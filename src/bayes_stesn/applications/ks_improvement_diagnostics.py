"""Matched diagnostics for the accepted KS Bayesian ST-ESN.

No fitting, selection, calibration, or evaluation-label access occurs here.
Values use the fitted input scale; the caller applies the verified inverse
transform. Only full_predictive is the original predictive distribution.
Zero-innovation parameter paths and fixed-parameter innovation paths intervene
on a nonlinear recurrence: their variances need not sum to full variance.
The deterministic conditional-readout rollout is NOT a predictive mean.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import numpy as np

from ..data import readonly
from ..gaussian import GaussianRegression
from ..io import content_hash, seed_for
from ..prediction import ForecastContext, joint_domain, simulate_predictive
from ..sampling import ParameterDraw
from ..benchmark.contracts import Forecast
from ..benchmark.classical import forecast_targets
from .ks_native import NativeFit, _validate_window, reservoir_synchronization

MODES = ("full_predictive", "parameter_only", "fixed_parameter_innovations",
         "deterministic_conditional_mean")


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _prepared_only(fitted):
    if not isinstance(fitted, NativeFit) or fitted.model_id != "bayes_stesn":
        raise TypeError("This diagnostic requires a Bayesian KS NativeFit")
    return fitted.core.prepared


def _accepted(fitted):
    p, posterior = _prepared_only(fitted), fitted.core.posterior
    covariance = p.spec.covariance
    if (p.spec.innovations != "iid" or p.spec.prior.kind != "ridge"
            or posterior.config.temporal != "iid" or covariance.kind != "matern"
            or covariance.scale_mode != "relative" or covariance.anisotropy
            or covariance.nonstationary or covariance.static_variance or covariance.infer_nu
            or posterior.metadata.get("spatial_backend") == "vecchia_operator"):
        raise ValueError("Diagnostics support only the accepted iid Gaussian scalar-ridge, stationary relative-scale Matérn model")
    return p, posterior


def _parameter_state(parameter):
    return {"C": parameter.C, "sigma2": parameter.sigma2,
            "covariance_parameters": parameter.covariance_parameters,
            "prior_state": parameter.prior_state,
            "temporal_parameters": parameter.temporal_parameters}


@dataclass(frozen=True)
class ConditionalReadoutPoint:
    """A plug-in coefficient mean at posterior-median log hyperparameters."""
    parameter: ParameterDraw
    fit_identity: str
    posterior_identity: str
    identity: str
    metadata: dict

    def validate(self, fitted):
        expected = content_hash({"fit_identity": self.fit_identity,
            "posterior_identity": self.posterior_identity,
            "parameter": _parameter_state(self.parameter), "metadata": self.metadata})
        if (self.fit_identity != fitted.identity or self.posterior_identity != fitted.core.posterior_identity
                or self.identity != expected):
            raise ValueError("Conditional readout point is mutated or belongs to another fit")


def conditional_mean_parameters(fitted):
    """Exact E[C | data, median log-hyperparameters], not marginal E[C | data].

    Retained hyperstates have equal weight. Medians use stored log coordinates.
    The common variance plug-in is conditional IG mean bn/(an-1). The original
    separable Gaussian algebra and all spatial/temporal features are reused.
    """
    p, posterior = _accepted(fitted)
    states = [state for chain in posterior.states for state in chain]
    prior_vectors = np.asarray([posterior._runtime.vector(s["prior_state"]) for s in states])
    covariance_vectors = np.asarray([posterior.covariance.parameters_to_vector(
        s["covariance_parameters"], posterior.coordinates) for s in states])
    prior_state = posterior._runtime.state_from_vector(np.median(prior_vectors, axis=0))
    cp = posterior.covariance.vector_to_parameters(np.median(covariance_vectors, axis=0),
                                                   posterior.coordinates)
    R = posterior.covariance.covariance(p.data.coordinates, cp)
    precision = posterior._runtime.precision(prior_state, R)
    engine = GaussianRegression(posterior.Y, posterior.Phi, posterior.Z,
        posterior.config.a0, posterior.config.b0, posterior.config.dense_threshold,
        posterior.config.max_dense_bytes)
    evaluation = engine.evaluate(R, precision=precision)
    if evaluation.an <= 1:
        raise ValueError("Conditional inverse-gamma mean does not exist")
    C, sigma2 = readonly(evaluation.mean), float(evaluation.bn/(evaluation.an-1))
    if not np.isfinite(C).all() or not np.isfinite(sigma2) or sigma2 <= 0:
        raise FloatingPointError("Nonfinite conditional readout point")
    parameter = ParameterDraw(C, sigma2, copy.deepcopy(cp), copy.deepcopy(prior_state), {},
                              {"method": "existing_separable_Gaussian_conditional_mean"})
    metadata = {"point_rule": "E[C|training,componentwise_median_stored_log_hyperparameters]",
        "sigma2_rule": "conditional_IG_mean_bn_over_an_minus_1",
        "conditional_IG_shape": float(evaluation.an), "conditional_IG_scale": float(evaluation.bn),
        "hyperstate_count": len(states), "chains": posterior.n_chains,
        "draws_per_chain": posterior.n_draws,
        "log_covariance_parameter_names": list(posterior.covariance.parameter_names(p.data.coordinates)),
        "log_covariance_point": posterior.covariance.parameters_to_vector(cp).tolist(),
        "readout_precision": float(precision.values),
        "temporal_feature_count": p.Z.shape[1], "spatial_basis_count": p.Phi.shape[1],
        "original_fit_qualified": bool(fitted.diagnostics.get("qualified", False)),
        "not_marginal_posterior_coefficient_mean": True,
        "not_full_posterior_predictive_mean": True}
    state = {"fit_identity": fitted.identity, "posterior_identity": fitted.core.posterior_identity,
             "parameter": _parameter_state(parameter), "metadata": metadata}
    return ConditionalReadoutPoint(parameter, fitted.identity, fitted.core.posterior_identity,
                                  content_hash(state), metadata)


def _context(fitted, history):
    p, _ = _accepted(fitted)
    origin, delta = _validate_window(fitted.training, history)
    if len(history.y) <= max(p.spec.features.lags):
        raise ValueError("History must exceed the fitted lag requirement")
    y = readonly((history.y-p.response_center)/p.response_scale)
    states, z = p.features.history(y)
    context = ForecastContext(origin, y, readonly(states), readonly(z), history.times,
                              fitted.core.posterior_identity)
    return p, context, delta


class _FixedPosterior:
    """A separate conditional experiment; never substitutes into stored fit."""
    n_chains = 1
    n_draws = 1
    metadata = {}

    def __init__(self, point):
        self.point = point

    def draw_parameters(self, chain, draw, rng):
        return self.point.parameter


def _draw_parameter(posterior, seed, path):
    # Exactly matches prediction.simulate_predictive, including recovery RNG.
    rng = np.random.default_rng(seed_for(seed, "parameter", path))
    chain = int(rng.integers(posterior.n_chains))
    draw = int(rng.integers(posterior.n_draws))
    return posterior.draw_parameters(chain, draw, rng), [chain, draw]


def diagnostic_forecast(fitted, history_panel, horizon, target_coordinates=None,
                        target_site_ids=None, *, mode="full_predictive", n_paths=256,
                        seed=0, path_start=0, point=None, max_output_bytes=128*1024**2):
    """One bounded diagnostic batch with unchanged fitted features/geometry.

    Full/parameter-only modes share exact posterior parameter streams.
    Full/fixed-parameter modes share exact innovation streams. Each complete
    path holds static parameters and feeds back its own INPUT-site values.
    deterministic_conditional_mean requires n_paths=1 and returns no samples.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown diagnostic mode {mode!r}")
    horizon, n_paths = _integer(horizon, "horizon"), _integer(n_paths, "n_paths")
    seed, path_start = _integer(seed, "seed", 0), _integer(path_start, "path_start", 0)
    budget = _integer(max_output_bytes, "max_output_bytes")
    if mode == "deterministic_conditional_mean" and (n_paths != 1 or path_start != 0):
        raise ValueError("Deterministic diagnostic has exactly one rollout: n_paths=1, path_start=0")
    if point is not None and mode in {"full_predictive", "parameter_only"}:
        raise ValueError("A plug-in point is not used by this mode; do not supply ignored controls")
    p, context, delta = _context(fitted, history_panel)
    targets, ids = forecast_targets(fitted.training, target_coordinates, target_site_ids)
    estimated_output_bytes = 8*horizon*len(targets)*(n_paths+1)
    if estimated_output_bytes > budget:
        raise MemoryError("Diagnostic outputs exceed max_output_bytes; use path batches")
    joint, target_index = joint_domain(p.data.coordinates, targets,
        input_ids=p.data.site_ids, target_ids=ids)
    if mode in {"fixed_parameter_innovations", "deterministic_conditional_mean"}:
        point = conditional_mean_parameters(fitted) if point is None else point
        point.validate(fitted)
    draw_ids = []
    if mode == "full_predictive":
        draw = simulate_predictive(fitted.core, context, horizon=horizon,
            target_coordinates=targets, target_site_ids=ids, paths=n_paths, seed=seed,
            path_start=path_start, components=("observation",), max_output_bytes=budget)
        samples, draw_ids = draw.observation, draw.parameter_draw_ids.tolist()
    elif mode == "fixed_parameter_innovations":
        conditional_identity = content_hash({"original_posterior": fitted.core.posterior_identity,
                                             "conditional_point": point.identity})
        conditional_core = replace(fitted.core, posterior=_FixedPosterior(point),
                                   posterior_identity=conditional_identity)
        conditional_context = replace(context, parameter_posterior_identity=conditional_identity)
        draw = simulate_predictive(conditional_core, conditional_context, horizon=horizon,
            target_coordinates=targets, target_site_ids=ids, paths=n_paths, seed=seed,
            path_start=path_start, components=("observation",), max_output_bytes=budget)
        samples = draw.observation
    else:
        phi = p.basis.transform(joint)
        samples = np.empty((n_paths, horizon, len(targets)), dtype=np.float64)
        for path in range(n_paths):
            if mode == "parameter_only":
                parameter, draw_id = _draw_parameter(fitted.core.posterior, seed, path_start+path)
                draw_ids.append(draw_id)
            else:
                parameter = point.parameter
            mean_map = phi@parameter.C
            previous = np.array(context.historical_y[-1], copy=True)
            state = np.array(context.states[-1], copy=True)
            for lead in range(horizon):
                state, z = p.features.update(previous, state)
                values = mean_map@z
                if not np.isfinite(values).all():
                    raise FloatingPointError(f"Nonfinite diagnostic rollout at lead {lead+1}")
                samples[path, lead] = values[target_index]*p.response_scale+p.response_center
                previous = values[:len(p.data.coordinates)]
    mean = np.mean(samples, axis=0, dtype=np.float64)
    deterministic = mode == "deterministic_conditional_mean"
    original_qualified = bool(fitted.diagnostics.get("qualified", False))
    metadata = {"schema": "ks-improvement-diagnostic-forecast/1", "mode": mode,
        "fit_identity": fitted.identity, "posterior_identity": fitted.core.posterior_identity,
        "feature_fingerprint": p.features.fingerprint, "diagnostic_only": mode != "full_predictive",
        "original_fit_qualified": original_qualified,
        "probabilistic_ranking_eligible": mode == "full_predictive" and original_qualified,
        "mean_policy": ("full_posterior_predictive_mean_estimated_from_complete_paths" if mode == "full_predictive"
            else "single_conditional_readout_zero_innovation_rollout_NOT_predictive_mean" if deterministic
            else "mean_of_diagnostic_complete_paths_NOT_full_posterior_predictive_mean"),
        "origin_index": context.origin, "first_target_index": context.origin+1,
        "first_target_time": float(context.times[-1]+delta), "history_length": len(history_panel.y),
        "horizon": horizon, "path_start": path_start, "path_stop": path_start+n_paths,
        "seed": seed, "parameter_draw_ids": draw_ids,
        "point_identity": None if point is None else point.identity,
        "feedback": "own_predicted_input_site_values; no hidden or future response inputs",
        "observation_access": "same permitted input history and fitted transform as original NativeFit",
        "output_scale": "fitted input scale; caller inverts delivered scalar transform",
        "uncertainty_interpretation": "intervention diagnostics, not additive variance attribution; nonlinear recurrence changes with removed uncertainty",
        "memory_scope": "output budget only; existing model/feature/spatial allocation gates remain relevant"}
    return Forecast._from_owned(mean, None if deterministic else samples, targets, ids, metadata)


def iter_diagnostic_forecasts(fitted, history_panel, horizon, target_coordinates=None,
                             target_site_ids=None, *, mode="full_predictive", n_paths=256,
                             seed=0, path_start=0, batch_size=64, point=None,
                             max_output_bytes=128*1024**2):
    """Yield stable global-path batches; callers consume scores incrementally."""
    n_paths, batch_size = _integer(n_paths, "n_paths"), _integer(batch_size, "batch_size")
    path_start = _integer(path_start, "path_start", 0)
    if mode == "deterministic_conditional_mean" and (n_paths != 1 or path_start != 0):
        raise ValueError("Deterministic diagnostic requires n_paths=1, path_start=0")
    if mode in {"fixed_parameter_innovations", "deterministic_conditional_mean"} and point is None:
        point = conditional_mean_parameters(fitted)
    for start in range(0, n_paths, batch_size):
        yield diagnostic_forecast(fitted, history_panel, horizon, target_coordinates,
            target_site_ids, mode=mode, n_paths=min(batch_size,n_paths-start), seed=seed,
            path_start=path_start+start, point=point, max_output_bytes=max_output_bytes)


def teacher_forced_readout(fitted, history_panel, future_input_panel,
                          target_coordinates=None, target_site_ids=None, *, point=None):
    """One-step plug-in means with declared observed-input updates.

    This separate diagnostic is NOT autonomous. Prediction at lead h accesses
    future_input_panel only through h-1; evaluation labels are not accepted.
    Delivered KS metadata restricts the helper to validation, not test times.
    """
    p, context, delta = _context(fitted, history_panel)
    _validate_window(fitted.training, future_input_panel)
    expected = context.times[-1]+delta*np.arange(1,len(future_input_panel.y)+1)
    if not np.allclose(future_input_panel.times, expected, rtol=0, atol=1e-10):
        raise ValueError("Teacher-forced inputs must immediately follow the history")
    phase = p.data.metadata.get("phase")
    stop = {"core": 8000, "pilot": 1600}.get(phase)
    if stop is not None:
        last_index = (future_input_panel.times[-1]-p.data.times[0])/delta
        if last_index >= stop-1e-8:
            raise ValueError("Teacher-forced diagnostic is restricted to training/validation, not the KS test interval")
    targets, ids = forecast_targets(fitted.training, target_coordinates, target_site_ids)
    point = conditional_mean_parameters(fitted) if point is None else point
    point.validate(fitted)
    mean_map = p.basis.transform(targets)@point.parameter.C
    means = np.empty((len(future_input_panel.y),len(targets)))
    state, previous = np.array(context.states[-1],copy=True), context.historical_y[-1]
    supplied = (future_input_panel.y-p.response_center)/p.response_scale
    for lead in range(len(means)):
        state, z = p.features.update(previous,state)
        means[lead] = (mean_map@z)*p.response_scale+p.response_center
        if lead+1 < len(means):
            previous = supplied[lead]
    return Forecast(means,None,targets,ids,{"schema":"ks-teacher-forced-diagnostic/1",
        "fit_identity":fitted.identity,"point_identity":point.identity,"diagnostic_only":True,
        "regime":"teacher_forced_one_step; NOT autonomous","probabilistic_ranking_eligible":False,
        "history_length":len(history_panel.y),"future_input_rows":len(future_input_panel.y),
        "future_input_access":"permitted input sites only; prediction h uses observations strictly before h",
        "output_scale":"fitted input scale; caller inverts delivered scalar transform"})


def training_readout_residuals(fitted, *, point=None):
    """Permitted-site training residual summaries; explicitly in-sample."""
    p, _ = _accepted(fitted)
    point = conditional_mean_parameters(fitted) if point is None else point
    point.validate(fitted)
    residual = p.Y.T-p.fit_Z@point.parameter.C.T@p.Phi.T
    return {"schema":"ks-training-readout-diagnostic/1","fit_identity":fitted.identity,
        "point_identity":point.identity,"regime":"in_sample_one_step_conditional_readout; not out_of_sample",
        "rows":len(residual),"sites":residual.shape[1],"fit_start":p.split.fitting_start,
        "fit_stop":p.split.fitting_stop,"rmse_fitted_scale":float(np.sqrt(np.mean(residual**2))),
        "mae_fitted_scale":float(np.mean(np.abs(residual))),
        "residual_mean_by_site":residual.mean(axis=0).tolist(),
        "residual_sd_by_site":residual.std(axis=0).tolist()}


def spatial_design_diagnostics(training_basis, target_basis, *, ridge_values=(0.,1e-8,1e-6,1e-4,.01,1.),
                               weak_rtol=1e-6, pseudoinverse_rtol=1e-12,
                               max_matrix_bytes=64*1024**2):
    """Geometry-only conditioning, weak directions and output-decoder gains.

    D_a = Phi_target (Phi_train'Phi_train/n + a I)^-1 Phi_train'/n.
    At a=0 an explicitly thresholded Moore-Penrose inverse is used. These
    operators are diagnostic output maps; they never change recurrence.
    All coefficient columns including the intercept are penalized for a>0.
    """
    P,T = np.asarray(training_basis,np.float64),np.asarray(target_basis,np.float64)
    if P.ndim!=2 or T.ndim!=2 or min(P.shape)<1 or min(T.shape)<1 or P.shape[1]!=T.shape[1]:
        raise ValueError("Finite training/target basis matrices with a common column axis required")
    if not np.isfinite(P).all() or not np.isfinite(T).all():
        raise ValueError("Basis matrices must be finite")
    if not 0<pseudoinverse_rtol<weak_rtol<1:
        raise ValueError("Require 0 < pseudoinverse_rtol < weak_rtol < 1")
    n,p = P.shape; k=min(P.shape); q=len(T)
    estimate = 8*(P.size+T.size+n*k+k*p+q*k+2*n*n+q*n)
    if estimate > _integer(max_matrix_bytes,"max_matrix_bytes"):
        raise MemoryError("Spatial diagnostic workspaces exceed max_matrix_bytes")
    U,s,Vt = np.linalg.svd(P,full_matrices=False)
    threshold=pseudoinverse_rtol*s[0]; active=s>threshold
    rank=int(np.count_nonzero(active)); full_rank=rank==p
    query_directions=T@Vt.T
    weakest=[]
    for j in np.argsort(s)[:min(5,k)]:
        weakest.append({"index":int(j),"training_singular_value":float(s[j]),
            "relative_training_singular_value":float(s[j]/s[0]) if s[0] else 0.,
            "target_direction_norm":float(np.linalg.norm(query_directions[:,j])),
            "target_to_training_gain":float(np.linalg.norm(query_directions[:,j])/s[j]) if s[j]>0 else None,
            "right_singular_direction":Vt[j].tolist()})
    rowspace = Vt[active]
    unresolved = T-(T@rowspace.T)@rowspace
    decoder_rows=[]
    for value in ridge_values:
        ridge=float(value)
        if not np.isfinite(ridge) or ridge<0:
            raise ValueError("Geometry ridge controls must be finite and nonnegative")
        gain=np.divide(1.,s,out=np.zeros_like(s),where=active) if ridge==0 else s/(s*s+n*ridge)
        D=(query_directions*gain)@U.T
        training_D=(U*(s*gain))@U.T
        decoder_rows.append({"ridge":ridge,"loss_convention":"SSE/n + ridge*all_coefficient_squared_norm",
            "operator_norm":float(np.linalg.norm(D,2)),
            "maximum_row_norm":float(np.linalg.norm(D,axis=1).max()),
            "training_smoother_trace":float(np.trace(training_D)),
            "relative_basis_reconstruction_error":float(np.linalg.norm(training_D@P-P)/max(np.linalg.norm(P),np.finfo(float).tiny)),
            "recurrence_policy":"diagnostic output map only; never feed back decoded targets"})
    return {"schema":"ks-spatial-design-diagnostic/1","training_sites":n,"target_sites":q,
        "basis_columns":p,"singular_values":s.tolist(),"numerical_rank":rank,
        "full_column_rank_at_declared_cutoff":full_rank,
        "condition_number":float(s[0]/s[-1]) if full_rank else None,
        "weak_rtol":weak_rtol,"pseudoinverse_rtol":pseudoinverse_rtol,
        "weak_directions_including_unidentified":int(np.count_nonzero(s<=weak_rtol*s[0])+p-k),
        "weakest_directions":weakest,
        "training_row_norms":np.linalg.norm(P,axis=1).tolist(),
        "target_row_norms":np.linalg.norm(T,axis=1).tolist(),
        "target_unresolved_direction_row_norms":np.linalg.norm(unresolved,axis=1).tolist(),
        "target_prior_variance_geometry_multipliers":np.sum(T*T,axis=1).tolist(),
        "prior_geometry_formula":"Var(phi(s)' C z | sigma2,lambda) = sigma2/lambda * ||phi(s)||² * ||z||²; not a posterior variance formula",
        "decoder_diagnostics":decoder_rows,"estimated_workspace_bytes":estimate,
        "uses_response_values":False,"basis_fitting":"already frozen on permitted training geometry"}


def geometry_diagnostics(fitted, target_coordinates, *, target_site_ids=None, **kwargs):
    """Inspect the actual frozen spatial basis at static query coordinates."""
    p = _prepared_only(fitted)
    targets,ids = forecast_targets(fitted.training,target_coordinates,target_site_ids)
    columns=p.Phi.shape[1]
    if 8*len(targets)*columns > kwargs.get("max_matrix_bytes",64*1024**2):
        raise MemoryError("Target basis exceeds geometry workspace budget")
    result=spatial_design_diagnostics(p.Phi,p.basis.transform(targets),**kwargs)
    result.update(fit_identity=fitted.identity,
        posterior_required=False,has_fitted_posterior=hasattr(fitted.core,"posterior"),
        target_coordinates=np.asarray(targets).tolist(),target_site_ids=list(ids),
        input_coordinates=p.data.coordinates.tolist(),
        basis_column_center=np.asarray(p.basis.center).tolist(),
        basis_column_scale=np.asarray(p.basis.scale).tolist(),
        basis_normalization=float(p.basis.normalization),basis_scale_floor=1e-6,
        basis_formula="[1,(raw-training_coordinate_mean)/max(training_coordinate_SD,1e-6)/sqrt(raw_column_count)]")
    return result


def synchronization_diagnostics(fitted, permitted_history, lengths=(128,512,1024), **kwargs):
    """Reuse the original state-and-feature replay check without fitting."""
    _prepared_only(fitted)
    return reservoir_synchronization(fitted,permitted_history,lengths=lengths,**kwargs)

