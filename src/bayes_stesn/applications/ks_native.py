"""KS adapters for the accepted Bayesian ST-ESN, separable GP and basis GRU.

Only permitted, delivered-standardized observations enter these adapters.
Coordinates are the caller's fixed circle embedding, not a compressed spatial
grid.  Forecast histories are explicit bounded windows and are never replaced
by the fitted panel or a longer prefix.  The generic model implementations and
their historical benchmark behavior remain unchanged.

The Bayesian adapter retains the scalar-ridge iid Gaussian hierarchy, spatial
mean rule, frozen reservoir equations and exact collapsed sampler.  Its one
application-specific preparation change is to retain the *delivered* response
scale (center=0, scale=1), rather than estimate a second response transform on
the reservoir-conditioning prefix.  PCA, feature scaling and reservoir input
calibration still use that prefix.  No likelihood, coefficient prior, residual
component or posterior draw is substituted.

``checkpoint_path`` names a completed portable native-fit artifact.  Bayesian
MCMC progress lives beside it in ``<name>.chains.npz`` and the existing durable
fit-phase ledger.  Completed GP/GRU fits are reusable; interruption *inside*
their optimizer currently requires deterministic refitting and is disclosed.
All forecasts use the average of the returned complete paths as their primary
point prediction, including the GP and GRU.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import copy
import hashlib
import importlib.metadata
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy.linalg import cho_solve, eigh

from ..basis import SpatialMean
from ..data import SplitPlan, readonly
from ..diagnostics import summarize_chains
from ..features import FrozenFeatures
from ..io import content_hash, environment, load_state, save_state, seed_for
from ..model import FittedModel, ModelSpec, PreparedModel
from ..prediction import ForecastContext, simulate_predictive
from ..sampling import SamplerConfig
from ..benchmark.basis_gru import (
    BasisGRUFit, _configuration as gru_configuration, basis_gru_from_state,
)
from ..benchmark.classical import (
    _panel_state, classical_from_state, matern_correlation, psd_root,
)
from ..benchmark.contracts import Forecast, Panel, forecast_targets
from ..benchmark.esn import _checkpoint_info, _fit_checkpoint
from ..benchmark.gaussian_gp import (
    DEFAULT_CONFIG as GP_DEFAULTS, SeparableGPFit, filter_modes,
)


NATIVE_IDS = ("bayes_stesn", "separable_gp", "basis_gru")
STATE_SCHEMA = "ks-native-fit-1"


def reference_configs():
    """Accepted conservative reference and existing comparator starting points.

    These are starting candidates, not claims of KS optimality or the exact
    configuration selected in the earlier simulation confirmation campaign.
    The Bayesian values match reference_reliability.yaml; its feature and MCMC
    seeds are replaced by named streams of the supplied model seed.
    """
    return {
        "bayes_stesn": {
            "conditioning": 80,
            "features": {"kind": "pca_esn", "modes": 8, "units": [32],
                "lags": [1, 2], "input_scales": [.5],
                "leak_patterns": [[.2, .5, .9]], "score_bound": 3.,
                "squares": 16, "products": 0, "recurrent_norm": .9,
                "connectivity": .08, "feature_scale_floor": .04},
            "spatial_mean": {"side": 3, "width": .3, "kernel": "matern32",
                "coordinate_terms": True},
            "prior": {"kind": "ridge", "mean_log_precision": float(np.log(.1)),
                "sd_log_precision": 1.5},
            "covariance": {"kind": "matern", "nu": 1.5,
                "anisotropy": False, "nonstationary": False,
                "static_variance": False, "infer_nu": False,
                "scale_mode": "relative", "range_prior": "lognormal",
                "log_q_mean": 0., "log_q_sd": 1.25,
                "log_ell_mean": float(np.log(.18)), "log_ell_sd": .65},
            "innovations": "iid",
            "sampler": {"chains": 4, "warmup": 1000, "draws": 2000,
                "parameterization": "ridge_total_scale", "checkpoint_every": 250,
                "a0": 2., "b0": .05},
            "diagnostics": {"functional_draws": 512, "rhat_limit": 1.01,
                "min_ess": 400, "min_chains": 4,
                "max_coefficient_bytes": 256 * 1024 ** 2},
            "continuation_draws": 2000,
        },
        "separable_gp": {**GP_DEFAULTS, "spatial_nu": 1.5,
            "temporal_nu": .5, "spatial_length": .2, "temporal_length": 4.,
            "max_iter": 60, "continuation_max_iter": 120},
        "basis_gru": {**gru_configuration({}), "rank": 8,
            "basis_width": .25, "lags": 8, "hidden_size": 32,
            "epochs": 80, "patience": 12, "validation_horizon": 100,
            "require_full_validation_horizon": True,
            "covariance_continuation_max_iter": 80},
    }


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _computation_identity(model_id):
    """Bind numerical adapters, dependencies and effective thread settings."""
    from threadpoolctl import threadpool_info

    root = Path(__file__).resolve().parents[1]
    names = ["applications/ks_native.py", "data.py", "io.py", "prediction.py",
        "benchmark/contracts.py", "benchmark/classical.py"]
    if model_id == "bayes_stesn":
        names += ["model.py", "basis.py", "features.py", "_reservoir.py",
            "sampling.py", "gaussian.py", "spatial.py", "priors.py",
            "assessment.py", "diagnostics.py", "benchmark/esn.py"]
    elif model_id == "separable_gp":
        names += ["benchmark/gaussian_gp.py", "benchmark/optimization.py"]
    else:
        names += ["benchmark/basis_gru.py", "benchmark/optimization.py"]
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    runtime = environment()
    runtime["blas"] = [{key: entry.get(key) for key in
        ("internal_api", "num_threads", "version", "architecture")}
        for entry in threadpool_info() if entry.get("user_api") == "blas"]
    if model_id == "basis_gru":
        runtime["torch"] = importlib.metadata.version("torch")
    return {"source_sha256": digest.hexdigest(), "runtime": runtime}


def _resolve_bayes(config, seed):
    defaults = reference_configs()["bayes_stesn"]
    supplied = copy.deepcopy(dict(config or {}))
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown Bayesian KS options: {sorted(unknown)}")
    resolved = {}
    for key, default in defaults.items():
        resolved[key] = ({**default, **supplied.get(key, {})}
            if isinstance(default, dict) else supplied.get(key, default))
    resolved["conditioning"] = _integer(resolved["conditioning"], "conditioning", 10)
    resolved["continuation_draws"] = _integer(resolved["continuation_draws"], "continuation_draws", 0)
    resolved["features"]["seed"] = seed_for(seed, "reservoir")
    resolved["sampler"]["seed"] = seed_for(seed, "mcmc")
    spec = ModelSpec.from_dict({key: resolved[key] for key in
        ("features", "spatial_mean", "prior", "covariance", "innovations")})
    if spec.prior.kind != "ridge" or spec.innovations != "iid":
        raise ValueError("bayes_stesn preserves the accepted scalar-ridge iid Gaussian hierarchy")
    if (spec.covariance.kind != "matern" or spec.covariance.scale_mode != "relative"
            or spec.covariance.static_variance or spec.covariance.nonstationary
            or spec.covariance.anisotropy or spec.covariance.infer_nu):
        raise ValueError("KS reference adapter preserves isotropic stationary fixed-nu relative-scale Matérn covariance")
    sampler = SamplerConfig(**resolved["sampler"])
    if sampler.temporal != "iid":
        raise ValueError("Accepted KS Bayesian reference has iid residual innovation fields")
    resolved.update({key: value for key, value in spec.to_dict().items() if key != "name"})
    resolved["sampler"] = asdict(sampler)
    return resolved, spec, sampler


def resolve_native_config(model_id, config=None, seed=0):
    """Resolve defaults and actual seeds; ignored aliases are not candidates."""
    seed = _integer(seed, "seed", 0)
    if model_id == "bayes_stesn":
        return _resolve_bayes(config, seed)[0]
    if model_id == "basis_gru":
        return gru_configuration({**reference_configs()[model_id], **dict(config or {})})
    if model_id == "separable_gp":
        from ..benchmark.classical import _classical_config
        return _classical_config("separable_stgp", {**reference_configs()[model_id], **dict(config or {})})
    raise ValueError(f"Unknown native KS model {model_id!r}")


def _prepare_delivered_scale(data, config, spec):
    conditioning = config["conditioning"]
    if conditioning >= len(data.y) - 5:
        raise ValueError("Bayesian ESN needs a conditioning prefix and at least six likelihood targets")
    if conditioning <= max(spec.features.lags):
        raise ValueError("Conditioning must exceed the maximum causal feature lag")
    split = SplitPlan(conditioning, len(data.y))
    y = readonly(data.y)
    features = FrozenFeatures.fit(y, conditioning, data.coordinates, spec.features)
    basis = SpatialMean.fit(data.coordinates, spec.spatial_mean)
    phi, z = readonly(basis.transform(data.coordinates)), readonly(features.Z)
    reference = readonly(z[conditioning // 3:conditioning])
    identity = content_hash({"schema": "ks-delivered-scale-preparation-1",
        "data": _panel_state(data), "input_metadata": dict(data.metadata),
        "split": asdict(split), "spec": spec.to_dict(),
        "features": features.fingerprint,
        "response_transform": {"center": 0., "scale": 1.,
            "policy": "caller_verified_delivered_training_only_KS_transform"}})
    return PreparedModel(spec, data, split, features, basis, 0., 1., y, phi, z,
        reference, features.groups, identity)


def _validate_window(training, history):
    if not isinstance(history, Panel):
        raise TypeError("KS forecasts require an explicit permitted observation history Panel")
    if (history.site_ids != training.site_ids
            or not np.array_equal(history.coordinates, training.coordinates)):
        raise ValueError("Window input-site identities/order must equal the fitted input panel")
    for name in ("data_identity_sha256", "phase", "trajectory", "spatial_protocol", "ks_protocol"):
        if name in training.metadata and history.metadata.get(name) != training.metadata[name]:
            raise ValueError(f"KS history has an incompatible trajectory/protocol/transform identity ({name})")
    if ("response_transform" in training.metadata
            and history.metadata.get("response_transform") != training.metadata["response_transform"]):
        raise ValueError("KS history response transform differs from the fitted training transform")
    delta = float(training.times[1] - training.times[0])
    if len(history.y) < 2 or not np.allclose(np.diff(history.times), delta, rtol=1e-10, atol=1e-12):
        raise ValueError("KS history must contain at least two states at the fitted cadence")
    if history.times[-1] < training.times[-1] - 1e-12:
        raise ValueError("Forecast origin cannot precede the parameter fitting endpoint")
    offset = (float(history.times[-1]) - float(training.times[0])) / delta
    if not np.isclose(offset, round(offset), rtol=0, atol=1e-8):
        raise ValueError("Forecast history must align with the fitted time lattice")
    return int(round(offset)), delta


def _diagnose_bayes(fitted, checks, seed):
    """All readout coefficients plus hyperparameters and predictive functionals.

    Coefficients are integrated by the MCMC target.  Each recovered coefficient
    and scale sample is a joint conditional draw at the corresponding retained
    hyperstate.  Chain order is retained; these are not treated as iid draws.
    The extra one-step functionals use only the complete permitted training
    history.  No validation/test response or posterior-mean rollout is used.
    """
    started = time.perf_counter()
    checks = dict(checks)
    allowed = {"functional_draws", "rhat_limit", "min_ess", "min_chains", "max_coefficient_bytes"}
    if set(checks) - allowed:
        raise ValueError(f"Unknown KS Bayesian diagnostics: {sorted(set(checks) - allowed)}")
    requested = _integer(checks.pop("functional_draws", 512), "functional_draws", 8)
    memory_limit = _integer(checks.pop("max_coefficient_bytes", 256 * 1024 ** 2), "max_coefficient_bytes")
    posterior, p = fitted.posterior, fitted.prepared
    count = min(requested, posterior.n_draws)
    if count < 8:
        raise ValueError("At least eight retained draws per chain are needed for KS diagnostics")
    indices = np.linspace(0, posterior.n_draws - 1, count, dtype=int)
    spatial, temporal = p.Phi.shape[1], p.fit_Z.shape[1]
    coefficient_bytes = 8 * posterior.n_chains * count * spatial * temporal
    if coefficient_bytes > memory_limit:
        raise MemoryError(f"All-coefficient KS diagnostics need {coefficient_bytes} bytes; "
            f"declared diagnostic cap is {memory_limit}. Increase only after resource review")
    coefficients = np.empty((posterior.n_chains, count, spatial * temporal))
    locations = np.array([0, len(p.Phi) // 2, len(p.Phi) - 1])
    names = ("log_sigma2_delivered_scale", "readout_first_site", "readout_middle_site",
        "readout_last_site", "readout_spatial_average",
        "one_step_predictive_mean_first_site", "one_step_predictive_mean_middle_site",
        "one_step_predictive_mean_last_site", "one_step_predictive_mean_spatial_average",
        "log_one_step_marginal_predictive_variance")
    values = np.empty((posterior.n_chains, count, len(names)))
    z = p.fit_Z[-1]
    _, next_z = p.features.update(p.standardized_y[-1], p.features.states[-1])
    for chain in range(posterior.n_chains):
        for position, draw in enumerate(indices):
            parameter = posterior.draw_parameters(chain, int(draw),
                np.random.default_rng(seed_for(seed, "diagnostic", chain, int(draw))))
            coefficients[chain, position] = parameter.C.reshape(-1, order="F")
            mean = p.Phi @ (parameter.C @ z)
            next_mean = p.Phi @ (parameter.C @ next_z)
            # Accepted residual: sigma² (I + q K), with K(s,s)=1.
            log_variance = np.log(parameter.sigma2) + np.logaddexp(0.,
                float(parameter.covariance_parameters["log_q"]))
            values[chain, position] = np.r_[np.log(parameter.sigma2), mean[locations],
                float(mean.mean()), next_mean[locations], float(next_mean.mean()), log_variance]
    hyper = summarize_chains(posterior.chains, posterior.parameter_names, **checks)
    functional = summarize_chains(values, names, **checks)
    coefficient_names = tuple(f"C[{i},{j}]" for j in range(temporal) for i in range(spatial))
    coefficient = summarize_chains(coefficients, coefficient_names, **checks)
    return {"passed": bool(hyper.passed and functional.passed and coefficient.passed),
        "hyperparameters": hyper.to_dict(), "functionals": functional.to_dict(),
        "coefficients": coefficient.to_dict(),
        "diagnostic_seconds": time.perf_counter() - started,
        "monitoring": {"all_readout_coefficients": True,
            "coefficient_count": spatial * temporal, "coefficient_storage_bytes": coefficient_bytes,
            "max_coefficient_bytes": memory_limit, "draws_per_chain": count,
            "retained_indices": indices.tolist(), "coefficient_vector_order": "column_major",
            "site_indices": locations.tolist(), "seed": int(seed),
            "recovery": "joint_conditional_Gaussian_and_IG_given_each_selected_hyperstate",
            "predictive_conditioning": "full_permitted_training_prefix_no_future_labels",
            "restricted_monitoring": "coefficients_and_predictive_functionals_use_evenly_spaced_retained_states; all_sampled_hyperstates_are_checked",
            "qualification_scope": "all_sampled_hyperparameters_all_readout_coefficients_and_ten_scale_readout_predictive_functionals"}}


def _recover_bayes(prepared, initial_config, continuation, checks, diagnostic_seed,
                  checkpoint, computation):
    """Exact existing sampler with one continuation under the stronger KS gate."""
    maximum = initial_config.draws + continuation
    state_path = checkpoint.with_name(checkpoint.name + ".ks-fit-phase.npz")
    transition = asdict(initial_config)
    transition.pop("checkpoint_every", None)
    identity = content_hash({"schema": "ks-bayes-fit-phase-1", "prepared": prepared.identity,
        "initial_sampler": transition, "continuation_draws": continuation,
        "diagnostics": checks, "diagnostic_seed": diagnostic_seed,
        "computation": computation})
    info = (_checkpoint_info(checkpoint, prepared, initial_config, maximum)
        if checkpoint.exists() else None)
    resumed = state_path.exists() or info is not None
    if state_path.exists():
        state = load_state(state_path, kind="ks_bayes_fit_phase")
        if (state.get("identity") != identity or state.get("initial_draws") != initial_config.draws
                or state.get("maximum_draws") != maximum):
            raise ValueError("KS Bayesian phase checkpoint has incompatible model, source, runtime or allowance")
        if state["phase"] not in {"initial_sampling", "initial_diagnostics",
                "continuation_sampling", "continuation_diagnostics", "complete"}:
            raise ValueError("Unknown KS Bayesian recovery phase")
        if info is None and state["phase"] != "initial_sampling":
            raise ValueError("KS Bayesian phase receipt requires its missing chain checkpoint")
        if info is not None and info["draw_target"] > state["target_draws"]:
            raise ValueError("KS Bayesian chain checkpoint exceeds its phase draw allowance")
        if state["phase"] == "complete" and (not info["complete_at_target"]
                or info["sha256"] != state["checkpoint"]["sha256"]):
            raise ValueError("Completed KS Bayesian phase receipt does not match its chain checkpoint")
    else:
        if info is not None:
            raise ValueError("Existing KS chain checkpoint has no matching KS phase receipt; preserve and inspect")
        state = {"schema": "ks-bayes-fit-phase-1", "identity": identity,
            "initial_draws": initial_config.draws, "maximum_draws": maximum,
            "phase": "initial_sampling", "target_draws": initial_config.draws,
            "gates": [], "checkpoint": None, "completed_phase_seconds": {}}
        save_state(state_path, state, kind="ks_bayes_fit_phase")
    fitted = None
    while True:
        phase = state["phase"]
        cfg = replace(initial_config, draws=int(state["target_draws"]))
        if phase.endswith("sampling"):
            start = time.perf_counter()
            fitted = _fit_checkpoint(prepared, cfg, checkpoint)
            state["completed_phase_seconds"][phase] = time.perf_counter() - start
            state["phase"] = phase.replace("sampling", "diagnostics")
            state["checkpoint"] = _checkpoint_info(checkpoint, prepared, initial_config, maximum)
            save_state(state_path, state, kind="ks_bayes_fit_phase")
            continue
        if fitted is None:
            fitted = _fit_checkpoint(prepared, cfg, checkpoint)
        if phase == "complete":
            if not state["gates"] or len(state["gates"]) > 2:
                raise ValueError("Completed KS fit has invalid diagnostic history")
            break
        start = time.perf_counter()
        gate = _diagnose_bayes(fitted, checks, diagnostic_seed)
        state["completed_phase_seconds"][phase] = time.perf_counter() - start
        state["gates"].append(gate)
        if phase == "initial_diagnostics" and not gate["passed"] and continuation:
            state["phase"] = "continuation_sampling"
            state["target_draws"] = maximum
            fitted = None
        else:
            state["phase"] = "complete"
        save_state(state_path, state, kind="ks_bayes_fit_phase")
    return fitted, state["gates"], {"resumed": resumed, "phase": state["phase"],
        "initial_draws": state["initial_draws"], "maximum_draws": state["maximum_draws"],
        "completed_phase_seconds": state["completed_phase_seconds"],
        "continuation_policy": "one extension after finite initial fit fails any unchanged KS hyperparameter_coefficient_functional gate",
        "partial_attempt_timing": "phase_timings_cover_completed_attempts; outer_process_ledger_records_interrupted_work"}


class NativeFit:
    """Portable native model with explicit observation-window conditioning."""

    def __init__(self, model_id, core, config, seed, identity, input_identity,
                 computation, diagnostics):
        self.model_id, self.core = model_id, core
        self.config, self.seed, self.identity = config, int(seed), identity
        self.input_identity, self.computation = input_identity, computation
        self.diagnostics = diagnostics

    @property
    def training(self):
        return self.core.prepared.data if self.model_id == "bayes_stesn" else self.core.data

    def forecast(self, history_panel, horizon, target_coordinates=None,
                 target_site_ids=None, n_paths=256, seed=0, *, path_start=0):
        horizon = _integer(horizon, "horizon")
        n_paths = _integer(n_paths, "n_paths")
        path_start = _integer(path_start, "path_start", 0)
        seed = _integer(seed, "forecast seed", 0)
        origin, delta = _validate_window(self.training, history_panel)
        coords, ids = forecast_targets(self.training, target_coordinates, target_site_ids)
        if self.model_id == "bayes_stesn":
            p = self.core.prepared
            if len(history_panel.y) <= max(p.spec.features.lags):
                raise ValueError("Forecast window is shorter than the reservoir lag initialization")
            y = readonly((history_panel.y - p.response_center) / p.response_scale)
            states, z = p.features.history(y)
            context = ForecastContext(origin, y, readonly(states), readonly(z),
                history_panel.times, self.core.posterior_identity)
            draw = simulate_predictive(self.core, context, horizon=horizon,
                target_coordinates=coords, target_site_ids=ids, paths=n_paths,
                seed=seed, path_start=path_start, components=("observation",))
            samples = draw.observation
        elif self.model_id == "separable_gp":
            samples = _gp_paths(self.core, history_panel, horizon, coords,
                n_paths, seed, path_start)
        else:
            samples = _gru_paths(self.core, history_panel, horizon, coords,
                n_paths, seed, path_start)
        # No clipping, imputation or deterministic fallback: Forecast rejects
        # nonfinite paths and the outer ledger records the failed origin.
        mean = np.mean(samples, axis=0, dtype=np.float64)
        metadata = {"model": self.model_id, "fit_identity": self.identity,
            "parameter_policy": "frozen", "target": "clean_simulated_field",
            "residual_interpretation": "fitted_statistical_model_discrepancy_on_clean_field; no_added_sensor_noise",
            "mean_estimator": "average_of_returned_complete_predictive_paths",
            "qualified": bool(self.diagnostics.get("qualified", False)),
            "origin_index": origin, "first_target_index": origin + 1,
            "origin_time": float(history_panel.times[-1]),
            "first_target_time": float(history_panel.times[-1] + delta),
            "history_length": len(history_panel.y),
            "history_first_time": float(history_panel.times[0]),
            "state_initialization": "zero_reservoir_at_window_start" if self.model_id == "bayes_stesn"
                else ("stationary_GP_prior_at_window_start" if self.model_id == "separable_gp"
                      else "zero_GRU_hidden_per_declared_lag_window"),
            "observation_access": "permitted_input_sites_in_supplied_window_only",
            "newsite_response_history": "never_used", "path_count": n_paths,
            "path_start": path_start, "path_stop": path_start + n_paths,
            "forecast_seed": seed,
            "uncertainty": "posterior_joint" if self.model_id == "bayes_stesn" else "joint_plugin_gaussian",
            "parameter_uncertainty": self.model_id == "bayes_stesn",
            "scale": "delivered_standardized_response; caller_inverts_verified_transform"}
        return Forecast._from_owned(mean, samples, coords, ids, metadata)

    def iter_forecasts(self, history_panel, horizon, target_coordinates=None,
                       target_site_ids=None, n_paths=256, seed=0, *, batch_size=32,
                       path_start=0):
        """Yield complete joint-path batches using stable global path streams."""
        n_paths = _integer(n_paths, "n_paths")
        batch_size = _integer(batch_size, "batch_size")
        path_start = _integer(path_start, "path_start", 0)
        for offset in range(0, n_paths, batch_size):
            yield self.forecast(history_panel, horizon, target_coordinates,
                target_site_ids, min(batch_size, n_paths - offset), seed,
                path_start=path_start + offset)

    def to_state(self):
        return {"schema": STATE_SCHEMA, "model_id": self.model_id,
            "config": self.config, "seed": self.seed, "identity": self.identity,
            "input_identity": self.input_identity, "computation": self.computation,
            "diagnostics": self.diagnostics, "core": self.core.to_state()}

    @classmethod
    def from_state(cls, state):
        if state.get("schema") != STATE_SCHEMA or state.get("model_id") not in NATIVE_IDS:
            raise ValueError("Unsupported KS native-fit artifact")
        model_id = state["model_id"]
        if model_id == "bayes_stesn":
            core = FittedModel.from_state(state["core"])
        elif model_id == "basis_gru":
            core = basis_gru_from_state(state["core"])
        else:
            core = classical_from_state(state["core"])
        return cls(model_id, core, **{key: state[key] for key in
            ("config", "seed", "identity", "input_identity", "computation", "diagnostics")})

    def save(self, path):
        return save_state(path, self.to_state(), kind="ks_native_fit")

    @classmethod
    def load(cls, path):
        return cls.from_state(load_state(path, kind="ks_native_fit"))


def fit_native(model_id, training_panel, config=None, seed=0, checkpoint_path=None):
    """Fit/recover one native model from permitted normalized training values."""
    if not isinstance(training_panel, Panel):
        raise TypeError("KS native fitting accepts only an observed training Panel")
    if len(training_panel.y) < 12 or len(training_panel.coordinates) < 2:
        raise ValueError("Native KS fitting needs at least twelve rows and two observed sites")
    seed = _integer(seed, "model seed", 0)
    config = resolve_native_config(model_id, config, seed)
    computation = _computation_identity(model_id)
    input_identity = content_hash({"panel": _panel_state(training_panel),
        "metadata": dict(training_panel.metadata)})
    identity = content_hash({"schema": STATE_SCHEMA, "model_id": model_id,
        "input_identity": input_identity, "config": config, "seed": seed,
        "computation": computation})
    path = None if checkpoint_path is None else Path(checkpoint_path)
    if path is not None and path.exists():
        saved = load_state(path, kind="ks_native_fit")
        if saved.get("identity") != identity:
            raise ValueError("KS fitted checkpoint has incompatible data, transform, model, configuration, source or runtime")
        return NativeFit.from_state(saved)

    started = time.perf_counter()
    if model_id == "bayes_stesn":
        config, spec, sampler = _resolve_bayes(config, seed)
        prepared = _prepare_delivered_scale(training_panel, config, spec)
        with tempfile.TemporaryDirectory(prefix="ks-bayes-fit-") as temporary:
            chain_path = (Path(temporary) / "chains.npz" if path is None
                else path.with_name(path.name + ".chains.npz"))
            core, gates, recovery = _recover_bayes(prepared, sampler,
                config["continuation_draws"], config["diagnostics"],
                seed_for(seed, "diagnostics"), chain_path, computation)
        dimensions = {"conditioning_rows": config["conditioning"],
            "likelihood_rows": len(training_panel.y) - config["conditioning"],
            "temporal_features": prepared.Z.shape[1], "spatial_basis": prepared.Phi.shape[1],
            "readout_coefficients": prepared.Phi.shape[1] * prepared.Z.shape[1]}
        diagnostics = {"qualified": bool(gates[-1]["passed"]), "gates": gates,
            "recovery": recovery, "continuations": len(gates) - 1, **dimensions,
            "inference": "exact_collapsed_random_walk_Metropolis_then_conditional_Gaussian_IG_recovery",
            "monitoring_scope": "all_sampled_scalar_hyperparameters_all_readout_coefficients_and_ten_scale_readout_predictive_functionals",
            "parameter_count": dimensions["readout_coefficients"] + 4,
            "response_transform": "identity_on_delivered_KS_training_only_standardized_scale",
            "feature_transform_rows": [0, config["conditioning"]],
            "interrupted_fit_policy": "exact_chain_RNG_and_phase_resume_same_numerical_runtime",
            "fixed_prior": "scalar_ridge_on_all_readout_coefficients_including_intercept"}
    elif model_id == "separable_gp":
        core = SeparableGPFit(training_panel, config, seed)
        diagnostics = {**core.diagnostics, "parameter_count": 5,
            "fit_rows": len(training_panel.y),
            "interrupted_fit_policy": "completed_fit_reuse; unfinished_optimizer_restarts_from_declared_initialization"}
    else:
        core = BasisGRUFit(training_panel, config, seed)
        diagnostics = {**core.diagnostics,
            "parameter_count": sum(parameter.numel() for parameter in core.network.parameters()) + 3,
            "fit_rows": len(training_panel.y),
            "internal_stopping": "training_only_chronological_tail; separate_from_official_validation_configuration_selection",
            "interrupted_fit_policy": "completed_fit_reuse; unfinished_training_restarts_from_fixed_model_seed"}
    diagnostics.update(fit_wall_seconds=time.perf_counter() - started,
        input_identity=input_identity, fit_identity=identity,
        supplied_training_rows=len(training_panel.y), supplied_training_sites=training_panel.y.shape[1],
        coordinate_policy="static_circle_embedding_Euclidean_Matern_restricted_to_circle",
        residual_target="clean_field_statistical_discrepancy_not_artificial_measurement_noise")
    fitted = NativeFit(model_id, core, config, seed, identity, input_identity, computation, diagnostics)
    if path is not None:
        fitted.save(path)
    return fitted


def _gp_paths(core, history, horizon, coords, paths, seed, path_start):
    # This is the existing exact separable GP state construction, with the
    # conditional filter reset at the explicitly supplied window boundary.
    required = 8 * (2 * paths * horizon * len(coords) +
        4 * paths * (len(core._values) + len(coords)) * len(core._f) + 4 * len(coords) ** 2)
    if required > int(core.config.get("max_forecast_bytes", 1_073_741_824)):
        raise MemoryError("KS GP forecast exceeds max_forecast_bytes; reduce path batch size")
    unique, mapping, loadings, residual = core._target_setup(coords)
    mean, covariance, _ = filter_modes((history.y - core.center) @ core._vectors,
        core._values * core.signal_sd ** 2, core.noise_sd ** 2,
        core._f, core._p_inf, core._q)
    values, vectors = eigh(residual, check_finite=False)
    if values[0] < -1e-8 * max(float(values[-1]), 1.):
        raise np.linalg.LinAlgError("KS GP conditional new-site covariance is not PSD")
    keep = values > 1e-12 * max(float(values[-1]), 1.)
    values, vectors = np.maximum(values[keep], 0.), vectors[:, keep]
    n, r, dimension, sites = len(core._values), len(values), len(core._f), len(unique)
    streams = [np.random.default_rng(seed_for(seed, core.model_id, "path", i))
        for i in range(path_start, path_start + paths)]
    initial = np.array([rng.normal(size=(n + r) * dimension) for rng in streams])
    roots = np.array([psd_root(one) for one in covariance])
    observed = mean + np.einsum("bnq,nrq->bnr", initial[:, :n * dimension].reshape(paths, n, dimension), roots)
    remainder = (initial[:, n * dimension:].reshape(paths, r, dimension) @ psd_root(core._p_inf).T)
    remainder *= core.signal_sd * np.sqrt(values)[None, :, None]
    process_root = psd_root(core._q)
    observed_scale = core.signal_sd * np.sqrt(core._values)[None, :, None]
    remainder_scale = core.signal_sd * np.sqrt(values)[None, :, None]
    samples = np.empty((paths, horizon, sites))
    for lead in range(horizon):
        innovations = np.array([rng.normal(size=(n + r) * dimension + sites) for rng in streams])
        observed = observed @ core._f.T + (innovations[:, :n * dimension].reshape(paths, n, dimension) @ process_root.T) * observed_scale
        remainder = remainder @ core._f.T + (innovations[:, n * dimension:(n + r) * dimension].reshape(paths, r, dimension) @ process_root.T) * remainder_scale
        samples[:, lead] = (core.center + observed[:, :, 0] @ loadings.T
            + remainder[:, :, 0] @ vectors.T + core.noise_sd * innovations[:, -sites:])
    return np.array(samples[:, :, mapping], dtype=np.float64, order="C", copy=True)


def _gru_paths(core, history, horizon, coords, paths, seed, path_start):
    if len(history.y) < core.lags:
        raise ValueError("KS origin history is shorter than the fitted GRU lag window")
    unique, mapping = np.unique(coords, axis=0, return_inverse=True)
    lookup = {tuple(point): i for i, point in enumerate(core.data.coordinates)}
    observed_indices = np.array([lookup.get(tuple(point), -1) for point in unique])
    new_indices = np.flatnonzero(observed_indices < 0)
    at_observed = np.flatnonzero(observed_indices >= 0)
    domain = len(core.data.coordinates) + len(new_indices)
    if domain > core.config["max_domain_sites"]:
        raise MemoryError("KS basis-GRU joint domain exceeds max_domain_sites")
    required = 8 * (2 * paths * horizon * len(coords) + 4 * domain * domain
        + 3 * paths * core.lags * core._phi.shape[1] + 4 * paths * len(core._phi))
    if required > core.config["max_forecast_bytes"]:
        raise MemoryError("KS basis-GRU forecast exceeds max_forecast_bytes; reduce path batch size")
    phi = core.basis.evaluate(unique)
    new_weights = new_root = None
    if len(new_indices):
        new = unique[new_indices]
        cross = core.signal_sd ** 2 * matern_correlation(
            (new - core.offset) / core.coordinate_scale, core._normalized_coordinates,
            core.spatial_length, core.spatial_nu)
        new_weights = cho_solve(core._observed_factor, cross.T, check_finite=False).T
        residual = core.innovation_covariance(new, standardized=True) - new_weights @ cross.T
        new_root = psd_root(residual)
    coefficients = ((history.y[-core.lags:] - core.center) / core.scale) @ core._phi / len(core._phi)
    contexts = np.repeat(coefficients[None], paths, axis=0)
    streams = [np.random.default_rng(seed_for(seed, core.model_id, "observed_path", i))
        for i in range(path_start, path_start + paths)]
    new_streams = [np.random.default_rng(seed_for(seed, core.model_id, "newsite_path", i))
        for i in range(path_start, path_start + paths)]
    samples = np.empty((paths, horizon, len(unique)))
    for lead in range(horizon):
        coefficients = core._predict(contexts)
        innovations = np.array([rng.normal(size=len(core._phi)) for rng in streams]) @ core._observed_root.T
        generated = coefficients @ core._phi.T + innovations
        target_noise = np.empty((paths, len(unique)))
        target_noise[:, at_observed] = innovations[:, observed_indices[at_observed]]
        if len(new_indices):
            normal = np.array([rng.normal(size=len(new_indices)) for rng in new_streams])
            target_noise[:, new_indices] = innovations @ new_weights.T + normal @ new_root.T
        samples[:, lead] = core.center + core.scale * (coefficients @ phi.T + target_noise)
        next_coefficients = generated @ core._phi / len(core._phi)
        contexts = np.concatenate((contexts[:, 1:], next_coefficients[:, None]), axis=1)
    return np.array(samples[:, :, mapping], dtype=np.float64, order="C", copy=True)


def reservoir_synchronization(fitted, permitted_history, lengths=(128, 256, 512),
                              *, absolute_tolerance=1e-8, relative_tolerance=1e-6):
    """Compare bounded-window and full permitted-history reservoir replay.

    This uses no future labels, posterior draws, validation loss or test data.
    It measures one fitted feature map on one permitted trajectory and does
    not claim a universal dynamical guarantee.  The study must freeze its
    shared causal history policy from pilot/training measurements before test.
    """
    if not isinstance(fitted, NativeFit) or fitted.model_id != "bayes_stesn":
        raise TypeError("Synchronization assessment needs a fitted Bayesian KS ESN")
    _validate_window(fitted.training, permitted_history)
    p = fitted.core.prepared
    y = (permitted_history.y - p.response_center) / p.response_scale
    full_state, _ = p.features.history(y)
    state, z = p.features.update(y[-1], full_state[-1])
    rows = []
    for length in lengths:
        length = _integer(length, "synchronization history length", 2)
        if length > len(y):
            rows.append({"history": length, "available": False})
            continue
        window_state, _ = p.features.history(y[-length:])
        actual_state, actual_z = p.features.update(y[-1], window_state[-1])
        state_difference = float(np.max(np.abs(actual_state - state)))
        z_difference = float(np.max(np.abs(actual_z - z)))
        state_scale = max(float(np.max(np.abs(state))), 1e-12)
        z_scale = max(float(np.max(np.abs(z))), 1e-12)
        passed = (state_difference <= absolute_tolerance + relative_tolerance * state_scale
            and z_difference <= absolute_tolerance + relative_tolerance * z_scale)
        rows.append({"history": length, "available": True,
            "state_max_absolute_difference": state_difference,
            "state_max_relative_difference": state_difference / state_scale,
            "feature_max_absolute_difference": z_difference,
            "feature_max_relative_difference": z_difference / z_scale,
            "passed": bool(passed)})
    core = p.features.core
    layer_bounds = []
    for weight, leaks in zip(core.W, core.leaks):
        layer_bounds.append({"recurrent_operator_norm": float(np.linalg.norm(weight, 2)),
            "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(weight)))),
            "sufficient_contraction_bound": float(np.max(1 - leaks) + np.linalg.norm(leaks[:, None] * weight, 2)),
            "minimum_leak": float(np.min(leaks))})
    return {"schema": "ks-reservoir-synchronization-1", "fit_identity": fitted.identity,
        "permitted_history_rows": len(y), "last_history_time": float(permitted_history.times[-1]),
        "full_replay_policy": "zero_state_at_first_permitted_history_row",
        "comparison": "state_and_features_for_first_future_target_after_last_history_observation",
        "absolute_tolerance": absolute_tolerance, "relative_tolerance": relative_tolerance,
        "rows": rows, "layer_bounds": layer_bounds,
        "scope": "empirical_fitted_map_pilot_or_training_check; no_universal_synchronization_guarantee"}
