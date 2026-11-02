"""Joint, observation-feedback posterior predictive trajectories.

Parameters remain fixed for an entire path. The reservoir is driven by that
path's simulated observations at the input sites. Extra locations are sampled
jointly, and duplicate locations refer to the same random observation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from .data import SpatioTemporalData, readonly
from .io import seed_for, save_state, load_state

_PREDICTIVE_COMPONENTS = ("observation", "readout_mean", "conditional_mean", "latent")
_DEFAULT_OUTPUT_BYTES = 1024 ** 3


def _integer(value, name, *, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if number != value or number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return number


def _prediction_components(components):
    try:
        components = (components,) if isinstance(components, str) else tuple(components)
    except TypeError as exc:
        raise ValueError("components must name the predictive components to store") from exc
    if (not components or any(not isinstance(k, str) or k not in _PREDICTIVE_COMPONENTS for k in components)
            or len(set(components)) != len(components) or "observation" not in components):
        raise ValueError("components must be distinct predictive components including observation")
    return tuple(k for k in _PREDICTIVE_COMPONENTS if k in components)


def predictive_output_bytes(paths, horizon, target_sites, *, components=_PREDICTIVE_COMPONENTS):
    """Bytes for selected float64 output components and int64 draw identifiers.

    This is an output-storage bound, not a bound on covariance factors,
    feature history, model parameters, or scoring workspaces.
    """
    paths, horizon, target_sites = (_integer(v, name) for v, name in
                                   ((paths, "paths"), (horizon, "horizon"), (target_sites, "target_sites")))
    return 8 * paths * horizon * target_sites * len(_prediction_components(components)) + 16 * paths


def _check_output_budget(paths, horizon, target_sites, components, max_output_bytes):
    required = predictive_output_bytes(paths, horizon, target_sites, components=components)
    if max_output_bytes is not None:
        budget = _integer(max_output_bytes, "max_output_bytes")
        if required > budget:
            raise MemoryError(f"Predictive outputs require {required:,} bytes, above max_output_bytes={budget:,}; "
                              "request fewer components/paths, use iter_predictive(), or explicitly increase the budget")
    return required


def _all_finite(array):
    # The internally allocated arrays are C-contiguous. The K-order view
    # also avoids a full copy for public Fortran-contiguous inputs.
    flat = array.ravel(order="K")
    return all(np.isfinite(flat[start:start + 1048576]).all() for start in range(0, flat.size, 1048576))


def _freeze_owned(array, dtype):
    """Transfer a private contiguous allocation; never use on caller arrays."""
    if (not isinstance(array, np.ndarray) or array.dtype != np.dtype(dtype)
            or not (array.flags.c_contiguous or array.flags.f_contiguous)):
        raise ValueError("Internal predictive ownership transfer requires an owned contiguous allocation")
    owner = array
    while isinstance(owner.base, np.ndarray):
        owner = owner.base
    if not owner.flags.owndata or owner.nbytes != array.nbytes:
        raise ValueError("Internal predictive ownership transfer requires the whole private allocation")
    # NumPy loads a Fortran NPZ member as a shape view of its private flat
    # allocation. Freeze that owner too, without copying the entire archive.
    owner.setflags(write=False)
    array.setflags(write=False)
    # The published view cannot enable writes while its private base is
    # frozen. No mutable caller-owned alias was accepted on this route.
    view = array.view()
    view.setflags(write=False)
    return view


@dataclass(frozen=True)
class ForecastContext:
    origin: int
    historical_y: np.ndarray
    states: np.ndarray
    Z: np.ndarray
    times: np.ndarray
    parameter_posterior_identity: str
    parameter_policy: str = "frozen"


def forecast_context(fitted, origin=None, *, history=None, parameter_policy="frozen"):
    if parameter_policy != "frozen":
        raise ValueError("Use refit() to update the parameter posterior; forecast contexts freeze parameters")
    prepared = fitted.prepared
    origin = fitted.training_end if origin is None else int(origin)
    if origin < fitted.training_end:
        raise ValueError("Forecast origin cannot precede the parameter training endpoint")
    if history is None:
        if origin != fitted.training_end:
            raise ValueError("Later origins require the newly available observed history")
        return ForecastContext(origin, prepared.standardized_y, prepared.features.states,
                               prepared.Z, prepared.data.times, fitted.posterior_identity)
    if not isinstance(history, SpatioTemporalData):
        raise TypeError("history must be SpatioTemporalData containing the historical prefix")
    if history.site_ids != prepared.data.site_ids or not np.array_equal(history.coordinates, prepared.data.coordinates):
        raise ValueError("Forecast history must have the identical input-site panel and order")
    if origin >= len(history.y):
        raise ValueError("Origin exceeds available history")
    n = len(prepared.data.y)
    if not np.array_equal(history.y[:n], prepared.data.y) or not np.array_equal(history.times[:n], prepared.data.times):
        raise ValueError("Historical prefix differs from the fitted observations")
    y = readonly((history.y[:origin+1]-prepared.response_center)/prepared.response_scale)
    states, Z = prepared.features.history(y)
    return ForecastContext(origin, y, readonly(states), readonly(Z), readonly(history.times[:origin+1]),
                           fitted.posterior_identity)


def advance_state(fitted, context, newly_available_observed_history):
    """Assimilate new observations into feature/filter history without refitting.

    The new data contains only additional rows. Latent filtering is done for
    each fixed parameter draw when forecasting, using all available history.
    """
    if context.parameter_posterior_identity != fitted.posterior_identity:
        raise ValueError("Context belongs to a different parameter posterior")
    new = newly_available_observed_history
    if not isinstance(new, SpatioTemporalData):
        raise TypeError("New history must be SpatioTemporalData")
    p = fitted.prepared
    if new.site_ids != p.data.site_ids or not np.array_equal(new.coordinates, p.data.coordinates):
        raise ValueError("Input-site identity/order changed")
    step = context.times[-1]-context.times[-2]
    expected = context.times[-1]+step*np.arange(1, len(new.y)+1)
    if not np.allclose(new.times, expected, rtol=0, atol=max(1e-10, abs(step)*1e-10)):
        raise ValueError("New observations must immediately follow the current history")
    standardized = (new.y-p.response_center)/p.response_scale
    y = np.vstack([context.historical_y, standardized])
    states, Z = p.features.history(y)
    return ForecastContext(len(y)-1, readonly(y), readonly(states), readonly(Z),
                           readonly(np.r_[context.times, new.times]), fitted.posterior_identity)


def joint_domain(input_coordinates, target_coordinates, *, input_ids=None, target_ids=None):
    inputs, targets = np.asarray(input_coordinates, float), np.asarray(target_coordinates, float)
    if targets.ndim != 2 or targets.shape[1] != inputs.shape[1] or not len(targets) or not np.isfinite(targets).all():
        raise ValueError("target_coordinates must be a nonempty finite matrix in the input geometry")
    if target_ids is not None and (len(target_ids) != len(targets) or len(set(target_ids)) != len(target_ids)):
        raise ValueError("target_site_ids must be unique and match targets")
    if input_ids is not None and target_ids is not None:
        byid = dict(zip(input_ids, inputs))
        for identity, point in zip(target_ids, targets):
            if identity in byid and not np.array_equal(byid[identity], point):
                raise ValueError("A target ID refers to a different input coordinate")
    points = [row.copy() for row in inputs]
    index = {tuple(row): i for i, row in enumerate(points)}
    mapping = []
    for point in targets:
        key = tuple(point)
        if key not in index:
            index[key] = len(points)
            points.append(point.copy())
        mapping.append(index[key])
    return np.asarray(points), np.asarray(mapping, dtype=int)


def _psd_root(matrix):
    values, vectors = eigh((matrix+matrix.T)/2, check_finite=False)
    tolerance = 1e-9*max(1., float(np.max(np.abs(values))))
    if values.min() < -tolerance:
        raise np.linalg.LinAlgError(f"Predictive covariance is not PSD: minimum eigenvalue {values.min()}")
    return vectors*np.sqrt(np.maximum(values, 0))


def _conditional_append(existing, covariance, n_inputs, rng, scale):
    if n_inputs == len(covariance):
        return np.array(existing, copy=True)
    cross = covariance[n_inputs:, :n_inputs]
    factor = cho_factor(covariance[:n_inputs, :n_inputs], lower=True, check_finite=False)
    weights = cho_solve(factor, cross.T, check_finite=False).T
    conditional = covariance[n_inputs:, n_inputs:]-weights@cross.T
    return np.r_[existing, weights@existing+np.sqrt(scale)*(_psd_root(conditional)@rng.normal(size=len(cross)))]


def _latent_terminal(fitted, context, C, sigma2, K, rho, rng):
    """Exact spatial-mode Kalman filtering at fixed C and covariance.

    Refiltering prevents the incorrect practice of treating an old sampled
    latent state as certain after new observations have arrived.
    """
    p = fitted.prepared
    k, U = eigh((K+K.T)/2, check_finite=False)
    if k.min() < -1e-9*max(1., k.max()):
        raise ValueError("Invalid latent covariance")
    k = np.maximum(k, 0)
    start = p.split.fitting_start
    residual = context.historical_y[start:] - context.Z[start:]@C.T@p.Phi.T
    transformed = residual@U
    m, v = np.zeros(len(K)), sigma2*k
    innovation = sigma2*(1-rho*rho)*k
    for i, residual_t in enumerate(transformed):
        if i:
            m, v = rho*m, rho*rho*v+innovation
        gain = v/(v+sigma2)
        m += gain*(residual_t-m)
        v *= 1-gain
    return U@(m+np.sqrt(np.maximum(v, 0))*rng.normal(size=len(K)))


@dataclass(frozen=True)
class PredictiveSample:
    observation: np.ndarray
    readout_mean: np.ndarray | None
    conditional_mean: np.ndarray | None
    latent: np.ndarray | None
    target_coordinates: np.ndarray
    target_site_ids: tuple[str, ...]
    parameter_draw_ids: np.ndarray
    metadata: dict

    def __post_init__(self):
        self._initialize(owned=False)

    def _initialize(self, *, owned):
        shape = np.asarray(self.observation).shape
        if len(shape) != 3 or min(shape) < 1:
            raise ValueError("Predictive arrays must be nonempty [path,lead,target_site]")
        for name in _PREDICTIVE_COMPONENTS:
            original = getattr(self, name)
            if original is None and name != "observation":
                continue
            value = _freeze_owned(original, float) if owned else readonly(original)
            if value.shape != shape or not _all_finite(value):
                raise ValueError("Predictive paths must be finite and aligned")
            object.__setattr__(self, name, value)
        coordinates = readonly(self.target_coordinates)
        identities = tuple(map(str, self.target_site_ids))
        if coordinates.ndim != 2 or len(coordinates) != shape[2] or not _all_finite(coordinates) or len(identities) != shape[2]:
            raise ValueError("Predictive target geometry and site IDs must match the site axis")
        draw_ids = _freeze_owned(self.parameter_draw_ids, int) if owned else readonly(self.parameter_draw_ids, int)
        if draw_ids.shape != (shape[0], 2):
            raise ValueError("parameter_draw_ids must have [path,chain_and_draw] axes")
        object.__setattr__(self, "target_coordinates", coordinates)
        object.__setattr__(self, "target_site_ids", identities)
        object.__setattr__(self, "parameter_draw_ids", draw_ids)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def _from_owned(cls, **values):
        """Internal factory for arrays created here or freshly deserialized.

        The public dataclass constructor always defensively copies its
        arrays. This private route transfers allocations with no external
        mutable aliases and avoids doubling large predictive outputs.
        """
        instance = object.__new__(cls)
        for name in cls.__dataclass_fields__:
            object.__setattr__(instance, name, values[name])
        instance._initialize(owned=True)
        return instance

    @property
    def mean(self):
        return self.observation.mean(axis=0)

    def interval(self, level=.9, target="observation"):
        if not 0 < level < 1:
            raise ValueError("Interval level must be between zero and one")
        if target not in _PREDICTIVE_COMPONENTS or getattr(self, target) is None:
            raise ValueError(f"Predictive component {target!r} was not stored")
        return np.quantile(getattr(self, target), [(1-level)/2, (1+level)/2], axis=0)

    def save(self, path):
        return save_state(path, self.__dict__, kind="predictive_sample")

    @classmethod
    def load(cls, path):
        return cls._from_owned(**load_state(path, kind="predictive_sample"))


def simulate_predictive(fitted, context=None, *, horizon, target_coordinates=None,
                        target_site_ids=None, paths=256, seed=20261102,
                        future_variance_covariates=None, components=_PREDICTIVE_COMPONENTS,
                        path_start=0, max_output_bytes=_DEFAULT_OUTPUT_BYTES):
    """Draw autonomous forecast paths; no future response argument exists.

    For heteroskedastic models, future variance predictors must be supplied for
    the *joint* domain returned by joint_domain(), in [lead,joint_site,q] order.
    Their availability at the forecast origin is the caller's responsibility.

    ``components=("observation",)`` avoids storing optional latent/readout
    arrays. Default storage remains all four for backward compatibility.
    ``path_start`` indexes named global RNG streams and makes path batching
    exactly reproducible. ``max_output_bytes`` limits selected output storage
    only; covariance/feature workspaces have separate model gates.
    """
    from .workflow import FittedEnsemble, simulate_ensemble
    if isinstance(fitted, FittedEnsemble):
        return simulate_ensemble(fitted, context, horizon=horizon,
            target_coordinates=target_coordinates, target_site_ids=target_site_ids,
            paths=paths, seed=seed, future_variance_covariates=future_variance_covariates,
            components=components, path_start=path_start, max_output_bytes=max_output_bytes)
    if context is None:
        context = forecast_context(fitted)
    if context.parameter_posterior_identity != fitted.posterior_identity:
        raise ValueError("Forecast context belongs to another fitted posterior")
    horizon, paths = _integer(horizon, "horizon"), _integer(paths, "paths")
    path_start = _integer(path_start, "path_start", minimum=0)
    components = _prediction_components(components)
    p = fitted.prepared
    targets = p.data.coordinates if target_coordinates is None else np.asarray(target_coordinates, float)
    ids = (p.data.site_ids if target_coordinates is None else tuple(f"target_{i}" for i in range(len(targets)))) if target_site_ids is None else tuple(map(str, target_site_ids))
    joint, target_index = joint_domain(p.data.coordinates, targets, input_ids=p.data.site_ids, target_ids=ids)
    n, m = len(p.data.coordinates), len(joint)
    output_bytes = _check_output_budget(paths, horizon, len(targets), components, max_output_bytes)
    phi = p.basis.transform(joint)
    shape = (paths, horizon, len(targets))
    model = p.spec.innovations
    operator_route = fitted.posterior.metadata.get("spatial_backend") == "vecchia_operator"
    if operator_route and model != "iid":
        raise ValueError("The Vecchia operator forecast currently supports iid Gaussian innovations only")
    spatial_geometry = (p.spec.covariance.prepare_geometry(joint,
                        reference_coordinates=p.data.coordinates) if operator_route else None)
    F = None
    if model == "heteroskedastic":
        if future_variance_covariates is None:
            raise ValueError("Known future variance predictors are required for heteroskedastic forecasts")
        F = np.asarray(future_variance_covariates, float)
        if F.ndim != 3 or F.shape[:2] != (horizon, m) or not np.isfinite(F).all():
            raise ValueError("Future variance predictors must be [lead,joint_site,covariate]")
    elif future_variance_covariates is not None:
        raise ValueError("Future variance predictors are only used by heteroskedastic models")
    output = {key: np.empty(shape) if key in components else None for key in _PREDICTIVE_COMPONENTS}
    draw_ids = np.empty((paths, 2), int)
    chains, retained = fitted.posterior.n_chains, fitted.posterior.n_draws
    solver_reports = []
    for path in range(paths):
        global_path = path_start + path
        parameter_rng = np.random.default_rng(seed_for(seed, "parameter", global_path))
        innovation_rng = np.random.default_rng(seed_for(seed, "innovation", global_path))
        chain, draw = int(parameter_rng.integers(chains)), int(parameter_rng.integers(retained))
        draw_ids[path] = (chain, draw)
        parameter = fitted.posterior.draw_parameters(chain, draw, parameter_rng)
        C, sigma2, cp = parameter.C, float(parameter.sigma2), parameter.covariance_parameters
        if not np.isfinite(sigma2) or sigma2 <= 0:
            raise ValueError("Invalid posterior variance draw")
        if operator_route:
            latent_factor = spatial_geometry.factor(cp)
            nugget_sd = np.sqrt(spatial_geometry.nugget_diagonal(cp))
        else:
            if m == n:
                K, V = p.spec.covariance.matrices(joint, cp)
            else:
                K, V = p.spec.covariance.joint_matrices(p.data.coordinates, joint[n:], cp)
            LK = _psd_root(K)
            nugget_sd = np.sqrt(np.diag(V))
        q = float(np.exp(cp["log_q"]))
        temporal = parameter.temporal_parameters or {}
        rho = float(temporal.get("rho", 0.))
        nu = float(temporal.get("nu", 10.))
        if model in {"field_t", "local_t"} and nu <= 2:
            raise ValueError("Student degrees of freedom must exceed two")
        mean_map = phi@C
        previous_y = np.array(context.historical_y[-1], copy=True)
        feature_state = np.array(context.states[-1], copy=True)
        w_previous = e_previous = v_previous = None
        if model == "latent_ar":
            w_input = _latent_terminal(fitted, context, C, sigma2, q*K[:n, :n], rho, innovation_rng)
            w_previous = _conditional_append(w_input, K, n, innovation_rng, sigma2*q)
        elif model == "full_residual_ar":
            R = V+q*K
            residual = previous_y-p.Phi@C@context.Z[-1]
            e_previous = _conditional_append(residual, R, n, innovation_rng, sigma2)
            factor = cho_factor(R, lower=True, check_finite=False)
            weights = q*cho_solve(factor, K, check_finite=False).T
            covariance = q*K-weights@(q*K)
            w_previous = weights@e_previous+np.sqrt(sigma2)*(_psd_root(covariance)@innovation_rng.normal(size=m))
            v_previous = e_previous-w_previous
        for lead in range(horizon):
            feature_state, z = p.features.update(previous_y, feature_state)
            mean = mean_map@z
            field_weight = innovation_rng.gamma(nu/2, 2/nu) if model == "field_t" else 1.
            field = (latent_factor.sample(innovation_rng) if operator_route else
                     LK@innovation_rng.normal(size=m))
            w = np.sqrt(sigma2*q/field_weight)*field
            measurement = np.sqrt(sigma2/field_weight)*nugget_sd*innovation_rng.normal(size=m)
            conditional = mean.copy()
            if model == "full_residual_ar":
                innovation_scale = np.sqrt(1-rho*rho)
                conditional += rho*e_previous
                w = rho*w_previous+innovation_scale*w
                measurement = rho*v_previous+innovation_scale*measurement
                w_previous, v_previous = w, measurement
                e_previous = w+measurement
            elif model == "latent_ar":
                conditional += rho*w_previous
                w = rho*w_previous+np.sqrt(1-rho*rho)*w
                w_previous = w
            elif model == "local_t":
                measurement /= np.sqrt(innovation_rng.gamma(nu/2, 2/nu, size=m))
            elif model == "heteroskedastic":
                beta = np.asarray(temporal["variance_eta"])
                if F.shape[2] != len(beta):
                    raise ValueError("Future variance predictor count differs from fitted coefficients")
                normalized_variance_features = (F[lead]-np.asarray(temporal["variance_covariate_center"])) / np.asarray(temporal["variance_covariate_scale"])
                log_variance = normalized_variance_features@beta
                if not np.isfinite(log_variance).all() or np.max(np.abs(log_variance)) > 700:
                    raise FloatingPointError("Heteroskedastic forecast variance overflow")
                measurement *= np.exp(.5*log_variance)
            latent = mean+w
            observation = latent+measurement
            previous_y = observation[:n]
            for key, value in (("observation", observation), ("readout_mean", mean),
                               ("conditional_mean", conditional), ("latent", latent)):
                if output[key] is not None:
                    output[key][path, lead] = value[target_index]*p.response_scale+p.response_center
        if parameter.solver_metadata:
            solver_reports.append(parameter.solver_metadata)
    metadata = {"forecast_origin": context.origin, "origin_time": float(context.times[-1]),
                "parameter_training_end": fitted.training_end, "parameter_policy": "frozen",
                "parameter_posterior_identity": fitted.posterior_identity,
                "response_units": p.data.response_units, "seed": int(seed),
                "rng_scheme": "SHA256 named parameter/innovation streams indexed by path",
                "path_start": path_start, "path_stop": path_start + paths,
                "stored_components": list(components), "output_storage_bytes": output_bytes,
                "forecast_feedback": "simulated observations at the input sites",
                "target_definitions": {"readout_mean": "Phi C z on each stochastic feedback path",
                    "conditional_mean": "next observation mean conditional on the path's residual state",
                    "latent": "readout_mean plus the spatial process", "observation": "latent plus measurement noise"},
                "spatial_approximation": p.spec.covariance.kind,
                "spatial_backend": "vecchia_ancestral" if operator_route else
                                   ("dense_exact" if p.spec.covariance.kind == "matern" else "dense_reference"),
                "spatial_dense_materialization": not operator_route,
                "spatial_rng_scheme": "vecchia-ancestral-v1" if operator_route else "dense-eigenroot-v1",
                "joint_sites": m, "target_index": target_index.tolist(),
                "solver_reports": solver_reports,
                "new_origin_interpretation": "fixed parameter posterior with conditionally updated states"}
    return PredictiveSample._from_owned(**output, target_coordinates=targets, target_site_ids=ids,
                                        parameter_draw_ids=draw_ids, metadata=metadata)


def iter_predictive(fitted, context=None, *, paths=256, batch_size=32, path_start=0, **kwargs):
    """Yield bounded path batches without changing any global-index trajectory.

    Consume or write a batch before requesting the next to keep memory
    bounded. Concatenating all batches recreates full output storage. The
    spatial domain remains joint in every batch; this never tiles sites
    independently. Every batch can be saved with ``PredictiveSample.save``.
    """
    paths, batch_size = _integer(paths, "paths"), _integer(batch_size, "batch_size")
    path_start = _integer(path_start, "path_start", minimum=0)
    for start in range(0, paths, batch_size):
        yield simulate_predictive(fitted, context, paths=min(batch_size, paths - start),
                                  path_start=path_start + start, **kwargs)

