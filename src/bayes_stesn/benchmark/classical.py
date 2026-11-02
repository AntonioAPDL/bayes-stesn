"""Classical forecasting comparators with explicit spatial capabilities.

All transformations see only the supplied training Panel.  ``history`` may
extend that exact prefix to condition states, without refitting parameters.
No point forecast is silently converted into a predictive distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.spatial.distance import cdist, pdist
from .contracts import Panel, Forecast, CapabilityError, validate_history, forecast_targets
from ..io import save_state, load_state, seed_for


CLASSICAL_CONFIG_KEYS = {
    "persistence": frozenset(),
    "seasonal_naive": frozenset({"period", "min_period", "max_period"}),
    "spatial_persistence": frozenset({"spatial_length", "ridge", "spatial_nu"}),
    "basis_krr": frozenset({"rank", "basis_width", "lags", "bandwidth",
        "bandwidth_multiplier", "ridge", "penalty", "exact", "max_exact_cases", "nystrom_rank"}),
    "basis_lgssm": frozenset({"rank", "basis_width", "initial_variance",
        "transition_norm_cap", "variance_floor", "max_iter", "tolerance",
        "mstep_steps", "max_forecast_bytes"}),
    "separable_stgp": frozenset({"max_fit_sites", "center", "spatial_nu", "temporal_nu",
        "spatial_length", "temporal_length", "signal_sd", "noise_sd",
        "fit_hyperparameters", "max_iter", "ftol", "gtol", "max_joint_dimension",
        "max_forecast_bytes"}),
}


def _integer_option(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _classical_config(model_id, config):
    """Keep typo detection identical for fitting and portable-state loading."""
    result = dict(config or {})
    allowed = CLASSICAL_CONFIG_KEYS.get(model_id)
    # Other explicitly validated forecasters reuse ClassicalFit's state helpers.
    if allowed is None:
        return result
    unknown = set(result)-allowed
    if unknown:
        raise ValueError(f"Unknown {model_id} configuration: {sorted(unknown)}")
    integers = {"period", "min_period", "max_period", "rank", "lags", "max_exact_cases",
        "nystrom_rank", "max_iter", "mstep_steps", "max_fit_sites",
        "max_joint_dimension", "max_forecast_bytes"}
    for key in integers & set(result):
        if key == "period" and result[key] is None:
            continue
        result[key] = _integer_option(result[key], key)
    for key in {"exact", "fit_hyperparameters"} & set(result):
        if not isinstance(result[key], (bool, np.bool_)):
            raise ValueError(f"{key} must be boolean")
        result[key] = bool(result[key])
    if model_id == "seasonal_naive" and result.get("min_period", 8) > result.get("max_period", 200):
        raise ValueError("min_period must not exceed max_period")
    return result


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _horizon(value):
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError("horizon must be a positive integer")
    return int(value)


def _paths(value):
    if isinstance(value, bool) or int(value) != value or value < 0:
        raise ValueError("n_paths must be a nonnegative integer")
    return int(value)


def matern_correlation(x, y, length, nu=1.5):
    """Normalized Euclidean Matérn correlation; length is not effective range."""
    d = cdist(np.asarray(x, float), np.asarray(y, float)) / _positive(length, "length")
    if nu == .5:
        return np.exp(-d)
    if nu == 1.5:
        d = np.sqrt(3) * d
        return (1 + d) * np.exp(-d)
    if nu == 2.5:
        d = np.sqrt(5) * d
        return (1 + d + d*d/3) * np.exp(-d)
    raise ValueError("Spatial Matérn nu must be 0.5, 1.5 or 2.5")


def psd_root(covariance, *, tolerance=1e-9):
    """Symmetric covariance root, rejecting materially negative eigenvalues."""
    c = np.asarray(covariance, float)
    values, vectors = eigh((c+c.T)/2, check_finite=False)
    scale = max(float(np.max(np.abs(values))), 1.)
    if values[0] < -tolerance * scale:
        raise np.linalg.LinAlgError("Predictive covariance is not positive semidefinite")
    return vectors * np.sqrt(np.maximum(values, 0.))


def _panel_state(data):
    return {"y": data.y, "coordinates": data.coordinates, "times": data.times,
            "site_ids": data.site_ids, "response_units": data.response_units,
            "coordinate_units": data.coordinate_units}


@dataclass
class CoordinateBasis:
    """Geometry-only RBF basis, orthonormal on fitted sites after scaling.

    Its extension at arbitrary coordinates is explicit.  Training-site
    loadings satisfy Phi'Phi / n = I, simplifying low-rank filtering.
    """
    offset: np.ndarray
    scale: float
    centers: np.ndarray
    width: float
    transform: np.ndarray

    @classmethod
    def fit(cls, coordinates, rank=16, width=.25):
        c = np.asarray(coordinates, float)
        rank = _integer_option(rank, "basis rank")
        rank = min(rank, len(c))
        offset = c.min(axis=0)
        scale = max(float(np.max(np.ptp(c, axis=0))), 1e-12)
        x = (c-offset)/scale
        # Coordinate sorting makes tie-breaking independent of site order.
        ordered = x[np.lexsort(tuple(x[:, j] for j in range(x.shape[1]-1, -1, -1)))]
        chosen = []
        if rank > 1:
            chosen = [int(np.argmin(np.sum((ordered-ordered.mean(axis=0))**2, axis=1)))]
            distance = np.sum((ordered-ordered[chosen[0]])**2, axis=1)
            for _ in range(rank-2):
                j = int(np.argmax(distance)); chosen.append(j)
                distance = np.minimum(distance, np.sum((ordered-ordered[j])**2, axis=1))
        centers = ordered[chosen] if chosen else np.empty((0, c.shape[1]))
        obj = cls(offset, scale, centers, _positive(width, "basis_width"), np.eye(rank))
        raw = obj._raw(c)
        _, singular, vt = np.linalg.svd(raw, full_matrices=False)
        keep = singular > max(singular[0]*1e-10, 1e-12)
        obj.transform = vt[keep].T * (np.sqrt(len(c))/singular[keep])
        return obj

    def _raw(self, coordinates):
        x = (np.asarray(coordinates, float)-self.offset)/self.scale
        radial = np.exp(-.5*(cdist(x, self.centers)/self.width)**2)
        return np.column_stack([np.ones(len(x)), radial])

    def evaluate(self, coordinates):
        return self._raw(coordinates) @ self.transform

    def to_state(self):
        return {k: getattr(self, k) for k in ("offset", "scale", "centers", "width", "transform")}

    @classmethod
    def from_state(cls, state):
        return cls(**state)


class _Conditioned:
    def __init__(self, fitted, history):
        self.fitted, self.history = fitted, validate_history(fitted.data, history)

    def forecast(self, *args, **kwargs):
        if "history" in kwargs:
            raise ValueError("A conditioned fit already specifies its history")
        return self.fitted.forecast(*args, history=self.history, **kwargs)


class ClassicalFit:
    """Portable numeric-state base class; no Python objects are unpickled."""
    model_id = "abstract"
    _state_fields = ()

    def __init__(self, data, config=None, seed=0):
        if not isinstance(data, Panel):
            raise TypeError("fit requires an observed Panel")
        if len(data.y) < 3:
            raise ValueError("At least three observed times are required")
        self.data, self.config, self.seed = data, _classical_config(self.model_id, config), int(seed)
        self.diagnostics = {"method": self.model_id, "parameter_policy": "frozen_after_fit"}

    def condition(self, history):
        return _Conditioned(self, history)

    def _metadata(self, history, **extra):
        return {"model": self.model_id, "training_rows": len(self.data.y),
                "origin_index": len(history.y), "parameter_policy": "frozen",
                "target": "observation", **extra}

    def to_state(self):
        state = {k: getattr(self, k) for k in self._state_fields}
        if hasattr(self, "basis"):
            state["basis"] = self.basis.to_state()
        return {"version": 1, "model_id": self.model_id, "data": _panel_state(self.data),
                "config": self.config, "seed": self.seed, "parameters": state,
                "diagnostics": self.diagnostics}

    def save(self, path):
        return save_state(path, self.to_state(), kind="benchmark_classical")


class NaiveFit(ClassicalFit):
    _state_fields = ("period",)

    def __init__(self, data, config=None, seed=0, *, seasonal=False):
        self.model_id = "seasonal_naive" if seasonal else "persistence"
        super().__init__(data, config, seed)
        self.period = 1
        if seasonal:
            period = self.config.get("period")
            if period is not None:
                if int(period) != period or not 1 <= period <= len(data.y):
                    raise ValueError("period must be an available positive integer")
                self.period = int(period)
                source = "explicit_training_selected_or_known_period"
            else:
                self.period, candidates = self._choose_period()
                self.diagnostics["period_candidates"] = candidates
                source = "training_prefix_periodogram_and_training_tail_validation"
            self.diagnostics.update(period=self.period, period_source=source)
        self.diagnostics.update(uncertainty="point_only", converged=True, qualified=True)

    def _choose_period(self):
        y = self.data.y
        split = max(4, int(.7*len(y)))
        prefix = y[:split]-y[:split].mean(axis=0)
        u, s, _ = np.linalg.svd(prefix, full_matrices=False)
        score = u[:, 0]*s[0]
        power = np.abs(np.fft.rfft(score))**2
        frequency = np.fft.rfftfreq(split)
        min_period = int(self.config.get("min_period", 8))
        max_period = min(int(self.config.get("max_period", 200)), split)
        indices = np.flatnonzero((frequency > 0) & (frequency >= 1/max_period) & (frequency <= 1/min_period)) if max_period >= min_period else np.array([], int)
        strongest = indices[np.argsort(power[indices])[-8:]]
        candidates = sorted({1, *(int(round(1/frequency[i])) for i in strongest)})
        h = len(y)-split
        errors = []
        for period in candidates:
            pred = y[split-period + np.arange(h) % period]
            errors.append(float(np.mean((pred-y[split:])**2)))
        winner = int(np.argmin(errors))
        # The fallback must improve on persistence, not merely be periodic.
        if errors[winner] >= .99*errors[0]:
            winner = 0
        return candidates[winner], candidates

    def forecast(self, horizon, target_coordinates=None, target_site_ids=None,
                 n_paths=256, seed=0, history=None):
        h = _horizon(horizon); observed = validate_history(self.data, history)
        coords, ids = forecast_targets(self.data, target_coordinates, target_site_ids)
        mapping = {tuple(c): i for i, c in enumerate(self.data.coordinates)}
        if any(tuple(c) not in mapping for c in coords):
            raise CapabilityError(f"{self.model_id} requires target-site histories")
        indices = [mapping[tuple(c)] for c in coords]
        rows = len(observed.y)-self.period + np.arange(h) % self.period
        mean = observed.y[rows][:, indices]
        return Forecast(mean=mean, samples=None, coordinates=coords, site_ids=ids,
                        metadata=self._metadata(observed, uncertainty="point_only", period=self.period))


class SpatialPersistenceFit(ClassicalFit):
    model_id = "spatial_persistence"
    _state_fields = ("offset", "coordinate_scale", "length", "ridge", "nu", "center")

    def __init__(self, data, config=None, seed=0):
        super().__init__(data, config, seed)
        self.offset = data.coordinates.min(axis=0)
        self.coordinate_scale = max(float(np.max(np.ptp(data.coordinates, axis=0))), 1e-12)
        self.length = _positive(self.config.get("spatial_length", .2), "spatial_length")
        self.ridge = _positive(self.config.get("ridge", .01), "ridge")
        self.nu = float(self.config.get("spatial_nu", 1.5))
        self.center = float(data.y.mean())
        self._after_load()
        self.diagnostics.update(converged=True, qualified=True, uncertainty="point_only")

    def _after_load(self):
        self._observed_coordinates = (self.data.coordinates-self.offset)/self.coordinate_scale
        kernel = matern_correlation(self._observed_coordinates, self._observed_coordinates,
                                    self.length, self.nu)
        self._factor = cho_factor(kernel+self.ridge*np.eye(len(kernel)), lower=True)

    def forecast(self, horizon, target_coordinates=None, target_site_ids=None,
                 n_paths=256, seed=0, history=None):
        h = _horizon(horizon); observed = validate_history(self.data, history)
        coords, ids = forecast_targets(self.data, target_coordinates, target_site_ids)
        o = self._observed_coordinates
        u = (coords-self.offset)/self.coordinate_scale
        weights = cho_solve(self._factor, observed.y[-1]-self.center)
        field = self.center + matern_correlation(u, o, self.length, self.nu) @ weights
        return Forecast(mean=np.tile(field, (h, 1)), samples=None, coordinates=coords, site_ids=ids,
                        metadata=self._metadata(observed, uncertainty="point_only", extension="coordinate_kernel_smoother"))


class KernelFit(ClassicalFit):
    model_id = "basis_krr"
    _state_fields = ("center", "lags", "x_center", "x_scale", "bandwidth", "ridge", "mode",
                     "landmarks", "kernel_transform", "weights", "n_training_pairs")

    def __init__(self, data, config=None, seed=0):
        super().__init__(data, config, seed)
        self.basis = CoordinateBasis.fit(data.coordinates, self.config.get("rank", 12), self.config.get("basis_width", .25))
        phi = self.basis.evaluate(data.coordinates)
        self.center = float(data.y.mean())
        self.lags = int(self.config.get("lags", 4))
        if not 1 <= self.lags < len(data.y)-2:
            raise ValueError("lags must leave at least three regression cases")
        coefficients = (data.y-self.center) @ phi / len(phi)
        x = np.concatenate([coefficients[self.lags-j-1:len(coefficients)-j-1] for j in range(self.lags)], axis=1)
        targets = coefficients[self.lags:]
        self.n_training_pairs = len(x)
        self.x_center = x.mean(axis=0)
        self.x_scale = np.maximum(x.std(axis=0), 1e-8)
        x = (x-self.x_center)/self.x_scale
        rng = np.random.default_rng(seed_for(seed, self.model_id, "landmarks"))
        subset = x[rng.choice(len(x), size=min(len(x), 256), replace=False)]
        distances = pdist(subset)
        positive = distances[distances > 1e-10]
        median = float(np.median(positive)) if len(positive) else 1.
        self.bandwidth = _positive(self.config.get("bandwidth", median*float(self.config.get("bandwidth_multiplier", 1.))), "bandwidth")
        self.ridge = _positive(self.config.get("ridge", self.config.get("penalty", 1e-3)), "ridge")
        if self.config.get("exact", False):
            if len(x) > int(self.config.get("max_exact_cases", 2000)):
                raise MemoryError("Exact KRR exceeds max_exact_cases; select Nyström approximation")
            self.mode = "exact"
            self.landmarks, self.kernel_transform = x, None
            kernel = self._kernel(x, x)
            self.weights = cho_solve(cho_factor(kernel+len(x)*self.ridge*np.eye(len(x)), lower=True), targets)
            effective_rank = len(x)
        else:
            self.mode = "nystrom"
            m = min(int(self.config.get("nystrom_rank", 128)), len(x))
            if m < 1:
                raise ValueError("nystrom_rank must be positive")
            self.landmarks = x[rng.choice(len(x), m, replace=False)]
            values, vectors = eigh(self._kernel(self.landmarks, self.landmarks))
            keep = values > max(values[-1]*1e-12, 1e-14)
            self.kernel_transform = vectors[:, keep]/np.sqrt(values[keep])
            features = self._kernel(x, self.landmarks) @ self.kernel_transform
            gram = features.T @ features + len(x)*self.ridge*np.eye(features.shape[1])
            self.weights = cho_solve(cho_factor(gram, lower=True), features.T @ targets)
            effective_rank = features.shape[1]
        self.diagnostics.update(converged=True, qualified=True, uncertainty="point_only", approximation=self.mode,
                                n_training_pairs=len(x), effective_rank=effective_rank,
                                library_ridge_equivalent=float(len(x)*self.ridge),
                                response_model="two_stage_projected_coefficients")

    def _kernel(self, x, y):
        return np.exp(-.5*cdist(x, y, metric="sqeuclidean")/self.bandwidth**2)

    def forecast(self, horizon, target_coordinates=None, target_site_ids=None,
                 n_paths=256, seed=0, history=None):
        h = _horizon(horizon); observed = validate_history(self.data, history)
        coords, ids = forecast_targets(self.data, target_coordinates, target_site_ids)
        phi = self.basis.evaluate(self.data.coordinates)
        target_phi = self.basis.evaluate(coords)
        a = (observed.y[-self.lags:]-self.center) @ phi / len(phi)
        result = np.empty((h, len(coords)))
        for lead in range(h):
            x = (np.concatenate(a[::-1])-self.x_center)/self.x_scale
            features = self._kernel(x[None], self.landmarks)
            if self.kernel_transform is not None:
                features = features @ self.kernel_transform
            next_a = (features @ self.weights)[0]
            result[lead] = self.center + target_phi @ next_a
            a = np.vstack([a[1:], next_a])
        return Forecast(mean=result, samples=None, coordinates=coords, site_ids=ids,
                        metadata=self._metadata(observed, uncertainty="point_only", approximation=self.mode))


def fit_classical(model_id, data, config=None, seed=0):
    """Fit one declared comparator. Extra target responses are never accepted."""
    if model_id in {"persistence", "seasonal_naive"}:
        return NaiveFit(data, config, seed, seasonal=model_id == "seasonal_naive")
    if model_id == "spatial_persistence":
        return SpatialPersistenceFit(data, config, seed)
    if model_id == "basis_krr":
        return KernelFit(data, config, seed)
    if model_id == "basis_lgssm":
        from .linear_state import LinearStateFit
        return LinearStateFit(data, config, seed)
    if model_id == "separable_stgp":
        from .gaussian_gp import SeparableGPFit
        return SeparableGPFit(data, config, seed)
    raise ValueError(f"Unknown classical comparator: {model_id}")


def classical_from_state(state):
    if state.get("version") != 1:
        raise ValueError("Unsupported classical comparator state version")
    model_id = state["model_id"]
    if model_id == "basis_lgssm":
        from .linear_state import LinearStateFit
        kind = LinearStateFit
    elif model_id == "separable_stgp":
        from .gaussian_gp import SeparableGPFit
        kind = SeparableGPFit
    else:
        kinds = {"persistence":NaiveFit, "seasonal_naive":NaiveFit,
                 "spatial_persistence":SpatialPersistenceFit, "basis_krr":KernelFit}
        if model_id not in kinds:
            raise ValueError("Unknown model in stored comparator state")
        kind = kinds[model_id]
    obj = kind.__new__(kind)
    obj.model_id = model_id
    ClassicalFit.__init__(obj, Panel(**state["data"]), state["config"], state["seed"])
    expected = set(obj._state_fields) | ({"basis"} if model_id in {"basis_lgssm", "basis_krr"} else set())
    if set(state["parameters"]) != expected:
        raise ValueError("Stored comparator parameter fields do not match its schema")
    for key, value in state["parameters"].items():
        setattr(obj, key, CoordinateBasis.from_state(value) if key == "basis" else value)
    obj.diagnostics = dict(state["diagnostics"])
    if hasattr(obj, "_after_load"):
        obj._after_load()
    return obj


def load_classical(path):
    return classical_from_state(load_state(path, kind="benchmark_classical"))
