"""Data-only KS vector controls with an explicit periodic output decoder.

Ridge solves minimize mean squared residuals plus lambda times the squared
non-intercept coefficients. POD is fitted only on the supplied training panel.
The decoder is used for reported outputs, never as a hidden recurrence step.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg, sparse

from ..benchmark.contracts import Forecast, Panel
from ..io import save_state, load_state


def angles(coordinates):
    c = np.asarray(coordinates, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 2 or not np.allclose(np.linalg.norm(c, axis=1), 1., atol=1e-10):
        raise ValueError("KS baselines require unit-circle coordinates")
    return np.mod(np.arctan2(c[:, 1], c[:, 0]), 2*np.pi)


def periodic_decoder(observed_coordinates, target_coordinates):
    """Piecewise linear interpolation around the circle, including its seam."""
    x, q = angles(observed_coordinates), angles(target_coordinates)
    order = np.argsort(x); xp = x[order]
    if len(x) < 2 or np.any(np.diff(xp) <= 1e-12):
        raise ValueError("At least two distinct observed sites required")
    right = np.searchsorted(xp, q, side="right") % len(x)
    left = (right-1) % len(x)
    distance = np.mod(xp[right]-xp[left], 2*np.pi)
    weight = np.mod(q-xp[left], 2*np.pi)/distance
    out = np.zeros((len(q), len(x)), dtype=np.float64)
    out[np.arange(len(q)), order[left]] = 1-weight
    out[np.arange(len(q)), order[right]] = weight
    if not np.allclose(out.sum(axis=1), 1.) or np.min(out) < -1e-12:
        raise ArithmeticError("Invalid periodic decoder weights")
    return out


def polynomial_features(delays):
    """One shared constant/linear/upper-triangle quadratic map for fit/rollout."""
    x = np.asarray(delays, dtype=np.float64)
    d = x.shape[-1]; i, j = np.triu_indices(d)
    return np.concatenate((np.ones(x.shape[:-1]+(1,)), x, x[..., i]*x[..., j]), axis=-1)


def feature_count(rank, delays):
    d = int(rank)*int(delays)
    return 1+d+d*(d+1)//2


def ridge_stream(chunks, n_features, n_outputs, ridge):
    """Accumulate sufficient statistics in float64 without a full design."""
    gram = np.zeros((n_features, n_features)); cross = np.zeros((n_features, n_outputs)); n = 0
    for x, y in chunks:
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        gram += x.T@x; cross += x.T@y; n += len(x)
    if not n or not np.isfinite(gram).all() or ridge <= 0:
        raise ValueError("Finite training features and positive ridge required")
    gram /= n; cross /= n
    gram.flat[::n_features+1] += ridge
    gram[0, 0] -= ridge  # declared unpenalized intercept
    weights = linalg.cho_solve(linalg.cho_factor(gram, lower=True, check_finite=False), cross, check_finite=False)
    return weights, n


def _panel_state(panel):
    return {"y": panel.y, "coordinates": panel.coordinates, "times": panel.times,
            "site_ids": panel.site_ids, "response_units": panel.response_units,
            "coordinate_units": panel.coordinate_units, "metadata": dict(panel.metadata)}


class VectorFit:
    """Mean, persistence, linear VAR/POD-VAR, POD-NVAR or sparse ridge ESN."""
    def __init__(self, model_id, training, config, seed):
        self.model_id, self.training, self.config, self.seed = model_id, training, dict(config), int(seed)
        self.diagnostics = {"qualified": True, "inference": "deterministic",
                            "spatial_extension": "periodic_piecewise_linear_output_only",
                            "ridge_objective": "sum_squared_error/N + lambda*sum(nonintercept_weights^2)"}
        y = np.asarray(training.y, np.float64); n, s = y.shape
        angles(training.coordinates)
        self.mean = y.mean(axis=0)
        if model_id in {"training_mean", "persistence"}:
            self.diagnostics.update(parameter_count=s if model_id == "training_mean" else 0, usable_targets=n)
            return
        if model_id == "ridge_esn":
            self._fit_esn(y)
            return
        if model_id not in {"linear_var", "pod_nvar"}:
            raise ValueError(f"Unknown vector family {model_id}")
        rank = int(config.get("rank", s))
        if not 1 <= rank <= s:
            raise ValueError("POD rank exceeds observed dimension")
        self.rank = rank
        if rank < s:
            _, _, vt = linalg.svd(y-self.mean, full_matrices=False, check_finite=False)
            self.components = vt[:rank].T
            state = (y-self.mean)@self.components
        else:
            self.components = np.eye(s); state = y-self.mean
        self.lags = int(config.get("lags", 1)); self.spacing = int(config.get("spacing", 1))
        if self.lags < 1 or self.spacing < 1:
            raise ValueError("Positive lag count and spacing required")
        start = 1+(self.lags-1)*self.spacing
        self.target = config.get("target", "increment" if model_id == "pod_nvar" else "state")
        if self.target not in {"state", "increment"}:
            raise ValueError("Target must be state or increment")
        p = feature_count(rank, self.lags) if model_id == "pod_nvar" else 1+rank*self.lags
        cap = int(config.get("max_features", 1024))
        if p > cap:
            raise MemoryError(f"Feature count {p} exceeds declared cap {cap}")
        chunk_size = int(config.get("chunk_size", 512))
        def chunks():
            for a in range(start, n, chunk_size):
                t = np.arange(a, min(a+chunk_size, n))
                delayed = np.concatenate([state[t-1-k*self.spacing] for k in range(self.lags)], axis=1)
                x = self._map(delayed)
                target = state[t]-(state[t-1] if self.target == "increment" else 0)
                yield x, target
        self.weights, count = ridge_stream(chunks(), p, rank, float(config["ridge"]))
        self.diagnostics.update(parameter_count=int(self.weights.size+self.components.size), usable_targets=count,
                                temporal_features=p, latent_rank=rank, target=self.target,
                                gram_bytes=8*p*p, chunk_design_bytes=8*chunk_size*p,
                                hypothetical_full_design_bytes=8*count*p, training_lag_loss=start)

    def _map(self, state):
        if self.model_id == "pod_nvar":
            return polynomial_features(state)
        return np.concatenate((np.ones(state.shape[:-1]+(1,)), state), axis=-1)

    def _fit_esn(self, y):
        cfg = self.config; n, s = y.shape; units = int(cfg.get("units", 256))
        if units < 2:
            raise ValueError("Reservoir requires at least two units")
        rng = np.random.default_rng(self.seed)
        connectivity = float(cfg.get("connectivity", .02))
        matrix = rng.uniform(-1, 1, (units, units))*(rng.random((units, units)) < connectivity)
        radius = float(np.max(np.abs(linalg.eigvals(matrix, check_finite=False))))
        if radius <= 1e-12:
            raise ValueError("Degenerate reservoir; seed is not silently redrawn")
        self.recurrent = matrix*(float(cfg.get("spectral_radius", .9))/radius)
        self._sparse = sparse.csr_matrix(self.recurrent)
        self.input = rng.uniform(-1, 1, (units, s+1))*float(cfg.get("input_scale", .2))
        self.leak = float(cfg.get("leak", .5))
        self.readout = cfg.get("readout", "states_and_squares")
        if self.readout not in {"states_and_squares", "alternating_squares"}:
            raise ValueError("Unknown readout feature map")
        self.target = cfg.get("target", "state")
        if self.target not in {"state", "increment"}:
            raise ValueError("Unknown ESN target")
        washout = int(cfg.get("washout", 128)); chunk_size = int(cfg.get("chunk_size", 512))
        if not 1 <= washout < n:
            raise ValueError("Washout must leave training targets")
        p = 1+(2*units if self.readout == "states_and_squares" else units)
        def chunks():
            state = np.zeros(units); xs, ys = [], []
            for t in range(1, n):
                state = self._update(y[t-1], state)
                if t >= washout:
                    xs.append(self._esn_features(state)); ys.append(y[t]-(y[t-1] if self.target == "increment" else 0))
                if len(xs) == chunk_size:
                    yield np.asarray(xs), np.asarray(ys); xs, ys = [], []
            if xs:
                yield np.asarray(xs), np.asarray(ys)
        self.weights, count = ridge_stream(chunks(), p, s, float(cfg["ridge"]))
        self.diagnostics.update(parameter_count=int(self.weights.size), random_feature_parameters=int(self.recurrent.size+self.input.size),
                                reservoir_nonzeros=int(self._sparse.nnz), usable_targets=count,
                                temporal_features=p, gram_bytes=8*p*p, chunk_design_bytes=8*chunk_size*p,
                                hypothetical_full_design_bytes=8*count*p, training_washout=washout,
                                target=self.target, readout=self.readout, intercept="explicit_unpenalized",
                                recurrent_scaling="spectral_radius", realized_radius=float(cfg.get("spectral_radius", .9)))

    def _update(self, y, state):
        return (1-self.leak)*state+self.leak*np.tanh(self._sparse@state+self.input[:, 0]+self.input[:, 1:]@y)

    def _esn_features(self, state):
        if self.readout == "states_and_squares":
            return np.concatenate(([1.], state, state*state))
        x = state.copy(); x[1::2] *= x[1::2]
        return np.concatenate(([1.], x))

    def forecast(self, history_panel, horizon, target_coordinates, target_site_ids, n_paths=0, seed=0):
        if history_panel.site_ids != self.training.site_ids or not np.array_equal(history_panel.coordinates, self.training.coordinates):
            raise ValueError("History sites must match fitted observed sites")
        for key in ("data_identity_sha256", "ks_protocol", "trajectory", "phase"):
            if key in self.training.metadata and history_panel.metadata.get(key) != self.training.metadata[key]:
                raise ValueError(f"History {key} differs from the fitted training episode/protocol")
        y = np.asarray(history_panel.y, np.float64)
        h = int(horizon); predictions = np.empty((h, y.shape[1]))
        if self.model_id == "training_mean":
            predictions[:] = self.mean
        elif self.model_id == "persistence":
            predictions[:] = y[-1]
        elif self.model_id == "ridge_esn":
            state = np.zeros(len(self.recurrent))
            for value in y:
                state = self._update(value, state)
            previous = y[-1]
            for j in range(h):
                nxt = self._esn_features(state)@self.weights
                if self.target == "increment":
                    nxt = previous+nxt
                predictions[j] = nxt
                if not np.isfinite(nxt).all():
                    raise FloatingPointError(f"Nonfinite autonomous ESN at lead {j+1}")
                previous = nxt; state = self._update(nxt, state)
        else:
            needed = 1+(self.lags-1)*self.spacing
            if len(y) < needed:
                raise ValueError("History shorter than delay context")
            latent = list((y[-needed:]-self.mean)@self.components)
            for j in range(h):
                delayed = np.concatenate([latent[-1-k*self.spacing] for k in range(self.lags)])
                with np.errstate(over="ignore", invalid="ignore"):
                    nxt = self._map(delayed)@self.weights
                    if self.target == "increment":
                        nxt = latent[-1]+nxt
                    predictions[j] = nxt@self.components.T+self.mean
                if not np.isfinite(nxt).all() or not np.isfinite(predictions[j]).all():
                    raise FloatingPointError(f"Nonfinite autonomous {self.model_id} at lead {j+1}")
                latent.append(nxt); latent = latent[-needed:]
        decoder = periodic_decoder(self.training.coordinates, target_coordinates)
        output = predictions@decoder.T
        return Forecast(output, None, target_coordinates, target_site_ids,
                        {"model": self.model_id, "qualified": True, "uncertainty": "point_only",
                         "parameter_policy": "frozen", "history_rows": len(y),
                         "spatial_extension": "periodic_piecewise_linear_output_only", "path_count": 0})

    def to_state(self):
        attrs = {k: v for k, v in self.__dict__.items() if k not in {"training", "_sparse"}}
        return {"training": _panel_state(self.training), "attributes": attrs}

    @classmethod
    def from_state(cls, state):
        instance = cls.__new__(cls); instance.__dict__.update(state["attributes"])
        instance.training = Panel(**state["training"])
        if instance.model_id == "ridge_esn":
            instance._sparse = sparse.csr_matrix(instance.recurrent)
        return instance

    def save(self, path):
        return save_state(path, self.to_state(), kind="ks_vector_fit")

    @classmethod
    def load(cls, path):
        return cls.from_state(load_state(path, kind="ks_vector_fit"))


def fit_vector(model_id, training_panel, config, seed=0, checkpoint_path=None):
    return VectorFit(model_id, training_panel, config, seed)
