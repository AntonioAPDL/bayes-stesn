"""Recursive quadratic echo-state ensemble comparator.

This is an explicitly recursive adaptation, not a claim to reproduce a
paper's direct-horizon E-QESN experiments.  PCA and all scales are estimated
from the supplied training observations only.  For member m,

    h[t] = (1-leak) h[t-1] + leak tanh(W h[t-1] + U x[t] + b),
    x[t] = concat(a[t-1], ..., a[t-lags]),
    a_hat[t] = [1, h[t], h[t]**2] beta.

The optional squared terms and an intercept form a ridge readout.  A member
is held fixed along its complete recursive path.  The native predictive
object is the actual finite ensemble; no innovation noise or interpolated
extra members are invented.  It therefore does not include observation
noise or full Bayesian parameter uncertainty.

Accepted configuration keys and their defaults are ``DEFAULT_CONFIG``.
``ridge`` penalizes the normalized objective mean squared score error;
the intercept is not penalized.  ``sparsity`` is a nonzero probability.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve

from ..io import seed_for
from .contracts import Panel, Forecast, CapabilityError, validate_history, forecast_targets


DEFAULT_CONFIG = {
    "width": 64, "lags": 3, "modes": 8, "ridge": 1e-3,
    "members": 30, "leak": .3, "input_scale": .5,
    "spectral_radius": .9, "sparsity": .1, "washout": 32,
    "quadratic": True,
}


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _config(config):
    supplied = {} if config is None else dict(config)
    unknown = set(supplied)-set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown recursive E-QESN options: {sorted(unknown)}")
    result = {**DEFAULT_CONFIG, **supplied}
    for key in ("width", "lags", "modes", "members"):
        result[key] = _integer(result[key], key)
    result["washout"] = _integer(result["washout"], "washout", 0)
    for key in ("ridge", "input_scale", "leak", "spectral_radius", "sparsity"):
        value = float(result[key])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
        result[key] = value
    if result["leak"] > 1 or result["sparsity"] > 1 or result["spectral_radius"] >= 1:
        raise ValueError("leak and sparsity must be <=1; spectral_radius must be <1")
    if not isinstance(result["quadratic"], (bool, np.bool_)):
        raise ValueError("quadratic must be boolean")
    result["quadratic"] = bool(result["quadratic"])
    return result


def _panel_state(data):
    return {"y": data.y, "coordinates": data.coordinates, "times": data.times,
            "site_ids": data.site_ids, "response_units": data.response_units,
            "coordinate_units": data.coordinate_units}


def _observed_targets(training, target_coordinates=None, target_site_ids=None):
    coordinates, ids = forecast_targets(training, target_coordinates, target_site_ids)
    lookup = {tuple(point): i for i, point in enumerate(training.coordinates)}
    if any(tuple(point) not in lookup for point in coordinates):
        raise CapabilityError("This fixed-site model has no native forecast at new coordinates")
    indices = np.array([lookup[tuple(point)] for point in coordinates], dtype=int)
    return coordinates, ids, indices


def reservoir_step(hidden, inputs, recurrent, input_weights, bias, leak):
    """One declared reservoir transition, useful for equation/reduction checks."""
    return (1-leak)*hidden + leak*np.tanh(recurrent@hidden + input_weights@inputs + bias)


def _lagged(scores, t, lags):
    return np.concatenate([scores[t-lag] if t >= lag else np.zeros(scores.shape[1])
                           for lag in range(1, lags+1)])


def _features(hidden, quadratic):
    return np.r_[1., hidden, hidden*hidden] if quadratic else np.r_[1., hidden]


def _ridge_readout(features, responses, penalty):
    """Minimize ||Y-X B||_F²/N + penalty ||B[1:]||_F² exactly."""
    n = len(features)
    gram = features.T@features/n
    gram.flat[::len(gram)+1] += penalty
    gram[0, 0] -= penalty
    return solve(gram, features.T@responses/n, assume_a="pos", check_finite=False)


@dataclass
class QuadraticReservoirEnsemble:
    training: Panel
    config: dict
    center: np.ndarray
    scale: np.ndarray
    loading: np.ndarray
    score_scale: np.ndarray
    members: list[dict]
    diagnostics: dict
    model_id: str = "eqesn_rollout"

    def _encode(self, y):
        return ((y-self.center)/self.scale)@self.loading/self.score_scale

    def _decode(self, scores):
        return (scores*self.score_scale)@self.loading.T*self.scale+self.center

    def forecast(self, horizon, target_coordinates=None, target_site_ids=None,
                 n_paths=256, seed=0, history=None):
        horizon = _integer(horizon, "horizon")
        n_paths = _integer(n_paths, "n_paths", 0)
        data = validate_history(self.training, history)
        coordinates, ids, indices = _observed_targets(self.training, target_coordinates, target_site_ids)
        scores = self._encode(data.y)
        config = self.config
        trajectories = np.empty((len(self.members), horizon, len(indices)))
        for m, member in enumerate(self.members):
            hidden = np.zeros(config["width"])
            # Transition h[t] uses information only through row t-1.
            for t in range(len(scores)):
                hidden = reservoir_step(hidden, _lagged(scores, t, config["lags"]),
                    member["recurrent"], member["input_weights"], member["bias"], config["leak"])
            path = np.empty((len(scores)+horizon, scores.shape[1]))
            path[:len(scores)] = scores
            for step in range(horizon):
                t = len(scores)+step
                hidden = reservoir_step(hidden, _lagged(path, t, config["lags"]),
                    member["recurrent"], member["input_weights"], member["bias"], config["leak"])
                feature = (_features(hidden, config["quadratic"])-member["feature_center"])/member["feature_scale"]
                path[t] = feature@member["readout"]
                trajectories[m, step] = self._decode(path[t])[indices]
        return Forecast(trajectories.mean(axis=0), trajectories, coordinates, ids, {
            "model_id": self.model_id, "origin_index": len(data.y),
            "uncertainty": "finite_reservoir_ensemble", "actual_member_count": len(self.members),
            "requested_n_paths": n_paths, "finite_distribution_enumerated": True,
            "path_policy": "one fitted member held fixed throughout each recursive path",
            "parameter_fit_rows": len(self.training.y), "conditioning_rows": len(data.y),
            "observation_noise_included": False,
        })

    def to_state(self):
        return {"model_id": self.model_id, "schema_version": 1,
                "training": _panel_state(self.training), "config": self.config,
                "center": self.center, "scale": self.scale, "loading": self.loading,
                "score_scale": self.score_scale, "members": self.members,
                "diagnostics": self.diagnostics}

    @classmethod
    def from_state(cls, state):
        if state.get("model_id") != "eqesn_rollout" or state.get("schema_version") != 1:
            raise ValueError("Unsupported recursive E-QESN state")
        return cls(Panel(**state["training"]), _config(state["config"]),
                   *(np.array(state[key], copy=True) for key in ("center", "scale", "loading", "score_scale")),
                   [{key: np.array(value, copy=True) for key, value in member.items()} for member in state["members"]],
                   dict(state["diagnostics"]))


def fit_reservoir(model_id, data: Panel, config=None, seed=0):
    """Fit a finite recursive E-QESN ensemble using only an observed Panel."""
    if model_id != "eqesn_rollout":
        raise CapabilityError(f"Unknown reservoir comparator {model_id!r}")
    if not isinstance(data, Panel):
        raise TypeError("Reservoir fitting accepts an observation Panel")
    config = _config(config)
    start = max(config["washout"], config["lags"])
    if len(data.y) < start+3:
        raise ValueError("Need at least three training targets after washout and lag initialization")
    center = data.y.mean(axis=0)
    scale = np.maximum(data.y.std(axis=0), 1e-8)
    standardized = (data.y-center)/scale
    _, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    rank = min(config["modes"], len(vt), len(data.y)-1)
    loading = vt[:rank].T
    signs = np.sign(loading[np.argmax(np.abs(loading), axis=0), np.arange(rank)])
    loading *= np.where(signs == 0, 1., signs)
    score_scale = np.maximum(singular[:rank]/np.sqrt(len(data.y)), 1e-8)
    scores = standardized@loading/score_scale
    lag_matrix = np.array([_lagged(scores, t, config["lags"]) for t in range(len(scores))])
    members, losses = [], []
    summed_prediction = np.zeros_like(scores[start:])
    width = config["width"]
    for index in range(config["members"]):
        rng = np.random.default_rng(seed_for(seed, model_id, "member", index))
        recurrent = rng.uniform(-1., 1., (width, width))*(rng.uniform(size=(width, width)) < config["sparsity"])
        radius = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
        if radius < 1e-12:
            recurrent[0, 0] = .5
            radius = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
        recurrent *= config["spectral_radius"]/radius
        input_weights = rng.uniform(-config["input_scale"], config["input_scale"], (width, rank*config["lags"]))
        bias = rng.uniform(-config["input_scale"], config["input_scale"], width)
        hidden = np.zeros(width)
        features = np.empty((len(scores)-start, 1+width*(2 if config["quadratic"] else 1)))
        for t, inputs in enumerate(lag_matrix):
            hidden = reservoir_step(hidden, inputs, recurrent, input_weights, bias, config["leak"])
            if t >= start:
                features[t-start] = _features(hidden, config["quadratic"])
        feature_center = features.mean(axis=0)
        feature_scale = np.maximum(features.std(axis=0), 1e-8)
        feature_center[0], feature_scale[0] = 0., 1.
        features = (features-feature_center)/feature_scale
        readout = _ridge_readout(features, scores[start:], config["ridge"])
        predicted = features@readout
        summed_prediction += predicted
        losses.append(float(np.mean((predicted-scores[start:])**2)))
        members.append(dict(recurrent=recurrent, input_weights=input_weights, bias=bias,
                            feature_center=feature_center, feature_scale=feature_scale, readout=readout))
    diagnostics = {
        "status": "fitted", "qualified": True,
        "qualification": "finite randomized reservoir ensemble, not posterior MCMC",
        "seed": int(seed), "members": config["members"], "effective_modes": rank,
        "training_rows": len(data.y), "readout_target_rows": len(scores)-start,
        "training_member_score_mse": losses,
        "training_ensemble_score_mse": float(np.mean((summed_prediction/config["members"]-scores[start:])**2)),
        "training_loss_scope": "in-sample one-step teacher-forced PCA-score prediction",
        "transform_scope": "supplied training observations only",
        "spectral_radius_note": "a scaled spectral radius below one is not a general echo-state-property proof",
    }
    return QuadraticReservoirEnsemble(data, config, center, scale, loading, score_scale, members, diagnostics)


def reservoir_from_state(state):
    return QuadraticReservoirEnsemble.from_state(state)
