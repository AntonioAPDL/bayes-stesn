"""Conditioning-only encoders and causal, frozen reservoir feature maps.

The PCA route preserves the V3 reservoir equations and random streams. Spatial
pooling, graph diffusion, and a small nonlinear autoencoder are optional input
encoders. No encoder or normalization is fitted on a forecast/evaluation block.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
import numpy as np
from scipy.linalg import svd
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from ._reservoir import TemporalFeaturesV3
from .raw_reservoir import RawFieldTemporalFeatures
from .io import content_hash


@dataclass(frozen=True)
class FeatureSpec:
    kind: str = "pca_esn"
    modes: int = 8
    units: tuple[int, ...] = (48, 48)
    lags: tuple[int, ...] = (1,)
    input_scales: tuple[float, ...] | float | None = None
    leak_patterns: tuple[tuple[float, ...], ...] | None = None
    score_bound: float | None = 3.0
    squares: int = 24
    products: int = 0
    recurrent_norm: float = 0.9
    connectivity: float = 0.08
    feature_scale_floor: float = 0.04
    seed: int = 731
    graph_neighbors: int = 6
    local_width: float = 0.25
    encoder_width: int = 16
    encoder_epochs: int = 120
    encoder_learning_rate: float = 0.01
    encoder_penalty: float = 1e-4
    max_feature_bytes: int = 512*1024**2
    # Raw-field options do not alter the original PCA route or its RNG stream.
    recurrent_scaling: str = "operator_norm"
    spectral_radius: float | None = None
    raw_readout_lags: str = "all"

    def __post_init__(self):
        kinds = {"pca_esn", "raw_esn", "local_features_esn", "graph_features_esn", "frozen_encoder_esn"}
        if self.kind not in kinds:
            raise ValueError(f"Unknown feature kind {self.kind}; expected one of {sorted(kinds)}")
        for name in ("units", "lags"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        scales = ((.4,)+(.8,)*(len(self.units)-1))[:len(self.units)] if self.input_scales is None else self.input_scales
        scales = (float(scales),)*len(self.units) if np.isscalar(scales) else tuple(scales)
        object.__setattr__(self, "input_scales", scales)
        if self.leak_patterns is not None:
            object.__setattr__(self, "leak_patterns", tuple(tuple(x) for x in self.leak_patterns))
        if self.modes < 1 or self.seed < 0 or self.graph_neighbors < 1 or self.local_width <= 0:
            raise ValueError("Invalid feature dimensions, seed or spatial encoder parameters")
        if self.kind == "frozen_encoder_esn":
            if self.encoder_width <= self.modes or self.encoder_epochs < 1 or self.encoder_learning_rate <= 0:
                raise ValueError("Encoder width must exceed modes, with positive epochs and learning rate")
            if self.encoder_penalty < 0:
                raise ValueError("Encoder penalty must be nonnegative")
        if self.max_feature_bytes < 1:
            raise ValueError("max_feature_bytes must be positive")
        if self.recurrent_scaling not in {"operator_norm", "spectral_radius"}:
            raise ValueError("recurrent_scaling must be operator_norm or spectral_radius")
        if self.raw_readout_lags not in {"all", "latest", "none"}:
            raise ValueError("raw_readout_lags must be all, latest, or none")
        if self.kind == "raw_esn":
            if len(self.units) != 1:
                raise ValueError("raw_esn currently requires exactly one reservoir layer")
            if self.raw_readout_lags == "latest" and 1 not in self.lags:
                raise ValueError("The latest direct readout requires lag 1")
            if self.recurrent_scaling == "spectral_radius" and (
                    self.spectral_radius is None or not np.isfinite(self.spectral_radius)
                    or self.spectral_radius <= 0):
                raise ValueError("spectral_radius must be positive and finite")
        elif (self.recurrent_scaling != "operator_norm" or self.spectral_radius is not None
              or self.raw_readout_lags != "all"):
            raise ValueError("Spectral-radius scaling and direct-readout controls require kind='raw_esn'")

    @classmethod
    def v3(cls):
        return cls(modes=16, units=(256, 256), squares=128, input_scales=(0.4, 0.8),
                   leak_patterns=((0.2, 0.5, 0.9), (0.05, 0.2, 0.6)))


class FrozenEncoder:
    def __init__(self, kind, parameters):
        self.kind, self.parameters = kind, parameters

    @classmethod
    def fit(cls, y, coordinates, spec):
        if spec.kind in {"pca_esn", "raw_esn"}:
            return cls(spec.kind, {})
        if spec.kind == "graph_features_esn":
            n = len(coordinates)
            if n < 2:
                raise ValueError("Graph features need at least two sites")
            distance = cdist(coordinates, coordinates)
            np.fill_diagonal(distance, np.inf)
            k = min(spec.graph_neighbors, n - 1)
            neighbors = np.argsort(distance, axis=1, kind="stable")[:, :k]
            weights = np.zeros((n, n))
            scale = max(float(np.median(np.take_along_axis(distance, neighbors, axis=1))), 1e-12)
            weights[np.arange(n)[:, None], neighbors] = np.exp(-np.take_along_axis(distance, neighbors, axis=1)**2 / (2*scale**2))
            weights = np.maximum(weights, weights.T)
            weights /= weights.sum(axis=1, keepdims=True)
            return cls(spec.kind, {"operator": weights})
        if spec.kind == "local_features_esn":
            n = len(coordinates)
            count = min(max(4, spec.modes), n)
            anchors = [int(np.argmin(np.linalg.norm(coordinates - coordinates.mean(axis=0), axis=1)))]
            distances = cdist(coordinates, coordinates)
            nearest = distances[:, anchors[0]]
            for _ in range(1, count):
                scores = nearest.copy()
                scores[anchors] = -np.inf
                anchors.append(int(np.argmax(scores)))
                nearest = np.minimum(nearest, distances[:, anchors[-1]])
            weights = np.exp(-0.5*(distances[np.asarray(anchors)]/spec.local_width)**2)
            weights /= weights.sum(axis=1, keepdims=True)
            return cls(spec.kind, {"operator": weights, "anchors": np.asarray(anchors)})
        # A deliberately small, transparent nonlinear encoder. Adam trains a
        # tanh bottleneck + linear decoder exclusively on the conditioning set.
        center, scale = y.mean(axis=0), np.maximum(y.std(axis=0), 0.05)
        x = (y-center)/scale
        rng = np.random.default_rng(spec.seed + 8003)
        n, width = x.shape[1], spec.encoder_width
        _, _, vt = svd(x, full_matrices=False)
        W = rng.normal(0, .02, (n, width))
        columns = min(width, len(vt))
        W[:, :columns] += 0.3*vt[:columns].T
        b = np.zeros(width)
        D = 0.3*W.T.copy()
        a = np.zeros(n)
        parameters = [W, b, D, a]
        moment = [np.zeros_like(v) for v in parameters]
        variance = [np.zeros_like(v) for v in parameters]
        losses = []
        for epoch in range(1, spec.encoder_epochs+1):
            h = np.tanh(x@W+b)
            residual = h@D+a-x
            loss = float(np.mean(residual**2) + spec.encoder_penalty*(np.sum(W**2)+np.sum(D**2)))
            losses.append(loss)
            dr = 2*residual/residual.size
            dh = (dr@D.T)*(1-h*h)
            gradients = [x.T@dh+2*spec.encoder_penalty*W, dh.sum(axis=0),
                         h.T@dr+2*spec.encoder_penalty*D, dr.sum(axis=0)]
            for j, (parameter, gradient) in enumerate(zip(parameters, gradients)):
                moment[j] = .9*moment[j]+.1*gradient
                variance[j] = .999*variance[j]+.001*gradient**2
                parameter -= spec.encoder_learning_rate*(moment[j]/(1-.9**epoch))/(np.sqrt(variance[j]/(1-.999**epoch))+1e-8)
        return cls(spec.kind, {"center": center, "scale": scale, "W": W, "b": b,
                               "decoder": D, "decoder_bias": a, "training_loss": np.asarray(losses)})

    def transform(self, y):
        y = np.asarray(y, dtype=float)
        if self.kind in {"pca_esn", "raw_esn"}:
            return y
        p = self.parameters
        if self.kind == "graph_features_esn":
            smooth = y@p["operator"].T
            return np.concatenate([y, smooth, smooth@p["operator"].T], axis=-1)
        if self.kind == "local_features_esn":
            return np.concatenate([y, y@p["operator"].T], axis=-1)
        return np.tanh(((y-p["center"])/p["scale"])@p["W"]+p["b"])


class FrozenFeatures:
    """Fitted feature transformations; callers must not mutate fitted arrays."""
    def __init__(self, core, encoder, spec, correction=None):
        self.core, self.encoder, self.spec = core, encoder, spec
        self.correction = correction

    @classmethod
    def fit(cls, y, conditioning, coordinates, spec, *, correction_base=None):
        if not spec.lags or min(spec.lags)<1:
            raise ValueError("Positive feature lags are required")
        if spec.kind == "raw_esn":
            values = np.asarray(y, dtype=float)
            if values.ndim != 2:
                raise ValueError("Raw-field features need a time-by-site matrix")
            # Canonicalize an inactive PCA dimension to the actual input count.
            spec = replace(spec, modes=values.shape[1])
            encoder = FrozenEncoder.fit(values[:conditioning], coordinates, spec)
            names = ("units", "lags", "input_scales", "leak_patterns", "score_bound",
                     "squares", "products", "seed", "recurrent_norm", "connectivity",
                     "feature_scale_floor", "recurrent_scaling", "spectral_radius",
                     "raw_readout_lags", "max_feature_bytes")
            core = RawFieldTemporalFeatures(values, conditioning,
                **{k: getattr(spec, k) for k in names})
        else:
            state_size=max(spec.lags)*spec.modes+sum(spec.units)
            dimension=1+len(spec.lags)*spec.modes+sum(spec.units)+spec.squares+spec.products
            estimate=8*(len(y)*(state_size+dimension)+sum(size*size for size in spec.units))
            if estimate>spec.max_feature_bytes:
                raise MemoryError(f"Feature history/weights need at least {estimate/1024**2:.1f} MiB; increase max_feature_bytes deliberately or reduce dimensions")
            encoder = FrozenEncoder.fit(np.asarray(y)[:conditioning], coordinates, spec)
            encoded = encoder.transform(y)
            names = ("modes", "units", "lags", "input_scales", "leak_patterns", "score_bound",
                     "squares", "products", "seed", "recurrent_norm", "connectivity", "feature_scale_floor")
            core = TemporalFeaturesV3(encoded, conditioning, **{k: getattr(spec, k) for k in names})
        correction = None
        if correction_base is not None:
            d0 = int(correction_base)
            if not 1 <= d0 < core.Z.shape[1]:
                raise ValueError("Correction prior base_columns must be between 1 and d-1")
            reference = core.Z[conditioning//3:conditioning]
            A = np.linalg.lstsq(reference[:, :d0], reference[:, d0:], rcond=1e-10)[0]
            correction = {"base_columns": d0, "A": A}
        return cls(core, encoder, spec, correction)

    def transform_z(self, z):
        if self.correction is None:
            return z
        d0, A = self.correction["base_columns"], self.correction["A"]
        z = np.array(z, copy=True)
        z[..., d0:] -= z[..., :d0]@A
        return z

    @property
    def Z(self):
        return self.transform_z(self.core.Z)

    @property
    def states(self):
        return self.core.states

    @property
    def groups(self):
        inputs = getattr(self.core, "readout_input_dimension", self.core.input_dimension)
        groups = ["intercept"] + ["score"]*inputs
        for j, size in enumerate(self.spec.units):
            groups.extend([f"reservoir_{j+1}"]*size)
        groups.extend(["squares"]*self.spec.squares)
        groups.extend(["products"]*self.spec.products)
        if self.correction is not None:
            d0 = self.correction["base_columns"]
            groups = ["base"]*d0 + ["correction"]*(len(groups)-d0)
        return tuple(groups)

    def update(self, previous_y, previous_state):
        state, z = self.core.update(self.encoder.transform(previous_y), previous_state)
        return state, self.transform_z(z)

    def update_batch(self, previous_y, previous_states):
        state, z = self.core.update_batch(self.encoder.transform(previous_y), previous_states)
        return state, self.transform_z(z)

    def history(self, y):
        """Causal replay under already fitted transforms; learns nothing."""
        y = np.asarray(y, dtype=float)
        states = np.zeros((len(y), self.core.state_size))
        raw0 = self.core._raw(states[0])
        z0 = np.r_[1., (raw0-self.core.center)/self.core.scale/self.core.norm]
        Z = np.empty((len(y), len(z0)))
        Z[0] = self.transform_z(z0)
        for t in range(1, len(y)):
            states[t], Z[t] = self.update(y[t-1], states[t-1])
        return states, Z

    def to_state(self, include_history=True):
        core = {k: v for k, v in self.core.__dict__.items() if k != "_sparse_W"}
        if not include_history:
            core.pop("Z", None)
            core.pop("states", None)
        return {"spec": asdict(self.spec), "core": core,
                "encoder": {"kind": self.encoder.kind, "parameters": self.encoder.parameters},
                "correction": self.correction}

    @classmethod
    def from_state(cls, state):
        kind = state["spec"].get("kind", "pca_esn")
        core_cls = RawFieldTemporalFeatures if kind == "raw_esn" else TemporalFeaturesV3
        core = core_cls.__new__(core_cls)
        core.__dict__.update(state["core"])
        core._sparse_W = [csr_matrix(w) for w in core.W]
        return cls(core, FrozenEncoder(**state["encoder"]), FeatureSpec(**state["spec"]), state["correction"])

    @property
    def fingerprint(self):
        return content_hash(self.to_state(include_history=False))

