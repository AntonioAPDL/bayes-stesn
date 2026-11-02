"""Causal shallow/deep reservoir features with optional bounded readout terms.

The response model remains linear in its Bayesian readout coefficients. All
data-dependent transforms use the conditioning segment only. For two layers,
the lower layer's conditioning states first determine the fixed center/scale
of its input to the upper layer; the complete sequence is then recomputed.

``states[t]`` and ``Z[t]`` use responses through t-1. Both ``update`` and
``update_batch`` incorporate one new previous observation and return the state
and predictors for the next response. Batches share one fixed feature map;
their histories and response values may differ.
"""
from __future__ import annotations

import json
import numpy as np
from scipy.linalg import svd
from scipy.sparse import csr_matrix

from .io import atomic_npz


def _per_layer(value, length, name):
    if length == 0:
        return ()
    if np.isscalar(value):
        return tuple(float(value) for _ in range(length))
    values = tuple(float(item) for item in value)
    if len(values) != length:
        raise ValueError(f"{name} needs one value per reservoir layer")
    return values


class TemporalFeaturesV3:
    """A fixed causal feature map supporting multiple recurrent layers.

    ``units=()`` gives a linear delayed-score map. Optional squares/products
    use bounded reservoir coordinates. ``score_bound=k`` replaces the raw
    lagged scores in the readout by k*tanh(score/k), while raw scores continue
    to drive the first reservoir. Thus all readout features are bounded when
    score_bound is specified, without changing Gaussian coefficient inference.
    """

    def __init__(
        self, y, conditioning, *, units=(256,), lags=(1,), input_scales=(0.4,),
        leak_patterns=None, modes=16, score_bound=None, squares=0, products=0,
        seed=731, recurrent_norm=0.9, connectivity=0.08, feature_scale_floor=0.04,
    ):
        y = np.asarray(y, dtype=float)
        if y.ndim != 2 or not np.isfinite(y).all():
            raise ValueError("y must be a finite time-by-observed-site matrix")
        self.conditioning = int(conditioning)
        self.units = tuple(int(size) for size in units)
        self.lags = tuple(int(lag) for lag in lags)
        self.modes, self.seed = int(modes), int(seed)
        if (self.conditioning < 3 or self.conditioning > len(y)
                or self.modes < 1 or self.modes >= min(self.conditioning, y.shape[1])):
            raise ValueError("Invalid conditioning length or PCA dimension")
        if (not self.lags or min(self.lags) < 1
                or self.lags != tuple(sorted(set(self.lags)))
                or any(size < 1 for size in self.units)):
            raise ValueError("Use increasing positive lags and positive layer sizes")
        if not 0 < recurrent_norm < 1 or not 0 < connectivity <= 1:
            raise ValueError("Require recurrent_norm in (0,1) and positive connectivity")
        if score_bound is not None and (not np.isfinite(score_bound) or score_bound <= 0):
            raise ValueError("score_bound must be positive or None")
        if not np.isfinite(feature_scale_floor) or feature_scale_floor <= 0:
            raise ValueError("feature_scale_floor must be positive")
        self.input_scales = _per_layer(input_scales, len(self.units), "input_scales")
        if any(scale <= 0 or not np.isfinite(scale) for scale in self.input_scales):
            raise ValueError("All input scales must be positive and finite")
        if leak_patterns is None:
            leak_patterns = [(0.2, 0.5, 0.9)] + [(0.05, 0.2, 0.6)] * (len(self.units) - 1)
            leak_patterns = leak_patterns[:len(self.units)]
        self.leak_patterns = tuple(tuple(float(x) for x in pattern) for pattern in leak_patterns)
        if len(self.leak_patterns) != len(self.units):
            raise ValueError("One leak pattern is required per layer")
        if any(not pattern or min(pattern) <= 0 or max(pattern) > 1 for pattern in self.leak_patterns):
            raise ValueError("Leak rates must lie in (0,1]")
        self.score_bound = None if score_bound is None else float(score_bound)
        self.recurrent_norm = float(recurrent_norm)
        self.connectivity = float(connectivity)
        self.feature_scale_floor = float(feature_scale_floor)
        self.squares, self.products = int(squares), int(products)
        self.maxlag = max(self.lags)
        self.history_size = self.maxlag * self.modes
        self.input_dimension = len(self.lags) * self.modes
        self.total_units = sum(self.units)
        self.state_size = self.history_size + self.total_units
        self.layer_offsets = self.history_size + np.r_[0, np.cumsum(self.units)].astype(int)
        self._lag_index = np.asarray(self.lags) - 1
        if not 0 <= self.squares <= self.total_units:
            raise ValueError("squares must not exceed the total reservoir size")
        if not 0 <= self.products <= self.total_units * (self.total_units - 1) // 2:
            raise ValueError("Invalid number of distinct reservoir products")

        initial = y[:self.conditioning]
        self.site_center = initial.mean(axis=0)
        _, singular, vt = svd(initial - self.site_center, full_matrices=False)
        if singular[self.modes - 1] <= 1e-12 * singular[0]:
            raise ValueError("Selected PCA modes have numerically zero variation")
        self.P = vt[:self.modes] / (singular[:self.modes, None] / np.sqrt(self.conditioning - 1))

        rng = np.random.default_rng(self.seed)
        self.W, self.U, self.bias, self.leaks = [], [], [], []
        incoming = self.input_dimension
        for size, scale, pattern in zip(self.units, self.input_scales, self.leak_patterns):
            recurrent = rng.normal(size=(size, size)) * (rng.uniform(size=(size, size)) < self.connectivity)
            norm = np.linalg.norm(recurrent, 2)
            # A tiny sparse reservoir can legitimately draw no edges. Keep
            # that zero recurrent matrix, preserving its RNG stream, instead
            # of dividing by zero. Nonzero draws reproduce V3 exactly.
            if norm > 0:
                recurrent *= self.recurrent_norm / norm
            self.W.append(recurrent)
            self.U.append(rng.normal(size=(size, incoming)) * scale / np.sqrt(incoming))
            self.bias.append(rng.uniform(-0.3, 0.3, size))
            self.leaks.append(np.resize(np.asarray(pattern), size))
            incoming = size
        self._sparse_W = [csr_matrix(weight) for weight in self.W]

        # Select nonlinear readout coordinates using a separate fixed stream;
        # adding readout terms never changes reservoir weights.
        extras = np.random.default_rng(self.seed + 991)
        self.square_index = np.sort(extras.choice(self.total_units, self.squares, replace=False))
        if self.products:
            i, j = np.triu_indices(self.total_units, k=1)
            chosen = extras.choice(len(i), self.products, replace=False)
            self.product_i, self.product_j = i[chosen], j[chosen]
        else:
            self.product_i = self.product_j = np.empty(0, dtype=int)

        # Conditioning-only interlayer calibration. Each higher layer receives
        # the current lower-layer state, already determined by lagged responses.
        self.layer_input_centers, self.layer_input_scales = [], []
        drivers = np.zeros((self.conditioning, self.input_dimension))
        initial_scores = (initial - self.site_center) @ self.P.T
        for lag_number, lag in enumerate(self.lags):
            drivers[lag:, lag_number * self.modes:(lag_number + 1) * self.modes] = initial_scores[:-lag]
        calibration = slice(self.conditioning // 3, self.conditioning)
        for layer, size in enumerate(self.units):
            if layer == 0:
                location = np.zeros(drivers.shape[1]); spread = np.ones(drivers.shape[1])
            else:
                location = drivers[calibration].mean(axis=0)
                spread = np.maximum(drivers[calibration].std(axis=0), self.feature_scale_floor)
            self.layer_input_centers.append(location)
            self.layer_input_scales.append(spread)
            normalized = (drivers - location) / spread
            layer_states = np.zeros((self.conditioning, size))
            state = np.zeros(size)
            for t in range(1, self.conditioning):
                state = ((1 - self.leaks[layer]) * state + self.leaks[layer]
                         * np.tanh(self._sparse_W[layer] @ state
                                   + self.U[layer] @ normalized[t] + self.bias[layer]))
                layer_states[t] = state
            drivers = layer_states

        raw_dimension = self.input_dimension + self.total_units + self.squares + self.products
        raw = np.empty((len(y), raw_dimension))
        self.states = np.zeros((len(y), self.state_size))
        state = np.zeros(self.state_size)
        raw[0] = self._raw(state)
        for t in range(1, len(y)):
            state = self._state_update(y[t - 1], state)
            self.states[t] = state
            raw[t] = self._raw(state)
        self.center = raw[calibration].mean(axis=0)
        self.scale = np.maximum(raw[calibration].std(axis=0), self.feature_scale_floor)
        self.norm = np.sqrt(raw_dimension)
        self.Z = np.column_stack([np.ones(len(y)), (raw - self.center) / self.scale / self.norm])
        self.config = dict(
            units=list(self.units), lags=list(self.lags), input_scales=list(self.input_scales),
            leak_patterns=[list(pattern) for pattern in self.leak_patterns], modes=self.modes,
            score_bound=self.score_bound, squares=self.squares, products=self.products,
            seed=self.seed, recurrent_norm=self.recurrent_norm, connectivity=self.connectivity,
            feature_scale_floor=self.feature_scale_floor,
        )

    def _state_update(self, previous_y, previous_state):
        history = np.asarray(previous_state[:self.history_size]).reshape(self.maxlag, self.modes)
        history = np.vstack([(np.asarray(previous_y) - self.site_center) @ self.P.T, history[:-1]])
        result = np.empty(self.state_size)
        result[:self.history_size] = history.ravel()
        incoming = history[self._lag_index].ravel()
        for layer in range(len(self.units)):
            start, end = self.layer_offsets[layer:layer + 2]
            old = previous_state[start:end]
            normalized = (incoming - self.layer_input_centers[layer]) / self.layer_input_scales[layer]
            incoming = ((1 - self.leaks[layer]) * old + self.leaks[layer]
                        * np.tanh(self._sparse_W[layer] @ old
                                  + self.U[layer] @ normalized + self.bias[layer]))
            result[start:end] = incoming
        return result

    def _raw(self, state):
        history = state[:self.history_size].reshape(self.maxlag, self.modes)
        scores = history[self._lag_index].ravel()
        if self.score_bound is not None:
            scores = self.score_bound * np.tanh(scores / self.score_bound)
        reservoir = state[self.history_size:]
        return np.concatenate([
            scores, reservoir,
            reservoir[self.square_index] ** 2,
            reservoir[self.product_i] * reservoir[self.product_j],
        ])

    def update(self, previous_y, previous_state):
        """Advance one history and return (new_state, next_response_features)."""
        state = self._state_update(np.asarray(previous_y), np.asarray(previous_state))
        return state, np.r_[1.0, (self._raw(state) - self.center) / self.scale / self.norm]

    def update_batch(self, previous_y, previous_states):
        """Advance independent histories with a shared fixed feature map.

        Inputs have shapes (batch, n_sites) and (batch, state_size). Sparse
        recurrent multiplication avoids doing dense work on the 92% zeros.
        """
        previous_y = np.asarray(previous_y)
        previous_states = np.asarray(previous_states)
        if (previous_y.ndim != 2 or previous_states.ndim != 2
                or len(previous_y) != len(previous_states)
                or previous_y.shape[1] != len(self.site_center)
                or previous_states.shape[1] != self.state_size):
            raise ValueError("Invalid batch response/state shapes")
        batch = len(previous_y)
        old_history = previous_states[:, :self.history_size].reshape(batch, self.maxlag, self.modes)
        history = np.concatenate([
            ((previous_y - self.site_center) @ self.P.T)[:, None, :], old_history[:, :-1]
        ], axis=1)
        states = np.empty_like(previous_states, dtype=float)
        states[:, :self.history_size] = history.reshape(batch, self.history_size)
        incoming = history[:, self._lag_index, :].reshape(batch, self.input_dimension)
        for layer in range(len(self.units)):
            start, end = self.layer_offsets[layer:layer + 2]
            old = previous_states[:, start:end]
            normalized = (incoming - self.layer_input_centers[layer]) / self.layer_input_scales[layer]
            incoming = ((1 - self.leaks[layer]) * old + self.leaks[layer]
                        * np.tanh((self._sparse_W[layer] @ old.T).T
                                  + normalized @ self.U[layer].T + self.bias[layer]))
            states[:, start:end] = incoming
        scores = history[:, self._lag_index, :].reshape(batch, self.input_dimension)
        if self.score_bound is not None:
            scores = self.score_bound * np.tanh(scores / self.score_bound)
        reservoir = states[:, self.history_size:]
        raw = np.concatenate([
            scores, reservoir, reservoir[:, self.square_index] ** 2,
            reservoir[:, self.product_i] * reservoir[:, self.product_j],
        ], axis=1)
        z = np.column_stack([np.ones(batch), (raw - self.center) / self.scale / self.norm])
        return states, z

    def save(self, path, *, include_history=False):
        """Save the fitted feature map; full history is optional to avoid duplication."""
        arrays = dict(
            P=self.P, site_center=self.site_center, center=self.center, scale=self.scale,
            norm=np.array(self.norm), last_state=self.states[-1], last_z=self.Z[-1],
            history_length=np.array(len(self.states)),
            conditioning=np.array(self.conditioning), layer_offsets=self.layer_offsets,
            square_index=self.square_index, product_i=self.product_i, product_j=self.product_j,
            configuration=np.array(json.dumps(self.config, sort_keys=True)),
        )
        if include_history:
            arrays.update(Z=self.Z, states=self.states)
        for layer in range(len(self.units)):
            arrays.update({
                f"W_{layer}": self.W[layer], f"U_{layer}": self.U[layer],
                f"bias_{layer}": self.bias[layer], f"leaks_{layer}": self.leaks[layer],
                f"layer_input_center_{layer}": self.layer_input_centers[layer],
                f"layer_input_scale_{layer}": self.layer_input_scales[layer],
            })
        atomic_npz(path, **arrays)
