"""Causal ESN features from every permitted input coordinate, without PCA.

This is a separate feature-map version.  It does not change the V3 PCA map or
its random streams.  Input normalization is fitted on the conditioning prefix;
every supplied site and configured lag drives the reservoir.  Direct readout
lags are an independent, explicit dimension-control choice.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigvals, svdvals
from scipy.sparse import csr_matrix

from ._reservoir import _per_layer


def _integer(value, name, minimum=1):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def raw_feature_allocation(n_times, n_sites, *, units, lags, squares=0,
                           products=0, raw_readout_lags="all"):
    """Conservative float64 feature-build allocation estimate, not peak RSS.

    Accounts for stored states/design, dense input and recurrent matrices,
    sparse recurrent storage, and a bounded allowance for dense eigen/SVD work.
    The sampler's Gram matrices, model copies and posterior diagnostics are
    outside this estimate and must be admitted separately by the runner.
    """
    t, n = _integer(n_times, "n_times"), _integer(n_sites, "n_sites")
    sizes = tuple(_integer(x, "units") for x in units)
    if len(sizes) != 1:
        raise ValueError("raw_esn currently requires exactly one reservoir layer")
    delays = tuple(_integer(x, "lags") for x in lags)
    if not delays or delays != tuple(sorted(set(delays))):
        raise ValueError("Use increasing distinct positive lags")
    if raw_readout_lags not in {"all", "latest", "none"}:
        raise ValueError("raw_readout_lags must be all, latest, or none")
    if raw_readout_lags == "latest" and 1 not in delays:
        raise ValueError("The latest direct readout requires lag 1")
    size = sizes[0]
    squares, products = _integer(squares, "squares", 0), _integer(products, "products", 0)
    if squares > size or products > size * (size - 1) // 2:
        raise ValueError("Nonlinear feature counts exceed available reservoir coordinates")
    incoming = n * len(delays)
    readout = incoming if raw_readout_lags == "all" else n if raw_readout_lags == "latest" else 0
    state = n * max(delays) + size
    dimension = 1 + readout + size + squares + products
    components = {
        "states_bytes": 8 * t * state,
        "temporal_design_bytes": 8 * t * dimension,
        "dense_input_weights_bytes": 8 * size * incoming,
        "dense_recurrent_weights_bytes": 8 * size * size,
        # This upper bound intentionally does not assume a realized edge count.
        "sparse_recurrent_upper_bytes": 12 * size * size + 4 * (size + 1),
        "feature_scalers_and_vectors_bytes": 8 * (4 * n + 4 * incoming + 6 * size + 3 * dimension),
        "normalization_work_bytes": 8 * t * n,
        "eigen_svd_workspace_allowance_bytes": 64 * size * size,
    }
    return {
        "n_times": t, "n_sites": n, "reservoir_units": size,
        "input_dimension": incoming, "readout_input_dimension": readout,
        "state_size": state, "temporal_columns": dimension,
        **components, "estimated_peak_feature_bytes": sum(components.values()),
        "measured_peak_rss": False,
        "scope": "feature construction only; excludes inference/design copies, posterior and forecasts",
    }


def scale_recurrent_matrix(matrix, *, recurrent_scaling, recurrent_norm,
                           spectral_radius):
    """Scale a finite square matrix and retain independently defined diagnostics.

    Dense LAPACK eigenvalues determine the spectral radius; singular values
    determine the operator norm.  The two are never substituted.  A zero or
    near-zero requested normalization is an explicit candidate failure, not a
    request to redraw weights or silently add edges.
    """
    value = np.array(matrix, dtype=np.float64, copy=True)
    if (value.ndim != 2 or value.shape[0] != value.shape[1] or len(value) == 0
            or not np.isfinite(value).all()):
        raise ValueError("Recurrent matrix must be finite, nonempty and square")
    if recurrent_scaling not in {"operator_norm", "spectral_radius"}:
        raise ValueError("recurrent_scaling must be operator_norm or spectral_radius")
    target = spectral_radius if recurrent_scaling == "spectral_radius" else recurrent_norm
    if target is None or not np.isfinite(target) or target <= 0:
        raise ValueError("The requested recurrent scale must be positive and finite")
    raw_radius = float(np.max(np.abs(eigvals(value, check_finite=False))))
    raw_norm = float(svdvals(value, check_finite=False)[0])
    denominator = raw_radius if recurrent_scaling == "spectral_radius" else raw_norm
    threshold = 1e-10 * max(1.0, raw_norm)
    if not np.isfinite(denominator) or denominator <= threshold:
        raise ValueError(
            f"Degenerate recurrent {recurrent_scaling}: {denominator:.6g}; "
            "cannot realize the requested scale without changing the seeded matrix")
    factor = float(target) / denominator
    value *= factor
    radius, norm = raw_radius * factor, raw_norm * factor
    achieved = radius if recurrent_scaling == "spectral_radius" else norm
    return value, {
        "recurrent_scaling": recurrent_scaling,
        "requested_spectral_radius": float(target) if recurrent_scaling == "spectral_radius" else None,
        "configured_operator_norm": float(target) if recurrent_scaling == "operator_norm" else None,
        "raw_spectral_radius": raw_radius, "raw_operator_norm": raw_norm,
        "spectral_radius": radius, "operator_norm": norm,
        "scale_multiplier": factor,
        "spectral_method": "dense_eigvals", "operator_norm_method": "dense_svdvals",
        "achieved_target": bool(np.isclose(achieved, target, rtol=1e-12, atol=1e-14)),
    }


class RawFieldTemporalFeatures:
    """Single-layer fixed reservoir retaining all supplied input coordinates.

    ``states[t]`` and ``Z[t]`` condition on responses through ``t-1``.
    ``raw_readout_lags`` changes only the direct readout: all configured lags
    still enter the reservoir.  No response-dependent projection is estimated.
    """

    def __init__(self, y, conditioning, *, units=(256,), lags=(1,),
                 input_scales=(.4,), leak_patterns=None, score_bound=None,
                 squares=0, products=0, seed=731, recurrent_norm=.9,
                 recurrent_scaling="spectral_radius", spectral_radius=.9,
                 raw_readout_lags="all", connectivity=.08,
                 feature_scale_floor=.04, max_feature_bytes=512 * 1024**2):
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 2 or not y.shape[1] or not np.isfinite(y).all():
            raise ValueError("y must be a finite nonempty time-by-input-site matrix")
        self.conditioning = _integer(conditioning, "conditioning", 3)
        self.units = tuple(_integer(x, "units") for x in units)
        self.lags = tuple(_integer(x, "lags") for x in lags)
        self.squares, self.products = _integer(squares, "squares", 0), _integer(products, "products", 0)
        self.allocation = raw_feature_allocation(
            len(y), y.shape[1], units=self.units, lags=self.lags,
            squares=self.squares, products=self.products, raw_readout_lags=raw_readout_lags)
        limit = _integer(max_feature_bytes, "max_feature_bytes")
        if self.allocation["estimated_peak_feature_bytes"] > limit:
            raise MemoryError(
                f"Raw-field features need an estimated {self.allocation['estimated_peak_feature_bytes']} "
                f"bytes; declared max_feature_bytes is {limit}")
        if not max(self.lags) < self.conditioning <= len(y):
            raise ValueError("Conditioning must exceed all lags and lie inside the supplied data")
        self.seed = _integer(seed, "seed", 0)
        if not np.isfinite(connectivity) or not 0 < connectivity <= 1:
            raise ValueError("Connectivity must lie in (0,1]")
        if not np.isfinite(feature_scale_floor) or feature_scale_floor <= 0:
            raise ValueError("feature_scale_floor must be positive and finite")
        if score_bound is not None and (not np.isfinite(score_bound) or score_bound <= 0):
            raise ValueError("score_bound must be positive or None")
        self.connectivity, self.feature_scale_floor = float(connectivity), float(feature_scale_floor)
        self.score_bound = None if score_bound is None else float(score_bound)
        self.recurrent_scaling, self.raw_readout_lags = recurrent_scaling, raw_readout_lags
        self.recurrent_norm = float(recurrent_norm)
        self.spectral_radius = None if spectral_radius is None else float(spectral_radius)
        self.input_scales = _per_layer(input_scales, 1, "input_scales")
        if any(not np.isfinite(x) or x <= 0 for x in self.input_scales):
            raise ValueError("Input scale must be positive and finite")
        leak_patterns = ((.2, .5, .9),) if leak_patterns is None else leak_patterns
        self.leak_patterns = tuple(tuple(float(x) for x in pattern) for pattern in leak_patterns)
        if (len(self.leak_patterns) != 1 or not self.leak_patterns[0]
                or any(not np.isfinite(x) or not 0 < x <= 1 for x in self.leak_patterns[0])):
            raise ValueError("One nonempty leak pattern in (0,1] is required")

        self.n_sites = y.shape[1]
        # ``modes`` is retained only as a legacy state-layout attribute.
        self.modes = self.n_sites
        self.maxlag = max(self.lags)
        self.total_units = self.units[0]
        self.history_size = self.maxlag * self.n_sites
        self.input_dimension = self.n_sites * len(self.lags)
        self.readout_input_dimension = self.allocation["readout_input_dimension"]
        self.state_size = self.history_size + self.total_units
        self.layer_offsets = np.asarray([self.history_size, self.state_size], dtype=int)
        self._lag_index = np.asarray(self.lags, dtype=int) - 1
        self._readout_lag_index = (self._lag_index if raw_readout_lags == "all" else
                                   np.asarray([0], dtype=int) if raw_readout_lags == "latest" else
                                   np.empty(0, dtype=int))

        initial = y[:self.conditioning]
        self.site_center = initial.mean(axis=0)
        self.site_scale = np.maximum(initial.std(axis=0), self.feature_scale_floor)
        rng = np.random.default_rng(self.seed)
        recurrent = rng.normal(size=(self.total_units, self.total_units))
        recurrent *= rng.uniform(size=recurrent.shape) < self.connectivity
        recurrent, diagnostic = scale_recurrent_matrix(
            recurrent, recurrent_scaling=self.recurrent_scaling,
            recurrent_norm=self.recurrent_norm, spectral_radius=self.spectral_radius)
        self.W = [recurrent]
        self.U = [rng.normal(size=(self.total_units, self.input_dimension))
                  * self.input_scales[0] / np.sqrt(self.input_dimension)]
        self.bias = [rng.uniform(-.3, .3, self.total_units)]
        self.leaks = [np.resize(np.asarray(self.leak_patterns[0]), self.total_units)]
        self._sparse_W = [csr_matrix(recurrent)]
        self.layer_input_centers = [np.zeros(self.input_dimension)]
        self.layer_input_scales = [np.ones(self.input_dimension)]
        diagnostic.update(
            minimum_leak=float(np.min(self.leaks[0])),
            maximum_leak=float(np.max(self.leaks[0])),
            sufficient_contraction_bound=float(1 - np.min(self.leaks[0])
                + np.max(self.leaks[0]) * diagnostic["operator_norm"]),
            contraction_bound_method="1-min(leak)+max(leak)*operator_norm")
        self.recurrent_diagnostics = [diagnostic]

        extras = np.random.default_rng(self.seed + 991)
        self.square_index = np.sort(extras.choice(self.total_units, self.squares, replace=False))
        if self.products:
            i, j = np.triu_indices(self.total_units, k=1)
            chosen = extras.choice(len(i), self.products, replace=False)
            self.product_i, self.product_j = i[chosen], j[chosen]
        else:
            self.product_i = self.product_j = np.empty(0, dtype=int)

        dimension = self.allocation["temporal_columns"]
        self.states = np.zeros((len(y), self.state_size), dtype=np.float64)
        self.Z = np.empty((len(y), dimension), dtype=np.float64)
        self.Z[:, 0] = 1.0
        raw = self.Z[:, 1:]  # Reuse the final design allocation for raw features.
        raw[0] = self._raw(self.states[0])
        for t in range(1, len(y)):
            self.states[t] = self._state_update(y[t - 1], self.states[t - 1])
            raw[t] = self._raw(self.states[t])
        calibration = slice(self.conditioning // 3, self.conditioning)
        self.center = raw[calibration].mean(axis=0)
        self.scale = np.maximum(raw[calibration].std(axis=0), self.feature_scale_floor)
        self.norm = float(np.sqrt(dimension - 1))
        raw -= self.center
        raw /= self.scale
        raw /= self.norm
        self.config = {
            "version": "raw-field-reservoir/1", "input_sites": self.n_sites,
            "input_scaling": "conditioning_site_center_and_population_sd",
            "units": list(self.units), "lags": list(self.lags),
            "input_scales": list(self.input_scales),
            "leak_patterns": [list(p) for p in self.leak_patterns],
            "score_bound": self.score_bound, "squares": self.squares,
            "products": self.products, "seed": self.seed,
            "recurrent_scaling": self.recurrent_scaling,
            "recurrent_norm": self.recurrent_norm if self.recurrent_scaling == "operator_norm" else None,
            "spectral_radius": self.spectral_radius if self.recurrent_scaling == "spectral_radius" else None,
            "raw_readout_lags": self.raw_readout_lags, "connectivity": self.connectivity,
            "feature_scale_floor": self.feature_scale_floor,
        }

    def _state_update(self, previous_y, previous_state):
        previous_y = np.asarray(previous_y, dtype=np.float64)
        previous_state = np.asarray(previous_state, dtype=np.float64)
        if previous_y.shape != (self.n_sites,) or previous_state.shape != (self.state_size,):
            raise ValueError("Raw-field update needs one full input field and one matching state")
        old_history = previous_state[:self.history_size].reshape(self.maxlag, self.n_sites)
        result = np.empty(self.state_size, dtype=np.float64)
        history = result[:self.history_size].reshape(self.maxlag, self.n_sites)
        history[0] = (previous_y - self.site_center) / self.site_scale
        history[1:] = old_history[:-1]
        incoming = history[self._lag_index].reshape(self.input_dimension)
        old = previous_state[self.history_size:]
        result[self.history_size:] = ((1 - self.leaks[0]) * old + self.leaks[0]
            * np.tanh(self._sparse_W[0] @ old + self.U[0] @ incoming + self.bias[0]))
        return result

    def _raw(self, state):
        history = state[:self.history_size].reshape(self.maxlag, self.n_sites)
        direct = history[self._readout_lag_index].reshape(self.readout_input_dimension)
        if self.score_bound is not None:
            direct = self.score_bound * np.tanh(direct / self.score_bound)
        reservoir = state[self.history_size:]
        return np.concatenate([direct, reservoir, reservoir[self.square_index] ** 2,
                               reservoir[self.product_i] * reservoir[self.product_j]])

    def update(self, previous_y, previous_state):
        state = self._state_update(previous_y, previous_state)
        return state, np.r_[1.0, (self._raw(state) - self.center) / self.scale / self.norm]

    def update_batch(self, previous_y, previous_states):
        previous_y = np.asarray(previous_y, dtype=np.float64)
        previous_states = np.asarray(previous_states, dtype=np.float64)
        if (previous_y.ndim != 2 or previous_states.ndim != 2
                or len(previous_y) != len(previous_states)
                or previous_y.shape[1] != self.n_sites
                or previous_states.shape[1] != self.state_size):
            raise ValueError("Invalid raw-field batch response/state shapes")
        batch = len(previous_y)
        old_history = previous_states[:, :self.history_size].reshape(batch, self.maxlag, self.n_sites)
        states = np.empty_like(previous_states, dtype=np.float64)
        history = states[:, :self.history_size].reshape(batch, self.maxlag, self.n_sites)
        history[:, 0] = (previous_y - self.site_center) / self.site_scale
        history[:, 1:] = old_history[:, :-1]
        incoming = history[:, self._lag_index].reshape(batch, self.input_dimension)
        old = previous_states[:, self.history_size:]
        states[:, self.history_size:] = ((1 - self.leaks[0]) * old + self.leaks[0]
            * np.tanh((self._sparse_W[0] @ old.T).T + incoming @ self.U[0].T + self.bias[0]))
        direct = history[:, self._readout_lag_index].reshape(batch, self.readout_input_dimension)
        if self.score_bound is not None:
            direct = self.score_bound * np.tanh(direct / self.score_bound)
        reservoir = states[:, self.history_size:]
        raw = np.concatenate([direct, reservoir, reservoir[:, self.square_index] ** 2,
                               reservoir[:, self.product_i] * reservoir[:, self.product_j]], axis=1)
        z = np.empty((batch, 1 + raw.shape[1]), dtype=np.float64)
        z[:, 0] = 1.0
        z[:, 1:] = (raw - self.center) / self.scale / self.norm
        return states, z
