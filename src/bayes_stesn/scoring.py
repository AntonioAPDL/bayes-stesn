"""Proper predictive scores, replication-level comparisons, and dependence checks.

Public forecast arrays use ``(path, lead, site)``. All scores are losses:
smaller is better. Truth must be supplied explicitly and is never imported
by fitting or forecasting code. Sample CRPS scores the empirical predictive
distribution exactly (the V-statistic, including its diagonal pairs).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

__all__ = [
    "crps_ensemble", "energy_score", "variogram_score", "geometry_pairs",
    "score_predictions", "paired_bootstrap", "dependence_summary",
]


def _samples(draws: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(draws, dtype=float), np.asarray(truth, dtype=float)
    if x.ndim != y.ndim + 1 or x.shape[1:] != y.shape or x.shape[0] < 1 or y.size == 0:
        raise ValueError("draws must have one leading path axis followed by the truth shape")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("forecast draws and truth must be finite")
    return x, y


def crps_ensemble(draws: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Exact empirical CRPS for ``draws[path,...]`` and ``truth[...]``.

    The sorted-sample identity avoids allocating a path-by-path distance
    matrix. A single predictive draw reduces to absolute error. No claim
    of unbiased estimation of the underlying continuous-distribution CRPS
    is made: this is the proper score of the supplied empirical forecast.
    """
    x, y = _samples(draws, truth)
    count = x.shape[0]
    ordered = np.sort(x, axis=0)
    coefficients = (2 * np.arange(1, count + 1) - count - 1).reshape((count,) + (1,) * y.ndim)
    result = np.mean(np.abs(x - y), axis=0) - np.sum(coefficients * ordered, axis=0) / count**2
    return np.maximum(result, 0.0)


def energy_score(
    draws: np.ndarray,
    truth: np.ndarray,
    *,
    pair_count: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Energy score over the last (joint-vector) axis.

    Input ``(path, *case, dimension)`` returns ``(*case,)``. With
    ``pair_count=None`` the empirical score is exact. Otherwise independent
    pairs with replacement approximate its second expectation; the fixed
    seed makes this reproducible, and Monte Carlo estimates can be slightly
    negative. Units are those of an unnormalized Euclidean vector norm.
    """
    x, y = _samples(draws, truth)
    if y.ndim < 1:
        raise ValueError("energy score requires a joint-vector axis")
    if pair_count is not None and (not isinstance(pair_count, (int, np.integer)) or pair_count < 1):
        raise ValueError("pair_count must be a positive integer or None")
    paths, dimension = x.shape[0], x.shape[-1]
    cases = int(np.prod(y.shape[:-1]))
    xx, yy = x.reshape(paths, cases, dimension), y.reshape(cases, dimension)
    out = np.empty(cases)
    pairs = None
    if pair_count is not None:
        pairs = np.random.default_rng(seed).integers(paths, size=(2, pair_count))
    for j in range(cases):
        first = np.linalg.norm(xx[:, j] - yy[j], axis=-1).mean()
        if pairs is None:
            # pdist contains each off-diagonal pair once; the 1/2 cancels
            # the corresponding pair in the full symmetric distance sum.
            second = pdist(xx[:, j], metric="euclidean").sum() / paths**2
        else:
            second = 0.5 * np.linalg.norm(xx[pairs[0], j] - xx[pairs[1], j], axis=-1).mean()
        out[j] = first - second
    return out.reshape(y.shape[:-1])


def geometry_pairs(
    coordinates: np.ndarray,
    *,
    max_pairs: int | None = 512,
    seed: int = 0,
    inverse_distance: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Select fixed geometry-only pairs and normalized nonnegative weights.

    Pairs are selected uniformly without replacement before seeing any
    responses. Default weights are inverse distance on that selected
    domain. The pair set, seed, and weights must be shared across methods.
    """
    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[0] < 2 or not np.isfinite(coords).all():
        raise ValueError("coordinates must be a finite (site, dimension) array with at least two sites")
    pairs = np.column_stack(np.triu_indices(coords.shape[0], k=1))
    if max_pairs is not None:
        if not isinstance(max_pairs, (int, np.integer)) or max_pairs < 1:
            raise ValueError("max_pairs must be positive or None")
        if pairs.shape[0] > max_pairs:
            selected = np.sort(np.random.default_rng(seed).choice(pairs.shape[0], max_pairs, replace=False))
            pairs = pairs[selected]
    distances = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
    if np.any(distances <= 0):
        raise ValueError("distinct sites must have distinct coordinates")
    weights = 1.0 / distances if inverse_distance else np.ones(pairs.shape[0])
    return pairs, weights / weights.sum()


def variogram_score(
    draws: np.ndarray,
    truth: np.ndarray,
    *,
    power: float = 0.5,
    pairs: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Variogram loss for the last site axis with explicitly weighted pairs.

    Returns one score per leading case (usually forecast lead). Weights
    are normalized to sum to one. With omitted pairs all unordered site
    pairs receive equal weight. A variogram score is insensitive to a
    common additive field shift; assess it together with marginal/energy
    scores. This implementation bounds temporary memory by pair chunks.
    """
    x, y = _samples(draws, truth)
    if y.ndim < 1 or y.shape[-1] < 2 or not 0 < power < 2:
        raise ValueError("variogram score requires >=2 sites and 0<power<2")
    n = y.shape[-1]
    if pairs is None:
        pairs = np.column_stack(np.triu_indices(n, k=1))
    else:
        supplied = np.asarray(pairs)
        if supplied.ndim != 2 or supplied.shape[1] != 2 or supplied.shape[0] == 0 or not np.isfinite(supplied).all() or np.any(supplied != np.floor(supplied)):
            raise ValueError("pairs must be a nonempty integer (pair, 2) array")
        pairs = supplied.astype(int)
    if np.any(pairs < 0) or np.any(pairs >= n) or np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError("pairs contain invalid site indices")
    w = np.ones(pairs.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (pairs.shape[0],) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative, and sum to a positive value")
    w = w / w.sum()
    paths = x.shape[0]
    cases = int(np.prod(y.shape[:-1]))
    xx, yy = x.reshape(paths, cases, n), y.reshape(cases, n)
    out = np.zeros(cases)
    for j in range(cases):
        for start in range(0, pairs.shape[0], 256):
            pair = pairs[start : start + 256]
            observed = np.abs(yy[j, pair[:, 0]] - yy[j, pair[:, 1]]) ** power
            predictive = np.mean(np.abs(xx[:, j, pair[:, 0]] - xx[:, j, pair[:, 1]]) ** power, axis=0)
            out[j] += np.dot(w[start : start + 256], (observed - predictive) ** 2)
    return out.reshape(y.shape[:-1])


def score_predictions(
    draws: np.ndarray,
    truth: np.ndarray,
    *,
    interval_levels: tuple[float, ...] = (0.9, 0.95),
    coordinates: np.ndarray | None = None,
    joint_scores: bool = True,
    energy_pairs: int | None = 2048,
    variogram_pairs: int | None = 512,
    seed: int = 0,
) -> dict:
    """Score observation or explicitly labeled alternative-target forecasts.

    Arrays must have ``(path, lead, site)`` and ``(lead, site)`` shapes.
    Point predictions are path means. Joint scores evaluate one spatial
    field per lead, not the complete temporal trajectory. The default
    energy estimate uses 2,048 sampled path pairs; pass ``None`` for exact
    computation. Marginal CRPS and interval scores are always exact for the
    empirical predictive distribution. No scale normalization is implicit.
    """
    x, y = _samples(draws, truth)
    if x.ndim != 3:
        raise ValueError("score_predictions requires (path, lead, site) draws")
    for level in interval_levels:
        if not 0 < level < 1:
            raise ValueError("interval levels must be between zero and one")
    if variogram_pairs is not None and (not isinstance(variogram_pairs, (int, np.integer)) or variogram_pairs < 1):
        raise ValueError("variogram_pairs must be a positive integer or None")
    error = x.mean(axis=0) - y
    crps = crps_ensemble(x, y)
    result = {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "crps": float(crps.mean()),
        "rmse_by_lead": np.sqrt(np.mean(error**2, axis=1)),
        "mae_by_lead": np.mean(np.abs(error), axis=1),
        "crps_by_lead": crps.mean(axis=1),
        "intervals": {},
        "paths": x.shape[0], "leads": x.shape[1], "sites": x.shape[2],
        "joint_domain": "one spatial field at each lead",
        "crps_estimator": "exact empirical distribution (V-statistic)",
    }
    for level in interval_levels:
        lo, hi = np.quantile(x, [(1 - level) / 2, (1 + level) / 2], axis=0)
        covered, width = (y >= lo) & (y <= hi), hi - lo
        result["intervals"][f"{level:g}"] = {
            "coverage": float(covered.mean()), "width": float(width.mean()),
            "coverage_by_lead": covered.mean(axis=1), "width_by_lead": width.mean(axis=1),
        }
    if joint_scores:
        energy = energy_score(x, y, pair_count=energy_pairs, seed=seed)
        result.update(energy=float(energy.mean()), energy_by_lead=energy, energy_pair_count=energy_pairs)
        if y.shape[1] >= 2:
            if coordinates is None:
                pairs = np.column_stack(np.triu_indices(y.shape[1], k=1))
                if variogram_pairs is not None and len(pairs) > variogram_pairs:
                    pairs = pairs[np.sort(np.random.default_rng(seed).choice(len(pairs), variogram_pairs, replace=False))]
                weights = np.full(len(pairs), 1 / len(pairs))
            else:
                if np.asarray(coordinates).shape[0] != y.shape[1]:
                    raise ValueError("coordinates must match the forecast site axis")
                pairs, weights = geometry_pairs(coordinates, max_pairs=variogram_pairs, seed=seed)
            variogram = variogram_score(x, y, pairs=pairs, weights=weights)
            result.update(variogram=float(variogram.mean()), variogram_by_lead=variogram, variogram_power=0.5, variogram_pair_count=len(pairs), variogram_pair_indices=pairs, variogram_weights=weights)
    return result


def paired_bootstrap(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> dict:
    """Percentile interval for a paired mean loss difference, ``a - b``.

    Inputs contain one pre-aggregated value per independent replication.
    Origins/sites/leads from a single run must be averaged within that
    replication first. The returned distribution exposes the actual
    resampling uncertainty; this is not a test of model correctness.
    """
    a, b = np.asarray(loss_a, dtype=float), np.asarray(loss_b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or a.size < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("losses must be paired finite vectors with >=2 independent replications")
    if not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1 or not 0 < level < 1:
        raise ValueError("invalid resample count or interval level")
    delta = a - b
    rng = np.random.default_rng(seed)
    # Bound memory for research profiles with many replications/resamples.
    bootstrap = np.empty(n_resamples)
    for first in range(0, n_resamples, 1024):
        count = min(1024, n_resamples - first)
        index = rng.integers(a.size, size=(count, a.size))
        bootstrap[first : first + count] = delta[index].mean(axis=1)
    lo, hi = np.quantile(bootstrap, [(1 - level) / 2, (1 + level) / 2])
    return {"difference": float(delta.mean()), "ci": (float(lo), float(hi)), "level": level, "replications": a.size, "paired_differences": delta, "bootstrap_means": bootstrap, "sign_convention": "loss_a - loss_b; negative favors a", "resampling_unit": "independent replication"}


def dependence_summary(
    fields: np.ndarray,
    coordinates: np.ndarray,
    *,
    max_lag: int = 40,
    distance_bins: int | np.ndarray = 12,
    max_pairs: int | None = 4096,
    seed: int = 0,
    grid_shape: tuple[int, int] | None = None,
) -> dict:
    """Temporal ACF, spatial/directional semivariograms, optional spectrum.

    ``fields`` has axes ``(time, site)``. Temporal ACF removes each site's
    time mean; semivariograms retain the structured spatial signal. The
    optional spectrum requires coordinates in row-major rectangular-grid
    order (x changes fastest); arbitrary locations are rejected. These
    descriptive checks do not supply confidence intervals under dependence.
    """
    x, c = np.asarray(fields, dtype=float), np.asarray(coordinates, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or c.ndim != 2 or c.shape[0] != x.shape[1] or c.shape[1] != 2 or not np.isfinite(x).all() or not np.isfinite(c).all():
        raise ValueError("require finite fields[time,site] and coordinates[site,2]")
    if not isinstance(max_lag, (int, np.integer)) or max_lag < 1:
        raise ValueError("max_lag must be a positive integer")
    max_lag = min(max_lag, x.shape[0] - 1)
    centered = x - x.mean(axis=0)
    variance = np.mean(centered**2)
    acf = np.array([np.mean(centered[lag:] * centered[:-lag]) / variance if variance > 0 else np.nan for lag in range(1, max_lag + 1)])
    pairs, _ = geometry_pairs(c, max_pairs=max_pairs, seed=seed, inverse_distance=False)
    displacement = c[pairs[:, 0]] - c[pairs[:, 1]]
    distances = np.linalg.norm(displacement, axis=1)
    if np.isscalar(distance_bins):
        if int(distance_bins) != distance_bins or distance_bins < 1:
            raise ValueError("distance_bins must be positive")
        edges = np.linspace(0, distances.max() * (1 + 1e-12), int(distance_bins) + 1)
    else:
        edges = np.asarray(distance_bins, dtype=float)
        if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0):
            raise ValueError("distance bin edges must strictly increase")
    semivariance = np.empty(len(pairs))
    for first in range(0, len(pairs), 256):
        selected = pairs[first : first + 256]
        semivariance[first : first + len(selected)] = 0.5 * np.mean((x[:, selected[:, 0]] - x[:, selected[:, 1]]) ** 2, axis=0)
    assignment = np.searchsorted(edges, distances, side="right") - 1
    # Directions are axial: horizontal/vertical cones of half-angle pi/8.
    angle = np.mod(np.arctan2(displacement[:, 1], displacement[:, 0]), np.pi)
    direction_masks = {
        "all": np.ones(len(pairs), dtype=bool),
        "horizontal": np.minimum(angle, np.pi - angle) <= np.pi / 8,
        "vertical": np.abs(angle - np.pi / 2) <= np.pi / 8,
    }
    variograms = {}
    for name, mask in direction_masks.items():
        counts = np.array([np.sum(mask & (assignment == j)) for j in range(len(edges) - 1)])
        values = np.array([np.mean(semivariance[mask & (assignment == j)]) if counts[j] else np.nan for j in range(len(edges) - 1)])
        variograms[name] = {"semivariance": values, "pair_counts": counts}
    result = {"lags": np.arange(1, max_lag + 1), "temporal_acf": acf, "distance_centers": (edges[:-1] + edges[1:]) / 2, "distance_edges": edges, "variograms": variograms, "pairs": pairs}
    if grid_shape is not None:
        ny, nx = grid_shape
        if nx < 2 or ny < 2 or nx * ny != x.shape[1]:
            raise ValueError("grid_shape must contain >=2 rows/columns and match the site count")
        gx, gy = np.unique(c[:, 0]), np.unique(c[:, 1])
        expected = np.column_stack((np.tile(gx, len(gy)), np.repeat(gy, len(gx))))
        if len(gx) != nx or len(gy) != ny or not np.allclose(c, expected, rtol=0, atol=1e-12) or not np.allclose(np.diff(gx), np.diff(gx)[0]) or not np.allclose(np.diff(gy), np.diff(gy)[0]):
            raise ValueError("spectrum requires a uniformly spaced row-major rectangular grid")
        grid = x.reshape(-1, ny, nx)
        grid = grid - grid.mean(axis=(1, 2), keepdims=True)
        power = np.abs(np.fft.rfft2(grid, axes=(1, 2), norm="ortho")) ** 2
        result["spatial_power"] = power.mean(axis=0)
        result["frequency_x"] = np.fft.rfftfreq(nx, d=float(np.diff(gx)[0]))
        result["frequency_y"] = np.fft.fftfreq(ny, d=float(np.diff(gy)[0]))
    return result
