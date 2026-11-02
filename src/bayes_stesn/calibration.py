"""Explicit postprocessing of historical forecast errors into interval bands.

Raw predictive draws are never modified. Exchangeability is an assumption,
not something a time-series calibration split establishes automatically.
In particular, multiple sites/leads are not independent calibration blocks.
The reservoir-weighted option here is an explicitly defined empirical
kernel-weighted procedure, not a claim to reproduce the published ResCP
algorithm or its asymptotic guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

__all__ = [
    "CalibrationMap", "conformal_quantile", "fit_standardized_calibration",
    "calibrated_intervals", "reservoir_weighted_calibration",
]


@dataclass(frozen=True)
class CalibrationMap:
    """Historical scores and the declared interval domain/policy metadata."""

    alpha: float
    target: str
    quantiles: np.ndarray
    scores: np.ndarray
    blocks: int
    domain_shape: tuple[int, int]
    scale_floor: float
    metadata: dict = field(default_factory=dict)
    weighted: bool = False
    weights: np.ndarray | None = None
    effective_blocks: float | None = None
    guarantee: str = "Finite-sample rank requires exchangeable calibration and forecast blocks; dependent time series receive empirical assessment only."


def conformal_quantile(scores: np.ndarray, alpha: float = 0.1, *, axis: int = 0) -> np.ndarray:
    """Exact conformal order statistic with its necessary infinity atom.

    For ``m`` block scores, use one-based rank
    ``ceil((m+1)*(1-alpha))``. If it exceeds ``m``, return infinity with
    the other axes preserved. Interpolation or a capped rank would give
    a different procedure and invalidate the usual rank argument.
    """
    s = np.asarray(scores, dtype=float)
    if s.ndim < 1 or not np.isfinite(s).all() or not 0 < alpha < 1:
        raise ValueError("scores must be finite and alpha must lie in (0,1)")
    if not isinstance(axis, (int, np.integer)) or not -s.ndim <= axis < s.ndim:
        raise ValueError("axis is outside the score array dimensions")
    axis = axis % s.ndim
    m = s.shape[axis]
    if m < 1:
        raise ValueError("at least one calibration score is required")
    rank = int(np.ceil((m + 1) * (1 - alpha)))
    if rank > m:
        shape = s.shape[:axis] + s.shape[axis + 1 :]
        return np.full(shape, np.inf)
    return np.take(np.partition(s, rank - 1, axis=axis), rank - 1, axis=axis)


def _scale(scale: np.ndarray, floor: float) -> np.ndarray:
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("scale_floor must be finite and strictly positive")
    supplied = np.asarray(scale, dtype=float)
    if not np.isfinite(supplied).all() or np.any(supplied < 0):
        raise ValueError("predictive scales must be finite and nonnegative")
    return np.maximum(supplied, floor)


def fit_standardized_calibration(
    truth: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    *,
    alpha: float = 0.1,
    target: str = "marginal",
    scale_floor: float = 1e-8,
    metadata: Mapping | None = None,
) -> CalibrationMap:
    """Fit standardized absolute errors on ``(block, lead, site)`` arrays.

    ``marginal`` produces a separate quantile for each lead/site.
    ``horizon`` takes the maximum across sites within every block and
    produces a simultaneous field band separately at each lead.
    ``simultaneous`` takes a maximum over the full lead/site domain and
    produces one quantile for a complete trajectory band. Each target
    retains the original number of calibration blocks: sites and leads
    never increase the exchangeability sample size.

    Scales come from each historical forecast, not its realized error.
    A declared positive floor handles degenerate empirical forecast SDs.
    ``metadata`` should identify the historical period, forecast policy,
    site IDs, and whether updates are permitted; high-level split-aware
    APIs enforce those restrictions before calling this array routine.
    """
    y, mu, supplied_scale = (np.asarray(value, dtype=float) for value in (truth, center, scale))
    if y.ndim != 3 or min(y.shape) < 1 or y.shape != mu.shape or y.shape != supplied_scale.shape or not np.isfinite(y).all() or not np.isfinite(mu).all():
        raise ValueError("truth, center and scale must share nonempty finite (block, lead, site) axes")
    standardized = np.abs(y - mu) / _scale(supplied_scale, scale_floor)
    if target == "marginal":
        scores = standardized
    elif target == "horizon":
        scores = np.max(standardized, axis=2, keepdims=True)
    elif target == "simultaneous":
        scores = np.max(standardized, axis=(1, 2), keepdims=True)
    else:
        raise ValueError("target must be marginal, horizon, or simultaneous")
    quantiles = conformal_quantile(scores, alpha, axis=0)
    info = dict(metadata or {})
    info["scale_floor"] = float(scale_floor)
    info["floored_scale_count"] = int(np.sum(supplied_scale < scale_floor))
    info["calibration_unit"] = "forecast block"
    info["raw_posterior_unchanged"] = True
    return CalibrationMap(alpha, target, quantiles, scores.copy(), y.shape[0], tuple(y.shape[1:]), scale_floor, info)


def calibrated_intervals(
    calibration: CalibrationMap,
    center: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lower/upper bands, keeping the fitted calibration map fixed.

    Arrays end in the original ``(lead, site)`` domain and may have leading
    batch axes. This function has no argument for future truth and performs
    no within-forecast update. Infinite bands are a legitimate signal of
    insufficient calibration blocks, not a numerical failure to suppress.
    """
    mu, sd = np.asarray(center, dtype=float), np.asarray(scale, dtype=float)
    if mu.ndim < 2 or mu.shape != sd.shape or tuple(mu.shape[-2:]) != calibration.domain_shape or not np.isfinite(mu).all():
        raise ValueError("prediction shape must match the calibrated lead/site domain")
    radius = calibration.quantiles * _scale(sd, calibration.scale_floor)
    return mu - radius, mu + radius


def reservoir_weighted_calibration(
    calibration: CalibrationMap,
    calibration_states: np.ndarray,
    forecast_state: np.ndarray,
    *,
    bandwidth: float = 1.0,
    temporal_decay: float | None = None,
    test_weight: float = 1.0,
) -> CalibrationMap:
    """Empirical state-local weighted calibration with an infinity test atom.

    Historical block weights are ``exp(-||r_b-r_new||²/(2*bandwidth²))``.
    Optional decay multiplies by ``temporal_decay**age_b`` with the last
    calibration block most recent. The forecast receives the declared
    ``test_weight`` mass at infinity. Sorting each score column and taking
    its left weighted ``1-alpha`` quantile defines the algorithm completely.
    Equal states, no decay, and test weight one recover the exact ordinary
    conformal rank. State normalization must be learned before calibration.

    State similarity weights are not known density ratios; no finite-sample
    coverage theorem follows. The effective historical block count is a
    weight concentration diagnostic, not the number of independent blocks.
    Hyperparameters must be fixed using earlier development data.
    """
    if calibration.weighted:
        raise ValueError("weight original historical scores, not an already weighted calibration map")
    states, new = np.asarray(calibration_states, dtype=float), np.asarray(forecast_state, dtype=float)
    if states.ndim != 2 or states.shape[0] != calibration.blocks or new.shape != (states.shape[1],) or states.shape[1] < 1 or not np.isfinite(states).all() or not np.isfinite(new).all():
        raise ValueError("require finite calibration states[block,feature] and forecast_state[feature]")
    if not np.isfinite(bandwidth) or bandwidth <= 0 or not np.isfinite(test_weight) or test_weight <= 0:
        raise ValueError("bandwidth and test_weight must be finite and positive")
    if temporal_decay is not None and (not np.isfinite(temporal_decay) or not 0 < temporal_decay <= 1):
        raise ValueError("temporal_decay must be in (0,1]")
    log_weights = -0.5 * np.sum(((states - new) / bandwidth) ** 2, axis=1)
    if temporal_decay is not None:
        log_weights += np.arange(calibration.blocks - 1, -1, -1) * np.log(temporal_decay)
    log_test = np.log(test_weight)
    shift = max(float(np.max(log_weights)), float(log_test))
    weights = np.exp(log_weights - shift)
    test = np.exp(log_test - shift)
    total = weights.sum() + test
    flat = calibration.scores.reshape(calibration.blocks, -1)
    quantiles = np.full(flat.shape[1], np.inf)
    threshold = 1 - calibration.alpha
    for j in range(flat.shape[1]):
        order = np.argsort(flat[:, j], kind="stable")
        cumulative = np.cumsum(weights[order]) / total
        index = int(np.searchsorted(cumulative, threshold, side="left"))
        if index < calibration.blocks:
            quantiles[j] = flat[order[index], j]
    historical_sum = weights.sum()
    effective = float(historical_sum**2 / np.dot(weights, weights)) if np.dot(weights, weights) > 0 else 0.0
    info = dict(calibration.metadata)
    info.update(weighting_rule="Gaussian reservoir-state kernel with optional geometric time decay", bandwidth=bandwidth, temporal_decay=temporal_decay, test_weight=test_weight, test_probability=float(test / total), coverage_status="empirical; no exchangeability or ResCP guarantee")
    return CalibrationMap(calibration.alpha, calibration.target, quantiles.reshape(calibration.quantiles.shape), calibration.scores.copy(), calibration.blocks, calibration.domain_shape, calibration.scale_floor, info, True, weights / total, effective, "Empirical kernel-weighted calibration; no automatic finite-sample or published ResCP guarantee.")
