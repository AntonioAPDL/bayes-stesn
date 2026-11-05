"""KS-specific mean-based, horizon-balanced and trajectory-clustered scores."""
from __future__ import annotations
import numpy as np
from ..scoring import crps_ensemble, energy_score, geometry_pairs, variogram_score

BANDS = ((0, 10), (10, 25), (25, 50), (50, 100))
LEVELS = (.5, .8, .9, .95)


def horizon_balance(values):
    x = np.asarray(values, np.float64)
    return float(np.mean([x[a:min(b, len(x))].mean() for a, b in BANDS if a < len(x)]))


def score(mean, samples, truth, training_scale, coordinates, *, pair_seed=20261105):
    """All point losses use the mean of complete paths for stochastic models."""
    y, m = np.asarray(truth, np.float64), np.asarray(mean, np.float64)
    if y.shape != m.shape or y.ndim != 2 or not np.isfinite(y).all() or not np.isfinite(m).all():
        raise ValueError("Finite [lead,site] predictions and truth required")
    if not np.isfinite(training_scale) or training_scale <= 0:
        raise ValueError("Positive permitted training scale required")
    if samples is not None:
        samples = np.asarray(samples, np.float64)
        if not np.allclose(m, samples.mean(axis=0), rtol=1e-12, atol=1e-12):
            raise ValueError("Primary point forecast must average complete predictive paths")
    error = m-y
    with np.errstate(over="ignore"):
        mse_h = np.mean(error*error, axis=1)
    if not np.isfinite(mse_h).all():
        raise FloatingPointError("Squared forecast error overflow; rollout fails")
    mae_h = np.mean(np.abs(error), axis=1)
    rmse_h = np.sqrt(mse_h)
    crossing = np.flatnonzero(rmse_h/training_scale > .4)
    vpt = int(crossing[0]+1) if len(crossing) else len(y)
    out = {"rmse": float(np.sqrt(mse_h.mean())), "mae": float(mae_h.mean()),
           "nrmse": float(np.sqrt(mse_h.mean())/training_scale),
           "mse_by_horizon": mse_h.tolist(), "mae_by_horizon": mae_h.tolist(),
           "rmse_by_horizon": rmse_h.tolist(), "training_scale": float(training_scale),
           "selection_mse": horizon_balance(mse_h), "vpt_steps": vpt, "vpt_physical": .25*vpt,
           "vpt_right_censored": not bool(len(crossing)),
           "diverged": bool(np.any(rmse_h/training_scale > 10.)),
           "point_policy": "mean_of_complete_predictive_paths" if samples is not None else "deterministic_point",
           "paths": 0 if samples is None else len(samples), "intervals": {}}
    if samples is None:
        return out
    crps_h = crps_ensemble(samples, y).mean(axis=1)
    out.update(crps=float(crps_h.mean()), normalized_crps=float(crps_h.mean()/training_scale),
               crps_by_horizon=crps_h.tolist(), selection_crps=horizon_balance(crps_h),
               crps_estimator="empirical V-statistic; no iid bias-correction claim")
    for level in LEVELS:
        lo, hi = np.quantile(samples, [(1-level)/2, (1+level)/2], axis=0)
        coverage = ((lo <= y) & (y <= hi)).mean(axis=1)
        width = (hi-lo).mean(axis=1)
        out["intervals"][str(level)] = {"coverage": float(coverage.mean()), "width": float(width.mean()),
                                              "coverage_by_horizon": coverage.tolist(), "width_by_horizon": width.tolist()}
    # Same path-pair scheme for all methods/origins, normalized spatial norm.
    energy = energy_score(samples, y, pair_count=1024, seed=pair_seed)/np.sqrt(y.shape[1])
    out.update(spatial_energy=float(energy.mean()), spatial_energy_by_horizon=energy.tolist(), energy_pairs=1024)
    pairs, weights = geometry_pairs(coordinates, max_pairs=128, seed=pair_seed)
    vg = variogram_score(samples, y, pairs=pairs, weights=weights, power=.5)
    out.update(spatial_variogram=float(vg.mean()), spatial_variogram_by_horizon=vg.tolist(),
               variogram_power=.5, spatial_pairs=pairs.tolist(), spatial_pair_weights=weights.tolist())
    return out


def cluster_summary(episode_values, *, seed=20261105, resamples=2000):
    """Equally weighted trajectories; uncertainty conditional on frozen selection."""
    x = np.asarray(episode_values, np.float64)
    if x.ndim != 1:
        raise ValueError("Cluster summary requires one scalar per trajectory")
    if not len(x) or not np.isfinite(x).all():
        return {"n": int(len(x)), "mean": None, "ci95": None}
    rng = np.random.default_rng(seed)
    draws = x[rng.integers(len(x), size=(resamples, len(x)))].mean(axis=1)
    return {"n": int(len(x)), "mean": float(x.mean()),
            "between_trajectory_standard_error": float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else None,
            "ci95": np.quantile(draws, [.025, .975]).tolist() if len(x)>1 else None,
            "method": "trajectory cluster bootstrap; all origins retained within each trajectory"}
