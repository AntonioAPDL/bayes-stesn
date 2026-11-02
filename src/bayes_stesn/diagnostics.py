"""Conservative rank-normalized MCMC diagnostics and SBC rank bookkeeping.

The computations follow Vehtari et al. (2021), DOI 10.1214/20-BA1221:
split chains, pooled Blom normal scores, folded rank R-hat, and Geyer's
initial-positive/initial-monotone autocorrelation estimates.  Passing these
screening thresholds is not a proof of convergence.  Constant, nonfinite,
short, or single-chain samples cannot pass the default gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

__all__ = ["DiagnosticSummary", "summarize_chains", "sbc_ranks", "sbc_histogram"]


@dataclass(frozen=True)
class DiagnosticSummary:
    """Scalar diagnostics flattened in ``parameter_names`` order."""

    parameter_names: tuple[str, ...]
    mean: np.ndarray
    sd: np.ndarray
    rhat: np.ndarray
    ess_bulk: np.ndarray
    ess_tail: np.ndarray
    mcse_mean: np.ndarray
    parameter_passed: np.ndarray
    passed: bool
    messages: tuple[str, ...]
    chains: int
    draws: int

    def to_dict(self) -> dict:
        """Return ordinary containers suitable for a report or JSON encoder.

        Undefined numerical diagnostics are represented by ``None`` rather
        than nonstandard JSON NaN/Infinity literals.
        """
        def finite(value: float) -> float | None:
            return float(value) if np.isfinite(value) else None

        return {
            "passed": self.passed,
            "chains": self.chains,
            "draws": self.draws,
            "messages": list(self.messages),
            "parameters": {
                name: {
                    "mean": finite(self.mean[j]),
                    "sd": finite(self.sd[j]),
                    "rhat": finite(self.rhat[j]),
                    "ess_bulk": finite(self.ess_bulk[j]),
                    "ess_tail": finite(self.ess_tail[j]),
                    "mcse_mean": finite(self.mcse_mean[j]),
                    "passed": bool(self.parameter_passed[j]),
                }
                for j, name in enumerate(self.parameter_names)
            },
        }


def _split(x: np.ndarray) -> np.ndarray:
    half = x.shape[1] // 2
    return np.concatenate((x[:, :half], x[:, -half:]), axis=0)


def _normal_scores(x: np.ndarray) -> np.ndarray:
    ranks = rankdata(x, method="average").reshape(x.shape)
    return ndtri((ranks - 0.375) / (x.size + 0.25))


def _rhat(x: np.ndarray) -> float:
    n = x.shape[1]
    within = np.var(x, axis=1, ddof=1).mean()
    if not np.isfinite(within) or within <= 0:
        return float("inf")
    between = n * np.var(x.mean(axis=1), ddof=1)
    return float(np.sqrt((n - 1 + between / within) / n))


def _autocovariance(x: np.ndarray) -> np.ndarray:
    """Biased autocovariances, using a padded FFT to avoid wraparound."""
    n = x.shape[1]
    size = 1 << (2 * n - 1).bit_length()
    centered = x - x.mean(axis=1, keepdims=True)
    transformed = np.fft.rfft(centered, n=size, axis=1)
    return np.fft.irfft(transformed * transformed.conj(), n=size, axis=1)[:, :n] / n


def _ess(x: np.ndarray) -> float:
    """Multi-chain effective sample size with positive, monotone pair sums.

    The last positive even lag and the lower bound on integrated correlation
    time match the finite-sample convention used by Stan/ArviZ. Constant
    chains return zero here deliberately: they cannot establish mixing.
    """
    chains, n = x.shape
    if n < 4 or not np.isfinite(x).all() or np.ptp(x) == 0:
        return 0.0
    acov = _autocovariance(x)
    within = acov[:, 0].mean() * n / (n - 1)
    var_plus = within * (n - 1) / n + np.var(x.mean(axis=1), ddof=1)
    if var_plus <= 0 or not np.isfinite(var_plus):
        return 0.0
    rho = np.zeros(n)
    rho[0] = 1.0
    even = 1.0
    odd = 1.0 - (within - acov[:, 1].mean()) / var_plus
    rho[1] = odd
    lag = 1
    while lag < n - 3 and even + odd > 0:
        even = 1.0 - (within - acov[:, lag + 1].mean()) / var_plus
        odd = 1.0 - (within - acov[:, lag + 2].mean()) / var_plus
        if even + odd >= 0:
            rho[lag + 1 : lag + 3] = (even, odd)
        lag += 2
    last = lag - 2
    if even > 0:
        rho[last + 1] = even
    for lag in range(1, last - 1, 2):
        previous = rho[lag - 1] + rho[lag]
        if rho[lag + 1] + rho[lag + 2] > previous:
            rho[lag + 1 : lag + 3] = previous / 2
    tau = -1.0 + 2.0 * rho[: last + 1].sum() + rho[last + 1 : last + 2].sum()
    tau = max(float(tau), 1.0 / np.log10(chains * n))
    return float(chains * n / tau)


def summarize_chains(
    draws: np.ndarray,
    parameter_names: Sequence[str] | None = None,
    *,
    min_chains: int = 4,
    min_draws: int = 8,
    rhat_limit: float = 1.01,
    min_ess: float = 400.0,
) -> DiagnosticSummary:
    """Diagnose samples with axes ``(chain, draw, *parameter_shape)``.

    Warmup must already have been discarded. Odd chain lengths drop their
    middle draw for split-chain diagnostics. R-hat is the larger of rank
    and folded-rank values. Bulk ESS uses rank scores; tail ESS is the
    minimum of lower-5% and upper-95% quantile-indicator ESS. Mean MCSE uses
    untransformed split draws. At least two chains and eight draws are
    required even if the caller asks for a weaker operational gate.
    """
    x = np.asarray(draws, dtype=float)
    if x.ndim < 2 or min(x.shape) < 1:
        raise ValueError("draws must have nonempty (chain, draw, ...) axes")
    if min_chains < 2 or min_draws < 8 or rhat_limit <= 1 or min_ess <= 0:
        raise ValueError("diagnostic gates require chains>=2, draws>=8, R-hat>1, ESS>0")
    chains, n = x.shape[:2]
    x = x.reshape(chains, n, -1)
    p = x.shape[2]
    names = tuple(parameter_names) if parameter_names is not None else tuple(f"parameter_{j}" for j in range(p))
    if len(names) != p or len(set(names)) != p:
        raise ValueError("parameter_names must uniquely label all flattened parameters")
    mean = np.full(p, np.nan)
    sd = np.full(p, np.nan)
    rhat = np.full(p, np.inf)
    bulk = np.zeros(p)
    tail = np.zeros(p)
    mcse = np.full(p, np.inf)
    messages: list[str] = []
    enough = chains >= min_chains and n >= min_draws
    if not enough:
        messages.append(f"Insufficient samples for the declared gate: {chains} chains × {n} draws.")
    for j in range(p):
        sample = x[:, :, j]
        if not np.isfinite(sample).all():
            messages.append(f"{names[j]} has nonfinite draws; diagnostics fail.")
            continue
        mean[j] = sample.mean()
        sd[j] = sample.std(ddof=1) if sample.size > 1 else np.nan
        if np.ptp(sample) == 0:
            messages.append(f"{names[j]} is constant; mixing cannot be established.")
            continue
        if chains < 2 or n < 8:
            continue
        split = _split(sample)
        rank = _normal_scores(split)
        folded = _normal_scores(np.abs(split - np.median(split)))
        rhat[j] = max(_rhat(rank), _rhat(folded))
        bulk[j] = _ess(rank)
        q05, q95 = np.quantile(sample, (0.05, 0.95))
        tail[j] = min(_ess(_split((sample <= q05).astype(float))), _ess(_split((sample <= q95).astype(float))))
        ess_mean = _ess(split)
        if ess_mean > 0:
            mcse[j] = sd[j] / np.sqrt(ess_mean)
    per_parameter = (
        enough & np.isfinite(rhat) & (rhat < rhat_limit)
        & (bulk >= min_ess) & (tail >= min_ess) & np.isfinite(mcse)
    )
    if enough and not np.all(per_parameter):
        failed = ", ".join(name for name, ok in zip(names, per_parameter) if not ok)
        messages.append(f"Diagnostic thresholds failed for: {failed}.")
    messages.append("Thresholds are screening criteria; assess relevant predictive summaries and MCSE as well.")
    return DiagnosticSummary(names, mean, sd, rhat, bulk, tail, mcse, per_parameter, bool(np.all(per_parameter)), tuple(messages), chains, n)


def sbc_ranks(
    true_parameters: np.ndarray,
    posterior_draws: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Randomized SBC ranks for arrays ``(replication, parameter...)``.

    Posterior draws have axes ``(replication, draw, parameter...)``. Each
    rank is uniform among the positions tied with the true value; for
    continuous posteriors ties usually vanish. Ranks range from zero to
    ``draw_count`` inclusive. The caller must supply independent prior
    simulations and adequately sampled posteriors. This function performs
    bookkeeping, not the recursive generative/refitting SBC experiment.
    """
    truth = np.asarray(true_parameters, dtype=float)
    posterior = np.asarray(posterior_draws, dtype=float)
    if truth.ndim < 1 or posterior.ndim != truth.ndim + 1 or posterior.shape[0] != truth.shape[0] or posterior.shape[2:] != truth.shape[1:] or posterior.shape[1] < 1:
        raise ValueError("SBC requires truth[replication,...] and draws[replication,draw,...]")
    if not np.isfinite(truth).all() or not np.isfinite(posterior).all():
        raise ValueError("SBC values must be finite")
    less = (posterior < truth[:, None, ...]).sum(axis=1)
    ties = (posterior == truth[:, None, ...]).sum(axis=1)
    rng = np.random.default_rng(seed)
    return less + rng.integers(0, ties + 1)


def sbc_histogram(ranks: np.ndarray, draw_count: int, bins: int = 10) -> dict:
    """Count SBC ranks per parameter with exact discrete-null expectations.

    Bins need not divide ``draw_count+1``. Expected probabilities therefore
    count the integer rank positions rather than assume identical bins.
    Parameters from one replication are not independent replicates.
    """
    ranks = np.asarray(ranks)
    if ranks.ndim < 1 or ranks.shape[0] < 1 or draw_count < 1 or not 1 <= bins <= draw_count + 1:
        raise ValueError("invalid rank array, draw count, or number of bins")
    if not np.isfinite(ranks).all() or np.any(ranks != np.floor(ranks)) or np.any((ranks < 0) | (ranks > draw_count)):
        raise ValueError("ranks must be integers from zero through draw_count")
    flat = ranks.reshape(ranks.shape[0], -1).astype(int)
    # Integer partition: every possible rank belongs to exactly one bin.
    assignment = np.minimum(np.arange(draw_count + 1) * bins // (draw_count + 1), bins - 1)
    counts = np.stack([np.bincount(assignment[flat[:, j]], minlength=bins) for j in range(flat.shape[1])])
    probabilities = np.bincount(assignment, minlength=bins) / (draw_count + 1)
    return {"counts": counts, "expected": ranks.shape[0] * probabilities, "probabilities": probabilities, "replications": ranks.shape[0], "draw_count": draw_count}
