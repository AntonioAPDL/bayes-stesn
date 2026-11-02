"""Coherent trajectory mixtures, predictive stacking, and finite model indices.

Stacking is a validation-based predictive optimization, not Bayesian model
selection. ``sample_model_index`` is a separate Gibbs/Metropolis algorithm
whose callbacks must evaluate normalized likelihoods and proper priors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

__all__ = [
    "MixtureDraws", "StackingResult", "IndexSamples", "mix_trajectories",
    "fit_logscore_stacking", "fit_crps_stacking", "sample_model_index",
]


@dataclass(frozen=True)
class MixtureDraws:
    """Complete mixture trajectories and their source component/path IDs."""

    values: np.ndarray
    component_index: np.ndarray
    draw_index: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class StackingResult:
    weights: np.ndarray
    objective: float
    method: str
    cases: int
    success: bool
    message: str


@dataclass(frozen=True)
class IndexSamples:
    """Joint retained draws, not point estimates of component evidences."""

    theta: np.ndarray
    model_index: np.ndarray
    acceptance_rate: np.ndarray
    model_probabilities: np.ndarray
    final_proposal_scale: np.ndarray
    warmup: int


def _weights(weights: np.ndarray | None, count: int) -> np.ndarray:
    w = np.ones(count) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (count,) or not np.isfinite(w).all() or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("weights must be finite nonnegative model weights with positive sum")
    scaled = w / np.max(w)
    return scaled / scaled.sum()


def mix_trajectories(
    components: Sequence[np.ndarray],
    weights: np.ndarray | None = None,
    *,
    paths: int | None = None,
    seed: int = 0,
) -> MixtureDraws:
    """Sample one component and one complete source path per output path.

    Components have ``(path, lead, site)`` axes. Their path counts may
    differ. Default output count is the smallest component count, so adding
    members does not silently increase the forecast simulation budget.
    A one-member ensemble with ``paths=None`` preserves its source draws
    exactly. Positive supplied weights are normalized.
    """
    arrays = [np.asarray(x, dtype=float) for x in components]
    if not arrays or any(x.ndim != 3 or min(x.shape) < 1 or not np.isfinite(x).all() for x in arrays):
        raise ValueError("components must be nonempty finite (path, lead, site) arrays")
    if any(x.shape[1:] != arrays[0].shape[1:] for x in arrays):
        raise ValueError("all components must share lead and site axes")
    w = _weights(weights, len(arrays))
    if len(arrays) == 1 and paths is None:
        n = len(arrays[0])
        return MixtureDraws(arrays[0].copy(), np.zeros(n, dtype=int), np.arange(n), w)
    n = min(len(x) for x in arrays) if paths is None else paths
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("paths must be a positive integer")
    rng = np.random.default_rng(seed)
    component_index = rng.choice(len(arrays), size=n, p=w)
    draw_index = np.empty(n, dtype=int)
    out = np.empty((n,) + arrays[0].shape[1:])
    for k, values in enumerate(arrays):
        locations = np.flatnonzero(component_index == k)
        selected = rng.integers(len(values), size=len(locations))
        out[locations] = values[selected]
        draw_index[locations] = selected
    return MixtureDraws(out, component_index, draw_index, w)


def _simplex_optimize(objective, gradient, count: int) -> tuple[np.ndarray, str]:
    initial = np.full(count, 1 / count)
    if count == 1:
        return initial, "single component"
    result = minimize(
        objective, initial, jac=gradient, method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(count)},
        options={"ftol": 1e-11, "maxiter": 1000},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"stacking optimization failed: {result.message}")
    weights = np.maximum(result.x, 0)
    weights /= weights.sum()
    return weights, str(result.message)


def fit_logscore_stacking(log_density: np.ndarray, *, case_weights: np.ndarray | None = None) -> StackingResult:
    """Maximize held-out log score on a ``(validation_block, model)`` array.

    Each entry must be a correctly integrated joint block predictive
    density. The parameter draw is shared inside that block likelihood;
    products of independently marginalized transition densities are not
    valid substitutes. Data splitting and frozen/refitted posterior policy
    are the caller's explicit responsibility. ``-inf`` densities are
    allowed if every block has at least one finite component.
    """
    lp = np.asarray(log_density, dtype=float)
    if lp.ndim != 2 or min(lp.shape) < 1 or np.isnan(lp).any() or np.isposinf(lp).any() or not np.isfinite(lp).any(axis=1).all():
        raise ValueError("log_density must have a finite-supported block-by-model matrix")
    b, k = lp.shape
    cw = _weights(case_weights, b)
    offset = np.max(lp, axis=1)
    likelihood = np.exp(lp - offset[:, None])

    def objective(w):
        mixture = likelihood @ w
        return -float(np.dot(cw, np.log(np.maximum(mixture, np.finfo(float).tiny))))

    def gradient(w):
        mixture = np.maximum(likelihood @ w, np.finfo(float).tiny)
        return -(likelihood.T @ (cw / mixture))

    weights, message = _simplex_optimize(objective, gradient, k)
    positive = weights > 0
    log_mix = logsumexp(lp[:, positive] + np.log(weights[positive]), axis=1)
    return StackingResult(weights, float(np.dot(cw, log_mix)), "validation joint log score (maximize)", b, True, message)


def _mean_cross_distance(a: np.ndarray, b: np.ndarray, case_weights: np.ndarray) -> float:
    """Mean |A-B| for independent draws from two empirical marginals."""
    total = 0.0
    # Binary searches against sorted b avoid an M-by-M temporary matrix.
    ordered = np.sort(b, axis=0)
    prefix = np.concatenate((np.zeros((1, b.shape[1])), np.cumsum(ordered, axis=0)), axis=0)
    for j, weight in enumerate(case_weights):
        below = np.searchsorted(ordered[:, j], a[:, j], side="left")
        distance = a[:, j] * (2 * below - b.shape[0]) + prefix[-1, j] - 2 * prefix[below, j]
        total += float(weight * distance.mean() / b.shape[0])
    return total


def fit_crps_stacking(
    component_draws: Sequence[np.ndarray],
    truth: np.ndarray,
    *,
    case_weights: np.ndarray | None = None,
) -> StackingResult:
    """Minimize marginal CRPS of the actual empirical mixture distribution.

    Each component has axes ``(path, *truth.shape)``. Truth/cases are from
    validation only. The quadratic objective retains cross-component
    distances; averaging separate component CRPS values would be wrong.
    This optimizes marginal behavior, not joint trajectory dependence.
    """
    y = np.asarray(truth, dtype=float)
    arrays = [np.asarray(value, dtype=float) for value in component_draws]
    if not arrays or not y.size or not np.isfinite(y).all():
        raise ValueError("finite validation truth and at least one component are required")
    if any(x.ndim != y.ndim + 1 or x.shape[1:] != y.shape or x.shape[0] < 1 or not np.isfinite(x).all() for x in arrays):
        raise ValueError("each component must have one path axis followed by the truth shape")
    arrays = [x.reshape(x.shape[0], -1) for x in arrays]
    cw = _weights(None if case_weights is None else np.asarray(case_weights).reshape(-1), y.size)
    observed = y.reshape(-1)
    linear = np.array([np.dot(cw, np.mean(np.abs(x - observed), axis=0)) for x in arrays])
    k = len(arrays)
    cross = np.empty((k, k))
    for i in range(k):
        for j in range(i + 1):
            cross[i, j] = cross[j, i] = _mean_cross_distance(arrays[i], arrays[j], cw)

    def objective(w):
        return float(linear @ w - 0.5 * w @ cross @ w)

    weights, message = _simplex_optimize(objective, lambda w: linear - cross @ w, k)
    return StackingResult(weights, objective(weights), "validation marginal CRPS (minimize)", y.size, True, message)


def sample_model_index(
    normalized_log_likelihood: Callable[[int, np.ndarray], float],
    log_hyperprior: Callable[[int, np.ndarray], float],
    initial: np.ndarray,
    n_models: int,
    *,
    model_prior: np.ndarray | None = None,
    draws: int = 1000,
    warmup: int = 500,
    proposal_scale: float | np.ndarray = 0.35,
    model_dimensions: Sequence[int] | None = None,
    seed: int = 0,
    adapt: bool = True,
) -> IndexSamples:
    """Common-dimensional finite-index Gibbs/Metropolis sampler.

    The target is ``pi[k] * likelihood(k,theta) * hyperprior(k,theta)``.
    First sample the full conditional model index, then apply a symmetric
    Gaussian random-walk MH update to its common hyperparameter vector.
    Warmup alone adapts a global proposal multiplier; retained transitions
    use a fixed proposal. Priors in transformed coordinates must include
    their Jacobians. Callbacks must be deterministic, properly normalized,
    and return finite values or ``-inf``; a likelihood proportional only
    within one model is not sufficient for index probabilities.

    ``initial`` has axes ``(chain, parameter)`` (a vector requests one
    chain). Empty hyperparameter vectors are valid for a completely
    enumerated model. Different hyperparameter dimensions are explicitly
    rejected; this is not a reversible-jump algorithm. Conditional index
    visits estimate posterior probabilities, not component evidences.
    """
    state = np.asarray(initial, dtype=float)
    if state.ndim == 1:
        state = state[None, :]
    if state.ndim != 2 or state.shape[0] < 1 or not np.isfinite(state).all():
        raise ValueError("initial must be a finite (chain, parameter) array")
    if not isinstance(n_models, (int, np.integer)) or n_models < 1 or not isinstance(draws, (int, np.integer)) or draws < 1 or not isinstance(warmup, (int, np.integer)) or warmup < 0:
        raise ValueError("invalid model, draw, or warmup count")
    chains, dimension = state.shape
    if model_dimensions is not None and (len(model_dimensions) != n_models or any(d != dimension for d in model_dimensions)):
        raise ValueError("all models must have the same hyperparameter dimension")
    prior = _weights(model_prior, n_models)
    log_prior = np.full(n_models, -np.inf)
    log_prior[prior > 0] = np.log(prior[prior > 0])
    scale = np.broadcast_to(np.asarray(proposal_scale, dtype=float), (dimension,)).copy()
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("proposal scales must be finite and positive")

    def target(k, theta):
        likelihood = float(normalized_log_likelihood(k, theta))
        hyperprior = float(log_hyperprior(k, theta))
        if np.isnan(likelihood) or np.isposinf(likelihood) or np.isnan(hyperprior) or np.isposinf(hyperprior):
            raise ValueError("index-sampler callbacks must return finite log densities or -inf")
        return likelihood + hyperprior

    theta_trace = np.empty((chains, draws, dimension))
    index_trace = np.empty((chains, draws), dtype=int)
    acceptance = np.zeros(chains)
    scales = np.empty((chains, dimension))
    streams = np.random.SeedSequence(seed).spawn(chains)
    for chain in range(chains):
        rng = np.random.default_rng(streams[chain])
        theta = state[chain].copy()
        log_factor = 0.0
        accepted_retained = 0
        for iteration in range(warmup + draws):
            values = np.array([target(k, theta) for k in range(n_models)])
            log_prob = log_prior + values
            normalization = logsumexp(log_prob)
            if not np.isfinite(normalization):
                raise ValueError("initial/current theta has no support under any model")
            index = int(rng.choice(n_models, p=np.exp(log_prob - normalization)))
            accepted = False
            if dimension:
                proposal = theta + rng.normal(size=dimension) * scale * np.exp(log_factor)
                proposed_target = target(index, proposal)
                accepted = bool(np.log(rng.uniform()) < proposed_target - values[index])
                if accepted:
                    theta = proposal
                if adapt and iteration < warmup:
                    log_factor += ((1.0 if accepted else 0.0) - 0.234) / np.sqrt(iteration + 10.0)
                    log_factor = float(np.clip(log_factor, -10, 10))
            if iteration >= warmup:
                retained = iteration - warmup
                theta_trace[chain, retained] = theta
                index_trace[chain, retained] = index
                accepted_retained += accepted
        acceptance[chain] = accepted_retained / draws if dimension else np.nan
        scales[chain] = scale * np.exp(log_factor)
    probability = np.bincount(index_trace.ravel(), minlength=n_models) / index_trace.size
    return IndexSamples(theta_trace, index_trace, acceptance, probability, scales, warmup)
