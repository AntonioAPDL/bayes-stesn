"""Validated latent, robust and heteroskedastic Gaussian-mixture samplers.

All samplers use ordinary Gibbs/MH sweeps. A collapsed update is never mixed
with stale augmented states. The implementation deliberately exposes resource
gates: these experimental models are reference-size routes, while the common
Gaussian model has a separately optimised collapsed sampler.
"""
from __future__ import annotations

import copy
import hashlib
import time
from typing import Any, Iterable

import numpy as np
from scipy import linalg, special
from scipy.sparse.linalg import LinearOperator, cg

Array = np.ndarray


def _chol(matrix: Array) -> Array:
    return linalg.cholesky((matrix+matrix.T)/2, lower=True, check_finite=False)


def _solve_chol(cholesky: Array, rhs: Array) -> Array:
    return linalg.cho_solve((cholesky, True), rhs, check_finite=False)


def _logdet_chol(cholesky: Array) -> float:
    return float(2*np.log(np.diag(cholesky)).sum())


def _filter_modes(residual: Array, K: Array, q: float, sigma2: float, rho: float
                  ) -> tuple[Array, Array, Array, Array]:
    residual = np.asarray(residual, dtype=float)
    if residual.ndim != 2 or residual.shape[0] != len(K) or residual.shape[1] < 1:
        raise ValueError("residual must be a complete n by T panel matching K")
    if q <= 0 or sigma2 <= 0 or abs(rho) >= 1:
        raise ValueError("q,sigma2 must be positive and |rho|<1")
    eigenvalues, U = linalg.eigh(K, check_finite=True)
    if eigenvalues[0] <= 0:
        raise ValueError("latent AR covariance must be positive definite at unique input sites")
    observations = U.T @ residual
    n, T = residual.shape
    means, variances, predictions = (np.empty((n, T)) for _ in range(3))
    next_mean = np.zeros(n)
    next_var = sigma2*q*eigenvalues
    innovation = sigma2*q*(1-rho*rho)*eigenvalues
    for t in range(T):
        predictions[:, t] = next_var
        gain = next_var/(next_var+sigma2)
        means[:, t] = next_mean+gain*(observations[:, t]-next_mean)
        # This expression is more stable than (1-gain)*next_var.
        variances[:, t] = next_var*sigma2/(next_var+sigma2)
        next_mean = rho*means[:, t]
        next_var = rho*rho*variances[:, t]+innovation
    return U, means, variances, predictions


def ffbs_ar(residual: Array, K: Array, q: float, sigma2: float, rho: float,
            rng: np.random.Generator) -> Array:
    """Exact scalar-eigenmode FFBS for a latent spatial AR(1) field.

    Stationary initial state: w1~N(0,sigma2*q*K). Innovations have covariance
    sigma2*q*(1-rho**2)*K, and measurements have iid variance sigma2. Missing
    observations or local measurement variances are not accepted by this API.
    """
    U, means, variances, predictions = _filter_modes(residual, K, q, sigma2, rho)
    n, T = means.shape
    states = np.empty((n, T))
    states[:, -1] = means[:, -1]+np.sqrt(variances[:, -1])*rng.normal(size=n)
    for t in range(T-2, -1, -1):
        gain = rho*variances[:, t]/predictions[:, t+1]
        variance = variances[:, t]-gain*gain*predictions[:, t+1]
        if np.any(variance < -1e-12*np.maximum(1, variances[:, t])):
            raise linalg.LinAlgError("negative backward sampling variance")
        states[:, t] = (means[:, t]+gain*(states[:, t+1]-rho*means[:, t])
                        +np.sqrt(np.maximum(variance, 0))*rng.normal(size=n))
    return U @ states


def ffbs_ar_moments(residual: Array, K: Array, q: float, sigma2: float, rho: float
                    ) -> tuple[Array, Array]:
    """Exact smoother mean and covariance for small independent verification.

    The full covariance is ordered by column-major vec(W). Its quadratic
    allocation is intentionally gated and is never used by production FFBS.
    """
    U, filtered, variances, predicted = _filter_modes(residual, K, q, sigma2, rho)
    n, T = filtered.shape
    if n*T > 1200:
        raise ValueError("full smoother covariance is limited to n*T<=1200")
    means = filtered.copy()
    covariance_modes = np.zeros((n, T, T))
    covariance_modes[:, -1, -1] = variances[:, -1]
    for t in range(T-2, -1, -1):
        gain = rho*variances[:, t]/predicted[:, t+1]
        means[:, t] += gain*(means[:, t+1]-rho*filtered[:, t])
        covariance_modes[:, t, t] = variances[:, t]+gain**2*(
            covariance_modes[:, t+1, t+1]-predicted[:, t+1])
        covariance_modes[:, t, t+1:] = gain[:, None]*covariance_modes[:, t+1, t+1:]
        covariance_modes[:, t+1:, t] = covariance_modes[:, t, t+1:]
    covariance = np.empty((n*T, n*T))
    for t in range(T):
        for s in range(T):
            covariance[t*n:(t+1)*n, s*n:(s+1)*n] = (U*covariance_modes[:, t, s]) @ U.T
    return U @ means, covariance


def latent_ar_quadratic(W: Array, K: Array, rho: float) -> float:
    """w1'K^-1w1 + sum innovations'K^-1innovations/(1-rho²)."""
    if abs(rho) >= 1:
        return np.inf
    L = _chol(K)
    innovations = np.column_stack((W[:, 0], (W[:, 1:]-rho*W[:, :-1])/np.sqrt(1-rho*rho)))
    return float(np.sum(innovations*_solve_chol(L, innovations)))


def latent_ar_log_density(W: Array, K: Array, q: float, sigma2: float, rho: float) -> float:
    """Complete latent-field density, excluding only the (2pi) constant."""
    n, T = W.shape
    if q <= 0 or sigma2 <= 0 or abs(rho) >= 1:
        return -np.inf
    return float(-n*T/2*np.log(sigma2*q)-T/2*_logdet_chol(_chol(K))
                 -n*(T-1)/2*np.log1p(-rho*rho)
                 -latent_ar_quadratic(W, K, rho)/(2*sigma2*q))


def field_t_weight_parameters(residual: Array, R: Array, sigma2: float,
                              nu: float) -> tuple[float, Array]:
    if nu <= 2 or sigma2 <= 0:
        raise ValueError("nu>2 and sigma2>0 required")
    quadratic = np.sum(residual*_solve_chol(_chol(R), residual), axis=0)
    return (nu+residual.shape[0])/2, (nu+quadratic/sigma2)/2


def local_t_weight_parameters(measurement_residual: Array, sigma2: float,
                              nu: float) -> tuple[float, Array]:
    if nu <= 2 or sigma2 <= 0:
        raise ValueError("nu>2 and sigma2>0 required")
    return (nu+1)/2, (nu+measurement_residual**2/sigma2)/2


def degrees_of_freedom_log_target(eta_nu: float, weights: Array) -> float:
    """Gamma-mixture df conditional with eta=log(nu-2)~N(log(8),1)."""
    if not np.isfinite(eta_nu) or np.any(weights <= 0):
        return -np.inf
    with np.errstate(over="ignore", invalid="ignore"):
        nu = 2+np.exp(eta_nu)
        half = nu/2
        value = (weights.size*(half*np.log(half)-special.gammaln(half))
                 +(half-1)*np.log(weights).sum()-half*weights.sum()
                 -0.5*(eta_nu-np.log(8.0))**2)
    return float(value) if np.isfinite(value) else -np.inf


def field_t_log_likelihood(residual: Array, R: Array, sigma2: float, nu: float) -> float:
    """Normalised independent-time multivariate Student-t likelihood."""
    n, T = residual.shape
    if nu <= 2 or sigma2 <= 0:
        return -np.inf
    L = _chol(R)
    quadratic = np.sum(residual*_solve_chol(L, residual), axis=0)/sigma2
    return float(T*(special.gammaln((nu+n)/2)-special.gammaln(nu/2)
                    -n/2*np.log(nu*np.pi*sigma2)-_logdet_chol(L)/2)
                 -(nu+n)/2*np.log1p(quadratic/nu).sum())


def weighted_precision_operator(Phi: Array, Z: Array, weights: Array,
                                 prior_precision: Any) -> LinearOperator:
    """Exact nonseparable coefficient precision action, with no Kronecker matrix."""
    p, d = Phi.shape[1], Z.shape[1]
    if weights.shape != (Phi.shape[0], Z.shape[0]) or np.any(weights <= 0):
        raise ValueError("weights must be positive and have shape (n,T)")

    def matvec(vector: Array) -> Array:
        M = vector.reshape(p, d, order="F")
        result = Phi.T @ ((Phi @ M @ Z.T)*weights) @ Z
        return prior_precision.apply(vector)+result.reshape(-1, order="F")

    return LinearOperator((p*d, p*d), matvec=matvec, rmatvec=matvec, dtype=float)


def weighted_coefficient_precision(Phi: Array, Z: Array, weights: Array,
                                    prior_precision: Any) -> Array:
    """Dense reference for the sum of time-specific Kronecker precisions."""
    precision = prior_precision.as_matrix()
    for t, z in enumerate(Z):
        gram = Phi.T @ (weights[:, t, None]*Phi)
        precision += np.kron(np.outer(z, z), gram)
    return precision


def draw_weighted_coefficients(Y: Array, Phi: Array, Z: Array, weights: Array,
                               prior_precision: Any, sigma2: float,
                               rng: np.random.Generator, *, backend: str = "dense",
                               rtol: float = 1e-10, atol: float = 0.0,
                               maxiter: int | None = None) -> tuple[Array, dict[str, Any]]:
    """Gaussian conditional under site-time independent measurement weights.

    CG uses an exact precision-noise perturbation before its checked numerical
    solve. A finite-tolerance result is identified as an approximation.
    """
    p, d = Phi.shape[1], Z.shape[1]
    h = (Phi.T @ (weights*Y) @ Z).reshape(-1, order="F")
    if backend == "dense":
        A = weighted_coefficient_precision(Phi, Z, weights, prior_precision)
        L = _chol(A)
        vector = (_solve_chol(L, h)+np.sqrt(sigma2)*
                  linalg.solve_triangular(L.T, rng.normal(size=p*d), lower=False))
        return vector.reshape(p, d, order="F"), {
            "solver": "dense_nonseparable_cholesky", "converged": True,
            "finite_tolerance_approximation": False}
    if backend != "cg":
        raise ValueError("weighted coefficient backend must be dense or cg")
    operator = weighted_precision_operator(Phi, Z, weights, prior_precision)
    noise = prior_precision.sample_precision_noise(rng)
    noise += Phi.T @ (np.sqrt(weights)*rng.normal(size=Y.shape)) @ Z
    rhs = h+np.sqrt(sigma2)*noise.reshape(-1, order="F")
    diagonal = (Phi*Phi).T @ weights @ (Z*Z)
    diagonal = diagonal.reshape(-1, order="F")+prior_precision.diagonal()
    preconditioner = LinearOperator(operator.shape, matvec=lambda x: x/diagonal)
    iterations = [0]

    def callback(_: Array) -> None:
        iterations[0] += 1

    solution, info = cg(operator, rhs, M=preconditioner, rtol=rtol, atol=atol,
                        maxiter=maxiter, callback=callback)
    residual = float(np.linalg.norm(operator @ solution-rhs))
    threshold = max(float(atol), float(rtol)*np.linalg.norm(rhs))
    if info != 0 or not np.isfinite(residual) or residual > threshold*1.05:
        raise RuntimeError(f"weighted CG failed its residual target: info={info}, residual={residual:g}")
    return solution.reshape(p, d, order="F"), {
        "solver": "perturb_and_cg_nonseparable", "converged": True,
        "iterations": iterations[0], "residual_norm": residual,
        "relative_residual": residual/max(np.linalg.norm(rhs), 1e-300),
        "finite_tolerance_approximation": True}


def general_coefficient_statistics(Y: Array, Phi: Array, Z: Array,
                                    row_covariances: Iterable[Array],
                                    prior_precision: Any) -> tuple[Array, Array]:
    """Exact dense precision and information for arbitrary declared R_t.

    The iterator consumes one covariance at a time, keeping spatial-factor
    memory bounded. No common-R Kronecker identity is used for varying R_t.
    """
    A = prior_precision.as_matrix(max_size=900)
    h = np.zeros((Phi.shape[1], Z.shape[1]))
    count = 0
    for t, R in enumerate(row_covariances):
        if t >= len(Z):
            raise ValueError("too many time covariance matrices")
        L = _chol(R)
        A += np.kron(np.outer(Z[t], Z[t]), Phi.T @ _solve_chol(L, Phi))
        h += np.outer(Phi.T @ _solve_chol(L, Y[:, t]), Z[t])
        count += 1
    if count != len(Z):
        raise ValueError("one row covariance matrix is required per time")
    return A, h.reshape(-1, order="F")


def _draw_general(Y: Array, Phi: Array, Z: Array, covariances: Iterable[Array],
                  precision: Any, sigma2: float, rng: np.random.Generator) -> tuple[Array, dict]:
    A, h = general_coefficient_statistics(Y, Phi, Z, covariances, precision)
    L = _chol(A)
    b = _solve_chol(L, h)+np.sqrt(sigma2)*linalg.solve_triangular(
        L.T, rng.normal(size=precision.size), lower=False, check_finite=False)
    return b.reshape(Phi.shape[1], Z.shape[1], order="F"), {
        "solver": "dense_general_covariance_cholesky", "converged": True,
        "finite_tolerance_approximation": False}


def draw_local_fields(residual: Array, K: Array, q: float, sigma2: float,
                       weights: Array, rng: np.random.Generator) -> Array:
    """Local-t Gaussian field conditional, without storing T factorizations."""
    Ki = _solve_chol(_chol(K), np.eye(len(K)))/q
    W = np.empty_like(residual)
    for t in range(residual.shape[1]):
        precision = Ki+np.diag(weights[:, t])
        L = _chol(precision)
        W[:, t] = (_solve_chol(L, weights[:, t]*residual[:, t])
                    +np.sqrt(sigma2)*linalg.solve_triangular(
                        L.T, rng.normal(size=len(K)), lower=False, check_finite=False))
    return W


def sample_augmented(Y: Array, Phi: Array, Z: Array, coordinates: Array,
                      groups: Any, prior: Any, covariance: Any, config: Any,
                      kind: str = "latent_ar", known_variance_covariates: Array | None = None,
                      *, variance_prior_sd: float = 0.3, reference_Z: Array | None = None,
                      checkpoint_callback: Any = None, resume_state: Any = None
                      ) -> Any:
    """Run complete ordinary Gibbs/MH samplers for four supported extensions.

    Supported: ``latent_ar`` (iid nugget), ``field_t`` (one whole-field weight
    per time), ``local_t`` (iid spatial fields plus local nugget weights), and
    ``heteroskedastic`` (known predictable site-time variance covariates).
    Unsupported compositions fail before sampling. All use a shared scale IG
    prior; the separate-physical-scale PC variant uses the Gaussian sampler.

    A ``reference_r2`` readout prior refers to trace(V+q*K)/n in these models:
    the baseline Gaussian scale at zero variance covariates. It does not use
    Student-t marginal variance inflation or time-average heteroskedastic
    variance, and is not a literal prior on realised response R².

    Reference limits are n<=100, T<=2000. General heteroskedasticity additionally
    requires p*d<=900, since it uses an explicit coefficient precision. Local-t
    coefficients support checked matrix-free CG. All spatial matrices remain
    dense in these augmented reference samplers, including approximate K from
    Vecchia/FEM. Their sparse operator APIs are a separate scaling facility.
    """
    from .gaussian import draw_coefficients
    from .priors import PriorRuntime
    from .sampling import RegressionPosterior, _resolve_resume, _signature

    if kind not in {"latent_ar", "field_t", "local_t", "heteroskedastic"}:
        raise ValueError("unknown augmented model kind")
    Y, Phi, Z, coordinates = (np.asarray(a, dtype=float) for a in (Y, Phi, Z, coordinates))
    if any(a.ndim != 2 or not np.isfinite(a).all() for a in (Y, Phi, Z, coordinates)):
        raise ValueError("all input matrices must be complete and finite")
    n, T = Y.shape
    p, d = Phi.shape[1], Z.shape[1]
    N, k = n*T, p*d
    if Phi.shape[0] != n or len(Z) != T or len(coordinates) != n or min(n, T, p, d) == 0:
        raise ValueError("incompatible augmented regression dimensions")
    if n > 100 or T > 2000 or (kind == "heteroskedastic" and k > 900):
        raise ValueError("experimental augmented sampler gate: n<=100, T<=2000; heteroskedastic p*d<=900")
    if kind == "heteroskedastic" and 4*8*k*k > config.max_dense_bytes:
        raise MemoryError("general heteroskedastic coefficient precision exceeds configured dense memory gate")
    if covariance.scale_mode != "relative":
        raise ValueError("augmented models require the common IG scale; separate scales use Gaussian inference")
    if kind in {"latent_ar", "local_t"} and covariance.static_variance:
        raise ValueError("latent_ar/local_t routes implement an iid reference nugget; static variance is unsupported")
    if kind in {"latent_ar", "local_t"} and covariance.kind == "spde":
        # These observed-site state samplers use K^-1 and log|K|. A proper
        # mesh precision alone does not make A Q^-1 A' nonsingular when the
        # mesh/interpolation has fewer independent modes than observations.
        operator = covariance.operator(coordinates, covariance.initial_parameters(coordinates))
        interpolation_gram = (operator.interpolation @ operator.interpolation.T).toarray()
        if np.linalg.matrix_rank(interpolation_gram) < n:
            raise ValueError("latent_ar/local_t require full-row-rank SPDE interpolation; refine the mesh or use the Gaussian residual model")
    if getattr(config, "temporal", "iid") != "iid":
        raise ValueError("full-residual AR cannot be combined with an augmented model in this backend")
    if getattr(config, "algorithm", "auto") == "collapsed":
        raise ValueError("augmented models require ordinary conditional Gibbs/MH, not collapsed inference")
    if config.backend == "temporal_whitened_spectral":
        raise ValueError("temporal_whitened_spectral is only a full-residual-AR backend")
    if config.backend == "latent_ffbs" and kind != "latent_ar":
        raise ValueError("latent_ffbs requires the latent_ar model")
    if config.backend == "weighted_spectral" and kind != "field_t":
        raise ValueError("weighted_spectral requires the whole-field-t model")
    if kind == "heteroskedastic" and config.backend not in {"auto", "dense"}:
        raise ValueError("general heteroskedasticity currently requires auto or dense coefficient backend")
    if kind == "local_t" and config.backend == "spectral_gaussian":
        raise ValueError("local-t site-time weights do not have a separable spectral precision")
    if kind == "local_t" and config.backend == "dense" and (
            k > config.dense_threshold or 4*8*k*k > config.max_dense_bytes):
        raise MemoryError("explicit dense local-t readout exceeds configured dense gate; request matrixfree")
    if not np.isfinite(variance_prior_sd) or variance_prior_sd <= 0:
        raise ValueError("variance_prior_sd must be positive")
    F = None
    variance_center = variance_scale = None
    if kind == "heteroskedastic":
        if known_variance_covariates is None:
            raise ValueError("heteroskedastic model requires known predictable variance covariates")
        F = np.asarray(known_variance_covariates, dtype=float)
        if F.ndim == 2 and F.shape[0] == T:
            F = np.broadcast_to(F[None, :, :], (n, T, F.shape[1])).copy()
        if F.ndim != 3 or F.shape[:2] != (n, T) or not 1 <= F.shape[2] <= 12 or not np.isfinite(F).all():
            raise ValueError("variance covariates must have shape (n,T,h) or (T,h), 1<=h<=12")
        variance_center = F.mean(axis=(0, 1))
        variance_scale = np.sqrt(np.mean((F-variance_center)**2, axis=(0, 1)))
        if np.any(variance_scale <= 1e-10):
            raise ValueError("variance covariates cannot contain an intercept or constant column")
        F = (F-variance_center)/variance_scale
    elif known_variance_covariates is not None:
        raise ValueError("variance covariates are only used by kind='heteroskedastic'")

    runtime = PriorRuntime(prior, p, d, groups, Phi, Z if reference_Z is None else reference_Z)
    if config.backend in {"weighted_spectral", "spectral_gaussian"} and not runtime.separable:
        raise ValueError("the requested spectral readout requires a column-separable prior")
    estimated_state_bytes = 8*config.chains*config.draws*(k+runtime.dimension+n+64)
    if estimated_state_bytes > config.max_stored_hyperstate_bytes:
        raise MemoryError("augmented retained states exceed configured storage budget")
    signature_hash = hashlib.sha256(_signature(Y, Phi, Z, coordinates, groups, prior,
        covariance, config, reference_Z).encode())
    signature_hash.update(f"{kind}:{variance_prior_sd:.17g}".encode())
    if known_variance_covariates is not None:
        raw_covariates = np.ascontiguousarray(known_variance_covariates, dtype=float)
        signature_hash.update(str(raw_covariates.shape).encode())
        signature_hash.update(memoryview(raw_covariates).cast("B"))
    signature = signature_hash.hexdigest()
    seed_sequences = np.random.SeedSequence(config.seed).spawn(config.chains)
    all_states: list[list[dict[str, Any]]] = []
    log_density = np.empty((config.chains, config.draws))
    acceptance = np.empty(config.chains)
    chain_metadata = []
    started = time.perf_counter()

    for chain, seed in enumerate(seed_sequences):
        rng = np.random.default_rng(seed)
        params = covariance.initial_parameters(coordinates)
        vector = covariance.parameters_to_vector(params, coordinates)
        if config.initial_jitter:
            vector += config.initial_jitter*rng.normal(size=len(vector))
            params = covariance.vector_to_parameters(vector, coordinates, nu=params["nu"])
        prior_state = runtime.initial_state()
        C = np.zeros((p, d))
        sigma2 = max(float(np.var(Y)), 0.01)
        W = np.zeros_like(Y)
        weights = np.ones(T if kind == "field_t" else (n, T))
        eta_rho = 0.0
        eta_nu = float(np.log(8.0))
        eta_variance = np.zeros(F.shape[2]) if F is not None else None
        step = float(config.proposal_scale)
        accepted = window_accepted = 0
        total = config.warmup+config.draws*config.thin
        retained: list[dict[str, Any]] = []
        solver_metadata: dict[str, Any] = {}
        start_iteration = 0
        restored = _resolve_resume(resume_state, chain)
        if restored is not None:
            if (restored.get("schema_version") != 1 or restored.get("signature") != signature
                    or restored.get("chain_index") != chain
                    or restored.get("schedule") != f"augmented:{kind}"):
                raise ValueError("augmented checkpoint schema, data, chain, model or transition settings do not match")
            rng.bit_generator.state = copy.deepcopy(restored["rng_state"])
            vector = np.asarray(restored["vector"], dtype=float).copy()
            params = covariance.vector_to_parameters(vector, coordinates, nu=restored["nu"])
            prior_state = runtime.state_from_vector(runtime.vector(restored["prior_state"]))
            C, W, weights = (np.asarray(restored[key], dtype=float).copy() for key in ("C", "W", "weights"))
            sigma2 = float(restored["sigma2"])
            eta_rho, eta_nu = float(restored["eta_rho"]), float(restored["eta_nu"])
            eta_variance = (None if restored["eta_variance"] is None
                            else np.asarray(restored["eta_variance"], dtype=float).copy())
            step = float(restored["proposal_scale"])
            accepted, window_accepted = int(restored["accepted_total"]), int(restored["window_accepted"])
            start_iteration = int(restored["iteration"])
            retained = copy.deepcopy(restored["retained_states"])
            if start_iteration > total or len(retained) > config.draws:
                raise ValueError("augmented checkpoint exceeds the requested target iterations")
            log_density[chain, :len(retained)] = restored["retained_log_density"]
            solver_metadata = copy.deepcopy(restored["solver_metadata"])

        def checkpoint(iteration: int) -> dict[str, Any]:
            return {"schema_version": 1, "signature": signature, "chain_index": chain,
                    "schedule": f"augmented:{kind}", "iteration": iteration,
                    "rng_state": copy.deepcopy(rng.bit_generator.state),
                    "vector": vector.copy(), "nu": float(params["nu"]),
                    "prior_state": copy.deepcopy(prior_state), "C": C.copy(), "W": W.copy(),
                    "weights": weights.copy(), "sigma2": sigma2,
                    "eta_rho": eta_rho, "eta_nu": eta_nu,
                    "eta_variance": None if eta_variance is None else eta_variance.copy(),
                    "proposal_scale": step, "accepted_total": accepted,
                    "window_accepted": window_accepted,
                    "retained_states": copy.deepcopy(retained),
                    "retained_log_density": log_density[chain, :len(retained)].tolist(),
                    "solver_metadata": copy.deepcopy(solver_metadata)}

        def matrices(candidate: dict[str, Any]) -> tuple[Array, Array, Array, float]:
            K, V = covariance.matrices(coordinates, candidate)
            q = float(np.exp(candidate["log_q"]))
            return K, V, V+q*K, q

        def hetero_covariances(K: Array, V: Array, q: float, eta: Array) -> Iterable[Array]:
            log_variance = np.einsum("nth,h->nt", F, eta)
            with np.errstate(over="raise", invalid="raise"):
                variance = np.diag(V)[:, None]*np.exp(log_variance)
            for t in range(T):
                yield np.diag(variance[:, t])+q*K

        def hetero_density(residual: Array, K: Array, V: Array, q: float,
                           eta: Array) -> tuple[float, float]:
            ld, quadratic = 0.0, 0.0
            for t, Rt in enumerate(hetero_covariances(K, V, q, eta)):
                L = _chol(Rt)
                ld += _logdet_chol(L)
                quadratic += float(residual[:, t] @ _solve_chol(L, residual[:, t]))
            return ld, quadratic

        def coefficient_prior(candidate_R: Array) -> float:
            precision = runtime.precision(prior_state, candidate_R)
            return precision.logdet/2-precision.quadratic(C)/(2*sigma2)

        def covariance_target(candidate: dict[str, Any]) -> float:
            value = covariance.log_prior(candidate, coordinates)
            if not np.isfinite(value):
                return -np.inf
            try:
                KK, VV, RR, qq = matrices(candidate)
                residual = Y-Phi @ C @ Z.T
                if kind == "latent_ar":
                    value += latent_ar_log_density(W, KK, qq, sigma2, np.tanh(eta_rho))
                elif kind == "local_t":
                    L = _chol(KK)
                    value += (-N/2*np.log(qq)-T/2*_logdet_chol(L)
                              -np.sum(W*_solve_chol(L, W))/(2*sigma2*qq))
                elif kind == "field_t":
                    L = _chol(RR)
                    quadratic = np.sum(residual*_solve_chol(L, residual), axis=0)
                    value += -T/2*_logdet_chol(L)-float(weights @ quadratic)/(2*sigma2)
                else:
                    ld, quadratic = hetero_density(residual, KK, VV, qq, eta_variance)
                    value += -ld/2-quadratic/(2*sigma2)
                value += coefficient_prior(RR)
                return float(value) if np.isfinite(value) else -np.inf
            except (linalg.LinAlgError, ValueError, FloatingPointError, OverflowError):
                return -np.inf

        for iteration in range(start_iteration, total):
            K, V, R, q = matrices(params)
            precision = runtime.precision(prior_state, R)
            residual = Y-Phi @ C @ Z.T
            nu = 2+np.exp(eta_nu)
            if kind == "latent_ar":
                W = ffbs_ar(residual, K, q, sigma2, np.tanh(eta_rho), rng)
            elif kind == "local_t":
                W = draw_local_fields(residual, K, q, sigma2, weights, rng)
                shape, rate = local_t_weight_parameters(residual-W, sigma2, nu)
                weights = rng.gamma(shape, 1/rate)
            elif kind == "field_t":
                shape, rate = field_t_weight_parameters(residual, R, sigma2, nu)
                weights = rng.gamma(shape, 1/rate)

            common_backend = ("auto" if config.backend in {"latent_ffbs", "weighted_spectral"}
                              else config.backend)
            common_kwargs = dict(backend=common_backend, rtol=config.solver_rtol,
                                 atol=config.solver_atol, maxiter=config.solver_maxiter,
                                 dense_threshold=config.dense_threshold,
                                 max_dense_bytes=config.max_dense_bytes)
            if kind == "latent_ar":
                C, solver_metadata = draw_coefficients(Y-W, Phi, Z, np.eye(n), precision,
                                                        sigma2, rng, **common_kwargs)
            elif kind == "field_t":
                root_weight = np.sqrt(weights)
                C, solver_metadata = draw_coefficients(Y*root_weight, Phi, Z*root_weight[:, None],
                                                        R, precision, sigma2, rng, **common_kwargs)
            elif kind == "local_t":
                dense_ok = k <= config.dense_threshold and 4*8*k*k <= config.max_dense_bytes
                backend = "cg" if config.backend in {"cg", "matrixfree", "gaussian_matrixfree"} or not dense_ok else "dense"
                C, solver_metadata = draw_weighted_coefficients(
                    Y-W, Phi, Z, weights, precision, sigma2, rng, backend=backend,
                    rtol=config.solver_rtol, atol=config.solver_atol,
                    maxiter=config.solver_maxiter)
            else:
                C, solver_metadata = _draw_general(Y, Phi, Z,
                    hetero_covariances(K, V, q, eta_variance), precision, sigma2, rng)

            residual = Y-Phi @ C @ Z.T
            penalty = precision.quadratic(C)
            if kind == "latent_ar":
                scale_quadratic = np.sum((residual-W)**2)+latent_ar_quadratic(W, K, np.tanh(eta_rho))/q
                shape = config.a0+(2*N+k)/2
            elif kind == "local_t":
                scale_quadratic = (np.sum(weights*(residual-W)**2)
                                   +np.sum(W*_solve_chol(_chol(K), W))/q)
                shape = config.a0+(2*N+k)/2
            elif kind == "field_t":
                scale_quadratic = weights @ np.sum(residual*_solve_chol(_chol(R), residual), axis=0)
                shape = config.a0+(N+k)/2
            else:
                _, scale_quadratic = hetero_density(residual, K, V, q, eta_variance)
                shape = config.a0+(N+k)/2
            sigma2 = float((config.b0+(scale_quadratic+penalty)/2)/rng.gamma(shape))
            prior_state = runtime.update_conditional(C, sigma2, prior_state, rng, R,
                rhs_tradeoff_scale=getattr(config, "rhs_tradeoff_scale", 0.5))

            current = covariance_target(params)
            if not np.isfinite(current):
                raise FloatingPointError("current augmented covariance conditional has nonfinite density")
            proposal_vector = vector+step*rng.normal(size=len(vector))/np.sqrt(len(vector))
            proposed = covariance.vector_to_parameters(proposal_vector, coordinates, nu=params["nu"])
            proposal_target = covariance_target(proposed)
            if np.log(rng.uniform()) < proposal_target-current:
                params, vector, current = proposed, proposal_vector, proposal_target
                accepted += 1
                window_accepted += 1
            if covariance.infer_nu:
                choices = [dict(params, nu=float(v)) for v in covariance.smoothness_choices]
                scores = np.asarray([covariance_target(candidate) for candidate in choices])
                if not np.isfinite(scores).any():
                    raise FloatingPointError("all augmented smoothness candidates have zero probability")
                probabilities = np.exp(scores-special.logsumexp(scores))
                params = choices[rng.choice(len(choices), p=probabilities)]
            K, V, R, q = matrices(params)

            if kind == "latent_ar":
                candidate_eta = eta_rho+0.12*rng.normal()
                current_rho_target = (latent_ar_log_density(W, K, q, sigma2, np.tanh(eta_rho))
                                      -0.5*(eta_rho/config.rho_prior_sd)**2)
                candidate_target = (latent_ar_log_density(W, K, q, sigma2, np.tanh(candidate_eta))
                                    -0.5*(candidate_eta/config.rho_prior_sd)**2)
                if np.log(rng.uniform()) < candidate_target-current_rho_target:
                    eta_rho = candidate_eta
            elif kind in {"local_t", "field_t"}:
                candidate_eta = eta_nu+0.15*rng.normal()
                if np.log(rng.uniform()) < (degrees_of_freedom_log_target(candidate_eta, weights)
                                           -degrees_of_freedom_log_target(eta_nu, weights)):
                    eta_nu = candidate_eta
            else:
                proposal_eta = eta_variance+0.06*rng.normal(size=len(eta_variance))
                ld0, quad0 = hetero_density(residual, K, V, q, eta_variance)
                try:
                    ld1, quad1 = hetero_density(residual, K, V, q, proposal_eta)
                    log_ratio = (-(ld1-ld0)/2-(quad1-quad0)/(2*sigma2)
                                 -(proposal_eta @ proposal_eta-eta_variance @ eta_variance)/(2*variance_prior_sd**2))
                except (linalg.LinAlgError, FloatingPointError, OverflowError):
                    log_ratio = -np.inf
                if np.log(rng.uniform()) < log_ratio:
                    eta_variance = proposal_eta

            if iteration < config.warmup and (iteration+1) % config.adapt_interval == 0:
                rate = window_accepted/config.adapt_interval
                step *= np.exp(np.clip(rate-config.target_accept, -0.5, 0.5))
                step = float(np.clip(step, 1e-5, 3.0))
                window_accepted = 0

            if iteration >= config.warmup and (iteration-config.warmup+1) % config.thin == 0:
                temporal: dict[str, Any] = {"kind": kind}
                if kind == "latent_ar":
                    temporal.update(rho=float(np.tanh(eta_rho)), eta_rho=float(eta_rho))
                elif kind in {"field_t", "local_t"}:
                    temporal.update(nu=float(2+np.exp(eta_nu)), eta_nu=float(eta_nu))
                else:
                    temporal.update(variance_eta=eta_variance.copy(),
                                    variance_covariate_center=variance_center.copy(),
                                    variance_covariate_scale=variance_scale.copy())
                state = {"C": C.copy(), "sigma2": sigma2, "prior_state": copy.deepcopy(prior_state),
                         "covariance_parameters": copy.deepcopy(params), "temporal_parameters": temporal,
                         "solver_metadata": dict(solver_metadata)}
                if kind == "latent_ar":
                    state["latent_state"] = W[:, -1].copy()
                # Complete augmented posterior density up to parameter-independent constants.
                density = covariance_target(params)+runtime.log_prior(prior_state)
                density -= (config.a0+1+k/2)*np.log(sigma2)+config.b0/sigma2
                if kind == "latent_ar":
                    density += -N/2*np.log(sigma2)-np.sum((residual-W)**2)/(2*sigma2)
                    density -= 0.5*(eta_rho/config.rho_prior_sd)**2
                elif kind == "local_t":
                    density += (-N*np.log(sigma2)+0.5*np.log(weights).sum()
                                -np.sum(weights*(residual-W)**2)/(2*sigma2))
                    density += degrees_of_freedom_log_target(eta_nu, weights)
                elif kind == "field_t":
                    density += -N/2*np.log(sigma2)+n/2*np.log(weights).sum()
                    density += degrees_of_freedom_log_target(eta_nu, weights)
                else:
                    density += -N/2*np.log(sigma2)-np.sum(eta_variance**2)/(2*variance_prior_sd**2)
                log_density[chain, len(retained)] = density
                retained.append(state)

            if checkpoint_callback is not None and ((config.checkpoint_every and
                    (iteration+1) % config.checkpoint_every == 0) or iteration+1 == total):
                checkpoint_callback(chain, checkpoint(iteration+1))

        all_states.append(retained)
        acceptance[chain] = accepted/total
        chain_metadata.append({"final_covariance_proposal_scale": step,
                               "covariance_acceptance": float(acceptance[chain]),
                               "iterations": total, "last_solver": solver_metadata})

    return RegressionPosterior(
        Y=Y, Phi=Phi, Z=Z, coordinates=coordinates, groups=groups, prior=prior,
        covariance=covariance, config=config, states=all_states,
        log_density=log_density, acceptance=acceptance,
        metadata={"algorithm": "ordinary_gibbs_mh", "augmentation_kind": kind,
                  "experimental": True, "spatial_backend_materialized_dense": True,
                  "n_sites": n, "n_times": T, "coefficient_dimension": k,
                  "elapsed_seconds": time.perf_counter()-started,
                  "chain_details": chain_metadata,
                  "checkpoint_schema_version": 1, "data_signature": signature,
                  "log_density_kind": "complete_augmented_posterior_up_to_constants",
                  "variance_prior_sd": variance_prior_sd,
                  "scale_shape": "a0+(2N+k)/2" if kind in {"latent_ar", "local_t"} else "a0+(N+k)/2"},
        reference_Z=reference_Z)
