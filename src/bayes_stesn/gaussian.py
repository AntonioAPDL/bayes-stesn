"""Gaussian tensor regression without constructing its Kronecker design.

The public data convention in this module is ``Y[n, T], Phi[n, p], Z[T, d]``.
Coefficients are vectorized in column-major order.  All coefficient priors are
``vec(C) | sigma2 ~ N(0, sigma2 * precision**-1)``.  The inverse-gamma uses shape
and scale: its density is proportional to ``x**(-a-1) * exp(-b/x)``.

Separable precision is integrated exactly, including normalizing constants.
General precision uses direct Gaussian sampling on small problems or a checked
perturb-and-CG solve.  A finite-tolerance CG draw is explicitly identified as a
numerical approximation; a failed solve is never silently returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from scipy import linalg, special
from scipy.sparse.linalg import LinearOperator, cg


def _readonly(x: Any) -> np.ndarray:
    out = np.array(x, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out


def psd_eigh(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalue root of a Gram matrix, correcting roundoff only."""
    values, vectors = linalg.eigh((x + x.T) * 0.5, check_finite=False)
    tol = 100 * np.finfo(float).eps * max(x.shape) * max(1., abs(values).max())
    if values[0] < -tol:
        raise linalg.LinAlgError("A purported Gram matrix has a negative eigenvalue")
    return np.maximum(values, 0.), vectors


@dataclass(frozen=True)
class PrecisionSpec:
    """Proper coefficient precision with a bounded dense conversion.

    ``kind`` is scalar, column, diagonal, or dense.  Diagonal vectors use
    column-major coefficient ordering.  Dense conversion is intentionally
    refused above ``max_size``; matrix-free consumers use ``apply`` instead.
    """

    p: int
    d: int
    kind: str
    values: Any

    def __post_init__(self) -> None:
        if self.p < 1 or self.d < 1:
            raise ValueError("Precision dimensions must be positive")
        expected = {"scalar": (), "column": (self.d,),
                    "diagonal": (self.size,), "dense": (self.size, self.size)}
        if self.kind not in expected:
            raise ValueError(f"Unknown precision kind {self.kind!r}")
        value = _readonly(self.values)
        if value.shape != expected[self.kind] or not np.isfinite(value).all():
            raise ValueError("Precision has invalid dimensions or nonfinite entries")
        if self.kind == "dense":
            if not np.allclose(value, value.T, rtol=1e-12, atol=1e-13):
                raise ValueError("A dense precision must be symmetric")
            linalg.cholesky(value, lower=True, check_finite=False)
        elif np.any(value <= 0):
            raise ValueError("Every coefficient precision must be strictly positive")
        object.__setattr__(self, "values", value)

    @property
    def size(self) -> int:
        return self.p * self.d

    @property
    def separable(self) -> bool:
        return self.kind in ("scalar", "column")

    @property
    def column_values(self) -> np.ndarray:
        if self.kind == "scalar":
            return np.full(self.d, float(self.values))
        if self.kind == "column":
            return np.array(self.values)
        raise ValueError("This precision is not constant across spatial rows")

    def diagonal(self) -> np.ndarray:
        if self.kind == "scalar":
            return np.full(self.size, float(self.values))
        if self.kind == "column":
            return np.repeat(self.values, self.p)
        if self.kind == "diagonal":
            return np.array(self.values)
        return np.diag(self.values).copy()

    @cached_property
    def logdet(self) -> float:
        if self.kind == "scalar":
            return self.size * float(np.log(self.values))
        if self.kind == "column":
            return self.p * float(np.log(self.values).sum())
        if self.kind == "diagonal":
            return float(np.log(self.values).sum())
        return 2 * float(np.log(np.diag(self._cholesky)).sum())

    @cached_property
    def _cholesky(self) -> np.ndarray:
        if self.kind != "dense":
            raise ValueError("A Cholesky matrix is only needed for dense precision")
        return linalg.cholesky(self.values, lower=True, check_finite=False)

    def as_matrix(self, max_size: int = 2000) -> np.ndarray:
        if self.size > max_size:
            raise MemoryError(f"Dense precision size {self.size} exceeds cap {max_size}")
        return np.array(self.values) if self.kind == "dense" else np.diag(self.diagonal())

    def apply(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.size,):
            raise ValueError("Precision.apply expects a column-major coefficient vector")
        if self.kind == "dense":
            return self.values @ vector
        return self.diagonal() * vector

    def quadratic(self, C: np.ndarray) -> float:
        vector = np.asarray(C).reshape(-1, order="F")
        return float(vector @ self.apply(vector))

    def sample_precision_noise(self, rng: np.random.Generator) -> np.ndarray:
        white = rng.normal(size=self.size)
        noise = self._cholesky @ white if self.kind == "dense" else np.sqrt(self.diagonal()) * white
        return noise.reshape(self.p, self.d, order="F")


@dataclass
class GaussianEvaluation:
    """Normalized coefficient/variance integrated likelihood and recovery factors."""

    log_marginal: float
    an: float
    bn: float
    Q: float
    logdet_R: float
    logdet_temporal: float
    precision: PrecisionSpec
    spatial_vectors: np.ndarray | None = None
    temporal_vectors: np.ndarray | None = None
    eigenmean: np.ndarray | None = None
    eigenprecision: np.ndarray | None = None
    column_unscale: np.ndarray | None = None
    dense_mean: np.ndarray | None = None
    dense_cholesky: np.ndarray | None = None
    fixed_sigma2: float | None = None

    @property
    def mean(self) -> np.ndarray:
        if self.dense_mean is not None:
            return self.dense_mean.reshape(self.precision.p, self.precision.d, order="F")
        return (self.spatial_vectors @ self.eigenmean @ self.temporal_vectors.T) * self.column_unscale

    def draw(self, rng: np.random.Generator, sigma2: float | None = None
             ) -> tuple[np.ndarray, float, dict]:
        scale = self.fixed_sigma2 if sigma2 is None else sigma2
        if scale is None:
            scale = self.bn / rng.gamma(self.an)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("sigma2 must be positive and finite")
        if self.dense_mean is not None:
            white = rng.normal(size=self.precision.size)
            vector = self.dense_mean + np.sqrt(scale) * linalg.solve_triangular(
                self.dense_cholesky.T, white, lower=False, check_finite=False)
            C = vector.reshape(self.precision.p, self.precision.d, order="F")
            solver = "dense_cholesky"
        else:
            U = self.eigenmean + np.sqrt(scale / self.eigenprecision) * rng.normal(size=self.eigenmean.shape)
            C = (self.spatial_vectors @ U @ self.temporal_vectors.T) * self.column_unscale
            solver = "spectral"
        return C, float(scale), {"solver": solver, "converged": True,
                                 "finite_tolerance_approximation": False}


class GaussianRegression:
    """Cached exact Gaussian algebra for a fixed causal design.

    Scalar ridge reuses the eigensystem of ``Z.T @ Z`` throughout a chain.
    General column precisions refresh only the prior-whitened temporal factor.
    Spatial and temporal caches each retain only the current factorization.
    AR(1) likelihoods use cached lagged cross-products, so proposals do not
    repeatedly transform the full response history.
    """

    def __init__(self, Y: np.ndarray, Phi: np.ndarray, Z: np.ndarray,
                 a0: float = 2., b0: float = .05, dense_threshold: int = 900,
                 max_dense_bytes: int = 256 * 1024**2) -> None:
        self.Y, self.Phi, self.Z = (np.asarray(x, dtype=float) for x in (Y, Phi, Z))
        if any(x.ndim != 2 or not np.isfinite(x).all() for x in (self.Y, self.Phi, self.Z)):
            raise ValueError("Y, Phi and Z must be finite matrices")
        self.n, self.T = self.Y.shape
        if min(self.n, self.T) < 1 or self.Phi.shape[0] != self.n or self.Z.shape[0] != self.T:
            raise ValueError("Response and design dimensions do not agree")
        self.p, self.d = self.Phi.shape[1], self.Z.shape[1]
        if min(self.p, self.d) < 1:
            raise ValueError("Both designs must have at least one column")
        if not np.isfinite(a0 + b0) or min(a0, b0) <= 0:
            raise ValueError("The inverse-gamma shape and scale must be positive")
        self.N, self.k = self.n * self.T, self.p * self.d
        self.a0, self.b0 = float(a0), float(b0)
        self.an = self.a0 + self.N / 2
        self.dense_threshold, self.max_dense_bytes = dense_threshold, max_dense_bytes
        self.G = self.Z.T @ self.Z
        self.YZ = self.Y @ self.Z
        self._g, self._Ug = psd_eigh(self.G)
        # Scalar-ridge proposals reuse this rotation as well as the Gram
        # eigensystem: no O(p*d^2) temporal rotation is repeated per proposal.
        self._YZ_eig = self.YZ @ self._Ug
        self._spatial_cache = None
        self._temporal_cache = None
        self._ar_cache = None
        self._stats_cache = None

    @cached_property
    def YY(self) -> np.ndarray:
        # Dense reference likelihoods use this cache. Operator likelihoods
        # deliberately never access or allocate an n-by-n response product.
        return self.Y @ self.Y.T

    def _check_precision(self, precision: PrecisionSpec) -> None:
        if (precision.p, precision.d) != (self.p, self.d):
            raise ValueError("Coefficient precision and design dimensions disagree")

    def _dense_allowed(self) -> bool:
        # Precision, Cholesky, and work copies, not merely one k-by-k array.
        return self.k <= self.dense_threshold and 4 * 8 * self.k**2 <= self.max_dense_bytes

    def statistics(self, temporal_rho: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        rho = 0. if temporal_rho is None else float(temporal_rho)
        if not np.isfinite(rho) or abs(rho) >= 1:
            raise ValueError("Stationary residual AR requires |rho| < 1")
        if rho == 0:
            return self.G, self.YY, self.YZ, 0.
        if self._stats_cache is not None and self._stats_cache[0] == rho:
            return self._stats_cache[1]
        if self._ar_cache is None:
            yc, yp, zc, zp = self.Y[:, 1:], self.Y[:, :-1], self.Z[1:], self.Z[:-1]
            y0, z0 = self.Y[:, 0], self.Z[0]
            self._ar_cache = (
                np.outer(y0, y0), yc @ yc.T, yp @ yp.T, yc @ yp.T + yp @ yc.T,
                np.outer(z0, z0), zc.T @ zc, zp.T @ zp, zc.T @ zp + zp.T @ zc,
                np.outer(y0, z0), yc @ zc, yp @ zp, yc @ zp + yp @ zc)
        yy0, yyc, yyp, yyl, g0, gc, gp, gl, j0, jc, jp, jl = self._ar_cache
        den = 1 - rho**2
        stats = (g0 + (gc + rho**2 * gp - rho * gl) / den,
                 yy0 + (yyc + rho**2 * yyp - rho * yyl) / den,
                 j0 + (jc + rho**2 * jp - rho * jl) / den,
                 (self.T - 1) * float(np.log1p(-rho**2)))
        self._stats_cache = (rho, stats)
        return stats

    def spatial_statistics(self, R: np.ndarray) -> tuple:
        R = np.asarray(R, dtype=float)
        if R.shape != (self.n, self.n) or not np.isfinite(R).all():
            raise ValueError("R must be a finite n-by-n covariance")
        if not np.allclose(R, R.T, rtol=1e-11, atol=1e-13):
            raise ValueError("R must be symmetric")
        if self._spatial_cache is not None and np.array_equal(R, self._spatial_cache[0]):
            return self._spatial_cache[1]
        L = linalg.cholesky(R, lower=True, check_finite=False)
        RiPhi = linalg.cho_solve((L, True), self.Phi, check_finite=False)
        B = self.Phi.T @ RiPhi
        beta, U = psd_eigh(B)
        # This small n-by-n operator is shared across cached cross-products;
        # the enormous coefficient precision/design is never inverted.
        Ri = linalg.cho_solve((L, True), np.eye(self.n), check_finite=False)
        ans = L, RiPhi, B, beta, U, Ri, 2 * float(np.log(np.diag(L)).sum())
        self._spatial_cache = (R.copy(), ans)
        return ans

    def _temporal_factor(self, G: np.ndarray, columns: np.ndarray,
                         rho: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        scalar = bool(np.all(columns == columns[0]))
        keyrho = 0. if rho is None else float(rho)
        if scalar:
            if keyrho == 0:
                g, V = self._g, self._Ug
            else:
                key = ("scalar", keyrho)
                if self._temporal_cache is not None and self._temporal_cache[0] == key:
                    g, V = self._temporal_cache[2:4]
                else:
                    g, V = psd_eigh(G)
                    self._temporal_cache = (key, None, g, V)
            return g, V, np.ones(self.d), float(columns[0])
        key = ("column", keyrho)
        unscale = 1 / np.sqrt(columns)
        if (self._temporal_cache is not None and self._temporal_cache[0] == key
                and np.array_equal(self._temporal_cache[1], columns)):
            g, V = self._temporal_cache[2:4]
        else:
            g, V = psd_eigh(unscale[:, None] * G * unscale[None, :])
            self._temporal_cache = (key, columns.copy(), g, V)
        return g, V, unscale, 1.

    def residual_quadratic(self, C: np.ndarray, R: np.ndarray,
                           temporal_rho: float | None = None,
                           chunk_size: int = 512) -> float:
        """Cancellation-safe residual norm with bounded history work arrays."""
        L = self.spatial_statistics(R)[0]
        rho = 0. if temporal_rho is None else float(temporal_rho)
        if abs(rho) >= 1:
            raise ValueError("|rho| must be less than one")
        total = 0.
        PC = self.Phi @ C
        for start in range(0, self.T, chunk_size):
            end = min(self.T, start + chunk_size)
            residual = self.Y[:, start:end] - PC @ self.Z[start:end].T
            if rho:
                if start == 0:
                    residual[:, 1:] = (residual[:, 1:] - rho * residual[:, :-1]) / np.sqrt(1-rho**2)
                else:
                    previous = self.Y[:, start-1:end-1] - PC @ self.Z[start-1:end-1].T
                    residual = (residual - rho * previous) / np.sqrt(1-rho**2)
            white = linalg.solve_triangular(L, residual, lower=True, check_finite=False)
            total += float(np.square(white).sum())
        return total

    def evaluate(self, R: np.ndarray, column_precision: np.ndarray | float | None = None,
                 precision: PrecisionSpec | None = None, sigma2: float | None = None,
                 temporal_rho: float | None = None, details: bool = True) -> GaussianEvaluation:
        """Normalized integrated likelihood.

        With ``sigma2=None`` both C and sigma2 are integrated.  Otherwise C is
        integrated and the supplied positive scale is held fixed (separate PC
        scale priors use this route).  Invalid proposals raise, allowing the
        sampler to reject them explicitly without concealing programming errors.
        """
        del details  # Evaluation always contains inexpensive recovery factors.
        if precision is not None and column_precision is not None:
            raise ValueError("Supply precision or column_precision, not both")
        if precision is None:
            column_precision = 1. if column_precision is None else column_precision
            cp = np.asarray(column_precision)
            precision = PrecisionSpec(self.p, self.d, "scalar" if cp.ndim == 0 else "column", cp)
        self._check_precision(precision)
        if sigma2 is not None and (not np.isfinite(sigma2) or sigma2 <= 0):
            raise ValueError("Fixed sigma2 must be positive and finite")
        G, YY, YZ, logdetT = self.statistics(temporal_rho)
        L, RiPhi, B, beta, U, Ri, logdetR = self.spatial_statistics(R)
        sY = float(np.sum(Ri * YY))
        extras = {}
        if precision.separable:
            columns = precision.column_values
            g, V, unscale, lam = self._temporal_factor(G, columns, temporal_rho)
            if (temporal_rho is None or temporal_rho==0) and np.all(columns==columns[0]):
                cross = U.T @ (RiPhi.T @ self._YZ_eig)
            else:
                J = RiPhi.T @ YZ
                cross = U.T @ ((J * unscale) @ V)
            den = lam + beta[:, None] * g[None, :]
            if not np.isfinite(den).all():
                raise FloatingPointError("Nonfinite spectral precision")
            eigenmean = cross / den
            Q = sY - float(np.sum(cross * eigenmean))
            logratio = self.k * np.log(lam) - float(np.log(den).sum())
            extras = dict(spatial_vectors=U, temporal_vectors=V, eigenmean=eigenmean,
                          eigenprecision=den, column_unscale=unscale)
        else:
            if not self._dense_allowed():
                raise MemoryError("Nonseparable collapse requires a bounded dense determinant; use conditional MCMC")
            J = RiPhi.T @ YZ
            A = precision.as_matrix(self.dense_threshold) + np.kron(G, B)
            LA = linalg.cholesky(A, lower=True, check_finite=False)
            h = J.reshape(-1, order="F")
            m = linalg.cho_solve((LA, True), h, check_finite=False)
            Q = sY - float(h @ m)
            logratio = precision.logdet - 2 * float(np.log(np.diag(LA)).sum())
            extras = dict(dense_mean=m, dense_cholesky=LA)
        ans = GaussianEvaluation(0., self.an, 0., Q, logdetR, logdetT, precision,
                                 fixed_sigma2=sigma2, **extras)
        if Q < 1e-7 * max(1., sY):
            Q = self.residual_quadratic(ans.mean, R, temporal_rho) + precision.quadratic(ans.mean)
        if Q < 0 or not np.isfinite(Q):
            raise FloatingPointError("Invalid marginal quadratic form")
        ans.Q, ans.bn = float(Q), self.b0 + .5 * float(Q)
        base = -.5 * self.N * np.log(2*np.pi) -.5 * (self.T*logdetR + self.n*logdetT) + .5*logratio
        if sigma2 is None:
            ans.log_marginal = float(base + self.a0*np.log(self.b0) + special.gammaln(self.an)
                                     - special.gammaln(self.a0) - self.an*np.log(ans.bn))
        else:
            ans.log_marginal = float(base -.5*self.N*np.log(sigma2) -.5*Q/sigma2)
        return ans

    def draw(self, evaluation: GaussianEvaluation, rng: np.random.Generator,
             sigma2: float | None = None) -> tuple[np.ndarray, float, dict]:
        return evaluation.draw(rng, sigma2)

    def draw_conditional(self, R: np.ndarray, precision: PrecisionSpec, sigma2: float,
                         rng: np.random.Generator, backend: str = "auto",
                         rtol: float = 1e-10, atol: float = 0., maxiter: int | None = None,
                         temporal_rho: float | None = None,
                         dense_fallback: bool = True) -> tuple[np.ndarray, dict]:
        self._check_precision(precision)
        if sigma2 <= 0 or not np.isfinite(sigma2):
            raise ValueError("sigma2 must be positive")
        if precision.separable and backend not in ("matrixfree", "gaussian_matrixfree", "dense"):
            ev = self.evaluate(R, precision=precision, sigma2=sigma2, temporal_rho=temporal_rho)
            C, _, meta = ev.draw(rng)
            return C, meta
        if backend not in ("auto", "dense", "matrixfree", "gaussian_matrixfree", "spectral_gaussian"):
            raise ValueError(f"Unsupported coefficient backend {backend!r}")
        G, _, YZ, _ = self.statistics(temporal_rho)
        _, RiPhi, B, _, _, _, _ = self.spatial_statistics(R)
        J = RiPhi.T @ YZ
        if backend == "dense" or (backend == "auto" and self._dense_allowed()):
            if not self._dense_allowed():
                raise MemoryError("Dense Gaussian draw exceeds the configured memory/size gate")
            A = precision.as_matrix(self.dense_threshold) + np.kron(G, B)
            L = linalg.cholesky(A, lower=True, check_finite=False)
            mean = linalg.cho_solve((L, True), J.reshape(-1, order="F"), check_finite=False)
            b = mean + np.sqrt(sigma2)*linalg.solve_triangular(L.T, rng.normal(size=self.k), check_finite=False)
            return b.reshape(self.p,self.d,order="F"), {"solver":"dense_cholesky","converged":True,
                "finite_tolerance_approximation":False,"relative_residual":0.,"iterations":0}
        beta, Ub = psd_eigh(B)
        gamma, Ug = psd_eigh(G)
        LB, LG = Ub * np.sqrt(beta), Ug * np.sqrt(gamma)
        noise = precision.sample_precision_noise(rng) + LB @ rng.normal(size=(self.p,self.d)) @ LG.T
        rhs = (J + np.sqrt(sigma2)*noise).reshape(-1,order="F")
        def matvec(v):
            M = v.reshape(self.p,self.d,order="F")
            return (B @ M @ G).reshape(-1,order="F") + precision.apply(v)
        Aop = LinearOperator((self.k,self.k), matvec=matvec, dtype=float)
        prediag = precision.diagonal() + np.kron(np.diag(G), np.diag(B))
        Mop = LinearOperator((self.k,self.k), matvec=lambda v:v/prediag,dtype=float)
        iterations = [0]
        def callback(_):
            iterations[0] += 1
        limit = maxiter if maxiter is not None else max(100, min(10000, 5*self.k))
        solution, info = cg(Aop,rhs,M=Mop,rtol=rtol,atol=atol,maxiter=limit,callback=callback)
        absolute = float(linalg.norm(matvec(solution)-rhs))
        normrhs = float(linalg.norm(rhs))
        relative = absolute / max(normrhs, np.finfo(float).tiny)
        converged = info == 0 and absolute <= 1.05*max(atol,rtol*normrhs)
        meta = {"solver":"conjugate_gradient","converged":bool(converged),"info":int(info),
                "relative_residual":relative,"absolute_residual":absolute,
                "iterations":iterations[0],"rtol":rtol,"atol":atol,"maxiter":limit,
                "finite_tolerance_approximation":True,"dense_fallback":False}
        if not converged:
            if dense_fallback and self._dense_allowed():
                A = precision.as_matrix(self.dense_threshold) + np.kron(G,B)
                solution = linalg.cho_solve((linalg.cholesky(A,lower=True),True),rhs)
                meta.update(solver="dense_cholesky_fallback",converged=True,dense_fallback=True,
                            finite_tolerance_approximation=False,
                            fallback_relative_residual=float(linalg.norm(A@solution-rhs)/max(normrhs,1e-300)))
            else:
                raise linalg.LinAlgError(f"Gaussian perturb-and-solve failed: {meta}")
        return solution.reshape(self.p,self.d,order="F"),meta

    def draw_sigma2_conditional(self, C: np.ndarray, R: np.ndarray, precision: PrecisionSpec,
                                rng: np.random.Generator,
                                temporal_rho: float | None = None) -> float:
        scale = self.b0 + .5*(self.residual_quadratic(C,R,temporal_rho)+precision.quadratic(C))
        return float(scale/rng.gamma(self.a0+(self.N+self.k)/2))


class OperatorGaussianRegression(GaussianRegression):
    """Exact iid separable Gaussian algebra using a covariance operator.

    The spatial object supplies ``solve`` and ``logdet`` for R; the sufficient
    statistics are Phi' R^-1 Phi, Phi' R^-1 Y Z and tr(Y' R^-1 Y).
    Response quadratic forms are accumulated in bounded time-column chunks.
    No dense n-by-n covariance, inverse or response cross-product is built.
    The finite p-by-p and d-by-d spectral readout factors are unchanged.
    """

    def spatial_statistics(self, R) -> tuple:
        if R.shape != (self.n, self.n) or not callable(getattr(R, "solve", None)):
            raise ValueError("R must supply a matching spatial solve/logdet operator")
        if self._spatial_cache is not None and self._spatial_cache[0] is R:
            return self._spatial_cache[1]
        RiPhi = R.solve(self.Phi)
        B = self.Phi.T @ RiPhi
        beta, U = psd_eigh(B)
        sY = 0.
        for start in range(0, self.T, 64):
            block = self.Y[:, start:start+64]
            sY += float(np.sum(block * R.solve(block)))
        values = RiPhi, B, beta, U, sY, float(R.logdet())
        self._spatial_cache = (R, values)
        return values

    def residual_quadratic(self, C, R, temporal_rho=None, chunk_size=64) -> float:
        if temporal_rho not in (None, 0):
            raise ValueError("The spatial operator regression supports iid time innovations only")
        PC = self.Phi @ C
        total = 0.
        for start in range(0, self.T, chunk_size):
            residual = self.Y[:, start:start+chunk_size] - PC @ self.Z[start:start+chunk_size].T
            total += float(np.sum(residual * R.solve(residual)))
        return total

    def evaluate(self, R, column_precision=None, precision=None, sigma2=None,
                 temporal_rho=None, details=True) -> GaussianEvaluation:
        del details
        if temporal_rho not in (None, 0):
            raise ValueError("The spatial operator regression supports iid time innovations only")
        if precision is not None and column_precision is not None:
            raise ValueError("Supply precision or column_precision, not both")
        if precision is None:
            cp = np.asarray(1. if column_precision is None else column_precision)
            precision = PrecisionSpec(self.p, self.d, "scalar" if cp.ndim == 0 else "column", cp)
        self._check_precision(precision)
        if not precision.separable:
            raise ValueError("The operator regression currently supports separable coefficient precision only")
        if sigma2 is not None and (not np.isfinite(sigma2) or sigma2 <= 0):
            raise ValueError("Fixed sigma2 must be positive and finite")
        RiPhi, _, beta, U, sY, logdetR = self.spatial_statistics(R)
        columns = precision.column_values
        g, V, unscale, lam = self._temporal_factor(self.G, columns, None)
        if np.all(columns == columns[0]):
            cross = U.T @ (RiPhi.T @ self._YZ_eig)
        else:
            J = RiPhi.T @ self.YZ
            cross = U.T @ ((J * unscale) @ V)
        den = lam + beta[:, None] * g[None, :]
        if not np.isfinite(den).all():
            raise FloatingPointError("Nonfinite spectral precision")
        eigenmean = cross / den
        Q = sY - float(np.sum(cross * eigenmean))
        logratio = self.k*np.log(lam) - float(np.log(den).sum())
        ans = GaussianEvaluation(0., self.an, 0., Q, logdetR, 0., precision,
            spatial_vectors=U, temporal_vectors=V, eigenmean=eigenmean,
            eigenprecision=den, column_unscale=unscale, fixed_sigma2=sigma2)
        if Q < 1e-7*max(1., sY):
            Q = self.residual_quadratic(ans.mean, R) + precision.quadratic(ans.mean)
        if Q < 0 or not np.isfinite(Q):
            raise FloatingPointError("Invalid marginal quadratic form")
        ans.Q, ans.bn = float(Q), self.b0 + .5*float(Q)
        base = -.5*self.N*np.log(2*np.pi) -.5*self.T*logdetR + .5*logratio
        if sigma2 is None:
            ans.log_marginal = float(base + self.a0*np.log(self.b0) + special.gammaln(self.an)
                                     - special.gammaln(self.a0) - self.an*np.log(ans.bn))
        else:
            ans.log_marginal = float(base -.5*self.N*np.log(sigma2) -.5*Q/sigma2)
        return ans

    def draw_conditional(self, R, precision, sigma2, rng, backend="auto",
                         rtol=1e-10, atol=0., maxiter=None, temporal_rho=None,
                         dense_fallback=False):
        if backend not in ("auto", "spectral_gaussian", "vecchia_operator"):
            raise ValueError("The operator regression supports spectral coefficient draws only")
        evaluation = self.evaluate(R, precision=precision, sigma2=sigma2, temporal_rho=temporal_rho)
        C, _, meta = evaluation.draw(rng)
        meta["spatial_solver"] = "sparse_lu"
        return C, meta


def draw_coefficients(Y: np.ndarray, Phi: np.ndarray, Z: np.ndarray, R: np.ndarray,
                      prior_precision: PrecisionSpec, sigma2: float, rng: np.random.Generator,
                      backend: str = "auto", rtol: float = 1e-10, atol: float = 0.,
                      maxiter: int | None = None, dense_threshold: int = 900,
                      max_dense_bytes: int = 256*1024**2,
                      temporal_rho: float | None = None) -> tuple[np.ndarray, dict]:
    """Convenient conditional draw; reuse GaussianRegression for repeated draws."""
    engine = GaussianRegression(Y,Phi,Z,dense_threshold=dense_threshold,max_dense_bytes=max_dense_bytes)
    return engine.draw_conditional(R,prior_precision,sigma2,rng,backend,rtol,atol,maxiter,temporal_rho)
