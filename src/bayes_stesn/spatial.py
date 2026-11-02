"""Spatial covariance families and honest sparse approximations.

The Matérn argument is distance / ``ell`` and includes ``sqrt(2 * nu)``.
Consequently the SPDE practical range ``sqrt(8 * nu) / kappa`` is ``2*ell``.
Geometry and nugget bases are centred on the *input* sites, never the targets.
Sparse backends expose sparse precision operations; ``matrices`` explicitly
materialises their approximate covariance and is intended for reference fits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import linalg, sparse
from scipy.sparse import linalg as splinalg
from scipy.spatial import Delaunay, cKDTree
from scipy.special import gamma, kv

Array = np.ndarray


def _coordinates(value: Array) -> Array:
    a = np.asarray(value, dtype=float)
    if a.ndim != 2 or len(a) == 0 or not np.isfinite(a).all():
        raise ValueError("coordinates must be a nonempty finite (n, dimension) array")
    return a


def matern_correlation(distance: Array, nu: float = 1.5) -> Array:
    """Matérn correlation for nonnegative dimensionless distance.

    Closed forms avoid special-function loss of precision for the three
    supported finite-mixture smoothness values. General positive ``nu`` is
    also available for testing and the integer-order FEM backend.
    """
    r = np.asarray(distance, dtype=float)
    if nu <= 0 or not np.isfinite(nu) or np.any(r < 0) or np.isnan(r).any():
        raise ValueError("nu must be positive and distances nonnegative")
    x = np.sqrt(2.0 * nu) * r
    if nu == 0.5:
        return np.exp(-x)
    if nu == 1.5:
        return (1.0 + x) * np.exp(-x)
    if nu == 2.5:
        return (1.0 + x + x*x/3.0) * np.exp(-x)
    ans = np.ones_like(x)
    use = x > 1e-8
    ans[use] = (2.0**(1.0-nu) / gamma(nu)) * x[use]**nu * kv(nu, x[use])
    return ans


def anisotropy_matrix(a: float, b: float) -> Array:
    """Unit-determinant SPD matrix exp([[a,b],[b,-a]])."""
    r = float(np.hypot(a, b))
    s = 1.0 if r < 1e-12 else np.sinh(r) / r
    return np.cosh(r) * np.eye(2) + s * np.array([[a, b], [b, -a]])


def _raw_basis(coords: Array, reference: Array) -> Array:
    if coords.shape[1] != 2 or reference.shape[1] != 2:
        raise ValueError("spatial variance/nonstationarity bases require two coordinates")
    centre = np.mean(reference, axis=0)
    scale = np.ptp(reference, axis=0) / 2.0
    scale = np.where(scale > 1e-12, scale, 1.0)
    x, y = ((coords-centre)/scale).T
    return np.column_stack((x, y, x*y, (3*x*x-1)/2, (3*y*y-1)/2,
                            np.sin(np.pi*x), np.sin(np.pi*y),
                            np.cos(np.pi*x), np.cos(np.pi*y)))


def geometry_basis(coords: Array, reference: Array, terms: int) -> Array:
    """Fixed low-dimensional basis, centred and scaled at reference sites."""
    if not 1 <= terms <= 9:
        raise ValueError("geometry basis terms must be between 1 and 9")
    r = _raw_basis(reference, reference)[:, :terms]
    centre = r.mean(axis=0)
    scale = np.sqrt(np.mean((r-centre)**2, axis=0))
    scale = np.where(scale > 1e-10, scale, 1.0)
    return (_raw_basis(coords, reference)[:, :terms]-centre)/scale


@dataclass(frozen=True)
class CovarianceSpec:
    """Immutable spatial model specification.

    ``kind='matern'`` is exact dense covariance. ``vecchia`` uses a fixed
    input ordering (the supplied row order), Euclidean predecessor neighbours,
    and appended targets. ``spde`` is a genuine mass-lumped, piecewise-linear
    FEM approximation on a fixed triangular mesh with natural Neumann boundary
    conditions. It supports integer operator orders 2 and 3, hence smoothness
    1 and 2 in two dimensions. It is not a fractional-order nu=1.5 SPDE.

    Nonstationarity uses four fixed-scale Gaussian coefficient blocks for
    local range, log amplitude and two anisotropy coordinates. Their scales
    are explicit settings, not estimated hyperparameters. The reference-scale
    PC prior for this family is not a joint PC prior on all geometry terms.
    """

    kind: str = "matern"
    anisotropy: bool = False
    nonstationary: bool = False
    static_variance: bool = False
    nu: float | None = None
    infer_nu: bool = False
    smoothness_choices: tuple[float, ...] = (0.5, 1.5, 2.5)
    smoothness_probabilities: tuple[float, ...] | None = None
    range_prior: str = "lognormal"
    scale_mode: str = "relative"
    log_q_mean: float = 0.0
    log_q_sd: float = 1.25
    log_ell_mean: float = float(np.log(0.18))
    log_ell_sd: float = 0.65
    anisotropy_sd: float = 0.5
    variance_terms: int = 2
    variance_sd: float = 0.3
    geometry_terms: int = 4
    nonstationary_sd: float = 0.15
    pc_range_threshold: float = 0.3
    pc_range_probability: float = 0.05
    pc_sigma_threshold: float = 1.0
    pc_sigma_probability: float = 0.05
    sigma_e_scale: float = 1.0
    vecchia_neighbors: int = 15
    spde_order: int = 2
    mesh_resolution: int = 8
    mesh_padding: float = 0.5
    mesh_nodes: tuple[tuple[float, float], ...] | None = None
    max_dense_sites: int = 2000

    def __post_init__(self) -> None:
        if self.kind not in {"matern", "vecchia", "spde"}:
            raise ValueError("kind must be matern, vecchia or spde")
        if self.range_prior not in {"lognormal", "pc"}:
            raise ValueError("range_prior must be lognormal or pc")
        if self.scale_mode not in {"relative", "separate"}:
            raise ValueError("scale_mode must be relative or separate")
        if self.scale_mode == "separate" and self.range_prior != "pc":
            raise ValueError("separate physical scales require the declared PC range prior")
        for name in ("log_q_sd", "log_ell_sd", "anisotropy_sd", "variance_sd",
                     "nonstationary_sd", "pc_range_threshold", "pc_sigma_threshold",
                     "sigma_e_scale"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.log_q_mean) or not np.isfinite(self.log_ell_mean):
            raise ValueError("log-scale prior means must be finite")
        if not (0 < self.pc_range_probability < 1 and 0 < self.pc_sigma_probability < 1):
            raise ValueError("PC prior probabilities must lie in (0,1)")
        if not (1 <= self.geometry_terms <= 9 and 1 <= self.variance_terms <= 9):
            raise ValueError("basis sizes must lie between 1 and 9")
        if (not isinstance(self.vecchia_neighbors, (int, np.integer)) or
                not isinstance(self.max_dense_sites, (int, np.integer)) or
                self.vecchia_neighbors < 1 or self.max_dense_sites < 1):
            raise ValueError("neighbour count and dense size gate must be positive")
        choices = tuple(float(x) for x in self.smoothness_choices)
        if not choices or any(not np.isfinite(x) or x <= 0 for x in choices) or len(set(choices)) != len(choices):
            raise ValueError("smoothness choices must be distinct and positive")
        object.__setattr__(self, "smoothness_choices", choices)
        probs = (np.ones(len(choices))/len(choices) if self.smoothness_probabilities is None
                 else np.asarray(self.smoothness_probabilities, dtype=float))
        if probs.shape != (len(choices),) or np.any(probs <= 0) or not np.isclose(probs.sum(), 1):
            raise ValueError("smoothness probabilities must form a positive simplex")
        object.__setattr__(self, "smoothness_probabilities", tuple(probs))
        if self.kind == "spde":
            if self.spde_order not in {2, 3}:
                raise ValueError("FEM backend supports integer alpha=2 or 3 only")
            if self.infer_nu or self.nonstationary:
                raise ValueError("SPDE backend does not support smoothness mixtures or nonstationarity")
            effective_nu = float(self.spde_order-1)
            if self.nu is not None and self.nu != effective_nu:
                raise ValueError("in 2D the implemented SPDE has nu=alpha-1; fractional order is unsupported")
            object.__setattr__(self, "nu", effective_nu)
            if (not isinstance(self.mesh_resolution, (int, np.integer)) or
                    self.mesh_resolution < 3 or not np.isfinite(self.mesh_padding) or self.mesh_padding < 0):
                raise ValueError("mesh_resolution>=3 and nonnegative padding required")
        elif self.nu is None:
            object.__setattr__(self, "nu", 1.5)
        if self.nu is None or not np.isfinite(self.nu) or self.nu <= 0:
            raise ValueError("nu must be positive")
        if self.infer_nu and self.nu not in choices:
            raise ValueError("initial nu must belong to smoothness_choices")
        if self.mesh_nodes is not None:
            nodes = _coordinates(np.asarray(self.mesh_nodes, dtype=float))
            if nodes.shape[1] != 2 or len(nodes) < 3:
                raise ValueError("mesh_nodes must contain at least three 2D nodes")
            object.__setattr__(self, "mesh_nodes", tuple(map(tuple, nodes.tolist())))

    def parameter_names(self, coordinates: Array | None = None) -> tuple[str, ...]:
        names = ["log_ell"]
        names += ["log_q"] if self.scale_mode == "relative" else ["log_sigma_e", "log_sigma_w"]
        if self.anisotropy or self.nonstationary:
            names += ["aniso_a", "aniso_b"]
        if self.static_variance:
            names += [f"variance_beta[{j}]" for j in range(self.variance_terms)]
        if self.nonstationary:
            names += [f"nonstationary_coefficients[{j},{h}]"
                      for j in range(4) for h in range(self.geometry_terms)]
        return tuple(names)

    def initial_parameters(self, coordinates: Array) -> dict[str, Any]:
        _coordinates(coordinates)
        params: dict[str, Any] = {"log_ell": self.log_ell_mean, "nu": float(self.nu)}
        if self.scale_mode == "relative":
            params["log_q"] = self.log_q_mean
        else:
            params.update(log_sigma_e=float(np.log(self.sigma_e_scale/2)),
                          log_sigma_w=float(np.log(self.pc_sigma_threshold/2)))
            params["log_q"] = 2*(params["log_sigma_w"]-params["log_sigma_e"])
        if self.anisotropy or self.nonstationary:
            params.update(aniso_a=0.0, aniso_b=0.0)
        if self.static_variance:
            params["variance_beta"] = np.zeros(self.variance_terms)
        if self.nonstationary:
            params["nonstationary_coefficients"] = np.zeros((4, self.geometry_terms))
        return params

    def parameters_to_vector(self, params: dict[str, Any], coordinates: Array | None = None) -> Array:
        x = [float(params["log_ell"])]
        if self.scale_mode == "relative":
            x.append(float(params["log_q"]))
        else:
            x.extend((float(params["log_sigma_e"]), float(params["log_sigma_w"])))
        if self.anisotropy or self.nonstationary:
            x.extend((float(params["aniso_a"]), float(params["aniso_b"])))
        if self.static_variance:
            x.extend(np.asarray(params["variance_beta"]).ravel())
        if self.nonstationary:
            x.extend(np.asarray(params["nonstationary_coefficients"]).ravel())
        ans = np.asarray(x, dtype=float)
        if ans.shape != (len(self.parameter_names()),):
            raise ValueError("covariance parameter shapes do not match the specification")
        return ans

    def vector_to_parameters(self, vector: Array, coordinates: Array | None = None,
                             nu: float | None = None) -> dict[str, Any]:
        x = np.asarray(vector, dtype=float)
        if x.shape != (len(self.parameter_names()),):
            raise ValueError("covariance vector has the wrong shape")
        p: dict[str, Any] = {"log_ell": float(x[0]), "nu": float(self.nu if nu is None else nu)}
        j = 1
        if self.scale_mode == "relative":
            p["log_q"] = float(x[j]); j += 1
        else:
            p["log_sigma_e"], p["log_sigma_w"] = map(float, x[j:j+2]); j += 2
            p["log_q"] = 2*(p["log_sigma_w"]-p["log_sigma_e"])
        if self.anisotropy or self.nonstationary:
            p["aniso_a"], p["aniso_b"] = map(float, x[j:j+2]); j += 2
        if self.static_variance:
            p["variance_beta"] = x[j:j+self.variance_terms].copy(); j += self.variance_terms
        if self.nonstationary:
            p["nonstationary_coefficients"] = x[j:].reshape(4, self.geometry_terms).copy()
        return p

    def log_prior(self, params: dict[str, Any], coordinates: Array | None = None) -> float:
        """Log prior in the actual sampled coordinates (constants omitted)."""
        try:
            if not np.isfinite(self.parameters_to_vector(params)).all():
                return -np.inf
            h = float(params["log_ell"])
            with np.errstate(over="raise", invalid="raise"):
                if self.range_prior == "lognormal":
                    value = -0.5*((h-self.log_ell_mean)/self.log_ell_sd)**2
                else:
                    lam_r = -self.pc_range_threshold*np.log(self.pc_range_probability)
                    value = -h-lam_r*np.exp(-h)/2.0
                if self.scale_mode == "relative":
                    value -= 0.5*((params["log_q"]-self.log_q_mean)/self.log_q_sd)**2
                else:
                    e, w = params["log_sigma_e"], params["log_sigma_w"]
                    lam_w = -np.log(self.pc_sigma_probability)/self.pc_sigma_threshold
                    value += e-np.exp(2*e)/(2*self.sigma_e_scale**2)+w-lam_w*np.exp(w)
                if self.anisotropy or self.nonstationary:
                    value -= 0.5*(params["aniso_a"]**2+params["aniso_b"]**2)/self.anisotropy_sd**2
                if self.static_variance:
                    value -= 0.5*np.sum(np.asarray(params["variance_beta"])**2)/self.variance_sd**2
                if self.nonstationary:
                    value -= 0.5*np.sum(np.asarray(params["nonstationary_coefficients"])**2)/self.nonstationary_sd**2
            nu = float(params.get("nu", self.nu))
            if self.infer_nu:
                if nu not in self.smoothness_choices:
                    return -np.inf
                value += np.log(self.smoothness_probabilities[self.smoothness_choices.index(nu)])
            elif nu != self.nu:
                return -np.inf
            return float(value) if np.isfinite(value) else -np.inf
        except (KeyError, ValueError, OverflowError, FloatingPointError):
            return -np.inf

    def _nugget_diagonal(self, coordinates: Array, reference: Array, params: dict[str, Any]) -> Array:
        if not self.static_variance:
            return np.ones(len(coordinates))
        return np.exp(geometry_basis(coordinates, reference, self.variance_terms)
                      @ np.asarray(params["variance_beta"]))

    def _kernel(self, left: Array, right: Array, reference: Array,
                params: dict[str, Any]) -> Array:
        ell = np.exp(float(params["log_ell"]))
        nu = float(params.get("nu", self.nu))
        if not np.isfinite(ell) or ell <= 0:
            raise ValueError("invalid length scale")
        delta = left[:, None, :]-right[None, :, :]
        if not self.nonstationary:
            metric = np.eye(left.shape[1])
            if self.anisotropy:
                if left.shape[1] != 2:
                    raise ValueError("anisotropy requires two coordinates")
                metric = anisotropy_matrix(params["aniso_a"], params["aniso_b"])
            squared = np.einsum("...i,ij,...j->...", delta, metric, delta)
            return matern_correlation(np.sqrt(np.maximum(squared, 0))/ell, nu)
        coeff = np.asarray(params["nonstationary_coefficients"])
        fl = geometry_basis(left, reference, self.geometry_terms) @ coeff.T
        fr = geometry_basis(right, reference, self.geometry_terms) @ coeff.T

        def local_h(f: Array) -> Array:
            mats = np.stack([anisotropy_matrix(-params["aniso_a"]-x[2],
                                              -params["aniso_b"]-x[3]) for x in f])
            return ell**2*np.exp(2*f[:, 0])[:, None, None]*mats

        hl, hr = local_h(fl), local_h(fr)
        hbar = (hl[:, None, :, :]+hr[None, :, :, :])/2
        h_inv = np.linalg.inv(hbar)
        squared = np.einsum("...i,...ij,...j->...", delta, h_inv, delta)
        det_l, det_r = np.linalg.det(hl), np.linalg.det(hr)
        volume = (det_l[:, None]*det_r[None, :])**0.25/np.sqrt(np.linalg.det(hbar))
        amplitude = np.exp(fl[:, 1, None]+fr[None, :, 1])
        return amplitude*volume*matern_correlation(np.sqrt(np.maximum(squared, 0)), nu)

    def operator(self, coordinates: Array, params: dict[str, Any], *,
                 reference_coordinates: Array | None = None) -> "SpatialOperator":
        """Build a sparse approximation supporting solve/logdet without dense K.

        For appended targets, pass the original input sites as the reference.
        This preserves centring, FEM mesh and every input Vecchia factor.
        """
        coords = _coordinates(coordinates)
        ref = coords if reference_coordinates is None else _coordinates(reference_coordinates)
        if self.kind == "matern":
            raise ValueError("the exact dense Matérn has no sparse precision operator")
        if self.kind == "vecchia":
            return self.prepare_geometry(coords, reference_coordinates=ref).operator(params)
        nugget = self._nugget_diagonal(coords, ref, params)
        q = float(np.exp(params["log_q"]))
        if not np.isfinite(q) or q <= 0 or not np.all(np.isfinite(nugget)):
            raise ValueError("invalid covariance scale")
        if coords.shape[1] != 2:
            raise ValueError("FEM SPDE is implemented in two spatial dimensions")
        nodes = (np.asarray(self.mesh_nodes) if self.mesh_nodes is not None
                 else rectangular_mesh(ref, self.mesh_resolution, self.mesh_padding))
        tri = Delaunay(nodes)
        interpolation = mesh_interpolation(tri, coords)
        a = params.get("aniso_a", 0.0)
        b = params.get("aniso_b", 0.0)
        diffusion = anisotropy_matrix(-a, -b)
        mass, stiffness = fem_matrices(nodes, tri.simplices, diffusion)
        kappa = np.sqrt(2*float(self.nu))/np.exp(params["log_ell"])
        base = kappa*kappa*sparse.diags(mass)+stiffness
        Q = base.copy()
        mass_inverse = sparse.diags(1/mass)
        for _ in range(1, self.spde_order):
            Q = Q @ mass_inverse @ base
        # Infinite-domain variance convention; mesh/boundary effects remain.
        tau2 = gamma(float(self.nu))/(gamma(self.spde_order)*4*np.pi*kappa**(2*float(self.nu)))
        Q = ((Q+Q.T)*(tau2/2)).tocsc()
        return SpatialOperator(Q, interpolation, q, nugget,
                               {"backend": "spde_fem", "alpha": self.spde_order,
                                "nu": float(self.nu), "mesh_nodes": len(nodes),
                                "mesh_triangles": len(tri.simplices),
                                "boundary": "natural Neumann; fixed padded domain",
                                "mass_lumping": True, "unit_marginal_variance": False})

    def prepare_geometry(self, coordinates: Array, *,
                         reference_coordinates: Array | None = None) -> "PreparedVecchia":
        """Cache the fixed Vecchia ordering and predecessor search.

        Conditional coefficients and sparse LU factors remain parameter
        dependent and are recomputed. Only the iid Gaussian, ridge, collapsed
        fit route currently consumes this cache end to end; other model
        combinations retain their size-gated dense reference implementation.
        """
        if self.kind != "vecchia":
            raise ValueError("Prepared sparse geometry currently supports Vecchia only")
        return PreparedVecchia(self, coordinates, reference_coordinates)

    def matrices(self, coordinates: Array, params: dict[str, Any], *,
                 reference_coordinates: Array | None = None) -> tuple[Array, Array]:
        """Return K,V in R=V+qK; sparse approximations are materialised here."""
        coords = _coordinates(coordinates)
        ref = coords if reference_coordinates is None else _coordinates(reference_coordinates)
        if len(coords) > self.max_dense_sites:
            raise ValueError(f"dense covariance limited to {self.max_dense_sites} sites; use operator()")
        if self.kind == "matern":
            K = self._kernel(coords, coords, ref, params)
            V = np.diag(self._nugget_diagonal(coords, ref, params))
        else:
            operator = self.operator(coords, params, reference_coordinates=ref)
            K = operator.spatial_covariance_dense()
            V = np.diag(operator.nugget_diagonal)
        if not (np.isfinite(K).all() and np.isfinite(V).all()):
            raise ValueError("nonfinite spatial covariance")
        return (K+K.T)/2, V

    def covariance(self, coordinates: Array, params: dict[str, Any], *,
                   reference_coordinates: Array | None = None) -> Array:
        K, V = self.matrices(coordinates, params, reference_coordinates=reference_coordinates)
        return V+np.exp(params["log_q"])*K

    def joint_matrices(self, input_coordinates: Array, target_coordinates: Array,
                       params: dict[str, Any]) -> tuple[Array, Array]:
        """K,V for [inputs,targets], retaining input geometry and factors.

        Exact duplicate targets should be deduplicated by the caller and their
        observation paths copied. A duplicate *latent* coordinate is legal for
        dense Matérn/FEM, but makes a no-nugget Vecchia conditional degenerate
        and is rejected there. A fixed master target ordering is required for
        Vecchia subset projectivity; rebuilding different target orderings
        defines different approximations.
        """
        inputs, targets = _coordinates(input_coordinates), _coordinates(target_coordinates)
        return self.matrices(np.vstack((inputs, targets)), params, reference_coordinates=inputs)

    def joint_covariance(self, input_coordinates: Array, target_coordinates: Array,
                         params: dict[str, Any]) -> Array:
        K, V = self.joint_matrices(input_coordinates, target_coordinates, params)
        return V+np.exp(params["log_q"])*K


def _sparse_logdet(factor: Any) -> float:
    return float(np.log(np.abs(factor.U.diagonal())).sum())


def _sparse_factor(matrix):
    """Normalize SuperLU's numerical-singularity exception at its boundary."""
    try:
        return splinalg.splu(matrix)
    except RuntimeError as exc:
        if "singular" in str(exc).lower():
            raise linalg.LinAlgError(f"Sparse precision factorization failed: {exc}") from exc
        raise


class SpatialOperator:
    """R=diag(v)+q A Q^{-1}A' with sparse Q and interpolation A.

    Sparse LU is used as the available portable factorisation, not described
    as sparse Cholesky. The underlying Q is symmetric positive definite.
    """

    def __init__(self, precision: sparse.spmatrix, interpolation: sparse.spmatrix,
                 q: float, nugget_diagonal: Array, metadata: dict[str, Any], *,
                 precision_logdet: float | None = None) -> None:
        self.precision = precision.tocsc()
        self.interpolation = interpolation.tocsc()
        self.q = q
        self.nugget_diagonal = np.asarray(nugget_diagonal, dtype=float)
        self.metadata = metadata
        self._precision_logdet = precision_logdet
        self._q_factor_cache = None
        if precision_logdet is None:
            self._q_factor_cache = _sparse_factor(self.precision)
        self._vinv = 1/self.nugget_diagonal
        a = self.interpolation
        self._identity_interpolation = (a.shape == self.precision.shape and
                                        a.nnz == a.shape[0] and np.all(a.diagonal() == 1))
        self._updated = self.precision+q*(a.T @ sparse.diags(self._vinv) @ a)
        self._updated_factor = _sparse_factor(self._updated.tocsc())

    @property
    def _q_factor(self):
        # Vecchia supplies log|Q| directly from its conditional variances.
        # Its prior LU is needed only for explicitly requested covariance
        # reference/marginal operations, not fit, recovery or prediction.
        if self._q_factor_cache is None:
            self._q_factor_cache = _sparse_factor(self.precision)
        return self._q_factor_cache

    @property
    def shape(self) -> tuple[int, int]:
        n = self.interpolation.shape[0]
        return n, n

    def solve(self, rhs: Array) -> Array:
        rhs = np.asarray(rhs, dtype=float)
        one = rhs.ndim == 1
        x = rhs[:, None] if one else rhs
        if x.shape[0] != self.shape[0]:
            raise ValueError("right-hand side has wrong leading dimension")
        if self._identity_interpolation:
            # V^-1 (Q + q V^-1)^-1 Q x is algebraically the same
            # Woodbury solve, without cancellation when q is very large.
            ans = self._vinv[:, None]*self._updated_factor.solve(self.precision @ x)
            return ans[:, 0] if one else ans
        d = self._vinv[:, None]*x
        correction = self.interpolation @ self._updated_factor.solve(self.interpolation.T @ d)
        ans = d-self.q*self._vinv[:, None]*correction
        return ans[:, 0] if one else ans

    def logdet(self) -> float:
        logdet_precision = (_sparse_logdet(self._q_factor) if self._precision_logdet is None
                            else self._precision_logdet)
        return float(np.log(self.nugget_diagonal).sum()
                     +_sparse_logdet(self._updated_factor)-logdet_precision)

    def spatial_covariance_dense(self) -> Array:
        a = self.interpolation
        ans = a @ self._q_factor.solve(a.T.toarray())
        return (np.asarray(ans)+np.asarray(ans).T)/2

    def marginal_variances(self, batch_size: int = 64) -> Array:
        """Actual diag(K_tilde), with bounded dense right-hand-side batches."""
        n = self.shape[0]
        result = np.empty(n)
        for start in range(0, n, batch_size):
            stop = min(start+batch_size, n)
            rows = self.interpolation[start:stop]
            solved = self._q_factor.solve(rows.T.toarray())
            result[start:stop] = np.asarray(rows.multiply(solved.T).sum(axis=1)).ravel()
        return result


def _vecchia_predecessors(coordinates: Array, neighbours: int) -> tuple[Array, ...]:
    """Fixed Euclidean predecessors, with deterministic boundary ties."""
    coords = _coordinates(coordinates)
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError("Vecchia latent sites must be unique; deduplicate target coordinates")
    if not isinstance(neighbours, (int, np.integer)) or neighbours < 1:
        raise ValueError("Vecchia neighbour count must be a positive integer")
    n = len(coords)
    tree = cKDTree(coords)
    result = []
    for i in range(n):
        count = min(neighbours, i)
        if count == 0:
            predecessors = np.empty(0, dtype=int)
        else:
            k = min(n, max(2*count+1, 8))
            while True:
                distances, ids = tree.query(coords[i], k=k)
                ids = np.atleast_1d(ids)
                distances = np.atleast_1d(distances)
                keep = ids < i
                ids, distances = ids[keep], distances[keep]
                if len(ids) >= count or k == n:
                    order = np.lexsort((ids, distances))
                    cutoff = distances[order[count-1]]
                    # Include every tie at the predecessor boundary before
                    # tie-breaking by row index. Appended targets can change
                    # KD-tree query ties, but must never alter input factors.
                    candidates = np.asarray(tree.query_ball_point(
                        coords[i], np.nextafter(cutoff, np.inf)), dtype=int)
                    candidates = candidates[candidates < i]
                    candidate_distance = np.linalg.norm(coords[candidates]-coords[i], axis=1)
                    order = np.lexsort((candidates, candidate_distance))
                    predecessors = candidates[order[:count]]
                    break
                k = min(n, 2*k)
        predecessors.setflags(write=False)
        result.append(predecessors)
    return tuple(result)


class VecchiaFactor:
    """Latent conditional factor B w = D**(1/2) z; Q = B' D**(-1) B.

    Simulation uses a sparse triangular solve in the declared joint ordering.
    No covariance root or spatial identity matrix is materialized.
    """

    def __init__(self, factor: sparse.spmatrix, conditional_variances: Array):
        self.factor = factor.tocsr()
        self.conditional_variances = np.asarray(conditional_variances, dtype=float)

    def precision(self) -> sparse.csc_matrix:
        return (self.factor.T @ sparse.diags(1/self.conditional_variances) @ self.factor).tocsc()

    def sample(self, rng: np.random.Generator, size: int | None = None) -> Array:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("An explicit NumPy Generator is required")
        if size is None:
            rhs = np.sqrt(self.conditional_variances)*rng.normal(size=len(self.conditional_variances))
        else:
            if not isinstance(size, (int, np.integer)) or size < 1:
                raise ValueError("sample size must be a positive integer")
            rhs = np.sqrt(self.conditional_variances)[:, None]*rng.normal(
                size=(len(self.conditional_variances), int(size)))
        return splinalg.spsolve_triangular(self.factor, rhs, lower=True, unit_diagonal=True)


def _vecchia_factor(coords: Array, predecessors: tuple[Array, ...], kernel: Any) -> VecchiaFactor:
    n = len(coords)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    variances = np.empty(n)
    for i, preceding in enumerate(predecessors):
        count = len(preceding)
        marginal = float(kernel(coords[i:i+1], coords[i:i+1])[0, 0])
        if count:
            cross = kernel(coords[preceding], coords[i:i+1])[:, 0]
            local = kernel(coords[preceding], coords[preceding])
            weights = linalg.solve(local, cross, assume_a="pos")
            conditional = marginal-float(cross @ weights)
            rows.extend([i]*count); cols.extend(preceding.tolist()); data.extend((-weights).tolist())
        else:
            conditional = marginal
        if not np.isfinite(conditional) or conditional <= 0:
            raise linalg.LinAlgError("nonpositive Vecchia conditional variance; geometry is numerically singular")
        variances[i] = conditional
        rows.append(i); cols.append(i); data.append(1.0)
    factor = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsc()
    return VecchiaFactor(factor, variances)


class PreparedVecchia:
    """Reusable immutable input geometry, independent of sampled parameters.

    The original inputs must remain the reference when targets are appended.
    Sparse precision construction costs O(n m²) storage in the worst case;
    sparse LU fill depends on geometry and is not guaranteed linear in n.
    """

    def __init__(self, specification: CovarianceSpec, coordinates: Array,
                 reference_coordinates: Array | None = None):
        if specification.kind != "vecchia":
            raise ValueError("PreparedVecchia requires a Vecchia specification")
        self.specification = specification
        self.coordinates = np.array(_coordinates(coordinates), copy=True)
        self.reference_coordinates = (self.coordinates if reference_coordinates is None else
                                      np.array(_coordinates(reference_coordinates), copy=True))
        if self.coordinates.shape[1] != self.reference_coordinates.shape[1]:
            raise ValueError("Input and reference coordinate dimensions disagree")
        self.coordinates.setflags(write=False)
        self.reference_coordinates.setflags(write=False)
        self.predecessors = _vecchia_predecessors(self.coordinates, specification.vecchia_neighbors)

    def factor(self, params: dict[str, Any]) -> VecchiaFactor:
        return _vecchia_factor(self.coordinates, self.predecessors,
            lambda a, b: self.specification._kernel(a, b, self.reference_coordinates, params))

    def nugget_diagonal(self, params: dict[str, Any]) -> Array:
        diagonal = self.specification._nugget_diagonal(
            self.coordinates, self.reference_coordinates, params)
        if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
            raise ValueError("Invalid nugget variances")
        return diagonal

    def operator(self, params: dict[str, Any]) -> SpatialOperator:
        q = float(np.exp(params["log_q"]))
        if not np.isfinite(q) or q <= 0:
            raise ValueError("Invalid spatial variance ratio")
        factor = self.factor(params)
        Q = factor.precision()
        result = SpatialOperator(Q, sparse.eye(len(self.coordinates), format="csc"),
                                 q, self.nugget_diagonal(params),
                                 {"backend": "vecchia", "conditional_variances": factor.conditional_variances,
                                  "ordering": "supplied rows; targets appended",
                                  "neighbors": self.specification.vecchia_neighbors,
                                  "geometry_cached": True, "unit_marginal_variance": False,
                                  "factorization": "sparse_lu", "precision_nonzeros": Q.nnz,
                                  "precision_logdet_method": "Vecchia conditional variances"},
                                 precision_logdet=-float(np.log(factor.conditional_variances).sum()))
        result.metadata["lu_nonzeros"] = int(result._updated_factor.L.nnz + result._updated_factor.U.nnz)
        return result


def vecchia_precision(coordinates: Array, neighbours: int, kernel: Any
                      ) -> tuple[sparse.csc_matrix, Array]:
    """Construct the fixed-order latent conditional approximation.

    Use ``CovarianceSpec.prepare_geometry`` to reuse predecessor searches
    across parameter proposals. This convenience function rebuilds geometry.
    """
    coords = _coordinates(coordinates)
    factor = _vecchia_factor(coords, _vecchia_predecessors(coords, neighbours), kernel)
    return factor.precision(), factor.conditional_variances


def rectangular_mesh(reference: Array, resolution: int, padding: float) -> Array:
    ref = _coordinates(reference)
    if ref.shape[1] != 2:
        raise ValueError("rectangular mesh requires 2D geometry")
    lower, upper = ref.min(axis=0), ref.max(axis=0)
    span = upper-lower
    if np.any(span <= 0):
        raise ValueError("automatic FEM mesh requires nondegenerate 2D input extent")
    x = np.linspace(lower[0]-padding*span[0], upper[0]+padding*span[0], resolution)
    y = np.linspace(lower[1]-padding*span[1], upper[1]+padding*span[1], resolution)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel()))


def fem_matrices(nodes: Array, triangles: Array, diffusion: Array | None = None
                 ) -> tuple[Array, sparse.csc_matrix]:
    """Mass-lumped P1 FEM mass diagonal and stiffness on a triangular mesh."""
    nodes = _coordinates(nodes)
    diffusion = np.eye(2) if diffusion is None else np.asarray(diffusion, dtype=float)
    if diffusion.shape != (2, 2) or np.linalg.eigvalsh(diffusion).min() <= 0:
        raise ValueError("diffusion must be a 2x2 SPD matrix")
    mass = np.zeros(len(nodes))
    rows, cols, data = [], [], []
    for tri in np.asarray(triangles, dtype=int):
        xy = nodes[tri]
        area = abs(np.linalg.det(np.column_stack((xy[1]-xy[0], xy[2]-xy[0]))))/2
        if area <= 0:
            raise ValueError("degenerate FEM triangle")
        coefficients = np.linalg.inv(np.column_stack((np.ones(3), xy)))
        gradients = coefficients[1:, :]
        element = area*(gradients.T @ diffusion @ gradients)
        mass[tri] += area/3
        for i in range(3):
            for j in range(3):
                rows.append(int(tri[i])); cols.append(int(tri[j])); data.append(float(element[i, j]))
    if np.any(mass <= 0):
        raise ValueError("every FEM node must belong to a nondegenerate triangle")
    stiffness = sparse.coo_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes))).tocsc()
    return mass, stiffness


def mesh_interpolation(triangulation: Delaunay, coordinates: Array) -> sparse.csc_matrix:
    """Barycentric P1 interpolation; sites outside the fixed mesh fail clearly."""
    coords = _coordinates(coordinates)
    simplex = triangulation.find_simplex(coords, tol=1e-10)
    if np.any(simplex < 0):
        raise ValueError("prediction coordinates fall outside the fixed FEM mesh; declare a larger mesh")
    transform = triangulation.transform[simplex]
    bary12 = np.einsum("nij,nj->ni", transform[:, :2], coords-transform[:, 2])
    bary = np.column_stack((bary12, 1-bary12.sum(axis=1)))
    return sparse.coo_matrix((bary.ravel(),
                              (np.repeat(np.arange(len(coords)), 3),
                               triangulation.simplices[simplex].ravel())),
                             shape=(len(coords), len(triangulation.points))).tocsc()
