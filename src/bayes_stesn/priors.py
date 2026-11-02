"""Proper noise-scaled readout priors and their exact conditional kernels.

The finite-slab regularized horseshoe uses slice updates of the actual RHS
density.  Its scale updates are deliberately not the ordinary horseshoe's
inverse-gamma shortcut.  GIGG uses a gamma group variance and inverse-gamma
entry variance, with an explicit generalized inverse Gaussian convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from scipy import linalg, special, stats

from .gaussian import PrecisionSpec, psd_eigh


def _immutable(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(_immutable(x) for x in value)
    return value


@dataclass(frozen=True)
class PriorSpec:
    """Immutable configuration for a proper, noise-scaled coefficient prior.

    Group labels align with temporal columns.  ``unshrunk_groups`` receive a
    fixed, proper variance, never an improper flat prior.  R2 allocation covers
    only the remaining component and uses a frozen reference design.
    """

    kind: str = "ridge"
    unit: str = "group"
    mean_log_precision: Any = float(np.log(.1))
    sd_log_precision: Any = 1.5
    tau0: float = .1
    slab_scale: float = 2.
    slab_df: float = 4.
    fixed_slab: bool = False
    unshrunk_groups: tuple[str, ...] = ("intercept",)
    fixed_variance: float = 10.
    shape_a: Any = .5
    shape_b: Any = .5
    tau_scale: float = 1.
    beta_a: float = 2.
    beta_b: float = 2.
    allocation: str = "dirichlet"
    alpha: Any = 1.
    logistic_mean: Any = ()
    logistic_cov: Any = ()
    base_columns: int | None = None
    baseline_groups: tuple[str, ...] = ("intercept", "score", "base")
    lambda0: float = .1
    penalty: Any = None
    max_penalty_size: int = 900

    def __post_init__(self):
        if self.kind not in ("ridge", "group", "rhs", "gigg", "reference_r2", "correction", "function_penalty"):
            raise ValueError(f"Unknown readout prior {self.kind!r}")
        if self.unit not in ("group", "column", "entry"):
            raise ValueError("RHS unit must be group, column or entry")
        if self.allocation not in ("dirichlet", "logistic_normal"):
            raise ValueError("Allocation must be dirichlet or logistic_normal")
        for name in ("tau0", "slab_scale", "slab_df", "fixed_variance", "tau_scale",
                     "beta_a", "beta_b", "lambda0"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("mean_log_precision", "sd_log_precision", "shape_a", "shape_b", "alpha",
                     "logistic_mean", "logistic_cov", "penalty", "unshrunk_groups", "baseline_groups"):
            object.__setattr__(self, name, _immutable(getattr(self, name)))
        if self.base_columns is not None and self.base_columns < 1:
            raise ValueError("base_columns must be positive")
        if self.max_penalty_size < 1:
            raise ValueError("max_penalty_size must be positive")

    @classmethod
    def ridge(cls, mean_log_precision=float(np.log(.1)), sd_log_precision=1.5, **kwargs):
        return cls(kind="ridge", mean_log_precision=mean_log_precision,
                   sd_log_precision=sd_log_precision, **kwargs)

    @classmethod
    def grouped(cls, mean_log_precision=float(np.log(.1)), sd_log_precision=1.5, **kwargs):
        return cls(kind="group", mean_log_precision=mean_log_precision,
                   sd_log_precision=sd_log_precision, **kwargs)

    group_ridge = grouped

    @classmethod
    def rhs(cls, unit="group", tau0=.1, slab_scale=2., slab_df=4., **kwargs):
        return cls(kind="rhs", unit=unit, tau0=tau0, slab_scale=slab_scale,
                   slab_df=slab_df, **kwargs)

    @classmethod
    def group_rhs(cls, **kwargs):
        return cls.rhs(unit="group", **kwargs)

    @classmethod
    def column_rhs(cls, **kwargs):
        return cls.rhs(unit="column", **kwargs)

    @classmethod
    def entry_rhs(cls, **kwargs):
        return cls.rhs(unit="entry", **kwargs)

    @classmethod
    def gigg(cls, shape_a=.5, shape_b=.5, tau_scale=.1, **kwargs):
        return cls(kind="gigg", shape_a=shape_a, shape_b=shape_b, tau_scale=tau_scale, **kwargs)

    @classmethod
    def reference_r2(cls, beta_a=2., beta_b=2., allocation="dirichlet", **kwargs):
        return cls(kind="reference_r2", beta_a=beta_a, beta_b=beta_b,
                   allocation=allocation, **kwargs)

    r2d2 = reference_r2

    @classmethod
    def correction(cls, base_columns=None, tau_scale=.1, **kwargs):
        return cls(kind="correction", base_columns=base_columns, tau_scale=tau_scale, **kwargs)

    subspace = correction

    @classmethod
    def function_penalty(cls, penalty=None, base_columns=None, lambda0=.1,
                         tau_scale=1., **kwargs):
        return cls(kind="function_penalty", penalty=penalty, base_columns=base_columns,
                   lambda0=lambda0, tau_scale=tau_scale, **kwargs)


def rhs_log_variance(log_local, log_tau: float, log_c2: float) -> np.ndarray:
    """Stable log(c² τ² ψ² / (c² + τ² ψ²))."""
    x = 2 * (log_tau + np.asarray(log_local, dtype=float))
    return log_c2 + x - np.logaddexp(log_c2, x)


def half_cauchy_log_coordinate(log_x, scale: float = 1.) -> np.ndarray:
    """Normalized log density with respect to d log(x), including Jacobian."""
    u = np.asarray(log_x, dtype=float)
    return np.log(2/np.pi) - np.log(scale) + u - np.logaddexp(0., 2*(u-np.log(scale)))


def inverse_gamma_log_coordinate(log_x, shape: float, scale) -> np.ndarray:
    u = np.asarray(log_x, dtype=float)
    return shape*np.log(scale) - special.gammaln(shape) - shape*u - np.asarray(scale)*np.exp(-u)


def slice_sample(log_target: Callable[[float], float], current: float,
                 rng: np.random.Generator, width: float = .8,
                 max_steps: int = 40, max_shrink: int = 1000) -> float:
    """Univariate stepping-out slice sampler with randomized finite expansion.

    The left/right expansion budgets sum to max_steps-1, preserving the
    reversibility of Neal's finite-step stepping-out construction.
    """
    if width <= 0 or max_steps < 1:
        raise ValueError("Slice width and expansion budget must be positive")
    value = float(log_target(current))
    if not np.isfinite(value):
        raise ValueError("Slice sampler must start from a finite log density")
    level = value - rng.exponential()
    left = current - width*rng.uniform()
    right = left + width
    J = int(rng.integers(max_steps))
    K = max_steps - 1 - J
    while J and log_target(left) > level:
        left -= width
        J -= 1
    while K and log_target(right) > level:
        right += width
        K -= 1
    for _ in range(max_shrink):
        proposed = float(rng.uniform(left, right))
        if log_target(proposed) >= level:
            return proposed
        if proposed < current:
            left = proposed
        else:
            right = proposed
    raise RuntimeError("Slice shrinkage budget exhausted; no invalid transition was retained")


def sample_gig(order: float, alpha: float, chi: float,
               rng: np.random.Generator, size=None):
    """Draw x^(order−1) exp{−(alpha*x+chi/x)/2}, with explicit convention.

    SciPy's standardized GIG is rescaled by sqrt(chi/alpha).  The legitimate
    chi=0 gamma limit is handled exactly; a nonintegrable boundary is rejected.
    """
    if not np.isfinite(order + alpha + chi) or alpha <= 0 or chi < 0:
        raise ValueError("GIG requires finite order, alpha>0 and chi>=0")
    if chi == 0:
        if order <= 0:
            raise ValueError("The zero-chi GIG boundary is improper for nonpositive order; draw coefficients first")
        return rng.gamma(order, 2/alpha, size=size)
    value = stats.geninvgauss.rvs(order, np.sqrt(alpha*chi),
                                scale=np.sqrt(chi/alpha), size=size, random_state=rng)
    if not np.isfinite(value).all() or np.any(np.asarray(value) <= 0):
        raise FloatingPointError("The GIG generator returned an invalid draw")
    return value


def residualize_correction(reference_Z: np.ndarray, Z: np.ndarray,
                            base_columns: int, rcond: float = 1e-10
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Fit the frozen reference residualization and apply it to supplied rows."""
    ref, design = np.asarray(reference_Z,float), np.asarray(Z,float)
    if ref.ndim != 2 or design.ndim != 2 or ref.shape[1] != design.shape[1]:
        raise ValueError("Reference and application designs must have equal column count")
    if not 0 < base_columns < ref.shape[1] or not 0 < rcond < 1:
        raise ValueError("A correction requires baseline and correction columns and a valid rcond")
    mapping = linalg.lstsq(ref[:,:base_columns],ref[:,base_columns:],cond=rcond)[0]
    transformed = design.copy()
    transformed[:,base_columns:] -= design[:,:base_columns] @ mapping
    return transformed,mapping


def _expand(value, count: int, name: str, positive: bool = False) -> np.ndarray:
    array = np.asarray(value,float)
    if array.ndim == 0:
        array = np.full(count,float(array))
    if array.shape != (count,) or not np.isfinite(array).all() or (positive and np.any(array <= 0)):
        raise ValueError(f"{name} must be a finite {'positive ' if positive else ''}scalar or length-{count} vector")
    return array


def copy_state(state: dict) -> dict:
    return {key: value.copy() if isinstance(value,np.ndarray) else value for key,value in state.items()}


class PriorRuntime:
    """Bind a prior specification to dimensions, groups and a frozen design."""

    def __init__(self, spec: PriorSpec, p: int, d: int, groups: Sequence,
                 Phi: np.ndarray, reference_Z: np.ndarray):
        self.spec,self.p,self.d,self.k = spec,int(p),int(d),int(p*d)
        labels = tuple(str(x) for x in groups)
        if len(labels) != d:
            raise ValueError("A group label is required for each temporal column")
        self.groups = labels
        self.group_names = tuple(dict.fromkeys(labels))
        self.group_index = np.array([self.group_names.index(g) for g in labels],dtype=int)
        self.ngroups = len(self.group_names)
        self.selected_columns = np.array([g not in spec.unshrunk_groups for g in labels])
        self.selected = np.repeat(self.selected_columns,p)
        self.selected_group_names = tuple(g for g in self.group_names if g not in spec.unshrunk_groups)
        self.selected_column_group = np.array([self.selected_group_names.index(g) for g in labels if g not in spec.unshrunk_groups],int)
        self.selected_group = np.repeat(self.selected_column_group,p)
        self.ns = int(self.selected.sum())
        self.Phi,self.reference_Z = np.asarray(Phi,float),np.asarray(reference_Z,float)
        if self.Phi.ndim != 2 or self.Phi.shape[1] != p or self.reference_Z.ndim != 2 or self.reference_Z.shape[1] != d:
            raise ValueError("Reference designs do not match the coefficient dimensions")
        if not np.isfinite(self.Phi).all() or not np.isfinite(self.reference_Z).all():
            raise ValueError("Reference designs must be finite")
        if spec.kind in ("rhs","gigg","reference_r2") and self.ns == 0:
            raise ValueError("This prior requires at least one non-intercept shrinkage coefficient")
        self.separable = spec.kind not in ("gigg","function_penalty") and not (spec.kind=="rhs" and spec.unit=="entry")
        self._layout = []
        if spec.kind in ("ridge","group"):
            count = 1 if spec.kind=="ridge" else self.ngroups
            self.mu = _expand(spec.mean_log_precision,count,"mean_log_precision")
            self.sd = _expand(spec.sd_log_precision,count,"sd_log_precision",True)
        if spec.kind=="rhs":
            if spec.unit=="group":
                self.unit_index = self.selected_group
            elif spec.unit=="column":
                self.unit_index = np.repeat(np.arange(self.selected_columns.sum()),p)
            else:
                self.unit_index = np.arange(self.ns)
            self.nunits = int(self.unit_index.max()+1)
            self.unit_counts = np.bincount(self.unit_index,minlength=self.nunits)
        if spec.kind=="gigg":
            count = len(self.selected_group_names)
            self.ag = _expand(spec.shape_a,count,"shape_a",True)
            self.bg = _expand(spec.shape_b,count,"shape_b",True)
            self.group_counts = np.bincount(self.selected_group,minlength=count)
        if spec.kind=="reference_r2":
            self._prepare_allocation()
        if spec.kind in ("correction","function_penalty"):
            if spec.base_columns is None:
                self.baseline_mask = np.array([g in spec.baseline_groups for g in labels])
            else:
                if spec.base_columns >= d:
                    raise ValueError("The subspace model needs at least one correction column")
                self.baseline_mask = np.arange(d)<spec.base_columns
            if not self.baseline_mask.any() or self.baseline_mask.all():
                raise ValueError("Both a proper baseline and a correction subspace are needed")
        if spec.kind=="function_penalty":
            if self.k > spec.max_penalty_size:
                raise MemoryError("The general function penalty exceeds its explicit dense research gate")
            if spec.penalty is None:
                base = self.reference_Z[:,self.baseline_mask]
                coefficient = linalg.lstsq(base,self.reference_Z,cond=1e-10)[0]
                remainder = self.reference_Z-base@coefficient
                self.F0 = np.kron(remainder.T@remainder,self.Phi.T@self.Phi)/(self.Phi.shape[0]*len(self.reference_Z))
            else:
                self.F0 = np.asarray(spec.penalty,float)
            if self.F0.shape != (self.k,self.k) or not np.isfinite(self.F0).all() or not np.allclose(self.F0,self.F0.T):
                raise ValueError("The proper function penalty must be a finite symmetric k-by-k PSD matrix")
            self.penalty_eigenvalues,self.penalty_vectors = psd_eigh(self.F0)
        # Freeze a deterministic vector layout for checkpointing and MH.
        initial = self.initial_state()
        offset = 0
        for key,value in initial.items():
            shape = np.asarray(value).shape
            size = np.asarray(value).size
            self._layout.append((key,shape,offset,offset+size))
            offset += size
        self.dimension = offset
        self.parameter_names = tuple(name if end-start==1 else f"{name}[{j}]"
                                     for name,shape,start,end in self._layout for j in range(end-start))

    def _prepare_allocation(self):
        count = len(self.selected_group_names)
        self.alpha = _expand(self.spec.alpha,count,"alpha",True)
        Z,Phi = self.reference_Z,self.Phi
        N0 = Z.shape[0]*Phi.shape[0]
        trace_phi = float(np.square(Phi).sum())
        sum_phi = float(np.square(Phi.sum(axis=0)).sum())
        energies = []
        for group in self.selected_group_names:
            z = Z[:,np.array(self.groups)==group]
            energy = (trace_phi*np.square(z).sum() - sum_phi*np.square(z.sum(axis=0)).sum()/N0)/N0
            if not np.isfinite(energy) or energy <= 0:
                raise ValueError(f"Reference group {group!r} has zero centered field energy; remove it or use a separate proper prior")
            energies.append(energy)
        self.reference_energy = np.array(energies)
        dimension = count-1
        self.logistic_mean = np.zeros(dimension) if not len(self.spec.logistic_mean) else _expand(self.spec.logistic_mean,dimension,"logistic_mean")
        self.logistic_cov = np.eye(dimension) if not len(self.spec.logistic_cov) else np.asarray(self.spec.logistic_cov,float)
        if (self.logistic_cov.shape != (dimension,dimension)
                or not np.isfinite(self.logistic_cov).all()
                or not np.allclose(self.logistic_cov,self.logistic_cov.T,rtol=1e-12,atol=1e-13)):
            raise ValueError("Logistic-normal covariance must be finite, symmetric and correctly dimensioned")
        if dimension:
            self.logistic_chol = linalg.cholesky(self.logistic_cov,lower=True)

    def initial_state(self) -> dict:
        s = self.spec
        if s.kind in ("ridge","group"):
            return {"log_precision": self.mu.copy()}
        if s.kind=="rhs":
            state = {"log_local":np.zeros(self.nunits),"log_tau":float(np.log(s.tau0))}
            if not s.fixed_slab:
                state["log_c2"] = float(2*np.log(s.slab_scale))
            return state
        if s.kind=="gigg":
            return {"log_group_variance":np.zeros(len(self.selected_group_names)),
                    "log_local_variance":np.zeros(self.ns),"log_tau2":float(2*np.log(s.tau_scale)),
                    "log_xi":float(-2*np.log(s.tau_scale))}
        if s.kind=="reference_r2":
            logits = self.logistic_mean.copy() if s.allocation=="logistic_normal" else np.log(self.alpha[:-1]/self.alpha[-1])
            return {"logit_r":float(np.log(s.beta_a/s.beta_b)),"allocation_logits":logits}
        return {"log_tau":float(np.log(s.tau_scale))}

    def vector(self,state: dict) -> np.ndarray:
        return np.concatenate([np.asarray(state[key],float).reshape(-1) for key,_,_,_ in self._layout])

    def state_from_vector(self,vector: np.ndarray) -> dict:
        vector = np.asarray(vector,float)
        if vector.shape != (self.dimension,):
            raise ValueError("Prior hyperparameter vector has the wrong length")
        return {key:float(vector[start]) if not shape else vector[start:end].reshape(shape).copy()
                for key,shape,start,end in self._layout}

    def allocation_weights(self,state):
        return special.softmax(np.r_[state["allocation_logits"],0.])

    def precision(self,state: dict,R: np.ndarray | None = None) -> PrecisionSpec:
        s = self.spec
        with np.errstate(over="raise",invalid="raise",divide="raise"):
            if s.kind=="ridge":
                return PrecisionSpec(self.p,self.d,"scalar",np.exp(np.asarray(state["log_precision"])[0]))
            if s.kind=="group":
                return PrecisionSpec(self.p,self.d,"column",np.exp(state["log_precision"])[self.group_index])
            if s.kind=="rhs":
                log_c2 = 2*np.log(s.slab_scale) if s.fixed_slab else state["log_c2"]
                logvar = rhs_log_variance(state["log_local"],state["log_tau"],log_c2)
                diagonal = np.full(self.k,1/s.fixed_variance)
                diagonal[self.selected] = np.exp(-logvar[self.unit_index])
                if self.separable:
                    return PrecisionSpec(self.p,self.d,"column",diagonal.reshape(self.p,self.d,order="F")[0])
                return PrecisionSpec(self.p,self.d,"diagonal",diagonal)
            if s.kind=="gigg":
                diagonal = np.full(self.k,1/s.fixed_variance)
                logvar = state["log_tau2"] + np.asarray(state["log_group_variance"])[self.selected_group] + np.asarray(state["log_local_variance"])
                diagonal[self.selected] = np.exp(-logvar)
                return PrecisionSpec(self.p,self.d,"diagonal",diagonal)
            if s.kind=="reference_r2":
                if R is None:
                    raise ValueError("The reference R2 precision must be recomputed with the current R")
                rbar = float(np.trace(R)/len(R))
                logW = np.log(rbar)+state["logit_r"]
                logpi = special.log_softmax(np.r_[state["allocation_logits"],0.])
                col = np.full(self.d,1/s.fixed_variance)
                col[self.selected_columns] = np.exp(np.log(self.reference_energy)[self.selected_column_group]
                    -logW-logpi[self.selected_column_group])
                return PrecisionSpec(self.p,self.d,"column",col)
            if s.kind=="correction":
                col = np.where(self.baseline_mask,1/s.fixed_variance,np.exp(-2*state["log_tau"]))
                return PrecisionSpec(self.p,self.d,"column",col)
            matrix = s.lambda0*np.eye(self.k)+np.exp(-2*state["log_tau"])*self.F0
            return PrecisionSpec(self.p,self.d,"dense",matrix)

    def log_prior(self,state: dict) -> float:
        """Normalized hyperprior on the explicitly sampled log/logit coordinates."""
        s = self.spec
        try:
            with np.errstate(over="raise",invalid="raise",divide="raise"):
                if s.kind in ("ridge","group"):
                    z = (np.asarray(state["log_precision"])-self.mu)/self.sd
                    result = -.5*np.square(z).sum()-np.log(self.sd).sum()-.5*len(z)*np.log(2*np.pi)
                elif s.kind=="rhs":
                    result = half_cauchy_log_coordinate(state["log_local"]).sum()
                    result += half_cauchy_log_coordinate(state["log_tau"],s.tau0)
                    if not s.fixed_slab:
                        result += inverse_gamma_log_coordinate(state["log_c2"],s.slab_df/2,s.slab_df*s.slab_scale**2/2)
                elif s.kind=="gigg":
                    v,w = np.asarray(state["log_group_variance"]),np.asarray(state["log_local_variance"])
                    result = np.sum(self.ag*v-np.exp(v)-special.gammaln(self.ag))
                    result += np.sum(-special.gammaln(self.bg[self.selected_group])-self.bg[self.selected_group]*w-np.exp(-w))
                    result += inverse_gamma_log_coordinate(state["log_tau2"],.5,np.exp(-state["log_xi"]))
                    result += inverse_gamma_log_coordinate(state["log_xi"],.5,1/s.tau_scale**2)
                elif s.kind=="reference_r2":
                    log_r = -np.logaddexp(0.,-state["logit_r"])
                    log_1mr = -np.logaddexp(0.,state["logit_r"])
                    result = s.beta_a*log_r+s.beta_b*log_1mr-special.betaln(s.beta_a,s.beta_b)
                    logits = np.asarray(state["allocation_logits"])
                    if s.allocation=="dirichlet":
                        logpi = special.log_softmax(np.r_[logits,0.])
                        result += special.gammaln(self.alpha.sum())-special.gammaln(self.alpha).sum()+self.alpha@logpi
                    elif len(logits):
                        white = linalg.solve_triangular(self.logistic_chol,logits-self.logistic_mean,lower=True)
                        result += -.5*float(white@white)-np.log(np.diag(self.logistic_chol)).sum()-.5*len(logits)*np.log(2*np.pi)
                else:
                    result = half_cauchy_log_coordinate(state["log_tau"],s.tau_scale)
                return float(result) if np.isfinite(result) else -np.inf
        except (FloatingPointError,ValueError,OverflowError):
            return -np.inf

    def coefficient_log_density(self,C,sigma2,state,R=None) -> float:
        precision = self.precision(state,R)
        return float(.5*precision.logdet-.5*self.k*np.log(2*np.pi*sigma2)-.5*precision.quadratic(C)/sigma2)

    def conditional_log_target(self,C,sigma2,state,R=None) -> float:
        try:
            return self.coefficient_log_density(C,sigma2,state,R)+self.log_prior(state)
        except (FloatingPointError,ValueError,linalg.LinAlgError):
            return -np.inf

    def rhs_tradeoff(self,state: dict,rng: np.random.Generator,scale: float = .5) -> tuple[dict,bool]:
        """Mix the RHS global/local direction at no likelihood cost.

        The symmetric translation log(tau)+=delta, log(psi_r)-=delta
        preserves every tau² psi_r² and hence the complete coefficient prior.
        Its coordinate Jacobian is one.  Only the hyperprior ratio enters MH;
        there is no approximation to either collapsed or conditional targets.
        """
        if self.spec.kind!="rhs":
            raise ValueError("The global/local tradeoff kernel is specific to RHS")
        if not np.isfinite(scale) or scale<0:
            raise ValueError("Tradeoff proposal scale must be finite and nonnegative")
        current = copy_state(state)
        if scale==0:
            return current,False
        delta = float(rng.normal(scale=scale))
        proposed = copy_state(state)
        proposed["log_tau"] = float(state["log_tau"]+delta)
        proposed["log_local"] = np.asarray(state["log_local"])-delta
        accepted = bool(np.log(rng.uniform())<self.log_prior(proposed)-self.log_prior(current))
        return (proposed if accepted else current),accepted

    def update_conditional(self,C: np.ndarray,sigma2: float,state: dict,
                           rng: np.random.Generator,R: np.ndarray | None = None,
                           rhs_tradeoff_scale: float = .5) -> dict:
        """One complete, valid prior block conditional on the current C and scale."""
        state = copy_state(state)
        b = np.asarray(C,float).reshape(-1,order="F")
        if b.shape != (self.k,) or sigma2<=0:
            raise ValueError("Invalid coefficient or variance state")
        if self.spec.kind=="gigg":
            return self._update_gigg(b,sigma2,state,rng)
        if self.spec.kind=="rhs":
            updated = self._update_rhs(b,sigma2,state,rng)
            return self.rhs_tradeoff(updated,rng,rhs_tradeoff_scale)[0]
        vector = self.vector(state)
        for j in range(len(vector)):
            def target(value,j=j):
                proposed = vector.copy()
                proposed[j] = value
                return self.conditional_log_target(C,sigma2,self.state_from_vector(proposed),R)
            vector[j] = slice_sample(target,float(vector[j]),rng)
        return self.state_from_vector(vector)

    def _update_rhs(self,b,sigma2,state,rng):
        s = self.spec
        sums = np.bincount(self.unit_index,weights=np.square(b[self.selected]),minlength=self.nunits)/sigma2
        local = np.array(state["log_local"])
        t = float(state["log_tau"])
        z = 2*np.log(s.slab_scale) if s.fixed_slab else float(state["log_c2"])
        def likelihood(logvars,counts,sumsq):
            with np.errstate(over="ignore",invalid="ignore"):
                ans = -.5*np.sum(counts*logvars+sumsq*np.exp(-logvars))
            return float(ans) if np.isfinite(ans) else -np.inf
        for j in range(self.nunits):
            def local_target(value,j=j):
                lv = rhs_log_variance(value,t,z)
                return likelihood(lv,self.unit_counts[j],sums[j])+float(half_cauchy_log_coordinate(value))
            local[j] = slice_sample(local_target,float(local[j]),rng)
        def tau_target(value):
            return likelihood(rhs_log_variance(local,value,z),self.unit_counts,sums)+float(half_cauchy_log_coordinate(value,s.tau0))
        t = slice_sample(tau_target,t,rng)
        if not s.fixed_slab:
            def slab_target(value):
                with np.errstate(over="ignore"):
                    hp = inverse_gamma_log_coordinate(value,s.slab_df/2,s.slab_df*s.slab_scale**2/2)
                return likelihood(rhs_log_variance(local,t,value),self.unit_counts,sums)+float(hp)
            z = slice_sample(slab_target,z,rng)
            state["log_c2"] = z
        state["log_local"],state["log_tau"] = local,t
        return state

    def _update_gigg(self,b,sigma2,state,rng):
        sq = np.square(b[self.selected])
        tau2,xi = np.exp(state["log_tau2"]),np.exp(state["log_xi"])
        vg = np.exp(state["log_group_variance"])
        scales = 1+sq/(2*sigma2*tau2*vg[self.selected_group])
        w = scales/rng.gamma(self.bg[self.selected_group]+.5)
        for group in range(len(vg)):
            mask = self.selected_group==group
            chi = float(np.sum(sq[mask]/(sigma2*tau2*w[mask])))
            vg[group] = sample_gig(self.ag[group]-.5*self.group_counts[group],2.,chi,rng)
        scale = 1/xi+.5*np.sum(sq/(vg[self.selected_group]*w))/sigma2
        tau2 = float(scale/rng.gamma((self.ns+1)/2))
        xi = float((1/self.spec.tau_scale**2+1/tau2)/rng.gamma(1.))
        return {"log_group_variance":np.log(vg),"log_local_variance":np.log(w),
                "log_tau2":float(np.log(tau2)),"log_xi":float(np.log(xi))}
