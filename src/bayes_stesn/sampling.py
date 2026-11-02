"""Reproducible MCMC schedules and deferred joint posterior recovery.

Separable Gaussian models use normalized collapsed likelihoods.  Entrywise
priors use ordinary Gibbs/MH, storing variance and hyperstates rather than
every coefficient matrix.  Checkpoints include RNG, proposal adaptation,
current coefficients when needed, and already retained draws.  Resumption is
bit-for-bit within the same numerical runtime; cross-platform BLAS equality
is not promised.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import linalg, optimize, special

from .gaussian import GaussianRegression, OperatorGaussianRegression
from .priors import PriorRuntime, PriorSpec, copy_state
from .spatial import CovarianceSpec


@dataclass(frozen=True)
class SamplerConfig:
    chains: int = 4
    warmup: int = 500
    draws: int = 1000
    seed: int = 2026
    backend: str = "auto"
    checkpoint_every: int = 100
    proposal_scale: float = .12
    target_accept: float = .3
    adapt_interval: int = 25
    solver_rtol: float = 1e-10
    solver_atol: float = 0.
    solver_maxiter: int | None = None
    dense_threshold: int = 900
    max_dense_bytes: int = 256*1024**2
    max_stored_hyperstate_bytes: int = 512*1024**2
    thin: int = 1
    initial_jitter: float = .15
    a0: float = 2.
    b0: float = .05
    temporal: str = "iid"
    rho_prior_sd: float = .8
    algorithm: str = "auto"
    rhs_tradeoff_scale: float = .5
    initialize: str = "auto"
    init_maxiter: int = 100
    init_max_evaluations: int = 2000
    init_max_dimension: int = 20
    init_dispersion: float = 2.

    def __post_init__(self):
        for key in ("chains","draws","adapt_interval","dense_threshold","max_dense_bytes",
                    "max_stored_hyperstate_bytes","thin"):
            if not isinstance(getattr(self,key),(int,np.integer)) or getattr(self,key)<1:
                raise ValueError(f"{key} must be a positive integer")
        if self.warmup<0 or self.checkpoint_every<0 or self.seed<0:
            raise ValueError("Warmup, checkpoint interval and seed must be nonnegative")
        if self.temporal not in ("iid","full_residual_ar"):
            raise ValueError("sample_regression supports iid or full_residual_ar; augmented models use sample_augmented")
        if self.algorithm not in ("auto","collapsed","conditional"):
            raise ValueError("algorithm must be auto, collapsed or conditional")
        if self.initialize not in ("auto","prior"):
            raise ValueError("initialize must be auto or prior")
        if self.backend not in ("auto","spectral_gaussian","temporal_whitened_spectral",
                                "gaussian_matrixfree","matrixfree","dense",
                                "weighted_spectral","latent_ffbs","vecchia_operator"):
            raise ValueError("Unknown inference backend")
        if not 0 < self.target_accept < 1:
            raise ValueError("target_accept must lie strictly between zero and one")
        for key in ("proposal_scale","solver_rtol","a0","b0","rho_prior_sd"):
            if not np.isfinite(getattr(self,key)) or getattr(self,key)<=0:
                raise ValueError(f"{key} must be positive and finite")
        if self.initial_jitter<0 or self.solver_atol<0:
            raise ValueError("Initial jitter and absolute solver tolerance cannot be negative")
        if self.solver_maxiter is not None and self.solver_maxiter<1:
            raise ValueError("solver_maxiter must be positive when supplied")
        if not np.isfinite(self.rhs_tradeoff_scale) or self.rhs_tradeoff_scale<0:
            raise ValueError("rhs_tradeoff_scale must be finite and nonnegative")
        for key in ("init_maxiter","init_max_evaluations","init_max_dimension"):
            if not isinstance(getattr(self,key),(int,np.integer)) or getattr(self,key)<1:
                raise ValueError(f"{key} must be a positive integer")
        if not np.isfinite(self.init_dispersion) or self.init_dispersion<=0:
            raise ValueError("init_dispersion must be finite and positive")


def _vecchia_operator_route(covariance, prior, config) -> bool:
    """The deliberately narrow, validated spatially sparse inference route."""
    supported = (covariance.kind == "vecchia" and prior.kind == "ridge" and
                 config.temporal == "iid" and config.algorithm in ("auto", "collapsed") and
                 config.backend in ("auto", "spectral_gaussian", "vecchia_operator"))
    if config.backend == "vecchia_operator" and not supported:
        raise ValueError("vecchia_operator requires Vecchia covariance, ridge prior, iid Gaussian "
                         "innovations and collapsed inference; other combinations use the "
                         "size-gated dense reference backend")
    return supported


@dataclass
class ParameterDraw:
    C: np.ndarray
    sigma2: float
    covariance_parameters: dict
    prior_state: dict
    temporal_parameters: dict
    solver_metadata: dict
    latent_state: Any = None


@dataclass
class RegressionPosterior:
    Y: np.ndarray
    Phi: np.ndarray
    Z: np.ndarray
    coordinates: np.ndarray
    groups: Sequence
    prior: PriorSpec
    covariance: CovarianceSpec
    config: SamplerConfig
    states: list[list[dict]]
    log_density: np.ndarray
    acceptance: np.ndarray
    metadata: dict
    reference_Z: np.ndarray | None = None

    def __post_init__(self):
        self.Y,self.Phi,self.Z,self.coordinates = (np.asarray(x,float) for x in
                                                  (self.Y,self.Phi,self.Z,self.coordinates))
        self.log_density,self.acceptance = np.asarray(self.log_density,float),np.asarray(self.acceptance,float)
        self.groups = tuple(str(g) for g in self.groups)
        if not self.states or any(not chain for chain in self.states):
            raise ValueError("A posterior requires at least one retained state in each chain")
        lengths = {len(chain) for chain in self.states}
        if len(lengths)!=1 or self.log_density.shape!=(len(self.states),len(self.states[0])):
            raise ValueError("Posterior states and log-density dimensions disagree")
        self._runtime = PriorRuntime(self.prior,self.Phi.shape[1],self.Z.shape[1],self.groups,
                                     self.Phi,self.Z if self.reference_Z is None else self.reference_Z)
        self._engine = None
        self._spatial_geometry = None

    @property
    def n_chains(self):
        return len(self.states)

    @property
    def n_draws(self):
        return len(self.states[0])

    def _scalar_layout(self):
        first = self.states[0][0]
        names = ["prior."+name for name in self._runtime.parameter_names]
        names.extend(self.covariance.parameter_names(self.coordinates))
        temporal = first.get("temporal_parameters",{})
        tkeys = []
        for key in sorted(temporal):
            value = temporal[key]
            # rho and nu are useful natural-coordinate diagnostics; retaining
            # both them and their transformed coordinates is intentional.
            if key.startswith(("variance_covariate_center","variance_covariate_scale")):
                continue
            arr = np.asarray(value)
            if np.issubdtype(arr.dtype,np.number):
                tkeys.append((key,arr.size))
                names.extend(["temporal."+key] if arr.size==1 else
                             [f"temporal.{key}[{j}]" for j in range(arr.size)])
        if self.covariance.infer_nu:
            names.append("covariance.nu")
        if "sigma2" in first:
            names.append("sigma2")
        return tuple(names),tkeys,"sigma2" in first

    @property
    def parameter_names(self) -> tuple[str,...]:
        return self._scalar_layout()[0]

    @property
    def chains(self) -> np.ndarray:
        """Scalar hyperparameter diagnostics in (chain, draw, parameter) order."""
        names,tkeys,has_sigma = self._scalar_layout()
        out = np.empty((self.n_chains,self.n_draws,len(names)))
        for c,chain in enumerate(self.states):
            for j,state in enumerate(chain):
                parts = [self._runtime.vector(state["prior_state"]),
                         self.covariance.parameters_to_vector(state["covariance_parameters"],self.coordinates)]
                parts.extend(np.asarray(state["temporal_parameters"][key]).reshape(-1) for key,_ in tkeys)
                if self.covariance.infer_nu:
                    parts.append(np.array([state["covariance_parameters"]["nu"]]))
                if has_sigma:
                    parts.append(np.array([state["sigma2"]]))
                out[c,j] = np.concatenate(parts)
        return out

    def draw_parameters(self,chain: int,draw: int,rng: np.random.Generator) -> ParameterDraw:
        """Recover one *joint* parameter draw; keep it fixed for a whole path."""
        if not isinstance(rng,np.random.Generator):
            raise TypeError("An explicit NumPy Generator is required")
        state = self.states[chain][draw]
        cov = copy.deepcopy(state["covariance_parameters"])
        prior_state = copy_state(state["prior_state"])
        temporal = copy.deepcopy(state.get("temporal_parameters",{}))
        if "C" in state:
            return ParameterDraw(np.array(state["C"],copy=True),float(state["sigma2"]),cov,
                                 prior_state,temporal,copy.deepcopy(state.get("solver_metadata",{})),
                                 copy.deepcopy(state.get("latent_state")))
        operator_route = self.metadata.get("spatial_backend") == "vecchia_operator"
        if self._engine is None:
            engine_type = OperatorGaussianRegression if operator_route else GaussianRegression
            self._engine = engine_type(self.Y,self.Phi,self.Z,self.config.a0,self.config.b0,
                                      self.config.dense_threshold,self.config.max_dense_bytes)
        if operator_route:
            if self._spatial_geometry is None:
                self._spatial_geometry = self.covariance.prepare_geometry(self.coordinates)
            R = self._spatial_geometry.operator(cov)
        else:
            R = self.covariance.covariance(self.coordinates,cov)
        precision = self._runtime.precision(prior_state,R)
        rho = temporal.get("rho") if self.config.temporal=="full_residual_ar" else None
        fixed = float(np.exp(2*cov["log_sigma_e"])) if self.covariance.scale_mode=="separate" else None
        if "sigma2" in state:
            sigma2 = float(state["sigma2"]) if fixed is None else fixed
            backend = self.config.backend if self.config.backend in ("dense","matrixfree","gaussian_matrixfree") else "auto"
            C,meta = self._engine.draw_conditional(R,precision,sigma2,rng,backend,
                self.config.solver_rtol,self.config.solver_atol,self.config.solver_maxiter,rho)
        else:
            evaluation = self._engine.evaluate(R,precision=precision,sigma2=fixed,temporal_rho=rho)
            C,sigma2,meta = evaluation.draw(rng)
        if operator_route:
            meta.update(spatial_solver="sparse_lu", spatial_backend="vecchia_operator",
                        precision_nonzeros=R.metadata["precision_nonzeros"],
                        lu_nonzeros=R.metadata["lu_nonzeros"])
        return ParameterDraw(C,sigma2,cov,prior_state,temporal,meta,copy.deepcopy(state.get("latent_state")))

    def to_state(self) -> dict:
        """Numeric/JSON-safe posterior payload; designs and specs are rebound."""
        return {"schema_version":1,"states":copy.deepcopy(self.states),
                "log_density":self.log_density.copy(),"acceptance":self.acceptance.copy(),
                "metadata":copy.deepcopy(self.metadata)}

    @classmethod
    def from_state(cls,state: dict,Y,Phi,Z,coordinates,groups,prior,covariance,config,reference_Z=None):
        if state.get("schema_version")!=1:
            raise ValueError("Unsupported RegressionPosterior state schema")
        return cls(Y,Phi,Z,coordinates,groups,prior,covariance,config,copy.deepcopy(state["states"]),
                   np.asarray(state["log_density"]),np.asarray(state["acceptance"]),
                   copy.deepcopy(state["metadata"]),reference_Z)


class _AdaptiveProposal:
    def __init__(self,dimension: int,config: SamplerConfig,state=None,initial_covariance=None):
        self.dimension,self.config = dimension,config
        if state is None:
            self.count = 0
            self.mean = np.zeros(dimension)
            self.M2 = np.zeros((dimension,dimension))
            if initial_covariance is None:
                self.base_covariance = np.eye(dimension)*config.proposal_scale**2*dimension/2.38**2
                self.factor = np.eye(dimension)*config.proposal_scale
            else:
                self.base_covariance = np.array(initial_covariance,float)
                self.factor = (2.38/np.sqrt(dimension))*linalg.cholesky(self.base_covariance,lower=True)
            self.log_scale = 0.
        else:
            self.count = int(state["count"])
            self.mean = np.array(state["mean"])
            self.M2 = np.array(state["M2"])
            self.factor = np.array(state["factor"])
            self.base_covariance = np.array(state["base_covariance"])
            self.log_scale = float(state["log_scale"])

    def propose(self,current,rng):
        return current+np.exp(self.log_scale)*(self.factor@rng.normal(size=self.dimension))

    def adapt(self,value,accepted):
        self.count += 1
        delta = value-self.mean
        self.mean += delta/self.count
        self.M2 += np.outer(delta,value-self.mean)
        gain = (self.count+10.)**(-.6)
        self.log_scale = float(np.clip(self.log_scale+gain*(float(accepted)-self.config.target_accept),-5.,4.))
        if self.count>=max(25,2*self.dimension) and self.count%self.config.adapt_interval==0:
            empirical = self.M2/(self.count-1)
            covariance = (2.38**2/self.dimension)*(.95*empirical+.05*self.base_covariance)
            self.factor = linalg.cholesky((covariance+covariance.T)/2,lower=True)

    def state(self):
        return {"count":self.count,"mean":self.mean.copy(),"M2":self.M2.copy(),
                "factor":self.factor.copy(),"base_covariance":self.base_covariance.copy(),
                "log_scale":self.log_scale}


def _calibrate_initialization(log_target,initial: np.ndarray,config: SamplerConfig) -> dict:
    """Bounded deterministic mode/curvature search used only to initialize MCMC.

    No optimizer bound or Hessian regularization changes the posterior target.
    The BFGS evaluation budget is separate from the bounded O(d²) curvature
    check.  An incomplete search keeps its best finite point and reports that
    limitation.  This is a local numerical calibration, not evidence of a
    global mode or converged posterior chains.
    """
    initial = np.asarray(initial,float)
    dimension = len(initial)
    best_x = initial.copy()
    best_log = float(log_target(initial))
    if not np.isfinite(best_log):
        raise ValueError("Mode initialization requires a finite prior-center target")
    calls = 0
    class BudgetExhausted(Exception):
        pass
    def objective(x):
        nonlocal calls,best_x,best_log
        if calls>=config.init_max_evaluations:
            raise BudgetExhausted()
        calls += 1
        value = float(log_target(x))
        if np.isfinite(value) and value>best_log:
            best_x,best_log = np.array(x),value
        return -value if np.isfinite(value) else 1e100
    success,message,iterations = False,"Evaluation budget exhausted",0
    try:
        result = optimize.minimize(objective,initial,method="BFGS",jac="3-point",
            options={"maxiter":config.init_maxiter,"gtol":1e-4,"finite_diff_rel_step":1e-4})
        success,message,iterations = bool(result.success),str(result.message),int(result.nit)
    except BudgetExhausted:
        pass
    # Central differences are large enough to avoid cancellation of large
    # log-likelihood constants, while staying local in unconstrained space.
    steps = .002*np.maximum(1.,abs(best_x))
    Hessian = np.empty((dimension,dimension))
    center = -float(log_target(best_x))
    curvature_calls = 1
    for i in range(dimension):
        ei = np.zeros(dimension)
        ei[i] = steps[i]
        plus,minus = -float(log_target(best_x+ei)),-float(log_target(best_x-ei))
        curvature_calls += 2
        Hessian[i,i] = (plus-2*center+minus)/steps[i]**2
        for j in range(i):
            ej = np.zeros(dimension)
            ej[j] = steps[j]
            terms = [-float(log_target(best_x+ei+ej)),-float(log_target(best_x+ei-ej)),
                     -float(log_target(best_x-ei+ej)),-float(log_target(best_x-ei-ej))]
            curvature_calls += 4
            Hessian[i,j] = Hessian[j,i] = (terms[0]-terms[1]-terms[2]+terms[3])/(4*steps[i]*steps[j])
    if np.isfinite(Hessian).all():
        eigenvalues,vectors = linalg.eigh((Hessian+Hessian.T)/2)
        floor = max(1e-4,1e-6*float(np.max(abs(eigenvalues))))
        regularized = np.maximum(eigenvalues,floor)
        covariance = (vectors/regularized)@vectors.T
        covariance = (covariance+covariance.T)/2
        fallback = False
    else:
        eigenvalues = np.zeros(dimension)
        floor = 1/config.proposal_scale**2
        regularized = np.full(dimension,floor)
        covariance = np.eye(dimension)*config.proposal_scale**2
        fallback = True
    return {"method":"local_map_finite_difference_laplace","mode":best_x,
            "mode_log_target":best_log,"covariance":covariance,
            "optimizer_success":success,"optimizer_message":message,"optimizer_iterations":iterations,
            "optimizer_evaluations":calls,"optimizer_evaluation_budget":config.init_max_evaluations,
            "curvature_evaluations":curvature_calls,"curvature_eigenvalues":eigenvalues,
            "curvature_floor":floor,"curvature_regularized":bool(np.any(eigenvalues<floor)),
            "curvature_fallback":fallback,"scope":"initialization only; posterior support and target unchanged"}


class _ResidualStatistics:
    def __init__(self,E,ar):
        self.T = E.shape[1]
        self.EE = E@E.T
        if ar:
            cur,prev = E[:,1:],E[:,:-1]
            self.first = np.outer(E[:,0],E[:,0])
            self.cur,self.prev,self.lag = cur@cur.T,prev@prev.T,cur@prev.T+prev@cur.T

    def get(self,rho):
        if rho==0:
            return self.EE,0.
        if not np.isfinite(rho) or abs(rho)>=1:
            raise ValueError("Invalid stationary AR parameter")
        return (self.first+(self.cur+rho*rho*self.prev-rho*self.lag)/(1-rho*rho),
                (self.T-1)*float(np.log1p(-rho*rho)))


def _signature(Y,Phi,Z,coordinates,groups,prior,covariance,config,reference_Z):
    digest = hashlib.sha256()
    if _vecchia_operator_route(covariance, prior, config):
        # Operator arithmetic has a different floating-point path from the
        # former dense reference implementation. Never silently resume one
        # algorithm's checkpoint using the other algorithm.
        digest.update(b"vecchia-operator-regression-v1")
    for array in (Y,Phi,Z,coordinates,reference_Z):
        if array is not None:
            a = np.ascontiguousarray(array,dtype=float)
            digest.update(str(a.shape).encode())
            digest.update(memoryview(a).cast("B"))
    cfg = asdict(config)
    # Extending the number of retained draws and changing checkpoint cadence
    # leaves the transition kernel and its deterministic continuation intact.
    cfg.pop("draws")
    cfg.pop("checkpoint_every")
    def convert(x):
        if isinstance(x,np.ndarray):
            return x.tolist()
        if isinstance(x,np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    digest.update(json.dumps([list(groups),asdict(prior),asdict(covariance),cfg],
                             sort_keys=True,default=convert).encode())
    return digest.hexdigest()


def _resolve_resume(resume_state,chain):
    if resume_state is None:
        return None
    if isinstance(resume_state,Mapping):
        if "chain_index" in resume_state:
            return resume_state if int(resume_state["chain_index"])==chain else None
        payload = resume_state.get("chains",resume_state)
        if isinstance(payload,Mapping):
            return payload.get(chain,payload.get(str(chain)))
        return payload[chain] if chain<len(payload) else None
    return resume_state[chain] if chain<len(resume_state) else None


def sample_regression(Y: np.ndarray,Phi: np.ndarray,Z: np.ndarray,coordinates: np.ndarray,
                      groups: Sequence,prior: PriorSpec,covariance: CovarianceSpec,
                      config: SamplerConfig,reference_Z: np.ndarray | None = None,
                      checkpoint_callback: Callable[[int,dict],None] | None = None,
                      resume_state: Mapping | None = None) -> RegressionPosterior:
    """Sample Gaussian or full-residual AR readout posterior.

    The callback receives ``(chain_index, checkpoint)``.  Pass a mapping from
    chain indices to these checkpoints as ``resume_state``.  Data, priors,
    transition settings and warmup must match their SHA-256 signature.  The
    requested draw count may increase.  Adaptation ends exactly at warmup.

    Auto selects collapsed inference for separable priors with at most 16
    prior hyperparameters.  Wider column RHS uses its efficient conditional
    local-scale sweep; entrywise priors always use ordinary conditional MCMC.
    Explicit ``algorithm='collapsed'`` can select a separable wide model.
    ``initialize='auto'`` calibrates a local MAP/Laplace proposal for collapsed
    blocks of at most init_max_dimension parameters. This bounded search is
    solely initialization; priors and MH support remain unrestricted.
    """
    operator_route = _vecchia_operator_route(covariance, prior, config)
    engine_type = OperatorGaussianRegression if operator_route else GaussianRegression
    engine = engine_type(Y,Phi,Z,config.a0,config.b0,config.dense_threshold,config.max_dense_bytes)
    Y,Phi,Z = engine.Y,engine.Phi,engine.Z
    coords = np.asarray(coordinates,float)
    if coords.ndim!=2 or coords.shape[0]!=engine.n or not np.isfinite(coords).all():
        raise ValueError("Coordinates require a finite row for every response site")
    dense_cap = getattr(covariance, "max_dense_sites", None)
    if not operator_route and dense_cap is not None and engine.n > dense_cap:
        raise ValueError("This model combination uses a dense spatial reference likelihood, "
                         f"limited to {dense_cap} sites. The spatial operator "
                         "route requires iid Gaussian ridge collapsed Vecchia inference.")
    spatial_geometry = covariance.prepare_geometry(coords) if operator_route else None
    reference = Z if reference_Z is None else np.asarray(reference_Z,float)
    runtime = PriorRuntime(prior,engine.p,engine.d,groups,Phi,reference)
    ar = config.temporal=="full_residual_ar"
    if config.backend in ("weighted_spectral","latent_ffbs"):
        raise ValueError("The requested backend requires the explicit augmented likelihood sampler")
    if config.backend=="temporal_whitened_spectral" and not ar:
        raise ValueError("temporal_whitened_spectral requires full_residual_ar")
    if config.algorithm=="collapsed":
        if not runtime.separable:
            raise ValueError("This prior requires ordinary conditional inference; generic collapsed determinants are not approximated")
        collapsed = True
    elif config.algorithm=="conditional":
        collapsed = False
    else:
        collapsed = runtime.separable and runtime.dimension<=16 and config.backend not in ("gaussian_matrixfree","matrixfree","dense")
    if config.backend in ("spectral_gaussian","temporal_whitened_spectral") and not runtime.separable:
        raise ValueError("An entrywise/general precision cannot use the separable spectral backend")
    if collapsed and config.backend in ("gaussian_matrixfree","matrixfree","dense"):
        raise ValueError("A coefficient-sampling backend is incompatible with fully collapsed inference")
    expected = config.chains*config.draws*(runtime.dimension+len(covariance.parameter_names(coords))+8)*8
    if expected>config.max_stored_hyperstate_bytes:
        raise MemoryError(f"Retained hyperstates need approximately {expected/1024**2:.1f} MiB; reduce retained draws or prior dimension")
    npdim = runtime.dimension
    ncdim = len(covariance.parameter_names(coords))
    dimension = (npdim if collapsed else 0)+ncdim+int(ar)
    separate = covariance.scale_mode=="separate"
    sig = _signature(Y,Phi,Z,coords,groups,prior,covariance,config,reference_Z)
    total_iterations = config.warmup+config.draws*config.thin
    all_states,all_density,all_acceptance = [],[],[]
    invalid_counts,solver_records,final_checkpoints,initial_states = [],[],[],[]

    def unpack(vector,nu,ps=None):
        if collapsed:
            ps = runtime.state_from_vector(vector[:npdim])
            start = npdim
        else:
            start = 0
        cp = covariance.vector_to_parameters(vector[start:start+ncdim],coords,nu=nu)
        eta = float(vector[-1]) if ar else 0.
        rho = float(np.tanh(eta)) if ar else 0.
        tp = {"eta_rho":eta,"rho":rho} if ar else {}
        return ps,cp,tp,rho

    def hyperprior(ps,cp,tp):
        ans = runtime.log_prior(ps)+covariance.log_prior(cp,coords)
        if ar:
            eta = tp["eta_rho"]
            ans += -.5*(eta/config.rho_prior_sd)**2-np.log(config.rho_prior_sd)-.5*np.log(2*np.pi)
        return float(ans)

    def collapsed_target(vector,nu):
        try:
            with np.errstate(over="raise",invalid="raise",divide="raise"):
                ps,cp,tp,rho = unpack(vector,nu)
                hp = hyperprior(ps,cp,tp)
                if not np.isfinite(hp):
                    return -np.inf,None
                R = spatial_geometry.operator(cp) if operator_route else covariance.covariance(coords,cp)
                precision = runtime.precision(ps,R)
                fixed = float(np.exp(2*cp["log_sigma_e"])) if separate else None
                evaluation = engine.evaluate(R,precision=precision,sigma2=fixed,temporal_rho=rho)
                value = evaluation.log_marginal+hp
                return (float(value),evaluation) if np.isfinite(value) else (-np.inf,None)
        except (FloatingPointError,ValueError,linalg.LinAlgError,OverflowError):
            return -np.inf,None

    # Reuse checkpointed deterministic calibration when extending chains.
    # Otherwise calibrate once per fit, never independently at each draw.
    initialization = None
    for chain in range(config.chains):
        saved = _resolve_resume(resume_state,chain)
        if saved is not None:
            if saved.get("signature")!=sig or saved.get("schema_version")!=1:
                raise ValueError("Checkpoint schema, data or transition configuration does not match")
            if initialization is None:
                initialization = copy.deepcopy(saved.get("initialization"))
    if initialization is None:
        if collapsed and config.initialize=="auto" and dimension<=config.init_max_dimension:
            initial_prior = runtime.initial_state()
            initial_cov = covariance.initial_parameters(coords)
            center = np.r_[runtime.vector(initial_prior),covariance.parameters_to_vector(initial_cov,coords),[0.] if ar else []]
            initialization = _calibrate_initialization(
                lambda x:collapsed_target(x,float(initial_cov["nu"]))[0],center,config)
        else:
            initialization = {"method":"prior_center_with_jitter",
                "reason":"explicit_prior" if config.initialize=="prior" else
                         ("conditional_schedule" if not collapsed else "dimension_gate"),
                "scope":"initialization only; posterior support and target unchanged"}

    for chain in range(config.chains):
        rng = np.random.default_rng(np.random.SeedSequence(config.seed,spawn_key=(chain,)))
        restored = _resolve_resume(resume_state,chain)
        invalid = 0
        if restored is None:
            ps = runtime.initial_state()
            cp = covariance.initial_parameters(coords)
            nu = float(cp["nu"])
            vector = np.r_[runtime.vector(ps) if collapsed else [],covariance.parameters_to_vector(cp,coords),[0.] if ar else []]
            initial_covariance = None
            if collapsed and initialization["method"]=="local_map_finite_difference_laplace":
                initial_covariance = np.asarray(initialization["covariance"])
                initial_chol = linalg.cholesky(initial_covariance,lower=True)
                mode = np.asarray(initialization["mode"])
                # Overdispersed starts in calibrated units. Very remote
                # numerical starts are retried; this is not a target bound.
                for attempt in range(30):
                    dispersion = config.init_dispersion*(1. if attempt<15 else .25)
                    vector = mode+dispersion*(initial_chol@rng.normal(size=dimension))
                    current,evaluation = collapsed_target(vector,nu)
                    if np.isfinite(current) and current>=initialization["mode_log_target"]-25*dimension:
                        break
                else:
                    vector = mode.copy()
            else:
                vector += config.initial_jitter*rng.normal(size=dimension)
            if collapsed:
                current,evaluation = collapsed_target(vector,nu)
                if not np.isfinite(current):
                    vector = np.r_[runtime.vector(ps),covariance.parameters_to_vector(cp,coords),[0.] if ar else []]
                    current,evaluation = collapsed_target(vector,nu)
                if not np.isfinite(current):
                    raise ValueError("The configured model has no finite initial collapsed density")
                C,sigma2 = None,None
            else:
                _,cp,_,rho = unpack(vector,nu,ps)
                R = covariance.covariance(coords,cp)
                precision = runtime.precision(ps,R)
                sigma2 = float(np.exp(2*cp["log_sigma_e"])) if separate else max(float(np.var(Y)),config.b0/config.a0)
                C = np.zeros((engine.p,engine.d))
                current = np.nan
            proposal = _AdaptiveProposal(dimension,config,initial_covariance=initial_covariance)
            initial_vector = vector.copy()
            start_iteration = 0
            retained,densities = [],[]
            accepted_total,accepted_sampling,proposals_sampling = 0,0,0
        else:
            if restored.get("schema_version")!=1 or restored.get("signature")!=sig or restored.get("chain_index")!=chain:
                raise ValueError("Checkpoint schema, chain identity, data or transition configuration does not match")
            if restored.get("schedule")!=("collapsed" if collapsed else "conditional"):
                raise ValueError("Cannot resume a different MCMC schedule")
            rng.bit_generator.state = copy.deepcopy(restored["rng_state"])
            vector = np.array(restored["vector"],float)
            nu = float(restored["nu"])
            ps = copy_state(restored["prior_state"])
            C = None if restored["C"] is None else np.array(restored["C"],float)
            sigma2 = restored["sigma2"]
            proposal = _AdaptiveProposal(dimension,config,restored["proposal"])
            initial_vector = np.array(restored["initial_vector"],float)
            start_iteration = int(restored["iteration"])
            retained = copy.deepcopy(restored["retained_states"])
            densities = list(restored["retained_log_density"])
            accepted_total = int(restored["accepted_total"])
            accepted_sampling = int(restored["accepted_sampling"])
            proposals_sampling = int(restored["proposals_sampling"])
            invalid = int(restored.get("invalid_proposals",0))
            if start_iteration>total_iterations or len(retained)>config.draws:
                raise ValueError("Checkpoint contains more iterations than the requested target")
            if collapsed:
                _,evaluation = collapsed_target(vector,nu)
                current = float(restored["current_log_target"])
            else:
                current = np.nan

        def checkpoint(iteration):
            prior_now,_,_,_ = unpack(vector,nu,ps)
            return {"schema_version":1,"signature":sig,"chain_index":chain,
                    "schedule":"collapsed" if collapsed else "conditional","iteration":iteration,
                    "current_log_target":float(current),
                    "vector":vector.copy(),"nu":nu,"prior_state":copy_state(prior_now),
                    "C":None if C is None else C.copy(),"sigma2":sigma2,
                    "rng_state":copy.deepcopy(rng.bit_generator.state),"proposal":proposal.state(),
                    "initialization":copy.deepcopy(initialization),"initial_vector":initial_vector.copy(),
                    "retained_states":list(retained),"retained_log_density":list(densities),
                    "accepted_total":accepted_total,"accepted_sampling":accepted_sampling,
                    "proposals_sampling":proposals_sampling,"invalid_proposals":invalid}

        for iteration in range(start_iteration,total_iterations):
            if collapsed:
                proposed = proposal.propose(vector,rng)
                candidate,candidate_evaluation = collapsed_target(proposed,nu)
                accepted = bool(np.log(rng.uniform())<candidate-current)
                if accepted:
                    vector,current,evaluation = proposed,candidate,candidate_evaluation
                if not np.isfinite(candidate):
                    invalid += 1
                if covariance.infer_nu:
                    candidates = [collapsed_target(vector,n) for n in covariance.smoothness_choices]
                    logs = np.array([item[0] for item in candidates])
                    if not np.isfinite(logs).any():
                        raise FloatingPointError("Every smoothness candidate has zero posterior probability")
                    selected = int(rng.choice(len(logs),p=np.exp(logs-special.logsumexp(logs))))
                    nu = float(covariance.smoothness_choices[selected])
                    current,evaluation = candidates[selected]
                if prior.kind=="rhs" and config.rhs_tradeoff_scale:
                    old_prior,_,_,_ = unpack(vector,nu)
                    new_prior,tradeoff_accepted = runtime.rhs_tradeoff(old_prior,rng,config.rhs_tradeoff_scale)
                    if tradeoff_accepted:
                        # Every effective RHS variance is unchanged.  Reuse
                        # Gaussian factors and change only its hyperprior term.
                        current += runtime.log_prior(new_prior)-runtime.log_prior(old_prior)
                        vector[:npdim] = runtime.vector(new_prior)
            else:
                ps,cp,tp,rho = unpack(vector,nu,ps)
                R = covariance.covariance(coords,cp)
                precision = runtime.precision(ps,R)
                if separate:
                    sigma2 = float(np.exp(2*cp["log_sigma_e"]))
                backend = config.backend if config.backend in ("matrixfree","gaussian_matrixfree","dense") else "auto"
                C,solver_meta = engine.draw_conditional(R,precision,sigma2,rng,backend,
                    config.solver_rtol,config.solver_atol,config.solver_maxiter,rho)
                # Keep bounded solver summaries, not one dictionary per iteration.
                if solver_meta.get("finite_tolerance_approximation") or solver_meta.get("dense_fallback"):
                    solver_records.append({"chain":chain,"iteration":iteration,**solver_meta})
                if not separate:
                    sigma2 = engine.draw_sigma2_conditional(C,R,precision,rng,rho)
                ps = runtime.update_conditional(C,sigma2,ps,rng,R,rhs_tradeoff_scale=config.rhs_tradeoff_scale)
                residual = _ResidualStatistics(Y-Phi@C@Z.T,ar)
                def conditional_target(value,smoothness):
                    try:
                        with np.errstate(over="raise",invalid="raise",divide="raise"):
                            prior_now,cov_now,temp_now,rho_now = unpack(value,smoothness,ps)
                            hp = hyperprior(prior_now,cov_now,temp_now)
                            if not np.isfinite(hp):
                                return -np.inf
                            matrix = covariance.covariance(coords,cov_now)
                            L = linalg.cholesky(matrix,lower=True,check_finite=False)
                            EE,logdetT = residual.get(rho_now)
                            rss = float(np.trace(linalg.cho_solve((L,True),EE,check_finite=False)))
                            scale = float(np.exp(2*cov_now["log_sigma_e"])) if separate else sigma2
                            logdetR = 2*float(np.log(np.diag(L)).sum())
                            loglik = -.5*engine.N*np.log(2*np.pi*scale)-.5*engine.T*logdetR-.5*engine.n*logdetT-.5*rss/scale
                            result = loglik+runtime.coefficient_log_density(C,scale,prior_now,matrix)+hp
                            if not separate:
                                result += config.a0*np.log(config.b0)-special.gammaln(config.a0)-(config.a0+1)*np.log(scale)-config.b0/scale
                            return float(result) if np.isfinite(result) else -np.inf
                    except (FloatingPointError,ValueError,linalg.LinAlgError,OverflowError):
                        return -np.inf
                current = conditional_target(vector,nu)
                proposed = proposal.propose(vector,rng)
                candidate = conditional_target(proposed,nu)
                accepted = bool(np.log(rng.uniform())<candidate-current)
                if accepted:
                    vector,current = proposed,candidate
                if not np.isfinite(candidate):
                    invalid += 1
                if covariance.infer_nu:
                    logs = np.array([conditional_target(vector,n) for n in covariance.smoothness_choices])
                    if not np.isfinite(logs).any():
                        raise FloatingPointError("Every smoothness candidate has zero posterior probability")
                    selected = int(rng.choice(len(logs),p=np.exp(logs-special.logsumexp(logs))))
                    nu = float(covariance.smoothness_choices[selected])
                    current = float(logs[selected])
                if separate:
                    _,cov_now,_,_ = unpack(vector,nu,ps)
                    sigma2 = float(np.exp(2*cov_now["log_sigma_e"]))
            accepted_total += int(accepted)
            if iteration<config.warmup:
                proposal.adapt(vector,accepted)
            else:
                accepted_sampling += int(accepted)
                proposals_sampling += 1
                if (iteration-config.warmup+1)%config.thin==0:
                    prior_now,cov_now,temp_now,_ = unpack(vector,nu,ps)
                    state = {"prior_state":copy_state(prior_now),"covariance_parameters":copy.deepcopy(cov_now),
                             "temporal_parameters":copy.deepcopy(temp_now)}
                    if not collapsed:
                        state["sigma2"] = float(sigma2)
                    retained.append(state)
                    densities.append(float(current))
            if checkpoint_callback is not None and ((config.checkpoint_every and (iteration+1)%config.checkpoint_every==0)
                                                     or iteration+1==total_iterations):
                checkpoint_callback(chain,checkpoint(iteration+1))
        final_checkpoints.append({"chain":chain,"iteration":total_iterations,
                                  "adaptation_steps":proposal.count,"proposal_log_scale":proposal.log_scale})
        initial_states.append(initial_vector)
        all_states.append(retained)
        all_density.append(densities)
        all_acceptance.append(accepted_sampling/max(1,proposals_sampling))
        invalid_counts.append(invalid)
    if operator_route:
        backend = "vecchia_operator_spectral"
    elif config.backend=="dense":
        backend = "dense_gaussian"
    elif config.backend in ("matrixfree","gaussian_matrixfree") or not runtime.separable:
        backend = "gaussian_matrixfree"
    else:
        backend = "temporal_whitened_spectral" if ar else "spectral_gaussian"
    metadata = {"schedule":"collapsed" if collapsed else "conditional","backend":backend,
                "spatial_backend":"vecchia_operator" if operator_route else
                                  ("dense_exact" if covariance.kind=="matern" else "dense_reference"),
                "spatial_dense_materialization":not operator_route,
                "spatial_operator_scope":"iid Gaussian + ridge + collapsed Vecchia" if operator_route else None,
                "spatial_factorization":"sparse_lu" if operator_route else "dense_cholesky",
                "spatial_scaling_note":"Sparse LU fill depends on geometry; no guaranteed linear memory" if operator_route else
                                       "Spatial covariance is materialized within max_dense_sites",
                "spatial_geometry_cached":bool(operator_route),
                "scale_integration":"coefficients_only" if separate else ("coefficients_and_sigma2" if collapsed else "none"),
                "stored_coefficients":False,"joint_recovery":"conditional_on_retained_sigma2" if not collapsed else "sigma2_then_coefficients",
                "warmup":config.warmup,"thin":config.thin,"adaptation_frozen_after_warmup":True,
                "initialization":initialization,"initial_states":np.array(initial_states),
                "rhs_tradeoff_scale":config.rhs_tradeoff_scale if prior.kind=="rhs" else 0.,
                "acceptance_definition":"primary_joint_or_covariance_random_walk",
                "invalid_proposals":invalid_counts,"solver_records":solver_records,
                "final_chain_summaries":final_checkpoints,"checkpoint_schema_version":1,"data_signature":sig,
                "likelihood_normalized":True,"hyperpriors_evaluated_in":"declared_log_or_logit_coordinates",
                "covariance_approximation":covariance.kind if covariance.kind!="matern" else None,
                "reference_allocation_energy":getattr(runtime,"reference_energy",None)}
    return RegressionPosterior(Y,Phi,Z,coords,groups,prior,covariance,config,all_states,
                               np.array(all_density),np.array(all_acceptance),metadata,reference_Z)
