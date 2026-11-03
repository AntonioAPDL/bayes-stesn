"""Separable Matérn space-time GP with exact temporal state filtering.

For temporal nu=1/2 or 3/2, the continuous covariance has an exact small
state representation. Independent homoskedastic noise permits eigenspatial
decoupling. Conditional new-site residuals are temporal processes, not
independently redrawn kriging errors at every forecast frame.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from .optimization import bounded_lbfgsb
from .classical import ClassicalFit, _positive, _horizon, _paths, matern_correlation, psd_root
from .contracts import Forecast, validate_history, forecast_targets
from ..io import seed_for


# Response-dependent center/signal/noise defaults are resolved only at fitting.
DEFAULT_CONFIG = {"max_fit_sites":512,"spatial_nu":1.5,"temporal_nu":.5,
    "spatial_length":.25,"fit_hyperparameters":True,
    "max_iter":25,"continuation_max_iter":0,"ftol":1e-8,"gtol":1e-5,
    "max_joint_dimension":2000,"max_forecast_bytes":1_073_741_824}


def temporal_system(length, nu, delta=1.):
    """Return F,P_infinity,Q for a unit-marginal-variance Matérn process."""
    ell = _positive(length, "temporal_length"); dt = _positive(delta, "time spacing")
    if nu == .5:
        f = np.array([[np.exp(-dt/ell)]])
        p = np.ones((1,1))
    elif nu == 1.5:
        rate = np.sqrt(3)/ell
        f = np.exp(-rate*dt)*np.array([[1+rate*dt,dt],[-rate*rate*dt,1-rate*dt]])
        p = np.diag([1.,rate*rate])
    else:
        raise ValueError("Exact temporal GP supports nu=0.5 or 1.5")
    q = p-f @ p @ f.T
    return f,p,(q+q.T)/2


def filter_modes(rotated_y, spatial_variances, noise_variance, f, p_inf, q,
                 *, initial_mean=None, initial_cov=None, predict_initial=False):
    """Vectorized independent spatial-mode Kalman filters."""
    z = np.asarray(rotated_y, float); variance = np.asarray(spatial_variances, float)
    n = len(variance); d = len(f)
    m = np.zeros((n,d)) if initial_mean is None else np.array(initial_mean,copy=True)
    p = variance[:,None,None]*p_inf if initial_cov is None else np.array(initial_cov,copy=True)
    ll=0.
    if d == 1:
        mean=m[:,0].copy(); cov=p[:,0,0].copy(); decay=float(f[0,0])
        innovation_variance=variance*float(q[0,0])
        for t,y in enumerate(z):
            if t or predict_initial:
                mean=decay*mean;cov=decay*decay*cov+innovation_variance
            s=cov+noise_variance; error=y-mean
            if np.any(s<=0):
                raise np.linalg.LinAlgError("Nonpositive GP innovation variance")
            gain=cov/s
            mean+=gain*error;cov*=1-gain
            ll-=.5*np.sum(np.log(2*np.pi*s)+error*error/s)
        return mean[:,None],cov[:,None,None],float(ll)
    for t,y in enumerate(z):
        if t or predict_initial:
            m=m @ f.T
            p=f @ p @ f.T+variance[:,None,None]*q
        s=p[:,0,0]+noise_variance;error=y-m[:,0]
        if np.any(s<=0):
            raise np.linalg.LinAlgError("Nonpositive GP innovation variance")
        gain=p[:,:,0]/s[:,None]
        m+=gain*error[:,None]
        p-=gain[:,:,None]*p[:,None,0,:]
        p=(p+p.transpose(0,2,1))/2
        ll-=.5*np.sum(np.log(2*np.pi*s)+error*error/s)
    return m,p,float(ll)


class SeparableGPFit(ClassicalFit):
    model_id="separable_stgp"
    _state_fields=("center","offset","coordinate_scale","spatial_length","temporal_length",
                   "signal_sd","noise_sd","spatial_nu","temporal_nu","delta",
                   "terminal_mean","terminal_covariance")

    def __init__(self,data,config=None,seed=0):
        super().__init__(data,config,seed)
        if len(data.coordinates)>int(self.config.get("max_fit_sites",512)):
            raise MemoryError("Dense spatial GP exceeds max_fit_sites; use a declared approximation")
        self.center=float(self.config.get("center",data.y.mean()))
        if not np.isfinite(self.center):
            raise ValueError("GP center must be finite")
        scale=max(float(data.y.std()),1e-6)
        self.offset=data.coordinates.min(axis=0)
        self.coordinate_scale=max(float(np.max(np.ptp(data.coordinates,axis=0))),1e-12)
        self.delta=float(data.times[1]-data.times[0])
        self.spatial_nu=float(self.config.get("spatial_nu",1.5))
        self.temporal_nu=float(self.config.get("temporal_nu",.5))
        spatial=_positive(self.config.get("spatial_length",.25),"spatial_length")
        temporal=_positive(self.config.get("temporal_length",10*self.delta),"temporal_length")
        signal=_positive(self.config.get("signal_sd",scale),"signal_sd")
        noise=_positive(self.config.get("noise_sd",.2*scale),"noise_sd")
        temporal_system(temporal,self.temporal_nu,self.delta)
        x=(data.coordinates-self.offset)/self.coordinate_scale
        centered=data.y-self.center

        @lru_cache(maxsize=4)
        def spatial_decomposition(length):
            covariance=matern_correlation(x,x,length,self.spatial_nu)
            values,vectors=eigh(covariance,check_finite=False)
            if values[0]<-1e-9*max(values[-1],1):
                raise np.linalg.LinAlgError("Spatial GP covariance is not PSD")
            return np.maximum(values,0.),vectors,centered @ vectors

        def objective(log_parameters):
            ell_s,ell_t,sd_f,sd_e=np.exp(log_parameters)
            try:
                values,_,z=spatial_decomposition(float(ell_s))
                f,p,q=temporal_system(ell_t,self.temporal_nu,self.delta)
                _,_,ll=filter_modes(z,sd_f*sd_f*values,sd_e*sd_e,f,p,q)
            except (np.linalg.LinAlgError,FloatingPointError):
                return 1e100
            return -ll if np.isfinite(ll) else 1e100

        initial=np.log([spatial,temporal,signal,noise])
        optimize=bool(self.config.get("fit_hyperparameters",True))
        bounds=np.log(np.array([[.005,3.],[.1*self.delta,max(4.,len(data.y))*self.delta],
                                 [1e-3*scale,5*scale],[1e-4*scale,3*scale]]))
        if optimize:
            initial=np.clip(initial,bounds[:,0],bounds[:,1])
            result,optimization=bounded_lbfgsb(objective,initial,bounds,
                            max_iter=int(self.config.get("max_iter",25)),
                            continuation_max_iter=int(self.config.get("continuation_max_iter",0)),
                            ftol=float(self.config.get("ftol",1e-8)),
                            gtol=float(self.config.get("gtol",1e-5)))
            parameters=result.x
            converged=optimization["qualified"]
            self.diagnostics.update(optimizer="L-BFGS-B_marginal_likelihood",optimizer_message=str(result.message),
                                    iterations=optimization["iterations"],function_evaluations=optimization["function_evaluations"],
                                    optimizer_details=optimization,
                                    optimizer_projected_gradient_inf=optimization["projected_gradient_inf"])
        else:
            parameters=initial;converged=True
            self.diagnostics.update(optimizer="fixed_declared_hyperparameters",iterations=0,function_evaluations=1)
        self.spatial_length,self.temporal_length,self.signal_sd,self.noise_sd=map(float,np.exp(parameters))
        self._after_load()
        self.terminal_mean,self.terminal_covariance,ll=filter_modes(
            centered @ self._vectors,self._values*self.signal_sd**2,self.noise_sd**2,
            self._f,self._p_inf,self._q)
        if not np.isfinite(ll):
            raise FloatingPointError("GP fit has nonfinite likelihood")
        boundary_hits=[]
        if optimize:
            for j,name in enumerate(("spatial_length","temporal_length","signal_sd","noise_sd")):
                if min(abs(parameters[j]-bounds[j,0]),abs(parameters[j]-bounds[j,1]))<1e-4:
                    boundary_hits.append(name)
        self.diagnostics.update(converged=converged,qualified=converged,finite_fit=True,
            log_likelihood=float(ll),boundary_hits=boundary_hits,temporal_state_dimension=len(self._f),
            uncertainty="Gaussian_joint_conditional_on_fitted_hyperparameters",
            parameter_uncertainty=False,spatial_length_units="maximum_observed_coordinate_span",
            temporal_length_units="same_as_times",spatial_decomposition_cache_entries=spatial_decomposition.cache_info().currsize)

    def _after_load(self):
        self._x=(self.data.coordinates-self.offset)/self.coordinate_scale
        self._space=matern_correlation(self._x,self._x,self.spatial_length,self.spatial_nu)
        self._values,self._vectors=eigh(self._space,check_finite=False)
        self._values=np.maximum(self._values,0.)
        self._space_factor=cho_factor(self._space,lower=True,check_finite=False)
        self._f,self._p_inf,self._q=temporal_system(self.temporal_length,self.temporal_nu,self.delta)

    def _terminal(self,observed):
        if len(observed.y)==len(self.data.y):
            return self.terminal_mean,self.terminal_covariance
        z=(observed.y[len(self.data.y):]-self.center) @ self._vectors
        m,p,_=filter_modes(z,self._values*self.signal_sd**2,self.noise_sd**2,self._f,self._p_inf,self._q,
                           initial_mean=self.terminal_mean,initial_cov=self.terminal_covariance,
                           predict_initial=True)
        return m,p

    def _target_setup(self,coords,*,residual_covariance=True):
        unique,mapping=np.unique(coords,axis=0,return_inverse=True)
        u=(unique-self.offset)/self.coordinate_scale
        cross=matern_correlation(u,self._x,self.spatial_length,self.spatial_nu)
        weights=cho_solve(self._space_factor,cross.T,check_finite=False).T
        residual=None
        if residual_covariance:
            residual=matern_correlation(u,u,self.spatial_length,self.spatial_nu)-weights @ cross.T
            residual=(residual+residual.T)/2
        return unique,mapping,weights @ self._vectors,residual

    def joint_moments(self,horizon,target_coordinates=None,target_site_ids=None,history=None):
        """Exact conditional mean/covariance for small reference cases only.

        Covariance axes flatten [lead,site] in C order. This deliberately
        bounded dense reference is never used by the ordinary sampler.
        """
        h=_horizon(horizon);observed=validate_history(self.data,history)
        coords,_=forecast_targets(self.data,target_coordinates,target_site_ids)
        if h*len(coords)>int(self.config.get("max_joint_dimension",2000)):
            raise MemoryError("Dense joint-moment reference exceeds max_joint_dimension")
        unique,mapping,loadings,residual=self._target_setup(coords)
        m,p=self._terminal(observed);n=len(self._values);d=len(self._f);s=len(unique)
        means=np.empty((h,s));vstates=[];current=m.copy();v=p.copy()
        for lead in range(h):
            current=current @ self._f.T
            v=self._f @ v @ self._f.T+self._values[:,None,None]*self.signal_sd**2*self._q
            vstates.append(v.copy());means[lead]=self.center+current[:,0] @ loadings.T
        cov=np.zeros((h*s,h*s))
        for k in range(h):
            cross_state=vstates[k]
            for j in range(k,h):
                if j>k:
                    cross_state=self._f @ cross_state
                temporal=matern_correlation(np.array([[j*self.delta]]),np.array([[k*self.delta]]),self.temporal_length,self.temporal_nu)[0,0]
                block=(loadings*cross_state[:,0,0]) @ loadings.T+self.signal_sd**2*temporal*residual
                if j==k:
                    block+=self.noise_sd**2*np.eye(s)
                cov[j*s:(j+1)*s,k*s:(k+1)*s]=block
                cov[k*s:(k+1)*s,j*s:(j+1)*s]=block.T
        indices=np.concatenate([j*s+mapping for j in range(h)])
        return means[:,mapping],cov[np.ix_(indices,indices)]

    def forecast(self,horizon,target_coordinates=None,target_site_ids=None,n_paths=256,seed=0,history=None):
        h=_horizon(horizon);paths=_paths(n_paths);observed=validate_history(self.data,history)
        coords,ids=forecast_targets(self.data,target_coordinates,target_site_ids)
        required=8*(2*paths*h*len(coords)+4*h*len(coords)
                    +(3*len(coords)**2+4*paths*(len(self._values)+len(coords))*len(self._f) if paths else 0))
        if required>int(self.config.get("max_forecast_bytes",1_073_741_824)):
            raise MemoryError("GP forecast allocation exceeds max_forecast_bytes")
        unique,mapping,loadings,residual=self._target_setup(coords,residual_covariance=bool(paths))
        m,p=self._terminal(observed)
        current=m.copy();mean=np.empty((h,len(unique)))
        for lead in range(h):
            current=current @ self._f.T
            mean[lead]=self.center+current[:,0] @ loadings.T
        samples=None
        if paths:
            values,vectors=eigh(residual,check_finite=False)
            if values[0]<-1e-8*max(float(values[-1]),1.):
                raise np.linalg.LinAlgError("New-site conditional GP covariance is not PSD")
            keep=values>1e-12*max(float(values[-1]),1.)
            values=np.maximum(values[keep],0.);vectors=vectors[:,keep]
            n=len(self._values);r=len(values);d=len(self._f);s=len(unique)
            streams=[np.random.default_rng(seed_for(seed,self.model_id,"path",i)) for i in range(paths)]
            initial=np.array([rng.normal(size=(n+r)*d) for rng in streams])
            roots=np.array([psd_root(one) for one in p])
            observed_states=m+np.einsum("bnq,nrq->bnr",initial[:,:n*d].reshape(paths,n,d),roots)
            residual_states=(initial[:,n*d:].reshape(paths,r,d) @ psd_root(self._p_inf).T)*self.signal_sd*np.sqrt(values)[None,:,None]
            process_root=psd_root(self._q)
            observed_scale=self.signal_sd*np.sqrt(self._values)[None,:,None]
            residual_scale=self.signal_sd*np.sqrt(values)[None,:,None]
            samples=np.empty((paths,h,s))
            for lead in range(h):
                innovations=np.array([rng.normal(size=(n+r)*d+s) for rng in streams])
                observed_states=observed_states @ self._f.T+(innovations[:,:n*d].reshape(paths,n,d) @ process_root.T)*observed_scale
                residual_states=residual_states @ self._f.T+(innovations[:,n*d:(n+r)*d].reshape(paths,r,d) @ process_root.T)*residual_scale
                samples[:,lead]=self.center+observed_states[:,:,0] @ loadings.T+residual_states[:,:,0] @ vectors.T+self.noise_sd*innovations[:,-s:]
            samples=samples[:,:,mapping]
        return Forecast(mean=mean[:,mapping],samples=samples,coordinates=coords,site_ids=ids,
                        metadata=self._metadata(observed,uncertainty="joint_plugin_gaussian",parameter_uncertainty=False,
                            temporal_nu=self.temporal_nu,newsite_residual="temporally_correlated_independent_conditional_GP",
                            path_seed_policy="global_path_index"))
