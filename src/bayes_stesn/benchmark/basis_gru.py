"""Optional probabilistic coordinate-basis GRU control.

This is a custom matched control, not a reproduction of a published model.
Let Phi be the geometry-only basis with Phi'Phi/n=I, and a_t=Phi' y_t/n
on the standardized response scale. The conditional model is

    y_t(s) = phi(s)' g_theta(a_{t-L:t-1}) + e_t(s),
    Cov(e_t(s),e_t(s')) = sigma_f^2 Matern(s,s') + sigma_e^2 1[s=s'].

The mean network is fitted by projected one-step MSE. A chronological
training tail selects the epoch; the selected epoch count is then refitted
on all supplied training data by default. Residual covariance is fitted
afterwards by Gaussian likelihood on that optimization data. This staged
fit is explicitly not joint maximum likelihood or Bayesian inference.

Forecasts are ancestral joint observation paths. Only generated observations
at the fitted sites are projected back into the GRU. New sites are decoded
from the same state and their innovations are drawn conditionally on the
observed-site innovations. A fixed number of ancestral paths estimates the
predictive mean by averaging conditional means (Rao--Blackwellization).
PyTorch is imported lazily, so the core package has no neural dependency.
"""
from __future__ import annotations

import importlib
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from .classical import (ClassicalFit, CoordinateBasis, _positive, _horizon,
                        _paths, _panel_state, matern_correlation, psd_root)
from .contracts import Panel, Forecast, validate_history, forecast_targets
from ..io import seed_for, save_state, load_state


DEFAULTS = {
    "rank":8, "basis_width":.25, "lags":8, "hidden_size":32,
    "epochs":100, "patience":12, "learning_rate":.001, "batch_size":64,
    "validation_fraction":.2, "validation_horizon":16, "weight_decay":1e-4,
    "gradient_clip":1., "min_delta":1e-6, "refit":True,
    "covariance_spatial_nu":1.5, "covariance_spatial_length":.25,
    "covariance_max_iter":40, "covariance_ftol":1e-9,
    "mean_paths":128, "prediction_batch_size":64, "max_fit_sites":512,
    "max_domain_sites":1024, "max_forecast_bytes":1_073_741_824,
}


def _torch():
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise ImportError("The optional basis_gru comparator requires PyTorch") from exc


def _configuration(config):
    result={**DEFAULTS, **dict(config or {})}
    unknown=set(result)-set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown basis_gru configuration: {sorted(unknown)}")
    for key in ("rank","lags","hidden_size","epochs","patience","batch_size",
                "validation_horizon","covariance_max_iter","mean_paths",
                "prediction_batch_size","max_fit_sites","max_domain_sites",
                "max_forecast_bytes"):
        value=result[key]
        if isinstance(value,bool) or int(value)!=value or value<1:
            raise ValueError(f"{key} must be a positive integer")
        result[key]=int(value)
    for key in ("basis_width","learning_rate","gradient_clip",
                "covariance_spatial_length","covariance_ftol"):
        result[key]=_positive(result[key],key)
    for key in ("weight_decay","min_delta"):
        result[key]=float(result[key])
        if not np.isfinite(result[key]) or result[key]<0:
            raise ValueError(f"{key} must be nonnegative and finite")
    if not 0<float(result["validation_fraction"])<.5:
        raise ValueError("validation_fraction must lie in (0,0.5)")
    result["validation_fraction"]=float(result["validation_fraction"])
    if result["covariance_spatial_nu"] not in (.5,1.5,2.5):
        raise ValueError("covariance_spatial_nu must be 0.5, 1.5 or 2.5")
    if not isinstance(result["refit"],bool):
        raise ValueError("refit must be boolean")
    return result


def _network(torch,rank,hidden_size):
    class CoordinateGRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru=torch.nn.GRU(rank,hidden_size,batch_first=True)
            self.readout=torch.nn.Linear(hidden_size,rank)

        def forward(self,x):
            _,hidden=self.gru(x)
            return self.readout(hidden[-1])
    return CoordinateGRU()


def _fit_network(data,phi,config,seed,*,cutoff,epochs,select_epoch):
    """Prefix transforms for stopping; full-data transforms for final refit."""
    torch=_torch(); n=len(phi); lags=config["lags"]
    center=float(data.y[:cutoff].mean());scale=max(float(data.y[:cutoff].std()),1e-8)
    coefficients=((data.y-center)/scale) @ phi/n
    tensor=torch.tensor(coefficients,dtype=torch.float32)
    origins=np.arange(lags,cutoff)
    indices=origins[:,None]+np.arange(-lags,0)
    windows=tensor[indices];targets=tensor[origins]
    init_seed=seed_for(seed,"basis_gru","initialization")%(2**63-1)
    order_seed=seed_for(seed,"basis_gru","optimization")%(2**63-1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        network=_network(torch,phi.shape[1],config["hidden_size"])
        optimizer=torch.optim.AdamW(network.parameters(),lr=config["learning_rate"],
                                    weight_decay=config["weight_decay"])
        generator=torch.Generator(device="cpu").manual_seed(order_seed)
        records=[];best_loss=np.inf;best_epoch=0;best_state=None;stale=0;stopped=False
        for epoch in range(1,epochs+1):
            network.train(); order=torch.randperm(len(origins),generator=generator)
            total=0.
            for start in range(0,len(order),config["batch_size"]):
                batch=order[start:start+config["batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                prediction=network(windows[batch])
                loss=torch.mean((prediction-targets[batch])**2)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite basis-GRU training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(),config["gradient_clip"],
                                               error_if_nonfinite=True)
                optimizer.step(); total+=float(loss.detach())*len(batch)
            record={"epoch":epoch,"training_mse":total/len(origins)}
            if select_epoch:
                network.eval()
                steps=min(config["validation_horizon"],len(data.y)-cutoff)
                context=tensor[cutoff-lags:cutoff][None].clone()
                predictions=[]
                with torch.no_grad():
                    for _ in range(steps):
                        nxt=network(context);predictions.append(nxt[0])
                        context=torch.cat((context[:,1:],nxt[:,None]),dim=1)
                    validation=float(torch.mean((torch.stack(predictions)-tensor[cutoff:cutoff+steps])**2))
                if not np.isfinite(validation):
                    raise FloatingPointError("Nonfinite basis-GRU stopping loss")
                record["validation_mse"]=validation
                if validation<best_loss-config["min_delta"]:
                    best_loss=validation;best_epoch=epoch;stale=0
                    best_state={key:value.detach().clone() for key,value in network.state_dict().items()}
                else:
                    stale+=1
                records.append(record)
                if stale>=config["patience"]:
                    stopped=True;break
            else:
                records.append(record);best_epoch=epoch
        if select_epoch:
            network.load_state_dict(best_state)
        network.eval()
    return network,center,scale,best_epoch,records,stopped


class BasisGRUFit(ClassicalFit):
    model_id="basis_gru"

    def __init__(self,data,config=None,seed=0):
        config=_configuration(config)
        super().__init__(data,config,seed)
        self.training=data
        if len(data.coordinates)>config["max_fit_sites"]:
            raise MemoryError("basis_gru residual covariance exceeds max_fit_sites")
        cutoff=int(np.floor((1-config["validation_fraction"])*len(data.y)))
        if cutoff<config["lags"]+3 or cutoff>=len(data.y):
            raise ValueError("Insufficient rows for lag context and chronological stopping tail")
        self.basis=CoordinateBasis.fit(data.coordinates,config["rank"],config["basis_width"])
        self._phi=self.basis.evaluate(data.coordinates)
        network,center,scale,selected,selection,stopped=_fit_network(
            data,self._phi,config,seed,cutoff=cutoff,epochs=config["epochs"],select_epoch=True)
        if config["refit"]:
            network,center,scale,_,refitting,_=_fit_network(
                data,self._phi,config,seed,cutoff=len(data.y),epochs=selected,select_epoch=False)
            covariance_rows=len(data.y)
        else:
            refitting=[];covariance_rows=cutoff
        self.network,self.center,self.scale=network,center,scale
        self.offset=data.coordinates.min(axis=0)
        self.coordinate_scale=max(float(np.max(np.ptp(data.coordinates,axis=0))),1e-12)
        self._normalized_coordinates=(data.coordinates-self.offset)/self.coordinate_scale
        self.lags=config["lags"]
        coefficients=((data.y[:covariance_rows]-center)/scale) @ self._phi/len(self._phi)
        origins=np.arange(self.lags,covariance_rows)
        context=coefficients[origins[:,None]+np.arange(-self.lags,0)]
        predicted=self._predict(context)
        residual=(data.y[self.lags:covariance_rows]-center)/scale-predicted @ self._phi.T
        self._fit_covariance(residual)
        self._cache_covariance()
        torch=_torch()
        self.diagnostics.update(
            qualified=bool(self.diagnostics["covariance_converged"]),finite_fit=True,
            neural_stop_reason="patience" if stopped else "epoch_budget",
            neural_optimizer_convergence="not_claimed",selected_epoch=selected,
            selection_training_rows=cutoff,selection_validation_rows=min(config["validation_horizon"],len(data.y)-cutoff),
            selection_history=selection,refit_history=refitting,
            covariance_fit_rows=covariance_rows-self.lags,
            covariance_fit_policy="one_step_residuals_on_optimization_rows_only",
            inference="staged_projected_MSE_then_Gaussian_residual_likelihood",
            mean_fit="chronological_free_running_stopping_then_selected_epoch_refit" if config["refit"] else "chronological_free_running_stopping",
            covariance="spatial_Matern_plus_independent_site_nugget; independent_innovation_fields_over_time",
            parameter_uncertainty=False,uncertainty="ancestral_joint_plugin_observation_paths",
            spatial_length_units="maximum_observed_coordinate_span",
            torch_version=str(torch.__version__),device="cpu",dtype="float32_network_float64_covariance",
            torch_threads=int(torch.get_num_threads()),rank=self._phi.shape[1],
            research_status="custom_matched_control_not_paper_reproduction")

    def _predict(self,contexts):
        """Fixed padded batches preserve path prefixes when B changes."""
        torch=_torch();batch=self.config["prediction_batch_size"]
        contexts=np.asarray(contexts,dtype=np.float32)
        result=np.empty((len(contexts),self._phi.shape[1]))
        with torch.no_grad():
            for start in range(0,len(contexts),batch):
                count=min(batch,len(contexts)-start)
                x=np.zeros((batch,self.config["lags"],self._phi.shape[1]),dtype=np.float32)
                x[:count]=contexts[start:start+count]
                result[start:start+count]=self.network(torch.from_numpy(x)).numpy()[:count]
        return result

    def _fit_covariance(self,residual):
        count,n=residual.shape;scatter=residual.T @ residual
        spread=max(float(np.sqrt(np.mean(residual**2))),1e-6)
        self.spatial_nu=float(self.config["covariance_spatial_nu"])
        bounds=np.log(np.array([[.005,3.],[1e-4*spread,5*spread],[1e-4*spread,5*spread]]))
        initial=np.log([self.config["covariance_spatial_length"],spread/np.sqrt(2),spread/np.sqrt(2)])

        def objective(parameters):
            length,signal,noise=np.exp(parameters)
            covariance=signal**2*matern_correlation(self._normalized_coordinates,
                self._normalized_coordinates,length,self.spatial_nu)+noise**2*np.eye(n)
            try:
                factor=cho_factor(covariance,lower=True,check_finite=False)
                value=.5*(count*(n*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum())
                           +np.trace(cho_solve(factor,scatter,check_finite=False)))
            except np.linalg.LinAlgError:
                return 1e100
            return float(value) if np.isfinite(value) else 1e100

        result=minimize(objective,np.clip(initial,bounds[:,0],bounds[:,1]),method="L-BFGS-B",bounds=bounds,
            options={"maxiter":self.config["covariance_max_iter"],"ftol":self.config["covariance_ftol"],"maxls":30})
        self.spatial_length,self.signal_sd,self.noise_sd=map(float,np.exp(result.x))
        if not np.isfinite(result.fun) or result.fun>=1e99:
            raise FloatingPointError("Residual covariance likelihood is nonfinite")
        boundary=[name for j,name in enumerate(("spatial_length","signal_sd","noise_sd"))
                  if min(abs(result.x[j]-bounds[j,0]),abs(result.x[j]-bounds[j,1]))<1e-4]
        self.diagnostics.update(covariance_converged=bool(result.success),
            covariance_optimizer="L-BFGS-B_Gaussian_residual_likelihood",
            covariance_optimizer_message=str(result.message),covariance_iterations=int(result.nit),
            covariance_function_evaluations=int(result.nfev),covariance_boundary_hits=boundary,
            covariance_log_likelihood_original=float(-result.fun-count*n*np.log(self.scale)))

    def _cache_covariance(self):
        self._phi=self.basis.evaluate(self.data.coordinates)
        self._normalized_coordinates=(self.data.coordinates-self.offset)/self.coordinate_scale
        covariance=self.signal_sd**2*matern_correlation(self._normalized_coordinates,
            self._normalized_coordinates,self.spatial_length,self.spatial_nu)+self.noise_sd**2*np.eye(len(self._phi))
        self._observed_root=np.linalg.cholesky(covariance)
        self._observed_factor=(self._observed_root,True)

    def innovation_covariance(self,coordinates,*,standardized=False):
        """One-step observation innovation covariance, with shared duplicates."""
        coords=np.asarray(coordinates,float)
        normalized=(coords-self.offset)/self.coordinate_scale
        same=np.all(coords[:,None,:]==coords[None,:,:],axis=2)
        covariance=self.signal_sd**2*matern_correlation(normalized,normalized,self.spatial_length,self.spatial_nu)+self.noise_sd**2*same
        return covariance if standardized else self.scale**2*covariance

    def forecast(self,horizon,target_coordinates=None,target_site_ids=None,n_paths=256,seed=0,history=None):
        h=_horizon(horizon);paths=_paths(n_paths);observed=validate_history(self.data,history)
        coords,ids=forecast_targets(self.data,target_coordinates,target_site_ids)
        unique,mapping=np.unique(coords,axis=0,return_inverse=True)
        lookup={tuple(c):i for i,c in enumerate(self.data.coordinates)}
        observed_indices=np.array([lookup.get(tuple(c),-1) for c in unique])
        new_indices=np.flatnonzero(observed_indices<0)
        domain=len(self.data.coordinates)+len(new_indices)
        if domain>self.config["max_domain_sites"]:
            raise MemoryError("Joint basis_gru forecast exceeds max_domain_sites")
        mean_paths=self.config["mean_paths"];total=max(paths,mean_paths)
        required=8*(2*paths*h*len(coords)+4*h*len(coords)+4*domain*domain
                    +3*total*self.lags*self._phi.shape[1]+4*total*len(self._phi))
        if required>self.config["max_forecast_bytes"]:
            raise MemoryError("Joint basis_gru forecast exceeds max_forecast_bytes")
        phi=self.basis.evaluate(unique)
        new_weights=new_root=None
        if paths and len(new_indices):
            new=unique[new_indices]
            cross=self.signal_sd**2*matern_correlation((new-self.offset)/self.coordinate_scale,
                self._normalized_coordinates,self.spatial_length,self.spatial_nu)
            new_weights=cho_solve(self._observed_factor,cross.T,check_finite=False).T
            residual=self.innovation_covariance(new,standardized=True)-new_weights @ cross.T
            new_root=psd_root(residual)
        coefficient_history=((observed.y[-self.lags:]-self.center)/self.scale) @ self._phi/len(self._phi)
        contexts=np.repeat(coefficient_history[None],total,axis=0)
        streams=[np.random.default_rng(seed_for(seed,self.model_id,"observed_path",i)) for i in range(total)]
        new_streams=[np.random.default_rng(seed_for(seed,self.model_id,"newsite_path",i)) for i in range(paths)]
        mean=np.empty((h,len(unique)))
        samples=np.empty((paths,h,len(unique))) if paths else None
        at_observed=np.flatnonzero(observed_indices>=0)
        for lead in range(h):
            coefficients=self._predict(contexts)
            mean[lead]=self.center+self.scale*(phi @ coefficients[:mean_paths].mean(axis=0))
            innovations=np.array([rng.normal(size=len(self._phi)) for rng in streams]) @ self._observed_root.T
            generated=coefficients @ self._phi.T+innovations
            if paths:
                target_noise=np.empty((paths,len(unique)))
                target_noise[:,at_observed]=innovations[:paths,observed_indices[at_observed]]
                if len(new_indices):
                    normal=np.array([rng.normal(size=len(new_indices)) for rng in new_streams])
                    target_noise[:,new_indices]=innovations[:paths] @ new_weights.T+normal @ new_root.T
                samples[:,lead]=self.center+self.scale*(coefficients[:paths] @ phi.T+target_noise)
            next_coefficients=generated @ self._phi/len(self._phi)
            contexts=np.concatenate((contexts[:,1:],next_coefficients[:,None]),axis=1)
        return Forecast(mean=mean[:,mapping],samples=None if samples is None else samples[:,:,mapping],
            coordinates=coords,site_ids=ids,metadata=self._metadata(observed,
            uncertainty="ancestral_joint_plugin_gaussian_innovations",parameter_uncertainty=False,
            mean_estimator="Rao_Blackwell_mean_over_ancestral_observed_histories",mean_paths=mean_paths,
            newsite_history="never_used",path_seed_policy="global_path_index_separate_observed_and_newsite_streams"))

    def to_state(self):
        parameters={key:getattr(self,key) for key in ("center","scale","offset","coordinate_scale",
            "lags","spatial_length","spatial_nu","signal_sd","noise_sd")}
        return {"version":1,"model_id":self.model_id,"data":_panel_state(self.data),
                "config":self.config,"seed":self.seed,"parameters":parameters,
                "basis":self.basis.to_state(),"diagnostics":self.diagnostics,
                "weights":{key:value.detach().cpu().numpy().copy() for key,value in self.network.state_dict().items()}}

    def save(self,path):
        return save_state(path,self.to_state(),kind="benchmark_basis_gru")


def fit_basis_gru(data,config=None,seed=0):
    return BasisGRUFit(data,config,seed)


def basis_gru_from_state(state):
    if state.get("version")!=1 or state.get("model_id")!="basis_gru":
        raise ValueError("Unsupported basis_gru state")
    parameters={"center","scale","offset","coordinate_scale","lags","spatial_length",
                "spatial_nu","signal_sd","noise_sd"}
    if set(state["parameters"])!=parameters:
        raise ValueError("Stored basis_gru parameter fields do not match its schema")
    obj=BasisGRUFit.__new__(BasisGRUFit)
    ClassicalFit.__init__(obj,Panel(**state["data"]),_configuration(state["config"]),state["seed"])
    obj.training=obj.data;obj.basis=CoordinateBasis.from_state(state["basis"])
    for key,value in state["parameters"].items():
        setattr(obj,key,value)
    obj.diagnostics=dict(state["diagnostics"]);obj._cache_covariance()
    torch=_torch()
    with torch.random.fork_rng(devices=[]):
        obj.network=_network(torch,obj._phi.shape[1],obj.config["hidden_size"])
    obj.network.load_state_dict({key:torch.tensor(value) for key,value in state["weights"].items()})
    obj.network.eval()
    return obj


def load_basis_gru(path):
    return basis_gru_from_state(load_state(path,kind="benchmark_basis_gru"))
