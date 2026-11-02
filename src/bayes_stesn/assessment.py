"""MCMC checks for both hyperparameters and prespecified readout functionals."""
from __future__ import annotations
import numpy as np
from .diagnostics import summarize_chains
from .io import seed_for


def diagnose(fitted, *, functional_draws=256, seed=90731,
             rhat_limit=1.01, min_ess=400, min_chains=4):
    """Joint scientific screening gate, not a proof of convergence.

    Recover one C,sigma2 draw for each of an evenly spaced set of retained
    hyperparameter states per chain. Inspect log observation-reference scale,
    three fixed-site readout means, and the spatially averaged readout at the
    final fitted feature vector. Functional draws remain in chain order.
    This consumes bounded memory and does not run any future forecasts.
    """
    if functional_draws < 8:
        raise ValueError("At least eight functional draws per chain are required")
    posterior=fitted.posterior
    count=min(int(functional_draws),posterior.n_draws)
    indices=np.linspace(0,posterior.n_draws-1,count,dtype=int)
    p=fitted.prepared
    locations=np.array([0,len(p.Phi)//2,len(p.Phi)-1])
    z=p.fit_Z[-1]
    names=('log_sigma2_original_units','readout_first_site','readout_middle_site',
           'readout_last_site','readout_spatial_average')
    values=np.empty((posterior.n_chains,count,len(names)))
    for chain in range(posterior.n_chains):
        for j,draw in enumerate(indices):
            parameter=posterior.draw_parameters(chain,int(draw),
                np.random.default_rng(seed_for(seed,'diagnostic',chain,int(draw))))
            mean=p.Phi@(parameter.C@z)*p.response_scale+p.response_center
            values[chain,j]=np.r_[np.log(parameter.sigma2)+2*np.log(p.response_scale),
                                  mean[locations],float(mean.mean())]
    kwargs=dict(rhat_limit=rhat_limit,min_ess=min_ess,min_chains=min_chains)
    hyper=summarize_chains(posterior.chains,posterior.parameter_names,**kwargs)
    functional=summarize_chains(values,names,**kwargs)
    return {'passed':bool(hyper.passed and functional.passed),
            'hyperparameters':hyper.to_dict(),'functionals':functional.to_dict(),
            'functional_policy':{'draws_per_chain':count,'retained_indices':indices.tolist(),
               'site_indices':locations.tolist(),'feature_time_index':fitted.training_end,
               'seed':int(seed),'meaning':'Joint conditional coefficient/scale recovery; no future observations',
               'screening_only':True}}

