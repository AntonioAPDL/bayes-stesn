"""A connected finite reservoir-index model for shared ridge hyperparameters.

All models use the same proper hyperprior and the same observations. Coefficient
dimensions may differ: normalized Gaussian integrated likelihoods retain their
dimension-dependent determinant terms. No trans-dimensional move is hidden here.
"""
from __future__ import annotations
import numpy as np
from .ensemble import sample_model_index
from .gaussian import GaussianRegression
from .priors import PriorRuntime
from .sampling import SamplerConfig, RegressionPosterior
from .model import FittedModel
from .workflow import FittedEnsemble, _alignment
from .io import content_hash


def fit_reservoir_index(prepared_models, config=None, *, model_prior=None):
    """Fit p(M,theta|Y), then retain its coherent empirical joint mixture.

    Initial supported family: iid Gaussian residuals, a scalar ridge prior,
    and a common covariance specification with fixed smoothness. Common
    physical PC scales are supported. Other priors must use the explicit
    low-level sample_model_index callback API with verified common dimensions.
    """
    prepared=tuple(prepared_models)
    if len(prepared)<2:
        raise ValueError("Reservoir index inference needs at least two feature maps")
    _alignment(prepared)
    first=prepared[0]
    for p in prepared:
        if (p.spec.innovations!='iid' or p.spec.prior.kind!='ridge' or p.spec.covariance.infer_nu
            or p.spec.prior!=first.spec.prior or p.spec.covariance!=first.spec.covariance):
            raise ValueError("High-level index inference requires shared ridge/covariance priors, iid Gaussian residuals and fixed smoothness")
    config=SamplerConfig() if config is None else config
    if config.temporal!='iid' or config.thin!=1:
        raise ValueError("Index sampler requires iid temporal policy and thin=1")
    if config.backend not in {'auto','spectral_gaussian'} or config.algorithm not in {'auto','collapsed'}:
        raise ValueError("Connected index inference uses normalized spectral collapse")
    runtimes=[PriorRuntime(p.spec.prior,p.Phi.shape[1],p.Z.shape[1],p.groups,p.Phi,p.reference_Z) for p in prepared]
    engines=[GaussianRegression(p.Y,p.Phi,p.fit_Z,config.a0,config.b0,config.dense_threshold,config.max_dense_bytes) for p in prepared]
    covariance=first.spec.covariance
    coordinates=first.data.coordinates
    prior_dimension=runtimes[0].dimension
    initial=np.r_[runtimes[0].vector(runtimes[0].initial_state()),
                  covariance.parameters_to_vector(covariance.initial_parameters(coordinates),coordinates)]
    initial=np.repeat(initial[None,:],config.chains,axis=0)
    initial+=config.initial_jitter*np.random.default_rng(config.seed).normal(size=initial.shape)

    def decode(model,theta):
        ps=runtimes[model].state_from_vector(theta[:prior_dimension])
        cp=covariance.vector_to_parameters(theta[prior_dimension:],coordinates)
        return ps,cp

    def likelihood(model,theta):
        try:
            ps,cp=decode(model,theta)
            R=covariance.covariance(coordinates,cp)
            fixed=np.exp(2*cp['log_sigma_e']) if covariance.scale_mode=='separate' else None
            return engines[model].evaluate(R,precision=runtimes[model].precision(ps,R),sigma2=fixed).log_marginal
        except (ValueError,np.linalg.LinAlgError,OverflowError,FloatingPointError):
            return -np.inf

    def hyperprior(model,theta):
        ps,cp=decode(model,theta)
        return runtimes[model].log_prior(ps)+covariance.log_prior(cp,coordinates)

    index=sample_model_index(likelihood,hyperprior,initial,len(prepared),model_prior=model_prior,
             draws=config.draws,warmup=config.warmup,proposal_scale=config.proposal_scale,
             model_dimensions=[len(initial[0])]*len(prepared),seed=config.seed)
    components=[];weights=[];component_ids=[]
    for model,p in enumerate(prepared):
        chosen=index.theta[index.model_index==model]
        if len(chosen)==0:
            continue
        states=[];density=[]
        for theta in chosen:
            ps,cp=decode(model,theta)
            states.append({'prior_state':ps,'covariance_parameters':cp,'temporal_parameters':{}})
            density.append(likelihood(model,theta)+hyperprior(model,theta))
        posterior=RegressionPosterior(p.Y,p.Phi,p.fit_Z,p.data.coordinates,p.groups,p.spec.prior,
            p.spec.covariance,config,[states],np.asarray(density)[None,:],np.array([float(index.acceptance_rate.mean())]),
            {'algorithm':'finite_index_conditional_recovery','original_model_index':model,
             'diagnostic_policy':'Use the joint model-index chain; conditional component samples are pooled'},p.reference_Z)
        identity=content_hash({'prepared':p.identity,'states':states})
        components.append(FittedModel(p,posterior,config,{'finite_index':True},identity))
        weights.append(index.model_probabilities[model]);component_ids.append(model)
    metadata={'weighting':'Bayesian common-dimensional reservoir index','experimental':True,
              'original_component_indices':component_ids,
              'index_trace':index.model_index,'theta_trace':index.theta,
              'acceptance':index.acceptance_rate,'posterior_model_probabilities':index.model_probabilities,
              'unvisited_models':'zero empirical posterior mass; finite-chain uncertainty remains',
              'parameters_fixed_per_trajectory':True}
    return FittedEnsemble(tuple(components),np.asarray(weights),metadata)

