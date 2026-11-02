"""Aligned ensembles, validation stacking and scoring convenience functions."""
from __future__ import annotations
from dataclasses import dataclass, field, replace
import numpy as np
from .data import EvaluationTruth, readonly
from .model import FittedModel, fit
from .prediction import PredictiveSample, simulate_predictive
from .ensemble import mix_trajectories, fit_crps_stacking, fit_logscore_stacking
from .scoring import score_predictions
from .io import content_hash, seed_for, save_state, load_state


@dataclass(frozen=True)
class EnsembleSpec:
    weighting: str = "equal"

    def __post_init__(self):
        if self.weighting not in {"equal", "validation_crps", "validation_log_score"}:
            raise ValueError("Unknown ensemble weighting method")


@dataclass(frozen=True)
class FittedEnsemble:
    components: tuple
    weights: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self,"components",tuple(self.components))
        weights=readonly(self.weights)
        if not self.components or weights.shape!=(len(self.components),) or np.any(weights<0) or not np.isfinite(weights).all() or not np.isclose(weights.sum(),1):
            raise ValueError("Ensemble needs fitted components and nonnegative simplex weights")
        object.__setattr__(self,"weights",weights)

    @property
    def posterior_identity(self):
        return content_hash({"components":[f.posterior_identity for f in self.components],"weights":self.weights})

    def save(self, path):
        return save_state(path, {"components":[f.to_state() for f in self.components],
                                "weights":self.weights,"metadata":self.metadata},kind="fitted_ensemble")

    @classmethod
    def load(cls, path):
        state=load_state(path,kind="fitted_ensemble")
        return cls(tuple(FittedModel.from_state(s) for s in state['components']),
                   state['weights'],state['metadata'])


def _alignment(prepared):
    first=prepared[0]
    for other in prepared[1:]:
        if other.split!=first.split or other.data.site_ids!=first.data.site_ids or not np.array_equal(other.data.y,first.data.y) or not np.array_equal(other.data.coordinates,first.data.coordinates) or not np.array_equal(other.data.times,first.data.times):
            raise ValueError("Ensemble components must share observed data, geometry and split policy")


def fit_ensemble(prepared_models, config=None, *, spec=None):
    """Fit aligned components; non-equal weights require a subsequent validation step."""
    prepared=tuple(prepared_models)
    if not prepared:
        raise ValueError("At least one prepared model is required")
    _alignment(prepared)
    spec=EnsembleSpec() if spec is None else spec
    components=tuple(fit(p,config) for p in prepared)
    return FittedEnsemble(components,np.ones(len(components))/len(components),
                          {"weighting":"equal", "requested_weighting":spec.weighting,
                           "validation_weights_fitted":False})


def fit_stacking_weights(ensemble, validation_predictions, observed_validation, *,
                         method="crps", block_log_densities=None):
    """Select weights using a declared validation block only.

    Log-score inputs must be correctly integrated *block* densities computed
    with one parameter draw shared across each whole block. The function never
    manufactures a path log score by multiplying marginal forecast densities.
    """
    samples=tuple(validation_predictions)
    if len(samples)!=len(ensemble.components):
        raise ValueError("One aligned validation prediction is required per component")
    shape=samples[0].observation.shape[1:]
    for fitted,sample in zip(ensemble.components,samples):
        if sample.metadata.get('parameter_posterior_identity')!=fitted.posterior_identity:
            raise ValueError("Validation prediction belongs to a different component posterior")
        if sample.metadata['forecast_origin']!=samples[0].metadata['forecast_origin']:
            raise ValueError("Validation prediction origins must agree")
        start=sample.metadata['forecast_origin']+1
        stop=start+sample.observation.shape[1]
        if not any(a<=start and stop<=b for a,b in fitted.prepared.split.validation_blocks):
            raise ValueError("Stacking predictions must lie entirely in a declared validation block")
        if (sample.observation.shape[1:]!=shape or not np.array_equal(sample.target_coordinates,samples[0].target_coordinates)
                or sample.target_site_ids!=samples[0].target_site_ids):
            raise ValueError("Validation prediction domains must align")
    truth=np.asarray(observed_validation,float)
    if truth.shape!=shape or not np.isfinite(truth).all():
        raise ValueError("Validation observations must be finite and aligned")
    if method=="crps":
        result=fit_crps_stacking([s.observation for s in samples],truth)
    elif method=="log_score":
        if block_log_densities is None:
            raise ValueError("Log-score stacking requires normalized joint block log densities")
        result=fit_logscore_stacking(block_log_densities)
    else:
        raise ValueError("Stacking method must be crps or log_score")
    if not result.success:
        raise RuntimeError("Stacking optimizer did not report convergence")
    return FittedEnsemble(ensemble.components,result.weights,
        {"weighting":f"validation_{method}","validation_weights_fitted":True,
         "objective":float(result.objective),"validation_identity":content_hash(truth),
         "prediction_identities":[s.metadata.get('parameter_posterior_identity') for s in samples]})


def simulate_ensemble(ensemble, contexts=None, *, horizon, target_coordinates=None,
                      target_site_ids=None, paths=256, seed=20261102,
                      future_variance_covariates=None):
    contexts=[None]*len(ensemble.components) if contexts is None else list(contexts)
    if len(contexts)!=len(ensemble.components):
        raise ValueError("One forecast context per ensemble component is required")
    samples=[simulate_predictive(f,c,horizon=horizon,target_coordinates=target_coordinates,
                    target_site_ids=target_site_ids,paths=paths,seed=seed_for(seed,'component',j),
                    future_variance_covariates=future_variance_covariates)
             for j,(f,c) in enumerate(zip(ensemble.components,contexts))]
    origins={s.metadata['forecast_origin'] for s in samples}
    if len(origins)!=1:
        raise ValueError("Ensemble forecast origins must agree")
    mixture=mix_trajectories([s.observation for s in samples],ensemble.weights,paths=paths,seed=seed_for(seed,'mixture'))
    values={}
    for key in ('observation','latent','readout_mean','conditional_mean'):
        values[key]=np.stack([getattr(samples[c],key)[d] for c,d in zip(mixture.component_index,mixture.draw_index)])
    parameter_ids=np.stack([samples[c].parameter_draw_ids[d] for c,d in zip(mixture.component_index,mixture.draw_index)])
    metadata=dict(samples[0].metadata)
    metadata.update(parameter_posterior_identity=ensemble.posterior_identity,
                    ensemble_weights=ensemble.weights.tolist(),
                    mixture_component_id_per_path=mixture.component_index.tolist(),
                    component_draw_id_per_path=mixture.draw_index.tolist(),
                    weighting_provenance=ensemble.metadata, seed=int(seed))
    return PredictiveSample(**values,target_coordinates=samples[0].target_coordinates,
                            target_site_ids=samples[0].target_site_ids,
                            parameter_draw_ids=parameter_ids,metadata=metadata)


def score(prediction, truth: EvaluationTruth, *, target="observation", **kwargs):
    if not isinstance(truth,EvaluationTruth):
        raise TypeError("score requires a scoring-only EvaluationTruth")
    if target=="generating_mean":
        observations, draws=truth.generating_mean,prediction.readout_mean
    elif target=="latent":
        observations,draws=truth.latent,prediction.latent
    elif target=="observation":
        observations,draws=truth.observation,prediction.observation
    else:
        raise ValueError("Unknown scientific scoring target")
    if observations is None:
        raise ValueError(f"The requested {target} truth is unavailable")
    result=score_predictions(draws,observations,coordinates=prediction.target_coordinates,**kwargs)
    result.update(target=target,forecast_origin=prediction.metadata['forecast_origin'],
                  response_units=prediction.metadata['response_units'])
    if target=="generating_mean":
        result['interpretation']='Generating-mean reconstruction evaluated using stochastic-feedback readout draws; not observation error'
    return result


@dataclass(frozen=True)
class CalibrationSpec:
    alpha: float = .1
    target: str = "marginal"
    scale_floor: float = 1e-8

    def __post_init__(self):
        if not 0<self.alpha<1 or self.target not in {"marginal","horizon","simultaneous"} or self.scale_floor<=0:
            raise ValueError("Invalid calibration level, domain or scale floor")


def fit_calibration(fitted, calibration_predictions, observed_blocks, *, spec=None):
    """Fit intervals on declared historical calibration blocks; keep draws intact."""
    from .calibration import fit_standardized_calibration
    spec=CalibrationSpec() if spec is None else spec
    samples=tuple(calibration_predictions)
    if not samples:
        raise ValueError("Calibration requires historical forecast blocks")
    components=fitted.components if isinstance(fitted,FittedEnsemble) else (fitted,)
    domain=samples[0].target_coordinates
    origins=[]
    for sample in samples:
        start=sample.metadata['forecast_origin']+1
        stop=start+sample.observation.shape[1]
        if sample.metadata['parameter_posterior_identity']!=fitted.posterior_identity:
            raise ValueError("Calibration prediction belongs to another parameter posterior")
        if not np.array_equal(sample.target_coordinates,domain):
            raise ValueError("Calibration domains must agree")
        for component in components:
            if not any(a<=start and stop<=b for a,b in component.prepared.split.calibration_blocks):
                raise ValueError("Predictions must lie within declared calibration blocks")
        origins.append((start,stop))
    if any(a[1]>b[0] for a,b in zip(sorted(origins),sorted(origins)[1:])):
        raise ValueError("Calibration forecast blocks must not overlap")
    centers=np.stack([s.observation.mean(axis=0) for s in samples])
    scales=np.stack([s.observation.std(axis=0) for s in samples])
    return fit_standardized_calibration(observed_blocks,centers,scales,
        alpha=spec.alpha,target=spec.target,scale_floor=spec.scale_floor,
        metadata={"parameter_posterior_identity":fitted.posterior_identity,
                  "blocks":origins,"domain_identity":content_hash(domain),
                  "forecast_policy":"frozen parameters; historical state updates",
                  "raw_posterior_draws_changed":False})


def apply_calibration(calibration, prediction):
    from .calibration import calibrated_intervals
    if calibration.metadata.get('domain_identity') not in {None,content_hash(prediction.target_coordinates)}:
        raise ValueError("Calibration is for a different target domain")
    if calibration.metadata.get('parameter_posterior_identity') not in {None,prediction.metadata['parameter_posterior_identity']}:
        raise ValueError("Calibration is for a different posterior")
    origin=prediction.metadata['forecast_origin']
    if any(stop>origin+1 for _,stop in calibration.metadata.get('blocks',())):
        raise ValueError("Calibration outcomes must all be available at the forecast origin")
    return calibrated_intervals(calibration,prediction.observation.mean(axis=0),
                                 prediction.observation.std(axis=0))


