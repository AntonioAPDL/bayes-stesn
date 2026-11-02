"""Blockwise post-fit calibration; raw laws and final labels remain separate."""
from __future__ import annotations
from dataclasses import asdict
import numpy as np
from scipy.spatial.distance import cdist
from ..calibration import fit_standardized_calibration, calibrated_intervals, CalibrationMap
from ..io import seed_for, save_state, load_state
from .contracts import Panel, select_panel
from .metrics import score_forecast


def _scale(forecast, training_scale):
    if forecast.samples is None or len(forecast.samples)<2:
        return np.full(forecast.mean.shape, training_scale)
    return np.maximum(np.std(forecast.samples,axis=0,ddof=1), .01*training_scale)


def calibrate_frozen_model(model, observed_history: Panel, *, origins, horizon,
                           calibration_coordinates=None, calibration_site_ids=None,
                           calibration_observations=None, levels=(.5,.8,.9,.95),
                           n_paths=256, seed=0):
    """Collect calibration block errors without refitting the base parameters.

    ``observed_history`` contains only O. For new-site calibration the optional
    response panel is C, disjoint from O and the final U; the enclosing study
    enforces those roles. C observations are used only after each forecast.
    Blocks must be disjoint and strictly follow model training. No independence
    of serial blocks or distribution-free time-series guarantee is asserted.
    """
    training=getattr(model,"training",getattr(model,"data",None))
    if not isinstance(training,Panel):
        raise TypeError("Fitted model must expose its observation-only training panel")
    origins=tuple(origins)
    if not origins or any(not isinstance(o,(int,np.integer)) for o in origins):
        raise ValueError("origins must be nonempty integer indices")
    if any(b-a<horizon for a,b in zip(origins,origins[1:])) or origins[0]<len(training.y):
        raise ValueError("Calibration blocks must not overlap training or one another")
    if origins[-1]+horizon>len(observed_history.y):
        raise ValueError("Observed calibration history is too short")
    coordinates=training.coordinates if calibration_coordinates is None else np.asarray(calibration_coordinates,float)
    ids=training.site_ids if calibration_site_ids is None else tuple(calibration_site_ids)
    if calibration_coordinates is None:
        target_y=observed_history.y
        transfer=False
    else:
        if set(ids)&set(training.site_ids):
            raise ValueError("C labels must be disjoint from base-model sites")
        target_y=np.asarray(calibration_observations,float)
        if target_y.shape!=(len(observed_history.y),len(coordinates)) or not np.isfinite(target_y).all():
            raise ValueError("C observations must align with the calibration history")
        transfer=True
    training_scale=float(np.std(training.y))
    if training_scale<=0:
        raise ValueError("Calibration needs positive training response scale")
    truth=[];centers=[];scales=[]
    for origin in origins:
        visible=select_panel(observed_history,stop=origin)
        prediction=model.forecast(horizon,target_coordinates=coordinates,target_site_ids=ids,
                    n_paths=n_paths,seed=seed_for(seed,"calibration_block",origin),history=visible)
        centers.append(prediction.median);scales.append(_scale(prediction,training_scale))
        truth.append(target_y[origin:origin+horizon])
    truth,centers,scales=map(np.asarray,(truth,centers,scales))
    maps={}
    for target in ("marginal","horizon","simultaneous"):
        for level in levels:
            maps[f"{target}_{level:g}"]=fit_standardized_calibration(truth,centers,scales,
                alpha=1-level,target=target,scale_floor=.01*training_scale,
                metadata={"origins":list(origins),"site_ids":ids,"parameter_policy":"frozen",
                          "state_policy":"condition O between blocks; open loop within block",
                          "transfer_to_U":transfer,"exchangeability_claim":False})
    return {"maps":maps,"coordinates":coordinates,"site_ids":ids,
            "training_scale":training_scale,"origins":origins,"horizon":horizon,
            "transfer":transfer,"blocks":len(origins)}


def evaluate_calibration(calibration, forecast, truth, *, transfer=False):
    """Evaluate fixed bands and WIS. Calibration never manufactures joint draws."""
    y=np.asarray(truth,float)
    if y.shape!=forecast.mean.shape:
        raise ValueError("truth/forecast shape mismatch")
    center=forecast.median;scale=_scale(forecast,calibration["training_scale"])
    if transfer:
        nearest=np.argmin(cdist(forecast.coordinates,calibration["coordinates"]),axis=1)
    elif tuple(forecast.site_ids)!=tuple(calibration["site_ids"]):
        raise ValueError("Calibration site IDs changed; declare empirical transfer explicitly")
    scores={};bands={};wis=[]
    for key,cal in calibration["maps"].items():
        if transfer:
            q=cal.quantiles[:,nearest] if cal.target=="marginal" else cal.quantiles
            radius=q*np.maximum(scale,cal.scale_floor)
            lo,hi=center-radius,center+radius
        else:
            lo,hi=calibrated_intervals(cal,center,scale)
        covered=(lo<=y)&(y<=hi);width=hi-lo
        penalty=(2/cal.alpha)*(np.maximum(lo-y,0)+np.maximum(y-hi,0))
        loss=width+penalty
        scores[key]={"coverage":float(np.mean(covered)),"width":float(np.mean(width)),
                     "interval_score":float(np.mean(loss)),"field_coverage":float(np.all(covered,axis=1).mean()),
                     "whole_path_covered":bool(np.all(covered)),"infinite_band":bool(np.isinf(width).any()),
                     "blocks":cal.blocks,"rank":int(np.ceil((cal.blocks+1)*(1-cal.alpha))),
                     "transfer_to_U":bool(transfer),"target":cal.target,
                     "coverage_claim":"empirical transfer only" if transfer else "empirical under serial dependence"}
        bands[key]=(lo,hi)
        if cal.target=="marginal":
            wis.append((cal.alpha/2)*loss)
    weighted=(.5*np.abs(y-center)+sum(wis))/(len(wis)+.5)
    return {"raw":score_forecast(forecast,y),"calibrated":scores,
            "wis_marginal":float(np.mean(weighted)),"bands":bands,
            "distribution_added":False,"raw_samples_unchanged":True}


def save_calibration(calibration,path):
    state=dict(calibration)
    state["maps"]={key:asdict(value) for key,value in state["maps"].items()}
    return save_state(path,state,kind="benchmark_calibration")


def load_calibration(path):
    state=load_state(path,kind="benchmark_calibration")
    state["maps"]={key:CalibrationMap(**value) for key,value in state["maps"].items()}
    return state
