"""Common point, marginal, spatial, and temporal scores with paired summaries."""
from __future__ import annotations
import numpy as np
from ..scoring import crps_ensemble, energy_score, geometry_pairs, variogram_score, paired_bootstrap
from .contracts import Forecast


def score_forecast(forecast: Forecast, truth, *, levels=(.5,.8,.9,.95),
                   pair_count=512, seed=0):
    y=np.asarray(truth,float)
    if y.shape!=forecast.mean.shape or not np.isfinite(y).all():
        raise ValueError("Finite truth must match the forecast lead/site axes")
    error=forecast.mean-y
    abs_error=np.abs(forecast.median-y)
    result={"rmse":float(np.sqrt(np.mean(error**2))),
            "mae":float(abs_error.mean()),
            "rmse_by_lead":np.sqrt(np.mean(error**2,axis=1)).tolist(),
            "mae_by_lead":abs_error.mean(axis=1).tolist(),
            "mean_error_by_lead":error.mean(axis=1).tolist(),
            "point_policy":{"squared_loss":"predictive_mean", "absolute_loss":"predictive_median"},
            "leads":len(y), "sites":y.shape[1], "intervals":{}, "bands":{}}
    for start,stop in ((1,10),(11,30),(31,len(y))):
        stop=min(stop,len(y))
        if start<=stop:
            result["bands"][f"{start}-{stop}"]={
                "rmse":float(np.sqrt(np.mean(error[start-1:stop]**2))),
                "mae":float(abs_error[start-1:stop].mean())}
    x=forecast.samples
    if x is None:
        result.update(uncertainty="point", paths=0)
        return result
    crps=crps_ensemble(x,y)
    result.update(crps=float(crps.mean()), crps_by_lead=crps.mean(axis=1).tolist(), paths=len(x),
                  crps_estimator="exact empirical predictive distribution; V-statistic",
                  uncertainty=forecast.metadata.get("uncertainty","joint_samples"))
    for key in result["bands"]:
        start,stop=map(int,key.split("-"))
        result["bands"][key]["crps"]=float(crps[start-1:stop].mean())
    for level in levels:
        if not 0<level<1:
            raise ValueError("Interval levels must lie in (0,1)")
        alpha=1-level
        lo,hi=np.quantile(x,[alpha/2,1-alpha/2],axis=0)
        covered=(lo<=y)&(y<=hi)
        width=hi-lo
        loss=width+(2/alpha)*(np.maximum(lo-y,0)+np.maximum(y-hi,0))
        result["intervals"][f"{level:g}"]={"coverage":float(covered.mean()),
            "width":float(width.mean()), "interval_score":float(loss.mean()),
            "coverage_by_lead":covered.mean(axis=1).tolist(),
            "field_coverage":float(np.all(covered,axis=1).mean()),
            "whole_path_covered":bool(np.all(covered)),
            "domain":"marginal equal-tail intervals; joint coverage is descriptive"}
    energy=energy_score(x,y,pair_count=pair_count,seed=seed)
    result.update(field_energy=float(energy.mean()),field_energy_by_lead=energy.tolist(),
                  energy_pairs=pair_count)
    if y.shape[1]>1:
        pairs,weights=geometry_pairs(forecast.coordinates,max_pairs=pair_count,seed=seed)
        vg=variogram_score(x,y,pairs=pairs,weights=weights)
        result.update(spatial_variogram=float(vg.mean()),spatial_variogram_by_lead=vg.tolist(),
                      spatial_pairs=pairs.tolist(),spatial_pair_weights=weights.tolist(),variogram_power=.5)
    lag_losses={}
    for lag in (1,5,20):
        if lag<len(y):
            expected=np.mean(np.abs(x[:,lag:]-x[:,:-lag])**.5,axis=0)
            observed=np.abs(y[lag:]-y[:-lag])**.5
            lag_losses[str(lag)]=float(np.mean((observed-expected)**2))
    result["temporal_variogram"]=lag_losses
    # These fixed functionals inspect temporal/spatial dependence without using
    # future data to choose a favorable projection.
    fx={"path_spatial_mean":x.mean(axis=(1,2)), "final_spatial_mean":x[:,-1].mean(axis=1),
        "path_maximum":x.max(axis=(1,2)), "path_minimum":x.min(axis=(1,2))}
    fy={"path_spatial_mean":y.mean(), "final_spatial_mean":y[-1].mean(),
        "path_maximum":y.max(), "path_minimum":y.min()}
    result["functional_crps"]={key:float(crps_ensemble(value,np.asarray(fy[key]))) for key,value in fx.items()}
    return result


def aggregate_results(receipts, *, seed=0, bootstrap_resamples=2000):
    """One independent dataset is the uncertainty unit, including failed fits."""
    groups={}
    for receipt in receipts:
        key=(receipt["scenario"],receipt["track"],receipt["model_id"])
        groups.setdefault(key,[]).append(receipt)
    table=[]
    for key,rows in sorted(groups.items()):
        qualified=[r for r in rows if r.get("qualified",False)]
        row={"scenario":key[0],"track":key[1],"model_id":key[2],
             "attempted":len(rows),"completed":sum(r.get("completed",False) for r in rows),
             "qualified":len(qualified),"fallback":sum(r.get("fallback",False) for r in rows),
             "selection_metric":rows[0]["selection_metric"],
             "fit_seconds_total":sum(r.get("fit_seconds",0.) for r in rows),
             "forecast_seconds_total":sum(r.get("forecast_seconds",0.) for r in rows),
             "tuning_seconds_total":sum(r.get("tuning_seconds",0.) for r in rows),
             "total_seconds_including_tuning":sum(r.get("total_seconds",0.)+r.get("tuning_seconds",0.) for r in rows),
             "mcse_units":"original response units"}
        for metric in ("rmse","mae"):
            values=[r["operational_scores"][metric] for r in rows]
            row[metric]=float(np.mean(values))
            row[metric+"_mcse"]=float(np.std(values,ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
        probabilities=[r for r in qualified if "crps" in r.get("native_scores",{})]
        row["probability_population"]=len(probabilities)
        for metric in ("crps","field_energy","spatial_variogram"):
            values=[r["native_scores"][metric] for r in probabilities if metric in r["native_scores"]]
            row[metric]=float(np.mean(values)) if values else None
            row[metric+"_mcse"]=float(np.std(values,ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
        coverages=[r["native_scores"]["intervals"]["0.9"]["coverage"] for r in probabilities]
        row["coverage_90"]=float(np.mean(coverages)) if coverages else None
        table.append(row)
    comparisons=[]
    for scenario,track,_ in sorted(groups):
        reference=groups.get((scenario,track,"bayes_esn_ridge"))
        if reference is None:
            continue
        a={r["replication"]:r for r in reference}
        for model in sorted({key[2] for key in groups if key[:2]==(scenario,track)}-{"bayes_esn_ridge"}):
            # Process each scenario/track once despite the outer three-part key.
            if any(r["scenario"]==scenario and r["track"]==track and r["other_model"]==model for r in comparisons):
                continue
            b={r["replication"]:r for r in groups[(scenario,track,model)]}
            paired=sorted(set(a)&set(b))
            item={"scenario":scenario,"track":track,"reference":"bayes_esn_ridge","other_model":model}
            for metric in ("rmse","crps"):
                valid=paired if metric=="rmse" else [r for r in paired if a[r]["qualified"] and b[r]["qualified"] and
                       metric in a[r].get("native_scores",{}) and metric in b[r].get("native_scores",{})]
                source="operational_scores" if metric=="rmse" else "native_scores"
                av=np.array([a[r][source][metric]/a[r]["training_scale"] for r in valid])
                bv=np.array([b[r][source][metric]/b[r]["training_scale"] for r in valid])
                item[metric+"_population"]=len(valid)
                if len(valid)>=2:
                    summary=paired_bootstrap(av,bv,n_resamples=bootstrap_resamples,seed=seed)
                    item[metric]={k:v for k,v in summary.items() if k!="bootstrap_means"}
                    item[metric]["mcse"]=float(np.std(av-bv,ddof=1)/np.sqrt(len(valid)))
                    item[metric]["units"]="dimensionless; each replication divided by its training response scale"
                elif len(valid)==1:
                    item[metric]={"difference":float(av[0]-bv[0]),"mcse":None,"ci":None,
                                  "paired_differences":(av-bv).tolist(),"replications":1,
                                  "units":"dimensionless; divided by training response scale",
                                  "sign_convention":"reference minus other; negative favors reference"}
            comparisons.append(item)
    return {"table":table,"paired_comparisons":comparisons,
            "uncertainty_unit":"independent complete DGP replication; no site/time pseudo-replication",
            "probability_population":"qualified native fits; paired scores use the common qualified intersection",
            "point_population":"all attempts, with declared point-only fallback on failure"}
