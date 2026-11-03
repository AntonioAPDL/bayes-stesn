"""Common point, marginal, spatial, and temporal scores with paired summaries."""
from __future__ import annotations
from statistics import NormalDist
import numpy as np
from ..io import content_hash, seed_for
from ..scoring import crps_ensemble, energy_score, geometry_pairs, variogram_score, paired_bootstrap
from .contracts import Forecast


def training_normalization(simulation, protocol):
    """Return a training-only scale and its explicit reference provenance.

    A learning curve supplies one common reference window/site subset in every
    condition. Ordinary legacy profiles retain their original training-window,
    population-SD convention exactly. The evaluator calls this helper; models
    never receive the simulation object or these response summaries.
    """
    supplied = getattr(protocol, "normalization_reference", None)
    reference = ({"start": protocol.training_start, "stop": protocol.initial_fit,
                  "observed_count": protocol.score_observed_count or protocol.n_observed,
                  "floor": 1e-12, "ddof": 0} if supplied is None else dict(supplied))
    reference.setdefault("floor", 1e-12)
    reference.setdefault("ddof", 0)
    if set(reference) != {"start", "stop", "observed_count", "floor", "ddof"}:
        raise ValueError("Normalization reference needs start/stop/observed_count/floor/ddof")
    for key in ("start", "stop", "observed_count", "ddof"):
        value = reference[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"Normalization {key} must be an integer")
    start, stop, count = (int(reference[key]) for key in ("start", "stop", "observed_count"))
    ddof, floor = int(reference["ddof"]), float(reference["floor"])
    if (not protocol.training_start <= start < stop <= protocol.initial_fit or
            not 1 <= count <= protocol.n_observed or ddof not in (0, 1) or
            (stop-start)*count <= ddof or not np.isfinite(floor) or floor <= 0):
        raise ValueError("Normalization must use a nonempty common permitted training window/site subset")
    sites = np.asarray(simulation.observed_indices[:count], int)
    values = np.asarray(simulation.observation[start:stop, sites], float)
    if values.shape != (stop-start, count) or not np.isfinite(values).all():
        raise ValueError("Normalization reference observations are unavailable or nonfinite")
    ids = tuple(simulation.site_ids[i] for i in sites)
    scale = max(float(np.std(values, ddof=ddof)), floor)
    metadata = {"start": start, "stop": stop, "observed_count": count,
                "site_ids": list(ids), "floor": floor, "ddof": ddof,
                "sample_count": int(values.size), "scale": scale,
                "policy": "legacy_training_window" if supplied is None else "common_training_reference",
                "reference_data_hash": content_hash({"observations": values, "site_ids": ids,
                    "start": start, "stop": stop}),
                "information_policy": "observed sites before first validation; no final or latent responses"}
    metadata["identity"] = content_hash(metadata)
    return scale, metadata


def qualification_interval(successes, attempts, *, level=.95):
    """Wilson interval for paired qualification, not a convergence guarantee."""
    if (isinstance(successes, (bool, np.bool_)) or isinstance(attempts, (bool, np.bool_)) or
            not isinstance(successes, (int, np.integer)) or not isinstance(attempts, (int, np.integer)) or
            not 0 <= successes <= attempts or not 0 < level < 1):
        raise ValueError("Invalid joint qualification count or confidence level")
    if not attempts:
        return {"qualified_pairs": 0, "attempted_pairs": 0, "rate": None,
                "interval": None, "level": level, "method": "Wilson score"}
    rate = successes/attempts
    z = NormalDist().inv_cdf((1+level)/2)
    denominator = 1+z*z/attempts
    center = (rate+z*z/(2*attempts))/denominator
    half = z*np.sqrt(rate*(1-rate)/attempts+z*z/(4*attempts*attempts))/denominator
    return {"qualified_pairs": int(successes), "attempted_pairs": int(attempts),
            "rate": float(rate), "interval": [float(max(0., center-half)), float(min(1., center+half))],
            "level": level, "method": "Wilson score"}


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


def _operational(receipt):
    state = receipt.get("job_state", receipt.get("status"))
    if (receipt.get("process_failure", False) or state in
            {"pending", "running", "interrupted", "retryable_infrastructure_interruption",
             "unresolved_infrastructure_failure", "invalidated"}):
        return False
    scores = receipt.get("operational_scores", {})
    return all(key in scores and np.isfinite(scores[key]) for key in ("rmse", "mae"))


def _native(receipt, metric="crps"):
    return (_operational(receipt) and bool(receipt.get("qualified", False)) and
            not receipt.get("fallback", False) and
            metric in receipt.get("native_scores", {}) and
            np.isfinite(receipt["native_scores"][metric]))


def _normalization_pair(a, b):
    scales = [float(row["training_scale"]) for row in (a, b)]
    if any(not np.isfinite(value) or value <= 0 for value in scales):
        raise ValueError("Paired losses need positive finite training scales")
    if not np.isclose(*scales, rtol=1e-12, atol=0):
        raise ValueError("Paired methods must use the same per-dataset normalization")
    refs = [row.get("normalization_reference") for row in (a, b)]
    if all(isinstance(value, dict) for value in refs):
        identities = [value.get("identity") for value in refs]
        if all(identities) and identities[0] != identities[1]:
            raise ValueError("Paired methods have different normalization reference identities")
    return scales[0]


def aggregate_results(receipts, *, seed=0, bootstrap_resamples=2000,
                      primary_comparisons=None):
    """Aggregate terminal scientific outcomes using independent datasets.

    Infrastructure-only records may be supplied for accounting, but are not
    converted into method failures or silently included in point-score means.
    Legacy receipts without explicit job states retain their original meaning.
    """
    groups = {}
    seen = set()
    for receipt in receipts:
        key = (receipt["scenario"], receipt["track"], receipt["model_id"])
        identity = key + (receipt["replication"],)
        if identity in seen:
            raise ValueError("Duplicate dataset/model receipt; execution retries are not independent replications")
        seen.add(identity)
        groups.setdefault(key, []).append(receipt)
    table = []
    for key, rows in sorted(groups.items()):
        operational = [r for r in rows if _operational(r)]
        qualified = [r for r in operational if r.get("qualified", False) and not r.get("fallback", False)]
        row = {"scenario": key[0], "track": key[1], "model_id": key[2],
               "scheduled": len(rows), "attempted": len(operational),
               "completed": sum(bool(r.get("completed", False)) for r in operational),
               "qualified": len(qualified), "fallback": sum(bool(r.get("fallback", False)) for r in operational),
               "incomplete": len(rows)-len(operational), "operational_population": len(operational),
               "qualification_rate": len(qualified)/len(operational) if operational else None,
               "selection_metric": rows[0].get("selection_metric"),
               "fit_seconds_total": sum(r.get("fit_seconds", 0.) for r in rows),
               "forecast_seconds_total": sum(r.get("forecast_seconds", 0.) for r in rows),
               "tuning_seconds_total": sum(r.get("tuning_seconds", 0.) for r in rows),
               "total_seconds_including_tuning": sum(r.get("total_pipeline_seconds",
                   r.get("total_seconds", 0.)+r.get("tuning_seconds", 0.)) for r in rows),
               "total_pipeline_cpu_seconds": (sum(r["total_pipeline_cpu_seconds"] for r in rows)
                   if all(r.get("total_pipeline_cpu_seconds") is not None for r in rows) else None),
               "compute_accounting_complete": all(r.get("compute_accounting_complete",False) for r in rows),
               "mcse_units": "original response units",
               "aggregation": "mean dataset-level loss; RMSE is not pooled root-MSE"}
        for metric in ("rmse", "mae"):
            values = [r["operational_scores"][metric] for r in operational]
            row[metric] = float(np.mean(values)) if values else None
            row[metric+"_mcse"] = float(np.std(values, ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
        probabilities = [r for r in qualified if _native(r)]
        row["probability_population"] = len(probabilities)
        for metric in ("crps", "field_energy", "spatial_variogram"):
            values = [r["native_scores"][metric] for r in probabilities if metric in r["native_scores"]]
            row[metric] = float(np.mean(values)) if values else None
            row[metric+"_population"] = len(values)
            row[metric+"_mcse"] = float(np.std(values, ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
        intervals = [r["native_scores"].get("intervals", {}).get("0.9", {}) for r in probabilities]
        for output, field in (("coverage_90", "coverage"), ("width_90", "width"),
                              ("interval_score_90", "interval_score")):
            values = [value[field] for value in intervals if field in value]
            row[output] = float(np.mean(values)) if values else None
        table.append(row)
    references = {"bayes_esn_ridge"}
    if primary_comparisons is not None:
        references.update(spec["reference"] for spec in canonical_primary_comparisons(primary_comparisons))
    comparisons = []
    for scenario, track in sorted({key[:2] for key in groups}):
        for reference_id in sorted(references):
            reference = groups.get((scenario, track, reference_id))
            if reference is None:
                continue
            a = {r["replication"]: r for r in reference}
            for model in sorted({key[2] for key in groups if key[:2] == (scenario, track)}-{reference_id}):
                b = {r["replication"]: r for r in groups[(scenario, track, model)]}
                paired = sorted(set(a) & set(b))
                attempted = [rep for rep in paired if _operational(a[rep]) and _operational(b[rep])]
                item = {"scenario": scenario, "track": track, "reference": reference_id,
                        "other_model": model, "paired_scheduled_population": len(paired),
                        "paired_attempted_population": len(attempted),
                        "attempted_replications": attempted,
                        "incomplete_pair_population": len(paired)-len(attempted)}
                for metric in ("rmse", "crps"):
                    valid = attempted if metric == "rmse" else [rep for rep in attempted if _native(a[rep]) and _native(b[rep])]
                    source = "operational_scores" if metric == "rmse" else "native_scores"
                    scales = [_normalization_pair(a[rep], b[rep]) for rep in valid]
                    av = np.array([a[rep][source][metric]/scale for rep, scale in zip(valid, scales)], float)
                    bv = np.array([b[rep][source][metric]/scale for rep, scale in zip(valid, scales)], float)
                    item[metric+"_population"] = len(valid)
                    item[metric+"_replications"] = valid
                    item[metric+"_attrition"] = qualification_interval(len(valid), len(attempted))
                    if len(valid) >= 2:
                        summary = paired_bootstrap(av, bv, n_resamples=bootstrap_resamples, seed=seed)
                        item[metric] = {k: v for k, v in summary.items() if k != "bootstrap_means"}
                        item[metric]["mcse"] = float(np.std(av-bv, ddof=1)/np.sqrt(len(valid)))
                    elif len(valid) == 1:
                        item[metric] = {"difference": float(av[0]-bv[0]), "mcse": None, "ci": None,
                                        "paired_differences": (av-bv).tolist(), "replications": 1,
                                        "sign_convention": "reference minus other; negative favors reference"}
                    if valid:
                        item[metric]["units"] = "dimensionless; paired methods share each dataset's training-only response scale"
                        item[metric]["paired_replications"] = valid
                        item[metric]["population"] = "operational paired outcomes" if metric == "rmse" else "conditional common-qualified native distributions"
                comparisons.append(item)
    result = {"table": table, "paired_comparisons": comparisons,
              "uncertainty_unit": "independent complete DGP replication; no site/time pseudo-replication",
              "probability_population": "qualified native fits; paired scores use the common qualified intersection",
              "point_population": "completed scientific attempts, with declared point-only fallback on method failure; infrastructure interruptions excluded",
              "incomplete_receipts": sum(row["incomplete"] for row in table)}
    if primary_comparisons is not None:
        result["primary_comparisons"] = canonical_primary_comparisons(primary_comparisons)
        result["macro_comparisons"] = macro_comparisons(result, primary_comparisons,
                                                       seed=seed, bootstrap_resamples=bootstrap_resamples)
    return result


def canonical_primary_comparisons(comparisons):
    """Validate fixed weighted scenario contrasts; never renormalize omissions."""
    if not isinstance(comparisons, (list, tuple)) or not comparisons:
        raise ValueError("Primary comparisons must be a nonempty sequence")
    output, seen = [], set()
    allowed = {"id", "reference", "other_model", "track", "metric", "scenario_weights"}
    for raw in comparisons:
        if not isinstance(raw, dict) or set(raw)-allowed:
            raise ValueError("Unknown primary comparison fields")
        reference = str(raw.get("reference", "bayes_esn_ridge"))
        other = raw.get("other_model")
        track, metric = raw.get("track"), raw.get("metric", "crps")
        weights = raw.get("scenario_weights")
        if (not isinstance(other, str) or not other or other == reference or
                track not in {"A", "B"} or metric not in {"rmse", "crps"} or
                not isinstance(weights, dict) or not weights):
            raise ValueError("Primary contrast needs distinct models, A/B, rmse/crps and fixed scenario weights")
        weights = {str(key): float(value) for key, value in sorted(weights.items())}
        if (any(not np.isfinite(value) or value <= 0 for value in weights.values()) or
                not np.isclose(sum(weights.values()), 1., rtol=0, atol=1e-12)):
            raise ValueError("Scenario weights must be positive finite values summing to one")
        # Stress arms share underlying innovations with their controls. This
        # explicitly independent-stratum estimator cannot claim their omitted
        # covariance is zero; those arms retain paired cell-level reporting.
        for stress, control in (("g5_noise_stress", "g4_existing_fronts"),
                                ("g6_velocity_shift", "g2_spectral_transport")):
            if stress in weights and control in weights:
                raise ValueError("Coupled stress/control strata require a separately specified cluster estimator")
        identity = str(raw.get("id", f"{reference}_vs_{other}_{track}_{metric}"))
        key = (reference, other, track, metric)
        if identity in seen or key in seen:
            raise ValueError("Duplicate primary comparison")
        seen.update((identity, key))
        output.append({"id": identity, "reference": reference, "other_model": other,
                       "track": track, "metric": metric, "scenario_weights": weights})
    return output


def macro_comparisons(summary, primary_comparisons, *, seed=0, bootstrap_resamples=2000):
    """Fixed-weight, independent-scenario paired contrasts and pointwise CIs.

    Every cell uses its own common-qualified population for CRPS. Missing cells
    produce an explicit incomplete contrast, not a change in scenario weights.
    Bootstrap samples whole dataset differences independently inside strata.
    """
    if (isinstance(bootstrap_resamples, (bool, np.bool_)) or
            not isinstance(bootstrap_resamples, (int, np.integer)) or bootstrap_resamples < 1):
        raise ValueError("bootstrap_resamples must be positive")
    lookup = {(row["scenario"], row["track"], row.get("reference", "bayes_esn_ridge"), row["other_model"]): row
              for row in summary.get("paired_comparisons", [])}
    output = []
    for spec in canonical_primary_comparisons(primary_comparisons):
        cells, draws, variance = [], np.zeros(bootstrap_resamples), 0.
        mean, complete, estimated = 0., True, True
        for scenario, weight in spec["scenario_weights"].items():
            row = lookup.get((scenario, spec["track"], spec["reference"], spec["other_model"]), {})
            metric = spec["metric"]
            differences = np.asarray(row.get(metric, {}).get("paired_differences", []), float)
            if differences.ndim != 1 or not np.isfinite(differences).all():
                raise ValueError("Paired differences must be finite one-dimensional arrays")
            reported = row.get(metric+"_population")
            if reported is not None and reported != len(differences):
                raise ValueError("Reported paired population does not match its saved differences")
            # Old summaries recorded operational pair counts under
            # rmse_population. If neither count exists, keep attrition unknown
            # rather than treating every observed qualified pair as a success.
            attempts = row.get("paired_attempted_population", row.get("rmse_population"))
            attrition = (qualification_interval(len(differences), attempts) if attempts is not None else
                         {"qualified_pairs": len(differences), "attempted_pairs": None,
                          "rate": None, "interval": None, "method": "attempt count unavailable"})
            cell = {"scenario": scenario, "weight": weight, "paired_differences": differences.tolist(),
                    "pairs": len(differences), "attempted_pairs": attempts,
                    "paired_replications": row.get(metric+"_replications", row.get(metric, {}).get("paired_replications")),
                    "joint_qualification": attrition,
                    "mean_difference": float(differences.mean()) if len(differences) else None,
                    "paired_variance": float(np.var(differences, ddof=1)) if len(differences)>1 else None}
            cells.append(cell)
            if not len(differences):
                complete = estimated = False
                continue
            mean += weight*float(differences.mean())
            if len(differences) < 2:
                estimated = False
                continue
            variance += weight*weight*float(np.var(differences, ddof=1))/len(differences)
            rng = np.random.default_rng(seed_for(seed, "macro_bootstrap", spec["id"], scenario))
            for start in range(0, bootstrap_resamples, 1024):
                count = min(1024, bootstrap_resamples-start)
                indices = rng.integers(len(differences), size=(count, len(differences)))
                draws[start:start+count] += weight*differences[indices].mean(axis=1)
        output.append({**spec, "complete": complete, "cells": cells,
                       "difference": float(mean) if complete else None,
                       "mcse": float(np.sqrt(variance)) if estimated else None,
                       "ci": np.quantile(draws, [.025, .975]).tolist() if estimated else None,
                       "level": .95, "interval_scope": "pointwise; no simultaneous superiority claim",
                       "variance_method": "sum of squared fixed weights times within-stratum paired variance divided by usable pairs",
                       "resampling_unit": "independent DGP datasets within independent scenario strata",
                       "population": "cell-specific common-qualified native distributions" if spec["metric"] == "crps" else "paired operational outcomes",
                       "sign_convention": "reference minus other; negative favors reference"})
    return output
