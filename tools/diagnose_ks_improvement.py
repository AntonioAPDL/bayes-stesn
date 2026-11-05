#!/usr/bin/env python3
"""Two validation-origin interventions on the retained original A Bayesian fit.

No fit, selection, test-label read, or model change. The existing full
predictive method remains the only qualified posterior predictive ranking.
All other scores are explicitly diagnostic. Example:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTHONPATH=src /path/to/env/bin/python \
  tools/diagnose_ks_improvement.py \
    --cache /path/to/verified/cache \
    --fit /path/to/ks_core_A_bayes_stesn_t00_s20261105.npz \
    --freeze /path/to/ks_frozen_specification_full.json \
    --output /path/to/results/initial_diagnostics
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import time

for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[_key]="1"

import numpy as np
from bayes_stesn.io import atomic_npz, content_hash, seed_for
from bayes_stesn.benchmark.classical import _panel_state
from bayes_stesn.applications.ks_data import KSAdapter, IO_LIMITATION
from bayes_stesn.applications.ks_metrics import score
from bayes_stesn.applications.ks_native import NativeFit, _computation_identity
from bayes_stesn.applications.ks_runtime import atomic_json, now
from bayes_stesn.applications.ks_improvement_diagnostics import (
    MODES, conditional_mean_parameters, iter_diagnostic_forecasts,
    geometry_diagnostics, synchronization_diagnostics,
    teacher_forced_readout, training_readout_residuals,
)

FIT_SHA="cfe7b6598deaf76beb7198232ac8a98c17eb22861956cf811808b78ea0d38049"
FREEZE_SHA="90e9a5e82e077eae15c19ae887496f1ec4b4fc706925d961ae539852efd699e8"
FREEZE_ID="ca6d8dac49208767f0073a747a3fbc7ee37648715a22abc32e539378b75756dc"
NATIVE_ID="31ed706dc9abc19821f7d11b60afecd9aa43096ce5ee064e1b7b42627c7d92c1"
MODEL_SEED=20261105
FORECAST_SEED=11622026
ORIGINS=(6000,6500)
HISTORY,HORIZON,PATHS,BATCH=128,100,256,128
LEADS=(1,10,25,100)
LEVELS=np.array([.05,.25,.5,.75,.95])
MEMORY_CAP=512*1024**2
DISK_RESERVE=12789245542


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024**2),b""):digest.update(block)
    return digest.hexdigest()


def snapshot(path):
    root=Path("/sys/fs/cgroup")
    def value(name):
        p=root/name
        v=p.read_text().strip() if p.exists() else None
        return int(v) if v and v.isdigit() else v
    limit,current=value("memory.max"),value("memory.current")
    return {"process_peak_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
        "container_memory_limit_bytes":limit,"container_memory_current_bytes":current,
        "container_available_bytes":limit-current if isinstance(limit,int) and isinstance(current,int) else None,
        "disk_free_bytes":shutil.disk_usage(path).free,"cpu_affinity_count":len(os.sched_getaffinity(0)),
        "cpu_quota":value("cpu.max")}


def guard(path):
    state=snapshot(path)
    if state["process_peak_rss_bytes"]>MEMORY_CAP:
        raise MemoryError("Diagnostic process exceeded its 512 MiB cap; do not schedule further work")
    if state["container_available_bytes"] is not None and state["container_available_bytes"]<768*1024**2:
        raise MemoryError("Insufficient measured container headroom for another diagnostic batch")
    if state["disk_free_bytes"]<DISK_RESERVE+32*1024**2:
        raise MemoryError("Insufficient disk reserve")
    return state


def verify(args):
    checks={}
    def require(name,condition):
        checks[name]=bool(condition)
        if not condition:raise ValueError("Retained-fit diagnostic binding failed: "+name)
    require("fit_sha256",sha(args.fit)==FIT_SHA)
    require("full_freeze_sha256",sha(args.freeze)==FREEZE_SHA)
    frozen=json.loads(args.freeze.read_text())
    require("freeze_identity",frozen["identity"]==FREEZE_ID and content_hash(
        {k:v for k,v in frozen.items() if k!="identity"})==FREEZE_ID)
    root=Path(__file__).resolve().parents[1]/"src/bayes_stesn"
    original_sources={name:sha(root/name) for name in frozen["source"] if name!="runtime"}
    require("all_51_original_numerical_sources",len(original_sources)==51 and original_sources==
        {k:v for k,v in frozen["source"].items() if k!="runtime"})
    fitted=NativeFit.load(args.fit)
    require("native_fit_identity",fitted.identity==NATIVE_ID)
    require("native_bayesian_runtime_and_sources",_computation_identity("bayes_stesn")==fitted.computation)
    require("qualified_original_posterior",fitted.diagnostics.get("qualified") is True)
    require("model_and_seed",fitted.model_id=="bayes_stesn" and fitted.seed==MODEL_SEED)
    require("frozen_A_bayes03",frozen["selected"]["A"]["bayes_stesn"]["id"]=="bayes_03" and
        json.loads(json.dumps(fitted.config))==frozen["selected"]["A"]["bayes_stesn"]["config"])
    packages={name:importlib.metadata.version(name) for name in ("numpy","scipy","h5py","threadpoolctl")}
    require("reader_and_numerical_package_versions",all(packages[n]==frozen["source"]["runtime"]["packages"][n] for n in packages))
    adapter=KSAdapter(args.cache,"interleaved","core")
    raw=adapter.folder/adapter.reader.files[0]["path"]
    require("raw_trajectory_hash",sha(raw)==adapter.reader.files[0]["sha256"])
    train=adapter.training(0)
    identity=content_hash({"panel":_panel_state(train.panel),"metadata":dict(train.panel.metadata)})
    require("permitted_training_identity",identity==fitted.input_identity)
    require("stored_training_values",np.array_equal(train.panel.y,fitted.training.y))
    require("stored_training_coordinates",np.array_equal(train.panel.coordinates,fitted.training.coordinates))
    require("stored_training_times",np.array_equal(train.panel.times,fitted.training.times))
    require("strict_6000_by_39_training",train.panel.y.shape==(6000,39))
    require("full_training_transform",train.transform.fit_time_bounds==(0,6000) and train.transform.training_value_count==234000)
    source={"script_sha256":sha(Path(__file__)),
        "diagnostic_module_sha256":sha(root/"applications/ks_improvement_diagnostics.py"),
        "original_frozen_source_files":original_sources,"native_computation":fitted.computation,
        "packages":packages}
    binding={"fit_sha256":FIT_SHA,"freeze_sha256":FREEZE_SHA,"freeze_identity":FREEZE_ID,
        "experiment_identity":frozen["experiment_identity"],"native_fit_identity":NATIVE_ID,
        "input_identity":identity,"training_identity":dict(train.identity),"source":source,
        "case":{"phase":"core","track":"A","trajectory":0,"first_targets":list(ORIGINS),
            "history":HISTORY,"horizon":HORIZON,"paths":PATHS,"path_batch":BATCH,
            "model_seed":MODEL_SEED,"forecast_seed_root":FORECAST_SEED},
        "checks":checks,"check_count":len(checks),
        "runtime_scope":"Exact Bayesian native computation and all original source files; not a full neural-study environment"}
    return fitted,adapter,train,binding


def compact_scores(metrics,samples,regime):
    scalar={k:metrics[k] for k in ("rmse","mae","nrmse","crps","normalized_crps") if k in metrics}
    scalar["intervals"]={level:{"coverage":value["coverage"],"width":value["width"]}
        for level,value in metrics["intervals"].items()}
    variance=None if samples is None else np.var(samples,axis=0,ddof=0).mean(axis=1)
    scalar["leads"]=[]
    for lead in LEADS:
        k=lead-1
        row={"lead":lead,"physical_lead_time":.25*lead,
             "rmse":metrics["rmse_by_horizon"][k],"mae":metrics["mae_by_horizon"][k]}
        if "crps_by_horizon" in metrics:row["crps"]=metrics["crps_by_horizon"][k]
        if variance is not None:row["empirical_path_variance_mean_over_sites"]=float(variance[k])
        row["intervals"]={level:{"coverage":value["coverage_by_horizon"][k],"width":value["width_by_horizon"][k]}
            for level,value in metrics["intervals"].items()}
        scalar["leads"].append(row)
    if regime!="teacher_forced_one_step":
        scalar.update(diverged=metrics["diverged"],vpt_steps=metrics["vpt_steps"],
            vpt_right_censored=metrics["vpt_right_censored"])
    return scalar


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--fit",type=Path,required=True)
    parser.add_argument("--freeze",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--receipt",type=Path,default=Path(__file__).resolve().parents[1]/
                        "docs/applications/evidence/ks_improvement_initial_diagnosis.json")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    initial=guard(args.output);started=time.perf_counter()
    try:
        fitted,adapter,train,binding=verify(args)
    except Exception as error:
        atomic_json(args.output/"binding_blocker.json",{"schema":"ks-improvement-initial-diagnosis-blocker/1",
            "status":"blocked_before_any_forecast","error":repr(error),"utc":now(),
            "new_fits":0,"test_label_reads":0,"source_sha256":sha(Path(__file__))})
        raise
    identity=content_hash(binding);state_path=args.output/"diagnosis_state.json"
    if state_path.exists():
        report=json.loads(state_path.read_text())
        if report["identity"]!=identity:raise ValueError("Existing diagnostic state belongs to different data/source/configuration")
        for case in report.get("cases",[]):
            for row in case["artifacts"]:
                p=args.output/row["name"]
                if not p.exists() or sha(p)!=row["sha256"]:raise ValueError("Saved diagnostic artifact checksum mismatch")
        if report["status"]=="complete":
            atomic_json(args.receipt,report)
            print(json.dumps({"status":"reused_completed_diagnosis","identity":identity}),flush=True)
            return
    else:
        point=conditional_mean_parameters(fitted)
        residual=training_readout_residuals(fitted,point=point)
        residual["rmse_physical"]=residual["rmse_fitted_scale"]*train.transform.scale
        residual["mae_physical"]=residual["mae_fitted_scale"]*train.transform.scale
        geometry=geometry_diagnostics(fitted,adapter.coordinates_all[adapter.target_sites("validation")],
            target_site_ids=adapter.target_sites("validation"))
        sync=synchronization_diagnostics(fitted,train.panel,lengths=(128,1024))
        report={"schema":"ks-improvement-initial-diagnosis/1","status":"running","identity":identity,
            "binding":binding,"started_utc":now(),"new_fits":0,"new_selections":0,"test_label_reads":0,
            "selection_use":"Separate diagnosis of the retained baseline; candidate roster not changed by this script",
            "scope":"Two specified validation origins of one retained A Bayesian fit; not new main-study evidence",
            "original_posterior":{"qualified":True,"chains":fitted.core.posterior.n_chains,
                "retained_draws_per_chain":fitted.core.posterior.n_draws},
            "point_identity":point.identity,"point_parameters":point.metadata,
            "point_sigma2":point.parameter.sigma2,
            "point_covariance_parameters":point.parameter.covariance_parameters,
            "training_one_step":residual,"geometry":geometry,"synchronization_training_only":sync,
            "interpretation":"Zero-innovation/fixed-parameter interventions are not additive variance components or causal attribution. Only full_predictive is the unchanged qualified posterior predictive distribution.",
            "io_limitation":IO_LIMITATION,"resource_initial":initial,
            "memory_cap_bytes":MEMORY_CAP,"disk_reserve_bytes":DISK_RESERVE,"cases":[]}
        atomic_json(state_path,report)
    point=conditional_mean_parameters(fitted)
    if point.identity!=report["point_identity"]:raise ValueError("Conditional readout point changed")
    for first in ORIGINS:
        if any(row["first_target"]==first for row in report["cases"]):continue
        guard(args.output)
        window=adapter.model_window(0,first,"validation",history=HISTORY,horizon=HORIZON)
        stream=seed_for(FORECAST_SEED,"core","A",0,"bayes_stesn",MODEL_SEED,first)
        generated={};mode_failures=[]
        for mode in MODES:
            guard(args.output);begin=time.perf_counter()
            try:
                count=1 if mode=="deterministic_conditional_mean" else PATHS
                kwargs={"point":point} if mode in {"fixed_parameter_innovations","deterministic_conditional_mean"} else {}
                samples=None if count==1 else np.empty((PATHS,HORIZON,len(window.target_site_ids)))
                draw_ids=[];mean=None;metadata=None
                for batch in iter_diagnostic_forecasts(fitted,window.panel,HORIZON,
                    window.target_coordinates,window.target_site_ids,mode=mode,n_paths=count,
                    seed=stream,batch_size=BATCH,**kwargs):
                    metadata=dict(batch.metadata)
                    if samples is None:mean=train.transform.inverse(batch.mean)
                    else:
                        a,b=metadata["path_start"],metadata["path_stop"]
                        samples[a:b]=train.transform.inverse(batch.samples)
                        draw_ids.extend(metadata["parameter_draw_ids"])
                    guard(args.output)
                if samples is not None:mean=samples.mean(axis=0,dtype=np.float64)
                generated[mode]={"mean":mean,"samples":samples,"forecast_seconds":time.perf_counter()-begin,
                    "metadata":metadata,"draw_ids":draw_ids}
                print(json.dumps({"event":"diagnostic_forecast","first_target":first,"mode":mode,
                    "seconds":generated[mode]["forecast_seconds"],"peak_rss_bytes":snapshot(args.output)["process_peak_rss_bytes"]}),flush=True)
            except (FloatingPointError,np.linalg.LinAlgError) as error:
                mode_failures.append({"mode":mode,"error":repr(error)})
        # This separate helper deliberately permits INPUT-site observations
        # within the validation interval; these never enter autonomous calls.
        future_inputs=adapter.model_window(0,first+HORIZON,"validation",history=HORIZON,horizon=HORIZON).panel
        begin=time.perf_counter()
        forced=teacher_forced_readout(fitted,window.panel,future_inputs,window.target_coordinates,
                                      window.target_site_ids,point=point)
        generated["teacher_forced_one_step"]={"mean":train.transform.inverse(forced.mean),
            "samples":None,"forecast_seconds":time.perf_counter()-begin,"metadata":dict(forced.metadata),"draw_ids":[]}
        # Evaluation labels are requested only after all these predictions.
        truth=adapter.evaluation_window(0,first,"validation",history=HISTORY,horizon=HORIZON).values
        case={"first_target":first,"forecast_stream":stream,"target_site_ids":window.target_site_ids.tolist(),
            "state":"complete" if not mode_failures else "complete_with_diagnostic_failures",
            "failures":mode_failures,"modes":{},"artifacts":[]}
        visual={"truth":truth,"x":window.target_x,"input_x":window.x,"site_ids":window.target_site_ids,
                "times":window.target_time,"quantile_levels":LEVELS}
        curves={}
        for mode,result in generated.items():
            m,s=result["mean"],result["samples"]
            metrics=score(m,s,truth,train.transform.scale,window.target_coordinates)
            case["modes"][mode]={"scores":compact_scores(metrics,s,mode),
                "qualified_full_predictive_ranking":mode=="full_predictive",
                "regime":"autonomous" if mode!="teacher_forced_one_step" else "teacher_forced_one_step_NOT_autonomous",
                "diagnostic_only":mode!="full_predictive","forecast_seconds":result["forecast_seconds"],
                "mean_interpretation":result["metadata"].get("mean_policy",result["metadata"].get("regime")),
                "draw_id_sha256":content_hash(np.asarray(result["draw_ids"],dtype=np.int64)),
                "point_identity":result["metadata"].get("point_identity")}
            curves[mode]=metrics
            visual["mean__"+mode]=m
            if s is not None:
                visual["quantiles__"+mode]=np.quantile(s,LEVELS,axis=0)
                visual["first16paths__"+mode]=s[:16]
        if "full_predictive" in generated and "parameter_only" in generated:
            case["posterior_draw_ids_match_exactly"]=generated["full_predictive"]["draw_ids"]==generated["parameter_only"]["draw_ids"]
            if not case["posterior_draw_ids_match_exactly"]:raise ValueError("Matched posterior streams disagree")
        arrays=args.output/f"validation_A_bayes03_t00_a{first}.npz"
        curve_path=args.output/f"validation_A_bayes03_t00_a{first}_curves.json"
        atomic_npz(arrays,**visual);atomic_json(curve_path,{"diagnostic_identity":identity,"first_target":first,"metrics":curves})
        for path in (arrays,curve_path):
            case["artifacts"].append({"name":path.name,"bytes":path.stat().st_size,"sha256":sha(path)})
        report["cases"].append(case);report["last_updated_utc"]=now()
        atomic_json(state_path,report);atomic_json(args.receipt,report)
        del generated,visual,curves
    report.update(status="complete",completed_utc=now(),total_wall_seconds=time.perf_counter()-started,
        resource_final=guard(args.output),completed_validation_origins=len(report["cases"]))
    atomic_json(state_path,report);atomic_json(args.receipt,report)
    print(json.dumps({"event":"diagnosis_complete","identity":identity,"receipt":str(args.receipt),
        "peak_rss_bytes":report["resource_final"]["process_peak_rss_bytes"],
        "total_wall_seconds":report["total_wall_seconds"]}),flush=True)


if __name__=="__main__":
    main()

