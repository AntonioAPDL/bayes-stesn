"""Reproducible chronological benchmarking with sealed final targets.

The evaluator owns Simulation. Every fit receives a fresh observation-only
Panel restricted to its permitted rows/sites. A manifest binds resume to the
resolved protocol, source files and numerical environment. Failed fits remain
in the operational point-score population through a declared fallback.
"""
from __future__ import annotations
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
import sys
import tempfile
import time
import traceback
import numpy as np

from ..io import content_hash, environment, seed_for
from .contracts import select_panel, Forecast
from .generators import generate, Simulation
from .metrics import score_forecast, aggregate_results
from .protocol import BenchmarkProtocol, candidate_configs, NATIVE_NEW_SITE
from .registry import MODELS, fit_model, save_model


_THREAD_ENV=("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS")
_MATCHED_CONTROL={"g5_noise_stress":"g4_existing_fronts",
                  "g6_velocity_shift":"g2_spectral_transport"}


@contextmanager
def single_thread_runtime():
    """Explicit one-thread numeric runtime; preserve caller state afterward.

    This process-global setting is intentionally used only by serial callers
    or isolated spawned workers, never concurrent fitting threads.
    """
    from threadpoolctl import threadpool_limits
    previous={key:os.environ.get(key) for key in _THREAD_ENV}
    torch=sys.modules.get("torch")
    old_torch_threads=None if torch is None else torch.get_num_threads()
    for key in _THREAD_ENV:
        os.environ[key]="1"
    if torch is not None:
        torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            yield
    finally:
        if torch is not None:
            torch.set_num_threads(old_torch_threads)
        for key,value in previous.items():
            if value is None:
                os.environ.pop(key,None)
            else:
                os.environ[key]=value


def dgp_seed_for(protocol, scenario, replication):
    """Pair stress arms with their controls, keeping final seeds independent."""
    namespace="final_dgp" if protocol.stage=="confirmation" else "development_dgp"
    return seed_for(protocol.master_seed,namespace,_MATCHED_CONTROL.get(scenario,scenario),replication)


def _worker_limit():
    count=os.cpu_count() or 1
    if hasattr(os,"sched_getaffinity"):
        count=min(count,len(os.sched_getaffinity(0)))
    try:
        quota,period=Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota!="max":
            count=min(count,max(1,int(quota)//int(period)))
    except (OSError,ValueError):
        pass
    return max(1,min(count,4))


def memory_estimate(protocol, workers=1):
    """Conservative planning estimate, not a measured peak or a guarantee.

    Include copied truth, native paths and score temporaries, optional Torch
    interpreter costs, and the existing dense Gaussian engine's memory gate.
    Individual comparator allocation guards still enforce their own limits.
    """
    sites=protocol.n_side**2
    target=max(protocol.n_observed,sites-protocol.n_observed-protocol.n_calibration)
    truth=8*protocol.n_times*sites*4
    paths=8*max(protocol.paths,protocol.ensemble_members)*protocol.horizon*target
    baseline=512*1024**2 if any(MODELS[m].family=="neural" for m in protocol.models) else 192*1024**2
    gaussian=256*1024**2 if "bayes_esn_ridge" in protocol.models else 0
    per_job=baseline+truth+5*paths+gaussian
    return {"estimated_per_worker_bytes":int(per_job),
            "estimated_total_bytes":int(workers*per_job+truth),
            "native_path_bytes":int(paths),"dataset_bytes":int(truth),
            "assumption":"planning allowance; peak depends on model ranks, linear algebra, and allocator behavior"}


def _memory_budget(max_memory_mb):
    if max_memory_mb is not None:
        if not np.isfinite(max_memory_mb) or max_memory_mb<=0:
            raise ValueError("max_memory_mb must be positive and finite")
        return int(max_memory_mb*1024**2)
    limits=[]
    try:
        limits.append(int(os.sysconf("SC_PAGE_SIZE")*os.sysconf("SC_PHYS_PAGES")))
    except (OSError,ValueError):
        pass
    try:
        limit=Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit!="max":
            limits.append(int(limit))
    except (OSError,ValueError):
        pass
    return int(.8*min(limits)) if limits else None


def jsonable(value):
    if isinstance(value,np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value,np.generic):
        return jsonable(value.item())
    if isinstance(value,dict):
        return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    """Atomic small receipt writes; failed replacement leaves the old receipt."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",dir=path.parent,
                                          prefix="."+path.name+".",suffix=".tmp",delete=False) as stream:
            temporary=Path(stream.name)
            json.dump(jsonable(value),stream,indent=2,sort_keys=True,allow_nan=False)
            stream.write("\n");stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def source_fingerprint():
    root=Path(__file__).resolve().parents[1]
    files=sorted(root.rglob("*.py"))
    digest=hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def benchmark_environment():
    env=environment()
    from threadpoolctl import threadpool_info
    env["numeric_thread_policy"]="one thread per process"
    env["threadpools"]=[{key:item.get(key) for key in ("user_api","internal_api","num_threads","prefix","version","architecture")}
                        for item in threadpool_info() if item.get("user_api")=="blas"]
    try:
        import importlib.metadata
        env["torch"]=importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        env["torch"]=None
    return env


def _peak_rss():
    try:
        import resource, sys
        value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform=="darwin" else value*1024)
    except ImportError:
        return None


def spatial_validation_split(panel):
    """Geometry-only contiguous pseudo-new-site block, entirely inside O."""
    n=panel.y.shape[1]
    if n<5:
        raise ValueError("Spatial validation needs at least five observed sites")
    # Lexicographic x/y ordering is independent of every response. Hold out the
    # rightmost fifth to challenge extrapolation; all transforms are refitted.
    order=np.lexsort(tuple(panel.coordinates[:,i] for i in reversed(range(panel.coordinates.shape[1]))))
    held=order[-max(2,int(np.ceil(.2*n))):]
    kept=np.setdiff1d(np.arange(n),held)
    return kept,held


def _fit_and_validate(model_id, config, sim, protocol, track, fit_seed, score_seed):
    all_visible=sim.observed_panel(protocol.final_origin)
    if track=="B":
        inputs,targets=spatial_validation_split(all_visible)
    else:
        inputs=targets=np.arange(all_visible.y.shape[1])
    training=select_panel(all_visible,stop=protocol.initial_fit,sites=inputs,start=protocol.training_start)
    started=time.perf_counter()
    fitted=fit_model(model_id,training,config,fit_seed)
    fit_seconds=time.perf_counter()-started
    coords=all_visible.coordinates[targets]
    ids=tuple(all_visible.site_ids[j] for j in targets)
    metric=MODELS[model_id].selection_metric
    losses=[]
    for origin in protocol.origins:
        history=select_panel(all_visible,stop=origin,sites=inputs,start=protocol.training_start)
        prediction=fitted.forecast(protocol.horizon,target_coordinates=coords,target_site_ids=ids,
                    n_paths=protocol.paths,seed=seed_for(score_seed,"forecast",origin),history=history)
        # Evaluator exposes validation truth only after forecast construction.
        truth=all_visible.y[origin:origin+protocol.horizon,targets]
        if metric=="rmse":
            loss=float(np.sqrt(np.mean((prediction.mean-truth)**2)))
        else:
            from ..scoring import crps_ensemble
            if prediction.samples is None:
                raise ValueError("Registered probabilistic selector needs native samples")
            loss=float(crps_ensemble(prediction.samples,truth).mean())
        if not np.isfinite(loss):
            raise FloatingPointError("Nonfinite validation score")
        losses.append(loss)
    return {"score":float(np.mean(losses)),"origin_scores":losses,
            "fit_seconds":fit_seconds,"seconds":time.perf_counter()-started,
            "diagnostics":jsonable(fitted.diagnostics),
            "qualified":bool(fitted.diagnostics.get("qualified",True)),
            "metric":metric,"fit_seed":int(fit_seed),
            "spatial_input_ids":training.site_ids,"validation_target_ids":ids}


def select_configuration(model_id, sim, protocol, track, seed, *, directory=None, progress=None):
    """Tune with frozen parameters between origins; record every candidate cost.

    Screening diagnostics are retained. Qualification of the final refit is a
    separate mandatory gate. Screening scores never become scientific test
    results and no candidate is selected using U or any final-test response.
    """
    configs=candidate_configs(model_id,protocol,track=track,seed=seed)
    records=[]
    def evaluate(index,fit_seed,label):
        path=None if directory is None else Path(directory)/f"{label}.json"
        if path is not None and path.exists():
            return json.loads(path.read_text())
        started=time.perf_counter()
        try:
            result=_fit_and_validate(model_id,configs[index],sim,protocol,track,fit_seed,seed_for(seed,label))
            result.update(status="completed",candidate=index,config=configs[index])
        except Exception as error:
            result={"status":"failed","candidate":index,"config":configs[index],
                    "score":None,"seconds":time.perf_counter()-started,
                    "error":f"{type(error).__name__}: {error}","traceback":traceback.format_exc()}
        if path is not None:
            write_json(path,result)
        if progress:
            progress({"event":"candidate","model":model_id,"track":track,"label":label,
                      "status":result["status"],"score":result["score"]})
        return result
    for index in range(len(configs)):
        records.append(evaluate(index,seed_for(seed,"candidate_fit",index),f"trial_{index:02d}"))
    valid=sorted((r for r in records if r["status"]=="completed"),key=lambda r:(r["score"],r["candidate"]))
    if not valid:
        return {"selected":None,"trials":records,"rechecks":[],"status":"all_candidates_failed",
                "compute_seconds":sum(r["seconds"] for r in records)}
    rechecks=[]
    best=valid[0]
    if MODELS[model_id].stochastic_fit and protocol.finalist_seeds>1:
        finalists=valid[:protocol.finalist_count]
        averaged=[]
        for candidate in finalists:
            index=candidate["candidate"]
            values=[]
            for j in range(protocol.finalist_seeds):
                value=evaluate(index,seed_for(seed,"finalist_fit",index,j),f"recheck_{index:02d}_{j:02d}")
                rechecks.append(value)
                if value["status"]=="completed":
                    values.append(value["score"])
            # Recheck failures are not discarded to reward fragile candidates.
            if len(values)==protocol.finalist_seeds:
                averaged.append((float(np.mean(values)),index,candidate))
        if not averaged:
            return {"selected":None,"trials":records,"rechecks":rechecks,
                    "status":"all_finalists_failed","compute_seconds":sum(r["seconds"] for r in records+rechecks)}
        best=min(averaged,key=lambda r:(r[0],r[1]))[2]
    return {"selected":best["config"],"selected_candidate":best["candidate"],
            "metric":MODELS[model_id].selection_metric,"trials":records,"rechecks":rechecks,
            "status":"selected","compute_seconds":sum(r["seconds"] for r in records+rechecks),
            "information_policy":"initial training only; validation responses score forecasts; U and final time sealed",
            "qualification_policy":"screening diagnostics recorded; final refit must pass independently"}


def _final_job(model_id,sim,protocol,track,seed,directory,selection):
    targets=sim.observed_indices[:protocol.score_observed_count] if track=="A" else sim.test_indices
    training=select_panel(sim.observed_panel(protocol.final_origin),start=protocol.training_start)
    coords=sim.coordinates[targets]; ids=tuple(sim.site_ids[j] for j in targets)
    started=time.perf_counter();cpu_start=time.process_time()
    info={"completed":False,"qualified":False,"fallback":False,
          "selection_metric":MODELS[model_id].selection_metric,
          "fit_seconds":0.,"forecast_seconds":0.,"tuning_seconds":selection["compute_seconds"],
          "training_scale":max(float(np.std(sim.observation[protocol.training_start:protocol.initial_fit,
                                    sim.observed_indices[:protocol.score_observed_count]])),1e-12),
          "final_fit_seed":int(seed_for(seed,"final_fit")),"forecast_seed":int(seed_for(seed,"final_forecast")),
          "native_scores":{},"diagnostics":{},"origin_index":protocol.final_origin,
          "target_ids":ids,"fit_config":selection["selected"]}
    forecast=None
    if selection["selected"] is not None:
        fitting_start=time.perf_counter()
        try:
            fitted=fit_model(model_id,training,selection["selected"],seed_for(seed,"final_fit"),
                             checkpoint_path=directory/"chain.npz" if model_id=="bayes_esn_ridge" else None)
            info["fit_seconds"]=time.perf_counter()-fitting_start
            info["diagnostics"]=jsonable(fitted.diagnostics)
            info["qualified"]=bool(fitted.diagnostics.get("qualified",True))
            save_model(fitted,directory/"model.npz")
            forecast_start=time.perf_counter()
            forecast=fitted.forecast(protocol.horizon,target_coordinates=coords,target_site_ids=ids,
                                     n_paths=protocol.paths,seed=seed_for(seed,"final_forecast"))
            info["forecast_seconds"]=time.perf_counter()-forecast_start
            info["completed"]=True
        except Exception as error:
            if not info["fit_seconds"]:
                info["fit_seconds"]=time.perf_counter()-fitting_start
            info.update(qualified=False,error=f"{type(error).__name__}: {error}",traceback=traceback.format_exc())
    else:
        info["error"]="No candidate survived the predeclared validation procedure"
    # Final truth is read only after hyperparameter selection, final fitting,
    # and any forecast generation. No base-model routine receives this array.
    truth=sim.observation[protocol.final_origin:protocol.n_times,targets]
    score_seed=seed_for(protocol.master_seed,"score_pairs",track,sim.metadata["scenario"])
    if forecast is not None:
        try:
            info["native_scores"]=score_forecast(forecast,truth,pair_count=protocol.score_pairs,seed=score_seed)
            info["forecast_metadata"]=dict(forecast.metadata)
        except Exception as error:
            info.update(qualified=False,scoring_error=f"{type(error).__name__}: {error}",
                        scoring_traceback=traceback.format_exc())
    if not info["qualified"] or forecast is None:
        info["fallback"]=True
        fallback_id="persistence" if track=="A" else "spatial_persistence"
        fallback_start=time.perf_counter()
        fallback=fit_model(fallback_id,training,{},seed_for(seed,"fallback"))
        operational=fallback.forecast(protocol.horizon,target_coordinates=coords,target_site_ids=ids,n_paths=1,
                                      seed=seed_for(seed,"fallback_forecast"))
        info["fallback_seconds"]=time.perf_counter()-fallback_start
        info["fallback_model"]=fallback_id
    else:
        operational=forecast
    info["operational_scores"]=dict(info["native_scores"]) if not info["fallback"] else score_forecast(operational,truth)
    info.update(total_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu_start,
                process_peak_rss_bytes=_peak_rss(),
                memory_measurement="process lifetime high-water mark; not an isolated per-model increment")
    return info,forecast,operational


def _finish_job(job, protocol, info, forecast, operational):
    directory=Path(job["directory"])
    info.update(model_id=job["model_id"],scenario=job["scenario"],track=job["track"],
                replication=job["replication"],dgp_seed=job["dgp_seed"],
                dgp_seed_namespace=_MATCHED_CONTROL.get(job["scenario"],job["scenario"]),
                manifest_identity=job["manifest_identity"],stage=protocol.stage)
    if job["replication"] in protocol.retain_paths_replications:
        if forecast is not None:
            forecast.save(directory/"forecast.npz")
        Forecast(operational.mean,None,operational.coordinates,operational.site_ids,
                 {"fallback":info["fallback"]}).save(directory/"operational_point.npz")
    write_json(directory/"receipt.json",info)
    return jsonable(info)


def _execute_job(job, progress=None):
    """Top-level spawn entry point; exclusively owns this job's directory."""
    with single_thread_runtime():
        if source_fingerprint()!=job["source_fingerprint"]:
            raise RuntimeError("Source changed after the run manifest was sealed")
        protocol=BenchmarkProtocol(**job["protocol"])
        sim=Simulation.load(job["data_path"])
        directory=Path(job["directory"])
        selection_path=directory/"selection.json"
        if selection_path.exists():
            selection=json.loads(selection_path.read_text())
        else:
            selection=select_configuration(job["model_id"],sim,protocol,job["track"],job["seed"],
                          directory=directory/"tuning",progress=progress)
            write_json(selection_path,selection)
        info,forecast,operational=_final_job(job["model_id"],sim,protocol,job["track"],
                                           job["seed"],directory,selection)
        return _finish_job(job,protocol,info,forecast,operational)


def _process_failure(job, error):
    """Keep infrastructure failures in the declared operational population."""
    protocol=BenchmarkProtocol(**job["protocol"])
    sim=Simulation.load(job["data_path"])
    selection={"selected":None,"compute_seconds":0.}
    cached=Path(job["directory"])/"selection.json"
    if cached.exists():
        selection["compute_seconds"]=json.loads(cached.read_text()).get("compute_seconds",0.)
    info,_,operational=_final_job(job["model_id"],sim,protocol,job["track"],job["seed"],
                                Path(job["directory"]),selection)
    info.update(error=f"Worker execution failed: {type(error).__name__}: {error}",
                process_failure=True,process_failure_scope="execution infrastructure; may include queued jobs after a worker failure",
                incomplete_compute_accounting=True)
    return _finish_job(job,protocol,info,None,operational)


def _run_benchmark(protocol, output, resume, progress, workers, power_freeze, max_memory_mb):
    limit=_worker_limit()
    if not isinstance(workers,(int,np.integer)) or isinstance(workers,bool) or not 1<=workers<=limit:
        raise ValueError(f"workers must be an integer between 1 and the host cap {limit}")
    frozen=None
    if protocol.stage=="confirmation":
        from .studies import verify_confirmation_freeze
        frozen=verify_confirmation_freeze(protocol,power_freeze)
    estimate=memory_estimate(protocol,workers)
    budget=_memory_budget(max_memory_mb)
    if budget is not None and estimate["estimated_total_bytes"]>budget:
        raise MemoryError("Planned worker memory exceeds the declared host budget; reduce workers/ranks/paths or set an explicit larger budget")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest={"protocol":protocol.to_dict(),"protocol_identity":protocol.identity,
              "source_fingerprint":source_fingerprint(),"environment":benchmark_environment(),
              "stage":protocol.stage,"schema":"benchmark-run-3.0",
              "confirmation_freeze":None if frozen is None else frozen["identity"],
              "execution":{"workers":int(workers),"worker_cap":limit,"threads_per_worker":1,
                           "process_start_method":"spawn" if workers>1 else "serial",
                           "memory_budget_bytes":budget,"memory_plan":estimate}}
    manifest=jsonable(manifest)
    manifest["identity"]=content_hash(manifest)
    manifest_path=output/"manifest.json"
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if not resume or previous.get("identity")!=manifest["identity"]:
            raise ValueError("Output exists with a different protocol/source/runtime/execution plan; use a new directory")
    else:
        write_json(manifest_path,manifest)
        protocol.save(output/"resolved_protocol.yaml")
        if frozen is not None:
            write_json(output/"confirmation_freeze.json",frozen)
    receipts=[];jobs=[]
    namespace="final_dgp" if protocol.stage=="confirmation" else "development_dgp"
    for scenario in protocol.scenarios:
        for replication in range(protocol.replications):
            dataset_dir=output/scenario/f"rep_{replication:03d}"
            dataset_dir.mkdir(parents=True,exist_ok=True)
            data_path=dataset_dir/"simulation.npz"
            dgp_seed=dgp_seed_for(protocol,scenario,replication)
            if data_path.exists():
                sim=Simulation.load(data_path)
                if sim.metadata["seed"]!=dgp_seed or len(sim.observation)!=protocol.n_times:
                    raise ValueError("Cached dataset does not match its manifest seed/dimensions")
            else:
                options=dict((protocol.generator_options or {}).get(scenario,{}))
                if scenario=="g6_velocity_shift":
                    options.setdefault("origin_index",protocol.final_origin)
                    options.setdefault("break_lead",min(25,protocol.horizon))
                sim=generate(scenario,protocol.n_times,n_side=protocol.n_side,n_observed=protocol.n_observed,
                             n_calibration=protocol.n_calibration,seed=dgp_seed,**options)
                sim.save(data_path)
            del sim
            if progress:
                progress({"event":"dataset","scenario":scenario,"replication":replication})
            for track in protocol.tracks:
                models=[m for m in protocol.models if track in MODELS[m].tracks]
                if track=="B" and "spatial_persistence" not in models:
                    models.insert(0,"spatial_persistence")
                for model_id in models:
                    directory=dataset_dir/track/model_id;directory.mkdir(parents=True,exist_ok=True)
                    receipt_path=directory/"receipt.json"
                    if receipt_path.exists():
                        saved=json.loads(receipt_path.read_text())
                        if saved.get("manifest_identity")!=manifest["identity"]:
                            raise ValueError("Cached job receipt belongs to a different run manifest")
                        receipts.append(saved);continue
                    jobs.append({"protocol":protocol.to_dict(),"data_path":str(data_path),
                         "directory":str(directory),"model_id":model_id,"scenario":scenario,
                         "track":track,"replication":replication,"dgp_seed":int(dgp_seed),
                         "seed":seed_for(protocol.master_seed,namespace,scenario,replication,track,model_id),
                         "manifest_identity":manifest["identity"],"source_fingerprint":manifest["source_fingerprint"]})
    def accept(info):
        receipts.append(info)
        receipts.sort(key=lambda r:(r["scenario"],r["track"],r["model_id"],r["replication"]))
        write_json(output/"summary.json",aggregate_results(receipts,seed=protocol.master_seed))
        if progress:
            progress({"event":"complete","scenario":info["scenario"],"replication":info["replication"],
                      "model":info["model_id"],"track":info["track"],"qualified":info["qualified"],
                      "fallback":info["fallback"],"rmse":info["operational_scores"]["rmse"],
                      "seconds":info["total_seconds"]})
    if workers==1:
        for job in jobs:
            try:
                info=_execute_job(job,progress)
            except Exception as error:
                info=_process_failure(job,error)
            accept(info)
    elif jobs:
        # Spawn imports fresh numerical runtimes under inherited one-thread env.
        # Models and random states never cross fitting threads.
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            pending={}
            for job in jobs:
                try:
                    pending[pool.submit(_execute_job,job)]=job
                except Exception as error:
                    accept(_process_failure(job,error))
            for future in as_completed(pending):
                job=pending[future]
                try:
                    info=future.result()
                except Exception as error:
                    info=_process_failure(job,error)
                accept(info)
    receipts.sort(key=lambda r:(r["scenario"],r["track"],r["model_id"],r["replication"]))
    summary=aggregate_results(receipts,seed=protocol.master_seed)
    final_source=source_fingerprint()
    stable=final_source==manifest["source_fingerprint"] and protocol.identity==manifest["protocol_identity"]
    summary.update(manifest_identity=manifest["identity"],stage=protocol.stage,
                   jobs=len(receipts),profile=protocol.name,source_stable=stable,
                   final_source_fingerprint=final_source,
                   comparison_status="execution only" if protocol.stage=="execution_test" else
                       "development evidence; no confirmatory superiority claim" if protocol.stage=="development" else
                       "confirmation profile; inspect qualifications, replication intervals and power freeze")
    if frozen is not None:
        summary.update(power_status=frozen["power_status"],power_cap_binding=frozen["power"]["cap_binding"])
    if not stable:
        summary["comparison_status"]="invalidated: source or protocol changed while the run was active"
    write_json(output/"summary.json",summary)
    if not stable:
        raise RuntimeError("Benchmark source/protocol changed during execution; results are retained but invalidated. Start a new run after edits finish.")
    return summary


def run_benchmark(protocol: BenchmarkProtocol, output, *, resume=True, progress=None,
                  workers=1, power_freeze=None, max_memory_mb=None):
    """Run serially or with at most four isolated spawned worker processes.

    Confirmation requires the verified receipt produced by
    ``studies.freeze_confirmation`` in both Python and CLI entry points.
    Smoke/development callers retain their existing API. Each worker owns
    one job directory; only the parent writes aggregate summaries. Seeds and
    sorted replication summaries are independent of scheduling order.
    """
    if not isinstance(protocol,BenchmarkProtocol):
        raise TypeError("run_benchmark requires a validated BenchmarkProtocol")
    with single_thread_runtime():
        return _run_benchmark(protocol,output,resume,progress,workers,power_freeze,max_memory_mb)
