"""Reproducible chronological benchmarking with sealed final targets.

The evaluator owns Simulation. Every fit receives a fresh observation-only
Panel restricted to its permitted rows/sites. A manifest binds resume to the
resolved protocol, source files and numerical environment. Failed fits remain
in the operational point-score population through a declared fallback.
"""
from __future__ import annotations
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from concurrent.futures.process import BrokenProcessPool
from collections import deque
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
from .protocol import BenchmarkProtocol, candidate_configs, NATIVE_NEW_SITE, effective_models
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
    from .execution import host_resources
    return host_resources()["worker_cap"]


def memory_estimate(protocol, workers=1):
    from .execution import estimate_memory
    return estimate_memory(protocol, workers)


def _memory_budget(max_memory_mb):
    from .execution import memory_allowance
    return memory_allowance(max_memory_mb)["effective_job_budget_bytes"]


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
    records=[{key:item.get(key) for key in ("user_api","internal_api","num_threads","prefix","version","architecture")}
             for item in threadpool_info() if item.get("user_api")=="blas"]
    # Dynamic-library enumeration follows import order, which can differ in a
    # fresh spawned worker. The complete inventory is a multiset: retain every
    # record (including duplicates) and every numerical identity field, while
    # removing incidental enumeration order from the computation identity.
    env["threadpools"]=sorted(records,key=lambda record:json.dumps(record,sort_keys=True,separators=(",",":")))
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


class FitTimeLimitExceeded(TimeoutError):
    """A fit exhausted its declared scientific wall-time allowance."""


@contextmanager
def _fit_time_limit(seconds):
    """A declared POSIX fit ceiling; native calls may return before delivery.

    This is a scientific stopping rule, not an infrastructure retry. The
    previous process timer is restored, including elapsed time, on exit.
    """
    if seconds is None:
        yield
        return
    import signal
    import threading
    if not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Declared fit-time limits require a POSIX main-thread worker")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0] > 0:
        raise RuntimeError("Cannot replace an existing process alarm with a scientific fit limit")
    def expired(signum, frame):
        raise FitTimeLimitExceeded(f"Declared fit allowance of {seconds:g} seconds exhausted")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous_handler)


def _fit_with_policy(model_id, panel, config, seed, protocol, **kwargs):
    with _fit_time_limit((protocol.fit_time_limits_seconds or {}).get(model_id)):
        return fit_model(model_id, panel, config, seed, **kwargs)


def _propagate_nonmethod_failure(error):
    """Only a model's declared numerical failure may become a fallback."""
    if isinstance(error, (StudyInvalidatedError, MemoryError)):
        raise error
    if isinstance(error, OSError) and not isinstance(error, FitTimeLimitExceeded):
        raise error
    # Kept lazy so core point-control workflows never import inference code.
    from . import esn
    integrity_type = getattr(esn, "CheckpointIntegrityError", ())
    if integrity_type and isinstance(error, integrity_type):
        raise StudyInvalidatedError(str(error)) from error


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


def spatial_validation_splits(panel, count=1):
    """Fixed geometry-only edge/interior blocks; no responses are inspected."""
    if not isinstance(count, (int, np.integer)) or isinstance(count, bool) or not 1 <= count <= 3:
        raise ValueError("Spatial validation uses one, two, or three fixed blocks")
    kept, held = spatial_validation_split(panel)
    result = [("edge_right", kept, held)]
    n = panel.y.shape[1]
    size = len(held)
    indices = np.arange(n)
    if count >= 2:
        center = np.median(panel.coordinates, axis=0)
        distance = np.sum((panel.coordinates-center)**2, axis=1)
        interior = np.lexsort((indices, distance))[:size]
        result.append(("interior", np.setdiff1d(indices, interior), interior))
    if count >= 3:
        order = np.lexsort(tuple(panel.coordinates[:, i] for i in reversed(range(panel.coordinates.shape[1]))))
        left = order[:size]
        result.append(("edge_left", np.setdiff1d(indices, left), left))
    return result


def _fit_and_validate(model_id, config, sim, protocol, track, fit_seed, score_seed,
                      *, checkpoint_directory=None):
    all_visible = sim.observed_panel(protocol.final_origin)
    if track == "B":
        folds = spatial_validation_splits(all_visible, getattr(protocol, "spatial_validation_blocks", 1))
    else:
        sites = np.arange(all_visible.y.shape[1])
        folds = [("observed", sites, sites)]
    started = time.perf_counter()
    cpu_started = time.process_time()
    metric = MODELS[model_id].selection_metric
    records = []
    legacy = protocol.schema_version == "3.0"
    for label, inputs, targets in folds:
        training = select_panel(all_visible, stop=protocol.initial_fit, sites=inputs,
                                start=protocol.training_start)
        fold_seed = fit_seed if legacy or track == "A" else seed_for(fit_seed, "spatial_fold", label)
        fit_options = {}
        if checkpoint_directory is not None and MODELS[model_id].family == "bayesian":
            fit_options["checkpoint_path"] = Path(checkpoint_directory)/label/"chain.npz"
        fit_started = time.perf_counter()
        fitted = _fit_with_policy(model_id, training, config, fold_seed, protocol, **fit_options)
        fit_seconds = time.perf_counter()-fit_started
        coords = all_visible.coordinates[targets]
        ids = tuple(all_visible.site_ids[j] for j in targets)
        losses = []
        forecast_seconds = scoring_seconds = 0.
        for origin in protocol.origins:
            history = select_panel(all_visible, stop=origin, sites=inputs, start=protocol.training_start)
            forecast_seed = seed_for(score_seed, "forecast", origin) if legacy else seed_for(
                score_seed, "forecast", label, origin)
            forecast_started = time.perf_counter()
            prediction = fitted.forecast(protocol.horizon, target_coordinates=coords,
                target_site_ids=ids, n_paths=protocol.paths, seed=forecast_seed, history=history)
            forecast_seconds += time.perf_counter()-forecast_started
            scoring_started = time.perf_counter()
            truth = all_visible.y[origin:origin+protocol.horizon, targets]
            if metric == "rmse":
                loss = float(np.sqrt(np.mean((prediction.mean-truth)**2)))
            else:
                from ..scoring import crps_ensemble
                if prediction.samples is None:
                    raise ValueError("Registered probabilistic selector needs native samples")
                loss = float(crps_ensemble(prediction.samples, truth).mean())
            if not np.isfinite(loss):
                raise FloatingPointError("Nonfinite validation score")
            losses.append(loss)
            scoring_seconds += time.perf_counter()-scoring_started
        records.append({"fold": label, "score": float(np.mean(losses)), "origin_scores": losses,
                        "fit_seconds": fit_seconds, "forecast_seconds": forecast_seconds,
                        "scoring_seconds": scoring_seconds,
                        "fit_seed": int(fold_seed), "diagnostics": jsonable(fitted.diagnostics),
                        "qualified": bool(fitted.diagnostics.get("qualified", True)),
                        "spatial_input_ids": training.site_ids, "validation_target_ids": ids})
    return {"score": float(np.mean([r["score"] for r in records])),
            "origin_scores": np.mean([r["origin_scores"] for r in records], axis=0).tolist(),
            "fit_seconds": sum(r["fit_seconds"] for r in records),
            "forecast_seconds": sum(r["forecast_seconds"] for r in records),
            "scoring_seconds": sum(r["scoring_seconds"] for r in records),
            "seconds": time.perf_counter()-started, "cpu_seconds": time.process_time()-cpu_started,
            "diagnostics": records[0]["diagnostics"] if len(records) == 1 else {"folds": records},
            "qualified": all(r["qualified"] for r in records), "metric": metric,
            "fit_seed": int(fit_seed), "folds": records,
            "spatial_input_ids": records[0]["spatial_input_ids"],
            "validation_target_ids": records[0]["validation_target_ids"],
            "validation_weights": "equal folds; equal chronological origins; all required fits qualify"}


def _select_legacy_configuration(model_id, sim, protocol, track, seed, *, directory=None, progress=None):
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
            _propagate_nonmethod_failure(error)
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


def select_configuration(model_id, sim, protocol, track, seed, *, directory=None, progress=None):
    """Resolved distinct candidates and diagnostic-qualified fresh finalists."""
    if getattr(protocol, "selection_policy", "legacy") == "legacy":
        return _select_legacy_configuration(model_id, sim, protocol, track, seed,
                                             directory=directory, progress=progress)
    from .protocol import candidate_manifest
    panel = sim.observed_panel(protocol.final_origin)
    frozen_mode = protocol.selection_mode == "development_frozen_configuration"
    folds = spatial_validation_splits(panel, protocol.spatial_validation_blocks) if track == "B" and not frozen_mode else [
        ("observed", np.arange(panel.y.shape[1]), np.arange(panel.y.shape[1]))]
    design = candidate_manifest(model_id, protocol, track=track, seed=seed,
        fitting_sites=min(len(kept) for _, kept, _ in folds),
        fitting_rows=(protocol.final_origin if frozen_mode else protocol.initial_fit)-protocol.training_start)
    configurations, hashes = design["candidates"], design["candidate_hashes"]
    expected_source = source_fingerprint()
    context_id = content_hash({"protocol": protocol.identity, "model": model_id, "track": track,
        "seed": int(seed), "source": expected_source, "y": panel.y,
        "coordinates": panel.coordinates, "times": panel.times, "site_ids": panel.site_ids})
    base = {"metric": MODELS[model_id].selection_metric, "candidate_design": design["metadata"],
            "candidate_hashes": hashes, "selection_context_identity": context_id,
            "selection_mode": protocol.selection_mode,
            "information_policy": "initial training only; O validation scores; U/final/latent truth sealed",
            "qualification_policy": "all planned finalist seed-by-fold fits qualify; equal fixed weights"}
    if protocol.selection_mode == "development_frozen_configuration":
        if len(configurations) != 1:
            raise ValueError("A frozen configuration must resolve to exactly one model setting")
        return {**base, "selected": configurations[0], "selected_candidate": 0,
                "selected_candidate_hash": hashes[0], "trials": [], "rechecks": [],
                "status": "selected", "compute_seconds": 0., "compute_cpu_seconds": 0.,
                "selection_policy": "development-selected configuration; independent final refit"}
    trials, rechecks = [], []

    def evaluate(index, phase, repetition=0):
        if source_fingerprint() != expected_source:
            raise StudyInvalidatedError("Source changed during model selection")
        candidate_hash = hashes[index]
        fit_seed = seed_for(seed, "candidate_fit" if phase == "screen" else "finalist_fit",
                            candidate_hash, repetition)
        label = f"{phase}_{candidate_hash[:16]}_{repetition:02d}"
        path = None if directory is None else Path(directory)/(label+".json")
        identity = content_hash({"context": context_id, "candidate_hash": candidate_hash,
                                 "phase": phase, "repetition": repetition, "fit_seed": fit_seed})
        if path is not None and path.exists():
            cached = json.loads(path.read_text())
            if (cached.get("evaluation_identity") != identity or
                cached.get("record_identity") != content_hash({k:v for k,v in cached.items() if k!="record_identity"})):
                raise StudyInvalidatedError("Cached tuning evaluation has incompatible identity or modified content")
            return cached
        started, cpu_started = time.perf_counter(), time.process_time()
        try:
            result = _fit_and_validate(model_id, configurations[index], sim, protocol, track,
                fit_seed, seed_for(seed, "validation_forecast", candidate_hash, phase, repetition),
                checkpoint_directory=None if directory is None else Path(directory)/(label+"_fit"))
            if source_fingerprint() != expected_source:
                raise StudyInvalidatedError("Source changed during a candidate fit")
            result.update(status="completed")
        except Exception as error:
            _propagate_nonmethod_failure(error)
            result = {"status": "failed", "score": None, "qualified": False,
                      "seconds": time.perf_counter()-started,
                      "cpu_seconds": time.process_time()-cpu_started,
                      "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
        result.update(candidate=index, candidate_hash=candidate_hash, config=configurations[index],
                      fit_seed=int(fit_seed), phase=phase, evaluation_identity=identity)
        result = jsonable(result)
        result["record_identity"] = content_hash(result)
        if path is not None:
            write_json(path, result)
        if progress:
            progress({"event": "candidate", "model": model_id, "track": track, "label": label,
                      "status": result["status"], "qualified": result["qualified"], "score": result["score"]})
        return result

    for i in range(len(configurations)):
        trials.append(evaluate(i, "screen"))
    ranked = sorted((r for r in trials if r["status"] == "completed" and r["score"] is not None),
                    key=lambda r: (r["score"], r["candidate_hash"]))
    eligible = []
    initial = min(protocol.finalist_count, len(ranked))
    maximum = min(len(ranked), initial+protocol.replacement_finalists)
    seeds = protocol.finalist_seeds if MODELS[model_id].stochastic_fit else 1
    for position, candidate in enumerate(ranked[:maximum]):
        if position >= initial and eligible:
            break
        index = candidate["candidate"]
        values = [evaluate(index, "finalist", repetition) for repetition in range(seeds)]
        rechecks.extend(values)
        if all(r["status"] == "completed" and r.get("qualified", False) for r in values):
            eligible.append((float(np.mean([r["score"] for r in values])), candidate["candidate_hash"], index))
    result = {**base, "trials": trials, "rechecks": rechecks,
              "compute_seconds": sum(r["seconds"] for r in trials+rechecks),
              "compute_cpu_seconds": sum(r.get("cpu_seconds", 0.) for r in trials+rechecks),
              "eligible_finalists": len(eligible),
              "replacement_limit": protocol.replacement_finalists,
              "finalist_seed_count": seeds}
    if not eligible:
        return {**result, "selected": None, "status": "all_candidates_failed" if not ranked else "all_finalists_unqualified"}
    score, selected_hash, index = min(eligible)
    return {**result, "selected": configurations[index], "selected_candidate": index,
            "selected_candidate_hash": selected_hash, "selected_validation_score": score, "status": "selected"}


def _final_job(model_id,sim,protocol,track,seed,directory,selection):
    targets=sim.observed_indices[:protocol.score_observed_count] if track=="A" else sim.test_indices
    training=select_panel(sim.observed_panel(protocol.final_origin),start=protocol.training_start)
    coords=sim.coordinates[targets]; ids=tuple(sim.site_ids[j] for j in targets)
    started=time.perf_counter();cpu_start=time.process_time()
    from .metrics import training_normalization
    training_scale, normalization_reference = training_normalization(sim, protocol)
    info={"completed":False,"qualified":False,"fallback":False,
          "selection_metric":MODELS[model_id].selection_metric,
          "fit_seconds":0.,"forecast_seconds":0.,"scoring_seconds":0.,"serialization_seconds":0.,
          "fallback_seconds":0.,"tuning_seconds":selection["compute_seconds"],
          "tuning_cpu_seconds":selection.get("compute_cpu_seconds"),
          "training_scale":training_scale,"normalization_reference":normalization_reference,
          "final_fit_seed":int(seed_for(seed,"final_fit")),"forecast_seed":int(seed_for(seed,"final_forecast")),
          "native_scores":{},"diagnostics":{},"origin_index":protocol.final_origin,
          "target_ids":ids,"fit_config":selection["selected"]}
    forecast=None
    if selection["selected"] is not None:
        fitting_start=time.perf_counter()
        try:
            fitted=_fit_with_policy(model_id,training,selection["selected"],seed_for(seed,"final_fit"),protocol,
                             checkpoint_path=directory/"chain.npz" if MODELS[model_id].family=="bayesian" else None)
            info["fit_seconds"]=time.perf_counter()-fitting_start
            info["diagnostics"]=jsonable(fitted.diagnostics)
            info["qualified"]=bool(fitted.diagnostics.get("qualified",True))
            serialization_start=time.perf_counter()
            try:
                save_model(fitted,directory/"model.npz")
            finally:
                info["serialization_seconds"]+=time.perf_counter()-serialization_start
            forecast_start=time.perf_counter()
            try:
                forecast=fitted.forecast(protocol.horizon,target_coordinates=coords,target_site_ids=ids,
                                         n_paths=protocol.paths,seed=seed_for(seed,"final_forecast"))
            finally:
                info["forecast_seconds"]=time.perf_counter()-forecast_start
            info["completed"]=True
        except Exception as error:
            _propagate_nonmethod_failure(error)
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
        scoring_start=time.perf_counter()
        try:
            info["native_scores"]=score_forecast(forecast,truth,pair_count=protocol.score_pairs,seed=score_seed)
            info["forecast_metadata"]=dict(forecast.metadata)
        except Exception as error:
            _propagate_nonmethod_failure(error)
            info.update(qualified=False,scoring_error=f"{type(error).__name__}: {error}",
                        scoring_traceback=traceback.format_exc())
        info["scoring_seconds"]+=time.perf_counter()-scoring_start
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
    scoring_start=time.perf_counter()
    info["operational_scores"]=dict(info["native_scores"]) if not info["fallback"] else score_forecast(operational,truth)
    info["scoring_seconds"]+=time.perf_counter()-scoring_start
    recovery = info["diagnostics"].get("recovery",{})
    info["diagnostic_seconds"] = sum(value for key,value in recovery.get(
        "completed_phase_seconds",{}).items() if "diagnostics" in key)
    info["fit_resumed"] = bool(recovery.get("resumed",False))
    info["diagnostic_timing_scope"] = "sum of completed ESN diagnostic phases across compatible invocations; not added again; may exceed current fit_seconds after recovery"
    info["timing_scope"] = "fit_seconds includes this invocation's diagnostics; total_seconds and cpu_seconds cover final fitting through scoring, excluding tuning and final artifact serialization"
    info.update(total_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu_start,
                process_peak_rss_bytes=_peak_rss(),
                memory_measurement="process lifetime high-water mark; not an isolated per-model increment")
    return info,forecast,operational


class StudyInvalidatedError(RuntimeError):
    """Source, data, or persisted scientific identity changed during a run."""


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for piece in iter(lambda: stream.read(1024*1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def _record_state(job, status, **fields):
    path = Path(job["directory"])/"state.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    if previous and previous.get("job_identity") != job["job_identity"]:
        raise StudyInvalidatedError("Cached job state has an incompatible scientific identity")
    state = {**previous, "schema": "benchmark-job-state-1.0",
             "job_identity": job["job_identity"], "manifest_identity": job["manifest_identity"],
             "status": status, **fields}
    write_json(path, state)
    return state


def _load_terminal(job):
    path = Path(job["directory"])/"receipt.json"
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    if saved.get("manifest_identity") != job["manifest_identity"]:
        raise StudyInvalidatedError("Cached job receipt belongs to another run manifest")
    if saved.get("process_failure") or not saved.get("terminal_scientific_result", False):
        raise StudyInvalidatedError("Nonterminal or legacy infrastructure receipt cannot be silently resumed")
    if saved.get("job_identity") != job["job_identity"]:
        raise StudyInvalidatedError("Cached scientific job identity differs")
    if saved.get("receipt_identity") != content_hash({k:v for k,v in saved.items() if k!="receipt_identity"}):
        raise StudyInvalidatedError("Cached scientific receipt content changed")
    for artifact in saved.get("artifacts", []):
        name = Path(artifact["file"])
        if name.is_absolute() or ".." in name.parts:
            raise StudyInvalidatedError("Unsafe scientific artifact path")
        member = Path(job["directory"])/name
        if not member.is_file() or member.stat().st_size != artifact["bytes"] or _file_hash(member) != artifact["sha256"]:
            raise StudyInvalidatedError("Cached scientific artifact is missing or changed")
    return saved


def _finish_job(job, protocol, info, forecast, operational):
    directory = Path(job["directory"])
    serialization_start = time.perf_counter()
    serialization_cpu_start = time.process_time()
    info.update(model_id=job["model_id"], scenario=job["scenario"], track=job["track"],
        replication=job["replication"], dgp_seed=job["dgp_seed"],
        dgp_seed_namespace=_MATCHED_CONTROL.get(job["scenario"], job["scenario"]),
        manifest_identity=job["manifest_identity"], computation_identity=job["computation_identity"],
        job_identity=job["job_identity"], data_identity=job.get("data_identity"),
        stage=protocol.stage, terminal_scientific_result=True,
        outcome="qualified" if info["qualified"] else "method_failure_with_fallback",
        receipt_schema="benchmark-job-receipt-4.0")
    if job["replication"] in protocol.retain_paths_replications:
        if forecast is not None:
            forecast.save(directory/"forecast.npz")
        Forecast(operational.mean, None, operational.coordinates, operational.site_ids,
                 {"fallback": info["fallback"]}).save(directory/"operational_point.npz")
    artifacts = []
    for name in ("model.npz", "forecast.npz", "operational_point.npz", "selection.json"):
        path = directory/name
        if path.is_file():
            artifacts.append({"file": name, "bytes": path.stat().st_size, "sha256": _file_hash(path)})
    info["artifacts"] = artifacts
    info["artifact_serialization_seconds"] = time.perf_counter()-serialization_start
    info["serialization_seconds"] = info.get("serialization_seconds",0.)+info["artifact_serialization_seconds"]
    info["total_pipeline_seconds"] = info.get("tuning_seconds",0.)+info["total_seconds"]+info["artifact_serialization_seconds"]
    info["artifact_serialization_cpu_seconds"] = time.process_time()-serialization_cpu_start
    info["total_pipeline_cpu_seconds"] = (None if info.get("tuning_cpu_seconds") is None else
        info["tuning_cpu_seconds"]+info["cpu_seconds"]+info["artifact_serialization_cpu_seconds"])
    info["process_peak_rss_bytes"] = _peak_rss()
    state_path = directory/"state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    info["infrastructure_retries"] = state.get("failures", [])
    info["compute_accounting_complete"] = not bool(info["infrastructure_retries"]) and not info.get("fit_resumed",False)
    info = jsonable(info)
    info["receipt_identity"] = content_hash(info)
    write_json(directory/"receipt.json", info)
    _record_state(job, "qualified" if info["qualified"] else "unqualified",
                  terminal=True, phase="complete", finished_unix=time.time())
    return info


def _execute_job(job, progress=None):
    """Spawn entry point; exclusively owns one scientific job directory."""
    with single_thread_runtime():
        if source_fingerprint() != job["source_fingerprint"]:
            raise StudyInvalidatedError("Source changed after the run manifest was sealed")
        if benchmark_environment() != job["expected_environment"]:
            raise StudyInvalidatedError("Worker numerical runtime differs from the sealed computation identity")
        _record_state(job, "running", phase="selection", started=True, worker_pid=os.getpid(),
                      started_unix=time.time(), attempt=job["attempt"])
        protocol = BenchmarkProtocol(**job["protocol"])
        try:
            sim = Simulation.load(job["data_path"])
        except (ValueError, KeyError) as error:
            raise StudyInvalidatedError("Worker dataset failed numeric artifact verification") from error
        if (_file_hash(job["data_path"]) != job["data_sha256"] or
            content_hash({"y":sim.observation,"times":sim.times,"coordinates":sim.coordinates}) != job["data_observation_hash"]):
            raise StudyInvalidatedError("Worker dataset differs from its admitted data receipt")
        directory = Path(job["directory"])
        selection_path = directory/"selection.json"
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())
            if (selection.get("job_identity") != job["job_identity"] or
                selection.get("selection_identity") != content_hash(
                    {k:v for k,v in selection.items() if k!="selection_identity"})):
                raise StudyInvalidatedError("Cached selection is incompatible or its content changed")
        else:
            selection = select_configuration(job["model_id"], sim, protocol, job["track"], job["seed"],
                          directory=directory/"tuning", progress=progress)
            selection = jsonable(selection)
            selection["job_identity"] = job["job_identity"]
            selection["selection_identity"] = content_hash(selection)
            write_json(selection_path, selection)
        _record_state(job, "running", phase="final_fit")
        info, forecast, operational = _final_job(job["model_id"], sim, protocol, job["track"],
                                               job["seed"], directory, selection)
        if source_fingerprint() != job["source_fingerprint"]:
            raise StudyInvalidatedError("Benchmark source changed during execution")
        return _finish_job(job, protocol, info, forecast, operational)


def _process_failure(job, error):
    """Record interruption; never invent a model fit, score, or fallback."""
    path = Path(job["directory"])/"state.json"
    current = json.loads(path.read_text()) if path.exists() else {}
    event = {"attempt": int(job.get("attempt", current.get("attempts", 0))),
             "started": bool(current.get("started", False)),
             "last_phase": current.get("phase", "queued"),
             "error": f"{type(error).__name__}: {error}",
             "finished_unix": time.time(), "incomplete_compute_accounting": True,
             "scientific_seed_unchanged": True}
    event["known_attempt_wall_seconds"] = (max(0.,time.time()-current["started_unix"])
        if current.get("started") and current.get("started_unix") is not None else None)
    event["known_attempt_wall_scope"] = "elapsed since the persisted worker start; may include time between process loss and its detection"
    return _record_state(job, "interrupted", terminal=False,
                         failures=[*current.get("failures", []), event],
                         phase=current.get("phase", "queued"))


def _ensure_dataset(job, protocol, verified):
    path = Path(job["data_path"])
    key = str(path)
    if key in verified:
        job.update(verified[key])
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = path.with_name("data_receipt.json")
    if path.exists():
        try:
            sim = Simulation.load(path)
        except (ValueError,KeyError) as error:
            raise StudyInvalidatedError("Cached dataset failed numeric artifact verification") from error
        if sim.metadata["seed"] != job["dgp_seed"] or len(sim.observation) != protocol.n_times:
            raise StudyInvalidatedError("Cached dataset has inconsistent seed or dimensions")
    else:
        options = dict((protocol.generator_options or {}).get(job["scenario"], {}))
        if job["scenario"] == "g6_velocity_shift":
            options.setdefault("origin_index", protocol.final_origin)
            options.setdefault("break_lead", min(25, protocol.horizon))
        sim = generate(job["scenario"], protocol.n_times, n_side=protocol.n_side,
                       n_observed=protocol.n_observed, n_calibration=protocol.n_calibration,
                       seed=job["dgp_seed"], **options)
        sim.save(path)
    data = {"schema": "benchmark-data-receipt-1.0", "manifest_identity": job["manifest_identity"],
            "dgp_seed": job["dgp_seed"], "sha256": _file_hash(path),
            "observation_hash": content_hash({"y": sim.observation, "times": sim.times,
                                            "coordinates": sim.coordinates}),
            "observed_ids": [sim.site_ids[i] for i in sim.observed_indices],
            "calibration_ids": [sim.site_ids[i] for i in sim.calibration_indices],
            "target_ids": [sim.site_ids[i] for i in sim.test_indices]}
    data = jsonable(data)
    data["identity"] = content_hash(data)
    if receipt_path.exists():
        if json.loads(receipt_path.read_text()) != data:
            raise StudyInvalidatedError("Cached dataset provenance or bytes changed")
    else:
        write_json(receipt_path, data)
    verified[key] = {"data_identity":data["identity"],"data_sha256":data["sha256"],
                     "data_observation_hash":data["observation_hash"]}
    job.update(verified[key])


def _run_benchmark(protocol, output, resume, progress, workers, power_freeze, max_memory_mb,
                   max_infrastructure_retries, max_jobs, max_wall_seconds):
    from .execution import plan_execution, host_resources, estimate_job_memory
    for name, value, minimum in (("max_infrastructure_retries", max_infrastructure_retries, 0),
                                ("max_jobs", max_jobs, 1)):
        if value is not None and (not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < minimum):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if max_wall_seconds is not None and (not np.isfinite(max_wall_seconds) or max_wall_seconds <= 0):
        raise ValueError("max_wall_seconds must be positive; it pauses between atomic jobs")
    frozen = None
    if protocol.stage == "confirmation":
        from .studies import verify_confirmation_freeze
        frozen = verify_confirmation_freeze(protocol, power_freeze)
    requested_workers = workers
    execution = plan_execution(protocol, workers=workers, max_memory_mb=max_memory_mb)
    while not execution["memory_admissible"] and workers > 1:
        workers -= 1
        execution = plan_execution(protocol, workers=workers, max_memory_mb=max_memory_mb)
    if not execution["memory_admissible"]:
        raise MemoryError("One job exceeds current safe admission allowance; no model dimensions or diagnostics were changed. Inspect benchmark plan or resume on a suitable host.")
    if any(MODELS[m].family == "neural" for m in protocol.models):
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("This roster requires the optional CPU-neural environment; install it before running.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = source_fingerprint()
    env = benchmark_environment()
    design_identity = content_hash({"protocol": protocol.to_dict(),
                                    "confirmation_freeze": None if frozen is None else frozen["identity"]})
    computation_identity = content_hash({"design_identity": design_identity,
        "source_fingerprint": fingerprint, "environment": env,
        "algorithm_schema": "benchmark-computation-4.0"})
    manifest = jsonable({"protocol": protocol.to_dict(), "protocol_identity": protocol.identity,
        "source_fingerprint": fingerprint, "environment": env, "stage": protocol.stage,
        "schema": "benchmark-run-4.0", "design_identity": design_identity,
        "computation_identity": computation_identity,
        "confirmation_freeze": None if frozen is None else frozen["identity"]})
    manifest["identity"] = content_hash(manifest)
    manifest_path = output/"manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if not resume or previous != manifest:
            raise ValueError("Output has a different scientific protocol/source/runtime; use a new directory")
    else:
        write_json(manifest_path, manifest)
        protocol.save(output/"resolved_protocol.yaml")
        if frozen is not None:
            write_json(output/"confirmation_freeze.json", frozen)
    execution.update(requested_workers=int(requested_workers), admitted_workers=int(workers),
                     max_infrastructure_retries=int(max_infrastructure_retries),
                     max_jobs=max_jobs, max_wall_seconds=max_wall_seconds,
                     started_unix=time.time(), manifest_identity=manifest["identity"])
    attempt_path = output/"execution_attempts"/f"{time.time_ns()}_{os.getpid()}.json"
    write_json(attempt_path, execution)
    write_json(output/"execution_plan.json", execution)
    receipts, jobs, all_jobs, verified_data = [], [], [], {}
    namespace = "final_dgp" if protocol.stage == "confirmation" else "development_dgp"
    for scenario in protocol.scenarios:
        for replication in range(protocol.replications):
            dataset_dir = output/scenario/f"rep_{replication:03d}"
            dgp_seed = dgp_seed_for(protocol, scenario, replication)
            for track in protocol.tracks:
                models = effective_models(protocol, track)
                for model_id in models:
                    directory = dataset_dir/track/model_id
                    directory.mkdir(parents=True, exist_ok=True)
                    job = {"protocol": protocol.to_dict(), "data_path": str(dataset_dir/"simulation.npz"),
                        "directory": str(directory), "model_id": model_id, "scenario": scenario,
                        "track": track, "replication": replication, "dgp_seed": int(dgp_seed),
                        "seed": seed_for(protocol.master_seed, namespace, scenario, replication, track, model_id),
                        "manifest_identity": manifest["identity"], "computation_identity": computation_identity,
                        "source_fingerprint": fingerprint,"expected_environment":env}
                    job["job_identity"] = content_hash({"manifest": manifest["identity"], "scenario": scenario,
                        "replication": replication, "track": track, "model_id": model_id, "seed": job["seed"]})
                    all_jobs.append(job)
                    saved = _load_terminal(job)
                    if saved is not None:
                        _ensure_dataset(job, protocol, verified_data)
                        if saved.get("data_identity") != job["data_identity"]:
                            raise StudyInvalidatedError("Completed job refers to changed data")
                        receipts.append(saved)
                    else:
                        jobs.append(job)
    started = time.perf_counter()
    submitted = 0
    pause_reason = None

    def budget_available():
        nonlocal pause_reason
        if max_jobs is not None and submitted >= max_jobs:
            pause_reason = "job-attempt chunk limit"
            return False
        if max_wall_seconds is not None and time.perf_counter()-started >= max_wall_seconds:
            pause_reason = "between-job wall-time limit"
            return False
        host = host_resources()
        ceiling = host["memory_limit_bytes"]
        current = host.get("memory_admission_estimate_bytes", host["memory_current_bytes"])
        if ceiling is not None and current is not None and current > ceiling-min(2*1024**3, int(.25*ceiling)):
            pause_reason = "current aggregate memory headroom"
            return False
        return True

    def prepare_attempt(job, active_jobs=()):
        nonlocal submitted, pause_reason
        state_path = Path(job["directory"])/"state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        attempts = int(state.get("attempts", 0))
        if state.get("status") in {"running","queued"} and attempts:
            job["attempt"] = attempts
            _process_failure(job,RuntimeError("Previous coordinator ended before a terminal receipt; interrupted attempt recovered"))
        if attempts >= 1+max_infrastructure_retries:
            _record_state(job, "unresolved_infrastructure_failure", terminal=False)
            return False
        estimate = estimate_job_memory(protocol,job["model_id"],job["track"],job["seed"])
        job["resource_estimate"] = estimate
        reserved = sum(item["resource_estimate"]["estimated_bytes"] for item in active_jobs)
        total = reserved+estimate["estimated_bytes"]+execution["memory"]["estimated_parent_bytes"]
        allowance = execution["allowance"]["effective_job_budget_bytes"]
        host = host_resources()
        ceiling = host["memory_limit_bytes"]
        current = host.get("memory_admission_estimate_bytes", host["memory_current_bytes"])
        reserve = execution["allowance"]["reserved_headroom_bytes"] or 0
        fits_budget = allowance is None or total <= allowance
        fits_headroom = ceiling is None or current is None or current+estimate["estimated_bytes"] <= ceiling-reserve
        job["resource_deferred"] = not (fits_budget and fits_headroom)
        if job["resource_deferred"]:
            _record_state(job,"pending_resource",terminal=False,resource_estimate=estimate)
            if not active_jobs:
                pause_reason = "actual resolved job exceeds available memory admission; scientific dimensions unchanged"
            return False
        _ensure_dataset(job, protocol, verified_data)
        job["attempt"] = attempts+1
        submitted += 1
        _record_state(job, "queued", attempts=attempts+1, attempt=attempts+1,
                      phase="queued", started=False, terminal=False,resource_estimate=estimate)
        return True

    def accept(info):
        receipts.append(info)
        receipts.sort(key=lambda r:(r["scenario"], r["track"], r["model_id"], r["replication"]))
        if progress:
            progress({"event": "complete", "scenario": info["scenario"], "replication": info["replication"],
                "model": info["model_id"], "track": info["track"], "qualified": info["qualified"],
                "fallback": info["fallback"], "rmse": info["operational_scores"]["rmse"],
                "seconds": info["total_seconds"]})

    def handle_error(job, error, failed):
        if isinstance(error, StudyInvalidatedError):
            _record_state(job, "invalidated", terminal=False, error=str(error))
            raise error
        saved = _load_terminal(job)
        if saved is not None:
            accept(saved)
            return
        state = _process_failure(job, error)
        failed.append(job)
        if progress:
            progress({"event": "interrupted", "model": job["model_id"], "track": job["track"],
                      "attempt": job["attempt"], "started": state.get("started", False),
                      "error": f"{type(error).__name__}: {error}"})

    invalidated = None
    pending = list(jobs)
    try:
        for recovery in range(1+max_infrastructure_retries):
            failed = []
            queue = deque(pending)
            if workers == 1:
                while queue and budget_available():
                    job = queue.popleft()
                    if not prepare_attempt(job):
                        if job.get("resource_deferred"):
                            queue.appendleft(job)
                            break
                        continue
                    try:
                        accept(_execute_job(job, progress))
                    except Exception as error:
                        handle_error(job, error, failed)
            elif queue:
                with ProcessPoolExecutor(max_workers=workers,
                                         mp_context=multiprocessing.get_context("spawn")) as pool:
                    active = {}
                    broken = False
                    while active or (queue and not broken):
                        while queue and not broken and len(active) < workers and budget_available():
                            job = queue.popleft()
                            if not prepare_attempt(job,tuple(active.values())):
                                if job.get("resource_deferred"):
                                    queue.appendleft(job)
                                    break
                                continue
                            try:
                                active[pool.submit(_execute_job, job)] = job
                            except Exception as error:
                                handle_error(job, error, failed)
                                broken = isinstance(error, BrokenProcessPool)
                        if not active:
                            break
                        completed, _ = wait(active, return_when=FIRST_COMPLETED)
                        for future in completed:
                            job = active.pop(future)
                            try:
                                accept(future.result())
                            except Exception as error:
                                handle_error(job, error, failed)
                                broken = broken or isinstance(error, BrokenProcessPool)
            pending = failed+list(queue)
            if not pending or pause_reason:
                break
        for job in pending:
            path = Path(job["directory"])/"state.json"
            if path.exists():
                state = json.loads(path.read_text())
                if int(state.get("attempts", 0)) >= 1+max_infrastructure_retries:
                    _record_state(job, "unresolved_infrastructure_failure", terminal=False)
    except StudyInvalidatedError as error:
        invalidated = str(error)
    except BaseException:
        execution.update(finished_unix=time.time(), elapsed_seconds=time.perf_counter()-started,
                         submitted_attempts=submitted, status="interrupted")
        write_json(attempt_path, execution)
        raise
    final_source = source_fingerprint()
    stable = final_source == fingerprint and protocol.identity == manifest["protocol_identity"] and invalidated is None
    receipts.sort(key=lambda r:(r["scenario"], r["track"], r["model_id"], r["replication"]))
    summary = aggregate_results(receipts, seed=protocol.master_seed,
        primary_comparisons=frozen.get("primary_comparisons") if frozen is not None
                            and frozen.get("planning_mode") == "paired_macro" else None)
    unresolved = 0
    for job in all_jobs:
        path = Path(job["directory"])/"state.json"
        if path.exists() and json.loads(path.read_text()).get("status") == "unresolved_infrastructure_failure":
            unresolved += 1
    complete = stable and len(receipts) == len(all_jobs)
    status = ("execution only" if protocol.stage == "execution_test" else
              "development evidence; no confirmatory superiority claim" if protocol.stage == "development" else
              "confirmation profile; inspect qualified populations and precision receipt")
    if not stable:
        status = "invalidated: source, protocol, or persisted identity changed during execution"
    elif not complete:
        status = "incomplete: unresolved infrastructure or intentionally paused work; "+status
    summary.update(manifest_identity=manifest["identity"], stage=protocol.stage,
                   design_identity=design_identity, computation_identity=computation_identity,
                   jobs=len(receipts), scheduled_jobs=len(all_jobs), pending_jobs=len(all_jobs)-len(receipts),
                   unresolved_infrastructure_failures=unresolved, complete=complete,
                   profile=protocol.name, source_stable=stable, final_source_fingerprint=final_source,
                   comparison_status=status)
    if frozen is not None:
        summary.update(power_status=frozen["power_status"], power_cap_binding=frozen["power"]["cap_binding"])
    if invalidated is not None:
        summary["invalidation_reason"] = invalidated
    write_json(output/"summary.json", summary)
    execution.update(finished_unix=time.time(), elapsed_seconds=time.perf_counter()-started,
        submitted_attempts=submitted, status="complete" if complete else "invalidated" if not stable else "paused" if pause_reason else "incomplete",
        pause_reason=pause_reason, completed_scientific_jobs=len(receipts),
        aggregate_memory_at_finish=host_resources())
    write_json(attempt_path, execution)
    if not stable:
        raise StudyInvalidatedError("Benchmark source/protocol or persisted identity changed during execution; results retained but invalidated.")
    return summary


def benchmark_status(output):
    """Read status without loading model arrays or mutating the campaign."""
    output = Path(output)
    manifest = json.loads((output/"manifest.json").read_text())
    summary_path = output/"summary.json"
    states = []
    for path in sorted(output.glob("*/rep_*/[AB]/*/state.json")):
        state = json.loads(path.read_text())
        states.append({"job": str(path.parent.relative_to(output)), "status": state["status"],
                       "phase": state.get("phase"), "attempts": state.get("attempts", 0)})
    return {"manifest_identity": manifest["identity"],
            "summary": json.loads(summary_path.read_text()) if summary_path.exists() else None,
            "states": states}


def run_benchmark(protocol: BenchmarkProtocol, output, *, resume=True, progress=None,
                  workers=1, power_freeze=None, max_memory_mb=None,
                  max_infrastructure_retries=2, max_jobs=None, max_wall_seconds=None):
    """Run/resume a scientifically frozen campaign with mutable scheduling.

    Worker loss is retryable and never becomes a fabricated scientific fallback.
    max_jobs limits submitted attempts; max_wall_seconds pauses between
    atomic jobs without changing any model's fitting budget.
    """
    if not isinstance(protocol, BenchmarkProtocol):
        raise TypeError("run_benchmark requires a validated BenchmarkProtocol")
    if protocol.stage == "confirmation":
        from .studies import verify_confirmation_freeze
        verify_confirmation_freeze(protocol, power_freeze)
    from .execution import campaign_lock
    with single_thread_runtime(), campaign_lock(output):
        try:
            return _run_benchmark(protocol, output, resume, progress, workers, power_freeze, max_memory_mb,
                                  max_infrastructure_retries, max_jobs, max_wall_seconds)
        except StudyInvalidatedError as error:
            output = Path(output)
            summary_path = output/"summary.json"
            previous = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            audit = {"status":"invalidated", "reason":str(error), "observed_unix":time.time(),
                     "previous_summary_identity":content_hash(previous), "previous_summary":previous}
            write_json(output/"invalidation.json", audit)
            if previous:
                previous.update(complete=False, provenance_stable=False,
                                comparison_status="invalidated: persisted evidence failed verification",
                                invalidation_reason=str(error))
                write_json(summary_path, previous)
            raise
