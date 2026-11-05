"""Resumable, resource-bounded full-field KS screening and validation follow-up.

Only training and official validation responses are accessed. Atomic origin
receipts are reused; retained native sampler state supplies exact fit resume.
Resource pilots have no scientific rankings and cannot reject a candidate for
the deliberately short pilot's convergence. One numerical worker runs at a
time. The coordinator can be detached with the companion launch command.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

from ..io import content_hash, seed_for
from .ks_data import KSAdapter, audit_cache, circle_embedding
from .ks_fullfield_candidates import (SITES, active_configuration, build_registry,
                                     estimate_cost)
from .ks_metrics import score, cluster_summary
from .ks_runtime import admission, atomic_json, file_hash, jsonable, now, resource_snapshot


MODULE = "bayes_stesn.applications.ks_fullfield_runner"
PROTOCOLS = {"A": "interleaved", "B": "all_observed", "C": "contiguous"}
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
TERMINAL = {"complete", "failed", "inference-unqualified", "resource-blocked"}
SEARCH_STAGES = {"pilot", "preflight", "anchors", "screen", "promote", "precision"}
SELECTION_TRAJECTORIES = [0, 5, 10, 15, 4, 9]
CONFIRMATION_TRAJECTORIES = [i for i in range(20) if i not in SELECTION_TRAJECTORIES]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_once(path, record):
    """Retain an existing bound receipt; never overwrite a terminal outcome."""
    path = Path(path)
    if path.exists():
        if read_json(path) != jsonable(record):
            raise ValueError(f"Immutable receipt differs: {path}")
        return
    atomic_json(path, record)


class RunLedger:
    """Small status view; immutable terminal records remain authoritative."""
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)

    def read(self, job_id):
        path = self.root / (job_id + ".json")
        return read_json(path) if path.exists() else None

    def write(self, job_id, state, **values):
        if state not in TERMINAL | {"pending", "running"}:
            raise ValueError("Unknown full-field ledger state")
        old = self.read(job_id) or {}
        atomic_json(self.root / (job_id + ".json"),
                    {**old, **values, "job_id": job_id, "state": state, "updated_utc": now()})


def source_identity():
    """Bind all core numerical modules and the actual application dependencies."""
    application = Path(__file__).resolve().parent
    core = application.parent
    sources = {"core/" + p.name: file_hash(p) for p in sorted(core.glob("*.py"))}
    for name in ("ks_fullfield_runner.py", "ks_fullfield_candidates.py", "ks_fullfield_native.py",
                 "ks_data.py", "ks_metrics.py", "ks_runtime.py", "ks_native.py", "ks_baselines.py"):
        sources["applications/" + name] = file_hash(application / name)
    for name in ("contracts.py", "classical.py", "esn.py"):
        sources["benchmark/" + name] = file_hash(core / "benchmark" / name)
    return {"files": sources, "python": sys.version,
            "packages": {n: importlib.metadata.version(n) for n in
                         ("numpy", "scipy", "h5py", "threadpoolctl")}}


def process_identity(pid=None):
    """Identity in the /proc mount's PID namespace, never os.getpid()'s alias."""
    folder = Path("/proc/self") if pid is None else Path(f"/proc/{int(pid)}")
    try:
        # Field 22 follows a potentially space-containing parenthesized comm.
        stat = (folder/"stat").read_text()
        visible_pid = int(stat.split(" ", 1)[0])
        tail = stat.rsplit(")", 1)[1].split()
        if tail[0] == "Z": return None
        status = (folder/"status").read_text().splitlines()
        nspid = next((line.split()[1:] for line in status if line.startswith("NSpid:")), [str(visible_pid)])
        return {"pid": visible_pid, "namespace_pid": int(nspid[-1]),
                "pid_namespace": os.stat(folder/"ns/pid").st_ino, "start_ticks": tail[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def process_matches(record):
    return record is not None and process_identity(record.get("pid", -1)) == record


def child_process_identity(namespace_pid):
    """Map a Popen.pid into the visible /proc mount without confusing aliases.

    The execution environment exposes the host /proc inside a nested PID
    namespace, and some proc mounts omit task/children. Match NSpid and the
    namespace inode instead of assuming the Popen integer is a visible PID.
    """
    namespace = os.stat("/proc/self/ns/pid").st_ino
    for folder in Path("/proc").iterdir():
        if not folder.name.isdigit(): continue
        try:
            if os.stat(folder/"ns/pid").st_ino != namespace: continue
            identity = process_identity(int(folder.name))
            if identity and identity["namespace_pid"] == int(namespace_pid): return identity
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return None


def process_rss(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"): return 1024*int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0


def settle_attempts(output):
    """Refuse live orphans; conservatively charge dead unobserved attempts.

    A terminal worker receipt gives its finish time even if its coordinator
    died. Otherwise the interval until this restart is charged as an explicit
    upper bound: a restart must never erase unobserved worker time.
    """
    root = Path(output)
    attempts = [(p, read_json(p)) for p in root.glob("**/attempts/*.json")]
    live = [r for _, r in attempts if r.get("state") == "running" and process_matches(r.get("process"))]
    if live:
        raise RuntimeError("An orphaned numerical worker is still active; do not launch another. "
            "Use tools/run_ks_fullfield.py stop --output RUN_ROOT, then resume. "
            f"Owned worker PIDs: {[r['process']['pid'] for r in live]}")
    for path, row in attempts:
        if row.get("state") != "running": continue
        terminal = path.parent.parent/"terminal"/(row["job_id"]+".json")
        finish = read_json(terminal).get("finished_utc") if terminal.exists() else None
        endpoint = datetime.fromisoformat(finish).timestamp() if finish else time.time()
        elapsed = max(float(row.get("elapsed_seconds", 0.)), endpoint-row["started_epoch"])
        atomic_json(path, {**row, "state": "interrupted", "elapsed_seconds": elapsed,
            "settled_utc": now(), "charge_policy": "worker_terminal_finish" if finish else
            "conservative_unobserved_interval_until_resume", "observed_exact_end": bool(finish)})


def evict_closed_artifact_pages(folder):
    """Advise release of closed task-owned NPZ pages; preserve every byte.

    This is scoped to verified regular files inside one completed fit folder.
    It never touches data HDF5, other projects, or global drop_caches.
    The advisory API need not actually reclaim pages on every filesystem.
    """
    folder = Path(folder).resolve()
    result = {"supported": hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"),
        "files_advised": 0, "bytes_advised": 0, "errors": [], "files_deleted": 0}
    if not result["supported"]: return result
    for path in sorted(folder.rglob("*.npz")):
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(folder): continue
        try:
            with path.open("rb") as handle:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            result["files_advised"] += 1; result["bytes_advised"] += path.stat().st_size
        except OSError as error:
            result["errors"].append({"file": str(path.relative_to(folder)), "error": str(error)})
    return result


def _job_identity(job):
    return content_hash(jsonable({k: v for k, v in job.items() if k != "job_id"}))


def terminal_result(root, job):
    path = Path(root) / "terminal" / f"{job['job_id']}.json"
    if not path.exists(): return None
    record = read_json(path)
    if record.get("job_identity") != _job_identity(job) or record.get("state") not in TERMINAL:
        raise ValueError("Incompatible immutable job outcome")
    return record


def fit_key(track, candidate, trajectory, seed, phase="core"):
    return content_hash({"track": track, "phase": phase,
                         "effective_hash": candidate["effective_hash"],
                         "trajectory": int(trajectory), "seed": int(seed)})[:32]


def preflight_key(track, candidate, trajectory, seed):
    active = candidate["active_config"]
    # Both targets fit the feature map to the original observed state. Prior
    # variants do not change the map. Only this training-only proof is shared.
    return content_hash({"track": track, "trajectory": int(trajectory), "seed": int(seed),
        "features": active["features"], "spatial_mean": active["spatial_mean"],
        "conditioning": active["conditioning"]})[:32]


def synchronization(prepared, lengths, *, absolute_tolerance=1e-8, relative_tolerance=1e-6):
    """Compare raw-state replays; the likelihood response may be an increment."""
    y, features = prepared.data.y, prepared.features
    state, z = features.update(y[-1], features.states[-1])
    rows = []
    for length in lengths:
        if length > len(y):
            rows.append({"history": int(length), "available": False}); continue
        states, _ = features.history(y[-length:])
        actual_state, actual_z = features.update(y[-1], states[-1])
        ds, dz = float(np.max(np.abs(actual_state-state))), float(np.max(np.abs(actual_z-z)))
        ss, sz = max(float(np.max(np.abs(state))), 1e-12), max(float(np.max(np.abs(z))), 1e-12)
        rows.append({"history": int(length), "available": True,
            "state_max_absolute_difference": ds, "feature_max_absolute_difference": dz,
            "state_max_relative_difference": ds/ss, "feature_max_relative_difference": dz/sz,
            "passed": bool(ds <= absolute_tolerance + relative_tolerance*ss
                           and dz <= absolute_tolerance + relative_tolerance*sz)})
        del states
    return {"rows": rows, "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "permitted_history_rows": len(y), "feature_identity": features.fingerprint,
        "recurrent_diagnostics": getattr(features.core, "recurrent_diagnostics", None),
        "replayed_values": "prepared.data.y: observed state, never likelihood increments",
        "scope": "empirical training-only endpoint synchronization; not a universal dynamical guarantee"}


def training_preflight(training, candidate, seed, lengths):
    from .ks_fullfield_native import prepare_fullfield
    prepared = prepare_fullfield(training.panel, candidate["config"], seed=seed)
    if prepared.spec.features.kind != "raw_esn":
        raise ValueError("A reduced-input model entered the full-field preflight")
    phi = np.asarray(prepared.Phi)
    singular = np.linalg.svd(phi, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    queries = circle_embedding(np.arange(64)*22./64.)
    query_phi = prepared.basis.transform(queries)
    return {"synchronization": synchronization(prepared, lengths),
        "geometry": {"training_basis_condition": condition,
            "training_basis_rank": int(np.linalg.matrix_rank(phi)),
            "training_basis_shape": list(phi.shape),
            "max_target_basis_norm": float(np.max(np.linalg.norm(query_phi, axis=1))),
            "max_training_basis_norm": float(np.max(np.linalg.norm(phi, axis=1)))},
        "dimensions": {"input_sites": int(training.panel.y.shape[1]),
            "spatial": int(phi.shape[1]), "temporal": int(prepared.Z.shape[1]),
            "readout": int(phi.shape[1]*prepared.Z.shape[1]),
            "likelihood_rows": int(prepared.Y.shape[1])},
        "training_identity": dict(training.identity)}


def _forecast_identity(job, fitted_identity, first):
    return content_hash({"native_fit": fitted_identity, "first": first,
        "history": job["history"], "horizon": job["horizon"], "paths": job["paths"],
        "forecast_seed": job["forecast_seed"], "parameter_seed": job.get("parameter_seed"),
        "phase": job["phase"], "partition": "validation",
        "score_source": job["source"]["files"]["applications/ks_metrics.py"]})


def _origin_path(root, job, first, identity):
    namespace = "precision_forecasts" if job["stage"] == "precision" else "forecasts"
    return Path(root)/namespace/job["fit_key"]/f"{first:05d}_{job['paths']}_{identity[:16]}.json"


def _physical_scores(adapter, training, fitted, job, first):
    window = adapter.model_window(job["trajectory"], first, "validation",
                                  history=job["history"], horizon=job["horizon"])
    stream = seed_for(job["forecast_seed"], job["track"], job["trajectory"], job["seed"], first)
    samples = np.empty((job["paths"], job["horizon"], len(window.target_site_ids)), dtype=np.float64)
    kwargs = {"batch_size": job["path_batch"]}
    if job.get("parameter_seed") is not None: kwargs["parameter_seed"] = job["parameter_seed"]
    expected_start = 0
    for batch in fitted.iter_forecasts(window.panel, job["horizon"], window.target_coordinates,
            window.target_site_ids, job["paths"], stream, **kwargs):
        a, b = batch.metadata["path_start"], batch.metadata["path_stop"]
        if a != expected_start or not a < b <= job["paths"]:
            raise ValueError("Predictive path batches have a gap, overlap or out-of-range index")
        samples[a:b] = training.transform.inverse(batch.samples)
        expected_start = b
    if expected_start != job["paths"]: raise ValueError("Incomplete predictive path stream")
    mean = samples.mean(axis=0, dtype=np.float64)
    # Evaluation is deliberately separate and happens after the autonomous path.
    truth = adapter.evaluation_window(job["trajectory"], first, "validation",
        history=job["history"], horizon=job["horizon"]).latent_clean
    metrics = score(mean, samples, truth, training.transform.scale, window.target_coordinates)
    return metrics, (window, samples, mean, truth)


def _scientific_failure(error):
    return isinstance(error, (FloatingPointError, np.linalg.LinAlgError)) or (
        isinstance(error, ValueError) and any(term in str(error).lower() for term in
        ("nonfinite", "finite", "spectral radius", "zero variation", "zero spectral")))


def _fit_receipt(fitted, path, job, training, start):
    return {"fit_key": job["fit_key"], "native_fit_identity": fitted.identity,
        "fit_sha256": file_hash(path), "fit_bytes": path.stat().st_size,
        "fit_seconds": float(fitted.diagnostics.get("fit_wall_seconds", time.perf_counter()-start)),
        "current_fit_or_load_seconds": time.perf_counter()-start,
        "diagnostics": fitted.diagnostics, "training_identity": dict(training.identity),
        "config": fitted.config, "track": job["track"], "phase": job["phase"],
        "trajectory": job["trajectory"], "seed": job["seed"],
        "candidate_id": job["candidate"]["id"], "completed_utc": now(),
        "resource": resource_snapshot(job["output"])}


def worker(job_path):
    """Exactly one process/writer; complete origins are restart boundaries."""
    threadpool_limits(1)
    job = read_json(job_path); root = Path(job["output"])
    if job["source"] != source_identity():
        raise ValueError("Numerical source/runtime changed; refusing the incompatible job")
    if job["partition"] != "validation":
        raise ValueError("The full-field campaign is validation-only")
    ledger = RunLedger(root/"ledger"); started = time.perf_counter()
    def finish(state, **values):
        result = {"job_id": job["job_id"], "state": state,
            "job_identity": _job_identity(job), "finished_utc": now(),
            "elapsed_seconds": time.perf_counter()-started,
            "resource": resource_snapshot(root), **values}
        write_once(root/"terminal"/f"{job['job_id']}.json", result)
        ledger.write(job["job_id"], state, job=job, **values)
    folder = root/"fits"/job["fit_key"]; folder.mkdir(parents=True, exist_ok=True)
    with (folder/"writer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger.write(job["job_id"], "running", job=job, process=process_identity(), started_utc=now())
        try:
            adapter = KSAdapter(job["cache"], PROTOCOLS[job["track"]], phase=job["phase"])
            training = adapter.training(job["trajectory"])
            if job["stage"] == "preflight":
                result = training_preflight(training, job["candidate"], job["seed"], job["lengths"])
                atomic_json(root/"preflight"/f"{job['preflight_key']}.json", result)
                finish("complete", result=result); return
            if job["stage"] in {"anchors", "anchor_confirm"}:
                from .ks_baselines import fit_vector
                summaries = {}
                for model_id, config in (("training_mean", {}), ("persistence", {}),
                    ("linear_var", {"lags": 4, "rank": SITES[job["track"]], "ridge": 1e-5,
                                    "target": "state", "max_features": 1024})):
                    fitted = fit_vector(model_id, training.panel, config, seed=job["seed"])
                    rows = []
                    for first in job["origins"]:
                        window = adapter.model_window(job["trajectory"], first, "validation",
                            history=job["history"], horizon=job["horizon"])
                        prediction = fitted.forecast(window.panel, job["horizon"],
                            window.target_coordinates, window.target_site_ids)
                        truth = adapter.evaluation_window(job["trajectory"], first, "validation",
                            history=job["history"], horizon=job["horizon"]).latent_clean
                        metrics = score(training.transform.inverse(prediction.mean), None, truth,
                                        training.transform.scale, window.target_coordinates)
                        rows.append({"first_target": first, "metrics": metrics})
                    summaries[model_id] = {"config": config, "diagnostics": fitted.diagnostics, "origins": rows}
                atomic_json(root/"anchors"/f"{job['track']}_t{job['trajectory']:02d}.json",
                    {"experiment_identity": job["experiment_identity"], "track": job["track"],
                     "trajectory": job["trajectory"], "history": job["history"], "models": summaries,
                     "schedule": "confirmation" if job["stage"] == "anchor_confirm" else "selection",
                     "partition": "validation", "model_seed": job["seed"],
                     "training_identity": dict(training.identity), "origins": job["origins"]})
                finish("complete"); return
            from .ks_fullfield_native import fit_fullfield
            path, receipt_path = folder/"fit.npz", folder/"receipt.json"
            previous = read_json(receipt_path) if receipt_path.exists() else None
            if previous and (not path.exists() or file_hash(path) != previous["fit_sha256"]):
                raise ValueError("Native fit checksum changed")
            fit_start = time.perf_counter()
            fitted = fit_fullfield(training.panel, config=job["candidate"]["config"],
                                  seed=job["seed"], checkpoint_path=path)
            if previous is None:
                previous = _fit_receipt(fitted, path, job, training, fit_start)
                write_once(receipt_path, previous)
            elif previous["native_fit_identity"] != fitted.identity:
                raise ValueError("Resumed native fit identity differs from its immutable receipt")
            if job["stage"] == "pilot":
                # Short-chain output only profiles resources. No labels/scores,
                # ranking, or inference-failure exclusion is inferred from it.
                first = int(adapter.splits["chronological"]["validation"][0])
                history = min(job["history"], len(training.panel.y))
                window = adapter.model_window(job["trajectory"], first, "validation",
                                              history=history, horizon=job["horizon"])
                forecast_start = time.perf_counter(); finite = True; batches = 0
                for batch in fitted.iter_forecasts(window.panel, job["horizon"], window.target_coordinates,
                        window.target_site_ids, job["paths"], job["forecast_seed"],
                        batch_size=job["path_batch"], require_qualified=False):
                    finite = finite and bool(np.isfinite(batch.samples).all()); batches += 1
                result = {"candidate_id": job["original_candidate_id"],
                    "fit_seconds": previous["fit_seconds"],
                    "forecast_seconds": time.perf_counter()-forecast_start,
                    "forecast_paths": job["paths"], "forecast_horizon": job["horizon"],
                    "forecast_batches": batches, "finite_paths": finite,
                    "pilot_qualified": bool(fitted.diagnostics.get("qualified")),
                    "diagnostics": fitted.diagnostics,
                    "training_bounds": list(adapter.train_bounds),
                    "validation_bounds": adapter.splits["chronological"]["validation"],
                    "training_identity": dict(training.identity),
                    "interpretation": "resource pilot only; short-chain nonconvergence is not a candidate rejection",
                    "evaluation_labels_exposed_to_resource_profile": False,
                    "facade_io_limitation": "The facade internally constructs labels; only its model output was consumed."}
                atomic_json(root/"pilots"/f"{job['fit_key']}.json", result)
                finish("complete", result=result); return
            if not fitted.diagnostics.get("qualified", False):
                finish("inference-unqualified", fit_key=job["fit_key"]); return
            sync_path = folder/f"synchronization_h{job['history']}.json"
            if sync_path.exists():
                sync = read_json(sync_path)
            else:
                sync = synchronization(fitted.core.prepared, (job["history"],))
                atomic_json(sync_path, sync)
            if not sync["rows"][0].get("passed", False):
                finish("failed", reason="prespecified_history_did_not_synchronize"); return
            completed, failed = 0, 0
            for first in job["origins"]:
                identity = _forecast_identity(job, fitted.identity, first)
                target = _origin_path(root, job, first, identity)
                if target.exists():
                    old = read_json(target)
                    if old["identity"] != identity: raise ValueError("Forecast identity mismatch")
                    completed += old["state"] == "complete"; failed += old["state"] == "failed"
                    continue
                snap = resource_snapshot(root)
                if snap["disk_free_bytes"] < job["budget"]["disk_reserve_bytes"]:
                    raise MemoryError("Disk reserve before forecast")
                begin = time.perf_counter()
                receipt = {"identity": identity, "fit_key": job["fit_key"],
                    "native_fit": fitted.identity, "track": job["track"], "phase": job["phase"],
                    "candidate_id": job["candidate"]["id"], "trajectory": job["trajectory"],
                    "seed": job["seed"], "first_target": first, "history": job["history"],
                    "horizon": job["horizon"], "paths": job["paths"], "qualified": True,
                    "forecast_seed": job["forecast_seed"], "parameter_seed": job.get("parameter_seed"),
                    "partition": "validation", "first_stage": job["stage"]}
                try:
                    metrics, arrays = _physical_scores(adapter, training, fitted, job, first)
                    if job.get("retain") and first == job["origins"][0]:
                        from ..io import atomic_npz
                        window, samples, mean, truth = arrays
                        atomic_npz(root/"examples"/f"{job['track']}_{job['candidate']['id']}_t{job['trajectory']:02d}_a{first}.npz",
                            mean=mean, truth=truth, lower=np.quantile(samples, .05, axis=0),
                            upper=np.quantile(samples, .95, axis=0),
                            coordinates=window.target_coordinates, input_coordinates=window.panel.coordinates,
                            first_target=np.array(first), training_scale=np.array(training.transform.scale))
                    del arrays
                    atomic_json(target, {**receipt, "state": "complete", "metrics": metrics,
                                         "seconds": time.perf_counter()-begin})
                    completed += 1
                except Exception as error:
                    if not _scientific_failure(error): raise
                    atomic_json(target, {**receipt, "state": "failed", "error": repr(error),
                                         "seconds": time.perf_counter()-begin})
                    failed += 1
            finish("complete" if failed == 0 else "failed",
                   completed_origins=completed, failed_origins=failed)
        except MemoryError as error:
            finish("resource-blocked", reason=str(error), traceback=traceback.format_exc())
        except Exception as error:
            if _scientific_failure(error):
                finish("failed", reason=repr(error), traceback=traceback.format_exc())
            else:
                ledger.write(job["job_id"], "pending", job=job, reason="execution_error",
                    error=repr(error), traceback=traceback.format_exc(),
                    elapsed_seconds=time.perf_counter()-started)
                raise


def default_configuration():
    """Prospective numerical defaults; budgets must be finalized after pilots."""
    return {
        "schema": "ks-fullfield-experiment/1", "tracks": ["A", "B"], "deferred_tracks": ["C"],
        "model_seed": 20261105, "forecast_seed": 2026110509, "path_batch": 32,
        "history_candidates": [512, 1024, 2048], "horizon": 100,
        "screen": {"trajectories": [0, 5], "origins": [6000, 6500, 7000, 7500], "paths": 256},
        "promote": {"trajectories": SELECTION_TRAJECTORIES,
            "origins": [6000, 6200, 6500, 6700, 7000, 7200, 7500, 7700],
            "paths": 512, "count": 6, "matched_prior_extra": 1},
        "precision": {"enabled": True, "trajectory": 0, "origins": [6000, 6500],
                      "paths": 512, "replicates": 4},
        "confirmation": {"trajectories": CONFIRMATION_TRAJECTORIES,
            "origins": list(range(6000, 8000, 100)), "paths": 512},
        "sensitivity": {"trajectories": [0, 5, 10, 15], "seeds": [20261106, 20261107],
            "origins": [6000, 6200, 6500, 6700, 7000, 7200, 7500, 7700], "paths": 512},
        "selection": {"early_horizon": 10, "early_rmse_relative_tolerance": .10,
            "relative_crps_near_tie": .01, "mc_sd_multiplier": 2.,
            "maximum_severe_divergent_origins": 0,
            "modes": ["distribution", "dynamics", "point"],
            "description": "Distribution: band-balanced CRPS. Dynamics: CRPS subject to early RMSE "
                "within10percent of best eligible early RMSE. Point: band-balanced MSE. "
                "CRPS near ties prefer longer valid prediction time, then fewer coefficients. "
                "All planned trajectory/origin outcomes and inference gates are required."},
        "pilot": {"labels": ["dynamics_a0.2_r0.9", "capacity1000_increment",
            "rhs_base_state", "rhs_capacity1000_increment", "lag32_all"],
            "reference_trajectories": [0, 1, 2], "other_trajectories": [0],
            "history": 512, "paths": 32, "warmup": 32, "draws": 32, "chains": 2,
            "functional_draws": 16, "init_maxiter": 10, "init_max_evaluations": 80},
        "budget": {"frozen_after_pilot": False, "workers": 1,
            "disk_reserve_bytes": 12781798196, "combined_worker_cap_bytes": 5368709120,
            "estimated_worker_peak_bytes": 512*1024**2,
            "total_worker_seconds_cap": 60*3600, "search_worker_seconds_cap": 26*3600,
            "wall_seconds_cap": 72*3600, "job_timeout_seconds": 7200,
            "pilot_job_timeout_seconds": 600, "preflight_job_timeout_seconds": 600,
            "heartbeat_seconds": 5, "minimum_container_headroom_bytes": 768*1024**2,
            "stage_worker_seconds_caps": {"preflight": 1800, "anchors": 900,
                "screen": 17*3600-120, "promote": 6*3600, "precision": 2*3600,
                "confirm": 24*3600, "sensitivity": 10*3600},
            "preparation_worker_seconds": 900,
            "preparation_charge_interpretation": "Conservative profiling/acceptance allowance; charged once to total and shared search, not separately to each stage.",
            "partial_roster_at_stage_cap": True,
            "evict_closed_task_artifact_pages": True,
            "measurements": None},
        "interpretation": "Exploratory validation-only expansion after original test inspection; "
            "original results remain frozen. No claim of universal optimality.",
    }


def validate_configuration(cfg):
    if cfg.get("schema") != "ks-fullfield-experiment/1": raise ValueError("Unknown configuration schema")
    if not cfg["tracks"] or len(set(cfg["tracks"])) != len(cfg["tracks"]) or set(cfg["tracks"]) - set(SITES):
        raise ValueError("Tracks must be unique A/B/C labels")
    deferred = cfg.get("deferred_tracks", [])
    if len(set(deferred)) != len(deferred) or set(deferred)-set(SITES) or set(deferred)&set(cfg["tracks"]):
        raise ValueError("Deferred tracks must be distinct supported tracks outside the main campaign")
    if cfg["budget"]["workers"] != 1:
        raise ValueError("This initial full-field execution permits one expensive worker")
    if cfg["horizon"] != 100: raise ValueError("The declared KS horizon is100")
    lengths = cfg["history_candidates"]
    if lengths != sorted(set(lengths)) or min(lengths) < 2 or max(lengths) > 6000:
        raise ValueError("Invalid common history candidates")
    for stage in ("screen", "promote", "confirmation", "sensitivity"):
        schedule = cfg[stage]
        if not schedule["trajectories"] or len(set(schedule["trajectories"])) != len(schedule["trajectories"]):
            raise ValueError("Duplicate or empty trajectory schedule")
        if any(not 0 <= i < 20 for i in schedule["trajectories"]): raise ValueError("Invalid core trajectory")
        if not schedule["origins"] or len(set(schedule["origins"])) != len(schedule["origins"]):
            raise ValueError("Duplicate or empty origin schedule")
        if any(a < 6000 or a+100 > 8000 or a < max(lengths) for a in schedule["origins"]):
            raise ValueError("Schedule crosses validation boundary or lacks possible common history")
        if not isinstance(schedule["paths"], int) or schedule["paths"] < 1: raise ValueError("Invalid path count")
    if set(cfg["promote"]["trajectories"]) & set(cfg["confirmation"]["trajectories"]):
        raise ValueError("Selection and confirmation trajectories overlap")
    if not set(cfg["screen"]["trajectories"]) <= set(cfg["promote"]["trajectories"]):
        raise ValueError("Promotion must contain screening trajectories")
    if cfg["precision"]["trajectory"] not in cfg["promote"]["trajectories"]:
        raise ValueError("Precision cases must belong to selection trajectories")
    if any(a < 6000 or a+100 > 8000 for a in cfg["precision"]["origins"]):
        raise ValueError("Precision schedule must use validation")
    for name in ("total_worker_seconds_cap", "search_worker_seconds_cap", "wall_seconds_cap",
                 "job_timeout_seconds", "pilot_job_timeout_seconds", "preflight_job_timeout_seconds",
                 "combined_worker_cap_bytes", "disk_reserve_bytes"):
        value = cfg["budget"][name]
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"Finite positive budget required: {name}")
    if cfg["budget"]["search_worker_seconds_cap"] >= cfg["budget"]["total_worker_seconds_cap"]:
        raise ValueError("Reserve worker time beyond screening")
    caps = cfg["budget"].get("stage_worker_seconds_caps", {})
    for stage, cap in caps.items():
        if stage not in SEARCH_STAGES | {"confirm", "anchor_confirm", "sensitivity"} or not np.isfinite(cap) or cap <= 0:
            raise ValueError("Stage worker caps must be finite, positive, and name supported stages")
    if cfg["budget"].get("partial_roster_at_stage_cap") and not all(s in caps for s in ("screen", "promote", "precision")):
        raise ValueError("Partial-roster selection requires separate screening, promotion and precision caps")
    if sum(v for s, v in caps.items() if s in SEARCH_STAGES) + cfg["budget"].get(
            "preparation_worker_seconds", 0.) > cfg["budget"]["search_worker_seconds_cap"]:
        raise ValueError("Stage search allowances exceed the shared search cap")
    if cfg["precision"]["enabled"] and cfg["precision"]["replicates"] < 2:
        raise ValueError("Predictive precision requires at least two independent repetitions")


def selection_frontier(rows):
    """Nondominated CRPS/early-RMSE/VPT set; directions are explicit."""
    eligible = [r for r in rows if r["eligible"]]
    result = []
    for r in eligible:
        v = (r["loss"], r["early_rmse"], -r["vpt_physical"])
        dominated = False
        for other in eligible:
            u = (other["loss"], other["early_rmse"], -other["vpt_physical"])
            if all(a <= b for a, b in zip(u, v)) and any(a < b for a, b in zip(u, v)):
                dominated = True; break
        if not dominated: result.append(r["id"])
    return sorted(result)


def choose_selections(rows, policy, mc_tolerances=None):
    eligible = [r for r in rows if r["eligible"]]
    if not eligible:
        return {"state": "no_eligible_candidate", "distribution": None, "dynamics": None,
                "point": None, "pareto_ids": [], "eligible_count": 0}
    mc_tolerances = mc_tolerances or {}
    def crps_choice(pool):
        best = min(r["loss"] for r in pool)
        best_id = min(pool, key=lambda r: (r["loss"], r["id"]))["id"]
        tolerances = {r["id"]: max(policy["relative_crps_near_tie"]*abs(best),
            mc_tolerances.get(best_id, {}).get(r["id"], 0.)) for r in pool}
        near = [r for r in pool if r["loss"] <= best+tolerances[r["id"]]]
        winner = min(near, key=lambda r: (-r["vpt_physical"],
            r["complexity"]["readout_coefficients"], r["loss"], r["id"]))
        return {"candidate": winner, "minimum_raw_loss_id": best_id,
                "near_tie_ids": sorted(r["id"] for r in near), "absolute_tolerances": tolerances}
    best_early = min(r["early_rmse"] for r in eligible)
    limit = (1+policy["early_rmse_relative_tolerance"])*best_early
    pool = [r for r in eligible if r["early_rmse"] <= limit]
    point = min(eligible, key=lambda r: (r["point_loss"], -r["vpt_physical"],
                                       r["complexity"]["readout_coefficients"], r["id"]))
    return {"state": "selected", "distribution": crps_choice(eligible),
        "dynamics": crps_choice(pool), "point": {"candidate": point},
        "early_rmse_best": best_early, "early_rmse_limit": limit,
        "dynamics_eligible_ids": sorted(r["id"] for r in pool),
        "pareto_ids": selection_frontier(rows), "eligible_count": len(eligible)}


def promotion_ids(rows, reference_id, count=6, matched_extra=1):
    eligible = [r for r in rows if r["eligible"]]
    if not eligible: return []
    by_id = {r["id"]: r for r in eligible}
    ids = []
    def add(cid):
        if cid in by_id and cid not in ids: ids.append(cid)
    add(reference_id)
    add(min(eligible, key=lambda r: (r["loss"], r["id"]))["id"])
    add(min(eligible, key=lambda r: (r["early_rmse"], r["id"]))["id"])
    add(min(eligible, key=lambda r: (-r["vpt_physical"], r["loss"], r["id"]))["id"])
    rhs = [r for r in eligible if r["prior_family"] == "rhs"]
    best_rhs = min(rhs, key=lambda r: (r["loss"], r["id"])) if rhs else None
    if best_rhs: add(best_rhs["id"])
    frontier = set(selection_frontier(rows))
    for r in sorted(eligible, key=lambda r: (r["id"] not in frontier, r["loss"], r["id"])):
        if len(ids) >= count: break
        add(r["id"])
    if len(ids) > count:
        raise ValueError("Promotion count cannot fit all prespecified mandatory roles")
    if best_rhs and best_rhs.get("matched_ridge_id") in by_id and best_rhs["matched_ridge_id"] not in ids:
        if matched_extra < 1: raise ValueError("Matched RHS comparison requires its declared extra control")
        add(best_rhs["matched_ridge_id"])
    return ids


def ordered_candidates(candidates):
    """Response-independent scientific controls, then balanced strata.

    Each candidate remains a two-trajectory bundle in the scheduler. Broad
    leak/radius proposals are ordered diagonally to cover both marginals early.
    Expensive RHS candidates receive genuine opportunities before the grid.
    """
    required = ["dynamics_a0.2_r0.9", "capacity128_increment", "rhs_base_state",
        "rhs_base_increment", "capacity1000_state", "capacity1000_increment",
        "lag32_latest", "lag32_all", "joint512_lag32_increment",
        "joint1000_lag8_increment", "joint1000_lag32_state",
        "rhs_joint1000_lag8_increment", "rhs_capacity1000_increment"]
    by_label = {c["label"]: c for c in candidates}
    result = [by_label[label] for label in required if label in by_label]
    seen = {c["id"] for c in result}; strata = {}
    for c in candidates:
        if c["id"] not in seen: strata.setdefault(c["stratum"], []).append(c)
    grid = strata.get("joint_leak_radius", [])
    if grid:
        leaks = sorted({c["config"]["features"]["leak_patterns"][0][0] for c in grid})
        radii = sorted({c["config"]["features"]["spectral_radius"] for c in grid})
        grid.sort(key=lambda c: ((radii.index(c["config"]["features"]["spectral_radius"])-
            leaks.index(c["config"]["features"]["leak_patterns"][0][0])) % len(radii),
            leaks.index(c["config"]["features"]["leak_patterns"][0][0])))
    while any(strata.values()):
        for group in sorted(strata):
            if strata[group]: result.append(strata[group].pop(0))
    if len(result) != len(candidates) or len({c["id"] for c in result}) != len(result):
        raise ValueError("Response-independent scheduling lost or repeated a candidate")
    return result


class Campaign:
    def __init__(self, config, cache, output, *, allow_unfrozen=False):
        threadpool_limits(1)
        self.cfg = read_json(config); validate_configuration(self.cfg)
        self.output = Path(output).resolve(); self.output.mkdir(parents=True, exist_ok=True)
        self.cache = str(Path(cache).resolve()); self.source = source_identity()
        self.registry = build_registry(self.cfg["model_seed"], self.cfg["tracks"] + self.cfg.get("deferred_tracks", []))
        if not allow_unfrozen and not self.cfg["budget"]["frozen_after_pilot"]:
            raise ValueError("Freeze the measured execution budget before scientific screening")
        audit_path = self.output/"cache_acceptance.json"
        if audit_path.exists():
            # Audit each coordinator invocation, not only path existence. Data
            # hashes and training transforms remain the authoritative boundary.
            audit = audit_cache(self.cache)
            if audit != read_json(audit_path): raise ValueError("Verified cache identity changed")
        else:
            audit = audit_cache(self.cache)
            if not audit["passed"]: raise ValueError("KS cache integrity check failed")
            atomic_json(audit_path, audit)
        self.cache_identity = content_hash(audit)
        self.identity = content_hash({"config": self.cfg, "registry": self.registry["identity"],
            "source": self.source, "cache": self.cache_identity})
        path = self.output/"experiment.json"
        if path.exists():
            old = read_json(path)
            if old["identity"] != self.identity:
                raise ValueError("Changed source/configuration/data: use a new versioned run root")
        else:
            atomic_json(path, {"identity": self.identity, "config": self.cfg,
                "registry_identity": self.registry["identity"], "source": self.source,
                "cache_identity": self.cache_identity, "created_utc": now(),
                "initial_resources": resource_snapshot(self.output), "test_scoring": False})
        write_once(self.output/"candidate_registry.json", self.registry)
        self.ledger = RunLedger(self.output/"ledger")
        settle_attempts(self.output)

    def candidates(self, track):
        return self.registry["tracks"][track]["candidates"]

    def job(self, stage, track, candidate, trajectory, origins=(), paths=256,
            seed=None, retain=False, phase="core", parameter_seed=None, forecast_seed=None):
        seed = self.cfg["model_seed"] if seed is None else int(seed)
        policy = self.output/"history_policy.json"
        history = read_json(policy)["history"] if policy.exists() else max(self.cfg["history_candidates"])
        job = {"output": str(self.output), "cache": self.cache, "source": self.source,
            "experiment_identity": self.identity, "stage": stage, "track": track, "phase": phase,
            "candidate": candidate, "trajectory": int(trajectory), "seed": seed,
            "fit_key": fit_key(track, candidate, trajectory, seed, phase),
            "preflight_key": preflight_key(track, candidate, trajectory, seed),
            "origins": list(origins), "paths": int(paths), "path_batch": self.cfg["path_batch"],
            "forecast_seed": self.cfg["forecast_seed"] if forecast_seed is None else int(forecast_seed),
            "parameter_seed": parameter_seed, "history": history,
            "horizon": self.cfg["horizon"], "lengths": self.cfg["history_candidates"],
            "retain": bool(retain), "partition": "validation", "budget": self.cfg["budget"]}
        job["job_id"] = _job_identity(job)[:24]
        return job

    def consumed(self, stages=None):
        total = 0.
        paths = (self.output.glob("**/attempts/*.json") if stages is None or set(stages) == SEARCH_STAGES
                 else (self.output/"attempts").glob("*.json"))
        for path in paths:
            row = read_json(path)
            if stages is not None and row["stage"] not in stages: continue
            elapsed = float(row.get("elapsed_seconds", 0.))
            if row.get("state") == "running":
                # A crash cannot reset spent worker time; retained heartbeat is
                # a lower bound, and dead attempts are settled during resume.
                elapsed = max(elapsed, time.time()-row["started_epoch"] if process_matches(row.get("process")) else elapsed)
            total += elapsed
        preparation = float(self.cfg["budget"].get("preparation_worker_seconds", 0.))
        carry = self.cfg["budget"].get("carried_worker_seconds", {})
        if stages is None:
            return total + preparation + float(carry.get("total", 0.))
        if set(stages) == SEARCH_STAGES:
            return total + preparation + float(carry.get("search", 0.))
        return total

    def budget_reason(self, stage):
        budget = self.cfg["budget"]
        start = datetime.fromisoformat(budget.get("global_started_utc") or
            read_json(self.output/"experiment.json")["created_utc"]).timestamp()
        if time.time()-start >= budget["wall_seconds_cap"]: return "finite_elapsed_time_budget"
        if self.consumed() >= budget["total_worker_seconds_cap"]: return "finite_worker_time_budget"
        cap = budget.get("stage_worker_seconds_caps", {}).get(stage)
        if cap is not None and self.consumed({stage}) >= cap:
            return "finite_"+stage+"_worker_budget"
        if stage in SEARCH_STAGES and self.consumed(SEARCH_STAGES) >= budget["search_worker_seconds_cap"]:
            return "finite_search_budget_with_final_fit_reserve"
        return None

    def _timeout(self, job):
        budget = self.cfg["budget"]
        return budget.get(job["stage"]+"_job_timeout_seconds", budget["job_timeout_seconds"])

    def _resource_block(self, job, reason, **values):
        result = {"job_id": job["job_id"], "state": "resource-blocked",
            "job_identity": _job_identity(job), "finished_utc": now(), "reason": reason,
            "scientific_failure": False, **values}
        write_once(self.output/"terminal"/f"{job['job_id']}.json", result)
        self.ledger.write(job["job_id"], "resource-blocked", job=job, reason=reason)
        return result

    def submit(self, job):
        previous = terminal_result(self.output, job)
        if previous is not None: return previous
        # A coordinator crash may leave its detached worker alive. Never admit
        # another expensive process while any bound attempt still matches.
        settle_attempts(self.output)
        reason = self.budget_reason(job["stage"])
        if reason:
            self.ledger.write(job["job_id"], "pending", job=job, reason=reason)
            return {"state": "pending", "reason": reason}
        path = self.output/"jobs"/f"{job['job_id']}.json"; write_once(path, job)
        budget = self.cfg["budget"]
        estimate = max(job["candidate"].get("cost", {}).get("estimated_peak_bytes", 0),
                       budget["estimated_worker_peak_bytes"])
        if job["stage"] in {"anchors", "anchor_confirm"}: estimate = min(estimate, 512*1024**2)
        cost = job["candidate"].get("cost", {})
        additional_disk = 2*int(cost.get("estimated_fit_storage_bytes", 0)) + int(
            cost.get("coefficient_diagnostic_storage_bytes", 0)) + int(
            job["candidate"]["config"].get("max_parameter_cache_disk_bytes", 0))
        if job["stage"] in {"anchors", "anchor_confirm", "preflight"}: additional_disk = 32*1024**2
        if resource_snapshot(self.output)["disk_free_bytes"] - additional_disk < budget["disk_reserve_bytes"]:
            self.ledger.write(job["job_id"], "pending", job=job, reason="disk_allocation_reserve",
                estimated_additional_bytes=additional_disk)
            return {"state": "pending", "reason": "disk_allocation_reserve"}
        try:
            admission(self.output, reserve_bytes=budget["disk_reserve_bytes"],
                      estimated_peak_bytes=estimate,
                      worker_cap_bytes=budget["combined_worker_cap_bytes"])
        except RuntimeError as error:
            # A temporary host/container condition keeps the job pending, not
            # scientifically failed. Its static allocation may be infeasible.
            if 1.5*estimate > budget["combined_worker_cap_bytes"]:
                return self._resource_block(job, "static_allocation_exceeds_declared_cap", estimated_peak_bytes=estimate)
            self.ledger.write(job["job_id"], "pending", job=job, reason="resource_admission", error=str(error))
            return {"state": "pending", "reason": "resource_admission"}
        env = dict(os.environ); env.update({name: "1" for name in THREAD_ENV})
        env.update(PYTHONHASHSEED="0", MPLBACKEND="Agg")
        log = self.output/"logs"/f"{job['job_id']}.log"; log.parent.mkdir(parents=True, exist_ok=True)
        begin = time.monotonic(); stopped = None
        before_memory = resource_snapshot(self.output)["memory"]
        before_current = int(before_memory["memory.current"]) if before_memory["memory.current"] else 0
        attempt_path = self.output/"attempts"/f"{job['job_id']}_{time.time_ns()}.json"
        print(json.dumps({"event": "start", "stage": job["stage"], "track": job["track"],
            "candidate": job["candidate"]["id"], "trajectory": job["trajectory"]}), flush=True)
        self.ledger.write(job["job_id"], "pending", job=job)
        with log.open("a") as stream:
            proc = subprocess.Popen([sys.executable, "-m", MODULE, "worker", "--job", str(path)],
                env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            child = child_process_identity(proc.pid)
            if child is None:
                proc.terminate(); proc.wait()
                raise RuntimeError("Cannot verify the numerical worker's process namespace identity")
            attempt = {"job_id": job["job_id"], "stage": job["stage"], "state": "running",
                "started_utc": now(), "started_epoch": time.time(), "process": child}
            atomic_json(attempt_path, attempt); last = -float("inf")
            last_stdout = begin; peak_rss = 0; peak_task_memory = 0; last_resource = None
            try:
                while proc.poll() is None:
                    elapsed = time.monotonic()-begin
                    if elapsed-last >= budget["heartbeat_seconds"]:
                        snap = resource_snapshot(self.output); mem = snap["memory"]
                        rss = process_rss(child["pid"])
                        task_memory = max(rss, int(mem["memory.current"] or 0)-before_current)
                        peak_rss = max(peak_rss, rss); peak_task_memory = max(peak_task_memory, task_memory)
                        last_resource = snap
                        if elapsed >= self._timeout(job): stopped = "per_job_time_budget"
                        elif snap["disk_free_bytes"] < budget["disk_reserve_bytes"]: stopped = "disk_reserve"
                        elif task_memory > budget["combined_worker_cap_bytes"]: stopped = "combined_task_memory_cap"
                        elif mem["memory.max"] not in (None, "max") and (
                            int(mem["memory.max"])-int(mem["memory.current"]) < budget["minimum_container_headroom_bytes"]):
                            stopped = "container_headroom"
                        elif self.budget_reason(job["stage"]): stopped = self.budget_reason(job["stage"])
                        atomic_json(attempt_path, {**attempt, "elapsed_seconds": elapsed,
                            "worker_rss_bytes": rss, "task_memory_upper_estimate_bytes": task_memory,
                            "resource": snap})
                        summarize_output(self.output, write=True, quiet=True)
                        if time.monotonic()-last_stdout >= 60:
                            print(json.dumps({"event": "heartbeat", "stage": job["stage"],
                                "candidate": job["candidate"]["id"], "elapsed_seconds": round(elapsed, 1)}), flush=True)
                            last_stdout = time.monotonic()
                        last = elapsed
                        if stopped: break
                    time.sleep(.25)
            except BaseException:
                stopped = "coordinator_interrupted"
                raise
            finally:
                if proc.poll() is None:
                    proc.send_signal(signal.SIGINT)
                    try: proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.terminate()
                        try: proc.wait(timeout=10)
                        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                elapsed = time.monotonic()-begin
                eviction = evict_closed_artifact_pages(self.output/"fits"/job["fit_key"]) if budget.get(
                    "evict_closed_task_artifact_pages", True) else {"enabled": False}
                atomic_json(attempt_path, {**attempt,
                    "state": "complete" if proc.returncode == 0 else "interrupted",
                    "returncode": proc.returncode, "reason": stopped,
                    "elapsed_seconds": elapsed, "finished_utc": now(), "page_cache_advice": eviction,
                    "monitored_peak_rss_bytes": peak_rss,
                    "monitored_peak_task_memory_upper_estimate_bytes": peak_task_memory,
                    "last_monitored_resource": last_resource})
        result = terminal_result(self.output, job)
        if result is None and stopped in {"per_job_time_budget", "combined_task_memory_cap"}:
            result = self._resource_block(job, stopped, elapsed_seconds=elapsed,
                                         checkpoint_retained=True)
        if result is None:
            self.ledger.write(job["job_id"], "pending", job=job,
                reason=stopped or "worker_execution_error", elapsed_seconds=elapsed)
            summarize_output(self.output, write=True, quiet=True)
            if stopped:
                return {"state": "pending", "reason": stopped}
            raise RuntimeError(f"Worker exited without a bound terminal receipt; inspect {log}")
        print(json.dumps({"event": "end", "stage": job["stage"], "track": job["track"],
            "candidate": job["candidate"]["id"], "trajectory": job["trajectory"],
            "state": result["state"], "seconds": elapsed}), flush=True)
        summarize_output(self.output, write=True, quiet=True)
        return result

    def batch(self, jobs):
        jobs = list(jobs)
        # Register the entire finite stage before execution, so status never
        # hides unscheduled members when a resource/budget stop occurs.
        for job in jobs:
            write_once(self.output/"jobs"/f"{job['job_id']}.json", job)
            if terminal_result(self.output, job) is None and self.ledger.read(job["job_id"]) is None:
                self.ledger.write(job["job_id"], "pending", job=job, reason="registered")
        for job in jobs:
            if self.submit(job)["state"] == "pending": return False
        return True

    def pilot(self, labels=None, tracks=None):
        from .ks_fullfield_native import resolve_fullfield_config
        policy = self.cfg["pilot"]; jobs = []
        for track in tracks or self.cfg["tracks"]:
            for original in self.candidates(track):
                if original["label"] not in (labels or policy["labels"]): continue
                candidate = copy.deepcopy(original); cfg = candidate["config"]
                cfg["sampler"].update(chains=policy["chains"], warmup=policy["warmup"], draws=policy["draws"],
                    init_maxiter=policy["init_maxiter"], init_max_evaluations=policy["init_max_evaluations"],
                    checkpoint_every=16)
                cfg["diagnostics"]["functional_draws"] = policy["functional_draws"]
                cfg["continuation_draws"] = 0
                candidate["config"] = resolve_fullfield_config(cfg, seed=self.cfg["model_seed"])
                candidate["active_config"] = active_configuration(candidate["config"], SITES[track])
                candidate["effective_hash"] = content_hash(candidate["active_config"])
                candidate["cost"] = estimate_cost(candidate["config"], SITES[track], training_rows=1200)
                candidate["id"] = original["id"]+"_resource_pilot"
                trajectories = policy["reference_trajectories"] if original["id"] == self.registry["tracks"][track]["reference_id"] else policy["other_trajectories"]
                for trajectory in trajectories:
                    job = self.job("pilot", track, candidate, trajectory, phase="pilot", paths=policy["paths"])
                    job["history"] = policy["history"]; job["original_candidate_id"] = original["id"]
                    job["job_id"] = _job_identity(job)[:24]; jobs.append(job)
        if not jobs: raise ValueError("No declared pilot candidates match")
        result = self.batch(jobs)
        atomic_json(self.output/"pilot_summary.json", {"experiment_identity": self.identity,
            "complete_schedule": result, "jobs": [{"job_id": j["job_id"], "candidate_id": j["original_candidate_id"],
                "track": j["track"], "trajectory": j["trajectory"],
                "result": terminal_result(self.output, j)} for j in jobs],
            "interpretation": "Resource and execution pilots only; no prediction-quality selection", "updated_utc": now()})
        return result

    def preflight(self):
        if (self.output/"history_policy.json").exists(): return True
        jobs = {}
        for track in self.cfg["tracks"]:
            for candidate in self.candidates(track):
                for trajectory in self.cfg["screen"]["trajectories"]:
                    job = self.job("preflight", track, candidate, trajectory)
                    jobs.setdefault(job["preflight_key"], job)
        if not self.batch(jobs.values()): return False
        excluded, usable = {}, []
        common = set(self.cfg["history_candidates"])
        for track in self.cfg["tracks"]:
            for candidate in self.candidates(track):
                passing = set(self.cfg["history_candidates"])
                for trajectory in self.cfg["screen"]["trajectories"]:
                    key = preflight_key(track, candidate, trajectory, self.cfg["model_seed"])
                    result = terminal_result(self.output, jobs[key])
                    if result is None: raise RuntimeError("Incomplete preflight must stay pending")
                    if result["state"] != "complete":
                        excluded[track+"/"+candidate["id"]] = {"reason": "training_preflight_"+result["state"],
                            "job_id": jobs[key]["job_id"]}; passing.clear(); break
                    passing &= {r["history"] for r in result["result"]["synchronization"]["rows"] if r.get("passed")}
                if not passing:
                    excluded.setdefault(track+"/"+candidate["id"], {"reason": "no_admissible_training_synchronization"})
                else:
                    common &= passing; usable.append(track+"/"+candidate["id"])
        if not usable or not common:
            raise RuntimeError("No common measured history among admissible candidates; no scientific screen launched")
        atomic_json(self.output/"history_policy.json", {"history": min(common),
            "history_candidates": self.cfg["history_candidates"], "excluded_candidates": excluded,
            "admissible_candidates": usable, "unique_preflight_jobs": len(jobs),
            "created_utc": now(), "selection_values_used": False,
            "basis": "common training-only state/feature synchronization across all retained tracks/candidates"})
        return True

    def anchors(self, confirmation=False):
        if not self.preflight(): return False
        jobs = []
        schedule = self.cfg["confirmation" if confirmation else "promote"]
        stage = "anchor_confirm" if confirmation else "anchors"
        for track in self.cfg["tracks"]:
            ref = next(c for c in self.candidates(track) if c["id"] == self.registry["tracks"][track]["reference_id"])
            for trajectory in schedule["trajectories"]:
                jobs.append(self.job(stage, track, ref, trajectory, schedule["origins"], paths=0))
        return self.batch(jobs)

    def partial_stage_allowed(self, stage):
        return (self.cfg["budget"].get("partial_roster_at_stage_cap", False)
                and self.budget_reason(stage) == "finite_"+stage+"_worker_budget")

    def ledger_fit_index(self):
        result = {}
        for path in (self.output/"ledger").glob("*.json"):
            row = read_json(path); job = row.get("job")
            if not job: continue
            terminal = terminal_result(self.output, job)
            if terminal is not None: row["state"] = terminal["state"]
            result.setdefault(job["fit_key"], []).append(row)
        return result

    def loss(self, track, candidate, schedule, state_index=None, seed=None):
        policy = read_json(self.output/"history_policy.json")
        excluded = policy["excluded_candidates"].get(track+"/"+candidate["id"])
        state_index = self.ledger_fit_index() if state_index is None else state_index
        episodes, failures, pending, unqualified, resources, diverged = [], 0, 0, 0, 0, 0
        resource_origins = 0
        for trajectory in schedule["trajectories"]:
            job = self.job("screen", track, candidate, trajectory, schedule["origins"], paths=schedule["paths"], seed=seed)
            receipt_path = self.output/"fits"/job["fit_key"]/"receipt.json"
            fit = read_json(receipt_path) if receipt_path.exists() else None
            fit_states = state_index.get(job["fit_key"], [])
            if excluded: continue
            if fit is not None and not fit["diagnostics"].get("qualified", False):
                unqualified += 1; continue
            if fit is None:
                terminal_fits = [row["state"] for row in fit_states]
                if "resource-blocked" in terminal_fits:
                    resources += 1; resource_origins += len(schedule["origins"])
                elif "failed" in terminal_fits: failures += 1
                elif "inference-unqualified" in terminal_fits: unqualified += 1
                else: pending += len(schedule["origins"])
                continue
            records = []; resource_blocked_episode = False
            for first in schedule["origins"]:
                identity = _forecast_identity(job, fit["native_fit_identity"], first)
                path = _origin_path(self.output, job, first, identity)
                if not path.exists():
                    outcome_states = [row["state"] for row in fit_states if
                        row["job"].get("paths") == job["paths"] and
                        row["job"].get("history") == job["history"] and
                        row["job"].get("forecast_seed") == job["forecast_seed"] and
                        row["job"].get("parameter_seed") == job["parameter_seed"] and
                        first in row["job"].get("origins", [])]
                    if "resource-blocked" in outcome_states:
                        resource_origins += 1; resource_blocked_episode = True
                    elif "failed" in outcome_states: failures += 1
                    else: pending += 1
                    continue
                record = read_json(path)
                if (record["identity"] != identity or record["native_fit"] != fit["native_fit_identity"]
                    or record["partition"] != "validation" or record["seed"] != job["seed"]
                    or not record.get("qualified") or record["trajectory"] != trajectory):
                    raise ValueError("Incompatible forecast receipt in selection")
                if record["state"] != "complete": failures += 1; continue
                records.append(record["metrics"])
            resources += int(resource_blocked_episode)
            if len(records) == len(schedule["origins"]):
                early = self.cfg["selection"]["early_horizon"]
                episodes.append({"trajectory": trajectory,
                    "selection_crps": float(np.mean([r["selection_crps"] for r in records])),
                    "selection_mse": float(np.mean([r["selection_mse"] for r in records])),
                    "early_rmse": float(np.sqrt(np.mean([r["mse_by_horizon"][:early] for r in records]))),
                    "lead1_rmse": float(np.sqrt(np.mean([r["mse_by_horizon"][0] for r in records]))),
                    "rmse": float(np.sqrt(np.mean([r["mse_by_horizon"] for r in records]))),
                    "vpt_physical": float(np.mean([r["vpt_physical"] for r in records])),
                    "crps": float(np.mean([r["crps"] for r in records])),
                    "divergent_origins": sum(r["diverged"] for r in records)})
                diverged += episodes[-1]["divergent_origins"]
        complete = len(episodes) == len(schedule["trajectories"]) and not (failures or pending or unqualified or resources or excluded)
        eligible = complete and diverged <= self.cfg["selection"]["maximum_severe_divergent_origins"]
        def average(key): return float(np.mean([e[key] for e in episodes])) if complete else None
        return {"id": candidate["id"], "label": candidate["label"], "stratum": candidate["stratum"],
            "prior_family": candidate["prior_family"], "eligible": bool(eligible), "complete": bool(complete),
            "loss": average("selection_crps"), "point_loss": average("selection_mse"),
            "early_rmse": average("early_rmse"), "lead1_rmse": average("lead1_rmse"),
            "vpt_physical": average("vpt_physical"), "rmse": average("rmse"), "crps": average("crps"),
            "expected_episodes": len(schedule["trajectories"]), "complete_episodes": len(episodes),
            "failed_outcomes": failures, "pending_origins": pending,
            "inference_unqualified_fits": unqualified, "resource_blocked_fits": resources,
            "resource_blocked_origins": resource_origins,
            "severe_divergent_origins": diverged, "preflight_exclusion": excluded,
            "episodes": episodes, "configuration": candidate["config"],
            "effective_hash": candidate["effective_hash"], "complexity": candidate["cost"],
            "matched_ridge_id": candidate.get("matched_ridge_id")}

    def rankings(self, stage, track, candidates, schedule):
        state_index = self.ledger_fit_index()
        rows = [self.loss(track, c, schedule, state_index) for c in candidates]
        rows.sort(key=lambda r: (not r["eligible"], r["loss"] if r["loss"] is not None else float("inf"), r["id"]))
        atomic_json(self.output/"selection"/f"{stage}_{track}.json", {"stage": stage, "track": track,
            "schedule": schedule, "rows": rows, "pareto_ids": selection_frontier(rows),
            "selection_policy": self.cfg["selection"], "created_utc": now()})
        return rows

    def screen(self):
        if not self.preflight(): return False
        if not self.anchors(): return False
        policy = read_json(self.output/"history_policy.json")
        schedule = self.cfg["screen"]; jobs = []
        queues = {track: ordered_candidates(self.candidates(track)) for track in self.cfg["tracks"]}
        for index in range(max(map(len, queues.values()))):
            for track in self.cfg["tracks"]:
                if index >= len(queues[track]): continue
                candidate = queues[track][index]
                if track+"/"+candidate["id"] in policy["excluded_candidates"]: continue
                for trajectory in schedule["trajectories"]:
                    jobs.append(self.job("screen", track, candidate, trajectory,
                                         schedule["origins"], paths=schedule["paths"]))
        complete = self.batch(jobs)
        coverage = {}
        for track in self.cfg["tracks"]:
            rows = self.rankings("screen", track, self.candidates(track), schedule)
            coverage[track] = {"proposed_candidates": len(rows),
                "eligible_candidates": sum(r["eligible"] for r in rows),
                "complete_candidate_outcomes": sum(r["complete"] for r in rows),
                "preflight_exclusions": sum(bool(r["preflight_exclusion"]) for r in rows),
                "pending_origins": sum(r["pending_origins"] for r in rows)}
        atomic_json(self.output/"screen_schedule.json", {"complete_terminal_schedule": complete,
            "partial_roster": not complete, "stop_reason": self.budget_reason("screen") if not complete else None,
            "promotion_from_completed_candidates_allowed": complete or self.partial_stage_allowed("screen"),
            "coverage": coverage, "candidate_order": {t: [c["id"] for c in queue] for t, queue in queues.items()},
            "rule": "Response-independent candidate bundles; all unfinished jobs remain pending; "
                "only qualified complete two-trajectory candidates are eligible", "updated_utc": now()})
        return complete

    def promote(self):
        if not self.screen() and not self.partial_stage_allowed("screen"): return False
        schedule = self.cfg["promote"]; promoted = {}; jobs = []
        for track in self.cfg["tracks"]:
            rows = read_json(self.output/"selection"/f"screen_{track}.json")["rows"]
            ids = promotion_ids(rows, self.registry["tracks"][track]["reference_id"],
                                schedule["count"], schedule["matched_prior_extra"])
            promoted[track] = [c for c in self.candidates(track) if c["id"] in ids]
            for candidate in promoted[track]:
                for trajectory in schedule["trajectories"]:
                    jobs.append(self.job("promote", track, candidate, trajectory,
                                         schedule["origins"], paths=schedule["paths"]))
        complete = self.batch(jobs)
        for track in self.cfg["tracks"]: self.rankings("promote", track, promoted[track], schedule)
        atomic_json(self.output/"promotion_schedule.json", {"complete_terminal_schedule": complete,
            "partial_roster": not complete, "stop_reason": self.budget_reason("promote") if not complete else None,
            "promoted_ids": {t: [c["id"] for c in rows] for t, rows in promoted.items()},
            "updated_utc": now()})
        return complete

    def precision(self):
        if not self.promote() and not self.partial_stage_allowed("promote"): return False
        policy = self.cfg["precision"]
        if not policy["enabled"]: return True
        jobs, selected = [], {}
        for track in self.cfg["tracks"]:
            rows = read_json(self.output/"selection"/f"promote_{track}.json")["rows"]
            eligible = {r["id"] for r in rows if r["eligible"]}
            selected[track] = [c for c in self.candidates(track) if c["id"] in eligible]
            for candidate in selected[track]:
                for repetition in range(policy["replicates"]):
                    job = self.job("precision", track, candidate, policy["trajectory"], policy["origins"],
                        paths=policy["paths"], forecast_seed=seed_for(self.cfg["forecast_seed"], "precision", repetition),
                        parameter_seed=seed_for(self.cfg["forecast_seed"], "parameter_bank", repetition))
                    jobs.append(job)
        complete = self.batch(jobs)
        partial_allowed = self.partial_stage_allowed("precision") or (
            self.cfg["budget"].get("partial_roster_at_stage_cap", False) and
            self.budget_reason("precision") == "finite_search_budget_with_final_fit_reserve")
        if not complete and not partial_allowed: return False
        for track in self.cfg["tracks"]:
            values, excluded = {}, {}
            for candidate in selected[track]:
                repeats, missing = [], []
                for repetition in range(policy["replicates"]):
                    job = self.job("precision", track, candidate, policy["trajectory"], policy["origins"],
                        paths=policy["paths"], forecast_seed=seed_for(self.cfg["forecast_seed"], "precision", repetition),
                        parameter_seed=seed_for(self.cfg["forecast_seed"], "parameter_bank", repetition))
                    fit = read_json(self.output/"fits"/job["fit_key"]/"receipt.json")
                    losses = []
                    for first in policy["origins"]:
                        identity = _forecast_identity(job, fit["native_fit_identity"], first)
                        path = _origin_path(self.output, job, first, identity)
                        record = read_json(path) if path.exists() else None
                        if record is None or record["state"] != "complete":
                            terminal = terminal_result(self.output, job)
                            missing.append({"repetition": repetition, "first_target": first,
                                "state": record["state"] if record else terminal["state"] if terminal else "pending"})
                            continue
                        if record["identity"] != identity:
                            raise ValueError("Predictive precision receipt identity changed")
                        losses.append(record["metrics"]["selection_crps"])
                    if len(losses) == len(policy["origins"]): repeats.append(float(np.mean(losses)))
                if missing:
                    excluded[candidate["id"]] = {"reason": "incomplete_declared_predictive_precision",
                        "missing_cases": missing, "complete_repetitions": len(repeats)}
                else: values[candidate["id"]] = repeats
            tolerances = {a: {b: self.cfg["selection"]["mc_sd_multiplier"]*float(np.std(
                np.asarray(values[a])-np.asarray(values[b]), ddof=1)) for b in values} for a in values}
            atomic_json(self.output/"selection"/f"precision_{track}.json", {
                "track": track, "policy": policy, "replicate_losses": values,
                "pairwise_absolute_tolerances": tolerances,
                "complete_terminal_schedule": complete, "partial_schedule": bool(excluded),
                "precision_eligible_ids": sorted(values), "excluded_candidates": excluded,
                "stop_reason": self.budget_reason("precision") if not complete else None,
                "interpretation": "Conditional predictive Monte Carlo precision from independent parameter banks and "
                    "innovation repetitions on fixed validation cases; not scientific uncertainty or a convergence test",
                "created_utc": now()})
        return True

    def freeze(self):
        path = self.output/"freeze.json"
        if path.exists():
            record = read_json(path)
            if record["experiment_identity"] != self.identity or record["identity"] != content_hash(
                    {k: v for k, v in record.items() if k != "identity"}):
                raise ValueError("Selection freeze identity differs")
            return record
        if not self.precision(): return None
        selections = {}
        for track in self.cfg["tracks"]:
            rows = read_json(self.output/"selection"/f"promote_{track}.json")["rows"]
            precision_path = self.output/"selection"/f"precision_{track}.json"
            precision = read_json(precision_path) if precision_path.exists() else None
            tolerances = precision["pairwise_absolute_tolerances"] if precision else {}
            if self.cfg["precision"]["enabled"]:
                if precision is None: raise RuntimeError("Predictive precision gate has no bound receipt")
                for row in rows:
                    row["pre_precision_eligible"] = row["eligible"]
                    row["precision_eligible"] = row["id"] in precision["precision_eligible_ids"]
                    row["precision_exclusion"] = precision["excluded_candidates"].get(row["id"])
                    row["eligible"] = row["eligible"] and row["precision_eligible"]
            selections[track] = choose_selections(rows, self.cfg["selection"], tolerances)
            selections[track]["reference_id"] = self.registry["tracks"][track]["reference_id"]
            selections[track]["precision_qualification"] = precision
        result = {"experiment_identity": self.identity, "selections": selections,
            "history": read_json(self.output/"history_policy.json")["history"],
            "confirmation": self.cfg["confirmation"], "policy": self.cfg["selection"],
            "created_utc": now(), "test_scoring": False,
            "screen_schedule": read_json(self.output/"screen_schedule.json"),
            "promotion_schedule": read_json(self.output/"promotion_schedule.json"),
            "interpretation": self.cfg["interpretation"]}
        result["identity"] = content_hash(result)
        write_once(path, result); return result

    def confirm(self):
        freeze = self.freeze()
        if freeze is None: return False
        if not self.anchors(confirmation=True): return False
        schedule = self.cfg["confirmation"]; jobs = []; selected = {}
        for track in self.cfg["tracks"]:
            result = freeze["selections"][track]
            ids = {result[mode]["candidate"]["id"] for mode in self.cfg["selection"]["modes"] if result[mode]}
            ids.add(result["reference_id"])
            selected[track] = [c for c in self.candidates(track) if c["id"] in ids]
            for candidate in selected[track]:
                for trajectory in schedule["trajectories"]:
                    jobs.append(self.job("confirm", track, candidate, trajectory, schedule["origins"],
                        paths=schedule["paths"], retain=trajectory == schedule["trajectories"][0]))
        complete = self.batch(jobs)
        for track in self.cfg["tracks"]:
            rows = self.rankings("confirmation", track, selected[track], schedule)
            for row in rows:
                row["uncertainty"] = {key: cluster_summary([e[key] for e in row["episodes"]])
                    for key in ("rmse", "crps", "early_rmse", "vpt_physical")}
            atomic_json(self.output/"confirmation"/f"{track}.json", {"rows": rows,
                "schedule": schedule, "freeze_identity": freeze["identity"],
                "population": "complete-trajectory summaries with failures separately counted; no test claims"})
        return complete

    def sensitivity(self):
        freeze = self.freeze()
        if freeze is None: return False
        schedule = self.cfg["sensitivity"]; jobs = []
        for track in self.cfg["tracks"]:
            result = freeze["selections"][track]
            ids = {result[mode]["candidate"]["id"] for mode in self.cfg["selection"]["modes"] if result[mode]}
            ids.add(result["reference_id"])
            for candidate in self.candidates(track):
                if candidate["id"] not in ids: continue
                # Include the base seed on the same trajectories and origins.
                for seed in [self.cfg["model_seed"], *schedule["seeds"]]:
                    for trajectory in schedule["trajectories"]:
                        jobs.append(self.job("sensitivity", track, candidate, trajectory, schedule["origins"],
                            paths=schedule["paths"], seed=seed))
        complete = self.batch(jobs)
        index = self.ledger_fit_index()
        for track in self.cfg["tracks"]:
            result = freeze["selections"][track]
            ids = {result[mode]["candidate"]["id"] for mode in self.cfg["selection"]["modes"] if result[mode]}
            ids.add(result["reference_id"])
            rows = []
            for candidate in self.candidates(track):
                if candidate["id"] not in ids: continue
                for seed in [self.cfg["model_seed"], *schedule["seeds"]]:
                    row = self.loss(track, candidate, schedule, index, seed=seed)
                    row["model_seed"] = seed; rows.append(row)
            atomic_json(self.output/"sensitivity"/f"{track}.json", {"rows": rows,
                "schedule": schedule, "freeze_identity": freeze["identity"],
                "interpretation": "Separate model-seed fits; posterior paths are never pooled across seeds."})
        return complete

    def deferred(self):
        """Run frozen robustness proposals after the main campaign, if budget remains.

        The child configuration carries the original elapsed deadline and all
        prior worker/search charges. It is written once, so restart cannot buy
        a new budget. C has its own transforms, fits, histories and selections.
        """
        if self.cfg.get("deferred_tracks"):
            main_path = self.output/"main_completion.json"
            if not main_path.exists() or read_json(main_path).get("experiment_identity") != self.identity:
                raise RuntimeError("Deferred robustness requires this experiment's verified main_completion.json")
        for track in self.cfg.get("deferred_tracks", []):
            folder = self.output/"deferred"/track; folder.mkdir(parents=True, exist_ok=True)
            config_path = folder/"configuration.json"
            if not config_path.exists():
                cfg = copy.deepcopy(self.cfg)
                cfg.update(tracks=[track], deferred_tracks=[], parent_experiment_identity=self.identity)
                budget = cfg["budget"]
                budget["carried_worker_seconds"] = {"total": self.consumed(),
                    "search": self.consumed(SEARCH_STAGES)}
                budget["preparation_worker_seconds"] = 0
                budget["global_started_utc"] = self.cfg["budget"].get("global_started_utc") or read_json(
                    self.output/"experiment.json")["created_utc"]
                remaining = max(0., budget["search_worker_seconds_cap"]-budget["carried_worker_seconds"]["search"])
                if remaining > 0:
                    preflight, anchors = min(900., .1*remaining), min(300., .05*remaining)
                    after = remaining-preflight-anchors
                    budget["stage_worker_seconds_caps"].update(preflight=preflight, anchors=anchors,
                        screen=.6*after, promote=.3*after, precision=.1*after)
                budget["carried_charge_interpretation"] = "Already spent main/preparation work; original global caps and elapsed start retained."
                write_once(config_path, cfg)
            with (folder/"scheduler.lock").open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                child = Campaign(config_path, self.cache, folder)
                record = {"process": process_identity(), "state": "running", "command": "deferred",
                    "parent_experiment_identity": self.identity, "experiment_identity": child.identity,
                    "started_utc": now()}
                atomic_json(folder/"coordinator.json", record)
                complete = False
                try:
                    complete = child.confirm()
                    if complete: complete = child.sensitivity()
                finally:
                    atomic_json(folder/"coordinator.json", {**record,
                        "state": "complete" if complete else "paused", "finished_utc": now()})
                    summarize_output(folder, write=True, quiet=True)
                    from .ks_fullfield_report import report
                    report(folder)
                if not complete: return False
        return True


def summarize_output(output, *, write=False, quiet=False):
    """Read-only status works without data access or loading the model runtime."""
    root = Path(output); experiment = read_json(root/"experiment.json")
    states, stages, outcomes = {}, {}, []
    for path in sorted((root/"jobs").glob("*.json")):
        job = read_json(path); ledger_path = root/"ledger"/(job["job_id"]+".json")
        record = terminal_result(root, job) or (read_json(ledger_path) if ledger_path.exists() else {"state": "pending"})
        state = record["state"]
        if state == "running" and not process_matches(record.get("process")): state = "pending"
        states[state] = states.get(state, 0)+1
        stages.setdefault(job["stage"], {})[state] = stages.setdefault(job["stage"], {}).get(state, 0)+1
    attempts = [read_json(p) for p in root.glob("**/attempts/*.json")]
    worker_seconds = sum(float(a.get("elapsed_seconds", 0.)) for a in attempts)
    search_seconds = sum(float(a.get("elapsed_seconds", 0.)) for a in attempts if a["stage"] in SEARCH_STAGES)
    coordinator_path = root/"coordinator.json"
    coordinator = read_json(coordinator_path) if coordinator_path.exists() else {}
    active = process_matches(coordinator.get("process")) and coordinator.get("state") == "running"
    budget = experiment["config"]["budget"]; carry = budget.get("carried_worker_seconds", {})
    preparation = float(budget.get("preparation_worker_seconds", 0.))
    result = {"experiment_identity": experiment["identity"], "states": states, "stages": stages,
        "worker_seconds": worker_seconds + preparation + float(carry.get("total", 0.)),
        "search_worker_seconds": search_seconds + preparation + float(carry.get("search", 0.)),
        "coordinator_active": active, "coordinator": coordinator,
        "screening_frozen": (root/"freeze.json").exists(),
        "registered_jobs": sum(states.values()), "updated_utc": now(),
        "resources": resource_snapshot(root), "test_scoring": False,
        "main_tracks": experiment["config"]["tracks"],
        "deferred_tracks": {track: {"output": str(root/"deferred"/track),
            "status": read_json(root/"deferred"/track/"status.json") if (root/"deferred"/track/"status.json").exists()
                      else "deferred_until_main_completion"}
            for track in experiment["config"].get("deferred_tracks", [])},
        "interpretation": "No scores are implied by queued jobs; resource/inference/forecast failures remain distinct."}
    if write: atomic_json(root/"status.json", result)
    if not quiet: print(json.dumps(result, indent=2))
    return result


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}; preserving checkpoints")
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("config", "worker", "pilot", "preflight", "screen",
        "promote", "precision", "freeze", "confirm", "sensitivity", "deferred", "all", "status"))
    parser.add_argument("--config"); parser.add_argument("--cache"); parser.add_argument("--output")
    parser.add_argument("--job"); parser.add_argument("--pilot-label", action="append")
    parser.add_argument("--pilot-track", action="append", choices=list(SITES))
    args = parser.parse_args()
    if args.command == "config":
        if not args.output: parser.error("--output is required")
        atomic_json(args.output, default_configuration()); return
    if args.command == "worker":
        if not args.job: parser.error("--job is required")
        worker(args.job); return
    if args.command == "status":
        if not args.output: parser.error("--output is required")
        summarize_output(args.output); return
    if not all((args.config, args.cache, args.output)):
        parser.error("--config, --cache and --output are required")
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    with (output/"scheduler.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign = Campaign(args.config, args.cache, output, allow_unfrozen=args.command == "pilot")
        record = {"process": process_identity(), "state": "running", "command": args.command,
            "started_utc": now(), "experiment_identity": campaign.identity}
        atomic_json(output/"coordinator.json", record)
        try:
            if args.command == "pilot": complete = campaign.pilot(args.pilot_label, args.pilot_track)
            elif args.command == "all":
                complete = campaign.confirm()
                if complete: complete = campaign.sensitivity()
                if complete:
                    atomic_json(output/"main_completion.json", {"experiment_identity": campaign.identity,
                        "main_tracks": campaign.cfg["tracks"], "completed_utc": now(),
                        "interpretation": "Scheduled main follow-up complete; any unfinished broad proposals remain explicitly pending."})
                    complete = campaign.deferred()
            elif args.command == "freeze": complete = campaign.freeze() is not None
            else: complete = getattr(campaign, args.command)()
            atomic_json(output/"coordinator.json", {**record,
                "state": "complete" if complete else "paused", "finished_utc": now()})
        except BaseException as error:
            atomic_json(output/"coordinator.json", {**record, "state": "interrupted",
                "error": repr(error), "finished_utc": now()})
            raise
        finally:
            summarize_output(output, write=True, quiet=True)
            from .ks_fullfield_report import report
            report(output)


if __name__ == "__main__":
    main()
