#!/usr/bin/env python3
"""Prepare or execute fixed point anchors on the improvement confirmation panel.

Without --execute this writes only a bound plan, without opening the data cache
or fitting any model. Execution requires the campaign's completed selection
freeze, uses one worker/thread, and checks the unchanged resource admission gate.
No new selection, test-partition labels, probabilistic intervals, or posterior
model averaging is introduced.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from bayes_stesn.io import content_hash
from bayes_stesn.applications.ks_baselines import VectorFit, fit_vector
from bayes_stesn.applications.ks_data import KSAdapter
from bayes_stesn.applications.ks_metrics import score
from bayes_stesn.applications.ks_runtime import admission, resource_snapshot

from build_ks_improvement_report import CONFIRMATION, ORIGINS, atomic_bytes, sha256, utc, write_json


MODELS = ("training_mean", "persistence", "linear_var")
AVAILABLE_MODELS = MODELS + ("ridge_esn", "pod_nvar")
PROTOCOLS = {"A": "interleaved", "B": "all_observed", "C": "contiguous"}


def read(path):
    return json.loads(Path(path).read_text())


def build_plan(results, original_freeze, repository, models=MODELS):
    models = tuple(dict.fromkeys(models))
    if not models or any(model not in AVAILABLE_MODELS for model in models):
        raise ValueError("Unknown fixed-reference model")
    experiment = read(results / "experiment.json")
    expected = content_hash({"config": experiment["config"], "source": experiment["source"],
                             "cache": experiment["cache_identity"]})
    if expected != experiment["identity"]:
        raise ValueError("Campaign experiment identity mismatch")
    old = read(original_freeze)
    if (old["identity"] != content_hash({k: v for k, v in old.items() if k != "identity"})
            or old["identity"] != experiment["config"]["registry"]["old_freeze_identity"]):
        raise ValueError("Original selected-config freeze mismatch")
    names = ("applications/ks_baselines.py", "applications/ks_data.py", "applications/ks_metrics.py",
             "applications/ks_runtime.py", "io.py", "scoring.py", "data.py")
    sources = {name: sha256(repository / "src/bayes_stesn" / name) for name in names}
    if any(old["source"].get(name) != value for name, value in sources.items()):
        raise ValueError("An original anchor, reader or scoring source differs from its published freeze")
    configurations = {}
    for track in experiment["config"]["tracks"]:
        source_track = "A" if track == "C" else track
        configurations[track] = {model: {"id": old["selected"][source_track][model]["id"],
                                      "config": old["selected"][source_track][model]["config"],
                                      "original_source_track": source_track} for model in models}
    plan = {"schema": "ks-improvement-point-anchors/1", "experiment_identity": expected,
            "original_freeze_identity": old["identity"], "original_freeze_sha256": sha256(original_freeze),
            "tracks": experiment["config"]["tracks"], "trajectories": CONFIRMATION, "origins": ORIGINS,
            "history": 512, "horizon": 100, "partition": "validation", "models": list(models),
            "configurations": configurations, "numerical_sources": sources,
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "h5py")},
            "builder_sha256": sha256(__file__),
            "model_seeds": {model: old["model_seed"] if model == "ridge_esn" else 0 for model in models},
            "seed_meaning": "ridge ESN uses the originally frozen model seed; zero is unused for closed-form deterministic families",
            "interpretation": "fixed original point references; C transfers A's selected point-reference hyperparameters for every included family and refits only permitted C training data",
            "disk_reserve_bytes": experiment["config"]["budget"]["disk_reserve_bytes"],
            "estimated_worker_peak_bytes": 256 * 1024 ** 2, "worker_cap_bytes": 512 * 1024 ** 2}
    plan["identity"] = content_hash(plan)
    return plan


def fit_identity(plan, track, model, trajectory, training_identity):
    return content_hash({"plan": plan["identity"], "track": track, "model": model,
                         "trajectory": trajectory, "training_identity_sha256": content_hash(training_identity)})


def physical_prediction(prediction, transform, shape):
    """Classify numerical divergence explicitly; preserve shape errors as bugs."""
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(transform.inverse(prediction), np.float64)
    if result.shape != shape:
        raise ValueError("Anchor forecast shape differs from the frozen target panel")
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite physical-unit autonomous anchor prediction")
    return result


def run(plan, results, cache, output, maximum_seconds):
    freeze = read(results / "freeze.json")
    if (freeze["identity"] != content_hash({k: v for k, v in freeze.items() if k != "identity"})
            or freeze["experiment_identity"] != plan["experiment_identity"]
            or freeze["confirmation_trajectories"] != plan["trajectories"]
            or freeze["confirmation_origins"] != plan["origins"]
            or freeze["history"] != plan["history"] or freeze["test_access"] is not False):
        raise ValueError("Anchor schedule is incompatible with the current confirmation freeze")
    binding = {"plan_identity": plan["identity"], "experiment_identity": plan["experiment_identity"],
               "freeze_identity": freeze["identity"], "freeze_sha256": sha256(results / "freeze.json")}
    binding_path = output / "binding.json"
    if binding_path.exists() and read(binding_path) != binding:
        raise ValueError("Anchor output root is bound to a different frozen study")
    write_json(binding_path, binding)
    started = time.perf_counter()
    completed_jobs = 0
    for track in plan["tracks"]:
        adapter = KSAdapter(cache, PROTOCOLS[track])
        for trajectory in plan["trajectories"]:
            training = None
            for model in plan["models"]:
                job_id = f"{track}_{model}_t{trajectory:02d}"
                terminal_path = output / "terminal" / f"{job_id}.json"
                if terminal_path.exists():
                    terminal = read(terminal_path)
                    if terminal["plan_identity"] != plan["identity"] or terminal["binding"] != binding:
                        raise ValueError("Incompatible completed anchor job")
                    completed_jobs += 1
                    continue
                if time.perf_counter() - started > maximum_seconds:
                    write_json(output / "status.json", {"state": "incomplete", "reason": "bounded_execution_time_reached",
                        "plan_identity": plan["identity"], "completed_jobs": completed_jobs, "updated_utc": utc()})
                    return False
                admission(output, reserve_bytes=plan["disk_reserve_bytes"],
                          estimated_peak_bytes=plan["estimated_worker_peak_bytes"], worker_cap_bytes=plan["worker_cap_bytes"])
                job_started = time.perf_counter()
                if training is None:
                    training = adapter.training(trajectory)
                config = plan["configurations"][track][model]["config"]
                identity = fit_identity(plan, track, model, trajectory, dict(training.identity))
                folder = output / "fits" / job_id
                folder.mkdir(parents=True, exist_ok=True)
                fit_path, receipt_path = folder / "fit.npz", folder / "receipt.json"
                if receipt_path.exists():
                    receipt = read(receipt_path)
                    if receipt["identity"] != identity or not fit_path.exists() or sha256(fit_path) != receipt["fit_sha256"]:
                        raise ValueError("Anchor checkpoint checksum/identity mismatch")
                    fitted = VectorFit.load(fit_path)
                    if fitted.config != config or fitted.model_id != model or fitted.seed != plan["model_seeds"][model]:
                        raise ValueError("Anchor checkpoint has incompatible effective parameters")
                else:
                    begin = time.perf_counter()
                    try:
                        fitted = fit_vector(model, training.panel, config, seed=plan["model_seeds"][model])
                    except (FloatingPointError, np.linalg.LinAlgError) as error:
                        write_json(terminal_path, {"job_id": job_id, "plan_identity": plan["identity"], "binding": binding,
                            "state": "failed", "reason": "numerical_fit_failure", "error": repr(error),
                            "observed_origins": 0, "blocked_origins": len(plan["origins"]),
                            "elapsed_seconds": time.perf_counter() - job_started, "resource": resource_snapshot(output), "completed_utc": utc()})
                        completed_jobs += 1
                        continue
                    fit_seconds = time.perf_counter() - begin
                    fitted.save(fit_path)
                    receipt = {"identity": identity, "plan_identity": plan["identity"], "binding": binding,
                        "track": track, "model": model, "trajectory": trajectory, "config": config,
                        "training_identity_sha256": content_hash(dict(training.identity)),
                        "fit_sha256": sha256(fit_path), "fit_bytes": fit_path.stat().st_size,
                        "fit_seconds": fit_seconds, "qualified": True, "diagnostics": fitted.diagnostics,
                        "resource": resource_snapshot(output), "completed_utc": utc()}
                    write_json(receipt_path, receipt)
                failures = 0
                for first in plan["origins"]:
                    destination = output / "forecasts" / job_id / f"{first:05d}.json"
                    prediction_identity = content_hash({"fit": identity, "first_target": first,
                                                        "history": plan["history"], "horizon": 100})
                    if destination.exists():
                        prior = read(destination)
                        if prior["identity"] != prediction_identity:
                            raise ValueError("Incompatible retained anchor forecast")
                        failures += prior["state"] == "failed"
                        continue
                    forecast_started = time.perf_counter()
                    base = {"identity": prediction_identity, "fit_identity": identity, "plan_identity": plan["identity"],
                            "track": track, "model": model, "trajectory": trajectory, "first_target": first,
                            "history": plan["history"], "horizon": 100, "partition": "validation", "qualified": True}
                    try:
                        window = adapter.model_window(trajectory, first, "validation", history=plan["history"], horizon=100)
                        prediction = fitted.forecast(window.panel, 100, window.target_coordinates, window.target_site_ids)
                        mean = physical_prediction(prediction.mean, training.transform, (100, len(window.target_site_ids)))
                        truth = adapter.evaluation_window(trajectory, first, "validation", history=plan["history"], horizon=100).latent_clean
                        metrics = score(mean, None, truth,
                                        training.transform.scale, window.target_coordinates)
                        write_json(destination, {**base, "state": "complete", "metrics": metrics,
                                                 "seconds": time.perf_counter() - forecast_started})
                    except (FloatingPointError, np.linalg.LinAlgError) as error:
                        write_json(destination, {**base, "state": "failed", "error": repr(error),
                                                 "seconds": time.perf_counter() - forecast_started})
                        failures += 1
                write_json(terminal_path, {"job_id": job_id, "plan_identity": plan["identity"], "binding": binding,
                    "state": "failed" if failures else "complete", "failed_origins": failures,
                    "elapsed_seconds": time.perf_counter() - job_started, "resource": resource_snapshot(output), "completed_utc": utc()})
                completed_jobs += 1
                del fitted
    write_json(output / "status.json", {"state": "complete", "completed_jobs": completed_jobs,
               "plan_identity": plan["identity"], "updated_utc": utc(), "last_invocation_seconds": time.perf_counter() - started})
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--original-freeze", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--models", nargs="+", choices=AVAILABLE_MODELS, default=list(MODELS),
                        help="Default cheap three; ridge ESN/NVAR are optional fixed original configurations, never retuned")
    parser.add_argument("--max-seconds", type=float, default=900)
    args = parser.parse_args(argv)
    threadpool_limits(1)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"Set {name}=1 before launching this numerical program")
    repository = Path(__file__).resolve().parents[1]
    plan = build_plan(args.results, args.original_freeze, repository, args.models)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.output.resolve() == args.results.resolve() or args.results.resolve() in args.output.resolve().parents:
        raise ValueError("Anchors must use a separate output root")
    path = args.output / "plan.json"
    if path.exists() and read(path) != plan:
        raise ValueError("Changed anchor source/configuration; use a new output root")
    write_json(path, plan)
    if not args.execute:
        print(json.dumps({"state": "prepared_only", "plan_identity": plan["identity"], "fits_run": 0,
                          "cache_opened": False, "scheduled_jobs": len(plan["tracks"]) * len(CONFIRMATION) * len(plan["models"])}))
        return
    with (args.output / "writer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = run(plan, args.results, args.cache, args.output, args.max_seconds)
    print(json.dumps({"state": "complete" if completed else "incomplete", "plan_identity": plan["identity"]}))


if __name__ == "__main__":
    main()
