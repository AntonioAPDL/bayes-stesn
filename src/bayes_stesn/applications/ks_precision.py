"""Validation-only predictive Monte Carlo precision checks for the KS study.

This examines simulation error conditional on each retained fitted posterior or
plug-in model.  Independent RNG streams are not independent posterior fits and
cannot establish MCMC convergence.  No model is fitted, calibrated or reselected.
"""
from __future__ import annotations

import gc
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np

from ..io import content_hash, seed_for
from ..scoring import crps_ensemble
from .ks_metrics import horizon_balance
from .ks_runtime import admission, atomic_json, file_hash, now, resource_snapshot

SCHEMA = "ks-validation-predictive-precision-1"
ORIGINS = (6000, 7500)
PATH_LEVELS = (256, 512, 1024)
REPLICATES = 4
BATCH_SIZE = 64
POLICY = {
    "trajectory": 0, "origins": list(ORIGINS), "partition": "validation",
    "replicates": REPLICATES, "paths": list(PATH_LEVELS), "batch_size": BATCH_SIZE,
    "normalized_crps_single_ensemble_sd_limit": .01,
    "mean_95_interval_width_cv_limit": .10, "runner_up_relative_tolerance": .01,
    "escalation": "extend_each_case_until_both_thresholds_pass_or_1024; common_final_paths_is_maximum_case_requirement",
    "conditional_scope": "simulation_error_given_retained_fitted_distribution; excludes_posterior_inference_error_and_scientific_sampling_uncertainty",
    "estimator": "empirical_distribution_CRPS_V_statistic; no_iid_finite_ensemble_bias_correction",
    "selection": "diagnose_near_ties_only; never_reselect_candidates_or_change_S1_S2_path_counts",
}


def _read(path):
    return json.loads(Path(path).read_text())


def _budget_gate(study):
    budget = study.cfg["budget"]
    cap = budget.get("total_worker_seconds_cap", budget["initial_total_wall_hours"]*3600*budget.get("workers", 1))
    if study.consumed_worker_seconds() >= cap:
        raise StopIteration("Precision cumulative compute budget reached; completed cases are resumable")


def _metrics(paths, truth, scale):
    """Chunk score work; accept [path,h,site] or [ensemble,path,h,site]."""
    x, y = np.asarray(paths, np.float64), np.asarray(truth, np.float64)
    if x.ndim not in (3, 4) or x.shape[-2:] != y.shape:
        raise ValueError("Precision score needs complete [path,lead,site] draws")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not (scale > 0):
        raise FloatingPointError("Nonfinite predictive precision case")
    axes = tuple(range(x.ndim-2))
    mean = x.mean(axis=axes, dtype=np.float64)
    with np.errstate(over="ignore"):
        mse_h = np.mean((mean-y)**2, axis=1)
    if not np.isfinite(mse_h).all():
        raise FloatingPointError("Predictive mean squared-error overflow")
    crps_h, width_h = np.empty(len(y)), np.empty(len(y))
    for start in range(0, len(y), 10):
        stop = min(start+10, len(y))
        block = x[..., start:stop, :].reshape(-1, stop-start, y.shape[1])
        crps_h[start:stop] = crps_ensemble(block, y[start:stop]).mean(axis=1)
        lo, hi = np.quantile(block, [.025, .975], axis=0)
        width_h[start:stop] = (hi-lo).mean(axis=1)
    return {"crps": float(crps_h.mean()), "selection_crps": horizon_balance(crps_h),
        "normalized_crps": float(crps_h.mean()/scale),
        "normalized_selection_crps": horizon_balance(crps_h)/scale,
        "mean_forecast_rmse": float(np.sqrt(mse_h.mean())),
        "mean_forecast_nrmse": float(np.sqrt(mse_h.mean())/scale),
        "mean_95_interval_width": float(width_h.mean()),
        "normalized_mean_95_interval_width": float(width_h.mean()/scale),
        "paths": int(np.prod(x.shape[:-2]))}


def _level(paths, truth, scale, count):
    ensembles = [_metrics(paths[r, :count], truth, scale) for r in range(REPLICATES)]
    pooled = _metrics(paths[:, :count], truth, scale)
    normalized_sd = float(np.std([r["normalized_crps"] for r in ensembles], ddof=1))
    widths = np.array([r["mean_95_interval_width"] for r in ensembles])
    # A degenerate point distribution has zero width and zero width variability.
    cv = float(widths.std(ddof=1)/widths.mean()) if widths.mean() > 0 else 0.
    flags = []
    if normalized_sd > POLICY["normalized_crps_single_ensemble_sd_limit"]:
        flags.append("normalized_single_ensemble_CRPS_sd_above_0.01")
    if cv > POLICY["mean_95_interval_width_cv_limit"]:
        flags.append("mean_95_interval_width_CV_above_0.10")
    return {"paths_per_ensemble": count, "ensembles": ensembles, "pooled": pooled,
        "normalized_single_ensemble_crps_sd": normalized_sd,
        "normalized_single_ensemble_selection_crps_sd": float(np.std(
            [r["normalized_selection_crps"] for r in ensembles], ddof=1)),
        "normalized_single_ensemble_mean_forecast_rmse_sd": float(np.std(
            [r["mean_forecast_nrmse"] for r in ensembles], ddof=1)),
        "mean_95_interval_width_cv": cv, "flags": flags, "thresholds_passed": not flags,
        "replicate_uncertainty_caveat": "only_four_simulation_replicates; thresholds_are_experiment_conventions_not_inference_guarantees"}


def _contrast(left, right):
    """Simulation contrast at common checked P; no scientific-replicate CI."""
    levels = sorted(set(left["levels"]) & set(right["levels"]), key=int)
    result = []
    for level in levels:
        a, b = left["levels"][level], right["levels"][level]
        differences = np.array([x["selection_crps"]-y["selection_crps"]
            for x, y in zip(a["ensembles"], b["ensembles"])])
        sd = float(differences.std(ddof=1))
        se = sd/np.sqrt(REPLICATES)
        result.append({"paths_per_ensemble": int(level),
            "selection_crps_difference_ensembles": differences.tolist(),
            "mean_difference": float(differences.mean()), "single_ensemble_difference_sd": sd,
            "replicate_mean_difference_mcse": float(se),
            "difference_within_twice_mcse": bool(abs(differences.mean()) <= 2*se),
            "pooled_selection_crps_difference": a["pooled"]["selection_crps"]-b["pooled"]["selection_crps"],
            "normalized_single_ensemble_difference_sd": sd/left["training_scale"]})
    return {"track": left["track"], "model": left["model"],
        "trajectory": 0, "first_target": left["first_target"],
        "difference": "chosen_minus_near_tie_runner_up", "chosen": left["candidate_id"],
        "runner_up": right["candidate_id"], "levels": result,
        "interpretation": "conditional_predictive_simulation_uncertainty_only; independently_seeded_models; four_replicates; not_a_trajectory_or_MCMC_confidence_interval"}


def _case(study, descriptor, adapter, train):
    """One fixed validation case; keep all other origins out of RAM."""
    from .ks_native import NativeFit

    fit_path = Path(descriptor["fit_path"])
    identity = content_hash({"schema": SCHEMA, "experiment": study.identity,
        "policy": POLICY, "case": descriptor, "training": dict(train.identity),
        "history": study.cfg["history"], "horizon": study.cfg["horizon"]})
    path = study.output/"precision_cases"/(identity+".json")
    if path.exists():
        saved = _read(path)
        if saved["identity"] != identity:
            raise ValueError("Precision receipt identity mismatch")
        if saved["state"] in {"complete", "failed"}:
            return saved
    raw_bytes = REPLICATES*PATH_LEVELS[-1]*study.cfg["horizon"]*(13 if descriptor["track"] == "A" else 64)*8
    estimate = study.cfg["budget"].get("estimated_peak_rss_bytes", {}).get(descriptor["model"], 512*1024**2)
    before = admission(study.output, reserve_bytes=study.cfg["budget"]["disk_reserve_bytes"],
        estimated_peak_bytes=estimate+raw_bytes+40*1024**2,
        worker_cap_bytes=study.cfg["budget"].get("combined_worker_cap_bytes", int(3.8*1024**3)))
    _budget_gate(study)
    fit = NativeFit.load(fit_path)
    if fit.model_id != descriptor["model"] or not fit.diagnostics.get("qualified", False):
        raise ValueError("A precision case must use its verified qualified native fit")
    if fit.diagnostics.get("input_identity") != fit.input_identity:
        raise ValueError("Native fit internal training identity mismatch")
    if (not np.array_equal(fit.training.y, train.panel.y) or
            not np.array_equal(fit.training.coordinates, train.panel.coordinates) or
            not np.array_equal(fit.training.times, train.panel.times) or
            fit.training.site_ids != train.panel.site_ids):
        raise ValueError("Precision fit does not match permitted trajectory-specific training data")
    start = time.perf_counter()
    origin = descriptor["first_target"]
    window = adapter.model_window(0, origin, "validation", history=study.cfg["history"], horizon=study.cfg["horizon"])
    streams = [seed_for(study.cfg["forecast_seed"], SCHEMA, descriptor["track"],
        descriptor["model"], descriptor["candidate_id"], 0, origin, "ensemble", r)
        for r in range(REPLICATES)]
    record = {**descriptor, "identity": identity, "native_fit_identity": fit.identity,
        "native_input_identity": fit.input_identity, "training_identity": dict(train.identity),
        "transform": train.transform.record(), "training_scale": train.transform.scale,
        "history": study.cfg["history"], "horizon": study.cfg["horizon"],
        "ensemble_seeds": streams, "source_sha256": file_hash(__file__),
        "levels": {}, "state": "running", "resource_before": before, "started_utc": now()}
    atomic_json(path, record)
    samples = np.empty((REPLICATES, PATH_LEVELS[-1], study.cfg["horizon"], len(window.target_site_ids)), dtype=np.float64)
    previous, truth = 0, None
    try:
        for count in PATH_LEVELS:
            for replicate, stream in enumerate(streams):
                _budget_gate(study)
                for batch in fit.iter_forecasts(window.panel, study.cfg["horizon"], window.target_coordinates,
                        window.target_site_ids, count-previous, stream, batch_size=BATCH_SIZE, path_start=previous):
                    a, b = batch.metadata["path_start"], batch.metadata["path_stop"]
                    samples[replicate, a:b] = train.transform.inverse(batch.samples)
                current = resource_snapshot(study.output)
                if current["disk_free_bytes"] < study.cfg["budget"]["disk_reserve_bytes"]:
                    raise RuntimeError("Precision disk reserve reached; completed receipts retained")
                mem = current["memory"]
                if mem.get("memory.max") not in (None, "max"):
                    available = int(mem["memory.max"])-int(mem["memory.current"])+int(.5*mem.get("inactive_file_bytes", 0))
                    if available < 256*1024**2:
                        raise RuntimeError("Precision container headroom reached; completed receipts retained")
            # The facade can internally read evaluation labels; only its causal
            # model section reached forecasting. Labels are requested here for scoring.
            if truth is None:
                truth = adapter.evaluation_window(0, origin, "validation",
                    history=study.cfg["history"], horizon=study.cfg["horizon"]).values
            result = _level(samples, truth, train.transform.scale, count)
            record["levels"][str(count)] = result
            record["last_completed_paths"] = count
            atomic_json(path, record)
            previous = count
            if result["thresholds_passed"]:
                break
        record.update(state="complete", required_paths=count,
            precision_thresholds_passed=result["thresholds_passed"],
            flags_at_cap=result["flags"] if count == PATH_LEVELS[-1] else [],
            elapsed_seconds=time.perf_counter()-start, resource_after=resource_snapshot(study.output), completed_utc=now())
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        record.update(state="failed", error=repr(error), required_paths=PATH_LEVELS[-1],
            precision_thresholds_passed=False, flags_at_cap=["nonfinite_or_numerical_predictive_failure"],
            elapsed_seconds=time.perf_counter()-start, resource_after=resource_snapshot(study.output), completed_utc=now())
    except ValueError as error:
        if not str(error).startswith(("Forecast mean must be finite", "Samples must be finite")):
            raise
        record.update(state="failed", error=repr(error), required_paths=PATH_LEVELS[-1],
            precision_thresholds_passed=False, flags_at_cap=["nonfinite_predictive_paths"],
            elapsed_seconds=time.perf_counter()-start, resource_after=resource_snapshot(study.output), completed_utc=now())
    finally:
        del samples, fit
        gc.collect()
    atomic_json(path, record)
    return record


def _run_validation_precision(study):
    """Check both frozen validation rosters and return the common path budget.

    Call after both tracks' promotion receipts exist and before test access.
    Completed matching case receipts are reused. An interrupted unfinished case
    restarts its deterministic simulation streams; no training is performed.
    """
    from threadpoolctl import threadpool_limits
    from .ks_data import KSAdapter
    from .ks_runner import NATIVE, PROTOCOLS, numerical_identity

    if (study.output/"test_access.json").exists():
        raise ValueError("Predictive path-budget selection is sealed after test access")
    if study.source != numerical_identity():
        raise ValueError("Precision module/source changed after experiment identity was created")
    if study.cfg["paths"] != PATH_LEVELS[0]:
        raise ValueError("Precision policy requires original screening ensembles of 256 paths")
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    selections, descriptors, skipped, provenance = {}, [], [], {}
    # Both complete rosters are checked before opening any validation case.
    for track in ("A", "B"):
        for model in study.cfg["candidates"][track]:
            path = study.output/"selection"/f"{track}_{model}_promote.json"
            if not path.exists():
                raise ValueError("Both main-track promotion rosters must finish before precision checks")
            record = _read(path)
            if not record.get("ranking"):
                raise ValueError("Empty promotion roster")
            selections[track, model] = record
            provenance[f"{track}/{model}"] = file_hash(path)
    for (track, model), selection in selections.items():
        if model not in NATIVE:
            continue
        ranking = selection["ranking"]
        chosen = ranking[0]
        candidates = [("chosen", chosen)]
        eligible = [r for r in ranking[1:] if r["eligible"] and r.get("loss") is not None]
        if chosen["eligible"] and eligible:
            runner_up = min(eligible, key=lambda r: r["loss"])
            denominator = max(abs(float(chosen["loss"])), abs(float(runner_up["loss"])), np.finfo(float).tiny)
            if abs(chosen["loss"]-runner_up["loss"])/denominator <= POLICY["runner_up_relative_tolerance"]:
                candidates.append(("near_tie_runner_up", runner_up))
        for role, entry in candidates:
            candidate = entry["candidate"]
            job = study.make_job("precision_identity", track, model, candidate, 0, ORIGINS)
            folder = study.output/"fits"/job["fit_id"]
            receipt_path, fit_path = folder/"receipt.json", folder/"fit.npz"
            base = {"track": track, "model": model, "candidate_id": candidate["id"],
                "candidate_role": role, "promotion_eligible": bool(entry["eligible"]),
                "promotion_loss": entry.get("loss"), "trajectory": 0, "fit_id": job["fit_id"]}
            if not receipt_path.exists():
                raise ValueError("Promotion precision fit receipt is missing")
            receipt = _read(receipt_path)
            if receipt["identity"] != job["fit_id"]:
                raise ValueError("Precision fit receipt identity mismatch")
            if receipt["state"] != "complete" or not receipt.get("diagnostics", {}).get("qualified", False):
                skipped.append({**base, "reason": "trajectory_0_fit_failed_or_inference_unqualified",
                    "fit_state": receipt["state"], "fit_receipt_sha256": file_hash(receipt_path)})
                continue
            if not fit_path.exists() or file_hash(fit_path) != receipt["fit_sha256"]:
                raise ValueError("Precision fitted artifact is missing or has failed hash verification")
            train = KSAdapter(study.cache, PROTOCOLS[track], "core").training(0)
            if content_hash(receipt["training_identity"]) != content_hash(dict(train.identity)):
                raise ValueError("Precision fit has incompatible training/transform identity")
            for origin in ORIGINS:
                descriptors.append({**base, "first_target": origin, "fit_path": str(fit_path),
                    "fit_sha256": receipt["fit_sha256"], "fit_receipt_sha256": file_hash(receipt_path)})
    identity = content_hash({"schema": SCHEMA, "experiment": study.identity, "policy": POLICY,
        "selection_hashes": provenance, "cases": descriptors, "skipped": skipped})
    destination = study.output/"validation_precision.json"
    if destination.exists():
        saved = _read(destination)
        if saved["identity"] != identity:
            raise ValueError("Predictive precision receipt no longer matches source/selection/fits")
        return saved
    started = time.perf_counter()
    cases = []
    attempt_path = study.output/"attempts"/f"precision_{identity[:20]}_{time.time_ns()}.json"
    attempt = {"job_id": "precision_"+identity[:20], "stage": "precision", "model": "native_predictive_precision",
        "track": "A+B", "state": "running", "started_epoch": time.time(), "started_utc": now(), "pid": os.getpid()}
    atomic_json(attempt_path, attempt)
    try:
        with threadpool_limits(limits=1):
            for descriptor in descriptors:
                adapter = KSAdapter(study.cache, PROTOCOLS[descriptor["track"]], "core")
                cases.append(_case(study, descriptor, adapter, adapter.training(0)))
    except BaseException as error:
        atomic_json(attempt_path, {**attempt, "state": "interrupted", "error": repr(error),
            "elapsed_seconds": time.perf_counter()-started, "finished_utc": now()})
        raise
    else:
        atomic_json(attempt_path, {**attempt, "state": "complete", "returncode": 0,
            "elapsed_seconds": time.perf_counter()-started, "finished_utc": now()})
    contrasts = []
    for left in cases:
        if left["candidate_role"] != "chosen" or left["state"] != "complete":
            continue
        for right in cases:
            if (right["candidate_role"] == "near_tie_runner_up" and right["state"] == "complete"
                    and (left["track"], left["model"], left["first_target"]) ==
                        (right["track"], right["model"], right["first_target"])):
                contrasts.append(_contrast(left, right))
    selected = max([case["required_paths"] for case in cases], default=PATH_LEVELS[0])
    result = {"schema": SCHEMA, "identity": identity, "experiment_identity": study.identity,
        "source": study.source, "selection_receipt_hashes": provenance, "policy": POLICY,
        "state": "complete" if cases else "no_qualified_cases", "selected_paths": selected,
        "cases": cases, "skipped": skipped, "pairwise_contrasts": contrasts,
        "all_checked_cases_passed": bool(cases) and all(c["precision_thresholds_passed"] for c in cases),
        "unresolved_cases": [c["identity"] for c in cases if not c["precision_thresholds_passed"]],
        "path_budget_note": "one_common_final_count_for_both_tracks; case_checks_stop_at_first_passing_level; an_increase_is_not_proof_of_precision_at_every_unchecked_origin",
        "screening_note": "S1_S2_remain_256_paths; original_selection_and_near_ties_are_preserved; no_validation_or_test_outcome_reselection",
        "pooled_note": "baseline_four_256_path_ensembles_form_1024_path_estimate; escalations_pool_2048_or_4096_paths_for_diagnostics_only",
        "test_data_access": False, "fits_performed": 0,
        "elapsed_seconds": time.perf_counter()-started, "completed_utc": now()}
    atomic_json(destination, result)
    return result


def run_validation_precision(study):
    """Serialize this validation-only stage; never duplicate fit/case writers."""
    with (study.output/"precision.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("An active validation precision process owns this study") from error
        return _run_validation_precision(study)
