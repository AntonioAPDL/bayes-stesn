#!/usr/bin/env python3
"""Collect compact, public-safe KS reporting evidence without running models.

Reads configuration, validation-selection metadata, attempt/fit receipts and
filesystem metadata. It never opens datasets, checkpoints, forecast receipts,
test summaries or prediction arrays. It can run before selection is frozen;
refresh it after execution for the final report. This command does not mutate
the scientific ledger or repair/reconcile running jobs.

    python tools/collect_ks_report_evidence.py --results RESULTS \
        --workspace WORKSPACE --published-commit COMMIT --local-revision COMMIT

The default output is RESULTS/report_evidence.json. Private absolute paths,
account details, remote URLs, raw errors and credentials are not copied.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile


SCHEMA = "ks-report-evidence/1"
SEARCH_STAGES = {"screen", "promote", "refine", "precision"}
OPTIMIZER_QUALIFICATION = (
    "Existing bounded L-BFGS-B gate: accepted SciPy OptimizeResult.success, finite objective below 1e99 and finite parameters, "
    "and no rejected worsening continuation. SciPy success may be reached by relative-objective ftol or projected-gradient gtol; "
    "the gate does not additionally require the recorded projected gradient to be <= gtol."
)
OPTIMIZER_MESSAGES = {
    "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL",
    "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
    "STOP: CPU EXCEEDING THE TIME LIMIT",
    "STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT",
    "STOP: PROJECTED GRADIENT IS SUFFICIENTLY SMALL",
    "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT",
    "STOP: CALLBACK REQUESTED HALT",
    "WARNING: ROUNDING ERRORS PREVENT PROGRESS", "WARNING: STP = STPMAX", "WARNING: STP = STPMIN",
    "WARNING: XTOL TEST SATISFIED", "ERROR: NO FEASIBLE SOLUTION", "ERROR: FACTR < 0", "ERROR: FTOL < 0",
    "ERROR: GTOL < 0", "ERROR: XTOL < 0", "ERROR: STP < STPMIN", "ERROR: STP > STPMAX",
    "ERROR: STPMIN < 0", "ERROR: STPMAX < STPMIN", "ERROR: INITIAL G >= 0", "ERROR: M <= 0",
    "ERROR: N <= 0", "ERROR: INVALID NBD", "ABNORMAL:", "ABNORMAL", "ABNORMAL_TERMINATION_IN_LNSRCH",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + "-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def distribution(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {"n": len(values), "min": min(values) if values else None,
            "median": statistics.median(values) if values else None, "max": max(values) if values else None}


def active_wall(intervals):
    """Length of the union of recorded attempt intervals, excluding idle gaps."""
    total, end = 0., None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop-start
        else:
            total += max(0., stop-end)
        end = max(stop, end) if end is not None else stop
    return total


def attempt_summary(attempts):
    intervals, seconds = [], 0.
    for attempt in attempts:
        duration = float(attempt.get("elapsed_seconds", attempt.get("last_observed_elapsed_seconds", 0.)))
        if duration < 0 or not math.isfinite(duration):
            raise ValueError("Invalid recorded attempt duration")
        seconds += duration
        if isinstance(attempt.get("started_epoch"), (int, float)):
            start = float(attempt["started_epoch"])
            intervals.append((start, start+duration))
    counts = Counter(attempt["state"] for attempt in attempts)
    return {"attempts": len(attempts), "worker_seconds": seconds, "active_wall_seconds": active_wall(intervals),
            "peak_monitored_rss_bytes": None,
            "attempt_rss_status": "unavailable: original Popen PID to mounted /proc namespace identity was not validated; do not use those samples",
            "running_attempts": counts["running"],
            "interrupted_attempts": sum(n for state, n in counts.items() if state not in {"running", "complete"}),
            "attempt_counts": dict(counts),
            "inexact_accounting_attempts": sum(r.get("accounting_exact") is False for r in attempts)}


def diagnostic_block(block):
    parameters = block.get("parameters", {})
    result = {"passed": bool(block.get("passed", False)), "chains": block.get("chains"), "draws_per_chain": block.get("draws"),
              "parameters": len(parameters), "failed_parameters": sum(not p.get("passed", False) for p in parameters.values())}
    for key, reducer in (("rhat", max), ("ess_bulk", min), ("ess_tail", min)):
        values = [p.get(key) for p in parameters.values()]
        finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
        result[("max_" if key == "rhat" else "min_") + key] = reducer(finite) if finite else None
        result["undefined_"+key] = len(values)-len(finite)
    # MCSE/SD is scale-free within a quantity; raw MCSEs across unlike
    # coefficients and hyperparameters are not collapsed into one number.
    relative = [p["mcse_mean"]/p["sd"] for p in parameters.values()
                if isinstance(p.get("mcse_mean"), (int, float)) and isinstance(p.get("sd"), (int, float))
                and p["sd"] > 0 and math.isfinite(p["mcse_mean"]) and math.isfinite(p["sd"])]
    result["max_mcse_over_posterior_sd"] = max(relative) if relative else None
    return result


def _finite(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _optimizer_message(value):
    text = " ".join(str(value or "").strip().split())
    return {"message": text if text in OPTIMIZER_MESSAGES else "UNRECOGNIZED OPTIMIZER MESSAGE (raw text not exported)",
            "message_allowlisted": text in OPTIMIZER_MESSAGES,
            "message_sha256": hashlib.sha256(text.encode()).hexdigest()}


def optimizer_fit(diag, config, model):
    """Report the accepted result; do not introduce a different success gate."""
    prefix = "covariance_" if model == "basis_gru" else ""
    details = diag.get(prefix+"optimizer_details")
    if not isinstance(details, dict) or not details.get("phases"):
        return None
    phases = []
    for phase in details["phases"]:
        phases.append({"phase": phase["phase"], "success": bool(phase["success"]), "status": phase["status"],
            **_optimizer_message(phase.get("message")), "iterations": phase["iterations"],
            "function_evaluations": phase["function_evaluations"], "projected_gradient_inf": _finite(phase.get("projected_gradient_inf")),
            "ftol": _finite(phase.get("ftol")), "gtol": _finite(phase.get("gtol"))})
    # With at most two phases, a rejected worsening continuation leaves the
    # first result accepted; otherwise the last phase is the retained result.
    rejected = bool(details.get("rejected_worsening_continuation", False))
    accepted = phases[0] if rejected else phases[-1]
    gradient = _finite(details.get("projected_gradient_inf"))
    gtol = _finite(config.get(prefix+"gtol", accepted["gtol"]))
    ftol = _finite(config.get(prefix+"ftol", accepted["ftol"]))
    boundaries = diag.get(prefix+"boundary_hits", [])
    if not isinstance(boundaries, list) or any(value not in {"spatial_length", "temporal_length", "signal_sd", "noise_sd"} for value in boundaries):
        raise ValueError("Unexpected optimizer boundary names; extend the public-safe whitelist explicitly")
    return {"component": "GRU residual covariance" if prefix else "GP marginal likelihood",
            "qualified_by_existing_gate": bool(details.get("qualified", False)),
            "qualification_criterion": OPTIMIZER_QUALIFICATION,
            "accepted_phase": accepted["phase"], "accepted_success": accepted["success"], "accepted_status": accepted["status"],
            **_optimizer_message(diag.get(prefix+"optimizer_message", accepted["message"])),
            "projected_gradient_inf": gradient, "configured_gtol": gtol, "configured_ftol": ftol,
            "gradient_within_gtol": None if gradient is None or gtol is None else gradient <= gtol,
            "boundary_hits": sorted(set(boundaries)), "rejected_worsening_continuation": rejected, "phases": phases,
            "neural_optimizer_convergence": "not_claimed" if prefix else "not_applicable"}


def optimizer_summary(fits):
    values = [fit["diagnostics"]["optimizer"] for fit in fits if fit["diagnostics"].get("optimizer")]
    if not values:
        return {}
    boundary_counts = Counter(name for value in values for name in value["boundary_hits"])
    return {"fit_count": len(values), "qualified_fits": sum(value["qualified_by_existing_gate"] for value in values),
            "termination_message_counts": dict(Counter(value["message"] for value in values)),
            "accepted_status_counts": dict(Counter(str(value["accepted_status"]) for value in values)),
            "accepted_success_count": sum(value["accepted_success"] for value in values),
            "unrecognized_termination_count": sum(not value["message_allowlisted"] for value in values),
            "projected_gradient_inf": distribution(value["projected_gradient_inf"] for value in values),
            "configured_gtol": distribution(value["configured_gtol"] for value in values),
            "configured_ftol": distribution(value["configured_ftol"] for value in values),
            "gradient_within_gtol_count": sum(value["gradient_within_gtol"] is True for value in values),
            "gradient_above_gtol_count": sum(value["gradient_within_gtol"] is False for value in values),
            "qualified_with_gradient_above_gtol_count": sum(value["qualified_by_existing_gate"] and value["gradient_within_gtol"] is False for value in values),
            "undefined_gradient_count": sum(value["projected_gradient_inf"] is None for value in values),
            "boundary_hit_counts": dict(boundary_counts), "fits_with_any_boundary": sum(bool(value["boundary_hits"]) for value in values),
            "rejected_worsening_continuation_count": sum(value["rejected_worsening_continuation"] for value in values),
            "qualification_criterion": OPTIMIZER_QUALIFICATION,
            "scope": "recorded likelihood/residual-covariance optimization; no neural mean-network convergence claim"}


def compact_fit(raw, config_id, selected_config, primary_seed):
    diag = raw.get("diagnostics", {}); config = raw.get("config", {})
    training = raw.get("training_identity", {}).get("transform", {})
    bounds = training.get("fit_time_bounds", [])
    supplied = diag.get("supplied_training_rows", diag.get("training_rows", bounds[1]-bounds[0] if len(bounds) == 2 else None))
    usable = next((diag[k] for k in ("likelihood_rows", "usable_targets", "usable_training_targets", "covariance_fit_rows", "fit_rows") if k in diag), supplied)
    selected = selected_config is not None and signature(config) == signature(selected_config)
    result = {"fit_id": raw["identity"], "track": raw["track"], "model": raw["model"], "phase": raw.get("phase"),
              "trajectory": raw.get("trajectory"), "seed": raw.get("seed"), "config_id": config_id,
              "config_sha256": signature(config), "state": raw["state"], "selected_configuration": selected,
              "selected_primary": selected and raw.get("seed") == primary_seed and raw.get("phase") == "core",
              "fit_receipt_seconds": raw.get("fit_seconds", 0.),
              "fit_receipt_peak_rss_bytes": raw.get("resource", {}).get("process_peak_rss_bytes", 0),
              "fit_bytes": raw.get("fit_bytes"), "fit_sha256": raw.get("fit_sha256"),
              "parameter_count": diag.get("parameter_count"), "supplied_training_rows": supplied,
              "supplied_training_sites": diag.get("supplied_training_sites", len(training["fit_site_indices"]) if "fit_site_indices" in training else None),
              "usable_training_targets": usable, "qualified": bool(diag.get("qualified", raw["state"] == "complete")),
              "effort": {}, "diagnostics": {}}
    for key in ("selected_epoch", "epochs_completed", "optimization_updates", "iterations", "function_evaluations",
                "covariance_iterations", "covariance_function_evaluations", "continuations", "conditioning_rows",
                "temporal_features", "spatial_basis", "readout_coefficients", "random_feature_parameters", "reservoir_nonzeros"):
        if key in diag:
            result["effort"][key] = diag[key]
    for key in ("selection_history", "refit_history"):
        if isinstance(diag.get(key), list):
            result["effort"][key.replace("history", "epochs")] = len(diag[key])
    for key in ("converged", "finite_fit", "optimizer_projected_gradient_inf", "covariance_converged", "covariance_projected_gradient_inf"):
        if key in diag:
            result["diagnostics"][key] = diag[key]
    for key in ("boundary_hits", "covariance_boundary_hits"):
        if key in diag:
            result["diagnostics"][key] = diag[key]
    optimizer = optimizer_fit(diag, config, raw["model"])
    if optimizer is not None:
        result["diagnostics"]["optimizer"] = optimizer
    gates = diag.get("gates", [])
    if gates:
        gate = gates[-1]
        result["diagnostics"]["bayesian_final_gate"] = {key: diagnostic_block(gate[key])
            for key in ("hyperparameters", "coefficients", "functionals") if key in gate}
        result["effort"]["chains"] = config.get("sampler", {}).get("chains")
        result["effort"]["warmup_per_chain"] = config.get("sampler", {}).get("warmup")
        result["effort"]["retained_draws_per_chain"] = gate.get("hyperparameters", {}).get("draws")
    return result


def tree_size(root):
    """Metadata-only size; skip symlinks and count each hardlinked inode once."""
    root = Path(root)
    if not root.exists():
        return {"exists": False, "files": 0, "logical_bytes": 0, "allocated_bytes": 0}
    seen, files, logical, allocated, skipped = set(), 0, 0, 0, 0
    for folder, directories, names in os.walk(root, followlinks=False):
        directories[:] = [name for name in directories if not (Path(folder)/name).is_symlink()]
        for name in names:
            path = Path(folder)/name
            try:
                if path.is_symlink():
                    skipped += 1; continue
                value = path.stat()
            except FileNotFoundError:
                skipped += 1; continue  # An atomic writer may replace its temporary file.
            inode = (value.st_dev, value.st_ino)
            if inode in seen:
                continue
            seen.add(inode); files += 1; logical += value.st_size; allocated += value.st_blocks*512
    return {"exists": True, "files": files, "logical_bytes": logical, "allocated_bytes": allocated,
            "skipped_symlinks_or_transient_files": skipped}


def resources(root):
    cgroup = Path("/sys/fs/cgroup"); memory = {}
    for name in ("memory.max", "memory.current", "memory.peak", "cpu.max"):
        path = cgroup/name
        if path.exists():
            value = path.read_text().strip(); memory[name] = int(value) if value.isdigit() else value
    disk = shutil.disk_usage(root); vfs = os.statvfs(root)
    status = Path("/proc/self/status").read_text()
    proc_pid = next(int(line.split()[1]) for line in status.splitlines() if line.startswith("Pid:"))
    return {"cgroup": memory, "affinity_cpus": len(os.sched_getaffinity(0)),
            "observed_pid_namespace_mismatch": proc_pid != os.getpid(),
            "disk_available_bytes": vfs.f_bavail*vfs.f_frsize, "disk_free_bytes": disk.free,
            "disk_total_bytes": disk.total, "inodes_available": vfs.f_favail,
            "scope": "snapshot shared with other workspace processes; not attributed entirely to this experiment"}


def git_value(repo, *args):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def collect(results, *, repo, workspace=None, published_commit=None, local_revision=None):
    results, repo = Path(results).resolve(), Path(repo).resolve()
    inputs = []
    def read(path):
        data = path.read_bytes()
        relative = str(path.relative_to(results)) if path.is_relative_to(results) else "repository/"+str(path.relative_to(repo))
        inputs.append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        return json.loads(data)
    experiment = read(results/"experiment.json"); config = experiment["config"]
    frozen = read(results/"freeze.json") if (results/"freeze.json").exists() else None
    access = read(results/"test_access.json") if (results/"test_access.json").exists() else None
    if frozen and frozen["experiment_identity"] != experiment["identity"]:
        raise ValueError("Freeze and experiment identities differ")
    if access and (not frozen or access["freeze_identity"] != frozen["identity"] or not access.get("both_track_selection_complete")):
        raise ValueError("Test-access receipt is not bound to completed two-track selection")
    jobs = {}
    for path in sorted((results/"jobs").glob("*.json")):
        value = read(path)
        jobs[value["job_id"]] = {key: value.get(key) for key in ("fit_id", "model", "track", "config_id", "stage")}
    fit_jobs = {value["fit_id"]: value for value in jobs.values()}
    attempts = [read(path) for path in sorted((results/"attempts").glob("*.json"))]
    by_stage = defaultdict(list)
    for value in attempts:
        by_stage[(value["stage"], value["track"], value["model"])].append(value)
    campaign = attempt_summary(attempts)
    campaign["stage_rows"] = [{"stage": stage, "track": track, "model": model, **attempt_summary(values)}
        for (stage, track, model), values in sorted(by_stage.items())]
    campaign["screening"] = attempt_summary([r for r in attempts if r["stage"] in SEARCH_STAGES])
    campaign["accounting"] = "recorded process elapsed time includes startup, fitting, forecasts, scoring and interrupted attempts; active wall is interval union excluding idle gaps; running entries use last recorded heartbeat only"
    ledger_states = Counter(); pending_reasons = Counter()
    for path in sorted((results/"ledger").glob("*.json")):
        value = read(path); ledger_states[value["state"]] += 1
        if value["state"] == "pending" and value.get("reason"):
            pending_reasons[value["reason"]] += 1
    campaign["ledger_states"] = dict(ledger_states); campaign["pending_reasons"] = dict(pending_reasons)
    fits = []
    for path in sorted((results/"fits").glob("*/receipt.json")):
        raw = read(path); selected = None
        if frozen:
            chosen = frozen["selected"].get("A" if raw["track"] == "C" else raw["track"], {}).get(raw["model"])
            selected = chosen["config"] if chosen else None
        fit = compact_fit(raw, fit_jobs.get(raw["identity"], {}).get("config_id"), selected, config["model_seed"])
        fits.append(fit)
    family_rows = []
    families = {(track, model) for track, models in config["candidates"].items() for model in models}
    families |= {("C", model) for model in config.get("robustness", {}).get("models", [])}
    for track, model in sorted(families):
        all_fits = [fit for fit in fits if fit["track"] == track and fit["model"] == model]
        selected_fits = [fit for fit in all_fits if fit["selected_primary"]]
        chosen = frozen["selected"].get("A" if track == "C" else track, {}).get(model) if frozen else None
        selected_ids = {fit["fit_id"] for fit in selected_fits}
        observed_counts = Counter(fit["state"] for fit in selected_fits)
        group_attempts = [a for a in attempts if a["track"] == track and a["model"] == model]
        group_search = [a for a in group_attempts if a["stage"] in SEARCH_STAGES]
        row = {"track": track, "model": model,
               "initial_candidates": len(config["candidates"].get(track, {}).get(model, [])),
               "evaluated_candidates": len({fit["config_sha256"] for fit in all_fits}),
               "candidate_count_definition": "distinct effective configuration dictionaries with a terminal fit receipt, including failed fits",
               "selected_config_id": chosen["id"] if chosen else None,
               "selection_eligible": chosen.get("selection_eligible") if chosen else None,
               "selected_primary_fit_count": len(selected_fits), "fit_state_counts": dict(observed_counts),
               "fit_receipt_seconds": sum(fit["fit_receipt_seconds"] for fit in selected_fits),
               "fit_receipt_peak_rss_bytes": max((fit["fit_receipt_peak_rss_bytes"] for fit in selected_fits), default=0),
               "attempts": attempt_summary(group_attempts), "screening": attempt_summary(group_search),
               "selected_primary_fit_ids": sorted(selected_ids),
               "parameter_count": distribution(fit["parameter_count"] for fit in selected_fits),
               "supplied_training_rows": distribution(fit["supplied_training_rows"] for fit in selected_fits),
               "supplied_training_sites": distribution(fit["supplied_training_sites"] for fit in selected_fits),
               "usable_training_targets": distribution(fit["usable_training_targets"] for fit in selected_fits),
               "effort": {}, "bayesian_diagnostics": {}, "optimizer_diagnostics": optimizer_summary(selected_fits)}
        for key in sorted({key for fit in selected_fits for key in fit["effort"]}):
            row["effort"][key] = distribution(fit["effort"].get(key) for fit in selected_fits)
        for block in ("hyperparameters", "coefficients", "functionals"):
            values = [fit["diagnostics"].get("bayesian_final_gate", {}).get(block) for fit in selected_fits]
            values = [value for value in values if value]
            if values:
                row["bayesian_diagnostics"][block] = {"fits": len(values), "passed_fits": sum(value["passed"] for value in values),
                    "failed_parameters_across_fits": sum(value["failed_parameters"] for value in values),
                    **{key: distribution(value.get(key) for value in values) for key in (
                        "parameters", "max_rhat", "min_ess_bulk", "min_ess_tail", "undefined_rhat", "max_mcse_over_posterior_sd")}}
        family_rows.append(row)
    optimizer_groups = defaultdict(list)
    for fit in fits:
        if fit["diagnostics"].get("optimizer"):
            optimizer_groups[(fit["track"], fit["model"], fit["config_id"], fit["seed"])].append(fit)
    optimizer_rows = [{"track": track, "model": model, "config_id": config_id, "seed": seed,
                       "selected_configuration": any(fit["selected_configuration"] for fit in values),
                       **optimizer_summary(values)}
                      for (track, model, config_id, seed), values in sorted(optimizer_groups.items(), key=lambda item: str(item[0]))]
    precision = None
    path = results/"validation_precision.json"
    if path.exists():
        raw = read(path)
        if raw["experiment_identity"] != experiment["identity"]:
            raise ValueError("Predictive precision belongs to another experiment")
        precision = {key: raw[key] for key in ("state", "selected_paths", "all_checked_cases_passed", "elapsed_seconds", "test_data_access", "fits_performed")}
        precision.update(cases=len(raw["cases"]), unresolved_cases=len(raw["unresolved_cases"]), skipped_cases=len(raw.get("skipped", [])))
        precision["policy"] = raw["policy"]
    preflight_path = repo/"docs/applications/evidence/ks_preflight.json"
    preflight = read(preflight_path) if preflight_path.exists() else None
    extra_acceptance = {}
    for name in ("ks_core_candidate_invariance.json", "ks_validation_audit_early.json", "ks_validation_promotion_A_audit.json",
                 "ks_contiguous_history_check.json", "ks_memory_recovery_20261105.json",
                 "ks_memory_recovery_20261105_event2.json"):
        path = repo/"docs/applications/evidence"/name
        if path.exists():
            value = read(path)
            if name == "ks_contiguous_history_check.json":
                if (frozen is None or value.get("freeze_identity") != frozen["identity"] or
                        value.get("freeze_sha256") != sha256(results/"freeze.json")):
                    raise ValueError("Contiguous history evidence belongs to another frozen selection")
            elif value.get("experiment_identity") != experiment["identity"]:
                raise ValueError("Additional acceptance evidence belongs to another experiment")
            extra_acceptance[name.removesuffix(".json")] = value
    external = None
    observer_path = repo/"tools/watch_ks_resources.py"
    observer_spec = importlib.util.spec_from_file_location("ks_reporting_observer", observer_path)
    observer = importlib.util.module_from_spec(observer_spec)
    observer_spec.loader.exec_module(observer)
    records = observer.read_session_records(results, read=read)
    if records:
        if any(row.get("experiment_identity", experiment["identity"]) != experiment["identity"] for row in records):
            raise ValueError("Observer session belongs to another experiment")
        raw = observer.merge_sessions(records)
        external = {key: raw.get(key) for key in ("started_utc", "last_observed_utc", "status", "samples",
            "combined_worker_rss_peak_bytes", "disk_free_min_bytes", "container_current_peak_bytes", "original_attempt_rss_valid",
            "session_count", "sessions", "coverage_gaps", "retained_job_rows")}
        external["method"] = "exact absolute --job path matched in outer proc cmdline; same-entry VmRSS/VmHWM"
        external["scope"] = ("Partial observation period beginning after campaign launch. Distinct session aggregates are merged by maxima and disk minima; "
            "samples are summed only across distinct sessions. Missing per-job rows remain unavailable and use fit-receipt RUSAGE_SELF evidence; "
            "retained_job_rows and observed_jobs count actual retained rows, not all jobs observed in sessions with incomplete records. "
            "Container memory includes unrelated processes.")
        external["alerts_count"] = len(raw.get("alerts", []))
        external["jobs"] = {job: {key: value.get(key) for key in ("model", "track", "stage", "first_seen_utc", "last_seen_utc", "peak_rss_bytes", "peak_hwm_bytes")}
                            for job, value in raw.get("jobs", {}).items()}
        external["observed_jobs"] = len(external["jobs"])
        for row in campaign["stage_rows"] + family_rows:
            seen = [value for value in external["jobs"].values() if value["track"] == row["track"] and value["model"] == row["model"]
                    and ("stage" not in row or value["stage"] == row["stage"])]
            row["externally_observed_jobs"] = len(seen)
            row["external_peak_rss_bytes"] = max((value["peak_rss_bytes"] for value in seen), default=None)
            row["external_peak_hwm_bytes"] = max((value["peak_hwm_bytes"] for value in seen), default=None)
    storage = {"results": tree_size(results)}
    if workspace is not None:
        for label in ("incoming", "cache", "env", "pilot", "private"):
            storage[label] = tree_size(Path(workspace)/label)
    current = resources(results)
    status = git_value(repo, "status", "--porcelain")
    evidence = {"schema": SCHEMA, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "binding": {"experiment_identity": experiment["identity"], "freeze_identity": frozen["identity"] if frozen else None,
                    "published_remote_commit": published_commit, "local_tree_revision": local_revision,
                    "observed_checkout_revision": git_value(repo, "rev-parse", "HEAD"), "branch": git_value(repo, "branch", "--show-current"),
                    "working_tree_changed_entries": len(status.splitlines()) if status else 0,
                    "collector_source_sha256": sha256(__file__),
                    "report_source_sha256": sha256(repo/"src/bayes_stesn/applications/ks_report.py")},
        "campaign": campaign, "family_rows": family_rows, "fits": fits, "optimizer_rows": optimizer_rows,
        "current_resources": current, "storage": storage, "preflight": preflight, "predictive_precision": precision,
        "external_resources": external, "extra_acceptance": extra_acceptance,
        "budget": {key: config["budget"][key] for key in ("workers", "threads_per_worker", "initial_total_wall_hours", "total_worker_seconds_cap",
            "search_wall_hours_cap", "search_worker_seconds_cap", "combined_worker_cap_bytes", "worker_cap_bytes", "disk_reserve_bytes") if key in config["budget"]},
        "limitations": ["No test scores, forecasts, dataset values or checkpoints read; final scientific results remain in their own summaries.",
            "Diagnostic extrema use finite values and separately retain undefined counts; different parameter-count definitions are not architecture-equivalent complexity measures.",
            "Fit-receipt time excludes earlier interrupted attempts; campaign attempt accounting includes recorded interruption and startup costs.",
            "Storage is a metadata snapshot; allocated bytes deduplicate hardlinks within each reported root. Running attempts and concurrent writes make an interim snapshot provisional.",
            "Original runner /proc RSS samples are not used: child PIDs were in a different namespace from the mounted /proc. RUSAGE_SELF peaks in fit receipts remain valid; external validated watchdog evidence is required for whole-worker peaks.",
            "Published/local revision identifiers are explicit reporting provenance supplied by the caller; no remote fetch or push is performed."],
        "inputs": inputs, "model_fit_or_scoring_performed": False, "test_outputs_read": False}
    # Refuse accidental expansion of the public-safe whitelist.
    text = json.dumps(evidence, allow_nan=False)
    if any(token in text for token in ("@gmail", "@ucsc", "drive.google", "/workspace/", "file_id", "folder_id")):
        raise ValueError("Public evidence unexpectedly contains private retrieval/path fields")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--published-commit")
    parser.add_argument("--local-revision")
    args = parser.parse_args()
    for name in ("published_commit", "local_revision"):
        value = getattr(args, name)
        if value is not None and not re.fullmatch(r"[0-9a-f]{40}", value):
            parser.error("Revision arguments must be full lowercase 40-character Git identifiers")
    evidence = collect(args.results, repo=args.repo, workspace=args.workspace,
                       published_commit=args.published_commit, local_revision=args.local_revision)
    output = args.output or args.results/"report_evidence.json"
    atomic_json(output, evidence)
    print(json.dumps({"schema": SCHEMA, "frozen": evidence["binding"]["freeze_identity"] is not None,
        "attempts": evidence["campaign"]["attempts"], "fit_receipts": len(evidence["fits"]),
        "worker_seconds": evidence["campaign"]["worker_seconds"], "test_outputs_read": False,
        "artifact_sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
