#!/usr/bin/env python3
"""Migrate a closed frozen KS run under one prospective operational allowance.

Default: metadata-only dry-run, requiring genuine closure evidence and all
existing source locks. --apply verifies current numerical source/runtime/cache,
then copies canonical checkpoint and evidence bytes to a new atomic run root.
Only experiment/output/job identities and their terminal bindings are rebased;
an explicit --source-cache-path also permits relocating the byte-identical cache.
Explicitly reviewed opaque fit sidecars can be preserved outside active fits
using --fit-sidecar-manifest; unlisted unknown files still stop migration.
One explicitly reviewed inconsistent inference-sidecar pair may be archived
with --inference-sidecar-manifest while its verified completed native fit stays
active; that archived pair is never declared an exact inference restart.
The scientific freeze, original start, all attempt charges, forecasts and fits
are preserved. No fitting, forecasting, selection, or resource-gate change runs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import copy
import ctypes
from datetime import datetime, timezone
import errno
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile


def helper(filename, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


N = helper("package_ks_improvement_fits.py", "ks_budget_native_validation")
TERMINAL = {"complete", "failed", "inference-unqualified"}
SCHEMA = "ks-improvement-budget-migration/1"
ALLOWED_BUDGET_CHANGES = {"total_worker_seconds_cap", "wall_seconds_cap", "extension_reason", "operational_extension"}
FORECAST_FIELDS = {"candidate_id", "first_target", "fit_key", "forecast_namespace", "forecast_seed",
    "history", "identity", "native_fit", "partition", "paths", "qualified", "seconds", "seed",
    "stage_first_scored", "state", "track", "trajectory"}
SCALAR = re.compile(r'^  "([^"\\]+)": (.*?)(?:,)?\n?$')
COMPLETED_PHASE_CHAIN_ERROR = "Completed phase does not match its exact chain checkpoint"


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return N.read_json(path)


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(N.encoded(value)); stream.flush(); os.fsync(stream.fileno())


def date(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Evidence timestamps must be timezone-aware")
    return parsed


def recorded_cache_path(value):
    """Validate an explicit historical location without requiring it to exist."""
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("The recorded source cache path must be an absolute path without parent traversal")
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Symlink paths are forbidden for the recorded source cache")
    if path.exists():
        N.safe_path(path, directory=True)
    return path


def verify_extension(old, new, previous_config_path, projection_path, decision_path, frozen):
    before, after = old["config"], new
    if ({k: v for k, v in before.items() if k != "budget"}
            != {k: v for k, v in after.items() if k != "budget"}):
        raise ValueError("Scientific configuration/registry changed")
    a, b = before["budget"], after["budget"]
    if "operational_extension" in a:
        raise ValueError("This tool permits only one prospective operational extension")
    if {k: v for k, v in a.items() if k not in ALLOWED_BUDGET_CHANGES} != {
            k: v for k, v in b.items() if k not in ALLOWED_BUDGET_CHANGES}:
        raise ValueError("Original start, search cap, concurrency or resource policy changed")
    if (a["total_worker_seconds_cap"], a["wall_seconds_cap"], b["total_worker_seconds_cap"], b["wall_seconds_cap"]) != (43200, 14400, 54000, 18000):
        raise ValueError("Allowance must be the single recorded12/4 to15/5 extension")
    if a["search_worker_seconds_cap"] != 21600 or b["campaign_started_utc"] != a["campaign_started_utc"]:
        raise ValueError("Search cap or original campaign start changed")
    ext = b["operational_extension"]
    expected = {"schema": "ks-improvement-prospective-budget-extension/1",
        "one_prospective_extension": True, "original_total_worker_seconds_cap": 43200,
        "original_wall_seconds_cap": 14400, "search_worker_seconds_cap_unchanged": 21600,
        "original_campaign_started_utc": a["campaign_started_utc"], "previous_budget": a,
        "frozen_selection_identity": frozen["identity"], "frozen_science_unchanged": True,
        "confirmation_scores_used_for_decision": False, "original_start_and_all_spent_time_carried": True}
    if any(ext.get(k) != v for k, v in expected.items()) or b["extension_reason"] != ext["reason"]:
        raise ValueError("Prospective extension provenance is inconsistent")
    previous = read(previous_config_path)
    if previous != before or N.digest(previous_config_path) != ext["previous_configuration_sha256"]:
        raise ValueError("Previous exact configuration does not match source experiment")
    projection = read(projection_path)
    if (N.digest(projection_path) != ext["timing_receipt_sha256"]
            or projection["experiment_identity"] != old["identity"]
            or projection["freeze_identity"] != frozen["identity"]
            or projection.get("confirmation_score_objects_decoded") != 0
            or projection.get("npz_payload_reads") != 0):
        raise ValueError("Extension lacks its bound timing-only projection")
    decision = read(decision_path)
    if (decision.get("schema") != "ks-improvement-budget-extension-decision/1"
            or decision["identity"] != N.identity({k: v for k, v in decision.items() if k != "identity"})
            or decision["timing_receipt_sha256"] != ext["timing_receipt_sha256"]
            or decision["frozen_selection_identity"] != frozen["identity"]
            or decision["previous_configuration_sha256"] != ext["previous_configuration_sha256"]
            or decision["original_campaign_started_utc"] != a["campaign_started_utc"]
            or decision.get("confirmation_scores_used_for_decision") is not False
            or decision.get("one_prospective_extension") is not True
            or decision.get("extended_caps") != {"total_worker_seconds_cap": 54000, "wall_seconds_cap": 18000, "search_worker_seconds_cap": 21600}
            or decision.get("previous_caps") != {"total_worker_seconds_cap": 43200, "wall_seconds_cap": 14400, "search_worker_seconds_cap": 21600}):
        raise ValueError("Public decision receipt does not bind this exact extension")
    if not date(frozen["created_utc"]) <= date(ext["authorized_utc"]) <= datetime.now(timezone.utc):
        raise ValueError("Extension authorization must follow the original selection freeze")
    return ext, decision


def source_records(source, cache):
    old = read(source/"experiment.json"); frozen = read(source/"freeze.json")
    if old["identity"] != N.identity({"config": old["config"], "source": old["source"], "cache": old["cache_identity"]}):
        raise ValueError("Original experiment identity mismatch")
    if (frozen["identity"] != N.identity({k: v for k, v in frozen.items() if k != "identity"})
            or frozen["experiment_identity"] != old["identity"] or frozen.get("test_access") is not False):
        raise ValueError("Original frozen selection identity mismatch")
    registry = {(t, c["id"]): c for t, rs in old["config"]["registry"]["tracks"].items() for c in rs["candidates"]}
    jobs = {}; ledgers = {}
    for p in sorted((source/"jobs").glob("*.json")):
        j = read(p)
        if p.stem != j["job_id"]:
            raise ValueError("Original job filename mismatch")
        jobs[j["job_id"]] = j
    for p in sorted((source/"ledger").glob("*.json")):
        value = read(p); ledgers[p.stem] = value
        if value.get("job"):
            j = value["job"]
            if p.stem != j["job_id"] or (p.stem in jobs and jobs[p.stem] != j):
                raise ValueError("Ledger job disagrees with immutable job evidence")
            jobs[p.stem] = j
    for jid, j in jobs.items():
        if (jid != N.identity({k: v for k, v in j.items() if k != "job_id"})[:24]
                or j["experiment_identity"] != old["identity"] or j["source"] != old["source"]
                or j["candidate"] != registry[(j["track"], j["candidate"]["id"])]
                or j["fit_key"] != N.identity({"track": j["track"], "effective_config": j["candidate"]["config"], "trajectory": int(j["trajectory"]), "seed": int(j["seed"])})[:32]
                or Path(j["output"]).absolute() != source or Path(j["cache"]).absolute() != cache
                or j.get("partition") != "validation"):
            raise ValueError("Original job/source/cache/candidate identity mismatch")
    terminals = {}
    for p in sorted((source/"terminal").glob("*.json")):
        value = read(p); j = jobs.get(p.stem)
        if (j is None or value.get("job_id") != p.stem or value.get("state") not in TERMINAL
                or value.get("job_identity") != N.identity({k: v for k, v in j.items() if k != "job_id"})):
            raise ValueError("Original immutable terminal binding/state mismatch")
        terminals[p.stem] = value
    attempts = {}
    for p in sorted((source/"attempts").glob("*.json")):
        value = read(p); seconds = value.get("elapsed_seconds")
        if (value.get("state") not in {"complete", "interrupted"} or not value.get("finished_utc")
                or not isinstance(value.get("returncode"), int) or isinstance(value["returncode"], bool)
                or not isinstance(seconds, (int, float)) or isinstance(seconds, bool)
                or not math.isfinite(seconds) or seconds < 0):
            raise ValueError("Every historical attempt needs its closed exact charge; no fabricated closure")
        attempts[p.name] = value
    if not attempts:
        raise ValueError("No spent-work records to preserve")
    return old, frozen, jobs, terminals, attempts, ledgers


def reviewed_fit_sidecars(source, experiment, manifest_path):
    """Bind opaque historical bytes without interpreting them as checkpoints."""
    if manifest_path is None:
        return {}, None
    manifest = read(manifest_path)
    historical_path = source/"migration"/"receipt.json"
    if (manifest.get("schema") != "ks-reviewed-fit-sidecars/1"
            or manifest.get("identity") != N.identity({k: v for k, v in manifest.items() if k != "identity"})
            or manifest.get("source_experiment_identity") != experiment["identity"]
            or manifest.get("historical_migration_receipt_sha256") != N.digest(historical_path)):
        raise ValueError("Reviewed sidecar manifest/source/historical receipt binding mismatch")
    historical = read(historical_path)
    if historical.get("destination_experiment") != experiment["identity"]:
        raise ValueError("Historical sidecar receipt belongs to a different experiment")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Reviewed sidecar manifest must list its exact opaque files")
    result = {}
    for entry in entries:
        name = entry.get("path")
        size = entry.get("bytes"); digest = entry.get("sha256")
        if (not isinstance(name, str)
                or not re.fullmatch(r"fits/[0-9a-f]{32}/\.fit\.npz\.[A-Za-z0-9]{6}", name)
                or name in result or not isinstance(size, int) or isinstance(size, bool) or size < 0
                or not isinstance(digest, str) or not N.SHA.fullmatch(digest)
                or entry.get("disposition") != "preserve_opaque_outside_active_fits"):
            raise ValueError("Sidecar listing must name unique reviewed noncanonical fit bytes")
        prior = historical.get("files", {}).get(name, {})
        if prior.get("bytes") != size or prior.get("sha256") != digest:
            raise ValueError("Reviewed sidecar bytes/hash differ from the historical receipt")
        path = N.safe_path(source/name)
        if path.stat().st_size != size:
            raise ValueError("Reviewed sidecar byte count changed")
        archive = "migration/source_records/fit_sidecars/"+path.parent.name+"/"+path.name[1:]+".opaque"
        N.safe_member(archive)
        result[name] = {"source_path": name, "archive_path": archive,
            "bytes": size, "sha256": digest,
            "disposition": "preserve_opaque_outside_active_fits"}
    return result, {"manifest_sha256": N.digest(manifest_path),
        "manifest_identity": manifest["identity"],
        "historical_migration_receipt_sha256": manifest["historical_migration_receipt_sha256"],
        "creator": manifest.get("creator", "unproven"),
        "interpretation": "Opaque historical bytes, preserved by hash; never loaded or adopted as canonical checkpoint state."}


def reviewed_inference_sidecars(source, experiment, manifest_path):
    """Authorize archival of one exact, independently examined historical pair."""
    if manifest_path is None:
        return {}, None
    manifest = read(manifest_path); historical_path = source/"migration"/"receipt.json"
    if (manifest.get("schema") != "ks-reviewed-inference-sidecars/1"
            or manifest.get("identity") != N.identity({k:v for k,v in manifest.items() if k != "identity"})
            or manifest.get("source_experiment_identity") != experiment["identity"]
            or manifest.get("historical_migration_receipt_sha256") != N.digest(historical_path)):
        raise ValueError("Reviewed inference-sidecar manifest binding mismatch")
    historical = read(historical_path)
    entries = manifest.get("fits")
    if historical.get("destination_experiment") != experiment["identity"] or not isinstance(entries,list) or len(entries) != 1:
        raise ValueError("Exactly one historically bound inference-sidecar review is required")
    entry = entries[0]; key = entry.get("fit_key")
    if (not isinstance(key,str) or not N.FIT.fullmatch(key)
            or entry.get("disposition") != "preserve_auxiliary_only_outside_active_fits"
            or entry.get("inference_restart_state_available_after_archival") is not False
            or entry.get("qualification_changed") is not False):
        raise ValueError("Reviewed auxiliary disposition or fit identity is invalid")
    for field, name in (("native",N.NATIVE),("chain",N.CHAIN),("phase",N.PHASE)):
        record = entry[field]; relative = f"fits/{key}/{name}"
        expected = {k:record.get(k) for k in ("bytes","sha256")}
        if (record.get("path") != relative or not isinstance(expected["bytes"],int)
                or isinstance(expected["bytes"],bool) or expected["bytes"] < 0
                or not isinstance(expected["sha256"],str) or not N.SHA.fullmatch(expected["sha256"])
                or entry.get("historical_entries",{}).get(relative) != expected
                or historical.get("files",{}).get(relative) != expected
                or N.safe_path(source/relative).stat().st_size != expected["bytes"]):
            raise ValueError("Reviewed native/auxiliary historical bytes do not match")
    native = entry["native"]; receipt_relative = f"fits/{key}/receipt.json"
    if (native.get("receipt_path") != receipt_relative
            or N.digest(N.safe_path(source/receipt_relative)) != native.get("receipt_sha256")
            or historical.get("files",{}).get(receipt_relative,{}).get("sha256") != native.get("receipt_sha256")):
        raise ValueError("Reviewed completed native receipt does not match")
    comparison = entry["prefix_comparison"]; filename = comparison.get("receipt_file")
    if (comparison.get("passed") is not True or not isinstance(filename,str)
            or Path(filename).name != filename):
        raise ValueError("An exact retained-prefix comparison receipt is required")
    N.safe_member(filename); proof_path = N.safe_path(Path(manifest_path).parent/filename)
    if N.digest(proof_path) != comparison.get("receipt_sha256"):
        raise ValueError("Retained-prefix comparison receipt checksum mismatch")
    proof = read(proof_path)
    if (proof.get("schema") != "ks-native-versus-auxiliary-prefix-check/1"
            or proof.get("source_experiment_identity") != experiment["identity"]
            or proof.get("fit_key") != key or proof.get("native_fit_identity") != native["native_fit_identity"]
            or proof.get("native_posterior_state_counts") != native["posterior_state_counts"]
            or proof.get("native_qualified") != native["qualified"]
            or proof.get("source_files_unchanged_during_read") is not True
            or proof.get("all_native_chain_phase_internal_checksums_verified_by_load_state") is not True
            or proof.get("native_vs_chain_sampler_equal") is not True
            or proof.get("exact_completed_inference_restart_available_from_auxiliary_pair") is not False):
        raise ValueError("Retained-prefix proof does not support this completed native artifact")
    for field,name in (("native",N.NATIVE),("chain",N.CHAIN),("phase",N.PHASE)):
        if any(proof.get("files",{}).get(name,{}).get(k) != entry[field][k] for k in ("bytes","sha256")):
            raise ValueError("Retained-prefix proof binds different artifact bytes")
    if proof.get("files",{}).get("receipt.json",{}).get("sha256") != native["receipt_sha256"]:
        raise ValueError("Retained-prefix proof binds a different native receipt")
    for name in ("retained_state_prefix_equal_per_chain","retained_log_density_prefix_equal_per_chain"):
        values = proof.get(name,{})
        if set(values) != set(entry["chain"]["progress"]) or not all(v is True for v in values.values()):
            raise ValueError("Not every retained chain prefix was independently verified")
    archives = {name:f"migration/source_records/inference_sidecars/{key}/{name}"
        for name in (N.CHAIN,N.PHASE)}
    return {key:{"entry":entry,"archives":archives,"proof_path":proof_path}}, {
        "manifest_sha256":N.digest(manifest_path),"manifest_identity":manifest["identity"],
        "historical_migration_receipt_sha256":manifest["historical_migration_receipt_sha256"]}


def validate_reviewed_native_and_archival_pair(source, row, job, experiment, review):
    """Keep the strict validator unchanged and recognize only its named failure."""
    entry = review["entry"]
    if not row["native_present"] or not row["receipt_present"]:
        raise ValueError("Reviewed auxiliary archival requires a completed native fit and receipt")
    try:
        N.validate_fit(source,row,job,experiment,"all-fit-checkpoints")
    except ValueError as error:
        if str(error) != COMPLETED_PHASE_CHAIN_ERROR:
            raise
    else:
        raise ValueError("Reviewed auxiliary pair no longer has its recorded consistency failure")
    summary, checked = N.validate_fit(source,row,job,experiment,"selected-only")
    if (not summary.get("native_receipt_verified") or not summary.get("forecast_state_available")
            or summary["native_fit_identity"] != entry["native"]["native_fit_identity"]
            or summary["qualified"] != entry["native"]["qualified"]
            or checked[N.NATIVE]["sha256"] != entry["native"]["sha256"]):
        raise ValueError("Reviewed native forecast authority failed independent verification")
    states = {}
    for field,name,kind in (("chain",N.CHAIN,"mcmc_checkpoint"),("phase",N.PHASE,"ks_bayes_fit_phase")):
        value = N.validate_npz(source/"fits"/row["fit_key"]/name,kind)
        if value["sha256"] != entry[field]["sha256"] or value["bytes"] != entry[field]["bytes"]:
            raise ValueError("Reviewed inference-sidecar bytes changed")
        states[field] = value["state"]
        checked[name] = {k:v for k,v in value.items() if k != "state"}
    chain = states["chain"]; phase = N.plain(states["phase"])
    completed = N.validate_npz(source/"fits"/row["fit_key"]/N.NATIVE,"ks_native_fit")
    native_core = completed["state"]["core"]
    native_counts = [len(values) for values in native_core["posterior"]["states"]]
    native_sampler = N.plain(native_core["sampler"])
    if (completed["sha256"] != checked[N.NATIVE]["sha256"]
            or native_counts != entry["native"]["posterior_state_counts"]
            or len(native_counts) != native_sampler["chains"]
            or any(count != native_sampler["draws"] for count in native_counts)
            or native_sampler != N.plain(chain["config"])
            or native_sampler["draws"] != entry["phase"]["target_draws"]):
        raise ValueError("Reviewed native posterior counts/sampler target are not complete and identical")
    progress = {key:{"iteration":value["iteration"],"retained_draws":len(value["retained_states"])}
        for key,value in chain["chains"].items()}
    if (progress != entry["chain"]["progress"] or phase["phase"] != "complete"
            or phase["phase"] != entry["phase"]["phase"]
            or phase["target_draws"] != entry["phase"]["target_draws"]
            or phase["checkpoint"]["sha256"] != entry["phase"]["expected_chain_sha256"]
            or not 1 <= len(phase["gates"]) <= 2
            or bool(phase["gates"][-1]["passed"]) != summary["qualified"]
            or bool(phase["gates"][-1]["passed"]) != entry["phase"]["qualified"]):
        raise ValueError("Reviewed auxiliary progress/phase/qualification changed")
    summary.pop("inference_sidecars_intentionally_excluded",None)
    summary.update(inference_restart_state_available=False,inference_sidecars_archived_only=True,
        reviewed_auxiliary_consistency_failure=COMPLETED_PHASE_CHAIN_ERROR,
        archived_chain_progress=progress,qualification_changed=False)
    return summary,checked


def closure_proof(path, session_id, source, attempts, terminals):
    receipt = read(path)
    if receipt.get("source") != "actual_unified_exec_write_stdin_completion_results":
        raise ValueError("A genuine observed coordinator-completion receipt is required")
    matching = [r for r in receipt.get("sessions", []) if r.get("session_id") == session_id
        and Path(r.get("root", "")).absolute() == source and r.get("observed_completed") is True
        and isinstance(r.get("exit_code"), int) and not isinstance(r["exit_code"], bool)]
    if len(matching) != 1:
        raise ValueError("The explicitly selected coordinator session was not observed closed")
    last = max([date(r["finished_utc"]) for r in attempts.values()]
               + [date(r["finished_utc"]) for r in terminals.values()])
    if date(receipt["observed_utc"]) < last:
        raise ValueError("Closure observation predates source work; use the actual latest closed session")
    return {"session_id": session_id, "receipt_sha256": N.digest(path),
            "observed_utc": receipt["observed_utc"], "latest_source_finish_utc": last.isoformat()}


def verify_runtime_cache(old, cache):
    from threadpoolctl import threadpool_limits
    from bayes_stesn.applications.ks_improvement_runner import source_identity
    from bayes_stesn.applications.ks_data import audit_cache
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError("Set every recorded numerical thread environment variable to1")
    with threadpool_limits(1):
        current_source = source_identity()
        if current_source != old["source"]:
            raise ValueError("Numerical source, packages or native runtime changed")
        audit = audit_cache(cache)
        if not audit["passed"] or N.identity(audit) != old["cache_identity"]:
            raise ValueError("Verified original KS cache identity changed")
    return {"numerical_source_runtime_identical": True, "cache_identity": old["cache_identity"]}


def forecast_metadata(path):
    # Do not decode nested confirmation performance metrics for this operation.
    result = {}; before = N.fingerprint(path.stat())
    with path.open() as stream:
        for line in stream:
            match = SCALAR.match(line)
            if match and match[1] in FORECAST_FIELDS:
                result[match[1]] = json.loads(match[2].rstrip().removesuffix(","))
    if N.fingerprint(path.stat()) != before:
        raise RuntimeError("Forecast receipt changed during metadata read")
    return result


def validate_forecasts(source, jobs, terminals, native_summaries):
    by_key = defaultdict(list)
    for j in jobs.values():
        by_key[j["fit_key"]].append(j)
    found = {}; counts = Counter()
    for namespace in ("forecasts", "precision_forecasts"):
        for p in sorted((source/namespace).glob("*/*.json")):
            N.safe_path(p); f = forecast_metadata(p)
            if f.get("state") not in {"complete", "failed"}:
                raise ValueError("Nonterminal forecast receipt in immutable forecast storage")
            possible = [j for j in by_key.get(p.parent.name, []) if
                j.get("forecast_namespace", "main") == ("precision" if namespace == "precision_forecasts" else "main")
                and j["paths"] == f.get("paths") and j["forecast_seed"] == f.get("forecast_seed")
                and f.get("first_target") in j["origins"]]
            if not possible:
                raise ValueError("Forecast has no bound original job")
            j = possible[0]; key = N.identity({"native_fit": f["native_fit"], "first": f["first_target"],
                "history": j["history"], "horizon": 100, "paths": j["paths"],
                "forecast_seed": j["forecast_seed"], "partition": "validation",
                "score_source": j["source"]["application"]["ks_metrics.py"]})
            pairs = {"fit_key": j["fit_key"], "track": j["track"], "trajectory": j["trajectory"],
                "candidate_id": j["candidate"]["id"], "seed": j["seed"], "history": j["history"],
                "paths": j["paths"], "forecast_seed": j["forecast_seed"], "partition": "validation",
                "forecast_namespace": j.get("forecast_namespace", "main")}
            native = native_summaries.get(j["fit_key"])
            if (any(f.get(k) != v for k, v in pairs.items()) or f["identity"] != key
                    or p.name != f'{f["first_target"]:05d}_{f["paths"]}_{key[:12]}.json'
                    or not f.get("qualified") or native is None or not native.get("qualified")
                    or f["native_fit"] != native.get("native_fit_identity")):
                raise ValueError("Forecast/native identity mismatch")
            found[(namespace, j["fit_key"], j["paths"], j["forecast_seed"], f["first_target"])] = f
            counts[f["state"]] += 1
    for jid, terminal in terminals.items():
        j = jobs[jid]
        if j["stage"] == "preflight":
            if terminal["state"] == "complete" and read(source/"preflight"/(j["fit_key"]+".json")) != terminal["result"]:
                raise ValueError("Preflight terminal differs from retained feature-only evidence")
            continue
        namespace = "precision_forecasts" if j.get("forecast_namespace") == "precision" else "forecasts"
        items = [found.get((namespace, j["fit_key"], j["paths"], j["forecast_seed"], first)) for first in j["origins"]]
        complete = sum(f is not None and f["state"] == "complete" for f in items)
        failed = sum(f is not None and f["state"] == "failed" for f in items)
        if ((terminal["state"] == "complete" and complete != len(j["origins"]))
                or ("completed_origins" in terminal and (complete != terminal["completed_origins"] or failed != terminal["failed_origins"]))):
            raise ValueError("Terminal completed/failed origin counts lack immutable evidence")
        if terminal["state"] == "inference-unqualified" and native_summaries.get(j["fit_key"], {}).get("qualified") is not False:
            raise ValueError("Unqualified terminal does not match its native fit")
    return dict(counts)


def inventory(source, jobs, sidecars=None, inference_sidecars=None):
    files = {}; skipped = []
    sidecars = sidecars or {}
    inference_sidecars = inference_sidecars or {}
    def add(path, name, expected_sha256=None):
        N.safe_path(path); N.safe_member(name)
        if name in files:
            raise ValueError("Duplicate migration destination")
        files[name] = {"source": path, "bytes": path.stat().st_size, "fingerprint": N.fingerprint(path.stat())}
        if expected_sha256 is not None:
            files[name]["expected_sha256"] = expected_sha256
    for folder in sorted((source/"fits").iterdir()):
        N.safe_path(folder, directory=True)
        for p in sorted(folder.iterdir()):
            reviewed_pair = inference_sidecars.get(folder.name)
            if reviewed_pair and p.name in (N.CHAIN,N.PHASE):
                field = "chain" if p.name == N.CHAIN else "phase"
                add(p,reviewed_pair["archives"][p.name],reviewed_pair["entry"][field]["sha256"])
            elif p.name in N.CANONICAL or p.name == "receipt.json" or re.fullmatch(r"synchronization_h[0-9]+\.json", p.name):
                add(p, p.relative_to(source).as_posix())
            elif p.relative_to(source).as_posix() in sidecars:
                reviewed = sidecars[p.relative_to(source).as_posix()]
                add(p, reviewed["archive_path"], reviewed["sha256"])
            elif p.name == "writer.lock" or (p.name.startswith(".") and (".bstesn-" in p.name or p.name.endswith(".tmp"))):
                skipped.append(p.relative_to(source).as_posix())
            else:
                raise ValueError("Unknown fit sidecar; preserve and explicitly review before migration")
    for sub in ("forecasts", "precision_forecasts", "preflight", "selection", "attempts", "examples", "logs"):
        if not (source/sub).exists():
            continue
        for p in sorted((source/sub).rglob("*")):
            if p.is_dir():
                N.safe_path(p, directory=True); continue
            if p.name.startswith(".") or p.name.endswith(".tmp"):
                skipped.append(p.relative_to(source).as_posix()); continue
            add(p, p.relative_to(source).as_posix())
    add(source/"history_policy.json", "history_policy.json")
    for name in ("experiment.json", "freeze.json", "history_policy.json", "status.json", "campaign.log"):
        p = source/name
        if p.exists():add(p, "migration/source_records/"+name)
    for sub in ("jobs", "terminal", "ledger", "migration", "selection"):
        if not (source/sub).exists():continue
        for p in sorted((source/sub).rglob("*")):
            if p.is_dir():N.safe_path(p, directory=True);continue
            if p.name.startswith(".") or p.name.endswith((".tmp", ".lock")):
                skipped.append(p.relative_to(source).as_posix());continue
            add(p, "migration/source_records/"+p.relative_to(source).as_posix())
    return files, skipped


def copy_verified(row, destination, reserve, expected=None):
    source = row["source"]; N.safe_path(source)
    if N.fingerprint(source.stat()) != row["fingerprint"]:
        raise RuntimeError("Source changed before migration copy")
    destination.parent.mkdir(parents=True, exist_ok=True)
    import hashlib
    digest = hashlib.sha256(); count = 0
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as incoming, destination.open("xb") as outgoing:
        while block := incoming.read(N.MIB):
            if shutil.disk_usage(destination.parent).free-len(block) < reserve:
                raise RuntimeError("Original disk reserve prevents completing migration")
            outgoing.write(block); digest.update(block); count += len(block)
        outgoing.flush(); os.fsync(outgoing.fileno())
    shutil.copystat(source, destination, follow_symlinks=False)
    actual = digest.hexdigest()
    if (count != row["bytes"] or (expected is not None and actual != expected)
            or N.digest(destination) != actual or N.fingerprint(source.stat()) != row["fingerprint"]):
        raise ValueError("Copied artifact byte count/hash/fingerprint mismatch")
    return {"bytes": count, "sha256": actual, "source_relative_scope": "original run retained unchanged"}


def publish_new(staging, destination):
    # Linux renameat2(RENAME_NOREPLACE) refuses even a concurrently created empty
    # destination. There is no unsafe overwrite fallback.
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, "renameat2"):
        raise RuntimeError("Atomic no-replace directory publication requires Linux renameat2")
    fn = libc.renameat2; fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    fn.restype = ctypes.c_int
    if fn(-100, os.fsencode(staging), -100, os.fsencode(destination), 1) != 0:
        code = ctypes.get_errno(); raise OSError(code, os.strerror(code), str(destination))
    descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def migrate(source, destination, config, cache, closure, session_id, projection, decision, previous_config, *, apply=False, source_cache_path=None, fit_sidecar_manifest=None, inference_sidecar_manifest=None):
    source = N.safe_path(source, directory=True); cache = N.safe_path(cache, directory=True)
    original_cache = cache if source_cache_path is None else recorded_cache_path(source_cache_path)
    cache_relocated = original_cache != cache
    destination = Path(os.path.abspath(destination))
    if (destination.exists() or source == destination or source in destination.parents or destination in source.parents
            or any(p.is_symlink() for p in (destination, *destination.parents))):
        raise ValueError("A new distinct destination outside the original run is required")
    N.safe_path(destination.parent, directory=True)
    with ExitStack() as stack:
        locks = N.acquire_locks(source, stack)
        old, frozen, jobs, terminals, attempts, ledgers = source_records(source, original_cache)
        closed = closure_proof(closure, session_id, source, attempts, terminals)
        new_config = read(config)
        ext, approved = verify_extension(old, new_config, previous_config, projection, decision, frozen)
        if N.digest(config) != approved["extended_configuration_sha256"]:
            raise ValueError("Exact budget-v2 config bytes differ from public decision")
        new_identity = N.identity({"config": new_config, "source": old["source"], "cache": old["cache_identity"]})
        new_exp = {**old, "identity": new_identity, "config": new_config, "created_utc": utc()}
        new_freeze = {**frozen, "experiment_identity": new_identity}
        new_freeze["identity"] = N.identity({k: v for k, v in new_freeze.items() if k != "identity"})
        if {k: v for k, v in frozen.items() if k not in {"experiment_identity", "identity"}} != {
                k: v for k, v in new_freeze.items() if k not in {"experiment_identity", "identity"}}:
            raise AssertionError("Scientific freeze changed")
        sidecars, sidecar_binding = reviewed_fit_sidecars(source, old, fit_sidecar_manifest)
        inference_sidecars, inference_binding = reviewed_inference_sidecars(source, old, inference_sidecar_manifest)
        files, skipped = inventory(source, jobs, sidecars, inference_sidecars)
        preparation = float(old["config"]["budget"].get("preparation_worker_seconds", 0))
        charge = sum(float(r["elapsed_seconds"]) for r in attempts.values())
        stats = {"schema": SCHEMA, "apply": apply, "source_experiment": old["identity"],
            "destination_experiment": new_identity, "source_freeze_identity": frozen["identity"],
            "destination_freeze_identity": new_freeze["identity"], "original_freeze_created_utc": frozen["created_utc"],
            "wall_budget_start": old["config"]["budget"]["campaign_started_utc"],
            "attempt_files": len(attempts), "attempt_elapsed_seconds": charge,
            "preparation_worker_seconds": preparation, "carried_worker_seconds": charge+preparation,
            "terminal_states": dict(Counter(r["state"] for r in terminals.values())),
            "job_count": len(jobs), "pending_or_interrupted_job_count": len(jobs)-len(terminals),
            "copy_files": len(files), "copy_bytes": sum(r["bytes"] for r in files.values()),
            "skipped_locks_and_staging_count": len(skipped), "closure": closed,
            "source_locks_acquired": len(locks), "scientific_recomputation": False,
            "cache_path_relocated": cache_relocated,
            "cache_relocation": {"explicit_source_path": source_cache_path is not None,
                "source_job_cache": str(original_cache), "destination_cache": str(cache),
                "expected_cache_identity": old["cache_identity"],
                "same_verified_cache_and_runtime_required": True},
            "reviewed_fit_sidecar_count": len(sidecars),
            "reviewed_fit_sidecar_bytes": sum(r["bytes"] for r in sidecars.values()),
            "reviewed_fit_sidecars": {"binding": sidecar_binding,
                "files": list(sidecars.values()), "canonical_checkpoint_adoption": False},
            "reviewed_inference_sidecars": {"binding":inference_binding,
                "fits":[{"fit_key":key,"archive_paths":review["archives"],
                    "native_sha256":review["entry"]["native"]["sha256"],
                    "inference_restart_state_available":False,"qualification_changed":False}
                    for key,review in inference_sidecars.items()]},
            "confirmation_score_objects_decoded": 0, "source_root_modified_or_deleted": False,
            "timing_receipt_sha256": N.digest(projection), "decision_receipt_sha256": N.digest(decision),
            "extended_configuration_sha256": N.digest(config), "binary_payload_reads": 0 if not apply else "guarded checksum validation and exact copy only"}
        if not apply:return stats
        reserve = max(2*1024**3, int(new_config["budget"]["disk_reserve_bytes"]))
        if shutil.disk_usage(destination.parent).free < reserve+stats["copy_bytes"]+128*N.MIB:
            raise RuntimeError("New run copy cannot preserve the original disk reserve")
        runtime = verify_runtime_cache(old, cache)
        known = {}; native_summaries = {}
        by_fit = defaultdict(list)
        for j in jobs.values():by_fit[j["fit_key"]].append(j)
        for folder in sorted((source/"fits").iterdir()):
            present = [n for n in N.CANONICAL if (folder/n).exists()]
            if not present:continue
            if folder.name not in by_fit:raise ValueError("Canonical checkpoint has no original job")
            j = by_fit[folder.name][0]
            row = {"fit_key": folder.name, "track": j["track"], "candidate_id": j["candidate"]["id"],
                "trajectory": j["trajectory"], "seed": j["seed"], "roles": [],
                "native_present": (folder/N.NATIVE).exists(), "receipt_present": (folder/"receipt.json").exists()}
            if folder.name in inference_sidecars:
                summary, checked = validate_reviewed_native_and_archival_pair(source,row,j,old,inference_sidecars[folder.name])
            else:
                summary, checked = N.validate_fit(source, row, j, old, "all-fit-checkpoints")
            native_summaries[folder.name] = summary
            for name, check in checked.items():known[f"fits/{folder.name}/{name}"] = check["sha256"]
        forecast_counts = validate_forecasts(source, jobs, terminals, native_summaries)
        for path in (source/"examples").glob("*.npz"):
            with zipfile.ZipFile(N.safe_path(path)) as archive:
                if archive.testzip() is not None:raise ValueError("Retained example ZIP CRC failure")
        staging = Path(tempfile.mkdtemp(prefix="."+destination.name+"-budget-extension-", dir=destination.parent))
        hashes = {}
        for name, row in sorted(files.items()):
            hashes[name] = copy_verified(row, staging/name, reserve, known.get(name, row.get("expected_sha256")))
        # Source jobs/terminals remain verbatim under migration/source_records.
        mapping = {}; destination_states = Counter()
        for old_id, j in sorted(jobs.items()):
            adopted = {**j, "output": str(destination), "experiment_identity": new_identity}
            if cache_relocated:
                adopted["cache"] = str(cache)
            adopted["job_id"] = N.identity({k: v for k, v in adopted.items() if k != "job_id"})[:24]
            new_id = adopted["job_id"]
            if new_id in mapping.values():raise ValueError("Job identity collision")
            mapping[old_id] = new_id
            write_new(staging/"jobs"/(new_id+".json"), adopted)
            if old_id in terminals:
                old_terminal = terminals[old_id]
                terminal = {**old_terminal, "job_id": new_id,
                    "job_identity": N.identity({k: v for k, v in adopted.items() if k != "job_id"})}
                write_new(staging/"terminal"/(new_id+".json"), terminal)
                ledger = {"state": terminal["state"], "job": adopted,
                    "adoption": {"source_job_id": old_id, "source_terminal_sha256": N.digest(source/"terminal"/(old_id+".json")), "scientific_recomputation": False}}
            else:
                ledger = {"state": "pending", "job": adopted, "reason": "closed_source_attempt_not_terminal",
                    "source_job_id": old_id, "source_ledger_state": ledgers.get(old_id, {}).get("state")}
            destination_states[ledger["state"]] += 1
            write_new(staging/"ledger"/(new_id+".json"), ledger)
        write_new(staging/"experiment.json", new_exp); write_new(staging/"freeze.json", new_freeze)
        for folder in ("jobs", "terminal", "ledger", "attempts", "selection", "preflight", "forecasts", "precision_forecasts", "examples", "logs", "fits"):
            (staging/folder).mkdir(exist_ok=True)
        for folder in sorted((source/"fits").iterdir()):
            target = staging/"fits"/folder.name; target.mkdir(parents=True, exist_ok=True)
            (target/"writer.lock").touch(exist_ok=False)
        (staging/"scheduler.lock").touch(exist_ok=False)
        extra_evidence = [("budget_extension_decision.json", decision), ("timing_projection.json", projection),
                           ("closed_sessions.json", closure), ("previous_configuration.json", previous_config),
                           ("extended_configuration.json", config)]
        if fit_sidecar_manifest is not None:
            extra_evidence.append(("reviewed_fit_sidecars.json", fit_sidecar_manifest))
        if inference_sidecar_manifest is not None:
            extra_evidence.append(("reviewed_inference_sidecars.json",inference_sidecar_manifest))
            extra_evidence.extend(("inference_sidecar_proofs/"+r["proof_path"].name,r["proof_path"])
                for r in inference_sidecars.values())
        for name, path in extra_evidence:
            p = Path(path);row = {"source": p, "bytes": p.stat().st_size, "fingerprint": N.fingerprint(p.stat())}
            hashes["migration/"+name] = copy_verified(row, staging/"migration"/name, reserve)
        after_attempts = {p.name: read(p) for p in sorted((staging/"attempts").glob("*.json"))}
        if after_attempts != attempts or preparation != new_config["budget"].get("preparation_worker_seconds", 0):
            raise ValueError("Attempt charges or one-time preparation changed during migration")
        carried = sum(float(r["elapsed_seconds"]) for r in after_attempts.values())+preparation
        if carried != charge+preparation:raise ValueError("Cumulative work accounting changed")
        for name, row in files.items():
            if N.fingerprint(row["source"].stat()) != row["fingerprint"]:
                raise RuntimeError("Original source changed before destination publication")
        receipt = {**stats, "created_utc": utc(), "passed": True, "runtime_and_cache": runtime,
            "files": hashes, "job_id_mapping": mapping, "skipped_source_locks_and_staging": skipped,
            "source_locks": locks, "destination_terminal_states": dict(destination_states),
            "native_checkpoint_count": sum(s.get("forecast_state_available", False) for s in native_summaries.values()),
            "forecast_receipt_states": forecast_counts, "attempt_bytes_preserved": True,
            "native_forecast_and_example_bytes_preserved": True, "history_and_selection_bytes_preserved": True,
            "freeze_scientific_fields_and_original_created_utc_preserved": True,
            "new_job_identities_created_for_relocation": True,
            "all_spent_work_including_prior_migration_attempts_carried_exactly": True,
            "native_identity_guards_remain_required": True}
        write_new(staging/"migration"/"receipt.json", receipt)
        publish_new(staging, destination)
        return {k: v for k, v in receipt.items() if k not in {"files", "job_id_mapping", "source_locks", "skipped_source_locks_and_staging"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "destination", "config", "cache", "closed-session-receipt", "projection-receipt", "decision-receipt"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--source-cache-path", type=Path,
        help="Explicit absolute cache path recorded in source jobs; may be missing. --cache must verify as the identical restored cache.")
    parser.add_argument("--fit-sidecar-manifest", type=Path,
        help="Explicit size/hash-bound review of opaque historical fit sidecars to preserve outside active fits.")
    parser.add_argument("--inference-sidecar-manifest",type=Path,
        help="One exact reviewed inconsistent auxiliary pair to archive while retaining its independently verified completed native fit.")
    parser.add_argument("--closed-session-id", type=int, required=True)
    parser.add_argument("--previous-config", type=Path, default=Path(__file__).resolve().parents[1]/"configs/applications/ks_improvement_v1.json")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    result = migrate(args.source, args.destination, args.config, args.cache, args.closed_session_receipt,
        args.closed_session_id, args.projection_receipt, args.decision_receipt, args.previous_config,
        apply=args.apply, source_cache_path=args.source_cache_path,
        fit_sidecar_manifest=args.fit_sidecar_manifest,inference_sidecar_manifest=args.inference_sidecar_manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":main()
