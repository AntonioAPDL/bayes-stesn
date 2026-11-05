#!/usr/bin/env python3
"""Migrate a closed frozen KS run under one prospective operational allowance.

Default: metadata-only dry-run, requiring genuine closure evidence and all
existing source locks. --apply verifies current numerical source/runtime/cache,
then copies canonical checkpoint and evidence bytes to a new atomic run root.
Only experiment/output/job identities and their terminal bindings are rebased;
an explicit --source-cache-path also permits relocating the byte-identical cache.
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


def inventory(source, jobs):
    files = {}; skipped = []
    def add(path, name):
        N.safe_path(path); N.safe_member(name)
        if name in files:
            raise ValueError("Duplicate migration destination")
        files[name] = {"source": path, "bytes": path.stat().st_size, "fingerprint": N.fingerprint(path.stat())}
    for folder in sorted((source/"fits").iterdir()):
        N.safe_path(folder, directory=True)
        for p in sorted(folder.iterdir()):
            if p.name in N.CANONICAL or p.name == "receipt.json" or re.fullmatch(r"synchronization_h[0-9]+\.json", p.name):
                add(p, p.relative_to(source).as_posix())
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


def migrate(source, destination, config, cache, closure, session_id, projection, decision, previous_config, *, apply=False, source_cache_path=None):
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
        files, skipped = inventory(source, jobs)
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
            hashes[name] = copy_verified(row, staging/name, reserve, known.get(name))
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
        for name, path in (("budget_extension_decision.json", decision), ("timing_projection.json", projection),
                           ("closed_sessions.json", closure), ("previous_configuration.json", previous_config),
                           ("extended_configuration.json", config)):
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
    parser.add_argument("--closed-session-id", type=int, required=True)
    parser.add_argument("--previous-config", type=Path, default=Path(__file__).resolve().parents[1]/"configs/applications/ks_improvement_v1.json")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    result = migrate(args.source, args.destination, args.config, args.cache, args.closed_session_receipt,
        args.closed_session_id, args.projection_receipt, args.decision_receipt, args.previous_config,
        apply=args.apply, source_cache_path=args.source_cache_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":main()
