#!/usr/bin/env python3
"""Observe a KS run across execution namespaces without modifying the run.

The scientific runner's exact namespace-inode check can be inaccessible from a
separate execution session. Host-visible PID, process start ticks, boot ID and
NSpid still identify the process without treating denied essential reads as
death. This standalone, standard-library tool never imports the model runtime,
signals processes, deletes locks, settles attempts, or writes status receipts.
Lock probes briefly request a nonblocking shared lock and immediately release
it; they establish only whether an exclusive lock was observed at that instant.
The resulting snapshot is observational, not authorization to restart a run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import time


TERMINAL = {"complete", "failed", "inference-unqualified", "resource-blocked"}


def _process_stat(path):
    text = path.read_text()
    tail = text.rsplit(")", 1)[1].split()
    return {"pid": int(text.split(" ", 1)[0]), "state": tail[0],
            "start_ticks": str(int(tail[19]))}


def inspect_process(record, *, proc_root=Path("/proc")):
    """Return active, inactive, or unknown; an optional inode is not essential.

    Inactive requires an absent/dead process or a mismatching lifetime identity.
    Permission failures for boot ID, stat, or NSpid instead return unknown.
    A second stat read detects process disappearance/reuse during inspection.
    """
    result = {"state": "unknown", "reason": "missing_or_invalid_process_receipt",
              "validated": {}, "legacy_match": None}
    if not isinstance(record, dict):
        return result
    try:
        for field in ("pid", "namespace_pid"):
            if type(record[field]) is not int or record[field] <= 0:
                return result
        expected = {"pid": record["pid"], "namespace_pid": record["namespace_pid"],
                    "start_ticks": str(int(record["start_ticks"])),
                    "boot_id": record["boot_id"]}
        if int(expected["start_ticks"]) < 0 or not isinstance(expected["boot_id"], str) or not expected["boot_id"]:
            return result
        if "pid_namespace" in record and (type(record["pid_namespace"]) is not int or record["pid_namespace"] <= 0):
            return result
    except (KeyError, TypeError, ValueError, OverflowError):
        return result
    result["pid"] = expected["pid"]
    folder = Path(proc_root)/str(expected["pid"])
    field = "boot_id"

    def mismatch(reason, observed):
        return {**result, "state": "inactive", "reason": reason,
                "observed_mismatch": observed, "legacy_match": False}

    try:
        boot = (Path(proc_root)/"sys/kernel/random/boot_id").read_text().strip()
        if boot != expected["boot_id"]:
            return mismatch("boot_id_changed", boot)
        result["validated"]["boot_id"] = boot
        field = "stat"
        first = _process_stat(folder/"stat")
        for name in ("pid", "start_ticks"):
            if first[name] != expected[name]:
                return mismatch(name+"_changed", first[name])
            result["validated"][name] = first[name]
        result["process_state"] = first["state"]
        if first["state"] in {"Z", "X", "x"}:
            return mismatch("process_exited_or_zombie", first["state"])
        field = "NSpid"
        status = (folder/"status").read_text().splitlines()
        line = next((line for line in status if line.startswith("NSpid:")), None)
        if line is None:
            return {**result, "reason": "essential_NSpid_unavailable"}
        nspid = [int(value) for value in line.split()[1:]]
        if not nspid or nspid[0] != expected["pid"] or nspid[-1] != expected["namespace_pid"]:
            return mismatch("namespace_pid_changed", nspid)
        result["validated"]["namespace_pid"] = nspid[-1]
        namespace = {"state": "unavailable"}
        try:
            inode = os.stat(folder/"ns/pid").st_ino
            namespace = {"state": "available", "inode": inode}
            if "pid_namespace" in record and inode != record["pid_namespace"]:
                return mismatch("pid_namespace_changed", inode)
        except OSError as error:
            namespace["error"] = type(error).__name__
            namespace["errno"] = error.errno
        result["namespace_inode"] = namespace
        field = "stat_recheck"
        last = _process_stat(folder/"stat")
        if last["pid"] != first["pid"] or last["start_ticks"] != first["start_ticks"]:
            return mismatch("process_changed_during_inspection", last)
        if last["state"] in {"Z", "X", "x"}:
            return mismatch("process_exited_or_zombie", last["state"])
        result.update(state="active", reason="matched_process_lifetime", process_state=last["state"])
        observed = dict(result["validated"])
        if namespace["state"] == "available":
            observed["pid_namespace"] = namespace["inode"]
            result["legacy_match"] = observed == record
        else:
            result["legacy_match"] = False
            result["legacy_disagreement"] = "namespace_inode_unavailable_but_essential_lifetime_fields_match"
        return result
    except (FileNotFoundError, ProcessLookupError) as error:
        if field != "boot_id":
            return {**result, "state": "inactive", "reason": "process_absent",
                    "failed_field": field, "legacy_match": False}
        return {**result, "reason": "essential_field_unavailable", "failed_field": field,
                "error": type(error).__name__, "errno": error.errno}
    except OSError as error:
        return {**result, "reason": "essential_field_unavailable", "failed_field": field,
                "error": type(error).__name__, "errno": error.errno}
    except (ValueError, IndexError) as error:
        return {**result, "reason": "malformed_process_information", "failed_field": field,
                "error": type(error).__name__}


def inspect_lock(path):
    """Observe an exclusive lock without creating a file or waiting for a lock."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return {"state": "missing"}
    except OSError as error:
        return {"state": "unknown", "error": type(error).__name__, "errno": error.errno}
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return {"state": "unknown", "reason": "not_a_regular_lock_file"}
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in {errno.EAGAIN, errno.EWOULDBLOCK}:
                return {"state": "held", "probe": "shared_nonblocking"}
            return {"state": "unknown", "error": type(error).__name__, "errno": error.errno}
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return {"state": "free", "probe": "shared_nonblocking"}
    finally:
        os.close(descriptor)


def _read_record(path, issues, *, required=False):
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value
    except FileNotFoundError:
        if required:
            issues.append({"path": str(path), "reason": "missing_record"})
    except (OSError, ValueError) as error:
        issues.append({"path": str(path), "reason": type(error).__name__})
    return None


def _component(value):
    return isinstance(value, str) and value not in {"", ".", ".."} and Path(value).name == value


def inspect_run(output, *, proc_root=Path("/proc")):
    root = Path(output).resolve()
    issues, processes, locks, jobs, worker_checks = [], [], [], {}, {}
    reported_workers = set()
    campaigns = [root] + sorted(path for path in (root/"deferred").glob("*")
                                if path.is_dir() and (path/"experiment.json").is_file())
    cache = {}

    def check(record):
        key = json.dumps(record, sort_keys=True)
        if key not in cache:
            cache[key] = inspect_process(record, proc_root=proc_root)
        return cache[key]

    coordinator = _read_record(root/"coordinator.json", issues, required=True)
    coordinator_check = check(coordinator.get("process") if coordinator else None)
    processes.append({"role": "coordinator", "recorded_state": (coordinator or {}).get("state"),
                      **coordinator_check})
    for campaign in campaigns:
        relative = str(campaign.relative_to(root))
        locks.append({"kind": "scheduler", "campaign": relative,
                      **inspect_lock(campaign/"scheduler.lock")})
        for path in sorted((campaign/"jobs").glob("*.json")):
            job = _read_record(path, issues, required=True)
            if job is None or not _component(job.get("job_id")) or job["job_id"] != path.stem:
                issues.append({"path": str(path), "reason": "invalid_job_record"})
                continue
            jobs[(relative, job["job_id"])] = job
        for path in sorted((campaign/"attempts").glob("*.json")):
            attempt = _read_record(path, issues, required=True)
            if not attempt or attempt.get("state") != "running":
                continue
            key = (relative, attempt.get("job_id"))
            observed = check(attempt.get("process"))
            worker_checks.setdefault(key, []).append(observed)
            reported_workers.add((*key, json.dumps(attempt.get("process"), sort_keys=True)))
            item = {"role": "worker", "campaign": relative, "job_id": attempt.get("job_id"),
                    "attempt": path.name, **observed}
            if isinstance(attempt.get("started_epoch"), (int, float)):
                item["heartbeat_age_seconds"] = round(time.time()-attempt["started_epoch"]-
                                                       attempt.get("elapsed_seconds", 0), 3)
            processes.append(item)
            job = jobs.get(key, {})
            if _component(job.get("fit_key")):
                locks.append({"kind": "writer", "campaign": relative, "job_id": job["job_id"],
                              **inspect_lock(campaign/"fits"/job["fit_key"]/"writer.lock")})
    states, stages, overrides, unknown_jobs = {}, {}, [], []
    for key, job in jobs.items():
        relative, job_id = key
        campaign = root/relative
        terminal_path = campaign/"terminal"/(job_id+".json")
        before = len(issues)
        terminal = _read_record(terminal_path, issues)
        terminal_unreadable = len(issues) != before
        before = len(issues)
        ledger = _read_record(campaign/"ledger"/(job_id+".json"), issues)
        ledger_unreadable = len(issues) != before
        checks = list(worker_checks.get(key, []))
        if ledger and ledger.get("state") == "running":
            observed = check(ledger.get("process"))
            checks.append(observed)
            worker_key = (*key, json.dumps(ledger.get("process"), sort_keys=True))
            if worker_key not in reported_workers:
                processes.append({"role": "worker", "campaign": relative, "job_id": job_id,
                                  "source": "ledger", **observed})
                reported_workers.add(worker_key)
                if _component(job.get("fit_key")):
                    locks.append({"kind": "writer", "campaign": relative, "job_id": job_id,
                                  **inspect_lock(campaign/"fits"/job["fit_key"]/"writer.lock")})
        recorded_state = (terminal or ledger or {"state": "pending"}).get("state", "unknown")
        state = recorded_state
        if terminal_unreadable or (terminal is None and ledger_unreadable):
            state = "unknown"
        elif terminal is not None:
            payload = {name: value for name, value in job.items() if name != "job_id"}
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False,
                                              separators=(",", ":")).encode()).hexdigest()
            if terminal.get("job_identity") != digest or state not in TERMINAL:
                state = "unknown"
                issues.append({"path": str(terminal_path), "reason": "incompatible_terminal_receipt"})
        elif state not in TERMINAL:
            live = any(row["state"] == "active" for row in checks)
            uncertain = any(row["state"] == "unknown" for row in checks)
            if live:
                state = "running"
                ledger_legacy_live = (ledger and ledger.get("state") == "running" and
                                      check(ledger.get("process"))["legacy_match"] is True)
                legacy_state = "running" if ledger_legacy_live else "pending"
                if state != legacy_state:
                    overrides.append({"campaign": relative, "job_id": job_id,
                                      "legacy_state": legacy_state, "state": state,
                                      "basis": "verified_live_worker"})
            elif uncertain:
                unknown_jobs.append({"campaign": relative, "job_id": job_id})
                if recorded_state == "running":
                    state = "unknown"
            elif recorded_state == "running":
                state = "pending"
        states[state] = states.get(state, 0)+1
        stage = str(job.get("stage", "unknown"))
        stages.setdefault(stage, {})[state] = stages.setdefault(stage, {}).get(state, 0)+1
    observed_checks = list(cache.values())
    if any(row["state"] == "active" for row in observed_checks):
        activity, reason = "active", "at_least_one_verified_live_process"
    elif any(row["state"] in {"held", "unknown", "missing"} for row in locks):
        activity, reason = "unknown", "lock_held_or_lock_observation_incomplete"
    elif any(row["state"] == "unknown" for row in observed_checks) or issues:
        activity, reason = "unknown", "process_or_receipt_observation_incomplete"
    else:
        activity, reason = "inactive", "recorded_processes_inactive_and_no_exclusive_lock_observed"
    return {"schema": "ks-execution-inspection/1", "output": str(root),
            "observed_utc": datetime.now(timezone.utc).isoformat(),
            "activity": activity, "reason": reason, "coordinator": coordinator_check["state"],
            "processes": processes, "locks": locks, "states": states, "stages": stages,
            "registered_jobs": sum(states.values()), "live_job_overrides": overrides,
            "unknown_jobs": unknown_jobs, "issues": issues,
            "count_scope": "main and existing deferred campaigns", "monitoring_only": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Existing immutable campaign output directory")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    result = inspect_run(args.output)
    print(json.dumps(result, indent=2 if args.pretty else None,
                     separators=None if args.pretty else (",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
