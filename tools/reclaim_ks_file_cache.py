#!/usr/bin/env python3
"""Advise eviction of task-owned file cache after KS scheduling has stopped.

Dry-run is the default. --apply requires the existing scheduler lock, every
existing fit writer lock, and no actual KS workers in the outer /proc view.
Only regular, current-user files beneath this workspace's cache/incoming/
results/deliverables roots are eligible. Eligible payloads are not read or
modified; experiment/job metadata are read. No files are deleted, and no global
kernel cache control or numerical gate changes are used. POSIX advice is a hint,
not a promise of reclaimed bytes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile


ROOT_NAMES = ("cache", "incoming", "results", "deliverables")
NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def utc():
    return datetime.now(timezone.utc).isoformat()


def memory_snapshot():
    root = Path("/sys/fs/cgroup")
    result = {}
    for name in ("memory.current", "memory.max", "memory.peak"):
        path = root/name
        if path.exists():
            value = path.read_text().strip()
            result[name] = value if value == "max" else int(value)
    for filename in ("memory.stat", "memory.events"):
        path = root/filename
        if path.exists():
            values = dict(line.split() for line in path.read_text().splitlines())
            wanted = ("anon", "file", "inactive_file", "active_file") if filename == "memory.stat" else values
            result[filename] = {name: int(values[name]) for name in wanted if name in values}
    return result


def fingerprint(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def atomic_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Choose a new receipt path; previous evidence is retained")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="."+path.name+"-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        # Link rather than replace: a simultaneous writer cannot lose evidence.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def worker_census(results):
    path = Path(__file__).with_name("clean_ks_staging.py")
    specification = importlib.util.spec_from_file_location("ks_cache_liveness", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    active, unknown, _ = module.actual_workers(results)
    return sum(len(value) for value in active.values()), len(unknown)


def inventory(workspace, minimum_bytes):
    candidates, skipped, seen = [], Counter(), set()
    for label in ROOT_NAMES:
        root = workspace/label
        if not root.exists():
            continue
        if root.is_symlink():
            raise ValueError("An eligible task root is a symlink")
        for directory, folders, names in os.walk(root, followlinks=False):
            parent = Path(directory)
            for folder in list(folders):
                if (parent/folder).is_symlink():
                    folders.remove(folder); skipped["symlink_directory"] += 1
            for name in names:
                path = parent/name
                try:
                    info = path.lstat()
                except FileNotFoundError:
                    skipped["disappeared_during_inventory"] += 1; continue
                reason = ("nonregular_or_symlink" if not stat.S_ISREG(info.st_mode)
                    else "foreign_owner" if info.st_uid != os.getuid()
                    else "below_size_threshold" if info.st_size < minimum_bytes
                    else "duplicate_hardlink" if (info.st_dev, info.st_ino) in seen else None)
                if reason:
                    skipped[reason] += 1; continue
                seen.add((info.st_dev, info.st_ino))
                candidates.append({"path": path.relative_to(workspace).as_posix(), "bytes": info.st_size,
                    "fingerprint": fingerprint(info)})
    return sorted(candidates, key=lambda row: row["path"]), dict(skipped)


def acquire_quiescence(results, locks):
    paths = [results/"scheduler.lock", *sorted((results/"fits").glob("*/writer.lock"))]
    for path in paths:
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError("Scheduler/fit writer lock must not be a symlink")
        lock = locks.enter_context(path.open("r+"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active, unresolved = worker_census(results)
    if active or unresolved:
        raise RuntimeError("KS workers are active or their liveness could not be verified")


def reclaim(workspace, results, *, apply=False, minimum_bytes=1024**2):
    workspace = Path(workspace).resolve(strict=True)
    results = Path(results).resolve(strict=True)
    if not results.is_relative_to(workspace/"results"):
        raise ValueError("Results must be inside this workspace's results root")
    if minimum_bytes < 1024**2:
        raise ValueError("Minimum eligible size is 1 MiB")
    experiment = json.loads((results/"experiment.json").read_text())
    receipt = {"schema": "ks-file-cache-advice/1", "utc": utc(),
        "experiment_identity": experiment["identity"], "apply_requested": apply,
        "operation": "POSIX_FADV_DONTNEED on existing regular task files",
        "eligible_root_labels": list(ROOT_NAMES), "minimum_file_bytes": minimum_bytes,
        "before": memory_snapshot(), "global_cache_operation": False,
        "eligible_file_payload_reads": 0, "file_content_changes": 0, "files_deleted": 0,
        "files_advised": 0, "logical_bytes_advised": 0, "errors": [],
        "limitation": "Kernel advice may retain dirty or mapped pages. Cgroup deltas include unrelated activity; no content hashes are computed because doing so would refill file cache."}
    with ExitStack() as locks:
        try:
            acquire_quiescence(results, locks)
        except (OSError, RuntimeError, ValueError) as error:
            if apply:
                raise RuntimeError("Cache advice refused: task is not verified quiescent") from error
            receipt["quiescent"] = False
            receipt["apply_blocker"] = type(error).__name__
        else:
            receipt["quiescent"] = True
        files, skipped = inventory(workspace, minimum_bytes)
        receipt.update(eligible_files=len(files), eligible_logical_bytes=sum(row["bytes"] for row in files), skipped=skipped)
        if apply:
            if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
                raise RuntimeError("POSIX_FADV_DONTNEED is unavailable; no fallback is applied")
            for row in files:
                path = workspace/row["path"]
                try:
                    descriptor = os.open(path, os.O_RDONLY | NOFOLLOW)
                    try:
                        before = os.fstat(descriptor)
                        if fingerprint(before) != row["fingerprint"] or not stat.S_ISREG(before.st_mode):
                            raise ValueError("File identity changed after inventory")
                        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
                        if fingerprint(os.fstat(descriptor)) != fingerprint(before):
                            raise ValueError("File metadata changed during advice")
                    finally:
                        os.close(descriptor)
                    receipt["files_advised"] += 1
                    receipt["logical_bytes_advised"] += row["bytes"]
                except (OSError, ValueError) as error:
                    receipt["errors"].append({"path": row["path"], "error_type": type(error).__name__})
        receipt["files"] = [{key: row[key] for key in ("path", "bytes")} for row in files]
        receipt["after"] = memory_snapshot()
        receipt["state"] = "applied" if apply and not receipt["errors"] else "applied_with_errors" if apply else "dry_run"
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--minimum-mib", default=1, type=int)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.receipt.exists():
        parser.error("Choose a new receipt path; previous evidence is retained")
    result = reclaim(args.workspace, args.results, apply=args.apply, minimum_bytes=args.minimum_mib*1024**2)
    atomic_json(args.receipt, result)
    print(json.dumps({key: result[key] for key in ("state", "quiescent", "eligible_files", "eligible_logical_bytes", "files_advised", "logical_bytes_advised")}))


if __name__ == "__main__":
    main()
