#!/usr/bin/env python3
"""Remove only byte-identical native checkpoints from a closed migration source.

Dry-run is metadata-only. Applying requires observed coordinator closure for
both roots, every existing scheduler/writer lock, and a complete SHA-256 match
of each source and replacement against the applied migration manifest. Data,
logs, receipts, sampler restart files and unidentified files are retained.
The active run is never modified. Run only after final evidence backup.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile

from run_ks_improvement_bounded import acquire_locks, closed_session_proof, safe_existing_path


NATIVE = re.compile(r"fits/[0-9a-f]{32}/fit\.npz\Z")


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stamp(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def verified_pair(source, replacement, expected):
    """Hash both complete files, reject links/mutations, and retain open handles."""
    opened, verified_stamps = [], []
    try:
        for path in (source, replacement):
            safe_existing_path(path)
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            opened.append(descriptor)
            info = os.fstat(descriptor)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                    or info.st_nlink != 1 or info.st_size != expected["bytes"]):
                raise ValueError("Expected a single-link owned file of the manifest size")
            with os.fdopen(os.dup(descriptor), "rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if (actual != expected["sha256"] or stamp(os.fstat(descriptor)) != stamp(info)
                    or stamp(path.lstat()) != stamp(info)):
                raise ValueError("Source/replacement hash or stable-file identity mismatch")
            verified_stamps.append(stamp(info))
        if (os.fstat(opened[0]).st_dev, os.fstat(opened[0]).st_ino) == (
                os.fstat(opened[1]).st_dev, os.fstat(opened[1]).st_ino):
            raise ValueError("Source and replacement must be distinct regular files")
        return opened, verified_stamps
    except BaseException:
        for descriptor in opened:
            os.close(descriptor)
        raise


def clean(source_root, active_root, closure, receipt_path, *, apply=False):
    source_root = safe_existing_path(source_root, directory=True)
    active_root = safe_existing_path(active_root, directory=True)
    if source_root == active_root or source_root in active_root.parents or active_root in source_root.parents:
        raise ValueError("Use separate, nonnested source and active roots")
    receipt_path = Path(receipt_path).absolute()
    if receipt_path.exists() or any(root == receipt_path or root in receipt_path.parents
                                    for root in (source_root, active_root)):
        raise ValueError("Choose a new receipt outside both immutable runs")
    migration_path = safe_existing_path(active_root / "migration/receipt.json")
    migration = json.loads(migration_path.read_text())
    source = json.loads((source_root / "experiment.json").read_text())
    active = json.loads((active_root / "experiment.json").read_text())
    if (migration["schema"] != "ks-improvement-budget-migration/1" or not migration["passed"]
            or not migration["apply"] or migration["scientific_recomputation"] is not False
            or source["identity"] != migration["source_experiment"]
            or active["identity"] != migration["destination_experiment"]):
        raise ValueError("Applied migration/source/destination binding mismatch")
    candidates = {name: row for name, row in migration["files"].items() if NATIVE.fullmatch(name)}
    if len(candidates) != migration["native_checkpoint_count"]:
        raise ValueError("Migration native-file census mismatch")
    record = {"schema": "ks-improvement-verified-duplicate-cleanup/1", "started_utc": utc(),
        "mode": "apply" if apply else "dry-run", "source_root": str(source_root),
        "active_root": str(active_root), "source_experiment": source["identity"],
        "active_experiment": active["identity"], "migration_sha256": digest(migration_path),
        "tool_sha256": digest(__file__), "candidate_files": len(candidates),
        "candidate_bytes": sum(row["bytes"] for row in candidates.values()),
        "disk_before": shutil.disk_usage(active_root)._asdict(), "files": [], "removed_bytes": 0,
        "protected": ["active run", "raw/canonical data", "all JSON receipts and logs",
                      "chain/phase checkpoints", "opaque or unidentified sidecars", "original study"],
        "complete": False}
    if not apply:
        record["binary_payloads_opened"] = 0
        record["files"] = [{"name": name, **row, "source_exists": (source_root/name).exists(),
                            "replacement_exists": (active_root/name).exists()}
                           for name, row in sorted(candidates.items())]
        record["complete"] = True
        atomic_json(receipt_path, record)
        return record
    record["closure"] = closed_session_proof(closure, [source_root, active_root])
    atomic_json(receipt_path, record)
    with ExitStack() as stack:
        record["locks_acquired"] = len(acquire_locks([source_root, active_root], stack))
        for name, expected in sorted(candidates.items()):
            source_path, target_path = source_root/name, active_root/name
            if not source_path.exists():
                record["files"].append({"name": name, "action": "already_absent", **expected})
                continue
            descriptors, verified_stamps = verified_pair(source_path, target_path, expected)
            try:
                for path, descriptor, verified_stamp in zip((source_path, target_path), descriptors, verified_stamps):
                    if stamp(path.lstat()) != verified_stamp or stamp(os.fstat(descriptor)) != verified_stamp:
                        raise RuntimeError("File changed immediately before removal")
                source_path.unlink()
                record["files"].append({"name": name, "action": "removed_verified_duplicate", **expected})
                record["removed_bytes"] += expected["bytes"]
                atomic_json(receipt_path, record)
            finally:
                for descriptor in descriptors:
                    os.close(descriptor)
    record.update(complete=True, finished_utc=utc(), disk_after=shutil.disk_usage(active_root)._asdict())
    atomic_json(receipt_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--active-run", type=Path, required=True)
    parser.add_argument("--closed-session-receipt", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    result = clean(args.source_run, args.active_run, args.closed_session_receipt, args.receipt, apply=args.apply)
    print(json.dumps({k: result[k] for k in ("mode", "candidate_files", "candidate_bytes", "removed_bytes", "complete")}))


if __name__ == "__main__":
    main()
