#!/usr/bin/env python3
"""One guarded pass over seven recorded KS archive staging files.

Dry-run is the default. This task-specific tool never scans arbitrary names for
deletion. An apply pass requires exclusive scheduler/fit locks, no actual KS
workers, no unresolved process liveness, and no visible writable descriptors on
the candidates or their canonical replacements. Transfers outside the visible
process namespace must already have finished. A persistent apply marker forbids
automatic retries, including deletion of a subsequently recreated staging file.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import time
import zipfile


def load_helper(name):
    path = Path(__file__).with_name(name + ".py")
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


stage_helper = load_helper("clean_ks_staging")
cache_helper = load_helper("reclaim_ks_file_cache")
NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
BLOCK = 1024**2
MARKER = "archive_staging_apply_once.json"
KS_SHA = "898f18deccc1fe6b3fb4d6eb0aa847244dc115efc56a260aceaae07bc22c227c"
GROUPS = (
    {"directory": "incoming", "canonical": "ks_prepared_v1.tar.gz", "bytes": 182334299,
     "sha256": KS_SHA, "kind": "binary", "staging": (
         ".ks_prepared_v1.tar.gz.TEMAos", ".ks_prepared_v1.tar.gz.TEMAos.bolKbc",
         ".ks_prepared_v1.tar.gz.TEMAos.bolKbc.1UaF5e")},
    {"directory": "cache/data/ks/exports", "canonical": "ks_prepared_v1.tar.gz", "bytes": 182334299,
     "sha256": KS_SHA, "kind": "binary", "staging": (
         ".ks_prepared_v1.tar.gz.w7SgB5", ".ks_prepared_v1.tar.gz.w7SgB5.SB6WtX")},
    {"directory": "deliverables", "canonical": "ks_validation_checkpoint_01.zip", "bytes": 347338663,
     "sha256": "ae78c23139e19c9ccb66eb06276e289aaf880d8b4ec5180833989fd7bed5ae67",
     "kind": "zip", "receipt": "ks_validation_checkpoint_01.receipt.json", "staging": (
         ".ks_validation_checkpoint_01.zip.7ZgKug", ".ks_validation_checkpoint_01.zip.tmp.BzkMur")},
)


def regular_owned(info, *, single_link):
    return (stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid()
            and (not single_link or info.st_nlink == 1))


def checked_directory(workspace, relative):
    """Open through directory descriptors so symlink ancestors are rejected."""
    descriptor = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | NOFOLLOW)
    try:
        for part in Path(relative).parts:
            if part in ("", ".", ".."):
                raise ValueError("unsafe_directory_component")
            next_descriptor = os.open(part, os.O_RDONLY | os.O_DIRECTORY | NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor); descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def open_regular(directory, name, *, single_link):
    descriptor = os.open(name, os.O_RDONLY | NOFOLLOW, dir_fd=directory)
    info = os.fstat(descriptor)
    if not regular_owned(info, single_link=single_link):
        os.close(descriptor)
        raise ValueError("not_regular_owned_or_required_single_link")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, info


def unchanged(directory, name, descriptor, before):
    expected = stage_helper.fingerprint(before)
    return (stage_helper.fingerprint(os.fstat(descriptor)) == expected
            and stage_helper.fingerprint(os.stat(name, dir_fd=directory, follow_symlinks=False)) == expected)


def writable_descriptors(targets, *, processes=None):
    """Check actual visible /proc entries; do not trust filename PID tokens."""
    writers, unresolved = [], 0
    for process in Path("/proc").iterdir() if processes is None else processes:
        if not process.name.isdigit():
            continue
        try:
            descriptors = list((process/"fd").iterdir())
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            unresolved += 1; continue
        for descriptor in descriptors:
            try:
                info = descriptor.stat()
                identity = (info.st_dev, info.st_ino)
                if identity not in targets:
                    continue
                content = (process/"fdinfo"/descriptor.name).read_text()
                fields = dict(line.split(":", 1) for line in content.splitlines() if ":" in line)
                flags = int(fields["flags"].strip(), 8)
                if flags & os.O_ACCMODE in (os.O_WRONLY, os.O_RDWR):
                    writers.append({"relative_path": targets[identity], "outer_pid": int(process.name),
                                    "descriptor": int(descriptor.name)})
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (PermissionError, OSError, ValueError, KeyError):
                unresolved += 1
    return writers, unresolved


def compare_prefix(stage_fd, canonical_fd, stage_bytes, canonical_bytes):
    if stage_bytes <= 0 or stage_bytes > canonical_bytes:
        raise ValueError("empty_or_larger_than_canonical")
    os.lseek(stage_fd, 0, os.SEEK_SET); os.lseek(canonical_fd, 0, os.SEEK_SET)
    remaining = stage_bytes; stage_hash = hashlib.sha256(); canonical_hash = hashlib.sha256(); equal = True
    while remaining:
        amount = min(BLOCK, remaining)
        left = os.read(stage_fd, amount); right = os.read(canonical_fd, amount)
        if len(left) != amount or len(right) != amount:
            raise ValueError("unexpected_end_during_prefix_check")
        stage_hash.update(left); canonical_hash.update(right)
        equal = equal and left == right
        remaining -= amount
    return {"sha256": stage_hash.hexdigest(), "canonical_prefix_sha256": canonical_hash.hexdigest(),
            "exact_prefix_match": equal,
            "redundancy": "full_copy" if stage_bytes == canonical_bytes else "shorter_exact_prefix"}


def canonical_verification(directory, group, locks):
    descriptor, info = open_regular(directory, group["canonical"], single_link=False)
    locks.callback(os.close, descriptor)
    if info.st_size != group["bytes"]:
        raise ValueError("canonical_size_mismatch")
    digest = stage_helper.sha256_fd(descriptor)
    if digest != group["sha256"]:
        raise ValueError("canonical_sha256_mismatch")
    result = {"relative_path": group["directory"]+"/"+group["canonical"], "bytes": info.st_size,
              "sha256": digest, "zip_crc_valid": None}
    if group["kind"] == "zip":
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb") as stream, zipfile.ZipFile(stream) as archive:
            if archive.testzip() is not None:
                raise ValueError("canonical_zip_crc_failure")
            result.update(zip_crc_valid=True, zip_members=len(archive.infolist()))
    dependencies = [(group["canonical"], descriptor, info)]
    if group.get("receipt"):
        receipt_fd, receipt_info = open_regular(directory, group["receipt"], single_link=False)
        locks.callback(os.close, receipt_fd)
        receipt = stage_helper.read_json_fd(receipt_fd)
        if receipt.get("bytes") != group["bytes"] or receipt.get("sha256") != group["sha256"]:
            raise ValueError("canonical_receipt_identity_mismatch")
        result["receipt_sha256"] = stage_helper.sha256_fd(receipt_fd)
        dependencies.append((group["receipt"], receipt_fd, receipt_info))
    return descriptor, info, dependencies, result


def clean(workspace, results, receipt_path, *, apply=False, minimum_age=60.):
    workspace = Path(workspace).resolve(strict=True); results = Path(results).resolve(strict=True)
    receipt_path = Path(receipt_path)
    if receipt_path.exists():
        raise FileExistsError("Choose a new receipt; prior evidence is retained")
    if not results.is_relative_to(workspace/"results") or minimum_age < 60 or not minimum_age < float("inf"):
        raise ValueError("Invalid results root or minimum staging age")
    experiment = json.loads((results/"experiment.json").read_text())
    if (experiment["config"].get("schema") != "ks-experiment-1"
            or experiment["config"].get("name") != "custom_literature_grounded_KS_20261105"):
        raise ValueError("This tool is restricted to the recorded KS application")
    record = {"schema": "ks-archive-staging-cleanup/1", "mode": "apply" if apply else "dry-run",
        "experiment_identity": experiment["identity"], "started_utc": stage_helper.utc_now(),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "policy_sha256": hashlib.sha256(json.dumps(GROUPS, sort_keys=True).encode()).hexdigest(),
        "minimum_age_seconds": minimum_age, "recorded_file_count": sum(len(group["staging"]) for group in GROUPS),
        "files": [], "canonical_verification": [],
        "deleted_files": 0, "deleted_logical_bytes": 0, "deleted_allocated_bytes": 0,
        "blockers": [], "unknown_staging_files_retained": [], "state": "started",
        "cause_assessment": "Not established; no mechanism is inferred from staging filenames or timestamps.",
        "scope": "Seven exact recorded filenames only; canonical archives/checkpoints, raw data, fit/chain/phase states and all other files are protected.",
        "liveness_limitation": "Actual visible outer processes and descriptors are inspected. Relevant transfers or synchronization writers outside this view must already have finished."}
    cache_helper.atomic_json(receipt_path, record)
    # Persist the same receipt atomically on every consequential state transition.
    def save():
        stage_helper.atomic_json(receipt_path, record)
    marker = results/"cleanup"/MARKER
    candidates, verified_canonicals = [], []
    apply_marker_created = False
    with ExitStack() as locks:
        try:
            cache_helper.acquire_quiescence(results, locks)
        except (OSError, ValueError, RuntimeError) as error:
            record["blockers"].append({"reason": "scheduler_or_workers_not_verified_quiescent", "error_type": type(error).__name__})
        if apply and marker.exists():
            record["blockers"].append({"reason": "previous_apply_pass_exists_no_automatic_retry"})
        targets = {}; groups = []
        for group in GROUPS:
            try:
                directory = checked_directory(workspace, group["directory"]); locks.callback(os.close, directory)
                names = set(os.listdir(directory))
                known = set(group["staging"])
                prefix = "."+group["canonical"]+"."
                record["unknown_staging_files_retained"].extend(group["directory"]+"/"+name
                    for name in sorted(names-known) if name.startswith(prefix))
                rows = []
                for name in group["staging"]:
                    relative = group["directory"]+"/"+name
                    row = {"relative_path": relative, "action": "retained"}; record["files"].append(row)
                    try:
                        info = os.stat(name, dir_fd=directory, follow_symlinks=False)
                        row.update(bytes=info.st_size, allocated_bytes=info.st_blocks*512,
                            age_seconds=time.time()-info.st_mtime, fingerprint=list(stage_helper.fingerprint(info)))
                        if not regular_owned(info, single_link=True):
                            row["reason"] = "not_regular_owned_single_link"; continue
                        targets[(info.st_dev, info.st_ino)] = relative
                        rows.append((name, info, row))
                    except FileNotFoundError:
                        row.update(action="absent", reason="recorded_staging_file_absent")
                for name in (group["canonical"], group.get("receipt")):
                    if name:
                        info = os.stat(name, dir_fd=directory, follow_symlinks=False)
                        if stat.S_ISREG(info.st_mode):
                            targets[(info.st_dev, info.st_ino)] = group["directory"]+"/"+name
                groups.append((group, directory, rows))
            except (OSError, ValueError) as error:
                record["blockers"].append({"reason": "group_path_or_canonical_unavailable", "directory": group["directory"], "error_type": type(error).__name__})
        writers, unresolved = writable_descriptors(targets)
        record.update(writable_descriptors=writers, unresolved_descriptor_checks=unresolved)
        if writers or unresolved:
            record["blockers"].append({"reason": "writable_descriptor_or_unresolved_descriptor_check"})
        if not record["blockers"]:
            for group, directory, rows in groups:
                if not rows:
                    continue
                try:
                    canonical_fd, canonical_info, dependencies, checked = canonical_verification(directory, group, locks)
                    record["canonical_verification"].append(checked)
                    verified_canonicals.append((group, directory, canonical_fd, canonical_info, checked))
                except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
                    for _, _, row in rows:
                        row["reason"] = str(error) if isinstance(error, ValueError) else type(error).__name__
                    continue
                for name, inventory_info, row in rows:
                    try:
                        if row["age_seconds"] <= minimum_age:
                            raise ValueError("staging_file_too_recent")
                        stage_fd, stage_info = open_regular(directory, name, single_link=True)
                        locks.callback(os.close, stage_fd)
                        if stage_helper.fingerprint(stage_info) != stage_helper.fingerprint(inventory_info):
                            raise ValueError("staging_identity_changed_since_inventory")
                        row.update(compare_prefix(stage_fd, canonical_fd, stage_info.st_size, canonical_info.st_size))
                        if not row["exact_prefix_match"]:
                            raise ValueError("staging_bytes_do_not_match_canonical_prefix")
                        checks = [(name, stage_fd, stage_info), *dependencies]
                        if not all(unchanged(directory, key, fd, info) for key, fd, info in checks):
                            raise ValueError("identity_changed_during_verification")
                        row.update(verified_redundant=True, action="would_delete", canonical_sha256=checked["sha256"])
                        candidates.append((directory, name, checks, row))
                    except (OSError, ValueError) as error:
                        row["reason"] = str(error) if isinstance(error, ValueError) else type(error).__name__
        if apply and candidates and not record["blockers"]:
            marker.parent.mkdir(parents=True, exist_ok=True)
            if marker.parent.is_symlink():
                raise ValueError("Cleanup marker parent must not be a symlink")
            cache_helper.atomic_json(marker, {"schema": "ks-archive-staging-apply-once/1",
                "experiment_identity": experiment["identity"], "started_utc": stage_helper.utc_now(),
                "policy_sha256": record["policy_sha256"], "state": "apply_pass_started_no_automatic_retry"})
            apply_marker_created = True
            for directory, name, checks, row in candidates:
                active, unknown = cache_helper.worker_census(results)
                writers, unresolved = writable_descriptors(targets)
                if active or unknown or writers or unresolved:
                    record["blockers"].append({"reason": "liveness_changed_before_deletion"}); break
                if not all(unchanged(directory, key, fd, info) for key, fd, info in checks):
                    row.update(action="retained", reason="identity_changed_before_deletion"); continue
                row["action"] = "delete_pending"; save()
                try:
                    os.unlink(name, dir_fd=directory)
                except OSError as error:
                    row.update(action="retained", reason="deletion_error_"+type(error).__name__); save(); continue
                row.update(action="deleted", deleted_utc=stage_helper.utc_now())
                record["deleted_files"] += 1; record["deleted_logical_bytes"] += row["bytes"]
                record["deleted_allocated_bytes"] += row["allocated_bytes"]
                try:
                    os.fsync(directory)
                except OSError as error:
                    row["directory_sync_error"] = type(error).__name__
                    record["blockers"].append({"reason": "directory_sync_failed_after_deletion"})
                    save(); break
                save()
            # Reappearance is reported after the single pass, never deleted again.
            record["reappeared_after_pass"] = [row["relative_path"] for _, name, _, row in candidates
                if row["action"] == "deleted" and (workspace/row["relative_path"]).exists()]
            for group, directory, fd, info, checked in verified_canonicals:
                checked["post_pass_sha256"] = stage_helper.sha256_fd(fd)
                checked["post_pass_unchanged"] = (unchanged(directory, group["canonical"], fd, info)
                    and checked["post_pass_sha256"] == group["sha256"])
                if not checked["post_pass_unchanged"]:
                    record["blockers"].append({"reason": "canonical_post_pass_verification_failed",
                                               "relative_path": checked["relative_path"]})
        if record["blockers"]:
            for row in record["files"]:
                if row["action"] not in {"deleted", "absent"}:
                    row.update(action="retained", reason=row.get("reason", "global_liveness_or_scope_blocker"))
        record.update(completed_utc=stage_helper.utc_now(),
            considered_existing_files=sum(row["action"] != "absent" for row in record["files"]),
            retained_files=sum(row["action"] == "retained" for row in record["files"]),
            absent_files=sum(row["action"] == "absent" for row in record["files"]),
            eligible_files=sum(row.get("verified_redundant", False) for row in record["files"]),
            eligible_logical_bytes=sum(row.get("bytes", 0) for row in record["files"] if row.get("verified_redundant", False)),
            state="blocked" if record["blockers"] else "apply_pass_complete" if apply else "dry_run_complete")
        save()
        if apply_marker_created:
            stage_helper.atomic_json(marker, {"schema": "ks-archive-staging-apply-once/1",
                "experiment_identity": experiment["identity"], "completed_utc": record["completed_utc"],
                "policy_sha256": record["policy_sha256"], "state": "apply_pass_finished_no_automatic_retry",
                "deleted_files": record["deleted_files"], "retained_files": record["retained_files"],
                "blockers": len(record["blockers"]), "reappeared_after_pass": record.get("reappeared_after_pass", [])})
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--minimum-age-seconds", type=float, default=60.)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    result = clean(args.workspace, args.results, args.receipt, apply=args.apply, minimum_age=args.minimum_age_seconds)
    print(json.dumps({key: result[key] for key in ("state", "mode", "considered_existing_files", "eligible_files", "eligible_logical_bytes",
                                                    "deleted_files", "deleted_logical_bytes", "deleted_allocated_bytes")}))
    if args.apply and result["state"] == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
