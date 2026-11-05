#!/usr/bin/env python3
"""Remove only redundant completed-fit KS staging archives; dry-run by default.

Only fits/<64-hex identity>/.fit.npz.bstesn-<host>-<pid>-<token>.npz
is considered. Canonical fits, chain/phase checkpoints, data and logs are never
deletion candidates. Outer /proc command lines identify actual workers; numeric
PID tokens from another PID namespace are not used as liveness evidence.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
import tempfile
import time
import zipfile


FIT_ID = re.compile(r"[0-9a-f]{64}\Z")
STAGING = re.compile(r"\.fit\.npz\.bstesn-([0-9a-f]{12})-([1-9][0-9]*)-([A-Za-z0-9_]+)\.npz\Z")
TERMINAL = {"complete", "inference-unqualified"}
NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="."+path.name+"-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def secure_open(directory, name, *, writable=False):
    flags = (os.O_RDWR if writable else os.O_RDONLY) | NOFOLLOW
    fd = os.open(name, flags, dir_fd=directory)
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
            or info.st_nlink != 1):
        os.close(fd)
        raise ValueError("Expected a single-link, regular, current-user file")
    return fd, info


def sha256_fd(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while block := os.read(fd, 1024*1024):
        digest.update(block)
    return digest.hexdigest()


def read_json_fd(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    with os.fdopen(os.dup(fd), "r") as stream:
        return json.load(stream)


def verify_zip_crc(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    with os.fdopen(os.dup(fd), "rb") as stream, zipfile.ZipFile(stream) as archive:
        if "__manifest__.npy" not in archive.namelist():
            raise ValueError("Canonical artifact lacks its portable-state manifest")
        if archive.testzip() is not None:
            raise ValueError("Canonical artifact failed a ZIP member CRC")
        return {"members": len(archive.infolist()), "crc_valid": True}


def stage_zip_status(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    try:
        with os.fdopen(os.dup(fd), "rb") as stream, zipfile.ZipFile(stream) as archive:
            return {"central_directory_readable": True, "members": len(archive.infolist())}
    except zipfile.BadZipFile:
        return {"central_directory_readable": False, "error": "BadZipFile"}


def actual_workers(results_root):
    """Inspect visible outer PIDs and their job records, without printing argv."""
    active, unknown, namespace_examples = {}, [], []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            argv = [os.fsdecode(x) for x in (process/"cmdline").read_bytes().split(b"\0") if x]
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            unknown.append({"outer_pid": int(process.name), "reason": "process_arguments_inaccessible"})
            continue
        is_runner = "bayes_stesn.applications.ks_runner" in argv or any(Path(x).name == "ks_runner.py" for x in argv)
        if not is_runner or "worker" not in argv:
            continue
        try:
            job_argument = Path(argv[argv.index("--job")+1])
            job_path = job_argument if job_argument.is_absolute() else (process/"cwd"/job_argument).resolve()
            # Foreign/inaccessible worker jobs are not assumed harmless.
            if job_path.resolve().parent != results_root/"jobs" or job_path.is_symlink():
                unknown.append({"outer_pid": int(process.name), "reason": "worker_job_outside_verified_run_jobs"})
                continue
            job = json.loads(job_path.read_text())
            if Path(job["output"]).resolve() != results_root or not FIT_ID.fullmatch(job["fit_id"]):
                raise ValueError("Worker job does not identify this run")
            status = (process/"status").read_text()
            nested = next((list(map(int, line.split()[1:])) for line in status.splitlines()
                           if line.startswith("NSpid:")), [int(process.name)])
            item = {"outer_pid": int(process.name), "namespace_pids": nested,
                    "job_id": job_path.stem}
            active.setdefault(job["fit_id"], []).append(item)
            if len(nested) > 1:
                namespace_examples.append(item)
        except (FileNotFoundError, ProcessLookupError):
            if process.exists():
                unknown.append({"outer_pid": int(process.name), "reason": "live_worker_job_or_status_unavailable"})
            continue
        except (PermissionError, ValueError, KeyError, IndexError, OSError):
            unknown.append({"outer_pid": int(process.name), "reason": "worker_identity_could_not_be_verified"})
    return active, unknown, namespace_examples


def clean(results_root, receipt_path, *, apply=False, minimum_age=60.):
    results_root = Path(results_root).resolve(strict=True)
    receipt_path = Path(receipt_path)
    if receipt_path.exists():
        raise FileExistsError("Choose a new cleanup-receipt filename; previous evidence is retained")
    if minimum_age < 60 or not minimum_age < float("inf"):
        raise ValueError("The minimum staging-file age must be at least 60 seconds")
    experiment_path = results_root/"experiment.json"
    experiment = json.loads(experiment_path.read_text())
    if (experiment["config"].get("schema") != "ks-experiment-1"
            or experiment["config"].get("name") != "custom_literature_grounded_KS_20261105"):
        raise ValueError("This cleanup is scoped to the verified KS application run")
    fits = results_root/"fits"
    if fits.is_symlink() or not fits.is_dir():
        raise ValueError("Expected the run's real fits directory")
    host = hashlib.sha256(platform.node().encode()).hexdigest()[:12]
    record = {"schema": "ks-redundant-staging-cleanup-1", "mode": "apply" if apply else "dry-run",
        "experiment_identity": experiment["identity"], "started_utc": utc_now(),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "minimum_age_seconds": minimum_age, "files": [],
        "protected_artifacts": ["canonical fit.npz", "chain and phase checkpoints", "data", "logs", "all nonmatching filenames"],
        "liveness_policy": "exclusive fit writer lock plus actual outer /proc worker command/job identity; never os.kill on a filename PID token",
        "cause_assessment": "Not established. This receipt records redundant staging archives and namespace evidence, without attributing an interruption mechanism."}
    atomic_json(receipt_path, record)
    for fit in sorted(fits.iterdir()):
        if fit.is_symlink() or not fit.is_dir() or not FIT_ID.fullmatch(fit.name):
            continue
        candidates = [p.name for p in fit.iterdir() if STAGING.fullmatch(p.name)]
        if not candidates:
            continue
        directory = os.open(fit, os.O_RDONLY | os.O_DIRECTORY | NOFOLLOW)
        lock_fd = receipt_fd = canonical_fd = None
        try:
            lock_fd, _ = secure_open(directory, "writer.lock", writable=True)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                for name in candidates:
                    record["files"].append({"relative_path": f"fits/{fit.name}/{name}", "action": "retained", "reason": "writer_lock_busy"})
                continue
            active, unknown, namespaces = actual_workers(results_root)
            record["pid_namespace_evidence"] = namespaces
            if unknown or fit.name in active:
                for name in candidates:
                    record["files"].append({"relative_path": f"fits/{fit.name}/{name}", "action": "retained",
                        "reason": "actual_worker_active_or_liveness_unresolved", "matching_workers": active.get(fit.name, []),
                        "unresolved_worker_count": len(unknown)})
                continue
            receipt_fd, receipt_stat = secure_open(directory, "receipt.json")
            fit_receipt = read_json_fd(receipt_fd)
            if fit_receipt.get("identity") != fit.name or fit_receipt.get("state") not in TERMINAL:
                raise ValueError("Fit receipt is not verified terminal state for this identity")
            completed_ns = int(dt.datetime.fromisoformat(fit_receipt["completed_utc"]).timestamp()*1e9)
            canonical_fd, canonical_stat = secure_open(directory, "fit.npz")
            canonical_hash = sha256_fd(canonical_fd)
            if canonical_hash != fit_receipt["fit_sha256"] or canonical_stat.st_size != fit_receipt["fit_bytes"]:
                raise ValueError("Canonical archive differs from its terminal receipt")
            crc = verify_zip_crc(canonical_fd)
            receipt_hash = sha256_fd(receipt_fd)
            for name in candidates:
                row = {"relative_path": f"fits/{fit.name}/{name}", "action": "retained"}
                stage_fd = None
                try:
                    match = STAGING.fullmatch(name)
                    if match.group(1) != host:
                        raise ValueError("Staging archive belongs to a different host token")
                    stage_fd, stage_stat = secure_open(directory, name)
                    age = time.time()-stage_stat.st_mtime
                    row.update(bytes=stage_stat.st_size, age_seconds=age,
                        staging_writer_pid_token=int(match.group(2)), staging_mtime_ns=stage_stat.st_mtime_ns)
                    if (age <= minimum_age or stage_stat.st_mtime_ns >= canonical_stat.st_mtime_ns
                            or stage_stat.st_mtime_ns >= receipt_stat.st_mtime_ns or stage_stat.st_mtime_ns >= completed_ns):
                        raise ValueError("Staging age or canonical/receipt timestamp ordering is insufficient")
                    row.update(sha256=sha256_fd(stage_fd), stage_zip=stage_zip_status(stage_fd),
                        canonical_sha256=canonical_hash, canonical_bytes=canonical_stat.st_size,
                        canonical_zip=crc, terminal_receipt_sha256=receipt_hash,
                        terminal_receipt_state=fit_receipt["state"], terminal_receipt_completed_utc=fit_receipt["completed_utc"])
                    active, unknown, namespaces = actual_workers(results_root)
                    if unknown or fit.name in active:
                        raise ValueError("Actual worker liveness changed before deletion")
                    for filename, before, fd in ((name, stage_stat, stage_fd),
                            ("fit.npz", canonical_stat, canonical_fd), ("receipt.json", receipt_stat, receipt_fd)):
                        after = os.stat(filename, dir_fd=directory, follow_symlinks=False)
                        if fingerprint(after) != fingerprint(before) or fingerprint(os.fstat(fd)) != fingerprint(before):
                            raise ValueError("Artifact identity or timestamp changed during verification")
                    row.update(actual_worker_for_fit=False, writer_lock_held=True,
                        verified_redundant=True, action="delete_pending" if apply else "would_delete")
                    record["files"].append(row)
                    atomic_json(receipt_path, record)
                    if apply:
                        os.unlink(name, dir_fd=directory)
                        os.fsync(directory)
                        row["action"] = "deleted"
                        row["deleted_utc"] = utc_now()
                except (OSError, ValueError, zipfile.BadZipFile) as error:
                    row.update(reason=str(error) if isinstance(error, ValueError) else type(error).__name__)
                    if row not in record["files"]:
                        record["files"].append(row)
                finally:
                    if stage_fd is not None:
                        os.close(stage_fd)
                atomic_json(receipt_path, record)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
            for name in candidates:
                if not any(r["relative_path"] == f"fits/{fit.name}/{name}" for r in record["files"]):
                    record["files"].append({"relative_path": f"fits/{fit.name}/{name}", "action": "retained",
                        "reason": str(error) if isinstance(error, ValueError) else type(error).__name__})
        finally:
            for fd in (canonical_fd, receipt_fd, lock_fd, directory):
                if fd is not None:
                    os.close(fd)
            atomic_json(receipt_path, record)
    record.update(completed_utc=utc_now(), considered_files=len(record["files"]),
        eligible_files=sum(r.get("verified_redundant", False) for r in record["files"]),
        deleted_files=sum(r["action"] == "deleted" for r in record["files"]),
        deleted_bytes=sum(r.get("bytes", 0) for r in record["files"] if r["action"] == "deleted"),
        eligible_bytes=sum(r.get("bytes", 0) for r in record["files"] if r.get("verified_redundant", False)))
    atomic_json(receipt_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--minimum-age-seconds", type=float, default=60.)
    args = parser.parse_args()
    result = clean(args.results, args.receipt, apply=args.apply, minimum_age=args.minimum_age_seconds)
    print(json.dumps({key: result[key] for key in ("mode", "considered_files", "eligible_files",
        "eligible_bytes", "deleted_files", "deleted_bytes")}))


if __name__ == "__main__":
    main()
