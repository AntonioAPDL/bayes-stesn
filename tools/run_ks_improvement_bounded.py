#!/usr/bin/env python3
"""Launch the unchanged KS improvement runner with bounded cache-only recovery.

Dry-run is the default. --execute enables runner launches and guarded file-cache
advice. A retry is allowed only after this wrapper's child exits nonzero with a
terminal ``RuntimeError: Memory admission gate:`` traceback. --reclaim-first
additionally requires a supplied observation of completed coordinator sessions.
Every existing scheduler and fit-writer lock is held during advice. Only
regular, current-user .npz/.h5 files under explicit run roots' fits directories
are eligible; --include-fit-receipts additionally permits closed .json receipts
there. No payload is read, changed or deleted. No /proc access, global
cache control, experiment modification or weakening of resource gates occurs.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
import uuid


MODULE = "bayes_stesn.applications.ks_improvement_runner"
MIB = 1024**2
NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text())


def write_new_json(path, value):
    """Atomic immutable receipt; concurrent callers cannot replace evidence."""
    descriptor, temporary = tempfile.mkstemp(prefix="."+path.name+"-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def safe_existing_path(path, *, directory=False):
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"Symlink paths are not eligible: {path}")
    info = path.stat()
    if directory and not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"Expected an existing directory: {path}")
    if info.st_uid != os.geteuid():
        raise ValueError(f"Task path is not owned by the current user: {path}")
    return path


def fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def memory_snapshot():
    root = Path("/sys/fs/cgroup"); out = {}
    for name in ("memory.current", "memory.max", "memory.peak"):
        value = (root/name).read_text().strip()
        out[name] = value if value == "max" else int(value)
    for name in ("memory.stat", "memory.events"):
        values = dict(line.split() for line in (root/name).read_text().splitlines())
        keys = ("anon", "file", "active_file", "inactive_file") if name == "memory.stat" else values
        out[name] = {key: int(values[key]) for key in keys if key in values}
    return out


def closed_session_proof(path, required_roots):
    if not required_roots:
        return None
    if path is None:
        raise ValueError("A closed-session receipt is required for initial/adjoined run roots")
    path = safe_existing_path(path)
    receipt = json_read(path)
    ended = {str(Path(row["root"]).absolute()) for row in receipt.get("sessions", [])
        if row.get("observed_completed") is True and isinstance(row.get("exit_code"), int)
        and not isinstance(row["exit_code"], bool)}
    missing = {str(root) for root in required_roots} - ended
    if missing:
        raise ValueError(f"No observed coordinator completion for: {sorted(missing)}")
    return {"receipt": str(path), "sha256": sha256(path),
        "required_roots": sorted(str(root) for root in required_roots),
        "scope": "Supplied observations of owned coordinator completion; additionally checked with existing locks."}


def acquire_locks(roots, stack):
    locked = []
    for root in sorted(roots):
        safe_existing_path(root, directory=True)
        fits = safe_existing_path(root/"fits", directory=True)
        paths = [root/"scheduler.lock"]
        for folder in sorted(fits.iterdir()):
            safe_existing_path(folder, directory=True)
            paths.append(folder/"writer.lock")
        for path in paths:
            safe_existing_path(path)
            descriptor = os.open(path, os.O_RDONLY | NOFOLLOW)
            stack.callback(os.close, descriptor)
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
                raise ValueError("Only existing regular current-user locks are accepted")
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if fingerprint(path.lstat()) != fingerprint(info):
                raise RuntimeError("Lock path identity changed during acquisition")
            locked.append({"root": str(root), "path": path.relative_to(root).as_posix(),
                "device": info.st_dev, "inode": info.st_ino})
    return locked


def inventory(roots, minimum_bytes, include_fit_receipts=False):
    result, seen = [], set()
    suffixes = {".npz", ".h5"} | ({".json"} if include_fit_receipts else set())
    skipped = {"non_payload": 0, "small": 0, "duplicate_inode": 0}
    for root in roots:
        for directory, folders, files in os.walk(root/"fits", followlinks=False):
            parent = Path(directory)
            for name in folders:
                safe_existing_path(parent/name, directory=True)
            for name in files:
                path = parent/name
                if path.suffix.lower() not in suffixes:
                    skipped["non_payload"] += 1; continue
                safe_existing_path(path)
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("A candidate payload is not a regular file")
                if info.st_size < minimum_bytes:
                    skipped["small"] += 1; continue
                inode = (info.st_dev, info.st_ino)
                if inode in seen:
                    skipped["duplicate_inode"] += 1; continue
                seen.add(inode)
                result.append({"root": str(root), "path": path.relative_to(root).as_posix(),
                    "bytes": info.st_size, "fingerprint": fingerprint(info)})
    return sorted(result, key=lambda row: (row["root"], row["path"])), skipped


def advise_cache(roots, *, apply, minimum_bytes=MIB, include_fit_receipts=False):
    if not isinstance(minimum_bytes, int) or isinstance(minimum_bytes, bool) or minimum_bytes < 64*1024:
        raise ValueError("Minimum eligible payload is 64 KiB")
    receipt = {"schema": "ks-improvement-guarded-cache-advice/1", "utc": utc(),
        "apply": apply, "run_roots": list(map(str, roots)), "before": memory_snapshot(),
        "minimum_eligible_bytes": minimum_bytes,
        "include_closed_fit_receipts": bool(include_fit_receipts),
        "payload_bytes_read": 0, "file_content_changes": 0, "files_deleted": 0,
        "global_kernel_cache_operation": False, "proc_access_attempted": False,
        "files_advised": 0, "logical_bytes_advised": 0, "file_fsync_calls": 0,
        "errors": [], "limitation": "POSIX advice is a hint; dirty/mapped pages may remain. Cgroup deltas include other activity and are not attributed exactly to this task."}
    with ExitStack() as locks:
        receipt["locks"] = acquire_locks(roots, locks)
        receipt["quiescent_under_owned_run_locks"] = True
        rows, skipped = inventory(roots, minimum_bytes, include_fit_receipts)
        receipt.update(files=rows, skipped=skipped, eligible_files=len(rows),
                       eligible_logical_bytes=sum(row["bytes"] for row in rows))
        if apply:
            if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
                raise RuntimeError("POSIX file-cache advice unavailable; no fallback")
            for row in rows:
                path = Path(row["root"])/row["path"]
                try:
                    safe_existing_path(path)
                    descriptor = os.open(path, os.O_RDONLY | NOFOLLOW)
                    try:
                        before = os.fstat(descriptor)
                        if (fingerprint(before) != tuple(row["fingerprint"])
                                or not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                                or fingerprint(path.lstat()) != fingerprint(before)):
                            raise RuntimeError("Payload identity changed before advice")
                        # Flush this existing file only; this is not global sync
                        # and does not change any application payload bytes.
                        os.fsync(descriptor); receipt["file_fsync_calls"] += 1
                        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
                        if (fingerprint(os.fstat(descriptor)) != fingerprint(before)
                                or fingerprint(path.lstat()) != fingerprint(before)):
                            raise RuntimeError("Payload identity changed during advice")
                    finally:
                        os.close(descriptor)
                    receipt["files_advised"] += 1
                    receipt["logical_bytes_advised"] += row["bytes"]
                except (OSError, ValueError, RuntimeError) as error:
                    receipt["errors"].append({"root": row["root"], "path": row["path"],
                        "error": type(error).__name__+": "+str(error)})
                    break
        receipt["after"] = memory_snapshot()
        receipt["observed_cgroup_file_delta_bytes"] = (
            receipt["before"]["memory.stat"]["file"]-receipt["after"]["memory.stat"]["file"])
        receipt["observed_cgroup_current_delta_bytes"] = (
            receipt["before"]["memory.current"]-receipt["after"]["memory.current"])
    receipt["state"] = "dry_run" if not apply else "applied_with_errors" if receipt["errors"] else "applied"
    return receipt


def terminal_memory_gate(log_path):
    with Path(log_path).open("rb") as stream:
        stream.seek(max(0, stream.seek(0, os.SEEK_END)-65536))
        tail = stream.read().decode("utf-8", errors="replace")
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    return bool(lines and "Traceback (most recent call last):" in tail
                and lines[-1].startswith("RuntimeError: Memory admission gate:"))


def bind_frozen(repository, config_path, output):
    config = json_read(config_path); experiment = json_read(output/"experiment.json")
    if experiment["config"] != config:
        raise ValueError("CLI configuration differs from the existing frozen experiment")
    source = repository/"src/bayes_stesn/applications"
    actual = {name: sha256(source/name) for name in experiment["source"]["application"]}
    if actual != experiment["source"]["application"]:
        raise ValueError("Application source differs from the existing frozen experiment")
    return {"config_sha256": sha256(config_path), "experiment_sha256": sha256(output/"experiment.json"),
        "experiment_identity": experiment["identity"], "source_application": actual,
        "budget": config["budget"], "experiment_created_utc": experiment["created_utc"]}


def unchanged_admission(output, budget):
    from bayes_stesn.applications.ks_runtime import admission
    return admission(output, reserve_bytes=budget["disk_reserve_bytes"],
        estimated_peak_bytes=budget["estimated_worker_peak_bytes"]*budget["workers"],
        worker_cap_bytes=budget["combined_worker_cap_bytes"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--additional-run", type=Path, action="append", default=[])
    parser.add_argument("--closed-session-receipt", type=Path)
    parser.add_argument("--logs-root", type=Path)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--stage", choices=("all", "precision", "confirm", "sensitivity"), default="all")
    parser.add_argument("--minimum-mib", type=float, default=1.)
    parser.add_argument("--include-fit-receipts", action="store_true")
    parser.add_argument("--reclaim-first", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.max_retries <= 6 or not math.isfinite(args.minimum_mib) or args.minimum_mib < .0625:
        parser.error("max-retries must be 0..6 and finite minimum-mib >=0.0625")
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository/"src"))
    output = safe_existing_path(args.output, directory=True)
    cache = safe_existing_path(args.cache, directory=True)
    config_path = safe_existing_path(args.config)
    roots = [output]+[safe_existing_path(p, directory=True) for p in args.additional_run]
    if len(set(roots)) != len(roots):
        parser.error("Explicit run roots must be unique")
    required = roots if args.reclaim_first else roots[1:]
    proof = closed_session_proof(args.closed_session_receipt, required)
    logs = (args.logs_root or output.parent/"operations"/"bounded_launches").absolute()
    if logs.resolve().is_relative_to(Path("/workspace")):
        parser.error("Launcher logs must be outside the synchronized workspace")
    logs.mkdir(parents=True, exist_ok=True); safe_existing_path(logs, directory=True)
    launch = logs/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid.uuid4().hex[:10])
    launch.mkdir()
    binding = bind_frozen(repository, config_path, output)
    command = [sys.executable, "-u", "-m", MODULE, args.stage, "--config", str(config_path),
        "--cache", str(cache), "--output", str(output)]
    invocation = {"schema": "ks-improvement-bounded-launcher/1", "utc": utc(),
        "execute": args.execute, "command": command, "binding": binding,
        "run_roots": list(map(str, roots)), "closed_session_proof": proof,
        "max_retries": args.max_retries, "reclaim_first": args.reclaim_first,
        "runner_stage": args.stage,
        "runner_modified": False, "budgets_reset": False, "gates_weakened": False,
        "launcher_sha256": sha256(Path(__file__))}
    write_new_json(launch/"invocation.json", invocation)
    print(json.dumps({"launcher": str(launch), "mode": "execute" if args.execute else "dry_run"}), flush=True)
    attempts = []; final = {"started_utc": utc(), "state": "starting"}

    def recover(label):
        if bind_frozen(repository, config_path, output) != binding:
            raise RuntimeError("Frozen bindings changed; recovery refused")
        receipt = advise_cache(roots, apply=args.execute, minimum_bytes=int(args.minimum_mib*MIB),
            include_fit_receipts=args.include_fit_receipts)
        receipt.update(closed_session_proof=proof,
            trigger=("dry_run_inventory_only" if not args.execute else
                "recorded_closed_sessions" if label == "initial" else
                "own_child_exited_with_exact_memory_admission_traceback"),
            original_campaign_budget=binding["budget"])
        if args.execute and not receipt["errors"]:
            try:
                receipt["unchanged_admission_after"] = unchanged_admission(output, binding["budget"])
                receipt["sufficient_headroom_after"] = True
            except RuntimeError as error:
                receipt["sufficient_headroom_after"] = False
                receipt["admission_error"] = str(error)
        write_new_json(launch/f"cache_{label}.json", receipt)
        if receipt["errors"] or (args.execute and not receipt.get("sufficient_headroom_after")):
            raise RuntimeError("Cache advice did not establish sufficient unchanged admission headroom; no retry")

    try:
        if not args.execute:
            recover("initial" if args.reclaim_first else "dry_run")
            final["state"] = "dry_run_complete"
            returncode = 0
        else:
            if args.reclaim_first:
                recover("initial")
            env = dict(os.environ)
            env.update({key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
            env.update(PYTHONPATH=str(repository/"src"), PYTHONHASHSEED="0", MPLBACKEND="Agg")
            for number in range(args.max_retries+1):
                if bind_frozen(repository, config_path, output) != binding:
                    raise RuntimeError("Frozen bindings changed; launch refused")
                log = launch/f"attempt_{number:02d}.log"
                descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o600)
                begin = time.monotonic(); started = utc()
                with os.fdopen(descriptor, "ab", buffering=0) as stream:
                    child = subprocess.Popen(command, cwd=repository, env=env, stdout=stream, stderr=subprocess.STDOUT)
                    returncode = child.wait()
                memory_failure = returncode != 0 and terminal_memory_gate(log)
                row = {"attempt": number, "started_utc": started, "finished_utc": utc(),
                    "pid": child.pid, "returncode": returncode, "elapsed_seconds": time.monotonic()-begin,
                    "terminal_memory_admission_failure": memory_failure,
                    "log": log.name, "log_sha256": sha256(log), "memory": memory_snapshot()}
                write_new_json(launch/f"attempt_{number:02d}.json", row); attempts.append(row)
                print(json.dumps(row), flush=True)
                if returncode == 0 or not memory_failure or number == args.max_retries:
                    final["state"] = "runner_exited_zero" if returncode == 0 else "retry_limit" if memory_failure else "non_memory_failure"
                    break
                recover(f"retry_{number+1:02d}")
    except Exception as error:
        final.update(state="stopped", error=type(error).__name__+": "+str(error))
        returncode = 1
    finally:
        final.update(finished_utc=utc(), attempts=attempts,
            limitation="A zero runner exit is not a claim that all scientific stages finished; inspect its status/terminal receipts. Cgroup changes are not exact task-attributed reclamation.")
        write_new_json(launch/"summary.json", final)
        print(json.dumps({"state": final["state"], "receipt": str(launch/"summary.json")}), flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
