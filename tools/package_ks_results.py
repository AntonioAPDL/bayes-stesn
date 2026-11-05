#!/usr/bin/env python3
"""Stream a quiescent, completed KS results root into verified <=32 MiB parts.

Uses only Python's standard library. The same script supplies read-only verify
and fresh-destination restore commands. It never edits archived job paths,
merges checkpoints into an existing run, or packages the data cache/source/env.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import datetime as dt
import fcntl
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time


SCHEMA = "ks-results-multipart-1"
CHUNK = 1024 * 1024
MAX_PART = 32 * CHUNK
SHA = re.compile(r"[0-9a-f]{64}\Z")
PART = re.compile(r"ks-results\.tar\.gz\.part([0-9]{6})\Z")
DIRECTORIES = {"fits", "forecasts", "ledger", "jobs", "attempts", "selection",
    "precision_cases", "predictions", "reports", "figures", "plots", "animations",
    "tables", "logs", "diagnostics", "evidence", "audits", "cleanup", "artifacts"}
REQUIRED = {"experiment.json", "freeze.json", "test_access.json", "summary.json"}
NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="."+path.name+"-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def digest_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def safe_name(name):
    if not isinstance(name, str) or not name or "\\" in name or "\0" in name:
        raise ValueError("Unsafe archive member name")
    p = PurePosixPath(name)
    if p.is_absolute() or name != p.as_posix() or any(x in ("", ".", "..") or x.startswith(".") for x in p.parts):
        raise ValueError("Unsafe or hidden archive member path")
    if ":" in p.parts[0]:
        raise ValueError("Drive-qualified archive paths are forbidden")
    return p


def excluded_name(name):
    return name.startswith(".") or name.endswith((".lock", ".tmp", "~")) or ".bstesn-" in name


def fingerprint(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def free_bytes(path):
    values = os.statvfs(path)
    return values.f_bavail * values.f_frsize


def inventory(root):
    members, excluded = [], []
    for directory, folders, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in list(folders):
            path = parent/name; rel = path.relative_to(root).as_posix()
            reason = ("symlink" if path.is_symlink() else "lock_staging_or_temporary" if excluded_name(name)
                      else "outside_results_directory_allowlist" if parent == root and name not in DIRECTORIES else None)
            if reason:
                folders.remove(name); excluded.append({"path": rel, "reason": reason})
            else:
                safe_name(rel)
        for name in files:
            path = parent/name; rel = path.relative_to(root).as_posix()
            info = path.lstat()
            reason = ("symlink" if stat.S_ISLNK(info.st_mode) else "nonregular_file" if not stat.S_ISREG(info.st_mode)
                      else "lock_staging_or_temporary" if excluded_name(name)
                      else "source_or_environment_file" if path.suffix in {".py", ".pyc", ".so", ".whl"} else None)
            if reason:
                excluded.append({"path": rel, "reason": reason}); continue
            safe_name(rel)
            members.append({"name": rel, "bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
                            "source_mode": stat.S_IMODE(info.st_mode), "fingerprint": fingerprint(info)})
    members.sort(key=lambda row: row["name"])
    if not REQUIRED.issubset({row["name"] for row in members}):
        raise ValueError("Completed experiment/freeze/test-access/summary files are required")
    return members, excluded


def active_workers(root):
    found = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = [os.fsdecode(x) for x in (process/"cmdline").read_bytes().split(b"\0") if x]
            if "bayes_stesn.applications.ks_runner" not in args or "worker" not in args:
                continue
            candidate = Path(args[args.index("--job")+1])
            if not candidate.is_absolute():
                candidate = (process/"cwd"/candidate).resolve()
            job = json.loads(candidate.read_text())
            if Path(job["output"]).resolve() == root:
                found.append(int(process.name))
        except (FileNotFoundError, ProcessLookupError):
            if process.exists():
                raise RuntimeError("A live worker's job identity could not be verified")
        except (PermissionError, KeyError, IndexError, ValueError) as error:
            raise RuntimeError("Worker liveness could not be verified") from error
    return found


def provenance(root):
    experiment = json.loads((root/"experiment.json").read_text())
    frozen = json.loads((root/"freeze.json").read_text())
    gate = json.loads((root/"test_access.json").read_text())
    summary = json.loads((root/"summary.json").read_text())
    frozen_values = dict(frozen); frozen_values.pop("identity")
    if json_hash({"config": experiment["config"], "source": experiment["source"]}) != experiment["identity"]:
        raise ValueError("Experiment identity failed reconstruction")
    if json_hash(frozen_values) != frozen["identity"]:
        raise ValueError("Freeze identity failed reconstruction")
    if (frozen["experiment_identity"] != experiment["identity"] or gate["freeze_identity"] != frozen["identity"]
            or summary["freeze_identity"] != frozen["identity"] or not gate.get("both_track_selection_complete")):
        raise ValueError("Experiment, freeze, test gate and summary are incompatible")
    rows = summary.get("rows", [])
    if not {"A", "B"}.issubset({r["track"] for r in rows}):
        raise ValueError("Both main-track summaries are required")
    if any(r.get("pending_origins", 0) or r.get("pending_fits", 0) for r in rows):
        raise ValueError("The final summary still has pending work")
    ledger = Counter()
    for path in (root/"ledger").glob("*.json"):
        ledger[json.loads(path.read_text())["state"]] += 1
    if ledger.get("running") or ledger.get("pending"):
        raise ValueError("The ledger still contains running or pending jobs")
    return {"experiment_identity": experiment["identity"], "freeze_identity": frozen["identity"],
        "base_revision": experiment["base_revision"], "numerical_source_and_runtime": experiment["source"],
        "ledger_states": dict(ledger), "configuration_hash": json_hash(experiment["config"]),
        "required_disk_reserve_bytes": experiment["config"]["budget"]["disk_reserve_bytes"]}


class PartWriter:
    """Writes compressed bytes directly to bounded on-disk parts."""
    def __init__(self, destination, limit, reserve):
        self.destination, self.limit, self.reserve = destination, limit, reserve
        self.parts, self.stream, self.current = [], None, None
        self.total, self.current_bytes = 0, 0
        self.digest, self.part_digest = hashlib.sha256(), None

    def _open(self):
        name = f"ks-results.tar.gz.part{len(self.parts)+1:06d}"
        self.current = self.destination/("."+name+".incomplete")
        self.stream = self.current.open("xb")
        self.part_digest, self.current_bytes = hashlib.sha256(), 0

    def _finish(self):
        if self.stream is None:
            return
        self.stream.flush(); os.fsync(self.stream.fileno()); self.stream.close()
        name = f"ks-results.tar.gz.part{len(self.parts)+1:06d}"
        final = self.destination/name
        if final.exists():
            raise FileExistsError("A package part already exists")
        os.rename(self.current, final)
        self.parts.append({"name": name, "bytes": self.current_bytes, "sha256": self.part_digest.hexdigest()})
        self.stream = None

    def write(self, data):
        view = memoryview(data); written = 0
        while written < len(view):
            if self.stream is None:
                self._open()
            count = min(len(view)-written, self.limit-self.current_bytes, CHUNK)
            if free_bytes(self.destination)-count < self.reserve:
                raise RuntimeError("Disk reserve reached; complete parts and the incomplete tail are retained")
            block = view[written:written+count]
            self.stream.write(block); self.part_digest.update(block); self.digest.update(block)
            self.current_bytes += count; self.total += count; written += count
            if self.current_bytes == self.limit:
                self._finish()
        return written

    def flush(self):
        if self.stream is not None:
            self.stream.flush()

    def finish(self):
        self._finish()

    def abort(self):
        if self.stream is not None:
            self.stream.flush(); os.fsync(self.stream.fileno()); self.stream.close()
            self.stream = None


class HashingReader:
    def __init__(self, stream):
        self.stream, self.digest, self.count = stream, hashlib.sha256(), 0

    def read(self, amount=-1):
        if amount < 0:
            raise ValueError("Unbounded source reads are forbidden")
        block = self.stream.read(amount)
        self.digest.update(block); self.count += len(block)
        return block


def create(root, output, *, part_mib=32, gzip_level=1, reserve_bytes=None, code_revision=None):
    started = time.perf_counter(); root = Path(root).resolve(strict=True)
    output = Path(output).resolve()
    if output.is_relative_to(root) or output.exists():
        raise ValueError("Use a new package directory outside the results root")
    if not 1 <= part_mib <= 32 or not 0 <= gzip_level <= 9:
        raise ValueError("Use 1-32 MiB parts and a gzip level from 0 through 9")
    if (root/"scheduler.lock").is_symlink() or (root/"fits").is_symlink():
        raise ValueError("Scheduler and fit paths must not be symbolic links")
    with ExitStack() as locks:
        scheduler = locks.enter_context((root/"scheduler.lock").open("r+"))
        fcntl.flock(scheduler, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if active_workers(root):
            raise RuntimeError("A worker is still active; stop scheduling and wait before packaging")
        # The scheduler lock prevents new normal scheduling. Existing fit locks
        # also catch a worker that has not appeared in a ledger yet.
        for path in sorted((root/"fits").glob("*/writer.lock")):
            if path.is_symlink() or path.parent.is_symlink():
                raise ValueError("A fit writer lock is a symlink")
            lock = locks.enter_context(path.open("r+"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identities = provenance(root)
        members, excluded = inventory(root)
        source_bytes = sum(row["bytes"] for row in members)
        reserve = max(int(identities["required_disk_reserve_bytes"]), int(reserve_bytes or 0), 2*1024**3)
        ancestor = output.parent
        while not ancestor.exists():
            ancestor = ancestor.parent
        tar_bound = source_bytes+8192*len(members)+20480
        compressed_bound = tar_bound+16*math.ceil(tar_bound/16384)+CHUNK
        if free_bytes(ancestor) < reserve+compressed_bound:
            raise RuntimeError("Insufficient quota-aware free space for the archive bound plus disk reserve")
        expected_fits = {}
        for path in (root/"fits").glob("*/receipt.json"):
            receipt = json.loads(path.read_text())
            if receipt.get("identity") != path.parent.name:
                raise ValueError("Fit receipt identity differs from its directory")
            if receipt["state"] in {"complete", "inference-unqualified"}:
                name = (path.parent/"fit.npz").relative_to(root).as_posix()
                expected_fits[name] = receipt["fit_sha256"]
        if not set(expected_fits).issubset({r["name"] for r in members}):
            raise ValueError("A completed canonical fit is missing")
        if code_revision is None:
            result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                                    capture_output=True, text=True, check=False)
            code_revision = result.stdout.strip() if result.returncode == 0 else None
        if not code_revision or not re.fullmatch(r"[0-9a-f]{40,64}", code_revision):
            raise ValueError("Provide --code-revision with the exact code commit")
        status = subprocess.run(["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parent,
                                capture_output=True, text=True, check=False)
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
        state = {"schema": SCHEMA, "state": "streaming", "started_utc": now(),
                 "experiment_identity": identities["experiment_identity"], "source_bytes": source_bytes,
                 "disk_reserve_bytes": reserve}
        atomic_json(output/"package_state.json", state)
        writer = PartWriter(output, part_mib*CHUNK, reserve)
        try:
            with gzip.GzipFile(filename="", mode="wb", compresslevel=gzip_level, fileobj=writer, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                    for row in members:
                        path = root/row["name"]
                        fd = os.open(path, os.O_RDONLY | NOFOLLOW)
                        with os.fdopen(fd, "rb") as stream:
                            if fingerprint(os.fstat(stream.fileno())) != row["fingerprint"]:
                                raise ValueError("Source artifact changed before packaging")
                            reader = HashingReader(stream)
                            entry = tarfile.TarInfo(row["name"]); entry.size = row["bytes"]
                            entry.mode, entry.uid, entry.gid = 0o600, 0, 0
                            entry.uname = entry.gname = ""; entry.mtime = row["mtime_ns"]//10**9
                            archive.addfile(entry, reader)
                            if reader.count != row["bytes"] or fingerprint(os.fstat(stream.fileno())) != row["fingerprint"]:
                                raise ValueError("Source artifact changed while packaging")
                            row["sha256"] = reader.digest.hexdigest()
                            if row["name"] in expected_fits and row["sha256"] != expected_fits[row["name"]]:
                                raise ValueError("Canonical fit failed its original receipt checksum")
            writer.finish()
            final_inventory, _ = inventory(root)
            if [(r["name"], r["fingerprint"]) for r in final_inventory] != [(r["name"], r["fingerprint"]) for r in members]:
                raise ValueError("The results root changed while packaging")
            helper = output/"restore_ks_results.py"
            helper.write_bytes(Path(__file__).read_bytes()); os.chmod(helper, 0o600)
            manifest = {"schema": SCHEMA, "state": "complete", "created_utc": now(),
                "archive_format": "single gzip-compressed POSIX PAX tar stream, split as binary parts",
                "part_limit_bytes": part_mib*CHUNK, "parts": writer.parts,
                "full_stream_bytes": writer.total, "full_stream_sha256": writer.digest.hexdigest(),
                "members": [{k:v for k,v in row.items() if k != "fingerprint"} for row in members],
                "member_count": len(members), "source_bytes": source_bytes, "excluded": excluded,
                "identities": {**identities, "code_revision": code_revision,
                    "code_worktree_dirty_at_packaging": dirty,
                    "source_authority": "Recorded numerical source hashes and runtime; a Git commit alone does not cover uncommitted work."},
                "tool_sha256": digest_file(__file__),
                "helper": {"name": helper.name, "bytes": helper.stat().st_size, "sha256": digest_file(helper)},
                "canonical_fit_receipts_checked": len(expected_fits), "disk_reserve_bytes": reserve,
                "compressed_size_preflight_bound_bytes": compressed_bound,
                "packaging_seconds": time.perf_counter()-started,
                "relocation": "Bytes are restored unchanged. Archived jobs and precision descriptors contain original absolute paths; do not execute archived worker jobs directly. Matching source/runtime/config/cache identities remain required for resume."}
            atomic_json(output/"manifest.json", manifest)
            (output/"manifest.sha256").write_text(digest_file(output/"manifest.json")+"  manifest.json\n")
            checked = verify(output)
            atomic_json(output/"verification.json", checked)
            atomic_json(output/"package_state.json", {**state, "state": "complete_verified", "completed_utc": now(),
                "parts": len(writer.parts), "full_stream_bytes": writer.total, "manifest_sha256": digest_file(output/"manifest.json")})
            return {"state": "complete_verified", "parts": len(writer.parts), "members": len(members),
                    "full_stream_bytes": writer.total, "full_stream_sha256": writer.digest.hexdigest()}
        except BaseException as error:
            writer.abort()
            atomic_json(output/"package_state.json", {**state, "state": "incomplete", "error_type": type(error).__name__,
                "completed_parts": writer.parts, "observed_stream_bytes": writer.total, "stopped_utc": now()})
            raise


def read_manifest(package):
    package = Path(package).resolve(strict=True)
    manifest_path, checksum = package/"manifest.json", package/"manifest.sha256"
    if manifest_path.is_symlink() or checksum.is_symlink():
        raise ValueError("Manifest files must be regular local files")
    tokens = checksum.read_text().split()
    if len(tokens) != 2 or tokens[1] != "manifest.json" or not SHA.fullmatch(tokens[0]) or digest_file(manifest_path) != tokens[0]:
        raise ValueError("Manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("state") != "complete":
        raise ValueError("A complete compatible package manifest is required")
    limit = manifest["part_limit_bytes"]
    if not isinstance(limit, int) or not 1 <= limit <= MAX_PART:
        raise ValueError("Invalid package part limit")
    parts = manifest["parts"]
    if not parts or sum(p["bytes"] for p in parts) != manifest["full_stream_bytes"]:
        raise ValueError("Package part byte counts do not match the full stream")
    for index, part in enumerate(parts, 1):
        match = PART.fullmatch(part["name"])
        path = package/part["name"]
        if not match or int(match.group(1)) != index or not SHA.fullmatch(part["sha256"]):
            raise ValueError("Parts must have contiguous safe numbered names and SHA256 hashes")
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or not 0 < part["bytes"] <= limit or info.st_size != part["bytes"]:
            raise ValueError("Part type/size mismatch")
        if index < len(parts) and part["bytes"] != limit:
            raise ValueError("Only the final part may be shorter than the part limit")
    names = []
    for row in manifest["members"]:
        safe_name(row["name"])
        if not isinstance(row["bytes"], int) or row["bytes"] < 0 or not SHA.fullmatch(row["sha256"]):
            raise ValueError("Invalid member size or checksum")
        names.append(row["name"])
    if names != sorted(set(names)) or len(names) != manifest["member_count"] or not REQUIRED.issubset(names):
        raise ValueError("Archive members are duplicated, unordered or incomplete")
    if sum(row["bytes"] for row in manifest["members"]) != manifest["source_bytes"]:
        raise ValueError("Archive member byte count mismatch")
    if not SHA.fullmatch(manifest["full_stream_sha256"]):
        raise ValueError("Invalid full-stream checksum")
    helper = manifest["helper"]; helper_path = package/helper["name"]
    if (helper["name"] != "restore_ks_results.py" or helper_path.is_symlink()
            or helper_path.stat().st_size != helper["bytes"] or digest_file(helper_path) != helper["sha256"]):
        raise ValueError("Standalone helper is missing or has changed")
    return package, manifest


class PartReader(io.RawIOBase):
    """Checks every part while supplying a concatenated stream without copying it."""
    def __init__(self, package, manifest):
        self.package, self.manifest = package, manifest
        self.index, self.stream, self.part_bytes, self.total = 0, None, 0, 0
        self.digest, self.part_digest = hashlib.sha256(), None

    def readable(self):
        return True

    def readinto(self, target):
        if len(target) == 0:
            return 0
        if self.index == len(self.manifest["parts"]):
            return 0
        if self.stream is None:
            part = self.manifest["parts"][self.index]
            fd = os.open(self.package/part["name"], os.O_RDONLY | NOFOLLOW)
            self.stream = os.fdopen(fd, "rb")
            self.part_digest, self.part_bytes = hashlib.sha256(), 0
        block = self.stream.read(len(target))
        if not block:
            part = self.manifest["parts"][self.index]
            self.stream.close(); self.stream = None
            if self.part_bytes != part["bytes"] or self.part_digest.hexdigest() != part["sha256"]:
                raise ValueError("Binary part checksum mismatch")
            self.index += 1
            return self.readinto(target)
        target[:len(block)] = block
        self.part_digest.update(block); self.digest.update(block)
        self.part_bytes += len(block); self.total += len(block)
        return len(block)

    def close(self):
        if self.stream is not None:
            self.stream.close(); self.stream = None
        super().close()


def scan_archive(package, manifest, destination=None):
    count = total = 0
    retained = {}
    reader = PartReader(package, manifest)
    with reader, io.BufferedReader(reader, buffer_size=CHUNK) as raw:
        with gzip.GzipFile(fileobj=raw, mode="rb") as compressed:
            with tarfile.open(fileobj=compressed, mode="r|", ignore_zeros=True) as archive:
                for entry in archive:
                    safe_name(entry.name)
                    if not entry.isfile() or entry.sparse is not None or count >= len(manifest["members"]):
                        raise ValueError("Unexpected, linked, sparse or non-file archive member")
                    expected = manifest["members"][count]
                    if entry.name != expected["name"] or entry.size != expected["bytes"]:
                        raise ValueError("Archive member name/size differs from manifest")
                    digest, bytes_read = hashlib.sha256(), 0
                    output = None; capture = bytearray() if entry.name in REQUIRED else None
                    if destination is not None:
                        target = destination.joinpath(*PurePosixPath(entry.name).parts)
                        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                        for parent in target.parents:
                            if parent == destination.parent:
                                break
                            if parent.is_symlink():
                                raise ValueError("Restore parent path became a symlink")
                        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW, 0o600)
                        output = os.fdopen(fd, "wb")
                    try:
                        with archive.extractfile(entry) as source:
                            while block := source.read(CHUNK):
                                digest.update(block); bytes_read += len(block)
                                if capture is not None:
                                    if len(capture)+len(block) > 64*CHUNK:
                                        raise ValueError("Required identity JSON exceeds its bounded size")
                                    capture.extend(block)
                                if output is not None:
                                    if free_bytes(destination)-len(block) < manifest["disk_reserve_bytes"]:
                                        raise RuntimeError("Restore disk reserve reached; incomplete destination retained")
                                    output.write(block)
                        if digest.hexdigest() != expected["sha256"] or bytes_read != entry.size:
                            raise ValueError("Archive member checksum mismatch")
                        if capture is not None:
                            retained[entry.name] = json.loads(capture)
                        if output is not None:
                            output.flush(); os.fsync(output.fileno())
                    finally:
                        if output is not None:
                            output.close()
                    if destination is not None:
                        os.utime(target, ns=(expected["mtime_ns"], expected["mtime_ns"]), follow_symlinks=False)
                    count += 1; total += bytes_read
            # Reading through EOF validates the gzip trailer and any concatenated
            # stream; tar ignore_zeros also detects members after padding blocks.
            if compressed.read(CHUNK):
                raise ValueError("Unexpected decompressed trailing data")
        if raw.read(1):
            raise ValueError("Unexpected compressed trailing data")
        if (reader.index != len(manifest["parts"]) or reader.total != manifest["full_stream_bytes"]
                or reader.digest.hexdigest() != manifest["full_stream_sha256"]):
            raise ValueError("Full compressed-stream checksum/count mismatch")
    if count != manifest["member_count"] or total != manifest["source_bytes"]:
        raise ValueError("Archive did not contain every declared member")
    experiment, frozen = retained["experiment.json"], retained["freeze.json"]
    check = dict(frozen); check.pop("identity")
    if (json_hash({"config": experiment["config"], "source": experiment["source"]}) != experiment["identity"]
            or json_hash(check) != frozen["identity"]
            or frozen["experiment_identity"] != experiment["identity"]
            or experiment["identity"] != manifest["identities"]["experiment_identity"]
            or experiment["source"] != manifest["identities"]["numerical_source_and_runtime"]
            or frozen["identity"] != manifest["identities"]["freeze_identity"]
            or retained["test_access.json"]["freeze_identity"] != frozen["identity"]
            or retained["summary.json"]["freeze_identity"] != frozen["identity"]):
        raise ValueError("Restored identities are incompatible")
    return count, total


def verify(package):
    started = time.perf_counter(); package, manifest = read_manifest(package)
    count, total = scan_archive(package, manifest)
    return {"schema": "ks-results-package-verification-1", "state": "verified", "verified_utc": now(),
        "manifest_sha256": digest_file(package/"manifest.json"), "parts": len(manifest["parts"]),
        "members": count, "uncompressed_member_bytes": total, "full_stream_bytes": manifest["full_stream_bytes"],
        "full_stream_sha256": manifest["full_stream_sha256"], "elapsed_seconds": time.perf_counter()-started,
        "verification": "manifest, helper, each part, full compressed stream, gzip trailer, each regular member and study identities",
        "writes": 0}


def restore(package, destination):
    checked = verify(package)  # Full read-only validation precedes any extraction.
    package, manifest = read_manifest(package)
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Restore requires a new destination; there is no overwrite or merge mode")
    parent = destination.parent.resolve(strict=True)
    destination = parent/destination.name
    if free_bytes(parent) < manifest["source_bytes"]+manifest["disk_reserve_bytes"]+CHUNK:
        raise RuntimeError("Insufficient restore space above the preserved disk reserve")
    destination.mkdir(mode=0o700, exist_ok=False)
    marker = destination/".ks-restore-state.json"
    atomic_json(marker, {"state": "incomplete", "started_utc": now(), "manifest_sha256": checked["manifest_sha256"]})
    count, total = scan_archive(package, manifest, destination)
    receipt = {"state": "complete_verified", "completed_utc": now(), "members": count,
        "bytes": total, "manifest_sha256": checked["manifest_sha256"],
        "experiment_identity": manifest["identities"]["experiment_identity"],
        "freeze_identity": manifest["identities"]["freeze_identity"],
        "job_paths_rewritten": False, "checkpoint_merge_performed": False}
    atomic_json(marker, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("create")
    make.add_argument("--results", required=True, type=Path)
    make.add_argument("--output", required=True, type=Path)
    make.add_argument("--part-mib", type=int, default=32)
    make.add_argument("--gzip-level", type=int, default=1)
    make.add_argument("--reserve-bytes", type=int)
    make.add_argument("--code-revision")
    check = commands.add_parser("verify"); check.add_argument("--package", required=True, type=Path)
    unpack = commands.add_parser("restore")
    unpack.add_argument("--package", required=True, type=Path)
    unpack.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "create":
        result = create(args.results, args.output, part_mib=args.part_mib, gzip_level=args.gzip_level,
                        reserve_bytes=args.reserve_bytes, code_revision=args.code_revision)
    elif args.command == "verify":
        result = verify(args.package)
    else:
        result = restore(args.package, args.destination)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
