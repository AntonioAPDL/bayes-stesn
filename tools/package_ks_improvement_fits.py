#!/usr/bin/env python3
"""Plan, create, verify and restore private KS native-checkpoint backups.

``plan`` reads small job/freeze metadata and stats binary files; it never opens
checkpoint payloads and can run during computation. ``create --apply`` requires
the existing scheduler and every fit writer lock. It packages either completed
native fits for frozen main/anchor/prior roles (all available trajectory/seed
fits), or all canonical native/chain/phase checkpoints. A current verified
compact-evidence manifest supplies source/configuration/selection/score records.

Independent ZIP segments are capped at 512 MiB. Already-compressed native NPZs
are stored unchanged; there is no giant intermediate archive. Full hashing,
NPZ CRC and internal state-checksum validation happen only during create.
Verification/restoration use the standard library; creating a backup also uses
NumPy only to validate array dtypes and bounded Fortran-order array blocks.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import ExitStack
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import tempfile
import uuid
import zipfile


SCHEMA = "ks-improvement-native-backup/1"
MIB = 1024**2
MAX_PART = 512*MIB
NATIVE = "fit.npz"
CHAIN = "fit.npz.chains.npz"
PHASE = "fit.npz.chains.npz.ks-fit-phase.npz"
CANONICAL = (NATIVE, CHAIN, PHASE)
SHA = re.compile(r"[0-9a-f]{64}\Z")
FIT = re.compile(r"[0-9a-f]{32}\Z")


def utc():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def seed_for(seed, *keys):
    raw = json.dumps([int(seed), *keys], sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def safe_path(path, directory=False):
    path = Path(os.path.abspath(path))
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Symlink paths are forbidden")
    info = path.stat()
    if (info.st_uid != os.geteuid() or (directory and not stat.S_ISDIR(info.st_mode))
            or (not directory and not stat.S_ISREG(info.st_mode))):
        raise ValueError("Expected a current-user regular file/directory")
    return path


def safe_member(name):
    p = PurePosixPath(name)
    if (not name or p.is_absolute() or name != p.as_posix() or "\\" in name or ":" in name
            or "\0" in name or any(x in ("", ".", "..") or x.startswith(".") for x in p.parts)):
        raise ValueError("Unsafe archive member name")
    return name


def reject_private(value):
    forbidden = {"password", "access_token", "refresh_token", "client_secret", "authorization",
                 "cookie", "private_key", "drive_file_id", "drive_folder_id", "owner_email"}
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in forbidden:
                raise ValueError("Credential/account/retrieval field is not eligible")
            reject_private(item)
    elif isinstance(value, list):
        for item in value:
            reject_private(item)
    elif isinstance(value, str) and re.search(r"drive\.google\.com|-----BEGIN [A-Z ]*PRIVATE KEY|\b(?:ghp_|github_pat_|ya29\.)[A-Za-z0-9_.-]+|[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", value, re.I):
        raise ValueError("Credential/account/retrieval text is not eligible")


def read_json(path, maximum=32*MIB):
    path = safe_path(path); before = path.stat()
    if before.st_size > maximum:
        raise ValueError("Metadata exceeds its declared size gate")
    payload = path.read_bytes()
    if fingerprint(path.stat()) != fingerprint(before):
        raise RuntimeError("Metadata changed during read")
    value = json.loads(payload); reject_private(value)
    return value


def plain(value):
    if isinstance(value, dict):
        if set(value) == {"__tuple__"}:
            return [plain(x) for x in value["__tuple__"]]
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plain(x) for x in value]
    return value


def current_records(root):
    exp = read_json(root/"experiment.json")
    if exp["identity"] != identity({"config": exp["config"], "source": exp["source"], "cache": exp["cache_identity"]}):
        raise ValueError("Experiment identity mismatch")
    registry = {(t, c["id"]): c for t, r in exp["config"]["registry"]["tracks"].items() for c in r["candidates"]}
    jobs = {}
    for path in sorted((root/"jobs").glob("*.json")):
        j = read_json(path)
        expected = identity({k: v for k, v in j.items() if k != "job_id"})
        key = identity({"track": j["track"], "effective_config": j["candidate"]["config"],
                        "trajectory": int(j["trajectory"]), "seed": int(j["seed"])})[:32]
        if (j["job_id"] != expected[:24] or path.stem != j["job_id"] or j["fit_key"] != key
                or j["experiment_identity"] != exp["identity"] or j["source"] != exp["source"]
                or j["candidate"] != registry[(j["track"], j["candidate"]["id"])]) :
            raise ValueError("Job/source/candidate/fit binding mismatch")
        jobs.setdefault(key, []).append(j)
    frozen = None
    if (root/"freeze.json").exists():
        frozen = read_json(root/"freeze.json")
        if (frozen["experiment_identity"] != exp["identity"]
                or frozen["identity"] != identity({k: v for k, v in frozen.items() if k != "identity"})):
            raise ValueError("Freeze identity mismatch")
    return exp, jobs, frozen


def role_map(frozen):
    roles = {}
    if frozen is None:
        return roles
    for track, row in frozen["selected"].items():
        for label, cid in (("main", row["main"]["id"]), ("anchor", row["anchor_id"]),
                           ("prior_sensitivity", (row.get("sensitivity") or {}).get("id"))):
            if cid:
                roles.setdefault((track, cid), []).append(label)
    return roles


def selected_coverage(exp, frozen, rows):
    if frozen is None:
        return {"known": False, "reason": "Selection is not yet frozen"}
    registry = {(t, c["id"]): c for t, r in exp["config"]["registry"]["tracks"].items() for c in r["candidates"]}
    roles = role_map(frozen); wanted = {}
    trajectories = sorted(set(exp["config"]["promote"]["trajectories"]) | set(frozen["confirmation_trajectories"]))
    for (track, cid), labels in roles.items():
        cases = {(i, exp["config"]["model_seed"]) for i in trajectories}
        if set(labels) & {"main", "anchor"}:
            cases |= {(i, seed) for i in exp["config"]["sensitivity"]["trajectories"]
                      for seed in exp["config"]["sensitivity"]["seeds"]}
        for trajectory, seed in cases:
            key = identity({"track": track, "effective_config": registry[(track, cid)]["config"],
                            "trajectory": trajectory, "seed": seed})[:32]
            wanted[key] = {"fit_key": key, "track": track, "candidate_id": cid,
                           "trajectory": trajectory, "seed": seed, "roles": labels}
    available = {r["fit_key"] for r in rows if r["native_present"] and r["receipt_present"]}
    return {"known": True, "expected_unique_selected_fits": len(wanted),
        "present_receipted_native_fits": len(set(wanted) & available),
        "missing_or_unreceipted": [wanted[k] for k in sorted(set(wanted)-available)],
        "scope": "Frozen selected/anchor/prior configurations across declared fit trajectories; additional declared seeds for main/anchor roles. Availability is not inference qualification."}


def inventory(root, mode):
    root = safe_path(root, directory=True)
    exp, jobs, frozen = current_records(root); roles = role_map(frozen)
    rows, skipped = [], Counter()
    for folder in sorted((root/"fits").iterdir()):
        safe_path(folder, directory=True)
        if not FIT.fullmatch(folder.name):
            raise ValueError("Unexpected fit-directory name")
        if folder.name not in jobs:
            if any((folder/name).exists() for name in CANONICAL):
                raise ValueError("Checkpoint has no bound job provenance")
            continue
        j = jobs[folder.name][0]; role = roles.get((j["track"], j["candidate"]["id"]), [])
        if mode == "selected-only" and not role:
            continue
        names = [NATIVE] if mode == "selected-only" else list(CANONICAL)
        files = []
        for name in names:
            path = folder/name
            if path.exists():
                safe_path(path)
                files.append({"name": f"fits/{folder.name}/{name}", "bytes": path.stat().st_size,
                              "fingerprint": fingerprint(path.stat())})
        if not files:
            continue
        has_receipt = (folder/"receipt.json").exists()
        if mode == "selected-only" and not has_receipt:
            skipped["selected_native_without_complete_receipt"] += 1; continue
        rows.append({"fit_key": folder.name, "track": j["track"], "candidate_id": j["candidate"]["id"],
            "trajectory": j["trajectory"], "seed": j["seed"], "roles": role,
            "native_present": (folder/NATIVE).exists(), "receipt_present": has_receipt,
            "files": files, "job_ids": [j["job_id"] for j in jobs[folder.name]]})
    counts = Counter(Path(f["name"]).name for row in rows for f in row["files"])
    total = sum(f["bytes"] for row in rows for f in row["files"])
    largest = max((f["bytes"] for row in rows for f in row["files"]), default=0)
    return {"schema": SCHEMA, "mode": mode, "experiment_identity": exp["identity"],
        "freeze_identity": None if frozen is None else frozen["identity"], "selection_frozen": frozen is not None,
        "fit_count": len(rows), "canonical_file_counts": dict(counts), "canonical_binary_bytes": total,
        "largest_binary_file_bytes": largest, "minimum_single_member_part_mib": math.ceil((largest+1024)/MIB),
        "ideal_512mib_payload_parts_lower_bound": math.ceil(total/MAX_PART), "skipped": dict(skipped),
        "selected_fit_coverage": selected_coverage(exp, frozen, rows),
        "binary_payloads_opened": 0, "quiescence_verified": False,
        "scope": "Stat-only planning estimate. Active writers may change these sizes; metadata and scores add storage.",
        "fits": rows}, exp, jobs, frozen


def acquire_locks(root, stack):
    paths = [root/"scheduler.lock"]
    paths.extend(folder/"writer.lock" for folder in sorted((root/"fits").iterdir()))
    result = []
    for path in paths:
        safe_path(path)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)); stack.callback(os.close, descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = os.fstat(descriptor)
        if fingerprint(path.stat()) != fingerprint(info):
            raise RuntimeError("Lock path changed while acquiring quiescence")
        result.append({"path": path.relative_to(root).as_posix(), "device": info.st_dev, "inode": info.st_ino})
    return result


def npy_header(stream):
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError("Invalid numeric member header")
    version = stream.read(2)
    if version == b"\x01\x00":
        width, fmt, encoding = 2, "<H", "latin1"
    elif version in (b"\x02\x00", b"\x03\x00"):
        width, fmt, encoding = 4, "<I", "utf-8" if version[0] == 3 else "latin1"
    else:
        raise ValueError("Unsupported numeric member version")
    length = struct.unpack(fmt, stream.read(width))[0]
    if length > 65536:
        raise ValueError("Oversized numeric header")
    value = ast.literal_eval(stream.read(length).decode(encoding))
    if set(value) != {"descr", "fortran_order", "shape"} or not isinstance(value["shape"], tuple):
        raise ValueError("Unexpected numeric header")
    return value


def array_keys(value):
    if isinstance(value, dict):
        if set(value) == {"__array__"}:
            yield value["__array__"]
        else:
            for key in sorted(value):
                yield from array_keys(value[key])
    elif isinstance(value, list):
        for item in value:
            yield from array_keys(item)


def validate_npz(path, kind, max_fortran_bytes=128*MIB):
    """Stream the repository's exact packed-state checksum, without fitting."""
    import numpy as np
    path = safe_path(path); stamp = fingerprint(path.stat()); byte_sha = digest(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names) or "__manifest__.npy" not in names:
            raise ValueError("Malformed native numeric archive")
        if archive.getinfo("__manifest__.npy").file_size > 128*MIB:
            raise ValueError("Native metadata exceeds the validation memory gate")
        if sum(info.file_size for info in archive.infolist()) > 2*1024**3:
            raise ValueError("Decoded native archive exceeds the declared 2 GiB gate")
        with archive.open("__manifest__.npy") as stream:
            header = npy_header(stream); dtype = np.dtype(header["descr"])
            if header["shape"] != () or dtype.kind != "U":
                raise ValueError("Expected a scalar Unicode native manifest")
            if dtype.itemsize > 128*MIB:
                raise ValueError("Oversized native manifest dtype")
            raw = stream.read(dtype.itemsize+1)
            if len(raw) != dtype.itemsize:
                raise ValueError("Native manifest byte count mismatch")
            text = raw.decode("utf-32-be" if dtype.str.startswith(">") else "utf-32-le").rstrip("\0")
        manifest = json.loads(text)
        if manifest.get("schema_version") != "1.0" or manifest.get("kind") != kind:
            raise ValueError("Incompatible native artifact schema/kind")
        state = manifest["state"]; reject_private(state)
        keys = list(array_keys(state))
        if keys != [f"a{i:06d}" for i in range(len(keys))] or set(names) != {"__manifest__.npy", *(key+".npy" for key in keys)}:
            raise ValueError("Array references differ from the native traversal")
        h = hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        decoded = 0
        for key in keys:
            with archive.open(key+".npy") as stream:
                header = npy_header(stream); dtype = np.dtype(header["descr"]); shape = header["shape"]
                if dtype.hasobject or dtype.fields is not None or any(not isinstance(x, int) or x < 0 for x in shape):
                    raise ValueError("Executable/structured/invalid array type forbidden")
                size = math.prod(shape)*dtype.itemsize
                # io.content_hash applies ascontiguousarray, which promotes a
                # scalar array to shape (1,) while preserving its one value.
                hash_shape = shape if shape else (1,)
                h.update(key.encode()); h.update(dtype.str.encode()); h.update(str(hash_shape).encode()); decoded += size
                if header["fortran_order"] and len(shape) > 1 and size:
                    if size > max_fortran_bytes:
                        raise MemoryError("Fortran array exceeds bounded validation workspace")
                    raw = stream.read(size+1)
                    if len(raw) != size:
                        raise ValueError("Numeric member byte count mismatch")
                    values = np.frombuffer(raw, dtype=dtype).reshape(shape, order="F")
                    stride = max(1, MIB//max(1, math.prod(shape[1:])*dtype.itemsize))
                    for first in range(0, shape[0], stride):
                        h.update(np.ascontiguousarray(values[first:first+stride]).tobytes())
                    del values, raw
                else:
                    count = 0
                    while block := stream.read(MIB):
                        count += len(block); h.update(block)
                    if count != size:
                        raise ValueError("Numeric member byte count mismatch")
        if h.hexdigest() != manifest["sha256"]:
            raise ValueError("Native internal state checksum mismatch")
    if fingerprint(path.stat()) != stamp:
        raise RuntimeError("Native checkpoint changed during validation")
    return {"state": state, "sha256": byte_sha, "bytes": path.stat().st_size,
            "internal_state_sha256": manifest["sha256"], "decoded_array_bytes": decoded,
            "fingerprint": stamp}


def expected_config(job):
    cfg = copy.deepcopy(job["candidate"]["config"])
    cfg["features"]["seed"] = seed_for(job["seed"], "reservoir")
    cfg["sampler"]["seed"] = seed_for(job["seed"], "mcmc")
    return cfg


def validate_fit(root, row, job, exp, mode):
    folder = root/"fits"/row["fit_key"]; checked = {}; cfg = expected_config(job)
    summary = {k: row[k] for k in ("fit_key", "track", "candidate_id", "trajectory", "seed", "roles")}
    native = None; prepared = None
    if row["native_present"]:
        checked[NATIVE] = validate_npz(folder/NATIVE, "ks_native_fit")
        native = checked[NATIVE]["state"]
        expected = identity({key: native[key] for key in ("schema", "model_id", "input_identity", "config", "seed", "computation")})
        if (native["schema"] != "ks-native-fit-1" or native["model_id"] != "bayes_stesn"
                or native["identity"] != expected or native["seed"] != job["seed"]
                or plain(native["config"]) != cfg or plain(native["computation"]) != exp["source"]["native"]):
            raise ValueError("Completed native fit/source/config identity mismatch")
        prepared = native["core"]["prepared"]["identity"]
        summary.update(native_fit_identity=native["identity"], prepared_identity=prepared,
            qualified=bool(native["diagnostics"]["qualified"]), forecast_state_available=True)
        if row["receipt_present"]:
            receipt = read_json(folder/"receipt.json")
            if (receipt["fit_key"] != row["fit_key"] or receipt["native_fit_identity"] != native["identity"]
                    or receipt["fit_sha256"] != checked[NATIVE]["sha256"] or receipt["fit_bytes"] != checked[NATIVE]["bytes"]
                    or receipt["config"] != cfg or receipt["seed"] != job["seed"]
                    or receipt["track"] != job["track"] or receipt["trajectory"] != job["trajectory"]
                    or receipt["candidate_id"] != job["candidate"]["id"]
                    or bool(receipt["diagnostics"]["qualified"]) != summary["qualified"]
                    or receipt["diagnostics"]["input_identity"] != native["input_identity"]):
                raise ValueError("Canonical native checkpoint failed its current fit receipt")
            summary["native_receipt_verified"] = True
        else:
            summary["native_receipt_verified"] = False
    else:
        summary.update(forecast_state_available=False, qualified=False, native_receipt_verified=False)
    if mode == "all-fit-checkpoints":
        chain = phase = None
        if (folder/CHAIN).exists():
            checked[CHAIN] = validate_npz(folder/CHAIN, "mcmc_checkpoint"); chain = checked[CHAIN]["state"]
            if prepared is not None and chain["identity"] != prepared:
                raise ValueError("Chain checkpoint belongs to a different prepared model")
            prepared = chain["identity"]
        if (folder/PHASE).exists():
            checked[PHASE] = validate_npz(folder/PHASE, "ks_bayes_fit_phase"); phase = plain(checked[PHASE]["state"])
        if chain is not None and phase is None:
            raise ValueError("A chain checkpoint without its required phase file cannot be resumed safely")
        if phase is not None:
            initial = cfg["sampler"]["draws"]; maximum = initial+cfg["continuation_draws"]
            if (phase["initial_draws"] != initial or phase["maximum_draws"] != maximum
                    or phase["target_draws"] not in {initial, maximum}
                    or phase["phase"] not in {"initial_sampling", "initial_diagnostics", "continuation_sampling", "continuation_diagnostics", "complete"}):
                raise ValueError("Phase file changed the frozen continuation allowance")
            if prepared is not None:
                transition = copy.deepcopy(cfg["sampler"]); transition.pop("checkpoint_every", None)
                expected = identity({"schema": "ks-bayes-fit-phase-1", "prepared": prepared,
                    "initial_sampler": transition, "continuation_draws": cfg["continuation_draws"],
                    "diagnostics": cfg["diagnostics"], "diagnostic_seed": seed_for(job["seed"], "diagnostics"),
                    "computation": exp["source"]["native"]})
                if phase["identity"] != expected:
                    raise ValueError("Phase/source/diagnostic identity mismatch")
            elif phase["phase"] != "initial_sampling" or phase.get("checkpoint") is not None:
                raise ValueError("Phase requires its missing chain checkpoint")
            summary.update(inference_phase=phase["phase"], initial_draws=initial, maximum_draws=maximum,
                phase_prepared_identity_verified=prepared is not None)
            if chain is not None:
                old, intended = plain(chain["config"]), copy.deepcopy(cfg["sampler"])
                draws = old.pop("draws"); intended.pop("draws")
                old.pop("checkpoint_every", None); intended.pop("checkpoint_every", None)
                if old != intended or draws not in {initial, maximum} or draws > phase["target_draws"]:
                    raise ValueError("Chain sampler transition/target mismatch")
                progress = {}
                for key, saved in chain["chains"].items():
                    required = {"schema_version", "signature", "chain_index", "schedule", "iteration",
                        "current_log_target", "vector", "nu", "prior_state", "C", "sigma2",
                        "rng_state", "proposal", "initialization", "initial_vector", "retained_states",
                        "retained_log_density", "accepted_total", "accepted_sampling", "proposals_sampling"}
                    if not required <= set(saved):
                        raise ValueError("Chain lacks exact restart state")
                    if (not key.isdigit() or not 0 <= int(key) < cfg["sampler"]["chains"]
                            or saved["schema_version"] != 1 or saved["chain_index"] != int(key)
                            or not SHA.fullmatch(saved["signature"])
                            or saved["schedule"] not in {"collapsed", "conditional"}
                            or not 0 <= saved["iteration"] <= cfg["sampler"]["warmup"]+draws*cfg["sampler"]["thin"]
                            or len(saved["retained_states"]) > draws
                            or len(saved["retained_states"]) != len(saved["retained_log_density"])):
                        raise ValueError("Chain restart schema, progress or identity is inconsistent")
                    progress[key] = {"iteration": saved["iteration"], "retained_draws": len(saved["retained_states"])}
                complete = len(progress) == cfg["sampler"]["chains"] and all(
                    v["iteration"] == cfg["sampler"]["warmup"]+draws*cfg["sampler"]["thin"]
                    and v["retained_draws"] == draws for v in progress.values())
                if phase["phase"] == "complete" and (not complete or phase["checkpoint"]["sha256"] != checked[CHAIN]["sha256"]):
                    raise ValueError("Completed phase does not match its exact chain checkpoint")
                if phase["phase"] == "complete" and (not 1 <= len(phase["gates"]) <= 2
                        or (native is not None and bool(phase["gates"][-1]["passed"]) != summary["qualified"])):
                    raise ValueError("Completed phase diagnostics disagree with native qualification")
                summary.update(chain_progress=progress, chain_complete_at_recorded_target=complete,
                    inference_restart_state_available=True,
                    chain_transition_signature_validation="Retained unchanged; original sampler rechecks it against prepared arrays on identity-guarded resume")
        summary.setdefault("inference_restart_state_available", False)
    else:
        summary["inference_restart_state_available"] = False
        summary["inference_sidecars_intentionally_excluded"] = True
    summary["continuation_limitation"] = "Phase-complete unqualified inference remains unqualified. This backup grants no extra draws or changed model/prior allowance."
    records = {name: {k: v for k, v in value.items() if k != "state"} for name, value in checked.items()}
    return summary, records


def compact_binding(manifest_path, root, exp, frozen):
    path = safe_path(manifest_path); manifest = read_json(path)
    if manifest["schema"] != "ks-improvement-compact-evidence/1" or manifest["experiment_identity"] != exp["identity"]:
        raise ValueError("Compact evidence belongs to another experiment")
    archive = safe_path(path.parent/safe_member(manifest["archive"]["name"]))
    if archive.stat().st_size != manifest["archive"]["bytes"] or digest(archive) != manifest["archive"]["sha256"]:
        raise ValueError("Compact evidence archive checksum mismatch")
    members = {row["name"]: row for row in manifest["members"]}
    if len(members) != len(manifest["members"]):
        raise ValueError("Duplicate compact-evidence member")
    with zipfile.ZipFile(archive) as zipped:
        if set(zipped.namelist()) != set(members) or len(zipped.namelist()) != len(members):
            raise ValueError("Compact archive/member manifest mismatch")
        for entry in zipped.infolist():
            safe_member(entry.filename); declared = members[entry.filename]
            if entry.is_dir() or entry.file_size != declared["bytes"]:
                raise ValueError("Compact member size mismatch")
            h = hashlib.sha256()
            with zipped.open(entry) as stream:
                while block := stream.read(MIB):
                    h.update(block)
            if h.hexdigest() != declared["sha256"]:
                raise ValueError("Compact member SHA-256/CRC mismatch")
    for name in ("experiment.json", "history_policy.json", *(('freeze.json',) if frozen is not None else ())):
        if members.get("run/"+name, {}).get("sha256") != digest(root/name):
            raise ValueError("Compact evidence is stale for the current experiment/history/freeze")
    # A matching terminal-name set does not establish a current accounting
    # snapshot: another interrupted attempt may consume work without finishing
    # a new job. Bind both frontiers and their exact small-file contents while
    # the caller holds the scheduler and fit-writer locks.
    for label in ("terminal", "attempts"):
        current = {}
        for p in (root/label).glob("*.json"):
            record = read_json(p)
            eligible = (record.get("state") in {"complete", "failed", "inference-unqualified"}
                if label == "terminal" else
                record.get("state") in {"complete", "interrupted"}
                and "finished_utc" in record and isinstance(record.get("returncode"), int))
            if eligible:
                current[f"run/{label}/"+p.name] = p
        archived = {name for name in members if name.startswith(f"run/{label}/")}
        description = "terminal receipts" if label == "terminal" else "closed attempts"
        if set(current) != archived:
            raise ValueError(f"Create a fresh compact snapshot after the coordinator stops: {description} frontier changed")
        for name, path_now in current.items():
            if (path_now.stat().st_size != members[name]["bytes"]
                    or digest(path_now) != members[name]["sha256"]):
                raise ValueError(f"Create a fresh compact snapshot after the coordinator stops: stale {description} bytes")
    return path, archive


def source_row(path, name, checked=None):
    safe_path(path); info = path.stat()
    return {"name": safe_member(name), "source": str(path), "bytes": info.st_size,
        "sha256": checked["sha256"] if checked else digest(path), "fingerprint": fingerprint(info)}


def store_parts(staging, members, part_bytes, reserve):
    parts, part_members = [], []; archive = None; planned = 22
    def finish():
        nonlocal archive
        if archive is None:
            return
        name = f"native_fits.part{len(parts)+1:04d}.zip"; archive.close(); archive = None
        path = staging/name
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        if path.stat().st_size > part_bytes:
            raise RuntimeError("ZIP part exceeded its hard upload cap")
        parts.append({"name": name, "bytes": path.stat().st_size, "sha256": digest(path),
                      "members": list(part_members)})
        part_members.clear()
    try:
        for row in members:
            bound = row["bytes"]+256+2*len(row["name"].encode())
            if bound+22 > part_bytes:
                raise ValueError("A single native file exceeds this part cap; choose a larger cap up to512 MiB")
            if archive is not None and planned+bound > part_bytes:
                finish(); planned = 22
            if archive is None:
                archive = zipfile.ZipFile(staging/f"native_fits.part{len(parts)+1:04d}.zip", "x", compression=zipfile.ZIP_STORED)
            path = Path(row["source"]); safe_path(path)
            if fingerprint(path.stat()) != row["fingerprint"]:
                raise RuntimeError("Verified source changed before archive copy")
            h = hashlib.sha256(); count = 0
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "rb") as source, archive.open(row["name"], "w") as target:
                while block := source.read(MIB):
                    if shutil.disk_usage(staging).free-len(block) < reserve:
                        raise RuntimeError("Original disk reserve reached; archive remains unpublished")
                    target.write(block); h.update(block); count += len(block)
            if count != row["bytes"] or h.hexdigest() != row["sha256"] or fingerprint(path.stat()) != row["fingerprint"]:
                raise ValueError("Source checksum/identity changed during archive copy")
            row["part"] = f"native_fits.part{len(parts)+1:04d}.zip"
            part_members.append(row["name"]); planned += bound
        finish()
    finally:
        if archive is not None:
            archive.close()
    return parts


RESTORE_TEXT = """# Restoring KS native fit checkpoints

Run `python restore_ks_improvement_fits.py verify --package PACKAGE` first.
Then use `restore --package PACKAGE --destination NEW_DIRECTORY`. The helper
refuses existing destinations; it does not overwrite or merge a live run.

Files are restored byte-for-byte. `fits/FIT_KEY/fit.npz` is the completed native
forecast state. It embeds the permitted training panel and derived features
because the existing NativeFit serializer requires them; this package does not
include the separate raw/canonical data cache or an environment.

Use the recorded repository numerical source and compatible NumPy/SciPy/BLAS
runtime. Obtain the original verified KS cache through its separately authorized
route. Construct the correct KSAdapter protocol and trajectory, then call the
unchanged `fit_native('bayes_stesn', training.panel, config, seed,
checkpoint_path=RESTORED_FIT_PATH)`. An existing completed fit is reused only if
its data, transform, configuration, source and runtime identity matches.
Do not bypass this guard or relabel an inference-unqualified checkpoint.

In all-fit-checkpoints mode, the exact canonical chain and KS phase sidecars are
retained when available. If native fit.npz is absent, matching `fit_native` may
resume these within their original allowance. A completed unqualified phase
does not permit another extension. The original data are needed to reconstruct
the prepared model for an incomplete fit. Phase-only files may contain no saved
chain progress; their prepared identity must be checked by the original adapter.

`evidence/` contains the separately verified compact archive and manifest with
configuration, source, selections, scores, terminal receipts and elapsed-work
accounting. Unzip that archive separately for inspection. Job files contain
original absolute paths: do not execute restored worker jobs directly. A new
coordinator root requires an explicit compatible import and new job identities;
preserve cumulative worker/wall budgets. No paths or identities are rewritten by
this restorer. Selected-only mode is forecast-state backup, not full campaign or
sampler continuation storage. Missing expected selected fits remain missing.
"""


def create(root, output, compact_manifest, mode, *, part_mib=512, apply=False):
    if not apply:
        raise ValueError("Binary checkpoint reads require explicit create --apply after coordinator exit")
    if not 1 <= part_mib <= 512:
        raise ValueError("Part cap must be 1..512 MiB")
    root = safe_path(root, directory=True); output = Path(os.path.abspath(output))
    if output.exists() or output.is_relative_to(root) or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Use a new backup directory outside the run")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        locks = acquire_locks(root, stack)
        plan, exp, jobs, frozen = inventory(root, mode)
        if mode == "selected-only" and frozen is None:
            raise ValueError("Selected-only backup requires a frozen selection")
        compact_path, compact_archive = compact_binding(compact_manifest, root, exp, frozen)
        if not plan["fits"]:
            raise ValueError("No eligible native fit/checkpoint files")
        reserve = max(2*1024**3, int(exp["config"]["budget"]["disk_reserve_bytes"]))
        approximate = plan["canonical_binary_bytes"]+compact_archive.stat().st_size
        approximate += sum((root/"fits"/r["fit_key"]/"receipt.json").stat().st_size for r in plan["fits"] if r["receipt_present"])
        if shutil.disk_usage(output.parent).free < reserve+approximate+64*MIB:
            raise RuntimeError("Insufficient free space above the original disk reserve")
        records, members = [], []
        for row in plan["fits"]:
            summary, checked = validate_fit(root, row, jobs[row["fit_key"]][0], exp, mode)
            records.append(summary)
            for name, validation in checked.items():
                p = root/"fits"/row["fit_key"]/name
                members.append(source_row(p, p.relative_to(root).as_posix(), validation))
            folder = root/"fits"/row["fit_key"]
            for p in ([folder/"receipt.json"] if row["receipt_present"] else [])+sorted(folder.glob("synchronization_*.json")):
                read_json(p); members.append(source_row(p, p.relative_to(root).as_posix()))
            for j in jobs[row["fit_key"]]:
                p = root/"jobs"/(j["job_id"]+".json")
                members.append(source_row(p, p.relative_to(root).as_posix()))
        members.extend([source_row(compact_path, "evidence/compact_manifest.json"),
                        source_row(compact_archive, "evidence/"+compact_archive.name)])
        unique = {row["name"]: row for row in members}
        if len(unique) != len(members):
            raise ValueError("Duplicate source member")
        members = [unique[key] for key in sorted(unique)]
        required = sum(row["bytes"]+512+2*len(row["name"].encode()) for row in members)
        if shutil.disk_usage(output.parent).free < reserve+required+MIB:
            raise RuntimeError("Final archive size bound violates the original disk reserve")
        staging = Path(tempfile.mkdtemp(prefix="."+output.name+"-", dir=output.parent))
        parts = store_parts(staging, members, part_mib*MIB, reserve)
        helper = staging/"restore_ks_improvement_fits.py"; helper.write_bytes(Path(__file__).read_bytes())
        instructions = staging/"RESTORE.md"; instructions.write_text(RESTORE_TEXT)
        manifest = {"schema": SCHEMA, "created_utc": utc(), "mode": mode,
            "experiment_identity": exp["identity"], "freeze_identity": plan["freeze_identity"],
            "source_and_runtime": exp["source"], "configuration_identity": identity(exp["config"]),
            "compact_evidence_manifest_sha256": digest(compact_path), "part_limit_bytes": part_mib*MIB,
            "parts": parts, "members": [{k: v for k, v in row.items() if k not in {"source", "fingerprint"}} for row in members],
            "source_bytes": sum(row["bytes"] for row in members), "fit_count": len(records), "fits": records,
            "disk_reserve_bytes": reserve, "quiescence_locks": locks, "source_payloads_changed": False,
            "original_run_deleted_or_modified": False, "job_paths_rewritten": False,
            "native_internal_state_checksums_verified": sum(len(r["files"]) for r in plan["fits"]),
            "native_receipt_verified_count": sum(bool(r["native_receipt_verified"]) for r in records),
            "forecast_state_count": sum(bool(r["forecast_state_available"]) for r in records),
            "selected_fit_coverage": plan["selected_fit_coverage"],
            "inference_restart_state_count": sum(bool(r["inference_restart_state_available"]) for r in records),
            "qualified_native_fit_count": sum(bool(r["qualified"]) for r in records),
            "scope": "Native state backup; completion and qualification are per-fit. Separate verified data/cache and matching code/runtime remain required. Archived worker jobs are not directly relocation-resumable.",
            "helper": {"name": helper.name, "bytes": helper.stat().st_size, "sha256": digest(helper)},
            "instructions": {"name": instructions.name, "bytes": instructions.stat().st_size, "sha256": digest(instructions)}}
        (staging/"manifest.json").write_bytes(encoded(manifest))
        (staging/"manifest.sha256").write_text(digest(staging/"manifest.json")+"  manifest.json\n")
        verified = verify(staging); (staging/"verification.json").write_bytes(encoded(verified))
        for path in staging.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        if output.exists():
            raise FileExistsError("Backup destination appeared concurrently")
        os.rename(staging, output)
        fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return {"package": str(output), **verified}


def package_manifest(package):
    package = safe_path(package, directory=True); path = safe_path(package/"manifest.json")
    expected, name = safe_path(package/"manifest.sha256").read_text().split()
    if name != "manifest.json" or not SHA.fullmatch(expected) or digest(path) != expected:
        raise ValueError("Manifest checksum mismatch")
    manifest = read_json(path)
    if manifest["schema"] != SCHEMA or not 1 <= manifest["part_limit_bytes"] <= MAX_PART:
        raise ValueError("Incompatible backup manifest")
    if len({r["name"] for r in manifest["members"]}) != len(manifest["members"]):
        raise ValueError("Duplicate manifest member")
    for row in manifest["members"]:
        safe_member(row["name"])
        if not SHA.fullmatch(row["sha256"]) or not isinstance(row["bytes"], int) or row["bytes"] < 0:
            raise ValueError("Invalid member checksum/size")
    for label in ("helper", "instructions"):
        row = manifest[label]; p = safe_path(package/safe_member(row["name"]))
        if p.stat().st_size != row["bytes"] or digest(p) != row["sha256"]:
            raise ValueError("Backup helper/instructions changed")
    return package, manifest


def verify(package, destination=None):
    package, manifest = package_manifest(package)
    expected = {row["name"]: row for row in manifest["members"]}; seen = set(); count = total = 0
    for index, part in enumerate(manifest["parts"], 1):
        if part["name"] != f"native_fits.part{index:04d}.zip":
            raise ValueError("Part names must be contiguous and safe")
        path = safe_path(package/part["name"])
        if not 0 < part["bytes"] <= manifest["part_limit_bytes"] or path.stat().st_size != part["bytes"] or digest(path) != part["sha256"]:
            raise ValueError("Part checksum/size mismatch")
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if [x.filename for x in entries] != part["members"]:
                raise ValueError("Part member inventory mismatch")
            for info in entries:
                name = safe_member(info.filename); row = expected.get(name)
                mode = info.external_attr >> 16
                if (name in seen or row is None or row["part"] != part["name"] or info.is_dir()
                        or stat.S_IFMT(mode) not in (0, stat.S_IFREG) or info.file_size != row["bytes"]):
                    raise ValueError("Unsafe or incompatible backup member")
                seen.add(name); h = hashlib.sha256(); size = 0
                target = None
                if destination is not None:
                    p = destination.joinpath(*PurePosixPath(name).parts); p.parent.mkdir(parents=True, exist_ok=True)
                    target = p.open("xb")
                try:
                    with archive.open(info) as stream:
                        while block := stream.read(MIB):
                            size += len(block); h.update(block)
                            if target is not None:
                                target.write(block)
                    if target is not None:
                        target.flush(); os.fsync(target.fileno())
                finally:
                    if target is not None:
                        target.close()
                if size != row["bytes"] or h.hexdigest() != row["sha256"]:
                    raise ValueError("Member SHA-256/ZIP CRC mismatch")
                count += 1; total += size
    if seen != set(expected) or total != manifest["source_bytes"]:
        raise ValueError("Incomplete native backup")
    return {"verified": True, "manifest_sha256": digest(package/"manifest.json"),
        "mode": manifest["mode"], "parts": len(manifest["parts"]), "members": count,
        "source_bytes": total, "fit_count": manifest["fit_count"],
        "experiment_identity": manifest["experiment_identity"], "freeze_identity": manifest["freeze_identity"]}


def restore(package, destination):
    checked = verify(package); package, manifest = package_manifest(package)
    destination = Path(os.path.abspath(destination))
    if destination.exists() or any(p.is_symlink() for p in (destination, *destination.parents)):
        raise FileExistsError("Restore requires a fresh destination; no merge/overwrite is provided")
    parent = safe_path(destination.parent, directory=True)
    if shutil.disk_usage(parent).free < manifest["source_bytes"]+manifest["disk_reserve_bytes"]+MIB:
        raise RuntimeError("Insufficient restore space above the retained disk reserve")
    staging = Path(tempfile.mkdtemp(prefix="."+destination.name+"-", dir=parent))
    verify(package, staging)
    receipt = {**checked, "restored_utc": utc(), "job_paths_rewritten": False,
        "existing_checkpoints_overwritten": False, "native_checkpoint_bytes_unchanged": True,
        "resume_condition": "Use unchanged native identity checks with original verified data and compatible numerical source/runtime; do not execute archived absolute-path jobs."}
    (staging/"RESTORE_RECEIPT.json").write_bytes(encoded(receipt))
    if destination.exists():
        raise FileExistsError("Restore destination appeared concurrently")
    os.rename(staging, destination)
    return {"destination": str(destination), **receipt}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "create"):
        p = sub.add_parser(command); p.add_argument("--run-root", type=Path, required=True)
        modes = p.add_mutually_exclusive_group(required=True)
        modes.add_argument("--selected-only", action="store_true")
        modes.add_argument("--all-fit-checkpoints", action="store_true")
        if command == "create":
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--compact-manifest", type=Path, required=True)
            p.add_argument("--part-mib", type=int, default=512)
            p.add_argument("--apply", action="store_true")
    p = sub.add_parser("verify"); p.add_argument("--package", type=Path, required=True)
    p = sub.add_parser("restore"); p.add_argument("--package", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in ("plan", "create"):
        mode = "selected-only" if args.selected_only else "all-fit-checkpoints"
        if args.command == "plan":
            result = inventory(args.run_root, mode)[0]
            result = {k: v for k, v in result.items() if k != "fits"}
        else:
            result = create(args.run_root, args.output, args.compact_manifest, mode,
                            part_mib=args.part_mib, apply=args.apply)
    elif args.command == "verify":
        result = verify(args.package)
    else:
        result = restore(args.package, args.destination)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
