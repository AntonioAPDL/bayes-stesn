"""Atomic JSON + numeric NPZ storage. No executable/pickled objects are loaded."""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict, is_dataclass
from collections.abc import Mapping
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
import time
import numpy as np
import scipy

SCHEMA_VERSION = "1.0"


def seed_for(seed: int, *keys) -> int:
    """Stable named substream; independent of process hashes or worker order."""
    payload = json.dumps([int(seed), *keys], sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def atomic_npz(path, **arrays):
    """Replace one numeric archive atomically, cleaning every handled failure.

    Before ``os.replace`` a failure leaves the previous canonical file intact.
    A directory-fsync failure after replacement leaves the complete new file
    in place, but indicates that rename durability could not be confirmed.
    Hard termination cannot run ``finally``; :func:`cleanup_stale_npz` only
    removes this writer's recognizable staging files from dead local PIDs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    host = hashlib.sha256(platform.node().encode()).hexdigest()[:12]
    prefix = f".{path.name}.bstesn-{host}-{os.getpid()}-"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=prefix, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if hasattr(os, "O_DIRECTORY"):
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def cleanup_stale_npz(path, *, older_than_seconds=86400, dry_run=True):
    """Inspect/remove owned staging archives left by a dead local writer.

    This opt-in recovery never touches canonical files, foreign-host files,
    symlinks, files owned by another user, or a PID that may still be alive.
    PID reuse can conservatively retain a stale file. The age threshold is
    an additional guard, not evidence that a live writer is abandoned.
    Returns the eligible paths; ``dry_run=True`` is the safe default.
    """
    if not np.isfinite(older_than_seconds) or older_than_seconds < 0:
        raise ValueError("older_than_seconds must be finite and nonnegative")
    path = Path(path)
    if not path.parent.exists():
        return []
    host = hashlib.sha256(platform.node().encode()).hexdigest()[:12]
    pattern = re.compile(re.escape(f".{path.name}.bstesn-{host}-") + r"([0-9]+)-[A-Za-z0-9_]+\.npz\Z")
    cutoff = time.time() - older_than_seconds
    eligible = []
    for candidate in path.parent.iterdir():
        match = pattern.fullmatch(candidate.name)
        if match is None or candidate.is_symlink() or not candidate.is_file():
            continue
        try:
            stat = candidate.stat()
        except FileNotFoundError:
            # Another cleanup or a completed writer may have removed it.
            continue
        if stat.st_mtime > cutoff or (hasattr(os, "geteuid") and stat.st_uid != os.geteuid()):
            continue
        try:
            os.kill(int(match.group(1)), 0)
        except ProcessLookupError:
            eligible.append(candidate)
            if not dry_run:
                candidate.unlink(missing_ok=True)
        except (PermissionError, OverflowError):
            # An inaccessible or unrepresentable PID is not proven dead.
            continue
    return sorted(eligible)


def _pack(value, arrays):
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            raise TypeError("Object arrays are forbidden in portable storage")
        key = f"a{len(arrays):06d}"
        arrays[key] = value
        return {"__array__": key}
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _pack(asdict(value), arrays)
    if isinstance(value, Mapping):
        # Stable traversal is essential: array keys must not depend on the
        # insertion order that changes when JSON is parsed after sort_keys.
        return {str(k): _pack(value[k], arrays) for k in sorted(value, key=str)}
    if isinstance(value, tuple):
        return {"__tuple__": [_pack(v, arrays) for v in value]}
    if isinstance(value, list):
        return [_pack(v, arrays) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsafe/unregistered serialization type: {type(value).__name__}")


def _unpack(value, arrays):
    if isinstance(value, list):
        return [_unpack(v, arrays) for v in value]
    if isinstance(value, dict):
        if set(value) == {"__array__"}:
            # NpzFile.__getitem__ already returns a fresh owned allocation.
            # It is detached from the archive when the file closes.
            return arrays[value["__array__"]]
        if set(value) == {"__tuple__"}:
            return tuple(_unpack(v, arrays) for v in value["__tuple__"])
        return {k: _unpack(v, arrays) for k, v in value.items()}
    return value


def content_hash(value) -> str:
    arrays = {}
    manifest = _pack(value, arrays)
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False, separators=(",", ":")).encode())
    for key, array in arrays.items():
        array = np.ascontiguousarray(array)
        digest.update(key.encode())
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        # A contiguous byte view preserves the historical hash exactly and
        # avoids another full-size allocation for large forecast archives.
        if array.nbytes:
            digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def save_state(path, state, *, kind="state"):
    arrays = {}
    packed = _pack(state, arrays)
    metadata = {"schema_version": SCHEMA_VERSION, "kind": kind,
                "sha256": content_hash(state), "state": packed}
    arrays["__manifest__"] = np.array(json.dumps(metadata, sort_keys=True, allow_nan=False))
    atomic_npz(path, **arrays)
    return metadata["sha256"]


def load_state(path, *, kind=None):
    with np.load(path, allow_pickle=False) as arrays:
        manifest = json.loads(str(arrays["__manifest__"]))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported state schema version")
        if kind is not None and manifest.get("kind") != kind:
            raise ValueError(f"Expected {kind} artifact, found {manifest.get('kind')}")
        state = _unpack(manifest["state"], arrays)
    if content_hash(state) != manifest["sha256"]:
        raise ValueError("Artifact content checksum does not match manifest")
    return state


def environment():
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "platform": platform.platform(),
            "blas_threads": {k: os.environ.get(k) for k in
                             ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}

