"""Atomic JSON + numeric NPZ storage. No executable/pickled objects are loaded."""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict, is_dataclass
from collections.abc import Mapping
import hashlib
import json
import os
import platform
import sys
import tempfile
import numpy as np
import scipy

SCHEMA_VERSION = "1.0"


def seed_for(seed: int, *keys) -> int:
    """Stable named substream; independent of process hashes or worker order."""
    payload = json.dumps([int(seed), *keys], sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
            return np.array(arrays[value["__array__"]], copy=True)
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
        digest.update(array.tobytes())
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

