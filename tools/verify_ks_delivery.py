#!/usr/bin/env python3
"""Verify the exact, already-downloaded KS delivery; optionally restore it.

This standard-library-only command has no network, authentication, or credential
handling. Download the four authorized binary objects first, preserving the
basenames below. No archive is extracted before all four hashes, the complete
catalog, and the catalog's KS/source-v2 selection have been checked.

    python tools/verify_ks_delivery.py --incoming DOWNLOAD_DIR
    python tools/verify_ks_delivery.py --incoming DOWNLOAD_DIR --restore CACHE_ROOT

The optional restore invokes the verified supplied restore.py twice: dry-run,
then actual KS-only restore. That helper preserves conflicting files and has no
force-overwrite mode. Receipts contain hashes and counts, never account details,
private Drive identifiers, URLs, or local absolute paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile


EXPECTED = {
    "ks_prepared_v1.tar.gz": (182334299, "898f18deccc1fe6b3fb4d6eb0aa847244dc115efc56a260aceaae07bc22c227c"),
    "source_and_metadata_v2.tar.gz": (9981003, "c88f8475b97a9b7159e643e06e424d0f58630400182957602e48b91097422965"),
    "drive_catalog.json": (633855, "510a14dfa23e66fc861066dc4707fc46a192ba13588af2ba30f1e69478ce6be0"),
    "restore.py": (23108, "9711bbb6a78c59e106134f3ebd64c151e45593b9d8b46304d2d81759c833b282"),
}
ARCHIVE_DESTINATIONS = {
    "ks_prepared_v1.tar.gz": "data/ks/exports/ks_prepared_v1.tar.gz",
    "source_and_metadata_v2.tar.gz": "exports/source_and_metadata_v2.tar.gz",
}
SOURCE_GROUPS = {"code", "sourcecode", "control"}


class DeliveryError(ValueError):
    """A preserved local object does not satisfy the exact delivery contract."""


def verify_file(path: Path, size: int, digest: str) -> dict:
    """Stream a regular file and reject changed size/content or symlink input."""
    if not path.is_file() or path.is_symlink():
        raise DeliveryError(f"Missing regular delivery file: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise DeliveryError(f"Delivery input is not a regular file: {path.name}")
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.fstat(stream.fileno())
    if before.st_size != size or actual != digest:
        raise DeliveryError(f"Size/SHA-256 mismatch for {path.name}; preserve it and obtain the exact supplied object")
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise DeliveryError(f"Delivery file changed while hashing: {path.name}")
    return {"name": path.name, "bytes": size, "sha256": actual, "verified": True}


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DeliveryError("Duplicate JSON object key in complete catalog")
        result[key] = value
    return result


def _constant(value):
    raise DeliveryError(f"Invalid nonfinite JSON value: {value}")


def _relative(value):
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise DeliveryError("Unsafe relative path in catalog")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")) or ":" in path.parts[0]:
        raise DeliveryError("Unsafe relative path in catalog")
    return path


def validate_catalog(catalog: dict) -> dict:
    """Validate every entry and the exact two-object default KS restore set."""
    if not isinstance(catalog, dict) or catalog.get("schema_version") != "benchmark-drive-catalog/1.0":
        raise DeliveryError("Unexpected complete-catalog schema")
    entries = catalog.get("artifacts")
    if not isinstance(entries, list) or len(entries) != 362 or catalog.get("file_count") != len(entries):
        raise DeliveryError("The final catalog must contain all 362 artifact records")
    if catalog.get("status") != "complete_verified" or catalog.get("incomplete") != []:
        raise DeliveryError("Catalog does not identify a complete verified collection")
    selected, destinations = [], set()
    required = {"relative_path", "name", "bytes", "sha256", "restore_kind", "dataset", "group", "file_id"}
    for entry in entries:
        if not isinstance(entry, dict) or not required <= entry.keys():
            raise DeliveryError("Incomplete artifact record in catalog")
        relative = str(_relative(entry["relative_path"]))
        if relative in destinations:
            raise DeliveryError("Duplicate artifact destination in catalog")
        destinations.add(relative)
        if len(_relative(entry["name"]).parts) != 1:
            raise DeliveryError("Catalog object name is not a basename")
        if type(entry["bytes"]) is not int or entry["bytes"] < 0:
            raise DeliveryError("Invalid catalog byte count")
        digest = entry["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise DeliveryError("Invalid catalog SHA-256")
        if entry["restore_kind"] not in {"file", "tar", "part"}:
            raise DeliveryError("Unknown catalog restore kind")
        # Matches the supplied helper with --dataset ks and no --include-raw.
        chosen = entry["group"] in SOURCE_GROUPS or (entry["dataset"] == "ks" and entry["group"] not in {"raw", "raw_continuations"})
        if chosen:
            selected.append(entry)
    if len(selected) != 2 or {entry["name"] for entry in selected} != set(ARCHIVE_DESTINATIONS):
        raise DeliveryError("KS selection is not exactly KS data v1 plus active source/metadata v2")
    safe = []
    for entry in selected:
        size, digest = EXPECTED[entry["name"]]
        if (entry["bytes"], entry["sha256"], entry["relative_path"], entry["restore_kind"]) != (size, digest, ARCHIVE_DESTINATIONS[entry["name"]], "tar"):
            raise DeliveryError("Selected catalog archive does not match the delivered object")
        if entry.get("verification_status") != "verified":
            raise DeliveryError("Selected catalog archive lacks its verification status")
        safe.append({key: entry[key] for key in ("name", "bytes", "sha256", "relative_path", "restore_kind", "dataset", "group")})
    return {"artifact_count": len(entries), "status": "complete_verified", "selected_artifact_count": 2,
            "selected_artifact_bytes": sum(entry["bytes"] for entry in selected),
            "selected": sorted(safe, key=lambda entry: entry["name"]),
            "additional_dataset_payloads_selected": False}


def verify_delivery(incoming: Path) -> dict:
    incoming = Path(incoming).resolve(strict=True)
    if not incoming.is_dir():
        raise DeliveryError("Incoming delivery root must be a directory")
    files = [verify_file(incoming / name, size, digest) for name, (size, digest) in EXPECTED.items()]
    # Read the complete verified catalog; never transmit or print its contents.
    catalog = json.loads((incoming / "drive_catalog.json").read_text(encoding="utf-8"),
                         object_pairs_hook=_pairs, parse_constant=_constant)
    return {"schema": "ks-delivery-verification/1", "status": "verified",
            "files": files, "file_count": len(files), "delivery_bytes": sum(item["bytes"] for item in files),
            "catalog": validate_catalog(catalog), "network_access": False, "credentials_used": False,
            "account_access_checked_by_this_offline_tool": False, "source_files_deleted": False}


def _restore(incoming: Path, destination: Path, reserve_bytes: int | None) -> list[dict]:
    repo = Path(__file__).resolve().parents[1]
    incoming, destination = incoming.resolve(strict=True), destination.resolve()
    for protected, label in ((repo, "model repository"), (incoming, "incoming delivery directory")):
        if destination == protected or destination.is_relative_to(protected) or protected.is_relative_to(destination):
            raise DeliveryError(f"Restore cache must be separate from the {label}")
    command = [sys.executable, str(incoming / "restore.py"), "--catalog", str(incoming / "drive_catalog.json"),
               "--incoming", str(incoming), "--destination", str(destination), "--dataset", "ks"]
    ancestor = destination
    while not ancestor.exists():
        ancestor = ancestor.parent
    initial_free = shutil.disk_usage(ancestor).free
    reserve = max(2*1024**3, int(.2*initial_free), reserve_bytes or 0)
    stages = []
    for dry_run in (True, False):
        if not dry_run:
            remaining = shutil.disk_usage(ancestor).free - stages[0]["estimated_additional_bytes_before_joined_tar_expansion"]
            if remaining < reserve:
                raise DeliveryError("Restore disk-reserve gate failed; use another authorized cache or reduce verified redundant storage")
        completed = subprocess.run(command + (["--dry-run"] if dry_run else []),
                                   capture_output=True, text=True, check=False)
        if completed.returncode:
            stage = "dry-run" if dry_run else "restore"
            try:
                failure_message = json.loads(completed.stdout).get("message", "")
            except (ValueError, AttributeError):
                failure_message = ""
            raise DeliveryError(f"Supplied {stage} failed with exit {completed.returncode}; files remain preserved. "
                                + (failure_message or completed.stderr.strip())[-2000:])
        raw = json.loads(completed.stdout, object_pairs_hook=_pairs, parse_constant=_constant)
        expected_status = "verified_dry_run" if dry_run else "restored_verified"
        if raw.get("status") != expected_status or raw.get("selected_artifact_count") != 2 or raw.get("datasets") != ["ks"]:
            raise DeliveryError("Supplied restore receipt disagrees with the exact KS selection")
        fields = ("status", "catalog_sha256", "datasets", "selected_artifact_count", "selected_artifact_bytes",
                  "verified_tar_member_count", "estimated_additional_bytes_before_joined_tar_expansion",
                  "multipart_archive_count", "source_files_deleted", "network_access")
        stages.append({key: raw[key] for key in fields})
    stages[-1]["disk_reserve_bytes"] = reserve
    return stages


def _write_receipt(path: Path, receipt: dict):
    data = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == data:
            return
        raise DeliveryError("Receipt already exists with different content; choose a new receipt filename")
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + "-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)  # Exclusive installation never replaces a different receipt.
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incoming", type=Path, required=True)
    parser.add_argument("--restore", type=Path, metavar="CACHE_ROOT")
    parser.add_argument("--receipt", type=Path, help="Optional public-safe JSON receipt; conflicting files are preserved")
    parser.add_argument("--disk-reserve-bytes", type=int, help="Optional stricter restore reserve, additional to max(2 GiB, 20%% initial free)")
    args = parser.parse_args()
    if args.disk_reserve_bytes is not None and args.disk_reserve_bytes < 0:
        parser.error("--disk-reserve-bytes must be nonnegative")
    try:
        receipt = verify_delivery(args.incoming)
        if args.restore is not None:
            receipt["restore"] = _restore(args.incoming, args.restore, args.disk_reserve_bytes)
            receipt["status"] = "restored_verified"
        if args.receipt is not None:
            _write_receipt(args.receipt, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    except (DeliveryError, OSError, json.JSONDecodeError) as error:
        parser.exit(1, f"KS delivery verification failed: {error}\n")


if __name__ == "__main__":
    main()
