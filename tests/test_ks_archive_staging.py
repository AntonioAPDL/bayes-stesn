"""Focused safety checks for the seven-file task archive cleanup policy."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import zipfile

import pytest


spec = importlib.util.spec_from_file_location(
    "ks_archive_cleanup_test", Path(__file__).resolve().parents[1]/"tools/clean_ks_archive_staging.py")
cleaner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleaner)


@pytest.fixture
def task(tmp_path, monkeypatch):
    results = tmp_path/"results"/"synthetic"; results.mkdir(parents=True)
    (results/"scheduler.lock").touch()
    (results/"experiment.json").write_text(json.dumps({"identity": "synthetic",
        "config": {"schema": "ks-experiment-1", "name": "custom_literature_grounded_KS_20261105"}}))
    directory = tmp_path/"deliverables"; directory.mkdir()
    canonical = directory/"canonical.zip"
    with zipfile.ZipFile(canonical, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("record.txt", b"a meaningful checkpoint payload"*100)
    payload = canonical.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    (directory/"canonical.receipt.json").write_text(json.dumps({"bytes": len(payload), "sha256": digest}))
    for name, content in {".canonical.zip.full": payload, ".canonical.zip.prefix": payload[:713],
                          ".canonical.zip.wrong": b"wrong contents", ".canonical.zip.unknown": payload}.items():
        path = directory/name; path.write_bytes(content)
        os.utime(path, (time.time()-300, time.time()-300))
    (directory/".canonical.zip.link").symlink_to(canonical)
    group = {"directory": "deliverables", "canonical": "canonical.zip", "bytes": len(payload),
        "sha256": digest, "kind": "zip", "receipt": "canonical.receipt.json",
        "staging": (".canonical.zip.full", ".canonical.zip.prefix", ".canonical.zip.wrong", ".canonical.zip.link")}
    monkeypatch.setattr(cleaner, "GROUPS", (group,))
    # Actual outer descriptor checks remain live, but this synthetic fixture is
    # independent of the concurrently executing real scientific campaign.
    monkeypatch.setattr(cleaner.cache_helper, "worker_census", lambda root: (0, 0))
    descriptor_check = cleaner.writable_descriptors
    monkeypatch.setattr(cleaner, "writable_descriptors",
        lambda targets: descriptor_check(targets, processes=[Path("/proc/self").resolve()]))
    return tmp_path, results, directory, canonical, payload


def run(task, name, **kwargs):
    workspace, results, *_ = task
    return cleaner.clean(workspace, results, workspace/(name+".json"), **kwargs)


def test_exact_prefix_full_copy_crc_and_one_pass_no_retry(task):
    workspace, results, directory, canonical, payload = task
    dry = run(task, "dry")
    assert dry["state"] == "dry_run_complete" and dry["eligible_files"] == 2
    assert dry["deleted_files"] == 0
    assert dry["canonical_verification"][0]["zip_crc_valid"]
    assert (directory/".canonical.zip.full").exists()
    applied = run(task, "apply", apply=True)
    assert applied["deleted_files"] == 2
    assert applied["deleted_logical_bytes"] == len(payload)+713
    assert canonical.read_bytes() == payload
    assert applied["canonical_verification"][0]["post_pass_unchanged"]
    assert (directory/".canonical.zip.wrong").exists()
    assert (directory/".canonical.zip.link").is_symlink()
    assert (directory/".canonical.zip.unknown").exists()
    (directory/".canonical.zip.full").write_bytes(payload)
    again = run(task, "no_retry", apply=True)
    assert again["state"] == "blocked" and again["deleted_files"] == 0
    assert (directory/".canonical.zip.full").exists()


def test_busy_scheduler_and_actual_writable_descriptor_block(task, monkeypatch):
    workspace, results, directory, canonical, payload = task
    with (results/"scheduler.lock").open("r+") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        busy = run(task, "busy", apply=True)
    assert busy["state"] == "blocked" and busy["deleted_files"] == 0
    with (directory/".canonical.zip.prefix").open("r+b"):
        writable = run(task, "writable", apply=True)
    assert writable["state"] == "blocked" and writable["deleted_files"] == 0
    assert writable["writable_descriptors"]
    assert not (results/"cleanup"/cleaner.MARKER).exists()
    monkeypatch.setattr(cleaner, "writable_descriptors", lambda targets: ([], 1))
    unresolved = run(task, "unresolved", apply=True)
    assert unresolved["state"] == "blocked" and unresolved["deleted_files"] == 0


def test_canonical_corruption_rejects_before_any_delete(task):
    workspace, results, directory, canonical, payload = task
    changed = bytearray(payload); changed[100] ^= 1; canonical.write_bytes(changed)
    result = run(task, "corrupt", apply=True)
    assert result["eligible_files"] == 0 and result["deleted_files"] == 0
    assert any(row.get("reason") == "canonical_sha256_mismatch" for row in result["files"])
    assert (directory/".canonical.zip.full").exists()
    assert not (results/"cleanup"/cleaner.MARKER).exists()


def test_identity_change_after_verification_is_retained(task, monkeypatch):
    workspace, results, directory, canonical, payload = task
    original = cleaner.writable_descriptors; calls = 0
    def changing(targets):
        nonlocal calls
        calls += 1
        # The first call is the initial census. Replace one file just before
        # the first deletion, after its content and fingerprint were verified.
        if calls == 2:
            (directory/".canonical.zip.full").unlink()
            (directory/".canonical.zip.full").write_bytes(b"replacement must survive")
        return original(targets)
    monkeypatch.setattr(cleaner, "writable_descriptors", changing)
    result = run(task, "changed", apply=True)
    full = next(row for row in result["files"] if row["relative_path"].endswith(".full"))
    assert full["action"] == "retained" and full["reason"] == "identity_changed_before_deletion"
    assert (directory/".canonical.zip.full").read_bytes() == b"replacement must survive"
    assert canonical.read_bytes() == payload
