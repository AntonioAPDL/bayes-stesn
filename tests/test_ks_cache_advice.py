"""Guarded operational cache advice, tested without advising real task files."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "ks_cache_advice_test", Path(__file__).resolve().parents[1]/"tools/reclaim_ks_file_cache.py")
advice = importlib.util.module_from_spec(spec)
spec.loader.exec_module(advice)


@pytest.fixture
def work(tmp_path, monkeypatch):
    results = tmp_path/"results"/"synthetic"; results.mkdir(parents=True)
    (results/"experiment.json").write_text(json.dumps({"identity": "synthetic"}))
    (results/"scheduler.lock").touch()
    (tmp_path/"cache").mkdir()
    payload = tmp_path/"cache"/"data.bin"; payload.write_bytes(b"x"*1024**2)
    (tmp_path/"cache"/"alias.bin").symlink_to(payload)
    os.link(payload, tmp_path/"cache"/"hardlink.bin")
    monkeypatch.setattr(advice, "worker_census", lambda root: (0, 0))
    return tmp_path, results, payload


def test_dry_run_reads_metadata_only_and_deduplicates_links(work, monkeypatch):
    workspace, results, payload = work
    monkeypatch.setattr(os, "posix_fadvise", lambda *args: pytest.fail("Dry-run must not advise"))
    result = advice.reclaim(workspace, results)
    assert result["state"] == "dry_run" and result["quiescent"]
    assert result["eligible_files"] == 1 and result["files_advised"] == 0
    assert result["skipped"]["nonregular_or_symlink"] == 1
    assert result["skipped"]["duplicate_hardlink"] == 1


def test_apply_uses_read_only_fd_preserves_payload_and_refuses_live_task(work, monkeypatch):
    workspace, results, payload = work
    calls = []
    def capture(fd, offset, length, operation):
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        assert flags & os.O_ACCMODE == os.O_RDONLY
        calls.append((offset, length, operation))
    monkeypatch.setattr(os, "posix_fadvise", capture)
    before = advice.fingerprint(payload.stat())
    result = advice.reclaim(workspace, results, apply=True)
    assert result["files_advised"] == 1 and len(calls) == 1
    assert advice.fingerprint(payload.stat()) == before
    assert payload.read_bytes() == b"x"*1024**2
    monkeypatch.setattr(advice, "worker_census", lambda root: (1, 0))
    with pytest.raises(RuntimeError, match="not verified quiescent"):
        advice.reclaim(workspace, results, apply=True)
    assert len(calls) == 1


def test_busy_scheduler_refuses_apply_and_existing_receipt_is_preserved(work):
    workspace, results, payload = work
    with (results/"scheduler.lock").open("r+") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not advice.reclaim(workspace, results)["quiescent"]
        with pytest.raises(RuntimeError, match="not verified quiescent"):
            advice.reclaim(workspace, results, apply=True)
    receipt = workspace/"receipt.json"
    advice.atomic_json(receipt, {"original": True})
    with pytest.raises(FileExistsError):
        advice.atomic_json(receipt, {"original": False})
    assert json.loads(receipt.read_text()) == {"original": True}
