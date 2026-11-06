"""Read-only cross-namespace monitoring: lifetime checks, uncertainty and locks."""
from __future__ import annotations

import errno
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("inspect_ks_execution",
    Path(__file__).resolve().parents[1]/"tools/inspect_ks_execution.py")
inspection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inspection)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def process_fixture(root, pid=4101, *, ticks="123456", state="S", namespace_pid=7):
    boot = root/"sys/kernel/random/boot_id"
    boot.parent.mkdir(parents=True, exist_ok=True)
    boot.write_text("fixture-boot\n")
    folder = root/str(pid)
    (folder/"ns").mkdir(parents=True, exist_ok=True)
    tail = [state, "1"] + ["0"]*17 + [ticks]
    (folder/"stat").write_text(f"{pid} (python (worker) with spaces) " + " ".join(tail))
    (folder/"status").write_text(f"Name:\tpython\nNSpid:\t{pid}\t{namespace_pid}\n")
    (folder/"ns/pid").write_text("namespace fixture")
    return {"pid": pid, "namespace_pid": namespace_pid, "start_ticks": ticks,
            "boot_id": "fixture-boot", "pid_namespace": (folder/"ns/pid").stat().st_ino}


def deny_namespace_stat(monkeypatch, proc):
    original = os.stat
    def inaccessible(path, *args, **kwargs):
        if str(path).startswith(str(proc)+"/") and str(path).endswith("/ns/pid"):
            raise PermissionError(errno.EACCES, "Namespace inode inaccessible", str(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(os, "stat", inaccessible)


def test_permission_limited_namespace_is_live_when_lifetime_matches(tmp_path, monkeypatch):
    record = process_fixture(tmp_path)
    deny_namespace_stat(monkeypatch, tmp_path)
    result = inspection.inspect_process(record, proc_root=tmp_path)
    assert result["state"] == "active"
    assert result["validated"] == {key: record[key] for key in
                                   ("pid", "namespace_pid", "start_ticks", "boot_id")}
    assert result["namespace_inode"]["errno"] == errno.EACCES
    assert result["legacy_match"] is False
    assert "namespace_inode_unavailable" in result["legacy_disagreement"]


@pytest.mark.parametrize("field,value,reason", [
    ("start_ticks", "999999", "start_ticks_changed"),
    ("boot_id", "older-boot", "boot_id_changed"),
    ("namespace_pid", 99, "namespace_pid_changed"),
    ("pid_namespace", 1, "pid_namespace_changed"),
])
def test_reused_or_mismatched_lifetime_is_inactive(tmp_path, field, value, reason):
    record = process_fixture(tmp_path)
    record[field] = value
    result = inspection.inspect_process(record, proc_root=tmp_path)
    assert result["state"] == "inactive"
    assert result["reason"] == reason


@pytest.mark.parametrize("state", ["Z", "X"])
def test_zombie_and_dead_processes_are_inactive(tmp_path, state):
    record = process_fixture(tmp_path, state=state)
    result = inspection.inspect_process(record, proc_root=tmp_path)
    assert result["state"] == "inactive"
    assert result["reason"] == "process_exited_or_zombie"


@pytest.mark.parametrize("relative", ["sys/kernel/random/boot_id", "4101/stat", "4101/status"])
def test_essential_permission_failure_is_unknown(tmp_path, monkeypatch, relative):
    record = process_fixture(tmp_path)
    original = Path.read_text
    def inaccessible(path, *args, **kwargs):
        if path == tmp_path/relative:
            raise PermissionError(errno.EACCES, "Essential field inaccessible", str(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", inaccessible)
    result = inspection.inspect_process(record, proc_root=tmp_path)
    assert result["state"] == "unknown"
    assert result["reason"] == "essential_field_unavailable"


def test_missing_process_is_inactive_but_missing_nspid_is_unknown(tmp_path):
    record = process_fixture(tmp_path)
    (tmp_path/"4101/status").write_text("Name:\tpython\n")
    assert inspection.inspect_process(record, proc_root=tmp_path)["state"] == "unknown"
    (tmp_path/"4101/stat").unlink()
    assert inspection.inspect_process(record, proc_root=tmp_path)["state"] == "inactive"


def test_process_replaced_during_inspection_is_not_live(tmp_path, monkeypatch):
    record = process_fixture(tmp_path)
    original = inspection._process_stat
    calls = []
    def changing(path):
        value = original(path)
        calls.append(path)
        if len(calls) == 2:
            value["start_ticks"] = "987654"
        return value
    monkeypatch.setattr(inspection, "_process_stat", changing)
    result = inspection.inspect_process(record, proc_root=tmp_path)
    assert result["state"] == "inactive"
    assert result["reason"] == "process_changed_during_inspection"


def test_lock_probe_preserves_file_and_owners_lock(tmp_path):
    path = tmp_path/"scheduler.lock"
    path.write_bytes(b"preserve existing lock file")
    before = path.stat()
    with path.open("rb") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert inspection.inspect_lock(path)["state"] == "held"
        with path.open("rb") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert inspection.inspect_lock(path)["state"] == "free"
    assert path.read_bytes() == b"preserve existing lock file"
    assert (path.stat().st_ino, path.stat().st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    missing = tmp_path/"absent.lock"
    assert inspection.inspect_lock(missing)["state"] == "missing"
    assert not missing.exists()


def test_count_correction_requires_live_identity_and_preserves_terminals(tmp_path, monkeypatch):
    proc, run = tmp_path/"proc", tmp_path/"run"
    live = process_fixture(proc)
    dead = {**live, "start_ticks": "999999"}
    unknown = process_fixture(proc, pid=4102)
    (proc/"4102/status").write_text("Name:\tpython\n")
    write_json(run/"coordinator.json", {"state": "running", "process": live})
    (run/"scheduler.lock").touch()
    for job_id, process in (("live", live), ("done", live), ("dead", dead), ("unknown", unknown)):
        job = {"job_id": job_id, "stage": "screen", "trajectory": 0}
        write_json(run/"jobs"/(job_id+".json"), job)
        write_json(run/"ledger"/(job_id+".json"), {"state": "running", "process": process})
        write_json(run/"attempts"/(job_id+".json"), {"job_id": job_id, "state": "running", "process": process})
        if job_id == "done":
            payload = {key: value for key, value in job.items() if key != "job_id"}
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            write_json(run/"terminal"/(job_id+".json"), {"state": "complete", "job_identity": digest})
    before = {str(path): path.read_bytes() for path in run.rglob("*") if path.is_file()}
    deny_namespace_stat(monkeypatch, proc)
    result = inspection.inspect_run(run, proc_root=proc)
    assert result["activity"] == "active"
    assert result["states"] == {"pending": 1, "complete": 1, "running": 1, "unknown": 1}
    assert result["registered_jobs"] == 4
    assert [row["job_id"] for row in result["live_job_overrides"]] == ["live"]
    assert [row["job_id"] for row in result["unknown_jobs"]] == ["unknown"]
    assert result["issues"] == []
    assert {str(path): path.read_bytes() for path in run.rglob("*") if path.is_file()} == before


def test_unidentified_lock_holder_is_unknown_not_inactive(tmp_path):
    proc, run = tmp_path/"proc", tmp_path/"run"
    record = process_fixture(proc)
    (proc/"4101/stat").unlink()
    write_json(run/"coordinator.json", {"state": "running", "process": record})
    path = run/"scheduler.lock"
    path.touch()
    with path.open("rb") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert inspection.inspect_run(run, proc_root=proc)["activity"] == "unknown"
    result = inspection.inspect_run(run, proc_root=proc)
    assert result["activity"] == "inactive"
    assert result["coordinator"] == "inactive"


def test_unreadable_terminal_is_not_overridden_by_a_live_worker(tmp_path):
    proc, run = tmp_path/"proc", tmp_path/"run"
    record = process_fixture(proc)
    write_json(run/"coordinator.json", {"state": "running", "process": record})
    (run/"scheduler.lock").touch()
    write_json(run/"jobs/job.json", {"job_id": "job", "stage": "screen"})
    write_json(run/"ledger/job.json", {"state": "running", "process": record})
    (run/"terminal").mkdir()
    (run/"terminal/job.json").write_text("invalid JSON")
    result = inspection.inspect_run(run, proc_root=proc)
    assert result["states"] == {"unknown": 1}
    assert result["live_job_overrides"] == []
    assert result["issues"][0]["reason"] == "JSONDecodeError"


def test_actual_observer_identity_uses_visible_pid_not_namespace_alias():
    proc = Path("/proc")
    observed = inspection._process_stat(proc/"self/stat")
    nspid = next(line.split()[1:] for line in (proc/"self/status").read_text().splitlines()
                 if line.startswith("NSpid:"))
    record = {"pid": observed["pid"], "start_ticks": observed["start_ticks"],
              "namespace_pid": int(nspid[-1]), "pid_namespace": (proc/"self/ns/pid").stat().st_ino,
              "boot_id": (proc/"sys/kernel/random/boot_id").read_text().strip()}
    result = inspection.inspect_process(record)
    assert result["state"] == "active"
    assert result["legacy_match"] is True
