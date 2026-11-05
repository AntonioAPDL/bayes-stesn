"""Small synthetic acceptance checks; never packages the live KS results."""
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import zipfile
import zlib

import pytest


SCRIPT = Path(__file__).resolve().parents[1]/"tools/package_ks_results.py"
SPEC = importlib.util.spec_from_file_location("ks_package_tool", SCRIPT)
package_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package_tool)


@pytest.fixture
def results(tmp_path, monkeypatch):
    root = tmp_path/"synthetic_results"; root.mkdir()
    (root/"scheduler.lock").touch()
    fit_id = "a"*64
    fit = root/"fits"/fit_id; fit.mkdir(parents=True)
    (fit/"writer.lock").touch()
    with zipfile.ZipFile(fit/"fit.npz", "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("synthetic.bin", os.urandom(2*1024**2+333333))
    package_tool.atomic_json(fit/"receipt.json", {"identity": fit_id, "state": "complete",
        "fit_sha256": package_tool.digest_file(fit/"fit.npz"), "fit_bytes": (fit/"fit.npz").stat().st_size})
    (fit/"fit.npz.chains.npz").write_bytes(b"synthetic retained chain")
    (fit/"fit.npz.chains.npz.ks-fit-phase.npz").write_bytes(b"synthetic phase state")
    (fit/".fit.npz.bstesn-123456789abc-1-incomplete.npz").write_bytes(b"redundant temporary")
    config = {"schema": "ks-experiment-1", "name": "synthetic_acceptance_only",
              "budget": {"disk_reserve_bytes": 2*1024**3}}
    source = {"synthetic_source.py": "0"*64}
    identity = package_tool.json_hash({"config": config, "source": source})
    package_tool.atomic_json(root/"experiment.json", {"identity": identity, "config": config,
        "source": source, "base_revision": "0"*40})
    frozen = {"experiment_identity": identity, "selected": {"A": {}, "B": {}}, "source": source}
    frozen["identity"] = package_tool.json_hash(frozen)
    package_tool.atomic_json(root/"freeze.json", frozen)
    package_tool.atomic_json(root/"test_access.json", {"freeze_identity": frozen["identity"],
        "both_track_selection_complete": True})
    package_tool.atomic_json(root/"summary.json", {"freeze_identity": frozen["identity"],
        "rows": [{"track": track, "pending_fits": 0, "pending_origins": 0} for track in ("A", "B")]})
    package_tool.atomic_json(root/"ledger"/"finished.json", {"state": "complete"})
    package_tool.atomic_json(root/"jobs"/"finished.json", {"output": str(root), "cache": "/synthetic/original/cache"})
    (root/"reports").mkdir(); (root/"reports"/"report.md").write_text("Synthetic backup acceptance fixture only.\n")
    (root/"cache").mkdir(); (root/"cache"/"must_not_package.bin").write_bytes(b"outside study results")
    external = tmp_path/"external.bin"; external.write_bytes(b"not included")
    (root/"external_link").symlink_to(external)
    monkeypatch.setattr(package_tool, "active_workers", lambda root: [])
    return root


@pytest.fixture
def backup(results):
    output = results.parent/"package"
    package_tool.create(results, output, part_mib=1, code_revision="0"*40)
    return output


def test_streamed_multipart_roundtrip_and_no_overwrite(results, backup):
    checked = package_tool.verify(backup)
    manifest = json.loads((backup/"manifest.json").read_text())
    assert checked["state"] == "verified" and checked["writes"] == 0
    assert len(manifest["parts"]) >= 3
    assert all(p["bytes"] <= 1024**2 for p in manifest["parts"])
    assert not list(backup.glob("*.tar.gz"))
    names = {r["name"] for r in manifest["members"]}
    assert not any(".bstesn-" in name or name.endswith(".lock") for name in names)
    assert "external_link" not in names and "cache/must_not_package.bin" not in names
    restored = results.parent/"restored"
    receipt = package_tool.restore(backup, restored)
    assert receipt["state"] == "complete_verified"
    for row in manifest["members"]:
        assert (restored/row["name"]).read_bytes() == (results/row["name"]).read_bytes()
    assert json.loads((restored/"jobs"/"finished.json").read_text())["output"] == str(results)
    with pytest.raises(FileExistsError, match="new destination"):
        package_tool.restore(backup, restored)
    with pytest.raises(ValueError, match="new package directory"):
        package_tool.create(results, backup, part_mib=1, code_revision="0"*40)


def test_part_corruption_is_detected_before_restore_writes(backup):
    manifest = json.loads((backup/"manifest.json").read_text())
    part = backup/manifest["parts"][1]["name"]
    with part.open("r+b") as stream:
        stream.seek(1000); value = stream.read(1)
        stream.seek(1000); stream.write(bytes([value[0] ^ 1]))
    destination = backup.parent/"should_not_exist"
    with pytest.raises((ValueError, gzip.BadGzipFile, EOFError, zlib.error)):
        package_tool.restore(backup, destination)
    assert not destination.exists()


def test_unsafe_manifest_path_is_rejected_before_restore_writes(backup):
    path = backup/"manifest.json"; manifest = json.loads(path.read_text())
    manifest["members"][0]["name"] = "../../outside"
    package_tool.atomic_json(path, manifest)
    (backup/"manifest.sha256").write_text(package_tool.digest_file(path)+"  manifest.json\n")
    destination = backup.parent/"unsafe_restore"
    with pytest.raises(ValueError, match="Unsafe"):
        package_tool.restore(backup, destination)
    assert not destination.exists()


def test_scheduler_lock_and_active_worker_block_creation(results, monkeypatch):
    output = results.parent/"blocked_package"
    with (results/"scheduler.lock").open("r+") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            package_tool.create(results, output, code_revision="0"*40)
    assert not output.exists()
    monkeypatch.setattr(package_tool, "active_workers", lambda root: [12345])
    with pytest.raises(RuntimeError, match="worker is still active"):
        package_tool.create(results, output, code_revision="0"*40)
    assert not output.exists()
