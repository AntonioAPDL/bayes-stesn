"""Compact evidence contracts using tiny synthetic JSON-only jobs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location("ks_compact", Path(__file__).resolve().parents[1]/"tools/package_ks_improvement_checkpoint.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(M.json_bytes(value))


def fixture(folder):
    root = Path(folder)/"run"; root.mkdir()
    candidate = {"id": "candidate1", "config": {"conditioning": 512}, "effective_hash": "fixture"}
    config = {"registry": {"identity": "registry-fixture", "tracks": {"A": {"candidates": [candidate]}}},
              "budget": {"disk_reserve_bytes": 0}}
    source = {"application": {"ks_metrics.py": "0"*64}}
    exp = {"config": config, "source": source, "cache_identity": "cache-fixture"}
    exp["identity"] = M.content_hash({"config": config, "source": source, "cache": exp["cache_identity"]})
    write(root/"experiment.json", exp); write(root/"history_policy.json", {"history": 512})
    diagnosis = Path(folder)/"diagnosis.json"
    write(diagnosis, {"status": "complete", "scope": "synthetic retained diagnostic", "test_label_reads": 0})
    return root, diagnosis, exp, candidate


def completed(root, exp, candidate, trajectory):
    job = {"experiment_identity": exp["identity"], "source": exp["source"], "candidate": candidate,
        "stage": "screen", "track": "A", "trajectory": trajectory, "seed": 17, "history": 512,
        "paths": 256, "forecast_seed": 19, "origins": [6000], "partition": "validation"}
    job["fit_key"] = M.content_hash({"track": "A", "effective_config": candidate["config"],
                                    "trajectory": trajectory, "seed": 17})[:32]
    identity = M.content_hash(job); job["job_id"] = identity[:24]
    write(root/"jobs"/(job["job_id"]+".json"), job)
    terminal = {"job_id": job["job_id"], "job_identity": identity, "state": "complete",
                "completed_origins": 1, "failed_origins": 0}
    write(root/"terminal"/(job["job_id"]+".json"), terminal)
    forecast = {"native_fit": "native-fit-fixture", "first_target": 6000, "state": "complete",
        "qualified": True, "metrics": {"selection_crps": 1.0}, "candidate_id": candidate["id"],
        "forecast_namespace": "main"}
    forecast.update({k: job[k] for k in ("track", "trajectory", "seed", "history", "paths", "forecast_seed", "partition", "fit_key")})
    forecast["identity"] = M.content_hash({"native_fit": forecast["native_fit"], "first": 6000,
        "history": 512, "horizon": 100, "paths": 256, "forecast_seed": 19,
        "partition": "validation", "score_source": exp["source"]["application"]["ks_metrics.py"]})
    path = root/"forecasts"/job["fit_key"]/f"06000_256_{forecast['identity'][:12]}.json"
    write(path, forecast)
    return job, path


class CompactCheckpointContracts(unittest.TestCase):
    def test_completed_frontier_excludes_live_partial_and_bulk_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root, diag, exp, candidate = fixture(folder)
            job, forecast = completed(root, exp, candidate, 0)
            write(root/"terminal"/"pending.json", {"state": "running"})
            write(root/"attempts"/"closed.json", {"state": "complete", "finished_utc": "fixture", "returncode": 0})
            write(root/"attempts"/"live.json", {"state": "running"})
            write(root/"ledger"/"mutable.json", {"state": "running"})
            (root/"fits").mkdir(); (root/"fits"/"never-open.npz").write_bytes(b"NOT A CHECKPOINT")
            result = M.create_snapshot(root, Path(folder)/"exports", diag)
            status = json.loads(Path(result["status"]).read_text())
            self.assertEqual(status["included_terminal_jobs"], 1)
            self.assertEqual(status["included_forecast_receipts"], 1)
            self.assertEqual(status["excluded_record_counts"], {"nonterminal_status_receipt": 1, "live_or_partial_attempt": 1})
            self.assertFalse(status["numerically_resumable_from_this_package_alone"])
            with zipfile.ZipFile(result["archive"]) as z:
                self.assertFalse(any("ledger/" in n or "/fits/" in n or n.endswith(".npz") for n in z.namelist()))
            self.assertTrue(M.verify_snapshot(Path(result["manifest"]))["verified"])

    def test_new_immutable_terminal_after_frontier_is_deferred(self):
        with tempfile.TemporaryDirectory() as folder:
            root, diag, exp, candidate = fixture(folder)
            completed(root, exp, candidate, 0)
            original = M.stable_json; added = False
            def concurrent(path):
                nonlocal added
                if Path(path).name == "experiment.json" and not added:
                    added = True; completed(root, exp, candidate, 5)
                return original(path)
            with patch.object(M, "stable_json", side_effect=concurrent):
                result = M.create_snapshot(root, Path(folder)/"exports", diag)
            status = json.loads(Path(result["status"]).read_text())
            self.assertEqual(status["included_terminal_jobs"], 1)
            self.assertEqual(status["new_terminal_files_after_frontier"], 1)

    def test_corrupt_or_partial_completed_job_cannot_publish(self):
        with tempfile.TemporaryDirectory() as folder:
            root, diag, exp, candidate = fixture(folder)
            job, forecast = completed(root, exp, candidate, 0)
            forecast.unlink()
            exports = Path(folder)/"exports"
            with self.assertRaisesRegex(ValueError, "missing its declared"):
                M.create_snapshot(root, exports, diag)
            self.assertFalse(any(not p.name.startswith(".") for p in exports.iterdir()))

    def test_sensitive_fields_and_unsafe_member_paths_are_refused(self):
        with self.assertRaisesRegex(ValueError, "Sensitive"):
            M.scan_private({"drive_file_id": "synthetic-private-identifier"})
        with self.assertRaisesRegex(ValueError, "Sensitive"):
            M.scan_private({"url": "https://drive.google.com/file/synthetic"})
        for name in ("../escape.json", "/absolute.json", "x/../escape.json", "x/.hidden.json"):
            with self.assertRaises(ValueError):
                M.member_name(name)


if __name__ == "__main__":
    unittest.main()
