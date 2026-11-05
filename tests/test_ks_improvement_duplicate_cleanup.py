"""Deletion is allowed only for an exact, locked migration duplicate."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from clean_ks_improvement_duplicates import clean
import clean_ks_improvement_duplicates as cleanup


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source, active = root/"old", root/"active"
        relative = "fits/" + "a"*32 + "/fit.npz"
        payload = b"known migration checkpoint bytes"
        for folder, identity in ((source, "source"), (active, "active")):
            (folder/relative).parent.mkdir(parents=True)
            (folder/relative).write_bytes(payload)
            (folder/relative).with_name("writer.lock").touch()
            (folder/"scheduler.lock").touch()
            (folder/"experiment.json").write_text(json.dumps({"identity": identity}))
        (active/"migration").mkdir()
        (active/"migration/receipt.json").write_text(json.dumps({
            "schema": "ks-improvement-budget-migration/1", "passed": True, "apply": True,
            "scientific_recomputation": False, "source_experiment": "source",
            "destination_experiment": "active", "native_checkpoint_count": 1,
            "files": {relative: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}}}))
        closure = root/"closure.json"
        closure.write_text(json.dumps({"sessions": [{"root": str(folder), "observed_completed": True,
            "exit_code": 0} for folder in (source, active)]}))
        yield root, source, active, relative, closure, payload


class DuplicateCleanupTests(unittest.TestCase):
    def test_only_identical_source_removed(self):
        with fixture() as (root, source, active, relative, closure, payload):
            result = clean(source, active, closure, root/"receipt.json", apply=True)
            self.assertTrue(result["complete"])
            self.assertEqual(result["removed_bytes"], len(payload))
            self.assertFalse((source/relative).exists())
            self.assertEqual((active/relative).read_bytes(), payload)
            self.assertTrue((source/"experiment.json").exists())

    def test_mismatched_replacement_preserves_source(self):
        with fixture() as (root, source, active, relative, closure, payload):
            (active/relative).write_bytes(b"x"*len(payload))
            with self.assertRaises(ValueError):
                clean(source, active, closure, root/"receipt.json", apply=True)
            self.assertEqual((source/relative).read_bytes(), payload)

    def test_active_lock_preserves_source(self):
        with fixture() as (root, source, active, relative, closure, payload):
            with (active/"scheduler.lock").open() as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    clean(source, active, closure, root/"receipt.json", apply=True)
            self.assertEqual((source/relative).read_bytes(), payload)

    def test_replacement_modified_after_hash_preserves_source(self):
        with fixture() as (root, source, active, relative, closure, payload):
            original = cleanup.verified_pair
            def mutate_after_validation(*args):
                result = original(*args)
                (active/relative).write_bytes(b"x"*len(payload))
                return result
            with patch.object(cleanup, "verified_pair", side_effect=mutate_after_validation):
                with self.assertRaises(RuntimeError):
                    clean(source, active, closure, root/"receipt.json", apply=True)
            self.assertEqual((source/relative).read_bytes(), payload)

    def test_dry_run_does_not_open_or_remove_binary_payload(self):
        with fixture() as (root, source, active, relative, closure, payload):
            result = clean(source, active, None, root/"receipt.json")
            self.assertEqual(result["binary_payloads_opened"], 0)
            self.assertEqual(result["removed_bytes"], 0)
            self.assertEqual((source/relative).read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
