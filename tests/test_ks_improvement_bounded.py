"""Operational guards only; no runner, fitted model or actual cache advice."""
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("ks_bounded", Path(__file__).resolve().parents[1]/"tools/run_ks_improvement_bounded.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class BoundedRecoveryGuards(unittest.TestCase):
    def test_only_terminal_memory_traceback_is_eligible(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/"child.log"
            for text, expected in [
                ("Traceback (most recent call last):\nRuntimeError: Memory admission gate: insufficient\n", True),
                ("RuntimeError: Memory admission gate: text only\n", False),
                ("Traceback (most recent call last):\nRuntimeError: Memory admission gate: insufficient\nValueError: different failure\n", False),
                ("Traceback (most recent call last):\nRuntimeError: Disk reserve gate: insufficient\n", False)]:
                p.write_text(text)
                self.assertEqual(M.terminal_memory_gate(p), expected)

    def test_existing_locks_are_required_and_busy_lock_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); fit = root/"fits"/"fit1"; fit.mkdir(parents=True)
            (root/"scheduler.lock").touch()
            with ExitStack() as stack, self.assertRaises(FileNotFoundError):
                M.acquire_locks([root], stack)
            (fit/"writer.lock").touch()
            with (fit/"writer.lock").open("rb") as busy:
                fcntl.flock(busy, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with ExitStack() as stack, self.assertRaises(BlockingIOError):
                    M.acquire_locks([root], stack)

    def test_guarded_inventory_deduplicates_and_mocked_advice_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); fit = root/"fits"/"fit1"; fit.mkdir(parents=True)
            (root/"scheduler.lock").touch(); (fit/"writer.lock").touch()
            p = fit/"fit.npz"; p.write_bytes(b"payload!"*(M.MIB//8))
            (fit/"copy.npz").hardlink_to(p)
            before = hashlib.sha256(p.read_bytes()).hexdigest()
            with patch.object(M.os, "posix_fadvise") as advice:
                dry = M.advise_cache([root], apply=False)
                self.assertEqual(dry["eligible_files"], 1); advice.assert_not_called()
                done = M.advise_cache([root], apply=True)
                self.assertEqual(done["files_advised"], 1); advice.assert_called_once()
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), before)
            self.assertEqual(done["file_content_changes"], 0)
            (fit/"unsafe.npz").symlink_to(p)
            with self.assertRaisesRegex(ValueError, "Symlink"):
                M.advise_cache([root], apply=False)

    def test_initial_completion_receipt_must_cover_every_declared_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); receipt = root/"closed.json"
            receipt.write_text(json.dumps({"sessions": [{"root": str(root),
                "exit_code": 1, "observed_completed": True}]}))
            proof = M.closed_session_proof(receipt, [root])
            self.assertEqual(proof["required_roots"], [str(root)])
            with self.assertRaises(ValueError):
                M.closed_session_proof(receipt, [root/"unobserved"])

    def test_explicit_small_checkpoint_advice_retains_locks_and_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); fit = root/"fits"/"fit1"; fit.mkdir(parents=True)
            (root/"scheduler.lock").touch(); (fit/"writer.lock").touch()
            p = fit/"fit.npz.chains.npz"; p.write_bytes(b"closed!!"*(128*1024//8))
            before = hashlib.sha256(p.read_bytes()).hexdigest()
            with patch.object(M.os, "posix_fadvise") as advice:
                self.assertEqual(M.advise_cache([root], apply=False)["eligible_files"], 0)
                done = M.advise_cache([root], apply=True, minimum_bytes=64*1024)
                self.assertEqual(done["files_advised"], 1); advice.assert_called_once()
                self.assertEqual(done["minimum_eligible_bytes"], 64*1024)
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), before)
            with self.assertRaises(ValueError):
                M.advise_cache([root], apply=False, minimum_bytes=1024)
            with (fit/"writer.lock").open("rb") as busy:
                fcntl.flock(busy, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    M.advise_cache([root], apply=True, minimum_bytes=64*1024)

    def test_closed_fit_receipt_scope_is_explicit_and_does_not_read_payloads(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); fit = root/"fits"/"fit1"; fit.mkdir(parents=True)
            (root/"scheduler.lock").touch(); (fit/"writer.lock").touch()
            p = fit/"fit.json"; p.write_bytes(b"receipt!"*(128*1024//8))
            outside = root/"unrelated.json"; outside.write_bytes(p.read_bytes())
            before = hashlib.sha256(p.read_bytes()).hexdigest()
            with patch.object(M.os, "posix_fadvise") as advice:
                self.assertEqual(M.advise_cache([root], apply=False, minimum_bytes=64*1024)["eligible_files"], 0)
                done = M.advise_cache([root], apply=True, minimum_bytes=64*1024, include_fit_receipts=True)
                self.assertEqual(done["files_advised"], 1); advice.assert_called_once()
                self.assertEqual(done["payload_bytes_read"], 0)
                self.assertEqual(done["files"][0]["path"], "fits/fit1/fit.json")
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), before)
            self.assertEqual(hashlib.sha256(outside.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
