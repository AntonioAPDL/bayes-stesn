"""Tiny checkpoint contracts; no sampler, real data or live fit is opened."""
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from bayes_stesn.io import content_hash, load_state, save_state


SPEC = importlib.util.spec_from_file_location("ks_native_package", Path(__file__).resolve().parents[1]/"tools/package_ks_improvement_fits.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(M.encoded(value))


def fixture(folder):
    root = Path(folder)/"run"; root.mkdir()
    for name in ("fits", "jobs", "terminal"):
        (root/name).mkdir()
    (root/"scheduler.lock").touch()
    base_seed = 17
    config = {"features": {"seed": M.seed_for(base_seed, "reservoir"), "units": [2]},
        "sampler": {"seed": M.seed_for(base_seed, "mcmc"), "draws": 2, "warmup": 1,
                    "thin": 1, "chains": 1, "checkpoint_every": 1},
        "continuation_draws": 2, "diagnostics": {"min_chains": 1}}
    candidate = {"id": "candidate1", "config": config, "effective_hash": M.identity(config)}
    cfg = {"registry": {"tracks": {"A": {"candidates": [candidate]}}},
        "model_seed": base_seed, "promote": {"trajectories": [0]},
        "sensitivity": {"seeds": [19], "trajectories": [0]},
        "budget": {"disk_reserve_bytes": 0}}
    source = {"native": {"source_sha256": "a"*64, "runtime": {"numpy": "fixture"}}}
    exp = {"config": cfg, "source": source, "cache_identity": "cache-fixture"}
    exp["identity"] = M.identity({"config": cfg, "source": source, "cache": exp["cache_identity"]})
    write(root/"experiment.json", exp); write(root/"history_policy.json", {"history": 512})
    frozen = {"experiment_identity": exp["identity"], "confirmation_trajectories": [],
        "selected": {"A": {"main": candidate, "anchor_id": candidate["id"], "sensitivity": None}}}
    frozen["identity"] = M.identity(frozen); write(root/"freeze.json", frozen)
    return root, exp, candidate


def job_for(root, exp, candidate, *, seed=17, trajectory=0):
    job = {"experiment_identity": exp["identity"], "source": exp["source"],
        "candidate": candidate, "track": "A", "seed": seed, "trajectory": trajectory,
        "stage": "confirmation", "partition": "validation", "origins": [6000]}
    job["fit_key"] = M.identity({"track": "A", "effective_config": candidate["config"],
        "trajectory": trajectory, "seed": seed})[:32]
    job["job_id"] = M.identity(job)[:24]
    write(root/"jobs"/(job["job_id"]+".json"), job)
    directory = root/"fits"/job["fit_key"]; directory.mkdir(exist_ok=True)
    (directory/"writer.lock").touch()
    return job, directory


def native(root, exp, candidate, *, seed=17, trajectory=0, payload_bytes=0, qualified=True):
    job, directory = job_for(root, exp, candidate, seed=seed, trajectory=trajectory)
    cfg = M.expected_config(job)
    saved_cfg = copy.deepcopy(cfg); saved_cfg["features"]["units"] = tuple(saved_cfg["features"]["units"])
    state = {"schema": "ks-native-fit-1", "model_id": "bayes_stesn", "config": saved_cfg,
        "seed": seed, "input_identity": "input-fixture", "computation": exp["source"]["native"]}
    state["identity"] = content_hash(state)
    state["diagnostics"] = {"qualified": qualified, "input_identity": "input-fixture"}
    state["core"] = {"prepared": {"identity": "prepared-fixture"},
        "payload": np.random.default_rng(seed).integers(0, 256, payload_bytes, dtype=np.uint8)}
    path = directory/M.NATIVE; save_state(path, state, kind="ks_native_fit")
    receipt = {"fit_key": job["fit_key"], "native_fit_identity": state["identity"],
        "fit_sha256": M.digest(path), "fit_bytes": path.stat().st_size,
        "config": cfg, "seed": seed, "track": "A", "trajectory": trajectory,
        "candidate_id": candidate["id"], "diagnostics": state["diagnostics"]}
    write(directory/"receipt.json", receipt)
    return job, directory, state


def partial(root, exp, candidate, *, trajectory=5):
    job, directory = job_for(root, exp, candidate, trajectory=trajectory)
    cfg = M.expected_config(job)
    restart = {"schema_version": 1, "signature": "b"*64, "chain_index": 0,
        "schedule": "collapsed", "iteration": 0, "current_log_target": 0.,
        "vector": np.zeros(1), "nu": 1.5, "prior_state": {}, "C": None, "sigma2": None,
        "rng_state": np.random.default_rng(17).bit_generator.state, "proposal": {"scale": 1.},
        "initialization": {}, "initial_vector": np.zeros(1), "retained_states": [],
        "retained_log_density": [], "accepted_total": 0, "accepted_sampling": 0,
        "proposals_sampling": 0}
    chain = {"identity": "prepared-partial-fixture", "config": cfg["sampler"],
        "chains": {"0": restart}}
    save_state(directory/M.CHAIN, chain, kind="mcmc_checkpoint")
    transition = dict(cfg["sampler"]); transition.pop("checkpoint_every")
    phase = {"schema": "ks-bayes-fit-phase-1", "initial_draws": 2, "maximum_draws": 4,
        "target_draws": 2, "phase": "initial_sampling", "gates": [], "checkpoint": None}
    phase["identity"] = M.identity({"schema": phase["schema"], "prepared": chain["identity"],
        "initial_sampler": transition, "continuation_draws": 2, "diagnostics": cfg["diagnostics"],
        "diagnostic_seed": M.seed_for(job["seed"], "diagnostics"), "computation": exp["source"]["native"]})
    save_state(directory/M.PHASE, phase, kind="ks_bayes_fit_phase")
    return job, directory, phase


def compact(root, exp):
    folder = root.parent/"compact"; folder.mkdir(exist_ok=True)
    members = []
    with zipfile.ZipFile(folder/"evidence.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        paths = [root/"experiment.json", root/"history_policy.json", root/"freeze.json"]
        paths += sorted((root/"terminal").glob("*.json"))
        paths += [path for path in sorted((root/"attempts").glob("*.json"))
                  if (record := M.read_json(path)).get("state") in {"complete", "interrupted"}
                  and "finished_utc" in record and isinstance(record.get("returncode"), int)]
        for path in paths:
            name = "run/"+path.relative_to(root).as_posix(); archive.write(path, name)
            members.append({"name": name, "bytes": path.stat().st_size, "sha256": M.digest(path)})
    archive = folder/"evidence.zip"
    manifest = {"schema": "ks-improvement-compact-evidence/1", "experiment_identity": exp["identity"],
        "archive": {"name": archive.name, "bytes": archive.stat().st_size, "sha256": M.digest(archive)}, "members": members}
    write(folder/"manifest.json", manifest)
    return folder/"manifest.json"


class NativeCheckpointContracts(unittest.TestCase):
    def test_streamed_checksum_matches_native_c_fortran_scalar_and_empty_arrays(self):
        with tempfile.TemporaryDirectory() as folder:
            state = {"scalar": np.array(3.5), "fortran": np.asfortranarray(np.arange(24.).reshape(2,3,4)),
                "strided": np.arange(18, dtype=">i4").reshape(3,6)[:,::2],
                "empty": np.empty((2,0,3)), "tuple": (np.array([4]), "constant")}
            path = Path(folder)/"fixture.npz"; expected = save_state(path, state, kind="fixture")
            checked = M.validate_npz(path, "fixture")
            self.assertEqual(checked["internal_state_sha256"], expected)
            with self.assertRaisesRegex(ValueError, "schema/kind"):
                M.validate_npz(path, "wrong-kind")

    def test_selected_segments_roundtrip_seed_coverage_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root, exp, candidate = fixture(folder)
            first, a, state = native(root, exp, candidate, payload_bytes=600_000)
            second, b, _ = native(root, exp, candidate, seed=19, payload_bytes=600_000, qualified=False)
            (a/".fit.npz.staging.npz").write_bytes(b"not eligible")
            manifest = compact(root, exp)
            result = M.create(root, Path(folder)/"package", manifest, "selected-only", part_mib=1, apply=True)
            self.assertEqual(result["parts"], 2)
            package = Path(result["package"]); saved = M.read_json(package/"manifest.json")
            self.assertEqual(saved["selected_fit_coverage"]["expected_unique_selected_fits"], 2)
            self.assertEqual(saved["selected_fit_coverage"]["missing_or_unreceipted"], [])
            self.assertEqual(saved["qualified_native_fit_count"], 1)
            self.assertEqual(saved["inference_restart_state_count"], 0)
            self.assertTrue(all(x["bytes"] <= M.MIB for x in saved["parts"]))
            self.assertFalse(any("staging" in x["name"] or x["name"].endswith(".lock") for x in saved["members"]))
            destination = Path(folder)/"restored"; M.restore(package, destination)
            for directory in (a,b):
                self.assertEqual(M.digest(directory/M.NATIVE), M.digest(destination/directory.relative_to(root)/M.NATIVE))
            with self.assertRaises(FileExistsError):
                M.restore(package, destination)
            part = package/saved["parts"][0]["name"]
            with part.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "Part checksum/size"):
                M.verify(package)

    def test_all_mode_retains_partial_chain_and_phase_without_authorizing_new_draws(self):
        with tempfile.TemporaryDirectory() as folder:
            root, exp, candidate = fixture(folder)
            native(root, exp, candidate)
            job, directory, phase = partial(root, exp, candidate)
            manifest = compact(root, exp)
            result = M.create(root, Path(folder)/"all", manifest, "all-fit-checkpoints", part_mib=1, apply=True)
            saved = M.read_json(Path(result["package"])/"manifest.json")
            record = next(x for x in saved["fits"] if x["fit_key"] == job["fit_key"])
            self.assertTrue(record["inference_restart_state_available"])
            self.assertFalse(record["forecast_state_available"])
            self.assertEqual(record["maximum_draws"], 4)
            phase["maximum_draws"] = 6; save_state(directory/M.PHASE, phase, kind="ks_bayes_fit_phase")
            with self.assertRaisesRegex(ValueError, "continuation allowance"):
                M.create(root, Path(folder)/"invalid", manifest, "all-fit-checkpoints", part_mib=1, apply=True)
            (directory/M.PHASE).unlink()
            with self.assertRaisesRegex(ValueError, "without its required phase"):
                M.create(root, Path(folder)/"missingphase", manifest, "all-fit-checkpoints", part_mib=1, apply=True)

    def test_busy_scheduler_or_writer_blocks_binary_reads_and_apply_is_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            root, exp, candidate = fixture(folder); _, directory, _ = native(root, exp, candidate)
            manifest = compact(root, exp)
            with self.assertRaisesRegex(ValueError, "explicit create --apply"):
                M.create(root, Path(folder)/"none", manifest, "selected-only")
            for lock in (root/"scheduler.lock", directory/"writer.lock"):
                with lock.open("rb") as handle:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
                    with patch.object(M, "validate_npz", side_effect=AssertionError("payload read")):
                        with self.assertRaises(BlockingIOError):
                            M.create(root, Path(folder)/"busy", manifest, "selected-only", apply=True)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def test_complete_phase_is_bound_to_exact_chain_bytes_and_original_target(self):
        with tempfile.TemporaryDirectory() as folder:
            root, exp, candidate = fixture(folder)
            job, directory, phase = partial(root, exp, candidate)
            chain = load_state(directory/M.CHAIN, kind="mcmc_checkpoint")
            chain["chains"]["0"].update(iteration=3, retained_states=[{}, {}], retained_log_density=[0., 0.])
            save_state(directory/M.CHAIN, chain, kind="mcmc_checkpoint")
            phase.update(phase="complete", checkpoint={"sha256": M.digest(directory/M.CHAIN)}, gates=[{"passed": False}])
            save_state(directory/M.PHASE, phase, kind="ks_bayes_fit_phase")
            manifest = compact(root, exp)
            result = M.create(root, Path(folder)/"completed", manifest, "all-fit-checkpoints", part_mib=1, apply=True)
            record = M.read_json(Path(result["package"])/"manifest.json")["fits"][0]
            self.assertTrue(record["chain_complete_at_recorded_target"])
            self.assertFalse(record["qualified"])
            phase["checkpoint"]["sha256"] = "0"*64
            save_state(directory/M.PHASE, phase, kind="ks_bayes_fit_phase")
            with self.assertRaisesRegex(ValueError, "exact chain checkpoint"):
                M.create(root, Path(folder)/"changed", manifest, "all-fit-checkpoints", part_mib=1, apply=True)

    def test_plan_never_opens_binary_and_receipt_mismatch_cannot_publish(self):
        with tempfile.TemporaryDirectory() as folder:
            root, exp, candidate = fixture(folder); _, directory, _ = native(root, exp, candidate)
            manifest = compact(root, exp)
            with patch.object(M, "validate_npz", side_effect=AssertionError("payload read")), patch.object(M, "digest", side_effect=AssertionError("payload hash")):
                plan = M.inventory(root, "all-fit-checkpoints")[0]
                self.assertEqual(plan["binary_payloads_opened"], 0)
            receipt = M.read_json(directory/"receipt.json"); receipt["fit_sha256"] = "0"*64
            write(directory/"receipt.json", receipt)
            with self.assertRaisesRegex(ValueError, "current fit receipt"):
                M.create(root, Path(folder)/"corrupt", manifest, "selected-only", apply=True)
            self.assertFalse((Path(folder)/"corrupt").exists())

    def test_stale_compact_frontier_and_unsafe_paths_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root, exp, candidate = fixture(folder); native(root, exp, candidate)
            manifest = compact(root, exp)
            write(root/"terminal"/"new.json", {"state": "complete"})
            with self.assertRaisesRegex(ValueError, "fresh compact snapshot"):
                M.create(root, Path(folder)/"stale", manifest, "selected-only", apply=True)
            for name in ("../escape", "/absolute", "x/../escape", "x/.hidden", "x\\unsafe"):
                with self.assertRaises(ValueError):
                    M.safe_member(name)
            with self.assertRaisesRegex(ValueError, "not eligible"):
                M.reject_private({"drive_file_id": "synthetic"})

    def test_same_terminal_names_do_not_hide_changed_receipt_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root, exp, candidate = fixture(folder)
            path = root/"terminal"/"synthetic.json"
            write(path, {"state": "complete", "elapsed_seconds": 10.})
            manifest = compact(root, exp)
            frozen = M.read_json(root/"freeze.json")
            M.compact_binding(manifest, root, exp, frozen)
            write(path, {"state": "complete", "elapsed_seconds": 20.})
            with self.assertRaisesRegex(ValueError, "stale terminal receipts bytes"):
                M.compact_binding(manifest, root, exp, frozen)

    def test_closed_attempt_accounting_must_match_even_without_new_terminal_job(self):
        for change in ("changed_existing", "new_interrupted_attempt"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                root, exp, candidate = fixture(folder)
                path = root/"attempts"/"synthetic_first.json"
                attempt = {"state": "interrupted", "returncode": -2,
                           "finished_utc": "2026-11-05T10:00:00+00:00", "elapsed_seconds": 10.}
                write(path, attempt)
                manifest = compact(root, exp)
                frozen = M.read_json(root/"freeze.json")
                M.compact_binding(manifest, root, exp, frozen)
                if change == "changed_existing":
                    write(path, {**attempt, "elapsed_seconds": 20.})
                else:
                    write(path.with_name("synthetic_second.json"), {**attempt, "elapsed_seconds": 30.})
                with self.assertRaisesRegex(ValueError, "closed attempts"):
                    M.compact_binding(manifest, root, exp, frozen)


if __name__ == "__main__":
    unittest.main()
