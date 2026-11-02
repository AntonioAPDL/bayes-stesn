"""Memory, failure recovery and path-law checks for the P0 engineering work."""
import hashlib
import os
from pathlib import Path
import platform
import selectors
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
from unittest import mock

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from bayes_stesn import (SpatioTemporalData, SplitPlan, FeatureSpec, SpatialMeanSpec,
                        ModelSpec, SamplerConfig, prepare, fit, FittedEnsemble)
from bayes_stesn.prediction import (PredictiveSample, simulate_predictive,
                                   iter_predictive, predictive_output_bytes)
from bayes_stesn.scoring import (_pairs_from_ranks, _unordered_pairs,
                                geometry_pairs, variogram_score, score_predictions)
from bayes_stesn.io import atomic_npz, cleanup_stale_npz, content_hash


class PredictiveStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(421)
        cls.coords = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        y = rng.normal(size=(50, 4))
        spec = ModelSpec(features=FeatureSpec(modes=2, units=(4,), input_scales=(.4,), squares=2),
                         spatial_mean=SpatialMeanSpec(side=2))
        cls.fitted = fit(prepare(SpatioTemporalData(y, cls.coords), SplitPlan(12, 50), spec),
                         SamplerConfig(chains=2, warmup=6, draws=8, seed=5))

    def test_public_constructor_defensively_owns_inputs(self):
        values = np.ones((2, 3, 4))
        ids = np.zeros((2, 2), int)
        sample = PredictiveSample(values, None, None, None, self.coords,
                                  tuple(map(str, range(4))), ids, {})
        values[:] = 9
        ids[:] = 5
        assert_array_equal(sample.observation, 1)
        assert_array_equal(sample.parameter_draw_ids, 0)
        self.assertFalse(sample.observation.flags.writeable)
        with self.assertRaisesRegex(ValueError, "not stored"):
            sample.interval(target="latent")

    def test_internal_transfer_does_not_copy_full_outputs(self):
        from bayes_stesn import prediction
        original = prediction.readonly

        def reject_full_copy(value, *args, **kwargs):
            self.assertNotEqual(np.asarray(value).ndim, 3, "an internal predictive output was copied")
            return original(value, *args, **kwargs)

        with mock.patch.object(prediction, "readonly", side_effect=reject_full_copy):
            sample = simulate_predictive(self.fitted, horizon=3, paths=2)
        self.assertFalse(sample.observation.flags.owndata)
        self.assertTrue(sample.observation.base.flags.owndata)
        self.assertFalse(sample.observation.base.flags.writeable)
        with self.assertRaises(ValueError):
            sample.observation.setflags(write=True)

    def test_selective_storage_preserves_observation_path_law(self):
        full = simulate_predictive(self.fitted, horizon=4, paths=5, seed=17)
        selected = simulate_predictive(self.fitted, horizon=4, paths=5, seed=17,
                                       components=("observation",))
        assert_array_equal(full.observation, selected.observation)
        assert_array_equal(full.parameter_draw_ids, selected.parameter_draw_ids)
        for name in ("latent", "readout_mean", "conditional_mean"):
            self.assertIsNone(getattr(selected, name))
        self.assertEqual(selected.metadata["output_storage_bytes"], selected.observation.nbytes + selected.parameter_draw_ids.nbytes)

    def test_global_path_batches_and_offsets_are_identical(self):
        full = simulate_predictive(self.fitted, horizon=4, paths=8, seed=28)
        batches = list(iter_predictive(self.fitted, horizon=4, paths=8, batch_size=3, seed=28))
        self.assertEqual([b.metadata["path_start"] for b in batches], [0, 3, 6])
        for name in ("observation", "latent", "readout_mean", "conditional_mean", "parameter_draw_ids"):
            assert_array_equal(getattr(full, name), np.concatenate([getattr(b, name) for b in batches]))
        offset = simulate_predictive(self.fitted, horizon=4, paths=3, path_start=3, seed=28)
        assert_array_equal(full.observation[3:6], offset.observation)

    def test_memory_preflight_precedes_posterior_sampling(self):
        with mock.patch.object(type(self.fitted.posterior), "draw_parameters", side_effect=AssertionError("sampling began")):
            with self.assertRaisesRegex(MemoryError, "iter_predictive"):
                simulate_predictive(self.fitted, horizon=10, paths=10, max_output_bytes=1)
        self.assertEqual(predictive_output_bytes(512, 100, 2500), 4096000000 + 16 * 512)
        with self.assertRaises(ValueError):
            predictive_output_bytes(1, 1, 1, components=("latent",))

    def test_selective_and_legacy_fortran_archives_roundtrip(self):
        selected = simulate_predictive(self.fitted, horizon=4, paths=3, components=("observation",))
        full = simulate_predictive(self.fitted, horizon=4, paths=3)
        legacy = PredictiveSample(np.asfortranarray(full.observation), full.readout_mean,
                                  full.conditional_mean, full.latent, full.target_coordinates,
                                  full.target_site_ids, full.parameter_draw_ids, full.metadata)
        with tempfile.TemporaryDirectory() as temp:
            for j, sample in enumerate((selected, legacy)):
                path = Path(temp) / f"sample{j}.npz"
                sample.save(path)
                restored = PredictiveSample.load(path)
                assert_array_equal(sample.observation, restored.observation)
                self.assertEqual(sample.latent is None, restored.latent is None)

    def test_ensemble_global_path_contract_and_selective_storage(self):
        ensemble = FittedEnsemble((self.fitted, self.fitted), np.array([.3, .7]))
        full = simulate_predictive(ensemble, horizon=4, paths=9, seed=37, components=("observation",))
        batches = list(iter_predictive(ensemble, horizon=4, paths=9, batch_size=4, seed=37,
                                       components=("observation",)))
        assert_array_equal(full.observation, np.concatenate([b.observation for b in batches]))
        assert_array_equal(full.parameter_draw_ids, np.concatenate([b.parameter_draw_ids for b in batches]))
        self.assertEqual(full.metadata["mixture_component_id_per_path"],
                         sum([b.metadata["mixture_component_id_per_path"] for b in batches], []))
        self.assertEqual(full.metadata["rng_scheme"], "ensemble-global-path-v2")
        self.assertIsNone(full.latent)


class BoundedPairTests(unittest.TestCase):
    def test_integer_rank_inverse_matches_dense_reference_and_large_limits(self):
        for n in (2, 3, 9, 37):
            assert_array_equal(_pairs_from_ranks(n, range(n * (n - 1) // 2)),
                               np.column_stack(np.triu_indices(n, k=1)))
        n = 2 ** 32
        total = n * (n - 1) // 2
        ranks = [0, 1, total // 2, total - 2, total - 1]
        pairs = _pairs_from_ranks(n, ranks)
        for (i, j), rank in zip(pairs, ranks):
            i, j = int(i), int(j)
            self.assertTrue(0 <= i < j < n)
            self.assertEqual(i * (2 * n - i - 1) // 2 + j - i - 1, rank)

    def test_sampling_memory_does_not_depend_on_all_pair_count(self):
        tracemalloc.start()
        try:
            with mock.patch("numpy.triu_indices", side_effect=AssertionError("all-pairs allocation")):
                pairs = _unordered_pairs(10_000_000, 128, 19)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 1024 ** 2)
        self.assertEqual(len(np.unique(pairs, axis=0)), 128)
        assert_array_equal(pairs, _unordered_pairs(10_000_000, 128, 19))

    def test_small_uniform_subset_distribution(self):
        # Six possible sites pairs, fifteen two-pair subsets; this fixed
        # simulation catches gross selection bias without depending on any
        # exact NumPy RNG realization from the previous algorithm.
        counts = {}
        for seed in range(3000):
            key = tuple(map(tuple, _unordered_pairs(4, 2, seed)))
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(len(counts), 15)
        self.assertTrue(all(abs(count - 200) < 6 * np.sqrt(200) for count in counts.values()))

    def test_geometry_and_score_without_geometry_use_bounded_pairs(self):
        coords = np.column_stack((np.arange(1000), np.zeros(1000)))
        draws = np.random.default_rng(1).normal(size=(3, 2, 1000))
        with mock.patch("numpy.triu_indices", side_effect=AssertionError("all-pairs allocation")):
            pairs, weights = geometry_pairs(coords, max_pairs=7, seed=8)
            score = score_predictions(draws, np.zeros((2, 1000)), variogram_pairs=7, seed=8)
        self.assertEqual(len(pairs), 7)
        assert_allclose(weights.sum(), 1)
        assert_array_equal(score["variogram_pair_indices"], pairs)
        self.assertEqual(score["variogram_pair_rng_scheme"], "floyd-upper-triangle-v2")

    def test_implicit_all_pair_variogram_is_exact_without_full_indices(self):
        rng = np.random.default_rng(2)
        x, y = rng.normal(size=(5, 3, 30)), rng.normal(size=(3, 30))
        pairs = np.column_stack(np.triu_indices(30, k=1))
        expected = variogram_score(x, y, pairs=pairs)
        with mock.patch("numpy.triu_indices", side_effect=AssertionError("all-pairs allocation")):
            actual = variogram_score(x, y)
        assert_allclose(actual, expected, rtol=1e-14)


class AtomicArchiveTests(unittest.TestCase):
    def test_handled_compression_fsync_and_replace_failures_preserve_canonical(self):
        def partial_write(stream, **arrays):
            stream.write(b"partial bytes")
            raise OSError("injected compression failure")

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.npz"
            atomic_npz(path, value=np.arange(4))
            previous = path.read_bytes()
            for name, side_effect in (("numpy.savez_compressed", partial_write),
                                      ("bayes_stesn.io.os.fsync", OSError("injected fsync failure")),
                                      ("bayes_stesn.io.os.replace", OSError("injected replace failure"))):
                with self.subTest(name=name), mock.patch(name, side_effect=side_effect):
                    with self.assertRaises(OSError):
                        atomic_npz(path, value=np.arange(8))
                self.assertEqual(path.read_bytes(), previous)
                self.assertEqual(list(Path(temp).iterdir()), [path])

    def test_hard_kill_recovery_only_removes_dead_local_owned_staging(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.npz"
            atomic_npz(path, value=np.arange(3))
            previous = path.read_bytes()
            script = """
import sys,time,numpy as np
from bayes_stesn.io import atomic_npz
def interrupted(stream,**arrays):
    stream.write(b'partial');stream.flush()
    print('ready',flush=True)
    time.sleep(60)
np.savez_compressed=interrupted
atomic_npz(sys.argv[1],value=np.arange(3))
"""
            proc = subprocess.Popen([sys.executable, "-c", script, str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(proc.stdout, selectors.EVENT_READ)
                    self.assertTrue(selector.select(timeout=10), "child did not create staging file")
                    self.assertEqual(proc.stdout.readline().strip(), b"ready")
                proc.kill()
                proc.wait(timeout=5)
            finally:
                if proc.poll() is None:
                    proc.kill(); proc.wait(timeout=5)
                proc.stdout.close(); proc.stderr.close()
            host = hashlib.sha256(platform.node().encode()).hexdigest()[:12]
            active = Path(temp) / f".{path.name}.bstesn-{host}-{os.getpid()}-active.npz"
            active.write_bytes(b"active")
            unrelated = Path(temp) / "unrelated.npz"
            unrelated.write_bytes(b"keep")
            eligible = cleanup_stale_npz(path, older_than_seconds=0)
            self.assertEqual(len(eligible), 1)
            self.assertTrue(eligible[0].exists())  # default is dry run
            self.assertEqual(cleanup_stale_npz(path, older_than_seconds=0, dry_run=False), eligible)
            self.assertFalse(eligible[0].exists())
            self.assertTrue(active.exists() and unrelated.exists())
            self.assertEqual(path.read_bytes(), previous)

    def test_hash_handles_empty_contiguous_and_noncontiguous_arrays(self):
        array = np.arange(20, dtype=float).reshape(4, 5)
        self.assertEqual(content_hash({"x": array}), content_hash({"x": array.copy()}))
        self.assertEqual(content_hash({"x": array[:, ::2]}), content_hash({"x": array[:, ::2].copy()}))
        self.assertEqual(content_hash({"x": np.empty((0, 2))}), content_hash({"x": np.empty((0, 2))}))


if __name__ == "__main__":
    unittest.main()
