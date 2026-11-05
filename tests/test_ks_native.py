"""Small contract checks for the KS adapters, independent of delivered data."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from bayes_stesn.applications.ks_native import (
    NativeFit, fit_native, reservoir_synchronization,
)
from bayes_stesn.benchmark.contracts import Panel
from bayes_stesn.benchmark.gaussian_gp import SeparableGPFit


def panel(rows=48, sites=8):
    angle = np.arange(sites) * 2 * np.pi / sites
    coordinates = .5 * np.column_stack((np.cos(angle), np.sin(angle)))
    times = .25 * np.arange(rows)
    rng = np.random.default_rng(48002)
    values = 2.3 + np.sin(.17 * np.arange(rows)[:, None] + angle)
    values += .2 * np.cos(.11 * np.arange(rows)[:, None] - 2 * angle)
    values += .05 * rng.normal(size=values.shape)
    return Panel(values, coordinates, times, tuple(f"s{i}" for i in range(sites)),
                 metadata={"transform_identity": "synthetic-independent-contract-test"})


def window(data, count):
    return Panel(data.y[-count:], data.coordinates, data.times[-count:], data.site_ids,
                 metadata=dict(data.metadata))


class NativeGPContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = panel()
        cls.config = {"fit_hyperparameters": False, "center": 0.,
            "signal_sd": 1.2, "noise_sd": .08,
            "spatial_length": .3, "temporal_length": 2.5, "temporal_nu": 1.5}
        cls.fitted = fit_native("separable_gp", cls.data, cls.config, seed=17)

    def test_bounded_conditioning_matches_same_fixed_gp_on_window(self):
        history = window(self.data, 12)
        reference = SeparableGPFit(history, self.config, seed=17)
        targets = .5 * np.array([[np.cos(.2), np.sin(.2)], [np.cos(1.3), np.sin(1.3)]])
        actual = self.fitted.forecast(history, 5, targets, ("new0", "new1"), 13, 91)
        expected = reference.forecast(5, targets, ("new0", "new1"), n_paths=13, seed=91)
        np.testing.assert_allclose(actual.samples, expected.samples, rtol=3e-13, atol=3e-13)
        np.testing.assert_array_equal(actual.mean, actual.samples.mean(axis=0))
        self.assertEqual(actual.metadata["history_length"], 12)
        self.assertEqual(actual.metadata["first_target_index"], len(self.data.y))

    def test_path_batches_resume_by_global_path_index(self):
        history = window(self.data, 12)
        whole = self.fitted.forecast(history, 4, n_paths=11, seed=9)
        batches = list(self.fitted.iter_forecasts(history, 4, n_paths=11, seed=9, batch_size=3))
        np.testing.assert_allclose(whole.samples, np.concatenate([part.samples for part in batches]),
                                   rtol=2e-13, atol=2e-13)

    def test_portable_reuse_rejects_changed_training_or_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.npz"
            fitted = fit_native("separable_gp", self.data, self.config, 17, path)
            loaded = fit_native("separable_gp", self.data, self.config, 17, path)
            self.assertEqual(fitted.identity, loaded.identity)
            np.testing.assert_array_equal(fitted.forecast(window(self.data, 10), 2, n_paths=3, seed=4).samples,
                                          loaded.forecast(window(self.data, 10), 2, n_paths=3, seed=4).samples)
            changed = Panel(self.data.y, self.data.coordinates, self.data.times, self.data.site_ids,
                            metadata={"transform_identity": "incompatible"})
            with self.assertRaisesRegex(ValueError, "incompatible"):
                fit_native("separable_gp", changed, self.config, 17, path)

    def test_wrong_sites_and_nonfinite_paths_are_never_silent(self):
        history = window(self.data, 12)
        wrong = Panel(history.y[:, ::-1], history.coordinates[::-1], history.times,
                      history.site_ids[::-1])
        with self.assertRaisesRegex(ValueError, "identities/order"):
            self.fitted.forecast(wrong, 3, n_paths=2)

    def test_same_grid_cannot_cross_trajectory_identity(self):
        data = Panel(self.data.y, self.data.coordinates, self.data.times, self.data.site_ids,
                     metadata={"data_identity_sha256": "trajectory-and-transform-A"})
        fitted = fit_native("separable_gp", data, self.config, 17)
        wrong = Panel(data.y[-12:], data.coordinates, data.times[-12:], data.site_ids,
                      metadata={"data_identity_sha256": "trajectory-and-transform-B"})
        with self.assertRaisesRegex(ValueError, "trajectory/protocol/transform"):
            fitted.forecast(wrong, 3, n_paths=2)


class NativeBayesianContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = panel()
        cls.fitted = fit_native("bayes_stesn", cls.data,
            {"conditioning": 16,
             "features": {"modes": 2, "units": [4], "squares": 2},
             "sampler": {"chains": 2, "warmup": 8, "draws": 12,
                         "init_maxiter": 15, "init_max_evaluations": 100},
             "diagnostics": {"functional_draws": 8}, "continuation_draws": 0}, seed=12)

    def test_preserves_supplied_scale_and_refuses_fake_qualification(self):
        p = self.fitted.core.prepared
        self.assertEqual(p.response_center, 0.)
        self.assertEqual(p.response_scale, 1.)
        np.testing.assert_array_equal(p.standardized_y, self.data.y)
        self.assertEqual(p.spec.prior.kind, "ridge")
        self.assertEqual(p.spec.innovations, "iid")
        self.assertFalse(self.fitted.diagnostics["qualified"])
        self.assertEqual(self.fitted.diagnostics["likelihood_rows"], 32)

    def test_unseen_sites_and_exact_named_path_batching(self):
        history = window(self.data, 20)
        targets = .5 * np.array([[np.cos(.3), np.sin(.3)], [np.cos(2.), np.sin(2.)]])
        whole = self.fitted.forecast(history, 3, targets, ("a", "b"), 7, 82)
        parts = list(self.fitted.iter_forecasts(history, 3, targets, ("a", "b"),
                     n_paths=7, seed=82, batch_size=3))
        np.testing.assert_array_equal(whole.samples, np.concatenate([part.samples for part in parts]))
        self.assertEqual(whole.samples.shape, (7, 3, 2))
        np.testing.assert_array_equal(whole.mean, whole.samples.mean(axis=0))

    def test_empirical_synchronization_reports_actual_windows(self):
        result = reservoir_synchronization(self.fitted, self.data, lengths=(12, 24, 48, 64))
        self.assertTrue(result["rows"][2]["passed"])
        self.assertEqual(result["rows"][2]["state_max_absolute_difference"], 0.)
        self.assertFalse(result["rows"][3]["available"])

    def test_interrupted_chain_and_single_stronger_gate_continuation(self):
        import bayes_stesn.model as core_model

        configuration = {"conditioning": 16,
            "features": {"modes": 2, "units": [4], "squares": 2},
            "sampler": {"chains": 2, "warmup": 8, "draws": 12, "checkpoint_every": 5,
                "init_maxiter": 15, "init_max_evaluations": 100},
            "diagnostics": {"functional_draws": 8}, "continuation_draws": 8}
        clean = fit_native("bayes_stesn", self.data, configuration, 24)
        original_save = core_model.save_state
        interrupted = False

        def save_then_interrupt(*args, **kwargs):
            nonlocal interrupted
            result = original_save(*args, **kwargs)
            if kwargs.get("kind") == "mcmc_checkpoint" and not interrupted:
                interrupted = True
                raise InterruptedError("intentional test interruption after durable RNG checkpoint")
            return result

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.npz"
            with patch.object(core_model, "save_state", side_effect=save_then_interrupt):
                with self.assertRaisesRegex(InterruptedError, "durable RNG"):
                    fit_native("bayes_stesn", self.data, configuration, 24, path)
            resumed = fit_native("bayes_stesn", self.data, configuration, 24, path)
            self.assertEqual(clean.core.posterior_identity, resumed.core.posterior_identity)
            self.assertEqual(resumed.core.posterior.n_draws, 20)
            self.assertEqual(resumed.diagnostics["continuations"], 1)
            self.assertEqual(len(resumed.diagnostics["gates"]), 2)
            self.assertTrue(resumed.diagnostics["gates"][0]["monitoring"]["all_readout_coefficients"])
            np.testing.assert_array_equal(clean.core.posterior.chains, resumed.core.posterior.chains)
            again = fit_native("bayes_stesn", self.data, configuration, 24, path)
            self.assertEqual(again.core.posterior.n_draws, 20)


class NativeGRUContracts(unittest.TestCase):
    def test_native_samples_preserve_existing_ancestral_rule(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch optional dependency is unavailable")
        torch.set_num_threads(1)
        data = panel()
        fitted = fit_native("basis_gru", data,
            {"rank": 3, "hidden_size": 4, "lags": 2, "epochs": 1,
             "patience": 1, "validation_horizon": 4, "batch_size": 32,
             "prediction_batch_size": 4, "covariance_max_iter": 3,
             "covariance_continuation_max_iter": 0}, seed=2)
        targets = .5 * np.array([[np.cos(.3), np.sin(.3)], [np.cos(2.), np.sin(2.)]])
        actual = fitted.forecast(window(data, 10), 4, targets, ("a", "b"), 7, 5)
        expected = fitted.core.forecast(4, targets, ("a", "b"), n_paths=7, seed=5)
        np.testing.assert_allclose(actual.samples, expected.samples, rtol=2e-12, atol=2e-12)
        np.testing.assert_array_equal(actual.mean, actual.samples.mean(axis=0))
        restored = NativeFit.from_state(fitted.to_state())
        np.testing.assert_array_equal(actual.samples,
            restored.forecast(window(data, 10), 4, targets, ("a", "b"), 7, 5).samples)


if __name__ == "__main__":
    unittest.main()
