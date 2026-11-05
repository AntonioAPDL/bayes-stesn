"""Targeted periodic-CNN checks; synthetic test signals are not KS results."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pytest

from bayes_stesn.applications import ks_cnn
from bayes_stesn.benchmark.contracts import CapabilityError, Panel
from bayes_stesn.io import load_state


pytestmark = pytest.mark.neural


def signal_panel(stop=48, *, start=0, values=None, protocol="all_observed"):
    angles = 2*np.pi*np.arange(64)/64
    coordinates = np.column_stack((np.cos(angles), np.sin(angles)))
    times = .25*np.arange(start, stop)
    if values is None:
        values = np.sin(angles[None, :]-.06*np.arange(start, stop)[:, None])
        values += .2*np.cos(2*angles[None, :]+.02*np.arange(start, stop)[:, None])
    return Panel(values, coordinates, times, tuple(f"site_{i:02d}" for i in range(64)),
                 "standardized", "circle_embedding",
                 {"ks_protocol": protocol, "input_mask": "none", "test_fixture": True})


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch is unavailable")
class TestKSCNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = signal_panel()
        cls.config = {"width": 4, "blocks": 1, "epochs": 2, "batch_size": 16,
                      "checkpoint_every_batches": 0}
        cls.fit = ks_cnn.fit_cnn(cls.data, cls.config, seed=71)

    def test_explicit_protocol_and_uniform_geometry_are_required(self):
        with self.assertRaisesRegex(CapabilityError, "ks_protocol"):
            ks_cnn.fit_cnn(signal_panel(protocol="strict"), self.config)
        masked = Panel(self.data.y, self.data.coordinates, self.data.times, self.data.site_ids,
                       metadata={"ks_protocol": "all_observed", "input_site_mask": [False]+[True]*63})
        with self.assertRaisesRegex(CapabilityError, "hidden zero"):
            ks_cnn.fit_cnn(masked, self.config)
        order = np.arange(64); order[[2, 3]] = order[[3, 2]]
        unordered = Panel(self.data.y[:, order], self.data.coordinates[order], self.data.times,
                          tuple(self.data.site_ids[index] for index in order),
                          metadata=dict(self.data.metadata))
        with self.assertRaisesRegex(ValueError, "uniformly ordered"):
            ks_cnn.fit_cnn(unordered, self.config)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            ks_cnn.fit_cnn(self.data, {"epocs": 2})

    def test_autonomous_point_forecast_and_coordinate_capability(self):
        forecast = self.fit.forecast(self.data, 7)
        self.assertEqual(forecast.mean.shape, (7, 64))
        self.assertEqual(forecast.mean.dtype, np.float64)
        self.assertIsNone(forecast.samples)
        self.assertEqual(forecast.metadata["history_rows_used"], 1)
        self.assertEqual(self.fit.diagnostics["usable_training_targets"], 47)
        self.assertEqual(self.fit.diagnostics["inference_qualification"], "not_applicable_point_model")
        self.assertEqual(self.fit.diagnostics["selection_loss"], "fixed_epoch_count")
        selected = [63, 0, 17]
        subset = self.fit.forecast(self.data, 7, self.data.coordinates[selected],
                                   tuple(self.data.site_ids[index] for index in selected))
        np.testing.assert_array_equal(subset.mean, forecast.mean[:, selected])
        with self.assertRaises(CapabilityError):
            self.fit.forecast(self.data, 2, np.array([[.3, .7]]))
        with self.assertRaises(CapabilityError):
            self.fit.forecast(self.data, 2, n_paths=2)
        with self.assertRaises(CapabilityError):
            forecast.interval(.9)

    def test_periodic_seam_and_translation_equivariance(self):
        shift = 17
        shifted = signal_panel(values=np.roll(self.data.y, shift, axis=1))
        forecast = self.fit.forecast(self.data, 5).mean
        moved = self.fit.forecast(shifted, 5).mean
        np.testing.assert_allclose(moved, np.roll(forecast, shift, axis=1), rtol=2e-6, atol=2e-7)

    def test_stateless_history_boundary_and_numeric_precision(self):
        reference = self.fit.forecast(self.data, 5).mean
        changed = self.data.y.copy(); changed[:-1] += 1000.
        history = signal_panel(values=changed)
        np.testing.assert_array_equal(reference, self.fit.forecast(history, 5).mean)
        other = signal_panel(values=-self.data.y)
        self.fit.forecast(other, 5)
        np.testing.assert_array_equal(reference, self.fit.forecast(self.data, 5, seed=900).mean)
        with self.assertRaisesRegex(ValueError, "origin precedes"):
            self.fit.forecast(signal_panel(47), 2)
        check = self.fit.precision_check(self.data, 10)
        self.assertLess(check["max_abs_difference"], 2e-6)
        self.assertLess(check["relative_rms_difference"], 2e-6)

    def test_safe_roundtrip_and_rng_isolation(self):
        import torch
        before = torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"cnn.npz"
            self.fit.save(path)
            restored = ks_cnn.load_cnn(path)
            np.testing.assert_array_equal(self.fit.forecast(self.data, 5).mean,
                                          restored.forecast(self.data, 5).mean)
            self.assertEqual(restored.diagnostics, self.fit.diagnostics)
            with np.load(path, allow_pickle=False) as archive:
                self.assertTrue(all(value.dtype.kind != "O" for value in archive.values()))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_mid_epoch_resume_retains_optimizer_rng_and_selected_epoch(self):
        import torch
        data = signal_panel(33)
        validation = ([data], signal_panel(39).y[33:39][None])
        config = {**self.config, "epochs": 3, "batch_size": 8,
                  "checkpoint_every_batches": 2, "patience": 10}
        before = torch.random.get_rng_state().clone()
        uninterrupted = ks_cnn.fit_cnn(data, config, seed=121, validation=validation)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"checkpoint.npz"
            real_save = ks_cnn.save_state

            def interrupt_after_durable_write(destination, state, *, kind):
                result = real_save(destination, state, kind=kind)
                if state["status"] == "running" and state["epoch"] == 1 and state["cursor"] == 16:
                    raise KeyboardInterrupt("simulated termination after an atomic partial-epoch checkpoint")
                return result

            with patch.object(ks_cnn, "save_state", side_effect=interrupt_after_durable_write):
                with self.assertRaises(KeyboardInterrupt):
                    ks_cnn.fit_cnn(data, config, seed=121, checkpoint_path=path, validation=validation)
            partial = load_state(path, kind="ks_periodic_cnn_checkpoint")
            self.assertEqual(partial["cursor"], 16)
            self.assertTrue(partial["optimizer"]["state"])
            self.assertEqual(partial["permutation_rng"].dtype, np.uint8)
            self.assertEqual(partial["torch_rng"].dtype, np.uint8)
            resumed = ks_cnn.fit_cnn(data, config, seed=121, checkpoint_path=path, validation=validation)
            for name, weights in uninterrupted.to_state()["weights"].items():
                np.testing.assert_array_equal(weights, resumed.to_state()["weights"][name])
            self.assertEqual(uninterrupted.diagnostics["history"], resumed.diagnostics["history"])
            self.assertEqual(uninterrupted.diagnostics["selected_epoch"], resumed.diagnostics["selected_epoch"])
            np.testing.assert_array_equal(uninterrupted.forecast(data, 5).mean, resumed.forecast(data, 5).mean)
            complete = load_state(path, kind="ks_periodic_cnn_checkpoint")
            self.assertEqual(complete["status"], "complete")
            with patch.object(torch.optim, "AdamW", side_effect=AssertionError("must not retrain")):
                cached = ks_cnn.fit_cnn(data, config, seed=121, checkpoint_path=path, validation=validation)
            np.testing.assert_array_equal(cached.forecast(data, 5).mean, resumed.forecast(data, 5).mean)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                ks_cnn.fit_cnn(data, {**config, "epochs": 4}, seed=121, checkpoint_path=path, validation=validation)
            changed_targets = validation[1].copy(); changed_targets[:, 0, 0] += .1
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                ks_cnn.fit_cnn(data, config, seed=121, checkpoint_path=path,
                               validation=(validation[0], changed_targets))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
