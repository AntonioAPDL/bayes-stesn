"""Equation, rollout, fitting-boundary and numeric serialization checks."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from bayes_stesn.benchmark.contracts import Panel, CapabilityError
from bayes_stesn.benchmark.reservoir import (
    fit_reservoir, reservoir_from_state, reservoir_step, _ridge_readout,
    _lagged, _features,
)
from bayes_stesn.benchmark.neural import (
    fit_neural, neural_from_state, coordinate_diffusion_operator,
    diffusion_features, _build_network, DEFAULT_CONFIG,
)
from bayes_stesn.io import save_state, load_state


def toy_panel(rows=90):
    t = np.arange(rows, dtype=float)
    y = np.column_stack([np.sin(.13*t), np.sin(.13*t+.3),
                         .7*np.cos(.13*t), .5*np.sin(.08*t)+.3*np.cos(.13*t)])
    coordinates = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    return Panel(y, coordinates, t, ("a", "b", "c", "d"))


class ReservoirTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = toy_panel(70)
        cls.config = dict(width=9, lags=2, modes=3, members=3, washout=6,
                          ridge=.01, leak=.4, sparsity=.4)
        cls.model = fit_reservoir("eqesn_rollout", cls.data, cls.config, seed=142)

    def test_declared_transition_and_ridge_normal_equations(self):
        h = np.array([.2, -.1]); x = np.array([.5])
        w = np.array([[.3, .4], [-.2, .1]]); u = np.array([[.8], [.2]])
        b = np.array([.1, -.3]); leak = .7
        expected = (1-leak)*h+leak*np.tanh(w@h+u@x+b)
        assert_allclose(reservoir_step(h, x, w, u, b, leak), expected, atol=0)
        assert_allclose(reservoir_step(h, x, w*0, u*0, b*0, leak), (1-leak)*h)
        features = np.c_[np.ones(7), np.arange(7.)-3]
        target = np.c_[np.arange(7.)+9, np.arange(7.)**2]
        penalty = .4
        beta = _ridge_readout(features, target, penalty)
        gradient = features.T@(features@beta-target)/len(features)+penalty*np.r_[np.zeros((1, 2)), beta[1:]]
        assert_allclose(gradient, 0, atol=1e-13)
        assert_allclose(beta[0], target.mean(axis=0))

    def test_complete_finite_ensemble_and_member_is_held_for_full_path(self):
        result = self.model.forecast(8, n_paths=999, seed=4)
        self.assertEqual(result.samples.shape, (3, 8, 4))
        self.assertEqual(result.metadata["actual_member_count"], 3)
        assert_allclose(result.mean, result.samples.mean(axis=0))
        assert_array_equal(result.samples, self.model.forecast(8, n_paths=1, seed=19).samples)
        assert_array_equal(result.samples[:, :3], self.model.forecast(3).samples)
        model = self.model; member = model.members[0]
        scores = model._encode(self.data.y)
        sequence = np.zeros((len(scores)+8, scores.shape[1])); sequence[:len(scores)] = scores
        hidden = np.zeros(model.config["width"])
        expected = []
        for t in range(len(sequence)):
            hidden = reservoir_step(hidden, _lagged(sequence, t, 2), member["recurrent"],
                                    member["input_weights"], member["bias"], .4)
            if t >= len(scores):
                feature = (_features(hidden, True)-member["feature_center"])/member["feature_scale"]
                sequence[t] = feature@member["readout"]
                expected.append(model._decode(sequence[t]))
        assert_allclose(result.samples[0], expected, atol=2e-14)

    def test_constant_field_reduction_and_spectral_scaling(self):
        data = Panel(np.tile([2., -3.], (35, 1)), np.array([[0.], [1.]]))
        fitted = fit_reservoir("eqesn_rollout", data, {**self.config, "modes": 2}, 4)
        assert_allclose(fitted.forecast(30).samples, np.broadcast_to([2., -3.], (3, 30, 2)), atol=1e-12)
        for member in fitted.members:
            self.assertAlmostEqual(np.max(np.abs(np.linalg.eigvals(member["recurrent"]))), .9)

    def test_numeric_checkpoint_replay_and_target_subsets(self):
        result = self.model.forecast(6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"reservoir.npz"
            save_state(path, self.model.to_state(), kind="benchmark_model")
            restored = reservoir_from_state(load_state(path, kind="benchmark_model"))
        assert_array_equal(result.samples, restored.forecast(6).samples)
        subset = self.model.forecast(6, self.data.coordinates[[3, 1]], ("d", "b"))
        assert_array_equal(subset.samples, result.samples[:, :, [3, 1]])
        with self.assertRaises(CapabilityError):
            self.model.forecast(1, np.array([[.5, .5]]))

    def test_frozen_history_uses_only_observed_extension(self):
        full = toy_panel(80)
        extended = self.model.forecast(4, history=full)
        self.assertEqual(extended.metadata["parameter_fit_rows"], 70)
        self.assertEqual(extended.metadata["conditioning_rows"], 80)
        self.assertGreater(np.max(np.abs(extended.mean-self.model.forecast(4).mean)), .01)
        corrupt = full.y.copy(); corrupt[2, 0] += 1
        with self.assertRaises(ValueError):
            self.model.forecast(4, history=Panel(corrupt, full.coordinates, full.times, full.site_ids))
        with self.assertRaises(ValueError):
            fit_reservoir("eqesn_rollout", self.data, {"misspelled_option": 1})


class OptionalImportTests(unittest.TestCase):
    def test_import_does_not_load_torch(self):
        script = "import sys; import bayes_stesn.benchmark.neural; assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Optional PyTorch not installed")
class DiffusionGRUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.data = toy_panel(70)
        cls.config = dict(hidden_size=5, layers=1, diffusion_order=1, graph_k=2,
                          context=5, forecast_training_steps=3, max_epochs=3,
                          patience=2, batch_size=16, stride=4)
        before = torch.get_rng_state().clone()
        cls.model = fit_neural("dcrnn", cls.data, cls.config, seed=51)
        cls.rng_preserved = torch.equal(before, torch.get_rng_state())

    def test_graph_and_diffusion_equations(self):
        p = coordinate_diffusion_operator(self.data.coordinates, 2)
        assert_allclose(p.sum(axis=1), 1, atol=1e-15)
        self.assertTrue(np.all(p.data >= 0))
        assert_array_equal(coordinate_diffusion_operator(self.data.coordinates, 0).toarray(), np.eye(4))
        values = np.arange(8.).reshape(4, 2)
        assert_allclose(diffusion_features(values, p, 2), np.c_[values, p@values, p@(p@values)])
        network = _build_network(self.torch, {**DEFAULT_CONFIG, "hidden_size": 2}, p)
        actual = network.diffusion(self.torch.tensor(values[None], dtype=self.torch.float32)).detach().numpy()[0]
        assert_allclose(actual, diffusion_features(values, p, 2), rtol=1e-6)

    def test_gate_equation_and_zero_diffusion_reduction(self):
        torch = self.torch
        config = {**DEFAULT_CONFIG, "hidden_size": 2, "diffusion_order": 0}
        p = coordinate_diffusion_operator(self.data.coordinates, 2)
        network = _build_network(torch, config, p)
        cell = network.cells[0]
        with torch.no_grad():
            cell.gates.weight.zero_(); cell.gates.bias.zero_()
            cell.candidate.weight.copy_(torch.tensor([[.7, .2, -.1], [-.3, .4, .5]]))
            cell.candidate.bias.copy_(torch.tensor([.1, -.2]))
        y = np.array([[.2, -.3, .4, .5]], dtype=np.float32)
        old = np.arange(8, dtype=np.float32).reshape(1, 4, 2)/20
        _, states = network.step(torch.tensor(y), [torch.tensor(old)])
        inputs = np.concatenate([y[:, :, None], .5*old], axis=-1)
        candidate = np.tanh(inputs@cell.candidate.weight.detach().numpy().T+cell.candidate.bias.detach().numpy())
        assert_allclose(states[0].detach().numpy(), .5*old+.5*candidate, atol=4e-8)
        changed = y.copy(); changed[0, 0] += 8
        _, alternate = network.step(torch.tensor(changed), [torch.tensor(old)])
        assert_array_equal(states[0].detach().numpy()[:, 1:], alternate[0].detach().numpy()[:, 1:])

    def test_deterministic_training_checkpoint_and_point_capability(self):
        self.assertTrue(self.rng_preserved)
        original = self.model.forecast(7)
        self.assertIsNone(original.samples)
        with self.assertRaises(CapabilityError):
            original.interval()
        repeated = fit_neural("dcrnn", self.data, self.config, seed=51)
        assert_array_equal(original.mean, repeated.forecast(7, seed=999, n_paths=1).mean)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"dcrnn.npz"
            save_state(path, self.model.to_state(), kind="benchmark_model")
            restored = neural_from_state(load_state(path, kind="benchmark_model"))
        assert_array_equal(original.mean, restored.forecast(7).mean)
        losses = self.model.diagnostics["selection_losses"]
        minimum_epoch = min(losses, key=lambda record: record["validation_free_running_mse"])["epoch"]
        self.assertEqual(self.model.diagnostics["selected_epoch"], minimum_epoch)
        self.assertEqual(len(self.model.diagnostics["refit_losses"]), minimum_epoch)
        self.assertEqual(self.model.diagnostics["parameter_fit_rows"], 70)
        assert_allclose(self.model.center, self.data.y.mean(axis=0), atol=0)

    def test_tail_cannot_train_gradients_or_scales(self):
        config = {**self.config, "max_epochs": 1, "refit_full": False, "train_fraction": .8}
        first = fit_neural("dcrnn", self.data, config, seed=72)
        cutoff = int(.8*len(self.data.y))
        mutated = self.data.y.copy(); mutated[cutoff:] += 100
        second = fit_neural("dcrnn", Panel(mutated, self.data.coordinates, self.data.times, self.data.site_ids), config, seed=72)
        assert_array_equal(first.center, second.center)
        assert_array_equal(first.scale, second.scale)
        for key, weights in first.to_state()["weights"].items():
            assert_array_equal(weights, second.to_state()["weights"][key])
        self.assertNotEqual(first.diagnostics["selection_losses"][0]["validation_free_running_mse"],
                            second.diagnostics["selection_losses"][0]["validation_free_running_mse"])

    def test_free_rollout_context_target_boundaries(self):
        result = self.model.forecast(9)
        assert_array_equal(result.mean[:3], self.model.forecast(3).mean)
        subset = self.model.forecast(9, self.data.coordinates[[1, 3]], ("b", "d"))
        assert_array_equal(subset.mean, result.mean[:, [1, 3]])
        full = toy_panel(80)
        extended = self.model.forecast(9, history=full)
        self.assertGreater(np.max(np.abs(extended.mean-result.mean)), .001)
        self.assertEqual(extended.metadata["conditioning_rows"], 80)
        self.assertEqual(extended.metadata["parameter_fit_rows"], 70)
        with self.assertRaises(CapabilityError):
            self.model.forecast(1, np.array([[.4, .3]]))
        with self.assertRaises(ValueError):
            fit_neural("dcrnn", self.data, {"learningrate": .01})


if __name__ == "__main__":
    unittest.main()
