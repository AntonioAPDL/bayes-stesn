"""Independent numerical/statistical references for evaluation and simulation."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import norm

from bayes_stesn.calibration import (
    calibrated_intervals, conformal_quantile, fit_standardized_calibration,
    reservoir_weighted_calibration,
)
from bayes_stesn.diagnostics import sbc_histogram, sbc_ranks, summarize_chains
from bayes_stesn.ensemble import (
    fit_crps_stacking, fit_logscore_stacking, mix_trajectories, sample_model_index,
)
from bayes_stesn.scoring import (
    crps_ensemble, dependence_summary, energy_score, geometry_pairs,
    paired_bootstrap, score_predictions, variogram_score,
)
from bayes_stesn.simulation import (
    SCENARIOS, named_rng, nested_input_design, phase_path, regular_grid,
    simulate_dataset, structured_mean,
)


class DiagnosticTests(unittest.TestCase):
    def test_rank_diagnostics_external_reference(self):
        # Reference independently obtained with arviz_stats.array_stats;
        # optional plotting/statistics dependencies are not required here.
        x = np.random.default_rng(42).normal(size=(4, 1000, 3))
        diagnostic = summarize_chains(x)
        assert_allclose(diagnostic.rhat, [1.0006539599387911, 1.0001233757423607, 0.9997195352076546], rtol=1e-12)
        assert_allclose(diagnostic.ess_bulk, [4424.766850250621, 3902.360083048475, 4185.591815485772], rtol=1e-11)
        assert_allclose(diagnostic.ess_tail, [3843.868956368816, 4013.464410466669, 3925.4427289358955], rtol=1e-11)
        self.assertTrue(diagnostic.passed)
        self.assertEqual(diagnostic.to_dict()["chains"], 4)

    def test_stuck_short_nonfinite_and_shifted_chains_fail(self):
        self.assertFalse(summarize_chains(np.zeros((4, 1000))).passed)
        self.assertFalse(summarize_chains(np.random.default_rng(1).normal(size=(4, 4))).passed)
        self.assertFalse(summarize_chains(np.ones((1, 1000))).passed)
        bad = np.random.default_rng(1).normal(size=(4, 1000))
        bad[0] += 3
        self.assertGreater(summarize_chains(bad).rhat[0], 1.2)
        bad[0, 0] = np.nan
        summary = summarize_chains(bad)
        self.assertFalse(summary.passed)
        self.assertIsNone(summary.to_dict()["parameters"]["parameter_0"]["rhat"])

    def test_autocorrelation_reduces_ess_and_folded_detects_scale(self):
        rng = np.random.default_rng(19)
        white = rng.normal(size=(4, 3000))
        correlated = white.copy()
        for t in range(1, correlated.shape[1]):
            correlated[:, t] = 0.9 * correlated[:, t - 1] + np.sqrt(0.19) * white[:, t]
        self.assertLess(summarize_chains(correlated).ess_bulk[0], summarize_chains(white).ess_bulk[0] / 8)
        scale_mismatch = white.copy()
        scale_mismatch[:2] *= 5
        self.assertGreater(summarize_chains(scale_mismatch).rhat[0], 1.1)

    def test_sbc_randomized_ties_and_discrete_bin_expectation(self):
        ranks = sbc_ranks(np.zeros((6000, 1)), np.zeros((6000, 9, 1)), seed=18)
        self.assertTrue(np.all((0 <= ranks) & (ranks <= 9)))
        histogram = sbc_histogram(ranks, draw_count=9, bins=4)
        assert_allclose(histogram["probabilities"], [0.3, 0.2, 0.3, 0.2])
        expected = histogram["expected"]
        self.assertTrue(np.all(np.abs(histogram["counts"][0] - expected) < 6 * np.sqrt(expected)))
        with self.assertRaises(ValueError):
            sbc_histogram(np.array([10]), 9)


class ScoreTests(unittest.TestCase):
    def test_crps_matches_independent_pairwise_definition(self):
        rng = np.random.default_rng(7)
        draws, truth = rng.normal(size=(9, 3, 4)), rng.normal(size=(3, 4))
        reference = np.mean(np.abs(draws - truth), axis=0) - 0.5 * np.mean(np.abs(draws[:, None] - draws[None, :]), axis=(0, 1))
        assert_allclose(crps_ensemble(draws, truth), reference, atol=1e-14)
        assert_allclose(crps_ensemble(draws[::-1], truth), reference, atol=1e-14)
        assert_allclose(crps_ensemble(draws[:1], truth), np.abs(draws[0] - truth))

    def test_energy_matches_dense_joint_definition(self):
        rng = np.random.default_rng(8)
        draws, truth = rng.normal(size=(11, 4, 3)), rng.normal(size=(4, 3))
        reference = np.linalg.norm(draws - truth, axis=-1).mean(axis=0) - 0.5 * np.linalg.norm(draws[:, None] - draws[None, :], axis=-1).mean(axis=(0, 1))
        assert_allclose(energy_score(draws, truth), reference, atol=1e-14)
        # A high-count Monte Carlo estimate agrees within simulation error.
        approx = energy_score(draws, truth, pair_count=100000, seed=90)
        assert_allclose(approx, reference, atol=0.008)

    def test_variogram_detects_dependence_missed_by_marginals(self):
        truth = np.zeros((1, 2))
        concordant = np.array([[[1, 1]], [[-1, -1]]], dtype=float)
        discordant = np.array([[[1, -1]], [[-1, 1]]], dtype=float)
        assert_allclose(crps_ensemble(concordant, truth), crps_ensemble(discordant, truth))
        assert_allclose(variogram_score(concordant, truth), [0])
        assert_allclose(variogram_score(discordant, truth), [2])

    def test_scores_use_declared_targets_and_geometry_pair_domain(self):
        rng = np.random.default_rng(4)
        draws = rng.normal(size=(20, 5, 9))
        truth = rng.normal(size=(5, 9))
        result = score_predictions(draws, truth, coordinates=regular_grid(3), variogram_pairs=6, energy_pairs=None)
        self.assertEqual(result["variogram_pair_count"], 6)
        self.assertEqual(result["rmse_by_lead"].shape, (5,))
        self.assertGreaterEqual(result["intervals"]["0.9"]["coverage"], 0)
        self.assertLessEqual(result["intervals"]["0.9"]["coverage"], 1)
        assert_allclose(result["rmse"] ** 2, np.mean(result["rmse_by_lead"] ** 2))
        pairs, weights = geometry_pairs(regular_grid(3), max_pairs=6)
        assert_array_equal(result["variogram_pair_indices"], pairs)
        assert_allclose(weights.sum(), 1)
        with self.assertRaises(ValueError):
            score_predictions(draws, np.zeros((5, 8)))

    def test_paired_bootstrap_uses_replication_unit(self):
        a = np.array([1, 3, 8, 2, 6], dtype=float)
        result = paired_bootstrap(a, a + 0.3, seed=8)
        assert_allclose(result["ci"], (-0.3, -0.3), atol=1e-14)
        self.assertEqual(result["replications"], 5)
        with self.assertRaises(ValueError):
            paired_bootstrap(a[:, None], a[:, None])
        with self.assertRaises(ValueError):
            paired_bootstrap(a[:1], a[:1])

    def test_dependence_summary_ar_and_regular_grid(self):
        rng = np.random.default_rng(11)
        fields = rng.normal(size=(6000, 9))
        for t in range(1, len(fields)):
            fields[t] = 0.8 * fields[t - 1] + 0.6 * fields[t]
        result = dependence_summary(fields, regular_grid(3), max_lag=5, distance_bins=4, grid_shape=(3, 3))
        assert_allclose(result["temporal_acf"][:3], [0.8, 0.64, 0.512], atol=0.02)
        self.assertEqual(result["spatial_power"].shape, (3, 2))
        self.assertTrue(np.all(result["spatial_power"] >= 0))
        with self.assertRaises(ValueError):
            dependence_summary(fields, regular_grid(3)[::-1], grid_shape=(3, 3))


class EnsembleTests(unittest.TestCase):
    def test_mixture_never_switches_component_inside_path(self):
        left, right = np.ones((10, 7, 3)), -np.ones((12, 7, 3))
        mixed = mix_trajectories([left, right], [0.25, 0.75], paths=10000, seed=12)
        self.assertTrue(np.all(mixed.values == mixed.values[:, :1, :1]))
        self.assertLess(abs(np.mean(mixed.component_index == 0) - 0.25), 0.02)
        for k in (0, 1):
            source = [left, right][k]
            selected = mixed.component_index == k
            assert_array_equal(mixed.values[selected], source[mixed.draw_index[selected]])
        assert_array_equal(mix_trajectories([left]).values, left)

    def test_log_stacking_analytic_optimum_and_numerical_stability(self):
        logp = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))
        result = fit_logscore_stacking(logp)
        assert_allclose(result.weights, [0.5, 0.5], atol=1e-8)
        shifted = fit_logscore_stacking(logp - 10000)
        assert_allclose(shifted.weights, result.weights, atol=1e-8)
        assert_allclose(shifted.objective, result.objective - 10000, atol=1e-10)
        dominant = fit_logscore_stacking(np.log([[0.7, 0.2], [0.6, 0.3]]))
        self.assertGreater(dominant.weights[0], 1 - 1e-7)
        with self.assertRaises(ValueError):
            fit_logscore_stacking(np.array([[-np.inf, -np.inf]]))

    def test_crps_stacking_includes_cross_component_dispersion(self):
        truth = np.array([0.0, 2.0])
        components = [np.zeros((3, 2)), np.full((5, 2), 2.0)]
        result = fit_crps_stacking(components, truth)
        assert_allclose(result.weights, [0.5, 0.5], atol=1e-8)
        assert_allclose(result.objective, 0.5, atol=1e-12)
        # A mixture's CRPS is not the weighted mean of component scores.
        averaged_separate_scores = sum(w * crps_ensemble(c, truth).mean() for w, c in zip(result.weights, components))
        self.assertGreater(averaged_separate_scores, result.objective + 0.4)

    def test_finite_index_matches_analytic_integrated_normal_evidence(self):
        y = 1.5
        sigma = np.array([0.5, 2.0])
        prior = np.array([0.4, 0.6])

        def likelihood(k, theta):
            return -0.5 * ((y - theta[0]) / sigma[k]) ** 2 - np.log(sigma[k]) - 0.5 * np.log(2 * np.pi)

        def hyperprior(k, theta):
            return -0.5 * theta[0] ** 2 - 0.5 * np.log(2 * np.pi)

        samples = sample_model_index(likelihood, hyperprior, np.array([[-1.0], [0.0], [1.0], [2.0]]), 2, model_prior=prior, draws=3000, warmup=750, seed=911)
        evidence = norm.pdf(y, loc=0, scale=np.sqrt(1 + sigma**2))
        expected = prior * evidence
        expected /= expected.sum()
        # Account for correlated visits using a deliberately conservative
        # ESS-based uncertainty rather than pretend every MCMC draw is iid.
        diagnostic = summarize_chains((samples.model_index == 0).astype(float), min_ess=1)
        effective = diagnostic.ess_bulk[0]
        standard_error = np.sqrt(expected[0] * (1 - expected[0]) / effective)
        self.assertLess(abs(samples.model_probabilities[0] - expected[0]), 6 * standard_error)
        self.assertGreater(effective, 200)
        self.assertTrue(np.all((samples.acceptance_rate > 0.05) & (samples.acceptance_rate < 0.8)))

    def test_finite_index_enumerated_reference_and_rejects_dimensions(self):
        # No remaining theta: exact enumeration is a direct probability
        # reference with independent Gibbs index draws.
        likelihoods = np.array([0.2, 0.6])
        sample = sample_model_index(lambda k, t: np.log(likelihoods[k]), lambda k, t: 0.0, np.empty((2, 0)), 2, draws=5000, warmup=0, seed=25)
        assert_allclose(sample.model_probabilities, [0.25, 0.75], atol=0.02)
        with self.assertRaises(ValueError):
            sample_model_index(lambda k, t: 0.0, lambda k, t: 0.0, np.zeros((2, 1)), 2, model_dimensions=(1, 2))


class CalibrationTests(unittest.TestCase):
    def test_exact_order_statistic_does_not_cap_insufficient_rank(self):
        self.assertTrue(np.isinf(conformal_quantile(np.arange(8), alpha=0.1)))
        self.assertEqual(conformal_quantile(np.arange(1, 10), alpha=0.1), 9)
        self.assertEqual(conformal_quantile(np.arange(1, 20), alpha=0.05), 19)
        self.assertEqual(conformal_quantile(np.arange(1, 10), alpha=0.5), 5)

    def test_simultaneous_target_keeps_block_unit(self):
        truth = np.ones((2, 200, 625))
        calibration = fit_standardized_calibration(truth, np.zeros_like(truth), np.ones_like(truth), target="simultaneous")
        self.assertEqual(calibration.blocks, 2)
        self.assertEqual(calibration.scores.shape, (2, 1, 1))
        self.assertTrue(np.isinf(calibration.quantiles).all())
        lo, hi = calibrated_intervals(calibration, truth[0], truth[0])
        self.assertTrue(np.isneginf(lo).all())
        self.assertTrue(np.isposinf(hi).all())

    def test_field_and_path_calibration_use_correct_maxima(self):
        truth = np.arange(1, 61, dtype=float).reshape(10, 2, 3)
        center, scale = np.zeros_like(truth), np.ones_like(truth)
        marginal = fit_standardized_calibration(truth, center, scale, alpha=0.2)
        field = fit_standardized_calibration(truth, center, scale, alpha=0.2, target="horizon")
        path = fit_standardized_calibration(truth, center, scale, alpha=0.2, target="simultaneous")
        assert_array_equal(marginal.quantiles, truth[8])
        assert_array_equal(field.quantiles, truth[8].max(axis=1, keepdims=True))
        assert_array_equal(path.quantiles, [[truth[8].max()]])
        before = center.copy()
        calibrated_intervals(marginal, center[0], scale[0])
        assert_array_equal(center, before)
        with self.assertRaises(ValueError):
            calibrated_intervals(path, np.zeros((3, 3)), np.ones((3, 3)))

    def test_weighted_equal_states_recovers_ordinary_rank(self):
        truth = np.arange(1, 61, dtype=float).reshape(10, 2, 3)
        calibration = fit_standardized_calibration(truth, np.zeros_like(truth), np.ones_like(truth), alpha=0.2)
        states = np.zeros((10, 4))
        weighted = reservoir_weighted_calibration(calibration, states, np.zeros(4))
        assert_array_equal(weighted.quantiles, calibration.quantiles)
        assert_allclose(weighted.effective_blocks, 10)
        self.assertTrue(weighted.weighted)
        remote = reservoir_weighted_calibration(calibration, states, np.full(4, 1000))
        self.assertTrue(np.isinf(remote.quantiles).all())
        self.assertEqual(remote.effective_blocks, 0)


class SimulationTests(unittest.TestCase):
    def test_structured_mean_matches_audited_v3_golden_fixture(self):
        # Produced by the original V3 simulate_data.simulate(side=3,
        # conditioning=1,fitting=1,horizon=1), including its 500-update
        # burn-in. The old implementation is not a runtime dependency.
        expected = np.array([
            [-0.7331171318881299, 0.5317530708048778, 0.02526434951497178, -0.4645409512567279, -0.26770196106071487, 0.3879850774569459, -0.12376424435886588, 0.37739068157324707, -0.05430728681767509],
            [-0.7236813738995231, 0.5467036272760354, 0.01907442857053486, -0.4634890507168602, -0.28876171408698564, 0.37079272306449074, -0.08192194271833937, 0.34805769203657677, -0.03651043717834242],
            [-0.7077727154151479, 0.5563821234806999, 0.01820507446062028, -0.4633398405134919, -0.30372621779310716, 0.3468337560974637, -0.04319957543971298, 0.3143311296630152, -0.013721842813778265],
        ])
        actual = structured_mean(phase_path(3), regular_grid(3))
        assert_allclose(actual, expected, rtol=0, atol=1e-13)

    def test_simultaneous_phase_update_and_break_leave_past_unchanged(self):
        initial = (0.3, 1.1)
        first = phase_path(1, initial, burnin=0)[0]
        expected = [0.3 + 0.115 + 0.035 * np.sin(0.3) + 0.018 * np.sin(1.1), 1.1 + 0.078 + 0.020 * np.sin(2 * 0.3 - 1.1)]
        assert_allclose(first, expected, atol=1e-15)
        baseline = phase_path(30, initial)
        changed = phase_path(30, initial, break_time=15)
        assert_array_equal(changed[:15], baseline[:15])
        assert_allclose(changed[15, 0] - baseline[15, 0], 0.035, atol=1e-14)

    def test_named_streams_are_independent_of_consumption_and_geometry_nested(self):
        first = named_rng(42, "test", 3, "phase").normal(size=10)
        named_rng(42, "test", 3, "reservoir").normal(size=10000)
        assert_array_equal(first, named_rng(42, "test", 3, "phase").normal(size=10))
        self.assertFalse(np.array_equal(first, named_rng(42, "test", 4, "phase").normal(size=10)))
        coordinates = regular_grid(10)
        design = nested_input_design(coordinates, (12, 35), seed=42)
        assert_array_equal(design[12], design[35][:12])
        chosen = coordinates[design[35]]
        self.assertFalse(np.any((chosen[:, 0] >= 0.6) & (chosen[:, 0] <= 0.84) & (chosen[:, 1] >= 0.12) & (chosen[:, 1] <= 0.36)))

    def test_scenarios_pair_mean_but_not_innovation_mechanism(self):
        baseline = simulate_dataset(70, grid_size=4, input_sizes=(4, 8), seed=7)
        for scenario in SCENARIOS:
            kwargs = {"break_time": 35} if scenario == "structural_break" else {}
            generated = simulate_dataset(70, grid_size=4, input_sizes=(4, 8), seed=7, scenario=scenario, **kwargs)
            self.assertTrue(np.isfinite(generated.observation).all())
            self.assertEqual(generated.parameters["scenario"], scenario)
            assert_array_equal(generated.input_indices[8], baseline.input_indices[8])
            if scenario != "structural_break":
                assert_array_equal(generated.generating_mean, baseline.generating_mean)
            else:
                assert_array_equal(generated.observation[:35], baseline.observation[:35])
            if scenario not in ("matched_gaussian", "structural_break"):
                self.assertFalse(np.array_equal(generated.observation, baseline.observation))
        changed = simulate_dataset(70, grid_size=4, input_sizes=(4, 8), seed=7, replication=1)
        self.assertFalse(np.array_equal(changed.generating_mean, baseline.generating_mean))

    def test_horizon_extension_does_not_change_existing_data(self):
        for scenario in ("matched", "latent_ar", "field_t", "local_t"):
            short = simulate_dataset(30, grid_size=4, input_sizes=(4,), scenario=scenario, seed=12)
            long = simulate_dataset(50, grid_size=4, input_sizes=(4,), scenario=scenario, seed=12)
            assert_allclose(short.observation, long.observation[:30], rtol=0, atol=1e-15)

    def test_variance_matched_heavy_tails_and_ar_stationarity(self):
        # Every time is an independent draw in the heavy-tailed scenario;
        # use MC uncertainty of squared errors rather than a hard relative
        # tolerance that ignores Student-t fourth moments.
        data = simulate_dataset(5000, grid_size=3, input_sizes=(4,), scenario="field_t", seed=98)
        nugget = data.observation[:, 0] - data.latent[:, 0]
        squared = nugget**2
        standard_error = squared.std(ddof=1) / np.sqrt(len(squared))
        self.assertLess(abs(squared.mean() - 0.05**2), 6 * standard_error)
        ar = simulate_dataset(5000, grid_size=3, input_sizes=(4,), scenario="latent_ar", seed=98)
        residual = ar.latent - ar.generating_mean
        correlation = np.corrcoef(residual[:-1, 0], residual[1:, 0])[0, 1]
        self.assertLess(abs(correlation - 0.8), 0.04)
        hetero = simulate_dataset(10, grid_size=7, input_sizes=(4,), scenario="static_hetero", seed=98)
        self.assertAlmostEqual(hetero.parameters["nugget_rms_sd"], 0.05, places=14)

    def test_invalid_simulation_domains_and_breaks_rejected(self):
        for kwargs in ({"n_times": 0}, {"scenario": "break"}, {"break_time": 1}, {"scenario": "break", "break_time": 100}, {"input_sizes": (1000,)}):
            with self.assertRaises(ValueError):
                simulate_dataset(grid_size=3, **({"n_times": 20} | kwargs))
        with self.assertRaises(ValueError):
            structured_mean(np.zeros((3, 2)), np.array([[0.5, 0.5], [0.5, 0.5]]))


if __name__ == "__main__":
    unittest.main()
