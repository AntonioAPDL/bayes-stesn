"""Dense-reference and resource-gate checks for the narrow Vecchia workflow."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg

from bayes_stesn import (SpatioTemporalData, SplitPlan, FeatureSpec, SpatialMeanSpec,
                        ModelSpec, FittedModel, SamplerConfig, prepare, fit)
from bayes_stesn.gaussian import GaussianRegression, OperatorGaussianRegression
from bayes_stesn.prediction import simulate_predictive, iter_predictive
from bayes_stesn.priors import PriorSpec
from bayes_stesn.sampling import sample_regression
from bayes_stesn.spatial import CovarianceSpec, SpatialOperator


class SparseGaussianAlgebraTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2701)
        self.coordinates = rng.uniform(size=(11, 2))
        self.Y = rng.normal(size=(11, 23))
        self.Phi = rng.normal(size=(11, 2))
        self.Z = rng.normal(size=(23, 3))
        self.spec = CovarianceSpec(kind="vecchia", vecchia_neighbors=3,
                                   static_variance=True, anisotropy=True)
        self.parameters = self.spec.initial_parameters(self.coordinates)
        self.parameters.update(log_q=np.log(1.7), aniso_a=.3, aniso_b=-.1,
                               variance_beta=np.array([.2, -.1]))

    def test_operator_marginal_and_sufficient_statistics_match_dense(self):
        operator = self.spec.prepare_geometry(self.coordinates).operator(self.parameters)
        dense_R = self.spec.covariance(self.coordinates, self.parameters)
        dense = GaussianRegression(self.Y, self.Phi, self.Z)
        sparse = OperatorGaussianRegression(self.Y, self.Phi, self.Z)
        for precision in (.7, np.array([.4, 1.1, 2.])):
            for sigma2 in (None, .6):
                expected = dense.evaluate(dense_R, column_precision=precision, sigma2=sigma2)
                actual = sparse.evaluate(operator, column_precision=precision, sigma2=sigma2)
                assert_allclose(actual.log_marginal, expected.log_marginal, atol=2e-11)
                assert_allclose([actual.Q, actual.bn, actual.logdet_R],
                                [expected.Q, expected.bn, expected.logdet_R], rtol=2e-12)
                assert_allclose(actual.mean, expected.mean, atol=2e-12)
        self.assertNotIn("YY", sparse.__dict__)
        with self.assertRaisesRegex(ValueError, "iid"):
            sparse.evaluate(operator, temporal_rho=.2)

    def test_recovered_joint_coefficient_and_variance_moments(self):
        operator = self.spec.prepare_geometry(self.coordinates).operator(self.parameters)
        R = self.spec.covariance(self.coordinates, self.parameters)
        engine = OperatorGaussianRegression(self.Y, self.Phi, self.Z)
        evaluation = engine.evaluate(operator, column_precision=.7)
        B = self.Phi.T @ linalg.solve(R, self.Phi, assume_a="pos")
        A = np.kron(self.Z.T @ self.Z, B) + .7*np.eye(6)
        expected_variance = evaluation.bn/(evaluation.an-1)
        covariance = expected_variance*linalg.inv(A)
        rng = np.random.default_rng(981)
        count = 12000
        samples = [evaluation.draw(rng)[:2] for _ in range(count)]
        coefficients = np.array([C.reshape(-1, order="F") for C, _ in samples])
        scales = np.array([s for _, s in samples])
        mean_error = np.abs(coefficients.mean(axis=0)-evaluation.mean.reshape(-1, order="F"))
        self.assertTrue(np.all(mean_error < 6*np.sqrt(np.diag(covariance)/count)))
        covariance_se = np.sqrt((np.outer(np.diag(covariance), np.diag(covariance)) + covariance**2)/count)
        self.assertTrue(np.all(np.abs(np.cov(coefficients, rowvar=False)-covariance) < 6*covariance_se))
        scale_se = expected_variance/np.sqrt((evaluation.an-2)*count)
        self.assertLess(abs(scales.mean()-expected_variance), 6*scale_se)

    def test_vecchia_solve_avoids_extreme_ratio_cancellation(self):
        geometry = self.spec.prepare_geometry(self.coordinates)
        rhs = np.random.default_rng(381).normal(size=(11, 3))
        for q in (1e-8, 1., 1e6, 1e14):
            parameters = dict(self.parameters, log_q=np.log(q))
            operator = geometry.operator(parameters)
            R = np.diag(operator.nugget_diagonal) + q*operator.spatial_covariance_dense()
            expected = linalg.solve(R, rhs, assume_a="pos")
            actual = operator.solve(rhs)
            relative = np.linalg.norm(actual-expected)/np.linalg.norm(expected)
            self.assertLess(relative, 1e-10)

    def test_sparse_numerical_singularity_is_rejectable_without_masking_other_errors(self):
        geometry = self.spec.prepare_geometry(self.coordinates)
        with mock.patch("bayes_stesn.spatial.splinalg.splu",
                        side_effect=RuntimeError("Factor is exactly singular")):
            with self.assertRaises(linalg.LinAlgError):
                geometry.operator(self.parameters)
        with mock.patch("bayes_stesn.spatial.splinalg.splu",
                        side_effect=RuntimeError("unrelated internal bug")):
            with self.assertRaisesRegex(RuntimeError, "unrelated internal bug"):
                geometry.operator(self.parameters)

    def test_prepared_neighbors_reused_and_latent_joint_law_preserved(self):
        from bayes_stesn import spatial
        inputs = self.coordinates
        targets = np.array([[.25, .3], [.7, .4]])
        joint = np.vstack((inputs, targets))
        original = spatial._vecchia_predecessors
        with mock.patch.object(spatial, "_vecchia_predecessors", wraps=original) as search:
            geometry = self.spec.prepare_geometry(joint, reference_coordinates=inputs)
            factor = geometry.factor(self.parameters)
            altered = dict(self.parameters, log_ell=np.log(.3))
            geometry.factor(altered)
            geometry.operator(altered)
            self.assertEqual(search.call_count, 1)
        K, _ = self.spec.joint_matrices(inputs, targets, self.parameters)
        input_K, _ = self.spec.matrices(inputs, self.parameters)
        assert_allclose(K[:len(inputs), :len(inputs)], input_K, atol=1e-12)
        # Independent analytic check of B^-1 D B^-T on this tiny fixture.
        root = linalg.solve_triangular(factor.factor.toarray(),
                    np.diag(np.sqrt(factor.conditional_variances)), lower=True)
        assert_allclose(root @ root.T, K, atol=2e-12)
        count = 20000
        draws = factor.sample(np.random.default_rng(912), size=count)
        standard_error = np.sqrt((np.outer(np.diag(K), np.diag(K)) + K*K)/(count-1))
        self.assertTrue(np.all(np.abs(np.cov(draws)-K) < 6*standard_error))
        self.assertTrue(np.all(np.abs(draws.mean(axis=1)) < 6*np.sqrt(np.diag(K)/count)))

    def test_sparse_checkpoint_continuation_is_identical(self):
        spec = replace(self.spec, max_dense_sites=4)
        config = SamplerConfig(chains=1, warmup=4, draws=5, initialize="prior", seed=19)
        args = (self.Y, self.Phi, self.Z, self.coordinates, ["g"]*3,
                PriorSpec.ridge(), spec)
        checkpoints = {}
        sample_regression(*args, replace(config, draws=2),
                          checkpoint_callback=lambda chain, state: checkpoints.update({chain: state}))
        full = sample_regression(*args, config)
        resumed = sample_regression(*args, config, resume_state=checkpoints)
        assert_array_equal(resumed.chains, full.chains)
        assert_array_equal(resumed.log_density, full.log_density)
        self.assertEqual(resumed.metadata["backend"], "vecchia_operator_spectral")

    def test_unsupported_routes_are_explicit_and_dense_reference_remains_gated(self):
        config = SamplerConfig(chains=1, warmup=1, draws=2, initialize="prior",
                               backend="vecchia_operator")
        args = (self.Y, self.Phi, self.Z, self.coordinates, ["g"]*3)
        with self.assertRaisesRegex(ValueError, "ridge prior"):
            sample_regression(*args, PriorSpec.grouped(), self.spec, config)
        with self.assertRaisesRegex(ValueError, "iid Gaussian"):
            sample_regression(*args, PriorSpec.ridge(), self.spec, replace(config, temporal="full_residual_ar"))
        with self.assertRaisesRegex(ValueError, "dense spatial reference"):
            sample_regression(*args, PriorSpec.grouped(), replace(self.spec, max_dense_sites=4),
                              replace(config, backend="auto"))
        from bayes_stesn.dynamics import sample_augmented
        with self.assertRaisesRegex(ValueError, "iid Gaussian ridge collapsed"):
            sample_augmented(*args, PriorSpec.ridge(), self.spec, config, kind="field_t")


class SparseWorkflowTests(unittest.TestCase):
    @staticmethod
    def specification(max_dense_sites=4):
        return ModelSpec(features=FeatureSpec(modes=2, units=(4,), input_scales=(.4,), squares=2),
                         spatial_mean=SpatialMeanSpec(side=2),
                         covariance=CovarianceSpec(kind="vecchia", vecchia_neighbors=4,
                                                   max_dense_sites=max_dense_sites))

    def test_end_to_end_fit_and_new_site_forecast_avoid_dense_spatial_paths(self):
        rng = np.random.default_rng(91)
        n = 384
        coordinates = rng.uniform(size=(n, 2))
        data = SpatioTemporalData(rng.normal(size=(38, n)), coordinates)
        original_eye = np.eye

        def bounded_identity(size, *args, **kwargs):
            if int(size) >= n:
                raise AssertionError("dense spatial identity allocation")
            return original_eye(size, *args, **kwargs)

        forbidden = AssertionError("dense spatial covariance or response product allocated")
        with mock.patch.object(CovarianceSpec, "covariance", side_effect=forbidden), \
             mock.patch.object(CovarianceSpec, "matrices", side_effect=forbidden), \
             mock.patch.object(CovarianceSpec, "joint_matrices", side_effect=forbidden), \
             mock.patch.object(SpatialOperator, "spatial_covariance_dense", side_effect=forbidden), \
             mock.patch.object(SpatialOperator, "_q_factor", new_callable=mock.PropertyMock,
                               side_effect=AssertionError("unneeded prior-precision LU")), \
             mock.patch.object(GaussianRegression, "YY", new_callable=mock.PropertyMock, side_effect=forbidden), \
             mock.patch("numpy.eye", side_effect=bounded_identity):
            fitted = fit(prepare(data, SplitPlan(12, 38), self.specification()),
                         SamplerConfig(chains=1, warmup=2, draws=3, initialize="prior", seed=7))
            targets = np.vstack((coordinates[:2], [[.3, .2], [.6, .7]]))
            sample = simulate_predictive(fitted, paths=2, horizon=3, target_coordinates=targets,
                                         components=("observation",), seed=31)
        self.assertEqual(sample.observation.shape, (2, 3, 4))
        self.assertTrue(np.isfinite(sample.observation).all())
        self.assertFalse(sample.metadata["spatial_dense_materialization"])
        self.assertEqual(sample.metadata["spatial_rng_scheme"], "vecchia-ancestral-v1")
        self.assertEqual(fitted.posterior.metadata["spatial_backend"], "vecchia_operator")
        self.assertGreater(sample.metadata["solver_reports"][0]["precision_nonzeros"], n)

    def test_saved_sparse_model_preserves_batches_and_duplicate_targets(self):
        rng = np.random.default_rng(92)
        coordinates = rng.uniform(size=(12, 2))
        data = SpatioTemporalData(rng.normal(size=(42, 12)), coordinates)
        fitted = fit(prepare(data, SplitPlan(12, 42), self.specification()),
                     SamplerConfig(chains=1, warmup=3, draws=4, initialize="prior", seed=8))
        targets = np.vstack((coordinates[:1], [[.2, .7], [.2, .7]]))
        options = dict(horizon=4, seed=13, target_coordinates=targets)
        full = simulate_predictive(fitted, paths=5, **options)
        assert_array_equal(full.observation[:, :, 1], full.observation[:, :, 2])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"sparse_model.npz"
            fitted.save(path)
            loaded = FittedModel.load(path)
            batches = list(iter_predictive(loaded, paths=5, batch_size=2, **options))
        for name in ("observation", "latent", "readout_mean", "parameter_draw_ids"):
            assert_array_equal(getattr(full, name), np.concatenate([getattr(b, name) for b in batches]))


if __name__ == "__main__":
    unittest.main()
