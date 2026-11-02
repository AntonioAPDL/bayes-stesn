"""Independent dense-reference checks for spatial and augmented components."""
from __future__ import annotations

import dataclasses
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy import linalg, special, stats
from scipy.spatial import Delaunay

from bayes_stesn.spatial import (
    CovarianceSpec, anisotropy_matrix, fem_matrices, geometry_basis,
    matern_correlation, mesh_interpolation, rectangular_mesh,
)
from bayes_stesn.dynamics import (
    degrees_of_freedom_log_target, draw_local_fields, draw_weighted_coefficients,
    ffbs_ar, ffbs_ar_moments, field_t_log_likelihood, field_t_weight_parameters,
    general_coefficient_statistics, latent_ar_log_density, local_t_weight_parameters,
    sample_augmented, weighted_coefficient_precision, weighted_precision_operator,
)
from bayes_stesn.gaussian import PrecisionSpec


class SpatialTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(324)
        self.coordinates = self.rng.uniform(size=(11, 2))

    def test_matern_parameterization_and_isotropy(self):
        x = np.array([0., .1, .9, 2.])
        assert_allclose(matern_correlation(x, .5), np.exp(-x))
        a = np.sqrt(3)*x
        assert_allclose(matern_correlation(x, 1.5), (1+a)*np.exp(-a))
        matrix = anisotropy_matrix(.7, -.3)
        assert_allclose(matrix, linalg.expm([[.7, -.3], [-.3, -.7]]), atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(matrix), 1., places=13)
        self.assertGreater(np.linalg.eigvalsh(matrix).min(), 0)

    def test_zero_extensions_reproduce_exact_model(self):
        baseline = CovarianceSpec()
        extension = CovarianceSpec(anisotropy=True, nonstationary=True, static_variance=True)
        a = baseline.covariance(self.coordinates, baseline.initial_parameters(self.coordinates))
        b = extension.covariance(self.coordinates, extension.initial_parameters(self.coordinates))
        assert_allclose(a, b, rtol=1e-13, atol=1e-13)

    def test_nonstationary_spd_and_amplitude_convention(self):
        spec = CovarianceSpec(nonstationary=True, geometry_terms=4)
        params = spec.initial_parameters(self.coordinates)
        params["nonstationary_coefficients"] = self.rng.normal(0, .17, (4, 4))
        params["aniso_a"], params["aniso_b"] = .4, -.2
        K, V = spec.matrices(self.coordinates, params)
        basis = geometry_basis(self.coordinates, self.coordinates, 4)
        amplitude = np.exp(2*basis @ params["nonstationary_coefficients"][1])
        assert_allclose(np.diag(K), amplitude, rtol=1e-12)
        assert_allclose(K, K.T, atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(K).min(), 0)
        assert_allclose(V, np.eye(len(K)))

    def test_joint_geometry_preserves_input_marginal(self):
        spec = CovarianceSpec(nonstationary=True, static_variance=True)
        params = spec.initial_parameters(self.coordinates)
        params["variance_beta"] = np.array([.25, -.1])
        params["nonstationary_coefficients"] = self.rng.normal(0, .1, (4, 4))
        R = spec.covariance(self.coordinates, params)
        targets = np.array([[1.2, -.1], [.2, .5]])
        joint = spec.joint_covariance(self.coordinates, targets, params)
        assert_allclose(joint[:11, :11], R, atol=1e-13)
        _, V = spec.matrices(self.coordinates, params)
        self.assertAlmostEqual(np.mean(np.log(np.diag(V))), 0., places=13)
        self.assertGreater(np.linalg.eigvalsh(joint).min(), 0)

    def test_parameter_round_trip_and_immutability(self):
        spec = CovarianceSpec(nonstationary=True, static_variance=True, infer_nu=True)
        x = spec.parameters_to_vector(spec.initial_parameters(self.coordinates))
        x += self.rng.normal(0, .03, len(x))
        params = spec.vector_to_parameters(x, self.coordinates, nu=2.5)
        assert_allclose(spec.parameters_to_vector(params), x)
        self.assertEqual(len(spec.parameter_names()), len(x))
        self.assertTrue(np.isfinite(spec.log_prior(params)))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            spec.nu = 0.5

    def test_pc_prior_transformed_density(self):
        spec = CovarianceSpec(range_prior="pc")
        p = spec.initial_parameters(self.coordinates)
        h0, h1 = -.8, -1.2
        lam = -spec.pc_range_threshold*np.log(spec.pc_range_probability)
        # Independently compute p(rho)*d(rho)/dh; constants cancel.
        def range_density(h):
            rho = 2*np.exp(h)
            return np.log(lam)-2*np.log(rho)-lam/rho+np.log(rho)
        p["log_ell"] = h0
        value0 = spec.log_prior(p)
        p["log_ell"] = h1
        assert_allclose(spec.log_prior(p)-value0, range_density(h1)-range_density(h0))

    def test_separate_scale_prior_has_physical_scale_jacobians(self):
        spec = CovarianceSpec(range_prior="pc", scale_mode="separate")
        x = spec.parameters_to_vector(spec.initial_parameters(self.coordinates))
        p0 = spec.vector_to_parameters(x)
        x[1:] += np.array([.2, -.15])
        p1 = spec.vector_to_parameters(x)
        lam = -np.log(spec.pc_sigma_probability)/spec.pc_sigma_threshold
        def density(p):
            e, w = np.exp(p["log_sigma_e"]), np.exp(p["log_sigma_w"])
            return stats.halfnorm.logpdf(e, scale=spec.sigma_e_scale)+np.log(e)+stats.expon.logpdf(w, scale=1/lam)+np.log(w)
        assert_allclose(spec.log_prior(p1)-spec.log_prior(p0), density(p1)-density(p0))
        self.assertAlmostEqual(np.exp(p1["log_q"]), np.exp(2*(x[2]-x[1])))

    def test_vecchia_all_predecessors_recovers_exact_kernel(self):
        exact = CovarianceSpec()
        spec = CovarianceSpec(kind="vecchia", vecchia_neighbors=20)
        params = spec.initial_parameters(self.coordinates)
        K0, _ = exact.matrices(self.coordinates, params)
        operator = spec.operator(self.coordinates, params)
        assert_allclose(operator.spatial_covariance_dense(), K0, rtol=2e-12, atol=2e-12)
        R = np.eye(len(K0))+K0
        rhs = self.rng.normal(size=(len(K0), 3))
        assert_allclose(operator.solve(rhs), linalg.solve(R, rhs), rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(operator.logdet(), np.linalg.slogdet(R)[1], places=11)

    def test_vecchia_fixed_order_preserves_marginal_and_reports_actual_diagonal(self):
        grid = np.array([(x, y) for y in [0., .5, 1.] for x in [0., .5, 1.]])
        spec = CovarianceSpec(kind="vecchia", vecchia_neighbors=2, static_variance=True)
        params = spec.initial_parameters(grid)
        params["log_ell"] = np.log(.8)
        params["variance_beta"] = np.array([.1, .2])
        K, _ = spec.matrices(grid, params)
        # Target locations create distance ties with the original grid.
        joint, _ = spec.joint_matrices(grid, np.array([[.25, .25], [.75, .75]]), params)
        assert_allclose(joint[:len(grid), :len(grid)], K, atol=1e-12)
        operator = spec.operator(grid, params)
        assert_allclose(operator.marginal_variances(batch_size=3), np.diag(K), atol=1e-13)
        self.assertGreater(np.max(np.abs(np.diag(K)-1)), 1e-4)
        self.assertFalse(operator.metadata["unit_marginal_variance"])
        with self.assertRaisesRegex(ValueError, "unique"):
            spec.joint_matrices(grid, grid[:1], params)

    def test_fem_triangle_matches_element_integrals(self):
        nodes = np.array([[0., 0.], [1., 0.], [0., 1.]])
        mass, stiffness = fem_matrices(nodes, np.array([[0, 1, 2]]))
        assert_allclose(mass, np.full(3, 1/6))
        assert_allclose(stiffness.toarray(), .5*np.array([[2, -1, -1], [-1, 1, 0], [-1, 0, 1]]))
        assert_allclose(stiffness @ np.ones(3), 0, atol=1e-15)

    def test_fem_interpolation_reproduces_affine_functions(self):
        nodes = rectangular_mesh(self.coordinates, 5, .5)
        tri = Delaunay(nodes)
        A = mesh_interpolation(tri, self.coordinates)
        assert_allclose(A @ nodes, self.coordinates, atol=1e-14)
        assert_allclose(A @ np.ones(len(nodes)), 1, atol=1e-14)
        with self.assertRaisesRegex(ValueError, "outside"):
            mesh_interpolation(tri, np.array([[100., 100.]]))

    def test_spde_proper_precision_and_woodbury(self):
        for order in (2, 3):
            spec = CovarianceSpec(kind="spde", spde_order=order, mesh_resolution=6,
                                  static_variance=True, anisotropy=True)
            params = spec.initial_parameters(self.coordinates)
            params["aniso_a"] = .2
            params["variance_beta"] = np.array([.2, -.1])
            operator = spec.operator(self.coordinates, params)
            self.assertGreater(np.linalg.eigvalsh(operator.precision.toarray()).min(), 0)
            self.assertEqual(spec.nu, order-1)
            K = operator.spatial_covariance_dense()
            R = np.diag(operator.nugget_diagonal)+np.exp(params["log_q"])*K
            rhs = self.rng.normal(size=(len(K), 2))
            assert_allclose(operator.solve(rhs), np.linalg.solve(R, rhs), atol=2e-11)
            assert_allclose(operator.logdet(), np.linalg.slogdet(R)[1], atol=2e-10)
            assert_allclose(operator.marginal_variances(), np.diag(K), atol=1e-12)
            targets = np.array([[.4, .6]])
            joint, _ = spec.joint_matrices(self.coordinates, targets, params)
            assert_allclose(joint[:11, :11], K, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "fractional"):
            CovarianceSpec(kind="spde", nu=1.5)

    def test_resource_gate_is_explicit(self):
        spec = CovarianceSpec(kind="vecchia", max_dense_sites=4)
        params = spec.initial_parameters(self.coordinates)
        with self.assertRaisesRegex(ValueError, "operator"):
            spec.matrices(self.coordinates, params)
        self.assertEqual(spec.operator(self.coordinates, params).shape, (11, 11))


class DynamicsAlgebraTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(98)
        self.n, self.T = 3, 4
        self.K = np.array([[1., .4, .1], [.4, 1., .3], [.1, .3, 1.]])
        self.residual = self.rng.normal(size=(self.n, self.T))

    def test_ffbs_smoother_matches_dense_conditional_distribution(self):
        q, sigma2, rho = 1.3, .7, -.4
        S = rho**np.abs(np.subtract.outer(np.arange(self.T), np.arange(self.T)))
        covariance = sigma2*q*np.kron(S, self.K)
        obs = covariance+sigma2*np.eye(self.n*self.T)
        mean_dense = covariance @ np.linalg.solve(obs, self.residual.ravel(order="F"))
        variance_dense = covariance-covariance @ np.linalg.solve(obs, covariance)
        mean, variance = ffbs_ar_moments(self.residual, self.K, q, sigma2, rho)
        assert_allclose(mean.ravel(order="F"), mean_dense, atol=1e-13)
        assert_allclose(variance, variance_dense, atol=1e-13)

    def test_ffbs_random_draws_match_independent_moments(self):
        mean, covariance = ffbs_ar_moments(self.residual, self.K, 1.1, .5, .7)
        draws = np.stack([ffbs_ar(self.residual, self.K, 1.1, .5, .7, self.rng).ravel(order="F")
                          for _ in range(5000)])
        errors = np.abs(draws.mean(axis=0)-mean.ravel(order="F"))
        self.assertTrue(np.all(errors < 5*np.sqrt(np.diag(covariance)/len(draws))))
        assert_allclose(np.cov(draws, rowvar=False), covariance, atol=.027, rtol=.10)

    def test_latent_density_matches_joint_gaussian_including_stationary_start(self):
        rho, sigma2, q = .6, .8, 1.4
        S = rho**np.abs(np.subtract.outer(np.arange(self.T), np.arange(self.T)))
        full = sigma2*q*np.kron(S, self.K)
        dense = stats.multivariate_normal.logpdf(self.residual.ravel(order="F"), cov=full)
        actual = latent_ar_log_density(self.residual, self.K, q, sigma2, rho)
        assert_allclose(actual-self.n*self.T/2*np.log(2*np.pi), dense, atol=1e-12)

    def test_latent_rho_zero_limit(self):
        q, sigma2 = .7, .4
        mean, covariance = ffbs_ar_moments(self.residual, self.K, q, sigma2, 0)
        marginal = np.eye(self.n)+q*self.K
        mean_one = q*self.K @ np.linalg.solve(marginal, self.residual)
        conditional = sigma2*(q*self.K-q*q*self.K @ np.linalg.solve(marginal, self.K))
        assert_allclose(mean, mean_one, atol=1e-13)
        assert_allclose(covariance, np.kron(np.eye(self.T), conditional), atol=1e-13)

    def test_field_t_likelihood_and_gamma_conditionals(self):
        R, sigma2, nu = np.eye(self.n)+.8*self.K, .7, 7.
        reference = sum(stats.multivariate_t.logpdf(y, shape=sigma2*R, df=nu) for y in self.residual.T)
        assert_allclose(field_t_log_likelihood(self.residual, R, sigma2, nu), reference, atol=1e-12)
        shape, rate = field_t_weight_parameters(self.residual, R, sigma2, nu)
        self.assertEqual(shape, (nu+self.n)/2)
        assert_allclose(rate, [(nu+y @ np.linalg.solve(R, y)/sigma2)/2 for y in self.residual.T])
        a, b = local_t_weight_parameters(self.residual, sigma2, nu)
        self.assertEqual(a, (nu+1)/2)
        assert_allclose(b, (nu+self.residual**2/sigma2)/2)

    def test_df_target_uses_prior_on_log_nu_minus_two(self):
        weights = np.array([.8, 1.2, .5, 2.])
        def independent(eta):
            nu = 2+np.exp(eta)
            return stats.gamma.logpdf(weights, a=nu/2, scale=2/nu).sum()+stats.norm.logpdf(eta, loc=np.log(8))
        eta0, eta1 = np.log(4), np.log(11)
        assert_allclose(degrees_of_freedom_log_target(eta1, weights)-degrees_of_freedom_log_target(eta0, weights),
                        independent(eta1)-independent(eta0), atol=1e-12)

    def test_student_t_gaussian_limit(self):
        R, sigma2 = np.eye(self.n)+self.K, .7
        gaussian = sum(stats.multivariate_normal.logpdf(y, cov=sigma2*R) for y in self.residual.T)
        self.assertAlmostEqual(field_t_log_likelihood(self.residual, R, sigma2, 1e6), gaussian, places=4)
        shape, rate = field_t_weight_parameters(self.residual, R, sigma2, 1e6)
        assert_allclose(shape/rate, 1, atol=1e-5)

    def test_general_nonseparable_statistics_match_full_block_design(self):
        Phi = self.rng.normal(size=(self.n, 2))
        Z = self.rng.normal(size=(self.T, 2))
        precision = PrecisionSpec(2, 2, "diagonal", [1., 2., 3., 4.])
        covariances = [self.K+np.diag(self.rng.uniform(.5, 2, self.n)) for _ in range(self.T)]
        A, h = general_coefficient_statistics(self.residual, Phi, Z, iter(covariances), precision)
        X = np.kron(Z, Phi)
        block = linalg.block_diag(*covariances)
        assert_allclose(A, precision.as_matrix()+X.T @ np.linalg.solve(block, X), atol=1e-13)
        assert_allclose(h, X.T @ np.linalg.solve(block, self.residual.ravel(order="F")), atol=1e-13)

    def test_weighted_operator_and_checked_perturbation_match_dense_reference(self):
        Phi = self.rng.normal(size=(self.n, 2))
        Z = self.rng.normal(size=(self.T, 2))
        precision = PrecisionSpec(2, 2, "diagonal", [1., 2., 3., 4.])
        weights = self.rng.uniform(.2, 2, (self.n, self.T))
        X = np.kron(Z, Phi)
        A = precision.as_matrix()+X.T @ (weights.ravel(order="F")[:, None]*X)
        assert_allclose(weighted_coefficient_precision(Phi, Z, weights, precision), A, atol=1e-13)
        op = weighted_precision_operator(Phi, Z, weights, precision)
        v = self.rng.normal(size=4)
        assert_allclose(op @ v, A @ v, atol=1e-13)
        sigma2, seed = .7, 678
        actual, metadata = draw_weighted_coefficients(self.residual, Phi, Z, weights, precision,
                            sigma2, np.random.default_rng(seed), backend="cg", rtol=1e-12)
        noise_rng = np.random.default_rng(seed)
        noise = np.sqrt(precision.diagonal())*noise_rng.normal(size=4)
        noise += X.T @ (np.sqrt(weights)*noise_rng.normal(size=weights.shape)).ravel(order="F")
        h = X.T @ (weights*self.residual).ravel(order="F")
        expected = np.linalg.solve(A, h+np.sqrt(sigma2)*noise)
        assert_allclose(actual.ravel(order="F"), expected, atol=1e-11)
        self.assertTrue(metadata["finite_tolerance_approximation"])
        self.assertLess(metadata["relative_residual"], 1e-11)

    def test_local_field_draws_match_direct_gaussian_conditioning(self):
        residual = self.residual[:, :1]
        weights = np.array([[.5], [1.3], [2.]])
        sigma2, q = .6, .9
        prior = sigma2*q*self.K
        noise = sigma2*np.diag(1/weights[:, 0])
        mean = prior @ np.linalg.solve(prior+noise, residual[:, 0])
        covariance = prior-prior @ np.linalg.solve(prior+noise, prior)
        draws = np.stack([draw_local_fields(residual, self.K, q, sigma2, weights, self.rng)[:, 0]
                          for _ in range(4000)])
        assert_allclose(draws.mean(axis=0), mean, atol=.035)
        assert_allclose(np.cov(draws, rowvar=False), covariance, atol=.022, rtol=.08)

    def test_general_design_rejects_missing_covariance_times(self):
        P = np.ones((self.n, 1))
        Z = np.ones((self.T, 1))
        precision = PrecisionSpec(1, 1, "scalar", 1.)
        with self.assertRaisesRegex(ValueError, "per time"):
            general_coefficient_statistics(self.residual, P, Z, iter([self.K]), precision)


class AugmentedSamplerTests(unittest.TestCase):
    def setUp(self):
        from bayes_stesn.priors import PriorSpec
        from bayes_stesn.sampling import SamplerConfig
        self.rng = np.random.default_rng(19)
        self.coords = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        self.Phi = np.column_stack((np.ones(4), self.coords[:, 0]-.5))
        self.Z = np.column_stack((np.ones(7), np.linspace(-1, 1, 7)))
        self.Y = self.Phi @ np.array([[.2, .5], [-.3, .1]]) @ self.Z.T+self.rng.normal(0, .2, (4, 7))
        self.prior = PriorSpec()
        self.config = SamplerConfig(chains=2, warmup=3, draws=4, seed=37,
                                    backend="auto", initial_jitter=.01)

    def test_all_augmented_routes_run_and_reproduce(self):
        for kind in ("latent_ar", "field_t", "local_t", "heteroskedastic"):
            kwargs = {}
            if kind == "heteroskedastic":
                kwargs["known_variance_covariates"] = self.Z[:, 1:]
            posterior = sample_augmented(self.Y, self.Phi, self.Z, self.coords, ["a", "b"],
                self.prior, CovarianceSpec(), self.config, kind=kind, **kwargs)
            repeat = sample_augmented(self.Y, self.Phi, self.Z, self.coords, ["a", "b"],
                self.prior, CovarianceSpec(), self.config, kind=kind, **kwargs)
            self.assertEqual(len(posterior.states), 2)
            self.assertEqual(len(posterior.states[0]), 4)
            self.assertTrue(np.isfinite(posterior.log_density).all())
            assert_allclose(posterior.log_density, repeat.log_density, rtol=0, atol=0)
            for chain in posterior.states:
                for state in chain:
                    self.assertGreater(state["sigma2"], 0)
                    self.assertTrue(np.isfinite(state["C"]).all())
                    self.assertEqual(state["temporal_parameters"]["kind"], kind)
                    if kind == "latent_ar":
                        self.assertEqual(state["latent_state"].shape, (4,))
                        self.assertLess(abs(state["temporal_parameters"]["rho"]), 1)
                    elif kind in {"field_t", "local_t"}:
                        self.assertGreater(state["temporal_parameters"]["nu"], 2)
            expected_shape = "a0+(2N+k)/2" if kind in {"latent_ar", "local_t"} else "a0+(N+k)/2"
            self.assertEqual(posterior.metadata["scale_shape"], expected_shape)

    def test_local_t_cg_sampler_route(self):
        config = dataclasses.replace(self.config, backend="matrixfree", solver_rtol=1e-11)
        posterior = sample_augmented(self.Y, self.Phi, self.Z, self.coords, ["a", "b"],
            self.prior, CovarianceSpec(), config, kind="local_t")
        self.assertTrue(posterior.states[0][0]["solver_metadata"]["finite_tolerance_approximation"])

    def test_augmented_checkpoint_continuation_matches_uninterrupted_chain(self):
        for kind in ("latent_ar", "field_t", "local_t", "heteroskedastic"):
            kwargs = ({"known_variance_covariates": self.Z[:, 1:]}
                      if kind == "heteroskedastic" else {})
            small = dataclasses.replace(self.config, draws=2, thin=2, checkpoint_every=2)
            target = dataclasses.replace(small, draws=4)
            snapshots = {}
            def save(chain, payload):
                snapshots[chain] = payload
            args = (self.Y, self.Phi, self.Z, self.coords, ["a", "b"], self.prior, CovarianceSpec())
            sample_augmented(*args, small, kind=kind, checkpoint_callback=save, **kwargs)
            resumed = sample_augmented(*args, target, kind=kind, resume_state=snapshots, **kwargs)
            full = sample_augmented(*args, target, kind=kind, **kwargs)
            assert_allclose(resumed.log_density, full.log_density, rtol=0, atol=0)
            for c in range(2):
                for j in range(4):
                    assert_allclose(resumed.states[c][j]["C"], full.states[c][j]["C"], rtol=0, atol=0)
            wrong = self.Y.copy()
            wrong[0, 0] += .01
            with self.assertRaisesRegex(ValueError, "do not match"):
                sample_augmented(wrong, *args[1:], target, kind=kind, resume_state=snapshots, **kwargs)

    def test_unsupported_compositions_fail_before_sampling(self):
        common = (self.Y, self.Phi, self.Z, self.coords, ["a", "b"], self.prior)
        with self.assertRaisesRegex(ValueError, "iid reference nugget"):
            sample_augmented(*common, CovarianceSpec(static_variance=True), self.config, kind="latent_ar")
        with self.assertRaisesRegex(ValueError, "separate scales"):
            sample_augmented(*common, CovarianceSpec(scale_mode="separate", range_prior="pc"), self.config)
        with self.assertRaisesRegex(ValueError, "requires known"):
            sample_augmented(*common, CovarianceSpec(), self.config, kind="heteroskedastic")
        with self.assertRaisesRegex(ValueError, "constant"):
            sample_augmented(*common, CovarianceSpec(), self.config, kind="heteroskedastic",
                             known_variance_covariates=np.ones((7, 1)))
        for backend, kind in (("temporal_whitened_spectral", "latent_ar"),
                              ("latent_ffbs", "field_t"),
                              ("weighted_spectral", "local_t"),
                              ("spectral_gaussian", "local_t")):
            with self.assertRaises(ValueError):
                sample_augmented(*common, CovarianceSpec(),
                                 dataclasses.replace(self.config, backend=backend), kind=kind)

    def test_coarse_spde_latent_state_rank_gate(self):
        axis = np.linspace(0, 1, 4)
        coords = np.array([(x, y) for y in axis for x in axis])
        Phi = np.column_stack((np.ones(16), coords[:, 0]-.5))
        Y = self.rng.normal(size=(16, 7))
        for kind in ("latent_ar", "local_t"):
            with self.assertRaisesRegex(ValueError, "full-row-rank"):
                sample_augmented(Y, Phi, self.Z, coords, ["a", "b"], self.prior,
                    CovarianceSpec(kind="spde", mesh_resolution=3), self.config, kind=kind)


if __name__ == "__main__":
    unittest.main()
