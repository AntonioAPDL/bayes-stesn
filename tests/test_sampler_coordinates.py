"""Independent density/reference and restart checks for proposal coordinates."""
from dataclasses import replace
import copy
import itertools
from unittest import mock

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg, special

from bayes_stesn.diagnostics import summarize_chains
from bayes_stesn.dynamics import sample_augmented
from bayes_stesn.priors import PriorSpec
from bayes_stesn.sampling import SamplerConfig, _SamplingCoordinates, sample_regression
from bayes_stesn.spatial import CovarianceSpec


def _problem():
    return (np.array([[.25, -.15, .5, .1], [-.2, .35, .2, -.1]]),
            np.array([[1.], [.7]]), np.array([[1.], [-.5], [.3], [.8]]),
            np.array([[0., 0.], [.35, .2]]), ("intercept",),
            PriorSpec.ridge(mean_log_precision=.1, sd_log_precision=.55),
            CovarianceSpec(log_q_mean=.2, log_q_sd=.5, log_ell_mean=-.7, log_ell_sd=.3))


def _dense_reference(value, args):
    """Observation-space integration, including independent original priors."""
    Y, Phi, Z, coordinates, _, prior, covariance = args
    a, h, xi = value
    distance = np.linalg.norm(coordinates[:, None]-coordinates[None, :], axis=2)
    scaled = np.sqrt(3.)*distance/np.exp(h)
    K = (1+scaled)*np.exp(-scaled)
    R = np.eye(len(coordinates))+np.exp(xi)*K
    X = np.kron(Z, Phi)
    y = Y.reshape(-1, order="F")
    Omega = np.kron(np.eye(Y.shape[1]), R)+np.exp(-a)*X@X.T
    factor = linalg.cho_factor(Omega, lower=True)
    Q = float(y@linalg.cho_solve(factor, y))
    an, bn = 2+len(y)/2, .05+Q/2
    loglik = (-.5*len(y)*np.log(2*np.pi)-np.log(np.diag(factor[0])).sum()
              +2*np.log(.05)+special.gammaln(an)-special.gammaln(2)-an*np.log(bn))
    hp = (-.5*((a-prior.mean_log_precision)/prior.sd_log_precision)**2
          -np.log(prior.sd_log_precision)-.5*np.log(2*np.pi)
          -.5*((h-covariance.log_ell_mean)/covariance.log_ell_sd)**2
          -.5*((xi-covariance.log_q_mean)/covariance.log_q_sd)**2)
    mean_coefficient = float((np.exp(-a)*X.T@linalg.cho_solve(factor, y))[0])
    return float(loglik+hp), float(loglik), mean_coefficient, float(bn/(an-1))


def test_stable_unit_jacobian_transform_roundtrips_and_names():
    coordinates = _SamplingCoordinates("ridge_total_scale", ("prior.log_precision", "log_ell", "log_q"))
    for a, xi in itertools.product((-5., .3, 8.), (-600., -50., 0., 50., 600.)):
        original = np.array([a, -.7, xi])
        actual = coordinates.to_scientific(coordinates.to_sampler(original))
        assert_allclose(actual, original, atol=6e-14, rtol=0)
    assert coordinates.log_abs_det_jacobian == 0
    assert coordinates.proposal_names == ("proposal.log_kappa", "log_ell", "log_q")
    with pytest.raises(ValueError, match="wrong shape"):
        coordinates.to_sampler([1., 2.])
    with pytest.raises(ValueError, match="parameterization"):
        SamplerConfig(parameterization="unregistered")


def test_transformed_target_matches_independent_observation_space_density():
    args = _problem()
    mapping = _SamplingCoordinates("ridge_total_scale", ("prior.log_precision", "log_ell", "log_q"))
    points = list(itertools.product((-1.1, .3, 1.2), (-1.5, -.4), (-1.3, .2, 1.4)))
    inspected = []
    def inspect_target(target, initial, config):
        for original in points:
            assert_allclose(target(mapping.to_sampler(original)), _dense_reference(original, args)[0],
                            atol=2e-11, rtol=2e-12)
        expected = mapping.to_sampler([.1, -.7, .2])
        assert_allclose(initial, expected, atol=1e-15)
        inspected.append(True)
        return {"method":"prior_center_with_jitter", "scope":"test initialization"}
    config = SamplerConfig(chains=2, warmup=3, draws=8, parameterization="ridge_total_scale")
    with mock.patch("bayes_stesn.sampling._calibrate_initialization", side_effect=inspect_target):
        fitted = sample_regression(*args, config)
    assert inspected == [True]
    assert fitted.parameter_names == ("prior.log_precision", "log_ell", "log_q")
    assert fitted.metadata["parameterization"] == "ridge_total_scale"
    for value, density in zip(fitted.chains.reshape(-1, 3), fitted.log_density.ravel()):
        assert_allclose(density, _dense_reference(value, args)[0], atol=2e-11)


@pytest.mark.parametrize("kwargs", [
    {"prior":PriorSpec.grouped()},
    {"covariance":CovarianceSpec(anisotropy=True)},
    {"covariance":CovarianceSpec(static_variance=True)},
    {"covariance":CovarianceSpec(nonstationary=True)},
    {"covariance":CovarianceSpec(infer_nu=True)},
    {"covariance":CovarianceSpec(kind="vecchia")},
    {"covariance":CovarianceSpec(scale_mode="separate", range_prior="pc")},
    {"algorithm":"conditional"}, {"temporal":"full_residual_ar"},
])
def test_unsupported_coordinates_are_rejected_explicitly(kwargs):
    args = list(_problem())
    kwargs = dict(kwargs)
    args[5] = kwargs.pop("prior", args[5])
    args[6] = kwargs.pop("covariance", args[6])
    config = SamplerConfig(chains=2, warmup=1, draws=8, parameterization="ridge_total_scale", **kwargs)
    with pytest.raises(ValueError, match="ridge_total_scale"):
        sample_regression(*args, config)


def test_augmented_entry_point_cannot_ignore_coordinate_policy():
    with pytest.raises(ValueError, match="ridge_total_scale"):
        sample_augmented(*_problem(), SamplerConfig(parameterization="ridge_total_scale"), kind="field_t")


@pytest.mark.parametrize("interrupt_at", [4, 12, 20])
def test_transformed_checkpoint_warmup_sampling_and_continuation_are_exact(interrupt_at):
    args = _problem()
    config = SamplerConfig(chains=2, warmup=8, draws=20, checkpoint_every=4,
                           initialize="prior", parameterization="ridge_total_scale", seed=103)
    full = sample_regression(*args, config)
    checkpoints = {}
    class Interrupted(Exception):
        pass
    def save(chain, state):
        checkpoints[chain] = copy.deepcopy(state)
        if chain == 0 and state["iteration"] == interrupt_at:
            raise Interrupted()
    with pytest.raises(Interrupted):
        sample_regression(*args, config, checkpoint_callback=save)
    resumed = sample_regression(*args, config, resume_state=checkpoints)
    assert_array_equal(resumed.chains, full.chains)
    assert_array_equal(resumed.log_density, full.log_density)
    assert_array_equal(resumed.acceptance, full.acceptance)
    with pytest.raises(ValueError, match="does not match"):
        sample_regression(*args, replace(config, parameterization="legacy"), resume_state=checkpoints)
    endpoints = {}
    sample_regression(*args, replace(config, draws=12),
                      checkpoint_callback=lambda c,s:endpoints.update({c:copy.deepcopy(s)}))
    extended = sample_regression(*args, config, resume_state=endpoints)
    assert_array_equal(extended.chains, full.chains)
    assert_array_equal(extended.log_density, full.log_density)


def _quadrature(args, order):
    """Independent Gauss-Hermite prior integration, no sampler target used."""
    prior, cov = args[-2:]
    center = np.array([prior.mean_log_precision, cov.log_ell_mean, cov.log_q_mean])
    scale = np.array([prior.sd_log_precision, cov.log_ell_sd, cov.log_q_sd])
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    values, log_weights = [], []
    for indices in itertools.product(range(order), repeat=3):
        point = center+np.sqrt(2)*scale*nodes[list(indices)]
        _, loglik, coefficient, variance = _dense_reference(point, args)
        values.append(np.r_[point, coefficient, variance])
        log_weights.append(loglik+np.log(weights[list(indices)]).sum())
    values, log_weights = np.asarray(values), np.asarray(log_weights)
    weights = np.exp(log_weights-special.logsumexp(log_weights))
    return weights@values


def test_both_coordinate_algorithms_agree_with_coupled_posterior_quadrature():
    args = _problem()
    reference = _quadrature(args, 15)
    refined = _quadrature(args, 23)
    reference_error = np.abs(refined-reference)
    assert np.max(reference_error) < 2e-5
    for policy in ("legacy", "ridge_total_scale"):
        config = SamplerConfig(chains=4, warmup=400, draws=900, seed=83,
                               parameterization=policy)
        posterior = sample_regression(*args, config)
        values = np.empty((4, 900, 5))
        values[:, :, :3] = posterior.chains
        for c, chain in enumerate(posterior.states):
            for j, state in enumerate(chain):
                draw = posterior.draw_parameters(c, j, np.random.default_rng(9101+c*900+j))
                values[c, j, 3:] = [draw.C[0, 0], draw.sigma2]
        diagnostics = summarize_chains(values)
        # A reference test of the target, not an assertion that this short
        # run is qualified for a scientific benchmark. Tolerance is tied to
        # measured chain MC error and independent quadrature refinement.
        assert np.isfinite(diagnostics.mcse_mean).all()
        errors = np.abs(diagnostics.mean-refined)
        tolerance = 6*diagnostics.mcse_mean+2*reference_error
        assert np.all(errors <= tolerance), (policy, errors, tolerance)
