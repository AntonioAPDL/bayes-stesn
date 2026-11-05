"""Independent algebra and information-boundary checks for the new KS variants."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.linalg import block_diag

from bayes_stesn.applications.ks_fullfield_native import (
    FullFieldKSFit, _diagnose_fullfield, fit_fullfield, load_fullfield_fit,
    prepare_fullfield, resolve_fullfield_config,
)
from bayes_stesn.applications.ks_baselines import periodic_decoder
from bayes_stesn.applications.ks_native import _resolve_bayes
from bayes_stesn.basis import SpatialMean, SpatialMeanSpec
from bayes_stesn.benchmark.contracts import Panel
from bayes_stesn.gaussian import GaussianRegression
from bayes_stesn.io import content_hash, load_state, save_state, seed_for
from bayes_stesn.prediction import ForecastContext, simulate_predictive
from bayes_stesn.sampling import ParameterDraw


def panel(rows=48, sites=8):
    theta = np.arange(sites)*2*np.pi/sites
    xy = np.column_stack((np.cos(theta), np.sin(theta)))
    t = np.arange(rows)
    y = np.sin(.17*t[:, None]+theta)+.2*np.cos(.11*t[:, None]-2*theta)
    y += .05*np.random.default_rng(48002).normal(size=y.shape)
    return Panel(y, xy, .25*t, tuple(f's{j}' for j in range(sites)),
        metadata={'data_identity_sha256': 'synthetic-training-only',
                  'transform_identity': 'synthetic'})


def window(data, count=20):
    return Panel(data.y[-count:], data.coordinates, data.times[-count:], data.site_ids,
                 metadata=dict(data.metadata))


def config(target='state', prior='ridge', **extra):
    result = {'target': target, 'conditioning': 16,
        'features': {'units': [6], 'squares': 3, 'lags': [1, 2],
                     'connectivity': .8, 'leak_patterns': [[.5]]},
        'spatial_mean': {'harmonics': 2},
        'prior': {'kind': prior, 'unit': 'group'},
        'sampler': {'chains': 2, 'warmup': 8, 'draws': 12, 'initialize': 'prior',
                    'checkpoint_every': 5},
        'diagnostics': {'functional_draws': 8, 'disk_reserve_bytes': 0,
                        'coefficient_chunk_size': 17},
        'continuation_draws': 0}
    result.update(extra)
    return result


@pytest.fixture(scope='module')
def state_fit():
    return fit_fullfield(panel(), config(), seed=12)


@pytest.fixture(scope='module')
def increment_fit():
    return fit_fullfield(panel(), config(target='increment', prior='rhs'), seed=13)


def test_fourier_geometry_normalization_and_seam_are_response_independent():
    xy = panel(sites=32).coordinates
    spec = SpatialMeanSpec(kernel='fourier', harmonics=4, coordinate_terms=False)
    whole = SpatialMean.fit(xy, spec)
    arc = SpatialMean.fit(xy[:5], spec)
    np.testing.assert_array_equal(whole.transform(xy), arc.transform(xy))
    phi = whole.transform(xy)
    np.testing.assert_allclose(phi.T@phi/32, np.diag([1.]+[1/8]*8), atol=4e-16)
    seam = np.array([[np.cos(1e-12), np.sin(1e-12)],
                     [np.cos(2*np.pi-1e-12), np.sin(2*np.pi-1e-12)]])
    np.testing.assert_allclose(whole.transform(seam)[0], whole.transform(seam)[1], atol=5e-12)
    restored = SpatialMean.from_state(whole.to_state())
    np.testing.assert_array_equal(phi, restored.transform(xy))
    with pytest.raises(ValueError, match='unit-circle'):
        whole.transform(xy*.5)


def test_increment_likelihood_and_raw_input_states_are_separate():
    data = panel()
    state = prepare_fullfield(data, config(), 17)
    delta = prepare_fullfield(data, config(target='increment'), 17)
    np.testing.assert_array_equal(delta.data.y, state.data.y)
    np.testing.assert_array_equal(delta.Z, state.Z)
    np.testing.assert_array_equal(delta.features.states, state.features.states)
    assert delta.features.fingerprint == state.features.fingerprint
    assert delta.identity != state.identity
    np.testing.assert_array_equal(delta.Y, np.diff(data.y, axis=0)[15:].T)
    assert delta.spec.features.modes == data.y.shape[1]
    assert not hasattr(delta.features.core, 'P')


@pytest.mark.parametrize('target', ['state', 'increment'])
def test_separable_likelihood_matches_independent_dense_regression(target):
    p = prepare_fullfield(panel(), config(target=target), 5)
    n, count = p.Y.shape
    phi, z = p.Phi, p.fit_Z
    R = np.eye(n)+.7*np.exp(-np.linalg.norm(p.data.coordinates[:, None]-p.data.coordinates, axis=-1))
    lam = .3
    ev = GaussianRegression(p.Y, phi, z).evaluate(R, column_precision=lam)
    # Only this deliberately tiny oracle materializes the tensor design.
    X = np.kron(z, phi)
    W = block_diag(*([np.linalg.inv(R)]*count))
    y = p.Y.reshape(-1, order='F')
    precision = lam*np.eye(X.shape[1])+X.T@W@X
    h = X.T@W@y
    expected = np.linalg.solve(precision, h)
    np.testing.assert_allclose(ev.mean.reshape(-1, order='F'), expected, rtol=2e-10, atol=2e-10)
    residual = y-X@expected
    Q = residual@W@residual+lam*(expected@expected)
    assert ev.Q == pytest.approx(Q, rel=3e-12, abs=1e-12)
    assert ev.an == 2+n*count/2
    assert ev.bn == pytest.approx(.05+Q/2, rel=3e-12)


def test_rhs_is_real_and_original_adapter_is_not_relaxed(increment_fit):
    assert increment_fit.core.prepared.spec.prior.kind == 'rhs'
    assert increment_fit.core.prepared.spec.prior.unit == 'group'
    assert increment_fit.config['sampler']['parameterization'] == 'legacy'
    assert any('tau' in key for key in increment_fit.core.posterior.parameter_names)
    assert not increment_fit.diagnostics['qualified']
    assert increment_fit.diagnostics['gates'][0]['monitoring']['all_readout_coefficients']
    with pytest.raises(ValueError):
        _resolve_bayes({'prior': {'kind': 'rhs'}}, 1)
    with pytest.raises(ValueError, match='weaken'):
        resolve_fullfield_config({'diagnostics': {'min_ess': 10}})


def test_state_forecast_matches_original_joint_gaussian_simulation(state_fit):
    data = panel(); history = window(data)
    fitted = FullFieldKSFit.from_state(state_fit.to_state())
    fitted.config['forecast_parameter_rng_policy'] = 'per_origin_v1'
    targets = np.column_stack((np.cos([.3, 2.]), np.sin([.3, 2.])))
    states, z = fitted.core.prepared.features.history(history.y)
    context = ForecastContext(len(data.y)-1, history.y, states, z, history.times,
                              fitted.core.posterior_identity)
    expected = simulate_predictive(fitted.core, context, horizon=5,
        target_coordinates=targets, target_site_ids=('a', 'b'), paths=7, seed=82,
        components=('observation',))
    actual = fitted.forecast(history, 5, targets, ('a', 'b'), 7, 82, require_qualified=False)
    np.testing.assert_allclose(actual.samples, expected.observation, rtol=8e-12, atol=8e-12)
    np.testing.assert_array_equal(actual.mean, actual.samples.mean(axis=0))


def test_unqualified_forecasts_require_an_explicit_diagnostic_override(state_fit):
    with pytest.raises(RuntimeError, match='inference-unqualified'):
        state_fit.forecast(window(panel()), 2, n_paths=3)
    result = state_fit.forecast(window(panel()), 2, n_paths=3, require_qualified=False)
    assert result.metadata['qualified'] is False
    assert result.metadata['qualification_override'] is True


def test_shared_parameter_bank_and_path_batching(increment_fit):
    fitted = FullFieldKSFit.from_state(increment_fit.to_state())
    one = fitted._parameter_draw(0, 22)
    two = fitted._parameter_draw(0, 99)
    np.testing.assert_array_equal(one[0].C, two[0].C)
    assert one[0].sigma2 == two[0].sigma2
    other = fitted._parameter_draw(0, 22, parameter_seed=943)
    assert not np.array_equal(one[0].C, other[0].C)
    history = window(panel())
    whole = fitted.forecast(history, 6, n_paths=11, seed=81, require_qualified=False)
    with patch.object(fitted.core.prepared.features, 'history',
                      wraps=fitted.core.prepared.features.history) as replay:
        batches = list(fitted.iter_forecasts(history, 6, n_paths=11, seed=81,
            batch_size=3, require_qualified=False))
    assert replay.call_count == 1
    np.testing.assert_allclose(whole.samples, np.concatenate([b.samples for b in batches]),
                               rtol=3e-12, atol=3e-12)
    repeated = fitted.forecast(history, 6, n_paths=11, seed=81, require_qualified=False)
    np.testing.assert_array_equal(whole.samples, repeated.samples)
    assert whole.metadata['forecast_internal_path_batch'] == 11


def test_increment_native_feedback_does_not_project_or_use_hidden_histories():
    data = panel(); prepared = prepare_fullfield(data, config(target='increment'), 10)
    covariance = {'log_q': 0., 'log_ell': np.log(.2)}
    draws = []

    class Posterior:
        n_chains, n_draws = 2, 10

        def draw_parameters(self, chain, draw, rng):
            draws.append((chain, draw))
            return ParameterDraw(np.zeros((prepared.Phi.shape[1], prepared.Z.shape[1])),
                1e-30, covariance, {}, {}, {})

    core = SimpleNamespace(prepared=prepared, posterior=Posterior(), posterior_identity='synthetic')
    fitted = FullFieldKSFit(core, resolve_fullfield_config(config(target='increment'), 10),
                           10, 'identity', 'inputs', {}, {'qualified': True})
    theta = [.3, 2.]
    targets = np.column_stack((np.cos(theta), np.sin(theta)))
    target_xy = np.vstack([data.coordinates, targets])
    ids = (*data.site_ids, 'new0', 'new1')
    history = window(data)
    prediction = fitted.forecast(history, 4, target_xy, ids, 3, 81)
    expected = np.r_[history.y[-1], periodic_decoder(data.coordinates, targets)@history.y[-1]]
    np.testing.assert_allclose(prediction.samples, np.broadcast_to(expected, prediction.samples.shape),
                               rtol=0, atol=2e-14)
    assert len(draws) == 3  # One static readout and covariance per complete path.
    assert prediction.metadata['newsite_response_history'] == 'never_used'
    wrong = Panel(history.y, history.coordinates, history.times, history.site_ids,
                  metadata={'data_identity_sha256': 'other-trajectory'})
    with pytest.raises(ValueError, match='trajectory/protocol/transform'):
        fitted.forecast(wrong, 4, n_paths=3)


def test_chunked_diagnostics_are_chunk_size_invariant(increment_fit, tmp_path):
    checks = dict(increment_fit.config['diagnostics'])
    checks['coefficient_chunk_size'] = 3
    small = _diagnose_fullfield(increment_fit.core, 'increment', checks, 201, tmp_path)
    checks['coefficient_chunk_size'] = 1000
    large = _diagnose_fullfield(increment_fit.core, 'increment', checks, 201, tmp_path)
    assert content_hash(small['coefficients']) == content_hash(large['coefficients'])
    assert content_hash(small['functionals']) == content_hash(large['functionals'])
    assert small['monitoring']['mapped_pages_counted_as_resident']
    assert small['monitoring']['increment_functionals_include_persistence_offset']
    assert not list(tmp_path.glob('fullfield-diagnostic-*'))


def test_portable_fit_cache_and_identity_rejections(increment_fit, tmp_path):
    path = tmp_path/'portable.npz'
    increment_fit.save(path)
    first = load_fullfield_fit(path)
    history = window(panel())
    expected = first.forecast(history, 3, n_paths=4, seed=80, require_qualified=False)
    cached = list(path.with_name(path.name+'.parameter-bank').rglob('*.npz'))
    assert len(cached) == 4
    second = load_fullfield_fit(path)
    with patch.object(second.core.posterior, 'draw_parameters', side_effect=AssertionError('cache should load')):
        actual = second.forecast(history, 3, n_paths=4, seed=80, require_qualified=False)
    np.testing.assert_array_equal(actual.samples, expected.samples)
    wrong = load_state(cached[0], kind='ks_fullfield_parameter_draw')
    wrong['identity'] = 'incompatible'
    save_state(cached[0], wrong, kind='ks_fullfield_parameter_draw')
    third = load_fullfield_fit(path)
    with pytest.raises(ValueError, match='incompatible identity'):
        third.forecast(history, 3, n_paths=4, seed=80, require_qualified=False)


def test_load_rejects_source_runtime_drift_without_diagnostic_override(state_fit, tmp_path):
    path = tmp_path/'drift.npz'
    state_fit.save(path)
    with patch('bayes_stesn.applications.ks_fullfield_native._computation_identity',
               return_value={'source_sha256': 'different', 'runtime': {}}):
        with pytest.raises(ValueError, match='numerical source/runtime'):
            load_fullfield_fit(path)
        diagnostic = load_fullfield_fit(path, allow_computation_mismatch=True)
    forecast = diagnostic.forecast(window(panel()), 2, n_paths=2, require_qualified=False)
    assert forecast.metadata['loaded_computation_mismatch_override'] is True


def test_interrupted_sampling_resumes_identical_rng_and_rejects_target_change(tmp_path):
    import bayes_stesn.model as core_model
    data = panel()
    configuration = config(target='increment', prior='rhs', continuation_draws=8)
    clean = fit_fullfield(data, configuration, 24)
    path = tmp_path/'resume.npz'
    original_save = core_model.save_state
    interrupted = False

    def save_then_interrupt(*args, **kwargs):
        nonlocal interrupted
        result = original_save(*args, **kwargs)
        if kwargs.get('kind') == 'mcmc_checkpoint' and not interrupted:
            interrupted = True
            raise InterruptedError('after durable RNG checkpoint')
        return result

    with patch.object(core_model, 'save_state', side_effect=save_then_interrupt):
        with pytest.raises(InterruptedError, match='durable RNG'):
            fit_fullfield(data, configuration, 24, path)
    resumed = fit_fullfield(data, configuration, 24, path)
    assert resumed.core.posterior_identity == clean.core.posterior_identity
    np.testing.assert_array_equal(resumed.core.posterior.chains, clean.core.posterior.chains)
    assert resumed.diagnostics['continuations'] == 1
    assert resumed.core.posterior.n_draws == 20
    again = fit_fullfield(data, configuration, 24, path)
    assert again.identity == resumed.identity
    with pytest.raises(ValueError, match='incompatible'):
        fit_fullfield(data, {**configuration, 'target': 'state'}, 24, path)


def test_pca_aliases_cannot_manufacture_distinct_candidate_hashes():
    first = resolve_fullfield_config(config(), 11)
    other = config(); other['features']['modes'] = 10000
    assert content_hash(first) == content_hash(resolve_fullfield_config(other, 11))
