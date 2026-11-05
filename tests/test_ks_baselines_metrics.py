import numpy as np
import pytest
from bayes_stesn.benchmark.contracts import Panel
from bayes_stesn.applications.ks_baselines import periodic_decoder, polynomial_features, feature_count, fit_vector, VectorFit
from bayes_stesn.applications.ks_metrics import score, horizon_balance


def panel(n=150, sites=8):
    a = np.arange(sites)*2*np.pi/sites
    c = np.column_stack((np.cos(a), np.sin(a)))
    t = np.arange(n)
    y = np.sin(.03*t[:, None]+a)
    return Panel(y, c, .25*t, tuple(map(str, range(sites))))


def test_decoder_identity_and_periodic_seam():
    p = panel()
    np.testing.assert_allclose(periodic_decoder(p.coordinates, p.coordinates), np.eye(8), atol=1e-14)
    eps = 1e-8
    points = np.array([[np.cos(eps), np.sin(eps)], [np.cos(-eps), np.sin(-eps)]])
    weights = periodic_decoder(p.coordinates, points)
    np.testing.assert_allclose(weights.sum(axis=1), 1.)
    assert np.max(np.abs(weights[0]-weights[1])) < 3e-8


def test_quadratic_intercept_and_count():
    x = polynomial_features(np.array([2., 3.]))
    np.testing.assert_array_equal(x, [1, 2, 3, 4, 6, 9])
    assert feature_count(16, 2) == 561
    assert feature_count(24, 2) == 1225


@pytest.mark.parametrize('model,cfg', [
    ('training_mean', {}), ('persistence', {}),
    ('linear_var', {'rank': 4, 'lags': 2, 'ridge': .001}),
    ('pod_nvar', {'rank': 4, 'lags': 2, 'ridge': .001}),
    ('ridge_esn', {'units': 24, 'washout': 32, 'ridge': .001, 'connectivity': .2})])
def test_roundtrip_and_independent_origin_state(tmp_path, model, cfg):
    p = panel(); fit = fit_vector(model, p, cfg, 17)
    h = Panel(p.y[-64:], p.coordinates, p.times[-64:], p.site_ids)
    f = fit.forecast(h, 8, p.coordinates, p.site_ids)
    # Another forecast cannot mutate the fitted state or an origin context.
    fit.forecast(Panel(p.y[-80:], p.coordinates, p.times[-80:], p.site_ids), 8, p.coordinates, p.site_ids)
    np.testing.assert_array_equal(f.mean, fit.forecast(h, 8, p.coordinates, p.site_ids).mean)
    path = tmp_path/'fit.npz'; fit.save(path)
    np.testing.assert_array_equal(f.mean, VectorFit.load(path).forecast(h, 8, p.coordinates, p.site_ids).mean)


def test_scores_known_answers_mean_mae_and_bands():
    truth = np.zeros((100, 2)); paths = np.broadcast_to(np.array([0., 0., 3.])[:, None, None], (3, 100, 2)).copy()
    coords = np.array([[1., 0.], [-1., 0.]])
    s = score(paths.mean(axis=0), paths, truth, 2., coords)
    assert s['rmse'] == s['mae'] == 1.
    assert s['nrmse'] == .5
    assert s['crps'] == pytest.approx(1/3)
    assert s['vpt_steps'] == 1 and not s['vpt_right_censored']
    assert horizon_balance(np.r_[np.ones(10), np.full(15, 2.), np.full(25, 3.), np.full(50, 4.)]) == 2.5
    zero = score(truth, None, truth, 2., coords)
    assert zero['vpt_steps'] == 100 and zero['vpt_right_censored']
    assert zero['rmse'] == 0.


def test_no_fake_probability_for_point_forecast():
    p = panel(); f = fit_vector('persistence', p, {}, 0).forecast(p, 5, p.coordinates, p.site_ids, n_paths=256)
    assert f.samples is None
    assert 'crps' not in score(f.mean, None, f.mean, 1., p.coordinates)
