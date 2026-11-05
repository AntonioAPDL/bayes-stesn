"""Fixed-fit KS forecast and learned-feature information-boundary checks.

Only the separate pilot library is used. Small configurations exercise the
same implementation; they are acceptance fixtures, not selected study models.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from bayes_stesn.applications.ks_baselines import fit_vector
from bayes_stesn.applications.ks_data import KSAdapter
from bayes_stesn.features import FeatureSpec, FrozenFeatures
from bayes_stesn.io import content_hash


CONFIGURATIONS = (
    ("training_mean", {}),
    ("persistence", {}),
    ("linear_var", {"rank": 8, "lags": 4, "spacing": 1, "ridge": .01}),
    ("pod_nvar", {"rank": 8, "lags": 2, "spacing": 2, "ridge": .01}),
    ("ridge_esn", {"units": 32, "washout": 128, "connectivity": .2,
                    "spectral_radius": .8, "leak": .5, "input_scale": .1,
                    "readout": "states_and_squares", "ridge": .01}),
)


@pytest.fixture(scope="module")
def adapter():
    value = os.environ.get("BAYES_STESN_KS_CACHE")
    if not value:
        pytest.skip("Set BAYES_STESN_KS_CACHE to the verified private KS cache")
    pytest.importorskip("h5py")
    return KSAdapter(Path(value), "interleaved", "pilot")


@pytest.mark.parametrize("model,config", CONFIGURATIONS)
def test_fixed_fit_forecast_and_refit_invariance_outside_information_boundary(adapter, monkeypatch, model, config):
    """Freeze fit/randomness before changing forbidden histories and futures."""
    with threadpool_limits(limits=1):
        training = adapter.training(0)
        history = adapter.model_window(0, 1200, "validation")
        fit = fit_vector(model, training.panel, config, seed=41)
        fit_identity = content_hash(fit.to_state())
        before = fit.forecast(history.panel, 100, history.target_coordinates,
                              history.target_site_ids, seed=73)
        original = adapter.native.read_observed
        forbidden = adapter.partition != 0

        def changed_read(handle, start=None, stop=None):
            values = original(handle, start, stop).copy()
            times = np.arange(0 if start is None else start,
                              adapter.n_states if stop is None else stop)
            values[:, forbidden] += 12345.
            values[times >= 1200] -= 24680.
            return values

        monkeypatch.setattr(adapter.native, "read_observed", changed_read)
        changed_history = adapter.model_window(0, 1200, "validation")
        after = fit.forecast(changed_history.panel, 100,
                             changed_history.target_coordinates,
                             changed_history.target_site_ids, seed=73)
        np.testing.assert_array_equal(before.mean, after.mean)
        assert before.samples is None and after.samples is None
        assert content_hash(fit.to_state()) == fit_identity
        # This is a separate preprocessing/refitting test; the forecast above
        # did not refit anything or repeat selection/calibration.
        changed_training = adapter.training(0)
        refit = fit_vector(model, changed_training.panel, config, seed=41)
        assert content_hash(refit.to_state()) == fit_identity


def test_accepted_frozen_encoder_refit_uses_permitted_prefix_only(adapter, monkeypatch):
    with threadpool_limits(limits=1):
        training = adapter.training(0)
        spec = FeatureSpec(modes=8, units=(32,), lags=(1, 2), squares=16,
                           recurrent_norm=.9, seed=101)
        first = FrozenFeatures.fit(training.values, 80, training.coordinates, spec)
        original = adapter.native.read_observed
        forbidden = adapter.partition != 0

        def changed_read(handle, start=None, stop=None):
            values = original(handle, start, stop).copy()
            times = np.arange(0 if start is None else start,
                              adapter.n_states if stop is None else stop)
            values[:, forbidden] *= -31.
            values[times >= 1200] += 54321.
            return values

        monkeypatch.setattr(adapter.native, "read_observed", changed_read)
        changed = adapter.training(0)
        second = FrozenFeatures.fit(changed.values, 80, changed.coordinates, spec)
        assert first.fingerprint == second.fingerprint
        np.testing.assert_array_equal(first.Z, second.Z)
        np.testing.assert_array_equal(first.states, second.states)


@pytest.mark.parametrize("model,config", CONFIGURATIONS[2:])
def test_query_decoder_does_not_change_observed_or_latent_recurrence(adapter, model, config):
    with threadpool_limits(limits=1):
        training = adapter.training(0)
        history = adapter.model_window(0, 1200, "validation")
        fit = fit_vector(model, training.panel, config, seed=41)
        first = fit.forecast(history.panel, 100, history.target_coordinates,
                             history.target_site_ids)
        # Change the output domain and its ordering. Observed state evolution
        # must not depend on whether unseen query coordinates were requested.
        joint_coordinates = np.vstack((training.coordinates, history.target_coordinates[::-1]))
        joint_ids = tuple(map(str, training.site_ids)) + tuple(map(str, history.target_site_ids[::-1]))
        joint = fit.forecast(history.panel, 100, joint_coordinates, joint_ids)
        np.testing.assert_allclose(first.mean, joint.mean[:, -len(history.target_site_ids):][:, ::-1],
                                   rtol=2e-14, atol=2e-14)


def test_vector_fit_rejects_same_grid_history_from_another_episode(adapter):
    fit = fit_vector("persistence", adapter.training_panel(0), {}, seed=41)
    wrong = adapter.model_window(1, 1200, "validation")
    with pytest.raises(ValueError, match="episode|protocol"):
        fit.forecast(wrong.panel, 100, wrong.target_coordinates, wrong.target_site_ids)


def test_saved_native_pilot_forecast_invariant_without_refitting(adapter, monkeypatch):
    """Use an existing provisional pilot artifact; this is not a new MCMC fit."""
    value = os.environ.get("BAYES_STESN_KS_NATIVE_PILOT")
    if not value:
        pytest.skip("Set BAYES_STESN_KS_NATIVE_PILOT to the existing native pilot fit")
    from bayes_stesn.applications.ks_native import NativeFit

    fit = NativeFit.load(Path(value))
    assert fit.training.metadata["phase"] == "pilot"
    assert fit.training.metadata["trajectory"] == 0
    assert fit.training.metadata["ks_protocol"] == "interleaved"
    before_window = adapter.model_window(0, 1200, "validation")
    with threadpool_limits(limits=1):
        before = fit.forecast(before_window.panel, 100, before_window.target_coordinates,
                              before_window.target_site_ids, n_paths=16, seed=83)
        original = adapter.native.read_observed
        forbidden = adapter.partition != 0

        def changed_read(handle, start=None, stop=None):
            values = original(handle, start, stop).copy()
            times = np.arange(0 if start is None else start,
                              adapter.n_states if stop is None else stop)
            values[:, forbidden] += 12345.
            values[times >= 1200] -= 24680.
            return values

        monkeypatch.setattr(adapter.native, "read_observed", changed_read)
        after_window = adapter.model_window(0, 1200, "validation")
        after = fit.forecast(after_window.panel, 100, after_window.target_coordinates,
                             after_window.target_site_ids, n_paths=16, seed=83)
        np.testing.assert_array_equal(before.samples, after.samples)
        np.testing.assert_array_equal(before.mean, after.mean)
        np.testing.assert_array_equal(before.mean, before.samples.mean(axis=0))
        wrong = adapter.model_window(1, 1200, "validation")
        with pytest.raises(ValueError, match="identity|trajectory|protocol"):
            fit.forecast(wrong.panel, 100, wrong.target_coordinates,
                         wrong.target_site_ids, n_paths=16, seed=83)
