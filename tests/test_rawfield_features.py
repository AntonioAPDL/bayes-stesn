"""Information boundaries and numerical references for raw-field reservoirs."""
from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from bayes_stesn.features import FeatureSpec, FrozenFeatures
from bayes_stesn.io import content_hash, load_state, save_state
from bayes_stesn.raw_reservoir import raw_feature_allocation, scale_recurrent_matrix


def panel(rows=90, sites=12):
    rng = np.random.default_rng(25)
    y = rng.normal(size=(rows, sites))
    theta = np.linspace(0, 2 * np.pi, sites, endpoint=False)
    return y, np.column_stack([np.cos(theta), np.sin(theta)])


def configuration(**overrides):
    options = dict(kind="raw_esn", units=(7,), lags=(1, 3), squares=4,
                   products=2, input_scales=(.4,), leak_patterns=((.2, .7),),
                   connectivity=1., recurrent_scaling="spectral_radius",
                   spectral_radius=1.2, score_bound=None, seed=8)
    options.update(overrides)
    return FeatureSpec(**options)


def test_all_input_sites_survive_without_a_response_projection():
    # More sites than conditioning rows and a rank-deficient field are both
    # valid: this feature map does not estimate a PCA subspace.
    y, xy = panel(rows=40, sites=25)
    y[:, 1] = y[:, 0]
    spec = configuration(lags=(1,), modes=2)
    with patch("bayes_stesn.features.svd", side_effect=AssertionError("No response SVD")):
        fitted = FrozenFeatures.fit(y, 10, xy, spec)
    assert fitted.spec.modes == y.shape[1]
    assert fitted.core.input_dimension == y.shape[1]
    assert not hasattr(fitted.core, "P")
    recovered_field = fitted.states[21, :25] * fitted.core.site_scale + fitted.core.site_center
    np.testing.assert_allclose(recovered_field, y[20], rtol=0, atol=3e-16)
    # In particular the last input coordinate cannot disappear under a
    # low-rank projection. Its direct readout changes at precisely lead one.
    changed = y.copy(); changed[20, -1] += 2.
    other = FrozenFeatures.fit(changed, 10, xy, spec)
    assert fitted.fingerprint == other.fingerprint
    np.testing.assert_array_equal(fitted.Z[:21], other.Z[:21])
    assert abs(fitted.Z[21, 25] - other.Z[21, 25]) > .1


def test_preprocessing_is_prefix_only_and_features_are_causal():
    y, xy = panel()
    first = FrozenFeatures.fit(y, 30, xy, configuration())
    changed = y.copy(); changed[45:] += 1000
    second = FrozenFeatures.fit(changed, 30, xy, configuration())
    assert first.fingerprint == second.fingerprint
    np.testing.assert_array_equal(first.Z[:46], second.Z[:46])
    assert np.max(np.abs(first.Z[46:] - second.Z[46:])) > .1
    np.testing.assert_array_equal(first.core.site_center, y[:30].mean(axis=0))
    np.testing.assert_array_equal(first.core.site_scale, y[:30].std(axis=0))


def test_radius_is_not_operator_norm_and_can_exceed_one():
    nonnormal = np.array([[.4, 10.], [0., .2]])
    scaled, diagnostic = scale_recurrent_matrix(nonnormal,
        recurrent_scaling="spectral_radius", recurrent_norm=.6, spectral_radius=1.25)
    # Triangular eigenvalues are its diagonal, an independent known answer.
    np.testing.assert_allclose(np.diag(scaled), [1.25, .625], rtol=0, atol=1e-15)
    assert np.linalg.norm(scaled, 2) > 30
    assert diagnostic["spectral_radius"] == pytest.approx(1.25)
    assert diagnostic["operator_norm"] == pytest.approx(np.linalg.norm(scaled, 2))
    assert diagnostic["achieved_target"]
    by_norm, norm_diagnostic = scale_recurrent_matrix(nonnormal,
        recurrent_scaling="operator_norm", recurrent_norm=.6, spectral_radius=None)
    assert np.linalg.norm(by_norm, 2) == pytest.approx(.6)
    assert norm_diagnostic["spectral_radius"] < .03


@pytest.mark.parametrize("matrix", [np.zeros((2, 2)), np.array([[0., 1.], [0., 0.]]),
                                    np.array([[1e-14, 1.], [0., 2e-14]])])
def test_degenerate_radius_is_rejected_without_seed_redraw(matrix):
    with pytest.raises(ValueError, match="Degenerate recurrent spectral_radius"):
        scale_recurrent_matrix(matrix, recurrent_scaling="spectral_radius",
                               recurrent_norm=.9, spectral_radius=.9)


def test_sparse_generated_radius_and_leak_are_recorded_separately():
    y, xy = panel()
    f = FrozenFeatures.fit(y, 30, xy, configuration(connectivity=.4))
    matrix = f.core.W[0]
    actual = np.max(np.abs(np.linalg.eigvals(matrix)))
    assert actual == pytest.approx(1.2, rel=2e-12, abs=2e-12)
    diagnostic = f.core.recurrent_diagnostics[0]
    assert diagnostic["spectral_radius"] == pytest.approx(actual, rel=2e-12)
    assert diagnostic["minimum_leak"] == .2
    assert np.count_nonzero(matrix) < matrix.size


def test_direct_readout_selection_does_not_change_reservoir_information():
    y, xy = panel()
    maps = {kind: FrozenFeatures.fit(y, 30, xy,
            configuration(raw_readout_lags=kind)) for kind in ["all", "latest", "none"]}
    for other in [maps["latest"], maps["none"]]:
        np.testing.assert_array_equal(maps["all"].states, other.states)
        np.testing.assert_array_equal(maps["all"].core.U[0], other.core.U[0])
    assert [maps[k].Z.shape[1] for k in ["all", "latest", "none"]] == [38, 26, 14]
    assert [maps[k].core.input_dimension for k in maps] == [24, 24, 24]
    for f in maps.values():
        assert len(f.groups) == f.Z.shape[1]
    with pytest.raises(ValueError, match="requires lag 1"):
        configuration(lags=(2, 4), raw_readout_lags="latest")


def test_spaced_and_long_lags_recover_the_exact_observed_field_history():
    y, xy = panel(rows=100, sites=4)
    fitted = FrozenFeatures.fit(y, 50, xy,
        configuration(lags=(1, 2, 8, 16, 32), raw_readout_lags="latest"))
    memory = fitted.states[80, :fitted.core.history_size].reshape(32, 4)
    recovered = memory * fitted.core.site_scale + fitted.core.site_center
    np.testing.assert_allclose(recovered, y[48:80][::-1], rtol=0, atol=5e-16)


@pytest.mark.parametrize("readout", ["all", "latest", "none"])
def test_serialization_batches_and_episode_reset(tmp_path, readout):
    y, xy = panel()
    fitted = FrozenFeatures.fit(y, 30, xy, configuration(raw_readout_lags=readout))
    path = tmp_path / "features.npz"
    save_state(path, fitted.to_state())
    restored = FrozenFeatures.from_state(load_state(path))
    assert fitted.fingerprint == restored.fingerprint
    np.testing.assert_array_equal(fitted.Z, restored.Z)
    state, z = fitted.update_batch(y[40:44], fitted.states[40:44])
    for j in range(4):
        scalar_state, scalar_z = restored.update(y[40 + j], fitted.states[40 + j])
        np.testing.assert_allclose(state[j], scalar_state, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(z[j], scalar_z, rtol=2e-14, atol=2e-14)
    history_state, history_z = fitted.history(y)
    np.testing.assert_array_equal(history_state, fitted.states)
    np.testing.assert_allclose(history_z, fitted.Z, rtol=2e-15, atol=2e-15)
    expected_state, expected_z = fitted.history(y[40:60])
    fitted.history(100 * y[60:80])  # An unrelated episode must leave no state.
    repeat_state, repeat_z = fitted.history(y[40:60])
    np.testing.assert_array_equal(repeat_state, expected_state)
    np.testing.assert_array_equal(repeat_z, expected_z)
    compact = FrozenFeatures.from_state(fitted.to_state(include_history=False))
    np.testing.assert_array_equal(compact.history(y[40:60])[0], expected_state)
    assert content_hash(fitted.core.recurrent_diagnostics) == content_hash(restored.core.recurrent_diagnostics)


def test_memory_accounting_includes_full_delay_inputs_and_precedes_weights():
    cost = raw_feature_allocation(6000, 64, units=(1000,), lags=tuple(range(1, 33)),
                                squares=1000, raw_readout_lags="latest")
    assert cost["input_dimension"] == 2048
    assert cost["readout_input_dimension"] == 64
    assert cost["dense_input_weights_bytes"] == 1000 * 2048 * 8
    assert cost["state_size"] == 3048
    assert cost["temporal_columns"] == 2065
    assert cost["estimated_peak_feature_bytes"] > cost["states_bytes"] + cost["temporal_design_bytes"]
    y, xy = panel()
    with patch("bayes_stesn.raw_reservoir.np.random.default_rng",
               side_effect=AssertionError("Weights must not be allocated")):
        with pytest.raises(MemoryError, match="max_feature_bytes"):
            FrozenFeatures.fit(y, 30, xy, configuration(max_feature_bytes=1))


def test_old_pca_route_rejects_inactive_raw_options():
    with pytest.raises(ValueError, match="require kind='raw_esn'"):
        FeatureSpec(spectral_radius=.9, recurrent_scaling="spectral_radius")
    # Legacy dictionaries with no raw-specific keys remain loadable.
    y, xy = panel()
    spec = FeatureSpec(modes=3, units=(7,), squares=3)
    fitted = FrozenFeatures.fit(y, 30, xy, spec)
    state = fitted.to_state()
    for name in ["recurrent_scaling", "spectral_radius", "raw_readout_lags"]:
        state["spec"].pop(name)
    restored = FrozenFeatures.from_state(state)
    np.testing.assert_array_equal(restored.history(y)[0], fitted.states)


def test_optional_bounded_direct_terms_do_not_change_recurrent_states():
    y, xy = panel()
    plain = FrozenFeatures.fit(y, 30, xy, configuration(score_bound=None))
    bounded = FrozenFeatures.fit(y, 30, xy, configuration(score_bound=3.))
    np.testing.assert_array_equal(plain.states, bounded.states)
    assert not np.array_equal(plain.Z, bounded.Z)
