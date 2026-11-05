"""Small application-contract checks; no KS regeneration or model selection.

Set BAYES_STESN_KS_CACHE to the verified restore destination to exercise the
delivered reader.  Tests with responses use the separate pilot trajectories.
"""
from __future__ import annotations

from dataclasses import fields
import os
from pathlib import Path

import numpy as np
import pytest

from bayes_stesn.applications.ks_data import (
    KSAdapter, KSModelWindow, ScalarTransform, circle_embedding,
)


@pytest.fixture(scope="module")
def cache():
    value = os.environ.get("BAYES_STESN_KS_CACHE")
    if not value:
        pytest.skip("Set BAYES_STESN_KS_CACHE to the verified private KS cache")
    pytest.importorskip("h5py")
    root = Path(value).expanduser().resolve()
    if not (root / "data/ks/manifests/core.json").is_file():
        pytest.fail("BAYES_STESN_KS_CACHE does not contain the restored KS manifest")
    return root


def test_circle_embedding_is_periodic_and_preserves_seam_neighbors():
    points = circle_embedding(np.array([0., 22., -22., 22 / 64, 22 - 22 / 64]))
    np.testing.assert_array_equal(points[0], points[1])
    np.testing.assert_array_equal(points[0], points[2])
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1., atol=2e-16)
    np.testing.assert_allclose(np.linalg.norm(points[0] - points[3]),
                               np.linalg.norm(points[0] - points[4]), rtol=2e-14)
    with pytest.raises(ValueError, match="positive"):
        circle_embedding(np.array([0.]), period=0)


def test_scalar_transform_inverse_has_known_physical_units():
    transform = ScalarTransform.from_record({"mean": 2., "scale": 3.,
                                              "fit_time_bounds": [0, 8],
                                              "fit_site_indices": [0, 2],
                                              "training_value_count": 16})
    np.testing.assert_array_equal(transform.forward(np.array([-1., 2., 5.])), [-1., 0., 1.])
    np.testing.assert_array_equal(transform.inverse(np.array([-1., 0., 1.])), [-1., 2., 5.])


@pytest.mark.parametrize("protocol,input_count,target_count", [
    ("interleaved", 39, 13), ("all_observed", 64, 64), ("contiguous", 40, 12),
])
def test_pilot_shapes_mask_selection_transform_and_time_index(cache, protocol, input_count, target_count):
    adapter = KSAdapter(cache, protocol, "pilot")
    train = adapter.training(0)
    window = adapter.model_window(0, 1200, "validation")
    evaluation = adapter.evaluation_window(0, 1200, "validation")
    assert train.values.shape == (1200, input_count)
    assert window.values.shape == (128, input_count)
    assert evaluation.values.shape == (100, target_count)
    assert window.origin == 1199 and window.first_target == 1200
    np.testing.assert_array_equal(window.panel.times, .25 * np.arange(1072, 1200))
    np.testing.assert_array_equal(window.target_time, .25 * np.arange(1200, 1300))
    assert train.panel.metadata["ks_protocol"] == protocol
    assert window.panel.metadata["ks_protocol"] == protocol
    np.testing.assert_allclose(train.values.mean(), 0., atol=2e-14)
    np.testing.assert_allclose(train.values.std(), 1., atol=2e-14)
    np.testing.assert_allclose(window.physical, train.physical[-128:], rtol=0, atol=2e-15)
    np.testing.assert_array_equal(window.coordinates, circle_embedding(window.x))
    np.testing.assert_array_equal(window.target_coordinates, circle_embedding(window.target_x))
    if protocol != "all_observed":
        assert set(window.site_ids).isdisjoint(window.target_site_ids)
    assert evaluation.mask.all()
    assert not window.panel.y.flags.writeable
    assert {f.name for f in fields(KSModelWindow)}.isdisjoint(
        {"evaluation", "latent_clean", "future_noisy_observation", "scoring_mask", "truth"})


def test_full_schedules_and_target_boundaries_are_explicit(cache):
    core = KSAdapter(cache, "interleaved", "core")
    pilot = KSAdapter(cache, "all_observed", "pilot")
    np.testing.assert_array_equal(core.starts("validation"), np.arange(6000, 8000, 100))
    np.testing.assert_array_equal(core.starts("test"), np.arange(8000, 10000, 100))
    np.testing.assert_array_equal(pilot.starts("validation"), np.arange(1200, 1600, 100))
    for first_target, partition, history in [(1599, "validation", 128),
                                             (1199, "validation", 128),
                                             (1200, "validation", 1201)]:
        with pytest.raises(ValueError, match="crosses.*partition|lacks"):
            pilot.model_window(0, first_target, partition, history=history)
    with pytest.raises(ValueError, match="integer"):
        pilot.model_window(0, 1200.0, "validation")


@pytest.mark.parametrize("perturbation", ["future", "hidden", "both"])
def test_future_and_hidden_truth_cannot_change_model_or_refitted_training_statistics(cache, monkeypatch, perturbation):
    adapter = KSAdapter(cache, "interleaved", "pilot")
    a = 1200
    before_train = adapter.training(0)
    before_window = adapter.model_window(0, a, "validation")
    before_evaluation = adapter.evaluation_window(0, a, "validation")
    original = adapter.native.read_observed
    forbidden = adapter.partition != 0

    def changed_read(handle, start=None, stop=None):
        result = original(handle, start, stop).copy()
        times = np.arange(0 if start is None else start,
                          adapter.n_states if stop is None else stop)
        if perturbation in ("hidden", "both"):
            result[:, forbidden] += 12345.6789
        if perturbation in ("future", "both"):
            result[times >= a] -= 24680.1357
        return result

    monkeypatch.setattr(adapter.native, "read_observed", changed_read)
    after_train = adapter.training(0)
    after_window = adapter.model_window(0, a, "validation")
    after_evaluation = adapter.evaluation_window(0, a, "validation")
    np.testing.assert_array_equal(after_train.values, before_train.values)
    np.testing.assert_array_equal(after_window.values, before_window.values)
    np.testing.assert_array_equal(after_window.target_coordinates, before_window.target_coordinates)
    assert after_window.identity == before_window.identity
    # Recompute, rather than merely inspect the frozen transform parameters.
    for statistic in (np.mean, np.std):
        assert statistic(after_train.physical) == statistic(before_train.physical)
    # The perturbation actually reached evaluator-only fields; this is not a vacuous test.
    assert not np.array_equal(before_evaluation.values, after_evaluation.values)


def test_episode_reads_are_independent_and_identity_separates_protocols(cache):
    strict = KSAdapter(cache, "interleaved", "pilot")
    temporal = KSAdapter(cache, "all_observed", "pilot")
    first = strict.model_window(0, 1200, "validation")
    other = strict.model_window(1, 1200, "validation")
    repeated = strict.model_window(0, 1200, "validation")
    np.testing.assert_array_equal(first.values, repeated.values)
    assert first.identity["identity_sha256"] != other.identity["identity_sha256"]
    assert strict.identity(0)["identity_sha256"] != temporal.identity(0)["identity_sha256"]
    assert not np.shares_memory(first.values, other.values)
