#!/usr/bin/env python3
"""Check a frozen full-field fit against delivered facade truth perturbations.

This diagnostic never fits a posterior, selects a candidate, or scores a
predictive comparison. Feature/preprocessing refitting is a separate check on
the unchanged permitted training panel. The facade may still read future
evaluation labels from disk; only output separation is established here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from bayes_stesn.applications.ks_data import KSAdapter
from bayes_stesn.applications.ks_fullfield_native import load_fullfield_fit, prepare_fullfield
from bayes_stesn.applications.ks_runtime import atomic_json
from bayes_stesn.io import content_hash


def check(cache, path, *, history=128, paths=16, allow_prefreeze=False):
    fit = load_fullfield_fit(path, allow_computation_mismatch=allow_prefreeze)
    metadata = fit.training.metadata
    phase, track, trajectory = metadata['phase'], metadata['ks_protocol'], metadata['trajectory']
    adapter = KSAdapter(Path(cache), track, phase)
    first = int(metadata['time_index_bounds'][1])
    before_window = adapter.model_window(trajectory, first, 'validation', history=history)
    labels = adapter.evaluation_window(trajectory, first, 'validation', history=history).values
    identity = content_hash(fit.to_state())
    before = fit.forecast(before_window.panel, 100, before_window.target_coordinates,
        before_window.target_site_ids, paths, 93, require_qualified=False)
    original = adapter.native.read_observed
    hidden = adapter.partition != 0
    calls = 0

    def changed_read(handle, start=None, stop=None):
        nonlocal calls
        calls += 1
        values = original(handle, start, stop).copy()
        times = np.arange(0 if start is None else start,
                          adapter.n_states if stop is None else stop)
        if track != 'all_observed':
            values[:, hidden] += 12345.
        values[times >= first] -= 24680.
        return values

    adapter.native.read_observed = changed_read
    try:
        changed_window = adapter.model_window(trajectory, first, 'validation', history=history)
        changed_labels = adapter.evaluation_window(trajectory, first, 'validation', history=history).values
        assert not np.array_equal(labels, changed_labels)
        after = fit.forecast(changed_window.panel, 100, changed_window.target_coordinates,
            changed_window.target_site_ids, paths, 93, require_qualified=False)
        np.testing.assert_array_equal(before_window.panel.y, changed_window.panel.y)
        np.testing.assert_array_equal(before.samples, after.samples)
        np.testing.assert_array_equal(before.mean, after.mean)
        assert content_hash(fit.to_state()) == identity
        training = adapter.training(trajectory)
        np.testing.assert_array_equal(training.panel.y, fit.training.y)
        prepared = prepare_fullfield(training.panel, fit.config, fit.seed)
        assert prepared.features.fingerprint == fit.core.prepared.features.fingerprint
        np.testing.assert_array_equal(prepared.Z, fit.core.prepared.Z)
        np.testing.assert_array_equal(prepared.Phi, fit.core.prepared.Phi)
        np.testing.assert_array_equal(prepared.Y, fit.core.prepared.Y)
    finally:
        adapter.native.read_observed = original
    return {'schema': 'ks-fullfield-boundary-check/1', 'status': 'passed',
        'fit': str(Path(path).resolve()), 'fit_identity': fit.identity,
        'phase': phase, 'track': track, 'trajectory': trajectory,
        'first_target_index': first, 'history': history, 'horizon': 100, 'paths': paths,
        'inference_qualified': fit.diagnostics['qualified'], 'require_qualified': False,
        'allow_prefreeze': allow_prefreeze, 'reader_calls_under_perturbation': calls,
        'future_evaluation_values_confirmed_changed': True,
        'hidden_histories_changed': track != 'all_observed',
        'model_inputs_unchanged': True, 'complete_predictive_paths_unchanged': True,
        'point_means_unchanged': True, 'fitted_artifacts_unchanged': True,
        'permitted_preprocessing_and_features_refit_unchanged': True,
        'likelihood_targets_unchanged': True, 'inference_rerun': False,
        'selection_or_calibration_rerun': False,
        'claim': 'fixed-fit output separation; no claim of future-free disk access or predictive improvement'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--fit', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--history', type=int, default=128)
    parser.add_argument('--paths', type=int, default=16)
    parser.add_argument('--allow-prefreeze', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        receipt = check(args.cache, args.fit, history=args.history,
                        paths=args.paths, allow_prefreeze=args.allow_prefreeze)
    atomic_json(args.output, receipt)
    print(receipt)


if __name__ == '__main__':
    main()
