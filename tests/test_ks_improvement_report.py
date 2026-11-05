"""Focused reporting checks; no model fitting or delivered data access."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def load_report():
    path = Path(__file__).resolve().parents[1] / "tools/build_ks_improvement_report.py"
    spec = importlib.util.spec_from_file_location("ks_improvement_report_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = load_report()


def origin(mse, *, diverged=False, scale=2., crps=None):
    metrics = {"training_scale": scale, "mse_by_horizon": [float(mse)] * 100,
               "mae_by_horizon": [float(np.sqrt(mse))] * 100,
               "vpt_physical": 25., "vpt_right_censored": True, "diverged": diverged}
    if crps is not None:
        metrics["crps_by_horizon"] = [float(crps)] * 100
        metrics["intervals"] = {str(level): {"coverage_by_horizon": [level] * 100,
                                            "width_by_horizon": [float(level * 2)] * 100}
                                for level in (.5, .8, .9, .95)}
    return {"state": "complete", "metrics": metrics}


def test_rmse_pools_within_trajectory_then_averages_trajectories():
    values = {(1, 6000): origin(1., crps=.2), (1, 6100): origin(9., crps=.4),
              (2, 6000): origin(16., crps=.6), (2, 6100): origin(16., crps=.8)}
    result = report.summarize_cohort(values, [1, 2], [6000, 6100],
                                     {1: "qualified", 2: "qualified"}, {1: "complete", 2: "complete"})
    expected = (np.sqrt(5.) + 4.) / 2
    assert result["metrics"]["rmse"]["mean"] == pytest.approx(expected)
    assert result["metrics"]["rmse"]["mean"] != pytest.approx(3.)  # mean of origin RMSEs
    assert result["complete_case_pooled_rmse"] == pytest.approx(np.sqrt(10.5))
    assert result["metrics"]["nrmse"]["mean"] == pytest.approx(expected / 2.)
    assert result["metrics"]["crps"]["mean"] == pytest.approx(.5)
    assert result["metrics"]["coverage90"]["mean"] == pytest.approx(.9)
    assert result["curves"]["rmse"]["mean"] == pytest.approx([expected] * 100)
    assert result["full_schedule_point_score_defined"]


def test_failed_rollouts_blocked_jobs_and_pending_are_distinct():
    outcomes = {(1, 6000): origin(1., diverged=True), (1, 6100): origin(9.),
                (2, 6000): origin(4.), (2, 6100): {"state": "failed", "error": "nonfinite"}}
    result = report.summarize_cohort(outcomes, [1, 2, 3, 4], [6000, 6100],
              {1: "qualified", 2: "qualified", 3: "missing", 4: "missing"},
              {1: "complete", 2: "failed", 3: "failed", 4: "pending"})
    assert result["scheduled_origins"] == 8
    assert result["finite_successful_origins"] == 3
    assert result["failed_rollout_origins"] == 1
    assert result["failed_job_blocked_origins"] == 2
    assert result["pending_origins"] == 2
    assert result["finite_divergent_origins"] == 1
    assert result["failure_or_divergence_origins"] == 2
    assert result["complete_trajectory_ids"] == [1]
    assert result["score_population_origins"] == 2
    assert result["finite_origins_outside_score_population"] == 1
    assert not result["full_schedule_point_score_defined"]
    assert not result["all_scheduled_jobs_terminal"]
    assert result["failure_frequency"]["ci95"] is None
    assert result["failure_frequency"]["identified_bounds"] == [1 / 8, 5 / 8]


def test_terminal_unqualified_work_is_complete_but_not_accuracy_evidence():
    result = report.summarize_cohort({}, [1], [6000, 6100], {1: "unqualified"}, {1: "inference-unqualified"})
    assert result["unqualified_blocked_origins"] == 2
    assert result["pending_origins"] == 0
    assert result["all_scheduled_jobs_terminal"]
    assert result["metrics"]["rmse"]["mean"] is None
    assert result["metrics"]["crps"]["mean"] is None
    with pytest.raises(ValueError, match="qualified"):
        report.summarize_cohort({(1, 6000): origin(1.)}, [1], [6000], {1: "unqualified"})


def test_pairs_use_trajectory_intersection_not_unmatched_population_means():
    left = {"candidate_id": "new", "seed": 1, "trajectories": [1, 2, 3],
            "episodes": [{"trajectory": 1, "rmse": 100.}, {"trajectory": 2, "rmse": 4.}]}
    right = {"candidate_id": "old", "seed": 1, "trajectories": [1, 2, 3],
             "episodes": [{"trajectory": 2, "rmse": 3.}, {"trajectory": 3, "rmse": .01}]}
    paired = report.paired_comparisons(left, right, metrics=("rmse", "crps"))
    assert paired[0]["paired_trajectories"] == [2]
    assert paired[0]["mean"] == 1.
    assert paired[0]["n"] == 1
    assert paired[0]["ci95"] is None
    assert paired[1]["paired_trajectories"] == []
    assert paired[1]["mean"] is None


def test_training_scale_and_forecast_nonfinites_fail_loudly():
    with pytest.raises(ValueError, match="one positive"):
        report.episode_summary([origin(1., scale=1.), origin(1., scale=2.)], 1)
    with pytest.raises(ValueError, match="finite horizon"):
        report.episode_summary([origin(float("inf"))], 1)
    with pytest.raises(ValueError, match="Negative"):
        bad = origin(1.)
        bad["metrics"]["mse_by_horizon"][0] = -1.
        report.episode_summary([bad], 1)


def test_controlled_screen_contrast_keeps_two_matched_episodes_and_category():
    reader = SimpleNamespace(cfg={"tracks": ["A"]}, registry={"tracks": {"A": {"anchor_id": "old"}}},
        candidates={("A", "old"): {"stratum": "main_fixed_prior", "single_factor_control": False,
                                    "aliases": ["old"], "changed_effective_fields": []},
                    ("A", "leak"): {"stratum": "main_fixed_prior", "single_factor_control": True,
                                     "aliases": ["scalar_leak"], "changed_effective_fields": ["features.leak_patterns"]}})
    def stage_row(cid, values):
        return {"id": cid, "eligible": True, "loss": float(np.mean(values)), "point_loss": float(np.mean(values)) * 2,
                "episodes": [{"trajectory": t, "selection_crps": v, "selection_mse": v * 2}
                             for t, v in zip([0, 5], values)]}
    rows = report.controlled_contrasts(reader, {"screen_A": {"rows": [stage_row("old", [2., 4.]), stage_row("leak", [1., 3.])]}})
    changed = next(r for r in rows if r["candidate_id"] == "leak")
    assert changed["category"] == "controlled_single_factor"
    assert changed["paired_trajectories"] == [0, 5]
    assert changed["delta_band_crps"] == -1.
    assert changed["delta_band_mse"] == -2.


def test_anchor_nonfinite_is_numerical_failure_but_shape_error_is_not():
    tools = Path(__file__).resolve().parents[1] / "tools"
    spec = importlib.util.spec_from_file_location("ks_anchor_guard_tests", tools / "evaluate_ks_improvement_anchors.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(tools))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(tools))
    transform = SimpleNamespace(inverse=lambda x: x)
    with pytest.raises(FloatingPointError, match="Nonfinite physical-unit"):
        module.physical_prediction(np.array([[float("nan")]]), transform, (1, 1))
    with pytest.raises(ValueError, match="shape"):
        module.physical_prediction(np.ones((1, 2)), transform, (1, 1))
    assert module.physical_prediction(np.array([[2.]]), transform, (1, 1)).item() == 2.


def test_display_scale_detects_negative_extremes_without_changing_signed_values():
    class Axis:
        transAxes = None

        def __init__(self):
            self.scale = None
            self.yaxis = SimpleNamespace(get_major_locator=lambda: SimpleNamespace(set_params=lambda **kwargs: None))

        def set_yscale(self, name, **kwargs):
            self.scale = (name, kwargs)

        def text(self, *args, **kwargs):
            pass

        def autoscale(self, **kwargs):
            pass

        def margins(self, **kwargs):
            pass

        def get_ylim(self):
            return -1., 1.

        def get_yticks(self):
            return []

    values = np.array([-1e9, -1., 0., 1., 2., 3.])
    original = values.copy()
    negative, positive = Axis(), Axis()
    report.readable_scale(negative, [values])
    report.readable_scale(positive, [-values])
    assert negative.scale == positive.scale == ("symlog", {"linthresh": 2.})
    np.testing.assert_array_equal(values, original)
    ordinary, empty = Axis(), Axis()
    report.readable_scale(ordinary, [np.array([-3., -1., 0., 1., 2.])])
    report.readable_scale(empty, [None])
    assert ordinary.scale is None and empty.scale is None


def test_display_scale_keeps_small_curve_visible_amid_large_majority():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    small = np.array([1., 2., 3.])
    large = np.array([1000., 2000., 3000.])
    fig, axis = plt.subplots()
    try:
        for values in (small, large, large * 2):
            axis.plot(values)
        originals = [line.get_ydata().copy() for line in axis.lines]
        report.readable_scale(axis, (small, large, large * 2))
        assert axis.get_yscale() == "symlog"
        assert axis.yaxis.get_transform().linthresh == 2.
        assert axis.get_ylim()[0] < small.min()
        assert axis.get_ylim()[1] > (large * 2).max()
        for line, original in zip(axis.lines, originals):
            np.testing.assert_array_equal(line.get_ydata(), original)
    finally:
        plt.close(fig)
