"""Focused, data-free regressions for the KS report's reliability census.

These fixtures create receipt JSON only. They never load prepared data,
fit a model, execute a rollout, import the PDF stack, or change the numerical
runner. Run with ``python -m pytest -q tests/test_ks_report_reliability.py``.
"""
from __future__ import annotations

import json

import pytest

from bayes_stesn.applications.ks_report import (
    RELIABILITY_COUNTS,
    collect_reliability_census,
    summarize_reliability,
)
from bayes_stesn.applications.ks_runner import forecast_key, forecast_path
from bayes_stesn.applications.ks_runtime import atomic_json, file_hash
from bayes_stesn.io import content_hash


def frozen_fixture(model, trajectories):
    """A synthetic specification with the actual twenty-origin convention."""
    value = {
        "experiment_identity": "synthetic-report-regression",
        "source": {"fixture": "no model code is executed"},
        "model_seed": 20261105,
        "sensitivity_seeds": [20261106, 20261107],
        "sensitivity_trajectories": list(trajectories)[:4],
        "test_trajectories": list(trajectories),
        "test_origins": list(range(8000, 10000, 100)),
        "history": 128,
        "horizon": 100,
        "paths": 256,
        "selected": {
            "A": {model: {"id": "synthetic_00", "config": {"fixture": True}, "selection_eligible": True}},
            "B": {},
        },
        "robustness_methods": [],
    }
    value["identity"] = content_hash(value)
    return value


def test_full_schedule_known_answer_retains_396_4_2_6_and_340_56():
    """Failed trajectories cannot erase their other finite origin outcomes."""
    frozen = frozen_fixture("pod_nvar", range(20))
    failed = {5: [8000, 8200], 11: [8800], 18: [9000]}
    divergent = {0: [9700], 1: [8300]}
    episodes = []
    for trajectory in frozen["test_trajectories"]:
        failure_count = len(failed.get(trajectory, []))
        divergence_count = len(divergent.get(trajectory, []))
        complete = failure_count == 0
        episode = {name: 0 for name in RELIABILITY_COUNTS}
        episode.update(
            track="A", model="pod_nvar", seed=20261105, trajectory=trajectory,
            fit_state="complete", scheduled_origins=20, observed_origins=20,
            finite_successful_origins=20-failure_count,
            finite_divergent_origins=divergence_count,
            failed_rollout_origins=failure_count, failed_origins=failure_count,
            severe_rollout_origins=failure_count+divergence_count,
            failure_or_divergence_origins=failure_count+divergence_count,
            qualified_successful_origins=20-failure_count,
            finite_successful_origins_in_score_population=20 if complete else 0,
            finite_successful_origins_excluded_incomplete_trajectories=0 if complete else 20-failure_count,
            complete=complete, qualified_complete=complete,
            failure_type_counts={"floating_point_or_nonfinite": failure_count} if failure_count else {},
            failed_rollout_first_targets=failed.get(trajectory, []),
            finite_divergent_first_targets=divergent.get(trajectory, []),
        )
        episodes.append(episode)

    census = summarize_reliability(
        episodes, frozen, source_manifest=[],
        episode_results_sha256="0"*64, freeze_sha256="1"*64,
    )
    row = census["rows"][0]
    assert row["scheduled_origins"] == 400
    assert row["finite_successful_origins"] == 396
    assert row["failed_rollout_origins"] == row["failed_origins"] == 4
    assert row["finite_divergent_origins"] == 2
    assert row["severe_rollout_origins"] == row["failure_or_divergence_origins"] == 6
    assert row["pending_origins"] == row["fit_failure_placeholder_origins"] == 0
    assert row["complete_trajectories"] == row["qualified_complete_trajectories"] == 17
    assert row["finite_successful_origins_in_score_population"] == 340
    assert row["finite_successful_origins_excluded_incomplete_trajectories"] == 56
    assert not row["full_schedule_point_score_defined"]
    assert row["full_schedule_point_score_undefined_reason"]
    assert row["full_schedule_probability_score_defined"] is None
    assert row["failure_fraction"]["mean"] == pytest.approx(.01)
    assert row["severe_rollout_fraction"]["mean"] == pytest.approx(.015)
    assert row["failure_fraction"]["n"] == 20
    assert row["failure_fraction"]["ci95"] is not None
    assert census["main_score_population_changed"] is False


@pytest.fixture
def receipt_fixture(tmp_path):
    """Finite, failed-fit, unscheduled and partially observed trajectories."""
    frozen = frozen_fixture("linear_var", [0, 1, 2, 3])
    experiment = {"identity": frozen["experiment_identity"], "config": {"forecast_seed": 42}}
    atomic_json(tmp_path/"freeze.json", frozen)
    atomic_json(tmp_path/"experiment.json", experiment)
    atomic_json(tmp_path/"test_access.json", {
        "freeze_identity": frozen["identity"], "both_track_selection_complete": True,
    })
    episodes = []
    first_successful_receipt = None
    for trajectory in frozen["test_trajectories"]:
        fit_state = {0: "complete", 1: "failed", 2: "pending", 3: "complete"}[trajectory]
        candidate = frozen["selected"]["A"]["linear_var"]
        fit_id = content_hash({"synthetic_fit": trajectory})
        job = {
            "phase": "core", "partition": "test", "track": "A", "model": "linear_var",
            "seed": 20261105, "trajectory": trajectory, "config_id": candidate["id"],
            "config": candidate["config"], "fit_id": fit_id,
            "experiment_identity": experiment["identity"], "source": frozen["source"],
            "origins": frozen["test_origins"], "history": 128, "horizon": 100,
            "paths": 256, "forecast_seed": 42,
        }
        job["job_id"] = content_hash(job)[:24]
        if fit_state != "pending":
            atomic_json(tmp_path/"jobs"/(job["job_id"]+".json"), job)
            atomic_json(tmp_path/"fits"/fit_id/"receipt.json", {
                "identity": fit_id, "state": fit_state, "phase": "core", "track": "A",
                "model": "linear_var", "seed": 20261105, "trajectory": trajectory,
                "config": candidate["config"],
            })
        successful = 20 if trajectory == 0 else (2 if trajectory == 3 else 0)
        for origin in frozen["test_origins"][:successful]:
            receipt = {
                "identity": forecast_key(job, origin), "fit_id": fit_id, "model": "linear_var",
                "track": "A", "phase": "core", "trajectory": trajectory, "seed": 20261105,
                "first_target": origin, "partition": "test", "qualified": True,
                "history": 128, "horizon": 100, "config_id": job["config_id"],
                "training_scale": 1., "scheduled_paths": 0, "state": "complete",
                "scores": {"rmse_by_horizon": [1.]*100, "diverged": False},
            }
            path = forecast_path(tmp_path, job, origin)
            atomic_json(path, receipt)
            if first_successful_receipt is None:
                first_successful_receipt = path
        episodes.append({
            "track": "A", "model": "linear_var", "seed": 20261105, "trajectory": trajectory,
            "successful_origins": successful, "failed_origins": 20 if fit_state == "failed" else 0,
            "pending_origins": 0 if fit_state == "failed" else 20-successful, "qualified": successful > 0,
        })
    atomic_json(tmp_path/"episode_results.json", episodes)

    def collect():
        return collect_reliability_census(
            tmp_path, frozen, experiment, episodes,
            episode_results_sha256=file_hash(tmp_path/"episode_results.json"),
            freeze_sha256=file_hash(tmp_path/"freeze.json"),
        )

    return collect, first_successful_receipt


def test_pending_and_failed_fit_are_distinct_from_observed_rollout_failure(receipt_fixture):
    collect, _ = receipt_fixture
    census = collect()
    row = census["rows"][0]
    assert row["scheduled_origins"] == 80
    assert row["observed_origins"] == row["finite_successful_origins"] == 22
    assert row["failed_rollout_origins"] == row["severe_rollout_origins"] == 0
    assert row["fit_failure_placeholder_origins"] == row["failed_origins"] == 20
    assert row["failure_or_divergence_origins"] == 20
    assert row["pending_origins"] == 38
    assert row["finite_successful_origins_in_score_population"] == 20
    assert row["finite_successful_origins_excluded_incomplete_trajectories"] == 2
    frequency = row["failure_fraction"]
    assert frequency["status"] == "observed_lower_bound_with_pending"
    assert frequency["identified_bounds"] == pytest.approx([20/80, 58/80])
    assert frequency["ci95"] is frequency["between_trajectory_standard_error"] is None
    per_trajectory = {value["trajectory"]: value for value in census["trajectories"]}
    assert per_trajectory[1]["fit_failure_placeholder_first_targets"] == list(range(8000, 10000, 100))
    assert per_trajectory[2]["pending_first_targets"] == list(range(8000, 10000, 100))


def test_changed_origin_receipt_rejects_stale_episode_summary(receipt_fixture):
    collect, path = receipt_fixture
    receipt = json.loads(path.read_text())
    receipt.pop("scores")
    receipt.update(state="failed", error="FloatingPointError('synthetic nonfinite rollout')")
    atomic_json(path, receipt)
    with pytest.raises(ValueError, match="regenerate a coherent summary"):
        collect()
