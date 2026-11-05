"""Small receipt-only checks; never open the KS dataset or fit a model."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from bayes_stesn.applications.ks_fullfield_report import (
    _receipt_index, cluster_curve, completed_blocks, matched_anchor_comparisons, report,
)
from bayes_stesn.applications.ks_runtime import atomic_json, file_hash


def metric(rmse):
    return {"mse_by_horizon": [rmse**2]*100, "crps_by_horizon": [rmse/2]*100,
            "mae_by_horizon": [.8*rmse]*100, "mae": .8*rmse, "training_scale": 2.,
            "rmse": rmse, "crps": rmse/2, "vpt_physical": 25/rmse, "vpt_right_censored": False,
            "spatial_energy": 1.1*rmse, "spatial_energy_by_horizon": [1.1*rmse]*100,
            "spatial_variogram": .2*rmse, "spatial_variogram_by_horizon": [.2*rmse]*100,
            "intervals": {str(level): {"coverage_by_horizon": [level]*100,
                          "width_by_horizon": [2*rmse]*100} for level in (.5, .8, .9, .95)}}


@pytest.fixture
def run(tmp_path):
    cfg = {"tracks": ["A"], "model_seed": 7, "forecast_seed": 19, "horizon": 100,
           "confirmation": {"trajectories": [1, 2], "origins": [6000, 6100], "paths": 32}}
    selected = {"id": "fixture", "label": "raw_reference", "configuration": {
        "target": "state", "prior": {"kind": "ridge"}, "features": {"units": [128], "lags": [1, 2],
        "leak_patterns": [[.2]], "spectral_radius": .9, "raw_readout_lags": "latest"}}}
    freeze = {"history": 512, "selections": {"A": {"distribution": {"candidate": selected},
              "dynamics": None, "point": None, "reference_id": "fixture"}}}
    atomic_json(tmp_path/"experiment.json", {"identity": "synthetic", "config": cfg})
    atomic_json(tmp_path/"freeze.json", freeze)
    atomic_json(tmp_path/"candidate_registry.json", {"tracks": {"A": {"candidates": [selected]}}})
    atomic_json(tmp_path/"coordinator.json", {"state": "paused", "command": "all"})
    atomic_json(tmp_path/"status.json", {"states": {"complete": 2, "pending": 2}, "stages": {"confirm": {"complete": 2}},
        "worker_seconds": 123, "search_worker_seconds": 45, "resources": {"disk_free_bytes": 9*1024**3}})
    episodes = []
    for trajectory, rmse in ((1, 2.), (2, 4.)):
        for first in cfg["confirmation"]["origins"]:
            atomic_json(tmp_path/"forecasts"/f"fit{trajectory}"/f"{first}.json", {
                "partition": "validation", "phase": "core", "track": "A", "candidate_id": "fixture",
                "trajectory": trajectory, "seed": 7, "first_target": first, "paths": 32, "horizon": 100,
                "forecast_seed": 19, "parameter_seed": None, "history": 512,
                "qualified": True, "state": "complete", "metrics": metric(rmse)})
        atomic_json(tmp_path/"anchors"/f"A_t{trajectory:02d}.json", {"track": "A", "trajectory": trajectory,
            "schedule": "confirmation", "partition": "validation", "history": 512,
            "models": {"persistence": {"origins": [{"first_target": first, "metrics": metric(rmse/2)}
                for first in cfg["confirmation"]["origins"]]}}})
        atomic_json(tmp_path/"fits"/f"fit{trajectory}"/"receipt.json", {"track": "A", "candidate_id": "fixture",
            "trajectory": trajectory, "seed": 7, "phase": "core", "fit_seconds": 30*trajectory})
        episodes.append({"trajectory": trajectory, "rmse": rmse, "crps": rmse/2})
    row = {**selected, "eligible": True, "complete": True, "prior_family": "ridge", "episodes": episodes,
           "complete_episodes": 2, "expected_episodes": 2, "loss": 1.5, "point_loss": 10.,
           "rmse": 3., "early_rmse": 3., "crps": 1.5, "vpt_physical": 9.375}
    atomic_json(tmp_path/"confirmation"/"A.json", {"rows": [row], "schedule": cfg["confirmation"]})
    return tmp_path, cfg, freeze


def test_root_after_average_and_cluster_uncertainty_known_answer():
    mean, lo, hi = cluster_curve([[1., 1.], [9., 9.]], square_root=True)
    np.testing.assert_allclose(mean, np.sqrt(5))
    np.testing.assert_allclose(lo, 1.)
    np.testing.assert_allclose(hi, 3.)
    mean, lo, hi = cluster_curve([[2., 3.]])
    assert lo is None and hi is None
    with pytest.raises(ValueError): cluster_curve([[np.nan]])


def test_pairing_uses_all_origins_and_exact_history(run):
    root, cfg, freeze = run
    index = _receipt_index(root, [])
    comparisons = matched_anchor_comparisons(root, cfg, freeze, index)
    rmse = next(row for row in comparisons if row["metric"] == "rmse")
    assert rmse["candidate_minus_anchor"] == pytest.approx(1.5)
    assert rmse["paired_trajectories"] == rmse["expected_trajectories"] == 2
    assert rmse["ci95_lower"] == pytest.approx(1.) and rmse["ci95_upper"] == pytest.approx(2.)
    path = root/"anchors"/"A_t02.json"
    anchor = json.loads(path.read_text()); anchor["history"] = 1024; atomic_json(path, anchor)
    rmse = next(row for row in matched_anchor_comparisons(root, cfg, freeze, index) if row["metric"] == "rmse")
    assert rmse["paired_trajectories"] == 1 and rmse["expected_trajectories"] == 2
    assert not rmse["full_schedule"] and rmse["ci95_lower"] is None
    (root/"forecasts"/"fit1"/"6100.json").unlink()
    blocks = completed_blocks(_receipt_index(root, []), cfg, "A", "fixture", cfg["confirmation"], 512)
    assert set(blocks) == {2}


def test_report_is_read_only_and_does_not_promote_stage_completion_to_study_completion(run):
    root, _, _ = run
    originals = {path: file_hash(path) for path in root.rglob("*.json")}
    result = report(root, make_figures=False)
    assert not result["complete_campaign"] and result["all_confirmation_rows_complete"]
    assert result["matched_comparison_rows"] == 3
    assert all(file_hash(path) == digest for path, digest in originals.items())
    text = (root/"report"/"report.md").read_text()
    assert "Partial campaign snapshot" in text and "not an untouched test holdout" in text
    assert "No new test results" in text and "coordinate basis remains" in text
    rows = list(csv.DictReader((root/"report"/"candidate_metrics.csv").open()))
    assert len(rows) == 1 and float(rows[0]["mean_recorded_fit_seconds"]) == 45


def test_incomplete_scores_keep_denominators_and_successful_only_columns(run):
    root, _, _ = run
    path = root/"confirmation"/"A.json"; packet = json.loads(path.read_text())
    row = packet["rows"][0]
    row.update(complete=False, eligible=False, complete_episodes=1, pending_origins=2,
               episodes=[row["episodes"][0]], rmse=None, crps=None, loss=None)
    atomic_json(path, packet)
    result = report(root, make_figures=False)
    assert not result["complete_campaign"]
    row = next(csv.DictReader((root/"report"/"candidate_metrics.csv").open()))
    assert row["rmse"] == "" and row["crps"] == ""
    assert row["complete_episodes"] == "1" and row["expected_episodes"] == "2"
    assert float(row["successful_episode_rmse"]) == 2


def test_running_empty_report_has_no_fabricated_score(tmp_path):
    atomic_json(tmp_path/"experiment.json", {"identity": "empty", "config": {"tracks": ["A", "B"]}})
    result = report(tmp_path, make_figures=False)
    assert result["candidate_rows"] == 0 and not result["complete_campaign"]
    assert "Queued jobs imply no measured performance" in (tmp_path/"report"/"report.md").read_text()


def test_horizon_and_cost_figures_render_from_receipts(run):
    pytest.importorskip("matplotlib")
    root, _, _ = run
    sites = np.arange(1, 64, 5)
    angles = 2*np.pi*sites/64
    coordinates = np.column_stack((np.cos(angles), np.sin(angles)))
    input_sites = np.setdiff1d(np.arange(64), sites)[:39]
    input_angles = 2*np.pi*input_sites/64
    truth = np.sin(np.arange(100)[:, None]/20+angles[None, :])
    (root/"examples").mkdir()
    np.savez(root/"examples"/"A_fixture_t01_a6000.npz", truth=truth, mean=truth+.1,
        lower=truth-.2, upper=truth+.4, coordinates=coordinates,
        input_coordinates=np.column_stack((np.cos(input_angles), np.sin(input_angles))),
        first_target=np.array(6000), training_scale=np.array(2.))
    result = report(root, make_figures=True)
    paths = {item["path"] for item in result["artifacts"]}
    assert "report/horizons_A_confirmation.png" in paths
    assert "report/horizons_A_confirmation.pdf" in paths
    assert "report/cost_A_confirmation.png" in paths
    assert "report/example_A_t01_a6000.png" in paths
    assert "report/intervals_A_t01_a6000.pdf" in paths
    rows = list(csv.DictReader((root/"report"/"horizon_metrics.csv").open()))
    first = next(row for row in rows if row["metric"] == "rmse" and row["horizon"] == "1")
    assert float(first["value"]) == pytest.approx(np.sqrt(10))
    assert first["complete_trajectories"] == "2" and first["expected_trajectories"] == "2"


def test_supplementary_metrics_milestones_and_explicit_missingness(run):
    root, _, _ = run
    report(root, make_figures=False)
    rows = {row["metric"]: row for row in csv.DictReader((root/"report"/"supplementary_metrics.csv").open())}
    assert float(rows["mae"]["value"]) == pytest.approx(2.4)
    assert float(rows["nrmse"]["value"]) == pytest.approx(1.5)
    assert float(rows["spatial_energy"]["value"]) == pytest.approx(3.3)
    for level in (50, 80, 90, 95):
        assert float(rows[f"coverage{level}"]["value"]) == pytest.approx(level/100)
        assert float(rows[f"width{level}"]["value"]) == pytest.approx(6.)
    leads = list(csv.DictReader((root/"report"/"milestone_metrics.csv").open()))
    assert {int(r["horizon"]) for r in leads} == {1, 5, 10, 25, 50, 100}
    last = next(r for r in leads if r["metric"] == "rmse" and r["horizon"] == "100")
    assert float(last["physical_time"]) == 25 and float(last["value"]) == pytest.approx(np.sqrt(10))
    path = root/"forecasts"/"fit2"/"6000.json"
    receipt = json.loads(path.read_text()); del receipt["metrics"]["intervals"]["0.95"]
    atomic_json(path, receipt); report(root, make_figures=False)
    rows = {row["metric"]: row for row in csv.DictReader((root/"report"/"supplementary_metrics.csv").open())}
    missing = rows["coverage95"]
    assert missing["value"] == "" and missing["missingness"] == "metric_absent_or_nonfinite"
    assert missing["metric_complete_trajectories"] == "1" and missing["complete_trajectories"] == "2"
    assert float(missing["available_episode_mean"]) == pytest.approx(.95)
