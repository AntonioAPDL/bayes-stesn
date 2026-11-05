"""Data-free regressions for report display and completion metadata.

No artifact files are rendered and no scientific fitting, forecasting or
scoring occurs. The synthetic tail bounds exercise the readable-companion
decision; one tiny in-memory canvas checks crowded cost-label placement.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from bayes_stesn.applications.ks_report import Report, _json_identity, verify_contiguous_diagnosis
from bayes_stesn.applications.ks_runtime import file_hash


def test_large_finite_tail_gets_companion_without_changing_estimates():
    estimates = {
        "bayes_stesn": {"mean": [.02, 1.1], "ci95": [[.01, .9], [.03, 1.4]]},
        "basis_gru": {"mean": [.01, 1.2], "ci95": [[.005, 1.], [.02, 1.5]]},
        "ridge_esn": {"mean": [.01, 1.1], "ci95": [[.005, .9], [.02, 1.4]]},
        "pod_nvar": {"mean": [.005, 43.], "ci95": [[.002, 1.], [.01, 129.]]},
    }
    before = deepcopy(estimates)
    assert Report._extreme_display_models(estimates) == {"pod_nvar"}
    assert estimates == before
    # A catastrophic finite tail still receives the same separate display.
    estimates["pod_nvar"]["mean"][-1] = 1e43
    estimates["pod_nvar"]["ci95"][1][-1] = 3e43
    assert Report._extreme_display_models(estimates) == {"pod_nvar"}


def test_crowded_cost_labels_stay_inside_axes_without_overlap():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(5, 3))
    axis.set_xscale("log"); axis.set_xlim(.7, 4000); axis.set_ylim(0, 14)
    points = [(3., 1.1, "8 (n=20)"), (3.1, 1.15, "9 (n=20)"),
              (350., 1.2, "2 (n=20)"), (370., 1.3, "3 (n=20)"),
              (780., .6, "7 (n=20)")]
    before = (axis.get_xlim(), axis.get_ylim())
    annotations = Report._annotate_compute_points(axis, points)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bounds = axis.get_window_extent(renderer)
    boxes = [item.get_window_extent(renderer) for item in annotations]
    assert (axis.get_xlim(), axis.get_ylim()) == before
    for index, box in enumerate(boxes):
        assert bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1)
        assert not any(box.overlaps(other) for other in boxes[:index])
    plt.close(figure)


def report_fixture():
    report = Report.__new__(Report)
    report.seed = 20261105
    report.frozen = {
        "model_seed": report.seed,
        "selected": {"A": {"bayes_stesn": {}}, "B": {"basis_gru": {}}},
        "robustness_methods": ["bayes_stesn"],
        "test_trajectories": list(range(20)),
        "test_origins": list(range(8000, 10000, 100)),
        "sensitivity_seeds": [20261106, 20261107],
        "sensitivity_trajectories": [0, 5, 10, 15],
    }
    cohorts = [("A", "bayes_stesn", report.seed, 20),
               ("B", "basis_gru", report.seed, 20),
               ("C", "bayes_stesn", report.seed, 20)]
    cohorts += [(track, model, seed, 4)
                for track, model in (("A", "bayes_stesn"), ("B", "basis_gru"))
                for seed in (20261106, 20261107)]
    report.rows = [{"track": track, "model": model, "seed": seed,
                    "scheduled_trajectories": n, "scheduled_origins": 20*n,
                    "failed_origins": 0, "pending_origins": 0,
                    "qualified_fits": 0, "inference_unqualified_fits": n}
                   for track, model, seed, n in cohorts]
    return report


def test_completion_rejects_missing_robustness_or_sensitivity_cohort():
    for missing in (("C", "bayes_stesn", 20261105), ("A", "bayes_stesn", 20261106)):
        report = report_fixture()
        report.rows = [row for row in report.rows
                       if (row["track"], row["model"], row["seed"]) != missing]
        with pytest.raises(ValueError, match="every frozen main, sensitivity and robustness cohort"):
            report._completion()


def test_terminal_status_retains_failed_and_unqualified_extension_rows():
    report = report_fixture()
    main = next(row for row in report.rows if row["track"] == "A" and row["seed"] == report.seed)
    contiguous = next(row for row in report.rows if row["track"] == "C")
    sensitivity = next(row for row in report.rows if row["seed"] == 20261106)
    main["failed_origins"] = 4
    contiguous["failed_origins"] = 2
    status = report._completion()
    assert status["main_execution_terminal"]
    assert status["all_requested_scheduled_origins_terminal"]
    assert status["main_failed_origins"] == 4
    assert status["extension_failed_origins"] == 2
    # Qualification and failure are not silently converted into pending work.
    assert all(row["inference_unqualified_fits"] for row in report.rows)
    contiguous["pending_origins"] = 20
    sensitivity["pending_origins"] = 3
    status = report._completion()
    assert status["main_execution_terminal"]
    assert not status["all_requested_scheduled_origins_terminal"]
    assert status["extension_pending_origins"] == 23
    assert status["extension_failed_origins"] == 2


def test_contiguous_diagnosis_rejects_wrong_freeze_or_changed_source(tmp_path):
    source = tmp_path/"saved_source.txt"
    source.write_text("fixed source\n")
    frozen = {"identity": "frozen", "experiment_identity": "experiment", "selected": {"A": {
        "bayes_stesn": {"id": "bayes_03"}, "basis_gru": {"id": "gru_02"}}}}
    manifest = [{"root": "results", "path": source.name, "bytes": source.stat().st_size,
                 "sha256": file_hash(source)}]
    receipt = {"schema": "ks-contiguous-spatial-representation-diagnosis/1",
               "status": "passed_identity_and_representation_checks",
               "freeze_identity": "frozen", "experiment_identity": "experiment",
               "source_manifest": manifest, "source_manifest_sha256": _json_identity(manifest),
               "rows": [{"track": track, "model": model, "trajectory": 0, "first_target": 8000,
                         "config_id": frozen["selected"]["A"][model]["id"], "checks": {"identity": True}}
                        for track in ("A", "C") for model in ("bayes_stesn", "basis_gru")]}
    assert len(verify_contiguous_diagnosis(receipt, frozen, results=tmp_path, repository=tmp_path)) == 4
    with pytest.raises(ValueError, match="frozen identity"):
        verify_contiguous_diagnosis({**receipt, "freeze_identity": "different"}, frozen,
                                    results=tmp_path, repository=tmp_path)
    source.write_text("changed source\n")
    with pytest.raises(ValueError, match="source artifact changed"):
        verify_contiguous_diagnosis(receipt, frozen, results=tmp_path, repository=tmp_path)
