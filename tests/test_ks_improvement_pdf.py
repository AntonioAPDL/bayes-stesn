"""Focused PDF presentation checks; synthetic summaries only, no rendering or data."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load_renderer():
    path = Path(__file__).resolve().parents[1] / "tools/build_ks_improvement_pdf.py"
    spec = importlib.util.spec_from_file_location("ks_improvement_pdf_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matched_seed_tables_keep_undefined_pairs_and_all_origin_outcomes():
    pdf = load_renderer()
    estimates = {"rmse": {"mean": 1.25, "ci95": [1., 1.5], "n": 4},
                 "crps": {"mean": .25, "ci95": [.2, .3], "n": 4}}
    main = {"track": "A", "candidate_id": "candidate", "roles": ["main"],
            "seed": 20261105, "stage": "matched_main_seed", "scheduled_trajectories": 4,
            "scheduled_origins": 32, "complete_trajectories": 4, "qualified_fits": 4,
            "finite_successful_origins": 32, "failed_rollout_origins": 0,
            "unqualified_blocked_origins": 0, "failed_job_blocked_origins": 0,
            "pending_origins": 0, "finite_divergent_origins": 0,
            "finite_origins_outside_score_population": 0, "metrics": estimates}
    alternative = {**main, "seed": 20261106, "stage": "sensitivity",
                   "complete_trajectories": 0, "qualified_fits": 1,
                   "finite_successful_origins": 7, "failed_rollout_origins": 1,
                   "unqualified_blocked_origins": 8, "failed_job_blocked_origins": 8,
                   "pending_origins": 8, "finite_divergent_origins": 1,
                   "finite_origins_outside_score_population": 7,
                   "metrics": {m: {"mean": None, "ci95": None, "n": 0} for m in estimates}}
    pairs = [{"track": "A", "left": "candidate", "right": "candidate",
              "left_seed": 20261106, "right_seed": 20261105, "metric": metric,
              "n": 0, "mean": None, "ci95": None, "paired_trajectories": [],
              "scheduled_trajectories": [0, 5, 10, 15], "difference": "left minus right",
              "conditional_on_complete_pairs": True} for metric in estimates]
    renderer = pdf.Renderer.__new__(pdf.Renderer)
    renderer.seed = {"rows": [main, alternative], "paired_differences": pairs}
    renderer.diagnosis = None
    tables, paragraphs = [], []
    renderer.heading = lambda *args, **kwargs: None
    renderer.p = lambda text, *args, **kwargs: paragraphs.append(text)
    renderer.table = lambda headers, rows, *args, **kwargs: tables.append((headers, rows))
    renderer.sensitivity_pages()
    assert len(tables) == 3
    scores, census, contrasts = [rows for _, rows in tables]
    assert scores[0][1] == "20261105\nMatched main"
    assert scores[0][3] == "1.2500 [1.0000, 1.5000]"
    assert scores[1][2:] == ["0/4", "Undefined", "Undefined"]
    assert census[1] == ["New", 20261106, "1/4", "7/32", 1, 8, "8 / 8", 1]
    assert len(contrasts) == 2
    assert [r[2] for r in contrasts] == ["RMSE", "CRPS"]
    assert all(row[3:] == ["0/4", "None", "Undefined"] for row in contrasts)
    assert any("seed 20261106: 7" in text for text in paragraphs)
    assert any("alternative seed minus the matched main seed" in text for text in paragraphs)


def test_applied_migration_disclosure_preserves_archive_only_restart_limitation():
    pdf = load_renderer()
    pair = {"fit_key": "historical_fit", "qualification_changed": False,
            "inference_restart_state_available": False}
    proof = {"fit_key": "historical_fit", "track": "A", "candidate_id": "bayes_02",
             "trajectory": 0, "seed": 20261105, "native_qualified": True,
             "native_posterior_state_counts": [2, 2, 2, 2],
             "comparison_counts": {"array_comparisons": 7},
             "exact_completed_inference_restart_available_from_auxiliary_pair": False}
    migration = {"schema": "ks-improvement-budget-migration/1", "apply": True, "passed": True,
                 "destination_experiment": "experiment", "destination_freeze_identity": "freeze",
                 "created_utc": "synthetic migration time", "receipt_sha256": "synthetic receipt hash",
                 "native_checkpoint_count": 1, "forecast_receipt_states": {"complete": 2},
                 "carried_worker_seconds": 31., "original_freeze_created_utc": "original freeze time",
                 "wall_budget_start": "original campaign time", "reviewed_inference_sidecars": {"fits": [pair]}}
    renderer = pdf.Renderer.__new__(pdf.Renderer)
    renderer.r = {"experiment_identity": "experiment", "freeze_identity": "freeze",
                  "operations": {"migration": migration, "public_evidence": {
                      "native_auxiliary_validation": {"sha256": "synthetic proof hash", "data": proof}}}}
    paragraphs, tables = [], []
    renderer.p = lambda text, *args, **kwargs: paragraphs.append(text)
    renderer.table = lambda headers, rows, *args, **kwargs: tables.append((headers, rows))
    renderer.applied_migration_evidence()
    assert "not final totals" in tables[0][0][0]
    assert any("archive-only evidence outside active fit storage" in text for text in paragraphs)
    assert any("Exact inference restart from this one archived pair is unavailable" in text for text in paragraphs)
    assert any("recorded qualification is unchanged" in text for text in paragraphs)
    assert any("All 7 retained auxiliary state-array comparisons" in text for text in paragraphs)
    migration["destination_freeze_identity"] = "wrong freeze"
    with pytest.raises(ValueError, match="reported destination"):
        renderer.applied_migration_evidence()
