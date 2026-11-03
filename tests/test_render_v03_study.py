"""Bounded integrity checks for the saved-artifact synthesis, with no fitting."""
from dataclasses import replace
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from bayes_stesn.benchmark.metrics import aggregate_results
from bayes_stesn.benchmark.protocol import profile
from bayes_stesn.benchmark.runner import jsonable
from bayes_stesn.io import content_hash, save_state


SPEC = importlib.util.spec_from_file_location("v03_renderer_tested", Path(__file__).resolve().parents[1] / "tools/render_v03_study.py")
renderer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = renderer
SPEC.loader.exec_module(renderer)
SOURCE = "a" * 64


def save(path, value, identity_key=None):
    value = jsonable(dict(value))
    if identity_key:
        value[identity_key] = content_hash({k: v for k, v in value.items() if k != identity_key})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


@pytest.fixture
def completed_run(tmp_path):
    root = tmp_path / "scientific_run"
    protocol = replace(profile("v03_pilot"), scenarios=("g1_gaussian_ar",), tracks=("A",),
                       models=("bayes_esn_ridge", "separable_stgp"), replications=2)
    design = content_hash({"protocol": protocol.to_dict(), "confirmation_freeze": None})
    computation = content_hash({"design_identity": design, "source_fingerprint": SOURCE,
        "environment": {}, "algorithm_schema": "benchmark-computation-4.0"})
    manifest = save(root / "manifest.json", {"schema": "benchmark-run-4.0", "protocol": protocol.to_dict(),
        "protocol_identity": protocol.identity, "design_identity": design, "computation_identity": computation,
        "confirmation_freeze": None, "source_fingerprint": SOURCE, "environment": {}, "stage": "development"}, "identity")
    receipts = []
    for replication in range(2):
        folder = root / "g1_gaussian_ar" / f"rep_{replication:03d}"
        folder.mkdir(parents=True)
        simulation = folder / "simulation.npz"
        simulation.write_bytes(b"synthetic checksum-only test fixture")
        data = save(folder / "data_receipt.json", {"manifest_identity": manifest["identity"],
            "sha256": renderer.sha256(simulation), "observation_hash": f"observation-{replication}"}, "identity")
        for j, model in enumerate(protocol.models):
            job = folder / "A" / model
            job.mkdir(parents=True)
            artifact = job / "model.npz"
            artifact.write_bytes(b"synthetic artifact")
            qualified = not (model == "bayes_esn_ridge" and replication == 1)
            scores = {"rmse": 1. + j + replication, "mae": .5 + j + replication,
                "crps": .6 + j + replication, "rmse_by_lead": [1., 2.],
                "mean_error_by_lead": [0., .1], "crps_by_lead": [.5, .8]}
            receipt = save(job / "receipt.json", {"scenario": "g1_gaussian_ar", "track": "A", "model_id": model,
                "replication": replication, "manifest_identity": manifest["identity"], "computation_identity": computation,
                "data_identity": data["identity"], "terminal_scientific_result": True, "completed": True,
                "qualified": qualified, "fallback": not qualified, "selection_metric": "crps",
                "operational_scores": scores, "native_scores": scores, "training_scale": 2.,
                "normalization_reference": {"identity": "common-training-reference"}, "origin_index": 520,
                "target_ids": ["s0"], "fit_seconds": 1., "forecast_seconds": .1, "tuning_seconds": 2.,
                "total_seconds": 1.2, "total_pipeline_seconds": 3.2, "compute_accounting_complete": True,
                "artifacts": [{"file": "model.npz", "bytes": artifact.stat().st_size, "sha256": renderer.sha256(artifact)}]}, "receipt_identity")
            receipts.append(receipt)
    summary = aggregate_results(receipts, bootstrap_resamples=10)
    summary.update(manifest_identity=manifest["identity"], complete=True, source_stable=True,
        final_source_fingerprint=SOURCE, jobs=4, scheduled_jobs=4, pending_jobs=0,
        unresolved_infrastructure_failures=0, stage="development")
    save(root / "summary.json", summary)
    return root


@pytest.fixture
def coordinates(tmp_path):
    root = tmp_path / "coordinate"
    manifest = save(root / "manifest.json", {"schema": "sampler-coordinate-development-1.0",
        "source_fingerprint": SOURCE, "scenarios": ["g1_gaussian_ar"], "replications": [0], "stage": "development"}, "identity")
    rows = []
    for algorithm in ("legacy", "ridge_total_scale"):
        folder = root / "g1_gaussian_ar/rep_000" / algorithm
        folder.mkdir(parents=True)
        save_state(folder / "chain.npz", {"chain": np.array([0., 1.])})
        save_state(folder / "chain.npz.fit-state.npz", {"phase": "complete", "gates": [],
            "checkpoint": {"sha256": renderer.sha256(folder / "chain.npz")}})
        rows.append(save(folder / "receipt.json", {"scenario": "g1_gaussian_ar", "replication": 0,
            "parameterization": algorithm, "complete": True, "qualified": True,
            "manifest_identity": manifest["identity"], "target_identity": "same-target",
            "data_identity": "same-data", "fit_seed": 100, "diagnostics": {"gates": []},
            "minimum_scientific_ess": 500, "minimum_ess_per_second": 20, "timing_complete": True}))
    save(root / "summary.json", {"manifest_identity": manifest["identity"], "complete": True,
        "test_forecasts_computed": False, "accuracy_claim": False, "rows": rows})
    return root


def test_complete_run_validates_receipts_scalar_means_and_qualified_pair_intersection(completed_run):
    inputs = renderer.Inputs()
    run = renderer.load_run(completed_run, "test", inputs, SOURCE)
    paired = run["summary"]["paired_comparisons"][0]
    assert paired["rmse_population"] == 2 and paired["crps_population"] == 1
    assert paired["rmse"]["difference"] == pytest.approx(-.5)
    assert run["summary"]["table"][0]["qualified"] == 1
    assert len(inputs.files) == 14
    inputs.verify_unchanged()


@pytest.mark.parametrize("fault", ["incomplete", "source", "receipt", "artifact", "data", "scalar_summary", "pair_summary", "invalidated"])
def test_saved_evidence_tampering_is_rejected(completed_run, fault):
    root = completed_run
    if fault in {"incomplete", "scalar_summary", "pair_summary"}:
        path = root / "summary.json"
        value = renderer.read_json(path)
        if fault == "incomplete":
            value["complete"] = False
        elif fault == "scalar_summary":
            value["table"][0]["rmse"] = 9000
        else:
            value["paired_comparisons"][0]["rmse"]["paired_differences"][0] = 9000
        save(path, value)
    elif fault == "source":
        with pytest.raises(ValueError, match="different scientific source"):
            renderer.load_run(root, "test", renderer.Inputs(), "b" * 64)
        return
    elif fault in {"artifact", "data"}:
        path = next(root.glob("*/rep_000/A/*/model.npz")) if fault == "artifact" else next(root.glob("*/rep_000/simulation.npz"))
        path.write_bytes(b"changed")
    elif fault == "invalidated":
        save(root / "invalidation.json", {"reason": "test"})
    else:
        path = next(root.glob("*/rep_000/A/*/receipt.json"))
        value = renderer.read_json(path)
        value["qualified"] = False
        save(path, value)
    with pytest.raises(ValueError):
        renderer.load_run(root, "test", renderer.Inputs(), SOURCE)


def test_artifact_paths_cannot_escape_or_follow_symlinks(tmp_path):
    root = tmp_path / "inside"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("outside")
    (root / "alias").symlink_to(outside)
    for value in ("../outside", str(outside), "alias"):
        with pytest.raises(ValueError):
            renderer.safe_member(root, value)


def test_coordinate_ancillary_source_requires_explicit_exact_exception(coordinates):
    with pytest.raises(ValueError, match="allow-coordinate-source"):
        renderer.load_coordinate(coordinates, renderer.Inputs(), "b" * 64)
    value = renderer.load_coordinate(coordinates, renderer.Inputs(), "b" * 64, SOURCE)
    assert value["ancillary_source_exception"] is True


def test_coordinate_checkpoint_hash_and_common_target_are_checked(coordinates):
    checkpoint = coordinates / "g1_gaussian_ar/rep_000/legacy/chain.npz"
    save_state(checkpoint, {"chain": np.array([3., 4.])})
    with pytest.raises(ValueError, match="hash mismatch"):
        renderer.load_coordinate(coordinates, renderer.Inputs(), SOURCE)


def test_coordinate_summary_must_match_its_receipt(coordinates):
    path = coordinates / "summary.json"
    value = renderer.read_json(path)
    value["rows"][0]["qualified"] = False
    save(path, value)
    with pytest.raises(ValueError, match="differs from saved"):
        renderer.load_coordinate(coordinates, renderer.Inputs(), SOURCE)


def test_optional_missing_studies_remain_explicit_and_bad_existing_evidence_fails(tmp_path):
    assert renderer.load_curve(None, renderer.Inputs(), SOURCE)["status"] == "not supplied"
    assert renderer.load_curve(tmp_path / "missing", renderer.Inputs(), SOURCE)["status"] == "not available"
    assert renderer.load_extensions(tmp_path / "missing.json", renderer.Inputs(), SOURCE)["status"] == "not available"
    path = tmp_path / "bad.json"
    path.write_text('{"extensions": [{"name": "RHS"}]}')
    with pytest.raises(ValueError, match="needs name, status and rationale"):
        renderer.load_extensions(path, renderer.Inputs(), SOURCE)


def test_scope_only_extensions_are_never_presented_as_ablation_results(tmp_path):
    path = tmp_path / "scope.json"
    save(path, {"scientific_results_in_this_file": False,
        "extensions": [{"id": "rhs", "default_disposition": "deferred", "trigger": "Needs explicit prior elicitation"}]})
    result = renderer.load_extensions(path, renderer.Inputs(), SOURCE)
    assert result["status"] == "implementation scope; no experiment results"
    assert result["entries"][0]["status"] == "deferred"


def test_scientific_audits_and_plans_are_not_mistaken_for_release_verification(tmp_path, monkeypatch):
    monkeypatch.setattr(renderer, "ROOT", tmp_path)
    folder = tmp_path / "docs/evidence/v03"
    save(folder / "reference_release_audit.json", {"schema": "v03-reference-audit-1.0", "status": "passed"})
    save(folder / "tuned_release_plan.json", {"memory_admissible": True})
    save(folder / "implementation_remote_ci.json", {"schema": "github-actions-verification-1.0", "commit": "fixed-head",
        "jobs": [{"status": "completed", "conclusion": "success"}]})
    evidence = renderer.load_engineering(renderer.Inputs())
    assert [(row["file"], row["kind"]) for row in evidence["release"]] == [
        ("implementation_remote_ci.json", "saved remote CI")]


def test_final_extension_disposition_supersedes_scope_independently_of_argument_order():
    scope = {"document": {"scientific_results_in_this_file": False},
             "entries": [{"id": "full_residual_ar", "status": "candidate", "rationale": "Training residual trigger"}]}
    final = {"document": {"schema": "p5-extension-findings-1.0"},
             "entries": [{"id": "full_residual_ar", "status": "not_promoted", "rationale": "One failed gate; no improvement in the other case"}]}
    for studies in ([scope, final], [final, scope]):
        assert renderer.current_extension_entries(studies) == final["entries"]
    conflicting = {**final, "entries": [{**final["entries"][0], "status": "promoted"}]}
    with pytest.raises(ValueError, match="Conflicting extension dispositions"):
        renderer.current_extension_entries([final, conflicting])
    overlay = {"document": {"schema": "v03-release-extension-dispositions-1.0"},
               "entries": [{"id": "full_residual_ar", "status": "not_promoted", "rationale": "Later completed evidence retains the decision"}]}
    for studies in ([scope, final, overlay], [overlay, final, scope]):
        assert renderer.current_extension_entries(studies) == overlay["entries"]


def test_release_disposition_overlay_binds_immutable_base_and_later_evidence(tmp_path):
    base_path, later_path = tmp_path / "p5_findings.json", tmp_path / "later_calibration.json"
    base = save(base_path, {"schema": "p5-extension-findings-1.0", "source_fingerprint": SOURCE,
                          "extensions": []}, "identity")
    save(later_path, {"complete": True, "raw_coverage": .8, "calibrated_coverage": .9}, "identity")
    value = {"schema": "v03-release-extension-dispositions-1.0", "source_fingerprint": SOURCE,
        "base_findings": {"path": str(base_path), "sha256": renderer.sha256(base_path), "identity": base["identity"]},
        "source_inputs": [{"path": str(p), "sha256": renderer.sha256(p), "bytes": p.stat().st_size} for p in (base_path, later_path)],
        "extensions": [{"id": "calibration", "disposition": "completed", "rationale": "Actual separately measured calibration"}]}
    path = tmp_path / "release_dispositions.json"
    save(path, value, "identity")
    assert renderer.load_extensions(path, renderer.Inputs(), SOURCE)["entries"][0]["status"] == "completed"
    assert renderer.read_json(base_path) == base
    save(later_path, {"complete": True, "raw_coverage": .8, "calibrated_coverage": .95}, "identity")
    with pytest.raises(ValueError, match="hash mismatch"):
        renderer.load_extensions(path, renderer.Inputs(), SOURCE)


def test_extension_findings_verify_inventory_contents_not_only_inventory_identity(tmp_path):
    artifact = tmp_path / "saved_result.json"
    save(artifact, {"qualified": False, "native_scores": None})
    inventory = save(tmp_path / "output_inventory.json", {"complete": True, "source_fingerprint": SOURCE,
        "mode": "inspect_only", "entries": [{"file": artifact.name, "bytes": artifact.stat().st_size,
        "sha256": renderer.sha256(artifact)}]}, "identity")
    path = tmp_path / "findings.json"
    save(path, {"source_fingerprint": SOURCE,
        "extensions": [{"id": "ar", "disposition": "not_promoted", "rationale": "Failed gate"}],
        "evidence_inventories": [{"path": str(tmp_path / "output_inventory.json"),
            "sha256": renderer.sha256(tmp_path / "output_inventory.json"), "identity": inventory["identity"]}]}, "identity")
    assert renderer.load_extensions(path, renderer.Inputs(), SOURCE)["entries"][0]["status"] == "not_promoted"
    save(artifact, {"qualified": True, "native_scores": {"rmse": 0.}})
    with pytest.raises(ValueError, match="hash mismatch"):
        renderer.load_extensions(path, renderer.Inputs(), SOURCE)


def test_extension_source_replay_preserves_the_independent_dataset_count(tmp_path):
    previous_path = tmp_path / "historical.json"
    previous = save(previous_path, {"source_fingerprint": "old-source", "extensions": []}, "identity")
    value = {"source_fingerprint": SOURCE, "extensions": [], "replay_provenance": {
        "new_independent_datasets": 0, "same_datasets_and_seeds": True,
        "new_model_selection_rules": False, "historical_artifacts_modified": False,
        "historical_findings_path": str(previous_path), "historical_findings_sha256": renderer.sha256(previous_path),
        "historical_findings_identity": previous["identity"], "historical_source_fingerprint": "old-source"}}
    path = tmp_path / "current.json"
    save(path, value, "identity")
    assert renderer.load_extensions(path, renderer.Inputs(), SOURCE)["document"]["replay_provenance"]["new_independent_datasets"] == 0
    value["replay_provenance"]["new_independent_datasets"] = 2
    save(path, value, "identity")
    with pytest.raises(ValueError, match="replication population"):
        renderer.load_extensions(path, renderer.Inputs(), SOURCE)


def test_path_budget_decision_checks_saved_scale_loss_and_predeclared_rule(completed_run, tmp_path):
    run = renderer.load_run(completed_run, "test", renderer.Inputs(), SOURCE)
    reference = completed_run / "g1_gaussian_ar/rep_000/A/bayes_esn_ridge/receipt.json"
    rule = {"candidate_counts": [128, 256, 512], "reference": 512, "cases": ["fixed_case"]}
    rule_path = tmp_path / "predeclared.json"
    save(rule_path, {"predictive_path_count_rule": rule})
    originals, rows = [], []
    for n, crps, coverage, rms in ((128, 1.1, .85, .2), (256, 1.02, .9, .15), (512, 1., .9, .1)):
        originals.append({"paths": n, "scores": {"rmse": 1., "crps": crps, "intervals": {"0.9": {"coverage": coverage}}},
                          "predictive_mean_mcse_rms": rms, "predictive_mean_mcse_max": 2 * rms})
        rows.append({"paths": n, "rmse": 1., "crps": crps, "coverage90": coverage,
            "normalized_absolute_crps_difference_from_512": abs(crps - 1.) / 2.,
            "absolute_coverage90_difference_from_512": abs(coverage - .9),
            "conditional_predictive_mean_mcse_rms": rms, "conditional_predictive_mean_mcse_rms_over_training_scale": rms / 2.,
            "conditional_predictive_mean_mcse_max": 2 * rms, "conditional_predictive_mean_mcse_max_over_training_scale": rms,
            "passes_both_rules": n == 512})
    paths = [{"posterior_identity": "fixed-posterior", "evaluation": {"scenario": "g1_gaussian_ar"}, "rows": originals}]
    assessment = {"predeclared_rule_path": str(rule_path), "predeclared_rule_document_sha256": renderer.sha256(rule_path),
        "predeclared_rule": rule, "selected_for_subsequent_studies": 512,
        "passes_on_both_cases": {"128": False, "256": False, "512": True}, "cases": [{
            "reference_receipt_path": str(reference), "reference_receipt_sha256": renderer.sha256(reference),
            "scenario": "g1_gaussian_ar", "posterior_identity": "fixed-posterior", "training_scale": 2., "rows": rows}]}
    renderer.validate_path_assessment(assessment, paths, [run], renderer.Inputs())
    rows[1]["normalized_absolute_crps_difference_from_512"] = .001
    with pytest.raises(ValueError, match="Path decision normalized CRPS"):
        renderer.validate_path_assessment(assessment, paths, [run], renderer.Inputs())


def test_curve_representation_audit_binds_source_inventory_and_fit_qualification(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    monkeypatch.setattr(renderer, "ROOT", root)
    tool = root / "tools/summarize_v03_learning_curves.py"
    recipe = root / "configs/benchmark/v03/studies/remaining_studies.yaml"
    design_path = root / "results/curve_design.json"
    for path in (tool, recipe, design_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source inventory fixture")
    files, runs, rows = [tool, recipe, design_path], [], []
    for count in (25, 50, 100):
        directory = root / "results" / str(count)
        base = directory / "g4_existing_fronts/rep_000"
        receipt = {"scenario": "g4_existing_fronts", "replication": 0, "model_id": "bayes_esn_ridge",
                   "track": "A", "receipt_identity": f"receipt-{count}", "qualified": True, "fallback": False}
        for path, value in ((directory / "manifest.json", {}), (directory / "summary.json", {}),
                            (base / "data_receipt.json", {}), (base / "A/bayes_esn_ridge/receipt.json", receipt)):
            save(path, value)
            files.append(path)
        runs.append({"value": count, "path": str(directory), "receipts": [receipt], "manifest": {"protocol": {"tracks": ["A"]}}})
        rows.append({"sensor_count": count, "track": "A", "receipt_identity": receipt["receipt_identity"],
                     "qualified": True, "fallback": False, "target_ids": ["u1", "u2"],
                     "fixed_reference_noise_amplification": {"target_unit_iid_noise_variance_amplification": [.25, .75],
                         "mean_unit_iid_noise_variance_amplification": .5, "maximum_unit_iid_noise_variance_amplification": .75}})
    curve = {"path": str(design_path), "design": {"identity": "curve-identity", "kind": "sensors"}, "runs": runs}
    value = {"schema": "v03-sensor-representation-audit-1.0", "source_fingerprint": SOURCE,
        "curve_identity": "curve-identity", "design_sha256": renderer.sha256(design_path),
        "tool_sha256": renderer.sha256(tool), "recipe_sha256": renderer.sha256(recipe),
        "source_inputs": [{"path": str(p), "sha256": renderer.sha256(p), "bytes": p.stat().st_size} for p in files],
        "independent_fields": 1, "replication_mcse": None, "rows": rows}
    audit_path = root / "audit.json"
    save(audit_path, value, "identity")
    assert renderer.load_curve_audits([audit_path], [curve], renderer.Inputs(), SOURCE)[0]["status"] == "complete"
    rows[0]["qualified"] = False
    save(audit_path, value, "identity")
    with pytest.raises(ValueError, match="saved fit/qualification"):
        renderer.load_curve_audits([audit_path], [curve], renderer.Inputs(), SOURCE)
    rows[0]["qualified"] = True
    save(audit_path, value, "identity")
    files[-1].write_text("modified scientific receipt")
    with pytest.raises(ValueError, match="hash mismatch"):
        renderer.load_curve_audits([audit_path], [curve], renderer.Inputs(), SOURCE)


def test_history_choice_uses_actual_paired_saved_losses_and_preserves_the_default(completed_run):
    run = renderer.load_run(completed_run, "test", renderer.Inputs(), SOURCE)
    value = {"confirmation_outcomes_inspected": False, "contrasts": [{"metric": "rmse", "track": "A",
        "score_family": "operational_scores", "is_decision_contrast": False, "sufficient_pairs": True,
        "model_id": "bayes_esn_ridge", "cells": [{"scenario": "g1_gaussian_ar", "usable_replications": [0, 1],
            "qualified1000": 1, "qualified2000": 1, "paired_differences": [0., 0.]}],
        "macro": {"paired_differences": {"g1_gaussian_ar": [0., 0.]}, "lower_one_sided95": 0.},
        "margin": None, "bound_passed": None}, {"metric": "crps", "track": "A",
        "score_family": "native_scores", "is_decision_contrast": True, "sufficient_pairs": False,
        "model_id": "bayes_esn_ridge", "cells": [{"scenario": "g1_gaussian_ar", "usable_replications": [0],
            "qualified1000": 1, "qualified2000": 1, "paired_differences": [0.]}],
        "macro": None, "margin": .02, "bound_passed": False}],
        "qualification_cells": [{"model_id": "bayes_esn_ridge", "track": "A", "scenario": "g1_gaussian_ar", "not_worse": True}],
        "summed_pipeline_seconds": {"1000": 12.8, "2000": 12.8}, "short_history_time_saving_fraction": 0.,
        "long_to_short_pipeline_time_ratio": 1., "promotion_gate_passed": False,
        "complete_compute_accounting": True, "cost_gate_passed": True, "score_bounds_passed": False,
        "qualification_not_worse": True, "selected_initial_history": 1000,
        "decision_rule": {"maximum_measured_long_to_short_pipeline_time_ratio": 2.5, "tracks": ["A"],
                          "minimum_common_qualified_pairs_per_native_cell": 3,
                          "decision_model": "bayes_esn_ridge", "minimum_normalized_crps_improvement": .02}}
    renderer._validate_history_choice(value, {1000: run, 2000: run})
    value["contrasts"][0]["macro"]["paired_differences"]["g1_gaussian_ar"][0] = -.3
    with pytest.raises(ValueError, match="paired losses differ"):
        renderer._validate_history_choice(value, {1000: run, 2000: run})


@pytest.fixture
def stress_effect_case():
    spec = importlib.util.spec_from_file_location("stress_producer_contract", renderer.ROOT / "tools/summarize_v03_stress.py")
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    pairs, originals = [], []
    for replication, scale in enumerate((1., 2., 4., 8., 16.)):
        control = {"replication": replication, "model_id": "bayes_esn_ridge", "track": "A",
            "qualified": replication != 4, "fallback": replication == 4, "receipt_identity": f"control-{replication}",
            "training_scale": scale, "normalization_reference": {"identity": f"scale-{replication}"},
            "native_scores": {"rmse": 1., "mae": .5, "crps": 2., "field_energy": 3.,
                              "intervals": {"0.9": {"coverage": .9, "width": 2., "whole_path_covered": True}}},
            "operational_scores": {"rmse": 1., "mae": .5}}
        stress = {**control, "receipt_identity": f"stress-{replication}", "training_scale": 1000000. + replication,
            "qualified": replication != 2, "fallback": replication == 2,
            "native_scores": {"rmse": 1. + scale, "mae": .5 + 2 * scale, "crps": 2. + .3 * scale,
                              "field_energy": 3. + .5 * scale,
                              "intervals": {"0.9": {"coverage": .8, "width": 2. + scale, "whole_path_covered": False}}},
            "operational_scores": {"rmse": 1. + scale, "mae": .5 + 2 * scale}}
        pairs.append(producer.pair_record(control, stress, scale=scale, normalization=control["normalization_reference"], arm="local_t"))
        originals.append((control, stress))
    recipe = {"inference": {"bootstrap_seed": 17, "bootstrap_resamples": 50}}
    arm = {"id": "local_t", "paired_jobs": pairs, "effects": producer.effects_from_pairs(pairs, "local_t", recipe)}
    return arm, recipe, originals


def test_stress_effects_verify_common_scale_exact_population_and_dimensionless_coverage(stress_effect_case):
    arm, recipe, originals = stress_effect_case
    renderer.validate_stress_effects(arm, recipe)
    for pair, (control, stress) in zip(arm["paired_jobs"], originals):
        renderer._validate_stress_pair(pair, control, stress, arm["id"])
    effects = {(row["population"], row["metric"]): row for row in arm["effects"]}
    rmse = effects["operational_scores", "rmse"]
    assert rmse["raw"]["mean"] == pytest.approx(31 / 5)
    assert rmse["common_normalized"]["mean"] == 1.
    assert rmse["common_normalized"]["mcse"] == 0.
    native = effects["native_scores", "crps"]
    assert native["N_pair"] == 3
    assert native["pair_ids"] == [f"rep_{n:03d}/A/bayes_esn_ridge" for n in (0, 1, 3)]
    assert native["common_normalized"]["mean"] == pytest.approx(.3)
    coverage = effects["native_scores", "intervals/0.9/coverage"]
    assert coverage["normalization_power"] == 0
    assert coverage["common_normalized"]["mean"] == pytest.approx(-.1)


@pytest.mark.parametrize("fault", ["standalone_scale", "population", "interval", "coverage_units", "missing_companion"])
def test_stress_report_rejects_incorrect_paired_numbers(stress_effect_case, fault):
    arm, recipe, _ = stress_effect_case
    effect = next(row for row in arm["effects"] if row["population"] == "native_scores" and row["metric"] == "crps")
    if fault == "standalone_scale":
        effect["normalized_differences"][0] = effect["raw_differences"][0] / 1000000.
    elif fault == "population":
        effect["N_pair"] = 5
    elif fault == "interval":
        effect["common_normalized"]["pointwise95"][1] += .1
    elif fault == "coverage_units":
        next(row for row in arm["effects"] if row["metric"] == "intervals/0.9/coverage")["normalization_power"] = 1
    else:
        arm["effects"] = [row for row in arm["effects"] if row["metric"] != "field_energy"]
    with pytest.raises(ValueError):
        renderer.validate_stress_effects(arm, recipe)


def test_stress_audit_rejects_resealed_source_input_tampering_before_reading_science(completed_run, tmp_path):
    run = renderer.load_run(completed_run, "control", renderer.Inputs(), SOURCE)
    bound = tmp_path / "stress_design.json"
    bound.write_text("sealed input")
    path = tmp_path / "audit.json"
    value = {"schema": "v03-paired-stress-summary-1.0", "complete": True, "source_fingerprint": SOURCE,
        "standalone_g5_normalization_is_cross_arm_comparable": False,
        "control_manifest_identity": run["manifest"]["identity"],
        "source_inputs": [{"path": str(bound), "sha256": renderer.sha256(bound), "bytes": bound.stat().st_size}]}
    save(path, value, "identity")
    bound.write_text("edited input")
    with pytest.raises(ValueError, match="hash mismatch"):
        renderer.load_stress_audits([path], [run], renderer.Inputs(), SOURCE)
    value["control_manifest_identity"] = "different control"
    save(path, value, "identity")
    with pytest.raises(ValueError, match="supplied completed control"):
        renderer.load_stress_audits([path], [run], renderer.Inputs(), SOURCE)


def test_saved_first_step_diagnostic_requires_bound_inputs_and_valid_mse_decomposition(completed_run, tmp_path):
    run = renderer.load_run(completed_run, "test", renderer.Inputs(), SOURCE)
    model = completed_run / "g1_gaussian_ar/rep_000/A/bayes_esn_ridge/model.npz"
    folder = tmp_path / "path_study"
    manifest_path, inventory_path = folder / "manifest.json", folder / "output_inventory.json"
    manifest = save(manifest_path, {"source_fingerprint": SOURCE, "input_posterior_identity": "posterior",
                                   "baseline_sha256": renderer.sha256(model)}, "identity")
    save(inventory_path, {"source_fingerprint": SOURCE, "complete": True, "mode": "path_stability",
        "manifest_identity": manifest["identity"], "entries": [{"file": "manifest.json",
            "sha256": renderer.sha256(manifest_path), "bytes": manifest_path.stat().st_size}]}, "identity")
    diagnostic = {"schema": "saved-G3-first-step-review-1.0", "source_fingerprint": SOURCE,
        "posterior_identity": "posterior", "input_hashes": {str(p): renderer.sha256(p) for p in (model, manifest_path, inventory_path)},
        "decoder": {"first_lead_unavoidable_outside_span_rmse": 1., "first_lead_within_span_readout_error_rmse64": .5,
                    "first_lead_outside_span_fraction_of_conditional_readout_mse64": .8},
        "one_step": {"conditional_readout_mean64_rmse": np.sqrt(1.25)},
        "saved_path_leads": [{"conditional_empirical_mean_mcse_rms": 2 / np.sqrt(512), "predictive_sd_rms": 2.}]}
    path = tmp_path / "diagnostic.json"
    save(path, diagnostic, "identity")
    assert renderer.load_diagnostics([path], [run], renderer.Inputs(), SOURCE)[0]["status"] == "complete"
    diagnostic["decoder"]["first_lead_outside_span_fraction_of_conditional_readout_mse64"] = .4
    save(path, diagnostic, "identity")
    with pytest.raises(ValueError, match="outside-span error fraction"):
        renderer.load_diagnostics([path], [run], renderer.Inputs(), SOURCE)


@pytest.fixture
def prior_check_case(tmp_path):
    """Small hand-computable prior design, no model fit or posterior use."""
    directory = tmp_path / "final_ar_g4_existing_fronts"
    prepared = {"Phi": np.array([[1.], [2.]]), "Z": np.array([[1.], [2.], [3.]]),
        "reference_Z": np.array([[1.]]), "split": {"fitting_start": 1, "fitting_stop": 3},
        "response_center": .5, "response_scale": 2.,
        "spec": {"prior": {"kind": "ridge", "mean_log_precision": 0., "sd_log_precision": 0.},
            "covariance": {"kind": "matern", "range_prior": "lognormal", "scale_mode": "relative",
                "anisotropy": False, "nonstationary": False, "static_variance": False, "infer_nu": False,
                "log_q_mean": 0., "log_q_sd": 0., "log_ell_mean": -1., "log_ell_sd": .2, "nu": 1.5},
            "features": {}, "spatial_mean": {}}}
    parameters = {"a0": 2., "b0": .5, "log_precision_mean": 0., "log_precision_sd": 0.,
        "log_q_mean": 0., "log_q_sd": 0., "log_ell_mean": -1., "log_ell_sd": .2, "nu": 1.5,
        "rho_eta_mean": 0., "rho_eta_sd": .8}
    for variant, temporal in (("scalar_reference", "iid"), ("full_residual_ar", "full_residual_ar")):
        save_state(directory / variant / "fitted.npz", {"prepared": prepared,
            "sampler": {"a0": 2., "b0": .5, "rho_prior_sd": .8, "temporal": temporal},
            "posterior": {"unused_array": np.arange(10.)}}, kind="fitted_model")
    manifest = save(directory / "manifest.json", {"source_fingerprint": SOURCE, "environment": {}}, "identity")
    save(directory / "output_inventory.json", {"source_fingerprint": SOURCE, "complete": True,
        "manifest_identity": manifest["identity"]}, "identity")
    def sd_summary(variance):
        variance = np.asarray(variance)
        sd = np.sqrt(variance)
        summary = {"minimum": float(sd.min()), "maximum": float(sd.max()),
                   "median": float(np.median(sd)), "rms": float(np.sqrt(variance.mean()))}
        return {"conditioning_standard_deviation_units": summary,
                "original_response_units": {key: 2 * value for key, value in summary.items()}}
    designs = []
    for name, z, variance in (("conditioning_reference", prepared["reference_Z"], np.array([[.5], [2.]])),
            ("observed_history_fitting_design", prepared["Z"][1:3], np.array([[2., 4.5], [8., 18.]]))):
        designs.append({"name": name, "frozen_design_hash": content_hash({"Phi": prepared["Phi"], "Z": z}),
            "sites": 2, "times": len(z), "spatial_columns": 1, "temporal_columns": 1,
            "readout_pointwise_prior_sd": sd_summary(variance), "observation_pointwise_prior_sd": sd_summary(variance + 1)})
    results = {"prior_parameters": parameters, "response_center": .5, "response_scale": 2.,
        "expected_sigma2_standardized": .5, "expected_inverse_lambda": 1., "expected_q": 1.,
        "marginal_coefficient_variance_standardized": .5, "marginal_coefficient_variance_original_units": 2.,
        "predictive_variance_finite": True, "predictive_fourth_moment_finite": False,
        "designs": designs, "stationary_residual_prior_sd": sd_summary(np.array([1., 1.])),
        "ar_absolute_rho_tail_probabilities": {"0.9": .06572796740759036, "0.95": .022037203554479936},
        "ar_absolute_rho_median": .4926789123765065,
        "quadrature": {"passed": True, "orders": [128, 256], "absolute_agreement_tolerance": 1e-8,
            "expected_kernel_max_absolute_difference": 0., "rho_moment_max_absolute_difference": 0.}}
    value = {"schema": "conditional-fixed-design-prior-check-1.0", "source_fingerprint": SOURCE,
        "environment": {}, "tool_sha256": renderer.sha256(renderer.ROOT / "tools/check_conditional_prior_v03.py"),
        **{key: False for key in ("posterior_samples_used", "future_labels_used", "models_refitted", "prior_retuning_performed",
                                 "extension_promotion", "autonomous_prior_rollouts_assessed")},
        "input_hashes": {str(directory / name): renderer.sha256(directory / name) for name in (
            "manifest.json", "output_inventory.json", "scalar_reference/fitted.npz", "full_residual_ar/fitted.npz")},
        "cases": [{"ablation_directory": str(directory), "manifest_identity": manifest["identity"],
            "identical_frozen_design_and_shared_priors_verified": True, "saved_scalar_sampler_temporal": "iid",
            "saved_ar_sampler_temporal": "full_residual_ar", "results": results}]}
    path = tmp_path / "prior.json"
    save(path, value, "identity")
    studies = [{"path": str(directory / "decisions.json"), "scientific_evidence": {"mode": "extensions"},
                "document": {"manifest_identity": manifest["identity"]}}]
    return path, value, studies


def test_prior_assessment_binds_actual_spec_and_design_without_reading_posterior_arrays(prior_check_case, monkeypatch):
    path, value, studies = prior_check_case
    original_load, accessed = np.load, []
    class ArrayAccess:
        def __init__(self, *args, **kwargs):
            self.archive = original_load(*args, **kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.archive.close()
        def __getitem__(self, key):
            accessed.append(key)
            # Sorted state traversal stores the synthetic posterior first.
            assert key != "a000000", "Posterior arrays must not enter a prior assessment"
            return self.archive[key]
    monkeypatch.setattr(renderer.np, "load", ArrayAccess)
    checks = renderer.load_prior_checks([path], studies, renderer.Inputs(), SOURCE)
    assert checks[0]["status"] == "complete"
    assert len(accessed) == 8
    paragraphs = " ".join(text for section in renderer.prior_sections(checks) for text in section["paragraphs"])
    assert "finite prior variance and an infinite fourth moment" in paragraphs
    assert "autonomous ESN prior rollouts" in paragraphs
    assert "negative draws give alternating correlation" in paragraphs


@pytest.mark.parametrize("fault", ["parameters", "moments", "units", "design", "scope", "missing_input"])
def test_prior_assessment_rejects_resealed_wrong_numbers_or_scope(prior_check_case, fault):
    path, value, studies = prior_check_case
    results = value["cases"][0]["results"]
    if fault == "parameters":
        results["prior_parameters"]["b0"] = 9.
    elif fault == "moments":
        results["predictive_fourth_moment_finite"] = True
    elif fault == "units":
        results["designs"][0]["observation_pointwise_prior_sd"]["original_response_units"]["rms"] = 99.
    elif fault == "design":
        results["designs"][0]["frozen_design_hash"] = "changed"
    elif fault == "scope":
        value["autonomous_prior_rollouts_assessed"] = True
    else:
        del value["input_hashes"][next(iter(value["input_hashes"]))]
    save(path, value, "identity")
    with pytest.raises(ValueError):
        renderer.load_prior_checks([path], studies, renderer.Inputs(), SOURCE)


def test_check_inputs_never_fits_generates_forecasts_or_writes_output(completed_run, tmp_path, monkeypatch, capsys):
    from bayes_stesn.benchmark import runner
    def forbidden(*args, **kwargs):
        raise AssertionError("Renderer crossed the saved-artifact boundary")
    monkeypatch.setattr(runner, "source_fingerprint", lambda: SOURCE)
    monkeypatch.setattr(runner, "fit_model", forbidden)
    monkeypatch.setattr(runner, "generate", forbidden)
    monkeypatch.setattr(renderer, "render_pdf", forbidden)
    monkeypatch.setattr(renderer, "plotting", forbidden)
    out = tmp_path / "never_written"
    renderer.main(["--run", f"synthetic={completed_run}", "--output", str(out), "--check-inputs"])
    assert json.loads(capsys.readouterr().out)["valid"] is True
    assert not out.exists()


def test_full_chart_and_html_recipe_from_synthetic_saved_values_without_pdf(completed_run, coordinates, tmp_path, monkeypatch):
    # Exercise every main figure/equation and escaped HTML without authoring a PDF.
    from bayes_stesn.benchmark import runner
    monkeypatch.setattr(runner, "source_fingerprint", lambda: SOURCE)
    args = renderer.parser().parse_args(["--run", f"synthetic={completed_run}", "--coordinate", str(coordinates),
                                         "--output", str(tmp_path / "charts")])
    data, inputs = renderer.collect(args)
    out = args.output
    out.mkdir()
    sections = renderer.sections_for(data, out)
    assert all(figure["minimum_pdf_font_pt"] >= 7.5 for section in sections for figure in section["figures"])
    renderer.render_html(sections, out, SOURCE)
    assert (out / "index.html").is_file()
    html = (out / "index.html").read_text()
    assert "Bayesian spatiotemporal forecasting" in html
    assert "max-width:100%" in html and "overflow-x:auto" in html
    assert len(list((out / "figures").glob("*.png"))) >= 6
    assert len(list((out / "figures").glob("*.svg"))) >= 6
    assert not list(out.glob("*.pdf"))
    inputs.verify_unchanged()


def test_source_bound_oracle_audit_checks_artifact_and_normalized_saved_numbers(completed_run, tmp_path):
    run = renderer.load_run(completed_run, "test", renderer.Inputs(), SOURCE)
    artifact = tmp_path / "oracle.npz"
    artifact.write_bytes(b"saved oracle checksum fixture")
    rows = [{"scenario": "g1_gaussian_ar", "replication": r, "track": "A",
        "moment_artifact": artifact.name, "moment_sha256": renderer.sha256(artifact),
        "scores": {"rmse": 1., "mae": .8, "crps": .5}, "training_scale": 2.,
        "normalized_rmse": .5, "normalized_crps": .25} for r in range(2)]
    path = tmp_path / "audit.json"
    audit = {"schema": "v03-reference-audit-1.0", "status": "passed", "source_fingerprint": SOURCE,
        "audit_source_fingerprint": SOURCE, "manifest_identity": run["manifest"]["identity"],
        "manifest_sha256": renderer.sha256(completed_run / "manifest.json"),
        "summary_sha256": renderer.sha256(completed_run / "summary.json"), "oracle": rows}
    save(path, audit, "identity")
    result = renderer.load_audits([("test", path)], [run], renderer.Inputs(), SOURCE)
    assert len(result[0]["document"]["oracle"]) == 2
    audit["oracle"][0]["normalized_crps"] = .75
    save(path, audit, "identity")
    with pytest.raises(ValueError, match="Oracle normalized score"):
        renderer.load_audits([("test", path)], [run], renderer.Inputs(), SOURCE)


def test_calibration_seal_requires_all_scientific_artifacts(completed_run, tmp_path):
    run = renderer.load_run(completed_run, "test", renderer.Inputs(), SOURCE)
    folder = tmp_path / "calibration"
    save(folder / "calibration_evidence.json", {"schema": "v03-calibration-evidence-1.0", "complete": True,
        "base_manifest_identity": run["manifest"]["identity"], "source_fingerprint": SOURCE,
        "scenario": "g1_gaussian_ar", "replication": 0, "track": "A", "model_id": "bayes_esn_ridge",
        "base_inputs": [], "files": []}, "identity")
    with pytest.raises(ValueError, match="required base/scientific artifacts"):
        renderer.load_calibrations([("test", folder)], [run], renderer.Inputs(), SOURCE)


def test_saved_calibration_number_verification_is_independent_of_forecasting(monkeypatch):
    from bayes_stesn.benchmark.calibration import evaluate_calibration
    from bayes_stesn.benchmark.contracts import Forecast
    from bayes_stesn.calibration import CalibrationMap
    monkeypatch.setitem(sys.modules, "render_v03_study", renderer)
    spec = importlib.util.spec_from_file_location("v03_calibration_sealer_tested", Path(__file__).resolve().parents[1] / "tools/seal_v03_calibration.py")
    sealer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sealer)
    truth = np.array([[0., 1.], [1., 0.]])
    samples = truth[None] + np.array([-.7, -.1, .4, 1.2])[:, None, None]
    forecast = Forecast(samples.mean(axis=0), samples, np.array([[0., 0.], [1., 0.]]), ("a", "b"), {})
    mapping = CalibrationMap(alpha=.1, target="marginal", quantiles=np.ones((2, 2)),
        scores=np.ones((25, 2, 2)), blocks=25, domain_shape=(2, 2), scale_floor=.01, metadata={})
    calibration = {"maps": {"marginal_0.9": mapping}, "coordinates": forecast.coordinates,
                   "site_ids": forecast.site_ids, "training_scale": 1., "transfer": False, "blocks": 25}
    expected = evaluate_calibration(calibration, forecast, truth)
    bands = expected.pop("bands")
    saved = {**calibration, "maps": {"marginal_0.9": asdict(mapping)}}
    sealer.verify_numbers(expected, forecast, truth, saved, bands)
    changed = {**expected, "raw": {**expected["raw"], "crps": 99.}}
    with pytest.raises(ValueError, match="empirical CRPS"):
        sealer.verify_numbers(changed, forecast, truth, saved, bands)
    changed_bands = {key: (lo + 1., hi) for key, (lo, hi) in bands.items()}
    with pytest.raises(ValueError, match="Saved bands"):
        sealer.verify_numbers(expected, forecast, truth, saved, changed_bands)
