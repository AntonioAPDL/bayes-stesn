"""Small independent mathematical references; no study datasets or model fits."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest
import yaml

from bayes_stesn.io import content_hash


def tool(relative="tools/summarize_v03_learning_curves.py"):
    path = Path(__file__).resolve().parents[1]/relative
    spec = importlib.util.spec_from_file_location("learning_curve_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_equal_stratum_macro_uses_paired_dataset_variance():
    result = tool().paired_macro({"first": [-1., 1.], "second": [0., 2.]}, seed=5)
    assert_allclose(result["difference"], .5, atol=1e-14)
    assert_allclose(result["mcse"], np.sqrt(.5), atol=1e-14)
    assert result["pair_counts"] == {"first": 2, "second": 2}


def test_geometry_only_ridge_noise_amplification_has_known_identity_solution():
    result = tool().reconstruction_diagnostics(np.eye(2), np.eye(2), ridge=.1)
    expected = (1/(1+2*.1))**2
    assert_allclose(result["target_unit_iid_noise_variance_amplification"], expected, atol=1e-14)
    assert_allclose(result["worst_direction_variance_amplification"], expected, atol=1e-14)
    assert_allclose(result["regularized_gram_condition"], 1., atol=1e-14)


def test_overcomplete_dictionary_is_not_reported_full_column_rank():
    result = tool().matrix_diagnostics(np.array([[1., 0., 1.], [0., 1., 1.]]))
    assert result["rank"] == 2
    assert not result["full_column_rank"]
    assert result["condition_full_column_space"] is None
    assert_allclose(result["condition_nonzero_subspace"], np.sqrt(3.), atol=1e-14)


def test_path_count_transfer_binds_the_completed_prespecified_rule(tmp_path, monkeypatch):
    module = tool("configs/benchmark/v03/studies/prepare.py")
    monkeypatch.setattr(module, "source_fingerprint", lambda: "frozen-package")
    value = {"schema": "p5-extension-findings-1.0", "source_fingerprint": "frozen-package",
             "path_count_assessment": {
                 "prespecification_verified_against_both_first_path_invocation_times": True,
                 "predeclared_rule": {"candidate_counts": [128, 256, 512]},
                 "passes_on_both_cases": {"128": False, "256": False, "512": True},
                 "selected_for_subsequent_studies": 512}}
    value["identity"] = content_hash(value)
    path = tmp_path/"decision.json"
    path.write_text(json.dumps(value))
    assert module.path_decision(path, 512)["identity"] == value["identity"]
    with pytest.raises(ValueError, match="differs from the completed P5"):
        module.path_decision(path, 128)
    value["path_count_assessment"]["prespecification_verified_against_both_first_path_invocation_times"] = False
    value["identity"] = content_hash({k: v for k, v in value.items() if k != "identity"})
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="prospective declaration"):
        module.path_decision(path, 512)


@pytest.mark.parametrize("pair_sufficient,qualifies,cost_complete,cost_ratio,chosen", [
    (True, True, True, 2.5, 2000),
    (False, True, True, 2., 1000),
    (True, False, True, 2., 1000),
    (True, True, False, 2., 1000),
    (True, True, True, 2.5001, 1000),
])
def test_prospective_history_gate_requires_pairs_qualification_and_complete_affordable_cost(
        pair_sufficient, qualifies, cost_complete, cost_ratio, chosen):
    module = tool()
    contrasts = [{"track": track, "is_decision_contrast": True, "sufficient_pairs": pair_sufficient,
                  "bound_passed": True} for track in ("A", "B")]
    result = module.apply_history_gate(contrasts, [{"not_worse": qualifies}],
        {1000: 10., 2000: 10.*cost_ratio}, cost_complete,
        {"tracks": ["A", "B"], "maximum_measured_long_to_short_pipeline_time_ratio": 2.5})
    assert result["selected_initial_history"] == chosen
    assert result["promotion_gate_passed"] == (chosen == 2000)
    assert bool(result["retention_reasons"]) == (chosen == 1000)


def test_history_gate_requires_the_improvement_bound_on_both_tracks():
    result = tool().apply_history_gate([
        {"track": "A", "is_decision_contrast": True, "sufficient_pairs": True, "bound_passed": True},
        {"track": "B", "is_decision_contrast": True, "sufficient_pairs": True, "bound_passed": False},
    ], [{"not_worse": True}], {1000: 10., 2000: 20.}, True,
       {"tracks": ["A", "B"], "maximum_measured_long_to_short_pipeline_time_ratio": 2.5})
    assert result["selected_initial_history"] == 1000
    assert not result["score_bounds_passed"]


@pytest.mark.parametrize("kind,execute_values", [("history", [1000, 2000]), ("sensors", [25, 50, 100])])
def test_resolved_complete_curve_receipt_matches_the_public_api_without_fits(tmp_path, monkeypatch, kind, execute_values):
    from bayes_stesn.benchmark import studies
    module = tool("configs/benchmark/v03/studies/prepare.py")
    recipe = yaml.safe_load(module.RECIPE.read_text())
    frozen, _, _ = module.selected_specifications(recipe,
        module.ROOT/"configs/benchmark/v03/reference_reliability.yaml")
    profiles = module.profiles_from_recipe(recipe, frozen)
    calls = []
    monkeypatch.setattr(studies, "run_benchmark", lambda p, output, **kwargs: calls.append(p.identity))
    studies.run_curve(profiles[kind+"_base"], tmp_path, kind=kind, values=recipe[kind]["values"],
                      execute_values=execute_values, common_endpoint=4000 if kind == "history" else None)
    assert json.loads((tmp_path/"curve_design.json").read_text()) == module.curve_design(profiles, recipe, kind)
    assert calls == [profiles[f"{kind}_{value}"].identity for value in execute_values]


def test_complete_summary_cannot_bypass_core_resume_integrity_checks(tmp_path, monkeypatch):
    from types import SimpleNamespace
    module = tool("configs/benchmark/v03/studies/run_chunks.py")
    output = tmp_path/"campaign"
    output.mkdir()
    (output/"summary.json").write_text(json.dumps({"complete": True, "jobs": 55}))
    monkeypatch.setattr(module, "source_fingerprint", lambda: "current-source")
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=20*1024**3))
    calls = []

    def rejected_core_resume(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(module.subprocess, "run", rejected_core_resume)
    assert module.main(["--output", str(output), "--max-invocations", "1"]) == 7
    assert len(calls) == 1
    assert calls[0][2:5] == ["bayes_stesn", "benchmark", "resume"]
