"""Saved-only P5 reduction checks; no simulation, fitting, or forecasting."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def reducer(monkeypatch):
    directory = Path(__file__).resolve().parents[1]/"tools"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("p5_reducer_tested", directory/"summarize_extensions_v03.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gate(passed=True):
    parameter = {"rhat":1. if passed else 1.1, "ess_bulk":1000., "ess_tail":1000., "passed":passed}
    group = {"chains":4, "draws":2000, "parameters":{"x":parameter}}
    return {"passed":passed, "hyperparameters":group, "functionals":deepcopy(group)}


def test_unqualified_ar_has_no_native_loss(reducer):
    scalar = {"variant":"scalar_reference", "qualified":True, "gates":[gate()],
              "posterior_identity":"scalar", "fit_invocation_seconds":1., "forecast_seconds":.1,
              "scores":{"crps":1., "mae":2., "rmse":3.,
                        "intervals":{"0.9":{"coverage":.9, "interval_score":2., "width":1.}}}}
    ar = {"variant":"full_residual_ar", "qualified":False, "gates":[gate(False)],
          "posterior_identity":"ar", "fit_invocation_seconds":2.}
    document = {"complete":True, "rows":[scalar,ar], "evaluation":{},
                "comparisons":[{"variant":"full_residual_ar", "common_qualified_datasets":0,
                                "loss_difference_extension_minus_scalar":None, "mcse_across_datasets":None}]}
    result = reducer.ablation_summary(document, {"settings":{"paths":128}}, "toy")
    assert result["rows"][1]["native_scores"] is None
    assert result["comparisons"][0]["loss_difference_extension_minus_scalar"] is None
    ar["scores"] = scalar["scores"]
    with pytest.raises(ValueError, match="Unqualified native scores"):
        reducer.ablation_summary(document, {"settings":{"paths":128}}, "toy")


def test_path_rule_uses_common_scale_and_does_not_gate_on_descriptive_mcse(reducer, tmp_path):
    receipt = tmp_path/"receipt.json"
    receipt.write_text("{}")
    rows = [{"paths":count, "scores":{"crps":crps, "rmse":1., "intervals":{"0.9":{"coverage":.9}}},
             "predictive_mean_mcse_rms":10., "predictive_mean_mcse_max":20.}
            for count,crps in ((128,.012),(256,.01),(512,0.))]
    result = reducer.path_case({"rows":rows, "posterior_identity":"fixed-fit"},
                               {"training_scale":2.}, receipt, "toy")
    assert [r["passes_both_rules"] for r in result["rows"]] == [False,True,True]
    assert result["rows"][1]["normalized_absolute_crps_difference_from_512"] == .005
    assert result["rows"][1]["conditional_predictive_mean_mcse_rms_over_training_scale"] == 5.


def test_rule_must_precede_both_original_path_invocations(reducer):
    rule = {"candidate_counts":[128,256,512], "reference":512, "rule":"CRPS <=0.005 and coverage <=0.02",
            "declared_at":"2026-11-03T07:00:00+00:00", "cases":["old/g3/A/esn","old/g4/A/esn"]}
    evidence = {"path_count_assessment":{"predeclared_rule":rule, "predeclared_rule_document_sha256":"rule-hash"},
                "resource_receipts":[{"invocations":[
                    {"command":["python","--baseline",f"results/{case}/model.npz","--path-stability"],
                     "started_utc":"2026-11-03T07:01:00+00:00"} for case in rule["cases"]]}]}
    _, starts = reducer.verify_original_rule({"predictive_path_count_rule":rule}, evidence, "rule-hash")
    assert set(starts) == set(rule["cases"])
    rule["declared_at"] = "2026-11-03T07:02:00+00:00"
    with pytest.raises(ValueError, match="did not precede"):
        reducer.verify_original_rule({"predictive_path_count_rule":rule}, evidence, "rule-hash")


def test_aggregation_input_tampering_is_rejected(reducer, tmp_path):
    path = tmp_path/"data.bin"
    path.write_bytes(b"fixed data")
    inputs = reducer.Inputs()
    inputs.bind(path)
    inputs.verify()
    path.write_bytes(b"changed data")
    with pytest.raises(ValueError, match="input changed"):
        inputs.verify()
