"""Confirmation admission from complete saved evidence, without model fitting."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from bayes_stesn.benchmark.metrics import aggregate_results
from bayes_stesn.benchmark.protocol import BenchmarkProtocol, profile
from bayes_stesn.benchmark.runner import (
    benchmark_environment, jsonable, single_thread_runtime, source_fingerprint,
)
from bayes_stesn.benchmark.studies import verify_confirmation_freeze
from bayes_stesn.io import content_hash


TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "v03_confirmation_assessment_tested", TOOLS / "assess_confirmation_v03.py"
)
assessment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = assessment
sys.path.insert(0, str(TOOLS))
try:
    SPEC.loader.exec_module(assessment)
finally:
    sys.path.remove(str(TOOLS))


def save(path, value, identity_key=None):
    value = jsonable(dict(value))
    if identity_key:
        value[identity_key] = content_hash(
            {key: item for key, item in value.items() if key != identity_key}
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


@pytest.fixture
def saved_pilot(tmp_path):
    """All data/job/artifact receipts are real files consumed by load_run.

    Their bounded synthetic values test admission and integrity, not numerical
    fitting or forecast accuracy. Four qualified pairs still have five terminal
    operational pairs, so a scientific block cannot be confused with missing work.
    """
    def make(*, qualified_pairs=5, source=None, wrong_runtime=False,
             scenario="g1_gaussian_ar"):
        root = tmp_path / "pilot"
        protocol = profile(
            "v03_pilot", initial_fit=120, horizon=8, n_side=5, n_observed=10,
            n_calibration=3, scenarios=(scenario,), tracks=("A",),
            models=("bayes_esn_ridge", "separable_stgp"), replications=5,
        )
        with single_thread_runtime():
            environment = jsonable(benchmark_environment())
        if wrong_runtime:
            environment["numpy"] = "incompatible-test-runtime"
        source = source_fingerprint() if source is None else source
        design = content_hash({"protocol": protocol.to_dict(), "confirmation_freeze": None})
        computation = content_hash({
            "design_identity": design, "source_fingerprint": source,
            "environment": environment, "algorithm_schema": "benchmark-computation-4.0",
        })
        manifest = save(root / "manifest.json", {
            "schema": "benchmark-run-4.0", "stage": "development",
            "protocol": protocol.to_dict(), "protocol_identity": protocol.identity,
            "design_identity": design, "computation_identity": computation,
            "confirmation_freeze": None, "source_fingerprint": source,
            "environment": environment,
        }, "identity")
        receipts = []
        for replication in range(5):
            folder = root / scenario / f"rep_{replication:03d}"
            folder.mkdir(parents=True)
            simulation = folder / "simulation.npz"
            np.savez(simulation, fixture_observation=np.array([float(replication)]))
            data = save(folder / "data_receipt.json", {
                "manifest_identity": manifest["identity"],
                "sha256": assessment.sha256(simulation),
                "observation_hash": f"independent-fixture-{scenario}-{replication}",
            }, "identity")
            for model in protocol.models:
                job = folder / "A" / model
                job.mkdir(parents=True)
                artifact = job / "model.npz"
                np.savez(artifact, fixture_parameter=np.array([0.]))
                qualified = model != "bayes_esn_ridge" or replication < qualified_pairs
                crps = 1. + (replication - 2) * .04 if model == "bayes_esn_ridge" else 1.
                scores = {"rmse": crps + .2, "mae": crps, "crps": crps}
                receipt = save(job / "receipt.json", {
                    "scenario": scenario, "track": "A", "model_id": model,
                    "replication": replication, "manifest_identity": manifest["identity"],
                    "computation_identity": computation, "data_identity": data["identity"],
                    "terminal_scientific_result": True, "completed": True,
                    "qualified": qualified, "fallback": not qualified,
                    "selection_metric": "crps", "operational_scores": scores,
                    "native_scores": scores if qualified else None, "training_scale": 2.,
                    "normalization_reference": {"identity": "common-training-reference"},
                    "origin_index": protocol.final_origin, "target_ids": ["s0"],
                    "fit_seconds": 1., "forecast_seconds": .1, "tuning_seconds": .2,
                    "total_seconds": 1.1, "total_pipeline_seconds": 1.3,
                    "compute_accounting_complete": True,
                    "artifacts": [{"file": "model.npz", "bytes": artifact.stat().st_size,
                                   "sha256": assessment.sha256(artifact)}],
                }, "receipt_identity")
                receipts.append(receipt)
        summary = aggregate_results(receipts, bootstrap_resamples=10)
        summary.update(
            manifest_identity=manifest["identity"], complete=True, source_stable=True,
            final_source_fingerprint=source, jobs=10, scheduled_jobs=10, pending_jobs=0,
            unresolved_infrastructure_failures=0, stage="development",
        )
        save(root / "summary.json", summary)
        draft = replace(protocol, name="fixture_confirmation", stage="confirmation", replications=20)
        config = tmp_path / "confirmation_draft.yaml"
        draft.save(config)
        primary = tmp_path / "primary.json"
        primary.write_text(json.dumps([{
            "reference": "bayes_esn_ridge", "other_model": "separable_stgp",
            "track": "A", "metric": "crps", "scenario_weights": {scenario: 1.},
        }]))
        return {
            "pilot": root, "config": config, "primary": primary,
            "output": tmp_path / "assessment.json", "freeze_output": tmp_path / "freeze.json",
            "resolved_config": tmp_path / "confirmation.yaml",
        }
    return make


def outputs(case):
    return [case[key] for key in ("output", "freeze_output", "resolved_config")]


@pytest.mark.parametrize("qualified_pairs", [4, 5])
def test_actual_freeze_factory_blocks_four_pairs_and_admits_five(saved_pilot, qualified_pairs):
    case = saved_pilot(qualified_pairs=qualified_pairs)
    original = {path: path.read_bytes() for path in case["pilot"].rglob("*") if path.is_file()}
    receipt = assessment.assess(**case)
    assert receipt == json.loads(case["output"].read_text())
    assert receipt["identity"] == content_hash({k: v for k, v in receipt.items() if k != "identity"})
    assert receipt["complete"] is True
    assert receipt["admitted"] is (qualified_pairs == 5)
    assert receipt["macros"][0]["cells"][0]["pairs"] == qualified_pairs
    assert receipt["freeze_attempt"]["called"] is True
    assert receipt["confirmation_outcomes_generated"] is False
    assert receipt["model_fit_calls"] == 0
    assert all(path.read_bytes() == data for path, data in original.items())
    inputs = {
        (Path(row["path"]) if Path(row["path"]).is_absolute()
         else assessment.ROOT / row["path"]).resolve()
        for row in receipt["source_inputs"]
    }
    assert set(original).issubset(inputs)
    assert (TOOLS / "assess_confirmation_v03.py").resolve() in inputs
    assert (TOOLS / "render_v03_study.py").resolve() in inputs
    assert receipt["fhn_prerequisite"] == "not_applicable"
    assert receipt["fhn_prerequisite_verified"] is False
    if qualified_pairs == 4:
        assert receipt["outcome"] == "blocked_insufficient_common_pairs"
        assert receipt["freeze"] is None
        assert receipt["deficient_cells"][0]["pairs"] == 4
        assert receipt["freeze_attempt"]["exception_type"] == "ValueError"
        assert not case["freeze_output"].exists()
        assert not case["resolved_config"].exists()
        # The missing native pair is a completed method outcome, not lost work.
        summary = json.loads((case["pilot"] / "summary.json").read_text())
        assert summary["paired_comparisons"][0]["rmse_population"] == 5
    else:
        assert receipt["outcome"] == "admitted"
        assert receipt["deficient_cells"] == []
        freeze = json.loads(case["freeze_output"].read_text())
        frozen = BenchmarkProtocol.load(case["resolved_config"])
        assert verify_confirmation_freeze(frozen, freeze)["identity"] == receipt["freeze"]["identity"]
        assert 20 <= receipt["freeze"]["R"] <= 60
        assert receipt["freeze"]["sha256"] == assessment.sha256(case["freeze_output"])


def test_admission_keeps_mcse_cap_shortfall_explicit(saved_pilot):
    case = saved_pilot()
    receipt = assessment.assess(**case, target_mcse=1e-6)
    assert receipt["admitted"] is True
    assert receipt["freeze"]["R"] == 60
    assert receipt["freeze"]["power_status"] == "mcse_target_not_met_at_cap"
    assert receipt["confirmation_outcomes_generated"] is False


@pytest.mark.parametrize("fault,match", [
    ("source", "different scientific source"),
    ("runtime", "current-runtime development evidence"),
    ("design", "share scientific dimensions"),
])
def test_wrong_provenance_is_an_error_even_when_pairs_are_deficient(saved_pilot, fault, match):
    case = saved_pilot(qualified_pairs=4, source="b" * 64 if fault == "source" else None,
                       wrong_runtime=fault == "runtime")
    if fault == "design":
        replace(BenchmarkProtocol.load(case["config"]), horizon=9).save(case["config"])
    with pytest.raises(ValueError, match=match):
        assessment.assess(**case)
    assert not any(path.exists() for path in outputs(case))


@pytest.mark.parametrize("fault", ["pending", "process_failure", "missing_receipt", "corrupt_artifact"])
def test_infrastructure_and_incomplete_artifacts_never_become_a_scientific_block(saved_pilot, fault):
    case = saved_pilot(qualified_pairs=4)
    if fault == "pending":
        path = case["pilot"] / "summary.json"
        value = json.loads(path.read_text())
        value.update(complete=False, pending_jobs=1, unresolved_infrastructure_failures=1)
        save(path, value)
    elif fault in {"process_failure", "missing_receipt"}:
        path = next(case["pilot"].glob("*/rep_000/A/*/receipt.json"))
        if fault == "missing_receipt":
            path.unlink()
        else:
            value = json.loads(path.read_text())
            value["process_failure"] = True
            save(path, value, "receipt_identity")
    else:
        next(case["pilot"].glob("*/rep_000/A/*/model.npz")).write_bytes(b"damaged artifact")
    with pytest.raises((ValueError, FileNotFoundError)):
        assessment.assess(**case)
    assert not any(path.exists() for path in outputs(case))


@pytest.mark.parametrize("error", [ValueError("unrelated validation failure"), OSError("disk unavailable")])
def test_arbitrary_factory_errors_propagate_without_a_completed_assessment(saved_pilot, monkeypatch, error):
    case = saved_pilot(qualified_pairs=4)
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(assessment, "freeze_confirmation", fail)
    with pytest.raises(type(error), match=str(error)):
        assessment.assess(**case)
    assert not any(path.exists() for path in outputs(case))


@pytest.mark.parametrize("destination", ["output", "freeze_output", "resolved_config"])
def test_preexisting_evidence_is_never_overwritten(saved_pilot, destination):
    case = saved_pilot(qualified_pairs=4)
    path = case[destination]
    path.write_bytes(b"existing evidence must survive")
    with pytest.raises(ValueError, match="Preserve existing"):
        assessment.assess(**case)
    assert path.read_bytes() == b"existing evidence must survive"
    assert [item for item in outputs(case) if item.exists()] == [path]


@pytest.mark.parametrize("pair", [("output", "freeze_output"), ("output", "resolved_config"),
                                  ("freeze_output", "resolved_config")])
def test_output_paths_must_be_distinct(saved_pilot, pair):
    case = saved_pilot()
    case[pair[1]] = case[pair[0]]
    with pytest.raises(ValueError, match="paths must differ"):
        assessment.assess(**case)
    assert not any(path.exists() for path in outputs(case))


@pytest.mark.parametrize("minimum", [4, True, 5.])
def test_no_lower_or_noninteger_pair_threshold(saved_pilot, minimum):
    case = saved_pilot(qualified_pairs=4)
    with pytest.raises(ValueError, match="At least five"):
        assessment.assess(**case, minimum_pairs=minimum)
    assert not any(path.exists() for path in outputs(case))


@pytest.mark.parametrize("target", [0., float("nan"), float("inf")])
def test_invalid_precision_request_is_not_classified_as_a_pair_block(saved_pilot, target):
    case = saved_pilot(qualified_pairs=4)
    with pytest.raises(ValueError, match="Invalid MCSE bounds"):
        assessment.assess(**case, target_mcse=target)
    assert not any(path.exists() for path in outputs(case))


def test_generator_prerequisite_is_required_before_recording_a_pair_block(saved_pilot):
    case = saved_pilot(qualified_pairs=4, scenario="g3_excitable_field")
    with pytest.raises(ValueError, match="requires time-step validation evidence"):
        assessment.assess(**case)
    assert not any(path.exists() for path in outputs(case))


@pytest.mark.parametrize("destination", ["freeze_output", "resolved_config"])
def test_partial_factory_output_is_not_reported_as_a_clean_scientific_block(saved_pilot, monkeypatch, destination):
    case = saved_pilot(qualified_pairs=4)
    authoritative = assessment.freeze_confirmation
    def partial_then_authoritative(*args, **kwargs):
        case[destination].write_text("incomplete factory output")
        return authoritative(*args, **kwargs)
    monkeypatch.setattr(assessment, "freeze_confirmation", partial_then_authoritative)
    with pytest.raises(RuntimeError, match="blocked gate unexpectedly wrote a freeze"):
        assessment.assess(**case)
    assert not case["output"].exists()


def test_in_use_evidence_change_invalidates_assessment(saved_pilot, monkeypatch):
    case = saved_pilot(qualified_pairs=4)
    authoritative = assessment.freeze_confirmation
    changed = case["pilot"] / "summary.json"
    def change_after_gate(*args, **kwargs):
        try:
            return authoritative(*args, **kwargs)
        finally:
            changed.write_bytes(changed.read_bytes() + b"\n")
    monkeypatch.setattr(assessment, "freeze_confirmation", change_after_gate)
    with pytest.raises(ValueError, match="Input changed during rendering"):
        assessment.assess(**case)
    assert not any(path.exists() for path in outputs(case))
