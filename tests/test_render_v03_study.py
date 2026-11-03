"""Bounded integrity checks for the saved-artifact synthesis, with no fitting."""
from dataclasses import replace
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
    renderer.render_html(sections, out, SOURCE)
    assert (out / "index.html").is_file()
    assert len(list((out / "figures").glob("*.png"))) >= 6
    assert len(list((out / "figures").glob("*.svg"))) >= 6
    assert not list(out.glob("*.pdf"))
    inputs.verify_unchanged()
