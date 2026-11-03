"""Native comparison integrity from tiny saved artifacts, without model fits."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from bayes_stesn.benchmark.contracts import Forecast
from bayes_stesn.benchmark.generators import Simulation
from bayes_stesn.io import content_hash


SPEC = importlib.util.spec_from_file_location(
    "comparison_renderer_tested", Path(__file__).resolve().parents[1] / "tools/render_model_comparison.py")
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)
SCENARIO = "g4_existing_fronts"


def save_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


@pytest.fixture
def saved_run(tmp_path):
    run = tmp_path / "saved_run"
    folder = run / SCENARIO / "rep_000"
    folder.mkdir(parents=True)
    protocol = {"scenarios": [SCENARIO], "tracks": ["B"], "replications": 1,
                "initial_fit": 2, "validation_blocks": 1, "horizon": 2, "training_start": 1}
    manifest = {"protocol": protocol, "stage": "development", "fixture": True}
    manifest["identity"] = content_hash(manifest)
    save_json(run / "manifest.json", manifest)
    observation = np.arange(24, dtype=float).reshape(6, 4) / 10
    sim = Simulation(observation, observation.copy(),
        np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]), np.arange(6, dtype=float),
        np.array([0]), np.array([1]), np.array([2, 3]), np.array([], dtype=int),
        {"scenario": SCENARIO, "fixture": True})
    sim.save(folder / "simulation.npz")
    targets = sim.test_indices
    ids = tuple(sim.site_ids[index] for index in targets)
    truth = observation[4:6, targets]
    for index, model in enumerate(renderer.MODELS):
        directory = folder / "B" / model
        directory.mkdir(parents=True)
        metadata = {"origin_index": 3, "model_id": model, "fixture": True}
        mean = truth + (index + 1) / 10
        Forecast(mean, None, sim.coordinates[targets], ids, metadata).save(directory / "forecast.npz")
        # A distinct operational artifact must never replace the requested native forecast.
        Forecast(np.full_like(mean, -999.), None, sim.coordinates[targets], ids,
                 {"fallback": True}).save(directory / "operational_point.npz")
        scores = {"rmse": float(np.sqrt(np.mean((mean - truth) ** 2))),
                  "rmse_by_lead": np.sqrt(np.mean((mean - truth) ** 2, axis=1)).tolist()}
        save_json(directory / "receipt.json", {"completed": True, "qualified": True, "fallback": False,
            "model_id": model, "scenario": SCENARIO, "replication": 0, "track": "B",
            "manifest_identity": manifest["identity"], "origin_index": 4, "target_ids": ids,
            "forecast_metadata": metadata, "native_scores": scores, "operational_scores": {"rmse": 999.}})
    return run


def change_first_receipt(run, **changes):
    path = run / SCENARIO / "rep_000/B" / renderer.MODELS[0] / "receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt.update(changes)
    save_json(path, receipt)
    return path


def test_unqualified_null_scores_remain_native_and_diagnostic(saved_run, tmp_path):
    pytest.importorskip("matplotlib")
    change_first_receipt(saved_run, qualified=False, fallback=True, native_scores=None)
    before = {path: renderer._sha256(path) for path in saved_run.rglob("*") if path.is_file()}
    sim, forecasts, receipts, origin, targets, _, _, checks = renderer._load_inputs(saved_run, SCENARIO, 0, "B")
    np.testing.assert_allclose(forecasts[0].mean, sim.observation[origin:origin + 2, targets] + .1)
    assert receipts[0]["native_scores"] is None and receipts[0]["qualified"] is False
    assert checks[renderer.MODELS[0]] == {"receipt_native_rmse_present": False}
    output = tmp_path / "comparison"
    renderer.render_comparison(saved_run, output, scenario=SCENARIO, animate=False)
    evidence = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert evidence["native_forecasts_only"] is True
    assert evidence["qualification"][renderer.MODELS[0]] is False
    assert evidence["operational_fallback_in_receipt"][renderer.MODELS[0]] is True
    assert evidence["checks"]["saved_native_scores"][renderer.MODELS[0]]["receipt_native_rmse_present"] is False
    assert "UNQUALIFIED · DIAGNOSTIC ONLY" in (output / "comparison.svg").read_text(encoding="utf-8")
    assert before == {path: renderer._sha256(path) for path in before}


@pytest.mark.parametrize("qualified", [False, True])
@pytest.mark.parametrize("scores", [[], "rmse", 1, False])
def test_nonmapping_native_scores_are_rejected(saved_run, qualified, scores):
    change_first_receipt(saved_run, qualified=qualified, native_scores=scores)
    with pytest.raises(ValueError, match="native scores must be a mapping"):
        renderer._load_inputs(saved_run, SCENARIO, 0, "B")


@pytest.mark.parametrize("scores", [None, {}, {"mae": .1}])
def test_qualified_forecast_requires_saved_native_rmse(saved_run, scores):
    change_first_receipt(saved_run, native_scores=scores)
    with pytest.raises(ValueError, match="native scores must be a mapping|requires its saved RMSE"):
        renderer._load_inputs(saved_run, SCENARIO, 0, "B")


@pytest.mark.parametrize("qualified", [False, True])
@pytest.mark.parametrize("scores", [{"rmse": 2.}, {"rmse": .1, "rmse_by_lead": [2., 2.]}])
def test_mismatched_native_rmse_is_rejected(saved_run, qualified, scores):
    change_first_receipt(saved_run, qualified=qualified, native_scores=scores)
    with pytest.raises(ValueError, match="saved forecast disagrees with .*native RMSE"):
        renderer._load_inputs(saved_run, SCENARIO, 0, "B")


@pytest.mark.parametrize(("changes", "message"), [
    ({"origin_index": 5}, "origin disagrees"),
    ({"target_ids": ["wrong", "sites"]}, "site IDs"),
    ({"forecast_metadata": {"origin_index": 3}}, "metadata disagree"),
])
def test_null_scores_do_not_bypass_identity_checks(saved_run, changes, message):
    change_first_receipt(saved_run, qualified=False, native_scores=None, **changes)
    with pytest.raises(ValueError, match=message):
        renderer._load_inputs(saved_run, SCENARIO, 0, "B")


def test_missing_native_forecast_is_not_replaced_by_operational_artifact(saved_run, tmp_path):
    receipt = change_first_receipt(saved_run, qualified=False, fallback=True, native_scores=None)
    (receipt.parent / "forecast.npz").unlink()
    assert (receipt.parent / "operational_point.npz").is_file()
    output = tmp_path / "unavailable_comparison"
    with pytest.raises(FileNotFoundError):
        renderer.render_comparison(saved_run, output, scenario=SCENARIO, animate=False)
    assert not output.exists()
