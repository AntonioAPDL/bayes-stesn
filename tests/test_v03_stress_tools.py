"""Independent stress-analysis references; no DGP generation or model fitting."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pytest

from bayes_stesn.benchmark.generators import Simulation
from bayes_stesn.io import content_hash, seed_for


def tool():
    path = Path(__file__).resolve().parents[1]/"tools/summarize_v03_stress.py"
    spec = importlib.util.spec_from_file_location("stress_tools", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def toy_simulation():
    y = np.arange(80., dtype=float).reshape(20, 4)
    return Simulation(y, y-.25, np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
                      np.arange(20.), np.array([0, 1]), np.array([2]), np.array([3]), np.array([], int), {})


def test_stress_uses_common_control_sd_and_exact_qualified_intersection():
    module = tool()
    pairs = []
    for r, scale in enumerate((1., 2., 4., 8., 16.)):
        common = {"replication": r, "model_id": "bayes_esn_ridge", "track": "A", "fallback": False,
                  "qualified": True, "receipt_identity": f"control-{r}",
                  "native_scores": {"rmse": 1., "mae": 1., "crps": 1.},
                  "operational_scores": {"rmse": 1., "mae": 1.}}
        stress = {**common, "receipt_identity": f"stress-{r}", "qualified": r != 2,
                  "training_scale": 1000000., "native_scores": {"rmse": 1.+scale, "mae": 1.+scale, "crps": 1.+scale},
                  "operational_scores": {"rmse": 1.+scale, "mae": 1.+scale}}
        pairs.append(module.pair_record(common, stress, scale=scale, normalization={"identity": f"scale-{r}"}, arm="local_t"))
    effects = module.effects_from_pairs(pairs, "local_t", {"inference": {"bootstrap_seed": 17, "bootstrap_resamples": 50}})
    operational = next(r for r in effects if r["population"] == "operational_scores" and r["metric"] == "rmse")
    native = next(r for r in effects if r["population"] == "native_scores" and r["metric"] == "crps")
    assert operational["N_pair"] == 5
    assert native["N_pair"] == 4
    assert native["pair_ids"] == [f"rep_{r:03d}/A/bayes_esn_ridge" for r in (0, 1, 3, 4)]
    assert_array_equal(operational["normalized_differences"], np.ones(5))
    assert operational["common_normalized"] == {"mean": 1., "mcse": 0., "pointwise95": [1., 1.]}
    assert_allclose(operational["raw"]["mean"], 31/5)


def test_g6_first_affected_row_and_prebreak_are_verified():
    from dataclasses import replace
    module = tool()
    control = toy_simulation()
    latent, observed = control.latent.copy(), control.observation.copy()
    latent[12:] += 2.
    observed[12:] += 2.
    stress = replace(control, latent=latent, observation=observed, metadata={"parameters": {"break_index": 12}})
    assert module.verify_g6_pair(control, stress, 10, break_lead=3) == 12
    observed[11, 0] += .1
    contaminated = replace(stress, observation=observed)
    with pytest.raises(ValueError, match="pre-break history"):
        module.verify_g6_pair(control, contaminated, 10, break_lead=3)


def test_forecast_information_is_only_original_O_prefix():
    from dataclasses import replace
    module = tool()
    control = toy_simulation()
    p = SimpleNamespace(final_origin=10, training_start=0)
    first = module.allowed_training(control, p)
    changed = control.observation.copy()
    changed[:, control.test_indices] += 100000.
    changed[:, control.calibration_indices] -= 200000.
    changed[10:, control.observed_indices] += 300000.
    second = module.allowed_training(replace(control, observation=changed), p)
    assert_array_equal(first.y, second.y)
    assert first.site_ids == ("site_00000", "site_00001")
    assert first.y.shape == (10, 2)
    assert first.times[-1] == 9.


def test_role_changes_do_not_count_as_a_matched_future():
    from dataclasses import replace
    module = tool()
    control = toy_simulation()
    changed = replace(control, calibration_indices=control.test_indices, test_indices=control.calibration_indices)
    with pytest.raises(ValueError, match="calibration_indices"):
        module.common_geometry(control, changed)


def test_all_metric_replay_check_rejects_a_hidden_companion_difference():
    module = tool()
    expected = {"rmse": 2., "intervals": {"0.9": {"coverage": .9}}, "path": [0., 1.]}
    module.compare_scores({"rmse": 2.+1e-11, "intervals": {"0.9": {"coverage": .9}}, "path": [0., 1.]}, expected)
    with pytest.raises(ValueError, match="numeric score"):
        module.compare_scores({"rmse": 2., "intervals": {"0.9": {"coverage": .91}}, "path": [0., 1.]}, expected)


def test_same_size_artifact_corruption_is_rejected_before_replay(tmp_path):
    module = tool()
    manifest = {"identity": "manifest"}
    p = SimpleNamespace(master_seed=7)
    scenario, model, track, rep = "g2_spectral_transport", "persistence", "A", 0
    directory = tmp_path/scenario/"rep_000"/track/model
    directory.mkdir(parents=True)
    artifact = directory/"model.npz"
    artifact.write_bytes(b"abc")
    seed = seed_for(7, "development_dgp", scenario, rep, track, model)
    value = {"manifest_identity": "manifest", "terminal_scientific_result": True,
             "job_identity": content_hash({"manifest": "manifest", "scenario": scenario, "replication": rep,
                 "track": track, "model_id": model, "seed": seed}),
             "forecast_seed": seed_for(seed, "final_forecast"), "final_fit_seed": seed_for(seed, "final_fit"),
             "artifacts": [{"file": "model.npz", "bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}]}
    value["receipt_identity"] = content_hash(value)
    (directory/"receipt.json").write_text(json.dumps(value))
    module.read_job((tmp_path, manifest, p), scenario, rep, track, model, module.Inputs())
    artifact.write_bytes(b"abd")
    with pytest.raises(ValueError, match="artifact changed"):
        module.read_job((tmp_path, manifest, p), scenario, rep, track, model, module.Inputs())


def test_spatial_fallback_preserves_a_constant_field_without_fitting():
    module = tool()
    from bayes_stesn.benchmark.contracts import Panel
    observed = Panel(np.full((5, 2), 3.5), np.array([[0., 0.], [1., 1.]]), np.arange(5.), ("O0", "O1"))
    forecast = module.fallback_forecast(observed, np.array([[.2, .8]]), ("U0",), 4, "B")
    assert_array_equal(forecast.mean, np.full((4, 1), 3.5))
    assert forecast.samples is None


def test_replay_completion_rejects_an_extra_receipt_instead_of_counting_it(tmp_path, monkeypatch):
    module = tool()
    p = SimpleNamespace(tracks=("A",))
    monkeypatch.setattr(module, "effective_models", lambda protocol, track: ("persistence",))
    unexpected = tmp_path/"rep_999"/"A"/"persistence"/"receipt.json"
    unexpected.parent.mkdir(parents=True)
    unexpected.write_text("{}")
    with pytest.raises(ValueError, match="outside the declared roster"):
        module.completed_g6_jobs(tmp_path, (tmp_path, {}, p), {}, module.Inputs())


def test_cached_replay_cannot_change_control_qualification_or_source(tmp_path):
    module = tool()
    p = module.BenchmarkProtocol(stage="development", initial_fit=40, horizon=2,
        n_side=3, n_observed=3, n_calibration=1, models=("persistence",), tracks=("A",), replications=5)
    original = {"receipt_identity": "control", "replication": 0, "track": "A", "model_id": "persistence",
        "qualified": True, "fallback": False, "forecast_seed": 4, "normalization_reference": {"rows": 40},
        "training_scale": 2., "native_scores": {"rmse": 1.}, "operational_scores": {"rmse": 1.}}
    design = {"identity": "design", "control_protocol": p.to_dict(), "control_manifest_identity": "manifest",
        "source_fingerprint": "frozen-source", "recipe": {"g6": {"score_replay_tolerance": {"rtol": 1e-10, "atol": 1e-10}}}}
    data = {"identity": "future"}
    row = {"schema": "v03-g6-replay-job-1.0",
        "job_identity": content_hash({"design": "design", "control_receipt": "control", "g6_data": "future"}),
        "design_identity": "design", "control_manifest_identity": "manifest", "control_receipt_identity": "control",
        "source_fingerprint": "frozen-source", "scenario": "g6_velocity_shift", "replication": 0, "track": "A",
        "model_id": "persistence", "qualified": True, "fallback": False, "new_parameter_fits": 0,
        "control_score_replay_verified": True, "forecast_seed": 4,
        "score_seed": seed_for(p.master_seed, "score_pairs", "A", "g2_spectral_transport"),
        "score_tolerance": {"rtol": 1e-10, "atol": 1e-10}, "origin_index": p.final_origin,
        "first_affected_row": p.final_origin+24, "allowed_training_rows": p.final_origin,
        "normalization_reference": {"rows": 40}, "training_scale": 2., "source_inputs": [], "artifacts": [],
        "replayed_control_native_scores": {"rmse": 1.}, "replayed_control_operational_scores": {"rmse": 1.}}

    def save(value):
        (tmp_path/"receipt.json").write_text(json.dumps({**value, "receipt_identity": content_hash(value)}))

    save(row)
    module.read_g6_job(tmp_path, original, data, design, module.Inputs())
    save({**row, "qualified": False})
    with pytest.raises(ValueError, match="qualification"):
        module.read_g6_job(tmp_path, original, data, design, module.Inputs())
    save({**row, "source_fingerprint": "unrelated-source"})
    with pytest.raises(ValueError, match="binding differs"):
        module.read_g6_job(tmp_path, original, data, design, module.Inputs())


def test_stress_preparation_binds_the_selected_history_and_resolved_protocol(tmp_path, monkeypatch):
    module = tool()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "source_fingerprint", lambda: "source")
    recipe = tmp_path/"configs/benchmark/v03/studies/remaining_studies.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("history_choice_gate:\n  default_if_evidence_is_insufficient: 1000\n")
    decision_tool = tmp_path/"tools/summarize_v03_learning_curves.py"
    decision_tool.parent.mkdir()
    decision_tool.write_text("# fixed prospective reducer\n")
    gate = {"default_if_evidence_is_insufficient": 1000}
    choice = {"schema": "v03-history-choice-1.0", "source_fingerprint": "source",
        "recipe_sha256": module.sha256(recipe), "tool_sha256": module.sha256(decision_tool),
        "decision_rule": gate, "confirmation_outcomes_inspected": False,
        "source_inputs": [], "selected_initial_history": 1000}
    choice_path = tmp_path/"history_choice.json"
    choice_path.write_text(json.dumps({**choice, "identity": content_hash(choice)}))
    profiles = tmp_path/"resolved"
    profiles.mkdir()

    def save_profile(initial):
        p = module.BenchmarkProtocol(stage="development", initial_fit=initial)
        p.save(profiles/"revised_L1000.yaml")
        provenance = {"execution_ready": True, "source_fingerprint": "source",
            "recipe_sha256": module.sha256(recipe), "profile_identities": {"revised_L1000": p.identity}}
        (profiles/"study_provenance.json").write_text(json.dumps({**provenance, "identity": content_hash(provenance)}))
        return p

    first = save_profile(1000)
    verified, binding = module.verified_revised_control(choice_path, profiles, module.Inputs())
    assert verified.identity == first.identity
    assert binding["history_choice_identity"] == content_hash(choice)
    save_profile(2000)
    with pytest.raises(ValueError, match="declared history choice"):
        module.verified_revised_control(choice_path, profiles, module.Inputs())
