#!/usr/bin/env python3
"""Seal and audit paired G5 stresses; replay frozen G2 forecasts for blind G6.

No parameter-fitting API is imported or called. G5 fits use ordinary frozen
benchmark profiles. G6 observes only the original pre-origin O history and
reproduces the control scores before evaluating a different future.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
from scipy.linalg import cho_factor, cho_solve
import yaml

from bayes_stesn.benchmark.classical import matern_correlation
from bayes_stesn.benchmark.contracts import Forecast, select_panel
from bayes_stesn.benchmark.execution import estimate_job_memory, memory_allowance
from bayes_stesn.benchmark.generators import Simulation, generate
from bayes_stesn.benchmark.metrics import score_forecast, training_normalization
from bayes_stesn.benchmark.protocol import BenchmarkProtocol, effective_models
from bayes_stesn.benchmark.registry import load_model
from bayes_stesn.benchmark.runner import (benchmark_environment, dgp_seed_for, jsonable,
    single_thread_runtime, source_fingerprint, write_json)
from bayes_stesn.io import content_hash, load_state, seed_for


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT/"configs/benchmark/v03/studies/stress_recipe.yaml"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024*1024), b""):
            digest.update(data)
    return digest.hexdigest()


def label(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT/path


class Inputs:
    def __init__(self):
        self.rows = {}

    def add(self, path):
        path = Path(path).resolve()
        self.rows[label(path)] = {"path": label(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        return path

    def json(self, path, identity_key=None):
        value = json.loads(self.add(path).read_text())
        if identity_key:
            require(value.get(identity_key) == content_hash({k: v for k, v in value.items() if k != identity_key}),
                    f"Content identity differs: {path}")
        return value

    def verify(self):
        for row in self.rows.values():
            path = resolve(row["path"])
            require(path.stat().st_size == row["bytes"] and sha256(path) == row["sha256"],
                    f"Source input changed: {path}")

    def inventory(self):
        return sorted(self.rows.values(), key=lambda r: r["path"])


def verify_inventory(rows):
    for row in rows:
        path = resolve(row["path"])
        require(path.is_file() and path.stat().st_size == row["bytes"] and sha256(path) == row["sha256"],
                f"Sealed study input changed: {path}")


def verify_artifacts(directory, artifacts):
    for artifact in artifacts:
        member = Path(artifact["file"])
        require(not member.is_absolute() and ".." not in member.parts, "Unsafe scientific artifact path")
        path = Path(directory)/member
        require(path.is_file() and path.stat().st_size == artifact["bytes"] and sha256(path) == artifact["sha256"],
                f"Scientific artifact changed: {path}")


def read_run(path, inputs, expected_protocol=None):
    root = Path(path).resolve()
    manifest = inputs.json(root/"manifest.json", "identity")
    summary = inputs.json(root/"summary.json")
    p = BenchmarkProtocol(**manifest["protocol"])
    require(p.stage == "development" and summary.get("complete") and summary.get("source_stable"),
            "Stress controls/arms must be complete source-stable development studies")
    require(manifest["source_fingerprint"] == source_fingerprint()
            and manifest["environment"] == benchmark_environment(), "Stress source or numerical runtime differs")
    require(manifest["protocol_identity"] == p.identity and summary["manifest_identity"] == manifest["identity"],
            "Run protocol/summary binding differs")
    require(expected_protocol is None or p.identity == expected_protocol.identity, "Stress arm differs from its sealed profile")
    return root, manifest, p


def read_job(run, scenario, replication, track, model, inputs):
    root, manifest, p = run
    directory = root/scenario/f"rep_{replication:03d}"/track/model
    row = inputs.json(directory/"receipt.json", "receipt_identity")
    seed = seed_for(p.master_seed, "development_dgp", scenario, replication, track, model)
    expected = content_hash({"manifest": manifest["identity"], "scenario": scenario,
        "replication": replication, "track": track, "model_id": model, "seed": seed})
    require(row.get("terminal_scientific_result") and row["manifest_identity"] == manifest["identity"]
            and row["job_identity"] == expected, "Stress job is nonterminal or mismatched")
    verify_artifacts(directory, row["artifacts"])
    require(row["forecast_seed"] == seed_for(seed, "final_forecast")
            and row["final_fit_seed"] == seed_for(seed, "final_fit"), "Original fit/forecast seed differs")
    return directory, row


def read_data(run, scenario, replication, inputs):
    root, manifest, p = run
    base = root/scenario/f"rep_{replication:03d}"
    receipt = inputs.json(base/"data_receipt.json", "identity")
    path = base/"simulation.npz"
    require(receipt["manifest_identity"] == manifest["identity"] and sha256(path) == receipt["sha256"],
            "Scientific dataset artifact changed")
    simulation = Simulation.load(path)
    require(simulation.metadata["seed"] == receipt["dgp_seed"] == dgp_seed_for(p, scenario, replication)
            and len(simulation.observation) == p.n_times, "Dataset seed or duration differs")
    return simulation


def compare_scores(actual, expected, *, rtol=1e-10, atol=1e-10, path="scores"):
    """Check every numeric leaf plus exact score structure/metadata."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"Replay score keys differ at {path}")
        for key in expected:
            compare_scores(actual[key], expected[key], rtol=rtol, atol=atol, path=f"{path}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"Replay score length differs at {path}")
        for j, (a, b) in enumerate(zip(actual, expected)):
            compare_scores(a, b, rtol=rtol, atol=atol, path=f"{path}[{j}]")
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        require(type(actual) is type(expected) and actual == expected, f"Replay metadata differs at {path}")
    elif isinstance(expected, (int, float)):
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and np.isfinite(actual) and np.isfinite(expected)
                and np.isclose(actual, expected, rtol=rtol, atol=atol), f"Replay numeric score differs at {path}")
    else:
        raise ValueError(f"Unsupported score value at {path}")


def common_geometry(control, stress):
    for name in ("coordinates", "times", "observed_indices", "calibration_indices", "test_indices", "hole_indices"):
        require(np.array_equal(getattr(control, name), getattr(stress, name)), f"Stress/control differ in {name}")


def verify_g5_pair(control, stress, arm):
    common_geometry(control, stress)
    require(np.array_equal(control.generating_mean, stress.generating_mean), "G5 changed its common generating mean")
    if arm == "local_t":
        require(np.array_equal(control.latent, stress.latent), "Local-noise stress changed the latent field")
    elif arm == "whole_field_t":
        require(np.allclose(control.observation-control.latent, stress.observation-stress.latent,
                            rtol=1e-12, atol=1e-12), "Whole-field stress changed its paired measurement residual")
    else:
        raise ValueError("Undeclared G5 arm")
    require(stress.metadata["parameters"]["noise_arm"] == arm
            and stress.metadata["parameters"]["student_t_df"] == 5., "G5 intervention differs")


def verify_g6_pair(control, stress, origin, break_lead=25):
    common_geometry(control, stress)
    first = origin+break_lead-1
    require(stress.metadata["parameters"]["break_index"] == first, "G6 break index differs")
    require(np.array_equal(control.observation[:first], stress.observation[:first])
            and np.array_equal(control.latent[:first], stress.latent[:first]), "G6 changed the pre-break history")
    require(first < len(stress.observation) and not np.array_equal(control.latent[first], stress.latent[first]),
            "The declared G6 first affected row did not change")
    return first


def allowed_training(control, p):
    return select_panel(control.observed_panel(p.final_origin), start=p.training_start)


def fallback_forecast(training, coordinates, site_ids, horizon, track):
    """Reconstruct the original deterministic fallback; no parameter-fit API."""
    if track == "A":
        mapping = {tuple(x): j for j, x in enumerate(training.coordinates)}
        require(all(tuple(x) in mapping for x in coordinates), "Persistence needs only original O sites")
        field = training.y[-1, [mapping[tuple(x)] for x in coordinates]]
    else:
        offset = training.coordinates.min(axis=0)
        scale = max(float(np.ptp(training.coordinates, axis=0).max()), 1e-12)
        o, u = (training.coordinates-offset)/scale, (coordinates-offset)/scale
        center = float(training.y.mean())
        kernel = matern_correlation(o, o, .2, 1.5)+.01*np.eye(len(o))
        weights = cho_solve(cho_factor(kernel, lower=True), training.y[-1]-center)
        field = center+matern_correlation(u, o, .2, 1.5)@weights
    return Forecast(np.tile(field, (horizon, 1)), None, coordinates, site_ids, {"uncertainty": "point_only"})


def verify_model_training(path, training, model_id):
    with zipfile.ZipFile(path) as archive:
        require(sum(row.file_size for row in archive.infolist()) <= 512*1024**2,
                "Saved model exceeds bounded replay-state size")
    state = load_state(path, kind="benchmark_model")
    require(state["model_id"] == model_id, "Saved model identity differs")
    data = state["fitted"]["prepared"]["data"] if model_id == "bayes_esn_ridge" else state["data"]
    for name in ("y", "coordinates", "times"):
        require(np.array_equal(data[name], getattr(training, name)), f"Saved model saw different {name}")
    require(tuple(data["site_ids"]) == tuple(training.site_ids), "Saved model contains different site roles")
    return content_hash({k: data[k] for k in ("y", "coordinates", "times", "site_ids")})


def disk_admission(path, *, reserve_gib=4, next_mib=512):
    free = shutil.disk_usage(Path(path)).free
    required = int(reserve_gib*1024**3+next_mib*1024**2)
    return {"free_bytes": free, "reserve_bytes": int(reserve_gib*1024**3),
            "next_output_and_temporary_allowance_bytes": int(next_mib*1024**2),
            "admitted": free >= required, "scope": "prospective estimate; concurrent unrelated writes remain outside this process"}


def verified_revised_control(history_choice, resolved_profiles, inputs):
    """Bind the completed development choice to its previously resolved profile."""
    choice = inputs.json(history_choice, "identity")
    recipe_path = inputs.add(ROOT/"configs/benchmark/v03/studies/remaining_studies.yaml")
    decision_tool = inputs.add(ROOT/"tools/summarize_v03_learning_curves.py")
    recipe = yaml.safe_load(recipe_path.read_text())
    require(choice["schema"] == "v03-history-choice-1.0"
            and choice["source_fingerprint"] == source_fingerprint()
            and choice["recipe_sha256"] == sha256(recipe_path)
            and choice["tool_sha256"] == sha256(decision_tool)
            and choice["decision_rule"] == recipe["history_choice_gate"]
            and choice["confirmation_outcomes_inspected"] is False,
            "Revised control history choice has different source or prospective rule")
    verify_inventory(choice["source_inputs"])
    selected = choice["selected_initial_history"]
    require(selected in (1000, 2000), "History choice is outside the declared revised profiles")
    root = Path(resolved_profiles)
    provenance = inputs.json(root/"study_provenance.json", "identity")
    profile = BenchmarkProtocol.load(inputs.add(root/f"revised_L{selected}.yaml"))
    require(provenance["execution_ready"] is True
            and provenance["source_fingerprint"] == source_fingerprint()
            and provenance["recipe_sha256"] == sha256(recipe_path)
            and provenance["profile_identities"][f"revised_L{selected}"] == profile.identity
            and profile.initial_fit == selected and profile.training_start == 0,
            "Resolved revised profile does not match the declared history choice")
    return profile, {"history_choice_identity": choice["identity"],
                     "resolved_revised_profile_identity": profile.identity,
                     "study_resolution_identity": provenance["identity"]}


def prepare_design(control_run, output, *, history_choice, resolved_profiles):
    inputs = Inputs()
    recipe = yaml.safe_load(inputs.add(RECIPE).read_text())
    require(recipe["source_fingerprint"] == source_fingerprint(), "Stress recipe is bound to another source")
    expected, choice_binding = verified_revised_control(history_choice, resolved_profiles, inputs)
    run = read_run(control_run, inputs, expected)
    root, manifest, p = run
    require(p.replications == 5 and p.horizon == 100 and p.initial_fit in (1000, 2000)
            and p.training_start == 0 and p.n_side == 7 and p.n_observed == 25 and p.n_calibration == 4
            and p.selection_mode == "development_frozen_configuration", "Unexpected revised control design")
    require(sum(len(effective_models(p, t)) for t in p.tracks) == 11, "Stress recipe requires the declared eleven-job roster")
    for scenario in ("g4_existing_fronts", "g2_spectral_transport"):
        require(scenario in p.scenarios, "Missing stress control mechanism")
        for replication in range(5):
            for track in p.tracks:
                for model in effective_models(p, track):
                    read_job(run, scenario, replication, track, model, inputs)
            base = root/scenario/f"rep_{replication:03d}"
            inputs.json(base/"data_receipt.json", "identity")
    profiles = {}
    for arm in recipe["g5"]["arms"]:
        options = dict(p.generator_options or {})
        options["g5_noise_stress"] = {**options.get("g4_existing_fronts", {}), "arm": arm, "nu": 5.}
        profiles[arm] = replace(p, name=f"v03_stress_g5_{arm}_L{p.initial_fit}",
                               scenarios=("g5_noise_stress",), generator_options=options)
        require(all(dgp_seed_for(p, "g4_existing_fronts", r) == dgp_seed_for(profiles[arm], "g5_noise_stress", r)
                    for r in range(5)), "G5/control DGP seed family mapping differs")
    require(all(dgp_seed_for(p, "g6_velocity_shift", r) == dgp_seed_for(p, "g2_spectral_transport", r)
                for r in range(5)), "G6/control DGP seed family mapping differs")
    inputs.add(__file__)
    inputs.add(ROOT/"configs/benchmark/v03/studies/stress_prepare.py")
    inputs.verify()
    result = {"schema": "v03-stress-design-1.0", "source_fingerprint": source_fingerprint(),
              "environment": benchmark_environment(), "recipe": recipe, "control_run": label(root),
              "control_manifest_identity": manifest["identity"], "control_protocol": p.to_dict(),
              **choice_binding,
              "g5_profiles": {arm: profile.to_dict() for arm, profile in profiles.items()},
              "source_inputs": inputs.inventory(), "tool_sha256": sha256(__file__),
              "required_new_g5_fits": 110, "new_g6_parameter_fits": 0,
              "standalone_g5_normalization_is_cross_arm_comparable": False,
              "scientific_outcomes_generated_by_preparation": False}
    result = jsonable(result)
    result["identity"] = content_hash(result)
    output = Path(output)
    path = output/"stress_design.json"
    if path.exists():
        require(json.loads(path.read_text()) == result, "Refuse to overwrite a different sealed stress design")
    output.mkdir(parents=True, exist_ok=True)
    for arm, profile in profiles.items():
        profile.save(output/f"g5_{arm}.yaml")
    write_json(path, result)
    return result


def load_design(path, inputs):
    design = inputs.json(path, "identity")
    require(design["schema"] == "v03-stress-design-1.0" and design["source_fingerprint"] == source_fingerprint()
            and design["environment"] == benchmark_environment() and design["tool_sha256"] == sha256(__file__),
            "Stress design source, tool, or numerical runtime differs")
    verify_inventory(design["source_inputs"])
    return design


def read_g6_job(directory, original, data, design, inputs):
    """Verify a terminal replay against its exact original control and future."""
    row = inputs.json(Path(directory)/"receipt.json", "receipt_identity")
    p = BenchmarkProtocol(**design["control_protocol"])
    expected = content_hash({"design": design["identity"], "control_receipt": original["receipt_identity"],
                             "g6_data": data["identity"]})
    require(row["schema"] == "v03-g6-replay-job-1.0" and row["job_identity"] == expected
            and row["design_identity"] == design["identity"]
            and row["control_manifest_identity"] == design["control_manifest_identity"]
            and row["control_receipt_identity"] == original["receipt_identity"]
            and row["source_fingerprint"] == design["source_fingerprint"], "G6 replay binding differs")
    require(row["scenario"] == "g6_velocity_shift" and row["replication"] == original["replication"]
            and row["track"] == original["track"] and row["model_id"] == original["model_id"]
            and row["qualified"] is original["qualified"] and row["fallback"] is original["fallback"],
            "G6 replay job roles or qualification differ")
    require(row["new_parameter_fits"] == 0 and row["control_score_replay_verified"] is True
            and row["forecast_seed"] == original["forecast_seed"]
            and row["score_seed"] == seed_for(p.master_seed, "score_pairs", row["track"], "g2_spectral_transport")
            and row["score_tolerance"] == design["recipe"]["g6"]["score_replay_tolerance"]
            and row["origin_index"] == p.final_origin and row["first_affected_row"] == p.final_origin+24
            and row["allowed_training_rows"] == p.final_origin-p.training_start,
            "G6 replay seed, information boundary, or refit policy differs")
    require(row["normalization_reference"] == original["normalization_reference"]
            and np.isclose(row["training_scale"], original["training_scale"], rtol=1e-12, atol=0.),
            "G6 replay normalization differs")
    verify_inventory(row["source_inputs"])
    verify_artifacts(directory, row["artifacts"])
    compare_scores(row["replayed_control_native_scores"], original["native_scores"], **row["score_tolerance"])
    compare_scores(row["replayed_control_operational_scores"], original["operational_scores"], **row["score_tolerance"])
    return row


def completed_g6_jobs(output, run, design, inputs):
    """Count only the exact expected, sealed jobs; reject extraneous receipts."""
    p = run[2]
    expected_paths = {Path(output)/f"rep_{r:03d}"/track/model/"receipt.json"
        for r in range(5) for track in p.tracks for model in effective_models(p, track)}
    saved = set(Path(output).glob("rep_*/*/*/receipt.json"))
    require(saved <= expected_paths, "Unexpected G6 replay receipt outside the declared roster")
    verified = []
    data_by_replication = {}
    for path in sorted(saved):
        replication, track, model = int(path.parts[-4][4:]), path.parts[-3], path.parts[-2]
        if replication not in data_by_replication:
            base = Path(output)/f"rep_{replication:03d}"
            data = inputs.json(base/"data_receipt.json", "identity")
            require(data["design_identity"] == design["identity"] and data["replication"] == replication
                    and data["dgp_seed"] == dgp_seed_for(p, "g6_velocity_shift", replication),
                    "G6 replay dataset identity differs")
            verify_artifacts(base, [{"file": "simulation.npz", "bytes": data["bytes"], "sha256": data["sha256"]}])
            data_by_replication[replication] = data
        _, original = read_job(run, "g2_spectral_transport", replication, track, model, inputs)
        verified.append(read_g6_job(path.parent, original, data_by_replication[replication], design, inputs))
    return verified, len(expected_paths)


def replay_g6(design_path, output, *, max_jobs=None, max_memory_mb=5120):
    inputs = Inputs()
    design = load_design(design_path, inputs)
    run = read_run(resolve(design["control_run"]), inputs, BenchmarkProtocol(**design["control_protocol"]))
    root, manifest, p = run
    output = Path(output).resolve()
    require(not output.is_relative_to(root), "G6 replay cannot write inside its immutable control run")
    output.mkdir(parents=True, exist_ok=True)
    job_count, data_receipts, all_receipts = 0, [], []
    tolerance = design["recipe"]["g6"]["score_replay_tolerance"]
    source = source_fingerprint()
    started = time.perf_counter()
    paused = False
    for replication in range(5):
        if max_jobs is not None and job_count >= max_jobs:
            break
        disk = disk_admission(output)
        if not disk["admitted"]:
            paused = True
            break
        control = read_data(run, "g2_spectral_transport", replication, inputs)
        base = output/f"rep_{replication:03d}"
        base.mkdir(exist_ok=True)
        data_path, receipt_path = base/"simulation.npz", base/"data_receipt.json"
        if data_path.exists() != receipt_path.exists():
            raise ValueError("Incomplete G6 data transaction retained; inspect it before a new measurement")
        if receipt_path.exists():
            data_receipt = inputs.json(receipt_path, "identity")
            require(data_receipt["design_identity"] == design["identity"] and sha256(data_path) == data_receipt["sha256"],
                    "Cached G6 dataset differs")
            stress = Simulation.load(data_path)
        else:
            before, cpu = time.perf_counter(), time.process_time()
            options = dict((p.generator_options or {}).get("g2_spectral_transport", {}))
            options.update(origin_index=p.final_origin, break_lead=25, new_velocity=(-.003, .008))
            stress = generate("g6_velocity_shift", p.n_times, n_side=p.n_side, n_observed=p.n_observed,
                n_calibration=p.n_calibration, seed=dgp_seed_for(p, "g6_velocity_shift", replication), **options)
            first = verify_g6_pair(control, stress, p.final_origin)
            stress.save(data_path)
            data_receipt = {"schema": "v03-g6-data-1.0", "design_identity": design["identity"],
                "replication": replication, "dgp_seed": dgp_seed_for(p, "g6_velocity_shift", replication),
                "first_affected_row": first, "break_lead": 25, "origin_index": p.final_origin,
                "sha256": sha256(data_path), "bytes": data_path.stat().st_size,
                "generation_and_serialization_seconds": time.perf_counter()-before,
                "generation_and_serialization_cpu_seconds": time.process_time()-cpu,
                "cost_scope": "generation, paired-prefix verification, numeric serialization and hashing; excludes this tiny receipt write"}
            data_receipt["identity"] = content_hash(data_receipt)
            write_json(receipt_path, data_receipt)
            inputs.add(receipt_path)
        verify_g6_pair(control, stress, p.final_origin)
        require(stress.metadata["seed"] == dgp_seed_for(p, "g6_velocity_shift", replication), "G6 seed differs")
        data_receipts.append(data_receipt)
        training = allowed_training(control, p)
        scale, normalization = training_normalization(control, p)
        for track in p.tracks:
            targets = control.observed_indices[:p.score_observed_count] if track == "A" else control.test_indices
            coordinates = control.coordinates[targets]
            ids = tuple(control.site_ids[j] for j in targets)
            truth_control = control.observation[p.final_origin:p.n_times, targets]
            truth_stress = stress.observation[p.final_origin:p.n_times, targets]
            score_seed = seed_for(p.master_seed, "score_pairs", track, "g2_spectral_transport")
            for model_id in effective_models(p, track):
                directory, original = read_job(run, "g2_spectral_transport", replication, track, model_id, inputs)
                job_dir = base/track/model_id
                path = job_dir/"receipt.json"
                expected = content_hash({"design": design["identity"], "control_receipt": original["receipt_identity"],
                                         "g6_data": data_receipt["identity"]})
                if path.exists():
                    cached = read_g6_job(job_dir, original, data_receipt, design, inputs)
                    all_receipts.append(cached)
                    continue
                if max_jobs is not None and job_count >= max_jobs:
                    break
                disk = disk_admission(output)
                allowance = memory_allowance(max_memory_mb)
                estimate = estimate_job_memory(p, model_id, track, original["forecast_seed"])
                if (not disk["admitted"] or allowance["effective_job_budget_bytes"] is not None
                        and estimate["estimated_bytes"] > allowance["effective_job_budget_bytes"]):
                    paused = True
                    break
                job_dir.mkdir(parents=True, exist_ok=True)
                state_path = job_dir/"state.json"
                previous = json.loads(state_path.read_text()) if state_path.exists() else None
                if previous:
                    require(previous["job_identity"] == expected, "Interrupted G6 replay has another identity")
                write_json(state_path, {"job_identity": expected, "status": "running", "started_unix": time.time(),
                                       "prior_interrupted_attempt": previous is not None})
                before, cpu = time.perf_counter(), time.process_time()
                native = None
                method = "no_original_native_forecast"
                training_hash = content_hash({"y": training.y, "coordinates": training.coordinates,
                                              "times": training.times, "site_ids": training.site_ids})
                if original["native_scores"]:
                    verified = verify_model_training(directory/"model.npz", training, model_id)
                    require(verified == training_hash, "Model data includes a different permitted history")
                    if (directory/"forecast.npz").exists():
                        native = Forecast.load(directory/"forecast.npz")
                        method = "reused_saved_forecast"
                    else:
                        fitted = load_model(directory/"model.npz")
                        native = fitted.forecast(p.horizon, target_coordinates=coordinates, target_site_ids=ids,
                                                n_paths=p.paths, seed=original["forecast_seed"], history=training)
                        del fitted
                        method = "replayed_frozen_model_original_seed"
                operational = fallback_forecast(training, coordinates, ids, p.horizon, track) if original["fallback"] else native
                require(operational is not None, "Original operational forecast cannot be reproduced")
                control_native = {} if native is None else score_forecast(native, truth_control,
                    pair_count=p.score_pairs, seed=score_seed)
                compare_scores(control_native, original["native_scores"], **tolerance)
                control_operational = score_forecast(operational, truth_control) if original["fallback"] else control_native
                compare_scores(control_operational, original["operational_scores"], **tolerance)
                native_scores = {} if native is None else score_forecast(native, truth_stress,
                    pair_count=p.score_pairs, seed=score_seed)
                operational_scores = score_forecast(operational, truth_stress) if original["fallback"] else native_scores
                artifacts = []
                if replication in p.retain_paths_replications:
                    if native is not None:
                        native.save(job_dir/"forecast.npz")
                    Forecast(operational.mean, None, coordinates, ids).save(job_dir/"operational_point.npz")
                    for name in ("forecast.npz", "operational_point.npz"):
                        member = job_dir/name
                        if member.exists():
                            artifacts.append({"file": name, "bytes": member.stat().st_size, "sha256": sha256(member)})
                receipt = {"schema": "v03-g6-replay-job-1.0", "job_identity": expected,
                    "design_identity": design["identity"], "control_manifest_identity": manifest["identity"],
                    "control_receipt_identity": original["receipt_identity"], "source_fingerprint": source,
                    "scenario": "g6_velocity_shift", "replication": replication, "track": track, "model_id": model_id,
                    "qualified": original["qualified"], "fallback": original["fallback"],
                    "native_scores": native_scores, "operational_scores": operational_scores,
                    "replayed_control_native_scores": control_native, "replayed_control_operational_scores": control_operational,
                    "forecast_seed": original["forecast_seed"], "score_seed": score_seed,
                    "training_scale": scale, "normalization_reference": normalization,
                    "allowed_training_hash": training_hash, "allowed_training_rows": len(training.y),
                    "allowed_training_site_ids": list(training.site_ids), "origin_index": p.final_origin,
                    "first_affected_row": p.final_origin+24, "forecast_method": method,
                    "new_parameter_fits": 0, "control_score_replay_verified": True, "score_tolerance": tolerance,
                    "new_pipeline_seconds": time.perf_counter()-before, "new_pipeline_cpu_seconds": time.process_time()-cpu,
                    "compute_accounting_complete": previous is None, "artifacts": artifacts,
                    "disk_admission": disk, "memory_admission": allowance,
                    "source_inputs": [r for r in inputs.inventory() if r["path"] in
                        {label(directory/"receipt.json"), label(receipt_path), label(design_path)}],
                    "cost_scope": "model loading or forecast reuse/replay, deterministic fallback reconstruction, full control-score verification, stress scoring and demonstration serialization; excludes this tiny receipt write"}
                receipt = jsonable(receipt)
                receipt["receipt_identity"] = content_hash(receipt)
                write_json(path, receipt)
                write_json(state_path, {"job_identity": expected, "status": "complete", "receipt_identity": receipt["receipt_identity"]})
                inputs.add(path)
                all_receipts.append(receipt)
                job_count += 1
                print(json.dumps({"event": "g6_replay_complete", "replication": replication,
                                  "track": track, "model": model_id, "new_parameter_fits": 0}), flush=True)
            if paused or max_jobs is not None and job_count >= max_jobs:
                break
        if paused:
            break
    # Verify all previously saved jobs as well, including contexts not visited
    # because the invocation cap was reached. A receipt count is not validation.
    saved, scheduled = completed_g6_jobs(output, run, design, inputs)
    complete = len(saved) == scheduled
    require(source_fingerprint() == source, "Source changed during G6 replay")
    result = {"schema": "v03-g6-replay-progress-1.0", "design_identity": design["identity"],
              "complete": complete, "jobs": len(saved), "scheduled_jobs": scheduled,
              "new_parameter_fits": 0, "invocation_new_jobs": job_count,
              "invocation_wall_seconds": time.perf_counter()-started, "paused_resource": paused,
              "source_fingerprint": source, "source_inputs": inputs.inventory()}
    inputs.verify()
    result["identity"] = content_hash(result)
    write_json(output/"replay_summary.json", result)
    return result


def paired_summary(values, *, seed, resamples=5000):
    values = np.asarray(values, float)
    require(values.ndim == 1 and np.isfinite(values).all(), "Finite paired effects required")
    if not len(values):
        return {"mean": None, "mcse": None, "pointwise95": None}
    interval = None
    if len(values) > 1:
        rng = np.random.default_rng(seed)
        means = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
        interval = np.quantile(means, (.025, .975)).tolist()
    return {"mean": float(values.mean()),
            "mcse": float(values.std(ddof=1)/np.sqrt(len(values))) if len(values) > 1 else None,
            "pointwise95": interval}


def sum_known(values):
    values = list(values)
    return None if any(value is None or not np.isfinite(value) for value in values) else float(sum(values))


def pair_record(control, stress, *, scale, normalization, arm):
    require(np.isfinite(scale) and scale > 0, "Positive control training SD required")
    require(control["replication"] == stress["replication"] and control["track"] == stress["track"]
            and control["model_id"] == stress["model_id"], "Stress/control job pairing differs")
    return {"pair_id": f"rep_{control['replication']:03d}/{control['track']}/{control['model_id']}",
        "replication": control["replication"], "track": control["track"], "model_id": control["model_id"],
        "arm": arm, "control_receipt_identity": control["receipt_identity"],
        "stress_receipt_identity": stress["receipt_identity"], "control_training_scale": float(scale),
        "control_normalization_reference": normalization,
        "standalone_stress_training_scale": stress.get("training_scale"),
        "control_qualified": bool(control["qualified"]), "stress_qualified": bool(stress["qualified"]),
        "control_fallback": bool(control["fallback"]), "stress_fallback": bool(stress["fallback"]),
        "control_scores": {key: control[key] for key in ("native_scores", "operational_scores")},
        "stress_scores": {key: stress[key] for key in ("native_scores", "operational_scores")},
        "normalization_policy": "Both sides use this same control training-only SD; standalone G5 scales are not used"}


def scalar_metrics():
    metrics = [((name,), 1) for name in ("rmse", "mae", "crps", "field_energy", "spatial_variogram")]
    for level in ("0.5", "0.8", "0.9", "0.95"):
        for name in ("coverage", "width", "interval_score", "field_coverage", "whole_path_covered"):
            metrics.append((("intervals", level, name), 1 if name in {"width", "interval_score"} else 0))
    metrics.extend((("temporal_variogram", lag), 1) for lag in ("1", "5", "20"))
    metrics.extend((("functional_crps", name), 1) for name in
                   ("path_spatial_mean", "final_spatial_mean", "path_maximum", "path_minimum"))
    return metrics


def lookup(value, keys):
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return float(value) if isinstance(value, (int, float, bool)) and np.isfinite(value) else None


def effects_from_pairs(pairs, arm, recipe):
    effects = []
    groups = sorted({(row["model_id"], row["track"]) for row in pairs})
    for model, track in groups:
        rows = [r for r in pairs if r["model_id"] == model and r["track"] == track]
        require(len(rows) == 5 and {r["replication"] for r in rows} == set(range(5)), "Five paired stress fields required")
        for population in ("operational_scores", "native_scores"):
            eligible = rows if population == "operational_scores" else [r for r in rows if
                r["control_qualified"] and r["stress_qualified"] and not r["control_fallback"] and not r["stress_fallback"]]
            for keys, power in scalar_metrics():
                if population == "operational_scores" and keys not in (("rmse",), ("mae",)):
                    continue
                usable, raw, normalized, scales = [], [], [], []
                for row in eligible:
                    a = lookup(row["control_scores"][population], keys)
                    b = lookup(row["stress_scores"][population], keys)
                    if a is None or b is None:
                        continue
                    scale = row["control_training_scale"]
                    raw.append(b-a)
                    normalized.append((b-a)/scale**power)
                    scales.append(scale)
                    usable.append(row["pair_id"])
                if not raw and all(lookup(r["control_scores"][population], keys) is None for r in rows):
                    continue
                metric = "/".join(keys)
                seed = seed_for(recipe["inference"]["bootstrap_seed"], arm, model, track, population, metric)
                resamples = recipe["inference"]["bootstrap_resamples"]
                effects.append({"model_id": model, "track": track, "population": population, "metric": metric,
                    "pair_ids": usable, "N_pair": len(raw), "attempted_pairs": 5,
                    "raw_differences": raw, "normalized_differences": normalized, "control_scales": scales,
                    "normalization_power": power, "raw": paired_summary(raw, seed=seed, resamples=resamples),
                    "common_normalized": paired_summary(normalized, seed=seed, resamples=resamples),
                    "bootstrap_seed": seed, "bootstrap_resamples": resamples,
                    "control_qualified_count": sum(r["control_qualified"] and not r["control_fallback"] for r in rows),
                    "stress_qualified_count": sum(r["stress_qualified"] and not r["stress_fallback"] for r in rows),
                    "sign": "stress minus control; positive is a larger loss except coverage indicators, where it is more coverage"})
    return effects


def summarize(design_path, whole_field_run, local_run, g6_output, output):
    inputs = Inputs()
    design = load_design(design_path, inputs)
    control_run = read_run(resolve(design["control_run"]), inputs, BenchmarkProtocol(**design["control_protocol"]))
    _, control_manifest, p = control_run
    output = Path(output).resolve()
    for root in (control_run[0], Path(whole_field_run).resolve(), Path(local_run).resolve(), Path(g6_output).resolve()):
        require(not output.is_relative_to(root), "Write the stress summary outside immutable source runs")
    arms, historical_control_costs = [], {}
    for arm, path in (("whole_field_t", whole_field_run), ("local_t", local_run)):
        run = read_run(path, inputs, BenchmarkProtocol(**design["g5_profiles"][arm]))
        pairs, source_jobs = [], []
        for replication in range(5):
            control = read_data(control_run, "g4_existing_fronts", replication, inputs)
            stress = read_data(run, "g5_noise_stress", replication, inputs)
            verify_g5_pair(control, stress, arm)
            scale, normalization = training_normalization(control, p)
            for track in p.tracks:
                for model in effective_models(p, track):
                    _, a = read_job(control_run, "g4_existing_fronts", replication, track, model, inputs)
                    _, b = read_job(run, "g5_noise_stress", replication, track, model, inputs)
                    require(a["fit_config"] == b["fit_config"], "G5 changed frozen model settings")
                    require(a["normalization_reference"] == normalization and np.isclose(a["training_scale"], scale, rtol=1e-12),
                            "Control normalization no longer reproduces")
                    pairs.append(pair_record(a, b, scale=scale, normalization=normalization, arm=arm))
                    source_jobs.append(b)
                    historical_control_costs[a["receipt_identity"]] = a["total_pipeline_seconds"]
        arms.append({"id": arm, "control_scenario": "g4_existing_fronts", "stress_scenario": "g5_noise_stress",
            "run_manifest_identity": run[1]["identity"], "paired_jobs": pairs,
            "effects": effects_from_pairs(pairs, arm, design["recipe"]),
            "costs": {"new_parameter_fits": 55,
                "incremental_pipeline_seconds": sum(r["total_pipeline_seconds"] for r in source_jobs),
                "incremental_pipeline_cpu_seconds": sum_known(r.get("total_pipeline_cpu_seconds") for r in source_jobs),
                "compute_accounting_complete": all(r.get("compute_accounting_complete", False) for r in source_jobs),
                "source_job_receipt_identities": [r["receipt_identity"] for r in source_jobs],
                "scope": "New G5 arm's ordinary benchmark fitting/forecast/scoring pipelines; excludes shared DGP generation and controller overhead, which the frozen core does not itemize; reused G4 controls accounted separately"},
            "normalization": "Control G4 training SD for both sides; standalone arm normalization is not cross-arm comparable"})
    g6_root = Path(g6_output).resolve()
    progress = inputs.json(g6_root/"replay_summary.json", "identity")
    require(progress["complete"] and progress["jobs"] == 55 and progress["design_identity"] == design["identity"],
            "G6 replay is not complete under this design")
    pairs, replay_jobs, generations = [], [], []
    for replication in range(5):
        control = read_data(control_run, "g2_spectral_transport", replication, inputs)
        base = g6_root/f"rep_{replication:03d}"
        data = inputs.json(base/"data_receipt.json", "identity")
        require(data["design_identity"] == design["identity"] and sha256(base/"simulation.npz") == data["sha256"],
                "G6 data receipt differs")
        stress = Simulation.load(base/"simulation.npz")
        verify_g6_pair(control, stress, p.final_origin)
        generations.append(data)
        scale, normalization = training_normalization(control, p)
        for track in p.tracks:
            for model in effective_models(p, track):
                _, a = read_job(control_run, "g2_spectral_transport", replication, track, model, inputs)
                directory = base/track/model
                b = read_g6_job(directory, a, data, design, inputs)
                require(b["normalization_reference"] == normalization and np.isclose(b["training_scale"], scale, rtol=1e-12),
                        "G6 common control scale differs")
                pairs.append(pair_record(a, b, scale=scale, normalization=normalization, arm="velocity_shift"))
                replay_jobs.append(b)
                historical_control_costs[a["receipt_identity"]] = a["total_pipeline_seconds"]
    arms.append({"id": "velocity_shift", "control_scenario": "g2_spectral_transport", "stress_scenario": "g6_velocity_shift",
        "replay_design_identity": design["identity"], "paired_jobs": pairs,
        "effects": effects_from_pairs(pairs, "velocity_shift", design["recipe"]),
        "costs": {"new_parameter_fits": 0, "forecast_jobs": 55,
            "replayed_model_forecasts": sum(r["forecast_method"] == "replayed_frozen_model_original_seed" for r in replay_jobs),
            "reused_saved_forecasts": sum(r["forecast_method"] == "reused_saved_forecast" for r in replay_jobs),
            "generation_and_serialization_seconds": sum(r["generation_and_serialization_seconds"] for r in generations),
            "generation_and_serialization_cpu_seconds": sum_known(r["generation_and_serialization_cpu_seconds"] for r in generations),
            "replay_verification_scoring_serialization_seconds": sum(r["new_pipeline_seconds"] for r in replay_jobs),
            "replay_verification_scoring_serialization_cpu_seconds": sum_known(r["new_pipeline_cpu_seconds"] for r in replay_jobs),
            "incremental_pipeline_seconds": sum(r["generation_and_serialization_seconds"] for r in generations)
                +sum(r["new_pipeline_seconds"] for r in replay_jobs),
            "incremental_pipeline_cpu_seconds": sum_known([r["generation_and_serialization_cpu_seconds"] for r in generations]
                +[r["new_pipeline_cpu_seconds"] for r in replay_jobs]),
            "compute_accounting_complete": all(r["compute_accounting_complete"] for r in replay_jobs),
            "source_job_receipt_identities": [r["receipt_identity"] for r in replay_jobs],
            "source_generation_receipt_identities": [r["identity"] for r in generations],
            "scope": "G6 generation, frozen forecast reuse/replay, deterministic original fallback reconstruction, all-metric G2 verification and G6 scoring; no parameter refits"},
        "normalization": "Same G2 control training SD for both futures", "adaptation": False,
        "break_lead": 25, "first_affected_row": p.final_origin+24})
    inputs.verify()
    result = {"schema": "v03-paired-stress-summary-1.0", "complete": True,
        "source_fingerprint": source_fingerprint(), "design_identity": design["identity"],
        "design_path": label(design_path), "design_sha256": sha256(design_path),
        "control_manifest_identity": control_manifest["identity"], "arms": arms,
        "source_inputs": inputs.inventory(), "new_parameter_fits": 110,
        "incremental_pipeline_seconds": sum(arm["costs"]["incremental_pipeline_seconds"] for arm in arms),
        "incremental_pipeline_cpu_seconds": sum_known(arm["costs"]["incremental_pipeline_cpu_seconds"] for arm in arms),
        "compute_accounting_complete": all(arm["costs"]["compute_accounting_complete"] for arm in arms),
        "historical_shared_control_pipeline_seconds": sum(historical_control_costs.values()),
        "historical_shared_control_costs_by_receipt": historical_control_costs,
        "standalone_g5_normalization_is_cross_arm_comparable": False,
        "limitations": ["Five independent paired fields per arm; pointwise exploratory uncertainty, no formal robustness superiority claim",
            "Native effects condition on the exact common-qualified intersection; operational effects retain every completed point forecast",
            "Coupled control/stress arms are not independent macro strata; reused control cost is counted once separately",
            "G5 uses ordinary scenario-specific fitting seeds, identical across G5 arms but distinct from G4; it measures the declared fitting procedure",
            "G5 pipeline sums exclude shared dataset generation and controller overhead; invocation wall-time logs include those costs but do not separately itemize their CPU use",
            "Blind G6 changes evaluation truth only after a hidden lead25 transition; no adaptation or post-shift calibration"]}
    result = jsonable(result)
    result["identity"] = content_hash(result)
    write_json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("replay-g6")
    replay.add_argument("--design", required=True)
    replay.add_argument("--output", required=True)
    replay.add_argument("--max-jobs", type=int)
    replay.add_argument("--max-memory-mb", type=float, default=5120)
    final = sub.add_parser("summarize")
    final.add_argument("--design", required=True)
    final.add_argument("--whole-field-run", required=True)
    final.add_argument("--local-run", required=True)
    final.add_argument("--g6-output", required=True)
    final.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    with single_thread_runtime():
        if args.command == "replay-g6":
            require(args.max_jobs is None or args.max_jobs >= 1, "max-jobs must be positive")
            result = replay_g6(args.design, args.output, max_jobs=args.max_jobs, max_memory_mb=args.max_memory_mb)
        else:
            result = summarize(args.design, args.whole_field_run, args.local_run, args.g6_output, args.output)
    print(json.dumps({"output": str(Path(args.output).resolve()), "complete": result["complete"],
                      "identity": result["identity"], "new_parameter_fits": result.get("new_parameter_fits", 0)}, indent=2))
    return 75 if result.get("paused_resource") else 0


if __name__ == "__main__":
    raise SystemExit(main())
