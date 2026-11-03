#!/usr/bin/env python3
"""Bind prospective studies to G4 validation selections; never generate data.

Without --preflight-dir, write explicit draft profiles for training-only cost
measurement. With complete preflight receipts, bind common finite model time
allowances and write execution-ready profiles. --planning-proxy writes only
resource estimates from conservative reference settings, never runnable claims
about unknown selected configurations.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import yaml

from bayes_stesn.benchmark.execution import estimate_job_memory, plan_execution
from bayes_stesn.benchmark.protocol import BenchmarkProtocol, candidate_manifest, effective_models
from bayes_stesn.benchmark.registry import MODELS
from bayes_stesn.benchmark.runner import jsonable, source_fingerprint, write_json
from bayes_stesn.benchmark.studies import (_fhn_evidence, default_primary_comparisons,
                                         history_curve_profiles, sensor_curve_profiles)
from bayes_stesn.io import content_hash, seed_for


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RECIPE = HERE/"remaining_studies.yaml"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_identity(path, key="identity"):
    value = json.loads(Path(path).read_text())
    require(value.get(key) == content_hash({k: v for k, v in value.items() if k != key}),
            f"Invalid content identity: {path}")
    return value


def path_decision(path, count):
    """Bind the completed, prespecified P5 rule; never reinterpret its results."""
    value = read_identity(path)
    assessment = value.get("path_count_assessment", {})
    require(value.get("schema") == "p5-extension-findings-1.0"
            and value.get("source_fingerprint") == source_fingerprint(),
            "Path-count decision must bind the current-source P5 evidence")
    require(assessment.get("prespecification_verified_against_both_first_path_invocation_times") is True,
            "Path-count decision lacks verified prospective declaration")
    counts = assessment.get("predeclared_rule", {}).get("candidate_counts")
    require(counts == [128, 256, 512], "Unexpected prospective path-count comparison")
    passes = assessment.get("passes_on_both_cases", {})
    eligible = [n for n in counts if passes.get(str(n)) is True]
    require(eligible and assessment.get("selected_for_subsequent_studies") == min(eligible) == count,
            "Requested path count differs from the completed P5 selection rule")
    return {"path": str(path), "sha256": file_hash(path), "identity": value["identity"],
            "selected_for_subsequent_studies": count,
            "limitations": assessment.get("limitations")}


def long_generator_evidence(path, protocol):
    """Strengthen the generic horizon check to cover the full history record."""
    checked = _fhn_evidence(protocol, path)
    value = checked["evidence"]
    delta = protocol.generator_options["g3_excitable_field"]["delta"]
    duration = (value["n_observations"]-1)*value["observation_spacing"]
    required = (protocol.n_times-1)*delta
    require(value["seed"] == 983417 and value["observation_spacing"] == delta,
            "Long-history generator evidence must use the independently declared seed and observation spacing")
    require(duration >= required, "Generator evidence is shorter than the complete history record")
    require(value["validation_tool_sha256"] == file_hash(ROOT/"tools/validate_generators.py"),
            "Generator evidence uses a different validation tool")
    return {"path": str(path), "sha256": file_hash(path), "identity": value["identity"],
            "validated_post_burnin_duration": duration, "required_full_record_duration": required,
            "time_gate_passed": checked["time_gate_passed"], "time_rmse": checked["time_rmse"],
            "mesh_gate_passed": checked["mesh_gate_passed"], "continuum_claim": False,
            "original_overall_status": value["status"],
            "scope": "Independent finite-grid time-step pilot covers this complete record duration; not a guarantee for every field or a continuum claim"}


def selected_specifications(recipe, reference, tuned=None):
    requested = list(recipe["common_protocol"]["models"])
    reference = BenchmarkProtocol.load(reference)
    model_pairs = [(model, track) for track in ("A", "B")
                   for model in (["spatial_persistence"] if track == "B" else [])+requested
                   if track in MODELS[model].tracks]
    frozen, provenance = {}, []
    manifest = None
    if tuned is not None:
        manifest = read_identity(Path(tuned)/"manifest.json")
        source = BenchmarkProtocol(**manifest["protocol"])
        require(source.stage == "development" and source.selection_mode == "retuned_pipeline",
                "Specifications must come from the declared retuned development procedure")
        require(manifest["source_fingerprint"] == source_fingerprint(),
                "G4 selection source differs from the current package; do not silently transfer it")
    for model, track in model_pairs:
        baseline = candidate_manifest(model, reference, track=track)["candidates"][0]
        origin = {"model": model, "track": track, "disposition": "planning_reference_proxy"}
        config = deepcopy(baseline)
        if tuned is not None:
            path = Path(tuned)/"g4_existing_fronts"/"rep_000"/track/model/"selection.json"
            require(path.is_file(), f"G4 selection is still pending: {path}")
            selection = read_identity(path, "selection_identity")
            fit_seed = seed_for(source.master_seed, "development_dgp", "g4_existing_fronts", 0, track, model)
            expected_job = content_hash({"manifest": manifest["identity"], "scenario": "g4_existing_fronts",
                "replication": 0, "track": track, "model_id": model, "seed": fit_seed})
            require(selection.get("job_identity") == expected_job, "G4 selection belongs to another job")
            if selection["selected"] is not None:
                require(selection["status"] == "selected", "Selection status is not terminal")
                chosen = selection["selected_candidate_hash"]
                require(content_hash(selection["selected"]) == chosen, "Selected effective hash differs")
                checks = [r for r in selection["rechecks"] if r["candidate_hash"] == chosen]
                expected = source.finalist_seeds if MODELS[model].stochastic_fit else 1
                require(len(checks) == expected and all(r["status"] == "completed" and r["qualified"] for r in checks),
                        "Chosen configuration lacks every required qualified finalist recheck")
                config = deepcopy(selection["selected"])
                origin["disposition"] = "qualified_G4_validation_selection"
            else:
                require(selection["status"] in {"all_candidates_failed", "all_finalists_unqualified"},
                        "Missing/interrupted selection cannot trigger a conservative fallback")
                origin["disposition"] = "declared_conservative_reference_after_failed_G4_selection"
            origin.update(selection_receipt=str(path), selection_sha256=file_hash(path),
                          selection_identity=selection["selection_identity"],
                          selection_context_identity=selection["selection_context_identity"],
                          original_effective_hash=content_hash(config))
        if model == "basis_gru":
            origin["declared_horizon_transfer"] = {
                "from": config["validation_horizon"], "to": recipe["common_protocol"]["horizon"]}
            config["validation_horizon"] = recipe["common_protocol"]["horizon"]
            config["require_full_validation_horizon"] = True
        if model == "eqesn_rollout":
            require(config["members"] == recipe["common_protocol"]["ensemble_members"] == 20,
                    "The declared E-QESN control retains its original twenty-member finite ensemble")
        frozen[f"{model}/{track}"] = config
        origin["transferred_effective_hash"] = content_hash(config)
        provenance.append(origin)
    return frozen, provenance, manifest


def profiles_from_recipe(recipe, frozen, *, caps=None, paths=512):
    common = {**deepcopy(recipe["common_protocol"]), "frozen_configurations": deepcopy(frozen),
              "fit_time_limits_seconds": caps, "paths": paths}
    fields = {"name", "master_seed", "initial_fit", "training_start", "n_side", "n_observed",
              "n_calibration", "replications", "scenarios", "normalization_reference"}
    bases = {name: BenchmarkProtocol(**{**common, **{k: v for k, v in recipe[name].items() if k in fields}})
             for name in ("history", "sensors")}
    profiles = {"history_base": bases["history"], "sensors_base": bases["sensors"]}
    history = history_curve_profiles(bases["history"], recipe["history"]["values"], common_endpoint=4000)
    sensors = sensor_curve_profiles(bases["sensors"], recipe["sensors"]["values"])
    profiles.update({f"history_{v}": p for v, p in zip(recipe["history"]["values"], history)})
    profiles.update({f"sensors_{v}": p for v, p in zip(recipe["sensors"]["values"], sensors)})
    revised = recipe["revised_development"]
    for length in revised["candidate_initial_histories"]:
        row = {**common, **{k: v for k, v in revised.items() if k in fields},
               "name": f"{revised['name']}_L{length}", "initial_fit": length,
               "normalization_reference": {"start": length-revised["normalization_window"], "stop": length,
                    "observed_count": 25, "floor": 1e-12, "ddof": revised["normalization_ddof"]}}
        p = BenchmarkProtocol(**row)
        profiles[f"revised_L{length}"] = p
        profiles[f"confirmation_L{length}"] = replace(p,
            name=f"{recipe['confirmation']['name']}_L{length}", stage="confirmation",
            replications=recipe["confirmation"]["draft_replications"])
    for profile in profiles.values():
        for track in profile.tracks:
            for model in effective_models(profile, track):
                candidate_manifest(model, profile, track=track,
                    fitting_sites=profile.n_observed,
                    fitting_rows=profile.final_origin-profile.training_start)
    return profiles


def preflight_jobs(draft_profiles, recipe):
    jobs = []
    for case in recipe["preflight"]["cases"]:
        p = draft_profiles[case]
        for model in recipe["preflight"]["models"]:
            choices = [(estimate_job_memory(p, model, track, 0)["estimated_bytes"], track)
                       for track in p.tracks if model in effective_models(p, track)]
            largest = max(value for value, _ in choices)
            track = min(track for value, track in choices if value == largest)
            jobs.append({"case": case, "model_id": model, "track": track,
                         "profile_identity": p.identity, "estimated_job_bytes": largest,
                         "selection_policy": "larger dimension-based memory estimate; lexical track tie break; no response score"})
    return jobs


def caps_from_preflight(directory, draft_profiles, recipe, existing):
    maxima = dict(existing)
    require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0 for value in maxima.values()),
            "Existing reference timings must be finite nonnegative seconds")
    evidence = []
    for job in preflight_jobs(draft_profiles, recipe):
        case, track, model = job["case"], job["track"], job["model_id"]
        path = Path(directory)/f"{case}__{track}__{model}.json"
        row = read_identity(path)
        require(row["profile_identity"] == job["profile_identity"] and row["source_fingerprint"] == source_fingerprint(),
                "Preflight uses different selected settings, geometry, paths or package source")
        profile = draft_profiles[case]
        config = candidate_manifest(model, profile, track=track, fitting_sites=profile.n_observed,
            fitting_rows=profile.final_origin-profile.training_start)["candidates"][0]
        expected = {"case": case, "model_id": model, "track": track,
                    "fit_seed": seed_for(recipe["preflight"]["master_seed"], "training_only_study_preflight", case, model, track),
                    "config_hash": content_hash(config), "recipe_sha256": file_hash(RECIPE),
                    "tool_sha256": file_hash(HERE/"training_preflight.py"),
                    "elapsed_allowance_seconds": recipe["preflight"]["elapsed_allowance_seconds_per_fresh_process"]}
        require(all(row.get(key) == value for key, value in expected.items()),
                "Preflight measurement differs from its exact planned model, seed, tool or budget")
        require(row["execution_status"] == "completed" and not row["forecast_computed"],
                "Incomplete/capped preflight cannot set a study allowance")
        require(row["master_seed"] == recipe["preflight"]["master_seed"], "Preflight seed differs")
        require(math.isfinite(row["fit_seconds"]) and row["fit_seconds"] > 0,
                "Fresh preflight fit time must be finite and positive")
        maxima[model] = max(float(row["fit_seconds"]), float(maxima.get(model, 0.)))
        evidence.append({"file": str(path), "sha256": file_hash(path), "identity": row["identity"]})
    models = {m for p in draft_profiles.values() for t in p.tracks for m in effective_models(p, t)}
    caps = {model: max(30, math.ceil(20*maxima[model]+10)) for model in sorted(models)}
    return caps, evidence


def curve_design(profiles, recipe, kind):
    values = recipe[kind]["values"]
    rows = [profiles[f"{kind}_{value}"] for value in values]
    result = jsonable({"schema": "benchmark-curve-2.0", "kind": "history" if kind == "history" else "sensors",
        "values": values, "profiles": [p.to_dict() for p in rows], "common_endpoint": rows[0].initial_fit,
        "fixed_U": kind == "sensors", "separate_experiments_not_full_factorial": True})
    result["identity"] = content_hash(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuned-run", "--tuned-dir", dest="tuned_run")
    parser.add_argument("--reference-config", default="configs/benchmark/v03/reference_reliability.yaml")
    parser.add_argument("--planning-proxy", action="store_true")
    parser.add_argument("--preflight-dir")
    parser.add_argument("--measurements", default="docs/evidence/v03/measured_fit_seconds.json")
    parser.add_argument("--paths", type=int, choices=(128, 256, 512), default=512)
    parser.add_argument("--path-decision", help="Completed P5 evidence/decision receipt binding the selected path count")
    parser.add_argument("--g3-validation", default="docs/evidence/v03/generator_validation_long_history_grid64.json",
                        help="Independent numerical validation covering the complete long-history record")
    parser.add_argument("--output", required=True, help="Directory for draft/final profiles or planning-only evidence")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-memory-mb", type=float, default=5120)
    args = parser.parse_args(argv)
    require(args.planning_proxy != bool(args.tuned_run), "Choose --planning-proxy or an actual --tuned-run")
    require(args.planning_proxy or args.path_decision, "Materialization needs the completed path-count decision receipt")
    recipe = yaml.safe_load(RECIPE.read_text())
    path_evidence = None if args.path_decision is None else path_decision(args.path_decision, args.paths)
    frozen, selections, source = selected_specifications(recipe, args.reference_config, args.tuned_run)
    draft = profiles_from_recipe(recipe, frozen, paths=args.paths)
    generator_evidence = (long_generator_evidence(args.g3_validation, draft["history_base"])
                          if Path(args.g3_validation).exists() else None)
    require(args.planning_proxy or generator_evidence is not None,
            "Materialization requires complete long-history G3 validation evidence")
    caps, timing_evidence = (None, [])
    if args.preflight_dir:
        require(not args.planning_proxy, "Proxy estimates cannot be finalized as selected specifications")
        caps, timing_evidence = caps_from_preflight(args.preflight_dir, draft, recipe,
                                                   json.loads(Path(args.measurements).read_text()))
    profiles = profiles_from_recipe(recipe, frozen, caps=caps, paths=args.paths)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    execution_ready = bool(args.tuned_run and caps is not None)
    label = "reference_proxy_planning_only" if args.planning_proxy else "execution_ready" if execution_ready else "draft_needs_training_only_preflight"
    plans = {name: plan_execution(p, workers=args.workers, max_memory_mb=args.max_memory_mb)
             for name, p in profiles.items() if name not in {"history_base", "sensors_base"}}
    # Low-dimensional reference timings are not silently extrapolated to these
    # larger studies; time projections stay null pending target-size preflight.
    provenance = {"schema": "v03-remaining-study-resolution-1.0", "status": label,
        "execution_ready": execution_ready, "scientific_outcomes_generated": False,
        "source_fingerprint": source_fingerprint(), "recipe_sha256": file_hash(RECIPE),
        "preparer_sha256": file_hash(__file__), "reference_config_sha256": file_hash(args.reference_config),
        "selection_manifest_identity": None if source is None else source["identity"],
        "selections": selections, "paths": args.paths,
        "path_decision_sha256": None if args.path_decision is None else file_hash(args.path_decision),
        "path_decision": path_evidence,
        "long_history_generator_evidence": generator_evidence,
        "fit_time_limits_seconds": caps, "timing_evidence": timing_evidence,
        "existing_reference_timing_sha256": file_hash(args.measurements) if caps is not None else None,
        "profile_identities": {name: p.identity for name, p in profiles.items()},
        "required_scientific_fit_calls": {"history_initial": 330, "sensors_illustrative": 33,
                                          "revised_one_chosen_length": 220, "total": 583},
        "additional_training_only_preflight_fit_calls": 6,
        "calibration_additional_fit_calls": 0,
        "condition_execution_status": {"history_1000": "initial_required", "history_2000": "initial_required",
            "history_500": "deferred_evidence_trigger", "history_4000": "deferred_needs_new4300_row_timing_review",
            "revised": "execute_only_the_history_chosen_by_the_completed_development_curve_gate",
            "confirmation": "requires_independent_valid_freeze_before_final_seed_generation"},
        "primary_comparisons": default_primary_comparisons(profiles["confirmation_L2000"])}
    provenance["identity"] = content_hash(provenance)
    write_json(output/"study_provenance.json", provenance)
    write_json(output/"resource_plans.json", {"status": label, "provenance_identity": provenance["identity"], "plans": plans})
    if not args.planning_proxy:
        suffix = ".yaml" if execution_ready else ".draft.yaml"
        for name, p in profiles.items():
            p.save(output/(name+suffix))
        for kind in ("history", "sensors"):
            write_json(output/(kind+"_curve_design.json"), curve_design(profiles, recipe, kind))
        write_json(output/"primary_comparisons.json", provenance["primary_comparisons"])
        write_json(output/"preflight_jobs.json", preflight_jobs(draft, recipe))
    print(json.dumps({"status": label, "output": str(output.resolve()),
                      "resolved_profiles": len(profiles), "required_scientific_fit_calls": 583,
                      "source": provenance["source_fingerprint"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
