#!/usr/bin/env python3
"""Read-only paired-history decision and sensor representation diagnostics.

Uses completed development artifacts only. It never refits a model, generates
a dataset, recalibrates a prior, or evaluates confirmation observations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

from bayes_stesn.basis import SpatialMean
from bayes_stesn.benchmark.generators import Simulation
from bayes_stesn.benchmark.metrics import training_normalization
from bayes_stesn.benchmark.protocol import BenchmarkProtocol, effective_models
from bayes_stesn.benchmark.registry import MODELS
from bayes_stesn.benchmark.runner import source_fingerprint, write_json
from bayes_stesn.io import content_hash, load_state, seed_for


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT/"configs/benchmark/v03/studies/remaining_studies.yaml"
SOURCE_INPUTS = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_json(path, key=None):
    row = json.loads(record_source(path))
    if key:
        require(row.get(key) == content_hash({k: v for k, v in row.items() if k != key}),
                f"Invalid content identity: {path}")
    return row


def record_source(path):
    path = Path(path).resolve()
    payload = path.read_bytes()
    label = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    SOURCE_INPUTS[label] = {"path": label, "sha256": hashlib.sha256(payload).hexdigest(),
                            "bytes": len(payload)}
    return payload


def completed_run(root, protocol):
    path = Path(root)/protocol.name
    manifest = verified_json(path/"manifest.json", "identity")
    summary = verified_json(path/"summary.json")
    require(protocol.stage == "development" and manifest["stage"] == "development",
            "Learning-curve selection cannot consume confirmation data")
    require(manifest["protocol_identity"] == protocol.identity
            and summary["manifest_identity"] == manifest["identity"], "Curve/run protocol identity differs")
    require(summary.get("complete") and summary.get("source_stable")
            and manifest["source_fingerprint"] == source_fingerprint(),
            "Curve requires complete source-stable development under the current package")
    return path, manifest


def read_job(path, manifest):
    row = verified_json(path/"receipt.json", "receipt_identity")
    require(row["manifest_identity"] == manifest["identity"] and row["terminal_scientific_result"],
            "Nonterminal or mismatched curve job")
    for artifact in row["artifacts"]:
        member = Path(artifact["file"])
        require(not member.is_absolute() and ".." not in member.parts, "Unsafe artifact path")
        target = path/member
        require(target.stat().st_size == artifact["bytes"] and file_hash(target) == artifact["sha256"],
                "Curve artifact differs from its receipt")
    return row


def paired_macro(differences, *, seed, resamples=5000):
    """Equal independent-scenario strata; resample whole paired datasets."""
    arrays = [np.asarray(values, float) for _, values in sorted(differences.items())]
    require(arrays and all(a.ndim == 1 and len(a) >= 2 and np.isfinite(a).all() for a in arrays),
            "A macro comparison needs at least two finite independent pairs per stratum")
    rng = np.random.default_rng(seed)
    weight = 1/len(arrays)
    estimate = float(sum(weight*a.mean() for a in arrays))
    mcse = float(np.sqrt(sum(weight**2*a.var(ddof=1)/len(a) for a in arrays)))
    bootstrap = sum(weight*a[rng.integers(0, len(a), size=(resamples, len(a)))].mean(axis=1) for a in arrays)
    return {"difference": estimate, "mcse": mcse,
            "lower_one_sided95": float(np.quantile(bootstrap, .05)),
            "upper_one_sided95": float(np.quantile(bootstrap, .95)),
            "pointwise95_interval": np.quantile(bootstrap, (.025, .975)).tolist(),
            "pair_counts": {key: len(value) for key, value in differences.items()},
            "paired_differences": {key: list(map(float, value)) for key, value in differences.items()},
            "scenario_weights": {key: weight for key in differences}, "bootstrap_resamples": resamples,
            "contrast_sign": "1000 minus 2000; positive favors the longer history",
            "interpretation": "exploratory development selection with small-sample stratified percentile bounds; no formal coverage or simultaneous superiority claim"}


def apply_history_gate(contrasts, qualification, times, complete_costs, gate):
    """Apply the prospective ESN-centered practical rule, without adaptation."""
    primary = [row for row in contrasts if row["is_decision_contrast"]]
    require({row["track"] for row in primary} == set(gate["tracks"])
            and len(primary) == len(gate["tracks"]), "Both declared ESN tracks must be present")
    require(all(np.isfinite(value) and value > 0 for value in times.values()),
            "Full-pipeline time sums must be finite and positive")
    score_pass = all(row["sufficient_pairs"] and row["bound_passed"] for row in primary)
    qualification_pass = all(row["not_worse"] for row in qualification)
    ratio = times[2000]/times[1000]
    cost_pass = bool(complete_costs and ratio <= gate["maximum_measured_long_to_short_pipeline_time_ratio"])
    reasons = []
    if not all(row["sufficient_pairs"] for row in primary):
        reasons.append("Fewer than the required common-qualified pairs in at least one ESN scenario/track cell")
    if not score_pass:
        reasons.append("Both ESN tracks did not establish the declared practical improvement lower bound")
    if not qualification_pass:
        reasons.append("Long-history qualification is lower in at least one ESN scenario/track cell")
    if not complete_costs:
        reasons.append("Full-pipeline compute accounting is incomplete")
    if ratio > gate["maximum_measured_long_to_short_pipeline_time_ratio"]:
        reasons.append("The recorded all-roster pipeline time ratio exceeds the declared resource preference")
    promoted = bool(score_pass and qualification_pass and cost_pass)
    return {"selected_initial_history": 2000 if promoted else 1000,
            "default_without_sufficient_evidence": 1000, "promotion_gate_passed": promoted,
            "retention_reasons": reasons, "score_bounds_passed": score_pass,
            "score_bounds_definition": "Both ESN tracks: lower one-sided95 bound for normalized native CRPS(1000)-CRPS(2000) >=0.02 with sufficient pairs",
            "qualification_not_worse": qualification_pass, "cost_gate_passed": cost_pass,
            "complete_compute_accounting": bool(complete_costs), "summed_pipeline_seconds": times,
            "long_to_short_pipeline_time_ratio": ratio,
            "short_history_time_saving_fraction": 1-times[1000]/times[2000]}


def history_decision(design_path, recipe):
    design = verified_json(design_path, "identity")
    require(design["kind"] == "history" and design["values"] == [500, 1000, 2000, 4000],
            "History decision requires the complete four-length declared curve")
    root = Path(design_path).parent
    profiles = {p.initial_fit-p.training_start: p for p in (BenchmarkProtocol(**r) for r in design["profiles"])}
    short, long = profiles[1000], profiles[2000]
    ignore = {"name", "training_start"}
    require({k: v for k, v in short.to_dict().items() if k not in ignore}
            == {k: v for k, v in long.to_dict().items() if k not in ignore},
            "History comparison changes more than its supplied start row")
    require(short.replications == 5 and short.initial_fit == 4000 and short.horizon == 100,
            "Unexpected paired history geometry or replication count")
    require(tuple(short.scenarios) == tuple(recipe["history"]["scenarios"]), "Unexpected history mechanisms")
    locations = {1000: completed_run(root, short), 2000: completed_run(root, long)}
    jobs, field_hashes = {}, []
    for scenario in short.scenarios:
        for replication in range(short.replications):
            simulations = []
            for length, (location, manifest) in locations.items():
                base = location/scenario/f"rep_{replication:03d}"
                data = verified_json(base/"data_receipt.json", "identity")
                require(data["manifest_identity"] == manifest["identity"]
                        and file_hash(base/"simulation.npz") == data["sha256"], "Curve data changed")
                sim = Simulation.load(base/"simulation.npz")
                simulations.append(sim)
                p = short if length == 1000 else long
                scale, normalization = training_normalization(sim, p)
                for track in short.tracks:
                    for model in effective_models(short, track):
                        row = read_job(base/track/model, manifest)
                        require(row["normalization_reference"] == normalization
                                and np.isclose(row["training_scale"], scale, rtol=1e-12),
                                "Saved training-only normalization does not reproduce")
                        jobs[length, scenario, replication, track, model] = row
            a, b = simulations
            for name in ("observation", "coordinates", "times", "observed_indices", "calibration_indices", "test_indices"):
                require(np.array_equal(getattr(a, name), getattr(b, name)), f"Paired field differs on {name}")
            field_hashes.append({"scenario": scenario, "replication": replication,
                                 "observation_hash": content_hash(a.observation)})
    gate = recipe["history_choice_gate"]
    contrasts, qualification = [], []
    all_models = sorted({m for t in short.tracks for m in effective_models(short, t)})
    for metric, source in (("crps", "native_scores"), ("rmse", "native_scores"),
                           ("rmse", "operational_scores")):
        for model in all_models:
            for track in short.tracks:
                if model not in effective_models(short, track) or (metric == "crps" and MODELS[model].uncertainty == "point"):
                    continue
                primary = model == gate["decision_model"] and metric == "crps" and source == "native_scores"
                differences, counts = {}, []
                for scenario in short.scenarios:
                    values, usable = [], []
                    qualified_short = qualified_long = 0
                    for replication in range(5):
                        a = jobs[1000, scenario, replication, track, model]
                        b = jobs[2000, scenario, replication, track, model]
                        require(a["normalization_reference"]["identity"] == b["normalization_reference"]["identity"]
                                and np.isclose(a["training_scale"], b["training_scale"], rtol=1e-12),
                                "History comparison has inconsistent score scale")
                        qa = bool(a["qualified"] and not a["fallback"])
                        qb = bool(b["qualified"] and not b["fallback"])
                        qualified_short += qa
                        qualified_long += qb
                        if source == "native_scores" and not (qa and qb and metric in a[source] and metric in b[source]):
                            continue
                        values.append((a[source][metric]-b[source][metric])/a["training_scale"])
                        usable.append(replication)
                    differences[scenario] = values
                    counts.append({"scenario": scenario, "usable_replications": usable,
                                   "qualified1000": qualified_short, "qualified2000": qualified_long,
                                   "paired_differences": values,
                                   "mean_difference": float(np.mean(values)) if values else None,
                                   "mcse": float(np.std(values, ddof=1)/np.sqrt(len(values))) if len(values) > 1 else None})
                minimum = gate["minimum_common_qualified_pairs_per_native_cell"] if source == "native_scores" else 5
                sufficient = all(len(v) >= minimum for v in differences.values())
                result = paired_macro(differences, seed=seed_for(gate["bootstrap_seed"], source, metric, model, track),
                                      resamples=gate["bootstrap_resamples"]) if all(len(v) >= 2 for v in differences.values()) else None
                margin = gate["minimum_normalized_crps_improvement"] if primary else None
                passed = bool(result and sufficient and result["lower_one_sided95"] >= margin) if primary else None
                contrasts.append({"model_id": model, "track": track, "metric": metric,
                    "score_family": source, "is_decision_contrast": primary,
                    "population": "common-qualified short/long" if source == "native_scores" else "all completed operational pairs",
                    "sufficient_pairs": sufficient, "margin": margin, "bound_passed": passed,
                    "cells": counts, "macro": result})
                if primary:
                    qualification.extend({"model_id": model, "track": track, **cell,
                        "not_worse": cell["qualified2000"] >= cell["qualified1000"]} for cell in counts)
    times = {length: sum(r["total_pipeline_seconds"] for key, r in jobs.items() if key[0] == length)
             for length in (1000, 2000)}
    complete_costs = all(r.get("compute_accounting_complete", False) for r in jobs.values())
    decision = apply_history_gate(contrasts, qualification, times, complete_costs, gate)
    return {"schema": "v03-history-choice-1.0", "curve_identity": design["identity"],
            **decision,
            "contrasts": contrasts, "qualification_cells": qualification, "paired_field_hashes": field_hashes,
            "decision_rule": gate, "confirmation_outcomes_inspected": False,
            "all_model_data_benefit_reported": True,
            "scientific_interpretation": "Exploratory development resource/specification choice. Native effects condition on pairwise qualification. Retaining 1000 with insufficient evidence is not a finding of no data benefit; no formal coverage or simultaneous superiority is established."}


def matrix_diagnostics(matrix):
    matrix = np.asarray(matrix, float)
    singular = np.linalg.svd(matrix, compute_uv=False)
    cutoff = float(max(matrix.shape)*np.finfo(float).eps*singular[0]) if len(singular) else 0.
    positive = singular[singular > cutoff]
    rank = len(positive)
    return {"shape": list(matrix.shape), "singular_values": singular.tolist(), "rank": rank,
            "rank_tolerance": cutoff, "rank_rule": "max(shape)*float64_epsilon*largest_singular_value",
            "full_column_rank": rank == matrix.shape[1],
            "condition_full_column_space": float(positive[0]/positive[-1]) if rank == matrix.shape[1] else None,
            "condition_nonzero_subspace": float(positive[0]/positive[-1]) if rank else None,
            "condition_null_meaning": "rank deficient in the stated full column space"}


def reconstruction_diagnostics(phi_observed, phi_target, ridge=.01):
    require(np.isfinite(ridge) and ridge > 0, "Geometry diagnostic ridge must be positive")
    n = len(phi_observed)
    gram = phi_observed.T@phi_observed/n + ridge*np.eye(phi_observed.shape[1])
    operator = phi_target @ np.linalg.solve(gram, phi_observed.T/n)
    amplification = np.sum(operator*operator, axis=1)
    singular = np.linalg.svd(operator, compute_uv=False)
    return {"ridge": ridge, "criterion": "||y-Phi_O c||^2/n + ridge*||c||^2, including the intercept",
            "target_unit_iid_noise_variance_amplification": amplification.tolist(),
            "mean_unit_iid_noise_variance_amplification": float(amplification.mean()),
            "maximum_unit_iid_noise_variance_amplification": float(amplification.max()),
            "worst_direction_variance_amplification": float(singular[0]**2),
            "regularized_gram_condition": float(np.linalg.cond(gram)),
            "operator_hash": content_hash(operator),
            "interpretation": "Geometry-only normalized-ridge reconstruction diagnostic; not correlated-residual forecast variance or a forecast score"}


def sensor_representations(design_path, *, ridge=.01):
    design = verified_json(design_path, "identity")
    require(design["kind"] == "sensors" and design["values"] == [25, 50, 100],
            "Representation audit requires the declared nested 25/50/100 design")
    root = Path(design_path).parent
    profiles = {p.n_observed: p for p in (BenchmarkProtocol(**r) for r in design["profiles"])}
    runs = {n: completed_run(root, p) for n, p in profiles.items()}
    geometry, prepared, receipts = {}, {}, {}
    for count, (location, manifest) in runs.items():
        p = profiles[count]
        require(p.replications == 1 and p.scenarios == ("g4_existing_fronts",),
                "This sensor audit is the prespecified one-field G4 sensitivity")
        base = location/"g4_existing_fronts"/"rep_000"
        data = verified_json(base/"data_receipt.json", "identity")
        require(data["manifest_identity"] == manifest["identity"]
                and file_hash(base/"simulation.npz") == data["sha256"], "Sensor dataset artifact changed")
        sim = Simulation.load(base/"simulation.npz")
        geometry[count] = {"observed_coordinates": np.array(sim.coordinates[sim.observed_indices]),
                           "observed_ids": [sim.site_ids[j] for j in sim.observed_indices],
                           "target_coordinates": np.array(sim.coordinates[sim.test_indices]),
                           "target_ids": [sim.site_ids[j] for j in sim.test_indices],
                           "field_hash": content_hash({"observations": sim.observation,
                                                       "coordinates": sim.coordinates, "times": sim.times})}
        for track in p.tracks:
            directory = base/track/"bayes_esn_ridge"
            row = read_job(directory, manifest)
            receipts[count, track] = row
            path = directory/"model.npz"
            if not path.exists():
                prepared[count, track] = None
                continue
            with zipfile.ZipFile(path) as archive:
                require(sum(z.file_size for z in archive.infolist()) <= 256*1024**2,
                        "Saved ESN exceeds the bounded representation-audit memory limit")
            # Read numeric state directly; do not rebuild a posterior or run inference.
            state = load_state(path, kind="benchmark_model")
            require(state["model_id"] == "bayes_esn_ridge", "Expected the declared ridge ESN")
            prepared[count, track] = state["fitted"]["prepared"]
    anchor_geometry = geometry[100]
    for count, values in geometry.items():
        require(values["field_hash"] == anchor_geometry["field_hash"]
                and values["target_ids"] == anchor_geometry["target_ids"]
                and values["observed_ids"] == anchor_geometry["observed_ids"][:count],
                "Sensor conditions are not the same field, nested O and fixed U")
    rows = []
    for track in profiles[100].tracks:
        anchor = prepared[100, track]
        fixed_basis = None if anchor is None else SpatialMean.from_state(anchor["basis"])
        fixed_hash = None if anchor is None else content_hash(anchor["basis"])
        for count in (25, 50, 100):
            p = prepared[count, track]
            receipt = receipts[count, track]
            row = {"sensor_count": count, "track": track, "receipt_identity": receipt["receipt_identity"],
                   "qualified": receipt["qualified"], "fallback": receipt["fallback"],
                   "observed_ids": geometry[count]["observed_ids"], "target_ids": geometry[count]["target_ids"],
                   "observed_coordinates": geometry[count]["observed_coordinates"].tolist(),
                   "target_coordinates": geometry[count]["target_coordinates"].tolist(),
                   "reference_basis_sensor_count": 100, "reference_basis_hash": fixed_hash}
            if p is None:
                row.update(status="saved_ESN_model_unavailable", temporal_rank=None, spatial_rank=None)
            else:
                basis = SpatialMean.from_state(p["basis"])
                require(np.allclose(basis.transform(geometry[count]["observed_coordinates"]), p["Phi"], rtol=1e-12, atol=1e-12),
                        "Saved Phi does not match its original basis transformation")
                split = p["split"]
                z = p["Z"][split["fitting_start"]:split["fitting_stop"]]
                row.update(status="available", spatial=matrix_diagnostics(p["Phi"]), temporal=matrix_diagnostics(z),
                           supplied_rows=len(p["Z"]), conditioning_rows=split["conditioning"], likelihood_rows=len(z),
                           coordinate_units=p["data"]["coordinate_units"],
                           fitted_basis_hash=content_hash(p["basis"]),
                           fitted_basis_transformation={k: np.asarray(v).tolist() if isinstance(v, np.ndarray) else v
                                                        for k, v in p["basis"].items()},
                           fitted_basis_matches_fixed_reference=content_hash(p["basis"]) == fixed_hash)
            if fixed_basis is not None:
                phi_o = fixed_basis.transform(geometry[count]["observed_coordinates"])
                phi_u = fixed_basis.transform(geometry[count]["target_coordinates"])
                row["fixed_reference_noise_amplification"] = reconstruction_diagnostics(phi_o, phi_u, ridge)
            rows.append(row)
    return {"schema": "v03-sensor-representation-audit-1.0", "curve_identity": design["identity"],
            "independent_fields": 1, "replication_mcse": None, "rows": rows,
            "basis_policy": "Fitted Phi uses its original sensor-specific geometry normalization. The diagnostic alone fixes the 100-sensor basis transformation within each track; no fit, normalization or prior is changed.",
            "interpretation": "One-field descriptive geometry/representation sensitivity; no replicated sensor-benefit or forecast-accuracy claim."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("history", "sensors"))
    parser.add_argument("--design", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reconstruction-ridge", type=float, default=.01)
    args = parser.parse_args(argv)
    require(not Path(args.output).resolve().is_relative_to(Path(args.design).resolve().parent),
            "Write the audit outside its immutable source curve directory")
    SOURCE_INPUTS.clear()
    recipe = yaml.safe_load(record_source(RECIPE))
    record_source(__file__)
    with threadpool_limits(limits=1):
        result = history_decision(args.design, recipe) if args.kind == "history" else sensor_representations(
            args.design, ridge=args.reconstruction_ridge)
    for item in SOURCE_INPUTS.values():
        path = Path(item["path"])
        path = path if path.is_absolute() else ROOT/path
        require(path.stat().st_size == item["bytes"] and file_hash(path) == item["sha256"],
                "An input changed during the read-only curve audit")
    result.update(source_fingerprint=source_fingerprint(), tool_sha256=file_hash(__file__),
                  recipe_sha256=file_hash(RECIPE), design_sha256=file_hash(args.design),
                  source_inputs=sorted(SOURCE_INPUTS.values(), key=lambda row: row["path"]))
    result["identity"] = content_hash(result)
    write_json(args.output, result)
    print(json.dumps({"output": str(Path(args.output).resolve()), "schema": result["schema"],
                      "selected_initial_history": result.get("selected_initial_history"), "model_refits": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
