"""Render a provenance-checked v0.3 study synthesis from saved artifacts only.

No model fitting, forecasting, selection, or simulation is performed. Optional
unexecuted studies are visible gaps; incomplete supplied benchmark runs fail.
Use --check-inputs before PDF authoring. See report/v03/README.md.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil
from statistics import NormalDist
import sys
from xml.etree import ElementTree
import zipfile

import numpy as np

from bayes_stesn.io import content_hash, load_state, seed_for


ROOT = Path(__file__).resolve().parents[1]
NAVY, TEAL, GRAY, PALE = "#10283F", "#008C82", "#5C7083", "#EDF5F6"
COLORS = {"persistence": "#94A3B8", "seasonal_naive": "#64748B",
          "spatial_persistence": "#94A3B8", "basis_lgssm": "#8066C2",
          "separable_stgp": "#2563AB", "basis_krr": "#BA8127",
          "bayes_esn_ridge": TEAL, "eqesn_rollout": "#D77851",
          "dcrnn": "#B64F83", "basis_gru": "#46764F"}
LABELS = {"persistence": "Persistence", "seasonal_naive": "Seasonal naive",
          "spatial_persistence": "Spatial persistence", "basis_lgssm": "Basis LGSSM",
          "separable_stgp": "Separable GP", "basis_krr": "Basis KRR",
          "bayes_esn_ridge": "Bayesian ESN", "eqesn_rollout": "Quadratic ESN ensemble",
          "dcrnn": "Diffusion GRU", "basis_gru": "Basis GRU"}
PLOT_LABELS = {**LABELS, "eqesn_rollout": "Quadratic ESN", "spatial_persistence": "Spatial persist."}
SCENARIOS = {"g1_gaussian_ar": "G1 / Gaussian spatial AR",
             "g2_spectral_transport": "G2 / Spectral transport",
             "g3_excitable_field": "G3 / Excitable field",
             "g4_existing_fronts": "G4 / Moving fronts",
             "g5_noise_stress": "G5 / Noise stress",
             "g6_velocity_shift": "G6 / Hidden velocity shift"}
FOCUS = ("bayes_esn_ridge", "separable_stgp", "basis_gru", "eqesn_rollout",
         "persistence", "spatial_persistence")
STRESS_ARMS = {"whole_field_t": "G5 / Whole-field Student-t noise",
               "local_t": "G5 / Local Student-t noise", "velocity_shift": "G6 / Hidden velocity shift"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=_reject_constant, object_pairs_hook=_unique_keys)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                   allow_nan=False) + "\n", encoding="utf-8")


def identity(value, key="identity", label="Receipt"):
    if value.get(key) != content_hash({k: v for k, v in value.items() if k != key}):
        raise ValueError(f"{label} content identity does not match")


def safe_member(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Unsafe artifact path: {relative}")
    path = root / relative
    if any(part.is_symlink() for part in [path, *path.parents] if part != root.parent):
        raise ValueError(f"Symlinked artifact path: {relative}")
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or escaping artifact: {relative}")
    return path


@dataclass
class Inputs:
    """An immutable-in-use input inventory; hashing is streamed, never buffered."""
    files: dict = field(default_factory=dict)

    def add(self, path, *, expected=None, size=None):
        path = Path(path).resolve()
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f"Artifact hash mismatch: {path}")
        if size is not None and path.stat().st_size != size:
            raise ValueError(f"Artifact byte count mismatch: {path}")
        if str(path) in self.files and self.files[str(path)]["sha256"] != digest:
            raise ValueError(f"Input changed during validation: {path}")
        self.files[str(path)] = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}
        return path

    def json(self, path):
        path = self.add(path)
        value = read_json(path)
        if sha256(path) != self.files[str(path)]["sha256"]:
            raise ValueError(f"Input changed while reading: {path}")
        return value

    def verify_unchanged(self):
        for item in self.files.values():
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError(f"Input changed during rendering: {item['path']}")


def bound_inputs(rows, inputs):
    paths = set()
    for row in rows:
        path = Path(row["path"])
        path = (path if path.is_absolute() else ROOT / path).resolve()
        if path in paths:
            raise ValueError("Bound source inventory contains duplicate files")
        paths.add(path)
        inputs.add(path, expected=row["sha256"], size=row["bytes"])
    return paths


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def _same_number(actual, expected, label):
    if expected is None:
        if actual is not None:
            raise ValueError(f"{label}: value should be unavailable")
    elif not _finite(actual) or not np.isclose(actual, expected, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{label}: summary disagrees with saved receipt values")


def validate_summary(summary, receipts):
    """Check reductions of saved scalars; never rescore observations or paths."""
    groups = defaultdict(list)
    for receipt in receipts:
        groups[(receipt["scenario"], receipt["track"], receipt["model_id"])].append(receipt)
    rows = summary.get("table", [])
    keys = [(r["scenario"], r["track"], r["model_id"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(groups):
        raise ValueError("Summary table does not contain exactly the saved scientific cells")
    for row, key in zip(rows, keys):
        saved = groups[key]
        native = [r for r in saved if r["qualified"] and not r["fallback"]
                  and "crps" in (r.get("native_scores") or {})]
        counts = {"scheduled": len(saved), "attempted": len(saved),
                  "qualified": sum(bool(r["qualified"]) for r in saved),
                  "fallback": sum(bool(r["fallback"]) for r in saved),
                  "completed": sum(bool(r["completed"]) for r in saved),
                  "operational_population": len(saved), "probability_population": len(native)}
        for name, count in counts.items():
            if row.get(name) != count:
                raise ValueError(f"Summary {name} population disagrees in {key}")
        for metric, population, field_name in (("rmse", saved, "operational_scores"),
                                               ("mae", saved, "operational_scores"),
                                               ("crps", native, "native_scores"),
                                               ("field_energy", native, "native_scores"),
                                               ("spatial_variogram", native, "native_scores")):
            values = [r[field_name][metric] for r in population if metric in r[field_name]]
            if not all(_finite(v) for v in values):
                raise ValueError(f"Non-finite saved {metric}")
            _same_number(row.get(metric), float(np.mean(values)) if values else None, f"{key}/{metric}")
            _same_number(row.get(metric + "_mcse"),
                         float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
                         f"{key}/{metric} MCSE")
        for output_name, field_name in (("coverage_90", "coverage"), ("width_90", "width"),
                                         ("interval_score_90", "interval_score")):
            values = [r["native_scores"].get("intervals", {}).get("0.9", {}).get(field_name) for r in native]
            values = [v for v in values if v is not None]
            _same_number(row.get(output_name), float(np.mean(values)) if values else None,
                         f"{key}/{output_name}")
    lookup = {(r["scenario"], r["track"], r["model_id"], r["replication"]): r for r in receipts}
    for pair in summary.get("paired_comparisons", []):
        scenario, track = pair["scenario"], pair["track"]
        reference, other = pair.get("reference", "bayes_esn_ridge"), pair["other_model"]
        shared = sorted({r["replication"] for r in receipts if (r["scenario"], r["track"], r["model_id"]) == (scenario, track, reference)}
                        & {r["replication"] for r in receipts if (r["scenario"], r["track"], r["model_id"]) == (scenario, track, other)})
        if pair.get("paired_attempted_population") != len(shared):
            raise ValueError("Paired attempted population differs from complete receipt roster")
        for metric in ("rmse", "crps"):
            ids = pair.get(metric + "_replications", [])
            eligible = shared if metric == "rmse" else [r for r in shared if all(
                lookup[(scenario, track, model, r)]["qualified"]
                and not lookup[(scenario, track, model, r)]["fallback"]
                and metric in lookup[(scenario, track, model, r)]["native_scores"] for model in (reference, other))]
            if ids != eligible:
                raise ValueError("Paired summary omitted or added eligible datasets")
            if len(ids) != pair.get(metric + "_population") or len(ids) != len(set(ids)):
                raise ValueError("Paired comparison population/replications disagree")
            expected = []
            for replication in ids:
                a = lookup[(scenario, track, reference, replication)]
                b = lookup[(scenario, track, other, replication)]
                key = "native_scores" if metric == "crps" else "operational_scores"
                if metric == "crps" and not (a["qualified"] and b["qualified"]):
                    raise ValueError("Unqualified fit entered a native paired comparison")
                _same_number(a["training_scale"], b["training_scale"], "Paired training scale")
                expected.append((a[key][metric] - b[key][metric]) / a["training_scale"])
            actual = pair.get(metric, {}).get("paired_differences", [])
            if len(actual) != len(expected) or not np.allclose(actual, expected, rtol=1e-10, atol=1e-12):
                raise ValueError("Paired summary differs from saved normalized receipt losses")
            _same_number(pair.get(metric, {}).get("difference"),
                         float(np.mean(expected)) if expected else None, "Paired mean")


def load_run(path, label, inputs, current_source):
    from bayes_stesn.benchmark.protocol import BenchmarkProtocol, effective_models

    path = Path(path).resolve()
    if (path / "invalidation.json").exists():
        raise ValueError(f"Invalidated study cannot be rendered as complete: {path}")
    manifest = inputs.json(path / "manifest.json")
    summary = inputs.json(path / "summary.json")
    identity(manifest, label="Benchmark manifest")
    if manifest.get("schema") != "benchmark-run-4.0":
        raise ValueError("This synthesis requires schema-4 benchmark runs; historical reports remain separate")
    if manifest.get("source_fingerprint") != current_source:
        raise ValueError(f"Main run uses a different scientific source: {label}")
    if (summary.get("manifest_identity") != manifest["identity"] or summary.get("complete") is not True
            or summary.get("source_stable") is not True or summary.get("pending_jobs") != 0
            or summary.get("unresolved_infrastructure_failures") != 0):
        raise ValueError(f"A complete source-stable run with no unresolved jobs is required: {label}")
    if summary.get("final_source_fingerprint") != manifest["source_fingerprint"]:
        raise ValueError("Final source fingerprint disagrees with the sealed manifest")
    protocol = BenchmarkProtocol(**manifest["protocol"])
    if manifest.get("protocol_identity") != protocol.identity:
        raise ValueError("Protocol identity mismatch")
    design = content_hash({"protocol": protocol.to_dict(), "confirmation_freeze": manifest.get("confirmation_freeze")})
    computation = content_hash({"design_identity": design, "source_fingerprint": current_source,
        "environment": manifest["environment"], "algorithm_schema": "benchmark-computation-4.0"})
    if manifest.get("design_identity") != design or manifest.get("computation_identity") != computation:
        raise ValueError("Scientific design/computation identity mismatch")
    if manifest.get("stage") == "confirmation":
        frozen = inputs.json(path / "confirmation_freeze.json")
        identity(frozen, label="Confirmation freeze")
        if frozen["identity"] != manifest.get("confirmation_freeze"):
            raise ValueError("Confirmation run lacks its matching frozen design")
    receipts, selections, data_ids, observation_hashes = [], [], {}, {}
    for scenario in protocol.scenarios:
        for replication in range(protocol.replications):
            folder = path / scenario / f"rep_{replication:03d}"
            data = inputs.json(folder / "data_receipt.json")
            identity(data, label="Dataset receipt")
            if data.get("manifest_identity") != manifest["identity"]:
                raise ValueError("Dataset belongs to another study")
            inputs.add(safe_member(folder, "simulation.npz"), expected=data["sha256"])
            data_ids[(scenario, replication)] = data["identity"]
            observation_hashes[f"{scenario}/{replication}"] = data["observation_hash"]
            for track in protocol.tracks:
                for model in effective_models(protocol, track):
                    job = folder / track / model
                    receipt = inputs.json(job / "receipt.json")
                    identity(receipt, "receipt_identity", "Scientific job receipt")
                    expected = {"manifest_identity": manifest["identity"], "computation_identity": computation,
                        "scenario": scenario, "track": track, "model_id": model, "replication": replication,
                        "data_identity": data["identity"], "terminal_scientific_result": True}
                    if any(receipt.get(k) != v for k, v in expected.items()):
                        raise ValueError(f"Scientific receipt identity mismatch: {job}")
                    if receipt.get("process_failure") or not receipt.get("operational_scores"):
                        raise ValueError("Infrastructure interruption is not a scientific result")
                    if bool(receipt["qualified"]) == bool(receipt["fallback"]):
                        raise ValueError("Scientific qualification/fallback flags are inconsistent")
                    if not _finite(receipt.get("training_scale")) or receipt["training_scale"] <= 0:
                        raise ValueError("Missing finite positive normalization scale")
                    artifacts = receipt.get("artifacts")
                    if not isinstance(artifacts, list):
                        raise ValueError("Scientific artifact inventory is missing")
                    names = [a["file"] for a in artifacts]
                    if len(names) != len(set(names)):
                        raise ValueError("Duplicate scientific artifact inventory entries")
                    for artifact in artifacts:
                        inputs.add(safe_member(job, artifact["file"]), expected=artifact["sha256"], size=artifact["bytes"])
                        if artifact["file"] == "selection.json":
                            selected = inputs.json(job / "selection.json")
                            identity(selected, "selection_identity", "Selection receipt")
                            if selected.get("job_identity") != receipt.get("job_identity"):
                                raise ValueError("Selection receipt belongs to a different scientific job")
                            if selected.get("selected") != receipt.get("fit_config"):
                                raise ValueError("Final configuration differs from its saved selection")
                            selections.append({"scenario": scenario, "replication": replication, "track": track,
                                "model_id": model, "mode": selected.get("selection_mode"), "status": selected.get("status"),
                                "effective_candidates": len(selected.get("candidate_hashes", [])),
                                "screen_evaluations": len(selected.get("trials", [])),
                                "finalist_rechecks": len(selected.get("rechecks", [])),
                                "eligible_finalists": selected.get("eligible_finalists"),
                                "selected_candidate_hash": selected.get("selected_candidate_hash"),
                                "selected": selected.get("selected"), "candidate_design": selected.get("candidate_design"),
                                "selection_identity": selected["selection_identity"]})
                    receipts.append(receipt)
    if summary.get("jobs") != len(receipts) or summary.get("scheduled_jobs") != len(receipts):
        raise ValueError("Completed study count disagrees with its declared design")
    validate_summary(summary, receipts)
    return {"label": label, "path": str(path), "manifest": manifest, "summary": summary,
            "receipts": receipts, "selections": selections, "data_identities": list(data_ids.values()),
            "data_observation_hashes": observation_hashes}


def load_audits(specs, runs, inputs, current_source):
    by_label = {run["label"]: run for run in runs}
    audits = []
    for label, path in specs:
        if label not in by_label:
            raise ValueError("Audit label must name a supplied run")
        run, path = by_label[label], Path(path).resolve()
        audit = inputs.json(path)
        identity(audit, label="Reference audit")
        if (audit.get("schema") != "v03-reference-audit-1.0" or audit.get("status") not in {"passed", "passed_with_findings"}
                or audit.get("manifest_identity") != run["manifest"]["identity"]
                or audit.get("source_fingerprint") != current_source
                or audit.get("audit_source_fingerprint") != current_source):
            raise ValueError("Audit does not match the completed current-source benchmark")
        inputs.add(Path(run["path"]) / "manifest.json", expected=audit["manifest_sha256"])
        inputs.add(Path(run["path"]) / "summary.json", expected=audit["summary_sha256"])
        found = set()
        protocol = run["manifest"]["protocol"]
        expected = {(rep, track) for rep in range(protocol["replications"]) for track in protocol["tracks"]} if "g1_gaussian_ar" in protocol["scenarios"] else set()
        for row in audit.get("oracle", []):
            key = (row["replication"], row["track"])
            if row["scenario"] != "g1_gaussian_ar" or key not in expected or key in found:
                raise ValueError("Known-parameter reference does not match the declared G1 datasets/tasks")
            found.add(key)
            inputs.add(safe_member(path.parent, row["moment_artifact"]), expected=row["moment_sha256"])
            for metric in ("rmse", "mae", "crps"):
                if not _finite(row["scores"][metric]):
                    raise ValueError("Known-parameter scores must be finite")
            for metric in ("rmse", "crps"):
                _same_number(row["normalized_" + metric], row["scores"][metric] / row["training_scale"], "Oracle normalized score")
        if found != expected:
            raise ValueError("Known-parameter audit omits a declared G1 dataset/task")
        audits.append({"label": label, "path": str(path), "document": audit})
    return audits


def load_calibrations(specs, runs, inputs, current_source):
    by_label = {run["label"]: run for run in runs}
    calibrations = []
    for label, directory in specs:
        if label not in by_label:
            raise ValueError("Calibration label must name a supplied base run")
        directory = Path(directory).resolve()
        seal = inputs.json(directory / "calibration_evidence.json")
        identity(seal, label="Calibration evidence seal")
        run = by_label[label]
        if (seal.get("schema") != "v03-calibration-evidence-1.0" or seal.get("complete") is not True
                or seal.get("base_manifest_identity") != run["manifest"]["identity"]
                or seal.get("source_fingerprint") != current_source):
            raise ValueError("Calibration seal does not bind a complete current-source base study")
        required = {"summary.json", "simulation.npz", "raw_forecast.npz", "calibration.npz", "bands.npz"}
        names = [item["file"] for item in seal["files"]]
        base_folder = f"{seal['scenario']}/rep_{seal['replication']:03d}"
        job_folder = f"{base_folder}/{seal['track']}/{seal['model_id']}"
        base_required = {"manifest.json", "summary.json", f"{job_folder}/model.npz", f"{job_folder}/receipt.json",
                         f"{base_folder}/simulation.npz", f"{base_folder}/data_receipt.json"}
        base_names = [item["file"] for item in seal["base_inputs"]]
        if set(names) != required or len(names) != len(required) or set(base_names) != base_required or len(base_names) != len(base_required):
            raise ValueError("Calibration seal omits or duplicates required base/scientific artifacts")
        if not all(seal.get("verification", {}).get(key) is True for key in (
                "original_prefix_exact", "same_late_evaluation_targets", "disjoint_calibration_blocks",
                "parameter_cutoff", "O_C_U_roles", "saved_raw_rmse_mae_crps_and_intervals", "saved_map_bands_and_calibrated_scores")):
            raise ValueError("Calibration seal lacks the required numerical/information checks")
        for item in seal["base_inputs"]:
            inputs.add(safe_member(run["path"], item["file"]), expected=item["sha256"], size=item["bytes"])
        for item in seal["files"]:
            inputs.add(safe_member(directory, item["file"]), expected=item["sha256"], size=item["bytes"])
        summary = inputs.json(directory / "summary.json")
        if summary.get("base_manifest_identity") != seal["base_manifest_identity"]:
            raise ValueError("Calibration numerical summary belongs to another base study")
        calibrations.append({"label": label, "path": str(directory), "seal": seal, "summary": summary})
    return calibrations


def load_diagnostics(paths, runs, inputs, current_source):
    results = []
    model_hashes = {a["sha256"] for run in runs for receipt in run["receipts"]
                    for a in receipt["artifacts"] if a["file"] == "model.npz"}
    for path in paths:
        path = Path(path).resolve()
        if not path.exists():
            results.append({"status": "not available", "path": str(path)})
            continue
        value = inputs.json(path)
        identity(value, label="Saved first-step diagnostic")
        if value.get("schema") != "saved-G3-first-step-review-1.0" or value.get("source_fingerprint") != current_source:
            raise ValueError("Unsupported or incompatible saved first-step diagnostic")
        matched_models, path_manifest, inventory = [], None, None
        for filename, digest in value["input_hashes"].items():
            member = Path(filename)
            if not member.is_absolute():
                member = ROOT / member
            inputs.add(member, expected=digest)
            if member.name == "model.npz":
                if digest in model_hashes:
                    matched_models.append(digest)
            elif member.name == "manifest.json":
                path_manifest = inputs.json(member)
                identity(path_manifest, label="Diagnostic path-study manifest")
            elif member.name == "output_inventory.json":
                inventory = inputs.json(member)
                identity(inventory, label="Diagnostic path-study inventory")
                for item in inventory["entries"]:
                    inputs.add(safe_member(member.parent, item["file"]), expected=item["sha256"], size=item["bytes"])
        if (len(matched_models) != 1 or not path_manifest or not inventory or inventory.get("complete") is not True
                or inventory.get("manifest_identity") != path_manifest["identity"]
                or inventory.get("mode") != "path_stability" or inventory.get("source_fingerprint") != current_source
                or path_manifest.get("baseline_sha256") != matched_models[0]
                or path_manifest.get("input_posterior_identity") != value["posterior_identity"]
                or path_manifest.get("source_fingerprint") != current_source):
            raise ValueError("Saved first-step diagnostic does not bind the supplied benchmark and fixed posterior")
        decoder, one = value["decoder"], value["one_step"]
        _same_number(decoder["first_lead_unavoidable_outside_span_rmse"] ** 2
                     + decoder["first_lead_within_span_readout_error_rmse64"] ** 2,
                     one["conditional_readout_mean64_rmse"] ** 2, "Diagnostic orthogonal MSE decomposition")
        _same_number(decoder["first_lead_outside_span_fraction_of_conditional_readout_mse64"],
                     (decoder["first_lead_unavoidable_outside_span_rmse"] / one["conditional_readout_mean64_rmse"]) ** 2,
                     "Diagnostic outside-span error fraction")
        for row in value["saved_path_leads"]:
            _same_number(row["conditional_empirical_mean_mcse_rms"], row["predictive_sd_rms"] / np.sqrt(512),
                         "Diagnostic conditional predictive mean MCSE")
        results.append({"status": "complete", "path": str(path), "document": value})
    return results


def _prior_model_metadata(path):
    """Read only frozen design/specification members, never posterior arrays."""
    with zipfile.ZipFile(path) as archive:
        if archive.getinfo("__manifest__.npy").file_size > 32 * 1024**2:
            raise ValueError("Prior-check model metadata exceeds the 32 MiB allowance")
        with np.load(path, allow_pickle=False) as arrays:
            manifest = json.loads(str(arrays["__manifest__"]), parse_constant=_reject_constant,
                                  object_pairs_hook=_unique_keys)
            if manifest.get("schema_version") != "1.0" or manifest.get("kind") != "fitted_model":
                raise ValueError("Prior check requires the bound portable fitted-model archive")
            state = manifest["state"]
            prepared = state["prepared"]
            names = ("Phi", "Z", "reference_Z")
            members = {key: prepared[key]["__array__"] for key in names}
            if sum(archive.getinfo(key + ".npy").file_size for key in members.values()) > 32 * 1024**2:
                raise ValueError("Prior-check frozen design exceeds the 32 MiB allowance")
            result = {key: arrays[member] for key, member in members.items()}
            result.update({key: prepared[key] for key in ("spec", "response_center", "response_scale", "split")})
            result["sampler"] = state["sampler"]
    return result


def load_prior_checks(paths, extension_studies, inputs, current_source):
    """Bind the saved conditional prior assessment; do not repeat quadrature or rollouts."""
    results = []
    for path in paths:
        path = Path(path).resolve()
        if not path.exists():
            results.append({"status": "not available", "path": str(path)})
            continue
        value = inputs.json(path)
        identity(value, label="Conditional prior check")
        if (value.get("schema") != "conditional-fixed-design-prior-check-1.0"
                or value.get("source_fingerprint") != current_source
                or any(value.get(key) is not False for key in ("posterior_samples_used", "future_labels_used",
                    "models_refitted", "prior_retuning_performed", "extension_promotion", "autonomous_prior_rollouts_assessed"))):
            raise ValueError("Prior check has an incompatible source or information scope")
        inputs.add(ROOT / "tools/check_conditional_prior_v03.py", expected=value["tool_sha256"])
        bound = set()
        for filename, digest in value["input_hashes"].items():
            member = Path(filename)
            member = (member if member.is_absolute() else ROOT / member).resolve()
            if member in bound:
                raise ValueError("Prior check duplicates a bound input")
            bound.add(member)
            inputs.add(member, expected=digest)
        required, seen = set(), set()
        if not 1 <= len(value["cases"]) <= 4:
            raise ValueError("Prior check requires one to four distinct bounded cases")
        for case in value["cases"]:
            directory = Path(case["ablation_directory"])
            directory = (directory if directory.is_absolute() else ROOT / directory).resolve()
            matched = [study for study in extension_studies
                       if (study.get("scientific_evidence") or {}).get("mode") == "extensions"
                       and Path(study["path"]).resolve().parent == directory
                       and study["document"].get("manifest_identity") == case["manifest_identity"]]
            if directory in seen or len(matched) != 1 or case.get("identical_frozen_design_and_shared_priors_verified") is not True:
                raise ValueError("Prior check must bind distinct supplied complete P5 ablations")
            seen.add(directory)
            manifest = inputs.json(directory / "manifest.json")
            identity(manifest, label="Prior-check P5 manifest")
            if (manifest["identity"] != case["manifest_identity"] or manifest.get("source_fingerprint") != current_source
                    or manifest.get("environment") != value["environment"]):
                raise ValueError("Prior check and saved P5 pair use different numerical environments")
            required.update(directory / name for name in ("manifest.json", "output_inventory.json",
                            "scalar_reference/fitted.npz", "full_residual_ar/fitted.npz"))
            if not required.issubset(bound):
                raise ValueError("Prior check inventory omits a required P5 pair input")
            states = [_prior_model_metadata(directory / variant / "fitted.npz")
                      for variant in ("scalar_reference", "full_residual_ar")]
            base, ar = states
            if (case["saved_scalar_sampler_temporal"] != "iid" or base["sampler"]["temporal"] != "iid"
                    or ar["sampler"]["temporal"] != "full_residual_ar"
                    or case["saved_ar_sampler_temporal"] != "full_residual_ar"):
                raise ValueError("Prior check mislabels the iid/full-residual-AR pair")
            for key in ("Phi", "Z", "reference_Z"):
                if not np.array_equal(base[key], ar[key]):
                    raise ValueError("Prior pair does not preserve the frozen feature design")
            for key in ("response_center", "response_scale", "split"):
                if base[key] != ar[key]:
                    raise ValueError("Prior pair changes its response units or fitting window")
            if any(base["spec"][key] != ar["spec"][key] for key in ("prior", "covariance", "features", "spatial_mean")):
                raise ValueError("Prior pair changes a shared specification")
            for key in ("a0", "b0"):
                if base["sampler"][key] != ar["sampler"][key]:
                    raise ValueError("Prior pair changes its inverse-gamma prior")
            saved, prior, covariance, sampler = case["results"], base["spec"]["prior"], base["spec"]["covariance"], ar["sampler"]
            if (prior["kind"] != "ridge" or covariance["kind"] != "matern"
                    or covariance["range_prior"] != "lognormal" or covariance["scale_mode"] != "relative"
                    or any(covariance[key] for key in ("anisotropy", "nonstationary", "static_variance", "infer_nu"))):
                raise ValueError("Prior assessment is outside its declared scalar-ridge covariance scope")
            parameters = {"a0": sampler["a0"], "b0": sampler["b0"],
                "log_precision_mean": prior["mean_log_precision"], "log_precision_sd": prior["sd_log_precision"],
                **{key: covariance[key] for key in ("log_q_mean", "log_q_sd", "log_ell_mean", "log_ell_sd", "nu")},
                "rho_eta_mean": 0., "rho_eta_sd": sampler["rho_prior_sd"]}
            for key, expected in parameters.items():
                _same_number(saved["prior_parameters"][key], expected, "Actual saved prior parameter " + key)
            if sampler["a0"] <= 1 or sampler["b0"] <= 0 or base["response_scale"] <= 0:
                raise ValueError("Prior assessment requires positive scale and finite variance")
            sigma = sampler["b0"] / (sampler["a0"] - 1)
            inverse_lambda = math.exp(-parameters["log_precision_mean"] + parameters["log_precision_sd"]**2 / 2)
            q = math.exp(parameters["log_q_mean"] + parameters["log_q_sd"]**2 / 2)
            for key, expected in (("response_center", base["response_center"]), ("response_scale", base["response_scale"]),
                    ("expected_sigma2_standardized", sigma), ("expected_inverse_lambda", inverse_lambda), ("expected_q", q),
                    ("marginal_coefficient_variance_standardized", sigma * inverse_lambda),
                    ("marginal_coefficient_variance_original_units", sigma * inverse_lambda * base["response_scale"]**2)):
                _same_number(saved[key], expected, "Prior moment / units " + key)
            if saved["predictive_variance_finite"] is not True or saved["predictive_fourth_moment_finite"] != (sampler["a0"] > 2):
                raise ValueError("Prior moment-existence claim disagrees with the inverse-gamma shape")
            expected_designs = {"conditioning_reference": base["reference_Z"],
                "observed_history_fitting_design": base["Z"][base["split"]["fitting_start"]:base["split"]["fitting_stop"]]}
            if {row["name"] for row in saved["designs"]} != set(expected_designs) or len(saved["designs"]) != 2:
                raise ValueError("Prior check omits or duplicates a declared frozen design")
            for row in saved["designs"]:
                z = expected_designs[row["name"]]
                if row["frozen_design_hash"] != content_hash({"Phi": base["Phi"], "Z": z}):
                    raise ValueError("Prior check frozen design hash differs from the fitted archive")
                if [row[key] for key in ("sites", "times", "spatial_columns", "temporal_columns")] != [
                        len(base["Phi"]), len(z), base["Phi"].shape[1], z.shape[1]]:
                    raise ValueError("Prior check misstates its frozen design dimensions")
                diagonal = sigma * inverse_lambda * np.outer(np.square(base["Phi"]).sum(axis=1), np.square(z).sum(axis=1))
                for key, variance in (("readout_pointwise_prior_sd", diagonal),
                                      ("observation_pointwise_prior_sd", diagonal + sigma * (1 + q))):
                    sd = np.sqrt(variance)
                    expected = {"minimum": sd.min(), "median": np.median(sd), "maximum": sd.max(), "rms": np.sqrt(variance.mean())}
                    for statistic, number in expected.items():
                        _same_number(row[key]["conditioning_standard_deviation_units"][statistic], number,
                                     "Prior fixed-design amplitude")
            sd_summaries = [saved["stationary_residual_prior_sd"]] + [row[key] for row in saved["designs"]
                            for key in ("readout_pointwise_prior_sd", "observation_pointwise_prior_sd")]
            for summary in sd_summaries:
                for key in ("minimum", "median", "maximum", "rms"):
                    _same_number(summary["original_response_units"][key],
                        summary["conditioning_standard_deviation_units"][key] * base["response_scale"], "Prior SD unit conversion")
            _same_number(saved["stationary_residual_prior_sd"]["conditioning_standard_deviation_units"]["rms"],
                         math.sqrt(sigma * (1 + q)), "Stationary residual prior amplitude")
            for threshold, probability in saved["ar_absolute_rho_tail_probabilities"].items():
                _same_number(probability, math.erfc(math.atanh(float(threshold)) / (sampler["rho_prior_sd"] * math.sqrt(2))),
                             "Signed-rho prior tail probability")
            _same_number(saved["ar_absolute_rho_median"], math.tanh(sampler["rho_prior_sd"] * NormalDist().inv_cdf(.75)),
                         "Signed-rho prior median absolute dependence")
            quadrature = saved["quadrature"]
            if (quadrature["passed"] is not True or quadrature["orders"] != [128, 256]
                    or any(not _finite(quadrature[key]) or not 0 <= quadrature[key] <= quadrature["absolute_agreement_tolerance"]
                           for key in ("expected_kernel_max_absolute_difference", "rho_moment_max_absolute_difference"))):
                raise ValueError("Saved prior quadrature lacks its declared agreement check")
        if bound != required:
            raise ValueError("Prior check inventory does not exactly cover its P5 pairs and fitted archives")
        results.append({"status": "complete", "path": str(path), "document": value})
    return results


def _small_numeric_state(path, inputs):
    inputs.add(path)
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 128 * 1024 ** 2:
            raise ValueError("Coordinate checkpoint exceeds the renderer's 128 MiB validation limit")
    return load_state(path)


def load_coordinate(path, inputs, current_source, allowed_source=None):
    path = Path(path).resolve()
    manifest, summary = inputs.json(path / "manifest.json"), inputs.json(path / "summary.json")
    identity(manifest, label="Coordinate manifest")
    source = manifest.get("source_fingerprint")
    if source != current_source and source != allowed_source:
        raise ValueError("Ancillary coordinate source differs; explicitly pass --allow-coordinate-source with its exact SHA")
    if (manifest.get("schema") != "sampler-coordinate-development-1.0" or
            summary.get("manifest_identity") != manifest["identity"] or summary.get("complete") is not True or
            summary.get("test_forecasts_computed") is not False or summary.get("accuracy_claim") is not False):
        raise ValueError("Coordinate evidence must be a complete algorithm-development comparison")
    expected = {(s, r, a) for s in manifest["scenarios"] for r in manifest["replications"]
                for a in ("legacy", "ridge_total_scale")}
    found = set()
    for row in summary["rows"]:
        key = (row["scenario"], row["replication"], row["parameterization"])
        if key not in expected or key in found or not row.get("complete") or row.get("manifest_identity") != manifest["identity"]:
            raise ValueError("Coordinate rows do not match the complete prespecified cases")
        found.add(key)
        folder = path / key[0] / f"rep_{key[1]:03d}" / key[2]
        if inputs.json(folder / "receipt.json") != row:
            raise ValueError("Coordinate summary differs from saved scientific receipt")
        state = _small_numeric_state(folder / "chain.npz.fit-state.npz", inputs)
        checkpoint = safe_member(folder, "chain.npz")
        inputs.add(checkpoint, expected=state["checkpoint"]["sha256"])
        _small_numeric_state(checkpoint, inputs)
        if state.get("phase") != "complete" or state.get("gates") != row["diagnostics"].get("gates"):
            raise ValueError("Coordinate fit-state is incomplete or disagrees with saved diagnostics")
    if found != expected:
        raise ValueError("Coordinate comparison has missing cases")
    for scenario in manifest["scenarios"]:
        for replication in manifest["replications"]:
            pair = [r for r in summary["rows"] if (r["scenario"], r["replication"]) == (scenario, replication)]
            for field_name in ("target_identity", "data_identity", "fit_seed"):
                if len({r[field_name] for r in pair}) != 1:
                    raise ValueError(f"Coordinate algorithms differ in {field_name}")
    return {"path": str(path), "manifest": manifest, "summary": summary,
            "ancillary_source_exception": source != current_source}


def load_curve(path, inputs, current_source):
    if path is None or not Path(path).exists():
        return {"status": "not supplied" if path is None else "not available", "path": None if path is None else str(path), "runs": []}
    path = Path(path).resolve()
    design_path = path / "curve_design.json" if path.is_dir() else path
    design = inputs.json(design_path)
    identity(design, label="Learning-curve design")
    if design.get("schema") != "benchmark-curve-2.0" or len(design["values"]) != len(design["profiles"]):
        raise ValueError("A resolved schema-2 matched curve design is required")
    runs, pending = [], []
    for value, profile in zip(design["values"], design["profiles"]):
        folder = design_path.parent / profile["name"]
        if not (folder / "summary.json").exists():
            pending.append(value)
            continue
        run = load_run(folder, f"{design['kind']}={value}", inputs, current_source)
        if run["manifest"]["protocol"] != profile:
            raise ValueError("Curve condition does not match its frozen design")
        run["value"] = value
        runs.append(run)
    if runs:
        first = runs[0]
        first_targets = {(r["scenario"], r["track"], r["model_id"], r["replication"]):
                         (r["origin_index"], r["target_ids"], r.get("normalization_reference"))
                         for r in first["receipts"]}
        for run in runs[1:]:
            if run["data_observation_hashes"] != first["data_observation_hashes"]:
                raise ValueError("Curve conditions do not contain identical generated fields")
            targets = {(r["scenario"], r["track"], r["model_id"], r["replication"]):
                       (r["origin_index"], r["target_ids"], r.get("normalization_reference"))
                       for r in run["receipts"]}
            if targets != first_targets:
                raise ValueError("Curve conditions do not share targets, origin and normalization reference")
    return {"status": "complete" if not pending else "partial", "path": str(design_path),
            "design": design, "runs": runs, "unexecuted_values": pending}


def load_curve_audits(paths, curves, inputs, current_source):
    results = []
    for path in paths:
        path = Path(path).resolve()
        if not path.exists():
            results.append({"status": "not available", "path": str(path)})
            continue
        value = inputs.json(path)
        identity(value, label="Learning-curve audit")
        kinds = {"v03-history-choice-1.0": "history", "v03-sensor-representation-audit-1.0": "sensors"}
        kind = kinds.get(value.get("schema"))
        matched = [curve for curve in curves if curve.get("design", {}).get("identity") == value.get("curve_identity")]
        if (kind is None or value.get("source_fingerprint") != current_source or len(matched) != 1
                or matched[0]["design"]["kind"] != kind):
            raise ValueError("Curve audit must match exactly one supplied current-source curve")
        curve = matched[0]
        inputs.add(curve["path"], expected=value["design_sha256"])
        supplied = set()
        for item in value["source_inputs"]:
            member = Path(item["path"])
            if not member.is_absolute():
                member = ROOT / member
            member = member.resolve()
            if member in supplied:
                raise ValueError("Curve audit source inventory contains duplicate files")
            supplied.add(member)
            inputs.add(member, expected=item["sha256"], size=item["bytes"])
        required = {Path(curve["path"]).resolve(), ROOT / "tools/summarize_v03_learning_curves.py",
                    ROOT / "configs/benchmark/v03/studies/remaining_studies.yaml"}
        values = (1000, 2000) if kind == "history" else (25, 50, 100)
        by_value = {run["value"]: run for run in curve["runs"] if run["value"] in values}
        if set(by_value) != set(values):
            raise ValueError("Curve audit requires all of its declared consumed conditions")
        for run in by_value.values():
            root = Path(run["path"])
            required.update((root / "manifest.json", root / "summary.json"))
            for row in run["receipts"]:
                base = root / row["scenario"] / f"rep_{row['replication']:03d}"
                required.add(base / "data_receipt.json")
                if kind == "history" or row["model_id"] == "bayes_esn_ridge":
                    required.add(base / row["track"] / row["model_id"] / "receipt.json")
        if not required.issubset(supplied):
            raise ValueError("Curve audit omits required design, recipe or scientific input receipts")
        inputs.add(ROOT / "tools/summarize_v03_learning_curves.py", expected=value["tool_sha256"])
        inputs.add(ROOT / "configs/benchmark/v03/studies/remaining_studies.yaml", expected=value["recipe_sha256"])
        if kind == "history":
            _validate_history_choice(value, by_value)
        else:
            if value.get("independent_fields") != 1 or value.get("replication_mcse") is not None:
                raise ValueError("Sensor representation audit overstates its single-field population")
            seen = set()
            for row in value["rows"]:
                key = (row["sensor_count"], row["track"])
                if key in seen or key[0] not in by_value:
                    raise ValueError("Sensor representation audit duplicates or invents a condition")
                seen.add(key)
                matches = [r for r in by_value[key[0]]["receipts"]
                           if r["track"] == key[1] and r["model_id"] == "bayes_esn_ridge"]
                if (len(matches) != 1 or matches[0]["receipt_identity"] != row["receipt_identity"]
                        or any(row.get(k) != matches[0][k] for k in ("qualified", "fallback"))):
                    raise ValueError("Sensor representation row differs from its saved fit/qualification")
                amplification = row.get("fixed_reference_noise_amplification")
                if amplification:
                    values_array = np.asarray(amplification["target_unit_iid_noise_variance_amplification"], float)
                    if (values_array.shape != (len(row["target_ids"]),) or not np.isfinite(values_array).all()
                            or np.any(values_array < 0)):
                        raise ValueError("Invalid geometry-only noise-amplification summary")
                    _same_number(amplification["mean_unit_iid_noise_variance_amplification"], values_array.mean(), "Sensor amplification mean")
                    _same_number(amplification["maximum_unit_iid_noise_variance_amplification"], values_array.max(), "Sensor amplification maximum")
            expected = {(n, track) for n, run in by_value.items() for track in run["manifest"]["protocol"]["tracks"]}
            if seen != expected:
                raise ValueError("Sensor representation audit omits a declared sensor/track condition")
        results.append({"status": "complete", "path": str(path), "kind": kind, "document": value})
    return results


def _stress_metric_powers():
    metrics = {name: 1 for name in ("rmse", "mae", "crps", "field_energy", "spatial_variogram")}
    for level in ("0.5", "0.8", "0.9", "0.95"):
        for name in ("coverage", "width", "interval_score", "field_coverage", "whole_path_covered"):
            metrics[f"intervals/{level}/{name}"] = int(name in {"width", "interval_score"})
    metrics.update({f"temporal_variogram/{lag}": 1 for lag in (1, 5, 20)})
    metrics.update({f"functional_crps/{name}": 1 for name in
                    ("path_spatial_mean", "final_spatial_mean", "path_maximum", "path_minimum")})
    return metrics


def _score_leaf(score, metric):
    for key in metric.split("/"):
        if not isinstance(score, dict) or key not in score:
            return None
        score = score[key]
    return float(score) if isinstance(score, (int, float, bool)) and np.isfinite(score) else None


def validate_stress_effects(arm, recipe):
    """Independently reduce declared paired losses, units and whole-dataset uncertainty."""
    effects = {(row["model_id"], row["track"], row["population"], row["metric"]): row for row in arm["effects"]}
    if len(effects) != len(arm["effects"]):
        raise ValueError("Stress audit duplicates a paired effect")
    expected_keys = set()
    for model, track in sorted({(row["model_id"], row["track"]) for row in arm["paired_jobs"]}):
        rows = [row for row in arm["paired_jobs"] if row["model_id"] == model and row["track"] == track]
        if len(rows) != 5 or [row["replication"] for row in rows] != list(range(5)):
            raise ValueError("Stress effect requires its five ordered independent paired fields")
        for population in ("native_scores", "operational_scores"):
            eligible = rows if population == "operational_scores" else [row for row in rows if
                row["control_qualified"] and row["stress_qualified"] and not row["control_fallback"] and not row["stress_fallback"]]
            for metric, power in _stress_metric_powers().items():
                if population == "operational_scores" and metric not in {"rmse", "mae"}:
                    continue
                if all(_score_leaf(row["control_scores"][population], metric) is None for row in rows):
                    continue
                key = (model, track, population, metric)
                expected_keys.add(key)
                if key not in effects:
                    raise ValueError("Stress audit omits a declared primary or companion effect")
                effect = effects[key]
                pairs, raw, normalized, scales = [], [], [], []
                for row in eligible:
                    a, b = (_score_leaf(row[name][population], metric) for name in ("control_scores", "stress_scores"))
                    if a is None or b is None:
                        continue
                    scale = row["control_training_scale"]
                    if not _finite(scale) or scale <= 0:
                        raise ValueError("Stress normalization requires a positive control training-only SD")
                    pairs.append(row["pair_id"])
                    raw.append(b - a)
                    normalized.append((b - a) / scale**power)
                    scales.append(scale)
                if (effect["pair_ids"] != pairs or effect["N_pair"] != len(pairs) or effect["attempted_pairs"] != 5
                        or effect["normalization_power"] != power
                        or effect["control_qualified_count"] != sum(row["control_qualified"] and not row["control_fallback"] for row in rows)
                        or effect["stress_qualified_count"] != sum(row["stress_qualified"] and not row["stress_fallback"] for row in rows)):
                    raise ValueError("Stress effect misstates its exact qualified population or normalization units")
                for field, expected in (("raw_differences", raw), ("normalized_differences", normalized), ("control_scales", scales)):
                    if len(effect[field]) != len(expected) or not np.allclose(effect[field], expected, rtol=1e-11, atol=1e-12):
                        raise ValueError("Stress effect differs from saved paired losses or the common control scale")
                bootstrap_seed = seed_for(recipe["inference"]["bootstrap_seed"], arm["id"], model, track, population, metric)
                resamples = recipe["inference"]["bootstrap_resamples"]
                if (effect["bootstrap_seed"] != bootstrap_seed or effect["bootstrap_resamples"] != resamples
                        or not isinstance(resamples, int) or not 1 <= resamples <= 10000):
                    raise ValueError("Stress paired bootstrap differs from its bounded declared rule")
                indices = np.random.default_rng(bootstrap_seed).integers(0, len(pairs), size=(resamples, len(pairs))) if len(pairs) > 1 else None
                for field, differences in (("raw", raw), ("common_normalized", normalized)):
                    array = np.asarray(differences, float)
                    stats = effect[field]
                    _same_number(stats["mean"], float(array.mean()) if len(array) else None, "Stress paired mean")
                    _same_number(stats["mcse"], float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else None, "Stress paired MCSE")
                    interval = np.quantile(array[indices].mean(axis=1), (.025, .975)) if indices is not None else None
                    if interval is None:
                        if stats["pointwise95"] is not None:
                            raise ValueError("Stress interval requires at least two paired datasets")
                    elif (not isinstance(stats["pointwise95"], list) or len(stats["pointwise95"]) != 2
                          or not np.allclose(stats["pointwise95"], interval, rtol=1e-11, atol=1e-12)):
                        raise ValueError("Stress pointwise interval disagrees with paired-dataset resampling")
    if set(effects) != expected_keys:
        raise ValueError("Stress audit invents undeclared effect populations or metrics")


def _sum_known(values):
    values = list(values)
    return sum(values) if all(_finite(value) and value >= 0 for value in values) else None


def _same_score_tree(actual, expected, *, rtol, atol):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError("G6 original-score replay changes a score object")
        for key in expected:
            _same_score_tree(actual[key], expected[key], rtol=rtol, atol=atol)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError("G6 original-score replay changes a score vector")
        for a, b in zip(actual, expected):
            _same_score_tree(a, b, rtol=rtol, atol=atol)
    elif isinstance(expected, (str, bool)) or expected is None:
        if actual != expected:
            raise ValueError("G6 original-score replay changes score metadata")
    elif not _finite(actual) or not np.isclose(actual, expected, rtol=rtol, atol=atol):
        raise ValueError("G6 original-score replay differs from a saved control score")


def _validate_stress_pair(pair, control, stress, arm):
    key = (control["replication"], control["track"], control["model_id"])
    if (tuple(pair[name] for name in ("replication", "track", "model_id")) != key
            or tuple(stress[name] for name in ("replication", "track", "model_id")) != key
            or pair["pair_id"] != f"rep_{key[0]:03d}/{key[1]}/{key[2]}" or pair["arm"] != arm
            or pair["control_receipt_identity"] != control["receipt_identity"]
            or pair["stress_receipt_identity"] != stress["receipt_identity"]
            or pair["control_normalization_reference"] != control["normalization_reference"]):
        raise ValueError("Stress pair changes its original receipt, role or training-normalization reference")
    _same_number(pair["control_training_scale"], control["training_scale"], "Stress common control scale")
    _same_number(pair["standalone_stress_training_scale"], stress.get("training_scale"), "Standalone stress scale provenance")
    for label, receipt in (("control", control), ("stress", stress)):
        if (any(pair[label + "_" + name] != receipt[name] for name in ("qualified", "fallback"))
                or pair[label + "_scores"] != {name: receipt[name] for name in ("native_scores", "operational_scores")}):
            raise ValueError("Stress pair changes its saved qualification or complete score objects")


def load_stress_audits(paths, runs, inputs, current_source):
    results = []
    for path in paths:
        path = Path(path).resolve()
        if not path.exists():
            results.append({"status": "not available", "path": str(path)})
            continue
        value = inputs.json(path)
        identity(value, label="Paired stress audit")
        if (value.get("schema") != "v03-paired-stress-summary-1.0" or value.get("complete") is not True
                or value.get("source_fingerprint") != current_source
                or value.get("standalone_g5_normalization_is_cross_arm_comparable") is not False):
            raise ValueError("Stress audit requires complete current-source paired results with a common control scale")
        matched = [run for run in runs if run["manifest"]["identity"] == value["control_manifest_identity"]]
        if len(matched) != 1:
            raise ValueError("Stress audit must bind exactly one supplied completed control run")
        control = matched[0]
        bound = bound_inputs(value["source_inputs"], inputs)
        design_path = Path(value["design_path"])
        design_path = (design_path if design_path.is_absolute() else ROOT / design_path).resolve()
        if design_path not in bound:
            raise ValueError("Stress audit omits its immutable design input")
        inputs.add(design_path, expected=value["design_sha256"])
        design = inputs.json(design_path)
        identity(design, label="Paired stress design")
        if (design.get("schema") != "v03-stress-design-1.0" or design["identity"] != value["design_identity"]
                or design.get("source_fingerprint") != current_source
                or design["control_manifest_identity"] != control["manifest"]["identity"]
                or design["control_protocol"] != control["manifest"]["protocol"]
                or design["environment"] != control["manifest"]["environment"]):
            raise ValueError("Stress audit differs from its bound control/source/design")
        design_bound = bound_inputs(design["source_inputs"], inputs)
        required_design = {ROOT / "tools/summarize_v03_stress.py", ROOT / "configs/benchmark/v03/studies/stress_recipe.yaml",
                           ROOT / "configs/benchmark/v03/studies/stress_prepare.py"}
        if not required_design.issubset(design_bound):
            raise ValueError("Stress design omits its immutable recipe and preparation tools")
        inputs.add(ROOT / "tools/summarize_v03_stress.py", expected=design["tool_sha256"])
        if design["recipe"]["replications"] != 5 or design["control_protocol"]["replications"] != 5:
            raise ValueError("This stress design requires five independent paired fields")
        documents = {member: inputs.json(member) for member in bound if member.suffix == ".json"}
        control_jobs = {(row["scenario"], row["replication"], row["track"], row["model_id"]): row for row in control["receipts"]}
        historical, seen_arms = {}, set()
        for arm in value["arms"]:
            name = arm["id"]
            if name not in STRESS_ARMS or name in seen_arms:
                raise ValueError("Stress audit duplicates or invents an intervention arm")
            seen_arms.add(name)
            control_scenario, stress_scenario = (("g2_spectral_transport", "g6_velocity_shift") if name == "velocity_shift"
                                                else ("g4_existing_fronts", "g5_noise_stress"))
            if (arm["control_scenario"], arm["stress_scenario"]) != (control_scenario, stress_scenario):
                raise ValueError("Stress arm is paired with the wrong control mechanism")
            expected = {key[1:]: row for key, row in control_jobs.items() if key[0] == control_scenario}
            pair_keys = [(pair["replication"], pair["track"], pair["model_id"]) for pair in arm["paired_jobs"]]
            if len(expected) != 55 or len(pair_keys) != 55 or set(pair_keys) != set(expected):
                raise ValueError("Stress audit omits or duplicates a declared model/task/dataset pair")
            costs, generation = arm["costs"], []
            if name != "velocity_shift":
                manifests = [member for member, document in documents.items() if member.name == "manifest.json"
                             and document.get("identity") == arm["run_manifest_identity"]]
                if len(manifests) != 1:
                    raise ValueError("Stress arm does not bind one complete benchmark manifest")
                stress_run = load_run(manifests[0].parent, "stress_" + name, inputs, current_source)
                if stress_run["manifest"]["protocol"] != design["g5_profiles"][name]:
                    raise ValueError("Stress arm differs from its prespecified frozen profile")
                stress_jobs = {(row["replication"], row["track"], row["model_id"]): row for row in stress_run["receipts"]}
            else:
                progress = [(member, document) for member, document in documents.items()
                            if document.get("schema") == "v03-g6-replay-progress-1.0"]
                if (len(progress) != 1 or progress[0][1].get("complete") is not True
                        or progress[0][1].get("jobs") != 55 or progress[0][1].get("new_parameter_fits") != 0
                        or progress[0][1].get("design_identity") != design["identity"]
                        or progress[0][1].get("source_fingerprint") != current_source
                        or arm["replay_design_identity"] != design["identity"] or arm.get("adaptation") is not False
                        or arm["break_lead"] != design["recipe"]["g6"]["break_lead"]):
                    raise ValueError("G6 requires its complete fixed-design replay with zero parameter fits or adaptation")
                identity(progress[0][1], label="G6 replay completion")
                stress_jobs = {}
                for member, receipt in documents.items():
                    if receipt.get("schema") == "v03-g6-data-1.0":
                        identity(receipt, label="G6 generated-data receipt")
                        if receipt["design_identity"] != design["identity"] or receipt["break_lead"] != arm["break_lead"]:
                            raise ValueError("G6 data changed the frozen hidden-transition design")
                        inputs.add(safe_member(member.parent, "simulation.npz"), expected=receipt["sha256"], size=receipt["bytes"])
                        generation.append(receipt)
                    elif receipt.get("schema") == "v03-g6-replay-job-1.0":
                        identity(receipt, "receipt_identity", "G6 replay job")
                        key = (receipt["replication"], receipt["track"], receipt["model_id"])
                        if (key in stress_jobs or key not in expected or receipt["design_identity"] != design["identity"]
                                or receipt["source_fingerprint"] != current_source or receipt["new_parameter_fits"] != 0
                                or receipt["control_score_replay_verified"] is not True
                                or receipt["control_receipt_identity"] != expected[key]["receipt_identity"]
                                or receipt["control_manifest_identity"] != control["manifest"]["identity"]
                                or receipt["score_tolerance"] != design["recipe"]["g6"]["score_replay_tolerance"]):
                            raise ValueError("G6 replay changes its original control fit or information policy")
                        bound_inputs(receipt["source_inputs"], inputs)
                        for item in receipt["artifacts"]:
                            inputs.add(safe_member(member.parent, item["file"]), expected=item["sha256"], size=item["bytes"])
                        for population in ("native_scores", "operational_scores"):
                            _same_score_tree(receipt["replayed_control_" + population], expected[key][population], **receipt["score_tolerance"])
                        stress_jobs[key] = receipt
                generation.sort(key=lambda row: row["replication"])
                if set(stress_jobs) != set(expected) or [row["replication"] for row in generation] != list(range(5)):
                    raise ValueError("G6 audit lacks its exact five fields and complete fixed-roster replay")
            source_jobs = []
            for pair in arm["paired_jobs"]:
                key = (pair["replication"], pair["track"], pair["model_id"])
                a, b = expected[key], stress_jobs[key]
                _validate_stress_pair(pair, a, b, name)
                if name != "velocity_shift" and (a["fit_config"] != b["fit_config"]
                        or a["origin_index"] != b["origin_index"] or a["target_ids"] != b["target_ids"]):
                    raise ValueError("G5 changes the control specification, forecast origin or target roles")
                historical[a["receipt_identity"]] = a["total_pipeline_seconds"]
                source_jobs.append(b)
            if costs["source_job_receipt_identities"] != [row["receipt_identity"] for row in source_jobs]:
                raise ValueError("Stress cost population differs from its actual job receipts")
            time_key = "new_pipeline_seconds" if name == "velocity_shift" else "total_pipeline_seconds"
            cpu_key = "new_pipeline_cpu_seconds" if name == "velocity_shift" else "total_pipeline_cpu_seconds"
            total = sum(row[time_key] for row in source_jobs) + sum(row["generation_and_serialization_seconds"] for row in generation)
            cpu = _sum_known([row.get(cpu_key) for row in source_jobs] + [row["generation_and_serialization_cpu_seconds"] for row in generation])
            _same_number(costs["incremental_pipeline_seconds"], total, "Stress incremental pipeline time")
            _same_number(costs["incremental_pipeline_cpu_seconds"], cpu, "Stress incremental CPU time")
            if (costs["new_parameter_fits"] != (0 if name == "velocity_shift" else 55)
                    or costs["compute_accounting_complete"] != all(row.get("compute_accounting_complete", False) for row in source_jobs)):
                raise ValueError("Stress audit misstates fitting work or accounting completeness")
            if name == "velocity_shift":
                if (costs["forecast_jobs"] != len(source_jobs)
                        or costs["source_generation_receipt_identities"] != [row["identity"] for row in generation]
                        or costs["replayed_model_forecasts"] != sum(row["forecast_method"] == "replayed_frozen_model_original_seed" for row in source_jobs)
                        or costs["reused_saved_forecasts"] != sum(row["forecast_method"] == "reused_saved_forecast" for row in source_jobs)):
                    raise ValueError("G6 misstates its reused versus replayed forecast work")
                for field, expected_value in (("generation_and_serialization_seconds", sum(row["generation_and_serialization_seconds"] for row in generation)),
                        ("replay_verification_scoring_serialization_seconds", sum(row[time_key] for row in source_jobs))):
                    _same_number(costs[field], expected_value, "G6 component time")
            validate_stress_effects(arm, design["recipe"])
        if seen_arms != set(STRESS_ARMS) or value["new_parameter_fits"] != 110:
            raise ValueError("Stress audit is missing a declared intervention or changes its fitting budget")
        if value["historical_shared_control_costs_by_receipt"] != historical:
            raise ValueError("Stress audit duplicates or changes a reused control cost")
        _same_number(value["historical_shared_control_pipeline_seconds"], sum(historical.values()), "Shared historical control cost")
        _same_number(value["incremental_pipeline_seconds"], sum(arm["costs"]["incremental_pipeline_seconds"] for arm in value["arms"]), "Total incremental stress time")
        _same_number(value["incremental_pipeline_cpu_seconds"], _sum_known(arm["costs"]["incremental_pipeline_cpu_seconds"] for arm in value["arms"]), "Total incremental stress CPU time")
        if value["compute_accounting_complete"] != all(arm["costs"]["compute_accounting_complete"] for arm in value["arms"]):
            raise ValueError("Stress total accounting completeness differs from its arms")
        results.append({"status": "complete", "path": str(path), "control_label": control["label"], "document": value})
    return results


def _validate_history_choice(value, runs):
    if value.get("confirmation_outcomes_inspected") is not False:
        raise ValueError("History choice crossed the declared development-only boundary")
    jobs = {n: {(r["scenario"], r["replication"], r["track"], r["model_id"]): r
                for r in run["receipts"]} for n, run in runs.items()}
    qualification_cells = {}
    for contrast in value["contrasts"]:
        metric = contrast["metric"]
        population = contrast["score_family"]
        if population not in {"native_scores", "operational_scores"}:
            raise ValueError("Unknown history comparison population")
        is_decision = (contrast["model_id"] == value["decision_rule"]["decision_model"]
                       and metric == "crps" and population == "native_scores")
        if contrast["is_decision_contrast"] != is_decision:
            raise ValueError("History audit changed its prespecified decision model/score")
        expected_margin = value["decision_rule"]["minimum_normalized_crps_improvement"] if is_decision else None
        if contrast["margin"] != expected_margin:
            raise ValueError("History audit changed its prespecified improvement margin")
        for cell in contrast["cells"]:
            key_tail = (contrast["track"], contrast["model_id"])
            actual, eligible, qualified_short, qualified_long = [], [], 0, 0
            for replication in range(runs[1000]["manifest"]["protocol"]["replications"]):
                key = (cell["scenario"], replication, *key_tail)
                short, long = jobs[1000][key], jobs[2000][key]
                qualified_short += bool(short["qualified"])
                qualified_long += bool(long["qualified"])
                if population == "native_scores" and not (short["qualified"] and long["qualified"]
                        and metric in short[population] and metric in long[population]):
                    continue
                actual.append((short[population][metric] - long[population][metric]) / short["training_scale"])
                eligible.append(replication)
            if (eligible != cell["usable_replications"] or qualified_short != cell["qualified1000"]
                    or qualified_long != cell["qualified2000"]):
                raise ValueError("History audit uses a different paired population")
            if contrast["is_decision_contrast"]:
                qualification_cells[(contrast["model_id"], contrast["track"], cell["scenario"])] = qualified_long >= qualified_short
            saved = cell["paired_differences"]
            if len(saved) != len(actual) or not np.allclose(saved, actual, rtol=1e-11, atol=1e-12):
                raise ValueError("History audit paired losses differ from the saved scientific receipts")
            if contrast["macro"] is not None:
                saved = contrast["macro"]["paired_differences"][cell["scenario"]]
                if len(saved) != len(actual) or not np.allclose(saved, actual, rtol=1e-11, atol=1e-12):
                    raise ValueError("History audit paired losses differ from the saved scientific receipts")
        minimum = value["decision_rule"]["minimum_common_qualified_pairs_per_native_cell"] if population == "native_scores" else runs[1000]["manifest"]["protocol"]["replications"]
        sufficient = all(len(cell["usable_replications"]) >= minimum for cell in contrast["cells"])
        if contrast["sufficient_pairs"] != sufficient:
            raise ValueError("History audit misstates its minimum paired population")
        passed = (contrast["macro"] is not None and sufficient
                  and contrast["macro"]["lower_one_sided95"] >= contrast["margin"]) if contrast["is_decision_contrast"] else None
        if contrast["bound_passed"] != passed:
            raise ValueError("History audit misstates its declared score-bound decision")
    for count, run in runs.items():
        _same_number(value["summed_pipeline_seconds"][str(count)],
                     sum(r["total_pipeline_seconds"] for r in run["receipts"]), "History summed pipeline time")
    saving = 1 - value["summed_pipeline_seconds"]["1000"] / value["summed_pipeline_seconds"]["2000"]
    _same_number(value["short_history_time_saving_fraction"], saving, "History time-saving fraction")
    ratio = value["summed_pipeline_seconds"]["2000"] / value["summed_pipeline_seconds"]["1000"]
    _same_number(value["long_to_short_pipeline_time_ratio"], ratio, "History long-to-short time ratio")
    complete = all(r["compute_accounting_complete"] for run in runs.values() for r in run["receipts"])
    cost_pass = complete and ratio <= value["decision_rule"]["maximum_measured_long_to_short_pipeline_time_ratio"]
    primary = [c for c in value["contrasts"] if c["is_decision_contrast"]]
    if {c["track"] for c in primary} != set(value["decision_rule"]["tracks"]) or len(primary) != len(value["decision_rule"]["tracks"]):
        raise ValueError("History audit omits a prespecified decision track")
    score_pass = all(c["sufficient_pairs"] and c["bound_passed"] for c in primary)
    saved_qualification = {(c["model_id"], c["track"], c["scenario"]): c["not_worse"] for c in value["qualification_cells"]}
    if saved_qualification != qualification_cells or len(saved_qualification) != len(value["qualification_cells"]):
        raise ValueError("History qualification decision differs from saved model outcomes")
    qualification_pass = all(qualification_cells.values())
    promoted = bool(cost_pass and score_pass and qualification_pass)
    selected = 2000 if promoted else 1000
    if (value["complete_compute_accounting"] != complete or value["cost_gate_passed"] != cost_pass
            or value["score_bounds_passed"] != score_pass or value["qualification_not_worse"] != qualification_pass
            or value["promotion_gate_passed"] != promoted or value["selected_initial_history"] != selected):
        raise ValueError("History choice does not follow its saved gate components")


def load_extensions(path, inputs, current_source, *, runs=()):
    if path is None or not Path(path).exists():
        return {"status": "not supplied" if path is None else "not available", "entries": []}
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "decisions.json"
    value = inputs.json(path)
    if not isinstance(value, dict):
        raise ValueError("Extension dispositions must be a JSON mapping")
    if "identity" in value:
        identity(value, label="Extension disposition")
    referenced_paths = []
    for reference in value.get("evidence_inventories", []):
        inventory_path = Path(reference["path"])
        if not inventory_path.is_absolute():
            inventory_path = ROOT / inventory_path
        inputs.add(inventory_path, expected=reference["sha256"])
        saved = inputs.json(inventory_path)
        identity(saved, label="Referenced extension output inventory")
        if (saved["identity"] != reference["identity"] or saved.get("complete") is not True
                or saved.get("source_fingerprint") != current_source):
            raise ValueError("Referenced extension evidence is incompatible or incomplete")
        names = [entry["file"] for entry in saved["entries"]]
        if len(names) != len(set(names)):
            raise ValueError("Referenced extension output inventory contains duplicate files")
        for entry in saved["entries"]:
            inputs.add(safe_member(inventory_path.parent, entry["file"]),
                       expected=entry["sha256"], size=entry["bytes"])
        if saved.get("mode") == "path_stability":
            if "path_stability.json" not in names:
                raise ValueError("Referenced path study lacks its numerical result")
            referenced_paths.append(inputs.json(inventory_path.with_name("path_stability.json")))
    if value.get("source_fingerprint", current_source) != current_source:
        raise ValueError("Extension dispositions refer to a different current source")
    if value.get("schema") == "v03-release-extension-dispositions-1.0":
        identity(value, label="Release extension-disposition overlay")
        if value.get("source_fingerprint") != current_source:
            raise ValueError("Release disposition overlay needs its current source identity")
        bound = bound_inputs(value["source_inputs"], inputs)
        reference = value["base_findings"]
        previous_path = Path(reference["path"])
        previous_path = (previous_path if previous_path.is_absolute() else ROOT / previous_path).resolve()
        if previous_path not in bound:
            raise ValueError("Release disposition overlay omits its immutable P5 base input")
        inputs.add(previous_path, expected=reference["sha256"])
        previous = inputs.json(previous_path)
        identity(previous, label="Immutable P5 disposition base")
        if (previous.get("schema") != "p5-extension-findings-1.0" or previous["identity"] != reference["identity"]
                or previous.get("source_fingerprint") != current_source):
            raise ValueError("Release disposition overlay binds an incompatible P5 base")
    replay = value.get("replay_provenance")
    if replay:
        if (replay.get("new_independent_datasets") != 0 or replay.get("same_datasets_and_seeds") is not True
                or replay.get("new_model_selection_rules") is not False or replay.get("historical_artifacts_modified") is not False):
            raise ValueError("A P5 source replay must preserve its original evidence and replication population")
        previous_path = Path(replay["historical_findings_path"])
        if not previous_path.is_absolute():
            previous_path = ROOT / previous_path
        inputs.add(previous_path, expected=replay["historical_findings_sha256"])
        previous = inputs.json(previous_path)
        identity(previous, label="Historical P5 disposition bound by replay")
        if (previous["identity"] != replay["historical_findings_identity"]
                or previous.get("source_fingerprint") != replay["historical_source_fingerprint"]):
            raise ValueError("P5 replay does not bind its declared historical disposition")
    if value.get("path_count_assessment"):
        validate_path_assessment(value["path_count_assessment"], referenced_paths, runs, inputs)
    entries = value.get("extensions", value.get("rows", value.get("decisions")))
    if not isinstance(entries, list) or any(not isinstance(row, dict) for row in entries):
        raise ValueError("Extension JSON requires an extensions/rows/decisions list")
    entries = [dict(row) for row in entries]
    for row in entries:
        row.setdefault("status", row.get("disposition", row.get("default_disposition")))
        row.setdefault("rationale", row.get("trigger", row.get("reason")))
        if not row.get("name", row.get("id")) or not row.get("status") or not row.get("rationale", row.get("reason")):
            raise ValueError("Every extension disposition needs name, status and rationale")
        if row.get("run_directory"):
            folder = Path(row["run_directory"])
            if not folder.is_absolute():
                folder = Path(path).resolve().parent / folder
            row["validated_run_identity"] = load_run(folder, row.get("name", row.get("id")), inputs, current_source)["manifest"]["identity"]
    status = "implementation scope; no experiment results" if value.get("scientific_results_in_this_file") is False else "supplied dispositions"
    scientific = None
    if value.get("manifest_identity"):
        manifest = inputs.json(Path(path).resolve().with_name("manifest.json"))
        identity(manifest, label="Extension study manifest")
        if manifest["identity"] != value["manifest_identity"] or manifest.get("source_fingerprint") != current_source:
            raise ValueError("Extension decisions do not match their source-bound study manifest")
        inventory = inputs.json(path.with_name("output_inventory.json"))
        identity(inventory, label="P5 output inventory")
        if (inventory.get("schema") != "p5-output-inventory-1.0" or inventory.get("complete") is not True
                or inventory.get("manifest_identity") != manifest["identity"]
                or inventory.get("source_fingerprint") != current_source):
            raise ValueError("P5 results require their matching complete output inventory")
        names = [item["file"] for item in inventory["entries"]]
        if len(names) != len(set(names)) or not {"manifest.json", "inspection.json", "decisions.json"}.issubset(names):
            raise ValueError("P5 terminal inventory is incomplete or duplicated")
        for item in inventory["entries"]:
            inputs.add(safe_member(path.parent, item["file"]), expected=item["sha256"], size=item["bytes"])
        baseline_runs = [run for run in runs if any(a["file"] == "model.npz" and a["sha256"] == manifest["baseline_sha256"]
                         for receipt in run["receipts"] for a in receipt["artifacts"])]
        if len(baseline_runs) != 1:
            raise ValueError("P5 baseline hash must identify exactly one supplied benchmark run")
        baseline_run = baseline_runs[0]
        inspection = inputs.json(path.with_name("inspection.json"))
        if inspection.get("manifest_identity") != manifest["identity"] or inspection.get("future_labels_used") is not False:
            raise ValueError("P5 inspection has incompatible identity or information policy")
        scientific = {"mode": inventory["mode"], "inventory_identity": inventory["identity"],
                      "base_run": baseline_run["label"], "qualified_baseline": inspection["qualified_baseline"],
                      "inspection": inspection, "ablation": None, "path_stability": None}
        if inventory["mode"] in {"extensions", "path_stability"}:
            filename = "paired_ablation.json" if inventory["mode"] == "extensions" else "path_stability.json"
            if filename not in names:
                raise ValueError("P5 requested scientific result is absent from its terminal inventory")
            result = inputs.json(path.with_name(filename))
            for field_name in ("simulation_sha256", "development_manifest_sha256"):
                if result["evaluation"].get(field_name) != manifest.get(field_name):
                    raise ValueError("P5 evaluation inputs differ from the sealed design")
            if manifest["development_manifest_sha256"] != sha256(Path(baseline_run["path"]) / "manifest.json"):
                raise ValueError("P5 evaluation is attached to another supplied study")
            if inventory["mode"] == "extensions":
                if result.get("complete") is not True or not result.get("rows") or result["rows"][0]["variant"] != "scalar_reference":
                    raise ValueError("P5 ablation needs a complete contemporaneous scalar reference")
                by_variant = {row["variant"]: row for row in result["rows"]}
                if len(by_variant) != len(result["rows"]):
                    raise ValueError("P5 ablation variant receipts are duplicated")
                for row in result["rows"]:
                    if inputs.json(path.parent / row["variant"] / "receipt.json") != row:
                        raise ValueError("P5 ablation summary differs from its variant receipt")
                    if "scores" in row and not row["qualified"]:
                        raise ValueError("Unqualified P5 variant entered a native score comparison")
                reference = by_variant["scalar_reference"]
                for pair in result.get("comparisons", []):
                    row = by_variant[pair["variant"]]
                    usable = "scores" in reference and "scores" in row
                    if pair["common_qualified_datasets"] != int(usable) or pair["mcse_across_datasets"] is not None:
                        raise ValueError("P5 comparison misstates its single-dataset population")
                    if usable:
                        for metric in ("rmse", "mae", "crps"):
                            _same_number(pair["loss_difference_extension_minus_scalar"][metric],
                                row["scores"][metric] - reference["scores"][metric], "P5 paired saved loss")
                    elif pair.get("loss_difference_extension_minus_scalar") is not None:
                        raise ValueError("P5 comparison invented a loss for an unqualified pair")
                scientific["ablation"] = result
            else:
                from bayes_stesn.benchmark.contracts import Forecast
                if (result.get("posterior_identity") != manifest["input_posterior_identity"]
                        or [row["paths"] for row in result["rows"]] != [128, 256, 512]
                        or result.get("global_path_indexing") is not True):
                    raise ValueError("P5 path study does not preserve the fixed posterior/nested path design")
                forecast_path = safe_member(path.parent, "path_stability_forecast.npz")
                with zipfile.ZipFile(forecast_path) as archive:
                    if sum(item.file_size for item in archive.infolist()) > 128 * 1024**2:
                        raise MemoryError("P5 saved path validation exceeds the 128 MiB archive allowance")
                forecast = Forecast.load(forecast_path)
                for row in result["rows"]:
                    if content_hash(forecast.samples[:row["paths"]]) != row["path_hash"]:
                        raise ValueError("P5 nested path prefix hash disagrees with saved joint paths")
                    for metric in ("rmse", "mae", "crps"):
                        _same_number(row["difference_from_512"][metric],
                                     row["scores"][metric] - result["rows"][-1]["scores"][metric], "P5 path-count sensitivity")
                scientific["path_stability"] = result
            status = "completed bounded P5 evidence; no automatic promotion"
    return {"status": status, "entries": entries, "path": str(path), "document": value,
            "scientific_evidence": scientific}


def validate_path_assessment(assessment, path_results, runs, inputs):
    """Check the recorded v0.3 budget decision against its saved rule and losses."""
    rule_path = Path(assessment["predeclared_rule_path"])
    if not rule_path.is_absolute():
        rule_path = ROOT / rule_path
    inputs.add(rule_path, expected=assessment["predeclared_rule_document_sha256"])
    rule = inputs.json(rule_path)["predictive_path_count_rule"]
    if (assessment["predeclared_rule"] != rule or rule["candidate_counts"] != [128, 256, 512]
            or rule["reference"] != 512):
        raise ValueError("Path-count decision does not match its supported predeclared rule")
    receipt_ids = {row["receipt_identity"] for run in runs for row in run["receipts"]}
    passing = {n: True for n in rule["candidate_counts"]}
    cases = assessment["cases"]
    if len(cases) != len(rule["cases"]) or len({c["posterior_identity"] for c in cases}) != len(cases):
        raise ValueError("Path-count decision misstates its prespecified cases")
    for case in cases:
        reference = Path(case["reference_receipt_path"])
        if not reference.is_absolute():
            reference = ROOT / reference
        inputs.add(reference, expected=case["reference_receipt_sha256"])
        receipt = inputs.json(reference)
        identity(receipt, "receipt_identity", "Path decision reference")
        if receipt["receipt_identity"] not in receipt_ids or not receipt["qualified"]:
            raise ValueError("Path-count decision must bind a supplied qualified benchmark receipt")
        _same_number(case["training_scale"], receipt["training_scale"], "Path decision training scale")
        saved = [p for p in path_results if p["posterior_identity"] == case["posterior_identity"]]
        if len(saved) != 1 or saved[0]["evaluation"]["scenario"] != case["scenario"]:
            raise ValueError("Path-count decision lacks exactly one matching saved sensitivity study")
        rows = case["rows"]
        if [r["paths"] for r in rows] != rule["candidate_counts"]:
            raise ValueError("Path-count decision changed its candidate prefixes")
        by_count = {r["paths"]: r for r in saved[0]["rows"]}
        for row in rows:
            original = by_count[row["paths"]]
            for key in ("rmse", "crps", "coverage90"):
                number = original["scores"][key] if key != "coverage90" else original["scores"]["intervals"]["0.9"]["coverage"]
                _same_number(row[key], number, "Path decision saved score")
            _same_number(row["normalized_absolute_crps_difference_from_512"],
                         abs(row["crps"] - rows[-1]["crps"]) / case["training_scale"], "Path decision normalized CRPS")
            _same_number(row["absolute_coverage90_difference_from_512"],
                         abs(row["coverage90"] - rows[-1]["coverage90"]), "Path decision coverage difference")
            for statistic in ("rms", "max"):
                key = "conditional_predictive_mean_mcse_" + statistic
                _same_number(row[key], original["predictive_mean_mcse_" + statistic], "Path decision conditional mean MCSE")
                _same_number(row[key + "_over_training_scale"], row[key] / case["training_scale"],
                             "Path decision normalized mean MCSE")
            passed = (row["normalized_absolute_crps_difference_from_512"] <= .005
                      and row["absolute_coverage90_difference_from_512"] <= .02)
            if row["passes_both_rules"] != passed:
                raise ValueError("Path-count decision changed its fixed CRPS/coverage rule")
            passing[row["paths"]] &= passed
    if (assessment["passes_on_both_cases"] != {str(n): p for n, p in passing.items()}
            or assessment["selected_for_subsequent_studies"] != next(n for n, passed in passing.items() if passed)):
        raise ValueError("Path-count decision does not follow its saved case results")


def current_extension_entries(studies):
    """A sealed release overlay supersedes P5 findings and scope; conflicts fail."""
    chosen = {}
    for study in studies:
        if study.get("scientific_evidence"):
            continue
        document = study.get("document", {})
        rank = {"v03-release-extension-dispositions-1.0": 3,
                "p5-extension-findings-1.0": 2}.get(document.get("schema"),
                    int(document.get("scientific_results_in_this_file") is not False))
        for row in study["entries"]:
            key = row.get("id", row.get("name"))
            current = chosen.get(key)
            if current and current[0] == rank and (current[1]["status"], current[1]["rationale"]) != (row["status"], row["rationale"]):
                raise ValueError("Conflicting extension dispositions at the same evidence priority")
            if current is None or rank > current[0]:
                chosen[key] = (rank, row)
    return [row for _, row in chosen.values()]


def load_media(specs, runs, inputs):
    by_label = {r["label"]: r for r in runs}
    output, destinations = [], {}
    for label, path in specs:
        if label not in by_label:
            raise ValueError(f"Media label must name a supplied run: {label}")
        path = Path(path).resolve()
        manifest_path = path / "report_manifest.json"
        if not manifest_path.exists():
            manifest_path = path / "manifest.json"
        manifest = inputs.json(manifest_path)
        if manifest.get("run_identity") != by_label[label]["manifest"]["identity"]:
            raise ValueError("Animation/report assets belong to another run")
        if manifest.get("render_only") is True:
            inventory = [(a["path"], a["sha256"], a.get("bytes")) for a in manifest["files"]]
        elif manifest.get("schema") == "saved-model-comparison-v1":
            inventory = [(name, digest, None) for name, digest in manifest["output_sha256"].items()]
            for name, digest in manifest.get("input_sha256", {}).items():
                inputs.add(safe_member(by_label[label]["path"], name), expected=digest)
        else:
            raise ValueError("Media needs an existing core report or comparison-renderer inventory")
        files = []
        for name, digest, size in inventory:
            member = safe_member(path, name)
            inputs.add(member, expected=digest, size=size)
            if member.suffix.lower() in {".mp4", ".gif", ".png"}:
                destination = (label, name)
                if destination in destinations and destinations[destination] != digest:
                    raise ValueError("Different supplied media assets share one report destination")
                destinations[destination] = digest
                files.append({"path": str(member), "name": name, "sha256": digest})
        output.append({"label": label, "path": str(path), "manifest": manifest, "files": files})
    return output


def load_engineering(inputs):
    folder = ROOT / "docs/evidence/v03"
    memory = inputs.json(folder / "forecast_ownership.json") if (folder / "forecast_ownership.json").exists() else None
    if memory:
        rows = memory.get("rows", [])
        if (memory.get("schema") != "benchmark-forecast-ownership-probe-1.0" or len(rows) != 2 or
                [r["mode"] for r in rows] != ["public_defensive_copy", "private_owned_transfer"] or
                not memory.get("numerical_payloads_equal") or len({r["path_sha256"] for r in rows}) != 1 or
                len({r["mean_sha256"] for r in rows}) != 1):
            raise ValueError("Memory ownership receipt lacks matching numerical payloads")
        incremental = [(r["peak_rss_bytes"] - r["peak_rss_before_constructor_bytes"]) for r in rows]
        if memory["incremental_peak_reduction_bytes"] != incremental[0] - incremental[1]:
            raise ValueError("Memory ownership reduction disagrees with its measurements")
    tests = []
    for path in sorted(folder.glob("*tests*.xml")):
        inputs.add(path)
        tree = ElementTree.parse(path).getroot()
        suites = list(tree.findall("testsuite")) or [tree]
        tests.append({"file": path.name, **{key: sum(int(s.get(key, 0)) for s in suites)
                     for key in ("tests", "failures", "errors", "skipped")},
                      "recorded_at": suites[0].get("timestamp", "unrecorded")})
    release = []
    for path in sorted(set(folder.glob("*release*.json")) | set(folder.glob("*ci*.json"))):
        receipt = inputs.json(path)
        if receipt.get("schema") == "github-actions-verification-1.0":
            release.append({"file": path.name, "kind": "saved remote CI", "receipt": receipt})
        elif (receipt.get("schema_version") == 2
              and {"passed", "steps", "source_hashes", "repository_published"}.issubset(receipt)):
            release.append({"file": path.name, "kind": "isolated local release gate", "receipt": receipt})
    admission_path = folder / "cache_admission_review.json"
    admission = inputs.json(admission_path) if admission_path.exists() else None
    if admission:
        identity(admission, label="Cache-admission engineering evidence")
        if admission.get("schema") != "cache-admission-review-1.0":
            raise ValueError("Unknown cache-admission evidence schema")
    return {"memory": memory, "tests": tests, "release": release, "memory_admission": admission}


def named_path(value):
    label, separator, path = value.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", label) or not path:
        raise argparse.ArgumentTypeError("Use a simple unique label=directory")
    return label, Path(path)


def collect(args):
    from bayes_stesn.benchmark.runner import source_fingerprint
    source, inputs = source_fingerprint(), Inputs()
    specs = args.run or []
    if len({label for label, _ in specs}) != len(specs):
        raise ValueError("Run labels must be unique")
    coordinate = load_coordinate(args.coordinate, inputs, source, args.allow_coordinate_source) if args.coordinate else None
    runs = [load_run(path, label, inputs, source) for label, path in specs]
    if not coordinate and not runs:
        raise ValueError("Supply at least one --coordinate or --run")
    curves = [load_curve(path, inputs, source) for path in args.curve] if args.curve else [load_curve(None, inputs, source)]
    curve_audits = load_curve_audits(args.curve_audit, curves, inputs, source)
    stress_audits = load_stress_audits(args.stress_audit, runs, inputs, source)
    extension_studies = [load_extensions(path, inputs, source, runs=runs) for path in args.extensions]
    extensions = ({"status": "; ".join(dict.fromkeys(x["status"] for x in extension_studies)),
                   "entries": [entry for x in extension_studies for entry in x["entries"]], "studies": extension_studies}
                  if extension_studies else {"status": "not supplied", "entries": [], "studies": []})
    extensions["current_entries"] = current_extension_entries(extension_studies)
    media = load_media(args.media or [], runs, inputs)
    audits = load_audits(args.audit or [], runs, inputs, source)
    calibrations = load_calibrations(args.calibration or [], runs, inputs, source)
    diagnostics = load_diagnostics(args.diagnostic, runs, inputs, source)
    prior_checks = load_prior_checks(args.prior_check, extension_studies, inputs, source)
    engineering = load_engineering(inputs)
    for name in ("docs/models.md", "docs/derivations/07_sampler_coordinates_recovery.md", "docs/study_design_v03.md"):
        inputs.add(ROOT / name)
    return {"source_fingerprint": source, "coordinate": coordinate, "runs": runs, "curves": curves,
            "curve_audits": curve_audits, "stress_audits": stress_audits,
            "extensions": extensions, "media": media, "audits": audits,
            "calibrations": calibrations, "diagnostics": diagnostics, "prior_checks": prior_checks,
            "engineering": engineering}, inputs


def fmt(value, digits=3):
    return "unavailable" if value is None else f"{value:.{digits}g}"


def short_scenario(value):
    return value.split("_")[0].upper()


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.titlesize": 14,
        "axes.labelsize": 13, "axes.titleweight": "bold", "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#CDD9E1", "text.color": NAVY,
        "axes.labelcolor": GRAY, "xtick.color": GRAY, "ytick.color": GRAY,
        "figure.facecolor": "white", "axes.facecolor": "white", "grid.color": "#E3EBEF",
        "savefig.dpi": 210, "svg.fonttype": "none", "pdf.fonttype": 42})
    return plt


def save_figure(figure, output, name, caption):
    from matplotlib.text import Text
    # Match render_pdf's available A4 width and maximum placed plot height.
    # Tight bounding boxes and explicit small legend fonts otherwise shrink
    # nominally readable labels to 4-6 pt in the exported PDF.
    minimum_pdf_font = None
    for attempt in range(4):
        figure.canvas.draw()
        texts = [text for text in figure.findobj(Text) if text.get_visible() and text.get_text()]
        bounds = figure.get_tightbbox(figure.canvas.get_renderer())
        scale = min((210 / 25.4 * 72 - 92) / (72 * (bounds.width + .36)),
                    525 / (72 * (bounds.height + .36)))
        minimum_pdf_font = min((text.get_fontsize() * scale for text in texts), default=8.)
        if minimum_pdf_font >= 7.7 or attempt == 3:
            break
        for text in texts:
            if text.get_fontsize() * scale < 7.9:
                text.set_fontsize(7.9 / scale)
    if minimum_pdf_font < 7.5:
        raise ValueError(f"Plot {name} requires a larger layout to retain 7.5 pt PDF labels")
    path = output / "figures" / (name + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", pad_inches=.18)
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=.18)
    plotting().close(figure)
    return {"kind": "figure", "path": str(path.relative_to(output)), "caption": caption,
            "minimum_pdf_font_pt": float(minimum_pdf_font)}


def coordinate_figure(data, output):
    plt = plotting()
    rows = data["summary"]["rows"]
    cases = list(dict.fromkeys((r["scenario"], r["replication"]) for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(10.6, max(4.2, .8 * len(cases) + 1.8)), layout="constrained")
    for ax, key, label in zip(axes, ("minimum_scientific_ess", "minimum_ess_per_second"),
                              ("Minimum scientific ESS", "Minimum ESS per elapsed second")):
        for algorithm, offset, color in (("legacy", -.13, GRAY), ("ridge_total_scale", .13, TEAL)):
            for i, case in enumerate(cases):
                row = next(r for r in rows if (r["scenario"], r["replication"]) == case and r["parameterization"] == algorithm)
                value = row.get(key)
                if value is not None:
                    ax.scatter(value, i + offset, s=72, color=color, marker="o" if row["qualified"] else "X",
                               label=("Legacy" if algorithm == "legacy" else "Total-scale coordinates") if i == 0 else None)
        ax.set_yticks(range(len(cases)), [f"{short_scenario(s)} / dataset {r}" for s, r in cases])
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=.7)
        ax.set_axisbelow(True)
        ax.invert_yaxis()
    axes[0].axvline(400, color="#C47933", ls="--", lw=1, label="ESS screening threshold")
    axes[1].tick_params(axis="y", labelleft=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=12.5, frameon=False)
    return save_figure(fig, output, "sampler_coordinates", "Same target and prespecified datasets. Crosses mark failed diagnostic gates; gate also checks rank-normalized R-hat. Minimum is across bulk/tail ESS of original hyperparameters and recovered functionals. ESS/second is omitted for resumed timings. These are development fits, not forecast accuracy results.")


def equation_figure(output, name, equations):
    plt = plotting()
    fig, ax = plt.subplots(figsize=(10.4, .52 * len(equations) + .35))
    ax.set_axis_off()
    for i, equation in enumerate(equations):
        ax.text(.02, 1 - (i + .5) / len(equations), equation, fontsize=14, va="center", color=NAVY)
    return save_figure(fig, output, name, "")


def score_figures(run, output):
    plt = plotting()
    figures, rows = [], run["summary"]["table"]
    for track in run["manifest"]["protocol"]["tracks"]:
        scenarios = list(dict.fromkeys(r["scenario"] for r in rows if r["track"] == track))
        for offset in range(0, len(scenarios), 2):
            subset = scenarios[offset:offset + 2]
            fig, axes = plt.subplots(len(subset), 2, figsize=(10.8, 5.2 * len(subset)), squeeze=False, layout="constrained")
            for k, scenario in enumerate(subset):
                selected = [r for r in rows if r["track"] == track and r["scenario"] == scenario]
                for ax, metric in zip(axes[k], ("rmse", "crps")):
                    for i, row in enumerate(selected):
                        value, error = row.get(metric), row.get(metric + "_mcse")
                        if value is not None:
                            ax.errorbar(value, i, xerr=None if error is None else 2 * error, fmt="o", capsize=3,
                                        color=COLORS.get(row["model_id"], GRAY), markersize=5)
                    population = "operational_population" if metric == "rmse" else "probability_population"
                    ax.set_yticks(range(len(selected)), [PLOT_LABELS.get(r['model_id'], r['model_id']) for r in selected], fontsize=12.5)
                    for i, row in enumerate(selected):
                        ax.text(1.02, i, f"n={row[population]}", transform=ax.get_yaxis_transform(), va="center", fontsize=12.5)
                    ax.set_ylim(len(selected) - .5, -.5)
                    ax.grid(axis="x", alpha=.6)
                    ax.set_title(f"{short_scenario(scenario)} / {metric.upper()} / track {track}", loc="left")
                    ax.set_xlabel("Original response units\n(smaller is better)")
                axes[k, 1].tick_params(axis="y", labelleft=False)
            figures.append(save_figure(fig, output, f"{run['label']}_scores_{track}_{offset}",
                "Dots are means of independent dataset losses; whiskers are +/-2 MCSE where at least two datasets contribute. Operational RMSE includes declared fallbacks. CRPS uses each method's qualified native distributions; marginal CRPS dots can involve different datasets. Use paired contrasts for direct comparisons."))
    return figures


def paired_figure(run, output):
    plt = plotting()
    rows = [r for r in run["summary"].get("paired_comparisons", [])
            if r.get("reference", "bayes_esn_ridge") == "bayes_esn_ridge" and r["other_model"] in ("separable_stgp", "basis_gru")]
    if not rows:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(10.8, max(4.8, .43 * len(rows) + 2.2)), layout="constrained")
    for ax, metric in zip(axes, ("rmse", "crps")):
        for i, row in enumerate(rows):
            estimate = row.get(metric, {})
            mean = estimate.get("difference")
            interval = estimate.get("ci")
            if mean is not None:
                color = COLORS[row["other_model"]]
                if interval is not None:
                    ax.plot(interval, [i, i], color=color, lw=1.7)
                ax.scatter([mean], [i], color=color, s=26, zorder=3)
        ax.set_yticks(range(len(rows)), [f"{short_scenario(r['scenario'])}/{r['track']} vs {PLOT_LABELS[r['other_model']]}" for r in rows], fontsize=12.5)
        for i, row in enumerate(rows):
            ax.text(1.02, i, f"n={row.get(metric + '_population', 0)}", transform=ax.get_yaxis_transform(), va="center", fontsize=12.5)
        ax.set_ylim(len(rows) - .5, -.5)
        ax.axvline(0, color=GRAY, ls="--", lw=.8)
        ax.grid(axis="x", alpha=.5)
        ax.set_title(f"Paired normalized {metric.upper()}", loc="left")
        ax.set_xlabel("ESN minus comparator\n(negative favors ESN)")
    axes[1].tick_params(axis="y", labelleft=False)
    return [save_figure(fig, output, run["label"] + "_paired", "Saved paired dataset contrasts and pointwise bootstrap intervals. Native CRPS uses the intersection of the two methods' qualified datasets. A missing dot means zero usable pairs; one usable pair has no estimated sampling uncertainty. Training-only normalization is identical within each pair. These are not simultaneous superiority intervals.")]


def calibration_figure(run, output):
    plt = plotting()
    rows = [r for r in run["summary"]["table"] if r["model_id"] in FOCUS and r.get("coverage_90") is not None]
    if not rows:
        return []
    figures = []
    for track in run["manifest"]["protocol"]["tracks"]:
        selected = [r for r in rows if r["track"] == track]
        if not selected:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10.8, max(5., .44 * len(selected) + 2.4)), layout="constrained")
        for ax, field_name, title in zip(axes, ("coverage_90", "interval_score_90"),
                                       ("Marginal 90% coverage", "90% interval score")):
            for i, row in enumerate(selected):
                if row.get(field_name) is not None:
                    ax.scatter(row[field_name], i, color=COLORS[row["model_id"]], s=28)
            ax.set_yticks(range(len(selected)), [f"{short_scenario(r['scenario'])} / {PLOT_LABELS[r['model_id']]}  n={r['probability_population']}" for r in selected], fontsize=12.5)
            ax.set_title(f"Track {track} / {title}", loc="left")
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=.4)
        axes[0].axvline(.9, color=GRAY, ls="--", lw=.8)
        axes[1].tick_params(axis="y", labelleft=False)
        axes[0].set_xlim(0, 1.03)
        axes[0].set_xlabel("Qualified native coverage\n(target = 0.90)")
        axes[1].set_xlabel("Original response units\n(smaller is better)")
        figures.append(save_figure(fig, output, f"{run['label']}_intervals_{track}", "Saved marginal interval coverage and proper interval scores averaged over eligible datasets, using a fixed model subset for legibility. Coverage is not simultaneous whole-field coverage, and small conditional populations do not establish calibration. The interval score jointly penalizes excessive width and missed observations."))
    return figures


def lead_figures(run, output):
    plt = plotting()
    output_figures = []
    for track in run["manifest"]["protocol"]["tracks"]:
        rows = [r for r in run["receipts"] if r["track"] == track and r["model_id"] in FOCUS]
        scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
        for offset in range(0, len(scenarios), 2):
            subset = scenarios[offset:offset + 2]
            fig, axes = plt.subplots(3, len(subset), figsize=(10.6 if len(subset) == 2 else 7, 10.4), squeeze=False, layout="constrained")
            for column, scenario in enumerate(subset):
                for model in FOCUS:
                    selected = [r for r in rows if r["scenario"] == scenario and r["model_id"] == model]
                    for index, (metric, field_name) in enumerate((("rmse_by_lead", "operational_scores"),
                            ("mean_error_by_lead", "operational_scores"), ("crps_by_lead", "native_scores"))):
                        eligible = [r for r in selected if field_name == "operational_scores" or (r["qualified"] and not r["fallback"])]
                        values = [r[field_name][metric] for r in eligible if r.get(field_name) and metric in r[field_name]]
                        if values:
                            values = np.asarray(values, float)
                            if values.ndim != 2 or not np.isfinite(values).all():
                                raise ValueError("Invalid saved leadwise score array")
                            axes[index, column].plot(np.arange(1, values.shape[1] + 1), values.mean(axis=0),
                                color=COLORS[model], lw=1.8 if model == "bayes_esn_ridge" else 1.2,
                                label=PLOT_LABELS[model])
                axes[0, column].set_title(f"{short_scenario(scenario)} / track {track}", loc="left")
                for index, title in enumerate(("Operational RMSE", "Operational signed bias", "Qualified-native CRPS")):
                    axes[index, column].set_ylabel(title)
                    axes[index, column].grid(alpha=.5)
                axes[1, column].axhline(0, color=GRAY, lw=.6)
                axes[2, column].set_xlabel("Forecast lead from the fixed origin")
            legend = {}
            for ax in axes.flat:
                handles, labels = ax.get_legend_handles_labels()
                legend.update(zip(labels, handles))
            fig.legend(list(legend.values()), list(legend), loc="outside lower center", ncol=3, fontsize=12.5, frameon=False)
            output_figures.append(save_figure(fig, output, f"{run['label']}_leads_{track}_{offset}",
                "Mean of saved dataset-level lead scores, without future-data assimilation. Bias is forecast mean minus observed response. Model subset is fixed for legibility; complete scores remain in the machine summaries. Exact operational/native dataset counts are in the preceding design and qualification table; native CRPS populations vary by model. No fit or forecast was recomputed."))
    return output_figures


def cost_figure(run, output):
    plt = plotting()
    models = list(dict.fromkeys(r["model_id"] for r in run["receipts"]))
    fig, axes = plt.subplots(1, 2, figsize=(10.6, max(4.7, len(models) * .44 + 1.8)), layout="constrained")
    for i, model in enumerate(models):
        rows = [r for r in run["receipts"] if r["model_id"] == model]
        total = np.mean([r.get("total_pipeline_seconds", r["total_seconds"] + r.get("tuning_seconds", 0)) for r in rows])
        axes[0].barh(i, total, color=COLORS.get(model, GRAY), height=.6)
        qualified = sum(r["qualified"] for r in rows)
        axes[1].barh(i, qualified / len(rows), color=TEAL, height=.6)
        axes[1].barh(i, 1 - qualified / len(rows), left=qualified / len(rows), color="#E7B38B", height=.6)
        axes[1].text(1.03, i, f"{qualified}/{len(rows)}", va="center", fontsize=8)
    for ax in axes:
        ax.set_yticks(range(len(models)), [PLOT_LABELS.get(m, m) for m in models], fontsize=12.5)
        ax.invert_yaxis()
    axes[0].set_xlabel("Mean recorded pipeline seconds / job")
    axes[1].tick_params(axis="y", labelleft=False)
    axes[0].set_title("Tuning + final fit + forecast + scoring", loc="left")
    axes[0].grid(axis="x", alpha=.4)
    axes[1].set_xlim(0, 1.24)
    axes[1].set_xticks([0, .5, 1], ["0%", "50%", "100%"])
    axes[1].set_xlabel("Qualified fraction; tan = declared fallback")
    axes[1].set_title("All declared final model/task jobs", loc="left")
    incomplete = sum(not r.get("compute_accounting_complete", False) for r in run["receipts"])
    return save_figure(fig, output, run["label"] + "_cost_qualification", f"Recorded costs averaged over the run's declared scenario/task mix; this is not machine-independent throughput. {incomplete} jobs have incomplete compute accounting after infrastructure interruption; their recorded times are lower bounds. Failed scientific attempts remain in cost and operational populations. Process peak RSS is kept in raw receipts with its measurement scope.")


def accuracy_time_figure(run, output):
    plt = plotting()
    rows = run["summary"]["table"]
    cells = sorted({(r["scenario"], r["track"]) for r in rows})
    figures = []
    for offset in range(0, len(cells), 4):
        chunk = cells[offset:offset + 4]
        fig, axes = plt.subplots((len(chunk) + 1) // 2, 2, figsize=(10.5, 4.7 * ((len(chunk) + 1) // 2)), squeeze=False, layout="constrained")
        for ax, (scenario, track) in zip(axes.flat, chunk):
            for row in rows:
                if (row["scenario"], row["track"]) != (scenario, track):
                    continue
                members = [r for r in run["receipts"] if (r["scenario"], r["track"], r["model_id"]) == (scenario, track, row["model_id"])]
                elapsed = np.mean([r.get("total_pipeline_seconds", r["total_seconds"] + r.get("tuning_seconds", 0)) for r in members])
                ax.scatter(elapsed, row["rmse"], color=COLORS.get(row["model_id"], GRAY), s=40,
                           label=PLOT_LABELS.get(row["model_id"], row["model_id"]))
            ax.set_xscale("log")
            ax.set_title(f"{short_scenario(scenario)} / track {track}", loc="left")
            ax.set_xlabel("Mean recorded pipeline seconds / job (log)")
            ax.set_ylabel("Mean operational RMSE")
            ax.grid(alpha=.4)
        for ax in list(axes.flat)[len(chunk):]:
            ax.set_axis_off()
        legend = {}
        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            legend.update(zip(labels, handles))
        fig.legend(list(legend.values()), list(legend), loc="outside lower center", ncol=3, fontsize=12.5, frameon=False)
        figures.append(save_figure(fig, output, f"{run['label']}_accuracy_time_{offset}", "Accuracy and recorded computational cost for the same declared jobs. Lower and left is preferable, but points have different qualification/fallback behavior; this plot is descriptive, not a Pareto-superiority test. Costs include tuning, final fitting, forecasting and saved-artifact serialization; lost work remains labeled incomplete."))
    return figures


def oracle_figure(audit, output):
    plt = plotting()
    rows = audit["document"].get("oracle", [])
    if not rows:
        return []
    tracks = sorted({r["track"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), layout="constrained")
    for track in tracks:
        eligible = [r for r in rows if r["track"] == track]
        for ax, field, label in zip(axes, ("rmse_by_lead", "crps_by_lead"), ("Realized RMSE", "Analytic marginal Gaussian CRPS")):
            array = np.asarray([r["scores"][field] for r in eligible])
            ax.plot(np.arange(1, array.shape[1] + 1), array.mean(axis=0), label=f"Track {track}; n={len(eligible)}", lw=1.8)
            ax.set_xlabel("Forecast lead")
            ax.set_ylabel(label)
            ax.grid(alpha=.4)
            ax.legend()
    return [save_figure(fig, output, audit["label"] + "_known_parameter_g1", "Privileged-parameter G1 reference, conditional on the same noisy observed-site history. It knows generating parameters; it does not observe the latent state or future labels. These are saved analytic reference scores, not a newly fitted competitor. There is no finite-realization dominance guarantee.")]


def curve_figures(curve, output, *, curve_index=0):
    plt = plotting()
    if not curve["runs"]:
        return []
    runs = sorted(curve["runs"], key=lambda r: r["value"])
    keys = sorted({(r["scenario"], r["track"]) for run in runs for r in run["summary"]["table"]})
    figures = []
    for offset in range(0, len(keys), 2):
        chunk = keys[offset:offset + 2]
        fig, axes = plt.subplots(len(chunk), 2, figsize=(10.6, 4.8 * len(chunk)), squeeze=False, layout="constrained")
        for k, (scenario, track) in enumerate(chunk):
            for model in FOCUS:
                for j, metric in enumerate(("rmse", "crps")):
                    points = [(run["value"], row) for run in runs for row in run["summary"]["table"]
                              if (row["scenario"], row["track"], row["model_id"]) == (scenario, track, model)]
                    if points:
                        population = "operational_population" if metric == "rmse" else "probability_population"
                        counts = "/".join(str(p[1][population]) for p in points)
                        axes[k, j].plot([p[0] for p in points], [p[1][metric] if p[1][metric] is not None else np.nan for p in points],
                                       "o-", color=COLORS[model], label=f"{PLOT_LABELS[model]} n={counts}")
            for j, metric in enumerate(("RMSE (operational)", "CRPS (qualified native)")):
                axes[k, j].set_title(f"{short_scenario(scenario)}/{track} / {metric}", loc="left")
                axes[k, j].set_xlabel("Initial training rows" if curve["design"]["kind"] == "history" else "Observed sensors")
                axes[k, j].grid(alpha=.4)
                axes[k, j].legend(fontsize=12.5, ncol=2, loc="upper center", handlelength=1.2, columnspacing=.7)
                low, high = axes[k, j].get_ylim()
                axes[k, j].set_ylim(low, high + .5 * (high - low))
        figures.append(save_figure(fig, output, f"learning_curve_{curve_index}_{offset}", "Resolved matched design; only completed conditions are shown. Legend populations follow conditions in ascending x order. Final forecast period and scored target domain are held fixed. Native CRPS is conditional on qualification; a zero population produces a gap. Connecting points illustrates the declared information intervention, not monotonicity or a proven improvement."))
    return figures


def section(title, paragraphs=(), table=None, figures=(), links=()):
    return {"title": title, "paragraphs": list(paragraphs), "table": table,
            "figures": list(figures), "links": list(links)}


def prior_sections(checks):
    sections = []
    for index, check in enumerate(checks):
        if check["status"] != "complete":
            sections.append(section("Conditional prior assessment: unavailable", [
                "The explicitly supplied prior assessment is absent. No prior adequacy or trajectory-stability claim is inferred."]))
            continue
        value, parameter_rows, amplitude_rows, dependence = check["document"], [], [], []
        for case in value["cases"]:
            name = next((short_scenario(key) for key in SCENARIOS if key in case["ablation_directory"]),
                        Path(case["ablation_directory"]).name)
            results = case["results"]
            parameters = results["prior_parameters"]
            parameter_rows.append([name, f"{fmt(parameters['a0'])} / {fmt(parameters['b0'])}",
                f"{fmt(parameters['log_precision_mean'])} / {fmt(parameters['log_precision_sd'])}",
                f"{fmt(parameters['log_q_mean'])} / {fmt(parameters['log_q_sd'])}",
                f"{fmt(parameters['log_ell_mean'])} / {fmt(parameters['log_ell_sd'])}", fmt(parameters["rho_eta_sd"])])
            for design in results["designs"]:
                sd = design["observation_pointwise_prior_sd"]
                design_label = {"conditioning_reference": "Reference design", "observed_history_fitting_design": "Fitting design"}[design["name"]]
                amplitude_rows.append([name, design_label,
                    fmt(sd["conditioning_standard_deviation_units"]["rms"]),
                    fmt(sd["original_response_units"]["rms"]),
                    fmt(results["stationary_residual_prior_sd"]["original_response_units"]["rms"])])
            dependence.append(f"{name}: median |rho| = {fmt(results['ar_absolute_rho_median'])}; "
                f"Pr(|rho| > 0.9) = {fmt(results['ar_absolute_rho_tail_probabilities']['0.9'])}.")
        links = [("Actual prior law, frozen designs and saved numerical checks", f"evidence/prior_check_{index}.json")]
        shape_statement = ("All supplied cases have a0 = 2: finite prior variance and an infinite fourth moment. "
                           if all(case["results"]["prior_parameters"]["a0"] == 2 for case in value["cases"]) else "")
        sections.append(section("Conditional prior specification: the actual saved parameters", [
            "These are first and second prior-predictive moments conditional on each saved encoder, spatial basis, historical feature path and response standardization. They integrate the original priors. They differ from posterior conditional scale summaries and do not assess autonomous ESN prior rollouts.",
            "The precision, relative spatial scale and correlation range have independent lognormal priors. The table gives normal means / standard deviations on their log scales. The scalar ridge prior applies to every coefficient, including intercept columns. Sigma squared has inverse-gamma shape a0 and scale b0.",
            shape_statement + "Finite prior variance requires a0 > 1; the fourth moment exists only for a0 > 2. These conditions apply to each scalar prior-predictive observation about its frozen center, which has positive residual variance. They concern prior tails, not posterior variance, and do not by themselves show that the prior is adequate.",
            "The renderer binds the source, tool, P5 inventories, saved model specifications and feature hashes, then verifies closed-form moment and unit reductions. It reads no posterior arrays and performs no refit, quadrature or trajectory simulation."],
            table=(["DGP", "IG a0 / b0", "log lambda mu / SD", "log q mu / SD", "log ell mu / SD", "rho eta SD"], parameter_rows), links=links))
        sections.append(section("Prior amplitude and signed temporal dependence", [
            "Pointwise prior observation SD is summarized by its root mean square across the fixed site/time design. Reference design uses the frozen conditioning rows; fitting design uses the observed historical features entering the likelihood. Conditioning-SD units use the saved response standardizer; original units multiply by that case's response scale. Stationary iid and full-residual AR specifications share the same pointwise prior amplitude under these common priors.",
            "For the AR prior, rho = tanh(eta) with a symmetric normal eta. Positive draws give persistent residual correlation and negative draws give alternating correlation. Averaging over both signs makes odd-lag residual covariance zero; individual draws remain temporally dependent. " + " ".join(dependence),
            "The recorded 128/256-point quadrature agreement checks concern integrated spatial correlation and rho moments. The frozen historical design does not certify self-generated trajectory stability, coverage or model adequacy. This retrospective assessment leaves prior choices, predictive comparisons and MCMC qualification unchanged."],
            table=(["DGP", "Frozen temporal design", "Observation RMS SD / conditioning SD", "Observation RMS SD, response units", "Residual SD, response units"], amplitude_rows), links=links))
    return sections


def stress_sections(audits):
    sections = []
    for index, audit in enumerate(audits):
        if audit["status"] != "complete":
            sections.append(section("Paired stress comparisons: unavailable", [
                "The explicitly supplied optional stress audit is absent. No noise-robustness or hidden-shift performance is inferred."]))
            continue
        value = audit["document"]
        links = [("Full paired primary, interval, joint/path and cost evidence", f"evidence/stress_audit_{index}.json")]
        scopes = list(dict.fromkeys(arm["costs"]["scope"] for arm in value["arms"]))
        sections.append(section("Paired stress design and incremental work", [
            f"Control study: {audit['control_label']}. Each intervention uses five paired fields and the same frozen roster. Whole-field and local Student-t G5 arms retain their matched G4 generating components. G6 evaluates the original G2 forecasts against a future with a hidden velocity transition at lead 25; it uses no parameter refit, post-origin assimilation, adaptation or post-shift calibration.",
            "Each difference is stress minus control. Positive means larger loss; for coverage indicators it means more coverage, and for interval width it means a wider band. Both sides of a dimensionful loss use the same control training-only SD. Standalone G5 arm scales are preserved in their receipts but are not the comparison denominator. Coverage differences remain dimensionless.",
            f"New G5 fitting jobs: {value['new_parameter_fits']}; incremental recorded pipeline time {fmt(value['incremental_pipeline_seconds'], 5)} seconds, CPU {fmt(value['incremental_pipeline_cpu_seconds'], 5)} seconds. Historical shared G4/G2 control pipeline time is {fmt(value['historical_shared_control_pipeline_seconds'], 5)} seconds, counted once separately. Complete incremental accounting: {value['compute_accounting_complete']}.",
            "These sums have the following recorded scopes and are not elapsed parallel-campaign time: " + " ".join(scopes),
            "Coupled control/stress arms are not independent macro strata and are not pooled into the core G1-G4 macro comparison. Pointwise paired-dataset bootstrap intervals and MCSEs are exploratory with five fields. Native comparisons condition on their exact common-qualified intersections; operational point comparisons retain declared fallbacks."],
            table=(["Intervention", "New fit jobs", "Incremental wall sec", "Incremental CPU sec", "Complete cost"], [
                [STRESS_ARMS[arm["id"]], str(arm["costs"]["new_parameter_fits"]),
                 fmt(arm["costs"]["incremental_pipeline_seconds"], 5), fmt(arm["costs"]["incremental_pipeline_cpu_seconds"], 5),
                 str(arm["costs"]["compute_accounting_complete"])] for arm in value["arms"]]), links=links))
        for arm in value["arms"]:
            selected = [row for row in arm["effects"] if (row["population"], row["metric"]) in
                        {("operational_scores", "rmse"), ("native_scores", "crps")}]
            selected.sort(key=lambda row: (row["track"], row["model_id"], row["population"]))
            rows = []
            for effect in selected:
                interval = effect["common_normalized"]["pointwise95"]
                rows.append([f"{LABELS.get(effect['model_id'], effect['model_id'])} / {effect['track']}",
                    "Operational RMSE" if effect["population"] == "operational_scores" else "Native CRPS",
                    f"{effect['N_pair']}/5", f"{fmt(effect['raw']['mean'])} ({fmt(effect['raw']['mcse'])})",
                    f"{fmt(effect['common_normalized']['mean'])} ({fmt(effect['common_normalized']['mcse'])})",
                    "unavailable" if interval is None else f"[{fmt(interval[0])}, {fmt(interval[1])}]"])
            paragraphs = [
                "The table preserves every compatible model/task cell for operational RMSE and native CRPS. Parentheses give MCSE across independent paired fields. Zero usable pairs remain unavailable; one pair has no estimated sampling uncertainty. All raw and common-normalized MAE, native RMSE and companion effects are retained in the linked evidence.",
                "Common normalization is applied to each field's difference before averaging. Dividing an average difference by an average scale would be a different estimand. Exact pair IDs, qualification/fallback outcomes and original complete score objects are included in the audit."]
            if arm["id"] == "velocity_shift":
                costs = arm["costs"]
                paragraphs.append(f"G6 used {costs['reused_saved_forecasts']} retained native forecasts and replayed {costs['replayed_model_forecasts']} frozen models with their original forecast seeds. Every original G2 score object was checked before G6 scoring; deterministic fallback reconstruction is recorded separately in each job receipt. New parameter fits: 0.")
            sections.append(section(STRESS_ARMS[arm["id"]] + ": paired marginal effects", paragraphs,
                table=(["Model / task", "Score population", "Pairs", "Raw delta (MCSE)", "Control-SD delta (MCSE)", "95% interval, control SD"], rows), links=links))
            metrics = ("field_energy", "functional_crps/path_maximum", "intervals/0.9/coverage",
                       "intervals/0.9/width", "intervals/0.9/whole_path_covered")
            companions = {(r["model_id"], r["track"], r["metric"]): r for r in arm["effects"]
                          if r["population"] == "native_scores" and r["metric"] in metrics}
            companion_rows = []
            for model, track in sorted({key[:2] for key in companions}, key=lambda key: (key[1], key[0])):
                cells = [f"{LABELS.get(model, model)} / {track}"]
                for metric in metrics:
                    effect = companions.get((model, track, metric))
                    cells.append("unavailable" if effect is None else f"{fmt(effect['common_normalized']['mean'])} (n={effect['N_pair']})")
                companion_rows.append(cells)
            sections.append(section(STRESS_ARMS[arm["id"]] + ": dependence and interval companions", [
                "Entries are stress-minus-control paired means, with the exact eligible count for that metric. Energy, path-maximum CRPS and interval width use the common control SD; marginal and whole-path coverage differences are unscaled. More coverage can reflect wider bands, so it is not a stand-alone improvement criterion.",
                "The full sealed audit also retains spatial and temporal variogram scores, all declared interval levels, interval scores, field coverage and the remaining path-function CRPS contrasts, with raw/common-normalized effects and their pointwise uncertainty. Their score objects preserve the original coherent joint paths; no forecast or calibration is recomputed by this report."],
                table=(["Model / task", "Field energy delta / SD", "Path-max CRPS delta / SD", "90% coverage delta", "90% width delta / SD", "Whole-path coverage delta"], companion_rows) if companion_rows else None,
                links=links))
    return sections


def sections_for(data, output):
    runs, coordinate = data["runs"], data["coordinate"]
    count = sum(len(r["receipts"]) for r in runs)
    qualified = sum(sum(x["qualified"] for x in r["receipts"]) for r in runs)
    sections = [section("Evidence at a glance", [
        f"This synthesis contains {len(runs)} completed benchmark run(s), {count} terminal model/task outcomes, and {qualified} qualified native fits. Each run has its own frozen scientific design and source identity. Separate runs are not pooled into a new experiment.",
        "Development results guide the next study. They do not establish a final ranking or a universal benefit from larger ESNs or longer histories. No confirmation claim is made unless a supplied run is explicitly labeled confirmation and carries its frozen design.",
        "The independent unit is a generated dataset. Sites, forecast leads, posterior paths, candidate seeds and repeated optimization steps are not additional replications."],
        table=(["Quantity", "Population and interpretation"], [
            ["Operational RMSE / MAE", "All terminal method outcomes, including the declared fallback after a method exhausts its scientific budget."],
            ["Native CRPS / interval coverage", "Qualified native predictive distributions only; the number of eligible datasets is displayed."],
            ["Paired CRPS", "Common-qualified intersection within each model pair and scenario. This is conditional performance, not all-attempt reliability."],
            ["Infrastructure interruption", "Unresolved execution, never relabeled as a scientific method failure or an extra replication."]]))]
    hierarchy = equation_figure(output, "model_hierarchy", [
        r"$u_{t-1}=E(y_{t-1},y_{t-2},\ldots),\quad r_t=(I-D)r_{t-1}+D\tanh(Wr_{t-1}+Uu_{t-1}+b_r)$",
        r"$z_t=\mathrm{FrozenScale}[1,u_{t-1},r_t,r_t^2,\ldots],\qquad y_t=\Phi C z_t+e_t$",
        r"$e_t\mid\sigma^2,q,\ell\sim N(0,\sigma^2R_q),\quad R_q=I+qK_\ell$",
        r"$b=\mathrm{vec}(C),\quad b\mid\sigma^2,\lambda\sim N(0,\sigma^2\lambda^{-1}I),\quad\sigma^2\sim\mathrm{IG}(a_0,b_0)$"])
    sections.append(section("Model hierarchy: the repaired reference", [
        "The spatial basis Phi maps a low-rank coefficient surface to fixed input or prediction sites. A frozen, causal encoder and reservoir construct nonlinear temporal features. Deep or multiscale reservoirs replace the first displayed recursion layer by layer; the saved configuration specifies the actual feature map.",
        "The iid Gaussian scalar-ridge reference uses the Matérn residual correlation and original proper hyperpriors. Conditioning-window preprocessing is frozen before likelihood fitting. Recursive forecasts feed back simulated observed-site values; they do not assimilate future observations or latent DGP states.",
        "Conditional Gaussian coefficient recovery and inverse-gamma scale recovery are exact for this reference. The likelihood is a product of causal conditional densities; recursive ESN predictors do not imply that the unconditional response sequence is jointly Gaussian.",
        "Other priors, dynamics and covariance extensions retain separate configuration labels and compatibility gates. Their implementation alone is not evidence that they improve prediction."], figures=[hierarchy], links=[("Full hierarchy and extension equations", "evidence/models.md")]))
    sections.extend(prior_sections(data.get("prior_checks", [])))
    transformed = equation_figure(output, "exact_coordinate_change", [
        r"$a=\log\lambda,\quad\xi=\log q,\quad\zeta=\log\ell,\quad\eta=a+\mathrm{softplus}(\xi)$",
        r"$a=\eta-\mathrm{softplus}(\xi),\qquad |\partial(\eta,\xi,\zeta)/\partial(a,\xi,\zeta)|=1$",
        r"$\log\widetilde\pi(\eta,\xi,\zeta\mid Y)=\log\pi(\eta-\mathrm{softplus}(\xi),\xi,\zeta\mid Y)$",
        r"$\alpha(x,x')=\min\{1,\exp[\log\widetilde\pi(x')-\log\widetilde\pi(x)]\}$"])
    text = ["The total-scale proposal changes coordinates, while preserving the normalized collapsed likelihood and original priors. The log-coordinate Jacobian is one. Sigma squared remains integrated out; no independent replacement prior is introduced for the transformed coordinate.",
        "Initialization, local curvature calibration and random-walk proposals use the transformed target. Adaptation stops at the declared warmup boundary. Diagnostics remain on original hyperparameters and recovered scientific functionals: four chains, rank-normalized R-hat below 1.01 and bulk/tail ESS at least 400 in the scientific comparison.",
        "The certified route is restricted to collapsed scalar ridge, iid Gaussian innovations, stationary isotropic relative-scale Matérn covariance, constant nugget and fixed smoothness. Other combinations must use their supported route."]
    figures = [transformed]
    if coordinate:
        rows = coordinate["summary"]["rows"]
        counts = {a: (sum(r["qualified"] for r in rows if r["parameterization"] == a), sum(r["parameterization"] == a for r in rows)) for a in ("legacy", "ridge_total_scale")}
        text.append(f"Saved comparison: legacy qualified {counts['legacy'][0]}/{counts['legacy'][1]} cases; total-scale coordinates qualified {counts['ridge_total_scale'][0]}/{counts['ridge_total_scale'][1]}. These counts describe the prespecified cases only. Timing is host- and workload-specific; an algorithm can improve one mechanism without improving every case.")
        if coordinate["ancillary_source_exception"]:
            text.append("This is explicitly accepted ancillary evidence from a different recorded source fingerprint. It is not pooled with current-source forecast benchmarks; the exact source and exception are preserved in the report manifest.")
        figures.append(coordinate_figure(coordinate, output))
    else:
        text.append("No coordinate-development results were supplied. The equations describe implemented behavior, not a measured efficiency result.")
    sections.append(section("Exact sampler change and observed qualification", text, figures=figures,
        links=[("Coordinate derivation and recovery contract", "evidence/07_sampler_coordinates_recovery.md")]))
    for run in runs:
        protocol, summary = run["manifest"]["protocol"], run["summary"]
        counts = [[short_scenario(r["scenario"]), r["track"], LABELS.get(r["model_id"], r["model_id"]),
                   f"{r['qualified']}/{r['attempted']}", str(r["probability_population"])] for r in summary["table"]]
        text = [f"Stage: {run['manifest']['stage']}. Protocol: {protocol['name']}. Selection mode: {protocol.get('selection_mode', 'legacy')}. The complete resolved protocol is included in the machine-readable evidence.",
            f"Independent datasets per scenario: {protocol['replications']}; observed sensors: {protocol['n_observed']}; evaluation grid: {protocol['n_side']} by {protocol['n_side']}; forecast horizon: {protocol['horizon']}. Initial training starts at row {protocol.get('training_start', 0)} and ends before row {protocol['initial_fit']}. Model parameters are refitted at the common final forecast origin.",
            f"Predictive path budget: {protocol['paths']}. Protocol MCMC allowance: {protocol['mcmc_chains']} chains, {protocol['mcmc_warmup']} warmup iterations and {protocol['mcmc_draws']} initially retained draws per chain, with at most {protocol['mcmc_continuation']} additional draws under the declared continuation policy. Resolved per-model configurations and tuning budgets remain authoritative.",
            str(summary.get("comparison_status", "Completed development study; no confirmatory ranking.")),
            "Qualification is assessed under each method's declared gate. Neural early stopping is a model-selection rule, not proof of a global optimizer optimum. Separable GP and basis GRU are custom matched controls, not certified reproductions of every similarly named published method."]
        if protocol["replications"] < 5:
            text.append("This run has fewer than five independent datasets per scenario. Its paired uncertainty and qualification-rate estimates are too sparse for the schema-4 confirmation planning gate; use it for reliability and cost diagnosis.")
        missing_native = [r for r in summary["table"] if r["model_id"] in ("bayes_esn_ridge", "separable_stgp", "basis_gru") and not r["probability_population"]]
        if missing_native:
            text.append("No qualified native distribution is available in these primary-model cells: " + "; ".join(
                f"{short_scenario(r['scenario'])}/{r['track']} {LABELS[r['model_id']]}" for r in missing_native) +
                ". Their operational results remain included through the declared fallback.")
        fallback_count = sum(r["fallback"] for r in summary["table"])
        text.append(f"The run retains {fallback_count} fallback outcome(s) among {len(run['receipts'])} terminal model/task jobs. They are scientific failures of the attempted procedure under its declared allowance, not missing infrastructure jobs.")
        sections.append(section(f"{run['label']}: design and qualification", text,
            table=(["DGP", "Task", "Model", "Qualified / attempted", "Native n"], counts),
            links=[("Resolved manifest", f"evidence/{run['label']}_manifest.json"), ("Complete numerical summary", f"evidence/{run['label']}_summary.json")]))
        if run.get("selections"):
            selected_rows = []
            for model in dict.fromkeys(s["model_id"] for s in run["selections"]):
                items = [s for s in run["selections"] if s["model_id"] == model]
                candidates = [s["effective_candidates"] for s in items]
                rechecks = [s["finalist_rechecks"] for s in items]
                distinct = {s["selected_candidate_hash"] for s in items if s["selected"] is not None}
                selected_rows.append([LABELS.get(model, model), f"{min(candidates)}-{max(candidates)}",
                    f"{min(rechecks)}-{max(rechecks)}", str(len(distinct)),
                    f"{sum(s['selected'] is not None for s in items)}/{len(items)}"])
            sections.append(section(f"{run['label']}: selection and information budget", [
                "Counts below come from hash-verified selection receipts. Candidate counts refer to distinct effective configurations after overrides and dimension constraints. Rechecks count actual saved finalist evaluation records; each record can include the declared spatial folds and chronological origins.",
                "Frozen-configuration experiments may have one declared configuration and zero screening/rechecks. A failed selection remains a terminal procedure outcome with its declared fallback. Exact selected parameters, selection identities and candidate-resolution metadata are included in the linked JSON."],
                table=(["Model", "Candidates / job", "Rechecks / job", "Distinct selected", "Selected / jobs"], selected_rows),
                links=[("Selected configurations and candidate accounting", f"evidence/{run['label']}_selected_configurations.json")]))
        sections.append(section(f"{run['label']}: predictive performance", [
            "Scenarios retain their original response units in these marginal score panels. Do not rank methods by comparing unpaired native CRPS means with different qualification populations. Every paired contrast below uses the common eligible datasets and its saved training-only scale."], figures=score_figures(run, output)))
        calibration = calibration_figure(run, output)
        if calibration:
            sections.append(section(f"{run['label']}: predictive uncertainty", [
                "A good forecast distribution must represent uncertainty as well as its mean. These saved interval summaries show whether narrow predictions miss observations and whether wider intervals buy useful coverage. Qualification and calibration are separate questions."], figures=calibration))
        dependence = [r for r in summary["table"] if r["model_id"] in FOCUS and r.get("field_energy") is not None]
        if dependence:
            sections.append(section(f"{run['label']}: joint predictive structure", [
                "Marginal CRPS alone cannot verify spatial or temporal dependence. These saved whole-field energy and spatial variogram scores retain coherent predictive paths. Their empirical populations are the qualified native distributions; a good marginal score does not imply a good joint score.",
                "Scores use the declared geometric pair policy and original response scale. Temporal variogram, functional path scores, field coverage and whole-path coverage remain available in the copied per-job score evidence. Smaller energy/variogram scores are preferable within the same task and population."],
                table=(["DGP / task", "Model", "Native n", "Field energy", "Spatial variogram"],
                    [[f"{short_scenario(r['scenario'])}/{r['track']}", LABELS[r['model_id']], str(r["probability_population"]),
                      fmt(r["field_energy"]), fmt(r["spatial_variogram"])] for r in dependence])))
        macros = summary.get("macro_comparisons", [])
        macro_table = None
        if macros:
            macro_table = (["Contrast / task", "Score", "Difference", "MCSE", "Pairs by DGP"], [[
                f"{LABELS.get(m['reference'], m['reference'])} vs {LABELS.get(m['other_model'], m['other_model'])} / {m['track']}", m["metric"],
                fmt(m.get("difference")), fmt(m.get("mcse")), ", ".join(f"{short_scenario(c['scenario'])}:{c['pairs']}/{c.get('attempted_pairs')}" for c in m["cells"])] for m in macros])
        sections.append(section(f"{run['label']}: paired comparisons", [
            "Negative differences favor the reference ESN. Macro estimates, when present, use the prespecified scenario weights; a missing scenario cell remains missing and weights are not renormalized. Their MCSE combines independent within-scenario paired variances. Intervals are pointwise and do not support simultaneous superiority claims.",
            "Qualification attrition is part of the result. A complete campaign can still have no usable native comparison in a cell. Confirmation planning needs at least five usable pairs in every primary cell, accounts for joint qualification, and fixes attempted replications before final seeds."], table=macro_table, figures=paired_figure(run, output)))
        sections.append(section(f"{run['label']}: error evolution", figures=lead_figures(run, output)))
        cpu = [r.get("total_pipeline_cpu_seconds") for r in run["receipts"]]
        total_cpu = sum(cpu) if all(_finite(value) for value in cpu) else None
        sections.append(section(f"{run['label']}: cost and reliability", [
            f"Recorded pipeline CPU summed over terminal jobs: {fmt(total_cpu, 5)} seconds. This sum includes tuning, final computation and final artifact serialization where the new accounting fields are present; it is not elapsed parallel campaign time.",
            "Fit wall time includes diagnostics. The diagnostic subcomponent is cumulative saved diagnostic-phase time and can include earlier invocations after resume; adding it again would double-count. Pipeline costs with interrupted unknown work are explicitly incomplete.",
            "Worker RSS is a process-lifetime high-water mark, including final artifact serialization. Reused-worker records are not isolated per-model memory increments, so this report does not infer a model-level accuracy-versus-memory ranking from them. The separately isolated forecast-ownership probe supports its narrower allocation comparison."],
            figures=[cost_figure(run, output), *accuracy_time_figure(run, output)]))
    if not data.get("audits"):
        sections.append(section("G1 known-parameter reference", ["No saved known-parameter audit was supplied. The renderer does not compute an oracle on demand. This required comparison remains unavailable until a source-bound audit is supplied with --audit."]))
    for audit in data.get("audits", []):
        run = next(r for r in runs if r["label"] == audit["label"])
        oracle = {(r["replication"], r["track"]): r for r in audit["document"]["oracle"]}
        comparisons = []
        for row in run["summary"]["table"]:
            if row["scenario"] != "g1_gaussian_ar":
                continue
            saved = [r for r in run["receipts"] if r["scenario"] == row["scenario"] and r["track"] == row["track"] and r["model_id"] == row["model_id"]]
            operational = [r["operational_scores"]["rmse"] - oracle[r["replication"], r["track"]]["scores"]["rmse"] for r in saved]
            native = [r["native_scores"]["crps"] - oracle[r["replication"], r["track"]]["scores"]["crps"] for r in saved
                      if r["qualified"] and not r["fallback"] and "crps" in r["native_scores"]]
            comparisons.append([row["track"], LABELS.get(row["model_id"], row["model_id"]),
                f"{fmt(float(np.mean(operational)))} (n={len(operational)})",
                f"{fmt(float(np.mean(native)) if native else None)} (n={len(native)})"])
        sections.append(section(f"{audit['label']}: G1 known-parameter reference", [
            "This reference knows the generating parameters and conditions on the same noisy observed-site history. It has privileged parameter information, but does not receive the latent state or future observations. It is a diagnostic reference, not another learned competitor.",
            "Table differences are learned-method loss minus oracle loss in original G1 response units. Operational RMSE uses all declared outcomes. Native CRPS uses each method's qualified datasets, and the oracle is restricted to those same datasets for that comparison. Negative realized differences can occur; optimal conditional information does not guarantee dominance on each finite realization.",
            f"Independent audit status: {audit['document']['status']}; {len(audit['document'].get('findings', []))} finding(s) retained in the copied audit. No reference or comparator was recomputed by this renderer."],
            table=(["Task", "Learned method", "Paired operational RMSE difference", "Paired native CRPS difference"], comparisons),
            figures=oracle_figure(audit, output), links=[("Complete reference audit", f"evidence/{audit['label']}_audit.json")]))
    if not data.get("calibrations"):
        sections.append(section("Separate post-fit calibration", [
            "No sealed raw-versus-calibrated study was supplied. Native interval coverage above describes the original predictive laws; it is not evidence that post-fit calibration has been completed or improved them."]))
    for index, calibration in enumerate(data.get("calibrations", [])):
        value = calibration["summary"]
        table = []
        for key, adjusted in value["calibrated"].items():
            target, level = key.rsplit("_", 1)
            raw = value["raw"].get("intervals", {}).get(level, {})
            table.append([f"{target} / {level}",
                f"{fmt(raw.get('coverage'))} -> {fmt(adjusted.get('coverage'))}",
                f"{fmt(raw.get('width'))} -> {fmt(adjusted.get('width'))}",
                f"{fmt(raw.get('interval_score'))} -> {fmt(adjusted.get('interval_score'))}"])
        sections.append(section(f"{calibration['label']}: raw versus calibrated intervals", [
            f"{short_scenario(value['scenario'])}, track {value['track']}, {LABELS.get(value['model_id'], value['model_id'])}; {value['blocks']} disjoint calibration blocks. Parameters are frozen at row {value['parameter_cutoff']}; both raw and adjusted bands are evaluated at the same later origin {value['origin_index']} on identical targets.",
            "O observations may update forecast state between calibration blocks. New-site calibration learns from reserved C sites and evaluates transfer to U. This is empirical calibration under serial dependence, not an exchangeability-based coverage guarantee.",
            f"Calibrated marginal WIS: {fmt(value.get('wis_marginal'))}. Interval adjustment adds no new joint distribution and leaves the raw predictive samples unchanged. Calibrated CRPS is therefore not invented. Saved band and raw-score calculations were independently verified by the read-only seal helper.",
            "Each table entry is raw -> calibrated. Marginal, horizon and simultaneous maps are distinct targets. Infinite bands, when required by a finite block rank, remain marked unavailable for finite width/score rather than being hidden."],
            table=(["Target / level", "Coverage", "Width", "Interval score"], table),
            links=[("Calibration numerical summary and verification seal", f"evidence/calibration_{index}_summary.json")]))
    for curve_index, curve in enumerate(data["curves"]):
        text = [f"Learning-curve evidence status: {curve['status']}.",
            "A valid history curve holds the final forecast interval and target sites fixed, varies the supplied historical window, and uses a common training-only normalization reference. A valid sensor curve nests observed sensors while preserving the unobserved target set. Fixed specifications and repeated tuning are distinct experiments."]
        if curve.get("unexecuted_values"):
            text.append("Unexecuted planned values: " + ", ".join(map(str, curve["unexecuted_values"])) + ". They are absent from the plots, not treated as failed or zero-valued results.")
        if not curve["runs"]:
            text.append("No completed curve conditions are included. This report cannot establish whether more history or more sensors improve prediction.")
        sections.append(section("More data and expressiveness: " + curve.get("design", {}).get("kind", "unavailable"), text,
            figures=curve_figures(curve, output, curve_index=curve_index),
            links=[("Matched information interventions and populations", "evidence/study_design_v03.md")]))
    for audit_index, audit in enumerate(data.get("curve_audits", [])):
        if audit["status"] != "complete":
            sections.append(section("Learning-curve audit: unavailable", [
                "The explicitly supplied optional audit is not available. No history choice or representation diagnostic is inferred from its absence."]))
            continue
        value = audit["document"]
        links = [("Source-bound learning-curve audit", f"evidence/curve_audit_{audit_index}.json")]
        if audit["kind"] == "history":
            rows = []
            for contrast in value["contrasts"]:
                macro = contrast["macro"] or {}
                pairs = ", ".join(f"{short_scenario(c['scenario'])}:{len(c['usable_replications'])}" for c in contrast["cells"])
                family = "Native" if contrast["score_family"] == "native_scores" else "Operational"
                decision = f"{fmt(contrast['margin'])} / {contrast['bound_passed']}" if contrast["is_decision_contrast"] else "descriptive"
                rows.append([f"{LABELS.get(contrast['model_id'], contrast['model_id'])} / {contrast['track']}",
                    family + " " + contrast["metric"].upper(), pairs, fmt(macro.get("difference")), fmt(macro.get("lower_one_sided95")), decision])
            sections.append(section("History length: recorded development choice", [
                f"Selected initial supplied history: {value['selected_initial_history']} rows. The prospective rule retains 1000 by default. It promotes 2000 only if both ESN tracks have at least three common-qualified pairs per mechanism and a macro normalized native-CRPS improvement lower bound of at least 0.02, long-history qualification is not worse, and complete summed pipeline cost is at most 2.5 times the shorter history.",
                f"Score bounds passed: {value['score_bounds_passed']}; long-history qualification not worse: {value['qualification_not_worse']}; cost gate passed: {value['cost_gate_passed']}. Summed pipeline time was {fmt(value['summed_pipeline_seconds']['1000'], 5)} seconds at 1000 and {fmt(value['summed_pipeline_seconds']['2000'], 5)} at 2000; long/short ratio {fmt(value['long_to_short_pipeline_time_ratio'])}.",
                "Differences are short minus long, divided by the same training-only scale; positive favors the longer history. Native CRPS and native RMSE use each model's common-qualified paired fields. Operational RMSE includes the declared fallbacks on all paired fields. Every model's effects remain visible, regardless of the ESN-centered choice.",
                "The lower bounds are saved, stratified, one-sided 95% percentile-bootstrap development summaries. Small samples do not establish formal interval coverage or simultaneous superiority. The 0.02 improvement and 2.5 cost margins are explicit practical preferences. Retaining 1000 when the gate fails is not evidence that additional data have no benefit. No confirmation outcomes entered this choice.",
                "Recorded retention reasons: " + ("; ".join(value["retention_reasons"]) if value["retention_reasons"] else "none; the declared promotion gate passed")],
                table=(["Model / task", "Score population", "Usable pairs", "Short - long", "Lower 95%", "Decision margin / pass"], rows), links=links))
        else:
            rows = []
            for row in value["rows"]:
                spatial, temporal = row.get("spatial", {}), row.get("temporal", {})
                amplification = row.get("fixed_reference_noise_amplification", {})
                rank = lambda item: f"{item['rank']}/{item['shape'][1]}" if item else "unavailable"
                rows.append([str(row["sensor_count"]), row["track"], str(row["qualified"]), rank(spatial), rank(temporal),
                    f"{fmt(amplification.get('mean_unit_iid_noise_variance_amplification'))} / {fmt(amplification.get('maximum_unit_iid_noise_variance_amplification'))}"])
            sections.append(section("Sensor geometry and representation", [
                "This audit uses one prespecified field and its saved ESN states. Spatial and temporal ranks use the recorded singular-value tolerance. Qualification is displayed even when a saved representation can be inspected after an unqualified fit.",
                value["basis_policy"],
                "The last column is mean / maximum unit-iid-noise variance amplification for the geometry-only normalized-ridge operator A = Phi_U (Phi_O' Phi_O / n + eta I)^(-1) Phi_O' / n. It is a fixed diagnostic of interpolation geometry; it is neither correlated-residual forecast variance nor a predictive score.",
                "Additional sensors can change fitted basis scaling, conditioning and identifiability as well as sample count. The fixed-reference diagnostic makes its basis choice explicit; it does not retroactively change any fitted model or prior. One field supports descriptive sensitivity, with no independent-replication MCSE or general sensor-benefit claim."],
                table=(["O sensors", "Task", "Qualified", "Phi rank / cols", "Z rank / cols", "IID noise mean / max"], rows), links=links))
    sections.extend(stress_sections(data.get("stress_audits", [])))
    extensions = data["extensions"]
    policy_entries = extensions.get("current_entries", current_extension_entries(extensions.get("studies", [])))
    extension_rows = [[r.get("name", r.get("id")).replace("_", " "), r["status"].replace("_", " "),
                       r.get("rationale", r.get("reason"))] for r in policy_entries]
    scope_links = [("Predeclared extension scope", f"evidence/p5_{index}.json")
                   for index, study in enumerate(extensions.get("studies", []))
                   if study.get("document", {}).get("scientific_results_in_this_file") is False]
    scope_links.extend(("Sealed release disposition overlay", f"evidence/p5_{index}.json")
                       for index, study in enumerate(extensions.get("studies", []))
                       if study.get("document", {}).get("schema") == "v03-release-extension-dispositions-1.0")
    sections.append(section("Extension dispositions", [f"Status: {extensions['status']}. These are supplied research decisions with their stated evidence, not automatic claims of improvement.",
        "The table shows one current disposition per extension. Final P5 findings supersede their earlier implementation scope. A separately sealed release overlay can update a disposition using later evidence; it binds the unchanged P5 base and every cited later receipt. Earlier scientific inputs remain immutable and individually linked.",
        "Grouped ridge, finite-slab regularized horseshoe, residual dynamics, anisotropy, calibration and larger feature maps should advance only when a diagnosed deficiency, compatible inference route and bounded development comparison justify them. Implemented optional routes may remain unvalidated at the benchmark scale."],
        table=(["Extension", "Disposition", "Reason / evidence"], extension_rows) if extension_rows else None,
        links=scope_links))
    for study_index, study in enumerate(extensions.get("studies", [])):
        if study.get("document", {}).get("replay_provenance"):
            sections.append(section("P5 execution replay and independent evidence", [
                "The final-source disposition records a fresh execution of the original P5 datasets, seeds, budgets and decisions. It reproduces the earlier numerical results and binds the unchanged historical disposition. This verifies execution under the current source and adds zero independent datasets.",
                "The historical and current executions are not pooled as separate replications. The ablation still has one development field per mechanism; its inference failures and negative predictive findings retain their original interpretation."],
                links=[("Replay provenance and unchanged historical evidence", f"evidence/p5_{study_index}.json")]))
        assessment = study.get("document", {}).get("path_count_assessment")
        if assessment:
            selected = assessment["selected_for_subsequent_studies"]
            table = []
            for case in assessment["cases"]:
                for row in case["rows"]:
                    table.append([short_scenario(case["scenario"]), str(row["paths"]),
                        fmt(row["normalized_absolute_crps_difference_from_512"]),
                        fmt(row["absolute_coverage90_difference_from_512"]),
                        fmt(row["conditional_predictive_mean_mcse_rms_over_training_scale"]),
                        str(row["passes_both_rules"])])
            sections.append(section("Predictive path budget: recorded development decision", [
                f"The separately predeclared decision selects {selected} paths for subsequent studies. The rule chooses the smallest of 128, 256 and 512 whose absolute CRPS difference from 512, divided by the common training scale, is at most 0.005 and whose absolute 90% coverage difference is at most 0.02 in both prespecified cases. Neither smaller count passes both cases.",
                "The sampling tool itself makes no automatic selection. This later policy decision is bound to the saved predeclared rule, reference receipts and sensitivity output inventories. It does not alter earlier ablation forecasts, which retain their original path budget.",
                assessment["limitations"],
                "A zero difference at 512 is equality to a finite reference, not proof of negligible predictive Monte Carlo error. Conditional mean MCSE is descriptive and excludes posterior MCMC error; it was not an extra adaptive selection criterion."],
                table=(["DGP", "Paths", "|CRPS diff| / train SD", "|Coverage diff|", "Mean MCSE RMS / train SD", "Rule passes"], table),
                links=[("Decision, predeclared rule and bound output inventories", f"evidence/p5_{study_index}.json")]))
        scientific = study.get("scientific_evidence")
        if not scientific:
            continue
        title = Path(study["path"]).parent.name
        paragraphs = [f"Terminal P5 mode: {scientific['mode']}; base benchmark: {scientific['base_run']}; qualified saved baseline: {scientific['qualified_baseline']}. The completed output inventory and original source-bound design were verified. A completed negative result remains in the report."]
        table = None
        if scientific["ablation"]:
            ablation = scientific["ablation"]
            table = (["Variant", "Qualified", "RMSE", "Native CRPS", "Fit invocation sec"], [[
                {"scalar_reference": "Scalar ridge", "full_residual_ar": "Full-residual AR", "grouped_ridge": "Grouped ridge"}.get(r["variant"], r["variant"]), str(r["qualified"]), fmt(r.get("scores", {}).get("rmse")),
                fmt(r.get("scores", {}).get("crps")), fmt(r.get("fit_invocation_seconds"), 4)] for r in ablation["rows"]])
            paragraphs.append("This is one prespecified development dataset with a contemporaneously refitted scalar reference, fixed observations and fixed feature design. Differences below are extension minus scalar; negative favors the extension. No across-dataset MCSE or general superiority claim is available from one case.")
            for pair in ablation["comparisons"]:
                difference = pair["loss_difference_extension_minus_scalar"]
                if difference is None:
                    paragraphs.append(f"{pair['variant']}: zero common-qualified pairs. Its native predictive comparison is unavailable because at least one fit did not qualify. Successful completion of the execution does not repair that scientific failure.")
                else:
                    paragraphs.append(f"{pair['variant']}: paired differences RMSE {fmt(difference['rmse'])}, MAE {fmt(difference['mae'])}, CRPS {fmt(difference['crps'])}; usable datasets = 1.")
                    if difference["rmse"] > 0 and difference["crps"] > 0:
                        paragraphs.append("Both RMSE and CRPS are worse than the scalar reference on this saved case. Passing the inference gate is useful engineering evidence, but this result does not support predictive promotion of the extension.")
            paragraphs.append("Fit invocation seconds can omit work before a recovered interruption. They are not an isolated end-to-end speed benchmark, and qualification never implies improved accuracy.")
        if scientific["path_stability"]:
            sensitivity = scientific["path_stability"]
            table = (["Paths", "RMSE", "CRPS", "CRPS - 512", "Mean MCSE RMS"], [[
                str(r["paths"]), fmt(r["scores"]["rmse"]), fmt(r["scores"]["crps"]),
                fmt(r["difference_from_512"]["crps"]), fmt(r["predictive_mean_mcse_rms"])] for r in sensitivity["rows"]])
            paragraphs.extend(["The 128/256/512 comparisons are nested prefixes of one saved 512-path forecast under one fixed posterior. Prefix hashes were independently checked against the saved joint paths. They are correlated Monte Carlo sensitivity conditions, not independent generated datasets.",
                "Predictive-mean MCSE is conditional on the stored posterior mixture and excludes posterior MCMC error. More predictive paths cannot rescue an unqualified chain. Exact empirical CRPS is reported for each prefix; no iid finite-ensemble correction is applied. The sampling tool does not select a path count; any separately supplied policy decision is reported explicitly above."])
        sections.append(section(f"Bounded extension evidence: {title}", paragraphs, table=table,
            links=[("P5 decisions, qualification and numerical results", f"evidence/p5_{study_index}.json")]))
    for diagnostic_index, diagnostic in enumerate(data.get("diagnostics", [])):
        if diagnostic["status"] != "complete":
            sections.append(section("Saved first-step diagnostic: unavailable", [
                "The explicitly supplied optional diagnostic is absent. No error decomposition is inferred from that absence."]))
            continue
        value = diagnostic["document"]
        decoder, one, alignment = value["decoder"], value["one_step"], value["alignment"]
        rows = [[str(row["lead"]), fmt(row["empirical_mean_rmse"]), fmt(row["decoder_projection_rmse"]),
                 fmt(row["predictive_sd_rms"]), fmt(row["conditional_empirical_mean_mcse_rms"])]
                for row in value["saved_path_leads"]]
        sections.append(section("G3 diagnosis: decoder restriction and rollout dispersion", [
            f"This saved track-A review checks the fixed reference fit and its 512-path sensitivity artifact. The causal feature/state replay, site ordering and response scale agree; reconstruction of the original first 64 saved observation paths differs by at most {fmt(alignment['first64_saved_observation_reconstruction_max_abs'])}. The report reads this diagnostic and does not repeat the replay.",
            f"The first-lead 64-draw conditional readout has RMSE {fmt(one['conditional_readout_mean64_rmse'])}. The evaluator-only projection of the noisy observed target onto the rank-{decoder['rank']} decoder span on {decoder['sites']} sites has residual RMSE {fmt(decoder['first_lead_unavoidable_outside_span_rmse'])}; {100 * decoder['first_lead_outside_span_fraction_of_conditional_readout_mse64']:.2f}% of this readout's realized squared error lies outside that fixed span. This is a bound for this target and decoder, not a universal irreducible-error or latent-field claim.",
            "The saved path dispersion grows at later leads. Additional predictive paths improve Monte Carlo precision of this existing predictive law; they do not expand the decoder or establish adequate stochastic rollout behavior. The result motivates a separately declared spatial-representation/dynamics investigation, with fresh validation before any promotion.",
            "The 64 conditional draws are a descriptive moment reference, not an exact posterior mean. This truth-based projection is an evaluator diagnostic only and never enters a forecast or model selection. Track B uses a different fit and target set; this replay does not verify its first step."],
            table=(["Lead", "Saved mean RMSE", "Projection residual RMSE", "Predictive SD RMS", "Mean MCSE RMS"], rows),
            links=[("Saved first-step review and input hashes", f"evidence/diagnostic_{diagnostic_index}.json")]))
    engineering = data["engineering"]
    memory = engineering["memory"]
    text = ["Scientific identity binds design, source, data, numerical runtime and transition policy. Worker count, memory ceiling and execution chunk limits belong to attempt receipts. Safe resume preserves scientific seeds and checkpoint trajectories; incompatible evidence is invalidated rather than reused.",
        "Memory admission combines a conservative workload estimate with host/cgroup headroom before each job. Jobs use one numerical thread each. A resource pause does not silently reduce model dimensions or relax diagnostics. Current benchmark scoring still materializes forecast samples; end-to-end disk-backed scoring is not claimed."]
    if engineering.get("memory_admission"):
        text.append("Shared cgroup file cache is recorded separately from the admission estimate. The scheduler credits at most half eligible clean file cache after conservatively excluding shared/tmpfs, dirty, writeback, mapped and unevictable bytes. Raw usage, hard limits, the 65% budget cap and up-to-2-GiB reserve remain visible and unchanged. Missing or invalid statistics revert to raw usage. This policy is a scheduling heuristic, not guaranteed reclaim or a claim of available RAM; no cgroup limits or reclaim controls are written.")
    if memory:
        a, b = memory["rows"]
        text.append(f"The isolated ownership probe used a {a['shape'][0]} x {a['shape'][1]} x {a['shape'][2]} float64 path array. Public-copy peak RSS was {a['peak_rss_bytes']/1024**2:.1f} MiB; private owned-transfer peak was {b['peak_rss_bytes']/1024**2:.1f} MiB. Incremental peak reduction was {memory['incremental_peak_reduction_bytes']/1024**2:.1f} MiB with identical numerical payload hashes. This measures the allocation/constructor workload only, not model-fit memory or universal forecasting speed.")
    tests = engineering["tests"]
    test_rows = [[r["file"], str(r["tests"]), str(r["failures"] + r["errors"]), str(r["skipped"])] for r in tests]
    text.append("Counts below are JUnit cases, including separately recorded subtests; they are not counts of distinct pytest test functions. These saved snapshots retain their original timestamps and do not certify later changes. Release/publication status is verified only by explicit supplied release receipts; building this report does not publish a release.")
    if not engineering["release"]:
        text.append("No release-verification receipt is included in this synthesis. Repository publication and asset checks remain separately recorded release gates.")
    sections.append(section("Engineering evidence and reproducibility", text,
        table=(["Saved test receipt", "JUnit cases", "Failures + errors", "Skipped"], test_rows) if test_rows else None))
    if engineering["release"]:
        release_rows = []
        for item in engineering["release"]:
            receipt = item["receipt"]
            status = (f"{sum(j.get('conclusion') == 'success' for j in receipt['jobs'])}/{len(receipt['jobs'])} jobs succeeded"
                      if item["kind"] == "saved remote CI" else str(receipt["passed"]))
            release_rows.append([item["file"], item["kind"], status,
                receipt.get("commit", receipt.get("source_fingerprint", "per-file source hashes in receipt"))])
        sections.append(section("Release and remote-CI snapshots", [
            "These are saved gate receipts for their recorded source or commit. Historical successful CI does not certify the report's later HEAD. A local isolated-installation gate does not establish public publication, and an audit or resource plan is not a release receipt.",
            "Publication still requires explicit exact-commit verification and asset readback in the release workflow. This renderer performs no publication or remote CI action. PDF tables show 12-character commit/source prefixes; full identities remain in the HTML and JSON evidence."],
            table=(["Saved receipt", "Gate", "Recorded outcome", "Commit / source scope"], release_rows),
            links=[("Complete engineering receipts", "evidence/engineering_evidence.json")]))
    media_links = []
    for group in data["media"]:
        for item in group["files"]:
            if Path(item["path"]).suffix.lower() in {".mp4", ".gif"}:
                media_links.append((f"{group['label']} / {Path(item['name']).name}", f"media/{group['label']}/{item['name']}"))
    sections.append(section("Animations and visual comparisons", [
        "Existing core report and model-comparison renderers display observed fields, saved forecast means and signed errors on shared scales. Gray cells mark unscored locations. Illustration replications are selected explicitly; animations are not independent evidence for a model ranking.",
        "Unqualified native trajectories, where displayed by those renderers, are diagnostic illustrations and are labeled accordingly. Operational score comparisons still use the declared fallback.",
        "No linked animation assets were supplied." if not media_links else "The linked assets passed their original report inventories and are copied without rerendering. Open the HTML report for playable video."], links=media_links))
    sections.append(section("Limits and decisions supported by this evidence", [
        "Improved implementation, exact recovery and stricter information boundaries make an experiment more trustworthy; they do not guarantee improved forecast losses. Development choices need a fresh, frozen confirmation design before broad performance claims.",
        "A failed gate remains a negative result. Native scores describe a selected qualified population, while operational scores include fallback behavior. Do not hide attrition, pool sites as replications, or reinterpret MCSE planning as hypothesis-test power.",
        "Finite-grid excitable-field results concern the declared semidiscrete generator. They do not establish continuum mesh convergence. Neural parameter uncertainty, encoder uncertainty, sparse-model approximations and experimental prior combinations require their explicitly documented interpretation.",
        "Timing observations depend on the declared runtime, host and concurrency. Recovery after lost worker time can leave incomplete cost accounting, which is reported. The constructor allocation probe does not prove scalability of the full inference-and-scoring pipeline.",
        "Machine-readable manifests, summaries and the report input/output inventory accompany this report. The renderer verifies saved data and artifacts, reduces saved scalar scores for consistency, and never fits, simulates, forecasts or chooses a favorable replication."],
        links=[("Synthesis evidence and source identities", "evidence_summary.json"), ("Complete report artifact inventory", "report_manifest.json")]))
    return sections


def _html_table(table):
    if not table:
        return ""
    headers, rows = table
    return "<div class='table-wrap'><table><thead><tr>" + "".join(f"<th>{html.escape(str(x))}</th>" for x in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def render_html(sections, output, source):
    navigation = "".join(f'<a href="#section-{i}">{html.escape(s["title"])}</a>' for i, s in enumerate(sections))
    pieces = []
    for i, item in enumerate(sections):
        paragraphs = "".join(f"<p>{html.escape(p)}</p>" for p in item["paragraphs"])
        figures = "".join(f'<figure><img src="{html.escape(f["path"])}" alt="{html.escape(f["caption"] or item["title"])}"><figcaption>{html.escape(f["caption"])}</figcaption></figure>' for f in item["figures"])
        links = "".join(f'<li><a href="{html.escape(href)}">{html.escape(label)}</a></li>' for label, href in item["links"])
        videos = "".join(f'<video controls preload="none" playsinline src="{html.escape(href)}"></video>' for _, href in item["links"] if href.endswith(".mp4"))
        pieces.append(f'<section id="section-{i}"><div class="eyebrow">{i+1:02d} / evidence</div><h2>{html.escape(item["title"])}</h2>{paragraphs}{_html_table(item["table"])}{figures}<ul>{links}</ul>{videos}</section>')
    css = """body{margin:0;background:#f0f5f7;color:#10283f;font:16px/1.65 system-ui,sans-serif}main{max-width:1140px;margin:auto;padding:30px 24px 70px}header{background:#10283f;color:#fff;border-radius:20px;padding:48px 44px}header p{max-width:800px;color:#c8dbe6}h1{font-size:clamp(36px,5vw,62px);line-height:1.05;letter-spacing:-.04em;max-width:850px;margin:18px 0 25px}h2{font-size:28px;line-height:1.2;margin:12px 0 24px;letter-spacing:-.02em}.eyebrow{text-transform:uppercase;letter-spacing:.18em;font-size:11px;font-weight:700;color:#008c82}header .eyebrow{color:#77d6cc}section{background:#fff;border-radius:16px;padding:32px 36px;margin:24px 0;scroll-margin-top:20px}nav{display:flex;flex-wrap:wrap;gap:9px;margin:24px 0}nav a{font-size:12px;padding:7px 11px;background:#e2edef;border-radius:7px;text-decoration:none}a{color:#007c75}figure{margin:28px 0}img,video{width:100%;max-width:100%;height:auto}figcaption{font-size:13px;color:#5c7083;line-height:1.55;margin-top:10px}table{border-collapse:collapse;width:100%;font-size:13px;margin:15px 0}td,th{text-align:left;vertical-align:top;padding:10px 12px;border-bottom:1px solid #dce7ed}th{background:#edf5f6;color:#10283f}tr:nth-child(even) td{background:#f8fbfc}.table-wrap{overflow-x:auto}code{overflow-wrap:anywhere;font-size:11px}footer{font-size:12px;color:#5c7083;margin-top:30px}@media(max-width:650px){main{padding:12px}header,section{padding:25px 22px}h2{font-size:24px}}"""
    content = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bayesian spatiotemporal forecasting / v0.3</title><style>{css}</style></head><body><main><header><div class="eyebrow">Bayes-STESN / v0.3 / study synthesis</div><h1>Bayesian spatiotemporal forecasting</h1><p>Development comparisons and inference: model hierarchy, predictive performance, qualification and computational cost. A synthesis of saved, versioned experiments.</p><p><a style="color:#77d6cc" href="bayes_stesn_v03_report.pdf">Download the PDF report</a></p></header><nav>{navigation}</nav>{"".join(pieces)}<footer>Scientific source fingerprint: <code>{source}</code><br>Saved-artifact renderer. No model fitting or forecasting occurs during report generation.</footer></main></body></html>'
    (output / "index.html").write_text(content, encoding="utf-8")


def render_pdf(sections, output, source):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                                   Image, Table, TableStyle, KeepTogether)
    from matplotlib import font_manager
    from PIL import Image as PILImage

    for name, weight in (("StudySans", "normal"), ("StudySansBold", "bold")):
        pdfmetrics.registerFont(TTFont(name, font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight=weight))))
    pdfmetrics.registerFontFamily("StudySans", normal="StudySans", bold="StudySansBold")
    styles = {"body": ParagraphStyle("Body", fontName="StudySans", fontSize=9.4, leading=14.1, textColor=colors.HexColor(NAVY), spaceAfter=9),
              "title": ParagraphStyle("Title", fontName="StudySansBold", fontSize=34, leading=38, textColor=colors.HexColor(NAVY), spaceAfter=25),
              "h2": ParagraphStyle("Heading", fontName="StudySansBold", fontSize=20, leading=25, textColor=colors.HexColor(NAVY), spaceAfter=15, keepWithNext=True),
              "caption": ParagraphStyle("Caption", fontName="StudySans", fontSize=8, leading=11.5, textColor=colors.HexColor(GRAY), spaceAfter=12),
              "cell": ParagraphStyle("Cell", fontName="StudySans", fontSize=7.7, leading=11, textColor=colors.HexColor(NAVY)),
              "head": ParagraphStyle("Head", fontName="StudySansBold", fontSize=7.5, leading=10.5, textColor=colors.HexColor(NAVY))}
    width = A4[0] - 92
    doc = SimpleDocTemplate(str(output / "bayes_stesn_v03_report.pdf"), pagesize=A4, rightMargin=46, leftMargin=46,
        topMargin=49, bottomMargin=45, title="Bayes-STESN v0.3 study synthesis", author="bayes-stesn project", pageCompression=1)
    story = [Spacer(1, 70), Paragraph("BAYES-STESN / v0.3", styles["caption"]),
             Paragraph("Bayesian spatiotemporal forecasting", styles["title"]),
             Paragraph("Development comparisons and inference", styles["h2"]),
             Paragraph("Model hierarchy, exact inference changes, qualification, predictive performance and computational cost. Built from saved, versioned evidence.", styles["body"]),
             Spacer(1, 25), Paragraph("Read the scope before the scores. Development evidence does not establish a confirmatory model ranking. Each result retains its actual qualification population, study design and numerical provenance.", styles["body"]),
             Spacer(1, 30), Paragraph("Open index.html for linked animations and full numerical evidence. The PDF is a static companion; the input and output inventories are recorded in report_manifest.json.", styles["caption"]), PageBreak()]
    for index, item in enumerate(sections):
        if index:
            story.append(PageBreak())
        story.extend([Paragraph(f"{index+1:02d} / EVIDENCE", styles["caption"]), Paragraph(html.escape(item["title"]), styles["h2"])])
        story.extend(Paragraph(html.escape(p), styles["body"]) for p in item["paragraphs"])
        if item["table"]:
            headers, rows = item["table"]
            fractions = {2: [.26, .74], 3: [.24, .19, .57], 4: [.52, .16, .18, .14], 5: [.12, .09, .38, .23, .18]}.get(len(headers), [1 / len(headers)] * len(headers))
            if headers[0] == "Contrast / task":
                fractions = [.36, .1, .15, .13, .26]
            elif headers[0] == "Model" and len(headers) == 5:
                fractions = [.3, .16, .18, .18, .18]
            elif headers[0] == "DGP / task":
                fractions = [.14, .34, .12, .2, .2]
            elif headers[0] == "Task" and len(headers) == 4:
                fractions = [.08, .34, .29, .29]
            elif headers[0] == "Target / level":
                fractions = [.28, .24, .24, .24]
            elif headers[0] == "Variant":
                fractions = [.25, .13, .18, .2, .24]
            elif len(headers) == 5 and headers[1] == "Frozen temporal design":
                fractions = [.13, .27, .22, .2, .18]
            elif headers[0] == "Intervention":
                fractions = [.32, .12, .2, .2, .16]
            elif headers[0] == "Model / task" and len(headers) == 6:
                fractions = [.23, .17, .07, .18, .19, .16] if headers[1] == "Score population" else [.23, .16, .17, .14, .15, .15]
            display_rows = rows
            if headers[-1] == "Commit / source scope":
                display_rows = [list(row[:-1]) + [str(row[-1])[:12] + "..." if re.fullmatch(r"[a-f0-9]{40,64}", str(row[-1])) else row[-1]] for row in rows]
            table = Table([[Paragraph(html.escape(str(x)), styles["head" if i == 0 else "cell"]) for x in row]
                           for i, row in enumerate([headers, *display_rows])], colWidths=[width * f for f in fractions], repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PALE)),
                ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7), ("LINEBELOW", (0, 0), (-1, -1), .35, colors.HexColor("#DCE7ED"))]))
            story.extend([table, Spacer(1, 15)])
        for figure in item["figures"]:
            path = output / figure["path"]
            with PILImage.open(path) as bitmap:
                image_width, image_height = bitmap.size
            scale = min(width / image_width, 525 / image_height)
            block = [Image(str(path), width=image_width * scale, height=image_height * scale)]
            if figure["caption"]:
                block.extend([Spacer(1, 7), Paragraph(html.escape(figure["caption"]), styles["caption"])])
            story.append(KeepTogether(block))
        if item["links"]:
            story.append(Paragraph("Companion evidence: " + "; ".join(html.escape(label) for label, _ in item["links"]) + ". Open the HTML version for links.", styles["caption"]))
    def footer(canvas, document):
        canvas.setStrokeColor(colors.HexColor("#DCE7ED"))
        canvas.line(46, 34, A4[0] - 46, 34)
        canvas.setFillColor(colors.HexColor(GRAY))
        canvas.setFont("StudySans", 7)
        canvas.drawString(46, 22, "BAYES-STESN v0.3  |  SAVED-ARTIFACT SYNTHESIS")
        canvas.drawRightString(A4[0] - 46, 22, f"{document.page}")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def export_evidence(data, output):
    folder = output / "evidence"
    folder.mkdir(parents=True, exist_ok=True)
    for run in data["runs"]:
        write_json(folder / f"{run['label']}_manifest.json", run["manifest"])
        write_json(folder / f"{run['label']}_summary.json", run["summary"])
        write_json(folder / f"{run['label']}_selected_configurations.json", run.get("selections", []))
        write_json(folder / f"{run['label']}_job_scores.json", [{key: row.get(key) for key in
            ("scenario", "track", "model_id", "replication", "qualified", "fallback", "operational_scores", "native_scores",
             "training_scale", "normalization_reference", "total_pipeline_seconds", "total_pipeline_cpu_seconds",
             "process_peak_rss_bytes", "memory_measurement", "timing_scope", "compute_accounting_complete")}
            for row in run["receipts"]])
    if data["coordinate"]:
        write_json(folder / "coordinate_manifest.json", data["coordinate"]["manifest"])
        write_json(folder / "coordinate_summary.json", data["coordinate"]["summary"])
    for index, curve in enumerate(data["curves"]):
        if curve.get("design"):
            write_json(folder / f"curve_{index}_design.json", curve["design"])
            for run in curve["runs"]:
                write_json(folder / f"curve_{index}_{run['value']}_summary.json", run["summary"])
    for index, audit in enumerate(data.get("curve_audits", [])):
        write_json(folder / f"curve_audit_{index}.json", audit)
    for index, audit in enumerate(data.get("stress_audits", [])):
        write_json(folder / f"stress_audit_{index}.json", audit)
    write_json(folder / "engineering_evidence.json", data["engineering"])
    write_json(folder / "extension_dispositions.json", data["extensions"])
    for index, study in enumerate(data["extensions"].get("studies", [])):
        write_json(folder / f"p5_{index}.json", study)
    for audit in data.get("audits", []):
        write_json(folder / f"{audit['label']}_audit.json", audit["document"])
    for index, calibration in enumerate(data.get("calibrations", [])):
        write_json(folder / f"calibration_{index}_summary.json", calibration["summary"])
        write_json(folder / f"calibration_{index}_seal.json", calibration["seal"])
    for index, diagnostic in enumerate(data.get("diagnostics", [])):
        write_json(folder / f"diagnostic_{index}.json", diagnostic)
    for index, check in enumerate(data.get("prior_checks", [])):
        write_json(folder / f"prior_check_{index}.json", check)
    for name in ("docs/models.md", "docs/derivations/07_sampler_coordinates_recovery.md", "docs/study_design_v03.md"):
        shutil.copyfile(ROOT / name, folder / Path(name).name)
    for group in data["media"]:
        for item in group["files"]:
            destination = output / "media" / group["label"] / item["name"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item["path"], destination)


def render(data, inputs, output):
    output = Path(output).resolve()
    # Never overwrite a numerical run, one of its ancestors, or any source input.
    sources = [Path(p).resolve() for p in inputs.files]
    if any(path.is_relative_to(output) for path in sources):
        raise ValueError("Report output must not contain any evidence/source input")
    if output.exists() and any(output.iterdir()) and not (output / "report_manifest.json").exists():
        raise ValueError("Use an empty output directory or a previously generated v0.3 synthesis")
    if (output / "report_manifest.json").exists():
        old = read_json(output / "report_manifest.json")
        if old.get("schema") != "saved-v03-study-report-1.0":
            raise ValueError("Existing directory belongs to another report recipe")
        for item in old.get("files", []):
            path = safe_member(output, item["path"])
            if sha256(path) != item["sha256"]:
                raise ValueError("Existing generated report has unrecorded edits")
        # Remove only earlier inventoried outputs so stale plots cannot masquerade as current evidence.
        for item in old.get("files", []):
            safe_member(output, item["path"]).unlink()
    output.mkdir(parents=True, exist_ok=True)
    export_evidence(data, output)
    sections = sections_for(data, output)
    render_html(sections, output, data["source_fingerprint"])
    render_pdf(sections, output, data["source_fingerprint"])
    evidence = {"schema": "v03-synthesis-evidence-1.0", "source_fingerprint": data["source_fingerprint"],
        "runs": [{"label": r["label"], "path": r["path"], "identity": r["manifest"]["identity"],
            "design_identity": r["manifest"]["design_identity"], "stage": r["manifest"]["stage"],
            "jobs": len(r["receipts"])} for r in data["runs"]],
        "coordinate": None if not data["coordinate"] else {"identity": data["coordinate"]["manifest"]["identity"],
            "source_fingerprint": data["coordinate"]["manifest"]["source_fingerprint"],
            "ancillary_source_exception": data["coordinate"]["ancillary_source_exception"]},
        "curves": [{k: v for k, v in curve.items() if k not in {"runs", "design"}} for curve in data["curves"]],
        "extension_status": data["extensions"]["status"], "render_only": True}
    evidence["audits"] = [{"label": audit["label"], "identity": audit["document"]["identity"]} for audit in data.get("audits", [])]
    evidence["calibrations"] = [{"label": value["label"], "identity": value["seal"]["identity"]} for value in data.get("calibrations", [])]
    evidence["curve_audits"] = [{"status": value["status"], "identity": value.get("document", {}).get("identity")}
                               for value in data.get("curve_audits", [])]
    evidence["stress_audits"] = [{"status": value["status"], "identity": value.get("document", {}).get("identity")}
                                for value in data.get("stress_audits", [])]
    evidence["diagnostics"] = [{"status": value["status"], "identity": value.get("document", {}).get("identity")}
                               for value in data.get("diagnostics", [])]
    evidence["prior_checks"] = [{"status": value["status"], "identity": value.get("document", {}).get("identity")}
                               for value in data.get("prior_checks", [])]
    write_json(output / "evidence_summary.json", evidence)
    inputs.verify_unchanged()
    files = [{"path": str(p.relative_to(output)), "bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(output.rglob("*")) if p.is_file() and p.name != "report_manifest.json"]
    import matplotlib
    import reportlab
    receipt = {"schema": "saved-v03-study-report-1.0", "render_only": True,
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_fingerprint": data["source_fingerprint"],
        "renderer_sha256": sha256(Path(__file__)), "command": sys.argv,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "matplotlib": matplotlib.__version__, "reportlab": reportlab.Version},
        "inputs": list(inputs.files.values()), "files": files, "evidence": evidence,
        "figure_typography": [{"path": figure["path"], "minimum_pdf_font_pt": figure["minimum_pdf_font_pt"]}
                              for section in sections for figure in section["figures"]],
        "checks": {"manifests": True, "complete_scientific_results": True,
                   "artifact_hashes": True, "saved_scalar_reductions": True, "inputs_unchanged": True,
                   "pdf_visual_qa": "pending; render and inspect before publication"}}
    receipt["identity"] = content_hash(receipt)
    write_json(output / "report_manifest.json", receipt)
    return receipt


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--coordinate", type=Path, help="Completed sampler-coordinate development directory")
    command.add_argument("--run", action="append", type=named_path, default=[], metavar="LABEL=DIRECTORY")
    command.add_argument("--curve", action="append", default=[], type=Path,
                         help="Optional resolved curve directory or curve_design.json; repeat for separate history/sensor designs")
    command.add_argument("--curve-audit", action="append", default=[], type=Path,
                         help="Optional source-bound history-choice or sensor-representation JSON; requires its matching --curve")
    command.add_argument("--stress-audit", action="append", default=[], type=Path,
                         help="Optional sealed G5/G6 paired-stress summary; requires its completed control --run")
    command.add_argument("--extensions", action="append", default=[], type=Path,
                         help="Optional scope JSON or completed P5 directory/decisions.json; repeat for multiple studies")
    command.add_argument("--audit", action="append", type=named_path, default=[], metavar="RUN_LABEL=JSON", help="Saved source-bound reference audit and G1 known-parameter scores")
    command.add_argument("--calibration", action="append", type=named_path, default=[], metavar="RUN_LABEL=DIRECTORY", help="Saved raw/calibrated study with calibration_evidence.json seal")
    command.add_argument("--diagnostic", action="append", type=Path, default=[], help="Optional saved source-bound G3 first-step diagnostic JSON; no replay")
    command.add_argument("--prior-check", action="append", type=Path, default=[], help="Saved conditional fixed-design prior assessment; requires its matching complete P5 --extensions inputs")
    command.add_argument("--media", action="append", type=named_path, default=[], metavar="RUN_LABEL=DIRECTORY", help="Existing core report/comparison assets; no new animation is generated")
    command.add_argument("--allow-coordinate-source", help="Explicit exact SHA for historical ancillary coordinate evidence only")
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--check-inputs", action="store_true", help="Validate only; do not create figures, HTML or PDF")
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    data, inputs = collect(args)
    if args.check_inputs:
        inputs.verify_unchanged()
        print(json.dumps({"valid": True, "input_files": len(inputs.files), "runs": [r["label"] for r in data["runs"]],
            "coordinate": data["coordinate"] is not None, "curve_status": [curve["status"] for curve in data["curves"]],
            "extension_status": data["extensions"]["status"], "source_fingerprint": data["source_fingerprint"]}, sort_keys=True))
    else:
        receipt = render(data, inputs, args.output)
        print(json.dumps({"output": str(args.output.resolve()), "identity": receipt["identity"],
                          "files": len(receipt["files"]), "pdf_visual_qa": "pending"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
