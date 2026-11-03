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
from pathlib import Path
import re
import shutil
import sys
from xml.etree import ElementTree
import zipfile

import numpy as np

from bayes_stesn.io import content_hash, load_state


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
SCENARIOS = {"g1_gaussian_ar": "G1 / Gaussian spatial AR",
             "g2_spectral_transport": "G2 / Spectral transport",
             "g3_excitable_field": "G3 / Excitable field",
             "g4_existing_fronts": "G4 / Moving fronts",
             "g5_noise_stress": "G5 / Noise stress",
             "g6_velocity_shift": "G6 / Hidden velocity shift"}
FOCUS = ("bayes_esn_ridge", "separable_stgp", "basis_gru", "eqesn_rollout",
         "persistence", "spatial_persistence")


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
    receipts, data_ids, observation_hashes = [], {}, {}
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
                    receipts.append(receipt)
    if summary.get("jobs") != len(receipts) or summary.get("scheduled_jobs") != len(receipts):
        raise ValueError("Completed study count disagrees with its declared design")
    validate_summary(summary, receipts)
    return {"label": label, "path": str(path), "manifest": manifest, "summary": summary,
            "receipts": receipts, "data_identities": list(data_ids.values()),
            "data_observation_hashes": observation_hashes}


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


def load_extensions(path, inputs, current_source):
    if path is None or not Path(path).exists():
        return {"status": "not supplied" if path is None else "not available", "entries": []}
    value = inputs.json(path)
    if not isinstance(value, dict):
        raise ValueError("Extension dispositions must be a JSON mapping")
    if "identity" in value:
        identity(value, label="Extension disposition")
    if value.get("source_fingerprint", current_source) != current_source:
        raise ValueError("Extension dispositions refer to a different current source")
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
    if value.get("manifest_identity"):
        manifest = inputs.json(Path(path).resolve().with_name("manifest.json"))
        identity(manifest, label="Extension study manifest")
        if manifest["identity"] != value["manifest_identity"] or manifest.get("source_fingerprint") != current_source:
            raise ValueError("Extension decisions do not match their source-bound study manifest")
    return {"status": status, "entries": entries, "path": str(Path(path).resolve()), "document": value}


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
    for path in sorted(folder.glob("*release*.json")):
        release.append({"file": path.name, "receipt": inputs.json(path)})
    return {"memory": memory, "tests": tests, "release": release}


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
    curve = load_curve(args.curve, inputs, source)
    extensions = load_extensions(args.extensions, inputs, source)
    media = load_media(args.media or [], runs, inputs)
    engineering = load_engineering(inputs)
    for name in ("docs/models.md", "docs/derivations/07_sampler_coordinates_recovery.md", "docs/study_design_v03.md"):
        inputs.add(ROOT / name)
    return {"source_fingerprint": source, "coordinate": coordinate, "runs": runs, "curve": curve,
            "extensions": extensions, "media": media, "engineering": engineering}, inputs


def fmt(value, digits=3):
    return "unavailable" if value is None else f"{value:.{digits}g}"


def short_scenario(value):
    return value.split("_")[0].upper()


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
        "axes.labelsize": 9, "axes.titleweight": "bold", "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#CDD9E1", "text.color": NAVY,
        "axes.labelcolor": GRAY, "xtick.color": GRAY, "ytick.color": GRAY,
        "figure.facecolor": "white", "axes.facecolor": "white", "grid.color": "#E3EBEF",
        "savefig.dpi": 210, "svg.fonttype": "none", "pdf.fonttype": 42})
    return plt


def save_figure(figure, output, name, caption):
    path = output / "figures" / (name + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", pad_inches=.18)
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=.18)
    plotting().close(figure)
    return {"kind": "figure", "path": str(path.relative_to(output)), "caption": caption}


def coordinate_figure(data, output):
    plt = plotting()
    rows = data["summary"]["rows"]
    cases = list(dict.fromkeys((r["scenario"], r["replication"]) for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(10.6, max(3.1, .63 * len(cases) + 1.4)), layout="constrained")
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
    axes[0].legend(loc="lower right", fontsize=7)
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
            fig, axes = plt.subplots(len(subset), 2, figsize=(10.8, 3.7 * len(subset)), squeeze=False, layout="constrained")
            for k, scenario in enumerate(subset):
                selected = [r for r in rows if r["track"] == track and r["scenario"] == scenario]
                for ax, metric in zip(axes[k], ("rmse", "crps")):
                    for i, row in enumerate(selected):
                        value, error = row.get(metric), row.get(metric + "_mcse")
                        if value is not None:
                            ax.errorbar(value, i, xerr=None if error is None else 2 * error, fmt="o", capsize=3,
                                        color=COLORS.get(row["model_id"], GRAY), markersize=5)
                    population = "operational_population" if metric == "rmse" else "probability_population"
                    ax.set_yticks(range(len(selected)), [f"{LABELS.get(r['model_id'], r['model_id'])}  n={r[population]}" for r in selected], fontsize=7)
                    ax.invert_yaxis()
                    ax.grid(axis="x", alpha=.6)
                    ax.set_title(f"{short_scenario(scenario)} / {metric.upper()} / track {track}", loc="left")
                    ax.set_xlabel("Original response units; smaller is better")
            figures.append(save_figure(fig, output, f"{run['label']}_scores_{track}_{offset}",
                "Dots are means of independent dataset losses; whiskers are +/-2 MCSE where at least two datasets contribute. Operational RMSE includes declared fallbacks. CRPS uses each method's qualified native distributions; marginal CRPS dots can involve different datasets. Use paired contrasts for direct comparisons."))
    return figures


def paired_figure(run, output):
    plt = plotting()
    rows = [r for r in run["summary"].get("paired_comparisons", [])
            if r.get("reference", "bayes_esn_ridge") == "bayes_esn_ridge" and r["other_model"] in ("separable_stgp", "basis_gru")]
    if not rows:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(10.8, max(3.5, .3 * len(rows) + 1.8)), layout="constrained")
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
        ax.set_yticks(range(len(rows)), [f"{short_scenario(r['scenario'])}/{r['track']} vs {LABELS[r['other_model']]}  n={r.get(metric + '_population', 0)}" for r in rows], fontsize=7)
        ax.invert_yaxis()
        ax.axvline(0, color=GRAY, ls="--", lw=.8)
        ax.grid(axis="x", alpha=.5)
        ax.set_title(f"Paired normalized {metric.upper()}", loc="left")
        ax.set_xlabel("ESN minus comparator; negative favors ESN")
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
        fig, axes = plt.subplots(1, 2, figsize=(10.8, max(3.3, .3 * len(selected) + 1.4)), layout="constrained")
        for ax, field_name, title in zip(axes, ("coverage_90", "interval_score_90"),
                                       ("Marginal 90% interval coverage", "90% interval score")):
            for i, row in enumerate(selected):
                if row.get(field_name) is not None:
                    ax.scatter(row[field_name], i, color=COLORS[row["model_id"]], s=28)
            ax.set_yticks(range(len(selected)), [f"{short_scenario(r['scenario'])} / {LABELS[r['model_id']]}  n={r['probability_population']}" for r in selected], fontsize=7)
            ax.set_title(f"Track {track} / {title}", loc="left")
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=.4)
        axes[0].axvline(.9, color=GRAY, ls="--", lw=.8)
        axes[0].set_xlim(0, 1.03)
        axes[0].set_xlabel("Qualified native population; target = 0.90")
        axes[1].set_xlabel("Original response units; smaller is better")
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
            fig, axes = plt.subplots(3, len(subset), figsize=(10.6 if len(subset) == 2 else 7, 7.5), squeeze=False, layout="constrained")
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
                                label=f"{LABELS[model]} (n={len(values)})")
                axes[0, column].set_title(f"{short_scenario(scenario)} / track {track}", loc="left")
                for index, title in enumerate(("Operational RMSE", "Operational signed bias", "Qualified-native CRPS")):
                    axes[index, column].set_ylabel(title)
                    axes[index, column].grid(alpha=.5)
                    axes[index, column].legend(fontsize=6, ncol=2, loc="upper right")
                axes[1, column].axhline(0, color=GRAY, lw=.6)
                axes[2, column].set_xlabel("Forecast lead from the fixed origin")
            output_figures.append(save_figure(fig, output, f"{run['label']}_leads_{track}_{offset}",
                "Mean of saved dataset-level lead scores, without future-data assimilation. Bias is forecast mean minus observed response. Model subset is fixed for legibility; complete scores remain in the machine summaries. Native CRPS populations vary by model. No fit or forecast was recomputed."))
    return output_figures


def cost_figure(run, output):
    plt = plotting()
    models = list(dict.fromkeys(r["model_id"] for r in run["receipts"]))
    fig, axes = plt.subplots(1, 2, figsize=(10.6, max(3.5, len(models) * .34 + 1)), layout="constrained")
    for i, model in enumerate(models):
        rows = [r for r in run["receipts"] if r["model_id"] == model]
        total = np.mean([r.get("total_pipeline_seconds", r["total_seconds"] + r.get("tuning_seconds", 0)) for r in rows])
        axes[0].barh(i, total, color=COLORS.get(model, GRAY), height=.6)
        qualified = sum(r["qualified"] for r in rows)
        axes[1].barh(i, qualified / len(rows), color=TEAL, height=.6)
        axes[1].barh(i, 1 - qualified / len(rows), left=qualified / len(rows), color="#E7B38B", height=.6)
        axes[1].text(1.03, i, f"{qualified}/{len(rows)}", va="center", fontsize=8)
    for ax in axes:
        ax.set_yticks(range(len(models)), [LABELS.get(m, m) for m in models], fontsize=8)
        ax.invert_yaxis()
    axes[0].set_xlabel("Mean recorded pipeline seconds / job")
    axes[0].set_title("Tuning + final fit + forecast + scoring", loc="left")
    axes[0].grid(axis="x", alpha=.4)
    axes[1].set_xlim(0, 1.24)
    axes[1].set_xticks([0, .5, 1], ["0%", "50%", "100%"])
    axes[1].set_xlabel("Qualified fraction; tan = declared fallback")
    axes[1].set_title("All declared final model/task jobs", loc="left")
    incomplete = sum(not r.get("compute_accounting_complete", False) for r in run["receipts"])
    return save_figure(fig, output, run["label"] + "_cost_qualification", f"Recorded costs averaged over the run's declared scenario/task mix; this is not machine-independent throughput. {incomplete} jobs have incomplete compute accounting after infrastructure interruption; their recorded times are lower bounds. Failed scientific attempts remain in cost and operational populations. Process peak RSS is kept in raw receipts with its measurement scope.")


def curve_figures(curve, output):
    plt = plotting()
    if not curve["runs"]:
        return []
    runs = sorted(curve["runs"], key=lambda r: r["value"])
    keys = sorted({(r["scenario"], r["track"]) for run in runs for r in run["summary"]["table"]})
    figures = []
    for offset in range(0, len(keys), 2):
        chunk = keys[offset:offset + 2]
        fig, axes = plt.subplots(len(chunk), 2, figsize=(10.6, 3.3 * len(chunk)), squeeze=False, layout="constrained")
        for k, (scenario, track) in enumerate(chunk):
            for model in FOCUS:
                for j, metric in enumerate(("rmse", "crps")):
                    points = [(run["value"], row) for run in runs for row in run["summary"]["table"]
                              if (row["scenario"], row["track"], row["model_id"]) == (scenario, track, model) and row.get(metric) is not None]
                    if points:
                        axes[k, j].plot([p[0] for p in points], [p[1][metric] for p in points], "o-", color=COLORS[model], label=LABELS[model])
            for j, metric in enumerate(("RMSE (operational)", "CRPS (qualified native)")):
                axes[k, j].set_title(f"{short_scenario(scenario)}/{track} / {metric}", loc="left")
                axes[k, j].set_xlabel("Initial training rows" if curve["design"]["kind"] == "history" else "Observed sensors")
                axes[k, j].grid(alpha=.4)
                axes[k, j].legend(fontsize=7)
        figures.append(save_figure(fig, output, f"learning_curve_{offset}", "Resolved matched design; only completed conditions are shown. Final forecast period and scored target domain are held fixed. Native CRPS is conditional on qualification and must be read with per-condition counts. Connecting points illustrates the response to the declared intervention, not monotonicity or a proven improvement."))
    return figures


def section(title, paragraphs=(), table=None, figures=(), links=()):
    return {"title": title, "paragraphs": list(paragraphs), "table": table,
            "figures": list(figures), "links": list(links)}


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
        sections.append(section(f"{run['label']}: predictive performance", [
            "Scenarios retain their original response units in these marginal score panels. Do not rank methods by comparing unpaired native CRPS means with different qualification populations. Every paired contrast below uses the common eligible datasets and its saved training-only scale."], figures=score_figures(run, output)))
        calibration = calibration_figure(run, output)
        if calibration:
            sections.append(section(f"{run['label']}: predictive uncertainty", [
                "A good forecast distribution must represent uncertainty as well as its mean. These saved interval summaries show whether narrow predictions miss observations and whether wider intervals buy useful coverage. Qualification and calibration are separate questions."], figures=calibration))
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
        sections.append(section(f"{run['label']}: cost and reliability", figures=[cost_figure(run, output)]))
    curve = data["curve"]
    text = [f"Learning-curve evidence status: {curve['status']}.",
        "A valid history curve holds the final forecast interval and target sites fixed, varies the supplied historical window, and uses a common training-only normalization reference. A valid sensor curve nests observed sensors while preserving the unobserved target set. Fixed specifications and repeated tuning are distinct experiments."]
    if curve.get("unexecuted_values"):
        text.append("Unexecuted planned values: " + ", ".join(map(str, curve["unexecuted_values"])) + ". They are absent from the plots, not treated as failed or zero-valued results.")
    if not curve["runs"]:
        text.append("No completed curve conditions are included. This report cannot establish whether more history or more sensors improve prediction.")
    sections.append(section("More data and expressiveness", text, figures=curve_figures(curve, output),
        links=[("Matched information interventions and populations", "evidence/study_design_v03.md")]))
    extensions = data["extensions"]
    extension_rows = [[r.get("name", r.get("id")), r["status"], r.get("rationale", r.get("reason"))] for r in extensions["entries"]]
    sections.append(section("Extension dispositions", [f"Status: {extensions['status']}. These are supplied research decisions with their stated evidence, not automatic claims of improvement.",
        "Grouped ridge, finite-slab regularized horseshoe, residual dynamics, anisotropy, calibration and larger feature maps should advance only when a diagnosed deficiency, compatible inference route and bounded development comparison justify them. Implemented optional routes may remain unvalidated at the benchmark scale."],
        table=(["Extension", "Disposition", "Reason / evidence"], extension_rows) if extension_rows else None))
    engineering = data["engineering"]
    memory = engineering["memory"]
    text = ["Scientific identity binds design, source, data, numerical runtime and transition policy. Worker count, memory ceiling and execution chunk limits belong to attempt receipts. Safe resume preserves scientific seeds and checkpoint trajectories; incompatible evidence is invalidated rather than reused.",
        "Memory admission combines a conservative workload estimate with host/cgroup headroom before each job. Jobs use one numerical thread each. A resource pause does not silently reduce model dimensions or relax diagnostics. Current benchmark scoring still materializes forecast samples; end-to-end disk-backed scoring is not claimed."]
    if memory:
        a, b = memory["rows"]
        text.append(f"The isolated ownership probe used a {a['shape'][0]} x {a['shape'][1]} x {a['shape'][2]} float64 path array. Public-copy peak RSS was {a['peak_rss_bytes']/1024**2:.1f} MiB; private owned-transfer peak was {b['peak_rss_bytes']/1024**2:.1f} MiB. Incremental peak reduction was {memory['incremental_peak_reduction_bytes']/1024**2:.1f} MiB with identical numerical payload hashes. This measures the allocation/constructor workload only, not model-fit memory or universal forecasting speed.")
    tests = engineering["tests"]
    test_rows = [[r["file"], str(r["tests"]), str(r["failures"] + r["errors"]), str(r["skipped"])] for r in tests]
    text.append("Test totals below are saved JUnit snapshots with their original timestamps. They do not automatically certify changes made after the snapshot. Release/publication status is verified only by explicit supplied release receipts; building this report does not publish a release.")
    if not engineering["release"]:
        text.append("No release-verification receipt is included in this synthesis. Repository publication and asset checks remain separately recorded release gates.")
    sections.append(section("Engineering evidence and reproducibility", text,
        table=(["Saved test receipt", "Tests", "Failures + errors", "Skipped"], test_rows) if test_rows else None))
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
    css = """body{margin:0;background:#f0f5f7;color:#10283f;font:16px/1.65 system-ui,sans-serif}main{max-width:1140px;margin:auto;padding:30px 24px 70px}header{background:#10283f;color:#fff;border-radius:20px;padding:48px 44px}header p{max-width:800px;color:#c8dbe6}h1{font-size:clamp(36px,5vw,62px);line-height:1.05;letter-spacing:-.04em;max-width:850px;margin:18px 0 25px}h2{font-size:28px;line-height:1.2;margin:12px 0 24px;letter-spacing:-.02em}.eyebrow{text-transform:uppercase;letter-spacing:.18em;font-size:11px;font-weight:700;color:#008c82}header .eyebrow{color:#77d6cc}section{background:#fff;border-radius:16px;padding:32px 36px;margin:24px 0;scroll-margin-top:20px}nav{display:flex;flex-wrap:wrap;gap:9px;margin:24px 0}nav a{font-size:12px;padding:7px 11px;background:#e2edef;border-radius:7px;text-decoration:none}a{color:#007c75}figure{margin:28px 0}img,video{width:100%;height:auto}figcaption{font-size:13px;color:#5c7083;line-height:1.55;margin-top:10px}table{border-collapse:collapse;width:100%;font-size:13px;margin:15px 0}td,th{text-align:left;vertical-align:top;padding:10px 12px;border-bottom:1px solid #dce7ed}th{background:#edf5f6;color:#10283f}tr:nth-child(even) td{background:#f8fbfc}.table-wrap{overflow-x:auto}code{overflow-wrap:anywhere;font-size:11px}footer{font-size:12px;color:#5c7083;margin-top:30px}@media(max-width:650px){main{padding:12px}header,section{padding:25px 22px}h2{font-size:24px}}"""
    content = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bayesian ESNs / v0.3 study synthesis</title><style>{css}</style></head><body><main><header><div class="eyebrow">Bayes-STESN / v0.3 / auditable research</div><h1>Forecasting spatial dynamics,<br>with evidence that travels.</h1><p>Model hierarchy, exact inference changes, predictive performance, qualification and computational cost. A synthesis of saved, versioned experiments.</p><p><a style="color:#77d6cc" href="bayes_stesn_v03_report.pdf">Download the PDF report</a></p></header><nav>{navigation}</nav>{"".join(pieces)}<footer>Scientific source fingerprint: <code>{source}</code><br>Saved-artifact renderer. No model fitting or forecasting occurs during report generation.</footer></main></body></html>'
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
             Paragraph("Forecasting spatial dynamics", styles["title"]),
             Paragraph("A reproducible study synthesis", styles["h2"]),
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
            table = Table([[Paragraph(html.escape(str(x)), styles["head" if i == 0 else "cell"]) for x in row]
                           for i, row in enumerate([headers, *rows])], colWidths=[width * f for f in fractions], repeatRows=1, hAlign="LEFT")
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
    if data["coordinate"]:
        write_json(folder / "coordinate_manifest.json", data["coordinate"]["manifest"])
        write_json(folder / "coordinate_summary.json", data["coordinate"]["summary"])
    if data["curve"].get("design"):
        write_json(folder / "curve_design.json", data["curve"]["design"])
        for run in data["curve"]["runs"]:
            write_json(folder / f"curve_{run['value']}_summary.json", run["summary"])
    write_json(folder / "engineering_evidence.json", data["engineering"])
    write_json(folder / "extension_dispositions.json", data["extensions"])
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
        "curve": {k: v for k, v in data["curve"].items() if k not in {"runs", "design"}},
        "extension_status": data["extensions"]["status"], "render_only": True}
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
    command.add_argument("--curve", type=Path, help="Optional resolved curve directory or curve_design.json")
    command.add_argument("--extensions", type=Path, help="Optional JSON extension dispositions")
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
            "coordinate": data["coordinate"] is not None, "curve_status": data["curve"]["status"],
            "extension_status": data["extensions"]["status"], "source_fingerprint": data["source_fingerprint"]}, sort_keys=True))
    else:
        receipt = render(data, inputs, args.output)
        print(json.dumps({"output": str(args.output.resolve()), "identity": receipt["identity"],
                          "files": len(receipt["files"]), "pdf_visual_qa": "pending"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
