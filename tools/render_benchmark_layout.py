#!/usr/bin/env python3
"""Render saved benchmark evidence with two explicit presentation corrections.

Usage::

    python tools/render_benchmark_layout.py RUN_DIR --output REPORT_DIR
    python tools/embed_report_fonts.py REPORT_DIR

The package renderer remains unchanged. This wrapper hides redundant y-axis tick
labels on forecast panels two and three, and gives CRPS comparison categories a
half-row margin. Numeric plot artists are hashed before and after each correction.
No fitting, scoring, model selection, forecast generation, or array modification
occurs. Existing complete reports must pass their inventory hashes; unaffected
assets, including custom comparison subdirectories, remain byte-identical.

An optional --evidence path outside REPORT_DIR saves the full render audit. The
report manifest also records the recipe, its hash and the display corrections.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _standalone_comparison(output, run_identity):
    """Import only the companion renderer's complete, self-inventoried output."""
    comparison = output / "comparison"
    if output.is_symlink() or set(output.iterdir()) != {comparison} or comparison.is_symlink() or not comparison.is_dir():
        raise ValueError("Without a report inventory, output may contain only a standalone comparison/ directory")
    metadata_path = comparison / "manifest.json"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise ValueError("Standalone comparison needs its regular manifest.json")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema") != "saved-model-comparison-v1" or metadata.get("run_identity") != run_identity:
        raise ValueError("Standalone comparison schema or run identity does not match")
    hashes = metadata.get("output_sha256")
    allowed = {"comparison.png", "comparison.svg", "comparison.mp4", "comparison.gif"}
    if not isinstance(hashes, dict) or not {"comparison.png", "comparison.svg"}.issubset(hashes) or not set(hashes).issubset(allowed):
        raise ValueError("Standalone comparison has an unsupported output inventory")
    expected = set(hashes) | {"manifest.json"}
    if {path.name for path in comparison.iterdir()} != expected:
        raise ValueError("Standalone comparison has missing or unlisted files")
    verified = {}
    for name in sorted(expected):
        path = comparison / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(comparison.resolve()):
            raise ValueError("Standalone comparison contains a symlink, non-file or escaping path")
        digest = _sha256(path)
        if name != "manifest.json" and hashes[name] != digest:
            raise ValueError(f"Standalone comparison output hash mismatch: {name}")
        verified[str(Path("comparison") / name)] = digest
    return verified


def _inventory(output, run_identity):
    manifest_path = output / "report_manifest.json"
    if not manifest_path.exists():
        if output.exists() and any(output.iterdir()):
            return _standalone_comparison(output, run_identity), None
        return {}, None
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("render_only"):
        raise ValueError("Existing output is not a saved-artifact report")
    verified = {}
    for record in manifest["files"]:
        name = Path(record["path"])
        path = output / name
        if name.is_absolute() or ".." in name.parts or path.is_symlink():
            raise ValueError("Unsafe report inventory path")
        if not path.resolve().is_relative_to(output.resolve()) or str(name) in verified:
            raise ValueError("Report inventory has a duplicate or escaping path")
        if not path.is_file() or path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise ValueError(f"Existing report asset does not match its inventory: {name}")
        verified[str(name)] = record["sha256"]
    actual = {str(path.relative_to(output)) for path in output.rglob("*")
              if path.is_file() and path != manifest_path}
    if actual != set(verified):
        raise ValueError("Existing report contains unlisted or missing files")
    return verified, manifest


def _numeric_artists_sha256(fig):
    """Hash plotted observations, means, error curves and interval coordinates."""
    digest = hashlib.sha256()
    def add(value):
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    for axis in fig.axes:
        for artist in axis.images:
            add(artist.get_array()); add(artist.get_extent()); add(artist.get_clim())
        for line in axis.lines:
            add(line.get_xdata()); add(line.get_ydata())
        for collection in axis.collections:
            add(collection.get_offsets())
            if hasattr(collection, "get_segments"):
                for segment in collection.get_segments():
                    add(segment)
    return digest.hexdigest()


def _apply_layout(fig, kind):
    before = _numeric_artists_sha256(fig)
    if kind == "forecast":
        if len(fig.axes) < 3:
            raise ValueError("Forecast figure does not have the expected field panels")
        for axis in fig.axes[1:3]:
            axis.tick_params(axis="y", labelleft=False)
        change = "Only the first field panel shows y-coordinate tick labels; adjacent colorbar labels stay unobstructed"
    elif kind == "crps":
        for axis in fig.axes:
            count = len(axis.get_yticks())
            if not count:
                raise ValueError("CRPS figure has no categorical model rows")
            axis.set_ylim(count - .5, -.5)
        change = "Half-row vertical padding keeps the first/last model annotations inside the comparison axes"
    else:
        raise ValueError(f"Unknown presentation correction: {kind}")
    after = _numeric_artists_sha256(fig)
    if after != before:
        raise RuntimeError("A presentation correction changed numerical plot artists")
    return {"kind": kind, "change": change, "numeric_artists_sha256": before,
            "numeric_artists_unchanged": True}


def render_layout(run, output, *, evidence=None):
    from matplotlib.animation import Animation
    from bayes_stesn.benchmark import report
    from bayes_stesn.benchmark.runner import source_fingerprint, write_json

    run, output = Path(run), Path(output)
    if evidence is not None and Path(evidence).resolve().is_relative_to(output.resolve()):
        raise ValueError("The optional evidence file must be outside the inventoried report directory")
    summary = json.loads((run / "summary.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    if summary.get("source_stable") is not True or not summary.get("jobs"):
        raise ValueError("A completed source-stable benchmark run is required")
    if summary.get("manifest_identity") != manifest["identity"]:
        raise ValueError("Run summary and manifest identities disagree")
    current_source = source_fingerprint()
    if current_source != manifest["source_fingerprint"]:
        raise ValueError("Render this evidence with its matching scientific source revision")
    baseline, previous = _inventory(output, manifest["identity"])
    if previous is not None and previous["run_identity"] != manifest["identity"]:
        raise ValueError("Existing report belongs to a different benchmark run")
    inputs = {run / "summary.json", run / "manifest.json"}
    scenarios = list(dict.fromkeys(row["scenario"] for row in summary["table"]))
    for scenario in scenarios:
        base = run / scenario / "rep_000"
        inputs.add(base / "simulation.npz")
        inputs.update(base.glob("*/*/receipt.json"))
        inputs.update(base.glob("*/bayes_esn_ridge/forecast.npz"))
    input_hashes = {str(path): _sha256(path) for path in sorted(inputs)}
    started = datetime.now(timezone.utc).isoformat()
    recipe_hash = _sha256(__file__)
    prior_manifest_hash = None if previous is None else _sha256(output / "report_manifest.json")
    changed, reused, corrections = [], [], []
    original_save, original_movie_save = report._save, Animation.save

    def save_figure(fig, path):
        path = Path(path)
        relative = path.relative_to(output)
        pair = (relative, relative.with_suffix(".svg"))
        kind = "forecast" if relative.parent.name == "media" and relative.stem.endswith("_forecast") else (
            "crps" if relative.parent.name == "figures" and "_crps_" in relative.stem else None)
        if kind is not None:
            corrections.append({"path": str(relative), **_apply_layout(fig, kind)})
        if kind is None and all(str(item) in baseline for item in pair):
            reused.extend(str(item) for item in pair)
            return None
        changed.extend(str(item) for item in pair)
        return original_save(fig, path)

    def save_movie(self, filename, *args, **kwargs):
        relative = Path(filename).relative_to(output)
        if not relative.stem.endswith("_forecast") and str(relative) in baseline:
            reused.append(str(relative))
            return None
        changed.append(str(relative))
        print(json.dumps({"event": "render_animation", "path": str(relative)}), flush=True)
        return original_movie_save(self, filename, *args, **kwargs)

    report._save, Animation.save = save_figure, save_movie
    try:
        with warnings.catch_warnings():
            # Fully reused movies require no Matplotlib draw call; their existing
            # byte-verified exports are intentionally retained.
            warnings.filterwarnings("ignore", message="Animation was deleted without rendering anything.*")
            receipt = report.create_report(run, output, animations=True)
    finally:
        report._save, Animation.save = original_save, original_movie_save
    if source_fingerprint() != current_source:
        raise RuntimeError("Scientific source changed during report rendering")
    for path, digest in input_hashes.items():
        if _sha256(path) != digest:
            raise RuntimeError(f"Scientific input changed during report rendering: {path}")
    mutable = set(changed) | {"benchmark_report.pdf", "report.html"}
    preserved = {name: digest for name, digest in baseline.items() if name not in mutable}
    for name, digest in preserved.items():
        if _sha256(output / name) != digest:
            raise RuntimeError(f"An unaffected report asset changed: {name}")
    command = [sys.executable, str(Path(__file__)), str(run), "--output", str(output)]
    if evidence is not None:
        command.extend(("--evidence", str(evidence)))
    audit = {
        "schema": "benchmark-presentation-correction-v1", "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "run_identity": manifest["identity"], "source_fingerprint": current_source,
        "recipe": "tools/render_benchmark_layout.py", "recipe_sha256": recipe_hash,
        "command": command, "prior_report_manifest_sha256": prior_manifest_hash,
        "standalone_comparison_imported": bool(baseline and previous is None),
        "scientific_input_sha256": input_hashes, "scientific_inputs_unchanged": True,
        "corrections": corrections, "regenerated_assets": sorted(changed),
        "unaffected_asset_sha256": preserved, "unaffected_assets_byte_identical": True,
        "custom_comparison_assets_unchanged": all(_sha256(output / name) == digest
            for name, digest in preserved.items() if name.startswith("comparison/")),
        "font_embedding_next": "python tools/embed_report_fonts.py " + str(output),
    }
    receipt["presentation"] = {key: audit[key] for key in (
        "schema", "recipe", "recipe_sha256", "command", "source_fingerprint",
        "corrections", "scientific_inputs_unchanged", "unaffected_assets_byte_identical",
        "custom_comparison_assets_unchanged")}
    write_json(output / "report_manifest.json", receipt)
    audit["report_manifest_sha256_before_font_embedding"] = _sha256(output / "report_manifest.json")
    if evidence is not None:
        write_json(Path(evidence), audit)
    return {"report": str(output), "run_identity": manifest["identity"],
            "corrected_figures": len(corrections), "regenerated_assets": len(changed),
            "preserved_assets": len(preserved), "numerical_inputs_unchanged": True,
            "evidence": None if evidence is None else str(evidence),
            "next_command": audit["font_embedding_next"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evidence", type=Path, help="Optional audit JSON outside the report directory")
    args = parser.parse_args(argv)
    print(json.dumps(render_layout(args.run, args.output, evidence=args.evidence), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
