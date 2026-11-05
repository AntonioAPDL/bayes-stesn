#!/usr/bin/env python3
"""Animate report-bound KS validation examples without fitting or forecasting.

The immutable example rule is trajectory 1, first target 6000, horizon 100:
old selected configuration versus the new preserved-prior selection, separately
for A/B/C. Only retained NPZ arrays and report/freeze metadata are read. No raw
dataset, posterior checkpoint, or additional forecast receipt is opened.

Run after generating the report with its example bindings:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONPATH=src python tools/animate_ks_improvement.py \
    --results /data/ks_improvement_20261105/run_v2 \
    --report docs/applications/improvement_results \
    --output docs/applications/improvement_results/animations

Existing movies are reused only when their complete render identity and checksum
match. An incompatible output directory is refused instead of overwriting it.
"""
from __future__ import annotations

import argparse
import datetime as dt
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import tempfile
import time

import numpy as np

from bayes_stesn.io import content_hash


SCHEMA = "ks-improvement-animation/1"
TRACKS = {"A": {"label": "Interleaved unseen sites", "inputs": 39, "targets": 13},
          "B": {"label": "All observed sites", "inputs": 64, "targets": 64},
          "C": {"label": "Contiguous unseen sites", "inputs": 40, "targets": 12}}
POLICY = {"trajectory": 1, "first_target": 6000, "partition": "validation", "horizon": 100,
          "saved_cadence": .25, "domain_length": 22., "panel_grid_sites": 64,
          "roles": ["anchor", "main"], "target": "clean_field", "interval": .9,
          "render_width": 1280, "render_height": 720,
          "site_display": "discrete target-site markers and vertical marginal intervals; no spatial interpolation",
          "scale_policy": "one physical amplitude scale shared by both model panels; symlog if range exceeds 50 times the case truth amplitude",
          "case_selection": "fixed by trajectory and origin, never by outcome"}
KEYS = ("mean", "truth", "lower", "upper", "coordinates", "input_coordinates", "first_target", "training_scale")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def verify_file(path, expected):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != expected["bytes"] or file_hash(path) != expected["sha256"]:
        raise ValueError(f"Bound file missing or changed: {path.name}")


def coordinate_geometry(coordinates, count):
    coordinates = np.asarray(coordinates, np.float64)
    if (coordinates.shape != (count, 2) or not np.isfinite(coordinates).all()
            or not np.allclose(np.linalg.norm(coordinates, axis=1), 1., rtol=0, atol=1e-10)):
        raise ValueError("Expected finite unit-circle coordinates with the declared site count")
    x = np.mod(np.arctan2(coordinates[:, 1], coordinates[:, 0]), 2 * np.pi) * 22. / (2 * np.pi)
    x[np.isclose(x, 22., atol=1e-10, rtol=0)] = 0.
    indices = np.rint(x * 64 / 22).astype(int) % 64
    if len(set(indices.tolist())) != count or not np.allclose(x, indices * 22. / 64, atol=1e-8, rtol=0):
        raise ValueError("Coordinates must be unique points of the delivered 64-site observation panel")
    return x, indices


def validate_example(arrays, track):
    geometry = TRACKS[track]
    for key in KEYS:
        if key not in arrays:
            raise ValueError(f"Retained example is missing {key}")
    shape = (100, geometry["targets"])
    for key in ("mean", "truth", "lower", "upper"):
        if np.shape(arrays[key]) != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid finite retained {key} array")
    if np.any(arrays["lower"] > arrays["upper"]):
        raise ValueError("Reversed marginal interval endpoints")
    if np.size(arrays["first_target"]) != 1 or float(np.asarray(arrays["first_target"]).item()) != 6000.:
        raise ValueError("Example is not the predeclared first target 6000")
    if np.size(arrays["training_scale"]) != 1 or not np.isfinite(arrays["training_scale"]).all() or float(arrays["training_scale"]) <= 0:
        raise ValueError("Invalid permitted training scale")
    x, targets = coordinate_geometry(arrays["coordinates"], geometry["targets"])
    input_x, inputs = coordinate_geometry(arrays["input_coordinates"], geometry["inputs"])
    if track == "B":
        if set(targets) != set(range(64)) or set(inputs) != set(range(64)):
            raise ValueError("All-observed panels must contain the complete 64-site grid")
    elif set(targets) & set(inputs):
        raise ValueError("Unseen-site input and validation target coordinates overlap")
    return {"x": x, "input_x": input_x, "target_grid_indices": targets,
            "input_grid_indices": inputs, "order": np.argsort(x)}


def matched_geometry(examples):
    available = [r for r in examples if r["state"] == "available"]
    if len(available) == 2:
        a, b = available[0]["arrays"], available[1]["arrays"]
        for key in ("truth", "coordinates", "input_coordinates", "first_target", "training_scale"):
            if not np.array_equal(a[key], b[key]):
                raise ValueError(f"Old/new retained examples are unmatched in {key}")
    return available


def scale_policy(examples):
    available = matched_geometry(examples)
    if not available:
        raise ValueError("No available example; do not render an empty movie")
    truth_amplitude = max(float(np.abs(available[0]["arrays"]["truth"]).max()), 1e-8)
    amplitude = max(max(float(np.abs(item["arrays"][key]).max()) for key in ("truth", "mean", "lower", "upper"))
                    for item in available)
    error_amplitude = max(float(np.abs(item["arrays"]["mean"] - item["arrays"]["truth"]).max()) for item in available)
    amplitude = max(amplitude, 1e-8)
    error_amplitude = max(error_amplitude, 1e-8)
    return {"value_limit": amplitude * 1.06, "error_limit": error_amplitude * 1.06,
            "linear_threshold": truth_amplitude,
            "value_scale": "symlog" if amplitude > 50 * truth_amplitude else "linear",
            "error_scale": "symlog" if error_amplitude > 50 * truth_amplitude else "linear",
            "all_finite_values_included": True, "clipping": False}


def checked_report(results, report):
    results, report = Path(results).resolve(), Path(report).resolve()
    manifest = read_json(report / "source_manifest.json")
    artifacts = read_json(report / "artifact_manifest.json")
    for value in (manifest, artifacts):
        if value["identity"] != content_hash({k: v for k, v in value.items() if k != "identity"}):
            raise ValueError("Report manifest identity mismatch")
    for name in ("source_manifest.json", "report.json", "selected_configurations.json"):
        verify_file(report / name, artifacts["files"][name])
    if artifacts["source_manifest_identity"] != manifest["identity"]:
        raise ValueError("Artifact/source manifest bindings disagree")
    for name in ("experiment.json", "freeze.json"):
        if name not in manifest["results_receipts"]:
            raise ValueError("Report has no bound experiment/freeze; wait for confirmation selection")
        verify_file(results / name, manifest["results_receipts"][name])
    experiment, freeze = read_json(results / "experiment.json"), read_json(results / "freeze.json")
    summary = read_json(report / "report.json")
    if (freeze["identity"] != content_hash({k: v for k, v in freeze.items() if k != "identity"})
            or freeze["test_access"] is not False or freeze["history"] != 512 or freeze["paths"] != 512
            or freeze["experiment_identity"] != experiment["identity"]
            or experiment["identity"] != content_hash({"config": experiment["config"], "source": experiment["source"], "cache": experiment["cache_identity"]})
            or freeze != read_json(report / "selected_configurations.json")
            or summary["freeze_identity"] != freeze["identity"]
            or artifacts["freeze_identity"] != freeze["identity"]
            or summary["experiment_identity"] != experiment["identity"]
            or manifest["experiment_identity"] != experiment["identity"]
            or 1 not in freeze["confirmation_trajectories"] or 6000 not in freeze["confirmation_origins"]):
        raise ValueError("Retained-example report has an incompatible scientific binding")
    binding = {"experiment_identity": experiment["identity"], "freeze_identity": freeze["identity"],
        "freeze_sha256": file_hash(results / "freeze.json"), "report_status": summary["status"],
        "source_manifest_identity": manifest["identity"], "source_manifest_sha256": file_hash(report / "source_manifest.json"),
        "artifact_manifest_identity": artifacts["identity"], "artifact_manifest_sha256": file_hash(report / "artifact_manifest.json"),
        "report_sha256": file_hash(report / "report.json"), "policy": POLICY}
    return manifest, experiment, freeze, binding


def load_examples(results, manifest, experiment, freeze, track):
    seed = int(experiment["config"]["model_seed"])
    selected = freeze["selected"][track]
    panels = []
    for role, cid in (("anchor", selected["anchor_id"]), ("main", selected["main"]["id"])):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", cid):
            raise ValueError("Unsafe candidate identifier")
        name = f"examples/{track}_{cid}_t01_s{seed}_a6000.npz"
        path = Path(results) / name
        expected = manifest["results_receipts"].get(name)
        panel = {"role": role, "candidate_id": cid, "relative_source": name, "expected": expected}
        if expected is None:
            if path.exists():
                raise ValueError("Existing retained example is not bound by this report snapshot; rebuild the report with example plots")
            panel.update(state="missing", reason="predeclared example absent from the report snapshot and retained files")
        else:
            verify_file(path, expected)
            if expected["bytes"] > 32 * 1024 ** 2:
                raise ValueError("Retained example unexpectedly exceeds the 32MiB input budget")
            with np.load(path, allow_pickle=False) as saved:
                arrays = {key: saved[key].copy() for key in KEYS}
            geometry = validate_example(arrays, track)
            panel.update(state="available", arrays=arrays, geometry=geometry)
        panels.append(panel)
    matched_geometry(panels)
    return panels


def panel_label(panel):
    role = "Old selected configuration" if panel["role"] == "anchor" else "New preserved-prior selection"
    return role + "\n" + panel["candidate_id"]


def source_panels(panels):
    return [{k: v for k, v in panel.items() if k not in ("arrays", "geometry")} for panel in panels]


def validate_probe(probe, *, fps, width=1280, height=720):
    streams = probe.get("streams", [])
    if len(streams) != 1:
        raise ValueError("Expected one selected video stream")
    video = streams[0]
    if (video.get("codec_name") != "h264" or video.get("pix_fmt") != "yuv420p"
            or int(video.get("width", 0)) != width or int(video.get("height", 0)) != height
            or int(video.get("nb_read_frames", -1)) != 100
            or Fraction(video.get("avg_frame_rate", "0/1")) != Fraction(fps, 1)
            or abs(float(video.get("duration", -1.)) - 100 / fps) > .001):
        raise ValueError("Encoded movie differs from the declared H.264/100-frame/size/cadence contract")
    return video


def probe_movie(path, fps):
    completed = subprocess.run([shutil.which("ffprobe") or "ffprobe", "-v", "error", "-threads", "1", "-count_frames",
        "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_read_frames,pix_fmt,duration",
        "-of", "json", str(path)], capture_output=True, text=True, check=True, timeout=60)
    return validate_probe(json.loads(completed.stdout), fps=fps)


def sparse_symlog_ticks(axis, minimum_fraction=.20):
    """Space signed tick labels in display coordinates without changing limits."""
    limits = axis.get_ylim()
    transform = axis.yaxis.get_transform()
    span = abs(float(np.diff(transform.transform(np.asarray(limits)))[0]))
    candidates = [float(t) for t in axis.get_yticks() if limits[0] <= t <= limits[1]]
    chosen = [0.] if limits[0] <= 0 <= limits[1] else []
    for tick in sorted(candidates, key=abs, reverse=True):
        if all(abs(float(transform.transform(tick) - transform.transform(other))) >= minimum_fraction * span
               for other in chosen):
            chosen.append(tick)
    axis.set_yticks(sorted(chosen))
    axis.set_ylim(limits)


def render_movie(path, track, panels, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    policy = scale_policy(panels)
    available = matched_geometry(panels)
    geometry = available[0]["geometry"]
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    grid = fig.add_gridspec(3, 2, height_ratios=(3., 1.25, .34),
                          left=.07, right=.985, top=.77, bottom=.155, hspace=.35, wspace=.19)
    artists = []
    for column, panel in enumerate(panels):
        axis, error_axis = fig.add_subplot(grid[0, column]), fig.add_subplot(grid[1, column])
        for item, limit, scaling in ((axis, policy["value_limit"], policy["value_scale"]),
                                     (error_axis, policy["error_limit"], policy["error_scale"])):
            item.set_xlim(-.25, 22.25); item.set_ylim(-limit, limit)
            item.set_xticks([0, 5.5, 11., 16.5, 22.])
            if scaling == "symlog":
                item.set_yscale("symlog", linthresh=policy["linear_threshold"])
                sparse_symlog_ticks(item)
            item.grid(alpha=.17)
            item.spines[["top", "right"]].set_visible(False)
            item.tick_params(labelsize=8)
        axis.set_title(panel_label(panel), fontsize=9, pad=8)
        axis.tick_params(labelbottom=False)
        axis.set_ylabel("Clean field u (physical units)" if column == 0 else "", fontsize=9)
        error_axis.set_ylabel("Mean − truth" if column == 0 else "", fontsize=9)
        error_axis.set_xlabel("Physical periodic coordinate x", fontsize=8)
        error_axis.axhline(0., color="#555555", linewidth=.7)
        if panel["state"] != "available":
            for item in (axis, error_axis):
                item.text(.5, .5, "Retained example unavailable\nNo replacement case was selected", transform=item.transAxes,
                          ha="center", va="center", color="#666666", fontsize=9)
            continue
        arrays, geo = panel["arrays"], panel["geometry"]
        order, x = geo["order"], geo["x"][geo["order"]]
        truth, = axis.plot(x, arrays["truth"][0, order], linestyle="none", marker="o", color="black", markersize=3.4, zorder=3)
        mean, = axis.plot(x, arrays["mean"][0, order], linestyle="none", marker="x", color="#2166ac", markersize=5., markeredgewidth=1.2, zorder=4)
        segments = np.stack((np.column_stack((x, arrays["lower"][0, order])), np.column_stack((x, arrays["upper"][0, order]))), axis=1)
        intervals = LineCollection(segments, colors="#67a9cf", linewidths=2.2, alpha=.75, zorder=2)
        axis.add_collection(intervals)
        error, = error_axis.plot(x, (arrays["mean"] - arrays["truth"])[0, order], linestyle="none", marker="o", markersize=3.4, color="#b2182b")
        artists.append((arrays, order, x, truth, mean, intervals, error))
    rug = fig.add_subplot(grid[2, :])
    rug.plot(geometry["input_x"], np.zeros(len(geometry["input_x"])), linestyle="none", marker="|", color="#555555", markersize=8)
    rug.plot(geometry["x"], np.ones(len(geometry["x"])), linestyle="none", marker="o", color="#d95f02", markersize=3.)
    rug.set_xlim(-.25, 22.25); rug.set_ylim(-.6, 1.6)
    rug.set_xticks([]); rug.set_yticks([0, 1], ["Inputs", "Targets"], fontsize=7)
    rug.spines[["top", "right", "bottom"]].set_visible(False)
    handles = [Line2D([], [], color="black", marker="o", linestyle="none", markersize=4, label="Clean truth at target sites"),
               Line2D([], [], color="#2166ac", marker="x", linestyle="none", markersize=5, label="Posterior predictive mean"),
               Line2D([], [], color="#67a9cf", linewidth=3, label="90% marginal predictive interval")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .855), ncol=3, frameon=False, fontsize=8)
    fig.text(.5, .966, f"Track {track} · {TRACKS[track]['label']} · predeclared trajectory 1 / first target 6000",
             ha="center", va="top", fontsize=12)
    dynamic = fig.text(.5, .917, "", ha="center", va="top", fontsize=10)
    scope = (f"{TRACKS[track]['targets']} official validation target sites; {TRACKS[track]['inputs']} permitted input sites. "
             "Discrete markers only; no full-field interpolation or error claim.")
    fig.text(.5, .090, scope, ha="center", fontsize=8)
    if policy["value_scale"] == "symlog" or policy["error_scale"] == "symlog":
        scale_note = (f"Shared y scales: field={policy['value_scale']}, error={policy['error_scale']}; "
                      f"symlog is linear within ±{policy['linear_threshold']:.3g}. Every finite value is retained.")
    else:
        scale_note = "Both model panels share fixed physical y scales across all 100 frames. Every finite value is retained."
    fig.text(.5, .060, scale_note, ha="center", fontsize=8)
    fig.text(.5, .030, "Validation-only exploratory follow-up; this single fixed example is not an aggregate ranking.", ha="center", fontsize=8, color="#555555")
    fd, temporary = tempfile.mkstemp(prefix="." + Path(path).stem + ".", suffix=".tmp.mp4", dir=Path(path).parent)
    os.close(fd)
    try:
        writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=1400,
            metadata={"title": f"KS validation {track}: fixed trajectory1 first6000, old versus preserved-prior selection",
                      "comment": "Report-bound retained arrays; official target-site markers only; no new forecast computation"},
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-preset", "veryfast", "-threads", "1", "-filter_threads", "1"])
        with writer.saving(fig, temporary, dpi=100):
            for frame in range(100):
                dynamic.set_text(f"Autonomous lead {frame + 1:3d}/100 · physical lead τ = {.25 * (frame + 1):.2f} · target index {6000 + frame}")
                for arrays, order, x, truth, mean, intervals, error in artists:
                    truth.set_ydata(arrays["truth"][frame, order]); mean.set_ydata(arrays["mean"][frame, order])
                    segments = np.stack((np.column_stack((x, arrays["lower"][frame, order])), np.column_stack((x, arrays["upper"][frame, order]))), axis=1)
                    intervals.set_segments(segments)
                    error.set_ydata((arrays["mean"][frame] - arrays["truth"][frame])[order])
                writer.grab_frame(facecolor="white")
        probe = probe_movie(temporary, fps)
        if Path(temporary).stat().st_size <= 0:
            raise ValueError("Encoder produced an empty movie")
        os.replace(temporary, path)
        return {"scale_policy": policy, "ffprobe": probe, "geometry": {
            "target_grid_indices": geometry["target_grid_indices"].tolist(),
            "input_grid_indices": geometry["input_grid_indices"].tolist(),
            "validation_target_count": TRACKS[track]["targets"], "permitted_input_count": TRACKS[track]["inputs"],
            "coordinate_unit": "physical x in [0,22)", "no_repeated_endpoint": True},
            "visual_review": "not performed by this program; inspect the encoded first, middle and final leads separately"}
    finally:
        Path(temporary).unlink(missing_ok=True)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tracks", nargs="+", choices=tuple(TRACKS), default=list(TRACKS))
    parser.add_argument("--fps", choices=(8, 10), type=int, default=10)
    args = parser.parse_args(argv)
    if len(set(args.tracks)) != len(args.tracks):
        raise ValueError("Duplicate requested track")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"Set {name}=1 before launching")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("Existing ffmpeg and ffprobe are required; no codec installation is attempted")
    output = args.output.resolve()
    results = args.results.resolve()
    if output == results or results in output.parents:
        raise ValueError("Animation output must be separate from immutable numerical receipts")
    manifest, experiment, freeze, binding = checked_report(results, args.report)
    plans = {track: load_examples(results, manifest, experiment, freeze, track) for track in args.tracks}
    render_binding = {**binding, "fps": args.fps, "tracks": args.tracks, "builder_sha256": file_hash(__file__),
                      "source_panels": {t: source_panels(p) for t, p in plans.items()}}
    render_identity = content_hash(render_binding)
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "animation_receipt.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["render_identity"] != render_identity or receipt["binding"] != render_binding:
            raise ValueError("Incompatible animation output identity; use a new output directory")
    else:
        receipt = {"schema": SCHEMA, "render_identity": render_identity, "binding": render_binding,
                   "created_utc": utc(), "movies": {}, "new_fits": 0, "new_forecasts": 0,
                   "raw_dataset_reads": 0, "additional_forecast_receipt_reads": 0, "frame_dump_files": 0}
        atomic_json(receipt_path, receipt)
    for track, panels in plans.items():
        old = receipt["movies"].get(track)
        if old and old["state"] in ("complete", "complete_with_missing_panel"):
            verify_file(output / old["file"], old)
            probe_movie(output / old["file"], args.fps)
            continue
        if not any(panel["state"] == "available" for panel in panels):
            receipt["movies"][track] = {"state": "unavailable", "reason": "both predeclared examples are missing; no empty movie rendered",
                                         "panels": source_panels(panels)}
            atomic_json(receipt_path, receipt)
            continue
        filename = f"ks_improvement_{track}_t01_a6000.mp4"
        path = output / filename
        if path.exists():
            raise ValueError("Movie exists without its verified completed receipt; preserve it and use another output root")
        started = time.perf_counter()
        details = render_movie(path, track, panels, args.fps)
        receipt["movies"][track] = {"state": "complete" if all(p["state"] == "available" for p in panels) else "complete_with_missing_panel",
            "file": filename, "bytes": path.stat().st_size, "sha256": file_hash(path), "frames": 100,
            "fps": args.fps, "duration_seconds": 100 / args.fps, "panels": source_panels(panels),
            "render_seconds": time.perf_counter() - started, **details}
        receipt["updated_utc"] = utc()
        atomic_json(receipt_path, receipt)
    receipt["all_requested_panels_available"] = all(p["state"] == "available" for ps in plans.values() for p in ps)
    receipt["state"] = "complete" if receipt["all_requested_panels_available"] else "partial_examples"
    receipt["updated_utc"] = utc()
    receipt["self_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    receipt["child_process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
    receipt["memory_scope"] = "separate process high-water marks, not combined simultaneous memory"
    atomic_json(receipt_path, receipt)
    print(json.dumps({"state": receipt["state"], "render_identity": render_identity,
                      "movies": {t: {k: v for k, v in m.items() if k in ("state", "file", "bytes", "sha256", "frames", "fps", "duration_seconds")}
                                 for t, m in receipt["movies"].items()}}))


if __name__ == "__main__":
    main()
