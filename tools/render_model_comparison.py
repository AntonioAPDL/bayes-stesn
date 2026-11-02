"""Render synchronized native forecasts from a completed benchmark replication.

This script only reads saved Simulation, Forecast and receipt artifacts. It does
not fit a model, select a winning model, or replace an unqualified native forecast
with an operational fallback. Columns are fixed: truth, Bayesian ESN, separable
GP, and the custom probabilistic basis GRU. The still uses the middle lead by
position; the animation includes every lead without temporal subsampling.

Run from an installed checkout, for example::

    python tools/render_model_comparison.py results/benchmark_v3/spatial_detail_demo \
        --scenario g4_existing_fronts --output report/spatial_detail_comparison

Use ``--no-animation`` for a PNG/SVG verification pass. Replication 0 and track B
are the defaults. Each explicitly chosen replication remains an illustration,
not evidence of an aggregate ranking. Incomplete model results fail explicitly.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from bayes_stesn.benchmark.contracts import Forecast
from bayes_stesn.benchmark.generators import Simulation


MODELS = ("bayes_esn_ridge", "separable_stgp", "basis_gru")
LABELS = ("Bayesian ESN", "Separable GP", "Probabilistic basis GRU")
COLORS = ("#008C82", "#2563EB", "#B65B26")
SCENARIOS = {
    "g1_gaussian_ar": "G1  /  Gaussian spatial autoregression",
    "g2_spectral_transport": "G2  /  Spectral transport",
    "g3_excitable_field": "G3  /  Nonlinear excitable field",
    "g4_existing_fronts": "G4  /  Moving nonlinear fronts",
    "g5_noise_stress": "G5  /  Process and observation stress",
    "g6_velocity_shift": "G6  /  Hidden velocity shift",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _load_inputs(run: Path, scenario: str, replication: int, track: str):
    """Validate result identity, times, scored domain and published native RMSE."""
    manifest_path = run / "manifest.json"
    manifest = _read_json(manifest_path)
    protocol = manifest["protocol"]
    if scenario not in protocol["scenarios"] or track not in protocol["tracks"]:
        raise ValueError("The requested scenario/track is absent from the saved protocol")
    if replication >= protocol["replications"]:
        raise ValueError("Replication is outside the saved protocol")
    simulation_path = run / scenario / f"rep_{replication:03d}" / "simulation.npz"
    sim = Simulation.load(simulation_path)
    if sim.metadata.get("scenario") != scenario:
        raise ValueError("Saved simulation has a different scenario identity")
    if sim.coordinates.shape[1] != 2:
        raise ValueError("Spatial comparison requires a two-dimensional evaluation grid")
    inputs = [manifest_path, simulation_path]
    forecasts, receipts, checks = [], [], {}
    origin = None
    reference = None
    expected_origin = int(protocol["initial_fit"] + protocol["validation_blocks"] * protocol["horizon"])
    history_start = int(protocol.get("training_start", 0))
    if track == "B":
        target_indices = sim.test_indices
    else:
        target_indices = sim.observed_indices[:protocol.get("score_observed_count")]
    expected_ids = tuple(sim.site_ids[j] for j in target_indices)
    for model in MODELS:
        directory = simulation_path.parent / track / model
        receipt_path, forecast_path = directory / "receipt.json", directory / "forecast.npz"
        receipt = _read_json(receipt_path)
        if not receipt.get("completed"):
            raise ValueError(f"{model}: native fit/forecast is incomplete; no substitute is rendered")
        if any(receipt.get(key) != value for key, value in (
            ("model_id", model), ("scenario", scenario),
            ("replication", replication), ("track", track),
        )):
            raise ValueError(f"{model}: receipt identity does not match its directory")
        if receipt.get("manifest_identity") != manifest.get("identity"):
            raise ValueError(f"{model}: receipt belongs to a different run manifest")
        qualified = receipt.get("qualified")
        if not isinstance(qualified, bool):
            raise ValueError(f"{model}: qualification must be explicitly true or false")
        forecast = Forecast.load(forecast_path)
        this_origin = _integer(receipt.get("origin_index"), f"{model} origin", minimum=1)
        if this_origin != expected_origin:
            raise ValueError(f"{model}: receipt origin disagrees with the saved protocol")
        metadata_origin = _integer(forecast.metadata.get("origin_index"), f"{model} metadata origin", minimum=1)
        if metadata_origin + history_start != this_origin:
            raise ValueError(f"{model}: forecast and receipt origins disagree")
        if len(forecast.mean) != protocol["horizon"]:
            raise ValueError(f"{model}: forecast horizon disagrees with the saved protocol")
        if tuple(receipt.get("target_ids", ())) != forecast.site_ids or forecast.site_ids != expected_ids:
            raise ValueError(f"{model}: forecast site IDs do not match the declared scored domain")
        if not np.array_equal(forecast.coordinates, sim.coordinates[target_indices]):
            raise ValueError(f"{model}: forecast geometry disagrees with the saved simulation")
        if receipt.get("forecast_metadata") != forecast.metadata:
            raise ValueError(f"{model}: receipt and forecast metadata disagree")
        if reference is not None and (
            this_origin != origin or forecast.mean.shape != reference.mean.shape
            or forecast.site_ids != reference.site_ids
            or not np.array_equal(forecast.coordinates, reference.coordinates)
        ):
            raise ValueError("Models must share the identical origin, horizon, site order and geometry")
        if this_origin + len(forecast.mean) > len(sim.observation):
            raise ValueError(f"{model}: actual future observations are missing")
        truth = sim.observation[this_origin:this_origin + len(forecast.mean), target_indices]
        rmse = np.sqrt(np.mean((forecast.mean - truth) ** 2, axis=1))
        scores = receipt.get("native_scores", {})
        check = {"receipt_native_rmse_present": "rmse" in scores}
        if "rmse" in scores:
            direct = float(np.sqrt(np.mean((forecast.mean - truth) ** 2)))
            if not np.isclose(direct, scores["rmse"], rtol=1e-10, atol=1e-12):
                raise ValueError(f"{model}: saved forecast disagrees with its native RMSE")
            check["native_rmse_absolute_difference"] = abs(direct - scores["rmse"])
        if "rmse_by_lead" in scores:
            published = np.asarray(scores["rmse_by_lead"], dtype=float)
            if published.shape != rmse.shape or not np.allclose(published, rmse, rtol=1e-10, atol=1e-12):
                raise ValueError(f"{model}: saved forecast disagrees with native RMSE by lead")
            check["native_lead_rmse_max_absolute_difference"] = float(np.max(np.abs(published - rmse)))
        checks[model] = check
        origin, reference = this_origin, forecast
        forecasts.append(forecast)
        receipts.append(receipt)
        inputs.extend((receipt_path, forecast_path))
    if not np.isfinite(sim.times).all() or not np.all(np.diff(sim.times) > 0):
        raise ValueError("Saved simulation times must be finite and strictly increasing")
    return sim, forecasts, receipts, origin, target_indices, manifest, inputs, checks


def _grid(sim):
    """Map actual coordinates to cells; do not assume flattened storage order."""
    x, y = np.unique(sim.coordinates[:, 0]), np.unique(sim.coordinates[:, 1])
    if len(x) < 2 or len(y) < 2 or len(x) * len(y) != len(sim.coordinates):
        raise ValueError("Renderer requires a complete Cartesian evaluation grid")
    xi = np.searchsorted(x, sim.coordinates[:, 0])
    yi = np.searchsorted(y, sim.coordinates[:, 1])
    if len(set(zip(yi.tolist(), xi.tolist()))) != len(sim.coordinates):
        raise ValueError("Evaluation grid has duplicate coordinates")
    def edges(values):
        middle = (values[:-1] + values[1:]) / 2
        return np.r_[values[0] - (middle[0] - values[0]), middle,
                     values[-1] + (values[-1] - middle[-1])]
    return x, y, edges(x), edges(y), xi, yi


def render_comparison(run, output, *, scenario, replication=0, track="B", animate=True, fps=8):
    """Export a fixed 2x4 comparison, all-lead movies, and an integrity manifest."""
    replication = _integer(replication, "replication")
    fps = _integer(fps, "fps", minimum=1)
    if track not in ("A", "B"):
        raise ValueError("track must be A or B")
    run, output = Path(run).resolve(), Path(output).resolve()
    sim, forecasts, receipts, origin, targets, run_manifest, inputs, checks = _load_inputs(
        run, scenario, replication, track)
    input_hashes = {str(path.relative_to(run)): _sha256(path) for path in inputs}
    horizon = len(forecasts[0].mean)
    truth = sim.observation[origin:origin + horizon, targets]
    means = np.stack([forecast.mean for forecast in forecasts])
    errors = means - truth[None, :, :]
    rmse = np.sqrt(np.mean(errors ** 2, axis=2))
    lo = float(min(truth.min(), means.min()))
    hi = float(max(truth.max(), means.max()))
    if lo == hi:
        width = max(1., abs(lo)) * .05
        lo, hi = lo - width, hi + width
    error_limit = max(float(np.max(np.abs(errors))), 1e-12)
    x, y, xe, ye, xi, yi = _grid(sim)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": "#142A3A", "axes.labelcolor": "#536878",
        "xtick.color": "#647789", "ytick.color": "#647789",
        "axes.edgecolor": "#D2DCE3", "axes.linewidth": .7,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "none",
        "svg.hashsalt": "bayes-stesn-model-comparison-v1",
    })
    figure = plt.figure(figsize=(20, 11.2), dpi=110)
    grid = figure.add_gridspec(2, 4, left=.055, right=.922, bottom=.16, top=.795,
                              wspace=.23, hspace=.34)
    axes = np.array([[figure.add_subplot(grid[row, col]) for col in range(4)] for row in range(2)])
    value_cmap = matplotlib.colormaps["viridis"].copy()
    error_cmap = matplotlib.colormaps["RdBu_r"].copy()
    gray = "#E8EDF1"
    value_cmap.set_bad(gray)
    error_cmap.set_bad(gray)

    def field(values):
        full = np.full((len(y), len(x)), np.nan)
        full[yi[targets], xi[targets]] = values
        return np.ma.masked_invalid(full)

    def spatial_axis(ax):
        ax.set_aspect("equal")
        ax.set_xlim(xe[0], xe[-1]); ax.set_ylim(ye[0], ye[-1])
        ax.set_xticks(np.linspace(x[0], x[-1], 3))
        ax.set_yticks(np.linspace(y[0], y[-1], 3))
        ax.tick_params(length=3, labelsize=9)
        ax.set_xlabel("Spatial coordinate x", fontsize=9, labelpad=6)
        ax.set_ylabel("Spatial coordinate y", fontsize=9, labelpad=5)

    artists = []
    top_values = (truth, *means)
    for column in range(4):
        ax = axes[0, column]
        spatial_axis(ax)
        mesh = ax.pcolormesh(xe, ye, field(top_values[column][0]), cmap=value_cmap,
                             vmin=lo, vmax=hi, shading="flat", linewidth=0)
        artists.append(mesh)
        title = "Actual observations" if column == 0 else LABELS[column - 1]
        ax.set_title(title, loc="left", pad=34, fontsize=14, fontweight="bold")
        if column == 0:
            label, color, background = "SAVED TEST OBSERVATIONS", "#536878", "#F0F3F6"
        elif receipts[column - 1]["qualified"]:
            label, color, background = "NATIVE FIT · QUALIFIED", "#17664F", "#EAF5EF"
        else:
            label, color, background = "UNQUALIFIED · DIAGNOSTIC ONLY", "#9F3A22", "#FCEEE8"
        ax.text(0, 1.035, label, transform=ax.transAxes, fontsize=8.1,
                color=color, ha="left", va="bottom",
                bbox={"boxstyle": "round,pad=.35", "facecolor": background, "edgecolor": "none"})
    error_artists, error_labels = [], []
    for column in range(1, 4):
        ax = axes[1, column]
        spatial_axis(ax)
        mesh = ax.pcolormesh(xe, ye, field(errors[column - 1, 0]), cmap=error_cmap,
                             vmin=-error_limit, vmax=error_limit, shading="flat", linewidth=0)
        error_artists.append(mesh)
        ax.set_title("Forecast − actual", loc="left", pad=30, fontsize=13, fontweight="bold")
        error_labels.append(ax.text(0, 1.025, "", transform=ax.transAxes,
                                    fontsize=9, color=COLORS[column - 1], va="bottom"))

    trace = axes[1, 0]
    leads = np.arange(1, horizon + 1)
    dots = []
    for index, (label, color) in enumerate(zip(LABELS, COLORS)):
        style = "-" if receipts[index]["qualified"] else "--"
        trace.plot(leads, rmse[index], color=color, lw=2.2, ls=style, label=label)
        dot, = trace.plot([1], [rmse[index, 0]], "o", color=color, ms=5, zorder=6)
        dots.append(dot)
    cursor = trace.axvline(1, lw=1.2, color="#7F8F9A", ls=":", zorder=4)
    trace.set_title("Error through the horizon", loc="left", pad=30, fontsize=13, fontweight="bold")
    trace.text(0, 1.025, "Per-lead RMSE on the same scored cells", transform=trace.transAxes,
               fontsize=9, color="#647789", va="bottom")
    trace.set_xlabel("Forecast lead", fontsize=10)
    trace.set_ylabel("RMSE  /  original response units", fontsize=10)
    trace.set_xlim(1, max(2, horizon))
    trace.set_ylim(0, max(float(rmse.max()) * 1.12, 1e-8))
    trace.grid(axis="both", color="#E4EAF0", linewidth=.7)
    trace.set_axisbelow(True)
    trace.legend(frameon=False, loc="upper left", fontsize=8.6,
                 bbox_to_anchor=(0, -.21), borderaxespad=0, handlelength=2.4)

    top_position = axes[0, -1].get_position()
    bottom_position = axes[1, -1].get_position()
    value_bar = figure.add_axes([.943, top_position.y0, .009, top_position.height])
    error_bar = figure.add_axes([.943, bottom_position.y0, .009, bottom_position.height])
    figure.colorbar(artists[0], cax=value_bar, label="Observation / forecast mean")
    figure.colorbar(error_artists[0], cax=error_bar, label="Signed error  /  forecast − actual")
    value_bar.tick_params(labelsize=9); error_bar.tick_params(labelsize=9)

    figure.text(.055, .963, "Forecasts in the same space and time", fontsize=26,
                fontweight="bold", ha="left", va="top")
    domain = "never-observed test sites" if track == "B" else "observed input sites"
    figure.text(.055, .917, f"{SCENARIOS.get(scenario, scenario)}   ·   Track {track}: {domain}",
                fontsize=13, color="#536878", ha="left", va="top")
    clock = figure.text(.055, .884, "", fontsize=11, color="#536878", ha="left", va="top")
    figure.text(.335, .078,
                f"SINGLE-REPLICATION ILLUSTRATION  ·  Replication {replication}  ·  {len(targets)} scored sites\n"
                "Every panel uses the same saved origin and target cells. Forecast parameters stay fixed; no future data are assimilated.\n"
                "Gray cells are outside the scored domain. Color limits are fixed over all models and leads; no values are clipped.\n"
                "Gaussian innovations do not imply Gaussian multistep GRU paths. This display does not establish an aggregate ranking.",
                fontsize=9, linespacing=1.55, color="#647789", va="top")

    def update(index):
        for column, artist in enumerate(artists):
            artist.set_array(field(top_values[column][index]).ravel())
        for model_index, artist in enumerate(error_artists):
            artist.set_array(field(errors[model_index, index]).ravel())
            error_labels[model_index].set_text(f"Current-lead RMSE  {rmse[model_index, index]:.3f}")
            dots[model_index].set_data([index + 1], [rmse[model_index, index]])
        cursor.set_xdata([index + 1, index + 1])
        clock.set_text(f"Lead {index + 1:02d} / {horizon}   ·   Actual time {sim.times[origin + index]:g}"
                       f"   ·   First future row {origin}   ·   Last observed time {sim.times[origin - 1]:g}")
        return artists + error_artists + error_labels + dots + [cursor, clock]

    output.mkdir(parents=True, exist_ok=True)
    files = []
    still_index = horizon // 2
    update(still_index)
    png_path, svg_path = output / "comparison.png", output / "comparison.svg"
    figure.savefig(png_path, dpi=200, metadata={"Title": "Saved native spatiotemporal forecast comparison"})
    figure.savefig(svg_path, metadata={"Date": None, "Title": "Saved native spatiotemporal forecast comparison"})
    files.extend((png_path, svg_path))
    animation_status = {"requested": bool(animate), "fps": fps, "frame_indices": list(range(horizon))}
    try:
        if animate:
            movie = FuncAnimation(figure, update, frames=horizon, interval=1000 / fps,
                                  blit=False, cache_frame_data=False)
            if shutil.which("ffmpeg"):
                mp4_path = output / "comparison.mp4"
                movie.save(mp4_path, writer=FFMpegWriter(
                    fps=fps, codec="libx264", bitrate=-1,
                    extra_args=["-crf", "20", "-pix_fmt", "yuv420p", "-threads", "1"]), dpi=120)
                files.append(mp4_path)
                animation_status["mp4"] = "written"
            else:
                animation_status["mp4"] = "not available: ffmpeg is not installed; GIF is written"
            gif_path = output / "comparison.gif"
            movie.save(gif_path, writer=PillowWriter(fps=fps), dpi=80)
            files.append(gif_path)
            animation_status["gif"] = "written"
    finally:
        plt.close(figure)
    # The result files may be rendered while unrelated benchmark jobs finish;
    # every input consumed here must remain unchanged throughout this rendering.
    for path in inputs:
        if _sha256(path) != input_hashes[str(path.relative_to(run))]:
            raise RuntimeError(f"An input changed during rendering: {path}")
    artifact_manifest = {
        "schema": "saved-model-comparison-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "single explicitly selected replication; no fitting, model selection or aggregate ranking",
        "run_directory": str(run), "run_identity": run_manifest["identity"],
        "run_stage": run_manifest.get("stage"),
        "source_fingerprint": run_manifest.get("source_fingerprint"),
        "renderer_sha256": _sha256(Path(__file__)),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "scenario": scenario, "track": track, "replication": replication,
        "models": list(MODELS), "origin_index": origin, "horizon": horizon,
        "target_site_ids": list(forecasts[0].site_ids),
        "forecast_times": sim.times[origin:origin + horizon].tolist(),
        "grid_shape": [len(y), len(x)], "scored_sites": len(targets),
        "still_lead": still_index + 1, "still_selection": "fixed middle by lead position",
        "truth": "actual noisy observations, not latent or generating-mean values",
        "native_forecasts_only": True, "gray_cells": "outside the common scored domain",
        "value_color_limits": [lo, hi], "error_color_limits": [-error_limit, error_limit],
        "color_policy": "common full min/max over all saved leads and displayed models; no clipping",
        "error_definition": "forecast mean minus actual observation",
        "qualification": {model: receipts[index]["qualified"] for index, model in enumerate(MODELS)},
        "operational_fallback_in_receipt": {model: receipts[index]["fallback"] for index, model in enumerate(MODELS)},
        "rmse_by_lead": {model: rmse[index].tolist() for index, model in enumerate(MODELS)},
        "checks": {"same_origin_horizon_site_order_geometry": True, "inputs_unchanged": True,
                   "saved_native_scores": checks},
        "animation": animation_status,
        "input_sha256": input_hashes,
        "output_sha256": {path.name: _sha256(path) for path in files},
    }
    destination = output / "manifest.json"
    destination.write_text(json.dumps(artifact_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {"output": str(output), "manifest": str(destination),
            "files": [str(path) for path in files], "qualification": artifact_manifest["qualification"],
            "origin_index": origin, "horizon": horizon, "scored_sites": len(targets)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="Saved benchmark run directory")
    parser.add_argument("--scenario", required=True, help="Saved scenario directory, selected explicitly")
    parser.add_argument("--output", required=True, type=Path, help="Directory for PNG/SVG, optional movies and manifest")
    parser.add_argument("--replication", type=int, default=0)
    parser.add_argument("--track", choices=("A", "B"), default="B")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--no-animation", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = render_comparison(args.run, args.output, scenario=args.scenario,
                                   replication=args.replication, track=args.track,
                                   animate=not args.no_animation, fps=args.fps)
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.exit(2, f"Saved comparison could not be rendered: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
