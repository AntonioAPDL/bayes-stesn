"""Render existing KS application results without fitting or selecting a model.

Usage::

    python -m bayes_stesn.applications.ks_report \
        --results RESULTS --output RESULTS/report

Only the predeclared core trajectory 0 / first-target 8000 examples are opened.
All quantitative tables come from the runner's saved summaries. Lead-curve
intervals resample trajectories, never individual sites or origins. Native
probability curves use qualified complete trajectories only. Missing, failed,
pending and inference-unqualified outcomes remain visible in the report.

PNG and PDF figures, compact streamed MP4s, a self-contained offline HTML
report, Markdown report and checksummed report manifest are produced. MP4
encoding does not create intermediate PNG frame dumps. No dataset access,
model execution, scientific selection or test-access gate mutation occurs.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import hashlib
import html
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .ks_runtime import atomic_json, file_hash, now


NATIVE = {"bayes_stesn", "separable_gp", "basis_gru"}
ORDER = ("bayes_stesn", "separable_gp", "basis_gru", "ridge_esn", "pod_nvar",
         "linear_var", "periodic_cnn", "persistence", "training_mean")
LABELS = {
    "bayes_stesn": "Bayesian ST-ESN", "separable_gp": "Separable ST-GP",
    "basis_gru": "Probabilistic basis GRU", "ridge_esn": "Sparse ridge ESN",
    "pod_nvar": "POD-NVAR", "linear_var": "Linear/POD VAR",
    "periodic_cnn": "Periodic residual CNN", "persistence": "Persistence",
    "training_mean": "Training mean",
}
COLORS = {
    "bayes_stesn": "#0072B2", "separable_gp": "#D55E00", "basis_gru": "#009E73",
    "ridge_esn": "#CC79A7", "pod_nvar": "#E69F00", "linear_var": "#56B4E9",
    "periodic_cnn": "#7B3294", "persistence": "#555555", "training_mean": "#999999",
}
TRACKS = {"A": "A · strict unseen sites", "B": "B · all-observed temporal",
          "C": "C · contiguous-site robustness"}
CASE_TRAJECTORY = 0
CASE_FIRST_TARGET = 8000
REPORTED_HORIZONS = (1, 5, 10, 25, 50, 100)
INTERVAL_INDICES = {"A": (0, 5, 11), "B": (0, 16, 32, 48), "C": (0, 5, 11)}
ANIMATION_MODELS = ("bayes_stesn", "basis_gru", "ridge_esn")
DT = .25
REFERENCES = (
    ("Pathak et al. (2018), reservoir forecasting", "https://doi.org/10.1103/PhysRevLett.120.024102"),
    ("Vlachas et al. (2020), recurrent and reservoir comparisons", "https://arxiv.org/abs/1910.05266"),
    ("Gauthier et al. (2021), next generation reservoir computing", "https://arxiv.org/abs/2106.07688"),
    ("Li et al. (2021), Fourier neural operators", "https://arxiv.org/abs/2010.08895"),
    ("Koehler et al. (2024), APEBench architecture context", "https://arxiv.org/abs/2411.00180"),
    ("Li et al. (2018), Hyperband allocation context", "https://www.jmlr.org/papers/v18/16-558.html"),
    ("Vehtari et al. (2021), rank-normalized MCMC diagnostics", "https://arxiv.org/abs/1903.08008"),
    ("Gneiting and Raftery (2007), proper scores", "https://doi.org/10.1198/016214506000001437"),
    ("Scheuerer and Hamill (2015), variogram scores", "https://doi.org/10.1175/MWR-D-14-00269.1"),
)


def _read_json(path: Path) -> tuple[Any, dict[str, Any]]:
    data = path.read_bytes()
    return json.loads(data), {"name": path.name, "bytes": len(data),
                              "sha256": hashlib.sha256(data).hexdigest()}


def _number(value, digits=3):
    if value is None:
        return "—"
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("Nonfinite reported metric must be a failure, not a formatted score")
    if value != 0 and (abs(value) >= 1e4 or abs(value) < 10**(-digits)):
        return f"{value:.{max(2, digits-1)}g}"
    return f"{value:.{digits}f}"


def _estimate(row, key, *, digits=3, interval=True):
    estimate = row.get(key, {})
    if not estimate or estimate.get("mean") is None:
        return "—"
    value = _number(estimate["mean"], digits)
    bounds = estimate.get("ci95")
    if interval and bounds is not None:
        value += f" [{_number(bounds[0], digits)}, {_number(bounds[1], digits)}]"
    return value


def _ordered(values):
    rank = {model: index for index, model in enumerate(ORDER)}
    return sorted(values, key=lambda item: (rank.get(item["model"], len(rank)), item["model"]))


def _table(headers, rows):
    """Return both offline HTML and CommonMark tables from plain strings."""
    header = "".join(f"<th>{html.escape(str(cell))}</th>" for cell in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>" for row in rows)
    markup = f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'
    clean = lambda cell: str(cell).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(clean, headers)) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(map(clean, row)) + " |" for row in rows]
    return markup, "\n".join(lines) + "\n"


def trajectory_curve(records, metric, *, qualified_only=False, seed=20261105, resamples=2000):
    """Pointwise percentile intervals for equally weighted trajectory means."""
    chosen = [record for record in records if metric in record
              and (not qualified_only or record.get("qualified", False))]
    if not chosen:
        return None
    chosen.sort(key=lambda record: record["trajectory"])
    ids = [record["trajectory"] for record in chosen]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate trajectory in one model/seed lead curve")
    values = np.asarray([record[metric] for record in chosen], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 100 or not np.isfinite(values).all():
        raise ValueError("Lead curves require finite 100-step trajectory arrays")
    mean = values.mean(axis=0, dtype=np.float64)
    bounds = None
    if len(values) > 1:
        # Multinomial counts are equivalent to resampling whole trajectories,
        # but avoid a resamples × episodes × leads temporary array.
        rng = np.random.default_rng(seed)
        counts = rng.multinomial(len(values), np.full(len(values), 1 / len(values)), size=resamples)
        draws = counts @ values / len(values)
        bounds = np.quantile(draws, [.025, .975], axis=0)
    return {"mean": mean, "ci95": bounds, "n": len(values), "trajectories": ids,
            "qualified_n": sum(bool(record.get("qualified", False)) for record in chosen),
            "uncertainty": "pointwise trajectory-cluster percentile bootstrap; conditional on selection"}


@dataclass
class Example:
    track: str
    model: str
    seed: int
    truth: np.ndarray
    mean: np.ndarray
    x: np.ndarray
    input_x: np.ndarray
    times: np.ndarray
    site_ids: np.ndarray
    qualified: bool
    levels: np.ndarray | None = None
    quantiles: np.ndarray | None = None


class Report:
    def __init__(self, results, output, *, dpi=200):
        self.results, self.output = Path(results).resolve(), Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "figures").mkdir(exist_ok=True)
        (self.output / "animations").mkdir(exist_ok=True)
        self.dpi = int(dpi)
        self.input_receipts = []
        objects = {}
        for name in ("summary.json", "episode_results.json", "horizon_curves.json", "freeze.json"):
            objects[name], receipt = _read_json(self.results / name)
            self.input_receipts.append(receipt)
        self.summary = objects["summary.json"]
        self.episodes = objects["episode_results.json"]
        self.curves = objects["horizon_curves.json"]
        self.frozen = objects["freeze.json"]
        if self.summary.get("freeze_identity") != self.frozen.get("identity"):
            raise ValueError("Summary and frozen selection identities differ")
        self.seed = int(self.frozen["model_seed"])
        self.rows = self.summary["rows"]
        self.primary_rows = [row for row in self.rows if row["seed"] == self.seed]
        self.by_episode = {(row["track"], row["model"], row["seed"], row["trajectory"]): row
                           for row in self.episodes}
        if len(self.by_episode) != len(self.episodes):
            raise ValueError("Duplicate episode-result identities")
        self.examples = {}
        self.figures, self.movies, self.notices = [], [], []
        self.artifact_paths = set()
        self.curve_cache = {}
        self.experiment = None
        path = self.results / "experiment.json"
        if path.exists():
            self.experiment, receipt = _read_json(path)
            self.input_receipts.append(receipt)
        for row in self.rows:
            observed = self.records(row["track"], row["model"], row["seed"])
            if len(observed) != row["complete_trajectories"]:
                raise ValueError("Summary/curve counts disagree; regenerate a coherent summary before plotting")

    def records(self, track, model, seed=None):
        seed = self.seed if seed is None else seed
        return [record for record in self.curves if record["track"] == track
                and record["model"] == model and record["seed"] == seed]

    def curve(self, track, model, metric, *, qualified=False):
        key = (track, model, metric, qualified)
        if key not in self.curve_cache:
            self.curve_cache[key] = trajectory_curve(self.records(track, model), metric,
                                                     qualified_only=qualified)
        return self.curve_cache[key]

    def load_examples(self):
        for row in self.primary_rows:
            track, model = row["track"], row["model"]
            if track not in ("A", "B"):
                continue
            name = f"core_{track}_{model}_s{self.seed}_t{CASE_TRAJECTORY:02d}_a{CASE_FIRST_TARGET}.npz"
            path = self.results / "predictions" / name
            if not path.exists():
                self.notices.append(f"{TRACKS[track]} / {LABELS.get(model, model)}: predeclared example unavailable (failed or pending); no substitute case selected.")
                continue
            if path.stat().st_size > 64 * 1024**2:
                raise MemoryError("A compact example unexpectedly exceeds the 64 MiB read budget")
            with np.load(path, allow_pickle=False) as archive:
                required = ("truth", "mean", "x", "input_x", "times", "site_ids")
                if any(key not in archive for key in required):
                    raise ValueError(f"Incomplete example fields: {name}")
                arrays = {key: np.array(archive[key], copy=True) for key in required}
                levels = np.array(archive["quantile_levels"], copy=True) if "quantile_levels" in archive else None
                quantiles = np.array(archive["quantiles"], copy=True) if "quantiles" in archive else None
            expected_sites = 12 if track == "A" else 64
            if arrays["truth"].shape != (100, expected_sites) or arrays["mean"].shape != arrays["truth"].shape:
                raise ValueError(f"Unexpected official example domain: {name}")
            if arrays["x"].shape != (expected_sites,) or arrays["times"].shape != (100,):
                raise ValueError(f"Unexpected example coordinate/time shape: {name}")
            if arrays["input_x"].shape != ((39 if track == "A" else 64),) or arrays["site_ids"].shape != (expected_sites,):
                raise ValueError(f"Unexpected example input/target partition: {name}")
            for coordinate in (arrays["x"], arrays["input_x"]):
                if len(np.unique(coordinate)) != len(coordinate) or np.any(coordinate < 0) or np.any(coordinate >= 22):
                    raise ValueError(f"Invalid periodic example coordinates: {name}")
            if any(not np.isfinite(value).all() for key, value in arrays.items() if key != "site_ids"):
                raise ValueError(f"Nonfinite example was incorrectly retained as a successful forecast: {name}")
            if not np.array_equal(arrays["times"], DT * np.arange(CASE_FIRST_TARGET, CASE_FIRST_TARGET + 100)):
                raise ValueError("Saved example is not the predeclared first-target interval")
            if (levels is None) != (quantiles is None):
                raise ValueError("Saved quantile levels and values must occur together")
            if levels is not None:
                if levels.ndim != 1 or quantiles.shape != (len(levels), 100, expected_sites) or not np.isfinite(quantiles).all():
                    raise ValueError("Saved interval dimensions/values are invalid")
                if not np.all(np.diff(levels) > 0) or np.any(levels <= 0) or np.any(levels >= 1) or np.any(np.diff(quantiles, axis=0) < 0):
                    raise ValueError("Saved interval levels/values are not ordered quantiles")
            episode = self.by_episode.get((track, model, self.seed, CASE_TRAJECTORY), {})
            qualified = bool(episode.get("qualified", False))
            self.examples[(track, model)] = Example(track, model, self.seed, **arrays,
                                                   qualified=qualified, levels=levels, quantiles=quantiles)
            self.input_receipts.append({"name": str(path.relative_to(self.results)),
                                        "bytes": path.stat().st_size, "sha256": file_hash(path)})

    def _save_figure(self, figure, stem, title, caption, track=None):
        paths = []
        for suffix in ("png", "pdf"):
            destination = self.output / "figures" / f"{stem}.{suffix}"
            temporary = destination.with_name("." + destination.stem + ".tmp." + suffix)
            try:
                figure.savefig(temporary, format=suffix, dpi=self.dpi, bbox_inches="tight", facecolor="white")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            relative = str(destination.relative_to(self.output))
            self.artifact_paths.add(relative)
            paths.append(relative)
        self.figures.append({"title": title, "caption": caption, "track": track,
                             "png": paths[0], "pdf": paths[1]})

    @staticmethod
    def _lead_axes(axis):
        axis.set_xlim(1, 100)
        axis.set_xticks([1, 25, 50, 75, 100])
        axis.set_xlabel("Forecast lead (saved steps)")
        physical = axis.secondary_xaxis("top", functions=(lambda h: DT*h, lambda t: t/DT))
        physical.set_xlabel("Elapsed physical time")
        physical.set_xticks([.25, 6.25, 12.5, 18.75, 25])
        axis.grid(alpha=.18, linewidth=.6)

    @staticmethod
    def _line(axis, result, model, *, probability=False):
        x = np.arange(1, 101)
        label = f"{LABELS.get(model, model)} (n={result['n']}"
        if model in NATIVE and not probability:
            label += f", q={result['qualified_n']}"
        label += ")"
        axis.plot(x, result["mean"], color=COLORS.get(model), label=label, linewidth=1.8)
        if result["ci95"] is not None:
            axis.fill_between(x, *result["ci95"], color=COLORS.get(model), alpha=.12, linewidth=0)

    @staticmethod
    def _optional_symlog(axis, arrays):
        finite = np.concatenate([np.asarray(value).ravel() for value in arrays]) if arrays else np.empty(0)
        positive = finite[finite > 0]
        if len(positive) and positive.max() > max(10., 25*float(np.median(positive))):
            threshold = max(.1, float(np.median(positive)))
            axis.set_yscale("symlog", linthresh=threshold)
            axis.set_ylabel(axis.get_ylabel() + "\n(symmetric log scale)")

    def error_curves(self, plt):
        for track in ("A", "B", "C"):
            models = [row["model"] for row in _ordered(self.primary_rows) if row["track"] == track]
            figure, axes = plt.subplots(1, 2, figsize=(11, 5.2))
            shown = False
            for axis, metric, label in zip(axes, ("rmse", "nrmse"), ("RMSE (physical field units)", "RMSE / permitted training SD")):
                arrays = []
                for model in models:
                    result = self.curve(track, model, metric)
                    if result is not None:
                        self._line(axis, result, model)
                        arrays.append(result["mean"]); shown = True
                axis.set_ylabel(label)
                self._lead_axes(axis)
                self._optional_symlog(axis, arrays)
            axes[1].axhline(.4, color="#333333", linestyle="--", linewidth=.8)
            axes[1].text(.98, .02, "VPT threshold = 0.4", transform=axes[1].transAxes, ha="right", fontsize=8)
            if shown:
                handles, labels = axes[0].get_legend_handles_labels()
                figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
                figure.subplots_adjust(left=.08, right=.98, bottom=.27, top=.78, wspace=.28)
                figure.suptitle(f"{TRACKS[track]} · point error by lead", y=.99)
                self._save_figure(figure, f"error_horizon_{track}", f"{TRACKS[track]}: point error",
                    "Equal trajectory means, with pointwise 95% trajectory-cluster bootstrap bands. "
                    "n is the number of complete trajectories; q is the qualified count for native probabilistic fits. "
                    "Point curves include complete unqualified cases as labeled diagnostics; probability curves use qualified cases only. "
                    "The horizontal 0.4 line is the prespecified VPT convention, not a Lyapunov normalization.", track)
            plt.close(figure)

    def probability_curves(self, plt):
        for track in ("A", "B", "C"):
            figure, axes = plt.subplots(1, 3, figsize=(12, 4.7))
            shown = False
            for axis, metric, label in zip(axes, ("crps_by_horizon", "coverage_0.9", "width_0.9"),
                                           ("CRPS (physical units)", "90% marginal coverage", "90% interval width (physical units)")):
                arrays = []
                for model in ORDER:
                    if model not in NATIVE:
                        continue
                    result = self.curve(track, model, metric, qualified=True)
                    if result is not None:
                        self._line(axis, result, model, probability=True)
                        arrays.append(result["mean"]); shown = True
                axis.set_ylabel(label); self._lead_axes(axis)
                if metric != "coverage_0.9":
                    self._optional_symlog(axis, arrays)
            axes[1].axhline(.9, color="#333333", linestyle="--", linewidth=.8)
            axes[1].set_ylim(0, 1.01)
            if shown:
                handles, labels = axes[0].get_legend_handles_labels()
                figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
                figure.subplots_adjust(left=.065, right=.98, bottom=.2, top=.77, wspace=.35)
                figure.suptitle(f"{TRACKS[track]} · qualified predictive distributions", y=.99)
                self._save_figure(figure, f"probability_horizon_{track}", f"{TRACKS[track]}: predictive distributions",
                    "Only qualified complete trajectories enter these curves; each legend gives its actual n. "
                    "Bands are pointwise 95% trajectory-cluster bootstrap intervals, conditional on selected configurations. "
                    "Coverage targets the clean simulated field. Native residual uncertainty is retained; no arbitrary interval floor or sensor noise was added.", track)
            else:
                self.notices.append(f"{TRACKS[track]}: no qualified complete probability curves are available.")
            plt.close(figure)

    def heatmaps(self, plt):
        from matplotlib.colors import Normalize, SymLogNorm

        for track in ("A", "B"):
            examples = [self.examples[(track, model)] for model in ORDER if (track, model) in self.examples]
            if not examples:
                continue
            field_limit = max(float(np.max(np.abs(value))) for example in examples for value in (example.truth, example.mean))
            error_limit = max(float(np.max(np.abs(example.mean-example.truth))) for example in examples)
            field_limit, error_limit = max(field_limit, 1e-8), max(error_limit, 1e-8)
            field_norm = (SymLogNorm(linthresh=2., vmin=-field_limit, vmax=field_limit)
                          if field_limit > 12 else Normalize(-field_limit, field_limit))
            error_norm = (SymLogNorm(linthresh=.5, vmin=-error_limit, vmax=error_limit)
                          if error_limit > 12 else Normalize(-error_limit, error_limit))
            normalization = "Field/error color limits are shared across all saved model examples within this track."
            if field_limit > 12 or error_limit > 12:
                normalization += " Symmetric logarithmic colors retain extreme forecasts without clipping; linear thresholds are |field|=2 and |error|=0.5 when that scale is used."
            for example in examples:
                figure = plt.figure(figsize=(11.5, 5.7))
                grid = figure.add_gridspec(2, 3, height_ratios=(7, 1.2), hspace=.56, wspace=.32)
                axes = [figure.add_subplot(grid[0, j]) for j in range(3)]
                arrays = (example.truth, example.mean, example.mean-example.truth)
                images = []
                for j, (axis, values, title) in enumerate(zip(axes, arrays, ("Truth", "Forecast mean", "Signed error: forecast − truth"))):
                    images.append(axis.imshow(values.T, origin="lower", interpolation="nearest", aspect="auto",
                                              extent=(.125, 25.125, -.5, len(example.x)-.5),
                                              cmap="RdBu_r", norm=field_norm if j < 2 else error_norm))
                    ticks = np.unique(np.linspace(0, len(example.x)-1, min(7, len(example.x))).round().astype(int))
                    axis.set_yticks(ticks, [f"{example.x[k]:.2f}" for k in ticks])
                    axis.set_xlabel("Elapsed physical time")
                    axis.set_ylabel("Official target x (one row per site)")
                    axis.set_title(title, fontsize=10)
                figure.colorbar(images[1], ax=axes[:2], fraction=.027, pad=.02, label="Field (physical units)")
                figure.colorbar(images[2], ax=axes[2], fraction=.046, pad=.04, label="Signed error (physical units)")
                rug = figure.add_subplot(grid[1, :])
                rug.plot(example.input_x, np.ones(len(example.input_x)), "|", color="#333333", markersize=10)
                rug.plot(example.x, np.zeros(len(example.x)), "o", color=COLORS[example.model], markersize=3)
                rug.set_xlim(0, 22); rug.set_ylim(-.6, 1.6)
                rug.set_yticks([0, 1], ["Official targets", "Permitted inputs"])
                rug.set_xticks([0, 5.5, 11, 16.5, 22]); rug.set_xlabel("Physical periodic coordinate x ∈ [0, 22)")
                for side in ("top", "left", "right"):
                    rug.spines[side].set_visible(False)
                qualifier = " · inference unqualified" if example.model in NATIVE and not example.qualified else ""
                figure.suptitle(f"{TRACKS[track]} · {LABELS[example.model]}{qualifier}\n"
                               f"Predeclared trajectory 0, first target 8000; {len(example.x)} official target sites", fontsize=12, y=1.02)
                self._save_figure(figure, f"heatmap_{track}_{example.model}",
                    f"{TRACKS[track]}: {LABELS[example.model]} example",
                    f"Predeclared case: core trajectory 0, first target 8000 (adapter origin 7999). "
                    f"Exactly {len(example.x)} official target-site rows are shown; row positions are categorical and labeled by physical x. "
                    "The lower rug shows static permitted input and target coordinates. No error is claimed at unscored coordinates. "
                    + normalization + (" The point forecast shown is diagnostic because this fit is inference-unqualified." if qualifier else ""), track)
                plt.close(figure)

    def interval_traces(self, plt):
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        for (track, model), example in self.examples.items():
            if model not in NATIVE or example.quantiles is None:
                continue
            if not example.qualified:
                self.notices.append(f"{TRACKS[track]} / {LABELS[model]}: interval trace omitted because the predeclared example is inference-unqualified.")
                continue
            def quantile(level):
                matches = np.flatnonzero(np.isclose(example.levels, level, rtol=0, atol=1e-12))
                if len(matches) != 1:
                    raise ValueError(f"Missing saved quantile level {level}")
                return example.quantiles[matches[0]]
            lower90, upper90, lower50, upper50 = map(quantile, (.05, .95, .25, .75))
            chosen = INTERVAL_INDICES[track]
            figure, axes = plt.subplots(len(chosen), 1, figsize=(8.6, 2.05*len(chosen)+1.2), sharex=True, squeeze=False)
            leads = np.arange(1, 101) * DT
            for axis, index in zip(axes[:, 0], chosen):
                axis.fill_between(leads, lower90[:, index], upper90[:, index], color=COLORS[model], alpha=.14)
                axis.fill_between(leads, lower50[:, index], upper50[:, index], color=COLORS[model], alpha=.30)
                axis.plot(leads, example.mean[:, index], color=COLORS[model], linewidth=1.5)
                axis.plot(leads, example.truth[:, index], color="black", linewidth=1.2)
                axis.set_ylabel("Field (physical units)")
                axis.set_title(f"Official target position {index}: site {example.site_ids[index]}, x={example.x[index]:.5g}", loc="left", fontsize=9)
                axis.grid(alpha=.18)
            axes[-1, 0].set_xlabel("Elapsed physical time (0.25 × lead)")
            handles = [Line2D([], [], color="black", label="Clean truth"),
                       Line2D([], [], color=COLORS[model], label="Mean of complete paths"),
                       Patch(color=COLORS[model], alpha=.30, label="50% central interval"),
                       Patch(color=COLORS[model], alpha=.14, label="90% central interval")]
            figure.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False)
            figure.suptitle(f"{TRACKS[track]} · {LABELS[model]}\nQualified predeclared case: trajectory 0, first target 8000", fontsize=12)
            figure.subplots_adjust(left=.12, right=.97, bottom=.13, top=.88, hspace=.42)
            self._save_figure(figure, f"intervals_{track}_{model}", f"{TRACKS[track]}: {LABELS[model]} intervals",
                "Fixed target positions were declared before test access: " + ", ".join(map(str, chosen)) + ". "
                "Lines show clean truth and the mean of complete predictive paths; shaded bands are empirical central 50% and 90% intervals. "
                "This individual case is qualified; aggregate coverage and width use all qualified complete trajectories.", track)
            plt.close(figure)

    def compute_plot(self, plt):
        figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
        shown = False
        for axis, track in zip(axes, ("A", "B")):
            for j, row in enumerate(_ordered([r for r in self.primary_rows if r["track"] == track])):
                estimate = row.get("rmse", {})
                seconds = float(row.get("fit_seconds", 0))
                if estimate.get("mean") is None or seconds <= 0:
                    continue
                model = row["model"]; mean = float(estimate["mean"])
                axis.scatter(seconds, mean, color=COLORS.get(model), s=42, zorder=3)
                if estimate.get("ci95") is not None:
                    axis.vlines(seconds, *estimate["ci95"], color=COLORS.get(model), linewidth=1.3)
                axis.annotate(f"{LABELS.get(model, model)}\nn={estimate['n']}", (seconds, mean),
                              xytext=(7, 5 if j % 2 == 0 else -17), textcoords="offset points", fontsize=7)
                shown = True
            axis.set_xscale("log"); axis.set_xlabel("Total selected-fit wall time (seconds; log scale)")
            axis.set_ylabel("Trajectory-mean RMSE (physical units)")
            axis.set_title(TRACKS[track]); axis.grid(alpha=.18)
            axis.margins(x=.25, y=.25)
        if shown:
            figure.suptitle("Predictive accuracy versus recorded fitting cost", fontsize=12)
            figure.subplots_adjust(left=.08, right=.95, bottom=.17, top=.85, wspace=.3)
            self._save_figure(figure, "accuracy_compute", "Accuracy versus fitting cost",
                "Costs are summed selected-configuration fit wall times recorded by the runner, including recorded failed fit attempts. "
                "They exclude screening and forecast time, which are separately reported. RMSE summarizes complete trajectories only; labels show its actual n. "
                "Vertical intervals are trajectory-cluster 95% intervals conditional on selection. Tracks remain separate.")
        plt.close(figure)

    def animate(self, plt, enabled=True):
        if not enabled:
            self.notices.append("Animation generation was explicitly disabled for this report invocation.")
            return
        from matplotlib.animation import FFMpegWriter

        if not FFMpegWriter.isAvailable():
            self.notices.append("MP4 generation unavailable: the existing ffmpeg executable could not be used. No encoder was downloaded.")
            return
        for track in ("A", "B"):
            available = [self.examples[(track, model)] for model in ANIMATION_MODELS if (track, model) in self.examples]
            if len(available) < 2:
                self.notices.append(f"{TRACKS[track]}: side-by-side animation requires at least two saved predeclared model examples.")
                continue
            limit = max(float(np.max(np.abs(value))) for example in available for value in (example.truth, example.mean))
            limit = max(limit, 1e-3)
            figure, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
            artists = []
            for axis, model in zip(axes, ANIMATION_MODELS):
                example = self.examples.get((track, model))
                axis.set_xlim(0, 22); axis.set_ylim(-1.08*limit, 1.08*limit)
                axis.set_xticks([0, 5.5, 11, 16.5, 22]); axis.set_xlabel("Physical periodic coordinate x")
                axis.grid(alpha=.16)
                if limit > 12:
                    axis.set_yscale("symlog", linthresh=2.)
                if example is None:
                    axis.set_title(LABELS[model], fontsize=10)
                    axis.text(.5, .5, "Saved case unavailable\n(failed or pending)", transform=axis.transAxes,
                              ha="center", va="center", fontsize=10)
                    continue
                style = {"linestyle": "none", "marker": "o", "markersize": 4} if track == "A" else {"linewidth": 1.5}
                truth_line, = axis.plot(example.x, example.truth[0], color="black", label="Truth", **style)
                mean_line, = axis.plot(example.x, example.mean[0], color=COLORS[model], label="Forecast mean", **style)
                axis.plot(example.input_x, np.full(len(example.input_x), -.10), "|", color="#777777", markersize=4,
                          transform=axis.get_xaxis_transform(), clip_on=False)
                qualifier = "\nInference unqualified; point diagnostic" if model in NATIVE and not example.qualified else ""
                axis.set_title(LABELS[model] + qualifier, fontsize=9)
                artists.append((example, truth_line, mean_line))
            axes[0].set_ylabel("Field (physical units)" + ("; symlog" if limit > 12 else ""))
            title = figure.suptitle("", fontsize=12)
            target_count = 12 if track == "A" else 64
            note = f"Predeclared trajectory 0 / first target 8000 · {target_count} official target sites only · gray rug: permitted input coordinates"
            figure.text(.5, .025, note, ha="center", fontsize=8)
            figure.subplots_adjust(left=.07, right=.98, bottom=.23, top=.78, wspace=.17)
            for axis in reversed(axes):
                if axis.get_legend_handles_labels()[0]:
                    axis.legend(loc="lower right", fontsize=8, frameon=False)
                    break
            destination = self.output / "animations" / f"ks_{track}_t00_a8000.mp4"
            temporary = destination.with_name("." + destination.stem + ".tmp.mp4")
            try:
                writer = FFMpegWriter(fps=10, codec="libx264", bitrate=1400,
                    extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-preset", "veryfast"])
                with writer.saving(figure, str(temporary), dpi=100):
                    for frame in range(100):
                        title.set_text(f"{TRACKS[track]} · lead {frame+1}/100 · elapsed physical time {(frame+1)*DT:.2f}")
                        for example, truth_line, mean_line in artists:
                            truth_line.set_ydata(example.truth[frame]); mean_line.set_ydata(example.mean[frame])
                        writer.grab_frame(facecolor="white")
                os.replace(temporary, destination)
                relative = str(destination.relative_to(self.output)); self.artifact_paths.add(relative)
                self.movies.append({"track": track, "path": relative, "frames": 100, "fps": 10,
                    "title": f"{TRACKS[track]}: predeclared autonomous forecast",
                    "caption": note + ". The same fixed physical amplitude scale is used across model panels; no forecast is clipped. "
                    + ("Only target-site markers are shown in the strict track; no interpolated full field is claimed." if track == "A" else "Lines connect the 64 observed evaluation locations.")})
            except Exception as error:
                self.notices.append(f"{TRACKS[track]}: MP4 encoding failed ({type(error).__name__}: {error}); static figures are retained.")
            finally:
                temporary.unlink(missing_ok=True)
                plt.close(figure)

    def _completion(self):
        expected = {(track, model) for track in ("A", "B") for model in self.frozen["selected"][track]}
        actual = {(row["track"], row["model"]): row for row in self.primary_rows if row["track"] in ("A", "B")}
        complete = expected == set(actual) and all(row["pending_origins"] == 0 for row in actual.values())
        pending = sum(row["pending_origins"] for row in actual.values())
        failures = sum(row["failed_origins"] for row in actual.values())
        extensions_pending = sum(row["pending_origins"] for row in self.rows if row["track"] == "C" or row["seed"] != self.seed)
        return {"main_execution_terminal": complete, "main_pending_origins": pending,
                "main_failed_origins": failures, "extension_pending_origins": extensions_pending,
                "all_requested_scheduled_origins_terminal": complete and extensions_pending == 0}

    def selected_horizons(self):
        """Export requested lead values from saved, trajectory-level curves."""
        values = []
        for row in _ordered(self.primary_rows):
            for horizon in REPORTED_HORIZONS:
                item = {"track": row["track"], "model": row["model"], "seed": self.seed,
                        "horizon": horizon, "elapsed_physical_time": DT*horizon,
                        "complete_trajectories": row["complete_trajectories"],
                        "qualified_complete_trajectories": row["qualified_trajectories"]}
                for metric, curve_key in (("rmse", "rmse"), ("nrmse", "nrmse"),
                                          ("mae", "mae"), ("crps", "crps_by_horizon"),
                                          ("coverage_0.9", "coverage_0.9"), ("width_0.9", "width_0.9")):
                    probability = metric in ("crps", "coverage_0.9", "width_0.9")
                    result = self.curve(row["track"], row["model"], curve_key, qualified=probability)
                    if result is not None:
                        item[metric] = {"n": result["n"], "mean": float(result["mean"][horizon-1]),
                                        "ci95": None if result["ci95"] is None else result["ci95"][:, horizon-1].tolist()}
                values.append(item)
        destination = self.output / "selected_horizons.json"
        atomic_json(destination, {"freeze_identity": self.frozen["identity"], "rows": values,
                                 "aggregation": "equal means of complete trajectory lead scores; qualified only for probabilities",
                                 "uncertainty": "pointwise trajectory-cluster percentile bootstrap, 2000 resamples, seed 20261105"})
        self.artifact_paths.add(destination.name)
        return values

    def write_reports(self):
        completion = self._completion()
        lead_values = self.selected_horizons()
        title = "Kuramoto–Sivashinsky: Bayesian ST-ESN and comparator study"
        status = ("All scheduled main-track origins have terminal outcomes." if completion["main_execution_terminal"]
                  else "Main-track execution is incomplete; pending work is explicitly retained below.")
        status += (f" Failed main-track origins: {completion['main_failed_origins']}; "
                   f"pending main-track origins: {completion['main_pending_origins']}; "
                   f"pending sensitivity/robustness origins: {completion['extension_pending_origins']}.")
        html_parts, md_parts = [], [f"# {title}\n", "## Execution status\n", status + "\n"]

        def section(heading, paragraphs=()):
            html_parts.append(f"<h2>{html.escape(heading)}</h2>")
            md_parts.append(f"## {heading}\n")
            for text in paragraphs:
                html_parts.append(f"<p>{html.escape(text)}</p>"); md_parts.append(text + "\n")

        def add_table(headers, rows):
            h, m = _table(headers, rows); html_parts.append(h); md_parts.append(m)

        section("Scope and interpretation", [
            "Custom, literature-grounded experiment on the unchanged prepared KS field: periodic [0,22), 64 observed grid locations from a 128-point solver, saved interval 0.25. This is not a published leaderboard reproduction or a state-of-the-art claim.",
            "Track A fits and conditions on 39 interleaved training sites and scores 12 never-trained test sites. Track B fits, conditions and scores all 64 observation-panel sites. Each core trajectory has separate parameters fitted using training times [0,6000); validation [6000,8000) selected configurations; test [8000,10000) is evaluation-only. The two tracks are never combined into one ranking.",
            "Each test trajectory has 20 autonomous 100-step forecasts, starting at 8000,8100,...,9900. Later origins may use permitted observations available by that origin; no parameters are refitted and no within-window future observations update a rollout. The frozen history length is " + str(self.frozen["history"]) + " saved states.",
            "All displayed values are in physical field units after the delivered training-only inverse transform. Native statistical residual/nugget terms represent model discrepancy for the clean field. They are retained in predictive paths; no added measurement-noise view or arbitrary interval floor is used.",
        ])
        section("Aggregation, qualification and uncertainty", [
            "Origin losses are summarized within trajectories before equally weighted trajectory aggregation. Point estimates use means of complete predictive paths for probabilistic families. Complete-trajectory successful-only point scores remain explicitly labeled; failed and pending origins remain in scheduled denominators. Probability scores use qualified complete trajectories only.",
            "Brackets are pointwise 95% trajectory-cluster bootstrap intervals conditional on the frozen validation-selected configurations. They do not treat sites, origins, MCMC draws or model seeds as independent scientific replicates, and they do not average over model selection. Differences use common eligible trajectories and are descriptive paired summaries without multiplicity adjustment.",
            "Valid prediction time (VPT) is the first lead at which spatial RMSE/training SD exceeds 0.4, right censored at 100. A normalized spatial RMSE above 10 is a separately flagged finite divergence. No forecast is silently clipped or replaced. CRPS scores the empirical predictive distribution; no iid unbiased finite-ensemble correction is claimed.",
            "Spatial energy uses the Euclidean norm divided by sqrt(target-site count), with a seeded 1,024-pair Monte Carlo approximation to the empirical path-pair term. Variogram score uses power 0.5 and at most 128 fixed geometry-only site pairs weighted by inverse circle-chord distance. These are spatial per-lead scores, not whole-temporal-path scores.",
        ])
        for track in ("A", "B", "C"):
            rows = _ordered([row for row in self.primary_rows if row["track"] == track])
            if not rows:
                continue
            section(TRACKS[track] + " · primary seed", [
                "n denotes complete scored trajectories; q denotes qualified complete trajectories. "
                "All failed/pending origins remain in the listed scheduled denominator. "
                "A dash means no eligible estimate, not zero loss."
            ])
            add_table(["Model", "n / scheduled", "q", "Failed / pending / origins", "RMSE [95% CI]", "NRMSE [95% CI]", "MAE", "VPT steps"], [
                [LABELS.get(row["model"], row["model"]), f"{row['complete_trajectories']} / {row['scheduled_trajectories']}",
                 row["qualified_trajectories"], f"{row['failed_origins']} / {row['pending_origins']} / {row['scheduled_origins']}",
                 _estimate(row, "rmse"), _estimate(row, "nrmse"), _estimate(row, "mae", interval=False),
                 _estimate(row, "vpt_steps", digits=1, interval=False)] for row in rows])
            native = [row for row in rows if row["model"] in NATIVE]
            if native:
                add_table(["Qualified distribution", "q", "CRPS [95% CI]", "90% coverage", "90% width", "Spatial energy", "Spatial variogram"], [
                    [LABELS[row["model"]], row.get("crps", {}).get("n", 0), _estimate(row, "crps"),
                     _estimate(row, "coverage_0.9", interval=False), _estimate(row, "width_0.9", interval=False),
                     _estimate(row, "spatial_energy", interval=False), _estimate(row, "spatial_variogram", interval=False)] for row in native])
                add_table(["Model", "Central level", "Eligible trajectories", "Coverage [95% CI]", "Width [95% CI]"], [
                    [LABELS[row["model"]], f"{100*level:g}%", row.get(f"coverage_{level}", {}).get("n", 0),
                     _estimate(row, f"coverage_{level}"), _estimate(row, f"width_{level}")]
                    for row in native for level in (.5, .8, .9, .95)])
            add_table(["Model", "Selected config", "Fit seconds", "Forecast seconds", "Peak worker MiB", "Finite divergence fraction", "VPT censored fraction"], [
                [LABELS.get(row["model"], row["model"]), row["selected_config"], _number(row["fit_seconds"], 1),
                 _number(row["forecast_seconds"], 1), _number(row["peak_worker_rss_bytes"] / 1024**2, 1),
                 _estimate(row, "divergence_fraction", interval=False), _estimate(row, "vpt_censored_fraction", interval=False)] for row in rows])
            if any("inference_unqualified_fits" in row for row in rows):
                add_table(["Model", "Scheduled fits", "Inference-unqualified fits", "Failed fits", "Pending fits"], [
                    [LABELS.get(row["model"], row["model"]), row["scheduled_trajectories"],
                     row.get("inference_unqualified_fits", "—"), row.get("failed_fits", "—"),
                     row.get("pending_fits", "—")] for row in rows])
            leads = [item for item in lead_values if item["track"] == track]
            lead_html, lead_md = _table(
                ["Model", "Lead", "Physical time", "n / q", "RMSE [95% CI]", "NRMSE", "MAE", "Qualified CRPS [95% CI]"],
                [[LABELS.get(item["model"], item["model"]), item["horizon"], _number(item["elapsed_physical_time"], 2),
                  f"{item['complete_trajectories']} / {item['qualified_complete_trajectories']}",
                  _estimate(item, "rmse"), _estimate(item, "nrmse", interval=False), _estimate(item, "mae", interval=False),
                  _estimate(item, "crps")] for item in leads])
            html_parts.append("<details><summary>Values at the six prespecified forecast leads</summary>" + lead_html + "</details>")
            md_parts += ["### Values at the six prespecified forecast leads\n", lead_md]

        pairs = [row for row in self.summary.get("paired_comparisons", []) if row["metric"] in ("nrmse", "crps")]
        if pairs:
            section("Paired Bayesian-minus-comparator differences", [
                "Negative differences favor Bayesian ST-ESN for the stated loss. Each row uses only its common eligible complete trajectories. A confidence interval covering zero is not evidence of equivalence. Native CRPS pairs require qualification; no inference-unqualified probability estimates enter these rows."
            ])
            add_table(["Track", "Comparator", "Metric", "Paired trajectories", "Mean difference [95% CI]"], [
                [row["track"], LABELS.get(row["comparator"], row["comparator"]), row["metric"], row["n"],
                 _estimate({"difference": row}, "difference")] for row in pairs])
        sensitivity = _ordered([row for row in self.rows if row["seed"] != self.seed])
        if sensitivity:
            section("Fixed-seed sensitivity", [
                "Additional model seeds are separately labeled sensitivity fits, not independent posterior paths or extra scientific trajectories. These rows use the prespecified episode subset and are not pooled into Bayesian predictive distributions."
            ])
            add_table(["Track", "Model", "Model seed", "n / scheduled", "q", "Failed / pending", "RMSE [95% CI]", "CRPS [95% CI]"], [
                [row["track"], LABELS.get(row["model"], row["model"]), row["seed"],
                 f"{row['complete_trajectories']} / {row['scheduled_trajectories']}", row["qualified_trajectories"],
                 f"{row['failed_origins']} / {row['pending_origins']}", _estimate(row, "rmse"), _estimate(row, "crps")]
                for row in sensitivity])

        section("Figures and predeclared examples", [
            "All example fields and traces use core trajectory 0 and first target 8000, fixed independently of model performance. Strict-site heatmaps contain exactly the 12 official test sites; the static rug distinguishes permitted input sites. No best-looking case was substituted. Lead bands resample trajectories. PDF figures are separate vector/raster publication exports of the same plots."
        ])
        for figure in self.figures:
            path = self.output / figure["png"]
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            html_parts.append(f'<figure><h3>{html.escape(figure["title"])}</h3><img alt="{html.escape(figure["title"], quote=True)}" '
                              f'src="data:image/png;base64,{data}"><figcaption>{html.escape(figure["caption"])}</figcaption>'
                              f'<p><a href="{html.escape(figure["pdf"], quote=True)}">PDF figure</a></p></figure>')
            md_parts += [f"### {figure['title']}\n", f"![{figure['title']}]({figure['png']})\n",
                         figure["caption"] + "\n", f"[PDF figure]({figure['pdf']})\n"]
        if self.movies:
            section("Compact animations", [
                "Each video has 100 frames at 10 frames/s and shows the same predeclared case. No intermediate frame dump is retained. Static figures remain the quantitative reporting evidence."
            ])
            for movie in self.movies:
                data = base64.b64encode((self.output / movie["path"]).read_bytes()).decode("ascii")
                html_parts.append(f'<figure><h3>{html.escape(movie["title"])}</h3><video controls preload="metadata" '
                                  f'src="data:video/mp4;base64,{data}"></video><figcaption>{html.escape(movie["caption"])}</figcaption></figure>')
                md_parts += [f"### {movie['title']}\n", f"[MP4 animation]({movie['path']})\n", movie["caption"] + "\n"]
        if self.notices:
            section("Missing or ineligible report elements")
            for notice in self.notices:
                html_parts.append(f"<p>{html.escape(notice)}</p>"); md_parts.append("- " + notice)
            md_parts.append("")

        section("Frozen configuration and reproducibility", [
            "The exact run identity is " + str(self.frozen["identity"]) + ".",
            "The Bayesian model preserves its accepted scalar-ridge Gaussian hierarchy, iid spatial innovation fields and collapsed MCMC. GP paths are conditional on fitted covariance hyperparameters; basis-GRU paths use fixed trained network/covariance parameters. Conventional controls are point-only. The selected configurations below are the actual saved choices, not claims of universal optimality.",
        ])
        safe_configuration = {"history": self.frozen["history"], "horizon": self.frozen["horizon"],
                              "paths": self.frozen["paths"], "model_seed": self.seed,
                              "test_origins": self.frozen["test_origins"], "selected": self.frozen["selected"],
                              "selection_issues": self.frozen.get("selection_issues", [])}
        configuration = json.dumps(safe_configuration, indent=2, sort_keys=True, allow_nan=False)
        html_parts.append("<details><summary>Exact frozen settings</summary><pre>" + html.escape(configuration) + "</pre></details>")
        md_parts += ["```json\n" + configuration + "\n```\n"]
        if self.experiment is not None:
            base_revision = self.experiment.get("base_revision")
            html_parts.append(f"<p>Base repository revision: <code>{html.escape(str(base_revision))}</code>.</p>")
            md_parts.append(f"Base repository revision: `{base_revision}`.\n")
        commands = '''python -m bayes_stesn.applications.ks_runner screen --config "$CONFIG" --cache "$CACHE_ROOT" --output "$RESULTS"
python -m bayes_stesn.applications.ks_runner final --config "$CONFIG" --cache "$CACHE_ROOT" --output "$RESULTS"
python -m bayes_stesn.applications.ks_runner sensitivity --config "$CONFIG" --cache "$CACHE_ROOT" --output "$RESULTS"
python -m bayes_stesn.applications.ks_runner robustness --config "$CONFIG" --cache "$CACHE_ROOT" --output "$RESULTS"
python -m bayes_stesn.applications.ks_runner resume --config "$CONFIG" --cache "$CACHE_ROOT" --output "$RESULTS"
python -m bayes_stesn.applications.ks_runner summarize --config "$CONFIG" --cache "$CACHE_ROOT" --output "$RESULTS"
python -m bayes_stesn.applications.ks_report --results "$RESULTS" --output "$RESULTS/report"'''
        html_parts.append("<pre>" + html.escape(commands) + "</pre>"); md_parts.append("```bash\n" + commands + "\n```\n")
        section("Primary methodological references")
        html_parts.append("<ul>" + "".join(f'<li><a href="{url}">{html.escape(label)}</a></li>' for label, url in REFERENCES) + "</ul>")
        md_parts.append("\n".join(f"- [{label}]({url})" for label, url in REFERENCES) + "\n")

        css = '''body{font:16px/1.55 system-ui,-apple-system,Arial,sans-serif;color:#17202a;background:#fff;margin:0}
main{max-width:1240px;margin:auto;padding:38px 32px 70px}h1{font-size:2rem;line-height:1.2;margin:.4em 0}
h2{font-size:1.35rem;border-top:1px solid #d9dfe5;padding-top:1.3em;margin-top:1.8em}h3{font-size:1.08rem}
.status{padding:16px 20px;background:#eef3f7;border-left:4px solid #24577a}.subtitle{color:#52616d}
.table-wrap{overflow-x:auto;margin:1.2em 0 1.6em}table{border-collapse:collapse;width:100%;font-size:.86rem}
th,td{text-align:left;border-bottom:1px solid #d9dfe5;padding:8px 9px;vertical-align:top}th{background:#f2f5f7}
th:first-child,td:first-child{min-width:125px}figure{margin:2em 0;border:1px solid #e3e7eb;padding:16px}
img,video{display:block;width:100%;height:auto}figcaption{font-size:.9rem;color:#47535c;margin-top:10px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:18px;font-size:.78rem}
code{overflow-wrap:anywhere}a{color:#006ca5}details{margin:1em 0}summary{cursor:pointer;font-weight:600}
@media print{main{padding:0}figure{break-inside:avoid}.table-wrap{overflow:visible}video{display:none}}
'''
        document = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
                    f"<title>{html.escape(title)}</title><style>{css}</style></head><body><main>"
                    f"<h1>{html.escape(title)}</h1><p class=\"subtitle\">Custom literature-grounded experiment · source results generated {html.escape(str(self.summary.get('generated_utc', 'unspecified')))}</p>"
                    f"<div class=\"status\">{html.escape(status)}</div>" + "\n".join(html_parts) + "</main></body></html>")
        for name, data in (("report.html", document), ("report.md", "\n".join(md_parts))):
            destination = self.output / name
            fd, temporary = tempfile.mkstemp(prefix="." + name + "-", suffix=".tmp", dir=self.output)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    stream.write(data); stream.flush(); os.fsync(stream.fileno())
                os.replace(temporary, destination)
            finally:
                Path(temporary).unlink(missing_ok=True)
            self.artifact_paths.add(name)
        return completion


def build_report(results, output, *, animations=True, dpi=200):
    """Generate figures/report from completed saved results; never run a model."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    report = Report(results, output, dpi=dpi)
    report.load_examples()
    report.error_curves(plt)
    report.probability_curves(plt)
    report.heatmaps(plt)
    report.interval_traces(plt)
    report.compute_plot(plt)
    report.animate(plt, enabled=animations)
    completion = report.write_reports()
    artifacts = [{"path": relative, "bytes": (report.output / relative).stat().st_size,
                  "sha256": file_hash(report.output / relative)} for relative in sorted(report.artifact_paths)]
    manifest = {"schema": "ks-report/1", "generated_utc": now(),
                "freeze_identity": report.frozen["identity"], "model_seed": report.seed,
                "completion": completion, "inputs": report.input_receipts,
                "figures": report.figures, "animations": report.movies,
                "notices": report.notices, "artifacts": artifacts,
                "artifact_bytes": sum(item["bytes"] for item in artifacts),
                "model_fit_or_selection_performed": False,
                "case": {"phase": "core", "trajectory": CASE_TRAJECTORY,
                         "first_target": CASE_FIRST_TARGET},
                "curve_uncertainty": "pointwise trajectory-cluster bootstrap, 2000 resamples, seed 20261105",
                "visual_qa_status": "requires_rendered_figure_and_video_inspection"}
    atomic_json(report.output / "report_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-animations", action="store_true")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    if not 100 <= args.dpi <= 400:
        parser.error("--dpi must be between 100 and 400")
    manifest = build_report(args.results, args.output, animations=not args.no_animations, dpi=args.dpi)
    print(json.dumps({"completion": manifest["completion"], "figures": len(manifest["figures"]),
                      "animations": len(manifest["animations"]), "artifact_bytes": manifest["artifact_bytes"],
                      "manifest": str(args.output / "report_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
