"""Receipt-only reporting for the exploratory, validation-only full-field KS study.

No dataset, fit, posterior checkpoint, test outcome, or selection procedure is
opened or executed. Reports are replaceable snapshots under RUN/report; the
numerical receipts remain immutable. Missing work never becomes a zero loss.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .ks_runtime import atomic_json, file_hash, now


TRACK_NAMES = {"A": "Interleaved unseen sites (13 validation targets)",
               "B": "All 64 observed sites", "C": "Contiguous unseen sites (12 validation targets)"}
STAGES = ("screen", "promote", "confirmation")
LEVELS = (.5, .8, .9, .95)
MILESTONES = (1, 5, 10, 25, 50, 100)
SUPPLEMENT_METRICS = ("rmse", "mae", "nrmse", "crps", "normalized_crps", "spatial_energy",
                      "spatial_variogram", "vpt_physical", "vpt_censoring_fraction",
                      *[name+str(int(100*level)) for level in LEVELS for name in ("coverage", "width")])
CSV_FIELDS = ("stage", "track", "candidate", "label", "eligible", "complete",
              "complete_episodes", "expected_episodes", "failed_outcomes", "pending_origins",
              "inference_unqualified_fits", "resource_blocked_fits", "severe_divergent_origins",
              "selection_crps", "selection_mse", "rmse", "early_rmse", "crps", "vpt_physical",
              "successful_episode_rmse", "successful_episode_crps", "mean_recorded_fit_seconds")


def _read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def _atomic_text(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="."+path.name+"-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            out.write(value); out.flush(); os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def _csv(path, rows, fields):
    output = io.StringIO(); writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader(); writer.writerows(rows); _atomic_text(path, output.getvalue())


def _finite(value):
    return isinstance(value, (int, float)) and np.isfinite(value)


def _fmt(value, digits=4):
    return f"{float(value):.{digits}g}" if _finite(value) else "—"


def _table(headers, rows):
    def cell(x): return str(x).replace("|", "/").replace("\n", " ")
    return "\n".join(["| "+" | ".join(map(cell, headers))+" |",
                      "| "+" | ".join("---" for _ in headers)+" |",
                      *["| "+" | ".join(map(cell, row))+" |" for row in rows]])


def cluster_curve(values, *, square_root=False, seed=20261105, resamples=1000):
    """Resample whole trajectories, preserving all leads and origins together.

    Input rows are already equal-origin trajectory means. Squared errors are
    averaged before taking the root, including inside each bootstrap sample.
    No interval is returned from a single completed trajectory.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Finite trajectory-by-horizon values are required")
    if square_root and np.any(x < 0): raise ValueError("Squared errors cannot be negative")
    transform = np.sqrt if square_root else lambda a: a
    center = transform(x.mean(axis=0))
    if len(x) == 1: return center, None, None
    weights = np.random.default_rng(seed).multinomial(len(x), np.full(len(x), 1/len(x)), size=resamples)
    draws = transform((weights@x)/len(x))
    lower, upper = np.quantile(draws, [.025, .975], axis=0)
    return center, lower, upper


def _receipt_index(root, warnings):
    index = {}
    for path in sorted((root/"forecasts").glob("*/*.json")):
        row = _read(path)
        # Precision repetitions live elsewhere and are not scientific replicates.
        if row.get("partition") != "validation" or row.get("phase") != "core":
            warnings.append(f"Excluded non-core/non-validation receipt: {path.name}"); continue
        key = (row.get("track"), row.get("candidate_id"), row.get("trajectory"),
               row.get("seed"), row.get("first_target"), row.get("paths"),
               row.get("forecast_seed"), row.get("parameter_seed"))
        if key in index:
            warnings.append(f"Ambiguous duplicate forecast receipt for {key}; case excluded")
            index[key] = None
        else:
            index[key] = row
    return index


def completed_blocks(index, cfg, track, candidate, schedule, history=None):
    """Use every planned origin or exclude that trajectory from score summaries."""
    blocks = {}
    for trajectory in schedule.get("trajectories", []):
        rows = []
        for first in schedule.get("origins", []):
            key = (track, candidate, trajectory, cfg.get("model_seed"), first,
                   schedule.get("paths"), cfg.get("forecast_seed"), None)
            row = index.get(key)
            if (row is None or row.get("state") != "complete" or not row.get("qualified")
                    or row.get("horizon") != cfg.get("horizon", 100)
                    or (history is not None and row.get("history") != history)):
                break
            rows.append(row)
        if rows and len(rows) == len(schedule.get("origins", [])):
            blocks[trajectory] = rows
    return blocks


def _curve_values(blocks, key, level=None):
    values = []
    for rows in blocks.values():
        origins = []
        for row in rows:
            metric = row["metrics"]
            if level is not None: metric = metric.get("intervals", {}).get(str(level), {})
            value = metric.get(key)
            if value is None: return None
            origins.append(value)
        values.append(np.mean(np.asarray(origins, dtype=np.float64), axis=0))
    return np.asarray(values, dtype=np.float64) if values else None


def _point_episode(rows, key):
    metrics = [row["metrics"] for row in rows]
    if key == "rmse": return float(np.sqrt(np.mean([m["mse_by_horizon"] for m in metrics])))
    if key == "early_rmse": return float(np.sqrt(np.mean([m["mse_by_horizon"][:10] for m in metrics])))
    return float(np.mean([m[key] for m in metrics]))


def _origin_metric(metric, name, horizon=None):
    """Use supplied metrics or an exact reduction; absent fields stay missing."""
    if name.startswith(("coverage", "width")):
        kind = "coverage" if name.startswith("coverage") else "width"
        level = int(name[len(kind):])/100
        value = metric.get("intervals", {}).get(str(level), {})
        values = value.get(kind+"_by_horizon")
        direct = value.get(kind)
    elif name == "vpt_censoring_fraction":
        value = metric.get("vpt_right_censored")
        return None if value is None or horizon is not None else float(value)
    elif name in ("rmse", "nrmse"):
        values = metric.get("mse_by_horizon")
        if values is None: return metric.get(name) if horizon is None else None
        x = np.asarray(values, dtype=np.float64)
        if (x.ndim != 1 or not np.isfinite(x).all() or np.any(x < 0)
                or (horizon is not None and horizon > len(x))): return None
        value = float(np.sqrt(x.mean() if horizon is None else x[horizon-1]))
        if name == "nrmse":
            scale = metric.get("training_scale")
            if not _finite(scale) or scale <= 0: return None
            value /= scale
        return value
    else:
        values, direct = metric.get(name+"_by_horizon"), metric.get(name)
        if name == "normalized_crps" and values is None:
            scale = metric.get("training_scale")
            raw = metric.get("crps_by_horizon")
            if raw is not None and _finite(scale) and scale > 0:
                values = np.asarray(raw, dtype=np.float64)/scale
    if horizon is None and _finite(direct): return float(direct)
    if values is None: return None
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all() or (horizon is not None and horizon > len(x)): return None
    return float(x.mean() if horizon is None else x[horizon-1])


def supplementary_metrics(index, cfg, freeze, collections):
    """Long-form aggregates and fixed-lead values with explicit missingness.

    Every included trajectory must have all scheduled origins. A metric is
    reported only if it exists for every origin in every such trajectory.
    The available-episode mean is separate when some metric fields are absent.
    """
    aggregates, milestones = [], []
    for (stage, track), (rows, schedule) in collections.items():
        for row in rows:
            blocks = completed_blocks(index, cfg, track, row["id"], schedule, freeze.get("history"))
            common = {"stage": stage, "track": track, "candidate": row["id"], "label": row.get("label", row["id"]),
                "complete_trajectories": len(blocks), "expected_trajectories": len(schedule.get("trajectories", [])),
                "origins_per_trajectory": len(schedule.get("origins", [])),
                "full_schedule": bool(blocks) and len(blocks) == len(schedule.get("trajectories", [])),
                "inference_qualified_receipts_only": True}
            for horizon in (None, *MILESTONES):
                names = SUPPLEMENT_METRICS if horizon is None else [n for n in SUPPLEMENT_METRICS if not n.startswith("vpt")]
                for name in names:
                    values = []
                    for records in blocks.values():
                        origins = [_origin_metric(record["metrics"], name, horizon) for record in records]
                        if not all(_finite(x) for x in origins): continue
                        # RMSE over a trajectory's equal-length origins is a
                        # root after averaging squared errors, not a mean root.
                        value = float(np.sqrt(np.mean(np.square(origins)))) if name in ("rmse", "nrmse") else float(np.mean(origins))
                        values.append(value**2 if horizon is not None and name in ("rmse", "nrmse") else value)
                    defined = bool(blocks) and len(values) == len(blocks)
                    center = lo = hi = None
                    if values:
                        a, b, c = cluster_curve(np.asarray(values)[:, None],
                            square_root=horizon is not None and name in ("rmse", "nrmse"))
                        center, lo, hi = float(a[0]), None if b is None else float(b[0]), None if c is None else float(c[0])
                    result = {**common, "metric": name, "metric_complete_trajectories": len(values),
                        "missing_metric_trajectories": len(blocks)-len(values), "metric_defined_for_completed_population": defined,
                        "value": center if defined else None, "ci95_lower": lo if defined else None,
                        "ci95_upper": hi if defined else None, "available_episode_mean": center,
                        "missingness": "none" if defined else "no_complete_trajectories" if not blocks else "metric_absent_or_nonfinite"}
                    if horizon is None: aggregates.append(result)
                    else: milestones.append({**result, "horizon": horizon, "physical_time": .25*horizon})
    return aggregates, milestones


def matched_anchor_comparisons(root, cfg, freeze, index):
    """Paired differences on identical complete confirmation trajectory blocks.

    Never substitute selection anchors, differing histories, or a convenient
    subset of origins. Incomplete schedules retain their original denominator.
    """
    schedule = cfg.get("confirmation", {}); result = []
    history = freeze.get("history")
    for track, selections in freeze.get("selections", {}).items():
        ids = {item["candidate"]["id"] for mode in ("distribution", "dynamics", "point")
               if (item := selections.get(mode))}
        if selections.get("reference_id"): ids.add(selections["reference_id"])
        for candidate in sorted(ids):
            blocks = completed_blocks(index, cfg, track, candidate, schedule, history)
            by_anchor = {}
            for trajectory, rows in blocks.items():
                anchor = _read(root/"anchors"/f"{track}_t{trajectory:02d}.json", {})
                if (anchor.get("schedule") != "confirmation" or anchor.get("partition") != "validation"
                        or anchor.get("track") != track or anchor.get("trajectory") != trajectory
                        or anchor.get("history") != history): continue
                for name, model in anchor.get("models", {}).items():
                    origins = model.get("origins", [])
                    keyed = {row["first_target"]: row for row in origins}
                    if len(keyed) != len(origins) or set(keyed) != set(schedule.get("origins", [])): continue
                    ordered = [keyed[first] for first in schedule["origins"]]
                    if any("metrics" not in r for r in ordered): continue
                    for metric in ("rmse", "early_rmse", "vpt_physical"):
                        difference = _point_episode(rows, metric)-_point_episode(ordered, metric)
                        by_anchor.setdefault((name, metric), []).append((trajectory, difference))
            for (anchor, metric), pairs in sorted(by_anchor.items()):
                values = np.array([v for _, v in pairs])[:, None]
                center, lo, hi = cluster_curve(values)
                result.append({"track": track, "candidate": candidate, "anchor": anchor, "metric": metric,
                    "paired_trajectories": len(pairs), "expected_trajectories": len(schedule.get("trajectories", [])),
                    "origins_per_trajectory": len(schedule.get("origins", [])),
                    "full_schedule": len(pairs) == len(schedule.get("trajectories", [])),
                    "candidate_minus_anchor": float(center[0]),
                    "ci95_lower": None if lo is None else float(lo[0]),
                    "ci95_upper": None if hi is None else float(hi[0]),
                    "trajectory_ids": ",".join(str(t) for t, _ in pairs)})
    return result


def _save_figure(fig, directory, stem):
    files = []
    for extension in ("png", "pdf"):
        path = directory/f"{stem}.{extension}"
        fd, temporary = tempfile.mkstemp(prefix="."+stem+"-", suffix="."+extension, dir=directory)
        os.close(fd)
        try:
            fig.savefig(temporary, dpi=180, bbox_inches="tight")
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
        files.append(path)
    return files


def _plots(root, destination, cfg, freeze, collections, index, warnings):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.append("Matplotlib unavailable: tables are complete; figure rendering is pending")
        return [], []
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    files, horizon_rows = [], []
    for track in cfg.get("tracks", []):
        stages = [s for s in reversed(STAGES) if (s, track) in collections]
        if not stages: continue
        stage = stages[0]; rows, schedule = collections[(stage, track)]
        selections = freeze.get("selections", {}).get(track, {})
        ids = {item["candidate"]["id"] for mode in ("distribution", "dynamics", "point")
               if (item := selections.get(mode))}
        if selections.get("reference_id"): ids.add(selections["reference_id"])
        # Before the freeze, these are explicitly provisional validation curves.
        if not ids: ids = {r["id"] for r in rows if r.get("eligible")}
        if not freeze: ids = set(sorted(ids, key=lambda cid: next(r["loss"] for r in rows if r["id"] == cid))[:3])
        fig, axes = plt.subplots(2, 2, figsize=(10, 7)); plotted = 0
        for row in rows:
            if row["id"] not in ids: continue
            blocks = completed_blocks(index, cfg, track, row["id"], schedule, freeze.get("history"))
            if not blocks: continue
            label = f"{row.get('label', row['id'])} ({len(blocks)}/{len(schedule.get('trajectories', []))})"
            metrics = (("mse_by_horizon", True, None), ("crps_by_horizon", False, None),
                       ("coverage_by_horizon", False, .9), ("width_by_horizon", False, .9))
            for ax, (key, square_root, level) in zip(axes.flat, metrics):
                values = _curve_values(blocks, key, level)
                if values is None: continue
                center, lo, hi = cluster_curve(values, square_root=square_root)
                lead = np.arange(1, len(center)+1); line, = ax.plot(.25*lead, center, label=label)
                if lo is not None: ax.fill_between(.25*lead, lo, hi, color=line.get_color(), alpha=.14, linewidth=0)
                metric = "rmse" if square_root else key.removesuffix("_by_horizon")
                for h in range(len(center)):
                    horizon_rows.append({"stage": stage, "track": track, "candidate": row["id"],
                        "metric": metric, "interval_level": level, "horizon": h+1, "physical_time": .25*(h+1),
                        "complete_trajectories": len(blocks), "expected_trajectories": len(schedule.get("trajectories", [])),
                        "value": float(center[h]), "ci95_lower": None if lo is None else float(lo[h]),
                        "ci95_upper": None if hi is None else float(hi[h])})
            plotted += 1
        if plotted:
            for ax, title in zip(axes.flat, ("RMSE", "CRPS", "90% marginal coverage", "90% interval width")):
                ax.set(title=title, xlabel="Forecast physical time (0.25 × lead)"); ax.grid(alpha=.2)
            axes[1, 0].axhline(.9, color="black", linestyle=":", linewidth=.8)
            axes[1, 0].set_ylim(0, 1)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.06), ncol=1)
            fig.suptitle(f"{TRACK_NAMES.get(track, track)} — {stage}, validation only\n"
                         "95% trajectory-bootstrap bands; completed planned-origin blocks only")
            fig.tight_layout(); files.extend(_save_figure(fig, destination, f"horizons_{track}_{stage}"))
        plt.close(fig)
        cost_rows = [r for r in rows if r.get("eligible") and _finite(r.get("mean_recorded_fit_seconds"))
                     and r["mean_recorded_fit_seconds"] > 0 and _finite(r.get("loss"))]
        if cost_rows:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for prior, marker in (("ridge", "o"), ("rhs", "^")):
                subset = [r for r in cost_rows if r.get("prior_family") == prior]
                if subset: ax.scatter([r["mean_recorded_fit_seconds"] for r in subset], [r["loss"] for r in subset],
                                      marker=marker, label=prior, alpha=.75)
            for row in cost_rows:
                if row["id"] in ids: ax.annotate(row.get("label", row["id"]),
                    (row["mean_recorded_fit_seconds"], row["loss"]), fontsize=7, xytext=(3, 4), textcoords="offset points")
            ax.set(xscale="log", xlabel="Recorded fit seconds per completed trajectory", ylabel="Band-balanced validation CRPS",
                   title=f"Track {track}: qualified {stage} fits; failures remain in tables")
            ax.grid(alpha=.2); ax.legend(); files.extend(_save_figure(fig, destination, f"cost_{track}_{stage}")); plt.close(fig)
    # Render only the frozen schedule's first case, never a score-selected case.
    schedule = cfg.get("confirmation", {})
    if freeze and schedule.get("trajectories") and schedule.get("origins"):
        trajectory, first = schedule["trajectories"][0], schedule["origins"][0]
        for track, selections in freeze.get("selections", {}).items():
            choice = selections.get("distribution")
            if not choice: continue
            candidate = choice["candidate"]["id"]
            path = root/"examples"/f"{track}_{candidate}_t{trajectory:02d}_a{first}.npz"
            if not path.exists(): continue
            with np.load(path, allow_pickle=False) as source:
                arrays = {key: np.array(source[key], copy=True) for key in
                          ("truth", "mean", "lower", "upper", "coordinates", "input_coordinates")}
            truth, mean = arrays["truth"], arrays["mean"]
            if truth.shape != mean.shape or truth.ndim != 2 or not np.isfinite(np.r_[truth.ravel(), mean.ravel()]).all():
                warnings.append(f"Invalid example arrays: {path.name}"); continue
            x = np.mod(np.arctan2(arrays["coordinates"][:, 1], arrays["coordinates"][:, 0]), 2*np.pi)*22/(2*np.pi)
            slots = np.round(x*64/22).astype(int) % 64
            if len(set(slots)) != len(slots):
                warnings.append(f"Example does not map uniquely to the observation grid: {path.name}"); continue
            input_x = np.mod(np.arctan2(arrays["input_coordinates"][:, 1], arrays["input_coordinates"][:, 0]), 2*np.pi)*22/(2*np.pi)
            color = max(float(np.max(np.abs(truth))), float(np.max(np.abs(mean))), 1e-12)
            error = max(float(np.max(np.abs(mean-truth))), 1e-12)
            fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharex=True, sharey=True)
            for ax, values, title, bound in zip(axes, (truth, mean, mean-truth), ("Truth", "Predictive mean", "Signed error"), (color, color, error)):
                field = np.full((len(truth), 64), np.nan); field[:, slots] = values
                image = ax.imshow(field, origin="lower", aspect="auto", extent=(-22/128, 22-22/128, 0, .25*len(truth)),
                                  cmap="RdBu_r", vmin=-bound, vmax=bound, interpolation="none")
                ax.scatter(input_x, np.full(len(input_x), .015), transform=ax.get_xaxis_transform(), marker="|", color="black", s=18)
                ax.set(title=title, xlabel="Periodic position x"); fig.colorbar(image, ax=ax, shrink=.8)
            axes[0].set_ylabel("Forecast physical time")
            fig.suptitle(f"Track {track}, trajectory {trajectory}, first target {first}; frozen distribution selection\n"
                         "Only official validation-target columns are colored; black ticks mark input sites")
            fig.tight_layout(); files.extend(_save_figure(fig, destination, f"example_{track}_t{trajectory:02d}_a{first}")); plt.close(fig)
            chosen = sorted(set(int(np.argmin(np.abs(np.angle(np.exp(2j*np.pi*(x-location)/22)))))
                                for location in (0., 22/3, 44/3)))
            fig, axes = plt.subplots(len(chosen), 1, figsize=(8, 2.3*len(chosen)), squeeze=False)
            times = .25*np.arange(1, len(truth)+1)
            for ax, coordinate in zip(axes.flat, chosen):
                ax.fill_between(times, arrays["lower"][:, coordinate], arrays["upper"][:, coordinate], alpha=.22, label="90% predictive interval")
                ax.plot(times, truth[:, coordinate], color="black", label="Truth")
                ax.plot(times, mean[:, coordinate], label="Predictive mean")
                ax.set(title=f"Validation site nearest fixed display coordinate: x={x[coordinate]:.4g}", xlabel="Forecast physical time", ylabel="Physical field u")
            axes[0, 0].legend(fontsize=8); fig.tight_layout()
            files.extend(_save_figure(fig, destination, f"intervals_{track}_t{trajectory:02d}_a{first}")); plt.close(fig)
    return files, horizon_rows


def report(output, *, make_figures=True, animation=False):
    """Write an honest partial or completed snapshot; return its compact manifest."""
    root = Path(output).resolve(); destination = root/"report"; destination.mkdir(parents=True, exist_ok=True)
    experiment = _read(root/"experiment.json", {})
    cfg = experiment.get("config", {}); freeze = _read(root/"freeze.json", {})
    status = _read(root/"status.json", {}); coordinator = _read(root/"coordinator.json", {})
    registry = _read(root/"candidate_registry.json", {}); warnings = []
    if animation: warnings.append("Animation omitted: this compact reporter renders static evidence only")
    index = _receipt_index(root, warnings)
    fit_receipts = [_read(p) for p in sorted((root/"fits").glob("*/receipt.json"))]
    collections, flat, coverage = {}, [], []
    for stage in STAGES:
        for track in cfg.get("tracks", registry.get("tracks", {}).keys()):
            path = root/"confirmation"/f"{track}.json" if stage == "confirmation" else root/"selection"/f"{stage}_{track}.json"
            packet = _read(path)
            if packet is None: continue
            rows = packet.get("rows", []); schedule = packet.get("schedule", cfg.get(stage, {}))
            coverage.append({"stage": stage, "track": track, "reported_candidates": len(rows),
                "eligible_candidates": sum(bool(r.get("eligible")) for r in rows),
                "complete_candidates": sum(bool(r.get("complete")) for r in rows),
                "declared_candidates": len(registry.get("tracks", {}).get(track, {}).get("candidates", [])),
                "expected_trajectories": len(schedule.get("trajectories", [])),
                "origins_per_trajectory": len(schedule.get("origins", []))})
            for row in rows:
                fits = [f for f in fit_receipts if f.get("track") == track and f.get("candidate_id") == row["id"]
                        and f.get("phase") == "core" and f.get("seed") == cfg.get("model_seed")
                        and f.get("trajectory") in [e["trajectory"] for e in row.get("episodes", [])]]
                timings = [f["fit_seconds"] for f in fits if _finite(f.get("fit_seconds"))]
                row["mean_recorded_fit_seconds"] = float(np.mean(timings)) if timings else None
                record = {key: row.get(key) for key in CSV_FIELDS}
                record.update(stage=stage, track=track, candidate=row["id"],
                              selection_crps=row.get("loss"), selection_mse=row.get("point_loss"))
                for metric in ("rmse", "crps"):
                    values = [e[metric] for e in row.get("episodes", []) if _finite(e.get(metric))]
                    record["successful_episode_"+metric] = float(np.mean(values)) if values else None
                flat.append(record)
            collections[(stage, track)] = rows, schedule
    comparisons = matched_anchor_comparisons(root, cfg, freeze, index)
    supplement, milestones = supplementary_metrics(index, cfg, freeze, collections)
    _csv(destination/"candidate_metrics.csv", flat, CSV_FIELDS)
    comparison_fields = ("track", "candidate", "anchor", "metric", "paired_trajectories", "expected_trajectories",
                         "origins_per_trajectory", "full_schedule", "candidate_minus_anchor", "ci95_lower", "ci95_upper", "trajectory_ids")
    _csv(destination/"paired_anchor_comparisons.csv", comparisons, comparison_fields)
    files = [destination/"candidate_metrics.csv", destination/"paired_anchor_comparisons.csv"]
    supplement_fields = ("stage", "track", "candidate", "label", "metric", "value", "ci95_lower", "ci95_upper",
        "complete_trajectories", "expected_trajectories", "origins_per_trajectory", "full_schedule",
        "metric_complete_trajectories", "missing_metric_trajectories", "metric_defined_for_completed_population",
        "available_episode_mean", "missingness", "inference_qualified_receipts_only")
    _csv(destination/"supplementary_metrics.csv", supplement, supplement_fields)
    _csv(destination/"milestone_metrics.csv", milestones, (*supplement_fields, "horizon", "physical_time"))
    files.extend((destination/"supplementary_metrics.csv", destination/"milestone_metrics.csv"))
    if make_figures:
        figures, horizons = _plots(root, destination, cfg, freeze, collections, index, warnings)
        files.extend(figures)
        _csv(destination/"horizon_metrics.csv", horizons, ("stage", "track", "candidate", "metric", "interval_level",
            "horizon", "physical_time", "complete_trajectories", "expected_trajectories", "value", "ci95_lower", "ci95_upper"))
        files.append(destination/"horizon_metrics.csv")
    terminal = [_read(p) for p in (root/"terminal").glob("*.json")]
    peak_rss = max((r.get("resource", {}).get("process_peak_rss_bytes", 0) for r in terminal), default=0)
    all_confirmation = bool(freeze) and all(("confirmation", track) in collections
        and all(r.get("complete") for r in collections[("confirmation", track)][0])
        and bool(collections[("confirmation", track)][0]) for track in cfg.get("tracks", []))
    complete = bool(all_confirmation and coordinator.get("command") == "all"
        and coordinator.get("state") == "complete" and status.get("stages", {}).get("sensitivity")
        and not any(status.get("states", {}).get(s, 0) for s in ("pending", "running")))
    heading = "Configured campaign finished; inspect qualification and failures" if complete else "Partial campaign snapshot — work remains"
    text = ["# Full-field Bayesian KS experiment", "", f"**{heading}.** Snapshot: {now()}.", "",
        "This is an exploratory, validation-only extension after historical test inspection. "
        "The 14-trajectory confirmation schedule is a separately reserved validation subset, not an untouched test holdout. "
        "No new test results are scored or imported here. It is a custom, literature-grounded experiment, not a published leaderboard reproduction.", "",
        "## Model and information access", "",
        "Every permitted observed site at every declared lag drives a sparse leaky ESN: 39 sites in A, 64 in B, and40 in C. "
        "There is no PCA, POD, response projection, or hidden-site input. Conditioning-prefix centering/scaling is frozen. "
        "The lagged raw fields, reservoir states, and selected squared states form the readout, with an explicit all/latest/none direct-field policy. "
        "All configured lag fields still drive the reservoir when the direct readout uses only the latest field. "
        "A spatial coordinate basis remains in the output model; full-field inputs do not mean removal of that coordinate decoder.", "",
        "The conditional Gaussian likelihood retains the spatial-basis readout and a valid Matérn covariance on the unit-circle embedding, "
        "with independent time innovations. Ridge and finite-slab regularized-horseshoe priors are distinct declared variants. "
        "State and increment targets are separate candidates; increments add native-site persistence and a declared periodic interpolation offset at new sites. "
        "A static parameter draw is held for each autonomous path. Point metrics average complete paths. "
        "Four-chain R-hat≤1.01 and ESS≥400 gates remain required for scientific probability scores. Resource pilots do not establish convergence.", "",
        "## Progress and denominators", "",
        f"Coordinator state: **{coordinator.get('state', 'not recorded')}**; command: `{coordinator.get('command', 'not recorded')}`. "
        f"Selection freeze present: **{bool(freeze)}**. Registered job states: `{json.dumps(status.get('states', {}), sort_keys=True)}`.", "",
        "Completion of a currently registered stage does not establish completion of the study. A missing, failed, unqualified, "
        "resource-blocked, or pending result is never converted to a numerical score. Successful-only summaries retain their original denominator. "
        "Any budget-limited selection is best among the configurations actually completed and eligible, with the pending roster retained.", ""]
    if coverage:
        text.append(_table(("Track", "Stage", "Reported / declared candidates", "Complete", "Eligible", "Trajectories × origins"),
            [(r["track"], r["stage"], f"{r['reported_candidates']} / {r['declared_candidates']}", r["complete_candidates"], r["eligible_candidates"],
              f"{r['expected_trajectories']} × {r['origins_per_trajectory']}") for r in coverage])); text.append("")
    else: text.extend(["No screening result table has been written yet. Queued jobs imply no measured performance.", ""])
    text.extend(["## Selected configurations", ""])
    selected_rows = []
    for track, selection in freeze.get("selections", {}).items():
        for role in ("distribution", "dynamics", "point"):
            if not selection.get(role): continue
            row = selection[role]["candidate"]; config = row.get("configuration", row.get("config", {})); f = config.get("features", {})
            selected_rows.append((track, role, row.get("label", row["id"]), config.get("prior", {}).get("kind"), config.get("target"),
                f.get("units"), f.get("lags"), f.get("leak_patterns"), f.get("spectral_radius"), f.get("raw_readout_lags")))
    text.append(_table(("Track", "Role", "Candidate", "Prior", "Target", "Units", "Lags", "Leak", "Radius", "Direct inputs"), selected_rows)
                if selected_rows else "Selection is not frozen. Provisional rankings do not constitute selected models.")
    text.extend(["", "## Available performance", "",
        "RMSE and CRPS in the table are equally weighted trajectory summaries from the runner. "
        "Horizon figures instead take the square root after averaging trajectory MSE at each lead. "
        "The complete CSV contains every reported candidate; the display below contains up to10 eligible rows per track/stage, followed by failure counts.", ""])
    for (stage, track), (rows, _) in collections.items():
        shown = sorted((r for r in rows if r.get("eligible")), key=lambda r: r["loss"])[:10]
        text.extend([f"### Track {track}: {stage}", ""])
        if shown:
            text.append(_table(("Candidate", "Complete episodes", "RMSE", "Early RMSE", "CRPS", "Balanced CRPS", "VPT (time)", "Fit seconds"),
                [(r.get("label", r["id"]), f"{r.get('complete_episodes', 0)}/{r.get('expected_episodes', 0)}",
                  *[_fmt(r.get(k)) for k in ("rmse", "early_rmse", "crps", "loss", "vpt_physical", "mean_recorded_fit_seconds")]) for r in shown]))
        else: text.append("No candidate currently has a complete, inference-qualified, eligible score on this stage.")
        text.extend(["", "Counts across candidate outcomes: "+", ".join(f"{key}={sum(int(r.get(key, 0) or 0) for r in rows)}"
            for key in ("failed_outcomes", "pending_origins", "inference_unqualified_fits", "resource_blocked_fits", "severe_divergent_origins"))+".", ""])
    text.extend(["## Supplementary point, interval, and spatial scores", "",
        "The complete supplementary_metrics.csv includes MAE, training-scale-normalized RMSE, CRPS, normalized CRPS, "
        "50/80/90/95% coverage and interval width, spatial energy and variogram scores, and VPT censoring when present. "
        "Every included trajectory has all planned origins. Missing metric fields are explicitly flagged; they are never imputed. "
        "milestone_metrics.csv gives h=1,5,10,25,50,100 and physical times0.25,1.25,2.5,6.25,12.5,25, with the same denominators and cluster intervals. "
        "The tables below display the latest reported stage and frozen selections, or at most3 eligible provisional candidates if selection is not frozen.", ""])
    for track in cfg.get("tracks", []):
        stages = [s for s in reversed(STAGES) if (s, track) in collections]
        if not stages: continue
        stage = stages[0]; rows = collections[(stage, track)][0]
        chosen = freeze.get("selections", {}).get(track, {})
        ids = {item["candidate"]["id"] for role in ("distribution", "dynamics", "point") if (item := chosen.get(role))}
        if chosen.get("reference_id"): ids.add(chosen["reference_id"])
        if not ids: ids = {r["id"] for r in sorted((r for r in rows if r.get("eligible")), key=lambda r: r["loss"])[:3]}
        points, intervals = [], []
        for row in rows:
            if row["id"] not in ids: continue
            metrics = {r["metric"]: r for r in supplement if r["stage"] == stage and r["track"] == track and r["candidate"] == row["id"]}
            if not metrics: continue
            count = next(iter(metrics.values()))
            denominator = f"{count['complete_trajectories']}/{count['expected_trajectories']}"
            label = row.get("label", row["id"])
            points.append((label, denominator, *[_fmt(metrics.get(key, {}).get("value"))
                for key in ("mae", "nrmse", "spatial_energy", "spatial_variogram")]))
            for level in (50, 80, 90, 95):
                coverage_row, width_row = metrics.get(f"coverage{level}", {}), metrics.get(f"width{level}", {})
                intervals.append((label, f"{level}%", denominator, _fmt(coverage_row.get("value")), _fmt(width_row.get("value")),
                    "available" if coverage_row.get("metric_defined_for_completed_population") and width_row.get("metric_defined_for_completed_population") else "missing; see CSV"))
        if points:
            text.extend([f"### Track {track}: {stage} supplement", "",
                _table(("Candidate", "Complete episodes", "MAE", "Normalized RMSE", "Spatial energy", "Spatial variogram"), points), "",
                _table(("Candidate", "Interval", "Complete episodes", "Coverage", "Width", "Data status"), intervals), ""])
    text.extend(["## Matched anchor comparisons", "",
        "Comparisons below pair each trajectory and every planned origin, using exactly matching histories. "
        "They compare the Bayesian candidate against training mean, persistence, or full-field linear VAR only when their confirmation receipts exist. "
        "Point-only anchors have no invented CRPS or intervals. Differences are candidate minus anchor: negative RMSE and positive VPT favor the candidate. "
        "Intervals resample whole trajectories, conditional on frozen selection; origins and grid cells are not independent replicates.", ""])
    if comparisons:
        text.append(_table(("Track", "Candidate", "Anchor", "Metric", "Paired / expected", "Difference", "95% interval"),
            [(r["track"], r["candidate"], r["anchor"], r["metric"], f"{r['paired_trajectories']}/{r['expected_trajectories']}",
              _fmt(r["candidate_minus_anchor"]), f"[{_fmt(r['ci95_lower'])}, {_fmt(r['ci95_upper'])}]") for r in comparisons]))
    else: text.append("No exact matched confirmation comparison is available yet.")
    text.extend(["", "Historical frozen benchmark metrics are a separate experiment. They are not merged with these validation rankings or used as matched comparisons.", "",
        "## Resources and interpretation", "",
        f"Recorded worker time: {_fmt(status.get('worker_seconds'))} seconds; recorded search time: {_fmt(status.get('search_worker_seconds'))} seconds. "
        f"Largest terminal worker peak RSS: {_fmt(peak_rss/1024**2 if peak_rss else None)} MiB. "
        f"Latest free disk: {_fmt(status.get('resources', {}).get('disk_free_bytes', 0)/1024**3)} GiB.", "",
        "Fit-time plot values are completed-fit receipt times, and can exclude earlier interrupted attempts. "
        "The attempt ledger is authoritative for total worker expenditure. Plotted bands are trajectory-bootstrap uncertainty, not posterior bands or Monte Carlo convergence diagnostics. "
        "A 0.4 training-SD RMSE crossing defines valid prediction time, right censored at100 steps (25 physical units); this convention is not a Lyapunov-time conversion. "
        "Strict-track heatmaps color only official validation target sites and mark observed input sites. Uncolored sites are not scored.", "",
        "## Reproduce this report", "", "```bash", "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \\",
        "  python tools/report_ks_fullfield.py --output /path/to/fullfield/run", "```", "",
        "This command only reads existing JSON receipts and predeclared example arrays. It does not fit, retune, forecast, open raw data, or score tests.", ""])
    if warnings: text.extend(["## Reporting notes", "", *["- "+w for w in warnings], ""])
    _atomic_text(destination/"report.md", "\n".join(text)); files.append(destination/"report.md")
    manifest = {"schema": "ks-fullfield-report/1", "created_utc": now(), "experiment_identity": experiment.get("identity"),
        "numerical_receipts_modified": False, "test_scoring": False, "validation_after_historical_test_inspection": True,
        "complete_campaign": complete, "all_confirmation_rows_complete": all_confirmation,
        "coordinator_state": coordinator.get("state"), "coverage": coverage, "candidate_rows": len(flat),
        "matched_comparison_rows": len(comparisons), "warnings": warnings,
        "supplementary_metric_rows": len(supplement), "milestone_metric_rows": len(milestones),
        "report_source_sha256": file_hash(Path(__file__)),
        "artifacts": [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": file_hash(p)} for p in files]}
    atomic_json(destination/"manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "--run", dest="output", required=True, help="Existing campaign run directory")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    result = report(args.output, make_figures=not args.no_figures)
    print(json.dumps({key: result[key] for key in ("complete_campaign", "candidate_rows", "matched_comparison_rows", "warnings")}, indent=2))


if __name__ == "__main__": main()
