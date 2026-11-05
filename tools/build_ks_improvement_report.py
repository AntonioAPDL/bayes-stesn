#!/usr/bin/env python3
"""Build a receipt-bound, validation-only KS improvement report; never fit models.

Example (from the repository):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
    python tools/build_ks_improvement_report.py --results /data/ks_improvement_20261105/run_v2 \
    --output docs/applications/improvement_results

This is a separate reporting program. It does not import the campaign driver,
open numerical data, load posterior checkpoints, select a model, or write into
the results root. All scientific comparisons use the frozen validation schedule.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from bayes_stesn.io import content_hash
from bayes_stesn.applications.ks_metrics import cluster_summary


SCHEMA = "ks-improvement-report/1"
TERMINAL = {"complete", "failed", "inference-unqualified"}
CONFIRMATION = [1, 2, 3, 6, 7, 8, 11, 12, 13, 14, 16, 17, 18, 19]
SELECTION = [0, 5, 10, 15, 4, 9]
ORIGINS = list(range(6000, 8000, 100))
TRACK_NAMES = {"A": "Interleaved unseen sites", "B": "All observed sites",
               "C": "Contiguous unseen sites"}
METRICS = ("rmse", "nrmse", "mae", "crps", "normalized_crps", "coverage50", "width50",
           "coverage80", "width80", "coverage90", "width90", "coverage95", "width95",
           "spatial_energy", "spatial_variogram", "vpt_physical", "vpt_censoring_fraction")
COUNTS = ("finite_successful_origins", "failed_rollout_origins", "unqualified_blocked_origins",
          "failed_job_blocked_origins", "pending_origins")
REFERENCES = [
    ("Pathak et al. (2018), reservoir prediction of spatiotemporal chaos",
     "https://doi.org/10.1103/PhysRevLett.120.024102"),
    ("Vlachas et al. (2020), recurrent and reservoir forecasting of chaotic systems",
     "https://arxiv.org/abs/1910.05266"),
    ("Vehtari et al. (2021), rank-normalized R-hat and effective sample sizes",
     "https://arxiv.org/abs/1903.08008"),
    ("Gneiting and Raftery (2007), proper scoring rules",
     "https://doi.org/10.1198/016214506000001437"),
    ("Li et al. (2018), Hyperband, motivation for staged resource allocation",
     "https://www.jmlr.org/papers/v18/16-558.html"),
]


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def write_csv(path, rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()})
    atomic_bytes(path, stream.getvalue().encode())


def finite_vector(value, name, length=100):
    result = np.asarray(value, np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"Invalid finite horizon vector: {name}")
    return result


def episode_summary(records, trajectory):
    """Pool squared loss within a trajectory, then take its square root.

    Every record represents the same horizon and target-site panel. Thus the
    mean of origin-level spatial MSE vectors is the pooled cell-level MSE.
    This is deliberately different from averaging origin-level RMSE values.
    """
    if not records or any(r["state"] != "complete" for r in records):
        raise ValueError("A complete episode needs every scheduled finite origin")
    metrics = [r["metrics"] for r in records]
    scales = np.asarray([m["training_scale"] for m in metrics], np.float64)
    if not np.isfinite(scales).all() or (scales <= 0).any() or not np.all(scales == scales[0]):
        raise ValueError("An episode must use one positive permitted training scale")
    mse = np.mean([finite_vector(m["mse_by_horizon"], "mse") for m in metrics], axis=0)
    if (mse < 0).any():
        raise ValueError("Negative squared error")
    curves = {"rmse": np.sqrt(mse).tolist(),
              "mae": np.mean([finite_vector(m["mae_by_horizon"], "mae") for m in metrics], axis=0).tolist()}
    result = {"trajectory": int(trajectory), "origins": len(records), "training_scale": float(scales[0]),
              "mse": float(mse.mean()), "rmse": float(np.sqrt(mse.mean())),
              "nrmse": float(np.sqrt(mse.mean()) / scales[0]), "mae": float(np.mean(curves["mae"])),
              "vpt_physical": float(np.mean([m["vpt_physical"] for m in metrics])),
              "vpt_censoring_fraction": float(np.mean([m["vpt_right_censored"] for m in metrics])),
              "finite_divergent_origins": sum(bool(m["diverged"]) for m in metrics)}
    has_crps = ["crps_by_horizon" in m for m in metrics]
    if any(has_crps) and not all(has_crps):
        raise ValueError("Mixed probabilistic and point-only records in one episode")
    if all(has_crps):
        curves["crps"] = np.mean([finite_vector(m["crps_by_horizon"], "crps") for m in metrics], axis=0).tolist()
        result["crps"] = float(np.mean(curves["crps"]))
        result["normalized_crps"] = result["crps"] / scales[0]
        for level in (.5, .8, .9, .95):
            for kind in ("coverage", "width"):
                name = f"{kind}{int(level * 100)}"
                curves[name] = np.mean([finite_vector(m["intervals"][str(level)][kind + "_by_horizon"], name)
                                        for m in metrics], axis=0).tolist()
                result[name] = float(np.mean(curves[name]))
        for name in ("spatial_energy", "spatial_variogram"):
            if all(name + "_by_horizon" in m for m in metrics):
                curves[name] = np.mean([finite_vector(m[name + "_by_horizon"], name) for m in metrics], axis=0).tolist()
                result[name] = float(np.mean(curves[name]))
    result["curves"] = curves
    return result


def horizon_summary(episodes, metric, *, seed=20261105, resamples=2000):
    values = [e["curves"][metric] for e in episodes if metric in e["curves"]]
    if not values:
        return {"n": 0, "mean": None, "ci95": None}
    array = np.asarray(values, np.float64)
    mean = array.mean(axis=0)
    if len(array) > 1:
        # 2,000 x 14 x 100 remains small. The same cluster is kept across leads.
        indices = np.random.default_rng(seed).integers(len(array), size=(resamples, len(array)))
        draws = array[indices].mean(axis=1)
        interval = np.quantile(draws, [.025, .975], axis=0).tolist()
    else:
        interval = None
    return {"n": len(array), "mean": mean.tolist(), "ci95": interval,
            "uncertainty": "trajectory bootstrap; pointwise 95% intervals, not simultaneous bands"}


def event_frequency(per_trajectory, key, scheduled_origins):
    observed = sum(r[key] for r in per_trajectory)
    unresolved = sum(r["pending_origins"] + r["failed_job_blocked_origins"] +
                     r["unqualified_blocked_origins"] for r in per_trajectory)
    out = {"events": observed, "denominator": scheduled_origins,
           "identified_bounds": [observed / scheduled_origins, (observed + unresolved) / scheduled_origins],
           "mean": None, "ci95": None}
    if unresolved == 0:
        out.update(cluster_summary([r[key] / r["scheduled_origins"] for r in per_trajectory]))
        out["interpretation"] = "empirical trajectory-cluster interval; zero observed events do not establish zero future risk"
    else:
        out["interpretation"] = "unobserved outcomes remain unidentified; no zero-filled bootstrap"
    return out


def summarize_cohort(outcomes, trajectories, origins, fit_states=None, terminal_states=None):
    """Pure aggregation used by the receipt reader and focused known-answer tests.

    outcomes[(trajectory, first_target)] is a validated immutable forecast receipt.
    fit_states values are qualified/unqualified/missing; terminal_states describe
    the scheduled job. Missing outcomes of a failed job are blocked, not counted
    as observed nonfinite rollouts or as still-pending work.
    """
    fit_states, terminal_states = fit_states or {}, terminal_states or {}
    episodes, per_trajectory = [], []
    for trajectory in trajectories:
        row = {"trajectory": int(trajectory), "scheduled_origins": len(origins),
               "fit_state": fit_states.get(trajectory, "missing"),
               "job_state": terminal_states.get(trajectory, "pending"),
               **{k: 0 for k in COUNTS}, "finite_divergent_origins": 0}
        finite = []
        for first in origins:
            record = outcomes.get((trajectory, first))
            if record is not None:
                if row["fit_state"] != "qualified":
                    raise ValueError("Forecast receipt without a qualified fitted-model receipt")
                if record["state"] == "complete":
                    row["finite_successful_origins"] += 1
                    row["finite_divergent_origins"] += bool(record["metrics"]["diverged"])
                    finite.append(record)
                elif record["state"] == "failed":
                    row["failed_rollout_origins"] += 1
                else:
                    raise ValueError("Unknown immutable forecast state")
            elif row["fit_state"] == "unqualified" or row["job_state"] == "inference-unqualified":
                row["unqualified_blocked_origins"] += 1
            elif row["job_state"] == "failed":
                row["failed_job_blocked_origins"] += 1
            else:
                row["pending_origins"] += 1
        if sum(row[k] for k in COUNTS) != len(origins):
            raise AssertionError("Incomplete origin census")
        row["observed_origins"] = row["finite_successful_origins"] + row["failed_rollout_origins"]
        row["failure_or_divergence_origins"] = row["failed_rollout_origins"] + row["finite_divergent_origins"]
        row["complete_for_accuracy"] = len(finite) == len(origins)
        if row["complete_for_accuracy"]:
            episodes.append(episode_summary(finite, trajectory))
        per_trajectory.append(row)
    counts = {k: sum(r[k] for r in per_trajectory) for k in
              (*COUNTS, "observed_origins", "finite_divergent_origins", "failure_or_divergence_origins")}
    scheduled = len(trajectories) * len(origins)
    complete_ids = [e["trajectory"] for e in episodes]
    summaries = {m: cluster_summary([e[m] for e in episodes if m in e]) for m in METRICS}
    curves = {m: horizon_summary(episodes, m) for m in
              ("rmse", "mae", "crps", "coverage50", "width50", "coverage80", "width80",
               "coverage90", "width90", "coverage95", "width95", "spatial_energy", "spatial_variogram")}
    return {"scheduled_trajectories": len(trajectories), "scheduled_origins": scheduled,
            "trajectories": list(trajectories), "origins": list(origins), **counts,
            "complete_trajectories": len(episodes), "complete_trajectory_ids": complete_ids,
            "score_population_origins": len(episodes) * len(origins),
            "finite_origins_outside_score_population": counts["finite_successful_origins"] - len(episodes) * len(origins),
            "full_schedule_point_score_defined": counts["finite_successful_origins"] == scheduled,
            "all_scheduled_jobs_terminal": all(terminal_states.get(t) in TERMINAL for t in trajectories),
            "metrics": summaries,
            "complete_case_pooled_rmse": float(np.sqrt(np.mean([e["mse"] for e in episodes]))) if episodes else None,
            "failure_frequency": event_frequency(per_trajectory, "failed_rollout_origins", scheduled),
            "finite_divergence_frequency": event_frequency(per_trajectory, "finite_divergent_origins", scheduled),
            "failure_or_divergence_frequency": event_frequency(per_trajectory, "failure_or_divergence_origins", scheduled),
            "per_trajectory": per_trajectory, "episodes": episodes, "curves": curves}


def paired_comparisons(left, right, *, metrics=("rmse", "nrmse", "mae", "crps", "coverage90", "width90")):
    """Left minus right, retaining only the explicit intersection of complete episodes."""
    a = {e["trajectory"]: e for e in left["episodes"]}
    b = {e["trajectory"]: e for e in right["episodes"]}
    result = []
    for metric in metrics:
        ids = sorted(i for i in a.keys() & b.keys() if metric in a[i] and metric in b[i])
        result.append({"metric": metric, "left": left["candidate_id"], "right": right["candidate_id"],
                       "left_seed": left["seed"], "right_seed": right["seed"],
                       "paired_trajectories": ids, "scheduled_trajectories": left["trajectories"],
                       "difference": "left minus right", "conditional_on_complete_pairs": True,
                       **cluster_summary([a[i][metric] - b[i][metric] for i in ids])})
    return result


def diagnostic_extrema(diagnostics):
    gates = diagnostics.get("gates", [])
    latest = gates[-1] if gates else {}
    blocks = {}
    for name, block in latest.items():
        if not isinstance(block, dict) or not isinstance(block.get("parameters"), dict):
            continue
        values = list(block["parameters"].values())
        def numeric(key):
            return [float(v[key]) for v in values if v.get(key) is not None and np.isfinite(v[key])]
        rhat, bulk, tail = numeric("rhat"), numeric("ess_bulk"), numeric("ess_tail")
        blocks[name] = {"monitored": len(values), "max_rhat": max(rhat, default=None),
                        "min_bulk_ess": min(bulk, default=None), "min_tail_ess": min(tail, default=None),
                        "failed_quantities": sum(not v.get("passed", False) for v in values),
                        "chains": block.get("chains"), "diagnostic_draws": block.get("draws")}
    return {"qualified": bool(diagnostics.get("qualified", False)), "blocks": blocks,
            "continuations": diagnostics.get("continuations"),
            "conditioning_rows": diagnostics.get("conditioning_rows"),
            "feature_transform_rows": diagnostics.get("feature_transform_rows"),
            "scalar_hyperparameter_summaries": latest.get("hyperparameters", {}).get("parameters", {})}


class ReceiptReader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = {}
        self.experiment = self.read("experiment.json")
        self.cfg = self.experiment["config"]
        self.registry = self.cfg["registry"]
        identity = content_hash({"config": self.cfg, "source": self.experiment["source"],
                                 "cache": self.experiment["cache_identity"]})
        if identity != self.experiment["identity"]:
            raise ValueError("Experiment identity mismatch")
        self.history = self.read("history_policy.json")
        self.freeze = self.read("freeze.json") if (self.root / "freeze.json").exists() else None
        if self.freeze:
            f = self.freeze
            if (f["identity"] != content_hash({k: v for k, v in f.items() if k != "identity"})
                    or f["experiment_identity"] != identity or f["test_access"] is not False
                    or f["confirmation_trajectories"] != CONFIRMATION
                    or f["confirmation_origins"] != ORIGINS or f["paths"] != 512
                    or f["history"] != self.history["history"]):
                raise ValueError("Incompatible confirmation freeze")
        self.jobs = {}
        self.terminals = {}
        self.fits = {}
        self.candidates = {(t, c["id"]): c for t in self.cfg["tracks"]
                           for c in self.registry["tracks"][t]["candidates"]}
        for path in sorted((self.root / "jobs").glob("*.json")):
            job = self.read(path)
            if (content_hash({k: v for k, v in job.items() if k != "job_id"})[:24] != job["job_id"]
                    or path.stem != job["job_id"] or job["experiment_identity"] != identity
                    or job["source"] != self.experiment["source"] or job["partition"] != "validation"
                    or job["candidate"] != self.candidates[(job["track"], job["candidate"]["id"])]):
                raise ValueError("Incompatible job receipt")
            self.jobs[job["job_id"]] = job
            terminal_path = self.root / "terminal" / (path.stem + ".json")
            if terminal_path.exists():
                terminal = self.read(terminal_path)
                if (terminal["job_identity"] != content_hash({k: v for k, v in job.items() if k != "job_id"})
                        or terminal["state"] not in TERMINAL or terminal["job_id"] != job["job_id"]):
                    raise ValueError("Incompatible immutable terminal receipt")
                self.terminals[job["job_id"]] = terminal

    def read(self, name):
        path = Path(name)
        if not path.is_absolute():
            path = self.root / path
        payload = path.read_bytes()
        relative = str(path.relative_to(self.root))
        self.manifest[relative] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        return json.loads(payload)

    def bind_file(self, path):
        path = Path(path)
        self.manifest[str(path.relative_to(self.root))] = {"bytes": path.stat().st_size, "sha256": sha256(path)}

    def key(self, track, candidate_id, trajectory, seed):
        candidate = self.candidates[(track, candidate_id)]
        return content_hash({"track": track, "effective_config": candidate["config"],
                             "trajectory": int(trajectory), "seed": int(seed)})[:32]

    def fit(self, track, candidate_id, trajectory, seed):
        key = self.key(track, candidate_id, trajectory, seed)
        if key in self.fits:
            return self.fits[key]
        path = self.root / "fits" / key / "receipt.json"
        if not path.exists():
            return None
        raw = self.read(path)
        if (raw["fit_key"] != key or raw["track"] != track or raw["candidate_id"] != candidate_id
                or raw["trajectory"] != trajectory or raw["seed"] != seed):
            raise ValueError("Incompatible fit receipt")
        compact = {k: raw[k] for k in ("fit_key", "native_fit_identity", "fit_sha256", "fit_bytes",
                                      "fit_seconds", "config", "track", "trajectory", "seed", "candidate_id")}
        compact["diagnostics"] = diagnostic_extrema(raw["diagnostics"])
        compact["peak_fit_process_rss_bytes"] = raw.get("resource", {}).get("process_peak_rss_bytes")
        compact["training_identity_sha256"] = content_hash(raw["training_identity"])
        self.fits[key] = compact
        return compact

    def matching_job(self, stage, track, candidate_id, trajectory, seed, origins, paths):
        matches = [j for j in self.jobs.values() if j["stage"] == stage and j["track"] == track
                   and j["candidate"]["id"] == candidate_id and j["trajectory"] == trajectory
                   and j["seed"] == seed and j["origins"] == list(origins) and j["paths"] == paths]
        if len(matches) > 1:
            raise ValueError("Ambiguous scheduled jobs")
        return matches[0] if matches else None

    def cohort(self, track, candidate_id, trajectories, origins, paths, seed, stage, roles):
        outcomes, fit_states, terminal_states, fits = {}, {}, {}, []
        forecast_seconds = 0.
        for trajectory in trajectories:
            key = self.key(track, candidate_id, trajectory, seed)
            fit = self.fit(track, candidate_id, trajectory, seed)
            fit_states[trajectory] = "missing" if fit is None else ("qualified" if fit["diagnostics"]["qualified"] else "unqualified")
            if fit:
                fits.append(fit)
            job = self.matching_job(stage, track, candidate_id, trajectory, seed, origins, paths)
            terminal = self.terminals.get(job["job_id"]) if job else None
            terminal_states[trajectory] = terminal["state"] if terminal else "pending"
            # Main-seed sensitivity is a fixed subset of the promoted schedule.
            if stage == "matched_main_seed":
                promotion = self.matching_job("promote", track, candidate_id, trajectory, seed,
                                              self.cfg["promote"]["origins"], self.cfg["promote"]["paths"])
                terminal = self.terminals.get(promotion["job_id"]) if promotion else None
                terminal_states[trajectory] = terminal["state"] if terminal else "pending"
            for first in origins:
                files = list((self.root / "forecasts" / key).glob(f"{first:05d}_{paths}_*.json"))
                if len(files) > 1:
                    raise ValueError("Ambiguous forecast receipts")
                if not files:
                    continue
                record = self.read(files[0])
                if fit is None:
                    raise ValueError("Forecast without fit receipt")
                expected = content_hash({"native_fit": fit["native_fit_identity"], "first": first,
                    "history": self.history["history"], "horizon": 100, "paths": paths,
                    "forecast_seed": self.cfg["forecast_seed"], "partition": "validation",
                    "score_source": self.experiment["source"]["application"]["ks_metrics.py"]})
                if (record["identity"] != expected or record["native_fit"] != fit["native_fit_identity"]
                        or record["fit_key"] != key or record["track"] != track or record["candidate_id"] != candidate_id
                        or record["trajectory"] != trajectory or record["seed"] != seed
                        or record["first_target"] != first or record["history"] != self.history["history"]
                        or record["partition"] != "validation" or record["paths"] != paths
                        or record.get("forecast_namespace", "main") != "main" or not record["qualified"]):
                    raise ValueError("Stale or incompatible forecast receipt")
                outcomes[trajectory, first] = record
                forecast_seconds += float(record["seconds"])
        result = summarize_cohort(outcomes, trajectories, origins, fit_states, terminal_states)
        result.update(track=track, candidate_id=candidate_id, seed=seed, stage=stage, roles=roles,
                      stratum=self.candidates[(track, candidate_id)]["stratum"],
                      fits_available=len(fits), qualified_fits=sum(f["diagnostics"]["qualified"] for f in fits),
                      fit_seconds=sum(f["fit_seconds"] for f in fits), forecast_seconds=forecast_seconds,
                      peak_fit_process_rss_bytes=max((f["peak_fit_process_rss_bytes"] or 0 for f in fits), default=None))
        return result


def search_evidence(reader):
    selections, rows = {}, []
    for track in reader.cfg["tracks"]:
        for stage in ("screen", "promote", "precision"):
            path = reader.root / "selection" / f"{stage}_{track}.json"
            if not path.exists():
                continue
            raw = reader.read(path)
            if raw["track"] != track:
                raise ValueError("Selection track mismatch")
            selections[f"{stage}_{track}"] = raw
            if stage == "precision":
                continue
            for plot_index, row in enumerate(raw["rows"], 1):
                candidate = reader.candidates[(track, row["id"])]
                if row["configuration"] != candidate["config"] or row["effective_hash"] != candidate["effective_hash"]:
                    raise ValueError("Selection configuration mismatch")
                rows.append({"stage": stage, "track": track, "plot_index": plot_index, "candidate_id": row["id"],
                    "stratum": row["stratum"], "eligible": row["eligible"], "band_crps": row["loss"],
                    "band_mse": row["point_loss"], "qualified_episodes": row["qualified_episodes"],
                    "expected_episodes": row["expected_episodes"], "failed_origins": row["failed_origins"],
                    "pending_origins": row["pending_origins"],
                    "unqualified_fits": row["inference_unqualified_fits"],
                    "preflight_exclusion": row["preflight_exclusion"], "effective_hash": row["effective_hash"]})
    if reader.freeze:
        for track, selected in reader.freeze["selected"].items():
            promoted = selections.get(f"promote_{track}")
            if promoted is None:
                raise ValueError("Freeze without promotion receipt")
            for role in ("main", "sensitivity"):
                if selected.get(role):
                    matches = [r for r in promoted["rows"] if r["id"] == selected[role]["id"]]
                    if matches != [selected[role]]:
                        raise ValueError("Frozen selected row differs from promotion receipt")
            precision_path = reader.root / "selection" / f"precision_{track}.json"
            expected = selected.get("precision_receipt_sha256")
            if expected and (not precision_path.exists() or sha256(precision_path) != expected):
                raise ValueError("Frozen precision receipt changed")
    return selections, rows


def controlled_contrasts(reader, selections):
    """Same-window screen contrasts; their n=2 is not confirmation evidence."""
    output = []
    for track in reader.cfg["tracks"]:
        screen = selections.get(f"screen_{track}")
        if screen is None:
            continue
        anchor_id = reader.registry["tracks"][track]["anchor_id"]
        by_id = {row["id"]: row for row in screen["rows"]}
        anchor = by_id.get(anchor_id)
        if anchor is None:
            raise ValueError("Screen receipt omits the declared original anchor")
        for cid, row in by_id.items():
            candidate = reader.candidates[(track, cid)]
            a = {e["trajectory"]: e for e in anchor["episodes"]}
            b = {e["trajectory"]: e for e in row["episodes"]}
            ids = sorted(a.keys() & b.keys())
            category = ("prior_sensitivity" if candidate["stratum"] == "prior_scale_sensitivity" else
                        "controlled_single_factor" if candidate["single_factor_control"] else
                        "original_anchor" if cid == anchor_id else
                        "legacy_reference" if candidate.get("origin") == "legacy_reference" else "joint_proposal")
            output.append({"track": track, "candidate_id": cid, "anchor_id": anchor_id,
                "category": category, "stratum": candidate["stratum"], "aliases": candidate["aliases"],
                "changed_effective_fields": candidate["changed_effective_fields"],
                "candidate_eligible": row["eligible"], "anchor_eligible": anchor["eligible"],
                "screen_band_crps": row["loss"], "screen_band_mse": row["point_loss"],
                "paired_trajectories": ids,
                "delta_band_crps": float(np.mean([b[i]["selection_crps"] - a[i]["selection_crps"] for i in ids])) if ids else None,
                "delta_band_mse": float(np.mean([b[i]["selection_mse"] - a[i]["selection_mse"] for i in ids])) if ids else None,
                "interpretation": "exploratory n<=2 selection diagnostic on the same four validation windows; not confirmation or causal attribution"})
    return output


def dynamics_evidence(reader):
    measured = {}
    for path in sorted((reader.root / "preflight").glob("*.json")):
        matching = [j for j in reader.jobs.values() if j["stage"] == "preflight" and j["fit_key"] == path.stem]
        if len(matching) != 1:
            raise ValueError("Preflight geometry lacks one bound job")
        job = matching[0]
        receipt = reader.read(path)
        key = (job["track"], job["candidate"]["id"])
        measured.setdefault(key, []).append({"trajectory": job["trajectory"], "seed": job["seed"],
            "layers": receipt["synchronization"]["layer_bounds"],
            "synchronization": receipt["synchronization"]["rows"],
            "geometry_condition": receipt["geometry"]["condition_number"],
            "readout_coefficients": receipt["dimensions"]["readout"]})
    rows = []
    for (track, cid), candidate in reader.candidates.items():
        config = candidate["config"]
        f = config["features"]
        observations = measured.get((track, cid), [])
        layers = [layer for item in observations for layer in item["layers"]]
        def extent(key):
            vals = [float(x[key]) for x in layers]
            return [min(vals), max(vals)] if vals else None
        rows.append({"track": track, "candidate_id": cid, "stratum": candidate["stratum"],
                     "effective_hash": candidate["effective_hash"],
                     "configured_operator_norm": f["recurrent_norm"],
                     "realized_operator_norm_range": extent("recurrent_operator_norm"),
                     "realized_spectral_radius_range": extent("spectral_radius"),
                     "units": f["units"], "leak_patterns": f["leak_patterns"], "input_scales": f["input_scales"],
                     "connectivity": f["connectivity"], "modes": f["modes"], "lags": f["lags"],
                     "squares": f["squares"], "products": f["products"], "score_bound": f["score_bound"],
                     "feature_scale_floor": f["feature_scale_floor"], "conditioning": config["conditioning"],
                     "readout_coefficients": candidate["cost"]["readout_coefficients"],
                     "observations": observations})
    return rows


def dynamics_ranges(rows):
    result = []
    for track, stratum in sorted({(r["track"], r["stratum"]) for r in rows}):
        group = [r for r in rows if r["track"] == track and r["stratum"] == stratum]
        def span(values):
            values = list(values)
            return [min(values), max(values)] if values else None
        result.append({"track": track, "stratum": stratum, "candidates": len(group),
            "configured_operator_norm": span(r["configured_operator_norm"] for r in group),
            "realized_operator_norm": span(x for r in group for x in (r["realized_operator_norm_range"] or [])),
            "realized_spectral_radius": span(x for r in group for x in (r["realized_spectral_radius_range"] or [])),
            "reservoir_units": span(sum(r["units"]) for r in group),
            "input_scale": span(x for r in group for x in r["input_scales"]),
            "leak": span(x for r in group for pattern in r["leak_patterns"] for x in pattern),
            "readout_coefficients": span(r["readout_coefficients"] for r in group)})
    return result


def anchor_cohorts(reader, root):
    """Import separately executed, deterministic anchors without re-fitting them."""
    root = Path(root)
    manifest = {}
    def read(name):
        path = root / name
        payload = path.read_bytes()
        manifest[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        return json.loads(payload)
    plan, binding = read("plan.json"), read("binding.json")
    if (not reader.freeze or plan["identity"] != content_hash({k: v for k, v in plan.items() if k != "identity"})
            or plan["experiment_identity"] != reader.experiment["identity"]
            or binding["experiment_identity"] != reader.experiment["identity"]
            or binding["plan_identity"] != plan["identity"]
            or binding["freeze_identity"] != reader.freeze["identity"]
            or binding["freeze_sha256"] != sha256(reader.root / "freeze.json")
            or plan["trajectories"] != CONFIRMATION or plan["origins"] != ORIGINS
            or plan["history"] != reader.history["history"] or plan["horizon"] != 100
            or plan["partition"] != "validation"):
        raise ValueError("Incompatible fixed-anchor study binding")
    rows = []
    for track in plan["tracks"]:
        for model in plan["models"]:
            outcomes, fit_states, terminal_states = {}, {}, {}
            fits_available, fit_seconds, forecast_seconds, peak = 0, 0., 0., 0
            for trajectory in CONFIRMATION:
                job_id = f"{track}_{model}_t{trajectory:02d}"
                fit_name = f"fits/{job_id}/receipt.json"
                fit = read(fit_name) if (root / fit_name).exists() else None
                fit_states[trajectory] = "missing" if fit is None else "qualified" if fit["qualified"] else "unqualified"
                if fit:
                    expected = content_hash({"plan": plan["identity"], "track": track, "model": model,
                        "trajectory": trajectory, "training_identity_sha256": fit["training_identity_sha256"]})
                    if (fit["identity"] != expected or fit["binding"] != binding or fit["track"] != track
                            or fit["trajectory"] != trajectory or fit["model"] != model
                            or fit["config"] != plan["configurations"][track][model]["config"]):
                        raise ValueError("Incompatible fixed-anchor fit receipt")
                    fits_available += 1
                    fit_seconds += float(fit["fit_seconds"])
                    peak = max(peak, fit["resource"]["process_peak_rss_bytes"])
                terminal_name = f"terminal/{job_id}.json"
                terminal = read(terminal_name) if (root / terminal_name).exists() else None
                if terminal and (terminal["binding"] != binding or terminal["plan_identity"] != plan["identity"]
                                 or terminal["job_id"] != job_id or terminal["state"] not in TERMINAL):
                    raise ValueError("Incompatible fixed-anchor terminal receipt")
                terminal_states[trajectory] = terminal["state"] if terminal else "pending"
                for first in ORIGINS:
                    name = f"forecasts/{job_id}/{first:05d}.json"
                    if not (root / name).exists():
                        continue
                    r = read(name)
                    if fit is None:
                        raise ValueError("Fixed-anchor forecast without fit receipt")
                    expected = content_hash({"fit": fit["identity"], "first_target": first,
                                             "history": plan["history"], "horizon": 100})
                    if (r["identity"] != expected or r["fit_identity"] != fit["identity"]
                            or r["plan_identity"] != plan["identity"] or r["track"] != track or r["model"] != model
                            or r["trajectory"] != trajectory or r["first_target"] != first
                            or r["partition"] != "validation" or r["history"] != plan["history"] or r["horizon"] != 100):
                        raise ValueError("Incompatible fixed-anchor forecast receipt")
                    if r["state"] == "complete" and (r["metrics"]["paths"] != 0 or "crps" in r["metrics"]):
                        raise ValueError("Point anchor has an invented predictive distribution")
                    outcomes[trajectory, first] = r
                    forecast_seconds += r["seconds"]
            row = summarize_cohort(outcomes, CONFIRMATION, ORIGINS, fit_states, terminal_states)
            row.update(track=track, candidate_id=model, seed=plan["model_seeds"][model], stage="confirm_point_anchor", roles=["cheap_anchor"],
                       stratum="fixed_original_point_anchor", fits_available=fits_available, qualified_fits=fits_available,
                       fit_seconds=fit_seconds, forecast_seconds=forecast_seconds, peak_fit_process_rss_bytes=peak,
                       config=plan["configurations"][track][model])
            rows.append(row)
    return rows, {"plan": plan, "binding": binding, "source_manifest": manifest}


def migration_evidence(reader):
    """Bind applied migration metadata without reopening fits or forecast scores."""
    path = reader.root / "migration" / "receipt.json"
    if not path.exists():
        return None, {}
    raw = reader.read(path)
    binding = reader.manifest["migration/receipt.json"]
    migration = {k: v for k, v in raw.items() if isinstance(v, (int, float, bool))
                 or (isinstance(v, str) and ("sha256" in k or "identity" in k or k in ("schema", "status")))}
    migration.update(receipt_sha256=binding["sha256"], receipt_bytes=binding["bytes"],
                     receipt_content_identity=content_hash(raw),
                     preflight_adoption_count=len(raw.get("preflight_adoptions", [])),
                     scope="Immutable migration snapshot; counts are not final campaign totals")
    for key in ("source_experiment", "destination_experiment", "created_utc"):
        if key in raw:
            migration[key] = raw[key]
    if raw.get("schema") != "ks-improvement-budget-migration/1":
        return migration, {}
    flags = ("passed", "apply", "attempt_bytes_preserved", "native_forecast_and_example_bytes_preserved",
             "history_and_selection_bytes_preserved", "freeze_scientific_fields_and_original_created_utc_preserved",
             "all_spent_work_including_prior_migration_attempts_carried_exactly", "native_identity_guards_remain_required")
    if (any(raw.get(key) is not True for key in flags) or reader.freeze is None
            or raw["destination_experiment"] != reader.experiment["identity"]
            or raw["destination_freeze_identity"] != reader.freeze["identity"]
            or raw["original_freeze_created_utc"] != reader.freeze["created_utc"]
            or raw["wall_budget_start"] != reader.cfg["budget"]["campaign_started_utc"]
            or raw["runtime_and_cache"]["cache_identity"] != reader.experiment["cache_identity"]
            or raw["runtime_and_cache"]["numerical_source_runtime_identical"] is not True
            or raw["scientific_recomputation"] is not False
            or raw["source_root_modified_or_deleted"] is not False
            or raw["confirmation_score_objects_decoded"] != 0
            or not math.isclose(raw["carried_worker_seconds"], raw["attempt_elapsed_seconds"] +
                                raw["preparation_worker_seconds"], rel_tol=0., abs_tol=1e-9)):
        raise ValueError("Applied budget migration lacks its frozen scientific/accounting binding")
    for key in ("source_freeze_identity", "destination_freeze_identity", "original_freeze_created_utc",
                "wall_budget_start", "terminal_states", "destination_terminal_states", "forecast_receipt_states"):
        migration[key] = raw[key]
    migration["runtime_and_cache"] = {k: raw["runtime_and_cache"][k] for k in
                                     ("cache_identity", "numerical_source_runtime_identical")}
    migration["cache_relocation"] = {k: raw["cache_relocation"][k] for k in
        ("explicit_source_path", "expected_cache_identity", "same_verified_cache_and_runtime_required")}
    opaque = raw["reviewed_fit_sidecars"]
    migration["reviewed_fit_sidecars"] = {"binding": opaque["binding"],
        "count": raw["reviewed_fit_sidecar_count"], "bytes": raw["reviewed_fit_sidecar_bytes"],
        "canonical_checkpoint_adoption": opaque["canonical_checkpoint_adoption"]}
    auxiliary = raw["reviewed_inference_sidecars"]
    migration["reviewed_inference_sidecars"] = {"binding": auxiliary["binding"], "fits": [
        {**{k: row[k] for k in ("fit_key", "native_sha256", "inference_restart_state_available", "qualification_changed")},
         "archived_artifact_count": len(row["archive_paths"])} for row in auxiliary["fits"]]}
    if (opaque["canonical_checkpoint_adoption"] is not False
            or any(row["inference_restart_state_available"] is not False or row["qualification_changed"] is not False
                   for row in auxiliary["fits"])):
        raise ValueError("Reviewed sidecars were relabeled as active checkpoint state")

    repository = Path(__file__).resolve().parents[1]
    public = {}
    def public_read(key, name, fields):
        relative = "docs/applications/evidence/" + name
        payload = (repository / relative).read_bytes()
        value = json.loads(payload)
        public[key] = {"repository_path": relative, "bytes": len(payload),
                       "sha256": hashlib.sha256(payload).hexdigest(),
                       "data": {k: value[k] for k in fields if k in value}}
        return value
    review = public_read("applied_migration_review", "ks_improvement_applied_migration_independent_review.json",
        ("schema", "created_utc", "passed", "scope", "scope_limits", "applied_migration_receipt", "identities",
         "budget_accounting", "migration_snapshot_counts", "scientific_contract_preserved", "reviewed_auxiliary_pair"))
    if (review.get("passed") is not True
            or review["applied_migration_receipt"]["sha256"] != binding["sha256"]
            or review["applied_migration_receipt"]["bytes"] != binding["bytes"]
            or review["identities"]["destination_experiment"] != reader.experiment["identity"]
            or review["identities"]["destination_freeze"] != reader.freeze["identity"]
            or review["identities"]["source_experiment"] != raw["source_experiment"]
            or review["identities"]["cache_identity_unchanged"] != reader.experiment["cache_identity"]
            or review["budget_accounting"]["carried_worker_seconds"] != raw["carried_worker_seconds"]
            or review["budget_accounting"]["original_campaign_started_utc"] != raw["wall_budget_start"]
            or review["scientific_contract_preserved"]["freeze_created_utc"] != reader.freeze["created_utc"]
            or review["migration_snapshot_counts"]["native_checkpoints"] != raw["native_checkpoint_count"]
            or review["migration_snapshot_counts"]["forecast_receipts"] != sum(raw["forecast_receipt_states"].values())):
        raise ValueError("Independent applied-migration review does not bind the current immutable receipt")
    if auxiliary["fits"]:
        proof = public_read("native_auxiliary_validation", "ks_improvement_native_auxiliary_validation_13103f.json",
            ("schema", "created_utc", "source_experiment_identity", "fit_key", "track", "candidate_id", "trajectory",
             "seed", "native_fit_identity", "native_qualified", "native_posterior_state_counts", "comparison_counts",
             "retained_state_prefix_equal_per_chain", "retained_log_density_prefix_equal_per_chain",
             "source_files_unchanged_during_read", "actual_chain_matches_completed_phase_sha256",
             "exact_completed_inference_restart_available_from_auxiliary_pair", "scientific_recomputation"))
        reviewed_pair = review["reviewed_auxiliary_pair"]
        matches = [r for r in auxiliary["fits"] if r["fit_key"] == proof["fit_key"]]
        if (len(matches) != 1 or len(auxiliary["fits"]) != 1
                or public["native_auxiliary_validation"]["sha256"] != reviewed_pair["prefix_proof_sha256"]
                or proof["source_experiment_identity"] != raw["source_experiment"]
                or proof["files"]["fit.npz"]["sha256"] != matches[0]["native_sha256"]
                or reviewed_pair["native_sha256"] != matches[0]["native_sha256"]
                or proof["native_qualified"] is not True or reviewed_pair["native_qualification_unchanged"] is not True
                or proof["exact_completed_inference_restart_available_from_auxiliary_pair"] is not False
                or not all(proof["retained_state_prefix_equal_per_chain"].values())
                or not all(proof["retained_log_density_prefix_equal_per_chain"].values())
                or proof["scientific_recomputation"] is not False):
            raise ValueError("Auxiliary-prefix evidence lacks its historical/native binding")
    resume_name = "ks_improvement_actual_resume_acceptance.json"
    if (repository / "docs/applications/evidence" / resume_name).exists():
        resumed = public_read("actual_resume_acceptance", resume_name,
            ("schema", "created_utc", "passed", "scope", "limits", "destination_experiment_identity",
             "migration_receipt_sha256", "immutable_migration_review_sha256", "original_campaign_started_utc", "checks"))
        if (resumed.get("passed") is not True
                or resumed["destination_experiment_identity"] != reader.experiment["identity"]
                or resumed["migration_receipt_sha256"] != binding["sha256"]
                or resumed["immutable_migration_review_sha256"] != public["applied_migration_review"]["sha256"]
                or resumed["original_campaign_started_utc"] != raw["wall_budget_start"]
                or resumed["checks"]["scientific_configuration_seed_history_and_schedule_changes"] != 0):
            raise ValueError("Observed resume acceptance has incompatible migration provenance")
        public["actual_resume_acceptance"]["data"]["cases"] = [
            {k: row[k] for k in ("track", "trajectory", "candidate_id", "model_seed", "forecast_seed", "history",
                "horizon", "paths", "completed_origins", "failed_origins", "sampler_refit",
                "native_and_fit_receipt_byte_identical_to_migration_snapshot")} for row in resumed["cases"]]
    return migration, public


def operation_evidence(reader, operations_root=None):
    attempts = []
    stage_seconds = {}
    for path in sorted((reader.root / "attempts").glob("*.json")):
        raw = reader.read(path)
        entry = {k: raw.get(k) for k in ("job_id", "stage", "elapsed_seconds", "state", "reason", "returncode")}
        attempts.append(entry)
        stage_seconds[raw["stage"]] = stage_seconds.get(raw["stage"], 0.) + float(raw["elapsed_seconds"])
    migration, public = migration_evidence(reader)
    bounded = []
    if operations_root and Path(operations_root).exists():
        for path in sorted(Path(operations_root).glob("*/cache_*.json")):
            raw = json.loads(path.read_text())
            bounded.append({"file": f"{path.parent.name}/{path.name}", "sha256": sha256(path),
                **{k: v for k, v in raw.items() if isinstance(v, (int, float, bool)) and
                   ("bytes" in k or "files" in k or "headroom" in k or "changed" in k)}})
    return {"attempts": attempts, "attempt_worker_seconds_by_stage": stage_seconds,
            "attempt_worker_seconds": sum(stage_seconds.values()),
            "preparation_worker_seconds": reader.cfg["budget"].get("preparation_worker_seconds", 0.),
            "unique_fit_seconds": sum(f["fit_seconds"] for f in reader.fits.values()),
            "migration": migration, "public_evidence": public, "bounded_cache_advice": bounded,
            "budget_semantics": "scheduling thresholds checked before job/group admission; running jobs are not killed when the aggregate threshold is crossed",
            "maximum_admitted_group_worker_seconds": reader.cfg["budget"]["workers"] * reader.cfg["budget"]["job_timeout_seconds"],
            "in_flight_overrun_qualification": "up to workers times per-job timeout plus shutdown overhead beyond a nominal worker threshold; actual final consumption must be checked separately",
            "scope": "attempt wall seconds include reload/resume overhead; unique fit receipts are counted once; no CPU-seconds interpretation"}


def save_figure(fig, path):
    """Export the same figure to PNG and a publication PDF before publishing either."""
    import matplotlib
    path = Path(path)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=170, facecolor="white", bbox_inches="tight")
    data = buffer.getvalue()
    if not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) < 1000:
        raise RuntimeError("Invalid rendered PNG")
    vector = io.BytesIO()
    with matplotlib.rc_context({"pdf.fonttype": 42}):
        fig.savefig(vector, format="pdf", dpi=300, facecolor="white", bbox_inches="tight",
                    metadata={"Title": path.stem, "Creator": "KS improvement report; retained scores only"})
    pdf = vector.getvalue()
    if not pdf.startswith(b"%PDF-") or not pdf.rstrip().endswith(b"%%EOF") or len(pdf) < 1000:
        raise RuntimeError("Invalid rendered figure PDF")
    atomic_bytes(path.with_suffix(".pdf"), pdf)
    atomic_bytes(path, data)


def label(row):
    roles = row.get("roles", [])
    role = " / ".join({"main": "Preserved-prior winner", "anchor": "Old configuration",
                       "sensitivity": "Prior sensitivity", "cheap_anchor": "Point anchor"}.get(x, x) for x in roles)
    return f"{role}: {row['candidate_id']} (n={row['complete_trajectories']})"


def readable_scale(axis, arrays):
    parts = [np.asarray(a, float).ravel() for a in arrays if a is not None]
    groups = [np.abs(part[np.isfinite(part) & (part != 0)]) for part in parts]
    groups = [part for part in groups if len(part)]
    magnitudes = np.concatenate(groups) if groups else np.array([])
    typical = [float(np.median(part)) for part in groups]
    between_series = bool(typical and max(typical) > 50 * min(typical))
    if len(magnitudes) and (magnitudes.max() > 50 * np.median(magnitudes) or between_series):
        # A majority of very large curves must not flatten a smaller model's
        # curve. This changes only the display transform, never signed data.
        threshold = min(typical) if between_series else float(np.median(magnitudes))
        axis.set_yscale("symlog", linthresh=max(threshold, 1e-8))
        axis.yaxis.get_major_locator().set_params(numticks=7)
        axis.autoscale(enable=True, axis="y")
        axis.margins(y=.08)
        low, high = axis.get_ylim()
        ticks = np.asarray([x for x in axis.get_yticks() if low <= x <= high], float)
        if len(ticks):
            transform = axis.yaxis.get_transform()
            span = abs(float(np.diff(transform.transform(np.array([low, high])))[0]))
            chosen = [0.] if low <= 0 <= high else []
            for tick in sorted(ticks, key=abs, reverse=True):
                if all(abs(float(transform.transform(tick) - transform.transform(other))) >= .14 * span
                       for other in chosen):
                    chosen.append(float(tick))
            axis.set_yticks(sorted(chosen))
        axis.text(.01, .99, f"symlog (linear ±{threshold:.3g}); all finite values retained",
                  transform=axis.transAxes, va="top", fontsize=7)


def plot_report(reader, cohorts, search_rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    paths = []
    for track in reader.cfg["tracks"]:
        rows = [r for r in cohorts if r["track"] == track]
        if any(r["complete_trajectories"] for r in rows):
            fig, axes = plt.subplots(2, 2, figsize=(11, 7.3), sharex=True)
            for ax, metric, title in zip(axes.flat, ("rmse", "crps", "coverage90", "width90"),
                                         ("Trajectory-first RMSE", "CRPS", "90% marginal coverage", "90% interval width")):
                magnitudes = []
                for row in rows:
                    curve = row["curves"][metric]
                    if curve["mean"] is None:
                        continue
                    x = np.arange(1, 101) * .25
                    line, = ax.plot(x, curve["mean"], lw=1.6, label=label(row))
                    magnitudes.append(curve["mean"])
                    if curve["ci95"]:
                        low, high = curve["ci95"]
                        ax.fill_between(x, low, high, color=line.get_color(), alpha=.13, lw=0)
                        magnitudes.extend([low, high])
                if metric == "coverage90":
                    ax.axhline(.9, ls="--", color="black", lw=.8)
                    ax.set_ylim(0, 1)
                elif magnitudes:
                    readable_scale(ax, magnitudes)
                ax.set_title(title)
                ax.grid(alpha=.2)
                ax.set_xlabel("Physical forecast lead (0.25 × saved steps)")
            handles, legends = axes[0, 0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, legends, loc="lower center", fontsize=8, ncol=1)
            fig.suptitle(f"{track} · {TRACK_NAMES[track]} · validation confirmation\nComplete trajectories only; reliability table retains the full schedule", fontsize=11)
            fig.tight_layout(rect=(0, .055 + .025 * len(handles), 1, .93))
            path = output / f"horizon_{track}.png"
            save_figure(fig, path); plt.close(fig); paths.append(path.name)
            fig, ax = plt.subplots(figsize=(7.5, 6.7))
            for row in rows:
                if not row["complete_trajectories"] or row["fits_available"] == 0:
                    continue
                mean = row["metrics"]["rmse"]["mean"]
                ax.scatter(row["fit_seconds"] / row["fits_available"], mean, s=45, label=label(row))
            ax.set_xscale("log")
            readable_scale(ax, [[r["metrics"]["rmse"]["mean"]] for r in rows if r["metrics"]["rmse"]["mean"] is not None])
            if ax.get_yscale() == "symlog":
                ax.set_ylim(bottom=0)  # RMSE is nonnegative; avoid stale linear padding.
            ax.set(xlabel="Mean fitted-model wall seconds (all available fits)", ylabel="Mean trajectory RMSE (complete cases)",
                   title=f"{track} · Accuracy and fitting cost; forecast/search cost reported separately")
            ax.grid(alpha=.2)
            handles, legends = ax.get_legend_handles_labels()
            fig.legend(handles, legends, loc="lower center", fontsize=7, ncol=1)
            fig.tight_layout(rect=(0, .035 + .027 * len(handles), 1, 1))
            path = output / f"compute_{track}.png"
            save_figure(fig, path); plt.close(fig); paths.append(path.name)
        raw = [r for r in search_rows if r["track"] == track]
        if raw:
            fig, axes = plt.subplots(2, 1, figsize=(10, 7.4))
            colors = {"main_fixed_prior": "#1f77b4", "prior_scale_sensitivity": "#d95f02"}
            for ax, stage in zip(axes, ("screen", "promote")):
                stage_rows = [r for r in raw if r["stage"] == stage]
                for row in stage_rows:
                    if row["band_crps"] is not None:
                        ax.scatter(row["plot_index"], row["band_crps"], color=colors.get(row["stratum"], "#777777"), s=32)
                indices = [r["plot_index"] for r in stage_rows]
                ticks = indices if len(indices) <= 12 else sorted(set(indices[::5] + indices[-1:]))
                ax.set_xticks(ticks, [str(i) for i in ticks], fontsize=10)
                ax.tick_params(axis="y", labelsize=10)
                ax.set_xlim(.3, max(indices, default=1) + .7)
                ax.set_title(f"{stage}: {sum(r['eligible'] for r in stage_rows)}/{len(stage_rows)} eligible", fontsize=11)
                ax.set_xlabel("Configuration index within this stage (plot_index in search_candidates.csv)", fontsize=10)
                ax.set_ylabel("Band-balanced validation CRPS", fontsize=10)
                vals = [r["band_crps"] for r in stage_rows if r["band_crps"] is not None]
                if vals: readable_scale(ax, [[value] for value in vals])
                ax.grid(axis="y", alpha=.2)
            fig.suptitle(f"{track} · Bounded staged search; exact candidate IDs and eligibility remain in search_candidates.csv\nBlue: preserved prior · orange: prior sensitivity · gray: other reference", fontsize=11)
            fig.tight_layout(); path = output / f"search_{track}.png"
            save_figure(fig, path); plt.close(fig); paths.append(path.name)
    if reader.freeze:
        paths.extend(plot_examples(reader, output))
    return paths


def physical_x(coordinates):
    coordinates = np.asarray(coordinates, float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.allclose(np.linalg.norm(coordinates, axis=1), 1.):
        raise ValueError("Example coordinates are not the required unit-circle embedding")
    result = np.mod(np.arctan2(coordinates[:, 1], coordinates[:, 0]), 2 * np.pi) * 22 / (2 * np.pi)
    return np.where(np.isclose(result, 22), 0, result)


def plot_examples(reader, output):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, SymLogNorm
    paths = []
    for track, chosen in reader.freeze["selected"].items():
        ids = list(dict.fromkeys([chosen["anchor_id"], chosen["main"]["id"]]))
        examples = []
        for cid in ids:
            path = reader.root / "examples" / f"{track}_{cid}_t01_s{reader.cfg['model_seed']}_a6000.npz"
            if not path.exists():
                continue
            reader.bind_file(path)
            with np.load(path, allow_pickle=False) as saved:
                data = {k: saved[k].copy() for k in saved.files}
            expected_sites = {"A": 13, "B": 64, "C": 12}[track]
            if (int(data["first_target"]) != 6000 or data["mean"].shape != (100, expected_sites)
                    or data["truth"].shape != data["mean"].shape
                    or len(data["input_coordinates"]) != {"A": 39, "B": 64, "C": 40}[track]
                    or not all(np.isfinite(data[k]).all() for k in ("mean", "truth", "lower", "upper"))):
                raise ValueError("Invalid predeclared example metadata")
            if examples and not np.array_equal(examples[0][1]["truth"], data["truth"]):
                raise ValueError("Matched examples contain different truth")
            examples.append((cid, data))
        if not examples:
            continue
        vmin = min(min(d["mean"].min(), d["truth"].min()) for _, d in examples)
        vmax = max(max(d["mean"].max(), d["truth"].max()) for _, d in examples)
        error_max = max(float(np.abs(d["mean"] - d["truth"]).max()) for _, d in examples)
        truth_scale = max(float(np.max(np.abs(examples[0][1]["truth"]))), 1e-8)
        use_log = max(abs(vmin), abs(vmax), error_max) > 50 * truth_scale
        amplitude = max(abs(vmin), abs(vmax), 1e-8)
        color_norm = (SymLogNorm(linthresh=truth_scale, vmin=-amplitude, vmax=amplitude)
                      if use_log else Normalize(vmin=vmin, vmax=vmax))
        error_norm = (SymLogNorm(linthresh=truth_scale, vmin=-max(error_max, 1e-8), vmax=max(error_max, 1e-8))
                      if use_log else Normalize(vmin=-max(error_max, 1e-8), vmax=max(error_max, 1e-8)))
        fig = plt.figure(figsize=(12, 3.8 * len(examples) + 1.3))
        grid = fig.add_gridspec(len(examples) + 1, 3, height_ratios=[1] * len(examples) + [.13], hspace=.42, wspace=.30)
        for row, (cid, data) in enumerate(examples):
            xs = physical_x(data["coordinates"]); order = np.argsort(xs)
            for col, (values, title) in enumerate(((data["truth"], "Clean truth"), (data["mean"], "Predictive mean"),
                                                  (data["mean"] - data["truth"], "Signed error: mean − truth"))):
                ax = fig.add_subplot(grid[row, col])
                image = ax.imshow(values[:, order].T, aspect="auto", origin="lower", extent=(.125, 25.125, -.5, len(xs) - .5),
                                  cmap="RdBu_r", norm=error_norm if col == 2 else color_norm)
                ticks = np.arange(len(xs)) if len(xs) <= 13 else np.arange(0, len(xs), 8)
                ax.set_yticks(ticks, [f"{xs[order[i]]:.2f}" for i in ticks], fontsize=7)
                ax.set_title(f"{cid}\n{title}")
                if row == len(examples) - 1:
                    ax.set_xlabel("Physical forecast lead")
                else:
                    ax.tick_params(axis="x", labelbottom=False)
                if col == 0:
                    ax.set_ylabel("Scored coordinate x (categorical rows)")
                else:
                    ax.tick_params(axis="y", labelleft=False)
                colorbar = fig.colorbar(image, ax=ax, fraction=.045, pad=.025,
                                       label="Physical u" + (" (symlog color)" if use_log else ""))
                if use_log:
                    # Display-only ticks spaced in color coordinates: preserve the
                    # exact signed norm endpoints and zero without crowding +/-1.
                    color_ticks = np.asarray(image.norm.inverse(np.linspace(0., 1., 5)))
                    color_ticks[2] = 0.
                    colorbar.set_ticks(color_ticks, labels=[f"{value:.2g}" for value in color_ticks])
                    colorbar.ax.tick_params(labelsize=7)
        rug = fig.add_subplot(grid[-1, :])
        data = examples[0][1]
        rug.plot(physical_x(data["input_coordinates"]), np.zeros(len(data["input_coordinates"])), "|", ms=10, color="black", label="Permitted input sites")
        rug.plot(physical_x(data["coordinates"]), np.ones(len(data["coordinates"])), "o", ms=3.5, color="#d95f02", label="Official validation target sites")
        rug.set(xlim=(-.2, 22.2), ylim=(-.5, 1.5), xlabel="Periodic physical domain [0,22)", yticks=[])
        rug.legend(loc="upper center", bbox_to_anchor=(.5, 2.1), ncol=2, fontsize=8)
        fig.suptitle(f"{track} · Predeclared trajectory 1, first target 6000\nOnly the official validation panel is scored; the displayed rows do not interpolate a complete field", fontsize=11, y=.99)
        path = output / f"example_{track}_t01_a6000.png"
        save_figure(fig, path); plt.close(fig); paths.append(path.name)
        indices = [0, (len(data["coordinates"]) - 1) // 2, len(data["coordinates"]) - 1]
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        x = np.arange(1, 101) * .25
        order = np.argsort(physical_x(data["coordinates"]))
        for ax, position in zip(axes, indices):
            j = order[position]
            ax.plot(x, data["truth"][:, j], color="black", lw=1.4, label="Clean truth")
            for cid, values in examples:
                line, = ax.plot(x, values["mean"][:, j], lw=1.25, label=cid + " predictive mean")
                ax.fill_between(x, values["lower"][:, j], values["upper"][:, j], color=line.get_color(), alpha=.16,
                                label=cid + " 90% predictive interval")
            readable_scale(ax, [data["truth"][:, j]] + [v[key][:, j] for _, v in examples for key in ("mean", "lower", "upper")])
            ax.set(ylabel="Physical u", title=f"Fixed coordinate x={physical_x(data['coordinates'])[j]:.4f}")
            ax.grid(alpha=.2)
        axes[-1].set_xlabel("Physical forecast lead (0.25 × saved steps)")
        handles, legends = axes[0].get_legend_handles_labels()
        fig.legend(handles, legends, loc="lower center", ncol=2, fontsize=7)
        fig.suptitle(f"{track} · t1 / target6000 · coordinate-only first/middle/last target-site rule\nSaved examples contain 90% intervals; no unretained quantiles are reconstructed", fontsize=10)
        fig.tight_layout(rect=(0, .10, 1, .91)); path = output / f"intervals_{track}_t01_a6000.png"
        save_figure(fig, path); plt.close(fig); paths.append(path.name)
    return paths


def format_number(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    return f"{value:.5g}" if isinstance(value, (float, int)) else str(value)


def with_ci(summary):
    result = format_number(summary["mean"])
    if summary.get("ci95"):
        result += " [" + ", ".join(format_number(x) for x in summary["ci95"]) + "]"
    return result


def seed_sensitivity_markdown(rows, pairs):
    """Display the already aggregated matched populations; never recompute scores."""
    text = ["## Model-seed sensitivity", "",
        "The two additional seeds use four declared selection trajectories and eight validation origins. "
        "The main-seed reference uses exactly the same four trajectories and origins, reusing compatible promotion receipts. "
        "It is not the 14-trajectory confirmation mean. Each seed/candidate/track has 32 scheduled origins. "
        "Model seeds are not posterior draws, and predictions from different seeds are not pooled.", ""]
    if not rows:
        return text + ["Matched seed-sensitivity outcomes are pending.", ""]
    text += ["| Track / candidate | Role | Seed / population | Complete trajectories | RMSE [95% CI] | CRPS [95% CI] |",
             "|---|---|---|---:|---:|---:|"]
    for row in rows:
        population = "matched main" if row["stage"] == "matched_main_seed" else "alternate"
        text.append(f"| {row['track']} / {row['candidate_id']} | {' / '.join(row['roles'])} | "
                    f"{row['seed']} / {population} | {row['complete_trajectories']}/{row['scheduled_trajectories']} | "
                    f"{with_ci(row['metrics']['rmse'])} | {with_ci(row['metrics']['crps'])} |")
    text += ["", "| Track / candidate / seed | Qualified fits | Finite / scheduled origins | Rollout failed | Inference blocked | Other failed-job blocked | Pending | Finite divergent | Full-schedule score defined |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        text.append(f"| {row['track']} / {row['candidate_id']} / {row['seed']} | "
                    f"{row['qualified_fits']}/{row['scheduled_trajectories']} | "
                    f"{row['finite_successful_origins']}/{row['scheduled_origins']} | {row['failed_rollout_origins']} | "
                    f"{row['unqualified_blocked_origins']} | {row['failed_job_blocked_origins']} | "
                    f"{row['pending_origins']} | {row['finite_divergent_origins']} | {row['full_schedule_point_score_defined']} |")
    text += ["", "Accuracy includes only complete trajectories; the census retains every scheduled origin. "
             "Inference-blocked and other blocked outcomes are not observed nonfinite rollouts. Finite divergence remains scored. "
             "A dash denotes an undefined or pending statistic, never zero error. The JSON retains per-trajectory counts and "
             "finite origins outside the accuracy population.", "",
             "| Track / candidate | Alternate − main seed | Metric | Paired trajectories | Exact paired IDs | Mean difference [95% CI] |",
             "|---|---|---|---:|---|---:|"]
    for pair in pairs:
        if pair["metric"] not in ("rmse", "crps"):
            continue
        text.append(f"| {pair['track']} / {pair['left']} | {pair['left_seed']} − {pair['right_seed']} | "
                    f"{pair['metric']} | {pair['n']}/{len(pair['scheduled_trajectories'])} | "
                    f"{pair['paired_trajectories']} | {with_ci(pair)} |")
    text += ["", "Differences are alternate minus matched main seed on the exact intersection of complete trajectories. "
             "Negative RMSE/CRPS differences favor the alternate seed. Intervals resample trajectories, with at most four "
             "independent trajectory clusters; no interval is reported for fewer than two pairs. Missing pairs are not imputed, "
             "and all scheduled zero-pair contrasts remain listed. These are conditional sensitivity checks after selection.", ""]
    return text


def migration_markdown(operations):
    migration = operations.get("migration")
    if migration is None:
        return ["No applied migration receipt is bound in this report snapshot.", ""]
    text = ["### Applied migration and retained restart scope", "",
            f"The consumed migration receipt has SHA-256 `{migration['receipt_sha256']}`. "
            "Its counts describe the immutable migration snapshot, not final campaign completion. "
            "`operations.json` preserves the sanitized receipt fields and hash-bound independent evidence.", ""]
    if migration.get("schema") != "ks-improvement-budget-migration/1":
        return text + ["This earlier operational receipt is retained with its recorded schema; it is not relabeled "
                       "as proof of a later budget-extension migration.", ""]
    text += [f"The applied migration was recorded at `{migration['created_utc']}`. It preserved "
             f"{migration['native_checkpoint_count']} native checkpoints and "
             f"{sum(migration['forecast_receipt_states'].values())} forecast receipts, with "
             f"{migration['attempt_files']} existing attempt records carrying "
             f"{migration['carried_worker_seconds']:.6f} worker-seconds including preparation. "
             "Prior interruptions and historical attempt IDs remain charged. Frozen scientific choices and their "
             f"original creation time `{migration['original_freeze_created_utc']}` were retained. "
             "No model fit, forecast, or selection was recomputed by migration.", "",
             f"Source experiment: `{migration['source_experiment']}`. Adopted experiment: "
             f"`{migration['destination_experiment']}`. The actual migration receipt and independent review bind "
             "the adopted freeze, original clock, identical numerical runtime, and exact verified cache identity. "
             "Explicit cache-path relocation changes an operational path, not the dataset or scientific protocol.", ""]
    opaque = migration["reviewed_fit_sidecars"]
    text += [f"{opaque['count']} exact-bound opaque sidecars ({opaque['bytes']} bytes) were preserved outside "
             "active fits and were not adopted as checkpoints. Their creator is unproven.", ""]
    auxiliary = migration["reviewed_inference_sidecars"]["fits"]
    for row in auxiliary:
        text += [f"For fit `{row['fit_key']}`, {row['archived_artifact_count']} reviewed auxiliary chain/phase files "
                 "were archived outside active inference state. The completed native posterior and its qualification "
                 "were preserved; this is not an additional failed fit or failed forecast. The archived pair is "
                 "unavailable as an exact final inference restart, and no final RNG state or additional posterior "
                 "draws were manufactured. All unlisted checkpoints remain subject to strict validation.", ""]
    public = operations.get("public_evidence", {})
    proof = public.get("native_auxiliary_validation", {}).get("data")
    if proof:
        counts = proof["comparison_counts"]
        text += [f"The bound prefix check for {proof['track']}/{proof['candidate_id']}, trajectory "
                 f"{proof['trajectory']}, seed {proof['seed']}, verified {counts['array_comparisons']} retained "
                 f"arrays and {counts['scalar_comparisons']} scalar values against the completed native posterior, "
                 f"whose per-chain retained state counts are {proof['native_posterior_state_counts']}. "
                 "It supports the retained native forecast state while preserving the auxiliary restart limitation. "
                 "The source of the historical discrepancy is unproven.", ""]
    resume = public.get("actual_resume_acceptance", {}).get("data")
    if resume:
        checks = resume["checks"]
        text += [f"A separate observed-resume check covered {checks['closed_jobs_checked']} closed jobs: "
                 f"{checks['existing_native_models_reused']} existing native fits were reused, "
                 f"{checks['original_origin_receipts_byte_identical']} prior origin receipts remained byte-identical, "
                 f"and {checks['previously_missing_origins_now_complete']} previously absent origins were completed "
                 "by the resumed coordinator. This operational check did not read performance scores or rerun "
                 "selection, and it does not establish completion of other campaign jobs.", ""]
    return text


def render_markdown(report, cohorts, comparisons, figures, diagnosis, *, seed_rows=(), seed_pairs=()):
    text = ["# Bayesian ST-ESN: validation-only KS improvement study", "",
        f"**Status: {report['status'].upper()}.** Receipt snapshot: {report['created_utc']}.", "",
        "This is a custom, literature-grounded KS experiment. It is an exploratory follow-up motivated by the original application results. "
        "This campaign uses no test-partition labels. Its 14 confirmation trajectories were withheld from this campaign's configuration selection; "
        "they do not constitute a newly untouched test set for the wider research project.", "",
        "The Gaussian likelihood, accepted scalar-ridge hierarchy, recursive path simulation, data adapter, and original scores remain the reference. "
        "A preserved-prior winner and, when eligible, a separate prior-scale sensitivity winner are selected within each spatial track. "
        "Prior-sensitivity results are never relabeled as the unchanged-prior model.", "",
        r"The retained hierarchy is $y_t=\Phi C z_t+e_t$, with $e_t\sim N(0,\sigma^2(I+qK_\ell))$ independently across time, "
        r"and $\mathrm{vec}(C)\mid\sigma^2,\lambda\sim N(0,\sigma^2\lambda^{-1}I)$. "
        "Every readout coefficient, including the intercept, is shrunk. The scalar precision is inferred rather than equated with a deterministic "
        "ridge penalty. The preserved-prior stratum retains its track's reference hyperprior constants; a separate stratum changes declared "
        "prior scales. Exact constants and spatial-basis settings appear in the frozen configuration files. Residual innovations represent "
        "statistical discrepancy for the clean field, not added measurement noise.", "",
        "## Scientific contract and aggregation", "",
        f"Training targets are [0,6000); all scored targets are within validation [6000,8000). History is {report['history']} observations; "
        "each autonomous forecast has 100 steps, equal to 25 physical time units. Confirmation uses the frozen 14-trajectory roster "
        "and 20 first targets 6000,6100,…,7900, with 512 complete predictive paths per origin.", "",
        "Track A uses 39 training sites and 13 interleaved validation sites. Track B uses all 64 observation sites. "
        "Track C uses 40 training sites and 12 contiguous withheld validation sites and has its own selection. "
        "No response-fitted artifact is transferred between these spatial protocols. Known coordinates are permitted. "
        "The delivered facade reads evaluation labels internally before separating its outputs; this is not a claim of future-free disk access.", "",
        "For each complete trajectory, squared error is pooled over all its scheduled origins, target sites, and horizons, then square-rooted. "
        "The primary RMSE is the arithmetic mean of those trajectory RMSEs. It is not the mean of origin RMSEs and not the square root of "
        "the all-trajectory pooled MSE. The latter is exported separately as `complete_case_pooled_rmse`. MAE, CRPS, coverage and width "
        "are equally weighted by trajectory after averaging within trajectory. Point forecasts are means of complete predictive paths.", "",
        "Uncertainty summaries resample entire trajectories, retaining all origins within each draw. Intervals are conditional on the frozen "
        "selection. Horizon bands are pointwise intervals. Paired differences use the explicit intersection of complete trajectories; "
        "missing or failed episodes are not imputed. Every finite catastrophic error remains in its score.", "",
        "The selection loss gives equal weight to horizon bands 1–10,11–25,26–50,51–100. The ordinary 100-step confirmation aggregates "
        "below use uniform horizon weight. Selected settings are best among the configurations evaluated under the recorded staged budget.", "",
        "A candidate's conditioning prefix calibrates its PCA/reservoir feature map and is excluded from its likelihood targets; the delivered "
        "scalar response transform separately uses permitted full-training observations. The usable "
        "training likelihood count is 6000 minus that conditioning length. This prefix, the 512-observation forecast synchronization history, "
        "the solver's physical spin-up and the sampler warm-up are distinct quantities. Candidate costs and fit diagnostics record the actual counts.", "",
        "When included, point references keep their original frozen configurations and are refitted on each permitted training panel. "
        "Training mean, persistence, POD-VAR, ridge ESN and POD-NVAR provide fixed references rather than a newly retuned comparator study. "
        "Their periodic piecewise-linear decoder acts only on output: VAR/ESN recur in their observed-site state and POD methods in their "
        "declared latent state. The conventional ESN uses its own [1,r,r²] ridge readout and eigenvalue-radius convention, distinct from the "
        "Bayesian architecture's operator-norm control. Point methods receive no artificial Gaussian intervals.", "",
        "## Confirmation results", ""]
    if not report["freeze_identity"]:
        text += ["Selection has not been frozen. No confirmation comparison is claimed.", ""]
    for track in report["tracks"]:
        rows = [r for r in cohorts if r["track"] == track]
        text += [f"### Track {track}: {TRACK_NAMES[track]}", "",
                 "| Candidate / role | Complete trajectories | Mean trajectory RMSE [95% CI] | CRPS [95% CI] | 90% coverage | 90% width |",
                 "|---|---:|---:|---:|---:|---:|"]
        for row in rows:
            text.append(f"| {row['candidate_id']} ({'/'.join(row['roles'])}) | {row['complete_trajectories']}/{row['scheduled_trajectories']} | "
                        f"{with_ci(row['metrics']['rmse'])} | {with_ci(row['metrics']['crps'])} | "
                        f"{format_number(row['metrics']['coverage90']['mean'])} | {format_number(row['metrics']['width90']['mean'])} |")
        if not rows:
            text.append("| Not frozen | — | — | — | — | — |")
        text += ["", "| Candidate | Scheduled origins | Finite | Nonfinite/numerical rollout failures | Finite divergence | Failed-job blocked | Unqualified blocked | Pending | Full-schedule RMSE defined |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            text.append(f"| {row['candidate_id']} | {row['scheduled_origins']} | {row['finite_successful_origins']} | {row['failed_rollout_origins']} | "
                        f"{row['finite_divergent_origins']} | {row['failed_job_blocked_origins']} | {row['unqualified_blocked_origins']} | "
                        f"{row['pending_origins']} | {row['full_schedule_point_score_defined']} |")
        text += ["", "Finite divergence means a point-forecast spatial RMSE exceeding 10 permitted-training standard deviations at any lead. "
                 "A finite divergence remains scored; it is distinct from a failed numerical rollout. Blocked fits have no forecast outcome and "
                 "are not counted as observed divergences. Reliability JSON retains every scheduled origin, the failure-or-divergence union, "
                 "and all per-trajectory counts. A zero-event empirical bootstrap interval is not evidence of zero risk on new trajectories.", ""]
        for row in rows:
            if row["finite_origins_outside_score_population"]:
                text.append(f"For {row['candidate_id']}, {row['finite_origins_outside_score_population']} additional finite origins belong to "
                            "incomplete trajectories and remain in the reliability census, but not in the complete-trajectory accuracy table.")
        pairs = [p for p in comparisons if p["track"] == track and p["metric"] in ("rmse", "crps")]
        if pairs:
            text += ["", "| Paired contrast (left − right) | Metric | Paired trajectories | Mean difference [95% CI] |", "|---|---|---:|---:|"]
            for row in pairs:
                text.append(f"| {row['left']} − {row['right']} | {row['metric']} | {row['n']}/14 | {with_ci(row)} |")
            text += ["", "Negative RMSE/CRPS differences favor the left candidate. Incomplete intersections support only conditional comparisons.", ""]
        for name in figures:
            if f"_{track}" in name:
                pdf_name = str(Path(name).with_suffix(".pdf"))
                text += [f"![{name.removesuffix('.png').replace('_', ' ')}]({name})", "",
                         f"[Publication PDF: {pdf_name}]({pdf_name})", ""]
    text += ["## Screening, selected settings, and inference", "",
        "`search_candidates.csv` retains every completed stage row, including ineligible candidates and their failure/qualification counts. "
        "`candidate_registry.json` contains the exact original proposal roster and `selected_configurations.json` the frozen choices. "
        "The proposal budget is 40 candidates for A,40 for B,24 for C, including reference and prior-sensitivity strata. "
        "The staged plan promotes four preserved-prior candidates plus one prior-sensitivity candidate and the old selected configuration. "
        "Predictive Monte Carlo repetitions assess simulation variability on fixed validation cases; this is different from sampler convergence "
        "or trajectory-sampling uncertainty. Near-tie selection follows the frozen precision rule rather than choosing the smallest noisy point estimate.", "",
        "`realized_dynamics.json` records each configured recurrent operator norm and the actual operator norm and eigenvalue spectral radius "
        "measured on training-only feature maps. Those two quantities are not interchangeable. It also records reservoir sizes, leak patterns, "
        "input scales, connectivity, PCA modes, lags, square features, conditioning length, and readout size. "
        "History synchronization is an empirical training-endpoint check, not a universal stability guarantee.", "",
        "`fit_diagnostics.json` records qualification, actual continuations, and extrema of the final monitored coefficient, hyperparameter and "
        "predictive-functional diagnostics. The established four-chain R-hat/ESS gate is unchanged. MCMC qualification does not imply accurate "
        "forecasts. Native predictive paths hold a parameter draw fixed along the entire path and propagate their own input-site predictions.", "",
        "| Track / stratum | Candidates | Configured operator norm range | Realized eigenvalue-radius range | Reservoir units | Input-scale range | Leak range |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["realized_parameter_ranges"]:
        def show(key):
            return "—" if row[key] is None else "–".join(format_number(x) for x in row[key])
        text.append(f"| {row['track']} / {row['stratum']} | {row['candidates']} | {show('configured_operator_norm')} | "
                    f"{show('realized_spectral_radius')} | {show('reservoir_units')} | {show('input_scale')} | {show('leak')} |")
    text += ["", "| Track / role | Selected candidate | Reservoir units | Operator norm | Leak pattern(s) | Input scale(s) | PCA modes / squares | Conditioning rows |",
             "|---|---|---:|---:|---|---|---:|---:|"]
    for row in report["selected_parameter_summary"]:
        f = row["features"]
        text.append(f"| {row['track']} / {row['role']} | {row['candidate_id']} | {f['units']} | {format_number(f['recurrent_norm'])} | "
                    f"{f['leak_patterns']} | {f['input_scales']} | {f['modes']} / {f['squares']} | {row['conditioning']} |")
    text += ["",
        "### Which individual controls changed screen performance?", "",
        "These candidate-minus-original-anchor changes use the same two trajectories and four validation windows per trajectory. "
        "They are exploratory selection diagnostics (n=2), not confirmation estimates or causal effects. "
        "Joint proposals and prior sensitivities are retained separately in `controlled_screen_contrasts.csv` and are not mixed into this table.", "",
        "| Track | Candidate alias | Changed effective fields | Screen CRPS | ΔCRPS vs old | Screen MSE | ΔMSE vs old | Paired n |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in report["controlled_screen_contrasts"]:
        if row["category"] == "controlled_single_factor":
            text.append(f"| {row['track']} | {', '.join(row['aliases'])} | {', '.join(row['changed_effective_fields'])} | "
                        f"{format_number(row['screen_band_crps'])} | {format_number(row['delta_band_crps'])} | "
                        f"{format_number(row['screen_band_mse'])} | {format_number(row['delta_band_mse'])} | {len(row['paired_trajectories'])} |")
    text += [""] + seed_sensitivity_markdown(seed_rows, seed_pairs)
    if report.get("selected_invariance"):
        receipt = report["selected_invariance"]
        text += ["## Selected-model acceptance", "",
                 f"The frozen-selected invariance receipt reports {receipt.get('passed_cases')} passing cases out of {receipt.get('expected_cases')} scheduled, "
                 f"with {receipt.get('failed_cases')} failed; complete={receipt.get('complete')}, passed={receipt.get('passed')}. "
                 "It checks the fixed-fit model-output boundary and separately refitted training-only preprocessing, without rerunning selection "
                 "or fitting a new posterior. Its experiment/freeze identities and exact file hash are bound in the source manifest.", ""]
    if diagnosis:
        text += ["## Initial fixed-fit diagnostic", "",
            "Before this search, the retained original A/trajectory0 fit was examined at the two prescribed validation origins6000 and6500. "
            "No new fit or test labels were used. The complete predictive distribution was compared with parameter-only paths, a fixed-parameter "
            "innovation experiment, and a deterministic conditional-readout rollout. These interventions are diagnostics, not interchangeable "
            "posterior predictive distributions or additive variance components.", "",
            "| Regime | RMSE at first target6000 | RMSE at first target6500 |", "|---|---:|---:|"]
        cases = diagnosis.get("cases", [])
        if isinstance(cases, dict):
            cases = list(cases.values())
        cases = sorted(cases, key=lambda x: x["first_target"])
        modes = list(cases[0]["modes"]) if cases else []
        for mode in modes:
            vals = [format_number(c["modes"][mode]["scores"].get("rmse")) for c in cases]
            text.append(f"| {mode} | " + " | ".join(vals) + " |")
        text += ["", "Low teacher-forced one-step error coexisted with large autonomous deterministic error. Removing innovations did not "
            "eliminate long-horizon spread or establish better forecasts. This supports investigating feedback and mean dynamics; it does not "
            "prove a unique cause. The two origins share one trajectory and were already part of validation evidence. Their stochastic score "
            "differences are not a new statistically independent ranking.", ""]
    budget = report["budget"]
    extension = budget.get("operational_extension")
    extension_text = []
    if extension:
        if (extension.get("one_prospective_extension") is not True
                or extension.get("frozen_science_unchanged") is not True
                or extension.get("confirmation_scores_used_for_decision") is not False
                or extension["original_campaign_started_utc"] != budget["campaign_started_utc"]
                or extension["search_worker_seconds_cap_unchanged"] != budget["search_worker_seconds_cap"]):
            raise ValueError("Operational budget extension lacks its unchanged-science/time boundary")
        extension_text = ["### One prospective operational budget extension", "",
            f"The original executed policy allowed {extension['original_total_worker_seconds_cap'] / 3600:g} total worker hours "
            f"and {extension['original_wall_seconds_cap'] / 3600:g} elapsed hours. The active extension allows "
            f"{budget['total_worker_seconds_cap'] / 3600:g} total worker hours and {budget['wall_seconds_cap'] / 3600:g} elapsed hours; "
            f"the search threshold remains {extension['search_worker_seconds_cap_unchanged'] / 3600:g} worker hours. "
            f"The original start `{extension['original_campaign_started_utc']}` and every recorded attempt remain in the cumulative budget. "
            "This single prospective operational decision used receipt timing and remaining-work estimates, not confirmation performance scores. "
            "It does not reopen selection, modify the scientific configuration or remove failures. Migration rebinds experiment/freeze identities "
            "while retaining the frozen choices and original freeze timestamp.", "",
            f"Recorded reason: {budget['extension_reason']}", "",
            f"Authorization recorded (UTC): `{extension['authorized_utc']}`. "
            f"{extension.get('authorization_timestamp_scope', '')} "
            f"Timing-only receipt SHA-256: `{extension['timing_receipt_sha256']}`. "
            f"Original frozen-selection identity: `{extension['frozen_selection_identity']}`. "
            f"Previous configuration SHA-256: `{extension['previous_configuration_sha256']}`. "
            "The complete prior budget is retained in `report.json` under `budget.operational_extension.previous_budget`.", ""]
    point_references = [r for r in cohorts if "cheap_anchor" in r.get("roles", [])]
    auxiliary_text = [
        "Fixed point-reference evaluations and selected-artifact checks are separate auxiliary work, outside the main campaign's "
        "worker-attempt ledger. The reference tool has a separate 900-second default invocation scheduling threshold, checked before "
        "each whole trajectory/model job; an admitted job may finish beyond it. The original selected point-reference hyperparameters "
        "are fixed, with C transferring A's configuration for every included family and refitting permitted C training data only.", ""]
    if point_references:
        auxiliary_text += [f"Recorded auxiliary reference fit time: {sum(r['fit_seconds'] for r in point_references):.2f} seconds. "
            f"Recorded auxiliary reference forecast time: {sum(r['forecast_seconds'] for r in point_references):.2f} seconds. "
            "These receipt sums include available recorded work and exclude orchestration overhead; they are not end-to-end invocation times.", ""]
    acceptance = report.get("selected_invariance")
    if acceptance and acceptance.get("last_invocation_seconds") is not None:
        auxiliary_text += [f"Latest recorded selected-artifact acceptance invocation: {acceptance['last_invocation_seconds']:.2f} seconds, "
            "separately from the main campaign ledger. This is a last-invocation field, not a cumulative timing of unrecorded earlier attempts.", ""]
    text += ["## Compute, provenance, and operational recovery", "",
        f"Attempt worker wall time recorded at this snapshot: {report['operations']['attempt_worker_seconds'] / 3600:.3f} hours, "
        f"plus {report['operations']['preparation_worker_seconds']:.1f} recorded preparation seconds. "
        "Fit receipts are deduplicated by fitted-model identity; the fit and forecast costs in each cohort are shown separately in `confirmation.csv`. "
        "Search, precision and seed runs have distinct stage totals in `operations.json`. Worker wall seconds are not CPU seconds. "
        "Fit-receipt peak RSS is a per-process high-water mark, not peak simultaneous campaign memory.", "",
        *auxiliary_text,
        f"The active configured thresholds are {report['budget']['total_worker_seconds_cap'] / 3600:g} total worker hours, "
        f"{report['budget']['search_worker_seconds_cap'] / 3600:g} search worker hours and "
        f"{report['budget']['wall_seconds_cap'] / 3600:g} elapsed hours from the original campaign start "
        f"`{report['budget']['campaign_started_utc']}`. These are scheduling thresholds: they are checked before admitting work and do not terminate "
        f"already-running jobs at the aggregate boundary. A group of up to {report['budget']['workers']} jobs, each with a "
        f"{report['budget']['job_timeout_seconds']:g}-second timeout, can overrun a nominal worker threshold by up to "
        f"{report['budget']['workers'] * report['budget']['job_timeout_seconds']:g} worker-seconds plus shutdown overhead. "
        "The final observed consumption must be compared with the "
        "nominal thresholds separately; the existence of this allowance does not establish that an overrun occurred.", "",
        *extension_text,
        *migration_markdown(report['operations']),
        "Bounded launcher memory recovery, where recorded, uses cache advice only after terminal memory-admission failures and a proved "
        "closed-worker state. It does not change model code, results, numerical gates or campaign budgets. Public receipts redact private paths. "
        "A partial snapshot does not imply the remaining stages completed.", "",
        f"Experiment identity: `{report['experiment_identity']}`. Freeze identity: `{report['freeze_identity']}`. "
        "`source_manifest.json` binds every consumed results receipt, and `artifact_manifest.json` binds all generated outputs. "
        "The report builder never loads a posterior or reads a KS data file.", "",
        "## Reproduction", "", "```bash", "# Existing project environment and authorized verified cache; one numerical thread.",
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
        "export PYTHONPATH=src", "python tools/build_ks_improvement_report.py \\",
        "  --results RUN_ROOT \\", "  --output docs/applications/improvement_results", "```", "",
        "The separate anchor tool requires explicit execution and a supplied original freeze; its dry preparation does not fit models. "
        "The original campaign runner owns all fits, selection and resume. Rebuilding this report cannot change selection or schedule new jobs.", "",
        "## Primary references", ""]
    for title, url in REFERENCES:
        text.append(f"- [{title}]({url}).")
    return "\n".join(text) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchors", type=Path)
    parser.add_argument("--operations", type=Path)
    parser.add_argument("--acceptance", type=Path,
                        help="Optional frozen-selected invariance receipt; defaults to RESULTS/../acceptance/selected_invariance.json")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    reader = ReceiptReader(args.results)
    if output == reader.root or reader.root in output.parents:
        raise ValueError("Report outputs must be separate from immutable scientific receipts")
    output.mkdir(parents=True, exist_ok=True)
    selections, search_rows = search_evidence(reader)
    controls = controlled_contrasts(reader, selections)
    dynamics = dynamics_evidence(reader)
    ranges = dynamics_ranges(dynamics)
    cohorts, comparisons, seed_rows, seed_pairs = [], [], [], []
    if reader.freeze:
        for track, chosen in reader.freeze["selected"].items():
            roles = {}
            for role, cid in (("anchor", chosen["anchor_id"]), ("main", chosen["main"]["id"]),
                              ("sensitivity", chosen["sensitivity"]["id"] if chosen.get("sensitivity") else None)):
                if cid:
                    roles.setdefault(cid, []).append(role)
            current = {}
            for cid, names in roles.items():
                row = reader.cohort(track, cid, CONFIRMATION, ORIGINS, 512, reader.cfg["model_seed"], "confirm", names)
                cohorts.append(row); current[cid] = row
            for left_id, right_id, purpose in ((chosen["main"]["id"], chosen["anchor_id"], "preserved_prior_vs_old"),
                    (chosen["sensitivity"]["id"] if chosen.get("sensitivity") else None, chosen["anchor_id"], "prior_sensitivity_vs_old"),
                    (chosen["sensitivity"]["id"] if chosen.get("sensitivity") else None, chosen["main"]["id"], "prior_sensitivity_vs_preserved_prior")):
                if left_id:
                    comparisons.extend({"track": track, "purpose": purpose, **x} for x in paired_comparisons(current[left_id], current[right_id]))
            sensitivity = reader.cfg["sensitivity"]
            for cid in dict.fromkeys([chosen["main"]["id"], chosen["anchor_id"]]):
                base = reader.cohort(track, cid, sensitivity["trajectories"], sensitivity["origins"], 512,
                                    reader.cfg["model_seed"], "matched_main_seed", roles[cid])
                seed_rows.append(base)
                for seed in sensitivity["seeds"]:
                    alternate = reader.cohort(track, cid, sensitivity["trajectories"], sensitivity["origins"], 512,
                                             seed, "sensitivity", roles[cid])
                    seed_rows.append(alternate)
                    seed_pairs.extend({"track": track, **x} for x in paired_comparisons(alternate, base))
    anchor_evidence = None
    if args.anchors:
        point_rows, anchor_evidence = anchor_cohorts(reader, args.anchors)
        for row in point_rows:
            chosen = next(r for r in cohorts if r["track"] == row["track"] and "main" in r["roles"])
            comparisons.extend({"track": row["track"], "purpose": "preserved_prior_vs_fixed_point_anchor", **x}
                               for x in paired_comparisons(chosen, row, metrics=("rmse", "nrmse", "mae")))
        cohorts.extend(point_rows)
    # Also include every actual fit once in diagnostic and compute summaries.
    for job in reader.jobs.values():
        if job["stage"] != "preflight":
            reader.fit(job["track"], job["candidate"]["id"], job["trajectory"], job["seed"])
    operations_root = args.operations or reader.root.parent / "operations/bounded_launches"
    operations = operation_evidence(reader, operations_root)
    complete = bool(reader.freeze and cohorts and seed_rows and all(r["all_scheduled_jobs_terminal"] for r in cohorts + seed_rows))
    repository = Path(__file__).resolve().parents[1]
    diagnosis_path = repository / "docs/applications/evidence/ks_improvement_initial_diagnosis.json"
    diagnosis = json.loads(diagnosis_path.read_text()) if diagnosis_path.exists() else None
    acceptance_path = args.acceptance or reader.root.parent / "acceptance/selected_invariance.json"
    acceptance = None
    if acceptance_path.exists():
        raw = json.loads(acceptance_path.read_text())
        if (not reader.freeze or raw["experiment_identity"] != reader.experiment["identity"]
                or raw["freeze_identity"] != reader.freeze["identity"]
                or raw["new_posterior_fits"] != 0 or raw["selection_rerun"] is not False
                or raw["test_partition_opened"] is not False):
            raise ValueError("Selected-model acceptance receipt has incompatible provenance")
        acceptance = {k: raw.get(k) for k in ("schema", "identity", "experiment_identity", "freeze_identity", "policy",
            "expected_cases", "passed_cases", "failed_cases", "complete", "passed", "new_posterior_fits",
            "selection_rerun", "test_partition_opened", "reader_scope", "last_invocation_seconds")}
        acceptance["sha256"] = sha256(acceptance_path)
    figures = [] if args.no_plots else plot_report(reader, cohorts, search_rows, output)
    vector_figures = [str(Path(name).with_suffix(".pdf")) for name in figures]
    manifest = {"schema": "ks-improvement-report-source-manifest/1", "experiment_identity": reader.experiment["identity"],
                "results_receipts": reader.manifest, "builder_sha256": sha256(__file__),
                "initial_diagnosis_sha256": sha256(diagnosis_path) if diagnosis_path.exists() else None,
                "anchors": anchor_evidence, "selected_invariance": acceptance,
                "operational_public_evidence": {key: {k: v for k, v in value.items() if k != "data"}
                                                for key, value in operations["public_evidence"].items()}}
    manifest["identity"] = content_hash(manifest)
    selected_parameters = []
    if reader.freeze:
        for track, chosen in reader.freeze["selected"].items():
            for role, cid in (("anchor", chosen["anchor_id"]), ("main", chosen["main"]["id"]),
                              ("sensitivity", chosen["sensitivity"]["id"] if chosen.get("sensitivity") else None)):
                if cid:
                    candidate = reader.candidates[(track, cid)]
                    selected_parameters.append({"track": track, "role": role, "candidate_id": cid,
                        "stratum": candidate["stratum"], "features": candidate["config"]["features"],
                        "conditioning": candidate["config"]["conditioning"],
                        "prior": candidate["config"]["prior"], "spatial_mean": candidate["config"]["spatial_mean"],
                        "covariance": candidate["config"]["covariance"], "cost": candidate["cost"]})
    stage_states = {}
    for job_id, job in reader.jobs.items():
        state = reader.terminals.get(job_id, {}).get("state", "pending")
        stage_states.setdefault(job["stage"], {})[state] = stage_states.setdefault(job["stage"], {}).get(state, 0) + 1
    report = {"schema": SCHEMA, "created_utc": utc(), "status": "complete" if complete else "incomplete",
              "scope": "validation-only exploratory post-test follow-up; not a fresh test set",
              "experiment_identity": reader.experiment["identity"],
              "freeze_identity": reader.freeze["identity"] if reader.freeze else None,
              "source_manifest_identity": manifest["identity"], "test_partition_reads_by_builder": 0,
              "tracks": reader.cfg["tracks"], "history": reader.history["history"],
              "confirmation_roster": CONFIRMATION, "confirmation_origins": ORIGINS,
              "primary_rmse": "mean across trajectories of sqrt(pooled within-trajectory squared error)",
              "confirmation": [{k: v for k, v in r.items() if k not in ("episodes", "curves")} for r in cohorts],
              "paired_comparisons": comparisons, "operations": operations, "figures": figures,
              "vector_figures": vector_figures,
              "figure_export_policy": "PNG and PDF from the same matplotlib figure; PDF axes, text and paths are vector, while heatmaps retain image layers",
              "controlled_screen_contrasts": controls,
              "realized_parameter_ranges": ranges, "selected_parameter_summary": selected_parameters,
              "immutable_job_states_by_stage": stage_states,
              "planned_initial_candidate_counts": {t: len(reader.registry["tracks"][t]["candidates"]) for t in reader.cfg["tracks"]},
              "budget": reader.cfg["budget"], "numerical_source": reader.experiment["source"]}
    report["selected_invariance"] = acceptance
    files = {"report.json": report, "source_manifest.json": manifest,
             "candidate_registry.json": reader.registry,
             "selected_configurations.json": reader.freeze,
             "search_stages.json": selections, "realized_dynamics.json": dynamics,
             "operations.json": operations, "fit_diagnostics.json": list(reader.fits.values()),
             "episode_results.json": [{"track": r["track"], "candidate_id": r["candidate_id"], "seed": r["seed"], **e}
                                      for r in cohorts for e in r["episodes"]],
             "horizon_curves.json": [{"track": r["track"], "candidate_id": r["candidate_id"], "seed": r["seed"],
                                      "complete_trajectory_ids": r["complete_trajectory_ids"], "curves": r["curves"]} for r in cohorts],
             "seed_sensitivity.json": {"experiment_identity": reader.experiment["identity"],
                                      "freeze_identity": report["freeze_identity"], "rows": seed_rows, "paired_differences": seed_pairs}}
    for name, value in files.items():
        write_json(output / name, value)
    write_csv(output / "search_candidates.csv", search_rows, ("stage", "track", "plot_index", "candidate_id", "stratum", "eligible",
              "band_crps", "band_mse", "qualified_episodes", "expected_episodes", "failed_origins", "pending_origins", "unqualified_fits", "preflight_exclusion", "effective_hash"))
    write_csv(output / "controlled_screen_contrasts.csv", controls, ("track", "candidate_id", "anchor_id", "category", "stratum", "aliases",
              "changed_effective_fields", "candidate_eligible", "anchor_eligible", "screen_band_crps", "delta_band_crps", "screen_band_mse", "delta_band_mse", "paired_trajectories", "interpretation"))
    flat = []
    for row in cohorts:
        flat.append({k: row[k] for k in ("track", "candidate_id", "roles", "stratum", "seed", "scheduled_trajectories",
            "complete_trajectories", "scheduled_origins", *COUNTS, "finite_divergent_origins", "score_population_origins",
            "finite_origins_outside_score_population", "complete_case_pooled_rmse", "full_schedule_point_score_defined",
            "fits_available", "qualified_fits", "fit_seconds", "forecast_seconds", "peak_fit_process_rss_bytes")} |
            {f"{m}_{k}": row["metrics"][m].get(k) for m in METRICS for k in ("mean", "ci95", "n")})
    write_csv(output / "confirmation.csv", flat, list(flat[0]) if flat else ("track", "candidate_id", "complete_trajectories"))
    write_csv(output / "paired_comparisons.csv", comparisons, ("track", "purpose", "left", "right", "left_seed", "right_seed", "metric", "n", "mean", "ci95", "paired_trajectories"))
    atomic_bytes(output / "README.md", render_markdown(report, cohorts, comparisons, figures, diagnosis,
                 seed_rows=seed_rows, seed_pairs=seed_pairs).encode())
    artifact_files = list(files) + ["search_candidates.csv", "controlled_screen_contrasts.csv", "confirmation.csv", "paired_comparisons.csv", "README.md"] + figures + vector_figures
    artifact_manifest = {"schema": "ks-improvement-report-artifacts/1", "experiment_identity": report["experiment_identity"],
                         "freeze_identity": report["freeze_identity"], "source_manifest_identity": manifest["identity"],
                         "files": {name: {"bytes": (output / name).stat().st_size, "sha256": sha256(output / name)} for name in artifact_files}}
    artifact_manifest["identity"] = content_hash(artifact_manifest)
    write_json(output / "artifact_manifest.json", artifact_manifest)
    print(json.dumps({"status": report["status"], "cohorts": len(cohorts), "complete_confirmation_episodes": sum(r["complete_trajectories"] for r in cohorts),
                      "source_receipts": len(reader.manifest), "artifact_identity": artifact_manifest["identity"], "output": str(output)}))


if __name__ == "__main__":
    main()
