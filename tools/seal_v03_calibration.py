"""Verify and seal an already saved calibration study without refitting.

The only new output is calibration_evidence.json. Base parameters, simulated
observations, saved forecasts, maps and bands are read, never regenerated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import zipfile

import numpy as np

from bayes_stesn.benchmark.contracts import Forecast
from bayes_stesn.benchmark.generators import Simulation
from bayes_stesn.benchmark.runner import source_fingerprint
from bayes_stesn.io import content_hash, load_state

from render_v03_study import Inputs, _same_number, load_run, safe_member, sha256, write_json


def finite_mean(value):
    result = float(np.mean(value))
    return result if np.isfinite(result) else None


def verify_numbers(summary, forecast, truth, calibration, bands):
    """Independent scalar/band reductions of saved arrays; no model methods."""
    truth = np.asarray(truth, float)
    if truth.shape != forecast.mean.shape:
        raise ValueError("Saved calibration truth and forecast shapes differ")
    error = forecast.mean - truth
    for key, expected in (("rmse", np.sqrt(np.mean(error**2))), ("mae", np.mean(np.abs(forecast.median - truth)))):
        _same_number(summary["raw"].get(key), float(expected), f"Calibration raw {key}")
    if not np.allclose(summary["raw"]["rmse_by_lead"], np.sqrt(np.mean(error**2, axis=1)), rtol=1e-10, atol=1e-12):
        raise ValueError("Calibration raw leadwise RMSE differs from saved arrays")
    samples = forecast.samples
    if samples is not None:
        ordered = np.sort(samples, axis=0)
        count = len(samples)
        weights = (2 * np.arange(1, count + 1) - count - 1) / count**2
        crps = np.mean(np.abs(samples - truth), axis=0) - np.tensordot(weights, ordered, axes=(0, 0))
        _same_number(summary["raw"].get("crps"), float(crps.mean()), "Calibration raw empirical CRPS")
        del ordered, crps
        for level, row in summary["raw"].get("intervals", {}).items():
            alpha = 1 - float(level)
            lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2], axis=0)
            covered = (lo <= truth) & (truth <= hi)
            loss = hi - lo + (2 / alpha) * (np.maximum(lo - truth, 0) + np.maximum(truth - hi, 0))
            for key, value in (("coverage", covered), ("width", hi - lo), ("interval_score", loss)):
                _same_number(row.get(key), finite_mean(value), f"Raw {level} {key}")
    center = forecast.median
    scale = (np.full(center.shape, calibration["training_scale"]) if samples is None or len(samples) < 2
             else np.maximum(np.std(samples, axis=0, ddof=1), .01 * calibration["training_scale"]))
    if calibration["transfer"]:
        distance = np.sum((forecast.coordinates[:, None, :] - np.asarray(calibration["coordinates"])[None, :, :])**2, axis=2)
        nearest = np.argmin(distance, axis=1)
    elif tuple(calibration["site_ids"]) != forecast.site_ids:
        raise ValueError("Observed-site calibration target identities changed")
    if set(bands) != set(calibration["maps"]) or set(summary["calibrated"]) != set(bands):
        raise ValueError("Calibration map, band and summary keys disagree")
    marginal_losses = []
    for key, mapping in calibration["maps"].items():
        quantiles = np.asarray(mapping["quantiles"])
        if calibration["transfer"] and mapping["target"] == "marginal":
            quantiles = quantiles[:, nearest]
        radius = quantiles * np.maximum(scale, mapping["scale_floor"])
        lo, hi = bands[key]
        if not (np.allclose(lo, center - radius, rtol=1e-11, atol=1e-12)
                and np.allclose(hi, center + radius, rtol=1e-11, atol=1e-12)):
            raise ValueError("Saved bands do not reproduce from the fixed calibration map")
        covered, width = (lo <= truth) & (truth <= hi), hi - lo
        loss = width + (2 / mapping["alpha"]) * (np.maximum(lo - truth, 0) + np.maximum(truth - hi, 0))
        row = summary["calibrated"][key]
        for name, value in (("coverage", covered), ("width", width), ("interval_score", loss),
                            ("field_coverage", np.all(covered, axis=1))):
            _same_number(row.get(name), finite_mean(value), f"Calibrated {key} {name}")
        if row["whole_path_covered"] != bool(np.all(covered)) or row["infinite_band"] != bool(np.isinf(width).any()):
            raise ValueError("Calibration coverage/infinite-band flags disagree")
        if (mapping["blocks"] != calibration["blocks"] or row["blocks"] != calibration["blocks"]
                or row["rank"] != int(np.ceil((calibration["blocks"] + 1) * (1 - mapping["alpha"])))):
            raise ValueError("Calibration finite-block count/rank differs")
        if mapping["target"] == "marginal":
            marginal_losses.append((mapping["alpha"] / 2) * loss)
    wis = (.5 * np.abs(truth - center) + sum(marginal_losses)) / (len(marginal_losses) + .5)
    _same_number(summary.get("wis_marginal"), finite_mean(wis), "Calibrated marginal WIS")


def seal(run_directory, calibration_directory, *, replication=0, max_uncompressed_mb=512):
    if not isinstance(replication, int) or isinstance(replication, bool) or replication < 0:
        raise ValueError("replication must be a nonnegative integer")
    if not np.isfinite(max_uncompressed_mb) or not 0 < max_uncompressed_mb <= 1024:
        raise ValueError("Use an aggregate saved-array allowance between 0 and 1024 MiB")
    source, inputs = source_fingerprint(), Inputs()
    run = load_run(run_directory, "base", inputs, source)
    directory = Path(calibration_directory).resolve()
    summary = inputs.json(directory / "summary.json")
    if (summary.get("base_manifest_identity") != run["manifest"]["identity"]
            or summary.get("distribution_added") is not False or summary.get("raw_samples_unchanged") is not True):
        raise ValueError("Calibration summary does not identify its unchanged base predictive law")
    key = (summary["scenario"], summary["track"], summary["model_id"], replication)
    matched = [r for r in run["receipts"] if (r["scenario"], r["track"], r["model_id"], r["replication"]) == key]
    if len(matched) != 1 or not matched[0]["qualified"] or matched[0]["fallback"]:
        raise ValueError("Calibration needs its explicitly identified qualified native base fit")
    receipt = matched[0]
    base = Path(run["path"]) / key[0] / f"rep_{replication:03d}"
    model = base / key[1] / key[2] / "model.npz"
    if not model.is_file():
        raise ValueError("Saved base model is required for provenance binding")
    names = ("simulation.npz", "raw_forecast.npz", "calibration.npz", "bands.npz")
    paths = [inputs.add(safe_member(directory, name)) for name in names]
    archives = [base / "simulation.npz", *paths]
    uncompressed = 0
    for archive in archives:
        with zipfile.ZipFile(archive) as stored:
            uncompressed += sum(item.file_size for item in stored.infolist())
    if uncompressed > max_uncompressed_mb * 1024**2:
        raise MemoryError("Saved calibration arrays exceed the explicit aggregate validation allowance")
    original, extended = Simulation.load(base / "simulation.npz"), Simulation.load(directory / "simulation.npz")
    if (not np.array_equal(original.observation, extended.observation[:len(original.observation)])
            or not np.array_equal(original.coordinates, extended.coordinates)
            or not np.array_equal(original.times, extended.times[:len(original.times)])
            or original.site_ids != extended.site_ids
            or extended.metadata["seed"] != receipt["dgp_seed"]
            or any(not np.array_equal(getattr(original, role), getattr(extended, role))
                   for role in ("observed_indices", "calibration_indices", "test_indices"))):
        raise ValueError("Calibration extension did not preserve its exact original field/history")
    forecast = Forecast.load(directory / "raw_forecast.npz")
    calibration = load_state(directory / "calibration.npz", kind="benchmark_calibration")
    bands = load_state(directory / "bands.npz", kind="calibrated_bands")
    origin, horizon = summary["origin_index"], forecast.mean.shape[0]
    if horizon != run["manifest"]["protocol"]["horizon"]:
        raise ValueError("Calibration forecast horizon differs from the declared base design")
    targets = extended.observed_indices if key[1] == "A" else extended.test_indices
    expected_ids = tuple(extended.site_ids[i] for i in targets)
    if forecast.site_ids != expected_ids or not np.array_equal(forecast.coordinates, extended.coordinates[targets]):
        raise ValueError("Calibration evaluation uses changed targets or coordinate order")
    origins = summary["calibration_origins"]
    if (summary["parameter_cutoff"] != receipt["origin_index"] or origins != list(calibration["origins"])
            or not origins or origins[0] < summary["parameter_cutoff"]
            or any(b - a < horizon for a, b in zip(origins, origins[1:]))
            or origins[-1] + horizon > origin or summary["blocks"] != len(origins)
            or calibration["blocks"] != len(origins) or calibration["horizon"] != horizon
            or origin + horizon > len(extended.observation)):
        raise ValueError("Calibration and evaluation windows overlap or disagree with their fixed parameter cutoff")
    if bool(calibration["transfer"]) != (key[1] == "B"):
        raise ValueError("Calibration target-transfer policy differs from its task")
    expected_calibration_ids = tuple(extended.site_ids[i] for i in (extended.calibration_indices if key[1] == "B" else extended.observed_indices))
    if tuple(calibration["site_ids"]) != expected_calibration_ids:
        raise ValueError("Calibration responses did not use the declared O/C roles")
    if key[1] == "B" and (set(calibration["site_ids"]) & set(expected_ids)):
        raise ValueError("Reserved calibration responses overlap final U targets")
    verify_numbers(summary, forecast, extended.observation[origin:origin + horizon, targets], calibration, bands)
    inputs.verify_unchanged()
    base_files = [Path(run["path"]) / "manifest.json", Path(run["path"]) / "summary.json", model,
                  base / "simulation.npz", base / "data_receipt.json", model.with_name("receipt.json")]
    entry = lambda path, root: {"file": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)}
    evidence = {"schema": "v03-calibration-evidence-1.0", "complete": True,
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_fingerprint": source,
        "base_manifest_identity": run["manifest"]["identity"], "base_receipt_identity": receipt["receipt_identity"],
        "scenario": key[0], "track": key[1], "model_id": key[2], "replication": replication,
        "base_inputs": [entry(path, Path(run["path"])) for path in base_files],
        "files": [entry(directory / name, directory) for name in ("summary.json", *names)],
        "verification": {"original_prefix_exact": True, "same_late_evaluation_targets": True,
            "disjoint_calibration_blocks": True, "parameter_cutoff": True, "O_C_U_roles": True,
            "saved_raw_rmse_mae_crps_and_intervals": True, "saved_map_bands_and_calibrated_scores": True,
            "fitted_models_loaded": False, "fits_or_forecasts_performed": False,
            "saved_arrays_uncompressed_bytes": uncompressed}, "sealer_sha256": sha256(Path(__file__))}
    evidence["identity"] = content_hash(evidence)
    write_json(directory / "calibration_evidence.json", evidence)
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--replication", default=0, type=int)
    parser.add_argument("--max-uncompressed-mb", default=512, type=float)
    args = parser.parse_args(argv)
    value = seal(args.run, args.calibration, replication=args.replication, max_uncompressed_mb=args.max_uncompressed_mb)
    print(value["identity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
