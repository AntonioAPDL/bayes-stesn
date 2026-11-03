#!/usr/bin/env python3
"""Audit saved development references and compute the G1 known-parameter oracle.

No fitted comparator is created, loaded, retuned, or forecast again. The oracle
receives only noisy observed-site history available at the saved final origin.
Future observations are read afterward for evaluation. Optional FHN validation
calls the existing independent generator validator; it is not a forecasting fit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm
from threadpoolctl import threadpool_limits

from bayes_stesn.benchmark.contracts import Forecast
from bayes_stesn.benchmark.generators import Simulation, gaussian_ar_oracle_moments
from bayes_stesn.benchmark.protocol import BenchmarkProtocol, effective_models
from bayes_stesn.benchmark.runner import source_fingerprint, write_json
from bayes_stesn.io import content_hash, save_state


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_json(path, identity_key=None):
    value = json.loads(Path(path).read_text())
    if identity_key is not None:
        require(value.get(identity_key) == content_hash(
            {key: item for key, item in value.items() if key != identity_key}),
            f"Content identity mismatch: {path}")
    return value


def archive_admission(path, maximum_bytes):
    with zipfile.ZipFile(path) as archive:
        size = sum(item.file_size for item in archive.infolist())
    require(size <= maximum_bytes, f"Archive exceeds the declared uncompressed audit limit: {path}")


def gaussian_scores(mean, variance, truth):
    """Analytic marginal Gaussian CRPS; no empirical-path approximation."""
    mean, variance, truth = (np.asarray(value, float) for value in (mean, variance, truth))
    require(mean.shape == variance.shape == truth.shape and mean.ndim == 2,
            "Oracle score arrays must align on [lead, site]")
    require(all(np.isfinite(value).all() for value in (mean, variance, truth))
            and np.all(variance >= 0), "Invalid Gaussian oracle moments")
    sd = np.sqrt(variance)
    error = truth - mean
    z = np.divide(error, sd, out=np.zeros_like(error), where=sd > 0)
    density = np.exp(-.5 * z*z) / np.sqrt(2*np.pi)
    crps = sd * (z * (2*ndtr(z)-1) + 2*density - 1/np.sqrt(np.pi))
    crps = np.where(sd > 0, crps, np.abs(error))
    intervals = {}
    for level in (.5, .8, .9, .95):
        radius = float(norm.ppf((1+level)/2)) * sd
        intervals[str(level)] = {
            "coverage": float(np.mean(np.abs(error) <= radius)),
            "width": float(np.mean(2*radius)),
        }
    return {"rmse": float(np.sqrt(np.mean(error*error))),
            "mae": float(np.mean(np.abs(error))), "crps": float(np.mean(crps)),
            "rmse_by_lead": np.sqrt(np.mean(error*error, axis=1)).tolist(),
            "crps_by_lead": np.mean(crps, axis=1).tolist(),
            "conditional_expected_mse": float(np.mean(variance)),
            "crps_estimator": "analytic marginal Gaussian CRPS", "intervals": intervals}


def oracle_moments(simulation, protocol, track):
    """Information boundary: no latent state, U history, or future O input."""
    require(simulation.metadata["scenario"] == "g1_gaussian_ar", "The oracle is specific to G1")
    require(track in {"A", "B"}, "Oracle track must be A or B")
    require(len(simulation.coordinates) <= 1024 and len(simulation.observed_indices) <= 256,
            "This bounded oracle audit supports at most 256 observed and 1024 domain sites")
    targets = (simulation.observed_indices[:protocol.score_observed_count]
               if track == "A" else simulation.test_indices)
    history = simulation.observation[protocol.training_start:protocol.final_origin,
                                     simulation.observed_indices]
    parameters = simulation.metadata["parameters"]
    require(parameters.get("stationary_initialization") is True
            and parameters.get("matern_smoothness") == 1.5,
            "Saved G1 is outside the declared stationary Matérn-3/2 oracle")
    result = gaussian_ar_oracle_moments(
        history, simulation.coordinates[simulation.observed_indices],
        simulation.coordinates[targets], protocol.horizon,
        **{key: parameters[key] for key in ("rho", "ell", "sigma_e")})
    result.update(target_ids=[simulation.site_ids[j] for j in targets],
                  origin_index=protocol.final_origin, history_start=protocol.training_start,
                  history_stop=protocol.final_origin,
                  history_hash=content_hash({"observations": history,
                      "site_ids": [simulation.site_ids[j] for j in simulation.observed_indices],
                      "times": simulation.times[protocol.training_start:protocol.final_origin]}))
    return result, targets


def audit_normalization(simulation, protocol, receipt):
    reference = receipt["normalization_reference"]
    require(reference["identity"] == content_hash(
        {key: value for key, value in reference.items() if key != "identity"}),
        "Normalization identity mismatch")
    start, stop, count = (reference[key] for key in ("start", "stop", "observed_count"))
    require(protocol.training_start <= start < stop <= protocol.initial_fit,
            "Normalization uses responses outside the permitted initial history")
    require(1 <= count <= protocol.n_observed and reference["ddof"] in (0, 1),
            "Invalid normalization population/convention")
    ids = [simulation.site_ids[j] for j in simulation.observed_indices[:count]]
    values = simulation.observation[start:stop, simulation.observed_indices[:count]]
    require(ids == reference["site_ids"] and values.size == reference["sample_count"],
            "Normalization sites or count changed")
    expected_hash = content_hash({"observations": values, "site_ids": tuple(ids),
                                  "start": start, "stop": stop})
    require(expected_hash == reference["reference_data_hash"], "Normalization data changed")
    scale = max(float(np.std(values, ddof=reference["ddof"])), float(reference["floor"]))
    require(np.isclose(scale, receipt["training_scale"], rtol=1e-12, atol=0)
            and np.isclose(scale, reference["scale"], rtol=1e-12, atol=0),
            "Saved response normalization does not reproduce")
    return scale


def audit_optimization(receipt):
    """Check recorded stopping arithmetic, without changing a qualification."""
    model, diagnostics = receipt["model_id"], receipt["diagnostics"]
    result = {"model_id": model, "diagnostic_qualified": diagnostics.get("qualified"),
              "final_qualified": receipt["qualified"]}
    if receipt["qualified"]:
        require(diagnostics.get("qualified", True), "Qualified receipt contradicts fitted diagnostics")
    if model == "basis_lgssm" and "log_likelihood" in diagnostics:
        trace = np.asarray(diagnostics["log_likelihood"], float)
        require(trace.ndim == 1 and len(trace) and np.isfinite(trace).all(), "Invalid EM trace")
        gains = np.diff(trace)/(1+np.abs(trace[:-1]))
        require(np.all(gains >= -1e-7), "Accepted EM trace decreases beyond its recorded tolerance")
        require(diagnostics["iterations"] == len(trace)-1, "EM iteration/trace mismatch")
        tolerance = diagnostics["optimizer_budget"]["tolerance"]
        if len(gains):
            require(np.isclose(gains[-1], diagnostics["final_relative_likelihood_gain"], atol=1e-14),
                    "EM terminal relative gain differs from its trace")
        if diagnostics["qualified"]:
            require(len(gains) and gains[-1] <= tolerance and diagnostics["converged"]
                    and not diagnostics["rejected_decreasing_step"], "Invalid EM qualification")
        result.update(stop_reason=diagnostics["stop_reason"], iterations=diagnostics["iterations"],
                      final_relative_gain=None if not len(gains) else float(gains[-1]),
                      tolerance=tolerance, monotonic_trace_checked=True)
    key = "covariance_optimizer_details" if model == "basis_gru" else "optimizer_details"
    if key in diagnostics:
        optimizer = diagnostics[key]
        phases = optimizer["phases"]
        require(1 <= len(phases) <= 2 and optimizer["continuations"] == len(phases)-1,
                "Optimizer exceeded its one-restart policy")
        require(optimizer["iterations"] == sum(p["iterations"] for p in phases)
                and optimizer["function_evaluations"] == sum(p["function_evaluations"] for p in phases),
                "Optimizer phase counts do not sum")
        for phase in phases:
            require(0 <= phase["iterations"] <= phase["max_iter"], "Optimizer exceeded iteration cap")
            require(phase["ftol"] == phases[0]["ftol"] and phase["gtol"] == phases[0]["gtol"],
                    "Optimizer changed its convergence tolerances")
        if len(phases) == 2:
            require(not phases[0]["success"] and phases[0]["status"] == 1
                    and phases[0]["iterations"] >= phases[0]["max_iter"],
                    "Restart did not follow iteration exhaustion")
            if not optimizer["rejected_worsening_continuation"]:
                a, b = phases[0]["objective"], phases[1]["objective"]
                require(a is not None and b is not None and b <= a+1e-10*(1+abs(a)),
                        "Accepted restart worsened the objective")
        if optimizer["qualified"]:
            require(any(p["success"] for p in phases) and not optimizer["rejected_worsening_continuation"],
                    "Optimizer qualification contradicts phase status")
        result.update(optimizer=optimizer,
                      interpretation="ftol or projected-gradient stopping; not a global-optimality certificate",
                      limitation="Initial objective value is not persisted, so first-phase improvement cannot be independently replayed")
    if model == "bayes_esn_ridge" and diagnostics.get("gates"):
        gate = diagnostics["gates"][-1]
        thresholds = receipt["fit_config"]["diagnostics"]
        parameters = [p for block in ("hyperparameters", "functionals")
                      for p in gate[block]["parameters"].values()]
        rhats = [np.nan if p["rhat"] is None else float(p["rhat"]) for p in parameters]
        ess = [np.nan if p[k] is None else float(p[k])
               for p in parameters for k in ("ess_bulk", "ess_tail")]
        if receipt["qualified"]:
            require(gate["passed"] and all(np.isfinite(rhats)) and all(np.isfinite(ess))
                    and max(rhats) <= thresholds["rhat_limit"] and min(ess) >= thresholds["min_ess"]
                    and all(gate[b]["chains"] >= thresholds["min_chains"]
                            for b in ("hyperparameters", "functionals")),
                    "Qualified ESN violates its scientific-quantity diagnostic thresholds")
        result.update(max_rhat=max(rhats) if rhats and all(np.isfinite(rhats)) else None,
                      minimum_bulk_or_tail_ess=min(ess) if ess and all(np.isfinite(ess)) else None,
                      nonfinite_diagnostic_entries=int(np.sum(~np.isfinite(rhats))+np.sum(~np.isfinite(ess))),
                      continuations=diagnostics["continuations"], gate_passed=gate["passed"])
    return result


def audit_cost(receipt, findings):
    fields = ("fit_seconds", "forecast_seconds", "scoring_seconds", "serialization_seconds",
              "artifact_serialization_seconds", "fallback_seconds", "tuning_seconds",
              "total_seconds", "total_pipeline_seconds", "cpu_seconds", "diagnostic_seconds")
    values = {key: float(receipt[key]) for key in fields}
    require(all(np.isfinite(v) and v >= 0 for v in values.values()), "Invalid recorded timing")
    require(np.isclose(values["total_pipeline_seconds"], values["total_seconds"]
            + values["tuning_seconds"] + values["artifact_serialization_seconds"], rtol=1e-10, atol=1e-8),
            "Pipeline cost arithmetic does not reproduce")
    require(values["serialization_seconds"]+1e-8 >= values["artifact_serialization_seconds"],
            "Artifact serialization exceeds recorded total serialization")
    ledger = receipt["diagnostics"].get("recovery", {}).get("completed_phase_seconds", {})
    ledger_diagnostics = sum(float(v) for key, v in ledger.items() if "diagnostic" in key)
    if ledger_diagnostics and not np.isclose(values["diagnostic_seconds"], ledger_diagnostics,
                                            rtol=1e-8, atol=1e-8):
        findings.append({"code": "diagnostic_subcomponent_scope_or_value_mismatch",
                         "job_identity": receipt["job_identity"],
                         "recorded_seconds": values["diagnostic_seconds"],
                         "cumulative_ledger_seconds": ledger_diagnostics,
                         "meaning": "Diagnostic work is inside fitting totals; inspect old key matching and current-versus-cumulative recovery scope. Do not add it again."})
    values.update(cumulative_diagnostic_ledger_seconds=ledger_diagnostics,
                  compute_accounting_complete=receipt.get("compute_accounting_complete", False),
                  timing_scope=receipt.get("timing_scope"),
                  cpu_scope="recorded final-job invocation; excludes tuning and later final-artifact writes",
                  peak_rss_scope=receipt.get("memory_measurement"))
    return values


def audit(run, output, *, max_archive_bytes=256*1024**2):
    run, output = Path(run).resolve(), Path(output).resolve()
    require(not output.is_relative_to(run), "Audit output must not overwrite the saved run")
    manifest = checked_json(run/"manifest.json", "identity")
    summary = checked_json(run/"summary.json")
    require(summary.get("complete") and summary.get("source_stable"), "Audit requires a completed source-stable run")
    require(summary["manifest_identity"] == manifest["identity"], "Summary/manifest identity mismatch")
    protocol = BenchmarkProtocol(**manifest["protocol"])
    require(protocol.identity == manifest["protocol_identity"], "Protocol identity mismatch")
    require(protocol.stage == "development", "This audit cannot inspect confirmation outcomes")
    findings, rows, oracles, receipts = [], [], [], {}
    expected_jobs = protocol.replications * len(protocol.scenarios) * sum(
        len(effective_models(protocol, track)) for track in protocol.tracks)
    require(summary["jobs"] == summary["scheduled_jobs"] == expected_jobs, "Incomplete reference roster")
    oracle_directory = output.parent/(output.stem+"_oracle")
    for scenario in protocol.scenarios:
        for replication in range(protocol.replications):
            base = run/scenario/f"rep_{replication:03d}"
            data = checked_json(base/"data_receipt.json", "identity")
            require(data["manifest_identity"] == manifest["identity"]
                    and data["sha256"] == sha256(base/"simulation.npz"), "Dataset binding mismatch")
            archive_admission(base/"simulation.npz", max_archive_bytes)
            sim = Simulation.load(base/"simulation.npz")
            require(content_hash({"y": sim.observation, "times": sim.times,
                                  "coordinates": sim.coordinates}) == data["observation_hash"],
                    "Observation/time/geometry hash mismatch")
            scales = set()
            for track in protocol.tracks:
                for model in effective_models(protocol, track):
                    directory = base/track/model
                    receipt = checked_json(directory/"receipt.json", "receipt_identity")
                    require(receipt["manifest_identity"] == manifest["identity"]
                            and receipt["data_identity"] == data["identity"]
                            and receipt["computation_identity"] == manifest["computation_identity"]
                            and receipt["terminal_scientific_result"] is True,
                            "Scientific receipt binding mismatch")
                    require((receipt["scenario"], receipt["replication"], receipt["track"], receipt["model_id"])
                            == (scenario, replication, track, model), "Receipt placed under the wrong job")
                    require(receipt["qualified"] != receipt["fallback"], "Invalid qualification/fallback combination")
                    require(receipt["outcome"] == ("qualified" if receipt["qualified"] else "method_failure_with_fallback"),
                            "Outcome disagrees with qualification")
                    for artifact in receipt["artifacts"]:
                        name = Path(artifact["file"])
                        require(not name.is_absolute() and ".." not in name.parts, "Unsafe artifact path")
                        path = directory/name
                        require(path.stat().st_size == artifact["bytes"] and sha256(path) == artifact["sha256"],
                                f"Saved artifact changed: {path}")
                    scale = audit_normalization(sim, protocol, receipt)
                    scales.add((scale, receipt["normalization_reference"]["identity"]))
                    target_indices = (sim.observed_indices[:protocol.score_observed_count]
                                      if track == "A" else sim.test_indices)
                    require(receipt["origin_index"] == protocol.final_origin
                            and receipt["target_ids"] == [sim.site_ids[j] for j in target_indices],
                            "Saved forecast origin or target identities changed")
                    point_path = directory/"operational_point.npz"
                    if point_path.exists():
                        archive_admission(point_path, max_archive_bytes)
                        point = Forecast.load(point_path)
                        truth = sim.observation[protocol.final_origin:protocol.n_times, target_indices]
                        rmse = float(np.sqrt(np.mean((point.mean-truth)**2)))
                        require(np.isclose(rmse, receipt["operational_scores"]["rmse"], rtol=1e-12),
                                "Saved operational point forecast does not reproduce its RMSE")
                    native_crps = (receipt["native_scores"].get("crps") if receipt["qualified"] else None)
                    row = {key: receipt[key] for key in ("scenario", "replication", "track", "model_id",
                                                       "qualified", "fallback", "job_identity", "receipt_identity")}
                    row.update(training_scale=scale,
                               normalized_rmse=receipt["operational_scores"]["rmse"]/scale,
                               normalized_native_crps=None if native_crps is None else native_crps/scale,
                               cost=audit_cost(receipt, findings), optimization=audit_optimization(receipt))
                    rows.append(row)
                    receipts[scenario, replication, track, model] = receipt
                if scenario == "g1_gaussian_ar":
                    moments, targets = oracle_moments(sim, protocol, track)
                    truth = sim.observation[protocol.final_origin:protocol.n_times, targets]
                    scores = gaussian_scores(moments["mean"], moments["variance"], truth)
                    destination = oracle_directory/f"rep_{replication:03d}_{track}.npz"
                    save_state(destination, moments, kind="v03_known_parameter_oracle")
                    oracles.append({"scenario": scenario, "replication": replication, "track": track,
                        "scores": scores, "training_scale": scale,
                        "normalized_rmse": scores["rmse"]/scale, "normalized_crps": scores["crps"]/scale,
                        "moment_artifact": str(destination.relative_to(output.parent)), "moment_sha256": sha256(destination),
                        "information_policy": moments["information_policy"], "history_hash": moments["history_hash"],
                        "parameters": moments["parameters"],
                        "interpretation": "Known true parameters, noisy O history only; a diagnostic reference with more parameter information than learned models. No finite-realization dominance guarantee."})
            require(len(scales) == 1, "Compared models/tracks use different per-dataset normalizations")
    paired_checked = 0
    for pair in summary["paired_comparisons"]:
        for metric in ("rmse", "crps"):
            values, ids = [], []
            for replication in pair["attempted_replications"]:
                a = receipts[pair["scenario"], replication, pair["track"], pair["reference"]]
                b = receipts[pair["scenario"], replication, pair["track"], pair["other_model"]]
                if metric == "crps" and not all(r["qualified"] and not r["fallback"]
                                                and "crps" in r["native_scores"] for r in (a, b)):
                    continue
                source = "operational_scores" if metric == "rmse" else "native_scores"
                values.append((a[source][metric]-b[source][metric])/a["training_scale"])
                ids.append(replication)
            require(ids == pair[metric+"_replications"] and len(ids) == pair[metric+"_population"],
                    "Paired qualified/operational population differs")
            if values:
                require(np.allclose(values, pair[metric]["paired_differences"], rtol=1e-11, atol=1e-12),
                        "Normalized paired differences do not reproduce")
                require(np.isclose(np.mean(values), pair[metric]["difference"], rtol=1e-11, atol=1e-12),
                        "Normalized paired mean does not reproduce")
                mcse = None if len(values) < 2 else float(np.std(values, ddof=1)/np.sqrt(len(values)))
                require(pair[metric]["mcse"] is None if mcse is None else
                        np.isclose(mcse, pair[metric]["mcse"], rtol=1e-11), "Paired MCSE differs")
                paired_checked += 1
    cost_by_model = defaultdict(lambda: defaultdict(float))
    table_checked = 0
    for table in summary["table"]:
        members = [r for key, r in receipts.items()
                   if (key[0], key[2], key[3]) == (table["scenario"], table["track"], table["model_id"])]
        qualified = [r for r in members if r["qualified"] and not r["fallback"]]
        require(len(members) == table["operational_population"]
                and len(qualified) == table["qualified"]
                and sum(r["fallback"] for r in members) == table["fallback"],
                "Aggregate qualification/population counts do not reproduce")
        require(np.isclose(table["rmse"], np.mean([r["operational_scores"]["rmse"] for r in members]), rtol=1e-12),
                "Aggregate mean dataset RMSE does not reproduce")
        for output_field, input_field in (("fit_seconds_total", "fit_seconds"),
                                          ("forecast_seconds_total", "forecast_seconds"),
                                          ("tuning_seconds_total", "tuning_seconds")):
            require(np.isclose(table[output_field], sum(r[input_field] for r in members), rtol=1e-12),
                    "Aggregate cost sum does not reproduce")
        table_checked += 1
    for row in rows:
        for key in ("fit_seconds", "tuning_seconds", "total_pipeline_seconds"):
            cost_by_model[row["model_id"]][key] += row["cost"][key]
    current_source = source_fingerprint()
    return {"schema": "v03-reference-audit-1.0", "status": "passed_with_findings" if findings else "passed",
            "input_run": str(run), "manifest_identity": manifest["identity"],
            "source_fingerprint": manifest["source_fingerprint"], "audit_source_fingerprint": current_source,
            "same_package_source": current_source == manifest["source_fingerprint"],
            "tool_sha256": sha256(__file__), "manifest_sha256": sha256(run/"manifest.json"),
            "summary_sha256": sha256(run/"summary.json"), "protocol_identity": protocol.identity,
            "scope": "read-only saved development audit; no comparator fits or forecasts; analytic G1 oracle only",
            "jobs_checked": len(rows), "aggregate_rows_checked": table_checked,
            "normalized_paired_cells_checked": paired_checked,
            "qualified_jobs": sum(r["qualified"] for r in rows), "fallback_jobs": sum(r["fallback"] for r in rows),
            "rows": rows, "oracle": oracles, "findings": findings,
            "recorded_cost_by_model": {k: dict(v) for k, v in cost_by_model.items()},
            "cost_qualification": "Summed recorded job time, not elapsed parallel campaign time; interrupted unknown work remains unknown. Runtime is never divided by response scale.",
            "scientific_conclusion": "Descriptive development references only; no new tuning, confirmation or general predictive-improvement claim."}


def validate_fhn(output):
    """Optional bounded independent H=100 finite-grid check; no model fitting."""
    path = Path(output).with_name(Path(output).stem+"_fhn.json")
    command = [sys.executable, str(ROOT/"tools/validate_generators.py"), "--extended",
               "--grid-size", "64", "--dt", "0.05", "--delta", "2.0",
               "--observations", "101", "--burnin", "200", "--output", str(path)]
    environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False, timeout=600)
    require(completed.returncode in (0, 2) and path.exists(), "Independent FHN validator did not complete")
    value = checked_json(path, "identity")
    policy = value["finite_grid_policy"]
    require(policy["grid_size"] == 64 and policy["dt"] == .05 and policy["delta"] == 2.
            and value["n_observations"] == 101 and value["physical_burnin"] == 200.,
            "FHN receipt does not match the requested pilot")
    return {"file": str(path.resolve()), "sha256": sha256(path), "identity": value["identity"],
            "validator_exit_code": completed.returncode, "validator_status": value["status"],
            "requested_forecast_horizon": 100, "physical_validation_window": 200,
            "finite_semidiscrete_time_and_morphology_passed": bool(policy["time_gate_passed"] and policy["morphology_gate_passed"]),
            "mesh_gate_passed": policy["mesh_gate_passed"], "continuum_claim": False,
            "scope": "Independent seed and post-burn-in window; finite-grid time/morphology and continuum mesh checks retain separate outcomes."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", default="results/v03/development/reference_reliability")
    parser.add_argument("--output", default="docs/evidence/v03/reference_audit.json")
    parser.add_argument("--max-archive-mb", type=float, default=256)
    parser.add_argument("--validate-fhn", action="store_true",
                        help="Also run the independent 64-grid, dt=.05, delta=2, H=100 generator check")
    args = parser.parse_args(argv)
    require(np.isfinite(args.max_archive_mb) and args.max_archive_mb > 0, "Positive finite archive budget required")
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        receipt = audit(args.run, args.output, max_archive_bytes=int(args.max_archive_mb*1024**2))
    if args.validate_fhn:
        receipt["fhn_validation"] = validate_fhn(args.output)
        if not receipt["fhn_validation"]["finite_semidiscrete_time_and_morphology_passed"]:
            receipt["status"] = "generator_gate_not_met"
    receipt["audit_seconds"] = time.perf_counter()-started
    receipt["identity"] = content_hash(receipt)
    write_json(args.output, receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "jobs_checked", "qualified_jobs",
          "fallback_jobs", "normalized_paired_cells_checked", "audit_seconds")}, indent=2))
    print(json.dumps({"output": str(Path(args.output).resolve()), "findings": len(receipt["findings"]),
                      "oracle_cases": len(receipt["oracle"]), "comparator_refits": 0}))
    return 2 if receipt["status"] == "generator_gate_not_met" else 0


if __name__ == "__main__":
    raise SystemExit(main())
