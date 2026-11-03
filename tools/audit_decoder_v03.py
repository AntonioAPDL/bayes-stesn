"""Reproduce the bounded saved G3 first-step/decoder diagnostic.

From the repository root, with the locked environment and numerical threads1:
  PYTHONPATH=src python tools/audit_decoder_v03.py \
    --model results/v03/development/reference_final/g3_excitable_field/rep_000/A/bayes_esn_ridge/model.npz \
    --simulation results/v03/development/reference_final/g3_excitable_field/rep_000/simulation.npz \
    --paths results/v03/extensions/final_paths_g3_excitable_field \
    --historical-review docs/evidence/v03/g3_saved_first_step_review.json \
    --output docs/evidence/v03/g3_saved_first_step_review_final.json

Exactly64 original path parameter draws are recovered to reconstruct their
first observation. All longer-lead results use already saved512x40 paths.
No fitting, full forecasting, or encoder estimation occurs. Projection floors
use noisy evaluator observations and cannot serve as permitted forecasts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from zipfile import ZipFile

import numpy as np

from bayes_stesn.benchmark.contracts import Forecast
from bayes_stesn.benchmark.execution import memory_allowance
from bayes_stesn.benchmark.generators import Simulation
from bayes_stesn.benchmark.runner import benchmark_environment, jsonable, single_thread_runtime, source_fingerprint, write_json
from bayes_stesn.io import content_hash, seed_for
from bayes_stesn.prediction import _psd_root, forecast_context
from evaluate_extensions_v03 import _load_baseline, _verify_terminal_inventory


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value) ** 2)))


def compare_previous(actual, previous, path="", differences=None):
    """Check shared scientific fields; metadata/timing and new labels excluded."""
    differences = [] if differences is None else differences
    if isinstance(previous, dict):
        for key, value in previous.items():
            if key not in actual:
                differences.append(path + "." + key)
            else:
                compare_previous(actual[key], value, path + "." + key, differences)
    elif isinstance(previous, list):
        if len(actual) != len(previous):
            differences.append(path)
        else:
            for index, (a, b) in enumerate(zip(actual, previous)):
                compare_previous(a, b, f"{path}[{index}]", differences)
    elif isinstance(previous, (float, int)) and not isinstance(previous, bool):
        if not np.isclose(actual, previous, rtol=1e-12, atol=1e-12):
            differences.append(path)
    elif actual != previous:
        differences.append(path)
    return differences


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True, help="Completed512-path P5 directory")
    parser.add_argument("--historical-review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-memory-mb", type=float, default=512)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Keep sealed diagnostic evidence intact; use a new output path")
    with single_thread_runtime():
        started = time.perf_counter()
        source, environment = source_fingerprint(), jsonable(benchmark_environment())
        manifest_path = args.paths / "manifest.json"
        inventory_path = args.paths / "output_inventory.json"
        summary_path = args.paths / "path_stability.json"
        forecast_path = args.paths / "path_stability_forecast.npz"
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("identity") != content_hash({k:v for k,v in manifest.items() if k != "identity"})
                or manifest.get("source_fingerprint") != source
                or manifest.get("baseline_sha256") != sha256(args.model)
                or manifest.get("simulation_sha256") != sha256(args.simulation)
                or manifest["settings"].get("track") != "A"
                or not _verify_terminal_inventory(args.paths, manifest, "path_stability")):
            raise ValueError("Path study does not bind these current-source model/data inputs")
        tool = Path(__file__).resolve()
        input_paths = [args.model, args.simulation, manifest_path, inventory_path, summary_path, forecast_path, tool]
        if args.historical_review:
            input_paths.append(args.historical_review)
        root = tool.parents[1]
        input_hashes = {str(path.resolve().relative_to(root)) if path.resolve().is_relative_to(root)
                        else str(path.resolve()): sha256(path) for path in input_paths}
        expanded = 0
        for path in (args.model, args.simulation, forecast_path):
            with ZipFile(path) as archive:
                expanded += sum(member.file_size for member in archive.infolist())
        estimated = 4 * expanded + 64 * 1024 ** 2
        allowance = memory_allowance(args.max_memory_mb)
        if allowance["effective_job_budget_bytes"] is not None and estimated > allowance["effective_job_budget_bytes"]:
            raise MemoryError("Saved-array diagnostic exceeds current admitted memory allowance")
        fitted, gate = _load_baseline(args.model)
        if not gate or not gate["passed"]:
            raise ValueError("The diagnostic requires the saved qualified benchmark fit")
        p = fitted.prepared
        if p.spec.innovations != "iid" or p.spec.covariance.kind != "matern":
            raise ValueError("This diagnostic reproduces the iid dense-Matérn reference only")
        simulation = Simulation.load(args.simulation)
        forecast = Forecast.load(forecast_path)
        paths = forecast.samples
        path_summary = json.loads(summary_path.read_text())
        if (simulation.metadata["scenario"] != "g3_excitable_field"
                or fitted.posterior_identity != manifest["input_posterior_identity"]
                or path_summary["posterior_identity"] != fitted.posterior_identity
                or paths is None or paths.shape != (512, 40, len(p.Phi))):
            raise ValueError("Expected the saved G3 posterior and512x40 observed-site paths")
        if forecast.site_ids != p.data.site_ids or not np.array_equal(forecast.coordinates, p.data.coordinates):
            raise ValueError("Saved path targets differ from the observed input panel")
        site_lookup = {label: i for i, label in enumerate(simulation.site_ids)}
        sites = np.array([site_lookup[label] for label in p.data.site_ids])
        rows = np.searchsorted(simulation.times, p.data.times)
        origin = int(rows[-1] + 1)
        if (not np.array_equal(rows, np.arange(len(rows)))
                or not np.array_equal(simulation.times[rows], p.data.times)
                or not np.array_equal(simulation.coordinates[sites], p.data.coordinates)
                or not np.array_equal(simulation.observation[np.ix_(rows, sites)], p.data.y)
                or not np.array_equal((p.data.y-p.response_center)/p.response_scale, p.standardized_y)
                or origin != path_summary["evaluation"]["origin_index"]
                or origin+40 > len(simulation.times)):
            raise ValueError("History, scale, geometry or first evaluation index differs")
        for row in path_summary["rows"]:
            if content_hash(paths[:row["paths"]]) != row["path_hash"]:
                raise ValueError("Saved forecast prefix differs from its path-study receipt")
        context = forecast_context(fitted, history=p.data)
        state_error = float(np.max(np.abs(context.states-p.features.states)))
        feature_error = float(np.max(np.abs(context.Z-p.Z)))
        next_state, next_z = p.features.update(context.historical_y[-1], context.states[-1])
        replay_states, replay_z = p.features.history(np.vstack((context.historical_y, np.zeros(len(p.Phi)))))
        next_exact = np.array_equal(next_state, replay_states[-1]) and np.array_equal(next_z, replay_z[-1])
        if state_error != 0 or feature_error != 0 or not next_exact:
            raise ValueError("Frozen causal feature/state replay differs")
        seed = seed_for(manifest["settings"]["seed"], "path_stability")
        means, last_means, innovation_variance, reconstructed, ids = [], [], [], [], []
        for global_path in range(64):
            rng = np.random.default_rng(seed_for(seed, "parameter", global_path))
            chain = int(rng.integers(fitted.posterior.n_chains))
            draw = int(rng.integers(fitted.posterior.n_draws))
            ids.append((chain, draw))
            parameter = fitted.posterior.draw_parameters(chain, draw, rng)
            C, sigma2, cp = parameter.C, float(parameter.sigma2), parameter.covariance_parameters
            mean_map = p.Phi @ C
            mean = mean_map @ next_z
            means.append(mean*p.response_scale+p.response_center)
            last_means.append((mean_map @ context.Z[-1])*p.response_scale+p.response_center)
            K, V = p.spec.covariance.matrices(p.data.coordinates, cp)
            q = float(np.exp(cp["log_q"]))
            innovation_variance.append(sigma2*np.diag(V+q*K)*p.response_scale**2)
            innovation_rng = np.random.default_rng(seed_for(seed, "innovation", global_path))
            spatial = np.sqrt(sigma2*q)*(_psd_root(K) @ innovation_rng.normal(size=len(K)))
            measurement = np.sqrt(sigma2)*np.sqrt(np.diag(V))*innovation_rng.normal(size=len(K))
            reconstructed.append((mean+spatial+measurement)*p.response_scale+p.response_center)
        means, last_means, innovation_variance = map(np.asarray, (means, last_means, innovation_variance))
        reconstruction_error = float(np.max(np.abs(np.asarray(reconstructed)-paths[:64, 0])))
        if reconstruction_error > 1e-12:
            raise ValueError("Named first-step reconstruction differs from the saved original paths")
        truth = simulation.observation[origin:origin+40, sites]
        fitted_truth = p.data.y[p.split.fitting_start:p.split.fitting_stop]
        # The decoder span includes an intercept. These are evaluator-only
        # projections of noisy observations in their original response units.
        projector = p.Phi @ np.linalg.pinv(p.Phi, rcond=1e-10)
        projected = truth @ projector.T
        outside = truth-projected
        readout_mean = means.mean(axis=0)
        within = projected[0]-readout_mean
        readout_variance = means.var(axis=0, ddof=1)
        saved_variance = paths.var(axis=0, ddof=1)
        saved_mean = paths.mean(axis=0)
        readout_rmse = rms(readout_mean-truth[0])
        outside_rmse = rms(outside[0])
        one = {"conditional_readout64_mcse_rms":rms(np.sqrt(readout_variance/64)),
            "conditional_readout_mean64_rmse":readout_rmse,
            "conditional_readout_variance_mean64":float(readout_variance.mean()),
            "empirical_total_variance_mean_saved512":float(saved_variance[0].mean()),
            "last_fitted_readout64_rmse":rms(last_means.mean(axis=0)-p.data.y[-1]),
            "mean_conditional_innovation_variance64":float(innovation_variance.mean()),
            "mean_conditional_readout64_vs_saved512_rms":rms(readout_mean-saved_mean[0]),
            "persistence_rmse":rms(p.data.y[-1]-truth[0]),
            "saved512_mean_rmse":rms(saved_mean[0]-truth[0]),
            "total_variance_reference64":float(readout_variance.mean()+innovation_variance.mean())}
        decoder = {"final40_true_field_projection_rmse":rms(outside),
            "final40_observed_target_projection_rmse":rms(outside),
            "projection_target":"Noisy observed field at the fitted O sites; the legacy true_field key is an observed-target alias, not latent truth",
            "first_lead_outside_span_fraction_of_conditional_readout_mse64":outside_rmse**2/readout_rmse**2,
            "first_lead_unavoidable_outside_span_rmse":outside_rmse,
            "first_lead_within_span_readout_error_rmse64":rms(within),
            "fitting_panel_projection_rmse":rms(fitted_truth-fitted_truth@projector.T),
            "orthogonal_mse_decomposition_error":float(abs(readout_rmse**2-outside_rmse**2-rms(within)**2)),
            "rank":int(np.linalg.matrix_rank(p.Phi)), "sites":len(p.Phi)}
        leads = [{"lead":lead, "empirical_mean_rmse":rms(saved_mean[lead-1]-truth[lead-1]),
            "predictive_sd_rms":float(np.sqrt(saved_variance[lead-1].mean())),
            "conditional_empirical_mean_mcse_rms":rms(np.sqrt(saved_variance[lead-1]/512)),
            "conditional_empirical_mean_mcse_max":float(np.sqrt(saved_variance[lead-1]/512).max()),
            "decoder_projection_rmse":rms(outside[lead-1]),
            "sample_abs_max":float(np.max(np.abs(paths[:, lead-1])))} for lead in (1,2,5,10,20,40)]
        result = {"schema":"saved-G3-first-step-review-1.0", "source_fingerprint":source,
            "tool_sha256":sha256(tool), "environment":environment, "input_hashes":input_hashes,
            "posterior_identity":fitted.posterior_identity,
            "scope":"Read-only saved fit and saved512-path artifact;64 original path parameter draws recovered; no refit and no new full forecast",
            "alignment":{"causal_next_feature_replay_exact":next_exact,
                "first64_saved_observation_reconstruction_max_abs":reconstruction_error,
                "first_evaluated_row":origin, "first_evaluated_time":float(simulation.times[origin]),
                "history_site_order_coordinates_scale_exact":True,
                "last_fitted_row":int(rows[-1]), "last_fitted_time":float(p.data.times[-1]),
                "observation_spacing":float(np.diff(p.data.times)[-1]),
                "response_center":p.response_center, "response_scale":p.response_scale,
                "saved_feature_replay_max_abs":feature_error, "saved_state_replay_max_abs":state_error,
                "stored_rows":len(p.data.y)}, "decoder":decoder, "one_step":one, "saved_path_leads":leads,
            "recovered_global_path_indices":[0,64], "recovered_parameter_ids_hash":content_hash(np.asarray(ids,dtype=np.int64)),
            "resource_admission":{"uncompressed_input_bytes":expanded,"estimated_additional_bytes":estimated,"allowance":allowance},
            "interpretation_limits":["64 conditional draws give a descriptive first-step moment reference, not an exact posterior mean.",
                "Decoder projection uses evaluator observations only as a diagnostic; it is not a permitted forecast or model selection result.",
                "Growing saved-path variance demonstrates forecast dispersion under stochastic observation feedback; it does not identify separate parameter-versus-innovation contributions at long leads.",
                "No off-by-one, site-order or response-scale discrepancy was found by these exact checks.",
                "This A-track replay reuses the same posterior, field and saved paths; it adds no independent dataset or capacity-improvement evidence and does not clear B-track interpolation errors."],
            "elapsed_diagnostic_seconds":time.perf_counter()-started}
        if args.historical_review:
            historical = json.loads(args.historical_review.read_text())
            if historical["identity"] != content_hash({k:v for k,v in historical.items() if k != "identity"}):
                raise ValueError("Historical review identity differs")
            keys = ("alignment", "decoder", "one_step", "saved_path_leads")
            differences = compare_previous({k:result[k] for k in keys}, {k:historical[k] for k in keys})
            if differences or historical["posterior_identity"] != fitted.posterior_identity:
                raise ValueError("Replayed scientific diagnostic differs: " + ", ".join(differences))
            result["replay_comparison"] = {"historical_review":str(args.historical_review),
                "historical_identity":historical["identity"], "scientific_fields_agree":True,
                "absolute_and_relative_tolerance":1e-12, "new_independent_datasets":0,
                "meaning":"Same saved-field/posterior diagnostic reproduced under final source; numerical roundoff within fixed tolerance is allowed"}
        if source_fingerprint() != source or jsonable(benchmark_environment()) != environment:
            raise ValueError("Package source or numerical runtime changed during the diagnostic")
        if any(sha256(root/path if not Path(path).is_absolute() else path) != value for path,value in input_hashes.items()):
            raise ValueError("A bound input changed during the diagnostic")
        result = jsonable(result)
        result["identity"] = content_hash(result)
        write_json(args.output, result)
    print(json.dumps({"output":str(args.output),"identity":result["identity"],
                      "reconstruction_max_abs":reconstruction_error,"historical_comparison":bool(args.historical_review)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
