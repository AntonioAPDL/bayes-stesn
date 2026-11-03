"""Inspect a saved scalar ESN; optionally run small, separately named ablations.

Default: no refitting and no future labels. Opt-in predictive modes require a
saved Simulation plus its development manifest and verify the fitted history.
They never alter the registered bayes_esn_ridge model or its archived scores.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from zipfile import ZipFile

import numpy as np

from bayes_stesn.assessment import diagnose
from bayes_stesn.benchmark.contracts import Forecast
from bayes_stesn.benchmark.esn import CheckpointIntegrityError, _recoverable_fit
from bayes_stesn.benchmark.execution import memory_allowance
from bayes_stesn.benchmark.generators import Simulation
from bayes_stesn.benchmark.metrics import score_forecast
from bayes_stesn.benchmark.protocol import BenchmarkProtocol
from bayes_stesn.benchmark.runner import benchmark_environment, dgp_seed_for, jsonable, single_thread_runtime, source_fingerprint, write_json
from bayes_stesn.io import content_hash, load_state, seed_for
from bayes_stesn.model import FittedModel, prepare
from bayes_stesn.prediction import iter_predictive
from bayes_stesn.priors import PriorSpec


SCOPE_PATH = Path(__file__).resolve().parents[1]/"docs/evidence/v03/extension_scope.json"
CHECKS = {"functional_draws":512, "rhat_limit":1.01, "min_ess":400, "min_chains":4}


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _admit_archive(path, memory_mb):
    with ZipFile(path) as archive:
        expanded = sum(member.file_size for member in archive.infolist())
    allowance = memory_allowance(memory_mb)
    # Numeric archives plus temporary parsing/recovery work. This is a
    # conservative admission estimate, not a claim of measured peak RSS.
    estimated = 4*expanded+512*1024**2
    effective = allowance["effective_job_budget_bytes"]
    if effective is not None and estimated > effective:
        raise MemoryError("Archive/recovery estimate exceeds current safe allowance; use a smaller compatible fit")
    return {"uncompressed_archive_bytes":expanded, "estimated_additional_bytes":estimated,
            "allowance":allowance}


def _load_baseline(path):
    state = load_state(path)
    stored_gate = None
    if state.get("model_id") == "bayes_esn_ridge" and "fitted" in state:
        gates = state.get("diagnostics", {}).get("gates", [])
        stored_gate = gates[-1] if gates else None
        state = state["fitted"]
    if not {"prepared", "posterior", "sampler", "posterior_identity"}.issubset(state):
        raise ValueError("Baseline must be a saved core FittedModel or bayes_esn_ridge benchmark model")
    fitted = FittedModel.from_state(state)
    p = fitted.prepared
    if (p.spec.prior.kind != "ridge" or p.spec.innovations != "iid"
            or fitted.sampler.temporal != "iid"):
        raise ValueError("P5 baseline must be scalar-ridge iid Gaussian; extensions are separately named")
    if len(p.data.coordinates) > 100 or p.Y.shape[1] > 2000:
        raise ValueError("This bounded P5 tool supports n<=100 and at most 2000 likelihood rows")
    if p.Y.shape[1] < 2:
        raise ValueError("Residual autocorrelation inspection requires at least two likelihood rows")
    if p.spec.covariance.kind != "matern":
        raise ValueError("This first P5 comparison supports dense Matérn baselines; sparse routes need their own ablation")
    return fitted, stored_gate


def inspect_fitted(fitted, *, draws_per_chain, seed):
    """Descriptive causal one-step fitting residuals; no future responses."""
    p, posterior = fitted.prepared, fitted.posterior
    count = min(draws_per_chain, posterior.n_draws)
    indices = np.linspace(0, posterior.n_draws-1, count, dtype=int)
    groups = tuple(dict.fromkeys(p.groups))
    masks = {group:np.array(p.groups) == group for group in groups}
    coefficient_sum = np.zeros((p.Phi.shape[1], p.fit_Z.shape[1]))
    residual_variance_sum = np.zeros(len(p.Phi))
    prior_variance = []
    rms = {group:[] for group in groups}
    for chain in range(posterior.n_chains):
        for draw_index in indices:
            draw = posterior.draw_parameters(chain, int(draw_index),
                np.random.default_rng(seed_for(seed, "inspection", chain, int(draw_index))))
            coefficient_sum += draw.C
            R = p.spec.covariance.covariance(p.data.coordinates, draw.covariance_parameters)
            residual_variance_sum += draw.sigma2*np.diag(R)
            precision = float(np.exp(np.asarray(draw.prior_state["log_precision"])[0]))
            prior_variance.append(draw.sigma2/precision*p.response_scale**2)
            for group, mask in masks.items():
                rms[group].append(float(np.sqrt(np.mean(draw.C[:, mask]**2))*p.response_scale))
    recovered = count*posterior.n_chains
    C_mean = coefficient_sum/recovered
    residual = p.Y-p.Phi@C_mean@p.fit_Z.T
    residual_original = residual*p.response_scale
    reference_sd = np.sqrt(residual_variance_sum/recovered)*p.response_scale
    standardized = residual_original/reference_sd[:, None]
    centered = standardized-standardized.mean(axis=1, keepdims=True)
    denominator = np.sum(centered**2, axis=1)
    acfs = {}
    for lag in (1, 2, 5, 10):
        if lag < residual.shape[1]:
            values = np.divide(np.sum(centered[:, lag:]*centered[:, :-lag], axis=1), denominator,
                               out=np.zeros_like(denominator), where=denominator > 0)
            acfs[str(lag)] = {"site_values":values.tolist(), "mean":float(values.mean()),
                             "median":float(np.median(values)), "mean_absolute":float(np.abs(values).mean())}
    lag_one = np.array(acfs["1"]["site_values"])
    signed = float(lag_one.mean())
    sign_agreement = float(np.mean(lag_one*np.sign(signed) > 0)) if signed else 0.
    group_rows = []
    phi_energy, phi_sum_energy = float(np.sum(p.Phi**2)), float(np.sum(p.Phi.sum(axis=0)**2))
    N0 = len(p.reference_Z)*len(p.Phi)
    for group, mask in masks.items():
        z0 = p.reference_Z[:, mask]
        # ||H0 (Z_g kron Phi)||_F^2 / N0, without the enormous Kronecker matrix.
        energy = max(0., (phi_energy*np.sum(z0**2)
                     -phi_sum_energy*np.sum(z0.sum(axis=0)**2)/N0)/N0)
        field = p.Phi@C_mean[:, mask]@p.fit_Z[:, mask].T*p.response_scale
        group_rows.append({"group":group, "temporal_columns":int(mask.sum()),
            "posterior_coefficient_rms_quantiles_05_50_95":np.quantile(rms[group], [.05, .5, .95]).tolist(),
            "posterior_mean_field_centered_rms":float(np.sqrt(np.mean((field-field.mean())**2))),
            "conditioning_reference_energy":float(energy),
            "prior_reference_field_sd":float(np.sqrt(np.mean(prior_variance)*energy))})
    nonintercept = [row["posterior_coefficient_rms_quantiles_05_50_95"][1]
                   for row in group_rows if row["group"] != "intercept"]
    ratio = max(nonintercept)/max(min(nonintercept), np.finfo(float).tiny) if len(nonintercept)>1 else None
    return {"definition":"Training y_t minus posterior mean Phi C z_t; causal frozen features, no held-out labels",
        "limitations":"Descriptive training residuals; not independent-site tests, causal feature importance or out-of-sample adequacy proof",
        "recovered_joint_draws":recovered, "draw_indices_per_chain":indices.tolist(),
        "likelihood_rows":p.Y.shape[1], "observed_sites":len(p.Phi),
        "residual_rms_original_units":float(np.sqrt(np.mean(residual_original**2))),
        "standardized_residual_rms":float(np.sqrt(np.mean(standardized**2))),
        "fraction_abs_standardized_residual_above_3":float(np.mean(np.abs(standardized)>3)),
        "autocorrelation_estimator":"Demeaned within-site lagged product divided by full within-site sum of squares",
        "acf":acfs, "lag1_sign_agreement":sign_agreement,
        "prior_common_coefficient_sd_quantiles_05_50_95":np.quantile(np.sqrt(prior_variance), [.05,.5,.95]).tolist(),
        "prior_scale_meaning":"Posterior summaries of the conditional prior scale sigma/sqrt(lambda); one shared scalar-ridge scale, with group field scales from frozen design energy",
        "groups":group_rows, "nonintercept_group_coefficient_rms_ratio":ratio,
        "signals":{"grouped_ridge":ratio is not None and ratio >= 3.,
                   "full_residual_ar":abs(signed) >= .15 and sign_agreement >= .75},
        "signal_interpretation":"Fixed engineering triage rules only; no automatic promotion or significance claim"}


def _dispositions(scope, inspection, qualified):
    rows = []
    for extension in scope["extensions"]:
        row = dict(extension)
        signal = None if inspection is None else inspection["signals"].get(extension["id"])
        if not qualified:
            row["disposition"] = "deferred_baseline_not_qualified"
        elif signal is True:
            row["disposition"] = "candidate_for_prespecified_bounded_ablation_not_promoted"
        elif signal is False:
            row["disposition"] = "optional_no_declared_training_signal_detected"
        else:
            row["disposition"] = extension["default_disposition"]
        row["scientific_ablation_executed"] = False
        rows.append(row)
    return rows


def _evaluation(fitted, simulation_path, manifest_path, track, horizon, memory_mb):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("stage") != "development":
        raise ValueError("Predictive extension/path decisions require a development manifest, never confirmation labels")
    if manifest.get("identity") != content_hash({k:v for k,v in manifest.items() if k!="identity"}):
        raise ValueError("Development manifest checksum differs")
    _admit_archive(simulation_path, memory_mb)
    simulation = Simulation.load(simulation_path)
    protocol = BenchmarkProtocol(**manifest["protocol"])
    if protocol.stage != "development" or track not in protocol.tracks:
        raise ValueError("The requested track must belong to a development protocol")
    scenario = simulation.metadata["scenario"]
    matched_replications = [r for r in range(protocol.replications)
                            if dgp_seed_for(protocol, scenario, r) == simulation.metadata["seed"]]
    if (scenario not in protocol.scenarios or len(matched_replications) != 1
            or len(simulation.observation) != protocol.n_times):
        raise ValueError("Simulation is not an identified dataset from this development design")
    data = fitted.prepared.data
    site_positions = {label:i for i,label in enumerate(simulation.site_ids)}
    if not set(data.site_ids).issubset({simulation.site_ids[i] for i in simulation.observed_indices}):
        raise ValueError("Baseline inputs must be observed-site IDs from this simulation")
    sites = np.array([site_positions[label] for label in data.site_ids])
    rows = np.searchsorted(simulation.times, data.times)
    if (np.any(rows >= len(simulation.times)) or not np.array_equal(simulation.times[rows], data.times)
            or not np.array_equal(simulation.coordinates[sites], data.coordinates)
            or not np.array_equal(simulation.observation[np.ix_(rows, sites)], data.y)):
        raise ValueError("Saved baseline history/coordinates do not exactly match the supplied simulation")
    if len(rows)>1 and not np.all(np.diff(rows) == 1):
        raise ValueError("Baseline requires a contiguous history window in the simulation")
    origin = int(rows[-1]+1)
    if origin != protocol.final_origin:
        raise ValueError("Use the final refitted development model; tuning-origin evaluation is a separate task")
    horizon = protocol.horizon if horizon is None else horizon
    targets = sites if track == "A" else simulation.test_indices
    if not 1 <= horizon <= 100 or origin+horizon > len(simulation.times):
        raise ValueError("Evaluation needs the immediate complete horizon, with 1<=H<=100")
    if not 1 <= len(targets) <= 128:
        raise ValueError("This bounded ablation uses at most 128 prespecified targets; use a compatible study")
    joint_sites = len(sites) if track == "A" else len(sites)+len(targets)
    if joint_sites > fitted.prepared.spec.covariance.max_dense_sites:
        raise ValueError("Joint input/target field exceeds this saved model's dense covariance limit")
    return {"truth":np.array(simulation.observation[origin:origin+horizon, targets], copy=True),
        "coordinates":simulation.coordinates[targets],
        "site_ids":tuple(simulation.site_ids[i] for i in targets),
        "metadata":{"simulation_sha256":_hash_file(simulation_path),
            "development_manifest_sha256":_hash_file(manifest_path), "manifest_identity":manifest.get("identity"),
            "scenario":scenario, "replication":matched_replications[0], "track":track, "origin_index":origin,
            "horizon":horizon, "target_count":len(targets), "independent_datasets":1,
            "history_matched_exactly":True}}


def _forecast(fitted, evaluation, paths, seed, batch_size):
    shape = (paths, len(evaluation["truth"]), len(evaluation["site_ids"]))
    samples = np.empty(shape, dtype=float)
    draw_ids = np.empty((paths, 2), dtype=np.int64)
    for batch in iter_predictive(fitted, paths=paths, batch_size=batch_size,
            horizon=shape[1], target_coordinates=evaluation["coordinates"],
            target_site_ids=evaluation["site_ids"], seed=seed, components=("observation",),
            max_output_bytes=128*1024**2):
        start = batch.metadata["path_start"]
        stop = start+len(batch.observation)
        samples[start:stop] = batch.observation
        draw_ids[start:stop] = batch.parameter_draw_ids
    return samples, draw_ids


def _score(samples, evaluation, seed):
    forecast = Forecast(samples.mean(axis=0), samples, evaluation["coordinates"], evaluation["site_ids"],
                        {"uncertainty":"posterior_joint", "parameter_policy":"frozen"})
    return forecast, score_forecast(forecast, evaluation["truth"], pair_count=256, seed=seed)


def _path_stability(fitted, evaluation, output, seed, batch_size):
    samples, draw_ids = _forecast(fitted, evaluation, 512, seed_for(seed, "path_stability"), batch_size)
    rows = []
    for count in (128, 256, 512):
        _, scores = _score(samples[:count], evaluation, seed_for(seed, "scores"))
        mcse = samples[:count].std(axis=0, ddof=1)/np.sqrt(count)
        rows.append({"paths":count, "scores":scores,
            "predictive_mean_mcse_rms":float(np.sqrt(np.mean(mcse**2))),
            "predictive_mean_mcse_max":float(mcse.max()),
            "path_hash":content_hash(samples[:count]), "draw_id_hash":content_hash(draw_ids[:count])})
    for row in rows:
        row["difference_from_512"] = {key:row["scores"][key]-rows[-1]["scores"][key]
                                      for key in ("rmse", "mae", "crps")}
    result = {"mode":"fixed_fit_nested_predictive_paths", "posterior_identity":fitted.posterior_identity,
        "evaluation":evaluation["metadata"], "rows":rows,
        "interpretation":"Conditional predictive Monte Carlo sensitivity; not independent datasets or MCMC error",
        "mcse_scope":"Predictive mean, conditional on the stored posterior mixture; excludes posterior MCMC error",
        "crps":"Exact empirical V-statistic for each prefix; no finite-ensemble correction applied",
        "global_path_indexing":True, "automatic_path_count_selection":False}
    forecast, _ = _score(samples, evaluation, seed_for(seed, "scores"))
    forecast.save(output/"path_stability_forecast.npz")
    write_json(output/"path_stability.json", result)
    return result


def _ablations(baseline, extensions, evaluation, output, args, source):
    # Refit a contemporaneous scalar reference, rather than treating a saved
    # run's different compute budget as a controlled new comparison.
    rows = []
    reference = prepare(baseline.prepared.data, baseline.prepared.split, baseline.prepared.spec)
    if (not np.array_equal(reference.Phi, baseline.prepared.Phi)
            or not np.array_equal(reference.Z, baseline.prepared.Z)):
        raise ValueError("Current preparation changes the saved baseline features; establish a new qualified reference")
    for label in ("scalar_reference", *extensions):
        if source_fingerprint() != source:
            raise RuntimeError("Source changed during P5 ablation")
        allowance = memory_allowance(args.max_memory_mb)["effective_job_budget_bytes"]
        n, t = reference.Y.shape
        p, d = reference.Phi.shape[1], reference.fit_Z.shape[1]
        estimated = (512*1024**2 + 8*(12*d*d+8*p*p+8*n*d+4*p*d+4*d*t)
                     + 4*(args.draws+args.continuation)*4096
                     + 4*8*args.paths*len(evaluation["truth"])*len(evaluation["site_ids"]))
        if allowance is not None and estimated > allowance:
            raise MemoryError("Insufficient current headroom for the next bounded ablation fit; resume later")
        spec = baseline.prepared.spec
        if label == "grouped_ridge":
            spec = replace(spec, prior=PriorSpec.grouped(
                mean_log_precision=float(np.asarray(spec.prior.mean_log_precision).reshape(-1)[0]),
                sd_log_precision=float(np.asarray(spec.prior.sd_log_precision).reshape(-1)[0])))
        elif label == "full_residual_ar":
            spec = replace(spec, innovations="full_residual_ar")
        prepared = prepare(baseline.prepared.data, baseline.prepared.split, spec)
        if not np.array_equal(reference.Phi, prepared.Phi) or not np.array_equal(reference.Z, prepared.Z):
            raise AssertionError("P5 prior/dynamics ablation unexpectedly changed the frozen feature design")
        policy = baseline.sampler.parameterization if label == "scalar_reference" else "legacy"
        cfg = replace(baseline.sampler, chains=4, warmup=args.warmup, draws=args.draws,
            seed=seed_for(args.seed, "p5_mcmc", label), parameterization=policy,
            backend="auto", algorithm="collapsed", checkpoint_every=0,
            temporal="full_residual_ar" if spec.innovations == "full_residual_ar" else "iid")
        directory = output/label
        directory.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        row = {"variant":label, "prepared_identity":prepared.identity,
               "specification":spec.to_dict(), "seed":cfg.seed, "chains":4,
               "estimated_additional_memory_bytes":estimated,
               "memory_measurement":"conservative admission estimate, not measured peak"}
        try:
            fitted, gates, recovery = _recoverable_fit(prepared, cfg, args.continuation, CHECKS,
                seed_for(args.seed, "p5_diagnostics", label), directory/"chain.npz")
            row.update(qualified=bool(gates[-1]["passed"]), gates=gates, recovery=recovery,
                       fit_invocation_seconds=time.perf_counter()-start,
                       posterior_identity=fitted.posterior_identity)
            fitted.save(directory/"fitted.npz")
            if row["qualified"]:
                forecast_start = time.perf_counter()
                samples, _ = _forecast(fitted, evaluation, args.paths,
                    seed_for(args.seed, "p5_forecast", label), args.batch_size)
                forecast, scores = _score(samples, evaluation, seed_for(args.seed, "p5_scores"))
                forecast.save(directory/"forecast.npz")
                row.update(scores=scores, forecast_seconds=time.perf_counter()-forecast_start)
            else:
                row["disposition"] = "unqualified_no_native_probability_comparison"
        except CheckpointIntegrityError:
            raise
        except Exception as error:
            row.update(qualified=False, error=f"{type(error).__name__}: {error}",
                       fit_invocation_seconds=time.perf_counter()-start, disposition="failed_development_fit")
        if source_fingerprint() != source:
            raise RuntimeError("Source changed during P5 ablation; evidence must not be promoted")
        rows.append(jsonable(row))
        write_json(directory/"receipt.json", row)
        write_json(output/"paired_ablation.json", {"rows":rows, "complete":len(rows)==1+len(extensions),
            "evaluation":evaluation["metadata"], "claim":"One prespecified development dataset; no general superiority claim"})
        print(json.dumps({"variant":label, "qualified":row["qualified"],
                          "fit_invocation_seconds":row["fit_invocation_seconds"]}), flush=True)
        fitted = samples = forecast = None
    comparisons = []
    for row in rows[1:]:
        usable = "scores" in rows[0] and "scores" in row
        comparisons.append({"variant":row["variant"], "common_qualified_datasets":int(usable),
            "loss_difference_extension_minus_scalar":({key:row["scores"][key]-rows[0]["scores"][key]
                for key in ("rmse", "mae", "crps")} if usable else None), "mcse_across_datasets":None})
    result = {"rows":rows, "comparisons":comparisons, "complete":True,
        "evaluation":evaluation["metadata"], "claim":"One development dataset; inference failures retained; no automatic promotion",
        "scalar_reference_refitted":True, "observations_and_feature_specification_fixed":True}
    write_json(output/"paired_ablation.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--extensions", help="Comma-separated grouped_ridge,full_residual_ar; opt-in fits")
    modes.add_argument("--path-stability", action="store_true")
    parser.add_argument("--simulation", help="Existing saved Simulation NPZ, predictive modes only")
    parser.add_argument("--study-manifest", help="Existing development manifest JSON, predictive modes only")
    parser.add_argument("--track", choices=("A", "B"), default="A")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--inspection-draws-per-chain", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--continuation", type=int, default=2000)
    parser.add_argument("--paths", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-memory-mb", type=float, default=2048)
    args = parser.parse_args(argv)
    extensions = tuple(args.extensions.split(",")) if args.extensions else ()
    if len(set(extensions)) != len(extensions) or set(extensions)-{"grouped_ridge", "full_residual_ar"}:
        raise ValueError("Only distinct grouped_ridge/full_residual_ar ablations are supported here")
    if (args.seed < 0 or not 1<=args.inspection_draws_per_chain<=128 or args.warmup < 0
            or args.draws < 8 or args.continuation < 0 or not 1<=args.paths<=512 or not 1<=args.batch_size<=64):
        raise ValueError("Invalid bounded inspection, inference, or forecast budget")
    if (extensions or args.path_stability) and not (args.simulation and args.study_manifest):
        raise ValueError("Predictive modes require --simulation and --study-manifest from development")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scope = json.loads(SCOPE_PATH.read_text())
    with single_thread_runtime():
        source = source_fingerprint()
        baseline_sha256 = _hash_file(args.baseline)
        admission = _admit_archive(args.baseline, args.max_memory_mb)
        fitted, stored_gate = _load_baseline(args.baseline)
        if baseline_sha256 != _hash_file(args.baseline):
            raise ValueError("Baseline file changed while it was being loaded")
        design = {"schema":"p5-evaluation-1.0", "baseline_sha256":baseline_sha256,
            "source_fingerprint":source, "tool_sha256":_hash_file(__file__),
            "scope_sha256":_hash_file(SCOPE_PATH), "environment":benchmark_environment(),
            "settings":{key:value for key,value in vars(args).items()
                        if key not in {"output", "baseline", "simulation", "study_manifest", "batch_size", "max_memory_mb"}},
            "simulation_sha256":_hash_file(args.simulation) if extensions or args.path_stability else None,
            "development_manifest_sha256":_hash_file(args.study_manifest) if extensions or args.path_stability else None,
            "input_posterior_identity":fitted.posterior_identity}
        design = jsonable(design)
        design["identity"] = content_hash(design)
        design_path = output/"manifest.json"
        if design_path.exists() and json.loads(design_path.read_text()) != design:
            raise ValueError("P5 output belongs to a different source/runtime/input/design; use a new directory")
        write_json(design_path, design)
        diagnostic_seed = (stored_gate or {}).get("functional_policy", {}).get("seed",
                             seed_for(args.seed, "p5_baseline_diagnostics"))
        gate = diagnose(fitted, seed=diagnostic_seed, **CHECKS)
        qualified = bool(gate["passed"])
        inspection = inspect_fitted(fitted, draws_per_chain=args.inspection_draws_per_chain,
                                    seed=args.seed) if qualified else None
        result = {"manifest_identity":design["identity"], "qualified_baseline":qualified,
            "gate":gate, "inspection":inspection, "resource_admission":admission,
            "future_labels_used":False, "dispositions":_dispositions(scope, inspection, qualified)}
        write_json(output/"inspection.json", result)
        decisions = {"manifest_identity":design["identity"], "extensions":result["dispositions"],
                     "automatic_promotion":False}
        write_json(output/"decisions.json", decisions)
        if not qualified and (extensions or args.path_stability):
            raise ValueError("The saved baseline does not pass the fixed four-chain scientific gate; inspection was saved")
        if extensions or args.path_stability:
            evaluation = _evaluation(fitted, args.simulation, args.study_manifest, args.track,
                                     args.horizon, args.max_memory_mb)
            for key in ("simulation_sha256", "development_manifest_sha256"):
                if evaluation["metadata"][key] != design[key]:
                    raise ValueError("Evaluation inputs changed after the design was frozen")
            if extensions:
                result["ablation"] = _ablations(fitted, extensions, evaluation, output, args, source)
                attempted = {row["variant"]:row for row in result["ablation"]["rows"]}
                for decision in decisions["extensions"]:
                    if decision["id"] in attempted:
                        row = attempted[decision["id"]]
                        decision.update(scientific_ablation_executed=True, qualified=row["qualified"],
                            disposition="evaluated_on_one_development_dataset_not_promoted" if row["qualified"]
                                        else row.get("disposition", "unqualified"))
                write_json(output/"decisions.json", decisions)
            else:
                result["path_stability"] = _path_stability(fitted, evaluation, output, args.seed, args.batch_size)
        if source_fingerprint() != source:
            raise RuntimeError("Source changed during P5 evaluation")
    print(json.dumps({"output":str(output.resolve()), "qualified_baseline":qualified,
                      "mode":"extensions" if extensions else "path_stability" if args.path_stability else "inspect_only"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
