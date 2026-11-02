"""Resource-piloted, protocol-locked controlled ESN forecasting experiment.

Commands, from the repository root::

    python benchmarks/controlled_study.py pilot --output report/controlled_study
    python benchmarks/controlled_study.py freeze --output report/controlled_study --replications 8
    python benchmarks/controlled_study.py run --output report/controlled_study

The final-stage DGP is constructed only after loading an immutable protocol
lock. Each case locks its validation-selected candidate before final truth
is passed to any scoring function. Selection never reads final metrics.
No report rendering, model-dependent stopping by final performance, or
silent substitution of failed MCMC fits occurs in this runner.
"""

from __future__ import annotations

import os
# Enforce one BLAS thread per worker before importing NumPy/SciPy. A process
# pool controls concurrency; nested BLAS pools would exceed the resource plan.
for _thread_name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback

import numpy as np
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bayes_stesn import (
    CovarianceSpec, FeatureSpec, ModelSpec, PriorSpec, SamplerConfig,
    SpatialMeanSpec, SpatioTemporalData, SplitPlan, fit, forecast_context,
    prepare, simulate_predictive, diagnose,
)
from bayes_stesn.io import atomic_npz, content_hash, environment, seed_for
from bayes_stesn.scoring import paired_bootstrap, score_predictions
from bayes_stesn.simulation import simulate_dataset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "controlled_study.yaml"
DEFAULT_OUTPUT = ROOT / "report" / "controlled_study"
CANDIDATES = ("ridge_short", "ridge_long", "ridge_expressive_long", "group_gaussian_expressive_long", "group_rhs_expressive_long")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_hashes() -> dict:
    """Identify the inference/prediction code used by the locked study."""
    scientific_modules = ("assessment", "_reservoir", "basis", "data", "diagnostics", "features",
                          "gaussian", "io", "model", "prediction", "priors", "sampling",
                          "scoring", "simulation", "spatial")
    paths = [ROOT / "src" / "bayes_stesn" / f"{name}.py" for name in scientific_modules]
    paths += [Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: dict, *, exclusive: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)


def load_config(path=DEFAULT_CONFIG) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    required = ("conditioning", "fitting_stop", "short_fit_times", "long_fit_times", "grid_size", "n_inputs", "paths")
    if any(not isinstance(config.get(key), int) or config[key] < 1 for key in required):
        raise ValueError("study sizes and boundaries must be positive integers")
    c, stop = config["conditioning"], config["fitting_stop"]
    if c < 3 or not 0 < config["short_fit_times"] < config["long_fit_times"] <= stop - c:
        raise ValueError("short and long likelihood windows must follow the same conditioning block")
    declared = tuple(config["candidate_names"])
    if len(declared) != 4 or declared[:3] != CANDIDATES[:3] or declared[3] not in CANDIDATES[3:]:
        raise ValueError("declare three controlled ridge contrasts and one explicit structured-prior candidate")
    if not 1 <= config["workers"] <= 4 or config["n_inputs"] > config["grid_size"]**2:
        raise ValueError("invalid worker or input-site budget")
    split_for(config, "ridge_short")
    split_for(config, "ridge_long")
    if len(config["validation_blocks"]) < 1 or len(config["final_blocks"]) < 1:
        raise ValueError("validation and final blocks must be explicitly declared")
    for name in declared:
        targets = draw_targets(config, name)
        if not targets or any(not isinstance(x, int) or x < 8 for x in targets) or targets != sorted(set(targets)):
            raise ValueError("draw targets must be strictly increasing integers >=8")
    if set(config["sampler"].get("draw_targets_by_candidate", {})) - set(CANDIDATES):
        raise ValueError("draw-target overrides must name declared candidates")
    if config["sampler"]["chains"] < 4 or config["sampler"]["min_ess"] < 400:
        raise ValueError("controlled-study convergence gates require four chains and ESS >=400")


def split_for(config: dict, candidate: str) -> SplitPlan:
    if candidate not in CANDIDATES:
        raise ValueError("unknown candidate")
    count = config["short_fit_times"] if candidate == "ridge_short" else config["long_fit_times"]
    return SplitPlan(
        conditioning=config["conditioning"], fitting_start=config["fitting_stop"] - count,
        fitting_stop=config["fitting_stop"],
        validation_blocks=tuple(map(tuple, config["validation_blocks"])),
        final_blocks=tuple(map(tuple, config["final_blocks"])),
    )


def candidate_spec(config: dict, name: str) -> ModelSpec:
    if name not in CANDIDATES:
        raise ValueError("unknown candidate")
    expressive = "expressive" in name
    features = FeatureSpec(
        modes=config["modes"], lags=tuple(config["lags"]),
        units=tuple(config["expressive_units"] if expressive else config["compact_units"]),
        squares=config["expressive_squares"] if expressive else config["compact_squares"],
        input_scales=(0.4, 0.8), leak_patterns=((0.2, 0.5, 0.9), (0.05, 0.2, 0.6)),
        seed=config["feature_seed"],
    )
    rhs = config.get("rhs_prior", {"tau0": 0.1, "slab_scale": 2.0, "slab_df": 4.0, "fixed_slab": False})
    if name.startswith("group_rhs"):
        prior = PriorSpec.group_rhs(unshrunk_groups=(), **rhs)
    elif name.startswith("group_gaussian"):
        prior = PriorSpec.grouped(unshrunk_groups=())
    else:
        prior = PriorSpec.ridge(unshrunk_groups=())
    return ModelSpec(
        features=features,
        spatial_mean=SpatialMeanSpec(side=config["spatial_basis_side"], width=config["spatial_basis_width"], bounds=((0, 1), (0, 1))),
        prior=prior, covariance=CovarianceSpec(nu=1.5), name=name,
    )


def draw_targets(config, candidate):
    return config["sampler"].get("draw_targets_by_candidate", {}).get(candidate, config["sampler"]["draw_targets"])


def choose_candidate(validation_records: dict) -> str | None:
    """Choose from validation losses and diagnostics only; no final argument."""
    eligible = [name for name, record in validation_records.items()
                if record.get("gate_passed", False) and record.get("validation") is not None
                and np.isfinite(record["validation"]["crps"]) and np.isfinite(record["validation"]["rmse"])]
    if not eligible:
        return None
    return min(eligible, key=lambda name: (validation_records[name]["validation"]["crps"],
                                          validation_records[name]["validation"]["rmse"],
                                          validation_records[name]["fit_seconds"], name))


def freeze_protocol(config: dict, output: Path, replications: int, pilot_summary: dict) -> dict:
    """Write an exclusive protocol lock before any final-stage generation."""
    validate_config(config)
    if not config["minimum_replications"] <= replications <= config["maximum_replications"]:
        raise ValueError("replication count lies outside the declared resource-planning range")
    output = Path(output)
    if (output / "final").exists() and any((output / "final").iterdir()):
        raise ValueError("cannot freeze a protocol after final-stage artifacts exist")
    protocol = {
        "schema_version": 1, "locked_at_utc": _now(), "config": config,
        "config_sha256": content_hash(config), "replications": replications,
        "replication_ids": list(range(replications)), "scenarios": list(config["scenarios"]),
        "stage": "final", "illustration_replication": config["illustration_replication"],
        "selection_rule": config["selection"], "failure_policy": config["failure_policy"],
        "candidate_specs": {name: candidate_spec(config, name).to_dict() for name in config["candidate_names"]},
        "splits": {name: asdict(split_for(config, name)) for name in config["candidate_names"]},
        "resource_pilot": pilot_summary, "environment": environment(),
        "source_sha256": source_hashes(),
        "claims": "Compact independent-replication experiment; no large-profile or universal-improvement claim.",
        "seed_manifest": [
            {"stage": "final", "replication": replication, "root_seed": config["root_seed"],
             "mcmc": {name: seed_for(config["root_seed"], "final", replication, name, "mcmc") for name in config["candidate_names"]}}
            for replication in range(replications)
        ],
    }
    # JSON serializes tuple axes as lists; hash the exact portable form so
    # a round trip does not change the identity of a legitimate lock.
    protocol = _json_safe(protocol)
    protocol["protocol_sha256"] = content_hash(protocol)
    write_json(output / "protocol.json", protocol, exclusive=True)
    return protocol


def load_protocol(output: Path) -> dict:
    path = Path(output) / "protocol.json"
    if not path.exists():
        raise ValueError("freeze the protocol before generating or scoring final-stage data")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    digest = protocol.pop("protocol_sha256")
    if content_hash(protocol) != digest or content_hash(protocol["config"]) != protocol["config_sha256"]:
        raise ValueError("protocol contents changed after freezing")
    protocol["protocol_sha256"] = digest
    return protocol


def fit_candidate(config, dataset, name, stage, replication, directory):
    indices = dataset.input_indices[config["n_inputs"]]
    # Only fitting history enters the fitting data constructor at all.
    fitting_stop = config["fitting_stop"]
    data = SpatioTemporalData(dataset.observation[:fitting_stop, indices], dataset.coordinates[indices],
                             times=dataset.times[:fitting_stop], site_ids=tuple(f"site_{i}" for i in indices))
    started = time.perf_counter()
    prepared = prepare(data, split_for(config, name), candidate_spec(config, name))
    preparation_seconds = time.perf_counter() - started
    checkpoint = directory / f"{name}.checkpoint.npz"
    history = []
    fitted = None
    for target in draw_targets(config, name):
        sampler = SamplerConfig(
            chains=config["sampler"]["chains"], warmup=config["sampler"]["warmup"], draws=target,
            seed=seed_for(config["root_seed"], stage, replication, name, "mcmc"),
            proposal_scale=config["sampler"]["proposal_scale"], checkpoint_every=config["sampler"]["checkpoint_every"],
            initialize=config["sampler"]["initialize"], init_maxiter=config["sampler"]["init_maxiter"],
            init_max_evaluations=config["sampler"]["init_max_evaluations"],
            init_max_dimension=config["sampler"]["init_max_dimension"],
            init_dispersion=config["sampler"]["init_dispersion"],
            rhs_tradeoff_scale=config["sampler"]["rhs_tradeoff_scale"],
        )
        extension_start = time.perf_counter()
        fitted = fit(prepared, sampler, checkpoint_path=checkpoint, resume=fitted is not None)
        diagnostic = fitted.diagnose(rhat_limit=config["sampler"]["rhat_limit"], min_ess=config["sampler"]["min_ess"])
        scientific = {"passed": False, "hyperparameters": diagnostic.to_dict(), "functionals": None,
                      "functional_policy": "not evaluated because hyperparameter gate failed"}
        if diagnostic.passed:
            scientific = diagnose(fitted, functional_draws=config["sampler"]["functional_draws"],
                                  seed=seed_for(config["root_seed"], stage, replication, name, "functionals"),
                                  rhat_limit=config["sampler"]["rhat_limit"], min_ess=config["sampler"]["min_ess"])
        history.append({"draws_per_chain": target, "extension_seconds": time.perf_counter() - extension_start,
                        "diagnostics": scientific, "acceptance": fitted.posterior.acceptance.tolist()})
        print(json.dumps({"event": "fit_gate", "stage": stage, "scenario": dataset.parameters["scenario"],
                          "replication": replication, "candidate": name, "draws": target,
                          "passed": scientific["passed"], "hyper_passed": diagnostic.passed,
                          "max_rhat": float(np.max(diagnostic.rhat)),
                          "min_bulk_ess": float(np.min(diagnostic.ess_bulk)),
                          "seconds": history[-1]["extension_seconds"]}), flush=True)
        if scientific["passed"]:
            break
    record = {
        "gate_passed": scientific["passed"], "diagnostics": scientific, "sampler_history": history,
        "fit_seconds": time.perf_counter() - started, "preparation_seconds": preparation_seconds,
        "fitting_window": [prepared.split.fitting_start, prepared.split.fitting_stop],
        "features": int(prepared.fit_Z.shape[1]), "spatial_basis": int(prepared.Phi.shape[1]),
        "prepared_identity": prepared.identity, "posterior_identity": fitted.posterior_identity,
        "sampler_metadata": fitted.posterior.metadata,
        "validation": None,
    }
    atomic_npz(directory / f"{name}.diagnostics.npz", chains=fitted.posterior.chains,
               parameter_names=np.asarray(fitted.posterior.parameter_names), log_density=fitted.posterior.log_density,
               acceptance=fitted.posterior.acceptance)
    return fitted, record


def predict_block(config, dataset, fitted, block, stage, replication, *, paths=None, stream="forecast"):
    first, stop = block
    indices = dataset.input_indices[config["n_inputs"]]
    history = SpatioTemporalData(dataset.observation[:first, indices], dataset.coordinates[indices],
                                times=dataset.times[:first], site_ids=tuple(f"site_{i}" for i in indices))
    context = forecast_context(fitted, origin=first - 1, history=history)
    return simulate_predictive(
        fitted, context, horizon=stop - first, target_coordinates=dataset.coordinates,
        target_site_ids=tuple(f"site_{i}" for i in range(len(dataset.coordinates))),
        paths=config["paths"] if paths is None else paths,
        seed=seed_for(config["root_seed"], stage, replication, stream, first, stop),
    )


def _scalar_scores(draws, truth, coordinates):
    score = score_predictions(draws, truth, coordinates=coordinates, energy_pairs=None, variogram_pairs=256)
    return {
        key: score[key] for key in ("rmse", "mae", "crps", "rmse_by_lead", "mae_by_lead", "crps_by_lead", "intervals", "energy", "energy_by_lead", "variogram", "variogram_by_lead")
    }


def score_block(prediction, dataset, block, input_indices):
    first, stop = block
    n = len(dataset.coordinates)
    domains = {"input": input_indices, "heldout": np.setdiff1d(np.arange(n), input_indices), "all": np.arange(n)}
    targets = {"observation": (prediction.observation, dataset.observation[first:stop]),
               "latent": (prediction.latent, dataset.latent[first:stop]),
               "generating_mean": (prediction.readout_mean, dataset.generating_mean[first:stop])}
    result = {}
    for domain, indices in domains.items():
        if len(indices) < 2:
            continue
        result[domain] = {name: _scalar_scores(draws[:, :, indices], truth[:, indices], dataset.coordinates[indices])
                          for name, (draws, truth) in targets.items()}
    return result


def _average_nested(values):
    first = values[0]
    if isinstance(first, dict):
        return {key: _average_nested([value[key] for value in values]) for key in first}
    array = np.asarray(values, dtype=float)
    return np.mean(array, axis=0)


def _forecast_summaries(prediction, prefix):
    lo90, hi90 = prediction.interval(0.9)
    lo95, hi95 = prediction.interval(0.95)
    return {prefix + key: value for key, value in {
        "mean": prediction.mean, "sd": prediction.observation.std(axis=0, ddof=1),
        "lower90": lo90, "upper90": hi90, "lower95": lo95, "upper95": hi95,
        "readout_mean": prediction.readout_mean.mean(axis=0),
        "latent_mean": prediction.latent.mean(axis=0), "example_path": prediction.observation[0],
        "parameter_draw_ids": prediction.parameter_draw_ids,
    }.items()}


def run_case(config, output, stage, scenario, replication, protocol_sha256=None, pilot_reuse=None):
    """Fit every candidate; lock validation choice; then evaluate final blocks."""
    if stage not in ("pilot", "final"):
        raise ValueError("stage must be pilot or final")
    if pilot_reuse is not None and stage != "pilot":
        raise ValueError("development-pilot record reuse is prohibited in final evaluation")
    output = Path(output)
    if stage == "final":
        protocol = load_protocol(output)
        if protocol_sha256 != protocol["protocol_sha256"] or content_hash(config) != protocol["config_sha256"]:
            raise ValueError("final run does not match its locked protocol")
        if scenario not in protocol["scenarios"] or replication not in protocol["replication_ids"]:
            raise ValueError("scenario/replication was not prespecified")
    directory = output / stage / scenario / f"replication_{replication:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "result.json").exists():
        existing = json.loads((directory / "result.json").read_text())
        if existing.get("protocol_sha256") != protocol_sha256 or existing.get("config_sha256") != content_hash(config):
            raise ValueError("existing result has a different protocol/configuration identity")
        return existing
    started = time.perf_counter()
    n_times = max(stop for _, stop in config["final_blocks"])
    if stage == "pilot":
        n_times = max(stop for _, stop in config["validation_blocks"])
    dataset = simulate_dataset(n_times, grid_size=config["grid_size"], input_sizes=(config["n_inputs"],),
                               scenario=scenario, seed=config["root_seed"], replication=replication, stage=stage)
    indices = dataset.input_indices[config["n_inputs"]]
    data_provenance = dataset.parameters
    records, fits = {}, {}
    for name in config["candidate_names"]:
        if pilot_reuse is not None and name in pilot_reuse["result"]["models"]:
            previous = pilot_reuse["result"]
            old_config = pilot_reuse["config"]
            same_spec = candidate_spec(config, name) == candidate_spec(old_config, name)
            same_split = split_for(config, name) == split_for(old_config, name)
            old_sampler = {key: value for key, value in old_config["sampler"].items()
                           if key not in {"draw_targets", "draw_targets_by_candidate", "checkpoint_every"}}
            new_sampler = {key: value for key, value in config["sampler"].items()
                           if key not in {"draw_targets", "draw_targets_by_candidate", "checkpoint_every"}}
            same_data = previous["data_provenance"]["observation_sha256"] == dataset.parameters["observation_sha256"]
            if same_spec and same_split and old_sampler == new_sampler and same_data:
                record = dict(previous["models"][name])
                record["reused_development_pilot_from"] = pilot_reuse["directory"]
                records[name] = record
                source = Path(pilot_reuse["directory"]) / f"{name}.diagnostics.npz"
                if source.exists():
                    shutil.copy2(source, directory / source.name)
                print(json.dumps({"event": "reuse_unchanged_development_fit", "candidate": name,
                                  "source": pilot_reuse["directory"]}), flush=True)
                continue
        try:
            fitted, record = fit_candidate(config, dataset, name, stage, replication, directory)
            fits[name] = fitted
            validation_scores = []
            prediction_start = time.perf_counter()
            # Validation is scored even when diagnostics fail, but such a
            # fit is explicitly ineligible for selection/final claims.
            for block in config["validation_blocks"]:
                prediction = predict_block(config, dataset, fitted, block, stage, replication, stream="validation")
                first, stop = block
                score = score_predictions(prediction.observation[:, :, indices], dataset.observation[first:stop, indices], joint_scores=False)
                validation_scores.append({"crps": score["crps"], "rmse": score["rmse"]})
            record["validation"] = _average_nested(validation_scores)
            record["validation_prediction_seconds"] = time.perf_counter() - prediction_start
            records[name] = record
        except Exception as error:
            records[name] = {"gate_passed": False, "validation": None, "fit_seconds": None,
                             "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        write_json(directory / "fit_progress.json", {"stage": stage, "scenario": scenario, "replication": replication, "models": records})
    chosen = choose_candidate(records)
    selection = {"selected": chosen, "locked_at_utc": _now(), "source": "validation only",
                 "protocol_sha256": protocol_sha256, "validation_records_sha256": content_hash(_json_safe(records)),
                 "candidates": {name: {"passed": record["gate_passed"], "validation": record.get("validation")} for name, record in records.items()}}
    selection_path = directory / "selection.json"
    if selection_path.exists():
        raise ValueError("an interrupted evaluated case already has a selection lock; inspect instead of overwriting it")
    write_json(selection_path, selection, exclusive=True)
    final_scores, forecast_arrays = {}, {}
    if stage == "final":
        for name, fitted in fits.items():
            if not records[name]["gate_passed"]:
                continue
            model_scores = []
            prediction_start = time.perf_counter()
            for block_index, block in enumerate(config["final_blocks"]):
                prediction = predict_block(config, dataset, fitted, block, stage, replication)
                model_scores.append(score_block(prediction, dataset, block, indices))
                forecast_arrays.update(_forecast_summaries(prediction, f"{name}__block{block_index}__"))
            final_scores[name] = {"by_origin": model_scores, "replication_average": _average_nested(model_scores)}
            records[name]["final_prediction_scoring_seconds"] = time.perf_counter() - prediction_start
        first_final = min(first for first, _ in config["final_blocks"])
        forecast_arrays.update(coordinates=dataset.coordinates, input_indices=indices, times=dataset.times[first_final:],
                               observation_truth=dataset.observation[first_final:], latent_truth=dataset.latent[first_final:],
                               generating_mean_truth=dataset.generating_mean[first_final:])
        atomic_npz(directory / "forecast_summaries.npz", **forecast_arrays)
        if replication == config["illustration_replication"] and chosen is not None:
            illustration_block = (first_final, n_times)
            illustration = predict_block(config, dataset, fits[chosen], illustration_block, stage, replication,
                                         paths=config["illustration_paths"], stream="locked_illustration")
            illustration_arrays = _forecast_summaries(illustration, "")
            illustration_arrays.update(observation_paths=illustration.observation,
                                       observation_truth=dataset.observation[first_final:],
                                       latent_truth=dataset.latent[first_final:], generating_mean_truth=dataset.generating_mean[first_final:],
                                       coordinates=dataset.coordinates, input_indices=indices, times=dataset.times[first_final:],
                                       selected_model=np.asarray(chosen), origin=np.asarray(first_final - 1),
                                       protocol_sha256=np.asarray(protocol_sha256))
            atomic_npz(directory / "illustration.npz", **illustration_arrays)
            fits[chosen].save(directory / "selected_fit.npz")
        # Baseline diagnostic: observed persistence on input sites only.
        # This has no fabricated spatial reconstruction or uncertainty law.
        persistence = []
        for first, stop in config["final_blocks"]:
            truth = dataset.observation[first:stop, indices]
            predicted = np.broadcast_to(dataset.observation[first - 1, indices], truth.shape)
            persistence.append({"rmse": float(np.sqrt(np.mean((predicted - truth)**2))), "mae": float(np.mean(np.abs(predicted - truth)))})
    else:
        persistence = []
    result = {
        "stage": stage, "scenario": scenario, "replication": replication, "protocol_sha256": protocol_sha256,
        "config_sha256": content_hash(config),
        "data_provenance": data_provenance, "selection": selection, "models": records,
        "final_scores": final_scores, "persistence_input_point_scores": persistence,
        "wall_seconds": time.perf_counter() - started,
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "environment": environment(), "completed_at_utc": _now(),
    }
    write_json(directory / "result.json", result)
    # Numeric hyperchains and the complete locked seeds remain reproducible;
    # finished checkpoints are temporary continuation aids, not result files.
    for path in directory.glob("*.checkpoint.npz"):
        path.unlink()
    print(json.dumps({"event": "case_complete", "stage": stage, "scenario": scenario, "replication": replication,
                      "selected": chosen, "wall_seconds": result["wall_seconds"]}), flush=True)
    return _json_safe(result)


def run_pilot(config: dict, output: Path, *, reuse_from: Path | None = None) -> dict:
    """Use a separate pilot stream; final-stage data are never generated."""
    output = Path(output)
    if (output / "protocol.json").exists():
        raise ValueError("resource pilot must precede the frozen final protocol")
    pilot_protocol = {"config": config, "source_sha256": source_hashes(), "created_at_utc": _now(),
                      "reuse_from": str(reuse_from) if reuse_from is not None else None,
                      "purpose": "development runtime, convergence, and prior calibration; not final performance evidence"}
    write_json(output / "pilot_protocol.json", pilot_protocol, exclusive=True)
    reuse = None
    if reuse_from is not None:
        previous_directory = Path(reuse_from) / "pilot" / config["scenarios"][0] / "replication_000"
        previous_protocol = json.loads((Path(reuse_from) / "pilot_protocol.json").read_text())
        reuse = {"result": json.loads((previous_directory / "result.json").read_text()),
                 "config": previous_protocol["config"], "directory": str(previous_directory)}
    started = time.perf_counter()
    case = run_case(config, output, "pilot", config["scenarios"][0], 0, pilot_reuse=reuse)
    # Budget includes three blocks of forecasting/scoring and a 35% cushion
    # for worker contention/anisotropy. It depends on runtime, not scores.
    fit_seconds = 0.0
    for name, record in case["models"].items():
        max_iterations = config["sampler"]["warmup"] + max(draw_targets(config, name))
        history = record.get("sampler_history", [])
        actual_draws = history[-1]["draws_per_chain"] if history else config["sampler"]["draw_targets"][0]
        actual_iterations = config["sampler"]["warmup"] + actual_draws
        fit_seconds += (record.get("fit_seconds") or 0) * max_iterations / actual_iterations
    validation_seconds = sum(record.get("validation_prediction_seconds") or 0 for record in case["models"].values())
    estimated_case_seconds = 1.35 * (fit_seconds + 4 * validation_seconds)
    projected = {count: estimated_case_seconds * count * len(config["scenarios"]) / config["workers"]
                 for count in range(config["minimum_replications"], config["maximum_replications"] + 1)}
    affordable = [count for count, seconds in projected.items() if seconds <= config["final_wall_budget_seconds"]]
    recommended = min(config["planned_replications"], max(affordable)) if affordable else None
    summary = {"stage": "pilot", "config_sha256": content_hash(config), "wall_seconds": time.perf_counter() - started,
               "case": {"scenario": case["scenario"], "wall_seconds": case["wall_seconds"],
                        "models": {name: {key: record.get(key) for key in ("gate_passed", "fit_seconds", "validation_prediction_seconds", "diagnostics", "sampler_history", "features", "error")}
                                   for name, record in case["models"].items()}},
               "estimated_case_seconds": estimated_case_seconds, "projected_final_wall_seconds": projected,
               "recommended_replications": recommended,
               "planning_rule": "runtime scaled conservatively to maximum retained-draw budget, plus four validation-prediction costs and a 35% cushion; no final-stage data generated",
               "created_at_utc": _now()}
    write_json(output / "pilot_summary.json", summary)
    return summary


def aggregate_results(protocol: dict, cases: list[dict]) -> dict:
    config = protocol["config"]
    summary = {"protocol_sha256": protocol["protocol_sha256"], "replications_per_scenario": protocol["replications"],
               "scenario_results": {}, "score_direction": "smaller is better", "bootstrap_unit": "independent simulation replication",
               "joint_score_domain": "one field per lead; exact empirical energy; fixed 256 geometry pairs for variograms"}
    for scenario in protocol["scenarios"]:
        local = sorted((case for case in cases if case["scenario"] == scenario), key=lambda case: case["replication"])
        model_means, failures, selection_counts = {}, {}, {name: 0 for name in config["candidate_names"]}
        for case in local:
            if case["selection"]["selected"] is not None:
                selection_counts[case["selection"]["selected"]] += 1
        for name in config["candidate_names"]:
            completed = [case["final_scores"][name]["replication_average"] for case in local if name in case["final_scores"]]
            model_means[name] = _average_nested(completed) if completed else None
            failures[name] = [case["replication"] for case in local if name not in case["final_scores"]]
        comparisons = {}
        pairs = [("ridge_long", "ridge_short"), ("ridge_expressive_long", "ridge_long"),
                 (config["candidate_names"][3], "ridge_expressive_long"), ("validation_selected", "ridge_short")]
        for left, right in pairs:
            differences = []
            for case in local:
                actual_left = case["selection"]["selected"] if left == "validation_selected" else left
                if actual_left not in case["final_scores"] or right not in case["final_scores"]:
                    continue
                aa = case["final_scores"][actual_left]["replication_average"]
                bb = case["final_scores"][right]["replication_average"]
                differences.append((case["replication"], aa, bb))
            entry = {"paired_replication_ids": [item[0] for item in differences],
                     "condition": "among replications where both fits passed the prespecified diagnostic gate", "domains": {}}
            for domain in ("input", "heldout", "all"):
                entry["domains"][domain] = {}
                for metric in ("crps", "rmse", "mae", "energy", "variogram"):
                    a = np.array([item[1][domain]["observation"][metric] for item in differences])
                    b = np.array([item[2][domain]["observation"][metric] for item in differences])
                    entry["domains"][domain][metric] = (paired_bootstrap(a, b, n_resamples=config["bootstrap_resamples"],
                        seed=seed_for(config["root_seed"], "bootstrap", scenario, left, right, domain, metric)) if len(a) >= 2 else None)
            comparisons[f"{left}_minus_{right}"] = entry
        summary["scenario_results"][scenario] = {"model_means": model_means, "failed_replication_ids": failures,
                                                  "selection_counts": selection_counts, "paired_comparisons": comparisons,
                                                  "case_wall_seconds": [case["wall_seconds"] for case in local]}
    return _json_safe(summary)


def run_final(output: Path) -> dict:
    protocol = load_protocol(output)
    if source_hashes() != protocol["source_sha256"]:
        raise ValueError("source files changed after protocol lock; inspect and predeclare a new study before final evaluation")
    config = protocol["config"]
    started = time.perf_counter()
    jobs = [(scenario, replication) for replication in protocol["replication_ids"] for scenario in protocol["scenarios"]]
    cases = []
    with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
        futures = {pool.submit(run_case, config, str(output), "final", scenario, replication, protocol["protocol_sha256"]): (scenario, replication)
                   for scenario, replication in jobs}
        for future in as_completed(futures):
            cases.append(future.result())
            write_json(Path(output) / "progress.json", {"completed_cases": len(cases), "planned_cases": len(jobs), "elapsed_seconds": time.perf_counter() - started})
    summary = aggregate_results(protocol, cases)
    summary["final_wall_seconds"] = time.perf_counter() - started
    summary["completed_cases"] = len(cases)
    summary["case_results"] = [{"scenario": case["scenario"], "replication": case["replication"],
                                "path": f"final/{case['scenario']}/replication_{case['replication']:03d}/result.json"} for case in cases]
    write_json(Path(output) / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pilot", "freeze", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replications", type=int)
    parser.add_argument("--reuse-pilot", type=Path, help="reuse only unchanged fits from this earlier development pilot")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "pilot":
        result = run_pilot(config, args.output, reuse_from=args.reuse_pilot)
    elif args.command == "freeze":
        pilot = json.loads((args.output / "pilot_summary.json").read_text())
        if pilot["config_sha256"] != content_hash(config):
            raise ValueError("pilot settings differ from the proposed frozen protocol")
        count = args.replications if args.replications is not None else pilot["recommended_replications"]
        if count is None:
            raise ValueError("pilot predicts insufficient budget: revise and rerun the resource profile before freezing")
        result = freeze_protocol(config, args.output, count, pilot)
    else:
        result = run_final(args.output)
    print(json.dumps({"command": args.command, "output": str(args.output), "completed": True,
                      "recommended_replications": result.get("recommended_replications"),
                      "replications": result.get("replications", result.get("replications_per_scenario"))}), flush=True)


if __name__ == "__main__":
    main()
