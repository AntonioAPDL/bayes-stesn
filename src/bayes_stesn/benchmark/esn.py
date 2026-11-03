"""Observation-only adapter for the audited Bayesian spatial ESN.

Scientific qualification checks both hyperparameters and recovered readout
functionals. A bounded continuation reuses the exact saved chain; a failed
gate is a reported result, never a reason to weaken the thresholds.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import tempfile
import time
from zipfile import BadZipFile
import numpy as np

from ..assessment import diagnose
from ..data import SplitPlan
from ..io import content_hash, environment, load_state, save_state, seed_for
from ..model import ModelSpec, FittedModel, prepare, fit
from ..prediction import forecast_context, simulate_predictive
from ..sampling import RegressionPosterior, SamplerConfig
from .contracts import Panel, Forecast, validate_history, forecast_targets


class CheckpointIntegrityError(ValueError):
    """Persisted computation identity/integrity is incompatible, not a bad fit."""


def _load_fit_artifact(path, kind):
    try:
        return load_state(path, kind=kind)
    except (ValueError, KeyError, TypeError, OSError, BadZipFile) as error:
        raise CheckpointIntegrityError(f"Cannot validate existing ESN {kind} artifact: {error}") from error


def _fit_checkpoint(prepared, config, checkpoint):
    resume = checkpoint.exists()
    try:
        return fit(prepared, config, checkpoint_path=checkpoint, resume=resume)
    except ValueError as error:
        # These are the core sampler's explicit checkpoint/transition checks,
        # not arbitrary model-domain ValueErrors, which remain fitting errors.
        if resume and str(error).startswith(("Checkpoint", "Cannot resume")):
            raise CheckpointIntegrityError(str(error)) from error
        raise


class BayesianESN:
    def __init__(self, fitted, diagnostics):
        self.fitted = fitted
        self.training = fitted.prepared.data
        self.diagnostics = diagnostics

    def forecast(self, horizon, target_coordinates=None, target_site_ids=None,
                 n_paths=256, seed=0, history=None):
        visible = validate_history(self.training, history)
        coordinates, ids = forecast_targets(self.training, target_coordinates, target_site_ids)
        context = forecast_context(self.fitted, origin=len(visible.y)-1, history=visible)
        draws = simulate_predictive(self.fitted, context, horizon=horizon,
                    target_coordinates=coordinates, target_site_ids=ids,
                    paths=n_paths, seed=seed, components=("observation",))
        return Forecast._from_owned(draws.observation.mean(axis=0), draws.observation, coordinates, ids,
                        {"model":"bayes_esn_ridge", "uncertainty":"posterior_joint",
                         "origin_index":len(visible.y), "parameter_policy":"frozen",
                         "qualified":bool(self.diagnostics["qualified"]),
                         "posterior_identity":self.fitted.posterior_identity,
                         "path_count":int(n_paths)})

    def to_state(self):
        return {"model_id":"bayes_esn_ridge", "fitted":self.fitted.to_state(),
                "diagnostics":self.diagnostics}

    @classmethod
    def from_state(cls, state):
        return cls(FittedModel.from_state(state["fitted"]), state["diagnostics"])


def _fit_state_path(checkpoint):
    return checkpoint.with_name(checkpoint.name+".fit-state.npz")


def _transition_config(config):
    """Checkpoint cadence is storage policy, not a different sampled chain."""
    value = dict(config)
    value.pop("checkpoint_every", None)
    value.setdefault("parameterization", "legacy")
    return value


def _fit_computation():
    """Local fit recovery additionally binds its implementation and runtime."""
    root = Path(__file__).resolve().parents[1]
    files = ("sampling.py", "model.py", "gaussian.py", "assessment.py", "diagnostics.py",
             "priors.py", "spatial.py", "features.py", "_reservoir.py", "basis.py",
             "dynamics.py", "data.py", "io.py", "benchmark/esn.py")
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode())
        digest.update((root/name).read_bytes())
    from threadpoolctl import threadpool_info
    runtime = environment()
    runtime["blas"] = [{key:item.get(key) for key in
                        ("user_api", "internal_api", "num_threads", "prefix", "version", "architecture")}
                       for item in threadpool_info() if item.get("user_api") == "blas"]
    return {"source_sha256":digest.hexdigest(), "runtime":runtime}


def _checkpoint_info(checkpoint, prepared, initial_config, maximum_draws):
    """Validate before any mutation; allow only the one declared draw target."""
    saved = _load_fit_artifact(checkpoint, "mcmc_checkpoint")
    old, current = _transition_config(saved["config"]), _transition_config(asdict(initial_config))
    saved_draws = old.pop("draws")
    initial_draws = current.pop("draws")
    if (saved.get("identity") != prepared.identity or old != current
            or saved_draws not in {initial_draws, maximum_draws}):
        raise CheckpointIntegrityError("Existing ESN checkpoint has a different model/data/config or draw allowance")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024), b""):
            digest.update(block)
    progress = {str(key):{"iteration":int(value["iteration"]),
                          "retained_draws":len(value["retained_states"])}
                for key, value in saved["chains"].items()}
    complete = (len(progress) == initial_config.chains
                and all(value["retained_draws"] == saved_draws
                        and value["iteration"] == initial_config.warmup+saved_draws*initial_config.thin
                        for value in progress.values()))
    return {"sha256":digest.hexdigest(), "draw_target":int(saved_draws),
            "chains":progress, "complete_at_target":bool(complete)}


def _diagnose_initial_prefix(fitted, initial_draws, diagnostic_seed, checks):
    """Recover a missing pre-continuation gate from an old retained prefix.

    Used only when importing a compatible pre-ledger checkpoint that already
    entered its declared continuation. It never restarts/extends that chain.
    Acceptance summaries are immaterial to diagnose(), which reads states.
    """
    posterior = fitted.posterior
    config = replace(fitted.sampler, draws=initial_draws)
    # diagnose() reads these states and draw_parameters() copies the selected
    # parameter state. Sharing this immutable prefix avoids duplicating every
    # retained hyperstate merely to reconstruct one missing gate.
    prefix = RegressionPosterior(posterior.Y, posterior.Phi, posterior.Z,
        posterior.coordinates, posterior.groups, posterior.prior, posterior.covariance,
        config, [chain[:initial_draws] for chain in posterior.states],
        posterior.log_density[:, :initial_draws], posterior.acceptance,
        dict(posterior.metadata), posterior.reference_Z)
    return diagnose(replace(fitted, posterior=prefix, sampler=config), seed=diagnostic_seed, **checks)


def _recoverable_fit(prepared, initial_config, continuation, checks, diagnostic_seed, checkpoint):
    """One initial fit, one optional extension, durable phase-aware recovery."""
    maximum = initial_config.draws+continuation
    state_path = _fit_state_path(checkpoint)
    identity = content_hash({"prepared":prepared.identity,
        "initial_sampler":_transition_config(asdict(initial_config)),
        "continuation_draws":continuation, "diagnostics":checks,
        "diagnostic_seed":diagnostic_seed, "computation":_fit_computation()})
    info = (_checkpoint_info(checkpoint, prepared, initial_config, maximum)
            if checkpoint.exists() else None)
    resumed = state_path.exists() or info is not None
    if state_path.exists():
        state = _load_fit_artifact(state_path, "benchmark_esn_fit")
        if (state.get("schema_version") != 1 or state.get("identity") != identity
                or state.get("initial_draws") != initial_config.draws
                or state.get("maximum_draws") != maximum):
            raise CheckpointIntegrityError("Existing ESN fit-state identity, runtime or declared budget differs")
        if state["phase"] not in {"initial_sampling", "initial_diagnostics",
                                  "continuation_sampling", "continuation_diagnostics", "complete"}:
            raise CheckpointIntegrityError("Unknown ESN fit recovery phase")
        if info is None and state["phase"] != "initial_sampling":
            raise CheckpointIntegrityError("ESN fit-state requires its missing MCMC checkpoint; existing state is preserved")
        if info is not None and info["draw_target"] > state["target_draws"]:
            raise CheckpointIntegrityError("ESN checkpoint is ahead of its declared fit-state phase")
        if state["phase"] == "complete" and (not info["complete_at_target"]
                or info["sha256"] != state["checkpoint"]["sha256"]):
            raise CheckpointIntegrityError("Completed ESN fit-state does not match its checkpoint")
    else:
        # Older artifacts have no benchmark phase receipt. Their config binds
        # either the initial target or its one declared continuation target.
        continued = info is not None and info["draw_target"] > initial_config.draws
        state = {"schema_version":1, "identity":identity,
            "initial_draws":initial_config.draws, "maximum_draws":maximum,
            "phase":"continuation_sampling" if continued else "initial_sampling",
            "target_draws":maximum if continued else initial_config.draws,
            "gates":[], "initial_gate_reconstructed":bool(continued),
            "imported_checkpoint_without_fit_state":info is not None,
            "checkpoint":info, "completed_phase_seconds":{}}
        save_state(state_path, state, kind="benchmark_esn_fit")

    fitted = None
    while True:
        phase = state["phase"]
        cfg = replace(initial_config, draws=int(state["target_draws"]))
        if phase.endswith("sampling"):
            start = time.perf_counter()
            fitted = _fit_checkpoint(prepared, cfg, checkpoint)
            state["completed_phase_seconds"][phase] = time.perf_counter()-start
            state["phase"] = phase.replace("sampling", "diagnostics")
            state["checkpoint"] = _checkpoint_info(checkpoint, prepared, initial_config, maximum)
            save_state(state_path, state, kind="benchmark_esn_fit")
            continue
        if fitted is None:
            fitted = _fit_checkpoint(prepared, cfg, checkpoint)
        if phase == "complete":
            if not state["gates"] or len(state["gates"]) > 2:
                raise CheckpointIntegrityError("Completed ESN fit-state has invalid diagnostic history")
            break
        if phase == "continuation_diagnostics" and not state["gates"]:
            state["gates"].append(_diagnose_initial_prefix(
                fitted, initial_config.draws, diagnostic_seed, checks))
        start = time.perf_counter()
        gate = diagnose(fitted, seed=diagnostic_seed, **checks)
        state["completed_phase_seconds"][phase] = time.perf_counter()-start
        state["gates"].append(gate)
        if phase == "initial_diagnostics" and not gate["passed"] and continuation:
            # Eligibility is executable: one finite/error-free completed fit
            # failing its unchanged diagnostic gate. fit/diagnose exceptions
            # do not silently earn another draw allowance.
            state["phase"] = "continuation_sampling"
            state["target_draws"] = maximum
            fitted = None
        else:
            state["phase"] = "complete"
        save_state(state_path, state, kind="benchmark_esn_fit")
    return fitted, state["gates"], {"resumed":resumed, "phase":state["phase"],
        "initial_draws":state["initial_draws"], "maximum_draws":state["maximum_draws"],
        "initial_gate_reconstructed":state["initial_gate_reconstructed"],
        "imported_checkpoint_without_fit_state":state["imported_checkpoint_without_fit_state"],
        "completed_phase_seconds":state["completed_phase_seconds"],
        "continuation_policy":"one extension after a finite error-free initial fit fails its diagnostic gate"}


def fit_esn(data: Panel, config=None, seed=0, *, checkpoint_path=None):
    if not isinstance(data, Panel):
        raise TypeError("ESN accepts only observed Panel inputs")
    if data.y.shape[1] < 2:
        raise ValueError("The ESN PCA encoder requires at least two observed sites")
    options = dict(config or {})
    allowed = {"conditioning", "features", "spatial_mean", "prior", "covariance",
               "innovations", "sampler", "diagnostics", "continuation_draws"}
    if set(options)-allowed:
        raise ValueError(f"Unknown ESN options: {sorted(set(options)-allowed)}")
    conditioning = int(options.pop("conditioning", min(200, max(20, len(data.y)//5))))
    if not 10 <= conditioning < len(data.y)-5:
        raise ValueError("ESN needs a conditioning prefix and at least six likelihood rows")
    sampler_options = options.pop("sampler", {})
    sampler = {"chains":4, "warmup":500, "draws":1000, "checkpoint_every":0,
               "seed":seed_for(seed, "mcmc"), **sampler_options}
    # A caller cannot accidentally reuse a candidate's random seed at final refit.
    sampler["seed"] = seed_for(seed, "mcmc")
    checks = {"functional_draws":512, "rhat_limit":1.01, "min_ess":400,
              "min_chains":4, **options.pop("diagnostics", {})}
    continuation = int(options.pop("continuation_draws", 0))
    if continuation < 0:
        raise ValueError("continuation_draws must be nonnegative")
    features = {"modes":min(8, data.y.shape[1]-1, conditioning//3),
                "units":(32,), "squares":16, "lags":(1,2),
                **options.pop("features", {})}
    features["seed"] = seed_for(seed, "reservoir")
    spec = ModelSpec.from_dict({"features":features,
                 "spatial_mean":{"side":3, "width":.3, **options.pop("spatial_mean", {})},
                 **options})
    if spec.prior.kind != "ridge" or spec.innovations != "iid":
        raise ValueError("bayes_esn_ridge identifies the scalar-ridge iid Gaussian benchmark model; "
                         "use the core prepare/fit API for separately named prior or innovation extensions")
    prepared = prepare(data, SplitPlan(conditioning, len(data.y)), spec)
    cfg = replace(SamplerConfig(**sampler),
                  temporal="full_residual_ar" if spec.innovations == "full_residual_ar" else "iid")
    started = time.perf_counter()
    # Chain endpoints always save the exact continuation state. Benchmark fits
    # default to endpoint-only writes to avoid repeatedly serializing every
    # retained hyperstate. An explicit caller cadence remains authoritative.
    # The public SamplerConfig default is deliberately unchanged.
    with tempfile.TemporaryDirectory(prefix="bstesn-fit-") as temporary:
        checkpoint = Path(temporary)/"chain.npz" if checkpoint_path is None else Path(checkpoint_path)
        fitted, gates, recovery = _recoverable_fit(prepared, cfg, continuation, checks,
            seed_for(seed, "diagnostics"), checkpoint)
    return BayesianESN(fitted, {"qualified":bool(gates[-1]["passed"]), "gates":gates,
                "continuations":len(gates)-1, "fit_seconds":time.perf_counter()-started,
                "recovery":recovery,
                "conditioning_rows":conditioning, "likelihood_rows":len(data.y)-conditioning,
                "spatial_basis":prepared.Phi.shape[1], "temporal_features":prepared.Z.shape[1],
                "checkpoint_cadence":{"every_iterations":int(cfg.checkpoint_every),
                    "chain_endpoints_always_saved":True,
                    "source":"caller" if "checkpoint_every" in sampler_options else "benchmark_default"},
                "uncertainty":"posterior_joint", "parameter_policy":"frozen"})
