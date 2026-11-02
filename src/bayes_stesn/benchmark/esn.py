"""Observation-only adapter for the audited Bayesian spatial ESN.

Scientific qualification checks both hyperparameters and recovered readout
functionals. A bounded continuation reuses the exact saved chain; a failed
gate is a reported result, never a reason to weaken the thresholds.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import time
import numpy as np

from ..assessment import diagnose
from ..data import SplitPlan
from ..io import seed_for
from ..model import ModelSpec, FittedModel, prepare, fit
from ..prediction import forecast_context, simulate_predictive
from ..sampling import SamplerConfig
from .contracts import Panel, Forecast, validate_history, forecast_targets


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
        return Forecast(draws.observation.mean(axis=0), draws.observation, coordinates, ids,
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


def fit_esn(data: Panel, config=None, seed=0, *, checkpoint_path=None):
    if not isinstance(data, Panel):
        raise TypeError("ESN accepts only observed Panel inputs")
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
    features = {"modes":min(8, data.y.shape[1], conditioning//3),
                "units":(32,), "squares":16, "lags":(1,2),
                **options.pop("features", {})}
    features["seed"] = seed_for(seed, "reservoir")
    spec = ModelSpec.from_dict({"features":features,
                 "spatial_mean":{"side":3, "width":.3, **options.pop("spatial_mean", {})},
                 **options})
    prepared = prepare(data, SplitPlan(conditioning, len(data.y)), spec)
    cfg = SamplerConfig(**sampler)
    started = time.perf_counter()
    # Chain endpoints always save the exact continuation state. Benchmark fits
    # default to endpoint-only writes to avoid repeatedly serializing every
    # retained hyperstate. An explicit caller cadence remains authoritative.
    # The public SamplerConfig default is deliberately unchanged.
    with tempfile.TemporaryDirectory(prefix="bstesn-fit-") as temporary:
        checkpoint = Path(temporary)/"chain.npz" if checkpoint_path is None else Path(checkpoint_path)
        fitted = fit(prepared, cfg, checkpoint_path=checkpoint)
        gate = diagnose(fitted, seed=seed_for(seed, "diagnostics"), **checks)
        gates = [gate]
        if not gate["passed"] and continuation:
            cfg = replace(cfg, draws=cfg.draws+continuation)
            fitted = fit(prepared, cfg, checkpoint_path=checkpoint, resume=True)
            gate = diagnose(fitted, seed=seed_for(seed, "diagnostics"), **checks)
            gates.append(gate)
    return BayesianESN(fitted, {"qualified":bool(gate["passed"]), "gates":gates,
                "continuations":len(gates)-1, "fit_seconds":time.perf_counter()-started,
                "conditioning_rows":conditioning, "likelihood_rows":len(data.y)-conditioning,
                "spatial_basis":prepared.Phi.shape[1], "temporal_features":prepared.Z.shape[1],
                "checkpoint_cadence":{"every_iterations":int(cfg.checkpoint_every),
                    "chain_endpoints_always_saved":True,
                    "source":"caller" if "checkpoint_every" in sampler_options else "benchmark_default"},
                "uncertainty":"posterior_joint", "parameter_policy":"frozen"})
