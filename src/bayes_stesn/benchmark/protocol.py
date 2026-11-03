"""Versioned experiment profiles and deterministic, validation-only searches."""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import json
import math
import numpy as np
import yaml
from ..io import seed_for, content_hash

CORE_MODELS = ("persistence", "seasonal_naive", "basis_lgssm", "separable_stgp",
               "basis_krr", "bayes_esn_ridge", "eqesn_rollout", "dcrnn")
NATIVE_NEW_SITE = ("spatial_persistence", "basis_lgssm", "separable_stgp", "basis_krr",
                   "bayes_esn_ridge")
CORE_SCENARIOS = ("g1_gaussian_ar", "g2_spectral_transport", "g3_excitable_field", "g4_existing_fronts")
V03_MODELS = CORE_MODELS + ("basis_gru",)
_NEW_FIELDS = ("selection_policy", "spatial_validation_blocks", "replacement_finalists",
               "selection_mode", "frozen_configurations", "candidate_proposal_cap",
               "normalization_reference", "fit_time_limits_seconds")


def effective_models(protocol, track):
    """The declared roster, including the mandatory new-site point control."""
    from .registry import MODELS
    if track not in protocol.tracks:
        raise ValueError("Track is absent from this protocol")
    models = [m for m in protocol.models if track in MODELS[m].tracks]
    if track == "B" and "spatial_persistence" not in models:
        models.insert(0, "spatial_persistence")
    return tuple(models)


@dataclass(frozen=True)
class BenchmarkProtocol:
    schema_version: str = "3.0"
    name: str = "main"
    stage: str = "confirmation"
    initial_fit: int = 2000
    training_start: int = 0
    horizon: int = 100
    validation_blocks: int = 3
    n_side: int = 25
    n_observed: int = 100
    n_calibration: int = 20
    paths: int = 256
    trials: int = 12
    finalist_count: int = 2
    finalist_seeds: int = 3
    replications: int = 20
    master_seed: int = 20261102
    models: tuple[str, ...] = CORE_MODELS
    scenarios: tuple[str, ...] = CORE_SCENARIOS
    tracks: tuple[str, ...] = ("A", "B")
    mcmc_warmup: int = 500
    mcmc_draws: int = 1000
    mcmc_continuation: int = 1000
    mcmc_functional_draws: int = 512
    mcmc_chains: int = 4
    neural_epochs: int = 200
    ensemble_members: int = 30
    score_pairs: int = 512
    score_observed_count: int | None = None
    retain_paths_replications: tuple[int, ...] = (0,)
    generator_options: dict | None = None
    model_options: dict | None = None
    fhn_target_policy: str = "finite_semidiscrete"
    fhn_time_tolerance: float = .02
    selection_policy: str = "legacy"
    spatial_validation_blocks: int = 1
    replacement_finalists: int = 0
    selection_mode: str = "retuned_pipeline"
    frozen_configurations: dict | None = None
    candidate_proposal_cap: int = 4096
    normalization_reference: dict | None = None
    fit_time_limits_seconds: dict | None = None

    def __post_init__(self):
        if self.schema_version not in {"3.0", "4.0"} or self.stage not in {"execution_test", "development", "confirmation"}:
            raise ValueError("Unknown benchmark schema or stage")
        positive = ("initial_fit", "horizon", "validation_blocks", "n_side", "n_observed", "paths",
                    "trials", "finalist_count", "finalist_seeds", "replications", "mcmc_warmup",
                    "mcmc_draws", "mcmc_functional_draws", "mcmc_chains", "neural_epochs",
                    "ensemble_members", "score_pairs")
        for key in positive:
            value = getattr(self, key)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.n_calibration < 0 or self.mcmc_continuation < 0 or self.initial_fit-self.training_start < 40 or self.training_start<0:
            raise ValueError("Invalid calibration, continuation, or training size")
        for key in ("training_start", "n_calibration", "mcmc_continuation"):
            value=getattr(self,key)
            if not isinstance(value,(int,np.integer)) or isinstance(value,bool):
                raise ValueError(f"{key} must be an integer")
        if self.n_side < 2 or self.n_observed+self.n_calibration>self.n_side**2:
            raise ValueError("Grid and observed/calibration role counts do not agree")
        if self.fhn_target_policy!="finite_semidiscrete":
            raise ValueError("G3 currently supports a fixed semidiscrete grid target, without a continuum claim")
        if not np.isfinite(self.fhn_time_tolerance) or not 0<self.fhn_time_tolerance<=.02:
            raise ValueError("FHN time-step tolerance must lie in (0,.02]")
        if self.score_observed_count is not None and not 1<=self.score_observed_count<=self.n_observed:
            raise ValueError("score_observed_count must be inside the observed domain")
        if self.n_side**2 <= self.n_observed+self.n_calibration and "B" in self.tracks:
            raise ValueError("Track B needs disjoint never-observed sites")
        if self.trials > 12:
            raise ValueError("The registered candidate budget is at most 12 trials")
        from .generators import SCENARIOS
        if not self.models or set(self.models)-set(CORE_MODELS+NATIVE_NEW_SITE+("basis_gru",)):
            raise ValueError("Profile contains an unknown or gated model")
        if not self.scenarios or set(self.scenarios)-set(SCENARIOS):
            raise ValueError("Unknown benchmark scenario")
        if not self.tracks or set(self.tracks)-{"A", "B"}:
            raise ValueError("Registered profiles currently support tracks A and B")
        for key in ("models", "scenarios", "tracks", "retain_paths_replications"):
            values = tuple(getattr(self, key))
            if len(set(values)) != len(values):
                raise ValueError(f"Duplicate {key}")
            object.__setattr__(self, key, values)
        for key in ("generator_options", "model_options", "frozen_configurations", "normalization_reference",
                    "fit_time_limits_seconds"):
            value=getattr(self,key)
            if value is not None and not isinstance(value,dict):
                raise ValueError(f"{key} must be a mapping or None")
            object.__setattr__(self,key,json.loads(json.dumps(value,allow_nan=False)))
        if self.selection_policy not in {"legacy", "qualified_finalists"}:
            raise ValueError("Unknown selection_policy")
        if self.selection_mode not in {"retuned_pipeline", "development_frozen_configuration"}:
            raise ValueError("Unknown selection_mode")
        for key, minimum in (("spatial_validation_blocks", 1), ("replacement_finalists", 0),
                             ("candidate_proposal_cap", 1)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        if self.spatial_validation_blocks > 3:
            raise ValueError("At most three spatial validation blocks are registered")
        if self.schema_version == "3.0":
            legacy = (self.selection_policy == "legacy" and self.spatial_validation_blocks == 1
                      and self.replacement_finalists == 0 and self.selection_mode == "retuned_pipeline"
                      and self.frozen_configurations is None and self.candidate_proposal_cap == 4096
                      and self.normalization_reference is None and self.fit_time_limits_seconds is None)
            if not legacy:
                raise ValueError("New selection/normalization policies require benchmark schema 4.0")
        elif self.selection_policy != "qualified_finalists":
            raise ValueError("Schema 4.0 requires qualified_finalists selection")
        if self.schema_version == "4.0":
            from .registry import MODELS, validate_model_options
            for model, seconds in (self.fit_time_limits_seconds or {}).items():
                if model not in MODELS or isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not np.isfinite(seconds) or seconds <= 0:
                    raise ValueError("fit_time_limits_seconds needs registered model names and finite positive seconds")
            for model,options in (self.model_options or {}).items():
                validate_model_options(model,options)
            for key,options in (self.frozen_configurations or {}).items():
                parts=key.split("/")
                if len(parts)>2 or parts[0] not in MODELS or (len(parts)==2 and parts[1] not in MODELS[parts[0]].tracks):
                    raise ValueError(f"Invalid frozen model/track key: {key}")
                validate_model_options(parts[0],options)
        if self.selection_mode == "development_frozen_configuration" and not self.frozen_configurations:
            raise ValueError("Frozen-configuration selection needs declared frozen_configurations")
        if self.frozen_configurations and self.selection_mode != "development_frozen_configuration":
            raise ValueError("frozen_configurations requires development_frozen_configuration mode")
        if self.normalization_reference is not None:
            reference = {"floor":1e-12, "ddof":0, **self.normalization_reference}
            if set(reference) != {"start", "stop", "observed_count", "floor", "ddof"}:
                raise ValueError("normalization_reference needs start, stop, observed_count, floor and ddof")
            for key in ("start", "stop", "observed_count", "ddof"):
                value = reference[key]
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                    raise ValueError(f"normalization_reference.{key} must be an integer")
            if not (self.training_start <= reference["start"] < reference["stop"] <= self.initial_fit):
                raise ValueError("Normalization reference must lie inside every supplied initial training window")
            if not 1 <= reference["observed_count"] <= self.n_observed or reference["ddof"] not in (0,1):
                raise ValueError("Invalid normalization observed_count or ddof")
            if not np.isfinite(reference["floor"]) or reference["floor"] <= 0:
                raise ValueError("Normalization floor must be finite and positive")
            if (reference["stop"]-reference["start"])*reference["observed_count"] <= reference["ddof"]:
                raise ValueError("Normalization reference has insufficient samples for ddof")
            object.__setattr__(self, "normalization_reference", reference)
        if self.stage == "confirmation" and (self.mcmc_chains < 4 or self.replications < 20):
            raise ValueError("Confirmation requires at least 20 independent replications and four MCMC chains")
        if self.stage == "confirmation" and self.replications>60:
            raise ValueError("The registered confirmation replication budget is capped at60")
        json.dumps(asdict(self), allow_nan=False)

    @property
    def origins(self):
        return tuple(self.initial_fit+j*self.horizon for j in range(self.validation_blocks))

    @property
    def final_origin(self):
        return self.initial_fit+self.validation_blocks*self.horizon

    @property
    def n_times(self):
        return self.final_origin+self.horizon

    @property
    def identity(self):
        return content_hash(self.to_dict())

    def to_dict(self):
        value = asdict(self)
        if self.schema_version == "3.0":
            for key in _NEW_FIELDS:
                value.pop(key)
        return value

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(json.loads(json.dumps(self.to_dict())), sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Benchmark profile must be a YAML mapping")
        return cls(**value)


def profile(name="pilot", **overrides):
    """Small profiles preserve chronological/task semantics, not scientific power."""
    if name == "main":
        base = BenchmarkProtocol()
    elif name == "pilot":
        base = BenchmarkProtocol(name="pilot", stage="development", initial_fit=400,
            horizon=40, n_side=7, n_observed=25, n_calibration=4,
            paths=128, trials=3, finalist_count=2, finalist_seeds=2, replications=5,
            mcmc_warmup=400, mcmc_draws=1000, mcmc_continuation=1000,
            neural_epochs=80, ensemble_members=20,
            generator_options={"g3_excitable_field":{"grid_size":64, "dt":.05, "delta":2.}})
    elif name == "smoke":
        base = BenchmarkProtocol(name="smoke", stage="execution_test", initial_fit=120,
            horizon=8, validation_blocks=2, n_side=5, n_observed=10, n_calibration=3,
            paths=16, trials=1, finalist_count=1, finalist_seeds=1, replications=1,
            mcmc_warmup=20, mcmc_draws=30, mcmc_continuation=0,
            mcmc_functional_draws=16, mcmc_chains=2, neural_epochs=2, ensemble_members=3,
            scenarios=("g1_gaussian_ar",),
            generator_options={"g3_excitable_field":{"grid_size":32, "dt":.05, "delta":2.}})
    elif name in {"v03_smoke", "v03_pilot", "v03_main"}:
        legacy_name = name.removeprefix("v03_")
        legacy = profile(legacy_name)
        options = dict(schema_version="4.0", name=name, models=V03_MODELS,
                       selection_policy="qualified_finalists", spatial_validation_blocks=2,
                       replacement_finalists=2, trials=8, finalist_count=2, finalist_seeds=3,
                       mcmc_warmup=1000, mcmc_draws=2000, mcmc_continuation=2000)
        if legacy_name == "smoke":
            options.update(trials=1, finalist_count=1, finalist_seeds=1,
                           replacement_finalists=0, mcmc_warmup=20, mcmc_draws=30,
                           mcmc_continuation=0)
        base = replace(legacy, **options)
    else:
        raise ValueError("profile must be smoke, pilot, main, or a v03_ version")
    return replace(base, **overrides)


def _choose(rng, items):
    return items[int(rng.integers(len(items)))]


def _legacy_candidate_configs(model_id, protocol: BenchmarkProtocol, *, track="A", seed=0):
    """Predeclared search; every random draw precedes validation/test scoring.

    The same trial cap is not claimed to be the same compute budget. First
    candidates are conservative reference settings. Search seeds do not use
    responses and are separate from fit, forecast, and final seeds.
    """
    rng = np.random.default_rng(seed_for(seed, "candidates", model_id, track))
    candidates=[]
    count=1 if model_id in {"persistence", "spatial_persistence"} else protocol.trials
    for j in range(count):
        if model_id == "persistence":
            c={}
        elif model_id == "spatial_persistence":
            c={"spatial_length":.2, "ridge":.01}
        elif model_id == "seasonal_naive":
            c={"period":int(_choose(rng, [12,24,48,96])) if j else 24}
        elif model_id == "basis_lgssm":
            c={"rank":min(protocol.n_observed, 8 if j==0 else _choose(rng,[8,16,32])),
               "basis_width":.2 if j==0 else _choose(rng,[.08,.2,.5]),
               "max_iter":50 if protocol.stage!="execution_test" else 5}
        elif model_id == "separable_stgp":
            c={"spatial_nu":1.5 if j==0 else _choose(rng,[.5,1.5]),
               "temporal_nu":.5 if j==0 else _choose(rng,[.5,1.5]),
               "spatial_length":.2 if j==0 else _choose(rng,[.03,.06,.12,.25,.5,1.]),
               "temporal_length":16. if j==0 else _choose(rng,[2.,8.,32.,64.]),
               "max_iter":60 if protocol.stage!="execution_test" else 8}
        elif model_id == "basis_krr":
            c={"rank":min(12, protocol.n_observed), "lags":4 if j==0 else _choose(rng,[1,4,8,16]),
               "bandwidth_multiplier":1. if j==0 else _choose(rng,[.5,1.,2.]),
               "ridge":1e-3 if j==0 else _choose(rng,[1e-6,1e-4,1e-2,1.]),
               "nystrom_rank":min(128 if j==0 else _choose(rng,[128,256,512]), protocol.initial_fit//2)}
        elif model_id == "bayes_esn_ridge":
            width=32 if j==0 else _choose(rng,[32,64,128])
            layers=1 if j==0 else _choose(rng,[1,2])
            history_rows=protocol.initial_fit-protocol.training_start
            modes=min(8, protocol.n_observed, max(2,history_rows//30))
            conditioning=min(200, max(20,history_rows//5))
            c={"conditioning":conditioning,
               "features":{"modes":modes, "units":[width]*layers, "lags":[1,2],
                           "squares":min(32,width//2) if j==0 or rng.random()<.5 else 0,
                           "input_scales":[.5 if j==0 else _choose(rng,[.1,.5,1.])]*layers},
               "spatial_mean":{"side":3,"width":.3},
               "sampler":{"chains":protocol.mcmc_chains,"warmup":protocol.mcmc_warmup,
                          "draws":protocol.mcmc_draws},
               "diagnostics":{"functional_draws":protocol.mcmc_functional_draws},
               "continuation_draws":protocol.mcmc_continuation}
            if protocol.stage=="execution_test":
                c["features"].update(units=[6], squares=3, input_scales=[.5], modes=3)
        elif model_id == "eqesn_rollout":
            c={"width":64 if j==0 else _choose(rng,[64,128,256]),
               "lags":3 if j==0 else _choose(rng,[1,3,6]),
               "modes":min(8,protocol.n_observed),
               "ridge":1e-3 if j==0 else float(10**rng.uniform(-6,3)),
               "members":protocol.ensemble_members,
               "leak":.3 if j==0 else _choose(rng,[.1,.3,.7,1.]),
               "input_scale":.5 if j==0 else _choose(rng,[.1,.5,1.])}
        elif model_id == "dcrnn":
            c={"hidden_size":32 if j==0 else _choose(rng,[32,64]),
               "layers":1 if j==0 else _choose(rng,[1,2]),
               "graph_k":min(protocol.n_observed-1,8 if j==0 else _choose(rng,[4,8,12])),
               "diffusion_order":2 if j==0 else _choose(rng,[1,2,3]),
               "context":24 if j==0 else _choose(rng,[12,24,48]),
               "learning_rate":.001 if j==0 else _choose(rng,[.0003,.001,.003]),
               "weight_decay":1e-5 if j==0 else _choose(rng,[0.,1e-5,.001]),
               "max_epochs":protocol.neural_epochs,"patience":20,"batch_size":32}
        elif model_id == "basis_gru":
            c={"rank":min(8,protocol.n_observed),"hidden_size":32, "lags":8, "epochs":protocol.neural_epochs}
        else:
            raise ValueError(f"No search registered for {model_id}")
        c.update((protocol.model_options or {}).get(model_id, {}))
        # Preserve the declared cap while removing duplicate settings before fitting.
        if not any(content_hash(c)==content_hash(old) for old in candidates):
            candidates.append(c)
    return candidates


def recursive_overrides(base, overrides):
    """Merge mappings recursively; arrays/scalars are replaced, never extended."""
    if not isinstance(base, dict) or not isinstance(overrides, dict):
        raise TypeError("Configuration overrides must be mappings")
    result = json.loads(json.dumps(base, allow_nan=False))
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = recursive_overrides(result[key], value)
        else:
            result[key] = json.loads(json.dumps(value, allow_nan=False))
    return result


def _v03_space(model_id):
    """Finite, declared factors; sampling is without replacement in raw space."""
    spaces = {
        "persistence": {},
        "spatial_persistence": {"spatial_length":(.1,.2,.5), "ridge":(.001,.01,.1)},
        "seasonal_naive": {"period":(12,24,48,96)},
        "basis_lgssm": {"rank":(8,16,32), "basis_width":(.08,.2,.5)},
        "separable_stgp": {"spatial_nu":(.5,1.5), "temporal_nu":(.5,1.5),
            "spatial_length":(.03,.06,.12,.25,.5,1.), "temporal_length":(2.,8.,32.,64.)},
        "basis_krr": {"rank":(8,16,24), "basis_width":(.12,.25,.5), "lags":(1,4,8,16),
            "bandwidth_multiplier":(.5,1.,2.), "ridge":(1e-6,1e-4,1e-2,1.),
            "nystrom_rank":(128,256,512)},
        "bayes_esn_ridge": {"architecture":((32,),(64,),(128,),(32,32),(64,64)),
            "modes":(4,8,12), "lags":((1,2),(1,2,4),(1,4,8)),
            "side":(3,4,5), "width":(.15,.3,.5), "input_scale":(.1,.5,1.),
            "leak_pattern":((.1,.3,.7),(.2,.5,.9),(.05,.2,.6)), "quadratic":(False,True)},
        "eqesn_rollout": {"width":(64,128,256), "lags":(1,3,6), "modes":(4,8,12),
            "ridge":(1e-6,1e-4,1e-2,1.,100.), "leak":(.1,.3,.7,1.),
            "input_scale":(.1,.5,1.), "quadratic":(False,True)},
        "dcrnn": {"hidden_size":(32,64), "layers":(1,2), "context":(12,24,48),
            "graph_k":(4,8,12), "diffusion_order":(1,2),
            "learning_rate":(.0003,.001,.003), "weight_decay":(0.,1e-5,.001),
            "forecast_training_steps":(1,4,8)},
        "basis_gru": {"rank":(8,16,24), "basis_width":(.12,.25,.5), "lags":(8,16,32),
            "hidden_size":(16,32,64), "learning_rate":(.0003,.001,.003),
            "weight_decay":(0.,1e-4,.001)},
    }
    if model_id not in spaces:
        raise ValueError(f"No search registered for {model_id}")
    return spaces[model_id]


def _v03_candidate(model_id, values, protocol):
    """Apply stage-specific fitting budgets, without response-dependent choices."""
    execution_test = protocol.stage == "execution_test"
    c = dict(values)
    if model_id == "basis_lgssm":
        c.update(max_iter=5 if execution_test else 200, tolerance=1e-5)
    elif model_id == "separable_stgp":
        c.update(max_iter=8 if execution_test else 60,
                 continuation_max_iter=0 if execution_test else 120)
    elif model_id == "basis_gru":
        c.update(epochs=protocol.neural_epochs, validation_horizon=protocol.horizon,
                 require_full_validation_horizon=True, covariance_max_iter=8 if execution_test else 40,
                 covariance_continuation_max_iter=0 if execution_test else 80)
    elif model_id == "dcrnn":
        c.update(max_epochs=protocol.neural_epochs, patience=20, batch_size=32,
                 validation_horizon=protocol.horizon, require_full_validation_horizon=True)
    elif model_id == "eqesn_rollout":
        c["members"] = protocol.ensemble_members
    elif model_id == "bayes_esn_ridge":
        units = list(c.pop("architecture", (32,)))
        pattern = c.pop("leak_pattern", (.2,.5,.9))
        scale = c.pop("input_scale", .5)
        quadratic = c.pop("quadratic", True)
        features = {"units":units, "modes":c.pop("modes",8), "lags":list(c.pop("lags",(1,2))),
                    "input_scales":[scale]*len(units), "leak_patterns":[list(pattern)]*len(units),
                    "squares":min(32,sum(units)//2) if quadratic else 0}
        if execution_test:
            features.update(units=[6], squares=3, modes=3, input_scales=[.5], leak_patterns=[[.2,.5,.9]])
        c = {"conditioning":min(200,max(20,(protocol.initial_fit-protocol.training_start)//5)),
             "features":features, "spatial_mean":{"side":c.pop("side",3),"width":c.pop("width",.3)},
             "sampler":{"chains":protocol.mcmc_chains, "warmup":protocol.mcmc_warmup,
                        "draws":protocol.mcmc_draws, "checkpoint_every":0},
             "diagnostics":{"functional_draws":protocol.mcmc_functional_draws},
             "continuation_draws":protocol.mcmc_continuation}
    return c


def _reference_candidate(model_id, protocol):
    values = {
        "persistence":{}, "spatial_persistence":{"spatial_length":.2,"ridge":.01},
        "seasonal_naive":{"period":24}, "basis_lgssm":{"rank":8,"basis_width":.2},
        "separable_stgp":{"spatial_nu":1.5,"temporal_nu":.5,"spatial_length":.2,"temporal_length":16.},
        "basis_krr":{"rank":12,"basis_width":.25,"lags":4,"bandwidth_multiplier":1.,"ridge":1e-3,"nystrom_rank":128},
        "bayes_esn_ridge":{"architecture":(32,),"modes":8,"lags":(1,2),"side":3,"width":.3,
                          "input_scale":.5,"leak_pattern":(.2,.5,.9),"quadratic":True},
        "eqesn_rollout":{"width":64,"lags":3,"modes":8,"ridge":1e-3,"leak":.3,"input_scale":.5,"quadratic":True},
        "dcrnn":{"hidden_size":32,"layers":1,"graph_k":8,"diffusion_order":2,"context":24,
                 "learning_rate":.001,"weight_decay":1e-5,"forecast_training_steps":4},
        "basis_gru":{"rank":8,"basis_width":.25,"hidden_size":32,"lags":8,"learning_rate":.001,"weight_decay":1e-4},
    }
    return _v03_candidate(model_id, values[model_id], protocol)


def candidate_manifest(model_id, protocol: BenchmarkProtocol, *, track="A", seed=0,
                       fitting_sites=None, fitting_rows=None):
    """Return effective candidates plus a reproducible search-design receipt.

    Only declared dimensions/geometry enter resolution. Data-dependent fitted
    values (e.g. GP initial scales) are documented as rules, never learned here.
    Caller-supplied fitting_sites should be the smallest actual spatial fold.
    """
    from .registry import MODELS, resolve_model_config, validate_model_options
    if model_id not in MODELS or track not in MODELS[model_id].tracks:
        raise ValueError(f"{model_id} does not support track {track}")
    if protocol.schema_version == "3.0":
        candidates = _legacy_candidate_configs(model_id, protocol, track=track, seed=seed)
        return {"candidates":candidates, "candidate_hashes":[content_hash(c) for c in candidates],
                "metadata":{"schema":"legacy-candidate-design-1.0", "requested":protocol.trials,
                            "effective_count":len(candidates), "policy":"unchanged_schema_3.0"}}
    frozen = protocol.selection_mode == "development_frozen_configuration"
    available_rows=(protocol.final_origin if frozen else protocol.initial_fit)-protocol.training_start
    sites = protocol.n_observed if fitting_sites is None else fitting_sites
    rows = available_rows if fitting_rows is None else fitting_rows
    if fitting_sites is None and track == "B" and not frozen:
        sites -= max(2,int(np.ceil(.2*sites)))
    for label,value in (("fitting_sites",sites),("fitting_rows",rows)):
        if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<1:
            raise ValueError(f"{label} must be a positive integer")
    if sites > protocol.n_observed or rows > available_rows:
        raise ValueError("Candidate fitting dimensions cannot exceed the allowed observed fitting panel")
    requested = 1 if model_id == "persistence" else protocol.trials
    overrides = (protocol.model_options or {}).get(model_id, {})
    if not isinstance(overrides,dict):
        raise ValueError("Per-model overrides must be a mapping")
    validate_model_options(model_id,overrides)
    if frozen:
        declared = protocol.frozen_configurations or {}
        key = f"{model_id}/{track}" if f"{model_id}/{track}" in declared else model_id
        if key not in declared and model_id != "spatial_persistence":
            raise ValueError(f"No frozen configuration declared for {model_id}/{track}")
        if overrides:
            raise ValueError("Frozen configurations are authoritative; remove additional model_options overrides")
        fixed = declared.get(key, {"spatial_length": .2, "ridge": .01})
        proposals = [("frozen" if key in declared else "fixed_spatial_control", fixed)]
        requested=1; total_space=1; proposal_limit=1
    else:
        space = _v03_space(model_id)
        keys = tuple(space)
        lengths = tuple(len(space[key]) for key in keys)
        total_space = math.prod(lengths)
        # Keep a stable raw sequence when trials increases: the number sampled
        # depends on the declared proposal cap, not requested effective count.
        rng = np.random.default_rng(seed_for(seed,"candidates-v4",model_id,track))
        proposal_limit = min(protocol.candidate_proposal_cap, total_space+1)
        raw_indices = rng.choice(total_space,size=min(total_space,max(0,proposal_limit-1)),replace=False)
        def proposed():
            yield "reference", _reference_candidate(model_id,protocol)
            for raw_index in raw_indices:
                index=int(raw_index); positions=[]
                for length in reversed(lengths):
                    positions.append(index%length); index//=length
                factors={key:space[key][position] for key,position in zip(keys,reversed(positions))}
                yield int(raw_index),_v03_candidate(model_id,factors,protocol)
        proposals=proposed()
    candidates=[];hashes=[];records=[];seen=set();invalid=[];duplicates=0;raw_count=0
    for label,raw in proposals:
        raw_count+=1
        try:
            merged=recursive_overrides(raw,overrides)
            resolved=resolve_model_config(model_id,merged,fitting_sites=int(sites),fitting_rows=int(rows))
            config=resolved["config"]; identity=content_hash(config)
        except (ValueError,TypeError) as error:
            invalid.append({"proposal":label,"reason":str(error)})
            continue
        if identity in seen:
            duplicates+=1;continue
        seen.add(identity); candidates.append(config);hashes.append(identity)
        records.append({"proposal":label,"hash":identity,
                        "constraints":resolved["constraints"],"dimensions":resolved["dimensions"],
                        "data_dependent_defaults":resolved["data_dependent_defaults"]})
        if len(candidates)>=requested:
            break
    if not candidates:
        detail=invalid[0]["reason"] if invalid else "empty candidate space"
        raise ValueError(f"No valid effective configuration for {model_id}: {detail}")
    reason=("requested_count_reached" if len(candidates)>=requested else
            "finite_space_exhausted" if frozen or raw_count>=total_space+1 else "proposal_cap_reached")
    metadata={"schema":"effective-candidate-design-1.0", "selection_mode":protocol.selection_mode,
              "requested":requested,"raw_proposals":raw_count,"effective_count":len(candidates),
              "duplicates":duplicates,"invalid_proposals":invalid,"stop_reason":reason,
              "raw_factorial_size":total_space,"proposal_cap":proposal_limit,
              "fitting_sites":int(sites),"fitting_rows":int(rows),"candidate_records":records,
              "seed_policy":"fit streams keyed by effective candidate hash; no responses used"}
    metadata["identity"]=content_hash(metadata)
    return {"candidates":candidates,"candidate_hashes":hashes,"metadata":metadata}


def candidate_configs(model_id, protocol: BenchmarkProtocol, *, track="A", seed=0,
                      fitting_sites=None, fitting_rows=None):
    """Backward-compatible list API; see candidate_manifest for provenance."""
    return candidate_manifest(model_id,protocol,track=track,seed=seed,
                              fitting_sites=fitting_sites,fitting_rows=fitting_rows)["candidates"]
