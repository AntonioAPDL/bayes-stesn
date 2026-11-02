"""Versioned experiment profiles and deterministic, validation-only searches."""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import json
import numpy as np
import yaml
from ..io import seed_for, content_hash

CORE_MODELS = ("persistence", "seasonal_naive", "basis_lgssm", "separable_stgp",
               "basis_krr", "bayes_esn_ridge", "eqesn_rollout", "dcrnn")
NATIVE_NEW_SITE = ("spatial_persistence", "basis_lgssm", "separable_stgp", "basis_krr",
                   "bayes_esn_ridge")
CORE_SCENARIOS = ("g1_gaussian_ar", "g2_spectral_transport", "g3_excitable_field", "g4_existing_fronts")


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

    def __post_init__(self):
        if self.schema_version != "3.0" or self.stage not in {"execution_test", "development", "confirmation"}:
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
        for key in ("generator_options", "model_options"):
            value=getattr(self,key)
            if value is not None and not isinstance(value,dict):
                raise ValueError(f"{key} must be a mapping or None")
            object.__setattr__(self,key,json.loads(json.dumps(value,allow_nan=False)))
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
        return content_hash(asdict(self))

    def to_dict(self):
        return asdict(self)

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
    else:
        raise ValueError("profile must be smoke, pilot, or main")
    return replace(base, **overrides)


def _choose(rng, items):
    return items[int(rng.integers(len(items)))]


def candidate_configs(model_id, protocol: BenchmarkProtocol, *, track="A", seed=0):
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
