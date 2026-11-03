"""Capability-aware model dispatch; optional dependencies are loaded on demand."""
from __future__ import annotations
from dataclasses import dataclass, asdict, fields
import json
import numpy as np
from ..io import save_state, load_state
from .contracts import Panel, CapabilityError


@dataclass(frozen=True)
class ModelInfo:
    name: str
    tracks: tuple[str, ...]
    uncertainty: str
    family: str
    stochastic_fit: bool = False
    selection_metric: str = "rmse"
    parameter_uncertainty: bool = False


MODELS = {
    "persistence":ModelInfo("Last-observation persistence", ("A",), "point", "classical"),
    "seasonal_naive":ModelInfo("Seasonal repetition", ("A",), "point", "classical"),
    "spatial_persistence":ModelInfo("Spatially interpolated persistence", ("A","B"), "point", "classical"),
    "basis_lgssm":ModelInfo("Coordinate-basis linear Gaussian state space", ("A","B"), "joint_plugin", "classical", selection_metric="crps"),
    "separable_stgp":ModelInfo("Separable spatial/temporal Matérn GP", ("A","B"), "joint_plugin", "classical", selection_metric="crps"),
    # Both the bandwidth subsample and Nyström landmarks use a fit-seed stream.
    "basis_krr":ModelInfo("Coordinate-basis kernel ridge", ("A","B"), "point", "classical", True),
    "bayes_esn_ridge":ModelInfo("Bayesian spatial ESN with ridge prior", ("A","B"), "posterior_joint", "bayesian", True, "crps", True),
    "eqesn_rollout":ModelInfo("Recursive quadratic reservoir ensemble", ("A",), "finite_ensemble", "reservoir", True, "crps"),
    "dcrnn":ModelInfo("Autoregressive diffusion-GRU comparator", ("A",), "point", "neural", True),
    "basis_gru":ModelInfo("Coordinate-basis probabilistic GRU", ("A","B"), "joint_plugin", "neural", True, "crps"),
}


def model_registry():
    return {key:asdict(value) for key,value in MODELS.items()}


def validate_model_options(model_id, supplied):
    """Reject misspelled keys before spending a candidate proposal budget."""
    if model_id not in MODELS or not isinstance(supplied,dict):
        raise ValueError("A registered model and option mapping are required")
    if model_id=="bayes_esn_ridge":
        from ..features import FeatureSpec
        from ..basis import SpatialMeanSpec
        from ..priors import PriorSpec
        from ..spatial import CovarianceSpec
        from ..sampling import SamplerConfig
        nested={"features":{f.name for f in fields(FeatureSpec)},
                "spatial_mean":{f.name for f in fields(SpatialMeanSpec)},
                "prior":{f.name for f in fields(PriorSpec)},
                "covariance":{f.name for f in fields(CovarianceSpec)},
                "sampler":{f.name for f in fields(SamplerConfig)},
                "diagnostics":{"functional_draws","rhat_limit","min_ess","min_chains"}}
        allowed=set(nested)|{"conditioning","innovations","continuation_draws"}
        if set(supplied)-allowed:
            raise ValueError(f"Unknown ESN options: {sorted(set(supplied)-allowed)}")
        for key,allowed_keys in nested.items():
            if key in supplied:
                if not isinstance(supplied[key],dict):
                    raise ValueError(f"ESN {key} must be a mapping")
                if set(supplied[key])-allowed_keys:
                    raise ValueError(f"Unknown ESN {key} options: {sorted(set(supplied[key])-allowed_keys)}")
        if (supplied.get("prior", {}).get("kind", "ridge") != "ridge"
                or supplied.get("innovations", "iid") != "iid"):
            raise ValueError("bayes_esn_ridge identifies the scalar-ridge iid Gaussian benchmark model; "
                             "use the core prepare/fit API for separately named prior or innovation extensions")
    elif model_id=="basis_gru":
        from .basis_gru import _configuration
        _configuration(supplied)
    elif model_id=="dcrnn":
        from .neural import _config
        _config(supplied)
    elif model_id=="eqesn_rollout":
        from .reservoir import _config
        _config(supplied)
    else:
        from .classical import _classical_config
        _classical_config(model_id,supplied)


def resolve_model_config(model_id, config, *, fitting_sites, fitting_rows):
    """Resolve a candidate without seeing responses or importing PyTorch.

    Shape-limited choices are clipped only where the adapter already has that
    mathematical behavior. Every such change is recorded. Defaults requiring
    response values remain named fitting rules, not leaked numerical values.
    """
    if model_id not in MODELS or not isinstance(config,dict):
        raise ValueError("A registered model and configuration mapping are required")
    validate_model_options(model_id,config)
    c=json.loads(json.dumps(config,allow_nan=False));changes=[];rules={}
    dimensions={"fitting_sites":int(fitting_sites),"fitting_rows":int(fitting_rows)}
    def clip(key, maximum, *, mapping=None, path=None):
        target=c if mapping is None else mapping
        value=target[key]
        if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<1 or maximum<1:
            raise ValueError(f"{path or key} must be positive and feasible")
        effective=min(int(value),int(maximum))
        if effective!=value:
            changes.append({"path":path or key,"requested":value,"effective":effective,
                            "reason":"actual fitting subset constraint"})
        target[key]=effective
    if model_id in {"dcrnn","basis_gru","eqesn_rollout"}:
        if model_id=="dcrnn":
            from .neural import _config, stopping_split
            c=_config(c)
            if c["graph_k"]>=fitting_sites:
                changes.append({"path":"graph_k","requested":c["graph_k"],
                                "effective":fitting_sites-1,"reason":"actual fitting subset constraint"})
                c["graph_k"]=fitting_sites-1
            cutoff,leads=stopping_split(fitting_rows,c)
            dimensions.update(stopping_training_rows=cutoff,stopping_validation_leads=leads,
                              hidden_size=c["hidden_size"],layers=c["layers"])
            rules={"center_scale":"allowed stopping prefix; supplied full history at selected-epoch refit"}
        elif model_id=="basis_gru":
            from .basis_gru import _configuration, stopping_split
            c=_configuration(c);clip("rank",fitting_sites)
            cutoff,leads=stopping_split(fitting_rows,c)
            dimensions.update(spatial_rank_cap=c["rank"],stopping_training_rows=cutoff,
                              stopping_validation_leads=leads,hidden_size=c["hidden_size"])
            rules={"center_scale":"allowed stopping prefix; supplied full history at selected-epoch refit",
                   "covariance_initial_scales":"RMS of allowed one-step training residuals"}
        else:
            from .reservoir import _config
            c=_config(c);clip("modes",min(fitting_sites,fitting_rows-1))
            if max(c["lags"],c["washout"])>=fitting_rows-2:
                raise ValueError("E-QESN lags/washout must leave at least three fitting rows")
            dimensions.update(modes=c["modes"],reservoir_units=c["width"],members=c["members"])
            rules={"center_scale_pca":"supplied observed training panel only"}
    elif model_id=="bayes_esn_ridge":
        from ..model import ModelSpec
        from ..sampling import SamplerConfig
        allowed={"conditioning","features","spatial_mean","prior","covariance","innovations",
                 "sampler","diagnostics","continuation_draws"}
        if set(c)-allowed:
            raise ValueError(f"Unknown ESN options: {sorted(set(c)-allowed)}")
        conditioning=c.get("conditioning",min(200,max(20,fitting_rows//5)))
        if isinstance(conditioning,bool) or not isinstance(conditioning,(int,np.integer)) or not 10<=conditioning<fitting_rows-5:
            raise ValueError("ESN conditioning must leave at least six likelihood rows")
        features={"modes":min(8,fitting_sites-1,conditioning//3),"units":[32],"squares":16,
                  "lags":[1,2],**c.get("features",{})}
        kind=features.get("kind","pca_esn")
        if kind=="frozen_encoder_esn":
            mode_limit=min(features.get("encoder_width",16),conditioning)-1
        elif kind in {"graph_features_esn","local_features_esn"}:
            mode_limit=min(fitting_sites,conditioning-1)
        else:
            mode_limit=min(fitting_sites,conditioning)-1
        clip("modes",mode_limit,mapping=features,path="features.modes")
        features["seed"]=0  # The adapter assigns the separate fit/reservoir stream.
        spec=ModelSpec.from_dict({"features":features,
              "spatial_mean":{"side":3,"width":.3,**c.get("spatial_mean",{})},
              **{key:c[key] for key in ("prior","covariance","innovations") if key in c}})
        features=asdict(spec.features)
        if kind=="graph_features_esn":
            clip("graph_neighbors",fitting_sites-1,mapping=features,path="features.graph_neighbors")
        if (any(isinstance(v,bool) or int(v)!=v or v<1 for v in features["units"])
            or not features["lags"] or tuple(features["lags"])!=tuple(sorted(set(features["lags"])))
            or min(features["lags"])<1):
            raise ValueError("Use positive reservoir units and increasing positive lags")
        layers=len(features["units"])
        if len(features["input_scales"])!=layers or any(not np.isfinite(x) or x<=0 for x in features["input_scales"]):
            raise ValueError("input_scales needs one positive finite value per reservoir layer")
        if features["leak_patterns"] is None:
            features["leak_patterns"]=[(.2,.5,.9)]+[(.05,.2,.6)]*max(0,layers-1)
            features["leak_patterns"]=features["leak_patterns"][:layers]
        if len(features["leak_patterns"])!=layers or any(not pattern or min(pattern)<=0 or max(pattern)>1 for pattern in features["leak_patterns"]):
            raise ValueError("One nonempty leak pattern in (0,1] is required per layer")
        units=sum(features["units"])
        if not 0<=features["squares"]<=units or not 0<=features["products"]<=units*(units-1)//2:
            raise ValueError("ESN square/product dimensions exceed reservoir coordinates")
        if not 0<features["recurrent_norm"]<1 or not 0<features["connectivity"]<=1:
            raise ValueError("Invalid recurrent norm/connectivity")
        if (features["score_bound"] is not None and
            (not np.isfinite(features["score_bound"]) or features["score_bound"]<=0)):
            raise ValueError("score_bound must be positive finite or None")
        sampler={"chains":4,"warmup":500,"draws":1000,"checkpoint_every":0,**c.get("sampler",{})}
        sampler["seed"]=0
        sampler=asdict(SamplerConfig(**sampler))
        checks={"functional_draws":512,"rhat_limit":1.01,"min_ess":400,"min_chains":4,**c.get("diagnostics",{})}
        if set(checks)-{"functional_draws","rhat_limit","min_ess","min_chains"}:
            raise ValueError("Unknown ESN diagnostic option")
        if checks["functional_draws"]<8 or checks["min_ess"]<=0 or checks["min_chains"]<1 or checks["rhat_limit"]<=1:
            raise ValueError("Invalid ESN diagnostic gate")
        continuation=c.get("continuation_draws",0)
        if isinstance(continuation,bool) or not isinstance(continuation,(int,np.integer)) or continuation<0:
            raise ValueError("continuation_draws must be a nonnegative integer")
        c={"conditioning":int(conditioning),"features":features,"spatial_mean":asdict(spec.spatial_mean),
           "prior":asdict(spec.prior),"covariance":asdict(spec.covariance),"innovations":spec.innovations,
           "sampler":sampler,"diagnostics":checks,"continuation_draws":int(continuation)}
        p=(len(spec.spatial_mean.centers) if spec.spatial_mean.centers is not None else spec.spatial_mean.side**2)+1+(2 if spec.spatial_mean.coordinate_terms else 0)
        d=1+len(features["lags"])*features["modes"]+units+features["squares"]+features["products"]
        dimensions.update(conditioning_rows=int(conditioning),likelihood_rows=fitting_rows-conditioning,
                          spatial_columns=p,temporal_columns=d,readout_coefficients=p*d)
        rules={"center_scale_pca":"conditioning prefix only",
               "feature_seed":"assigned by adapter from fit seed", "sampler_seed":"assigned by adapter from fit seed"}
    else:
        from .classical import _classical_config
        defaults={
            "persistence":{}, "seasonal_naive":{"period":None,"min_period":8,"max_period":200},
            "spatial_persistence":{"spatial_length":.2,"ridge":.01,"spatial_nu":1.5},
            "basis_krr":{"rank":12,"basis_width":.25,"lags":4,"bandwidth_multiplier":1.,
                         "ridge":1e-3,"exact":False,"max_exact_cases":2000,"nystrom_rank":128},
        }
        if model_id=="basis_lgssm":
            from .linear_state import DEFAULT_CONFIG
            defaults[model_id]=DEFAULT_CONFIG
        elif model_id=="separable_stgp":
            from .gaussian_gp import DEFAULT_CONFIG
            defaults[model_id]=DEFAULT_CONFIG
        if model_id not in defaults:
            raise ValueError(f"No resolver for {model_id}")
        supplied=_classical_config(model_id,c)
        if model_id=="basis_krr" and "penalty" in supplied:
            if "ridge" not in supplied:
                supplied["ridge"]=supplied["penalty"]
            supplied.pop("penalty")
        c=_classical_config(model_id,{**defaults[model_id],**supplied})
        if "rank" in c:
            clip("rank",fitting_sites); dimensions["spatial_rank_cap"]=c["rank"]
        for key in ("basis_width","spatial_length","temporal_length","ridge","bandwidth_multiplier",
                    "initial_variance","variance_floor","tolerance","ftol","gtol"):
            if key in c and (not np.isfinite(c[key]) or c[key]<=0):
                raise ValueError(f"{key} must be positive and finite")
        if model_id=="seasonal_naive" and c["period"] is not None and c["period"]>fitting_rows:
            raise ValueError("Seasonal period exceeds supplied history")
        if model_id=="basis_lgssm" and not 0<c["transition_norm_cap"]<1:
            raise ValueError("transition_norm_cap must lie in (0,1)")
        if model_id=="basis_krr":
            if c["lags"]>=fitting_rows-2:
                raise ValueError("KRR lags must leave at least three regression cases")
            if c["exact"]:
                if fitting_rows-c["lags"]>c["max_exact_cases"]:
                    raise ValueError("Exact KRR exceeds its declared case cap")
                c.pop("nystrom_rank",None)
            else:
                clip("nystrom_rank",fitting_rows-c["lags"])
                c.pop("max_exact_cases",None)
            if "bandwidth" in c:
                c.pop("bandwidth_multiplier",None)
            else:
                rules["bandwidth"]="median positive distance on a seeded training subsample times multiplier"
        if model_id=="separable_stgp":
            if c["spatial_nu"] not in (.5,1.5,2.5) or c["temporal_nu"] not in (.5,1.5):
                raise ValueError("Unsupported Matérn smoothness")
            if fitting_sites>c["max_fit_sites"]:
                raise ValueError("GP fitting sites exceed max_fit_sites")
            for key,rule in {"center":"training mean", "temporal_length":"10 * observed time spacing", "signal_sd":"max(training SD,1e-6)",
                             "noise_sd":"0.2 * max(training SD,1e-6)"}.items():
                if key not in c:
                    rules[key]=rule
                elif not np.isfinite(c[key]) or (key!="center" and c[key]<=0):
                    raise ValueError(f"Invalid {key}")
        if model_id in {"basis_lgssm","basis_krr","spatial_persistence"}:
            rules["center"]="supplied observed training mean"
    return {"config":json.loads(json.dumps(c,allow_nan=False)),"constraints":changes,
            "dimensions":dimensions,"data_dependent_defaults":rules}


def fit_model(model_id, data: Panel, config=None, seed=0, *, checkpoint_path=None):
    if model_id not in MODELS:
        raise CapabilityError(f"Unknown model {model_id!r}; inspect model_registry()")
    if not isinstance(data, Panel):
        raise TypeError("Model fitting accepts an observation Panel, never a Simulation")
    family=MODELS[model_id].family
    if family=="classical":
        from .classical import fit_classical
        return fit_classical(model_id, data, config=config, seed=seed)
    if family=="bayesian":
        from .esn import fit_esn
        return fit_esn(data, config=config, seed=seed, checkpoint_path=checkpoint_path)
    if family=="reservoir":
        from .reservoir import fit_reservoir
        return fit_reservoir(model_id, data, config=config, seed=seed)
    if model_id=="basis_gru":
        from .basis_gru import fit_basis_gru
        return fit_basis_gru(data, config=config, seed=seed)
    from .neural import fit_neural
    return fit_neural(model_id, data, config=config, seed=seed)


def save_model(model, path):
    """Numeric arrays + JSON only; no untrusted pickle or Torch object loading."""
    return save_state(path, model.to_state(), kind="benchmark_model")


def load_model(path):
    state=load_state(path, kind="benchmark_model")
    model_id=state["model_id"]
    family=MODELS[model_id].family
    if family=="classical":
        from .classical import classical_from_state
        return classical_from_state(state)
    if family=="bayesian":
        from .esn import BayesianESN
        return BayesianESN.from_state(state)
    if family=="reservoir":
        from .reservoir import reservoir_from_state
        return reservoir_from_state(state)
    if model_id=="basis_gru":
        from .basis_gru import basis_gru_from_state
        return basis_gru_from_state(state)
    from .neural import neural_from_state
    return neural_from_state(state)
