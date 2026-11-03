"""A compact high-level API with explicit data, feature and posterior identities."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
import time
import numpy as np
from .data import SpatioTemporalData, SplitPlan, readonly
from .features import FeatureSpec, FrozenFeatures
from .basis import SpatialMeanSpec, SpatialMean
from .priors import PriorSpec
from .spatial import CovarianceSpec
from .io import content_hash, save_state, load_state, environment


@dataclass(frozen=True)
class ModelSpec:
    features: FeatureSpec = field(default_factory=FeatureSpec)
    spatial_mean: SpatialMeanSpec = field(default_factory=SpatialMeanSpec)
    prior: PriorSpec = field(default_factory=PriorSpec.ridge)
    covariance: CovarianceSpec = field(default_factory=CovarianceSpec)
    innovations: str = "iid"
    name: str = "stesn"

    def __post_init__(self):
        if self.innovations not in {"iid", "full_residual_ar", "latent_ar", "field_t", "local_t", "heteroskedastic"}:
            raise ValueError("Unknown innovation hierarchy")
        for name, cls in (("features", FeatureSpec), ("spatial_mean", SpatialMeanSpec),
                          ("prior", PriorSpec), ("covariance", CovarianceSpec)):
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f"{name} must be a {cls.__name__}")
        if self.innovations in {"latent_ar", "local_t"} and self.covariance.static_variance:
            raise ValueError("latent_ar and local_t currently require an iid measurement nugget")
        if self.innovations not in {"iid", "full_residual_ar"} and self.covariance.scale_mode == "separate":
            raise ValueError("Separate physical PC scales are implemented for Gaussian residual models only")

    @classmethod
    def v3(cls):
        return cls(features=FeatureSpec.v3(), spatial_mean=SpatialMeanSpec.v3(), name="v3_reference")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for key, constructor in (("features", FeatureSpec), ("spatial_mean", SpatialMeanSpec),
                                 ("prior", PriorSpec), ("covariance", CovarianceSpec)):
            if key in value:
                value[key] = constructor(**value[key])
        return cls(**value)


@dataclass(frozen=True)
class PreparedModel:
    spec: ModelSpec
    data: SpatioTemporalData
    split: SplitPlan
    features: FrozenFeatures
    basis: SpatialMean
    response_center: float
    response_scale: float
    standardized_y: np.ndarray
    Phi: np.ndarray
    Z: np.ndarray
    reference_Z: np.ndarray
    groups: tuple[str, ...]
    identity: str
    variance_covariates: np.ndarray | None = None

    @property
    def training_end(self):
        return self.split.fitting_stop - 1

    @property
    def Y(self):
        return self.standardized_y[self.split.fitting_start:self.split.fitting_stop].T

    @property
    def fit_Z(self):
        return self.Z[self.split.fitting_start:self.split.fitting_stop]

    def to_state(self):
        data = {"y": self.data.y, "coordinates": self.data.coordinates, "times": self.data.times,
                "site_ids": self.data.site_ids, "response_units": self.data.response_units,
                "coordinate_units": self.data.coordinate_units, "metadata": dict(self.data.metadata)}
        return {"spec": self.spec.to_dict(), "data": data, "split": asdict(self.split),
                "features": self.features.to_state(), "basis": self.basis.to_state(),
                "response_center": self.response_center, "response_scale": self.response_scale,
                "standardized_y": self.standardized_y, "Phi": self.Phi, "Z": self.Z,
                "reference_Z": self.reference_Z, "groups": self.groups, "identity": self.identity,
                "variance_covariates": self.variance_covariates}

    @classmethod
    def from_state(cls, s):
        return cls(spec=ModelSpec.from_dict(s["spec"]), data=SpatioTemporalData(**s["data"]),
                   split=SplitPlan(**s["split"]), features=FrozenFeatures.from_state(s["features"]),
                   basis=SpatialMean.from_state(s["basis"]),
                   **{k: s[k] for k in ("response_center", "response_scale", "standardized_y", "Phi",
                                        "Z", "reference_Z", "groups", "identity", "variance_covariates")})


def prepare(data: SpatioTemporalData, split: SplitPlan, spec: ModelSpec | None = None,
            *, variance_covariates=None) -> PreparedModel:
    """Freeze preprocessing and discard all rows after the fitting endpoint.

    This function has no random stream beyond the explicit feature seed and
    never reads a latent truth object. Values after fitting_stop cannot change
    either the returned features or its model/data identity.
    """
    if not isinstance(data, SpatioTemporalData) or not isinstance(split, SplitPlan):
        raise TypeError("prepare requires observed SpatioTemporalData and a SplitPlan")
    spec = ModelSpec() if spec is None else spec
    if split.fitting_stop > len(data.y):
        raise ValueError("Fitting endpoint exceeds available observed history")
    historical = data.historical(split.fitting_stop)
    initial = historical.y[:split.conditioning]
    center, scale = float(initial.mean()), float(initial.std())
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("Conditioning observations need nonzero response variation")
    y = readonly((historical.y-center)/scale)
    base = None
    if spec.prior.kind == "correction":
        base = spec.prior.base_columns or (1+len(spec.features.lags)*spec.features.modes)
    features = FrozenFeatures.fit(y, split.conditioning, data.coordinates, spec.features, correction_base=base)
    basis = SpatialMean.fit(data.coordinates, spec.spatial_mean)
    Phi, Z = readonly(basis.transform(data.coordinates)), readonly(features.Z)
    reference_Z = readonly(Z[split.conditioning//3:split.conditioning])
    F = None
    if spec.innovations == "heteroskedastic":
        if variance_covariates is None:
            raise ValueError("heteroskedastic needs observed-history variance_covariates[time,site,covariate]")
        F = readonly(np.asarray(variance_covariates)[:split.fitting_stop])
        if F.ndim != 3 or F.shape[:2] != y.shape or not np.isfinite(F).all():
            raise ValueError("variance covariates must be finite and aligned with historical rows and sites")
    elif variance_covariates is not None:
        raise ValueError("variance_covariates are used only by the heteroskedastic hierarchy")
    identity = content_hash({"data": [y, data.coordinates, historical.times, historical.site_ids],
                             "split": asdict(split), "spec": spec.to_dict(),
                             "features": features.fingerprint, "variance_covariates": F})
    return PreparedModel(spec, historical, split, features, basis, center, scale, y, Phi, Z,
                         reference_Z, features.groups, identity, F)


@dataclass(frozen=True)
class FittedModel:
    prepared: PreparedModel
    posterior: object
    sampler: object
    metadata: dict
    posterior_identity: str

    @property
    def training_end(self):
        return self.prepared.training_end

    def diagnose(self, **kwargs):
        from .diagnostics import summarize_chains
        return summarize_chains(self.posterior.chains, self.posterior.parameter_names, **kwargs)

    def save(self, path):
        return save_state(path, self.to_state(), kind="fitted_model")

    def to_state(self):
        return {"prepared": self.prepared.to_state(), "posterior": self.posterior.to_state(),
                "sampler": asdict(self.sampler), "metadata": self.metadata,
                "posterior_identity": self.posterior_identity}

    @classmethod
    def load(cls, path):
        return cls.from_state(load_state(path, kind="fitted_model"))

    @classmethod
    def from_state(cls, state):
        from .sampling import SamplerConfig, RegressionPosterior
        prepared = PreparedModel.from_state(state["prepared"])
        config = SamplerConfig(**state["sampler"])
        args = dict(Y=prepared.Y, Phi=prepared.Phi, Z=prepared.fit_Z,
                    coordinates=prepared.data.coordinates, groups=prepared.groups,
                    prior=prepared.spec.prior, covariance=prepared.spec.covariance,
                    config=config, reference_Z=prepared.reference_Z)
        posterior = RegressionPosterior.from_state(state["posterior"], **args)
        return cls(prepared, posterior, config, state["metadata"], state["posterior_identity"])


def fit(prepared: PreparedModel, config=None, *, checkpoint_path=None, resume=False):
    """Run the registered sampler. A checkpoint cannot be rebound to other data."""
    from .sampling import SamplerConfig, sample_regression
    if not isinstance(prepared, PreparedModel):
        raise TypeError("fit requires a PreparedModel")
    config = SamplerConfig() if config is None else config
    temporal = "full_residual_ar" if prepared.spec.innovations == "full_residual_ar" else "iid"
    config = replace(config, temporal=temporal)
    if config.parameterization != "legacy" and prepared.spec.innovations != "iid":
        raise ValueError("ridge_total_scale is supported only for the certified iid Gaussian hierarchy")
    checkpoint = {"identity": prepared.identity, "config": asdict(config), "chains": {}}
    if resume:
        if checkpoint_path is None:
            raise ValueError("resume=True requires checkpoint_path")
        checkpoint = load_state(checkpoint_path, kind="mcmc_checkpoint")
        old_config, new_config = dict(checkpoint["config"]), asdict(config)
        old_draws, new_draws = old_config.pop("draws"), new_config.pop("draws")
        old_config.pop("checkpoint_every", None)
        new_config.pop("checkpoint_every", None)
        # Missing means the original v0.2 transition.  Keep old checkpoints
        # readable/resumable without permitting a different proposal policy.
        old_config.setdefault("parameterization", "legacy")
        new_config.setdefault("parameterization", "legacy")
        if (checkpoint["identity"] != prepared.identity or old_config != new_config
                or new_draws < old_draws):
            raise ValueError("Checkpoint model/data/config identity mismatch")
        checkpoint["config"] = asdict(config)

    def callback(chain, state):
        checkpoint["chains"][str(chain)] = state
        save_state(checkpoint_path, checkpoint, kind="mcmc_checkpoint")

    args = dict(Y=prepared.Y, Phi=prepared.Phi, Z=prepared.fit_Z,
                coordinates=prepared.data.coordinates, groups=prepared.groups,
                prior=prepared.spec.prior, covariance=prepared.spec.covariance,
                config=config, reference_Z=prepared.reference_Z)
    started = time.perf_counter()
    if prepared.spec.innovations in {"iid", "full_residual_ar"}:
        posterior = sample_regression(**args,
                      checkpoint_callback=callback if checkpoint_path is not None else None,
                      resume_state={int(k): v for k, v in checkpoint["chains"].items()} if resume else None)
    else:
        from .dynamics import sample_augmented
        args["kind"] = prepared.spec.innovations
        if prepared.variance_covariates is not None:
            args["known_variance_covariates"] = np.swapaxes(
                prepared.variance_covariates[prepared.split.fitting_start:], 0, 1)
        posterior = sample_augmented(**args,
                      checkpoint_callback=callback if checkpoint_path is not None else None,
                      resume_state={int(k): v for k, v in checkpoint["chains"].items()} if resume else None)
    metadata = {"fit_seconds": time.perf_counter()-started, "environment": environment(),
                "prepared_identity": prepared.identity, "training_end": prepared.training_end,
                "response_units": prepared.data.response_units,
                "posterior_policy": "parameters frozen after fitting",
                "sampler_metadata": posterior.metadata}
    # Wall-clock timings are provenance, not posterior identity. Numerically
    # identical runs must identify the same parameter distribution artifact.
    identity = content_hash({"prepared": prepared.identity, "config": asdict(config),
                             "states": posterior.states, "log_density": posterior.log_density,
                             "acceptance": posterior.acceptance})
    return FittedModel(prepared, posterior, config, metadata, identity)


def refit(fitted, expanded_data, split, config=None, **kwargs):
    """Explicitly replace the parameter posterior using an expanded fit period."""
    if split.fitting_stop <= fitted.training_end+1:
        raise ValueError("A refit must extend the parameter training endpoint")
    if expanded_data.site_ids != fitted.prepared.data.site_ids or not np.array_equal(
            expanded_data.coordinates, fitted.prepared.data.coordinates):
        raise ValueError("Refit requires the same input-site panel")
    variance_covariates = kwargs.pop("variance_covariates", None)
    return fit(prepare(expanded_data, split, fitted.prepared.spec,
                       variance_covariates=variance_covariates), config, **kwargs)

