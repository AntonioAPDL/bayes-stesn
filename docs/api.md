# API and reproducible workflows

## Data, axes and information boundaries

`SpatioTemporalData(y, coordinates, times=None, site_ids=None, ...)` owns a
read-only float64 copy of its inputs. `y` has axes **time × input site**;
`coordinates` has axes **input site × spatial dimension**. Numeric times or
`datetime64` are accepted, with strict ordering and equal spacing. One response
channel and complete input observations are required. Project geographic degrees
before fitting. Coordinate units determine the meaning of every covariance and
spatial-basis range parameter.

`EvaluationTruth(observation, generating_mean=None, latent=None)` is a separate,
scoring-only object. It is never accepted by `prepare`, `fit`, or forecasting.
Forecast arrays have axes **path × lead × target site** and return to the original
response units.

All split boundaries are **zero-based row indices with an exclusive stop**:

```python
split = SplitPlan(
    conditioning=160,
    fitting_start=160,
    fitting_stop=1700,
    validation_blocks=((1700, 1740),),
    calibration_blocks=((1740, 1780), (1780, 1820)),
    final_blocks=((1820, 1860),),
)
```

Only rows `[0,160)` fit centering, scale, PCA/encoder, layer normalization and
readout normalization. The likelihood uses rows `[160,1700)`. Validation,
calibration and final blocks are disjoint. `prepare` retains only rows before the
fitting endpoint, even if a longer observed array was supplied. The model identity
is unaffected by changes to later values. Spatial holdouts must be excluded from
the input panel itself, not merely from a scoring index.

## Define and fit a model

```python
from bayes_stesn import *

spec = ModelSpec(
    features=FeatureSpec(modes=10, units=(32, 32), lags=(1, 2, 4), squares=16),
    spatial_mean=SpatialMeanSpec(side=7, width=.18, bounds=((0, 1), (0, 1))),
    prior=PriorSpec.group_rhs(tau0=.1, slab_scale=2., slab_df=4.),
    covariance=CovarianceSpec(anisotropy=True),
    innovations="iid",
    name="group_rhs_anisotropic",
)
prepared = prepare(data, split, spec)
config = SamplerConfig(chains=4, warmup=1000, draws=2000, seed=17)
fitted = fit(prepared, config, checkpoint_path="results/checkpoint.npz")
```

The PCA reservoir input can be replaced by `local_features_esn`,
`graph_features_esn`, or `frozen_encoder_esn`. The last option trains a small tanh
autoencoder with a linear decoder exclusively on the conditioning data, then
freezes it before Bayesian readout inference. Its weights are not sampled; its
representation uncertainty is not included in the conditional readout posterior.
An ensemble of frozen maps is a separate, explicit uncertainty choice.

Feature group names are `intercept`, `score`, `reservoir_1`, `reservoir_2`,
`squares` and `products` when those terms exist. The correction prior labels the
frozen baseline and residualized part as `base` and `correction`. Input scales
default to `.4` in the first layer and `.8` thereafter; supplying a scalar repeats
that scale across layers. Explicit per-layer tuples must match the layer count.

An empty recurrent graph in a tiny sparse reservoir remains a valid zero
recurrent matrix. Nonempty draws are normalized to the stated operator norm.
This avoids undefined division without perturbing the established V3 streams.

`ModelSpec.v3()` reproduces the original architecture. `configs/baseline_v3.yaml`
contains its explicit dimensions and scale conventions. The range of the spatial
mean dictionary is independent of the innovation covariance range.

## Prior and covariance choices

| Requirement | API |
|---|---|
| Scalar Gaussian shrinkage | `PriorSpec.ridge()` |
| Group Gaussian shrinkage | `PriorSpec.grouped()` |
| Regularized horseshoe | `group_rhs()`, `column_rhs()`, `entry_rhs()` |
| Group inverse-gamma/gamma shrinkage | `PriorSpec.gigg()` |
| Frozen-reference R² budget | `PriorSpec.reference_r2(allocation="dirichlet")` or `"logistic_normal"` |
| Proper baseline + residual correction | `PriorSpec.correction(base_columns=...)` |
| Proper Gaussian function penalty | `PriorSpec.function_penalty(...)` |
| Global anisotropy | `CovarianceSpec(anisotropy=True)` |
| Static nugget variance | `CovarianceSpec(static_variance=True)` |
| Finite smoothness mixture | `CovarianceSpec(infer_nu=True)` |
| Nonstationary valid kernel | `CovarianceSpec(nonstationary=True)` |
| PC range prior | `CovarianceSpec(range_prior="pc")` |
| Separate physical PC scales | add `scale_mode="separate"` |
| Vecchia covariance approximation | `CovarianceSpec(kind="vecchia", vecchia_neighbors=15)` |
| Integer-order FEM/SPDE | `CovarianceSpec(kind="spde", spde_order=2)` |

See [models.md](models.md) and [implementation_registry.yaml](implementation_registry.yaml)
for the full hierarchies, defaults and support boundaries. Priors are on the
standardized response scale; inspect prior predictive implications when changing
feature normalization, dimensionality or measurement units. The default RHS slab
is a choice to assess, not a universally optimal setting.

## Temporal and robust observation models

`ModelSpec.innovations` accepts `iid`, `full_residual_ar`, `latent_ar`, `field_t`,
`local_t` or `heteroskedastic`.

The full residual AR and latent-process AR are different likelihoods. The first
correlates the complete residual, including its nugget component; the second
adds serially independent measurement noise to a dynamic spatial process.
The whole-field Student model shares one scale mixture across an entire time
slice, while the local Student model puts separate scales on measurement errors.

For predictable site-time variance, supply
`prepare(..., variance_covariates=F)` with shape `[time,input_site,covariate]`.
The fitting segment determines fixed centers/scales for these variance
covariates. Constant columns are rejected; the baseline variance is already a
separate model component. At forecast time, supply
`future_variance_covariates[lead,joint_site,covariate]`. The joint site order is
returned by `joint_domain(input_coordinates, target_coordinates)`.

The caller must ensure those future covariates are known at the origin. A future
response cannot be used as a variance predictor merely because its array fits
the required shape. The public CLI's simple `forecast` command handles models
without these extra predictors; use the Python API for this explicit case.

## Forecasts, new origins and refitting

```python
context = forecast_context(fitted)
prediction = simulate_predictive(
    fitted, context, horizon=80, target_coordinates=new_locations,
    target_site_ids=new_ids, paths=256, seed=23,
)
```

The initial context is at `fitted.training_end`, a row index. Parameters stay
fixed throughout each path. Generated observations at the input sites drive the
next reservoir input; truth is never fed in during an autonomous forecast.
Input and target sites share a joint spatial draw. Exact duplicate coordinates
are deduplicated, and their observation paths are copied. An input ID associated
with inconsistent target coordinates is rejected.

The prediction exposes four explicitly different targets:

| Attribute | Meaning |
|---|---|
| `observation` | Spatial latent field plus measurement noise |
| `latent` | Readout mean plus spatial process |
| `readout_mean` | `Phi C z` on each stochastic feedback trajectory |
| `conditional_mean` | Next observation mean conditional on that path's residual state |

`prediction.mean` averages observation paths. `prediction.interval(.9)` returns
pointwise 90% intervals. They are not simultaneous path bands. Readout draws are
not automatically draws of a simulator's deterministic mean under a misspecified
model; generating-mean scores are separately labeled reconstruction diagnostics.

To advance with newly available observations:

```python
later = advance_state(fitted, context, new_observations)
later_prediction = simulate_predictive(fitted, later, horizon=40, paths=256, seed=24)
```

Here `new_observations` contains **only new rows**, with matching site IDs and
consecutive times. Alternatively,
`forecast_context(fitted, origin=..., history=full_observed_prefix)` replays the
same frozen feature map. The earlier fitted prefix must match exactly.

For a latent AR process, each chosen parameter draw is filtered using the entire
likelihood period plus new observations, beginning at the original likelihood
initial state. A sampled old terminal latent state is not incorrectly treated
as known. The parameter posterior remains the training-period posterior;
updating states alone does not perform a full Bayesian parameter update.

Use `refit(fitted, expanded_data, expanded_split, config)` when the parameter
posterior should change. It creates a new model/posterior identity.

## Diagnostics and continuation

```python
summary = diagnose(fitted, functional_draws=512)
if not summary["passed"]:
    # Inspect chain agreement and MCSE before choosing an appropriate response.
    from dataclasses import replace
    extended = fit(prepared, replace(config, draws=4000),
                   checkpoint_path="results/checkpoint.npz", resume=True)
```

`fitted.diagnose()` checks scalar hyperparameter chains. `diagnose(fitted)` adds
joint coefficient/scale recovery at prespecified sites and the spatial average,
using the final fitted feature vector. The default gate is at least four chains,
rank/folded split R-hat below 1.01, and bulk/tail ESS at least 400. The result
includes MCSE. Passing a gate is evidence about these summaries, not a guarantee
of complete posterior exploration or model correctness.

Warmup alone adapts proposal geometry. Retained transitions keep the resulting
proposal fixed. For small collapsed hyperparameter blocks, automatic local
MAP/Laplace initialization is a bounded computation used to improve geometry;
it does not truncate the prior or change the target. `initialize="prior"`
reproduces prior-centered initialization. RHS global/local tradeoff moves preserve
coefficient variances exactly and use a proper hyperprior MH ratio.

Checkpoints store parameter state, any required coefficient/latent augmentation,
random-number generator state, proposal adaptation and retained draws. The
high-level API verifies model, data and transition settings before resumption.
Increasing `draws` and changing checkpoint cadence are allowed; reducing the
target or changing substantive sampler settings is rejected. The same-runtime
continuation tests use actual JSON/NPZ round trips.

## Ensembles and finite Bayesian indices

`fit_ensemble(prepared_models, config)` fits aligned components and begins with
equal weights. `simulate_predictive(ensemble, contexts, ...)` chooses one complete
component trajectory per path and retains component provenance. It never changes
components halfway through a trajectory.

`fit_stacking_weights(ensemble, validation_predictions, observed_validation)`
uses empirical marginal CRPS, including between-component dispersion. Every
prediction must lie within a declared validation block. `method="log_score"`
requires normalized, integrated joint block log densities from the caller; it
does not multiply one-step marginal densities and call that a path score.

`fit_reservoir_index(prepared_models, config)` is a separate Bayesian model-index
analysis. The connected initial implementation supports shared ridge/covariance
hyperpriors, iid Gaussian residuals and fixed smoothness. The normalized collapsed
likelihood includes dimension-dependent coefficient determinants. Its mixture
weights and hyperparameters come from their joint chain. Zero-visit models have
zero empirical mass, with finite-chain uncertainty explicitly recorded.
The advanced `ensemble.sample_model_index` callback API supports other verified
common-dimensional hyperparameter spaces. Neither route implements reversible
jump MCMC or separate marginal-likelihood estimation.

## Calibration and scoring

`fit_calibration(fitted, predictions, observed_blocks, spec=CalibrationSpec(...))`
uses only declared, nonoverlapping calibration blocks. Choose `target="marginal"`
for separate lead/site intervals, `"horizon"` for simultaneous spatial fields at
each lead, or `"simultaneous"` for the full lead/site domain. The number of blocks
remains the sample size, regardless of the number of sites or leads.

`apply_calibration(calibration, prediction)` returns intervals without changing
the raw posterior sample. Domain and posterior identities must match. Too few
blocks produce an infinite conformal quantile when required by the exact rank;
the code does not cap that rank. Dependent time series require empirical
assessment and appropriate assumptions. The optional reservoir-weighted
calibration is an explicitly described empirical procedure, not an assertion
that a particular published method or its guarantee has been reproduced.

`score(prediction, EvaluationTruth(...), target="observation")` labels the target
and origin. `score_predictions` provides the lower-level array interface. It
returns RMSE, MAE, exact empirical CRPS, interval width/coverage, spatial-field
energy and variogram scores, with per-lead and pooled summaries kept separate.
Use `energy_pairs=None` for exact empirical energy scores; the default uses 2,048
sampled pairs and records this approximation. Replication-level paired bootstrap
is provided separately; dependent site/time errors are not independent replicates.

## Serialization, resources and command line

`FittedModel.save/load`, `FittedEnsemble.save/load`, and `PredictiveSample.save/load`
use atomic numeric NPZ plus JSON metadata, schema versions and SHA256 validation.
They load with `allow_pickle=False`. Timings are provenance, not posterior
identity. Every named forecast random stream is independent of worker completion
order. Exact same-runtime forecast reproduction and path-prefix reproduction
are tested. Cross-platform bitwise equality is not promised.

Feature history defaults to a 512 MiB allocation gate; spatial mean dictionaries
default to at most 4,096 columns. Matrix-free coefficient samplers report solver
residuals and finite-tolerance approximation status. Sparse spatial operators
are available separately; current general fitting adapters materialize bounded
spatial matrices. Consult the registry before increasing dimensions.

```bash
bstesn demo --smoke --output results/smoke
bstesn demo --config configs/demo.yaml --output results/demo
bstesn validate-config configs/baseline_v3.yaml
bstesn fit observations.npz --config model.yaml --output model.npz --checkpoint checkpoint.npz
bstesn forecast model.npz --horizon 40 --paths 256 --output forecast.npz
bstesn diagnose model.npz --functional-draws 512 --output diagnostics.json
```

The smoke demonstration intentionally uses short chains and reports a failed
scientific diagnostic gate. Its purpose is installation/execution verification.
The controlled study has its own protocol-locked runner, independent pilot
streams and documented resource budget. Reporting reads saved results and never
refits or selects a model.

