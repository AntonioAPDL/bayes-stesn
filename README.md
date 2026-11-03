# bayes-stesn

Bayesian spatiotemporal echo state networks with explicit probabilistic models,
causal feature construction, posterior predictive trajectories, and reproducible
numerical checks. Python 3.11+, NumPy and SciPy; no GPU or deep-learning framework
is required.

The current implementation is a research release. The scalar/group/column
Gaussian routes use exact spectral calculations. Entrywise shrinkage and richer
latent/robust models use documented conditional samplers, with explicit numerical
and resource limits. Successful unit tests are distinct from passing an MCMC
diagnostic gate or demonstrating improved forecasting performance.

## Release and reports

[Version 0.2.1](https://github.com/AntonioAPDL/bayes-stesn/releases/tag/v0.2.1)
includes the checkpoint-continuation correction verified by the
[passing public CI run](https://github.com/AntonioAPDL/bayes-stesn/actions/runs/41580781455).
The original three-commit history and `v0.2.0` scientific snapshot are preserved.

| Download | Contents |
| --- | --- |
| [Benchmark report (PDF)](https://github.com/AntonioAPDL/bayes-stesn/releases/download/v0.2.1/bayes_stesn_v02_benchmark_report.pdf) | 27-page replicated development comparison |
| [Detailed forecast report (PDF)](https://github.com/AntonioAPDL/bayes-stesn/releases/download/v0.2.1/bayes_stesn_v02_detailed_forecast_report.pdf) | 10-page preselected nonlinear forecasting illustration |
| [Forecasts, truth, and errors (MP4)](https://github.com/AntonioAPDL/bayes-stesn/releases/download/v0.2.1/bayes_stesn_v02_forecast_truth_errors.mp4) | Synchronized ESN, GP, and GRU forecasts, signed errors, and RMSE evolution |
| [Simulated data evolution (MP4)](https://github.com/AntonioAPDL/bayes-stesn/releases/download/v0.2.1/bayes_stesn_v02_data_evolution.mp4) | The nonlinear field evolving through time |

The release also contains both complete-result archive parts, the original Git
bundle, and checksums. These numerical reports retain their original `v0.2.0`
source identity; the patch adds no new simulation results. See
[publication and validation evidence](docs/publication.md) for exact commits and
the distinction between software checks and scientific fit qualification.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[report,dev]'
python -m pytest -q tests
```

## API

```python
from bayes_stesn import (
    SpatioTemporalData, SplitPlan, ModelSpec, FeatureSpec, SpatialMeanSpec,
    PriorSpec, CovarianceSpec, SamplerConfig, prepare, fit, diagnose,
    forecast_context, simulate_predictive,
)

# y: [time, input_site], coordinates: [input_site, 2].
data = SpatioTemporalData(y, coordinates, response_units="measurement")
spec = ModelSpec(
    features=FeatureSpec(modes=10, units=(32, 32), lags=(1, 2, 4), squares=16),
    spatial_mean=SpatialMeanSpec(side=7, width=0.18),
    prior=PriorSpec.ridge(), covariance=CovarianceSpec(),
)
prepared = prepare(data, SplitPlan(conditioning=200, fitting_stop=len(y)), spec)
fitted = fit(prepared, SamplerConfig(chains=4, warmup=1000, draws=2000),
             checkpoint_path="checkpoints/model.npz")
checks = diagnose(fitted, functional_draws=512)
if not checks["passed"]:
    raise RuntimeError("Inspect diagnostics before interpreting this fit.")
prediction = simulate_predictive(
    fitted, forecast_context(fitted), horizon=40,
    target_coordinates=prediction_locations, paths=256, seed=17,
)
fitted.save("model.npz")
prediction.save("prediction.npz")
```

`prediction.observation` has dimensions `[path, lead, target_site]` in the original
response units. A parameter draw stays fixed throughout a trajectory; its own
simulated observations drive its future reservoir inputs. Readout means, latent
fields and conditional observation means are labeled separately.

## Version 0.2: model comparison framework

The package now includes shared G1-G6 simulators, eight core observed-site
comparators, native new-site comparators, and an optional probabilistic
coordinate-basis GRU control. Chronological/spatial validation, independent
final seeds, explicit failed-fit fallbacks, block calibration, matched learning
curves, process-parallel execution, and saved-result reports use one API.

```bash
# PyTorch is optional; install it for dcrnn and basis_gru.
python -m pip install -e '.[neural,report,dev]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 bstesn benchmark run \
  --profile smoke --output results/benchmark_smoke
bstesn benchmark report results/benchmark_smoke --output report/benchmark_smoke
```

The smoke profile verifies execution; its deliberately short MCMC/optimization
budgets do not qualify it as scientific evidence. Development and confirmation
profiles are separate, and confirmation requires a frozen design/power receipt.
The completed development pilot contains **300 model/task attempts across 20
independent datasets**: 251 qualified fits and 49 declared point fallbacks,
with no execution or scoring failures. Nineteen ESN fits failed MCMC diagnostics;
30 comparator fits reached their optimizer limits. The confirmation gate remains
blocked because ten primary comparisons lack three common qualified pilot
replications. These are development findings, not a completed confirmatory study.
See [the benchmark guide](docs/benchmark_guide.md),
[model equations](docs/benchmark_models.md), and
[the version 0.2 validation report](docs/benchmark_validation.md).

![Saved-data forecast and error comparison](docs/assets/v02_comparison.png)

The image shows a preselected 100-step forecast at 505 never-observed sites.
All three native fits qualified in this separate one-dataset illustration;
the replicated pilot is a distinct comparison. See the
[saved numeric summary](docs/evidence/v02/detail_summary.json).

Memory-controlled predictive batches, bounded spatial-pair selection, atomic
failure cleanup, and the narrow end-to-end latent-Vecchia route address the
highest-impact engineering issues from the prior audit. Public data and
forecast constructors retain defensive-copy semantics.

## Documentation

- `docs/api.md`: objects, workflows, serialization and supported combinations.
- `docs/models.md`: hierarchical models and implementation decisions.
- `docs/validation.md`: mathematical, statistical and integration verification.
- `docs/reproduction.md`: isolated installation, release checks and study commands.
- `docs/implementation_registry.yaml`: all 23 extension IDs and their limits.
- `docs/publication.md`: current public-repository and CI status.
- `docs/derivations/`: the complete mathematical blueprint and MCMC derivations.
- `configs/`: explicit reproducible experiment configurations.
- `benchmarks/`: resource pilots and controlled comparisons.
- `report/`: report-generation source; rendering never refits a model.

The primary controlled study compares scalar ridge with more data, larger
reservoirs and grouped Gaussian shrinkage. Two horseshoe development pilots
failed strict mixing checks and remain documented; those failures are not
presented as validated forecasting comparisons. Every implemented prior has
numerical and complete-workflow tests, which are a distinct claim.

## Earlier controlled study (version 0.1)

The locked study completed **48 four-chain fits**: four candidates, two spatial
innovation scenarios and six independent simulation replications. **47/48 fits
passed** both the hyperparameter and readout-functional gates (R-hat <= 1.01;
bulk/tail ESS >= 400). One grouped-Gaussian fit failed and remains visible.

| Change | Isotropic scenario: input CRPS difference | Anisotropic scenario: input CRPS difference |
|---|---:|---:|
| More likelihood history | -0.00266 [-0.00389, -0.00158] | -0.00199 [-0.00303, -0.00080] |
| Larger reservoirs, same long history | -0.01292 [-0.01869, -0.00399] | -0.01405 [-0.02074, -0.00626] |
| Group scales instead of scalar ridge | +0.00057 [-0.00001, +0.00102] | -0.00061 [-0.00139, -0.00002] |
| Historical validation-selection policy vs recent-history ridge | -0.01256 [-0.01969, -0.00451] | -0.01614 [-0.02188, -0.00876] |

Negative differences favor the change. Brackets are paired 95% bootstrap
intervals across independent replications, conditional on both compared fits
passing diagnostics. The grouped-prior isotropic comparison has five pairs;
the others have six. The validation-selected policy reduced mean input CRPS
by **13.7% and 17.8%** in these two scenarios, respectively.

Larger reservoirs delivered the clearest gain. More history alone slightly
worsened withheld-site accuracy, despite its input-site gains. Group-specific
shrinkage did not consistently improve on expressive ridge; both horseshoe
pilots failed convergence. These results favor expressive ridge as the first
working baseline for this DGP, while leaving more elaborate priors available
for tasks that justify them.

![Version 0.1 forecast, observations and errors](docs/assets/v01_forecast_montage.png)

The study uses 49 observed inputs on an 11-by-11 grid, 740 versus 1,540 likelihood
times, a fixed conditioning history, and two 40-step final forecast origins.
The animation is a separately labeled, preselected uninterrupted 80-step
forecast. Six replications give limited precision; this is not a new full-size
comparison against the earlier V3 forecast. Full records and media are included
in the release download. [The archived compact summary](docs/evidence/v01/compact_summary.json) and
[protocol](docs/evidence/v01/protocol.json) retain the numerical evidence used above.

The version 0.1 release gate passed **149 source tests and the same 149 tests against the
installed wheel**, with no failures, errors, skips or warnings, in an isolated
Python 3.12 environment. Full-size fixed-parameter V3 parity covered 113,004
coefficients; the coefficient-mean relative discrepancy was 2.48e-12.
Conditional Gaussian SBC and sparse-covariance fidelity exercises are recorded
separately. See the release manifest and validation document for their scope.

## Scope

One response, a complete fixed panel of input sites, equally spaced times and
Euclidean coordinates are supported. Observations at additional locations may be
forecast jointly. Missing inputs, irregular-time dynamics, vector-valued responses,
unprojected geographic degrees, and unsupported combinations fail explicitly.

The iid Gaussian scalar-ridge Vecchia route now supports end-to-end operator
inference and ancestral prediction without dense site covariance matrices.
Sparse LU fill remains geometry-dependent. Other Vecchia combinations and
FEM/SPDE routes remain explicitly bounded reference implementations. No
million-site performance claim is made. The FEM implementation uses integer
operator orders with the stated Matérn smoothness.

## Reproducibility

The statistical core uses float64; optional PyTorch networks train on CPU in
float32. Artifacts are atomic, checksum-verified numeric NPZ files
with JSON metadata and `allow_pickle=False`. Named random streams do not depend on
process completion order. Seeds, software versions, numerical approximations and
forecast information policies are recorded. Same-environment reproducibility is
tested; bitwise equality across BLAS implementations is not promised.

## Attribution

The PCA reservoir core preserves the equations and random streams of the earlier
V3 project, with a numerical compatibility fixture. Established mathematical
constructions are cited in the derivation chapters. The ESN-specific reference-R²
and subspace constructions are adaptations, without a claim of new theory.

`ModelSpec.v3()` reproduces the reference architecture. Exact full-size parity
reuses the original standardized observations and frozen transformations. By
default, a newly prepared data set estimates response center/scale from all
supplied input sites during conditioning; the historical V3 experiment used
its preserved common-site normalization. These preprocessing policies should
not be silently equated when comparing raw-data fits.

Original project code and documentation: MIT license. Referenced papers and data
retain their respective licenses.

