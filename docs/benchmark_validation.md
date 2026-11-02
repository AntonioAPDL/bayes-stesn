# Version 0.2 validation and evidence

This release separates four claims: correctness of the implementation, numerical
and resource behavior, predictive performance on independent simulated datasets,
and remote publication. Passing one does not establish the others.

The authoritative software receipt is `results/release_manifest.json` in the
release archive. The benchmark receipts and reports identify the exact source
fingerprint, resolved configuration, numerical runtime, random streams, and
qualification status used for each result. A Git bundle establishes a portable
local repository; it does not establish that a remote push or CI run succeeded.

## 1. Mathematical and implementation checks

The final source suite contains **210 tests and 70 subtests**, reported as
**280 JUnit cases**. The source verification completed with no failures, errors,
or skips. The release verifier independently reruns the suite against the built
wheel from outside the source checkout, checks every imported package-module
origin, and rejects unexpected skips, warnings, dependency drift, or source
changes during validation. Consult its receipt for the actual installed-wheel
outcome; the test count alone is not a release qualification.

| Area | What is checked | Risk addressed |
|---|---|---|
| Existing Bayesian model | Dense/spectral likelihood and conditional-Gaussian parity; prior, covariance, robust and latent-model reductions | A fast implementation sampling a different law |
| Sparse Vecchia workflow | Tiny dense-reference likelihood and posterior parity; forecast moments; extreme variance ratios; dense-route guards | Incorrect nugget treatment, cancellation, or hidden dense allocation |
| Forecast batching | Global-path identity, retained-component checks, saved-model continuation | Batch size or scheduling changing a seeded forecast |
| Persistence and kernel models | Known reference forecasts, exact/Nyström kernel calculations, strict configuration schemas | Incorrect lag alignment, regularization scaling, or ignored options |
| LGSSM | Kalman/RTS reference calculations, full observation likelihood, monotone constrained EM, joint forecasts | Omitted-space likelihood terms or an invalid state update |
| Separable GP | Dense covariance/conditional predictions against temporal state-space calculations | Temporal scaling or new-site cross-covariance errors |
| Reservoir and neural controls | Free rollout, training-only transforms, deterministic CPU fits, serialization, query consistency | Teacher forcing or target leakage during evaluation |
| Benchmark policy | Final-time and new-site response perturbations leave selection unchanged; common roles/endpoints; serial/spawn equality | Leakage, incomparable information, or schedule-dependent data |
| Randomized kernel tuning | Fresh independent seeds for finalist bandwidth/landmark choices; selection uses their average validation score | Treating a randomized approximation as a deterministic fit |
| Calibration | Finite-sample ranks, unavailable high-coverage ranks, frozen parameters, disjoint C/U roles | Using final labels to tune a fit or claiming an invented joint law |
| Failure handling | Failed candidates and final fits remain counted; operational fallbacks are explicit | Selectively deleting difficult or failed replications |
| Artifact writes | Fault injection through compression, fsync, and atomic replacement; ownership and checksums | Partial files replacing valid results or accidental mutation |

The suite includes independent numerical references and invariant checks.
Reproducing the same algebra in a second function is not treated as sufficient
validation of an inference method. Smoke profiles test execution only; their
short chains cannot meet the scientific MCMC gate.

## 2. Reproducible release environment

Core development supports Python 3.11 and later. The complete CPU-neural release
gate uses an isolated **CPython 3.12** environment, exact dependency locks,
NumPy/SciPy float64 statistical calculations, and explicit CPU float32 PyTorch
training. No system or user site packages may enter the isolated environment.
The PyTorch installation must come from the official CPU wheel index and have
no CUDA build. The wheel and source distribution are checked before testing.

```bash
python tools/release_check.py --with-neural \
  --fresh-venv /tmp/bstesn-release
```

The core-only gate uses an environment without PyTorch and explicitly deselects
only the optional Torch test classes. That narrower gate does not certify the
neural models. The complete gate includes them. CI configuration is committed;
remote CI is only verified when an actual remote run is observed.

## 3. Measured engineering improvements

### Exact checkpoint policy

Repeatedly rewriting all retained MCMC states into a temporary archive dominated
one representative benchmark-size fit. Benchmark fits now default to writing at
completed-chain endpoints; an explicit `sampler.checkpoint_every` retains an
intermediate cadence when recovery granularity matters. The original standalone
sampler default is unchanged.

With the same data, seeds, four chains, warmup and retained-draw continuation,
the measured fit fell from **33.64 s to 9.56 s**, a **3.52-fold wall-time ratio**.
Checkpoint writes fell from 96 to 8. Posterior chains, densities, acceptances,
recovered coefficients/variances, forecasts, diagnostic gates, and continuation
states were equal in the recorded comparison. Both fits qualified. This is one
measured workload, not a universal speedup guarantee. An incomplete chain loses
its intermediate recovery point under the new default.

Evidence: [checkpoint_cadence.json](evidence/v02/checkpoint_cadence.json).
Reproduce with `benchmarks/profile_checkpoint_cadence.py`.

### End-to-end sparse spatial execution

The supported operator route is collapsed scalar-ridge inference with iid
Gaussian innovations and a latent Vecchia covariance. It uses sparse geometry,
operator sufficient statistics, nugget-updated solves, and ancestral prediction.
It does not silently substitute an observation-Vecchia approximation. The tests
forbid dense spatial materialization on this route.

| Sites | Fit time (s) | Prediction time (s) | Process peak RSS (MiB) |
|---:|---:|---:|---:|
| 128 | 0.14 | 0.14 | 90.1 |
| 512 | 0.39 | 0.67 | 91.5 |
| 2,048 | 2.79 | 3.76 | 125.2 |
| 4,096 | 5.44 | 6.07 | 184.0 |

These are deliberately small **execution tests**: 50 time rows, 35 likelihood
rows, 8 predecessors, small readout dimensions, one chain with 2 warmup and
3 retained iterations, and 4 forecast paths over 8 steps. They establish
execution and resource behavior; they do not establish convergence or forecast
quality. Sparse factor fill-in and dense readout/basis Gram matrices still
limit scale. The current sparse route does not certify every prior or dynamic
extension as equally scalable.

Evidence: [sparse_profile.json](evidence/v02/sparse_profile.json) and
[sparse_4096_profile.json](evidence/v02/sparse_4096_profile.json).
Reproduce with `benchmarks/profile_sparse_workflow.py`.

## 4. Shared predictive study

`configs/benchmark/pilot.yaml` declares the replicated development comparison:
G1 Gaussian spatial autoregression, G2 spectral transport, G3 an excitable field,
and G4 nonlinear moving fronts. Each has five independent datasets shared by
all eligible models. The 7-by-7 grid has 25 observed O sites, 4 reserved
calibration C sites, and 20 never-observed U sites.

The first fit uses 400 historical rows. Three 40-step validation blocks precede
the final fit through row 519 and the fixed final forecast `[520:560)`. The
profile permits three candidate settings, two finalist candidates with two fresh
seeds where the fitting procedure is randomized, and 128 predictive paths.
It includes nine eligible A models and six B models, for **300 final model/task
attempts** across 20 shared simulated datasets. Candidate fits are additional
and their costs are retained.

Native Bayesian forecasts require four chains, rank-normalized R-hat below
1.01, bulk and tail ESS of at least 400, and recovered-readout functional
checks. One predeclared continuation is allowed; thresholds never relax after
a failed fit. Point-error tables retain every attempt through a declared
persistence fallback. Native probability tables contain qualified distributions
only and report their population. Paired probability comparisons use the common
qualified subset, which is a conditional comparison.

Full results are stored in `results/benchmark_v3/pilot_release`; its summary
and report provide actual completed/qualified/fallback counts and paired MCSE.
The release packager requires a complete, source-stable run, matching per-job
receipts, and a report bound to the same manifest. Interrupted engineering runs
are excluded from the selected release evidence.

This is a development study with limited simulation precision. The confirmation
factory requires at least three common qualified development replications for
each prespecified primary comparison, binds a fresh resolved main profile to
its power/design receipt, and plans 20–60 independent datasets. A failed freeze
remains a failed gate. No large confirmatory result is implied by implementing
that workflow or generating a main draft configuration.

## 5. Higher-resolution illustration and media

`configs/benchmark/visual_demo.yaml` fixes a separate, preselected G4 replication:
a 25-by-25 grid with O=100, C=20, U=505, 2,000 initial historical rows, three
100-step validation blocks, and a 100-step final forecast from row 2,300.
Its ESN has two 64-unit reservoirs, 12 input modes, lags 1/2/4, 32 quadratic
features, and a richer coordinate basis. Separable GP and basis-GRU comparators
receive the same information. This illustration is not pooled with the
replicated pilot and does not establish a general model ranking.

All seven final model/task fits qualified in this saved illustration. Its
point and probability results are:

| Model | A RMSE | B RMSE | A CRPS | B CRPS | B 90% coverage |
|---|---:|---:|---:|---:|---:|
| Bayesian ESN | 0.2020 | 0.2315 | 0.1088 | 0.1237 | 91.21% |
| Separable GP | 0.4725 | 0.4943 | 0.2764 | 0.2886 | 64.18% |
| Probabilistic basis GRU | 0.3032 | 0.2936 | 0.1697 | 0.1675 | 96.16% |
| Spatial persistence | — | 0.6155 | — | — | — |

The ESN performed best among these controls on this particular preselected
dataset. These values are descriptive, with no replication-based uncertainty
interval. Evidence: [detail_summary.json](evidence/v02/detail_summary.json).

`report/benchmark_detail` contains the saved-data report and synchronized media.
The standalone comparison renderer shows truth, three forecast means, three
signed-error fields, and per-lead RMSE on the exact same target cells. It checks
saved scores and hashes its inputs/outputs. Color scales remain fixed across
models and time; unscored cells are gray. Native-fit qualification labels remain
visible, including when a forecast is only a diagnostic illustration.

```bash
python tools/render_model_comparison.py results/benchmark_v3/detail_release \
  --scenario g4_existing_fronts --output report/benchmark_detail/comparison
bstesn benchmark report results/benchmark_v3/detail_release \
  --output report/benchmark_detail
python tools/embed_report_fonts.py report/benchmark_detail
```

The report is rendered from saved numerical artifacts. It does not refit models,
choose favorable random seeds, interpolate missing target truth, or assimilate
future observations during a forecast horizon. MP4/GIF trajectories, PNG/SVG
figures, HTML, PDF, and their hash inventory are retained in the release archive.

## 6. Calibration example

The preselected G1 replication 0 GP was calibrated separately with its parameters
frozen at row 520. Twenty-five blocks cover `[520:1520)`; the later evaluation
block is `[1520:1560)`. For B, labels from the four C sites construct bands
evaluated at the disjoint 20 U sites. Model and package-source hashes remain
unchanged; the extended DGP exactly preserves the original data prefix.

| Quantity | A raw | A calibrated | B raw | B calibrated |
|---|---:|---:|---:|---:|
| 90% marginal coverage | 87.3% | 91.4% | 84.5% | 88.5% |
| Mean interval width | 2.496 | 2.927 | 2.872 | 3.333 |
| Marginal WIS | 0.31567 | 0.32215 | 0.41936 | 0.41636 |

Coverage rose through wider intervals; WIS improved only in B. One replication
and one later test block cannot establish uniform improvement. Serial blocks
are not assumed exchangeable, and C-to-U transfer remains empirical. Calibration
changes bands, never the native joint draws. Verified ranks for 50/80/90/95%
levels with 25 blocks are 13/21/24/25; a requested 99% rank would be unavailable
and yield an infinite band rather than a fabricated finite guarantee.

Evidence: [calibration_verification.json](evidence/v02/calibration_verification.json).
The archive also includes both calibration arrays and full evaluation receipts.

## 7. Excitable-field numerical scope

The FHN target is an explicitly declared finite semidiscrete system. For the
main draft's 128-by-128 grid, halving `dt=.05` to `.025` produced noise-free RMSE
**0.000560**, below the predeclared 0.02 threshold. The 128-to-256 mesh comparison
gave **0.04188**, above that threshold. Therefore the release makes **no continuum
mesh-convergence claim**.

The pilot has its own 64-by-64 grid with observation spacing 2. A separate
time-halving check generated all 560 rows with matched initial state, stimulus
and physical burn-in. On its predeclared final 40-row window the noise-free RMSE
was **0.000830**, below 0.02; the maximum absolute difference was **0.006723**.
That supports this pilot discretization only, without extending the failed mesh
claim to another grid.

The main receipt is [generator_validation.json](evidence/v02/generator_validation.json).
The pilot receipt is [pilot_time_validation.json](evidence/v02/pilot_time_validation.json)
and is also included with the archived engineering evidence.
Both record parameters, random seeds, numerical versions and source identities.

## 8. Remaining scope boundaries

The [model registry](benchmark_registry.yaml) distinguishes implemented models
from conditional external integrations. R/ngme2, Julia cross-scale reservoirs,
complete-grid neural operators, diffusion forecasting, pretrained-model tracks,
and track C are not exposed as implemented native comparators. They need their
own dependency, information-budget and validation work. This release does not
claim irregular/missing/multivariate-panel support where the current API requires
complete univariate panels.

Model-specific approximations and uncertainty interpretations are documented in
[benchmark_models.md](benchmark_models.md); engineering limits are in
[optimization_and_scope.md](optimization_and_scope.md). Reproducibility is tied
to the recorded runtime. Bitwise equality is not promised across arbitrary
BLAS libraries, compilers, devices or dependency versions.
