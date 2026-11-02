# Reproducible model comparisons

Version 0.2 adds an executable comparison framework to the existing Bayesian
spatial ESN package. The scientific objective is to learn which models predict
future **noisy observations** well, how uncertainty behaves, and what that costs.
It does not assume the ESN will win every scenario.

The complete model equations are in [benchmark_models.md](benchmark_models.md).
The original ESN hierarchies and MCMC derivations remain in `docs/derivations/`.
The implementation registry distinguishes implemented core comparators from
optional external integrations that are deliberately gated.

## 1. Install and verify

The statistical and reservoir models need only the core dependencies. Neural
models use an optional CPU PyTorch installation; GPU hardware is not required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.lock -r requirements-report.lock
python -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-neural-cpu.lock
python -m pip install --no-deps -e .
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
bstesn benchmark registry
```

The locks record the tested Linux CPython 3.12 versions. They are version locks,
not claims of byte-identical wheels across operating systems. The ordinary
`.[neural,report,dev]` extras provide a more portable, less tightly pinned install.
The module does not import PyTorch until a neural model is fitted or loaded.
All model serialization uses numeric arrays and JSON; no pickle is loaded.

## 2. The common API

```python
from bayes_stesn.benchmark import fit_model, save_model, load_model
from bayes_stesn.benchmark.generators import generate

simulation = generate("g2_spectral_transport", n_times=600,
                      n_side=7, n_observed=25, n_calibration=4, seed=42)
training = simulation.observed_panel(stop=500)
fitted = fit_model("separable_stgp", training,
                   config={"spatial_nu": 1.5, "temporal_nu": 0.5,
                           "max_iter": 60}, seed=43)
u = simulation.test_indices
prediction = fitted.forecast(
    40, target_coordinates=simulation.coordinates[u],
    target_site_ids=tuple(simulation.site_ids[j] for j in u),
    n_paths=256, seed=44,
)
save_model(fitted, "results/gp/model.npz")
prediction.save("results/gp/forecast.npz")
```

`Panel.y` uses `[time, site]`; `Forecast.mean` uses `[lead, site]`; optional
`Forecast.samples` uses `[path, lead, site]`. The samples represent a joint
trajectory law or the explicitly enumerated finite reservoir ensemble. A point
forecast has `samples=None`; requesting its native interval fails explicitly.
`Forecast.median` is used for absolute-error scores; `Forecast.mean` is used for
squared-error scores. Parameter uncertainty is Bayesian only for the ESN.

Models never receive `Simulation`, latent truth, an evaluator callback, or
never-observed-site histories. An extended `history=Panel(...)` must preserve
the exact fitted prefix, geometry, identities, and time values. It conditions
states using newly observed O values while keeping fitted parameters frozen.
Within any forecast horizon, all recursive inputs are model-generated.

The benchmark convention is: origin `o` exposes rows `[0:o]` and predicts
`[o:o+H]`. The legacy ESN API instead takes the last available row `o-1`; the
adapter makes that conversion explicitly. History-window experiments preserve
actual time coordinates while shortening the prefix supplied to the model.

## 3. Which models are comparable?

| Model ID | Observed sites A | Never-observed sites B | Native uncertainty |
|---|---|---|---|
| `persistence` | Yes | No | Point |
| `seasonal_naive` | Yes | No | Point |
| `spatial_persistence` | Yes | Yes | Point |
| `basis_lgssm` | Yes | Yes | Gaussian paths, plug-in parameters |
| `separable_stgp` | Yes | Yes | Gaussian paths, plug-in parameters |
| `basis_krr` | Yes | Yes | Point |
| `bayes_esn_ridge` | Yes | Yes | Joint posterior predictive paths |
| `eqesn_rollout` | Yes | No | Actual finite reservoir ensemble |
| `dcrnn` | Yes | No | Point; declared autoregressive diffusion-GRU variant |
| `basis_gru` | Yes | Yes | Plug-in ancestral paths with Gaussian innovations |

The eight-model core track A roster excludes the optional `basis_gru` control
and the spatial interpolation baseline. Track B adds `spatial_persistence`
automatically and filters unsupported models. A failed native new-site request
is not silently replaced by an invented interpolation of learned node loadings.

The recursive E-QESN adaptation is different from the cited paper's particular
experiments. It holds one random reservoir member fixed through each complete
rollout. Its distribution has exactly `members` atoms, regardless of a larger
requested `n_paths`; it adds no invented measurement innovations. The diffusion
GRU shares its transition between context processing and autonomous rollout; it
is not advertised as the original separate DCRNN encoder/decoder architecture.
The optional basis GRU is a custom control, not a named-paper reproduction.

## 4. Shared datasets and difficulty

Each independent dataset is generated once and reused across every model.
Observed O, label-only calibration C, and final new-site U roles are chosen from
geometry before looking at responses. C and U are disjoint from O and each
other. A contiguous spatial hole challenges extrapolation; interleaved U sites
challenge interpolation. Coordinates and immutable master site IDs are saved.

| Generator | What it isolates | Important limitation |
|---|---|---|
| G1 Gaussian spatial AR | Easy, correctly specified stochastic reference | Oracle uses noisy O history, not hidden states |
| G2 spectral transport | Smooth periodic advection and diffusion | Finite Fourier modes define the exact DGP |
| G3 excitable field | Nonlinear hidden-state dynamics and moving patterns | Explicit finite-grid FitzHugh-Nagumo system; no continuum convergence claim |
| G4 existing fronts | Compatibility with the earlier nonlinear benchmark | Exact existing Gaussian DGP is retained |
| G5 noise stress | Spatial field t, local measurement t, or hidden-state measurement variance | Separate arms; no hidden variance covariate exposed to models |
| G6 velocity shift | A hidden change during the forecast horizon | First affected row is `origin + break_lead - 1` |

G1 starts in its stationary distribution; the spectral simulator respects
conjugacy, exact transition variance, and stationary initialization. G5 t noise
is variance matched. Stress arms use separate named process and measurement
streams so the unaffected component stays paired. G3's temporal discretization
and morphology receipts are retained: the initial half-step check passed, while
the 128-to-256 mesh difference exceeded the proposed continuum tolerance. That
is reported, not hidden by changing the tolerance after seeing the result.

The calibration workflow rejects a blind future-break interpretation if its
blocks already contain the regime change; post-break recovery is a different
scientific question and requires a separately declared profile.

## 5. Profiles, tuning and final evaluation

```bash
bstesn benchmark profile smoke --output configs/benchmark/smoke.yaml
bstesn benchmark run --config configs/benchmark/smoke.yaml --output results/smoke
bstesn benchmark profile pilot --output configs/benchmark/pilot.yaml
bstesn benchmark run --config configs/benchmark/pilot.yaml --output results/pilot --workers 4
```

`smoke` is an execution test. Its two short MCMC chains intentionally cannot
establish the four-chain scientific gate. `pilot` supplies development evidence
with five independent datasets, a 7-by-7 evaluation grid, 25 O sites, 4 C sites,
400 initial historical rows, three 40-step validation blocks, and a final
40-step forecast at row 520. Its candidate/epoch/path budgets are smaller than
the proposed confirmatory study. Neither is mislabeled a completed large study.

The main design uses a 25-by-25 grid, O=100, C=20, U=505, training `[0:2000)`,
validation origins 2000, 2100, 2200, final refit `[0:2300)`, and test
`[2300:2400)`. All three validation forecasts keep candidate parameters fixed
but condition on O between origins. Track B tuning hides a contiguous fifth of
O, refits all response-dependent transforms on the remaining sites, and scores
the hidden O block. U responses are not used for any tuning.

The search contains at most 12 declared candidate configurations per model and
track. Native probabilistic models select average validation CRPS; point models
select average validation RMSE. The same trial cap does not imply equal compute.
The best stochastic candidates are rechecked on fresh fit seeds; a candidate
with a failed recheck is not rewarded by dropping that failure. Final fit seeds
are predetermined, independent, and never chosen by favorable test performance.

Neural models select epochs using a chronological tail within the supplied
training data, with free-running validation. They then refit all supplied
training rows for the selected epoch count. State-space likelihood optimization
and Gaussian GP optimization report convergence and boundary behavior. ESN
mixing is assessed again after the final refit, independently of screening.

Confirmation requires a frozen design/power receipt bound to its exact resolved
profile; use `bstesn benchmark --help` for the freeze/finalization command. R is
planned from paired development variances at target normalized MCSE 0.02 and
bounded to 20-60 independent DGP datasets. If the 60-dataset cap binds, the
unmet precision target remains explicit. The final seed namespace differs from
development. Do not look at final results and then change R or the search space.

## 6. Qualifications, scores and uncertainty

The Bayesian ESN requires four chains, rank-normalized R-hat below 1.01,
bulk/tail ESS at least 400, and checks of recovered readout functionals as well
as hyperparameters. One predeclared retained-draw continuation is permitted.
It resumes the exact saved state. Thresholds do not relax after a failed fit.
Benchmark fits default to checkpointing at each completed chain. An explicit
`sampler.checkpoint_every` restores a chosen intermediate cadence. This avoids
repeatedly rewriting thousands of tiny NPZ entries during tuning while retaining
exact chain-end continuation; an interrupted incomplete chain must be rerun.
The original standalone `SamplerConfig` default remains 100 iterations.

All final attempts enter the operational RMSE/MAE tables. Failed fits use
persistence for A and fixed spatial persistence for B, and fallback costs are
recorded. Native CRPS, coverage, interval width, interval score, energy, spatial
variogram, temporal-lag variogram, and fixed path-functional scores use qualified
native distributions. Their populations and failure counts appear alongside
the scores. Pairwise probability comparisons use the common qualified subset;
this is a conditional comparison, not an unconditional model ranking.

Empirical CRPS is evaluated exactly using the sorted-sample identity. The
energy score uses a saved seeded pair budget. Spatial pair selection depends
only on geometry and uses bounded-memory sampling. A common additive spatial
error is invisible to variograms, so variogram scores are never used alone.
Marginal coverage does not establish full-field or full-trajectory coverage.

Uncertainty bars and paired bootstrap intervals resample **independent complete
DGP datasets**. Sites, forecast leads, overlapping forecasts, and fit seeds are
not counted as independent replications. Reported MCSE describes finite
simulation precision, not a universal population guarantee.

## 7. Calibration and learning curves

```bash
bstesn benchmark calibrate results/pilot --scenario g1_gaussian_ar \
  --model separable_stgp --track B --blocks 25 --output results/calibration_gp
bstesn benchmark curve history --config configs/benchmark/pilot.yaml \
  --values 500,1000,2000,4000 --output results/history_curve
```

Calibration extends the same seeded dataset and verifies its original prefix.
Base parameters stay frozen. At each block, only previously observed O values
condition the state; forecasts remain autonomous within the horizon. Raw and
calibrated predictions are compared at the same later test origin.

Pointwise, field-at-lead and whole-path maximum scores are separate calibration
targets. With m blocks, the quantile rank is `ceil((m+1)*(1-alpha))`; a rank
greater than m yields infinity. For 25 blocks, 90% uses rank 24, 95% rank 25,
and 99% is infinite. Serial blocks are not assumed exchangeable. Transferring
bands from C to U is explicitly empirical; the C maximum is not a coverage
guarantee for the larger U domain. Calibration changes intervals, never native
joint samples. Finite interval sets receive interval scores/WIS, not an invented
exact distributional CRPS.

History curves use a common initial validation endpoint and final test period.
Sensor curves reserve the largest O master set, keep C/U fixed, and score A on
the common smallest sensor subset. Increase the grid before requesting 200
sensors. Grid-scale experiments should use the resource-appropriate operator
route and remain separate from these history/density experiments; no giant
unjustified full factorial is run by default.

## 8. Efficiency, reproducibility and reports

For the final portable PDF export, run `python tools/embed_report_fonts.py
report/benchmark_pilot` after the report command. It embeds the pinned
Matplotlib DejaVu fonts, re-renders only the PDF from verified saved figures,
and updates its hash inventory. Numerical artifacts and animations are unchanged.

The Bayesian forecast API supports `components=("observation",)`, a preflight
output-byte budget, `path_start`, and `iter_predictive(..., batch_size=...)`.
Global path IDs make batch concatenation reproduce a monolithic call in the
same numerical environment. Internally generated arrays are frozen without
redundant complete copies; public constructors still protect callers' arrays.
The new ensemble path scheme and pair-selection scheme are versioned because
their exact seeded realizations intentionally changed.

The end-to-end sparse route is explicitly limited to iid Gaussian scalar ridge
with collapsed latent-Vecchia inference. It uses sparse solves/log determinants,
operator sufficient statistics and ancestral predictive generation. It avoids
dense site covariance matrices in that supported combination; sparse LU fill
still depends on geometry. SPDE, non-Gaussian and residual-AR combinations retain
their documented bounded reference routes. See the engineering receipts for
measured resource costs; no million-site claim follows from a small profile.

Parallel jobs use processes, with one BLAS thread per process and a small worker
cap. Named seeds do not depend on worker completion order. Final output paths
are disjoint; only the parent writes aggregate summaries. Resume requires the
same protocol, package source fingerprint and numerical environment. Saving is
atomic; numeric archives carry content checksums. A per-job record preserves
fit diagnostics, the selected settings, every tuning attempt and its cost,
failure traces, stage, data/fit/forecast seeds, and source identity.

```bash
bstesn benchmark report results/pilot --output report/benchmark_pilot
```

Rendering never fits a model. It produces PNG/SVG scientific figures, a PDF,
an HTML report, MP4 when FFmpeg is available, and GIF fallback animations. The
preselected replication-zero ESN illustration shows actual noisy observations,
forecast means, signed errors, and leadwise error evolution on synchronized
panels. Colors stay fixed across time; missing target cells are gray. Native
forecasts that fail qualification are labeled diagnostic. Aggregate tables,
rather than selected animations, support performance comparisons.
