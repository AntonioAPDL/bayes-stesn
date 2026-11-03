# Reproducible model comparisons

The benchmark compares forecasts of future **noisy observations**, their
uncertainty, and their computational cost. The v0.3 workflow adds effective
candidate resolution, strict finalist qualification, matched study designs,
and recoverable resource-aware execution. These are implemented methods;
their presence does not establish a forecasting improvement or a completed
confirmatory study.

The complete model equations are in [benchmark_models.md](benchmark_models.md).
The original ESN hierarchies and MCMC derivations remain in `docs/derivations/`.
The implementation registry distinguishes implemented core comparators from
optional external integrations that are deliberately gated.
See [tuning details](benchmark_tuning_v03.md),
[study estimands and learning curves](study_design_v03.md), and the
[P0–P8 requirements/evidence map](plans/v03_requirements.yaml) for the new work.

## 1. Install and verify

The statistical and reservoir models need only the core dependencies. Neural
models use an optional CPU PyTorch installation; GPU hardware is not required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.lock -r requirements-report.lock
python -m pip install --no-deps -e .
bstesn benchmark registry
```

The locks record the tested Linux CPython 3.12 versions. They are version locks,
not claims of byte-identical wheels across operating systems. Before executing
the full roster or optional neural tests, follow the CPU-Torch installation in
[reproduction.md](reproduction.md): install its support packages from PyPI, then
the pinned Torch wheel from the CPU index with `--no-deps`. The CPU index alone
does not supply the complete dependency closure. The ordinary
`.[neural,report,dev]` extras provide a less tightly pinned installation.
Candidate resolution requires no neural fitting. Model serialization uses
numeric arrays and JSON; no pickle is loaded.

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

The legacy schema-3 track A roster has eight models and excludes `basis_gru`
and the spatial interpolation baseline. The schema-4 `v03_*` profiles include
`basis_gru`, so their full execution needs CPU Torch. Track B adds
`spatial_persistence` automatically and filters unsupported models. Validation,
resource plans, and execution use this same effective roster. A failed native
new-site request is not silently replaced by an invented interpolation of
learned node loadings. The named `bayes_esn_ridge` comparator enforces scalar
ridge and iid innovations; other core ESN variants require separately declared
benchmark models.

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

Profiles are versioned scientific procedures. The original `smoke`, `pilot`,
and `main` names retain schema 3.0, their dictionaries and identities, and their
legacy selection rules. `v03_smoke`, `v03_pilot`, and `v03_main` use schema 4.0.
The default of `benchmark run` remains legacy `smoke`; explicitly choose the
new profile or a checked-in YAML to use the new policy.

| Profile/configuration | Purpose | Initial rows / H | Replications | Selection |
|---|---|---:|---:|---|
| `v03_smoke` | Execution only | 120 / 8 | 1 | 1 candidate, 1 finalist seed |
| `v03_pilot` | Small development | 400 / 40 | 5 | Up to 8 distinct candidates, 2 finalists × 3 fresh seeds |
| `configs/benchmark/v03/development.yaml` | Development at intended main dimensions | 2,000 / 100 | 5 | Same substantive schema-4 policy |
| `v03_main` / `confirmation_draft.yaml` | Unfrozen confirmation draft | 2,000 / 100 | Draft only | Requires revised development evidence and a freeze |

The smoke configuration has deliberately short fitting budgets and cannot
establish the four-chain ESN scientific gate. A smaller development profile
does not supply matched-dimension confirmation evidence for the main design.

### Resolve and budget before running

```bash
bstesn benchmark validate --config configs/benchmark/v03/pilot.yaml \
  --output results/v03/planning/resolved.json
bstesn benchmark plan --config configs/benchmark/v03/pilot.yaml \
  --workers 2 --max-memory-mb 5120 --output results/v03/planning/costs.json
```

`validate` (alias `resolve`) checks configuration and writes effective candidate
receipts without generating a dataset or fitting a model. `plan` reports fit
counts, memory/storage estimates and admission status. Candidate proposals vary
with each dataset's prespecified seed; a plan uses a representative search seed
and labels it. The runner also checks the actual job's resolved dimensions.
Supply `--measurements fit_seconds.json`, a JSON mapping from model IDs to
representative per-fit wall seconds, to obtain time projections; otherwise
those values are null. This is planning evidence, not a measured benchmark.

The main geometry is a 25-by-25 grid, O=100, C=20, U=505, training `[0:2000)`,
validation origins 2000, 2100, 2200, final refit `[0:2300)`, and test
`[2300:2400)`. All three validation forecasts keep candidate parameters fixed
but condition on O between origins. Schema-4 track B uses prespecified edge
and interior pseudoheld O blocks, refits all response-dependent transforms on
each remaining subset, and averages folds equally. U histories never enter
tuning. A third geometric fold is optional and changes the frozen design.

### Distinct candidates and qualified finalists

The schema-4 resolver applies recursive overrides, adapter defaults and actual
fitting-subset constraints before deduplicating canonical effective options.
It records hashes, constrained dimensions, invalid proposals, and whether it
reached the requested count, exhausted the finite space, or hit its proposal
cap. A simple baseline may have fewer than eight distinct configurations.

Native probabilistic candidates use average validation CRPS; point models use
average validation RMSE. Screening ranks candidates. A finalist must qualify
in **every** planned fresh-seed and spatial-fold recheck. Its selection score
averages those rechecks; failed rechecks cannot be dropped to improve its rank.
Deterministic models receive one recheck. If neither initial finalist qualifies,
up to two further ranked candidates receive the same rechecks. If none qualifies,
the recorded outcome is the declared fallback. Final refit seeds are independent
and fixed in advance. Final fitting must qualify again.

Eight screening configurations plus two finalists at three seeds require 14
selection fits on A or 28 with two B folds for one stochastic model/dataset,
before replacements. Across G1–G4 and 20 replications that is 3,360 selection
fit calls for that model, plus 160 final fits. Equal trial counts do not imply
equal computation. Internal neural epoch-selection/refitting, bounded MCMC or
optimizer continuation, forecasting and serialization add work; some are inside
one counted fit call. Use the executable plan for declared roster, fold and
allowance accounting; actual attempted calls remain in execution receipts.

`selection_mode: development_frozen_configuration` is a separate estimand: it
fits prespecified effective configurations on each new dataset and performs no
validation fits. Declare configurations by model or `model/A`, `model/B` in
`frozen_configurations`; model/track entries take precedence. The auto-added
spatial-persistence B control has an explicit fixed reference
`{spatial_length: 0.2, ridge: 0.01}` if omitted, recorded as
`fixed_spatial_control`. Other missing declarations fail. Select and freeze the
mode before final outcomes; switching modes is not a transparent cost reduction.

Neural models select epochs using a chronological tail within the supplied
training data, with H-step free-running validation in new profiles. A prefix
may shorten to leave H leads; infeasible context/history combinations fail.
The selected epoch count is refitted on all supplied rows. This does not change
the training objective into an H-step loss. Bounded EM and GP/GRU optimizer
allowances preserve their convergence criteria and report cap exhaustion,
objective progress and boundary behavior; see [the tuning specification](benchmark_tuning_v03.md).

Confirmation requires a frozen design/power receipt bound to its exact resolved
profile. Schema 4 uses prespecified paired macro contrasts and at least five
usable development pairs in each constituent primary cell, at matched geometry,
horizon, source, runtime and tuning policy. Attempted R accounts for joint
qualification attrition and is bounded to 20–60 independent datasets per
mechanism at target normalized MCSE 0.02. A cap shortfall remains explicit.
These are approximate precision calculations, not hypothesis-test power.
Pointwise intervals do not establish simultaneous superiority. The
[study-design document](study_design_v03.md) defines the formulas, populations,
freeze fields and limits. No final results may choose R or the search space.

## 6. Qualifications, scores and uncertainty

The Bayesian ESN requires four chains, rank-normalized R-hat below 1.01,
bulk/tail ESS at least 400, and checks of recovered readout functionals as well
as hyperparameters. One predeclared retained-draw continuation is permitted.
It resumes the exact saved state. Thresholds do not relax after a failed fit.
Benchmark fits default to checkpointing at each completed chain. An explicit
`sampler.checkpoint_every` restores a chosen intermediate cadence. This avoids
repeatedly rewriting thousands of tiny NPZ entries during tuning while retaining
exact chain-end continuation. After interruption, only work since the last
committed chain checkpoint needs replay. A fit-phase sidecar records initial
sampling/diagnostics and the one permitted continuation so restarting the
process neither repeats a completed phase nor grants a fresh draw allowance.
The original standalone `SamplerConfig` default remains 100 iterations.
The opt-in exact `ridge_total_scale` sampler coordinates and recovery contract
are derived in [chapter 07](derivations/07_sampler_coordinates_recovery.md).

Completed scientific outcomes enter the operational RMSE/MAE tables. Methods
that exhaust their declared numerical or diagnostic allowance use persistence
for A and fixed spatial persistence for B, with fallback costs recorded.
Unstarted or interrupted infrastructure work is incomplete, not a scientific
fallback. Native CRPS, coverage, interval width, interval score, energy, spatial
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
bstesn benchmark calibrate results/v03/pilot --scenario g1_gaussian_ar \
  --model separable_stgp --track B --blocks 25 --output results/calibration_gp
bstesn benchmark curve history --config configs/benchmark/v03/development.yaml \
  --values 500,1000,2000,4000 --common-endpoint 4000 \
  --execute-values 1000,2000 --workers 2 --max-memory-mb 5120 \
  --output results/v03/history_curve
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

Resolve the full history curve even when executing only a subset. With endpoint
4,000 and H=100, every condition validates at 4,000/4,100/4,200 and tests at
4,300. The schema-4 default score normalization uses the common final 500
initial-fitting rows and common O sites, with `ddof=1`; its data hash and scale
are saved. A later subset uses the same immutable `curve_design.json`.
Sensor curves reserve the largest O master set, keep C/U fixed, and score A on
the common smallest sensor subset and scale. Increase the grid before requesting
more sensors than it can support. Grid-scale experiments use a resource-appropriate operator
route and remain separate from these history/density experiments; no giant
unjustified full factorial is run by default.

## 8. Run, pause, inspect and resume

After installing the full roster's optional dependencies, start with an explicit
small execution profile in a new output directory:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bstesn benchmark run --config configs/benchmark/v03/smoke.yaml \
  --output results/v03/smoke --workers 2 --max-memory-mb 5120 --max-jobs 4
bstesn benchmark status results/v03/smoke
bstesn benchmark resume --output results/v03/smoke \
  --workers 1 --max-memory-mb 4096 --max-jobs 4 --max-wall-seconds 60
```

Repeat `resume` until `status` reports completion. Without `--config`, resume
loads the saved `resolved_protocol.yaml`. A confirmation resume must also supply
its original `--power-freeze` receipt. `--max-jobs` limits submitted attempts;
`--max-wall-seconds` pauses scheduling between atomic jobs and can overrun while
active work finishes. Neither changes a fit's iteration budget. Worker count,
memory admission and chunk limits may change on resume.

The run records three identities: scientific design; computation (design plus
source and numerical runtime); and mutable execution attempts. Checkpoint,
tuning, dataset and terminal-job receipts bind the relevant inputs and content
hashes. Changed scientific settings, source, runtime or corrupted artifacts
invalidate reuse; preserve the old run and use a new directory. Preserving
schema-3 profile identity does not migrate an old run directory to the new
execution manifest automatically.

One coordinator holds a local filesystem lock. Infrastructure failures receive
the same seeds/checkpoints and at most the declared retries (default two after
the initial attempt). Exhaustion remains unresolved infrastructure work and
keeps the campaign incomplete. Both qualified and unqualified completed method
outcomes are terminal. A killed process may have unmeasured work; receipts flag
incomplete compute accounting rather than assigning zero cost.

`fit_time_limits_seconds` optionally declares per-model elapsed-time allowances
in the protocol. A timer covers each fitting call, not forecasting/scoring.
The default is no elapsed-time fit cap; algorithmic caps still apply. Choose
timer values from representative fits and freeze them before the study. The
implementation uses POSIX `setitimer` in a worker's main thread, rejects an
existing active alarm, and restores the signal handler. Native calls can delay
Python signal delivery, so this is not a hard process deadline. The coordinator
uses POSIX `flock`; this execution workflow is currently Linux/POSIX oriented.

Memory admission respects the enforced host/cgroup limit, current headroom,
an explicit reserve, and the requested budget. The runner can reduce worker
count, or pause an oversized job, without shrinking scientific dimensions.
These are conservative model-based estimates, not an OS memory sandbox or a
guaranteed peak. See [optimization and measurement scope](optimization_and_scope.md).

`memory_current_bytes` always records the raw cgroup charge. On cgroup v2,
complete valid `memory.stat` fields permit a separate admission estimate:
eligible cache is `max(0, file - shmem - file_dirty - file_writeback -
file_mapped - unevictable)`, capped by raw usage; the credit is at most half
that eligible cache. The exclusions can overlap, so subtracting all of them
deliberately understates eligible cache. Anonymous and kernel memory receive
no credit. Missing, malformed or inconsistent statistics use raw usage.
The 50% fraction is a documented scheduling heuristic, **not** a kernel
reclaim guarantee or a measurement of available RAM; active clean file pages
can still be in use by another workload. Raw statistics, eligible bytes,
credit and policy are retained in execution receipts. The enforced limit,
65% budget cap, up-to-2-GiB reserve, requested budget and active-job
reservations are unchanged. The runner writes neither cgroup limits nor
reclaim controls. The kernel's [cgroup v2 memory documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory)
defines the underlying counters; shared workloads can still change resource
pressure between admission checks.

## 9. Efficiency, reproducibility and reports

For the presentation export, use `python tools/render_benchmark_layout.py
RUN_DIR --output REPORT_DIR`. It applies shared-axis label spacing and categorical
row padding while rendering saved numerical results. Then run
`python tools/embed_report_fonts.py REPORT_DIR` for the portable PDF. It embeds the pinned
Matplotlib DejaVu fonts, re-renders only the PDF from verified saved figures,
and updates its hash inventory. Numerical artifacts and animations are unchanged.

The Bayesian forecast API supports `components=("observation",)`, a preflight
output-byte budget, `path_start`, and `iter_predictive(..., batch_size=...)`.
Global path IDs make batch concatenation reproduce a monolithic call in the
same numerical environment. Internally generated arrays are frozen without
redundant complete copies; public constructors still protect callers' arrays.
The benchmark scorer still materializes its sample arrays: this is not an
end-to-end streaming implementation. The fresh-process ownership probe in
[the study-design document](study_design_v03.md) measures one avoided copy,
not total model-fitting memory or universal speed.
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
bstesn benchmark report results/v03/pilot --output report/benchmark_pilot --no-animations
python tools/render_benchmark_layout.py results/v03/pilot --output report/benchmark_pilot
python tools/embed_report_fonts.py report/benchmark_pilot
```

Rendering never fits a model. It produces PNG/SVG scientific figures, a PDF,
an HTML report, MP4 when FFmpeg is available, and GIF fallback animations. The
preselected replication-zero ESN illustration shows actual noisy observations,
forecast means, signed errors, and leadwise error evolution on synchronized
panels. Colors stay fixed across time; missing target cells are gray. Native
forecasts that fail qualification are labeled diagnostic. Aggregate tables,
rather than selected animations, support performance comparisons.
