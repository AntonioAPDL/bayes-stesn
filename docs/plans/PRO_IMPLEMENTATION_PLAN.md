# PRO implementation plan and execution prompt: bayes-stesn

**Prepared:** 2026-11-03. **Purpose:** a complete execution brief for the next implementation phase.

**Status:** proposed work; this document does not claim that its new algorithms, experiments, or release have been completed.

**Repository:** https://github.com/AntonioAPDL/bayes-stesn

**Audit baseline:** `fda5ec5e4997e858f22c84b520b1b609eefa3c4e`.

**Current tested software:** v0.2.1, `ebd2d1a031796cadae5dbb32a992e10b3f7cf653`.

**Archived scientific experiment identity:** v0.2.0, `8ee53dfe9f2727d4bf25e26f38d4690007140fd9`.

## How to use this document

Give this entire document to a tool-enabled PRO session with access to the repository. It is written as an instruction to that session. All paths below are relative to the repository root unless stated otherwise. The audit, its evidence, and the current source are authoritative; do not rely on an earlier conversation being available.

Begin at the latest verified remote head that contains the audit. If development has advanced, reconcile this brief against the intervening changes and preserve completed work. The proposed next release is v0.3.0 if that version remains unused. Software version, benchmark schema version, algorithm version, and experiment identity are separate concepts.

# Execution prompt

## 1. Mission, authority, and completion standard

You are responsible for implementing, validating, documenting, benchmarking, and publishing the next research-quality version of bayes-stesn.

The objective is to improve fitting reliability and achievable prediction, make comparisons fair and interpretable, and keep computation reproducible and feasible on modest hardware. An improvement in every predictive score is not an acceptance criterion: a correct negative finding is a valid scientific result. Every promoted numerical or statistical change needs evidence.

The user has already authorized ordinary project work: repository edits, tests, experiments within the available resources, commits, pushes, branch/PR work, and publication to AntonioAPDL/bayes-stesn. Continue autonomously through authorized steps. Respect provider permissions, protected branches, and actual tool capabilities. Do not request repeated confirmation for routine work; if a real access boundary blocks an action, finish independent work, preserve a concrete reviewable deliverable, and identify that exact boundary.

Use a single coordinator for shared protocol interfaces, source integration, and publication. Delegate independent inference, benchmark, and systems work when useful. Assign file ownership before parallel editing. Do not launch full scientific jobs while agents are modifying the source tree those jobs import. Send concise progress updates during active work, explaining findings and the next gate. Use existing resources; paid infrastructure provisioning or broad account-permission changes require separate explicit scope.

Complete every mandatory requirement below. For each conditional extension, implement or verify its bounded evaluation path, assess whether its trigger is met, and record one disposition: promoted, retained as optional, rejected for this study, or deferred with a specific technical reason. Existing implementations should be reused. Avoid a new experiment framework, a large dependency stack, or an exhaustive factorial study without measured need.

## 2. Read first and establish the evidence baseline

Read:

1. [The complete post-release audit](../audits/post_release_2026_11_03/README.md) and its evidence directory.
2. [Model hierarchy](../models.md), [implementation registry](../implementation_registry.yaml), and [derivations](../derivations/).
3. [Benchmark guide](../benchmark_guide.md), [model definitions](../benchmark_models.md), and [benchmark registry](../benchmark_registry.yaml).
4. [Validation evidence](../benchmark_validation.md), [engineering limits](../optimization_and_scope.md), [reproduction](../reproduction.md), and [publication status](../publication.md).
5. Applicable repository instructions, tests, configuration profiles, dependency locks, and the current Git history.

Carry these audited facts into planning:

| Finding | Consequence |
|---|---|
| Pilot ESN qualified in 21/40 attempts; LGSSM in 15/40 | Repair fitting before expanding the campaign |
| Every final pilot ESN required its first continuation | Inspect parameter geometry and adaptation before adding draws everywhere |
| Original scalar-ridge chains show a strong precision/amplitude tradeoff | Test the exact coordinate change in Section 6 |
| Raising the LGSSM cap from 50 to 200 rescued five of six selected jobs | A bounded allowance is justified; convergence criteria remain unchanged |
| Basis GRU currently has one effective candidate | Implement genuine tuning and include it explicitly in the revised roster |
| Duplicate proposals and shallow nested overrides reduce effective searches | Resolve configurations before deduplication |
| Neural stopping horizons are shorter than the benchmark horizons | Align actual stopping data and objectives |
| A richer demonstration changed history, sensors, horizon, and architecture together | Its gain does not identify the value of any one change |
| G1 GP point errors were close to the known-parameter reference | Additional data cannot remove future stochastic innovation |
| Forecast and memory receipts have avoidable copies or ambiguous measurement scope | Profile and repair the consequential paths |
| Interrupted queued work and fresh ESN checkpoint recovery have source-confirmed risks | Repair resumption before a large campaign |

These are development findings from the pinned audit, not claims about the future implementation. Preserve the old experiment results and tags.

## 3. Working structure and traceability

Extend the existing public interfaces: `ModelSpec`, `SamplerConfig`, `Panel`, `Forecast`, `fit_model`, `save_model`, `load_model`, and `bstesn benchmark`. Keep complete univariate panels as the supported input contract unless a separately scoped change adds missingness or multivariate support.

Create a small progress ledger and machine-readable requirement table, for example:

~~~text
docs/plans/v03_progress.md
docs/plans/v03_requirements.yaml
configs/benchmark/v03/
docs/evidence/v03/
results/v03/development/
results/v03/confirmation/
report/v03/
~~~

These are proposed destinations, not existing artifacts. Keep large results and movies out of ordinary Git commits.

For every requirement record: ID, source finding, implementation files, tests, evidence receipt, status, and any justified limitation. Each milestone records its commit, remaining gates, exact next command, and locations of recoverable checkpoints. Distinguish implemented code, an executed check, a completed experiment, and a scientific conclusion.

Primary module map:

| Workstream | Existing modules to extend |
|---|---|
| Sampler coordinates and inference | `sampling.py`, `gaussian.py`, `model.py`, `assessment.py` |
| ESN benchmark fitting/recovery | `benchmark/esn.py` |
| Optimizer reliability | `benchmark/linear_state.py`, `benchmark/gaussian_gp.py`, `benchmark/basis_gru.py` |
| Search and validation | `benchmark/protocol.py`, `benchmark/runner.py`, model adapters |
| Learning curves and confirmation | `benchmark/studies.py` |
| Resource planning and job states | `benchmark/runner.py`, `benchmark/cli.py`, `io.py` |
| Forecast ownership and batching | `prediction.py`, `benchmark/contracts.py`, `benchmark/esn.py`, `benchmark/metrics.py` |
| Reporting and release | `benchmark/report.py`, existing renderers, `tools/release_check.py`, `tools/package_release.py` |

Source-module paths in this table are under `src/bayes_stesn/`; paths beginning `tools/` are relative to the repository root.

## 4. Stage P0 — baseline, environment, and bounded execution

### Actions

- Read the remote head, local status, release tags, and relevant changes since the audit. Preserve unrelated work. Use an isolated branch/worktree when necessary.
- Record the starting commit, source fingerprint, Python/dependency/BLAS versions, CPU quota, affinity, cgroup memory ceiling/current use, swap, and free disk.
- Use the existing locked CPython 3.12 CPU-neural environment for the full release gate where available. Preserve core support and lazy optional Torch imports.
- Run the existing targeted baseline tests and a tiny execution smoke. Investigate pre-existing failures before attributing them to your changes.
- Build and save a job-count/resource estimate before scheduling each wave. Include candidates, spatial folds, finalist seeds, rescue, final refits, prediction, and storage. Persist projected total wall/CPU time, peak memory, output/temporary storage, caps, and the next runnable chunk. For one stochastic model, eight screens plus two finalists times three seeds means 14 A selection fits; two B spatial folds mean 28 B fits. Across four mechanisms and 20 datasets each, that model alone needs 3,360 selection fits before replacements, rescue, and final refits. Validation origins add forecast work without necessarily adding parameter fits. Use safely resumable chunks; do not assume the whole campaign fits one session.
- Keep short smoke tests, numerical-reference tests, statistical development experiments, and confirmation as separately labeled activities.

### Default resource policy

The audit observed an 8 GiB memory ceiling and eight CPU equivalents; remeasure rather than assuming those limits still apply.

Start with **at most two model workers, capped by effective CPU availability and memory admission, and one numerical thread per worker**. On an 8 GiB host, admit roughly **5 GiB of aggregate job memory**, leaving roughly **2 GiB of headroom plus space for other current processes**. The effective budget must respect the smallest enforced limit and current occupancy. A user-supplied allowance cannot override the cgroup ceiling.

Use resolved model dimensions and isolated measurements to refine admission. Include covariance factorizations, coefficient work arrays, neural state, retained hyperstates, joint forecast sites, predictive samples, copies, and scoring buffers. Reject or defer oversized jobs before their principal allocation.

When memory is tight, reduce concurrency or a proven-equivalent batch size. Changes to data, model rank, retained draws, paths, or diagnostic thresholds require a new declared scientific configuration.

Use algorithmic caps throughout. Derive candidate time ceilings and campaign estimates from representative measured fits, write them to the resolved protocol before the affected search, and include safety margin. A time ceiling that can terminate a candidate is part of the scientific procedure. Resource estimates are not guarantees.

**Exit gate:** a clean baseline record, a reproducible environment, a feasible initial job plan, and no unexplained failure in the relevant existing checks.

## 5. Stage P1 — orchestration, identity, recovery, and resource accounting

Implement this before scaling experiments.

### P1.1 Separate three identities

- **Design identity:** data and DGP definitions, temporal/spatial roles, model roster, effective search space, selection objectives, all scientific seeds, diagnostic gates, optimization/inference budgets, continuation and retry-to-new-seed rules, path counts, scoring conventions, primary comparisons.
- **Computation identity:** design identity plus source fingerprint, numerical-runtime compatibility, algorithm/RNG versions, and serialization schema.
- **Execution attempt:** scheduling workers, current hardware, memory admission allowance, equivalent batching, checkpoint-write cadence, timestamps, process IDs, and interruptions.

Only output-preserving execution settings are mutable on resume. An EM cap, maximum epochs, candidate timeout, or MCMC continuation allowance remains frozen even if described as a resource parameter.

Version new schemas. Read old artifacts without silently reinterpreting their meaning. Exact continuation may require the original runtime and legacy algorithm.

### P1.2 Distinguish scientific failure from interrupted work

Use explicit job/attempt states: pending, running with phase, retryable infrastructure interruption, qualified terminal result, unqualified terminal result with declared fallback, unresolved infrastructure failure, and invalidated study.

Only a verified terminal scientific receipt may be skipped on resume. A queued job that never ran remains pending. Source drift or inconsistent provenance invalidates the run; it does not manufacture a model fallback.

Default infrastructure recovery: initial attempt plus at most **two retries**, preserving the same scientific seeds and compatible checkpoint. Record all known costs. Exhausted infrastructure recovery leaves an incomplete campaign. A model exhausting its declared fitting budget receives the prescribed operational fallback and remains in the relevant denominator.

### P1.3 Make ESN recovery phase-aware

Persist the initial retained-draw target, permitted continuation target, current phase, completed diagnostic gates, chain counters, checkpoint identity/hash, and serialization status.

Recover a compatible checkpoint on a new benchmark invocation. Never overwrite surviving state by default, resume a continued chain to a smaller target, repeat warmup on retained samples, duplicate draws, or grant an extra continuation after each restart.

Retain endpoint-only checkpoint writing for inexpensive fits. Add measured intermediate cadence for expensive jobs when the cost of lost work warrants it. Preserve the v0.2.1 array-layout correction.

### P1.4 Measure useful costs

Record selection, fitting, diagnostics, prediction, scoring, serialization, fallback, and retry costs. Distinguish summed job wall time, campaign elapsed time, and CPU time.

Measure representative model peaks in fresh subprocesses. Label reused-worker high-water RSS as process-lifetime data. Monitor aggregate cgroup/process-group memory separately. Keep estimated and measured peaks distinct.

### Acceptance evidence

- Interrupt a tiny campaign with queued tasks; resume with fewer workers and obtain identical same-runtime numerical payloads, seed assignments, and diagnostic decisions. Attempt timestamps, measured costs, and retry/resource metadata should differ appropriately.
- Preserve already verified results; retry only unfinished work.
- Reject incompatible data, source, seed, algorithm, or numerical-runtime changes.
- Inject a declared method failure and verify its fallback/count; inject source drift and verify invalidation.
- Interrupt ESN warmup, initial sampling, the first diagnostic boundary, and continuation; verify same-runtime uninterrupted/resumed chains and forecasts.
- Clamp/reject an allowance above the actual memory ceiling; make planning respond to effective dimensions.
- Preserve single ownership and atomic writes for each job directory.

Use focused fault-injection tests on tiny fixtures. Do not exercise these tests by endangering the full host or corrupting released evidence.

## 6. Stage P2 — reliable inference and optimization

### P2.1 Exact sampler-coordinate improvement

For the audited iid Gaussian scalar-ridge model,

\[
R_q=I+qK,\qquad
b\mid\sigma^2,\lambda\sim N(0,\sigma^2\lambda^{-1}I),
\]

where K has unit diagonal. Define

\[
a=\log\lambda,\quad \xi=\log q,\quad \zeta=\log\ell,\quad
\eta=a+\operatorname{softplus}(\xi),
\qquad \operatorname{softplus}(\xi)=\operatorname{logaddexp}(0,\xi).
\]

The inverse is

\[
a=\eta-\operatorname{softplus}(\xi).
\]

The determinant of the Jacobian between the original and new **log coordinates** is one. For the existing collapsed density with respect to those log coordinates,

\[
\widetilde{\log\pi}(\eta,\xi,\zeta\mid Y)
=\log\pi\{\eta-\operatorname{softplus}(\xi),\xi,\zeta\mid Y\}.
\]

A symmetric random-walk proposal in the transformed coordinates uses the ordinary transformed log-target difference in its MH acceptance ratio.

The interpretation is

\[
v=\sigma^2(1+q),\qquad
\kappa=\lambda(1+q),\qquad
\widetilde R_q=\frac{I+qK}{1+q},
\]

giving residual covariance \(v\widetilde R_q\) and prior covariance \(v\kappa^{-1}I\). Keep \(\sigma^2\) collapsed in this first implementation. Do not introduce independent new priors on \(\eta,\kappa,v\); sampling v explicitly would require its induced prior and the appropriate Jacobian.

Add an explicit sampler-coordinate option, initially opt-in. Reuse the existing normalized likelihood, original priors, and posterior recovery. Apply the transform consistently to initialization, MAP/Laplace calibration, proposals, adaptation, and checkpoint state. Recompute proposal geometry in the chosen coordinates. Adapt only during warmup.

Retain original scientific quantities in saved posterior states and diagnostics. Record the coordinate option in the checkpoint signature. Initially certify the collapsed scalar-ridge, iid, relative-scale, stationary Matérn route. Validate support for other combinations separately; a shared interpretation cannot be assumed for separate physical scales, varying nugget, nonstationary covariance, or sparse approximations.

### Required mathematical checks

- Stable forward/inverse transformations over small and large finite amplitudes.
- Pointwise target equality, including original prior and normalization terms.
- Independent tiny-data integration/quadrature for the coupled precision/amplitude posterior and recovered predictive functionals.
- Distributional agreement of old/new algorithms within quantified Monte Carlo/reference error.
- Exact same-runtime restart equivalence **within** each algorithm; old/new trajectories need not be bitwise identical.
- Unchanged meaning of covariance, recovered observation scale, readout functional, and prediction diagnostics.
- Explicit compatibility errors instead of accidental unsupported dispatch.

### Bounded sampler development

Use prespecified G1/G3/G4 development datasets and archived settings. Compare old/new algorithms on identical statistical targets with independently prespecified chain seeds. Report initialization, warmup, retained sampling, qualification, invalid proposals, minimum bulk/tail ESS per second, and Monte Carlo error for fixed predictive functionals.

First proposed budget: **four chains, 1,000 warmup iterations, 2,000 retained draws per chain, one continuation to 4,000 retained draws**. Use this executable default: stop if the initial diagnostic gate passes; otherwise extend once if every chain completed, retained scientific quantities/log densities are finite, and no unresolved computation error occurred, subject to the frozen time cap. Evaluate the same gate at the maximum and retain the outcome. Persist the decision so restart never grants another extension. An improvement-based alternative must specify numerical windows and thresholds before execution; subjective trace assessment cannot decide the automated budget.

Keep **rank-normalized R-hat < 1.01 and bulk/tail ESS >= 400**, including the established recovered functionals. These are diagnostic gates, not a proof of exact convergence. Compare ESS for the same original scientific quantities. Stored-chain decorrelation is motivation, not a performance result.

Promote the new default only when reference checks pass and development evidence improves qualification or efficiency without a new material failure mechanism. Otherwise retain the legacy default and document the experiment.

### P2.2 Optimizer reliability

For LGSSM, provide a bounded allowance up to **200 EM iterations**, retaining the current **1e-5 relative likelihood criterion**, monotonicity checks, and stability constraints. A simple larger declared cap is acceptable before implementing elaborate EM checkpointing. Verify the historical first 50 likelihood values under the original numerical runtime. Otherwise compare the extended-cap fit against a cap-50 reference under the same current runtime with justified numerical tolerance; report historical/runtime discrepancies separately.

Record objective/parameter traces, active bounds, variance-floor behavior, iteration count, and the exact final reason. A finite fit at the cap remains unqualified.

For GP and basis-GRU covariance fits, inspect optimizer messages, objective values, projected gradients, and active bounds. Add a bounded continuation or at most three deterministic starts where supported and justified. Never replace convergence with “finite parameters returned.” Charge all starts and rescue to the corresponding method.

### P2.3 Reliability gate

Run the bounded sentinel suite, then a revised development panel spanning G1–G4. Report independent datasets and A/B jobs separately. A 90% qualification aim may guide engineering, but five datasets do not estimate that probability precisely.

Posterior-reference failures, corrupt checkpoints, or near-universal failure in a primary scenario block promotion to a large scientific campaign. Continue diagnosis within the bounded development scope. Keep difficult scenarios in the evidence.

## 7. Stage P3 — genuine tuning and task-aligned validation

### P3.1 Resolve before deduplicating

The required sequence is:

1. Generate a candidate.
2. Apply validated recursive overrides.
3. Resolve adapter defaults and fitting-subset constraints.
4. Canonicalize the complete effective configuration.
5. Deduplicate by its effective hash.
6. Assign fitting seeds in a separate named namespace keyed by that stable hash.

Do not count two raw proposals as distinct when their effective models are identical. Replenish duplicates until the requested distinct count is reached or a finite space is honestly exhausted. Bound rejection loops and record invalid proposals. Use actual Track B fitting-subset size for constrained ranks/modes. Overcomplete regularized ESN dictionaries are not universally invalid; respect the specific adapter's mathematics.

Reject unknown keys. Preserve unspecified sampled nested settings. Record dimensions, resolved options, candidate counts, exhaustion, seeds, and every cost. Explicitly include basis GRU in the revised main roster.

Default A roster: `persistence`, `seasonal_naive`, `basis_lgssm`, `separable_stgp`, `basis_krr`, `bayes_esn_ridge`, `eqesn_rollout`, `dcrnn`, `basis_gru`. Default B roster: `spatial_persistence`, `basis_lgssm`, `separable_stgp`, `basis_krr`, `bayes_esn_ridge`, `basis_gru`. Additional ESN/uncertainty variants receive explicit distinct identities. Unsupported capabilities remain explicit.

### P3.2 Bounded selection policy

Default: **eight distinct effective configurations**, **two initial finalists**, and **three fresh fitting seeds per stochastic finalist**. A later increase to twelve requires a declared development decision; do not run an exhaustive grid.

Use a declared screening fidelity. Short-budget unqualified posterior scores may be provisional screening values, but they cannot establish an eligible final candidate.

At full finalist fidelity, require every planned recheck to qualify after its allowed rescue. For Track B, this includes every required spatial-fold fit. Freeze seed, fold, and temporal-origin averaging weights; retain per-fold diagnostics. Rank eligible candidates by the resulting weighted mean score across all planned rechecks. Keep failed seed records. If neither initial finalist is eligible, allow at most **two replacement candidates** in the original screening order. If none qualifies, record selection failure and the prescribed operational fallback. Deterministic methods do not need duplicate identical seed evaluations.

Final-refit seeds are predetermined and independent. A failed final fit does not authorize selecting a favorable seed or using final truth to choose another configuration.

Selection objective: native probabilistic models use validation CRPS; point models use validation RMSE. State that resulting RMSE for a CRPS-selected model is a secondary property of that procedure. If a best-point-prediction claim is wanted, add a separately labeled RMSE-selected analysis under its own budget.

### P3.3 Initial search dimensions

These are proposed development ranges, to be validated against real API constraints:

| Family | First bounded search |
|---|---|
| Bayesian ESN | Spatial side 3/4/5 and a small prespecified width set, paired with shrinkage; encoder modes 4/8/12; lags (1,2), (1,2,4), (1,4,8); reservoir 32/64/128 or two modest layers; a small leak set and quadratic feature choice |
| Basis GRU | Rank 8/16/24 where valid; context 8/16/32; hidden width 16/32/64; a few learning-rate/decay choices |
| Diffusion GRU adaptation | Space-filling configurations emphasizing context and rollout training before adding layers |
| LGSSM | Converged optimizer first, then spatial rank and width |
| Separable GP | Converged, bounded multistart optimization; optional declared mean/kernel variants after adequacy checks |
| Basis KRR | Spatial rank/width plus its existing temporal kernel and regularization |
| E-QESN | Small reservoir/member specification search; retain the original finite-distribution interpretation |
| Simple controls | Small training-only seasonal or spatial choices where applicable; keep fallback settings separately fixed |

Select eight informative combinations, including a conservative reference, using a deterministic random or space-filling design. Do not take the Cartesian product. Separate spatial/regularization decisions from temporal-capacity expansion where that improves interpretability.

### P3.4 Match the actual forecast task

Keep three forward temporal validation origins. Candidate parameters are fitted on the initial permitted training window; new legitimate O history may update states between origins. Forecasts within each scored horizon remain autonomous.

Neural early stopping must evaluate the declared H=40 or H=100 target horizon, or a fixed declared lead-weighted objective. Ensure the historical stopping tail actually contains those leads. Do not silently truncate to a shorter horizon; record requested and realized leads. Refit supplied training history for the selected epoch count. A multistep training curriculum is a separate training-method ablation.

For Track B, use **two fixed geometry-only spatial blocks initially: one edge and one interior gap**. Add a third only when needed. Freeze site IDs, sizes, weights, and generators. Refit all response-dependent transforms inside each training subset. Use only retained-site histories to condition forecasts. Report edge/interior behavior separately as well as the declared weighted average.

### Acceptance tests

Use inexpensive fake adapters for selection/failure logic and small actual fits for integration:

- Stable candidates/seeds across scheduling and extension of a candidate list.
- Genuine basis-GRU diversity; correct finite-space exhaustion.
- Nested override preservation and correct effective dimension checks.
- An unqualified best provisional score cannot win final selection.
- Failed rechecks remain visible and replacement limits are enforced.
- Final/U/latent response perturbations leave fitting, tuning, and validation unchanged.
- Neural stopping consumes the promised historical horizon.
- Spatial folds refit preprocessing and never expose held histories to model inputs.

## 8. Stage P4 — identify the value of more history, sensors, and capacity

### P4.1 Common-endpoint history curves

Resolve all profiles with

\[
L\in\{500,1000,2000,4000\},\quad E=4000,\quad H=100,
\]

and

\[
\begin{aligned}
\text{initial fit}&=[E-L,E),\\
\text{validation origins}&=\{4000,4100,4200\},\\
\text{final refit}&=[E-L,4300),\\
\text{test}&=[4300,4400).
\end{aligned}
\]

Implement an explicit common-endpoint argument or resolve the complete four-length suite before executing a subset. The current maximum-requested-length convention cannot be reused carelessly when adding lengths later.

Start with the resolved **1,000 and 2,000** profiles on **five paired development datasets each for G1, G3, and G4**. Default initial geometry: the pilot's **7-by-7 evaluation grid, O=25, C=4, U=20**, with all role IDs resolved before viewing responses. Resolve and retain DGP settings; for G3 begin with the pilot finite-grid settings (grid size 64, dt=0.05, observation spacing 2), subject to the required numerical check. Use reliable ESN, GP, basis GRU, KRR, and applicable persistence controls. Execute 500/4,000 only if needed to clarify saturation, harm from older history, or an unresolved benefit.

Freeze the common fixed-specification configuration using prior development evidence before inspecting curve scores, and record its selection provenance. Materialize or hash-verify the same generated field realization across lengths; shared seed labels alone do not establish identity.

Separate:

- **Fixed specification:** same architecture, prior, conditioning policy, fitting targets, and seed policy; refit parameters on each permitted history.
- **Retuned pipeline:** repeat the declared tuning procedure at each length.

Fixed specification does not require learned PCA/scaling estimates to be numerically identical. A frozen-encoder comparison is an additional, separately named experiment if needed.

The current prefix-only encoder policy must be explicit. Record conditioning rows, actual likelihood rows, and supplied history L+300. Do not automatically assume longer readout history improves a prefix-trained encoder.

### P4.2 Common normalization

Use raw within-DGP losses and, where normalization is needed, **one common per-dataset scale** available to every condition. For the specified history curve, an example is the sample SD of observations from [3500:4000) at the common smallest O set, with a prespecified small numerical floor.

Reuse that same denominator across lengths and sensor counts. Do not use each condition's changing training-scale denominator. Verify that the reference contains no validation or final labels.

### P4.3 Sensor curve and representation

At fixed history and field resolution, use **25/50/100 nested O sites**. Default sensor-curve geometry: a **15-by-15 evaluation grid, O master=100, C=10, U=115**. Resolve the fixed history and DGP settings before running. Reserve the 100-site O master set before defining C/U; inactive potential sensors never become U. Keep U fixed and score A on the common smallest O subset. Preserve the same generated field. This separate sensor study has different geometry from the initial history curve; comparisons across the two studies do not isolate sensor count.

The ESN mean is

\[
\mu_t(s)=\phi(s)^\top C z_t.
\]

More temporal units cannot add missing spatial basis functions. Compare spatial size/width jointly with regularization using temporal and spatial validation. Record numerical rank, conditioning, and relevant held-site noise amplification. Treat contemporaneous reconstruction as a geometry diagnostic, not a forecast score.

A basis normalization or orthogonalization changes an isotropic coefficient prior unless its induced covariance is also transformed. Either preserve the original function-space prior exactly or label the change as a statistical variant.

Keep this stage small. Do not launch a history-by-sensors-by-width-by-prior factorial. Advance a larger configuration only when paired development performance and its uncertainty justify its additional cost.

## 9. Stage P5 — evidence-gated model and uncertainty extensions

Many extensions already exist. Verify their integration, derive only changed mathematics, and run compatible bounded ablations. Start with the simplest extension addressing an observed deficiency.

| Extension | Trigger and required evaluation | Scope control |
|---|---|---|
| Grouped ridge | Different feature groups need different shrinkage; compare against repaired scalar ridge | First prior ablation; reuse column-separable algebra |
| Group/column regularized horseshoe | A plausible sparsity assumption and remaining validation benefit | Calibrate global scale/slab; diagnose all new scales; no automatic Tn independent-design heuristic |
| Complete-residual AR | Persistent one-step residual dependence under an explicitly defined diagnostic | Preserve stationary first-row density, determinant, whitening of both Y and Z, and coherent prediction |
| Latent spatial AR | Persistent spatial signal with fresh measurement noise | Reuse FFBS, terminal filtering, and coherent new-site extension |
| Whole-field t / local measurement t | Field-wide shocks / local outliers in separate G5 arms | Preserve the correct augmentation, variance convention, and future innovation generation |
| Anisotropy / range-prior alternatives | Directional residual dependence or meaningful spatial scale knowledge | Fix coordinate units and prior elicitation; retain simple reference |
| Nonstationarity / heteroskedasticity | Residual structure persists after simpler repairs | Small declared bases; identifiable scales; future covariates available at origin |
| E-QESN residual innovations | Strong point accuracy with missing observation uncertainty | New model ID; training/validation innovation estimate; retain original finite ensemble as control |
| Interval calibration | A declared coverage objective beyond native uncertainty | Separate calibration/evaluation blocks; report width and interval score; preserve native draws |
| Periodic GP / structured LGSSM / multistep neural training | Specific temporal inadequacy supported by validation | Individually named algorithm/model variants with their own costs |

Do not bundle several statistical changes into one unidentifiable “improved ESN.”

Existing augmented dynamics/tail routes have explicit size and compatibility guards, including reference limits around n<=100 and T<=2000. Verify the actual backend and definition of T. A 4,000-row history study must not automatically bypass those guards. Use compatible ablation profiles or record a separately justified scaling task.

The efficient sparse route is currently narrowly certified for iid Gaussian scalar-ridge collapsed latent Vecchia. Preserve that supported combination and benchmark sparse fill/dense readout work. The 4,096-site execution demonstration does not establish large-data MCMC convergence. Entrywise RHS, arbitrary nonstationarity, and augmented dynamics do not automatically inherit that complexity.

For calibration, allocate blocks separately from the three selection blocks. Default to 25 nonoverlapping H-step calibration blocks from a declared post-fit origin o, with base parameters frozen; evaluate raw and calibrated forecasts on the same later block [o+25H, o+26H), using identical allowed history. This is a separately identified evaluation period. A refit changes the base predictive law and requires a declared new calibration procedure.

Maintain separate marginal, field-at-lead, and whole-trajectory targets. Preserve the rank ceil((m+1)(1-alpha)); if it exceeds m, report the infinite/unavailable finite band. Finite 90% and 95% ranks require at least 9 and 19 calibration scores respectively; these are algebraic rank requirements, not coverage guarantees. Serial blocks and C-to-U transfer do not acquire an exchangeability guarantee. Calibration alters intervals, not native joint draws; use interval score/WIS for interval-only outputs.

Keep G5 noise stress and G6 velocity shift as separate declared studies after basic reliability. G5/G4 and G6/G2 share control streams; respect their pairing. A blind future shift and post-shift adaptation are different tasks with different allowed information.

Each extension closes with its configuration, rationale, derivation reference, compatibility, prior predictive checks, diagnostics, paired scores, costs, and disposition. An extension that fails its diagnostic or resource gate remains documented optional/rejected work.

## 10. Stage P6 — fair comparison, precision planning, and confirmation

### Scientific contract

Models receive only allowed `Panel` inputs. O supplies observed histories; C supplies labels only for its declared calibration role; U remains unavailable to training/tuning. The evaluator alone may use latent truth as a separately labeled diagnostic.

Report both:

1. **Operational performance:** all completed method attempts, including prespecified fallbacks, with qualification and failure rates.
2. **Native distribution performance:** qualified predictions, with the exact common qualified dataset set for every comparison.

Unresolved infrastructure failures make a campaign incomplete. Keep attempted datasets, completed method outcomes, qualified forecasts, and pair counts distinct.

Use mean forecasts for squared loss and medians for absolute loss. Include RMSE/MAE, native CRPS, coverage with width and interval score, spatial/temporal variograms, and fixed trajectory functionals. Define averaging over sites/leads/datasets; distinguish mean dataset RMSE from a pooled root-MSE.

### Confirmation selection mode and workload

Default to the **retuned pipeline**: execute the frozen search policy independently on each new dataset. If its measured cost exceeds available resources, a **development-frozen configuration** study is a legitimate separately named option chosen before confirmation generation. Freeze configurations using development only, declare whether global or scenario/track-specific, refit statistical parameters on each new dataset, and retain fresh predetermined fit/reservoir seeds. Specify any within-fit epoch stopping, optimization, and MCMC behavior. Re-estimate variance and joint qualification with a bounded revised pilot under the chosen mode.

A fixed-configuration study estimates performance of development-selected specifications; it cannot claim best achievable per-dataset tuning. The mode enters design identity and report titles. A switch after confirmation inspection requires a new study. Prefer resumable execution of the intended estimand over an undeclared budget-driven substitution.

### Default primary comparison family

Define exactly one named ESN variant or one frozen validation-selection procedure as the primary ESN. Choose it using development only, retain repaired scalar ridge as a separate reference, and bind comparisons to explicit registry IDs. A promoted grouped/RHS/AR model must not masquerade as scalar ridge; do not select the best ESN variant separately in each final test cell.

Before confirmation, freeze a small explicit primary set. Proposed default:

- ESN versus separable GP.
- ESN versus basis GRU.
- Each separately for A and B.
- Primary score: normalized native CRPS, with **equal fixed weights across G1–G4**.

This produces four primary macro contrasts. Report all scenario-specific results, operational errors, qualification, and costs as mandatory companion outputs. State that each native cell estimates a method-pair-specific conditional common-qualified population; macro-averaging does not remove that conditioning. A favorable macro native score does not establish better all-attempt performance or reliability.

For independent core scenarios, combine scenario-specific paired means using the fixed weights. Do not require a dataset bundle to qualify in all scenarios merely to enter one cell. Use stratified dataset-level resampling; preserve any deliberately coupled scenario clusters. Pointwise intervals are labeled pointwise. If making formal multiple-comparison superiority claims, predeclare a simultaneous/adjusted procedure such as Holm for this primary family.

### Revised development and precision

Run development at the intended confirmation geometry, horizon, fitting/tuning policy, and selected data size. Start with five independent datasets per core mechanism; expand toward ten only when variance/qualification estimates remain too weak and the budget permits. Never estimate the new study's precision solely from the old small pilot.

For a single paired cell,

\[
d_r=L_{r,\mathrm{ESN}}-L_{r,\mathrm{control}},\qquad
\widehat{\mathrm{MCSE}}(\bar d)=s_d/\sqrt{N_{\mathrm{pair}}}.
\]

For a macro contrast with independent scenario strata,

\[
\widehat{\mathrm{MCSE}}^2(\widehat\Delta)
=\sum_g w_g^2\,s_{d,g}^2/N_{\mathrm{pair},g}.
\]

Use the paired-cluster analogue when strata are coupled. Report per-scenario MCSE as well as macro MCSE: a precise macro average can conceal imprecise individual mechanisms. Fit/reservoir seeds, sites, leads, and validation origins are not independent DGP replications.

Distinguish attempted R from usable N_pair. Estimate **joint** qualification from the paired revised pilot, with uncertainty; do not multiply marginal rates or substitute R for N_pair. A planning approximation is N_pair approximately R times p_pair. Carry uncertainty in both variance and qualification through a sensitivity calculation.

Proposed precision policy: normalized MCSE target **0.02**, with **20–60 attempted independent datasets per core mechanism**. Plan for the maximum requirement across the primary contrasts. Require at least five common qualified development pairs for each constituent primary cell before relying on its variance, and assess sensitivity because five remains imprecise. This is a proposed stricter design gate, not a general theorem.

The formula s²/epsilon² is an MCSE calculation, not hypothesis-test power. If the cap cannot attain the target, report the projected shortfall. Freeze attempted R; do not keep generating datasets until enough favorable/qualified outcomes appear.

### Freeze and run

Freeze source/runtime identities, resolved model roster/search rules, all seeds and seed namespaces, DGPs, data/sensor geometry, splits, priors, diagnostic/rescue budgets, scoring/normalization/path conventions, comparison weights, R, and failure rules. Revalidate G3 time discretization at the actual study settings and preserve its finite semidiscrete target; make no unsupported continuum claim.

Generate final-seed data only after freeze. Execute the sealed design with resumable infrastructure. Do not alter it after seeing final results. A source or scientific-policy change starts a new declared study. Negative or imprecise findings remain in the report.

## 11. Stage P7 — improve efficiency where measurements justify it

Several changes in this section can be developed earlier; apply them before large runs once verified.

### Forecast ownership and batching

Retain safe public defensive copying, but add a private validated owned-array construction route for internal producers where ownership is certain. Arrays exposed to users remain immutable and cannot alias mutable caller buffers.

Use the existing global path indexing and `iter_predictive` for bounded output. Avoid collecting every batch into an in-memory list. Use a numeric memory map or explicit batch consumer if measurements justify it.

Each path must remain a coherent whole-field trajectory, with its chosen posterior/ensemble component fixed as required. Spatially independent tiling changes the forecast law. Marginal scoring may process target/lead chunks after joint simulation.

Exact empirical CRPS and quantiles need the relevant sample axis. Preserve it per bounded chunk, use disk-backed samples, or explicitly define and validate an approximation. Do not claim that running sums alone recover exact quantiles/CRPS.

For float64 draws, one component needs 8MHn bytes: 256 paths by 100 leads by 505 sites is about 98.6 MiB before other arrays. Account for every retained component, duplicate, and workspace.

### Accuracy and performance gates

- Same-runtime batch concatenation reproduces path values and global IDs.
- Scores and means agree under declared numerical tolerances; changed floating-point reduction order need not be bitwise identical.
- Joint spatial/temporal scores preserve dependence.
- Serialization preserves target order, units, shapes, and metadata.
- An isolated representative workload demonstrates lower peak resident memory.
- Cache identities include all numerical inputs; share geometry only when semantics agree.
- Existing dense/reference and supported sparse numerical tests continue to pass.

Check 128/256/512 predictive-path stability on fixed qualified fits before increasing every run. The empirical CRPS is exact for its empirical forecast; finite-M Monte Carlo approximation to a continuous law is a separate issue. Do not apply an iid draw correction to a fully enumerated E-QESN distribution.

Keep the fast separable sufficient-statistic algebra and the measured checkpoint optimization. Do not spend engineering time on summary-writing micro-optimizations while tuning dominates. Accuracy-versus-total-compute plots are required; equal trial counts are not described as equal compute. An equal-compute experiment is optional and separately frozen.

## 12. Stage P8 — tests, API, documentation, reports, and publication

### Testing strategy

During editing, run focused tests addressing the changed risk. Before the release, run the complete applicable numerical/API suite, optional CPU-neural tests, installed-wheel tests, packaging checks, and CLI smoke workflow.

Include independent references rather than a second copy of the same algebra. Keep expensive convergence and performance checks outside the routine tiny unit suite, with executable profiles and receipts. Do not turn a noisy speed threshold into a flaky universal CI assertion.

Maintain core-without-Torch coverage and the locked full CPU-neural gate. Preserve legacy artifacts/serialization or document and test an explicit migration. Validate failure messages, unsupported combinations, shapes/units, numerical dtype, source stability, and information boundaries.

No fixed old test-count target applies: report the actual final numbers, skipped optional scopes, warnings, runtime, and commit. A configured workflow is not evidence of remote execution.

### Smooth API

Extend the existing CLI with narrowly scoped validate/resolve, resource-plan/dry-run, status, and explicit resume capabilities as needed. These are **proposed interfaces until implemented**. Keep a single source of truth for configuration validation and resolution.

Document a short sequence to:

1. Install the locked environment.
2. Validate/resolve a profile and inspect its candidates/budget.
3. Run or resume a small example.
4. Inspect job status and diagnostic failures.
5. Run a frozen study.
6. Render from saved artifacts.
7. Verify/package the release.

Examples must execute in a fresh environment. A notebook may illustrate the API, but must not be required to reproduce results.

Existing commands to preserve include:

~~~bash
bstesn benchmark registry
# Core-only execution smoke:
bstesn benchmark run --profile smoke \
  --models persistence,seasonal_naive,basis_lgssm,separable_stgp,basis_krr,eqesn_rollout \
  --scenarios g1_gaussian_ar --output results/v03/smoke_core
# Full smoke and release gate require the CPU-neural dependency environment:
bstesn benchmark run --profile smoke --output results/v03/smoke
python tools/release_check.py --with-neural --fresh-venv /tmp/bstesn-v03-release
~~~

Inspect current --help before composing new invocations. Add exact commands for any newly implemented flags; do not present planned flags as currently available.

### Documentation and derivations

Update the model hierarchy, every changed prior/conditional/MH target, parameter transformations/Jacobians, algorithm pseudocode, and inference/backend complexity. Distinguish exact computation, bounded reference computation, and approximation.

Update benchmark capability labels, search definitions, seed policy, fitting gates, normalization, estimands, resource scope, and limitations. Describe basis GRU, diffusion GRU, and E-QESN adaptations accurately. Do not imply a named-paper reproduction without verifying it.

### Reports and synchronized animations

Create a polished HTML report and an exportable PDF from saved, verified numeric artifacts. Include:

- Model hierarchy and implemented variants.
- Experiment design and information budgets.
- Candidate counts, selected configurations, qualification and failures.
- Paired scores/intervals and exact populations.
- Native uncertainty, calibration, joint dependence, and lead-wise errors.
- Learning curves and accuracy-versus-time/memory comparisons.
- The G1 known-parameter reference, clearly labeled as privileged parameters with the same noisy observation history.
- Reproducibility, negative findings, scope, and release provenance.

For predeclared demonstration datasets, produce:

1. True noisy observation evolution.
2. Truth and model forecasts side by side.
3. Synchronized signed-error fields.
4. Lead-wise error evolution, with comparable models on the same targets.

Use fixed cross-model/time color scales, common masks, explicit units, clear titles, readable legends, color-accessible palettes, and identical frame timing. Gray unscored cells; do not invent missing truth or smooth it into scored data. Keep failed-native/diagnostic labels visible.

Select demonstration replication IDs before viewing final errors. Report aggregate replicated results independently of the illustrative animation. Optional representative/failure-case selections require an explicit selection rule and label.

Retain PNG/SVG figures, MP4 and a practical fallback, HTML/PDF, machine-readable score tables, input/output hashes, and rendering commands. Check first/middle/last frames, duration, dimensions, target alignment, fixed scales, numerical score agreement, and PDF clipping/fonts. Renderers must not refit models.

### GitHub and release procedure

Use the already functioning authenticated repository connection. Verify the remote head before writing. The repository/history bootstrap is finished; do not repeat it or repeatedly attempt a known unavailable CLI credential path.

Make coherent commits with expected-parent checks and non-force updates. If the remote moved, inspect and reconcile. Respect branch protection. Keep ordinary CI at contents-read; grant publishing jobs only their needed repository-local writes.

Workflow changes can require a separate Workflows permission. A release target that changes workflow files relative to the default branch can also require that capability. Authorization does not create a missing provider scope. Diagnose actual 403/404 responses; do not disable checks or request global write-all permissions.

A push performed using a workflow's GITHUB_TOKEN does not generally trigger another push workflow. Use an available permitted dispatch or normal supported trigger, then verify CI on the actual commit. Check current official GitHub documentation when executing these operations.

Finalize implementation, configuration, and version before frozen experiments. Later report/document commits must preserve the experiments' source fingerprints. Rerun release validation when its hashed inputs change; do not relabel old receipts as validation of new code.

Publish a new version only after the applicable gates. Preserve existing tags. Upload large results, reports, animations, dependency/runtime receipts, and SHA-256 inventories as release assets. Keep the source history readable. Verify the remote commit/tag, release status, and asset checksums/metadata after publication.

If a genuine permission block remains, deliver the completed local package, exact blocked action, and minimal needed capability. Do not claim publication or remote CI succeeded until observed.

## 13. Requirement and acceptance matrix

| ID | Mandatory resolution | Minimum acceptance evidence |
|---|---|---|
| R01 | Baseline and provenance | Starting source, environment, resources, existing-check receipt |
| R02 | Design/computation/execution identities | Mutable scheduling resumes; scientific changes invalidate |
| R03 | Honest job states and bounded retries | Worker-loss fault injection; no terminal receipts for unattempted work |
| R04 | Phase-aware ESN resume | Same-runtime checkpoint/continuation equivalence |
| R05 | Exact sampler-coordinate option | Target/reference tests and bounded mixing comparison |
| R06 | Optimizer continuation | Preserved criteria and traces; honest cap failures |
| R07 | Effective candidate resolution | Nested override, uniqueness, exhaustion, rank/subset checks |
| R08 | Real basis-GRU tuning and roster | Distinct configurations and declared main capabilities |
| R09 | Qualified finalists | Full recheck/failure/replacement-policy tests |
| R10 | Temporal/spatial validation | Real H-lead stopping; edge/interior information-boundary tests |
| R11 | Fixed-endpoint learning curves | Same field/window/targets and common normalization |
| R12 | Nested sensors/representation | Common O/U roles, regularization, encoder-policy receipts |
| R13 | Extension disposition | Compatible ablation or specific evidence-backed deferral |
| R14 | Correct estimands and MCSE | Pair counts, qualification attrition, clustered uncertainty |
| R15 | Confirmation freeze | Revised evidence, primary contrasts, frozen R/source/seeds |
| R16 | Resource admission and metrics | Hard ceiling respected; scoped isolated/aggregate measurements |
| R17 | Forecast memory assessment and justified improvement | Ownership/joint-path/score invariants; measured benefit or explicit measured deferral |
| R18 | Full verification and API | Source/wheel/optional-neural/CLI/serialization checks |
| R19 | Scientific reports and media | Saved-data provenance, score checks, visual/media QA |
| R20 | Verified public delivery | Exact commit, CI, release/assets, checksums, preservation of history |

A failed scientific promotion gate does not prevent publishing useful validated engineering improvements. It does prevent claiming that a confirmation or predictive improvement was completed. Label a development-only release accordingly.

## 14. Order of execution and handoff discipline

The dependency order is:

**P0 baseline → P1 recovery/resource controls → P2 reliable fitting → P3 tuning → P4 data/capacity → P5 justified variants → P6 frozen confirmation → P8 report/release.**

P7 profiling and efficiency work runs alongside the relevant stages and must be verified before it is used by expensive experiments. Documentation and targeted tests accompany every stage.

Do not stop after restating this plan. Execute the next authorized stage, inspect its evidence, and proceed when its gate passes. Continue useful independent work when one branch is blocked. Stop optional expansion at its declared cap and retain its result.

At each durable milestone, update the progress ledger with:

- What changed and why.
- What was actually tested or run.
- Exact source, design, and computation identities.
- Diagnostic and resource outcomes.
- Promoted/rejected/deferred decisions.
- Pending tasks, recoverable states, and the exact next command.

If the session ends or context is exhausted, save a precise restart note. Do not imply that jobs continue after the environment has stopped.

The final delivery must state: implemented changes; measured scientific gains or lack of gains; qualification/failure counts; paired uncertainty; total compute and scoped memory; completed and incomplete gates; reproduction commands; and verified links to source, report, animations, and release assets.

# Reference basis

The requirements above are project-specific proposals grounded in the audit. The following sources support the general methods; they do not prove that a particular budget or extension will improve this project.

- [Pinned project audit](https://github.com/AntonioAPDL/bayes-stesn/blob/fda5ec5e4997e858f22c84b520b1b609eefa3c4e/docs/audits/post_release_2026_11_03/README.md): all reported project counts, failure mechanisms, bounded probes, and historical resource observations.
- [Vehtari et al., rank-normalized R-hat and ESS](https://avehtari.github.io/rhat_ess/rhat_ess.html): diagnostics and their interpretation.
- [Bergstra and Bengio, random hyperparameter search](https://jmlr.org/papers/v13/bergstra12a.html): a practical baseline for bounded configuration search.
- [Cawley and Talbot, model-selection overfitting](https://www.jmlr.org/papers/v11/cawley10a.html): selection uncertainty and independent performance evaluation.
- [Morris, White, and Crowther, simulation-study design](https://doi.org/10.1002/sim.8086): ADEMP, Monte Carlo precision, and transparent simulation reporting.
- [Piironen and Vehtari, regularized horseshoe](https://arxiv.org/abs/1707.01694): declared sparsity information and finite-slab regularization.
- [GitHub GITHUB_TOKEN documentation](https://docs.github.com/en/actions/concepts/security/github_token) and [release API](https://docs.github.com/en/rest/releases/releases): permissions and workflow/release behavior, checked while preparing this brief and to be rechecked when used.
