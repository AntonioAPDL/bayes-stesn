# KS improvement study: literature and search-design review

## Scope and status

This is a design review for the exploratory improvement branch based on
`2ecd8052ce2a44c9b4ed3862382594bd84598ad7`. It does not claim that a proposed
candidate has run or improved a score. The generated configuration, resource
receipt, execution ledger, and selection receipts are authoritative for actual
execution. Preserve the previous experiment and its conclusions as a separate
snapshot.

The original test results have already been inspected. The present revision is
therefore an **exploratory improvement study**, even when all new selection uses
validation data. A newly scored subset of validation trajectories offers an
additional check of selection stability; it does not create a new independent
test experiment retrospectively.

## What the primary literature supports

| Source | Verified methodological point | Consequence for this experiment |
|---|---|---|
| [Pathak et al. (2018)](https://link.aps.org/accepted/10.1103/PhysRevLett.120.024102), with [authors' code](https://github.com/jdppthk/parallelized-reservoir-computing) | The reservoir uses a sparse recurrent matrix scaled by eigenvalue radius and an output map containing reservoir states and their elementwise squares. Their scalable construction uses local parallel reservoirs. | Retain nonlinear readout terms and inspect recurrence strength. This motivates a comparator and possible extensions; it does not make our global Bayesian spatial model an exact reproduction. |
| [Vlachas et al. (2020)](https://arxiv.org/html/1910.05266v2), especially Section 2.5, Appendix A, and Table 7; [authors' code](https://github.com/pvlachas/RNN-RC-Chaos) | Their parallel KS study considers 500–12,000 nodes per reservoir, radius 0.6, input scaling 1, degree 10, 100,000 training states, and a 2,000-step testing warmup. Their reduced-observation experiments also expose autonomous instability. Appendix A accumulates regression statistics in blocks. | These settings are not optima for our domain, cadence, spatial panel, Bayesian hierarchy, or data budget. They support examining capacity, input scaling, synchronization and autonomous stability, while exploiting sufficient statistics. |
| [Jiang and Lai (2019)](https://doi.org/10.1103/PhysRevResearch.1.033056) | Favorable ranges of reservoir spectral radius depend on the predicted system and network construction. | A single fixed recurrence control cannot establish adequate tuning. The native model's **operator norm** must still be distinguished from eigenvalue radius. |
| [Gauthier et al. (2021)](https://arxiv.org/abs/2106.07688), with [authors' code](https://github.com/quantinfo/ng-rc-paper-code) | Nonlinear delay regression combines delayed coordinates and polynomial features with a regularized linear readout. The supplied examples concern other dynamical systems. | Keep POD-NVAR as an explicitly adapted comparison. Its quadratic feature count, conditioning, and autonomous stability require their own checks. |
| [Zhang et al. (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12221348/) | For NGRC, a better fitted flow map can coexist with unstable autonomous dynamics; delayed coordinates and regularization are involved in the studied mechanism. | Low one-step regression error is insufficient evidence of good recursive forecasting. This paper does not establish that the same mechanism explains the Bayesian KS result. |
| [Bergstra and Bengio (2012)](https://www.jmlr.org/papers/v13/bergstra12a.html) | Random search can allocate a finite search budget more effectively than a Cartesian grid when only some controls matter strongly. | Combine controlled contrasts with reproducible space-filling proposals rather than multiplying every range into a grid. |
| [Li et al. (2018)](https://www.jmlr.org/papers/v18/16-558.html) | Hyperband allocates increasing resources to promising configurations. | Use staged evaluation on more trajectories and origins. This custom schedule is not an implementation of all Hyperband assumptions or guarantees. |

No additional dataset, large benchmark framework, or new deep-learning ecosystem
is required to investigate the present failure modes.

## Separate the scientific questions

The accepted main model retains its Gaussian likelihood, scalar Gaussian ridge
hierarchy, covariance family, exact collapsed inference, and recursive
observation-feedback mechanism. Changes to the architecture and its existing
feature controls are selected within that family. Include all four earlier
configurations, including each track's previous winner.

Use separate, explicitly labeled analyses for the following questions:

1. **Reservoir and feature adequacy.** Vary recurrence strength, input scale,
   leaks, connectivity, capacity, PCA dimension and lag structure. Include
   controlled contrasts that change one setting around an accepted reference.
2. **Spatial extension.** Examine spatial rank, center placement, basis width,
   and training-geometry conditioning. Diagnose performance at observed and
   withheld coordinates separately. Track C needs its own training/validation
   selection; transferring Track A's winner measures transfer robustness only.
3. **Readout and uncertainty propagation.** Compare the native posterior
   predictive distribution with matched deterministic and component-removal
   diagnostics. A fixed-parameter rollout, a parameter-only ensemble, and an
   innovation-only ensemble answer different questions. None replaces the
   native posterior predictive mean or its probabilistic scores.
4. **Prior sensitivity.** A small, separately identified prior-scale stratum
   tests dependence on shrinkage assumptions. It is not silently pooled into a
   claim about the original fixed hyperprior.

One-step validation and 100-step autonomous validation should both be reported.
For the former, each one-step origin may condition on its own permitted past;
those predictions must not be presented as a single autonomous 100-step path.
Do not remove innovations from the main model because a diagnostic point
trajectory looks better.

## Broad but finite opportunity

The generated [campaign configuration](../../configs/applications/ks_improvement_v1.json)
contains **104 candidates**: 40 for A, 40 for B, and 24 for the independently
selected contiguous track C. These are configured opportunities, not completed
fit counts. Their strata are:

| Track | Preserved-hyperprior candidates | Legacy prior references | Prior-scale sensitivity | Total |
|---|---:|---:|---:|---:|
| A | 34 | 2 | 4 | 40 |
| B | 34 | 2 | 4 | 40 |
| C | 18 | 2 | 4 | 24 |

The configuration determines the exact values. The following scientific ranges
motivated that construction and possible later bounded refinements:

| Control | Proposed coverage or constraint |
|---|---|
| Recurrent operator norm | Several values from about 0.2 to 0.99, always within the current implementation's admissible interval |
| Leak | Homogeneous slow, intermediate and fast updates, plus the accepted heterogeneous pattern |
| Input scale | Approximately 0.05–2 on the native Gaussian-weight scale |
| Connectivity | Sparse alternatives; record both probability and expected degree so changing reservoir size is interpretable |
| Reservoir capacity | Accepted 32/64/128 sizes plus 256; admit 512 only after a measured large-capacity pilot |
| PCA modes | Approximately 4–24, bounded by available input dimension and conditioning sample size |
| Lags | One, two and several lagged states; include spacing changes as distinct choices |
| Squared readout terms | None, a subset, and all states where measured cost permits |
| Spatial dictionary | Small through moderate ranks, varied widths and permitted existing kernels; audit target amplification |
| Conditioning prefix | Controlled comparisons at 128/512/1,024 where valid, retaining the original 80-state reference |

These are proposals, not published KS optima. They are not a Cartesian grid.
Holding the conditioning prefix fixed across most candidates avoids conflating
reservoir capacity, PCA estimation, feature scaling and usable target count.
Record the exact temporal dimension and spatial-by-temporal coefficient count
before admitting each candidate.

Also report the realized eigenvalue radius from the synchronization preflight.
Since the eigenvalue radius is bounded by, but need not equal, the operator norm,
this native sweep does not reproduce a conventional spectral-radius sweep.
Changing the reservoir normalization or topology beyond the accepted constraint
is a separately identified architecture extension.

Use one common model seed for selection. Do not select a favorable seed. After
selection, repeat the chosen configuration on the fixed sensitivity seeds and
report their results separately. Changing a prior while using the same random
reservoir requires the reservoir stream itself to remain fixed.

### Stages and qualification

- Initial screen: trajectories `[0,5]`, validation starts
  `[6000,6500,7000,7500]`, full permitted training data and the complete 100-step
  forecast horizon.
- Promotion: four main candidates per track, trajectories
  `[0,5,10,15,4,9]`, starts
  `[6000,6200,6500,6700,7000,7200,7500,7700]`. Evaluate prior sensitivity as a
  distinct stratum. Screening uses 256 paths and promotion uses 512.
- Predictive precision: four repeated 512-path ensembles at trajectory 0 and
  validation starts `[6000,7500]` for every eligible promoted main candidate.
  Match forecast random streams across candidates and keep each model seed
  unchanged.
- Freeze one main candidate per track, the comparison with its previous
  selected configuration, random streams, history policy, and diagnostics.
- Validate the frozen main candidate, previous selected configuration, and best
  qualified prior-sensitivity candidate on the remaining 14 trajectories and
  all 20 official validation starts. Do not use this check to choose another
  winner while continuing to call it independent confirmation.
- Repeat the main candidate and previous selection at two additional fixed
  model seeds on trajectories `[0,5,10,15]`. The driver stops at validation
  confirmation and seed sensitivity. It does not open the original test
  partition; any later evaluation there remains exploratory.

Preserve the native CRPS selection objective and report the same horizon-band
RMSE as a secondary outcome. A different point-optimal configuration can be a
separate labeled analysis, not a post hoc replacement of the CRPS winner.
Finalists need adequate predictive Monte Carlo precision; a nominal 1% loss gap
is not decisive when uncertainty from predictive simulation is of similar size.
The implemented near-tie tolerance is candidate-specific: the maximum of 1% of
the best promotion loss and twice the paired replicate standard deviation for
that candidate versus the promotion-loss winner. An unrelated noisy candidate
cannot widen another candidate's tie threshold. This is a predeclared heuristic
based on conditional simulation variation at the fixed cases, not a confidence
interval for scientific performance or a replacement for chain diagnostics.

Keep four-chain qualification and the accepted convergence checks. Reducing the
number of trajectories or scored origins is preferable to judging configurations
from deliberately unconverged posteriors. Any inexpensive point-fit or
teacher-forced proxy is diagnostic or explicitly provisional; it cannot
establish a probabilistic winner. Preserve pending and inference-unqualified
candidates instead of treating them as scientific failures.

## Why geometry requires a separate diagnostic

The implemented basis standardizes each nonconstant column using the permitted
training coordinates. A column that changes little on a contiguous observed arc
can be strongly amplified when evaluated in an unobserved gap. The code's small
standard-deviation floor prevents division by zero; it does not guarantee a
well-behaved extension.

For the accepted coefficient prior, at fixed temporal features,

\[
\operatorname{Var}\{\phi(s)^\mathsf TCz\mid\sigma^2,\lambda,z\}
=\frac{\sigma^2}{\lambda}\|\phi(s)\|^2\|z\|^2.
\]

This algebraic relation identifies an amplification mechanism, not a proof that
it alone caused the observed errors. Record training singular values, target
row norms, weak-direction target loadings, and posterior predictive variance.
These geometry checks need no forbidden responses. Positive ridge
regularization can make a solve numerically valid without ensuring that the
unseen-coordinate prediction is informative.

A predetermined periodic Fourier representation or a different normalization
may be useful extensions, but they alter the spatial representation and hence
the induced function prior. Identify them explicitly, preserve the original
model as a reference, and verify arbitrary-coordinate prediction and the seam.
Do not apply a uniform-grid convolution to the compressed irregular input sites.

## Efficiency and implementation findings

The retained early audit records qualified Bayesian fits of about 29.6–40.6
seconds and mean 100-step, 256-path forecasts of about 2.24 seconds per origin
for the old candidate sizes. These historical numbers are estimates for
planning, not measurements of the larger new configurations. With those costs,
40 candidates across two tracks and two trajectories cost roughly two worker
hours for the first screen. Larger capacities and additional inference work
must be profiled before using that estimate.

The new training-only 512-unit profile recorded approximately **64.82 seconds
for a qualified fit**, **5.53 seconds for one 100-step, 256-path forecast**, and
**537,538,560 bytes of peak RSS**. This profile motivated the explicitly recorded
extension from the initial eight-worker-hour allocation to a **twelve-worker-hour
scheduling threshold**. It is resource evidence, not validation evidence for choosing that
configuration.

The configured limits are six worker hours for preflight, screening, promotion
and precision combined, reserving six for confirmation and subsequent
diagnostics; a separate four-hour wall scheduling threshold; and at most four workers with one
numerical thread each. The per-worker admission estimate is 600 MiB, checked
with the existing 1.5 multiplier against a combined allowance of
4,531,695,001 bytes and current container headroom. The disk reserve is
12,565,873,460 bytes. These are finite limits, not a promise that every planned
stage will fit. The coordinator checks cumulative worker and wall budgets before
admitting a batch, while each running job has a 1,800-second timeout. Already
admitted work can therefore exceed a scheduling threshold by at most one such
batch: up to 7,200 worker seconds or 1,800 wall seconds, plus bounded shutdown
overhead. Report actual consumption and retain any unfinished work as pending;
do not describe these admission checks as exact running-job cutoffs.

Source review identifies these practical constraints:

- `ks_runner.numerical_identity()` requires Torch package metadata even for a
  Bayesian-only study. A new native-only driver can reuse `fit_native`,
  `KSAdapter`, scores and the ledger without installing Torch, provided its own
  runtime and executed source are bound in its identities.
- `NativeFit.iter_forecasts()` rebuilds history features per batch. Larger
  batches can remove some repeated setup when measured memory permits, but
  `simulate_predictive()` still loops over paths and leads. Profile before
  undertaking a numerical rewrite.
- Forecast generation uses stable per-path random streams. Increasing an
  ensemble from 256 to 512 can reuse the first 256 paths exactly if that data
  were retained; a score JSON alone is insufficient to reconstruct a larger
  empirical distribution without regenerating those paths.
- Per-origin forecast scores and fit checkpoints are reusable only when
  effective configuration, training data, seed, source and runtime identities
  match. The old runner hashes the entire source/runtime identity, so do not
  claim old fit reuse after changing a numerical dependency without an explicit
  compatibility check.
- At 512 paths, 100 leads and 64 targets, one float64 predictive array requires
  about 26 MB. Keeping one origin at a time is modest; accumulating all origins,
  configurations and seeds is unnecessary.
- Keep float64 solves and scores. Batch arrays and sufficient statistics, keep
  BLAS threads at one per worker, and preserve each complete path's fixed
  parameter draw and its own recursive predictions.

No broader neural-model search is needed to answer why the native Bayesian
model underperformed. The old selected comparators supply context; a matched
readout ablation and controlled reservoir/spatial contrasts address the specific
unresolved questions more directly. New model classes should be added only
after those comparisons identify a question they cannot answer.

## Relevant repository evidence

- [Original configuration](../../configs/applications/ks_experiment.json)
- [Accepted model audit](ks_native_audit.md)
- [Early timing and qualification audit](evidence/ks_validation_audit_early.json)
- [Native adapter](../../src/bayes_stesn/applications/ks_native.py)
- [Original resumable runner](../../src/bayes_stesn/applications/ks_runner.py)
- [Spatial basis](../../src/bayes_stesn/basis.py)
- [Predictive path implementation](../../src/bayes_stesn/prediction.py)

All proposed changes remain subject to forecast/feature boundary checks, seam
checks, known-answer scores, numerical qualification, and exact resume identity.
Successful checks establish the property tested; they do not guarantee a
predictive improvement.
