# Independent final-results audit checklist

## Status and scope

This is an audit plan, prepared from the frozen configuration, scientific source,
and reporting source before reviewing the current confirmation or sensitivity
performance results. Its checkboxes are not completed acceptance findings. The
eventual audit must record the exact files and hashes it inspected, discrepancies,
and any unavailable evidence. It must not refit, reforecast, alter selection, or
modify immutable scientific records.

The improvement experiment is validation-only and exploratory after inspection
of the original KS study's test results. Its 14 confirmation trajectories were
excluded from this improvement study's selection. They are not a new historically
untouched test set. Keep tracks A, B, and C separate.

The expected adopted experiment identity is
`067aaf08f46d7a21895ce15876d482a4a837d58627026a59e07a63db37f5c376`.
The expected freeze identity is
`96774e83e2f737ceed30abc232b3a328b79894b0173a8e7a629c0289f0c3f4d2`.
The original freeze creation time remains
`2026-11-05T10:03:57.054419+00:00`. Migration changes operational identities and
paths; it does not constitute a new scientific selection or reset the clock.

## 1. Bind the evidence before evaluating claims

- [ ] Recompute the experiment and freeze identities using their declared
  canonical hashing rules. Check `test_access=false`, immutable candidate
  configurations, source/runtime identity, cache identity, and history policy.
- [ ] Bind every consumed job, terminal, fit receipt, origin receipt, selection
  record, precision receipt, acceptance receipt, and operations record by exact
  bytes and SHA-256. Verify the report's source manifest against that census.
- [ ] Check job identity, effective candidate configuration, track, trajectory,
  model seed, stage, validation partition, schedule, and forecast identity.
  Reject ambiguous duplicate origin receipts or incompatible identities.
- [ ] Check native fit identity and fit checksum references independently of
  report labels. Do not confuse a JSON fit receipt with newly validating all
  posterior payloads. State which native payloads were actually revalidated.
- [ ] Check that report/output manifests identify the actual final snapshot;
  report generation time alone does not establish evidence freshness.

## 2. Reconstruct the frozen schedule and denominators

| Population | Trajectories | Origins | Paths | Model seeds | Expected origins per candidate/track |
| --- | --- | --- | --- | --- | --- |
| Confirmation | 1, 2, 3, 6, 7, 8, 11, 12, 13, 14, 16, 17, 18, 19 | 6000, 6100, ..., 7900 | 512 | 20261105 | 280 |
| Matched main-seed sensitivity | 0, 5, 10, 15 | 6000, 6200, 6500, 6700, 7000, 7200, 7500, 7700 | 512 | 20261105 | 32 |
| Alternate-seed sensitivity | 0, 5, 10, 15 | Same eight origins | 512 | 20261106 and 20261107 | 32 per seed; 64 alternate |

- [ ] Enumerate confirmation roles from the frozen selection: fixed-prior main,
  legacy anchor, and separately selected prior sensitivity when present.
  Deduplicate identical candidate IDs without losing their role labels.
- [ ] Enumerate matched seed rows for main and legacy anchor, deduplicating
  identical candidates. Each distinct candidate/track has 96 scheduled origins
  across all three seeds. Verify main-seed forecasts are the exact compatible
  retained promotion cases, not the 14-trajectory confirmation aggregate.
- [ ] Verify any cheap point anchors use the same confirmation trajectories,
  origins, information access, history, and physical-unit scoring. Require their
  separate artifact identity and expose missing or failed anchor cases.
- [ ] Verify training is `[0,6000)`, all current forecast target intervals are
  within validation `[6000,8000)`, history is 512, horizon is 100, and cadence is
  0.25 physical time units. The last target interval is `[7900,8000)`.
- [ ] Verify actual target-site IDs and permitted input histories against frozen
  protocol records: A uses 39 training and 13 validation sites; B uses all 64;
  C uses 40 training and 12 validation sites. Do not evaluate this study on the
  original withheld test sites or include their response histories as inputs.
- [ ] Reset state between trajectories. At first target `a`, permitted history
  is `[a-512,a)`, with adapter origin `a-1`; every forecast is autonomous through
  `a+99`. Later origins may use newly available earlier permitted observations.

## 3. Separate reliability from conditional accuracy

- [ ] For every scheduled trajectory and origin, assign exactly one of:
  finite successful forecast, observed failed rollout, inference-unqualified
  blocked origin, other failed-job blocked origin, or pending origin. Check
  these mutually exclusive counts sum to the complete schedule.
- [ ] Distinguish terminal job status from fit status and origin status. A job
  can fail after retaining finite origins; a completed native fit can have
  failed forecasts. Do not relabel blocked or absent forecasts as observed
  nonfinite rollouts.
- [ ] Count finite divergence separately: the frozen rule is any lead whose
  primary point-forecast spatial RMSE divided by its permitted training SD is
  greater than 10. Preserve all finite errors, however large, in applicable
  scores. Count nonfinite/overflow failures without clipping or replacement.
- [ ] Reconcile observed failure-plus-divergence counts over all scheduled
  origins. For unresolved outcomes show identified frequency bounds; do not
  zero-fill unknown outcomes or compute a misleading complete frequency CI.
- [ ] Reconstruct the complete-trajectory scoring population. One nonfinite or
  unavailable origin excludes that trajectory from complete-case accuracy.
  Report its ID, the number of excluded finite origins, and conditional score
  population alongside all 280 scheduled confirmation origins.
- [ ] Mark full-schedule point accuracy undefined unless every scheduled origin
  has finite output. A terminal all-failed or zero-complete-trajectory cohort
  has undefined accuracy, not zero error and not an omitted table row.
- [ ] Preserve empirical zero-event bootstrap intervals while saying "no
  observed failures/divergences"; `[0,0]` from observed clusters is not evidence
  of zero population risk on new trajectories.

## 4. Reconstruct point and probabilistic metrics independently

Use saved per-origin/per-lead quantities, not rounded report cells. For each
complete trajectory `i`, let `m_iah` denote the origin-level spatial MSE at
origin `a` and lead `h`. Reconstruct

\[
\mathrm{RMSE}_i=\sqrt{\frac{1}{|A_i|\,100}\sum_{a,h}m_{iah}},
\qquad
\mathrm{NRMSE}_i=\mathrm{RMSE}_i/s_i.
\]

The reported primary aggregate is the equal-weight mean of trajectory RMSEs,
not the mean of origin RMSEs or the square root of MSE pooled across all
trajectories. The separately named complete-case pooled RMSE may also be
reported, with its different definition explicit.

- [ ] Verify a positive constant training scale `s_i` within each trajectory,
  its permitted training-only transform identity, and physical-unit output.
  Reconstruct MAE, normalized RMSE and normalized CRPS at trajectory level.
- [ ] Reconstruct lead-wise RMSE as `sqrt(mean_origin(m_iah))` within each
  trajectory, then average trajectories. Check the full 100-lead arrays and
  reported leads 1, 5, 10, 25, 50, 100 with physical times 0.25, 1.25, 2.5,
  6.25, 12.5, 25. Record numerical comparison tolerances and maximum mismatch.
- [ ] Confirm native primary point forecasts are means of 512 complete
  predictive paths. A deterministic rollout under mean parameters is a
  separately labeled diagnostic, not a substitute predictive mean.
- [ ] Confirm CRPS uses the saved empirical ensemble V-statistic consistently,
  without claiming an iid unbiased ensemble correction for MCMC-derived paths.
  Average site/lead/origin loss within each episode before averaging episodes.
- [ ] Reconstruct 50%, 80%, 90%, and 95% marginal coverage and width. Retain
  lead-wise behavior; near-nominal pooled coverage is not evidence of uniformly
  calibrated horizons, sharpness, good point accuracy, or useful extrapolation.
- [ ] Check spatial energy-score normalization by square root of target-site
  count, the recorded 1024 path-pair scheme, and fixed variogram power 0.5,
  geometry pairs, and weights. Do not substitute a different score definition.
- [ ] Reconstruct valid prediction time from the primary point forecast:
  first lead with normalized spatial RMSE greater than 0.4; if none, report
  100 steps/25 physical units with right censoring. A crossing at lead 100 is
  distinct from a right-censored noncrossing. This is an experiment convention,
  not a distribution of individual path crossing times or a Lyapunov time.
- [ ] Keep actual native predictive distributions distinct from point-only
  anchors. Missing native CRPS remains pending/undefined, not "point only".

## 5. Verify inference and causal access evidence

- [ ] Reconcile each native receipt's qualification with its actual frozen
  diagnostic gates, monitored hyperparameters, coefficients and predictive
  functionals, chain counts, retained draws, continuations, and convergence
  extrema. Report any restricted monitoring or undefined diagnostics.
- [ ] Preserve inference-unqualified fits and their blocked-origin counts.
  Do not turn them into qualified probabilistic results or drop them from
  eligibility denominators. Passing inference gates is not predictive success.
- [ ] Confirm parameter draws remain fixed throughout complete paths, each
  path advances its own predictions, and residual innovations follow the
  accepted spatial/temporal model. Chains, reservoir seeds and predictive
  paths are separate quantities.
- [ ] Check the selected-model invariance receipt's expected cases, completed
  cases, fit checksums/native identities, fixed randomness, future-label and
  forbidden-history perturbations. Verify no refit or reselection took place.
- [ ] Check training-only feature/basis/scaler boundaries and history
  synchronization evidence. Do not describe the unchanged facade as future-free
  disk access merely because model inputs passed perturbation invariance.

## 6. Audit paired uncertainty and matched seed sensitivity

- [ ] Reconstruct trajectory-level paired differences on the exact intersection
  of complete eligible episodes. Confirm contrast direction is left minus
  right, with explicit candidate IDs, seeds and paired trajectory IDs.
- [ ] Check trajectory bootstrap settings (2000 resamples; recorded deterministic
  seed), equal trajectory weights, and pointwise horizon intervals. Site-time
  cells, origins, chains, paths and seeds are not independent replicates.
- [ ] Report conditional inference after frozen selection. A CI crossing zero
  does not support a definite ordering; lack of separation is not equivalence.
  Do not present 100 pointwise intervals as a simultaneous confidence band.
- [ ] For every alternate seed contrast, compare its four matched episodes
  against the same four main-seed episodes and eight origins. State `n<=4`,
  exact paired IDs, all 32-origin denominators per seed, and missing outcomes.
  Do not substitute the main-seed confirmation mean across 14 trajectories.
- [ ] Keep seed sensitivity separate from Bayesian uncertainty. Do not pool
  different reservoir seeds into the posterior predictive distribution.

## 7. Verify selection and interpretation

- [ ] Reconcile selected IDs and eligibility with saved screening, promotion,
  precision, and freeze receipts. No confirmation outcome may change selection.
  Keep candidate counts, effort and bounded-search limitations explicit.
- [ ] Reconstruct the declared horizon-balanced selection loss and the local
  paired predictive Monte Carlo near-tie tolerance from its fixed validation
  cases. Distinguish this simulation error estimate from posterior inference
  error, trajectory sampling uncertainty, and a global precision guarantee.
- [ ] State that fixed-prior main candidates preserve the declared scalar
  hierarchy and numerical hyperprior constants conditional on architecture.
  Changing features or spatial representation changes the induced field prior.
  Label the separately selected prior-sensitivity family and legacy alternative
  prior distinctly; do not present their constants as fixed-prior evidence.
- [ ] Treat controlled screening contrasts with only two selection trajectories
  as limited diagnostics, not the 14-trajectory confirmation comparison or a
  causal decomposition. Report actual winners, losses and unstable cases.

## 8. Reconcile migration, budget and resource evidence

- [ ] Verify the applied migration's source inventory, preserved fit and forecast
  bytes, raw source records, original attempt bytes/IDs, terminal states and
  original freeze creation time. Do not count adopted attempts twice or erase
  interrupted/retried work. Distinguish scientific records from operational
  rebindings of output/cache paths and job/experiment identities.
- [ ] Recompute cumulative worker charges from every original and new closed
  attempt plus preparation. Reconcile stage totals, current elapsed wall time,
  and fit-receipt/forecast time; these are different quantities.
- [ ] Describe the single documented prospective extension from 12 to 15 worker
  hours and 4 to 5 wall hours, retaining the original start
  `2026-11-05T07:38:04.272689+00:00`; search remains capped at 6 worker hours.
  These are admission thresholds: already admitted jobs may finish. With four
  workers and 1800-second timeouts, nominal in-flight worker overrun is at most
  7200 seconds, plus monitoring/shutdown overhead. Report actual consumption.
- [ ] Reconcile resource admission pauses, memory/file-cache pressure, measured
  fit RSS versus external monitoring, disk reserve, and remaining coverage
  limitations. A missing process observation is not measured zero memory.
- [ ] Confirm exact verified cache identity after explicit cache relocation.
  Do not attribute unproven file creators or chain discrepancies to an actor.
- [ ] Preserve six exact-bound opaque sidecars outside active fits. For the
  single reviewed inconsistent auxiliary chain/phase pair, verify native
  forecast authority remains unchanged and `inference_restart_state_available`
  is false. Do not claim the archived partial chain is an exact final RNG state.
  Strict validation remains in force for all unlisted checkpoints.

## 9. Check figures, report claims and durable handoff

- [ ] Bind the predeclared trajectory 1/origin 6000 examples to their exact saved
  arrays and fit/forecast identities. Check observed input versus withheld
  validation sites, physical units, signed errors, color scales and interval
  labels. Do not invent full-field truth or heatmaps from aggregate tables.
- [ ] Check extreme finite values remain in full-range plots and tables. Any
  readable companion panel must disclose its scope and not erase failures.
- [ ] Check animation frames come from saved fixed examples, not new forecasts;
  verify resolution, frame count, duration, codec and representative frames.
- [ ] Reconcile Markdown/HTML/PDF/CSV/JSON tables, diagnostic ranges, scientific
  conclusions, denominators, and complete/pending labels against the same
  immutable report snapshot. Inspect every PDF page and critical figure labels.
- [ ] Check artifact checksums and durable readback receipts. State the scope
  of selected-only native backup and any omitted auxiliary restart state;
  a compact metrics package is not an all-fit posterior backup.
- [ ] Deliver a final audit receipt with passed, failed and unavailable checks,
  maximum metric discrepancies, exact scope and source hashes. Keep unfinished
  scheduled work explicitly pending. Terminal failed outcomes can complete the
  census while leaving their scientific score undefined.

## Source map for the later audit

Scientific contracts: `ks_improvement_runner.py`,
`ks_improvement_candidates.py`, `ks_improvement_diagnostics.py`,
`ks_native.py`, `ks_data.py`, and `ks_metrics.py` under
`src/bayes_stesn/applications/`.

Reporting and acceptance: `tools/build_ks_improvement_report.py`,
`tools/build_ks_improvement_pdf.py`, `tools/check_ks_improvement_selected.py`,
the animation tool, and their focused tests. Native custody:
`tools/package_ks_improvement_fits.py` and
`tools/extend_ks_improvement_budget.py` with applied migration/review receipts.
The audit must bind the actual current files, not assume this checklist records
their final hashes or proves that their checks have passed.
