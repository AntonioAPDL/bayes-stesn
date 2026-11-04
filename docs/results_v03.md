# v0.3 recovered execution: findings and remaining gates

This is the **2026-11-04 current-execution record**. Strict tuning, revised
development, all six preselected calibration cases and all three paired stress
arms are complete. The admitted independent confirmation is in progress. This
record keeps its outcome, final accounting/disposition and release gates
pending; it does not close R01–R20 or replace the final generated acceptance
receipt.

The numerical package and dependency locks remain frozen at scientific source
fingerprint
`9f73b28a6ee6b8eec4bd21c3366b0c260c9433f798610be16697b2e44166d5ac`.
Recovery repeats the original declared designs, fields and seeds; replaying the
same field adds no independent replication. The [progress ledger](plans/v03_progress.md)
preserves the earlier execution snapshots, and the
[requirements ledger](plans/v03_requirements.yaml) separates implementation,
executed evidence and scientific conclusions. Generated `results/` evidence
links resolve in the restored full release payload; they are not promises that
large numerical artifacts are tracked in Git.

## Completed evidence

| Study | Observed outcome | Authority |
|---|---|---|
| Reference reliability | 60 terminal model/track outcomes; all qualified under their declared method gates | [Reference summary](../results/v03/development/reference_final/summary.json), [independent saved-result audit and G1 known-parameter reference](../results/v03/reference_audit_reproduced.json) |
| Sampler coordinates | Six terminal cases on one field each from G1, G3 and G4; legacy qualified 2/3, `ridge_total_scale` 3/3 | [Coordinate summary](../results/v03/coordinate-final/summary.json) |
| History 1000 | 165 terminal outcomes: 163 qualified and two declared fallbacks | [History-1000 summary](../results/v03/history/v03_fixed_history_history_1000/summary.json) |
| History 2000 | 165 terminal outcomes: 164 qualified and one declared fallback | [History-2000 summary](../results/v03/history/v03_fixed_history_history_2000/summary.json) |
| Nested sensors | 11/11 qualified outcomes at each of 25, 50 and 100 sensors on the same single G4 field | [25-sensor summary](../results/v03/sensors/v03_sensor_sensitivity_sensors_25/summary.json), [50-sensor summary](../results/v03/sensors/v03_sensor_sensitivity_sensors_50/summary.json), [100-sensor summary](../results/v03/sensors/v03_sensor_sensitivity_sensors_100/summary.json) |
| Bounded P5 studies | Two residual-AR ablations and two fixed-fit path-count checks completed; no automatic model promotion | [Current-source P5 findings](../results/v03/extension_findings_reproduced.json) |
| Saved mathematical/forecast diagnostics | Conditional prior moments and the G3 saved first-step/decoder replay completed without refitting | [Prior assessment](../results/v03/conditional_prior_reproduced.json), [decoder review](../results/v03/g3_saved_first_step_review_reproduced.json) |
| Strict tuning | 60 terminal outcomes: 59 qualified and one declared LGSSM fallback | [Tuning summary](../results/v03/development/tuned_final/summary.json) and complete per-job selection/recheck traces |
| Revised development | 220/220 qualified outcomes, zero fallback, complete native integrity checks | [Revised summary](../results/v03/development/revised_L1000/summary.json) |
| Separate calibration | All six preselected G4 replication-0 cases completed and sealed | [Calibration disposition](../results/v03/release_evidence/calibration_disposition.json) |
| Paired stress | Both G5 arms complete with 55/55 qualified outcomes each; corrected G6 complete with 55 replay jobs and zero parameter refits | [Paired stress summary](evidence/v03/stress_summary.json), [G6 correction equivalence](../results/v03/release_evidence/g6_schema_validation/correction_equivalence.json) |
| Confirmation admission | Valid frozen R=60 per scenario; uncapped R=102; cap shortfall retained; independent execution in progress | [Assessment](../results/v03/confirmation/assessment.json), [freeze](../results/v03/confirmation/freeze.json) |
| Revised ordinary report and fixed comparisons | All 14 cases mechanically verified; 84 sampled MP4/GIF images have actual source-bound review; all 26 ordinary PDF pages viewed | [Corrected report QA](../results/v03/rendering/revised_corrected_visual_qa.json); browser blocked, continuous playback unperformed |

The coordinate comparison concerns the same original posterior targets and
scientific functionals. One field per mechanism is a bounded development
comparison, not an estimate of a general qualification probability. The public
default remains legacy. The reference's G1 oracle has privileged known
parameters but the same permitted noisy observation history; it is separately
labeled from fitted comparators. Neither result establishes overall predictive
superiority.

## Strict tuning, revised pilot and admitted confirmation

The [strict-tuning campaign](../results/v03/development/tuned_final/summary.json)
has all 60 declared outcomes, 59 qualified fits and one declared point fallback.
For G3/rep000/A LGSSM, all four finite finalist rechecks reached 200 EM
iterations without satisfying the fixed relative-likelihood tolerance of
1e-5. The [selection failure](../results/v03/development/tuned_final/g3_excitable_field/rep_000/A/basis_lgssm/selection.json)
and [fallback receipt](../results/v03/development/tuned_final/g3_excitable_field/rep_000/A/basis_lgssm/receipt.json)
are retained. This is a bounded method failure under the original candidate,
budget and convergence rules; no favorable replacement or enlarged allowance
was introduced. Three interrupted selection scopes have incomplete accounting,
which is distinct from incomplete scientific terminal receipts.

The [selected revised pilot](../results/v03/development/revised_L1000/summary.json)
completed 220/220 qualified outcomes with no fallback and complete cost
accounting on all terminal model/task receipts. It uses uncropped initial
history 1000, three validation blocks and final forecast origin 1300 on the
matched 7-by-7 core design. This differs from the fixed-endpoint history
experiment's origin 4300. Every one of the 16 primary scenario/track/comparator
cells has all five common-qualified pairs. The independent sampling units are
20 generated datasets, five in each of four equally weighted scenarios.

The [saved assessment](../results/v03/confirmation/assessment.json) reports the
following normalized native-CRPS macros. Positive values mean larger ESN loss;
these development results do not establish an ESN advantage. The paired-field
bootstrap intervals are pointwise and condition on the exact common-qualified
population in each cell. The [independent arithmetic review](../results/v03/release_evidence/independent_reviews/confirmation_macro_independent_review_20261103T233957037400Z.json)
checked the stored reductions and planning quantities without generating
confirmation outcomes.

| Comparator | Track | Pilot ESN − comparator CRPS | Pilot MCSE | Pointwise 95% interval | Requested R | Planned MCSE at R=60 |
|---|---|---:|---:|---|---:|---:|
| Separable GP | A | +0.071260 | 0.036190 | [0.007275, 0.135495] | 29 | 0.013892 |
| Separable GP | B | +1.528008 | 0.066739 | [1.424844, 1.650895] | 99 | 0.025619 |
| Basis GRU | A | +0.163532 | 0.041489 | [0.094460, 0.241628] | 39 | 0.015927 |
| Basis GRU | B | +1.461655 | 0.067847 | [1.348315, 1.582651] | 102 | 0.026045 |

The unchanged confirmation authority admitted
[the frozen design](../results/v03/confirmation/freeze.json), identity
`9b4b784c3c6447566986910148af6b6778510f48bfd0b0acd04896a380e84065`.
It fixes **60 independent datasets per scenario**, 240 datasets total and
2640 model/task jobs, with independent final seeds. The uncapped maximum
request is 102 per scenario, so `mcse_target_not_met_at_cap` remains explicit.
The planned Track B MCSEs exceed the target 0.02. These are approximate
qualification-adjusted MCSE projections, not hypothesis-test power or a
promise of realized precision. The independent campaign at
`results/v03/confirmation/frozen_run` is in progress; no completed confirmation
outcome or achieved MCSE is claimed at this update.

## History choice: retain 1000

The [sealed history decision](evidence/v03/history_choice.json), identity
`4b2121ba3e2bbcbbc4e2b8aa8753a36686acd6a71947759c54d5e74bb28569b7`,
selects **1000 initial fitting rows** for the revised study. The
[complete curve design](../results/v03/history/curve_design.json) fixes the
endpoint, realized field, forecast times, site roles and common training-only
normalization across lengths. It uses five paired independent fields per
G1/G3/G4 mechanism on a 7-by-7 grid with 25 observed, four calibration and
20 unobserved sites. Each field's common normalization is the sample SD from
rows 3500–3999 at the same 25 observed sites, with the declared numerical floor.
The initial windows end at row 4000; final fits
use the additional three validation blocks through row 4300, and evaluation
uses rows 4300–4399. The two executed lengths therefore supply 1300 and 2300
rows at final fitting. Learned transforms may differ within the same fixed
specification; this is not a frozen-encoder experiment.

The predeclared ESN-centered rule requires both tracks to establish a normalized
native-CRPS improvement lower bound of at least 0.02, at least three
common-qualified pairs in every G1/G3/G4 cell, no lower qualification count for
the longer history in any cell, and an all-roster pipeline-time ratio at most
2.5 with complete cost accounting. All conditions must hold to promote 2000.

Here a positive difference is **CRPS(1000) − CRPS(2000)**, so it favors the
longer history. Each macro gives G1, G3 and G4 equal weight and uses the exact
common-qualified short/long population in each scenario.

| ESN track | Macro difference | MCSE | Pointwise 95% interval | One-sided 95% lower bound | Common-qualified pairs, G1/G3/G4 | Lower bound ≥0.02 |
|---|---:|---:|---:|---:|---|---|
| A | +0.141085 | 0.044531 | [0.068923, 0.218632] | +0.077927 | 4 / 5 / 5 | Pass |
| B | +0.126062 | 0.166107 | [−0.152159, 0.405591] | −0.106575 | 4 / 5 / 5 | Fail |

These are exploratory, small-sample, stratified complete-dataset bootstrap
bounds with 5000 resamples. Sites, leads, fitting seeds and validation origins
are not independent DGP replications. The intervals are pointwise; neither
nominal coverage in this small sample nor simultaneous superiority is
established. Native comparisons condition on pairwise qualification. The
decision receipt also retains every compatible model's native CRPS, native
RMSE and all-field operational RMSE, including declared fallbacks.

The scenario means show why the positive Track B macro is not a general benefit:

| Scenario | Track A CRPS difference | Track B CRPS difference |
|---|---:|---:|
| G1 Gaussian AR | +0.427142 | +0.863514 |
| G3 excitable field | +0.009707 | −0.194510 |
| G4 existing fronts | −0.013593 | −0.290818 |

Long-history qualification is not worse in any ESN decision cell, and all pair
minimums pass. Track B nevertheless fails the required improvement bound.
Independently, the cost gate fails because one resumed history-1000 receipt
has incomplete compute accounting. The saved ratio 1.315047 and saving fraction
0.239571 are reductions of **recorded, partial pipeline fields**. They do not
establish a complete measured speedup or a complete campaign cost.

The optional 500/4000 endpoints remain unexecuted. They are not necessary to
apply the already sealed 1000/2000 decision, cannot restore lost timing, and
would reuse the declared fields rather than add independent fields. The curve
shape outside these two executed lengths remains unresolved: there is no
saturation, no-benefit or no-harm-from-old-history conclusion. A 4000-row
extension would also require fresh training-only admission for its 4300-row
final fitting window. Additional field replication, separately frozen encoder
comparisons and capacity/prior ablations remain distinct possible studies.

## Sensors and representation: one-field diagnostics

The [sensor design](../results/v03/sensors/curve_design.json) uses fixed initial
history 2000, a 15-by-15 grid, nested 25/50/100 observed sites, ten reserved
calibration sites and 115 fixed unobserved sites. Track A scores the common
25-site observed subset; Track B scores the fixed unobserved set. The later
selection of history 1000 for revised development does not change this already
declared sensor study. Its geometry differs from the history curve, so a
comparison between those studies does not isolate sensor count.

The [representation audit](evidence/v03/sensor_representation.json), identity
`6027236d04bdbd07681945c2fdba9831530d0c55722436d7293ac69f6a1cfcf6`,
records the following saved ESN design properties:

| Sensors | Track A spatial rank / columns | Track B spatial rank / columns | Track A mean noise amplification | Track B mean noise amplification |
|---|---:|---:|---:|---:|
| 25 | 19 / 19 | 25 / 28 | 0.276749 | 0.480117 |
| 50 | 19 / 19 | 28 / 28 | 0.138092 | 0.251804 |
| 100 | 19 / 19 | 28 / 28 | 0.072617 | 0.138552 |

Noise amplification here is a **geometry-only normalized-ridge reconstruction
diagnostic** at the fixed 115-site unobserved set, under unit iid observation
noise and ridge 0.01, including the intercept. The diagnostic fixes the
100-sensor basis transformation separately within each track. Actual fits keep
their original sensor-specific basis normalization and priors. These numbers
are not correlated-residual forecast variances or forecast scores.

Track B's 25-by-28 spatial design cannot have full column rank; at 50 and 100
sensors its saved spatial design has rank 28. All six temporal designs have
full column rank, with 89 columns in A and 101 in B, 80 conditioning rows and
2220 likelihood rows from 2300 supplied rows. Full rank alone does not establish
stable extrapolation, a sufficient decoder span or predictive improvement.
The audit retains singular values and condition measures rather than treating
rank as a binary adequacy result.

All 33 model/track outcomes qualified, but the experiment has only **one
independent field**. Across-field MCSE is unavailable. The observed forecast
differences and representation sensitivity are descriptive; they do not
establish a replicated sensor benefit or a general accuracy ranking. No
history-by-sensors-by-capacity-by-prior factorial or frozen-encoder ablation
was performed.

## Bounded extensions, priors and the G3 decoder

The [current-source P5 aggregation](../results/v03/extension_findings_reproduced.json)
keeps the scalar-ridge ESN as the primary model. On the single G4 Track A field,
both scalar and complete-residual AR fits qualified: native CRPS was 0.165096
and 0.165848 respectively, an AR-minus-scalar difference of +0.000752. On G3,
the scalar reference qualified after its allowed continuation, but residual
AR remained unqualified at the maximum budget. There is no native G3 AR loss
to rank. Failure of inference is distinct from worse forecasting, and one
common-qualified field provides no across-dataset MCSE or confirmation.

The predeclared 128/256/512 path-sensitivity rule selected **512** for later
studies because the smaller prefixes did not meet the joint criteria in both
development cases. The reference ceiling of 512 passes by construction when
compared with itself. It does not establish negligible predictive Monte Carlo
error or model adequacy; the G3 RMS conditional predictive-mean MCSE at 512 is
about 0.206963 training standard deviations, excluding posterior MCMC error.
The already completed residual-AR comparisons retain their original 128 paths.

The [conditional prior assessment](../results/v03/conditional_prior_reproduced.json)
checks first and second moments with the saved encoder, basis, historical
features and response standardization held fixed. The shared iid and stationary
AR priors have the same pointwise amplitude and different temporal dependence.
The saved inverse-gamma shape of two gives finite variance but an infinite
fourth moment for a scalar predictive observation about its frozen center.
Zero odd-lag covariance after averaging over the symmetric rho prior does not
remove dependence in individual draws. No autonomous prior rollouts, prior
retuning, posterior refit or prior-adequacy/coverage gate was performed.

The [saved G3 first-step review](../results/v03/g3_saved_first_step_review_reproduced.json)
reconstructs the original first 64 path draws and their features exactly. It
finds no off-by-one, site-order or response-scale discrepancy within those
checks. For that reference Track A fit, the saved decoder has rank 12 at 25
observed sites. Projection of the first noisy observed target outside its
span contributes about 98.93% of the 64-draw conditional-readout mean's squared
error; the associated projection RMSE is 0.800102. This evaluator-only
projection is a bound for that particular target and span, not a usable
forecast, a latent-field result or a universal irreducible error. The replay
does not diagnose Track B interpolation or demonstrate that a larger decoder
would improve forecasting. Growing saved-path dispersion does not separate
long-lead parameter and innovation contributions.

Grouped ridge was not triggered by the declared training-inspection threshold;
group/column horseshoe ablations lack the required sparsity and prior-scale
evidence. Other covariance, dynamics, tail and neural-training variants retain
their individual P5 rationales. No extension is promoted from implementation
availability or descriptive residual summaries. The separate base-model G5/G6
stress and all eligible preselected calibration cases are now complete, as
described below. Their completion does not promote richer variants. A final
disposition overlay is pending at
`results/v03/release_evidence/extension_dispositions.json`; it must retain the
immutable P5 findings and bind the actual later evidence.

## Separate calibration: six cases on one field

The [six-case disposition](../results/v03/release_evidence/calibration_disposition.json)
verifies all preselected ESN, GP and GRU A/B cases as completed and sealed.
They share **one G4 replication-0 field**, not six independent fields. Parameters
remain fixed after row 1300. Each of 25 calibration forecasts uses only the
allowed observed history available at its own origin; the common later
100-step evaluation starts at row 3800. Track A evaluates 25 observed sites.
Track B learns its map from reserved calibration-site labels and evaluates
20 unobserved sites; evaluation labels never train the map.

The [source-bound interpretation record](../results/v03/release_evidence/scientific_interpretation/paired_stress_calibration_review.json)
retains all raw and calibrated levels, targets, widths, interval scores and
marginal WIS. At nominal 90%, the marginal maps give the following descriptive
results on that single later block. Width and interval-score units are the
original response units; a smaller interval score is better.

| Model | Track | Raw 90% coverage | Calibrated marginal 90% coverage | Raw width | Calibrated width | Raw interval score | Calibrated interval score |
|---|---|---:|---:|---:|---:|---:|---:|
| Bayesian ESN | A | 88.00% | 91.88% | 0.720333 | 0.852390 | 1.014265 | 0.995619 |
| Bayesian ESN | B | 92.70% | 96.90% | 2.941989 | 3.284739 | 3.594129 | 3.372750 |
| Separable GP | A | 89.64% | 92.88% | 1.168804 | 1.378428 | 1.549671 | 1.517957 |
| Separable GP | B | 90.10% | 97.45% | 1.236558 | 1.967105 | 1.550296 | 2.055155 |
| Basis GRU | A | 68.84% | 89.92% | 0.727535 | 1.156507 | 1.841698 | 1.361791 |
| Basis GRU | B | 68.05% | 97.05% | 0.664316 | 1.646646 | 1.827064 | 1.744437 |

The GRU examples show substantial widening and improved empirical marginal
coverage; this is a within-block description. Calibration does not uniformly
improve interval score: GP Track B increases from 1.550296 to 2.055155.
Horizon and simultaneous maps answer different coverage questions and are
reported separately. All six simultaneous 95% bands cover the one later path,
but widths can be large: ESN Track B has mean width 18.659978 versus raw
marginal width 3.479684. One covered path is not evidence of a 95% path-coverage
probability. The 25 serial blocks remain dependent, and C-to-U transfer is
empirical only; no distribution-free time-series or spatial-transfer guarantee
is established. All raw predictive samples remain unchanged. The maps add no
predictive distribution, so raw CRPS remains the original distribution's score;
a calibrated joint-distribution CRPS is not defined by these band maps.

## Paired G5/G6 stress: five independent fields per arm

The [complete paired stress audit](evidence/v03/stress_summary.json) retains
both 55-job G5 arms and the complete 55-job G6 replay. Every primary model/track
cell has five common-qualified paired fields. Each reported contrast is
**stress minus its own control**, normalized by that pair's control
training-only standard deviation. Positive values mean larger loss. The G5
arms share their G4 controls; G6 uses G2 controls. The arm estimates remain
separate and are not pooled as independent scenario strata. The G5
arms refit the prescribed Gaussian models with frozen scenario-specific seeds;
G5 fitting seeds match between its two arms but differ from G4 control seeds.
Their contrast concerns the full declared fitting procedure. G6 alters only
the evaluation future after lead 25, using 44 frozen-model forecasts, 11 saved
forecast reuses and **zero parameter refits**.

| Arm | Model | Track | Normalized CRPS change | MCSE | Pointwise 95% interval |
|---|---|---|---:|---:|---|
| Whole-field t | Bayesian ESN | A | +0.013709 | 0.014424 | [-0.009917, 0.037775] |
| Whole-field t | Bayesian ESN | B | -0.069880 | 0.030169 | [-0.123325, -0.016435] |
| Whole-field t | Separable GP | A | +0.000778 | 0.000860 | [-0.000867, 0.002040] |
| Whole-field t | Separable GP | B | -0.002926 | 0.002866 | [-0.007992, 0.002139] |
| Whole-field t | Basis GRU | A | -0.017711 | 0.030083 | [-0.079562, 0.023914] |
| Whole-field t | Basis GRU | B | -0.002475 | 0.003702 | [-0.008412, 0.004008] |
| Local t | Bayesian ESN | A | -0.006464 | 0.010748 | [-0.027094, 0.011005] |
| Local t | Bayesian ESN | B | -0.051897 | 0.041885 | [-0.115267, 0.023217] |
| Local t | Separable GP | A | -0.000135 | 0.000662 | [-0.001310, 0.001039] |
| Local t | Separable GP | B | -0.002269 | 0.002897 | [-0.007581, 0.002453] |
| Local t | Basis GRU | A | -0.028258 | 0.028849 | [-0.086833, 0.015158] |
| Local t | Basis GRU | B | -0.001394 | 0.005733 | [-0.011067, 0.008785] |
| Velocity shift | Bayesian ESN | A | +0.011085 | 0.011183 | [-0.006573, 0.032356] |
| Velocity shift | Bayesian ESN | B | +0.037132 | 0.014409 | [0.009330, 0.063407] |
| Velocity shift | Separable GP | A | +0.012818 | 0.023281 | [-0.029292, 0.050277] |
| Velocity shift | Separable GP | B | +0.050026 | 0.019535 | [0.018058, 0.083300] |
| Velocity shift | Basis GRU | A | -0.001463 | 0.021680 | [-0.040115, 0.034553] |
| Velocity shift | Basis GRU | B | +0.053940 | 0.022138 | [0.019925, 0.096881] |

All intervals are pointwise percentile intervals from 5000 paired-field
bootstrap resamples, with only five independent fields per arm. The
[arithmetic review](../results/v03/release_evidence/scientific_interpretation/paired_stress_calibration_review.json)
retains full precision, pair IDs, all 18 primary CRPS and 18 RMSE contrasts and
144 independent arithmetic checks. The later
[serialization binding](../results/v03/release_evidence/stress_serialization_current/interpretation_binding_compact.json)
verifies that whitespace-only compaction preserves the complete parsed stress
audit, all estimates, intervals, pair IDs and its scientific identity. It
resolves the historical interpretation's original file hash to preserved exact
bytes; the [fresh strict verification](../results/v03/release_evidence/stress_serialization_current/stress_renderer_verification_compact.json)
binds the current compact file and all 1286 native input entries through the
unchanged renderer. No scores or bootstrap samples were recomputed.
Whole-field t gives a negative ESN Track B
contrast in these fields; every other primary whole-field CRPS interval and
every local-t primary interval includes zero. This does not establish general
robustness, equivalence or absence of an adverse effect. Under the blind
velocity shift, all three Track B primary intervals lie above zero; all
Track A intervals include zero. These within-model effects do not establish
a between-model robustness ranking or a simultaneous superiority result.

### Preserved G6 validator correction

The first replay attempt stopped after five completed jobs because the saved
E-QESN training state uses its native schema key. The original design at
`results/v03/stress/design/stress_design.json` and failed-attempt tree at
`results/v03/stress/g6_velocity_shift/` remain preserved. Commit `99b0f8d`
corrects strict saved-state validation without changing scientific package,
seeds, fitting budgets, source cutoff, frozen renderer or confirmation assessor.
The [equivalence receipt](../results/v03/release_evidence/g6_schema_validation/correction_equivalence.json)
records byte-identical G5 YAMLs and the same saved fits and scientific settings.
The completed recovered run uses
[`design_schema_fix/stress_design.json`](../results/v03/stress/design_schema_fix/stress_design.json)
and
[`g6_velocity_shift_schema_fix/replay_summary.json`](../results/v03/stress/g6_velocity_shift_schema_fix/replay_summary.json).
Replaying those same five fields does not add independent replication. See the
[stress restoration note](../configs/benchmark/v03/studies/stress_README.md#recovered-v03-release-artifacts)
for the distinction between these completed inputs and fresh-run recipes.

## Compute and memory scope

The [first interruption](../results/v03/recovery_preflight/session_interruption_20261103T2104Z.json)
and [second interruption](../results/v03/recovery_preflight/session_interruption_20261103T2220Z.json)
record lost execution sessions without assigning an unverified cause. The
history-1000 G3/rep004/B ESN and sensor-100 G4/rep000/B ESN receipts explicitly
flag incomplete accounting. The original outer history-1000 measurement and
the first sensor-100 chunk's terminal envelope were not recovered. Interrupted
selection work likewise retains its actual incomplete/unknown scope. Across
the completed pre-confirmation core studies there are six incomplete
model/task accounting scopes: one history-1000 ESN, one sensor-100 ESN, three
tuning selections (G1/A ESN, G2/A DCRNN, G3/A DCRNN), and one G5 whole-field-t
ESN at replication 2/A. The tuning final-fit receipts do not label those three
fits resumed; the missing scope is selection accounting. Revised development
has 220 complete cost receipts, and local-t G5 has 55. A successful final
scientific receipt does not repair an earlier lost measurement envelope.

The [initial ordinary-report rendering attempt](../results/v03/rendering/revised_render_interruption_20261103T2334Z.json)
also lacks its terminal measurement envelope. Its partial output, log and plan
remain preserved separately from the successful continuation into the fresh
canonical report directory and the later measured presentation correction.
The interrupted render's unknown CPU/elapsed scope is not replaced by the
successful rerender's measurements, and no interruption cause is inferred.

Finite fields on an incomplete receipt can be partial timings or elapsed
estimates that include downtime. They are neither exact compute totals nor
necessarily lower bounds. A final companion at
`results/v03/release_evidence/execution_costs.json` is pending until all required
phases and monitor windows are closed. It must retain complete-receipt subtotals,
partial/estimated fields and unavailable totals separately. Its role is to
account for the evidence that exists; lost work does not become measured merely
because the scientific outcomes later finish.

Inclusive command/chunk envelopes and nested model pipelines are overlapping
views and must not be added. Diagnostic time already charged inside a fit must
not be counted again. Cache verification/copy work is separate from the
original field generation, whose interrupted source envelope is incomplete.
Reused stress controls are charged once. Generation, controller overhead and
lost CPU time remain unavailable where they were not isolated. Summed job wall
time is not elapsed time of a parallel campaign.

Shared-cgroup monitor peaks describe their retained observation windows and
can include other processes and cache. Interrupted windows are not continuous
coverage of the campaign. Ordinary fit RSS fields are process-lifetime high
water marks, not isolated peaks for each fit. The
[forecast-ownership experiment](evidence/v03/forecast_ownership.json) measures
one avoided complete array copy on its declared workload; it does not establish
an all-fit or all-scoring memory reduction.

## Revised ordinary report and actual QA

The completed ordinary report is
[`report/v03/revised_media_complete`](../report/v03/revised_media_complete/report.html),
with its [26-page PDF](../report/v03/revised_media_complete/benchmark_report.pdf).
The two fixed three-model G4 comparisons are at
[`revised_comparison`](../report/v03/revised_comparison/manifest.json) and
[`sensor100_comparison`](../report/v03/sensor100_comparison/manifest.json).
They illustrate preselected replication 0 and retain native qualification,
fixed scales, target masks and large unfavorable ESN errors; they are not
aggregate rankings.

Actual review found four forecast-legend contacts and a small G4/A right-edge
tick-label clip. The [approved presentation correction](../results/v03/rendering/revised_forecast_layout_correction.json)
regenerated only those four PNG/SVG/MP4/GIF sets, replaced their exact embedded
HTML images, and rebuilt the PDF through the unchanged font helper. It changed
18 payloads, left 54 payloads byte-identical, retained numerical artists and
axis limits, and verified all 69 bound scientific/presentation input files.
The original report, original manifest and earlier QA remain preserved whole.
No model fit or scientific forecast was made by this correction.

The [new combined QA receipt](../results/v03/rendering/revised_corrected_visual_qa.json)
records all 14 mechanical cases passing, 24 newly viewed corrected samples and
60 previously viewed samples reused only by identical encoded-asset and
extracted-PNG hashes. Every one of the 26 corrected PDF pages was actually
viewed and every used font is embedded. The
[HTML structural check](../results/v03/rendering/revised_corrected_html_structural_checks.json)
verifies 22 embedded PNGs, 12 local videos and no scripts or external references.
This closes the stated saved-artifact and sampled-frame review. Continuous
playback was not performed; wide/narrow browser review remains
[blocked by environment policy](../results/v03/rendering/browser_preview_policy_block.json).
Static checks do not replace either activity, and this ordinary-report receipt
does not claim final synthesis or confirmation-report QA.

## Remaining decisions and delivery gates

The following remaining destinations are authoritative only when their own
completed, validated receipts exist. An input directory or configured workflow
does not establish a passed gate.

| Pending scope at this update | Canonical authority |
|---|---|
| Independent confirmation completion and realized common-qualified precision | `results/v03/confirmation/frozen_run/summary.json`, its per-job receipts and completed final comparison; bound to the existing assessment/freeze |
| Closed execution-cost and extension dispositions | `results/v03/release_evidence/execution_costs.json`, `results/v03/release_evidence/extension_dispositions.json` |
| Final covered-source/installed-wheel gate | `results/release_manifest.json`, `results/validation/source_pytest.xml`, `results/validation/installed_pytest.xml` bound to the final covered bytes |
| Final saved-artifact synthesis, any confirmation report and their actual QA | `report/v03/rendered/report_manifest.json`, `results/v03/release_evidence/inspection_records/final_media_pdf_qa.json` |
| R01–R20 closure and verified public delivery | `results/v03/release_evidence/r01_r20_closure.json` and the separate post-upload publication-verification sidecar |

An admitted freeze, a completed confirmation and achieved precision remain
different outcomes. No confirmation result, achieved MCSE target or
simultaneous model ranking is claimed here. The long-history
[G3 numerical validation](evidence/v03/generator_validation_long_history_grid64.json)
supports the declared finite semidiscrete time/activity checks; its failed mesh
refinement does not support a continuum claim.

Use the [ordered reproduction guide](reproduction_v03.md),
[saved-artifact report guide](../report/v03/README.md) and
[delivery procedure](release_v03.md) for the remaining commands. Complete all
covered factual edits before the final gate. Subsequent generated closure,
rendering and publication receipts must preserve those covered source bytes
or require a new gate.
