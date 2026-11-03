# Matched learning curves, paired precision, and forecast ownership

This document describes the schema-4 study interfaces. The changes implement
the [post-release audit](audits/post_release_2026_11_03/README.md); their presence
does not establish improved predictive performance. Numerical study results
require separately identified experiments and qualified fits.

## Common history endpoints and scale references

History length is an information intervention. All conditions must forecast
the same observations at the same times and sites. Resolve the full curve
before execution, or pass `common_endpoint` explicitly:

```python
from dataclasses import replace
from bayes_stesn.benchmark import profile
from bayes_stesn.benchmark.studies import history_curve_profiles

base = replace(profile("v03_pilot"), horizon=100)
profiles = history_curve_profiles(
    base, (500, 1000, 2000, 4000), common_endpoint=4000
)
assert {p.final_origin for p in profiles} == {4300}
assert {p.n_times for p in profiles} == {4400}
```

For each length L, initial fitting uses `[4000-L:4000)`, temporal validation
starts at 4000, 4100, and 4200, final refitting uses `[4000-L:4300)`, and testing
uses `[4300:4400)`. The supplied final history is L+300 rows; ESN likelihood
rows additionally exclude its declared conditioning prefix. Fixed specification
and repeated tuning remain different experiments. In a fixed specification
experiment, freeze suitable configurations through the protocol's
`development_frozen_configuration` selection mode before this stage.

`run_curve(..., execute_values=(1000, 2000))` executes a subset of the complete
resolved design. A later call with the same `values` and `common_endpoint` can
execute 500/4000 without moving the test period. `curve_design.json` binds all
profiles and rejects a different design in the same output directory. Execution
subsets are not part of its immutable design identity.

Schema-4 history profiles default to a common normalization window consisting
of the final `min(500, min(lengths))` initial-fitting rows. With the lengths
above this is `[3500:4000)`. All conditions use the same observed-site subset,
sample standard deviation (`ddof=1`), and numerical floor. An explicit
`normalization_reference` is validated against every condition:

```python
reference = {
    "start": 3500, "stop": 4000, "observed_count": 25,
    "ddof": 1, "floor": 1e-12,
}
```

The reference must be inside each initial supplied training window; it cannot
contain validation or final responses. The evaluator saves row bounds, site
identities, observation hash, convention, sample count, scale, and a content
identity in each final receipt. No model receives this evaluator-owned object.
Changing final/U/latent responses leaves the scale unchanged. Raw within-DGP
scores remain available alongside normalized paired losses.

For schema-3 profiles, the previous training-window population-SD convention is
retained and labeled `legacy_training_window`. New normalization policies
require schema 4. Merely loading a historical result does not upgrade its
normalization or scientific interpretation.

## Nested sensor curves

`sensor_curve_profiles(replace(base, n_side=25), (25, 50, 100))` reserves the largest O master set
before assigning C/U, keeps U fixed, and scores A on the smallest shared O set.
Inactive potential sensors do not join U. The generated field is unchanged.
Schema-4 conditions use a common training window and the same smallest
observed subset for normalization. The grid must contain enough locations for
the largest O set, reserved calibration sites, and a nonempty U population.

Historical-window and sensor experiments are separate stages. A larger plotting
grid, additional forecast paths, and additional independent DGP replications
are not substitutes for more training observations.

## Operational and native populations

`aggregate_results` preserves one terminal scientific outcome per
scenario/track/model/independent dataset. Duplicate receipt keys are rejected;
infrastructure retries do not create extra replications.

- Operational RMSE/MAE include qualified native predictions and the declared
  fallback when a method exhausts its fitting/diagnostic allowance.
- Native distribution scores use qualified native distributions. Pairwise
  CRPS uses the intersection of the two methods' qualified dataset sets.
- Infrastructure-only records are counted as incomplete and excluded from
  scientific loss means. A partially executed campaign is labeled incomplete.

The summary saves the actual dataset IDs, usable pair counts, attempted pair
counts, and joint qualification rate for each comparison. A failed fit is not
dropped to improve a method's operational mean. A conditional native comparison
does not estimate the all-attempt performance of an operational forecasting
system. Means of dataset-level RMSE are labeled accordingly; they are not
pooled root-mean-square errors over every field entry.

## Explicit primary macro contrasts

A primary specification fixes the reference, comparator, task, score, and
positive scenario weights summing to one. For example:

```python
primary = [{
    "reference": "bayes_esn_ridge",
    "other_model": "separable_stgp",
    "track": "B", "metric": "crps",
    "scenario_weights": {
        "g1_gaussian_ar": 0.25, "g2_spectral_transport": 0.25,
        "g3_excitable_field": 0.25, "g4_existing_fronts": 0.25,
    },
}]
```

The default schema-4 family compares ESN with GP and basis GRU on A and B over
the declared core scenarios. All required models/tasks must be in the roster;
the factory never silently removes an unavailable comparator. The full family,
including its weights and selection mode, is bound into the confirmation freeze.

Let d_rg be a normalized paired dataset-level loss difference, with N_g usable
pairs in independent scenario stratum g. Then

\[
\widehat\Delta=\sum_g w_g\bar d_g,\qquad
\widehat{\operatorname{MCSE}}(\widehat\Delta)
=\left(\sum_g w_g^2\frac{s_{d,g}^2}{N_g}\right)^{1/2}.
\]

`macro_comparisons(summary, primary)` also supplies a stratified dataset-level
bootstrap interval. Intervals are **pointwise**, not simultaneous superiority
claims. Each CRPS cell retains its own common-qualified population; datasets
need not qualify across every independent scenario. A missing cell makes the
macro contrast incomplete. Weights are never renormalized over available cells.

The implementation deliberately rejects an independent-stratum macro contrast
containing both G5/G4 or G6/G2: their control streams are paired. Their ordinary
paired cell analyses remain available. A cross-scenario macro estimand for
those coupled arms needs an explicitly implemented cluster estimator; its
covariance cannot be discarded.

## Attempted-replication precision planning

Schema-4 `freeze_confirmation` defaults to `planning_mode="paired_macro"` and
requires a completed, source-stable development run at the intended scientific
dimensions, model/tuning policies, and numerical runtime. Only run name,
development/confirmation stage, replication count, and retained illustration
IDs may differ. The new mode requires at least five usable paired development
differences in every constituent primary cell. This is a minimum evidence gate,
not a claim that five pairs precisely estimate a variance.

For native CRPS, let p_g be the **joint** qualification probability for the
two methods in cell g. A planning approximation for R attempted independent
datasets per core mechanism is

\[
N_g\approx Rp_g,\qquad
\operatorname{MCSE}^2(\widehat\Delta)
\approx\frac1R\sum_g\frac{w_g^2s_{d,g}^2}{p_g}.
\]

The factory records the observed joint rate and its Wilson score interval.
It uses the lower interval endpoint as a conservative planning input, takes
the largest requirement across primary contrasts, and bounds attempted R at
20–60. Operational RMSE retains all completed outcomes and therefore receives
no native-qualification correction. Marginal qualification rates are never
multiplied, and qualified-pair counts alone cannot identify attempted R.

The receipt includes observed/lower/upper qualification sensitivity, variance
multipliers 0.5/1/2, expected usable counts, requested uncapped R, MCSE at the
cap, and cap status. The variance multipliers are sensitivity scenarios, not
confidence bounds. Wilson limits are not a simultaneous precision guarantee.
These are approximate MCSE calculations, **not hypothesis-test power**.

Final attempted R is fixed before final seeds. A cap shortfall remains visible;
the runner does not generate datasets until enough qualified or favorable
outcomes appear. Freeze verification recomputes the complete plan, checks its
family and source/runtime/design identities, and rejects changed assumptions.

Legacy schema-3 all-pairs freezes remain readable and verifiable through their
original rule. Schema-4 cannot downgrade to the legacy planning gate. Receipt
fields named `power`/`power_status` are retained for interface compatibility;
new receipts explicitly identify their MCSE meaning and planning mode.

## Forecast ownership and measured memory scope

The public `Forecast(...)` constructor still copies arbitrary finite caller
arrays, including noncontiguous and Fortran layouts. Its published numerical
arrays are immutable views of frozen private allocations. `Forecast.load`
transfers freshly deserialized numeric buffers without copying them again.

The private `Forecast._from_owned(...)` factory accepts complete contiguous
float64 allocations or whole-allocation immutable views from the core
`PredictiveSample` producer. Internal callers must relinquish mutable aliases.
Partial/strided/wrong-dtype transfers are rejected. This removes the additional
complete observation-path copy at the ESN benchmark boundary without changing
any generated path or its spatial/temporal dependence.

The [bounded allocation probe](../benchmarks/profile_forecast_ownership.py) uses
two fresh sequential subprocesses, one numerical thread each, and a fixed
256-by-100-by-505 float64 path array (98.6 MiB). The
[saved receipt](evidence/v03/forecast_ownership.json) records identical complete
path/mean SHA-256 values. On that host, total peak RSS was approximately
307.1 MiB for the public-copy path and 205.4 MiB for the private transfer;
the constructor's incremental high-water increases were approximately
100.1 MiB and 1.0 MiB. This is evidence about that allocation workload, not
about total fitting memory or universal prediction speed.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
python benchmarks/profile_forecast_ownership.py results/forecast_ownership.json
```

Disk-backed scoring was not added merely to satisfy an architecture preference.
The existing core prediction iterator preserves global path IDs and coherent
whole-field paths. The full benchmark scorer still materializes its samples;
an end-to-end streaming claim would be incorrect. Larger workloads must pass
resource admission or receive a separately validated batch/disk-scoring route.
Exact empirical CRPS and quantiles require the relevant sample axis, not only
running means.

## Focused checks

`tests/test_benchmark_studies_v03.py` covers endpoint/subset invariance,
training-only normalization, nested sensor roles and common fields, explicit
populations, independently computed macro variances, joint-rate attrition,
freeze recomputation and tamper rejection, public array isolation, private
ownership transfer, and legacy interfaces. Existing numerical, model, scoring,
installed-wheel, and release tests remain separate integration gates.
