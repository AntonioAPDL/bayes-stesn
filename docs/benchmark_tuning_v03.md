# Versioned effective tuning and bounded model fitting

The v0.3 workflow uses benchmark schema **4.0**. The previous `smoke`, `pilot`,
and `main` profiles still use schema 3.0; their serialized dictionaries,
identities, and candidate-generation rules are preserved. Do not reinterpret
old results as if they used the new search or fitting policy.

## Profiles and prospective budgets

| Python profile | Purpose | Initial rows / horizon | Distinct candidates | Finalists × fresh seeds |
|---|---|---:|---:|---:|
| `v03_smoke` | Execution check only | 120 / 8 | 1 | 1 × 1 |
| `v03_pilot` | Small development | 400 / 40 | 8 | 2 × 3 |
| `v03_main` | Unfrozen confirmation draft | 2,000 / 100 | 8 | 2 × 3 |

The new roster includes `basis_gru`. The [configuration directory](../configs/benchmark/v03/)
also contains a five-replication development profile at the draft main geometry.
Loading a confirmation draft does not authorize its execution: the independent
development and confirmation-freeze gates still apply.

Eight means distinct **effective** configurations. Persistence has one. Seasonal
repetition exhausts its four declared periods. Other small spaces can exhaust
after actual fitting-subset restrictions or explicit overrides. Seed repetitions
do not increase the number of configurations.

For the substantive profiles, ESN allowances are four chains, 1,000 warmup
iterations, 2,000 retained draws, and at most 2,000 additional retained draws.
The original diagnostic thresholds remain unchanged. These are prospective
budgets, not evidence that every model will converge or improve its prediction.
The tiny smoke configuration is deliberately inadequate for scientific MCMC
qualification and must never be cited as convergence evidence.

## Inspect a search without fitting models

```python
from bayes_stesn.benchmark.protocol import profile, candidate_manifest

protocol = profile("v03_pilot")
design = candidate_manifest(
    "basis_gru", protocol, track="B", seed=923,
    fitting_sites=20, fitting_rows=400,
)
configs = design["candidates"]
hashes = design["candidate_hashes"]
receipt = design["metadata"]
assert len(configs) == len(set(hashes)) == 8
```

Planning imports no PyTorch. `candidate_configs(...)` remains the list-returning
API and accepts the same optional `fitting_sites` and `fitting_rows` arguments.
Pass the smallest actual spatial training-fold size when folds differ.

The resolution order is candidate construction, recursive overrides, adapter
defaults and subset constraints, canonicalization, then deduplication. Fitting
seeds use a separate namespace keyed by the effective hash. The raw proposal
sequence stays fixed when the requested number of effective candidates grows.

The receipt distinguishes `requested_count_reached`, `finite_space_exhausted`,
and `proposal_cap_reached`. It records proposals, duplicates, invalid proposals,
effective hashes, constrained dimensions, and each resolved option. A fixed
proposal cap bounds rejection work; hitting it is not evidence that the finite
space was exhausted.

The search uses a conservative reference followed by a deterministic sample
without replacement from declared finite factors. It does not execute a full
Cartesian grid. Rank/mode constraints apply to actual training subsets. An
overcomplete regularized ESN spatial dictionary is not automatically rejected.
Actual numerical rank remains a fitted-model diagnostic.

Response-dependent defaults are explicit *rules* until permitted fitting data
are available. Examples include a GP training mean and initial signal/noise
scales, a KRR median-distance bandwidth, and an ESN conditioning-prefix encoder.
Those values must not be estimated from validation or test responses merely to
complete a configuration receipt.

## Overrides and fixed configurations

Nested overrides preserve sampled settings they do not mention:

```python
protocol = profile("v03_pilot", model_options={
    "bayes_esn_ridge": {"features": {"recurrent_norm": 0.75}}
})
```

This preserves the sampled width, layers, input scales, and lag choices.
Unknown top-level or nested keys are errors. Arrays are replaced as whole
arrays. Consequently a manually supplied layer-specific array must agree with
the resulting number of layers; incompatible candidates are recorded as
invalid rather than silently assigned different architectures.

The alternate `development_frozen_configuration` mode requires explicit
`frozen_configurations`. Keys can be a model ID or `model_id/A` and `model_id/B`.
The latter take precedence. The automatically added `spatial_persistence` B
control is the sole omission exception: its recorded `fixed_spatial_control`
reference is `{spatial_length: 0.2, ridge: 0.01}`, without tuning. An explicit
frozen declaration can replace that reference. Other missing model/track
configurations are errors. Additional
`model_options` overrides for a frozen model are rejected: its effective
configuration is authoritative. Every new dataset still receives a fresh
prespecified fit seed and refits its statistical parameters. This is a different
scientific procedure from tuning each dataset and needs its own development
precision assessment and frozen study identity.

Frozen configurations are resolved against the full final refitting panel.
Unused pseudoheld validation folds must not reduce their spatial rank or
conditioning history.

The named `bayes_esn_ridge` benchmark identity enforces scalar ridge and iid
innovations. A nested override cannot turn it into a horseshoe or residual-AR
comparator under the same name. Those core APIs remain available for separately
declared and validated model comparisons.

Schema-4 `qualified_finalists` selection requires every planned fresh-seed and
spatial-fold recheck to qualify. Screening scores only rank the evaluation
order. The finalist score averages all successful planned rechecks, and a
single failed recheck makes that finalist ineligible. Deterministic methods
receive one recheck. If the initial finalists do not produce an eligible
configuration, the fixed replacement allowance evaluates the next candidates.
If none qualifies, the declared fallback is retained. The independent final
refit must pass its own gate.

## Actual neural stopping horizons

The new profiles explicitly set `validation_horizon` to the benchmark horizon
and `require_full_validation_horizon=True` for both neural adapters. The
chronological stopping prefix is shortened when necessary to reserve that
historical tail. If insufficient training rows remain for the context/lags and
training objective, resolution fails before allocating a network.

The adapters record requested horizon, actual stopping leads, prefix rows, and
whether truncation occurred. Legacy adapter calls retain the previous default
`require_full_validation_horizon=False`; their shorter realized horizon is now
visible in diagnostics. Epoch stopping evaluates autonomous rollouts, and the
selected epoch count is refitted on the supplied training history as declared.
This correction does not silently change the neural training loss into an
H-step objective.

## Optimizer allowances and diagnostics

The new LGSSM search permits 200 EM iterations at the unchanged relative
likelihood tolerance `1e-5`. Its diagnostic trace distinguishes convergence,
iteration exhaustion, and a rejected decreasing step; it records the actual
final relative likelihood gain. Increasing the cap preserves the existing EM
sequence and does not turn an exhausted fit into a qualified one.

The new GP search starts with 60 L-BFGS-B iterations and permits one bounded
restart of at most 120 iterations. The basis-GRU residual-covariance optimizer
starts with 40 and permits one restart of at most 80. Restart is attempted only
when the first phase is finite, improves its initial objective, and terminates
at its iteration cap. Abnormal line-search failures are not automatically
restarted. Tolerances remain unchanged. A worsening restart is rejected and
cannot qualify the result.

This is a restart from the last parameter vector; it does not restore SciPy's
private quasi-Newton history. The default continuation allowance for existing
adapter calls is zero. Each phase records its status/message, objective,
iterations, evaluations, convergence tolerances, and projected-gradient
infinity norm. Boundaries are reported separately. A small projected gradient
is a KKT diagnostic, not proof of a global likelihood optimum. Added optimizer
allowances are algorithmic budgets and remain frozen across infrastructure
resumption.

An optional schema-4 `fit_time_limits_seconds` mapping supplies a per-model
elapsed-time allowance around each fitting call. It is part of the scientific
identity and defaults to `None`; it does not wrap forecasting or scoring.
The POSIX timer requires a main-thread worker, cannot share an existing active
alarm, and can be delayed by native calls. Scheduling chunks controlled by
`--max-wall-seconds` are separate and do not truncate a fit. See
[the execution guide](benchmark_guide.md) for retry and resource policies.

## Targeted verification

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_v03_tuning.py \
  tests/test_benchmark_optimizer_policy.py \
  tests/test_benchmark_classical.py tests/test_benchmark_basis_gru.py \
  tests/test_benchmark_neural.py
```

Tests cover legacy identities, stable distinct candidates, recursive overrides,
actual-subset constraints, finite-space versus proposal-cap exhaustion, fixed
configuration semantics, training-only normalization, actual neural stopping
leads, bounded continuation, bound-aware gradients, unchanged EM trace prefixes,
and existing classical/neural serialization and prediction contracts. Tiny
neural tests require the optional CPU-neural environment.
