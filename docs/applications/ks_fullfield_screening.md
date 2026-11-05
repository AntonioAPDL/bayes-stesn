# Full-field KS screening contract and operations

This document accompanies `configs/applications/ks_fullfield_v1.json` and the
response-independent proposal generator in
`src/bayes_stesn/applications/ks_fullfield_candidates.py`. The exact hierarchy
is documented in [the model specification](../ks_fullfield_model.md), and
measured resource and data receipts accompany the
[execution handoff](ks_fullfield_launch.md).

This is a new, exploratory **validation-only** extension. Historical test
results were already inspected. No command in this coordinator scores test
labels, changes the original adapters, or overwrites frozen historical results.

## Information and model boundaries

Each trajectory supplies its complete permitted training field over
`[0,6000)`: 39 interleaved sites in A, 64 sites in B, or 40 contiguous sites
in C. There is no PCA/POD input encoder. All permitted sites at every declared
lag drive the fixed sparse reservoir. Direct readout terms are independently
declared as the latest field, all field lags, or none. Thus choosing the latest
direct term does not remove older lags from the reservoir input.

The conditioning prefix is 512 states, leaving 5,488 likelihood targets.
The supplied trajectory/protocol-specific scalar transform uses the complete
permitted training interval. Feature scaling uses the model's conditioning
prefix. No validation-fitted scaler, basis, calibration wrapper, or covariance
is transferred into fitting. The spatial Fourier dictionary is predetermined
from coordinates; it is an output representation, not a response-fitted input
projection.

The registry has 79 candidates in each main track and 53 in the separately
deferred contiguous track. It includes a full six-leak by six-radius grid at
128 units in each main track, consecutive lag windows of 1, 2, 4, 8, 16 and 32,
spaced windows, matched 128/256/512/1000-unit capacity comparisons, and explicit
large-reservoir/long-memory joint proposals. Input scales, connectivity,
state/increment targets, direct readout terms, squared features, and spatial
dictionary capacity are also varied. Every group-RHS configuration has an
identified matching ridge control. Effective hashes canonicalize inactive
encoder/normalization fields so unused options cannot create fake candidates.

All reservoir units enter the readout. The default capacity ladder adds 64
squared reservoir states; a 128-unit control includes all 128 squares.
For B with 1,000 units and latest-field direct terms, the readout has
`1 + 64 + 1000 + 64 = 1129` temporal columns. With 33 spatial columns, this is
37,257 readout coefficients, not 1,000 total weights.

True matrix spectral radius is the declared tuning parameter; operator norm
is separately measured. Candidates with radius above one are permitted, but
must pass the same empirical synchronization check. A zero/degenerate seeded
matrix is a recorded failure rather than an excuse to select another seed.

## Frozen stages

| Stage | Trajectories | Validation first targets | Predictive paths |
|---|---|---|---:|
| Training preflight | 0, 5 | No evaluation responses | None |
| Screen | 0, 5 | 6000, 6500, 7000, 7500 | 256 |
| Promote | 0, 5, 10, 15, 4, 9 | 6000, 6200, 6500, 6700, 7000, 7200, 7500, 7700 | 512 |
| Predictive precision | 0 | 6000, 6500; four independent banks/repetitions | 512 |
| Confirmation | Remaining 14 trajectories | 6000 through 7900 by 100 | 512 |
| Seed sensitivity | 0, 5, 10, 15 | Same eight promoted origins | 512 per seed |

All forecasts are autonomous for 100 steps. A first target `a` uses only
`y[a-history:a]`; the adapter origin is `a-1`. Histories are initialized anew
for each origin and use permitted sites only. The facade internally constructs
evaluation labels, so the guarantee is model/evaluator output separation,
not future-free disk access.

Preflight checks histories 512, 1024 and 2048 against the training-end state
and feature vector under the same fitted feature map. The common passing
length is frozen before screening. If no length passes on both preflight
trajectories, the candidate is excluded with its receipt. This is an empirical
training-end synchronization check, not a global dynamical proof. The same
frozen history is checked on each later fitted trajectory and model seed.
Increment candidates replay original observed states, never increment labels.

Only feature-map synchronization evidence is shared between exactly matching
maps. Bayesian fits are distinct across target, prior, model seed, spatial
track, training trajectory, and effective configuration. Completed identical
fits and origin receipts are reused.

Each scientific fit uses four chains, 1,000 warm-up iterations, 2,000 retained
draws per chain, and at most one declared 2,000-draw continuation. Rank-normalized
R-hat and bulk/tail ESS gates remain at 1.01 and 400. All readout coefficients
are checked through bounded diagnostic storage, alongside hyperparameters and
monitored predictive functionals. Short resource pilots are explicitly
provisional; their forecasts cannot enter scientific rankings.

The mean of complete predictive paths supplies point scores. Each path holds
its parameter draw fixed across the horizon, generates the accepted spatially
correlated residual innovations, and propagates its own predictions. A
fit-specific parameter bank may be reused across origins; innovations remain
origin-specific. Precision repetitions use explicitly independent banks.
Model seeds are never pooled into posterior samples.

## Selection and finite-budget behavior

Selection is equally weighted across trajectories and across the four horizon
bands 1–10, 11–25, 26–50 and 51–100. The three published decisions are separate:

1. **Distribution:** minimum band-balanced CRPS.
2. **Dynamics:** minimum CRPS among eligible configurations whose mean
   trajectory-level RMSE over horizons 1–10 is within 10% of the best eligible
   early RMSE.
3. **Point:** minimum band-balanced squared error.

CRPS near ties use the larger of 1% of the lowest loss and twice the paired
predictive Monte Carlo standard deviation on the fixed precision cases.
Within that set, longer valid prediction time precedes fewer readout
coefficients, raw loss and deterministic identifier. The CRPS/early-RMSE/VPT
Pareto set is retained. VPT is the first spatial RMSE/training SD crossing of
0.4, in 0.25-time-unit steps, right-censored at 100. A finite severe divergence
is the existing normalized RMSE greater than 10 convention.

Promotion retains the eligible raw reference, best CRPS, best early RMSE,
best VPT, and best eligible RHS candidate, then fills the declared count of six
from the Pareto/CRPS ordering. An additional exact ridge counterpart is allowed
for the best RHS candidate. Every eligible candidate must complete every
planned trajectory and origin, pass inference, and have no severe-divergent
origin. Incomplete or failed episodes are never silently dropped from that
eligibility requirement.

Candidate order is response-independent. Required state/increment, RHS,
1,000-unit, long-lag and large-reservoir/long-lag controls precede a balanced
round-robin over strata. A candidate's two screening trajectories remain a
bundle, and A/B bundles are interleaved. At a finite screen or promotion cap,
unfinished jobs remain explicitly pending. Only fully completed eligible
candidates may advance. At a precision cap, only candidates with every
predeclared precision repetition and origin complete may receive a frozen
choice; missing cases are preserved as precision exclusions. A resulting
partial-roster selection is labeled as such and is not described as full
screening completion or universal optimality.

Both main tracks are frozen before confirmation. Training-mean, persistence,
and a fixed full-input VAR (`lags=4`, `rank=number of permitted sites`,
mean-loss ridge `1e-5`, state target) are evaluated on the same confirmation
trajectories, histories and origins. Its objective is mean squared training
error plus `1e-5` times the squared non-intercept weights. This VAR is a fixed anchor for the new
Bayesian expansion, not a newly tuned competitor. The historical competitor
study is preserved separately.

C runs only after matching `main_completion.json` is present. Its data,
transforms, histories, fits and selections are separate. The child campaign
carries all already-spent worker/search charges and the original elapsed
deadline. Its immutable remaining-stage allocation is derived from remaining
time, without predictive responses. If no budget remains, C stays pending.

## Execution budget and protection

The frozen configuration allows one expensive process and one numerical
thread. Caps are 60 worker-hours, 72 elapsed hours, and 26 search worker-hours.
The 900-second preparation allowance is charged once to total and shared
search, not to each stage. Stage allowances are:

| Stage | Worker-time cap |
|---|---:|
| Preflight | 1,800 s |
| Selection anchors | 900 s |
| Screen | 61,080 s (16 h 58 min) |
| Promotion | 21,600 s |
| Predictive precision | 7,200 s |
| Confirmation | 86,400 s |
| Seed sensitivity | 36,000 s |

Global caps govern overlapping allowances and deferred work. The search
allocation leaves 120 seconds for heartbeat stopping overhead. Individual
scientific jobs are bounded at 7,200 seconds; pilot and preflight jobs at 600.
The measured initial estimate is 12–18 hours for the main screen and 24–50
hours overall. These are resource extrapolations, not completion guarantees.

Admission includes a 1.5 multiplier and the full coefficient diagnostic
memmap as potentially resident memory. Task workers share a 5 GiB cap, with
container-aware headroom checks and a 12,781,798,196-byte disk reserve.
Prospective fit/checkpoint, diagnostic and parameter-bank storage is checked
before launch. Global tensor-product designs and coefficient covariances are
never allocated. The monitor checks memory and disk every five seconds and
emits a concise terminal keepalive every minute.

After a worker exits, optional `POSIX_FADV_DONTNEED` is applied only to closed,
task-owned regular NPZ files in that fit folder. This preserves file bytes and
checkpoints while advising the operating system to release their page cache.
There is no global cache dropping, raw-data eviction or deletion. Ridge banks
use RAM reuse and deterministic regeneration; expensive RHS banks also have
a declared 512 MiB disk cap per fit.

## Reproducible commands

Run from the repository root using the project environment. The numerical
source and configuration must remain unchanged after the run starts.

```bash
PYTHON=/data/ks_fullfield_20261105/env/bin/python
CACHE=/data/ks_fullfield_20261105/cache
RUN=/data/ks_fullfield_20261105/runs/fullfield_v1_20261105T210330Z
CONFIG=configs/applications/ks_fullfield_v1.json

# Run in a retained terminal execution session, tmux, or an authorized scheduler.
"$PYTHON" tools/run_ks_fullfield.py run --stage all \
  --config "$CONFIG" --cache "$CACHE" --output "$RUN"

# The same immutable source/config/output resumes interrupted work.
"$PYTHON" tools/run_ks_fullfield.py resume --stage all \
  --config "$CONFIG" --cache "$CACHE" --output "$RUN"

# Read current status without fitting or data reads.
"$PYTHON" tools/run_ks_fullfield.py status --output "$RUN"

# Refresh tables/figures from existing receipts only.
"$PYTHON" tools/report_ks_fullfield.py --output "$RUN"
```

Stages `preflight`, `screen`, `promote`, `precision`, `freeze`, `confirm`,
`sensitivity`, and `deferred` can also be selected. Later stages reuse their
required predecessors. A separate `pilot` output is required for a provisional
budget or shortened inference settings; it must not become the scientific run
by changing its configuration in place.

`tools/run_ks_fullfield.py stop --output "$RUN"` signals only verified live
processes in the same PID namespace. Otherwise interrupt the original retained
terminal session. This environment exposes host-visible `/proc` inside a
nested PID namespace; process identities therefore bind visible PID, namespace
PID, namespace inode, start ticks and boot identity. Live orphan workers block
new scheduling. If a crashed coordinator has no final worker receipt, the
unobserved interval until resumption is conservatively charged, with that
uncertainty recorded. Resume cannot erase spent time.

## Persistent outputs and acceptance

`experiment.json` binds source/runtime, data audit, configuration and registry.
`candidate_registry.json` records exact effective proposals and cost estimates.
`history_policy.json`, `selection/`, `freeze.json`, `confirmation/`, and
`sensitivity/` preserve the scientific decisions. `jobs/`, `ledger/`,
`attempts/`, `terminal/`, `fits/`, and per-origin `forecasts/` separate execution
and scientific outcomes. `report/` contains replaceable Markdown, CSV, PNG and
PDF snapshots generated automatically at completion or pause. Optional compact
animation uses only the predeclared saved example; no test data are read.

Targeted runner acceptance covers unique effective proposals, exact matched
controls, JSON-stable identities, diagnostic-memory accounting, distinct
selection objectives, true nested-namespace child identity, orphan refusal,
conservative crash charging, partial-stage progression, precision completeness,
failure counting after a successful fit, and preservation of advised NPZ
bytes. An actual-cache subprocess preflight and fresh-coordinator resume also
passed: the same completed job was reused without another worker attempt.
