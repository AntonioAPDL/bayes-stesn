# Full-field Bayesian KS screening: execution handoff

This is a new, validation-only extension of the custom KS experiment. It does
not replace the accepted model or alter the previous frozen results. The
previous test results have already been inspected, so this extension must not
be described as a new untouched-test assessment or a published leaderboard
reproduction.

Implementation branch: `application/ks-fullfield-20261105`.
Parent revision: `5d3b06a173d7a62fd38e8e26e9870ea3dc00e676`.
Workspace: `/data/ks_fullfield_20261105/repo`.
Verified data root: `/data/ks_fullfield_20261105/cache/data`.
Application cache root: `/data/ks_fullfield_20261105/cache`.

## What changed

The additional `bayes_stesn_fullfield` adapter uses every permitted input site
at every configured lag. It does not fit PCA, POD, or another response-dependent
projection. The strict interleaved track supplies 39 input sites; the
all-observed track supplies 64; the contiguous robustness track supplies 40.
Hidden histories and the 128-point numerical solver field remain excluded.

The reservoir is a fixed sparse leaky tanh reservoir with exact seeded matrix
spectral-radius scaling. Spectral radius and operator norm are separately
calculated and reported. A zero or degenerate normalization is an explicit
failure, without a replacement seed. The new adapter retains the existing
Gaussian spatial likelihood, noise-scaled prior implementations, collapsed
sampler, and exact conditional Gaussian/inverse-gamma recovery. It exposes
both scalar ridge and regularized horseshoe variants; the finite search uses
the group horseshoe for a tractable matched prior comparison.

The response is either the next state or an increment added to a persistence
baseline. Forecast paths keep one parameter draw fixed for all 100 autonomous
steps and propagate their own predictions with joint spatial innovations.
Increment forecasts advance the native observed-site state. The unseen-site
baseline is an output-only periodic interpolation; it does not feed hidden
values or a smoothed decode/reprojection back into the recurrence.

Full-field **inputs** do not remove the spatial output model. The readout still
uses a coordinate-evaluated spatial dictionary, which is needed to predict
never-trained coordinates. Predetermined periodic Fourier dictionaries are
compared with the original radial dictionaries. Their dimensions and prior
normalizations are recorded explicitly. The Fourier dictionary is not PCA.

See [the hierarchical model specification](../ks_fullfield_model.md)
for equations, prior scaling, feature definitions, inference, and forecast
semantics. The old `ks_native.py` adapter is unchanged.

## Frozen search scope

| Component | Declared proposals |
|---|---|
| Main tracks | 79 configurations each for strict interleaved A and all-observed B |
| Deferred robustness | 53 contiguous-site C configurations, after the main work |
| Reservoir units | 128, 256, 512, 1,000 |
| Leak | 0.02, 0.05, 0.1, 0.2, 0.5, 1 |
| Spectral radius | 0.1, 0.3, 0.6, 0.9, 1.2, 1.5 |
| Joint dynamics coverage | Full 6-by-6 leak/radius grid at 128 units in A and B |
| Lags | Consecutive windows through 32 states and explicit spaced lag sets |
| Capacity and memory interactions | Fixed 512/1,000-unit proposals with long or spaced lags |
| Target | State and persistence correction/increment |
| Readout prior | Existing global ridge and matched finite-slab group horseshoe |
| Spatial dictionary | Periodic Fourier capacity ladder and matched radial controls |
| Direct readout | Latest field, all configured field lags, or reservoir-only controls |

Reservoir size counts **units**, not total matrix entries or readout weights.
For example, the B configuration with 1,000 units, the latest 64-site direct
term, 64 squared reservoir features, and an intercept has 1,129 temporal
features. With 33 spatial dictionary columns, its Bayesian readout has 37,257
coefficients. All configured lags still enter the reservoir when only the
latest field is included directly in the readout.

Per-trajectory training remains `[0,6000)`. The default conditioning prefix of
512 states leaves 5,488 likelihood targets, while the delivered scalar
transform uses the complete permitted training subset. A common causal
history length is selected from 512, 1,024, and 2,048 states using training-only
reservoir synchronization checks, before validation selection. Training
conditioning, origin history, and MCMC warm-up are distinct quantities.

Screening uses trajectories 0 and 5 and validation starts
6000, 6500, 7000, and 7500. Promotion uses the six predeclared selection
trajectories and eight starts. Selected configurations are frozen before
confirmation on the remaining 14 trajectories, with all 20 nonoverlapping
validation origins per trajectory. Additional fixed reservoir seeds are a
sensitivity analysis, not posterior draws. Mean, persistence, and a fixed
full-input VAR provide matched anchors. Previous competitor results remain
historical unless explicitly matched to this experiment.

Three predeclared selections are reported separately: distributional CRPS;
CRPS subject to early-horizon RMSE being within 10% of the best eligible
candidate; and point-forecast squared error. Losses weight the horizon bands
1–10, 11–25, 26–50, and 51–100 equally. Predictive means average complete
paths. Near-tie tolerances incorporate a fixed-case Monte Carlo precision
check, then prefer longer valid prediction time and fewer coefficients.

## Resource measurements and budget

The container has an 8 GiB memory limit and an 8-CPU quota. The task permits
one expensive worker and one numerical thread at a time, with a 5 GiB task
memory cap. The disk reserve is 12,781,798,196 bytes, 20% of initially available
space. Allocation estimates include the complete coefficient diagnostic
tensor even when it is backed by a memory-mapped temporary file.

| Resource case | Fit wall time | Forecast wall time | Peak RSS | Inference use |
|---|---:|---:|---:|---|
| B, 128 units, 6,000 training states | 62.57 s | 1.18 s for 256 paths | 326 MB | Full four-chain gate passed |
| B, 512 units, eight lags, pilot | 17.49 s | 5.73 s for 256 paths | 252 MB | Short resource pilot only |
| B, 1,000 units, 32 lags, pilot | 34.55 s | 12.68 s for 256 paths | 383 MB | Short resource pilot only |
| A, 128 units, horseshoe, pilot | 6.73 s | 2.74 s for 256 paths | 145 MB | Short resource pilot only |
| B, 1,000 units, horseshoe, pilot | 88.47 s | 96.89 s for an independent cold bank of 512 paths | 504 MB | Short resource pilot only |

All forecasts in this table have horizon 100. The full reference fit used four
chains, 1,000 warm-up iterations and 2,000 retained draws per chain. Its
hyperparameters, all 8,481 readout coefficients, and monitored predictive
functionals passed the R-hat/ESS gates without continuation. Resource pilots
do not supply scientific performance scores.

The prospective finite budget is 60 worker-hours and 72 elapsed hours. At most
26 worker-hours cover preflight, anchors, screening, promotion and predictive
precision checks. The remainder is reserved for confirmation and seed
sensitivity. The initial estimate is 12–18 hours for the main screening and
24–50 hours overall; convergence, selected model sizes, and synchronization
failures can change these estimates. The finite limits are authoritative.

The main screen has 158 configurations, or 316 initial per-trajectory fits
before exclusions. The staged roster is response-independent. If a stage cap
is reached, unfinished jobs remain pending and any selection from the
completed subset is explicitly labeled partial-roster. A failed or
inference-unqualified fit never receives a substitute point forecast for
probabilistic scoring. Per-fit time limits retain compatible checkpoints.

## Verified acceptance evidence

The four supplied objects passed exact byte and SHA-256 checks. The complete
catalog was parsed, the supplied restorer passed its dry-run, and 458 selected
archive members were verified after restoration. All 23 HDF5 external links
resolved with the raw/canonical relative layout intact. No KS data were
generated or substituted.

Actual-data fixed-fit checks changed future labels and hidden input histories
while holding fitted artifacts and randomness fixed. The strict-site check
preserved all 16 complete 100-step paths exactly, and separately refitted
permitted preprocessing/features remained unchanged. The facade constructs
labels internally; this establishes output separation, not future-free disk
access.

Evidence files in `docs/applications/evidence/` contain the retrieval hashes,
resource measurements, explicit budget, and boundary checks. Private accounts,
retrieval links and storage manifests are excluded from the public report.

The final combined feature, model, data-boundary, report, backup, and legacy
regression run passed 138 tests and seven subtests. Four tests were skipped:
three require optional PyTorch, and one requires the historical native pilot
artifact that is not present in this restored workspace. The new full-field
adapter has its own actual-data fixed-fit boundary checks described above.
Coordinator-specific tests and its real-cache resume check are recorded
separately with the execution specification: all 16 coordinator tests passed.
The combined totals are 154 passed tests, seven passed subtests and four
documented skips.

## Execution, resumption and persistence

The coordinator writes stable configuration/source/data identities, atomic
checkpoints, per-origin receipts, and a ledger with separate pending, running,
complete, failed, inference-unqualified and resource-blocked states. Numerical
source or runtime mismatches prevent checkpoint reuse. Process identity uses
host-visible PIDs, start ticks and boot identity because this environment has
nested PID namespaces; cross-namespace file locking was verified.

A plain-pipe resource process disappeared after approximately 123 seconds
without a Python traceback or OOM event. Its saved fit and 167 predictive
parameter draws resumed successfully. The scientific launch therefore uses
a persistent terminal execution session. A 190-heartbeat probe completed in
that mode. This is evidence of the tested launch behavior, not a guarantee
that the hosting workspace will remain available indefinitely.

Recomputable predictive banks reuse exact fixed parameter draws across origins
and use fresh innovation streams. Independent parameter banks are used for
the Monte Carlo precision check. This shares Monte Carlo error across origins;
it does not change an origin's posterior predictive law and is not treated as
independent scientific replication. Batched reservoir updates and a single
history replay per origin reduce repeated work.

The final operational configuration, current status and resume commands are
recorded with the run. The automatic report distinguishes screening,
confirmation, failures and pending work. Improvement claims require the
completed scientific results; implementation and a passing resource fit alone
do not establish an improvement.

## Reproducible run commands

The frozen execution configuration is
`configs/applications/ks_fullfield_v1.json`. Its SHA-256 is
`979ceb5ad58366e9a89fe4ddcf685041d982c121283bab24711b1a3e165325c6`.
The configuration contains the finite roster, seed, origin, selection and
budget policies. The coordinator additionally binds numerical source files,
package versions, and verified cache identity before running a job.

Run or resume from the repository in a retained terminal or authorized batch
scheduler:

```bash
/data/ks_fullfield_20261105/env/bin/python tools/run_ks_fullfield.py run \
  --stage all \
  --config configs/applications/ks_fullfield_v1.json \
  --cache /data/ks_fullfield_20261105/cache \
  --output /data/ks_fullfield_20261105/runs/fullfield_v1_20261105T210330Z
```

Replacing `run` with `resume` reuses completed, identity-compatible fits and
origins. It does not create a new budget. Separate stage commands use
`--stage preflight`, `screen`, `promote`, `precision`, `freeze`, `confirm`,
or `sensitivity`; dependencies remain enforced. The deferred C stage requires
a matching main-completion record. The launcher sets the numerical thread
limits explicitly before importing the model runtime.

Inspect status and regenerate the report without fitting:

```bash
/data/ks_fullfield_20261105/env/bin/python tools/run_ks_fullfield.py status \
  --output /data/ks_fullfield_20261105/runs/fullfield_v1_20261105T210330Z

/data/ks_fullfield_20261105/env/bin/python tools/run_ks_fullfield.py report \
  --output /data/ks_fullfield_20261105/runs/fullfield_v1_20261105T210330Z
```

The supplied retrieval helper restores the already verified four-object
download set into a separate cache. Never restore over the repository:

```bash
python restore.py --catalog drive_catalog.json --incoming DOWNLOAD_DIR \
  --destination CACHE_ROOT --dataset ks --dry-run
python restore.py --catalog drive_catalog.json --incoming DOWNLOAD_DIR \
  --destination CACHE_ROOT --dataset ks
```

Use the existing authorized private retrieval route to obtain those exact
objects; their expected bytes and hashes are in the public acceptance receipt.
Keep the reader's `CACHE_ROOT/data/ks/raw` and `canonical` relative paths intact.
Incremental private backup preparation and strict remote-verification
acknowledgment are documented in [the backup guide](../ks_fullfield_backup.md).
