# Reproducing and resuming the Bayesian KS improvement campaign

## Scope

This branch preserves the original KS application at revision
`2ecd8052ce2a44c9b4ed3862382594bd84598ad7`. The new driver reuses the original
data adapter, Bayesian likelihood and prior implementation, sampler, native
fit/checkpoint code, predictive path generator, and score definitions. Its
104-candidate registry and original finite budget are in
[`ks_improvement_v1.json`](../../configs/applications/ks_improvement_v1.json).
The one prospective operational extension is recorded separately in
[`ks_improvement_v2_budget.json`](../../configs/applications/ks_improvement_v2_budget.json);
it preserves that scientific registry and the original campaign start.
The [design review](ks_improvement_literature.md) explains the scientific choices.

This is exploratory work after the original test results were inspected. New
screening, selection, precision checks, confirmation, and seed sensitivity use
validation intervals only. The 14 confirmation trajectories are excluded from
the new selection; they are not a retrospectively untouched test experiment.
The original results remain a separate snapshot.

## Recorded completion and evidence

The executed improvement campaign and its fixed-reference evaluations are now
terminal. The [final metadata audit](evidence/ks_improvement_final_evidence_audit.json)
records 758 terminal scheduler jobs: 752 complete and six inference-unqualified
screening jobs. All 126 confirmation fits qualified and all 2,520 confirmation
origins were finite. The matched-main-seed and additional-seed schedules retain
24 fits/192 origins and 48 fits/384 origins respectively, with no missing or
blocked outcomes. The selected-artifact boundary checks passed all nine cases.
These counts describe completion and numerical eligibility, not predictive
adequacy.

The five fixed point-reference families contribute 210 terminal trajectory/model
jobs: 191 complete and 19 failed. POD-NVAR retains two failed origins in A, three
in B and 181 in C; its contiguous-site aggregate is undefined because no complete
C trajectory remains. Successful-only results and the full outcome census stay
separate. The [independent numerical audit](evidence/ks_improvement_final_numerical_audit.json)
reconciles the compact exports against the retained receipts. The integrated
report has 24 cohorts and 317 complete cohort episodes; this is not a count of
independent scientific replicates across methods.

The current human-readable report and movie review is a separate delivery step.
An unpublished QA preview is not a published final PDF. Use the final source-bound
visual review receipt, once present, for the exact inspected artifact hashes;
the historical pre-render receipts below do not establish that review.

## Environment and data

Use CPython 3.12 on Linux and a project environment. The observed compatible
versions needed by this native-only campaign are in
[`requirements-ks-improvement.lock`](../../requirements-ks-improvement.lock).
Do not upgrade an active scientific environment to render a report. The larger
original `requirements-ks.lock` includes the neural comparators and is not
required for this Bayesian improvement campaign.

For a new environment, from this repository:

```bash
python3.12 -m venv /data/ks-improvement-env
/data/ks-improvement-env/bin/python -m pip install -r requirements-ks-improvement.lock
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0 MPLBACKEND=Agg
```

Reuse the verified delivered cache; do not generate another KS dataset. Its
common reader root is `CACHE_ROOT/data`, with both `data/ks/raw` and
`data/ks/canonical` preserved. Canonical HDF5 external links require that relative
layout. The original KS retrieval receipt documents the source v2 and data v1
archive identities. If restoring elsewhere, use the supplied `restore.py` and
complete catalog, with the exact archive bytes and hashes checked first:

```bash
python restore.py --catalog drive_catalog.json --incoming DOWNLOAD_DIR --destination CACHE_ROOT --dataset ks --dry-run
python restore.py --catalog drive_catalog.json --incoming DOWNLOAD_DIR --destination CACHE_ROOT --dataset ks
```

Private retrieval/account details are intentionally absent from this public
document. The campaign itself audits the complete restored cache before use.

## Resume the recorded run

The executed scientific source files and package versions are hashed into every
job. A checkpoint with an incompatible identity is refused. Keep the numerical
source and configuration unchanged while resuming.

### Restored authoritative workspace; run_v3 migration applied

The repository, verified cache and compatible project interpreter have been
restored under `/data`. The original scratch paths are historical provenance,
not paths to recreate with symlinks. Use:

| Component | Restored path |
|---|---|
| Repository | `/data/ks_improvement_20261105/repo` |
| Verified delivered cache | `/data/ks_improvement_20261105/cache` |
| Project Python | `/data/ks_improvement_20261105/env/bin/python` |
| Original comparator-selection freeze | `/data/ks_improvement_20261105/recovered_original_evidence/ks_frozen_specification_full.json` |
| Preserved source campaign | `/data/ks_improvement_20261105/run_v2` |
| Terminal campaign after verified migration | `/data/ks_improvement_20261105/run_v3` |

```bash
cd /data/ks_improvement_20261105/repo
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0 MPLBACKEND=Agg
KS_PYTHON=/data/ks_improvement_20261105/env/bin/python
KS_CACHE_ROOT=/data/ks_improvement_20261105/cache
KS_ORIGINAL_FREEZE=/data/ks_improvement_20261105/recovered_original_evidence/ks_frozen_specification_full.json
KS_RUN_ROOT=/data/ks_improvement_20261105/run_v3
```

`KS_ORIGINAL_FREEZE` supplies `--original-freeze` to the fixed point-reference
tool. The improvement campaign's own selected models remain bound separately
by `freeze.json` in the active run root.

The guarded migration **passed and was applied** on 2026-11-05. Its receipt,
`run_v3/migration/receipt.json`, was created at 11:31:25.644123 UTC and has SHA-256
`95099d0c7d7eeaea0b9a82357912fcd012b86a45559c30ecb8c43eb0ab94a751`.
It preserves 348 completed native checkpoints, 3,138 complete forecast receipts,
and 31,138.42515946799 cumulative worker seconds, including the original
90-second preparation charge. No scientific computation was performed by the
migration, and the source run_v2 was neither modified nor deleted.
These are counts at the migration frontier, not a claim that confirmation,
seed sensitivity, or the auxiliary reference and acceptance schedules finished.

| Binding | Active run_v3 value |
|---|---|
| Experiment identity | `067aaf08f46d7a21895ce15876d482a4a837d58627026a59e07a63db37f5c376` |
| Freeze identity | `96774e83e2f737ceed30abc232b3a328b79894b0173a8e7a629c0289f0c3f4d2` |
| Original selection time, preserved | `2026-11-05T10:03:57.054419+00:00` |
| Original elapsed-budget start, preserved | `2026-11-05T07:38:04.272689+00:00` |

Use the v2 budget configuration with run_v3. Do not point it at the differently
bound run_v2 source, rerun migration into the existing destination, or reset the
elapsed-budget start.

### Recorded first run_v3 continuation

The first confirmation continuation began at 11:32:20.623753 UTC, under observed
coordinator session 48611. Its immutable launcher record is
`operations/bounded_launches/20261105T113220_524a265d73/invocation.json`.
The command below records that historical launch. That coordinator has finished;
do not execute it merely to rebuild a report, or start another coordinator while
any current writer is active.

```bash
"$KS_PYTHON" tools/run_ks_improvement_bounded.py \
  --stage confirm \
  --config configs/applications/ks_improvement_v2_budget.json \
  --cache "$KS_CACHE_ROOT" \
  --output "$KS_RUN_ROOT" \
  --additional-run /data/ks_improvement_20261105/run_v2 \
  --closed-session-receipt /data/ks_improvement_20261105/operations/closed_sessions_before_budget_extension.json \
  --minimum-mib 0.0625 --include-fit-receipts \
  --max-retries 6 --execute
```

This first launch deliberately omits `--reclaim-first`: no previous run_v3
coordinator had been observed to finish. Its supplied closure receipt covers
the preserved run_v2 source only. If its own child later ends with the exact
memory-admission failure, the wrapper can perform the existing bounded cache
advice and retry under its unchanged guards. Any later launch requesting initial
advice must instead supply fresh, genuine closure evidence for the latest
coordinator of **every** declared root, including run_v3; the historical source
closure receipt alone is insufficient. Do not fabricate a destination closure
or reuse a stale session number.

Current status, without scheduling fits:

```bash
"$KS_PYTHON" -m bayes_stesn.applications.ks_improvement_runner status \
  --config configs/applications/ks_improvement_v2_budget.json \
  --cache "$KS_CACHE_ROOT" \
  --output "$KS_RUN_ROOT"
```

### Historical v1/run_v2 invocation

The following preserves the earlier invocation and its scratch paths for
provenance. It is not a current command for the restored workspace. Relocation
uses the explicit reviewed migration below, rather than editing old jobs or
recreating missing source paths.

```bash
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0 MPLBACKEND=Agg

/workspace/scratch/ks_recovery_20261105T0615/env/bin/python \
  tools/run_ks_improvement_bounded.py \
  --config configs/applications/ks_improvement_v1.json \
  --cache /workspace/scratch/ks_recovery_20261105T0615/cache \
  --output /data/ks_improvement_20261105/run_v2 \
  --additional-run /workspace/scratch/ks_improvement_20261105/run_v1 \
  --closed-session-receipt /data/ks_improvement_20261105/operations/closed_sessions_after_97432.json \
  --minimum-mib 0.0625 --include-fit-receipts \
  --max-retries 6 --execute
```

The command has its own nonblocking scheduler lock and never creates duplicate
writers. Completed immutable terminal receipts and compatible fits/origin scores
are reused. The launcher can advise the operating system to release cached
pages of closed fit files after an observed child memory-admission failure;
it never deletes those files or weakens the admission rule. A finite retry limit
is operational, and does not renew the scientific time budget.

The wrapper defaults to `--stage all`. Its optional stage selector accepts
`all`, `precision`, `confirm`, or `sensitivity` and dispatches the unchanged
runner entrypoint. With a frozen selection already present, `--stage confirm`
skips the screening and promotion stage passes while rechecking the retained
promotion, precision, and frozen-selection records.
**`confirm` does not run seed sensitivity.** After that coordinator exits,
invoke the wrapper separately with `--stage sensitivity`, retaining the same
configuration, cache, output root, and original budget. Do not run the two
coordinators concurrently. A zero exit from either individual stage does not
certify completion of the other stage or of budget-limited pending work.

Two explicit options extend file-cache advice without changing that sequence:

| Option | Effect |
|---|---|
| `--minimum-mib 0.0625` | Includes eligible files of at least 64 KiB. The default remains 1 MiB; the supplied cutoff must be finite and at least 0.0625 MiB. |
| `--include-fit-receipts` | Also permits closed JSON files under the explicitly listed run roots' `fits/` directories. Without it, only NPZ and HDF5 files are eligible. |

The recorded run_v3 continuation above includes these cache-advice options.
After confirmation has exited, invoke `--stage sensitivity` using the same
active configuration, cache, and output root, with any required closure proof
refreshed to the actual latest coordinator. Every existing
scheduler and fit-writer lock must be acquired before inventory and advice.
Eligible files must be regular, owned by the current user, and reached without
symlinks. The tool checks file identity, flushes each existing file, and issues
`POSIX_FADV_DONTNEED`; it does not read, rewrite, or delete checkpoint or receipt
contents. JSON outside `fits/` remains outside this advice scope.

Recovery retries still require this launcher's own child to have exited with
the exact terminal memory-admission traceback. Initial advice using
`--reclaim-first` additionally requires genuine recorded coordinator-completion
observations covering every declared run root. After advice, the unchanged
memory and disk admission check must pass before any retry. The advice is a
kernel hint, so it cannot guarantee that pages are released.

The original executed campaign's four-hour elapsed scheduling budget starts at
`2026-11-05T07:38:04.272689+00:00`; its cumulative worker cap is 12 hours, with six
reserved for confirmation and subsequent diagnostics. Ordinary resuming preserves
the bound limits. One prospective operational extension, documented below,
raises total worker hours to 15 and elapsed hours to five while retaining the
six-hour search threshold, every spent attempt and that same original start.
These limits are checked before admitting another batch: already running jobs
can finish after a threshold is crossed. With at most four workers and a
1,800-second per-job timeout, the maximum additional in-flight allowance is
7,200 worker seconds, plus bounded shutdown overhead; wall overrun is at most
one job timeout plus shutdown overhead. The final report must show actual
consumption rather than describe admission checks as exact running-job cutoffs.
An expired scheduling budget deliberately leaves unfinished work pending.
Starting another wrapper invocation does not reset it.

### The single prospective budget decision and applied migration

The extension uses receipt timing and remaining-work estimates, not confirmation
performance scores. Its complete reason, authorization time, timing-receipt
SHA-256, original frozen-selection identity, previous-configuration SHA-256 and
verbatim previous budget are retained in
`ks_improvement_v2_budget.json` under `budget.operational_extension` and the
[public decision receipt](evidence/ks_improvement_budget_extension_decision.json).
The selected models and original selection timestamp are retained. No candidate,
prior, likelihood, numerical source, forecast schedule, score definition or
failure record is changed by this operational extension.

The migration described here has already been applied. The following command
documents its guarded procedure; it is **not a current continuation command**
and must not be applied again to the existing run_v3 destination. Any future
relocation must wait until the coordinator and every writer have actually exited.
It requires a fresh, genuine closure receipt for the latest observed coordinator;
neither a stale process number nor an invented receipt establishes quiescence.
The tool acquires the existing locks and rejects active writers. It preserves
the source root and uses a new destination; no history is reset in place.

```bash
"$KS_PYTHON" tools/extend_ks_improvement_budget.py \
  --source /data/ks_improvement_20261105/run_v2 \
  --destination /data/ks_improvement_20261105/run_v3 \
  --config configs/applications/ks_improvement_v2_budget.json \
  --source-cache-path /workspace/scratch/ks_recovery_20261105T0615/cache \
  --cache /data/ks_improvement_20261105/cache \
  --fit-sidecar-manifest /data/ks_improvement_20261105/operations/reviewed_fit_sidecars.json \
  --inference-sidecar-manifest /data/ks_improvement_20261105/operations/reviewed_inference_sidecars.json \
  --closed-session-receipt FRESH_CLOSED_SESSION_RECEIPT \
  --closed-session-id OBSERVED_LATEST_SESSION_ID \
  --projection-receipt TIMING_RECEIPT \
  --decision-receipt docs/applications/evidence/ks_improvement_budget_extension_decision.json
```

Without `--apply`, the command is a metadata-only dry run after acquiring locks.
The recorded authorized migration used `--apply`; its passed receipt verifies
all carried attempt time, unchanged scientific content and retained artifact
bytes. Only the
operational experiment/job bindings and corresponding freeze identity change;
the original freeze time and selected rows do not. Old and new identities remain
linked by the migration receipt.

`--source-cache-path` names the old path recorded in source jobs; it does not
require recreating or reading a second cache there. `--cache` supplies the
restored, independently verified cache. The two sidecar manifests are exact
reviewed allowlists, not general permission to ignore unexpected files. The
fit-sidecar manifest covers six reviewed opaque ancillary files. The inference
manifest covers only the separately validated historical pair described next;
unknown additional sidecars remain rejected.

#### One historical auxiliary sampler pair

The strict migration check identified one historical mismatch for unselected
candidate `bayes_02`, track A, trajectory 0, seed 20261105, fit key
`13103f3581fde7430a4128f52b3be460`. Its completed `fit.npz` independently passes
the original source/runtime, configuration, receipt and internal checksum
checks, contains four chains with 2,000 retained states each, and retains its
recorded inference qualification. Its auxiliary chain file instead contains
2,000, 2,000, 2,000 and 1,750 retained states, while the phase file declares the
completed target and a different chain checksum.

The [read-only prefix validation](evidence/ks_improvement_native_auxiliary_validation_13103f.json)
verified that **all 7,750 retained auxiliary state arrays and their log-density
prefixes match the completed native posterior exactly**, with zero mismatches.
The earlier migration receipt already recorded the same bytes. The creator and
cause of the auxiliary mismatch remain unproven. This verification performed no
model fit, forecast or extra MCMC draws.

The applied migration preserves this pair as **archive-only evidence outside
active fit storage**, under
`run_v3/migration/source_records/inference_sidecars/13103f3581fde7430a4128f52b3be460/`,
and retains the validated completed native checkpoint unchanged. Its passed
receipt records `inference_restart_state_available: false` and
`qualification_changed: false`. Exact completed inference restart from this auxiliary pair
is unavailable; no final RNG state is invented and no partial chain is relabeled
as complete. Existing completed native forecasting state, selected models,
qualification, scores and failures are unchanged. The proof receipt's SHA-256 is
`594e82111eee09987002388c5ffdeb16413569dfbd452578963b9feec4fb09c3`.

The current continuation uses
`--config configs/applications/ks_improvement_v2_budget.json` and
`--output /data/ks_improvement_20261105/run_v3` and the restored
`--cache /data/ks_improvement_20261105/cache`. Use
`--stage confirm` and, after its coordinator exits, `--stage sensitivity` as
documented above. Keep every historical root included in any bounded cache-advice
inventory covered by genuine closure evidence. The new elapsed deadline is five
hours after the original start, not five additional hours from migration. The
final report derives active limits from the recorded budget and shows both the
original and extended policy.

Historical v1/run_v2 status, without scheduling fits:

```bash
/workspace/scratch/ks_recovery_20261105T0615/env/bin/python \
  -m bayes_stesn.applications.ks_improvement_runner status \
  --config configs/applications/ks_improvement_v1.json \
  --cache /workspace/scratch/ks_recovery_20261105T0615/cache \
  --output /data/ks_improvement_20261105/run_v2
```

`terminal/*.json` is the authoritative job completion record. `ledger/` is a
mutable status view. `status.json` reports actual completed, failed,
inference-unqualified, and pending work; a process exit code alone does not
establish that the study finished. See the retained
[operational recovery receipt](evidence/ks_improvement_preflight_status_recovery.json)
for the original run's status-file inconsistency and its bounded recovery.

## Independent reproduction from the beginning

A new reproduction needs a new output root and a new dated configuration. The
historical configuration's elapsed-time start cannot serve as a future run's
start. Copy the original v1 specification prospectively before any new fit,
preserving its scientific registry, seeds, selection rule and schedules, and
declare a fresh allocation of **15 total worker hours, five elapsed hours and
six search worker hours**. This is the initial allocation of an independent
reproduction, not another extension of the historical run. Starting from v1
avoids carrying the historical `operational_extension` record into a new study.
The example refuses to replace an existing configuration:

```bash
python - <<'PY'
import datetime, json
from pathlib import Path
source = Path('configs/applications/ks_improvement_v1.json')
destination = Path('/data/ks-improvement-reproduction/config.json')
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.exists():
    raise FileExistsError('Use the existing configuration to resume; do not reset its budget')
config = json.loads(source.read_text())
config['name'] = 'ks_bayesian_improvement_independent_15workerh_5elapsedh'
config['budget'].update(
    campaign_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    total_worker_seconds_cap=15 * 3600,
    wall_seconds_cap=5 * 3600,
    search_worker_seconds_cap=6 * 3600,
    initial_worker_seconds=15 * 3600,
    preparation_worker_seconds=0.0,
    extension_reason=(
        'Prospective initial allocation for an independent reproduction: '
        '15 worker hours, 5 elapsed hours and 6 search worker hours, based on '
        'the recorded campaign cost projection; no historical clock reset or extension.'
    ),
)
destination.write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
PY

/data/ks-improvement-env/bin/python -m bayes_stesn.applications.ks_improvement_runner preflight \
  --config /data/ks-improvement-reproduction/config.json \
  --cache CACHE_ROOT \
  --output /data/ks-improvement-reproduction/run

/data/ks-improvement-env/bin/python tools/run_ks_improvement_bounded.py \
  --config /data/ks-improvement-reproduction/config.json \
  --cache CACHE_ROOT \
  --output /data/ks-improvement-reproduction/run \
  --minimum-mib 0.0625 --include-fit-receipts \
  --max-retries 6 --execute
```

The first command creates and binds the fresh experiment root. The bounded
launcher requires that existing root and does not invent a new experiment or
reset its start time. An additional historical run root requires a genuine
closed-session observation receipt as well as acquiring all existing locks;
do not fabricate that receipt for an unobserved process.

This fresh configuration begins with no separately incurred preparation time;
if new resource profiling is performed before launch, record its measured time
prospectively in this new configuration. Historical preparation time is not
claimed as work performed by the independent reproduction. The original v1/v2
configuration files and all existing run roots remain unchanged. An existing
independent run must resume with its already bound configuration and start.

Before doing this, inspect actual CPU quota, container memory, free disk and
other jobs. The recorded resource admission values are feasible measurements
for the executed environment, not promises for every machine. A required budget
or resource-policy change must be recorded prospectively in that new version.

The driver also exposes individual stages, with the same `--config`, `--cache`,
and `--output` arguments: `preflight`, `screen`, `promote`, `precision`, `freeze`,
`confirm`, and `sensitivity`. It automatically executes required earlier stages
and reuses matching completed work. Do not run a second coordinator concurrently.

## Targeted verification

```bash
PYTHONPATH=src /data/ks-improvement-env/bin/python -m pytest -q \
  tests/test_ks_improvement_candidates.py \
  tests/test_ks_improvement_diagnostics.py \
  tests/test_ks_improvement_runner.py \
  tests/test_ks_improvement_bounded.py
```

These are small acceptance fixtures for candidate identity, scientific
boundaries, diagnostic interventions, selection/resume behavior, and the cache
launcher. They do not replace actual chain diagnostics, forecast qualification,
or completed scientific results.

The native-backup and animation contracts have a separate, small fixture suite:

```bash
PYTHONPATH=src /data/ks-improvement-env/bin/python -m pytest -q \
  tests/test_ks_improvement_fits.py \
  tests/test_ks_improvement_animation.py
```

These 14 checks use synthetic checkpoint/geometry files. They do not open a real
fit, run MCMC, produce new forecasts, package the live campaign, or render movies.

## Evidence and reporting

Keep bulk data, posterior checkpoints, full run logs, and private retrieval
manifests outside Git. Keep the compact configuration, implementation, source
identities, diagnostic summaries, metrics, selected examples, and report in the
application-results area. Large checkpoint snapshots need their own checksum
manifest and authorized private storage.

A report from a running campaign must state its partial status and preserve the
expected denominators. Inference failures are never replaced by point forecasts
or removed from those counts. A process that exits cleanly may still have
budget-limited pending jobs; retain the recorded status and coverage counts.

### Compact evidence snapshots

The compact package captures a fixed frontier of immutable completed, failed,
and inference-unqualified terminal receipts, their bound jobs and forecast
scores, source/configuration identities, available history/selection/freeze
records, closed attempts, migration evidence, and the original diagnostic
summary. It excludes mutable status ledgers, live attempts, locks, raw data,
and native/posterior checkpoints. **It is evidence storage, not numerical
resumability by itself.** A later immutable terminal receipt is deferred to the
next snapshot, so compact snapshots can be made while the campaign is running.

For the recorded workspace, use its compatible Python and the thread settings
above. Set these task-specific paths once:

```bash
KS_PYTHON=/data/ks_improvement_20261105/env/bin/python
KS_CACHE_ROOT=/data/ks_improvement_20261105/cache
KS_ORIGINAL_FREEZE=/data/ks_improvement_20261105/recovered_original_evidence/ks_frozen_specification_full.json
# Active destination; its guarded migration receipt passed on 2026-11-05:
KS_RUN_ROOT=/data/ks_improvement_20261105/run_v3
KS_EXPORT_ROOT=/data/ks_improvement_20261105/exports
KS_REPORT_ROOT=/data/ks_improvement_20261105/report

"$KS_PYTHON" tools/package_ks_improvement_checkpoint.py create \
  --run-root "$KS_RUN_ROOT" \
  --destination "$KS_EXPORT_ROOT" \
  --diagnostic-summary docs/applications/evidence/ks_improvement_initial_diagnosis.json
```

The command prints the unique package directory and `manifest` path. Set
`KS_COMPACT_MANIFEST` to that exact returned path, then verify it:

```bash
KS_COMPACT_MANIFEST=/absolute/path/to/the/returned/manifest.json
"$KS_PYTHON" tools/package_ks_improvement_checkpoint.py verify \
  --manifest "$KS_COMPACT_MANIFEST"
```

### Native fit and sampler checkpoint backups

`tools/package_ks_improvement_fits.py plan` reads small experiment/job/freeze
metadata and stats canonical binary files; it does not open checkpoint payloads
and can be used during execution. Its byte counts are provisional until writers
stop. Selection must be frozen before the selected-only subset is meaningful.

```bash
"$KS_PYTHON" tools/package_ks_improvement_fits.py plan \
  --run-root "$KS_RUN_ROOT" --selected-only

"$KS_PYTHON" tools/package_ks_improvement_fits.py plan \
  --run-root "$KS_RUN_ROOT" --all-fit-checkpoints
```

Create a native package **after the coordinator and fit workers have exited**.
The tool must acquire the existing scheduler lock and every fit writer lock; it
does not create substitute locks. A busy, missing, or incompatible lock stops
the operation before checkpoint payload validation. Run the compact creation
command again after this exit and use its **fresh manifest**. The native tool
rejects stale experiment/history/freeze bindings or a changed terminal frontier.
Locks establish quiescence; the manifests separately disclose missing fits and
inference qualification rather than assuming every requested job completed.

Choose the scope needed for the private backup:

| Mode | Included native state | Resume meaning |
|---|---|---|
| `--selected-only` | Receipted `fit.npz` files for frozen main, anchor, and selected prior-sensitivity configurations, across available trajectory/model-seed fits; original receipts, synchronization records, bound jobs, and compact evidence | Completed forecast state, including declared seed-sensitivity fits when present; sampler sidecars intentionally excluded |
| `--all-fit-checkpoints` | Every unique closed canonical `fit.npz`, `fit.npz.chains.npz`, and `fit.npz.chains.npz.ks-fit-phase.npz` available in this run, with receipts/provenance and compact evidence | Completed forecasts plus retained exact sampler RNG/proposal/phase state for interrupted inference, subject to the original continuation allowance |

For a selected-fit package:

```bash
"$KS_PYTHON" tools/package_ks_improvement_fits.py create \
  --run-root "$KS_RUN_ROOT" \
  --output "$KS_EXPORT_ROOT/ks_native_selected_final" \
  --compact-manifest "$KS_COMPACT_MANIFEST" \
  --selected-only --part-mib 512 --apply
```

For the broader package, use this alternative. It already includes the selected
files, so making both packages is unnecessary unless separate delivery subsets
are useful:

```bash
"$KS_PYTHON" tools/package_ks_improvement_fits.py create \
  --run-root "$KS_RUN_ROOT" \
  --output "$KS_EXPORT_ROOT/ks_native_all_final" \
  --compact-manifest "$KS_COMPACT_MANIFEST" \
  --all-fit-checkpoints --part-mib 512 --apply
```

`--apply` is the tool's explicit execution flag. The output directory must be
new and outside the run. There is no force-overwrite mode. The tool verifies
current receipt hashes, native internal state checksums, ZIP CRCs, and
data/configuration/source/runtime identities; chain/phase checks preserve the
original bounded draw allowance and qualification. It writes independent ZIP
segments capped at 512 MiB, a per-member/per-part SHA-256 manifest, verification
receipt, standalone restore helper, and restore instructions. Already compressed
NPZ files are stored unchanged without a giant intermediate archive. A single
member must fit within the requested segment cap; the stat plan reports its
minimum size. Original data/checkpoints are retained, and creation preserves at
least the recorded disk reserve or 2 GiB, whichever is larger.

The native serializer embeds the permitted training panel, derived features,
frozen basis and posterior because these are needed by the accepted forecast
implementation. The separate raw/canonical KS cache and environment are not
included. An inference-unqualified completed fit remains unqualified; archiving
its chain/phase state does not authorize additional draws or a changed prior.
The compact archive is included under `evidence/` with its own verified manifest.

### Verify and restore a native package

Keep all numbered parts, `manifest.json`, `manifest.sha256`,
`restore_ks_improvement_fits.py`, and `RESTORE.md` together. For example, for the
selected package:

```bash
KS_PACKAGE="$KS_EXPORT_ROOT/ks_native_selected_final"
"$KS_PYTHON" "$KS_PACKAGE/restore_ks_improvement_fits.py" verify \
  --package "$KS_PACKAGE"

"$KS_PYTHON" "$KS_PACKAGE/restore_ks_improvement_fits.py" restore \
  --package "$KS_PACKAGE" \
  --destination /data/ks_improvement_restored_native
```

Verification and extraction need only the Python standard library. Restore
checks every part/member SHA-256 and ZIP CRC, rejects unsafe paths, preserves
the recorded disk reserve, and writes atomically to a **new destination**. It
does not merge with or overwrite an existing run. Extract the nested compact
evidence ZIP separately for inspection if required.

For forecast reuse, supply the original verified KS data, protocol, trajectory,
configuration, seed, and compatible numerical runtime to the unchanged
`fit_native('bayes_stesn', training.panel, config, seed,
checkpoint_path=RESTORED_FIT_PATH)`. Its existing identity guard decides whether
the completed checkpoint is reusable. When native `fit.npz` is absent, retain
the canonical chain and phase filenames beside that intended path so the same
adapter can resume the recorded inference phase within its original allowance.
Some phase-only files contain no saved chain progress; the manifest distinguishes
these cases. The original sampler rechecks its transition signature against the
reconstructed prepared model.

Archived job JSON contains original absolute paths. **Restored worker jobs are
not directly relocation-resumable.** A coordinator in another root needs an
explicit compatible import with new job identities and preserved cumulative
wall/worker accounting. Neither this restorer nor a new wrapper invocation
rewrites those paths or renews an exhausted campaign budget.

### Reassemble the 96 MiB transport edition before native restore

The selected-native backup also has a transport edition for the storage
connector's per-object limit. Its schema is `ks-native-package-transport/1`.
The transport layout contains **88 uniquely named objects**: 79 binary fragments
that reconstruct the original 14 canonical ZIP parts, five original package
metadata/helper files, and four transport manifest/helper files. Preserve every
filename when downloading the authorized backup into one directory. Use the
final transfer receipt to confirm all objects are available and verified;
creating the local transport layout alone does not prove remote upload completion.

The transport manifest SHA-256 is
`82aa2042ca028e4c023a234912bb0a897c6781590509e3029c93beddd08e6e41`.
The supplied checksum sidecar binds the manifest only, so also verify the
transport helper against its separately recorded hash before executing it.
First verify both files, then reconstruct the canonical package into a
**fresh destination**. The transport restorer checks every fragment and each
assembled canonical file; it does not fit a model or reinterpret an NPZ payload.

```bash
cd /data/ks_downloaded_native_transport
sha256sum -c KS_improvement_native_FINAL_20261105_transport_manifest.sha256
sha256sum -c <<'KS_TRANSPORT_SHA256'
82aa2042ca028e4c023a234912bb0a897c6781590509e3029c93beddd08e6e41  KS_improvement_native_FINAL_20261105_transport_manifest.json
ddfb67c377522b42fbbc8b0bfc3580a2bcaa9fcb60a45aeeb274c779969d7578  KS_improvement_native_FINAL_20261105_restore_native_transport.py
KS_TRANSPORT_SHA256

python KS_improvement_native_FINAL_20261105_restore_native_transport.py restore \
  --manifest KS_improvement_native_FINAL_20261105_transport_manifest.json \
  --destination /data/ks_restored_native_package
```

The reconstructed canonical `manifest.json` must retain SHA-256
`c001bba3d223fbb887f663b01b29fb8951acc38a1e7cea624e063324b67f96bb`.
Verify that package and then restore its members into another **fresh
destination**:

```bash
python /data/ks_restored_native_package/restore_ks_improvement_fits.py verify \
  --package /data/ks_restored_native_package

python /data/ks_restored_native_package/restore_ks_improvement_fits.py restore \
  --package /data/ks_restored_native_package \
  --destination /data/ks_restored_selected_fits
```

This selected edition preserves 228 completed native forecast states and their
bound evidence. It is not a mid-chain RNG backup, and it does not make archived
absolute-path jobs directly relocation-resumable. The unchanged native
data/configuration/source/runtime checks described above still govern reuse.
Keep the transport originals until the assembled canonical package and restored
members have both passed verification.

### Reports and fixed-case animations

For a partial native-only snapshot, build the report from retained receipts into
a location outside the numerical run root:

```bash
"$KS_PYTHON" tools/build_ks_improvement_report.py \
  --results "$KS_RUN_ROOT" \
  --output "$KS_REPORT_ROOT" \
  --operations /data/ks_improvement_20261105/operations/bounded_launches
```

For the final integrated report, set the following variables to the **actual
completed** fixed-reference output and selected-artifact acceptance receipt,
then include both arguments. Completion means every scheduled job is terminal;
observed numerical or inference failures must remain visible. Do not create
placeholder files or omit an unfinished auxiliary schedule to describe the
whole requested application as complete.

```bash
KS_ANCHOR_ROOT=/data/ks_improvement_20261105/fixed_references
KS_ACCEPTANCE_RECEIPT=/data/ks_improvement_20261105/acceptance/selected_invariance.json
KS_REPORT_ROOT=/data/ks_improvement_20261105/repo/docs/applications/improvement_results

"$KS_PYTHON" tools/build_ks_improvement_report.py \
  --results "$KS_RUN_ROOT" \
  --output "$KS_REPORT_ROOT" \
  --operations /data/ks_improvement_20261105/operations/bounded_launches \
  --anchors "$KS_ANCHOR_ROOT" \
  --acceptance "$KS_ACCEPTANCE_RECEIPT"
```

The auxiliary variables above name this recorded campaign's completed artifacts.
For an independent reproduction, replace them with that run's actual bound
outputs. The recorded campaign's `KS_RUN_ROOT` is its verified terminal run_v3
destination; run_v2 remains the historical source. The
report's `source_manifest.json` and
`artifact_manifest.json` bind the numerical receipts and retained examples used
by the animation tool. Generating a report does not fit, forecast, or rerun
selection.

After the reporting changes have been committed and the final compact report
has been generated, render its human-readable PDF with the required revision
argument:

```bash
KS_REPORT_REVISION=$(git rev-parse HEAD)
"$KS_PYTHON" tools/build_ks_improvement_pdf.py \
  --report "$KS_REPORT_ROOT" \
  --output "$KS_REPORT_ROOT/ks_improvement_report.pdf" \
  --revision "$KS_REPORT_REVISION"
```

The renderer creates a missing output parent, checks every compact artifact's
manifest entry, and verifies the temporary PDF before publishing it by atomic
rename. Its separate `ks_improvement_report.manifest.json` records the exact
renderer and PDF hashes, page count, source revision and report-manifest hash.
Structural verification is not visual review: its `visual_qa` field remains
`pending` until a separate receipt documents inspection of the actual pages.

With the frozen selection and report-bound example arrays present, render the
three tracks separately in compact MP4s:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD/src" MPLBACKEND=Agg
"$KS_PYTHON" tools/animate_ks_improvement.py \
  --results "$KS_RUN_ROOT" \
  --report "$KS_REPORT_ROOT" \
  --output "$KS_REPORT_ROOT/animations" \
  --tracks A B C --fps 10
```

The tool uses existing `ffmpeg` and `ffprobe`; it does not install codecs. The
output must be outside the numerical run root. The fixed example is trajectory
1, validation first target 6000, horizon 100, comparing the old selected anchor
with the new preserved-prior main selection. It reads only saved example arrays
and their report/freeze bindings: no dataset, posterior checkpoint, new forecast,
or selection access is needed. A/B/C remain separate protocols, with 13/64/12
validation target sites and 39/64/40 input sites respectively. Withheld target
sites and permitted input sites are marked; the display does not interpolate
unscored locations.

Each movie contains all 100 frames at 1280 by 720 pixels using H.264/yuv420p.
At 10 fps its playback lasts 10 seconds; the forecast spans 25 physical time
units. `--fps 8` is the other supported playback rate. The physical amplitude
scale is shared across the two panels; extreme finite forecasts use an explicitly
labeled symmetric log scale rather than clipping. Missing cases remain missing
and are recorded in `animation_receipt.json`; an entirely missing track gets no
empty movie. Matching completed movies are checksum-verified and reused.
Incompatible outputs require a new directory; the tool does not overwrite them
or substitute a more favorable example.

### Final artifact review remains a separate required step

The [pre-render QA receipt](evidence/ks_improvement_report_pre_render_qa.json)
covers its stated source, synthetic checks and pre-freeze search-figure review.
It explicitly leaves **final populated artifact QA pending**. Neither that
receipt nor a successful renderer exit certifies the final tables or layout.
Generate the final compact report first, then its PDF and movies; rebuilding
the compact report changes the bound animation identity, so use a new animation
output directory if a prior render no longer matches.

Before final delivery, record a separate review of:

- Every PDF page, including table continuation headers, complete/failure
  denominators, fixed point-reference labels, physical lead times, and the
  distinction between confirmation and exploratory selection diagnostics.
- The populated horizon, calibration, width, compute, heatmap and interval
  figures, retaining extreme finite values and clearly identifying the official
  validation sites and permitted input sites.
- The main-versus-alternative seed comparisons on their identical trajectory
  rosters, with failed, unqualified and pending outcomes retained.
- Each actual encoded movie's first, middle and final frames, as well as its
  recorded codec, resolution, 100-frame count and playback duration. The movie
  renderer already checks those technical properties but does not claim visual
  inspection.

Bind that final review to the exact PDF, figure and animation hashes. Temporary
page/frame images used for inspection are QA working files, not a redundant
scientific output archive.
