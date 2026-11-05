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
source and configuration unchanged while resuming. The actual running workspace
used the following v1-budget/run_v2 paths; substitute only equivalent verified roots when moving
the complete workspace, using the explicit migration tool if job output paths
must change. After the separately documented budget-only migration below, use
the v2 budget configuration with the new run_v3 root. Do not point a changed
configuration at a root bound to the original experiment identity.

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

For example, add `--stage confirm --minimum-mib 0.0625
--include-fit-receipts` to the resume command above, then use the same command
with `--stage sensitivity` after confirmation has exited. Every existing
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

### One prospective budget-only extension

The extension uses receipt timing and remaining-work estimates, not confirmation
performance scores. Its complete reason, authorization time, timing-receipt
SHA-256, original frozen-selection identity, previous-configuration SHA-256 and
verbatim previous budget are retained in
`ks_improvement_v2_budget.json` under `budget.operational_extension` and the
[public decision receipt](evidence/ks_improvement_budget_extension_decision.json).
The selected models and original selection timestamp are retained. No candidate,
prior, likelihood, numerical source, forecast schedule, score definition or
failure record is changed by this operational extension.

Migration must wait until the coordinator and every writer have actually exited.
Supply a fresh, genuine closure receipt for the latest observed coordinator;
neither a stale process number nor an invented receipt establishes quiescence.
The tool acquires the existing locks and rejects active writers. It preserves
the source root and uses a new destination; no history is reset in place.

```bash
python tools/extend_ks_improvement_budget.py \
  --source /data/ks_improvement_20261105/run_v2 \
  --destination /data/ks_improvement_20261105/run_v3 \
  --config configs/applications/ks_improvement_v2_budget.json \
  --cache /workspace/scratch/ks_recovery_20261105T0615/cache \
  --closed-session-receipt FRESH_CLOSED_SESSION_RECEIPT \
  --closed-session-id OBSERVED_LATEST_SESSION_ID \
  --projection-receipt TIMING_RECEIPT \
  --decision-receipt docs/applications/evidence/ks_improvement_budget_extension_decision.json
```

The command above is a metadata-only dry run after acquiring locks. Add
`--apply` to perform the authorized migration. Its receipt must verify all carried
attempt time, unchanged scientific content and retained artifact bytes. Only the
operational experiment/job bindings and corresponding freeze identity change;
the original freeze time and selected rows do not. Old and new identities remain
linked by the migration receipt.

After the destination receipt passes, resume with
`--config configs/applications/ks_improvement_v2_budget.json` and
`--output /data/ks_improvement_20261105/run_v3`, retaining the same cache. Use
`--stage confirm` and, after its coordinator exits, `--stage sensitivity` as
documented above. Keep every historical root included in any bounded cache-advice
inventory covered by genuine closure evidence. The new elapsed deadline is five
hours after the original start, not five additional hours from migration. The
final report derives active limits from the recorded budget and shows both the
original and extended policy.

For status without scheduling fits:

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
start. Copy it prospectively before any new fit, preserving its scientific
registry, seeds, selection rule, schedules, and finite limits. For example:

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
config['name'] = 'ks_bayesian_improvement_independent_reproduction'
config['budget']['campaign_started_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
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
  --max-retries 6 --execute
```

The first command creates and binds the fresh experiment root. The bounded
launcher requires that existing root and does not invent a new experiment or
reset its start time. An additional historical run root requires a genuine
closed-session observation receipt as well as acquiring all existing locks;
do not fabricate that receipt for an unobserved process.

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
KS_PYTHON=/workspace/scratch/ks_recovery_20261105T0615/env/bin/python
KS_RUN_ROOT=/data/ks_improvement_20261105/run_v2
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

### Reports and fixed-case animations

Build the report from retained receipts into a location outside the numerical
run root:

```bash
"$KS_PYTHON" tools/build_ks_improvement_report.py \
  --results "$KS_RUN_ROOT" \
  --output "$KS_REPORT_ROOT" \
  --operations /data/ks_improvement_20261105/operations/bounded_launches
```

The report builder also accepts `--anchors PATH` for the separately generated
cheap-anchor results and `--acceptance PATH` for a completed acceptance receipt.
Pass these only when those artifacts exist. Its `source_manifest.json` and
`artifact_manifest.json` bind the numerical receipts and retained examples used
by the animation tool. Generating a report does not fit, forecast, or rerun
selection.

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
