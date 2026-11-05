# Reproducing and resuming the Bayesian KS improvement campaign

## Scope

This branch preserves the original KS application at revision
`2ecd8052ce2a44c9b4ed3862382594bd84598ad7`. The new driver reuses the original
data adapter, Bayesian likelihood and prior implementation, sampler, native
fit/checkpoint code, predictive path generator, and score definitions. Its
104-candidate registry and finite budget are in
[`ks_improvement_v1.json`](../../configs/applications/ks_improvement_v1.json).
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
uses the following paths; substitute only equivalent verified roots when moving
the complete workspace, using the explicit migration tool if job output paths
must change.

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
  --max-retries 6 --execute
```

The command has its own nonblocking scheduler lock and never creates duplicate
writers. Completed immutable terminal receipts and compatible fits/origin scores
are reused. The launcher can advise the operating system to release cached
pages of closed fit files after an observed child memory-admission failure;
it never deletes those files or weakens the admission rule. A finite retry limit
is operational, and does not renew the scientific time budget.

The original campaign's four-hour elapsed scheduling budget starts at
`2026-11-05T07:38:04.272689+00:00`; its cumulative worker cap is 12 hours, with six
reserved for confirmation and subsequent diagnostics. Resuming preserves those
limits. They are checked before admitting another batch: already running jobs
can finish after a threshold is crossed. With at most four workers and a
1,800-second per-job timeout, the maximum additional in-flight allowance is
7,200 worker seconds, plus bounded shutdown overhead; wall overrun is at most
one job timeout plus shutdown overhead. The final report must show actual
consumption rather than describe admission checks as exact running-job cutoffs.
An expired scheduling budget deliberately leaves unfinished work pending.
Starting another wrapper invocation does not reset it.

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

/data/ks-improvement-env/bin/python tools/run_ks_improvement_bounded.py \
  --config /data/ks-improvement-reproduction/config.json \
  --cache CACHE_ROOT \
  --output /data/ks-improvement-reproduction/run \
  --max-retries 6 --execute
```

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

## Evidence and reporting

Keep bulk data, posterior checkpoints, full run logs, and private retrieval
manifests outside Git. Keep the compact configuration, implementation, source
identities, diagnostic summaries, metrics, selected examples, and report in the
application-results area. Large checkpoint snapshots need their own checksum
manifest and authorized private storage.

The report-building and optional fixed-anchor commands are documented with their
delivered tools once those stages are complete. A report from a running campaign
must state its partial status and preserve the expected denominators. Inference
failures are never replaced by point forecasts or removed from those counts.
