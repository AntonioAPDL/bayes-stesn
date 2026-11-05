# KS private results backup and restore

The standalone [package_ks_results.py](../../tools/package_ks_results.py) creates a single gzip-compressed tar stream and writes it directly into numbered binary parts of at most **32 MiB each**. It never creates a second, unsplit archive. It also provides read-only verification and restore into a new destination. Python 3.11 or newer is sufficient; the backup and restore tool itself requires no numerical or deep-learning packages.

This tool is prepared for the final private backup. Its acceptance tests use small synthetic fixtures. **The running KS study was not packaged during those tests.** Run it only after all workers have stopped and the final reports are finished.

## Included and excluded content

The package includes regular result files at the results root and the declared result directories: `fits`, `forecasts`, `ledger`, `jobs`, `attempts`, `selection`, `precision_cases`, `predictions`, `report`, `reports`, `figures`, `plots`, `animations`, `tables`, `logs`, `diagnostics`, `evidence`, `audits`, `cleanup`, `artifacts`, and `observer_sessions`. The actual final-report destination is `report` (singular); its nested figures, animations, tables and PDF are included recursively. The plural alias remains supported for separately placed report material.

This retains canonical fitted models, Bayesian chain and phase checkpoints, neural optimizer checkpoints, model receipts, forecasts and scores, selected predictions, precision assessments, scheduling records, selection/freeze identities, figures, animations, and reports that actually reside inside the results root. Reports saved elsewhere must be backed up separately; the packager does not follow links to retrieve them.

Scheduler stdout and admission-pause tracebacks may reside in workspace `private/main*.log` files, while the per-worker files under the results `logs/` directory can be empty. Those external driver logs are not automatically included. After the scheduler and all relevant writers have stopped, copy the useful immutable driver logs into results `logs/` with distinct `driver_*` filenames, and place any required operational receipts inside a declared result directory. Do not copy active logs into a purported final snapshot. The original logs remain intact, and private backup contents may retain their original workspace paths.

Locks, recognizable staging files, temporary files, hidden paths, symbolic links, nonregular files, and source/environment binaries are excluded. Unknown top-level directories are excluded and listed in the manifest. The repository, data/cache root, Python environment, original data archives, and unrelated simulation results are outside this backup. Source code and the compatible numerical requirements remain in the versioned repository; the verified KS cache must be restored separately when computation is needed.

The archive is a **private snapshot**. Archived jobs and logs can contain original absolute workspace paths. The tool preserves their bytes and hashes rather than rewriting scientific provenance. Public reports should continue to use the separately redacted evidence.

## Preconditions and disk budget

Creation acquires the run's existing scheduler lock and every existing fit-writer lock nonblockingly. It checks actual visible worker command lines and job records using outer process IDs, without interpreting an archived PID token as a liveness guarantee. It refuses an active scheduler or worker, a pending/running job ledger, or a summary with pending fits/origins. Failed scientific fits and inference-unqualified fits remain valid completed evidence and are retained.

The experiment, freeze, test-access gate and summary must agree. The experiment and freeze identity hashes are reconstructed. Both main tracks must be represented in the summary. Canonical fitted-model hashes are checked against their original terminal fit receipts during streaming.

The required disk reserve is the larger of the experiment's existing reserve, any larger `--reserve-bytes` value, and 2 GiB. It cannot be lowered by a command-line option. Preflight requires space for a conservative compressed-size bound plus that reserve, while the original results remain intact. Quota-aware available blocks are read with `statvfs`, and the reserve is checked again during output writes. NPZ checkpoints are already compressed, so substantial further compression should not be assumed.

Only one part file is open for writing at a time. The tool writes in bounded chunks and does not hold a part or the complete archive in RAM. It holds file metadata and the manifest in memory. Source inode, size and timestamp identities are checked before/after each member and against a second inventory, so concurrent report changes cause creation to fail instead of silently mixing snapshots.

## Create and verify the final package

Activate the project environment or use any compatible Python 3.11+ executable. Choose a **new package directory outside the results root**:

```bash
KS_CODE_REVISION=$(git rev-parse HEAD)

python tools/package_ks_results.py create \
  --results "$KS_WORKSPACE/results/ks_main" \
  --output "$KS_WORKSPACE/backups/ks_main_final_parts" \
  --part-mib 32 \
  --code-revision "$KS_CODE_REVISION"
```

Creation automatically performs a separate, read-only verification pass before returning `complete_verified`. The default gzip level is 1 to avoid spending substantial CPU recompressing NPZ files. A smaller `--part-mib` value between 1 and 32 is supported if a storage service has a lower per-upload limit.

The output directory contains:

- `ks-results.tar.gz.part000001`, `...000002`, and subsequent contiguous numbered parts.
- `manifest.json` and its detached `manifest.sha256` checksum.
- `restore_ks_results.py`, a byte-identical standalone copy of the tool.
- `verification.json` and `package_state.json`.

The manifest records every archive member's relative name, bytes, SHA-256, original modification time and source mode; every part's bytes and SHA-256; the complete concatenated compressed-stream bytes and SHA-256; excluded paths and reasons; and code, experiment, freeze, numerical-source and runtime identities. It also records whether the code working tree was dirty. A Git commit alone is not a substitute for the recorded source hashes when uncommitted work exists.

Upload **all files in the completed package directory**, including the manifest, checksum and helper, to the authorized private backup folder. Binary parts must remain binary; do not paste or encode them into a conversation. Individual uploads can be retried independently using their recorded hashes. No new public share or account credentials are required by this tool.

An independent verification can be run at any time:

```bash
python tools/package_ks_results.py verify \
  --package "$KS_WORKSPACE/backups/ks_main_final_parts"
```

The verification command writes nothing. It checks the manifest and helper hashes, numbered part sizes and hashes, full compressed-stream hash, gzip integrity, archive completeness/order, every regular member's content hash, and the study identities. It rejects traversal paths, links, sparse/device members, duplicate/unexpected members, missing parts, and corrupted data. Hashes establish integrity relative to the retained manifest; they are not an independent digital signature.

## Restore without an unsplit intermediate archive

Download the complete package directory, then use its included helper:

```bash
python restore_ks_results.py verify --package /data/ks_backup_parts

python restore_ks_results.py restore \
  --package /data/ks_backup_parts \
  --destination /data/ks_results_restored
```

The destination must not exist. There is no force-overwrite or merge option. Restore first verifies the entire package without writing the destination, then streams through it again while extracting and checking each member. Output files are private regular files; archive ownership, executable permissions, and special-file metadata are not applied. Original file content and modification times are retained. A failed extraction leaves a clearly incomplete restore marker; it is not a validated run. A successful restore records its experiment and freeze identities without rewriting any job or checkpoint.

Do not concatenate the parts into another large archive merely to verify or restore them. Both operations read the parts as one stream.

## Relocation and checkpoint reuse

The backup preserves artifact bytes, not a complete executable machine environment. The delivered readers, verified data cache and compatible source/runtime must be available separately for forecasting or continuation.

The runner's experiment identity binds its configuration and numerical-source/runtime fields. Native Bayesian fitting/recovery also has its own model, data, configuration, sampler and computation identity. **Restoring a file does not waive these gates.** A different implementation or incompatible numerical runtime needs a new versioned run; do not edit hashes or receipts to make a checkpoint appear compatible. Loading a portable model directly is not, by itself, proof of an identical execution environment or bitwise reproducibility on another platform.

The current runner contains these path dependencies:

| Artifact or identifier | Relocation behavior |
|---|---|
| Fitted-model identity | Derived from model, track, training/data identity, configuration, seed and numerical identity; the results-root path is not a field. |
| Forecast cache key | Derived from fit identity, origin schedule, history/horizon/path budget and forecast/scoring policy; it does not depend on the results-root path. |
| Job JSON and job ID | Job JSON stores absolute `output` and `cache` paths, and the job hash includes those fields. Moving the run changes newly generated job IDs. |
| Precision descriptors | They retain absolute fitted-artifact paths and hashes. Their audit identities are not rewritten during restore. |
| Frozen selection | Preserved exactly. The runner seals selection after test access; it must not be rerun during restoration. |

Do **not** execute an archived `worker --job ...` command blindly after relocation: its JSON can still point to the original output/cache paths. Restore to the original authorized absolute locations when exact job-ledger continuation is required. At a new location, use the same compatible runner/configuration with explicit new `--output` and `--cache` arguments so it generates current job metadata. Matching canonical fits and forecast receipts can then be reused, but new job IDs/ledger entries can be created. Old precision paths remain historical evidence, and a frozen completed study does not require repeating its precision assessment or selection.

For a completed study, read-only reporting from the restored metrics and predictions is usually sufficient. The backup tool intentionally provides no automatic path migration, environment recreation, sampler restart, or checkpoint merge.

## Acceptance evidence

`tests/test_ks_package.py` has four targeted tests using synthetic results only; measured execution receipts are in `docs/applications/evidence/ks_package_acceptance.json`. They exercise a multi-part stream with 1 MiB parts, complete verify/restore and byte equality, inclusion of the actual singular `report` directory and durable observer sessions, exclusion of staging/locks/symlinks/cache data, unchanged archived absolute job paths, refusal to overwrite an existing package/destination, corruption detection before extraction, path-traversal rejection, and scheduler/active-worker blocking. No final KS archive was produced during these checks.

If package creation is interrupted, completed part files and its bounded incomplete tail are retained with an incomplete package-state receipt. An incomplete package is not accepted by restore. Creation has no append/resume shortcut: choose a fresh output directory after resolving the blocker, and preserve the failed attempt until any redundant files can be reviewed safely. No numerical result or checkpoint in the source results root is deleted by packaging.
