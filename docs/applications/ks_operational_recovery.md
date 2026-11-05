# KS operational recovery and final staging cleanup

These procedures manage resource accounting and redundant task-created temporary
files. They do not change the frozen experiment, numerical source, inference
qualification, admission thresholds, or checkpoint identities.

## Memory-admission pause and observed recovery

The main scheduler paused between final Track A and Track B when its conservative
memory admission test rejected another pair of workers. The test credits only
half of inactive file cache and requires a 1.5 multiplier on the estimated
combined worker RSS. The initial stop message and the later resource snapshot
are different observations; their rounded allowances must not be conflated.

At the later recovery snapshot, container memory accounting was 7,700,688,896
bytes against an 8,589,934,592-byte limit. File-backed cache accounted for
7,240,908,800 bytes; anonymous memory was 98,856,960 bytes. After confirming no
live KS workers, the operator applied `POSIX_FADV_DONTNEED` to 504 existing task
files of at least 1 MiB, covering 4,167,321,724 logical bytes. It did not delete
files, alter their contents, or use a global cache-control operation.

The subsequent container reading was 3,417,071,616 bytes, file-backed cache was
3,072,589,824 bytes, and anonymous memory was 99,360,768 bytes. The unchanged
scheduler admitted resumed work. A subsequent independent snapshot recorded zero
`oom` and `oom_kill` events, terminal states for all 160 Track A final jobs, and
two running Track B final jobs with retained canonical fits and fit receipts.
The cgroup includes other activity, so these are observed accounting changes,
not a per-file attribution or a guaranteed effect of the advice.

Exact measurements and the resume census are in
[`evidence/ks_memory_recovery_20261105.json`](evidence/ks_memory_recovery_20261105.json).
The public report collector exposes this receipt as
`extra_acceptance.ks_memory_recovery_20261105`.

### Second pause: recorded memory-limit pressure

A second admission stop occurred at 04:44 UTC during Track A seed sensitivity,
after all eight additional GRU fits and the four fits for the first additional
Bayesian seed had completed. The required safety-adjusted estimate was
900,000,000 bytes, exceeding the then-current allowance of 473,687,552 bytes.
After confirming no live workers and acquiring the existing scheduler and fit
locks, the operator used the guarded cache-advice tool at 04:46:07 UTC. It advised
660 files covering 5,040,298,737 logical bytes, with zero errors, content changes,
or deletions. The frozen study resumed with the same memory gates; the resumed
invocation used a two-hour wall-time cap.

| Cgroup quantity | Before advice | After advice |
|---|---:|---:|
| Current accounted bytes | 8,583,749,632 | 3,525,160,960 |
| File-backed accounted bytes | 8,173,711,360 | 3,132,698,624 |
| Anonymous accounted bytes | 150,716,416 | 155,820,032 |
| Configured limit, bytes | 8,589,934,592 | 8,589,934,592 |
| Cumulative `memory.events:max` | 5,842 | 5,868 |
| Cumulative `memory.events:oom` | 0 | 0 |
| Cumulative `memory.events:oom_kill` | 0 | 0 |

This event involved actual memory-limit pressure: the before-advice accounting
was only 6,184,960 bytes below the limit. The recorded cgroup peak was
8,590,311,424 bytes. The zero OOM counters do not imply an absence of pressure.
The [Linux cgroup v2 documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html)
defines `max` as occasions when usage was about to exceed the maximum boundary,
separately from the OOM and process-kill counters; these event counters are
hierarchical. The before/after difference is not attributed exclusively to the
advice call or to the KS workers, because the cgroup includes other activity.

The independently checked resume snapshot retained 698 ledger entries: 686
complete, nine failed, one inference-unqualified, and two running. Previously
recorded scientific failures were preserved. All 51 experiment-bound numerical
Python files still matched their frozen hashes. The event-1 receipt is unchanged;
the appended event has its own evidence file,
[`evidence/ks_memory_recovery_20261105_event2.json`](evidence/ks_memory_recovery_20261105_event2.json),
exposed as `extra_acceptance.ks_memory_recovery_20261105_event2`. No additional
cache advice or deletion was performed by this documentation audit.

### Reusable guarded command

`tools/reclaim_ks_file_cache.py` defaults to a dry-run. It checks the existing
scheduler and fit writer locks and identifies actual workers through the outer
`/proc` view, rather than interpreting a nested namespace PID as an outer PID.
An apply request is refused if the task is active or liveness is unresolved.
Only existing regular files owned by the current user under the task's `cache`,
`incoming`, `results`, and `deliverables` roots are eligible. Symlinks are
excluded and hardlinks are deduplicated. File descriptors are read-only; file
identity, size, and timestamps must remain unchanged. Payload hashing is
deliberately absent because it would read the bytes back into cache.

From the repository, using the existing task environment:

```bash
python tools/reclaim_ks_file_cache.py \
  --workspace "$KS_WORKSPACE" --results "$KS_WORKSPACE/results/ks_main" \
  --receipt "$KS_WORKSPACE/private/cache_advice_dry_run_NEW.json"
```

Only after scheduling has stopped and workers have exited, use a fresh receipt
name and add `--apply`. The tool does not restart scheduling, change concurrency,
or weaken any resource gate. Resume using the existing frozen experiment command
and its existing cache/output paths. Kernel advice may retain dirty or mapped
pages; an unsuccessful reclamation remains an operational blocker rather than
permission to bypass the admission test.

## Resource observer continuity

The original observer replaced its root summary when restarted. Its exact first
session aggregate was recovered from final process output: 1,821 samples from
02:45:29 to 03:46:18 UTC, combined-worker RSS peak 1,008,340,992 bytes, container
peak 8,191,492,096 bytes, minimum free disk 56,911,650,816 bytes, and no alerts.
The final per-job map was not retained. Six earlier rows ending at 02:48:55 UTC
survive; the total number of jobs observed during that first session is unknown.
Missing job peaks are not inferred from the aggregate. Fit-receipt
`RUSAGE_SELF` high-water marks remain the fallback evidence for those jobs.

The updated observer stores an atomic record per session under
`observer_sessions/`, then produces a merged root summary. Repeated snapshots
of one session are deduplicated; samples are summed across distinct sessions,
peaks are maxima, and free-disk measurements are minima. Retained per-job rows
are merged only from actual observations. `coverage_gaps` records incomplete
sessions, and `total_observed_jobs` is null for such a session. The collector
reads both these durable records and the current legacy observer summary, so
the active old-code observer need not be interrupted. A new observer refuses
to overlap another observer for the same results root. Final result packages
include the durable session records.

At the second restart, the new observer correctly archived the entire previous
legacy session before beginning another session. The archive is byte-identical
to the separately retained final summary: 1,634 samples from 03:49:58 to 04:44:31
UTC and all 169 recorded per-job rows. Its combined-worker RSS peak was
1,065,906,176 bytes, container-accounting peak was 8,589,918,208 bytes, and minimum
free disk was 55,894,646,784 bytes. This successful preservation does not repair
the earlier session's missing job rows. The third observer session started at
04:46:20 UTC and maintains the durable-session format.

## Deferred final staging cleanup

The read-only inventory identified the following task-created archive staging
groups. No files in these groups were deleted during live execution.

| Location relative to task workspace | Staging files | Logical bytes |
|---|---:|---:|
| `incoming` | 3 | 279,969,792 |
| `deliverables` | 2 | 133,169,152 |
| `cache/data/ks/exports` | 2 | 256,114,688 |
| Total | 7 | 669,253,632 |

These counts are a snapshot, not a claim that every file has already passed a
redundancy check. Canonical incoming archives currently have a link count of one;
the prior hardlink arrangement cannot be assumed. The inventory records exact
relative paths and file identities privately. It does not establish why the
staging files exist or whether an external writer will recreate them.

The final cleanup sequence is deliberately narrow:

1. Wait for all experiment workers, report generation, checkpoint packaging, and
   relevant transfers to finish. Acquire the scheduler lock and every relevant
   fit writer lock nonblocking. Check actual outer-process command/job identities.
   Inspect open descriptors for the candidate and canonical inodes; any writable
   descriptor or unresolved writer identity blocks deletion. A transfer or sync
   writer outside the available process view must be known to have finished.
2. Re-inventory only the recorded task staging patterns and paths. Require a
   regular, non-symlink file owned by this user, link count one, age over 60 seconds,
   and a stable device/inode/size/mtime/ctime fingerprint. Keep unrecognized files.
3. Verify each canonical replacement before considering a staging copy. For the
   KS archive, require exactly 182,334,299 bytes and SHA-256
   `898f18deccc1fe6b3fb4d6eb0aa847244dc115efc56a260aceaae07bc22c227c`.
   The other delivered archive, catalog, and restore helper retain their verified
   retrieval receipt and are not deletion candidates. For the validation
   checkpoint ZIP, require its retained receipt's 347,338,663 bytes and SHA-256
   `ae78c23139e19c9ccb66eb06276e289aaf880d8b4ec5180833989fd7bed5ae67`,
   plus a complete ZIP-member CRC check. Apply the same exact KS archive check
   to the canonical cache export when assessing its staging copies.
4. Verify that each proposed archive staging file is either byte-identical to its
   verified canonical replacement or is an exact shorter prefix of that file.
   Record its complete SHA-256 and compare it with the same-length canonical
   prefix. A different prefix, oversized file, changing fingerprint, or unknown
   writer means retain and report it. A staging filename alone is insufficient.
5. Recheck locks, writer liveness, and fingerprints immediately before deletion.
   Delete only the verified redundant staging file, retaining canonical archives,
   raw/canonical KS data, transforms, all model/chain/phase checkpoints, receipts,
   logs, and unrelated work. For model-fit staging, use the existing guarded
   `tools/clean_ks_staging.py` with a new receipt; it additionally requires the
   terminal fit receipt, canonical SHA, and ZIP CRC checks.
6. Write the before/after relative-path, size, hash, and decision receipt. Verify
   canonical hashes again after the one cleanup pass and report actual allocated
   storage separately from logical bytes. If a staging copy reappears, stop and
   preserve the observation rather than repeatedly deleting it or guessing a
   mechanism. The final package and manifest exclude staging files regardless.

Because workers remain live and disk reserve is ample, this final pass is
deferred. The original data and useful canonical checkpoint ZIP are preserved.

### Narrow archive-cleanup command and visibility limitation

`tools/clean_ks_archive_staging.py` implements this plan for exactly the seven
recorded names and the canonical identities above. Its default dry-run writes a
new receipt and makes no deletions:

```bash
python tools/clean_ks_archive_staging.py \
  --workspace "$KS_WORKSPACE" --results "$KS_WORKSPACE/results/ks_main" \
  --receipt "$KS_WORKSPACE/private/archive_staging_dry_run_NEW.json"
```

An authorized final pass uses a fresh receipt name and `--apply`. A persistent
marker in the results cleanup directory is created before the first deletion;
any later apply invocation is refused, including after interruption or staging
reappearance. There is no force flag or automatic retry. All nonmatching names,
failed prefix checks, symlinks, and unstable files are retained.

This environment denies access to some visible processes' descriptor metadata.
The tool records those checks as unresolved and refuses apply instead of assuming
the descriptors are harmless. The production dry-run therefore may remain blocked
even after scientific workers exit. If this visibility limitation persists, retain
the seven files and report their 669,253,632 logical bytes with the blocker; the
remaining disk reserve does not require bypassing access controls or guessing
writer liveness. Synthetic acceptance uses an explicitly controlled descriptor
view and also checks a real writable descriptor in its own process. It does not
establish full production visibility.
