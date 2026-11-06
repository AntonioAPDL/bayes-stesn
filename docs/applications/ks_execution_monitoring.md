# Inspecting a retained KS screening run

Use the standalone observer when checking a running campaign from a separate
execution session:

```bash
/data/ks_fullfield_20261105/env/bin/python -B tools/inspect_ks_execution.py \
  --output /data/ks_fullfield_20261105/runs/fullfield_v1_20261105T210330Z \
  --pretty
```

The observer is read-only. It does not import the numerical model, change
checkpoints or job outcomes, settle budget charges, signal processes, or remove
locks. Its presence does not change the scientific source identity of the
campaign implemented at `457fdc4bb116c4ee9fc4e62e47143547e52ede5a`.

## Why an external observer is needed

In the recovered execution environment, a separate session could read a live
process's host-visible PID, start ticks, boot ID and `NSpid`, while access to
`/proc/PID/ns/pid` failed with `EACCES`. The original exact-namespace detector
returned no identity after that permission failure. Consequently, an external
`status` command could incorrectly show an active coordinator as inactive and
a live job as pending. Fresh retained-terminal heartbeats and held scheduler
and writer locks established that the processes were still running.

The observer verifies the accessible process lifetime fields and rereads the
process start ticks to detect disappearance or PID reuse during inspection.
It compares the namespace inode whenever it is available. A denied namespace
inode is explicitly reported as unavailable; a missing essential identity
field produces `unknown`, not a declaration that the process has died.

The JSON snapshot reports `active`, `inactive`, or `unknown`, verified process
identities, nonblocking lock observations, heartbeat age, and job counts.
Immutable terminal receipts take precedence. A job is changed from the legacy
`pending` view to `running` only when a live worker identity is verified.
These are observed counts; the original ledger is never rewritten.

## Resume decisions

An `active` observation is a reason to leave the run alone. An `unknown`
observation, including a held lock without an identified owner, requires
investigation. A free lock is only an observation at one instant and cannot
authorize a duplicate launch by itself. The scientific coordinator's own
exclusive scheduler lock remains the authoritative launch guard.

For a confirmed inactive campaign, first inspect its original budget and
terminal outcomes, then use the unchanged `resume` command in the
[execution handoff](ks_fullfield_launch.md). Retained-terminal execution is
verified current operation, not a guarantee of hosting-environment lifetime.
Do not erase checkpoints, clear locks, reset budgets, or retry terminal
diagnostic exclusions as part of a monitoring repair.

The frozen coordinator conservatively charges an interrupted attempt without
a terminal finish receipt through its next resume. That is an upper bound on
unobserved execution time, not measured CPU use. Preserve and label that
distinction when reporting cost.

## Focused verification

```bash
/data/ks_fullfield_20261105/env/bin/python -m pytest -q \
  tests/test_ks_execution_inspection.py
```

The tests cover denied namespace inspection, essential-field uncertainty,
PID/start-tick/boot/namespace mismatches, process death and reuse, preservation
of lock ownership and bytes, terminal-receipt precedence, read-only count
correction, and actual host-visible process identification. The helper is
also checked against a live coordinator and worker from a different execution
session before operational use.
