# Paired G5 noise stresses and blind G6 shift

`stress_recipe.yaml` declares these studies before their outcomes. They follow
the completed, uncropped, five-field revised development pilot selected by the
separate history rule. They retain its source, configurations, inference
budgets, fit caps, roster, geometry, and horizon. The stress preparation tool
only validates inputs and writes ordinary frozen benchmark profiles; it does
not generate data or fit a model.

| Arm | Shared control | Independent paired fields | New parameter fits |
|---|---|---:|---:|
| G5 whole-field Student-t innovations, ν=5 | Revised G4 | 5 | 55 |
| G5 local Student-t measurement noise, ν=5 | Revised G4 | 5 | 55 |
| G6 hidden velocity shift at lead 25 | Revised G2 | 5 | 0 |

Each field has six observed-site and five new-site jobs. G5 changes the observed
training record and therefore requires a new fit. Both noise arms use the
existing variance-matching generator. The ordinary scenario-specific fitting
seed policy is preserved: G5 fitting seeds match across its two arms and differ
from G4. This estimates the declared fitting procedure, rather than attributing
every finite-fit difference to a fixed-seed intervention.

G6 changes only the future field after the hidden transition. The original
parameters and information at the forecast origin are identical, so a fresh
parameter fit would be redundant. The tool reuses a saved forecast when one
exists; otherwise it loads the saved model and replays its forecast with its
original seed and exact pre-origin observed history. This is a forecast replay,
not an adaptation experiment. Its generation, replay, verification, scoring,
and output costs are measured and charged.

## Runtime and prerequisites

Run from the repository root in the locked environment from
`docs/reproduction.md`:

```bash
export STESN_PY=.venv/bin/python
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Choose the already completed revised pilot identified by the sealed history
decision. The following example assumes that decision selected 1000. If it
selected 2000, replace only the revised-control path with `revised_L2000`.
Do not choose a control from its forecast scores or replace a failed fit.

```bash
"$STESN_PY" configs/benchmark/v03/studies/stress_prepare.py \
  --control-run results/v03/development/revised_L1000 \
  --history-choice docs/evidence/v03/history_choice.json \
  --resolved-profiles results/v03/study_design/resolved \
  --output results/v03/stress/design
```

The resulting `stress_design.json` seals the recipe timestamp, tool hash,
source and runtime, control manifest, all required control receipt identities,
the sealed history decision and exact matching resolved revised profile, and
two complete G5 profiles. The completed control protocol must equal that
selected profile; a different 1000/2000 branch or changed frozen configuration
is rejected. Its inventory is verified before any replay or
summary. A conflicting existing design is rejected. From-scratch reproduction
must create its own matching completed controls and design; changing saved
hashes or substituting different-source evidence does not reproduce this study.
Finish all helper edits before sealing or executing the design.

## Execute the two G5 arms on one assigned worker

```bash
"$STESN_PY" configs/benchmark/v03/studies/run_chunks.py \
  --config results/v03/stress/design/g5_whole_field_t.yaml \
  --output results/v03/stress/g5_whole_field_t \
  --max-invocations 55 --jobs-per-invocation 1 --max-memory-mb 5120 \
  --reserve-gib 4 --next-output-mib 512

"$STESN_PY" configs/benchmark/v03/studies/run_chunks.py \
  --config results/v03/stress/design/g5_local_t.yaml \
  --output results/v03/stress/g5_local_t \
  --max-invocations 55 --jobs-per-invocation 1 --max-memory-mb 5120 \
  --reserve-gib 4 --next-output-mib 512
```

The controller checks disk before each atomic job and resumes ordinary immutable
runner receipts. Execute these commands sequentially, within the global model
worker allowance. Exit 75 is a resource/no-progress pause; preserve its partial
evidence and inspect the cause. The same command resumes when adequate memory
and storage become available. A 4 GiB reserve plus a 512 MiB prospective output
allowance is required for each invocation. Unrelated concurrent writes remain
outside this process's control.

## Replay G6 without fitting parameters

```bash
"$STESN_PY" tools/summarize_v03_stress.py replay-g6 \
  --design results/v03/stress/design/stress_design.json \
  --output results/v03/stress/g6_velocity_shift \
  --max-jobs 55 --max-memory-mb 5120
```

`--max-jobs` limits newly completed replay jobs in this invocation. Set it to 1
for a one-job chunk, then repeat the same command. The tool always uses one
numerical thread, checks memory and disk before a new job, validates cached
receipts and artifacts, and writes `replay_summary.json` with a verified job
count. A successful chunk exit alone does not mean all 55 jobs are complete.
An interrupted job retains its state and cannot claim complete compute
accounting if its earlier consumption is unknown.

Before a G6 score is accepted, the tool verifies all of the following:

- The G2/G6 DGP family seed mapping, parameters, coordinate/time arrays, and
  O/C/U/hole roles match the sealed design.
- Every observation and latent value before `final_origin + 24` is identical;
  that row is the first affected observation for lead 25.
- Only the original pre-origin O history enters model-state prediction. U and
  C labels, post-origin observations, and the shifted future never condition a
  forecast. Saved model training arrays must match this allowed history.
- Every numeric leaf of the original G2 score object is reproduced with
  `rtol=1e-10` and `atol=1e-10`, and structural metadata match exactly.
- The original native qualification and operational fallback remain unchanged.
  If the core saved an unqualified raw model forecast for diagnostics, its
  replay keeps that failed qualification. It never enters common-qualified
  effects; replay does not promote a failed fit into an eligible native result.

## Audit paired effects and costs

After both G5 arms and all 55 G6 jobs are complete:

```bash
"$STESN_PY" tools/summarize_v03_stress.py summarize \
  --design results/v03/stress/design/stress_design.json \
  --whole-field-run results/v03/stress/g5_whole_field_t \
  --local-run results/v03/stress/g5_local_t \
  --g6-output results/v03/stress/g6_velocity_shift \
  --output docs/evidence/v03/stress_summary.json
```

The sealed `v03-paired-stress-summary-1.0` report retains both raw score objects,
receipt identities, paired field IDs, raw differences, shared-scale
differences, exact eligible counts, and pointwise paired-bootstrap uncertainty.
For each G5 pair, **both losses use the control G4 training-only SD**. The raw
standalone G5 normalization receipts remain intact; they are not the scale for
cross-arm comparisons. The same rule uses G2's training SD for both G6 futures.
Dimensionless coverage stays dimensionless; losses scale by their declared
metric power. Positive stress-minus-control effects indicate larger loss,
except coverage indicators, for which a positive difference is more coverage.

Native effects use each exact common-qualified intersection. Operational RMSE
and MAE retain every completed pair, including declared fallbacks. Joint/path
and interval companions retain their native score populations. Five paired
fields support exploratory estimates and pointwise intervals, not precise
robustness or simultaneous superiority claims. Coupled arms are shown
separately, never counted as independent macro strata.

New G5 pipelines and G6 generation/replay/scoring are summed separately, with
wall and CPU time and an explicit completeness flag. Reused historical control
cost is deduplicated by its receipt identity and reported separately. This
completeness flag concerns each stated timing scope: the frozen core's G5 job
pipeline excludes shared DGP generation and controller overhead. The resource
guard's invocation logs measure wall time including that work, but do not
separately itemize its CPU consumption. Do not describe job-pipeline sums as
total campaign compute.

This study does not execute robust Student-t model variants, a heteroskedastic G5
arm, post-shift adaptation, or post-shift calibration. Those remain conditional
extensions with separate scientific requirements.
