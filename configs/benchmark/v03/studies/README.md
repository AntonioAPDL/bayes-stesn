# Remaining v0.3 studies: resolve, measure, execute, and audit

This directory makes the remaining development studies reproducible without
changing the frozen package or weakening any model budget. The recipe is
`remaining_studies.yaml`; it is **not** a runnable `BenchmarkProtocol`.
`prepare.py` binds the recipe to completed G4 validation selections and the
completed predictive path-count decision. Its first output is a set of explicit
drafts for a small training-only timing preflight. Only complete timing receipts
produce profiles with common finite fit allowances.

No history, sensor, revised-development, calibration, or confirmation outcome is
implied by these files. The planning-only JSON in
`docs/evidence/v03/remaining_studies_preflight/` uses conservative reference
settings because the intended G4 selection receipts must exist before they can
be transferred. Estimates are not observed peaks or runtimes.

## Scope and information budget

| Study | Independent fields | Geometry and common forecast | Required fits |
|---|---:|---|---:|
| History 1000 and 2000 | Five per G1/G3/G4 | 7×7; O25/C4/U20; common initial endpoint 4000, final origin 4300; H100 | 330 |
| Nested sensors 25/50/100 | One G4 field | 15×15; reserved O100/C10/U115; common observed score sites 25; final origin 4300; H100 | 33 |
| Revised development | Five per G1–G4 | 7×7; O25/C4/U20; one chosen initial history; H100 | 220 |
| Training-only timing | One separate G4 field | Largest planned sensor geometry; 2300 supplied rows; six fresh processes | 6 |
| Separate calibration | One preselected revised G4 field | Frozen ESN/GP/GRU fits on A/B; 25 H100 calibration blocks and a later common evaluation block | 0 new fits |

There are **583 required scientific fits**, plus six timing fits. The roster has
six models on A (persistence, KRR, GP, ridge ESN, basis GRU, E-QESN), and five on B
(spatial persistence, KRR, GP, ridge ESN, basis GRU). E-QESN retains its original
20 finite ensemble members. The selected 512 predictive paths apply to models
that sample paths; they do not manufacture 512 E-QESN members.

History labels specify the rows available **before the first nominal validation
origin**. Final fitting also includes the next three H100 blocks, so the 1000
and 2000 cases supply 1300 and 2300 rows at the identical final forecast origin.
Frozen specifications perform no new candidate or finalist selection fits.
The complete history design also resolves 500 and 4000. Neither deferred length
executes without a documented trigger; 4000 additionally requires a new timing
review at 4300 supplied rows.

The history scale is computed once from observed sites 0:25 and rows 3500:4000,
with `ddof=1`, identically across conditions. Sensor experiments retain that same
time window and the first 25 observed sites, with nested O and fixed C/U roles.
Their 15×15 field is a separate experiment from the 7×7 history experiment.
One sensor field supports descriptive sensitivity, not a replication MCSE.

Separate seed namespaces are declared in the recipe: history 2026110301,
sensors 2026110302, revised/confirmation 2026110303, training-only timing
2026110304, and history-decision resampling 2026110305. Confirmation uses the
package's independent `final_dgp` namespace and cannot start before a valid
freeze. The revised data do not reuse the G4 tuning field.

## 1. Establish the runtime and resolve the drafts

Run from the repository root using the same locked runtime as the scientific
receipts. `STESN_PY` can point to that environment's Python executable.
The completed long-history G3 receipt is already available. If reproducing
from scratch without that receipt, complete Section 3 before this resolution
step; the builder deliberately refuses to materialize the studies without it.

```bash
export STESN_PY=/workspace/scratch/158260e8300f/bstesn-v03-env/bin/python
export STESN_PATH_DECISION=docs/evidence/v03/extension_findings.json
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$STESN_PY" configs/benchmark/v03/studies/prepare.py \
  --tuned-dir results/v03/development/tuned_final \
  --reference-config configs/benchmark/v03/reference_reliability.yaml \
  --path-decision "$STESN_PATH_DECISION" \
  --g3-validation docs/evidence/v03/generator_validation_long_history_grid64.json \
  --output results/v03/study_design/drafts
```

Each model/track uses its qualified G4 validation selection, with all required
finalist seeds and folds checked. A completed failed selection uses the declared
conservative reference and records that fact. A missing or interrupted receipt
stops resolution. Final test losses never choose configurations.
`--tuned-dir` and `--tuned-run` are aliases; neither fixes a directory name in
code. Set `STESN_PATH_DECISION` to the actual completed findings receipt from the
current package source. The earlier `tuned_release` and P5 evidence remain
historical if a later resource-accounting source freeze changes the package
fingerprint. A fresh source requires fresh matching selection/path evidence;
the builder supplies no compatibility override.

The deterministic transfer changes basis-GRU stopping to the actual H100 task
and requires a complete 100-step validation tail. Architecture, regularization,
conditioning, full optimizer/epoch budgets, MCMC budgets, and diagnostic
thresholds remain as selected. This is an explicit development-frozen
specification experiment, not per-dataset retuning.

The path-count receipt must match the current package, its verified prospective
rule, and the requested count. The completed P5 rule selects 512. It is a finite
reference ceiling: its own evidence reports non-negligible residual predictive
Monte Carlo variation in G3, rather than certifying universal stability.

For an estimate before G4 selection finishes, run `prepare.py --planning-proxy`
with a separate output directory. That mode emits no runnable YAML.

## 2. Run at most six isolated training-only timing fits

Wait for a free science slot and recheck current cgroup headroom. The following
loop uses exactly the model/track pairs chosen by the builder's dimension-based
memory rule. It starts one fresh process at a time, never computes a forecast,
and generates no observations beyond the fitting cutoff.

```bash
"$STESN_PY" - <<'PY'
import json
from pathlib import Path
import subprocess
import sys

profiles = Path("results/v03/study_design/drafts")
for job in json.loads((profiles / "preflight_jobs.json").read_text()):
    subprocess.run([
        sys.executable, "configs/benchmark/v03/studies/training_preflight.py",
        "--profiles", str(profiles), "--case", job["case"],
        "--model", job["model_id"], "--track", job["track"],
        "--output", "results/v03/study_design/timing", "--max-memory-mb", "5120",
    ], check=True)
PY
```

Each full-budget fit has a separately declared 1800-second safety allowance.
The optional POSIX alarm can be delivered after a native call returns; it is
not an OS hard memory or instruction-time bound. A failed, interrupted, or
capped case cannot derive a final allowance. Do not resume a partial fit and
call the residual work a full fresh-fit timing measurement.

The common cap for model `m` is
`max(30, ceil(20 * max(large_geometry_seconds[m], reference_seconds[m]) + 10))`.
Ordinary persistence keeps the 30-second floor. The resulting cap is identical
across initial history, sensor, revised, and confirmation conditions for that
model. The factor is a conservative prospective resource rule, not a complexity
guarantee; measuring the larger memory dimension does not prove a larger CPU
cost for every possible model configuration.

Bind the resulting finite caps in a new directory:

```bash
"$STESN_PY" configs/benchmark/v03/studies/prepare.py \
  --tuned-dir results/v03/development/tuned_final \
  --reference-config configs/benchmark/v03/reference_reliability.yaml \
  --path-decision "$STESN_PATH_DECISION" \
  --g3-validation docs/evidence/v03/generator_validation_long_history_grid64.json \
  --preflight-dir results/v03/study_design/timing \
  --output results/v03/study_design/resolved
```

Inspect `study_provenance.json`, `resource_plans.json`, and the resolved YAML.
These bind the actual selections, all declared profiles, timing evidence, and
shared budgets. Planning forecasts remain null instead of extrapolating short
reference timings to every larger study.

## 3. Validate the long G3 trajectory before history outcomes

The previous H100 validation does not establish phase accuracy over 4400 saved
observations. The independent long-trajectory check is:

```bash
"$STESN_PY" tools/validate_generators.py --extended \
  --grid-size 64 --dt 0.05 --delta 2.0 --observations 4401 --burnin 200 \
  --seed 983417 --output docs/evidence/v03/generator_validation_long_history_grid64.json
```

Inspect the actual receipt before proceeding. The finite-grid scientific target
requires the declared time-discretization and morphology checks. A separate
mesh-refinement failure must remain visible; a failed long-trajectory time gate
requires a prospectively revised generator design and validation, not a silent
acceptance or a claim that the old H100 receipt covers all later rows.

The saved long-history receipt now passes the finite-grid time and activity
checks over 8800 physical units: time-halving RMS 0.001980 is below 0.02, and
the mean temporal SD activity proxy is 1.226906. Mesh-refinement RMS 0.272371
fails its separate criterion, so the original overall `refinement_gate_not_met`
status remains unchanged. These are one independent numerical pilot's results,
not a guarantee for every random initial field. `prepare.py` binds that receipt
and additionally requires coverage of the complete 8798-unit planned record,
beyond the generic API's forecast-window check. Read
`docs/evidence/v03/scientific_review.json` for the verified scope. Reuse this
valid receipt instead of repeating the expensive pilot unnecessarily.

## 4. Execute the complete declared curves in durable chunks

The `curve` API resolves all conditions before running an allowed subset:

```bash
"$STESN_PY" -m bayes_stesn benchmark curve history \
  --config results/v03/study_design/resolved/history_base.yaml \
  --values 500,1000,2000,4000 --common-endpoint 4000 \
  --execute-values 1000,2000 --output results/v03/history \
  --workers 2 --max-memory-mb 5120

"$STESN_PY" -m bayes_stesn benchmark curve sensors \
  --config results/v03/study_design/resolved/sensors_base.yaml \
  --values 25,50,100 --execute-values 25,50,100 \
  --output results/v03/sensors --workers 2 --max-memory-mb 5120
```

`curve` itself has no invocation job/time chunk flags. For bounded sessions,
first install the builder's exact complete design receipt in an empty curve
directory; refuse any conflicting existing identity. Then use the resolved
individual profiles with the ordinary chunked runner:

```bash
"$STESN_PY" - <<'PY'
import json
from pathlib import Path

for kind in ("history", "sensors"):
    source = Path("results/v03/study_design/resolved") / f"{kind}_curve_design.json"
    target = Path("results/v03") / kind / "curve_design.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and json.loads(target.read_text()) != json.loads(source.read_text()):
        raise ValueError(f"Existing curve design differs: {target}")
    if not target.exists():
        target.write_bytes(source.read_bytes())
PY

"$STESN_PY" -m bayes_stesn benchmark run \
  --config results/v03/study_design/resolved/history_1000.yaml \
  --output results/v03/history/v03_fixed_history_history_1000 \
  --workers 2 --max-memory-mb 5120 --max-jobs 20 --max-wall-seconds 600

"$STESN_PY" -m bayes_stesn benchmark status \
  results/v03/history/v03_fixed_history_history_1000
"$STESN_PY" -m bayes_stesn benchmark resume \
  --output results/v03/history/v03_fixed_history_history_1000 \
  --workers 2 --max-memory-mb 5120 --max-jobs 20 --max-wall-seconds 600
```

Repeat for `history_2000` and `sensors_25`, `sensors_50`, `sensors_100`, using the
`name` in each YAML as its child directory. Invocation limits pause between
atomic jobs; they do not reduce scientific fitting budgets. Preserve existing
receipts and checkpoint identities. Never add `--models`, `--scenarios`, or a
different replication count to an already initialized campaign.

## 5. Choose the revised history and audit sensor representation

Once both initial history conditions are complete, execute the read-only
prespecified rule in `remaining_studies.yaml`:

```bash
"$STESN_PY" tools/summarize_v03_learning_curves.py history \
  --design results/v03/history/curve_design.json \
  --output docs/evidence/v03/history_choice.json

"$STESN_PY" tools/summarize_v03_learning_curves.py sensors \
  --design results/v03/sensors/curve_design.json \
  --reconstruction-ridge 0.01 \
  --output docs/evidence/v03/sensor_representation.json
```

The history tool checks the identical realized fields, times, roles, scale, and
model budgets; it reports whole-dataset paired effects, independent-field MCSE,
stratified bootstrap bounds, qualification counts, and complete recorded costs.
Native comparisons condition on the common-qualified intersection. Operational
comparisons retain every completed dataset, including declared fallbacks.
Neither population should be described as an unconditional native accuracy
estimate. The history rule is a development resource/specification decision,
not a confirmatory superiority test.

The rule was declared at **2026-11-03 07:26:33 UTC, before any curve outcomes**.
It defaults to 1000 and promotes 2000 only if all of the following hold:

- On **both ESN tracks**, the one-sided 95% stratified whole-dataset bootstrap
  lower bound for equal-G1/G3/G4 mean normalized `CRPS(1000) − CRPS(2000)` is at
  least 0.02. Positive values favor 2000; the bootstrap uses 5000 resamples.
- Each ESN scenario/track cell has at least three common-qualified pairs, and
  long-history qualification is no lower than short-history qualification in
  every cell.
- All-roster full-pipeline costs are completely recorded, and the summed
  long/short time ratio is at most 2.5.

The practical margin and resource ratio are declared preferences. Small-sample
percentile bounds do not establish formal coverage or simultaneous superiority.
Any failed gate retains 1000 with reasons; insufficient pairs are not evidence
that additional data have no benefit. Independently of this ESN-centered choice,
the output reports native CRPS for every probabilistic model and both native
and operational RMSE for every compatible model. The four confirmation
contrasts remain unchanged.

The sensor tool reads numeric saved model state without refitting or rebuilding
a posterior. It reports the singular values, numerical rank, tolerance, and
condition of the actual spatial dictionary and likelihood temporal design.
The original fitted dictionary can change its geometry normalization with O.
For an interpretable geometry-only reconstruction audit, a separate diagnostic
holds the 100-sensor fitted basis fixed within each track and applies

`A = Phi_U @ solve(Phi_O.T @ Phi_O / n + 0.01 I, Phi_O.T / n)`.

Its row squared norms describe target variance under unit independent
observation noise; its squared spectral norm describes the worst-direction
amplification. These are not correlated-residual predictive variances or
forecast scores. The actual prior, basis normalization, and fits are unchanged.
Missing saved models remain unavailable. Report one-field sensitivity without
an invented replication interval.

## 6. Execute exactly one revised pilot, then calibration and freeze

Read `selected_initial_history` from the sealed history decision. Use only the
corresponding `revised_L1000.yaml` or `revised_L2000.yaml` and its matching
confirmation draft. For example, after a declared choice of 1000:

```bash
"$STESN_PY" -m bayes_stesn benchmark run \
  --config results/v03/study_design/resolved/revised_L1000.yaml \
  --output results/v03/development/revised_L1000 \
  --workers 2 --max-memory-mb 5120 --max-jobs 20 --max-wall-seconds 600
```

Resume the same campaign until all 220 jobs are terminal. Keep all five fields
per mechanism, including diagnostic failures. Do not quietly expand a failed
cell to obtain enough favorable qualifying fits.

The separate calibration exercise reuses the uncropped revised G4 replication
0 fits of ESN, GP, and basis GRU on A and B. Check each preselected base receipt's
qualification before invoking it; mark an unqualified case unavailable instead
of substituting another field. Each eligible case follows this pattern:

```bash
"$STESN_PY" -m bayes_stesn benchmark calibrate \
  results/v03/development/revised_L1000 \
  --scenario g4_existing_fronts --replication 0 \
  --model bayes_esn_ridge --track A --blocks 25 \
  --output results/v03/calibration/bayes_esn_ridge_A
```

Run the other five preselected model/track cases only if eligible. There are at
most 150 calibration forecasts and six later raw forecasts, with no parameter
refit. C observations calibrate B; U stays out of calibration. Serial blocks
are not independent datasets. Raw and calibrated intervals use the same later
evaluation block, and calibration does not alter native joint draws. The
current API rejects cropped history-curve fits, which is why it reuses the
uncropped revised pilot.

Finally, the frozen four primary contrasts are ridge ESN against GP and basis
GRU on A/B, using normalized native CRPS and equal G1–G4 weights:

```bash
"$STESN_PY" -m bayes_stesn benchmark freeze \
  results/v03/development/revised_L1000 \
  --config results/v03/study_design/resolved/confirmation_L1000.yaml \
  --planning-mode paired_macro \
  --primary-comparisons results/v03/study_design/resolved/primary_comparisons.json \
  --minimum-pairs 5 --target-mcse 0.02 \
  --fhn-validation docs/evidence/v03/generator_validation_long_history_grid64.json \
  --output results/v03/confirmation/freeze.json \
  --resolved-config results/v03/confirmation/frozen.yaml
```

This command is a gate, not a guarantee that five attempted pilot fields yield
five common-qualified pairs. If the gate cannot be met, preserve the evidence
and report the blocked precision claim. A valid freeze fixes the attempted
replications within the declared 20–60 cap before any final seed dataset is
generated; cap-limited MCSE shortfalls remain explicit. Only then use `benchmark
run` with the frozen YAML and `--power-freeze` receipt. See
`docs/study_design_v03.md` and `docs/derivations/07_sampler_coordinates_recovery.md`
for the statistical and recovery contracts.
