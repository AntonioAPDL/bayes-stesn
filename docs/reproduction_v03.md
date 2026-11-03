# Reproduce the v0.3 research campaign

This is the ordered entry point for the approved PRO study. The commands below
use the existing public CLI and small study helpers. They do not replace the
scientific configurations with shorter demonstration budgets. Read the final
release manifest and evidence before treating an archived outcome as a result
of a fresh execution.

## Choose the kind of reproduction

| Goal | Work to run | What it establishes |
|---|---|---|
| Check the released evidence | Download and verify the release archives; validate report inputs and inventories | The supplied artifacts are internally consistent with their recorded source and design |
| Check the software | Install the locked environment; run `tools/release_check.py` | Mathematical/API, source, installed-package and small execution checks |
| Rerun the study | Execute the stages below in a fresh checkout/output tree | New execution receipts for the same declared procedure; measured costs and artifact hashes will change |

Exact numerical restart requires the compatible recorded numerical runtime.
Cross-platform bitwise identity is not promised. Replaying the same generated
field is not an additional independent replication. Never change an old
receipt's hash, source, diagnostic decision, or seed to make a rerun acceptable.

## 1. Install, inspect resources, and keep one scientific source

Create the isolated CPython 3.12 core/report/development environment and install
the optional CPU neural dependencies using [reproduction.md](reproduction.md).
The full model roster requires those neural dependencies. From the repository
root:

```bash
export STESN_PY=.venv/bin/python
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Use at most two model workers **in total across all simultaneous commands**,
one numerical thread per worker, and the runner's actual memory admission.
The commands here are sequential, one-worker examples. A `5120` MiB allowance
is a ceiling request, not permission to exceed the host/cgroup limit. The disk
controller checks a 4 GiB reserve plus the declared next output/temporary
allowance before an invocation. Its estimates cannot reserve space against
other users' concurrent writes. Exit 75 means inspect the resource or
infrastructure pause and resume the same command when it is admissible.

Do not edit `src/`, locked runtime inputs, or a helper currently executing a
sealed study. Report/documentation changes have their own final release hash
gate. The public package source fingerprint and the Git commit are separate
identities.

## 2. Reference reliability and exact sampler coordinates

Run the complete 60-job conservative reference, including its failed outcomes
if any arise in the new execution:

```bash
"$STESN_PY" configs/benchmark/v03/studies/run_chunks.py \
  --config configs/benchmark/v03/reference_reliability.yaml \
  --output results/v03/development/reference_final \
  --max-invocations 60 --jobs-per-invocation 1 --max-memory-mb 5120 \
  --reserve-gib 4 --next-output-mib 512

"$STESN_PY" tools/audit_v03_references.py \
  results/v03/development/reference_final \
  --output results/v03/reference_audit_reproduced.json

"$STESN_PY" tools/compare_sampler_coordinates.py \
  --profile pilot --scenarios g1,g3,g4 --replications 0 \
  --warmup 1000 --draws 2000 --continuation 2000 --chains 4 \
  --output results/v03/coordinate-final --max-memory-mb 5120
```

The coordinate comparison deliberately uses the named `pilot` profile and its
explicit sampler overrides. It fits the same statistical target with both
transition kernels and does not score the held-out future. The reference audit
checks saved scalar results and computes the separately labeled G1
known-parameter Gaussian reference; it performs no comparator refit.

## 3. Strict development retuning

```bash
"$STESN_PY" configs/benchmark/v03/studies/run_chunks.py \
  --config configs/benchmark/v03/tuned_final.yaml \
  --output results/v03/development/tuned_final \
  --max-invocations 60 --jobs-per-invocation 1 --max-memory-mb 5120 \
  --reserve-gib 4 --next-output-mib 512
```

This is the full prespecified tuning procedure, including spatial folds,
finalist seeds, qualified-finalist rules, bounded replacements, all declared
fitting allowances and the final fit. It is substantially more work than 60
single fits. G4 selections appear first so that the dependent frozen-setting
studies can be resolved while the other tuning mechanisms finish. All 60
terminal outcomes remain required; a missing selection is not a fallback.

## 4. Separate AR and predictive-path evaluations

Use the saved A-task ridge ESNs from the reference study. Run the two mechanisms
and two modes sequentially, with a free science slot:

```bash
for STESN_SCENARIO in g4_existing_fronts g3_excitable_field; do
  STESN_CASE="results/v03/development/reference_final/$STESN_SCENARIO/rep_000"
  "$STESN_PY" tools/evaluate_extensions_v03.py \
    --baseline "$STESN_CASE/A/bayes_esn_ridge/model.npz" \
    --simulation "$STESN_CASE/simulation.npz" \
    --study-manifest results/v03/development/reference_final/manifest.json \
    --extensions full_residual_ar --track A --seed 20261103 \
    --warmup 1000 --draws 2000 --continuation 2000 --paths 128 \
    --inspection-draws-per-chain 32 --max-memory-mb 5120 \
    --output "results/v03/extensions/final_ar_$STESN_SCENARIO"

  "$STESN_PY" tools/evaluate_extensions_v03.py \
    --baseline "$STESN_CASE/A/bayes_esn_ridge/model.npz" \
    --simulation "$STESN_CASE/simulation.npz" \
    --study-manifest results/v03/development/reference_final/manifest.json \
    --path-stability --track A --seed 20261103 \
    --warmup 1000 --draws 2000 --continuation 2000 --paths 128 \
    --inspection-draws-per-chain 32 --max-memory-mb 5120 \
    --output "results/v03/extensions/final_paths_$STESN_SCENARIO"
done
```

The path-stability mode uses its declared nested 128/256/512 prefixes. The
`--paths 128` argument above records the original invocation settings; it does
not truncate that mode's 512-path reference. Read the actual saved decisions
instead of inferring the chosen count from this argument.

Produce a **new** findings receipt, leaving the released findings immutable:

```bash
"$STESN_PY" tools/summarize_extensions_v03.py \
  --reference-run results/v03/development/reference_final \
  --g4-ar results/v03/extensions/final_ar_g4_existing_fronts \
  --g3-ar results/v03/extensions/final_ar_g3_excitable_field \
  --g4-paths results/v03/extensions/final_paths_g4_existing_fronts \
  --g3-paths results/v03/extensions/final_paths_g3_excitable_field \
  --output results/v03/extension_findings_reproduced.json

export STESN_PATH_DECISION=results/v03/extension_findings_reproduced.json

"$STESN_PY" tools/check_conditional_prior_v03.py \
  --ablation results/v03/extensions/final_ar_g4_existing_fronts \
  --ablation results/v03/extensions/final_ar_g3_excitable_field \
  --output results/v03/conditional_prior_reproduced.json --max-memory-mb 5120
```

The reducer reconstructs numerical findings from the supplied current-source
outputs. Historical metadata verify when the original decision rule was
declared; old numerical outcomes do not fill missing current results. The prior
tool integrates the actual priors conditional on frozen spatial/temporal
designs. It does not certify autonomous prior trajectories or posterior mixing.

## 5. Resolve and execute the remaining shared-data studies

Follow [the study guide](../configs/benchmark/v03/studies/README.md) in order:

1. Verify the saved full-duration G3 finite-grid numerical evidence, or reproduce
   it before resolving the long-history profiles.
2. Resolve drafts from all eleven actual G4 validation selections and the fresh
   path-count decision.
3. Complete the six training-only timing fits, then freeze common model caps.
4. Complete the paired history1000/history2000 design and apply its previously
   declared selection rule.
5. Complete the separate nested-sensor study and the chosen revised pilot.
6. Run only the eligible preselected calibration cases; seal each saved result.
7. Execute [the paired G5/G6 stress study](../configs/benchmark/v03/studies/stress_README.md)
   using the completed revised controls.
8. Attempt the confirmation freeze. Generate final-seed datasets only if that
   gate succeeds. A blocked gate is a reportable development-only result, not
   permission to select favorable replacements or weaken diagnostics.

The guide's released-campaign example sets `STESN_PATH_DECISION` to its archived
receipt. For a fresh rerun, retain the new path from Section 4 instead. Histories,
sensor counts, future origins, role assignments and normalization references
are different declared interventions; do not combine their losses as if they
were additional independent fields from one design.

### Preserve the confirmation decision

The core `benchmark freeze` command refuses insufficient common-qualified
pilot pairs without writing a freeze. The following saved-artifact assessment
records that specific scientific rejection as well as a successful admission.
Use the history selected by the sealed `history_choice.json`; this example is
for a selected initial history of 1000:

```bash
PYTHONPATH=src .venv/bin/python tools/assess_confirmation_v03.py \
  --pilot results/v03/development/revised_L1000 \
  --config results/v03/study_design/resolved/confirmation_L1000.yaml \
  --primary-comparisons results/v03/study_design/resolved/primary_comparisons.json \
  --fhn-validation docs/evidence/v03/generator_validation_long_history_grid64.json \
  --minimum-pairs 5 --target-mcse .02 \
  --output results/v03/confirmation/assessment.json \
  --freeze-output results/v03/confirmation/freeze.json \
  --resolved-config results/v03/confirmation/frozen.yaml
```

Retain the one-thread environment from Section 1. If the history rule selects
2000, use `revised_L2000` and `confirmation_L2000.yaml` together. The tool
verifies all saved pilot artifacts, the matched design and generator evidence,
then calls the unchanged confirmation factory. It records the exact primary
cell pair counts and binds its inputs by SHA256. An arbitrary configuration,
runtime, integrity or infrastructure error remains an error; it cannot become
a completed scientific rejection.

Inspect the assessment's `admitted` field before continuing. A zero process
exit and `complete: true` mean the assessment completed, including when it
reports `blocked_insufficient_common_pairs`. A blocked assessment creates no
freeze or runnable frozen configuration. Admission is also distinct from
completed confirmation or achieved precision; a valid freeze can record an
MCSE target shortfall at the prespecified replication cap. The assessment
generates no confirmation data or forecasts and never replaces deficient
pilot seeds. Preserve existing assessments and freezes by using new output
paths for a separately justified reassessment.

## 6. Audit, render, and publish verified artifacts

Use [the report guide](../report/v03/README.md) to bind the completed runs,
learning-curve audits, P5 findings, prior check, separate calibration seals,
stress summary and saved animation assets. Run `--check-inputs` before rendering.
The renderer reads saved evidence and does not refit models. Its output
inventory does not by itself establish visual quality: inspect the PDF pages
and representative first/middle/last animation frames, and record the scope
actually inspected.

Finish source, tools, configurations, requirements and documentation edits
before the final source/installed-wheel release gate. The release packager
checks those hashes again, requires completed benchmark evidence and sealed
report assets, and creates verified archives and a Git history bundle. Verify
the exact public commit's CI and the public tag/assets after publication. A
local check, a configured workflow and a published release are three separate
states.

See [the requirements ledger](plans/v03_requirements.yaml) and
[completion dependencies](plans/v03_completion_checklist.md) for conditional
extensions, justified deferrals and the remaining evidence obligations.
