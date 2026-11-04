# v0.3 delivery: evidence closure, report verification, and bounded archives

This is an executable delivery procedure, not a statement that any experiment,
gate, visual review, or public upload has completed. The scientific completion
standard is [the PRO acceptance matrix](plans/PRO_IMPLEMENTATION_PLAN.md#13-requirement-and-acceptance-matrix).
Keep the package source fingerprint frozen throughout the scientific campaign.
Reporting tools, documentation, and packaging changes must be finalized before
the final local source/installed-wheel gate.

The [current findings](results_v03.md) and
[requirements ledger](plans/v03_requirements.yaml) give the dated execution
status. Their pending release entries are resolved by the actual generated
closure and publication receipts, not by editing covered source after the gate.
The previous requirements YAML is preserved exactly under
`results/v03/release_evidence/historical_source_snapshots/`, with its own
retention inventory; historical progress and public-release records keep their
original meaning.

## 1. Close the evidence before packaging

Retain every terminal scientific outcome, failed qualification, declared
fallback, selection/recheck trace, checkpoint, and source/design/data/runtime
receipt. A complete attempted campaign does not mean every fit qualified.
Use the saved-artifact synthesis's `--check-inputs` recipe to verify the actual
completed campaigns and their companion studies. Do not rewrite historical
receipts to match the recovered run.

The release needs an explicit disposition for every acceptance ID. Save the
final closure under `results/v03/release_evidence/`, with each claim linked to
its actual receipt path, SHA256, and scope. The following inventory describes
what must be retained; it does not mark a row as passed.

| ID | Evidence to retain and check |
|---|---|
| R01 | Starting commit/source, exact runtime/locks, recovery/baseline checks, and dated resource observations. |
| R02 | Design, computation and attempt identities; resume/invalidation tests; actual immutable campaign manifests. |
| R03 | Worker-loss/retry tests, interruption traces and terminal-state accounting; no synthetic receipts for unattempted jobs. |
| R04 | Phase-aware checkpoint/restart and continuation equivalence tests, with the declared runtime and preserved checkpoints. |
| R05 | Exact-coordinate mathematical/reference tests and all six bounded coordinate cases, including qualification and mixing/cost summaries. |
| R06 | Optimizer continuation tests and actual fit/selection traces, including exhausted caps and nonqualified outcomes. |
| R07 | Resolved candidate configurations, canonical uniqueness/exhaustion evidence and selection tests. |
| R08 | Actual basis-GRU candidates and declared roster/capabilities; no claim of an unverified named-paper reproduction. |
| R09 | Every finalist seed/fold recheck and the fixed replacement policy, with failed selections retained. |
| R10 | H100 temporal stopping and edge/interior spatial validation receipts and information-boundary tests. |
| R11 | Complete declared history design, executed 1000/2000 conditions, paired field/scale checks and the fresh sealed history decision. Deferred 500/4000 conditions stay explicit. |
| R12 | Nested 25/50/100 sensor conditions, common targets/scales, representation audit and one-field uncertainty limitation. |
| R13 | Actual bounded P5 outputs, prior assessment, calibration/stress evidence and an immutable final disposition overlay. Each deferred extension needs its specific technical rationale. |
| R14 | Revised paired native and all-attempt operational summaries, exact common-qualified populations, MCSE/intervals and qualification attrition. |
| R15 | Saved confirmation assessment, exact deficient cells or a valid freeze, source/runtime/seed/R bindings, and any actually completed independent confirmation. Admission is separate from execution and achieved precision. |
| R16 | Admission plans, actual attempt resource/cost receipts and a sealed monitor summary with its measured time window and ownership scope. Shared cgroup usage is not isolated model memory. |
| R17 | Forecast-ownership/joint-path/score invariants and the measured allocation comparison; retain limits on full-fit/scoring-memory claims. |
| R18 | Final current-source and installed-wheel gate, optional CPU-neural scope, CLI/serialization checks, complete validation inventory and distribution hashes. |
| R19 | Verified synthesis, PNG/SVG figures, planned MP4/GIF/still assets, availability of every planned illustration, and recorded PDF/HTML/media QA. |
| R20 | Exact public commit and CI, unchanged historical tags, new tag/release and asset readback checksums. Post-upload verification is delivered as a separate sidecar with its own checksum, not inserted into the archive it verifies. |

The [completion checklist](plans/v03_completion_checklist.md) separately
requires the two G5 arms and blind G6 replay after reliability, and the eligible
preselected calibration cases. Deferring a statistical robust-noise extension
does not silently waive those base studies. Conditional branches must retain
their actual admission decision and limitation.

`tools/assess_confirmation_v03.py` calls the unchanged freeze authority. If a
primary cell has fewer than the required five common-qualified pairs, retain
the completed blocked assessment, its exact factory error, input hashes and
deficient cells. Do not manufacture `freeze.json` or `frozen.yaml`. A valid
freeze with `power_status=mcse_target_not_met_at_cap` is not achieved precision.
A scientific block supports a clearly labeled development-only release; it
does not support a completed-confirmation or predictive-superiority claim.

### Retained execution and validation support

The generated `results/v03/release_evidence/execution_helpers/` tree retains
the exact reviewed operational helper bytes with `retention_inventory.json`.
These include the one-worker measurement helpers, simulation-cache reuse,
execution-cost aggregation and saved-media QA helper/plan. The separate
`results/v03/release_evidence/validation_support/supplemental_retention_inventory.json`
binds closed targeted XML snapshots, historical parity receipts, helper test
stdout/stderr and preparation/review evidence. Preserve each inventory
immutably; any later retained files need a new supplemental inventory. These
snapshots are not the final current-source gate or final visual QA.

The subsequent report correction has a separate immutable inventory at
`results/v03/release_evidence/presentation_helpers/supplemental_retention_inventory_20261104T003300460251Z.json`.
It retains the presentation-only helper, approved input plan, original whole
report, corrected report, both fixed comparisons, preview and final QA evidence,
and every measured correction/check attempt. The
`results/v03/rendering/revised_corrected_visual_qa.json` receipt closes the
14-case saved-artifact/sample review and all 26 ordinary PDF pages. Its
explicit browser block and unperformed continuous playback remain part of R19;
do not turn them into successful checks or infer final-synthesis QA from this
ordinary-report receipt.

Set `PYTHONDONTWRITEBYTECODE=1` when importing retained helpers so the selected
evidence tree does not acquire import caches. Bind helper calls to explicit
repo-relative paths and actual inputs. The one-worker measurement helpers must
not label a two-worker child as a one-worker measurement. If confirmation is
admitted with another execution scope, retain the native execution receipts
and honestly scoped external command evidence separately.

After all required scientific phases and actual monitor windows are closed,
write `results/v03/release_evidence/execution_costs.json` using the retained
aggregation helper's `--require-closed` mode. Its complete-scope subtotals,
incomplete observations, missing measurements and command/monitor scopes must
remain distinct. Lost CPU or elapsed time stays unavailable, including after
successful scientific recovery. Never add nested pipeline and command costs,
recount in-fit diagnostic time, or recharge reused control fits. Shared-cgroup
observations and process-lifetime RSS do not establish isolated fit peaks. This
companion closes the accounting disclosure; it does not manufacture a complete
whole-campaign total.

## 2. Avoid source-hash cycles

1. Finish the required scientific outcomes and assessment, stop all actual
   monitor windows and seal their observed scope, then complete execution-cost,
   calibration-disposition and extension-overlay records. Preserve interrupted
   monitor windows and missing costs. These records must not depend on a future
   synthesis, QA, release manifest, archive or publication sidecar. Finish all
   covered code/documentation edits before the gate.
2. Generate and inspect the ordinary benchmark report and the predeclared
   model-comparison media. Read [the report guide](../report/v03/README.md) for
   source checks, fixed replication0 selection, unavailable native artifacts,
   diagnostic labels and actual visual QA. Use the external layout wrapper and
   subsequent font embedding for the ordinary revised report; both preserve
   package source and update the saved report inventory:

   ```bash
   "$STESN_PY" tools/render_benchmark_layout.py "$STESN_REVISED_RUN" \
     --output report/v03/revised_media_complete \
     --evidence results/v03/rendering/revised_layout.json
   "$STESN_PY" tools/embed_report_fonts.py report/v03/revised_media_complete
   ```

   The recovered execution selects `report/v03/revised_media_complete` after
   preserving an interrupted, unsealed output separately. The
   `results/v03/rendering/revised_output_path_decision.json` receipt binds the
   original and continuation recipes; neither partial output is a completed
   report input. Here `STESN_REVISED_RUN` is the history-rule-selected revised directory.
   The wrapper retains its numerical-artist checks around the recorded display
   corrections. It generates the twelve planned core movies; the two explicit
   G4 comparison-tool invocations supply the remaining planned movies when
   their requested native artifacts are available. Keep availability and QA
   scope explicit.

   The restored release additionally includes an audited correction of four
   ordinary forecast legends and the G4/A tick-label margin. Its authority is
   `results/v03/rendering/revised_forecast_layout_correction.json`; numerical
   artists, limits, scientific inputs and all unaffected payload bytes remain
   unchanged. The original report is preserved whole, and the corrected
   `report_manifest.json` explicitly lists the presentation supplement. The
   commands above describe fresh generation; do not use them to overwrite the
   sealed restored report. Preserve the exact retained helper/plan and both
   report versions when reproducing or auditing the correction.
3. If an older synthesis exported Markdown copies, rerender it with the current
   renderer before the final gate. Current copies use `.md.txt`; their bytes
   are unchanged but they no longer enter the covered report-source set.
4. Run the final local gate in the locked CPU-neural environment, commit the
   exact covered source and verify the public commit's CI. Do not copy the
   generated final gate into `docs/` after validation. A commit/tag operation
   itself does not change covered source bytes.
5. Render the final synthesis with `--release-evidence
   results/release_manifest.json`, both bound `--test-evidence` XML files and
   the actual scientific receipts. Complete PDF/HTML/media QA and save its
   generated receipt under `results/v03/release_evidence/`. If QA requires a
   renderer or covered-doc edit, rerun the applicable final gate for that source.
6. Package the closed artifacts. Upload and read back the exact immutable
   assets. The post-upload R20 receipt is a separate additional release asset,
   referencing the archive/index hashes; it must not change those archives.

Final local gate, using a new isolated environment:

```bash
export STESN_PY=.venv/bin/python
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$STESN_PY" tools/release_check.py --with-neural \
  --fresh-venv /tmp/bstesn-v03-final-release
```

Inspect the resulting receipt and source/installed test counts, warnings,
optional scope, distributions and CLI checks. Do not substitute this runbook
or an old successful receipt for an observed current gate.

## 3. Independent ZIP volumes

The default packager still emits the original single ZIP. For a full numerical
delivery, use `--archive-volume-bytes 251658240` to emit ordinary, independently
extractable ZIPs of **at most 240 MiB each**. Naming and membership are
deterministic for the same logical inventory and cap:

```text
bayes_stesn_implementation.part00001.zip
bayes_stesn_implementation.part00002.zip
...
release_artifacts.json
release_volumes.json
RESTORE_VOLUMES.txt
bayes_stesn_repository.bundle
SHA256SUMS
```

The full logical payload inventory and provenance remain in
`release_artifacts.json`. Every volume embeds that identical document, the
extraction guide and a distinct per-volume manifest. `release_volumes.json`
maps every original payload to one volume and records each ZIP's actual byte
count and SHA256. No original payload is fragmented, omitted, or duplicated
across volumes; common generated metadata is intentionally repeated.

Volume mode uses `ZIP_STORED`, fixed ZIP timestamps and preserved file
permissions. Most retained numeric/movie/image artifacts are already compressed.
Exact header/member/metadata sizes are planned before writing, then actual
sizes, ZIP CRCs and every archived payload SHA256 are checked. Each input is
also hashed while streaming into the archive. Only verified staged outputs
are moved into the delivery directory; there is no intermediate monolithic ZIP
or second complete archive copy. Existing named outputs are not overwritten.

A single original file that cannot fit together with required metadata is an
explicit error before volumes are written. Inspect the named artifact and
resolve the delivery constraint; never drop a checkpoint or change a scientific
output to make the package fit. The 240-MiB ceiling is a delivery constraint,
not an estimate of model memory or evidence volume.

The default volume admission retains **4 GiB free disk** beyond the planned
archive and sidecars. `--archive-reserve-bytes` makes that reserve explicit;
retain the project reserve for an actual campaign. Input evidence remains on
disk, so budget its existing size plus one archive set and the reserve. A later
confirmation campaign requires a new measured storage estimate.

The existing supplemental budget remains 2 GiB and 10,000 files by default.
`--supplemental-max-bytes` and `--supplemental-max-files` permit larger explicit
measured budgets. They do not authorize incomplete benchmark campaigns, unsafe
paths, symlinks or temporary files. Overlapping primary/supplemental inputs are
deduplicated in the final logical payload mapping; the declared supplemental
budget counts the explicitly selected supplemental trees.

## 4. Concrete full-evidence packaging recipe

Run this only after all selected artifacts are closed. It uses the conventional
paths from the study and report guides. The ordinary revised benchmark report
identifies the selected run; the lookup below requires exactly one matching
manifest and does not select a history by its outcome. Keep QA rasters outside
these delivery subtrees. Missing planned comparison movies require a recorded
availability reason, without substituting another replication.

Before measuring the final payload, require quiescent writers and inspect all
excluded atomic-staging candidates. A candidate may be moved only after proving
that it is absent from every sealed artifact/input inventory: job/selection/data
receipts, coordinate checkpoint bindings, P5 inventories, reducer inputs,
media/report inputs and the cost summary's immutable input set. Search retained
JSON/JSONL/YAML and NPZ state manifests for its relative and absolute paths,
unique basename and SHA256; classify every hit. An identical digest at another
canonical path is a hit to investigate, not automatic proof of a reference to
the staging path. Any reference or unresolved uncertainty stops movement.

For a verified orphan, preserve its original path, bytes and hash in a reviewed
external quarantine, check the bytes immediately before and after movement,
and retain restoration instructions. Preserve actual checkpoints. Keep the
small finalized quarantine receipt with the release evidence; do not delete
staging files to make the packager pass or move anything in an active scope.
Remeasure the exact selected payload after this review.

Measure packaging with its plan, growing log and terminal resource receipt in
an **external directory outside every selected payload tree and outside the
new archive output**. A package cannot inventory its own active log or a
receipt that is written only after packaging finishes. Deliver that completed
measurement with the separately checksummed packaging/publication sidecar.
The complete `results/v03` selection below includes the retained support;
standalone files elsewhere under `results/validation` are not automatically
selected unless bound by the final gate's validation inventory.

```bash
export PYTHONPATH=src
"$STESN_PY" - <<'PY'
import json
from pathlib import Path
import subprocess
import sys

from bayes_stesn.io import content_hash

root = Path.cwd()
primary_report = root / "report/v03/revised_media_complete"
synthesis = root / "report/v03/rendered"
primary_manifest = json.loads((primary_report / "run_manifest.json").read_text())
candidates = [path for path in (root / "results/v03/development").glob("revised_L*")
              if (path / "manifest.json").is_file()
              and json.loads((path / "manifest.json").read_text())["identity"]
                  == primary_manifest["identity"]]
if len(candidates) != 1:
    raise ValueError("Need exactly one revised run matching the saved ordinary report")

# Validate the final synthesis's exact recorded input recipe, without rendering.
saved = json.loads((synthesis / "report_manifest.json").read_text())
if saved.get("schema") != "saved-v03-study-report-1.0":
    raise ValueError("Need the completed saved-artifact synthesis")
if saved["identity"] != content_hash({key: value for key, value in saved.items() if key != "identity"}):
    raise ValueError("Synthesis receipt identity changed")
command = saved["command"]
if Path(command[0]).name != "render_v03_study.py":
    raise ValueError("Unexpected saved report command")

# Check every recorded input and generated output before validating the same
# input recipe. A newer file at the same input path cannot stand in for the
# bytes used to generate this final PDF/HTML.
sys.path.insert(0, str(root / "tools"))
from render_v03_study import Inputs, safe_member, sha256
if saved["renderer_sha256"] != sha256(root / "tools/render_v03_study.py"):
    raise ValueError("Synthesis was generated by a different renderer")
verified = Inputs()
for entry in saved["inputs"]:
    verified.add(entry["path"], expected=entry["sha256"], size=entry["bytes"])
for entry in saved["files"]:
    verified.add(safe_member(synthesis, entry["path"]), expected=entry["sha256"], size=entry["bytes"])
verified.verify_unchanged()
subprocess.run([sys.executable, "tools/render_v03_study.py", *command[1:], "--check-inputs"], check=True)

supplements = [root / "results/v03", synthesis]
for name in ("revised_comparison", "sensor100_comparison"):
    directory = root / "report/v03" / name
    if directory.is_dir():
        supplements.append(directory)
files = [path for directory in supplements for path in directory.rglob("*") if path.is_file()]
byte_budget = sum(path.stat().st_size for path in files)
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
output = root.parent / "deliverables" / ("v03-" + commit[:12])
command = [sys.executable, "tools/package_release.py",
    "--benchmark-run", str(candidates[0]), "--benchmark-report", str(primary_report),
    "--supplemental-max-bytes", str(max(1, byte_budget)),
    "--supplemental-max-files", str(max(1, len(files))),
    "--archive-volume-bytes", str(240 * 1024**2),
    "--archive-reserve-bytes", str(4 * 1024**3), "--output", str(output)]
for directory in supplements:
    command += ["--supplemental-evidence", str(directory)]
print(json.dumps({"output": str(output), "supplemental_files": len(files),
                  "supplemental_bytes": byte_budget}, indent=2))
subprocess.run(command, check=True)
PY
```

This deliberately includes the full `results/v03` tree: reference, tuning,
coordinates, study designs/timing, history, sensors, chosen revised pilot, P5,
calibration, stress, assessments, resource logs and closure records. The normal
packager separately includes the current final validation inventory,
distributions, committed source and verified Git history bundle. Its rejection
of excluded staging/interim files is intentional; inspect any failure instead
of silently narrowing the evidence selection.

If independent confirmation actually completed, retain its complete run and
ordinary saved report and add their pair with `--additional-benchmark RUN
REPORT`; retain the corresponding `--run confirmation=RUN` in the synthesis.
Do not add this pair for an admitted-but-unexecuted or blocked assessment.

## 5. Readback and public completion

Verify the produced `SHA256SUMS`, extract every volume into a new directory,
and verify the logical payload mapping against the extracted files. Validate
the Git bundle and its intended commit. Retain the measured file/byte counts,
volume count and checks in the release closure.

Publish only through the authorized repository connection, with a checked
expected parent and no forced history changes. Verify the public HEAD/CI on the
exact commit, the tag target, release visibility and every uploaded asset's
downloaded bytes/SHA256. Provider metadata alone must be labeled as such if
full byte readback is unavailable. A local package is not a public release.

After readback, emit a separate publication-verification receipt with the
exact commit, CI run/jobs, tag/release/asset identifiers, local/remote hashes,
timestamps and any genuine permission or readback limitation. Give that receipt
its own checksum sidecar and upload it without modifying the earlier archives
or `SHA256SUMS`. This resolves R20 without a self-referential hash cycle.
