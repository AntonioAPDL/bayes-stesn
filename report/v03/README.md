# v0.3 study synthesis

`tools/render_v03_study.py` builds a portable HTML report, a typeset PDF,
PNG/SVG figures, copied machine-readable summaries, and an input/output
inventory. It reads completed saved experiments. It does not generate data,
fit models, tune candidates, forecast, or select a favorable illustration.

The PDF and HTML explain the model hierarchy, exact unit-Jacobian sampler
change, study designs, selected configurations and candidate/recheck counts,
qualification, operational and conditional-native populations, paired and
joint scores, error evolution, measured wall/CPU cost, accuracy versus time,
learning curves, the saved G1 privileged-parameter reference, separate
raw-versus-calibrated intervals, bounded extension/path results, memory
ownership evidence, an explicitly supplied conditional prior assessment,
separate paired G5/G6 stress comparisons, tests, and research limits.
Numerical conclusions come from the supplied artifacts; missing optional
studies remain visibly unexecuted.

## 1. Validate the inputs first

Install the package's `report` extra in the declared numerical environment.
The following command is a template: substitute the actual **completed,
current-source** experiment directories selected for the release.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
python tools/render_v03_study.py \
  --coordinate /path/to/completed/coordinate-development \
  --run reference=/path/to/completed/reference-reliability \
  --run tuned=/path/to/completed/tuned-development \
  --output report/v03/rendered \
  --check-inputs
```

`--check-inputs` does not create the output directory, figures, HTML or PDF.
It validates:

- Content identities of run, design, computation, data and scientific receipts.
- A complete, source-stable schema-4 study with the exact declared terminal
  jobs, no pending jobs, no unresolved infrastructure failures, and no
  invalidation receipt.
- The current scientific source fingerprint for every main benchmark run.
- Every retained artifact's byte count and SHA-256, safe relative paths, and
  unchanged inputs throughout validation.
- Population counts, means and MCSEs against saved dataset-level scalar
  receipts; paired normalized differences against the saved method losses.
- Coordinate checkpoint internal integrity, terminal fit-state/checkpoint
  correspondence, and the common data/target/seed for both algorithms.
- A confirmation run's matching frozen design, when confirmation evidence is
  explicitly supplied.

Validation of scalar reductions does not recompute scores from observations
or predictive samples. Large run artifacts are hashed in bounded chunks;
coordinate checkpoint integrity is checked one file at a time, with an
explicit 128 MiB uncompressed checkpoint limit.

## 2. Generate the report

After input validation, use the same command without `--check-inputs`.
The output directory must be empty or contain an earlier report made by this
specific recipe. A numerical input directory cannot be an output directory.
Rerendering first verifies earlier output hashes, removes only inventoried
generated files, and leaves unrelated files untouched.

The output includes:

| File or directory | Contents |
|---|---|
| `index.html` | Responsive report, figures, evidence links and supplied video players |
| `bayes_stesn_v03_report.pdf` | Static print companion with embedded fonts |
| `figures/` | PNG and editable-text SVG charts |
| `evidence/` | Run manifests and summaries, model/derivation notes, engineering evidence |
| `media/` | Copies of explicitly supplied, hash-verified animation assets |
| `evidence_summary.json` | Study/source identities and explicit optional-study status |
| `report_manifest.json` | Renderer/runtime identity, all input/output hashes, validation scope |

The report manifest initially records PDF visual QA as pending. Before
publication, render the PDF with Poppler and inspect every page for clipping,
unreadable charts, unexpected blank space, and broken glyphs. Also inspect
HTML on wide and narrow screens and play representative supplied animations.
Record visual QA separately; changing the report manifest requires recomputing
its content identity. Do not manually alter inventoried assets without a new
render or an explicit audited postprocessing recipe.

Plot typography is checked against its actual tight bounding box and the PDF's
A4 placement limits. A plot whose smallest visible label would be below 7.5 pt
is rejected. Tables use short display labels while exact model/configuration
identifiers remain in linked JSON. Selection tables aggregate across datasets;
adding replications does not duplicate a fixed configuration in the PDF.
JUnit totals are labeled **cases, including subtests**, rather than distinct
test functions. PDF commit/source cells use 12-character prefixes; HTML and
machine evidence retain full hashes. Responsive CSS constrains media to its
container and gives wide tables horizontal scrolling. Static layout/link checks
and actual browser review are separate QA activities and must be recorded as
such; blocked browser access is not a successful browser review.

When operating inside the ChatGPT PDF workflow, run that environment's
required PDF authoring marker immediately before the first generation command.
The renderer itself remains an ordinary reproducible Python CLI outside that
environment.

## 3. Optional evidence

### Current local verification receipts

Historical test and release snapshots in `docs/evidence/v03` remain visible as
historical evidence. To include the final local source/installed-wheel gate,
pass its generated receipts directly instead of copying them into that source
directory:

```text
--release-evidence results/release_manifest.json
--test-evidence results/validation/source_pytest.xml
--test-evidence results/validation/installed_pytest.xml
```

The optional `--release-evidence` accepts one complete schema-2 receipt from
`tools/release_check.py`. The renderer uses the release checker's exact source
inventory function and requires the full file/hash mapping to match the current
checkout, including documentation, tests and reporting tools. Added source
files are detected as well as changed or omitted files. Every source input,
distribution and validation-inventory member is hashed; unsafe paths, duplicate
entries and mismatched sizes or hashes are rejected. Recorded step-log hashes,
source/installed JUnit totals and the corresponding harness receipts must agree.
A receipt marked passed cannot contain failed steps or unsuccessful tests. The
complete source set and input hashes are checked again before rendering ends.
The displayed current source-scope identity is the canonical hash of the entire
source-file/hash mapping, not a Git commit or a scientific-source fingerprint.

`--test-evidence` is repeatable. With `--release-evidence`, supplied XML must be
the source or installed test receipt bound by that current gate. An explicitly
skipped source suite does not authorize leftover source-test XML. Without a
current release receipt, additional XML is accepted as a **supplied snapshot
with unbound source**, never labeled as current verification. JUnit totals can
include successful subtests without separate `testcase` elements; the renderer
retains the producer's counts and does not claim a count of distinct test
functions. Malformed explicit XML and incomplete outcome counts are rejected.

Finish all covered source, tool and documentation edits before the final local
gate. The subsequent synthesis can consume receipts under `results/` without
changing the source hash set. Generated PDF/HTML/JSON/media outputs are separate
from the release checker's covered reporting scripts and Markdown. Exported
copies of the three source documents use `.md.txt` names and retain their exact
bytes, so rendering does not create additional covered Markdown source files.
If reusing an older synthesis that contains `.md` copies, rerender it with the
current tool **before** the final gate to replace those legacy generated copies.
Keep visual QA in a separate generated receipt, and do not edit covered
documentation after the gate without rerunning it. These inputs establish the
recorded local gate; exact remote commit, CI and published-asset verification
remain separate.

### Explicit confirmation admission or scientific block

Pass the saved assessment produced by `tools/assess_confirmation_v03.py`, along
with its complete development run:

```text
--run revised=results/v03/development/revised_L1000
--confirmation-evidence results/v03/confirmation/assessment.json
```

Use the actual history selected by the sealed rule and the actual new assessment
path. This option does not run the freeze factory, simulate final data, fit
parameters or produce confirmation forecasts. The assessment helper and report
renderer are themselves bound in the assessment's source inventory, so finalize
these tools before producing the assessment intended for this release.

The renderer checks the assessment identity, current package and one-thread
numerical runtime, matching supplied pilot, complete source/artifact inventory,
draft and primary-comparison inputs. It verifies the independent G3 numerical
prerequisite when G3 is declared; a roster without G3 is **not applicable**, not
a failed prerequisite. It independently recomputes the saved fixed-weight macro
comparisons with the declared seed and 1000 bootstrap resamples, every usable
pair count and the exact deficient-cell list.

A blocked assessment must retain the unchanged factory's specific
insufficient-common-pairs `ValueError`, with `admitted: false` and `freeze: null`.
An unrelated error is not accepted as this scientific block. The report lists
every primary cell and its required/usable pair counts, preserves a
development-only conclusion and creates no substitute freeze.

For admission, the saved freeze and resolved configuration must validate with
the existing core verifier, bind the same pilot/draft and reproduce the
replication/precision plan. The report displays frozen attempted R, uncapped
requests, planned MCSEs and the actual cap status. `mcse_target_not_met_at_cap`
remains an explicit shortfall. **A completed assessment is not admission;
admission is not confirmation execution or achieved precision.** A separately
supplied completed confirmation run is recognized only against its exact freeze.

The portable report exports `evidence/confirmation_assessment.json` and
`evidence/confirmation_protocol.json`. It exports
`evidence/confirmation_freeze.json` only for a validated admission. These are
generated JSON copies, not additional covered Markdown source files. Omitting
the option produces an explicit “not supplied” gate section without inferring a
decision from a complete development campaign.

### Matched learning curves

Pass `--curve DIRECTORY` or `--curve DIRECTORY/curve_design.json`. Repeat the
option for separate history and sensor experiments. The renderer
reads the resolved `benchmark-curve-2.0` design and its completed condition
directories. It verifies matching generated fields, target site IDs, forecast
origin and normalization-reference metadata across conditions. Missing
condition summaries are listed as unexecuted planned values. A supplied
condition with an incomplete or invalidated summary is rejected; it is not
silently removed to improve the plotted curve.

History and sensor curves retain separate interpretations. Connected points
are descriptive comparisons at the declared intervention values, not a
monotonicity claim. A fixed architecture and a newly tuned architecture are
different procedures and retain their selection mode in the evidence.

Add the sealed post-study decision/representation outputs with repeatable
`--curve-audit JSON` options, alongside their matching `--curve` inputs:

```text
--curve-audit docs/evidence/v03/history_choice.json
--curve-audit docs/evidence/v03/sensor_representation.json
```

The renderer verifies the audit's content identity, source fingerprint,
curve identity, design/tool/recipe hashes and complete source-receipt
inventory. It checks history paired differences and qualification against
the supplied condition receipts, and checks the recorded time sums and
decision gates. The frozen history rule defaults to 1000 rows, promoting
2000 only if both ESN tracks meet the declared practical native-CRPS lower
bound, common-qualified-pair minimum, qualification and all-roster cost
conditions. Every comparator's native CRPS, native RMSE and operational RMSE
effects remain visible; failure of this exploratory gate is not evidence
that more data have no benefit. The sensor audit displays saved spatial/temporal ranks and
geometry-only unit-iid-noise amplification, with fit qualification visible.
Its fixed 100-sensor reference basis is a diagnostic choice; it does not
change a fitted basis, prior or forecast, and it is not actual correlated
forecast variance. The one-field sensor study has no replication MCSE.

### Separate paired G5/G6 stress comparisons

Pass the completed `v03-paired-stress-summary-1.0` audit with its corresponding
revised control run:

```text
--run revised=/path/to/completed/revised-development
--stress-audit /path/to/sealed/stress_summary.json
```

The stress audit is a separate experiment. It binds the immutable stress design,
recipe/tool inputs, original control manifest, complete G5 arm manifests, G6
replay receipts and saved artifact hashes. Both G5 noise arms are compared with
their matched G4 fields; blind G6 compares the same original G2 forecasts with a
future whose hidden velocity shift starts at lead 25. G6 reuses retained native
forecasts or replays frozen models with their original seeds, verifies every
original G2 score, and performs **zero parameter refits or post-shift adaptation**.
The renderer itself never generates a field or replays a forecast.

For each field, both sides of a paired dimensionful loss use the saved **control
training-only SD**. Standalone G5 normalization receipts remain unchanged and
are not used as a cross-arm denominator. Coverage indicators remain dimensionless.
The renderer independently checks original score copies, exact pair IDs,
qualification/fallback intersections, normalization powers, means, MCSEs and
the recorded paired-dataset bootstrap intervals. Native effects use the exact
common-qualified population; operational RMSE/MAE retain declared fallbacks.
Every primary and declared interval, joint and path companion effect remains
in the exported audit, even when the printed table shows a compact subset.

G5 costs sum recorded ordinary model-job pipelines and exclude shared DGP
generation/controller overhead that the original dataset receipts do not itemize.
G6 separately records generation, forecast reuse/replay, all-score verification,
scoring and serialization. Unknown CPU work remains unavailable and interrupted
accounting remains incomplete. Reused G4/G2 control costs are deduplicated and
shown separately from incremental work. These are measured sums, not elapsed
parallel-campaign time. Coupled controls and stress arms are not pooled as
independent strata in the core macro comparison. Five paired fields support
exploratory pointwise uncertainty, not a precise universal robustness claim.

### G1 known-parameter reference

First run the separate reference-audit tool on the completed current-source
benchmark. The audit is read-only for comparator results; it computes and
saves the analytic G1 reference using known parameters and noisy O history.
The report itself never computes this reference:

```bash
PYTHONPATH=src python tools/audit_v03_references.py \
  /path/to/completed/run --output /path/to/reference_audit.json
```

Pass `--audit RUN_LABEL=/path/to/reference_audit.json` to the report command.
The label must match a supplied `--run`. The audit's content identity, base
manifest/summary hashes, scientific source and all saved oracle-moment
artifact hashes must validate. The report shows paired learned-minus-oracle
losses on identical eligible datasets. The known parameters are privileged
information; realized finite-sample dominance is not guaranteed.

### Separate post-fit calibration

After the existing calibration workflow has saved its results, seal them:

```bash
PYTHONPATH=src python tools/seal_v03_calibration.py \
  --run /path/to/completed/base/run \
  --calibration /path/to/saved/calibration-study --replication 0
```

This writes only `calibration_evidence.json`. It never loads a fitted model
object, refits parameters, regenerates observations, or calls a forecast
method. It binds the actual base manifest, model, receipt and simulation
hashes; validates the exact shared simulation prefix, O/C/U roles, disjoint
calibration windows and fixed parameter cutoff; and checks saved raw
RMSE/median-MAE/empirical-CRPS, interval bands and calibrated scores/WIS.
The default aggregate uncompressed-array allowance is 512 MiB; a larger
explicit allowance is capped at 1 GiB. Temporary numerical working arrays
also consume memory, so run this verifier outside heavy fitting waves.

Pass `--calibration RUN_LABEL=/path/to/saved/calibration-study` to the report.
Both the raw and adjusted intervals are evaluated at the same later origin.
The original predictive samples remain unchanged, and no calibrated joint
law or calibrated CRPS is invented. Empirical transfer to U is not an
exchangeability-based coverage guarantee. Infinite finite-block bands remain
visible instead of being discarded.

### Extension decisions and actual bounded studies

Pass `--extensions decisions.json`. This option is repeatable: provide the
policy/disposition document and every completed P5 run directory you want
to include. The simple supported disposition shape is:

```json
{
  "schema": "extension-dispositions-1.0",
  "extensions": [
    {
      "name": "Grouped ridge",
      "status": "deferred",
      "rationale": "Await a qualified reference and a prespecified group-shrinkage comparison."
    }
  ]
}
```

The `rows` and `decisions` list aliases are accepted. The P5 scope document
with `scientific_results_in_this_file: false`, `id`, `default_disposition`,
and `trigger` is labeled **implementation scope; no experiment results**.
Saved P5 decisions may use `disposition`. A completed P5 directory must have
the matching current-source manifest, `decisions.json`, `inspection.json`,
and terminal `output_inventory.json`. Every inventoried output is checked.
Its baseline-model and evaluation-run hashes must bind a supplied main run.
Actual ablations display qualification, exact paired score differences and
negative findings; an unqualified variant has no invented native predictive
comparison. Path-sensitivity studies validate the 128/256/512 sample-prefix
hashes against the saved joint paths. Their MCSE excludes posterior MCMC
error and is not across-dataset precision.

For example, add each available actual study explicitly:

```text
--extensions docs/evidence/v03/extension_scope.json
--extensions docs/evidence/v03/extension_findings.json
--extensions results/v03/extensions/release_ar_g4_existing_fronts
--extensions results/v03/extensions/release_ar_g3_excitable_field
--extensions results/v03/extensions/release_paths_g4_existing_fronts
--extensions results/v03/extensions/release_paths_g3_excitable_field
```

Disposition documents can bind terminal inventories using
`evidence_inventories: [{path, identity, sha256}]`; repository-relative paths
are accepted. Every referenced inventory's output hashes are checked as well
as its own identity. When both original scope and final findings are supplied,
the current disposition table uses the final findings once per extension;
the original scope is separately linked, not presented as a second current
decision. The path-count policy is shown separately from the path sampler:
its predeclared rule, saved normalized differences, baseline training scales
and final choice must agree. Selecting 512 as the finite reference ceiling
does not establish negligible Monte Carlo error or adequate model dynamics.
An entry can also supply `run_directory` to bind its decision
to a complete benchmark run. A narrative recommendation never becomes a
measured ablation result merely by being included in the report.

If a final-source disposition includes a source replay, its historical
disposition identity and hash remain bound separately. The report explicitly
counts the reused fields once: exact replay under a new source adds zero
independent datasets and does not turn a failed gate into a qualified result.

A later release closure uses a new immutable overlay rather than changing a
P5 findings file already referenced by study configurations. Pass the overlay
with another `--extensions` option. Its schema is
`v03-release-extension-dispositions-1.0`, with `identity`, `source_fingerprint`,
`base_findings: {path, identity, sha256}`, `source_inputs: [{path, sha256, bytes}]`
and an `extensions` list containing the actual later dispositions/rationales.
The inventory includes the unchanged base findings and every cited later
calibration, stress or confirmation-gate receipt. The renderer verifies these
bindings and gives the overlay priority over P5 findings and implementation
scope, independently of argument order. Conflicting overlays at the same
priority are rejected. Earlier scientific findings and their hashes remain
unchanged.

If an explicitly optional curve or extension path is absent, the report says
`not available`. If it is omitted, the report says `not supplied`. Existing
malformed or incompatible evidence is an error.

### Saved first-step diagnostic

Add `--diagnostic docs/evidence/v03/g3_saved_first_step_review.json` to include
the separate G3 alignment, decoder and saved-path review. The input hashes
must bind a supplied benchmark model and the matching complete fixed-posterior
path-study inventory. The report checks the scalar orthogonal MSE decomposition
and conditional mean-MCSE relationship; it never replays the decoder or calls
prediction. The projection uses the realized **noisy observed target** and
the saved fixed decoder span. It is an evaluator-only lower bound for that
target/span, not a latent-field result, a permitted forecast, or a universal
irreducible-error estimate. A larger path budget does not repair a restricted
decoder or poor long-horizon dynamics.

### Conditional prior moments

Add the separately saved mathematical assessment alongside its two matching
completed P5 ablations:

```text
--extensions results/v03/extensions/final_ar_g4_existing_fronts
--extensions results/v03/extensions/final_ar_g3_excitable_field
--prior-check docs/evidence/v03/conditional_prior_check_final.json
```

The typed `conditional-fixed-design-prior-check-1.0` input binds the scientific
source, prior-check tool hash, numerical environment, P5 manifests/inventories
and both saved fitted archives. The renderer reads only the model metadata and
frozen `Phi`, `Z` and `reference_Z` arrays; it never loads posterior arrays.
Metadata and selected design members each have a 32 MiB uncompressed limit.
It checks actual saved prior parameters, feature hashes, closed-form variance
reductions, response-unit conversion, rho tail probabilities and the recorded
quadrature agreement. It does not repeat quadrature, generate prior trajectories
or refit a model.

These are prior moments conditional on the frozen encoder and historical feature
paths. They are distinct from posterior conditional scale summaries and from
autonomous prior-predictive rollouts. The report shows the actual inverse-gamma
shape and scale; the current shape of two gives finite prior variance and an
infinite fourth moment for a scalar predictive observation about its frozen
center. A symmetric signed rho prior averages positive and alternating residual
dependence: zero odd-lag marginal covariance does not remove dependence from
individual draws. This retrospective mathematical check makes no model-adequacy,
forecast-coverage, promotion or additional-independent-replication claim.

### Historical coordinate evidence

The default requires the coordinate comparison to match the current source.
An ancillary historical comparison can be included only with
`--allow-coordinate-source EXACT_64_CHARACTER_SHA`. The supplied SHA must
equal its recorded source fingerprint. The report visibly labels this
exception and keeps the coordinate experiment separate from current-source
forecast studies. This flag never relaxes the source requirement for a
main `--run`.

## 4. Reuse the existing animation tools

The synthesis renderer does not introduce a second animation implementation.
The existing tools render actual observations, saved forecasts, signed errors,
and leadwise error evolution from retained arrays:

```bash
PYTHONPATH=src python tools/render_benchmark_layout.py \
  /path/to/completed/run --output /path/to/core-report

PYTHONPATH=src python tools/render_model_comparison.py \
  /path/to/completed/run \
  --scenario g3_excitable_field --replication 0 --track B \
  --output /path/to/model-comparison
```

The comparison tool requires its named models' retained native forecasts.
Unqualified fits, when available, remain explicitly diagnostic illustrations.
It never substitutes an operational fallback without labeling it.

To include existing media in the synthesis, add:

```text
--run tuned=/path/to/completed/run
--media tuned=/path/to/core-report
--media tuned=/path/to/model-comparison
```

Each media label must match a supplied run label. Its report/comparison
manifest must carry that run's identity and a valid asset inventory. Files
are copied without reforecasting or rerendering. The core animations use
shared truth/forecast scales and symmetric signed-error scales, with unscored
locations shown in gray. A displayed replication is an illustration, not
aggregate evidence of superiority.

## 5. Interpretation and verification

Operational means include terminal scientific failures through the declared
fallback. Native CRPS and interval scores are conditional on qualification.
Paired CRPS uses the common-qualified dataset intersection. Counts appear
beside comparisons; no site/lead/path pseudo-replication is used. Macro
weights are never renormalized over missing scenario cells. Pointwise
intervals do not imply simultaneous superiority, and MCSE planning is not
hypothesis-test power.

The ownership probe concerns one isolated constructor/allocation workload.
It does not establish full-model memory or universal speed. Saved test XML
files retain their timestamps and do not automatically validate source changes
made afterward. A report build does not publish a GitHub release; publication
and release-asset verification remain separately recorded operations.

Focused checks run without any model fitting or PDF authoring:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
python -m pytest tests/test_render_v03_study.py -q
```

These checks use explicitly synthetic saved receipts, exercise source and
artifact rejection, verify populations and paired reductions, and render the
equations, charts and HTML. Final PDF pagination and animation playback still
require the authoring/visual QA step on the actual release inputs.
