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
ownership evidence, tests, and research limits.
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

When operating inside the ChatGPT PDF workflow, run that environment's
required PDF authoring marker immediately before the first generation command.
The renderer itself remains an ordinary reproducible Python CLI outside that
environment.

## 3. Optional evidence

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
decision gates. The sensor audit displays saved spatial/temporal ranks and
geometry-only unit-iid-noise amplification, with fit qualification visible.
Its fixed 100-sensor reference basis is a diagnostic choice; it does not
change a fitted basis, prior or forecast, and it is not actual correlated
forecast variance. The one-field sensor study has no replication MCSE.

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
