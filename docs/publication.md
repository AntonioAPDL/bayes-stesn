# Repository publication status

Status recorded on **2026-11-04**. The MIT-licensed project and its original
history are public at
[AntonioAPDL/bayes-stesn](https://github.com/AntonioAPDL/bayes-stesn).
The default branch is `main`.

Subsequent development uses the [GitHub execution workflow](github_workflow.md):
verified-parent source updates, exact-commit CI, and separately checked release
assets. The one-time history import below does not need to be repeated.

## v0.3: scientific evidence and generated delivery authorities

The recovered v0.3 execution has completed the two initial history conditions,
the one-field sensor study, bounded reference/coordinate/P5 diagnostics,
60 strict-tuning outcomes, 220 revised-pilot outcomes, six calibration cases,
both G5 arms and the corrected zero-refit G6 replay.
The [current findings](results_v03.md) and
[requirements ledger](plans/v03_requirements.yaml) identify their saved
authorities, declared limitations and remaining delivery work. Historical public checkpoints and
their CI receipts do not establish that a later documentation/tool tree has
passed its final source and installed-wheel gate.

The unchanged authority admitted a frozen confirmation of 60 datasets per
scenario. Its uncapped planning request was 102, and
`mcse_target_not_met_at_cap` remains the recorded planning shortfall. Actual
qualification and achieved precision are separate findings.

Final confirmation execution and native integrity:
The frozen campaign completed all 2,640 scheduled jobs over 240 independent
fields, with 60 fields per scenario: 2,630 outcomes qualified and 10 used
declared fallbacks (seven ESN and three basis-GRU outcomes). No jobs remain
pending or incomplete. The [completed interpretation](../results/v03/release_evidence/confirmation_result_interpretation.json)
binds these counts and their statistical scope. The separate native integrity
invocation used positive `--max-jobs 1`, exited zero and recorded zero submitted
attempts; all 13,221 preexisting immutable files and their membership were
unchanged. Its [postcheck](../results/v03/confirmation/operational/integrity_000/postcheck.json)
and [root exit observation](../results/v03/confirmation/operational/integrity_exit_000/root_exit_observation.json)
retain the actual native result and successful outer exit.

Scientific disposition: Within the frozen four-scenario experiment, equal-quarter macro normalized
native CRPS was higher for Bayes ESN than for separable GP or basis GRU in both
tracks. All four ESN-minus-comparator pointwise 95% bootstrap intervals were
positive. Track A has conditional ESN advantages over GP in G3 and G4 and over
GRU in G4; all eight primary track-B scenario differences were positive.
Realized macro MCSE was below 0.02 for both track-A comparisons and above 0.02
for both track-B comparisons. These achieved values do not reclassify the
frozen `mcse_target_not_met_at_cap` planning status at R=60 or its uncapped
request of R=102.

These native CRPS contrasts condition on each cell's common-qualified fits.
Fallback point predictions remain in operational point-loss populations and
are not substituted into distributional comparisons. The intervals are
pointwise; they do not establish multiplicity-adjusted simultaneous superiority
or generalization beyond this experiment.

The [compute and recovery disclosure](results_v03.md#compute-and-memory-scope)
distinguishes confirmation interruptions 000 and 001 from filesystem-loss
event 002. The latter restored the verified checkpoint 33 prefix of 1,895 jobs;
the original stdout observation at 1,917 remains unverified. Original lost
post-checkpoint artifacts and measurements remain unavailable; valid later
recomputation has its own receipts. Lost-segment CPU and other unmeasured costs
remain unavailable, and the unbound finish-only guard is excluded from complete
invocation totals. Missing v10 evidence leaves its time coverage, sample count,
shared-cgroup memory and disk extrema unavailable; that monitor did not sample
CPU. Scientific completion does not repair those accounting gaps or change
the original native flags.

The revised ordinary report and two fixed comparisons have complete
saved-artifact and sampled visual reviews; all 26 revised ordinary PDF pages
were actually viewed. That recorded scope excludes continuous playback and
retains its blocked browser QA. Separate confirmation-media review scope:
The [actual confirmation QA](../results/v03/rendering/confirmation_visual_qa.json)
records all 12 planned cases passing mechanical checks, actual review of 72
first/middle/last MP4 and GIF images and all 26 PDF pages, embedded fonts and
zero blocking visual findings. Static HTML checks cover 22 embedded PNGs and
13 local references. Browser viewport review remains blocked and continuous
playback remains unperformed. The preselected replication-0 cases are
illustrations, not aggregate model rankings; this ordinary-report receipt does
not establish final-synthesis QA or publication readiness.

The [delivery procedure](release_v03.md) identifies the generated final local
gate at `results/release_manifest.json`, the synthesis inventory at
`report/v03/rendered/report_manifest.json`, and the actual inspection record at
`results/v03/release_evidence/inspection_records/final_media_pdf_qa.json`.
Source text and preparation reviews do not establish successful execution of
those later checks. R01–R20 are resolved by the generated
`results/v03/release_evidence/r01_r20_closure.json` and a separate post-upload
publication-verification sidecar containing the actual commit, CI, tag,
release/asset identities and downloaded-byte checksums. The sidecar must not
be inserted into or change the archive whose hashes it verifies. Source
documentation is finalized before the gate; later generated receipts record
the actual gate, synthesis QA, archive readback and publication outcomes
without a source-hash cycle. R20 remains `pending_separate_publication_sidecar`
until that separate publication authority has been produced and verified.

The v0.2.x records below retain their original commits, test counts, numerical
identities and download links. They are historical completed releases, not
v0.3 validation or publication claims.

## Historical v0.2.x source and preserved history

The public [v0.2.1 release](https://github.com/AntonioAPDL/bayes-stesn/releases/tag/v0.2.1)
was published on 2026-11-03 from
`ebd2d1a031796cadae5dbb32a992e10b3f7cf653`. It adds a checkpoint-continuation
correction and retains CI validation evidence. Documentation-only commits may
advance `main`; the release tag identifies the tested code.

The imported `main` and release tag `v0.2.0` were both verified at
`8ee53dfe9f2727d4bf25e26f38d4690007140fd9`. All three original commits,
their parent relationships, authors, and timestamps were preserved:

| Commit | Description |
| --- | --- |
| `fdf8b7604f85b68d15b8832abade13e2c3d5c5d8` | Implement Bayesian spatiotemporal ESNs with verified inference, experiments and reports |
| `48d7e51b1e56f467a52f4e22e83a56a784f11732` | Add reproducible spatiotemporal benchmarks and scalable ESN workflows |
| `8ee53dfe9f2727d4bf25e26f38d4690007140fd9` | Polish benchmark reports and record completed development evidence |

Standard HTTPS Git transport lacked a local authentication credential. After
the user explicitly authorized the public destination and history publication,
the authenticated repository owner created a separate `publication-bootstrap`
branch containing the approved CI workflow, verified source bundle, and
one-time importer. The importer used a repository-scoped GitHub Actions token,
checked the bundle SHA-256 and exact source commit, and created the missing
`main` and `v0.2.0` refs without force. It refused conflicting existing refs.

[History import run 41580072870](https://github.com/AntonioAPDL/bayes-stesn/actions/runs/41580072870)
completed successfully. The remote refs and original history were then read
back and verified, and `main` became the default branch.

## Historical v0.2.1 remote validation and release assets

[CI run 41580781455](https://github.com/AntonioAPDL/bayes-stesn/actions/runs/41580781455)
passed all four jobs on the exact `v0.2.1` source commit:

| Check | Observed result |
| --- | --- |
| Torch-free core, Python 3.11 | 199 tests passed |
| Torch-free core, Python 3.12 | 199 tests passed |
| Torch-free core, Python 3.13 | 199 tests passed |
| Locked CPython 3.12 CPU neural, source | 211 tests and 74 subtests passed |
| Locked CPython 3.12 CPU neural, isolated wheel | 211 tests and 74 subtests passed |

The core jobs intentionally deselect the 12 optional-PyTorch tests; the neural
job exercises them. Both neural receipts record zero failures, errors, skips,
or warnings. The release gate also passed the installed CLI, safe serialization
roundtrip, all 13 benchmark smoke jobs, packaging, and dependency checks. There
was no source drift during the run. Smoke jobs verify execution, not statistical
convergence or model superiority.

The [initial remote run](https://github.com/AntonioAPDL/bayes-stesn/actions/runs/41580207396)
did not pass. Two strict checkpoint tests on Python 3.11 exposed rounding
differences up to 4.44e-16. Checkpoint copies changed proposal-factor storage
from Fortran to C order, permitting a different BLAS accumulation order. The
patch preserves Fortran layout, keeps exact-equality assertions, and adds four
initialized/adapted and array/JSON regression subcases. Four representative
uninterrupted fits remained bitwise identical across 12 compared arrays. No
model equation or saved scientific result was changed.

The release contains the 27-page benchmark PDF, 10-page detailed forecast PDF,
forecast/truth/error and data-evolution MP4s, comparison figure, both complete
result archive parts, original source bundle, and `SHA256SUMS.txt`. All original
nine uploaded file sizes and SHA-256 digests matched the local release files.
Their scientific snapshot remains `v0.2.0`; the patch has not been presented as
a new simulation study. The fresh CI validation ZIP and its separate checksum
preserve all 25 logs and receipts beyond GitHub Actions artifact retention.

Machine-readable records are in `docs/evidence/v021/`. The complete hierarchical
models and MCMC derivations remain in `docs/models.md` and `docs/derivations/`.

## Historical records

An earlier publication attempt was blocked by automatic approval review until
the user confirmed this exact public destination and history publication.
Earlier version 0.1 authentication was also unsuccessful. Those blockers are
resolved for the completed history import above.

The immutable `v0.2.0` source snapshot intentionally retains the publication
status written when that source was archived. This updated document on
`main` supersedes that historical status; the original receipt at
`results/benchmark_v3/publication/receipt.json` in the release archive remains
an accurate record of the earlier attempt. Subsequent documentation commits
may advance `main` without changing the release tag or original history.

