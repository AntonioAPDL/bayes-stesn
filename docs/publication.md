# Repository publication status

Status recorded on **2026-11-04**. The MIT-licensed project and its original
history are public at
[AntonioAPDL/bayes-stesn](https://github.com/AntonioAPDL/bayes-stesn).
The default branch is `main`.

Subsequent development uses the [GitHub execution workflow](github_workflow.md):
verified-parent source updates, exact-commit CI, and separately checked release
assets. The one-time history import below does not need to be repeated.

## v0.3: final delivery gates pending at this update

The recovered v0.3 execution has completed the two initial history conditions,
the one-field sensor study, bounded reference/coordinate/P5 diagnostics,
60 strict-tuning outcomes, 220 revised-pilot outcomes, six calibration cases,
both G5 arms and the corrected zero-refit G6 replay.
The [current findings](results_v03.md) and
[requirements ledger](plans/v03_requirements.yaml) identify their saved
authorities and remaining scientific work. Historical public checkpoints and
their CI receipts do not establish that a later documentation/tool tree has
passed its final source and installed-wheel gate.

The unchanged authority admitted a frozen confirmation of 60 datasets per
scenario, while retaining its MCSE planning shortfall at the cap. That campaign
is in progress at this 2026-11-04 update. The revised ordinary report and two
fixed comparisons have complete saved-artifact and sampled visual reviews;
all 26 revised ordinary PDF pages were actually viewed. Browser QA remains
blocked and continuous playback unperformed. Final scientific disposition,
complete synthesis and its QA, the local final gate, exact release-commit CI
and new public assets remain pending. The [delivery procedure](release_v03.md)
requires a generated
`results/v03/release_evidence/r01_r20_closure.json` and a separate post-upload
publication-verification sidecar containing the actual commit, CI, tag,
release/asset identities and downloaded-byte checksums. The sidecar must not
be inserted into or change the archive whose hashes it verifies. Source
documentation is finalized before the gate; later generated receipts record
the actual outcomes without a source-hash cycle.

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

