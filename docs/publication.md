# Repository publication status

Status recorded on **2026-11-03**. The MIT-licensed project and its original
history are public at
[AntonioAPDL/bayes-stesn](https://github.com/AntonioAPDL/bayes-stesn).
The default branch is `main`.

## Published source and preserved history

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

## Remote validation and release assets

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

