# Repository publication status

The intended public repository is `AntonioAPDL/bayes-stesn`. The source is
MIT-licensed. Local tests, a source commit, and a portable Git bundle are
separate from publishing that commit to a public remote.

## Current version 0.2 status

The attempted version 0.2 push to `AntonioAPDL/bayes-stesn` was rejected by
automatic approval review before Git executed. The review requires explicit
confirmation of this exact public destination and the publication of the project
history; general authorization to implement, commit, and push was not sufficient
for the selected owner/repository. Publication has not been retried through an
alternative route. The original rejected-attempt receipt is retained in the
release archive under `results/benchmark_v3/publication/receipt.json`.

This checkout has no configured Git remote. The connected GitHub capability
does not offer repository creation, and the browser session was observed signed
out. No version 0.2 browser authentication handoff was attempted. No version 0.2
public URL, successful push, or remote CI result is claimed by the local release
checker. Local source commits, tests, reports, and the portable archive remain
available while the destination-specific confirmation is outstanding.

The final publication receipt and verified remote should supersede this pending
status when publication completes. A receipt should identify the public
repository URL, the pushed commit, the remote's observed commit, the observed
visibility, and the verification time. Record CI run identifiers and outcomes
only after observing those runs; a workflow file alone does not establish CI
success. If publication remains blocked, record the actual blocked action and
reason without describing an unperformed attempt as completed.

## Historical version 0.1 authentication attempt

During the earlier version 0.1 work, repository creation was unavailable through
the connected GitHub operations, and the browser remained at the sign-in page
after an authentication attempt. That historical failure is not the result of
a version 0.2 publication attempt.

## Portable delivery and remote verification

The release packager creates and verifies a Git bundle for the committed
source. It can reconstruct the project independently of remote publication:

```bash
git clone bayes_stesn_repository.bundle bayes-stesn
cd bayes-stesn
```

The version 0.2 release ZIP also carries the explicitly selected benchmark
runs, numeric results, current validation receipts, PDF/HTML reports, vector
figures, and animation assets. These larger generated artifacts need not be
ordinary source-history objects. The default legacy packaging mode instead
requires the earlier controlled-study evidence.

After authenticated creation or verification of the intended repository, push
the final committed branch to that verified remote and read back its commit
and visibility. The package includes a Torch-free GitHub Actions compatibility
matrix for Python 3.11, 3.12 and 3.13, plus a separate locked CPython 3.12 CPU
neural release job. Their presence describes configured automation; observed
remote runs establish whether it actually executed and passed.

