# GitHub execution and publication workflow

This project already has a public repository and preserved Git history at
[AntonioAPDL/bayes-stesn](https://github.com/AntonioAPDL/bayes-stesn). The original
history import is complete. Ordinary development should extend that history;
the one-time bootstrap/import procedure is historical evidence.

## Authority and access are separate

The project owner has authorized implementation, bounded experiments, testing,
commits, pushes and release publication to this repository. Routine reversible
work within that scope does not need repeated confirmation. Actual provider
permissions, branch protections, resource limits and available tool operations
still apply. An agent should finish independent work and preserve a concrete
reviewable result if one external step is blocked.

Use an existing repository-scoped connection or a normally configured Git
credential. Check the current branch/ref and the connection's actual supported
operations before writing. A successful connected GitHub operation does not
imply that a shell has an HTTPS credential, and an unavailable shell credential
does not imply that the connected repository operation is unavailable.

The v0.3 work used the authenticated GitHub connector for Git objects, refs and
Actions evidence. Its Git data operations were sufficient for source changes.
Release creation and file attachment were not exposed as connector mutations;
the already authorized repository UI is the appropriate fallback for those
specific unsupported actions. Account-wide permission expansion and credential
extraction are not part of this workflow.

## Publish the exact reviewed source

1. Read the current remote `main` ref and compare it with the intended parent.
   Preserve unrelated local changes and inspect unexpected remote advances.
2. Finish the intended code, tests, documentation and configuration changes.
   Stage an explicit file list; keep large results, checkpoints and movies in
   the release evidence assets.
3. Build the Git tree from exact staged blobs. When using the Git data API,
   verify each returned blob SHA and the full tree SHA against the local Git
   objects. Chunking a file transfer must preserve its exact bytes.
4. Create a commit with the verified parent and tree, then update `main` with
   the expected old SHA and without force. A ref conflict requires inspecting
   the new remote state; it is not a reason to overwrite that state.
5. Read back the remote commit/tree/ref. Synchronize the local repository while
   preserving the index and working changes. When Git transport is unavailable,
   any reconstruction of a returned Git commit object must match the exact
   remote object SHA before it is installed locally; equivalent-looking metadata
   are insufficient.
6. Verify CI for that exact commit. A previous green commit or a configured
   workflow does not validate later edits. Retain the actual job conclusions,
   source and installed-package test receipts, and validation archive digest.

Do not copy or print authentication tokens in commands, logs, receipts or
repository files. Normal publication needs no token embedded in a remote URL.
Keep provider-specific authentication outside scientific configuration and
experiment identities.

## Finish a release from verified evidence

The source commit, scientific source fingerprint, experiment identities and
release tag have different roles. The final release gate binds all hashed
source/tool/configuration/documentation inputs; completed scientific receipts
retain the numerical source that actually produced them.

Use the existing `tools/release_check.py` and `tools/package_release.py` gates,
then inspect the finished report and media. Package only the declared current
evidence subtrees, with explicit byte/file budgets and room for temporary
archive duplication. The packager validates inventories, ZIP contents and
checksums; it does not make an incomplete experiment complete.

Create the release against the exact tested commit. Attach the source
distribution, complete scientific evidence, report/media, history bundle and
checksums before publishing the finished release. If an asset exceeds the
provider's current per-file limit, use separately checksummed archive parts
and document how to extract/reassemble them. Preserve existing tags.

After publication, read back the tag target and every asset's name, byte count
and available digest. Retain a public release receipt and the validation
archive so verification does not depend on temporary Actions retention. A
documentation-only post-release status update may advance `main`; it must not
move the release tag or imply that the tag contains that later update.

If automatic approval review rejects an external action, record the actual
action and stated reason. Continue through a materially safer supported route
when available. Ask the owner only for a genuinely unresolved boundary, after
the concrete deliverable is ready, and identify the exact blocked final step.

See [publication history](publication.md), [release verification](reproduction.md),
and [the ordered study reproduction guide](reproduction_v03.md).
