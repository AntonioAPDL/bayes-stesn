# Repository publication status

The project is a local, committed Git repository with MIT-licensed source and
a verified portable Git bundle. Its intended public repository name is
`AntonioAPDL/bayes-stesn`.

**Remote publication is pending authentication.** During this release, the
connected GitHub operations did not offer repository creation, and the browser
remained at the GitHub sign-in page after the authentication attempt. No public
repository URL or remote CI success is claimed. All implementation, tests,
scientific evidence and release files are available independently of that step.

The included bundle can reconstruct the committed project:

```bash
git clone bayes_stesn_repository.bundle bayes-stesn
cd bayes-stesn
```

After authenticated repository creation, the committed `main` branch can be
pushed to the verified remote. The source includes a GitHub Actions workflow
for Python 3.11, 3.12 and 3.13. The current recorded verification is the local
isolated Python 3.12 release check; remote workflow results remain unverified
until those jobs actually run.

The full release ZIP also contains numerical results, posterior illustration
paths, wheel/source distributions and report/media assets. These larger
generated artifacts need not be committed as source-history objects.

