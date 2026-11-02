# Reproducing the implementation and experiments

The release checks separate four questions: whether the mathematical and API tests pass, whether the distributable package installs correctly, whether a small complete fit and forecast executes, and whether a scientific experiment meets its convergence and prediction gates. A successful smoke run answers the execution question only.

## Verified environment and dependency groups

The version locks target **CPython 3.12 on Linux x86_64**. The package declares its broader supported Python range in `pyproject.toml`; identical floating-point results across operating systems, BLAS builds, or Python versions are not promised.

| File | Purpose |
|---|---|
| `requirements-core.lock` | NumPy, SciPy and PyYAML, the complete runtime dependency set |
| `requirements-report.lock` | Core dependencies plus plotting, image, PDF and required transitive packages |
| `requirements-dev.lock` | Core dependencies plus pytest, package-building tools and thread control for benchmarks |

These files pin versions rather than every platform-specific wheel hash. The release check downloads from `https://pypi.org/simple`, saves pip's installation receipt with distribution URLs and hashes, records the installed environment, and hashes the produced wheel and source archive. It does not copy the unrelated packages in the development machine's environment into a lockfile.

Create an isolated environment without `--system-site-packages`:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --index-url https://pypi.org/simple -r requirements-dev.lock -r requirements-report.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
source .venv/bin/activate
```

For a runtime-only installation, install `requirements-core.lock` and then the supplied wheel using `pip install --no-deps dist/bayes_stesn-0.1.0-py3-none-any.whl`. Report generation additionally uses the report lock. MP4 rendering requires the external `ffmpeg` executable; GIF output and the numerical package do not require FFmpeg. PDF generation uses ReportLab and does not require LaTeX.

## Mathematical and API verification

Set a predictable thread budget before launching Python:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q --junitxml=results/validation/manual_pytest.xml
```

The suite checks Gaussian algebra against independent dense references, posterior moments, valid covariance constructions, AR and robust likelihood calculations, prior transformation Jacobians, solver failures, information boundaries, checkpoint continuation, serialization and the complete prediction API. Tests for small MCMC runs check correctness and execution; their intentionally short chains are not labelled converged scientific fits.

The compact standalone algebra benchmark is:

```bash
.venv/bin/python benchmarks/verify_v3_parity.py
```

When the original V3 directory is available, run the full fixed-parameter comparison:

```bash
.venv/bin/python benchmarks/verify_v3_parity.py --legacy-root /path/to/spatiotemporal_esn_v3
```

This loads the original numeric artifacts with `allow_pickle=False`, replays their frozen feature map, and compares likelihoods and posterior quantities at matched hyperparameters. Its result is not a new full-size MCMC run or evidence of better forecasts. The full reference dimensions are 256 sites, 7,600 fitting times, 172 spatial basis columns and 657 temporal features.

## Build and installed-package release gate

The full gate builds both distributions, creates a separate isolated environment, installs only the pinned dependency closure from the public package registry, installs the wheel, and runs `pip check`. It then runs the complete tests from a temporary directory against the installed wheel, checks module origins, executes the installed CLI, and verifies exact saved-model/prediction roundtrips.

```bash
.venv/bin/python tools/release_check.py --fresh-venv /tmp/bstesn-release-venv
```

The default path must not already exist. To revalidate the same isolated environment after a rebuild, use:

```bash
.venv/bin/python tools/release_check.py --fresh-venv /tmp/bstesn-release-venv --reuse-venv
```

Reuse still verifies that system packages are excluded, reinstalls the built wheel, and checks the complete dependency and test gates. The manifest records whether the environment was created during that run. The gate writes:

- `results/release_manifest.json`: test counts, skipped tests, warnings, build hashes, package origins, versions, CLI and roundtrip status, and source provenance.
- `results/validation/source_pytest.xml` and `installed_pytest.xml`: JUnit test receipts.
- `results/validation/*.log`: captured commands and their outputs.
- `results/validation/fresh_install_report.json`: pip's dependency-install receipt; a separate `public_registry_install_report.json` may record downloads made during an approved network-enabled bootstrap.
- `results/validation/fresh_environment.json`: the isolated environment and installed versions.
- `dist/*.whl` and `dist/*.tar.gz`: the verified distributions.

If code changes after a check, rebuild and rerun before describing that revision as verified. The manifest explicitly distinguishes local validation from publication and remote CI. A configured GitHub workflow is not an executed GitHub workflow, and a local repository is not a published public repository.

## Complete smoke and demonstration workflows

The installed CLI can exercise the whole workflow cheaply:

```bash
bstesn demo --smoke --output results/smoke
```

This produces the resolved YAML configuration, a checkpoint, a fitted model, posterior predictive paths, evaluation truth and a score/diagnostic summary. Its small warmup and retained sample count are intentional. `smoke_only: true` and failed scientific diagnostics remain visible in the output.

A larger configurable example is:

```bash
bstesn validate-config configs/demo.yaml
bstesn demo --config configs/demo.yaml --output results/demo
bstesn diagnose results/demo/model.npz --functional-draws 256
bstesn forecast results/demo/model.npz --horizon 40 --paths 256 --seed 407 --output results/demo/new_prediction.npz
```

The `diagnose` command exits with status 2 when its convergence gate fails. A demonstration should not be interpreted as a successful scientific fit merely because fitting returned without an exception.

For user data, supply a numeric NPZ containing `y[time,site]` and `coordinates[site,spatial_dimension]`; optional `times` and `site_ids` retain identities. The `fit` command uses the model and split in YAML:

```bash
bstesn fit observed_data.npz --config configs/demo.yaml --checkpoint results/custom/checkpoint.npz --output results/custom/model.npz
```

To continue an interrupted chain, rerun the same command with `--resume`. The data, model, preprocessing, seed, warmup, transition settings and thinning must match. Increasing the retained draw count is supported; changing the scientific model or sampler kernel is not a checkpoint extension. Saved artifacts use JSON metadata and numeric arrays, with checksums, rather than executable pickle objects.

## Sampler settings and reproducibility

`SamplerConfig` records the exact transition settings. The current initialization and RHS mixing controls are:

```python
SamplerConfig(
    initialize="auto",
    init_maxiter=100,
    init_max_evaluations=2000,
    init_max_dimension=20,
    init_dispersion=2.0,
    rhs_tradeoff_scale=0.5,
)
```

Automatic initialization uses a bounded local MAP search and finite-difference curvature for eligible small collapsed hyperparameter blocks. Curvature regularization and optimization status are recorded. Initialization changes the starting points and proposal scale, not the posterior support or target. Set `initialize="prior"` to use prior-center starts. Adaptation stops at the warmup boundary.

The additional RHS move shifts `log_tau` upward and all `log_local` values downward by the same random amount. It preserves the coefficient variances and accepts on the hyperprior ratio. Set `rhs_tradeoff_scale=0` to disable it. Both settings are part of checkpoint identity. Exact continuation includes RNG state and adaptation state; cross-platform bitwise agreement is not promised.

Named random streams control feature construction, chains and predictive paths. A predictive trajectory keeps one joint parameter draw throughout its horizon and feeds its own simulated observations back into the reservoir. Increasing the number of paths preserves the earlier path prefix. Advancing the state with subsequently observed history and refitting the parameter posterior are separate API operations.

## Controlled comparisons and final evaluation

Use the controlled study's dedicated configuration, rather than changing the demonstration ad hoc:

```bash
.venv/bin/python benchmarks/controlled_study.py pilot --config configs/controlled_study.yaml --output report/reproduction_study
.venv/bin/python benchmarks/controlled_study.py freeze --config configs/controlled_study.yaml --output report/reproduction_study --replications 6
.venv/bin/python benchmarks/controlled_study.py run --output report/reproduction_study
```

The pilot measures cost and checks diagnostics. Review it before freezing. The frozen protocol records the configuration, scientific source hashes, scenarios, candidate models, forecast blocks and replication count. The final runner rejects changes to that protocol and evaluates only predeclared final replications. Candidate selection uses validation forecasts and excludes failed chains; final outcomes do not choose models, draw counts or stopping rules.

Use a new output directory for a changed protocol. Preserve failed pilots as pilot evidence rather than overwriting them into an apparent successful run. When reproducing a delivered study, use its recorded configuration and protocol rather than assuming that the latest development configuration is identical. A flexible model that fails diagnostic gates remains a failed candidate, even when some of its predictive summaries look attractive.

## Verify and render the delivered scientific evidence

The portable unit suite does not silently skip a large-data study check. Run the
separate evidence validator when the full release arrays are present:

```bash
.venv/bin/python tools/verify_study_artifacts.py
.venv/bin/python report/render_figures.py --study report/controlled_study --output report
.venv/bin/python report/render_report.py --study report/controlled_study --assets report --output report
```

The validator fails explicitly if required study artifacts are absent. It checks
case completeness, protocol and source identities, validation choices, failed-fit
exclusion, canonical archive integrity, saved-mean/truth RMSE equality and paired
comparisons. Its receipt is `report/controlled_study/integrity_receipt.json`.
The two renderers use saved results only. External output directories are
supported; `--assets` points the PDF/HTML renderer to the directory containing
the already generated `figures/` and `media/` folders.

After the release gate and a source commit, package the complete download:

```bash
.venv/bin/python tools/package_release.py --output ../deliverables
```

The packager checks the successful release manifest against current source,
creates and verifies a Git bundle, hashes every included payload, and checks ZIP
integrity. It excludes environments, cache files and incomplete atomic staging
files. Public-repository status is documented in `docs/publication.md`.
