# Reproducing the implementation and experiments

The release checks separate four questions: whether the mathematical and API tests pass, whether the distributable package installs correctly, whether a small complete fit and forecast executes, and whether a scientific experiment meets its convergence and prediction gates. A successful smoke run answers the execution question only.

## Verified environment and dependency groups

The version locks target **CPython 3.12 on Linux x86_64**. The package declares its broader supported Python range in `pyproject.toml`; identical floating-point results across operating systems, BLAS builds, or Python versions are not promised.

| File | Purpose |
|---|---|
| `requirements-core.lock` | NumPy, SciPy, PyYAML and threadpoolctl, the complete runtime dependency set |
| `requirements-report.lock` | Core dependencies plus plotting, image, PDF and required transitive packages |
| `requirements-dev.lock` | Core dependencies plus pytest, package-building tools and thread control for benchmarks |
| `requirements-neural-cpu.lock` | Optional CPU PyTorch and its pinned dependencies for Linux CPython 3.12 |

These files pin versions rather than every platform-specific wheel hash. The release check uses `https://pypi.org/simple` for core/report/build dependencies and the non-Torch support packages in the neural lock. Only the pinned CPU Torch wheel comes from `https://download.pytorch.org/whl/cpu`, installed with `--no-deps` after its support packages. The CPU index does not provide every support-package version. The checker saves separate installation receipts, checks all installed versions against the combined locks, records the isolated environment, and hashes the produced wheel and source archive. It does not copy unrelated packages from the development machine into a lockfile.

Create an isolated environment without `--system-site-packages`:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --index-url https://pypi.org/simple -r requirements-dev.lock -r requirements-report.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
source .venv/bin/activate
```

For a runtime-only installation, install `requirements-core.lock` and then the supplied wheel using `pip install --no-deps dist/bayes_stesn-0.2.0-py3-none-any.whl`. To exercise `dcrnn`, `basis_gru`, and their optional tests, also install:

```bash
python - <<'PY'
from pathlib import Path
import subprocess, sys
sys.path.insert(0, "tools")
from release_check import locked_versions
pins = locked_versions(Path("requirements-neural-cpu.lock"))
torch_version = pins.pop("torch")
subprocess.run([sys.executable, "-m", "pip", "install", "--index-url",
                "https://pypi.org/simple",
                *(f"{name}=={version}" for name, version in sorted(pins.items()))], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--index-url",
                "https://download.pytorch.org/whl/cpu", f"torch=={torch_version}"], check=True)
PY
```

The statistical core does not require PyTorch. Report generation additionally uses the report lock. MP4 rendering requires the external `ffmpeg` executable; GIF output and the numerical package do not require FFmpeg. PDF generation uses ReportLab and does not require LaTeX.

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

The full version 0.2 gate builds both distributions, creates a separate isolated CPython 3.12 environment, installs the pinned dependency closure and CPU Torch, installs the wheel, and runs `pip check`. It runs source and installed-wheel tests in that environment, checks package/module origins, rejects external or unpinned distributions, executes the installed CLI and benchmark smoke, and verifies exact saved-model/prediction roundtrips. Every selected test must run without skips or pytest warnings.

```bash
.venv/bin/python tools/release_check.py --with-neural --fresh-venv /tmp/bstesn-v02-release
```

The default path must not already exist. To revalidate the same isolated environment after a rebuild, use:

```bash
.venv/bin/python tools/release_check.py --with-neural \
  --fresh-venv /tmp/bstesn-v02-release --reuse-venv
```

Reuse verifies that system/user packages are excluded, reinstalls the built wheel and CPU Torch, and checks the dependency and test gates. Neural support pins are installed from a temporary exact requirements file through PyPI; the CPU-index step force-reinstalls only Torch with `--no-deps`. Network access can still be needed; this is not an offline flag. Already installed PyPI packages may remain in place, so their installation receipts can contain no new downloads. The manifest records reuse explicitly and records both the support-package receipt and the official-host CPU wheel receipt.

Omit `--with-neural` to validate a separate Torch-free core environment. That mode explicitly deselects declared optional-PyTorch test classes, records their IDs, and still rejects skips among selected tests. It rejects a reused environment that already contains Torch. The CI core matrix and the separate locked CPU-neural job exercise these distinct claims.

The gate writes:

- `results/release_manifest.json`: test counts, skipped tests, warnings, build hashes, package origins, versions, CLI and roundtrip status, and source provenance.
- `results/validation/source_pytest.xml` and `installed_pytest.xml`: JUnit test receipts.
- `results/validation/*.log`: captured commands and their outputs.
- `results/validation/fresh_install_report.json`: pip's current core/report/build receipt; `neural_support_install_report.json` records the PyPI neural support step and `neural_install_report.json` records the CPU Torch installation when enabled.
- `results/validation/fresh_environment.json`: the isolated environment and installed versions.
- `dist/*.whl` and `dist/*.tar.gz`: the verified distributions.

The manifest's `validation_artifacts` inventory seals the current receipts and smoke artifacts with sizes and SHA256 values. It does not present older files in `results/validation` as current release evidence. Each CLI and benchmark smoke uses a new output directory, recorded as `cli_smoke.artifact_directory` and `benchmark_smoke.artifact_directory`, so a rebuild cannot resume a previous build's sealed benchmark or include old smoke artifacts. These smoke fits are execution checks, not convergence evidence.

Finish source, documentation, configuration, dependency-lock and workflow edits before running the final gate. Their hashes, including `CONTRIBUTING.md`, `MANIFEST.in` and `.gitignore`, are checked for changes during validation and again during packaging. Point documentation to the manifest for final test counts instead of editing hashed documentation after a successful check. The manifest distinguishes local validation from publication and remote CI. A configured GitHub workflow is not an executed workflow, and a local repository is not a published public repository.

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

## Version 0.2 benchmark evidence and packaging

The comparison API, chronology, capability gates, development/confirmation distinction, and commands are documented in [benchmark_guide.md](benchmark_guide.md). A changed package source fingerprint requires a new run directory; completed job receipts must not be migrated from an earlier source revision.

The release's primary development run and separately declared spatial illustration use `results/benchmark_v3/pilot_release` and `results/benchmark_v3/detail_release`. Render from their saved results:

```bash
python tools/render_benchmark_layout.py results/benchmark_v3/pilot_release --output report/benchmark_pilot
python tools/embed_report_fonts.py report/benchmark_pilot
python tools/render_benchmark_layout.py results/benchmark_v3/detail_release --output report/benchmark_detail
python tools/embed_report_fonts.py report/benchmark_detail
```

The layout wrapper removes redundant y-axis tick labels from the second and
third shared-geometry forecast panels and supplies half-row categorical padding.
It delegates numerical content to the unchanged package renderer and never fits
or forecasts a model. The wrapper and font helper are presentation tools outside
the scientific package fingerprint; their source is still covered by the release
validation hashes and committed history. Report manifests bind their exported
files to the original run identity. The benchmark data were not rerun or relabeled
when these presentation details were corrected.

After the release gate succeeds and the validated source is committed, package the current benchmark evidence without requiring historical version 0.1 study arrays:

```bash
python tools/package_release.py --benchmark-run results/benchmark_v3/pilot_release \
  --benchmark-report report/benchmark_pilot \
  --additional-benchmark results/benchmark_v3/detail_release report/benchmark_detail \
  --supplemental-evidence results/benchmark_v3/calibration_release_A \
  --supplemental-evidence results/benchmark_v3/calibration_release_B \
  --supplemental-evidence results/benchmark_v3/confirmation_release \
  --supplemental-evidence results/benchmark_v3/publication \
  --output ../deliverables_v02
```

Run this only after every named result and report is complete. Each additional benchmark/report pair undergoes the same source, identity, numeric-summary, job-receipt and report-hash checks as the primary run. Supplemental evidence must be an explicitly named subdirectory of `results/` or `report/`; symlinks and interim files are rejected. The combined default supplemental caps are 2 GiB and 10,000 files; `--supplemental-max-bytes` changes the byte cap explicitly. Its file inventory preserves the saved evidence without asserting convergence or treating serial calibration blocks as independent replications.

The package includes SVG, numeric artifacts, current validation receipts, a verified Git bundle, and a SHA256 inventory. Interim render/integration folders and unrelated historical validation files are excluded. This creates a local deliverable; pushed/public status is established separately as described in [publication.md](publication.md).

## Earlier controlled comparisons (version 0.1)

Use the controlled study's dedicated configuration, rather than changing the demonstration ad hoc:

```bash
.venv/bin/python benchmarks/controlled_study.py pilot --config configs/controlled_study.yaml --output report/reproduction_study
.venv/bin/python benchmarks/controlled_study.py freeze --config configs/controlled_study.yaml --output report/reproduction_study --replications 6
.venv/bin/python benchmarks/controlled_study.py run --output report/reproduction_study
```

The pilot measures cost and checks diagnostics. Review it before freezing. The frozen protocol records the configuration, scientific source hashes, scenarios, candidate models, forecast blocks and replication count. The final runner rejects changes to that protocol and evaluates only predeclared final replications. Candidate selection uses validation forecasts and excludes failed chains; final outcomes do not choose models, draw counts or stopping rules.

Use a new output directory for a changed protocol. Preserve failed pilots as pilot evidence rather than overwriting them into an apparent successful run. When reproducing a delivered study, use its recorded configuration and protocol rather than assuming that the latest development configuration is identical. A flexible model that fails diagnostic gates remains a failed candidate, even when some of its predictive summaries look attractive.

## Verify and render earlier version 0.1 evidence

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

When the historical controlled-study artifacts are available, the original package mode remains supported:

```bash
.venv/bin/python tools/package_release.py --output ../deliverables
```

The packager checks the successful release manifest against current source,
creates and verifies a Git bundle, hashes every included payload, and checks ZIP
integrity. It excludes environments, cache files and incomplete atomic staging
files. Public-repository status is documented in `docs/publication.md`.
