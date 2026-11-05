# Reproducing and resuming the KS application

This is the repository's **custom, literature-grounded Kuramoto–Sivashinsky experiment**. It uses the supplied numerical data unchanged and preserves the Bayesian ST-ESN likelihood/prior and native comparator backends. Read the [protocol](ks_protocol.md), [model registry](ks_model_registry.md) and [native implementation audit](ks_native_audit.md) before interpreting results. The initial specification is [`configs/applications/ks_experiment.json`](../../configs/applications/ks_experiment.json); a run's `freeze.json` and ledger establish its actual selected settings and completion status.

## 1. Preserve the checkout and choose separate paths

Work from the existing authorized repository. Inspect its actual revision, branch and changes before creating a new worktree or editing anything. Do not reset to a revision recalled from an older conversation. The implementation was based on revision `4bae378922b824e0e934a3aaf08a9d8a1a65046e` on the isolated branch `application/ks-20261105`; the final handoff records the committed application revision.

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git worktree list
```

Choose a writable project cache outside the model checkout and outside an actively synchronized Drive mount. The example paths below are placeholders: change them to authorized local storage with adequate measured headroom. Do not use another task's directories or repurpose shell variables such as `HOME`.

```bash
export KS_REPO=/absolute/path/to/bayes-stesn
export KS_WORKSPACE=/data/ks_application_run
export DOWNLOAD_DIR="$KS_WORKSPACE/incoming"
export CACHE_ROOT="$KS_WORKSPACE/cache"
export RESULTS="$KS_WORKSPACE/results"
export CONFIG="$KS_REPO/configs/applications/ks_experiment.json"
mkdir -p "$DOWNLOAD_DIR" "$RESULTS" "$KS_WORKSPACE/private"
cd "$KS_REPO"
```

Keep original data, the private catalog, the environment, checkpoints and large outputs outside Git. Preserve unrelated simulations and frozen benchmark results. Only redundant task-created files with verified replacements are cleanup candidates.

## 2. Reuse or create a compatible project environment

Use a compatible existing project environment where available. The tested application stack is recorded in [`requirements-ks.lock`](../../requirements-ks.lock): CPython 3.12, NumPy/SciPy, h5py, Matplotlib and CPU PyTorch, with no second deep-learning ecosystem. These are the observed compatible application versions, not an instruction to upgrade an unrelated environment to the preparation environment's pins.

For a fresh isolated environment, with no `sudo`:

```bash
python3.12 -m venv "$KS_WORKSPACE/env"
source "$KS_WORKSPACE/env/bin/activate"
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  -r "$KS_REPO/requirements-ks.lock"
python -m pip install --no-deps -e "$KS_REPO"
```

The CPU wheel index supplies the explicitly pinned `+cpu` Torch build. Reuse an already compatible installed CPU build when it is available; no GPU is required for the small CNN. The runner records numerical versions in its identity and refuses incompatible checkpoint reuse. `ffmpeg` enables optional MP4 rendering; static reports remain available if it is absent.

Set thread limits before imports:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0
export MPLBACKEND=Agg
python -c 'import numpy, scipy, h5py, torch, matplotlib; print(numpy.__version__, scipy.__version__, h5py.__version__, torch.__version__, matplotlib.__version__)'
```

Before scheduling, inspect the current CPU quota/affinity, cgroup memory limit/current use, disk and inodes, applicable quota, and existing jobs. The admission gate rechecks cgroup-aware headroom and available disk during execution; it does not treat host RAM as this job's budget. On a different machine, profile before choosing resource settings. Any configuration/source change requires a **new results directory**; do not edit an initialized run's identity to make old checkpoints appear compatible.

The measured initial allocation is four active wall hours with at most two single-thread workers, eight cumulative worker hours, and at most 2.5 worker hours for screening. A `--wall-hours` argument bounds one invocation; it does not reset the cumulative worker budget across resumes. The current configuration reserves 12,463,156,428 bytes of disk and applies combined-worker admission with a 1.5 multiplier on estimated peak RSS. The [protocol](ks_protocol.md) records the pilot timings and 1.5–3.5-hour initial estimate. These limits are not a promise that every configuration will converge within that time.

## 3. Retrieve only the four authorized delivery objects

Use the existing authorized Drive connection and the private delivery instructions supplied separately. Verify that the presently connected account can read the completed collection, then download the **complete binary objects** below into `DOWNLOAD_DIR`, preserving these basenames. Public repository files intentionally contain no private account details, Drive object identifiers, or collection URLs. The local verifier does not authenticate, download, or prove present account access.

| Required object | Exact bytes | SHA-256 |
|---|---:|---|
| `ks_prepared_v1.tar.gz` | 182,334,299 | `898f18deccc1fe6b3fb4d6eb0aa847244dc115efc56a260aceaae07bc22c227c` |
| `source_and_metadata_v2.tar.gz` | 9,981,003 | `c88f8475b97a9b7159e643e06e424d0f58630400182957602e48b91097422965` |
| `drive_catalog.json` | 633,855 | `510a14dfa23e66fc861066dc4707fc46a192ba13588af2ba30f1e69478ce6be0` |
| `restore.py` | 23,108 | `9711bbb6a78c59e106134f3ebd64c151e45593b9d8b46304d2d81759c833b282` |

The total is 192,972,265 bytes. The unchanged KS package is **v1**; the active source package is **v2**. Do not substitute source v1 or download the other datasets. Small handoff documentation may be retrieved separately. Never transfer archives/base64 or a truncated catalog through conversational context.

Verify all four files and parse the complete catalog programmatically:

```bash
python tools/verify_ks_delivery.py \
  --incoming "$DOWNLOAD_DIR" \
  --receipt "$KS_WORKSPACE/private/ks_delivery_verified.json"
```

The tool checks every byte count and SHA-256, parses all 362 catalog records, validates their structure and confirms that the default KS restore selection contains exactly the KS archive and active source v2 archive. It never prints the catalog or private metadata. A missing or mismatched file remains an explicit error; no replacement dataset is generated.

## 4. Dry-run and restore to the separate cache

The verifier can invoke the exact, hash-verified supplied helper, first for a dry-run and then for actual restoration:

```bash
python tools/verify_ks_delivery.py \
  --incoming "$DOWNLOAD_DIR" \
  --restore "$CACHE_ROOT" \
  --disk-reserve-bytes 12463156428 \
  --receipt "$KS_WORKSPACE/private/ks_delivery_restored.json"
```

The reserve flag may be increased to match a stricter project policy. The wrapper always requires at least the larger of 2 GiB and 20% of initially available space. It refuses a cache overlapping the model repository or incoming directory. It invokes these supplied commands without shell interpolation:

```bash
python "$DOWNLOAD_DIR/restore.py" \
  --catalog "$DOWNLOAD_DIR/drive_catalog.json" \
  --incoming "$DOWNLOAD_DIR" --destination "$CACHE_ROOT" --dataset ks --dry-run

python "$DOWNLOAD_DIR/restore.py" \
  --catalog "$DOWNLOAD_DIR/drive_catalog.json" \
  --incoming "$DOWNLOAD_DIR" --destination "$CACHE_ROOT" --dataset ks
```

Use the wrapper to enforce its disk reserve when running both steps. The helper itself verifies archive safety and destination conflicts before extraction, reuses byte-identical files and preserves conflicting files. It has no force-overwrite option. The original verified dry-run selected two archives totaling 192,315,302 bytes, with 458 TAR members and 469,202,764 additional bytes including the retained archive copies. Existing verified files reduce that additional allocation on a repeat restore.

The reader's common root is `CACHE_ROOT/data`. Preserve the relative `data/ks/canonical` and `data/ks/raw` layout: canonical HDF5 files link to raw trajectories. Do not copy canonical files alone. Read `START_HERE.txt`, `PRO_HANDOFF.txt`, the delivered `README.md`, `dynamics/README.md`, and the KS configuration, split, transform and QC records before changing adapter behavior. The application's verified cache audit checks those identities and external links; no generator rerun is required.

## 5. Targeted data and model-boundary checks

Run the small application acceptance suite in the compatible environment:

```bash
export BAYES_STESN_KS_CACHE="$CACHE_ROOT"
python -m pytest -q \
  tests/test_ks_data.py \
  tests/test_ks_model_boundary.py \
  tests/test_ks_baselines_metrics.py \
  tests/test_ks_native.py \
  tests/test_ks_cnn.py
```

The data tests use the distinct pilot trajectories for numerical windows; core test values are not used for fitting or model selection. They check masks, time conventions, inverse transforms, the periodic seam and outside-boundary perturbation invariance. Native tests include interrupted-chain recovery, static-path parameter coherence and returned-path means. CNN tests check the explicit all-observed requirement and exact interrupted optimization state. Tiny synthetic fits in tests are implementation checks, not reported KS results.

The saved native-pilot boundary test is optional when its matching artifact is not present. To reproduce that additional fixed-fit check using the retained application artifact, set `BAYES_STESN_KS_NATIVE_PILOT` to its `native_bayes_reference.npz` path before the command. It does not refit or reselect that model. A separate core selected-configuration check remains necessary before interpreting final scientific results.

For a complete local cache audit, without fitting models:

```bash
KS_AUDIT_CACHE="$CACHE_ROOT" python - <<'PY'
import json
import os
from bayes_stesn.applications.ks_data import audit_cache

receipt = audit_cache(os.environ["KS_AUDIT_CACHE"])
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
```

The original audit verified all 23 raw trajectory hashes and canonical external links, 16 derived files, 69 training-only transforms, and 23 distinct initial states. The pre-screen history decision used all 72 candidate/seed/pilot/main-track combinations and selected 128 from `[128,256,512,1024]` under the declared numerical tolerance. No posterior fit or future label was required for that synchronization sweep. [The public preflight receipt](evidence/ks_preflight.json) preserves counts and hashes without private retrieval details.

## 6. Screen and freeze both main tracks

```bash
python -m bayes_stesn.applications.ks_runner screen \
  --config "$CONFIG" --cache "$CACHE_ROOT" --output "$RESULTS"
```

The initial stage uses core trajectories `[0,5]`, one common model seed and validation starts `[6000,6500,7000,7500]`. Promotion uses `[0,5,10,15,4,9]` and eight fixed validation starts. The runner reuses matching fits and origin forecasts. Native probabilistic families select by band-balanced empirical CRPS; point-only families select by band-balanced squared error. The four lead bands are 1–10, 11–25, 26–50 and 51–100. Eligible candidates within 1% of the best score follow the predeclared lower-complexity tie rule.

The configured boundary rule permits at most one additional candidate per eligible family/track: extend a selected VAR/ESN/NVAR ridge boundary one decade, or a selected 60-epoch CNN budget to 90, keeping other settings and the seed fixed. All such refinement uses validation information and remains inside the screening budget.

Selection for A and B and validation predictive-precision checks must finish before `freeze.json` is written and either test track is opened. Check `selection/`, the precision receipt and `freeze.json` for actual candidate qualifications, near-ties, selected path count and selected settings. No fully qualified candidate is an explicit family issue, not grounds for inventing a successful posterior. A pilot or incomplete stage is not a completed study.

## 7. Fit selected models and evaluate the main tracks

```bash
python -m bayes_stesn.applications.ks_runner final \
  --config "$CONFIG" --cache "$CACHE_ROOT" --output "$RESULTS" --wall-hours 4
```

Each selected family/track is fitted separately on each of the 20 core trajectories using only its permitted training rows/sites. Every trajectory has the same 20 test starts `8000,8100,...,9900`, with 100 autonomous steps per origin and the frozen causal history. Matching training fits from screening are reused. The test gate requires both main selections to be frozen; the runner does not add validation times/sites to training.

For native probability models, one origin's complete paths are generated in bounded batches and scored before proceeding. Large tensors covering all origins or trajectories are never accumulated. Physical-unit means, marginal quantiles and a small path subset are retained only for predeclared example cases. Point-only competitors have no invented intervals. Summary tables keep failed/pending origin counts and inference qualification visible.

## 8. Prespecified sensitivity and contiguous robustness

After main-track completion:

```bash
python -m bayes_stesn.applications.ks_runner sensitivity \
  --config "$CONFIG" --cache "$CACHE_ROOT" --output "$RESULTS" --wall-hours 4

python -m bayes_stesn.applications.ks_runner robustness \
  --config "$CONFIG" --cache "$CACHE_ROOT" --output "$RESULTS" --wall-hours 4
```

Sensitivity uses seeds `20261106` and `20261107` on trajectories `[0,5,10,15]` for selected stochastic families. Seeds remain distinct from posterior paths and scientific replicates. Contiguous robustness uses the predeclared eight strict-capable families and the interleaved-validation-selected configuration, with new contiguous transforms/features/parameters. It does not choose methods from the main-test ranking. Optional pooled or measurement-noise studies are separate experiments and are not silently included here.

## 9. Status, interruption and exact reuse

```bash
python -m bayes_stesn.applications.ks_runner status \
  --config "$CONFIG" --cache "$CACHE_ROOT" --output "$RESULTS"

python -m bayes_stesn.applications.ks_runner resume \
  --config "$CONFIG" --cache "$CACHE_ROOT" --output "$RESULTS" --wall-hours 4
```

`resume` continues unfinished selection if needed, then selected fitting/evaluation, sensitivity and robustness. It retains completed fit/forecast identities and skips terminal jobs. A scheduler lock and distinct fit locks prevent concurrent writers; do not launch a duplicate scheduler for the same results root. `status` is read-only with respect to scheduled jobs. Inspect `logs/`, `ledger/` and `attempts/` when an interrupted or resource-blocked job remains pending.

Atomic checkpoints bind data, preprocessing, configuration, seed, numerical source and runtime. Bayesian sampler and CNN checkpoints retain their actual RNG/transition or optimizer/batch state. GP and GRU completed fits are reusable; an interrupted unfinished optimizer/training fit restarts deterministically from its declared initialization rather than claiming exact optimizer-state continuation. Completed forecasts retain their named per-path randomness.

All attempt worker times count toward cumulative limits, including interrupted work. Resuming does not erase the screening or total compute budget. A genuine capacity or time blocker requires an explicitly versioned budget/run decision; do not delete receipts, change hashes, silently weaken diagnostics or mark pending jobs complete. Failed scientific runs are preserved and reported, not retried under new seeds until they pass.

## 10. Summarize, render and inspect the deliverables

```bash
python -m bayes_stesn.applications.ks_runner summarize \
  --config "$CONFIG" --cache "$CACHE_ROOT" --output "$RESULTS"

python -m bayes_stesn.applications.ks_report \
  --results "$RESULTS" --output "$RESULTS/report"
```

Summarization does not fit models or select new settings. Its complete-trajectory score population and all failure/pending denominators are explicit. Point/probability tables, episode results, paired trajectory differences and lead curves are saved in `summary.json`, `episode_results.json` and `horizon_curves.json`. Probability metrics use qualified complete trajectories only. Trajectory-cluster intervals are conditional on the selected configurations; origins, sites and posterior draws are not independent replicates.

The report command renders saved outputs without reading the dataset or running a model. It produces publication PNG/PDF figures, `selected_horizons.json`, offline `report.html`, `report.md`, optional streamed MP4s and `report_manifest.json`. Use `--no-animations` when static exports suffice. Examples are fixed at trajectory 0, first target 8000. Strict heatmaps show the 12 official target sites with a separate observed-site rug; unavailable cases are not replaced by favorable ones.

Inspect representative PNG/PDF layouts and MP4 frames before declaring visual QA complete. The manifest initially labels that inspection as pending. Actual scientific conclusions, final resource measurements, selected configurations and completed counts must come from the saved results; the renderer never turns partial execution into a completion claim.

## Output map

| Location under `RESULTS` | Purpose |
|---|---|
| `experiment.json` | Initial source/runtime/configuration identity |
| `jobs/`, `ledger/`, `attempts/`, `logs/` | Jobs, state, measured worker consumption and failure evidence |
| `fits/` | Identity-bound fitted objects and diagnostic receipts |
| `forecasts/` | Compact scores and per-origin status receipts |
| `selection/`, `validation_precision.json`, `freeze.json` | Validation choices, predictive precision and sealed specification |
| `test_access.json` | Evidence that test scoring began only after both main selections |
| `predictions/` | Compact predeclared physical-unit examples |
| `summary.json`, `episode_results.json`, `horizon_curves.json` | Aggregate, episode-level and lead-specific results with eligibility counts |
| `report/` | Figures, compact animations, offline report and checksums |

Retain code, configuration, diagnostics, compact metrics, selected predictions, receipts and the final report. Public Git artifacts must exclude bulk data, environments, private manifests and credentials. Any larger durable artifact copy should use already authorized private storage and include checksums.
