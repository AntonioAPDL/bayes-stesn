# Contributing

Use Python 3.11+ for core development; the fully pinned CPU-neural release gate
uses CPython 3.12. The quick start and scientific information policies are in
`docs/benchmark_guide.md`.

```bash
python -m pip install -e '.[dev,report,neural]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
python -m compileall -q src
```

A new forecaster accepts only an observed `Panel`, validates capabilities,
exposes `forecast(..., history=...)`, and saves numeric arrays/JSON through the
common state API. State conditioning preserves the fitted prefix and parameters.
Declare whether outputs are points, marginal intervals, a finite ensemble, or
coherent joint paths; do not infer an unimplemented probability distribution.
Register candidate settings and the selection metric before final evaluation.

Add reference or reduction tests that resolve real mathematical risks: a tiny
independent dense calculation, known moments, frozen-history information flow,
serialization identity, or a resource boundary. Tests that merely repeat an
implementation expression are insufficient evidence of a new inference method.
Use the existing suite after changes to shared numerical kernels.

For an efficiency change, identify the actual bottleneck, preserve the sampled
law and RNG contract when possible, and version any deliberately changed
arithmetic/random-stream convention. Separate execution/resource tests from
convergence checks and predictive comparisons. Never weaken a diagnostic gate
to obtain a favorable result.

Use one BLAS thread per process in comparisons. Keep generated checkpoints,
large data arrays and movies in release artifacts, not ordinary source commits.
Save resolved profiles, source/runtime identities and failed attempts. The
release tooling builds the wheel and reruns the tests from an isolated location
to detect accidental source imports.

```bash
python tools/release_check.py --with-neural --fresh-venv /tmp/bstesn-release
```

After validation and a source commit, package the selected comparison and its
render-only report:

```bash
python tools/package_release.py --benchmark-run results/benchmark_v3/pilot_release \
  --benchmark-report report/benchmark_pilot
```

The package command rejects source drift or mismatched evidence.
Repository/CI publication status must be reported from observed remote
results; a configured workflow alone is not evidence that it ran.
