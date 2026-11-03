# v0.3 implementation progress

Starting source: `37d2d918aa2d03cb4f76477231df33c06ca448bd`.

## Current source freeze and execution

Published numerical implementation commit: `001024f5dfdcfa5288b0742cf5cbb917e6b875ae`.
Subsequent packaging correction: `7633dd9b28051f13ff1c3f1905fe893a790287d1`.
Scientific package fingerprint:
`9f73b28a6ee6b8eec4bd21c3366b0c260c9433f798610be16697b2e44166d5ac`.
The milestones below this section are preserved historical snapshots, not claims
about completion of the current-source campaigns.

- Current locked source: 388 tests and 76 subtests passed. Current installed
  Torch-free wheel with newer numerical dependencies: 374 passed, two optional
  Torch skips, 12 neural deselections, and 76 subtests passed. See the
  `cache_admission_*_tests.xml` receipts in `docs/evidence/v03/`.
- Resource admission now keeps raw cgroup charge visible while granting a
  conservative credit for eligible clean file cache. This is an explicit
  scheduling heuristic, not guaranteed reclaim. The hard ceiling, aggregate
  job cap, reserve and active-job reservations remain enforced.
- Remote CI for `001024f` passed all three core jobs and locked source tests;
  extracted-sdist tests exposed omitted Python study helpers under `configs/`.
  The packaging-only correction passed all four jobs in run `41600288423`
  on exact commit `7633dd9`: locked source and installed wheel each passed
  388 tests plus 76 subtests, with no failures, skips or warnings. The downloaded
  25-file validation archive matches GitHub's SHA-256. See
  `docs/evidence/v03/packaging_corrected_remote_ci.json`. The final release's
  later report/tool commit still requires its own exact-head gate.
- Current-source scientific replay destinations are
  `results/v03/development/reference_final`, `results/v03/coordinate-final`,
  and `results/v03/extensions/final_*`. These are complete: reference60/60
  qualified; coordinate legacy2/3 versus transformed3/3; all four P5
  evaluations terminal. They reuse the same declared fields; they are
  provenance replays, not new independent replications.
- Strict retuning now runs at `results/v03/development/tuned_final`: all 60
  final outcomes remain required. G4 B/A selections run first, enabling later
  studies as soon as their needed validation receipts are terminal. An absent
  or incomplete selection never authorizes a reference substitution.
- The first 14 tuned outcomes are qualified and terminal, including all eleven
  model/track selections required for the next studies. Six isolated timing fits
  on the largest planned sensor geometry have completed; measured fit durations
  range from less than 0.001 seconds for the spatial control to 23.09 seconds for
  ridge ESN. These timings set prospective caps, not universal complexity bounds.
- Six fresh, training-only largest-geometry timing fits precede the 330 history,
  33 sensor and 220 revised-development fits. The prospective history gate
  retains 1000 by default and promotes 2000 only when all accuracy, qualification,
  paired-population and cost conditions pass.
- History1000 has completed all165 terminal outcomes with stable source.
  History2000 is running on the same assigned worker. No history choice has
  yet been made and no revised or confirmation datasets have been generated.
- The conditional-prior moment audit and exact saved G3 first-step reconstruction
  are complete. They are bounded fixed-design diagnostics, not autonomous
  prior-rollout certification or additional independent forecast experiments.
- Fresh P5 findings can now be regenerated from completed outputs with
  `tools/summarize_extensions_v03.py`; a separate reaggregation exactly matches
  all sealed scientific fields and leaves the original findings immutable.
- Separate calibration uses 25 H100 blocks and one later common evaluation
  block, with no parameter refits. G5 whole-field and local-noise arms and G6
  blind velocity shift remain distinct studies and must close explicitly.
- No confirmation fields have been generated. Revised-development precision
  and qualification must pass before freezing a confirmation design.
- Science uses at most two model workers across all campaigns and one numerical
  thread per worker. Recheck both memory and disk before each wave because the
  host is shared. Do not attribute raw aggregate cache usage to this project.

The long G3 numerical receipt covers the complete planned history: time-halving
RMS 0.001980 passes the 0.02 tolerance, while mesh-refinement RMS 0.272371 fails.
The target remains the finite semidiscrete grid; no continuum claim is made.
Historical P5 did not promote residual AR. Its prospective path-count rule
selected 512, which remains a finite reference ceiling rather than a guarantee
of negligible predictive Monte Carlo error.

Current resumable command, after activating the locked environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
python configs/benchmark/v03/studies/run_chunks.py \
  --config configs/benchmark/v03/tuned_final.yaml \
  --output results/v03/development/tuned_final \
  --max-invocations 60 --jobs-per-invocation 1 --max-memory-mb 5120 \
  --reserve-gib 4 --next-output-mib 512
bstesn benchmark status results/v03/development/tuned_final
```

Use `docs/reproduction_v03.md` for the ordered from-scratch workflow and
`configs/benchmark/v03/studies/README.md` for dependency-bound study commands.
No numerical source changes are permitted during the current runs.
Report and packaging inputs are hashed separately at the final release gate.

## Historical verified implementation milestone

Implementation and corrective commit: `e68cb8dbff0b5d1ba92d0e62d6c68f742dcb0c8f`.
Scientific package fingerprint: `e5180c486ecaa849bf5efeb0a67fe8ec30e963407f3eabe8c78009f88c40dc68`.

- P0 baseline: 211 tests and 74 subtests passed in the isolated locked CPU environment.
- P1–P7 interfaces and numerical changes implemented, with explicit supported scopes.
- Corrected a demonstrated false runtime mismatch from BLAS enumeration order; all actual version, architecture and thread-count fields remain protected.
- Corrected diagnostic/CPU/serialization cost accounting and retained unknown interrupted-attempt costs.
- Current local source gate: 353 tests and 76 subtests passed with optional CPU neural models.
- Newer numerical libraries in installed Torch-free core: 339 passed, two optional skips, 12 optional deselections, 76 subtests passed.
- Remote Actions run `41592946471`: Python3.11/3.12/3.13 core jobs and locked CPU-neural source/wheel gate all passed. See `docs/evidence/v03/implementation_remote_ci.json`.
- Ordinary Git updates used expected-parent, non-force writes and exact blob readback. Existing release tags were preserved.

## Executed scientific evidence

- Latest coordinate comparison: `results/v03/coordinate-verified`, identical G1/G3/G4 targets, legacy qualified2/3 and transformed qualified3/3; one field per mechanism. Public default remains legacy.
- Historical reference campaign: 60/60 qualified, no fallback, archived under its own source identity. Rerunning the same declared reference under final source at `results/v03/development/reference_release`.
- Interrupted early tuning wave is preserved at `results/v03/development/tuned`; it is incomplete and will not be silently resumed under changed source. Explicit interruption receipt records why cost accounting was corrected.
- G3 grid64,dt0.05,H100 validation passed the time-discretization and morphology gates for the finite semidiscrete target; the mesh-to-continuum gate failed. No continuum claim.
- Owned-forecast transfer saved approximately99MiB incremental peak allocation on the declared 256×100×505 workload with identical arrays. This is not a whole-pipeline memory claim.
- Preliminary training-residual inspection supports a bounded complete-residual AR ablation; grouped-shrinkage and sparsity triggers did not hold.

## Active and next gates

1. Complete final-source reference campaign and strict retuned development at `results/v03/development/tuned_release` (60 final outcomes; declared eight effective candidates where available, finalist/fold/seed policy unchanged).
2. Freeze later fixed specifications using the predeclared G4 validation-only rule in `docs/evidence/v03/development_decisions.json`.
3. Execute paired history curves, a bounded nested-sensor study, compatible AR/path-count checks, and separate calibration.
4. Run five revised development datasets per core mechanism at the intended confirmation design. Do not generate confirmation data unless every declared precision/qualification gate passes.
5. Seal saved-data report/media, exact-source/wheel/remote release validation, archives/checksums and public release.

### Recoverable commands

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bstesn benchmark run \
  --config configs/benchmark/v03/tuned_development.yaml \
  --output results/v03/development/tuned_release --workers 2 --max-wall-seconds 5400
```

`benchmark status` distinguishes verified terminal outcomes, pending work and unresolved infrastructure. The frozen source must not change while scientific runs import it. Report/document/tool edits outside the package do not change this scientific fingerprint, but final release validation hashes those inputs separately.
