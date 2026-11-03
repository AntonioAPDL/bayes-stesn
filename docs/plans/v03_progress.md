# v0.3 implementation progress

Starting source: `37d2d918aa2d03cb4f76477231df33c06ca448bd`.

## Verified implementation milestone

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
