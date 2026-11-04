# v0.3 implementation progress

Starting source: `37d2d918aa2d03cb4f76477231df33c06ca448bd`.

## Recovery execution from the preserved source

The recovery checkout starts at
`51cb672b07d83d14ced663bf4bc2673128ef00b7`. Its numerical package and dependency
locks are unchanged. The preceding session's untracked numerical workspace was
not available, so the preserved v0.3 designs are being executed again with the
same seeds, budgets and qualification rules. These re-executions do not add
independent datasets to the historical experiments.

Completed phase archives retain the numerical arrays, checkpoints, manifests,
receipts, logs, checksums and restoration instructions. Partial tuning archives
are explicitly labeled partial and are superseded for restoration when a
complete snapshot of that same campaign becomes available. Historical
documentation and test receipts remain historical; the new release requires
its own completed evidence and final source/installed-wheel gate.

Use `docs/reproduction_v03.md` for the ordered procedure. The completed revised
pilot now meets the unchanged authority's five-pair requirement in every
primary cell, and its valid freeze admits 60 datasets per scenario while
retaining the MCSE shortfall at the cap. Successful assessment, admission,
completed confirmation and achieved precision remain separate outcomes.

## Current execution update — 2026-11-04

The numerical fingerprint remains
`9f73b28a6ee6b8eec4bd21c3366b0c260c9433f798610be16697b2e44166d5ac`.
The [current results record](../results_v03.md) gives the quantitative findings,
exact populations and limits. This section is the current factual update;
everything beginning at the next historical heading is retained verbatim.

- Both history conditions are complete: 165 terminal outcomes each, with
  163 qualified at 1000 and 164 at 2000. The
  [sealed history decision](../evidence/v03/history_choice.json) selects 1000.
  The ESN Track A macro gain passes the practical lower-bound threshold;
  Track B fails it and has heterogeneous scenario effects. Missing
  short-history accounting independently fails the cost gate. The recorded
  time ratio is partial, and no complete measured speedup is claimed.
- All 33 outcomes in the nested 25/50/100-sensor study qualify. Its
  [representation audit](../evidence/v03/sensor_representation.json) uses one
  G4 field and fixed initial history 2000, separately from the chosen revised
  history 1000. Spatial rank and noise-amplification summaries are descriptive
  diagnostics, not replicated forecast-benefit evidence.
- The recovered reference panel is 60/60 qualified. The six coordinate cases
  give legacy 2/3 and transformed 3/3 qualification, with one field per
  mechanism and no general default-promotion claim. Their authorities are
  [the reference summary](../../results/v03/development/reference_final/summary.json)
  and [coordinate summary](../../results/v03/coordinate-final/summary.json).
- The [current P5 findings](../../results/v03/extension_findings_reproduced.json)
  retain residual AR as optional, without primary promotion. The G3 AR fit
  fails qualification; the one G4 native comparison does not establish
  improvement. The 512-path choice is a finite sensitivity ceiling.
  [Conditional prior moments](../../results/v03/conditional_prior_reproduced.json)
  and [the saved G3 decoder replay](../../results/v03/g3_saved_first_step_review_reproduced.json)
  add no independent field, autonomous prior validation or capacity gain.
- The [first](../../results/v03/recovery_preflight/session_interruption_20261103T2104Z.json)
  and [second](../../results/v03/recovery_preflight/session_interruption_20261103T2220Z.json)
  session interruptions retain explicit unknown cost. History-1000 and
  sensor-100 recovered ESN receipts have incomplete accounting; missing
  command envelopes remain unavailable. No cause is inferred. Incomplete
  elapsed estimates may include downtime and are not exact compute totals or
  guaranteed lower bounds. Shared-resource monitor windows remain separately
  scoped.
- Strict tuning at `results/v03/development/tuned_final` is complete:
  59/60 outcomes qualify, and G3/rep000/A LGSSM retains its declared fallback
  after all four finalist rechecks exhaust 200 EM iterations. Three interrupted
  selection scopes have incomplete cost accounting. The selected revised
  pilot at `results/v03/development/revised_L1000` is 220/220 qualified with
  complete terminal cost receipts, initial history 1000 and forecast origin
  1300. Its actual primary cells all contain five common-qualified pairs.
- The [six-case calibration disposition](../../results/v03/release_evidence/calibration_disposition.json)
  verifies completed seals for ESN/GP/GRU on both tracks. All share one G4
  field, 25 serial blocks and a later evaluation at origin 3800; raw samples
  remain unchanged. Coverage and C-to-U transfer are empirical demonstrations,
  with no cross-field coverage estimate or distribution-free guarantee.
- Both G5 arms have 55/55 qualified outcomes. The complete corrected G6 replay
  has 55 jobs and zero parameter refits, under
  `results/v03/stress/design_schema_fix/stress_design.json` and
  `results/v03/stress/g6_velocity_shift_schema_fix`. Commit `99b0f8d` corrects
  strict saved-state validation; original design and failed five-job attempt
  remain preserved, and G5 YAMLs are byte-identical. The
  [paired stress audit](../evidence/v03/stress_summary.json) and
  [interpretation review](../../results/v03/release_evidence/scientific_interpretation/paired_stress_calibration_review.json)
  retain five paired fields per arm and pointwise intervals. G6 Track B loss
  increases do not establish a between-model robustness ranking.
- The unchanged [confirmation assessment](../../results/v03/confirmation/assessment.json)
  admitted R=60 per scenario: 240 independent datasets and 2640 model/task jobs.
  The uncapped requirement is 102, so `mcse_target_not_met_at_cap` remains
  explicit. The independent frozen campaign is in progress; planned MCSE is
  not realized precision or completed confirmation.
- The revised ordinary report and two fixed comparisons have
  [complete saved-artifact QA](../../results/v03/rendering/revised_corrected_visual_qa.json):
  14 mechanical cases, 84 actual sampled-image reviews and every one of the
  26 ordinary PDF pages. Four observed legend contacts and one tick-label
  margin were corrected by an audited presentation-only helper, preserving
  scientific inputs and unaffected bytes. Browser review remains blocked and
  continuous playback unperformed. Final synthesis, its QA, execution-cost
  and extension closure, the final source/wheel gate and public delivery
  remain pending.

The optional 500/4000 history endpoints, extra independent sensor fields,
frozen-encoder comparisons and capacity/prior ablations remain unexecuted.
The executed history choice does not establish saturation, no data benefit or
absence of harm from older history. Existing implementation tests and public
checkpoints remain evidence for their exact recorded scope. Final full
source/wheel validation follows all covered factual edits; R01–R20 is not
marked closed by this progress update.

## Previous execution snapshot retained at the recovered commit

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
