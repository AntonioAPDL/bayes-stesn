# v0.3 completion dependencies and remaining obligations

This checklist interprets the approved PRO plan and the separately declared
study recipes. It is an execution checklist, not a completion or improvement
claim. The numerical package is frozen at
`9f73b28a6ee6b8eec4bd21c3366b0c260c9433f798610be16697b2e44166d5ac`.
Older source receipts remain historical. Run at most two model workers overall,
with one numerical thread per worker; this workstream owns one worker.

## Dependency checklist

| Order | Action and dependency | Completion evidence |
|---:|---|---|
| 1 | Finish current-source reference, coordinate/P5 replays and the full 60-job strict retuned development campaign. G4 selections may unlock the dependent fixed-specification studies before the other tuning mechanisms finish. | Verified current-source manifests, complete inventories, qualification and measured costs; retain older interrupted campaigns. |
| 2 | Bind all eleven required G4 model/track selections, completed P5 path-count findings, and the full-duration G3 numerical receipt. No missing/incomplete selection may become a reference fallback. | `study_design/drafts/study_provenance.json`; all eleven transferred selections are qualified. |
| 3 | Execute six fresh training-only timing fits at O100/T2300; freeze common model time caps without changing algorithm/diagnostic budgets. | Six timing receipts, no forecasts or final observations; `study_design/resolved/study_provenance.json`. This stage has completed. |
| 4 | Execute history 1000/2000: five paired fields per G1/G3/G4, 330 fits. Keep full 500/1000/2000/4000 design immutable. | Same realized fields, roles, test window, scale and seed policy; all terminal outcomes; `history_choice.json` from the previously declared rule. |
| 5 | Execute the separate 25/50/100 sensor design: one G4 field, 33 fits. | Nested O, fixed C/U, common 25-site score set and scale; saved spatial/temporal rank and fixed-basis geometry diagnostics. One-field sensitivity has no replication MCSE. |
| 6 | Execute only the chosen uncropped revised history at intended confirmation geometry: five fields per G1–G4, 220 fits. | Matched-design complete pilot, exact qualified intersections, all-attempt outcomes, paired variances and qualification rates. Do not substitute favorable fields. |
| 7 | Execute the separate declared calibration cases from qualified revised G4 replication0 fits: ESN/GP/GRU × A/B, up to 150 calibration forecasts plus six later forecasts, zero parameter refits. | Sealed raw/calibrated outputs on the same later block; C only calibrates B, U remains sealed; unavailable preselected fits stay unavailable. |
| 8 | Execute the separate stress recipe after basic reliability: whole-field-t G5 and local-t G5, five paired fields each (110 new fits), plus blind G6 against five saved G2 controls (zero parameter refits; at most55 forecast-replay jobs). | Unchanged frozen specifications; actual coupled stream/role checks; both sides normalized by control training SD; G6 original-score replay verification and no post-origin conditioning. |
| 9 | Close every P5 extension disposition, including prior-predictive evidence for an actually evaluated statistical extension. Do not equate posterior residual checks with prior-predictive checks. | Current-source extension findings and explicit promoted/optional/rejected/deferred rationale; no robustness or prior-improvement claim without its experiment. |
| 10 | Apply the independent confirmation gate only after revised evidence is complete and sufficiently qualified. | Four primary contrasts, at least five common-qualified pairs per constituent pilot cell, frozen attempted R20–60, source/runtime, settings, seeds and G3 evidence. If blocked, retain a development-only conclusion. |
| 11 | Execute the sealed confirmation unchanged if admitted; no outcome-adaptive replication or seed replacement. | Complete qualified/unqualified outcomes, paired native and all-attempt operational results, actual precision, costs, negative findings and cap shortfalls. |
| 12 | Render from verified saved artifacts, inspect media/PDF, run final source/wheel/core/neural/CLI checks, then verify public commit/CI/tag/assets/checksums. | Final input/output inventories and actual public receipts; old checks cannot validate changed hashed inputs. |

The core fixed-specification branch is **583 fits**. Including the two newly
declared G5 arms makes **693 scientific fits**, plus six timing fits. The active
reference/tuning/coordinate/P5 work, forecast-only calibration/G6 work, and any
eventual confirmation are separate costs. Confirmation at the current eleven-job
roster is `44 × R` fits: 880–2640 for R20–60. Do not describe 583 as the complete
remaining workload.

## Which branches are conditional?

- **G5/G6 base stress comparisons:** PRO Section9 explicitly retains these
  separate studies after basic reliability; the new `stress_recipe.yaml`
  prospectively closes their design. They are not in the 583-fit recipe and
  cannot be marked executed by deferring robust-model variants. If a real
  reliability/resource gate blocks execution, report that remaining scope and
  make no robustness or shift-performance claim.
- **Robust t, RHS/grouped priors, latent AR, anisotropy, nonstationarity,
  heteroskedasticity, E-QESN innovations and other statistical variants:**
  conditional. R13 permits compatible bounded evidence or a specific technical
  deferral. A new model/algorithm identity, supported inference path, elicited
  prior and meaningful validation evidence are required before promotion.
- **Prior-predictive checks:** required for a statistical extension actually
  evaluated or promoted. Existing prior algebra/shape tests and posterior
  residual diagnostics do not by themselves establish them. The exact sampler
  coordinate change preserves the prior and instead needs the already required
  target/reference/restart checks. No broad simulation-based calibration or
  new-prior campaign is required by this plan. The AR prior-predictive evidence
  must be supplied or remain an explicit gap.
- **Interval calibration:** conditional in the generic extension table, but now
  a committed separate experiment in the accepted remaining-study recipe.
  Eligibility remains conditional on each preselected base fit qualifying.
  Native coverage, interval score and width are mandatory companion reporting
  regardless of whether an interval adjustment can be run. Serial calibration
  blocks do not establish exchangeability or independent replication coverage.
- **History500/4000, a full capacity/prior factorial, additional sensor fields,
  equal-compute tuning, G5 heteroskedasticity and G6 post-shift adaptation:**
  conditional expansions, not automatic additions. History4000 also needs a
  new 4300-row timing review. The blind G6 task cannot be reused as an
  adaptation or post-shift calibration experiment.

## Storage and scheduling

A recent local measurement found about **13.5GiB available of63GiB**, not the
earlier55GiB. Own `results/v03` was about0.4–0.5GiB, reports about43MiB and the
locked runtime about1.27GiB. Unrelated shared activity changed free space by
several GiB within minutes. Do not touch that activity or rely on a stale `df`.

The frozen runner handles memory admission but does not reserve disk. Use
`configs/benchmark/v03/studies/run_chunks.py` for one-worker invocations. It
checks **at least4GiB reserve plus the declared next-job output/temporary
allowance** before every invocation, defaults to one job, and exits75 on low
headroom or no terminal progress. Every complete campaign still goes through
one ordinary core resume to verify source/manifests/artifacts before the wrapper
accepts success. Concurrent external writes can still fill disk during an
atomic job; any resulting infrastructure failure remains incomplete and
recoverable, not a model fallback.

Use512MiB for each core/history/revised job's provisional output/temporary
allowance and1024MiB for a sensor job until actual archives refine it. These are
planning margins, not measured upper bounds. The raw retained 512×100 path cube
is about9.8MiB at25 targets and44.9MiB at115 targets, before compression, model
state, copies, JSON and archive temporaries. The long-history simulation arrays
are only part of the total: keep all models, posterior checkpoints and receipts
in the estimate. A provisional **6GiB incremental envelope for the583-fit core
branch** is conservative relative to current short-reference artifact sizes
but must be checked against the first actual selected-model files. Budget the
110 G5 fits, calibration/G6 artifacts and reporting separately.

Before archiving or releasing, allow for the archive itself and temporary
duplication as well as the4GiB reserve. Before confirmation, recalculate storage
from the now-measured selected-model files and frozen R; its880–2640 fits are
not covered by the core6GiB planning envelope. Pause honestly if the full stage
or its next atomic job cannot be admitted; do not alter a sealed scientific
design to make a disk estimate pass.
