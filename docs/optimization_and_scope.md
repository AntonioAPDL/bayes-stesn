# Optimization decisions and supported scope

The changes target measured or directly visible bottlenecks. They
preserve the original probabilistic model where an optimization is advertised
as exact; model approximations remain separately labeled.

## Current v0.3 execution and measurement policy

The new execution workflow separates immutable scientific/computation identities
from mutable scheduling attempts. Source, numerical runtime, effective candidate
design, fitting limits, final seed policy and saved artifact hashes bind reuse.
Worker count, admitted memory and between-job chunk limits can change without
silently changing the scientific procedure. One POSIX-locked coordinator owns
a run directory. Phase-aware ESN checkpoints preserve the initial fit,
diagnostic and one-continuation allowance across process interruption; see
[the derivation and recovery contract](derivations/07_sampler_coordinates_recovery.md).

### Conservative resource admission

Start at two workers and one numerical thread per worker on a measured 8-GiB
host, then use the recorded headroom. These are starting settings, not a
universal optimal concurrency. `benchmark plan` reports the enforced
host/cgroup memory ceiling and CPU quota/affinity. The worker cap is at most
four. Job memory is clamped to the requested budget, 65% of the memory ceiling,
and remaining headroom after reserving the smaller of 2 GiB or 25% of the
ceiling. Current cgroup usage can reduce the allowance further.

The dimension-based estimate includes model work, prediction/scoring arrays,
parent work and runtime overhead. Planning uses a representative search seed;
execution checks each actual job's resolved configurations before admission.
The scheduler reduces concurrency or leaves an oversized job pending. It does
not reduce training rows, model dimensions, draws, paths or diagnostic standards
to make a run fit. Admission is a conservative estimate, not a kernel-enforced
memory sandbox: sparse-factor fill, libraries and allocator behavior still need
isolated measurements. An out-of-memory interruption is infrastructure failure,
not evidence that a statistical method produced a bad forecast.

`--max-jobs` and `--max-wall-seconds` stop submitting work between atomic jobs.
Active fits finish under their scientific allowance, so a scheduling deadline
can overrun. Optional `fit_time_limits_seconds` is different: it is a frozen
per-model allowance around each fitting call, recorded in the protocol and
treated as a declared method failure if exhausted. It does not cover forecasting
or scoring. The default is no elapsed-time cap; iteration/draw limits still
apply. The POSIX `setitimer` implementation requires a worker main thread and
refuses an already-active process alarm. Long native calls may delay signal
delivery, so it is not a hard subprocess deadline. Coordinator locking uses
POSIX `flock`; this workflow is currently Linux/POSIX oriented.

### Cost and memory scopes

| Quantity | Interpretation and limit |
|---|---|
| Planned fit calls | Screening, rechecks, possible replacements and final refits; internal neural refits/MCMC continuation are not separate calls |
| Projected fit wall time | Representative per-model fit seconds multiplied by planned calls; null without measurements and distinct from elapsed parallel campaign time |
| Recorded selection/final costs | Actual completed fitting/forecast stages and recorded attempts; resume reuses receipts without pretending original selection was free |
| Interrupted-worker cost | May be unobservable after process death; incomplete compute accounting remains explicit |
| Process peak RSS | Process-lifetime high-water mark, including earlier jobs when workers are reused; not a per-fit incremental peak |
| Cgroup memory usage | Aggregate current group occupancy, including other live work; not attributable to one model |
| Isolated allocation/fit profile | Fresh subprocess measurement at explicit dimensions, host and runtime; useful for a bounded comparison, not a universal scale law |
| Planned artifact bytes | Array estimates before compression; models, checkpoints, figures, temporary writes and dependencies add storage |

The paired forecast-ownership probe compares a public defensive copy with a
private transfer of the same 256×100×505 float64 paths in fresh processes.
Its [saved receipt](evidence/v03/forecast_ownership.json) establishes identical
path/mean hashes and a reduction of one complete path-buffer allocation at the
benchmark adapter boundary. See [the study-design discussion](study_design_v03.md)
for the measured RSS and its scope. Public constructors continue to copy caller
arrays, including arbitrary strided layouts. Internal ownership transfer is
restricted to complete immutable/privately owned contiguous allocations.

The benchmark scorer still materializes its full sample arrays. Global path IDs
and the existing core batch iterator preserve coherent trajectories, but they
do not constitute disk-backed or end-to-end streaming benchmark scoring. Add
such a route only when measured dimensions justify it, with exact score and
joint-path invariants checked. No covariance, prediction or qualification
change is justified solely by an allocator measurement.

The [tuning specification](benchmark_tuning_v03.md) documents bounded optimizer
restart rules and unchanged convergence criteria. The
[requirements/evidence map](plans/v03_requirements.yaml) separates implemented
engineering, executed measurements and scientific conclusions.

## Earlier v0.2 optimizations and measurements

| Area | Implemented change | Benefit and remaining boundary |
|---|---|---|
| Forecast storage | Select requested components; retain only observations for benchmark scoring | Avoids allocating four equal-sized outputs; observation component remains required |
| Ownership | Freeze privately generated contiguous allocations without copying them again | Public constructors still copy user arrays; round-trip ownership tested |
| Predictive batching | Named global path IDs and `path_start`; batch iterator | Batches concatenate exactly in the same runtime; frame and spatial workspaces still exist |
| Mixtures | Globally indexed ensemble-member and within-member paths | Worker/batch ordering no longer changes ensemble draws; RNG scheme deliberately versioned |
| Spatial scores | Floyd sampling of unordered pairs in O(requested pairs) storage | Avoids O(n squared) upper-triangle construction; explicit all-pair requests still imply large outputs |
| Archive writes | Cleanup covers compression, flush, fsync and atomic replacement | Existing canonical output survives pre-replacement failures; post-rename durability failures remain visible |
| Benchmark checkpoints | Default to chain endpoints; preserve explicit caller cadence | Historical v0.2 pilot fit fell from 33.64 s to 9.56 s with identical numerical results; without intermediate checkpoints, interrupted chain work after the last endpoint must replay |
| Checksum reads | Contiguous byte views and direct private NPZ arrays | Removes unnecessary full-size byte/copy allocations |
| Vecchia geometry | Cache fixed predecessor choices | Hyperparameter-dependent conditional coefficients are recomputed correctly |
| Vecchia likelihood | Sparse nugget-updated LU and operator sufficient statistics | No dense spatial inverse, covariance or YY matrix for the supported route; fill-in remains geometry dependent |
| Vecchia determinant | Exact conditional-variance determinant; prior LU is lazy | Removes an unused second factor in the supported inference workflow |
| Vecchia solves | Non-subtractive identity-interpolation solve | Avoids Woodbury cancellation for extreme spatial/nugget variance ratios |
| Vecchia prediction | Sparse triangular ancestral latent simulation | Samples the existing latent approximation plus separate nugget; no observation-Vecchia substitution |
| LGSSM | Coordinate basis with orthonormal observed-site loadings | Filtering works in small state dimension; full observation likelihood includes omitted-space residual terms |
| GP | Spatial eigensystem and low-order exact temporal state representation | Avoids an (nT)-by-(nT) covariance; new-site joint residual covariance is still bounded dense |
| KRR | Nyström features and normalized ridge | Trades a controlled kernel approximation for bounded feature dimension; exact reference mode is retained |
| Neural graph | Sparse geometric diffusion and window stride | Reduces graph/window work; architecture and training precision are recorded |
| Experiments | Process workers, one numerical thread each, strict manifests and per-job resumes | Avoids oversubscription and repeated completed jobs; worker memory preflight is a planning bound, not a measured peak |

## Exact scope of the end-to-end operator route

The automatic operator path requires `CovarianceSpec(kind="vecchia")`, iid
Gaussian innovations, `PriorSpec.ridge()`, and collapsed inference. It also can
be requested explicitly with `SamplerConfig(backend="vecchia_operator")`.
Incompatible explicit requests fail. Dense/reference modes retain their site
caps. The arbitrary-target forecast appends targets after observed sites, so
the observed factors and response history are unchanged.

The spatial approximation is the **latent** predecessor-based Gaussian field.
The observation nugget remains separate. Sparse precision multiplication and
sparse solves replace dense spatial algebra; the temporal readout Gram and small
spatial-basis Gram remain dense. Therefore large reservoir dimension can still
be expensive even when the number of sites is handled sparsely.

The measured engineering profile uses small readout dimensions and deliberately
short chains. It establishes execution/resource behavior only. It does not
establish MCMC convergence, comparative forecasting accuracy, mesh convergence,
or million-site scalability. The final sparse-factor change reduced the
2048-site process peak from about 153.6 to 125.2 MiB in the local profile. Wall
times varied, so a speedup is not inferred from that pair of measurements.
An additional 4096-site execution check completed at 184.0 MiB process peak RSS,
with 5.44 s fit and 6.07 s prediction for the same deliberately small dimensions.

## Conditional extensions

The full machine-readable roster is [benchmark_registry.yaml](benchmark_registry.yaml).
The core comparison and the optional matched basis-GRU control are implemented.
The earlier plan explicitly made external R/Julia, complete-grid neural
operator, diffusion, and pretrained-model additions conditional on their own
capability, dependency, cost, and data-budget gates. Those integrations remain
listed as `deferred_optional`; they are not exposed as callable fitted models.

Complete-grid methods require track C with the same complete-grid information
for every comparator. Pretrained models require a separate external-data
budget. Neither should be ranked as if it received only the sparse historical
panel used in tracks A/B. Installing a package alone would not meet these
scientific comparison requirements.

## Reproducibility boundaries

Named streams make scheduling order irrelevant; installed-wheel parity checks
guard against importing an editable source checkout accidentally. Float64 is
used for the statistical core. The PyTorch mean models use explicit CPU
float32 training. Exact seeded byte equality is only promised within the same
supported numerical runtime, not across arbitrary BLAS, compiler or device
versions. The profile records the inference and forecast components actually
retained and the exact candidate and continuation budgets.

The objective is efficient, auditable inference under these declared limits.
There is no claim of globally optimal hyperparameters, universally superior
forecasting performance, or fully converged inference merely because unit
tests pass.
