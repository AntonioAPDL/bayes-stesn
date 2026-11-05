# Kuramoto–Sivashinsky application protocol

## Status and scope

This document specifies a **custom, literature-grounded KS experiment**. It is not an exact reproduction of a published leaderboard. The initial candidate lists, schedules, history policy and resource limits are recorded in [`configs/applications/ks_experiment.json`](../../configs/applications/ks_experiment.json). Pilot measurements and targeted acceptance evidence are summarized in [the public preflight receipt](evidence/ks_preflight.json). Selected configurations and scientific results remain separate execution outputs: `freeze.json`, the ledger and result tables determine what has actually completed. This protocol does not infer completed selection or test results from the existence of code or pilots.

The application preserves the current repository's Bayesian spatial/ST-ESN and the independent frozen simulation study. Application code is under `src/bayes_stesn/applications/`; data, environments, fits and large results belong outside Git. The accepted model hierarchy and each comparator adaptation are described in [the model registry](ks_model_registry.md) and [native implementation audit](ks_native_audit.md). [Reproduction instructions](ks_reproduction.md) cover authorized delivery, verification, restoration and every runner stage. The repository revision and actual effective configuration hashes belong in every run receipt.

## Data identity and interpretation

The target is the clean simulated field of

\[
u_t=-u u_x-u_{xx}-u_{xxxx},\qquad x\in[0,22),
\]

with periodic boundaries and no repeated endpoint. This is the Kuramoto–Sivashinsky PDE, not a Kuramoto oscillator collection. The delivered solver uses 128 spatial points; the model observation panel uses the 64 even-indexed solver points. The core library contains 20 distinct trajectories with 10,000 saved states each. Three separate pilot trajectories contain 2,000 states each. The saved interval is 0.25 physical time units; 100 forecast steps span 25 physical time units.

The prepared arrays, generation settings, seeds and numerical method are unchanged. Solver truth at the additional 64 locations, derivatives and the known equations are unavailable as training inputs. No public KS replacement is permitted. The dataset's configuration governs all units; no Lyapunov exponent from another KS experiment is imported. [Pathak et al. (2018)](https://doi.org/10.1103/PhysRevLett.120.024102) and [Vlachas et al. (2020)](https://arxiv.org/abs/1910.05266) motivate data-driven forecasting of chaotic spatiotemporal systems, not an assertion that their data or protocols match this application.

Public provenance uses hashes rather than private account details or retrieval manifests:

| Object | Version or SHA-256 |
|---|---|
| KS archive | `ks_prepared_v1.tar.gz` |
| KS archive SHA-256 | `898f18deccc1fe6b3fb4d6eb0aa847244dc115efc56a260aceaae07bc22c227c` |
| Active source archive | `source_and_metadata_v2.tar.gz` |
| Source archive SHA-256 | `c88f8475b97a9b7159e643e06e424d0f58630400182957602e48b91097422965` |
| Complete catalog SHA-256 | `510a14dfa23e66fc861066dc4707fc46a192ba13588af2ba30f1e69478ce6be0` |
| KS configuration SHA-256 | `5f82a0e0e44a62925167bf2f55fb44ab7be086c3074f7325e76748288e9a0332` |

Restore into a separate cache root. The reader data root is `CACHE_ROOT/data`; preserve `data/ks/canonical` and `data/ks/raw` because canonical HDF5 files contain relative external links. The supplied restorer verifies complete byte counts and SHA-256 values and refuses conflicting destination files. Original archives and scientific data are retained; only verified redundant task-created copies may be removed.

## Temporal and spatial experiments

Fit each core trajectory separately. Select **one common configuration per model family and spatial track**, not one winning configuration or seed per trajectory.

All intervals are zero-based and half-open:

| Phase | Training targets | Validation targets | Test targets | Role |
|---|---|---|---|---|
| Core | `[0,6000)` | `[6000,8000)` | `[8000,10000)` | 20 scientific replicates |
| Pilot | `[0,1200)` | `[1200,1600)` | `[1600,2000)` | Three separate resource/implementation cases |

Lags, feature-conditioning rows and reservoir washout reduce usable likelihood/regression targets; every fit records that actual count. Solver spin-up, feature conditioning, reservoir washout, history length and MCMC warm-up are distinct quantities.

| Track | Permitted training and origin histories | Validation targets | Test targets |
|---|---|---|---|
| **A: strict unseen sites, primary** | Frozen interleaved 39 training sites only | 13 validation sites at validation times | 12 test sites at test times |
| **B: all observed, temporal** | All 64 observation-panel sites | The same 64 sites at validation times | The same 64 sites at test times |
| Contiguous robustness, separate | Frozen contiguous 40 training sites only | 12 validation sites | 12 test sites |

The facade protocol `all_observed` maps to native protocol `none`. The two main tracks have separate transforms, selection decisions and fits. They must not appear in one ranking. The strict facade's 64-column history array contains a mask; only 39 columns are available in Track A. `KSAdapter` verifies that mask and removes hidden columns before creating the model `SpatioTemporalData` object. Hidden zero fills are never treated as observed zeros.

The separate whole-trajectory 12/4/4 split is optional and is not implemented by the primary chronological adapter. A pooled extension requires its own pooled training-only transform, explicit episode resets and separately qualified claims based on four test trajectories.

## Scaling, geometry and information boundaries

For a trajectory and protocol, the supplied global scalar transform is

\[
z_t(s)=\{u_t(s)-\mu_{\mathrm{train}}\}/s_{\mathrm{train}}.
\]

Both parameters use only the permitted training times and training sites. Full-training fits use these exact delivered values. A smaller-data experiment must fit and version a new subset-only transform. Additional PCA/POD, feature standardization, covariance estimation, learned response bases and any calibration must observe their own declared training boundary.

The Bayesian adapter accepts the delivered standardized responses with response center 0 and scale 1. It retains its existing conditioning-prefix PCA and feature calibration. This is an explicit application adapter, not a change of likelihood or prior. The GP and GRU retain documented internal training-only centering/conditioning; their outputs remain on the input response scale before the delivered inverse transform.

Spatial covariance APIs receive the fixed unit-circle embedding

\[
c(x)=(\cos(2\pi x/22),\sin(2\pi x/22)).
\]

The existing Euclidean Matérn restricted to this embedded circle is positive definite. It is not a Matérn obtained by arbitrarily substituting wrapped distances. Physical coordinates are retained for interpolation and plots. A compressed irregular 39-site vector is never passed to an FFT or equally spaced convolution. Geometry-only coordinate bases may evaluate unseen coordinates; hidden response histories may not fit that basis or initialize a state.

The delivered dynamics facade internally constructs histories and future evaluation labels. The application enforces **output separation** and perturbation invariance. It does not claim future-free disk access. Evaluator labels, including future availability/scoring masks, never enter model objects or recurrence states.

## Forecast origins and causal context

The initial frozen context policy is **history 128 and horizon 100**. For first target index `a`:

- History is `y[a-history:a]`.
- Targets are `y[a:a+100]`.
- The delivered adapter origin is `a-1`.
- Lead `h` refers to physical elapsed time `0.25*h`.

Full validation starts are `6000,6100,...,7900`; full test starts are `8000,8100,...,9900`. Pilot validation starts are `1200,1300,1400,1500` under the same horizon. Each issued forecast is autonomous for all 100 steps. A later origin may use newly observed earlier values at permitted sites, but parameters remain fixed and its own future interval stays unavailable.

**Synchronization is an execution gate.** The legacy benchmark API required replay of the entire fitted prefix. The KS native adapter instead replays the explicitly supplied window with fixed fitted transformations. Before screening, all four Bayesian candidate architectures were checked on all three pilot trajectories and both main spatial tracks: 24 cases for the main model seed and 48 cases for the two sensitivity seeds. Histories 128, 256, 512 and 1,024 were compared with full permitted training replay at the training endpoint. All 72 cases passed the state-and-feature tolerance `atol=1e-8`, `rtol=1e-6` at history 128, the smallest declared candidate. Worst absolute state discrepancy was `3.515e-10`; worst feature discrepancy was `1.390e-10`. These checks fitted feature transformations only, performed no MCMC and read no future evaluation labels.

This evidence fixes 128 as the common initial context policy; it is an empirical pilot check, not a universal synchronization theorem. Selected fitted configurations still require the declared core training/validation checks. A failed later pre-test check must produce a versioned causal-context decision before test access, with the same permitted history offered to comparators. No implementation may silently replay more than the reported 128 states. The checked feature maps include conditioning prefixes of 80, 512 or 1,024 training rows; those preprocessing boundaries do not increase the history supplied at a forecast origin.

Reservoir, recurrent and lag states reset at every new forecast initialization and between trajectories. Conventional vector baselines advance their observed-site state; POD baselines advance the specified latent state. Their periodic decoder is applied only to output. Neither decoder smoothing nor hidden truth is silently fed back into the recurrence.

## Bounded screening and inference

The execution configuration fixes the following initial staged search. It is bounded by the measured resource allocation below:

| Stage | Core trajectories | First-target validation indices | Opportunity |
|---|---|---|---|
| Screen | `[0,5]` | `[6000,6500,7000,7500]` | Up to four expensive-family configurations |
| Promote | `[0,5,10,15,4,9]` | `[6000,6200,6500,6700,7000,7200,7500,7700]` | Two candidates per family/track |
| Selected fitting | All 20 | Full official schedule | One common configuration per family/track |
| Seed sensitivity | `[0,5,10,15]` | Frozen schedule | Two additional fixed model seeds for stochastic fits |

The finite initial opportunity is four Bayesian ST-ESN candidates, two GP candidates, four basis-GRU candidates, eight linear/POD-VAR candidates, eight conventional ridge-ESN candidates and eight POD-NVAR candidates per main track, plus one configuration for each anchor. Track B additionally has four CNN candidates. These are structured proposals, not Cartesian products. The conventional regression families receive more proposals because their measured closed-form fits are inexpensive. Exact dictionaries, including parameters that differ together, are in the configuration and [registry](ks_model_registry.md).

Screening uses model seed `20261105`; sensitivity uses the separately reported seeds `20261106` and `20261107`. Deterministic fits do not receive artificial duplicate seeds. No completed identical effective configuration is rerun; hashes include resolved parameters and actual data identities. The accepted Bayesian reference is `bayes_00`. Short MCMC pilots measure cost and identify problems; lack of convergence under deliberately reduced effort is not evidence against the model class.

Native probabilistic families select by episode-balanced validation CRPS; point-only families select by episode-balanced validation squared error. Average each loss within leads 1–10, 11–25, 26–50 and 51–100, then give the four bands equal weight. Also report ordinary 100-step aggregates and lead-specific values. These are different objectives across probabilistic and point-only families. Among eligible candidates within 1% of the best validation loss, prefer lower predeclared complexity and retain the near-tie flag. Staged allocation is inspired by [Hyperband](https://www.jmlr.org/papers/v18/16-558.html), without claiming its exact algorithm or theoretical guarantees.

The configured boundary rule permits at most one additional validation-only candidate per family/track for linear VAR, ridge ESN, POD-NVAR or CNN. A selected numeric ridge boundary is extended one decade with all other settings and the seed fixed; a selected 60-epoch CNN budget may extend to 90 epochs with architecture fixed. Bayesian/GRU capacity remains the profiled roster; a capacity-boundary winner is reported as a limitation. Refinements are recorded as additional evaluated settings before predictive-precision assessment and freezing. No test value selects candidates, seeds, calibration, failure policies or visual examples. Finish selection and predictive Monte Carlo precision checks for **both main tracks before scoring either test track**. Final fitting continues to use training data only. Posterior inference is conditional on validation-selected architecture and hyperparameters.

The CNN uses common fixed training-budget candidates of 20 or 60 epochs, selected by official-validation autonomous lead-band MSE. Its selected epoch budget is shared across trajectories within Track B. The study does not use the optional per-trajectory validation early-stopping API. The existing GRU retains its distinct internal training-only stopping/refit procedure; neither uses test values for stopping.

The accepted Bayesian final diagnostic target is four chains, rank-normalized/folded split R-hat strictly below 1.01 and bulk/tail ESS at least 400 for consequential quantities, with predictive Monte Carlo error small relative to a claimed difference. Monitor covariance/noise/shrinkage parameters, readout coefficients and predictive functionals. Record any monitoring restriction. This sampler is adaptive random-walk MCMC during warm-up, not HMC; HMC divergence statistics do not apply. See [Vehtari et al.](https://arxiv.org/abs/1903.08008).

The GP and GRU use their actual likelihood-optimization qualification, not artificial chain diagnostics. A failed posterior is never replaced by a point forecast and then given Bayesian coverage. Preserve inference-unqualified jobs, permit only a bounded training/validation repair, and exclude unqualified probability estimates from qualified probabilistic tables while reporting failure counts.

## Predictive distributions and scoring

The primary data arm has `noise_fraction=0` and `input_mask=none`; the evaluation target is the **clean simulated field**. Native statistical residual/nugget terms represent fitted model discrepancy on that field. They are not an added known sensor-noise process and are not removed or floored to improve coverage.

For Bayesian paths, choose one static parameter/readout draw per complete path, keep it fixed through the horizon, simulate the accepted residual process and feed back the path's own permitted-site predictions. GRU paths preserve spatial innovation covariance; GP paths preserve both spatial and temporal covariance, including conditional unseen-site processes. Independent horizon-wise marginal draws do not satisfy this contract. Model seeds, MCMC chains and predictive paths remain distinct objects.

Start with 256 predictive paths per origin, generated in batches of 64. Before `freeze.json`, the validation-only precision procedure uses trajectory 0, origins 6000 and 7500, and four independently seeded predictive ensembles conditional on each retained fitted distribution. It checks path counts 256, 512 and 1,024, increasing until normalized single-ensemble CRPS standard deviation is at most 0.01 and the coefficient of variation of mean 95% interval width is at most 0.10, or the cap is reached. The common final count is the largest case requirement recorded in the freeze. These thresholds are experiment conventions based on four simulation replicates, not an MCMC convergence guarantee or scientific-replicate confidence interval; cap failures remain flagged. Near-tie simulation contrasts diagnose uncertainty without reselecting candidates or changing earlier screening path counts. Primary point predictions are the mean of complete predictive paths, including for GP and GRU. A posterior-mean-parameter rollout or a separate Rao–Blackwell estimate is not substituted. Point-only controls have no invented intervals.

All final metrics use physical units after the supplied inverse transform. Report RMSE, MAE, RMSE divided by permitted training SD, horizon curves and leads 1, 5, 10, 25, 50 and 100. Qualified distributions additionally receive empirical CRPS, central 50%, 80%, 90% and 95% marginal coverage/width, spatial energy and variogram scores. CRPS uses the empirical-distribution V-statistic; no iid finite-ensemble unbiasedness is claimed for MCMC-derived paths. Proper-score definitions follow [Gneiting and Raftery (2007)](https://doi.org/10.1198/016214506000001437).

The implemented energy score evaluates the spatial vector separately at each lead, with Euclidean norm divided by the square root of the official target-site count. Its pair term uses 1,024 seeded pairs drawn with replacement from the empirical paths, including possible diagonal pairs. This Monte Carlo approximation can be slightly negative; it is not an exact all-pairs evaluation. The variogram score follows [Scheuerer and Hamill (2015)](https://doi.org/10.1175/MWR-D-14-00269.1), with power 0.5 and up to 128 geometry-only site pairs with normalized inverse chord-distance weights. Pair selection is fixed across methods within a target domain. These are spatial per-lead scores, not a claim to score the complete temporal trajectory jointly.

Valid prediction time is the first lead where spatial RMSE divided by training SD exceeds 0.4; a path that never crosses is right censored at lead 100. This is an experiment convention. A finite normalized spatial RMSE exceeding 10 at any lead is separately flagged as divergence; nonfinite or overflowing rollouts fail rather than being clipped. Report these events and denominators explicitly.

Average origins within each trajectory, then give trajectories equal weight. Pair method comparisons by trajectory and origin. Confidence summaries use paired trajectory-level differences or a cluster bootstrap over the 20 trajectories, retaining all dependent origins within each cluster. Sites, times, origins, chains and posterior paths are not independent scientific replicates. Report both eligibility/failure counts and successful-case score summaries; no silent dropping of unfavorable episodes is allowed.

## Resources, resumability and reporting gates

The measured initial environment had an 8 GiB cgroup memory limit, an eight-CPU quota with nine affinity-visible CPUs, no detected GPU, and 62,315,782,144 bytes of available disk. A filesystem quota command was unavailable, so `statvfs` available space was used conservatively. Host RAM was not substituted for container headroom. The retained disk reserve is 12,463,156,428 bytes, or 20% of that initially available space, exceeding 2 GiB. No model fits were already running at inspection.

Expensive pilots ran one at a time with one numerical thread. Measured process peak RSS was approximately 118 MB for the short Bayesian pilot, 101 MB for GP, 447 MB for GRU and 454 MB for CNN. The corresponding fit times were 3.708 s, 1.797 s, 1.537 s and 0.527 s; these are deliberately reduced-effort resource profiles, not adequate final fits or predictive rankings. The 100-step CNN frozen-weight float32/float64 comparison had maximum absolute difference about `1.15e-6`; the GRU comparison passed its `atol=1e-6`, `rtol=1e-5` criterion. These checks concern fixed-weight inference, not equivalence of training in both precisions.

The initial campaign allocation is **four hours of active wall time at up to two single-thread workers**, with a cumulative eight worker-hour ceiling. Screening has at most 1.25 wall hours per invocation and a cumulative 2.5 worker-hour ceiling, reserving at least 65% of the allocation for selected fitting, forecasts, diagnostics and reporting. The measured extrapolation is 1.5–3.5 hours overall, based on pilot fit/forecast times, training-row and inference-effort scaling, two workers and an I/O margin. It is an estimate to be revised from measured core timings, not a completion claim. Invocation wall limits and cumulative worker-time ceilings are distinct; all retained attempts contribute to the latter across resumes.

The configured combined-worker hard cap is 4,080,218,931 bytes, with 2,040,109,465 bytes per worker. Before a concurrent group is admitted, the scheduler sums the largest simultaneous estimated worker peaks and requires a 1.5 multiplier to fit inside `min(combined cap, 0.65 × current container-aware headroom)`. Current headroom is raw cgroup headroom plus only half of inactive file cache; the full group is admitted together and has distinct checkpoint owners. Per-worker admission and two-second resource monitoring continue during execution. A fixed cap never substitutes for the dynamic headroom gate. Resource or time stops preserve checkpoints and leave jobs pending.

Use sufficient statistics and valid separable algebra. Do not construct the full space-time Kronecker regression design or giant readout covariance. NVAR features are chunked, with explicit feature/design/Gram costs. Sensitive linear algebra and metric accumulation use float64; neural float32 needs its own numerical check. Forecast batches and scores are bounded; full path cubes across all episodes/models/origins are not retained.

The ledger distinguishes pending, running, complete, failed and inference-unqualified jobs. Atomic numeric checkpoints bind data/configuration/source/environment identity and RNG state. Completed fits are reused; incompatible checkpoints are refused. The native Bayesian sampler and CNN can resume interrupted computation. Any implementation that must restart an interrupted internal optimization discloses that limitation rather than claiming exact mid-fit resume.

Before expensive screening, verify archives, schema, masks, index conventions, inverse transforms, relative links, coordinate prediction, periodic seam behavior, autonomous shapes/units, known-answer scores and interruption/resume behavior. Test fixed-fit/fixed-randomness forecast invariance to altered future labels and forbidden histories; separately test refitted preprocessing invariance outside its fitting boundary. Do not rerun selection or calibration as part of that invariance test.

The configured report case is core trajectory 0, first target 8000, separately for Tracks A, B and C. Main-track interval traces use target-array positions `[0,5,11]` for A and `[0,16,32,48]` for B. These positions are fixed before test access; plots label their actual physical coordinates. Figures distinguish strict-track observed inputs from withheld scored sites and use consistent physical color scales. Report truth/forecast/signed-error heatmaps, lead curves with trajectory uncertainty, probability scores and coverage/width, interval traces, and performance versus compute cost. The compact side-by-side MP4 compares the predeclared Bayesian ST-ESN, basis GRU and ridge ESN at no more than 100 frames, 10 frames/s, without retained frame dumps. Figures and text must honestly distinguish accuracy, calibration, cost and unseen-site performance.

Contiguous-site robustness runs after completion of the two main tracks. All eight strict-capable families are prespecified, using their interleaved-validation-selected configuration with newly fitted contiguous training transforms/features and parameters; no methods are chosen from main-test rankings. This is configuration transfer across spatial masks, not an assertion that those settings are optimal for the contiguous split. Optional noise fractions 0.05 and 0.15 remain separate, unscheduled one-factor extensions; their delivered clean-training normalization must not be mislabeled as noisy-only training. No incomplete pilot or partial job list constitutes the completed application.
