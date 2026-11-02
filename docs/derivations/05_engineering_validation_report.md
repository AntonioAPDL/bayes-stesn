# Engineering, validation, reproducibility, and reporting

## Status and implementation boundary

This chapter is a specification for a future implementation. The API examples and commands below are proposed interfaces, not claims that an installable package already exists. The existing V3 scripts are the numerical reference for the baseline; none of the proposed extensions has been fitted or benchmarked as part of this planning deliverable.

The proposed package name is **bayes-stesn**, with Python import name **bayes_stesn**. Confirm name availability when creating the repository and publishing a package. The first implementation should use the established NumPy/SciPy numerical core. An R wrapper can follow after the data and serialization contracts stabilize; a second independent sampler in R would duplicate verification work. A public documentation repository can precede implementation, with its status prominently marked.

## Supported data and scientific scope

The first release models one scalar response over space and time. Public arrays use **time by observed site** ordering; the internal Gaussian derivations use its transpose, $Y\in\mathbb R^{n\times T}$. Conversion occurs in one tested boundary function.

**SpatioTemporalData** contains responses, ordered site identifiers, coordinates, equally spaced time identifiers, response units, coordinate units, and optional metadata. The initial implementation requires a complete, balanced response matrix at the input sites. Spatially withheld evaluation locations are held in a separate **EvaluationTruth** object that cannot be supplied to the fitting API. A spatial holdout must be removed from both the reservoir inputs and the likelihood. Missingness and evaluation masks are different concepts.

The common isotropic Matérn model and explicit-center radial mean dictionary should support Euclidean coordinates in one, two, or three dimensions. The initial anisotropy and nonstationarity implementations are restricted to two dimensions, matching the derivations. Geographic coordinates in degrees require an explicitly supported metric or projection; the API must not silently interpret them as planar distances. Irregular temporal sampling, multivariate response channels, and missing input-site observations are later features that require separate statistical specifications. They must initially raise an informative unsupported-configuration error.

Target locations may be irregular and need not equal the observed sites. Future simulation always generates a joint field over the union of input and requested target locations, with duplicate locations resolved by identifier and coordinate checks. This preserves spatial dependence between values used to update the reservoir and values reported to the user.

## Public object model

Objects should be typed, validated, immutable after construction, and serializable without executable objects. Mutation of fitted transformations would invalidate the meaning of posterior draws.

| Object | Responsibility |
|---|---|
| SpatioTemporalData | Observed data, units, coordinates, time ordering, and immutable data identity |
| SplitPlan | Conditioning, fitting, validation, calibration, and final evaluation boundaries; spatial holdout identities |
| FeatureSpec | PCA/local/graph input map, reservoir layers, leaks, nonlinear readout terms, and feature seed |
| SpatialMeanSpec | Radial centers, widths, coordinate terms, and normalization policy |
| PriorSpec | Readout, covariance, scale, and latent-process priors with explicit parameterization |
| ModelSpec | Composition of mean, innovations, observation noise, and prior specifications |
| SamplerConfig | Chains, adaptation, retained iterations, numerical tolerances, and checkpoint policy |
| PreparedModel | Frozen transformations, design metadata, initial history, and chosen inference backend |
| FittedModel | Posterior state, diagnostics, feature map, conditioning provenance, and complete model specification |
| ForecastContext | Forecast origin and available history/state, separate from the parameter posterior |
| PredictiveSample | Joint trajectories, physical units, target definitions, draw identities, and forecast policy |
| StudyResult | Comparable fits, predictions, scores, computational measurements, and locked analysis choices |
| EnsembleSpec / FittedEnsemble | Candidate feature-map identities, fitting policy, component posteriors and predictive weights |
| CalibrationSpec / CalibrationMap | Calibration target, historical calibration partition, scores and fitted interval adjustment |

**prepare** may use only the partitions authorized by the split policy. The compatibility mode for V3 preserves its 600-time conditioning transformations and common 128-site response scaling. The general mode records whichever observed sites and historical window define preprocessing. These policies must be explicit rather than hidden constants.

### Proposed end-to-end use

The following is the intended API, to be implemented and tested later:

~~~python
from bayes_stesn import (
    SpatioTemporalData, SplitPlan, FeatureSpec, SpatialMeanSpec,
    ModelSpec, PriorSpec, SamplerConfig, prepare, fit,
    forecast_context, simulate_predictive, diagnose, score, render_report,
)

data = SpatioTemporalData(
    y=y,                         # (time, observed_site)
    coordinates=coordinates,     # (observed_site, spatial_dimension)
    times=times,
    site_ids=site_ids,
    response_units="standardized field units",
    coordinate_units="unit square",
)

split = SplitPlan(
    conditioning=conditioning_times,
    fitting=fitting_times,
    validation_blocks=validation_blocks,
    final_blocks=sealed_final_blocks,
)

model = ModelSpec(
    features=FeatureSpec.deep_esn(
        input_modes=16, units=(256, 256),
        input_scales=(0.4, 0.8),
        leaks=((0.2, 0.5, 0.9), (0.05, 0.2, 0.6)),
        score_bound=3.0, squared_states=128, seed=731,
    ),
    spatial_mean=SpatialMeanSpec.matern_basis(
        centers=centers, width=0.18,
    ),
    innovations="spatial_gaussian_iid_time",
    observation_noise="iid_gaussian",
    covariance="matern32",
    initial_state_convention="condition_on_initial_history",
    prior=PriorSpec.group_ridge(
        groups=("intercept", "scores", "layer_1", "layer_2", "squares"),
    ),
)

prepared = prepare(data, split=split, model=model)
posterior = fit(
    prepared,
    sampler=SamplerConfig(
        chains=4, warmup=1500, draws=3000, seed=20261102,
        backend="auto", checkpoint_every=250,
    ),
)

context = forecast_context(
    posterior, origin=forecast_origin,
    parameter_policy="frozen",
)
prediction = simulate_predictive(
    posterior, context=context, horizon=200,
    target_coordinates=target_coordinates,
    target_site_ids=target_site_ids,
    paths=512, seed=9011,
    targets=("conditional_mean", "latent", "observation"),
)

diagnostics = diagnose(posterior, prediction=prediction)
scores = score(prediction, truth=evaluation_truth, horizons=(40, 100, 200))
render_report(posterior, prediction, scores, diagnostics, output=output_path)
~~~

Factory names above are exact proposed interface names; string-valued choices must eventually be backed by validated enumerations. The machine-readable companion contract is **specs/api_contract.yaml**. Public documentation must show the supported choices and their computational consequences.

### Updating history is not updating parameters

Three operations must remain distinct:

1. **simulate_predictive** propagates generated observations within a forecast block. Actual future observations are unavailable to this function.
2. **advance_state** accepts newly available actual history and returns a new ForecastContext with the same parameter posterior. In a latent-state model, each fixed parameter draw receives the corresponding filtering update. This is explicitly a frozen-parameter forecasting policy.
3. **refit** constructs a new parameter posterior using the expanded data. It returns a new FittedModel with a new training-data identity.

The frozen policy is not the posterior obtained by fully updating parameters with intervening observations. Code, report labels, and score tables must preserve that distinction.

Forecast origins cannot precede the training endpoint of their parameter posterior. Ensemble operations are separately declared as **fit_ensemble** and **fit_stacking_weights**, returning a FittedEnsemble whose component posteriors remain identifiable. **simulate_predictive** accepts either a FittedModel or FittedEnsemble and preserves one component per trajectory. **fit_calibration** uses an explicitly identified historical calibration partition to return a CalibrationMap; **apply_calibration** returns interval summaries without mutating posterior draws. Final EvaluationTruth is inaccessible to all fitting, ensemble-weight and calibration operations. Finite Bayesian index sampling remains a distinct inference mode with the common-dimension requirements in its derivation.

For each recursive trajectory, the parameter draw and any reservoir-mixture component remain fixed across all horizons. Point predictions should average conditional future means when this removes zero-mean innovation Monte Carlo noise; the recursive state still receives the generated observation, preserving the predictive distribution of later nonlinear states. For iid spatial innovations the conditional mean equals the ESN readout mean. With persistent latent innovations it includes the conditional expected residual. The outputs **readout_mean**, **conditional_mean**, **latent**, and **observation** therefore have distinct definitions.

## Inference backend routing

The dispatcher should resolve the mathematical structure before allocating large arrays. An explicit backend request that does not support a configuration must fail. The automatic dispatcher may choose among verified implementations, but must return the selected backend, its assumptions, and its numerical approximation status in metadata.

| Backend | Eligible structure | Required checks |
|---|---|---|
| spectral_gaussian | Common spatial covariance; Gaussian innovations; scalar, group, or column-separable readout precision | Positive-definite covariance; correct determinant terms; coefficient and scale recovery |
| weighted_spectral | Whole-field scalar time weights, including the specified Student-$t$ augmentation; compatible separable prior | Recompute weighted cross-products; correct weight and degrees-of-freedom updates |
| temporal_whitened_spectral | Verified full-residual AR construction with a common spatial covariance and separable readout prior | Initial-state convention and temporal determinant; distinguish full residual from latent-only AR |
| latent_ffbs | Latent spatial AR process with iid Gaussian measurement noise and compatible spatial structure | Forward-filter/backward-sample identities; stationary initialization; conditional readout updates |
| gaussian_matrixfree | Entrywise readout scales or supported time-varying Gaussian precision | Operator symmetry and definiteness; residual tolerance; sampling covariance against dense reference |

The first release must use a positive whitelist of model combinations. In particular, local Student-$t$ measurement scales are not interchangeable with whole-field Student-$t$ innovations; a latent AR process is not interchangeable with AR on the combined residual; and arbitrary site-by-time heteroskedasticity is not scalar temporal weighting. Complex combinations enter the whitelist only after their joint sampler has a derivation and an independent numerical reference.

Iterative linear algebra introduces numerical error. A successful function return is insufficient: record iteration counts, convergence codes, and independently computed relative residuals. Validate the distribution of Gaussian draws against dense calculations over a range of condition numbers. Failure of a prescribed tolerance must stop that chain or checkpoint it as failed; it must not silently change the posterior target or return approximate draws labeled exact. SciPy's documented conjugate-gradient interface provides the operator and residual-convergence mechanisms needed for this design.[^scipycg]

## Repository structure and public delivery

The future repository should have a conventional source layout and a single configuration system:

~~~text
bayes-stesn/
  pyproject.toml
  README.md
  LICENSE
  CITATION.cff
  CHANGELOG.md
  src/bayes_stesn/
    data.py
    specifications.py
    preparation.py
    features/
    spatial/
    priors/
    inference/
    prediction/
    diagnostics/
    scoring/
    reporting/
    serialization/
    cli.py
  tests/
    algebra/
    posterior/
    predictive/
    reproducibility/
  docs/
    model-reference/
    derivations/
    api/
    tutorials/
    limitations/
  configs/
    smoke/
    baseline/
    extensions/
    study/
  examples/
  benchmarks/
  report/
  .github/workflows/
~~~

Package metadata and builds should follow the Python Packaging Authority's **pyproject.toml** workflow.[^pypa] The exact dependency versions and supported Python versions must be locked after installation and compatibility checks during implementation. This plan does not invent an environment lock or claim package publication.

The initial public repository should contain the planning documents, derivations, citations, model-status table, and design decisions. Later commits add working code in independently reviewable milestones. The existing **exdqlm** repository remains a separate project unless integration is requested.

Public release contents should include code, small synthetic examples, reproducible configuration files, and compact reference outputs. Full posterior arrays and large movies belong in versioned release assets or an archival data deposit, referenced by checksums. A release must distinguish included artifacts, regenerable artifacts, and any data subject to separate distribution terms. No account credentials, local absolute paths, user-specific environment files, or private source documents belong in the repository.

## Reproducibility contract

Every fit and prediction must retain:

- source commit, repository URL when published, package version, schema version, and source-tree hash when uncommitted changes are deliberately allowed;
- normalized model/configuration serialization and hashes of the input arrays and ordered identifiers;
- preprocessing windows, observed-site identities, feature scaling constants, spatial centers, PCA loadings, reservoir matrices, and feature-group assignments;
- master seed and stable identifiers for data generation, feature generation, each chain, posterior recovery, each prediction path, spatial innovations, and nugget innovations;
- environment lock, numerical-library versions, BLAS implementation, operating system, thread settings, and numerical tolerances;
- posterior warmup policy, retained iteration ranges, failures, adaptation state, diagnostics, and wall-clock/resource measurements;
- target units, forecast origins, parameter-update policy, prediction path count, and ensemble-weight provenance.

Use NumPy SeedSequence or an equivalent documented mechanism to derive reproducible streams.[^numpy] Derive named child streams from stable integer identifiers or a specified cryptographic digest, never Python's process-dependent hash or task completion order. Parallel scheduling must not change which random stream belongs to a chain or trajectory. Byte-for-byte reproduction is a target for the same locked environment; cross-platform reproducibility is numerical agreement within declared tolerances.

Checkpoints contain JSON metadata and safe numeric arrays, with no pickle-dependent deserialization. Write to a temporary path, validate, then atomically replace the checkpoint. A restart requires matching model, data, source/schema compatibility, dimensions, RNG state, and iteration counters. Changes to the model cannot silently resume an old chain.

Do not store all $113{,}004$ readout coefficients at every iteration by default. For collapsed samplers, retain hyperparameter chains and recover coefficients in bounded batches for requested posterior summaries and forecasts. Cache small sufficient statistics and the frozen feature map. Latent-state samplers need explicit storage policies: current latent state for continuation, selected retained latent summaries or paths for diagnostics, and enough information to regenerate future predictions.

The intended implementation commands are:

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/locked-runtime.txt
python -m pip install --no-deps -e .
bstesn validate configs/baseline/v3_compatibility.yaml
bstesn run configs/smoke/gaussian.yaml --output runs/smoke
bstesn run configs/study/registered_comparison.yaml --output runs/study
bstesn report runs/study --output report/results
~~~

These commands become runnable only when the listed files and entry point exist. Reproduction of the documentation must be separate from fitting and prediction, so rebuilding a PDF cannot silently rerun a costly simulation.

## Verification strategy

### Verification completed for this blueprint

The accompanying standalone audit performed **39 deterministic identity checks across seven mathematical families** on tiny dense systems. All passed with absolute tolerance $10^{-10}$ and relative tolerance $10^{-9}$; the largest observed absolute difference was $9.68\times10^{-13}$. The checked families are normalized Gaussian/scale integration, grouped spectral recovery, complete-residual AR whitening, latent-AR smoothing moments, nonseparable precision perturbations, reference expected-variance allocation, and Vecchia inverse/determinant identities.

The executable audit and individual measurements are in `audit/verify_blueprint_algebra.py` and `audit/algebra_checks.json`. They use Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0 in the observed environment. These checks support the algebra; they do not validate a complete new MCMC implementation, its convergence, large-scale runtime, or forecast performance. The remaining verification below is an implementation requirement.

### Algebra and probability checks

For tiny dimensions, explicitly construct the full design matrix and covariance, and compare the structured implementation with independent dense Gaussian calculations. Required comparisons include normalized marginal likelihoods, posterior means, posterior covariance actions, conditional scale shapes, and predictive joint covariance. Check invariance to consistent permutations of sites and features, not only agreement of a single example.

For each new prior, verify the actual density in the sampled coordinates, including logarithmic or logistic Jacobians. For each sampler, verify that the sequence of collapsed, augmented, and conditional steps leaves the intended joint posterior invariant. The recovery of coefficients after a collapsed update must not use stale conditioning variables.

For stochastic tests, compare Monte Carlo discrepancies with their sampling uncertainty. Fixed tolerances on a few random draws are unsuitable evidence of a correct sampler. Test a range of weakly and strongly identified designs, collinear features, and spatial ranges close to the numerical limits. Positive-semidefinite roundoff handling must not mask a materially indefinite model covariance.

### Simulation-based calibration

SBC should condition on a fixed, valid feature-generating map and initial conditioning history. Draw parameters from the specified prior, generate a complete response trajectory recursively, recompute its reservoir states from that generated past, and fit using the same conditional model. Holding the realized training design $Z$ fixed while simulating responses does not test the response-driven ESN model.

For priors depending on a reference design, freeze that conditioning-only reference and use the same conditional prior in generation and inference. Include ranks for interpretable covariance/scale parameters, selected linear projections of $C$, and predictive functionals. Do not treat every coefficient rank or time point as an independent replication. SBC checks computation under the assumed model; misspecification experiments answer a separate predictive question.[^sbc]

### Diagnostics and acceptance gates

Use at least four independently initialized chains for the benchmark fits. Initial operational gates are rank-normalized $\widehat R<1.01$ and bulk/tail ESS of at least 400 for primary scalar parameters and prespecified predictive summaries. These thresholds are screening criteria, not proof of convergence. Require Monte Carlo standard errors small relative to reported scientific differences, inspect chain agreement and tails, and examine relevant parameter couplings.[^rhat]

When the number of local scales is large, also inspect scale-group summaries and selected extreme scales. Compare predicted field averages, maxima, and long-horizon variability across chains; satisfactory diagnostics on only three covariance parameters cannot validate every augmented state.

Limit cases must recover the simpler models: equal group precisions recover ridge; isotropy parameters zero recover isotropic Matérn; heteroskedastic coefficients zero recover constant variance; AR persistence zero recovers independent latent residual fields under the matching initialization; Student-$t$ degrees of freedom increasing recover the Gaussian limit; and an ensemble with one component reproduces that component.

### Prediction and leakage checks

One-step fitting, one-step updating with newly observed history, fixed-origin recursive forecasting, spatial interpolation, and prediction in an unobserved region are different tasks. Each receives a distinct evaluation label and test fixture.

Assert that changing final evaluation responses cannot change fitted parameters, selected architecture, reservoir transforms, or forecast arrays. The recursive predictor must not accept evaluation truth. Changing the scheduling or prediction batch size should preserve the assigned parameter/path streams. For innovations with spatial correlation, simulate the requested field jointly; independently drawing locations is a correctness failure.

## Controlled study and resource gates

Preserve the current model as **M0**. Compare **M1** grouped Gaussian shrinkage against M0 with the same data, features, covariance, and seed. Compare **M1 + E1**, a small prespecified reservoir ensemble, with M1 to isolate seed averaging. The identifier **M2** remains reserved for the regularized-horseshoe family in the extension register. Then add spatial-input or covariance extensions one at a time when the development diagnostics motivate them. Maintain a registry of hypotheses, changed components, prior calibration, candidate counts, selection criteria, and stopping rules.

The expanded study should include independent innovation realizations and varied initial phase/dynamic regimes. In the existing DGP, changing only the random seed does not necessarily vary its deterministic phase evolution. Add prespecified cases with anisotropy, latent residual persistence, heteroskedasticity, and heavy-tailed contamination, alongside the original matched Gaussian case. Distinguish changes in predictability from changes in sample size.

Use common prespecified splits and forecast origins across competing methods. Keep the primary 100-step score and secondary 40/200-step scores distinct. For novel covariance comparisons, use both marginal scores and joint-field scores. Compute uncertainty in paired performance differences across independent DGP realizations or appropriately separated forecast blocks, not by treating all locations and horizons as independent observations. Repeated tuning against the existing final test block is prohibited; reserve new final data for the next study.

Start each backend with a tiny reference problem and a representative-size pilot. Measure runtime per effective draw, memory peaks, factorization costs, and prediction throughput before scheduling the full study. Set user-visible compute and storage budgets from those measurements. Do not promise benchmark runtimes in advance.

Parallelize independent chains or models subject to a memory budget; cap BLAS threads within each worker to prevent oversubscription. If the estimated peak allocation exceeds 70% of available memory, reduce concurrency or batch size before execution. A pilot that fails numerical, convergence, or resource gates must be investigated before expanding the run. Save all completed milestones so an interrupted study resumes from validated artifacts.

## Milestones and release gates

| Milestone | Depends on | Deliverable | Acceptance condition |
|---|---|---|---|
| P0: specification | None | Public-ready documentation, derivations, API contract, status ledger | Independent mathematical review; no unresolved ambiguity in supported models |
| P1: baseline package | P0 | Migrated V3 core and explicit API | Dense-reference checks and reproduction of stored V3 metrics within declared tolerance |
| P2: structured priors | P1 | Group ridge, column/group RHS, separable R-squared allocation and correction-space shrinkage | Correct transformed targets; posterior checks; predictive ablation |
| P3: covariance extensions | P1 | Anisotropy, static variance, selected smoothness and PC options | Positive-definite construction; exact reduction tests; consistent scale conventions |
| P4: augmented dynamics | P1; P3 for combined variants | Complete-residual AR, latent spatial AR and whole-field robustness | Verified temporal whitening/mixture updates/FFBS; memory and mixing gates |
| P5: advanced priors | P2 and validated precision operator | Entrywise RHS/GIGG and general function-penalty variants | Matrix-free draw validation and configuration-specific performance gate |
| P6: representation/ensemble | P1; P2 for designated comparison | Local or graph inputs, prespecified reservoir mixtures | Training-only representation fitting; coherent path mixtures; validation-only weights |
| P7: controlled study | All participating P1--P6 implementations | Locked comparisons over replicated scenarios | All sampling gates satisfied; honest final evaluation; paired uncertainty |
| P8: release/report | P7 | Versioned package, complete report, figures, animations, archival assets | Fresh-environment smoke reproduction; artifact and citation checks |

Stages P3--P6 need not activate every possible combination. Each release records implemented, experimental, unsupported, and planned configurations. The final implementation is complete when all approved model families have verified samplers and documented limitations, not when an unrestricted combinatorial menu accepts every flag.

The extension registry maps every recommendation to these milestones. P3 includes low-dimensional nonstationarity and separate-scale PC variants. P5 includes local Student-t noise and general site-time heteroskedasticity after the general precision backend is validated. P6 includes frozen learned encoders, finite-index research inference, and predictive calibration. Scalable Vecchia/SPDE backends belong to P3 with a separate resource-triggered research gate; they are implemented and checked at feasible sizes before being used in a full study.

## Final report and visual specification

The eventual results report should be a typeset PDF with a companion navigable HTML report and downloadable numerical tables. Its order should make scientific conclusions easy to assess:

1. Executive findings and a compact model comparison.
2. Data-generating mechanisms, information available to each model, and split diagrams.
3. Model hierarchy, differences from M0, and prior predictive behavior.
4. Computational validation, convergence, and resource costs.
5. Forecast accuracy and calibration, separated by target and horizon.
6. Joint spatial/temporal dependence and aggregate-function uncertainty.
7. Ablations, sensitivity, failure cases, and limits.
8. Reproduction instructions and mathematical appendices.

Use consistent typography, restrained color, vector plots in the PDF, and high-resolution raster exports when necessary. Parameter and error figures need original units, common limits for comparable panels, clear reference levels, and readable labels. Use perceptually ordered sequential colors for magnitudes and a symmetric diverging scale for signed errors. Report posterior means separately from representative trajectories, and pointwise intervals separately from simultaneous bands.

The animation set should include the generating mean, latent field, noisy observations, forecasts, and signed errors with explicit target labels. Provide one synchronized comparison animation for each prespecified representative origin, plus an errors-only view with RMSE and MAE over time. Color limits and error axes remain fixed within a comparison. Show the full forecast window and annotate the information policy. Select representative origins by a prespecified rule rather than visual appeal.

Export MP4 for quality and efficient playback, GIF previews for compatibility, and static first/middle/final frames. Include a timestamp, forecast lead, model label, color bars, and error sign convention. Verify frame counts, duration, decoding, axis consistency, and agreement with saved numerical scores. The report generator reads immutable result artifacts; it must not choose models, recalibrate intervals, or change metric definitions during rendering.

[^numpy]: NumPy documentation, *Parallel random number generation*. https://numpy.org/doc/stable/reference/random/parallel.html. Consulted November 2, 2026.

[^scipycg]: SciPy documentation, *scipy.sparse.linalg.cg*. https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html. Consulted November 2, 2026.

[^pypa]: Python Packaging Authority, *Packaging Python Projects*. https://packaging.python.org/en/latest/tutorials/packaging-projects/. Consulted November 2, 2026.

[^sbc]: Talts, S., Betancourt, M., Simpson, D., Vehtari, A., and Gelman, A. (2018). *Validating Bayesian Inference Algorithms with Simulation-Based Calibration*. https://arxiv.org/abs/1804.06788.

[^rhat]: Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., and Bürkner, P.-C. (2021). *Rank-Normalization, Folding, and Localization: An Improved $\widehat R$ for Assessing Convergence of MCMC*. Bayesian Analysis, 16(2), 667--718. https://doi.org/10.1214/20-BA1221.
