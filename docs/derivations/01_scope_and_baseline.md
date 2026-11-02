# Purpose, scope, and implementation decisions

**Document status:** implementation blueprint, version 1.0, dated 2026-11-02. The proposed extensions have not been fitted. The accompanying API examples are contracts for the next implementation stage. The existing V3 study is the reference baseline, not evidence that the proposed extensions improve forecasting.

The objective is a public, documented, reproducible package for Bayesian nonlinear spatiotemporal forecasting with fixed echo-state features. The package must keep the likelihood, prior, computational method, and evaluation target explicit. A successful implementation supports several independently testable model families; it does not silently combine every option into one maximal model.

## Deliverables and publication target

The proposed repository is **AntonioAPDL/bayes-stesn**, with Python import name **bayes_stesn**. The authenticated account was verified; the repository name is a proposal, and a remote repository has not been created in this planning stage. Name availability must be checked at creation. A public documentation-first repository is the publication target for this blueprint, followed by tagged research releases during implementation. The connected GitHub toolset exposes operations on existing repositories but does not expose repository creation; the proposed remote returned not found in the planning-stage check. The publication step therefore remains pending, and no live public URL is claimed. No existing research repository will be repurposed.

The first public implementation release must contain:

- An installable, typed Python package and command-line interface.
- Mathematical documentation matching the actual parameterization and sampler.
- Runnable examples for complete panels at arbitrary planar locations.
- Versioned YAML configurations, a dependency lock, deterministic simulation generators, and immutable run manifests.
- Meaningful numerical and statistical tests, including a small dense reference implementation.
- Baseline reproduction, matched ablations, and a report generated directly from saved draws and scores.
- A permissive code license, citation metadata, source attribution, and separate licenses for external data.
- A documented public release procedure and checksums for large downloadable artifacts.

Python is the first implementation language because the audited V3 computation already uses NumPy and SciPy. An R-facing interface is a later adapter to the same engine; it must not duplicate a second inference implementation. C++ or compiled kernels are introduced only after profiling identifies an actual bottleneck.

## What is preserved from V3

V3 has a fixed causal two-layer reservoir and a spatial radial mean dictionary. It fits 256 input sites at 7,600 times after 600 conditioning times. There are 172 spatial features and 657 temporal features, giving 113,004 regression coefficients. Its three numerical MCMC parameters are log ridge precision, log spatial-to-nugget variance ratio, and log correlation length. Coefficients and the common variance are integrated analytically and recovered for prediction.

The exact model, on the response scale standardized from conditioning data, is
$$
\begin{aligned}
y_t(s)&=\phi(s)^\top C z_t+w_t(s)+\epsilon_t(s),\\
w_t&\sim N_n(0,\sigma^2qK_\ell),\qquad
\epsilon_t\sim N_n(0,\sigma^2I_n),\\
C_{jk}\mid\sigma^2,\lambda&\sim N(0,\sigma^2/\lambda),\\
\sigma^2&\sim\operatorname{IG}(2,0.05).
\end{aligned}
$$
Spatial innovations and nuggets are independent across time and mutually independent conditional on parameters. The independent log priors are
$$
\log\lambda\sim N(\log0.1,1.5^2),\quad
\log q\sim N(0,1.25^2),\quad
\log\ell\sim N(\log0.18,0.65^2).
$$

The V3 spatial dictionary contains an intercept, the two coordinates, and 169 Matérn-3/2 radial functions on a 13 by 13 lattice, with fixed mean-basis width 0.18. Nonintercept columns are centered and scaled over observed geometry and divided by the square root of their count. This fixed basis width is distinct from the inferred residual correlation length.

The temporal input is 16 standardized principal-component scores,
$$u_t=P(y_t-a).$$
The two 256-unit reservoirs satisfy
$$
\begin{aligned}
r_t^{(1)}
&=(I-D_1)r_{t-1}^{(1)}
+D_1\tanh(W_1r_{t-1}^{(1)}+U_1u_{t-1}+b_1),\\
r_t^{(2)}
&=(I-D_2)r_{t-1}^{(2)}
+D_2\tanh\{W_2r_{t-1}^{(2)}
+U_2[(r_t^{(1)}-a_r)\oslash s_r]+b_2\}.
\end{aligned}
$$
The raw readout is
$$
v_t=\left[
3\tanh(u_{t-1}/3)^\top,\,
(r_t^{(1)})^\top,\,
(r_t^{(2)})^\top,\,
(r_{t,j}^2)_{j\in\mathcal Q}^\top
\right]^\top,\quad|\mathcal Q|=128,
$$
and
$$z_t=\left[1,\{(v_{t,j}-\bar v_j)/(s_{v,j}\sqrt{656})\}_{j=1}^{656}\right]^\top.$$
The full fixed settings are in the baseline provenance and configuration files. They include seed 731, recurrent operator norm 0.9, connectivity 0.08, input scales 0.4 and 0.8, and leak patterns (0.2,0.5,0.9) and (0.05,0.2,0.6).

The likelihood is the product of causal transition densities given the conditioning history. Its Gaussian regression algebra does not imply an unconditional jointly Gaussian law for the response sequence. All transforms, reservoir matrices, architecture choices, and reference-design quantities are fixed before each evaluation block.

## Extension register

Every recommendation is assigned a concrete implementation or evaluation track. These are alternatives and combinations with explicit compatibility rules.

| ID | Family | Deliverable and inference |
|:--|:--|:--|
| M0 | Current ridge baseline | Exact spectral collapse and posterior recovery; reproduction gate |
| M1 | Group Gaussian shrinkage | Five temporal groups; exact whitened spectral calculation |
| M2 | Column/group regularized horseshoe | Noise-scaled finite-slab prior; log-scale slice/MH; separable coefficient calculation |
| M3 | Entrywise RHS | Conditional Gaussian, matrix-free precision sampling; no falsely cheap determinant |
| M4 | GIGG | Group and local normal mixtures; GIG/IG updates; generic coefficient sampler |
| M5 | Reference-design R-squared allocation | Precisely defined expected-variance calibration; Dirichlet or logistic-normal allocations |
| M6 | Subspace/functional shrinkage | Proper Gaussian prior penalizing departure from a prespecified simpler mean; clear distinction from published fHS theorems |
| C1 | Anisotropic Matérn | Valid determinant-one spatial metric; two extra parameters |
| C2 | Spatially varying nugget variance | Common spatial covariance across times; centered variance covariates |
| C3 | Discrete Matérn smoothness | Finite prior over smoothness values; exact conditional probabilities |
| C4 | Nonstationary covariance | Valid spatially varying local geometry and amplitude; shrinkage toward stationarity |
| C5 | PC covariance priors | Range-only conjugate route and separate-scale nonconjugate route |
| D1 | Full-residual AR | Separately named model; separable temporal whitening |
| D2 | Latent spatial AR plus iid nugget | Spatial-mode FFBS and ordinary Gibbs/MH |
| L1 | Whole-field Student-t innovations | One mixing precision per time; weighted spectral regression |
| L2 | Local Student-t measurement errors | One mixing precision per observed site-time; latent fields and generic weighted regression |
| L3 | General site-time heteroskedasticity | Time-varying covariance operators; conditional Gaussian sampler with resource gates |
| F1 | Spatially informed inputs | PCA expansion, local/graph filters, fixed training-only encoders |
| F2 | Learned spatial encoder | Separately trained, frozen encoder; conditional uncertainty clearly labeled |
| E1 | Reservoir ensemble | Equal weights or validation-based stacking of joint trajectories |
| E2 | Bayesian reservoir index | Common-dimensional collapsed index sampling, or validated model evidences |
| U1 | Predictive calibration | Separate conformal/weighted calibration; no invented MCMC block or coverage guarantee |
| S1 | Scalable spatial backend | Vecchia or SPDE as explicitly approximate/model-changing computation, activated by measured need |

Sections below derive M0--M6, C1--C5, D1--D2, and L1--L3. Feature maps, ensembles, calibration, and scalable spatial backends have separate inference definitions. The fact that a choice has no additional MCMC parameter is documented rather than obscured by an artificial sampler.

## Execution order and acceptance criteria

The P0--P8 milestone table in the engineering chapter is the authoritative delivery ledger. The stages here explain the scientific order; several independent tracks may proceed in parallel.

### Stage 0: preserve and reproduce the baseline

Preserve the verified V3 source hashes, model settings and seeds before migration. Audit fixed covariance posterior quantities and tiny-case marginal likelihoods. After the common contracts in Stage 1 are established, port these equations into the package without changing the statistical model. The acceptance gate is numerical agreement with the frozen reference calculation at fixed hyperparameters and statistically compatible posterior summaries under the same specification. Matching a stochastic run bit-for-bit is only expected in the same locked numerical environment.

### Stage 1: establish common contracts

Implement immutable data, feature-map, model, prior, sampler, and prediction specifications. Add shape, coordinate-unit, time-order, and causal-history checks. Separate fitting, recursive forecasting, state-only updating, full refitting, and spatial reconstruction. Reject unsupported combinations by name before computation starts.

### Stage 2: implement the exact common-covariance families

Deliver M1, C1--C3, and the range-only part of C5 first. Add M2 and M5 on the same conditional Gaussian backend. Each family must reduce to its baseline limit numerically and reproduce a direct dense posterior on tiny problems.

### Stage 3: implement conditional latent/weighted families

Deliver D1, D2, and L1. Add C2 combinations that preserve the required diagonalization or whitening. Introduce M3/M4/L2/L3 only after the generic precision-operator backend passes small-case distributional tests and solve-error diagnostics. C4 is a separate covariance module with fixed small basis dimension and explicit priors. M6 is a separate research prior with a proper positive-definite precision.

### Stage 4: implement representation and ensemble tracks

Add PCA-dimension and graph/local-input ablations before a learned encoder. Fit reservoir seeds independently with a fixed architecture. Compare equal-weight mixtures first. Estimate stacking weights from validation origins only. A Bayesian reservoir-index option is a separate inferential target. It requires either validated evidences or the derived common-dimensional joint index sampler using fully normalized integrated likelihoods.

### Stage 5: run a staged simulation study

Screen computational feasibility and failure modes on tiny and medium problems. Freeze the candidate set, priors, forecast policy, scenarios, evaluation seeds, and scoring rules before final evaluation. Compare the baseline and a small number of justified candidates under identical data budgets. Expand to advanced alternatives only when the earlier results identify a relevant deficiency.

### Stage 6: generate and audit the results report

Render tables, uncertainty summaries, animations, and text from machine-readable artifacts. Every displayed improvement must have a matched comparison, uncertainty summary across independent replications, computational cost, and stated target. Keep training fit, spatial reconstruction, future latent fields, future observations, and predictable DGP mean distinct.

### Stage 7: publish a reproducible public release

Publish the documentation as soon as the new public repository is available, before model implementation. After required implementation checks pass, publish the source and tag a reproducible release. Publish large results as release assets or a research archive rather than committing arbitrary simulation caches. Record the git commit, configuration and data hashes, environment lock, seeds, numerical backend, and release checksums. Provide a one-command reproduction entry point for each study profile.

## Scientific decisions that must remain explicit

1. The default posterior conditions on the fitted feature map. A frozen autoencoder does not create posterior uncertainty over its weights.
2. Residual temporal independence is a modeling assumption. Full-residual AR and latent-spatial AR are different models.
3. A Gaussian conditional posterior does not guarantee the same fast spectral sampler.
4. A finite iterative-solver tolerance is a numerical approximation. The package reports it and validates its statistical effect.
5. Conformal calibration and stacking are predictive procedures, not extra posterior parameters unless a separate probabilistic model is specified.
6. The innovation variance, latent-field variance, and observation variance must remain distinct, especially under Student-t and autoregressive alternatives.
7. No architecture, prior, calibration parameter, or ensemble weight may be selected using final evaluation outcomes.
8. The current synthetic DGP already matches the fitted residual Matérn family. Covariance extensions must also be evaluated under prespecified misspecification.
9. Three origins from one trajectory are not three independent simulation replications.
10. An extension is considered beneficial only relative to a stated goal: mean accuracy, tail calibration, spatial dependence, robustness, or computation.

## Position relative to the literature

The foundational ESN and Bayesian DESN references are McDermott and Wikle (2017, 2019). Regularized-horseshoe reservoir readouts and reservoir averaging have precedent in Grieshop and Wikle (2024). Recent work by Wang, Holan and Wikle (2025), Wang and colleagues (2026), and Bonas and colleagues (2026) supports investigating spatial inputs and spatial reconstruction. The present plan is a specific computational and software synthesis. It does not assert methodological novelty from combining these components.

The references file separates established constructions from the explicit adaptations proposed here. In particular, the reference-design R-squared prior and the proper subspace penalty below must not be advertised as unchanged implementations of GDR2 or the functional horseshoe.

