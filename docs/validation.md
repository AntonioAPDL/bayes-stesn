# Validation evidence and limits

The release distinguishes implementation verification, statistical calibration, MCMC diagnostics and comparative forecasting evidence. Each answers a different question. A working sampler or a passing unit test does not establish that an extension improves forecasting on a new dataset.

The exact release test count, commands, environment, result and artifact checksums are recorded in [`results/release_manifest.json`](../results/release_manifest.json). The manifest is authoritative for the released revision. This page records the substantial numerical/statistical checks and their measured results.

## 1. Reproduce the checks

From an installed checkout:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests
python benchmarks/validate_statistics.py --output results
python benchmarks/validate_spatial_approximations.py --output results
python benchmarks/verify_v3_parity.py --legacy-root ../spatiotemporal_esn_v3 --output results/validation/v3_full_parity.json
```

The original-artifact parity command requires the historical V3 data/features/source checkout. The repository also includes a compact independent reference test runnable without those external artifacts. The full parity JSON records the SHA-256 hash of every source artifact used. Recomputing statistical validation uses the declared seed without rerolling an inconvenient outcome.

## 2. Independent algebra and distribution checks

The tests compare optimized implementations with independently constructed small dense problems:

| Component | Independent check |
|---|---|
| Gaussian regression | Full observation-space Gaussian covariance and explicit Kronecker design; normalized marginal likelihood, posterior mean/covariance, common-scale recovery |
| General coefficient precision | Explicit dense precision against matrix-free action and the checked precision-noise perturbation draw |
| Full-residual AR | Dense temporal covariance $S_\rho\otimes R$, stationary initialization and determinant against whitened calculations |
| Readout priors | Transformed half-Cauchy/IG/Beta/Dirichlet densities, finite-slab RHS variance, GIG scaling, reference energy, proper function-penalty determinant |
| Spatial kernels | Isotropic/anisotropic limits, nonstationary positive definiteness and exact diagonal amplitude, frozen-reference target extension |
| Vecchia | All-predecessor exact recovery, truncated actual diagonals, fixed-order appended-target marginal preservation including distance ties |
| FEM SPDE | Triangle mass/stiffness integrals, affine interpolation, positive proper precision, Woodbury solve/log determinant, explicit boundary and smoothness scope |
| Latent AR | Full dense Gaussian smoother mean/covariance and Monte Carlo draws against eigenmode FFBS; stationary state density and $\rho=0$ limit |
| Student-t variants | SciPy multivariate Student-t likelihood, correct Gamma conditionals, Gaussian limit, local spatial-field conditional moments |
| Site-time heteroskedasticity | Sum of Kronecker products against a complete block-diagonal observation-space design |
| Reproducibility | Data/spec mismatch rejection; exact interrupted/continued chains including latent fields, weights, thinning and adaptation state |
| Forecasting | Frozen parameters, causal observation feedback, new-site joint paths, duplicate-site identity, state updating and serialization |
| Evaluation | Independent CRPS/energy-score expressions, rank diagnostic references, whole-path mixture sampling, held-out stacking, exact conformal order statistics |

The 29 spatial/dynamics tests passed in approximately 1.6 seconds under one BLAS thread during component verification. They include 5,000 FFBS and 4,000 local-field draws checked against independently derived conditional distributions. Short augmented chains establish functioning routes and reproducibility; they are not presented as evidence of convergence of those research models.

## 3. Original V3 parity at full dimensions

[`v3_full_parity.json`](../results/validation/v3_full_parity.json) reports a successful fixed-hyperparameter comparison using all original input rows, times and features:

| Quantity | Value |
|---|---:|
| Input sites $n$ | 256 |
| Likelihood times $T$ | 7,600 |
| Spatial features $p$ | 172 |
| Temporal features $d$ | 657 |
| Readout coefficients $pd$ | 113,004 |
| Observations $nT$ | 1,945,600 |

The fixed coordinates were $(\log\lambda,\log q,\log\ell)=(-5.4235934407,\,0.6062029351,\,-2.4645438365)$. This check evaluates the same model and frozen features; it does not rerun the original MCMC or claim a new full-dimensional forecast comparison.

| Check | Observed difference | Declared tolerance |
|---|---:|---:|
| Frozen feature endpoint, maximum absolute | 0 | $10^{-11}$ |
| Spatial $R$, maximum absolute | $4.44\times10^{-16}$ | $10^{-12}$ |
| Normalized log likelihood, absolute | $1.12\times10^{-8}$ | $10^{-5}$ |
| Marginal quadratic $Q$, relative | $1.08\times10^{-14}$ | $10^{-10}$ |
| Posterior IG scale, relative | $1.08\times10^{-14}$ | $10^{-10}$ |
| Coefficient mean, relative Frobenius | $2.48\times10^{-12}$ | $10^{-8}$ |
| Coefficient mean, maximum absolute | $2.85\times10^{-10}$ | $10^{-7}$ |
| Nine field covariance functionals, relative Frobenius | $1.01\times10^{-14}$ | $10^{-9}$ |
| Posterior mean of $\sigma^2$, relative | $1.09\times10^{-14}$ | $10^{-10}$ |

The recorded run took about 1.75 seconds, with peak resident memory approximately 236 MiB. The spectral computation avoided a coefficient precision array of 102,159,232,128 bytes. These timings describe this fixed-parameter parity run and environment, not the time required for a full MCMC study.

## 4. Conditional conjugate simulation-based calibration

[`benchmarks/validate_statistics.py`](../benchmarks/validate_statistics.py) performed **200 independent prior-predictive replications**, retaining **255 independent joint posterior draws per replication**. The design, row covariance and coefficient precision were fixed. Each replication drew

$$
\sigma^2\sim\operatorname{IG}(3,.9),\qquad
\operatorname{vec}(C)\mid\sigma^2\sim N\{0,\sigma^2(D\otimes I_2)^{-1}\},
\qquad D=\operatorname{diag}(1.5,3.0,2.2),
$$

then generated $Y=\Phi CZ^\top+E$ for $n=5,T=8,p=2,d=3$, and recovered joint posterior draws through the public spectral implementation. The rank is the count of posterior draws below the prior-generated truth. A fresh future observation was generated independently conditional on the true parameters; its predictive draws each used the same sampled $C,\sigma^2$ pair.

| Statistic | Exact 90% interval coverage | Wilson 95% interval for coverage | Randomized-rank KS $p$ |
|---|---:|---:|---:|
| $\sigma^2$ | 87.0% | 81.6–91.0% | .236 |
| $C_{00}$ | 90.5% | 85.6–93.8% | .262 |
| $C_{11}$ | 91.5% | 86.8–94.6% | .752 |
| Future readout mean | 90.5% | 85.6–93.8% | .921 |
| Fresh future observation | 89.5% | 84.5–93.0% | .092 |

Coverage Monte Carlo standard error at the nominal rate is 2.12 percentage points. Every reported Wilson interval contains 90%. None of the five KS rank tests crossed the prespecified Bonferroni threshold of .002, corresponding to a 1% familywise diagnostic level. This is a failure to detect a large calibration discrepancy at this experiment's resolution; non-significance is not proof of correctness.

The independent full-design posterior calculation agreed with the public implementation to $6.66\times10^{-16}$ in coefficient means and $3.55\times10^{-14}$ in the posterior IG scale. The recorded experiment took approximately 0.83 seconds with one BLAS thread.

Complete ranks, true values, posterior statistic draws, exact marginal intervals, design matrices and training panels are in [`statistical_validation.npz`](../results/statistical_validation.npz). The readable result, prior settings, seed and environment are in [`statistical_validation.json`](../results/statistical_validation.json).

**Scope:** this is conditional conjugate Gaussian SBC. It validates joint coefficient/common-scale recovery under a fixed design. It does not validate recursive ESN-generated predictors, covariance/shrinkage hyperparameter MCMC, learned encoder uncertainty, or the full latent/robust model family. Those require separately designed prior-predictive experiments and adequate chain diagnostics.

## 5. Spatial approximation fidelity

[`validate_spatial_approximations.py`](../benchmarks/validate_spatial_approximations.py) evaluates 36 fixed grid sites with $\ell=.35$ and $q=.8$. Fidelity is measured against the infinite-domain Matérn with **matching smoothness**: $\nu=1.5$ for Vecchia, $\nu=1$ for the integer-order FEM backend. Every raw marginal variance is preserved; no diagonal renormalization is performed.

| Vecchia predecessors | Relative Frobenius error in $K$ | Mean actual diagonal | Maximum diagonal discrepancy |
|---|---:|---:|---:|
| 4 | .26372 | 1.00750 | .03279 |
| 8 | .07086 | .99677 | .01596 |
| 16 | .00764 | .99986 | .00095 |
| 35, all predecessors | $2.45\times10^{-15}$ | 1.00000 | $3.33\times10^{-15}$ |

| FEM resolution per axis | Padding per side | Relative Frobenius error in $K$ | Mean actual diagonal | Maximum diagonal discrepancy |
|---|---:|---:|---:|---:|
| 6 | .5 | .19649 | .73009 | .38496 |
| 10 | .5 | .06333 | .98871 | .14671 |
| 18 | .5 | .05982 | 1.01039 | .10995 |
| 26 | .5 | .05693 | 1.01859 | .09086 |
| 26 | 1.0 | .03525 | .97021 | .14569 |
| 25 | .5 | .06328 | 1.02857 | .15646 |
| 37 | 1.0 | .01918 | .98380 | .09344 |

The last two FEM rows have the same grid spacing $1/12$. Their difference illustrates the importance of a larger boundary domain, while Delaunay interpolation/triangulation still contributes. At a fixed resolution, increasing padding also coarsens the mesh, so that comparison does not isolate boundary error. Mesh refinement need not improve every local marginal variance monotonically, and a refined bounded-domain SPDE remains different from an infinite-domain Matérn.

Separately, sparse solves/log determinants were compared with each approximation's **own** dense covariance. The maximum relative solve error was $3.30\times10^{-15}$ and the maximum absolute log-determinant error $5.22\times10^{-13}$. The complete study took approximately .48 seconds with one BLAS thread. This verifies fidelity choices and numerical operators at fixed parameters; it is not a large-site MCMC scaling result.

The machine-readable results include every actual diagonal, mean/max discrepancy and per-case timing: [`spatial_approximation_validation.json`](../results/spatial_approximation_validation.json). Full reference/approximate covariance arrays are in [`spatial_approximation_validation.npz`](../results/spatial_approximation_validation.npz).

## 6. MCMC diagnostic policy

Diagnostics include rank-normalized and folded split $\widehat R$, bulk/tail ESS and Monte Carlo standard errors. Default screening requires four chains, $\widehat R\le1.01$ and minimum ESS 400. Constant, short or nonfinite traces cannot demonstrate convergence. High-level diagnostics include bounded, fixed readout functionals as well as scalar hyperparameters; the saved diagnostic report identifies its quantities and thresholds.

A passing screen does not prove convergence or forecast calibration. A failed screen remains visible in the result ledger and prevents that run being described as a validated comparison. Warmup adaptation is frozen before retained sampling. Solver diagnostics are separate from MCMC diagnostics: a linear-system residual target checks numerical error, not posterior exploration.

## 7. Independent implementation review

Cross-review examined the parts most likely to produce plausible but incorrect posteriors:

- The finite-slab RHS conditional uses the actual regularized variance and transformed half-Cauchy/IG priors.
- GIGG group/local/global/auxiliary conditionals use a declared GIG convention and the noise-scaled coefficient hierarchy.
- M5 recomputes covariance-dependent coefficient precision for proposed $R$ and retains its determinant/quadratic. Its Beta/logit and Dirichlet/ALR Jacobians are present.
- M6 includes the complete $\log|\lambda_0I+\tau^{-2}F_0|$ contribution rather than a substitute scalar shrinkage determinant.
- Separate physical scales integrate only coefficients in the collapsed route and never reuse an IG common-scale update.
- Augmented schedules retain the correct $N+k$ versus $2N+k$ scale count, stationary latent initialization and full conditional state dependence.
- Forecast paths use the correct latent versus complete-residual AR recursions, future mixture weights, variance transforms and joint new-location conditionals.

Review found and corrected a nonsymmetric logistic-normal covariance validation gap, an inconsistent matrix-free backend alias, and inadequate exclusion of frozen variance-feature normalization constants from parameter diagnostics. Explicit backend compatibility guards and exact augmented checkpoint continuation were added. Coarse SPDE meshes with rank-deficient observation interpolation now fail clearly before latent-AR/local-t sampling, since those observed-site samplers require a nonsingular latent covariance. These findings were resolved in code rather than recorded as acceptable approximations.

## 8. Remaining research limits

The nonlinear forecasting study is a separate, configuration-driven experiment. Its generated report and saved score tables contain the actual completed replications, diagnostic failures, model selection, final evaluation and paired uncertainty. Conclusions about additional data, reservoir width or structured shrinkage must come from those saved comparisons; the algebra and SBC tables above are not forecast-improvement evidence.

The main limitations are explicit:

- Augmented samplers have small/reference-size spatial and coefficient gates. Their conditional algebra is checked; broad full-model SBC and long-chain performance across all combinations remain research work.
- Nonstationarity uses fixed coefficient shrinkage scales. More elaborate scale hyperpriors or specialized interweaving are not implied.
- Sparse Vecchia/FEM primitives are working, but fitting currently materializes approximate observed covariance. No complete large-site sparse-MCMC benchmark is claimed.
- The 2D SPDE backend uses order 2 with smoothness $\nu=1$, or order 3 with $\nu=2$, and natural Neumann boundaries. Its mesh and boundary error are model choices; fractional order 2.5 is absent.
- Frozen encoder uncertainty, reversible-jump reservoir selection and arbitrary missing-data multivariate state-space inference are outside this release.
- Dependent-series and state-weighted calibration is assessed empirically; an exchangeability-based finite-sample guarantee is not assumed.

See [models.md](models.md) for exact equations and [implementation_registry.yaml](implementation_registry.yaml) for every implementation and compatibility status. The rank diagnostic definitions follow [Vehtari et al. (2021)](https://doi.org/10.1214/20-BA1221).
