# Models, scales and inference

This release provides a common Gaussian engine and several working research extensions. A model's implementation status is separate from evidence that it improves forecasting. The [implementation registry](implementation_registry.yaml) records all 23 blueprint identifiers, their APIs, compatibility rules and remaining limits. Experimental routes execute genuine samplers; their large-data mixing and comparative forecasting performance have not been established by their small verification examples.

The original six [derivation chapters](derivations/01_scope_and_baseline.md) are retained as the mathematical design history. Their planning-stage status statements are historical. This document and the registry describe the implemented release.

## 1. Common hierarchy

### Observations and deterministic features

The fitting panel has $n$ fixed input sites and $T$ likelihood times. The public data API stores observations as `[time,site]`; the numerical inference API uses

$$
Y\in\mathbb R^{n\times T},\quad
\Phi_{ij}=\phi_j(s_i),\quad
Z_{tk}=z_{tk},\quad
C\in\mathbb R^{p\times d},\quad N=nT,\quad k=pd.
$$

Response centring/scaling, PCA or encoder fitting, reservoir calibration and feature scaling use the declared conditioning window. The fitting likelihood starts after that window. The spatial dictionary is fixed from input geometry and is evaluated with the same transformation at prediction sites.

The reservoir is deterministic conditional on its seed and frozen transformations:

$$
u_{t-1}=E(y_{t-1},y_{t-2},\ldots),\qquad
r_t^{(1)}=(I-D_1)r_{t-1}^{(1)}+
D_1\tanh(W_1r_{t-1}^{(1)}+U_1u_{t-1}+b_1),
$$

$$
r_t^{(l)}=(I-D_l)r_{t-1}^{(l)}+
D_l\tanh(W_lr_{t-1}^{(l)}+U_l\widetilde r_t^{(l-1)}+b_l),
\qquad
z_t=\operatorname{FrozenScale}[1,u_{t-1},r_t,r_t^2,\text{selected products}].
$$

`FeatureSpec` selects PCA, local pooling, graph diffusion, or a separately fitted nonlinear encoder. The frozen encoder's uncertainty is not included in the posterior. Raw reservoir inputs, optional bounded score readout terms and the exact retained transformations are recorded in the fitted object; the equations in [chapter 1](derivations/01_scope_and_baseline.md) specify the V3 configuration.

### Conditional Gaussian readout

$$
\boxed{y_t=\Phi C z_t+e_t,\qquad
e_t\mid\theta,\sigma^2\sim N_n(0,\sigma^2R_\theta)}
$$

$$
R_\theta=V_\theta+qK_\theta,\qquad
b=\operatorname{vec}_{\rm column}(C),\qquad
b\mid\sigma^2,\psi,\theta\sim N_k(0,\sigma^2\Lambda_{\psi,\theta}^{-1}),
\qquad \sigma^2\sim\operatorname{IG}(a_0,b_0).
$$

`IG(a,b)` has density proportional to $v^{-a-1}\exp(-b/v)$; Gamma distributions below use **shape and rate**. Defaults are $(a_0,b_0)=(2,0.05)$. Every coefficient prior is proper, including any designated unshrunk baseline: its default fixed variance is $10\sigma^2$.

The likelihood is a product of causal conditional densities. In recursive ESN forecasting, the predictors depend on past observations; the full response sequence is not asserted to have an unconditional multivariate Gaussian distribution.

### Exact Gaussian calculation

For independent time innovations, define

$$
B=\Phi^\top R^{-1}\Phi,\quad G=Z^\top Z,\quad
h=\operatorname{vec}(\Phi^\top R^{-1}YZ),\quad
A=\Lambda+G\otimes B,
$$

$$
m=A^{-1}h,\quad
Q=\operatorname{tr}(Y^\top R^{-1}Y)-h^\top A^{-1}h,\quad
a_n=a_0+N/2,\quad b_n=b_0+Q/2.
$$

Then

$$
\sigma^2\mid Y,\theta,\psi\sim\operatorname{IG}(a_n,b_n),\qquad
b\mid\sigma^2,Y,\theta,\psi\sim N(m,\sigma^2A^{-1}).
$$

The normalized integrated likelihood is

$$
\begin{aligned}
\log p(Y\mid\theta,\psi)
={}&-\tfrac N2\log(2\pi)-\tfrac T2\log|R|
+\tfrac12\log|\Lambda|-\tfrac12\log|A|\\
&+a_0\log b_0-\log\Gamma(a_0)+\log\Gamma(a_n)-a_n\log b_n.
\end{aligned}
$$

Column-separable precision uses exact spatial/temporal eigensystems without constructing the $k\times k$ coefficient precision. General precision uses bounded dense Cholesky or checked perturb-and-CG coefficient sampling; the sampler does not claim a cheap general collapsed determinant. Cancellation-prone $Q$ is recomputed as residual energy plus the prior quadratic when necessary. See [Gaussian derivation](derivations/02_gaussian_priors_mcmc.md).

## 2. Readout priors M0–M6

All variances in this table are multiplied by the common $\sigma^2$. Let $S$ denote the coefficients subject to shrinkage; excluded groups retain their fixed proper Gaussian prior.

| ID / API | Hierarchy | Implemented update |
|---|---|---|
| M0 `PriorSpec.ridge()` | $\Lambda=\lambda I_k$; $\log\lambda\sim N(\log .1,1.5^2)$ by default | Collapsed MH or conditional Gaussian/slice |
| M1 `.grouped()` | Column $j$ has precision $\lambda_{g(j)}$; proper normal priors on $\log\lambda_g$ | Prior-whitened exact spectral algebra; MH or log-scale slice |
| M2 `.group_rhs()` / `.column_rhs()` | $b_j\mid\cdot\sim N(0,\sigma^2v_j)$, $v_j=c^2\tau^2\lambda_{u(j)}^2/(c^2+\tau^2\lambda_{u(j)}^2)$; $\lambda_u\sim C^+(0,1)$, $\tau\sim C^+(0,\tau_0)$, $c^2\sim\operatorname{IG}(\nu_c/2,\nu_cs_c^2/2)$ | Exact finite-slab log targets; slice or collapsed MH; optional fixed slab |
| M3 `.entry_rhs()` | Same finite-slab hierarchy with one local scale per selected entry | Conditional coefficient draw; actual RHS slice targets |
| M4 `.gigg()` | $b_j\sim N(0,\sigma^2\tau^2v_{g(j)}w_j)$; $v_g\sim\Gamma(a_g,1)$, $w_j\sim\operatorname{IG}(b_g,1)$, $\tau^2\mid\xi\sim\operatorname{IG}(1/2,1/\xi)$, $\xi\sim\operatorname{IG}(1/2,1/s_\tau^2)$ | Local IG, group GIG, global IG and auxiliary IG; fixed positive shape settings |
| M5 `.reference_r2()` | $v_j=W\pi_{g(j)}/a_{g(j)}$, $W=\bar r\,r/(1-r)$, $r\sim\operatorname{Beta}(a_r,b_r)$; Dirichlet or logistic-normal $\pi$ | Logit/ALR coordinates with required Jacobians; covariance-dependent precision recomputed |
| M6 `.correction()` | Frozen residualized correction features; baseline variance fixed, correction variance $\tau^2$, $\tau\sim C^+(0,s_\tau)$ | Conditional Gaussian plus log-scale slice, or separable collapse |
| M6 `.function_penalty()` | $\Lambda=\lambda_0I_k+\tau^{-2}F_0$, $\lambda_0>0$, fixed $F_0\succeq0$, $\tau\sim C^+(0,s_\tau)$ | General Gaussian conditional and full determinant in the scale target; bounded dense penalty |

The M5 frozen reference energy is $a_g=\|H_0X_{0g}\|_F^2/N_0$, where $H_0$ centres the reference field and $X_{0g}=Z_{0g}\otimes\Phi$. The reference residual scale is $\bar r=\operatorname{tr}(V+qK)/n$. This defines an expected signal-variance budget, not a Beta prior on the realized coefficient of determination. In augmented Student-t or heteroskedastic models it remains the **baseline Gaussian reference scale**: it excludes the Student-t variance inflation and evaluates variance covariates at zero. Those combinations are proper experimental variants, not a claim that $r$ equals their actual marginal response $R^2$.

For M4, the group conditional is $\operatorname{GIG}(a_g-m_g/2,2,\sum_{j\in g}b_j^2/(\sigma^2\tau^2w_j))$, under density $x^{p-1}\exp\{-(ax+\chi/x)/2\}$. The local conditional is $\operatorname{IG}(b_g+1/2,1+b_j^2/(2\sigma^2\tau^2v_g))$. The global/auxiliary IG steps are given in [chapter 2](derivations/02_gaussian_priors_mcmc.md). Finite-slab RHS uses its actual density; ordinary horseshoe inverse-gamma shortcuts are not substituted.

Conditional RHS sweeps also support a symmetric tradeoff proposal
$\log\tau' = \log\tau+\delta$, $\log\lambda_u'=\log\lambda_u-\delta$ for every local unit. All finite-slab coefficient variances remain unchanged, so its MH ratio contains only the transformed hyperpriors. `SamplerConfig.rhs_tradeoff_scale` controls its proposal standard deviation; zero disables it. This is a valid additional conditional transition, not a change to the prior or a partially collapsed step.

M6 is a proper Gaussian subspace penalty inspired by functional shrinkage. It does not inherit published functional-horseshoe asymptotic guarantees. Full transformed targets, reference normalizations and prior interpretations are in [chapter 4](derivations/04_structured_priors_features_ensembles.md).

## 3. Covariance C1–C5

### Matérn and global anisotropy

$$
K_{ij}=M_\nu\!\left(\frac{\sqrt{(s_i-s_j)^\top A(s_i-s_j)}}{\ell}\right),\quad
M_\nu(r)=\frac{2^{1-\nu}}{\Gamma(\nu)}(\sqrt{2\nu}r)^\nu K_\nu(\sqrt{2\nu}r),
$$

$$
A=\exp\begin{pmatrix}a&b\\b&-a\end{pmatrix},\qquad
(a,b)\sim N(0,s_A^2I_2),\quad s_A=.5.
$$

$\det A=1$ separates global range from anisotropy. Isotropy sets $a=b=0$. Defaults are $\nu=1.5$, $\log\ell\sim N(\log .18,.65^2)$ and $\log q\sim N(0,1.25^2)$. `infer_nu=True` uses exact categorical probabilities over the declared finite smoothness set; defaults are $(.5,1.5,2.5)$ with equal mass.

The argument called `ell` is not the SPDE practical range. With $\kappa=\sqrt{2\nu}/\ell$, that convention gives $\rho_{\rm range}=\sqrt{8\nu}/\kappa=2\ell$.

### Static nugget and valid nonstationarity

For C2,

$$V=\operatorname{diag}\exp(F_\epsilon\delta),\qquad
\delta\sim N(0,s_\delta^2I),\quad s_\delta=.3.$$

The basis has no constant shift and is centred/scaled on the original input geometry. Thus $\sigma^2$ is the reference geometric nugget scale, while average nugget variance is $\sigma^2\operatorname{tr}(V)/n$.

C4 uses a small fixed geometry basis $F(s)$ with four coefficient vectors $\gamma_u,\gamma_v,\gamma_a,\gamma_b\sim N(0,s_{NS}^2I)$; the implemented default is fixed $s_{NS}=.15$. It does not sample an additional half-normal geometry-scale hierarchy. Define $u=F\gamma_u$, $v=F\gamma_v$, and

$$
H(s)=\ell^2\exp\{2u(s)I-E(s)\},\quad
E(s)=\begin{pmatrix}a+F(s)\gamma_a&b+F(s)\gamma_b\\b+F(s)\gamma_b&-a-F(s)\gamma_a\end{pmatrix}.
$$

With $\bar H=(H(s)+H(s'))/2$ and $r^2=(s-s')^\top\bar H^{-1}(s-s')$,

$$
K_{NS}(s,s')=e^{v(s)+v(s')}
\frac{|H(s)|^{1/4}|H(s')|^{1/4}}{|\bar H|^{1/2}}M_\nu(r).
$$

Its diagonal is $e^{2v(s)}$. Therefore $q$ is a reference amplitude, not a common spatial marginal variance. The zero-coefficient limit recovers the stationary anisotropic kernel. Geometry centring is frozen when targets are appended.

### PC range and separate physical scales

The two-dimensional PC range option replaces the lognormal range prior:

$$
\lambda_r=-r_0\log\alpha_r,\qquad
\log p(h)=-h-\lambda_r/(2e^h)+\text{constant},\quad h=\log\ell.
$$

This transformed density includes the range Jacobian. The default elicitation is $P(\rho_{\rm range}<.3)=.05$, in the supplied coordinate units.

`scale_mode="separate", range_prior="pc"` instead uses

$$
\sigma_\epsilon\sim N^+(0,s_\epsilon^2),\quad
\sigma_w\sim\operatorname{Exp}(\lambda_w),\quad
q=\sigma_w^2/\sigma_\epsilon^2,\quad \sigma^2=\sigma_\epsilon^2,
\qquad\lambda_w=-\log\alpha_w/w_0.
$$

The sampled coordinates are $u_\epsilon=\log\sigma_\epsilon$ and $u_w=\log\sigma_w$, with log priors $u_\epsilon-e^{2u_\epsilon}/(2s_\epsilon^2)$ and $u_w-\lambda_we^{u_w}$. Only $C$ is integrated in a collapsed update. There is no additional prior on derived $q$, and no old inverse-gamma update for $\sigma^2$. This variant is available for iid Gaussian and full-residual AR models. Its use for nonstationary geometry is a declared reference-scale prior, not a newly derived joint PC prior for the whole nonstationary family.

## 4. Dynamics and robust likelihoods

The following are distinct alternative innovation hierarchies. Let $\mu_t=\Phi Cz_t$.

| Model / API `innovations` | Complete hierarchy | Main augmentation |
|---|---|---|
| D1 `full_residual_ar` | $e_1\sim N(0,\sigma^2R)$; $e_t=\rho e_{t-1}+\eta_t$, $\eta_t\sim N(0,\sigma^2(1-\rho^2)R)$ | Whiten $Y,Z$ together; first row unchanged; temporal determinant $(T-1)\log(1-\rho^2)$ |
| D2 `latent_ar` | $y_t=\mu_t+w_t+\epsilon_t$, $\epsilon_t\sim N(0,\sigma^2I)$; $w_1\sim N(0,\sigma^2qK)$; $w_t=\rho w_{t-1}+\eta_t$, $\eta_t\sim N(0,\sigma^2q(1-\rho^2)K)$ | Exact scalar-eigenmode FFBS |
| L1 `field_t` | $e_t\mid\omega_t\sim N(0,\sigma^2R/\omega_t)$; $\omega_t\sim\Gamma(\nu_T/2,\nu_T/2)$ | One precision per complete spatial field |
| L2 `local_t` | $w_t\sim N(0,\sigma^2qK)$ iid; $\epsilon_{it}\mid\omega_{it}\sim N(0,\sigma^2/\omega_{it})$; $\omega_{it}\sim\Gamma(\nu_T/2,\nu_T/2)$ | Spatial field plus a weight per observation |
| L3 `heteroskedastic` | $e_t\sim N(0,\sigma^2R_t)$; $R_t=\operatorname{diag}\{V_{ii}\exp(f_{it}^\top\eta)\}+qK$; $\eta\sim N(0,.3^2I)$ | General time-dependent covariance precision |

For D1/D2, $\operatorname{atanh}(\rho)\sim N(0,.8^2)$ by default. For L1/L2, $\nu_T=2+e^{\eta_\nu}$ with $\eta_\nu\sim N(\log 8,1)$. These priors are directly on transformed coordinates, so another Jacobian is not added. L3 covariates are centred and scaled over the fitting panel; that frozen transformation is applied to known future covariates.

In L1, $\operatorname{Var}(e_t)=\sigma^2R\nu_T/(\nu_T-2)$. In L2 it is $\sigma^2qK+\sigma^2\nu_TI/(\nu_T-2)$; this convolution is not a multivariate Student-t distribution.

The robust full conditionals are

$$
\omega_t\mid\cdot\sim\Gamma\left(\frac{\nu_T+n}{2},\frac{\nu_T+e_t^\top R^{-1}e_t/\sigma^2}{2}\right),
\qquad
\omega_{it}\mid\cdot\sim\Gamma\left(\frac{\nu_T+1}{2},\frac{\nu_T+(y_{it}-\mu_{it}-w_{it})^2/\sigma^2}{2}\right).
$$

For local-t fields, $P_t=q^{-1}K^{-1}+\operatorname{diag}\omega_t$ and
$w_t\mid\cdot\sim N(P_t^{-1}\operatorname{diag}(\omega_t)(y_t-\mu_t),\sigma^2P_t^{-1})$.
General heteroskedasticity uses

$$
A=\Lambda+\sum_t(z_tz_t^\top)\otimes(\Phi^\top R_t^{-1}\Phi),\qquad
h=\operatorname{vec}\sum_t\Phi^\top R_t^{-1}y_tz_t^\top.
$$

The exact FFBS recursions, state quadratics, transformed covariance targets and degrees-of-freedom target are derived in [chapter 3](derivations/03_covariance_latent_mcmc.md) and implemented in `dynamics.py`.

### Safe sampler schedules

For Gaussian noncollapsed inference: draw $C$, draw $\sigma^2$ when it has the IG prior, update prior scales, then update covariance and AR parameters. Covariance-dependent coefficient-prior determinants and quadratics remain in every relevant target.

For D2: draw $W$ by FFBS, then $C$, $\sigma^2$, prior scales, covariance and $\rho$. For L1: weights, $C$, $\sigma^2$, prior scales, covariance, degrees of freedom. For L2: $W$, weights, $C$, $\sigma^2$, prior scales, covariance, degrees of freedom. For L3: $C$, $\sigma^2$, prior scales, covariance, variance coefficients. These are ordinary Gibbs/MH schedules, without a stale-state partial-collapse step.

The conditional IG shape is $a_0+(N+k)/2$ for Gaussian, L1 and L3. It is $a_0+(2N+k)/2$ when the $nT$ latent spatial states are explicitly sampled in D2/L2. The Gaussian **collapsed** IG shape is $a_0+N/2$. Mixing these three shapes would change the posterior.

All adaptation stops at warmup. Checkpoints retain the RNG, transition/adaptation state and all required latent states; resuming can increase the retained-draw target while keeping the model, data and transition configuration fixed.

The opt-in `SamplerConfig(parameterization="ridge_total_scale")` uses
`log(kappa) = log(lambda) + log(1+q)` as a proposal coordinate for the
certified scalar-ridge, isotropic fixed-smoothness iid Gaussian Matérn route.
Its unit log-coordinate Jacobian and inverse evaluation preserve the
original posterior and priors. The default remains `legacy`; original
scientific parameters remain the diagnostic quantities. Benchmark fits now
persist their phase and initial/maximum draw targets so an interruption
cannot award an extra continuation. See [the coordinate and recovery
derivation](derivations/07_sampler_coordinates_recovery.md).

## 5. Sparse spatial reference implementations S1

**Vecchia:** input row order and Euclidean predecessor neighbours are fixed independently of sampled parameters. Conditional factors give $\widetilde Q=L^\top D^{-1}L$. Targets are appended, preserving every input factor. Rebuilding different target sets/orderings is not claimed to be projective; use a fixed master target domain. Duplicate latent coordinates must be deduplicated. Truncated conditioning need not give unit marginal variance: `operator(...).marginal_variances()` returns the actual diagonal.

**FEM SPDE:** a genuine triangular P1 mesh supplies lumped mass matrix $C_h$ and stiffness $G_h$. With $B=\kappa^2C_h+G_h$, the implemented precision is

$$
Q=\tau^2B(C_h^{-1}B)^{\alpha-1},\qquad
\alpha\in\{2,3\},\quad\nu=\alpha-1\ \text{in two dimensions}.
$$

The default is $\alpha=2,\nu=1$; this explicitly changes the smoothness from Matérn $3/2$. Natural Neumann boundaries, a fixed padded mesh and barycentric interpolation define the approximation. Prediction outside that mesh fails. The infinite-domain variance convention is $\tau^2=\Gamma(\nu)/(4\pi\Gamma(\alpha)\kappa^{2\nu})$; finite mesh/boundary marginal variances remain unequal and are reported. Mesh refinement alone does not remove boundary effects. Fractional $\alpha=2.5$ is not implemented.

For both backends, `SpatialOperator` computes solves and log determinants with sparse precision algebra and bounded marginal-diagonal batches. The available factorization is sparse LU, not claimed sparse Cholesky. With interpolation $A_h$,

$$
R=V+qA_hQ^{-1}A_h^\top,\quad
R^{-1}=V^{-1}-qV^{-1}A_h(Q+qA_h^\top V^{-1}A_h)^{-1}A_h^\top V^{-1}.
$$

Generic model fitting currently materializes the corresponding approximate covariance. Consequently this release has real sparse primitives and small/reference fitting integration; it does **not** claim an end-to-end large-site sparse MCMC workflow.

## 6. Compatibility and resource limits

| Route | Supported components and explicit limits |
|---|---|
| iid Gaussian / D1 | All readout priors; supported stationary/nonstationary/static-variance covariance settings; separate physical scales supported; no unvalidated arbitrary likelihood callback |
| D2 / L2 | No static nugget variation; complete fixed-site panel; common IG scale; no combination of latent AR with local-t measurements; SPDE interpolation must have full row rank for observed-site latent-state inference |
| L1 | Whole-field weights with common spatial covariance, including static variance; common IG scale |
| L3 | Known covariates `[n,T,h]` or `[T,h]`, $1\le h\le12$; no intercept/constant variance column; known future predictors required |
| All augmented routes | $n\le100$, $T\le2000$; experimental dense spatial computation; explicit backend compatibility checks |
| L3 coefficient solve | $pd\le900$, `auto` or `dense`; additionally satisfies the configured dense-memory gate |
| L2 coefficient solve | Dense within configured gates or checked matrix-free weighted CG |
| General dense Gaussian | Default $pd\le900$ and 256 MiB work-memory gate; larger general models use conditional matrix-free sampling |
| General M6 penalty | Default $pd\le900$; proper dense penalty only |
| Covariance materialization | Default at most 2,000 sites; sparse `operator()` does not use this dense materialization route |
| SPDE | Two dimensions, integer order 2/3; no nonstationary geometry or inferred smoothness mixture |

Finite-tolerance CG draws are explicitly labelled numerical approximations and report achieved residuals. Failed solves are rejected or use a reported bounded dense fallback where implemented. Checkpoint and retained-state memory budgets are also enforced.

## 7. Forecasts, ensembles and calibration

Each forecast path keeps one joint parameter draw for its whole horizon and feeds back its simulated **observations at the input sites**. Inputs and targets share one spatial draw; duplicate locations refer to the same observation path. Readout mean, conditional predictive mean, latent field and noisy observations are separately available.

D1 propagates the complete residual. D2 filters the terminal field under fixed drawn parameters and all likelihood/newly available history, then propagates the field with fresh iid nugget noise. New-location terminal states use a joint spatial conditional. Student-t future weights are drawn from their prior predictive Gamma distribution, not reused from training conditionals.

E1 selects one reservoir member per whole trajectory; validation stacking optimizes predictive scores and is not a posterior model probability. E2 supplies a separate common-hyperparameter-dimensional model-index sampler with normalized likelihood callbacks, plus a constrained integrated ridge/iid Gaussian high-level workflow. Integrated coefficient dimensions may differ because their normalization terms remain in the marginal likelihood. The sampled hyperparameter dimensions must agree; reversible-jump inference and posterior plug-in substitutes for model evidences are not implemented.

U1 keeps raw predictive draws unchanged. Marginal, per-horizon field, and whole-trajectory bands use explicitly different block scores. State-weighted calibration is an empirical Gaussian-kernel weighting rule; it is not a claim to reproduce the published ResCP algorithm. Temporal dependence does not automatically satisfy exchangeability or produce a finite-sample coverage guarantee.

## Primary references and attribution

- [Piironen and Vehtari (2017)](https://doi.org/10.1214/17-EJS1337SI): finite-slab regularized horseshoe motivation.
- [Boss et al. (2024)](https://doi.org/10.1214/23-BA1371): GIGG prior family.
- [Paciorek and Schervish (2006)](https://doi.org/10.1002/env.785): valid nonstationary covariance construction.
- [Fuglstad et al. (2019)](https://doi.org/10.1080/01621459.2017.1415907): stationary Gaussian-field PC range/amplitude priors.
- [Aguilar and Bürkner (2025)](https://doi.org/10.1214/25-BA1524): generalized variance-allocation prior motivation.
- [Shin, Bhattacharya and Johnson (2020)](https://doi.org/10.1080/01621459.2019.1654875): functional shrinkage motivation; no theorem transfer claimed.
- [Lindgren, Rue and Lindström (2011)](https://doi.org/10.1111/j.1467-9868.2011.00777.x): FEM/SPDE Gaussian-field connection.
- [Yao et al. (2018)](https://doi.org/10.1214/17-BA1091): predictive stacking.

The model-specific tensor algebra, priors, compatibility choices and sampler schedules in this repository are verified as properties of this implementation. Literature citations are not evidence that these particular combinations improve forecasting.
