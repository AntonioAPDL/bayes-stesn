# Variance allocation, subspace shrinkage, features, and ensembles

This chapter specifies the remaining research extensions and distinguishes posterior inference from feature engineering, predictive combination, and calibration. These are proposed modules, not completed implementations. Mathematical identities below are derived for the stated hierarchy; published results for related priors are not automatically inherited by these adaptations.

## M5. A reference-design R-squared allocation prior

R2D2, spatial R2D2, and GDR2 motivate allocating a total signal-variance budget among coefficient groups. GDR2 in particular replaces a Dirichlet allocation with more flexible logistic-normal allocations [AB25, YBR24]. An autoregressive ESN needs a precise reference distribution for its predictors: its realized design is correlated and depends on response history. We therefore define the following **reference expected-variance prior**, not a literal Beta prior on the realized coefficient of determination.

Freeze a reference design $X_0=Z_0\otimes\Phi_0$ using conditioning data and known geometry. Let $N_0$ be its row count, $H_0=I_{N_0}-\mathbf1\mathbf1^\top/N_0$, and partition its columns into temporal groups containing all spatial rows. Define

$$
h_g=\frac1{N_0}\operatorname{tr}(X_{0g}^\top H_0X_{0g})>0,
\qquad \bar r=\frac{\operatorname{tr}(R)}n.
$$

The trace is computed from small Gram matrices and column sums; the enormous tensor design is never materialized. A column that is constant over the entire reference field has zero centered energy and receives a separate proper intercept prior. Its exclusion must be explicit. More generally, zero-energy columns are removed or separately regularized before defining $h_g$.

Specify

$$
\begin{aligned}
r&\sim\operatorname{Beta}(a_r,b_r),\qquad
W=\bar r\frac r{1-r},\\
\pi&\sim\operatorname{Dirichlet}(\alpha_1,\ldots,\alpha_G),\\
b_a\mid\sigma^2,r,\pi,R
&\sim N\!\left(0,\sigma^2W\frac{\pi_{g(a)}}{h_{g(a)}}\right).
\end{aligned}
$$

All $a_r,b_r,\alpha_g$ are positive, exposed configuration parameters. Beta$(2,2)$ and equal expected allocations are starting sensitivity settings, not prescribed universal defaults. The frozen reference design and prior predictive field amplitudes guide their choice.

For $V_0(b)=N_0^{-1}\|H_0X_0b\|^2$, independence and zero prior means give

$$
\begin{aligned}
E\{V_0(b)\mid\sigma^2,r,\pi,R\}
&=\sum_g\sigma^2W\frac{\pi_g}{h_g}h_g=\sigma^2W,\\
\frac{E\{V_0(b)\mid\cdots\}}
{E\{V_0(b)\mid\cdots\}+\sigma^2\bar r}
&=r.
\end{aligned}
$$

This is a ratio of expected variances. It is not generally the expectation of $V_0/(V_0+\sigma^2\bar r)$, and it does not control the realized long-run variance of a nonlinear recursive forecast. Correlated feature columns do not invalidate the displayed expectation because cross-coefficient prior covariances are zero. They still affect posterior identifiability and forecast behavior.

### MCMC for the allocation prior

For complete temporal groups, the column precision is

$$
D_{jj}=\frac{h_{g(j)}}{W\pi_{g(j)}}.
$$

The constant field-intercept coefficient is a special case: separating only this single coefficient breaks the simple $D\otimes I_p$ form. To preserve the spectral backend, either assign the entire temporal-intercept column a separately calibrated proper variance and exclude that column from the allocation, or use the generic precision backend. The supported initial implementation uses the former. In this case the variance identity above concerns the allocated component only, with its separate intercept-column contribution reported separately.

Set $\xi=\operatorname{logit}(r)$ and $v_g=\log(\pi_g/\pi_G)$ for $g<G$. The complete collapsed target, up to constants, is

$$
\begin{aligned}
\mathcal T(\xi,v,\theta)
={}&\mathcal L_c\{R_\theta,D(\xi,v,\theta)\}
+\log p(\theta)\\
&+a_r\log r+b_r\log(1-r)
+\sum_{g=1}^G\alpha_g\log\pi_g.
\end{aligned}
$$

The Beta logit Jacobian changes $(a_r-1,b_r-1)$ to $(a_r,b_r)$; the additive-log-ratio Jacobian is $\prod_g\pi_g$. Symmetric logit/log-ratio random-walk Metropolis or scalar slice updates target this expression. Recover $\sigma^2,C$ from the Gaussian chapter after each retained hyperparameter state or only when needed for prediction. Because $W$ depends on $R$, a covariance proposal must update both $R$ and $D$.

For a GDR2-inspired alternative, directly specify

$$
v\sim N_{G-1}(\mu_v,\Sigma_v),\qquad
\pi=\operatorname{softmax}(v_1,\ldots,v_{G-1},0).
$$

Replace the last term by the Gaussian log density of $v$. There is **no softmax Jacobian** when $v$ itself is the parameter with a Gaussian prior. Fix a proper $\Sigma_v$ initially; estimating an unrestricted allocation covariance from one allocation vector is not a useful default.

Elliptical slice sampling [MAM10] is a valid optional update for this Gaussian-prior block: draw $e\sim N(0,\Sigma_v)$ and propose $v'=\mu_v+(v-\mu_v)\cos\omega+e\sin\omega$, shrinking an angular bracket until the collapsed log likelihood exceeds its sampled slice level. The other parameters are held fixed. This is a sampler choice, not an additional model extension.

## M6. Shrinkage toward a simpler dynamic model

Functional horseshoe work motivates shrinkage toward a function subspace [SBJ20]. Two explicit proper alternatives are useful here. Neither is advertised as an unchanged implementation of the published functional horseshoe.

### Efficient correction-space formulation

Partition the temporal features into a simple baseline $z_{0t}$, such as an intercept and linear lag scores, and nonlinear features $z_{1t}$. On a frozen reference window compute

$$
A_0=(Z_0^\top Z_0)^+Z_0^\top Z_1,
\qquad \widetilde z_{1t}=z_{1t}-A_0^\top z_{0t}.
$$

Use a documented singular-value tolerance in the pseudoinverse. The future residualization uses the same frozen $A_0$. The mean is

$$
\mu_t=\Phi C_0z_{0t}+\Phi C_1\widetilde z_{1t},
\qquad C_{0,ij}\sim N(0,\sigma^2v_{0j}),
\quad C_{1,ij}\sim N(0,\sigma^2\tau_1^2),
$$

where $v_{0j}>0$ are proper fixed variances and $\tau_1\sim C^+(0,s_1)$. Let $k_1$ be the number of correction coefficients and $u=\log\tau_1$. Its conditional log target is

$$
\mathcal T(u)
=-k_1u-\frac{\|C_1\|_F^2}{2\sigma^2e^{2u}}
+u-\log(1+e^{2u}/s_1^2)+\mathrm{const}.
$$

Use a scalar slice or Metropolis update. The collapsed version replaces the first two terms by $\mathcal L_c(R,D(u))$. This is a separable prior, so the fast solver remains available. As $\tau_1\downarrow0$, the model approaches the simpler mean. A finite-slab RHS on correction groups can replace the single half-Cauchy scale using the already derived RHS target.

Residualizing nonlinear features makes the intended simpler subspace explicit over the reference window. It does not guarantee orthogonality under every future design or spatial covariance, and it does not by itself remove all mean-versus-residual confounding.

### General proper function-penalty formulation

Let $P_0$ be a fixed orthogonal projector onto a desired simpler mean subspace at reference inputs. With fixed $\lambda_0>0$, define

$$
F_0=\frac1{N_0}X_0^\top(I-P_0)X_0,\qquad
\Lambda(\tau)=\lambda_0I_k+\tau^{-2}F_0,
\qquad b\mid\sigma^2,\tau\sim N(0,\sigma^2\Lambda(\tau)^{-1}).
$$

The ridge term makes the prior proper, including directions inside the simpler subspace and the null space of $X_0$. With $\tau\sim C^+(0,s_\tau)$ and $u=\log\tau$, its conditional target is

$$
\mathcal T(u)
=\frac12\log|\Lambda(e^u)|
-\frac{b^\top\Lambda(e^u)b}{2\sigma^2}
+u-\log(1+e^{2u}/s_\tau^2)+\mathrm{const}.
$$

The prior determinant cannot be dropped. If the spectrum $f_j\geq0$ of the fixed $F_0$ is available, $\log|\Lambda|=\sum_j\log(\lambda_0+e^{-2u}f_j)$. Otherwise this form requires a justified determinant computation and is limited to manageable or specially structured problems. The general form is initially a research backend; the correction-space version is the production candidate.

The coefficient step uses $A=\Lambda+H$ and the Gaussian chapter. A prior perturbation with covariance $\Lambda$ is

$$
\sqrt{\lambda_0}\,e_0+
\frac{1}{\tau\sqrt{N_0}}X_0^\top(I-P_0)e_1,
$$

for independent standard Gaussian vectors $e_0,e_1$ of dimensions $k,N_0$. Operator implementations avoid dense projectors but may still be too costly; numerical resource checks are required. For these priors the variance update uses the same full $b^\top\Lambda b$ penalty and shape $a_0+(N+k)/2$.

## F1--F2. More expressive inputs without hidden inference changes

The primary representation comparison varies input information before simply enlarging the reservoir. Initial candidates use 16, 32, and 64 training-estimated PCA scores, subject to rank and computational limits. Keep the fitted data budget constant when comparing architectures. A separate learning-curve experiment varies the number of observed sites and fitting times.

For a fixed graph on input sites, let $A_G$ be a symmetric, nonnegative adjacency matrix built from coordinates, and let

$$
S_G=D_G^{-1/2}(A_G+I)D_G^{-1/2},
\qquad (D_G)_{ii}=\sum_j(A_G+I)_{ij}.
$$

One auditable spatial input map is

$$
u_t=\operatorname{standardize}\left[
P_0y_t,\ P_1S_Gy_t,\ P_2S_G^2y_t
\right],
$$

where each fixed projection $P_j$ and the normalization are fitted only on the authorized historical window. Local weighted averages or regional PCA can be additional declared options. The graph and projection must respect input-site identities; target-only observations are never used to build reservoir inputs. This is a simple proposed spatial-input construction, not a claim to reproduce a published architecture exactly.

A convolutional or graph encoder is a later alternative, motivated by recent spatial DESN and CESAR work [WHW25, BGCC26, GCG26]. Train $E_\eta$ on the training fold, freeze $\widehat\eta$, and use $u_t=E_{\widehat\eta}(y_t)$. The readout posterior then conditions on $\widehat\eta$; it does not include uncertainty over encoder weights. Full Bayesian learning of the recurrent and encoder weights is a separate model and outside the first implementation release.

With any fixed feature map $z_t=\mathcal R(y_{<t};\omega)$, all readout MCMC equations remain valid. No new MCMC step is invented for $\omega$ when it is fixed. Hyperparameter optimization and architecture selection use validation origins, and a final sealed evaluation is untouched by those choices. Future exogenous inputs must be known or jointly simulated from a separately specified model.

Increasing the reservoir spectral radius is not a sufficient argument for stable recursion. For the proposed leaky tanh update, a useful sufficient contraction bound for a single layer is

$$
\|I-D_L\|_2+\|D_LW\|_2<1.
$$

It is sufficient, not necessary, and can be conservative for heterogeneous leaks. The baseline constrains the recurrent operator norm; the implementation records both operator norm and spectral radius, tests washout and long rollout behavior, and distinguishes stability of the input-driven reservoir from stability of the full observation-feedback forecast.

## E1. Reservoir ensembles and validation-based stacking

For independently generated fixed reservoir maps $\omega_1,\ldots,\omega_K$, fit separate posteriors and form the trajectory distribution

$$
p_{\mathrm{mix}}(y_{T+1:T+H}\mid\mathcal F_T)
=\sum_{k=1}^K w_k\,p_k(y_{T+1:T+H}\mid\mathcal F_T),
\qquad w_k\geq0,\quad\sum_kw_k=1.
$$

Start with equal weights. To draw a trajectory, choose one model index $K^*$, draw one posterior parameter state from that model, and recursively simulate the entire trajectory. Do not independently switch models or posterior coefficients at every horizon unless a different switching model is explicitly intended. The mixture includes both within-model and between-model uncertainty:

$$
\operatorname{Var}_{\mathrm{mix}}(Y)
=\sum_kw_k\operatorname{Var}_k(Y)
+\sum_kw_k(\mu_k-\bar\mu)(\mu_k-\bar\mu)^\top.
$$

For stacking, let $b$ index held-out validation blocks with forecasting histories available at their origins. Fit weights by

$$
\widehat w=\arg\max_{w\in\Delta^{K-1}}
\sum_b\log\left[\sum_kw_k
p_k(y_b\mid\mathcal F_{\mathrm{origin}(b)})\right].
$$

Use log-sum-exp calculations. This is a predictive optimization, not a Bayesian posterior over models [YVSG18]. The posterior used for a block must respect the declared frozen or refitted parameter policy. Time-ordered validation follows the information discipline of leave-future-out evaluation [BGV20].

Within a model, the block density integrates a product of sequential transition densities with a **shared** parameter draw:

$$
p_k(y_b\mid\mathcal F_o)
=\int\prod_{t\in b}p_k(y_t\mid\mathcal F_{t-1},\vartheta)
\,\Pi_{k,o}^{\mathrm{policy}}(d\vartheta).
$$

Here $\Pi_{k,o}^{\mathrm{policy}}$ is the parameter distribution actually used at origin $o$: either the frozen training posterior or a posterior refitted through that origin. These are different policies. An origin earlier than its parameter posterior's training endpoint is invalid for a forecast evaluation. The realized earlier observations within the block appear when evaluating its joint density by the chain rule; they must not be fed into a fixed-origin predictive rollout used for scoring the forecast paths. Multiplying independently marginalized one-step densities is generally not this joint predictive density. Latent-state models use their appropriate filtering likelihood. Monte Carlo block densities must be checked for weight degeneracy; very long high-dimensional blocks may be unsuitable for stable stacking.

An alternative is stacking by a declared marginal proper score, such as average CRPS. For draws from component distributions $F_k$,

$$
\operatorname{CRPS}(F_w,y)
=\sum_kw_kE|X_k-y|
-\frac12\sum_{k,l}w_kw_lE|X_k-X_l'|.
$$

Here the primed draw is independent. This can be estimated from saved draws without multivariate density estimation, but optimizes marginal forecasts rather than full trajectory dependence. Report that distinction and assess dependence separately.

## E2. A genuinely Bayesian finite reservoir index

A separate finite-model alternative specifies $P(M=k)=\pi_k^0$ and proper within-model priors. Then

$$
P(M=k\mid Y)
=\frac{\pi_k^0m_k(Y)}{\sum_l\pi_l^0m_l(Y)},
\qquad
m_k(Y)=\int p(Y\mid\theta_k,M=k)p(\theta_k\mid M=k)\,d\theta_k.
$$

For collapsed Gaussian models, $p(Y\mid\theta_k,M=k)$ is the **normalized** integrated likelihood derived earlier. It is not the evidence until the remaining hyperparameters have been integrated. On small common-dimensional cases, deterministic quadrature can provide an evidence reference; larger cases require validated importance, bridge, or sequential Monte Carlo calculations with repeated estimates and numerical uncertainty. These are optional evidence backends, not a claimed consequence of running an ordinary posterior chain.

If $\theta$ is genuinely common-dimensional with the same proper hyperprior under every $M$, a valid collapsed Gibbs/MH scheme alternates

$$
P(M=k\mid\theta,Y)\propto\pi_k^0p(Y\mid\theta,M=k)
$$

and an ordinary Metropolis update of $\theta\mid M,Y$. All candidate feature maps are fixed and the coefficient and variance dimensions are analytically integrated. If the hyperprior differs, include $p(\theta\mid M=k)$ in the index update. If hyperparameter dimensions differ, this formula does not define an ordinary Gibbs update; use separately validated evidences or a properly specified trans-dimensional algorithm. The first release should use equal weights or stacking while this research path is validated.

## U1. Calibration is a separate prediction layer

Calibration can correct empirical interval behavior but does not replace a likelihood or posterior. A transparent initial procedure uses an independent calibration period and standardized scores

$$
s_{bh}=\frac{|y_{bh}-\widehat\mu_{bh}|}{\widehat s_{bh}},
\qquad
I_{oh}=\left[\widehat\mu_{oh}-q_{1-\alpha,h}\widehat s_{oh},\;
\widehat\mu_{oh}+q_{1-\alpha,h}\widehat s_{oh}\right].
$$

The scale must be positive and computed using only the forecast distribution. For $m$ exchangeable calibration scores, the conformal order statistic has rank $\lceil(m+1)(1-\alpha)\rceil$, with the upper endpoint treated as infinity if the rank exceeds $m$. Ordinary exchangeability cannot be assumed for overlapping spatiotemporal forecast errors. Consequently this baseline is an empirically assessed calibration method on such data, not an automatic finite-sample guarantee.

Separate marginal site-horizon intervals from simultaneous field/path bands. A simultaneous calibration score can use a maximum over its declared sites and horizons; it generally needs wider bands and enough independent calibration blocks. Reservoir-weighted methods such as ResCP are a research option whose asymptotic coverage conditions must be checked for the intended target [NCBB26]. They do not supply an unconditional guarantee for arbitrary nonstationary fields.

A calibration object records its data period, forecast policy, target, weighting rule, and whether updates are permitted. In a fixed-origin $H$-step forecast, it cannot consume future realized errors during the simulated horizon. Calibration and raw posterior intervals are reported together. There is no MCMC equation for this postprocessing unless a new probabilistic calibration model is separately introduced.

## S1. Scalable spatial computation when required

Dense covariance is an intentional initial choice at the current 256 fitted sites. A scalable backend is introduced when profiling shows spatial factorizations dominate. Vecchia and SPDE choices change or approximate a specified spatial model; they are not intrinsically better correlation functions.

For a Vecchia approximation to the latent spatial correlation, choose an ordering and fixed predecessor neighbor sets $N_i$. From the intended covariance $K$, compute

$$
a_i=K_{i,N_i}K_{N_i,N_i}^{-1},\qquad
d_i=K_{ii}-K_{i,N_i}K_{N_i,N_i}^{-1}K_{N_i,i}.
$$

With $L_V$ unit lower triangular and row entries $-a_i$ at $N_i$,

$$
\widetilde Q_K=L_V^\top\operatorname{diag}(d_i^{-1})L_V,
\quad\widetilde K=\widetilde Q_K^{-1},\quad
\log|\widetilde Q_K|=-\sum_i\log d_i.
$$

For $R=I+q\widetilde K$, sparse solves can use

$$
R^{-1}=I-q(\widetilde Q_K+qI)^{-1},\qquad
\log|R|=\log|\widetilde Q_K+qI|-\log|\widetilde Q_K|.
$$

These identities preserve the Gaussian readout equations for the **approximated spatial model**. A truncated-neighbor approximation need not preserve unit marginal diagonals even if $K$ is a correlation matrix. The initial backend retains the resulting $\widetilde K$ and reports its diagonal and average variance: $q$ is an amplitude multiplier, not automatically an exact common sitewise variance ratio. Any alternative diagonal normalization is a separate registered choice.

Sparse Cholesky fill, neighbor count, ordering, and predictive behavior at new sites all require checks. Freeze the fitted input-site ordering and predecessor sets. Append target sites after the input sites without altering existing conditional factors, preserving the fitted input marginal. For reproducible comparisons define a fixed master target domain and its ordering, then subset its joint predictions; separately requested target sets are not assumed to define a projectively consistent process under arbitrary neighbor changes. It is not enough to approximate separate marginal covariances independently.

An SPDE alternative defines a latent mesh field with proper sparse precision $Q_h$ and an observation interpolation matrix $A_h$. The corresponding observed-site covariance and operators are

$$
\begin{aligned}
K_h&=A_hQ_h^{-1}A_h^\top,\qquad R=I+qK_h,\\
R^{-1}&=I-qA_h(Q_h+qA_h^\top A_h)^{-1}A_h^\top,\\
\log|R|&=\log|Q_h+qA_h^\top A_h|-\log|Q_h|.
\end{aligned}
$$

Mesh precision is not observed-site precision. Specify amplitude normalization, boundary conditions, and the interpolation domain with the model. For the standard Matérn SPDE relation, the operator order is $\alpha=\nu_K+d_s/2$. Classical local Markov constructions use integer $\alpha$; with $\nu_K=3/2$ in two dimensions, $\alpha=5/2$ is not an integer. Preserving that smoothness therefore requires an appropriate fractional/rational approximation, or the model must explicitly change its smoothness. Mesh, boundary, and interpolation errors become recorded approximation choices [LRL11, WKCG26].

For both spatial backends, conditional covariance proposals use the same Gaussian log-density form with the new determinant and quadratic operators. Hyperparameters keep the declared proper priors. Sampler acceptance must use a deterministic, reproducible likelihood for the selected approximate model, unless a separately justified stochastic-likelihood method is implemented. Dense agreement as approximation fidelity increases is a required validation experiment. For SPDE this comparison must use matching boundary conditions or appropriately enlarge the mesh domain: mesh refinement alone does not remove bounded-domain effects relative to an unbounded stationary Matérn model.

## Sources for this chapter

- **AB25:** Aguilar and Bürkner (2025), *Generalized Decomposition Priors on R2*. https://doi.org/10.1214/25-BA1524
- **YBR24:** Yanchenko, Bondell and Reich (2024), spatial R2D2 shrinkage. https://doi.org/10.1002/env.2829
- **SBJ20:** Shin, Bhattacharya and Johnson (2020), *Functional Horseshoe Priors for Subspace Shrinkage*. https://doi.org/10.1080/01621459.2019.1654875
- **MAM10:** Murray, Adams and MacKay (2010), *Elliptical slice sampling*. https://proceedings.mlr.press/v9/murray10a.html
- **WHW25:** Wang, Holan and Wikle (2025), *Echo State Networks for Spatio-Temporal Area-Level Data*. https://doi.org/10.1080/26941899.2025.2554883
- **BGCC26:** Bonas, Giani, Crippa and Castruccio (2026), *CESAR: A Convolutional Echo State AutoencodeR for High-Resolution Wind Forecasting*. https://doi.org/10.1029/2026JH001234
- **GCG26:** Geng, Castruccio and Genton (2026), *Spatial-Awared Deep Echo State Networks for High-Resolution Spatio-Temporal Wind Speed Nowcasting*. https://doi.org/10.1002/env.70119
- **YVSG18:** Yao, Vehtari, Simpson and Gelman (2018), *Using Stacking to Average Bayesian Predictive Distributions*. https://doi.org/10.1214/17-BA1091
- **BGV20:** Bürkner, Gabry and Vehtari (2020), *Approximate leave-future-out cross-validation for Bayesian time series models*. https://doi.org/10.1080/00949655.2020.1783262
- **NCBB26:** Neglia, Cini, Bronstein and Bianchi (2026), *ResCP: Reservoir Conformal Prediction for Time Series Forecasting*. https://arxiv.org/abs/2510.05060
- **LRL11:** Lindgren, Rue and Lindström (2011), *An explicit link between Gaussian fields and Gaussian Markov random fields: the stochastic partial differential equation approach*. https://doi.org/10.1111/j.1467-9868.2011.00777.x
- **WKCG26:** Wang, Kim, Castruccio and Genton (2026), *Modelling high-resolution spatio-temporal wind with deep echo state networks and stochastic partial differential equations*. https://doi.org/10.1093/jrsssc/qlaf045

The general Gaussian identities and the exact reference-variance/subspace specifications in this chapter are derivations for this blueprint. Citation of related work does not transfer that work's asymptotic guarantees to the new construction.
