# Covariance, latent dynamics, and robust likelihoods

This chapter specifies the proposed covariance and likelihood extensions. It is an implementation specification, not evidence that these extensions improve the existing forecasts. The inexpensive common-covariance models should be implemented before the latent-state and general weighted models.

## Conventions and a common marginal target

Write $Y=[y_1,\ldots,y_T]\in\mathbb R^{n\times T}$, $Z=[z_1,\ldots,z_T]^\top\in\mathbb R^{T\times d}$, $C\in\mathbb R^{p\times d}$, $N=nT$, and $k=pd$. The temporal features are predictable functions of preceding observed responses. In the fitted likelihood their realized values are fixed; this does not make the entire response sequence unconditionally Gaussian. Every covariance model below conditions on the same frozen feature construction.

Let $D$ be the $d\times d$ positive diagonal column-prior precision and

$$
\operatorname{vec}(C)\mid\sigma^2,D
\sim N_k(0,\sigma^2(D^{-1}\otimes I_p)),
\qquad
\sigma^2\sim\operatorname{IG}(a_0,b_0).
$$

The inverse-gamma convention is $p(v)\propto v^{-a-1}\exp(-b/v)$. Gamma distributions below use shape and rate. For common spatial covariance $R$,

$$
\begin{aligned}
B&=\Phi^\top R^{-1}\Phi,&G&=Z^\top Z,\\
j&=\operatorname{vec}(\Phi^\top R^{-1}YZ),&
A&=D\otimes I_p+G\otimes B,\\
Q&=\operatorname{tr}(Y^\top R^{-1}Y)-j^\top A^{-1}j,&
a_n&=a_0+N/2.
\end{aligned}
$$

Define the *integrated likelihood*, excluding priors on covariance and shrinkage parameters, by

$$
\ell_{\mathrm{col}}(R,D;Y,Z)
=-\frac T2\log|R|+\frac p2\log|D|
-\frac12\log|A|-a_n\log(b_0+Q/2)+\mathrm{const}.
$$

The Gaussian chapter derives its normalizing constant, stable evaluation of $Q$, spectral computation, and posterior recovery. Every occurrence of $\ell_{\mathrm{col}}$ in this chapter must use the complete parameter-dependent expression, including the prior determinant. If a prior makes $D$ depend on covariance parameters, that dependence must also be retained.

The same rule applies to *conditional* covariance targets below. At fixed $C,\sigma^2$, add

$$
\log p(C\mid\sigma^2,D(\vartheta))
=\frac p2\log|D(\vartheta)|
-\frac{\operatorname{tr}\{CD(\vartheta)C^\top\}}{2\sigma^2}
+\mathrm{const}
$$

whenever the updated parameter enters $D$. The displayed simpler scalar conditionals assume that $D$ is independent of that parameter. For example, an $R^2$ allocation prior tying coefficient scale to $q$ changes the conditional update for $q$. The global-scale inverse-gamma updates also assume that $D$ is independent of $\sigma^2$; the separate-scale specification in C5 deliberately uses a different sampler.

For a general separable temporal covariance $S$ and spatial covariance $R$,

$$
\operatorname{vec}(E)\mid\sigma^2\sim
N_N(0,\sigma^2(S\otimes R)).
$$

Choose $L$ such that $LSL^\top=I_T$, and form

$$
Y'=YL^\top,\qquad Z'=LZ.
$$

The integrated likelihood is then

$$
\ell_{\mathrm{col}}(R,D;Y',Z')-\frac n2\log|S|.
$$

The determinant term is required even when whitening is performed by a sparse or banded transformation.

## C1. Stationary Matérn and global anisotropy

For spatial coordinates in the declared distance units, define

$$
M_{\nu_K}(r)=
\frac{2^{1-\nu_K}}{\Gamma(\nu_K)}
(\sqrt{2\nu_K}\,r)^{\nu_K}
K_{\nu_K}(\sqrt{2\nu_K}\,r),
\qquad M_{\nu_K}(0)=1.
$$

The symbol $K_{\nu_K}$ on the right is the modified Bessel function. The spatial matrix is

$$
(K_{\ell,A_s})_{ij}
=M_{\nu_K}\!\left(
\frac{\sqrt{(s_i-s_j)^\top A_s(s_i-s_j)}}{\ell}
\right),
\qquad R=I_n+qK_{\ell,A_s}.
$$

Here $\sigma^2$ is nugget variance and $\sigma^2q$ is spatial innovation variance. The baseline has $\nu_K=3/2$ and $A_s=I_2$.

An identifiable anisotropy extension is

$$
A_s=\exp\!\begin{pmatrix}a&b\\b&-a\end{pmatrix}.
$$

The exponential is a matrix exponential. It gives $A_s\succ0$ and $\det A_s=1$, separating overall range from directional deformation. Isotropy is $a=b=0$. Unlike an angle-and-axis-ratio representation, this representation does not introduce an unidentified angle at isotropy.

An initial proper regularizing specification is

$$
(a,b)^\top\sim N_2(0,s_A^2I_2),\qquad s_A=0.5,
$$

with $s_A$ exposed in configuration and assessed through prior predictive spatial correlations. The existing lognormal priors on $q$ and $\ell$ remain an initial comparator.

For $\vartheta=(\log q,\log\ell,a,b,\ldots)$, the collapsed update targets

$$
\log\pi(\vartheta\mid Y)
=\ell_{\mathrm{col}}(R(\vartheta),D(\vartheta);Y,Z)
+\log p_{\vartheta}(\vartheta)+\mathrm{const}.
$$

A symmetric random-walk Metropolis proposal accepts with probability

$$
\min\{1,\exp[\log\pi(\vartheta^\star\mid Y)
-\log\pi(\vartheta\mid Y)]\}.
$$

Priors stated directly on log parameters require no additional Jacobian. Priors stated on positive physical parameters do require the log transformation Jacobian. Proposal adaptation ends after warmup.

Only $R$ changes, so this extension retains the exact common-covariance Gaussian solver. It adds no covariance factorization over the $nT$ observations.

### C3. Optional discrete smoothness

To compare smoothness without a continuously varying special-function parameter, use

$$
\nu_K\in\{1/2,3/2,5/2\},\qquad
P(\nu_K=v)=\pi_v,\qquad \pi_v=1/3
$$

as a prespecified initial prior. In a fully collapsed chain,

$$
P(\nu_K=v\mid Y,\vartheta_{-\nu_K})
\propto
\pi_v\exp\{\ell_{\mathrm{col}}(R_v,D_v;Y,Z)\}.
$$

Normalize these three probabilities with log-sum-exp and draw a categorical state. This is a within-model posterior mixture, distinct from selecting smoothness by validation. Use one declared approach in each experiment.

Under the usual SPDE practical-range convention,

$$
\rho_{\mathrm{range}}
=\frac{\sqrt{8\nu_K}}{\kappa},
\qquad
\kappa=\frac{\sqrt{2\nu_K}}{\ell},
\qquad
\rho_{\mathrm{range}}=2\ell.
$$

Thus a prior specified on this practical range can be held comparable across the three smoothness values. The term “range” must always identify its parameterization.

## C4. A valid nonstationary covariance extension

Changing the length scale separately at each endpoint inside a stationary kernel does not by itself define a symmetric positive-definite covariance. Use a valid construction.

Let $F(s)\in\mathbb R^h$ be a small, fixed spatial basis, centered over a declared reference geometry. Set

$$
\begin{aligned}
u(s)&=F(s)^\top\gamma_u,&
v(s)&=F(s)^\top\gamma_v,\\
a(s)&=a_{\mathrm{ref}}+F(s)^\top\gamma_a,&
b(s)&=b_{\mathrm{ref}}+F(s)^\top\gamma_b,
\end{aligned}
$$

and

$$
E(s)=
\begin{pmatrix}a(s)&b(s)\\b(s)&-a(s)\end{pmatrix},
\qquad
H(s)=\ell^2\exp\{2u(s)I_2-E(s)\}.
$$

The reference anisotropy parameters $(a_{\mathrm{ref}},b_{\mathrm{ref}})$ have the proper Gaussian prior from C1 and are distinct from the inverse-gamma hyperparameters. The local geometry $H(s)$ is positive definite. Let

$$
\bar H_{ss'}=\frac{H(s)+H(s')}{2},
\qquad
r_{ss'}^2=(s-s')^\top\bar H_{ss'}^{-1}(s-s').
$$

A Paciorek--Schervish-type covariance is

$$
K_{\mathrm{NS}}(s,s')
=e^{v(s)+v(s')}
\frac{|H(s)|^{1/4}|H(s')|^{1/4}}
{|\bar H_{ss'}|^{1/2}}
M_{\nu_K}(r_{ss'}).
$$

This construction follows [Paciorek and Schervish (2006)](https://doi.org/10.1002/env.785). Recent work by [Coube-Sisqueille, Banerjee, and Liquet (2026; online 2025)](https://doi.org/10.1080/10618600.2025.2516020) develops scalable nonstationary covariance models and investigates identifiability. The proposed first implementation uses the exact dense spatial matrix at $n=256$; it does not claim to reproduce that paper's sparse-process or interweaving algorithm.

It reduces to the preceding anisotropic Matérn when the varying coefficients are zero. Its diagonal is $e^{2v(s)}$. Consequently, $q$ is a reference spatial scale, and average spatial variance is

$$
\sigma^2q\,n^{-1}\sum_i e^{2v(s_i)}.
$$

It must not be reported as $\sigma^2q$ when $v$ varies. Reference-centering $u$ and $v$ removes arbitrary constant shifts that would otherwise duplicate $\ell$ and $q$.

The initial nonstationary specification should use a small geometry basis, for example $h=4$ to $9$, rather than one parameter per observed location. Its size is selected using development data only. Specify noncentered coefficients

$$
\gamma_j=\tau_j\xi_j,\quad
\xi_j\sim N_h(0,I_h),\quad
\tau_j\sim N^+(0,s_j^2),
\quad j\in\{u,v,a,b\}.
$$

For initial implementation, fixed small $\tau_j$ values are also allowed as explicit model settings, with sensitivity analysis. Neither choice is a universal prior recommendation.

### Elliptical-slice updates

Conditional on the scales, update $\xi_j$ using its standard Gaussian prior and the collapsed likelihood, following [Murray, Adams, and MacKay (2010)](https://proceedings.mlr.press/v9/murray10a.html). Draw an auxiliary $v_j\sim N_h(0,I_h)$, a slice threshold

$$
\log h_{\mathrm{slice}}
=\ell_{\mathrm{col}}(\xi_j)+\log U,\qquad U\sim U(0,1),
$$

and propose

$$
\xi_j(\theta)=\xi_j\cos\theta+v_j\sin\theta.
$$

Draw an initial angle and bracket of width $2\pi$, shrink the bracket on rejection, and accept once the integrated likelihood exceeds the threshold. The Gaussian prior is represented by the ellipse and must not be multiplied into the slice likelihood a second time.

For $\zeta_j=\log\tau_j$, the half-normal scale prior contributes

$$
\log p(\zeta_j)
=\zeta_j-\frac{e^{2\zeta_j}}{2s_j^2}+\mathrm{const}.
$$

Update $\zeta_j$ by univariate slice sampling or Metropolis against this prior plus the integrated likelihood. Other covariance parameters use the same collapsed target. Because $R=I+qK_{\mathrm{NS}}$ remains common over time, the common-covariance solver is retained.

## C2. Heteroskedastic Gaussian observations

### Static spatial nugget variation

A tractable extension is

$$
\epsilon_t\mid\sigma^2,\delta
\sim N_n(0,\sigma^2V_\delta),
\qquad
V_\delta=\operatorname{diag}\{\exp(F_\epsilon\delta)\},
$$

where $F_\epsilon$ has no intercept and its columns are centered on the fixed reference geometry. Set

$$
R=V_\delta+qK,\qquad
\delta\sim N_h(0,s_\delta^2I_h).
$$

The baseline is $\delta=0$. Here $\sigma^2$ is the geometric-reference nugget scale, while the average nugget variance is $\sigma^2n^{-1}\operatorname{tr}(V_\delta)$. Distinguish variance and standard-deviation predictors: $F_\epsilon\delta$ is log variance.

Update $\delta$ by elliptical slice sampling with $\ell_{\mathrm{col}}$ as its likelihood, or by Gaussian-prior Metropolis. Recover $C,\sigma^2$ using the common-covariance formulas. Static nonstationary spatial covariance and static heteroskedasticity can be combined, provided their scales are identifiable and the combination passes validation.

### L3. General site-by-time variation

A concrete first specification uses fixed predictable variance covariates $f_{it}\in\mathbb R^h$, with no intercept and a declared training-reference centering, and

$$
R_t(\eta)=\operatorname{diag}_i\{\exp(f_{it}^\top\eta)\}+qK,
\qquad \eta\sim N_h(0,s_\eta^2I_h).
$$

The scale $s_\eta>0$ is calibrated by prior predictive variance ratios and then fixed in configuration. Covariates may include fixed spatial effects, known seasonal terms, and prespecified interactions; future covariates must be known or computed from simulated past information. They cannot use future evaluation responses. Static spatial covariates reduce to C2. Update $\eta$ by elliptical slice sampling using the Gaussian conditional likelihood below, or by Metropolis with its proper Gaussian prior. The same general formulas support other explicitly registered $R_t$ families, but an arbitrary covariance function without a prior and prediction rule is not a supported model.

For

$$
y_t\mid C,\sigma^2,\eta
\sim N_n(\Phi Cz_t,\sigma^2R_t(\eta)),
$$

the conditional coefficient precision and information vector are

$$
\begin{aligned}
A_\eta
&=D\otimes I_p+
\sum_t(z_tz_t^\top)\otimes(\Phi^\top R_t^{-1}\Phi),\\
j_\eta
&=\operatorname{vec}\left[
\sum_t\Phi^\top R_t^{-1}y_tz_t^\top\right].
\end{aligned}
$$

Then

$$
\begin{aligned}
\operatorname{vec}(C)\mid\cdots
&\sim N(A_\eta^{-1}j_\eta,\sigma^2A_\eta^{-1}),\\
\sigma^2\mid C,\eta,D,Y
&\sim\operatorname{IG}\left(
a_0+\frac{N+k}{2},
b_0+\frac12\left[
\sum_t e_t^\top R_t^{-1}e_t+
\operatorname{tr}(CD C^\top)\right]\right),
\end{aligned}
$$

where $e_t=y_t-\Phi Cz_t$. Conditional covariance-parameter updates target

$$
-\frac12\sum_t\log|R_t|
-\frac1{2\sigma^2}\sum_t e_t^\top R_t^{-1}e_t
+\log p(\eta),
$$

plus the appropriate transformation Jacobians and any parameter-dependent coefficient-prior terms.

If $R_t=r_tR$, temporal weighting restores separability. For unrestricted $R_t$, the sum of Kronecker products generally has no common eigensystem. Use the general matrix-free Gaussian sampler from the preceding chapter and explicit per-time covariance operations. Do not silently approximate $R_t$ by an average matrix.

An exact Gaussian draw, assuming exact linear solves, has a particularly useful perturbation construction. Write $R_t=L_tL_t^\top$, $\Lambda=D\otimes I_p$, and independently draw $e_0\sim N_k(0,I_k)$ and $g_t\sim N_n(0,I_n)$. Form

$$
\xi=\Lambda^{1/2}e_0+
\operatorname{vec}\left\{
\sum_t\Phi^\top L_t^{-\top}g_tz_t^\top
\right\}.
$$

Its covariance is exactly $A_\eta$, because each time term contributes $(z_tz_t^\top)\otimes(\Phi^\top R_t^{-1}\Phi)$. Solve

$$
A_\eta\operatorname{vec}(C)=j_\eta+\sigma\xi.
$$

This yields the stated conditional covariance $\sigma^2A_\eta^{-1}$. For a generic prior precision, substitute a noise factor whose product with its transpose is $\Lambda$. The single-Kronecker likelihood-noise factor from the common-covariance sampler is not valid for a general sum of Kronecker products.

## C5. PC priors on range and separate scales

The two-dimensional joint PC prior on practical range $\rho_{\mathrm{range}}>0$ and spatial standard deviation $\sigma_w>0$ has density

$$
p(\rho_{\mathrm{range}},\sigma_w)
=\lambda_r\lambda_w\rho_{\mathrm{range}}^{-2}
\exp\left(-\frac{\lambda_r}{\rho_{\mathrm{range}}}
-\lambda_w\sigma_w\right).
$$

Elicit its parameters by

$$
\begin{aligned}
P(\rho_{\mathrm{range}}<r_0)&=\alpha_r,
&\lambda_r&=-r_0\log\alpha_r,\\
P(\sigma_w>w_0)&=\alpha_w,
&\lambda_w&=-\log\alpha_w/w_0.
\end{aligned}
$$

The thresholds are scientific or prior-predictive choices recorded in configuration, not fitted to the final evaluation outcomes. The prior shrinks range toward infinity and spatial amplitude toward zero; it does not assert that short ranges are impossible.

This is the stationary-GRF PC construction of [Fuglstad et al. (2019)](https://doi.org/10.1080/01621459.2017.1415907), expressed in two spatial dimensions. Reusing these distributions for the reference range and amplitude of C4 is a stated reference-scale prior; it is not a derivation of a joint PC prior for the entire nonstationary family. The additional anisotropy, amplitude, and geometry coefficients still require their own proper priors.

### Range-only replacement

Replacing only the lognormal range prior preserves the common inverse-gamma scale integration. With $h=\log\ell$ and $\rho_{\mathrm{range}}=2e^h$, the transformed prior is

$$
\log p(h)
=-h-\frac{\lambda_r}{2e^h}+\mathrm{const}.
$$

This includes the Jacobian $d\rho_{\mathrm{range}}/dh=\rho_{\mathrm{range}}$. Do not retain the old lognormal prior on $\ell$ as well.

### Separate nugget and spatial-scale priors

For this model variant specify

$$
\sigma_\epsilon\sim N^+(0,s_\epsilon^2),\qquad
\sigma_w\sim\operatorname{Exp}(\lambda_w),
$$

with the PC range prior above. Let

$$
u_\epsilon=\log\sigma_\epsilon,\quad
u_w=\log\sigma_w,\quad
q=e^{2(u_w-u_\epsilon)},\quad
R=I+qK,
$$

and retain the coefficient prior

$$
\operatorname{vec}(C)\mid\sigma_\epsilon^2,D
\sim N(0,\sigma_\epsilon^2(D^{-1}\otimes I_p)).
$$

Only $C$ is integrated out. Its integrated log likelihood is

$$
\ell_C
=-Nu_\epsilon-\frac T2\log|R|
+\frac p2\log|D|-\frac12\log|A|
-\frac{Q}{2e^{2u_\epsilon}}+\mathrm{const}.
$$

The transformed log target adds

$$
\begin{aligned}
\log p(u_\epsilon)&=
u_\epsilon-\frac{e^{2u_\epsilon}}{2s_\epsilon^2}+\mathrm{const},\\
\log p(u_w)&=u_w-\lambda_we^{u_w}+\mathrm{const},\\
\log p(h)&=-h-\frac{\lambda_r}{2e^h}+\mathrm{const},
\end{aligned}
$$

and the prior on shrinkage parameters. Update these coordinates with slice sampling or Metropolis. There is no separate prior or Jacobian for $q$ because it is a derived parameter in these sampling coordinates.

Recover

$$
\operatorname{vec}(C)\mid\cdots
\sim N(A^{-1}j,e^{2u_\epsilon}A^{-1}).
$$

Do not draw $\sigma^2$ from the old inverse-gamma update: that prior and conditional no longer belong to this variant. For spatially varying nugget or amplitude, the physical interpretation and elicitation of these reference scales require the normalization conventions of C4 and C2.

## D1. Autoregression of the complete residual

This model applies temporal dependence to the *combined* spatial and nugget residual:

$$
\begin{aligned}
e_t&=y_t-\Phi Cz_t,\\
e_1&\sim N_n(0,\sigma^2R),\\
e_t\mid e_{t-1}
&\sim N_n(\rho e_{t-1},\sigma^2(1-\rho^2)R),
\quad t\ge2,\quad |\rho|<1.
\end{aligned}
$$

Thus $\sigma^2R$ is stationary residual covariance. The first modeled residual is initialized independently of the fixed ESN conditioning history. This explicit boundary convention is different from filtering a residual process through all conditioning observations.

Let $(S_\rho)_{tt'}=\rho^{|t-t'|}$. Then

$$
|S_\rho|=(1-\rho^2)^{T-1}.
$$

Whitening leaves the first column unchanged and transforms subsequent columns as

$$
y_t'=\frac{y_t-\rho y_{t-1}}{\sqrt{1-\rho^2}},
\qquad
z_t'=\frac{z_t-\rho z_{t-1}}{\sqrt{1-\rho^2}}.
$$

The first transformed feature is $z_1'=z_1$. The collapsed target for $\eta_\rho=\operatorname{atanh}\rho$, under $\eta_\rho\sim N(0,s_\rho^2)$, is

$$
\ell_{\mathrm{col}}(R,D;Y',Z')
-\frac{n(T-1)}2\log(1-\rho^2)
-\frac{\eta_\rho^2}{2s_\rho^2}
+\text{other transformed log priors}.
$$

No Jacobian is added for a prior stated directly on $\eta_\rho$. A prior stated on $\rho$ would instead contribute $\log(1-\rho^2)$ in these coordinates.

Update $\eta_\rho$ and covariance/prior parameters in the collapsed chain, then recover $C,\sigma^2$ from the whitened regression. Forecast by propagating each posterior path's terminal residual with new innovations of covariance $\sigma^2(1-\rho^2)R$ and feeding simulated observations to the ESN.

At unobserved forecast locations, the terminal full residual is sampled conditionally as

$$
e_{*,T}\mid e_{o,T},\cdots
\sim N\!\left(
R_{*o}R_{oo}^{-1}e_{o,T},
\sigma^2[R_{**}-R_{*o}R_{oo}^{-1}R_{o*}]
\right).
$$

Use the corresponding kernel blocks, with nugget covariance only for identical observation coordinates; copy already represented locations exactly. Propagate the unique union of input and target locations jointly, using its full $R$ matrix. This retains covariance between the simulated inputs and targets throughout recursive prediction.

This extension is computationally modest, but it makes the nugget component serially correlated as part of $e_t$. It must have a distinct model identifier and must not be described as an AR spatial field with iid measurement errors.

## D2. Autoregressive spatial field with iid measurement noise

### Model and initialization

The scientifically different latent-state hierarchy is

$$
\begin{aligned}
y_t&=\Phi Cz_t+w_t+\epsilon_t,&
\epsilon_t&\overset{\mathrm{iid}}{\sim}N_n(0,\sigma^2I_n),\\
w_1&\sim N_n(0,\sigma^2qK),\\
w_t\mid w_{t-1}
&\sim N_n(\rho w_{t-1},\sigma^2q(1-\rho^2)K),
&t&\ge2.
\end{aligned}
$$

Use $\eta_\rho=\operatorname{atanh}\rho\sim N(0,s_\rho^2)$, the declared positive-parameter priors, and the same Gaussian coefficient hierarchy. The initial $w_1$ is independent of the fixed conditioning history. An alternative initialization based on filtering that history is a separate explicit likelihood policy, not an implicit change.

Now $q$ is the stationary spatial-to-nugget variance ratio. Differencing $y_t-\Phi Cz_t$ gives innovation

$$
\eta_t+\epsilon_t-\rho\epsilon_{t-1}.
$$

These errors are temporally correlated, so the complete-residual whitening likelihood in D1 is incorrect for this hierarchy.

### Exact FFBS through a spatial eigensystem

The following forward-filtering backward-sampling construction specializes Gaussian state sampling in [Carter and Kohn (1994)](https://doi.org/10.1093/biomet/81.3.541) to a common spatial eigensystem. The eigen reduction is an algebraic consequence of a scalar AR coefficient and iid nugget covariance.

At fixed $K$, diagonalize

$$
K=U\operatorname{diag}(\kappa_1,\ldots,\kappa_n)U^\top.
$$

Set $\widetilde r_t=U^\top(y_t-\Phi Cz_t)$ and $x_t=U^\top w_t$. For each spatial mode $i$,

$$
\widetilde r_{it}=x_{it}+\widetilde\epsilon_{it},
\qquad \widetilde\epsilon_{it}\sim N(0,\sigma^2),
$$

and

$$
x_{i1}\sim N(0,\sigma^2q\kappa_i),\qquad
x_{it}\mid x_{i,t-1}\sim
N(\rho x_{i,t-1},Q_i),
\quad Q_i=\sigma^2q(1-\rho^2)\kappa_i.
$$

The modes are conditionally independent. Initialize prediction mean $a_{i1}=0$ and variance $P_{i1}=\sigma^2q\kappa_i$. The scalar Kalman recursions are

$$
\begin{aligned}
F_{it}&=P_{it}+\sigma^2,&
g_{it}&=P_{it}/F_{it},\\
m_{it}&=a_{it}+g_{it}(\widetilde r_{it}-a_{it}),&
V_{it}&=(1-g_{it})P_{it},\\
a_{i,t+1}&=\rho m_{it},&
P_{i,t+1}&=\rho^2V_{it}+Q_i.
\end{aligned}
$$

Draw $x_{iT}\sim N(m_{iT},V_{iT})$. For $t=T-1,\ldots,1$, define

$$
J_{it}=\frac{\rho V_{it}}{P_{i,t+1}},
$$

and draw

$$
x_{it}\mid x_{i,t+1},Y,\cdots
\sim N\!\left(
m_{it}+J_{it}(x_{i,t+1}-\rho m_{it}),
V_{it}-J_{it}^2P_{i,t+1}
\right).
$$

Finally set $w_t=Ux_t$. Positive-definite covariance evaluation and stable square-root or Joseph-form filtering should be used where needed. Materially negative variances are numerical failures, not values to clip silently.

### Coefficient and scale updates

Given $W=[w_1,\ldots,w_T]$, the observation regression is

$$
Y-W=\Phi CZ^\top+\epsilon.
$$

Use the Gaussian coefficient update with $R=I_n$:

$$
\begin{aligned}
A_0&=D\otimes I_p+(Z^\top Z)\otimes(\Phi^\top\Phi),\\
j_0&=\operatorname{vec}\{\Phi^\top(Y-W)Z\},\\
\operatorname{vec}(C)\mid\cdots
&\sim N(A_0^{-1}j_0,\sigma^2A_0^{-1}).
\end{aligned}
$$

Define

$$
\begin{aligned}
Q_w(\rho,K)
&=w_1^\top K^{-1}w_1\\
&\quad+\frac1{1-\rho^2}
\sum_{t=2}^T(w_t-\rho w_{t-1})^\top
K^{-1}(w_t-\rho w_{t-1}),\\
S_y&=\|Y-\Phi CZ^\top-W\|_F^2,\\
S_C&=\operatorname{tr}(CD C^\top).
\end{aligned}
$$

The full conditional is

$$
\boxed{
\sigma^2\mid C,W,\cdots
\sim\operatorname{IG}\left(
a_0+\frac{N_{\mathrm{obs}}+N_{\mathrm{latent}}+k}{2},
b_0+\frac12\{S_y+Q_w/q+S_C\}
\right).
}
$$

For this complete-panel model $N_{\mathrm{obs}}=N_{\mathrm{latent}}=nT$. The latent Gaussian state density contributes to the inverse-gamma shape even though $W$ is unobserved.

### Remaining full conditional targets

The spatial-state log density, omitting parameter-independent terms, is

$$
\begin{aligned}
\ell_W
={}&-\frac{nT}{2}\log q-\frac T2\log|K|
-\frac{n(T-1)}2\log(1-\rho^2)\\
&-\frac{Q_w(\rho,K)}{2\sigma^2q}.
\end{aligned}
$$

Update $\log q$, covariance parameters of $K$, and $\eta_\rho$ by slice sampling or Metropolis with $\ell_W$ plus their declared transformed priors. In particular, the existing normal prior on $\log q$ gives

$$
\log\pi(\log q\mid\cdots)
=-\frac N2\log q-\frac{Q_w}{2\sigma^2q}
-\frac{(\log q-\mu_q)^2}{2s_q^2}+\mathrm{const}.
$$

These expressions assume that $D$ is independent of the updated covariance parameter. If it is not, add the conditional coefficient-prior log density given in the opening conventions; for a general precision $\Lambda(\vartheta)$, it is $\tfrac12\log|\Lambda(\vartheta)|-\operatorname{vec}(C)^\top\Lambda(\vartheta)\operatorname{vec}(C)/(2\sigma^2)$.

For group precision $\lambda_g$ covering $d_g$ temporal columns,

$$
\log\pi(\log\lambda_g\mid C,\sigma^2)
=\frac{pd_g}{2}\log\lambda_g
-\frac{\lambda_g\|C_g\|_F^2}{2\sigma^2}
-\frac{(\log\lambda_g-\mu_g)^2}{2s_g^2}
+\mathrm{const}.
$$

More elaborate coefficient priors use their full conditional updates from the prior chapter. Discrete Matérn smoothness can be updated from $\pi_v\exp(\ell_W(K_v))$ conditional on $W$, rather than the integrated target used in C1.

A safe ordinary sweep is: update $W$ by FFBS; update $C$; update $\sigma^2$; update coefficient-prior scales; update covariance and AR parameters. Each step conditions on the current values of all other variables. This schedule requires no partial-collapse argument.

### Forecasts at observed and new locations

For each posterior draw retain its $w_T$. At new target locations, draw the terminal state jointly using

$$
w_{*,T}\mid w_T,\cdots
\sim N\!\left(
K_{*o}K_{oo}^{-1}w_T,\,
\sigma^2q[K_{**}-K_{*o}K_{oo}^{-1}K_{o*}]
\right).
$$

Matched observed locations are copied exactly, not simulated twice. This conditional is sufficient because the scalar-AR spatial process is separable and the spatial conditional remainder is independent of the full observed-location state history.

Propagate

$$
w_{\mathcal U,t}=\rho w_{\mathcal U,t-1}+
\eta_{\mathcal U,t},\qquad
\eta_{\mathcal U,t}\sim
N(0,\sigma^2q(1-\rho^2)K_{\mathcal U\mathcal U}),
$$

where $\mathcal U$ is the union of reservoir-input locations and requested target locations, and draw new iid nugget noise over that union. Propagate these locations jointly to preserve the cross-location covariance between the observations driving the reservoir and the displayed forecast. Only simulated observations at the input locations update the reservoir within a forecast block.

### Resource implications

The scalar filtering recursions cost $O(nT)$ after spatial transformations; rotating full arrays costs $O(n^2T)$, and changing $K$ requires an $O(n^3)$ eigendecomposition. The coefficient draw retains the common-covariance algebra conditional on $W$. Storing one complete $W$ array at $256\times7600$ in float64 uses approximately 15.6 MB; storing every MCMC state would be much larger and is not the default.

This backend initially requires a complete panel at fixed observed locations and isotropic nugget covariance in the rotated representation. Time-varying observation masks or arbitrary local measurement variances generally couple the modes and require a different state-space backend.

## L1. Whole-field Student-t innovations

Let $\nu_T>2$ denote Student-t degrees of freedom, distinct from Matérn smoothness $\nu_K$. Specify

$$
\begin{aligned}
e_t\mid\omega_t,\sigma^2,R
&\sim N_n(0,\sigma^2R/\omega_t),\\
\omega_t\mid\nu_T
&\overset{\mathrm{iid}}{\sim}
\operatorname{Gamma}(\nu_T/2,\nu_T/2).
\end{aligned}
$$

Then $e_t$ is multivariate Student-t with *scale matrix* $\sigma^2R$, and its covariance is

$$
\operatorname{Cov}(e_t\mid\sigma^2,R,\nu_T)
=\frac{\nu_T}{\nu_T-2}\sigma^2R.
$$

If latent and observation predictions are both required, define $w_t$ and $\epsilon_t$ as conditionally independent Gaussians given the same $\omega_t$, with covariances $\sigma^2qK/\omega_t$ and $\sigma^2I/\omega_t$. Marginally they share the field-scale shock. This model represents whole-field shocks; it is not independent robustness for individual sensors.

The training sampler in this section integrates $W$ out throughout. Conditional latent-field draws for reporting are a postprocessing operation, and are not inserted into the following sweep. The weight conditional is

$$
\boxed{
\omega_t\mid\cdots
\sim\operatorname{Gamma}\left(
\frac{\nu_T+n}{2},
\frac{\nu_T+e_t^\top R^{-1}e_t/\sigma^2}{2}
\right).
}
$$

For postprocessing at the observed sites, the conditional latent draw is

$$
w_t\mid e_t,\omega_t,\cdots
\sim N\!\left(
qKR^{-1}e_t,\,
\frac{\sigma^2}{\omega_t}
[qK-qKR^{-1}qK]
\right).
$$

With $\Omega=\operatorname{diag}(\omega_1,\ldots,\omega_T)$, use

$$
G_\omega=Z^\top\Omega Z,\qquad
j_\omega=\operatorname{vec}(\Phi^\top R^{-1}Y\Omega Z)
$$

for the Gaussian coefficient conditional. The scale conditional is

$$
\sigma^2\mid C,\omega,\cdots
\sim\operatorname{IG}\left(
a_0+\frac{N+k}{2},
b_0+\frac12\left[
\sum_t\omega_t e_t^\top R^{-1}e_t+S_C\right]\right).
$$

Conditional covariance updates use

$$
-\frac T2\log|R|
-\frac1{2\sigma^2}\sum_t\omega_t e_t^\top R^{-1}e_t
+\log p(\text{covariance parameters}).
$$

For degrees of freedom, set

$$
\nu_T=2+\exp\eta_\nu,\qquad
\eta_\nu\sim N(\log8,1).
$$

The second normal argument is variance, so this gives prior median $\nu_T=10$. The scalar update targets

$$
\begin{aligned}
\log\pi(\eta_\nu\mid\omega)
={}&T\left[
\frac{\nu_T}{2}\log\frac{\nu_T}{2}
-\log\Gamma(\nu_T/2)\right]\\
&+\left(\frac{\nu_T}{2}-1\right)\sum_t\log\omega_t
-\frac{\nu_T}{2}\sum_t\omega_t
-\frac{(\eta_\nu-\log8)^2}{2}
+\mathrm{const}.
\end{aligned}
$$

There is no additional Jacobian because the prior is specified on $\eta_\nu$. Use univariate slice sampling or Metropolis with numerically stable log-gamma evaluation.

An ordinary sweep updates weights, coefficients, scale, coefficient-prior parameters, covariance parameters, and degrees of freedom using their full conditionals. Alternatively, update the weights conditional on the previous $C,\sigma^2$, then perform a *joint* block update of covariance/prior hyperparameters and $(C,\sigma^2)$: first use the weighted integrated likelihood, then immediately regenerate $(C,\sigma^2)$. Do not insert an update conditioning on stale coefficients between collapse and recovery.

The weighted integrated likelihood uses $Y'=Y\Omega^{1/2}$ and $Z'=\Omega^{1/2}Z$, and includes

$$
\frac n2\sum_t\log\omega_t.
$$

This term is constant for covariance moves conditional on weights but is not removable from a target that updates weights marginally.

Future weights are sampled from their *prior predictive* Gamma distribution, with posterior $\nu_T$, rather than from training full conditionals. The Gaussian limit as $\nu_T\to\infty$ is an implementation check.

If a future implementation instead conditions on sampled $W$ during training, these are different full conditionals: the weight shape becomes $(\nu_T+2n)/2$, its rate uses both $w_t^\top(qK)^{-1}w_t/\sigma^2$ and $\|y_t-\Phi Cz_t-w_t\|^2/\sigma^2$, and the global-scale shape becomes $a_0+(2N+k)/2$. Mixing those two conditioning regimes in one ordinary Gibbs sweep is incorrect.

## L2. Local Student-t measurement noise

The individual-measurement alternative retains a Gaussian spatial field:

$$
\begin{aligned}
y_{it}&=(\Phi Cz_t)_i+w_{it}+\epsilon_{it},\\
w_t&\overset{\mathrm{ind}}{\sim}N_n(0,\sigma^2qK),\\
\epsilon_{it}\mid\omega_{it}
&\sim N(0,\sigma^2/\omega_{it}),\\
\omega_{it}\mid\nu_T
&\overset{\mathrm{iid}}{\sim}
\operatorname{Gamma}(\nu_T/2,\nu_T/2).
\end{aligned}
$$

The marginal residual covariance is

$$
\operatorname{Cov}(y_t-\Phi Cz_t\mid\cdots)
=\sigma^2qK+\frac{\nu_T}{\nu_T-2}\sigma^2I_n.
$$

This Gaussian spatial-field plus local Student-t convolution is generally not a multivariate Student-t distribution. Here $q$ compares the spatial variance to the *nugget scale* $\sigma^2$; the actual marginal nugget variance is inflated by $\nu_T/(\nu_T-2)$. Local Student-t likelihoods are a standard robust-regression construction; [Jylänki, Vanhatalo, and Vehtari (2011)](https://www.jmlr.org/papers/v12/jylanki11a.html) study their use with Gaussian processes. The ESN-specific conditionals below follow directly from the displayed Gamma mixture.

Define $r_t=y_t-\Phi Cz_t$, $\Omega_t=\operatorname{diag}(\omega_{1t},\ldots,\omega_{nt})$, and $P_t=q^{-1}K^{-1}+\Omega_t$. The exact conditionals are

$$
\begin{aligned}
w_t\mid\cdots
&\sim N(P_t^{-1}\Omega_t r_t,\sigma^2P_t^{-1}),\\
\omega_{it}\mid\cdots
&\sim\operatorname{Gamma}\left(
\frac{\nu_T+1}{2},
\frac{\nu_T+(r_{it}-w_{it})^2/\sigma^2}{2}
\right).
\end{aligned}
$$

The coefficient precision and information are

$$
\begin{aligned}
A_\omega
&=D\otimes I_p+
\sum_t(z_tz_t^\top)\otimes(\Phi^\top\Omega_t\Phi),\\
j_\omega
&=\operatorname{vec}\left[
\sum_t\Phi^\top\Omega_t(y_t-w_t)z_t^\top\right].
\end{aligned}
$$

Draw $C$ with the general Gaussian sampler. For matrix-free products, if $V=\operatorname{unvec}(v)$, the likelihood contribution is computed as

$$
\operatorname{unvec}\{(A_\omega-D\otimes I_p)v\}
=\Phi^\top\left[(\Phi VZ^\top)\odot\omega\right]Z,
$$

where $\omega$ is the $n\times T$ weight array.

In particular, independently draw $e_0\sim N_k(0,I_k)$ and a matrix $E$ with iid standard-normal entries. Use the perturbation

$$
\xi=(D^{1/2}\otimes I_p)e_0+
\operatorname{vec}\{\Phi^\top(\sqrt{\omega}\odot E)Z\},
$$

and solve $A_\omega\operatorname{vec}(C)=j_\omega+\sigma\xi$. The square root of $\omega$ is elementwise. This provides the required nonseparable Gaussian noise without constructing a $k\times k$ matrix.

Let

$$
S_{\epsilon,\omega}
=\sum_{it}\omega_{it}
\left[y_{it}-(\Phi Cz_t)_i-w_{it}\right]^2,
\qquad
S_w=\sum_tw_t^\top K^{-1}w_t.
$$

Then

$$
\sigma^2\mid\cdots
\sim\operatorname{IG}\left(
a_0+\frac{N_{\mathrm{obs}}+N_{\mathrm{latent}}+k}{2},
b_0+\frac12\{S_{\epsilon,\omega}+S_w/q+S_C\}
\right).
$$

Update $\log q$ and covariance parameters against

$$
-\frac{nT}{2}\log q-\frac T2\log|K|
-\frac{S_w}{2\sigma^2q}
+\text{transformed log priors}.
$$

As in D2, include the coefficient-prior determinant and quadratic if its precision depends on any updated covariance parameter.

The degrees-of-freedom update is L1 with $T$ replaced by the number of local weights and sums over $(i,t)$. Coefficient-prior updates are unchanged given $C,\sigma^2$.

A safe sweep updates $W$, local weights, $C$, $\sigma^2$, shrinkage parameters, covariance parameters, and degrees of freedom. For predictions draw independent local Gamma weights, a joint Gaussian spatial field, and conditionally independent nugget noise at each horizon.

This extension is substantially more expensive: the spatial-state conditional differs at every time, and the coefficient update is not separable. Process time slices in bounded batches; do not store all $T$ dense spatial factorizations. A latent-AR field combined with local Student-t measurements generally invalidates the scalar-mode FFBS in D2, because the rotated observation covariance is no longer diagonal. That combination requires a separately validated multivariate state-space solver.

## Supported combinations and promotion gates

| Combination | Initial sampler route | Qualification |
|---|---|---|
| Common stationary or anisotropic Matérn + group Gaussian prior | Fully collapsed spectral sampler | First implementation priority |
| Common nonstationary covariance or static nugget variation | Collapsed spectral sampler + elliptical slice/MH | Low-dimensional spatial parameterization |
| Range-only PC prior | Existing collapse with changed transformed prior | Replace, rather than multiply, the old range prior |
| Separate nugget/spatial-scale PC hierarchy | Collapse coefficients only | Numerical scale sampling |
| AR on complete residual | Banded temporal whitening + spectral sampler | Nugget is serially correlated in this model |
| Latent spatial AR + iid Gaussian nugget | Scalar-mode FFBS + conditional spectral coefficient sampler | Complete fixed-site panel initially |
| Whole-field Student-t innovations | Time-weighted Gaussian sampler + Gamma updates | Scale matrix differs from covariance |
| Local Student-t nugget | Latent spatial draws + matrix-free coefficients | Expensive; pilot and resource gate required |
| Arbitrary $R_t$ | General weighted Gaussian backend | No claim of a single cached eigensystem |
| Latent AR + local Student-t nugget or time-varying observation masks | General multivariate state-space backend | Not automatically enabled by composing components |

A combination is supported only when its likelihood, sampler, forecast recursion, and numerical validation have all been implemented. Unsupported combinations must fail explicitly before sampling. Diagnostics must not silently substitute a simpler covariance or impute unavailable future observations.

Promotion from a small example to the full $256\times7600$ experiment requires measured memory and per-iteration timing, agreement with a dense tiny-system reference, and successful chain diagnostics. An iterative linear solve defines an approximation unless its error is controlled to the declared numerical target; the software must record achieved residuals and reject numerically unreliable draws.

## Mathematical and implementation checks

The later test suite should verify the following identities and limiting cases before benchmarking:

1. Zero anisotropy/nonstationarity/heteroskedasticity reproduces the baseline covariance and integrated target.
2. Nonstationary covariance matrices are symmetric positive definite at varied geometries; test diagonal amplitude conventions explicitly.
3. Matérn practical-range conversion is consistent with the implemented kernel parameterization.
4. Complete-residual AR whitening agrees with dense $S_\rho\otimes R$ likelihood evaluation, including determinants.
5. Latent-AR FFBS moments agree with direct conditioning in a small Gaussian state-space example.
6. The latent-AR model at $\rho=0$ agrees with the corresponding iid spatial-latent model after integration.
7. Full-conditional and valid blocked samplers agree on small examples; incorrect inverse-gamma shapes are detectable through posterior moments.
8. Whole-field and local Student-t weights have their stated Gamma conditionals and converge to Gaussian behavior for large degrees of freedom.
9. General weighted coefficient operators agree with explicit tiny Kronecker-sum matrices and their exact Gaussian draws.
10. Each forecast retains one parameter draw and the appropriate terminal latent state throughout its trajectory.

## Sources and attribution

The covariance constructions and prior principles draw on the following primary literature. The model-specific conditionals and solver routing above are derivations for the proposed ESN extension and require the numerical verification described in this blueprint.

- Paciorek, C. J., and Schervish, M. J. (2006). Spatial modelling using a new class of nonstationary covariance functions. *Environmetrics*, 17, 483--506. [DOI](https://doi.org/10.1002/env.785).
- Fuglstad, G.-A., Simpson, D., Lindgren, F., and Rue, H. (2019). Constructing priors that penalize the complexity of Gaussian random fields. *Journal of the American Statistical Association*, 114, 445--452. [DOI](https://doi.org/10.1080/01621459.2017.1415907).
- Porcu, E., Bevilacqua, M., Schaback, R., and Oates, C. J. (2024). The Matérn model: a journey through statistics, numerical analysis and machine learning. *Statistical Science*, 39, 469--492. [DOI](https://doi.org/10.1214/24-STS923).
- Coube-Sisqueille, S., Banerjee, S., and Liquet, B. (2026; online 2025). Nonstationary spatial process models with spatially varying covariance kernels. *Journal of Computational and Graphical Statistics*, 35, 173--183. [DOI](https://doi.org/10.1080/10618600.2025.2516020).
- Jylänki, P., Vanhatalo, J., and Vehtari, A. (2011). Robust Gaussian process regression with a Student-t likelihood. *Journal of Machine Learning Research*, 12, 3227--3257. [Article](https://www.jmlr.org/papers/v12/jylanki11a.html).
- Carter, C. K., and Kohn, R. (1994). On Gibbs sampling for state space models. *Biometrika*, 81, 541--553. [DOI](https://doi.org/10.1093/biomet/81.3.541).
- Murray, I., Adams, R. P., and MacKay, D. J. C. (2010). Elliptical slice sampling. *AISTATS*, PMLR 9, 541--548. [Article](https://proceedings.mlr.press/v9/murray10a.html).
