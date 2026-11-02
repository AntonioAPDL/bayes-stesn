# Gaussian readouts, structured shrinkage, and MCMC

**Status:** implementation specification; no new fit or performance claim is made here. All normal coefficient priors in this chapter scale with the observation variance. This convention preserves the inverse-gamma updates below and must be recorded in the public API. The finite-slab and GIGG specifications are explicit adaptations to this model, not claims that the entire proposed ESN already appears in a published paper.

## Model, dimensions, and probability conventions

For times $t=1,\ldots,T$, let $y_t\in\mathbb R^n$ and let $z_t\in\mathbb R^d$ be features computed from the available history, fixed reservoir weights, and known inputs. With a fixed spatial basis $\Phi\in\mathbb R^{n\times p}$,

$$
\begin{aligned}
y_t\mid\mathcal F_{t-1},C,\sigma^2,\theta
  &\sim N_n(\Phi C z_t,\sigma^2R_\theta),\\
R_\theta&=I_n+qK_\ell,\qquad q,\ell>0,\\
C&\in\mathbb R^{p\times d},\qquad N=nT,\quad k=pd.
\end{aligned}
$$

The innovations are independent across time conditional on the past. Because $z_t$ may depend on earlier responses, the likelihood is a product of causal transition densities, conditional on initial history. The matrix notation $Y=\Phi CZ^\top+E$, where $Y=[y_1,\ldots,y_T]$ and row $t$ of $Z$ is $z_t^\top$, is an algebraic representation of that likelihood. It does not assert an unconditional matrix-normal distribution for the complete response path. In the updates below, the realized predictable design is fixed and does not depend on the parameters being updated.

Use column-wise vectorization:

$$
b=\operatorname{vec}(C),\quad X=Z\otimes\Phi,\quad
\operatorname{vec}(\Phi CZ^\top)=Xb.
$$

The complete-data derivation assumes a common positive-definite $R$ at every time. Missing observations, time-varying innovation covariance, and latent autoregressive residuals require the corresponding likelihood or latent-state update from their own modules; the tensor identities must not be reused unchanged.

Distribution conventions are

$$
\operatorname{IG}(a,b):\quad
f(v)=\frac{b^a}{\Gamma(a)}v^{-a-1}e^{-b/v},\quad v>0,
$$

and $\operatorname{Gamma}(a,b)$ uses shape $a$ and **rate** $b$. Write $C^+(0,s)$ for a half-Cauchy distribution with scale $s$.

The common coefficient hierarchy is

$$
b\mid\sigma^2,\psi\sim N_k(0,\sigma^2\Lambda_\psi^{-1}),
\qquad \sigma^2\sim\operatorname{IG}(a_0,b_0),
$$

where $\Lambda_\psi$ is positive definite. The original reference configuration uses $a_0=2$, $b_0=0.05$, $\Lambda=\lambda I_k$, and independent priors

$$
\begin{aligned}
\log\lambda&\sim N(\log 0.1,1.5^2),\\
\log q&\sim N(0,1.25^2),\\
\log\ell&\sim N(\log 0.18,0.65^2).
\end{aligned}
$$

These are reproducibility settings, not universal prior recommendations. Their meaning depends on the fixed feature and spatial-coordinate scales. All preprocessing is estimated inside the training fold and then frozen.

## Common Gaussian derivation

Define the sufficient statistics

$$
\begin{aligned}
B&=\Phi^\top R^{-1}\Phi\in\mathbb R^{p\times p},&
G&=Z^\top Z\in\mathbb R^{d\times d},\\
J&=\Phi^\top R^{-1}YZ\in\mathbb R^{p\times d},&
h&=\operatorname{vec}(J),\\
s_Y&=\operatorname{tr}(Y^\top R^{-1}Y),&
H&=G\otimes B.
\end{aligned}
$$

Then

$$
p(Y\mid b,\sigma^2,R)
=(2\pi\sigma^2)^{-N/2}|R|^{-T/2}
\exp\left[-\frac{s_Y-2b^\top h+b^\top Hb}{2\sigma^2}\right].
$$

Combining this likelihood with the normalized coefficient prior gives

$$
\begin{aligned}
A&=\Lambda+H,\qquad m=A^{-1}h,\\
Q&=s_Y-h^\top A^{-1}h,\\
s_Y-2b^\top h+b^\top Ab
 &=Q+(b-m)^\top A(b-m).
\end{aligned}
$$

No inverse is formed in software: notation such as $A^{-1}h$ means a solve. Since $\Lambda\succ0$ and $H\succeq0$, $A\succ0$ even if a basis or feature matrix is rank deficient.

### Full conditional distributions

Completing the square gives

$$
\boxed{b\mid\sigma^2,\psi,R,Y\sim N_k(m,\sigma^2A^{-1}).}
$$

At fixed coefficients, count the $k$ Gaussian prior normalizers as well as the $N$ likelihood normalizers:

$$
\boxed{
\sigma^2\mid b,\psi,R,Y\sim
\operatorname{IG}\left(
a_0+\frac{N+k}{2},\,
b_0+\frac{s_Y-2b^\top h+b^\top Ab}{2}
\right).}
$$

For numerical evaluation, use the equivalent scale

$$
b_0+\frac12\left[
\|L_R^{-1}(Y-\Phi CZ^\top)\|_F^2+b^\top\Lambda b
\right],\qquad R=L_RL_R^\top.
$$

### Exact integration and normalizers

The Gaussian integral over $b$ contributes $(2\pi\sigma^2)^{k/2}|A|^{-1/2}$. Consequently,

$$
p(Y\mid\sigma^2,\psi,R)
=(2\pi\sigma^2)^{-N/2}|R|^{-T/2}
\frac{|\Lambda|^{1/2}}{|A|^{1/2}}
e^{-Q/(2\sigma^2)}.
$$

Let

$$
a_n=a_0+N/2,\qquad b_n=b_0+Q/2.
$$

Then

$$
\boxed{\sigma^2\mid\psi,R,Y\sim\operatorname{IG}(a_n,b_n)}
$$

and integrating $\sigma^2$ gives

$$
\boxed{
p(Y\mid\psi,R)
=(2\pi)^{-N/2}|R|^{-T/2}
\frac{|\Lambda|^{1/2}}{|A|^{1/2}}
\frac{b_0^{a_0}\Gamma(a_n)}{\Gamma(a_0)b_n^{a_n}}.}
$$

This is the integrated **likelihood**; hyperpriors must still be multiplied in. For fixed dimensions its useful log kernel is

$$
\mathcal L_c(\psi,\theta)
=-\frac T2\log|R_\theta|
+\frac12\log|\Lambda_\psi|
-\frac12\log|A|
-a_n\log b_n.
$$

Normalizing terms that are constant within a chain are still needed when comparing models with different dimensions or priors. This formula alone is not a model evidence calculation because $\psi$ and $\theta$ have not been integrated out.

### Stable evaluation of $Q$

Subtraction in $s_Y-h^\top A^{-1}h$ can lose precision when the fit is strong. If $M=\operatorname{unvec}(m)$,

$$
\boxed{
Q=\|L_R^{-1}(Y-\Phi MZ^\top)\|_F^2+m^\top\Lambda m.}
$$

Both terms are nonnegative. Evaluate the residual norm in time chunks when needed. A substantially negative subtraction-based $Q$ is an error signal; silently clipping it to zero changes the posterior. Cached $YY^\top$, $YZ$, and $Z^\top Z$ avoid reprocessing the full history in ordinary covariance proposals. Use the residual identity as the accurate reference where cancellation is material.

## Separable Gaussian shrinkage and exact recovery

Assign a precision to each temporal column, allowing columns to share group precisions:

$$
C_{ij}\mid\sigma^2,D\sim N(0,\sigma^2/D_{jj}),\quad
D=\operatorname{diag}(\lambda_{g(1)},\ldots,\lambda_{g(d)}),\quad
\Lambda=D\otimes I_p.
$$

Natural groups are the intercept, bounded lagged PCA features, first reservoir layer, second reservoir layer, and squared states. A separate proper intercept variance is allowed; an improper flat prior is not covered by these normalized collapse formulas.

Whiten the prior by setting

$$
\widetilde C=CD^{1/2},\quad
\widetilde Z=ZD^{-1/2},\quad
\widetilde G=D^{-1/2}GD^{-1/2},\quad
\widetilde J=JD^{-1/2}.
$$

With eigendecompositions

$$
B=P\operatorname{diag}(\beta_i)P^\top,\qquad
\widetilde G=V\operatorname{diag}(\gamma_j)V^\top,
$$

define

$$
a_{ij}=1+\beta_i\gamma_j,\qquad
\widehat J=P^\top\widetilde JV.
$$

The generalized determinant and quadratic reduce to

$$
\frac{|\Lambda|^{1/2}}{|A|^{1/2}}
=\prod_{i=1}^p\prod_{j=1}^d a_{ij}^{-1/2},
\qquad
Q=s_Y-\sum_{i,j}\frac{\widehat J_{ij}^2}{a_{ij}}.
$$

After drawing $\sigma^2\sim\operatorname{IG}(a_n,b_n)$, draw independent $\xi_{ij}\sim N(0,1)$ and recover a joint posterior coefficient draw by

$$
\begin{aligned}
U_{ij}&=\frac{\widehat J_{ij}}{a_{ij}}
+\sqrt{\frac{\sigma^2}{a_{ij}}}\,\xi_{ij},\\
\widetilde C&=PUV^\top,\qquad C=\widetilde CD^{-1/2}.
\end{aligned}
$$

For the original ridge prior $D=\lambda I_d$, use eigenvalues $\lambda+\beta_i\gamma_j$ with the **unwhitened** $G$ and $\widehat J=P^\top JV$. The log determinant contribution becomes

$$
\frac{k}{2}\log\lambda
-\frac12\sum_{i,j}\log(\lambda+\beta_i\gamma_j).
$$

The eigendecomposition of $G$ is then cached for the entire chain. General group or column precisions require a new eigendecomposition of $D^{-1/2}GD^{-1/2}$ when $D$ changes. A spatial-covariance-only proposal can reuse the current temporal decomposition.

### Hyperparameter updates

For group precision priors, specify $\eta_g=\log\lambda_g\sim N(\mu_g,s_g^2)$. Define $\vartheta=(\eta_1,\ldots,\eta_G,\log q,\log\ell)$ and propose

$$
\vartheta'=\vartheta+L_{\mathrm{prop}}\epsilon,\qquad
\epsilon\sim N(0,I).
$$

For symmetric proposals, accept with probability

$$
\alpha=\min\{1,\exp[\mathcal L_c(\vartheta')
-\mathcal L_c(\vartheta)+\log\pi_{\log}(\vartheta')
-\log\pi_{\log}(\vartheta)]\}.
$$

The priors here are defined **on log parameters**, so there is no additional Jacobian. If a prior is instead specified on a positive natural parameter $x$, a log-space target includes $\log p_x(e^u)+u$, where $u=\log x$. Include the proposal-density ratio for nonsymmetric proposals. Adapt proposal scales during warmup and freeze them for retained sampling.

The group precisions should not be calibrated by group size alone. Under the stated independent coefficient prior, a group's conditional expected fitted-field energy is

$$
E\big[\|\Phi C_gZ_g^\top\|_{R^{-1},F}^2
\mid\sigma^2,\lambda_g\big]
=\frac{\sigma^2}{\lambda_g}
\operatorname{tr}(B)\operatorname{tr}(Z_g^\top Z_g).
$$

Use this identity and prior predictive rollouts to check group-scale choices. Copying the scalar ridge hyperprior to every group is a reference initialization, not evidence of equal regularization.

## Regularized horseshoe: group, column, and entrywise variants

The regularized horseshoe adds a finite slab to the horseshoe's global and local shrinkage [PV17]. Here the slab is in **noise-standardized coefficient units**. This is a deliberate convention; using a slab in absolute response units produces a different model and generally removes the inverse-gamma scale conditional derived above.

Let $S$ index coefficients assigned this prior, and let $r(a)$ identify the shrinkage unit containing coefficient $b_a$. Specify

$$
\begin{aligned}
b_a\mid\sigma^2,\tau,\psi,c^2&\sim N(0,\sigma^2v_a),\\
v_a&=\frac{c^2\tau^2\psi_{r(a)}^2}
{c^2+\tau^2\psi_{r(a)}^2},\\
\psi_r&\sim C^+(0,1),\qquad \tau\sim C^+(0,\tau_0),\\
c^2&\sim\operatorname{IG}(a_c,b_c),\qquad
a_c=\nu_c/2,\quad b_c=\nu_cs_c^2/2.
\end{aligned}
$$

Thus $v_a\leq c^2$. Coefficients outside $S$, such as intercepts, receive specified proper Gaussian variances $\sigma^2v_{0a}$. The full diagonal precision has entries $1/v_a$ or $1/v_{0a}$. Fixing $c$ is an allowed simpler configuration; then its sampling step is omitted.

| Shrinkage unit | Coefficients sharing $\psi_r$ | Computational consequence |
|---|---|---|
| Feature group | All spatial rows in one temporal feature group | $\Lambda=D\otimes I_p$; few hyperparameters |
| Temporal column | All $p$ spatial coefficients for one feature | Same separable algebra; many local scales |
| Individual entry | One coefficient of $C$ | Generic diagonal precision; tensor eigensolver unavailable |

These are distinct priors, not equivalent implementations. Column shrinkage asks whether a temporal feature is useful across space. Entrywise shrinkage permits spatially uneven use of that feature. For correlated reservoir features, neither selection pattern has an automatic causal interpretation.

### Correct log-coordinate target

Set $u_r=\log\psi_r$, $t=\log\tau$, and $z=\log c^2$. In these coordinates,

$$
\log v_a=z+2t+2u_{r(a)}
-\operatorname{logaddexp}(z,2t+2u_{r(a)}).
$$

Conditional on the current $b$ and $\sigma^2$, the complete relevant log target, including transformation Jacobians, is

$$
\begin{aligned}
\mathcal T(u,t,z)
={}&-\frac12\sum_{a\in S}\log v_a
-\frac{1}{2\sigma^2}\sum_{a\in S}\frac{b_a^2}{v_a}\\
&+\sum_r\left[u_r-\log(1+e^{2u_r})\right]\\
&+t-\log(1+e^{2t}/\tau_0^2)
-a_cz-b_ce^{-z}+\mathrm{const}.
\end{aligned}
$$

The $u_r$ and $t$ terms include the half-Cauchy change of variables; the inverse-gamma prior contributes $-a_cz-b_ce^{-z}$ after its Jacobian. Compute the logarithms with stable softplus/log-sum-exp routines.

Update each $u_r$, $t$, and $z$ with a univariate slice kernel or symmetric random-walk Metropolis using this target. A local update only needs its affected coefficients and its local prior. A global or slab update needs all coefficients in $S$. This specifies every RHS scale update without requiring derivatives. The ordinary horseshoe's inverse-gamma auxiliary Gibbs steps [MS16] **do not remain valid unchanged for the finite-slab RHS**.

For group or column RHS, a second valid option replaces the first line of $\mathcal T$ by the collapsed likelihood $\mathcal L_c$, retaining exactly the same transformed hyperpriors. This integrates both $C$ and $\sigma^2$. Columnwise collapsed proposals can be expensive because each accepted or rejected candidate needs a new temporal eigendecomposition. A practical alternative is to update all scales conditionally on $C$ and perform one coefficient draw per sweep. The two kernels target the same specified posterior; their mixing and cost must be measured.

For this correlated tensor design, $\tau_0$, $s_c$, and $\nu_c$ are configuration parameters calibrated against the actual training design and desired prior predictive amplitudes. An orthogonal-regression sparsity formula is not a universal default for these features.

## Generic diagonal precision: matrix-free Gaussian sampling

Entrywise RHS and GIGG need a coefficient sampler without forming the $k\times k$ precision or the $N\times k$ design. For a matrix $M\in\mathbb R^{p\times d}$,

$$
\mathcal A(M)=BMG+\operatorname{unvec}
\{\Lambda\operatorname{vec}(M)\}.
$$

Let $B=L_BL_B^\top$ and $G=L_GL_G^\top$; positive-semidefinite eigenvalue roots are allowed when Cholesky does not exist. Draw independent matrices $E_0,E_1$ of independent standard normal entries and construct

$$
\Xi=\operatorname{unvec}\{\Lambda^{1/2}\operatorname{vec}(E_0)\}
+L_BE_1L_G^\top.
$$

Because

$$
\operatorname{Cov}\{\operatorname{vec}(\Xi)\}
=\Lambda+G\otimes B=A,
$$

the solution of

$$
\boxed{\mathcal A(C)=J+\sigma\Xi}
$$

has mean $\operatorname{unvec}(A^{-1}h)$ and vectorized covariance $\sigma^2A^{-1}$. This is a direct perturb-and-solve Gaussian derivation. It is mathematically exact when the linear system is solved exactly.

A finite-tolerance conjugate-gradient solve produces a numerically approximate draw. Calling that draw exact without qualification is incorrect. The implementation contract must record the solver, stopping rule, achieved residual, iteration count, and failures; use small dense reference problems to check means and covariances; and demonstrate stability under tighter tolerances. A small relative residual is useful but is not by itself a bound on all posterior functionals. If the solve fails its criterion, reject the computational result and raise an error or retry under a documented policy; never silently retain a failed draw.

One inexpensive preconditioner uses

$$
\operatorname{diag}(A)_{ij}=\Lambda_{ij,ij}+B_{ii}G_{jj}.
$$

A separable approximation may give better conditioning. Each dense Gram-based matrix-vector product costs $O(p^2d+pd^2)$, with $O(p^2+d^2+pd)$ working storage. Forming and storing $A$ would instead cost $O(k^2)$ memory. At the reference dimensions $p=172,d=657$, $k=113{,}004$; a dense double-precision $A$ alone would occupy approximately 102 GB (decimal), before factorization workspace.

The generic backend uses **ordinary conditional** updates for hyperparameters, which avoid $\log|A|$. A noisy or approximate log-determinant inserted into the collapsed target does not automatically define exact MCMC. A pseudo-marginal method would require its own valid nonnegative unbiased likelihood estimator and proof; it is not part of this plan.

### Conditional covariance update

Given $C$ and $\sigma^2$, set $E=Y-\Phi CZ^\top$. For $\theta=(\log q,\log\ell)$ with priors on these logs and coefficient priors independent of $R$,

$$
\mathcal T_R(\theta)
=-\frac T2\log|R_\theta|
-\frac{\operatorname{tr}(E^\top R_\theta^{-1}E)}{2\sigma^2}
+\log\pi_{\log}(\theta).
$$

Metropolis or slice updates use this target. If another module makes the coefficient prior depend on $R$, add that conditional prior density as well. Cache $EE^\top$ during a covariance-update block; invalidate it whenever $C$ changes.

## Optional research extension: GIGG

The GIGG prior uses a gamma group variance and inverse-gamma local variances, enabling conjugate or generalized-inverse-Gaussian scale updates [BDWPKM24]. The following is a **noise-scaled adaptation** of its normal-mixture structure. It is not the finite-slab RHS, and the source paper's coefficient variance convention is not silently substituted for ours.

For shrinkage coefficients $a\in S$, write $g(a)$ for the group and use positive variance factors $v_g$ and $w_a$:

$$
\begin{aligned}
b_a\mid\sigma^2,\tau^2,v,w
&\sim N(0,\sigma^2\tau^2v_{g(a)}w_a),\\
v_g&\sim\operatorname{Gamma}(a_g,1),\qquad
w_a\sim\operatorname{IG}(b_g,1),\\
\tau^2\mid\xi&\sim\operatorname{IG}(1/2,1/\xi),\qquad
\xi\sim\operatorname{IG}(1/2,1/s_\tau^2).
\end{aligned}
$$

The last two lines imply $\tau\sim C^+(0,s_\tau)$ [MS16]. Fix $a_g,b_g>0$ in the initial research implementation. The special case $a_g=b_g=1/2$ is the group-horseshoe choice in the GIGG construction. Estimating these shape parameters by empirical Bayes or giving them hyperpriors is a further model choice, not an omitted automatic step.

### Local and group conditionals

Combining the local inverse-gamma density with its single Gaussian coefficient yields

$$
\boxed{
w_a\mid\cdots\sim\operatorname{IG}\left(
b_{g(a)}+\frac12,\,
1+\frac{b_a^2}{2\sigma^2\tau^2v_{g(a)}}
\right).}
$$

Let $m_g$ be the number of shrinkage coefficients in group $g$. Define

$$
\operatorname{GIG}(\rho,\alpha,\chi):\quad
f(x)\propto x^{\rho-1}
\exp\left[-\frac12\left(\alpha x+\frac\chi x\right)\right],
\quad x>0.
$$

The product of the group gamma prior and its $m_g$ normal densities is

$$
p(v_g\mid\cdots)\propto
v_g^{a_g-m_g/2-1}
\exp\left[-v_g
-\frac{1}{2v_g}\sum_{a\in g}
\frac{b_a^2}{\sigma^2\tau^2w_a}\right],
$$

so

$$
\boxed{
v_g\mid\cdots\sim\operatorname{GIG}\left(
a_g-\frac{m_g}{2},\,
2,\,
\sum_{a\in g}\frac{b_a^2}{\sigma^2\tau^2w_a}
\right).}
$$

Negative GIG order $\rho$ is permitted. If a library instead samples $1/v_g$, the order changes sign and the last two parameters swap. The implementation must explicitly test that convention. An all-zero coefficient initialization can yield a degenerate boundary conditional; draw coefficients before these updates and avoid arbitrary positive floors that alter the prior.

### Global scale and auxiliary conditional

With $k_S=|S|$,

$$
\boxed{
\tau^2\mid\cdots\sim\operatorname{IG}\left(
\frac{k_S+1}{2},\,
\frac1\xi+\frac{1}{2\sigma^2}
\sum_{a\in S}\frac{b_a^2}{v_{g(a)}w_a}
\right),}
$$

$$
\boxed{
\xi\mid\tau^2\sim\operatorname{IG}\left(
1,\frac1{s_\tau^2}+\frac1{\tau^2}
\right).}
$$

The coefficient precision is $\Lambda_{aa}=1/(\tau^2v_{g(a)}w_a)$ on $S$ and the prescribed proper precision elsewhere. Draw coefficients with the generic sampler and draw $\sigma^2$ with the **full** conditional in Section 2.1. Its shape includes all $k$ noise-scaled coefficients, whereas the global GIGG update counts only $k_S$. The unregularized heavy tails and high-dimensional scale geometry make this a research branch requiring comparison with simpler priors; no predictive advantage is presumed.

## Valid complete sampling schedules

### Schedule A: collapsed separable priors

1. Initialize valid positive covariance and prior hyperparameters; build the causal design once from the training data.
2. Update group/ridge/RHS scales and covariance hyperparameters using $\mathcal L_c$ plus their correctly transformed priors. These kernels target the marginal hyperparameter posterior with both $C$ and $\sigma^2$ integrated out.
3. After the collapsed block, draw $\sigma^2\sim\operatorname{IG}(a_n,b_n)$ at the **current** hyperparameters.
4. Draw $C\mid\sigma^2,\psi,R,Y$ using the separable eigensolver. Store these jointly matched draws or use them for posterior predictive simulation.

Several hyperparameter kernels may run before recovery. However, after a collapsed update the old $C$ and $\sigma^2$ are stale relative to the new hyperparameters. Do not insert a coefficient-dependent scale or covariance update before recovering both in the order above. This is the central partially collapsed MCMC ordering constraint [VP08].

### Schedule B: ordinary conditional priors

1. Draw $C\mid\sigma^2,\psi,R,Y$ using the separable or generic Gaussian sampler.
2. Draw $\sigma^2\mid C,\psi,R,Y$ with shape $a_0+(N+k)/2$.
3. Update the prior parameters conditionally on the current $C,\sigma^2$: use the RHS log target, GIGG conditionals, or a Gaussian-prior Metropolis target.
4. Update $R$ using the conditional covariance target.

For a diagonal Gaussian prior with generic hyperparameters $\psi$, the coefficient-prior contribution to step 3 is

$$
\frac12\log|\Lambda_\psi|
-\frac{b^\top\Lambda_\psi b}{2\sigma^2}
+\log p(\psi),
$$

with the applicable Jacobians in transformed coordinates. Thus Schedule B also completely specifies uncollapsed grouped Gaussian updates. Every step uses the most recently updated state; no marginal and conditional scale formulas are mixed.

Posterior predictive paths must preserve each joint $(C,\sigma^2,R,\psi)$ draw and recursively update the reservoir from that path's history. Independently shuffling marginal parameter draws breaks posterior dependence. Observation noise is included when predicting observations and omitted only for an explicitly defined conditional mean target.

## Implementation gates and cost accounting

| Component | Required implementation contract |
|---|---|
| Algebra | One shared complete-square derivation; dense small-problem checks of determinant, $Q$, and moments |
| Scalar ridge | Cache the $G$ eigensystem; recover the existing posterior under identical inputs and seeds within numerical tolerance |
| Group/column precision | Invalidate the temporal eigensystem whenever $D$ changes; account for $O(d^3)$ refresh cost |
| Spatial covariance | Factor $R$ stably; record conditioning; never silently reinterpret an added jitter as learned noise |
| RHS | Test log-coordinate Jacobians and finite-slab variance; no ordinary-horseshoe Gibbs shortcut |
| Generic Gaussian draw | Reference dense covariance checks, solver failure handling, tolerance refinement, and honest numerical labeling |
| GIGG | Check shape/rate conventions, GIG reciprocal convention, and global-scale subset counts |
| MCMC | Separate warmup from retained draws; diagnose chains and posterior predictive quantities; measure effective samples per second |

With a changing $R$, factorization costs $O(n^3)$; the spatial eigensystem costs $O(p^3)$. A changed group precision adds $O(d^3)$ for the temporal eigensystem, and coefficient rotations cost $O(p^2d+pd^2)$. The reference dimensions are $n=256,T=7600,p=172,d=657$; increasing reservoir width raises both statistical flexibility and these computational costs. Benchmarking must include preprocessing, factorization, scale-update counts, and posterior recovery, rather than reporting only an isolated matrix multiplication.

Correlated spatial and temporal features can exchange explanatory roles, and a flexible mean can compete with a spatial residual process. Proper priors ensure a defined model but do not guarantee useful identification or rapid mixing. Report predictive calibration, field-level uncertainty, and sensitivity to prior choices alongside coefficient summaries. Baseline ridge, grouped Gaussian, grouped RHS, and entrywise RHS should be compared as separate, reproducible configurations before promoting the GIGG research branch.

## References

- **[PV17]** Piironen, J., and Vehtari, A. (2017). *Sparsity information and regularization in the horseshoe and other shrinkage priors*. Electronic Journal of Statistics, 11(2), 5018–5051. [Primary manuscript](https://arxiv.org/abs/1707.01694); [DOI](https://doi.org/10.1214/17-EJS1337SI).
- **[MS16]** Makalic, E., and Schmidt, D. F. (2016). *A simple sampler for the horseshoe estimator*. IEEE Signal Processing Letters, 23(1), 179–182. [Primary manuscript](https://arxiv.org/abs/1508.03884).
- **[BDWPKM24]** Boss, J., Datta, J., Wang, X., Park, S. K., Kang, J., and Mukherjee, B. (2024). *Group Inverse-Gamma Gamma Shrinkage for Sparse Linear Models with Block-Correlated Regressors*. Bayesian Analysis, 19(3), 785–814. [Primary manuscript](https://arxiv.org/abs/2102.10670); [DOI](https://doi.org/10.1214/23-BA1371).
- **[VP08]** van Dyk, D. A., and Park, T. (2008). *Partially collapsed Gibbs samplers: Theory and methods*. Journal of the American Statistical Association, 103(482), 790–796. [DOI](https://doi.org/10.1198/016214508000000409).

The Gaussian identities, tensor reductions, finite-slab log targets, and noise-scaled GIGG conditionals above are explicitly derived for the hierarchy in this chapter. They should be tested as properties of this implementation rather than cited as empirical performance results from the source papers.
