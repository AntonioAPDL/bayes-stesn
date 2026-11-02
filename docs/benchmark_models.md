# Benchmark models: equations, inference and supported scope

This page specifies the implemented models. The [benchmark guide](benchmark_guide.md)
describes the experiment, tuning, scoring and command-line workflow. A published
method name identifies its source of ideas; the two explicitly named adaptations
below are not claimed to reproduce the original papers' experiments.

## 1. Shared information and spatial conventions

Models receive only an observed `Panel`: responses `y[time,site]`, coordinates,
equally spaced times and site identifiers. At a forecast origin, `history` may
extend the identical fitted prefix. Parameters and transformations remain fixed;
only filtering states or lag contexts update. Future responses and responses at
never-observed sites are absent from the model interface.

Let \(O\) contain the \(n\) observed sites, \(U\) contain new sites, and let
\(\widetilde y_t=(y_t-\mu)/s_y\). The coordinate-basis methods use a global
training mean \(\mu\), not an undefined per-site intercept at \(U\). LGSSM and
basis-GRU also use a global training response standard deviation; KRR only centers.
The geometry-only Gaussian RBF dictionary includes a constant, selects centers by
deterministic farthest-point selection, and is whitened by SVD so that

\[
\Phi_O^\top\Phi_O/n=I_r,\qquad
\Phi_U=\{\phi(s)^\top:s\in U\},\qquad
a_t=\Phi_O^\top\widetilde y_t/n.
\]

The same fitted function \(\phi\) evaluates at any target coordinate. Coordinates
are shifted by their training minimum and divided by the maximum training
coordinate span. Spatial length parameters in the classical comparators therefore
refer to this normalized geometry. The underlying Bayesian ESN retains its own
declared `CovarianceSpec` conventions; these are not silently overwritten.

For distance \(d\), the spatial kernels available to classical comparators are

\[
M_{1/2}(d;\ell)=e^{-d/\ell},\qquad
M_{3/2}(d;\ell)=(1+\sqrt3d/\ell)e^{-\sqrt3d/\ell},
\]

\[
M_{5/2}(d;\ell)=\left(1+\sqrt5d/\ell+5d^2/(3\ell^2)\right)e^{-\sqrt5d/\ell}.
\]

`Forecast.mean` has axes `[lead,site]`; `samples`, when available, has axes
`[path,lead,site]`. Duplicate queried coordinates refer to the same observation
variable and share a draw. Point methods do not receive invented uncertainty.

| Registry ID | Forecast at O | Forecast at U | Native uncertainty |
|---|---|---|---|
| `persistence` | Yes | No | None |
| `seasonal_naive` | Yes | No | None |
| `spatial_persistence` | Yes | Yes | None |
| `basis_lgssm` | Yes | Yes | Gaussian states and observations at fitted parameters |
| `separable_stgp` | Yes | Yes | Joint Gaussian conditional at fitted hyperparameters |
| `basis_krr` | Yes | Yes | None |
| `bayes_esn_ridge` | Yes | Yes | Joint Bayesian posterior prediction |
| `eqesn_rollout` | Yes | No | Actual finite reservoir ensemble |
| `dcrnn` | Yes | No | None; declared diffusion-GRU adaptation |
| `basis_gru` | Yes | Yes | Ancestral observations at fitted parameters |

## 2. Persistence controls

Here \(o\) denotes the last observed time; the API's `origin_index` is instead
the number of available rows, which is the index of the first future row.
At origin \(o\), persistence predicts \(\widehat y_{o+h}=y_o\). Seasonal naive
repeats the final observed cycle of length \(P\), including horizons longer than
one cycle. A supplied period must be known or selected without future data.
Otherwise the implementation takes periodogram candidates from the first
principal component of a 70% training prefix, checks them on the remaining
training tail, and retains persistence unless a candidate improves its MSE by
at least 1%. Default candidate periods span 8–200 rows when the prefix permits.

The separate spatial-persistence control holds the reconstructed last field fixed:

\[
\widehat y_{o+h}(U)=\mu\mathbf1+
K_{UO}(K_{OO}+\lambda I)^{-1}(y_o(O)-\mu\mathbf1).
\]

Its kernel factorization is cached. This spatial extension is a separate model,
not a hidden modification of naive persistence.

## 3. Coordinate-basis LGSSM

On the standardized response scale,

\[
a_0\sim N(0,v_0I_r),\qquad
a_t=Aa_{t-1}+\eta_t,\quad \eta_t\sim N(0,Q),\quad Q=\operatorname{diag}(q),
\]

\[
\widetilde y_t(S)=\Phi_Sa_t+\epsilon_t(S),\qquad
\epsilon_t(S)\sim N(0,R I),\qquad \|A\|_2\le c<1.
\]

The scalar measurement variance defines the model at both observed and new
sites. The spectral-norm constraint is stronger than a spectral-radius bound;
it is the declared fitted model, not a post-fit alteration.

Kalman filtering uses the reduced observations
\(b_t=\Phi_O^\top\widetilde y_t/n\), whose measurement covariance is \(R I_r/n\).
The full observation log likelihood adds the Gaussian density of the orthogonal
residual and the Jacobian term \(-Tr\log(n)/2\). Thus filtering costs depend on
rank rather than factoring an \(n\times n\) innovation covariance every time.

The E-step obtains exact RTS smoothed moments. Define

\[
S_{00}=\sum_{t=1}^{T-1}E[a_{t-1}a_{t-1}^\top\mid y],\quad
S_{10}=\sum_{t=1}^{T-1}E[a_ta_{t-1}^\top\mid y],\quad
S_{11}=\sum_{t=1}^{T-1}E[a_ta_t^\top\mid y].
\]

The generalized M-step minimizes
\(\operatorname{tr}\{Q^{-1}(S_{11}-AS_{10}^\top-S_{10}A^\top+AS_{00}A^\top)\}\)
over the spectral-norm ball, using the unconstrained solution when feasible and
projected gradient otherwise. Then

\[
q_j\leftarrow\frac{[S_{11}-AS_{10}^\top-S_{10}A^\top+AS_{00}A^\top]_{jj}}{T-1},
\qquad
R\leftarrow\frac{\sum_tE[\|\widetilde y_t-\Phi_Oa_t\|^2\mid y]}{Tn},
\]

subject to the declared positive variance floor. The implementation records the
observed likelihood sequence, rejects a materially decreasing update, and requires
the convergence criterion for a qualified fit. Forecasts simulate the filtered
terminal state, state innovations and observation errors jointly. These intervals
omit parameter-estimation uncertainty. The [Dynamax LGSSM documentation][lgssm]
provides a primary implementation reference for Kalman/EM methodology; the fixed
coordinate decoder and norm constraint here are this repository's specification.

## 4. Separable spatiotemporal GP

\[
y(s,t)=\mu+f(s,t)+\epsilon(s,t),\qquad
\operatorname{Cov}\{f(s,t),f(s',t')\}
=\sigma_f^2M_{\nu_s}(\|s-s'\|;\ell_s)M_{\nu_t}(|t-t'|;\ell_t),
\]

with independent observation noise of variance \(\sigma_e^2\). The constant
\(\mu\) is the training observation mean, or an explicitly supplied `center`;
it is held fixed during covariance fitting. The four positive covariance
parameters are fitted by normalized marginal likelihood using bounded L-BFGS-B;
discrete smoothness choices are external tuning choices. Optimizer status,
boundary hits and likelihood are saved. A successful local optimizer exit is
not a guarantee of the global likelihood maximum. `fit_hyperparameters=False`
supports declared fixed-parameter references. The resulting predictive Gaussian
law conditions on fitted hyperparameters and omits their estimation uncertainty.

The temporal choices have exact state representations. For spacing \(\Delta\),
\(\nu_t=1/2\) uses \(F=e^{-\Delta/\ell_t}\), \(P_\infty=1\),
\(Q_t=1-F^2\). For \(\nu_t=3/2\), let \(a=\sqrt3/\ell_t\):

\[
F=e^{-a\Delta}
\begin{bmatrix}1+a\Delta&\Delta\\-a^2\Delta&1-a\Delta\end{bmatrix},\qquad
P_\infty=\operatorname{diag}(1,a^2),\qquad Q_t=P_\infty-FP_\infty F^\top.
\]

Rotating by the spatial correlation eigenvectors creates independent one- or
two-dimensional filters. Complete panels and homoskedastic noise are required.
Per changed spatial length, cost includes \(O(n^3+Tn^2+Tnq^3)\), where \(q\le2\);
both spatial eigensystems and data rotations are cached when reusable.
[Särkkä and Hartikainen][stgp] establish the state-space approach.

For new sites, define \(W=M_{UO}M_{OO}^{-1}\) and
\(S=M_{UU}-WM_{OU}\). The exact decomposition is

\[
f_U(t)=Wf_O(t)+\xi_U(t),\qquad
\operatorname{Cov}\{\xi_U(t),\xi_U(t')\}=\sigma_f^2S M_{\nu_t}(|t-t'|;\ell_t).
\]

The independent conditional residual \(\xi_U\) is simulated as a temporal process.
Redrawing independent kriging errors at each lead would give the wrong joint law.
Dense small-case `joint_moments` verifies the filter against direct Gaussian
conditioning. Default dense gates are 512 fitted sites and 2,000 scalar variables
for the reference joint-moment routine. New-site ensemble generation also needs
a dense target residual factor; this is not a large-domain sparse GP implementation.

## 5. Coordinate-basis lag KRR / Nyström

Concatenate \(L\) past projected fields to obtain \(x_t\), then standardize every
input feature using training pairs only. With an RBF kernel \(k\), the normalized
ridge objective gives

\[
\widehat f(x)=k(x,X)(K+N_{\mathrm{pair}}\lambda I)^{-1}A_{\mathrm{next}},\qquad
\widehat y_t(S)=\mu\mathbf1+\Phi_S\widehat f(x_t).
\]

The equivalent unnormalized library penalty is \(N_{\mathrm{pair}}\lambda\),
where \(N_{\mathrm{pair}}=T-L\). The default uses randomly selected training
landmarks and whitened Nyström features; exact KRR is available below the declared
case limit. The bandwidth defaults to a training-distance median times a tuning
multiplier. Recursive rollout feeds predicted coefficients into subsequent lags.
This is a two-stage point predictor, not a latent measurement-error fit. See the
official [KRR][krr] and [Nyström][nystrom] references for the corresponding methods.

## 6. Bayesian spatial ESN and its sparse route

The benchmark adapter uses the existing causal ESN feature pipeline, coordinate
decoder and ridge prior. In matrix notation, with \(Y\in\mathbb R^{n\times T}\),

\[
Y=\Phi CZ^\top+E,\qquad E_{\cdot t}\sim N(0,\sigma^2R_\theta),\qquad
\operatorname{vec}(C)\mid\sigma^2,\lambda\sim N(0,\sigma^2\lambda^{-1}I),
\quad \sigma^2\sim\operatorname{IG}(a_0,b_0).
\]

`PriorSpec.ridge` places its declared normal prior on \(\log\lambda\);
`CovarianceSpec` supplies the covariance hyperpriors. Let
\(G=Z^\top Z\), \(B=\Phi^\top R^{-1}\Phi\),
\(J=\Phi^\top R^{-1}YZ\), \(H=\lambda I+G\otimes B\),
\(h=\operatorname{vec}(J)\), \(N=nT\), \(k=pd\), and

\[
a_n=a_0+N/2,\qquad
b_n=b_0+\tfrac12\{\operatorname{tr}(Y^\top R^{-1}Y)-h^\top H^{-1}h\}.
\]

The fully normalized collapsed likelihood is

\[
p(Y\mid\theta,\lambda)=
(2\pi)^{-N/2}|R|^{-T/2}\lambda^{k/2}|H|^{-1/2}
\frac{b_0^{a_0}\Gamma(a_n)}{\Gamma(a_0)b_n^{a_n}}.
\]

Metropolis updates use this likelihood and the hyperpriors. For each retained
hyperstate, recover \(\sigma^2\sim\operatorname{IG}(a_n,b_n)\), then
\(\operatorname{vec}(C)\mid\sigma^2,y\sim N(H^{-1}h,\sigma^2H^{-1})\).
The same joint parameter draw remains fixed along a complete forecast trajectory.
Spectral algebra avoids constructing the displayed \(pd\times pd\) matrix.
Separate covariance-scale priors retain their existing coefficients-only collapse.

### Exact algebra for the declared latent Vecchia approximation

The new operator route applies only to **Vecchia covariance + ridge prior + iid
Gaussian innovations + collapsed inference**. It accelerates the same latent
approximation, not a different approximation to observed noisy responses:

\[
B_vw=D_v^{1/2}z,\quad z\sim N(0,I),\quad
Q_v=B_v^\top D_v^{-1}B_v,\qquad R=V+qQ_v^{-1}.
\]

The unit-diagonal triangular factor contains conditional regressions on selected
predecessors. \(V\) is the separate positive diagonal nugget covariance. With
\(M=Q_v+qV^{-1}\),

\[
R^{-1}b=V^{-1}M^{-1}Q_vb,\qquad
\log|R|=\log|V|+\log|M|-\log|Q_v|,\qquad
\log|Q_v|=-\sum_i\log D_{v,ii}.
\]

The identity-interpolation solve avoids subtracting nearly equal Woodbury terms.
The operator computes \(B,J\) and the response quadratic in bounded time-column
chunks. It never creates dense spatial covariance, inverse, identity or
\(YY^\top\) arrays. Only the nugget-updated sparse LU is required for fitting;
the prior LU is lazy for explicitly requested dense-reference operations.

Forecasts append unique targets after the fitted inputs, preserve the input
reference geometry and factors, draw \(w\) by a sparse triangular solve, and add
independent nugget noise. Only generated input-site observations feed the ESN.
Target subsets need a fixed master ordering for Vecchia consistency: independently
rebuilding different target orderings can define different approximations.
[Katzfuss and Guinness][vecchia] give the broader latent/observed Vecchia framework.

This route does **not** promise linear memory or a wholly matrix-free model.
Sparse precision construction can require \(O(nm^2)\) storage; sparse LU fill
depends on geometry. Training responses, \(n\times p\) and \(n\times d\) products,
and dense \(p\times p\) and \(d\times d\) readout factors remain. AR residuals,
heavy tails, heteroskedastic dynamics, other priors and SPDE inference retain
their size-gated dense reference routes. Predictive output budgeting separately
counts retained components; `iter_predictive` batches complete paths without
independently tiling spatial sites.

## 7. Recursive E-QESN adaptation

`eqesn_rollout` follows the quadratic-reservoir construction of
[McDermott and Wikle][eqesn], with a declared recursive driver. Training-only
site scales and PCA loadings encode scores. For each independently randomized
member \(m\),

\[
x_t=(a_{t-1},\ldots,a_{t-L}),\quad
h_t^{(m)}=(1-\alpha)h_{t-1}^{(m)}+
\alpha\tanh(W_mh_{t-1}^{(m)}+U_mx_t+b_m),
\]

\[
\widehat a_t^{(m)}=[1,h_t^{(m)},h_t^{(m)\,2}]\beta_m.
\]

The readout minimizes normalized squared score loss plus ridge on non-intercept
coefficients. Each member remains fixed through its recursive forecast. The
returned distribution enumerates the actual fitted ensemble, even if a different
`n_paths` is requested; no extra members or observation noise are fabricated.
This differs from the paper's direct-horizon experiments and is not a Bayesian
posterior. PCA loadings define only fitted sites, so native new-site prediction is
unsupported. A spectral radius below one is not asserted to prove the echo-state
property for every nonlinear configuration.

## 8. Diffusion-GRU adaptation under `dcrnn`

The optional implementation uses a symmetric geographic nearest-neighbor graph,
row-normalized to \(P\), with sparse diffusion features
\(\mathcal D_K(X)=[X,PX,\ldots,P^KX]\). Each recurrent layer applies

\[
(r_t,z_t)=\operatorname{sigmoid}(\mathcal D_K([y_t,h_{t-1}])W_{rz}+b_{rz}),
\]

\[
c_t=\tanh(\mathcal D_K([y_t,r_t\odot h_{t-1}])W_c+b_c),\qquad
h_t=z_t\odot h_{t-1}+(1-z_t)\odot c_t.
\]

A shared linear head predicts the next field. Multistep training uses scheduled
sampling; training-tail stopping uses free-running forecasts and restores the
best epoch. By default that epoch count is refitted on all supplied training rows.
Forecasts use observed context once, then feed back predictions.

This is the **shared-transition autoregressive diffusion-GRU variant** inspired by
[Li et al.][dcrnn]. It does not implement their full encoder–decoder architecture
or claim reproduction of their traffic experiments. It is a fixed-site point
forecasting comparator. The sparse graph does not itself define a decoder at
never-observed coordinates.

## 9. Matched probabilistic coordinate-basis GRU

`basis_gru` is a custom control using the same coordinate basis as LGSSM/KRR:

\[
\widetilde y_t(S)=\Phi_Sg_\omega(a_{t-L:t-1})+e_t(S),\qquad
e_t\stackrel{\mathrm{iid}}\sim\operatorname{GP}(0,C_e),
\]

\[
C_e(s,s')=\tau_f^2M_{\nu_s}(\|s-s'\|;\ell_s)
+\tau_e^2\mathbf1\{s=s'\}.
\]

`g` is a GRU followed by a linear coefficient readout. Projected one-step MSE
fits the mean, a chronological free-running training tail selects the epoch,
and a selected-epoch refit uses all supplied training observations by default.
Gaussian residual likelihood then fits \((\ell_s,\tau_f,\tau_e)\) on optimization
rows only. This staged fit is neither joint maximum likelihood nor Bayesian
inference. `refit=False` keeps both mean fitting and covariance fitting restricted
to the inner optimization prefix. PyTorch's [GRU documentation][gru] defines the
recurrent primitive; this complete spatial probabilistic construction is custom.

At each forecast lead, simulate a joint innovation field on \(O\cup U\), add the
decoded mean, and project generated observations at \(O\) into the next context.
New-site innovations are drawn conditionally on the observed-site innovations;
independent RNG streams ensure that querying new sites cannot alter observed-site
paths. Unobserved-site history is never an input. Output means average conditional
means over a fixed `mean_paths` ancestral sample, avoiding the incorrect assumption
that a nonlinear mean rollout equals the multistep predictive expectation.
The one-step innovations are Gaussian; nonlinear recursive feedback generally
makes the multistep predictive distribution non-Gaussian.

The covariance fit must converge for qualification. Neural stopping records the
selected epoch and whether patience or the epoch budget stopped training; neither
is reported as proof of neural optimizer convergence. Forecasts exclude parameter
uncertainty. Default caps are 512 fitted sites and 1,024 sites in the joint
prediction domain. The dense spatial covariance is an explicit limit, not a
scalable GP claim.

## 10. Verification and reproducibility contracts

The focused tests check dense GP likelihood and conditional moments for both
temporal smoothnesses; new-site temporal path covariance; full/reduced Kalman and
RTS lag-moment parity; monotone constrained EM; exact/full-landmark KRR agreement;
joint spatial GRU innovation moments; and portable numeric save/load. Sparse
workflow tests independently check determinant/statistic parity, joint coefficient
and variance moments, latent triangular sampling, checkpoint continuation, and
execution with dense spatial materialization APIs forbidden.

The sampling APIs use named global path seeds. Horizon/path prefixes, duplicate
coordinates and resumed numeric artifacts are checked explicitly. All neural
modules import PyTorch lazily and serialize numeric weights without pickle.
Classical fitted objects expose `fit_classical`, `to_state`, `save` and
`load_classical`; the optional basis-GRU exposes corresponding dedicated functions.
The registry provides the common dispatch used by the benchmark runner.
Classical configuration names are checked against each model's explicit schema
during fitting and loading; integer budgets and dimensions must be positive
integers, so misspellings and silently truncated fractional ranks are rejected.

## 11. Implemented and deferred comparison scope

The table above is the callable model roster. The machine-readable
[registry](benchmark_registry.yaml) separately marks external advection-SPDE,
FNO/DST, cross-scale reservoir, DiffSTG and pretrained Chronos extensions as
`deferred_optional`. They require their own dependency, information-budget and
forecast-law checks before integration. The complete-grid comparison track is
also deferred. These features are not established by the current unit tests or
the sparse observed-panel development pilot. See
[optimization and scope](optimization_and_scope.md) for resource limits and
the distinction between implementation verification and empirical evidence.

[lgssm]: https://probml.github.io/dynamax/notebooks/linear_gaussian_ssm/lgssm_learning.html
[stgp]: https://proceedings.mlr.press/v22/sarkka12.html
[krr]: https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html
[nystrom]: https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html
[eqesn]: https://arxiv.org/abs/1708.05094
[dcrnn]: https://arxiv.org/abs/1707.01926
[vecchia]: https://arxiv.org/abs/1708.06302
[gru]: https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html
