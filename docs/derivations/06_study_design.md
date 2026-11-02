# Preregistered simulation and comparison plan

**Status:** proposed design, to be locked before evaluation. Numerical constants below are starting specifications, not optimized settings or results. This chapter separates reproduction of the existing V3 study from new experiments testing statistical improvements.

## Nonlinear generating mean and its limitations

The audited `spatiotemporal_esn_v3/simulate_data.py`, function `simulate`, generates a moving field independently of the fitted reservoir. On $s=(x,y)\in[0,1]^2$, its two phases evolve simultaneously:

$$
\begin{aligned}
\theta_{t+1}&=[\theta_t+0.115+0.035\sin\theta_t+0.018\sin\phi_t]_{2\pi},\\
\phi_{t+1}&=[\phi_t+0.078+0.020\sin(2\theta_t-\phi_t)]_{2\pi}.
\end{aligned}
$$

Discard the first 500 phase updates; each recorded time uses the next updated phases, matching the source indexing. Define

$$
\begin{aligned}
c_{1t}&=(0.50+0.27\cos\theta_t,\ 0.50+0.23\sin\theta_t),\\
c_{2t}&=(0.50+0.28\cos(\phi_t+0.70),\ 0.50+0.25\sin\phi_t),\\
d_{1t}&=0.110+0.014\sin\phi_t,\qquad
d_{2t}=0.135+0.012\cos(\theta_t-\phi_t).
\end{aligned}
$$

The structured mean is exactly

$$
\begin{aligned}
f_t(s)={}&(1.25+0.20\sin\phi_t)
e^{-\|s-c_{1t}\|^2/(2d_{1t}^2)}\\
&-(1.05+0.15\cos\theta_t)
e^{-\|s-c_{2t}\|^2/(2d_{2t}^2)}\\
&+0.40\sin\{2\pi(1.15x-0.70y)-\theta_t\}
\cos(\pi y+\phi_t)\\
&+0.30\tanh\{9[x-0.50-0.16\sin\phi_t
-0.12\sin(2\pi y+\theta_t)]\}\\
&+0.12\sin\{3\pi(x+y)+\theta_t-2\phi_t\}.
\end{aligned}
$$

The original observations are $y_t=f_t+w_t+\epsilon_t$, with independent timewise $w_t\sim N(0,0.065^2K)$, $\epsilon_t\sim N(0,0.05^2I)$, and Matérn-3/2 $K$ with length parameter $0.085$. Its initial phases are fixed at $(0.30,1.10)$: changing its random seed changes innovations and potentially geometry tie-breaking, but not this deterministic mean evolution. It therefore does not establish robustness across dynamical systems. Its residual covariance also matches the fitted baseline family.

## Paired scenarios and replication

Use a fixed $25\times25$ evaluation grid, nested geometry-only input designs of 128 and 256 sites, and the same unobserved rectangle $[0.60,0.84]\times[0.12,0.36]$. Freeze geometry before simulation; retain a common unobserved-site set outside the 256 inputs for comparisons.

Each independent replication draws $\theta_0,\phi_0\overset{\mathrm{iid}}\sim U(0,2\pi)$, applies the same burn-in, and receives separately named phase, spatial-innovation, nugget, mixture-weight, reservoir, and MCMC seed streams. Use root seed 20261102 and immutable spawn keys identifying stage, replication, and component; model-specific streams additionally identify the model. Dynamics coefficients remain fixed initially. The same phase path, geometry, and base random variates are paired across methods and, where meaningful, scenarios. Distinct replication identifiers use independent streams. A manifest records stream derivation and hashes; changing a model's RNG consumption must not change its data.

Let $a=0.065$, $b=0.05$, $\ell=0.085$, and let $K_A$ denote the Matérn kernel using metric $A$ in the covariance chapter.

| Scenario | Prespecified stochastic component |
|---|---|
| Matched Gaussian | Independent $w_t\sim N(0,a^2K_I)$ and $\epsilon_t\sim N(0,b^2I)$ |
| Anisotropic Gaussian | Replace $K_I$ by $K_A$, with $A=\exp\{\left(\begin{smallmatrix}0.55&0.25\\0.25&-0.55\end{smallmatrix}\right)\}$; determinant one |
| Latent spatial AR | $w_1\sim N(0,a^2K_I)$ and $w_t=0.8w_{t-1}+\eta_t$, $\eta_t\sim N(0,a^2(1-0.8^2)K_I)$; nugget remains iid |
| Static heteroskedasticity | Keep Gaussian $w_t$; nugget SD $b\exp[g(s)]/\sqrt{\overline{\exp(2g)}}$, $g(s)=0.7\sin(2\pi x)\cos(\pi y)$; the bar averages the fixed full grid |
| Whole-field Student-$t$ | One $\omega_t\sim\operatorname{Gamma}(5/2,5/2)$ per time; conditional covariances $\operatorname{Cov}(w_t\mid\omega_t)=(3/5)a^2K_I/\omega_t$ and $\operatorname{Cov}(\epsilon_t\mid\omega_t)=(3/5)b^2I/\omega_t$ |
| Local Student-$t$ nugget | Gaussian $w_t$ as baseline; independent $\omega_{it}\sim\operatorname{Gamma}(5/2,5/2)$ and $\epsilon_{it}\mid\omega_{it}\sim N(0,(3/5)b^2/\omega_{it})$ |
| Nonstationary Gaussian | Use the valid local-geometry construction below, with constant marginal spatial SD $a$ |

The heteroskedastic scenario has RMS nugget SD exactly $0.05$ on the declared grid. The factors $(\nu-2)/\nu=3/5$ variance-match the heavy-tailed scenarios to the Gaussian comparator; Student-$t$ scale matrices must not be reported as covariances. Whole-field latent and nugget components are conditionally independent given their shared weight, not marginally independent.

For nonstationarity, use the positive-definite Paciorek--Schervish construction in [the covariance chapter](03_covariance_latent_mcmc.md), with amplitude function zero and

$$
\begin{aligned}
u(s)&=0.35\sin(2\pi x)\sin(2\pi y),\\
E(s)&=\begin{pmatrix}
0.4\sin(2\pi y)&0.25\cos(2\pi x)\\
0.25\cos(2\pi x)&-0.4\sin(2\pi y)
\end{pmatrix},\\
\Sigma(s)&=\ell^2\exp\{2u(s)I-E(s)\}.
\end{aligned}
$$

Do not substitute an arbitrary pairwise average of length scales. An optional structural-break stress case increases the first phase drift from $0.115$ to $0.150$ at the first final forecast time. It is reported separately as adaptation under distribution shift, with no model tuning on that break.

The first release evaluates matched Gaussian, anisotropy, latent AR, and whole-field $t$. The remaining scenarios belong to the full research profile after their samplers pass the computational gates. No full factorial crossing of all violations is proposed initially.

## Time splits and information policy

All intervals are zero-based and half-open. **V3 compatibility** preserves its actual protocol: conditioning $[0,600)$; prevalidation likelihood $[600,7800)$; validation $[7800,8200)$; final refit $[600,8200)$ of exactly 7,600 times; final holdout $[8200,8800)$. Its three origins are 8199, 8399, and 8599. This profile reproduces a historical study; it is not the untouched final experiment for newly selected methods.

**The new preregistered profile** uses the following common calendar:

| Partition | Times | Allowed use |
|---|---|---|
| Conditioning | $[0,600)$ | Response/PCA/feature normalization and initial states |
| Maximum fitting window | $[600,8200)$ | Coefficient and covariance likelihood |
| Validation | $[8200,8600)$ | Candidate choice and ensemble weights |
| Calibration reserve | $[8600,9000)$ | Predictive calibration only, if requested |
| Final holdout | $[9000,9600)$ | Locked evaluation |

For likelihood size $m\in\{1900,3800,7600\}$, fit exactly $[8200-m,8200)$. All budgets share the initial preprocessing and causal history used to initialize reservoir states; only the likelihood window changes. Thus this is explicitly a **parameter-learning-window** comparison, not a claim that the model saw only $m$ historical observations in total. Architecture, priors, seed sets, and spatial basis remain fixed within this learning curve. The 128/256-site contrast is a separate axis with common target locations.

The primary policy freezes parameters after this fitting window. Validation uses four consecutive 100-step recursive forecasts. Calibration, if used, follows the same frozen-parameter policy on two nonoverlapping 200-step blocks, with its horizon grouping fixed during development. Final origins 8999, 9199, and 9399 each produce 200-step recursive paths; lead 1--100 is primary, with 40 and 200 secondary. At later origins, earlier observations at input sites may update reservoir/filter states; they never update parameters. No observations inside an active forecast block are supplied to that forecast. Keep the calibration reserve unused for uncalibrated models, preserving identical final times.

A separately labeled expanding-window refit sensitivity fits $[8200-m,8600)$ after selection, giving $m+400$ likelihood times, then freezes through calibration and final evaluation. It requires forecasts and calibration generated under that same refit policy. Its results are not pooled with the primary $m$-time study.

The 400-time calibration reserve supports an initial empirical marginal calibration study. Two independent 200-step blocks are insufficient for useful distribution-free 90% or 95% simultaneous path bands: the usual conformal rank exceeds the available number of block scores. Do not convert the sites or leads into fictitiously independent blocks. A simultaneous-band study must allocate a substantially longer calibration reserve or separately justified independent calibration trajectories before final evaluation and update the calendar accordingly; it is outside the initial short-reserve profile.

## Candidate budgets and release gates

Run 10 independent pilot replications, initially at $(m,n)=(1900,128)$, using development data and separate pilot evaluation streams. Then lock 30 new final replications, their seed manifest, scenarios, selected candidates, numerical tolerances, and scoring code. Thirty is the proposed target, not a power guarantee. Resource feasibility must be settled before any final outcomes are opened; if a reduced final profile is needed, amend and timestamp the protocol first. Do not extend or stop replications according to favorable effects.

Each core scenario compares persistence, a linear lag/PCA model with lags $(1,2,4,8)$ and no nonlinear reservoir, V3 ridge, grouped Gaussian shrinkage, grouped RHS, and at most one scenario-specific covariance/likelihood challenger. This is at most six method families. Permit at most 12 explicitly enumerated configurations per scenario/size cell, including at most three per family, and at most 24 fits including ensemble members. Persistence at unobserved sites uses one declared geometry-based reconstruction fitted from input sites only; its probabilistic version resamples joint residual paths from training-only rolling origins.

Select by validation observation CRPS at the common 128 input sites, with RMSE then runtime as prespecified tie-breakers. No generating means, latent truths, or spatially withheld responses enter selection. The Gaussian learning curve spans all six $(m,n)$ cells. Other core scenarios first use $(7600,256)$. A separate architecture comparison fixes that data budget and grouped Gaussian prior, testing two-layer widths $(128,128)$, $(256,256)$, and $(384,384)$; it does not accompany a simultaneous increase in training size.

All methods receive identical authorized data and forecast targets. Use common chain-diagnostic requirements and 512 predictive paths initially; report time, memory, failed fits, effective sample sizes, and actual draws. An equal-compute sensitivity is separate from the primary statistically converged comparison. MCMC failure does not justify scoring a shorter unconverged chain or silently replacing a method.

## Scoring, uncertainty, and presentation

The primary ranking loss is observation CRPS over leads 1--100. Also report RMSE and MAE against observations, latent $f+w$, and structured $f$ separately; never label one as another. Report 90%/95% pointwise coverage and interval width, field energy scores, and spatial variogram scores with exponent $1/2$ and fixed geometry-based weights. Use the same target grid and normalization across methods. State whether joint scores evaluate one spatial field or a whole trajectory; predeclare any geometry-selected subset used for costly trajectory scores.

Compare posterior predictive and observed spatial spectra, directional semivariograms, and temporal autocorrelation at lags 1--40. These assess dependence, not just marginal interval coverage. Calibration summaries must identify calibrated versus raw posterior predictions and must not imply independent-data guarantees for a dependent time series.

Average the three final origins within each replication before calculating paired method differences. Those origins are correlated observations from one trajectory, not three independent replications. Report all 30 paired differences, their mean, and a 95% replication-level bootstrap interval using 2,000 resamples. Present scenario-specific effects rather than selecting favorable scenarios or pooling incompatible targets.

Animations use the first locked final replication and first final origin, with synchronized truth, posterior mean, predictive trajectory, and signed-error panels. Fixed scales, original units, uncertainty labels, and error-versus-lead curves follow the report specification. A preselected example is illustrative; only the full replicated scores support performance claims. The completed report must include null improvements and failures as well as successes.
