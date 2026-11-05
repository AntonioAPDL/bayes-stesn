# KS native-model and inference audit

## Scope and source authority

This document describes the native-model portion of the **custom, literature-grounded KS experiment**. It is an implementation and resource audit, not a table of final comparative results. The prior simulation study remains unchanged. The audited repository base is `4bae378922b824e0e934a3aaf08a9d8a1a65046e`; the new adapter is [ks_native.py](../../src/bayes_stesn/applications/ks_native.py). The existing [model specification](../models.md), [Gaussian engine](../../src/bayes_stesn/gaussian.py), [sampler](../../src/bayes_stesn/sampling.py), [forecast engine](../../src/bayes_stesn/prediction.py), [separable GP](../../src/bayes_stesn/benchmark/gaussian_gp.py), and [probabilistic basis GRU](../../src/bayes_stesn/benchmark/basis_gru.py) supply the mathematical and numerical backends.

The conservative reference returned by `reference_configs()` comes from [reference_reliability.yaml](../../configs/benchmark/v03/reference_reliability.yaml). It preserves the accepted Bayesian model and provides a mandatory starting candidate. It is **not represented as the configuration selected in the earlier final simulation campaign**: the tracked `confirmation_draft.yaml` has no populated final freeze, and an absent final selection manifest cannot be reconstructed by guessing. Selection for KS uses the application’s own declared training/validation protocol.

The public provenance uses content hashes and model specifications. Private retrieval account information and access manifests are not needed here.

## Common information and scaling boundary

The input `Panel` contains only the permitted training sites and the supplied time interval. The strict track has 39 fitted sites; the all-observed track has 64. An origin supplies a separate bounded history panel. It does not silently supply the entire prefix required by the older generic benchmark adapter. The history must match the fitted site IDs, coordinate order, dataset/trajectory/protocol identity, response transform, and time cadence. No recurrent state is retained between episodes or origins by this interface. Forecast parameters remain frozen.

The application uses the verified delivered scalar transform

\[
\bar y_t(s)=\{y_t(s)-m_{\mathrm{train}}\}/s_{\mathrm{train}},
\]

specific to the trajectory and spatial protocol. Native forecasts are returned on this standardized scale; the evaluator applies that same transform’s inverse to every complete path and point summary. The Bayesian KS preparation explicitly retains response center zero and scale one on the supplied standardized data. This avoids the generic `prepare()` function’s second, conditioning-prefix response transform. Its conditioning-only PCA, reservoir calibration and feature scaling remain unchanged. The Bayesian prior and likelihood below are the accepted hierarchy, now expressed on the application’s mandated delivered scale.

Coordinates are the predetermined embedding

\[
e(x)=\bigl(\cos(2\pi x/22),\sin(2\pi x/22)\bigr),\qquad x\in[0,22).
\]

Every spatial kernel is evaluated in this Euclidean two-dimensional space. Restricting a valid Euclidean Matérn to the circle gives a positive-definite periodic covariance. The code does not insert wrapped arc distances into a Euclidean Matérn and does not apply an FFT or equally spaced convolution to the irregular 39-site input vector. The periodic input geometry, static coordinate queries, and training-fitted basis are available to unseen-site forecasts; hidden response histories are not.

The delivered facade internally constructs model and evaluation sections. The application discards evaluation outputs before model access and tests output invariance. This separation does not establish future-free physical disk access by the unchanged facade.

## Bayesian spatial/ST-ESN

### Frozen causal features and spatial readout

Let \(Y\in\mathbb R^{n\times T}\), \(\Phi\in\mathbb R^{n\times p}\), \(Z\in\mathbb R^{T\times d}\), and \(C\in\mathbb R^{p\times d}\). The reference conditioning prefix has 80 states. It fits the PCA site center and eight whitened principal-component scores, feature centers/scales, and any interlayer input calibration. These transforms are subsequently frozen.

The shallow reference reservoir uses two delayed eight-dimensional score vectors. Its input is \(u_{t-1}=(a_{t-1}^{\mathsf T},a_{t-2}^{\mathsf T})^{\mathsf T}\). With 32 reservoir units,

\[
r_t=(I-D)r_{t-1}+D\tanh(Wr_{t-1}+Uu_{t-1}+b).
\]

The recurrent matrix has independently masked Gaussian entries at connectivity 0.08 and is scaled to **operator norm** 0.9. This is not a specification of spectral radius 0.9. Input weights use scale \(0.5/\sqrt{16}\), biases are uniform on \([-0.3,0.3]\), and leak rates repeat \((0.2,0.5,0.9)\). These weights and the PCA are fixed conditional on the named model seed.

The readout features are an intercept, the 16 lagged scores bounded as \(3\tanh(a/3)\), all 32 reservoir states, and 16 selected elementwise squared states. The square-coordinate selection has its own fixed random stream; adding readout terms does not change the reservoir weights. No product terms are used by this reference. Raw PCA scores still drive the reservoir. Nonintercept readout features are centered/scaled from the latter two-thirds of the conditioning prefix, with the existing scale floor 0.04, and divided by the square root of their dimension. Thus the reference has \(d=65\) features and 48 stored state coordinates, including delayed scores. These choices remain candidates for KS screening rather than presumed optimal settings.

The accepted spatial mean rule is the existing fixed radial dictionary evaluated at input or target coordinates. With `side=3`, width 0.3 and coordinate terms, it has nine Matérn-3/2 radial terms, two embedded coordinate terms, and an intercept: \(p=12\). Its centering/scaling uses input geometry only. The same centers and transformation are evaluated at never-trained coordinates. A target’s response is never needed to construct \(\phi(s)\).

The likelihood uses targets after the conditioning prefix. For the reference, the core fit therefore uses \(T=6000-80=5920\) targets; the pilot uses \(1200-80=1120\). The lags are initialized inside the conditioning prefix. This 80-state prefix, the 128-state candidate forecast history, the solver spin-up, and MCMC warmup are distinct quantities.

### Preserved likelihood and prior

The fitted model is

\[
\bar y_t=\Phi C z_t+e_t,\qquad
e_t\mid\sigma^2,q,\ell\overset{\mathrm{iid}}\sim
N_n\{0,\sigma^2(I_n+qK_\ell)\},
\]

where \(K_\ell\) is the isotropic Matérn-3/2 correlation on the circle embedding. For column-major \(b=\operatorname{vec}(C)\),

\[
b\mid\sigma^2,\lambda\sim N_{pd}(0,\sigma^2\lambda^{-1}I),\qquad
\sigma^2\sim\operatorname{IG}(2,0.05),
\]

\[
\log\lambda\sim N(\log 0.1,1.5^2),\quad
\log q\sim N(0,1.25^2),\quad
\log\ell\sim N(\log 0.18,0.65^2).
\]

The inverse-gamma density is proportional to \(v^{-a-1}\exp(-b/v)\). These proper priors are independent at the hyperparameter level. Scalar ridge applies to **all 780 reference readout coefficients, including intercept coefficients**. The generic prior configuration’s `unshrunk_groups` field does not exempt intercepts in the scalar-ridge route. The range \(\ell\) is measured in raw unit-circle chord coordinates, not physical arc length or a Lyapunov unit.

The primary data are clean simulated fields. Both the spatial residual and independent-site residual in the accepted model are retained as statistical discrepancy for that clean field. The core forecast engine historically names the latter a measurement component, but the KS application does not claim that artificial future sensor noise was added. Removing this residual component, introducing a score-dependent variance floor, or treating a deterministic rollout as Bayesian uncertainty would change the experiment.

### Exact inference without a tensor-product design

Define \(R=I+qK_\ell\),

\[
B=\Phi^{\mathsf T}R^{-1}\Phi,\quad G=Z^{\mathsf T}Z,\quad
h=\operatorname{vec}(\Phi^{\mathsf T}R^{-1}YZ),\quad
A=\lambda I+G\otimes B.
\]

The core engine evaluates the normalized collapsed likelihood integrating \(C\) and \(\sigma^2\), using the eigensystems of \(G\) and \(B\). It never materializes \(Z\otimes\Phi\) or the \(pd\)-dimensional coefficient covariance. The temporal Gram eigensystem and rotated sufficient statistics are reused under scalar ridge. A cancellation-sensitive marginal quadratic is recomputed as residual energy plus prior quadratic when needed, using bounded chunks.

For each retained hyperstate, joint posterior recovery uses

\[
\sigma^2\mid Y,\lambda,q,\ell\sim
\operatorname{IG}\{2+nT/2,\;0.05+Q/2\},\qquad
b\mid\sigma^2,Y,\lambda,q,\ell\sim
N(A^{-1}h,\sigma^2A^{-1}).
\]

The sampler is adaptive random-walk Metropolis on the collapsed hyperparameters, with all adaptation ending at warmup. The accepted `ridge_total_scale` proposal coordinate is
\(\log\kappa=\log\lambda+\log(1+q)\).
Its Jacobian between log-coordinate systems is one. It evaluates the original likelihood and priors through the inverse map; it does not put a new prior on \(\kappa\).

Final inference retains four sequential chains, 1,000 warmup iterations and 2,000 retained draws per chain, with one predeclared additional 2,000-draw continuation when the complete initial diagnostic gate fails. An infrastructure exception or changed configuration does not create another allowance.

### Strengthened KS diagnostics and recovery

The rank-normalized split/folded R-hat, bulk/tail ESS and mean MCSE implementation follows [Vehtari et al. (2021)](https://doi.org/10.1214/20-BA1221); the [primary paper](https://arxiv.org/abs/1903.08008) describes the rank-based diagnostic and local efficiency checks. The unchanged numerical criteria are R-hat strictly below 1.01, bulk and tail ESS at least 400, at least four chains and finite diagnostics. These are screening criteria, not a proof of convergence.

KS strengthens the previously accepted five-functional check:

- All sampled scalar hyperparameters are checked using all retained hyperstates.
- Every readout coefficient is checked using joint conditional \(C,\sigma^2\) recovery at 512 evenly spaced retained states per chain.
- Ten fixed functionals cover log residual scale; last-fitted readout means at three fixed sites and their spatial average; one-step predictive means at the same sites and their spatial average; and log one-step marginal predictive variance. Predictive functionals use only the permitted training prefix.

Reference coefficient diagnostics use \(4\times512\times780\times8=12{,}779{,}520\) bytes for their coefficient array, about 12.2 MiB. The explicit cap is 256 MiB; exceeding it is a resource error requiring review rather than silent monitoring reduction. The selected hyperstates remain in chain order. Recovered coefficients are not assumed to be independent scientific replicates or iid posterior draws.

Requiring every coefficient to pass is stricter than checking a few summaries. With hundreds or thousands of diagnostics, an isolated estimated R-hat/ESS threshold failure becomes more likely than with five checks. The study reports failing counts and their extrema, retains the single continuation rule, and does not weaken the gate or change diagnostic seeds to obtain a pass. A failed final gate is recorded as inference-unqualified; any point or diagnostic forecast remains clearly separated from qualified probabilistic rankings.

The KS-local phase ledger wraps the unchanged core sampler. It records initial sampling, initial diagnostics, optional continuation sampling, continuation diagnostics, and completion. Checkpoints contain exact RNG/adaptation/transition state and are bound to data, transforms, effective configuration, implementation and numerical runtime. A targeted interruption after a durable chain write reproduced the uninterrupted posterior exactly, including the one strengthened-gate continuation. Completed fits are reused without retraining. The GP and GRU currently support completed-fit reuse; an interrupted unfinished optimizer is restarted from the declared initialization/seed and must not be described as an exact optimizer-state continuation.

### Recursive joint forecast

Each predictive path draws one hyperstate and then one joint \(C,\sigma^2\) realization. These static parameters remain fixed for all 100 steps. At each step, the model generates a spatially correlated residual plus the independent-site residual on the joint input/target domain. Only that path’s generated values at the fitted input sites update the reservoir. Unseen-site values are decoded and sampled jointly but never supplied as observations to the next state. Iid residual fields are independent across time conditional on the static parameters, while the forecast remains temporally dependent through its recurrent feedback.

Named path-index streams make batches resumable. The primary point forecast is the average of **complete returned paths**. It is not a posterior-mean-parameter rollout. No path is clipped or replaced with persistence; nonfinite outputs raise an explicit failure.

## Existing separable spatiotemporal GP

This comparator retains the exact separable Matérn GP with a small temporal state representation. Its latent covariance is

\[
k\{(s,t),(s',t')\}=\sigma_f^2K_s(s,s')K_t(t,t'),
\]

with a constant training-estimated mean and independent-site/time residual variance \(\sigma_e^2\). Spatial smoothness defaults to 3/2; the existing exact temporal implementations support 1/2 and 3/2. Spatial lengths are measured after dividing embedding coordinates by the maximum observed-coordinate span. Temporal lengths use the supplied physical times, so a value of 4 means 16 saved KS intervals, not four intervals.

The spatial covariance eigensystem decouples temporal modes because the residual variance is homoskedastic. A temporal Matérn-1/2 mode has transition \(F=\exp(-\Delta/\ell_t)\); Matérn-3/2 uses the existing exact two-state transition. The marginal likelihood is evaluated by these filters, and bounded L-BFGS-B estimates spatial length, temporal length, signal SD and residual SD. Its declared continuation is a restart from the last optimizer point with unchanged tolerances, not an exact recovery of private quasi-Newton memory. Boundary hits, termination reason and projected gradient are reported. A deliberately iteration-limited pilot is not evidence that the model family is ineligible.

At a forecast origin, the KS adapter starts from the stationary GP state prior and filters exactly the supplied permitted history. It does not reuse a full-prefix terminal filter while claiming a 128-state context. For unseen sites, the conditional spatial remainder is itself a temporally correlated process. Forecasting jointly simulates this process and the filtered observed-site modes. Independent horizon-wise marginal simulation would not reproduce this model and is not used.

Uncertainty is the joint Gaussian predictive distribution conditional on fitted hyperparameters; it does not include a Bayesian hyperparameter posterior. Although the underlying implementation also supplies an analytic mean, primary KS point scores use the sample mean of the same complete paths used for empirical probabilistic scoring.

## Existing probabilistic basis GRU

This is the existing custom matched control. Its geometry-only radial basis satisfies \(\Phi^{\mathsf T}\Phi/n=I\). Projected coefficients are

\[
a_t=\Phi^{\mathsf T}(\bar y_t-c)/\{n s\},\qquad
\bar y_t(s)=c+s\phi(s)^{\mathsf T}g_\theta(a_{t-L:t-1})+e_t(s).
\]

The reference has rank eight, lag window eight and 32 hidden units. A PyTorch GRU resets its hidden state for each explicit lag window; a linear layer maps its final hidden state to the next coefficient vector. The target is the next coefficient state, not an increment. The mean network is trained by projected one-step MSE with AdamW, gradient clipping and the existing weight decay. A chronological training-only tail chooses an epoch by autonomous rollout loss, then the chosen epoch count is refitted on the complete supplied training panel. This internal stopping procedure is distinct from the official validation-based choice among family configurations.

Residual covariance is subsequently fitted by Gaussian likelihood on one-step residuals from the optimization training rows:

\[
\operatorname{Cov}\{e_t(s),e_t(s')\}
=s^2\{\sigma_f^2K_\ell(s,s')+\sigma_e^2\mathbf1(s=s')\}.
\]

This staged procedure is neither joint maximum likelihood nor full Bayesian neural inference. It is a native conditional innovation model, not an added post-hoc interval-calibration wrapper. The reference network has 4,296 trainable neural parameters and three covariance parameters; preprocessing and a fixed basis are recorded separately.

Each future observation field is sampled jointly. Generated fitted-site observations are projected back into coefficient space to advance that path’s next lag window. New-site innovations are drawn conditionally on the fitted-site innovations, preserving their spatial cross-covariance. The existing architecture’s projection is explicit; no hidden response enters it. The wrapper replaces the older returned Rao–Blackwell mean with the average of the actual complete sampled paths for the requested primary point-scoring convention.

Network operations use float32, while covariance fitting, transforms, predictive output storage and score accumulation use float64. A fixed-weight numerical check is provided by [check_ks_gru_precision.py](../../tools/check_ks_gru_precision.py). On the saved five-epoch pilot, 64 permitted-training contexts had maximum float32-versus-float64 latent readout difference \(7.59\times10^{-8}\). A 100-step autonomous zero-innovation field path, retaining the native observed-site projection, differed by at most \(1.233\times10^{-6}\) standardized response units (relative to the maximum reference magnitude: \(5.62\times10^{-7}\)). Both passed the declared elementwise `atol=1e-6, rtol=1e-5`. This checks fixed-weight inference roundoff, not equivalence of training in the two precisions or final predictive accuracy.

## Resource pilot and synchronization evidence

Each resource fit was performed in an isolated single-thread process on the separate pilot trajectories, using all 1,200 permitted training states. No core test outcome was accessed. These deliberately reduced-effort pilots set costs; they do not select final configurations or qualify final inference.

| Native family | Pilot and input sites | Actual pilot effort | Fit wall seconds | H100, 256-path forecast seconds | Peak process RSS |
|---|---|---|---:|---:|---:|
| Bayesian ST-ESN reference | Pilot 0, strict 39 sites | Four chains, 200 warmup + 400 retained; 128 conditional diagnostic draws per chain | 3.708 | 1.878, 13 validation sites | 117,669,888 bytes |
| Separable GP | Pilot 1, strict 39 sites | 15 L-BFGS-B iterations, no continuation | 1.797 | Not profiled in this receipt | 100,954,112 bytes |
| Probabilistic basis GRU | Pilot 2, all 64 sites | Five stopping epochs + five refit epochs; 21 covariance iterations | 1.537 | 0.274, all 64 sites | 446,603,264 bytes |

Forecasts were generated in 64-path batches and used the 128-state candidate history. The Bayesian short diagnostic did not satisfy final thresholds; the GP pilot exhausted its deliberate 15-iteration cap. Neither outcome rejects the family. The GRU covariance optimizer converged, but five neural epochs remain a resource pilot rather than the final declared training opportunity. Fit timings exclude interpreter/import startup; process RSS includes the loaded runtime. Scaling these costs to final budgets is an estimate requiring measured campaign accounting.

The reference Bayesian map on pilot 0 was synchronized from histories 128, 256, 512 and 1,024 versus a zero-state replay of the complete 1,200-state training prefix. At length 128, maximum first-future state difference was \(5.91\times10^{-13}\) and feature difference \(1.02\times10^{-12}\); the three longer windows matched numerically. The frozen numerical criterion was absolute tolerance \(10^{-8}\) plus relative tolerance \(10^{-6}\), applied separately to the state and readout-feature maximum differences relative to their respective full-replay maximum magnitudes.

Before screening, this feature-only assessment was extended to all four Bayesian candidate architectures, both spatial tracks, all three pilot trajectories, and all three declared model seeds (20261105, 20261106, 20261107): 72 cases in total. Every case passed at all four candidate history lengths. Thus 128 is the smallest tested common history satisfying the declared numerical criterion. The worst relative state or feature discrepancy at history 128 was \(1.39\times10^{-10}\), for the second Bayesian candidate on all-observed pilot 2 with seed 20261106. The screening-seed sweep took 4.08 seconds and the two sensitivity-seed sweeps took 7.13 seconds, with peak process RSS below 128 MB. These checks fitted only permitted-training feature maps; they ran no MCMC and read no future evaluation labels. They support the common causal context policy empirically on the declared pilot cases, without claiming a uniform guarantee across every possible input history.

The general sufficient layer contraction expression is
\(\|I-D\|_2+\|DW\|_2\).
It was approximately 1.470 for this reference map, so it did not certify a uniform contraction bound even though the empirical replay check passed. The measured spectral radius was approximately 0.402 and operator norm 0.9. Neither input-driven synchronization nor a small spectral radius proves stable autonomous observation feedback.

Private workspace evidence files are `private/bayes_pilot.json`, `private/gp_pilot.json`, `private/gru_pilot.json`, `private/gru_precision_check.json`, `private/ks_all_candidate_sync.json`, and `private/ks_sensitivity_candidate_sync.json`. The synchronization receipts retain every case, effective configuration hash, numerical-source identity, and permitted-training data identity. Portable fitted objects are under the application’s separate `pilot/` directory. The native adapter source hash for these pilots is `dfb6e8c2d9baefa54cb87b162175ac3672f8bc17f9e8951272ba6e329798fd6e`.

### Memory quantities for the reference

For the full core likelihood, a naive tensor-product regression design would occupy approximately 1.34 GiB at 39 sites or 2.20 GiB at 64 sites, before any work copies. The implementation does not allocate it. The principal exact sufficient statistics have dimensions \(65\times65\), \(n\times n\), \(n\times65\) and \(12\times12\); the spectral coefficient precision has only \(12\times65\) entries. Temporal state/history and feature arrays scale with the number of training times, not \(nTpd\). The full 780-by-780 coefficient covariance is also unnecessary.

One 64-path, 100-step output batch occupies 665,600 bytes for 13 strict validation sites or 3,276,800 bytes for 64 all-observed sites, before small working arrays. This batch dimension is an output-storage choice; each path still simulates the joint input/target domain. Combining all origins, seeds and trajectories into one in-memory path tensor is not part of the runner. Posterior files store retained hyperstates and sufficient model state, not a dense coefficient draw for every MCMC iteration.

The measured pilot RSS is evidence for these configurations only. Capacity increases, more expensive diagnostics, altered basis ranks and simultaneous workers require the application’s current container-aware admission check and safety multiplier. Host RAM is not the available job budget.

## Targeted acceptance checks and reproduction

[test_ks_native.py](../../tests/test_ks_native.py) passed ten tests in 4.26 seconds in the measured environment. They verify explicit delivered-scale preservation, exact named Bayesian path batching, GP bounded-window equivalence to the same fixed-hyperparameter GP conditioned on that window, native GRU ancestral-path equivalence, primary sample-average means, portable fitted-state reuse, rejection of mismatched sites or trajectory identities, synchronization reporting, and exact interrupted-chain recovery with only one continuation. Tiny synthetic inference runs in these tests are deliberately unqualified and are not KS scientific evidence.

From the repository with the compatible application environment active:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_ks_native.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python tools/check_ks_gru_precision.py \
  --fit "$KS_WORKSPACE/pilot/native_gru_resource.npz" \
  --output "$KS_WORKSPACE/private/gru_precision_check.json"
```

`KS_WORKSPACE` is a task-specific application workspace, not the model repository or a synchronized Drive mount. The complete experiment’s configuration, job ledger and stage commands are documented by the main application handoff. Native calls are `fit_native(model_id, training.panel, config, seed, checkpoint_path)`, `fit.forecast(window.panel, horizon, target_coordinates, target_site_ids, n_paths, seed)`, and `fit.iter_forecasts(..., batch_size=64, path_start=...)`. All returned distributions remain conditional on the validation-selected architecture and hyperparameters; they do not average over the selection procedure.
