# KS model registry

## Reading this registry

This is a custom, literature-grounded comparator study using the prepared KS data and the existing Python numerical stack. The code paths below are authoritative descriptions of the implemented families; reference papers motivate model classes and adaptations. No method is called an exact reproduction of a fully observed published KS experiment.

**Track A** is strict interleaved unseen-site forecasting (39 observed inputs, 13 validation and 12 test coordinates). **Track B** observes and forecasts all 64 panel sites. Every fit uses one trajectory's permitted training interval and delivered scalar transform. The initial candidates and resource allocation are recorded in [`ks_experiment.json`](../../configs/applications/ks_experiment.json); winners remain pending until the runner writes its frozen selection. The configuration consists of structured proposals, not a Cartesian grid or claimed optima. The [protocol](ks_protocol.md) specifies information boundaries, history 128, horizon 100, schedules, the 4-hour active-wall/8-worker-hour allocation, and the measured 1.5–3.5-hour run estimate. [Reproduction instructions](ks_reproduction.md) explain execution and resume.

| Family ID | Tracks | Fitted mechanism | Predictive output | Unseen-site rule |
|---|---|---|---|---|
| `bayes_stesn` | A, B | Accepted Bayesian spatial tensor readout with frozen causal ESN features | Joint posterior predictive paths | Existing coordinate basis plus joint spatial residual covariance |
| `separable_gp` | A, B | Separable Matérn GP fitted by marginal likelihood | Joint Gaussian paths conditional on fitted hyperparameters | GP conditional spatial process, preserving temporal covariance |
| `basis_gru` | A, B | Coordinate-basis GRU, staged MSE/network and Gaussian residual fitting | Ancestral joint Gaussian-innovation paths with fixed fitted parameters | Same coordinate basis and conditional spatial innovations |
| `training_mean` | A, B | Per-observed-site temporal training mean | Point only | Periodic piecewise-linear interpolation |
| `persistence` | A, B | Last permitted observation | Point only | Periodic piecewise-linear interpolation |
| `linear_var` | A, B | Ridge linear VAR, optional training-only POD | Point only | Output-only periodic piecewise-linear decoder |
| `ridge_esn` | A, B | Conventional sparse tanh reservoir and ridge readout | Point only | Output-only periodic piecewise-linear decoder |
| `pod_nvar` | A, B | Quadratic delay regression, optional training-only POD | Point only | Output-only periodic piecewise-linear decoder |
| `periodic_cnn` | B only | Small circular residual convolutional increment emulator | Point only | Full uniform panel only; no strict-site adaptation claimed |

In all probabilistic families, the primary point forecast is the average of the same complete paths used for scores and intervals. Seeds are not pooled into a Bayesian predictive distribution. Native residual terms describe statistical discrepancy for the clean field; no future sensor noise is added by the data adapter.

### Finite initial search

The same candidate definitions are evaluated separately in A and B; their transformations, fitted parameters and selections are never shared. The CNN is evaluated only in B. One model seed, `20261105`, governs selection. Additional seeds `20261106` and `20261107` are sensitivity analyses for selected stochastic families.

| Family | Candidates per supported track | Actual varied settings in the initial file |
|---|---:|---|
| Bayesian ST-ESN | 4 | Preserved reference plus spatial columns 20/32, PCA modes 8/16, units 32/64/128, squared coordinates 16/32/64, conditioning 80/512/1,024; same ridge/IG/lognormal prior families |
| Separable GP | 2 | Temporal Matérn smoothness 0.5 or 1.5; both start at spatial length 0.2 and temporal length 4.0 before likelihood fitting |
| Basis GRU | 4 | Basis rank 8/16/24, hidden size 32/64, coefficient lags 8/16, learning rate `1e-3` or `3e-4` |
| Linear/POD VAR | 8 | Rank 8/16/32, lags 1/2/4, spacing 1, state target, selected ridge values spanning `1e-8`–`1e-2` |
| Sparse ridge ESN | 8 | Units 256/512, radius 0.5–1.2, leak 0.2/0.5/1, input scale 0.05–1, ridge `1e-8`–`1e-2`; connectivity 0.02 and washout 128 fixed |
| POD-NVAR | 8 | Rank 8/16, one/two delayed states, spacing 1/2, increment target, selected ridge values spanning `1e-6`–`1e-1` |
| Periodic CNN | 4, B only | Width/block/epoch tuples `(16,2,20)`, `(16,2,60)`, `(32,3,20)`, `(32,3,60)`; the last uses learning rate `3e-4`, others `1e-3` |
| Each anchor | 1 | No hyperparameter selection |

These ranges summarize the exact candidate tuples; they do not authorize generating every combination. The two-stage search first uses trajectories `[0,5]` at four validation origins, then promotes two candidates per family (or the sole anchor) to six trajectories and eight origins. Native families select band-balanced CRPS; point-only families select band-balanced squared error. A 1% relative near-tie rule chooses lower predeclared complexity among eligible candidates. Screening has a 1.25-wall-hour/2.5-worker-hour allocation. All candidate attempts, actual optimization/inference effort and qualification remain in the ledger. One bounded validation-only refinement may extend a selected VAR/ESN/NVAR ridge boundary by one decade, or a selected CNN 60-epoch budget to 90 epochs; architecture and seed remain fixed. Bayesian/GRU capacity boundaries are reported as limitations of the profiled roster.

## Accepted Bayesian spatial/ST-ESN

Implementation: `applications/ks_native.py`, with existing `model.py`, `features.py`, `_reservoir.py`, `basis.py`, `gaussian.py`, `sampling.py`, `spatial.py` and `prediction.py`.

Let \(y_t\in\mathbb R^S\) be the delivered standardized observations at the permitted sites, \(\Phi\in\mathbb R^{S\times p}\) the spatial dictionary, and \(z_t\in\mathbb R^d\) the frozen causal temporal feature vector. The retained accepted reference, included unchanged as candidate `bayes_00`, has hierarchy

\[
y_t=\Phi C z_t+e_t,
\qquad e_t\overset{\mathrm{iid}}\sim
N_S\{0,\sigma^2(I_S+qK_\ell)\},
\]

\[
\operatorname{vec}(C)\mid\sigma^2,\lambda
\sim N_{pd}(0,\sigma^2\lambda^{-1}I_{pd}),
\quad \sigma^2\sim\operatorname{IG}(2,0.05),
\]

\[
\log\lambda\sim N\{\log(0.1),1.5^2\},\quad
\log q\sim N(0,1.25^2),\quad
\log\ell\sim N\{\log(0.18),0.65^2\}.
\]

The inverse-gamma convention is shape/scale, with density proportional to \(x^{-a-1}\exp(-b/x)\). The reference kernel is isotropic stationary Matérn with fixed smoothness \(\nu=3/2\), restricted to unit-circle embedded coordinates. Its length parameter is measured in raw embedding chord units. All readout coefficients, including intercept terms, receive the scalar ridge prior. The nugget is the \(\sigma^2I_S\) term; \(q\) is relative spatial variance. No horseshoe, quantile likelihood, deterministic approximation or alternative prior is silently substituted.

The spatial dictionary is the existing geometry-only radial Matérn-3/2 dictionary, with an intercept and optional coordinate terms. Its radial centers, coordinate centering/scaling and column normalization use permitted static coordinates. A new coordinate is evaluated by the same frozen function. The conservative reference `side=3`, `coordinate_terms=True` in two embedding dimensions gives 9 radial columns, 2 coordinate columns and 1 intercept, or 12 spatial columns. The basis is not trained on hidden responses.

The accepted reference feature map uses conditioning-prefix PCA scores, lagged scores, a fixed sparse leaky tanh reservoir, selected elementwise reservoir squares and optional cross-products. PCA whitening, site centering, interlayer calibration and temporal-feature centering/scaling are fitted on the conditioning prefix only. The conservative reference has 8 PCA modes, lags 1 and 2, one 32-unit layer, 16 squared coordinates, no products, and a bounded readout score map \(3\tanh(a/3)\). Raw scores still drive the reservoir. This reference has \(d=1+16+32+16=65\) temporal columns and 780 readout coefficients when paired with the 12-column spatial dictionary.

For a reservoir layer,

\[
r_t=(1-\alpha)\odot r_{t-1}
 +\alpha\odot\tanh(Wr_{t-1}+U v_t+b).
\]

The retained Bayesian reservoir uses heterogeneous leak rates and recurrent **operator-norm** scaling, with reference norm 0.9. This differs from the conventional comparator's spectral-radius scaling. Input weights, biases and selected nonlinear coordinates are fixed by named seed streams. Every feature for target time \(t\) uses observations through \(t-1\).

The KS adapter explicitly retains the delivered response scale (`response_center=0`, `response_scale=1`) instead of estimating a second response scale on the conditioning prefix. All other feature machinery, hierarchy and Gaussian algebra are retained. Reference conditioning is 80 rows, so a 6,000-row core training panel contributes 5,920 likelihood targets. The initial application candidates are:

| ID | Geometry-only radial centers | Spatial columns | PCA modes / units / squares | Temporal columns | Readout coefficients | Conditioning / usable core targets | Basis width / median range prior |
|---|---|---:|---|---:|---:|---|---|
| `bayes_00` | Existing 3×3 embedding dictionary | 12 | 8 / 32 / 16 | 65 | 780 | 80 / 5,920 | 0.3 / 0.18 |
| `bayes_01` | 17 predetermined equally spaced circle centers | 20 | 8 / 64 / 32 | 113 | 2,260 | 512 / 5,488 | 0.4 / 0.3 |
| `bayes_02` | 17 predetermined equally spaced circle centers | 20 | 16 / 64 / 32 | 129 | 2,580 | 512 / 5,488 | 0.4 / 0.3 |
| `bayes_03` | 29 predetermined equally spaced circle centers | 32 | 16 / 128 / 64 | 225 | 7,200 | 1,024 / 4,976 | 0.3 / 0.18 |

All four use lags `[1,2]`, recurrent operator norm 0.9, leak pattern `[0.2,0.5,0.9]`, input scale 0.5 and the same scalar ridge/IG/lognormal prior families. The log-range prior median is an explicitly selected hyperparameter in candidates 01/02, not an unreported change to the accepted reference. Centers and widths use only static coordinates. Larger dictionaries do not allocate a dense covariance over all readout coefficients.

History 128 passed the pre-screen state-and-feature synchronization tolerance (`atol=1e-8`, `rtol=1e-6`) in all 72 combinations of four candidates, three fixed model seeds, three pilot trajectories and both main tracks. The longest compared history was 1,024, with full permitted training replay as reference. This numerical check uses the actual feature maps and no posterior fitting or future labels; it does not guarantee synchronization on every possible future origin. See [the preflight receipt](evidence/ks_preflight.json) and [native audit](ks_native_audit.md).

Inference integrates the Gaussian readout and common variance exactly for the collapsed hyperparameter target. Adaptive random-walk proposals operate in the existing `ridge_total_scale` parameterization; warm-up adaptation stops before retained draws. This is a coordinate change preserving the normalized likelihood and prior, not a different model. Conditional Gaussian/inverse-gamma recovery yields readout/variance draws. Separable spectral sufficient-statistic algebra avoids constructing the full space-time Kronecker design or a full \(pd\times pd\) covariance.

Each complete forecast path holds a recovered readout and static hyperparameter draw fixed, generates spatially correlated innovation fields according to the retained iid-in-time residual model, and advances its own observed-site reservoir history. New-site predictions share the same spatial basis and joint residual field; hidden response histories are never supplied. All initial candidates use four sequential chains, 1,000 warm-up iterations and 2,000 retained draws per chain, with at most one continuation of 2,000 draws per chain when the qualification gate fails. A deliberately short pilot is provisional. Final qualification follows [Vehtari et al.](https://arxiv.org/abs/1903.08008): R-hat strictly below 1.01 and bulk/tail ESS at least 400 for all sampled hyperparameters, all feasible readout coefficients and 10 consequential variance/readout/one-step predictive functionals. Conditional readout diagnostics use at most 512 evenly spaced retained states per chain and a 256 MiB storage cap; that restriction is recorded. No HMC diagnostic is claimed.

The exact initial dictionaries are fixed in the configuration. Candidate order does not imply predictive rank. Any bounded validation-only refinement and the final chosen settings must be recorded by the runner; this registry does not invent a selected KS setting.

## Separable spatiotemporal GP

Implementation: `benchmark/gaussian_gp.py`, wrapped by `applications/ks_native.py`.

The observation model is

\[
y_t(s)=m+f_t(s)+\epsilon_t(s),\qquad
\operatorname{Cov}\{f_t(s),f_{t'}(s')\}
=\sigma_f^2 k_s(c(s),c(s'))k_t(t,t'),
\]

with independent nugget variance \(\sigma_e^2\). The spatial kernel is Matérn on embedded coordinates; the implementation normalizes those coordinates by the maximum observed coordinate span. Thus its reported spatial length differs in units from the Bayesian covariance's raw-embedding length. Temporal Matérn smoothness 1/2 or 3/2 has an exact one- or two-dimensional state representation. Eigenspatial decomposition yields independent small Kalman filters, avoiding a dense \((TS)\times(TS)\) covariance. The present 39/64-site spatial matrices are small and exact; no undeclared inducing-point approximation is introduced.

The fitted center and marginal-likelihood parameters use permitted training responses only. L-BFGS-B qualification, optimizer continuation, boundaries and finite likelihood are retained. The initial two candidates vary temporal smoothness between 0.5 and 1.5, holding initial spatial length at 0.2 and initial temporal length at 4.0. Each allows 60 optimizer iterations and one bounded continuation of up to 120, with `ftol=1e-8`, `gtol=1e-5`. Signal/noise and length parameters are learned from the training likelihood. Forecasts condition only on the explicitly supplied causal window, initialized from its stationary state prior, with fitted parameters fixed.

The predictive sampler propagates observed-mode conditional states and the conditional unseen-site residual process jointly across time. It retains the fitted nugget in the clean-field predictive target as statistical discrepancy. Its uncertainty is **conditional on estimated hyperparameters**; it is not full Bayesian parameter uncertainty. Primary point scores average returned complete paths even though an analytic conditional mean also exists.

## Probabilistic coordinate-basis GRU

Implementation: `benchmark/basis_gru.py`, wrapped by `applications/ks_native.py`.

A geometry-only coordinate basis is normalized so \(\Phi^T\Phi/S=I\). Training response coefficients are \(a_t=\Phi^T y_t/S\) on the network's training-standardized scale. A single-layer GRU maps a fixed coefficient history of length \(L\) to next-step coefficients; a linear readout decodes its final hidden state. The field mean is the same frozen coordinate basis evaluated at observed or new coordinates.

Network training minimizes projected one-step MSE with AdamW. An internal chronological tail of the permitted training interval selects the epoch using an autonomous forecast; the selected epoch count is refitted on all permitted training rows. Center/scale used by this stopping fit exclude that internal tail. The application still uses official validation to select the complete family configuration. There is no post hoc uncertainty calibration on validation/test labels.

After mean fitting, a Matérn spatial-plus-independent-nugget Gaussian innovation covariance is fitted by likelihood to one-step optimization residuals. This is staged MSE then Gaussian likelihood, not joint maximum likelihood or Bayesian neural inference. Innovations are independent over time conditional on the path history. Generated observed-site values are projected into the next coefficient context; new-site innovations are drawn conditionally on the same observed-site innovation field. The inherent projection is an explicit part of this basis-GRU model, unlike the output-only decoder of conventional vector controls.

The application uses fixed fitted neural/covariance parameters and coherent ancestral paths. It replaces the generic comparator's separate Rao–Blackwell point estimate with the mean of those returned paths for primary scoring. Network computation is float32; covariance algebra, path outputs and scores are float64. The four initial `(rank, hidden size, coefficient lags, learning rate)` tuples are `(8,32,8,1e-3)`, `(16,32,8,1e-3)`, `(16,64,16,1e-3)` and `(24,64,16,3e-4)`. All use basis width 0.25, at most 80 stopping-fit epochs, patience 12 and a 20% chronological training-only stopping tail, followed by full-training refit. Gaussian residual optimization uses at most 40 iterations plus one continuation of up to 80. Actual selected/refit epoch counts are recorded per fit. This is a custom matched recurrent comparator; [Vlachas et al.](https://arxiv.org/abs/1910.05266) motivate recurrent forecasting comparisons, not this exact probabilistic architecture.

## Shared conventional decoder and anchors

Implementation: `applications/ks_baselines.py`.

For strict forecasting, all conventional vector/POD controls use the same parameter-free periodic piecewise-linear decoder. Query coordinates interpolate between their two adjacent permitted sites on the circle, including the seam. The weights are nonnegative and sum to one; queries at observed sites reproduce those values. The decoder uses only coordinates, with no response fitting or per-method tuning.

The training-mean anchor estimates a separate temporal mean at each permitted observed site. Persistence repeats the final permitted observation. Strict persistence therefore includes interpolation-only spatial extrapolation from the observed panel. The output-only decoder never changes the native recurrent state. On Track B it is the identity at all 64 evaluation sites. Both anchors remain in final tables and define no predictive distribution.

All ridge controls minimize

\[
\frac1N\lVert XW-Y\rVert_F^2+
\lambda\sum_{j>0,k}W_{jk}^2,
\]

where column 0 is an explicit unpenalized intercept. The implementation accumulates \(X^TX/N\) and \(X^TY/N\) in float64 chunks. The numerical penalty therefore has a mean-loss convention; it is not the same numerical parameter as a summed-loss ridge penalty or the Bayesian prior precision without accounting for likelihood and scaling.

## Linear VAR and POD-VAR

The training temporal mean is removed at each permitted input site. If `rank` is smaller than the observed dimension, SVD of the permitted full-training centered panel determines a POD basis; otherwise the state is the full centered observed vector. A linear readout predicts the next state from one, two or four lagged states, with optional explicit lag spacing. The default target is next state; increment prediction is supported only when recorded in the configuration.

For \(r\) latent coordinates, \(k\) delayed states and spacing \(s\), target time \(t\) uses states \(t-1,t-1-s,\ldots,t-1-(k-1)s\). The first usable target is \(1+(k-1)s\), and the feature dimension is \(1+rk\). Forecasts advance the predicted native or latent state. POD reconstructions and the periodic decoder are output operations; reconstructed values are not silently projected back into the state.

Starting ranges are lags 1/2/4, POD ranks 8/16/32 subject to the input dimension, and a broad logarithmic ridge range. The exact structured proposals and ridge values are set by the finite run budget. This is a regularized linear control, not a PDE-informed integrator.

## Conventional sparse ridge ESN

The conventional comparator has a random sparse tanh reservoir, fixed input weights and bias, scalar leak, and a ridge readout. Its random recurrent matrix is rescaled by its **spectral radius**. Input weights and input bias use the declared input scale. Its default target is the next observed-site state; an increment variant must be explicitly labeled.

The default readout is \([1,r_t,r_t\odot r_t]\), giving \(1+2R\) features for \(R\) reservoir units. The implemented alternative replaces zero-based indices 1,3,5,... by their squares and retains an intercept, giving \(1+R\) features. The same feature map is used in training and autonomous prediction. With 256/512 units the default readout has 513/1,025 features; its float64 Gram allocation is about 2.0/8.0 MiB before solver work arrays. Training targets start after the declared washout; state synchronization at each forecast origin uses only the supplied history.

[Pathak et al.](https://doi.org/10.1103/PhysRevLett.120.024102) describe linear/quadratic reservoir readouts. Their [author MATLAB training code](https://github.com/jdppthk/parallelized-reservoir-computing/blob/master/KSBasicSingleReservoir/train.m) uses alternating squares, a summed Gram penalty and no explicit intercept. Our explicit intercept, loss normalization, finite search and strict periodic decoder are declared adaptations. [Vlachas et al.](https://arxiv.org/abs/1910.05266) provide additional sparse-reservoir and streamed-regression context. This comparator is distinct from a deterministic-readout ablation of the accepted Bayesian architecture.

Initial proposals use reservoirs 256 or 512 (1,024 only if measured feasible), spectral radius about 0.5–1.2, leak about 0.2–1, input scale about 0.05–1, sparse connectivity and logarithmic ridge values. One reservoir seed is predeclared for screening; additional fixed seeds assess sensitivity, not seed selection.

## Quadratic delay regression / POD-NVAR

This comparator uses the same training-only centering/POD policy as the linear control. Its feature function is

\[
\psi(v)=\bigl[1, v_1,\ldots,v_d,
\{v_i v_j:1\leq i\leq j\leq d\}\bigr],\qquad
p=1+d+d(d+1)/2,
\]

where \(d=rk\) concatenates \(k\) delayed latent/observed states. The same function sets the intercept to 1 in fitting and rollout. The default ridge target is the next-state increment, which is added to the most recent predicted state. Native/POD states advance directly; spatial decoding is output-only.

[Gauthier et al. (2021)](https://arxiv.org/abs/2106.07688) and the [authors' delay/quadratic increment example](https://github.com/quantinfo/ng-rc-paper-code/blob/master/LorenzConstLinQuadraticNVARtimedelay-RK23.py) motivate NG-RC. Their supplied examples do not establish this prepared KS/POD/unseen-site protocol. Training-only POD, the periodic decoder, consistent intercept and training-only scale are explicit application choices.

Initial ranks are 8 or 16, one or two delayed states, spacing one or two saved steps, degree two and a 1,024-feature cap. Rank 16 with two delays gives 561 features; rank 24 with two delays gives 1,225 and is excluded under that initial cap. Designs are generated in chunks; the fit records actual feature count, usable targets, chunk-design bytes, Gram bytes and hypothetical full-design bytes. No full-field quadratic expansion is allocated without its explicit cost check.

## Small periodic neural emulator

Implementation: `applications/ks_cnn.py`. This is a compact circular residual CNN using the existing PyTorch CPU environment. A genuine five-epoch pilot completed on pilot trajectory 0, Track B: 0.527 s fitting, 0.028 s for a 100-step autonomous forecast, 454,123,520 bytes process peak RSS and 5,233 parameters for width 16/two blocks. Its fixed-weight 100-step float32/float64 maximum difference was approximately `1.15e-6`. The pilot establishes feasibility without a GPU; its decreasing training loss is not a selected or adequately trained final result.

The network lifts one field channel by a 1×1 convolution, applies two to four small residual blocks, and maps back through a 1×1 head. A block is `GELU(h + conv2(GELU(conv1(h))))`, with circular padding, an odd small kernel and no normalization layers. The head starts at zero. Training predicts the next-state increment by mean squared error with AdamW; rollout adds that predicted increment to the current state. There are no PDE residuals, derivative inputs, conservation corrections or forecast clips.

The initial candidates use `(width, blocks, epochs)` equal to `(16,2,20)`, `(16,2,60)`, `(32,3,20)` and `(32,3,60)`, all with kernel size 5, batch size 128 and weight decay `1e-6`. Learning rate is `1e-3`, except `3e-4` in the last tuple. Each candidate trains for its fixed declared epoch count. **Official-validation autonomous lead-band MSE selects one common complete configuration, including its 20- or 60-epoch budget, for all Track B trajectories.** The study does not call the optional per-trajectory validation early-stopping API; its `patience`/`min_delta` options do not define extra candidates here. Gradients and normalization use permitted training data only. This is point-only: it receives no Gaussian interval wrapper.

The model requires explicit `ks_protocol=all_observed` metadata and the complete uniformly ordered 64-site circle. It rejects strict masked panels even if their outer shape has 64 entries. Each forecast starts from the last permitted field and then uses its own outputs; the full history remains equally available but is not artificially consumed by a Markov emulator.

[Li et al. (2021)](https://arxiv.org/abs/2010.08895) motivate Fourier neural operators, whose ordinary FFT implementation assumes a regular grid. The compact CNN is the permitted resource-conscious alternative, motivated by the periodic convolutional controls in [APEBench, Appendix H.1](https://arxiv.org/abs/2411.00180). APEBench uses its own ecosystem, solver and protocols; none replaces this data or makes this implementation an APEBench reproduction.

## Qualification and reporting obligations

Training likelihood optimization, stochastic optimization, MCMC and deterministic regression have different diagnostic requirements. Report their actual effort, convergence/qualification, parameter counts, time and peak memory. Native probabilistic models optimize validation CRPS; point-only families optimize validation squared error. Report both point and probabilistic performance without pretending these objectives are identical.

Model/configuration selection, preprocessing and uncertainty calibration never use test outcomes. There is currently no post hoc calibration wrapper in this registry. A future wrapper would need genuine training-only out-of-sample hindcasts and a separately frozen complete-pipeline objective. Report failures and ineligible distributions rather than silently clipping paths, substituting another model or dropping bad trajectories.

Claims are limited to the configurations actually evaluated under the measured budget. Better RMSE, calibration, compute cost and unseen-site prediction are separate conclusions. Completed scientific outcomes, final configurations and sensitivity results must come from the run ledger and result tables, not from this registry.
