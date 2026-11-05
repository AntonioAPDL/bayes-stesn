# Full-field Bayesian KS extension

This document specifies the additional `bayes_stesn_fullfield` application in
`src/bayes_stesn/applications/ks_fullfield_native.py`. The accepted
`applications/ks_native.py` model remains unchanged. The new variant uses the
existing Gaussian likelihood, noise-scaled priors, collapsed inference engine,
and exact Gaussian/inverse-gamma parameter recovery. Its changed components
are explicitly named in each resolved configuration and fit identity.

The implementation and its tests establish mathematical and execution
contracts. They do not establish a predictive improvement. Resource pilots
with deliberately short chains remain inference-unqualified.

## Observed field and information boundary

Write `y[t]` for the delivered, scalar-standardized field at the permitted
input sites. There are 39 input sites in the strict interleaved track, 64 in
the all-observed track, and 40 in the contiguous robustness track. All these
permitted coordinates enter every configured delay vector. The 128-point
solver truth and hidden observation-panel histories are not model inputs.

`prepare_fullfield` accepts an observation `Panel`, not an evaluator or a
full data facade. The supplied panel contains the permitted training rows
and sites only. Its complete actual states remain in `prepared.data.y`.
`prepared.standardized_y` and `prepared.Y` hold the regression response:
actual states for `target="state"`, first differences for
`target="increment"`. Causal replay must use `prepared.data.y`; generic
forecast helpers that replay `prepared.standardized_y` are deliberately not
used by this adapter.

With conditioning length `c`, feature transformations use rows `[0,c)`;
likelihood targets use rows `[c,T)`. The default `c=512` leaves 5,488 targets
from the full 6,000-row training panel. The delivered scalar transformation
still belongs to the full permitted training protocol. An experiment claiming
fewer training observations must provide an independently refitted transform.

## Fixed recurrent features without PCA

Each input site's center and population standard deviation are estimated from
the conditioning prefix only. The standard deviation has the declared
`feature_scale_floor`, which is a feature-normalization safeguard rather than
an observation-noise floor. No response SVD, PCA, POD, or hidden-site
projection is used.

For delay set `L`, the reservoir driver is the concatenation of every
standardized input-site field at delays in `L`. If there are `n` input sites,
its dimension is `n*len(L)`. The state retains `n*max(L)` historical values
plus `R` reservoir values. At index `t`, the saved state and features have
observed data only through `t-1`.

The single-layer reservoir is

\[
h_t=(1-\alpha)\odot h_{t-1}
    +\alpha\odot\tanh(W h_{t-1}+U v_t+b).
\]

The independently seeded recurrent matrix has sparse Gaussian entries and is
scaled by its actual maximum eigenvalue magnitude to the requested spectral
radius. It is not scaled by its operator norm. Both quantities are recorded.
The input matrix entries have standard deviation
`input_scale/sqrt(n*len(L))`; biases are uniform on `[-0.3,0.3]`. The leak
pattern is explicitly repeated across reservoir coordinates. A degenerate
recurrent matrix fails the candidate; the code does not redraw seeds until a
matrix succeeds.

The temporal readout contains one intercept, declared raw delay terms,
reservoir states, selected squared reservoir states, and optional selected
pair products. `raw_readout_lags` independently chooses all configured raw
lags, the latest raw field, or none of the direct raw terms. This choice
does not remove any delay or site from the recurrent driver. If `D` direct
terms, `S` squares and `P` products are retained,

\[
d=1+D+R+S+P.
\]

Non-intercept readout terms are centered and scaled from prefix feature rows
`[floor(c/3),c)` and divided by `sqrt(d-1)`. The primary raw configuration
has no saturation of the direct readout terms; a declared `score_bound`
option is an explicit additional ablation. The reservoir nonlinearity remains
`tanh` in either case.

The fitted reservoir is a frozen feature map. `R=1000` means 1,000 recurrent
units, not 1,000 Bayesian regression coefficients. There are `p*d` learned
readout coefficients, in addition to the fixed random recurrent/input weights.

## Spatial mean dictionary

Physical coordinates are mapped to the unit circle with angle
`theta=2*pi*x/22`. The new periodic dictionary with `K` harmonics is

\[
\phi(\theta)=\left(1,
\frac{\cos\theta}{\sqrt K},\frac{\sin\theta}{\sqrt K},\ldots,
\frac{\cos K\theta}{\sqrt K},\frac{\sin K\theta}{\sqrt K}\right)^\top.
\]

Thus `p=1+2*K`. Its circle normalization is predetermined and does not use
responses or the variance of basis columns over a training arc. The same
analytic rule evaluates never-trained coordinates. It is a spatial output
dictionary, distinct from compressing reservoir input responses. Candidate
registries must report its rank explicitly: using the entire permitted input
field does not imply an unrestricted spatial mean at every possible output
coordinate. The existing radial dictionary remains an explicitly named
alternative where requested.

## Hierarchical regression

For input-site basis matrix `Phi`, fixed features `z[t]`, and coefficient
matrix `C` of shape `[p,d]`, let

\[
r_t=\begin{cases}
y_t,&\text{state target},\\
y_t-y_{t-1},&\text{increment target}.
\end{cases}
\qquad
r_t\mid C,\sigma^2,q,\ell
\sim N_n(\Phi C z_t,\sigma^2 R_{q,\ell}),
\]

\[
R_{q,\ell}=I_n+q K_\ell,
\qquad
\sigma^2\sim\operatorname{IG}(a_0,b_0).
\]

The inverse-gamma convention is shape/scale with density proportional to
`x^(-a0-1)*exp(-b0/x)`. The default remains `(a0,b0)=(2,0.05)`.
Residual vectors are conditionally independent over time. The covariance
`K` is Matérn-3/2 in Euclidean chord distance on the unit-circle embedding,
which is a restriction of a valid Euclidean covariance. There is no arbitrary
wrapped-distance substitution. The active lognormal priors for `q` and `ell`
are retained in the resolved configuration.

The two prior families are distinct:

1. **Scalar Bayesian ridge.** Conditional on precision `lambda`, every
   coefficient, including intercept coefficients, has independent variance
   `sigma2/lambda`. `log(lambda)` has the configured Gaussian prior. The
   `ridge_total_scale` sampler coordinates are a target-preserving change of
   proposal coordinates; they do not change the prior.
2. **Regularized horseshoe.** For a shrunk temporal group or column `g`,
   `psi[g] ~ half-Cauchy(0,1)`, `tau ~ half-Cauchy(0,tau0)`, and by default
   `c2 ~ IG(slab_df/2, slab_df*slab_scale^2/2)`. Its coefficient variance is

   \[
   C_{ij}\mid\sigma^2,\tau,\psi,c^2
   \sim N\!\left(0,\sigma^2
      \frac{c^2\tau^2\psi_{g(j)}^2}{c^2+\tau^2\psi_{g(j)}^2}\right).
   \]

   Group mode shares a local scale across all spatial rows and temporal
   columns belonging to `score`, `reservoir_1`, `squares`, or `products`.
   Column mode shares across spatial rows of one temporal column. The
   default unshrunk intercept group has the finite proper variance
   `sigma2*fixed_variance`, with default `fixed_variance=10`. It does not
   receive an improper flat prior. A fixed slab is an explicitly configured
   alternative. No extra group-size variance normalization is silently added:
   the declared feature normalization and group-sharing hierarchy determine
   the scale.

These priors are conditional on the validation-selected architecture and
hyperprior settings. They are not Bayesian averaging over the selection
procedure.

## Exact inference and diagnostics

The adapter uses the existing column-separable Gaussian regression engine.
Its likelihood and normalizing determinants are exact for the declared
finite basis model. It never constructs the full tensor-product regression
design or a `(p*d)` by `(p*d)` coefficient covariance. Collapsed
Metropolis-Hastings samples prior/covariance hyperparameters; conditional
Gaussian and inverse-gamma recovery samples `C` and `sigma2` jointly given a
retained hyperparameter state. Proper finite-slab horseshoe densities are
used; no ordinary-horseshoe conjugate shortcut is substituted.

Scalar ridge reuses a temporal eigensystem. Horseshoe proposals changing
column precisions generally require a new prior-whitened `d` by `d`
eigendecomposition. Large horseshoe fits can therefore be much more expensive
than equally sized ridge fits even when their stored arrays fit memory.
Resource pilots must measure that cost before admission.

Final qualification requires the retained standard of four chains,
rank-normalized split/folded R-hat below 1.01, bulk/tail ESS at least 400, and
finite MCSE. All sampled scalar hyperparameters are checked. Every readout
coefficient and ten scale/conditional-mean functionals are checked on a
declared evenly spaced retained-state subset. Increment functionals include
their persistence offsets. Restricted draw monitoring is recorded explicitly;
it does not silently omit most coefficients.

Coefficient draws are written once to a task-owned temporary memmap, then
diagnosed in coefficient chunks. The complete mapped tensor is charged as
potentially resident memory, with further working/eigensystem allocations,
against measured container headroom. `flush`, scoped `MADV_DONTNEED`, and
`POSIX_FADV_DONTNEED` are additional reductions where available. A memmap
alone is not claimed to bound cgroup memory. Temporary storage is removed
after the diagnostic summaries have been computed.

One bounded continuation uses the identical sampler target and gates when
the first diagnostic pass fails. A deliberately short resource pilot is
expected to remain unqualified. Forecast calls reject unqualified posteriors
unless `require_qualified=False` explicitly requests a diagnostic/resource
forecast; this status is carried in the output metadata.

## Autonomous predictive paths

Every path holds one joint draw of `C`, `sigma2`, covariance parameters, and
prior state fixed through the entire horizon. It generates joint spatial
Gaussian innovations at input and requested output coordinates and advances
its own generated input-site field. The residual is a statistical discrepancy
for the clean simulated field; no separate sensor-noise component is added or
removed to adjust coverage.

For the increment variant, input-site recursion adds the preceding generated
native input field exactly. At new coordinates, the offset is periodic
piecewise-linear interpolation of that same input-site field. The new-site
conditional mean is that offset plus the Fourier readout correction. The
adapter never decodes and reprojects the native input field, and never uses
hidden response histories. New-site draws are reported outputs; they are not
substituted into the observed-site state.

The explicit supplied history window initializes a zero reservoir at its
start, then replays all permitted observations in that window. The runner
must choose and report a history long enough to pass the synchronization
check before validation/test use. The implementation does not silently replay
more observations than the reported window.

`forecast_parameter_rng_policy="fit_shared_bank_v1"` shares a reproducible
bank of parameter draws across origins of a fitted model. The bank identity
binds posterior, fit, numerical source/runtime identity, model seed, bank
seed and global path index. Innovations use separate origin-specific named
streams. An explicit `parameter_seed` requests an independent parameter bank
for Monte Carlo precision checks. Increasing path count preserves existing
path indices. This policy changes cross-origin Monte Carlo dependence, not
any individual origin's posterior predictive distribution. Scores use the
empirical predictive distribution; no IID unbiased finite-ensemble correction
is claimed. `per_origin_v1` is the explicitly named alternative.

Parameter caches have independent RAM/disk caps. Eviction or a skipped cache
write regenerates exactly the same draw, rather than changing its law.
Reservoir updates are batched, with fixed per-path random streams. Batches
agree within declared floating-point tolerance; identical calls reproduce
exactly. The primary point prediction is the average of complete returned
paths, not a rollout under posterior-mean parameters.

## API and resource keys

```python
from bayes_stesn.applications.ks_fullfield_native import (
    resolve_fullfield_config, prepare_fullfield,
    fit_fullfield, load_fullfield_fit,
)

resolved = resolve_fullfield_config(config, seed=73)
prepared = prepare_fullfield(training_panel, resolved, seed=73)
fitted = fit_fullfield(training_panel, resolved, seed=73,
                      checkpoint_path="/data/project/results/fit.npz")
restored = load_fullfield_fit("/data/project/results/fit.npz")

for prediction in restored.iter_forecasts(
        history_panel, horizon=100,
        target_coordinates=target_coordinates, target_site_ids=target_site_ids,
        n_paths=256, batch_size=32, seed=91):
    consume(prediction)  # Score/save incrementally; do not retain all origins.
```

The exact same `fit_fullfield` call resumes partial chains/diagnostics or loads
a completed fit. Source, runtime, training protocol, target, prior, features,
transforms, sampler and diagnostic identities must match. Durable chain
checkpoints contain sampler/RNG state. The phase receipt prevents repeated
continuations or accidental restart under a different allowance.

| Setting | Meaning |
|---|---|
| `features.max_feature_bytes` | Conservative feature-construction gate |
| `max_forecast_bytes` | Returned path-output allocation cap |
| `max_forecast_working_bytes` | Internal readout/state/innovation block cap |
| `max_parameter_cache_bytes` | Per-fit parameter-bank RAM cap |
| `max_parameter_cache_disk_bytes` | Per-fit parameter-bank disk cap |
| `diagnostics.max_coefficient_bytes` | Entire diagnostic draw tensor cap |
| `diagnostics.coefficient_chunk_size` | Number of coefficient series per diagnostic block |
| `diagnostics.disk_reserve_bytes` | Required remaining free disk, including temporary diagnostics/cache |

These local gates supplement the runner's process/job admission and measured
RSS budget. They do not replace monitoring of other concurrent work.

## Targeted acceptance evidence

`tests/test_ks_fullfield_native.py` covers the following independent or
information-boundary risks:

- Fourier normalization on a known circle, seam continuity and serialization.
- Identical raw feature states for state and increment regressions, with
  separately correct likelihood responses.
- Both likelihoods against a tiny independently assembled dense Gaussian
  regression, including posterior mean and inverse-gamma quadratic/scale.
- Actual group-RHS inference and retained rejection of RHS by the original
  accepted adapter.
- State-path agreement with the accepted joint Gaussian simulator.
- Static parameter draws, native persistence, unseen-site interpolation and
  trajectory-identity rejection.
- Shared-bank identity, changed precision-check bank, batch invariance,
  durable cache and incompatible cache rejection.
- All-coefficient diagnostics invariant to chunk size.
- Interrupted-chain RNG continuation identical to uninterrupted inference.
- Explicit failure of incompatible model-target checkpoint reuse and
  prevention of inactive PCA aliases inflating candidate counts.

Delivered-data facade invariance remains a separate acceptance check. A
model-only API without an argument for future labels is necessary, but does
not by itself prove that the facade never read future truth from disk.
