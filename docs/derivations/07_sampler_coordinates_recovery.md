# Exact ridge coordinates and recoverable benchmark fitting

This implementation adds an opt-in proposal parameterization and durable
benchmark fit recovery. The default remains `SamplerConfig(parameterization="legacy")`.
Neither passing a short execution check nor introducing the new coordinates
establishes better mixing on the scientific benchmark. Promotion requires the
separate development comparison described below.

## 1. Original model and transformed target

For the supported iid Gaussian readout,

\[
y_t=\Phi C z_t+e_t,\quad e_t\mid\sigma^2,q,\ell\sim
N(0,\sigma^2R_q),\quad R_q=I+qK_\ell,
\]

\[
b=\operatorname{vec}(C),\qquad
b\mid\sigma^2,\lambda\sim N(0,\sigma^2\lambda^{-1}I),\qquad
\sigma^2\sim\operatorname{IG}(a_0,b_0).
\]

The original normalized Gaussian integrated likelihood and the original
hyperpriors are unchanged. Let the original collapsed target density be
\(\pi(a,\xi,\zeta\mid Y)\), with respect to
\(da\,d\xi\,d\zeta\), where

\[
a=\log\lambda,\qquad \xi=\log q,\qquad \zeta=\log\ell.
\]

Define

\[
\eta=a+\log(1+e^\xi),\qquad
a=\eta-\log(1+e^\xi).
\]

The forward Jacobian is triangular:

\[
\frac{\partial(\eta,\xi,\zeta)}{\partial(a,\xi,\zeta)}
=\begin{pmatrix}1&\operatorname{logistic}(\xi)&0\\0&1&0\\0&0&1\end{pmatrix},
\qquad |J|=1.
\]

Therefore the exact transformed log target is

\[
\log\widetilde\pi(\eta,\xi,\zeta\mid Y)
=\log\pi\{\eta-\operatorname{softplus}(\xi),\xi,\zeta\mid Y\}.
\]

`numpy.logaddexp(0, xi)` computes softplus without directly forming
`exp(xi)`. There is no extra Jacobian term in these log coordinates. Original
priors are evaluated at the inverse-transformed values. Assigning independent
replacement priors to the new coordinates would define a different model.

For interpretation, define

\[
v=\sigma^2(1+q),\qquad \kappa=\lambda(1+q),\qquad
\widetilde R_q=\frac{I+qK_\ell}{1+q}.
\]

Then residual covariance is \(v\widetilde R_q\) and coefficient-prior
covariance is \(v\kappa^{-1}I\). The implemented proposal coordinate is
\(\eta=\log\kappa\). The sampler continues integrating out \(\sigma^2\);
it does not explicitly sample \(v\) or change its induced distribution.

## 2. Transition and supported scope

With `parameterization="ridge_total_scale"`, initialization and local
MAP/curvature calibration operate on the transformed target. An adaptive
Gaussian random-walk proposal is symmetric conditional on its current
warmup adaptation state. Its MH probability is

\[
\alpha(x,x')=\min\{1,\exp[\log\widetilde\pi(x')-
\log\widetilde\pi(x)]\}.
\]

Adaptation stops at the unchanged warmup boundary. Retained states store
the original prior/covariance parameters. Conditional recovery of the
coefficient matrix and observation scale is unchanged. Diagnostic names
remain `prior.log_precision`, `log_ell`, `log_q`, and the existing recovered
scale/readout functionals; proposal-coordinate names are separately recorded
in sampler metadata. The code vector order is described by that metadata,
not assumed from the order chosen in the equations above.

The certified first route requires collapsed scalar ridge, iid Gaussian
innovations, and relative-scale isotropic stationary Matérn covariance with
constant nugget and fixed smoothness. Other prior, dynamics, anisotropy,
smoothness-mixture, separate-scale, nonstationary, or sparse combinations
raise an explicit compatibility error for this option. Their legacy routes
remain available. Exact change-of-variable algebra is not a claim that this
particular geometry is efficient for every possible extension.

Example:

```python
from bayes_stesn import SamplerConfig

sampler = SamplerConfig(
    parameterization="ridge_total_scale",
    chains=4,
    warmup=1000,
    draws=2000,
    seed=2026,
)
```

The benchmark adapter accepts the same option inside its existing `sampler`
mapping. Counts above are a proposed development budget, not a validated
universal optimum.

The named benchmark ID `bayes_esn_ridge` requires scalar ridge and iid
Gaussian innovations. Prior/dynamics extensions remain available through the
core `prepare`/`fit` API and must be labeled separately when compared. The
registry and adapter reject those extensions under the baseline ID before
fitting; changing a prior cannot silently retain a misleading ridge label.

## 3. Checkpoints and one bounded continuation

The core checkpoint signature includes a nonlegacy coordinate option. A
checkpoint cannot cross from one proposal policy to another. The newly
named legacy default is omitted from signature hashing, preserving the
original legacy signature. Missing legacy configuration fields are resolved
to the legacy default during high-level recovery. Existing fitted posterior
artifacts remain readable.

The benchmark adapter writes a numeric/JSON-safe sibling artifact:

```text
chain.npz
chain.npz.fit-state.npz
```

The fit-state receipt binds prepared data/model identity, the initial draw
target, the maximum draw target, diagnostics, diagnostic seed, implementation
fingerprint, and numerical runtime. Checkpoint-write cadence is deliberately
excluded from the transition identity. Its phases are:

1. Initial sampling.
2. Initial diagnostics.
3. Continuation sampling, only if the initial completed fit fails the gate
   and a continuation allowance was declared.
4. Continuation diagnostics.
5. Complete.

Transitions are atomic. The receipt stores completed gates and checkpoint
hash/progress summaries at boundaries. On a new invocation, compatible
surviving state resumes automatically. Warmup is not repeated, retained draws
are not duplicated, and a restart does not grant a second continuation.
Complete receipts must match the final checkpoint exactly. Incompatible
data, source/runtime, or declared fitting/diagnostic budgets are rejected
before overwriting the existing state.

Such persisted-identity, phase, and checkpoint-integrity failures use
`bayes_stesn.benchmark.esn.CheckpointIntegrityError`, a `ValueError` subclass.
Campaign controllers should invalidate incompatible evidence rather than
record this condition as a model convergence failure. Ordinary model-domain
errors remain distinct.

A compatible older core checkpoint without a fit-state receipt can be
imported at either its initial target or its declared continuation target.
If already continued, the initial diagnostic gate is reconstructed from the
retained initial prefix. No additional draw allowance is granted. Such
imports are explicitly labeled; historical source/runtime provenance must
come from the caller's original run records because the older core artifact
does not contain the newer benchmark computation receipt.

The continuation rule is executable: one extension after a finite,
error-free initial fit fails the unchanged diagnostic gate. Numerical
exceptions preserve recoverable state and propagate to the enclosing job
controller. They are not silently declared successful fits.

## 4. Verification and development comparison

Focused tests cover:

- Stable coordinate inversion and the unit Jacobian.
- The transformed target against an independent observation-space Gaussian
  integral including the original priors.
- Both coordinate algorithms against independent Gauss-Hermite integration
  of a coupled precision/amplitude posterior and recovered coefficient/scale
  moments, with quadrature refinement and chain MCSE accounting.
- Exact same-runtime interruption/resumption during warmup and retained
  draws, and exact continuation to a larger retained target.
- Benchmark interruption at sampling, diagnostic, and terminal-save
  boundaries, including exact forecast replay.
- Legacy checkpoint import, changed-cadence compatibility, wrong-budget
  rejection, and terminal-checkpoint mismatch rejection.

The old and new algorithms need not share bitwise trajectories. They must
target the same posterior; each algorithm's resumed trajectory must match
its uninterrupted trajectory within the same numerical runtime.

Run a small execution check after installing the project and development
dependencies:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=src python tools/compare_sampler_coordinates.py \
  --profile smoke --scenarios g1 --output results/coordinate-smoke
```

For a prespecified G1/G3/G4 development comparison:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=src python tools/compare_sampler_coordinates.py \
  --profile pilot --scenarios g1,g3,g4 --replications 0 \
  --chains 4 --warmup 1000 --draws 2000 --continuation 2000 \
  --output results/coordinate-development
```

Use `--esn-config selected.json` to supply an archived ESN configuration or a
selection receipt containing `selected`. Without it, the command uses an
explicit modest reference architecture. The algorithms share predetermined
reservoir and chain seeds to preserve the exact model target; their
transition paths differ. It checks the prepared target identities directly.

This command generates only the fitted observed-history prefix and computes
no final forecast scores. It reports original-coordinate qualification,
minimum scientific ESS, elapsed fit/diagnostic time, and ESS per second for
fresh runs. A recovered partial run lacks its lost wall time, so its speed
ratio is unavailable rather than being reported as an artificial speedup.
All short-smoke qualification failures remain visible. Source/runtime and
data/configuration identities are sealed in the output manifest.

Promote the new default only after evidence shows improved qualification or
efficiency without a new material failure mechanism. The implementation and
unit/reference tests alone do not establish that improvement.
