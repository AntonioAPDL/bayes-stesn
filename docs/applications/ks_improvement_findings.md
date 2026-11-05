# What the KS improvement study establishes

The revised Bayesian ST-ESN improves on its earlier configuration, but the
conclusion depends on spatial protocol and evaluation criterion. Track A shows
clear paired improvements in RMSE and CRPS. Track B improves CRPS while its RMSE
contrast remains uncertain. Track C greatly reduces the earlier catastrophic
extrapolation errors, yet still performs poorly against simple point references.

This interpretation accompanies the [complete report](improvement_results/README.md),
[confirmation table](improvement_results/confirmation.csv),
[paired comparisons](improvement_results/paired_comparisons.csv), and
[selected configurations](improvement_results/selected_configurations.json).
The [numerical audit](evidence/ks_improvement_final_numerical_audit.json) independently
checked the receipt arithmetic, complete-case populations, matched seeds and
saved examples; the [metadata audit](evidence/ks_improvement_final_evidence_audit.json)
checks inference qualification and execution provenance.

## Scientific population and preserved model

This is an exploratory, validation-only follow-up after inspection of the
original application's test results. Fourteen trajectories were excluded from
this follow-up's configuration selection; they are not a new historically
untouched test set. Each confirmation fit uses permitted observations in
`[0,6000)`, followed by twenty autonomous 100-step forecasts in `[6000,8000)`.
Every origin uses a 512-state permitted history and 512 predictive paths. Track A
predicts thirteen interleaved validation sites from thirty-nine training sites;
B uses all sixty-four sites; C predicts twelve contiguous validation sites from
forty training sites. Scores are reported separately.

The accepted model remains
\(y_t=\Phi C z_t+e_t\), with independent temporal innovations
\(e_t\sim N\{0,\sigma^2(I+qK_\ell)\}\), and
\(\operatorname{vec}(C)\mid\sigma^2,\lambda\sim
N(0,\sigma^2\lambda^{-1}I)\). The main search preserves each track's numerical
hyperprior settings. Changing the basis or reservoir nevertheless changes the
induced prior over fields: preservation is conditional on architecture.
Residual innovations describe discrepancy for the clean field, not additional
measurement noise. Point forecasts average complete predictive paths.

## Search opportunity and the selected architectures

The [registry](improvement_results/candidate_registry.json) contains 104 distinct
track/configuration candidates: 40 for A, 40 for B and 24 for C. These comprise 86
fixed-prior candidates, 12 prior-scale sensitivities and 6 legacy-prior references.
Structured controls and joint proposals were screened on two trajectories;
promotion used six trajectories. This was a bounded staged search, not a full
Cartesian optimization.

| Main-search range | A and B | C |
|---|---|---|
| Reservoir units |32–512|32–128|
| Recurrent operator norm |0.30–0.98|0.30–0.90|
| Leak values |0.05–1.00|0.05–0.90|
| Input scale |0.05–2.00|0.20–2.00|
| PCA coordinates |8–32|8–16|
| Spatial columns / width |8–32 /0.15–1.00|8–32 /0.15–1.00|

Connectivity ranged from 0.02 to 0.20. Lag proposals were `[1]`, `[1,2]`, and
`[1,2,4]`; conditioning prefixes ranged from 80 to 2048 observations in A/C and
512 to 2048 in B. These operator-norm bounds are not conventional ESN spectral
radius bounds. Reservoir forecasting studies motivate the comparison, without
making this a reproduction of their protocols
([Pathak et al., 2018](https://doi.org/10.1103/PhysRevLett.120.024102);
[Vlachas et al., 2020](https://arxiv.org/abs/1910.05266)).

A selected 32 units, input scale 2, leak 1, eight PCA coordinates and a narrower
32-column spatial basis. B selected its 64-unit architecture with one lag instead
of two. C selected eight spatial columns, width 1, leak 0.05 and a 2048-observation
conditioning prefix. C therefore uses 3952 likelihood targets, versus 4976 for its
earlier configuration. These are meaningful simultaneous changes, not isolated
causal interventions. Conditioning fits the feature map and excludes that prefix
from the likelihood; it differs from forecast history and sampler warm-up.

B did **not** select the smallest raw promotion loss. Its 256-unit candidate had
band-balanced CRPS 0.554944, versus 0.575614 for the selected one-lag model. Their
0.020670 gap fell within the predeclared 0.023138 tolerance derived from paired
predictive-simulation repetitions. The rule favored 2100 over 6100 readout
coefficients. Its tolerance is the larger of 1% of the best promotion loss and
twice the paired replicate SD. This local precision heuristic used two fixed validation origins
and four repeated ensembles; it is neither a global equivalence test nor proof
that the smaller model is faster. Exact evidence is in
[search stages](improvement_results/search_stages.json).

## Paired confirmation outcomes

RMSE first pools squared errors within a trajectory, then averages trajectory
RMSEs equally. The following differences are new minus earlier Bayesian
configuration, with 95% trajectory-bootstrap intervals over fourteen pairs,
conditional on the selected configurations.

| Track | Earlier → new RMSE | Paired RMSE difference | Earlier → new CRPS | Paired CRPS difference |
|---|---:|---:|---:|---:|
|A|1.40254 →1.23462|−0.16792 [−0.26095,−0.08540]|0.83460 →0.73910|−0.09550 [−0.16246,−0.04187]|
|B|1.19380 →1.15683|−0.03696 [−0.08751,0.01095]|0.72702 →0.66126|−0.06576 [−0.10985,−0.02647]|
|C|2442.36301 →2.19702|−2440.16599 [−4166.74888,−1442.55062]|1136.44100 →2.51351|−1133.92749 [−1801.75867,−753.84855]|

A supports improvement under both losses. B supports a distributional-score
gain; its RMSE interval includes zero. C's reduction is substantial, but the
comparison starts from an extremely poor reference. All native confirmation
forecasts are finite and inference-qualified; C still has five finite divergent
origins out of 280. Passing the monitored four-chain convergence gates does not
establish predictive adequacy
([Vehtari et al., 2021](https://arxiv.org/abs/1903.08008)).

The separately selected prior controls produced small A/B changes: RMSE
differences of −0.00235 and −0.00085, and CRPS differences of −0.00127 and −0.00051.
All four paired intervals include zero. This is little evidence of benefit from
those evaluated controls, not proof that prior choice is irrelevant.

## Distribution quality, coverage and prediction time

Lower CRPS supports better marginal predictive distributions under this score;
coverage alone cannot establish forecast quality
([Gneiting and Raftery, 2007](https://doi.org/10.1198/016214506000001437)).
A's 90% coverage rises from 91.37% to 94.95%, while width falls from 5.617 to 5.244.
Thus improved CRPS coexists with greater aggregate overcoverage. Its mean
horizon-restricted valid prediction time declines from 1.162 to 0.994 physical
units. That diagnostic uses the predictive mean and a 0.4-training-SD threshold;
it measures something different from average 100-step loss.

B's 90% coverage changes from 97.91% to 92.68%, width from 6.067 to 4.211, and restricted
prediction time from 1.222 to 1.878. These jointly support a more useful forecast
distribution under this protocol, although early-horizon coverage remains below
nominal.

C's 99.35% aggregate coverage comes with width 33.36. At lead 1, RMSE is 4.885 and 90%
coverage only 49.91%; by lead 10 coverage reaches 100%. Broad later intervals conceal
early undercoverage in the aggregate. Restricted prediction time is 0.263 physical
units. The contiguous-site problem remains unresolved.

## Fixed references and failure populations

Fixed-configuration VAR achieves RMSE 0.98914, 0.99506 and 1.32206 in A/B/C,
respectively, outperforming the new Bayesian point mean in all three paired
comparisons. C's training-mean reference is better still at 1.22275. The new
Bayesian model beats ridge ESN in A/B; their C RMSE contrast includes zero.
These references have no invented probabilistic intervals.

NVAR's attractive successful-only RMSEs require their denominators: A has 0.74464
over twelve complete trajectories, with two nonfinite origins out of 280; B has
0.28558 over eleven, with three failures. Two additional A origins are finite but
divergent. C has 181 nonfinite failures and 99 finite forecasts, including seventeen
finite divergences; no trajectory is complete, so its aggregate RMSE is undefined.
All 210 reference fits completed: the nineteen failed jobs reflect forecast
failures. A/B NVAR cannot be ranked as successful over the complete schedule.

## Seed sensitivity, cost and mechanism

The [matched seed analysis](improvement_results/seed_sensitivity.json) uses the
same four trajectories and eight origins for seeds 20261105–20261107. New-model
RMSEs are 1.166/1.174/1.194 in A, 1.180/1.093/1.133 in B, and 2.120/2.406/1.743 in C.
Each remains below its corresponding earlier-model mean in this subset, but C
shows material variability. These four-trajectory checks are conditional
sensitivity evidence, not comparisons with the fourteen-trajectory main mean
or a reason to select a favorable seed.

Recorded [confirmation costs](improvement_results/confirmation.csv) distinguish
complexity from measured execution:

| Track | Mean fit seconds, earlier → new | Summed fit + forecast seconds, earlier → new |
|---|---:|---:|
|A|38.83 →29.35|1835.55 →1571.79|
|B|40.40 →62.32|2227.75 →2544.47|
|C|39.85 →33.84|2007.62 →1770.38|

These totals cover fourteen fits and 280 origins, excluding search and orchestration.
B is more expensive despite reducing readout coefficients from 2260 to 2100.
The Bayesian campaign consumed 12.059 worker-hours including recorded retries and
preparation; search consumed 5.865 hours. The documented prospective allowance
was 15 worker-hours, with the original clock and charges retained.

Training-coordinate basis condition numbers fall from 4097 to 379 in A and from
\(5.08\times10^9\) to 124 in C
([geometry evidence](improvement_results/realized_dynamics.json)). This is
consistent with improved spatial identification, but joint changes in width,
rank, leak and conditioning prevent a single-cause explanation.

The study establishes budget-specific improvements, not an exhaustive optimum
or a new leaderboard. GP, GRU and CNN were not retuned in this follow-up, and
several selected settings lie on proposal boundaries. Future experiments could
test increment targets, training-only POD quadratic features inspired by
[Gauthier et al., 2021](https://arxiv.org/abs/2106.07688), or a predetermined periodic
Fourier spatial basis. These are unexecuted model variants requiring renewed
prior, leakage, stability and distributional checks; none is a demonstrated remedy.
