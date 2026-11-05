# KS results recovered from the completed application

**Scientific computation completed; the complete original artifact handoff remains unavailable after automated workspace maintenance.** See [recovery status](ks_recovery_status.md). These results belong to a custom, literature-grounded experiment, conditional on validation-selected configurations. They are not a published leaderboard reproduction.

## Evidence and aggregation

E1 tables below are generated directly from the exact published [main-track audit](evidence/ks_main_tracks_audit_v2.json), SHA-256 `fe4448818eda7e7f3ead715d989e5900db5464d47ddd446ce357c161e792f4c0`. Each trajectory receives equal weight. RMSE is calculated within trajectory before averaging trajectory RMSEs; intervals use 2,000 trajectory-cluster bootstrap resamples. Twenty origins in a trajectory are dependent. All errors are in physical KS units.

## Track A: Strict interleaved unseen-site forecasting (E1)

| Model | Complete trajectories | RMSE [95% interval] | MAE | NRMSE | VPT, steps | Failed origins | Finite-divergent origins |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Bayesian ST-ESN | 20/20 | 1.4298 [1.3330, 1.5495] | 1.1372 | 1.2018 | 4.6625 | 0/400 | 0/400 |
| Separable GP | 20/20 | 1.1701 [1.1568, 1.1828] | 0.8572 | 0.9835 | 14.3825 | 0/400 | 0/400 |
| Basis GRU | 20/20 | 1.1553 [1.1336, 1.1759] | 0.8293 | 0.9710 | 21.6600 | 0/400 | 0/400 |
| Ridge ESN | 20/20 | 1.3511 [1.3007, 1.3986] | 0.9585 | 1.1357 | 22.0300 | 0/400 | 0/400 |
| POD-NVAR | 17/20 | 8.439e+41 [0.7030, 2.532e+42] | 1.146e+40 | 7.078e+41 | 65.3765 | 4/400 | 2/400 |
| Linear/POD VAR | 20/20 | 0.9907 [0.9731, 1.0095] | 0.7469 | 0.8328 | 18.6500 | 0/400 | 0/400 |
| Persistence | 20/20 | 1.1962 [1.1830, 1.2085] | 0.8797 | 1.0055 | 14.5025 | 0/400 | 0/400 |
| Training mean | 20/20 | 1.2156 [1.1984, 1.2361] | 1.0040 | 1.0218 | 1.0000 | 0/400 | 0/400 |

| Native predictive distribution | CRPS | 90% coverage | 90% mean width | Spatial energy | Variogram |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bayesian ST-ESN | 0.8447 | 91.32% | 5.6167 | 1.0209 | 0.3308 |
| Separable GP | 0.6298 | 83.25% | 3.0479 | 0.8035 | 0.2156 |
| Basis GRU | 0.6429 | 63.17% | 1.9011 | 0.7852 | 0.1895 |

## Track B: All-observed temporal forecasting (E1)

| Model | Complete trajectories | RMSE [95% interval] | MAE | NRMSE | VPT, steps | Failed origins | Finite-divergent origins |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Bayesian ST-ESN | 20/20 | 1.2255 [1.2002, 1.2506] | 0.9867 | 1.0299 | 4.8175 | 0/400 | 0/400 |
| Separable GP | 20/20 | 1.1657 [1.1544, 1.1766] | 0.8532 | 0.9797 | 14.2250 | 0/400 | 0/400 |
| Basis GRU | 20/20 | 1.0951 [1.0562, 1.1360] | 0.7889 | 0.9202 | 23.0900 | 0/400 | 0/400 |
| Ridge ESN | 20/20 | 1.2567 [1.2248, 1.2903] | 0.9157 | 1.0562 | 14.9150 | 0/400 | 0/400 |
| POD-NVAR | 20/20 | 4.5454 [0.3091, 12.9436] | 0.2106 | 3.8098 | 91.0225 | 0/400 | 2/400 |
| Linear/POD VAR | 20/20 | 0.9930 [0.9778, 1.0091] | 0.7360 | 0.8346 | 19.1875 | 0/400 | 0/400 |
| Circular CNN | 20/20 | 0.6675 [0.6020, 0.7382] | 0.3763 | 0.5609 | 60.7825 | 0/400 | 0/400 |
| Persistence | 20/20 | 1.2436 [1.2308, 1.2557] | 0.9086 | 1.0452 | 13.7400 | 0/400 | 0/400 |
| Training mean | 20/20 | 1.2177 [1.2011, 1.2379] | 1.0048 | 1.0235 | 1.0000 | 0/400 | 0/400 |

| Native predictive distribution | CRPS | 90% coverage | 90% mean width | Spatial energy | Variogram |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bayesian ST-ESN | 0.7379 | 97.10% | 5.9241 | 0.8962 | 0.2092 |
| Separable GP | 0.6401 | 94.52% | 4.6224 | 0.8266 | 0.1526 |
| Basis GRU | 0.5866 | 76.50% | 2.4304 | 0.7187 | 0.1229 |

**Reliability qualifications.** A POD-NVAR has four nonfinite origins across three trajectories. Its displayed accuracy is conditional on 17 complete trajectories and includes two further finite catastrophes. Its full 400-origin RMSE is undefined. B POD-NVAR has all 400 finite origins but two finite divergences; rare extreme errors dominate its RMSE. No forecast was clipped or replaced.

VPT is first lead with spatial RMSE/training SD greater than 0.4, capped/right-censored at 100. A finite divergence has any lead with normalized spatial RMSE greater than 10. Multiply steps by 0.25 for physical time. These are prespecified experiment conventions, not Lyapunov-time estimates.

## Paired evidence about the Bayesian model (E1)

| Track | Comparator | Bayesian minus comparator RMSE [95% interval] |
| --- | --- | --- |
| A | Basis GRU | 0.2745 [0.1782, 0.4006] |
| A | Linear/POD VAR | 0.4391 [0.3471, 0.5563] |
| A | Persistence | 0.2336 [0.1381, 0.3536] |
| A | Ridge ESN | 0.0787 [-0.0237, 0.2045] |
| A | Separable GP | 0.2597 [0.1646, 0.3794] |
| A | Training mean | 0.2142 [0.1202, 0.3357] |
| B | Basis GRU | 0.1303 [0.0864, 0.1715] |
| B | Linear/POD VAR | 0.2325 [0.2031, 0.2636] |
| B | Circular CNN | 0.5580 [0.4919, 0.6199] |
| B | Persistence | -0.0182 [-0.0475, 0.0105] |
| B | Ridge ESN | -0.0312 [-0.0730, 0.0114] |
| B | Separable GP | 0.0597 [0.0312, 0.0869] |
| B | Training mean | 0.0077 [-0.0248, 0.0381] |

A positive difference favors the comparator. The Bayesian model has worse RMSE and CRPS than both GP and GRU in both main tracks. Its mean RMSE difference from the ridge ESN has an interval spanning zero in both tracks. In B, its small mean improvement over persistence is also unresolved. An interval spanning zero is not proof of equivalence. Aggregate coverage alone is insufficient: the Bayesian A intervals under-cover early leads and become much wider later.

## Contiguous-site robustness and seed sensitivity (E2)

The original final extension-audit bytes were not recovered. The following are recorded historical results preserved in [an explicitly transcript-derived summary](evidence/ks_recovered_extensions_transcript.json). They are not newly reconstructed per-origin records. C transferred the frozen A configurations and refitted on its own permitted data; it is not a C-optimized main ranking.

| C model | Complete trajectories | RMSE | CRPS | Nonfinite origins | Finite-divergent origins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bayesian ST-ESN | 20/20 | 2106.688962 | 905.339301 | 0/400 | 400/400 |
| Separable GP | 20/20 | 1.517825 | 0.857928 | 0/400 | 0/400 |
| Basis GRU | 20/20 | 5.036093 | 2.933332 | 0/400 | 95/400 |
| Ridge ESN | 20/20 | 2.230277 | — | 0/400 | 0/400 |
| Linear/POD VAR | 20/20 | 1.302969 | — | 0/400 | 0/400 |
| Persistence | 20/20 | 1.500100 | — | 0/400 | 0/400 |
| Training mean | 20/20 | 1.184368 | — | 0/400 | 0/400 |
| POD-NVAR | 0/20 | Undefined | — | 260/400 | 37/400 |

All 20 C Bayesian fits passed the recorded inference gate. Its forecasting failure was already present at lead 1 (RMSE 3607.925348); 93.99% aggregate 90% coverage accompanied an interval width of 6877.457476. This is not useful uncertainty quantification. The bounded fixed-case diagnosis supports amplification from weakly identified unseen-site spatial basis directions; it is not a complete causal explanation or a corrected model. The accepted prior and likelihood were preserved.

All 56 additional seed jobs were recorded complete. Seed comparisons use the same four trajectories for all three model seeds; they do not compare a four-trajectory average with the twenty-trajectory main average, select a favorable seed, or pool model seeds into posterior uncertainty.

## Code, figures and continuation

The [model registry](ks_model_registry.md), [native implementation audit](ks_native_audit.md), [frozen protocol](ks_protocol.md), and [reproduction instructions](ks_reproduction.md) specify the actual models and adaptations. All 51 frozen numerical source hashes and all 43 authoritative core-source files were verified unchanged in the recovered checkout.

The standalone PDF/HTML recovery report contains full main probability/interval tables, six-lead curves with cluster intervals, compute and reliability figures, seed/contiguous summaries, and provenance. A separately recomputed A Bayesian trajectory-0 / first-target-8000 example uses the retained original fit, frozen 128-state history, 100-step horizon, 512 paths and original random stream. Its six recorded historical lead RMSEs match exactly; it is not described as recovery of the missing original prediction-file bytes.

To reproduce these recovery deliverables, use `tools/build_ks_recovery_report.py` and `tools/recover_ks_example.py` with the recovered private inputs, as documented in their `--help`. The complete original final report and multipart backup still require the unavailable full result directory. Do not bypass their evidence guards or create a fabricated runner summary.
