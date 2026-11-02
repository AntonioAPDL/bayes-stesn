# Optimization decisions and supported scope

The version 0.2 changes target measured or directly visible bottlenecks. They
preserve the original probabilistic model where an optimization is advertised
as exact; model approximations remain separately labeled.

| Area | Implemented change | Benefit and remaining boundary |
|---|---|---|
| Forecast storage | Select requested components; retain only observations for benchmark scoring | Avoids allocating four equal-sized outputs; observation component remains required |
| Ownership | Freeze privately generated contiguous allocations without copying them again | Public constructors still copy user arrays; round-trip ownership tested |
| Predictive batching | Named global path IDs and `path_start`; batch iterator | Batches concatenate exactly in the same runtime; frame and spatial workspaces still exist |
| Mixtures | Globally indexed ensemble-member and within-member paths | Worker/batch ordering no longer changes ensemble draws; RNG scheme deliberately versioned |
| Spatial scores | Floyd sampling of unordered pairs in O(requested pairs) storage | Avoids O(n squared) upper-triangle construction; explicit all-pair requests still imply large outputs |
| Archive writes | Cleanup covers compression, flush, fsync and atomic replacement | Existing canonical output survives pre-replacement failures; post-rename durability failures remain visible |
| Benchmark checkpoints | Default to chain endpoints; preserve explicit caller cadence | One pilot-size fit fell from 33.64 s to 9.56 s with identical chains, recovered parameters, forecasts, diagnostics and continuation states; incomplete chains lose their intermediate recovery point |
| Checksum reads | Contiguous byte views and direct private NPZ arrays | Removes unnecessary full-size byte/copy allocations |
| Vecchia geometry | Cache fixed predecessor choices | Hyperparameter-dependent conditional coefficients are recomputed correctly |
| Vecchia likelihood | Sparse nugget-updated LU and operator sufficient statistics | No dense spatial inverse, covariance or YY matrix for the supported route; fill-in remains geometry dependent |
| Vecchia determinant | Exact conditional-variance determinant; prior LU is lazy | Removes an unused second factor in the supported inference workflow |
| Vecchia solves | Non-subtractive identity-interpolation solve | Avoids Woodbury cancellation for extreme spatial/nugget variance ratios |
| Vecchia prediction | Sparse triangular ancestral latent simulation | Samples the existing latent approximation plus separate nugget; no observation-Vecchia substitution |
| LGSSM | Coordinate basis with orthonormal observed-site loadings | Filtering works in small state dimension; full observation likelihood includes omitted-space residual terms |
| GP | Spatial eigensystem and low-order exact temporal state representation | Avoids an (nT)-by-(nT) covariance; new-site joint residual covariance is still bounded dense |
| KRR | Nyström features and normalized ridge | Trades a controlled kernel approximation for bounded feature dimension; exact reference mode is retained |
| Neural graph | Sparse geometric diffusion and window stride | Reduces graph/window work; architecture and training precision are recorded |
| Experiments | Process workers, one numerical thread each, strict manifests and per-job resumes | Avoids oversubscription and repeated completed jobs; worker memory preflight is a planning bound, not a measured peak |

## Exact scope of the end-to-end operator route

The automatic operator path requires `CovarianceSpec(kind="vecchia")`, iid
Gaussian innovations, `PriorSpec.ridge()`, and collapsed inference. It also can
be requested explicitly with `SamplerConfig(backend="vecchia_operator")`.
Incompatible explicit requests fail. Dense/reference modes retain their site
caps. The arbitrary-target forecast appends targets after observed sites, so
the observed factors and response history are unchanged.

The spatial approximation is the **latent** predecessor-based Gaussian field.
The observation nugget remains separate. Sparse precision multiplication and
sparse solves replace dense spatial algebra; the temporal readout Gram and small
spatial-basis Gram remain dense. Therefore large reservoir dimension can still
be expensive even when the number of sites is handled sparsely.

The measured engineering profile uses small readout dimensions and deliberately
short chains. It establishes execution/resource behavior only. It does not
establish MCMC convergence, comparative forecasting accuracy, mesh convergence,
or million-site scalability. The final sparse-factor change reduced the
2048-site process peak from about 153.6 to 125.2 MiB in the local profile. Wall
times varied, so a speedup is not inferred from that pair of measurements.
An additional 4096-site execution check completed at 184.0 MiB process peak RSS,
with 5.44 s fit and 6.07 s prediction for the same deliberately small dimensions.

## Conditional extensions

The full machine-readable roster is [benchmark_registry.yaml](benchmark_registry.yaml).
The core comparison and the optional matched basis-GRU control are implemented.
The earlier plan explicitly made external R/Julia, complete-grid neural
operator, diffusion, and pretrained-model additions conditional on their own
capability, dependency, cost, and data-budget gates. Those integrations remain
listed as `deferred_optional`; they are not exposed as callable fitted models.

Complete-grid methods require track C with the same complete-grid information
for every comparator. Pretrained models require a separate external-data
budget. Neither should be ranked as if it received only the sparse historical
panel used in tracks A/B. Installing a package alone would not meet these
scientific comparison requirements.

## Reproducibility boundaries

Named streams make scheduling order irrelevant; installed-wheel parity checks
guard against importing an editable source checkout accidentally. Float64 is
used for the statistical core. The PyTorch mean models use explicit CPU
float32 training. Exact seeded byte equality is only promised within the same
supported numerical runtime, not across arbitrary BLAS, compiler or device
versions. The profile records the inference and forecast components actually
retained and the exact candidate and continuation budgets.

The objective is efficient, auditable inference under these declared limits.
There is no claim of globally optimal hyperparameters, universally superior
forecasting performance, or fully converged inference merely because unit
tests pass.
