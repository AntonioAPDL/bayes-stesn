#!/usr/bin/env python3
"""Conditional conjugate Gaussian simulation-based calibration.

Run from a checkout with::

    python benchmarks/validate_statistics.py --output results

This experiment fixes the design, spatial covariance and coefficient precision.
It draws sigma² and C from their proper joint prior, generates a training panel,
and recovers *joint* posterior samples with the public Gaussian implementation.
It checks sigma², two coefficients, a future readout mean and a fresh future
observation. It is NOT SBC for recursive ESN-generated covariates, estimated
covariance hyperparameters, shrinkage MCMC, or augmented likelihood samplers.

Independent full-design Gaussian algebra supplies a numerical cross-check and
exact marginal 90% intervals. Rank tests are stochastic diagnostics, not proof
of correctness. Their realised p-values and Monte Carlo uncertainty are saved
without selecting a more favourable seed or suppressing failed diagnostics.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

# Set before NumPy/SciPy imports, so this small experiment has a predictable
# one-thread CPU footprint even in a machine with a large BLAS default.
for thread_option in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                      "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_option] = "1"

import numpy as np
import scipy
from scipy import linalg, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from bayes_stesn.gaussian import GaussianRegression, PrecisionSpec


def _wilson(successes: int, total: int, confidence: float = 0.95) -> list[float]:
    return list(stats.binomtest(successes, total).proportion_ci(confidence, method="wilson"))


def run_validation(replicates: int = 200, posterior_draws: int = 255,
                   seed: int = 20261102, output: str | Path = "results") -> dict:
    """Execute the declared experiment, save complete numerical evidence."""
    if replicates < 32 or posterior_draws < 31:
        raise ValueError("use at least 32 independent replicates and 31 posterior draws")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    n, T, p, d = 5, 8, 2, 3
    site = np.linspace(-1, 1, n)
    Phi = np.column_stack((np.ones(n), site))
    phase = np.linspace(0, 2*np.pi, T, endpoint=False)
    Z = np.column_stack((np.ones(T), np.sin(phase), np.cos(phase)))
    # A declared fixed, proper row covariance; no fitted kernel parameters.
    R = .65**np.abs(np.subtract.outer(np.arange(n), np.arange(n)))+.35*np.eye(n)
    LR = linalg.cholesky(R, lower=True)
    columns = np.array([1.5, 3.0, 2.2])
    precision = PrecisionSpec(p, d, "column", columns)
    a0, b0 = 3.0, .9
    future_site = 3
    future_z = np.array([1., np.sin(1.3), np.cos(1.3)])
    future_x = np.kron(future_z, Phi[future_site])
    names = ("sigma2", "coefficient_0_0", "coefficient_1_1",
             "future_readout_mean", "future_observation")
    selected_coefficient_indices = (0, 1+p)
    count_statistics = len(names)
    truths = np.empty((replicates, count_statistics))
    posterior_values = np.empty((replicates, posterior_draws, count_statistics))
    ranks = np.empty((replicates, count_statistics), dtype=np.int64)
    intervals = np.empty((replicates, count_statistics, 2))
    responses = np.empty((replicates, n, T))
    true_coefficients = np.empty((replicates, p, d))
    reference_errors = np.zeros((replicates, 2))

    # Independent dense full-design posterior, not the implementation's
    # Kronecker eigensystem or cached sufficient-statistic code.
    X = np.kron(Z, Phi)
    observation_precision = np.kron(np.eye(T), np.linalg.inv(R))
    A = np.diag(np.repeat(columns, p))+X.T @ observation_precision @ X
    A_inverse = np.linalg.inv(A)
    an = a0+n*T/2
    linear_functionals = np.vstack((np.eye(p*d)[selected_coefficient_indices[0]],
                                    np.eye(p*d)[selected_coefficient_indices[1]],
                                    future_x, future_x))
    function_variance = np.einsum("ij,jk,ik->i", linear_functionals, A_inverse, linear_functionals)
    function_variance[-1] += R[future_site, future_site]
    t_quantiles = stats.t.ppf([.05, .95], df=2*an)

    for replicate in range(replicates):
        prior_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(replicate, 0)))
        posterior_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(replicate, 1)))
        truth_future_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(replicate, 2)))
        true_sigma2 = float(b0/prior_rng.gamma(a0))
        C = np.sqrt(true_sigma2/columns)[None, :]*prior_rng.normal(size=(p, d))
        Y = Phi @ C @ Z.T+np.sqrt(true_sigma2)*(LR @ prior_rng.normal(size=(n, T)))
        responses[replicate], true_coefficients[replicate] = Y, C
        true_mean = float(Phi[future_site] @ C @ future_z)
        future_observation = true_mean+np.sqrt(true_sigma2*R[future_site, future_site])*truth_future_rng.normal()
        truths[replicate] = (true_sigma2, C[0, 0], C[1, 1], true_mean, future_observation)
        engine = GaussianRegression(Y, Phi, Z, a0=a0, b0=b0)
        evaluation = engine.evaluate(R, precision=precision)
        for draw in range(posterior_draws):
            sampled_C, sampled_sigma2, _ = engine.draw(evaluation, posterior_rng)
            sampled_mean = float(Phi[future_site] @ sampled_C @ future_z)
            sampled_observation = sampled_mean+np.sqrt(sampled_sigma2*R[future_site, future_site])*posterior_rng.normal()
            posterior_values[replicate, draw] = (sampled_sigma2, sampled_C[0, 0], sampled_C[1, 1],
                                                  sampled_mean, sampled_observation)
        ranks[replicate] = np.sum(posterior_values[replicate] < truths[replicate], axis=0)

        vector = Y.ravel(order="F")
        information = X.T @ observation_precision @ vector
        dense_mean = A_inverse @ information
        dense_bn = float(b0+.5*(vector @ observation_precision @ vector-information @ dense_mean))
        reference_errors[replicate] = (np.max(np.abs(evaluation.mean.ravel(order="F")-dense_mean)),
                                       abs(evaluation.bn-dense_bn))
        intervals[replicate, 0] = stats.invgamma.ppf([.05, .95], a=an, scale=dense_bn)
        means = linear_functionals @ dense_mean
        intervals[replicate, 1:] = means[:, None]+np.sqrt(dense_bn/an*function_variance)[:, None]*t_quantiles

    if reference_errors.max() > 1e-10:
        raise AssertionError(f"conjugate solver disagrees with independent dense algebra: {reference_errors.max():g}")
    covered = (truths >= intervals[:, :, 0]) & (truths <= intervals[:, :, 1])
    jitter_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(replicates, 99)))
    randomized_rank = (ranks+jitter_rng.uniform(size=ranks.shape))/(posterior_draws+1)
    bins = 16
    discrete_levels = (np.arange(posterior_draws+1)+.5)/(posterior_draws+1)
    histogram_probability = np.histogram(discrete_levels, bins=bins, range=(0, 1))[0]/(posterior_draws+1)
    summaries = {}
    familywise_threshold = .01/count_statistics
    for j, name in enumerate(names):
        counts = np.histogram((ranks[:, j]+.5)/(posterior_draws+1), bins=bins, range=(0, 1))[0]
        chi = stats.chisquare(counts, f_exp=replicates*histogram_probability)
        ks = stats.kstest(randomized_rank[:, j], "uniform")
        successes = int(covered[:, j].sum())
        coverage_p = stats.binomtest(successes, replicates, p=.9).pvalue
        summaries[name] = {
            "rank_histogram": counts.tolist(),
            "rank_histogram_expected": (replicates*histogram_probability).tolist(),
            "rank_chisquare_statistic": float(chi.statistic),
            "rank_chisquare_p_value": float(chi.pvalue),
            "randomized_rank_ks_statistic": float(ks.statistic),
            "randomized_rank_ks_p_value": float(ks.pvalue),
            "mean_randomized_rank": float(randomized_rank[:, j].mean()),
            "nominal_interval_coverage": .9,
            "exact_interval_coverage": successes/replicates,
            "coverage_successes": successes,
            "coverage_wilson_95_interval": _wilson(successes, replicates),
            "coverage_binomial_p_value": float(coverage_p),
            "severe_rank_discrepancy": bool(ks.pvalue < familywise_threshold),
        }
    result = {
        "experiment": "conditional_conjugate_gaussian_sbc",
        "scope": "Fixed design, fixed row covariance, fixed proper column precision; joint C,sigma2 posterior recovery.",
        "excluded_claims": ["Not SBC for recursive ESN feature construction.",
                            "Not validation of covariance/prior-hyperparameter MCMC.",
                            "Not validation of robust or latent augmented chains.",
                            "A non-significant rank test does not prove correctness."],
        "seed": seed, "replicates": replicates, "posterior_draws_per_replicate": posterior_draws,
        "dimensions": {"n_sites": n, "n_times": T, "p": p, "d": d},
        "prior": {"sigma2": "InverseGamma(shape=a0,scale=b0)", "a0": a0, "b0": b0,
                  "C": "vec(C)|sigma2 ~ N(0,sigma2 * (D kron I_p)^-1)",
                  "column_precision": columns.tolist()},
        "rank_definition": "Count of independent joint posterior draws strictly below the prior-generated truth, range 0..M.",
        "future_functional": {"site_index": future_site, "known_temporal_predictor": future_z.tolist(),
                              "new_observation_independent_of_training_given_parameters": True},
        "coverage_intervals": "Independent full-design exact Student-t / inverse-gamma 90% marginal intervals.",
        "coverage_mc_standard_error_at_nominal": float(np.sqrt(.9*.1/replicates)),
        "rank_mean_mc_standard_error": float(np.sqrt(1/(12*replicates))),
        "rank_flag_rule": {"method": "Bonferroni for five KS rank tests at familywise 1%",
                           "individual_p_threshold": familywise_threshold,
                           "requires_investigation": any(s["severe_rank_discrepancy"] for s in summaries.values())},
        "dense_reference_max_abs_error": {"posterior_mean": float(reference_errors[:, 0].max()),
                                           "posterior_ig_scale": float(reference_errors[:, 1].max())},
        "statistics": summaries,
        "runtime": {"seconds": time.perf_counter()-started, "blas_threads_requested": 1,
                    "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    }
    np.savez_compressed(output/"statistical_validation.npz", statistic_names=np.asarray(names),
                        true_statistics=truths, posterior_statistics=posterior_values, ranks=ranks,
                        randomized_ranks=randomized_rank, exact_90_intervals=intervals,
                        exact_90_covered=covered, Phi=Phi, Z=Z, R=R, column_precision=columns,
                        training_responses=responses, true_coefficients=true_coefficients,
                        reference_errors=reference_errors)
    (output/"statistical_validation.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--posterior-draws", type=int, default=255)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    result = run_validation(args.replicates, args.posterior_draws, args.seed, args.output)
    print(json.dumps({"experiment": result["experiment"], "replicates": args.replicates,
                      "runtime_seconds": result["runtime"]["seconds"],
                      "rank_flag": result["rank_flag_rule"]["requires_investigation"],
                      "coverage": {name: stats["exact_interval_coverage"]
                                   for name, stats in result["statistics"].items()},
                      "json": str(args.output/"statistical_validation.json"),
                      "npz": str(args.output/"statistical_validation.npz")}, indent=2))


if __name__ == "__main__":
    main()
