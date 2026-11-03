"""Bounded analytical prior check for a saved scalar/iid versus full-AR pair.

This integrates the actual priors conditional on a frozen encoder/design.
It does not use posterior draws, future labels, refitting, or autonomous
rollouts. The conditioning and fitting feature paths are held fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy import special, stats

from bayes_stesn.benchmark.execution import memory_allowance
from bayes_stesn.benchmark.runner import (
    benchmark_environment, jsonable, single_thread_runtime, source_fingerprint,
    write_json,
)
from bayes_stesn.io import content_hash, load_state
from bayes_stesn.model import ModelSpec
from bayes_stesn.sampling import SamplerConfig
from evaluate_extensions_v03 import _verify_terminal_inventory


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def integrated_correlation(covariance, coordinates, rho_sd, order):
    nodes, weights = special.roots_hermitenorm(order)
    weights = weights / np.sqrt(2 * np.pi)
    expected_kernel = np.zeros((len(coordinates), len(coordinates)))
    identity = np.eye(len(coordinates))
    for node, weight in zip(nodes, weights):
        parameters = {"log_q": 0.0, "nu": covariance.nu,
                      "log_ell": covariance.log_ell_mean + covariance.log_ell_sd * node}
        expected_kernel += weight * (covariance.covariance(coordinates, parameters) - identity)
    rho = np.tanh(rho_sd * nodes)
    moments = {str(lag): float(weights @ (rho ** lag))
               for lag in (0, 1, 2, 5, 10, 20, 40)}
    return expected_kernel, moments


def summarize_sd(variance, response_scale):
    sd = np.sqrt(variance)
    standardized = {"minimum": float(sd.min()), "median": float(np.median(sd)),
                    "maximum": float(sd.max()), "rms": float(np.sqrt(np.mean(variance)))}
    return {"conditioning_standard_deviation_units": standardized,
            "original_response_units": {key: value * response_scale for key, value in standardized.items()}}


def prior_moments(prepared, sampler):
    spec = ModelSpec.from_dict(prepared["spec"])
    prior, covariance = spec.prior, spec.covariance
    if (prior.kind != "ridge" or covariance.kind != "matern"
            or covariance.range_prior != "lognormal" or covariance.scale_mode != "relative"
            or covariance.anisotropy or covariance.nonstationary or covariance.static_variance
            or covariance.infer_nu or sampler.temporal not in ("iid", "full_residual_ar")):
        raise ValueError("This check supports only the evaluated scalar ridge, fixed-nu isotropic dense Matérn iid/full-AR pair")
    mu, sd = np.asarray(prior.mean_log_precision), np.asarray(prior.sd_log_precision)
    if mu.size != 1 or sd.size != 1 or sampler.a0 <= 1:
        raise ValueError("A scalar lognormal precision and finite prior variance are required")
    mu, sd = float(mu.reshape(-1)[0]), float(sd.reshape(-1)[0])
    coordinates = np.asarray(prepared["data"]["coordinates"])
    phi, all_z, reference_z = (np.asarray(prepared[key]) for key in ("Phi", "Z", "reference_Z"))
    split = prepared["split"]
    fit_z = all_z[split["fitting_start"]:split["fitting_stop"]]
    if len(phi) > 100 or len(fit_z) > 2000 or phi.shape[1] > 128 or all_z.shape[1] > 1024:
        raise ValueError("Frozen design exceeds this bounded diagnostic's declared dimensions")
    scale = float(prepared["response_scale"])
    sigma_mean = sampler.b0 / (sampler.a0 - 1)
    inverse_lambda_mean = float(np.exp(-mu + sd * sd / 2))
    q_mean = float(np.exp(covariance.log_q_mean + covariance.log_q_sd ** 2 / 2))
    coefficient_variance = sigma_mean * inverse_lambda_mean
    k128, rho128 = integrated_correlation(covariance, coordinates, sampler.rho_prior_sd, 128)
    k256, rho256 = integrated_correlation(covariance, coordinates, sampler.rho_prior_sd, 256)
    difference_kernel = float(np.max(np.abs(k128 - k256)))
    difference_rho = max(abs(rho128[key] - rho256[key]) for key in rho256)
    if max(difference_kernel, difference_rho) > 1e-8:
        raise ArithmeticError("128/256-point quadrature agreement is insufficient; no unchecked prior summary was written")
    expected_r = np.eye(len(phi)) + q_mean * k256
    innovation_variance = sigma_mean * np.diag(expected_r)
    off_diagonal = expected_r[np.triu_indices(len(phi), 1)] / (1 + q_mean)
    phi_energy = np.square(phi).sum(axis=1)
    designs = []
    for name, z in (("conditioning_reference", reference_z), ("observed_history_fitting_design", fit_z)):
        readout_variance = coefficient_variance * np.outer(phi_energy, np.square(z).sum(axis=1))
        observation_variance = readout_variance + innovation_variance[:, None]
        # Independent tiny reference to the tensor covariance and variance diagonal.
        small_phi, small_z = phi[:3], z[:4]
        x = np.kron(small_z, small_phi)
        explicit = coefficient_variance * (x @ x.T)
        tensor = coefficient_variance * np.kron(small_z @ small_z.T, small_phi @ small_phi.T)
        covariance_error = float(np.max(np.abs(explicit - tensor)))
        diagonal_error = float(np.max(np.abs(np.diag(explicit).reshape(len(small_z), len(small_phi)).T - readout_variance[:3, :4])))
        tolerance = 1e-12 * max(1.0, float(np.max(np.abs(explicit))))
        if max(covariance_error, diagonal_error) > tolerance:
            raise AssertionError("Tensor prior covariance disagrees with the explicit design")
        designs.append({"name": name, "times": len(z), "sites": len(phi),
            "spatial_columns": phi.shape[1], "temporal_columns": z.shape[1],
            "frozen_design_hash": content_hash({"Phi": phi, "Z": z}),
            "readout_pointwise_prior_sd": summarize_sd(readout_variance, scale),
            "observation_pointwise_prior_sd": summarize_sd(observation_variance, scale),
            "explicit_tensor_covariance_max_error": covariance_error,
            "explicit_diagonal_variance_max_error": diagonal_error})
    # Symmetry gives exact zero at odd lags. Keep numerical errors separately.
    residual_correlation = {key: (0.0 if int(key) % 2 else value) for key, value in rho256.items()}
    return {"prior_parameters": {"a0": sampler.a0, "b0": sampler.b0,
        "log_precision_mean": mu, "log_precision_sd": sd,
        "log_q_mean": covariance.log_q_mean, "log_q_sd": covariance.log_q_sd,
        "log_ell_mean": covariance.log_ell_mean, "log_ell_sd": covariance.log_ell_sd,
        "nu": covariance.nu, "rho_eta_mean": 0.0, "rho_eta_sd": sampler.rho_prior_sd},
        "response_center": prepared["response_center"], "response_scale": scale,
        "coefficient_scope": "Every scalar-ridge coefficient, including intercept columns; other-prior unshrunk_groups settings do not override scalar ridge",
        "expected_sigma2_standardized": sigma_mean,
        "expected_inverse_lambda": inverse_lambda_mean,
        "expected_q": q_mean,
        "marginal_coefficient_variance_standardized": coefficient_variance,
        "marginal_coefficient_variance_original_units": coefficient_variance * scale ** 2,
        "stationary_residual_prior_sd": summarize_sd(innovation_variance, scale),
        "spatial_residual_marginal_correlation_off_diagonal": {
            "minimum": float(off_diagonal.min()), "median": float(np.median(off_diagonal)),
            "maximum": float(off_diagonal.max())},
        "iid_residual_prior_marginal_autocorrelation": {key: float(key == "0") for key in residual_correlation},
        "ar_residual_prior_marginal_autocorrelation": residual_correlation,
        "ar_absolute_rho_tail_probabilities": {str(threshold): float(2 * stats.norm.sf(np.arctanh(threshold) / sampler.rho_prior_sd)) for threshold in (.9, .95, .99)},
        "ar_absolute_rho_median": float(np.tanh(sampler.rho_prior_sd * stats.norm.ppf(.75))),
        "predictive_variance_finite": True,
        "predictive_fourth_moment_finite": sampler.a0 > 2,
        "fourth_moment_scope": "A scalar observation about its frozen prior center with positive residual variance has an infinite fourth moment when IG shape a0<=2; a0>1 here ensures finite variance",
        "quadrature": {"orders": [128, 256], "absolute_agreement_tolerance": 1e-8,
            "expected_kernel_max_absolute_difference": difference_kernel,
            "rho_moment_max_absolute_difference": difference_rho, "passed": True},
        "designs": designs,
        "interpretation": "iid and stationary full-residual AR have exactly the same pointwise prior amplitude under their shared priors. AR changes temporal dependence. A symmetric rho prior has zero odd-lag marginal covariance while individual draws retain temporal dependence."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", action="append", required=True, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-memory-mb", type=float, default=512)
    args = parser.parse_args(argv)
    if len(args.ablation) > 4 or len(set(args.ablation)) != len(args.ablation):
        raise ValueError("Provide at most four distinct, saved P5 pairs")
    input_hashes, cases = {}, []
    with single_thread_runtime():
        source, environment = source_fingerprint(), jsonable(benchmark_environment())
        tool_sha = file_hash(__file__)
        for directory in args.ablation:
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            if manifest["source_fingerprint"] != source:
                raise ValueError("Prior receipt must bind a pair from the current package source")
            if not _verify_terminal_inventory(directory, manifest, "extensions"):
                raise ValueError("The saved P5 pair lacks a completed terminal inventory")
            for name in ("manifest.json", "output_inventory.json"):
                input_hashes[str(directory / name)] = file_hash(directory / name)
            states = []
            for name in ("scalar_reference", "full_residual_ar"):
                path = directory / name / "fitted.npz"
                with ZipFile(path) as archive:
                    expanded = sum(entry.file_size for entry in archive.infolist())
                allowance = memory_allowance(args.max_memory_mb)
                # At most two small archives are held together for exact design checks.
                estimated = 8 * expanded + 64 * 1024 ** 2
                if allowance["effective_job_budget_bytes"] is not None and estimated > allowance["effective_job_budget_bytes"]:
                    raise MemoryError("Prior diagnostic archive estimate exceeds current safe allowance")
                input_hashes[str(path)] = file_hash(path)
                state = load_state(path)
                states.append((state["prepared"], SamplerConfig(**state["sampler"])))
                del state  # Posterior states are never used by this diagnostic.
            (base, base_sampler), (ar, ar_sampler) = states
            if base_sampler.temporal != "iid" or ar_sampler.temporal != "full_residual_ar":
                raise ValueError("Expected the evaluated iid/full-residual-AR pair")
            for key in ("Phi", "Z", "reference_Z"):
                if not np.array_equal(base[key], ar[key]):
                    raise ValueError("The two variants do not have the identical frozen feature design")
            for key in ("response_center", "response_scale", "split"):
                if base[key] != ar[key]:
                    raise ValueError("The two variants have different response units or fitting windows")
            if not np.array_equal(base["data"]["coordinates"], ar["data"]["coordinates"]):
                raise ValueError("The two variants use different spatial coordinates")
            for key in ("prior", "covariance", "features", "spatial_mean"):
                if base["spec"][key] != ar["spec"][key]:
                    raise ValueError("Pair differs beyond the declared temporal residual law")
            if base_sampler.a0 != ar_sampler.a0 or base_sampler.b0 != ar_sampler.b0:
                raise ValueError("Pair differs in its noise-scale prior")
            moments = prior_moments(base, ar_sampler)
            cases.append({"ablation_directory": str(directory), "manifest_identity": manifest["identity"],
                "identical_frozen_design_and_shared_priors_verified": True,
                "saved_scalar_sampler_temporal": base_sampler.temporal,
                "saved_ar_sampler_temporal": ar_sampler.temporal, "results": moments})
        result = {"schema": "conditional-fixed-design-prior-check-1.0", "source_fingerprint": source,
            "tool_sha256": tool_sha, "environment": environment, "input_hashes": input_hashes,
            "posterior_samples_used": False, "future_labels_used": False, "models_refitted": False,
            "prior_retuning_performed": False, "extension_promotion": False,
            "autonomous_prior_rollouts_assessed": False,
            "scope": "Analytical prior-predictive first/second moments conditional on the saved encoder, basis, historical features and response standardization; does not certify a self-generated trajectory prior or model adequacy",
            "prior_law": {
                "sigma2": "p(s)=b0^a0/Gamma(a0) * s^(-a0-1) * exp(-b0/s), s>0",
                "lambda_q_ell": "Independent lognormal hyperpriors with the actual per-case means and standard deviations recorded below; sigma2 and rho are independent of them",
                "C": "vec(C)|sigma2,lambda ~ N(0,(sigma2/lambda) I)",
                "rho": "eta~N(0,rho_prior_sd^2), rho=tanh(eta); p(rho)=phi(atanh(rho)/sd)/(sd*(1-rho^2)), -1<rho<1",
                "residual": "iid: e_t|sigma2,q,ell~N(0,sigma2*R), independent in time. AR: e_1~N(0,sigma2*R), e_t=rho*e_(t-1)+sqrt(1-rho^2)*xi_t, xi_t~N(0,sigma2*R). R=I+q*K_ell."},
            "mathematical_identities": {
                "mean": "E[Y_it|fixed design]=response_center",
                "coefficient_variance": "E[sigma2]*E[1/lambda]",
                "readout_covariance_standardized": "E[sigma2]*E[1/lambda]*(Phi Phi^T)_ij*(Z Z^T)_tu",
                "residual_covariance_standardized": "E[sigma2]*(I+E[q]*E[K_ell])_ij*E[rho^|t-u|] for AR; replace the last factor by 1(t=u) for iid",
                "original_units": "Multiply every standardized covariance by response_scale^2; add the frozen response_center to the prior mean"},
            "limitations": ["Conditional historical feature paths are held fixed; autonomous ESN feedback, stability and prior trajectory plausibility are unassessed.",
                "Pointwise amplitude checks and finite prior variance do not establish a well-calibrated prior or forecast coverage.",
                "Zero odd-lag covariance after averaging over a symmetric rho prior is not absence of dependence in individual draws.",
                "The check is retrospective documentation of unchanged priors; it is not a prospective acceptance gate or evidence of additional predictive replications.",
                "The G3 AR fit's qualification failure does not prevent checking its mathematical prior, and these moments do not remedy that failure."],
            "cases": cases}
        if source_fingerprint() != source or file_hash(__file__) != tool_sha or jsonable(benchmark_environment()) != environment:
            raise ValueError("Source, diagnostic tool, or numerical environment changed during the check")
        if any(file_hash(path) != value for path, value in input_hashes.items()):
            raise ValueError("A bound input changed during the check")
        result = jsonable(result)
        result["identity"] = content_hash(result)
        if args.output.exists() and json.loads(args.output.read_text()) != result:
            raise ValueError("Existing prior evidence differs; use a new output path")
        write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "identity": result["identity"],
                      "cases": len(cases), "quadrature_checks_passed": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
