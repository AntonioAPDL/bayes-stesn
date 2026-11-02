#!/usr/bin/env python3
"""Verify the new Gaussian engine against the supplied original V3 artifacts.

Usage from the repository root::

    OPENBLAS_NUM_THREADS=1 python benchmarks/verify_v3_parity.py \
      --legacy-root ../spatiotemporal_esn_v3

The optional legacy directory supplies numeric NPZ files and the original
``bayes_core.py`` source.  No pickle is loaded.  The frozen feature map is
replayed from its saved weights; no preprocessing or model selection is
re-estimated.  With no legacy directory, a compact independent dense
observation-space comparison runs and is labelled as such.

This verifies fixed-parameter likelihood and conditional posterior algebra.
It is not a new full-size MCMC fit, a convergence claim, or a forecast result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

import numpy as np
from scipy import linalg, sparse, special
from threadpoolctl import threadpool_limits

from bayes_stesn.gaussian import GaussianRegression, PrecisionSpec
from bayes_stesn.spatial import CovarianceSpec


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def load_numeric(path):
    with np.load(path,allow_pickle=False) as data:
        return {key:data[key] for key in data.files}


def replay_frozen_features(observed,feature):
    configuration = json.loads(str(feature["configuration"]))
    lags = np.asarray(configuration["lags"],int)
    modes = int(configuration["modes"])
    maxlag = int(lags.max())
    history_size = maxlag*modes
    layer_offsets = np.asarray(feature["layer_offsets"],int)
    units = configuration["units"]
    size = history_size+sum(units)
    state = np.zeros(size)
    recurrent = [sparse.csr_matrix(feature[f"W_{j}"]) for j in range(len(units))]
    Z = np.empty((len(observed),len(feature["center"])+1))

    def readout(current):
        history = current[:history_size].reshape(maxlag,modes)
        scores = history[lags-1].ravel()
        if configuration["score_bound"] is not None:
            bound = configuration["score_bound"]
            scores = bound*np.tanh(scores/bound)
        reservoir = current[history_size:]
        raw = np.concatenate((scores,reservoir,reservoir[feature["square_index"]]**2,
                              reservoir[feature["product_i"]]*reservoir[feature["product_j"]]))
        return np.r_[1.,(raw-feature["center"])/feature["scale"]/float(feature["norm"])]

    Z[0] = readout(state)
    for t in range(1,len(observed)):
        history = state[:history_size].reshape(maxlag,modes)
        history = np.vstack(((observed[t-1]-feature["site_center"])@feature["P"].T,history[:-1]))
        updated = np.empty(size)
        updated[:history_size] = history.ravel()
        incoming = history[lags-1].ravel()
        for j in range(len(units)):
            start,end = layer_offsets[j:j+2]
            old = state[start:end]
            normalized = (incoming-feature[f"layer_input_center_{j}"])/feature[f"layer_input_scale_{j}"]
            leaks = feature[f"leaks_{j}"]
            incoming = ((1-leaks)*old+leaks*np.tanh(recurrent[j]@old+
                        feature[f"U_{j}"]@normalized+feature[f"bias_{j}"]))
            updated[start:end] = incoming
        state = updated
        Z[t] = readout(state)
    return Z,state


def functional_covariance(Phi,Z,spatial_vectors,temporal_vectors,denominator,sigma_mean):
    spatial = Phi@spatial_vectors
    temporal = Z@temporal_vectors
    count = len(Phi)
    result = np.empty((count,count))
    inverse_precision = 1/denominator
    for i in range(count):
        for j in range(count):
            result[i,j] = sigma_mean*((spatial[i]*spatial[j])@inverse_precision@(temporal[i]*temporal[j]))
    return result


def full_comparison(root):
    started = time.perf_counter()
    paths = {"training":root/"results/training_observations.npz",
             "features":root/"results/expanded_observations_features.npz",
             "spatial":root/"results/expanded_observations_spatial_dictionary.npz",
             "chains":root/"results/expanded_observations_chains.npz",
             "summary":root/"results/expanded_observations_mcmc_summary.json",
             "legacy_engine":root/"bayes_core.py"}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required numeric/source artifact unavailable: {path}")
    observations = load_numeric(paths["training"])
    feature = load_numeric(paths["features"])
    spatial = load_numeric(paths["spatial"])
    summary = json.loads(paths["summary"].read_text())
    theta = np.median(load_numeric(paths["chains"])["theta"],axis=(0,1))
    observed = observations["observed"]
    Z,state = replay_frozen_features(observed,feature)
    replay_seconds = time.perf_counter()-started
    endpoint_error = max(float(abs(state-feature["last_state"]).max()),float(abs(Z[-1]-feature["last_z"]).max()))
    conditioning = int(feature["conditioning"])
    Y = ((observed[conditioning:]-summary["response_center"])/summary["response_scale"]).T
    Phi = spatial["phi"][spatial["observed_idx"]]
    Z = Z[conditioning:]
    coordinates = observations["coordinates"]
    module_spec = importlib.util.spec_from_file_location("legacy_v3_bayes_core",paths["legacy_engine"])
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    then = time.perf_counter()
    old = module.MatrixRidge(Y,Phi,Z,coordinates)
    old_precompute_seconds = time.perf_counter()-then
    then = time.perf_counter()
    original = old.evaluate(theta,details=True)
    old_evaluation_seconds = time.perf_counter()-then
    if not np.isfinite(original["logpost"]):
        raise FloatingPointError(original.get("error","Legacy evaluation failed"))
    then = time.perf_counter()
    new = GaussianRegression(Y,Phi,Z)
    new_precompute_seconds = time.perf_counter()-then
    covariance = CovarianceSpec()
    params = {"log_q":float(theta[1]),"log_ell":float(theta[2]),"nu":1.5}
    R = covariance.covariance(coordinates,params)
    then = time.perf_counter()
    current = new.evaluate(R,column_precision=float(np.exp(theta[0])))
    new_evaluation_seconds = time.perf_counter()-then
    mean = current.mean
    sigma_old = original["bn"]/(original["an"]-1)
    sigma_new = current.bn/(current.an-1)
    sites = np.repeat([0,len(Phi)//2,len(Phi)-1],3)
    times = np.tile([0,len(Z)//2,len(Z)-1],3)
    fields_old = functional_covariance(Phi[sites],Z[times],original["eigvec_b"],original["eigvec_g"],
                                      original["precision_eigenvalues"],sigma_old)
    fields_new = functional_covariance(Phi[sites],Z[times],current.spatial_vectors,current.temporal_vectors,
                                      current.eigenprecision,sigma_new)
    checks = {
        "frozen_feature_endpoint_max_abs":endpoint_error,
        "R_max_abs":float(abs(R-original["R"]).max()),
        "log_likelihood_abs":abs(current.log_marginal-original["loglik"]),
        "Q_relative":abs(current.Q-original["Q"])/max(abs(original["Q"]),1e-300),
        "bn_relative":abs(current.bn-original["bn"])/abs(original["bn"]),
        "coefficient_mean_relative_frobenius":float(linalg.norm(mean-original["mcoef"])/linalg.norm(original["mcoef"])),
        "coefficient_mean_max_abs":float(abs(mean-original["mcoef"]).max()),
        "field_covariance_relative_frobenius":float(linalg.norm(fields_new-fields_old)/linalg.norm(fields_old)),
        "sigma2_posterior_mean_relative":abs(sigma_new-sigma_old)/sigma_old,
    }
    tolerances = {"frozen_feature_endpoint_max_abs":1e-11,"R_max_abs":1e-12,
                  "log_likelihood_abs":1e-5,"Q_relative":1e-10,"bn_relative":1e-10,
                  "coefficient_mean_relative_frobenius":1e-8,"coefficient_mean_max_abs":1e-7,
                  "field_covariance_relative_frobenius":1e-9,"sigma2_posterior_mean_relative":1e-10}
    dimensions = {"n":new.n,"T":new.T,"p":new.p,"d":new.d,"coefficients":new.k}
    return {"mode":"original_v3_full_dimension_fixed_parameter_parity","passed":all(checks[k]<=v for k,v in tolerances.items()),
            "dimensions":dimensions,"theta_order":["log_lambda","log_q","log_ell"],"theta":theta.tolist(),
            "checks":checks,"tolerances":tolerances,
            "normalized_log_likelihood":{"legacy":original["loglik"],"new":current.log_marginal},
            "timings_seconds":{"feature_replay":replay_seconds,"legacy_precompute":old_precompute_seconds,
                                "new_precompute":new_precompute_seconds,"legacy_evaluation":old_evaluation_seconds,
                                "new_evaluation":new_evaluation_seconds,"total":time.perf_counter()-started},
            "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            "dense_coefficient_precision_bytes_avoided":8*new.k**2,
            "sources":{key:{"file":str(path.relative_to(root)),"sha256":file_hash(path)} for key,path in paths.items()},
            "new_engine_sha256":file_hash(Path(__file__).resolve().parents[1]/"src/bayes_stesn/gaussian.py"),
            "scope":"Fixed hyperparameters, all original training rows/sites/features; no new MCMC or forecast claim.",
            "conditional_covariance_check":"Nine matched spatial/temporal linear functionals; sigma2 integrated analytically.",
            "numerical_note":"Tolerance permits double-precision accumulation/cancellation across 1,945,600 observations; raw errors reported."}


def compact_comparison():
    rng = np.random.default_rng(199)
    Y,Phi,Z = rng.normal(size=(5,8)),rng.normal(size=(5,3)),rng.normal(size=(8,4))
    R = np.eye(5)+.3*np.ones((5,5))
    precision = PrecisionSpec(3,4,"column",[.2,.7,1.5,4.])
    engine = GaussianRegression(Y,Phi,Z)
    ev = engine.evaluate(R,precision=precision)
    X = np.kron(Z,Phi)
    Omega = np.kron(np.eye(8),R)+X@linalg.solve(precision.as_matrix(),X.T,assume_a="pos")
    y = Y.reshape(-1,order="F")
    chol = linalg.cholesky(Omega,lower=True)
    Q = float(y@linalg.cho_solve((chol,True),y))
    expected = (-20*np.log(2*np.pi)-np.log(np.diag(chol)).sum()+2*np.log(.05)
                +special.gammaln(22)-special.gammaln(2)-22*np.log(.05+Q/2))
    error = float(abs(ev.log_marginal-expected))
    return {"mode":"compact_independent_observation_space_reference","passed":error<1e-10,
            "dimensions":{"n":5,"T":8,"p":3,"d":4},"log_likelihood_abs":error,
            "scope":"Standalone algebra check only; original V3 artifacts were not supplied."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root",type=Path)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = full_comparison(args.legacy_root.resolve()) if args.legacy_root else compact_comparison()
    output = args.output or Path("results/validation/v3_full_parity.json" if args.legacy_root else
                                  "results/validation/compact_gaussian_parity.json")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(output),**{k:v for k,v in result.items() if k not in ("sources",)}},indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
