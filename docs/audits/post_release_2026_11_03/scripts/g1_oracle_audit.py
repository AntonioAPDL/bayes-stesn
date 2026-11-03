#!/usr/bin/env python3
"""Known-parameter, observation-history-only G1 diagnostic on archived pilot data.

Example (Linux, project dependencies already installed):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python g1_oracle_audit.py --source-root /path/to/bayes-stesn \
      --archive /path/to/bayes_stesn_v02_pilot_results.zip --output g1_oracle_audit.json

No refitting or simulation is performed. Hidden-state and latent-field NPZ
members are never loaded. Future observations enter scoring only after the
oracle returns. The oracle knows true parameters and is a privileged reference,
not a learned competitor. Its analytic Gaussian CRPS is distinguished from the
archived models' finite-ensemble empirical CRPS.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import zipfile

def gaussian_crps(mean, variance, observation):
    import numpy as np
    from scipy.special import ndtr

    sd = np.sqrt(variance)
    z = (observation - mean) / sd
    density = np.exp(-0.5 * z*z) / np.sqrt(2*np.pi)
    return sd * (z*(2*ndtr(z)-1) + 2*density - 1/np.sqrt(np.pi))


def load_permitted_fields(blob):
    """Read only evaluator observations, geometry, roles, and DGP metadata."""
    import numpy as np

    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        manifest = json.loads(str(archive["__manifest__"]))
        assert manifest["kind"] == "benchmark_simulation"
        state = manifest["state"]
        fields = {key: archive[state[key]["__array__"]] for key in
                  ("observation", "coordinates", "observed_indices", "test_indices")}
        fields["metadata"] = state["metadata"]
    observation_hash = hashlib.sha256(
        np.ascontiguousarray(fields["observation"], dtype="<f8").tobytes()).hexdigest()
    assert observation_hash == fields["metadata"]["observation_sha256"]
    return fields


def score(mean, variance, truth):
    import numpy as np
    from scipy.stats import norm

    squared = (mean-truth)**2
    crps = gaussian_crps(mean, variance, truth)
    radius = norm.ppf(.95)*np.sqrt(variance)
    bands = {}
    for first, last in ((1,10),(11,30),(31,40)):
        sl = slice(first-1,last)
        bands[f"{first}-{last}"] = {
            "rmse": float(np.sqrt(squared[sl].mean())),
            "crps": float(crps[sl].mean()),
            "root_mean_conditional_variance": float(np.sqrt(variance[sl].mean()))}
    return {"rmse": float(np.sqrt(squared.mean())),
            "crps": float(crps.mean()),
            "coverage_90": float(np.mean(np.abs(truth-mean)<=radius)),
            "root_mean_conditional_variance": float(np.sqrt(variance.mean())),
            "rmse_by_lead": np.sqrt(squared.mean(axis=1)).tolist(),
            "crps_by_lead": crps.mean(axis=1).tolist(),
            "root_mean_conditional_variance_by_lead": np.sqrt(variance.mean(axis=1)).tolist(),
            "bands": bands}


def averages(records, key):
    import numpy as np

    return {metric: float(np.mean([r[key][metric] for r in records]))
            for metric in ("rmse", "crps")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    signal.alarm(60)
    import numpy as np

    started = time.perf_counter()
    sys.path.insert(0,str(args.source_root.resolve()/"src"))
    from bayes_stesn.benchmark.generators import gaussian_ar_oracle_moments

    rows=[]
    with zipfile.ZipFile(args.archive) as archive:
        suffix="/pilot_release/g1_gaussian_ar/rep_000/simulation.npz"
        first=next(name for name in archive.namelist() if name.endswith(suffix))
        prefix=first[:-len("g1_gaussian_ar/rep_000/simulation.npz")]
        for replication in range(5):
            base=f"{prefix}g1_gaussian_ar/rep_{replication:03d}/"
            member=base+"simulation.npz"
            blob=archive.read(member)
            fields=load_permitted_fields(blob)
            y=fields["observation"]
            coordinates=fields["coordinates"]
            observed=fields["observed_indices"]
            params=fields["metadata"]["parameters"]
            assert y.shape==(560,49) and len(observed)==25
            assert fields["metadata"]["scenario"]=="g1_gaussian_ar"
            origin,horizon=520,40
            for track,targets in (("A",observed),("B",fields["test_indices"])):
                moments=gaussian_ar_oracle_moments(
                    y[:origin,observed],coordinates[observed],coordinates[targets],horizon,
                    **{k:params[k] for k in ("rho","ell","sigma_e")})
                assert np.isfinite(moments["mean"]).all()
                assert np.isfinite(moments["variance"]).all() and np.min(moments["variance"])>0
                truth=y[origin:origin+horizon,targets]
                oracle=score(moments["mean"],moments["variance"],truth)
                models={}
                for model in ("bayes_esn_ridge","separable_stgp"):
                    receipt=json.loads(archive.read(base+f"{track}/{model}/receipt.json"))
                    expected_ids=[f"site_{j:05d}" for j in targets]
                    assert receipt["origin_index"]==origin
                    assert receipt["target_ids"]==expected_ids
                    assert receipt["dgp_seed"]==fields["metadata"]["seed"]
                    native=receipt["native_scores"]
                    models[model]={
                        "qualified":receipt["qualified"],"fallback":receipt["fallback"],
                        "native_rmse":native.get("rmse"),"native_crps":native.get("crps"),
                        "native_paths":native.get("paths"),
                        "operational_rmse":receipt["operational_scores"]["rmse"],
                        "fallback_model":receipt.get("fallback_model"),
                        "native_bands":native.get("bands"),
                        "operational_bands":receipt["operational_scores"].get("bands")}
                rows.append({"replication":replication,"track":track,"dgp_seed":fields["metadata"]["seed"],
                             "origin":origin,"horizon":horizon,"observed_sites":len(observed),
                             "target_sites":len(targets),"parameters":moments["parameters"],
                             "simulation_member":member,"simulation_member_sha256":hashlib.sha256(blob).hexdigest(),
                             "observation_sha256":fields["metadata"]["observation_sha256"],
                             "oracle":oracle,"models":models})
    summaries=[]
    for track in ("A","B"):
        selected=[r for r in rows if r["track"]==track]
        item={"track":track,"replications":len(selected),"oracle_all":averages(selected,"oracle"),
              "oracle_coverage_90":float(np.mean([r["oracle"]["coverage_90"] for r in selected])),
              "oracle_root_mean_conditional_variance":float(np.mean([r["oracle"]["root_mean_conditional_variance"] for r in selected])),
              "oracle_bands":{band:{metric:float(np.mean([r["oracle"]["bands"][band][metric] for r in selected]))
                                   for metric in ("rmse","crps","root_mean_conditional_variance")}
                              for band in ("1-10","11-30","31-40")},"models":{}}
        for model in ("bayes_esn_ridge","separable_stgp"):
            qualified=[r for r in selected if r["models"][model]["qualified"]]
            item["models"][model]={
                "qualified_replications":[r["replication"] for r in qualified],
                "qualified_count":len(qualified),
                "operational_rmse_all":float(np.mean([r["models"][model]["operational_rmse"] for r in selected])),
                "qualified_native_rmse":float(np.mean([r["models"][model]["native_rmse"] for r in qualified])) if qualified else None,
                "qualified_native_crps":float(np.mean([r["models"][model]["native_crps"] for r in qualified])) if qualified else None,
                "oracle_on_same_qualified":averages(qualified,"oracle") if qualified else None}
        summaries.append(item)
    result={"schema":"g1-known-parameter-oracle-audit-1","source_git_head":subprocess.check_output(
        ["git","-C",str(args.source_root),"rev-parse","HEAD"],text=True).strip(),
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generator_source_sha256":hashlib.sha256((args.source_root/"src/bayes_stesn/benchmark/generators.py").read_bytes()).hexdigest(),
        "archive_name":args.archive.name,"information_policy":"known true parameters; O noisy history only through row 519; future O/U observations used only for scoring; no latent/hidden-state members loaded",
        "scope":"post hoc development diagnostic; privileged known-parameter reference, not a fitted competitor or confirmatory comparison",
        "crps_policy":"oracle analytic Gaussian CRPS; archived native models empirical predictive-distribution CRPS with 128 paths; finite-ensemble bias differs",
        "conditional_variance_policy":"sqrt(mean predictive observation variance) is a root conditional MSE of the oracle mean, not the expected realized RMSE",
        "summaries":summaries,"rows":rows,"wall_seconds":time.perf_counter()-started,
        "peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "limits":{"cpu_seconds":60,"wall_seconds":60,"address_space_bytes":1024**3,"blas_threads":1}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    signal.alarm(0)
    print(json.dumps({k:result[k] for k in ("summaries","wall_seconds","peak_rss_bytes")},indent=2))


if __name__=="__main__":
    main()
