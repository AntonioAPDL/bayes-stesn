#!/usr/bin/env python3
"""Measure exact checkpoint-cadence equivalence on a pilot-size ESN fit.

Run from the repository root after installing the package::

    python benchmarks/profile_checkpoint_cadence.py

Defaults use400 observed times,25 observed sites,width32,4 chains,400 warmup,
1000 draws and one allowed1000-draw continuation. This is an engineering
experiment, not evidence of predictive superiority. CPU thread counts are
fixed before importing numerical libraries. No sampler or model law changes.
"""
from __future__ import annotations

import os
for _thread_key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_thread_key]="1"

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
from unittest import mock
import zipfile

import numpy as np

from bayes_stesn.benchmark.esn import fit_esn
from bayes_stesn.benchmark.generators import generate
from bayes_stesn.io import content_hash,environment,load_state,save_state


def _write(path,receipt):
    path.parent.mkdir(parents=True,exist_ok=True)
    staging=path.with_name(path.name+".tmp")
    staging.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(staging,path)


def run(args):
    root=Path(__file__).resolve().parents[1]
    output=Path(args.output)
    sources=("benchmark/esn.py","sampling.py","gaussian.py","model.py","prediction.py","io.py")
    hashes={name:hashlib.sha256((root/"src"/"bayes_stesn"/name).read_bytes()).hexdigest() for name in sources}
    revision=subprocess.run(["git","rev-parse","HEAD"],cwd=root,text=True,capture_output=True,check=True).stdout.strip()
    simulation=generate("g1",400,n_side=7,n_observed=25,n_calibration=4,seed=args.data_seed)
    data=simulation.observed_panel()
    config={"conditioning":80,
        "features":{"modes":8,"units":[args.width]*args.layers,"squares":min(32,args.width//2),
                    "lags":[1,2],"input_scales":[.5]*args.layers},
        "spatial_mean":{"side":3,"width":.3},
        "sampler":{"chains":args.chains,"warmup":args.warmup,"draws":args.draws},
        "diagnostics":{"functional_draws":args.functional_draws},
        "continuation_draws":args.continuation}
    receipt={"schema_version":1,"status":"running","purpose":"engineering_checkpoint_cadence_equivalence",
        "scientific_performance_evidence":False,"git_head":revision,"source_sha256":hashes,
        "environment":environment(),"config":config,"fit_seed":args.seed,"data_seed":args.data_seed,
        "observation_sha256":simulation.metadata["observation_sha256"],
        "observed_panel_sha256":content_hash(data.y),"times":400,"observed_sites":25,
        "method":"unprofiled wall/CPU timings with checkpoint-writer timing instrumentation",
        "cases":{}}
    _write(output,receipt)
    fitted={};checkpoint_hashes={}
    with tempfile.TemporaryDirectory(prefix="bstesn-cadence-profile-") as temporary:
        for name,cadence in (("periodic_100",100),("benchmark_default",None)):
            path=Path(temporary)/(name+".npz")
            writes=[]
            def tracked_save(destination,state,**kwargs):
                wall=time.perf_counter();cpu=time.process_time()
                result=save_state(destination,state,**kwargs)
                writes.append({"wall_seconds":time.perf_counter()-wall,
                    "cpu_seconds":time.process_time()-cpu,
                    "retained_states":sum(len(chain["retained_states"]) for chain in state["chains"].values()),
                    "archive_bytes":Path(destination).stat().st_size})
                return result
            candidate={**config,"sampler":dict(config["sampler"])}
            if cadence is not None:
                candidate["sampler"]["checkpoint_every"]=cadence
            wall=time.perf_counter();cpu=time.process_time()
            with mock.patch("bayes_stesn.model.save_state",side_effect=tracked_save):
                model=fit_esn(data,candidate,args.seed,checkpoint_path=path)
            elapsed_wall=time.perf_counter()-wall;elapsed_cpu=time.process_time()-cpu
            checkpoint=load_state(path,kind="mcmc_checkpoint")
            checkpoint_hashes[name]=content_hash(checkpoint["chains"])
            with zipfile.ZipFile(path) as archive:
                members=len(archive.namelist())
            fitted[name]=model
            p=model.fitted.posterior
            receipt["cases"][name]={
                "checkpoint_cadence":model.diagnostics["checkpoint_cadence"],
                "fit_wall_seconds":elapsed_wall,"fit_cpu_seconds":elapsed_cpu,
                "checkpoint_writes":len(writes),
                "retained_states_rewritten":sum(write["retained_states"] for write in writes),
                "checkpoint_wall_seconds":sum(write["wall_seconds"] for write in writes),
                "checkpoint_cpu_seconds":sum(write["cpu_seconds"] for write in writes),
                "checkpoint_bytes_written":sum(write["archive_bytes"] for write in writes),
                "final_checkpoint_bytes":path.stat().st_size,"final_checkpoint_zip_members":members,
                "final_checkpoint_chain_state_sha256":checkpoint_hashes[name],
                "posterior_chains_sha256":content_hash(p.chains),
                "posterior_log_density_sha256":content_hash(p.log_density),
                "diagnostic_gates_sha256":content_hash(model.diagnostics["gates"]),
                "qualified":model.diagnostics["qualified"],"continuations":model.diagnostics["continuations"],
                "draws_per_chain":p.n_draws,"spatial_basis":model.diagnostics["spatial_basis"],
                "temporal_features":model.diagnostics["temporal_features"]}
            _write(output,receipt)
            print(json.dumps({"case":name,**receipt["cases"][name]},sort_keys=True),flush=True)
    old,new=fitted["periodic_100"],fitted["benchmark_default"]
    a,b=old.fitted.posterior,new.fitted.posterior
    pa=a.draw_parameters(0,a.n_draws-1,np.random.default_rng(81))
    pb=b.draw_parameters(0,b.n_draws-1,np.random.default_rng(81))
    forecast_a=old.forecast(8,n_paths=8,seed=219)
    forecast_b=new.forecast(8,n_paths=8,seed=219)
    equal={"posterior_chains":bool(np.array_equal(a.chains,b.chains)),
        "posterior_log_density":bool(np.array_equal(a.log_density,b.log_density)),
        "acceptance":bool(np.array_equal(a.acceptance,b.acceptance)),
        "diagnostic_gates":old.diagnostics["gates"]==new.diagnostics["gates"],
        "qualification":old.diagnostics["qualified"]==new.diagnostics["qualified"],
        "checkpoint_chain_states":checkpoint_hashes["periodic_100"]==checkpoint_hashes["benchmark_default"],
        "recovered_sigma2":pa.sigma2==pb.sigma2,"recovered_C":bool(np.array_equal(pa.C,pb.C)),
        "forecast_mean":bool(np.array_equal(forecast_a.mean,forecast_b.mean)),
        "forecast_samples":bool(np.array_equal(forecast_a.samples,forecast_b.samples))}
    old_result,new_result=receipt["cases"]["periodic_100"],receipt["cases"]["benchmark_default"]
    receipt.update(status="complete" if all(equal.values()) else "equivalence_failed",
        exact_numeric_equality=equal,
        wall_speedup=old_result["fit_wall_seconds"]/new_result["fit_wall_seconds"],
        cpu_speedup=old_result["fit_cpu_seconds"]/new_result["fit_cpu_seconds"],
        elapsed_and_checkpoint_policy_excluded_from_equality=True,
        posterior_identity_note="Artifact identities include sampler execution configuration; numerical posterior arrays are compared directly.")
    _write(output,receipt)
    print(json.dumps({"status":receipt["status"],"wall_speedup":receipt["wall_speedup"],
                      "cpu_speedup":receipt["cpu_speedup"],"exact_numeric_equality":equal},sort_keys=True),flush=True)
    if not all(equal.values()):
        raise AssertionError("Checkpoint cadence changed numerical results")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="results/benchmark_v3/engineering/checkpoint_cadence.json")
    parser.add_argument("--warmup",type=int,default=400)
    parser.add_argument("--draws",type=int,default=1000)
    parser.add_argument("--continuation",type=int,default=1000)
    parser.add_argument("--chains",type=int,default=4)
    parser.add_argument("--functional-draws",type=int,default=512)
    parser.add_argument("--width",type=int,default=32)
    parser.add_argument("--layers",type=int,default=1)
    parser.add_argument("--seed",type=int,default=411)
    parser.add_argument("--data-seed",type=int,default=7611)
    run(parser.parse_args())
