#!/usr/bin/env python3
"""Run a numerical DGP pilot; never fit or select forecasting models.

Use --extended for a longer, post-burn-in FHN refinement study. The default
uses one CPU thread when launched with OPENBLAS_NUM_THREADS=1 and checks a
short matched physical window. Refinement failures are retained, not hidden.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import scipy
from bayes_stesn.benchmark.generators import generate, spectral_spec
from bayes_stesn.io import content_hash


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="results/benchmark_v2/generator_validation.json")
    parser.add_argument("--extended",action="store_true")
    parser.add_argument("--grid-size",type=int,default=128)
    parser.add_argument("--dt",type=float,default=.05)
    parser.add_argument("--delta",type=float,default=1.)
    parser.add_argument("--observations",type=int)
    parser.add_argument("--burnin",type=float)
    parser.add_argument("--seed",type=int,default=983417)
    args=parser.parse_args()
    started=time.perf_counter()
    common=dict(n_side=7,n_observed=25,n_calibration=4,seed=args.seed,
                n_times=args.observations or (101 if args.extended else 40),
                burnin=args.burnin if args.burnin is not None else (200. if args.extended else 20.),
                delta=args.delta,sigma_e=0)
    runs={}
    names=(f"mesh{args.grid_size//2}",f"reference{args.grid_size}",
           f"time_half{args.grid_size}",f"mesh{2*args.grid_size}")
    for name,grid,dt in [(names[0],args.grid_size//2,args.dt),(names[1],args.grid_size,args.dt),
                         (names[2],args.grid_size,args.dt/2),(names[3],2*args.grid_size,args.dt/2)]:
        before=time.perf_counter()
        sim=generate("g3",grid_size=grid,dt=dt,**common)
        runs[name]={"simulation":sim,"seconds":time.perf_counter()-before}
        print(json.dumps({"run":name,"seconds":runs[name]["seconds"]}),flush=True)
    def difference(a,b):
        x=runs[a]["simulation"].latent-runs[b]["simulation"].latent
        return {"rmse":float(np.sqrt(np.mean(x*x))),"max_abs":float(np.max(np.abs(x)))}
    time_error=difference(names[1],names[2])
    mesh_error=difference(names[2],names[3])
    coarse_error=difference(names[0],names[3])
    dynamics=runs[names[1]]["simulation"].metadata["parameters"]
    spec=spectral_spec()
    receipt={
        "schema":"benchmark-generator-validation-2.0",
        "status":"passed" if time_error["rmse"]<.02 and mesh_error["rmse"]<.02 and dynamics["mean_temporal_sd"]>.05 else "refinement_gate_not_met",
        "scope":"simulator-only validation; no final benchmark data or model scores inspected",
        "extended":args.extended,"seed":common["seed"],"physical_burnin":common["burnin"],
        "n_observations":common["n_times"],"observation_spacing":common["delta"],
        "fhn_time_halving":time_error,"fhn_mesh_reference_to_fine":mesh_error,
        "fhn_mesh_coarse_to_fine":coarse_error,
        "finite_grid_policy":{"target":"finite_semidiscrete","grid_size":args.grid_size,
            "dt":args.dt,"refined_dt":args.dt/2,"delta":args.delta,
            "time_gate_passed":time_error["rmse"]<.02,"morphology_gate_passed":dynamics["mean_temporal_sd"]>.05,
            "continuum_claim":False,"mesh_gate_passed":mesh_error["rmse"]<.02,
            "physical_parameters":{key:dynamics[key] for key in ("a","b","epsilon","kappa")}},
        "proposed_threshold_rms":.02,"observation_noise_sd_reference":.1,
        "fhn_mean_temporal_sd":dynamics["mean_temporal_sd"],
        "fhn_energy_bounds":[dynamics["energy_min"],dynamics["energy_max"]],
        "fhn_max_abs_state":dynamics["max_abs_state_at_observation_times"],
        "spectral_stationary_variance":float(spec["zero_power"]+2*sum(spec["power"])),
        "spectral_max_transition_modulus":float(max(spec["zero_multiplier"],np.max(np.abs(spec["multiplier"])))),
        "runtimes":{k:v["seconds"] for k,v in runs.items()},"total_seconds":time.perf_counter()-started,
        "qualification":"Refinement establishes only the declared pilot window; convergence must be assessed at longer horizons before continuum-fidelity claims.",
        "numerical_versions":{"numpy":np.__version__,"scipy":scipy.__version__},
    }
    import bayes_stesn.benchmark.generators as generators
    receipt["generator_source_sha256"]=hashlib.sha256(Path(generators.__file__).read_bytes()).hexdigest()
    receipt["validation_tool_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipt["identity"]=content_hash(receipt)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))
    return 0 if receipt["status"]=="passed" else 2


if __name__=="__main__":
    raise SystemExit(main())
