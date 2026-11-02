"""Profile the narrow sparse fit/forecast route in isolated processes.

Execution-only evidence: these deliberately tiny chains do not establish
MCMC convergence, predictive quality, or a cross-platform speed guarantee.
Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1.
The optional first argument is the destination JSON file.
"""
import json
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

CODE = r'''
import json, resource, sys, time
import numpy as np
from bayes_stesn import SpatioTemporalData, SplitPlan, FeatureSpec, SpatialMeanSpec, ModelSpec, SamplerConfig, prepare, fit
from bayes_stesn.prediction import simulate_predictive
from bayes_stesn.spatial import CovarianceSpec
n=int(sys.argv[1])
rng=np.random.default_rng(1987)
coords=rng.uniform(size=(n,2))
y=rng.normal(size=(50,n))
spec=ModelSpec(features=FeatureSpec(modes=2,units=(4,),input_scales=(.4,),squares=2),spatial_mean=SpatialMeanSpec(side=2),covariance=CovarianceSpec(kind='vecchia',vecchia_neighbors=8,max_dense_sites=8))
start=time.perf_counter()
p=prepare(SpatioTemporalData(y,coords),SplitPlan(15,50),spec)
prepared=time.perf_counter()
f=fit(p,SamplerConfig(chains=1,warmup=2,draws=3,initialize='prior',seed=31))
fitted=time.perf_counter()
s=simulate_predictive(f,paths=4,horizon=8,components=('observation',),seed=88)
finished=time.perf_counter()
assert np.isfinite(s.observation).all()
report=s.metadata['solver_reports'][0]
print(json.dumps(dict(sites=n,times=50,fitting_rows=35,neighbors=8,chains=1,warmup=2,draws=3,paths=4,horizon=8,prepare_seconds=prepared-start,fit_seconds=fitted-prepared,predict_seconds=finished-fitted,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,output_bytes=s.metadata['output_storage_bytes'],precision_nonzeros=report['precision_nonzeros'],lu_nonzeros=report['lu_nonzeros'],backend=f.posterior.metadata['backend'],spatial_dense_materialization=f.posterior.metadata['spatial_dense_materialization'])))
'''

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",nargs="?",default=str(Path(__file__).resolve().parents[1]/"results/benchmark_v3/engineering/sparse_profile.json"))
    parser.add_argument("--sites",nargs="+",type=int,default=[128,512,2048])
    args=parser.parse_args()
    if any(n<16 for n in args.sites):
        parser.error("site counts must be at least16")
    rows = []
    environment = dict(os.environ)
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for n in args.sites:
        completed = subprocess.run([sys.executable, "-c", CODE, str(n)], check=True,
                                   capture_output=True, text=True, env=environment, timeout=55)
        row = json.loads(completed.stdout)
        rows.append(row)
        print(json.dumps(row), flush=True)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "execution_only": True, "inference_converged": False,
        "workload_sha256":hashlib.sha256(CODE.encode()).hexdigest(),
        "scope": "engineering smoke only; tiny chains are not convergence evidence; peak RSS includes imports; one process per size; single BLAS thread",
        "rows": rows}, indent=2))
