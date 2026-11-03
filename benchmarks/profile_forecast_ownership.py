"""Fresh-process ownership allocation probe; no statistical fit is performed.

Run from an installed environment or the source tree::

    PYTHONPATH=src python benchmarks/profile_forecast_ownership.py OUTPUT.json

The workload is fixed at 256 paths x 100 leads x 505 targets, stored as float64.
Both fresh subprocesses have one numerical thread and a 60-second timeout.
Peak RSS is scoped to this allocation/constructor workload, not a model fit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

CHILD = r'''
import hashlib,inspect,json,platform,resource,sys,time
from pathlib import Path
import numpy as np
from bayes_stesn.benchmark.contracts import Forecast
mode=sys.argv[1]
shape=(256,100,505)
coordinates=np.column_stack((np.linspace(0,1,shape[2]),np.zeros(shape[2])))
paths=np.empty(shape,dtype=np.float64);paths.fill(1.125)
mean=paths.mean(axis=0)
rss_multiplier=1 if sys.platform=="darwin" else 1024
before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*rss_multiplier
start=time.perf_counter()
factory=Forecast if mode=="public_defensive_copy" else Forecast._from_owned
forecast=factory(mean,paths,coordinates,tuple(map(str,range(shape[2]))))
seconds=time.perf_counter()-start
peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*rss_multiplier
result={"mode":mode,"shape":shape,"paths_bytes":paths.nbytes,"mean_bytes":mean.nbytes,
"peak_rss_before_constructor_bytes":before,"peak_rss_bytes":peak,
"constructor_seconds":seconds,"shares_path_memory":np.shares_memory(paths,forecast.samples),
"runtime":{"python":platform.python_version(),"numpy":np.__version__,"platform":platform.platform()},
"constructor_source_sha256":hashlib.sha256(Path(inspect.getsourcefile(Forecast)).read_bytes()).hexdigest(),
"path_sha256":hashlib.sha256(memoryview(forecast.samples).cast("B")).hexdigest(),
"mean_sha256":hashlib.sha256(memoryview(forecast.mean).cast("B")).hexdigest()}
print(json.dumps(result))
'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    environment=dict(os.environ)
    environment.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    rows=[]
    for mode in ('public_defensive_copy','private_owned_transfer'):
        completed=subprocess.run([sys.executable,'-c',CHILD,mode],env=environment,text=True,
                                 capture_output=True,timeout=60,check=True)
        rows.append(json.loads(completed.stdout))
    assert rows[0]['path_sha256']==rows[1]['path_sha256']
    assert rows[0]['mean_sha256']==rows[1]['mean_sha256']
    assert not rows[0]['shares_path_memory'] and rows[1]['shares_path_memory']
    result={'schema':'benchmark-forecast-ownership-probe-1.0','rows':rows,
            'peak_rss_reduction_bytes':rows[0]['peak_rss_bytes']-rows[1]['peak_rss_bytes'],
            'incremental_peak_reduction_bytes':(rows[0]['peak_rss_bytes']-rows[0]['peak_rss_before_constructor_bytes'])-
                (rows[1]['peak_rss_bytes']-rows[1]['peak_rss_before_constructor_bytes']),
            'method':'two fresh sequential subprocesses; one numeric thread; process-lifetime peak isolated per constructor workload',
            'scope':'fixed 256x100x505 float64 allocation and validation; no claim about total model fit memory or forecast throughput',
            'numerical_payloads_equal':True,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output=args.output;output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
