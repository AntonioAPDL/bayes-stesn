"""Bounded post-hoc optimizer diagnostic; no forecast or test scoring.

Uses the first failed final fit in each scenario/track, ordered lexically,
capped at six. The sole changed setting is max_iter: 50 -> 200. Training,
configuration, seed, initial values, and stopping tolerance are unchanged.
Run in a fresh process. Requires the existing bayes-stesn dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import signal
import time
import zipfile

AUDIT_BASE_COMMIT = 'f279c7a47ebe76bc282e0592fa266e96b163acf9'
SCIENTIFIC_SNAPSHOT_COMMIT = '8ee53dfe9f2727d4bf25e26f38d4690007140fd9'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=Path.cwd()/'publication_release',
                        help='Directory containing bayes_stesn_v02_pilot_results.zip')
    parser.add_argument('--output', type=Path, default=Path.cwd()/'audit_output',
                        help='Output directory for receipt and small simulation files')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data_dir=args.output/'data';data_dir.mkdir(exist_ok=True)
    out=args.output/'lgssm_iteration_probe.json'
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='1'
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(110,115))

    import numpy as np
    from bayes_stesn.benchmark.classical import fit_classical
    from bayes_stesn.benchmark.generators import Simulation
    from bayes_stesn.benchmark.linear_state import LinearStateFit

    source_path=Path(inspect.getsourcefile(LinearStateFit))
    result={
        'audit_base_commit':AUDIT_BASE_COMMIT,
        'scientific_snapshot_commit':SCIENTIFIC_SNAPSHOT_COMMIT,
        'scientific_snapshot_tag':'v0.2.0',
        'linear_state_source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'scope':'Post-hoc development optimizer diagnostic; no forecasts/test scores',
        'selection':'First failed receipt per scenario/track in lexical order, maximum six',
        'only_changed_setting':{'max_iter':[50,200]},
        'tolerance':1e-5,
        'resource_limits':{'address_space_bytes':1024**3,'cpu_soft_seconds':110,
                           'cpu_hard_seconds':115,'wall_seconds':115,'threads':1},
        'rescue_rate_interpretation':'Selected jobs are dependent: A/B can use the same data and deterministic fit; this is not a qualification-rate estimate on independent replications.',
        'cases':[],
    }

    def save():
        out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

    def stop(signum,frame):
        raise TimeoutError('Bounded audit probe reached its resource time limit')

    signal.signal(signal.SIGXCPU,stop)
    signal.signal(signal.SIGALRM,stop)
    signal.alarm(115)
    started=time.perf_counter()
    try:
        with zipfile.ZipFile(args.assets/'bayes_stesn_v02_pilot_results.zip') as z:
            selected={}
            for name in sorted(z.namelist()):
                if not name.endswith('/basis_lgssm/receipt.json'):
                    continue
                r=json.loads(z.read(name))
                key=(r['scenario'],r['track'])
                if not r['qualified'] and key not in selected:
                    selected[key]=r
            for (scenario,track),r in list(sorted(selected.items()))[:6]:
                rep=r['replication']
                member=f'bayes-stesn/results/benchmark_v3/pilot_release/{scenario}/rep_{rep:03d}/simulation.npz'
                data_path=data_dir/f'{scenario}_rep{rep:03d}_simulation.npz'
                data_path.write_bytes(z.read(member))
                sim=Simulation.load(data_path)
                # Only the supplied observed training prefix enters fitting.
                training=sim.observed_panel(stop=r['origin_index'])
                config=dict(r['fit_config']);config['max_iter']=200
                tick=time.perf_counter()
                fit=fit_classical('basis_lgssm',training,config,r['final_fit_seed'])
                elapsed=time.perf_counter()-tick
                old=r['diagnostics'];new=fit.diagnostics
                old_ll=np.asarray(old['log_likelihood']);new_ll=np.asarray(new['log_likelihood'])
                prefix_error=float(np.max(np.abs(old_ll-new_ll[:len(old_ll)])))
                entry={
                    'scenario':scenario,'track':track,'replication':rep,
                    'training_rows':len(training.y),'observed_sites':training.y.shape[1],
                    'seed':r['final_fit_seed'],'original_config':r['fit_config'],
                    'baseline_qualified':r['qualified'],'new_qualified':new['qualified'],
                    'baseline_iterations':old['iterations'],'new_iterations':new['iterations'],
                    'baseline_log_likelihood':old_ll[-1].item(),'new_log_likelihood':new_ll[-1].item(),
                    'original_likelihood_prefix_max_abs_difference':prefix_error,
                    'original_likelihood_prefix_matches_rtol1e_12_atol1e_10':bool(np.allclose(old_ll,new_ll[:len(old_ll)],rtol=1e-12,atol=1e-10)),
                    'new_last_relative_ll_improvement':float((new_ll[-1]-new_ll[-2])/(1+abs(new_ll[-2]))),
                    'new_rejected_decreasing_step':new['rejected_decreasing_step'],
                    'new_variance_floor_hits':new['variance_floor_hits'],
                    'baseline_recorded_fit_seconds':r['fit_seconds'],'new_fit_seconds':elapsed,
                }
                result['cases'].append(entry);save()
                print(json.dumps(entry),flush=True)
        result['status']='completed'
    except Exception as error:
        result['status']='bounded_or_failed';result['error']=f'{type(error).__name__}: {error}'
    finally:
        signal.alarm(0)
        result['wall_seconds']=time.perf_counter()-started
        result['cpu_seconds']=time.process_time()
        result['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        result['distinct_dataset_count']=len({(r['scenario'],r['replication']) for r in result['cases']})
        result['distinct_dataset_config_count']=len({(r['scenario'],r['replication'],json.dumps(r['original_config'],sort_keys=True)) for r in result['cases']})
        save()
        print(out,result['status'],flush=True)


if __name__=='__main__':
    main()
