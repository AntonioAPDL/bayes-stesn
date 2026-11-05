#!/usr/bin/env python3
"""Training-only check of the already-frozen contiguous robustness history.

No posterior fit, forecast score, selection or history-policy change occurs.
This is an empirical feature-map replay check, not a contraction proof.
"""
from pathlib import Path
import argparse
import hashlib
import json
import resource
import time
import numpy as np
from bayes_stesn.applications.ks_data import KSAdapter
from bayes_stesn.applications.ks_native import resolve_native_config, _resolve_bayes, _prepare_delivered_scale


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--freeze',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    frozen=json.loads(args.freeze.read_text())
    chosen=frozen['selected']['A']['bayes_stesn']
    history=frozen['history'];seed=frozen['model_seed']
    config=resolve_native_config('bayes_stesn',chosen['config'],seed)
    config,spec,_=_resolve_bayes(config,seed)
    started=time.perf_counter();rows=[]
    for phase,episodes in [('pilot',range(3)),('core',range(20))]:
        adapter=KSAdapter(args.cache,'contiguous',phase)
        for episode in episodes:
            tr=adapter.training(episode)
            prepared=_prepare_delivered_scale(tr.panel,config,spec)
            y=(tr.panel.y-prepared.response_center)/prepared.response_scale
            full,_=prepared.features.history(y)
            reference,z=prepared.features.update(y[-1],full[-1])
            bounded,_=prepared.features.history(y[-history:])
            state,z_bounded=prepared.features.update(y[-1],bounded[-1])
            state_error=float(np.max(np.abs(state-reference)))
            feature_error=float(np.max(np.abs(z_bounded-z)))
            state_scale=max(float(np.max(np.abs(reference))),1e-12)
            feature_scale=max(float(np.max(np.abs(z))),1e-12)
            passed=state_error<=1e-8+1e-6*state_scale and feature_error<=1e-8+1e-6*feature_scale
            rows.append({'phase':phase,'trajectory':episode,'permitted_sites':tr.panel.y.shape[1],
                'permitted_training_rows':len(y),'history':history,'passed':bool(passed),
                'state_max_absolute_difference':state_error,'state_max_relative_difference':state_error/state_scale,
                'feature_max_absolute_difference':feature_error,'feature_max_relative_difference':feature_error/feature_scale})
    result={'schema':'ks-contiguous-frozen-history-check-1','freeze_identity':frozen['identity'],
        'freeze_sha256':hashlib.sha256(args.freeze.read_bytes()).hexdigest(),
        'selected_config':chosen['id'],'model_seed':seed,'history':history,
        'absolute_tolerance':1e-8,'relative_tolerance':1e-6,'cases':rows,
        'passed':sum(x['passed'] for x in rows),'total':len(rows),
        'posterior_fits':0,'forecast_scores_computed':0,'history_or_selection_changed':False,
        'observation_boundary':'only permitted contiguous training rows/sites returned to feature computation; no validation/test losses read',
        'scope':'post-freeze training-only check of the transferred history; no claim of future-free physical disk access or universal synchronization',
        'elapsed_seconds':time.perf_counter()-started,
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
    if result['passed']!=result['total']:raise SystemExit(2)


if __name__=='__main__':main()
