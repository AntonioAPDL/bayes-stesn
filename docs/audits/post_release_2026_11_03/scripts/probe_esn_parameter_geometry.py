"""Descriptive checkpoint geometry; no additional MCMC fit."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np

AUDIT_BASE_COMMIT = 'f279c7a47ebe76bc282e0592fa266e96b163acf9'
SCIENTIFIC_SNAPSHOT_COMMIT = '8ee53dfe9f2727d4bf25e26f38d4690007140fd9'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,default=Path.cwd()/'publication_release',
                        help='Directory containing bayes_stesn_v02_pilot_results.zip')
    parser.add_argument('--output',type=Path,default=Path.cwd()/'audit_output',
                        help='Output directory; existing esn_parameter_geometry.json is replaced')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    result={
        'audit_base_commit':AUDIT_BASE_COMMIT,
        'scientific_snapshot_commit':SCIENTIFIC_SNAPSHOT_COMMIT,
        'scientific_snapshot_tag':'v0.2.0',
        'scope':'Descriptive existing chains; G3 chains are unqualified, so correlations are diagnostic, not established posterior quantities',
        'original_parameters':['prior.log_precision','log_ell','log_q'],
        'transform':'eta = log_precision + logaddexp(0, log_q); xi = log_q',
        'correlation_pairs':['log_precision, log_q','eta, xi'],
        'cases':[],
    }
    with zipfile.ZipFile(args.assets/'bayes_stesn_v02_pilot_results.zip') as z:
        for scenario in ['g1_gaussian_ar','g3_excitable_field','g4_existing_fronts']:
            n=f'bayes-stesn/results/benchmark_v3/pilot_release/{scenario}/rep_000/A/bayes_esn_ridge/chain.npz'
            with np.load(io.BytesIO(z.read(n)),allow_pickle=False) as f:
                s=json.loads(str(f['__manifest__']))['state']
                chains=[]
                for c in s['chains'].values():
                    d=c['retained_states']
                    mat=np.array([[float(f[t['prior_state']['log_precision']['__array__']][0]),
                                   t['covariance_parameters']['log_ell'],t['covariance_parameters']['log_q']] for t in d])
                    eta=mat[:,0]+np.logaddexp(0,mat[:,2])
                    chains.append({
                        'chain':c['chain_index'],
                        'original_correlation':float(np.corrcoef(mat[:,0],mat[:,2])[0,1]),
                        'transformed_correlation':float(np.corrcoef(eta,mat[:,2])[0,1]),
                        'original_correlation_matrix':np.corrcoef(mat.T).tolist(),
                        'initialization':c['initialization']['method'],
                        'initialization_optimizer_success':c['initialization'].get('optimizer_success'),
                        'draws':len(d),'sampling_acceptance':c['accepted_sampling']/c['proposals_sampling'],
                    })
                result['cases'].append({'scenario':scenario,'track':'A','replication':0,'chains':chains})
    out=args.output/'esn_parameter_geometry.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    for case in result['cases']:
        print(case['scenario'],[(c['original_correlation'],c['transformed_correlation']) for c in case['chains']])
    print(out)


if __name__=='__main__':
    main()
