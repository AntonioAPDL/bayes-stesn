"""Read-only audit of release ZIP JSON receipts; no model fitting or extraction."""
from __future__ import annotations

import io
import argparse
import json
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

AUDIT_BASE_COMMIT = 'f279c7a47ebe76bc282e0592fa266e96b163acf9'
SCIENTIFIC_SNAPSHOT_COMMIT = '8ee53dfe9f2727d4bf25e26f38d4690007140fd9'


def stats(values):
    a = np.asarray(values, dtype=float)
    return {'n': len(a), 'min': float(a.min()), 'median': float(np.median(a)),
            'max': float(a.max()), 'mean': float(a.mean())} if len(a) else None


def flatten(d, prefix=''):
    result = {}
    for k, v in d.items():
        key = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            result.update(flatten(v, key))
        else:
            result[key] = json.dumps(v, sort_keys=True)
    return result


def read_campaign(archive, campaign, assets):
    prefix = f'bayes-stesn/results/benchmark_v3/{campaign}/'
    with zipfile.ZipFile(assets / archive) as z:
        receipts = []
        selections = []
        chain_rates = []
        for name in z.namelist():
            if not name.startswith(prefix):
                continue
            if name.endswith('/receipt.json'):
                d = json.loads(z.read(name))
                d['archive_member'] = name
                receipts.append(d)
            if name.endswith('/selection.json'):
                d = json.loads(z.read(name))
                pieces = name[len(prefix):].split('/')
                selections.append(dict(scenario=pieces[0], replication=pieces[1],
                                       track=pieces[2], model_id=pieces[3], **d))
            if name.endswith('/bayes_esn_ridge/chain.npz'):
                with np.load(io.BytesIO(z.read(name)), allow_pickle=False) as f:
                    state = json.loads(str(f['__manifest__']))['state']
                    pieces = name[len(prefix):].split('/')
                    for c in state['chains'].values():
                        chain_rates.append(dict(scenario=pieces[0], replication=pieces[1],
                                                track=pieces[2], chain=c['chain_index'],
                                                acceptance=c['accepted_sampling']/c['proposals_sampling'],
                                                invalid_proposals=c['invalid_proposals'],
                                                proposals=c['proposals_sampling'],
                                                initialization=c['initialization']['method'],
                                                initialization_optimizer_success=c['initialization'].get('optimizer_success')))
        manifest = json.loads(z.read(prefix + 'manifest.json'))
        summary = json.loads(z.read(prefix + 'summary.json'))
    return receipts, selections, chain_rates, manifest, summary


def summarize(receipts, selections, chain_rates, manifest, summary):
    by_model = defaultdict(list)
    by_cell = defaultdict(list)
    for r in receipts:
        by_model[r['model_id']].append(r)
        by_cell[(r['scenario'], r['track'], r['model_id'])].append(r)
    model_table = []
    for model, rows in sorted(by_model.items()):
        model_table.append(dict(model=model, attempts=len(rows), qualified=sum(r['qualified'] for r in rows),
                                fallbacks=sum(r['fallback'] for r in rows),
                                median_total_seconds_including_tuning=statistics.median(r['total_seconds']+r['tuning_seconds'] for r in rows),
                                sum_total_seconds_including_tuning=sum(r['total_seconds']+r['tuning_seconds'] for r in rows),
                                process_lifetime_peak_rss_bytes=stats([r['process_peak_rss_bytes'] for r in rows])))
    cells = []
    for (scenario, track, model), rows in sorted(by_cell.items()):
        qualified = [r for r in rows if r['qualified']]
        prob = [r for r in qualified if r['native_scores'].get('crps') is not None]
        cells.append(dict(scenario=scenario, track=track, model=model, attempts=len(rows),
                          qualified=len(qualified), operational_rmse_mean=statistics.mean(r['operational_scores']['rmse'] for r in rows),
                          qualified_rmse_mean=statistics.mean(r['native_scores']['rmse'] for r in qualified) if qualified else None,
                          qualified_probability_count=len(prob),
                          qualified_crps_mean=statistics.mean(r['native_scores']['crps'] for r in prob) if prob else None,
                          coverage90_mean=statistics.mean(r['native_scores']['intervals']['0.9']['coverage'] for r in prob) if prob else None,
                          width90_mean=statistics.mean(r['native_scores']['intervals']['0.9']['width'] for r in prob) if prob else None,
                          interval_score90_mean=statistics.mean(r['native_scores']['intervals']['0.9']['interval_score'] for r in prob) if prob else None,
                          operational_rmse_by_band={band:statistics.mean(r['operational_scores']['bands'][band]['rmse'] for r in rows)
                                                    for band in rows[0]['operational_scores']['bands']},
                          qualified_crps_by_band={band:statistics.mean(r['native_scores']['bands'][band]['crps'] for r in prob)
                                                 for band in prob[0]['native_scores']['bands']} if prob else {}))
    esn_diagnostics = []
    for r in by_model.get('bayes_esn_ridge', []):
        gates = []
        for g in r['diagnostics']['gates']:
            quantities = {f'{group}.{key}':value for group in ('hyperparameters', 'functionals')
                          for key, value in g[group]['parameters'].items()}
            gates.append(dict(passed=g['passed'], hyperparameter_draws=g['hyperparameters']['draws'],
                              functional_draws=g['functionals']['draws'], quantities=quantities))
        esn_diagnostics.append(dict(scenario=r['scenario'], track=r['track'], replication=r['replication'],
                                    qualified=r['qualified'], continuation=r['diagnostics']['continuations'],
                                    gates=gates))
    esn_first_pass = sum(x['gates'][0]['passed'] for x in esn_diagnostics)
    esn_final_pass = sum(x['gates'][-1]['passed'] for x in esn_diagnostics)
    failures = Counter()
    for x in esn_diagnostics:
        for key, value in x['gates'][-1]['quantities'].items():
            if not value['passed']:
                failures[key] += 1
    tuning = []
    for model in sorted(by_model):
        rows = [r for r in selections if r['model_id'] == model]
        choices = defaultdict(Counter)
        for r in rows:
            for k, v in flatten(r.get('selected', {})).items():
                choices[k][v] += 1
        selected_rechecks = [t for r in rows for t in r.get('rechecks', []) if t['candidate']==r.get('selected_candidate')]
        trials = [t for r in rows for t in r.get('trials', [])]
        tuning.append(dict(model=model, jobs=len(rows), trials_per_job=dict(Counter(len(r.get('trials', [])) for r in rows)),
                           selected_candidate_counts=dict(Counter(r.get('selected_candidate') for r in rows)),
                           selected_values=dict(choices), trial_fits=len(trials),
                           trial_qualified_count=sum(t.get('qualified', False) for t in trials),
                           selected_recheck_fits=len(selected_rechecks),
                           selected_recheck_qualified_count=sum(t.get('qualified', False) for t in selected_rechecks)))
    optimizer_details=[]
    for model in ['basis_lgssm','basis_gru','separable_stgp']:
        for r in by_model.get(model, []):
            d=r['diagnostics']
            keep=['iterations','covariance_iterations','converged','covariance_converged','optimizer_message',
                  'covariance_optimizer_message','boundary_hits','covariance_boundary_hits',
                  'variance_floor_hits','transition_spectral_norm','transition_spectral_radius','rejected_decreasing_step']
            details={k:d[k] for k in keep if k in d}
            if isinstance(d.get('log_likelihood'),list) and len(d['log_likelihood'])>1:
                l=d['log_likelihood']
                details['last_ll_improvement']=l[-1]-l[-2]
                details['last_ll_relative_improvement']=(l[-1]-l[-2])/(1+abs(l[-2]))
            optimizer_details.append(dict(scenario=r['scenario'],track=r['track'],model=model,replication=r['replication'],qualified=r['qualified'],**details))
    return dict(protocol=manifest['protocol'],execution=manifest['execution'],environment=manifest['environment'],
                raw_receipts=len(receipts), model_table=model_table, cells=cells,
                esn_continuation=dict(attempts=len(esn_diagnostics),passed_initial=esn_first_pass,
                                      continued=sum(x['continuation']>0 for x in esn_diagnostics),passed_final=esn_final_pass,
                                      final_failed_quantity_counts=dict(failures)),
                esn_diagnostics=esn_diagnostics, esn_chain_acceptance=chain_rates,
                tuning=tuning,optimizer_details=optimizer_details,
                recorded_summary_table=summary['table'],paired_comparisons=summary['paired_comparisons'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=Path.cwd()/'publication_release',
                        help='Directory containing the two published v0.2.0 scientific ZIP archives')
    parser.add_argument('--output', type=Path, default=Path.cwd()/'audit_output',
                        help='Output directory; existing results_evidence.json is replaced')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    output={'audit_base_commit':AUDIT_BASE_COMMIT,
            'scientific_snapshot_commit':SCIENTIFIC_SNAPSHOT_COMMIT,
            'scientific_snapshot_tag':'v0.2.0'}
    for key, archive, campaign in [
        ('pilot','bayes_stesn_v02_pilot_results.zip','pilot_release'),
        ('detail','bayes_stesn_v02_code_reports_detail.zip','detail_release'),
    ]:
        output[key]=summarize(*read_campaign(archive,campaign,args.assets))
    output['notes']=[
        'No fits run; read-only extraction from distributed release ZIP receipts.',
        'Operational RMSE includes declared persistence/spatial-persistence fallbacks.',
        'Qualified conditional means condition on fitting success and may be selected populations.',
        'Process RSS is a process lifetime high-water mark; not isolated model memory.',
        'Runtime is the sum of per-job elapsed time, not campaign wall time.',
        'DGP replications, not sites, times, or model/task jobs, are independent units.',
    ]
    path=args.output/'results_evidence.json'
    path.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(path, path.stat().st_size)
    for key in ['pilot','detail']:
        print(key,output[key]['esn_continuation'])
        print(json.dumps(output[key]['model_table'],indent=1))


if __name__ == '__main__':
    main()
