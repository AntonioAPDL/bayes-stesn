"""Move a stopped improvement run using verified, immutable scientific evidence.

The repair changes execution bookkeeping only. It refuses any changed model,
candidate, data, forecast or scoring specification and retains spent budgets.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil

from bayes_stesn.io import content_hash
from bayes_stesn.applications.ks_improvement_runner import Campaign, candidate_records
from bayes_stesn.applications.ks_runtime import atomic_json, file_hash, now


def load(path):
    return json.loads(Path(path).read_text())


def verified_copy(source, destination):
    source, destination = Path(source), Path(destination)
    digest = file_hash(source)
    if destination.exists():
        if file_hash(destination) != digest:
            raise ValueError(f'Conflicting recovery destination: {destination}')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    if source.stat().st_size != destination.stat().st_size or file_hash(destination) != digest:
        raise ValueError('Recovery byte-count/checksum mismatch')
    return {'bytes': source.stat().st_size, 'sha256': digest}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True); ap.add_argument('--destination', required=True)
    ap.add_argument('--config', required=True); ap.add_argument('--cache', required=True)
    a = ap.parse_args(); source, destination = Path(a.source).resolve(), Path(a.destination).resolve()
    if source == destination: raise ValueError('Recovery needs a distinct root')
    old = load(source/'experiment.json'); campaign = Campaign(a.config, a.cache, destination)
    old_cfg, new_cfg = copy.deepcopy(old['config']), copy.deepcopy(campaign.cfg)
    old_cfg.pop('budget'); new_cfg.pop('budget')
    if old_cfg != new_cfg: raise ValueError('Scientific experiment changed; migration refused')
    if campaign.cfg['budget'].get('campaign_started_utc') != old['created_utc']:
        raise ValueError('Original wall-budget start must be retained')
    if old['cache_identity'] != campaign.cache_identity: raise ValueError('Delivered cache changed')
    old_source, new_source = copy.deepcopy(old['source']), copy.deepcopy(campaign.source)
    old_driver = old_source['application'].pop('ks_improvement_runner.py')
    new_driver = new_source['application'].pop('ks_improvement_runner.py')
    if old_source != new_source: raise ValueError('Native numerical source/runtime changed')
    proof = {'schema': 'ks-improvement-operational-migration-1', 'created_utc': now(),
        'source_experiment': old['identity'], 'destination_experiment': campaign.identity,
        'old_driver_sha256': old_driver, 'new_driver_sha256': new_driver,
        'native_data_candidates_predictions_scores_unchanged': True,
        'wall_budget_start': old['created_utc'], 'files': {}, 'preflight_adoptions': []}
    # All completed fits and origin receipts survive exactly. No posterior is refitted here.
    for folder in ('fits', 'forecasts', 'preflight', 'attempts'):
        for path in sorted((source/folder).rglob('*')):
            if path.is_file():
                relative = path.relative_to(source)
                proof['files'][str(relative)] = verified_copy(path, destination/relative)
    for name in ('experiment.json', 'history_policy.json', 'imported_checkpoints.json',
            'cache_audit.json', 'initial_resources.json', 'resource_profile.json'):
        path = source/name
        if path.exists(): verified_copy(path, destination/'migration/source_records'/name)
    # Retain original status evidence rather than rewriting it.
    for path in (source/'ledger').glob('*.json'):
        verified_copy(path, destination/'migration/source_ledger'/path.name)
    jobs = [load(p) for p in (source/'jobs').glob('*.json')]
    jobs = [j for j in jobs if j['stage'] == 'preflight']
    by_fit = {j['fit_key']: j for j in jobs}
    expected_count = sum(len(candidate_records(campaign.registry, t)) for t in campaign.cfg['tracks'])*len(campaign.cfg['screen']['trajectories'])
    if len(jobs) != expected_count or len(by_fit) != expected_count:
        raise ValueError('Incomplete or duplicate source preflight roster')
    for track in campaign.cfg['tracks']:
        for candidate in candidate_records(campaign.registry, track):
            for trajectory in campaign.cfg['screen']['trajectories']:
                job = campaign.job('preflight', track, candidate, trajectory)
                original = by_fit[job['fit_key']]
                if original['job_id'] != content_hash({k: v for k, v in original.items() if k != 'job_id'})[:24]:
                    raise ValueError('Source job identity mismatch')
                if any(original[k] != job[k] for k in ('track', 'trajectory', 'seed', 'candidate', 'fit_key', 'lengths')):
                    raise ValueError('Preflight configuration or schedule changed')
                if original['source'] != old['source'] or original['experiment_identity'] != old['identity']:
                    raise ValueError('Preflight source binding mismatch')
                artifact = source/'preflight'/f"{job['fit_key']}.json"; result = load(artifact)
                attempts = list((source/'attempts').glob(f"{original['job_id']}_*.json"))
                if len(attempts) != 1: raise ValueError('Ambiguous original preflight attempts')
                attempt = load(attempts[0])
                if attempt['state'] != 'complete' or attempt['returncode'] != 0:
                    raise ValueError('Original process did not complete successfully')
                if not any(r.get('passed') for r in result['synchronization']['rows']):
                    raise ValueError('No usable history in recovered preflight')
                if result['geometry']['fit_identity'] != result['synchronization']['fit_identity']:
                    raise ValueError('Feature preparation identity mismatch')
                original_status = load(source/'ledger'/f"{original['job_id']}.json")
                if original_status['state'] == 'complete' and original_status['result'] != result:
                    raise ValueError('Original terminal result differs from immutable artifact')
                provenance = {'source_job_id': original['job_id'],
                    'source_job_sha256': file_hash(source/'jobs'/f"{original['job_id']}.json"),
                    'artifact_sha256': file_hash(artifact), 'attempt_sha256': file_hash(attempts[0]),
                    'source_status': original_status['state'],
                    'elapsed_seconds_scope': 'original_parent_process_attempt_wall_seconds',
                    'scientific_recomputation': False}
                terminal = {'job_id': job['job_id'], 'state': 'complete',
                    'job_identity': content_hash({k: v for k, v in job.items() if k != 'job_id'}),
                    'finished_utc': now(), 'result': result,
                    'elapsed_seconds': attempt['elapsed_seconds'], 'adoption': provenance}
                if original_status.get('state') == 'complete' and 'resource' in original_status:
                    terminal['resource'] = original_status['resource']
                atomic_json(destination/'jobs'/f"{job['job_id']}.json", job)
                atomic_json(destination/'terminal'/f"{job['job_id']}.json", terminal)
                campaign.ledger.write(job['job_id'], 'complete', job=job, result=result, adoption=provenance)
                proof['preflight_adoptions'].append({'destination_job_id': job['job_id'], **provenance})
    if not campaign.preflight(): raise ValueError('Recovered preflight did not complete')
    policy = load(destination/'history_policy.json')
    if policy['history'] != load(source/'history_policy.json')['history'] or policy['excluded_candidates']:
        raise ValueError('Recovered history changed or any verified candidate remains excluded')
    proof.update(recovered_preflight_jobs=expected_count, history=policy['history'],
        fitted_checkpoint_count=len(list((destination/'fits').glob('*/fit.npz'))),
        forecast_receipt_count=len(list((destination/'forecasts').glob('*/*.json'))),
        carried_worker_seconds=campaign.consumed(), passed=True)
    atomic_json(destination/'migration/receipt.json', proof)
    print(json.dumps({k: v for k, v in proof.items() if k not in ('files', 'preflight_adoptions')}, indent=2))


if __name__ == '__main__': main()
