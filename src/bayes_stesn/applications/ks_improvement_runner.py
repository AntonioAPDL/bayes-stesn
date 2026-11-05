"""Native-only, resumable validation expansion of the frozen KS application.

The original model, data reader, sampler and scores are reused unchanged.
This driver never opens the test partition. Prior sensitivity is a separate
stratum, and confirmation trajectories are excluded from candidate selection.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from ..io import content_hash, seed_for
from .ks_data import KSAdapter, audit_cache
from .ks_metrics import cluster_summary, score
from .ks_native import (NativeFit, _computation_identity, _prepare_delivered_scale,
    _resolve_bayes, fit_native, reservoir_synchronization)
from .ks_runtime import Ledger, admission, atomic_json, file_hash, now, resource_snapshot

PROTOCOLS = {'A': 'interleaved', 'B': 'all_observed', 'C': 'contiguous'}
MODULE = 'bayes_stesn.applications.ks_improvement_runner'
SELECTION_TRAJECTORIES = [0, 5, 10, 15, 4, 9]
CONFIRMATION_TRAJECTORIES = [i for i in range(20) if i not in SELECTION_TRAJECTORIES]
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def source_identity():
    """Bind this revision without requiring an unused neural ecosystem."""
    root = Path(__file__).resolve().parent
    names = ['ks_improvement_runner.py', 'ks_improvement_candidates.py',
        'ks_improvement_diagnostics.py', 'ks_data.py', 'ks_metrics.py', 'ks_runtime.py']
    return {'native': _computation_identity('bayes_stesn'),
        'application': {n: file_hash(root/n) for n in names},
        'packages': {n: importlib.metadata.version(n) for n in
            ('numpy', 'scipy', 'h5py', 'threadpoolctl')}}


def read_json(path):
    return json.loads(Path(path).read_text())


def candidate_records(registry, track):
    return registry['tracks'][track]['candidates']


def fit_key(track, candidate, trajectory, seed):
    return content_hash({'track': track, 'effective_config': candidate['config'],
        'trajectory': int(trajectory), 'seed': int(seed)})[:32]


def scientific_error(error):
    return isinstance(error, (FloatingPointError, np.linalg.LinAlgError)) or (
        isinstance(error, ValueError) and any(s in str(error) for s in
            ('finite', 'zero variation', 'Forecast mean', 'Samples must')))


def terminal_result(root, job):
    """Immutable terminal evidence takes precedence over a mutable status view."""
    path = Path(root)/'terminal'/f"{job['job_id']}.json"
    if not path.exists(): return None
    r = read_json(path)
    identity = content_hash({k: v for k, v in job.items() if k != 'job_id'})
    if r['job_identity'] != identity or r['state'] not in ('complete', 'failed', 'inference-unqualified'):
        raise ValueError('Incompatible terminal job receipt')
    return r


def training_preflight(training, candidate, seed, lengths):
    resolved, spec, _ = _resolve_bayes(candidate['config'], seed)
    p = _prepare_delivered_scale(training.panel, resolved, spec)
    # Only the already-fitted feature map is needed. No posterior is invented.
    shell = NativeFit('bayes_stesn', SimpleNamespace(prepared=p), resolved, seed,
        p.identity, p.identity, {}, {'qualified': False})
    sync = reservoir_synchronization(shell, training.panel, lengths=tuple(lengths))
    from .ks_improvement_diagnostics import geometry_diagnostics
    # Geometry of all static circle sites is permitted, including withheld sites.
    from .ks_data import circle_embedding
    coords = circle_embedding(np.arange(64)*22./64.)
    geometry = geometry_diagnostics(shell, coords)
    return {'synchronization': sync, 'geometry': geometry,
        'dimensions': {'spatial': int(p.Phi.shape[1]), 'temporal': int(p.Z.shape[1]),
            'readout': int(p.Phi.shape[1]*p.Z.shape[1]),
            'likelihood_rows': int(len(training.panel.y)-resolved['conditioning'])},
        'training_identity': dict(training.identity)}


def worker(job_path):
    """One process and one writer per fit; completed origins are immutable."""
    threadpool_limits(1)
    job = read_json(job_path); root = Path(job['output'])
    if job['source'] != source_identity():
        raise ValueError('Changed numerical source/runtime; incompatible job refused')
    if job.get('partition', 'validation') != 'validation':
        raise ValueError('This improvement campaign is validation-only')
    ledger = Ledger(root/'ledger'); started = time.perf_counter()
    def finish(state, **values):
        result = {'job_id': job['job_id'], 'state': state,
            'job_identity': content_hash({k: v for k, v in job.items() if k != 'job_id'}),
            'finished_utc': now(), **values}
        atomic_json(root/'terminal'/f"{job['job_id']}.json", result)
        ledger.write(job['job_id'], state, **values)
    folder = root/'fits'/job['fit_key']; folder.mkdir(parents=True, exist_ok=True)
    with (folder/'writer.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger.write(job['job_id'], 'running', job=job, pid=os.getpid(), started_utc=now())
        try:
            adapter = KSAdapter(job['cache'], PROTOCOLS[job['track']])
            training = adapter.training(job['trajectory'])
            if job['stage'] == 'preflight':
                result = training_preflight(training, job['candidate'], job['seed'], job['lengths'])
                atomic_json(root/'preflight'/f"{job['fit_key']}.json", result)
                finish('complete', result=result,
                    elapsed_seconds=time.perf_counter()-started, resource=resource_snapshot(root))
                return
            path = folder/'fit.npz'; receipt_path = folder/'receipt.json'
            previous = read_json(receipt_path) if receipt_path.exists() else None
            if previous and (not path.exists() or file_hash(path) != previous['fit_sha256']):
                raise ValueError('Fitted checkpoint failed its retained checksum')
            fit_start = time.perf_counter()
            fitted = fit_native('bayes_stesn', training.panel, job['candidate']['config'],
                job['seed'], checkpoint_path=path)
            if previous is None:
                previous = {'fit_key': job['fit_key'], 'native_fit_identity': fitted.identity,
                    'fit_sha256': file_hash(path), 'fit_bytes': path.stat().st_size,
                    'fit_seconds': float(fitted.diagnostics['fit_wall_seconds']),
                    'current_fit_or_load_seconds': time.perf_counter()-fit_start, 'diagnostics': fitted.diagnostics,
                    'training_identity': dict(training.identity), 'config': fitted.config,
                    'track': job['track'], 'trajectory': job['trajectory'], 'seed': job['seed'],
                    'candidate_id': job['candidate']['id'], 'completed_utc': now(),
                    'resource': resource_snapshot(root)}
                atomic_json(receipt_path, previous)
            qualified = bool(fitted.diagnostics['qualified'])
            if not qualified:
                finish('inference-unqualified', fit_key=job['fit_key'],
                    elapsed_seconds=time.perf_counter()-started, resource=resource_snapshot(root))
                return
            sync_path = folder/f"synchronization_h{job['history']}.json"
            if not sync_path.exists():
                sync = reservoir_synchronization(fitted, training.panel, lengths=(job['history'],))
                atomic_json(sync_path, sync)
            else:
                sync = read_json(sync_path)
            if not sync['rows'][0].get('passed', False):
                finish('failed', reason='prespecified_history_did_not_synchronize',
                    elapsed_seconds=time.perf_counter()-started, resource=resource_snapshot(root))
                return
            failed = 0; completed = 0
            for first in job['origins']:
                key = content_hash({'native_fit': fitted.identity, 'first': first,
                    'history': job['history'], 'horizon': 100, 'paths': job['paths'],
                    'forecast_seed': job['forecast_seed'], 'partition': 'validation',
                    'score_source': job['source']['application']['ks_metrics.py']})
                forecast_root = 'precision_forecasts' if job.get('forecast_namespace') == 'precision' else 'forecasts'
                target = root/forecast_root/job['fit_key']/f'{first:05d}_{job["paths"]}_{key[:12]}.json'
                if target.exists():
                    old = read_json(target)
                    if old['identity'] != key: raise ValueError('Forecast identity mismatch')
                    completed += old['state'] == 'complete'; failed += old['state'] != 'complete'
                    continue
                origin_started = time.perf_counter()
                base = {'identity': key, 'fit_key': job['fit_key'], 'native_fit': fitted.identity,
                    'track': job['track'], 'candidate_id': job['candidate']['id'],
                    'trajectory': job['trajectory'], 'seed': job['seed'], 'first_target': first,
                    'history': job['history'], 'paths': job['paths'], 'qualified': True,
                    'forecast_seed': job['forecast_seed'], 'forecast_namespace': job.get('forecast_namespace', 'main'),
                    'partition': 'validation', 'stage_first_scored': job['stage']}
                try:
                    window = adapter.model_window(job['trajectory'], first, 'validation',
                        history=job['history'], horizon=100)
                    stream = seed_for(job['forecast_seed'], job['track'], job['trajectory'],
                        job['seed'], first)
                    samples = np.empty((job['paths'], 100, len(window.target_site_ids)))
                    for batch in fitted.iter_forecasts(window.panel, 100, window.target_coordinates,
                            window.target_site_ids, job['paths'], stream, batch_size=job['path_batch']):
                        a, b = batch.metadata['path_start'], batch.metadata['path_stop']
                        samples[a:b] = training.transform.inverse(batch.samples)
                    mean = samples.mean(axis=0, dtype=np.float64)
                    truth = adapter.evaluation_window(job['trajectory'], first, 'validation',
                        history=job['history'], horizon=100).latent_clean
                    metrics = score(mean, samples, truth, training.transform.scale,
                        window.target_coordinates)
                    if job.get('retain') and first == job['origins'][0]:
                        from ..io import atomic_npz
                        atomic_npz(root/'examples'/f"{job['track']}_{job['candidate']['id']}_t{job['trajectory']:02d}_s{job['seed']}_a{first}.npz",
                            mean=mean, truth=truth, lower=np.quantile(samples, .05, axis=0),
                            upper=np.quantile(samples, .95, axis=0),
                            coordinates=window.target_coordinates,
                            input_coordinates=window.panel.coordinates,
                            first_target=np.array(first), training_scale=np.array(training.transform.scale))
                    atomic_json(target, {**base, 'state': 'complete', 'metrics': metrics,
                        'seconds': time.perf_counter()-origin_started})
                    completed += 1
                except Exception as error:
                    if not scientific_error(error): raise
                    atomic_json(target, {**base, 'state': 'failed', 'error': repr(error),
                        'seconds': time.perf_counter()-origin_started})
                    failed += 1
            finish('complete' if failed == 0 else 'failed',
                completed_origins=completed, failed_origins=failed,
                elapsed_seconds=time.perf_counter()-started, resource=resource_snapshot(root))
        except Exception as error:
            state = 'failed' if scientific_error(error) else 'pending'
            values = dict(error=repr(error), traceback=traceback.format_exc(),
                elapsed_seconds=time.perf_counter()-started, resource=resource_snapshot(root))
            if state == 'failed': finish(state, **values)
            else:
                ledger.write(job['job_id'], state, **values)
                raise


class Campaign:
    def __init__(self, config, cache, output):
        threadpool_limits(1)
        self.cfg = read_json(config); self.registry = self.cfg['registry']
        self.output = Path(output).resolve(); self.output.mkdir(parents=True, exist_ok=True)
        self.cache = str(Path(cache).resolve()); self.source = source_identity()
        verified_cache = audit_cache(self.cache)
        if not verified_cache['passed']: raise ValueError('Delivered cache integrity check failed')
        self.cache_identity = content_hash(verified_cache)
        self.identity = content_hash({'config': self.cfg, 'source': self.source, 'cache': self.cache_identity})
        path = self.output/'experiment.json'
        if path.exists() and read_json(path)['identity'] != self.identity:
            raise ValueError('Changed experiment/source; use a new versioned output root')
        if not path.exists():
            atomic_json(path, {'identity': self.identity, 'config': self.cfg, 'source': self.source,
                'cache_identity': self.cache_identity,
                'created_utc': now(), 'scope': 'exploratory_validation_expansion_after_original_test_inspection'})
        self.ledger = Ledger(self.output/'ledger')

    def job(self, stage, track, candidate, trajectory, origins=(), paths=256, seed=None, retain=False):
        seed = self.cfg['model_seed'] if seed is None else int(seed)
        history_path = self.output/'history_policy.json'
        j = {'output': str(self.output), 'cache': self.cache, 'source': self.source,
            'experiment_identity': self.identity, 'stage': stage, 'track': track,
            'candidate': candidate, 'trajectory': trajectory, 'seed': seed,
            'fit_key': fit_key(track, candidate, trajectory, seed), 'origins': list(origins),
            'paths': paths, 'path_batch': self.cfg['path_batch'], 'forecast_seed': self.cfg['forecast_seed'],
            'history': read_json(history_path)['history'] if stage != 'preflight' and history_path.exists() else None,
            'lengths': self.cfg['history_candidates'], 'retain': retain, 'partition': 'validation'}
        j['job_id'] = content_hash(j)[:24]
        return j

    def consumed(self, stages=None):
        records = [read_json(p) for p in (self.output/'attempts').glob('*.json')]
        preparation = float(self.cfg['budget'].get('preparation_worker_seconds', 0.)) if stages is None or 'preflight' in stages else 0.
        return preparation+sum(float(r.get('elapsed_seconds', 0.)) for r in records
            if stages is None or r['stage'] in stages)

    def budget_reason(self, stage):
        budget = self.cfg['budget']
        created_utc = budget.get('campaign_started_utc') or read_json(self.output/'experiment.json')['created_utc']
        created = datetime.datetime.fromisoformat(created_utc).timestamp()
        if time.time()-created >= budget.get('wall_seconds_cap', 4*3600):
            return 'finite_campaign_wall_budget'
        if self.consumed() >= budget['total_worker_seconds_cap']:
            return 'finite_campaign_budget'
        if stage in ('preflight', 'screen', 'promote', 'precision') and self.consumed({'preflight', 'screen', 'promote', 'precision'}) >= budget['search_worker_seconds_cap']:
            return 'search_budget_final_confirmation_reserved'
        return None

    def submit(self, job):
        old = terminal_result(self.output, job)
        if old is not None: return old
        budget = self.cfg['budget']
        reason = self.budget_reason(job['stage'])
        if reason:
            self.ledger.write(job['job_id'], 'pending', job=job, reason=reason)
            return {'state': 'pending'}
        path = self.output/'jobs'/f"{job['job_id']}.json"; atomic_json(path, job)
        log = self.output/'logs'/f"{job['job_id']}.log"; log.parent.mkdir(exist_ok=True)
        env = dict(os.environ); env.update({n: '1' for n in THREAD_ENV})
        env.update(PYTHONHASHSEED='0', MPLBACKEND='Agg')
        begin = time.monotonic(); attempt_path = self.output/'attempts'/f"{job['job_id']}_{time.time_ns()}.json"
        self.ledger.write(job['job_id'], 'pending', job=job)
        print(json.dumps({'event': 'start', 'stage': job['stage'], 'track': job['track'],
            'candidate': job['candidate']['id'], 'trajectory': job['trajectory']}), flush=True)
        with log.open('a') as stream:
            proc = subprocess.Popen([sys.executable, '-m', MODULE, 'worker', '--job', str(path)],
                env=env, stdout=stream, stderr=subprocess.STDOUT)
            attempt = {'job_id': job['job_id'], 'stage': job['stage'], 'state': 'running',
                'started_utc': now(), 'started_epoch': time.time(), 'pid': proc.pid}
            atomic_json(attempt_path, attempt)
            last = -10.; stopped = None
            while proc.poll() is None:
                elapsed = time.monotonic()-begin
                if elapsed-last >= 5:
                    snap = resource_snapshot(self.output); mem = snap['memory']
                    if elapsed > budget['job_timeout_seconds']: stopped = 'job_timeout'
                    elif snap['disk_free_bytes'] < budget['disk_reserve_bytes']: stopped = 'disk_reserve'
                    elif mem['memory.max'] not in (None, 'max') and int(mem['memory.max'])-int(mem['memory.current']) < 384*1024**2:
                        stopped = 'container_headroom'
                    atomic_json(attempt_path, {**attempt, 'elapsed_seconds': elapsed, 'resource': snap})
                    last = elapsed
                    if stopped:
                        import signal
                        proc.send_signal(signal.SIGINT)
                        try: proc.wait(timeout=20)
                        except subprocess.TimeoutExpired: proc.terminate(); proc.wait(timeout=10)
                        break
                time.sleep(.25)
        elapsed = time.monotonic()-begin
        atomic_json(attempt_path, {**attempt, 'state': 'complete' if proc.returncode == 0 else 'interrupted',
            'returncode': proc.returncode, 'reason': stopped, 'elapsed_seconds': elapsed, 'finished_utc': now()})
        result = terminal_result(self.output, job) or {'state': 'pending'}
        print(json.dumps({'event': 'end', 'stage': job['stage'], 'track': job['track'],
            'candidate': job['candidate']['id'], 'trajectory': job['trajectory'],
            'state': result['state'], 'seconds': elapsed}), flush=True)
        if proc.returncode != 0:
            self.ledger.write(job['job_id'], 'pending', reason=stopped or 'worker_interruption',
                elapsed_seconds=elapsed)
            raise RuntimeError(f'Worker interrupted; inspect {log}')
        if result['state'] == 'pending':
            raise RuntimeError(f'Worker exited without an immutable terminal receipt; inspect {log}')
        return result

    def batch(self, jobs, workers=None):
        workers = int(workers or self.cfg['budget']['workers'])
        remaining = list(jobs)
        while remaining:
            group = []; seen = set(); deferred = []
            for j in remaining:
                if len(group) < workers and j['fit_key'] not in seen:
                    group.append(j); seen.add(j['fit_key'])
                else: deferred.append(j)
            remaining = deferred
            pending = [j for j in group if terminal_result(self.output, j) is None]
            if not pending: continue
            reason = self.budget_reason(pending[0]['stage'])
            if reason:
                for j in pending: self.ledger.write(j['job_id'], 'pending', job=j, reason=reason)
                return False
            admission(self.output, reserve_bytes=self.cfg['budget']['disk_reserve_bytes'],
                estimated_peak_bytes=self.cfg['budget']['estimated_worker_peak_bytes']*len(pending),
                worker_cap_bytes=self.cfg['budget']['combined_worker_cap_bytes'])
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(self.submit, pending))
            if any(r['state'] == 'pending' for r in results): return False
        return True

    def preflight(self):
        if (self.output/'history_policy.json').exists(): return True
        jobs = [self.job('preflight', t, c, i) for t in self.cfg['tracks']
            for c in candidate_records(self.registry, t) for i in self.cfg['screen']['trajectories']]
        if not self.batch(jobs): return False
        eligible = []; failed = []; passing = set(self.cfg['history_candidates']); excluded = {}; outcomes = []
        for j in jobs:
            r = terminal_result(self.output, j)
            if r is None: raise RuntimeError('Incomplete preflight execution; do not classify candidate as a scientific failure')
            if r['state'] != 'complete':
                failed.append(j['job_id']); excluded[j['track']+'/'+j['candidate']['id']] = 'training_preflight_failed'
                continue
            good = {x['history'] for x in r['result']['synchronization']['rows'] if x.get('passed')}
            if not good:
                failed.append(j['job_id']); excluded[j['track']+'/'+j['candidate']['id']] = 'no_admissible_measured_history'
                continue
            outcomes.append((j, good))
        for j, good in outcomes:
            if j['track']+'/'+j['candidate']['id'] in excluded: continue
            passing &= good; eligible.append(j['job_id'])
        if not passing or not eligible:
            raise RuntimeError('No common measured history; preserve preflight and revise new campaign before screening')
        atomic_json(self.output/'history_policy.json', {'history': min(passing),
            'candidates': self.cfg['history_candidates'], 'checked_jobs': len(jobs),
            'passed_jobs': len(eligible), 'failed_jobs': failed, 'excluded_candidates': excluded, 'created_utc': now(),
            'basis': 'permitted_training_history_only; common_for_all_admissible_tracks_and_candidates'})
        return True

    def loss(self, track, candidate, trajectories, origins, paths=256, seed=None):
        seed = self.cfg['model_seed'] if seed is None else seed
        episodes = []; failures = 0; pending = 0; rows = []; unqualified = 0
        history_policy = read_json(self.output/'history_policy.json')
        excluded = history_policy.get('excluded_candidates', {}).get(track+'/'+candidate['id'])
        for i in trajectories:
            key = fit_key(track, candidate, i, seed); values = []; rmse = []
            receipt = self.output/'fits'/key/'receipt.json'
            fit_receipt = read_json(receipt) if receipt.exists() else None
            qualified = fit_receipt is not None and fit_receipt['diagnostics']['qualified']
            unqualified += int(fit_receipt is not None and not qualified)
            for a in origins:
                if excluded or (fit_receipt is not None and not qualified): continue
                files = list((self.output/'forecasts'/key).glob(f'{a:05d}_{paths}_*.json'))
                if len(files) > 1: raise ValueError('Ambiguous origin receipts')
                if not files: pending += 1; continue
                r = read_json(files[0]); rows.append(r)
                if fit_receipt is None: raise ValueError('Forecast without a fitted-model receipt')
                expected = content_hash({'native_fit': fit_receipt['native_fit_identity'], 'first': a,
                    'history': history_policy['history'], 'horizon': 100, 'paths': paths,
                    'forecast_seed': self.cfg['forecast_seed'], 'partition': 'validation',
                    'score_source': self.source['application']['ks_metrics.py']})
                if (r['identity'] != expected or r['native_fit'] != fit_receipt['native_fit_identity']
                        or r['history'] != history_policy['history'] or r['seed'] != seed
                        or r['track'] != track or r['trajectory'] != i or r['first_target'] != a
                        or r['partition'] != 'validation' or not r['qualified']):
                    raise ValueError('Stale or incompatible forecast receipt')
                if r['state'] != 'complete' or not qualified: failures += 1; continue
                values.append(r['metrics']['selection_crps']); rmse.append(r['metrics']['selection_mse'])
            if len(values) == len(origins):
                episodes.append({'trajectory': i, 'selection_crps': float(np.mean(values)),
                    'selection_mse': float(np.mean(rmse))})
        eligible = len(episodes) == len(trajectories) and failures == 0
        return {'id': candidate['id'], 'stratum': candidate['stratum'], 'eligible': eligible,
            'loss': float(np.mean([e['selection_crps'] for e in episodes])) if eligible else None,
            'point_loss': float(np.mean([e['selection_mse'] for e in episodes])) if eligible else None,
            'qualified_episodes': len(episodes), 'expected_episodes': len(trajectories),
            'failed_origins': failures, 'pending_origins': pending, 'episodes': episodes,
            'inference_unqualified_fits': unqualified, 'preflight_exclusion': excluded,
            'configuration': candidate['config'], 'effective_hash': candidate['effective_hash'],
            'complexity': candidate.get('cost', candidate.get('cost_estimate', {}))}

    def rankings(self, stage, track, candidates, schedule, paths=256):
        rows = [self.loss(track, c, schedule['trajectories'], schedule['origins'], paths) for c in candidates]
        rows.sort(key=lambda r: (not r['eligible'], float('inf') if r['loss'] is None else r['loss'], r['id']))
        atomic_json(self.output/'selection'/f'{stage}_{track}.json', {'stage': stage, 'track': track,
            'schedule': schedule, 'paths': paths, 'rows': rows, 'created_utc': now(),
            'objective': 'episode_balanced_equal_horizon_band_CRPS',
            'prior_strata': 'main_fixed_prior selected separately; legacy and sensitivity retained explicitly'})
        return rows

    def screen(self):
        if not self.preflight(): return False
        schedule = self.cfg['screen']
        excluded = read_json(self.output/'history_policy.json').get('excluded_candidates', {})
        paths = schedule.get('paths', 256)
        jobs = [self.job('screen', t, c, i, schedule['origins'], paths=paths) for t in self.cfg['tracks']
            for c in candidate_records(self.registry, t) if t+'/'+c['id'] not in excluded
            for i in schedule['trajectories']]
        if not self.batch(jobs): return False
        for t in self.cfg['tracks']:
            self.rankings('screen', t, candidate_records(self.registry, t), schedule, paths)
        return True

    def promote(self):
        if not all((self.output/'selection'/f'screen_{t}.json').exists() for t in self.cfg['tracks']):
            if not self.screen(): return False
        schedule = self.cfg['promote']; selected = {}
        excluded = read_json(self.output/'history_policy.json').get('excluded_candidates', {})
        for t in self.cfg['tracks']:
            rows = read_json(self.output/'selection'/f'screen_{t}.json')['rows']
            main = [r for r in rows if r['eligible'] and r['stratum'] == 'main_fixed_prior'][:schedule['count']]
            sensitivity = [r for r in rows if r['eligible'] and r['stratum'] == 'prior_scale_sensitivity'][:1]
            ids = {r['id'] for r in main+sensitivity}
            # Always compare to the old selected configuration on exactly the same windows.
            anchor = self.registry['tracks'][t]['anchor_id']
            if t+'/'+anchor not in excluded: ids.add(anchor)
            selected[t] = [c for c in candidate_records(self.registry, t) if c['id'] in ids]
        paths = schedule.get('paths', 512)
        jobs = [self.job('promote', t, c, i, schedule['origins'], paths=paths) for t in self.cfg['tracks']
            for c in selected[t] for i in schedule['trajectories']]
        if not self.batch(jobs): return False
        for t in self.cfg['tracks']: self.rankings('promote', t, selected[t], schedule, paths)
        return True

    def freeze(self):
        if not all((self.output/'selection'/f'promote_{t}.json').exists() for t in self.cfg['tracks']):
            if not self.promote(): return None
        if self.cfg.get('precision', {}).get('enabled'):
            if not self.precision(): return None
        selections = {}
        for t in self.cfg['tracks']:
            rows = read_json(self.output/'selection'/f'promote_{t}.json')['rows']
            eligible = [r for r in rows if r['eligible'] and r['stratum'] == 'main_fixed_prior']
            if not eligible: raise RuntimeError(f'No qualified main Bayesian candidate for track {t}')
            best = min(r['loss'] for r in eligible)
            precision_path = self.output/'selection'/f'precision_{t}.json'
            tolerances = {r['id']: .01*best for r in eligible}
            if precision_path.exists():
                measured = read_json(precision_path)['candidate_absolute_mc_tolerances']
                tolerances = {r['id']: max(.01*best, measured[r['id']]) for r in eligible}
            near = [r for r in eligible if r['loss'] <= best+tolerances[r['id']]]
            def size(row):
                c = row['configuration']; f = c['features']; s = c['spatial_mean']
                p = len(s['centers'])+3 if s.get('centers') is not None else s['side']**2+3
                return p*(1+len(f['lags'])*f['modes']+sum(f['units'])+f['squares']+f['products'])
            chosen = min(near, key=lambda r: (size(r), r['loss'], r['id']))
            sensitivity = [r for r in rows if r['eligible'] and r['stratum'] == 'prior_scale_sensitivity']
            selections[t] = {'main': chosen, 'near_tie_ids': [r['id'] for r in near],
                'absolute_near_tie_tolerance': tolerances[chosen['id']],
                'candidate_absolute_near_tie_tolerances': tolerances,
                'precision_receipt_sha256': file_hash(precision_path) if precision_path.exists() else None,
                'sensitivity': min(sensitivity, key=lambda r: r['loss']) if sensitivity else None,
                'anchor_id': self.registry['tracks'][t]['anchor_id']}
        freeze = {'experiment_identity': self.identity, 'selected': selections,
            'history': read_json(self.output/'history_policy.json')['history'],
            'confirmation_trajectories': CONFIRMATION_TRAJECTORIES,
            'confirmation_origins': list(range(6000, 8000, 100)), 'paths': 512,
            'created_utc': now(), 'test_access': False,
            'interpretation': 'newly withheld validation trajectories within an exploratory post-test study; not a fresh untouched test set'}
        freeze['identity'] = content_hash(freeze)
        path = self.output/'freeze.json'
        if path.exists():
            previous = read_json(path)
            if (previous.get('identity') != content_hash({k: v for k, v in previous.items() if k != 'identity'})
                    or previous['experiment_identity'] != self.identity or previous['selected'] != selections
                    or previous['history'] != freeze['history']):
                raise ValueError('Frozen selection cannot be replaced')
            return previous
        atomic_json(path, freeze)
        return freeze

    def precision(self):
        policy = self.cfg.get('precision', {})
        if not policy.get('enabled'): return True
        if all((self.output/'selection'/f'precision_{t}.json').exists() for t in self.cfg['tracks']): return True
        if not all((self.output/'selection'/f'promote_{t}.json').exists() for t in self.cfg['tracks']):
            if not self.promote(): return False
        chosen = {}; jobs = []
        for t in self.cfg['tracks']:
            rows = read_json(self.output/'selection'/f'promote_{t}.json')['rows']
            ids = {r['id'] for r in rows if r['eligible'] and r['stratum'] == 'main_fixed_prior'}
            chosen[t] = [c for c in candidate_records(self.registry, t) if c['id'] in ids]
            for c in chosen[t]:
                for repetition in range(policy['replicates']):
                    j = self.job('precision', t, c, policy['trajectory'], policy['origins'], paths=policy['paths'])
                    j['forecast_namespace'] = 'precision'
                    j['forecast_seed'] = seed_for(self.cfg['forecast_seed'], 'precision', repetition)
                    j['job_id'] = content_hash({k: v for k, v in j.items() if k != 'job_id'})[:24]
                    jobs.append(j)
        if not self.batch(jobs): return False
        for t in self.cfg['tracks']:
            rows = []; replicated = {}
            history = read_json(self.output/'history_policy.json')['history']
            for c in chosen[t]:
                key = fit_key(t, c, policy['trajectory'], self.cfg['model_seed'])
                receipts = [read_json(p) for p in (self.output/'precision_forecasts'/key).glob('*.json')]
                fitted_receipt = read_json(self.output/'fits'/key/'receipt.json')
                values = []
                for repetition in range(policy['replicates']):
                    stream = seed_for(self.cfg['forecast_seed'], 'precision', repetition)
                    selected = [r for r in receipts if r['forecast_seed'] == stream and
                        r['first_target'] in policy['origins'] and r['paths'] == policy['paths']]
                    if (sorted(r['first_target'] for r in selected) != sorted(policy['origins'])
                            or any(r['state'] != 'complete' for r in selected)):
                        raise RuntimeError('Predictive precision assessment incomplete or failed; no selection freeze')
                    for r in selected:
                        expected = content_hash({'native_fit': fitted_receipt['native_fit_identity'],
                            'first': r['first_target'], 'history': history, 'horizon': 100,
                            'paths': policy['paths'], 'forecast_seed': stream, 'partition': 'validation',
                            'score_source': self.source['application']['ks_metrics.py']})
                        if (r['identity'] != expected or r['native_fit'] != fitted_receipt['native_fit_identity']
                                or r['history'] != history or r['track'] != t
                                or r['trajectory'] != policy['trajectory'] or not r['qualified']
                                or r['partition'] != 'validation' or r['forecast_namespace'] != 'precision'
                                or not fitted_receipt['diagnostics']['qualified']):
                            raise ValueError('Stale or incompatible precision receipt')
                    values.append(float(np.mean([r['metrics']['selection_crps'] for r in selected])))
                replicated[c['id']] = values
                rows.append({'id': c['id'], 'replicate_case_mean_crps': values,
                    'conditional_mc_sd_at_declared_path_count': float(np.std(values, ddof=1))})
            if not rows: raise RuntimeError('No main candidates for precision checks')
            differences = []
            for a in range(len(rows)):
                for b in range(a):
                    delta = np.asarray(replicated[rows[a]['id']])-np.asarray(replicated[rows[b]['id']])
                    differences.append({'a': rows[a]['id'], 'b': rows[b]['id'],
                        'paired_difference_mc_sd': float(np.std(delta, ddof=1))})
            promoted = read_json(self.output/'selection'/f'promote_{t}.json')['rows']
            best_id = min((r for r in promoted if r['eligible'] and r['stratum'] == 'main_fixed_prior'),
                key=lambda r: (r['loss'], r['id']))['id']
            tolerances = {cid: 2*float(np.std(np.asarray(values)-np.asarray(replicated[best_id]), ddof=1))
                for cid, values in replicated.items()}
            atomic_json(self.output/'selection'/f'precision_{t}.json', {'track': t, 'policy': policy,
                'rows': rows, 'paired_differences': differences,
                'promotion_loss_best_id': best_id,
                'candidate_absolute_mc_tolerances': tolerances, 'created_utc': now(),
                'interpretation': 'conditional predictive-simulation variability on fixed validation cases, not posterior-chain MC error or scientific sampling uncertainty',
                'selection_rule': 'candidate-specific near ties use maximum of1percent best promotion loss and twice paired replicate SD against that loss winner on designated cases; lower coefficient count wins'})
        return True

    def confirm(self):
        freeze = self.freeze(); jobs = []
        if freeze is None: return False
        for t in self.cfg['tracks']:
            ids = {freeze['selected'][t]['main']['id'], freeze['selected'][t]['anchor_id']}
            if freeze['selected'][t].get('sensitivity'):
                ids.add(freeze['selected'][t]['sensitivity']['id'])
            for c in candidate_records(self.registry, t):
                if c['id'] in ids:
                    for i in CONFIRMATION_TRAJECTORIES:
                        jobs.append(self.job('confirm', t, c, i, freeze['confirmation_origins'],
                            paths=512, retain=i == CONFIRMATION_TRAJECTORIES[0]))
        finished = self.batch(jobs)
        self.summarize()
        return finished

    def sensitivity(self):
        freeze = self.freeze()
        if freeze is None: return False
        jobs = []
        for t in self.cfg['tracks']:
            ids = {freeze['selected'][t]['main']['id'], freeze['selected'][t]['anchor_id']}
            for c in candidate_records(self.registry, t):
                if c['id'] in ids:
                    for seed in self.cfg['sensitivity']['seeds']:
                        for i in self.cfg['sensitivity']['trajectories']:
                            jobs.append(self.job('sensitivity', t, c, i,
                                self.cfg['sensitivity']['origins'], paths=512, seed=seed))
        finished = self.batch(jobs)
        self.summarize()
        return finished

    def summarize(self):
        states = {}; stages = {}
        ids = {p.stem for p in (self.output/'ledger').glob('*.json')} | {p.stem for p in (self.output/'terminal').glob('*.json')}
        for jid in sorted(ids):
            r = self.ledger.read(jid) or {}
            job_path = self.output/'jobs'/f'{jid}.json'
            job = read_json(job_path) if job_path.exists() else r.get('job')
            if job is not None:
                terminal = terminal_result(self.output, job)
                if terminal is not None: r = {**r, **terminal}
            if not r: raise ValueError('Status evidence without a bound job')
            states[r['state']] = states.get(r['state'], 0)+1
            stage = (job or {}).get('stage', 'unknown')
            stages.setdefault(stage, {})[r['state']] = stages.setdefault(stage, {}).get(r['state'], 0)+1
        out = {'experiment_identity': self.identity, 'states': states, 'stages': stages,
            'worker_seconds': self.consumed(), 'resources': resource_snapshot(self.output),
            'updated_utc': now(), 'test_access': False}
        if (self.output/'freeze.json').exists():
            freeze = read_json(self.output/'freeze.json'); comparisons = {}
            for t in self.cfg['tracks']:
                ids = {freeze['selected'][t]['main']['id'], freeze['selected'][t]['anchor_id']}
                if freeze['selected'][t].get('sensitivity'):
                    ids.add(freeze['selected'][t]['sensitivity']['id'])
                comparisons[t] = [self.loss(t, c, CONFIRMATION_TRAJECTORIES,
                    freeze['confirmation_origins'], paths=512)
                    for c in candidate_records(self.registry, t) if c['id'] in ids]
            out['confirmation'] = comparisons
        atomic_json(self.output/'status.json', out)
        print(json.dumps({k: v for k, v in out.items() if k not in ('confirmation', 'resources')}, indent=2))
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('worker', 'preflight', 'screen', 'promote', 'precision', 'freeze', 'confirm', 'sensitivity', 'all', 'status'))
    p.add_argument('--config'); p.add_argument('--cache'); p.add_argument('--output'); p.add_argument('--job')
    args = p.parse_args()
    if args.command == 'worker': worker(args.job); return
    if not all((args.config, args.cache, args.output)): p.error('config, cache and output are required')
    c = Campaign(args.config, args.cache, args.output)
    if args.command == 'status': c.summarize(); return
    with (c.output/'scheduler.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'all':
            if c.screen() and c.promote() and c.confirm(): c.sensitivity()
        else: getattr(c, args.command)()


if __name__ == '__main__': main()
