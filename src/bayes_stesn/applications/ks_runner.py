"""Resumable, per-trajectory KS experiment; selection precedes test access.

Each expensive fit runs in its own single-threaded worker. Atomic fit and
origin receipts allow an interrupted study to continue without repeating
completed configurations. The old simulation runner and snapshots are unused.
"""
from __future__ import annotations

import argparse
import copy
import gc
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np

from ..io import content_hash, seed_for, atomic_npz
from .ks_runtime import Ledger, admission, atomic_json, file_hash, now, resource_snapshot

PROTOCOLS = {"A": "interleaved", "B": "all_observed", "C": "contiguous"}
NATIVE = {"bayes_stesn", "separable_gp", "basis_gru"}
STOCHASTIC = {"bayes_stesn", "basis_gru", "ridge_esn", "periodic_cnn"}


def numerical_identity():
    """Bind executed implementation, data adapter and metric definitions."""
    folder = Path(__file__).parent
    names = ('ks_data.py', 'ks_native.py', 'ks_baselines.py', 'ks_cnn.py', 'ks_metrics.py', 'ks_runner.py', 'ks_runtime.py', 'ks_precision.py')
    out = {f'applications/{name}': file_hash(folder/name) for name in names if (folder/name).exists()}
    package = folder.parent
    for p in sorted(package.glob('*.py'))+sorted((package/'benchmark').glob('*.py')):
        out[str(p.relative_to(package))] = file_hash(p)
    import scipy.linalg  # ensure both NumPy and SciPy BLAS are loaded before binding
    from threadpoolctl import threadpool_info, threadpool_limits
    threadpool_limits(1)
    out['runtime'] = {'python': sys.version, 'packages': {name: importlib.metadata.version(name)
        for name in ('numpy','scipy','torch','h5py','threadpoolctl')},
        'blas': [{key: item.get(key) for key in ('internal_api','num_threads','version','architecture')}
            for item in threadpool_info() if item.get('user_api')=='blas' and
                any(part in item.get('filepath','') for part in ('numpy.libs','scipy.libs'))],
        'torch_policy':'CPU; one intraop/inter-op thread; deterministic algorithms for neural fits'}
    return out


def forecast_key(job, first_target):
    return content_hash({'fit': job['fit_id'], 'first_target': first_target,
        'history': job['history'], 'horizon': job['horizon'], 'paths': job['paths'],
        'partition': job['partition'], 'forecast_seed': job['forecast_seed'],
        'score_definition': 'ks-mean-horizon-balanced-v1'})


def forecast_path(root, job, first_target):
    key = forecast_key(job, first_target)
    return Path(root)/'forecasts'/job['fit_id']/job['partition']/f'{first_target:05d}_{key[:12]}.json'


def _load_fit(model, path):
    if model in NATIVE:
        from .ks_native import NativeFit
        return NativeFit.load(path)
    if model == 'periodic_cnn':
        from .ks_cnn import load_cnn
        return load_cnn(path)
    from .ks_baselines import VectorFit
    return VectorFit.load(path)


def _fit(model, training, config, seed, path, validation=None):
    if model in NATIVE:
        from .ks_native import fit_native
        return fit_native(model, training, config, seed, checkpoint_path=path)
    if model == 'periodic_cnn':
        from .ks_cnn import fit_cnn
        fit = fit_cnn(training, config, seed,
            checkpoint_path=path.with_name('optimizer_checkpoint.npz'), validation=validation)
        fit.save(path)
        return fit
    from .ks_baselines import fit_vector
    fit = fit_vector(model, training, config, seed)
    fit.save(path)
    return fit


def _scientific_failure(error):
    return isinstance(error, (FloatingPointError, np.linalg.LinAlgError)) or (
        isinstance(error, ValueError) and str(error).startswith((
            'Forecast mean must be finite', 'Samples must be finite')))


def execute_job(job):
    """Single writer per fitted model, including recovery of orphaned workers."""
    folder = Path(job['output'])/'fits'/job['fit_id']; folder.mkdir(parents=True, exist_ok=True)
    with (folder/'writer.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('A live worker already owns this fit; do not duplicate or overwrite it') from error
        try:
            return _execute_job_locked(job)
        except BaseException as error:
            ledger=Ledger(Path(job['output'])/'ledger')
            previous=ledger.read(job['job_id'])
            if previous and previous['state']=='running':
                ledger.write(job['job_id'],'pending',reason='data_access_or_integrity_blocker',error=repr(error))
            raise


def _execute_job_locked(job):
    """Worker: one reusable fitted model, a finite list of forecast origins."""
    from .ks_data import KSAdapter
    from .ks_metrics import score
    if job['model'] in {'basis_gru', 'periodic_cnn'}:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
    root = Path(job['output']); root.mkdir(parents=True, exist_ok=True)
    if job['source'] != numerical_identity():
        raise ValueError('Job numerical source changed; refuse incompatible execution')
    if job['partition'] == 'test':
        frozen_path, access_path = root/'freeze.json', root/'test_access.json'
        if not frozen_path.exists() or not access_path.exists():
            raise ValueError('Test access requires completed selection and an explicit frozen gate')
        frozen = json.loads(frozen_path.read_text()); access = json.loads(access_path.read_text())
        if (frozen['experiment_identity'] != job['experiment_identity'] or
                access['freeze_identity'] != frozen['identity'] or
                not access.get('both_track_selection_complete')):
            raise ValueError('Test job is not bound to the completed two-track selection')
        chosen = frozen['selected']['A' if job['track']=='C' else job['track']][job['model']]
        if chosen['id'] != job['config_id'] or chosen['config'] != job['config']:
            raise ValueError('Test configuration differs from frozen selection')
        if job['origins'] != frozen['test_origins'] or job['history'] != frozen['history']:
            raise ValueError('Test schedule differs from frozen selection')
        if job['horizon'] != frozen['horizon'] or job['paths'] != frozen['paths']:
            raise ValueError('Test horizon/path count differs from frozen selection')
        if job['seed'] != frozen['model_seed']:
            if (job['seed'] not in frozen['sensitivity_seeds'] or job['model'] not in STOCHASTIC or
                    job['trajectory'] not in frozen['sensitivity_trajectories'] or job['track']=='C'):
                raise ValueError('Test model seed/episode is outside frozen sensitivity policy')
        if job['trajectory'] not in frozen['test_trajectories']:
            raise ValueError('Test trajectory is outside the frozen roster')
        if job['track']=='C' and job['model'] not in frozen['robustness_methods']:
            raise ValueError('Robustness method was not frozen before test access')
    ledger = Ledger(root/'ledger'); jid = job['job_id']
    started = time.perf_counter()
    ledger.write(jid, 'running', job=job, pid=os.getpid(), started_utc=now())
    adapter = KSAdapter(job['cache'], PROTOCOLS[job['track']], job['phase'])
    train = adapter.training(job['trajectory'])
    fitdir = root/'fits'/job['fit_id']; fitdir.mkdir(parents=True, exist_ok=True)
    fitpath, receiptpath = fitdir/'fit.npz', fitdir/'receipt.json'
    fit_identity = content_hash({'model': job['model'], 'track': job['track'],
        'phase': job['phase'], 'training': dict(train.identity), 'config': job['config'],
        'model_seed': job['seed'], 'source': job['source']})
    if fit_identity != job['fit_id']:
        raise ValueError('Training or configuration identity no longer matches scheduled fit')
    try:
        reused = False
        if receiptpath.exists():
            prior = json.loads(receiptpath.read_text())
            if prior.get('identity') != fit_identity:
                raise ValueError('Mismatched fit receipt')
            if prior['state'] == 'failed':
                ledger.write(jid, 'failed', error=prior['error'], fit_id=job['fit_id'], reused_failure=True)
                return
            if fitpath.exists() and file_hash(fitpath) == prior['fit_sha256']:
                fit = _load_fit(job['model'], fitpath); reused = True
            else:
                raise ValueError('Completed fit artifact missing or hash mismatch; never retrain silently')
        if not reused:
            start = time.perf_counter()
            try:
                fit = _fit(job['model'], train.panel, job['config'], job['seed'], fitpath)
                qualified = bool(fit.diagnostics.get('qualified', True))
                prior = {'identity': fit_identity, 'state': 'complete' if qualified else 'inference-unqualified',
                    'model': job['model'], 'track': job['track'], 'phase': job['phase'],
                    'trajectory': job['trajectory'], 'seed': job['seed'], 'config': job['config'],
                    'training_identity': dict(train.identity), 'fit_seconds': time.perf_counter()-start,
                    'diagnostics': fit.diagnostics, 'fit_sha256': file_hash(fitpath),
                    'fit_bytes': fitpath.stat().st_size, 'resource': resource_snapshot(root), 'completed_utc': now()}
                atomic_json(receiptpath, prior)
            except Exception as error:
                if not _scientific_failure(error):
                    raise
                prior = {'identity': fit_identity, 'state': 'failed', 'error': repr(error),
                    'traceback': traceback.format_exc(), 'fit_seconds': time.perf_counter()-start,
                    'model':job['model'], 'track':job['track'], 'phase':job['phase'],
                    'trajectory':job['trajectory'], 'seed':job['seed'], 'config':job['config'],
                    'resource': resource_snapshot(root), 'completed_utc': now()}
                atomic_json(receiptpath, prior)
                ledger.write(jid, 'failed', fit_id=fit_identity, error=repr(error), elapsed_seconds=time.perf_counter()-started)
                return
        qualified = bool(fit.diagnostics.get('qualified', True))
        forecastdir = root/'forecasts'/job['fit_id']/job['partition']; forecastdir.mkdir(parents=True, exist_ok=True)
        completed, failed, forecast_total, reuse_origins = 0, 0, 0., 0
        for first_target in job['origins']:
            key = forecast_key(job, first_target)
            outpath = forecast_path(root, job, first_target)
            if outpath.exists():
                previous = json.loads(outpath.read_text())
                if previous['identity'] != key:
                    raise ValueError('Forecast identity mismatch')
                completed += previous['state'] == 'complete'; failed += previous['state'] == 'failed'
                reuse_origins += 1; continue
            base = {'identity': key, 'fit_id': fit_identity, 'model': job['model'],
                'track': job['track'], 'phase': job['phase'], 'trajectory': job['trajectory'],
                'seed': job['seed'], 'first_target': first_target, 'partition': job['partition'],
                'qualified': qualified, 'history': job['history'], 'horizon': job['horizon'],
                'config_id': job['config_id'], 'training_scale': train.transform.scale,
                'model_seed': job['seed'], 'scheduled_paths': job['paths'] if job['model'] in NATIVE else 0}
            start = time.perf_counter()
            try:
                window = adapter.model_window(job['trajectory'], first_target, job['partition'],
                    history=job['history'], horizon=job['horizon'])
                stream = seed_for(job['forecast_seed'], job['phase'], job['track'], job['trajectory'],
                    job['model'], job['seed'], first_target)
                if job['model'] in NATIVE:
                    samples = np.empty((job['paths'], job['horizon'], len(window.target_site_ids)), dtype=np.float64)
                    for batch in fit.iter_forecasts(window.panel, job['horizon'], window.target_coordinates,
                            window.target_site_ids, job['paths'], stream, batch_size=job['path_batch']):
                        a = batch.metadata['path_start']; b = batch.metadata['path_stop']
                        samples[a:b] = train.transform.inverse(batch.samples)
                    mean = samples.mean(axis=0, dtype=np.float64)
                else:
                    prediction = fit.forecast(window.panel, job['horizon'], window.target_coordinates,
                        window.target_site_ids, n_paths=0, seed=stream)
                    mean = train.transform.inverse(prediction.mean); samples = None
                elapsed = time.perf_counter()-start
                # Evaluation is first requested after prediction has completed.
                evaluation = adapter.evaluation_window(job['trajectory'], first_target, job['partition'],
                    history=job['history'], horizon=job['horizon'])
                metrics = score(mean, samples, evaluation.values, train.transform.scale, window.target_coordinates)
                payload = {**base, 'state': 'complete', 'scores': metrics, 'forecast_seconds': elapsed,
                    'score_seconds': time.perf_counter()-start-elapsed, 'completed_utc': now()}
                if job['retain_example'] and first_target == job['origins'][0]:
                    visual = root/'predictions'/f"{job['phase']}_{job['track']}_{job['model']}_s{job['seed']}_t{job['trajectory']:02d}_a{first_target}.npz"
                    arrays = {'truth': evaluation.values, 'mean': mean, 'x': window.target_x,
                        'input_x': window.x, 'times': window.target_time, 'site_ids': window.target_site_ids}
                    if samples is not None:
                        arrays['quantile_levels'] = np.array([.025, .05, .1, .25, .5, .75, .9, .95, .975])
                        arrays['quantiles'] = np.quantile(samples, arrays['quantile_levels'], axis=0)
                        arrays['paths_first_32'] = samples[:32]
                    atomic_npz(visual, **arrays); payload['example_artifact'] = str(visual.relative_to(root))
                atomic_json(outpath, payload); completed += 1; forecast_total += elapsed
                del samples, mean
            except Exception as error:
                if not _scientific_failure(error):
                    raise
                atomic_json(outpath, {**base, 'state': 'failed', 'error': repr(error),
                    'forecast_seconds': time.perf_counter()-start, 'traceback': traceback.format_exc(),
                    'completed_utc': now()})
                failed += 1
        state = 'complete' if qualified else 'inference-unqualified'
        if failed:
            state = 'failed'
        ledger.write(jid, state, fit_id=fit_identity, qualified=qualified,
            completed_origins=completed, failed_origins=failed, reused_fit=reused,
            reused_origins=reuse_origins, new_forecast_seconds=forecast_total,
            elapsed_seconds=time.perf_counter()-started, resource=resource_snapshot(root))
    except BaseException as error:
        ledger.write(jid, 'pending', reason='infrastructure_interruption_or_integrity_blocker', error=repr(error), traceback=traceback.format_exc(),
            elapsed_seconds=time.perf_counter()-started)
        raise


class Study:
    def __init__(self, config, cache, output):
        self.config_path = Path(config).resolve(); self.cfg = json.loads(self.config_path.read_text())
        self.cache = Path(cache).resolve(); self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.source = numerical_identity(); self.ledger = Ledger(self.output/'ledger')
        self.identity = content_hash({'config': self.cfg, 'source': self.source})
        plan = self.output/'experiment.json'
        if plan.exists() and json.loads(plan.read_text())['identity'] != self.identity:
            raise ValueError('Experiment source/config changed; create a new versioned run root')
        if not plan.exists():
            atomic_json(plan, {'identity': self.identity, 'config': self.cfg, 'source': self.source,
                'created_utc': now(), 'base_revision': self.cfg['base_revision']})
        self.started = time.monotonic(); self.new_jobs = 0; self.stop_scheduling = Event()

    def submit_jobs(self, jobs, **limits):
        workers = int(self.cfg['budget'].get('workers', 1))
        if workers == 1:
            return [self.submit(j, **limits) for j in jobs]
        # Jobs in this batch have distinct fit identities: no shared checkpoint writers.
        if len({j['fit_id'] for j in jobs}) != len(jobs):
            raise ValueError('Concurrent jobs cannot share one fit checkpoint')
        peaks = sorted([self.cfg['budget'].get('estimated_peak_rss_bytes', {}).get(j['model'],150*1024**2)
                        for j in jobs], reverse=True)[:workers]
        # This method waits for its entire group before another group can be
        # admitted. Reserve all simultaneous workers together before spawning.
        admission(self.output, reserve_bytes=self.cfg['budget']['disk_reserve_bytes'],
            estimated_peak_bytes=sum(peaks),
            worker_cap_bytes=self.cfg['budget'].get('combined_worker_cap_bytes', int(3.8*1024**3)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            def guarded(job):
                if self.stop_scheduling.is_set():
                    self.ledger.write(job['job_id'],'pending',job=job,reason='earlier_scheduling_blocker')
                    raise StopIteration('Earlier scheduling blocker; retained work is resumable')
                try:
                    return self.submit(job,**limits)
                except BaseException:
                    self.stop_scheduling.set()
                    raise
            futures = [pool.submit(guarded, j) for j in jobs]
            return [f.result() for f in futures]

    def consumed_worker_seconds(self, stages=None):
        total=0.
        for p in (self.output/'attempts').glob('*.json'):
            r=json.loads(p.read_text())
            if stages is not None and r['stage'] not in stages:
                continue
            if r['state']=='running':
                ledger=self.ledger.read(r['job_id']) or {}
                try:
                    command=Path(f"/proc/{r['pid']}/cmdline").read_bytes().decode(errors='replace')
                    alive=r['job_id'] in command
                except (OSError,KeyError):
                    alive=False
                if alive:
                    total+=max(0.,time.time()-r['started_epoch'])
                else:
                    elapsed=float(r.get('last_observed_elapsed_seconds',0.))
                    exact=ledger.get('pid')==r.get('pid') and 'elapsed_seconds' in ledger
                    elapsed=max(elapsed,float(ledger['elapsed_seconds'])) if exact else elapsed+10.
                    atomic_json(p,{**r,'state':'reconciled_after_interruption','elapsed_seconds':elapsed,
                        'accounting':'worker terminal receipt' if exact else 'last heartbeat plus10s bounded estimate',
                        'accounting_exact':bool(exact),'reconciled_utc':now()})
                    total+=elapsed
            else:
                total+=r['elapsed_seconds']
        return total

    def make_job(self, stage, track, model, candidate, trajectory, origins, *, phase='core', seed=None, partition='validation', retain=False):
        from .ks_data import KSAdapter
        seed = self.cfg['model_seed'] if seed is None else seed
        train = KSAdapter(self.cache, PROTOCOLS[track], phase).training(trajectory)
        config = candidate['config']; cid = candidate['id']
        fit_id = content_hash({'model': model, 'track': track, 'phase': phase,
            'training': dict(train.identity), 'config': config, 'model_seed': seed, 'source': self.source})
        job = {'output': str(self.output), 'cache': str(self.cache), 'source': self.source,
            'experiment_identity': self.identity, 'stage': stage, 'track': track, 'phase': phase,
            'model': model, 'config': config, 'config_id': cid, 'trajectory': trajectory,
            'origins': list(map(int, origins)), 'seed': int(seed), 'fit_id': fit_id,
            'partition': partition, 'history': self.cfg['history'], 'horizon': self.cfg['horizon'],
            'paths': self.cfg['paths'], 'path_batch': self.cfg['path_batch'],
            'forecast_seed': self.cfg['forecast_seed'], 'retain_example': bool(retain)}
        if partition=='test' and (self.output/'freeze.json').exists():
            job['paths']=int(json.loads((self.output/'freeze.json').read_text())['paths'])
        job['job_id'] = content_hash(job)[:24]
        return job

    def submit(self, job, *, max_jobs=None, wall_seconds=None):
        current = self.ledger.read(job['job_id'])
        if current and current['state'] in {'complete', 'failed', 'inference-unqualified'}:
            return current
        if current and current['state']=='running' and current.get('pid'):
            try: os.kill(int(current['pid']),0)
            except ProcessLookupError: pass
            else: raise RuntimeError('An earlier live worker owns this job; wait for its completion')
        total_cap=self.cfg['budget'].get('total_worker_seconds_cap',self.cfg['budget']['initial_total_wall_hours']*3600*self.cfg['budget'].get('workers',1))
        if self.consumed_worker_seconds()>=total_cap:
            self.ledger.write(job['job_id'],'pending',job=job,reason='cumulative_campaign_compute_cap')
            raise StopIteration('Cumulative campaign compute budget reached')
        if job['stage'] in {'screen','promote','refine','precision'}:
            search_cap=self.cfg['budget'].get('search_worker_seconds_cap',self.cfg['budget']['search_wall_hours_cap']*3600*self.cfg['budget'].get('workers',1))
            if self.consumed_worker_seconds({'screen','promote','refine','precision'})>=search_cap:
                self.ledger.write(job['job_id'],'pending',job=job,reason='cumulative_screening_compute_cap')
                raise StopIteration('Cumulative screening compute cap reached; final budget reserved')
        if ((max_jobs is not None and self.new_jobs >= max_jobs) or
                (wall_seconds is not None and time.monotonic()-self.started >= wall_seconds)):
            self.ledger.write(job['job_id'], 'pending', job=job, reason='bounded_invocation_budget')
            raise StopIteration('Invocation budget reached; resume uses all retained results')
        estimate = self.cfg['budget'].get('estimated_peak_rss_bytes', {}).get(job['model'], 150*1024**2)
        snap = admission(self.output, reserve_bytes=self.cfg['budget']['disk_reserve_bytes'], estimated_peak_bytes=estimate,
            worker_cap_bytes=self.cfg['budget'].get('combined_worker_cap_bytes', int(3.8*1024**3)))
        self.ledger.write(job['job_id'], 'pending', job=job, admission=snap)
        jobs = self.output/'jobs'; logs = self.output/'logs'; jobs.mkdir(exist_ok=True); logs.mkdir(exist_ok=True)
        path = jobs/(job['job_id']+'.json'); atomic_json(path, job)
        env = dict(os.environ)
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            env[name] = '1'
        env['PYTHONHASHSEED'] = '0'; env['MPLBACKEND'] = 'Agg'
        timeout = self.cfg['budget'].get('job_timeout_seconds', {}).get(job['model'], 1800)
        print(json.dumps({'event': 'start', 'stage': job['stage'], 'track': job['track'],
            'model': job['model'], 'candidate': job['config_id'], 'trajectory': job['trajectory'], 'job_id': job['job_id']}), flush=True)
        with (logs/(job['job_id']+'.log')).open('a') as stream:
            process = subprocess.Popen([sys.executable, '-m', 'bayes_stesn.applications.ks_runner', 'worker', '--job', str(path)],
                env=env, stdout=stream, stderr=subprocess.STDOUT)
            attempt_path=self.output/'attempts'/f"{job['job_id']}_{time.time_ns()}.json"
            attempt={'job_id':job['job_id'],'stage':job['stage'],'model':job['model'],'track':job['track'],
                'state':'running','started_epoch':time.time(),'started_utc':now(),'pid':process.pid}
            atomic_json(attempt_path,attempt)
            begin = time.monotonic(); last_check = -10.; peak_rss = 0; stopped = None
            while process.poll() is None:
                elapsed = time.monotonic()-begin
                if elapsed-last_check >= 2.:
                    current_resources = resource_snapshot(self.output)
                    try:
                        status = Path(f'/proc/{process.pid}/status').read_text().splitlines()
                        rss = next(int(line.split()[1])*1024 for line in status if line.startswith('VmRSS:'))
                        peak_rss = max(peak_rss, rss)
                    except (OSError, StopIteration):
                        pass
                    mem = current_resources['memory']
                    memory_limit = int(mem['memory.max']) if mem.get('memory.max') not in (None,'max') else None
                    if elapsed > timeout:
                        stopped = 'runtime_timeout_checkpoint_preserved'
                    elif current_resources['disk_free_bytes'] < self.cfg['budget']['disk_reserve_bytes']:
                        stopped = 'disk_reserve_checkpoint_preserved'
                    elif peak_rss > self.cfg['budget'].get('worker_cap_bytes', int(3.8*1024**3)):
                        stopped = 'worker_memory_budget_checkpoint_preserved'
                    elif memory_limit and memory_limit-int(mem['memory.current'])+int(mem.get('inactive_file_bytes',0)) < 256*1024**2:
                        stopped = 'container_headroom_checkpoint_preserved'
                    if stopped:
                        process.send_signal(signal.SIGINT)
                        try: process.wait(timeout=20)
                        except subprocess.TimeoutExpired: process.terminate(); process.wait(timeout=10)
                        self.ledger.write(job['job_id'], 'pending', reason=stopped,
                            monitored_peak_rss_bytes=peak_rss, resource=current_resources)
                        atomic_json(attempt_path,{**attempt,'state':'interrupted','elapsed_seconds':time.monotonic()-begin,
                            'reason':stopped,'peak_rss_bytes':peak_rss,'finished_utc':now()})
                        raise RuntimeError(f"Measured resource blocker: {stopped}; retained checkpoint is resumable")
                    if int(elapsed)//10 > int(last_check)//10:
                        atomic_json(attempt_path,{**attempt,'last_observed_elapsed_seconds':elapsed,
                            'last_observed_utc':now(),'peak_rss_bytes':peak_rss})
                    last_check = elapsed
                time.sleep(.2)
            returncode = process.returncode
            atomic_json(attempt_path,{**attempt,'state':'complete' if returncode==0 else 'interrupted',
                'returncode':returncode,'elapsed_seconds':time.monotonic()-begin,
                'peak_rss_bytes':peak_rss,'finished_utc':now()})
        self.new_jobs += 1
        current = self.ledger.read(job['job_id'])
        if returncode:
            self.ledger.write(job['job_id'], 'pending', reason='worker_interruption', exit_code=returncode)
            raise RuntimeError('Worker interrupted; inspect log before exact resume')
        print(json.dumps({'event': 'end', 'job_id': job['job_id'], 'state': current['state'],
            'seconds': current.get('elapsed_seconds'), 'failed_origins': current.get('failed_origins')}), flush=True)
        return current

    def candidate_loss(self, track, model, candidate, trajectories, origins):
        losses = []; failures = 0; qualified_episodes = 0; rows = []
        for trajectory in trajectories:
            job = self.make_job('identity', track, model, candidate, trajectory, origins)
            episode = []
            by_origin = {}
            for origin in origins:
                p = forecast_path(self.output, job, origin)
                if p.exists():
                    r = json.loads(p.read_text())
                    if r['identity'] != forecast_key(job, origin):
                        raise ValueError('Invalid forecast key in selection')
                    by_origin[origin] = r
            qualified = True
            for origin in origins:
                r = by_origin.get(origin)
                if r is None or r['state'] != 'complete':
                    failures += 1; qualified = False; continue
                qualified &= r['qualified']
                metric = 'selection_crps' if model in NATIVE else 'selection_mse'
                episode.append(r['scores'][metric])
            if len(episode) == len(origins):
                losses.append(float(np.mean(episode)))
            if qualified and len(episode) == len(origins):
                qualified_episodes += 1
            rows.append({'trajectory': trajectory, 'complete_origins': len(episode),
                'qualified': qualified, 'loss': float(np.mean(episode)) if len(episode) == len(origins) else None})
        valid = len(losses) == len(trajectories) and qualified_episodes == len(trajectories)
        return {'candidate': candidate, 'eligible': valid, 'failed_origins': failures,
            'qualified_episodes': qualified_episodes, 'episodes': rows,
            'loss': float(np.mean(losses)) if len(losses) == len(trajectories) else None,
            'objective': 'episode_balanced_horizon_band_CRPS' if model in NATIVE else 'episode_balanced_horizon_band_MSE'}

    def screen(self, **limits):
        if (self.output/'test_access.json').exists():
            raise ValueError('Validation selection is sealed after test access')
        selection = self.output/'selection'; selection.mkdir(exist_ok=True)
        cap = self.cfg['budget']['search_wall_hours_cap']*3600
        limits['wall_seconds'] = min(cap, limits['wall_seconds']) if limits.get('wall_seconds') is not None else cap
        for stage in ('screen', 'promote'):
            settings = self.cfg[stage]
            for track in ('A', 'B'):
                for model, candidates in self.cfg['candidates'][track].items():
                    stage_path=selection/f'{track}_{model}_{stage}.json'
                    if stage_path.exists():
                        saved=json.loads(stage_path.read_text())
                        if saved.get('experiment_identity')!=self.identity or saved['trajectories']!=settings['trajectories'] or saved['origins']!=settings['origins']:
                            raise ValueError('Existing selection receipt has incompatible experiment/schedule')
                        continue
                    chosen = candidates
                    if stage == 'promote':
                        first = json.loads((selection/f'{track}_{model}_screen.json').read_text())
                        chosen = [r['candidate'] for r in first['ranking'][:min(self.cfg['promote']['count'], len(candidates))]]
                    for candidate in chosen:
                        jobs = [self.make_job(stage, track, model, candidate, trajectory, settings['origins'])
                            for trajectory in settings['trajectories']]
                        self.submit_jobs(jobs, **limits)
                    ranking = [self.candidate_loss(track, model, c, settings['trajectories'], settings['origins']) for c in chosen]
                    # Eligibility/failure counts precede any successful-only loss.
                    ranking.sort(key=lambda r: (not r['eligible'], r['failed_origins'],
                        -r['qualified_episodes'], float('inf') if r['loss'] is None else r['loss'], r['candidate']['id']))
                    if ranking[0]['eligible']:
                        threshold = ranking[0]['loss']*(1+self.cfg['scoring']['selection_near_tie_relative_tolerance'])
                        tied = [r for r in ranking if r['eligible'] and r['loss'] <= threshold]
                        simple = min(tied, key=lambda r: r['candidate'].get('complexity_order', 0))
                        for r in ranking:
                            r['validation_near_tie'] = r in tied
                        ranking.remove(simple); ranking.insert(0, simple)
                    atomic_json(stage_path, {'experiment_identity':self.identity, 'stage': stage, 'ranking': ranking,
                        'trajectories': settings['trajectories'], 'origins': settings['origins'], 'completed_utc': now()})
        from .ks_precision import run_validation_precision
        self.refine_boundaries(**limits)
        run_validation_precision(self)
        return self.freeze()

    def refine_boundaries(self, **limits):
        """One predeclared cheap penalty/epoch-budget refinement per family/track."""
        selection=self.output/'selection'
        policy=self.cfg.get('boundary_refinement',{})
        for track in ('A','B'):
            for model in policy.get('models',[]):
                if model not in self.cfg['candidates'][track]:
                    continue
                receipt=selection/f'{track}_{model}_refinement.json'
                if receipt.exists():
                    continue
                promoted_path=selection/f'{track}_{model}_promote.json'
                promoted=json.loads(promoted_path.read_text());best=promoted['ranking'][0]
                proposal=None;reason='not an eligible consequential declared boundary'
                if best['eligible']:
                    cfg=copy.deepcopy(best['candidate']['config'])
                    if model in {'linear_var','pod_nvar','ridge_esn'}:
                        values=[c['config']['ridge'] for c in self.cfg['candidates'][track][model]]
                        if cfg['ridge']==min(values):
                            cfg['ridge']/=10.;proposal=cfg;reason='lower evaluated ridge boundary'
                        elif cfg['ridge']==max(values):
                            cfg['ridge']*=10.;proposal=cfg;reason='upper evaluated ridge boundary'
                    elif model=='periodic_cnn':
                        ceiling=max(c['config']['epochs'] for c in self.cfg['candidates'][track][model])
                        if cfg['epochs']==ceiling:
                            cfg['epochs']=min(90,ceiling+30);proposal=cfg;reason='largest evaluated training epoch budget'
                if proposal is None:
                    atomic_json(receipt,{'performed':False,'reason':reason,'completed_utc':now()});continue
                candidate={'id':best['candidate']['id']+'_refine01','config':proposal,
                    'complexity_order':best['candidate'].get('complexity_order',0)+.5}
                settings=self.cfg['promote']
                jobs=[self.make_job('refine',track,model,candidate,t,settings['origins']) for t in settings['trajectories']]
                self.submit_jobs(jobs,**limits)
                result=self.candidate_loss(track,model,candidate,settings['trajectories'],settings['origins'])
                original_hash=file_hash(promoted_path)
                ranking=promoted['ranking']+[result]
                ranking.sort(key=lambda r:(not r['eligible'],r['failed_origins'],-r['qualified_episodes'],
                    float('inf') if r['loss'] is None else r['loss'],r['candidate']['id']))
                if ranking[0]['eligible']:
                    threshold=ranking[0]['loss']*(1+self.cfg['scoring']['selection_near_tie_relative_tolerance'])
                    tied=[r for r in ranking if r['eligible'] and r['loss']<=threshold]
                    simple=min(tied,key=lambda r:r['candidate'].get('complexity_order',0))
                    for r in ranking:r['validation_near_tie']=r in tied
                    ranking.remove(simple);ranking.insert(0,simple)
                atomic_json(selection/f'{track}_{model}_before_refinement.json',promoted)
                atomic_json(promoted_path,{**promoted,'ranking':ranking,'bounded_refinement':True,'completed_utc':now()})
                atomic_json(receipt,{'performed':True,'reason':reason,'candidate':candidate,
                    'original_promotion_sha256':original_hash,'result':result,'completed_utc':now()})

    def freeze(self):
        path = self.output/'freeze.json'
        if path.exists():
            return json.loads(path.read_text())
        precision_path=self.output/'validation_precision.json'
        if not precision_path.exists():
            raise ValueError('Validation predictive Monte Carlo precision must be checked before freezing test access')
        precision=json.loads(precision_path.read_text())
        selected = {}; issues = []
        for track in ('A', 'B'):
            selected[track] = {}
            for model in self.cfg['candidates'][track]:
                p = self.output/'selection'/f'{track}_{model}_promote.json'
                if not p.exists():
                    raise ValueError('Both main tracks must finish validation before any test score')
                result = json.loads(p.read_text()); best = result['ranking'][0]
                # A failed family remains represented, not silently removed.
                selected[track][model] = {**best['candidate'], 'selection_eligible': best['eligible'],
                    'selection_loss': best['loss'], 'selection_objective': best['objective'],
                    'selection_receipt_sha256': file_hash(p)}
                if not best['eligible']:
                    issues.append(f'{track}/{model}: no fully qualified candidate on all promotion cases')
        frozen = {'experiment_identity': self.identity, 'selected': selected, 'created_utc': now(),
            'history': self.cfg['history'], 'horizon': self.cfg['horizon'], 'paths': int(precision['selected_paths']),
            'validation_precision_sha256':file_hash(precision_path),
            'model_seed': self.cfg['model_seed'], 'sensitivity_seeds': self.cfg['sensitivity']['seeds'],
            'sensitivity_trajectories': self.cfg['sensitivity']['trajectories'],
            'test_origins': self.cfg['test_origins'], 'test_trajectories': list(range(20)),
            'plot_cases': self.cfg['plot_cases'], 'selection_issues': issues,
            'robustness_methods': self.cfg['robustness']['models'],
            'robustness_configuration_policy': 'selected_interleaved_configuration_transferred_without_test_ranking; fit new contiguous training transform/features',
            'scoring': self.cfg['scoring'], 'source': self.source}
        frozen['identity'] = content_hash(frozen); atomic_json(path, frozen)
        return frozen

    def final(self, *, sensitivity=False, robustness=False, **limits):
        frozen = self.freeze()
        # Opening this gate happens only after both tracks' selection receipts exist.
        gate = self.output/'test_access.json'
        if not gate.exists():
            atomic_json(gate, {'freeze_identity': frozen['identity'], 'first_opened_utc': now(),
                'both_track_selection_complete': True})
        if sensitivity:
            tracks = ('A', 'B'); episodes = self.cfg['sensitivity']['trajectories']; seeds = self.cfg['sensitivity']['seeds']
        else:
            tracks = ('C',) if robustness else ('A', 'B'); episodes = range(20); seeds = [self.cfg['model_seed']]
        for track in tracks:
            selected = frozen['selected']['A' if track == 'C' else track]
            for model, candidate in selected.items():
                if sensitivity and model not in STOCHASTIC:
                    continue
                if robustness and model not in self.cfg['robustness']['models']:
                    continue
                for seed in seeds:
                    jobs = [self.make_job('sensitivity' if sensitivity else ('robustness' if robustness else 'final'),
                            track, model, candidate, trajectory, self.cfg['test_origins'], seed=seed, partition='test',
                            retain=trajectory == 0 and seed == self.cfg['model_seed'])
                        for trajectory in episodes]
                    self.submit_jobs(jobs, **limits)
        return self.summarize()

    def summarize(self):
        from .ks_metrics import cluster_summary
        frozen = self.freeze(); rows = []; episodes = []; curves = []; paired = []
        for track in ('A', 'B', 'C'):
            selected = frozen['selected']['A' if track == 'C' else track]
            for model, candidate in selected.items():
                if track == 'C' and model not in self.cfg['robustness']['models']:
                    continue
                for seed in [self.cfg['model_seed']]+(self.cfg['sensitivity']['seeds'] if model in STOCHASTIC and track != 'C' else []):
                    subset = range(20) if seed == self.cfg['model_seed'] else self.cfg['sensitivity']['trajectories']
                    values = []; n_fail = 0; qualified_trajectories = 0; pending = 0; fit_seconds = 0.; forecast_seconds = 0.; maxrss = 0
                    fit_counts={'complete':0,'failed':0,'inference-unqualified':0,'pending':0}
                    for trajectory in subset:
                        job = self.make_job('identity', track, model, candidate, trajectory, self.cfg['test_origins'], seed=seed, partition='test')
                        fitreceipt = self.output/'fits'/job['fit_id']/'receipt.json'
                        if fitreceipt.exists():
                            fitr = json.loads(fitreceipt.read_text()); fit_seconds += fitr.get('fit_seconds', 0.)
                            maxrss = max(maxrss, fitr.get('resource', {}).get('process_peak_rss_bytes', 0))
                            fit_counts[fitr['state']]=fit_counts.get(fitr['state'],0)+1
                        else:
                            fit_counts['pending']+=1
                        observed = {}
                        for origin in self.cfg['test_origins']:
                            p = forecast_path(self.output, job, origin)
                            if p.exists():
                                r = json.loads(p.read_text())
                                if r['identity'] != forecast_key(job, origin):
                                    raise ValueError('Invalid forecast key in final summary')
                                observed[origin] = r
                        valid = [r for r in observed.values() if r['state'] == 'complete']
                        failed = sum(r['state'] == 'failed' for r in observed.values())
                        missing = len(self.cfg['test_origins'])-len(observed); pending += missing; n_fail += failed
                        if fitreceipt.exists() and fitr['state'] == 'failed':
                            n_fail += missing; pending -= missing; failed += missing
                        forecast_seconds += sum(r.get('forecast_seconds', 0.) for r in observed.values())
                        ep = {'track': track, 'model': model, 'seed': seed, 'trajectory': trajectory,
                            'successful_origins': len(valid), 'failed_origins': failed, 'pending_origins': missing if not (fitreceipt.exists() and fitr['state']=='failed') else 0,
                            'qualified': bool(valid) and all(r['qualified'] for r in valid)}
                        if len(valid) == len(self.cfg['test_origins']):
                            q = ep['qualified']; qualified_trajectories += q
                            # Mean trajectory RMSE of all site/origin/horizon cells.
                            mse = np.mean([r['scores']['mse_by_horizon'] for r in valid], axis=0)
                            mae = np.mean([r['scores']['mae_by_horizon'] for r in valid], axis=0)
                            ep.update(rmse=float(np.sqrt(mse.mean())), nrmse=float(np.sqrt(mse.mean())/valid[0]['training_scale']),
                                mae=float(mae.mean()), vpt_steps=float(np.mean([r['scores']['vpt_steps'] for r in valid])),
                                vpt_censored_fraction=float(np.mean([r['scores']['vpt_right_censored'] for r in valid])),
                                divergence_fraction=float(np.mean([r['scores']['diverged'] for r in valid])))
                            curve = {'track': track, 'model': model, 'seed': seed, 'trajectory': trajectory,
                                'rmse': np.sqrt(mse).tolist(), 'nrmse': (np.sqrt(mse)/valid[0]['training_scale']).tolist(), 'mae': mae.tolist(), 'qualified': q}
                            if model in NATIVE and q:
                                for metric in ('crps', 'normalized_crps', 'spatial_energy', 'spatial_variogram'):
                                    ep[metric] = float(np.mean([r['scores'][metric] for r in valid]))
                                for metric in ('crps_by_horizon', 'spatial_energy_by_horizon', 'spatial_variogram_by_horizon'):
                                    curve[metric] = np.mean([r['scores'][metric] for r in valid], axis=0).tolist()
                                for level in (.5, .8, .9, .95):
                                    for metric in ('coverage', 'width'):
                                        ep[f'{metric}_{level}'] = float(np.mean([r['scores']['intervals'][str(level)][metric] for r in valid]))
                                        curve[f'{metric}_{level}'] = np.mean([r['scores']['intervals'][str(level)][metric+'_by_horizon'] for r in valid], axis=0).tolist()
                            values.append(ep); curves.append(curve)
                        episodes.append(ep)
                    row = {'track': track, 'model': model, 'seed': seed, 'selected_config': candidate['id'],
                        'scheduled_trajectories': len(list(subset)), 'complete_trajectories': len(values),
                        'qualified_trajectories': qualified_trajectories, 'scheduled_origins': len(list(subset))*len(self.cfg['test_origins']),
                        'failed_origins': n_fail, 'pending_origins': pending,
                        'qualified_fits':fit_counts['complete'], 'inference_unqualified_fits':fit_counts['inference-unqualified'],
                        'failed_fits':fit_counts['failed'], 'pending_fits':fit_counts['pending'],
                        'fit_seconds': fit_seconds, 'forecast_seconds': forecast_seconds, 'peak_worker_rss_bytes': maxrss,
                        'selection_eligible': candidate['selection_eligible'],
                        'score_population': 'complete-trajectory successful-only; failures and pending retained in denominators'}
                    for metric in ('rmse','mae','nrmse','vpt_steps','vpt_censored_fraction','divergence_fraction','crps','normalized_crps','spatial_energy','spatial_variogram'):
                        row[metric] = cluster_summary([r[metric] for r in values if metric in r])
                    for level in (.5, .8, .9, .95):
                        for metric in ('coverage', 'width'):
                            key = f'{metric}_{level}'; row[key] = cluster_summary([r[key] for r in values if key in r])
                    rows.append(row)
        for track in ('A','B','C'):
            base = {r['trajectory']: r for r in episodes if r['track']==track and r['model']=='bayes_stesn' and r['seed']==self.cfg['model_seed'] and 'rmse' in r}
            for model in sorted(set(r['model'] for r in episodes if r['track']==track)-{'bayes_stesn'}):
                other = {r['trajectory']:r for r in episodes if r['track']==track and r['model']==model and r['seed']==self.cfg['model_seed'] and 'rmse' in r}
                for metric in ('rmse','nrmse','mae','crps','normalized_crps'):
                    pairs = sorted(k for k in base.keys() & other.keys() if metric in base[k] and metric in other[k])
                    paired.append({'track':track,'reference':'bayes_stesn','comparator':model,'metric':metric,
                        'paired_trajectories':pairs,'difference_direction':'Bayesian ST-ESN minus comparator; negative favors Bayesian',
                        **cluster_summary([base[k][metric]-other[k][metric] for k in pairs])})
        summary = {'freeze_identity': frozen['identity'], 'generated_utc':now(), 'rows':rows,'paired_comparisons':paired,
            'aggregation':'equal trajectory means; reported RMSE is mean trajectory root-MSE, not root of pooled trajectory MSE; origins clustered within trajectory; conditional on validation-selected configurations'}
        atomic_json(self.output/'summary.json',summary);atomic_json(self.output/'episode_results.json',episodes);atomic_json(self.output/'horizon_curves.json',curves)
        return summary


def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['worker','screen','freeze','final','sensitivity','robustness','summarize','resume','status'])
    p.add_argument('--job');p.add_argument('--config');p.add_argument('--cache');p.add_argument('--output')
    p.add_argument('--max-jobs',type=int);p.add_argument('--wall-hours',type=float)
    a=p.parse_args()
    if a.command=='worker':
        execute_job(json.loads(Path(a.job).read_text()));return
    if not all((a.config,a.cache,a.output)):
        p.error('--config --cache --output are required')
    study=Study(a.config,a.cache,a.output)
    ownership=None
    if a.command not in {'status','summarize'}:
        ownership=(study.output/'scheduler.lock').open('a+')
        try: fcntl.flock(ownership,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another live scheduler owns this experiment; use status instead of duplicate resume') from error
    limits={'max_jobs':a.max_jobs,'wall_seconds':None if a.wall_hours is None else a.wall_hours*3600}
    try:
        if a.command=='screen':study.screen(**limits)
        elif a.command=='freeze':study.freeze()
        elif a.command=='final':study.final(**limits)
        elif a.command=='sensitivity':study.final(sensitivity=True,**limits)
        elif a.command=='robustness':study.final(robustness=True,**limits)
        elif a.command=='summarize':study.summarize()
        elif a.command=='resume':
            if not (study.output/'freeze.json').exists():study.screen(**limits)
            study.final(**limits);study.final(sensitivity=True,**limits);study.final(robustness=True,**limits)
        else:
            counts={}
            for f in (study.output/'ledger').glob('*.json'):
                state=json.loads(f.read_text())['state'];counts[state]=counts.get(state,0)+1
            print(json.dumps({'identity':study.identity,'ledger':counts,'frozen':(study.output/'freeze.json').exists()},indent=2))
    except StopIteration as e:
        print(str(e),flush=True)


if __name__=='__main__':
    main()
