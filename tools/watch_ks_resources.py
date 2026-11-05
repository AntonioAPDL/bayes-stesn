#!/usr/bin/env python3
"""Read-only outer-/proc resource observer for a running KS application.

The executor may give Popen a nested-namespace PID while mounting the outer
namespace's /proc. Identify workers by their exact, absolute --job path, then
read the same proc entry's status. This observer never changes model state.
The scheduler's valid cgroup/disk guards remain responsible for stopping work.
"""
from __future__ import annotations
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def session_id(record):
    return hashlib.sha256(record['started_utc'].encode()).hexdigest()[:20]


def merge_sessions(records):
    """Combine distinct observation sessions, deduplicating repeated snapshots.

    Aggregate maxima do not imply that missing per-job rows can be recovered.
    The completeness flag belongs to the latest/fullest sample-count snapshot.
    """
    grouped = {}
    for record in records:
        if record.get('schema') == 'ks-external-resource-aggregate-2':
            continue  # Its individual sessions are read from the session folder.
        grouped.setdefault(session_id(record), []).append(record)
    if not grouped:
        return None
    sessions, jobs, alerts = [], {}, set()
    for identity, snapshots in sorted(grouped.items()):
        latest = max(snapshots, key=lambda r: (r.get('samples',0),r.get('last_observed_utc','')))
        retained = {}
        for snapshot in snapshots:
            for key, value in snapshot.get('jobs',{}).items():
                old = retained.get(key,{})
                retained[key] = {**old, **value,
                    'first_seen_utc': min(old.get('first_seen_utc',value['first_seen_utc']),value['first_seen_utc']),
                    'last_seen_utc': max(old.get('last_seen_utc',value['last_seen_utc']),value['last_seen_utc']),
                    'peak_rss_bytes':max(old.get('peak_rss_bytes',0),value.get('peak_rss_bytes',0)),
                    'peak_hwm_bytes':max(old.get('peak_hwm_bytes',0),value.get('peak_hwm_bytes',0))}
            alerts.update(snapshot.get('alerts',[]))
        complete = bool(latest.get('job_observations_complete', 'jobs' in latest))
        one = {key:latest.get(key) for key in ('started_utc','last_observed_utc','status','samples',
            'combined_worker_rss_peak_bytes','disk_free_min_bytes','container_current_peak_bytes','active_workers')}
        one.update(session_id=identity, retained_job_rows=len(retained), job_observations_complete=complete,
            aggregate_provenance=latest.get('aggregate_provenance','observer snapshot'),
            per_job_provenance=latest.get('per_job_provenance','retained observer rows'),
            total_observed_jobs=len(retained) if complete else None)
        sessions.append(one)
        for key,value in retained.items():
            old=jobs.get(key,{})
            for field in ('model','track','stage'):
                if field in old and field in value and old[field] != value[field]:
                    raise ValueError('Observer snapshots disagree on a job identity')
            jobs[key]={**old, **value,
                'first_seen_utc':min(old.get('first_seen_utc',value['first_seen_utc']),value['first_seen_utc']),
                'last_seen_utc':max(old.get('last_seen_utc',value['last_seen_utc']),value['last_seen_utc']),
                'peak_rss_bytes':max(old.get('peak_rss_bytes',0),value.get('peak_rss_bytes',0)),
                'peak_hwm_bytes':max(old.get('peak_hwm_bytes',0),value.get('peak_hwm_bytes',0))}
    sessions.sort(key=lambda r:r['started_utc'])
    latest=max(sessions,key=lambda r:r.get('last_observed_utc') or r['started_utc'])
    gaps=[{'session_id':r['session_id'],'retained_job_rows':r['retained_job_rows'],
        'total_observed_jobs':None,'limitation':'Final per-job observation rows unavailable; aggregate extrema remain valid; use fit-receipt RUSAGE_SELF for missing worker peaks.'}
        for r in sessions if not r['job_observations_complete']]
    return {'schema':'ks-external-resource-aggregate-2','started_utc':min(r['started_utc'] for r in sessions),
        'last_observed_utc':latest['last_observed_utc'],'status':latest['status'],
        'active_workers':latest.get('active_workers'),'samples':sum(r.get('samples',0) for r in sessions),
        'combined_worker_rss_peak_bytes':max(r.get('combined_worker_rss_peak_bytes',0) for r in sessions),
        'container_current_peak_bytes':max(r.get('container_current_peak_bytes',0) for r in sessions),
        'disk_free_min_bytes':min(r['disk_free_min_bytes'] for r in sessions if r.get('disk_free_min_bytes') is not None),
        'jobs':jobs,'retained_job_rows':len(jobs),'sessions':sessions,'session_count':len(sessions),
        'coverage_gaps':gaps,'alerts':sorted(alerts),'original_attempt_rss_valid':False,
        'method':'exact absolute --job path matched in outer proc cmdline; same-entry VmRSS/VmHWM',
        'scope':'session aggregates merged without summing concurrent peaks; missing job peaks are not reconstructed'}


def read_session_records(results, read=None):
    read = read or (lambda path: json.loads(path.read_text()))
    records=[read(path) for path in sorted((results/'observer_sessions').glob('session_*.json'))]
    current=results/'external_resources.json'
    if current.exists():
        value=read(current)
        if value.get('schema') != 'ks-external-resource-aggregate-2':
            records.append(value)
    return records


def persist_session(results, summary):
    path=results/'observer_sessions'/('session_'+session_id(summary)+'.json')
    if path.exists():
        previous=json.loads(path.read_text())
        if previous.get('samples',0)>summary.get('samples',0):
            raise ValueError('Refuse to replace a fuller observer session with an older snapshot')
    atomic_json(path,summary)


def another_observer(results):
    outer_self=next(int(line.split()[1]) for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('Pid:'))
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==outer_self:
            continue
        try:
            args=[os.fsdecode(x) for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            if any(Path(x).name=='watch_ks_resources.py' for x in args) and '--results' in args:
                if Path(args[args.index('--results')+1]).resolve()==results:
                    return True
        except (OSError,ValueError,IndexError):
            continue
    return False


def processes(results):
    observed = []
    scheduler = False
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = [v.decode() for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
            if 'bayes_stesn.applications.ks_runner' not in args:
                continue
            if '--output' in args and args[args.index('--output')+1] == str(results):
                scheduler = True
            if 'worker' not in args or '--job' not in args:
                continue
            job_path = Path(args[args.index('--job')+1])
            if job_path.parent != results/'jobs':
                continue
            status = dict(line.split(':',1) for line in (proc/'status').read_text().splitlines() if ':' in line)
            job = json.loads(job_path.read_text())
            observed.append({'job_id':job['job_id'], 'model':job['model'], 'track':job['track'],
                'stage':job['stage'], 'outer_pid':int(proc.name),
                'namespace_pids':[int(v) for v in status.get('NSpid','').split()],
                'rss_bytes':int(status.get('VmRSS','0 kB').split()[0])*1024,
                'process_hwm_bytes':int(status.get('VmHWM','0 kB').split()[0])*1024})
        except (OSError, ValueError, KeyError, UnicodeDecodeError):
            continue
    return scheduler, observed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=2.0)
    args = parser.parse_args()
    results = args.results.resolve()
    ownership=(results/'observer.lock').open('a+')
    fcntl.flock(ownership,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if another_observer(results):
        raise RuntimeError('An observer is already running; do not start a duplicate')
    budget = json.loads((results/'experiment.json').read_text())['config']['budget']
    out = results/'external_resources.json'
    if out.exists():
        previous=json.loads(out.read_text())
        if previous.get('schema')!='ks-external-resource-aggregate-2':
            persist_session(results,previous)
    summary = {'schema':'ks-external-resource-session-2','experiment_identity':json.loads((results/'experiment.json').read_text())['identity'],
        'started_utc':utc(), 'status':'running', 'method':'exact absolute --job path matched in outer proc cmdline; same-entry VmRSS/VmHWM',
        'purpose':'read-only independent resource observation', 'jobs':{}, 'samples':0,
        'combined_worker_rss_peak_bytes':0, 'disk_free_min_bytes':None,
        'container_current_peak_bytes':0, 'alerts':[],
        'original_attempt_rss_valid':False,
        'original_attempt_rss_limitation':'nested Popen PID did not identify the mounted outer /proc entry',
        'job_observations_complete':True,
        'earlier_jobs':'use fit receipt RUSAGE_SELF high-water marks; external sampling started after campaign launch'}
    while True:
        scheduler, workers = processes(results)
        now = utc()
        for w in workers:
            key=w['job_id']; old=summary['jobs'].get(key,{})
            summary['jobs'][key]={**w,'first_seen_utc':old.get('first_seen_utc',now),'last_seen_utc':now,
                'peak_rss_bytes':max(old.get('peak_rss_bytes',0),w['rss_bytes']),
                'peak_hwm_bytes':max(old.get('peak_hwm_bytes',0),w['process_hwm_bytes'])}
            if w['process_hwm_bytes'] > budget['worker_cap_bytes']:
                alert='worker memory budget exceeded: '+key
                if alert not in summary['alerts']:summary['alerts'].append(alert)
        combined=sum(w['rss_bytes'] for w in workers)
        summary['combined_worker_rss_peak_bytes']=max(summary['combined_worker_rss_peak_bytes'],combined)
        if combined > budget['combined_worker_cap_bytes'] and 'combined worker memory budget exceeded' not in summary['alerts']:
            summary['alerts'].append('combined worker memory budget exceeded')
        free=shutil.disk_usage(results).free
        summary['disk_free_min_bytes']=min(summary['disk_free_min_bytes'] or free,free)
        if free < budget['disk_reserve_bytes'] and 'disk reserve breached' not in summary['alerts']:
            summary['alerts'].append('disk reserve breached')
        cgroup=int(Path('/sys/fs/cgroup/memory.current').read_text())
        summary['container_current_peak_bytes']=max(summary['container_current_peak_bytes'],cgroup)
        summary['last_observed_utc']=now;summary['samples']+=1
        summary['active_workers']=len(workers)
        summary['status']='running' if scheduler or workers else 'scheduler_finished'
        persist_session(results,summary)
        atomic_json(out,merge_sessions(read_session_records(results)))
        if not scheduler and not workers:break
        time.sleep(args.interval)
    print(json.dumps({k:v for k,v in summary.items() if k!='jobs'}))


if __name__=='__main__':main()
