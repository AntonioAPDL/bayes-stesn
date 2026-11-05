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
import json
import os
from pathlib import Path
import shutil
import time


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


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
    budget = json.loads((results/'experiment.json').read_text())['config']['budget']
    out = results/'external_resources.json'
    summary = {'started_utc':utc(), 'status':'running', 'method':'exact absolute --job path matched in outer proc cmdline; same-entry VmRSS/VmHWM',
        'purpose':'read-only independent resource observation', 'jobs':{}, 'samples':0,
        'combined_worker_rss_peak_bytes':0, 'disk_free_min_bytes':None,
        'container_current_peak_bytes':0, 'alerts':[],
        'original_attempt_rss_valid':False,
        'original_attempt_rss_limitation':'nested Popen PID did not identify the mounted outer /proc entry',
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
        temporary=out.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');os.replace(temporary,out)
        if not scheduler and not workers:break
        time.sleep(args.interval)
    print(json.dumps({k:v for k,v in summary.items() if k!='jobs'}))


if __name__=='__main__':main()
