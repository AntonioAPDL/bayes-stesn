"""Atomic KS job ledger and conservative container-aware admission checks."""
from __future__ import annotations
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import tempfile
import time

import numpy as np


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False)+"\n"
    fd, name = tempfile.mkstemp(prefix='.'+p.name+'-', suffix='.tmp', dir=p.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            out.write(data); out.flush(); os.fsync(out.fileno())
        os.replace(name, p)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def file_hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def resource_snapshot(path):
    c = Path('/sys/fs/cgroup'); mem = {}
    for name in ('memory.max', 'memory.current', 'memory.peak', 'cpu.max'):
        f = c/name
        mem[name] = f.read_text().strip() if f.exists() else None
    if (c/'memory.stat').exists():
        d = dict(line.split() for line in (c/'memory.stat').read_text().splitlines())
        mem['inactive_file_bytes'] = int(d.get('inactive_file', 0))
        mem['anon_bytes'] = int(d.get('anon', 0))
    disk = shutil.disk_usage(path)
    v = os.statvfs(path)
    return {'utc': now(), 'memory': mem, 'disk_free_bytes': disk.free,
            'inode_available': v.f_favail, 'affinity_cpus': len(os.sched_getaffinity(0)),
            'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}


def admission(path, *, reserve_bytes, estimated_peak_bytes=1024**3, worker_cap_bytes=int(3.8*1024**3)):
    r = resource_snapshot(path)
    if r['disk_free_bytes'] < reserve_bytes:
        raise RuntimeError('Disk reserve gate: stop scheduling; preserve all completed jobs')
    m = r['memory']
    if m.get('memory.max') not in (None, 'max'):
        raw_headroom = int(m['memory.max'])-int(m['memory.current'])
        # Only half of inactive file cache is credited as reclaimable. Host RAM
        # is never substituted for the cgroup limit or this job's headroom.
        headroom = raw_headroom + int(.5*m.get('inactive_file_bytes', 0))
        allowance = min(worker_cap_bytes, int(.65*max(0, headroom)))
        r['raw_container_headroom_bytes'] = raw_headroom
        r['headroom_with_half_inactive_cache_bytes'] = headroom
        r['current_worker_allowance_bytes'] = allowance
        if 1.5*estimated_peak_bytes > allowance:
            raise RuntimeError(f'Memory admission gate: 1.5*pilot RSS={1.5*estimated_peak_bytes:g} exceeds {allowance}')
    return r


class Ledger:
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)

    def path(self, job_id):
        return self.root/(job_id+'.json')

    def read(self, job_id):
        p = self.path(job_id)
        return json.loads(p.read_text()) if p.exists() else None

    def write(self, job_id, state, **values):
        if state not in {'pending', 'running', 'complete', 'failed', 'inference-unqualified'}:
            raise ValueError('Unknown ledger state')
        old = self.read(job_id) or {}
        atomic_json(self.path(job_id), {**old, **values, 'job_id': job_id, 'state': state, 'updated_utc': now()})
