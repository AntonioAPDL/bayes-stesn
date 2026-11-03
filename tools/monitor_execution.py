#!/usr/bin/env python3
"""Sample shared cgroup use and disk availability without changing execution.

The cgroup can contain unrelated activity. Process namespaces can hide model
workers, so process RSS is deliberately not used as a campaign memory measure.
This is an observed resource envelope, not an isolated per-model allocation peak.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import signal
import time


def read_integer(path):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def sample(project):
    memory = {}
    try:
        memory = {k: int(v) for k, v in
                  (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
    except (OSError, ValueError):
        pass
    return {'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'shared_cgroup_bytes': read_integer('/sys/fs/cgroup/memory.current'),
            'shared_cgroup_anon_bytes': memory.get('anon'),
            'shared_cgroup_file_bytes': memory.get('file'),
            'available_disk_bytes': shutil.disk_usage(project).free}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--interval-seconds', type=float, default=2)
    parser.add_argument('--stop-file', help='Stop after this file appears; never removed.')
    args = parser.parse_args()
    if args.interval_seconds < 1:
        parser.error('Sampling interval must be at least one second.')
    project = Path(args.project).resolve(strict=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error('Use a new output path; existing measurements are immutable.')
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    count = 0
    peaks = {}
    disk_minimum = None
    with output.open('x') as stream:
        while not stopping and not (args.stop_file and Path(args.stop_file).exists()):
            record = sample(project)
            stream.write(json.dumps(record, sort_keys=True) + '\n')
            stream.flush()
            count += 1
            for key, value in record.items():
                if key.startswith('shared_cgroup_') and value is not None:
                    peaks[key] = max(peaks.get(key, 0), value)
            value = record['available_disk_bytes']
            disk_minimum = value if disk_minimum is None else min(disk_minimum, value)
            time.sleep(args.interval_seconds)
    summary = {'schema': 'shared-resource-sampling-2.0', 'started_utc': started,
               'ended_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
               'project': str(project), 'samples': count, 'peaks': peaks,
               'minimum_available_disk_bytes': disk_minimum,
               'interval_seconds': args.interval_seconds,
               'scope': 'Observed interval only. Shared cgroup includes unrelated activity; '
                        'process RSS is unavailable across worker namespaces. '
                        'Sampling can miss between-sample peaks.'}
    output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
