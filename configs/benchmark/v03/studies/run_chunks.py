#!/usr/bin/env python3
"""One-worker benchmark chunks with an external disk-reserve guard.

This changes execution scheduling only. Scientific YAML, seeds, fitting caps,
and old receipts are never edited. Exit75 means an honest resource/no-progress
pause. A concurrent writer can still exhaust disk during an atomic job.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from bayes_stesn.benchmark.runner import source_fingerprint


def admission(path, reserve_gib, next_output_mib):
    free = shutil.disk_usage(path).free
    required = int(reserve_gib*1024**3+next_output_mib*1024**2)
    return {"free_bytes": free, "reserve_bytes": int(reserve_gib*1024**3),
            "next_output_and_temporary_allowance_bytes": int(next_output_mib*1024**2),
            "admitted": free >= required}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Benchmark run, or guard-log directory for --command")
    parser.add_argument("--config")
    parser.add_argument("--max-invocations", type=int, default=20)
    parser.add_argument("--jobs-per-invocation", type=int, default=1)
    parser.add_argument("--max-wall-seconds", type=float, default=600)
    parser.add_argument("--max-memory-mb", type=float, default=5120)
    parser.add_argument("--reserve-gib", type=float, default=4)
    parser.add_argument("--next-output-mib", type=float, default=512)
    parser.add_argument("--command", nargs=argparse.REMAINDER,
                        help="One explicit non-benchmark command; invoke in a fresh guarded subprocess")
    args = parser.parse_args(argv)
    if (args.reserve_gib < 4 or args.next_output_mib <= 0 or args.max_invocations < 1
            or args.jobs_per_invocation < 1 or args.max_wall_seconds <= 0):
        raise ValueError("Reserve must be at least4GiB and all execution allowances positive")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    guards = output.parent/"resource_guards"
    guards.mkdir(exist_ok=True)
    source = source_fingerprint()
    stamp = f"{output.name}_{time.time_ns()}"
    journal = guards/(stamp+".jsonl")

    def record(row):
        with journal.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True)+"\n")
        print(json.dumps(row, sort_keys=True), flush=True)

    for index in range(1 if args.command else args.max_invocations):
        if source_fingerprint() != source:
            raise RuntimeError("Package source changed during the chunk controller")
        summary_path = output/"summary.json"
        previous = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        # A summary flag is not an integrity receipt. Even completed campaigns
        # go through one bounded core resume, which verifies every source,
        # manifest and terminal artifact before this wrapper accepts success.
        budget = admission(output.parent, args.reserve_gib, args.next_output_mib*args.jobs_per_invocation)
        if not budget["admitted"]:
            record({"event": "paused_disk_headroom", "source_fingerprint": source, **budget})
            return 75
        command = args.command or [sys.executable, "-m", "bayes_stesn", "benchmark", "resume",
            "--output", str(output), "--workers", "1", "--max-memory-mb", str(args.max_memory_mb),
            "--max-jobs", str(args.jobs_per_invocation), "--max-wall-seconds", str(args.max_wall_seconds)]
        if args.config and not args.command:
            command += ["--config", args.config]
        log = guards/(stamp+f"_{index:03d}.log")
        record({"event": "starting_guarded_invocation", "index": index,
                "source_fingerprint": source, "command": command, "stdout_stderr": str(log),
                "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **budget})
        start = time.perf_counter()
        with log.open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        elapsed = time.perf_counter()-start
        current = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        record({"event": "guarded_invocation_finished", "index": index, "exit_code": result.returncode,
                "wall_seconds": elapsed, "jobs": current.get("jobs"), "complete": current.get("complete"),
                "free_bytes_after": shutil.disk_usage(output.parent).free, "stdout_stderr": str(log)})
        if result.returncode:
            return result.returncode
        if args.command or current.get("complete"):
            return 0
        if current.get("jobs", 0) <= previous.get("jobs", 0):
            record({"event": "paused_without_terminal_progress", "run": str(output),
                    "meaning": "Inspect retained state/resources; no fallback or completion invented"})
            return 75
    record({"event": "paused_at_invocation_limit", "run": str(output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
