#!/usr/bin/env python3
"""Run/resume the KS campaign in a persistent terminal; inspect it elsewhere.

Use an execution service's retained TTY session, tmux, or an authorized batch
scheduler to keep ``run`` alive. A detached process is not claimed to survive
destruction of the terminal's container/PID namespace.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONHASHSEED", "0")

from bayes_stesn.applications.ks_fullfield_runner import (
    MODULE, process_identity, process_matches, read_json, summarize_output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "resume", "status", "stop", "report"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--cache")
    parser.add_argument("--stage", default="all", choices=("pilot", "preflight", "screen",
        "promote", "precision", "freeze", "confirm", "sensitivity", "deferred", "all"))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.command == "status":
        summarize_output(output); return
    if args.command == "report":
        from bayes_stesn.applications.ks_fullfield_report import report
        print(json.dumps(report(output), indent=2)); return
    if args.command == "stop":
        records = []
        coordinator = output/"coordinator.json"
        if coordinator.exists(): records.append(read_json(coordinator).get("process"))
        records += [read_json(p).get("process") for p in output.glob("**/attempts/*.json")
                    if read_json(p).get("state") == "running"]
        live = {r["pid"]: r for r in records if process_matches(r)}
        own = process_identity()
        if any(r["pid_namespace"] != own["pid_namespace"] for r in live.values()):
            parser.error("The worker belongs to another retained execution/PID namespace. "
                "Interrupt its original terminal session (Ctrl-C) or use its authorized scheduler. "
                "No unrelated local PID has been signalled.")
        for record in live.values():
            if process_matches(record): os.kill(record["namespace_pid"], signal.SIGTERM)
        print(json.dumps({"stop_requested_for_visible_pids": sorted(live)})); return
    if not args.config or not args.cache:
        parser.error("run/resume require --config and --cache; use the same immutable configuration and output")
    config, cache = Path(args.config).resolve(), Path(args.cache).resolve()
    if not config.is_file() or not cache.is_dir(): parser.error("Configuration/cache path does not exist")
    os.environ["PYTHONPATH"] = str(ROOT/"src") + (os.pathsep+os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    # Keep this process attached to its durable terminal. The coordinator owns
    # its exclusive lock and checkpoints; restart is the exact same command.
    os.execv(sys.executable, [sys.executable, "-m", MODULE, args.stage,
        "--config", str(config), "--cache", str(cache), "--output", str(output)])


if __name__ == "__main__":
    main()
