"""Benchmark subcommands, separate from the original model fit/forecast CLI."""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path


def add_parser(subparsers):
    parent=subparsers.add_parser("benchmark",help="Run, inspect, and reproduce registered model comparisons")
    sub=parent.add_subparsers(dest="benchmark_command",required=True)
    sub.add_parser("registry",help="List implemented native task/uncertainty capabilities")
    profiles=sub.add_parser("profile",help="Write a resolved experiment profile")
    profiles.add_argument("name",choices=("smoke","pilot","main"));profiles.add_argument("--output",required=True)
    run=sub.add_parser("run",help="Run/resume a chronology- and capability-checked benchmark")
    run.add_argument("--config");run.add_argument("--profile",choices=("smoke","pilot","main"),default="smoke")
    run.add_argument("--output",required=True);run.add_argument("--models");run.add_argument("--scenarios")
    run.add_argument("--tracks");run.add_argument("--replications",type=int)
    run.add_argument("--power-freeze",help="Required confirmation-stage development power receipt")
    run.add_argument("--workers",type=int,default=1,help="Isolated spawned processes, at most4 and bounded by host CPUs")
    run.add_argument("--max-memory-mb",type=float,help="Explicit total planning budget; default80%% of host/container limit")
    freeze=sub.add_parser("freeze",help="Seal a confirmation design and runnable YAML using development evidence")
    freeze.add_argument("pilot");freeze.add_argument("--output",required=True)
    freeze.add_argument("--config",help="Draft confirmation YAML; defaults to the registered main profile")
    freeze.add_argument("--resolved-config",help="Output frozen YAML; defaults to the receipt path with .yaml suffix")
    freeze.add_argument("--fhn-validation",help="Versioned time-step evidence, required when confirmation includes G3")
    freeze.add_argument("--target-mcse",type=float,default=.02)
    curve=sub.add_parser("curve",help="Run matched history or sensor learning curves")
    curve.add_argument("kind",choices=("history","sensors"));curve.add_argument("--config",required=True)
    curve.add_argument("--values",required=True);curve.add_argument("--output",required=True)
    curve.add_argument("--workers",type=int,default=1);curve.add_argument("--max-memory-mb",type=float)
    calibration=sub.add_parser("calibrate",help="Compare raw/calibrated forecasts at the same later origin")
    calibration.add_argument("run");calibration.add_argument("--scenario",required=True)
    calibration.add_argument("--model",required=True);calibration.add_argument("--track",choices=("A","B"),default="A")
    calibration.add_argument("--replication",type=int,default=0);calibration.add_argument("--blocks",type=int,default=25)
    calibration.add_argument("--output",required=True)
    report=sub.add_parser("report",help="Render verified plots, synchronized animations and an HTML report")
    report.add_argument("run");report.add_argument("--output",required=True)
    report.add_argument("--no-animations",action="store_true")
    return parent


def main(args):
    from .protocol import profile,BenchmarkProtocol
    from .runner import run_benchmark,jsonable
    def progress(event):
        print(json.dumps(jsonable(event),sort_keys=True),flush=True)
    if args.benchmark_command=="registry":
        from .registry import model_registry
        print(json.dumps(model_registry(),indent=2))
    elif args.benchmark_command=="profile":
        profile(args.name).save(args.output);print(str(Path(args.output).resolve()))
    elif args.benchmark_command=="run":
        p=BenchmarkProtocol.load(args.config) if args.config else profile(args.profile)
        changes={}
        for key in ("models","scenarios","tracks"):
            value=getattr(args,key)
            if value:
                changes[key]=tuple(x.strip() for x in value.split(",") if x.strip())
        if args.replications is not None:
            changes["replications"]=args.replications
        p=replace(p,**changes)
        result=run_benchmark(p,args.output,progress=progress,workers=args.workers,
                             power_freeze=args.power_freeze,max_memory_mb=args.max_memory_mb)
        print(json.dumps({"output":str(Path(args.output).resolve()),"jobs":result["jobs"],"stage":result["stage"]},indent=2))
    elif args.benchmark_command=="freeze":
        from .studies import freeze_confirmation
        draft=BenchmarkProtocol.load(args.config) if args.config else profile("main")
        receipt=freeze_confirmation(args.pilot,args.output,protocol=draft,
                    fhn_validation=args.fhn_validation,resolved_config_output=args.resolved_config,
                    target_mcse=args.target_mcse)
        print(json.dumps({"identity":receipt["identity"],"R":receipt["power"]["R"],
                          "power_status":receipt["power_status"],"receipt":str(Path(args.output).resolve()),
                          "config":str(Path(args.resolved_config or Path(args.output).with_suffix('.yaml')).resolve())},indent=2))
    elif args.benchmark_command=="curve":
        from .studies import run_curve
        run_curve(BenchmarkProtocol.load(args.config),args.output,kind=args.kind,
                  values=[int(v) for v in args.values.split(",")],progress=progress,
                  workers=args.workers,max_memory_mb=args.max_memory_mb)
    elif args.benchmark_command=="calibrate":
        from .studies import run_calibration_study
        run_calibration_study(args.run,args.output,scenario=args.scenario,model_id=args.model,
            replication=args.replication,track=args.track,blocks=args.blocks,progress=progress)
    elif args.benchmark_command=="report":
        from .report import create_report
        print(json.dumps(create_report(args.run,args.output,animations=not args.no_animations),indent=2))
    return 0
