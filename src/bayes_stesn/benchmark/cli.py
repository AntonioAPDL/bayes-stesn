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
    names=("smoke","pilot","main","v03_smoke","v03_pilot","v03_main")
    profiles.add_argument("name",choices=names);profiles.add_argument("--output",required=True)
    run=sub.add_parser("run",aliases=["resume"],help="Run/resume a chronology- and capability-checked benchmark")
    run.add_argument("--config");run.add_argument("--profile",choices=names,default="smoke")
    run.add_argument("--output",required=True);run.add_argument("--models");run.add_argument("--scenarios")
    run.add_argument("--tracks");run.add_argument("--replications",type=int)
    run.add_argument("--power-freeze",help="Required confirmation-stage development power receipt")
    run.add_argument("--workers",type=int,default=1,help="Isolated spawned processes, at most4 and bounded by host CPUs")
    run.add_argument("--max-memory-mb",type=float,help="Job-memory ceiling, clamped to 65%% of the enforced limit and current headroom")
    run.add_argument("--no-resume",action="store_true",help="Reject an existing run rather than resuming it")
    run.add_argument("--max-infrastructure-retries",type=int,default=2)
    run.add_argument("--max-jobs",type=int,help="Pause after this many submitted job attempts")
    run.add_argument("--max-wall-seconds",type=float,help="Pause between atomic jobs; never shorten a scientific fit")
    plan=sub.add_parser("plan",help="Inspect resolved fit counts and safe resource admission without fitting")
    plan.add_argument("--config");plan.add_argument("--profile",choices=names,default="v03_pilot")
    plan.add_argument("--workers",type=int,default=1);plan.add_argument("--max-memory-mb",type=float)
    plan.add_argument("--measurements",help="JSON mapping model IDs to representative fit wall seconds")
    plan.add_argument("--output")
    validate=sub.add_parser("validate",aliases=["resolve"],help="Validate and resolve effective candidates without fitting")
    validate.add_argument("--config",required=True);validate.add_argument("--output")
    status=sub.add_parser("status",help="Inspect terminal, interrupted and pending work")
    status.add_argument("run")
    freeze=sub.add_parser("freeze",help="Seal a confirmation design and runnable YAML using development evidence")
    freeze.add_argument("pilot");freeze.add_argument("--output",required=True)
    freeze.add_argument("--config",help="Draft confirmation YAML; defaults to the registered main profile")
    freeze.add_argument("--resolved-config",help="Output frozen YAML; defaults to the receipt path with .yaml suffix")
    freeze.add_argument("--fhn-validation",help="Versioned time-step evidence, required when confirmation includes G3")
    freeze.add_argument("--target-mcse",type=float,default=.02)
    freeze.add_argument("--planning-mode",choices=("legacy","paired_macro"))
    freeze.add_argument("--primary-comparisons",help="JSON file with the frozen primary comparison specifications")
    freeze.add_argument("--minimum-pairs",type=int,default=5)
    curve=sub.add_parser("curve",help="Run matched history or sensor learning curves")
    curve.add_argument("kind",choices=("history","sensors"));curve.add_argument("--config",required=True)
    curve.add_argument("--values",required=True);curve.add_argument("--output",required=True)
    curve.add_argument("--workers",type=int,default=1);curve.add_argument("--max-memory-mb",type=float)
    curve.add_argument("--common-endpoint",type=int,help="Shared initial validation endpoint for all history lengths")
    curve.add_argument("--execute-values",help="Comma-separated subset to execute after resolving the full curve")
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
    elif args.benchmark_command in {"run","resume"}:
        config=args.config
        if args.benchmark_command=="resume" and config is None:
            config=Path(args.output)/"resolved_protocol.yaml"
        p=BenchmarkProtocol.load(config) if config else profile(args.profile)
        changes={}
        for key in ("models","scenarios","tracks"):
            value=getattr(args,key)
            if value:
                changes[key]=tuple(x.strip() for x in value.split(",") if x.strip())
        if args.replications is not None:
            changes["replications"]=args.replications
        p=replace(p,**changes)
        result=run_benchmark(p,args.output,progress=progress,workers=args.workers,
                             power_freeze=args.power_freeze,max_memory_mb=args.max_memory_mb,
                             resume=not args.no_resume,max_infrastructure_retries=args.max_infrastructure_retries,
                             max_jobs=args.max_jobs,max_wall_seconds=args.max_wall_seconds)
        print(json.dumps({"output":str(Path(args.output).resolve()),"jobs":result["jobs"],
                          "scheduled_jobs":result["scheduled_jobs"],"complete":result["complete"],
                          "stage":result["stage"]},indent=2))
    elif args.benchmark_command=="plan":
        from .execution import plan_execution
        from .runner import write_json
        p=BenchmarkProtocol.load(args.config) if args.config else profile(args.profile)
        measured=None if args.measurements is None else json.loads(Path(args.measurements).read_text())
        result=plan_execution(p,workers=args.workers,max_memory_mb=args.max_memory_mb,
                              measured_fit_seconds=measured)
        if args.output:
            write_json(args.output,result)
        print(json.dumps(result,indent=2))
    elif args.benchmark_command in {"validate","resolve"}:
        from .protocol import candidate_manifest, effective_models
        from .runner import write_json
        p=BenchmarkProtocol.load(args.config)
        result={"protocol_identity":p.identity,"resolved_protocol":p.to_dict(),
                "candidates":{f"{model}/{track}":candidate_manifest(model,p,track=track,seed=p.master_seed)
                              for track in p.tracks for model in effective_models(p,track)}}
        if args.output:
            write_json(args.output,result)
        print(json.dumps(result,indent=2))
    elif args.benchmark_command=="status":
        from .runner import benchmark_status
        print(json.dumps(benchmark_status(args.run),indent=2))
    elif args.benchmark_command=="freeze":
        from .studies import freeze_confirmation
        draft=BenchmarkProtocol.load(args.config) if args.config else profile("main")
        receipt=freeze_confirmation(args.pilot,args.output,protocol=draft,
                    fhn_validation=args.fhn_validation,resolved_config_output=args.resolved_config,
                    target_mcse=args.target_mcse,planning_mode=args.planning_mode,
                    primary_comparisons=None if args.primary_comparisons is None else
                        json.loads(Path(args.primary_comparisons).read_text()),
                    minimum_pairs=args.minimum_pairs)
        print(json.dumps({"identity":receipt["identity"],"R":receipt["power"]["R"],
                          "power_status":receipt["power_status"],"receipt":str(Path(args.output).resolve()),
                          "config":str(Path(args.resolved_config or Path(args.output).with_suffix('.yaml')).resolve())},indent=2))
    elif args.benchmark_command=="curve":
        from .studies import run_curve
        run_curve(BenchmarkProtocol.load(args.config),args.output,kind=args.kind,
                  values=[int(v) for v in args.values.split(",")],progress=progress,
                  workers=args.workers,max_memory_mb=args.max_memory_mb,
                  common_endpoint=args.common_endpoint,
                  execute_values=None if args.execute_values is None else
                      [int(v) for v in args.execute_values.split(",")])
    elif args.benchmark_command=="calibrate":
        from .studies import run_calibration_study
        run_calibration_study(args.run,args.output,scenario=args.scenario,model_id=args.model,
            replication=args.replication,track=args.track,blocks=args.blocks,progress=progress)
    elif args.benchmark_command=="report":
        from .report import create_report
        print(json.dumps(create_report(args.run,args.output,animations=not args.no_animations),indent=2))
    return 0
