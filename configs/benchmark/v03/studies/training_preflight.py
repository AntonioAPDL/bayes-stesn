#!/usr/bin/env python3
"""One isolated full-budget training-only timing case; no future rows or scores.

Run once per case/model/track in a fresh process. Interrupted/capped cases are
not used to derive prospective elapsed-time allowances. This tool never changes
model defaults or promotes a configuration based on a forecast outcome.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import yaml

from bayes_stesn.benchmark.contracts import select_panel
from bayes_stesn.benchmark.execution import estimate_job_memory, memory_allowance
from bayes_stesn.benchmark.generators import generate
from bayes_stesn.benchmark.protocol import BenchmarkProtocol, candidate_manifest, effective_models
from bayes_stesn.benchmark.registry import MODELS, fit_model
from bayes_stesn.benchmark.runner import (_fit_time_limit, _peak_rss, FitTimeLimitExceeded,
    benchmark_environment, dgp_seed_for, jsonable, single_thread_runtime, source_fingerprint, write_json)
from bayes_stesn.io import content_hash, seed_for


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--case", choices=("sensors_100",), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--track", choices=("A", "B"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-memory-mb", type=float, default=5120)
    args = parser.parse_args(argv)
    recipe = yaml.safe_load(Path(__file__).with_name("remaining_studies.yaml").read_text())
    profile_path = Path(args.profiles)/(args.case+".draft.yaml")
    p = BenchmarkProtocol.load(profile_path)
    if args.model not in effective_models(p, args.track):
        raise ValueError("Model/track is not part of the declared preflight roster")
    scenario = recipe["preflight"]["scenario"]
    master = recipe["preflight"]["master_seed"]
    seed = seed_for(master, "training_only_study_preflight", args.case, args.model, args.track)
    p_seed = replace(p, master_seed=master)
    config = candidate_manifest(args.model, p, track=args.track, fitting_sites=p.n_observed,
        fitting_rows=p.final_origin-p.training_start)["candidates"][0]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.case}__{args.track}__{args.model}"
    receipt_path = output/(stem+".json")
    fit_directory = output/(stem+"_fit")
    identity_input = {"profile_identity": p.identity, "model_id": args.model, "track": args.track,
        "master_seed": master, "fit_seed": seed, "source_fingerprint": source_fingerprint(),
        "config_hash": content_hash(config), "case": args.case,
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "recipe_sha256": hashlib.sha256(Path(__file__).with_name("remaining_studies.yaml").read_bytes()).hexdigest(),
        "elapsed_allowance_seconds": recipe["preflight"]["elapsed_allowance_seconds_per_fresh_process"]}
    preflight_identity = content_hash(identity_input)
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        if (previous.get("identity") != content_hash({k: v for k, v in previous.items() if k != "identity"})
                or previous.get("preflight_identity") != preflight_identity):
            raise ValueError("Existing preflight receipt has another identity")
        if previous["execution_status"] != "completed":
            raise ValueError("Retain the failed preflight; use a separately declared fresh measurement directory")
        print(json.dumps({"cached_measurement": str(receipt_path), "fit_seconds": previous["fit_seconds"]}))
        return 0
    if fit_directory.exists():
        raise ValueError("Interrupted preflight work exists; exact resume would not measure a fresh complete fit")
    with single_thread_runtime():
        estimate = estimate_job_memory(p, args.model, args.track, seed)
        allowance = memory_allowance(args.max_memory_mb)
        budget = allowance["effective_job_budget_bytes"]
        if budget is not None and estimate["estimated_bytes"] > budget:
            raise MemoryError("Training-only preflight exceeds current estimated memory admission")
        # No observation beyond the parameter-fitting cutoff is generated.
        sim = generate(scenario, p.final_origin, n_side=p.n_side, n_observed=p.n_observed,
            n_calibration=p.n_calibration, seed=dgp_seed_for(p_seed, scenario, 0),
            **(p.generator_options or {}).get(scenario, {}))
        panel = select_panel(sim.observed_panel(), start=p.training_start)
        initial = {**identity_input, "preflight_identity": preflight_identity,
            "schema": "v03-training-only-preflight-1.0", "forecast_computed": False,
            "final_test_rows_generated": 0, "scenarios": [scenario],
            "generated_rows": p.final_origin, "supplied_training_rows": len(panel.y),
            "data_hash": content_hash({"y": panel.y, "coordinates": panel.coordinates,
                                      "times": panel.times, "site_ids": panel.site_ids}),
            "environment": benchmark_environment(), "memory_admission": allowance,
            "estimated_job_memory": estimate,
            "measurement_scope": "fresh process; one full fitting invocation including diagnostics; no forecast or scoring"}
        fit_directory.mkdir()
        start, cpu = time.perf_counter(), time.process_time()
        try:
            kwargs = {"checkpoint_path": fit_directory/"chain.npz"} if MODELS[args.model].family == "bayesian" else {}
            with _fit_time_limit(identity_input["elapsed_allowance_seconds"]):
                fitted = fit_model(args.model, panel, config, seed, **kwargs)
            row = {**initial, "execution_status": "completed", "method_qualified": bool(fitted.diagnostics.get("qualified", True)),
                   "diagnostics": jsonable(fitted.diagnostics)}
        except Exception as error:
            row = {**initial, "execution_status": "capped" if isinstance(error, FitTimeLimitExceeded) else "failed",
                   "method_qualified": None, "error": f"{type(error).__name__}: {error}"}
        row.update(fit_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu,
                   fresh_process_peak_rss_bytes=_peak_rss())
        if source_fingerprint() != initial["source_fingerprint"]:
            raise RuntimeError("Package changed during the training-only preflight")
        row = jsonable(row)
        row["identity"] = content_hash(row)
        write_json(receipt_path, row)
    print(json.dumps({k: row[k] for k in ("case", "model_id", "track", "execution_status", "method_qualified", "fit_seconds")}, indent=2))
    return 0 if row["execution_status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
