"""Bounded development comparison of two exact ESN sampling algorithms.

This command fits observed history only and never calculates final test
scores. The default is an intentionally short execution check, not evidence
of convergence. Use --profile pilot (or a development --config) for a
prespecified scientific comparison, and --esn-config to reuse an archived
selected configuration. Both algorithms share the exact feature map/target.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from bayes_stesn.benchmark.contracts import select_panel
from bayes_stesn.benchmark.esn import fit_esn
from bayes_stesn.benchmark.generators import ALIASES, generate
from bayes_stesn.benchmark.protocol import BenchmarkProtocol, profile
from bayes_stesn.benchmark.runner import (benchmark_environment, dgp_seed_for, jsonable,
    memory_estimate, _memory_budget, single_thread_runtime, source_fingerprint, write_json)
from bayes_stesn.io import content_hash, seed_for


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--profile", default="smoke", choices=("smoke", "pilot", "v03_pilot"))
    command.add_argument("--config", help="Existing development protocol YAML; overrides --profile")
    command.add_argument("--esn-config", help="JSON ESN settings, or a selection receipt with a selected mapping")
    command.add_argument("--scenarios", default="g1,g3,g4")
    command.add_argument("--replications", default="0", help="Prespecified comma-separated dataset indices")
    command.add_argument("--output", required=True)
    command.add_argument("--warmup", type=int)
    command.add_argument("--draws", type=int)
    command.add_argument("--continuation", type=int)
    command.add_argument("--chains", type=int)
    command.add_argument("--max-cases", type=int, default=6)
    command.add_argument("--max-memory-mb", type=float)
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    protocol = BenchmarkProtocol.load(args.config) if args.config else profile(args.profile)
    if protocol.stage == "confirmation":
        raise ValueError("Sampler development cannot consume a confirmation protocol or final seed namespace")
    scenarios = tuple(ALIASES.get(name.strip(), name.strip()) for name in args.scenarios.split(","))
    replications = tuple(int(value) for value in args.replications.split(","))
    if (not scenarios or not replications or any(r < 0 for r in replications)
            or len(set(scenarios)) != len(scenarios) or len(set(replications)) != len(replications)
            or len(scenarios)*len(replications) > args.max_cases):
        raise ValueError("Use distinct nonnegative replications and a case count within --max-cases")
    if args.esn_config:
        config = json.loads(Path(args.esn_config).read_text())
        config = config.get("selected", config)
        if not isinstance(config, dict):
            raise ValueError("--esn-config must contain an ESN configuration mapping")
    else:
        config = {"features":{"units":[32], "squares":16, "lags":[1, 2]},
                  "spatial_mean":{"side":3, "width":.3}}
    sampler = dict(config.get("sampler", {}))
    sampler.update(chains=protocol.mcmc_chains if args.chains is None else args.chains,
                   warmup=protocol.mcmc_warmup if args.warmup is None else args.warmup,
                   draws=protocol.mcmc_draws if args.draws is None else args.draws)
    config["sampler"] = sampler
    config["continuation_draws"] = protocol.mcmc_continuation if args.continuation is None else args.continuation
    config["diagnostics"] = {"functional_draws":protocol.mcmc_functional_draws,
                             "min_chains":4, "rhat_limit":1.01, "min_ess":400}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with single_thread_runtime():
        source = source_fingerprint()
        manifest = {"schema":"sampler-coordinate-development-1.0", "protocol":protocol.to_dict(),
            "config":config, "scenarios":scenarios, "replications":replications,
            "source_fingerprint":source, "environment":benchmark_environment(),
            "tool_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "stage":protocol.stage, "final_scores_computed":False,
            "seed_policy":"Prespecified common reservoir/chain seeds; different transition kernels, identical target",
            "scope":"algorithm development; smoke runs are execution checks only"}
        manifest = jsonable(manifest)
        manifest["identity"] = content_hash(manifest)
        path = output/"manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Existing comparison has a different source, runtime, dataset or fitting design")
        write_json(path, manifest)
        estimate, budget = memory_estimate(protocol, workers=1), _memory_budget(args.max_memory_mb)
        if budget is not None and estimate["estimated_total_bytes"] > budget:
            raise MemoryError("The comparison profile exceeds the current admitted memory budget")
        for scenario in scenarios:
            for replication in replications:
                if source_fingerprint() != source:
                    raise RuntimeError("Source changed during sampler development")
                # Generate only the fitting prefix; no final test labels are
                # needed even by this evaluator-side development command.
                simulation = generate(scenario, protocol.final_origin, n_side=protocol.n_side,
                    n_observed=protocol.n_observed, n_calibration=protocol.n_calibration,
                    seed=dgp_seed_for(protocol, scenario, replication),
                    **(protocol.generator_options or {}).get(scenario, {}))
                panel = select_panel(simulation.observed_panel(), start=protocol.training_start)
                data_identity = content_hash({"y":panel.y, "coordinates":panel.coordinates,
                                              "times":panel.times, "sites":panel.site_ids})
                fit_seed = seed_for(protocol.master_seed, "coordinate_comparison", scenario, replication)
                target_identities = []
                for policy in ("legacy", "ridge_total_scale"):
                    directory = output/scenario/f"rep_{replication:03d}"/policy
                    directory.mkdir(parents=True, exist_ok=True)
                    receipt_path = directory/"receipt.json"
                    if receipt_path.exists():
                        row = json.loads(receipt_path.read_text())
                        if row.get("manifest_identity") != manifest["identity"] or row.get("data_identity") != data_identity:
                            raise ValueError("Saved comparison receipt does not match its observed data/design")
                        rows.append(row)
                        if "target_identity" in row:
                            target_identities.append(row["target_identity"])
                        continue
                    settings = deepcopy(config)
                    settings["sampler"]["parameterization"] = policy
                    started = time.perf_counter()
                    row = {"manifest_identity":manifest["identity"], "scenario":scenario,
                        "replication":replication, "parameterization":policy,
                        "data_identity":data_identity, "fit_seed":fit_seed, "configuration":settings}
                    try:
                        fitted = fit_esn(panel, settings, fit_seed, checkpoint_path=directory/"chain.npz")
                        elapsed = time.perf_counter()-started
                        gate = fitted.diagnostics["gates"][-1]
                        scalars = [entry for group in ("hyperparameters", "functionals")
                                   for entry in gate[group]["parameters"].values()]
                        minimum = min(float(entry[key]) for entry in scalars
                                      for key in ("ess_bulk", "ess_tail") if entry[key] is not None)
                        # A resumed partial fit lacks the wall time lost before
                        # interruption. Do not turn that into a speedup claim.
                        fresh_timing = not fitted.diagnostics["recovery"]["resumed"]
                        row.update(qualified=bool(fitted.diagnostics["qualified"]), complete=True,
                            target_identity=fitted.fitted.prepared.identity,
                            invocation_seconds=elapsed, timing_complete=fresh_timing,
                            minimum_scientific_ess=minimum,
                            minimum_ess_per_second=minimum/elapsed if fresh_timing else None,
                            diagnostics=fitted.diagnostics, sampling=fitted.fitted.posterior.metadata)
                        target_identities.append(row["target_identity"])
                    except Exception as exc:
                        row.update(complete=False, qualified=False, error=f"{type(exc).__name__}: {exc}",
                                   invocation_seconds=time.perf_counter()-started,
                                   meaning="Uncompleted development fit, not a scientific operational fallback")
                    if source_fingerprint() != source:
                        raise RuntimeError("Source changed during sampler development; results require review")
                    write_json(receipt_path, row)
                    rows.append(jsonable(row))
                    print(json.dumps({key:row.get(key) for key in
                        ("scenario", "replication", "parameterization", "complete", "qualified", "invocation_seconds")}), flush=True)
                if len(target_identities) == 2 and len(set(target_identities)) != 1:
                    raise AssertionError("Coordinate comparison changed the statistical target")
        summary = {"manifest_identity":manifest["identity"], "rows":rows,
                   "complete":all(row["complete"] for row in rows),
                   "minimum_ess_scope":"Same original hyperparameters and recovered scientific functionals",
                   "accuracy_claim":False, "test_forecasts_computed":False}
        write_json(output/"summary.json", summary)
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
