"""Reduce four completed P5 studies into a fresh, verified findings receipt.

No fitting or forecasting occurs. Numerical findings are reconstructed only
from the supplied current-source outputs and reference receipts. The original
historical findings are read solely to verify the prospective rule's two
timestamped original path invocations and original dataset identities.

Example, from the repository root with the locked one-thread environment:
  PYTHONPATH=src python tools/summarize_extensions_v03.py \
    --reference-run results/v03/development/reference_final \
    --g4-ar results/v03/extensions/final_ar_g4_existing_fronts \
    --g3-ar results/v03/extensions/final_ar_g3_excitable_field \
    --g4-paths results/v03/extensions/final_paths_g4_existing_fronts \
    --g3-paths results/v03/extensions/final_paths_g3_excitable_field \
    --output results/v03/extension_findings_reproduced.json

The immutable scope, prospective-rule document and historical timing evidence
have repository defaults. Optional --compare-findings verifies scientific
fields against a sealed receipt; it is never used to construct the result.
Sibling DIRECTORY_resources.json files are optional measured CLI receipts.
Without all four, total CLI resources remain unavailable, not estimated.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path

from bayes_stesn.benchmark.runner import benchmark_environment, jsonable, single_thread_runtime, source_fingerprint, write_json
from bayes_stesn.io import content_hash
from evaluate_extensions_v03 import _verify_terminal_inventory


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("g4_existing_fronts", "g3_excitable_field")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


class Inputs:
    def __init__(self):
        self.hashes = {}

    def bind(self, path):
        digest = sha(path)
        self.hashes[label(path)] = digest
        return digest

    def read(self, path, identity=False):
        self.bind(path)
        value = json.loads(Path(path).read_text())
        if identity:
            require(value.get("identity") == content_hash({k:v for k,v in value.items() if k != "identity"}),
                    f"Content identity differs: {path}")
        return value

    def verify(self):
        for path, digest in self.hashes.items():
            require(sha(ROOT/path if not Path(path).is_absolute() else path) == digest,
                    f"An aggregation input changed: {path}")


def compact_gate(gate):
    entries = {f"{group}.{name}": value for group in ("hyperparameters", "functionals")
               for name, value in gate[group]["parameters"].items()}
    return {"passed": gate["passed"], "chains": gate["hyperparameters"]["chains"],
            "retained_per_chain": gate["hyperparameters"]["draws"],
            "maximum_rhat": max(row["rhat"] for row in entries.values()),
            "minimum_bulk_ess": min(row["ess_bulk"] for row in entries.values()),
            "minimum_tail_ess": min(row["ess_tail"] for row in entries.values()),
            "failed_parameters": {key:value for key,value in entries.items() if not value["passed"]}}


def verify_original_rule(rule_document, evidence, rule_sha):
    rule = rule_document["predictive_path_count_rule"]
    assessment = evidence["path_count_assessment"]
    require(assessment["predeclared_rule"] == rule and assessment["predeclared_rule_document_sha256"] == rule_sha,
            "Historical timing evidence does not bind the supplied prospective rule")
    require(rule["candidate_counts"] == [128, 256, 512] and rule["reference"] == 512,
            "This reducer implements the original 128/256/512 rule only")
    # The precise thresholds are part of the preserved rule, not CLI options.
    require("<=0.005" in rule["rule"] and "<=0.02" in rule["rule"], "Unexpected original rule thresholds")
    declared = datetime.fromisoformat(rule["declared_at"])
    require(declared.tzinfo is not None, "The original rule needs an explicit timestamp timezone")
    starts = {}
    for resource in evidence["resource_receipts"]:
        for invocation in resource["invocations"]:
            command = invocation["command"]
            if "--path-stability" not in command:
                continue
            baseline = command[command.index("--baseline") + 1]
            for case in rule["cases"]:
                if baseline.endswith(case + "/model.npz"):
                    timestamp = datetime.fromisoformat(invocation["started_utc"])
                    require(timestamp.tzinfo is not None and declared < timestamp,
                            "The rule did not precede an original path invocation")
                    starts[case] = min(starts.get(case, timestamp), timestamp)
    require(set(starts) == set(rule["cases"]) and len(starts) == 2,
            "Both timestamped original path invocations are required")
    return rule, {case:timestamp.isoformat() for case,timestamp in starts.items()}


def inspection_summary(document, path, scenario):
    d = document["inspection"]
    require(document["qualified_baseline"] and document["future_labels_used"] is False,
            "P5 findings require a qualified baseline and training-only inspection")
    return {"scenario":scenario, "path":label(path), "sha256":sha(path),
        "qualified_baseline":document["qualified_baseline"], "gate":compact_gate(document["gate"]),
        "future_labels_used":False, "observed_sites":d["observed_sites"], "likelihood_rows":d["likelihood_rows"],
        "signals":d["signals"], "group_coefficient_rms_ratio":d["nonintercept_group_coefficient_rms_ratio"],
        "lag1_sign_agreement":d["lag1_sign_agreement"],
        "mean_residual_acf":{key:value["mean"] for key,value in d["acf"].items()},
        "standardized_residual_rms":d["standardized_residual_rms"],
        "fraction_abs_standardized_residual_above_3":d["fraction_abs_standardized_residual_above_3"]}


def ablation_summary(document, manifest, scenario):
    require(document["complete"] and len(document["rows"]) == 2
            and [r["variant"] for r in document["rows"]] == ["scalar_reference", "full_residual_ar"],
            "Expected one completed scalar/AR pair per scenario")
    rows = []
    for row in document["rows"]:
        scores = row.get("scores")
        require(bool(row["qualified"]) == bool(row["gates"][-1]["passed"]), "Qualification/gate mismatch")
        require((scores is not None) == bool(row["qualified"]), "Unqualified native scores or missing qualified scores")
        rows.append({"variant":row["variant"], "qualified":row["qualified"],
            "posterior_identity":row["posterior_identity"],
            "initial_and_final_gate_passes":[gate["passed"] for gate in row["gates"]],
            "final_gate":compact_gate(row["gates"][-1]),
            "native_scores":None if scores is None else {key:scores[key] for key in ("crps", "mae", "rmse")},
            "native_90_percent_interval":None if scores is None else {key:scores["intervals"]["0.9"][key] for key in ("coverage", "interval_score", "width")},
            "fit_invocation_seconds":row["fit_invocation_seconds"], "forecast_seconds":row.get("forecast_seconds")})
    scalar, ar = ({r["variant"]:r for r in rows}[key] for key in ("scalar_reference", "full_residual_ar"))
    common = scalar["qualified"] and ar["qualified"]
    expected = {"variant":"full_residual_ar", "common_qualified_datasets":int(common),
        "loss_difference_extension_minus_scalar":None if not common else
            {key:ar["native_scores"][key]-scalar["native_scores"][key] for key in ("crps", "mae", "rmse")},
        "mcse_across_datasets":None}
    require(document["comparisons"] == [expected], "Paired loss reduction does not reproduce")
    interpretation = ("Observed losses on one common-qualified dataset; no general predictive superiority claim"
                      if common else "At least one fit failed qualification; no native paired loss exists")
    return {"scenario":scenario, "rows":rows, "comparisons":[expected], "evaluation":document["evaluation"],
            "forecast_paths":manifest["settings"]["paths"], "interpretation":interpretation}


def path_case(document, receipt, receipt_path, scenario):
    require([row["paths"] for row in document["rows"]] == [128,256,512], "Path prefixes differ from the original rule")
    scale = receipt["training_scale"]
    require(isinstance(scale, (float,int)) and scale > 0, "A positive saved common training scale is required")
    reference = document["rows"][-1]["scores"]
    rows = []
    for row in document["rows"]:
        score = row["scores"]
        crps = abs(score["crps"]-reference["crps"])/scale
        coverage = abs(score["intervals"]["0.9"]["coverage"]-reference["intervals"]["0.9"]["coverage"])
        rows.append({"paths":row["paths"], "crps":score["crps"], "rmse":score["rmse"],
            "coverage90":score["intervals"]["0.9"]["coverage"],
            "normalized_absolute_crps_difference_from_512":crps,
            "absolute_coverage90_difference_from_512":coverage, "passes_both_rules":crps <= .005 and coverage <= .02,
            "conditional_predictive_mean_mcse_rms":row["predictive_mean_mcse_rms"],
            "conditional_predictive_mean_mcse_max":row["predictive_mean_mcse_max"],
            "conditional_predictive_mean_mcse_rms_over_training_scale":row["predictive_mean_mcse_rms"]/scale,
            "conditional_predictive_mean_mcse_max_over_training_scale":row["predictive_mean_mcse_max"]/scale})
    return {"scenario":scenario, "rows":rows, "training_scale":scale,
            "posterior_identity":document["posterior_identity"],
            "reference_receipt_path":label(receipt_path), "reference_receipt_sha256":sha(receipt_path)}


def dispositions(scope, inspections, ablations):
    rows = deepcopy(scope["extensions"])
    group_signal = any(row["signals"]["grouped_ridge"] for row in inspections)
    ar_pairs = sum(row["comparisons"][0]["common_qualified_datasets"] for row in ablations)
    for row in rows:
        name = row["id"]
        row["scientific_ablation_executed"] = name == "full_residual_ar"
        if name == "grouped_ridge":
            row["disposition"] = ("optional_training_signal_requires_separate_paired_ablation" if group_signal
                                  else "optional_not_triggered_by_current_source_training_inspections")
            row["rationale"] = ("At least one current inspection crosses the descriptive group-amplitude threshold; no grouped fit or promotion is supplied here."
                if group_signal else "Neither current inspection crosses the declared group-amplitude threshold. No grouped fit was run; this does not prove it cannot help.")
        elif name == "full_residual_ar":
            row.update(disposition="optional_not_promoted_primary_scalar_ridge_retained",
                attempted_ablation_datasets=2, qualified_ablation_datasets=ar_pairs,
                rationale=f"Two bounded paired attempts produced {ar_pairs} common-qualified native comparison(s). Read their actual losses and gates separately; failed inference has no native loss. No automatic promotion follows from these two fields.")
        else:
            row["rationale"] = "No scientific ablation for this extension was performed by these P5 runs. " + row["trigger"]
    return rows


def scientific_view(value):
    """Ignore narrative, file bindings, and resource metadata when comparing."""
    inspections = [{k:v for k,v in row.items() if k not in ("path","sha256")}
                   for row in value["current_source_training_inspections"]]
    ablations = [{k:v for k,v in row.items() if k != "interpretation"} for row in value["ablation_results"]]
    cases = [{k:v for k,v in row.items() if k not in ("reference_receipt_path","reference_receipt_sha256")}
             for row in value["path_count_assessment"]["cases"]]
    statuses = [{k:row[k] for k in ("id","disposition","scientific_ablation_executed",
                 "attempted_ablation_datasets","qualified_ablation_datasets") if k in row} for row in value["extensions"]]
    return {"inspections":inspections, "ablations":ablations, "path_cases":cases, "statuses":statuses,
        "selected_paths":value["path_count_assessment"]["selected_for_subsequent_studies"],
        "passes":value["path_count_assessment"]["passes_on_both_cases"], "primary_model":value["primary_model"],
        "automatic_model_promotion":value["automatic_model_promotion"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("reference-run", "g4-ar", "g3-ar", "g4-paths", "g3-paths", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--scope", type=Path, default=ROOT/"docs/evidence/v03/extension_scope.json")
    parser.add_argument("--rule-document", type=Path, default=ROOT/"docs/evidence/v03/development_decisions.json")
    parser.add_argument("--rule-evidence", type=Path, default=ROOT/"docs/evidence/v03/extension_findings.json")
    parser.add_argument("--compare-findings", type=Path)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Keep sealed evidence intact; use a new output path")
    inputs = Inputs()
    with single_thread_runtime():
        source, environment = source_fingerprint(), jsonable(benchmark_environment())
        aggregation_tool_sha = inputs.bind(__file__)
        evaluation_tool_sha = inputs.bind(ROOT/"tools/evaluate_extensions_v03.py")
        scope = inputs.read(args.scope)
        require(scope["schema"] == "p5-extension-scope-1.0" and scope["inspection_rules"]["no_automatic_promotion"], "Unexpected P5 scope")
        rule_document = inputs.read(args.rule_document)
        timing_evidence = inputs.read(args.rule_evidence, identity=True)
        require(timing_evidence["frozen_scope_sha256"] == sha(args.scope)
                and timing_evidence["tool_sha256"] == evaluation_tool_sha,
                "The supplied scope or P5 evaluator differs from the original prospective assessment")
        rule, original_starts = verify_original_rule(rule_document, timing_evidence, sha(args.rule_document))
        baseline_manifest = inputs.read(args.reference_run/"manifest.json", identity=True)
        require(baseline_manifest["source_fingerprint"] == source and baseline_manifest["stage"] == "development",
                "Reference must be a current-source development run")
        directories = {SCENARIOS[0]:(args.g4_ar,args.g4_paths), SCENARIOS[1]:(args.g3_ar,args.g3_paths)}
        require(len({path.resolve() for pair in directories.values() for path in pair}) == 4, "Four distinct P5 directories are required")
        inventories, inspections, ablations, cases, resources, verified = [], [], [], [], [], []
        original_simulations = {row["scenario"]:row["evaluation"]["simulation_sha256"] for row in timing_evidence["ablation_results"]}
        for scenario, pair in directories.items():
            reference_dir = args.reference_run/scenario/"rep_000"/"A"/"bayes_esn_ridge"
            reference_receipt = inputs.read(reference_dir/"receipt.json")
            model_sha = inputs.bind(reference_dir/"model.npz")
            simulation_sha = inputs.bind(reference_dir.parents[1]/"simulation.npz")
            require(reference_receipt["receipt_identity"] == content_hash({k:v for k,v in reference_receipt.items() if k != "receipt_identity"})
                    and reference_receipt["qualified"] and not reference_receipt["fallback"]
                    and reference_receipt["completed"]
                    and reference_receipt["manifest_identity"] == baseline_manifest["identity"]
                    and reference_receipt["scenario"] == scenario and reference_receipt["replication"] == 0
                    and reference_receipt["track"] == "A" and reference_receipt["model_id"] == "bayes_esn_ridge"
                    and any(item["file"] == "model.npz" and item["sha256"] == model_sha
                            for item in reference_receipt["artifacts"]), "Reference job binding or qualification differs")
            previous_manifest = None
            for directory, mode in zip(pair, ("extensions", "path_stability")):
                manifest = inputs.read(directory/"manifest.json", identity=True)
                require(manifest["source_fingerprint"] == source and manifest["environment"] == environment
                    and manifest["scope_sha256"] == sha(args.scope)
                    and manifest["tool_sha256"] == evaluation_tool_sha
                    and manifest["development_manifest_sha256"] == sha(args.reference_run/"manifest.json")
                    and manifest["baseline_sha256"] == model_sha
                    and manifest["simulation_sha256"] == simulation_sha
                    and manifest["simulation_sha256"] == original_simulations[scenario], "P5 current-source or original-case bindings differ")
                require(_verify_terminal_inventory(directory, manifest, mode), "P5 terminal inventory is missing")
                settings = manifest["settings"]
                expected_settings = {"track":"A", "warmup":1000, "draws":2000,
                    "continuation":2000, "paths":128, "seed":20261103,
                    "inspection_draws_per_chain":32, "horizon":None,
                    "path_stability":mode == "path_stability",
                    "extensions":"full_residual_ar" if mode == "extensions" else None}
                require(settings == expected_settings, "Original P5 settings, seeds or budgets were changed")
                if previous_manifest is not None:
                    for key in ("input_posterior_identity", "simulation_sha256", "baseline_sha256"):
                        require(manifest[key] == previous_manifest[key], "AR/path studies use different baselines")
                    require(settings["seed"] == previous_manifest["settings"]["seed"], "Paired P5 seeds differ")
                previous_manifest = manifest
                inv = inputs.read(directory/"output_inventory.json", identity=True)
                decisions = inputs.read(directory/"decisions.json")
                require(decisions["manifest_identity"] == manifest["identity"], "P5 decisions belong to another design")
                inventories.append({"path":label(directory/"output_inventory.json"), "sha256":sha(directory/"output_inventory.json"),
                    "identity":inv["identity"], "manifest_identity":manifest["identity"], "mode":mode, "scenario":scenario,
                    "decisions_path":label(directory/"decisions.json"), "decisions_sha256":sha(directory/"decisions.json"), "inventoried_files":len(inv["entries"])})
                inspected = inputs.read(directory/"inspection.json")
                require(inspected["manifest_identity"] == manifest["identity"], "Inspection belongs to another design")
                if mode == "extensions":
                    inspections.append(inspection_summary(inspected, directory/"inspection.json", scenario))
                    raw = inputs.read(directory/"paired_ablation.json")
                    for row in raw["rows"]:
                        require(inputs.read(directory/row["variant"]/"receipt.json") == row, "Variant receipt differs from paired result")
                    ablations.append(ablation_summary(raw, manifest, scenario))
                else:
                    raw = inputs.read(directory/"path_stability.json")
                    require(raw["posterior_identity"] == manifest["input_posterior_identity"], "Path posterior differs")
                    cases.append(path_case(raw, reference_receipt, reference_dir/"receipt.json", scenario))
                require(raw["evaluation"]["scenario"] == scenario and raw["evaluation"]["replication"] == 0
                    and raw["evaluation"]["track"] == "A" and raw["evaluation"]["manifest_identity"] == baseline_manifest["identity"]
                    and raw["evaluation"]["simulation_sha256"] == simulation_sha
                    and raw["evaluation"]["development_manifest_sha256"] == manifest["development_manifest_sha256"], "P5 evaluated another case")
                resource_path = Path(str(directory)+"_resources.json")
                if resource_path.exists():
                    invocations = inputs.read(resource_path)
                    require(invocations and all(row["exit_code"] == 0 for row in invocations), "A supplied resource receipt contains failed invocations")
                    for invocation in invocations:
                        command = invocation["command"]
                        require("--output" in command
                                and Path(command[command.index("--output")+1]).resolve() == directory.resolve(),
                                "A supplied resource receipt belongs to a different output directory")
                    resources.append({"path":label(resource_path),"sha256":sha(resource_path),"invocations":invocations})
                verified.append((directory,manifest,mode))
        passes = {str(count):all(next(row for row in case["rows"] if row["paths"] == count)["passes_both_rules"] for case in cases) for count in (128,256,512)}
        selected = min(int(count) for count, passed in passes.items() if passed)
        invocations = [row for resource in resources for row in resource["invocations"]]
        complete_resources = len(resources) == 4
        g3 = next(case for case in cases if case["scenario"] == SCENARIOS[1])["rows"][-1]
        limitations = (f"512 is the finite reference ceiling, not evidence that predictive Monte Carlo error is negligible. G3 RMS conditional predictive-mean MCSE at 512 is {g3['conditional_predictive_mean_mcse_rms_over_training_scale']:.6g} training standard deviations. No unplanned larger path run is included; these MCSE estimates exclude posterior MCMC error.")
        result = {"schema":"p5-extension-findings-1.0", "artifact_type":"current_source_same_design_aggregation_not_new_independent_evidence",
            "scientific_results_in_this_file":True, "source_fingerprint":source, "tool_sha256":evaluation_tool_sha,
            "aggregation_tool_sha256":aggregation_tool_sha, "frozen_scope_path":label(args.scope), "frozen_scope_sha256":sha(args.scope),
            "primary_model":"bayes_esn_ridge", "automatic_model_promotion":False,
            "evidence_inventories":inventories, "current_source_training_inspections":inspections,
            "ablation_results":ablations, "extensions":dispositions(scope, inspections, ablations),
            "path_count_assessment":{"cases":cases, "predeclared_rule":rule, "predeclared_rule_path":label(args.rule_document),
                "predeclared_rule_document_sha256":sha(args.rule_document), "prespecification_verified_against_both_first_path_invocation_times":True,
                "prespecification_scope":"The original rule preceded both timestamped original path invocations. This aggregation maps fresh outputs to those same data cases; it does not redeclare a prospective rule.",
                "passes_on_both_cases":passes, "selected_for_subsequent_studies":selected,
                "existing_ablation_forecast_path_count_unchanged":128, "limitations":limitations},
            "resource_receipts":resources, "resource_summary":{"model_workers_per_p5_invocation":1, "numeric_threads_per_model_worker":1,
                "cross_invocation_concurrency":None,
                "complete_cli_resource_receipts":complete_resources,
                "total_cli_wall_seconds":sum(row["wall_seconds"] for row in invocations) if complete_resources else None,
                "total_cli_cpu_seconds":sum(row["user_seconds"]+row["system_seconds"] for row in invocations) if complete_resources else None,
                "maximum_single_cli_process_lifetime_peak_rss_bytes":max(row["process_lifetime_peak_rss_bytes"] for row in invocations) if complete_resources else None,
                "scope":"Optional measured CLI receipts only; missing total resources are not extrapolated from ablation fit/forecast timers. Per-process peaks are not aggregate cgroup peaks."},
            "aggregation_provenance":{"historical_rule_timing_evidence":label(args.rule_evidence), "historical_rule_evidence_identity":timing_evidence["identity"],
                "original_first_path_invocation_times":original_starts, "new_independent_datasets":0,
                "original_dataset_hashes_verified":True, "new_model_selection_rules":False,
                "meaning":"Fresh current-source outputs are reduced independently. Historical evidence supplies only original prospective-rule timing and case identities; its numerical findings are not copied."},
            "claim_limits":["One dataset per scenario; no across-dataset MCSE or confirmatory superiority claim.",
                "Unqualified AR fits have no native losses; failed inference is distinct from worse forecasting.",
                "Training ACF/group summaries are descriptive triage, not independent-site tests or sparsity evidence.",
                "The path-count rule controls later numerical budgets; it neither changes these 128-path AR comparisons nor establishes model adequacy.",
                "No additional extension is automatically promoted by this aggregation."]}
        if args.compare_findings:
            expected = inputs.read(args.compare_findings, identity=True)
            require(scientific_view(result) == scientific_view(expected), "Reaggregated scientific fields differ from the supplied sealed findings")
            result["aggregation_verification"] = {"compared_receipt":label(args.compare_findings),
                "compared_identity":expected["identity"], "scientific_fields_exactly_equal":True,
                "scope":"Inspections, numerical ablation results and costs, path-count case metrics and decisions, extension dispositions/counts; presentation and file/resource metadata excluded"}
        require(source_fingerprint() == source and jsonable(benchmark_environment()) == environment, "Source/runtime changed during aggregation")
        for directory,manifest,mode in verified:
            require(_verify_terminal_inventory(directory,manifest,mode), "P5 inventory changed during aggregation")
        inputs.verify()
        result["aggregation_input_hashes"] = inputs.hashes
        result = jsonable(result)
        result["identity"] = content_hash(result)
        write_json(args.output,result)
    print(json.dumps({"output":str(args.output),"identity":result["identity"],"selected_paths":selected,
                      "verified_inventories":4,"scientific_comparison":bool(args.compare_findings)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
