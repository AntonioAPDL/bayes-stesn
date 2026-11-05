#!/usr/bin/env python3
"""Check fixed core Bayesian artifacts against perturbed forbidden responses.

This report-only acceptance tool never fits a model, repeats selection, or
opens a test target interval. It covers every declared Bayesian candidate on
trajectory 0 in both main tracks, so any later selected architecture is covered.
Missing completed artifacts remain explicitly pending; rerunning reuses matching
completed checks. Output contains relative evidence identifiers and hashes only.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from bayes_stesn.applications.ks_data import KSAdapter
from bayes_stesn.applications.ks_native import NativeFit
from bayes_stesn.applications.ks_runtime import admission, atomic_json, file_hash, now
from bayes_stesn.io import content_hash, seed_for


SCHEMA = "ks-core-fixed-fit-invariance-1"
PROTOCOLS = {"A": "interleaved", "B": "all_observed"}


def _bitwise_equal(left, right):
    a, b = np.asarray(left), np.asarray(right)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _input_equal(first, second):
    checks = {
        "history_values": _bitwise_equal(first.panel.y, second.panel.y),
        "history_coordinates": _bitwise_equal(first.panel.coordinates, second.panel.coordinates),
        "history_times": _bitwise_equal(first.panel.times, second.panel.times),
        "history_site_ids": first.panel.site_ids == second.panel.site_ids,
        "history_metadata": dict(first.panel.metadata) == dict(second.panel.metadata),
        "target_coordinates": _bitwise_equal(first.target_coordinates, second.target_coordinates),
        "target_site_ids": _bitwise_equal(first.target_site_ids, second.target_site_ids),
        "target_times": _bitwise_equal(first.target_time, second.target_time),
        "response_transform": first.transform == second.transform,
        "data_identity": dict(first.identity) == dict(second.identity),
    }
    return {key: bool(value) for key, value in checks.items()}


def _case(fit_path, receipt, adapter, policy, candidate):
    started = time.perf_counter()
    fit = NativeFit.load(fit_path)
    if fit.model_id != "bayes_stesn":
        raise ValueError("The fixed-fit acceptance case requires its Bayesian artifact")
    if fit.training.metadata.get("phase") != "core" or fit.training.metadata.get("trajectory") != 0:
        raise ValueError("The acceptance artifact must be a core trajectory-0 fit")
    if fit.training.metadata.get("ks_protocol") != adapter.spatial_protocol:
        raise ValueError("The acceptance artifact has the wrong spatial protocol")
    native_state = content_hash(fit.to_state())
    origin, history, horizon = policy["first_target"], policy["history"], policy["horizon"]
    before_window = adapter.model_window(0, origin, "validation", history=history, horizon=horizon)
    forecast_seed = seed_for(policy["seed"], SCHEMA, receipt["track"], candidate)
    before = fit.forecast(before_window.panel, horizon, before_window.target_coordinates,
        before_window.target_site_ids, n_paths=policy["predictive_paths"], seed=forecast_seed)
    # Evaluation is used only to demonstrate that the adversarial change reaches
    # the reader. It is never forwarded to the fitted model or used for selection.
    before_labels = adapter.evaluation_window(0, origin, "validation", history=history, horizon=horizon).values
    forbidden = adapter.partition != 0
    perturbations = ("future", "hidden", "both") if forbidden.any() else ("future",)
    original = adapter.native.read_observed
    results = []
    for perturbation in perturbations:
        counts = {"read_calls": 0, "modified_future_values": 0, "modified_hidden_history_values": 0}

        def changed_read(handle, start=None, stop=None):
            values = original(handle, start, stop).copy()
            times = np.arange(0 if start is None else start,
                              adapter.n_states if stop is None else stop)
            if len(times) != len(values):
                raise ValueError("Perturbation reader indexing does not match returned values")
            counts["read_calls"] += 1
            if perturbation in ("hidden", "both"):
                values[:, forbidden] += 12345.
                counts["modified_hidden_history_values"] += int(np.count_nonzero(times < origin) * forbidden.sum())
            if perturbation in ("future", "both"):
                values[times >= origin] -= 24680.
                counts["modified_future_values"] += int(np.count_nonzero(times >= origin) * values.shape[1])
            return values

        adapter.native.read_observed = changed_read
        try:
            changed_window = adapter.model_window(0, origin, "validation", history=history, horizon=horizon)
            changed = fit.forecast(changed_window.panel, horizon, changed_window.target_coordinates,
                changed_window.target_site_ids, n_paths=policy["predictive_paths"], seed=forecast_seed)
            changed_labels = adapter.evaluation_window(0, origin, "validation", history=history, horizon=horizon).values
        finally:
            adapter.native.read_observed = original
        inputs = _input_equal(before_window, changed_window)
        labels_changed = not np.array_equal(before_labels, changed_labels)
        samples_equal = _bitwise_equal(before.samples, changed.samples)
        means_equal = _bitwise_equal(before.mean, changed.mean)
        physical_samples_equal = _bitwise_equal(before_window.transform.inverse(before.samples),
                                                changed_window.transform.inverse(changed.samples))
        passed = bool(all(inputs.values()) and labels_changed and samples_equal and means_equal and physical_samples_equal)
        results.append({"perturbation": perturbation, "passed": passed,
            "input_equalities": inputs, "evaluation_labels_changed": labels_changed,
            "samples_bitwise_equal": bool(samples_equal), "means_bitwise_equal": bool(means_equal),
            "physical_samples_bitwise_equal": bool(physical_samples_equal),
            "maximum_sample_absolute_difference": float(np.max(np.abs(before.samples-changed.samples))),
            "maximum_mean_absolute_difference": float(np.max(np.abs(before.mean-changed.mean))),
            **counts})
    unchanged = native_state == content_hash(fit.to_state())
    artifact_unchanged = file_hash(fit_path) == receipt["fit_sha256"]
    result = {"track": receipt["track"], "candidate": candidate, "trajectory": 0,
        "fit_identity": receipt["identity"], "native_fit_identity": fit.identity,
        "fit_sha256": receipt["fit_sha256"], "fit_qualified": bool(fit.diagnostics["qualified"]),
        "input_data_identity_sha256": before_window.identity["identity_sha256"],
        "forecast_seed": forecast_seed, "samples_shape": list(before.samples.shape),
        "mean_is_average_of_complete_paths": bool(np.array_equal(before.mean, before.samples.mean(axis=0))),
        "fitted_state_unchanged": bool(unchanged), "artifact_file_unchanged": bool(artifact_unchanged),
        "hidden_history_check": "performed" if forbidden.any() else "not_applicable_all_64_sites_are_permitted",
        "forbidden_observation_panel_sites": int(forbidden.sum()), "perturbations": results,
        "passed": bool(unchanged and artifact_unchanged and all(x["passed"] for x in results)),
        "elapsed_seconds": time.perf_counter()-started, "completed_utc": now()}
    del fit, before
    gc.collect()
    return result


def run(results_root, cache, output, *, paths=16, seed=83):
    started = time.perf_counter()
    results_root, output = Path(results_root), Path(output)
    experiment = json.loads((results_root/"experiment.json").read_text())
    configuration = experiment["config"]
    if paths < 1 or seed < 0:
        raise ValueError("Use a positive acceptance path count and a nonnegative fixed seed")
    package = Path(__file__).resolve().parents[1]/"src/bayes_stesn"
    numerical_hashes = {key: value for key, value in experiment["source"].items() if key != "runtime"}
    if any(file_hash(package/name) != expected for name, expected in numerical_hashes.items()):
        raise ValueError("Numerical source changed after the experiment was frozen")
    policy = {"first_target": 6000, "adapter_origin": 5999, "partition": "validation",
        "history": configuration["history"], "horizon": configuration["horizon"],
        "predictive_paths": int(paths), "seed": int(seed), "trajectory": 0,
        "future_perturbation": "Subtract 24680 from all observation-panel responses at time indices at least 6000.",
        "hidden_perturbation": "Add 12345 to all forbidden observation-panel site responses.",
        "criterion": "bitwise_equal_model_inputs_complete_paths_and_means; evaluator_labels_must_change",
        "fixed_fit_policy": "load_existing_artifact; no_refit_no_selection_rerun_no_calibration_rerun"}
    identity = content_hash({"schema": SCHEMA, "experiment": experiment["identity"],
        "policy": policy, "tool_sha256": file_hash(__file__)})
    record = {"schema": SCHEMA, "state": "running", "identity": identity, "experiment_identity": experiment["identity"],
        "tool_sha256": file_hash(__file__), "numerical_source_sha256": numerical_hashes,
        "policy": policy, "cases": [], "pending": [], "core_test_labels_opened": False,
        "new_model_fits": 0, "preprocessing_refitted": False,
        "preprocessing_scope": "This check freezes fitted preprocessing. Separate training-boundary refit invariance is covered by tests/test_ks_model_boundary.py.",
        "reader_scope": "The delivered facade can internally construct labels; this establishes output invariance, not future-free physical disk access.",
        "scientific_scope": f"Acceptance only; {paths}-path examples do not replace the selected study's predictive ensembles or inference diagnostics."}
    if output.exists():
        previous = json.loads(output.read_text())
        if previous["identity"] != identity:
            raise ValueError("Existing acceptance receipt has an incompatible policy/source; use a new versioned output")
        record["cases"] = previous["cases"]
    by_candidate = {}
    expected = []
    for track in PROTOCOLS:
        for candidate in configuration["candidates"][track]["bayes_stesn"]:
            expected.append((track, candidate["id"]))
            by_candidate[track, content_hash(candidate["config"])] = candidate["id"]
    artifacts = {}
    for path in (results_root/"fits").glob("*/receipt.json"):
        receipt = json.loads(path.read_text())
        if (receipt["model"] != "bayes_stesn" or receipt["phase"] != "core"
                or receipt["trajectory"] != 0 or receipt["seed"] != configuration["model_seed"]
                or receipt["track"] not in PROTOCOLS):
            continue
        candidate = by_candidate.get((receipt["track"], content_hash(receipt["config"])))
        if candidate is None or receipt["state"] == "failed":
            continue
        key = receipt["track"], candidate
        if key in artifacts:
            raise ValueError("Multiple incompatible artifacts represent one acceptance case")
        artifacts[key] = path.parent/"fit.npz", receipt
    completed = {(r["track"], r["candidate"]): r for r in record["cases"]}
    reused = 0
    for key in expected:
        if key not in artifacts:
            record["pending"].append({"track": key[0], "candidate": key[1],
                                      "reason": "matching_completed_core_fit_not_available"})
            continue
        fit_path, receipt = artifacts[key]
        if file_hash(fit_path) != receipt["fit_sha256"]:
            raise ValueError("Acceptance fit artifact no longer matches its completed receipt")
        if key in completed:
            if completed[key]["fit_sha256"] != receipt["fit_sha256"]:
                raise ValueError("Completed acceptance case has a changed fitted artifact")
            reused += 1
            continue
        admission(results_root, reserve_bytes=configuration["budget"]["disk_reserve_bytes"],
            estimated_peak_bytes=512*1024**2,
            worker_cap_bytes=configuration["budget"]["combined_worker_cap_bytes"])
        adapter = KSAdapter(cache, PROTOCOLS[key[0]], "core")
        result = _case(fit_path, receipt, adapter, policy, key[1])
        record["cases"].append(result)
        atomic_json(output, record)
    record.update(state="complete" if len(record["cases"]) == len(expected) else "pending_artifacts",
        expected_cases=len(expected), completed_cases=len(record["cases"]),
        passed_cases=sum(r["passed"] for r in record["cases"]),
        failed_cases=sum(not r["passed"] for r in record["cases"]),
        perturbation_checks=sum(len(r["perturbations"]) for r in record["cases"]),
        reused_cases=reused, last_invocation_seconds=time.perf_counter()-started,
        total_measured_case_seconds=sum(r["elapsed_seconds"] for r in record["cases"]), updated_utc=now())
    atomic_json(output, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=16)
    parser.add_argument("--seed", type=int, default=83)
    args = parser.parse_args()
    with threadpool_limits(1):
        record = run(args.results, args.cache, args.output, paths=args.paths, seed=args.seed)
    print(json.dumps({key: record[key] for key in ("state", "expected_cases", "completed_cases",
        "passed_cases", "failed_cases", "perturbation_checks", "reused_cases", "last_invocation_seconds")}))
    if record["failed_cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
