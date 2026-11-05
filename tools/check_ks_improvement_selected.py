#!/usr/bin/env python3
"""Acceptance checks on frozen improvement selections, without posterior refitting.

Uses the original fixed-fit adversarial reader check. Additionally refits only
the selected preprocessing/features on unchanged permitted training data under
an adversarial reader. Selection, posterior parameters, and randomness remain
fixed. Run after the main coordinator exits; the scheduler lock is required.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from threadpoolctl import threadpool_limits

from check_ks_selected_invariance import _case, _bitwise_equal
from bayes_stesn.io import content_hash
from bayes_stesn.applications.ks_data import KSAdapter, circle_embedding
from bayes_stesn.applications.ks_native import _prepare_delivered_scale, _resolve_bayes
from bayes_stesn.applications.ks_improvement_runner import source_identity, fit_key
from bayes_stesn.applications.ks_runtime import admission, atomic_json, file_hash, now, resource_snapshot

PROTOCOLS = {"A": "interleaved", "B": "all_observed", "C": "contiguous"}


def read(path):
    return json.loads(Path(path).read_text())


def preprocessing_case(adapter, candidate, seed):
    """The modified reader reaches hidden and future values, never fit inputs."""
    training = adapter.training(0)
    resolved, spec, _ = _resolve_bayes(candidate["config"], seed)
    original = adapter.native.read_observed
    forbidden = adapter.partition != 0
    first = _prepare_delivered_scale(training.panel, resolved, spec)
    first_identity = first.identity
    first_features = first.features.fingerprint
    first_phi = first.Phi.copy()
    first_z = first.Z.copy()
    del first
    gc.collect()
    counts = {"read_calls": 0, "modified_hidden_values": 0, "modified_future_values": 0}

    def changed_read(handle, start=None, stop=None):
        values = original(handle, start, stop).copy()
        times = np.arange(0 if start is None else start,
                          adapter.n_states if stop is None else stop)
        if len(times) != len(values):
            raise ValueError("Reader indexing does not match the perturbation")
        counts["read_calls"] += 1
        values[:, forbidden] += 12345.
        values[times >= 6000] -= 24680.
        counts["modified_hidden_values"] += int(len(times)*forbidden.sum())
        counts["modified_future_values"] += int(np.count_nonzero(times >= 6000)*values.shape[1])
        return values

    adapter.native.read_observed = changed_read
    try:
        changed = adapter.training(0)
        second = _prepare_delivered_scale(changed.panel, resolved, spec)
        checks = {
            "permitted_training_values_bitwise_equal": _bitwise_equal(training.panel.y, changed.panel.y),
            "delivered_transform_identical": training.transform == changed.transform,
            "training_identity_identical": dict(training.identity) == dict(changed.identity),
            "prepared_identity_identical": first_identity == second.identity,
            "feature_fingerprint_identical": first_features == second.features.fingerprint,
            "spatial_basis_bitwise_equal": _bitwise_equal(first_phi, second.Phi),
            "temporal_features_bitwise_equal": _bitwise_equal(first_z, second.Z),
        }
    finally:
        adapter.native.read_observed = original
    return {"checks": {k: bool(v) for k, v in checks.items()}, "passed": bool(all(checks.values())),
            "perturbation_counts": counts,
            "scope": "preprocessing/features refitted only; no likelihood or posterior refit; validation selection remains frozen"}


def run(args):
    threadpool_limits(1)
    root, output = args.results.resolve(), args.output.resolve()
    experiment = read(root/"experiment.json")
    freeze = read(root/"freeze.json")
    if freeze["identity"] != content_hash({k: v for k, v in freeze.items() if k != "identity"}):
        raise ValueError("Freeze identity is invalid")
    if freeze["experiment_identity"] != experiment["identity"] or experiment["source"] != source_identity():
        raise ValueError("Selection and current numerical source are incompatible")
    cfg = experiment["config"]
    policy = {"first_target": 6000, "history": freeze["history"], "horizon": 100,
              "predictive_paths": 16, "seed": 83, "trajectory": 0,
              "selected_roles": ["main", "anchor", "sensitivity"],
              "max_seconds": 180., "partition": "validation"}
    identity = content_hash({"experiment": experiment["identity"], "freeze": freeze["identity"],
                             "policy": policy, "source": file_hash(__file__),
                             "reused_check_source": file_hash(Path(__file__).with_name("check_ks_selected_invariance.py"))})
    record = {"schema": "ks-improvement-selected-acceptance/1", "identity": identity,
              "experiment_identity": experiment["identity"], "freeze_identity": freeze["identity"],
              "policy": policy, "cases": [], "pending": [], "new_posterior_fits": 0,
              "selection_rerun": False, "test_partition_opened": False,
              "reader_scope": "adversarial output separation; unchanged facade may internally construct labels"}
    if output.exists():
        old = read(output)
        if old["identity"] != identity:
            raise ValueError("Existing acceptance receipt has a different identity")
        record["cases"] = old["cases"]
    expected = []
    for track in cfg["tracks"]:
        selected = freeze["selected"][track]
        roles = {selected["anchor_id"]: ["anchor"]}
        roles.setdefault(selected["main"]["id"], []).append("main")
        if selected.get("sensitivity"):
            roles.setdefault(selected["sensitivity"]["id"], []).append("sensitivity")
        for candidate in cfg["registry"]["tracks"][track]["candidates"]:
            if candidate["id"] in roles:
                expected.append((track, candidate, roles[candidate["id"]]))
    if not args.execute:
        print(json.dumps({"execute": False, "expected_cases": len(expected), "policy": policy}, indent=2))
        return
    started = time.monotonic()
    completed = {(r["track"], r["candidate"]) for r in record["cases"]}
    with (root/"scheduler.lock").open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for track, candidate, roles in expected:
            key = fit_key(track, candidate, 0, cfg["model_seed"])
            fit_path = root/"fits"/key/"fit.npz"
            receipt = read(root/"fits"/key/"receipt.json")
            if (receipt["fit_key"] != key or receipt["track"] != track
                    or receipt["candidate_id"] != candidate["id"] or receipt["trajectory"] != 0
                    or receipt["seed"] != cfg["model_seed"]
                    or file_hash(fit_path) != receipt["fit_sha256"]):
                raise ValueError("Fitted artifact checksum/identity mismatch")
            if (track, candidate["id"]) in completed:
                retained = next(r for r in record["cases"] if r["track"] == track and r["candidate"] == candidate["id"])
                if (retained["fit_sha256"] != receipt["fit_sha256"]
                        or retained["native_fit_identity"] != receipt["native_fit_identity"]
                        or retained["fit_qualified"] != bool(receipt["diagnostics"]["qualified"])):
                    raise ValueError("Retained acceptance case no longer matches its fitted artifact")
                continue
            if time.monotonic()-started >= policy["max_seconds"]:
                record["pending"].append({"track": track, "candidate": candidate["id"], "reason": "acceptance_time_budget"})
                continue
            admission(root, reserve_bytes=cfg["budget"]["disk_reserve_bytes"],
                      estimated_peak_bytes=cfg["budget"]["estimated_worker_peak_bytes"],
                      worker_cap_bytes=cfg["budget"]["combined_worker_cap_bytes"])
            if not receipt["diagnostics"]["qualified"]:
                record["pending"].append({"track": track, "candidate": candidate["id"], "reason": "inference_unqualified"})
                continue
            adapter = KSAdapter(args.cache, PROTOCOLS[track])
            case = _case(fit_path, {**receipt, "identity": receipt["native_fit_identity"]},
                         adapter, policy, candidate["id"])
            case["roles"] = roles
            case["preprocessing_refit"] = preprocessing_case(adapter, candidate, cfg["model_seed"])
            case["periodic_embedding_seam"] = bool(np.allclose(circle_embedding(np.array([0.])),
                  circle_embedding(np.array([22.])), rtol=0., atol=1e-14))
            case["passed"] = bool(case["passed"] and case["preprocessing_refit"]["passed"] and case["periodic_embedding_seam"])
            record["cases"].append(case)
            record["resource"] = resource_snapshot(root)
            record["updated_utc"] = now()
            atomic_json(output, record)
            gc.collect()
    record["expected_cases"] = len(expected)
    record["passed_cases"] = sum(bool(r["passed"]) for r in record["cases"])
    record["failed_cases"] = sum(not r["passed"] for r in record["cases"])
    record["complete"] = len(record["cases"]) == len(expected) and not record["pending"]
    record["passed"] = bool(record["complete"] and record["failed_cases"] == 0)
    record["last_invocation_seconds"] = time.monotonic()-started
    record["updated_utc"] = now()
    atomic_json(output, record)
    print(json.dumps({k: record[k] for k in ("expected_cases", "passed_cases", "failed_cases", "pending", "complete", "passed", "last_invocation_seconds")}, indent=2))
    if record["failed_cases"]:
        raise RuntimeError("Selected-model acceptance check failed; preserve evidence and investigate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    run(parser.parse_args())
