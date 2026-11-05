"""Focused identity, inference-gate, budget and stage-contract checks."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from bayes_stesn.io import content_hash
from bayes_stesn.applications.ks_fullfield_candidates import (
    SITES, active_configuration, build_registry, estimate_cost)
from bayes_stesn.applications import ks_fullfield_runner as runner
from bayes_stesn.applications.ks_runtime import atomic_json


@pytest.fixture(scope="module")
def registry():
    return build_registry()


def lightweight_campaign(tmp_path, registry):
    campaign = runner.Campaign.__new__(runner.Campaign)
    campaign.cfg = runner.default_configuration()
    campaign.output = tmp_path
    campaign.source = {"files": {"applications/ks_metrics.py": "metric-source"}}
    campaign.cache = "unused"
    campaign.registry = registry
    campaign.identity = "fixture-campaign"
    campaign.ledger = runner.RunLedger(tmp_path/"ledger")
    atomic_json(tmp_path/"history_policy.json", {"history": 512, "excluded_candidates": {}})
    atomic_json(tmp_path/"experiment.json", {"created_utc": runner.now()})
    return campaign


def row(name, loss, early, vpt, point=None, prior="ridge", coefficients=20):
    return {"id": name, "eligible": True, "loss": loss, "early_rmse": early,
        "vpt_physical": vpt, "point_loss": loss if point is None else point,
        "prior_family": prior, "complexity": {"readout_coefficients": coefficients}}


def test_roster_is_raw_unique_joint_and_json_stable(registry):
    assert {t: len(v["candidates"]) for t, v in registry["tracks"].items()} == {"A": 79, "B": 79, "C": 53}
    restored = json.loads(json.dumps(registry))
    assert content_hash({k: v for k, v in restored.items() if k != "identity"}) == registry["identity"]
    for track, block in registry["tracks"].items():
        candidates = block["candidates"]
        assert len({c["effective_hash"] for c in candidates}) == len(candidates)
        assert len(block["coverage"]["joint_leak_radius_pairs"]) == (24 if track == "C" else 36)
        assert block["coverage"]["units"] == [128, 256, 512, 1000]
        for c in candidates:
            assert c["config"]["features"]["kind"] == "raw_esn"
            assert c["active_config"]["features"]["input_sites"] == SITES[track]
            assert c["cost"]["reservoir_input_dimension"] == SITES[track]*len(c["config"]["features"]["lags"])
        joint = [c for c in candidates if c["stratum"] == "large_capacity_memory_joint"]
        assert {c["config"]["features"]["units"][0] for c in joint} == {512, 1000}
        assert any(max(c["config"]["features"]["lags"]) == 32 for c in joint)


def test_capacity_ladder_is_matched_and_rhs_has_exact_control(registry):
    for block in registry["tracks"].values():
        by_label = {c["label"]: c for c in block["candidates"]}
        reference = by_label["dynamics_a0.2_r0.9"]["config"]["features"]
        for units in (256, 512, 1000):
            for target in ("state", "increment"):
                f = by_label[f"capacity{units}_{target}"]["config"]["features"]
                for name in ("leak_patterns", "spectral_radius", "input_scales", "connectivity", "lags"):
                    assert f[name] == reference[name]
        ids = {c["id"] for c in block["candidates"]}
        assert all(c["matched_ridge_id"] in ids for c in block["candidates"] if c["prior_family"] == "rhs")


def test_effective_hash_ignores_inactive_encoder_fields_but_binds_forecast_policy(registry):
    cfg = registry["tracks"]["A"]["candidates"][0]["config"]
    changed = copy.deepcopy(cfg)
    changed["features"].update(modes=999, recurrent_norm=.17)
    assert active_configuration(cfg, 39) == active_configuration(changed, 39)
    changed["forecast_parameter_rng_policy"] = "per_origin_v1"
    assert active_configuration(cfg, 39) != active_configuration(changed, 39)


def test_memory_estimate_counts_entire_coefficient_memmap(registry):
    cfg = registry["tracks"]["B"]["candidates"][0]["config"]
    small = estimate_cost(cfg, 64)
    changed = copy.deepcopy(cfg); changed["diagnostics"]["functional_draws"] *= 2
    big = estimate_cost(changed, 64)
    assert big["estimated_peak_bytes"]-small["estimated_peak_bytes"] == small["coefficient_diagnostic_storage_bytes"]
    assert small["dense_coefficient_covariance_if_materialized_bytes"] > small["temporal_gram_bytes"]


def test_distinct_frozen_selection_objectives_and_mc_ties():
    rows = [row("distribution", 1., .8, 2., point=3.), row("dynamics", 1.2, .2, 4., point=2.),
            row("point", 1.4, .3, 3., point=1.)]
    policy = runner.default_configuration()["selection"]
    choices = runner.choose_selections(rows, policy)
    assert choices["distribution"]["candidate"]["id"] == "distribution"
    assert choices["dynamics"]["candidate"]["id"] == "dynamics"
    assert choices["point"]["candidate"]["id"] == "point"
    choices = runner.choose_selections(rows, policy, {"distribution": {"dynamics": .25}})
    assert choices["distribution"]["candidate"]["id"] == "dynamics"
    rows[1]["eligible"] = False
    assert "dynamics" not in runner.selection_frontier(rows)


def test_response_independent_order_is_complete_and_prioritizes_controls(registry):
    for block in registry["tracks"].values():
        ordered = runner.ordered_candidates(block["candidates"])
        assert {c["id"] for c in ordered} == {c["id"] for c in block["candidates"]}
        assert ordered[0]["id"] == block["reference_id"]
        first_labels = {c["label"] for c in ordered[:13]}
        assert {"rhs_base_state", "capacity1000_state", "joint1000_lag8_increment"} <= first_labels


def test_configuration_rejects_test_access_and_budget_reset():
    cfg = runner.default_configuration(); runner.validate_configuration(cfg)
    cfg["screen"]["origins"] = [8000]
    with pytest.raises(ValueError, match="validation boundary"): runner.validate_configuration(cfg)
    cfg = runner.default_configuration(); cfg["budget"]["workers"] = 2
    with pytest.raises(ValueError, match="one expensive worker"): runner.validate_configuration(cfg)
    cfg = runner.default_configuration(); cfg["budget"]["wall_seconds_cap"] = float("inf")
    with pytest.raises(ValueError, match="Finite positive budget"): runner.validate_configuration(cfg)


def test_atomic_json_roundtrip_keeps_job_identity(tmp_path, registry):
    campaign = lightweight_campaign(tmp_path, registry)
    candidate = registry["tracks"]["A"]["candidates"][0]
    job = campaign.job("screen", "A", candidate, 0, [6000], paths=256)
    job["tuple_fixture"] = (1, 2)
    job["job_id"] = runner._job_identity(job)[:24]
    runner.write_once(tmp_path/"job.json", job)
    runner.write_once(tmp_path/"job.json", job)
    assert runner._job_identity(runner.read_json(tmp_path/"job.json")) == runner._job_identity(job)


def test_process_identity_maps_real_namespace_child():
    own = runner.process_identity()
    assert own["pid"] == int(Path("/proc/self/stat").read_text().split()[0])
    assert own["namespace_pid"] == os.getpid()
    assert runner.process_matches(own)
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"])
    try:
        child = runner.child_process_identity(process.pid)
        assert child and child["namespace_pid"] == process.pid
        assert runner.process_matches(child)
        assert child["pid_namespace"] == own["pid_namespace"]
    finally:
        process.terminate(); process.wait(timeout=5)
    assert not runner.process_matches(child)


def test_resume_refuses_live_orphan_and_charges_dead_interval(tmp_path):
    path = tmp_path/"attempts"/"job.json"
    atomic_json(path, {"job_id": "job", "stage": "screen", "state": "running",
        "process": runner.process_identity(), "started_epoch": time.time()-20, "elapsed_seconds": 5})
    with pytest.raises(RuntimeError, match="orphaned numerical worker"): runner.settle_attempts(tmp_path)
    record = runner.read_json(path); record["process"] = None; atomic_json(path, record)
    runner.settle_attempts(tmp_path)
    settled = runner.read_json(path)
    assert settled["elapsed_seconds"] >= 20
    assert settled["state"] == "interrupted"
    assert not settled["observed_exact_end"]


def test_preparation_charge_occurs_once_only_in_total_shared_search(tmp_path, registry):
    campaign = lightweight_campaign(tmp_path, registry)
    atomic_json(tmp_path/"attempts"/"one.json", {"stage": "screen", "state": "complete", "elapsed_seconds": 10.})
    atomic_json(tmp_path/"attempts"/"two.json", {"stage": "confirm", "state": "complete", "elapsed_seconds": 20.})
    assert campaign.consumed() == 930
    assert campaign.consumed(runner.SEARCH_STAGES) == 910
    assert campaign.consumed({"screen"}) == 10
    assert campaign.consumed({"confirm"}) == 20


def test_terminal_resource_failure_is_counted_even_after_qualified_fit(tmp_path, registry):
    campaign = lightweight_campaign(tmp_path, registry)
    candidate = registry["tracks"]["A"]["candidates"][0]
    job = campaign.job("screen", "A", candidate, 0, [6000], paths=256)
    atomic_json(tmp_path/"fits"/job["fit_key"]/"receipt.json",
        {"native_fit_identity": "posterior", "diagnostics": {"qualified": True}})
    campaign.ledger.write(job["job_id"], "resource-blocked", job=job)
    schedule = {"trajectories": [0], "origins": [6000], "paths": 256}
    result = campaign.loss("A", candidate, schedule)
    assert result["resource_blocked_fits"] == 1 and result["resource_blocked_origins"] == 1
    assert result["pending_origins"] == 0 and not result["eligible"]
    campaign.ledger.write(job["job_id"], "failed", job=job)
    result = campaign.loss("A", candidate, schedule)
    assert result["failed_outcomes"] == 1 and result["pending_origins"] == 0


def test_partial_screen_cap_allows_only_completed_candidates_to_promote(tmp_path, registry, monkeypatch):
    campaign = lightweight_campaign(tmp_path, registry)
    campaign.cfg["tracks"] = ["A"]
    candidate = registry["tracks"]["A"]["candidates"][0]
    selected = row(candidate["id"], 1., .2, 2.)
    atomic_json(tmp_path/"selection"/"screen_A.json", {"rows": [selected]})
    monkeypatch.setattr(campaign, "screen", lambda: False)
    monkeypatch.setattr(campaign, "budget_reason", lambda stage: "finite_screen_worker_budget" if stage == "screen" else None)
    scheduled = []
    monkeypatch.setattr(campaign, "batch", lambda jobs: scheduled.extend(jobs) or True)
    monkeypatch.setattr(campaign, "rankings", lambda *args: [])
    assert campaign.promote()
    assert len(scheduled) == len(campaign.cfg["promote"]["trajectories"])
    assert {j["candidate"]["id"] for j in scheduled} == {candidate["id"]}


def test_page_cache_advice_preserves_artifact_bytes_and_skips_symlinks(tmp_path):
    folder = tmp_path/"fit"; folder.mkdir()
    np.savez(folder/"fit.npz", value=np.arange(10))
    outside = tmp_path/"outside.npz"; np.savez(outside, value=np.arange(3))
    (folder/"link.npz").symlink_to(outside)
    before = (folder/"fit.npz").read_bytes()
    receipt = runner.evict_closed_artifact_pages(folder)
    assert (folder/"fit.npz").read_bytes() == before
    assert outside.exists() and receipt["files_deleted"] == 0
    if receipt["supported"]: assert receipt["files_advised"] == 1


def test_precision_cap_excludes_incomplete_banks_without_blocking_complete_ones(tmp_path, registry, monkeypatch):
    campaign = lightweight_campaign(tmp_path, registry)
    campaign.cfg["tracks"] = ["A"]
    campaign.cfg["precision"].update(replicates=2, origins=[6000], paths=8)
    candidates = registry["tracks"]["A"]["candidates"][:2]
    rows = [row(c["id"], 1.+i, .2, 2.) for i, c in enumerate(candidates)]
    atomic_json(tmp_path/"selection"/"promote_A.json", {"rows": rows})
    monkeypatch.setattr(campaign, "promote", lambda: True)
    monkeypatch.setattr(campaign, "batch", lambda jobs: False)
    monkeypatch.setattr(campaign, "budget_reason", lambda stage: "finite_precision_worker_budget")
    for i, candidate in enumerate(candidates):
        for repetition in range(2):
            job = campaign.job("precision", "A", candidate, 0, [6000], paths=8,
                forecast_seed=runner.seed_for(campaign.cfg["forecast_seed"], "precision", repetition),
                parameter_seed=runner.seed_for(campaign.cfg["forecast_seed"], "parameter_bank", repetition))
            atomic_json(tmp_path/"fits"/job["fit_key"]/"receipt.json", {"native_fit_identity": candidate["id"]})
            if i == 1 and repetition == 1: continue
            identity = runner._forecast_identity(job, candidate["id"], 6000)
            atomic_json(runner._origin_path(tmp_path, job, 6000, identity), {
                "identity": identity, "state": "complete", "metrics": {"selection_crps": 1.+.01*repetition}})
    assert campaign.precision()
    receipt = runner.read_json(tmp_path/"selection"/"precision_A.json")
    assert receipt["precision_eligible_ids"] == [candidates[0]["id"]]
    assert receipt["excluded_candidates"][candidates[1]["id"]]["complete_repetitions"] == 1
    assert receipt["partial_schedule"]


def test_deferred_run_requires_completed_matching_main_experiment(tmp_path, registry):
    campaign = lightweight_campaign(tmp_path, registry)
    with pytest.raises(RuntimeError, match="main_completion"):
        campaign.deferred()
    atomic_json(tmp_path/"main_completion.json", {"experiment_identity": "a-different-run"})
    with pytest.raises(RuntimeError, match="main_completion"):
        campaign.deferred()
