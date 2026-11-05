"""Small acceptance checks for orchestration boundaries; no scientific fits."""
from __future__ import annotations

import copy
import datetime
import fcntl
import hashlib
import math
import sys
import threading
from types import SimpleNamespace

import pytest

from bayes_stesn.applications import ks_improvement_runner as runner
from bayes_stesn.applications.ks_runtime import Ledger, atomic_json, now
from bayes_stesn.io import content_hash


def _candidate(name, units=4):
    config = {
        "conditioning": 16,
        "features": {"lags": [1], "modes": 1, "units": [units],
                     "squares": 2, "products": 0},
        "spatial_mean": {"side": 2, "centers": None,
                         "coordinate_terms": True},
    }
    return {"id": name, "stratum": "main_fixed_prior", "config": config,
            "effective_hash": content_hash(config)}


@pytest.fixture
def campaign(tmp_path):
    """Construct only the scheduler state needed by the tested methods."""
    c = runner.Campaign.__new__(runner.Campaign)
    candidates = [_candidate("anchor"), _candidate("challenger", units=8)]
    c.registry = {"tracks": {"A": {"candidates": candidates,
                                      "anchor_id": "anchor"}}}
    c.cfg = {
        "registry": c.registry, "tracks": ["A"], "model_seed": 123,
        "forecast_seed": 456, "path_batch": 64,
        "history_candidates": [128, 512],
        "screen": {"trajectories": [0, 5], "origins": [6000, 6500],
                   "paths": 256},
        "promote": {"trajectories": list(runner.SELECTION_TRAJECTORIES),
                    "origins": [6000, 6500], "count": 1, "paths": 512},
        "budget": {"workers": 1, "total_worker_seconds_cap": 1_000,
                   "search_worker_seconds_cap": 500, "wall_seconds_cap": 3600,
                   "disk_reserve_bytes": 0, "estimated_worker_peak_bytes": 1,
                   "combined_worker_cap_bytes": 1_000_000,
                   "job_timeout_seconds": 30},
    }
    c.output = tmp_path
    c.cache = str(tmp_path / "unused-cache")
    c.identity = "fixture-experiment"
    c.source = {"application": {"ks_metrics.py": "fixture-score-source"}}
    c.ledger = Ledger(tmp_path / "ledger")
    atomic_json(tmp_path / "experiment.json", {
        "identity": c.identity, "created_utc": now()})
    return c


def _history(c, *, history=128, excluded=None):
    atomic_json(c.output / "history_policy.json", {
        "history": history, "excluded_candidates": excluded or {}})


def _terminal_receipt(c, job, state="complete", **values):
    """Emulate a worker's durable evidence separately from its status view."""
    result = {
        "job_id": job["job_id"], "state": state,
        "job_identity": content_hash({k: v for k, v in job.items() if k != "job_id"}),
        "finished_utc": now(), **values,
    }
    atomic_json(c.output / "terminal" / f"{job['job_id']}.json", result)
    c.ledger.write(job["job_id"], state, job=job, **values)
    return result


def _fit_receipt(c, candidate, trajectory=0, *, qualified=True):
    key = runner.fit_key("A", candidate, trajectory, c.cfg["model_seed"])
    folder = c.output / "fits" / key
    folder.mkdir(parents=True, exist_ok=True)
    payload = b"fixture-only checkpoint; never loaded as a fitted model"
    (folder / "fit.npz").write_bytes(payload)
    receipt = {
        "fit_key": key, "native_fit_identity": "native-" + key,
        "fit_sha256": hashlib.sha256(payload).hexdigest(),
        "diagnostics": {"qualified": qualified}, "config": candidate["config"],
        "track": "A", "trajectory": trajectory, "seed": c.cfg["model_seed"],
        "candidate_id": candidate["id"], "training_identity": {"fixture": True},
    }
    atomic_json(folder / "receipt.json", receipt)
    return receipt


def _forecast_receipt(c, fit, first=6000, *, state="complete", changes=None,
                      paths=256, forecast_seed=None, namespace="main"):
    history = runner.read_json(c.output / "history_policy.json")["history"]
    forecast_seed = c.cfg["forecast_seed"] if forecast_seed is None else forecast_seed
    identity = content_hash({
        "native_fit": fit["native_fit_identity"], "first": first,
        "history": history, "horizon": 100, "paths": paths,
        "forecast_seed": forecast_seed, "partition": "validation",
        "score_source": c.source["application"]["ks_metrics.py"],
    })
    receipt = {
        "identity": identity, "fit_key": fit["fit_key"],
        "native_fit": fit["native_fit_identity"], "track": "A",
        "candidate_id": fit["candidate_id"], "trajectory": fit["trajectory"],
        "seed": c.cfg["model_seed"], "first_target": first,
        "history": history, "paths": paths, "qualified": True,
        "forecast_seed": forecast_seed, "forecast_namespace": namespace,
        "partition": "validation", "state": state,
        "metrics": {"selection_crps": 0.4, "selection_mse": 0.25},
    }
    receipt.update(changes or {})
    folder = "precision_forecasts" if namespace == "precision" else "forecasts"
    path = c.output / folder / fit["fit_key"] / f"{first:05d}_{paths}_{identity[:12]}.json"
    atomic_json(path, receipt)
    return path


def _preflight_batch(c, outcomes):
    """Return a stand-in executor with explicit candidate/trajectory outcomes."""
    def execute(jobs, workers=None):
        for job in jobs:
            passed = outcomes[(job["candidate"]["id"], job["trajectory"])]
            if passed is None:
                _terminal_receipt(c, job, "failed")
            else:
                rows = [{"history": length, "passed": length in passed}
                        for length in c.cfg["history_candidates"]]
                _terminal_receipt(c, job, "complete",
                    result={"synchronization": {"rows": rows}})
        return True
    return execute


def test_all_failed_preflights_do_not_publish_a_history(campaign, monkeypatch):
    outcomes = {(candidate["id"], trajectory): None
                for candidate in campaign.registry["tracks"]["A"]["candidates"]
                for trajectory in campaign.cfg["screen"]["trajectories"]}
    monkeypatch.setattr(campaign, "batch", _preflight_batch(campaign, outcomes))
    with pytest.raises(RuntimeError, match="common measured history"):
        campaign.preflight()
    assert not (campaign.output / "history_policy.json").exists()


def test_excluded_candidate_does_not_restrict_common_history(campaign, monkeypatch):
    # The anchor fails on one trajectory. Its other trajectory must not constrain
    # the surviving challenger's valid 512-observation history.
    outcomes = {("anchor", 0): {128}, ("anchor", 5): None,
                ("challenger", 0): {512}, ("challenger", 5): {512}}
    monkeypatch.setattr(campaign, "batch", _preflight_batch(campaign, outcomes))
    assert campaign.preflight()
    policy = runner.read_json(campaign.output / "history_policy.json")
    assert policy["history"] == 512
    assert "A/anchor" in policy["excluded_candidates"]


def test_screen_does_not_fit_preflight_excluded_candidates(campaign, monkeypatch):
    _history(campaign, excluded={"A/anchor": "training_preflight_failed"})
    captured = []
    monkeypatch.setattr(campaign, "batch", lambda jobs: captured.extend(jobs) or True)
    monkeypatch.setattr(campaign, "rankings", lambda *args, **kwargs: [])
    assert campaign.screen()
    assert captured
    assert {j["candidate"]["id"] for j in captured} == {"challenger"}


def test_promotion_does_not_reintroduce_an_excluded_anchor(campaign, monkeypatch):
    _history(campaign, excluded={"A/anchor": "training_preflight_failed"})
    atomic_json(campaign.output / "selection" / "screen_A.json", {"rows": [
        {"id": "challenger", "stratum": "main_fixed_prior", "eligible": True},
        {"id": "anchor", "stratum": "main_fixed_prior", "eligible": False},
    ]})
    captured = []
    monkeypatch.setattr(campaign, "batch", lambda jobs: captured.extend(jobs) or True)
    monkeypatch.setattr(campaign, "rankings", lambda *args, **kwargs: [])
    assert campaign.promote()
    assert captured
    assert {j["candidate"]["id"] for j in captured} == {"challenger"}


def test_preflight_job_identity_does_not_change_after_history_selection(campaign):
    candidate = campaign.registry["tracks"]["A"]["candidates"][0]
    before = campaign.job("preflight", "A", candidate, 0)
    _history(campaign, history=512)
    after = campaign.job("preflight", "A", candidate, 0)
    assert before["job_id"] == after["job_id"]
    assert after["history"] is None


def test_budget_stop_propagates_without_reading_missing_stage_files(campaign, monkeypatch):
    monkeypatch.setattr(campaign, "screen", lambda: False)
    assert campaign.promote() is False
    monkeypatch.setattr(campaign, "promote", lambda: False)
    assert campaign.freeze() is None
    monkeypatch.setattr(campaign, "freeze", lambda: None)
    assert campaign.confirm() is False
    assert not (campaign.output / "freeze.json").exists()


@pytest.mark.parametrize("changes", [
    {"history": 512}, {"native_fit": "another-posterior"},
    {"qualified": False}, {"partition": "test"},
])
def test_incompatible_forecasts_cannot_enter_selection(campaign, changes):
    _history(campaign)
    candidate = campaign.registry["tracks"]["A"]["candidates"][0]
    fit = _fit_receipt(campaign, candidate)
    _forecast_receipt(campaign, fit, changes=changes)
    with pytest.raises(ValueError, match="incompatible forecast receipt"):
        campaign.loss("A", candidate, [0], [6000])


def test_missing_origin_preserves_the_expected_denominator(campaign):
    _history(campaign)
    candidate = campaign.registry["tracks"]["A"]["candidates"][0]
    fit = _fit_receipt(campaign, candidate)
    _forecast_receipt(campaign, fit)
    result = campaign.loss("A", candidate, [0], [6000, 6500])
    assert not result["eligible"]
    assert result["loss"] is None
    assert result["pending_origins"] == 1
    assert result["expected_episodes"] == 1
    assert result["qualified_episodes"] == 0


def test_failed_origin_is_not_dropped_from_selection(campaign):
    _history(campaign)
    candidate = campaign.registry["tracks"]["A"]["candidates"][0]
    fit = _fit_receipt(campaign, candidate)
    _forecast_receipt(campaign, fit)
    _forecast_receipt(campaign, fit, 6500, state="failed")
    result = campaign.loss("A", candidate, [0], [6000, 6500])
    assert not result["eligible"]
    assert result["failed_origins"] == 1
    assert result["pending_origins"] == 0
    assert result["loss"] is None


def test_unqualified_fit_is_distinct_from_a_pending_fit(campaign):
    _history(campaign)
    candidate = campaign.registry["tracks"]["A"]["candidates"][0]
    _fit_receipt(campaign, candidate, qualified=False)
    result = campaign.loss("A", candidate, [0], [6000, 6500])
    assert not result["eligible"]
    assert result["inference_unqualified_fits"] == 1
    assert result["pending_origins"] == 0
    assert result["loss"] is None


def test_confirmation_jobs_use_only_the_remaining_trajectories(campaign, monkeypatch):
    _history(campaign)
    freeze = {"selected": {"A": {"main": {"id": "challenger"},
                                     "anchor_id": "anchor"}},
              "confirmation_origins": list(range(6000, 8000, 100)), "paths": 512}
    monkeypatch.setattr(campaign, "freeze", lambda: freeze)
    monkeypatch.setattr(campaign, "summarize", lambda: None)
    captured = []
    monkeypatch.setattr(campaign, "batch", lambda jobs: captured.extend(jobs) or True)
    assert campaign.confirm()
    actual = {j["trajectory"] for j in captured}
    screened = set(campaign.cfg["screen"]["trajectories"])
    promoted = set(campaign.cfg["promote"]["trajectories"])
    assert actual == set(range(20)) - screened - promoted
    assert len(actual) == 14
    assert len(captured) == 28
    assert all(j["partition"] == "validation" and j["paths"] == 512 for j in captured)


def test_frozen_metadata_tampering_is_rejected(campaign):
    _history(campaign)
    candidate = campaign.registry["tracks"]["A"]["candidates"][0]
    row = {"id": candidate["id"], "stratum": candidate["stratum"],
           "eligible": True, "loss": 0.5, "configuration": candidate["config"]}
    atomic_json(campaign.output / "selection" / "promote_A.json", {"rows": [row]})
    freeze = campaign.freeze()
    assert freeze is not None
    changed = copy.deepcopy(freeze)
    changed["history"] += 1
    atomic_json(campaign.output / "freeze.json", changed)
    with pytest.raises(ValueError):
        campaign.freeze()


def test_search_budget_preserves_confirmation_allocation(campaign):
    atomic_json(campaign.output / "attempts" / "search.json", {
        "stage": "screen", "elapsed_seconds": 500})
    assert campaign.budget_reason("screen") == "search_budget_final_confirmation_reserved"
    assert campaign.budget_reason("confirm") is None


@pytest.mark.parametrize("state", ["complete", "failed", "inference-unqualified"])
def test_terminal_jobs_are_not_reexecuted(campaign, monkeypatch, state):
    _history(campaign)
    candidate = campaign.registry["tracks"]["A"]["candidates"][0]
    job = campaign.job("screen", "A", candidate, 0, [6000])
    _terminal_receipt(campaign, job, state)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("terminal job was rerun"))
    assert campaign.submit(job)["state"] == state


def test_stale_running_ledger_cannot_override_terminal_completion(campaign, monkeypatch):
    _history(campaign)
    candidate = campaign.registry["tracks"]["A"]["candidates"][0]
    job = campaign.job("screen", "A", candidate, 0, [6000])
    completed = _terminal_receipt(campaign, job)
    campaign.ledger.write(job["job_id"], "running", job=job, pid=123456)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("completed work was rerun"))
    assert campaign.submit(job) == completed
    monkeypatch.setattr(campaign, "submit", lambda *a: pytest.fail("batch ignored durable completion"))
    assert campaign.batch([job]) is True
    assert campaign.ledger.read(job["job_id"])["state"] == "running"


def test_successful_process_without_terminal_receipt_is_operational_error(campaign, monkeypatch):
    _history(campaign)
    candidate = campaign.registry["tracks"]["A"]["candidates"][0]
    job = campaign.job("screen", "A", candidate, 0, [6000])
    process = SimpleNamespace(pid=123456, returncode=0, poll=lambda: 0)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: process)
    with pytest.raises(RuntimeError, match="without an immutable terminal receipt"):
        campaign.submit(job)
    assert campaign.ledger.read(job["job_id"])["state"] == "pending"
    assert runner.terminal_result(campaign.output, job) is None


def test_terminal_receipt_for_changed_job_is_rejected(campaign):
    _history(campaign)
    candidate = campaign.registry["tracks"]["A"]["candidates"][0]
    job = campaign.job("screen", "A", candidate, 0, [6000])
    _terminal_receipt(campaign, job)
    incompatible = copy.deepcopy(job)
    incompatible["history"] = 512
    with pytest.raises(ValueError, match="Incompatible terminal job receipt"):
        runner.terminal_result(campaign.output, incompatible)


def test_missing_preflight_terminal_is_not_a_scientific_exclusion(campaign, monkeypatch):
    def incomplete_execution(jobs):
        for job in jobs:
            campaign.ledger.write(job["job_id"], "complete", job=job)
        return True

    monkeypatch.setattr(campaign, "batch", incomplete_execution)
    with pytest.raises(RuntimeError, match="Incomplete preflight execution"):
        campaign.preflight()
    assert not (campaign.output / "history_policy.json").exists()


def test_second_scheduler_cannot_start(campaign, monkeypatch):
    monkeypatch.setattr(runner, "Campaign", lambda *args: campaign)
    monkeypatch.setattr(campaign, "screen", lambda: pytest.fail("duplicate scheduler entered screen"))
    monkeypatch.setattr(sys, "argv", ["runner", "screen", "--config", "unused",
                                     "--cache", "unused", "--output", str(campaign.output)])
    with (campaign.output / "scheduler.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            runner.main()


def test_wall_budget_is_distinct_from_worker_budget(campaign):
    created = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=2)
    campaign.cfg["budget"]["wall_seconds_cap"] = 3600
    atomic_json(campaign.output / "experiment.json", {
        "identity": campaign.identity, "created_utc": created.isoformat()})
    assert campaign.consumed() == 0
    assert campaign.budget_reason("confirm") == "finite_campaign_wall_budget"


def test_migration_cannot_restart_the_original_campaign_wall_budget(campaign):
    original_start = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=2)
    campaign.cfg["budget"]["campaign_started_utc"] = original_start.isoformat()
    campaign.cfg["budget"]["wall_seconds_cap"] = 3600
    # The new workspace's experiment receipt was created just now by the fixture.
    assert campaign.budget_reason("confirm") == "finite_campaign_wall_budget"


def test_repeated_jobs_for_one_fit_are_serialized(campaign, monkeypatch):
    """Independent fits can overlap, but replicas of the same fit cannot."""
    campaign.cfg["budget"]["workers"] = 4
    monkeypatch.setattr(campaign, "budget_reason", lambda stage: None)
    monkeypatch.setattr(runner, "admission", lambda *args, **kwargs: {})
    jobs = [{"job_id": f"{key}-rep{rep}", "fit_key": key, "stage": "precision"}
            for key in ("fit-a", "fit-b") for rep in range(4)]
    guard = threading.Lock()
    rendezvous = threading.Barrier(2, timeout=5)
    active = set()
    finished = []
    maximum_active = 0

    def submit(job):
        nonlocal maximum_active
        with guard:
            assert job["fit_key"] not in active
            active.add(job["fit_key"])
            maximum_active = max(maximum_active, len(active))
        rendezvous.wait()
        with guard:
            active.remove(job["fit_key"])
            finished.append(job["job_id"])
        return _terminal_receipt(campaign, job)

    monkeypatch.setattr(campaign, "submit", submit)
    assert campaign.batch(jobs)
    assert maximum_active == 2
    assert sorted(finished) == sorted(j["job_id"] for j in jobs)


def _precision_setup(c, scores, promotion_losses):
    _history(c)
    c.cfg["precision"] = {"enabled": True, "trajectory": 0,
        "origins": [6000, 7500], "paths": 512, "replicates": 4}
    candidates = c.registry["tracks"]["A"]["candidates"]
    fits = {candidate["id"]: _fit_receipt(c, candidate) for candidate in candidates}
    rows = [{"id": candidate["id"], "stratum": candidate["stratum"],
             "eligible": True, "loss": promotion_losses[candidate["id"]],
             "configuration": candidate["config"]} for candidate in candidates]
    atomic_json(c.output / "selection" / "promote_A.json", {"rows": rows})
    repetition = {runner.seed_for(c.cfg["forecast_seed"], "precision", k): k
                  for k in range(4)}
    captured = []

    def batch(jobs):
        captured.extend(jobs)
        for job in jobs:
            rep = repetition[job["forecast_seed"]]
            value = scores[job["candidate"]["id"]][rep]
            for first in job["origins"]:
                _forecast_receipt(c, fits[job["candidate"]["id"]], first,
                    paths=job["paths"], forecast_seed=job["forecast_seed"],
                    namespace="precision", changes={"metrics": {
                        "selection_crps": value, "selection_mse": 0.25}})
        return True

    return batch, captured


def test_precision_uses_paired_streams_and_known_difference_sd(campaign, monkeypatch):
    scores = {"anchor": [0.50, 0.54, 0.50, 0.54],
              "challenger": [0.50, 0.50, 0.50, 0.50]}
    batch, captured = _precision_setup(campaign, scores,
        {"anchor": 0.52, "challenger": 0.50})
    monkeypatch.setattr(campaign, "batch", batch)
    assert campaign.precision()
    receipt = runner.read_json(campaign.output / "selection" / "precision_A.json")
    assert receipt["paired_differences"][0]["paired_difference_mc_sd"] == pytest.approx(
        0.04 / math.sqrt(3))
    by_candidate = {
        name: {j["forecast_seed"] for j in captured if j["candidate"]["id"] == name}
        for name in scores}
    assert by_candidate["anchor"] == by_candidate["challenger"]
    assert len(by_candidate["anchor"]) == 4
    assert {j["seed"] for j in captured} == {campaign.cfg["model_seed"]}


def test_precision_uncertainty_can_make_a_cheaper_candidate_a_near_tie(campaign, monkeypatch):
    batch, _ = _precision_setup(campaign,
        {"anchor": [0.50, 0.54, 0.50, 0.54], "challenger": [0.50] * 4},
        {"anchor": 0.52, "challenger": 0.50})
    monkeypatch.setattr(campaign, "batch", batch)
    frozen = campaign.freeze()
    assert frozen["selected"]["A"]["main"]["id"] == "anchor"
    assert set(frozen["selected"]["A"]["near_tie_ids"]) == {"anchor", "challenger"}


def test_unrelated_noisy_candidate_does_not_expand_other_near_ties(campaign, monkeypatch):
    # The anchor is reliably worse than the best candidate. A third model's
    # large simulation variance must not turn that precise contrast into a tie.
    campaign.registry["tracks"]["A"]["candidates"].append(_candidate("unstable", units=16))
    batch, _ = _precision_setup(campaign,
        {"anchor": [0.52] * 4, "challenger": [0.50] * 4,
         "unstable": [100.0, 300.0, 50.0, 250.0]},
        {"anchor": 0.52, "challenger": 0.50, "unstable": 100.0})
    monkeypatch.setattr(campaign, "batch", batch)
    frozen = campaign.freeze()
    assert frozen["selected"]["A"]["main"]["id"] == "challenger"
    assert "anchor" not in frozen["selected"]["A"]["near_tie_ids"]


@pytest.mark.parametrize("changes", [
    {"history": 512}, {"qualified": False}, {"native_fit": "foreign-posterior"},
])
def test_precision_rejects_incompatible_forecast_receipts(campaign, monkeypatch, changes):
    batch, _ = _precision_setup(campaign,
        {"anchor": [0.52] * 4, "challenger": [0.50] * 4},
        {"anchor": 0.52, "challenger": 0.50})

    def changed_batch(jobs):
        batch(jobs)
        path = next((campaign.output / "precision_forecasts").glob("*/*.json"))
        value = runner.read_json(path)
        value.update(changes)
        atomic_json(path, value)
        return True

    monkeypatch.setattr(campaign, "batch", changed_batch)
    with pytest.raises((ValueError, RuntimeError)):
        campaign.precision()


def test_precision_requires_each_origin_once_not_only_the_receipt_count(campaign, monkeypatch):
    batch, _ = _precision_setup(campaign,
        {"anchor": [0.52] * 4, "challenger": [0.50] * 4},
        {"anchor": 0.52, "challenger": 0.50})

    def duplicate_origin_batch(jobs):
        batch(jobs)
        for path in (campaign.output / "precision_forecasts").glob("*/07500_*.json"):
            path.unlink()
        for path in list((campaign.output / "precision_forecasts").glob("*/06000_*.json")):
            path.with_name("duplicate_" + path.name).write_bytes(path.read_bytes())
        return True

    monkeypatch.setattr(campaign, "batch", duplicate_origin_batch)
    with pytest.raises((ValueError, RuntimeError)):
        campaign.precision()
