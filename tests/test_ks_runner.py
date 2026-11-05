"""Targeted KS orchestration checks using synthetic data and cheap anchors.

No delivered test responses, neural training, or posterior fits are used.
The native sampler's exact RNG recovery is tested in test_ks_native.py.
"""
from __future__ import annotations

import copy
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayes_stesn.applications import ks_data, ks_runner, ks_runtime
from bayes_stesn.benchmark.contracts import Panel
from bayes_stesn.io import content_hash


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """An adapter stand-in makes forbidden data access directly observable."""
    angle = np.arange(4) * 2 * np.pi / 4
    coordinates = np.column_stack((np.cos(angle), np.sin(angle)))
    times = .25 * np.arange(32)
    values = np.sin(.1 * np.arange(32)[:, None] + angle)
    panel = Panel(values, coordinates, times, tuple(map(str, range(4))))
    events = []

    class Adapter:
        def __init__(self, cache, protocol, phase):
            self.protocol, self.phase = protocol, phase

        def training(self, trajectory):
            events.append("training")
            identity = {"synthetic_fixture": True, "protocol": self.protocol,
                        "phase": self.phase, "trajectory": trajectory}
            return SimpleNamespace(panel=panel, identity=identity,
                transform=SimpleNamespace(scale=1., inverse=lambda x: np.asarray(x)))

        def model_window(self, trajectory, first_target, partition, *, history, horizon):
            events.append("model_history")
            history_panel = Panel(panel.y[-history:], panel.coordinates,
                                  panel.times[-history:], panel.site_ids)
            return SimpleNamespace(panel=history_panel,
                target_coordinates=coordinates, target_site_ids=panel.site_ids,
                target_x=angle, x=angle,
                target_time=.25 * np.arange(first_target, first_target+horizon))

        def evaluation_window(self, trajectory, first_target, partition, *, history, horizon):
            events.append("evaluation_labels")
            return SimpleNamespace(values=np.zeros((horizon, 4)))

    monkeypatch.setattr(ks_data, "KSAdapter", Adapter)
    monkeypatch.setattr(ks_runner, "numerical_identity", lambda: {"fixture_source": "v1"})
    candidate = {"id": "mean_0", "config": {}, "complexity_order": 0}
    config = {"base_revision": "synthetic-fixture", "history": 8, "horizon": 6,
        "paths": 256, "path_batch": 64, "model_seed": 17, "forecast_seed": 23,
        "test_origins": [32], "plot_cases": [], "scoring": {},
        "sensitivity": {"seeds": [18, 19], "trajectories": [0]},
        "robustness": {"models": ["training_mean"]},
        "candidates": {t: {"training_mean": [candidate]} for t in ("A", "B")},
        "budget": {"workers": 1, "disk_reserve_bytes": 0,
            "initial_total_wall_hours": 1., "search_wall_hours_cap": .25,
            "total_worker_seconds_cap": 100., "search_worker_seconds_cap": 50.}}
    config_path = tmp_path/"config.json"
    ks_runtime.atomic_json(config_path, config)
    study = ks_runner.Study(config_path, tmp_path/"cache", tmp_path/"output")
    job = study.make_job("screen", "A", "training_mean", candidate, 0, [32])
    events.clear()
    return SimpleNamespace(study=study, job=job, candidate=candidate,
        events=events, panel=panel, config_path=config_path)


def install_test_gate(harness):
    study, job = harness.study, harness.job
    frozen = {"experiment_identity": study.identity,
        "selected": {t: {"training_mean": harness.candidate} for t in ("A", "B")},
        "test_origins": job["origins"], "test_trajectories": [0],
        "history": job["history"], "horizon": job["horizon"], "paths": job["paths"],
        "model_seed": job["seed"], "sensitivity_seeds": [18, 19],
        "sensitivity_trajectories": [0], "robustness_methods": ["training_mean"]}
    frozen["identity"] = content_hash(frozen)
    ks_runtime.atomic_json(study.output/"freeze.json", frozen)
    ks_runtime.atomic_json(study.output/"test_access.json",
        {"freeze_identity": frozen["identity"], "both_track_selection_complete": True})
    return frozen


def test_test_gate_blocks_before_any_model_or_label_read(harness):
    job = {**harness.job, "partition": "test"}
    with pytest.raises(ValueError, match="Test access requires"):
        ks_runner.execute_job(job)
    assert harness.events == []


@pytest.mark.parametrize("changed,value", [
    ("config", {"unselected": True}), ("origins", [33]),
    ("history", 9), ("horizon", 7), ("paths", 512),
    ("seed", 18), ("trajectory", 1),
])
def test_frozen_test_contract_rejects_changes_before_reads(harness, changed, value):
    install_test_gate(harness)
    job = {**harness.job, "partition": "test", changed: value}
    with pytest.raises(ValueError, match="Test |configuration"):
        ks_runner.execute_job(job)
    assert harness.events == []


def test_source_change_blocks_before_any_data_read(harness):
    with pytest.raises(ValueError, match="numerical source changed"):
        ks_runner.execute_job({**harness.job, "source": {"fixture_source": "v2"}})
    assert harness.events == []


def test_test_jobs_use_the_frozen_escalated_path_budget(harness):
    frozen = install_test_gate(harness)
    frozen["paths"] = 512
    ks_runtime.atomic_json(harness.study.output/"freeze.json", frozen)
    test_job = harness.study.make_job("final", "A", "training_mean",
        harness.candidate, 0, [32], partition="test")
    validation_job = harness.study.make_job("promote", "A", "training_mean",
        harness.candidate, 0, [32], partition="validation")
    assert test_job["paths"] == 512
    assert validation_job["paths"] == 256
    assert test_job["fit_id"] == validation_job["fit_id"]


def test_completed_fit_and_origin_reused_and_corruption_not_retrained(harness, monkeypatch):
    job, study = harness.job, harness.study
    ks_runner.execute_job(job)
    assert harness.events.index("model_history") < harness.events.index("evaluation_labels")
    first_receipt = study.ledger.read(job["job_id"])
    assert first_receipt["state"] == "complete"
    artifact = study.output/"fits"/job["fit_id"]/"fit.npz"
    original_hash = ks_runtime.file_hash(artifact)

    def must_not_refit(*args, **kwargs):
        raise AssertionError("A completed fit must not be retrained")

    monkeypatch.setattr(ks_runner, "_fit", must_not_refit)
    harness.events.clear()
    ks_runner.execute_job(job)
    receipt = study.ledger.read(job["job_id"])
    assert receipt["state"] == "complete" and receipt["reused_fit"]
    assert receipt["reused_origins"] == 1
    assert harness.events == ["training"]
    assert ks_runtime.file_hash(artifact) == original_hash

    with artifact.open("ab") as stream:
        stream.write(b"intentional-test-corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        ks_runner.execute_job(job)
    assert study.ledger.read(job["job_id"])["state"] == "pending"
    assert (study.output/"fits"/job["fit_id"]/"receipt.json").exists()


def test_infrastructure_interruption_retains_pending_checkpoint_then_resumes(harness, monkeypatch):
    job, study = harness.job, harness.study
    original_fit = ks_runner._fit
    checkpoint = study.output/"retained-rng-checkpoint.bin"

    def interrupted_fit(*args, **kwargs):
        checkpoint.write_bytes(b"already-durable-state")
        raise InterruptedError("simulated worker interruption")

    monkeypatch.setattr(ks_runner, "_fit", interrupted_fit)
    with pytest.raises(InterruptedError):
        ks_runner.execute_job(job)
    assert study.ledger.read(job["job_id"])["state"] == "pending"
    assert not (study.output/"fits"/job["fit_id"]/"receipt.json").exists()
    assert checkpoint.read_bytes() == b"already-durable-state"
    assert "evaluation_labels" not in harness.events

    monkeypatch.setattr(ks_runner, "_fit", original_fit)
    ks_runner.execute_job(job)
    assert study.ledger.read(job["job_id"])["state"] == "complete"
    assert checkpoint.read_bytes() == b"already-durable-state"


def test_numerical_fit_failure_is_recorded_and_not_rerolled(harness, monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise FloatingPointError("nonfinite fitted numerical state")

    monkeypatch.setattr(ks_runner, "_fit", fail)
    ks_runner.execute_job(harness.job)
    first = harness.study.ledger.read(harness.job["job_id"])
    assert first["state"] == "failed"
    ks_runner.execute_job(harness.job)
    second = harness.study.ledger.read(harness.job["job_id"])
    assert second["state"] == "failed" and second["reused_failure"]
    assert len(calls) == 1
    assert "evaluation_labels" not in harness.events


def test_active_fit_has_only_one_writer(harness):
    path = harness.study.output/"fits"/harness.job["fit_id"]/"writer.lock"
    path.parent.mkdir(parents=True)
    with path.open("a+") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="live worker"):
            ks_runner.execute_job(harness.job)
    assert harness.events == []


def test_unqualified_and_missing_origins_remain_in_selection_denominator(harness):
    ks_runner.execute_job(harness.job)
    path = ks_runner.forecast_path(harness.study.output, harness.job, 32)
    result = json.loads(path.read_text())
    result["qualified"] = False
    ks_runtime.atomic_json(path, result)
    ranking = harness.study.candidate_loss("A", "training_mean", harness.candidate, [0], [32])
    assert ranking["loss"] is not None and not ranking["eligible"]
    assert ranking["qualified_episodes"] == 0 and ranking["failed_origins"] == 0
    missing = harness.study.candidate_loss("A", "training_mean", harness.candidate, [0], [32, 38])
    assert missing["loss"] is None and not missing["eligible"]
    assert missing["failed_origins"] == 1
    assert missing["episodes"][0]["complete_origins"] == 1


def test_freeze_requires_precision_and_both_track_selection(harness):
    study = harness.study
    with pytest.raises(ValueError, match="Monte Carlo precision"):
        study.freeze()
    ks_runtime.atomic_json(study.output/"validation_precision.json", {"selected_paths": 256})
    ranking = {"ranking": [{"candidate": harness.candidate, "eligible": True,
        "loss": .2, "objective": "episode_balanced_horizon_band_MSE"}]}
    ks_runtime.atomic_json(study.output/"selection"/"A_training_mean_promote.json", ranking)
    with pytest.raises(ValueError, match="Both main tracks"):
        study.freeze()
    assert not (study.output/"freeze.json").exists()
    ks_runtime.atomic_json(study.output/"selection"/"B_training_mean_promote.json", ranking)
    frozen = study.freeze()
    assert set(frozen["selected"]) == {"A", "B"}
    assert frozen["paths"] == 256
    assert not (study.output/"test_access.json").exists()


def test_compute_budget_persists_across_a_new_scheduler(harness):
    study = harness.study
    for name, stage, seconds in [("first", "screen", 30.), ("second", "promote", 25.)]:
        ks_runtime.atomic_json(study.output/"attempts"/(name+".json"),
            {"state": "interrupted", "stage": stage, "elapsed_seconds": seconds})
    resumed = ks_runner.Study(harness.config_path, study.cache, study.output)
    assert resumed.consumed_worker_seconds() == 55.
    assert resumed.consumed_worker_seconds({"final"}) == 0.
    with pytest.raises(StopIteration, match="Cumulative screening"):
        resumed.submit(harness.job)
    receipt = resumed.ledger.read(harness.job["job_id"])
    assert receipt["state"] == "pending"
    assert receipt["reason"] == "cumulative_screening_compute_cap"


def test_resource_admission_uses_container_headroom_and_preserves_disk_reserve(monkeypatch, tmp_path):
    snapshot = {"disk_free_bytes": 20_000, "memory": {
        "memory.max": "10000", "memory.current": "9000", "inactive_file_bytes": 1000}}
    monkeypatch.setattr(ks_runtime, "resource_snapshot", lambda path: copy.deepcopy(snapshot))
    with pytest.raises(RuntimeError, match="Memory admission"):
        ks_runtime.admission(tmp_path, reserve_bytes=1000, estimated_peak_bytes=651,
                             worker_cap_bytes=999_999)
    allowed = ks_runtime.admission(tmp_path, reserve_bytes=1000,
                                   estimated_peak_bytes=650, worker_cap_bytes=999_999)
    assert allowed["raw_container_headroom_bytes"] == 1000
    assert allowed["current_worker_allowance_bytes"] == 975
    snapshot["disk_free_bytes"] = 999
    with pytest.raises(RuntimeError, match="Disk reserve"):
        ks_runtime.admission(tmp_path, reserve_bytes=1000, estimated_peak_bytes=0)
