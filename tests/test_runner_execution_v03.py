"""Bounded orchestration fault injection and independent resource contracts."""
from dataclasses import replace
import json

import numpy as np
import pytest

from bayes_stesn.benchmark import profile
from bayes_stesn.benchmark import execution, runner


GIB = 1024**3


def _resources(current=GIB):
    return {"memory_limit_bytes": 8*GIB, "memory_current_bytes": current,
            "effective_cpus": 2, "worker_cap": 2, "memory_measurement": "test fixture"}


def _tiny(**changes):
    options = dict(models=("persistence", "seasonal_naive"), tracks=("A",),
                   scenarios=("g1_gaussian_ar",), initial_fit=60, replications=2)
    options.update(changes)
    return profile("v03_smoke", **options)


def _receipts(directory):
    return {str(path.relative_to(directory)): json.loads(path.read_text())
            for path in sorted(directory.rglob("receipt.json"))}


def _assert_same_numerical_results(first, second):
    assert set(first) == set(second)
    for key in first:
        for field in ("qualified", "fallback", "operational_scores", "native_scores",
                      "final_fit_seed", "forecast_seed", "training_scale",
                      "normalization_reference", "manifest_identity", "computation_identity"):
            assert first[key][field] == second[key][field], (key, field)


def test_job_chunk_resume_changes_workers_without_changing_science(tmp_path, monkeypatch):
    # Inject host capabilities only; the second phase exercises real spawn.
    monkeypatch.setattr(execution, "host_resources", lambda: _resources())
    p = _tiny(models=("persistence", "eqesn_rollout"),
              model_options={"eqesn_rollout": {"width": 9, "washout": 6}})
    uninterrupted = runner.run_benchmark(p, tmp_path/"uninterrupted", workers=1)
    chunk = tmp_path/"chunked"
    first = runner.run_benchmark(p, chunk, workers=1, max_jobs=1)
    assert first["jobs"] == 1 and first["pending_jobs"] == 3 and not first["complete"]
    manifest_before = (chunk/"manifest.json").read_bytes()
    retained_path = next(chunk.rglob("receipt.json"))
    retained = retained_path.read_bytes()
    second = runner.run_benchmark(p, chunk, workers=2, max_memory_mb=4096)
    assert second["complete"] and second["jobs"] == uninterrupted["jobs"] == 4
    assert (chunk/"manifest.json").read_bytes() == manifest_before
    assert retained_path.read_bytes() == retained
    _assert_same_numerical_results(_receipts(tmp_path/"uninterrupted"), _receipts(chunk))
    attempts = [json.loads(path.read_text()) for path in (chunk/"execution_attempts").glob("*.json")]
    assert {attempt["requested_workers"] for attempt in attempts} == {1, 2}
    assert len({attempt["manifest_identity"] for attempt in attempts}) == 1
    assert runner.run_benchmark(p, chunk, workers=1) == second


def test_transient_worker_loss_reuses_seed_and_preserves_failure_history(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "host_resources", lambda: _resources())
    p = _tiny(models=("persistence",), replications=1)
    expected = runner.run_benchmark(p, tmp_path/"reference")
    original = runner._execute_job
    seeds = []

    def transient(job, progress=None):
        seeds.append(job["seed"])
        if len(seeds) == 1:
            runner._record_state(job, "running", started=True, phase="final_fit")
            raise RuntimeError("injected transient worker loss")
        return original(job, progress)

    monkeypatch.setattr(runner, "_execute_job", transient)
    recovered = runner.run_benchmark(p, tmp_path/"recovered", max_infrastructure_retries=1)
    assert recovered["complete"] and recovered["jobs"] == expected["jobs"] == 1
    assert seeds[0] == seeds[1]
    _assert_same_numerical_results(_receipts(tmp_path/"reference"), _receipts(tmp_path/"recovered"))
    state = json.loads(next((tmp_path/"recovered").rglob("state.json")).read_text())
    assert state["attempts"] == 2 and len(state["failures"]) == 1
    assert state["failures"][0]["started"]
    assert not any(row["fallback"] for row in recovered["table"])


def test_persistent_infrastructure_is_incomplete_without_fabricated_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "host_resources", lambda: _resources())
    p = _tiny(replications=1)
    original = runner._execute_job
    attempted = []

    def broken(job, progress=None):
        if job["model_id"] == "persistence":
            attempted.append(job["seed"])
            raise RuntimeError("injected persistent infrastructure failure")
        return original(job, progress)

    monkeypatch.setattr(runner, "_execute_job", broken)
    result = runner.run_benchmark(p, tmp_path, max_infrastructure_retries=1)
    assert not result["complete"] and result["jobs"] == 1 and result["pending_jobs"] == 1
    assert result["unresolved_infrastructure_failures"] == 1
    assert {row["model_id"] for row in result["table"]} == {"seasonal_naive"}
    assert len(attempted) == 2 and len(set(attempted)) == 1
    assert not list(tmp_path.glob("*/rep_*/A/persistence/receipt.json"))
    # The retry budget applies across process restarts, not afresh on each call.
    runner.run_benchmark(p, tmp_path, max_infrastructure_retries=1)
    assert len(attempted) == 2
    monkeypatch.setattr(runner, "_execute_job", original)
    recovered = runner.run_benchmark(p, tmp_path, max_infrastructure_retries=2)
    assert recovered["complete"] and recovered["jobs"] == 2
    state = json.loads(next(tmp_path.glob("*/rep_*/A/persistence/state.json")).read_text())
    assert state["attempts"] == 3 and len(state["failures"]) == 2


@pytest.mark.parametrize("kind", ("receipt", "artifact", "dataset"))
def test_cached_tamper_is_rejected_and_persists_invalidated_status(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(execution, "host_resources", lambda: _resources())
    p = _tiny(models=("persistence",), replications=1)
    runner.run_benchmark(p, tmp_path)
    receipt_path = next(tmp_path.rglob("receipt.json"))
    if kind == "receipt":
        receipt = json.loads(receipt_path.read_text())
        receipt["operational_scores"]["rmse"] += 1
        receipt_path.write_text(json.dumps(receipt))
    elif kind == "artifact":
        path = receipt_path.parent/"model.npz"
        path.write_bytes(path.read_bytes()+b"changed")
    else:
        # A validly serialized changed dataset still must fail its sealed
        # byte/content provenance; the test does not rely on ZIP corruption.
        from bayes_stesn.benchmark.generators import Simulation
        path = next(tmp_path.rglob("simulation.npz"))
        sim = Simulation.load(path)
        observations = sim.observation.copy();observations[0, 0] += .5
        replace(sim, observation=observations).save(path)
    with pytest.raises(RuntimeError, match="changed|provenance"):
        runner.run_benchmark(p, tmp_path)
    summary = json.loads((tmp_path/"summary.json").read_text())
    assert not summary["complete"]
    assert summary["comparison_status"].startswith("invalidated")
    assert runner.benchmark_status(tmp_path)["summary"]["complete"] is False


def test_worker_runtime_mismatch_invalidates_before_model_fitting(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "host_resources", lambda: _resources())
    p = _tiny(models=("persistence",), replications=1)
    original = runner._execute_job

    def changed_worker(job, progress=None):
        altered = dict(runner.benchmark_environment())
        altered["numpy"] = "injected-incompatible-runtime"
        with monkeypatch.context() as inside_worker:
            inside_worker.setattr(runner, "benchmark_environment", lambda: altered)
            return original(job, progress)

    def forbidden(*args, **kwargs):
        raise AssertionError("An incompatible worker reached model fitting")

    monkeypatch.setattr(runner, "_execute_job", changed_worker)
    monkeypatch.setattr(runner, "fit_model", forbidden)
    with pytest.raises(runner.StudyInvalidatedError):
        runner.run_benchmark(p, tmp_path)
    assert not list(tmp_path.rglob("receipt.json"))
    state = json.loads(next(tmp_path.rglob("state.json")).read_text())
    assert state["status"] == "invalidated" and "numerical runtime" in state["error"]
    assert not runner.benchmark_status(tmp_path)["summary"]["complete"]


def test_worker_rechecks_valid_dataset_rewritten_after_admission(tmp_path, monkeypatch):
    from bayes_stesn.benchmark.generators import Simulation
    monkeypatch.setattr(execution, "host_resources", lambda: _resources())
    p = _tiny(models=("persistence",), replications=1)
    original = runner._execute_job

    def rewritten_data(job, progress=None):
        sim = Simulation.load(job["data_path"])
        observations = sim.observation.copy();observations[0, 0] += .25
        replace(sim, observation=observations).save(job["data_path"])
        np.testing.assert_array_equal(Simulation.load(job["data_path"]).observation, observations)
        return original(job, progress)

    def forbidden(*args, **kwargs):
        raise AssertionError("A worker reached fitting with changed admitted data")

    monkeypatch.setattr(runner, "_execute_job", rewritten_data)
    monkeypatch.setattr(runner, "fit_model", forbidden)
    with pytest.raises(runner.StudyInvalidatedError):
        runner.run_benchmark(p, tmp_path)
    assert not list(tmp_path.rglob("receipt.json"))
    state = json.loads(next(tmp_path.rglob("state.json")).read_text())
    assert state["status"] == "invalidated" and "admitted data" in state["error"]
    assert not runner.benchmark_status(tmp_path)["summary"]["complete"]


def test_memory_allowance_clamps_user_budget_to_current_hard_headroom():
    allowance = execution.memory_allowance(16384, resources=_resources())
    assert allowance["effective_job_budget_bytes"] == 5*GIB
    assert allowance["reserved_headroom_bytes"] == 2*GIB and allowance["clamped"]
    pressure = execution.memory_allowance(16384, resources=_resources(7*GIB))
    assert pressure["effective_job_budget_bytes"] == 0
    for value in (-1, 0, np.inf, np.nan):
        with pytest.raises(ValueError):
            execution.memory_allowance(value, resources=_resources())


def _cache_stat(file=6*GIB, **changes):
    result = {"file": file, "shmem": 0, "file_dirty": 0, "file_writeback": 0,
              "file_mapped": 0, "unevictable": 0}
    result.update(changes)
    return result


def test_cache_admission_keeps_raw_charge_ceiling_and_requested_budget_clamps():
    resources = {**_resources(7*GIB), **execution._cache_admission(7*GIB, _cache_stat())}
    assert resources["memory_current_bytes"] == 7*GIB
    assert resources["memory_cache_eligible_bytes"] == 6*GIB
    assert resources["memory_cache_credit_bytes"] == 3*GIB
    assert resources["memory_admission_estimate_bytes"] == 4*GIB
    large = execution.memory_allowance(16384, resources=resources)
    assert large["effective_job_budget_bytes"] == 2*GIB and large["clamped"]
    assert large["reserved_headroom_bytes"] == 2*GIB
    assert large["host"]["memory_limit_bytes"] == 8*GIB
    assert execution.memory_allowance(1024, resources=resources)["effective_job_budget_bytes"] == GIB
    anonymous = {**_resources(7*GIB), **execution._cache_admission(7*GIB, _cache_stat(file=0))}
    assert execution.memory_allowance(16384, resources=anonymous)["effective_job_budget_bytes"] == 0


@pytest.mark.parametrize("exclusion", ["shmem", "file_dirty", "file_writeback", "file_mapped", "unevictable"])
def test_excluded_file_memory_cannot_earn_admission_credit(exclusion):
    result = execution._cache_admission(7*GIB, _cache_stat(**{exclusion: 6*GIB}))
    assert result["memory_cache_credit_bytes"] == 0
    assert result["memory_admission_estimate_bytes"] == 7*GIB
    overlap = execution._cache_admission(7*GIB, _cache_stat(shmem=4*GIB, file_mapped=4*GIB))
    assert overlap["memory_cache_eligible_bytes"] == 0


@pytest.mark.parametrize("stat", [None, {}, {"file": 6*GIB}, _cache_stat(shmem=-1),
                                 _cache_stat(file_dirty=True), _cache_stat(file=9*GIB)])
def test_missing_or_invalid_cache_statistics_fall_back_to_raw_usage(stat):
    result = execution._cache_admission(7*GIB, stat)
    assert result["memory_admission_estimate_bytes"] == 7*GIB
    assert result["memory_cache_credit_bytes"] == 0
    assert result["memory_admission_policy"] == "raw_usage_missing_or_invalid_cache_stats"


def test_host_resource_receipt_separates_raw_cgroup_usage_from_cache_estimate(monkeypatch):
    values = {"/sys/fs/cgroup/memory.max": str(8*GIB), "/sys/fs/cgroup/memory.current": str(7*GIB),
              "/sys/fs/cgroup/memory.stat": "\n".join(f"{k} {v}" for k, v in _cache_stat().items()),
              "/sys/fs/cgroup/cpu.max": "200000 100000"}
    def read(path, *args, **kwargs):
        if str(path) not in values:
            raise OSError("unavailable fixture")
        return values[str(path)]
    monkeypatch.setattr(execution.Path, "read_text", read)
    monkeypatch.setattr(execution.os, "sysconf", lambda name: {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 2**21}[name])
    result = execution.host_resources()
    assert result["memory_limit_bytes"] == 8*GIB
    assert result["memory_current_bytes"] == 7*GIB
    assert result["memory_admission_estimate_bytes"] == 4*GIB
    assert result["cgroup_memory_stat"] == _cache_stat()


def test_all_job_admission_checks_use_cache_estimate_and_pause_for_anonymous_pressure(tmp_path, monkeypatch):
    pressure = {"anonymous": False}
    def resources():
        stat = _cache_stat(file=0 if pressure["anonymous"] else 6*GIB)
        return {**_resources(7*GIB), **execution._cache_admission(7*GIB, stat)}
    monkeypatch.setattr(execution, "host_resources", resources)
    original = runner._execute_job
    def first_then_anonymous_pressure(job, progress=None):
        result = original(job, progress)
        pressure["anonymous"] = True
        return result
    monkeypatch.setattr(runner, "_execute_job", first_then_anonymous_pressure)
    p = _tiny(replications=1)
    first = runner.run_benchmark(p, tmp_path)
    assert first["jobs"] == 1 and first["pending_jobs"] == 1 and not first["complete"]
    retained = next(tmp_path.rglob("receipt.json"))
    original_bytes = retained.read_bytes()
    pressure["anonymous"] = False
    monkeypatch.setattr(runner, "_execute_job", original)
    final = runner.run_benchmark(p, tmp_path)
    assert final["complete"] and final["jobs"] == 2
    assert retained.read_bytes() == original_bytes


def test_dimension_planning_responds_to_resolved_capacity_and_path_count():
    p = _tiny(models=("bayes_esn_ridge",))
    small = execution._candidate_dimensions("bayes_esn_ridge", {
        "features": {"units": [16], "modes": 2, "squares": 0},
        "spatial_mean": {"side": 2}}, p)
    large = execution._candidate_dimensions("bayes_esn_ridge", {
        "features": {"units": [128, 128], "modes": 8, "squares": 32},
        "spatial_mean": {"side": 5}}, p)
    paths = execution._candidate_dimensions("bayes_esn_ridge", {
        "features": {"units": [16], "modes": 2, "squares": 0},
        "spatial_mean": {"side": 2}}, replace(p, paths=p.paths*4))
    assert large["dimensions"]["readout_coefficients_upper_bound"] > small["dimensions"]["readout_coefficients_upper_bound"]
    assert large["estimated_bytes"] > small["estimated_bytes"]
    assert paths["predictive_tensor_bytes_upper_bound"] == 4*small["predictive_tensor_bytes_upper_bound"]
    assert paths["estimated_bytes"] > small["estimated_bytes"]


def test_dynamic_pressure_pauses_before_next_job_and_can_resume(tmp_path, monkeypatch):
    pressure = {"high": False}
    monkeypatch.setattr(execution, "host_resources", lambda: _resources(7*GIB if pressure["high"] else GIB))
    p = _tiny(replications=1)
    original = runner._execute_job

    def first_then_pressure(job, progress=None):
        result = original(job, progress)
        pressure["high"] = True
        return result

    monkeypatch.setattr(runner, "_execute_job", first_then_pressure)
    result = runner.run_benchmark(p, tmp_path)
    assert result["jobs"] == 1 and result["pending_jobs"] == 1 and not result["complete"]
    assert len(list(tmp_path.rglob("receipt.json"))) == 1
    attempt = json.loads(next((tmp_path/"execution_attempts").glob("*.json")).read_text())
    assert "memory" in attempt["pause_reason"]
    pressure["high"] = False
    monkeypatch.setattr(runner, "_execute_job", original)
    assert runner.run_benchmark(p, tmp_path)["complete"]


def test_actual_job_admission_precedes_dataset_allocation_and_fitting(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "host_resources", lambda: _resources())
    p = _tiny(models=("persistence",), replications=1)
    original = execution.estimate_job_memory
    resolved = []

    def oversized(protocol, model_id, track, seed):
        resolved.append((protocol.identity, model_id, track, seed))
        return {"estimated_bytes": 9*GIB, "dimensions": {"injected_large_workload": True},
                "predictive_tensor_bytes_upper_bound": 0}

    monkeypatch.setattr(execution, "estimate_job_memory", oversized)
    result = runner.run_benchmark(p, tmp_path)
    assert not result["complete"] and result["jobs"] == 0 and result["pending_jobs"] == 1
    assert resolved and resolved[0][:3] == (p.identity, "persistence", "A")
    assert not list(tmp_path.rglob("simulation.npz"))
    assert not list(tmp_path.rglob("receipt.json"))
    state = json.loads(next(tmp_path.rglob("state.json")).read_text())
    assert state["status"] == "pending_resource" and state.get("attempts", 0) == 0
    monkeypatch.setattr(execution, "estimate_job_memory", original)
    assert runner.run_benchmark(p, tmp_path)["complete"]


def test_campaign_lock_excludes_another_coordinator_and_releases_after_error(tmp_path):
    with execution.campaign_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another coordinator"):
            with execution.campaign_lock(tmp_path):
                pytest.fail("A second coordinator acquired the same output")
    with pytest.raises(LookupError):
        with execution.campaign_lock(tmp_path):
            raise LookupError("injected interruption")
    with execution.campaign_lock(tmp_path):
        assert (tmp_path/"run.lock").exists()


def test_cli_plan_resolve_and_status_are_read_only_about_science(tmp_path, monkeypatch, capsys):
    from bayes_stesn.cli import main
    monkeypatch.setattr(execution, "host_resources", lambda: _resources())
    p = _tiny(models=("persistence",), replications=1)
    config = tmp_path/"profile.yaml";p.save(config)

    def forbidden(*args, **kwargs):
        raise AssertionError("A planning/inspection command attempted a simulation or fit")

    with monkeypatch.context() as checks:
        checks.setattr(runner, "generate", forbidden)
        checks.setattr(runner, "fit_model", forbidden)
        assert main(["benchmark", "plan", "--config", str(config), "--output", str(tmp_path/"plan.json")]) == 0
        capsys.readouterr()
        assert main(["benchmark", "resolve", "--config", str(config), "--output", str(tmp_path/"resolved.json")]) == 0
        capsys.readouterr()
    plan = json.loads((tmp_path/"plan.json").read_text())
    assert plan["protocol_identity"] == p.identity and plan["jobs"] == 1
    assert not list(tmp_path.rglob("simulation.npz"))
    run = tmp_path/"run"
    runner.run_benchmark(p, run)
    before = {str(path.relative_to(run)): path.read_bytes() for path in run.rglob("*") if path.is_file()}
    with monkeypatch.context() as checks:
        checks.setattr(runner, "generate", forbidden)
        checks.setattr(runner, "fit_model", forbidden)
        assert main(["benchmark", "status", str(run)]) == 0
        status = json.loads(capsys.readouterr().out)
    assert status["summary"]["complete"] and status["summary"]["jobs"] == 1
    assert before == {str(path.relative_to(run)): path.read_bytes() for path in run.rglob("*") if path.is_file()}
