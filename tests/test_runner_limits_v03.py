"""Declared scientific time ceilings remain distinct from interrupted work."""
from dataclasses import replace
import signal
import time
import pytest
from bayes_stesn.benchmark.protocol import profile
from bayes_stesn.benchmark import runner
from bayes_stesn.benchmark.esn import CheckpointIntegrityError


def test_fit_limit_is_scientific_and_restores_handler():
    handler = signal.getsignal(signal.SIGALRM)
    with pytest.raises(runner.FitTimeLimitExceeded):
        with runner._fit_time_limit(.015):
            time.sleep(.1)
    assert signal.getsignal(signal.SIGALRM) is handler
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)
    # A frozen scientific ceiling must not earn another infrastructure retry.
    runner._propagate_nonmethod_failure(runner.FitTimeLimitExceeded("budget"))


def test_fit_limits_are_frozen_scientific_configuration():
    base = profile("v03_smoke")
    changed = replace(base, fit_time_limits_seconds={"bayes_esn_ridge": 15.})
    assert changed.identity != base.identity
    assert changed.to_dict()["fit_time_limits_seconds"] == {"bayes_esn_ridge": 15.}
    for bad in ({"bayes_esn_ridge": 0}, {"unknown": 2}, {"basis_gru": True}):
        with pytest.raises(ValueError, match="fit_time_limits_seconds"):
            replace(base, fit_time_limits_seconds=bad)
    with pytest.raises(ValueError, match="schema 4.0"):
        replace(profile("smoke"), fit_time_limits_seconds={"basis_gru": 1.})


def test_integrity_and_infrastructure_failures_do_not_become_fallbacks():
    with pytest.raises(runner.StudyInvalidatedError, match="runtime mismatch"):
        runner._propagate_nonmethod_failure(CheckpointIntegrityError("runtime mismatch"))
    with pytest.raises(MemoryError):
        runner._propagate_nonmethod_failure(MemoryError("pressure"))
    with pytest.raises(OSError):
        runner._propagate_nonmethod_failure(OSError("filesystem unavailable"))
    runner._propagate_nonmethod_failure(ValueError("declared model cannot fit"))


def test_fixed_time_allowance_covers_fit_not_forecast(monkeypatch):
    p = replace(profile("v03_smoke"), fit_time_limits_seconds={"persistence": .01})
    calls = []
    def fit(model, panel, config, seed, **kwargs):
        calls.append((model, seed, kwargs))
        time.sleep(.1)
    monkeypatch.setattr(runner, "fit_model", fit)
    with pytest.raises(runner.FitTimeLimitExceeded):
        runner._fit_with_policy("persistence", None, {}, 15, p)
    assert calls == [("persistence", 15, {})]


def test_actual_esn_diagnostic_cost_is_positive_and_included(tmp_path):
    from bayes_stesn.benchmark.generators import generate
    from bayes_stesn.benchmark.protocol import candidate_configs
    p=replace(profile('smoke'),models=('bayes_esn_ridge',),tracks=('A',))
    sim=generate('g1',p.n_times,n_side=p.n_side,n_observed=p.n_observed,n_calibration=p.n_calibration,seed=719)
    config=candidate_configs('bayes_esn_ridge',p,seed=7)[0]
    info,_,_=runner._final_job('bayes_esn_ridge',sim,p,'A',8,tmp_path,{'selected':config,'compute_seconds':0.})
    assert info['completed'],info.get('error')
    assert info['diagnostic_seconds']>0
    assert info['fit_seconds']>=info['diagnostic_seconds']
    assert info['total_seconds']>=info['fit_seconds']+info['forecast_seconds']


def test_external_interruption_retains_unknown_cost_on_resume(tmp_path,monkeypatch):
    import json
    p=replace(profile('v03_smoke'),models=('persistence',),tracks=('A',))
    original=runner._execute_job
    def stopped(job,progress=None):
        runner._record_state(job,'running',started=True,started_unix=time.time(),phase='selection')
        raise KeyboardInterrupt('external interruption fixture')
    monkeypatch.setattr(runner,'_execute_job',stopped)
    with pytest.raises(KeyboardInterrupt):runner.run_benchmark(p,tmp_path/'run')
    monkeypatch.setattr(runner,'_execute_job',original)
    summary=runner.run_benchmark(p,tmp_path/'run')
    assert summary['complete']
    r=json.loads(next((tmp_path/'run').glob('*/rep_*/A/*/receipt.json')).read_text())
    assert len(r['infrastructure_retries'])==1
    assert r['infrastructure_retries'][0]['incomplete_compute_accounting']
    assert not r['compute_accounting_complete']
    assert r['total_pipeline_seconds']>=r['total_seconds']+r['tuning_seconds']
    assert r['total_pipeline_cpu_seconds']>=r['cpu_seconds']
    assert summary['table'][0]['total_seconds_including_tuning']==r['total_pipeline_seconds']
