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
