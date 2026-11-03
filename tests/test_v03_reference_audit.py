"""Bounded independent checks for the saved-reference audit tool."""
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose

from bayes_stesn.benchmark.generators import generate


def audit_tool():
    path = Path(__file__).resolve().parents[1]/"tools/audit_v03_references.py"
    spec = importlib.util.spec_from_file_location("reference_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analytic_gaussian_crps_center_and_point_mass_limits():
    tool = audit_tool()
    result = tool.gaussian_scores(np.zeros((1, 2)), np.array([[4., 0.]]), np.array([[0., 3.]]))
    expected = (2*(np.sqrt(2)-1)/np.sqrt(np.pi)+3)/2
    assert_allclose(result["crps"], expected, rtol=1e-14)
    assert_allclose(result["rmse"], 3/np.sqrt(2), rtol=1e-14)
    assert result["conditional_expected_mse"] == 2.


def test_oracle_moments_ignore_latent_future_and_unobserved_histories():
    tool = audit_tool()
    sim = generate("g1", 50, n_side=4, n_observed=5, n_calibration=2, seed=137)
    protocol = SimpleNamespace(training_start=10, final_origin=46, horizon=4, score_observed_count=None)
    altered = np.array(sim.observation, copy=True)
    altered[:10] += 100
    altered[46:] -= 300
    never_visible = np.setdiff1d(np.arange(len(sim.coordinates)), sim.observed_indices)
    altered[:, never_visible] *= -100
    poisoned = replace(sim, observation=altered, latent=np.full_like(sim.latent, 400.))
    for track in ("A", "B"):
        original, targets = tool.oracle_moments(sim, protocol, track)
        checked, other_targets = tool.oracle_moments(poisoned, protocol, track)
        assert_allclose(original["mean"], checked["mean"], rtol=0, atol=0)
        assert_allclose(original["variance"], checked["variance"], rtol=0, atol=0)
        assert_allclose(targets, other_targets, rtol=0, atol=0)
        assert original["history_hash"] == checked["history_hash"]


def test_audit_retains_original_total_and_flags_diagnostic_subcomponent():
    tool = audit_tool()
    receipt = dict(fit_seconds=5., forecast_seconds=.3, scoring_seconds=.2,
                   serialization_seconds=.4, artifact_serialization_seconds=.2,
                   fallback_seconds=0., tuning_seconds=7., total_seconds=6.,
                   total_pipeline_seconds=13.2, cpu_seconds=5., diagnostic_seconds=0.,
                   job_identity="fixture", compute_accounting_complete=True,
                   diagnostics={"recovery": {"completed_phase_seconds": {
                       "initial_sampling": 3., "initial_diagnostics": 1., "continuation_diagnostics": 1.}}})
    findings = []
    result = tool.audit_cost(receipt, findings)
    assert result["total_pipeline_seconds"] == 13.2
    assert result["cumulative_diagnostic_ledger_seconds"] == 2.
    assert len(findings) == 1
    assert findings[0]["code"] == "diagnostic_subcomponent_scope_or_value_mismatch"


def test_audit_preserves_unqualified_nonfinite_esn_diagnostics():
    tool = audit_tool()
    block = {"chains": 4, "parameters": {"constant": {"rhat": None, "ess_bulk": None, "ess_tail": None}}}
    receipt = {"model_id": "bayes_esn_ridge", "qualified": False,
               "fit_config": {"diagnostics": {"min_chains": 4, "rhat_limit": 1.01, "min_ess": 400}},
               "diagnostics": {"qualified": False, "continuations": 1,
                               "gates": [{"passed": False, "hyperparameters": block, "functionals": block}]}}
    result = tool.audit_optimization(receipt)
    assert result["max_rhat"] is None
    assert result["minimum_bulk_or_tail_ess"] is None
    assert result["nonfinite_diagnostic_entries"] == 6
    assert result["final_qualified"] is False
