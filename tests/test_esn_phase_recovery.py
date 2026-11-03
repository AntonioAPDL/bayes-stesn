"""Phase-boundary fault injection for one bounded ESN continuation."""
from dataclasses import replace
from unittest import mock

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from bayes_stesn.benchmark.contracts import Panel
from bayes_stesn.benchmark.esn import CheckpointIntegrityError, _fit_state_path, fit_esn
from bayes_stesn.io import content_hash, load_state, save_state
from bayes_stesn.model import fit


@pytest.fixture
def problem():
    rng = np.random.default_rng(416)
    panel = Panel(rng.normal(size=(60, 4)), rng.uniform(size=(4, 2)))
    config = {"conditioning":12,
        "features":{"modes":2, "units":[4], "squares":2, "lags":[1]},
        "spatial_mean":{"side":2, "width":.3},
        "sampler":{"chains":2, "warmup":8, "draws":12, "initialize":"prior",
                   "checkpoint_every":4, "parameterization":"ridge_total_scale"},
        "diagnostics":{"functional_draws":8, "min_chains":4}, "continuation_draws":8}
    return panel, config, 71


@pytest.mark.parametrize("boundary", ["warmup", "sampling", "initial_diagnostics",
                                      "continuation", "continuation_diagnostics", "complete"])
def test_each_interruption_resumes_exactly_and_never_adds_a_continuation(tmp_path, problem, boundary):
    panel, config, seed = problem
    baseline = fit_esn(panel, config, seed, checkpoint_path=tmp_path/"reference.npz")
    path = tmp_path/"interrupted.npz"
    class Interrupted(Exception):
        pass
    fired = False
    def save_checkpoint(name, state, **kwargs):
        nonlocal fired
        result = save_state(name, state, **kwargs)
        chain = state["chains"].get("0", {})
        target = state["config"]["draws"]
        iteration = chain.get("iteration", -1)
        condition = ((boundary == "warmup" and target == 12 and iteration == 4)
                     or (boundary == "sampling" and target == 12 and iteration == 16)
                     or (boundary == "continuation" and target == 20 and iteration == 24))
        if condition and not fired:
            fired = True
            raise Interrupted()
        return result
    def save_phase(name, state, **kwargs):
        nonlocal fired
        result = save_state(name, state, **kwargs)
        if state["phase"] == boundary and not fired:
            fired = True
            raise Interrupted()
        return result
    with mock.patch("bayes_stesn.model.save_state", side_effect=save_checkpoint), \
         mock.patch("bayes_stesn.benchmark.esn.save_state", side_effect=save_phase):
        with pytest.raises(Interrupted):
            fit_esn(panel, config, seed, checkpoint_path=path)
    assert fired
    resumed = fit_esn(panel, config, seed, checkpoint_path=path)
    assert resumed.diagnostics["recovery"]["resumed"]
    assert resumed.diagnostics["continuations"] == 1
    assert resumed.fitted.posterior.n_draws == 20
    assert resumed.diagnostics["gates"] == baseline.diagnostics["gates"]
    assert_array_equal(resumed.fitted.posterior.chains, baseline.fitted.posterior.chains)
    assert_array_equal(resumed.fitted.posterior.log_density, baseline.fitted.posterior.log_density)
    assert_array_equal(resumed.fitted.posterior.acceptance, baseline.fitted.posterior.acceptance)
    a, b = resumed.forecast(3, n_paths=4, seed=910), baseline.forecast(3, n_paths=4, seed=910)
    assert_array_equal(a.samples, b.samples)
    again = fit_esn(panel, config, seed, checkpoint_path=path)
    assert again.diagnostics["continuations"] == 1
    assert_array_equal(again.fitted.posterior.chains, baseline.fitted.posterior.chains)


def test_changed_allowance_is_rejected_without_overwriting_checkpoint(tmp_path, problem):
    panel, config, seed = problem
    path = tmp_path/"chain.npz"
    fit_esn(panel, config, seed, checkpoint_path=path)
    before = path.read_bytes(), _fit_state_path(path).read_bytes()
    for changes in ({"continuation_draws":9}, {"diagnostics":{"functional_draws":12, "min_chains":4}}):
        with pytest.raises(CheckpointIntegrityError, match="differs|allowance"):
            fit_esn(panel, {**config, **changes}, seed, checkpoint_path=path)
        assert (path.read_bytes(), _fit_state_path(path).read_bytes()) == before


def test_preledger_continued_checkpoint_reconstructs_gate_without_more_draws(tmp_path, problem):
    panel, config, seed = problem
    path = tmp_path/"old.npz"
    reference = fit_esn(panel, config, seed, checkpoint_path=path)
    _fit_state_path(path).unlink()
    recovered = fit_esn(panel, config, seed, checkpoint_path=path)
    assert recovered.diagnostics["recovery"]["initial_gate_reconstructed"]
    assert recovered.diagnostics["gates"] == reference.diagnostics["gates"]
    assert_array_equal(recovered.fitted.posterior.chains, reference.fitted.posterior.chains)
    assert recovered.fitted.posterior.n_draws == 20


def test_old_legacy_checkpoint_config_and_new_cadence_remain_compatible(tmp_path, problem):
    panel, config, seed = problem
    config = {**config, "sampler":{**config["sampler"], "parameterization":"legacy"}}
    path = tmp_path/"legacy.npz"
    fitted = fit_esn(panel, config, seed, checkpoint_path=path)
    checkpoint = load_state(path, kind="mcmc_checkpoint")
    checkpoint["config"].pop("parameterization")
    save_state(path, checkpoint, kind="mcmc_checkpoint")
    # Old core files have neither a named legacy option nor a benchmark ledger.
    _fit_state_path(path).unlink()
    resumed = fit_esn(panel, {**config, "sampler":{**config["sampler"], "checkpoint_every":0}},
                      seed, checkpoint_path=path)
    assert_array_equal(fitted.fitted.posterior.chains, resumed.fitted.posterior.chains)
    direct = fit(fitted.fitted.prepared, replace(fitted.fitted.sampler, checkpoint_every=0),
                 checkpoint_path=path, resume=True)
    assert_array_equal(direct.posterior.log_density, fitted.fitted.posterior.log_density)


def test_terminal_checkpoint_mismatch_is_not_silently_accepted(tmp_path, problem):
    panel, config, seed = problem
    path = tmp_path/"terminal.npz"
    fit_esn(panel, config, seed, checkpoint_path=path)
    state = load_state(_fit_state_path(path), kind="benchmark_esn_fit")
    state["checkpoint"]["sha256"] = "wrong"
    save_state(_fit_state_path(path), state, kind="benchmark_esn_fit")
    before = content_hash(load_state(path, kind="mcmc_checkpoint"))
    with pytest.raises(CheckpointIntegrityError, match="does not match"):
        fit_esn(panel, config, seed, checkpoint_path=path)
    assert content_hash(load_state(path, kind="mcmc_checkpoint")) == before


def test_core_checkpoint_signature_and_source_mismatches_have_distinct_error(tmp_path, problem):
    panel, config, seed = problem
    path = tmp_path/"signature.npz"
    fit_esn(panel, config, seed, checkpoint_path=path)
    before = path.read_bytes()
    with mock.patch("bayes_stesn.benchmark.esn._fit_computation", return_value={"source":"changed"}):
        with pytest.raises(CheckpointIntegrityError, match="identity, runtime"):
            fit_esn(panel, config, seed, checkpoint_path=path)
    assert path.read_bytes() == before
    # A safe numeric artifact with an incompatible lower-level transition
    # signature must not be categorized as a scientific convergence failure.
    state = load_state(path, kind="mcmc_checkpoint")
    state["chains"]["0"]["signature"] = "different-transition"
    save_state(path, state, kind="mcmc_checkpoint")
    _fit_state_path(path).unlink()
    with pytest.raises(CheckpointIntegrityError, match="Checkpoint"):
        fit_esn(panel, config, seed, checkpoint_path=path)


def test_default_encoder_modes_fit_a_small_spatial_subset(tmp_path, problem):
    panel, _, seed = problem
    config = {"features":{"units":[4], "squares":2},
              "sampler":{"chains":2, "warmup":2, "draws":8, "initialize":"prior"},
              "diagnostics":{"functional_draws":8, "min_chains":4}}
    fitted = fit_esn(panel, config, seed, checkpoint_path=tmp_path/"small.npz")
    assert fitted.fitted.prepared.spec.features.modes == 3
    one_site = Panel(panel.y[:, :1], panel.coordinates[:1])
    with pytest.raises(ValueError, match="at least two"):
        fit_esn(one_site, config, seed)
