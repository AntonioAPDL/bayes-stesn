"""Search, effective-configuration, and historical-horizon contracts."""
from dataclasses import replace
import json

import numpy as np
import pytest

from bayes_stesn.io import content_hash
from bayes_stesn.benchmark.protocol import BenchmarkProtocol, profile, candidate_configs, candidate_manifest
from bayes_stesn.benchmark.registry import resolve_model_config, fit_model
from bayes_stesn.benchmark.contracts import Panel


@pytest.mark.parametrize("name,identity", [
    ("smoke","1bb651c77b16947ef4e2af5c058c3955abb77114ec01f3ef41787f1b708d747d"),
    ("pilot","acb5a88946423ddeb228a7b3ba375070f523ad196e854be23953d62b9c7995c9"),
    ("main","84bde79d6f7299e9bdebb49cef600d373ff45467a4edaa99a15508ca81d9fcc4"),
])
def test_legacy_protocol_identity_is_preserved(name,identity,tmp_path):
    p=profile(name)
    assert p.identity==identity
    assert "selection_policy" not in p.to_dict()
    p.save(tmp_path/"protocol.yaml")
    assert BenchmarkProtocol.load(tmp_path/"protocol.yaml").identity==identity
    with pytest.raises(ValueError,match="schema 4.0"):
        replace(p,selection_policy="qualified_finalists")


@pytest.mark.parametrize("model", ["basis_lgssm","separable_stgp","basis_krr",
    "bayes_esn_ridge","eqesn_rollout","dcrnn","basis_gru"])
def test_eight_unique_effective_candidates_and_stable_extension(model):
    p=profile("v03_pilot")
    first=candidate_manifest(model,p,seed=923)
    second=candidate_manifest(model,replace(p,trials=12),seed=923)
    assert len(first["candidates"])==8
    assert len(set(first["candidate_hashes"]))==8
    assert first["candidates"]==second["candidates"][:8]
    assert first["candidate_hashes"]==second["candidate_hashes"][:8]
    assert first==candidate_manifest(model,p,seed=923)
    assert all(content_hash(c)==h for c,h in zip(first["candidates"],first["candidate_hashes"]))


def test_basis_gru_is_present_and_searches_real_architectures():
    p=profile("v03_main")
    assert "basis_gru" in p.models
    candidates=candidate_configs("basis_gru",p,seed=4)
    assert len({(c["rank"],c["lags"],c["hidden_size"]) for c in candidates})>1
    assert all(c["validation_horizon"]==100 and c["require_full_validation_horizon"] for c in candidates)


def test_nested_override_preserves_other_sampled_features():
    p=profile("v03_pilot",model_options={"bayes_esn_ridge":{"features":{"recurrent_norm":.75}}})
    result=candidate_manifest("bayes_esn_ridge",p,seed=177)
    assert all(c["features"]["recurrent_norm"]==.75 for c in result["candidates"])
    assert len({tuple(c["features"]["units"]) for c in result["candidates"]})>1
    assert len({tuple(c["features"]["input_scales"]) for c in result["candidates"]})>1
    assert all("prior" in c and "covariance" in c and "min_ess" in c["diagnostics"] for c in result["candidates"])


def test_deduplication_happens_after_actual_subset_constraints():
    p=profile("v03_pilot",trials=12)
    result=candidate_manifest("basis_lgssm",p,track="B",fitting_sites=5,seed=11)
    assert len(result["candidates"])==3  # Three widths; all proposed ranks become five.
    assert {c["rank"] for c in result["candidates"]}=={5}
    assert result["metadata"]["stop_reason"]=="finite_space_exhausted"
    assert result["metadata"]["duplicates"]>0
    esn=candidate_configs("bayes_esn_ridge",p,track="B",fitting_sites=5,seed=9)
    assert all(c["features"]["modes"]<5 for c in esn)
    assert any(c["spatial_mean"]["side"]**2+3>5 for c in esn)  # A regularized dictionary may be overcomplete.


def test_complete_override_collapses_finite_space_honestly():
    p=profile("v03_pilot",model_options={"basis_lgssm":{"rank":3,"basis_width":.2}})
    result=candidate_manifest("basis_lgssm",p,seed=9)
    assert len(result["candidates"])==1
    assert result["metadata"]["stop_reason"]=="finite_space_exhausted"
    assert result["metadata"]["effective_count"]==1
    assert result["metadata"]["requested"]==8


def test_proposal_cap_is_distinct_from_finite_exhaustion():
    p=profile("v03_pilot",candidate_proposal_cap=2,
              model_options={"basis_lgssm":{"rank":3,"basis_width":.2}})
    result=candidate_manifest("basis_lgssm",p,seed=9)
    assert result["metadata"]["stop_reason"]=="proposal_cap_reached"
    assert result["metadata"]["raw_proposals"]==2


@pytest.mark.parametrize("options", [
    {"basis_gur":{"rank":8}},
    {"basis_gru":{"hidden_siz":8}},
    {"bayes_esn_ridge":{"features":{"leak_pattern":.5}}},
    {"bayes_esn_ridge":{"sampler":{"draw":200}}},
])
def test_misspelled_options_rejected_before_fitting(options):
    with pytest.raises(ValueError):
        profile("v03_pilot",model_options=options)


def test_frozen_configuration_is_resolved_once_and_missing_track_is_an_error():
    p=profile("v03_pilot")
    config=candidate_configs("basis_krr",p,seed=8)[0]
    frozen=replace(p,selection_mode="development_frozen_configuration",
                   frozen_configurations={"basis_krr/A":config})
    result=candidate_manifest("basis_krr",frozen,track="A",seed=200)
    assert result["candidates"]==[config]
    assert result["metadata"]["requested"]==1
    with pytest.raises(ValueError,match="No frozen configuration"):
        candidate_manifest("basis_krr",frozen,track="B")


def test_frozen_new_site_configuration_uses_full_refit_panel():
    p=profile("v03_pilot",selection_mode="development_frozen_configuration",
              frozen_configurations={"basis_lgssm/B":{"rank":24,"max_iter":200}})
    result=candidate_manifest("basis_lgssm",p,track="B")
    assert result["candidates"][0]["rank"]==24
    assert result["metadata"]["fitting_sites"]==25
    assert result["metadata"]["fitting_rows"]==520


def test_normalization_reference_cannot_use_validation_or_unavailable_history(tmp_path):
    reference={"start":200,"stop":400,"observed_count":10,"floor":1e-10,"ddof":1}
    p=profile("v03_pilot",normalization_reference=reference)
    p.save(tmp_path/"new.yaml")
    assert BenchmarkProtocol.load(tmp_path/"new.yaml").to_dict()==p.to_dict()
    for change in ({"stop":401},{"start":-1},{"observed_count":26},{"ddof":2},{"floor":0}):
        with pytest.raises(ValueError):
            replace(p,normalization_reference={**reference,**change})
    with pytest.raises(ValueError,match="initial training"):
        replace(p,training_start=250)


@pytest.mark.parametrize("model",["basis_gru","dcrnn"])
def test_horizon_resolution_reserves_the_requested_historical_tail(model):
    if model=="basis_gru":
        raw={"lags":3,"validation_horizon":40,"require_full_validation_horizon":True}
    else:
        raw={"context":4,"forecast_training_steps":2,"validation_horizon":40,
             "require_full_validation_horizon":True}
    resolved=resolve_model_config(model,raw,fitting_sites=6,fitting_rows=80)
    assert resolved["dimensions"]["stopping_training_rows"]==40
    assert resolved["dimensions"]["stopping_validation_leads"]==40
    raw["validation_horizon"]=78
    with pytest.raises(ValueError,match="horizon"):
        resolve_model_config(model,raw,fitting_sites=6,fitting_rows=80)


@pytest.mark.neural
@pytest.mark.parametrize("model",["basis_gru","dcrnn"])
def test_tiny_neural_fit_consumes_full_stopping_horizon(model):
    torch=pytest.importorskip("torch")
    torch.set_num_threads(1)
    rng=np.random.default_rng(82)
    data=Panel(rng.normal(size=(70,5)),rng.uniform(size=(5,2)))
    common={"validation_horizon":30,"require_full_validation_horizon":True}
    config=({**common,"rank":3,"hidden_size":4,"lags":3,"epochs":1,
             "covariance_max_iter":2,"mean_paths":4} if model=="basis_gru" else
            {**common,"hidden_size":4,"layers":1,"context":4,"forecast_training_steps":2,
             "max_epochs":1,"batch_size":8})
    fitted=fit_model(model,data,config,seed=91)
    assert fitted.diagnostics["selection_validation_rows"]==30
    assert fitted.diagnostics["selection_training_rows"]==40
    assert not fitted.diagnostics["selection_horizon_truncated"]
    assert len(fitted.data.y if model=="basis_gru" else fitted.training.y)==70
