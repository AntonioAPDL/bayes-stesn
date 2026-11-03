"""Independent, inexpensive checks of selection, cached evidence and CLI routing."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from bayes_stesn.cli import main as cli_main
from bayes_stesn.io import content_hash, seed_for
from bayes_stesn.benchmark.contracts import Forecast
from bayes_stesn.benchmark.generators import generate
from bayes_stesn.benchmark.protocol import profile, candidate_manifest
import bayes_stesn.benchmark.protocol as protocol_module
import bayes_stesn.benchmark.runner as runner


def _simulation(p):
    return generate("g1",p.n_times,n_side=p.n_side,n_observed=p.n_observed,
                    n_calibration=p.n_calibration,seed=817)


def _fake_selection(monkeypatch, *, candidates=5, seeds=2, replacements=2,
                    model="basis_krr", eligible=None, finalist_scores=None,
                    failures=None):
    p=profile("v03_smoke",models=(model,),tracks=("A",),trials=candidates,
              finalist_count=2,finalist_seeds=seeds,replacement_finalists=replacements)
    configurations=[{"rank":index+1} for index in range(candidates)]
    hashes=[content_hash(c) for c in configurations]
    design={"candidates":configurations,"candidate_hashes":hashes,
            "metadata":{"schema":"unit_fixture","effective_count":candidates}}
    monkeypatch.setattr(protocol_module,"candidate_manifest",lambda *args,**kwargs:design)
    monkeypatch.setattr(runner,"source_fingerprint",lambda:{"fixture_source":"stable"})
    namespace={}
    for index,h in enumerate(hashes):
        namespace[seed_for(98,"candidate_fit",h,0)]=(index,"screen",0)
        for repetition in range(seeds):
            namespace[seed_for(98,"finalist_fit",h,repetition)]=(index,"finalist",repetition)
    calls=[]
    def fake_validate(model_id,config,simulation,protocol,track,fit_seed,score_seed,**kwargs):
        index,phase,repetition=namespace[fit_seed]
        assert config==configurations[index]
        calls.append((index,phase,repetition))
        if failures and (index,phase,repetition) in failures:
            raise RuntimeError("declared fixture fit failure")
        qualified=True if phase=="screen" or eligible is None else eligible(index,repetition)
        score=float(index) if phase=="screen" or finalist_scores is None else float(finalist_scores(index,repetition))
        return {"score":score,"qualified":qualified,"seconds":2.,"cpu_seconds":1.,
                "fit_seconds":1.,"forecast_seconds":1.,"origin_scores":[score]*len(p.origins),
                "diagnostics":{"qualified":qualified}}
    monkeypatch.setattr(runner,"_fit_and_validate",fake_validate)
    return p,_simulation(p),calls,configurations


def test_unqualified_top_finalists_are_replaced_without_dropping_seed_costs(monkeypatch):
    p,sim,calls,configs=_fake_selection(monkeypatch,eligible=lambda index,rep:index>=2)
    result=runner.select_configuration("basis_krr",sim,p,"A",98)
    assert result["selected"]==configs[2]
    assert result["selected_candidate"]==2
    assert len(result["trials"])==5 and len(result["rechecks"])==6
    assert sum(not row["qualified"] for row in result["rechecks"])==4
    assert not any(phase=="finalist" and index>2 for index,phase,_ in calls)
    assert result["compute_seconds"]==2*len(calls)
    assert result["compute_cpu_seconds"]==len(calls)
    assert len({row["fit_seed"] for row in result["trials"]+result["rechecks"]})==len(calls)


def test_one_failed_seed_disqualifies_a_candidate_and_successful_scores_are_not_cherry_picked(monkeypatch):
    p,sim,calls,configs=_fake_selection(monkeypatch,seeds=3,
        eligible=lambda index,rep:not(index==0 and rep==1))
    result=runner.select_configuration("basis_krr",sim,p,"A",98)
    assert result["selected"]==configs[1]
    first=[row for row in result["rechecks"] if row["candidate"]==0]
    assert len(first)==3 and [row["qualified"] for row in first]==[True,False,True]
    assert len(result["rechecks"])==6  # One initial finalist was eligible: no replacements.


def test_finalist_score_is_mean_of_fresh_rechecks_not_screening_winner(monkeypatch):
    scores={(0,0):10.,(0,1):12.,(1,0):3.,(1,1):5.}
    p,sim,_,configs=_fake_selection(monkeypatch,finalist_scores=lambda i,r:scores[(i,r)])
    result=runner.select_configuration("basis_krr",sim,p,"A",98)
    assert result["trials"][0]["score"]<result["trials"][1]["score"]
    assert result["selected"]==configs[1]
    assert result["selected_validation_score"]==4.


def test_replacement_limit_does_not_search_until_a_candidate_eventually_qualifies(monkeypatch):
    p,sim,calls,_=_fake_selection(monkeypatch,eligible=lambda index,rep:index==4)
    result=runner.select_configuration("basis_krr",sim,p,"A",98)
    assert result["selected"] is None
    assert result["status"]=="all_finalists_unqualified"
    assert len(result["rechecks"])==8
    assert {i for i,phase,_ in calls if phase=="finalist"}=={0,1,2,3}


def test_failed_finalist_fit_is_recorded_and_cannot_be_dropped(monkeypatch):
    p,sim,_,configs=_fake_selection(monkeypatch,failures={(0,"finalist",1)})
    result=runner.select_configuration("basis_krr",sim,p,"A",98)
    assert result["selected"]==configs[1]
    failed=[row for row in result["rechecks"] if row["status"]=="failed"]
    assert len(failed)==1 and failed[0]["candidate"]==0
    assert failed[0]["score"] is None and not failed[0]["qualified"]
    assert "declared fixture" in failed[0]["error"]


def test_deterministic_finalists_do_not_duplicate_identical_seed_evaluations(monkeypatch):
    p,sim,calls,_=_fake_selection(monkeypatch,model="basis_lgssm",seeds=3)
    result=runner.select_configuration("basis_lgssm",sim,p,"A",98)
    assert result["finalist_seed_count"]==1
    assert len(result["rechecks"])==2
    assert all(rep==0 for _,phase,rep in calls if phase=="finalist")


def test_cached_candidate_evaluations_resume_without_refitting_and_charge_original_cost(monkeypatch,tmp_path):
    p,sim,calls,_=_fake_selection(monkeypatch)
    first=runner.select_configuration("basis_krr",sim,p,"A",98,directory=tmp_path)
    number=len(calls)
    resumed=runner.select_configuration("basis_krr",sim,p,"A",98,directory=tmp_path)
    assert len(calls)==number
    assert resumed==first
    assert resumed["compute_seconds"]==2*number
    for path in tmp_path.glob("*.json"):
        row=json.loads(path.read_text())
        assert row["record_identity"]==content_hash({k:v for k,v in row.items() if k!="record_identity"})


@pytest.mark.parametrize("change",["corrupt_record","different_source","different_history"])
def test_cached_selection_evidence_mismatch_invalidates_the_study(monkeypatch,tmp_path,change):
    p,sim,_,_=_fake_selection(monkeypatch)
    runner.select_configuration("basis_krr",sim,p,"A",98,directory=tmp_path)
    if change=="corrupt_record":
        path=next(tmp_path.glob("screen_*.json"))
        row=json.loads(path.read_text());row["score"]+=100.
        path.write_text(json.dumps(row))
    elif change=="different_source":
        monkeypatch.setattr(runner,"source_fingerprint",lambda:{"fixture_source":"different"})
    else:
        values=sim.observation.copy();values[0,sim.observed_indices[0]]+=1.
        sim=replace(sim,observation=values)
    with pytest.raises(runner.StudyInvalidatedError,match="Cached tuning"):
        runner.select_configuration("basis_krr",sim,p,"A",98,directory=tmp_path)


def test_track_b_requires_every_spatial_fold_and_preserves_input_information(monkeypatch):
    p=profile("v03_smoke",models=("basis_krr",),tracks=("B",),spatial_validation_blocks=2)
    sim=_simulation(p);fits=[];forecasts=[]
    def fit(model,data,config,seed,**kwargs):
        index=len(fits);fits.append(data)
        class Fitted:
            diagnostics={"qualified":index==0}
            def forecast(self,horizon,target_coordinates,target_site_ids,n_paths,seed,history):
                assert history.site_ids==data.site_ids
                assert not set(target_site_ids)&set(history.site_ids)
                assert len(data.y)==p.initial_fit
                assert len(history.y) in p.origins
                forecasts.append((index,len(history.y),tuple(target_site_ids)))
                return Forecast(np.zeros((horizon,len(target_coordinates))),None,
                                target_coordinates,target_site_ids)
        return Fitted()
    monkeypatch.setattr(runner,"fit_model",fit)
    result=runner._fit_and_validate("basis_krr",{},sim,p,"B",76,77)
    assert len(fits)==2 and len(forecasts)==2*len(p.origins)
    assert [row["qualified"] for row in result["folds"]]==[True,False]
    assert not result["qualified"]
    assert result["score"]==pytest.approx(np.mean([r["score"] for r in result["folds"]]))
    assert {row["fold"] for row in result["folds"]}=={"edge_right","interior"}


def test_frozen_mode_uses_full_final_panel_and_never_fits_for_selection(monkeypatch):
    p=profile("v03_smoke",models=("basis_krr",),tracks=("B",),
        selection_mode="development_frozen_configuration",
        frozen_configurations={"basis_krr/B":{"rank":9}})
    sim=_simulation(p)
    def forbidden(*args,**kwargs):
        pytest.fail("Frozen configuration selection must not create validation folds or fit a model")
    monkeypatch.setattr(runner,"spatial_validation_splits",forbidden)
    monkeypatch.setattr(runner,"_fit_and_validate",forbidden)
    selected=runner.select_configuration("basis_krr",sim,p,"B",83)
    assert selected["selected"]["rank"]==9
    assert selected["candidate_design"]["fitting_sites"]==p.n_observed
    assert selected["candidate_design"]["fitting_rows"]==p.final_origin-p.training_start
    assert selected["trials"]==selected["rechecks"]==[]
    assert selected["compute_seconds"]==selected["compute_cpu_seconds"]==0.


def test_cli_resolve_and_plan_include_the_same_implicit_frozen_spatial_control(monkeypatch,tmp_path,capsys):
    p=profile("v03_smoke",models=("basis_krr",),tracks=("B",),
        selection_mode="development_frozen_configuration",
        frozen_configurations={"basis_krr/B":{"rank":9}})
    config=tmp_path/"configuration.yaml";p.save(config)
    def forbidden(*args,**kwargs):
        pytest.fail("CLI resolution/planning must not fit or generate scientific data")
    monkeypatch.setattr(runner,"fit_model",forbidden)
    monkeypatch.setattr(runner,"generate",forbidden)
    assert cli_main(["benchmark","resolve","--config",str(config)])==0
    resolved=json.loads(capsys.readouterr().out)
    assert set(resolved["candidates"])=={"basis_krr/B","spatial_persistence/B"}
    control=resolved["candidates"]["spatial_persistence/B"]
    assert control["candidates"][0]["spatial_length"]==.2
    assert control["candidates"][0]["ridge"]==.01
    assert control["metadata"]["candidate_records"][0]["proposal"]=="fixed_spatial_control"
    output=tmp_path/"plan.json"
    assert cli_main(["benchmark","plan","--config",str(config),"--output",str(output)])==0
    plan=json.loads(capsys.readouterr().out)
    assert plan==json.loads(output.read_text())
    assert {row["model_id"] for row in plan["models"]}=={"basis_krr","spatial_persistence"}
    assert plan["selection_fit_calls"]==plan["selection_fit_calls_with_replacements"]==0
    assert plan["jobs"]==2
    assert "spatial_persistence" in plan["memory"]["per_model"]


def test_cli_plan_only_implicit_control_has_nonzero_model_memory(tmp_path,capsys):
    p=profile("v03_smoke",models=("persistence",),tracks=("B",))
    config=tmp_path/"configuration.yaml";p.save(config)
    assert cli_main(["benchmark","plan","--config",str(config)])==0
    plan=json.loads(capsys.readouterr().out)
    assert {row["model_id"] for row in plan["models"]}=={"spatial_persistence"}
    assert set(plan["memory"]["per_model"])=={"spatial_persistence"}
    assert plan["memory"]["estimated_per_worker_bytes"]>0


def test_cli_resume_loads_saved_protocol_and_routes_execution_only_controls(monkeypatch,tmp_path,capsys):
    p=profile("v03_smoke",models=("persistence",),tracks=("A",))
    p.save(tmp_path/"resolved_protocol.yaml")
    calls=[]
    def fake_run(protocol,output,**kwargs):
        calls.append((protocol,Path(output),kwargs))
        return {"jobs":1,"scheduled_jobs":1,"complete":True,"stage":protocol.stage}
    monkeypatch.setattr(runner,"run_benchmark",fake_run)
    assert cli_main(["benchmark","resume","--output",str(tmp_path),"--workers","1",
                     "--max-jobs","2","--max-wall-seconds","10"])==0
    assert len(calls)==1 and calls[0][0].identity==p.identity
    options=calls[0][2]
    assert options["resume"] and options["max_jobs"]==2 and options["max_wall_seconds"]==10.
    assert json.loads(capsys.readouterr().out)["complete"]
