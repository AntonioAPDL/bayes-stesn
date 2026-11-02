"""End-to-end information, resumption, scoring and calibration contracts."""
from dataclasses import replace
import json
import hashlib
from pathlib import Path
import numpy as np
import pytest

from bayes_stesn.benchmark import Panel,Forecast,CapabilityError,profile,fit_model,save_model,load_model
from bayes_stesn.benchmark.contracts import select_panel,validate_history
from bayes_stesn.benchmark.generators import generate
from bayes_stesn.benchmark.metrics import score_forecast,aggregate_results
from bayes_stesn.benchmark.protocol import BenchmarkProtocol,candidate_configs
from bayes_stesn.benchmark.runner import run_benchmark,select_configuration,spatial_validation_split
from bayes_stesn.benchmark.calibration import calibrate_frozen_model,evaluate_calibration,save_calibration,load_calibration
from bayes_stesn.benchmark.studies import (history_curve_profiles,sensor_curve_profiles,freeze_replications,
    freeze_confirmation,verify_confirmation_freeze,run_calibration_study,_fhn_evidence)


def test_forecast_contract_roundtrip_defensive_and_point_capability(tmp_path):
    mean=np.zeros((3,2));samples=np.ones((5,3,2));coords=np.array([[0.,0.],[1.,1.]])
    forecast=Forecast(mean,samples,coords,("a","b"))
    mean[:]=77;samples[:]=88
    assert forecast.mean[0,0]==0 and forecast.samples[0,0,0]==1
    assert not forecast.samples.flags.writeable
    forecast.save(tmp_path/"forecast.npz")
    loaded=Forecast.load(tmp_path/"forecast.npz")
    np.testing.assert_array_equal(loaded.samples,forecast.samples)
    with pytest.raises(CapabilityError):
        Forecast(mean,None,coords,("a","b")).interval()


def test_observation_contract_rejects_truth_objects_and_changed_prefix():
    sim=generate("g1",45,n_side=4,n_observed=8,n_calibration=2,seed=7)
    with pytest.raises(TypeError):
        fit_model("persistence",sim)
    training=sim.observed_panel(30);history=sim.observed_panel(40)
    assert validate_history(training,history) is history
    changed=history.y.copy();changed[0,0]+=1
    with pytest.raises(ValueError):
        validate_history(training,Panel(changed,history.coordinates,history.times,history.site_ids))
    cropped=select_panel(history,start=10,stop=30)
    assert len(cropped.y)==20 and cropped.times[0]==history.times[10]


def test_profile_roundtrip_and_invalid_budget(tmp_path):
    for name in ("smoke","pilot","main"):
        p=profile(name);p.save(tmp_path/f"{name}.yaml")
        assert p.identity==BenchmarkProtocol.load(tmp_path/f"{name}.yaml").identity
        assert p.origins[-1]+p.horizon==p.final_origin
    with pytest.raises(ValueError):
        replace(profile("pilot"),trials=13)
    with pytest.raises(ValueError):
        replace(profile("main"),replications=2)


def test_all_candidate_schemas_fit_classical():
    p=profile("smoke")
    sim=generate("g1",p.n_times,n_side=5,n_observed=10,n_calibration=3,seed=71)
    for model in ("persistence","spatial_persistence","seasonal_naive","basis_krr","basis_lgssm","separable_stgp"):
        config=candidate_configs(model,p,seed=9)[0]
        fitted=fit_model(model,sim.observed_panel(p.initial_fit),config,seed=10)
        result=fitted.forecast(3,n_paths=5,seed=11)
        assert result.mean.shape==(3,10)


def test_metric_mean_median_and_empirical_crps():
    samples=np.array([0.,0.,9.])[:,None,None]*np.ones((1,2,2))
    pred=Forecast(samples.mean(axis=0),samples,np.array([[0.,0.],[1.,1.]]),("a","b"))
    scores=score_forecast(pred,np.zeros((2,2)),pair_count=10)
    assert scores["rmse"]==3 and scores["mae"]==0
    assert scores["crps"]==pytest.approx(1.)
    assert scores["functional_crps"]["path_maximum"]==pytest.approx(1.)
    assert "1" in scores["temporal_variogram"]


def test_spatial_validation_uses_geometry_only():
    sim=generate("g1",50,n_side=5,n_observed=10,n_calibration=3)
    data=sim.observed_panel();kept,held=spatial_validation_split(data)
    shuffled=Panel(data.y[::-1],data.coordinates,site_ids=data.site_ids)
    a,b=spatial_validation_split(shuffled)
    np.testing.assert_array_equal(a,kept);np.testing.assert_array_equal(b,held)
    assert not set(kept)&set(held)


def test_selection_invariant_to_all_final_and_U_responses():
    p=profile("smoke",models=("basis_krr",),trials=2)
    sim=generate("g1",p.n_times,n_side=p.n_side,n_observed=p.n_observed,n_calibration=p.n_calibration,seed=72)
    changed=sim.observation.copy();changed[p.final_origin:]+=1e6;changed[:,sim.test_indices]-=1e6
    other=replace(sim,observation=changed)
    a=select_configuration("basis_krr",sim,p,"B",8)
    b=select_configuration("basis_krr",other,p,"B",8)
    assert a["selected"]==b["selected"]
    assert [r["score"] for r in a["trials"]]==[r["score"] for r in b["trials"]]


def test_randomized_kernel_finalists_use_fresh_seeds_and_average_scores():
    p=profile("smoke",models=("basis_krr",),trials=2,finalist_count=2,finalist_seeds=2)
    sim=generate("g1",p.n_times,n_side=p.n_side,n_observed=p.n_observed,
                 n_calibration=p.n_calibration,seed=72)
    selected=select_configuration("basis_krr",sim,p,"A",8)
    trials,rechecks=selected["trials"],selected["rechecks"]
    assert len(trials)==2 and len(rechecks)==4
    assert len({row["fit_seed"] for row in trials+rechecks})==6
    means=[]
    for index in range(2):
        scores=[row["score"] for row in rechecks if row["candidate"]==index]
        assert len(scores)==2 and np.isfinite(scores).all()
        means.append(np.mean(scores))
    assert selected["selected_candidate"]==int(np.argmin(means))
    assert selected["compute_seconds"]==pytest.approx(sum(row["seconds"] for row in trials+rechecks))


def test_runner_resume_and_failed_model_population(tmp_path,monkeypatch):
    import bayes_stesn.benchmark.runner as runner
    original=runner.fit_model
    def broken(model,data,*args,**kwargs):
        assert isinstance(data,Panel)
        if model=="basis_krr":
            raise RuntimeError("deliberate fit failure")
        return original(model,data,*args,**kwargs)
    monkeypatch.setattr(runner,"fit_model",broken)
    p=profile("smoke",models=("persistence","basis_krr"),tracks=("A",))
    result=run_benchmark(p,tmp_path)
    failed=next(row for row in result["table"] if row["model_id"]=="basis_krr")
    assert failed["attempted"]==1 and failed["fallback"]==1 and failed["qualified"]==0
    assert failed["probability_population"]==0 and failed["crps"] is None
    again=run_benchmark(p,tmp_path)
    assert result==again
    with pytest.raises(ValueError):
        run_benchmark(replace(p,master_seed=71),tmp_path)


def test_block_calibration_ranks_storage_and_frozen_model(tmp_path):
    rng=np.random.default_rng(52)
    y=rng.normal(size=(200,4)).cumsum(axis=0)
    data=Panel(y,np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]]))
    model=fit_model("persistence",select_panel(data,stop=40))
    origins=tuple(range(40,165,5))
    calibration=calibrate_frozen_model(model,select_panel(data,stop=165),origins=origins,horizon=5,
                                      levels=(.9,.95,.99),n_paths=1)
    assert calibration["blocks"]==25
    assert np.isinf(calibration["maps"]["marginal_0.99"].quantiles).all()
    assert len(model.data.y)==40
    save_calibration(calibration,tmp_path/"calibration.npz")
    restored=load_calibration(tmp_path/"calibration.npz")
    forecast=model.forecast(5,history=select_panel(data,stop=165))
    result=evaluate_calibration(restored,forecast,y[165:170])
    assert result["calibrated"]["marginal_0.9"]["rank"]==24
    assert result["calibrated"]["simultaneous_0.95"]["rank"]==25
    assert not result["distribution_added"] and result["raw_samples_unchanged"]
    with pytest.raises(ValueError):
        calibrate_frozen_model(model,data,origins=(40,43),horizon=5)


def test_learning_curve_common_endpoints_and_roles():
    base=profile("pilot")
    curves=history_curve_profiles(base,(80,120,240))
    assert len({p.final_origin for p in curves})==1
    assert [p.initial_fit-p.training_start for p in curves]==[80,120,240]
    curves=sensor_curve_profiles(replace(base,n_side=9),(8,16,25))
    targets=[];observed=[]
    for p in curves:
        sim=generate("g1",4,n_side=p.n_side,n_observed=p.n_observed,n_calibration=p.n_calibration,
                     **p.generator_options["g1_gaussian_ar"])
        targets.append(sim.test_indices);observed.append(sim.observed_indices)
        assert p.score_observed_count==8
    np.testing.assert_array_equal(targets[0],targets[-1])
    np.testing.assert_array_equal(observed[0],observed[-1][:8])


def test_independent_replication_aggregation_and_power_freeze(tmp_path):
    rows=[]
    for rep in range(5):
        for name,shift in (("bayes_esn_ridge",.1),("persistence",.2)):
            rows.append({"model_id":name,"scenario":"g1_gaussian_ar","track":"A","replication":rep,
                         "completed":True,"qualified":True,"fallback":False,"selection_metric":"rmse",
                         "training_scale":2.,"operational_scores":{"rmse":rep/10+shift,"mae":.1},"native_scores":{}})
    summary=aggregate_results(rows,bootstrap_resamples=10)
    assert summary["table"][0]["attempted"]==5
    assert summary["paired_comparisons"][0]["rmse_population"]==5
    from bayes_stesn.benchmark.runner import write_json
    write_json(tmp_path/"summary.json",summary)
    write_json(tmp_path/"manifest.json",{"stage":"development","identity":"test"})
    freeze=freeze_replications(tmp_path,tmp_path/"freeze.json")
    assert freeze["R"]==20 and not freeze["cap_binding"]


def _development_power_fixture(directory):
    """Synthetic paired losses exercise gates only; no scientific run claimed."""
    from bayes_stesn.benchmark.runner import (write_json,jsonable,source_fingerprint,
        benchmark_environment,single_thread_runtime)
    from bayes_stesn.io import content_hash
    p=profile("pilot",models=("bayes_esn_ridge","persistence"),tracks=("A",),
              scenarios=("g1_gaussian_ar",))
    with single_thread_runtime():
        runtime=benchmark_environment()
    manifest=jsonable({"stage":"development","schema":"benchmark-run-3.0",
        "protocol":p.to_dict(),"protocol_identity":p.identity,
        "source_fingerprint":source_fingerprint(),"environment":runtime})
    manifest["identity"]=content_hash(manifest)
    rows=[]
    for rep in range(5):
        for model,loss in (("bayes_esn_ridge",.4+.02*rep),("persistence",.6+.03*rep)):
            rows.append({"model_id":model,"scenario":"g1_gaussian_ar","track":"A","replication":rep,
                "completed":True,"qualified":True,"fallback":False,"selection_metric":"rmse",
                "training_scale":1.,"operational_scores":{"rmse":loss,"mae":loss},"native_scores":{}})
    summary=jsonable(aggregate_results(rows,bootstrap_resamples=10))
    summary.update(source_stable=True,manifest_identity=manifest["identity"],stage="development")
    write_json(directory/"manifest.json",manifest);write_json(directory/"summary.json",summary)
    return profile("main",models=p.models,scenarios=p.scenarios,tracks=p.tracks)


def test_confirmation_factory_binds_full_design_source_runtime_and_cap(tmp_path):
    from bayes_stesn.io import content_hash
    draft=_development_power_fixture(tmp_path/"pilot")
    path=tmp_path/"freeze.json"
    frozen=freeze_confirmation(tmp_path/"pilot",path,protocol=draft)
    resolved=BenchmarkProtocol.load(path.with_suffix(".yaml"))
    verified=verify_confirmation_freeze(resolved,path)
    assert verified["identity"]==frozen["identity"]
    assert resolved.replications==20 and verified["power_status"]=="planned_mcse_target_within_budget"
    with pytest.raises(ValueError,match="differs"):
        verify_confirmation_freeze(replace(resolved,paths=resolved.paths+1),path)
    altered=dict(frozen);altered["source_fingerprint"]="changed"
    altered["identity"]=content_hash({k:v for k,v in altered.items() if k!="identity"})
    with pytest.raises(ValueError,match="source/runtime"):
        verify_confirmation_freeze(resolved,altered)
    capped=freeze_confirmation(tmp_path/"pilot",tmp_path/"capped.json",protocol=draft,target_mcse=1e-5)
    assert capped["power"]["R"]==60 and capped["power"]["cap_binding"]
    assert capped["power_status"]=="power_target_not_met_at_cap"
    assert verify_confirmation_freeze(BenchmarkProtocol(**capped["protocol"]),capped)["power_status"]==capped["power_status"]


def test_confirmation_api_and_cli_cannot_bypass_freeze(tmp_path,capsys):
    with pytest.raises(ValueError,match="requires freeze_confirmation"):
        run_benchmark(profile("main"),tmp_path/"api")
    assert not (tmp_path/"api").exists()
    from bayes_stesn.cli import main
    with pytest.raises(SystemExit) as stopped:
        main(["benchmark","run","--profile","main","--output",str(tmp_path/"cli")])
    assert stopped.value.code==2
    assert "requires freeze_confirmation" in capsys.readouterr().err
    assert not (tmp_path/"cli").exists()


def test_fhn_finite_grid_time_gate_does_not_claim_mesh_convergence():
    import scipy
    import bayes_stesn.benchmark.generators as generators
    from bayes_stesn.io import content_hash
    evidence={"schema":"benchmark-generator-validation-2.0",
        "generator_source_sha256":hashlib.sha256(Path(generators.__file__).read_bytes()).hexdigest(),
        "numerical_versions":{"numpy":np.__version__,"scipy":scipy.__version__},
        "n_observations":101,"observation_spacing":1.,"physical_burnin":200.,"fhn_mean_temporal_sd":1.,
        "fhn_time_halving":{"rmse":.001},
        "finite_grid_policy":{"target":"finite_semidiscrete","grid_size":128,"dt":.05,"refined_dt":.025,
            "continuum_claim":False,"time_gate_passed":True,"morphology_gate_passed":True,"mesh_gate_passed":False,
            "physical_parameters":{"a":.7,"b":.8,"epsilon":.08,"kappa":.2}}}
    evidence["identity"]=content_hash(evidence)
    accepted=_fhn_evidence(profile("main"),evidence)
    assert accepted["time_gate_passed"] and not accepted["mesh_gate_passed"] and not accepted["continuum_claim"]
    evidence["fhn_time_halving"]["rmse"]=.04
    evidence["identity"]=content_hash({k:v for k,v in evidence.items() if k!="identity"})
    with pytest.raises(ValueError,match="time-step error"):
        _fhn_evidence(profile("main"),evidence)


def test_stress_control_stream_pairing_and_final_independence():
    from bayes_stesn.benchmark.runner import dgp_seed_for
    p=profile("smoke")
    assert dgp_seed_for(p,"g4_existing_fronts",2)==dgp_seed_for(p,"g5_noise_stress",2)
    assert dgp_seed_for(p,"g2_spectral_transport",2)==dgp_seed_for(p,"g6_velocity_shift",2)
    assert dgp_seed_for(p,"g1_gaussian_ar",2)!=dgp_seed_for(profile("main"),"g1_gaussian_ar",2)


def test_parallel_spawn_matches_serial_scores(tmp_path):
    from bayes_stesn.benchmark.runner import _worker_limit
    if _worker_limit()<2:
        pytest.skip("Host has one permitted CPU")
    p=profile("smoke",models=("persistence","basis_krr","eqesn_rollout"),tracks=("A",),replications=2,
              model_options={"eqesn_rollout":{"width":9,"washout":6}})
    first=run_benchmark(p,tmp_path/"serial",workers=1)
    second=run_benchmark(p,tmp_path/"parallel",workers=2)
    assert first["jobs"]==second["jobs"]==6
    for serial in sorted((tmp_path/"serial").rglob("receipt.json")):
        parallel=tmp_path/"parallel"/serial.relative_to(tmp_path/"serial")
        a=json.loads(serial.read_text());b=json.loads(parallel.read_text())
        assert a["operational_scores"]==b["operational_scores"]
        assert a["native_scores"]==b["native_scores"]
        assert a["final_fit_seed"]==b["final_fit_seed"]
    resumed=run_benchmark(p,tmp_path/"parallel",workers=2)
    assert second==resumed


def test_worker_failure_retained_and_midrun_source_change_invalidated(tmp_path,monkeypatch):
    import bayes_stesn.benchmark.runner as runner
    p=profile("smoke",models=("persistence",),tracks=("A",))
    original=runner._execute_job
    def crash(*args,**kwargs):
        raise RuntimeError("simulated worker boundary failure")
    monkeypatch.setattr(runner,"_execute_job",crash)
    summary=run_benchmark(p,tmp_path/"worker_failure")
    assert summary["table"][0]["fallback"]==1 and summary["table"][0]["qualified"]==0
    receipt=json.loads(next((tmp_path/"worker_failure").rglob("receipt.json")).read_text())
    assert receipt["process_failure"] and receipt["incomplete_compute_accounting"]
    monkeypatch.setattr(runner,"_execute_job",original)
    fingerprints=iter(("stable","stable","changed"))
    monkeypatch.setattr(runner,"source_fingerprint",lambda:next(fingerprints))
    with pytest.raises(RuntimeError,match="changed during execution"):
        run_benchmark(p,tmp_path/"changed_source")
    changed=json.loads((tmp_path/"changed_source"/"summary.json").read_text())
    assert not changed["source_stable"] and changed["comparison_status"].startswith("invalidated")


def test_calibration_preserves_negligible_prefix_rounding_and_rejects_g6(tmp_path,monkeypatch):
    import bayes_stesn.benchmark.studies as studies
    p=profile("smoke",models=("persistence",),tracks=("A",))
    run_benchmark(p,tmp_path/"base")
    original=studies.generate
    def rounding(*args,**kwargs):
        sim=original(*args,**kwargs)
        observations=sim.observation.copy();observations[:p.n_times]+=1e-14
        return replace(sim,observation=observations)
    monkeypatch.setattr(studies,"generate",rounding)
    result=run_calibration_study(tmp_path/"base",tmp_path/"calibration",scenario="g1_gaussian_ar",
                                model_id="persistence",blocks=3,levels=(.5,))
    assert 0<result["extension_prefix_max_abs_discrepancy"]<1e-13
    assert result["parameter_cutoff"]==p.final_origin
    with pytest.raises(ValueError,match="future-break"):
        run_calibration_study(tmp_path/"base",tmp_path/"unsupported",scenario="g6_velocity_shift",model_id="persistence")
