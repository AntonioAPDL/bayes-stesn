"""Powered replication planning, matched learning curves and calibration runs."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from ..io import content_hash, seed_for
from .protocol import BenchmarkProtocol, profile
from .runner import run_benchmark, write_json, jsonable, source_fingerprint, benchmark_environment, single_thread_runtime
from .generators import Simulation, generate
from .registry import load_model, MODELS
from .contracts import select_panel
from .calibration import calibrate_frozen_model,evaluate_calibration,save_calibration


def _power_plan(manifest, summary, target_mcse, minimum, maximum, comparison_keys=None):
    if manifest["stage"]!="development":
        raise ValueError("Power planning requires independent development datasets, not test results")
    if not np.isfinite(target_mcse) or not 0<target_mcse or not 2<=minimum<=maximum:
        raise ValueError("Invalid MCSE target or replication bounds")
    requested=[];details=[]
    for row in summary["paired_comparisons"]:
        for metric in ("rmse","crps"):
            key=(row["scenario"],row["track"],row["other_model"],metric)
            if comparison_keys is not None and key not in comparison_keys:
                continue
            value=row.get(metric,{})
            differences=np.asarray(value.get("paired_differences",[]),float)
            if len(differences)<3:
                continue
            if differences.ndim!=1 or not np.isfinite(differences).all():
                raise ValueError("Pilot paired differences must be finite one-dimensional arrays")
            variance=float(np.var(differences,ddof=1))
            required=max(minimum,int(math.ceil(variance/target_mcse**2)))
            requested.append(required)
            details.append({"scenario":row["scenario"],"track":row["track"],
                            "other_model":row["other_model"],"metric":metric,
                            "pilot_pairs":len(differences),"normalized_variance":variance,
                            "requested_R":required,"mcse_at_cap":float(np.sqrt(variance/maximum))})
    if not requested:
        raise ValueError("At least three paired development replications are needed for a primary comparison")
    count=min(maximum,max(requested))
    freeze={"schema":"benchmark-power-freeze-1.0","pilot_identity":manifest["identity"],
            "target_normalized_mcse":target_mcse,"R":count,"minimum":minimum,"maximum":maximum,
            "uncapped_R":max(requested),"cap_binding":max(requested)>maximum,"comparisons":details,
            "meaning":"Approximate MCSE planning from pilot variances; no guarantee at final DGPs/configurations",
            "fixed_before_final_seeds":True}
    freeze["identity"]=content_hash(freeze)
    return freeze


def freeze_replications(pilot_directory, output=None, *, target_mcse=.02, minimum=20, maximum=60):
    """Power-only planning utility; it does not authorize confirmation runs.

    ``freeze_confirmation`` additionally binds the exact design, source,
    numerical runtime and generator evidence and writes a runnable YAML.
    """
    pilot_directory=Path(pilot_directory)
    manifest=json.loads((pilot_directory/"manifest.json").read_text())
    summary=json.loads((pilot_directory/"summary.json").read_text())
    freeze=_power_plan(manifest,summary,target_mcse,minimum,maximum)
    if output is not None:
        write_json(output,freeze)
    return freeze


def _verify_identity(value, name):
    if not isinstance(value,dict) or "identity" not in value:
        raise ValueError(f"{name} has no verified content identity")
    body={key:item for key,item in value.items() if key!="identity"}
    if content_hash(body)!=value["identity"]:
        raise ValueError(f"{name} content does not match its identity")


def _required_comparisons(protocol):
    required=set()
    for scenario in protocol.scenarios:
        for track in protocol.tracks:
            models=[m for m in protocol.models if track in MODELS[m].tracks]
            if track=="B" and "spatial_persistence" not in models:
                models.append("spatial_persistence")
            if "bayes_esn_ridge" not in models:
                continue
            for other in models:
                if other=="bayes_esn_ridge":
                    continue
                required.add((scenario,track,other,"rmse"))
                if MODELS[other].uncertainty!="point":
                    required.add((scenario,track,other,"crps"))
    if not required:
        raise ValueError("Registered confirmation power planning needs Bayesian ESN plus at least one compatible comparator")
    return required


def _design_views(protocol):
    dimensions={key:getattr(protocol,key) for key in ("initial_fit","training_start","horizon","validation_blocks",
                    "n_side","n_observed","n_calibration","score_observed_count","replications")}
    dimensions.update(n_times=protocol.n_times,final_origin=protocol.final_origin,domain_sites=protocol.n_side**2)
    budgets={key:getattr(protocol,key) for key in ("trials","finalist_count","finalist_seeds","paths",
                  "mcmc_chains","mcmc_warmup","mcmc_draws","mcmc_continuation","mcmc_functional_draws",
                  "ensemble_members","neural_epochs","score_pairs")}
    return dimensions,budgets


def _fhn_evidence(protocol, evidence):
    if "g3_excitable_field" not in protocol.scenarios:
        return None
    if evidence is None:
        raise ValueError("Confirmation with G3 requires time-step validation evidence; run tools/validate_generators.py --extended")
    value=json.loads(Path(evidence).read_text()) if isinstance(evidence,(str,Path)) else dict(evidence)
    _verify_identity(value,"FHN numerical evidence")
    if value.get("schema")!="benchmark-generator-validation-2.0":
        raise ValueError("FHN evidence must use the versioned numerical validation schema")
    import bayes_stesn.benchmark.generators as generator_module
    current_hash=hashlib.sha256(Path(generator_module.__file__).read_bytes()).hexdigest()
    if value.get("generator_source_sha256")!=current_hash:
        raise ValueError("FHN evidence does not match the current generator implementation")
    import scipy
    if value.get("numerical_versions")!={"numpy":np.__version__,"scipy":scipy.__version__}:
        raise ValueError("FHN evidence uses a different NumPy/SciPy runtime")
    options=(protocol.generator_options or {}).get("g3_excitable_field",{})
    policy=value.get("finite_grid_policy",{})
    grid=int(options.get("grid_size",128));dt=float(options.get("dt",.05));delta=float(options.get("delta",1.))
    if (policy.get("target")!="finite_semidiscrete" or policy.get("continuum_claim") is not False or
            policy.get("grid_size")!=grid or not np.isclose(policy.get("dt",np.nan),dt,rtol=0,atol=1e-14) or
            not np.isclose(policy.get("refined_dt",np.nan),dt/2,rtol=0,atol=1e-14)):
        raise ValueError("FHN evidence does not validate the declared finite grid and time step")
    physical={key:float(options.get(key,default)) for key,default in (("a",.7),("b",.8),("epsilon",.08),("kappa",.2))}
    if policy.get("physical_parameters")!=physical:
        raise ValueError("FHN evidence uses different physical parameters")
    duration=(value.get("n_observations",0)-1)*value.get("observation_spacing",0)
    if duration+1e-12<(protocol.horizon-1)*delta:
        raise ValueError("FHN time-step pilot is shorter than the intended forecast window")
    burnin=options.get("burnin",(200.,400.))
    earliest_burnin=float(burnin if np.ndim(burnin)==0 else min(burnin))
    if value.get("physical_burnin",-1)<earliest_burnin:
        raise ValueError("FHN numerical pilot precedes the declared post-burn-in regime")
    rmse=value.get("fhn_time_halving",{}).get("rmse",np.inf)
    if not np.isfinite(rmse) or rmse>protocol.fhn_time_tolerance or not policy.get("time_gate_passed"):
        raise ValueError("FHN time-step error did not pass the frozen tolerance")
    if not policy.get("morphology_gate_passed") or value.get("fhn_mean_temporal_sd",0)<=.05:
        raise ValueError("FHN pilot did not retain nontrivial temporal activity")
    return {"evidence":value,"target_policy":protocol.fhn_target_policy,
            "time_gate_passed":True,"time_rmse":float(rmse),"time_tolerance":protocol.fhn_time_tolerance,
            "mesh_gate_passed":bool(policy.get("mesh_gate_passed")),"continuum_claim":False,
            "scope":"fixed semidiscrete spatial grid; time-step error checked on the declared independent pilot window"}


def freeze_confirmation(pilot_directory, output, *, protocol=None, fhn_validation=None,
                        resolved_config_output=None, target_mcse=.02):
    """Write a self-contained frozen confirmation design from development only.

    Every registered primary comparison needs at least three independent
    paired pilot differences. The worst planned MCSE determines R in[20,60].
    If60 is insufficient, ``power_target_not_met_at_cap`` remains explicit;
    it is never represented as achievement of the precision target.
    """
    pilot_directory=Path(pilot_directory)
    manifest=json.loads((pilot_directory/"manifest.json").read_text())
    summary=json.loads((pilot_directory/"summary.json").read_text())
    _verify_identity(manifest,"Development manifest")
    if manifest.get("stage")!="development" or summary.get("source_stable") is not True:
        raise ValueError("A completed, source-stable development run is required before confirmation")
    if summary.get("manifest_identity")!=manifest["identity"]:
        raise ValueError("Development summary and manifest do not match")
    draft=profile("main") if protocol is None else protocol
    if not isinstance(draft,BenchmarkProtocol) or draft.stage!="confirmation":
        raise ValueError("The requested frozen protocol must be a validated confirmation profile")
    with single_thread_runtime():
        runtime=benchmark_environment()
    source=source_fingerprint()
    if manifest.get("source_fingerprint")!=source or manifest.get("environment")!=runtime:
        raise ValueError("Development results must use the current source and one-thread numerical runtime before freezing")
    required=_required_comparisons(draft)
    power=_power_plan(manifest,summary,target_mcse,20,60,required)
    available={(r["scenario"],r["track"],r["other_model"],r["metric"]) for r in power["comparisons"]}
    missing=required-available
    if missing:
        raise ValueError(f"At least three paired development results are missing for primary comparisons: {sorted(missing)}")
    frozen=replace(draft,replications=power["R"])
    fhn=_fhn_evidence(frozen,fhn_validation)
    dimensions,budgets=_design_views(frozen)
    receipt=jsonable({"schema":"benchmark-confirmation-freeze-2.0","protocol":frozen.to_dict(),
        "protocol_identity":frozen.identity,"source_fingerprint":source,"environment":runtime,
        "dimensions":dimensions,"candidate_budgets":budgets,"power":power,
        "power_status":"power_target_not_met_at_cap" if power["cap_binding"] else "planned_mcse_target_within_budget",
        "pilot_manifest":manifest,"pilot_summary":summary,"pilot_summary_sha256":content_hash(summary),
        "primary_comparisons":[list(key) for key in sorted(required)],"fhn_validation":fhn,
        "seed_policy":"development_dgp and final_dgp are independent namespaces; g5/g4 and g6/g2 share control DGP streams",
        "creation_policy":"No final simulation or forecast was generated by this factory"})
    receipt["identity"]=content_hash(receipt)
    output=Path(output)
    config_path=Path(resolved_config_output) if resolved_config_output else output.with_suffix(".yaml")
    if config_path.resolve()==output.resolve():
        raise ValueError("Resolved configuration and freeze receipt must use different paths")
    frozen.save(config_path)
    write_json(output,receipt)
    return receipt


def verify_confirmation_freeze(protocol, receipt):
    """Enforce the same complete confirmation gate in Python and the CLI."""
    if receipt is None:
        raise ValueError("Confirmation requires freeze_confirmation development evidence; pass power_freeze to the API or --power-freeze to the CLI")
    value=json.loads(Path(receipt).read_text()) if isinstance(receipt,(str,Path)) else jsonable(dict(receipt))
    _verify_identity(value,"Confirmation freeze")
    if value.get("schema")!="benchmark-confirmation-freeze-2.0":
        raise ValueError("A power-only receipt is insufficient; use freeze_confirmation")
    frozen=BenchmarkProtocol(**value["protocol"])
    if protocol.stage!="confirmation" or frozen.identity!=protocol.identity or value.get("protocol_identity")!=protocol.identity:
        raise ValueError("Confirmation protocol differs from its frozen dimensions, options, seed or candidate budget")
    with single_thread_runtime():
        runtime=benchmark_environment()
    if value.get("source_fingerprint")!=source_fingerprint() or value.get("environment")!=runtime:
        raise ValueError("Confirmation source/runtime differs from the pre-final freeze; rerun development and freeze the new design")
    dimensions,budgets=_design_views(protocol)
    if value.get("dimensions")!=dimensions or value.get("candidate_budgets")!=budgets:
        raise ValueError("Frozen dimension/budget declarations disagree with the protocol")
    pilot=value["pilot_manifest"];summary=value["pilot_summary"]
    _verify_identity(pilot,"Embedded development manifest")
    if (pilot.get("stage")!="development" or pilot.get("source_fingerprint")!=value["source_fingerprint"] or
            pilot.get("environment")!=runtime or summary.get("source_stable") is not True or
            summary.get("manifest_identity")!=pilot["identity"] or content_hash(summary)!=value["pilot_summary_sha256"]):
        raise ValueError("Embedded pilot evidence does not match the frozen development run")
    required=_required_comparisons(protocol)
    if [list(key) for key in sorted(required)]!=value.get("primary_comparisons"):
        raise ValueError("Frozen primary comparison set is inconsistent")
    power=value["power"]
    _verify_identity(power,"Embedded power plan")
    expected=_power_plan(pilot,summary,power["target_normalized_mcse"],20,60,required)
    available={(r["scenario"],r["track"],r["other_model"],r["metric"]) for r in expected["comparisons"]}
    if required-available:
        raise ValueError("Frozen pilot lacks three paired results for every primary comparison")
    if expected["identity"]!=power["identity"] or protocol.replications!=power["R"]:
        raise ValueError("Frozen replication count does not follow the recorded pilot variance rule")
    status="power_target_not_met_at_cap" if power["cap_binding"] else "planned_mcse_target_within_budget"
    if value.get("power_status")!=status:
        raise ValueError("Power-cap status was misreported")
    fhn=value.get("fhn_validation")
    verified=_fhn_evidence(protocol,None if fhn is None else fhn.get("evidence"))
    if verified!=fhn:
        raise ValueError("Frozen FHN numerical policy/evidence does not match")
    return value


def history_curve_profiles(base: BenchmarkProtocol, lengths=(500,1000,2000,4000)):
    """Fixed chronological endpoints; vary the initial historical window only."""
    lengths=tuple(map(int,lengths));endpoint=max(lengths)
    if min(lengths)<40:
        raise ValueError("Each history window needs at least40 rows")
    return [replace(base,name=f"{base.name}_history_{length}",initial_fit=endpoint,
                    training_start=endpoint-length) for length in lengths]


def sensor_curve_profiles(base: BenchmarkProtocol, counts=(25,50,100,200)):
    """Nested O, fixed master-reserved U, common observed-site score domain."""
    counts=tuple(map(int,counts));master=max(counts)
    if master+base.n_calibration>=base.n_side**2:
        raise ValueError("Sensor curve needs a grid with fixed disjoint U beyond the largest sensor set")
    configs=[]
    for count in counts:
        options={key:dict(value) for key,value in (base.generator_options or {}).items()}
        for scenario in base.scenarios:
            options.setdefault(scenario,{})["master_observed"]=master
        configs.append(replace(base,name=f"{base.name}_sensors_{count}",n_observed=count,
                               score_observed_count=min(counts),generator_options=options))
    return configs


def run_curve(base, output, *, kind, values, progress=None, workers=1, max_memory_mb=None):
    if base.stage=="confirmation":
        raise ValueError("Confirmation learning curves need a separately frozen receipt for each resolved profile; use development curves to plan them")
    profiles=history_curve_profiles(base,values) if kind=="history" else sensor_curve_profiles(base,values) if kind=="sensors" else None
    if profiles is None:
        raise ValueError("kind must be history or sensors")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    write_json(output/"curve_design.json",{"kind":kind,"values":values,
          "profiles":[p.to_dict() for p in profiles],"common_endpoint":True,
          "fixed_U":kind=="sensors","separate_experiments_not_full_factorial":True})
    return {p.name:run_benchmark(p,output/p.name,progress=progress,workers=workers,
                               max_memory_mb=max_memory_mb) for p in profiles}


def run_calibration_study(run_directory, output, *, scenario, model_id, replication=0,
                          track="A", blocks=25, levels=(.5,.8,.9,.95), progress=None):
    """Extend a saved dataset; freeze base parameters; compare raw/calibrated
    intervals at one common later origin. No labels from U enter calibration.
    """
    if scenario=="g6_velocity_shift":
        raise ValueError("G6 calibration is not the registered blind future-break task: later blocks would learn the break. A separately named recovery study is required.")
    root=Path(run_directory);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/"manifest.json").read_text());p=BenchmarkProtocol(**manifest["protocol"])
    if p.training_start:
        raise ValueError("Calibration extension currently uses an uncropped base training prefix")
    base=root/scenario/f"rep_{replication:03d}"
    receipt=json.loads((base/track/model_id/"receipt.json").read_text())
    if not receipt["qualified"]:
        raise ValueError("The base fit must pass its native gate before calibration")
    if blocks<1 or track not in {"A","B"}:
        raise ValueError("Invalid calibration blocks/track")
    short=Simulation.load(base/"simulation.npz")
    model=load_model(base/track/model_id/"model.npz")
    origin=p.final_origin+blocks*p.horizon
    options=dict((p.generator_options or {}).get(scenario,{}))
    extended=generate(scenario,origin+p.horizon,n_side=p.n_side,n_observed=p.n_observed,
                      n_calibration=p.n_calibration,seed=receipt["dgp_seed"],**options)
    prefix=extended.observation[:len(short.observation)]
    discrepancy=float(np.max(np.abs(short.observation-prefix)))
    if not np.allclose(short.observation,prefix,rtol=1e-13,atol=1e-13):
        raise ValueError("Generator extension changed the prefix; calibration cannot reuse this fitted model")
    if (not np.array_equal(short.coordinates,extended.coordinates) or
            not np.array_equal(short.times,extended.times[:len(short.times)]) or
            not np.array_equal(short.observed_indices,extended.observed_indices)):
        raise ValueError("Generator extension changed coordinates, time identifiers or observed roles")
    # Reuse the original floating prefix exactly after accepting only harmless
    # matrix-kernel rounding. This preserves the fitted-history identity and
    # keeps evaluator truth consistent with the observations used for fitting.
    if discrepancy:
        changes={}
        for field in ("observation","latent","generating_mean","hidden_state"):
            old=getattr(short,field);new=getattr(extended,field)
            if old is not None:
                if new is None or not np.allclose(old,new[:len(old)],rtol=1e-13,atol=1e-13):
                    raise ValueError(f"Generator extension changed hidden {field} beyond numerical rounding")
                joined=np.array(new,copy=True);joined[:len(old)]=old;changes[field]=joined
        metadata=dict(extended.metadata)
        metadata["observation_sha256"]=hashlib.sha256(np.ascontiguousarray(changes["observation"],dtype="<f8").tobytes()).hexdigest()
        metadata["preserved_prefix_rows"]=len(short.observation)
        changes["metadata"]=metadata
        extended=replace(extended,**changes)
    extended.save(output/"simulation.npz")
    origins=tuple(range(p.final_origin,origin,p.horizon))
    history=extended.observed_panel(origin)
    args={}
    if track=="B":
        c=extended.calibration_indices
        if not len(c) or set(c)&set(extended.test_indices):
            raise ValueError("Track B needs disjoint reserved C labels and U targets")
        args=dict(calibration_coordinates=extended.coordinates[c],
                  calibration_site_ids=tuple(extended.site_ids[j] for j in c),
                  calibration_observations=extended.observation[:origin,c])
    calibration=calibrate_frozen_model(model,history,origins=origins,horizon=p.horizon,
                  levels=levels,n_paths=p.paths,seed=seed_for(p.master_seed,"calibration",scenario,replication,track,model_id),**args)
    save_calibration(calibration,output/"calibration.npz")
    targets=extended.observed_indices if track=="A" else extended.test_indices
    forecast=model.forecast(p.horizon,target_coordinates=extended.coordinates[targets],
                  target_site_ids=tuple(extended.site_ids[j] for j in targets),
                  history=history,n_paths=p.paths,seed=seed_for(p.master_seed,"late_forecast",scenario,replication,track,model_id))
    forecast.save(output/"raw_forecast.npz")
    results=evaluate_calibration(calibration,forecast,extended.observation[origin:origin+p.horizon,targets],transfer=track=="B")
    from ..io import save_state
    save_state(output/"bands.npz",results.pop("bands"),kind="calibrated_bands")
    results.update(origin_index=origin,parameter_cutoff=p.final_origin,calibration_origins=origins,
                   scenario=scenario,model_id=model_id,track=track,base_manifest_identity=manifest["identity"],
                   role_policy="C only for new-site calibration; U sealed",blocks=blocks,
                   extension_prefix_max_abs_discrepancy=discrepancy,
                   extension_prefix_tolerance={"rtol":1e-13,"atol":1e-13},
                   extension_prefix_policy="original prefix retained exactly after tight numerical verification")
    write_json(output/"summary.json",results)
    if progress:
        progress({"event":"calibration_complete","origin":origin,"blocks":blocks,"track":track})
    return results
