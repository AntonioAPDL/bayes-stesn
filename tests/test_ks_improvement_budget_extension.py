"""Synthetic budget migration: no real model, cache, fit or score is opened."""
from collections import Counter
from datetime import datetime, timedelta, timezone
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


M = load(ROOT/"tools/extend_ks_improvement_budget.py", "budget_migration_test")
F = load(ROOT/"tests/test_ks_improvement_fits.py", "budget_native_fixture")
COMPACT = load(ROOT/"tools/package_ks_improvement_checkpoint.py", "budget_compact_fixture")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(M.N.encoded(value))


def fixture(folder):
    folder = Path(folder); source, exp, candidate = F.fixture(folder)
    cache = folder/"cache"; cache.mkdir()
    when = datetime.now(timezone.utc)
    start = (when-timedelta(hours=2)).isoformat(); finish = (when-timedelta(minutes=3)).isoformat()
    cfg = exp["config"]
    cfg["tracks"] = ["A"]
    cfg["registry"]["identity"] = "fixture-registry"
    cfg["forecast_seed"] = 27
    cfg["budget"] = {"campaign_started_utc": start, "total_worker_seconds_cap": 43200,
        "wall_seconds_cap": 14400, "search_worker_seconds_cap": 21600, "workers": 4,
        "preparation_worker_seconds": 90., "disk_reserve_bytes": 0,
        "threads_per_worker": 1, "extension_reason": "earlier finite pilot-based allowance"}
    exp["source"]["application"] = {"ks_metrics.py": "0"*64}
    exp["identity"] = M.N.identity({"config": cfg, "source": exp["source"], "cache": exp["cache_identity"]})
    exp["created_utc"] = (when-timedelta(hours=1, minutes=20)).isoformat()
    write(source/"experiment.json", exp)
    frozen = {"experiment_identity": exp["identity"], "test_access": False,
        "selected": {"A": {"main": candidate, "anchor_id": candidate["id"], "sensitivity": None}},
        "created_utc": (when-timedelta(hours=1)).isoformat(), "history": 512, "paths": 512,
        "confirmation_trajectories": [1,2,3,6,7,8,11,12,13,14,16,17,18,19],
        "confirmation_origins": list(range(6000,8000,100))}
    frozen["identity"] = M.N.identity(frozen); write(source/"freeze.json", frozen)
    write(source/"history_policy.json", {"history": 512, "passed_jobs": ["historical-job-id"], "failed_jobs": []})
    write(source/"selection/promote_A.json", {"rows": [{"id": candidate["id"], "eligible": True}], "created_utc": finish})
    jobs = {}
    for trajectory, state in enumerate(("complete", "failed", "inference-unqualified", "pending", "preflight", "ledger-only")):
        if state in {"complete", "failed", "inference-unqualified"}:
            original, directory, _ = F.native(source, exp, candidate, trajectory=trajectory, qualified=state!="inference-unqualified")
        elif state == "pending":
            original, directory, _ = F.partial(source, exp, candidate, trajectory=trajectory)
        else:
            original, directory = F.job_for(source, exp, candidate, trajectory=trajectory)
        (source/"jobs"/(original["job_id"]+".json")).unlink()
        j = {**original, "output": str(source), "cache": str(cache),
            "stage": "preflight" if state=="preflight" else "confirm", "partition": "validation",
            "origins": [] if state=="preflight" else [6000], "history": 512,
            "paths": 512, "forecast_seed": 27, "path_batch": 256, "retain": False,
            "lengths": [128,256,512]}
        j["job_id"] = M.N.identity({k:v for k,v in j.items() if k!="job_id"})[:24]
        jobs[state] = j
        if state != "ledger-only":write(source/"jobs"/(j["job_id"]+".json"), j)
        write(source/"ledger"/(j["job_id"]+".json"), {"state": "running" if state=="pending" else "pending", "job": j})
        if state != "ledger-only":
            write(source/"attempts"/(j["job_id"]+"_10.json"), {"job_id": j["job_id"], "stage": j["stage"],
                "state": "interrupted" if state=="pending" else "complete", "returncode": 1 if state=="pending" else 0,
                "finished_utc": finish, "elapsed_seconds": .1*(trajectory+1)})
        if state in {"complete", "failed", "inference-unqualified", "preflight"}:
            terminal = {"job_id":j["job_id"], "job_identity":M.N.identity({k:v for k,v in j.items() if k!="job_id"}),
                "state":"complete" if state=="preflight" else state, "finished_utc":finish, "elapsed_seconds":.1}
            if state=="preflight":
                result={"training_identity":{"trajectory":trajectory,"transform":{"fit_time_bounds":[0,6000]}},
                    "geometry":{"fit_identity":"feature-fixture"}, "synchronization":{"fit_identity":"feature-fixture"}}
                write(source/"preflight"/(j["fit_key"]+".json"), result);terminal["result"]=result
            elif state in {"complete", "failed"}:
                native = json.loads((directory/"receipt.json").read_text())["native_fit_identity"]
                forecast={"native_fit":native,"first_target":6000,"forecast_namespace":"main", "qualified":True,
                    "state":state, "metrics":{"synthetic_uninspected_score":3.}, "seconds":.01,
                    "stage_first_scored":"confirm", "candidate_id":candidate["id"]}
                forecast.update({k:j[k] for k in ("fit_key","track","trajectory","seed","history","paths","forecast_seed","partition")})
                forecast["identity"]=M.N.identity({"native_fit":native,"first":6000,"history":512,"horizon":100,
                    "paths":512,"forecast_seed":27,"partition":"validation","score_source":"0"*64})
                write(source/"forecasts"/j["fit_key"]/f"06000_512_{forecast['identity'][:12]}.json", forecast)
                terminal.update(completed_origins=int(state=="complete"),failed_origins=int(state=="failed"))
            write(source/"terminal"/(j["job_id"]+".json"),terminal)
    # Prior-migration attempts deliberately have no current job counterpart.
    write(source/"attempts/000_prior_migration.json",{"job_id":"old-v1-job","stage":"preflight","state":"complete", "returncode":0,"finished_utc":finish,"elapsed_seconds":1e-12})
    write(source/"attempts/zzz_prior_interruption.json",{"job_id":"old-v1-interrupted","stage":"screen","state":"interrupted", "returncode":1,"finished_utc":finish,"elapsed_seconds":100.})
    write(source/"migration/receipt.json",{"passed":True,"destination_experiment":exp["identity"],"prior_fixture":True})
    (source/"examples").mkdir();np.savez_compressed(source/"examples/example.npz",mean=np.arange(4.))
    (source/"logs").mkdir();(source/"logs/old_closed.log").write_text("closed fixture\n")
    (source/"fits"/jobs["complete"]["fit_key"]/".fit.npz.bstesn-fixture-orphan.npz").write_bytes(b"retained source staging")
    previous=folder/"old.json";write(previous,cfg)
    projection=folder/"projection.json";write(projection,{"experiment_identity":exp["identity"],"freeze_identity":frozen["identity"], "confirmation_score_objects_decoded":0,"npz_payload_reads":0})
    new=copy.deepcopy(cfg);authorized=(when-timedelta(minutes=10)).isoformat()
    ext={"schema":"ks-improvement-prospective-budget-extension/1","one_prospective_extension":True,
        "original_total_worker_seconds_cap":43200,"original_wall_seconds_cap":14400,"search_worker_seconds_cap_unchanged":21600,
        "original_campaign_started_utc":start,"previous_budget":copy.deepcopy(cfg["budget"]),
        "previous_configuration_sha256":M.N.digest(previous),"timing_receipt_sha256":M.N.digest(projection),
        "frozen_selection_identity":frozen["identity"],"frozen_science_unchanged":True,
        "confirmation_scores_used_for_decision":False,"original_start_and_all_spent_time_carried":True,
        "authorized_utc":authorized,"recorded_utc":authorized,"reason":"fixture prospective allowance"}
    new["budget"].update(total_worker_seconds_cap=54000,wall_seconds_cap=18000,extension_reason=ext["reason"],operational_extension=ext)
    config=folder/"new.json";write(config,new)
    decision=folder/"decision.json";d={"schema":"ks-improvement-budget-extension-decision/1",
        "timing_receipt_sha256":M.N.digest(projection),"frozen_selection_identity":frozen["identity"],
        "previous_configuration_sha256":M.N.digest(previous),"extended_configuration_sha256":M.N.digest(config),
        "original_campaign_started_utc":start,"confirmation_scores_used_for_decision":False,"one_prospective_extension":True,
        "previous_caps":{"total_worker_seconds_cap":43200,"wall_seconds_cap":14400,"search_worker_seconds_cap":21600},
        "extended_caps":{"total_worker_seconds_cap":54000,"wall_seconds_cap":18000,"search_worker_seconds_cap":21600}}
    d["identity"]=M.N.identity(d);write(decision,d)
    closure=folder/"closure.json";write(closure,{"source":"actual_unified_exec_write_stdin_completion_results",
        "observed_utc":(when-timedelta(minutes=1)).isoformat(),"sessions":[{"session_id":42,"root":str(source),"observed_completed":True,"exit_code":0}]})
    args=(source,folder/"new-run",config,cache,closure,42,projection,decision,previous)
    return args,jobs,exp,frozen


def opaque_sidecar_fixture(args, job):
    source=args[0];path=source/"fits"/job["fit_key"]/".fit.npz.A1b2C3"
    path.write_bytes(b"opaque historical bytes; not a native checkpoint")
    relative=path.relative_to(source).as_posix()
    entry={"path":relative,"bytes":path.stat().st_size,"sha256":M.N.digest(path),
        "disposition":"preserve_opaque_outside_active_fits"}
    historical_path=source/"migration/receipt.json";historical=M.read(historical_path)
    historical.setdefault("files",{})[relative]={k:entry[k] for k in ("bytes","sha256")}
    write(historical_path,historical)
    manifest={"schema":"ks-reviewed-fit-sidecars/1",
        "source_experiment_identity":M.read(source/"experiment.json")["identity"],
        "historical_migration_receipt_sha256":M.N.digest(historical_path),
        "files":[entry],"creator":"unproven","rationale":"Synthetic preservation fixture"}
    manifest["identity"]=M.N.identity(manifest)
    manifest_path=source.parent/"reviewed-sidecars.json";write(manifest_path,manifest)
    return path,manifest_path


class BudgetExtensionContracts(unittest.TestCase):
    def test_all_states_payloads_freeze_and_historical_charges_survive_rebinding(self):
        with tempfile.TemporaryDirectory() as folder:
            args,jobs,old,frozen=fixture(folder);source,dest=args[:2]
            before={p.relative_to(source).as_posix():M.N.digest(p) for p in source.rglob("*") if p.is_file()}
            original_glob=Path.glob
            def reversed_glob(path,pattern):return iter(reversed(list(original_glob(path,pattern))))
            with patch.object(M,"verify_runtime_cache",return_value={"fixture":True}), patch.object(Path,"glob",reversed_glob):
                result=M.migrate(*args,apply=True)
            receipt=M.read(dest/"migration/receipt.json")
            self.assertTrue(result["passed"]);self.assertEqual(result["attempt_files"],7)
            self.assertEqual(result["carried_worker_seconds"],191.500000000001)
            self.assertEqual(result["wall_budget_start"],old["config"]["budget"]["campaign_started_utc"])
            self.assertNotEqual(result["wall_budget_start"],old["created_utc"])
            newfreeze=M.read(dest/"freeze.json")
            self.assertEqual({k:v for k,v in newfreeze.items() if k not in ("identity","experiment_identity")},
                {k:v for k,v in frozen.items() if k not in ("identity","experiment_identity")})
            self.assertEqual(result["terminal_states"],{"complete":2,"failed":1,"inference-unqualified":1})
            self.assertEqual(result["destination_terminal_states"]["pending"],2)
            self.assertEqual(result["forecast_receipt_states"],{"complete":1,"failed":1})
            for key,oldjob in jobs.items():
                nid=receipt["job_id_mapping"][oldjob["job_id"]];j=M.read(dest/"jobs"/(nid+".json"))
                self.assertNotEqual(nid,oldjob["job_id"])
                self.assertEqual(j["job_id"],M.N.identity({k:v for k,v in j.items() if k!="job_id"})[:24])
                self.assertEqual(j["fit_key"],oldjob["fit_key"])
                self.assertEqual(j["output"],str(dest))
                if key not in ("pending","ledger-only"):
                    terminal=M.read(dest/"terminal"/(nid+".json"));self.assertEqual(terminal["finished_utc"],M.read(source/"terminal"/(oldjob["job_id"]+".json"))["finished_utc"])
                    self.assertEqual(terminal["job_identity"],M.N.identity({k:v for k,v in j.items() if k!="job_id"}))
            for relative,digest in before.items():
                self.assertEqual(M.N.digest(source/relative),digest)
                if relative.startswith(("attempts/","forecasts/","examples/","selection/")) or relative=="history_policy.json":
                    self.assertEqual(M.N.digest(dest/relative),digest)
                if relative.startswith("fits/") and Path(relative).name in M.N.CANONICAL:
                    self.assertEqual(M.N.digest(dest/relative),digest)
            self.assertFalse(any("bstesn" in p.name for p in dest.rglob("*")))
            # Existing compact reader accepts the new binding without any relaxation.
            diagnosis=Path(folder)/"diagnosis.json";write(diagnosis,{"status":"complete","test_label_reads":0,"scope":"fixture"})
            package=COMPACT.create_snapshot(dest,Path(folder)/"exports",diagnosis)
            self.assertTrue(COMPACT.verify_snapshot(Path(package["manifest"]))["verified"])
            report=load(ROOT/"tools/build_ks_improvement_report.py","budget_report_reader")
            report.ReceiptReader(dest)

    def test_dry_run_is_metadata_only_and_busy_existing_locks_refuse(self):
        with tempfile.TemporaryDirectory() as folder:
            args,jobs,_,_=fixture(folder)
            with patch.object(M,"verify_runtime_cache",side_effect=AssertionError("runtime/cache read")), patch.object(M.N,"validate_fit",side_effect=AssertionError("binary payload read")):
                result=M.migrate(*args)
                self.assertEqual(result["binary_payload_reads"],0);self.assertFalse(args[1].exists())
                for lock in (args[0]/"scheduler.lock",args[0]/"fits"/jobs["complete"]["fit_key"]/"writer.lock"):
                    with lock.open("rb") as f:
                        fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                        with self.assertRaises(BlockingIOError):M.migrate(*args,apply=True)
                        fcntl.flock(f.fileno(),fcntl.LOCK_UN)

    def test_stale_closure_or_unfinished_attempt_cannot_be_adopted(self):
        with tempfile.TemporaryDirectory() as folder:
            args,_,_,_=fixture(folder);closure=M.read(args[4])
            closure["observed_utc"]="2000-01-01T00:00:00+00:00";write(args[4],closure)
            with self.assertRaisesRegex(ValueError,"predates"):M.migrate(*args)
        with tempfile.TemporaryDirectory() as folder:
            args,_,_,_=fixture(folder);p=next((args[0]/"attempts").glob("*.json"));a=M.read(p);a.pop("finished_utc");write(p,a)
            with self.assertRaisesRegex(ValueError,"closed exact charge"):M.migrate(*args)

    def test_scientific_change_start_reset_and_extra_resource_change_are_refused(self):
        for change in ("science","start","workers"):
            with tempfile.TemporaryDirectory() as folder:
                args,_,_,_=fixture(folder);c=M.read(args[2])
                if change=="science":c["forecast_seed"]+=1
                elif change=="start":c["budget"]["campaign_started_utc"]=datetime.now(timezone.utc).isoformat()
                else:c["budget"]["workers"]=5
                write(args[2],c)
                with self.assertRaisesRegex(ValueError,"Scientific configuration|Original start"):M.migrate(*args)

    def test_native_receipt_and_forecast_identity_corruption_fail_before_publish(self):
        for corrupt in ("native","forecast"):
            with tempfile.TemporaryDirectory() as folder:
                args,jobs,_,_=fixture(folder);source=args[0]
                if corrupt=="native":
                    p=source/"fits"/jobs["complete"]["fit_key"]/"receipt.json";r=M.read(p);r["fit_sha256"]="0"*64;write(p,r)
                else:
                    p=next((source/"forecasts"/jobs["complete"]["fit_key"]).glob("*.json"));r=M.read(p);r["native_fit"]="changed";write(p,r)
                with patch.object(M,"verify_runtime_cache",return_value={}):
                    with self.assertRaisesRegex(ValueError,"current fit receipt|Forecast/native identity"):M.migrate(*args,apply=True)
                self.assertFalse(args[1].exists())

    def test_existing_or_concurrently_created_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            args,_,_,_=fixture(folder);args[1].mkdir();sentinel=args[1]/"user.txt";sentinel.write_text("keep")
            with self.assertRaisesRegex(ValueError,"new distinct"):M.migrate(*args,apply=True)
            self.assertEqual(sentinel.read_text(),"keep")
        with tempfile.TemporaryDirectory() as folder:
            a=Path(folder)/"staging";b=Path(folder)/"existing";a.mkdir();b.mkdir()
            (a/"new").write_text("new")
            with self.assertRaises(FileExistsError):M.publish_new(a,b)
            self.assertTrue(a.exists());self.assertEqual(list(b.iterdir()),[])

    def test_explicit_missing_source_cache_relocation_rebinds_only_location_and_job_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            original_args,jobs,old,frozen=fixture(folder)
            original_cache=original_args[3];original_cache.rmdir()
            restored_cache=Path(folder)/"restored-cache";restored_cache.mkdir()
            args=(*original_args[:3],restored_cache,*original_args[4:])
            source,destination=args[:2]
            before={p.relative_to(source).as_posix():M.N.digest(p)
                for p in source.rglob("*") if p.is_file()}
            # A changed location is never inferred from --cache alone.
            with self.assertRaisesRegex(ValueError,"job/source/cache/candidate"):
                M.migrate(*args)
            with patch.object(M,"verify_runtime_cache",return_value={"fixture":True}) as verify:
                result=M.migrate(*args,source_cache_path=original_cache,apply=True)
            verify.assert_called_once_with(old,restored_cache)
            self.assertTrue(result["cache_path_relocated"])
            self.assertTrue(result["cache_relocation"]["explicit_source_path"])
            self.assertEqual(result["cache_relocation"]["source_job_cache"],str(original_cache))
            self.assertEqual(result["cache_relocation"]["destination_cache"],str(restored_cache))
            self.assertEqual(result["cache_relocation"]["expected_cache_identity"],old["cache_identity"])
            receipt=M.read(destination/"migration/receipt.json")
            for original in jobs.values():
                new_id=receipt["job_id_mapping"][original["job_id"]]
                adopted=M.read(destination/"jobs"/(new_id+".json"))
                self.assertEqual(adopted["cache"],str(restored_cache))
                excluded={"cache","output","experiment_identity","job_id"}
                self.assertEqual({k:v for k,v in original.items() if k not in excluded},
                    {k:v for k,v in adopted.items() if k not in excluded})
            adopted_freeze=M.read(destination/"freeze.json")
            self.assertEqual({k:v for k,v in frozen.items() if k not in {"identity","experiment_identity"}},
                {k:v for k,v in adopted_freeze.items() if k not in {"identity","experiment_identity"}})
            self.assertEqual(result["carried_worker_seconds"],191.500000000001)
            for relative,digest in before.items():
                self.assertEqual(M.N.digest(source/relative),digest)
                if relative.startswith(("attempts/","forecasts/")) or (
                        relative.startswith("fits/") and Path(relative).name in M.N.CANONICAL):
                    self.assertEqual(M.N.digest(destination/relative),digest)

    def test_cache_relocation_rejects_wrong_relative_traversal_and_symlink_source_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            original_args,_,_,_=fixture(folder)
            restored=Path(folder)/"restored-cache";restored.mkdir()
            alias=Path(folder)/"linked-cache";alias.symlink_to(restored,target_is_directory=True)
            args=(*original_args[:3],restored,*original_args[4:])
            rejected=[(Path(folder)/"wrong-recorded-cache","job/source/cache/candidate"),
                (Path("relative-cache"),"absolute path"),
                (Path(folder)/"missing"/".."/"cache","parent traversal"),
                (alias,"Symlink")]
            for recorded,message in rejected:
                with self.subTest(recorded=recorded):
                    with self.assertRaisesRegex(ValueError,message):
                        M.migrate(*args,source_cache_path=recorded)
                    self.assertFalse(args[1].exists())
            linked_args=(*args[:3],alias,*args[4:])
            with self.assertRaisesRegex(ValueError,"Symlink"):
                M.migrate(*linked_args,source_cache_path=original_args[3])

    def test_current_cache_and_runtime_must_match_the_original_content_identities(self):
        with tempfile.TemporaryDirectory() as folder:
            cache=Path(folder)/"restored-cache";cache.mkdir()
            expected_source={"fixture":"unchanged-runtime"}
            audit={"passed":True,"fixture":"same-cache-content"}
            old={"source":expected_source,"cache_identity":M.N.identity(audit)}
            current=[copy.deepcopy(expected_source)];checked=[copy.deepcopy(audit)];calls=[]
            def audit_cache(path):
                calls.append(path)
                return checked[0]
            modules={"bayes_stesn.applications.ks_improvement_runner":SimpleNamespace(source_identity=lambda:current[0]),
                "bayes_stesn.applications.ks_data":SimpleNamespace(audit_cache=audit_cache)}
            environment={name:"1" for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS")}
            with patch.dict("sys.modules",modules),patch.dict(M.os.environ,environment):
                result=M.verify_runtime_cache(old,cache)
                self.assertEqual(result["cache_identity"],old["cache_identity"])
                self.assertEqual(calls,[cache])
                current[0]={"fixture":"changed-runtime"}
                with self.assertRaisesRegex(ValueError,"source, packages or native runtime"):
                    M.verify_runtime_cache(old,cache)
                self.assertEqual(calls,[cache])
                current[0]=copy.deepcopy(expected_source)
                for replacement in [{"passed":False},{"passed":True,"fixture":"different-cache-content"}]:
                    checked[0]=replacement
                    with self.assertRaisesRegex(ValueError,"cache identity changed"):
                        M.verify_runtime_cache(old,cache)

    def test_reviewed_opaque_sidecar_is_archived_and_never_adopted_as_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            args,jobs,_,_=fixture(folder)
            sidecar,manifest=opaque_sidecar_fixture(args,jobs["complete"])
            before=sidecar.read_bytes()
            with self.assertRaisesRegex(ValueError,"Unknown fit sidecar"):
                M.migrate(*args)
            with patch.object(M,"verify_runtime_cache",side_effect=AssertionError("runtime read")),patch.object(M.N,"validate_fit",side_effect=AssertionError("payload read")):
                planned=M.migrate(*args,fit_sidecar_manifest=manifest)
            self.assertEqual(planned["reviewed_fit_sidecar_count"],1)
            self.assertEqual(planned["reviewed_fit_sidecar_bytes"],len(before))
            self.assertEqual(planned["binary_payload_reads"],0)
            self.assertFalse(args[1].exists())
            with patch.object(M,"verify_runtime_cache",return_value={"fixture":True}):
                result=M.migrate(*args,fit_sidecar_manifest=manifest,apply=True)
            reviewed=result["reviewed_fit_sidecars"]
            self.assertFalse(reviewed["canonical_checkpoint_adoption"])
            archive=args[1]/reviewed["files"][0]["archive_path"]
            self.assertEqual(archive.read_bytes(),before)
            self.assertEqual(sidecar.read_bytes(),before)
            self.assertFalse((args[1]/sidecar.relative_to(args[0])).exists())
            self.assertEqual(M.N.digest(args[1]/"migration/reviewed_fit_sidecars.json"),M.N.digest(manifest))
            self.assertEqual(result["carried_worker_seconds"],191.500000000001)

    def test_unlisted_or_incompatibly_reviewed_sidecars_still_stop_migration(self):
        for change in ("unlisted","hash","experiment","canonical","unsafe"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as folder:
                args,jobs,_,_=fixture(folder)
                sidecar,manifest_path=opaque_sidecar_fixture(args,jobs["complete"])
                manifest=M.read(manifest_path)
                if change=="unlisted":sidecar.with_name(".fit.npz.Z9y8X7").write_bytes(b"unreviewed")
                elif change=="hash":manifest["files"][0]["sha256"]="0"*64
                elif change=="experiment":manifest["source_experiment_identity"]="wrong-experiment"
                elif change=="canonical":manifest["files"][0]["path"]=sidecar.with_name("fit.npz").relative_to(args[0]).as_posix()
                else:manifest["files"][0]["path"]="fits/../../outside"
                manifest["identity"]=M.N.identity({k:v for k,v in manifest.items() if k!="identity"})
                write(manifest_path,manifest)
                with self.assertRaisesRegex(ValueError,"Unknown fit sidecar|sidecar|Sidecar"):
                    M.migrate(*args,fit_sidecar_manifest=manifest_path)
                self.assertFalse(args[1].exists())

    def test_reviewed_opaque_bytes_must_match_the_expected_hash_before_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            args,jobs,_,_=fixture(folder)
            sidecar,manifest=opaque_sidecar_fixture(args,jobs["complete"])
            changed=b"X"*sidecar.stat().st_size;sidecar.write_bytes(changed)
            with patch.object(M,"verify_runtime_cache",return_value={"fixture":True}):
                with self.assertRaisesRegex(ValueError,"byte count/hash/fingerprint mismatch"):
                    M.migrate(*args,fit_sidecar_manifest=manifest,apply=True)
            self.assertFalse(args[1].exists())
            self.assertEqual(sidecar.read_bytes(),changed)


if __name__=="__main__":unittest.main()
