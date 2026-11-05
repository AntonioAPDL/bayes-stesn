#!/usr/bin/env python3
"""Audit a closed KS improvement campaign using immutable metadata only.

This independent reporting tool never loads a model, posterior, dataset, or
forecast array and never decodes the nested forecast metrics. It checks receipt
identities and recorded diagnostic arithmetic, not a new posterior diagnostic
calculation or a fresh binary-checkpoint integrity check. Execute only after an
actually observed coordinator completion; existing scheduler/writer locks are
held throughout the snapshot. A new output file is required.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

import package_ks_improvement_fits as N


SCHEMA = "ks-improvement-final-evidence-audit/1"
TERMINAL = {"complete", "failed", "inference-unqualified"}
PROTOCOLS = {"A": "interleaved", "B": "all_observed", "C": "contiguous"}
SAFE_FORECAST = {"identity", "fit_key", "native_fit", "track", "candidate_id",
    "trajectory", "seed", "first_target", "history", "paths", "qualified",
    "forecast_seed", "forecast_namespace", "partition", "stage_first_scored",
    "state", "seconds"}
SCALAR_LINE = re.compile(r'^  "([^"\\]+)": (.*?)(?:,)?$')


def date(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("A timezone-qualified evidence timestamp is required")
    return result


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def scalar_forecast(path):
    """Read top-level scalar metadata without JSON-decoding nested metrics."""
    path = N.safe_path(path)
    before = N.fingerprint(path.stat())
    if before[2] > 4*N.MIB:
        raise ValueError("Unexpectedly large forecast metadata")
    payload = path.read_bytes()
    if N.fingerprint(path.stat()) != before:
        raise RuntimeError("Forecast receipt changed during metadata read")
    result = {}
    for line in payload.decode().splitlines():
        match = SCALAR_LINE.fullmatch(line)
        if match is not None and match[1] in SAFE_FORECAST:
            if match[1] in result:
                raise ValueError("Duplicate forecast metadata field")
            value = json.loads(match[2])
            if isinstance(value, (dict, list)):
                raise ValueError("Forecast identity fields must be scalar")
            result[match[1]] = value
    if not SAFE_FORECAST <= result.keys():
        raise ValueError("Missing required top-level forecast fields")
    return result, hashlib.sha256(payload).hexdigest()


def diagnostic_block(block, policy, names, chains, draws):
    """Independently recompute the gate from the retained scalar summaries."""
    if (block["chains"] != chains or block["draws"] != draws
            or set(block["parameters"]) != set(names)):
        raise ValueError("Diagnostic dimensions or complete parameter names disagree")
    enough = chains >= policy["min_chains"] and draws >= 8
    failures, rhats, bulks, tails = [], [], [], []
    for name, row in block["parameters"].items():
        passed = (enough and finite(row["rhat"]) and row["rhat"] < policy["rhat_limit"]
            and finite(row["ess_bulk"]) and row["ess_bulk"] >= policy["min_ess"]
            and finite(row["ess_tail"]) and row["ess_tail"] >= policy["min_ess"]
            and finite(row["mcse_mean"]))
        if row["passed"] is not passed:
            raise ValueError("Per-parameter recorded qualification disagrees with arithmetic")
        if not passed:
            failures.append(name)
        if finite(row["rhat"]): rhats.append(row["rhat"])
        if finite(row["ess_bulk"]): bulks.append(row["ess_bulk"])
        if finite(row["ess_tail"]): tails.append(row["ess_tail"])
    passed = not failures
    if block["passed"] is not passed:
        raise ValueError("Block qualification disagrees with per-parameter arithmetic")
    return {"passed": passed, "chains": chains, "draws_per_chain": draws,
        "parameters": len(names), "failed_parameter_count": len(failures),
        "failed_parameter_names": failures,
        "maximum_finite_rhat": max(rhats, default=None),
        "minimum_finite_bulk_ess": min(bulks, default=None),
        "minimum_finite_tail_ess": min(tails, default=None)}


def fit_metadata(receipt, job, experiment):
    cfg = N.expected_config(job)
    for field, expected in (("fit_key", job["fit_key"]), ("config", cfg),
            ("track", job["track"]), ("trajectory", job["trajectory"]),
            ("seed", job["seed"]), ("candidate_id", job["candidate"]["id"])):
        if receipt[field] != expected:
            raise ValueError(f"Fit receipt disagrees with bound job: {field}")
    d = receipt["diagnostics"]
    # Native identity preserves tuple/list types. JSON receipts erase that
    # distinction; reconstruct only the deterministic configuration, never a
    # fitted model or feature array, before checking this typed identity.
    from bayes_stesn.applications.ks_native import resolve_native_config
    from bayes_stesn.io import content_hash
    typed_cfg = resolve_native_config("bayes_stesn", job["candidate"]["config"], job["seed"])
    if json.loads(json.dumps(typed_cfg)) != cfg:
        raise ValueError("Effective configuration resolution changed")
    native_id = content_hash({"schema": "ks-native-fit-1", "model_id": "bayes_stesn",
        "input_identity": d["input_identity"], "config": typed_cfg, "seed": job["seed"],
        "computation": experiment["source"]["native"]})
    if receipt["native_fit_identity"] != native_id or d["fit_identity"] != native_id:
        raise ValueError("Recorded native identity does not bind its source/config/input")
    train = receipt["training_identity"]
    transform = train["transform"]
    if (train["identity_sha256"] != N.identity({k:v for k,v in train.items() if k != "identity_sha256"})
            or train["trajectory"] != job["trajectory"] or train["phase"] != "core"
            or train["spatial_protocol"] != PROTOCOLS[job["track"]]
            or train["temporal_protocol"] != "chronological"
            or train["noise_fraction"] != 0 or train["input_mask"] != "none"
            or transform["fit_time_bounds"] != [0, 6000]
            or len(transform["fit_site_indices"]) != {"A":39, "B":64, "C":40}[job["track"]]
            or transform["training_value_count"] != 6000*len(transform["fit_site_indices"])):
        raise ValueError("Training-only transform/protocol identity disagreement")
    spatial, features = cfg["spatial_mean"], cfg["features"]
    p = len(spatial["centers"])+3 if spatial.get("centers") is not None else spatial["side"]**2+3
    width = 1+len(features["lags"])*features["modes"]+sum(features["units"])+features["squares"]+features["products"]
    if (d["spatial_basis"] != p or d["temporal_features"] != width
            or d["readout_coefficients"] != p*width or d["supplied_training_rows"] != 6000
            or d["conditioning_rows"] != cfg["conditioning"]
            or d["likelihood_rows"] != 6000-cfg["conditioning"]):
        raise ValueError("Reported native dimensions/conditioning disagree with effective configuration")
    sampler, policy, gates = cfg["sampler"], cfg["diagnostics"], d["gates"]
    if len(gates) not in (1, 2) or d["continuations"] != len(gates)-1:
        raise ValueError("Unrecognized bounded inference history")
    functional_names = {"log_sigma2_delivered_scale", "readout_first_site", "readout_middle_site",
        "readout_last_site", "readout_spatial_average", "one_step_predictive_mean_first_site",
        "one_step_predictive_mean_middle_site", "one_step_predictive_mean_last_site",
        "one_step_predictive_mean_spatial_average", "log_one_step_marginal_predictive_variance"}
    results = []
    for index, gate in enumerate(gates):
        draws = sampler["draws"] + index*cfg["continuation_draws"]
        count = min(policy["functional_draws"], draws)
        monitor = gate["monitoring"]
        if (monitor["all_readout_coefficients"] is not True or monitor["coefficient_count"] != p*width
                or monitor["draws_per_chain"] != count
                or monitor["coefficient_storage_bytes"] != 8*sampler["chains"]*count*p*width
                or monitor["max_coefficient_bytes"] != policy["max_coefficient_bytes"]
                or monitor["coefficient_storage_bytes"] > policy["max_coefficient_bytes"]
                or len(monitor["retained_indices"]) != count
                or monitor["retained_indices"][0] != 0 or monitor["retained_indices"][-1] != draws-1
                or monitor["retained_indices"] != sorted(set(monitor["retained_indices"]))):
            raise ValueError("All-coefficient monitoring scope disagrees with effective gate")
        blocks = {
            "hyperparameters": diagnostic_block(gate["hyperparameters"], policy,
                {"prior.log_precision", "log_q", "log_ell"}, sampler["chains"], draws),
            "functionals": diagnostic_block(gate["functionals"], policy,
                functional_names, sampler["chains"], count),
            "coefficients": diagnostic_block(gate["coefficients"], policy,
                {f"C[{i},{j}]" for j in range(width) for i in range(p)}, sampler["chains"], count)}
        passed = all(b["passed"] for b in blocks.values())
        if gate["passed"] is not passed or (index and results[-1]["passed"]):
            raise ValueError("Inference gate/continuation semantics disagree")
        results.append({"passed": passed, "blocks": blocks})
    if d["qualified"] is not results[-1]["passed"]:
        raise ValueError("Final recorded qualification disagrees with all monitored blocks")
    if (len(gates)==1 and not d["qualified"] and cfg["continuation_draws"]>0) or (len(gates)==2 and cfg["continuation_draws"]==0):
        raise ValueError("Completed inference does not follow its bounded continuation policy")
    return {"qualified": d["qualified"], "native_fit_identity": native_id,
        "fit_sha256_recorded_not_rehashed": receipt["fit_sha256"], "fit_bytes": receipt["fit_bytes"],
        "p": p, "d": width, "readout_coefficients": p*width, "continuations": len(gates)-1,
        "final_gate": results[-1], "fit_seconds": receipt["fit_seconds"]}


def frozen_cases(experiment, freeze):
    cfg = experiment["config"]
    for track in cfg["tracks"]:
        row = freeze["selected"][track]
        roles = {"main": row["main"]["id"], "anchor": row["anchor_id"]}
        if row.get("sensitivity"):
            roles["prior_sensitivity"] = row["sensitivity"]["id"]
        for role, cid in roles.items():
            for trajectory in freeze["confirmation_trajectories"]:
                yield {"cohort":"confirmation", "stage":"confirm", "track":track,
                    "role":role, "candidate_id":cid, "trajectory":trajectory,
                    "seed":cfg["model_seed"], "origins":freeze["confirmation_origins"], "paths":512}
        for role, cid in roles.items():
            if role == "prior_sensitivity": continue
            for seed in [cfg["model_seed"], *cfg["sensitivity"]["seeds"]]:
                for trajectory in cfg["sensitivity"]["trajectories"]:
                    yield {"cohort":"seed_baseline" if seed == cfg["model_seed"] else "seed_sensitivity",
                        "stage":"promote" if seed == cfg["model_seed"] else "sensitivity",
                        "track":track, "role":role, "candidate_id":cid, "trajectory":trajectory,
                        "seed":seed, "origins":cfg["sensitivity"]["origins"], "paths":512}


def attempt_accounting(attempts, budget):
    stage = defaultdict(list)
    for name, row in sorted(attempts.items()):
        seconds = row["elapsed_seconds"]
        if not finite(seconds) or seconds < 0:
            raise ValueError("Invalid immutable attempt charge")
        stage[row["stage"]].append(seconds)
    preparation = budget.get("preparation_worker_seconds", 0.)
    all_seconds = math.fsum(r["elapsed_seconds"] for _,r in sorted(attempts.items()))
    search = math.fsum(math.fsum(stage[s]) for s in ("preflight","screen","promote","precision"))
    return {"attempt_files":len(attempts), "attempt_elapsed_seconds":all_seconds,
        "preparation_worker_seconds":preparation, "total_charged_worker_seconds":all_seconds+preparation,
        "search_charged_worker_seconds":search+preparation,
        "attempt_counts_by_stage":dict(Counter(r["stage"] for r in attempts.values())),
        "attempt_elapsed_seconds_by_stage":{s:math.fsum(v) for s,v in sorted(stage.items())},
        "original_campaign_started_utc":budget["campaign_started_utc"],
        "total_worker_seconds_cap":budget["total_worker_seconds_cap"],
        "search_worker_seconds_cap":budget["search_worker_seconds_cap"],
        "wall_seconds_cap":budget["wall_seconds_cap"],
        "interpretation":"All immutable attempt elapsed times, including historical job IDs and interrupted attempts, plus preparation once. Worker wall time is not CPU time; admission limits are checked before groups, not hard-kill deadlines."}


def closed_proof(path, session_id, root, attempts, terminals):
    receipt = N.read_json(path)
    rows = [r for r in receipt.get("sessions",[]) if r.get("session_id") == session_id
        and Path(r.get("root", "")).absolute() == root and r.get("observed_completed") is True
        and isinstance(r.get("exit_code"),int) and not isinstance(r["exit_code"],bool)]
    if receipt.get("source") != "actual_unified_exec_write_stdin_completion_results" or len(rows) != 1:
        raise ValueError("Require actual observed completion of the selected coordinator session")
    latest = max([date(r["finished_utc"]) for r in attempts.values()]
        + [date(r["finished_utc"]) for r in terminals.values()])
    if date(receipt["observed_utc"]) < latest:
        raise ValueError("Closure observation predates retained work")
    return {"session_id":session_id, "exit_code":rows[0]["exit_code"],
        "receipt_sha256":N.digest(path), "observed_utc":receipt["observed_utc"],
        "latest_worker_finish_utc":latest.isoformat()}


def audit(root, closure, session_id, *, acceptance=None):
    root = N.safe_path(root, directory=True)
    # A real observed closure is required before even attempting the locks.
    raw_closure = N.read_json(closure)
    if not any(r.get("session_id") == session_id and r.get("observed_completed") is True
            and Path(r.get("root", "")).absolute() == root for r in raw_closure.get("sessions",[])):
        raise ValueError("Selected coordinator is not recorded closed")
    with ExitStack() as stack:
        locks = N.acquire_locks(root, stack)
        exp, grouped_jobs, freeze = N.current_records(root)
        if freeze is None or freeze.get("test_access") is not False:
            raise ValueError("Require a frozen validation-only study")
        from bayes_stesn.applications.ks_improvement_runner import source_identity
        if source_identity() != exp["source"]:
            raise ValueError("Frozen numerical source/runtime has changed")
        jobs = {j["job_id"]:j for rows in grouped_jobs.values() for j in rows}
        terminals = {}
        for path in sorted((root/"terminal").glob("*.json")):
            row = N.read_json(path)
            if (path.stem not in jobs or row["job_id"] != path.stem or row["state"] not in TERMINAL
                    or row["job_identity"] != N.identity({k:v for k,v in jobs[path.stem].items() if k != "job_id"})):
                raise ValueError("Terminal job identity/state mismatch")
            terminals[path.stem] = row
        attempts = {p.name:N.read_json(p) for p in sorted((root/"attempts").glob("*.json"))}
        closed = closed_proof(closure,session_id,root,attempts,terminals)
        history = N.read_json(root/"history_policy.json")
        if history["history"] != freeze["history"]:
            raise ValueError("Freeze/history policy mismatch")
        by_case = defaultdict(list)
        for j in jobs.values():
            if j["output"] != str(root) or j.get("partition") != "validation":
                raise ValueError("Job root/partition mismatch")
            by_case[(j["stage"],j["track"],j["candidate"]["id"],j["trajectory"],j["seed"])].append(j)
        fits, fit_receipt_hashes, forecast_hashes, rows = {}, {}, {}, []
        for case in frozen_cases(exp,freeze):
            matches = by_case[(case["stage"],case["track"],case["candidate_id"],case["trajectory"],case["seed"])]
            matches = [j for j in matches if j["paths"] == case["paths"]
                and j["origins"] == case["origins"] and j["forecast_seed"] == exp["config"]["forecast_seed"]]
            if len(matches) > 1:
                raise ValueError("Duplicate jobs for the frozen case")
            row = {k:v for k,v in case.items() if k != "origins"}
            row.update(scheduled_origins=len(case["origins"]),job_state="unscheduled",
                qualified=None,successful_origins=0,failed_origins=0,blocked_origins=0,pending_origins=len(case["origins"]))
            if not matches:
                rows.append(row); continue
            j = matches[0]; term = terminals.get(j["job_id"])
            if j["history"] != freeze["history"] or j["path_batch"] != exp["config"]["path_batch"]:
                raise ValueError("Final job/history/batch policy mismatch")
            row.update(job_id=j["job_id"],fit_key=j["fit_key"],job_state="pending" if term is None else term["state"])
            folder=root/"fits"/j["fit_key"]; rp=folder/"receipt.json"
            if rp.exists() and j["fit_key"] not in fits:
                rec=N.read_json(rp); verified=fit_metadata(rec,j,exp)
                native=N.safe_path(folder/"fit.npz")
                if native.stat().st_size != rec["fit_bytes"]:
                    raise ValueError("Native checkpoint recorded size mismatch")
                fits[j["fit_key"]]=verified; fit_receipt_hashes[j["fit_key"]]=N.digest(rp)
            fit=fits.get(j["fit_key"])
            if fit:
                row["qualified"]=fit["qualified"]
                if term and term["state"] == "inference-unqualified" and fit["qualified"]:
                    raise ValueError("Terminal inference failure conflicts with fit gate")
                for first in case["origins"]:
                    identity=N.identity({"native_fit":fit["native_fit_identity"],"first":first,
                        "history":j["history"],"horizon":100,"paths":j["paths"],
                        "forecast_seed":j["forecast_seed"],"partition":"validation",
                        "score_source":exp["source"]["application"]["ks_metrics.py"]})
                    path=root/"forecasts"/j["fit_key"]/f'{first:05d}_{j["paths"]}_{identity[:12]}.json'
                    if not path.exists(): continue
                    f,h=scalar_forecast(path)
                    expected={"identity":identity,"fit_key":j["fit_key"],"native_fit":fit["native_fit_identity"],
                        "track":j["track"],"candidate_id":j["candidate"]["id"],"trajectory":j["trajectory"],
                        "seed":j["seed"],"first_target":first,"history":j["history"],"paths":j["paths"],
                        "forecast_seed":j["forecast_seed"],"partition":"validation","forecast_namespace":"main","qualified":True}
                    if not fit["qualified"] or any(f.get(k)!=v for k,v in expected.items()) or f["state"] not in {"complete","failed"}:
                        raise ValueError("Forecast identity/qualification mismatch")
                    forecast_hashes[path.relative_to(root).as_posix()]=h
                    row["successful_origins" if f["state"] == "complete" else "failed_origins"]+=1
                observed=row["successful_origins"]+row["failed_origins"]
                row["pending_origins"]-=observed
                if observed:
                    sync=N.read_json(folder/f'synchronization_h{j["history"]}.json')
                    if sync["rows"][0].get("passed") is not True:
                        raise ValueError("Forecasts lack passing fitted-model synchronization")
            if term:
                if term["state"] == "inference-unqualified" and fit is None:
                    raise ValueError("Inference-unqualified terminal lacks its fitted diagnostic receipt")
                if term["state"] == "complete" and (fit is None or not fit["qualified"] or row["pending_origins"] or row["failed_origins"]):
                    raise ValueError("Complete terminal lacks its full eligible forecast schedule")
                if "completed_origins" in term and (term["completed_origins"] != row["successful_origins"] or term["failed_origins"] != row["failed_origins"]):
                    raise ValueError("Terminal forecast counters disagree with immutable origins")
                if term["state"] in {"failed","inference-unqualified"}:
                    row["blocked_origins"],row["pending_origins"]=row["pending_origins"],0
            rows.append(row)
        aggregate={}
        for cohort in ("confirmation","seed_baseline","seed_sensitivity"):
            selected=[r for r in rows if r["cohort"]==cohort]
            aggregate[cohort]={"expected_cases":len(selected),"terminal_cases":sum(r["job_state"] in TERMINAL for r in selected),
                "states":dict(Counter(r["job_state"] for r in selected)),
                "qualified_cases":sum(r["qualified"] is True for r in selected),
                "inference_unqualified_cases":sum(r["qualified"] is False for r in selected),
                **{k:sum(r[k] for r in selected) for k in ("scheduled_origins","successful_origins","failed_origins","blocked_origins","pending_origins")}}
        complete=all(r["job_state"] in TERMINAL for r in rows) and len(terminals)==len(jobs)
        operations=attempt_accounting(attempts,exp["config"]["budget"])
        operations.update(closure=closed,locks_acquired=len(locks),
            campaign_elapsed_through_latest_worker_seconds=(date(closed["latest_worker_finish_utc"])-date(exp["config"]["budget"]["campaign_started_utc"])).total_seconds(),
            historical_attempt_job_ids_outside_current_jobs=sum(r["job_id"] not in jobs for r in attempts.values()),
            process_peak_rss_interpretation="Receipt process_peak_rss is a per-process maximum; it is not simultaneous campaign RSS. Cgroup memory includes page cache; no OOM conclusion is inferred from this audit.")
        migration=N.read_json(root/"migration/receipt.json")
        if (not migration["passed"] or migration["destination_experiment"]!=exp["identity"]
                or migration["destination_freeze_identity"]!=freeze["identity"]
                or migration["wall_budget_start"]!=exp["config"]["budget"]["campaign_started_utc"]):
            raise ValueError("Applied migration/current campaign identity mismatch")
        migration_hash=N.digest(root/"migration/receipt.json")
        carried_attempts={name:record for name,record in migration["files"].items()
            if re.fullmatch(r"attempts/[^/]+\.json",name)}
        if len(carried_attempts)!=migration["attempt_files"]:
            raise ValueError("Historical attempt manifest census mismatch")
        for name,record in carried_attempts.items():
            path=N.safe_path(root/name)
            if path.stat().st_size!=record["bytes"] or N.digest(path)!=record["sha256"]:
                raise ValueError("A carried immutable attempt changed or disappeared")
        operations["migration_attempts_independently_reverified"]=len(carried_attempts)
        evidence={}
        evidence_root=Path(__file__).resolve().parents[1]/"docs/applications/evidence"
        for name in ("ks_improvement_applied_migration_independent_review.json","ks_improvement_actual_resume_acceptance.json"):
            value=N.read_json(evidence_root/name)
            if value.get("passed") is not True:
                raise ValueError("Prior migration/resume evidence does not pass")
            if name=="ks_improvement_applied_migration_independent_review.json":
                if (value["identities"]["destination_experiment"]!=exp["identity"]
                        or value["identities"]["destination_freeze"]!=freeze["identity"]
                        or value["applied_migration_receipt"]["sha256"]!=migration_hash):
                    raise ValueError("Independent migration review belongs to another study")
            elif (value["destination_experiment_identity"]!=exp["identity"]
                    or value["migration_receipt_sha256"]!=migration_hash
                    or value["immutable_migration_review_sha256"]!=evidence["ks_improvement_applied_migration_independent_review.json"]["sha256"]):
                raise ValueError("Actual resume acceptance lineage disagrees")
            evidence[name]={"sha256":N.digest(evidence_root/name),"passed":True}
        accepted=None
        if acceptance is not None:
            accepted=N.read_json(acceptance)
            if accepted["experiment_identity"]!=exp["identity"] or accepted["freeze_identity"]!=freeze["identity"]:
                raise ValueError("Selected-model acceptance belongs to another frozen study")
            accepted={k:accepted[k] for k in ("passed","complete","expected_cases","passed_cases","failed_cases")}
            accepted["sha256"]=N.digest(acceptance)
        return {"schema":SCHEMA,"created_utc":datetime.now(timezone.utc).isoformat(),
            "structural_checks_passed":True,"complete":complete,"experiment_identity":exp["identity"],
            "freeze_identity":freeze["identity"],"original_freeze_created_utc":freeze["created_utc"],
            "cache_identity_bound_not_rehashed":exp["cache_identity"],"source_runtime_exact_match":True,
            "numerical_source":exp["source"],"config_identity":N.identity(exp["config"]),
            "history":freeze["history"],"cohorts":aggregate,"cases":rows,"inference_fits":fits,
            "fit_receipt_hashes":fit_receipt_hashes,"forecast_receipt_count":len(forecast_hashes),
            "forecast_receipt_inventory_sha256":N.identity(forecast_hashes),
            "all_scheduled_job_states":dict(Counter(terminals[j]["state"] if j in terminals else "pending" for j in jobs)),
            "operations":operations,"migration_receipt_sha256":migration_hash,
            "applied_migration_and_actual_resume_evidence":evidence,
            "archival_auxiliary_scope":"One reviewed unselected legacy fit retains its verified canonical native forecast checkpoint; inconsistent historical chain/phase files are preserved outside active fits and do not provide an exact inference restart. No qualification was changed.",
            "selected_acceptance":accepted,"binary_payloads_opened":0,"forecast_metrics_decoded":0,
            "limits":["Scalar diagnostic arithmetic is independently checked from receipts; posterior diagnostics are not recomputed from chains.",
                "Native existence/size and receipt identity are checked, but current binary hashes and cache contents are not revalidated here; backup validation is separate.",
                "Forecast metadata are bound by receipt hashes; nested forecast metrics and scoring arithmetic are deliberately outside this audit and are checked by reporting tools.",
                "Complete means the frozen jobs are terminal, including reported scientific/inference failures. Eligibility and origin success are separate counts.",
                "This is an exploratory validation expansion after original-test inspection, not a new untouched test experiment."],
            "tool_sha256":N.digest(__file__)}


def publish_new(path,value):
    path=Path(path).absolute()
    if path.exists() or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError("Choose a new non-symlink evidence output")
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary=tempfile.mkstemp(prefix=".audit-",dir=path.parent)
    try:
        with os.fdopen(descriptor,"wb") as stream:
            stream.write(N.encoded(value));stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path)  # no overwrite, including a concurrent publication
    finally:
        os.unlink(temporary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--closed-session-receipt",type=Path)
    parser.add_argument("--closed-session-id",type=int)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--acceptance",type=Path)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps({"schema":SCHEMA,"mode":"plan","reads_current_results":False,
            "binary_payloads_opened":0,"requires":"Observed final coordinator completion, all existing scheduler/writer locks, and a new evidence output."}));return
    if None in (args.closed_session_receipt,args.closed_session_id,args.output):
        parser.error("Execution requires closure receipt/session ID and new output")
    if args.output.absolute()==args.results.absolute() or args.results.absolute() in args.output.absolute().parents:
        parser.error("Write evidence outside the frozen numerical results root")
    result=audit(args.results,args.closed_session_receipt,args.closed_session_id,acceptance=args.acceptance)
    publish_new(args.output,result)
    print(json.dumps({k:result[k] for k in ("structural_checks_passed","complete","cohorts","binary_payloads_opened","forecast_metrics_decoded")}))
    if not result["complete"]: raise SystemExit(2)


if __name__=="__main__":
    main()
