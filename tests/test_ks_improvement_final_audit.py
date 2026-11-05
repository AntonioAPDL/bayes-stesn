"""Focused metadata-audit checks; no numerical model or production run access."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import audit_ks_improvement_final as A


def block(passed=True, rhat=1.009):
    return {"chains":4,"draws":512,"passed":passed,"parameters":{"x":{
        "rhat":rhat,"ess_bulk":400.,"ess_tail":400.,"mcse_mean":.001,"passed":passed}}}


class FinalEvidenceAuditTests(unittest.TestCase):
    def test_strict_rhat_boundary_and_recorded_flag(self):
        policy={"min_chains":4,"min_ess":400.,"rhat_limit":1.01}
        self.assertTrue(A.diagnostic_block(block(),policy,{"x"},4,512)["passed"])
        self.assertFalse(A.diagnostic_block(block(False,1.01),policy,{"x"},4,512)["passed"])
        with self.assertRaisesRegex(ValueError,"arithmetic"):
            A.diagnostic_block(block(True,1.01),policy,{"x"},4,512)
        with self.assertRaisesRegex(ValueError,"parameter names"):
            A.diagnostic_block(block(),policy,{"x","omitted"},4,512)

    def test_nonfinite_mcse_cannot_pass(self):
        value=block();value["parameters"]["x"]["mcse_mean"]=None
        with self.assertRaisesRegex(ValueError,"arithmetic"):
            A.diagnostic_block(value,{"min_chains":4,"min_ess":400,"rhat_limit":1.01},{"x"},4,512)

    def test_forecast_reader_does_not_decode_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"forecast.json"
            fields={key:("complete" if key=="state" else 1) for key in A.SAFE_FORECAST}
            payload=json.dumps({**fields,"metrics":{"test":123}},sort_keys=True,indent=2)
            # Deliberately invalid nested JSON demonstrates that metrics are
            # neither interpreted nor used for this metadata-only evidence.
            payload=payload.replace('"test": 123','"test": SHOULD_NOT_BE_DECODED')
            path.write_text(payload)
            values,digest=A.scalar_forecast(path)
            self.assertEqual(values,fields);self.assertEqual(len(digest),64)
            self.assertNotIn("metrics",values)

    def test_frozen_schedule_separates_roles_and_baseline_seeds(self):
        config={"tracks":["A","B","C"],"model_seed":1,
            "sensitivity":{"seeds":[2,3],"trajectories":[0,5,10,15],"origins":[6000,6200]}}
        freeze={"selected":{t:{"main":{"id":"main"},"anchor_id":"anchor","sensitivity":{"id":"prior"}}
            for t in config["tracks"]},"confirmation_trajectories":list(range(14)),
            "confirmation_origins":list(range(6000,8000,100))}
        rows=list(A.frozen_cases({"config":config},freeze))
        self.assertEqual(sum(r["cohort"]=="confirmation" for r in rows),126)
        self.assertEqual(sum(r["cohort"]=="seed_baseline" for r in rows),24)
        self.assertEqual(sum(r["cohort"]=="seed_sensitivity" for r in rows),48)
        self.assertFalse(any(r["role"]=="prior_sensitivity" and r["cohort"]!="confirmation" for r in rows))

    def test_all_historical_attempts_charged_once_in_order_independent_sum(self):
        attempts={"z":{"job_id":"legacy_missing","stage":"preflight","elapsed_seconds":.1},
            "a":{"job_id":"current","stage":"confirm","elapsed_seconds":.2},
            "b":{"job_id":"interrupted","stage":"confirm","elapsed_seconds":.3}}
        budget={"preparation_worker_seconds":90.,"campaign_started_utc":"2026-11-05T07:38:04+00:00",
            "total_worker_seconds_cap":54000,"search_worker_seconds_cap":21600,"wall_seconds_cap":18000}
        first=A.attempt_accounting(attempts,budget)
        second=A.attempt_accounting(dict(reversed(list(attempts.items()))),budget)
        self.assertEqual(first,second)
        self.assertEqual(first["total_charged_worker_seconds"],90.6)
        self.assertEqual(first["search_charged_worker_seconds"],90.1)

    def test_real_closure_source_freshness_and_output_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/"closure.json"
            value={"source":"actual_unified_exec_write_stdin_completion_results",
                "observed_utc":"2026-11-05T12:10:00+00:00","sessions":[{"root":str(root),
                "session_id":42,"observed_completed":True,"exit_code":0}]}
            attempts={"a":{"finished_utc":"2026-11-05T12:09:59+00:00"}}
            path.write_text(json.dumps(value))
            self.assertEqual(A.closed_proof(path,42,root,attempts,{})["exit_code"],0)
            value["observed_utc"]="2026-11-05T12:09:00+00:00";path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,"predates"):
                A.closed_proof(path,42,root,attempts,{})
            value["source"]="assumed";path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,"actual observed"):
                A.closed_proof(path,42,root,attempts,{})
            output=root/"new.json";A.publish_new(output,{"passed":True})
            with self.assertRaisesRegex(ValueError,"new non-symlink"):
                A.publish_new(output,{"passed":False})
            self.assertEqual(json.loads(output.read_text()),{"passed":True})


if __name__=="__main__":
    unittest.main()
