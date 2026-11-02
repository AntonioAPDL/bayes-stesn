"""Study information boundaries and protocol identities, without costly fits."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "controlled_study.py"
_SPEC = importlib.util.spec_from_file_location("controlled_study_contract_target", _PATH)
study = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(study)


class StudyContractTests(unittest.TestCase):
    def setUp(self):
        self.config = study.load_config()

    def test_more_data_contrast_shares_endpoint_and_preprocessing(self):
        short = study.split_for(self.config, "ridge_short")
        long = study.split_for(self.config, "ridge_long")
        self.assertEqual(short.conditioning, long.conditioning)
        self.assertEqual(short.fitting_stop, long.fitting_stop)
        self.assertEqual(short.validation_blocks, long.validation_blocks)
        self.assertEqual(short.final_blocks, long.final_blocks)
        self.assertEqual(short.fitting_stop - short.fitting_start, self.config["short_fit_times"])
        self.assertEqual(long.fitting_stop - long.fitting_start, self.config["long_fit_times"])
        a = study.candidate_spec(self.config, "ridge_short").to_dict()
        b = study.candidate_spec(self.config, "ridge_long").to_dict()
        a.pop("name"); b.pop("name")
        self.assertEqual(a, b)

    def test_expressiveness_and_prior_contrasts_change_declared_components_only(self):
        compact = study.candidate_spec(self.config, "ridge_long")
        expressive = study.candidate_spec(self.config, "ridge_expressive_long")
        rhs = study.candidate_spec(self.config, "group_rhs_expressive_long")
        grouped = study.candidate_spec(self.config, "group_gaussian_expressive_long")
        self.assertEqual(compact.prior, expressive.prior)
        self.assertEqual(compact.spatial_mean, expressive.spatial_mean)
        self.assertEqual(compact.covariance, expressive.covariance)
        self.assertEqual(expressive.features, rhs.features)
        self.assertEqual(expressive.spatial_mean, rhs.spatial_mean)
        self.assertEqual(expressive.covariance, rhs.covariance)
        self.assertNotEqual(expressive.prior, rhs.prior)
        self.assertEqual(expressive.features, grouped.features)
        self.assertEqual(expressive.spatial_mean, grouped.spatial_mean)
        self.assertEqual(expressive.covariance, grouped.covariance)
        self.assertEqual(grouped.prior.kind, "group")

    def test_selection_uses_validation_and_excludes_failed_chains(self):
        records = {
            "overfit": {"gate_passed": False, "validation": {"crps": 0.001, "rmse": 0.001}, "fit_seconds": 1},
            "good": {"gate_passed": True, "validation": {"crps": 0.12, "rmse": 0.2}, "fit_seconds": 10},
            "other": {"gate_passed": True, "validation": {"crps": 0.13, "rmse": 0.1}, "fit_seconds": 1},
        }
        self.assertEqual(study.choose_candidate(records), "good")
        # Even an accidentally attached final-score field is irrelevant.
        records["other"]["final_score"] = -1e300
        records["good"]["final_score"] = 1e300
        self.assertEqual(study.choose_candidate(records), "good")
        for record in records.values():
            record["gate_passed"] = False
        self.assertIsNone(study.choose_candidate(records))

    def test_freeze_roundtrip_is_immutable_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = study.freeze_protocol(self.config, root, 6, {"runtime_only": True})
            self.assertEqual(study.load_protocol(root), protocol)
            with self.assertRaises(FileExistsError):
                study.freeze_protocol(self.config, root, 6, {})
            path = root / "protocol.json"
            changed = json.loads(path.read_text())
            changed["replications"] += 1
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                study.load_protocol(root)

    def test_final_data_requires_prior_lock_and_predeclared_replication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                study.run_case(self.config, root, "final", "matched_gaussian", 0)
            protocol = study.freeze_protocol(self.config, root, 6, {})
            with self.assertRaises(ValueError):
                study.run_case(self.config, root, "final", "matched_gaussian", 6, protocol["protocol_sha256"])
            self.assertFalse((root / "final").exists())

    def test_cannot_lock_after_final_artifacts_exist(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "final").mkdir()
            (root / "final" / "old_result.json").write_text("{}")
            with self.assertRaises(ValueError):
                study.freeze_protocol(self.config, root, 6, {})

    def test_invalid_temporal_and_resource_profiles_fail(self):
        changed = copy.deepcopy(self.config)
        changed["short_fit_times"] = changed["long_fit_times"] + 1
        with self.assertRaises(ValueError):
            study.validate_config(changed)
        changed = copy.deepcopy(self.config)
        changed["validation_blocks"] = [[1745, 1750]]
        with self.assertRaises(ValueError):
            study.validate_config(changed)
        changed = copy.deepcopy(self.config)
        changed["workers"] = 5
        with self.assertRaises(ValueError):
            study.validate_config(changed)


if __name__ == "__main__":
    unittest.main()
