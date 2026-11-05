"""Finite proposal, identity and prior-boundary checks; no model/data fits."""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from bayes_stesn.applications.ks_improvement_candidates import (
    MAIN_STRATUM, LEGACY_STRATUM, PRIOR_STRATUM,
    build_candidate_registry, effective_config_hash, estimate_candidate_cost,
    _write_new_registry,
)


ROOT = Path(__file__).resolve().parents[1]


class ImprovementCandidateContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = json.loads((ROOT / "configs/applications/ks_experiment.json").read_text())
        cls.frozen = json.loads((ROOT / "docs/applications/evidence/ks_frozen_specification.json").read_text())
        cls.registry = build_candidate_registry(cls.original, cls.frozen, tracks=("A", "B", "C"))

    def test_deterministic_detached_unique_registry_and_old_aliases(self):
        before = copy.deepcopy((self.original, self.frozen))
        repeated = build_candidate_registry(self.original, self.frozen, tracks=("A", "B", "C"))
        self.assertEqual(repeated, self.registry)
        self.assertEqual((self.original, self.frozen), before)
        for track, expected in (("A", 40), ("B", 40), ("C", 24)):
            row = repeated["tracks"][track]
            candidates = row["candidates"]
            self.assertEqual(len(candidates), expected)
            self.assertEqual(len({c["effective_hash"] for c in candidates}), expected)
            self.assertEqual(len({c["id"] for c in candidates}), expected)
            source = "A" if track == "C" else track
            old = self.original["candidates"][source]["bayes_stesn"]
            for reference in old:
                actual = next(c for c in candidates if c["id"] == reference["id"])
                self.assertEqual(actual["effective_hash"], effective_config_hash(
                    reference["config"], model_seed=self.original["model_seed"]))
            selected = [c for c in candidates if f"old_selected_{source}" in c["aliases"]]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["id"], row["anchor_id"])
            self.assertEqual(selected[0]["effective_hash"], row["anchor_effective_hash"])
        repeated["tracks"]["A"]["candidates"][0]["config"]["conditioning"] = 999
        self.assertEqual((self.original, self.frozen), before)
        payload = copy.deepcopy(self.registry)
        identity = payload.pop("identity")
        self.assertEqual(identity, hashlib.sha256(json.dumps(payload, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest())

    def test_exact_prior_preservation_and_explicit_sensitivity_strata(self):
        for track, row in self.registry["tracks"].items():
            counts = Counter(c["stratum"] for c in row["candidates"])
            self.assertEqual(counts, {MAIN_STRATUM: 18 if track == "C" else 34,
                                      LEGACY_STRATUM: 2, PRIOR_STRATUM: 4})
            base = next(c["config"] for c in row["candidates"] if c["id"] == row["anchor_id"])
            for candidate in row["candidates"]:
                config = candidate["config"]
                self.assertFalse(candidate["history_synchronization_passed"])
                self.assertFalse(candidate["inference_qualified"])
                self.assertFalse(candidate["validation_selected"])
                self.assertEqual(candidate["main_selection_eligible"], candidate["stratum"] == MAIN_STRATUM)
                if candidate["stratum"] == MAIN_STRATUM:
                    self.assertEqual(config["prior"], base["prior"])
                    self.assertEqual(config["covariance"], base["covariance"])
                    self.assertEqual((config["sampler"]["a0"], config["sampler"]["b0"]),
                                     (base["sampler"]["a0"], base["sampler"]["b0"]))
                if candidate["single_factor_control"]:
                    self.assertEqual(len(candidate["changed_effective_fields"]), 1)
                if candidate["stratum"] == PRIOR_STRATUM:
                    self.assertEqual(candidate["origin"], "prior_sensitivity")
                    self.assertIn(candidate["changed_effective_fields"],
                        [["prior.mean_log_precision"], ["covariance.log_ell_mean"]])
        c = self.registry["tracks"]["C"]
        self.assertEqual(c["anchor_source_track"], "A")
        self.assertTrue(c["independent_validation_selection_required"])
        self.assertFalse(c["reuse_A_fitted_artifacts_permitted"])

    def test_actual_broad_coverage_and_allocation_costs(self):
        for track, row in self.registry["tracks"].items():
            main = [c for c in row["candidates"] if c["stratum"] == MAIN_STRATUM]
            for candidate in row["candidates"]:
                config, cost = candidate["config"], candidate["cost"]
                f = config["features"]
                expected_d = 1 + len(f["lags"]) * f["modes"] + sum(f["units"]) + f["squares"] + f["products"]
                self.assertEqual(cost["temporal_columns"], expected_d)
                self.assertEqual(cost["readout_coefficients"], cost["spatial_columns"] * expected_d)
                self.assertLessEqual(cost["readout_coefficients"], 16000)
                self.assertEqual(cost["likelihood_targets"], 6000-config["conditioning"])
                self.assertLessEqual(cost["all_coefficient_diagnostic_bytes"],
                                     config["diagnostics"]["max_coefficient_bytes"])
                self.assertLessEqual(cost["feature_history_and_recurrent_weights_bytes"],
                                     f["max_feature_bytes"])
                self.assertFalse(cost["measured_peak_rss"])
            if track in ("A", "B"):
                self.assertEqual({c["cost"]["spatial_columns"] for c in main}, {8,12,20,32})
                self.assertEqual({c["cost"]["reservoir_units"] for c in main}, {32,64,128,256,512})
                self.assertEqual({c["config"]["features"]["recurrent_norm"] for c in main}, {.3,.6,.9,.98})
                self.assertEqual({c["config"]["features"]["input_scales"][0] for c in main}, {.05,.2,.5,1.,2.})
                self.assertTrue({.15,.3,.6,1.} <= {c["config"]["spatial_mean"]["width"] for c in main})
                self.assertTrue(any(c["cost"]["pilot_tier"] == "heavy_pilot" for c in main))
            else:
                self.assertLessEqual(max(c["cost"]["reservoir_units"] for c in main), 128)
        self.assertFalse(self.registry["response_data_read"])
        self.assertEqual(self.registry["fits_run"], 0)

    def test_effective_aliases_cannot_manufacture_candidates(self):
        seed = self.original["model_seed"]
        base = copy.deepcopy(self.frozen["selected"]["A"]["bayes_stesn"]["config"])
        changed = copy.deepcopy(base)
        changed["features"].update(input_scales=.5, leak_patterns=None,
            encoder_learning_rate=.123, graph_neighbors=20, seed=123456)
        changed["prior"].update(tau0=.9, slab_scale=99., lambda0=9.)
        changed["covariance"].update(pc_range_threshold=.7, spde_order=3, mesh_resolution=20)
        changed["spatial_mean"].update(side=7, bounds=[[-3,3],[-3,3]])
        changed["sampler"]["seed"] = 654321
        original_hash = effective_config_hash(base, model_seed=seed)
        self.assertEqual(effective_config_hash(changed, model_seed=seed), original_hash)
        changed["features"]["recurrent_norm"] = .6
        self.assertNotEqual(effective_config_hash(changed, model_seed=seed), original_hash)
        self.assertNotEqual(effective_config_hash(base, model_seed=seed+1), original_hash)

    def test_finite_reduced_rosters_optional_sensitivity_and_seeded_joint_proposals(self):
        reduced = build_candidate_registry(self.original, self.frozen,
            max_candidates=32, max_readout_coefficients=12000)
        for row in reduced["tracks"].values():
            self.assertEqual(len(row["candidates"]), 32)
            self.assertLessEqual(max(c["cost"]["readout_coefficients"] for c in row["candidates"]), 12000)
        alternate = build_candidate_registry(self.original, self.frozen, proposal_seed=7,
                                              include_prior_sensitivity=False)
        for track, row in alternate["tracks"].items():
            self.assertEqual(len(row["candidates"]), 40)
            self.assertFalse(row["selection_strata"][PRIOR_STRATUM])
            prior_row = self.registry["tracks"][track]
            def pairs(r, origins):
                return {(c["id"], c["effective_hash"]) for c in r["candidates"] if c["origin"] in origins}
            self.assertEqual(pairs(row, {"legacy_reference", "single_factor_control"}),
                             pairs(prior_row, {"legacy_reference", "single_factor_control"}))
            self.assertNotEqual(pairs(row, {"balanced_joint"}), pairs(prior_row, {"balanced_joint"}))

    def test_native_cap_and_reference_failures_are_explicit(self):
        for options in ({"max_readout_coefficients": 7000}, {"max_candidates": 41},
                        {"max_candidates_c": 25}, {"tracks": ("A", "A")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                build_candidate_registry(self.original, self.frozen, **options)
        frozen = copy.deepcopy(self.frozen)
        frozen["selected"]["A"]["bayes_stesn"]["config"]["features"]["recurrent_norm"] = .6
        with self.assertRaisesRegex(ValueError, "must match"):
            build_candidate_registry(self.original, frozen)
        base = copy.deepcopy(self.registry["tracks"]["A"]["candidates"][0]["config"])
        base["features"]["recurrent_norm"] = 1.
        with self.assertRaisesRegex(ValueError, "strictly in"):
            estimate_candidate_cost(base, observed_sites=39)
        base["features"]["recurrent_norm"] = .9
        base["features"]["leak_patterns"] = [[float("nan")]]
        with self.assertRaisesRegex(ValueError, "Input-scale/leak"):
            estimate_candidate_cost(base, observed_sites=39)

    def test_atomic_publication_refuses_overwrite_and_removes_only_own_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roster.json"
            payload = b'{"schema":"test","complete":true}\n'
            _write_new_registry(path, payload)
            self.assertEqual(path.read_bytes(), payload)
            with self.assertRaises(FileExistsError):
                _write_new_registry(path, b"incompatible")
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
