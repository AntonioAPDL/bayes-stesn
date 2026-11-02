"""Validate completed controlled-study evidence separately from unit tests.

Run from the repository root::

    python tools/verify_study_artifacts.py --study report/controlled_study

The command requires an already completed study and fails explicitly when
its evidence is missing. It does not simulate data, fit models, render a
report, or change any scientific result. It writes an integrity receipt
containing hashes and the numerical cross-checks that actually ran.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayes_stesn.io import load_state


def _study_module():
    path = ROOT / "benchmarks" / "controlled_study.py"
    spec = importlib.util.spec_from_file_location("controlled_study_artifact_validator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_study_artifacts(study_root: Path, receipt_path: Path | None = None) -> dict:
    """Verify committed numeric outputs and write a portable JSON receipt.

    This is evidence validation, not a portable unit test. Missing results
    raise ``FileNotFoundError``; no absent evidence is represented as a
    skipped or passing scientific check.
    """
    if not __debug__:
        raise RuntimeError("Artifact verification requires assertions enabled; do not run Python with -O")
    started = time.perf_counter()
    root = Path(study_root).resolve()
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Completed study evidence is required: {summary_path}")
    study = _study_module()
    protocol = study.load_protocol(root)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert study.source_hashes() == protocol["source_sha256"], "Scientific source changed after protocol lock"
    expected_cases = len(protocol["scenarios"]) * protocol["replications"]
    assert summary["completed_cases"] == expected_cases
    assert summary["protocol_sha256"] == protocol["protocol_sha256"]
    canonical = [root / name for name in ("protocol.json", "summary.json", "pilot_summary.json", "pilot_protocol.json")]
    if (root / "compact_summary.json").is_file():
        canonical.append(root / "compact_summary.json")
    results = {}
    passes = 0
    failures = []
    npz_count = 0
    rmse_checks = 0
    paired_checks = 0
    maximum_error = 0.0
    config = protocol["config"]
    first_final = min(first for first, _ in config["final_blocks"])
    final_stop = max(stop for _, stop in config["final_blocks"])

    for case in summary["case_results"]:
        path = (root / case["path"]).resolve()
        assert path.is_relative_to(root), "Case artifact path escapes the study directory"
        result = json.loads(path.read_text(encoding="utf-8"))
        identity = (result["scenario"], result["replication"])
        assert identity not in results, "Duplicate scenario/replication result"
        assert identity[0] in protocol["scenarios"] and identity[1] in protocol["replication_ids"]
        results[identity] = result
        assert result["protocol_sha256"] == protocol["protocol_sha256"]
        assert result["data_provenance"]["stage"] == "final"
        assert result["selection"]["selected"] == study.choose_candidate(result["models"])
        assert result["selection"]["source"] == "validation only"
        assert datetime.fromisoformat(result["selection"]["locked_at_utc"]) <= datetime.fromisoformat(result["completed_at_utc"])
        directory = path.parent
        canonical.extend((path, directory / "selection.json", directory / "forecast_summaries.npz"))
        selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
        assert selection == result["selection"]

        for name, record in result["models"].items():
            assert (name in result["final_scores"]) == record["gate_passed"], "An ineligible fit was scored or an eligible fit was omitted"
            diagnostic_file = directory / f"{name}.diagnostics.npz"
            canonical.append(diagnostic_file)
            with np.load(diagnostic_file, allow_pickle=False) as diagnostic:
                assert diagnostic["chains"].shape[0] == config["sampler"]["chains"]
                assert np.isfinite(diagnostic["chains"]).all()
                assert np.isfinite(diagnostic["log_density"]).all()
                assert diagnostic["chains"].shape[-1] == len(diagnostic["parameter_names"])
                npz_count += 1
            if record["gate_passed"]:
                passes += 1
                assert record["diagnostics"]["hyperparameters"]["passed"]
                assert record["diagnostics"]["functionals"]["passed"]
            else:
                failures.append({"scenario": identity[0], "replication": identity[1], "model": name,
                                 "diagnostics": record.get("diagnostics"), "error": record.get("error")})

        with np.load(directory / "forecast_summaries.npz", allow_pickle=False) as arrays:
            npz_count += 1
            inputs = arrays["input_indices"]
            all_sites = np.arange(len(arrays["coordinates"]))
            domains = {"input": inputs, "heldout": np.setdiff1d(all_sites, inputs), "all": all_sites}
            assert np.isfinite(arrays["observation_truth"]).all()
            for name, scores in result["final_scores"].items():
                assert len(scores["by_origin"]) == len(config["final_blocks"])
                for block_id, score in enumerate(scores["by_origin"]):
                    first, stop = config["final_blocks"][block_id]
                    predicted = arrays[f"{name}__block{block_id}__mean"]
                    truth = arrays["observation_truth"][first - first_final : stop - first_final]
                    assert predicted.shape == truth.shape
                    assert np.isfinite(predicted).all()
                    for domain, indices in domains.items():
                        direct = float(np.sqrt(np.mean((predicted[:, indices] - truth[:, indices]) ** 2)))
                        error = abs(direct - score[domain]["observation"]["rmse"])
                        assert error < 1e-12, "Saved means/truth disagree with reported RMSE"
                        maximum_error = max(maximum_error, error)
                        rmse_checks += 1
            illustration = directory / "illustration.npz"
            if illustration.exists():
                canonical.append(illustration)
                with np.load(illustration, allow_pickle=False) as image:
                    expected_shape = (config["illustration_paths"], final_stop - first_final, len(all_sites))
                    assert image["observation_paths"].shape == expected_shape
                    assert np.isfinite(image["observation_paths"]).all()
                    assert str(image["selected_model"]) == result["selection"]["selected"]
                    assert_allclose(image["mean"], image["observation_paths"].mean(axis=0), rtol=0, atol=1e-12)
                    assert_array_equal(image["observation_truth"], arrays["observation_truth"])
                    assert int(image["origin"]) + 1 == int(image["times"][0])
                    assert str(image["protocol_sha256"]) == protocol["protocol_sha256"]
                    npz_count += 1
                selected = directory / "selected_fit.npz"
                canonical.append(selected)
                selected_state = load_state(selected, kind="fitted_model")
                selected_name = result["selection"]["selected"]
                assert selected_state["posterior_identity"] == result["models"][selected_name]["posterior_identity"]
                npz_count += 1

    assert len(results) == expected_cases
    for scenario, value in summary["scenario_results"].items():
        for name in config["candidate_names"]:
            failed = [rep for rep in protocol["replication_ids"] if not results[(scenario, rep)]["models"][name]["gate_passed"]]
            assert value["failed_replication_ids"][name] == failed
        for comparison_name, comparison in value["paired_comparisons"].items():
            left, right = comparison_name.split("_minus_")
            for domain, metrics in comparison["domains"].items():
                for metric, record in metrics.items():
                    if record is None:
                        continue
                    expected = []
                    for rep in comparison["paired_replication_ids"]:
                        case = results[(scenario, rep)]
                        actual_left = case["selection"]["selected"] if left == "validation_selected" else left
                        aa = case["final_scores"][actual_left]["replication_average"][domain]["observation"][metric]
                        bb = case["final_scores"][right]["replication_average"][domain]["observation"][metric]
                        expected.append(aa - bb)
                    assert_allclose(record["paired_differences"], expected, rtol=0, atol=1e-12)
                    assert abs(record["difference"] - float(np.mean(expected))) < 1e-12
                    assert record["replications"] == len(expected)
                    assert record["resampling_unit"] == "independent replication"
                    paired_checks += 1

    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in canonical}
    receipt = {
        "status": "passed", "protocol_sha256": protocol["protocol_sha256"],
        "final_wall_seconds": summary["final_wall_seconds"],
        "planned_cases": expected_cases, "completed_cases": len(results),
        "planned_fits": expected_cases * len(config["candidate_names"]),
        "scientific_gate_passes": passes, "scientific_gate_failures": failures,
        "canonical_artifacts_sha256": hashes, "canonical_artifact_count": len(hashes),
        "numeric_npz_archives_validated": npz_count, "cross_artifact_rmse_checks": rmse_checks,
        "maximum_rmse_discrepancy": maximum_error, "paired_comparison_checks": paired_checks,
        "source_hashes_unchanged": True,
        "checks": [
            "frozen protocol identity and all scientific source hashes",
            "complete prespecified scenario/replication cases",
            "validation-only selection and matching saved selection locks",
            "finite diagnostic chains and retained log densities",
            "failed-fit exclusion and scientific gate counts",
            "saved posterior means reproduce input/heldout/full-grid JSON RMSE",
            "illustration path shapes, means, truths, origins, and selected-model identity",
            "selected posterior archives pass embedded content checksums",
            "replication-level paired differences reproduce original case results",
        ],
        "disposable_tmp_npz_excluded": len(list(root.glob("**/tmp*.npz"))),
        "validator": "tools/verify_study_artifacts.py",
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_seconds": time.perf_counter() - started,
    }
    destination = root / "integrity_receipt.json" if receipt_path is None else Path(receipt_path)
    study.write_json(destination, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "report" / "controlled_study")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    result = verify_study_artifacts(args.study, args.receipt)
    print(json.dumps({key: result[key] for key in (
        "status", "canonical_artifact_count", "numeric_npz_archives_validated",
        "cross_artifact_rmse_checks", "paired_comparison_checks", "maximum_rmse_discrepancy",
        "scientific_gate_passes", "audit_seconds",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
