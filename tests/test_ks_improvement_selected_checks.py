"""A resumed acceptance receipt must still bind the current fitted artifact.

These checks use opaque synthetic bytes and mock every scientific operation.
They never load KS data, deserialize a fitted model, fit or forecast.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from bayes_stesn.io import content_hash
from bayes_stesn.applications.ks_runtime import file_hash


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


@pytest.fixture
def acceptance(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location(
        "_ks_improvement_selected_checks_test", TOOLS / "check_ks_improvement_selected.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = {"id": "synthetic_candidate", "config": {"synthetic": True}}
    seed = 20261105
    config = {"tracks": ["A"], "model_seed": seed,
              "registry": {"tracks": {"A": {"candidates": [candidate]}}}}
    source = {"synthetic_source_identity": True}
    experiment = {"identity": "synthetic_experiment", "source": source, "config": config}
    freeze = {"experiment_identity": experiment["identity"], "history": 512,
              "selected": {"A": {"anchor_id": candidate["id"],
                                  "main": {"id": candidate["id"]}}}}
    freeze["identity"] = content_hash(freeze)
    results = tmp_path / "results"
    results.mkdir()
    (results / "scheduler.lock").touch()

    def write(path, value):
        path.write_text(json.dumps(value))

    write(results / "experiment.json", experiment)
    write(results / "freeze.json", freeze)
    key = module.fit_key("A", candidate, 0, seed)
    folder = results / "fits" / key
    folder.mkdir(parents=True)
    fit_path = folder / "fit.npz"
    fit_path.write_bytes(b"opaque synthetic checkpoint; never deserialized")
    receipt = {"fit_key": key, "track": "A", "candidate_id": candidate["id"],
               "trajectory": 0, "seed": seed, "fit_sha256": file_hash(fit_path),
               "native_fit_identity": "synthetic_native_fit", "diagnostics": {"qualified": True}}
    write(folder / "receipt.json", receipt)
    policy = {"first_target": 6000, "history": 512, "horizon": 100,
              "predictive_paths": 16, "seed": 83, "trajectory": 0,
              "selected_roles": ["main", "anchor", "sensitivity"],
              "max_seconds": 180., "partition": "validation"}
    identity = content_hash({"experiment": experiment["identity"], "freeze": freeze["identity"],
                             "policy": policy, "source": file_hash(module.__file__),
                             "reused_check_source": file_hash(TOOLS / "check_ks_selected_invariance.py")})
    case = {"track": "A", "candidate": candidate["id"], "passed": True,
            "fit_sha256": receipt["fit_sha256"], "native_fit_identity": receipt["native_fit_identity"],
            "fit_qualified": True}
    output = tmp_path / "acceptance.json"
    write(output, {"identity": identity, "cases": [case]})

    def no_scientific_work(*args, **kwargs):
        pytest.fail("A retained acceptance case must not rerun scientific operations")

    monkeypatch.setattr(module, "source_identity", lambda: source)
    for name in ("_case", "preprocessing_case", "KSAdapter", "admission"):
        monkeypatch.setattr(module, name, no_scientific_work)
    args = SimpleNamespace(results=results, cache=tmp_path / "no_cache", output=output, execute=True)
    return SimpleNamespace(module=module, args=args, fit_path=fit_path,
                           receipt_path=folder / "receipt.json", receipt=receipt, write=write)


def test_unchanged_retained_case_reuses_without_scientific_work(acceptance):
    acceptance.module.run(acceptance.args)
    result = json.loads(acceptance.args.output.read_text())
    assert result["complete"] and result["passed"]
    assert result["expected_cases"] == result["passed_cases"] == 1
    assert result["new_posterior_fits"] == 0


@pytest.mark.parametrize("change", ["missing", "changed"])
def test_retained_case_rejects_missing_or_changed_payload(acceptance, change):
    before = acceptance.args.output.read_bytes()
    if change == "missing":
        acceptance.fit_path.unlink()
        error, message = FileNotFoundError, None
    else:
        acceptance.fit_path.write_bytes(b"different opaque checkpoint")
        error, message = ValueError, "checksum/identity mismatch"
    with pytest.raises(error, match=message):
        acceptance.module.run(acceptance.args)
    assert acceptance.args.output.read_bytes() == before


def test_retained_case_rejects_changed_native_identity(acceptance):
    before = acceptance.args.output.read_bytes()
    changed = {**acceptance.receipt, "native_fit_identity": "foreign_native_fit"}
    acceptance.write(acceptance.receipt_path, changed)
    with pytest.raises(ValueError, match="no longer matches"):
        acceptance.module.run(acceptance.args)
    assert acceptance.args.output.read_bytes() == before
