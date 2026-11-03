"""Runtime identity uses the BLAS inventory multiset, not import order."""
from copy import deepcopy

import pytest
import threadpoolctl

from bayes_stesn.benchmark import runner


def records():
    common = {"user_api": "blas", "internal_api": "openblas", "num_threads": 1,
              "prefix": "libscipy_openblas", "architecture": "SkylakeX"}
    first = {**common, "version": "0.3.34.106.0"}
    second = {**common, "version": "0.3.31.dev"}
    return [first, second, first.copy()]


def environment_in_policy():
    with runner.single_thread_runtime():
        return runner.benchmark_environment()


def test_runtime_identity_is_invariant_to_library_enumeration_order_and_retains_duplicates(monkeypatch):
    original = records()
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: deepcopy(original))
    parent = environment_in_policy()
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: deepcopy(list(reversed(original))))
    worker = environment_in_policy()
    assert parent == worker
    assert len(parent["threadpools"]) == 3
    assert parent["threadpools"].count(original[0]) == 2


@pytest.mark.parametrize("field,new_value", [
    ("version", "changed-version"), ("architecture", "changed-architecture"),
    ("num_threads", 2), ("internal_api", "changed-api"), ("prefix", "changed-library"),
])
def test_genuine_runtime_field_change_still_invalidates_before_fitting(monkeypatch, field, new_value):
    original = records()
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: deepcopy(original))
    expected = environment_in_policy()
    changed = deepcopy(original)
    changed[1][field] = new_value
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: deepcopy(changed))
    assert environment_in_policy() != expected
    monkeypatch.setattr(runner, "source_fingerprint", lambda: "unchanged-test-source")
    # The environment check precedes state creation, data loading and fitting;
    # this descriptor intentionally contains no data or job-directory fields.
    with pytest.raises(runner.StudyInvalidatedError, match="numerical runtime"):
        runner._execute_job({"source_fingerprint": "unchanged-test-source", "expected_environment": expected})


def test_missing_duplicate_library_still_changes_inventory_identity(monkeypatch):
    original = records()
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: deepcopy(original))
    expected = environment_in_policy()
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: deepcopy(original[:-1]))
    assert environment_in_policy() != expected
