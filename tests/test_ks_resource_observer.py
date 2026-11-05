"""Resource-provenance tests; no scientific fits or data access."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "ks_observer_test", Path(__file__).resolve().parents[1] / "tools/watch_ks_resources.py")
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


def record(start, samples, peak, free, jobs=None, complete=True):
    return {"started_utc": start, "last_observed_utc": start + "Z", "status": "scheduler_finished",
            "samples": samples, "combined_worker_rss_peak_bytes": peak,
            "container_current_peak_bytes": peak * 4, "disk_free_min_bytes": free,
            "active_workers": 0, "jobs": jobs or {}, "alerts": [],
            "job_observations_complete": complete}


def job(first, last, peak):
    return {"model": "basis_gru", "track": "A", "stage": "final",
            "first_seen_utc": first, "last_seen_utc": last,
            "peak_rss_bytes": peak, "peak_hwm_bytes": peak + 10}


def test_merge_deduplicates_samples_and_preserves_missing_rows():
    earlier = record("2026-01-01", 5, 40, 1000, {"one": job("a", "b", 25)})
    recovered = record("2026-01-01", 20, 90, 900, complete=False)
    later = record("2026-01-02", 7, 70, 800, {"two": job("c", "d", 60)})
    result = observer.merge_sessions([earlier, recovered, later, copy.deepcopy(later)])
    assert result["samples"] == 27
    assert result["session_count"] == 2
    assert result["combined_worker_rss_peak_bytes"] == 90
    assert result["container_current_peak_bytes"] == 360
    assert result["disk_free_min_bytes"] == 800
    assert result["retained_job_rows"] == 2
    assert result["jobs"]["one"]["peak_rss_bytes"] == 25
    assert result["sessions"][0]["total_observed_jobs"] is None
    assert result["coverage_gaps"][0]["retained_job_rows"] == 1


def test_same_job_union_keeps_observed_extrema_and_checks_identity():
    a = record("2026-01-01", 2, 40, 100, {"one": job("b", "c", 30)})
    b = record("2026-01-02", 4, 45, 90, {"one": job("a", "d", 35)})
    result = observer.merge_sessions([a, b])
    assert result["jobs"]["one"]["first_seen_utc"] == "a"
    assert result["jobs"]["one"]["last_seen_utc"] == "d"
    assert result["jobs"]["one"]["peak_hwm_bytes"] == 45
    b["jobs"]["one"]["track"] = "B"
    with pytest.raises(ValueError, match="identity"):
        observer.merge_sessions([a, b])


def test_legacy_current_and_durable_session_are_not_double_counted(tmp_path):
    a = record("2026-01-01", 3, 40, 100)
    observer.persist_session(tmp_path, a)
    (tmp_path / "external_resources.json").write_text(json.dumps(a))
    result = observer.merge_sessions(observer.read_session_records(tmp_path))
    assert result["samples"] == 3
    observer.atomic_json(tmp_path / "external_resources.json", result)
    assert len(observer.read_session_records(tmp_path)) == 1
    with pytest.raises(ValueError, match="older snapshot"):
        observer.persist_session(tmp_path, {**a, "samples": 2})
