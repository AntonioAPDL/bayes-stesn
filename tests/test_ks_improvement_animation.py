"""Small geometry, provenance and encoding-contract checks; no movie renders."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_tool():
    path = Path(__file__).resolve().parents[1] / "tools/animate_ks_improvement.py"
    spec = importlib.util.spec_from_file_location("ks_improvement_animation_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = load_tool()


def circle(indices):
    angle = np.asarray(indices) * 2 * np.pi / 64
    return np.column_stack((np.cos(angle), np.sin(angle)))


def example(track="A"):
    if track == "A":
        inputs = [i for i in range(64) if i % 5 < 3]
        targets = [i for i in range(64) if i % 5 == 3]
    elif track == "C":
        inputs, targets = list(range(40)), list(range(40, 52))
    else:
        inputs = targets = list(range(64))
    truth = np.sin(np.arange(100)[:, None] * .1 + np.asarray(targets)[None, :] * .2)
    return {"truth": truth, "mean": truth + .1, "lower": truth - .2, "upper": truth + .4,
            "coordinates": circle(targets), "input_coordinates": circle(inputs),
            "first_target": np.array(6000), "training_scale": np.array(1.)}


def panel(arrays, role="anchor"):
    return {"state": "available", "role": role, "candidate_id": role,
            "arrays": arrays, "geometry": tool.validate_example(arrays, "A")}


def test_periodic_coordinates_and_protocol_specific_site_counts():
    for track in ("A", "B", "C"):
        arrays = example(track)
        geometry = tool.validate_example(arrays, track)
        assert len(geometry["x"]) == tool.TRACKS[track]["targets"]
        assert len(geometry["input_x"]) == tool.TRACKS[track]["inputs"]
        assert np.all((geometry["x"] >= 0) & (geometry["x"] < 22))
    x, indices = tool.coordinate_geometry(circle([0, 63]), 2)
    assert x[0] == 0.
    assert x[1] == pytest.approx(63 * 22 / 64)
    assert indices.tolist() == [0, 63]


def test_strict_geometry_rejects_hidden_full_panel_and_overlap():
    arrays = example()
    arrays["input_coordinates"] = circle(range(64))
    with pytest.raises(ValueError, match="site count"):
        tool.validate_example(arrays, "A")
    arrays = example()
    arrays["coordinates"][0] = arrays["input_coordinates"][0]
    with pytest.raises(ValueError, match="overlap"):
        tool.validate_example(arrays, "A")


def test_matched_examples_need_identical_truth_and_training_geometry():
    a, b = example(), example()
    b["mean"] += .4  # A prediction may differ; conditioning/truth may not.
    assert len(tool.matched_geometry([panel(a), panel(b, "main")])) == 2
    b["truth"][0, 0] += .01
    with pytest.raises(ValueError, match="unmatched in truth"):
        tool.matched_geometry([panel(a), panel(b, "main")])


def test_extreme_finite_value_is_retained_with_shared_explicit_symlog():
    a, b = example(), example()
    a["mean"][99, 5] = 1e9
    scale = tool.scale_policy([panel(a), panel(b, "main")])
    assert scale["value_scale"] == "symlog"
    assert scale["error_scale"] == "symlog"
    assert scale["value_limit"] > 1e9
    assert scale["error_limit"] > abs(a["mean"][99, 5] - a["truth"][99, 5])
    assert scale["clipping"] is False
    assert scale["all_finite_values_included"] is True
    with pytest.raises(ValueError, match="empty movie"):
        tool.scale_policy([{"state": "missing"}, {"state": "missing"}])


def test_missing_case_stays_missing_and_bound_bytes_cannot_change(tmp_path):
    root = tmp_path
    name = "examples/A_old_t01_s7_a6000.npz"
    path = root / name
    path.parent.mkdir()
    np.savez_compressed(path, **example())
    manifest = {"results_receipts": {name: {"bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}}
    experiment = {"config": {"model_seed": 7}}
    freeze = {"selected": {"A": {"anchor_id": "old", "main": {"id": "new"}}}}
    panels = tool.load_examples(root, manifest, experiment, freeze, "A")
    assert panels[0]["state"] == "available"
    assert panels[1]["state"] == "missing"
    assert panels[1]["candidate_id"] == "new"
    assert panels[1]["relative_source"] == "examples/A_new_t01_s7_a6000.npz"
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        tool.load_examples(root, manifest, experiment, freeze, "A")


def test_existing_unbound_example_is_not_silently_substituted(tmp_path):
    path = tmp_path / "examples/A_old_t01_s7_a6000.npz"
    path.parent.mkdir()
    np.savez_compressed(path, **example())
    with pytest.raises(ValueError, match="not bound"):
        tool.load_examples(tmp_path, {"results_receipts": {}}, {"config": {"model_seed": 7}},
                           {"selected": {"A": {"anchor_id": "old", "main": {"id": "new"}}}}, "A")


def test_ffprobe_contract_requires_all_100_frames_and_browser_codec():
    correct = {"streams": [{"codec_name": "h264", "pix_fmt": "yuv420p", "width": 1280,
                            "height": 720, "nb_read_frames": "100", "avg_frame_rate": "8/1", "duration": "12.500000"}]}
    assert tool.validate_probe(correct, fps=8)["nb_read_frames"] == "100"
    for key, value in (("nb_read_frames", "99"), ("codec_name", "mpeg4"), ("height", 719), ("duration", "12.1")):
        changed = copy.deepcopy(correct)
        changed["streams"][0][key] = value
        with pytest.raises(ValueError, match="contract"):
            tool.validate_probe(changed, fps=8)
