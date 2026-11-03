"""Independent references for matched curves, populations and precision plans."""
from dataclasses import replace
import json
import math

import numpy as np
import pytest

from bayes_stesn.benchmark.contracts import Forecast
from bayes_stesn.benchmark.generators import generate
from bayes_stesn.benchmark.metrics import (aggregate_results, canonical_primary_comparisons,
    macro_comparisons, qualification_interval, training_normalization)
from bayes_stesn.benchmark.protocol import BenchmarkProtocol, profile
from bayes_stesn.benchmark.studies import (history_curve_profiles, sensor_curve_profiles,
    run_curve, default_primary_comparisons, freeze_confirmation, verify_confirmation_freeze,
    _paired_macro_plan)
from bayes_stesn.io import content_hash, load_state


def _row(scenario, model, replication, *, difference=0., qualified=True, track="A", scale=2.):
    native = 1.+difference if model == "bayes_esn_ridge" else 1.
    return {"scenario": scenario, "model_id": model, "replication": replication,
            "track": track, "completed": True, "qualified": qualified,
            "fallback": not qualified, "selection_metric": "crps", "training_scale": scale,
            "native_scores": {"crps": native},
            "operational_scores": {"rmse": native if qualified else 4., "mae": native if qualified else 3.}}


def _spec(weights=None, other="separable_stgp", track="A"):
    return {"reference": "bayes_esn_ridge", "other_model": other, "track": track,
            "metric": "crps", "scenario_weights": weights or {"g1_gaussian_ar": 1.}}


def test_explicit_history_endpoint_and_normalization_are_subset_stable():
    base = profile("v03_pilot")
    full = history_curve_profiles(base, (500, 1000, 2000, 4000), common_endpoint=4000)
    subset = history_curve_profiles(base, (1000, 2000), common_endpoint=4000)
    assert [p.to_dict() for p in subset] == [p.to_dict() for p in full[1:3]]
    for p in full:
        assert p.initial_fit == 4000 and p.final_origin == 4120
        assert p.normalization_reference == {"start": 3500, "stop": 4000,
                 "observed_count": 25, "floor": 1e-12, "ddof": 1}
    with pytest.raises(ValueError, match="common_endpoint"):
        history_curve_profiles(base, (500, 1000), common_endpoint=800)
    for values in ((), (500, 500), (500., 1000), (True, 1000)):
        with pytest.raises(ValueError):
            history_curve_profiles(base, values)


def test_normalization_uses_only_common_observed_training_reference():
    base = profile("v03_pilot", scenarios=("g1_gaussian_ar",))
    curves = history_curve_profiles(base, (500, 1000, 2000, 4000), common_endpoint=4000)
    sim = generate("g1", curves[0].n_times, n_side=7, n_observed=25, n_calibration=4, seed=29)
    normalized = [training_normalization(sim, p) for p in curves]
    assert all(pair == normalized[0] for pair in normalized)
    expected = np.std(sim.observation[3500:4000, sim.observed_indices], ddof=1)
    assert normalized[0][0] == expected
    changed = sim.observation.copy()
    changed[4000:] += 1000
    changed[:, sim.test_indices] *= -4
    perturbed = replace(sim, observation=changed, latent=np.zeros_like(sim.latent))
    assert training_normalization(perturbed, curves[0]) == normalized[0]
    legacy = profile("pilot", scenarios=("g1_gaussian_ar",))
    scale, metadata = training_normalization(sim, legacy)
    assert scale == np.std(sim.observation[:400, sim.observed_indices])
    assert metadata["policy"] == "legacy_training_window" and metadata["ddof"] == 0


def test_sensor_curve_shares_field_targets_and_normalization():
    base = profile("v03_pilot", n_side=9, initial_fit=120, scenarios=("g1_gaussian_ar",))
    curves = sensor_curve_profiles(base, (8, 16, 25))
    simulations = [generate("g1", p.n_times, n_side=p.n_side, n_observed=p.n_observed,
        n_calibration=p.n_calibration, seed=17, **p.generator_options["g1_gaussian_ar"]) for p in curves]
    first = simulations[0]
    for sim, p in zip(simulations, curves):
        np.testing.assert_array_equal(sim.observation, first.observation)
        np.testing.assert_array_equal(sim.test_indices, first.test_indices)
        np.testing.assert_array_equal(sim.calibration_indices, first.calibration_indices)
        np.testing.assert_array_equal(sim.observed_indices[:8], first.observed_indices)
        assert p.score_observed_count == 8
        assert training_normalization(sim, p) == training_normalization(first, curves[0])


def test_curve_subset_execution_preserves_full_design(tmp_path, monkeypatch):
    import bayes_stesn.benchmark.studies as studies
    seen = []
    monkeypatch.setattr(studies, "run_benchmark", lambda p, *args, **kwargs: seen.append(p.name) or {"name": p.name})
    base = profile("v03_pilot", scenarios=("g1_gaussian_ar",))
    kwargs = dict(kind="history", values=(500, 1000, 2000, 4000), common_endpoint=4000)
    run_curve(base, tmp_path, execute_values=(1000, 2000), **kwargs)
    original = (tmp_path/"curve_design.json").read_bytes()
    assert seen == ["v03_pilot_history_1000", "v03_pilot_history_2000"]
    run_curve(base, tmp_path, execute_values=(500, 4000), **kwargs)
    assert (tmp_path/"curve_design.json").read_bytes() == original
    value = json.loads(original)
    assert content_hash({k:v for k,v in value.items() if k != "identity"}) == value["identity"]
    with pytest.raises(ValueError, match="Curve design changed"):
        run_curve(base, tmp_path, kind="history", values=(1000, 2000), common_endpoint=4000)


def test_population_counts_do_not_turn_infrastructure_into_method_failure():
    rows = []
    for rep in range(4):
        rows.extend((_row("g1_gaussian_ar", "bayes_esn_ridge", rep, difference=.2, qualified=rep != 1),
                     _row("g1_gaussian_ar", "separable_stgp", rep)))
    # Deliberately include an old fabricated fallback; infrastructure status
    # must override its score presence and keep it outside scientific means.
    for row in rows[-2:]:
        row.update(process_failure=True, qualified=False, fallback=True)
    result = aggregate_results(rows, bootstrap_resamples=20)
    esn = next(row for row in result["table"] if row["model_id"] == "bayes_esn_ridge")
    assert (esn["scheduled"], esn["attempted"], esn["qualified"], esn["fallback"], esn["incomplete"]) == (4, 3, 2, 1, 1)
    assert esn["rmse"] == pytest.approx((1.2+4.+1.2)/3)
    paired = result["paired_comparisons"][0]
    assert paired["rmse_replications"] == [0, 1, 2]
    assert paired["crps_replications"] == [0, 2]
    assert paired["crps_attrition"]["rate"] == 2/3
    assert paired["crps_attrition"]["attempted_pairs"] == 3
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_results(rows+[rows[0]])
    bad = [dict(row) for row in rows]
    bad[1]["training_scale"] = 3.
    with pytest.raises(ValueError, match="same per-dataset"):
        aggregate_results(bad)


def test_macro_mcse_uses_weighted_paired_variances_and_usable_counts():
    values = {"g1_gaussian_ar": np.array([.1, .2, .4, .6, .8]),
              "g2_spectral_transport": np.array([-.2, -.1, 0, .1, .2, .3, .4, .5])}
    rows = []
    for scenario, differences in values.items():
        for rep, difference in enumerate(differences):
            rows.extend((_row(scenario, "bayes_esn_ridge", rep, difference=difference),
                         _row(scenario, "separable_stgp", rep)))
    weights = {"g1_gaussian_ar": .25, "g2_spectral_transport": .75}
    result = aggregate_results(rows, primary_comparisons=[_spec(weights)], bootstrap_resamples=100, seed=23)
    macro = result["macro_comparisons"][0]
    expected_mean = sum(weights[g]*d.mean()/2 for g, d in values.items())
    expected_variance = sum(weights[g]**2*np.var(d/2, ddof=1)/len(d) for g, d in values.items())
    assert macro["difference"] == pytest.approx(expected_mean)
    assert macro["mcse"] == pytest.approx(np.sqrt(expected_variance))
    assert [cell["pairs"] for cell in macro["cells"]] == [5, 8]
    # No complete-case intersection across independent scenario strata.
    assert macro["complete"] and macro["interval_scope"].startswith("pointwise")
    absent = {**_spec(), "scenario_weights": {"g1_gaussian_ar": .5, "g4_existing_fronts": .5}}
    missing = macro_comparisons(result, [absent], bootstrap_resamples=10)[0]
    assert not missing["complete"] and missing["difference"] is None and missing["mcse"] is None
    with pytest.raises(ValueError, match="Coupled"):
        canonical_primary_comparisons([_spec({"g4_existing_fronts": .5, "g5_noise_stress": .5})])


def test_joint_qualification_planning_uses_joint_counts_and_exposes_cap():
    rows = []
    differences = np.array([-.4, -.2, .1, .3, .7])
    for rep in range(10):
        rows.extend((_row("g1_gaussian_ar", "bayes_esn_ridge", rep,
                          difference=float(differences[rep % 5]), qualified=rep < 5),
                     _row("g1_gaussian_ar", "separable_stgp", rep, qualified=rep != 9)))
    summary = aggregate_results(rows, bootstrap_resamples=20)
    draft = profile("v03_main", scenarios=("g1_gaussian_ar",), tracks=("A",))
    plan = _paired_macro_plan({"stage": "development", "identity": "fixture"},
                              summary, draft, .01, [_spec()])
    cell = plan["primary_contrasts"][0]["cells"][0]
    assert cell["joint_qualification"]["rate"] == .5  # not .5*.9
    lower = qualification_interval(5, 10)["interval"][0]
    coefficient = np.var(differences/2, ddof=1)/lower
    assert plan["uncapped_R"] == max(20, math.ceil(coefficient/.01**2))
    assert plan["R"] == 60 and plan["cap_binding"]
    assert cell["expected_usable_pairs_at_R"]["observed_joint_rate"] == 30
    assert plan["primary_contrasts"][0]["variance_sensitivity"][-1]["multiplier"] == 2.
    fewer = aggregate_results(rows[2:], bootstrap_resamples=20)
    with pytest.raises(ValueError, match="At least 5"):
        _paired_macro_plan({"stage": "development", "identity": "fixture"}, fewer, draft, .02, [_spec()])
    unknown = json.loads(json.dumps(summary, default=lambda value: value.tolist()))
    for row in unknown["paired_comparisons"]:
        row.pop("paired_attempted_population")
        row.pop("rmse_population")
    with pytest.raises(ValueError, match="attempted-pair counts"):
        _paired_macro_plan({"stage": "development", "identity": "fixture"}, unknown, draft, .02, [_spec()])


def _freeze_fixture(directory):
    from bayes_stesn.benchmark.runner import (write_json, jsonable, source_fingerprint,
                                             benchmark_environment, single_thread_runtime)
    p = profile("v03_pilot", initial_fit=120, horizon=8, n_side=5, n_observed=10,
                n_calibration=3, scenarios=("g1_gaussian_ar",),
                models=("bayes_esn_ridge", "separable_stgp", "basis_gru"), tracks=("A",))
    with single_thread_runtime():
        runtime = benchmark_environment()
    manifest = jsonable({"stage": "development", "schema": "benchmark-run-4.0",
                "protocol": p.to_dict(), "protocol_identity": p.identity,
                "source_fingerprint": source_fingerprint(), "environment": runtime})
    manifest["identity"] = content_hash(manifest)
    rows = [_row("g1_gaussian_ar", model, rep, difference=(rep-2)*.04)
            for rep in range(5) for model in p.models]
    summary = jsonable(aggregate_results(rows, bootstrap_resamples=20))
    summary.update(source_stable=True, manifest_identity=manifest["identity"], complete=True)
    write_json(directory/"manifest.json", manifest)
    write_json(directory/"summary.json", summary)
    return replace(p, stage="confirmation", replications=20, name="frozen")


def test_macro_confirmation_freeze_recomputes_family_design_and_attrition(tmp_path):
    draft = _freeze_fixture(tmp_path/"pilot")
    result = freeze_confirmation(tmp_path/"pilot", tmp_path/"freeze.json", protocol=draft)
    assert result["schema"] == "benchmark-confirmation-freeze-3.0"
    assert result["planning_mode"] == "paired_macro"
    assert result["primary_comparisons"] == default_primary_comparisons(draft)
    frozen = BenchmarkProtocol(**result["protocol"])
    assert verify_confirmation_freeze(frozen, result)["identity"] == result["identity"]
    with pytest.raises(ValueError, match="share scientific dimensions"):
        freeze_confirmation(tmp_path/"pilot", tmp_path/"changed.json", protocol=replace(draft, horizon=9))
    with pytest.raises(ValueError, match="requires paired_macro"):
        freeze_confirmation(tmp_path/"pilot", tmp_path/"legacy.json", protocol=draft, planning_mode="legacy")
    changed = json.loads(json.dumps(result))
    changed["primary_comparisons"][0]["metric"] = "rmse"
    changed["identity"] = content_hash({k:v for k,v in changed.items() if k != "identity"})
    with pytest.raises(ValueError, match="family differs"):
        verify_confirmation_freeze(frozen, changed)
    summary_path = tmp_path/"pilot/summary.json"
    summary = json.loads(summary_path.read_text());summary["complete"] = False
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="incomplete"):
        freeze_confirmation(tmp_path/"pilot", tmp_path/"incomplete.json", protocol=draft)


def test_forecast_owned_arrays_no_copy_and_public_arrays_remain_defensive(tmp_path):
    from bayes_stesn.prediction import PredictiveSample
    samples = np.arange(8*5*3, dtype=float).reshape(8, 5, 3).copy()
    mean = samples.mean(axis=0)
    coordinates = np.array([[0., 0.], [.5, .5], [1., 1.]])
    public = Forecast(mean, samples, coordinates, ("a", "b", "c"))
    assert not np.shares_memory(public.samples, samples)
    with pytest.raises(ValueError):
        public.samples.setflags(write=True)
    # Simulate the actual producer's frozen whole-allocation view.
    produced = PredictiveSample._from_owned(observation=samples, readout_mean=None,
        conditional_mean=None, latent=None, target_coordinates=coordinates,
        target_site_ids=("a", "b", "c"), parameter_draw_ids=np.zeros((8, 2), int), metadata={})
    owned = Forecast._from_owned(produced.observation.mean(axis=0), produced.observation,
                                 coordinates, ("a", "b", "c"), {"example": True})
    assert np.shares_memory(owned.samples, produced.observation)
    np.testing.assert_array_equal(owned.samples, public.samples)
    for value in (owned.samples, owned.mean):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    owned.save(tmp_path/"forecast.npz")
    restored = Forecast.load(tmp_path/"forecast.npz")
    np.testing.assert_array_equal(restored.samples, owned.samples)
    assert restored.metadata == owned.metadata
    assert load_state(tmp_path/"forecast.npz", kind="benchmark_forecast")["samples"].dtype == float
    with pytest.raises(ValueError, match="whole private allocation"):
        Forecast._from_owned(mean, samples[:4], coordinates, ("a", "b", "c"))
    with pytest.raises(ValueError, match="owned contiguous"):
        Forecast._from_owned(mean, samples.astype(np.float32), coordinates, ("a", "b", "c"))


def test_public_forecast_accepts_noncontiguous_and_fortran_inputs():
    raw = np.arange(5*8*3, dtype=float).reshape(5, 8, 3)
    strided = raw.transpose(1, 0, 2)
    assert not strided.flags.c_contiguous and not strided.flags.f_contiguous
    coordinates = np.array([[0., 0.], [.5, .5], [1., 1.]])
    forecast = Forecast(strided.mean(axis=0), strided, coordinates, ("a", "b", "c"))
    np.testing.assert_array_equal(forecast.samples, strided)
    raw[:] = -1
    assert np.all(forecast.samples >= 0)
    fortran = np.asfortranarray(forecast.samples)
    owned = Forecast._from_owned(forecast.mean.copy(), fortran, coordinates, ("a", "b", "c"))
    assert np.shares_memory(owned.samples, fortran)
    np.testing.assert_array_equal(owned.samples, forecast.samples)
