"""Finite, response-independent proposals for the full-field KS extension.

This is a new model experiment, not a replacement of the accepted PCA/ridge
study. All permitted sites at every configured lag drive the reservoir. The
direct lag terms in the readout are separately declared to control the cost of
the Gaussian calculations. No response values enter this registry builder.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from ..io import content_hash
from .ks_runtime import atomic_json, jsonable


SCHEMA = "ks-fullfield-candidates/1"
SITES = {"A": 39, "B": 64, "C": 40}
LEAKS = (.02, .05, .1, .2, .5, 1.)
RADII = (.1, .3, .6, .9, 1.2, 1.5)


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def base_configuration(track):
    """A declared reference, not a claim of a previously optimal setting."""
    if track not in SITES:
        raise ValueError("Track must be A, B or C")
    return {
        "conditioning": 512,
        "target": "state",
        "features": {
            "kind": "raw_esn", "modes": SITES[track], "units": [128],
            "lags": [1], "raw_readout_lags": "latest",
            "recurrent_scaling": "spectral_radius", "spectral_radius": .9,
            "connectivity": .02, "input_scales": [.2],
            "leak_patterns": [[.2]], "score_bound": None,
            "squares": 64, "products": 0, "feature_scale_floor": .04,
            "max_feature_bytes": 1536 * 1024**2,
        },
        "spatial_mean": {
            "kernel": "fourier", "harmonics": {"A": 12, "B": 16, "C": 4}[track],
            "coordinate_terms": False, "centers": None,
        },
        "prior": {"kind": "ridge", "mean_log_precision": math.log(.1),
                  "sd_log_precision": 1.5},
        "covariance": {"kind": "matern", "nu": 1.5,
            "log_ell_mean": math.log(.3 if track == "B" else .18)},
        "sampler": {"chains": 4, "warmup": 1000, "draws": 2000,
            "parameterization": "ridge_total_scale", "checkpoint_every": 100,
            "a0": 2., "b0": .05},
        "diagnostics": {"functional_draws": 512, "min_chains": 4,
            "min_ess": 400, "rhat_limit": 1.01,
            "max_coefficient_bytes": 2 * 1024**3,
            "disk_reserve_bytes": 12781798196},
        "continuation_draws": 2000,
        "forecast_parameter_rng_policy": "fit_shared_bank_v1",
        "max_parameter_cache_bytes": 256 * 1024**2,
        "max_parameter_cache_disk_bytes": 0,
        "max_forecast_working_bytes": 256 * 1024**2,
    }


def active_configuration(config, observed_sites):
    """Canonical active numerical specification; ignore dormant dataclass fields.

    Fitted artifacts still retain the full resolved configuration. This key
    prevents unused aliases from inflating the number of evaluated candidates.
    """
    cfg = jsonable(copy.deepcopy(config))
    f, s, p = cfg["features"], cfg["spatial_mean"], cfg["prior"]
    features = {k: f[k] for k in (
        "kind", "units", "lags", "input_scales", "leak_patterns",
        "score_bound", "squares", "products", "connectivity",
        "feature_scale_floor", "max_feature_bytes", "raw_readout_lags",
        "recurrent_scaling")}
    features["input_sites"] = int(observed_sites)
    features["seed"] = f.get("seed")
    features["spectral_radius" if f["recurrent_scaling"] == "spectral_radius"
             else "recurrent_norm"] = f["spectral_radius" if f["recurrent_scaling"] == "spectral_radius"
                                       else "recurrent_norm"]
    if s["kernel"] == "fourier":
        spatial = {"kernel": "fourier", "harmonics": int(s["harmonics"]),
                   "coordinate_terms": False, "max_columns": s.get("max_columns", 4096)}
    else:
        spatial = {k: s.get(k) for k in
                   ("kernel", "width", "coordinate_terms", "max_columns", "centers")}
        if s.get("centers") is None:
            spatial.update(side=s.get("side"), bounds=s.get("bounds"))
    if p["kind"] == "ridge":
        prior = {k: p[k] for k in ("kind", "mean_log_precision", "sd_log_precision")}
    elif p["kind"] == "rhs":
        prior = {k: p[k] for k in ("kind", "unit", "tau0", "slab_scale", "slab_df",
                 "fixed_slab", "unshrunk_groups", "fixed_variance")}
    else:
        raise ValueError("This declared search supports ridge and regularized horseshoe only")
    covariance = {k: cfg["covariance"][k] for k in (
        "kind", "nu", "scale_mode", "range_prior", "log_q_mean", "log_q_sd",
        "log_ell_mean", "log_ell_sd", "max_dense_sites", "anisotropy",
        "nonstationary", "static_variance", "infer_nu")}
    result = {"conditioning": cfg["conditioning"], "target": cfg["target"],
              "features": features, "spatial_mean": spatial, "prior": prior,
              "covariance": covariance, "sampler": cfg["sampler"],
              "diagnostics": cfg["diagnostics"],
              "continuation_draws": cfg["continuation_draws"],
              "innovations": cfg.get("innovations", "iid"),
              "forecast_parameter_rng_policy": cfg.get("forecast_parameter_rng_policy", "fit_shared_bank_v1"),
              "max_forecast_bytes": cfg.get("max_forecast_bytes", 256 * 1024**2),
              "max_parameter_cache_bytes": cfg.get("max_parameter_cache_bytes", 256 * 1024**2),
              "max_parameter_cache_disk_bytes": cfg.get("max_parameter_cache_disk_bytes", 2 * 1024**3),
              "max_forecast_working_bytes": cfg.get("max_forecast_working_bytes", 256 * 1024**2)}
    # An explicit baseline decoder, when present, is scientifically active.
    for key in ("baseline", "decoder", "baseline_decoder", "target_decoder"):
        if key in cfg:
            result[key] = cfg[key]
    return result


def estimate_cost(config, observed_sites, training_rows=6000):
    """Conservative component accounting; never claim an unmeasured peak RSS."""
    n = _integer(int(observed_sites), "observed_sites", 2)
    t = _integer(int(training_rows), "training_rows", 2)
    f, s = config["features"], config["spatial_mean"]
    if f["kind"] != "raw_esn":
        raise ValueError("The full-field study forbids PCA and reduced input encoders")
    if len(f["units"]) != 1:
        raise ValueError("This finite study declares one reservoir layer")
    units = _integer(f["units"][0], "units")
    lags = list(f["lags"])
    if not lags or lags != sorted(set(lags)) or min(lags) < 1:
        raise ValueError("Lags must be increasing unique positive integers")
    policy = f["raw_readout_lags"]
    if policy not in ("all", "latest", "none") or (policy == "latest" and 1 not in lags):
        raise ValueError("Invalid direct-readout lag policy")
    driver = n * len(lags)
    direct = driver if policy == "all" else n if policy == "latest" else 0
    squares, products = int(f["squares"]), int(f["products"])
    d, state = 1 + direct + units + squares + products, n * max(lags) + units
    if s["kernel"] == "fourier":
        p = 1 + 2 * int(s["harmonics"])
    else:
        p = 1 + 2 * int(s["coordinate_terms"]) + (
            len(s["centers"]) if s.get("centers") is not None else int(s["side"])**2)
    c = int(config["conditioning"])
    if c <= max(lags) or c >= t - 5:
        raise ValueError("Conditioning must initialize lags and retain at least six targets")
    feature_history = 8 * t * (state + 2 * d + n)
    weights = 8 * (units * units + units * driver)
    spectral_workspace = 8 * 10 * units * units
    gram = 8 * d * d
    conditional_workspace = 8 * (8 * d*d + 8*p*d + 8*n*n)
    diagnostics = 8 * int(config["sampler"]["chains"]) * int(
        config["diagnostics"]["functional_draws"]) * p * d
    forecast_buffers = int(config.get("max_parameter_cache_bytes", 256*1024**2)) + int(
        config.get("max_forecast_working_bytes", 256*1024**2))
    return {
        "training_rows": t, "observed_sites": n, "conditioning_rows": c,
        "likelihood_targets": t-c, "reservoir_units": units,
        "reservoir_input_dimension": driver, "direct_readout_input_dimension": direct,
        "state_dimension": state, "temporal_columns": d, "spatial_columns": p,
        "readout_coefficients": p*d,
        "feature_history_bytes": feature_history, "weight_bytes": weights,
        "spectral_initialization_workspace_bytes": spectral_workspace,
        "temporal_gram_bytes": gram,
        "conditional_algebra_workspace_bytes": conditional_workspace,
        "coefficient_diagnostic_storage_bytes": diagnostics,
        "coefficient_diagnostic_residency_assumption": "Entire memmap counted as resident/file-cache memory",
        "forecast_working_and_parameter_cache_bytes": forecast_buffers,
        "estimated_peak_bytes": int(256*1024**2 + feature_history + weights
            + diagnostics + forecast_buffers + max(spectral_workspace, conditional_workspace)),
        "estimated_fit_storage_bytes": int(32*1024**2 + feature_history + weights),
        "dense_tensor_design_if_materialized_bytes": 8*(t-c)*n*p*d,
        "dense_coefficient_covariance_if_materialized_bytes": 8*(p*d)**2,
        "measured_peak_rss": False,
        "limitations": "Static estimates exclude some allocator/library workspaces and receive a separate 1.5 admission multiplier. "
            "All-coefficient diagnostic storage is conservatively included in memory; dense warning quantities must never be allocated.",
    }


def _ring(columns):
    count = columns - 3
    return [[math.cos(2*math.pi*j/count), math.sin(2*math.pi*j/count)] for j in range(count)]


def _proposals(track):
    base = base_configuration(track)
    proposals = []
    def add(label, cfg, group, **metadata):
        proposals.append((label, copy.deepcopy(cfg), group, metadata))
    leaks = LEAKS if track != "C" else (.02, .05, .2, 1.)
    for leak in leaks:
        for radius in RADII:
            c = copy.deepcopy(base)
            c["features"].update(leak_patterns=[[leak]], spectral_radius=radius)
            add(f"dynamics_a{leak:g}_r{radius:g}", c, "joint_leak_radius")
    memory = []
    for length in (2, 4, 8, 16, 32):
        c = copy.deepcopy(base); c["features"]["lags"] = list(range(1, length+1))
        memory.append((f"lag{length}_latest", c))
        if track != "C" or length in (4, 8):
            c = copy.deepcopy(c); c["features"]["raw_readout_lags"] = "all"
            memory.append((f"lag{length}_all", c))
    for lags in ((1, 2, 4, 8, 16, 32), (1, 4, 8, 16, 32))[:1 if track == "C" else 2]:
        c = copy.deepcopy(base); c["features"]["lags"] = list(lags)
        memory.append(("spaced_"+"_".join(map(str, lags)), c))
    for label, c in memory:
        add(label, c, "lag_and_direct_readout")
    capacity = {}
    for units in (256, 512, 1000):
        for target in ("state", "increment"):
            c = copy.deepcopy(base); c["target"] = target
            c["features"].update(units=[units], squares=min(64, units//2))
            label = f"capacity{units}_{target}"
            capacity[label] = c
            add(label, c, "matched_capacity_target")
    c = copy.deepcopy(base); c["target"] = "increment"
    capacity["capacity128_increment"] = c
    add("capacity128_increment", c, "matched_target")
    c = copy.deepcopy(base)
    c["features"].update(squares=128, input_scales=[2.], connectivity=.1)
    add("squares_full_large_input", c, "nonlinear_input_joint")
    joint = {}
    joint_specs = [
        (512, "lag8", tuple(range(1, 9)), .1, .6, .05, .05, "state"),
        (512, "lag32", tuple(range(1, 33)), .5, 1.2, .5, .01, "increment"),
        (512, "spaced", (1, 2, 4, 8, 16, 32), 1., .9, 1., .02, "state"),
        (1000, "lag8", tuple(range(1, 9)), .2, .9, .2, .02, "increment"),
        (1000, "lag32", tuple(range(1, 33)), .05, .3, .05, .01, "state"),
        (1000, "spaced", (1, 2, 4, 8, 16, 32), .5, 1.5, .5, .05, "increment"),
    ]
    if track == "C": joint_specs = [joint_specs[i] for i in (0, 1, 3, 5)]
    for units, name, lags, leak, radius, scale, connectivity, target in joint_specs:
        c = copy.deepcopy(base); c["target"] = target
        c["features"].update(units=[units], lags=list(lags), squares=64,
            leak_patterns=[[leak]], spectral_radius=radius,
            input_scales=[scale], connectivity=connectivity)
        label = f"joint{units}_{name}_{target}"
        joint[label] = c
        add(label, c, "large_capacity_memory_joint")
    radial = {}
    for target in ("state", "increment"):
        c = copy.deepcopy(base); c["target"] = target
        c["spatial_mean"] = {"kernel": "matern32", "coordinate_terms": True,
            "centers": _ring({"A": 32, "B": 32, "C": 8}[track]),
            "width": {"A": .15, "B": .4, "C": 1.}[track]}
        radial[target] = c
        add(f"radial_{target}", c, "matched_decoder")
    low, high = {"A": (8, 16), "B": (8, 31), "C": (2, 8)}[track]
    harmonic = {}
    for label, harmonics in (("low", low), ("high", high)):
        for target in (("state", "increment") if track != "C" else
                       ("state",) if label == "low" else ("increment",)):
            c = copy.deepcopy(base); c["target"] = target
            c["spatial_mean"]["harmonics"] = harmonics
            harmonic[(label, target)] = c
            add(f"fourier_{label}_{target}", c, "spatial_capacity")
    if track != "C":
        c = copy.deepcopy(base)
        c["features"].update(score_bound=3., input_scales=[.1], connectivity=.1)
        add("bounded_scores", c, "bounded_feature_joint")
        c = copy.deepcopy(base)
        c["features"].update(raw_readout_lags="none", input_scales=[1.], connectivity=.05)
        add("reservoir_only_readout", c, "direct_readout_joint")
    paired = [("base_state", base), ("base_increment", capacity["capacity128_increment"]),
              ("capacity512_increment", capacity["capacity512_increment"]),
              ("capacity1000_increment", capacity["capacity1000_increment"]),
              ("joint1000_lag8_increment", joint["joint1000_lag8_increment"])]
    if track != "C":
        paired += [("capacity256_increment", capacity["capacity256_increment"]),
                   ("lag8_latest", next(c for name, c in memory if name == "lag8_latest")),
                   ("radial_state", radial["state"]),
                   ("fourier_high_state", harmonic[("high", "state")])]
    for label, matched in paired:
        c = copy.deepcopy(matched)
        c["prior"] = {"kind": "rhs", "unit": "group", "tau0": .1,
            "slab_scale": 2., "slab_df": 4., "fixed_slab": False,
            "unshrunk_groups": ["intercept"], "fixed_variance": 10.}
        c["sampler"]["parameterization"] = "legacy"
        c["max_parameter_cache_disk_bytes"] = 512 * 1024**2
        add(f"rhs_{label}", c, "matched_prior", matched_ridge_config=matched)
    return proposals


def build_registry(model_seed=20261105, tracks=("A", "B", "C")):
    from .ks_fullfield_native import resolve_fullfield_config
    model_seed = _integer(model_seed, "model_seed", 0)
    registry = {"schema": SCHEMA, "model_seed": model_seed, "tracks": {},
        "scope": "raw permitted fields; exploratory validation after historical test inspection",
        "global_optimum_claim": False, "response_values_used": False,
        "prior_interpretation": "RHS is an explicit new readout prior; ridge is a matched control. "
            "Architecture changes alter the induced function prior.",
        "raw_input_policy": "Every permitted site at every configured lag enters the reservoir; "
            "raw_readout_lags declares only the additional direct readout terms."}
    for track in tracks:
        candidates, seen, provenance = [], {}, []
        for label, proposed, group, metadata in _proposals(track):
            resolved = resolve_fullfield_config(proposed, seed=model_seed)
            if isinstance(resolved, tuple):
                resolved = resolved[0]
            resolved = jsonable(resolved)
            active = active_configuration(resolved, SITES[track])
            key = content_hash(active)
            if key in seen:
                raise ValueError(f"Duplicate effective proposal {track}/{label} equals {seen[key]}")
            cid = f"ff_{track.lower()}_{label}_{key[:8]}"
            seen[key] = cid
            row = {"id": cid, "label": label, "track": track, "stratum": group,
                "prior_family": resolved["prior"]["kind"], "config": resolved,
                "effective_hash": key, "active_config": active,
                "cost": estimate_cost(resolved, SITES[track])}
            if "matched_ridge_config" in metadata:
                match = resolve_fullfield_config(metadata["matched_ridge_config"], seed=model_seed)
                if isinstance(match, tuple): match = match[0]
                row["matched_ridge_hash"] = content_hash(active_configuration(match, SITES[track]))
            candidates.append(row)
        for c in candidates:
            if "matched_ridge_hash" in c:
                if c["matched_ridge_hash"] not in seen:
                    raise ValueError("A declared RHS comparison lacks its exact ridge counterpart")
                c["matched_ridge_id"] = seen[c["matched_ridge_hash"]]
        expected = 53 if track == "C" else 79
        if len(candidates) != expected:
            raise ValueError(f"Proposal count drift: {track} has {len(candidates)}, expected {expected}")
        reference = next(c for c in candidates if c["label"] == "dynamics_a0.2_r0.9")
        coverage = {"candidate_count": len(candidates),
            "prior_counts": {p: sum(c["prior_family"] == p for c in candidates) for p in ("ridge", "rhs")},
            "units": sorted({c["config"]["features"]["units"][0] for c in candidates}),
            "leaks": sorted({c["config"]["features"]["leak_patterns"][0][0] for c in candidates}),
            "spectral_radii": sorted({c["config"]["features"]["spectral_radius"] for c in candidates}),
            "input_scales": sorted({c["config"]["features"]["input_scales"][0] for c in candidates}),
            "connectivity": sorted({c["config"]["features"]["connectivity"] for c in candidates}),
            "joint_leak_radius_pairs": [[c["config"]["features"]["leak_patterns"][0][0],
                c["config"]["features"]["spectral_radius"]] for c in candidates
                if c["stratum"] == "joint_leak_radius"],
            "lag_schedules": sorted({tuple(c["config"]["features"]["lags"]) for c in candidates}),
            "target_counts": {p: sum(c["config"]["target"] == p for c in candidates)
                              for p in ("state", "increment")},
            "large_capacity_memory_joint": [{"id": c["id"],
                "units": c["config"]["features"]["units"][0],
                "lags": c["config"]["features"]["lags"],
                "leak": c["config"]["features"]["leak_patterns"][0][0],
                "spectral_radius": c["config"]["features"]["spectral_radius"],
                "input_scale": c["config"]["features"]["input_scales"][0],
                "connectivity": c["config"]["features"]["connectivity"],
                "target": c["config"]["target"]} for c in candidates
                if c["stratum"] == "large_capacity_memory_joint"]}
        registry["tracks"][track] = {"reference_id": reference["id"],
            "coverage": coverage, "candidates": candidates}
    registry = jsonable(registry)
    registry["identity"] = content_hash(registry)
    return registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-seed", type=int, default=20261105)
    parser.add_argument("--tracks", nargs="+", default=["A", "B", "C"], choices=list(SITES))
    args = parser.parse_args()
    registry = build_registry(args.model_seed, args.tracks)
    atomic_json(args.output, registry)
    print(json.dumps({"identity": registry["identity"],
        "counts": {t: len(v["candidates"]) for t, v in registry["tracks"].items()}}))


if __name__ == "__main__":
    main()
