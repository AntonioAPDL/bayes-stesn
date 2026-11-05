"""Response-independent proposals for a broader KS Bayesian ST-ESN study.

This module builds specifications, not fits or rankings.  It leaves the accepted
native implementation unchanged.  New main candidates retain the selected
track's exact prior hyperparameters.  Original configurations with a different
range prior and explicitly requested prior sensitivities have separate strata.
No candidate is synchronized, inference-qualified or selected by this builder.

The two main tracks receive at most 40 proposals each: the four original
references, one-factor controls, bounded joint proposals and, optionally, four
prior sensitivities.  Track C uses a separate, smaller, spatially focused roster
based on the old A configuration; it requires its own validation selection and
fresh C fits.  The old selected reference is an alias, never a duplicate fit.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping


SCHEMA = "ks-bayesian-improvement-candidates/1"
SITES = {"A": 39, "B": 64, "C": 40}
MAIN_STRATUM = "main_fixed_prior"
LEGACY_STRATUM = "legacy_prior_reference"
PRIOR_STRATUM = "prior_scale_sensitivity"


def _copy(value):
    """JSON types, finite numbers, detached caller-owned dictionaries."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _hash(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _resolve(config, model_seed):
    # Importing the adapter does not fit a model or require the optional Torch
    # runtime.  Its own resolver binds the original named RNG streams/defaults.
    from .ks_native import resolve_native_config

    result = _copy(resolve_native_config("bayes_stesn", _copy(config), model_seed))
    if result["features"]["kind"] != "pca_esn":
        raise ValueError("This roster preserves the existing PCA-ESN encoder")
    if result["covariance"]["range_prior"] != "lognormal":
        raise ValueError("This roster preserves the existing lognormal covariance hyperpriors")
    return result


def _active(config):
    """Canonical fields that affect this supported likelihood/feature route.

    Dataclass dumps also contain inactive encoder, horseshoe, SPDE and PC-prior
    options.  They are preserved in ``config`` for native compatibility, but
    cannot manufacture distinct candidates.  Explicit centers make side/bounds
    inactive.  The full resolved configuration has a separate strict hash.
    """
    f = config["features"]
    feature = {k: f[k] for k in (
        "kind", "modes", "units", "lags", "squares", "products", "seed",
        "max_feature_bytes")}
    for k in ("recurrent_norm", "connectivity", "feature_scale_floor"):
        feature[k] = float(f[k])
    feature["score_bound"] = None if f["score_bound"] is None else float(f["score_bound"])
    feature["input_scales"] = [float(x) for x in f["input_scales"]]
    patterns = f["leak_patterns"]
    if patterns is None:
        patterns = [[.2, .5, .9]] + [[.05, .2, .6]] * (len(f["units"]) - 1)
    feature["leak_patterns"] = [[float(x) for x in row] for row in patterns]
    s = config["spatial_mean"]
    spatial = {k: s[k] for k in ("kernel", "coordinate_terms", "max_columns")}
    spatial["width"] = float(s["width"])
    if s["centers"] is None:
        spatial.update(centers=None, side=s["side"], bounds=s["bounds"])
    else:
        spatial["centers"] = [[float(x) for x in row] for row in s["centers"]]
    p = config["prior"]
    location = p["mean_log_precision"]
    prior = {"kind": p["kind"], "mean_log_precision": float(location)
             if location is not None else math.log(p["lambda0"]),
             "sd_log_precision": float(p["sd_log_precision"])}
    c = config["covariance"]
    covariance = {k: c[k] for k in (
        "kind", "nu", "scale_mode", "range_prior", "log_q_mean", "log_q_sd",
        "log_ell_mean", "log_ell_sd", "max_dense_sites", "anisotropy",
        "nonstationary", "static_variance", "infer_nu")}
    return {"conditioning": config["conditioning"], "features": feature,
            "spatial_mean": spatial, "prior": prior, "covariance": covariance,
            "innovations": config["innovations"], "sampler": config["sampler"],
            "diagnostics": config["diagnostics"],
            "continuation_draws": config["continuation_draws"]}


def effective_config_hash(config: Mapping[str, Any], *, model_seed: int) -> str:
    """Hash resolved, active options; ignored aliases are not new candidates."""
    return _hash(_active(_resolve(config, _integer(model_seed, "model_seed", 0))))


def _prior_signature(config):
    active = _active(config)
    covariance = active["covariance"]
    return {"readout": active["prior"],
            "variance": {"family": "inverse_gamma", "a0": config["sampler"]["a0"],
                         "b0": config["sampler"]["b0"]},
            "covariance_hyperpriors": {k: covariance[k] for k in
                ("range_prior", "log_q_mean", "log_q_sd", "log_ell_mean", "log_ell_sd")}}


def _flatten(value, prefix=""):
    out = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            out.update(_flatten(item, path))
        else:
            out[path] = item
    return out


def _changed(base, candidate):
    left, right = _flatten(_active(base)), _flatten(_active(candidate))
    return sorted(k for k in left.keys() | right.keys() if left.get(k) != right.get(k))


def estimate_candidate_cost(config: Mapping[str, Any], *, observed_sites: int,
                            training_rows: int = 6000,
                            max_readout_coefficients: int = 16000) -> dict:
    """Static float64 allocation components, never a measured peak-RSS claim.

    ``config`` must be resolved.  Dense tensor-design/covariance costs are shown
    only as warnings about forbidden materialization; no arrays are allocated.
    """
    n = _integer(observed_sites, "observed_sites", 2)
    t = _integer(training_rows, "training_rows", 12)
    cap = _integer(max_readout_coefficients, "max_readout_coefficients")
    f, s = config["features"], config["spatial_mean"]
    conditioning = _integer(config["conditioning"], "conditioning", 10)
    modes = _integer(f["modes"], "features.modes")
    units = [_integer(x, "features.units") for x in f["units"]]
    lags = [_integer(x, "features.lags") for x in f["lags"]]
    if not units or not lags or sorted(set(lags)) != lags:
        raise ValueError("Use nonempty reservoirs and increasing distinct positive lags")
    if modes >= min(conditioning, n) or conditioning <= max(lags) or conditioning > t - 6:
        raise ValueError("PCA/conditioning/lag dimensions violate the native data boundary")
    norm = float(f["recurrent_norm"])
    if not 0 < norm < 1:
        raise ValueError("The preserved native operator norm must lie strictly in (0,1)")
    if not 0 < float(f["connectivity"]) <= 1:
        raise ValueError("Connectivity must lie in (0,1]")
    active = _active(config)["features"]
    if (len(active["input_scales"]) != len(units)
            or len(active["leak_patterns"]) != len(units)
            or any(not math.isfinite(x) or x <= 0 for x in active["input_scales"])
            or any(not row or any(not math.isfinite(x) or not 0 < x <= 1 for x in row)
                   for row in active["leak_patterns"])):
        raise ValueError("Input-scale/leak settings do not match the native layers")
    total = sum(units)
    squares = _integer(f["squares"], "features.squares", 0)
    products = _integer(f["products"], "features.products", 0)
    if squares > total or products > total * (total - 1) // 2:
        raise ValueError("Nonlinear feature counts exceed available reservoir coordinates")
    centers = len(s["centers"]) if s["centers"] is not None else s["side"] ** 2
    p = 1 + (2 if s["coordinate_terms"] else 0) + centers
    dimension = 1 + len(lags) * modes + total + squares + products
    state = max(lags) * modes + total
    coefficients = p * dimension
    feature_bytes = 8 * (t * (state + dimension) + sum(x*x for x in units))
    diagnostic_bytes = (8 * config["sampler"]["chains"]
                        * config["diagnostics"]["functional_draws"] * coefficients)
    if coefficients > cap:
        raise ValueError(f"Readout coefficient budget exceeded: {coefficients} > {cap}")
    if p > s["max_columns"] or feature_bytes > f["max_feature_bytes"]:
        raise ValueError("Spatial/temporal feature allocation exceeds an unchanged native gate")
    if diagnostic_bytes > config["diagnostics"]["max_coefficient_bytes"]:
        raise ValueError("All-coefficient diagnostics exceed the unchanged native memory cap")
    return {"observed_sites": n, "training_rows": t, "conditioning_rows": conditioning,
            "likelihood_targets": t - conditioning, "spatial_columns": p,
            "temporal_columns": dimension, "readout_coefficients": coefficients,
            "reservoir_units": total, "state_dimension": state,
            "feature_history_and_recurrent_weights_bytes": feature_bytes,
            "temporal_gram_bytes": 8 * dimension**2,
            "spatial_gram_bytes": 8 * p**2, "response_panel_bytes": 8 * t * n,
            "all_coefficient_diagnostic_bytes": diagnostic_bytes,
            "dense_tensor_design_if_materialized_bytes": 8 * (t-conditioning) * n * coefficients,
            "dense_coefficient_covariance_if_materialized_bytes": 8 * coefficients**2,
            "eigendecomposition_dimension_cubed_proxy": dimension**3 + p**3,
            "pilot_tier": "heavy_pilot" if total >= 256 or coefficients >= 12000 else "standard_pilot",
            "measured_peak_rss": False,
            "limitations": "Allocation components exclude library workspaces, copies, Python objects, posterior storage and forecast paths; profile actual RSS before admission. Dense warning quantities must not be allocated."}


def _ring(spatial_columns):
    count = spatial_columns - 3  # intercept plus two static circle coordinates
    if count < 2:
        raise ValueError("At least five spatial columns are required")
    return [[math.cos(2*math.pi*j/count), math.sin(2*math.pi*j/count)]
            for j in range(count)]


def _patch(base, changes):
    result = copy.deepcopy(base)
    for path, value in changes.items():
        target = result
        parts = path.split(".")
        for key in parts[:-1]:
            target = target[key]
        target[parts[-1]] = copy.deepcopy(value)
    return result


def _controls(base, track):
    f = base["features"]
    other_modes = 8 if f["modes"] != 8 else 16
    other_conditioning = 512 if base["conditioning"] != 512 else 1024
    rank = lambda p: {"spatial_mean.centers": _ring(p)}
    if track == "C":
        return [("spatial_rank_8", rank(8)), ("spatial_rank_12", rank(12)),
                ("spatial_rank_20", rank(20)),
                ("spatial_width_015", {"spatial_mean.width": .15}),
                ("spatial_width_060", {"spatial_mean.width": .6}),
                ("spatial_width_100", {"spatial_mean.width": 1.}),
                ("pca_8", {"features.modes": 8}),
                ("operator_norm_060", {"features.recurrent_norm": .6}),
                ("input_scale_020", {"features.input_scales": [.2]}),
                ("conditioning_2048", {"conditioning": 2048})]
    # Order first covers distinct factors, then explores additional levels.
    return [("operator_norm_030", {"features.recurrent_norm": .3}),
            ("scalar_leak_010", {"features.leak_patterns": [[.1]]}),
            ("input_scale_005", {"features.input_scales": [.05]}),
            ("connectivity_002", {"features.connectivity": .02}),
            ("lag_1", {"features.lags": [1]}),
            ("pca_alternative", {"features.modes": other_modes}),
            ("conditioning_alternative", {"conditioning": other_conditioning}),
            ("spatial_rank_12", rank(12)),
            ("spatial_width_060", {"spatial_mean.width": .6}),
            ("units_256", {"features.units": [256]}),
            ("operator_norm_098", {"features.recurrent_norm": .98}),
            ("scalar_leak_100", {"features.leak_patterns": [[1.]]}),
            ("input_scale_200", {"features.input_scales": [2.]}),
            ("connectivity_020", {"features.connectivity": .2}),
            ("lags_1_2_4", {"features.lags": [1, 2, 4]}),
            ("pca_32", {"features.modes": 32}),
            ("conditioning_2048", {"conditioning": 2048}),
            ("spatial_rank_8", rank(8)),
            ("spatial_width_015", {"spatial_mean.width": .15}),
            ("spatial_width_100", {"spatial_mean.width": 1.}),
            ("scalar_leak_005", {"features.leak_patterns": [[.05]]}),
            ("no_square_terms", {"features.squares": 0})]


def _joint_templates(track):
    if track == "C":
        return [("low_rank_wide", {"spatial_mean.centers": _ring(8), "spatial_mean.width": 1.,
                 "features.units": [64], "features.squares": 32, "features.modes": 8,
                 "features.recurrent_norm": .6, "features.leak_patterns": [[.2]],
                 "features.input_scales": [.2], "features.connectivity": .02}),
                ("medium_rank_wide", {"spatial_mean.centers": _ring(12), "spatial_mean.width": .6,
                 "conditioning": 2048, "features.lags": [1], "features.input_scales": [.2]})]
    return [("units_512_small_spatial", {"spatial_mean.centers": _ring(12), "spatial_mean.width": .6,
             "features.units": [512], "features.squares": 128, "features.modes": 16,
             "features.lags": [1, 2], "conditioning": 1024, "features.recurrent_norm": .6,
             "features.leak_patterns": [[.2]], "features.input_scales": [.2],
             "features.connectivity": .02}),
            ("slow_sparse", {"spatial_mean.centers": _ring(20), "spatial_mean.width": 1.,
             "features.units": [128], "features.squares": 64, "features.modes": 16,
             "features.lags": [1, 2, 4], "conditioning": 2048, "features.recurrent_norm": .98,
             "features.leak_patterns": [[.05]], "features.input_scales": [.05],
             "features.connectivity": .02})]


def _joint_space(track):
    return {"features.recurrent_norm": [.3, .6, .9] if track == "C" else [.3, .6, .9, .98],
            "features.leak_patterns": [[[.05]], [[.1]], [[.2]], [[.5]], [[1.]], [[.2, .5, .9]]],
            "features.input_scales": [[.05], [.2], [.5], [1.], [2.]],
            "features.connectivity": [.02, .08, .2],
            "features.lags": [[1], [1, 2], [1, 2, 4]],
            "features.modes": [8, 16] if track == "C" else [8, 16, 24, 32],
            "features.units": [[32], [64], [128]] if track == "C" else [[32], [64], [128], [256], [512]],
            "square_fraction": [0., .5, 1.],
            "conditioning": [512, 1024, 2048],
            "spatial_columns": [8, 12, 16, 20] if track == "C" else [8, 12, 20, 32],
            "spatial_mean.width": [.15, .3, .6, 1.]}


def _balanced_proposals(space, seed, count):
    """Independently shuffled cycles, not a Cartesian-product enumeration.

    Proposal marginals are balanced before rejection.  Cost filtering and
    truncation may change accepted marginals; the registry records actual
    coverage and does not claim a Latin-hypercube or Hyperband guarantee.
    """
    rng = random.Random(seed)
    columns = {}
    for key, levels in space.items():
        values = []
        while len(values) < count:
            block = copy.deepcopy(levels)
            rng.shuffle(block)
            values.extend(block)
        columns[key] = values[:count]
    for i in range(count):
        row = {key: column[i] for key, column in columns.items()}
        row["features.squares"] = int(sum(row["features.units"]) * row.pop("square_fraction"))
        row["spatial_mean.centers"] = _ring(row.pop("spatial_columns"))
        yield row


def build_candidate_registry(original_config: Mapping[str, Any], frozen_spec: Mapping[str, Any], *,
                             tracks=("A", "B"), proposal_seed=20261105,
                             max_candidates=40, max_candidates_c=24,
                             max_readout_coefficients=16000,
                             include_prior_sensitivity=True, training_rows=6000) -> dict:
    """Build independent A/B/C rosters from configuration records only.

    ``max_candidates`` includes all strata (24--40 for A/B); C has its own
    16--24 cap.  Use ``selection_strata`` to keep the primary fixed-prior ranking
    distinct from prior sensitivity.  No source config, data, or fit is written.
    A previous test-informed motivation does not create a new untouched test
    set: the orchestrator must label the follow-up's evaluation status.
    """
    original, frozen = _copy(original_config), _copy(frozen_spec)
    proposal_seed = _integer(proposal_seed, "proposal_seed", 0)
    model_seed = _integer(original["model_seed"], "model_seed", 0)
    max_candidates = _integer(max_candidates, "max_candidates", 24)
    max_candidates_c = _integer(max_candidates_c, "max_candidates_c", 16)
    if max_candidates > 40 or max_candidates_c > 24:
        raise ValueError("Finite roster caps are 40 per A/B track and 24 for C")
    if not isinstance(include_prior_sensitivity, bool):
        raise ValueError("include_prior_sensitivity must be Boolean")
    tracks = tuple(tracks)
    if not tracks or len(set(tracks)) != len(tracks) or any(t not in SITES for t in tracks):
        raise ValueError("tracks must be distinct members of A, B, C")
    registry = {"schema": SCHEMA, "model": "bayes_stesn", "proposal_seed": proposal_seed,
                "model_seed": model_seed, "original_configuration_sha256": _hash(original),
                "old_freeze_identity": frozen.get("freeze_identity", frozen.get("identity")),
                "max_readout_coefficients": max_readout_coefficients,
                "response_data_read": False, "fits_run": 0, "tracks": {},
                "selection_policy": "Separate validation selection within each track and prior stratum; no seed optimization or prior-sensitivity winner relabeled as the original fixed-prior model.",
                "evaluation_status": "Follow-up motivated by previous results; this builder supplies no new untouched test set.",
                "admission_policy": "Every candidate needs training-only PCA/numerical, spatial geometry, reservoir synchronization and measured-RSS admission before expensive inference. Old history128 evidence does not qualify new dynamics."}
    for track in tracks:
        source_track = "A" if track == "C" else track
        old = original["candidates"][source_track]["bayes_stesn"]
        if len(old) != 4 or len({x["id"] for x in old}) != 4:
            raise ValueError("Exactly the four distinct original Bayesian references are required")
        chosen = frozen["selected"][source_track]["bayes_stesn"]
        base = _resolve(chosen["config"], model_seed)
        base_hash, base_prior = _hash(_active(base)), _prior_signature(base)
        if not any(effective_config_hash(x["config"], model_seed=model_seed) == base_hash for x in old):
            raise ValueError("Old selected configuration must match one of the four original references")
        limit = max_candidates_c if track == "C" else max_candidates
        candidates, by_hash, rejections = [], {}, []

        def add(name, config, origin, *, require_single=False, strict=False):
            resolved = _resolve(config, model_seed)
            digest = _hash(_active(resolved))
            if digest in by_hash:
                by_hash[digest]["aliases"].append(name)
                return False
            prior_matches = _prior_signature(resolved) == base_prior
            stratum = (MAIN_STRATUM if prior_matches else
                       LEGACY_STRATUM if origin == "legacy_reference" else PRIOR_STRATUM)
            if origin not in {"legacy_reference", "prior_sensitivity"} and not prior_matches:
                raise ValueError("A main candidate changed the selected prior hyperparameters")
            changed = _changed(base, resolved)
            if require_single and len(changed) != 1:
                raise ValueError(f"Control {name} changes {len(changed)} active fields: {changed}")
            try:
                cost = estimate_candidate_cost(resolved, observed_sites=SITES[track],
                    training_rows=training_rows, max_readout_coefficients=max_readout_coefficients)
            except ValueError as error:
                if strict:
                    raise ValueError(f"Required candidate {name}: {error}") from error
                rejections.append({"proposal": name, "reason": str(error), "effective_hash": digest})
                return False
            record = {"id": name if origin == "legacy_reference" else f"imp_{track.lower()}_{name}_{digest[:8]}",
                      "model": "bayes_stesn", "track": track, "config": resolved,
                      "effective_hash": digest, "resolved_config_sha256": _hash(resolved),
                      "origin": origin, "stratum": stratum, "aliases": [name],
                      "main_selection_eligible": stratum == MAIN_STRATUM,
                      "prior_signature_sha256": _hash(_prior_signature(resolved)),
                      "anchor_id": chosen["id"], "anchor_effective_hash": base_hash,
                      "changed_effective_fields": changed, "single_factor_control": require_single,
                      "cost": cost, "training_only_preflight_required": True,
                      "history_synchronization_passed": False,
                      "inference_qualified": False, "validation_selected": False}
            if digest == base_hash:
                record["aliases"].append(f"old_selected_{source_track}")
            candidates.append(record)
            by_hash[digest] = record
            return True

        for old_candidate in old:
            add(old_candidate["id"], old_candidate["config"], "legacy_reference", strict=True)
        prior_count = 4 if include_prior_sensitivity else 0
        main_limit = limit - prior_count
        reserve_joint = 6 if track == "C" else max(6, limit - 30)
        control_limit = max(0, main_limit - len(candidates) - reserve_joint)
        for name, change in _controls(base, track)[:control_limit]:
            add(name, _patch(base, change), "single_factor_control", require_single=True, strict=True)
        for name, change in _joint_templates(track):
            if len(candidates) < main_limit:
                add(name, _patch(base, change), "structured_joint", strict=True)
        joint_seed = int(_hash([SCHEMA, proposal_seed, track])[:16], 16)
        space = _joint_space(track)
        for i, change in enumerate(_balanced_proposals(space, joint_seed, 256)):
            if len(candidates) >= main_limit:
                break
            add(f"joint_{i:03d}", _patch(base, change), "balanced_joint")
        if len(candidates) != main_limit:
            raise ValueError("Finite proposal pool could not fill the requested unique cost-admissible roster")
        if include_prior_sensitivity:
            sensitivities = [
                ("prior_precision_x01", {"prior.mean_log_precision": base["prior"]["mean_log_precision"]-math.log(10)}),
                ("prior_precision_x10", {"prior.mean_log_precision": base["prior"]["mean_log_precision"]+math.log(10)}),
                ("prior_range_x05", {"covariance.log_ell_mean": base["covariance"]["log_ell_mean"]-math.log(2)}),
                ("prior_range_x2", {"covariance.log_ell_mean": base["covariance"]["log_ell_mean"]+math.log(2)})]
            for name, change in sensitivities:
                add(name, _patch(base, change), "prior_sensitivity", require_single=True, strict=True)
        if len(candidates) != limit:
            raise ValueError("Prior sensitivity aliases must not silently change the total roster budget")
        # Numerical complexity is explicit and applies equally to old/new
        # candidates; it is not the old candidate index or measured wall time.
        ordered = sorted(candidates, key=lambda c: (
            c["cost"]["readout_coefficients"],
            c["cost"]["eigendecomposition_dimension_cubed_proxy"],
            c["cost"]["feature_history_and_recurrent_weights_bytes"], c["id"]))
        for rank, candidate in enumerate(ordered):
            candidate["complexity_order"] = rank
        coverage = {}
        for candidate in candidates:
            if candidate["stratum"] != MAIN_STRATUM:
                continue
            for key, value in _flatten(_active(candidate["config"])).items():
                coverage.setdefault(key, {})[_hash(value)] = value
        coverage = {key: list(values.values()) for key, values in coverage.items() if len(values) > 1}
        registry["tracks"][track] = {
            "anchor_source_track": source_track, "anchor_id": chosen["id"],
            "anchor_effective_hash": base_hash, "fixed_prior_signature": base_prior,
            "independent_validation_selection_required": True,
            "reuse_A_fitted_artifacts_permitted": False if track == "C" else None,
            "candidate_budget_includes_all_strata": limit,
            "selection_strata": {label: [c["id"] for c in candidates if c["stratum"] == label]
                for label in (MAIN_STRATUM, LEGACY_STRATUM, PRIOR_STRATUM)},
            "joint_proposal_seed": joint_seed, "joint_proposal_space": space,
            "main_actual_active_parameter_coverage": coverage,
            "cost_rejected_proposals": rejections, "candidates": candidates}
    registry["identity"] = _hash(registry)
    return registry


def _write_new_registry(path, payload):
    """Publish a complete new specification atomically; never overwrite one."""
    fd, staging = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staging, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(staging).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracks", nargs="+", choices=tuple(SITES), default=["A", "B"])
    parser.add_argument("--proposal-seed", type=int, default=20261105)
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--max-candidates-c", type=int, default=24)
    parser.add_argument("--max-readout-coefficients", type=int, default=16000)
    parser.add_argument("--without-prior-sensitivity", action="store_true")
    args = parser.parse_args(argv)
    result = build_candidate_registry(json.loads(args.config.read_text()), json.loads(args.frozen.read_text()),
        tracks=args.tracks, proposal_seed=args.proposal_seed, max_candidates=args.max_candidates,
        max_candidates_c=args.max_candidates_c, max_readout_coefficients=args.max_readout_coefficients,
        include_prior_sensitivity=not args.without_prior_sensitivity)
    payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    # Linking a fully written staging file preserves an existing specification
    # even when two callers race; no truncated output becomes a valid artifact.
    _write_new_registry(args.output, payload)
    print(json.dumps({"identity": result["identity"], "path": str(args.output),
                      "counts": {t: len(r["candidates"]) for t, r in result["tracks"].items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
