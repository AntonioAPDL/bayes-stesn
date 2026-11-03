"""Resource planning and local execution ownership, independent of science.

These are conservative admission estimates, not complexity or peak-RSS
guarantees. They never reduce data, model dimensions, draws, or diagnostics.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import os
import shutil
import time

import numpy as np


def _cache_admission(current, stat):
    """Discount at most half eligible clean file cache for admission only.

    The subsets can overlap, so subtracting all exclusions is deliberately
    conservative. This is a scheduling heuristic, not a reclaim guarantee.
    The raw cgroup charge and its enforced limit are never changed.
    """
    fields = ("file", "shmem", "file_dirty", "file_writeback", "file_mapped", "unevictable")
    result = {"memory_admission_estimate_bytes": current,
              "memory_cache_eligible_bytes": 0, "memory_cache_credit_bytes": 0,
              "memory_admission_policy": "raw_usage_missing_or_invalid_cache_stats"}
    if (current is None or not isinstance(stat, dict)
            or any(not isinstance(stat.get(key), int) or isinstance(stat.get(key), bool)
                   or stat[key] < 0 for key in fields) or stat["file"] > current):
        return result
    eligible = min(current, max(0, stat["file"] - sum(stat[key] for key in fields[1:])))
    credit = eligible // 2
    result.update(memory_admission_estimate_bytes=current-credit,
                  memory_cache_eligible_bytes=eligible, memory_cache_credit_bytes=credit,
                  memory_admission_policy="half_eligible_clean_file_cache_v1")
    return result


def host_resources():
    limits = []
    try:
        limits.append(int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")))
    except (OSError, ValueError, AttributeError):
        pass
    current, current_path = None, None
    for filename in ("/sys/fs/cgroup/memory.max",
                     "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            value = Path(filename).read_text().strip()
            if value != "max" and 0 < int(value) < 2**60:
                limits.append(int(value))
        except (OSError, ValueError):
            pass
    for filename in ("/sys/fs/cgroup/memory.current",
                     "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            value = int(Path(filename).read_text().strip())
            if value >= 0:
                current, current_path = value, Path(filename)
                break
        except (OSError, ValueError):
            pass
    stat = None
    if current_path is not None and current_path.name == "memory.current":
        try:
            stat = {}
            for line in current_path.with_name("memory.stat").read_text().splitlines():
                key, value = line.split()
                if key in stat:
                    raise ValueError("Duplicate memory.stat key")
                stat[key] = int(value)
        except (OSError, ValueError):
            stat = None
    cpus = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        cpus = min(cpus, len(os.sched_getaffinity(0)))
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            cpus = min(cpus, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return {"memory_limit_bytes": min(limits) if limits else None,
            "memory_current_bytes": current, "effective_cpus": int(cpus),
            "worker_cap": max(1, min(int(cpus), 4)),
            "memory_measurement": "raw aggregate cgroup usage when available; admission cache credit is a heuristic, not guaranteed reclaim",
            "cgroup_memory_stat": stat, **_cache_admission(current, stat)}


def memory_allowance(max_memory_mb=None, *, resources=None):
    resources = host_resources() if resources is None else resources
    requested = None
    if max_memory_mb is not None:
        if not np.isfinite(max_memory_mb) or max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive and finite")
        requested = int(max_memory_mb * 1024**2)
    ceiling = resources.get("memory_limit_bytes")
    current = resources.get("memory_admission_estimate_bytes", resources.get("memory_current_bytes"))
    reserve = None if ceiling is None else min(2 * 1024**3, int(.25 * ceiling))
    bounds = []
    if requested is not None:
        bounds.append(requested)
    if ceiling is not None:
        bounds.extend((int(.65 * ceiling), ceiling - reserve))
        if current is not None:
            bounds.append(max(0, ceiling - current - reserve))
    effective = min(bounds) if bounds else None
    return {"requested_job_budget_bytes": requested,
            "effective_job_budget_bytes": effective,
            "reserved_headroom_bytes": reserve,
            "clamped": requested is not None and effective < requested,
            "host": dict(resources)}


def _candidate_dimensions(model_id, config, protocol):
    n = protocol.n_observed
    total_sites = protocol.n_side**2
    target = max(n, total_sites - n - protocol.n_calibration)
    times = protocol.final_origin - protocol.training_start
    paths = max(protocol.paths, protocol.ensemble_members)
    path_bytes = 8 * paths * protocol.horizon * (n + target)
    truth_bytes = 4 * 8 * protocol.n_times * total_sites
    neural = model_id in {"basis_gru", "dcrnn"}
    runtime_bytes = (512 if neural else 192) * 1024**2
    arrays = 8 * (6*n*times + 12*n*n)
    dims = {"observed_sites": n, "joint_forecast_sites_upper_bound": n+target,
            "supplied_history": times, "paths_upper_bound": paths}
    if model_id.startswith("bayes_esn"):
        features = config.get("features", {})
        units = features.get("units", (32,))
        units = (units,) if isinstance(units, int) else units
        modes = int(features.get("modes", min(8, n)))
        lags = features.get("lags", (1, 2))
        d = 1 + sum(units) + modes*len(lags) + int(features.get("squares", 16))
        p = int(config.get("spatial_mean", {}).get("side", 3))**2 + 3
        k = p*d
        arrays += 8*(12*d*d + 8*p*p + 8*n*d + 4*k + 4*d*times)
        sampler = config.get("sampler", {})
        prior = config.get("prior", {})
        prior_kind = prior.get("kind", "ridge") if isinstance(prior, dict) else "ridge"
        if prior_kind in {"entry_rhs", "gigg", "function_penalty"} or config.get("innovations", "iid") not in {"iid", "full_residual_ar"}:
            arrays += min(8*4*k*k, int(sampler.get("max_dense_bytes", 256*1024**2)))
        # Retained scalar hyperstates have substantial Python-container overhead.
        draws = int(sampler.get("draws", protocol.mcmc_draws))
        draws += int(config.get("continuation_draws", protocol.mcmc_continuation))
        arrays += int(sampler.get("chains", protocol.mcmc_chains))*draws*4096
        dims.update(spatial_basis=p, temporal_features_upper_bound=d,
                    readout_coefficients_upper_bound=k)
    elif model_id in {"basis_gru", "basis_lgssm", "basis_krr"}:
        rank = int(config.get("rank", 8))
        hidden = int(config.get("hidden", config.get("hidden_size", 32)))
        context = int(config.get("lags", config.get("context", 8)))
        arrays += 8*(8*rank*rank + 8*times*rank + 8*times*rank*context)
        if neural:
            batch = min(times, int(config.get("batch_size", 32)))
            arrays += 4*12*batch*max(1, context)*max(hidden, rank)
        if model_id == "basis_krr":
            centers = min(times, int(config.get("nystrom_rank", 256)))
            arrays += 8*(8*times*centers + 8*centers*centers)
        dims.update(basis_rank=rank, hidden_width=hidden, context=context)
    elif model_id == "dcrnn":
        hidden = int(config.get("hidden_size", 32))
        context = int(config.get("context", 24))
        batch = min(times, int(config.get("batch_size", 32)))
        steps = int(config.get("forecast_training_steps", 4))
        layers = int(config.get("layers", 1))
        arrays += 4*times*n*context + 4*16*batch*n*hidden*(context+steps)*layers
        dims.update(hidden_width=hidden, context=context)
    # Include output, constructor/scoring work, and representation uncertainty.
    estimate = runtime_bytes + truth_bytes + arrays + 4*path_bytes
    return {"estimated_bytes": int(estimate), "dimensions": dims,
            "predictive_tensor_bytes_upper_bound": int(path_bytes)}


def estimate_memory(protocol, workers=1):
    from .protocol import candidate_configs, effective_models
    from .registry import MODELS
    per_model = {}
    for model_id in dict.fromkeys(m for track in protocol.tracks for m in effective_models(protocol,track)):
        configurations = [config for track in protocol.tracks if track in MODELS[model_id].tracks
                          for config in candidate_configs(model_id, protocol, track=track, seed=0)]
        if not configurations:
            continue
        estimates = [_candidate_dimensions(model_id, c, protocol) for c in configurations]
        per_model[model_id] = max(estimates, key=lambda x: x["estimated_bytes"])
    maximum = max((x["estimated_bytes"] for x in per_model.values()), default=0)
    parent = 64*1024**2 + 4*8*protocol.n_times*protocol.n_side**2
    return {"estimated_per_worker_bytes": maximum,
            "estimated_total_bytes": int(workers*maximum + parent),
            "estimated_parent_bytes": int(parent), "per_model": per_model,
            "assumption": "conservative resolved-dimension admission estimate; "
                          "representative candidate seed 0, not a measured peak or guarantee; sparse fill and "
                          "allocator behavior need isolated profiles"}


def estimate_job_memory(protocol, model_id, track, seed):
    """Resolve the actual job search seed before admitting its allocations."""
    from .protocol import candidate_configs
    configurations = candidate_configs(model_id, protocol, track=track, seed=seed)
    return max((_candidate_dimensions(model_id,c,protocol) for c in configurations),
               key=lambda item:item["estimated_bytes"])


def plan_execution(protocol, *, workers=1, max_memory_mb=None, measured_fit_seconds=None):
    from .protocol import candidate_configs, effective_models
    from .registry import MODELS
    resources = host_resources()
    if not isinstance(workers, (int, np.integer)) or isinstance(workers, bool) or not 1 <= workers <= resources["worker_cap"]:
        raise ValueError(f"workers must lie between 1 and {resources['worker_cap']}")
    allowance = memory_allowance(max_memory_mb, resources=resources)
    estimates = estimate_memory(protocol, workers)
    rows = []
    for track in protocol.tracks:
        model_ids = effective_models(protocol, track)
        for model_id in model_ids:
            configurations = candidate_configs(model_id, protocol, track=track, seed=0)
            folds = getattr(protocol, "spatial_validation_blocks", 1) if track == "B" else 1
            if not isinstance(folds, int):
                folds = len(folds)
            finalists = min(len(configurations), protocol.finalist_count)
            seeds = protocol.finalist_seeds if MODELS[model_id].stochastic_fit else 1
            strict = getattr(protocol, "selection_policy", "legacy") == "qualified_finalists"
            rechecks = finalists*seeds if strict or seeds > 1 else 0
            replacement = min(max(0, len(configurations)-finalists),
                              getattr(protocol, "replacement_finalists", 0)) if strict else 0
            fit_calls = folds*(len(configurations) + rechecks)
            max_calls = fit_calls + folds*replacement*seeds
            if getattr(protocol, "selection_mode", "retuned_pipeline") == "development_frozen_configuration":
                fit_calls = max_calls = 0
            datasets = protocol.replications*len(protocol.scenarios)
            per_fit = (measured_fit_seconds or {}).get(model_id)
            rows.append({"model_id": model_id, "track": track,
                         "representative_distinct_candidates": len(configurations),
                         "spatial_folds": folds, "datasets": datasets,
                         "selection_fit_calls": fit_calls*datasets,
                         "selection_fit_calls_with_replacements": max_calls*datasets,
                         "final_fit_calls": datasets,
                         "projected_fit_wall_seconds": None if per_fit is None else
                             float(per_fit)*(max_calls+1)*datasets})
    budget = allowance["effective_job_budget_bytes"]
    feasible = budget is None or estimates["estimated_total_bytes"] <= budget
    jobs = sum(r["final_fit_calls"] for r in rows)
    datasets = protocol.replications*len(protocol.scenarios)
    retained = sum(0 <= r < protocol.replications for r in protocol.retain_paths_replications)
    raw_data = datasets*4*8*protocol.n_times*protocol.n_side**2
    raw_paths = retained*len(protocol.scenarios)*sum(
        8*protocol.paths*protocol.horizon*(protocol.n_observed if track=="A" else
                                         protocol.n_side**2-protocol.n_observed-protocol.n_calibration)
        for track in protocol.tracks for _ in effective_models(protocol,track))
    measured = [r["projected_fit_wall_seconds"] for r in rows]
    summed_fit_seconds = sum(measured) if all(x is not None for x in measured) else None
    return {"schema": "benchmark-execution-plan-1.0", "protocol_identity": protocol.identity,
            "workers": int(workers), "threads_per_worker": 1,
            "allowance": allowance, "memory": estimates, "memory_admissible": feasible,
            "jobs": jobs, "models": rows,
            "projected_summed_fit_wall_seconds": summed_fit_seconds,
            "ideal_parallel_fit_seconds_lower_bound": None if summed_fit_seconds is None else summed_fit_seconds/workers,
            "fit_time_limits_seconds":protocol.fit_time_limits_seconds,
            "storage": {"uncompressed_data_bytes":int(raw_data), "uncompressed_retained_paths_bytes":int(raw_paths),
                        "temporary_atomic_write_allowance_bytes":int(max(raw_data/max(datasets,1),raw_paths/max(retained*len(protocol.scenarios),1))),
                        "free_bytes_at_planning":shutil.disk_usage(Path.cwd()).free,
                        "scope":"array estimates before compression; fitted models, checkpoints, figures and dependency environment add storage"},
            "selection_fit_calls": sum(r["selection_fit_calls"] for r in rows),
            "selection_fit_calls_with_replacements": sum(r["selection_fit_calls_with_replacements"] for r in rows),
            "time_estimate_policy": "null until representative per-model timings are supplied; "
                                    "fit calls exclude internal neural epoch-selection refits, "
                                    "MCMC continuation, forecasts and serialization"}


@contextmanager
def campaign_lock(output):
    """One local coordinator; the OS releases the lock after interruption."""
    import fcntl
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/"run.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another coordinator owns this benchmark output") from error
        stream.seek(0); stream.truncate()
        json.dump({"pid": os.getpid(), "acquired_unix": time.time()}, stream)
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
