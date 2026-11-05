"""Full-field KS Bayesian variants, separate from the accepted PCA adapter.

This module changes the declared feature/mean/prior specification, not the
Gaussian inference equations.  Raw permitted-site states drive a frozen ESN.
For ``target='increment'``, the regression response is y[t]-y[t-1], while
features are still constructed from the original states.  At new coordinates
the persistence offset is a periodic interpolation of *input-site* states;
the native input-site recursion never decodes and reprojects its own state.

Both scalar ridge and finite-slab group/column RHS use the existing exact
column-separable Gaussian engine and collapsed hyperparameter sampler.
Coefficient diagnostics cover every coefficient with bounded RAM via a
temporary draw memmap.  The original ks_native.py adapter is unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from collections import OrderedDict
import copy
import hashlib
import mmap
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from ..basis import SpatialMean, SpatialMeanSpec
from ..data import SplitPlan, readonly
from ..diagnostics import summarize_chains
from ..features import FeatureSpec, FrozenFeatures
from ..io import content_hash, environment, load_state, save_state, seed_for
from ..model import FittedModel, ModelSpec, PreparedModel
from ..prediction import joint_domain, _psd_root
from ..priors import PriorSpec
from ..sampling import SamplerConfig, ParameterDraw
from ..benchmark.classical import _panel_state
from ..benchmark.contracts import Forecast, Panel, forecast_targets
from ..benchmark.esn import _checkpoint_info, _fit_checkpoint
from .ks_baselines import angles, periodic_decoder
from .ks_native import reference_configs, _integer, _validate_window


MODEL_ID = "bayes_stesn_fullfield"
STATE_SCHEMA = "ks-fullfield-native-fit/1"


def default_fullfield_config():
    config = copy.deepcopy(reference_configs()["bayes_stesn"])
    config.update(target="state", max_forecast_bytes=256*1024**2,
        max_forecast_working_bytes=256*1024**2,
        forecast_parameter_rng_policy="fit_shared_bank_v1",
        max_parameter_cache_bytes=256*1024**2,
        max_parameter_cache_disk_bytes=2*1024**3)
    config["conditioning"] = 512
    config["features"].update(kind="raw_esn", modes=1, units=[256],
        lags=[1, 2, 4, 8], squares=128, score_bound=None,
        recurrent_scaling="spectral_radius", spectral_radius=.9,
        raw_readout_lags="latest", max_feature_bytes=1024**3)
    config["spatial_mean"] = {"kernel": "fourier", "harmonics": 8,
        "coordinate_terms": False, "side": 2, "width": 1.}
    config["diagnostics"].update(max_coefficient_bytes=2*1024**3,
        coefficient_chunk_size=256, disk_reserve_bytes=2*1024**3)
    return config


def _resolve_fullfield(config=None, seed=0):
    seed = _integer(seed, "model seed", 0)
    supplied = copy.deepcopy(dict(config or {}))
    defaults = default_fullfield_config()
    unknown = set(supplied)-set(defaults)
    if unknown:
        raise ValueError(f"Unknown full-field KS options: {sorted(unknown)}")
    resolved = {key: ({**default, **supplied.get(key, {})}
                     if isinstance(default, dict) else supplied.get(key, default))
                for key, default in defaults.items()}
    if resolved["target"] not in {"state", "increment"}:
        raise ValueError("Full-field target must be state or increment")
    resolved["conditioning"] = _integer(resolved["conditioning"], "conditioning", 10)
    resolved["continuation_draws"] = _integer(resolved["continuation_draws"], "continuation_draws", 0)
    resolved["max_forecast_bytes"] = _integer(resolved["max_forecast_bytes"], "max_forecast_bytes")
    resolved["max_forecast_working_bytes"] = _integer(
        resolved["max_forecast_working_bytes"], "max_forecast_working_bytes")
    for key in ("max_parameter_cache_bytes", "max_parameter_cache_disk_bytes"):
        resolved[key] = _integer(resolved[key], key, 0)
    if resolved["forecast_parameter_rng_policy"] not in {"fit_shared_bank_v1", "per_origin_v1"}:
        raise ValueError("Unknown full-field predictive parameter RNG policy")

    f = dict(resolved["features"])
    f["seed"] = seed_for(seed, "reservoir")
    if f.get("kind") != "raw_esn" or f.get("recurrent_scaling") != "spectral_radius":
        raise ValueError("The full-field variant requires raw_esn with actual spectral-radius scaling")
    # PCA/other encoder defaults are inactive, and cannot manufacture distinct
    # effective configurations. Raw dimension is the supplied panel's n_sites.
    feature_keys = ("kind", "units", "lags", "input_scales", "leak_patterns",
        "score_bound", "squares", "products", "seed", "connectivity",
        "feature_scale_floor", "max_feature_bytes", "recurrent_scaling",
        "spectral_radius", "raw_readout_lags")
    features = FeatureSpec(**{key: f[key] for key in feature_keys if key in f})
    if len(features.units) != 1:
        raise ValueError("This declared full-field study supports one reservoir layer")
    if resolved["conditioning"] <= max(features.lags):
        raise ValueError("Conditioning must exceed the maximum feature lag")

    prior_input = PriorSpec(**resolved["prior"])
    if prior_input.kind == "ridge":
        prior = PriorSpec.ridge(mean_log_precision=prior_input.mean_log_precision,
                                sd_log_precision=prior_input.sd_log_precision)
        parameterization = "ridge_total_scale"
    elif prior_input.kind == "rhs" and prior_input.unit in {"group", "column"}:
        prior = PriorSpec.rhs(unit=prior_input.unit, tau0=prior_input.tau0,
            slab_scale=prior_input.slab_scale, slab_df=prior_input.slab_df,
            fixed_slab=prior_input.fixed_slab, unshrunk_groups=prior_input.unshrunk_groups,
            fixed_variance=prior_input.fixed_variance)
        parameterization = "legacy"
    else:
        raise ValueError("Full-field readout supports scalar ridge or group/column regularized horseshoe")

    spatial_input = SpatialMeanSpec(**resolved["spatial_mean"])
    if spatial_input.kernel == "fourier":
        spatial = SpatialMeanSpec(kernel="fourier", harmonics=spatial_input.harmonics,
            coordinate_terms=False, side=2, width=1., max_columns=spatial_input.max_columns)
    else:
        spatial = spatial_input
    spec = ModelSpec.from_dict({"features": asdict(features),
        "spatial_mean": asdict(spatial), "prior": asdict(prior),
        "covariance": resolved["covariance"], "innovations": resolved["innovations"],
        "name": MODEL_ID})
    covariance = spec.covariance
    if (spec.innovations != "iid" or covariance.kind != "matern"
            or covariance.scale_mode != "relative" or covariance.static_variance
            or covariance.nonstationary or covariance.anisotropy or covariance.infer_nu
            or covariance.nu != 1.5):
        raise ValueError("Full-field variants retain iid Gaussian stationary isotropic Matérn-3/2 residuals")

    sampler_input = dict(resolved["sampler"])
    # The target prior decides its compatible proposal coordinates. This is
    # an exposed resolved option, never a scalar-ridge proposal applied to RHS.
    sampler_input.update(seed=seed_for(seed, "mcmc"), parameterization=parameterization)
    if sampler_input.get("algorithm", "auto") not in {"auto", "collapsed"}:
        raise ValueError("This full-field adapter uses the exact separable collapsed sampler")
    sampler_input["algorithm"] = "collapsed"
    if sampler_input.get("backend", "auto") not in {"auto", "spectral_gaussian"}:
        raise ValueError("This full-field adapter requires the exact spectral Gaussian backend")
    sampler_input["backend"] = "spectral_gaussian"
    sampler = SamplerConfig(**sampler_input)
    if sampler.temporal != "iid":
        raise ValueError("Full-field residuals are independent over time")
    checks = dict(resolved["diagnostics"])
    allowed = {"functional_draws", "rhat_limit", "min_ess", "min_chains",
               "max_coefficient_bytes", "coefficient_chunk_size", "disk_reserve_bytes"}
    if set(checks)-allowed:
        raise ValueError(f"Unknown full-field diagnostic options: {sorted(set(checks)-allowed)}")
    for key, minimum in (("functional_draws", 8), ("min_chains", 4),
                         ("max_coefficient_bytes", 1), ("coefficient_chunk_size", 1),
                         ("disk_reserve_bytes", 0)):
        checks[key] = _integer(checks[key], f"diagnostics.{key}", minimum)
    if not (1 < float(checks["rhat_limit"]) <= 1.01) or float(checks["min_ess"]) < 400:
        raise ValueError("Full-field inference cannot weaken R-hat<=1.01 or ESS>=400 gates")
    resolved.update({key: value for key, value in spec.to_dict().items() if key != "name"})
    resolved.update(sampler=asdict(sampler), diagnostics=checks)
    return resolved, spec, sampler


def resolve_fullfield_config(config=None, seed=0):
    """Return canonical effective settings; no training data or labels read."""
    return _resolve_fullfield(config, seed)[0]


def prepare_fullfield(training_panel, config=None, seed=0):
    """Prepare conditional regression while keeping original states for inputs.

    ``data.y`` always contains the permitted states. ``standardized_y`` is the
    likelihood response (states or increments). Consumers must use the former
    for causal replay; this module deliberately does not use forecast_context.
    """
    if not isinstance(training_panel, Panel):
        raise TypeError("Full-field KS fitting accepts only a permitted observation Panel")
    config, spec, _ = _resolve_fullfield(config, seed)
    c, data = config["conditioning"], training_panel
    if c >= len(data.y)-5:
        raise ValueError("Conditioning must leave at least six likelihood targets")
    angles(data.coordinates)
    y = readonly(data.y)
    features = FrozenFeatures.fit(y, c, data.coordinates, spec.features)
    spec = replace(spec, features=features.spec)
    basis = SpatialMean.fit(data.coordinates, spec.spatial_mean)
    phi, z = readonly(basis.transform(data.coordinates)), readonly(features.Z)
    if config["target"] == "state":
        response = y
    else:
        response = np.zeros_like(y)
        response[1:] = y[1:]-y[:-1]
        response = readonly(response)
    reference = readonly(z[c//3:c])
    split = SplitPlan(c, len(data.y))
    identity = content_hash({"schema": "ks-fullfield-preparation/1",
        "data": _panel_state(data), "split": asdict(split), "spec": spec.to_dict(),
        "features": features.fingerprint, "target": config["target"],
        "likelihood_response": response,
        "response_transform": "identity_on_delivered_training_only_scale",
        "increment_offset": "previous_input_state; periodic_interpolation_at_new_sites"})
    return PreparedModel(spec, data, split, features, basis, 0., 1., response,
                         phi, z, reference, features.groups, identity)


def _computation_identity():
    from threadpoolctl import threadpool_info
    root = Path(__file__).resolve().parents[1]
    names = ("applications/ks_fullfield_native.py", "applications/ks_native.py",
        "applications/ks_baselines.py", "features.py", "raw_reservoir.py", "_reservoir.py",
        "basis.py", "model.py", "sampling.py", "gaussian.py", "spatial.py", "priors.py",
        "diagnostics.py", "prediction.py", "data.py", "io.py", "benchmark/esn.py",
        "benchmark/contracts.py", "benchmark/classical.py")
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update((root/name).read_bytes())
    runtime = environment()
    runtime["blas"] = [{key: item.get(key) for key in
        ("internal_api", "num_threads", "version", "architecture")}
        for item in threadpool_info() if item.get("user_api") == "blas"]
    return {"source_sha256": digest.hexdigest(), "runtime": runtime}


def _diagnose_fullfield(fitted, target, checks, seed, workdir):
    """Check every coefficient, charging the full mapped tensor to headroom.

    Memmaps reduce anonymous allocations, but mapped/file-cache pages can
    remain resident. Admission therefore includes the complete tensor; scoped
    page eviction is only an additional reduction, not a memory guarantee.
    """
    started = time.perf_counter()
    posterior, p = fitted.posterior, fitted.prepared
    count = min(checks["functional_draws"], posterior.n_draws)
    if count < 8:
        raise ValueError("Full-field diagnostics need at least eight retained states per chain")
    indices = np.linspace(0, posterior.n_draws-1, count, dtype=int)
    spatial, temporal = p.Phi.shape[1], p.fit_Z.shape[1]
    ncoef = spatial*temporal
    total_bytes = 8*posterior.n_chains*count*ncoef
    if total_bytes > checks["max_coefficient_bytes"]:
        raise MemoryError(f"All-coefficient diagnostic temporary storage needs {total_bytes} bytes; "
                          f"cap is {checks['max_coefficient_bytes']}")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(workdir).free-total_bytes < checks["disk_reserve_bytes"]:
        raise MemoryError("Diagnostic temporary storage would cross the declared disk reserve")
    gates = {key: checks[key] for key in ("rhat_limit", "min_ess", "min_chains")}
    hyper = summarize_chains(posterior.chains, posterior.parameter_names, **gates).to_dict()
    locations = np.array([0, len(p.Phi)//2, len(p.Phi)-1])
    names = ("log_sigma2_delivered_scale", "last_fitted_mean_first_site",
        "last_fitted_mean_middle_site", "last_fitted_mean_last_site", "last_fitted_spatial_mean",
        "next_mean_first_site", "next_mean_middle_site", "next_mean_last_site",
        "next_spatial_mean", "log_conditional_innovation_variance")
    values = np.empty((posterior.n_chains, count, len(names)))
    state_y = p.data.y
    _, next_z = p.features.update(state_y[-1], p.features.states[-1])
    last_offset = state_y[-2] if target == "increment" else 0.
    next_offset = state_y[-1] if target == "increment" else 0.
    coefficient = {"passed": True, "chains": posterior.n_chains, "draws": count,
                   "messages": [], "parameters": {}}
    chunk = min(checks["coefficient_chunk_size"], ncoef)
    # A second bounded copy is made for each diagnostic block; whole-chain
    # recovery writes each parameter draw once rather than redrawing it per block.
    peak_chunk_bytes = 8*posterior.n_chains*count*chunk
    if peak_chunk_bytes > 256*1024**2:
        raise MemoryError("Coefficient diagnostic chunk exceeds the 256 MiB working-array limit")
    # An existing prepared fit is already resident. Charge all additional
    # mapped pages plus working copies against actual container headroom.
    # Do not credit potentially reclaimable caches for this admission gate.
    worst_additional_bytes = total_bytes+4*peak_chunk_bytes+32*temporal**2+16*ncoef
    memory_limit, memory_current = Path('/sys/fs/cgroup/memory.max'), Path('/sys/fs/cgroup/memory.current')
    measured_headroom = None
    if memory_limit.exists() and memory_current.exists():
        limit = memory_limit.read_text().strip()
        if limit != 'max':
            measured_headroom = max(0, int(limit)-int(memory_current.read_text()))
            if 1.5*worst_additional_bytes > .65*measured_headroom:
                raise MemoryError("All-coefficient diagnostics exceed conservative container-aware "
                    f"headroom: 1.5*{worst_additional_bytes} > .65*{measured_headroom}")
    with tempfile.TemporaryDirectory(prefix="fullfield-diagnostic-", dir=workdir) as temporary:
        path = Path(temporary)/"coefficient_draws.dat"
        samples = np.memmap(path, mode="w+", dtype=np.float64,
                            shape=(posterior.n_chains, count, ncoef))
        def release_owned_pages():
            samples.flush()
            if hasattr(samples._mmap, 'madvise') and hasattr(mmap, 'MADV_DONTNEED'):
                samples._mmap.madvise(mmap.MADV_DONTNEED)
            if hasattr(os, 'posix_fadvise') and hasattr(os, 'POSIX_FADV_DONTNEED'):
                with path.open('rb') as owned_file:
                    os.posix_fadvise(owned_file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        try:
            for chain in range(posterior.n_chains):
                for position, draw in enumerate(indices):
                    parameter = posterior.draw_parameters(chain, int(draw),
                        np.random.default_rng(seed_for(seed, "diagnostic", chain, int(draw))))
                    samples[chain, position] = parameter.C.reshape(-1, order="F")
                    mean = last_offset+p.Phi@(parameter.C@p.fit_Z[-1])
                    nxt = next_offset+p.Phi@(parameter.C@next_z)
                    log_variance = (np.log(parameter.sigma2)
                        +np.logaddexp(0., float(parameter.covariance_parameters["log_q"])))
                    values[chain, position] = np.r_[np.log(parameter.sigma2), mean[locations],
                        float(mean.mean()), nxt[locations], float(nxt.mean()), log_variance]
                    if (position+1) % 32 == 0:
                        release_owned_pages()
                release_owned_pages()
            for start in range(0, ncoef, chunk):
                stop = min(start+chunk, ncoef)
                block = np.array(samples[:, :, start:stop], copy=True)
                labels = tuple(f"C[{j % spatial},{j // spatial}]" for j in range(start, stop))
                summary = summarize_chains(block, labels, **gates).to_dict()
                coefficient["passed"] = bool(coefficient["passed"] and summary["passed"])
                coefficient["parameters"].update(summary["parameters"])
                coefficient["messages"].extend(summary["messages"])
                del block
                release_owned_pages()
        finally:
            samples._mmap.close()
            del samples
    coefficient["messages"] = list(dict.fromkeys(coefficient["messages"]))
    functional = summarize_chains(values, names, **gates).to_dict()
    return {"passed": bool(hyper["passed"] and functional["passed"] and coefficient["passed"]),
        "hyperparameters": hyper, "coefficients": coefficient, "functionals": functional,
        "diagnostic_seconds": time.perf_counter()-started,
        "monitoring": {"all_readout_coefficients": True, "coefficient_count": ncoef,
            "coefficient_storage_bytes": total_bytes, "coefficient_storage": "temporary_memmap",
            "coefficient_working_block_bytes": peak_chunk_bytes,
            "conservative_additional_memory_bytes": worst_additional_bytes,
            "container_headroom_before_diagnostics_bytes": measured_headroom,
            "mapped_pages_counted_as_resident": True,
            "max_coefficient_bytes": checks["max_coefficient_bytes"],
            "draws_per_chain": count, "retained_indices": indices.tolist(),
            "coefficient_vector_order": "column_major", "site_indices": locations.tolist(),
            "target": target, "increment_functionals_include_persistence_offset": target == "increment",
            "seed": int(seed),
            "recovery": "joint_conditional_Gaussian_and_IG_given_each_selected_hyperstate",
            "restricted_monitoring": "coefficients_and_functionals_use_evenly_spaced_retained_states; all_hyperstates_checked"}}


def _recover_fullfield(prepared, config, sampler, checkpoint, computation, seed):
    """Use the existing exact sampler and one durable diagnostic continuation."""
    maximum = sampler.draws+config["continuation_draws"]
    phase_path = checkpoint.with_name(checkpoint.name+".fullfield-phase.npz")
    transition = asdict(sampler)
    transition.pop("checkpoint_every", None)
    identity = content_hash({"schema": "ks-fullfield-fit-phase/1", "prepared": prepared.identity,
        "sampler": transition, "continuation_draws": config["continuation_draws"],
        "diagnostics": config["diagnostics"], "seed": seed, "computation": computation})
    info = (_checkpoint_info(checkpoint, prepared, sampler, maximum)
            if checkpoint.exists() else None)
    resumed = phase_path.exists() or info is not None
    if phase_path.exists():
        state = load_state(phase_path, kind="ks_fullfield_fit_phase")
        if state.get("identity") != identity:
            raise ValueError("Full-field phase has incompatible model, data, source, runtime or allowance")
        if state["phase"] not in {"initial_sampling", "initial_diagnostics",
                "continuation_sampling", "continuation_diagnostics", "complete"}:
            raise ValueError("Unknown full-field recovery phase")
        if info is None and state["phase"] != "initial_sampling":
            raise ValueError("Full-field phase requires its missing chain checkpoint")
        if info is not None and info["draw_target"] > state["target_draws"]:
            raise ValueError("Full-field chain exceeds its phase allowance")
        if state["phase"] == "complete" and (not info["complete_at_target"]
                or info["sha256"] != state["checkpoint"]["sha256"]):
            raise ValueError("Completed full-field phase differs from its retained chain")
    else:
        if info is not None:
            raise ValueError("Existing full-field chain has no phase receipt; preserve and inspect")
        state = {"identity": identity, "phase": "initial_sampling", "target_draws": sampler.draws,
            "initial_draws": sampler.draws, "maximum_draws": maximum,
            "gates": [], "checkpoint": None, "completed_phase_seconds": {}}
        save_state(phase_path, state, kind="ks_fullfield_fit_phase")
    fitted = None
    while True:
        phase = state["phase"]
        cfg = replace(sampler, draws=int(state["target_draws"]))
        if phase.endswith("sampling"):
            started = time.perf_counter()
            fitted = _fit_checkpoint(prepared, cfg, checkpoint)
            state["completed_phase_seconds"][phase] = time.perf_counter()-started
            state["phase"] = phase.replace("sampling", "diagnostics")
            state["checkpoint"] = _checkpoint_info(checkpoint, prepared, sampler, maximum)
            save_state(phase_path, state, kind="ks_fullfield_fit_phase")
            continue
        if fitted is None:
            fitted = _fit_checkpoint(prepared, cfg, checkpoint)
        if phase == "complete":
            if not 1 <= len(state["gates"]) <= 2:
                raise ValueError("Completed full-field fit has invalid diagnostic history")
            break
        started = time.perf_counter()
        gate = _diagnose_fullfield(fitted, config["target"], config["diagnostics"],
                                  seed, checkpoint.parent)
        state["completed_phase_seconds"][phase] = time.perf_counter()-started
        state["gates"].append(gate)
        if phase == "initial_diagnostics" and not gate["passed"] and config["continuation_draws"]:
            state["phase"], state["target_draws"] = "continuation_sampling", maximum
            fitted = None
        else:
            state["phase"] = "complete"
        save_state(phase_path, state, kind="ks_fullfield_fit_phase")
    return fitted, state["gates"], {"resumed": resumed, "phase": state["phase"],
        "initial_draws": sampler.draws, "maximum_draws": maximum,
        "completed_phase_seconds": state["completed_phase_seconds"],
        "continuation_policy": "one extension after any unchanged coefficient/hyperparameter/functional gate fails"}


class FullFieldKSFit:
    """Bayesian raw-state KS forecaster with an explicit conditional mean target."""
    model_id = MODEL_ID

    def __init__(self, core, config, seed, identity, input_identity, computation, diagnostics):
        self.core, self.config, self.seed = core, config, int(seed)
        self.identity, self.input_identity = identity, input_identity
        self.computation, self.diagnostics = computation, diagnostics
        self._parameter_cache = OrderedDict()
        self._parameter_cache_bytes = 0
        self._parameter_cache_dir = None

    @property
    def training(self):
        return self.core.prepared.data

    def forecast(self, history_panel, horizon, target_coordinates=None,
                 target_site_ids=None, n_paths=256, seed=0, *, path_start=0,
                 parameter_seed=None, require_qualified=True):
        return self._forecast_with_context(history_panel, horizon, target_coordinates,
            target_site_ids, n_paths, seed, path_start=path_start,
            parameter_seed=parameter_seed, require_qualified=require_qualified)

    def _origin_context(self, history_panel):
        origin, delta = _validate_window(self.training, history_panel)
        p = self.core.prepared
        if len(history_panel.y) <= max(p.spec.features.lags):
            raise ValueError("Forecast history must exceed the largest reservoir lag")
        states, _ = p.features.history(history_panel.y)
        return origin, delta, readonly(states[-1])

    def _forecast_with_context(self, history_panel, horizon, target_coordinates=None,
                 target_site_ids=None, n_paths=256, seed=0, *, path_start=0,
                 parameter_seed=None, require_qualified=True, context=None):
        horizon = _integer(horizon, "horizon")
        paths = _integer(n_paths, "n_paths")
        path_start = _integer(path_start, "path_start", 0)
        seed = _integer(seed, "forecast seed", 0)
        if parameter_seed is not None:
            parameter_seed = _integer(parameter_seed, "parameter seed", 0)
        if require_qualified and not self.diagnostics.get("qualified", False):
            raise RuntimeError("Full-field posterior is inference-unqualified; set require_qualified=False "
                               "only for explicitly labeled resource or diagnostic forecasts")
        if require_qualified and getattr(self, '_loaded_computation_mismatch', False):
            raise RuntimeError("A source/runtime-mismatch artifact is restricted to explicit diagnostic forecasts")
        # Context is private and created only from this supplied immutable
        # history. iter_forecasts reuses it for this origin's path batches.
        origin, delta, initial_state = self._origin_context(history_panel) if context is None else context
        coords, ids = forecast_targets(self.training, target_coordinates, target_site_ids)
        p, posterior = self.core.prepared, self.core.posterior
        required = 8*paths*horizon*len(coords)
        if required > self.config["max_forecast_bytes"]:
            raise MemoryError("Forecast output allocation exceeds max_forecast_bytes; use path batches")
        joint, target_index = joint_domain(p.data.coordinates, coords,
                                          input_ids=p.data.site_ids, target_ids=ids)
        n, m = len(p.data.coordinates), len(joint)
        phi = p.basis.transform(joint)
        increment = self.config["target"] == "increment"
        decoder = periodic_decoder(p.data.coordinates, joint[n:]) if increment and m > n else None
        output = np.empty((paths, horizon, len(coords)), dtype=np.float64)
        draw_ids = np.empty((paths, 2), dtype=np.int64)
        d = p.fit_Z.shape[1]
        # Bound transformed readouts, innovations and recurrent work arrays.
        # Static parameter recovery remains once per path (and cached across
        # origins); the expensive dense reservoir-input multiply is a GEMM.
        per_path_bytes = 8*(m*d+3*horizon*m+4*p.features.core.state_size+4*d+2*m*m)
        block_size = min(32, paths, self.config["max_forecast_working_bytes"]//per_path_bytes)
        if block_size < 1:
            raise MemoryError("One forecast path exceeds max_forecast_working_bytes")
        for start in range(0, paths, block_size):
            stop = min(start+block_size, paths)
            size = stop-start
            mean_maps = np.empty((size, m, d), dtype=np.float64)
            innovations = np.empty((size, horizon, m), dtype=np.float64)
            for local, path in enumerate(range(start, stop)):
                global_path = path_start+path
                innovation_rng = np.random.default_rng(seed_for(seed, "innovation", global_path))
                parameter, chain, draw, bank_seed = self._parameter_draw(
                    global_path, seed, parameter_seed)
                draw_ids[path] = chain, draw
                sigma2, cp = float(parameter.sigma2), parameter.covariance_parameters
                if not np.isfinite(sigma2) or sigma2 <= 0:
                    raise ValueError("Invalid posterior residual variance")
                K, V = (p.spec.covariance.matrices(joint, cp) if m == n else
                        p.spec.covariance.joint_matrices(p.data.coordinates, joint[n:], cp))
                factor = _psd_root(K)
                q = float(np.exp(cp["log_q"]))
                latent_scale = np.sqrt(sigma2*q)
                nugget_scale = np.sqrt(sigma2)*np.sqrt(np.diag(V))
                mean_maps[local] = phi@parameter.C
                # The interleaved lead/channel order is identical to two
                # normal(size=m) calls at each lead of this path. No random
                # stream depends on batching or another path's outcome.
                white = innovation_rng.normal(size=(horizon, 2, m))
                innovations[local] = (latent_scale*(white[:, 0]@factor.T)
                                      +white[:, 1]*nugget_scale)
            previous = np.tile(history_panel.y[-1], (size, 1))
            state = np.tile(initial_state, (size, 1))
            for lead in range(horizon):
                state, z = p.features.update_batch(previous, state)
                mean = np.einsum('pmd,pd->pm', mean_maps, z, optimize=False)
                if increment:
                    # The first n entries are updated in native observed-site
                    # coordinates, exactly. Only new-site offsets are decoded.
                    mean[:, :n] += previous
                    if m > n:
                        mean[:, n:] += previous@decoder.T
                value = mean+innovations[:, lead]
                if not np.isfinite(value).all():
                    raise FloatingPointError(f"Nonfinite autonomous full-field forecast at lead {lead+1}")
                output[start:stop, lead] = value[:, target_index]
                previous = value[:, :n]
        mean = np.mean(output, axis=0, dtype=np.float64)
        metadata = {"model": MODEL_ID, "fit_identity": self.identity,
            "parameter_policy": "one_static_joint_draw_per_whole_path",
            "parameter_uncertainty": True, "uncertainty": "posterior_joint",
            "target": "clean_simulated_field", "regression_target": self.config["target"],
            "residual_interpretation": "statistical_discrepancy_for_clean_field_no_added_sensor_noise",
            "mean_estimator": "average_of_returned_complete_predictive_paths",
            "qualified": bool(self.diagnostics.get("qualified", False)),
            "qualification_override": not require_qualified,
            "loaded_computation_mismatch_override": bool(
                getattr(self, '_loaded_computation_mismatch', False)),
            "origin_index": origin, "first_target_index": origin+1,
            "origin_time": float(history_panel.times[-1]),
            "first_target_time": float(history_panel.times[-1]+delta),
            "history_length": len(history_panel.y), "history_first_time": float(history_panel.times[0]),
            "state_initialization": "zero_reservoir_at_explicit_window_start",
            "observation_access": "all_and_only_permitted_input_sites_in_supplied_window",
            "forecast_feedback": "native_generated_input_site_states_no_decoder_reprojection",
            "increment_offset": "native_persistence_at_inputs_periodic_interpolation_at_unseen_sites"
                if increment else "none",
            "newsite_response_history": "never_used", "path_count": paths,
            "path_start": path_start, "path_stop": path_start+paths, "forecast_seed": seed,
            "parameter_draw_ids_sha256": content_hash(draw_ids),
            "rng_scheme": "SHA256_parameter_and_innovation_streams_by_global_path",
            "forecast_parameter_rng_policy": self.config["forecast_parameter_rng_policy"],
            "parameter_seed": bank_seed,
            "forecast_internal_path_batch": block_size,
            "forecast_working_allocation_estimate_bytes": block_size*per_path_bytes,
            "parameter_MC_dependence": "shared_parameter_draws_across_origins; fresh_origin_seed_innovations"
                if self.config["forecast_parameter_rng_policy"] == "fit_shared_bank_v1"
                else "fresh_origin_seed_parameter_recovery",
            "finite_ensemble_score_policy": "empirical_predictive_distribution; no_iid_unbiased_correction_claim",
            "scale": "delivered_standardized_response; evaluator_inverts_verified_transform"}
        return Forecast._from_owned(mean, output, coords, ids, metadata)

    def iter_forecasts(self, history_panel, horizon, target_coordinates=None,
                       target_site_ids=None, n_paths=256, seed=0, *, batch_size=32, path_start=0,
                       parameter_seed=None, require_qualified=True):
        n_paths = _integer(n_paths, "n_paths")
        batch_size = _integer(batch_size, "batch_size")
        path_start = _integer(path_start, "path_start", 0)
        if require_qualified and not self.diagnostics.get("qualified", False):
            raise RuntimeError("Full-field posterior is inference-unqualified; set require_qualified=False "
                               "only for explicitly labeled resource or diagnostic forecasts")
        context = self._origin_context(history_panel)
        for offset in range(0, n_paths, batch_size):
            yield self._forecast_with_context(history_panel, horizon, target_coordinates, target_site_ids,
                min(batch_size, n_paths-offset), seed, path_start=path_start+offset,
                parameter_seed=parameter_seed, require_qualified=require_qualified, context=context)

    def _parameter_draw(self, global_path, forecast_seed, parameter_seed=None):
        """A reproducible joint posterior draw, optionally shared across origins.

        Sharing changes cross-origin Monte Carlo dependence, not any origin's
        posterior predictive law. A different parameter_seed requests a fresh
        precision-check bank. Cache eviction only causes exact regeneration.
        """
        shared = self.config["forecast_parameter_rng_policy"] == "fit_shared_bank_v1"
        bank_seed = (seed_for(self.seed, "fullfield_predictive_parameter_bank")
                     if parameter_seed is None and shared else
                     (forecast_seed if parameter_seed is None else int(parameter_seed)))
        key = (int(bank_seed), int(global_path))
        if shared and key in self._parameter_cache:
            item, size = self._parameter_cache.pop(key)
            self._parameter_cache[key] = item, size
            return (*item, bank_seed)
        posterior = self.core.posterior
        # Retain the original named stream for per_origin_v1 equivalence. The
        # shared policy binds the stream to this posterior and implementation.
        random_seed = (seed_for(bank_seed, "parameter", global_path) if not shared else
            seed_for(bank_seed, "fullfield-shared-parameter/1", self.identity,
                     self.core.posterior_identity, global_path))
        parameter_rng = np.random.default_rng(random_seed)
        chain = int(parameter_rng.integers(posterior.n_chains))
        draw = int(parameter_rng.integers(posterior.n_draws))
        expected = content_hash({"schema": "ks-fullfield-parameter-bank/1",
            "fit_identity": self.identity, "posterior_identity": self.core.posterior_identity,
            "computation": self.computation, "bank_seed": bank_seed, "path": global_path,
            "chain": chain, "draw": draw})
        cache_path = None
        if shared and self._parameter_cache_dir is not None:
            cache_path = self._parameter_cache_dir/str(bank_seed)/f"path_{global_path:08d}.npz"
        if cache_path is not None and cache_path.exists():
            saved = load_state(cache_path, kind="ks_fullfield_parameter_draw")
            if saved.get("identity") != expected:
                raise ValueError("Shared parameter-bank cache has an incompatible identity")
            parameter = ParameterDraw(**saved["parameter"])
        else:
            parameter = posterior.draw_parameters(chain, draw, parameter_rng)
            if cache_path is not None and self.config["max_parameter_cache_disk_bytes"]:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                # This is a recomputable cache, so exceeding its own storage
                # allowance omits caching rather than changing the draw.
                required = int(parameter.C.nbytes)+65536
                used = sum(p.stat().st_size for p in self._parameter_cache_dir.rglob("*.npz"))
                reserve = self.config["diagnostics"]["disk_reserve_bytes"]
                if (used+required <= self.config["max_parameter_cache_disk_bytes"]
                        and shutil.disk_usage(cache_path.parent).free-required >= reserve):
                    save_state(cache_path, {"identity": expected, "parameter": asdict(parameter)},
                               kind="ks_fullfield_parameter_draw")
        if shared:
            size = int(parameter.C.nbytes)+4096
            cap = self.config["max_parameter_cache_bytes"]
            if size <= cap:
                while self._parameter_cache and self._parameter_cache_bytes+size > cap:
                    _, (_, old_size) = self._parameter_cache.popitem(last=False)
                    self._parameter_cache_bytes -= old_size
                self._parameter_cache[key] = (parameter, chain, draw), size
                self._parameter_cache_bytes += size
        return parameter, chain, draw, bank_seed

    def _attach_parameter_cache(self, path):
        path = Path(path)
        self._parameter_cache_dir = path.with_name(path.name+".parameter-bank")
        return self

    def to_state(self):
        return {"schema": STATE_SCHEMA, "model_id": MODEL_ID, "core": self.core.to_state(),
            "config": self.config, "seed": self.seed, "identity": self.identity,
            "input_identity": self.input_identity, "computation": self.computation,
            "diagnostics": self.diagnostics}

    @classmethod
    def from_state(cls, state):
        if state.get("schema") != STATE_SCHEMA or state.get("model_id") != MODEL_ID:
            raise ValueError("Unsupported full-field KS fit artifact")
        return cls(FittedModel.from_state(state["core"]), **{key: state[key] for key in
            ("config", "seed", "identity", "input_identity", "computation", "diagnostics")})

    def save(self, path):
        return save_state(path, self.to_state(), kind="ks_fullfield_fit")

    @classmethod
    def load(cls, path, *, allow_computation_mismatch=False):
        fitted = cls.from_state(load_state(path, kind="ks_fullfield_fit"))
        mismatch = content_hash(fitted.computation) != content_hash(_computation_identity())
        if mismatch and not allow_computation_mismatch:
            raise ValueError("Full-field artifact has incompatible numerical source/runtime; "
                "only an explicitly labeled diagnostic may allow_computation_mismatch=True")
        fitted._loaded_computation_mismatch = bool(mismatch)
        return fitted._attach_parameter_cache(path)


def load_fullfield_fit(path, *, allow_computation_mismatch=False):
    return FullFieldKSFit.load(path, allow_computation_mismatch=allow_computation_mismatch)


def fit_fullfield(training_panel, config=None, seed=0, checkpoint_path=None):
    """Fit/resume one declared Bayesian variant from permitted training states."""
    if not isinstance(training_panel, Panel):
        raise TypeError("Full-field KS fitting accepts an observed Panel only")
    if len(training_panel.y) < 12 or training_panel.y.shape[1] < 2:
        raise ValueError("Full-field KS requires at least twelve rows and two permitted sites")
    seed = _integer(seed, "model seed", 0)
    config, _, sampler = _resolve_fullfield(config, seed)
    computation = _computation_identity()
    input_identity = content_hash({"panel": _panel_state(training_panel),
                                  "metadata": dict(training_panel.metadata)})
    identity = content_hash({"schema": STATE_SCHEMA, "model_id": MODEL_ID,
        "input_identity": input_identity, "config": config, "seed": seed,
        "computation": computation})
    path = None if checkpoint_path is None else Path(checkpoint_path)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            state = load_state(path, kind="ks_fullfield_fit")
            if state.get("identity") != identity:
                raise ValueError("Full-field checkpoint has incompatible data, transform, model, source or runtime")
            return FullFieldKSFit.from_state(state)._attach_parameter_cache(path)
    started = time.perf_counter()
    prepared = prepare_fullfield(training_panel, config, seed)
    with tempfile.TemporaryDirectory(prefix="ks-fullfield-fit-",
            dir=None if path is None else path.parent) as temporary:
        chain_path = (Path(temporary)/"chains.npz" if path is None else
                      path.with_name(path.name+".chains.npz"))
        core, gates, recovery = _recover_fullfield(prepared, config, sampler, chain_path,
            computation, seed_for(seed, "diagnostics"))
    ncoef = prepared.Phi.shape[1]*prepared.Z.shape[1]
    diagnostics = {"qualified": bool(gates[-1]["passed"]), "gates": gates,
        "recovery": recovery, "continuations": len(gates)-1,
        "conditioning_rows": config["conditioning"],
        "likelihood_rows": len(training_panel.y)-config["conditioning"],
        "temporal_features": prepared.Z.shape[1], "spatial_basis": prepared.Phi.shape[1],
        "readout_coefficients": ncoef,
        "parameter_count": ncoef+len(core.posterior.parameter_names)+1,
        "inference": "exact_separable_collapsed_MH_and_conditional_Gaussian_IG_recovery",
        "prior": config["prior"]["kind"], "target": config["target"],
        "raw_field_input_sites": training_panel.y.shape[1], "pca_used": False,
        "feature_transform_rows": [0, config["conditioning"]],
        "fixed_prior": "all_coefficients_global_ridge" if config["prior"]["kind"] == "ridge"
            else "finite_slab_RHS_with_declared_proper_unshrunk_groups",
        "interrupted_fit_policy": "exact_chain_RNG_and_phase_resume_same_source_runtime",
        "recurrent_diagnostics": getattr(prepared.features.core, "recurrent_diagnostics", None),
        "fit_wall_seconds": time.perf_counter()-started,
        "input_identity": input_identity, "fit_identity": identity,
        "supplied_training_rows": len(training_panel.y),
        "supplied_training_sites": training_panel.y.shape[1],
        "response_transform": "identity_on_delivered_KS_training_only_scale",
        "coordinate_policy": "unit_circle_embedding_and_valid_Euclidean_Matern_restriction"}
    fitted = FullFieldKSFit(core, config, seed, identity, input_identity, computation, diagnostics)
    if path is not None:
        fitted.save(path)
        fitted._attach_parameter_cache(path)
    return fitted
