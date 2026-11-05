"""Small data-only periodic residual CNN for the all-observed KS track.

The architecture is a custom PyTorch adaptation motivated by APEBench's
periodic convolutional controls (Koehler et al., 2024, arXiv:2411.00180,
Appendix H.1), not a reproduction of its dataset or training protocol.
An input 1x1 convolution, residual blocks with two circular convolutions,
and an output 1x1 convolution predict the next-state *increment*. Each
block is ``GELU(h + conv2(GELU(conv1(h))))``. There is no normalization,
PDE loss, derivative input, conservation correction, or forecast clipping.
The output layer starts at zero, hence the initial model is persistence.

Panels must explicitly declare ``metadata['ks_protocol']='all_observed'``
and contain all 64 ordered observation sites on a circle. The caller owns
the delivered training-only scalar transform and its inverse: predictions
are on exactly the input response scale. A forecast uses only the last
available state of the supplied causal history and then its own outputs.

Training minimizes mean one-step increment squared error. If official
validation histories/targets are supplied, autonomous forecasts select an
epoch by equal-weight lead-band MSE (1--10, 11--25, 26--50, 51--100).
Without validation, the configured epoch count is fixed. Final-study calls
must therefore use the frozen epoch count and omit validation. Validation
labels never enter gradient updates, normalization, or a rollout state.

Atomic numeric NPZ checkpoints retain Adam state, the current permutation
and batch cursor, both CPU/permutation RNG states, and early-stopping state.
Resume requires identical data, validation, effective configuration, seed,
and numerical environment; completed fits are returned without retraining.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import importlib
from pathlib import Path
import resource
import time

import numpy as np
from threadpoolctl import threadpool_limits

from ..benchmark.contracts import CapabilityError, Forecast, Panel
from ..io import content_hash, load_state, save_state, seed_for


MODEL_ID = "periodic_cnn"
SCHEMA_VERSION = 1
DEFAULTS = {
    "width": 16, "blocks": 2, "kernel_size": 5, "epochs": 30,
    "learning_rate": 1e-3, "batch_size": 128, "weight_decay": 1e-6,
    "gradient_clip": 1., "patience": 6, "min_delta": 1e-7,
    "validation_every": 1, "checkpoint_every_batches": 50,
}
LEAD_BANDS = ((0, 10), (10, 25), (25, 50), (50, 100))


def _torch():
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise ImportError("The optional periodic CNN requires the existing PyTorch environment") from exc


def _configuration(config):
    config = dict(config or {})
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown periodic_cnn configuration: {sorted(unknown)}")
    result = {**DEFAULTS, **config}
    for key in ("width", "blocks", "kernel_size", "epochs", "batch_size",
                "patience", "validation_every", "checkpoint_every_batches"):
        value = result[key]
        minimum = 0 if key == "checkpoint_every_batches" else 1
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
        result[key] = int(value)
    if result["width"] > 64 or result["blocks"] > 4:
        raise ValueError("This compact comparator supports width <=64 and at most four blocks")
    if result["kernel_size"] % 2 != 1 or result["kernel_size"] > 9:
        raise ValueError("kernel_size must be odd and at most nine")
    for key in ("learning_rate", "weight_decay", "gradient_clip", "min_delta"):
        result[key] = float(result[key])
        if not np.isfinite(result[key]) or result[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if result["learning_rate"] == 0 or result["gradient_clip"] == 0:
        raise ValueError("learning_rate and gradient_clip must be positive")
    return result


def _domain(panel):
    if not isinstance(panel, Panel):
        raise TypeError("periodic_cnn accepts an observation Panel, not a truth/facade object")
    if panel.metadata.get("ks_protocol") != "all_observed":
        raise CapabilityError("periodic_cnn requires an explicit ks_protocol=all_observed panel")
    if panel.metadata.get("input_mask", "none") not in (None, "none"):
        raise CapabilityError("The clean periodic CNN does not support masked inputs")
    for name in ("observed_mask", "input_site_mask"):
        if name in panel.metadata and not np.asarray(panel.metadata[name], bool).all():
            raise CapabilityError("All 64 sites must be observed; hidden zero fills are not inputs")
    coords = panel.coordinates
    if coords.shape != (64, 2):
        raise CapabilityError("periodic_cnn supports only the full 64-site circle embedding")
    centered = coords - coords.mean(axis=0)
    radius = np.linalg.norm(centered, axis=1)
    if radius.mean() <= 0 or not np.allclose(radius, radius.mean(), rtol=1e-6, atol=1e-10):
        raise ValueError("Input coordinates must lie on one circle")
    angles = np.arctan2(centered[:, 1], centered[:, 0])
    step = (np.roll(angles, -1) - angles + np.pi) % (2*np.pi) - np.pi
    expected = np.copysign(2*np.pi/64, step[0])
    if not np.allclose(step, expected, rtol=1e-6, atol=1e-10):
        raise ValueError("Circular convolutions require a uniformly ordered complete panel")


def _panel_identity(panel):
    return {"y": panel.y, "coordinates": panel.coordinates, "times": panel.times,
            "site_ids": panel.site_ids, "response_units": panel.response_units,
            "coordinate_units": panel.coordinate_units, "metadata": dict(panel.metadata)}


@contextmanager
def _cpu_runtime(torch):
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with threadpool_limits(limits=1):
            yield
    finally:
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
        torch.set_num_threads(old_threads)


def _network(torch, config):
    width, kernel = config["width"], config["kernel_size"]

    class PeriodicBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            args = dict(kernel_size=kernel, padding=kernel//2,
                        padding_mode="circular", dtype=torch.float32, device="cpu")
            self.conv1 = torch.nn.Conv1d(width, width, **args)
            self.conv2 = torch.nn.Conv1d(width, width, **args)
            self.activation = torch.nn.GELU()

        def forward(self, value):
            return self.activation(value + self.conv2(self.activation(self.conv1(value))))

    class PeriodicIncrementCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lift = torch.nn.Conv1d(1, width, 1, dtype=torch.float32, device="cpu")
            self.blocks = torch.nn.ModuleList([PeriodicBlock() for _ in range(config["blocks"])])
            self.head = torch.nn.Conv1d(width, 1, 1, dtype=torch.float32, device="cpu")
            torch.nn.init.zeros_(self.head.weight)
            torch.nn.init.zeros_(self.head.bias)

        def forward(self, value):
            state = self.lift(value)
            for block in self.blocks:
                state = block(state)
            return self.head(state)

    return PeriodicIncrementCNN()


def _numpy_tree(value, torch):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, dict):
        return {str(key): _numpy_tree(item, torch) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_numpy_tree(item, torch) for item in value)
    if isinstance(value, list):
        return [_numpy_tree(item, torch) for item in value]
    return value


def _tensor_tree(value, torch):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, dict):
        return {key: _tensor_tree(item, torch) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tensor_tree(item, torch) for item in value)
    if isinstance(value, list):
        return [_tensor_tree(item, torch) for item in value]
    return value


def _rollout(network, last_states, horizon, torch, *, dtype=None):
    """No observations or labels enter after the supplied initial states."""
    dtype = torch.float32 if dtype is None else dtype
    state = torch.tensor(np.array(last_states, copy=True), dtype=dtype)[:, None, :]
    predictions = np.empty((len(state), horizon, 64), dtype=np.float64)
    network.eval()
    with torch.no_grad():
        for lead in range(horizon):
            state = state + network(state)
            if not bool(torch.isfinite(state).all()):
                raise FloatingPointError(f"periodic_cnn autonomous rollout is nonfinite at lead {lead+1}")
            predictions[:, lead] = state[:, 0].cpu().numpy()
    return predictions


def _validation(training, validation):
    if validation is None:
        return None, None, None
    if not isinstance(validation, (tuple, list)) or len(validation) != 2:
        raise ValueError("validation must be (causal history Panels, target arrays)")
    histories, targets = validation
    histories = [histories] if isinstance(histories, Panel) else list(histories)
    if not histories:
        raise ValueError("Validation requires at least one origin")
    for panel in histories:
        _domain(panel)
        if panel.site_ids != training.site_ids or not np.array_equal(panel.coordinates, training.coordinates):
            raise ValueError("Validation histories must have the fitted site order and geometry")
        if panel.times[-1] < training.times[-1]:
            raise ValueError("Validation origins cannot precede the end of parameter fitting")
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim == 2 and len(histories) == 1:
        targets = targets[None]
    if (targets.ndim != 3 or targets.shape[0] != len(histories)
            or targets.shape[2] != 64 or not 1 <= targets.shape[1] <= 100
            or not np.isfinite(targets).all()):
        raise ValueError("Validation targets must be finite [origin,1..100,64]")
    signature = content_hash({"histories": [_panel_identity(panel) for panel in histories],
                              "targets": targets})
    return np.stack([panel.y[-1] for panel in histories]), targets.copy(), signature


def _validation_loss(network, last_states, targets, torch):
    try:
        predicted = _rollout(network, last_states, targets.shape[1], torch)
    except FloatingPointError:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        losses = np.mean((predicted - targets)**2, axis=2, dtype=np.float64)
        bands = [np.mean(losses[:, left:min(right, losses.shape[1])], axis=1)
                 for left, right in LEAD_BANDS if left < losses.shape[1]]
        result = float(np.mean(np.stack(bands), dtype=np.float64))
    return result if np.isfinite(result) else None


class PeriodicCNNFit:
    """Frozen deterministic CNN; coordinate queries cannot create new sites."""
    model_id = MODEL_ID

    def __init__(self, network, config, seed, domain, diagnostics, identity):
        self.network, self.config, self.seed = network, dict(config), int(seed)
        self.domain = copy.deepcopy(domain)
        self.coordinates = np.array(domain["coordinates"], dtype=float, copy=True)
        self.site_ids = tuple(domain["site_ids"])
        self.diagnostics = copy.deepcopy(diagnostics)
        self.identity = str(identity)

    def _history(self, history_panel):
        _domain(history_panel)
        if (history_panel.site_ids != self.site_ids
                or not np.array_equal(history_panel.coordinates, self.coordinates)):
            raise ValueError("History must preserve all fitted site IDs and their circular order")
        if history_panel.times[-1] < self.domain["last_training_time"]:
            raise ValueError("Forecast origin precedes the parameter-fitting boundary")
        if len(history_panel.times) > 1 and not np.isclose(
                np.diff(history_panel.times)[0], self.domain["time_step"], rtol=1e-10, atol=0):
            raise ValueError("History cadence differs from the fitted cadence")
        return history_panel.y[-1:]

    def _targets(self, target_coordinates, target_site_ids):
        if target_coordinates is None and target_site_ids is None:
            return np.arange(64), self.coordinates, self.site_ids
        by_id = {name: index for index, name in enumerate(self.site_ids)}
        if target_site_ids is not None:
            ids = tuple(map(str, target_site_ids))
            if not ids or len(set(ids)) != len(ids) or any(name not in by_id for name in ids):
                raise CapabilityError("periodic_cnn predicts only trained all-observed site IDs")
            indices = np.array([by_id[name] for name in ids], dtype=int)
            coords = self.coordinates[indices]
            if target_coordinates is not None and not np.array_equal(np.asarray(target_coordinates), coords):
                raise CapabilityError("Target IDs and trained coordinates disagree")
            return indices, coords, ids
        coords = np.asarray(target_coordinates, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2 or not len(coords):
            raise ValueError("Target coordinates must be a nonempty [site,2] array")
        indices = []
        for point in coords:
            match = np.flatnonzero(np.all(self.coordinates == point, axis=1))
            if len(match) != 1:
                raise CapabilityError("Unseen-coordinate prediction is unsupported by the periodic CNN")
            indices.append(int(match[0]))
        if len(set(indices)) != len(indices):
            raise ValueError("Target sites must be unique")
        return np.asarray(indices), coords, tuple(self.site_ids[index] for index in indices)

    def forecast(self, history_panel, horizon, target_coordinates=None,
                 target_site_ids=None, n_paths=0, seed=0):
        """Autonomous point forecast; ``seed`` cannot create predictive samples."""
        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if n_paths != 0:
            raise CapabilityError("A deterministic CNN has no native predictive paths or intervals")
        last = self._history(history_panel)
        indices, coords, ids = self._targets(target_coordinates, target_site_ids)
        torch = _torch()
        started = time.perf_counter()
        with _cpu_runtime(torch):
            mean = _rollout(self.network, last, int(horizon), torch)[0][:, indices]
        return Forecast(mean, None, coords, ids, {
            "model_id": self.model_id, "ks_protocol": "all_observed",
            "forecast_mechanism": "fixed_weights_autonomous_increment_recursion",
            "history_rows_available": len(history_panel.y), "history_rows_used": 1,
            "native_distribution": False, "forecast_randomness": "none",
            "input_output_scale": "caller_provided_training_only_transform",
            "response_units": self.domain["response_units"],
            "forecast_seconds": time.perf_counter() - started,
            "dtype": "float32_network_and_recursion_float64_return",
        })

    def precision_check(self, history_panel, horizon=10):
        """Compare the same frozen weights/initial state in float32 and float64."""
        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        last = self._history(history_panel)
        torch = _torch()
        with _cpu_runtime(torch):
            single = _rollout(self.network, last, int(horizon), torch)
            double = _rollout(copy.deepcopy(self.network).double(), last, int(horizon), torch,
                              dtype=torch.float64)
        difference = single - double
        scale = max(float(np.sqrt(np.mean(double**2))), np.finfo(float).tiny)
        return {"horizon": int(horizon), "max_abs_difference": float(np.max(np.abs(difference))),
                "rms_difference": float(np.sqrt(np.mean(difference**2))),
                "relative_rms_difference": float(np.sqrt(np.mean(difference**2))/scale),
                "comparison": "identical_float32_weights_promoted_to_float64_no_refit"}

    def to_state(self):
        torch = _torch()
        return {"schema_version": SCHEMA_VERSION, "model_id": self.model_id,
                "config": self.config, "seed": self.seed, "identity": self.identity,
                "domain": self.domain, "diagnostics": self.diagnostics,
                "weights": _numpy_tree(self.network.state_dict(), torch)}

    def save(self, path):
        return save_state(path, self.to_state(), kind="ks_periodic_cnn_fit")

    @classmethod
    def from_state(cls, state):
        if state.get("schema_version") != SCHEMA_VERSION or state.get("model_id") != MODEL_ID:
            raise ValueError("Incompatible periodic CNN fitted-state schema")
        config = _configuration(state["config"])
        torch = _torch()
        with torch.random.fork_rng(devices=[]):
            network = _network(torch, config)
        network.load_state_dict(_tensor_tree(state["weights"], torch), strict=True)
        if not all(bool(torch.isfinite(value).all()) for value in network.parameters()):
            raise FloatingPointError("Stored periodic CNN weights are nonfinite")
        network.eval()
        return cls(network, config, state["seed"], state["domain"],
                   state["diagnostics"], state["identity"])


def fit_cnn(training_panel, config=None, seed=0, checkpoint_path=None, validation=None):
    """Fit all permitted transitions, optionally select an epoch on validation.

    Existing matching checkpoints resume automatically, including mid-epoch.
    The saved epoch budget is part of the identity; changing it is a new fit.
    Checkpoint cadence alone can change without changing the fitted model.
    """
    _domain(training_panel)
    if len(training_panel.y) < 3:
        raise ValueError("At least three permitted training states are required")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    config = _configuration(config)
    last_states, targets, validation_signature = _validation(training_panel, validation)
    torch = _torch()
    effective = {key: value for key, value in config.items() if key != "checkpoint_every_batches"}
    if validation is None:
        for key in ("patience", "min_delta", "validation_every"):
            effective[key] = None
    identity = content_hash({"schema": SCHEMA_VERSION, "model": MODEL_ID, "config": effective,
                             "seed": int(seed), "training": _panel_identity(training_panel),
                             "validation_signature": validation_signature,
                             "torch": str(torch.__version__), "numpy": np.__version__,
                             "threads": 1, "deterministic_algorithms": True})
    path = None if checkpoint_path is None else Path(checkpoint_path)
    state = load_state(path, kind="ks_periodic_cnn_checkpoint") if path is not None and path.exists() else None
    if state is not None:
        if (state.get("schema_version") != SCHEMA_VERSION or state.get("identity") != identity
                or state.get("status") not in ("running", "complete")):
            raise ValueError("CNN checkpoint data/configuration/validation/seed/environment identity mismatch")
        if state["status"] == "complete":
            return PeriodicCNNFit.from_state(state["fit_state"])

    started = time.perf_counter()
    domain = {"coordinates": training_panel.coordinates, "site_ids": training_panel.site_ids,
              "last_training_time": float(training_panel.times[-1]),
              "time_step": float(training_panel.times[1] - training_panel.times[0]),
              "response_units": training_panel.response_units,
              "coordinate_units": training_panel.coordinate_units,
              "training_rows": len(training_panel.y), "ks_protocol": "all_observed"}
    inputs = torch.tensor(training_panel.y[:-1].copy(), dtype=torch.float32)[:, None, :]
    increments = torch.tensor(np.diff(training_panel.y, axis=0).copy(), dtype=torch.float32)[:, None, :]
    init_seed = seed_for(seed, MODEL_ID, "initialization") % (2**63-1)
    order_seed = seed_for(seed, MODEL_ID, "optimization") % (2**63-1)
    with _cpu_runtime(torch), torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        network = _network(torch, config)
        optimizer = torch.optim.AdamW(network.parameters(), lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"])
        generator = torch.Generator(device="cpu").manual_seed(order_seed)
        if state is None:
            state = {"schema_version": SCHEMA_VERSION, "identity": identity, "status": "running",
                     "epoch": 1, "permutation": None, "cursor": 0, "epoch_loss_sum": 0.,
                     "updates": 0, "records": [], "best_loss": None, "best_epoch": 0,
                     "best_weights": None, "stale": 0, "stopped": False, "elapsed_seconds": 0.}
        else:
            network.load_state_dict(_tensor_tree(state["network"], torch), strict=True)
            restored_optimizer = _tensor_tree(state["optimizer"], torch)
            restored_optimizer["state"] = {int(key): value for key, value in restored_optimizer["state"].items()}
            optimizer.load_state_dict(restored_optimizer)
            generator.set_state(torch.from_numpy(state["permutation_rng"].copy()))
            torch.random.set_rng_state(torch.from_numpy(state["torch_rng"].copy()))
        previous_seconds = state["elapsed_seconds"]

        def checkpoint():
            state["network"] = _numpy_tree(network.state_dict(), torch)
            state["optimizer"] = _numpy_tree(optimizer.state_dict(), torch)
            state["permutation_rng"] = generator.get_state().numpy().copy()
            state["torch_rng"] = torch.random.get_rng_state().numpy().copy()
            state["elapsed_seconds"] = previous_seconds + time.perf_counter() - started
            if path is not None:
                save_state(path, state, kind="ks_periodic_cnn_checkpoint")

        while state["epoch"] <= config["epochs"] and not state["stopped"]:
            epoch = state["epoch"]
            if state["permutation"] is None:
                state["permutation"] = torch.randperm(len(inputs), generator=generator).numpy().copy()
                state["cursor"] = 0
                state["epoch_loss_sum"] = 0.
            network.train()
            while state["cursor"] < len(inputs):
                stop = min(state["cursor"] + config["batch_size"], len(inputs))
                batch = torch.from_numpy(state["permutation"][state["cursor"]:stop].copy())
                optimizer.zero_grad(set_to_none=True)
                loss = torch.mean((network(inputs[batch]) - increments[batch])**2)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Nonfinite periodic CNN training loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), config["gradient_clip"],
                                               error_if_nonfinite=True)
                optimizer.step()
                if not all(bool(torch.isfinite(value).all()) for value in network.parameters()):
                    raise FloatingPointError(f"Nonfinite periodic CNN parameters at epoch {epoch}")
                state["epoch_loss_sum"] += float(loss.detach()) * len(batch)
                state["cursor"] = stop
                state["updates"] += 1
                if config["checkpoint_every_batches"] and state["updates"] % config["checkpoint_every_batches"] == 0:
                    checkpoint()
            record = {"epoch": epoch, "training_increment_mse": state["epoch_loss_sum"]/len(inputs)}
            if validation is not None and (epoch % config["validation_every"] == 0 or epoch == config["epochs"]):
                loss = _validation_loss(network, last_states, targets, torch)
                record.update(validation_band_mse=loss, validation_finite=loss is not None)
                if loss is not None and (state["best_loss"] is None or loss < state["best_loss"]-config["min_delta"]):
                    state.update(best_loss=loss, best_epoch=epoch, stale=0,
                                 best_weights=_numpy_tree(network.state_dict(), torch))
                else:
                    state["stale"] += 1
                state["stopped"] = state["stale"] >= config["patience"]
            elif validation is None:
                state["best_epoch"] = epoch
            state["records"].append(record)
            state["epoch"] += 1
            state["permutation"] = None
            state["cursor"] = 0
            state["epoch_loss_sum"] = 0.
            checkpoint()

        if validation is not None and state["best_weights"] is None:
            raise FloatingPointError("No finite autonomous validation forecast within the CNN epoch budget")
        # Capture the last optimization state before restoring the selected epoch.
        checkpoint()
        if validation is not None:
            network.load_state_dict(_tensor_tree(state["best_weights"], torch), strict=True)
        network.eval()
        diagnostics = {
            "model_id": MODEL_ID, "qualified": True, "finite_fit": True,
            "inference": "AdamW_point_estimate", "inference_qualification": "not_applicable_point_model",
            "native_distribution": False, "parameter_uncertainty": False,
            "neural_optimizer_convergence": "not_claimed",
            "neural_stop_reason": "patience" if state["stopped"] else "epoch_budget",
            "epochs_completed": len(state["records"]), "selected_epoch": state["best_epoch"],
            "optimization_updates": state["updates"], "history": state["records"],
            "training_rows": len(training_panel.y), "usable_training_targets": len(inputs),
            "training_target_interval_local": [1, len(training_panel.y)],
            "gradient_training_loss": "mean_one_step_increment_squared_error",
            "selection_loss": "autonomous_equal_lead_band_MSE" if validation is not None else "fixed_epoch_count",
            "selection_bands_half_open": [list(band) for band in LEAD_BANDS],
            "validation_origins": 0 if validation is None else len(targets),
            "validation_horizon": None if validation is None else targets.shape[1],
            "best_validation_loss": state["best_loss"], "validation_signature": validation_signature,
            "history_rows_used": 1, "ks_protocol": "all_observed",
            "preprocessing": "none; supplied training-only scalar transform retained",
            "architecture": "1x1_lift; GELU(h+conv2(GELU(conv1(h)))); 1x1_increment_head",
            "padding": "circular", "output_initialization": "zero_increment",
            "parameter_count": sum(parameter.numel() for parameter in network.parameters()),
            "torch_version": str(torch.__version__), "device": "cpu", "torch_threads": 1,
            "torch_interop_threads": int(torch.get_num_interop_threads()),
            "dtype": "float32_training_float64_metrics_and_return",
            "training_cast_max_abs_error": float(np.max(np.abs(training_panel.y.astype(np.float32).astype(float)-training_panel.y))),
            "fit_seconds": previous_seconds + time.perf_counter() - started,
            "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
            "research_status": "custom_APEBench_motivated_KS_adaptation_not_paper_reproduction",
        }
        fitted = PeriodicCNNFit(network, config, seed, domain, diagnostics, identity)
        state["fit_state"] = fitted.to_state()
        state["status"] = "complete"
        state["elapsed_seconds"] = diagnostics["fit_seconds"]
        if path is not None:
            save_state(path, state, kind="ks_periodic_cnn_checkpoint")
    return fitted


def cnn_from_state(state):
    return PeriodicCNNFit.from_state(state)


def load_cnn(path):
    return cnn_from_state(load_state(path, kind="ks_periodic_cnn_fit"))
