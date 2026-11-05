"""The delivered KS data contract, with explicit model/evaluator separation.

``cache_root`` is the verified restoration destination, not its ``data/ks``
leaf.  The delivered ``dynamics`` and ``benchprep`` readers remain authoritative
and are imported from that root.  No generator is called and no prepared file
is changed.  The native facade reads future labels internally; this adapter
provides *output separation*, not a claim of future-free disk access.

The public model objects contain only permitted observation histories.  In a
strict spatial protocol the delivered 64-site mask is checked before selecting
the 39 (interleaved) or 40 (contiguous) training columns.  Their zeros at hidden
sites never enter the model as observations.  Euclidean spatial APIs receive a
unit-circle embedding; physical periodic coordinates remain available for
decoders and plots.  The full-training scalar transform is used without refit.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib
import json
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..data import SpatioTemporalData, readonly


KS_CONFIGURATION_SHA256 = "5f82a0e0e44a62925167bf2f55fb44ab7be086c3074f7325e76748288e9a0332"
KS_DATA_SHA256 = "898f18deccc1fe6b3fb4d6eb0aa847244dc115efc56a260aceaae07bc22c227c"
KS_SOURCE_SHA256 = "c88f8475b97a9b7159e643e06e424d0f58630400182957602e48b91097422965"
PERIOD = 22.0
SAVED_INTERVAL = 0.25
PARTITIONS = {"train": 0, "validation": 1, "test": 2}
IO_LIMITATION = (
    "The delivered DynamicsDataset.batch constructs histories and future labels "
    "internally. Only its model section is exposed to forecasting code; this is "
    "output separation, not future-free disk access."
)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def circle_embedding(x: np.ndarray, period: float = PERIOD) -> np.ndarray:
    """Map physical periodic coordinates to a unit circle in Euclidean R².

    A Euclidean Matérn evaluated on these points is a restriction of a valid
    covariance in R².  Its range parameter is in unit-circle chord units, not
    physical arc length.  This does not use wrapped distance in a Matérn.
    """
    values = np.asarray(x, dtype=np.float64)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Periodic coordinates must be a finite one-dimensional vector")
    if not np.isfinite(period) or period <= 0:
        raise ValueError("The period must be positive and finite")
    angle = (2 * np.pi / period) * np.mod(values, period)
    return np.column_stack((np.cos(angle), np.sin(angle)))


def _load_readers(cache_root: Path):
    """Use the restored source, refusing a conflicting imported reader copy."""
    names = ("dynamics.adapters", "benchprep.batches")
    for relative in ("dynamics/adapters.py", "benchprep/batches.py"):
        if not (cache_root / relative).is_file():
            raise FileNotFoundError(f"Restore the v2 source package first: {cache_root / relative}")
    root = str(cache_root)
    sys.path.insert(0, root)
    try:
        modules = tuple(importlib.import_module(name) for name in names)
    finally:
        sys.path.remove(root)
    for name, module in zip(names, modules):
        actual = Path(module.__file__).resolve()
        expected = (cache_root / (name.replace(".", "/") + ".py")).resolve()
        if actual != expected:
            raise RuntimeError(f"Conflicting delivered reader {name}: {actual}; expected {expected}. "
                               "Start a new process when switching restored source roots.")
    return modules


@dataclass(frozen=True)
class ScalarTransform:
    """An exact delivered, per-trajectory/per-protocol full-training transform."""

    mean: float
    scale: float
    fit_time_bounds: tuple[int, int]
    fit_site_indices: tuple[int, ...]
    training_value_count: int

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ScalarTransform":
        result = cls(float(record["mean"]), float(record["scale"]),
                     tuple(map(int, record["fit_time_bounds"])),
                     tuple(map(int, record["fit_site_indices"])),
                     int(record["training_value_count"]))
        if not np.isfinite(result.mean) or not np.isfinite(result.scale) or result.scale <= 0:
            raise ValueError("The delivered transform must have finite mean and positive scale")
        return result

    @property
    def center(self) -> float:
        return self.mean

    def forward(self, physical: np.ndarray) -> np.ndarray:
        return (np.asarray(physical, dtype=np.float64) - self.mean) / self.scale

    def inverse(self, standardized: np.ndarray) -> np.ndarray:
        return np.asarray(standardized, dtype=np.float64) * self.scale + self.mean

    def record(self) -> dict[str, Any]:
        return {"mean": self.mean, "scale": self.scale,
                "fit_time_bounds": list(self.fit_time_bounds),
                "fit_site_indices": list(self.fit_site_indices),
                "training_value_count": self.training_value_count}


@dataclass(frozen=True)
class KSTraining:
    panel: SpatioTemporalData
    x: np.ndarray
    site_ids: np.ndarray
    transform: ScalarTransform
    identity: Mapping[str, Any]

    @property
    def values(self) -> np.ndarray:
        return self.panel.y

    @property
    def physical(self) -> np.ndarray:
        return self.transform.inverse(self.panel.y)

    @property
    def coordinates(self) -> np.ndarray:
        return self.panel.coordinates


@dataclass(frozen=True)
class KSModelWindow:
    """A causal permitted-site history plus static forecast query metadata."""

    panel: SpatioTemporalData
    x: np.ndarray
    site_ids: np.ndarray
    target_x: np.ndarray
    target_site_ids: np.ndarray
    target_coordinates: np.ndarray
    target_time: np.ndarray
    first_target: int
    origin: int
    history: int
    horizon: int
    transform: ScalarTransform
    identity: Mapping[str, Any]

    @property
    def values(self) -> np.ndarray:
        return self.panel.y

    @property
    def physical(self) -> np.ndarray:
        return self.transform.inverse(self.panel.y)

    @property
    def coordinates(self) -> np.ndarray:
        return self.panel.coordinates


@dataclass(frozen=True)
class KSEvaluationWindow:
    """Evaluator-only physical clean-field values and official eligibility."""

    values: np.ndarray
    mask: np.ndarray
    x: np.ndarray
    site_ids: np.ndarray
    coordinates: np.ndarray
    time: np.ndarray
    first_target: int
    transform: ScalarTransform

    @property
    def latent_clean(self) -> np.ndarray:
        return self.values


class KSAdapter:
    """Per-trajectory chronological KS application reader.

    Track A uses ``interleaved`` and Track B uses ``all_observed``.  Contiguous
    is a separately reported robustness protocol.  Pooled whole-trajectory
    training and noisy-only transforms are intentionally outside this API.
    ``training`` always returns the full permitted training interval; a smaller
    training experiment must implement and version its own subset transform.
    """

    def __init__(self, cache_root: str | Path, spatial_protocol: str = "interleaved",
                 phase: str = "core"):
        if phase not in ("core", "pilot"):
            raise ValueError("phase must be core or pilot")
        if spatial_protocol not in ("interleaved", "all_observed", "contiguous"):
            raise ValueError("spatial_protocol must be interleaved, all_observed, or contiguous")
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.data_root = self.cache_root / "data"
        self.folder = self.data_root / "ks"
        self.phase = phase
        self.spatial_protocol = spatial_protocol
        self.native_protocol = "none" if spatial_protocol == "all_observed" else spatial_protocol
        self.configuration = json.loads((self.folder / "manifests/configuration.json").read_text())
        if self.configuration["configuration_sha256"] != KS_CONFIGURATION_SHA256:
            raise ValueError("This application requires the exact delivered KS configuration")
        required = {"benchmark": "ks", "grid_size": 128, "observed_sites": 64,
                    "length": PERIOD, "saved_interval": SAVED_INTERVAL}
        if any(self.configuration.get(key) != value for key, value in required.items()):
            raise ValueError("The delivered KS schema/geometry differs from the frozen application")
        self.native, self.facade = _load_readers(self.cache_root)
        self.reader = self.native.DynamicsDataset(
            self.data_root, "ks", phase=phase, temporal_protocol="chronological",
            spatial_protocol=self.native_protocol)
        self.splits = self.reader.splits
        self.n_states = int(self.reader.n)
        self.trajectory_ids = tuple(sorted(map(int, self.reader.files)))
        self.train_bounds = tuple(map(int, self.splits["chronological"]["train"]))
        self.partition = readonly(self.reader.partition, dtype=np.int64)
        self.input_sites = readonly(np.flatnonzero(self.partition == 0), dtype=np.int64)
        expected_counts = {"interleaved": (39, 13, 12), "contiguous": (40, 12, 12),
                           "all_observed": (64, 0, 0)}[spatial_protocol]
        counts = tuple(int(np.count_nonzero(self.partition == i)) for i in range(3))
        if counts != expected_counts:
            raise ValueError(f"Unexpected frozen site counts: {counts}, expected {expected_counts}")
        expected_bounds = {"core": {"train": [0, 6000], "validation": [6000, 8000],
                                    "test": [8000, 10000]},
                           "pilot": {"train": [0, 1200], "validation": [1200, 1600],
                                     "test": [1600, 2000]}}[phase]
        if self.splits["chronological"] != expected_bounds:
            raise ValueError("Chronological target intervals differ from the delivered contract")
        expected_ids = tuple(range(20 if phase == "core" else 3))
        if self.trajectory_ids != expected_ids:
            raise ValueError("The complete declared trajectory manifest is required")
        self.x_all = readonly(np.arange(64, dtype=np.float64) * PERIOD / 64)
        self.coordinates_all = readonly(circle_embedding(self.x_all))
        self.metadata_hashes = {
            "split_sha256": _file_hash(self.folder / f"splits/{phase}.json"),
            "transform_file_sha256": _file_hash(self.folder / f"transforms/{phase}.json"),
            "native_reader_sha256": _file_hash(self.cache_root / "dynamics/adapters.py"),
            "facade_sha256": _file_hash(self.cache_root / "benchprep/batches.py"),
        }

    def _trajectory(self, trajectory: int) -> int:
        trajectory = _integer(trajectory, "trajectory")
        if trajectory not in self.reader.files:
            raise ValueError(f"Unknown {self.phase} trajectory {trajectory}")
        return trajectory

    def transform(self, trajectory: int) -> ScalarTransform:
        trajectory = self._trajectory(trajectory)
        result = ScalarTransform.from_record(
            self.reader.transforms["chronological"][str(trajectory)][self.native_protocol])
        if result.fit_time_bounds != self.train_bounds:
            raise ValueError("Transform fitting times do not match this training interval")
        if result.fit_site_indices != tuple(map(int, self.input_sites)):
            raise ValueError("Transform fitting sites do not match the permitted inputs")
        if result.training_value_count != (self.train_bounds[1] - self.train_bounds[0]) * len(self.input_sites):
            raise ValueError("Unexpected transform fitting sample count")
        return result

    def identity(self, trajectory: int) -> dict[str, Any]:
        trajectory = self._trajectory(trajectory)
        identity = {"application_data_schema": "ks-per-trajectory/1",
                    "configuration_sha256": KS_CONFIGURATION_SHA256,
                    "data_archive_sha256": KS_DATA_SHA256,
                    "source_archive_sha256": KS_SOURCE_SHA256,
                    "phase": self.phase, "trajectory": trajectory,
                    "temporal_protocol": "chronological",
                    "spatial_protocol": self.spatial_protocol,
                    "raw_trajectory_sha256": self.reader.files[trajectory]["sha256"],
                    "noise_fraction": 0.0, "input_mask": "none",
                    "transform": self.transform(trajectory).record(),
                    **self.metadata_hashes}
        identity["identity_sha256"] = _json_hash(identity)
        return identity

    def target_sites(self, partition: str) -> np.ndarray:
        if partition not in PARTITIONS:
            raise ValueError("partition must be train, validation, or test")
        if self.native_protocol == "none":
            return readonly(np.arange(64), dtype=np.int64)
        return readonly(np.flatnonzero(self.partition == PARTITIONS[partition]), dtype=np.int64)

    def starts(self, partition: str, horizon: int = 100, stride: int = 100,
               history: int = 128) -> np.ndarray:
        """Return first-target indices; adapter origins are these values minus one."""
        if partition not in PARTITIONS:
            raise ValueError("partition must be train, validation, or test")
        horizon = _integer(horizon, "horizon", 1)
        history = _integer(history, "history", 1)
        stride = _integer(stride, "stride", 1)
        lo, hi = self.splits["chronological"][partition]
        return np.arange(max(lo, history), hi - horizon + 1, stride, dtype=np.int64)

    def _window(self, trajectory: int, first_target: int, partition: str,
                history: int, horizon: int) -> tuple[int, int, int, int]:
        trajectory = self._trajectory(trajectory)
        first_target = _integer(first_target, "first_target", 1)
        history = _integer(history, "history", 1)
        horizon = _integer(horizon, "horizon", 1)
        if partition not in PARTITIONS:
            raise ValueError("partition must be train, validation, or test")
        lo, hi = self.splits["chronological"][partition]
        if first_target < max(lo, history) or first_target + horizon > hi:
            raise ValueError("Target window crosses its partition or lacks the declared history")
        return trajectory, first_target, history, horizon

    def _panel(self, values: np.ndarray, times: np.ndarray, trajectory: int,
               time_bounds: tuple[int, int], purpose: str) -> SpatioTemporalData:
        return SpatioTemporalData(
            values, self.coordinates_all[self.input_sites], times=times,
            site_ids=tuple(map(str, self.input_sites)),
            response_units="delivered train-standardized KS field",
            coordinate_units="unit-circle embedding; Euclidean chord distance",
            metadata={"dataset": "KS", "phase": self.phase, "trajectory": trajectory,
                      "spatial_protocol": self.spatial_protocol, "purpose": purpose,
                      "ks_protocol": self.spatial_protocol,
                      "time_index_bounds": list(time_bounds),
                      "physical_period": PERIOD, "saved_interval": SAVED_INTERVAL,
                      "data_identity_sha256": self.identity(trajectory)["identity_sha256"],
                      "response_transform": self.transform(trajectory).record(),
                      "model_payload_has_future_labels": False})

    def training(self, trajectory: int) -> KSTraining:
        """Read only the permitted full-training interval and observed-site panel.

        ``read_observed`` is the delivered native reader.  Its 64 observation
        columns are selected immediately; the 128-point solver field is never
        an input to fitting, and neither hidden-site responses nor later times
        are exposed in the returned object.
        """
        import h5py

        trajectory = self._trajectory(trajectory)
        lo, hi = self.train_bounds
        source = self.folder / self.reader.files[trajectory]["path"]
        with h5py.File(source, "r") as handle:
            values = self.native.read_observed(handle, lo, hi)[:, self.input_sites]
            coordinates = np.asarray(handle["coordinates/observed"][:, 0], dtype=np.float64)
        if not np.array_equal(coordinates, self.x_all):
            raise ValueError("Stored observation coordinates do not match the frozen periodic grid")
        transform = self.transform(trajectory)
        panel = self._panel(transform.forward(values), np.arange(lo, hi) * SAVED_INTERVAL,
                            trajectory, (lo, hi), "full-training observations")
        return KSTraining(panel, readonly(self.x_all[self.input_sites]), self.input_sites,
                          transform, MappingProxyType(self.identity(trajectory)))

    def training_panel(self, trajectory: int) -> SpatioTemporalData:
        return self.training(trajectory).panel

    def model_window(self, trajectory: int, first_target: int, partition: str,
                     history: int = 128, horizon: int = 100) -> KSModelWindow:
        trajectory, a, history, horizon = self._window(
            trajectory, first_target, partition, history, horizon)
        packet = self.facade.load_model_batch(
            "ks", self.data_root, trajectory=trajectory, phase=self.phase,
            temporal_protocol="chronological", spatial_protocol=self.spatial_protocol,
            partition=partition, origin=a - 1, history=history, horizon=horizon,
            noise_fraction=0.0, input_mask="none")
        # Do not forward the envelope or opaque native metadata to any model.
        model = packet["model"]
        values = np.asarray(model["values"], dtype=np.float64)
        mask = np.asarray(model["input_mask"], dtype=bool)
        if values.shape != (history, 64, 1) or mask.shape != values.shape:
            raise ValueError("Unexpected native model history/mask shape")
        expected_mask = np.zeros(64, dtype=bool)
        expected_mask[self.input_sites] = True
        if not np.array_equal(mask[:, :, 0], np.broadcast_to(expected_mask, (history, 64))):
            raise ValueError("Native availability does not match the clean permitted-site protocol")
        targets = self.target_sites(partition)
        if not np.array_equal(model["target_site_ids"], targets):
            raise ValueError("Native forecast query sites differ from the frozen target partition")
        if not np.array_equal(np.asarray(model["coordinates"])[:, 0], self.x_all):
            raise ValueError("Native history coordinates differ from the frozen periodic grid")
        times = np.arange(a - history, a, dtype=np.float64) * SAVED_INTERVAL
        target_time = np.arange(a, a + horizon, dtype=np.float64) * SAVED_INTERVAL
        if not np.array_equal(model["time"], times) or not np.array_equal(model["target_time"], target_time):
            raise ValueError("Native history/target indexing differs from the declared first target")
        transform = self.transform(trajectory)
        if ScalarTransform.from_record(packet["metadata"]["transform"]) != transform:
            raise ValueError("Native facade changed the delivered full-training transform")
        panel = self._panel(values[:, self.input_sites, 0], times, trajectory,
                            (a - history, a), "origin history; " + IO_LIMITATION)
        return KSModelWindow(
            panel, readonly(self.x_all[self.input_sites]), self.input_sites,
            readonly(self.x_all[targets]), targets, readonly(self.coordinates_all[targets]),
            readonly(target_time), a, a - 1, history, horizon, transform,
            MappingProxyType(self.identity(trajectory)))

    def history_panel(self, trajectory: int, first_target: int, partition: str,
                      history: int = 128, horizon: int = 100) -> SpatioTemporalData:
        return self.model_window(trajectory, first_target, partition, history, horizon).panel

    def evaluation_window(self, trajectory: int, first_target: int, partition: str,
                          history: int = 128, horizon: int = 100) -> KSEvaluationWindow:
        trajectory, a, history, horizon = self._window(
            trajectory, first_target, partition, history, horizon)
        packet = self.facade.load_evaluation_batch(
            "ks", self.data_root, trajectory=trajectory, phase=self.phase,
            temporal_protocol="chronological", spatial_protocol=self.spatial_protocol,
            partition=partition, origin=a - 1, history=history, horizon=horizon,
            noise_fraction=0.0, input_mask="none")
        evaluation = packet["evaluation"]
        targets = self.target_sites(partition)
        values = np.asarray(evaluation["latent_clean"], dtype=np.float64)
        mask = np.asarray(evaluation["scoring_mask"], dtype=bool)
        if values.shape != (horizon, len(targets), 1) or mask.shape != values.shape:
            raise ValueError("Unexpected native evaluation shape")
        if not mask.all() or not np.isfinite(values).all():
            raise ValueError("The clean verified KS target must be fully finite and eligible")
        return KSEvaluationWindow(
            readonly(values[:, :, 0]), readonly(mask[:, :, 0], dtype=bool),
            readonly(self.x_all[targets]), targets, readonly(self.coordinates_all[targets]),
            readonly(np.arange(a, a + horizon, dtype=np.float64) * SAVED_INTERVAL),
            a, self.transform(trajectory))


def audit_cache(cache_root: str | Path) -> dict[str, Any]:
    """Verify restored bytes, metadata, external links and training transforms.

    This performs no generator tests or model fitting and does not read core
    validation/test response values.  Full file hashes necessarily cover their
    opaque bytes; schema/time/coordinate checks inspect metadata only.  Scalar
    transforms are independently recomputed on permitted training values.
    """
    import h5py

    root = Path(cache_root).expanduser().resolve()
    folder = root / "data/ks"
    receipt: dict[str, Any] = {"schema": "ks-cache-acceptance/1", "passed": False,
                               "configuration_sha256": KS_CONFIGURATION_SHA256,
                               "model_fit_performed": False,
                               "core_validation_test_response_values_inspected": False,
                               "facade_io_limitation": IO_LIMITATION,
                               "phases": {}, "derived_file_count": 0}
    initial_hashes = set()
    for phase in ("pilot", "core"):
        adapters = {protocol: KSAdapter(root, protocol, phase)
                    for protocol in ("interleaved", "all_observed", "contiguous")}
        adapter = adapters["interleaved"]
        derived = json.loads((folder / f"qc/{phase}_derived_integrity.json").read_text())
        for record in derived["files"]:
            path = folder / record["path"]
            if path.stat().st_size != record["bytes"] or _file_hash(path) != record["sha256"]:
                raise ValueError(f"Derived KS bytes/hash mismatch: {record['path']}")
            receipt["derived_file_count"] += 1
        phase_receipt = {"trajectory_count": len(adapter.trajectory_ids),
                         "states_per_trajectory": adapter.n_states,
                         "raw_files": [], "external_links_verified": 0,
                         "transforms_recomputed": 0,
                         "chronological_bounds": adapter.splits["chronological"],
                         "site_counts": {p: {k: len(a.target_sites(k)) for k in PARTITIONS}
                                         for p, a in adapters.items()},
                         "metadata_hashes": adapter.metadata_hashes}
        with h5py.File(folder / f"canonical/{phase}.h5", "r") as canonical:
            expected_names = {f"trajectory_{t:03d}" for t in adapter.trajectory_ids}
            if set(canonical.keys()) != expected_names:
                raise ValueError("Canonical trajectory links do not match the complete manifest")
            for trajectory in adapter.trajectory_ids:
                record = adapter.reader.files[trajectory]
                path = folder / record["path"]
                digest = _file_hash(path)
                if path.stat().st_size != record["bytes"] or digest != record["sha256"]:
                    raise ValueError(f"Raw trajectory bytes/hash mismatch: {record['path']}")
                if record["initial_state_sha256"] in initial_hashes:
                    raise ValueError("Pilot/core initial-state hashes are not distinct")
                initial_hashes.add(record["initial_state_sha256"])
                name = f"trajectory_{trajectory:03d}"
                link = canonical.get(name, getlink=True)
                if not isinstance(link, h5py.ExternalLink) or link.filename != "../" + record["path"] or link.path != "/":
                    raise ValueError(f"Incorrect relative HDF5 external link: {name}")
                group = canonical[name]
                if group["truth/field"].shape != (adapter.n_states, 128, 1):
                    raise ValueError("Incorrect solver truth schema")
                if group["truth/field"].dtype != np.dtype("float64"):
                    raise ValueError("Solver truth must remain float64")
                if group.attrs["configuration_sha256"] != KS_CONFIGURATION_SHA256:
                    raise ValueError("Raw trajectory configuration mismatch")
                if float(group.attrs["saved_interval"]) != SAVED_INTERVAL:
                    raise ValueError("Incorrect saved cadence")
                if not np.array_equal(group["time"][:], np.arange(adapter.n_states) * SAVED_INTERVAL):
                    raise ValueError("Time labels are not the declared zero-based saved cadence")
                if not np.array_equal(group["coordinates/observed"][:, 0], adapter.x_all):
                    raise ValueError("Incorrect observation-panel coordinates")
                if not np.array_equal(group["coordinates/observed_grid_indices"][:], np.arange(0, 128, 2)):
                    raise ValueError("Observation panel differs from every second solver grid point")
                if not np.allclose(group["coordinates/periodic_embedding"][:], adapter.coordinates_all,
                                   rtol=0, atol=2e-15):
                    raise ValueError("Stored circle embedding differs from periodic geometry")
                # Read only training response values. No test result is opened.
                train = adapter.native.read_observed(group, *adapter.train_bounds)
                for protocol, protocol_adapter in adapters.items():
                    transform = protocol_adapter.transform(trajectory)
                    permitted = train[:, protocol_adapter.input_sites]
                    if not np.isclose(permitted.mean(), transform.mean, rtol=2e-13, atol=2e-14):
                        raise ValueError(f"Training-only mean mismatch: {phase}/{trajectory}/{protocol}")
                    if not np.isclose(permitted.std(), transform.scale, rtol=2e-13, atol=2e-14):
                        raise ValueError(f"Training-only scale mismatch: {phase}/{trajectory}/{protocol}")
                    reconstructed = transform.inverse(transform.forward(permitted))
                    if not np.allclose(reconstructed, permitted, rtol=2e-15, atol=2e-15):
                        raise ValueError("Scalar inverse transform failed")
                    phase_receipt["transforms_recomputed"] += 1
                phase_receipt["external_links_verified"] += 1
                phase_receipt["raw_files"].append({"trajectory": trajectory, "bytes": path.stat().st_size,
                                                   "sha256": digest, "passed": True})
        receipt["phases"][phase] = phase_receipt
    receipt["distinct_initial_state_count"] = len(initial_hashes)
    receipt["passed"] = True
    return receipt
