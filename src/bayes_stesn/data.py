"""Immutable observed data and explicit, half-open chronological splits.

The fitting API deliberately has no field for a simulator's latent truth.
Coordinates are Euclidean; geographic degrees must be projected beforehand.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping
import json
import numpy as np


def readonly(value, dtype=float):
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SpatioTemporalData:
    y: np.ndarray
    coordinates: np.ndarray
    times: np.ndarray | None = None
    site_ids: tuple[str, ...] | None = None
    response_units: str = "response"
    coordinate_units: str = "Euclidean"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        y, c = readonly(self.y), readonly(self.coordinates)
        if y.ndim != 2 or c.ndim != 2 or c.shape[0] != y.shape[1]:
            raise ValueError("y is [time,site]; coordinates must have one row per site")
        if min(y.shape) < 1 or c.shape[1] not in (1, 2, 3):
            raise ValueError("Require a nonempty scalar response and 1D, 2D or 3D coordinates")
        if not np.isfinite(y).all() or not np.isfinite(c).all():
            raise ValueError("Missing or nonfinite observations/coordinates are unsupported")
        if len(np.unique(c, axis=0)) != len(c):
            raise ValueError("Input coordinates must be unique; aggregate repeated measurements explicitly")
        if str(self.coordinate_units).lower() in {"degrees", "latlon", "latitude_longitude", "epsg:4326"}:
            raise ValueError("Project geographic degrees to an appropriate Euclidean coordinate system")
        times = np.arange(len(y), dtype=float) if self.times is None else np.asarray(self.times)
        if times.ndim != 1 or len(times) != len(y):
            raise ValueError("times must have one entry per observation row")
        if times.dtype.kind == "M":
            times = times.astype("datetime64[ns]").astype(np.int64)
        times = readonly(times)
        if not np.isfinite(times).all():
            raise ValueError("times must be finite numeric identifiers or datetime64")
        if len(times) > 1:
            differences = np.diff(times)
            if differences[0] <= 0 or not np.allclose(differences, differences[0], rtol=1e-10, atol=0):
                raise ValueError("times must be strictly increasing and equally spaced")
        ids = tuple(str(i) for i in range(y.shape[1])) if self.site_ids is None else tuple(map(str, self.site_ids))
        if len(ids) != y.shape[1] or len(set(ids)) != len(ids):
            raise ValueError("site_ids must be unique and match the observed columns")
        metadata = json.loads(json.dumps(dict(self.metadata), allow_nan=False))
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "coordinates", c)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "site_ids", ids)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    def historical(self, stop: int) -> "SpatioTemporalData":
        """Return observations through exclusive row ``stop`` with the same geometry."""
        if not 1 <= stop <= len(self.y):
            raise ValueError("Historical stop must be within available data")
        return SpatioTemporalData(self.y[:stop], self.coordinates, self.times[:stop],
                                  self.site_ids, self.response_units, self.coordinate_units,
                                  dict(self.metadata))


@dataclass(frozen=True)
class SplitPlan:
    """All boundaries are row indices, half-open [start, stop).

    Only [0, conditioning) may estimate feature transformations. The
    likelihood uses [fitting_start, fitting_stop). Other blocks are metadata
    that the study runner uses, and are never supplied to the likelihood.
    """
    conditioning: int
    fitting_stop: int
    fitting_start: int | None = None
    validation_blocks: tuple[tuple[int, int], ...] = ()
    calibration_blocks: tuple[tuple[int, int], ...] = ()
    final_blocks: tuple[tuple[int, int], ...] = ()

    def __post_init__(self):
        start = self.conditioning if self.fitting_start is None else int(self.fitting_start)
        if self.conditioning < 3 or start < self.conditioning or self.fitting_stop <= start:
            raise ValueError("Require 3 <= conditioning <= fitting_start < fitting_stop")
        object.__setattr__(self, "fitting_start", start)
        used = [(0, self.conditioning, "conditioning"), (start, self.fitting_stop, "fitting")]
        for name in ("validation_blocks", "calibration_blocks", "final_blocks"):
            blocks = tuple((int(a), int(b)) for a, b in getattr(self, name))
            for a, b in blocks:
                if a < self.fitting_stop or b <= a:
                    raise ValueError(f"{name} must follow fitting and have positive length")
                used.append((a, b, name))
            object.__setattr__(self, name, blocks)
        used.sort()
        if any(left[1] > right[0] for left, right in zip(used, used[1:])):
            raise ValueError("Split blocks must not overlap")
        if self.validation_blocks and self.final_blocks:
            if max(b for _, b in self.validation_blocks) > min(a for a, _ in self.final_blocks):
                raise ValueError("Validation must precede the sealed final evaluation")
        if self.calibration_blocks and self.final_blocks:
            if max(b for _, b in self.calibration_blocks) > min(a for a, _ in self.final_blocks):
                raise ValueError("Calibration must precede final evaluation")


@dataclass(frozen=True)
class EvaluationTruth:
    """A separate scoring-only object. Never accepted by prepare or fit."""
    observation: np.ndarray
    generating_mean: np.ndarray | None = None
    latent: np.ndarray | None = None

    def __post_init__(self):
        shape = np.asarray(self.observation).shape
        if len(shape) != 2:
            raise ValueError("Evaluation arrays must have shape [lead,site]")
        for name in ("observation", "generating_mean", "latent"):
            value = getattr(self, name)
            if value is not None:
                value = readonly(value)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError("All available truth arrays must be finite and aligned")
                object.__setattr__(self, name, value)

