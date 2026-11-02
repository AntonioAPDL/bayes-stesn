"""Shared, observation-only contracts for forecasting benchmarks.

``origin_index`` is the number of available rows. A forecast starts at that
row. The original ESN API uses the last observed row; its adapter performs
the one-step indexing conversion explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Any, Mapping
import json
import numpy as np

from ..data import SpatioTemporalData, readonly

Panel = SpatioTemporalData


class CapabilityError(ValueError):
    """The requested task is not defined by this fitted model."""


@dataclass(frozen=True)
class Forecast:
    """Point forecasts and, when supplied, coherent joint trajectory samples.

    Samples are ``[path, lead, site]``. No distribution is inferred from a
    point forecast; covariance and calibration capabilities remain explicit.
    """
    mean: np.ndarray
    samples: np.ndarray | None
    coordinates: np.ndarray
    site_ids: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        mean = readonly(self.mean)
        coords = readonly(self.coordinates)
        ids = tuple(map(str, self.site_ids))
        if mean.ndim != 2 or min(mean.shape) < 1 or not np.isfinite(mean).all():
            raise ValueError("Forecast mean must be finite [lead,site]")
        if coords.ndim != 2 or len(coords) != mean.shape[1] or not np.isfinite(coords).all():
            raise ValueError("Forecast coordinates must match site axis")
        if len(ids) != len(coords) or len(set(ids)) != len(ids):
            raise ValueError("Forecast site IDs must be unique and aligned")
        samples = None
        if self.samples is not None:
            samples = readonly(self.samples)
            if samples.ndim != 3 or samples.shape[1:] != mean.shape or not len(samples) or not np.isfinite(samples).all():
                raise ValueError("Samples must be finite nonempty [path,lead,site]")
        metadata = json.loads(json.dumps(dict(self.metadata), allow_nan=False))
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "coordinates", coords)
        object.__setattr__(self, "site_ids", ids)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "metadata", metadata)

    @property
    def median(self):
        return self.mean if self.samples is None else np.median(self.samples, axis=0)

    def interval(self, level=.9):
        if self.samples is None:
            raise CapabilityError("A point forecast does not define native prediction intervals")
        if not 0 < level < 1:
            raise ValueError("level must lie in (0,1)")
        return np.quantile(self.samples, [(1-level)/2, (1+level)/2], axis=0)

    def save(self, path):
        from ..io import save_state
        return save_state(path, {"mean":self.mean, "samples":self.samples,
                                "coordinates":self.coordinates, "site_ids":self.site_ids,
                                "metadata":self.metadata}, kind="benchmark_forecast")

    @classmethod
    def load(cls, path):
        from ..io import load_state
        return cls(**load_state(path, kind="benchmark_forecast"))


class FittedForecaster(Protocol):
    diagnostics: dict

    def forecast(self, horizon: int, target_coordinates=None, target_site_ids=None,
                 n_paths: int=256, seed: int=0, history: Panel | None=None) -> Forecast: ...


def select_panel(data: Panel, *, stop: int | None=None, sites=None, start: int=0) -> Panel:
    """Copy only permitted observed rows/sites into an isolated model input."""
    if not isinstance(data, Panel):
        raise TypeError("Models receive Panel observations, never a simulation truth object")
    stop = len(data.y) if stop is None else int(stop)
    if not 0 <= start < stop <= len(data.y):
        raise ValueError("Require a nonempty permitted historical slice")
    indices = np.arange(data.y.shape[1]) if sites is None else np.asarray(sites, dtype=int)
    if indices.ndim != 1 or not len(indices) or np.any(indices < 0) or np.any(indices >= data.y.shape[1]) or len(np.unique(indices)) != len(indices):
        raise ValueError("sites must contain unique valid input column indices")
    return Panel(data.y[start:stop, indices], data.coordinates[indices], data.times[start:stop],
                 tuple(data.site_ids[i] for i in indices), data.response_units,
                 data.coordinate_units, {"information_policy":"observed_prefix_only"})


def validate_history(training: Panel, history: Panel | None) -> Panel:
    """For frozen-parameter state conditioning, require the exact fitted prefix."""
    if history is None:
        return training
    if not isinstance(history, Panel):
        raise TypeError("history must be an observation Panel")
    n=len(training.y)
    if (history.site_ids != training.site_ids or
            not np.array_equal(history.coordinates, training.coordinates) or
            len(history.y) < n or not np.array_equal(history.y[:n], training.y) or
            not np.array_equal(history.times[:n], training.times)):
        raise ValueError("History must extend the identical fitted observations and site order")
    return history


def forecast_targets(training: Panel, target_coordinates=None, target_site_ids=None):
    coords = training.coordinates if target_coordinates is None else np.asarray(target_coordinates, float)
    if coords.ndim != 2 or coords.shape[1] != training.coordinates.shape[1] or not len(coords) or not np.isfinite(coords).all():
        raise ValueError("Target coordinates must match input geometry")
    if target_site_ids is None:
        ids = training.site_ids if target_coordinates is None else tuple(f"target_{i}" for i in range(len(coords)))
    else:
        ids=tuple(map(str,target_site_ids))
    if len(ids)!=len(coords) or len(set(ids))!=len(ids):
        raise ValueError("Target IDs must be unique and aligned")
    by_id=dict(zip(training.site_ids, training.coordinates))
    for identity,point in zip(ids,coords):
        if identity in by_id and not np.array_equal(point,by_id[identity]):
            raise ValueError("A target ID refers to a different input coordinate")
    return coords,ids
