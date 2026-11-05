"""Fixed radial spatial mean dictionaries with arbitrary-coordinate evaluation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class SpatialMeanSpec:
    side: int = 5
    width: float = .22
    kernel: str = "matern32"
    coordinate_terms: bool = True
    bounds: tuple[tuple[float, float], ...] | None = None
    centers: tuple[tuple[float, ...], ...] | None = None
    max_columns: int = 4096
    harmonics: int = 8

    def __post_init__(self):
        if self.side < 2 or not np.isfinite(self.width) or self.width <= 0:
            raise ValueError("Spatial mean side >= 2 and positive finite width required")
        if self.kernel not in {"matern32", "gaussian", "fourier"}:
            raise ValueError("Supported spatial mean kernels: matern32, gaussian, fourier")
        if (isinstance(self.harmonics, (bool, np.bool_))
                or not isinstance(self.harmonics, (int, np.integer))
                or self.harmonics < 1):
            raise ValueError("Fourier harmonics must be a positive integer")
        if self.kernel == "fourier" and (self.coordinate_terms or self.centers is not None
                                         or self.bounds is not None):
            raise ValueError("Fourier basis uses harmonics only; disable coordinate_terms and omit centers/bounds")
        if self.max_columns < 2:
            raise ValueError("max_columns must be at least two")
        for name in ("bounds", "centers"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, tuple(tuple(map(float, v)) for v in getattr(self, name)))

    @classmethod
    def v3(cls):
        return cls(side=13, width=.18, bounds=((0, 1), (0, 1)))


class SpatialMean:
    def __init__(self, spec, centers, center, scale, normalization):
        self.spec, self.centers = spec, centers
        self.center, self.scale, self.normalization = center, scale, float(normalization)

    @classmethod
    def fit(cls, coordinates, spec):
        coordinates = np.asarray(coordinates, dtype=float)
        if spec.kernel == "fourier":
            if 1+2*spec.harmonics > spec.max_columns:
                raise MemoryError("Fourier dictionary exceeds max_columns")
            # Predetermined L2(circle) normalization. In particular, no small
            # training-arc standard deviation inflates unseen-sector columns.
            instance = cls(spec, np.empty((0, 2)), np.zeros(2*spec.harmonics),
                           np.full(2*spec.harmonics, 1/np.sqrt(2)),
                           np.sqrt(2*spec.harmonics))
            instance._raw(coordinates)  # validate the declared circle geometry
            return instance
        dim = coordinates.shape[1]
        count = len(spec.centers) if spec.centers is not None else spec.side**dim
        if count+1+(dim if spec.coordinate_terms else 0)>spec.max_columns:
            raise MemoryError("Spatial dictionary exceeds max_columns; choose centers explicitly or increase the gate deliberately")
        if spec.centers is not None:
            centers = np.asarray(spec.centers)
            if centers.ndim != 2 or centers.shape[1] != dim or not np.isfinite(centers).all():
                raise ValueError("Spatial centers must match the observed coordinate dimension")
        else:
            bounds = spec.bounds or tuple(zip(coordinates.min(axis=0), coordinates.max(axis=0)))
            if len(bounds) != dim or any(b <= a for a, b in bounds):
                raise ValueError("Each spatial dictionary dimension needs increasing bounds")
            axes = [np.linspace(a, b, spec.side) for a, b in bounds]
            centers = np.column_stack([v.ravel() for v in np.meshgrid(*axes)])
        instance = cls(spec, centers, None, None, 1.)
        raw = instance._raw(coordinates)
        instance.center = raw.mean(axis=0)
        instance.scale = np.maximum(raw.std(axis=0), 1e-6)
        instance.normalization = np.sqrt(raw.shape[1])
        return instance

    def _raw(self, coordinates):
        if self.spec.kernel == "fourier":
            coordinates = np.asarray(coordinates, dtype=float)
            if (coordinates.ndim != 2 or coordinates.shape[1] != 2
                    or not np.isfinite(coordinates).all()
                    or not np.allclose(np.linalg.norm(coordinates, axis=1), 1.,
                                       rtol=0., atol=1e-10)):
                raise ValueError("Fourier spatial means require finite unit-circle coordinates")
            angle = np.arctan2(coordinates[:, 1], coordinates[:, 0])
            phase = angle[:, None]*np.arange(1, self.spec.harmonics+1)
            # Pair order: cos(1), sin(1), cos(2), sin(2), ... .
            return np.stack((np.cos(phase), np.sin(phase)), axis=-1).reshape(len(angle), -1)
        distance = cdist(np.asarray(coordinates, dtype=float), self.centers)
        if self.spec.kernel == "matern32":
            u = np.sqrt(3)*distance/self.spec.width
            radial = (1+u)*np.exp(-u)
        else:
            radial = np.exp(-.5*(distance/self.spec.width)**2)
        return np.column_stack([coordinates, radial]) if self.spec.coordinate_terms else radial

    def transform(self, coordinates):
        raw = self._raw(coordinates)
        return np.column_stack([np.ones(len(raw)), (raw-self.center)/self.scale/self.normalization])

    def to_state(self):
        from dataclasses import asdict
        return {"spec": asdict(self.spec), "centers": self.centers, "center": self.center,
                "scale": self.scale, "normalization": self.normalization}

    @classmethod
    def from_state(cls, state):
        return cls(SpatialMeanSpec(**state["spec"]), state["centers"], state["center"],
                   state["scale"], state["normalization"])

