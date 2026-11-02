"""Paired nonlinear spatial-field experiments with independent named RNG streams.

The structured mean reproduces the audited V3 moving-field equations in
``docs/derivations/06_study_design.md``. Random initial phases provide
independent dynamical trajectories. Residual scenarios change only their
declared mechanism, and variance-matched Student-t scenarios distinguish
scale matrices from marginal covariances.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np
from scipy.linalg import expm
from scipy.spatial.distance import cdist

__all__ = [
    "SCENARIOS", "SyntheticDataset", "named_rng", "regular_grid",
    "nested_input_design", "phase_path", "structured_mean", "simulate_dataset",
]

SCENARIOS = (
    "matched_gaussian", "anisotropic_gaussian", "latent_ar",
    "static_heteroskedastic", "whole_field_t", "local_t",
    "nonstationary_gaussian", "structural_break",
)
_ALIASES = {
    "matched": "matched_gaussian", "anisotropic": "anisotropic_gaussian",
    "static_hetero": "static_heteroskedastic", "field_t": "whole_field_t",
    "nonstationary": "nonstationary_gaussian", "break": "structural_break",
}


@dataclass(frozen=True)
class SyntheticDataset:
    """Distinct observation, latent-field, and structured-mean truths.

    Every response array has axes ``(time, site)``. ``latent`` equals the
    structured mean plus the spatial process; only ``observation`` includes
    the nugget. ``input_indices`` maps requested counts to nested geometry
    selections, never to response-informed sensor choices. ``parameters``
    contains JSON-safe provenance; phase paths are separate numeric arrays.
    """

    times: np.ndarray
    coordinates: np.ndarray
    observation: np.ndarray
    generating_mean: np.ndarray
    latent: np.ndarray
    input_indices: dict[int, np.ndarray]
    parameters: dict
    phases: np.ndarray


def _stream_digest(seed: int, stage: str, replication: int, component: str) -> str:
    if not isinstance(seed, (int, np.integer)) or seed < 0 or not isinstance(replication, (int, np.integer)) or replication < 0:
        raise ValueError("seed and replication must be nonnegative integers")
    if not isinstance(stage, str) or not stage or not isinstance(component, str) or not component:
        raise ValueError("stage and component must be nonempty strings")
    # Length prefixes prevent accidental ambiguity between identity fields.
    identity = f"bayes-stesn-v1|{int(seed)}|{len(stage)}:{stage}|{int(replication)}|{len(component)}:{component}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def named_rng(seed: int, stage: str, replication: int, component: str) -> np.random.Generator:
    """Stable stream identity unaffected by other RNG consumption/order.

    SHA256 digest words seed NumPy SeedSequence/PCG64. Scenario and model
    names are deliberately absent from DGP stream identities, allowing
    paired base innovations. Model-specific streams should include their
    model identity in ``component``. Python's randomized ``hash`` is never
    used. Results are deterministic within the recorded NumPy/BLAS platform.
    """
    digest = bytes.fromhex(_stream_digest(seed, stage, replication, component))
    words = np.frombuffer(digest, dtype="<u4").tolist()
    return np.random.default_rng(np.random.SeedSequence(words))


def _coordinates(coordinates: np.ndarray) -> np.ndarray:
    c = np.asarray(coordinates, dtype=float)
    if c.ndim != 2 or c.shape[1] != 2 or c.shape[0] < 1 or not np.isfinite(c).all() or np.any((c < 0) | (c > 1)):
        raise ValueError("coordinates must be finite sites in the unit square with shape (site,2)")
    if len(np.unique(c, axis=0)) != len(c):
        raise ValueError("coordinates must be distinct")
    return c


def regular_grid(grid_size: int = 25) -> np.ndarray:
    """Unit-square grid, row-major with x changing fastest."""
    if not isinstance(grid_size, (int, np.integer)) or grid_size < 2:
        raise ValueError("grid_size must be an integer >=2")
    axis = np.linspace(0, 1, grid_size)
    return np.column_stack((np.tile(axis, grid_size), np.repeat(axis, grid_size)))


def nested_input_design(
    coordinates: np.ndarray,
    sizes: Sequence[int],
    *,
    seed: int = 20261102,
    excluded_rectangle: tuple[float, float, float, float] = (0.60, 0.84, 0.12, 0.36),
) -> dict[int, np.ndarray]:
    """Geometry-only nested maximin sensor sets, frozen across replications.

    The rectangle is ``(xmin,xmax,ymin,ymax)`` and includes its boundary.
    A stable geometry stream breaks ties; selected sites at a smaller
    budget are a prefix of every larger design. Selection depends on no
    observations, phases, forecast horizons, or future inputs.
    """
    c = _coordinates(coordinates)
    requested = tuple(sizes)
    if not requested or any(not isinstance(n, (int, np.integer)) or n < 1 for n in requested) or len(set(requested)) != len(requested):
        raise ValueError("sizes must be distinct positive integers")
    rectangle = np.asarray(excluded_rectangle, dtype=float)
    if rectangle.shape != (4,) or not np.isfinite(rectangle).all() or rectangle[0] > rectangle[1] or rectangle[2] > rectangle[3]:
        raise ValueError("invalid excluded rectangle")
    xmin, xmax, ymin, ymax = rectangle
    eligible = np.flatnonzero(~((c[:, 0] >= xmin) & (c[:, 0] <= xmax) & (c[:, 1] >= ymin) & (c[:, 1] <= ymax)))
    if max(requested) > len(eligible):
        raise ValueError("input budget exceeds the sites outside the held-out rectangle")
    rng = named_rng(seed, "geometry", 0, "maximin-order")
    # Stable randomized tie priority, independent of all simulated data.
    priority = rng.permutation(len(eligible))
    selected = np.empty(max(requested), dtype=int)
    chosen = int(priority[0])
    nearest = np.full(len(eligible), np.inf)
    used = np.zeros(len(eligible), dtype=bool)
    for j in range(len(selected)):
        selected[j] = eligible[chosen]
        used[chosen] = True
        distance = np.sum((c[eligible] - c[eligible[chosen]]) ** 2, axis=1)
        nearest = np.minimum(nearest, distance)
        nearest[used] = -np.inf
        if j + 1 < len(selected):
            maximum = np.max(nearest)
            tied = np.flatnonzero(np.isclose(nearest[priority], maximum, rtol=1e-12, atol=1e-15))
            chosen = int(priority[tied[0]])
    return {int(size): selected[:size].copy() for size in sorted(requested)}


def phase_path(
    n_times: int,
    initial: Sequence[float] = (0.30, 1.10),
    *,
    burnin: int = 500,
    break_time: int | None = None,
) -> np.ndarray:
    """Two simultaneous nonlinear phase updates with source-faithful indexing.

    Exactly ``burnin`` updates are discarded; recorded time zero uses the
    next update. The optional break changes theta's drift from .115 to .150
    on the update generating ``break_time``. It never changes past values.
    """
    if not isinstance(n_times, (int, np.integer)) or n_times < 1 or not isinstance(burnin, (int, np.integer)) or burnin < 0:
        raise ValueError("n_times must be positive and burnin nonnegative")
    phases = np.asarray(initial, dtype=float)
    if phases.shape != (2,) or not np.isfinite(phases).all():
        raise ValueError("initial phases must be two finite values")
    if break_time is not None and (not isinstance(break_time, (int, np.integer)) or not 0 <= break_time < n_times):
        raise ValueError("break_time must be a recorded time in [0,n_times)")
    theta, phi = phases
    out = np.empty((n_times, 2))
    for iteration in range(burnin + n_times):
        recorded = iteration - burnin
        drift = 0.150 if break_time is not None and recorded >= break_time else 0.115
        theta, phi = (
            (theta + drift + 0.035 * np.sin(theta) + 0.018 * np.sin(phi)) % (2 * np.pi),
            (phi + 0.078 + 0.020 * np.sin(2 * theta - phi)) % (2 * np.pi),
        )
        if recorded >= 0:
            out[recorded] = (theta, phi)
    return out


def structured_mean(phases: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Evaluate the exact moving bumps, wave, front, and interaction field."""
    phase = np.asarray(phases, dtype=float)
    if phase.ndim != 2 or phase.shape[1] != 2 or phase.shape[0] < 1 or not np.isfinite(phase).all():
        raise ValueError("phases must have finite (time,2) axes")
    c = _coordinates(coordinates)
    x, y = c.T
    out = np.empty((len(phase), len(c)))
    # Bound temporary storage even for the full 9,600 × 625 study.
    for first in range(0, len(phase), 256):
        theta, phi = phase[first : first + 256].T[:, :, None]
        cx1, cy1 = 0.5 + 0.27 * np.cos(theta), 0.5 + 0.23 * np.sin(theta)
        cx2, cy2 = 0.5 + 0.28 * np.cos(phi + 0.70), 0.5 + 0.25 * np.sin(phi)
        d1, d2 = 0.110 + 0.014 * np.sin(phi), 0.135 + 0.012 * np.cos(theta - phi)
        field = (1.25 + 0.20 * np.sin(phi)) * np.exp(-((x - cx1) ** 2 + (y - cy1) ** 2) / (2 * d1**2))
        field -= (1.05 + 0.15 * np.cos(theta)) * np.exp(-((x - cx2) ** 2 + (y - cy2) ** 2) / (2 * d2**2))
        field += 0.40 * np.sin(2 * np.pi * (1.15 * x - 0.70 * y) - theta) * np.cos(np.pi * y + phi)
        field += 0.30 * np.tanh(9 * (x - 0.50 - 0.16 * np.sin(phi) - 0.12 * np.sin(2 * np.pi * y + theta)))
        field += 0.12 * np.sin(3 * np.pi * (x + y) + theta - 2 * phi)
        out[first : first + len(field)] = field
    return out


def _matern32(distance: np.ndarray) -> np.ndarray:
    scaled = np.sqrt(3.0) * np.maximum(distance, 0)
    return (1 + scaled) * np.exp(-scaled)


def _spatial_correlation(coordinates: np.ndarray, scenario: str, lengthscale: float) -> np.ndarray:
    if scenario == "anisotropic_gaussian":
        metric = expm(np.array([[0.55, 0.25], [0.25, -0.55]]))
        transformed = coordinates @ np.linalg.cholesky(metric)
        return _matern32(cdist(transformed, transformed) / lengthscale)
    if scenario != "nonstationary_gaussian":
        return _matern32(cdist(coordinates, coordinates) / lengthscale)
    x, y = coordinates.T
    u = 0.35 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
    e = np.empty((len(x), 2, 2))
    e[:, 0, 0] = 0.4 * np.sin(2 * np.pi * y)
    e[:, 1, 1] = -e[:, 0, 0]
    e[:, 0, 1] = e[:, 1, 0] = 0.25 * np.cos(2 * np.pi * x)
    local = np.stack([lengthscale**2 * expm(2 * u[j] * np.eye(2) - e[j]) for j in range(len(x))])
    average = (local[:, None] + local[None, :]) / 2
    determinant = np.linalg.det(local)
    det_average = np.linalg.det(average)
    difference = coordinates[:, None] - coordinates[None, :]
    radius_squared = np.einsum("...i,...ij,...j->...", difference, np.linalg.inv(average), difference)
    amplitude = determinant[:, None] ** 0.25 * determinant[None, :] ** 0.25 / np.sqrt(det_average)
    return amplitude * _matern32(np.sqrt(np.maximum(radius_squared, 0)))


def simulate_dataset(
    n_times: int = 9600,
    *,
    coordinates: np.ndarray | None = None,
    grid_size: int = 25,
    scenario: str = "matched_gaussian",
    seed: int = 20261102,
    replication: int = 0,
    stage: str = "study",
    input_sizes: Sequence[int] | None = None,
    burnin: int = 500,
    random_initial_phase: bool = True,
    break_time: int | None = None,
) -> SyntheticDataset:
    """Generate a complete paired nonlinear field with declared truth targets.

    Scenarios are listed in ``SCENARIOS``; short aliases are accepted.
    The structural-break stress case requires an explicit break time.
    No split, final forecast origin, or future exogenous input is inferred.
    No reservoir features enter this DGP. Spatial covariance is dense and
    uses no silently added jitter; duplicate coordinates are rejected.

    Defaults reproduce the blueprint's scale parameters and 25×25 domain.
    With ``input_sizes=None``, choose distinct min(128,available) and
    min(256,available) budgets, making small validation experiments usable.
    Named spatial/nugget/mixture streams are separate and paired across
    scenarios. Changing the simulation horizon preserves earlier draws.
    """
    canonical = _ALIASES.get(scenario, scenario)
    if canonical not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
    if canonical == "structural_break" and break_time is None:
        raise ValueError("structural_break requires an explicit break_time")
    if canonical != "structural_break" and break_time is not None:
        raise ValueError("break_time is only meaningful for structural_break")
    c = regular_grid(grid_size) if coordinates is None else _coordinates(coordinates).copy()
    if input_sizes is None:
        excluded = (c[:, 0] >= 0.60) & (c[:, 0] <= 0.84) & (c[:, 1] >= 0.12) & (c[:, 1] <= 0.36)
        available = int(np.sum(~excluded))
        input_sizes = tuple(sorted({min(128, available), min(256, available)}))
    inputs = nested_input_design(c, input_sizes, seed=seed)
    phase_rng = named_rng(seed, stage, replication, "phase")
    initial = phase_rng.uniform(0, 2 * np.pi, 2) if random_initial_phase else np.array([0.30, 1.10])
    phases = phase_path(n_times, initial, burnin=burnin, break_time=break_time)
    mean = structured_mean(phases, c)
    a, b, lengthscale = 0.065, 0.05, 0.085
    kernel = _spatial_correlation(c, canonical, lengthscale)
    kernel = (kernel + kernel.T) / 2
    factor = np.linalg.cholesky(kernel)
    spatial_rng = named_rng(seed, stage, replication, "spatial-innovation")
    nugget_rng = named_rng(seed, stage, replication, "nugget")
    # Separate field/site weights avoid a horizon-dependent RNG offset.
    weight_rng = named_rng(seed, stage, replication, "mixture-weight")
    spatial = a * (spatial_rng.normal(size=(n_times, len(c))) @ factor.T)
    nugget = b * nugget_rng.normal(size=(n_times, len(c)))
    nugget_sd = np.full(len(c), b)
    nu, rho = 5.0, 0.0
    if canonical == "latent_ar":
        rho = 0.8
        for t in range(1, n_times):
            spatial[t] = rho * spatial[t - 1] + np.sqrt(1 - rho**2) * spatial[t]
    elif canonical == "static_heteroskedastic":
        g = 0.7 * np.sin(2 * np.pi * c[:, 0]) * np.cos(np.pi * c[:, 1])
        multiplier = np.exp(g) / np.sqrt(np.mean(np.exp(2 * g)))
        nugget_sd *= multiplier
        nugget *= multiplier
    elif canonical == "whole_field_t":
        omega = weight_rng.gamma(nu / 2, 2 / nu, size=(n_times, 1))
        multiplier = np.sqrt((nu - 2) / (nu * omega))
        spatial *= multiplier
        nugget *= multiplier
    elif canonical == "local_t":
        omega = weight_rng.gamma(nu / 2, 2 / nu, size=(n_times, len(c)))
        nugget *= np.sqrt((nu - 2) / (nu * omega))
    latent = mean + spatial
    observation = latent + nugget
    if not np.isfinite(observation).all():
        raise FloatingPointError("DGP produced nonfinite observations")
    streams = {name: _stream_digest(seed, stage, replication, name) for name in ("phase", "spatial-innovation", "nugget", "mixture-weight")}
    streams["geometry"] = _stream_digest(seed, "geometry", 0, "maximin-order")
    parameters = {
        "scenario": canonical, "root_seed": int(seed), "replication": int(replication),
        "stage": stage, "seed_derivation": "SHA256 named identity -> little-endian uint32 SeedSequence -> PCG64", "stream_hashes": streams,
        "numpy_version": np.__version__, "n_times": int(n_times), "n_sites": int(len(c)),
        "initial_phase": initial.tolist(), "random_initial_phase": bool(random_initial_phase),
        "phase_burnin_updates": int(burnin), "spatial_sd": a, "nugget_rms_sd": float(np.sqrt(np.mean(nugget_sd**2))),
        "lengthscale": lengthscale, "matern_smoothness": 1.5, "rho": rho,
        "student_t_df": nu if canonical in {"whole_field_t", "local_t"} else None,
        "student_t_variance_matched": canonical in {"whole_field_t", "local_t"},
        "break_time": int(break_time) if break_time is not None else None,
        "input_sizes": list(inputs), "excluded_rectangle": [0.60, 0.84, 0.12, 0.36],
        "covariance_jitter": 0.0,
        "geometry_sha256": hashlib.sha256(c.astype("<f8").tobytes()).hexdigest(),
        "observation_sha256": hashlib.sha256(observation.astype("<f8").tobytes()).hexdigest(),
        "truth_definitions": {"generating_mean": "f", "latent": "f+w", "observation": "f+w+nugget"},
    }
    return SyntheticDataset(np.arange(n_times), c, observation, mean, latent, inputs, parameters, phases)
