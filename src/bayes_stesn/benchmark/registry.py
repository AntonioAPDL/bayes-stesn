"""Capability-aware model dispatch; optional dependencies are loaded on demand."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from ..io import save_state, load_state
from .contracts import Panel, CapabilityError


@dataclass(frozen=True)
class ModelInfo:
    name: str
    tracks: tuple[str, ...]
    uncertainty: str
    family: str
    stochastic_fit: bool = False
    selection_metric: str = "rmse"
    parameter_uncertainty: bool = False


MODELS = {
    "persistence":ModelInfo("Last-observation persistence", ("A",), "point", "classical"),
    "seasonal_naive":ModelInfo("Seasonal repetition", ("A",), "point", "classical"),
    "spatial_persistence":ModelInfo("Spatially interpolated persistence", ("A","B"), "point", "classical"),
    "basis_lgssm":ModelInfo("Coordinate-basis linear Gaussian state space", ("A","B"), "joint_plugin", "classical", selection_metric="crps"),
    "separable_stgp":ModelInfo("Separable spatial/temporal Matérn GP", ("A","B"), "joint_plugin", "classical", selection_metric="crps"),
    # Both the bandwidth subsample and Nyström landmarks use a fit-seed stream.
    "basis_krr":ModelInfo("Coordinate-basis kernel ridge", ("A","B"), "point", "classical", True),
    "bayes_esn_ridge":ModelInfo("Bayesian spatial ESN with ridge prior", ("A","B"), "posterior_joint", "bayesian", True, "crps", True),
    "eqesn_rollout":ModelInfo("Recursive quadratic reservoir ensemble", ("A",), "finite_ensemble", "reservoir", True, "crps"),
    "dcrnn":ModelInfo("Autoregressive diffusion-GRU comparator", ("A",), "point", "neural", True),
    "basis_gru":ModelInfo("Coordinate-basis probabilistic GRU", ("A","B"), "joint_plugin", "neural", True, "crps"),
}


def model_registry():
    return {key:asdict(value) for key,value in MODELS.items()}


def fit_model(model_id, data: Panel, config=None, seed=0, *, checkpoint_path=None):
    if model_id not in MODELS:
        raise CapabilityError(f"Unknown model {model_id!r}; inspect model_registry()")
    if not isinstance(data, Panel):
        raise TypeError("Model fitting accepts an observation Panel, never a Simulation")
    family=MODELS[model_id].family
    if family=="classical":
        from .classical import fit_classical
        return fit_classical(model_id, data, config=config, seed=seed)
    if family=="bayesian":
        from .esn import fit_esn
        return fit_esn(data, config=config, seed=seed, checkpoint_path=checkpoint_path)
    if family=="reservoir":
        from .reservoir import fit_reservoir
        return fit_reservoir(model_id, data, config=config, seed=seed)
    if model_id=="basis_gru":
        from .basis_gru import fit_basis_gru
        return fit_basis_gru(data, config=config, seed=seed)
    from .neural import fit_neural
    return fit_neural(model_id, data, config=config, seed=seed)


def save_model(model, path):
    """Numeric arrays + JSON only; no untrusted pickle or Torch object loading."""
    return save_state(path, model.to_state(), kind="benchmark_model")


def load_model(path):
    state=load_state(path, kind="benchmark_model")
    model_id=state["model_id"]
    family=MODELS[model_id].family
    if family=="classical":
        from .classical import classical_from_state
        return classical_from_state(state)
    if family=="bayesian":
        from .esn import BayesianESN
        return BayesianESN.from_state(state)
    if family=="reservoir":
        from .reservoir import reservoir_from_state
        return reservoir_from_state(state)
    if model_id=="basis_gru":
        from .basis_gru import basis_gru_from_state
        return basis_gru_from_state(state)
    from .neural import neural_from_state
    return neural_from_state(state)
