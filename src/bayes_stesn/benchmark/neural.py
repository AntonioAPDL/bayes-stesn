"""Optional CPU diffusion-GRU comparator with honest point forecasts.

``dcrnn`` here is the explicitly labelled **shared-transition autoregressive
diffusion-GRU** variant.  It does not pretend to reproduce the original
DCRNN encoder/decoder architecture.  A symmetric geographic k-nearest-
neighbor graph is row normalized to P.  Diffusion features are

    D_K(X) = [X, P X, ..., P**K X].

At each layer, with X=[y,h],

    r,z = sigmoid(D_K(X) W_rz + b_rz),
    c = tanh(D_K([y,r*h]) W_c + b_c),
    h_new = z*h + (1-z)*c.

A shared linear head predicts the next response.  A fixed observed context
initializes hidden states; subsequent responses are recursively generated.
Training uses a multistep loss and scheduled sampling.  Chronological
train-tail early stopping uses wholly free-running forecasts and restores
the best checkpoint.  By default, the selected epoch count is then refitted
on all supplied training rows with newly fitted training scales.  No outer
validation, calibration, or final observations are available to this module.

PyTorch is imported only when fitting/loading a neural model.  Numeric NPZ
serialization stores arrays, not pickled Torch objects.  Defaults and the
complete accepted config schema are ``DEFAULT_CONFIG``. ``stride`` spaces
training windows in rows to reduce redundant optimization work. The graph
is sparse during message passing and depends only on supplied coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from ..io import seed_for
from .contracts import Panel, Forecast, CapabilityError, validate_history
from .reservoir import _integer, _observed_targets, _panel_state


DEFAULT_CONFIG = {
    "hidden_size": 32, "layers": 1, "diffusion_order": 2, "graph_k": 8,
    "context": 24, "learning_rate": .001, "weight_decay": 1e-5,
    "max_epochs": 100, "patience": 15, "batch_size": 32,
    "train_fraction": .85, "gradient_clip": 1.,
    "forecast_training_steps": 4, "stride": 4,
    "validation_horizon": 64, "refit_full": True,
}


def _config(config):
    supplied = {} if config is None else dict(config)
    unknown = set(supplied)-set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown diffusion-GRU options: {sorted(unknown)}")
    result = {**DEFAULT_CONFIG, **supplied}
    for key in ("hidden_size", "layers", "context", "max_epochs", "patience", "batch_size",
                "forecast_training_steps", "stride", "validation_horizon"):
        result[key] = _integer(result[key], key)
    for key in ("diffusion_order", "graph_k"):
        result[key] = _integer(result[key], key, 0)
    for key in ("learning_rate", "gradient_clip", "train_fraction"):
        result[key] = float(result[key])
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    result["weight_decay"] = float(result["weight_decay"])
    if not np.isfinite(result["weight_decay"]) or result["weight_decay"] < 0:
        raise ValueError("weight_decay must be nonnegative and finite")
    if not .5 <= result["train_fraction"] < 1:
        raise ValueError("train_fraction must lie in [0.5,1)")
    if not isinstance(result["refit_full"], (bool, np.bool_)):
        raise ValueError("refit_full must be boolean")
    result["refit_full"] = bool(result["refit_full"])
    return result


def _torch():
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise CapabilityError("dcrnn requires the optional PyTorch dependency") from exc


def coordinate_diffusion_operator(coordinates, neighbors=8):
    """Sparse, row-stochastic geometric graph with deterministic tie handling.

    Ties in the k-nearest query are resolved by site index.  The undirected
    edge weight is exp(-distance²/(2*median_neighbor_distance²)); a zero-
    neighbor graph, or a single site, reduces to identity diffusion.
    """
    coordinates = np.asarray(coordinates, float)
    neighbors = _integer(neighbors, "neighbors", 0)
    if coordinates.ndim != 2 or not len(coordinates) or not np.isfinite(coordinates).all():
        raise ValueError("coordinates must be a nonempty finite matrix")
    n = len(coordinates)
    if n == 1 or neighbors == 0:
        return sparse.eye(n, format="csr")
    k = min(neighbors, n-1)
    tree = cKDTree(coordinates)
    distances, indices = tree.query(coordinates, k=k+1)
    rows, columns, lengths = [], [], []
    for i in range(n):
        # Include all exact kth-distance ties before deterministic selection.
        radius = float(distances[i, -1])
        candidates = tree.query_ball_point(coordinates[i], np.nextafter(radius, np.inf))
        ranked = sorted((float(np.linalg.norm(coordinates[i]-coordinates[j])), int(j))
                        for j in candidates if j != i)
        for distance, j in ranked[:k]:
            rows.append(i); columns.append(j); lengths.append(distance)
    length = max(float(np.median(lengths)), np.finfo(float).eps)
    adjacency = sparse.csr_matrix((np.exp(-.5*(np.asarray(lengths)/length)**2), (rows, columns)), shape=(n, n))
    adjacency = adjacency.maximum(adjacency.T)
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    return (sparse.diags(1./degrees)@adjacency).tocsr()


def diffusion_features(values, operator, order):
    """NumPy/SciPy reference for [X,PX,...,P^K X] at one time."""
    order = _integer(order, "order", 0)
    current = np.asarray(values, float)
    result = [current]
    for _ in range(order):
        current = operator@current
        result.append(current)
    return np.concatenate(result, axis=-1)


def _build_network(torch, config, operator):
    """Construct modules lazily, preserving a torch-free package import."""
    nn = torch.nn
    coo = operator.tocoo()
    index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.int64)
    values = torch.tensor(coo.data, dtype=torch.float32)
    graph = torch.sparse_coo_tensor(index, values, coo.shape, check_invariants=True).coalesce()
    order, width = config["diffusion_order"], config["hidden_size"]

    class DiffusionGRUCell(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            expanded = (order+1)*(input_size+width)
            self.gates = nn.Linear(expanded, 2*width)
            self.candidate = nn.Linear(expanded, width)

        def forward(self, x, hidden, diffusion):
            reset, update = torch.sigmoid(self.gates(diffusion(torch.cat([x, hidden], dim=-1)))).chunk(2, dim=-1)
            candidate = torch.tanh(self.candidate(diffusion(torch.cat([x, reset*hidden], dim=-1))))
            return update*hidden+(1-update)*candidate

    class DiffusionGRUNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            # Graph state is serialized once as a SciPy CSR, not as a Torch object.
            self.register_buffer("operator", graph, persistent=False)
            self.cells = nn.ModuleList([DiffusionGRUCell(1 if layer == 0 else width)
                                       for layer in range(config["layers"])])
            self.head = nn.Linear(width, 1)

        def diffusion(self, x):
            terms = [x]
            batch, nodes, features = x.shape
            for _ in range(order):
                mixed = torch.sparse.mm(self.operator, x.permute(1, 0, 2).reshape(nodes, batch*features))
                x = mixed.reshape(nodes, batch, features).permute(1, 0, 2)
                terms.append(x)
            return torch.cat(terms, dim=-1)

        def step(self, response, hidden=None):
            x = response.unsqueeze(-1)
            if hidden is None:
                hidden = [response.new_zeros(response.shape[0], response.shape[1], width)
                          for _ in self.cells]
            new_hidden = []
            for cell, previous in zip(self.cells, hidden):
                x = cell(x, previous, self.diffusion)
                new_hidden.append(x)
            return self.head(x).squeeze(-1), new_hidden

        def forward(self, context, steps, targets=None, teacher_probability=0., generator=None):
            hidden = None
            for t in range(context.shape[1]):
                prediction, hidden = self.step(context[:, t], hidden)
            result = [prediction]
            for lead in range(1, steps):
                previous = prediction
                if targets is not None and teacher_probability > 0:
                    use_observed = torch.rand((len(context), 1), generator=generator) < teacher_probability
                    previous = torch.where(use_observed, targets[:, lead-1], previous)
                prediction, hidden = self.step(previous, hidden)
                result.append(prediction)
            return torch.stack(result, dim=1)

    return DiffusionGRUNetwork()


def _standardizer(y):
    return y.mean(axis=0), np.maximum(y.std(axis=0), 1e-6)


def _fit_epochs(torch, raw, graph, config, seed, *, cutoff=None, epochs=None):
    """One training phase; validation values enter only the stopping score."""
    fit_stop = len(raw) if cutoff is None else cutoff
    center, scale = _standardizer(raw[:fit_stop])
    observed = torch.tensor((raw-center)/scale, dtype=torch.float32)
    context, steps = config["context"], config["forecast_training_steps"]
    origins = torch.arange(context, fit_stop-steps+1, config["stride"])
    if not len(origins):
        raise ValueError("Training prefix is too short for the declared context and multistep loss")
    window_offsets = torch.arange(-context, 0)
    target_offsets = torch.arange(steps)
    initialization_seed = seed_for(seed, "dcrnn", "initialization") % (2**63-1)
    optimization_seed = seed_for(seed, "dcrnn", "optimization") % (2**63-1)
    records = []
    # Restore the caller's global Torch random state after deterministic fitting.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initialization_seed)
        network = _build_network(torch, config, graph)
        optimizer = torch.optim.AdamW(network.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        generator = torch.Generator(device="cpu").manual_seed(optimization_seed)
        best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
        limit = config["max_epochs"] if epochs is None else epochs
        for epoch in range(limit):
            network.train()
            total_loss, cases = 0., 0
            order = torch.randperm(len(origins), generator=generator)
            teacher_probability = max(0., 1.-epoch/max(config["max_epochs"]-1, 1))
            for start in range(0, len(origins), config["batch_size"]):
                batch_origins = origins[order[start:start+config["batch_size"]]]
                inputs = observed[batch_origins[:, None]+window_offsets]
                targets = observed[batch_origins[:, None]+target_offsets]
                optimizer.zero_grad(set_to_none=True)
                predicted = network(inputs, steps, targets, teacher_probability, generator)
                loss = torch.mean((predicted-targets)**2)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Diffusion-GRU optimization produced nonfinite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                total_loss += float(loss.detach())*len(batch_origins)
                cases += len(batch_origins)
            record = {"epoch": epoch+1, "training_mse": total_loss/cases,
                      "teacher_probability": teacher_probability}
            if cutoff is not None:
                network.eval()
                horizon = min(config["validation_horizon"], len(raw)-cutoff)
                with torch.no_grad():
                    prediction = network(observed[cutoff-context:cutoff][None], horizon)
                    validation_loss = float(torch.mean((prediction[0]-observed[cutoff:cutoff+horizon])**2))
                if not np.isfinite(validation_loss):
                    raise FloatingPointError("Diffusion-GRU free-running validation was nonfinite")
                record["validation_free_running_mse"] = validation_loss
                if validation_loss < best_loss-1e-9:
                    best_loss, best_epoch, stale = validation_loss, epoch+1, 0
                    best_state = {key: value.detach().clone() for key, value in network.state_dict().items()}
                else:
                    stale += 1
            records.append(record)
            if cutoff is not None and stale >= config["patience"]:
                break
        if cutoff is not None:
            network.load_state_dict(best_state)
        else:
            best_epoch = limit
        if not all(bool(torch.isfinite(parameter).all()) for parameter in network.parameters()):
            raise FloatingPointError("Diffusion-GRU checkpoint contains nonfinite parameters")
        network.eval()
    return network, center, scale, records, best_epoch


@dataclass
class DiffusionGRU:
    training: Panel
    config: dict
    graph: sparse.csr_matrix
    center: np.ndarray
    scale: np.ndarray
    network: object
    diagnostics: dict
    model_id: str = "dcrnn"

    def forecast(self, horizon, target_coordinates=None, target_site_ids=None,
                 n_paths=256, seed=0, history=None):
        horizon = _integer(horizon, "horizon")
        _integer(n_paths, "n_paths", 0)
        data = validate_history(self.training, history)
        coordinates, ids, indices = _observed_targets(self.training, target_coordinates, target_site_ids)
        torch = _torch()
        context = (data.y[-self.config["context"]:]-self.center)/self.scale
        self.network.eval()
        with torch.no_grad():
            predicted = self.network(torch.tensor(context[None], dtype=torch.float32), horizon)[0].numpy()
        response = predicted*self.scale+self.center
        return Forecast(response[:, indices], None, coordinates, ids, {
            "model_id": self.model_id,
            "variant": "shared-transition autoregressive diffusion-GRU",
            "uncertainty": "point_only", "origin_index": len(data.y),
            "parameter_fit_rows": self.diagnostics["parameter_fit_rows"],
            "conditioning_rows": len(data.y), "context_rows": self.config["context"],
            "rollout_policy": "free running; observed context followed exclusively by model outputs",
        })

    def to_state(self):
        return {"model_id": self.model_id, "schema_version": 1,
                "training": _panel_state(self.training), "config": self.config,
                "center": self.center, "scale": self.scale,
                "graph": {"data": self.graph.data, "indices": self.graph.indices,
                          "indptr": self.graph.indptr, "shape": self.graph.shape},
                "weights": {key: value.detach().cpu().numpy().copy()
                            for key, value in self.network.state_dict().items()},
                "diagnostics": self.diagnostics}

    @classmethod
    def from_state(cls, state):
        if state.get("model_id") != "dcrnn" or state.get("schema_version") != 1:
            raise ValueError("Unsupported diffusion-GRU state")
        torch = _torch()
        config = _config(state["config"])
        saved = state["graph"]
        graph = sparse.csr_matrix((saved["data"], saved["indices"], saved["indptr"]), shape=tuple(saved["shape"]))
        # Building initialized layers must not perturb the caller's RNG state.
        with torch.random.fork_rng(devices=[]):
            network = _build_network(torch, config, graph)
        network.load_state_dict({key: torch.tensor(value) for key, value in state["weights"].items()})
        network.eval()
        return cls(Panel(**state["training"]), config, graph,
                   np.array(state["center"], copy=True), np.array(state["scale"], copy=True),
                   network, dict(state["diagnostics"]))


def fit_neural(model_id, data: Panel, config=None, seed=0):
    """Fit the declared optional diffusion-GRU using only observed training data."""
    if model_id != "dcrnn":
        raise CapabilityError(f"Unknown or unimplemented neural comparator {model_id!r}")
    if not isinstance(data, Panel):
        raise TypeError("Neural fitting accepts an observation Panel")
    config = _config(config)
    cutoff = int(len(data.y)*config["train_fraction"])
    if cutoff < config["context"]+config["forecast_training_steps"] or cutoff >= len(data.y):
        raise ValueError("Training prefix is too short for chronological tail validation and the chosen context")
    torch = _torch()
    graph = coordinate_diffusion_operator(data.coordinates, config["graph_k"])
    selected, center, scale, selection_records, selected_epoch = _fit_epochs(
        torch, data.y, graph, config, seed, cutoff=cutoff)
    if config["refit_full"]:
        network, center, scale, refit_records, _ = _fit_epochs(
            torch, data.y, graph, config, seed, epochs=selected_epoch)
        parameter_fit_rows = len(data.y)
    else:
        network, refit_records, parameter_fit_rows = selected, [], cutoff
    diagnostics = {
        "status": "fitted", "qualified": True,
        "qualification": "finite optimized point predictor; no claim of global optimization or MCMC convergence",
        "variant": "shared-transition autoregressive diffusion-GRU",
        "seed": int(seed), "torch_version": str(torch.__version__), "device": "cpu",
        "torch_threads": int(torch.get_num_threads()), "graph_nonzeros": int(graph.nnz),
        "graph_definition": "symmetric geometric kNN, Gaussian distance weights, row-stochastic single-direction diffusion",
        "context": config["context"], "selection_training_rows": cutoff,
        "selection_validation_rows": min(config["validation_horizon"], len(data.y)-cutoff),
        "selected_epoch": selected_epoch, "best_checkpoint_restored": True,
        "parameter_fit_rows": parameter_fit_rows, "refit_full": config["refit_full"],
        "selection_losses": selection_records, "refit_losses": refit_records,
        "loss_units": "training-standardized response MSE",
        "selection_policy": "chronological supplied-training tail; free-running validation",
        "scaling_policy": "selection-prefix scales during stopping; all supplied training rows for final refit",
    }
    return DiffusionGRU(data, config, graph, center, scale, network, diagnostics)


def neural_from_state(state):
    return DiffusionGRU.from_state(state)
