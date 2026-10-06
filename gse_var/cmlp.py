from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from torch import nn

from .data import construct_lagged_dataset
from .regularizers import RegularizerName

OptimizerName = Literal["adam", "ista"]


@dataclass(frozen=True)
class CMLPTrainingConfig:
    order: int
    hidden_layer_size: int
    num_hidden_layers: int = 1
    max_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    lambda_ngc: float = 0.0
    ridge_lambda: float = 0.0
    regularizer: RegularizerName = "sparse_group_lasso"
    sparse_group_lambda: float = 0.0
    sparse_l1_lambda: float = 0.0
    optimizer: OptimizerName = "ista"
    seed: int | None = 42
    shuffle: bool = True
    causal_threshold: float = 1e-8
    verbose: int = 1
    log_every: int = 10
    device: str | torch.device | None = None


@dataclass
class CMLPFitResult:
    model: "CMLP"
    history: dict[str, list[float]] = field(default_factory=dict)
    coeffs: np.ndarray | None = None
    causal_strength: np.ndarray | None = None
    causal_graph: np.ndarray | None = None


class _TargetMLP(nn.Module):
    """Neural-GC cMLP target network.

    The first layer is a Conv1d over the full lag window, matching the reference
    implementation's first-layer weight shape ``[hidden, source, lag]``.
    """

    def __init__(self, num_vars: int, order: int, hidden_layer_size: int, num_hidden_layers: int) -> None:
        super().__init__()
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        layers: list[nn.Module] = [nn.Conv1d(num_vars, hidden_layer_size, order)]
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Conv1d(hidden_layer_size, hidden_layer_size, 1))
        layers.append(nn.Conv1d(hidden_layer_size, 1, 1))
        self.layers = nn.ModuleList(layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.transpose(2, 1)
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx != 0:
                x = torch.relu(x)
            x = layer(x)
        return x.squeeze(2)

    @property
    def first_weight(self) -> torch.Tensor:
        return self.layers[0].weight


class CMLP(nn.Module):
    """Component-wise MLP baseline from Neural-GC."""

    def __init__(
        self,
        num_vars: int,
        order: int,
        hidden_layer_size: int,
        num_hidden_layers: int = 1,
    ) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.order = order
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.networks = nn.ModuleList(
            [
                _TargetMLP(num_vars, order, hidden_layer_size, num_hidden_layers)
                for _ in range(num_vars)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [batch, lag, variables]")
        if inputs.shape[1:] != (self.order, self.num_vars):
            raise ValueError(
                f"expected inputs shape [batch, {self.order}, {self.num_vars}], "
                f"got {tuple(inputs.shape)}"
            )

        # All target networks have the same architecture. Stacking their
        # parameters evaluates the independent cMLPs in one batched operation
        # while preserving the reference model's parameters and gradients.
        first_weights = torch.stack([network.layers[0].weight for network in self.networks])
        first_biases = torch.stack([network.layers[0].bias for network in self.networks])
        flat_weights = first_weights.permute(0, 1, 3, 2).reshape(
            self.num_vars * self.hidden_layer_size,
            self.order * self.num_vars,
        )
        activations = torch.nn.functional.linear(
            inputs.reshape(inputs.shape[0], -1),
            flat_weights,
            first_biases.reshape(-1),
        ).reshape(inputs.shape[0], self.num_vars, self.hidden_layer_size)

        for layer_idx in range(1, len(self.networks[0].layers)):
            activations = torch.relu(activations)
            weights = torch.stack(
                [network.layers[layer_idx].weight.squeeze(-1) for network in self.networks]
            )
            biases = torch.stack([network.layers[layer_idx].bias for network in self.networks])
            activations = torch.bmm(
                activations.transpose(0, 1),
                weights.transpose(1, 2),
            ).transpose(0, 1) + biases.unsqueeze(0)

        return activations.squeeze(-1)

    @torch.no_grad()
    def gc_strength(self, ignore_lag: bool = True) -> torch.Tensor:
        if ignore_lag:
            strengths = [torch.linalg.vector_norm(net.first_weight, ord=2, dim=(0, 2)) for net in self.networks]
        else:
            strengths = [torch.linalg.vector_norm(net.first_weight, ord=2, dim=0) for net in self.networks]
        return torch.stack(strengths)

    @torch.no_grad()
    def causal_graph(self, threshold: float = 1e-8) -> torch.Tensor:
        return (self.gc_strength(ignore_lag=True) > threshold).to(torch.int64)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _validate_config(config: CMLPTrainingConfig) -> None:
    if config.lambda_ngc < 0:
        raise ValueError("lambda_ngc must be non-negative")
    if config.ridge_lambda < 0:
        raise ValueError("ridge_lambda must be non-negative")
    if config.sparse_group_lambda < 0:
        raise ValueError("sparse_group_lambda must be non-negative")
    if config.sparse_l1_lambda < 0:
        raise ValueError("sparse_l1_lambda must be non-negative")
    if config.regularizer == "sparse_group_lasso":
        if config.lambda_ngc != 0:
            raise ValueError(
                "sparse_group_lasso does not use lambda_ngc. "
                "Use sparse_group_lambda and sparse_l1_lambda instead."
            )
    elif config.sparse_group_lambda != 0 or config.sparse_l1_lambda != 0:
        raise ValueError(
            "sparse_group_lambda and sparse_l1_lambda are only used with "
            "regularizer='sparse_group_lasso'. Use lambda_ngc otherwise."
        )


def _iter_batches(n_samples: int, batch_size: int, shuffle: bool, rng: np.random.Generator, device: torch.device):
    if not shuffle:
        for start in range(0, n_samples, batch_size):
            yield slice(start, start + batch_size)
        return
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    indices_t = torch.as_tensor(indices, dtype=torch.long, device=device)
    for start in range(0, n_samples, batch_size):
        yield indices_t[start : start + batch_size]


def _ridge_penalty(model: CMLP, lam: float) -> torch.Tensor:
    if lam == 0:
        return next(model.parameters()).new_zeros(())
    penalty = next(model.parameters()).new_zeros(())
    for network in model.networks:
        for layer in network.layers[1:]:
            penalty = penalty + layer.weight.pow(2).sum()
    return lam * penalty


def _cmlp_regularize(model: CMLP, config: CMLPTrainingConfig) -> torch.Tensor:
    if config.regularizer == "none":
        return next(model.parameters()).new_zeros(())

    weights = torch.stack([network.first_weight for network in model.networks])
    if config.regularizer == "sparse_group_lasso":
        penalty = weights.new_zeros(())
        if config.sparse_l1_lambda:
            penalty = penalty + config.sparse_l1_lambda * torch.linalg.vector_norm(
                weights, ord=2, dim=1
            ).sum()
        if config.sparse_group_lambda:
            penalty = penalty + config.sparse_group_lambda * torch.linalg.vector_norm(
                weights, ord=2, dim=(1, 3)
            ).sum()
        return penalty
    if config.regularizer == "group_lasso":
        return config.lambda_ngc * torch.linalg.vector_norm(weights, ord=2, dim=(1, 3)).sum()
    if config.regularizer == "hierarchical_group_lasso":
        return config.lambda_ngc * sum(
            torch.linalg.vector_norm(weights[..., : lag_idx + 1], ord=2, dim=(1, 3)).sum()
            for lag_idx in range(weights.shape[3])
        )
    raise ValueError(f"unsupported regularizer: {config.regularizer}")


@torch.no_grad()
def _prox_cmlp_(model: CMLP, config: CMLPTrainingConfig) -> None:
    if config.regularizer == "none":
        return

    weights = torch.stack([network.first_weight for network in model.networks])
    if config.regularizer == "sparse_group_lasso":
        if config.sparse_l1_lambda:
            _prox_weight_group_(weights, config.sparse_l1_lambda, config.learning_rate, dims=1)
        if config.sparse_group_lambda:
            _prox_weight_group_(weights, config.sparse_group_lambda, config.learning_rate, dims=(1, 3))
    elif config.regularizer == "group_lasso":
        _prox_weight_group_(weights, config.lambda_ngc, config.learning_rate, dims=(1, 3))
    elif config.regularizer == "hierarchical_group_lasso":
        threshold = config.lambda_ngc * config.learning_rate
        for lag_idx in range(weights.shape[3]):
            block = weights[..., : lag_idx + 1]
            norms = torch.linalg.vector_norm(block, ord=2, dim=(1, 3), keepdim=True)
            scale = torch.clamp(1.0 - threshold / torch.clamp(norms, min=1e-12), min=0.0)
            block.mul_(scale)
            block.masked_fill_(norms <= threshold, 0.0)
    else:
        raise ValueError(f"unsupported regularizer: {config.regularizer}")

    for target_idx, network in enumerate(model.networks):
        network.first_weight.copy_(weights[target_idx])


@torch.no_grad()
def _prox_weight_group_(weight: torch.Tensor, lam: float, lr: float, dims: int | tuple[int, ...]) -> None:
    threshold = lam * lr
    norms = torch.linalg.vector_norm(weight, ord=2, dim=dims, keepdim=True)
    scale = torch.clamp(1.0 - threshold / torch.clamp(norms, min=1e-12), min=0.0)
    weight.mul_(scale)
    weight.masked_fill_(norms <= threshold, 0.0)


def _make_optimizer(config: CMLPTrainingConfig, model: CMLP) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.optimizer == "ista":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    raise ValueError(f"unsupported optimizer: {config.optimizer}")


def _log_epoch(epoch: int, config: CMLPTrainingConfig, metrics: dict[str, float], model: CMLP) -> None:
    if config.verbose <= 0:
        return
    log_every = max(config.log_every, 1)
    if epoch != 1 and epoch != config.max_epochs and epoch % log_every != 0:
        return
    with torch.no_grad():
        graph = model.causal_graph(config.causal_threshold)
        active = int(graph.sum().detach().cpu())
        total = graph.numel()
    print(
        f"Epoch {epoch:>4}/{config.max_epochs:<4} | "
        f"loss={metrics['loss']:.6g} | "
        f"mse={metrics['mse']:.6g} | "
        f"ngc={metrics['ngc']:.6g} | "
        f"ridge={metrics['ridge']:.6g} | "
        f"active_edges={active}/{total} ({100.0 * active / max(total, 1):.2f}%)"
    )


def fit_cmlp(
    data: np.ndarray | list[np.ndarray],
    config: CMLPTrainingConfig,
    model: CMLP | None = None,
) -> CMLPFitResult:
    """Fit Neural-GC cMLP baseline with the same high-level API as GSE-VAR."""
    _validate_config(config)
    if config.seed is not None:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    dataset = construct_lagged_dataset(data, config.order)
    device = _resolve_device(config.device)
    predictors = torch.as_tensor(dataset.predictors, dtype=torch.float32, device=device)
    responses = torch.as_tensor(dataset.responses, dtype=torch.float32, device=device)

    if model is None:
        model = CMLP(
            num_vars=predictors.shape[-1],
            order=config.order,
            hidden_layer_size=config.hidden_layer_size,
            num_hidden_layers=config.num_hidden_layers,
        )
    elif model.order != config.order or model.num_vars != predictors.shape[-1]:
        raise ValueError("provided CMLP model is incompatible with data/config")
    model.to(device)
    optimizer = _make_optimizer(config, model)
    criterion = nn.MSELoss(reduction="mean")
    rng = np.random.default_rng(config.seed)
    history: dict[str, list[float]] = {"loss": [], "mse": [], "ngc": [], "ridge": []}

    for epoch in range(1, config.max_epochs + 1):
        totals = {key: torch.zeros((), device=device) for key in history}
        n_batches = 0
        model.train()
        for batch_idx in _iter_batches(
            predictors.shape[0],
            config.batch_size,
            config.shuffle,
            rng,
            device,
        ):
            inputs = predictors[batch_idx]
            targets = responses[batch_idx]
            preds = model(inputs)
            mse = criterion(preds, targets)
            ridge = _ridge_penalty(model, config.ridge_lambda)
            ngc_penalty = _cmlp_regularize(model, config)
            # Neural-GC optimizes the sum of the p target-wise MSE losses.
            # ``criterion`` averages over targets, so rescale it to preserve
            # the reference objective and the published lambda scale.
            smooth_loss = mse * model.num_vars + ridge
            if config.optimizer == "ista":
                loss = smooth_loss
                logged_loss = (loss + ngc_penalty.detach()) / model.num_vars
            else:
                loss = smooth_loss + ngc_penalty
                logged_loss = loss / model.num_vars

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if config.optimizer == "ista":
                _prox_cmlp_(model, config)
                ngc_penalty = _cmlp_regularize(model, config)
                logged_loss = (
                    mse.detach() * model.num_vars + ridge.detach() + ngc_penalty.detach()
                ) / model.num_vars

            totals["loss"] = totals["loss"] + logged_loss.detach()
            totals["mse"] = totals["mse"] + mse.detach()
            totals["ngc"] = totals["ngc"] + ngc_penalty.detach() / model.num_vars
            totals["ridge"] = totals["ridge"] + ridge.detach() / model.num_vars
            n_batches += 1

        metrics = {key: float((value / max(n_batches, 1)).detach().cpu()) for key, value in totals.items()}
        for key, value in metrics.items():
            history[key].append(value)
        _log_epoch(epoch, config, metrics, model)

    with torch.no_grad():
        strength = model.gc_strength(ignore_lag=True).detach().cpu().numpy()
        graph = model.causal_graph(config.causal_threshold).detach().cpu().numpy()

    return CMLPFitResult(
        model=model,
        history=history,
        coeffs=None,
        causal_strength=strength,
        causal_graph=graph,
    )
