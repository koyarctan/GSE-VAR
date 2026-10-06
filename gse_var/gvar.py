from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .data import construct_lagged_dataset
from .models import AggregationName, GSEVAR
from .training import (
    SmoothnessMode,
    _iter_batches,
    _resolve_device,
    _to_torch_dataset,
    temporal_smoothness_penalty,
)


@dataclass(frozen=True)
class GVARBaselineTrainingConfig:
    order: int
    hidden_layer_size: int
    num_hidden_layers: int = 1
    max_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    lambda_coeff: float = 0.0
    elastic_net_alpha: float = 0.5
    lambda_smooth: float = 0.0
    coefficient_weight_decay: float = 0.0
    seed: int | None = 42
    shuffle: bool = True
    causal_threshold: float = 1e-8
    strength_aggregation: AggregationName = "max"
    smoothness_mode: SmoothnessMode = "absolute"
    smoothness_eps: float = 1e-8
    verbose: int = 1
    log_every: int = 10
    device: str | torch.device | None = None


@dataclass
class GVARBaselineFitResult:
    model: GSEVAR
    history: dict[str, list[float]] = field(default_factory=dict)
    coeffs: np.ndarray | None = None
    causal_strength: np.ndarray | None = None
    causal_graph: np.ndarray | None = None


def _validate_config(config: GVARBaselineTrainingConfig) -> None:
    if config.lambda_coeff < 0:
        raise ValueError("lambda_coeff must be non-negative")
    if config.lambda_smooth < 0:
        raise ValueError("lambda_smooth must be non-negative")
    if config.coefficient_weight_decay < 0:
        raise ValueError("coefficient_weight_decay must be non-negative")
    if not 0 <= config.elastic_net_alpha <= 1:
        raise ValueError("elastic_net_alpha must be between 0 and 1")


def _make_optimizer(
    config: GVARBaselineTrainingConfig,
    model: GSEVAR,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.coefficient_weight_decay,
    )


def _coefficient_penalty(
    coeffs: torch.Tensor,
    lam: float,
    alpha: float,
) -> torch.Tensor:
    if lam == 0:
        return coeffs.new_zeros(())

    l1 = coeffs.abs().mean()
    group = torch.linalg.vector_norm(coeffs, ord=2, dim=1).mean()
    return lam * (alpha * l1 + (1.0 - alpha) * group)


def _epoch(
    model: GSEVAR,
    dataset,
    config: GVARBaselineTrainingConfig,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": torch.zeros((), device=device),
        "mse": torch.zeros((), device=device),
        "coeff": torch.zeros((), device=device),
        "smooth": torch.zeros((), device=device),
    }
    n_batches = 0
    use_smoothness = config.lambda_smooth != 0

    for batch_idx in _iter_batches(
        dataset.predictors.shape[0],
        config.batch_size,
        shuffle=config.shuffle,
        rng=rng,
        device=device,
    ):
        inputs = dataset.predictors[batch_idx]
        targets = dataset.responses[batch_idx]

        preds, coeffs = model(inputs)
        mse = criterion(preds, targets)
        coeff_penalty = _coefficient_penalty(
            coeffs,
            lam=config.lambda_coeff,
            alpha=config.elastic_net_alpha,
        )
        if use_smoothness:
            time_index = dataset.time_index[batch_idx]
            smooth = config.lambda_smooth * temporal_smoothness_penalty(
                coeffs,
                time_index,
                mode=config.smoothness_mode,
                eps=config.smoothness_eps,
            )
        else:
            smooth = mse.new_zeros(())

        loss = mse + coeff_penalty + smooth

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        totals["loss"] = totals["loss"] + loss.detach()
        totals["mse"] = totals["mse"] + mse.detach()
        totals["coeff"] = totals["coeff"] + coeff_penalty.detach()
        totals["smooth"] = totals["smooth"] + smooth.detach()
        n_batches += 1

    return {
        key: float((value / max(n_batches, 1)).detach().cpu())
        for key, value in totals.items()
    }


@torch.no_grad()
def _active_edges_from_coeffs(
    model: GSEVAR,
    coeffs: torch.Tensor,
    config: GVARBaselineTrainingConfig,
) -> tuple[int, int]:
    strength = model.coefficient_strength(coeffs, aggregation=config.strength_aggregation)
    graph = strength > config.causal_threshold
    return int(graph.sum().detach().cpu()), graph.numel()


@torch.no_grad()
def _infer(
    model: GSEVAR,
    dataset,
    config: GVARBaselineTrainingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    coeffs_all = []
    for start in range(0, dataset.predictors.shape[0], config.batch_size):
        stop = start + config.batch_size
        _, coeffs = model(dataset.predictors[start:stop])
        coeffs_all.append(coeffs.detach().cpu())

    coeffs_t = torch.cat(coeffs_all, dim=0)
    strength_t = model.coefficient_strength(coeffs_t, aggregation=config.strength_aggregation)
    graph_t = (strength_t > config.causal_threshold).to(torch.int64)
    return coeffs_t.numpy(), strength_t.numpy(), graph_t.numpy()


def _log_epoch(
    epoch: int,
    config: GVARBaselineTrainingConfig,
    metrics: dict[str, float],
    model: GSEVAR,
    dataset,
    log_fn: Callable[[str], None] = print,
) -> None:
    if config.verbose <= 0:
        return
    log_every = max(config.log_every, 1)
    if epoch != 1 and epoch != config.max_epochs and epoch % log_every != 0:
        return

    with torch.no_grad():
        sample_inputs = dataset.predictors[: config.batch_size]
        _, coeffs = model(sample_inputs)
        active_edges, total_edges = _active_edges_from_coeffs(model, coeffs, config)

    log_fn(
        f"Epoch {epoch:>4}/{config.max_epochs:<4} | "
        f"loss={metrics['loss']:.6g} | "
        f"mse={metrics['mse']:.6g} | "
        f"coeff={metrics['coeff']:.6g} | "
        f"smooth={metrics['smooth']:.6g} | "
        f"active_edges={active_edges}/{total_edges} "
        f"({100.0 * active_edges / max(total_edges, 1):.2f}%)"
    )


def fit_gvar(
    data: np.ndarray | list[np.ndarray],
    config: GVARBaselineTrainingConfig,
    model: GSEVAR | None = None,
) -> GVARBaselineFitResult:
    """Fit the original GVAR-style baseline without NGC causal gates."""
    _validate_config(config)

    if config.seed is not None:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    dataset_np = construct_lagged_dataset(data, order=config.order)
    device = _resolve_device(config.device)
    dataset = _to_torch_dataset(dataset_np, device)

    num_vars = dataset.predictors.shape[-1]
    if model is None:
        model = GSEVAR(
            num_vars=num_vars,
            order=config.order,
            hidden_layer_size=config.hidden_layer_size,
            num_hidden_layers=config.num_hidden_layers,
            use_causal_gate=False,
        )
    elif model.causal_gate is not None:
        raise ValueError("fit_gvar expects a model with use_causal_gate=False")
    model.to(device)

    optimizer = _make_optimizer(config, model)
    criterion = nn.MSELoss(reduction="mean")
    rng = np.random.default_rng(config.seed)
    history: dict[str, list[float]] = {"loss": [], "mse": [], "coeff": [], "smooth": []}

    for epoch in range(1, config.max_epochs + 1):
        metrics = _epoch(
            model=model,
            dataset=dataset,
            config=config,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            rng=rng,
        )
        for key, value in metrics.items():
            history[key].append(value)
        _log_epoch(epoch, config, metrics, model, dataset)

    coeffs, strength, graph = _infer(model, dataset, config)
    return GVARBaselineFitResult(
        model=model,
        history=history,
        coeffs=coeffs,
        causal_strength=strength,
        causal_graph=graph,
    )
