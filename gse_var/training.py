from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from typing import Literal

import numpy as np
import torch
from torch import nn

from .data import LaggedDataset, construct_lagged_dataset
from .models import AggregationName, GSEVAR
from .regularizers import NGCRegularizer, RegularizerName

OptimizerName = Literal["adam", "ista"]
SmoothnessMode = Literal["absolute", "relative"]


@dataclass(frozen=True)
class GSEVARTrainingConfig:
    order: int
    hidden_layer_size: int
    num_hidden_layers: int = 1
    max_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    lambda_ngc: float = 0.0
    lambda_smooth: float = 0.0
    lambda_jacobian: float = 0.0
    coefficient_weight_decay: float = 0.0
    regularizer: RegularizerName = "sparse_group_lasso"
    sparse_group_lambda: float = 0.0
    sparse_l1_lambda: float = 0.0
    optimizer: OptimizerName = "ista"
    gate_init: float = 1.0
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
class GSEVARFitResult:
    model: GSEVAR
    history: dict[str, list[float]] = field(default_factory=dict)
    coeffs: np.ndarray | None = None
    causal_strength: np.ndarray | None = None
    causal_graph: np.ndarray | None = None


@dataclass(frozen=True)
class _TorchLaggedDataset:
    predictors: torch.Tensor
    responses: torch.Tensor
    time_index: torch.Tensor
    series_index: torch.Tensor


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _validate_config(config: GSEVARTrainingConfig) -> None:
    if config.lambda_ngc < 0:
        raise ValueError("lambda_ngc must be non-negative")
    if config.sparse_group_lambda < 0:
        raise ValueError("sparse_group_lambda must be non-negative")
    if config.sparse_l1_lambda < 0:
        raise ValueError("sparse_l1_lambda must be non-negative")
    if not math.isfinite(config.lambda_jacobian) or config.lambda_jacobian < 0:
        raise ValueError("lambda_jacobian must be non-negative")
    if not math.isfinite(config.gate_init) or config.gate_init < 0:
        raise ValueError("gate_init must be non-negative")
    if config.regularizer == "sparse_group_lasso":
        if config.lambda_ngc != 0:
            raise ValueError(
                "sparse_group_lasso does not use lambda_ngc. "
                "Use sparse_group_lambda and sparse_l1_lambda instead."
            )
    elif config.sparse_group_lambda != 0 or config.sparse_l1_lambda != 0:
        raise ValueError(
            "sparse_group_lambda and sparse_l1_lambda are only used with "
            "regularizer='sparse_group_lasso'. Use lambda_ngc for group_lasso "
            "or hierarchical_group_lasso."
        )


def _make_optimizer(
    config: GSEVARTrainingConfig,
    model: nn.Module,
) -> torch.optim.Optimizer:
    gate_params = []
    coeff_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name == "causal_gate":
            gate_params.append(param)
        else:
            coeff_params.append(param)

    param_groups = [
        {
            "params": coeff_params,
            "weight_decay": config.coefficient_weight_decay,
        },
        {"params": gate_params, "weight_decay": 0.0},
    ]
    if config.optimizer == "adam":
        return torch.optim.Adam(param_groups, lr=config.learning_rate)
    if config.optimizer == "ista":
        return torch.optim.SGD(param_groups, lr=config.learning_rate)
    raise ValueError(f"unsupported optimizer: {config.optimizer}")


def _to_torch_dataset(
    dataset: LaggedDataset,
    device: torch.device,
) -> _TorchLaggedDataset:
    return _TorchLaggedDataset(
        predictors=torch.as_tensor(
            dataset.predictors,
            dtype=torch.float32,
            device=device,
        ),
        responses=torch.as_tensor(
            dataset.responses,
            dtype=torch.float32,
            device=device,
        ),
        time_index=torch.as_tensor(
            dataset.time_index,
            dtype=torch.long,
            device=device,
        ),
        series_index=torch.as_tensor(
            dataset.series_index,
            dtype=torch.long,
            device=device,
        ),
    )


def _iter_batches(
    n_samples: int,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
    device: torch.device,
):
    if not shuffle:
        for start in range(0, n_samples, batch_size):
            yield slice(start, start + batch_size)
        return

    indices = np.arange(n_samples)
    if shuffle:
        rng.shuffle(indices)
    indices_t = torch.as_tensor(indices, dtype=torch.long, device=device)
    for start in range(0, n_samples, batch_size):
        yield indices_t[start : start + batch_size]


def temporal_smoothness_penalty(
    coeffs: torch.Tensor,
    time_index: torch.Tensor,
    mode: SmoothnessMode = "absolute",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalty for adjacent coefficient changes within the same continuous run."""
    if coeffs.shape[0] < 2:
        return coeffs.new_zeros(())

    order = torch.argsort(time_index)
    sorted_t = time_index[order]
    sorted_coeffs = coeffs[order]
    adjacent = (sorted_t[1:] - sorted_t[:-1]) == 1
    if not torch.any(adjacent):
        return coeffs.new_zeros(())

    diffs = sorted_coeffs[1:][adjacent] - sorted_coeffs[:-1][adjacent]
    if mode == "absolute":
        return torch.mean(diffs.pow(2))
    if mode == "relative":
        denom = (
            sorted_coeffs[:-1][adjacent]
            .pow(2)
            .mean(dim=(1, 2, 3), keepdim=True)
            + eps
        )
        return torch.mean(diffs.pow(2) / denom)
    raise ValueError(f"unsupported smoothness mode: {mode}")


def _epoch(
    model: GSEVAR,
    dataset: _TorchLaggedDataset,
    config: GSEVARTrainingConfig,
    optimizer: torch.optim.Optimizer | None,
    ngc: NGCRegularizer,
    criterion: nn.Module,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals = {
        "loss": torch.zeros((), device=device),
        "mse": torch.zeros((), device=device),
        "ngc": torch.zeros((), device=device),
        "smooth": torch.zeros((), device=device),
        "jacobian": torch.zeros((), device=device),
    }
    n_batches = 0
    use_smoothness = config.lambda_smooth != 0
    use_jacobian = config.lambda_jacobian != 0

    for batch_idx in _iter_batches(
        dataset.predictors.shape[0],
        config.batch_size,
        shuffle=config.shuffle and train,
        rng=rng,
        device=device,
    ):
        inputs = dataset.predictors[batch_idx]
        targets = dataset.responses[batch_idx]
        if use_jacobian:
            # Prediction and structural sparsity retain their normal gate
            # gradients.  Only the Jacobian regularizer treats the gate as
            # fixed, preventing edge deletion from being a shortcut to J=C.
            preds, coeffs = model(inputs)
            mismatch = model.coefficient_jacobian_mismatch_for_regularization(
                inputs,
                create_graph=train,
            )
            jacobian = config.lambda_jacobian * mismatch.pow(2).mean()
        else:
            preds, coeffs = model(inputs)
            jacobian = preds.new_zeros(())
        mse = criterion(preds, targets)

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

        if model.causal_gate is None:
            raise RuntimeError("NGC regularization requires use_causal_gate=True")
        ngc_penalty = ngc.penalty(model.causal_gate)

        if config.optimizer == "ista" and train:
            loss = mse + smooth + jacobian
            logged_loss = loss + ngc_penalty.detach()
        else:
            loss = mse + smooth + jacobian + ngc_penalty
            logged_loss = loss

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if config.optimizer == "ista":
                if model.causal_gate is None:
                    raise RuntimeError("ISTA requires use_causal_gate=True")
                ngc.prox_(
                    model.causal_gate,
                    config.learning_rate,
                    nonnegative=True,
                )
                ngc_penalty = ngc.penalty(model.causal_gate)
                logged_loss = (
                    mse.detach()
                    + smooth.detach()
                    + jacobian.detach()
                    + ngc_penalty.detach()
                )
            else:
                # Adam optimizes the regularized objective directly. Projection
                # maintains the structural interpretation of the gate without
                # replacing the exact-zero role of ISTA.
                model.project_causal_gate_()
                ngc_penalty = ngc.penalty(model.causal_gate)
                logged_loss = (
                    mse.detach()
                    + smooth.detach()
                    + jacobian.detach()
                    + ngc_penalty.detach()
                )

        totals["loss"] = totals["loss"] + logged_loss.detach()
        totals["mse"] = totals["mse"] + mse.detach()
        totals["ngc"] = totals["ngc"] + ngc_penalty.detach()
        totals["smooth"] = totals["smooth"] + smooth.detach()
        totals["jacobian"] = totals["jacobian"] + jacobian.detach()
        n_batches += 1

    return {
        key: float((value / max(n_batches, 1)).detach().cpu())
        for key, value in totals.items()
    }


@torch.no_grad()
def _gate_usage(
    model: GSEVAR,
    threshold: float,
) -> tuple[float, int, int]:
    if model.causal_gate is None:
        return 0.0, 0, 0
    graph = model.causal_graph_from_gate(threshold=threshold)
    active = int(graph.sum().detach().cpu())
    total = graph.numel()
    return 100.0 * active / max(total, 1), active, total


def _log_epoch(
    epoch: int,
    config: GSEVARTrainingConfig,
    metrics: dict[str, float],
    model: GSEVAR,
    log_fn: Callable[[str], None] = print,
) -> None:
    if config.verbose <= 0:
        return
    log_every = max(config.log_every, 1)
    if epoch != 1 and epoch != config.max_epochs and epoch % log_every != 0:
        return

    usage_pct, active_edges, total_edges = _gate_usage(
        model,
        config.causal_threshold,
    )
    log_fn(
        f"Epoch {epoch:>4}/{config.max_epochs:<4} | "
        f"loss={metrics['loss']:.6g} | "
        f"mse={metrics['mse']:.6g} | "
        f"ngc={metrics['ngc']:.6g} | "
        f"smooth={metrics['smooth']:.6g} | "
        f"jacobian={metrics['jacobian']:.6g} | "
        f"active_edges={active_edges}/{total_edges} ({usage_pct:.2f}%)"
    )


@torch.no_grad()
def _infer(
    model: GSEVAR,
    dataset: _TorchLaggedDataset,
    config: GSEVARTrainingConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    coeffs_all = []
    for start in range(0, dataset.predictors.shape[0], config.batch_size):
        stop = start + config.batch_size
        inputs = dataset.predictors[start:stop]
        _, coeffs = model(inputs)
        coeffs_all.append(coeffs.cpu())

    coeffs_t = torch.cat(coeffs_all, dim=0)
    strength_t = model.coefficient_strength(
        coeffs_t,
        aggregation=config.strength_aggregation,
    )
    if model.causal_gate is not None:
        graph_t = model.causal_graph_from_gate(
            threshold=config.causal_threshold,
        )
    else:
        graph_t = (strength_t > config.causal_threshold).to(torch.int64)
    return coeffs_t.numpy(), strength_t.numpy(), graph_t.cpu().numpy()


def fit_gse_var(
    data: np.ndarray | list[np.ndarray],
    config: GSEVARTrainingConfig,
    model: GSEVAR | None = None,
) -> GSEVARFitResult:
    """Fit GSE-VAR with gated, structured Granger sparsity."""
    _validate_config(config)
    if config.seed is not None:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    dataset_np = construct_lagged_dataset(data, order=config.order)
    device = _resolve_device(config.device)
    num_vars = dataset_np.predictors.shape[-1]

    if model is None:
        model = GSEVAR(
            num_vars=num_vars,
            order=config.order,
            hidden_layer_size=config.hidden_layer_size,
            num_hidden_layers=config.num_hidden_layers,
            use_causal_gate=True,
            gate_init=config.gate_init,
        )
    model.to(device)

    if model.causal_gate is None:
        raise ValueError("fit_gse_var requires a model with use_causal_gate=True")
    if model.order != config.order:
        raise ValueError(
            f"model order {model.order} does not match config order {config.order}"
        )
    if model.num_vars != num_vars:
        raise ValueError(
            f"model num_vars {model.num_vars} does not match data num_vars {num_vars}"
        )
    # A caller may pass a manually modified model. Project before the first
    # forward pass so negative gates can never alter input signs.
    model.project_causal_gate_()

    ngc = NGCRegularizer(
        name=config.regularizer,
        lam=config.lambda_ngc,
        reduction="sum",
        lag_dim=0,
        sparse_l1_lambda=config.sparse_l1_lambda,
        sparse_group_lambda=config.sparse_group_lambda,
    )
    optimizer = _make_optimizer(config, model)
    criterion = nn.MSELoss(reduction="mean")
    rng = np.random.default_rng(config.seed)
    history: dict[str, list[float]] = {
        "loss": [],
        "mse": [],
        "ngc": [],
        "smooth": [],
        "jacobian": [],
    }
    dataset = _to_torch_dataset(dataset_np, device)

    for epoch in range(1, config.max_epochs + 1):
        metrics = _epoch(
            model=model,
            dataset=dataset,
            config=config,
            optimizer=optimizer,
            ngc=ngc,
            criterion=criterion,
            device=device,
            rng=rng,
        )
        for key, value in metrics.items():
            history[key].append(value)
        _log_epoch(epoch, config, metrics, model)

    coeffs, strength, graph = _infer(model, dataset, config, device)
    return GSEVARFitResult(
        model=model,
        history=history,
        coeffs=coeffs,
        causal_strength=strength,
        causal_graph=graph,
    )
