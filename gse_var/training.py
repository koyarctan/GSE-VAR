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
from .regularizers import ExogenousGroupRegularizer, NGCRegularizer, RegularizerName

SmoothnessMode = Literal["absolute", "relative"]


@dataclass(frozen=True)
class GSEVARTrainingConfig:
    """Edge-wise sparse group lasso with an unpenalized intercept.

    Each target-source pair groups all its lags. Penalty strengths are
    user-supplied; self-edges receive the same penalty as other edges.
    """
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
    gate_init: float = 1.0
    seed: int | None = 42
    shuffle: bool = True
    strength_aggregation: AggregationName = "max"
    smoothness_mode: SmoothnessMode = "absolute"
    smoothness_eps: float = 1e-8
    verbose: int = 1
    log_every: int = 10
    device: str | torch.device | None = None
    gate_learning_rate: float | None = field(default=None, kw_only=True)


@dataclass
class GSEVARFitResult:
    model: GSEVAR
    history: dict[str, list[float]] = field(default_factory=dict)
    coeffs: np.ndarray | None = None
    causal_strength: np.ndarray | None = None
    causal_graph: np.ndarray | None = None

    @property
    def intercept(self) -> np.ndarray:
        return self.model.intercept.detach().cpu().numpy().copy()


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
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if config.gate_learning_rate is not None and (
        not math.isfinite(config.gate_learning_rate) or config.gate_learning_rate <= 0
    ):
        raise ValueError("gate_learning_rate must be finite and positive")
    for name in (
        "lambda_ngc", "sparse_group_lambda", "sparse_l1_lambda",
        "lambda_smooth", "coefficient_weight_decay",
    ):
        value = getattr(config, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
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
    return _make_gated_optimizer(config, model, gate_names=("causal_gate",))


def _make_gated_optimizer(
    config: GSEVARTrainingConfig,
    model: nn.Module,
    *,
    gate_names: tuple[str, ...],
) -> torch.optim.Optimizer:
    """Adam for coefficients/intercept; gate groups only hold proximal metadata."""
    gate_params = []
    coeff_params = []
    intercept_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in gate_names:
            gate_params.append(param)
        elif name == "intercept":
            intercept_params.append(param)
        else:
            coeff_params.append(param)

    param_groups = [
        {
            "params": coeff_params,
            "weight_decay": 0.0,
            "objective_weight_decay": config.coefficient_weight_decay,
        },
        {"params": gate_params, "weight_decay": 0.0, "objective_weight_decay": 0.0},
        {"params": intercept_params, "weight_decay": 0.0, "objective_weight_decay": 0.0},
    ]
    if config.gate_learning_rate is not None:
        param_groups[1]["lr"] = config.gate_learning_rate
    return torch.optim.Adam(param_groups, lr=config.learning_rate)


def _make_endogenous_regularizer(
    config: GSEVARTrainingConfig,
) -> NGCRegularizer:
    return NGCRegularizer(
        name=config.regularizer, lam=config.lambda_ngc, reduction="sum", lag_dim=0,
        sparse_l1_lambda=config.sparse_l1_lambda,
        sparse_group_lambda=config.sparse_group_lambda,
    )


_OPTIMIZATION_METRICS = (
    "weight_decay", "coefficient_step_size", "gate_step_size",
    "coefficient_gradient_norm", "gate_proximal_gradient_norm",
)


def _weight_decay_penalty(
    optimizer: torch.optim.Optimizer,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Explicit coefficient L2; Adam's built-in decay stays disabled."""
    penalty = reference.new_zeros(())
    for group in optimizer.param_groups:
        decay = group["objective_weight_decay"]
        if decay:
            for parameter in group["params"]:
                penalty = penalty + 0.5 * decay * parameter.square().sum()
    return penalty


def _smooth_objective(metrics: dict[str, torch.Tensor]) -> torch.Tensor:
    return metrics["mse"] + metrics["smooth"] + metrics["jacobian"] + metrics["weight_decay"]


def _alternating_proximal_step(
    optimizer: torch.optim.Optimizer,
    evaluate: Callable[[bool], dict[str, torch.Tensor]],
    gate_regularizers: tuple[
        tuple[torch.Tensor, NGCRegularizer | ExogenousGroupRegularizer], ...
    ],
) -> dict[str, torch.Tensor]:
    """Adam coefficient/intercept update, then Euclidean gate prox.

    Evaluate once per active block to get its gradient; never evaluate trial
    steps or change learning rates. Log all loss components from the initial
    same-batch state, without an extra forward pass solely for logging.
    """
    coefficient_group, gate_group, intercept_group = optimizer.param_groups
    blocks = ((coefficient_group, intercept_group), (gate_group,))
    metrics = evaluate(True)
    logged = {name: value.detach() for name, value in metrics.items()}
    diagnostics = {name: metrics["loss"].new_zeros(()) for name in _OPTIMIZATION_METRICS[1:]}
    for block_index, groups in enumerate(blocks):
        parameters = [parameter for group in groups for parameter in group["params"]]
        if not parameters:
            continue
        smooth = _smooth_objective(metrics)
        gradients = (torch.autograd.grad(smooth, parameters, allow_unused=True)
                     if smooth.requires_grad else (None,) * len(parameters))
        gradients = [torch.zeros_like(parameter) if gradient is None else gradient.detach()
                     for parameter, gradient in zip(parameters, gradients)]
        finite = torch.stack([torch.isfinite(metrics["loss"])] +
                             [torch.isfinite(gradient).all() for gradient in gradients]).all()
        if not finite:
            raise FloatingPointError("non-finite loss or gradient in fixed-step proximal update")
        step_size = groups[0]["lr"]
        with torch.no_grad():
            # Only the small gate block needs a copy to measure the proximal
            # residual. No full-network copies or FP64 reductions are needed.
            originals = [parameter.clone() for parameter in parameters] if block_index == 1 else []
            if block_index == 0:
                optimizer.zero_grad(set_to_none=True)
                for parameter, gradient in zip(parameters, gradients):
                    parameter.grad = gradient
                optimizer.step()  # Gate gradients are None: no Adam gate update.
                optimizer.zero_grad(set_to_none=True)
            else:
                for parameter, gradient in zip(parameters, gradients):
                    parameter.add_(gradient, alpha=-step_size)
            if block_index == 1:
                for gate, regularizer in gate_regularizers:
                    if gate.requires_grad:
                        regularizer.prox_(gate, step_size, nonnegative=True)
                residual = sum((parameter - original).square().sum()
                               for parameter, original in zip(parameters, originals)).sqrt() / step_size
            else:
                residual = torch.linalg.vector_norm(torch.stack([
                    torch.linalg.vector_norm(gradient) for gradient in gradients
                ]))
        name = "coefficient" if block_index == 0 else "gate"
        diagnostics[f"{name}_step_size"] = metrics["loss"].new_tensor(step_size)
        residual_name = ("coefficient_gradient_norm" if block_index == 0
                         else "gate_proximal_gradient_norm")
        diagnostics[residual_name] = residual
        if block_index == 0 and gate_group["params"]:
            metrics = evaluate(True)
    logged.update(diagnostics)
    optimizer.zero_grad(set_to_none=True)
    return logged


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
        name: torch.zeros((), device=device)
        for name in ("loss", "mse", "ngc", "smooth", "jacobian", *_OPTIMIZATION_METRICS)
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
        if model.causal_gate is None:
            raise RuntimeError("NGC regularization requires use_causal_gate=True")

        def evaluate(create_graph: bool) -> dict[str, torch.Tensor]:
            with torch.set_grad_enabled(create_graph):
                preds, coeffs = model(inputs)
                mse = criterion(preds, targets)
                smooth = (config.lambda_smooth * temporal_smoothness_penalty(
                    coeffs, dataset.time_index[batch_idx],
                    mode=config.smoothness_mode, eps=config.smoothness_eps,
                ) if use_smoothness else mse.new_zeros(()))
                if use_jacobian:
                    # Preserve the gate-detached Jacobian regularizer.
                    mismatch = model.coefficient_jacobian_mismatch_for_regularization(
                        inputs, create_graph=create_graph,
                    )
                    jacobian = config.lambda_jacobian * mismatch.pow(2).mean()
                else:
                    jacobian = mse.new_zeros(())
                penalty = ngc.penalty(model.causal_gate)
                decay = (_weight_decay_penalty(optimizer, mse) if optimizer is not None
                         else config.coefficient_weight_decay * 0.5 * sum(
                             (parameter.square().sum() for parameter in model.coeff_net.parameters()
                              if parameter.requires_grad), mse.new_zeros(())))
                return dict(loss=mse + smooth + jacobian + penalty + decay,
                            mse=mse, smooth=smooth, jacobian=jacobian,
                            ngc=penalty, weight_decay=decay)

        if train:
            metrics = _alternating_proximal_step(
                optimizer, evaluate, ((model.causal_gate, ngc),),
            )
        else:
            metrics = evaluate(False)
        for name in totals:
            totals[name] += metrics.get(name, totals[name].new_zeros(())).detach()
        n_batches += 1

    return {
        key: float((value / max(n_batches, 1)).detach().cpu())
        for key, value in totals.items()
    }


@torch.no_grad()
def _gate_usage(
    model: GSEVAR,
) -> tuple[float, int, int]:
    if model.causal_gate is None:
        return 0.0, 0, 0
    graph = model.causal_graph_from_gate()
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

    usage_pct, active_edges, total_edges = _gate_usage(model)
    log_fn(
        f"Epoch {epoch:>4}/{config.max_epochs:<4} | "
        f"loss={metrics['loss']:.6g} | "
        f"mse={metrics['mse']:.6g} | "
        f"ngc={metrics['ngc']:.6g} | "
        f"smooth={metrics['smooth']:.6g} | "
        f"jacobian={metrics['jacobian']:.6g} | "
        f"l2={metrics['weight_decay']:.6g} | "
        f"gate_step={metrics['gate_step_size']:.3g} | "
        f"gate_residual={metrics['gate_proximal_gradient_norm']:.3g} | "
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
    graph_t = model.causal_graph_from_gate()
    return coeffs_t.numpy(), strength_t.numpy(), graph_t.cpu().numpy()


def fit_gse_var(
    data: np.ndarray | list[np.ndarray],
    config: GSEVARTrainingConfig,
    model: GSEVAR | None = None,
    *,
    epoch_callback: Callable[[int, GSEVAR, dict[str, float]], None] | None = None,
) -> GSEVARFitResult:
    """Fit GSE-VAR; optional callbacks observe the post-update epoch state.

    Adam updates coefficients/intercept; a nonnegative proximal step updates
    gates. The algorithm and zero-only graph selection are fixed, not options.
    Validation checkpoint selection remains the caller's responsibility.
    """
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
    if model.intercept is None:
        raise ValueError("GSE-VAR requires an intercept")
    # A caller may pass a manually modified model. Project before the first
    # forward pass so negative gates can never alter input signs.
    model.project_causal_gate_()

    ngc = _make_endogenous_regularizer(config)
    optimizer = _make_optimizer(config, model)
    criterion = nn.MSELoss(reduction="mean")
    rng = np.random.default_rng(config.seed)
    history: dict[str, list[float]] = {
        name: [] for name in ("loss", "mse", "ngc", "smooth", "jacobian", *_OPTIMIZATION_METRICS)
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
        if epoch_callback is not None:
            epoch_callback(epoch, model, metrics)

    coeffs, strength, graph = _infer(model, dataset, config, device)
    return GSEVARFitResult(
        model=model,
        history=history,
        coeffs=coeffs,
        causal_strength=strength,
        causal_graph=graph,
    )
