"""Bernoulli structure learning by DisARM and a joint Adam update."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal
import math
import numpy as np
import torch
from torch import nn
from .data import LaggedDataset, construct_lagged_dataset
from .models import AggregationName, GSEVAR
from .disarm import _antithetic_masks, _disarm_gradient

SmoothnessMode = Literal["absolute", "relative"]

@dataclass(frozen=True)
class GSEVARTrainingConfig:
    """DisARM plus ``lambda_l0 * sum(sigmoid(gate_logits))``.

    ``learning_rate`` controls coefficients. ``gate_learning_rate=None`` keeps
    the original shared rate; a positive value sets a separate logit rate.
    Adam updates both parameter blocks jointly.
    Self-routes have the same selection cost as other routes. No temperature,
    annealing schedule, or post-selection refit is used.
    """

    order: int
    hidden_layer_size: int
    num_hidden_layers: int = 1
    max_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    lambda_l0: float = 1e-3
    lambda_smooth: float = 0.0
    lambda_jacobian: float = 0.0
    coefficient_weight_decay: float = 0.0
    gate_probability_init: float = 0.9
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
    for name in ("order", "hidden_layer_size", "num_hidden_layers", "max_epochs",
                 "batch_size", "log_every"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("learning_rate", "smoothness_eps"):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if config.gate_learning_rate is not None and (
        not math.isfinite(config.gate_learning_rate) or config.gate_learning_rate <= 0
    ):
        raise ValueError("gate_learning_rate must be finite and positive")
    for name in ("lambda_l0", "lambda_smooth", "lambda_jacobian", "coefficient_weight_decay"):
        value = getattr(config, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not math.isfinite(config.gate_probability_init) or not 0 < config.gate_probability_init < 1:
        raise ValueError("gate_probability_init must be strictly between 0 and 1")
    if not isinstance(config.shuffle, bool):
        raise ValueError("shuffle must be a bool")
    if config.smoothness_mode not in ("absolute", "relative"):
        raise ValueError("unsupported smoothness_mode")
    if config.strength_aggregation not in ("max", "mean", "median"):
        raise ValueError("unsupported strength_aggregation")


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


def _smoothness_losses(coefficients, times, mode, eps):
    """Original scalar penalty and valid loss differences for each target.

    Absolute smoothness is additive across targets. Relative smoothness has a
    shared, mask-dependent denominator, so every target must retain the full
    relative penalty in its score estimator, not just its numerator's share.
    """
    num_targets = coefficients.shape[2]
    zero = coefficients.new_zeros(())
    if len(coefficients) < 2:
        return zero, zero.expand(num_targets)
    order = torch.argsort(times)
    sorted_coefficients = coefficients[order]
    adjacent = (times[order][1:] - times[order][:-1]) == 1
    if not torch.any(adjacent):
        return zero, zero.expand(num_targets)
    previous = sorted_coefficients[:-1][adjacent]
    squared_differences = (sorted_coefficients[1:][adjacent] - previous).square()
    if mode == "relative":
        denominator = previous.square().mean(dim=(1, 2, 3), keepdim=True) + eps
        penalty = (squared_differences / denominator).mean()
        return penalty, penalty.detach().expand(num_targets)
    if mode != "absolute":
        raise ValueError(f"unsupported smoothness mode: {mode}")
    target_losses = squared_differences.detach().mean(dim=(0, 1, 3)) / num_targets
    return squared_differences.mean(), target_losses


def _masked_losses(model, inputs, targets, times, mask, config):
    """Unchanged scalar objectives plus detached per-target score losses.

    Each target's MSE/absolute-smoothness/Jacobian contribution includes 1/p,
    preserving the original mean over all targets instead of amplifying the
    prediction-loss gradient relative to the expected-L0 selection cost.
    """
    prediction, coefficients, mismatch = model._masked_forward(
        inputs, mask, need_mismatch=config.lambda_jacobian != 0,
    )
    squared_errors = (prediction - targets).square()
    mse = squared_errors.mean()
    target_losses = squared_errors.detach().mean(dim=0) / model.num_vars
    smooth = mse.new_zeros(())
    if config.lambda_smooth:
        penalty, target_penalties = _smoothness_losses(
            coefficients, times, config.smoothness_mode, config.smoothness_eps,
        )
        smooth = config.lambda_smooth * penalty
        target_losses = target_losses + config.lambda_smooth * target_penalties
    jacobian = mse.new_zeros(())
    if mismatch is not None:
        squared_mismatch = mismatch.square()
        jacobian = config.lambda_jacobian * squared_mismatch.mean()
        target_losses = target_losses + config.lambda_jacobian * (
            squared_mismatch.detach().mean(dim=(0, 1, 3)) / model.num_vars
        )
    return dict(mse=mse, smooth=smooth, jacobian=jacobian), target_losses


def _make_optimizer(config, model):
    logits = set(model.gate_logits_by_branch.values())
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("fitting requires trainable parameters")
    groups = [{"params": [p for p in parameters if p not in logits]},
              {"params": [p for p in parameters if p in logits],
               "lr": config.gate_learning_rate or config.learning_rate}]
    return torch.optim.Adam(groups, lr=config.learning_rate)


def _adam_disarm_step(model, evaluate, costs, config, optimizer, generator):
    """Evaluate two masks at unchanged parameters, then perform one Adam step.

    ``evaluate`` returns scalar loss components and target-local score losses.
    Coupled penalties must contribute their full value to each target score.
    """
    logits = model.gate_logits_by_branch
    pairs = {name: _antithetic_masks(value, generator) for name, value in logits.items()}
    optimizer.zero_grad(set_to_none=True)
    first, first_targets = evaluate({name: pair[0] for name, pair in pairs.items()})
    first_loss = sum(first.values())
    if first_loss.requires_grad:
        (0.5 * first_loss).backward()
    second, second_targets = evaluate({name: pair[1] for name, pair in pairs.items()})
    second_loss = sum(second.values())
    gate_parameters = set(logits.values())
    decay = 0.5 * config.coefficient_weight_decay * sum(
        (p.square().sum() for name, p in model.named_parameters()
         if p.requires_grad and p not in gate_parameters and name != "intercept"),
        second_loss.new_zeros(()))
    if (second_loss + decay).requires_grad:
        (0.5 * second_loss + decay).backward()
    with torch.no_grad():
        metrics = {key: .5 * (first[key].detach() + second[key].detach()) for key in first}
        total_cost = second_loss.new_zeros(())
        expected = second_loss.new_zeros(())
        norm_squared = second_loss.new_zeros(())
        for name, value in logits.items():
            probabilities = value.sigmoid()
            cost = costs[name] * probabilities.sum()
            gradient = _disarm_gradient(value, *pairs[name],
                                       first_targets[None, :, None], second_targets[None, :, None])
            gradient += costs[name] * probabilities * (1 - probabilities)
            if value.requires_grad:
                value.grad = gradient
            metrics["l0_" + name] = cost
            total_cost += cost
            expected += probabilities.sum()
            norm_squared += gradient.square().sum()
        metrics.update(weight_decay=decay.detach(), l0=total_cost,
                       expected_lag_edges=expected, gate_gradient_norm=norm_squared.sqrt())
        metrics["loss"] = sum(metrics[k] for k in ("mse", "smooth", "jacobian", "weight_decay", "l0"))
        checks = [torch.isfinite(value).all() for value in metrics.values()]
        checks += [torch.isfinite(p).all() for p in model.parameters()]
        checks += [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
        if not torch.stack(checks).all():
            raise FloatingPointError("non-finite DisARM loss or gradient")
    optimizer.step()
    return metrics


def _disarm_step(model, inputs, targets, times, config, optimizer, generator):
    metrics = _adam_disarm_step(
        model, lambda masks: _masked_losses(model, inputs, targets, times, masks["endogenous"], config),
        {"endogenous": config.lambda_l0}, config, optimizer, generator)
    model.update_causal_gate_()
    return metrics


def fit_gse_var(
    data: np.ndarray | list[np.ndarray],
    config: GSEVARTrainingConfig,
    model: GSEVAR | None = None,
    *,
    epoch_callback: Callable[[int, GSEVAR, dict[str, float]], None] | None = None,
) -> GSEVARFitResult:
    """Fit binary structural gates using expected L0 regularization and DisARM.

    Callbacks run after each epoch with the refreshed deterministic mask. They
    can validate/save checkpoints without resetting optimizer or sampling RNG.
    Recorded training losses are pre-update two-mask Monte Carlo estimates;
    returned coefficients and graph are from the final fixed binary mask.
    """
    _validate_config(config)
    if config.seed is not None:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
    dataset_np = construct_lagged_dataset(data, order=config.order)
    device = _resolve_device(config.device)
    num_vars = dataset_np.predictors.shape[-1]
    if model is None:
        model = GSEVAR(num_vars, config.order, config.hidden_layer_size,
                            config.num_hidden_layers, config.gate_probability_init)
    if not isinstance(model, GSEVAR):
        raise ValueError("fit_gse_var requires a GSEVAR model")
    if (model.num_vars, model.order) != (num_vars, config.order):
        raise ValueError("model dimensions/order do not match data and config")
    if model.intercept is None:
        raise ValueError("DisARM requires an intercept")
    model.to(device)
    model.update_causal_gate_()
    optimizer = _make_optimizer(config, model)
    generator = torch.Generator(device=device)
    generator.seed() if config.seed is None else generator.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    dataset = _to_torch_dataset(dataset_np, device)
    history = {name: [] for name in ("loss", "mse", "smooth", "jacobian", "weight_decay",
                                    "l0", "expected_lag_edges", "gate_gradient_norm")}
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        totals = {name: torch.zeros((), device=device) for name in history}
        samples = 0
        for indices in _iter_batches(len(dataset.predictors), config.batch_size,
                                     config.shuffle, rng, device):
            inputs, targets = dataset.predictors[indices], dataset.responses[indices]
            metrics = _disarm_step(model, inputs, targets, dataset.time_index[indices],
                                   config, optimizer, generator)
            for name in totals:
                totals[name] += metrics[name].detach() * len(inputs)
            samples += len(inputs)
        record = {name: float((value / samples).cpu()) for name, value in totals.items()}
        for name, value in record.items():
            history[name].append(value)
        if config.verbose and (epoch == 1 or epoch == config.max_epochs or epoch % config.log_every == 0):
            active = int(model.causal_graph_from_gate().sum().cpu())
            print(f"Epoch {epoch:>4}/{config.max_epochs:<4} | loss={record['loss']:.6g} | "
                  f"mse={record['mse']:.6g} | l0={record['l0']:.6g} | "
                  f"smooth={record['smooth']:.6g} | jacobian={record['jacobian']:.6g} | "
                  f"l2={record['weight_decay']:.6g} | expected_lag_edges={record['expected_lag_edges']:.2f} | "
                  f"active_edges={active}/{num_vars ** 2}", flush=True)
        if epoch_callback is not None:
            epoch_callback(epoch, model, record)
    model.eval()
    with torch.no_grad():
        coefficients = torch.cat([model(dataset.predictors[start:start + config.batch_size])[1].cpu()
                                  for start in range(0, len(dataset.predictors), config.batch_size)])
        strength = model.coefficient_strength(coefficients, config.strength_aggregation).numpy()
        graph = model.causal_graph_from_gate().cpu().numpy()
    return GSEVARFitResult(model=model, history=history, coeffs=coefficients.numpy(),
                           causal_strength=strength, causal_graph=graph)
