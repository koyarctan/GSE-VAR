"""Bernoulli structural gates trained with DisARM, separate from ISTA.

Dong, Mnih and Tucker (NeurIPS 2020), equations (6)--(8):
https://arxiv.org/abs/2006.10680

Each antithetic mask is shared across the batch's time points and masks both
coefficient-generator inputs and outputs. Inference always uses a fixed mask.
The sparsity cost counts expected *lag routes*, not lag-integrated graph edges.
Target-separable losses use target-local DisARM differences at the original
global-mean scale; only relative smoothness retains its coupled difference.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math

import numpy as np
import torch
from torch import nn

from .data import construct_lagged_dataset
from .models import AggregationName, GSEVAR, _prediction_jacobian_components
from .training import (
    GSEVARFitResult, SmoothnessMode, _iter_batches, _resolve_device,
    _to_torch_dataset,
)


@dataclass(frozen=True)
class GSEVARDisARMTrainingConfig:
    """DisARM plus ``lambda_l0 * sum(sigmoid(gate_logits))``.

    ``learning_rate`` controls coefficients. ``gate_learning_rate=None`` keeps
    the original shared rate; a positive value sets a separate logit rate.
    Adam updates both blocks; no proximal step or line search is used.
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


class GSEVARDisARM(GSEVAR):
    """The existing GSE-VAR architecture with input-independent binary gates.

    ``gate_logits`` are trainable Bernoulli logits. ``causal_gate`` is a saved
    binary buffer, never a learnable amplitude. It is refreshed after updates
    with ``sigmoid(gate_logits) >= 0.5`` (equivalently, logits >= 0).
    Forward calls do not sample, even in train mode. Only the DisARM training
    step explicitly evaluates sampled masks, so prediction, Jacobian diagnostics
    and checkpoint selection all use the same deterministic graph.
    """

    def __init__(
        self, num_vars: int, order: int, hidden_layer_size: int,
        num_hidden_layers: int = 1, gate_probability_init: float = 0.9,
    ) -> None:
        if not math.isfinite(gate_probability_init) or not 0 < gate_probability_init < 1:
            raise ValueError("gate_probability_init must be strictly between 0 and 1")
        super().__init__(num_vars, order, hidden_layer_size, num_hidden_layers,
                         use_causal_gate=True, gate_init=1.0)
        shape = self.causal_gate.shape
        del self.causal_gate
        self.register_buffer("causal_gate", torch.ones(shape))
        logit = math.log(gate_probability_init) - math.log1p(-gate_probability_init)
        self.gate_logits = nn.Parameter(torch.full(shape, logit))
        self.update_causal_gate_()

    @property
    def gate_probabilities(self) -> torch.Tensor:
        """Opening probabilities, not probabilities that true causal edges exist."""
        return self.gate_logits.sigmoid()

    @torch.no_grad()
    def update_causal_gate_(self) -> torch.Tensor:
        self.causal_gate.copy_((self.gate_logits >= 0).to(self.causal_gate.dtype))
        return self.causal_gate

    def _masked_forward(self, inputs, mask, *, need_mismatch=False):
        if inputs.ndim != 3 or inputs.shape[1:] != (self.order, self.num_vars):
            raise ValueError("inputs must have shape [batch, order, num_vars]")
        if mask.shape != self.causal_gate.shape:
            raise ValueError("mask must have shape [order, target, source]")
        gate = mask.detach().to(device=inputs.device, dtype=inputs.dtype)
        coefficient_inputs = inputs.unsqueeze(2) * gate.unsqueeze(0)
        if need_mismatch and not coefficient_inputs.requires_grad:
            coefficient_inputs.requires_grad_(True)
        raw = self.coeff_net(coefficient_inputs)
        coefficients = raw * gate.unsqueeze(0)
        mismatch = None
        if need_mismatch:
            _, _, mismatch = _prediction_jacobian_components(
                raw, coefficients, coefficient_inputs, gate, create_graph=True,
            )
        prediction = self._add_intercept(torch.einsum("bkij,bkj->bi", coefficients, inputs))
        return prediction, coefficients, mismatch

    def forward_with_mask(self, inputs: torch.Tensor, mask: torch.Tensor):
        """Evaluate one explicit binary mask without changing the inference mask."""
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("mask must contain only 0 and 1")
        prediction, coefficients, _ = self._masked_forward(inputs, mask)
        return prediction, coefficients


@torch.no_grad()
def _antithetic_masks(logits: torch.Tensor, generator: torch.Generator):
    probabilities = logits.sigmoid()
    uniform = torch.rand(logits.shape, dtype=logits.dtype, device=logits.device,
                         generator=generator)
    return uniform < probabilities, (1 - uniform) < probabilities


@torch.no_grad()
def _disarm_gradient(logits, first_mask, second_mask, first_loss, second_loss):
    """Equation (7), with scalar or broadcastable target-local loss differences."""
    sign = 1 - 2 * second_mask.to(logits.dtype)
    different = (first_mask != second_mask).to(logits.dtype)
    return 0.5 * (first_loss - second_loss) * sign * different * logits.abs().sigmoid()


def _validate_config(config: GSEVARDisARMTrainingConfig) -> None:
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


def _disarm_step(model, inputs, targets, times, config, optimizer, generator):
    """Two mask evaluations at the same parameters; one joint optimizer step.

    Coefficient gradients average the two unchanged scalar sampled losses.
    Each logit uses only its target's separable loss differences at the original
    global-mean scale. Coupled relative smoothness uses the full scalar loss
    difference; the analytic expected-selection-cost gradient is unchanged.
    This does not claim that a detached local derivative through a binary gate
    is an unbiased estimator. No surrogate/straight-through gradient is used.
    """
    first_mask, second_mask = _antithetic_masks(model.gate_logits, generator)
    optimizer.zero_grad(set_to_none=True)
    first, first_target_losses = _masked_losses(model, inputs, targets, times, first_mask, config)
    first_loss = sum(first.values())
    if first_loss.requires_grad:
        (0.5 * first_loss).backward()
    second, second_target_losses = _masked_losses(model, inputs, targets, times, second_mask, config)
    second_loss = sum(second.values())
    decay = 0.5 * config.coefficient_weight_decay * sum(
        (parameter.square().sum() for parameter in model.coeff_net.parameters()
         if parameter.requires_grad), second_loss.new_zeros(()),
    )
    if (second_loss + decay).requires_grad:
        (0.5 * second_loss + decay).backward()
    with torch.no_grad():
        probabilities = model.gate_probabilities
        expected = probabilities.sum()
        l0 = config.lambda_l0 * expected
        gradient = _disarm_gradient(model.gate_logits, first_mask, second_mask,
                                   first_target_losses[None, :, None],
                                   second_target_losses[None, :, None])
        gradient += config.lambda_l0 * probabilities * (1 - probabilities)
        if model.gate_logits.requires_grad:
            model.gate_logits.grad = gradient
        metrics = {key: 0.5 * (first[key].detach() + second[key].detach()) for key in first}
        metrics.update(weight_decay=decay.detach(), l0=l0, expected_lag_edges=expected,
                       gate_gradient_norm=gradient.norm())
        metrics["loss"] = (metrics["mse"] + metrics["smooth"] + metrics["jacobian"]
                           + metrics["weight_decay"] + metrics["l0"])
        checks = [torch.isfinite(model.gate_logits).all()]
        checks += [torch.isfinite(value).all() for value in metrics.values()]
        checks += [torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                   if parameter.grad is not None]
        if not torch.stack(checks).all():
            raise FloatingPointError("non-finite DisARM loss or gradient")
    optimizer.step()
    model.update_causal_gate_()
    return metrics


def fit_gse_var_disarm(
    data: np.ndarray | list[np.ndarray],
    config: GSEVARDisARMTrainingConfig,
    model: GSEVARDisARM | None = None,
    *,
    epoch_callback: Callable[[int, GSEVARDisARM, dict[str, float]], None] | None = None,
) -> GSEVARFitResult:
    """Fit binary gates without changing the existing ``fit_gse_var`` route.

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
        model = GSEVARDisARM(num_vars, config.order, config.hidden_layer_size,
                            config.num_hidden_layers, config.gate_probability_init)
    if not isinstance(model, GSEVARDisARM):
        raise ValueError("fit_gse_var_disarm requires a GSEVARDisARM model")
    if (model.num_vars, model.order) != (num_vars, config.order):
        raise ValueError("model dimensions/order do not match data and config")
    if model.intercept is None:
        raise ValueError("DisARM requires an intercept")
    model.to(device)
    model.update_causal_gate_()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("DisARM fitting requires trainable parameters")
    if config.gate_learning_rate is None:
        optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    else:
        groups = [{"params": [parameter for name, parameter in model.named_parameters()
                               if name != "gate_logits" and parameter.requires_grad]},
                  {"params": [model.gate_logits] if model.gate_logits.requires_grad else [],
                   "lr": config.gate_learning_rate}]
        optimizer = torch.optim.Adam(groups, lr=config.learning_rate)
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
