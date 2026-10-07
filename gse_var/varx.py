from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from numbers import Integral

import numpy as np
import torch
from torch import nn

from .data import VARXLaggedDataset, construct_varx_lagged_dataset
from .models import GSEVARX
from .regularizers import ExogenousGroupRegularizer, NGCRegularizer, RegularizerName
from .exogenous import ExogenousLayout
from .training import (
    GSEVARTrainingConfig,
    _iter_batches,
    _resolve_device,
    _validate_config,
    temporal_smoothness_penalty,
)


@dataclass(frozen=True)
class GSEVARXTrainingConfig(GSEVARTrainingConfig):
    """Training configuration for generic endogenous/exogenous GSEVARX.

    Existing regularization fields apply to the endogenous branch. Optional
    ``*_exog`` values override them for the exogenous branch. Gate penalty
    weights set to ``None`` reuse the corresponding endogenous value only
    when supported by the selected exogenous regularizer; otherwise they
    resolve to zero. Other ``None`` overrides always reuse the endogenous value.
    """

    exog_order: int = 0
    include_current_exog: bool = True
    lambda_ngc_exog: float | None = None
    lambda_smooth_exog: float | None = None
    lambda_jacobian_exog: float | None = None
    sparse_group_lambda_exog: float | None = None
    sparse_l1_lambda_exog: float | None = None
    exogenous_gate_init: float | None = None
    regularizer_exog: RegularizerName | None = None
    exogenous_group_size_weights: bool = True
    early_stopping: bool = False
    early_stopping_patience: int = 100
    early_stopping_min_delta: float = 0.0
    restore_best_weights: bool = True


@dataclass
class GSEVARXFitResult:
    model: GSEVARX
    history: dict[str, list[float]] = field(default_factory=dict)
    endogenous_coeffs: np.ndarray | None = None
    exogenous_coeffs: np.ndarray | None = None
    endogenous_strength: np.ndarray | None = None
    exogenous_strength: np.ndarray | None = None
    endogenous_graph: np.ndarray | None = None
    exogenous_graph: np.ndarray | None = None
    exogenous_lags: np.ndarray | None = None
    exogenous_layout: ExogenousLayout | None = None
    endogenous_names: tuple[str, ...] | None = None
    causal_threshold: float = 0.0
    exogenous_term_graph: np.ndarray | None = None
    best_epoch: int | None = None
    best_validation_mse: float | None = None
    epochs_trained: int = 0
    early_stopped: bool = False
    stopped_epoch: int | None = None

    @property
    def exogenous_source_names(self):
        return None if self.exogenous_layout is None else self.exogenous_layout.sources

    @property
    def exogenous_gate_by_lag(self):
        gate = self.model.exogenous_gate.detach().cpu().numpy()
        return gate if self.exogenous_layout is None else self.exogenous_layout.gate_by_lag(gate)

    @property
    def exogenous_graph_by_lag(self):
        gate = self.exogenous_gate_by_lag
        return np.where(np.isfinite(gate), (gate > self.causal_threshold).astype(float), np.nan)

    @property
    def exogenous_coeffs_by_lag(self):
        if self.exogenous_coeffs is None:
            return None
        return (self.exogenous_coeffs if self.exogenous_layout is None else
                self.exogenous_layout.coefficients_by_lag(self.exogenous_coeffs))


@dataclass(frozen=True)
class _TorchVARXLaggedDataset:
    endogenous_predictors: torch.Tensor
    exogenous_predictors: torch.Tensor
    responses: torch.Tensor
    time_index: torch.Tensor
    series_index: torch.Tensor


def _resolved(value: float | None, fallback: float) -> float:
    return fallback if value is None else value


def _exogenous_regularizer_kwargs(config: GSEVARXTrainingConfig) -> dict:
    """Resolve compatible inherited gate weights without mutating config."""
    name = config.regularizer_exog or config.regularizer
    uses_ngc = name in ("group_lasso", "hierarchical_group_lasso")
    uses_sparse = name == "sparse_group_lasso"
    return dict(
        name=name,
        lam=_resolved(
            config.lambda_ngc_exog,
            config.lambda_ngc if uses_ngc else 0.0,
        ),
        sparse_l1_lambda=_resolved(
            config.sparse_l1_lambda_exog,
            config.sparse_l1_lambda if uses_sparse else 0.0,
        ),
        sparse_group_lambda=_resolved(
            config.sparse_group_lambda_exog,
            config.sparse_group_lambda if uses_sparse else 0.0,
        ),
    )


def _validate_optional_nonnegative(
    name: str,
    value: float | None,
) -> None:
    if value is not None and (not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be non-negative")


def _validate_varx_config(config: GSEVARXTrainingConfig) -> None:
    _validate_config(config)
    if not isinstance(config.early_stopping, bool):
        raise ValueError("early_stopping must be a bool")
    if not isinstance(config.restore_best_weights, bool):
        raise ValueError("restore_best_weights must be a bool")
    if (isinstance(config.early_stopping_patience, bool)
            or not isinstance(config.early_stopping_patience, Integral)
            or config.early_stopping_patience <= 0):
        raise ValueError("early_stopping_patience must be a positive integer")
    if (not math.isfinite(config.early_stopping_min_delta)
            or config.early_stopping_min_delta < 0):
        raise ValueError("early_stopping_min_delta must be finite and non-negative")
    if config.early_stopping and config.max_epochs < 1:
        raise ValueError("early stopping requires max_epochs >= 1")
    if config.exog_order < 0:
        raise ValueError("exog_order must be non-negative")
    if config.exog_order == 0 and not config.include_current_exog:
        raise ValueError(
            "include_current_exog must be True when exog_order is zero"
        )
    for name in (
        "lambda_ngc_exog",
        "lambda_smooth_exog",
        "lambda_jacobian_exog",
        "sparse_group_lambda_exog",
        "sparse_l1_lambda_exog",
        "exogenous_gate_init",
    ):
        _validate_optional_nonnegative(name, getattr(config, name))

    exog_name = config.regularizer_exog or config.regularizer
    if exog_name not in ("none", "group_lasso", "sparse_group_lasso", "hierarchical_group_lasso"):
        raise ValueError("unsupported regularizer_exog")
    exogenous_kwargs = _exogenous_regularizer_kwargs(config)
    if exog_name == "sparse_group_lasso":
        if exogenous_kwargs["lam"] != 0:
            raise ValueError(
                "sparse_group_lasso does not use lambda_ngc_exog. Use "
                "sparse_group_lambda_exog and sparse_l1_lambda_exog instead."
            )
    elif (
        exogenous_kwargs["sparse_group_lambda"] != 0
        or exogenous_kwargs["sparse_l1_lambda"] != 0
    ):
        raise ValueError(
            "sparse_group_lambda_exog and sparse_l1_lambda_exog are only "
                "used with the exogenous sparse_group_lasso regularizer."
        )


def _to_torch_varx_dataset(
    dataset: VARXLaggedDataset,
    device: torch.device,
) -> _TorchVARXLaggedDataset:
    return _TorchVARXLaggedDataset(
        endogenous_predictors=torch.as_tensor(
            dataset.endogenous_predictors,
            dtype=torch.float32,
            device=device,
        ),
        exogenous_predictors=torch.as_tensor(
            dataset.exogenous_predictors,
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


def _make_optimizer(
    config: GSEVARXTrainingConfig,
    model: GSEVARX,
) -> torch.optim.Optimizer:
    gate_parameters = []
    coefficient_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name in ("endogenous_gate", "exogenous_gate"):
            gate_parameters.append(parameter)
        else:
            coefficient_parameters.append(parameter)

    parameter_groups = [
        {
            "params": coefficient_parameters,
            "weight_decay": config.coefficient_weight_decay,
        },
        {"params": gate_parameters, "weight_decay": 0.0},
    ]
    if config.optimizer == "adam":
        return torch.optim.Adam(
            parameter_groups,
            lr=config.learning_rate,
        )
    if config.optimizer == "ista":
        return torch.optim.SGD(
            parameter_groups,
            lr=config.learning_rate,
        )
    raise ValueError(f"unsupported optimizer: {config.optimizer}")


def _prepare_validation_dataset(
    validation_data,
    training: VARXLaggedDataset,
    lag_kwargs: dict,
) -> VARXLaggedDataset | None:
    """Reuse explicit validation inputs, without splitting or fitting scales."""
    if validation_data is None:
        return None
    if isinstance(validation_data, VARXLaggedDataset):
        dataset = validation_data
    elif isinstance(validation_data, tuple) and len(validation_data) == 2:
        dataset = construct_varx_lagged_dataset(*validation_data, **lag_kwargs)
    else:
        raise ValueError("validation_data must be a VARXLaggedDataset or an (endog, exog) tuple")

    y = np.asarray(dataset.endogenous_predictors, dtype=np.float32)
    x = np.asarray(dataset.exogenous_predictors, dtype=np.float32)
    responses = np.asarray(dataset.responses, dtype=np.float32)
    if responses.ndim != 2 or not len(responses):
        raise ValueError("validation responses must be nonempty [sample, target]")
    n = len(responses)
    if y.shape != (n,) + training.endogenous_predictors.shape[1:]:
        raise ValueError("validation endogenous predictor shape does not match training")
    if x.shape != (n,) + training.exogenous_predictors.shape[1:]:
        raise ValueError("validation exogenous predictor shape does not match training")
    if responses.shape[1:] != training.responses.shape[1:]:
        raise ValueError("validation response shape does not match training")
    if dataset.exogenous_layout != training.exogenous_layout:
        raise ValueError("validation exogenous identities do not match training")
    if not np.array_equal(dataset.exogenous_lags, training.exogenous_lags):
        raise ValueError("validation exogenous lags do not match training")
    if not all(np.isfinite(values).all() for values in (y, x, responses)):
        raise ValueError("validation predictors and responses must be finite")
    times, series = np.asarray(dataset.time_index), np.asarray(dataset.series_index)
    if times.shape != (n,) or series.shape != (n,):
        raise ValueError("validation sample indices do not match response count")
    return VARXLaggedDataset(
        endogenous_predictors=y, exogenous_predictors=x, responses=responses,
        time_index=times, series_index=series,
        exogenous_lags=dataset.exogenous_lags, exogenous_layout=dataset.exogenous_layout,
    )


@torch.no_grad()
def _validation_mse(
    model: GSEVARX,
    dataset: _TorchVARXLaggedDataset,
    batch_size: int,
) -> float:
    """Pure prediction MSE, weighted by all response elements, not batches."""
    if batch_size <= 0:
        raise ValueError("validation requires a positive batch_size")
    was_training = model.training
    model.eval()
    try:
        total = torch.zeros((), dtype=torch.float64, device=dataset.responses.device)
        for start in range(0, len(dataset.responses), batch_size):
            stop = start + batch_size
            predictions, _, _ = model(
                dataset.endogenous_predictors[start:stop],
                dataset.exogenous_predictors[start:stop],
            )
            total += nn.functional.mse_loss(
                predictions, dataset.responses[start:stop], reduction="sum",
            ).double()
        value = float((total / dataset.responses.numel()).cpu())
    finally:
        model.train(was_training)
    if not math.isfinite(value):
        raise ValueError("validation MSE must be finite; check training stability")
    return value


def _make_regularizers(
    config: GSEVARXTrainingConfig,
    layout: ExogenousLayout | None = None,
) -> tuple[NGCRegularizer, NGCRegularizer]:
    endogenous_regularizer = NGCRegularizer(
        name=config.regularizer,
        lam=config.lambda_ngc,
        reduction="sum",
        lag_dim=0,
        sparse_l1_lambda=config.sparse_l1_lambda,
        sparse_group_lambda=config.sparse_group_lambda,
    )
    exogenous_kwargs = _exogenous_regularizer_kwargs(config)
    exogenous_regularizer = (
        NGCRegularizer(**exogenous_kwargs, reduction="sum", lag_dim=0)
        if layout is None else ExogenousGroupRegularizer(
            **exogenous_kwargs, layout=layout, size_weights=config.exogenous_group_size_weights
        )
    )
    return endogenous_regularizer, exogenous_regularizer


def _epoch(
    model: GSEVARX,
    dataset: _TorchVARXLaggedDataset,
    config: GSEVARXTrainingConfig,
    optimizer: torch.optim.Optimizer,
    endogenous_regularizer: NGCRegularizer,
    exogenous_regularizer: NGCRegularizer,
    criterion: nn.Module,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    model.train()
    metric_names = (
        "loss",
        "mse",
        "ngc",
        "ngc_endogenous",
        "ngc_exogenous",
        "smooth",
        "smooth_endogenous",
        "smooth_exogenous",
        "jacobian",
        "jacobian_endogenous",
        "jacobian_exogenous",
    )
    totals = {
        name: torch.zeros((), device=device)
        for name in metric_names
    }
    n_batches = 0
    exogenous_smoothness_weight = _resolved(
        config.lambda_smooth_exog,
        config.lambda_smooth,
    )
    exogenous_jacobian_weight = _resolved(
        config.lambda_jacobian_exog,
        config.lambda_jacobian,
    )
    use_jacobian = (
        config.lambda_jacobian != 0 or exogenous_jacobian_weight != 0
    )

    for batch_idx in _iter_batches(
        dataset.endogenous_predictors.shape[0],
        config.batch_size,
        shuffle=config.shuffle,
        rng=rng,
        device=device,
    ):
        endogenous_inputs = dataset.endogenous_predictors[batch_idx]
        exogenous_inputs = dataset.exogenous_predictors[batch_idx]
        targets = dataset.responses[batch_idx]
        predictions, endogenous_coeffs, exogenous_coeffs = model(
            endogenous_inputs,
            exogenous_inputs,
        )
        mse = criterion(predictions, targets)

        time_index = dataset.time_index[batch_idx]
        if config.lambda_smooth != 0:
            smooth_endogenous = (
                config.lambda_smooth
                * temporal_smoothness_penalty(
                    endogenous_coeffs,
                    time_index,
                    mode=config.smoothness_mode,
                    eps=config.smoothness_eps,
                )
            )
        else:
            smooth_endogenous = mse.new_zeros(())
        if exogenous_smoothness_weight != 0:
            smooth_exogenous = (
                exogenous_smoothness_weight
                * temporal_smoothness_penalty(
                    exogenous_coeffs,
                    time_index,
                    mode=config.smoothness_mode,
                    eps=config.smoothness_eps,
                )
            )
        else:
            smooth_exogenous = mse.new_zeros(())
        smooth = smooth_endogenous + smooth_exogenous

        if use_jacobian:
            endogenous_mismatch, exogenous_mismatch = (
                model.coefficient_jacobian_mismatches_for_regularization(
                    endogenous_inputs,
                    exogenous_inputs,
                    create_graph=True,
                )
            )
            jacobian_endogenous = (
                config.lambda_jacobian
                * endogenous_mismatch.pow(2).mean()
            )
            jacobian_exogenous = (
                exogenous_jacobian_weight
                * exogenous_mismatch.pow(2).mean()
            )
        else:
            jacobian_endogenous = mse.new_zeros(())
            jacobian_exogenous = mse.new_zeros(())
        jacobian = jacobian_endogenous + jacobian_exogenous

        ngc_endogenous = endogenous_regularizer.penalty(
            model.endogenous_gate
        )
        ngc_exogenous = exogenous_regularizer.penalty(model.exogenous_gate)
        ngc = ngc_endogenous + ngc_exogenous

        differentiable_loss = mse + smooth + jacobian
        if config.optimizer == "adam":
            loss = differentiable_loss + ngc
        else:
            loss = differentiable_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if config.optimizer == "ista":
            endogenous_regularizer.prox_(
                model.endogenous_gate,
                config.learning_rate,
                nonnegative=True,
            )
            exogenous_regularizer.prox_(
                model.exogenous_gate,
                config.learning_rate,
                nonnegative=True,
            )
        else:
            model.project_causal_gates_()

        ngc_endogenous = endogenous_regularizer.penalty(
            model.endogenous_gate
        )
        ngc_exogenous = exogenous_regularizer.penalty(model.exogenous_gate)
        ngc = ngc_endogenous + ngc_exogenous
        logged_loss = (
            mse.detach()
            + smooth.detach()
            + jacobian.detach()
            + ngc.detach()
        )

        batch_metrics = {
            "loss": logged_loss,
            "mse": mse,
            "ngc": ngc,
            "ngc_endogenous": ngc_endogenous,
            "ngc_exogenous": ngc_exogenous,
            "smooth": smooth,
            "smooth_endogenous": smooth_endogenous,
            "smooth_exogenous": smooth_exogenous,
            "jacobian": jacobian,
            "jacobian_endogenous": jacobian_endogenous,
            "jacobian_exogenous": jacobian_exogenous,
        }
        for name, value in batch_metrics.items():
            totals[name] = totals[name] + value.detach()
        n_batches += 1

    return {
        name: float((value / max(n_batches, 1)).detach().cpu())
        for name, value in totals.items()
    }


@torch.no_grad()
def _gate_usage(
    model: GSEVARX,
    threshold: float,
) -> tuple[int, int, int, int]:
    endogenous_graph = model.endogenous_graph_from_gate(threshold)
    exogenous_graph = model.exogenous_graph_from_gate(threshold)
    return (
        int(endogenous_graph.sum().cpu()),
        endogenous_graph.numel(),
        int(exogenous_graph.sum().cpu()),
        exogenous_graph.numel(),
    )


def _log_epoch(
    epoch: int,
    config: GSEVARXTrainingConfig,
    metrics: dict[str, float],
    model: GSEVARX,
    log_fn: Callable[[str], None] = print,
) -> None:
    if config.verbose <= 0:
        return
    log_every = max(config.log_every, 1)
    if epoch != 1 and epoch != config.max_epochs and epoch % log_every != 0:
        return

    endogenous_active, endogenous_total, exogenous_active, exogenous_total = (
        _gate_usage(model, config.causal_threshold)
    )
    validation_log = f"val_mse={metrics['val_mse']:.6g} | " if "val_mse" in metrics else ""
    log_fn(
        f"Epoch {epoch:>4}/{config.max_epochs:<4} | "
        f"loss={metrics['loss']:.6g} | "
        f"mse={metrics['mse']:.6g} | "
        f"ngc={metrics['ngc']:.6g} | "
        f"smooth={metrics['smooth']:.6g} | "
        f"jacobian={metrics['jacobian']:.6g} | "
        f"{validation_log}"
        f"endog_edges={endogenous_active}/{endogenous_total} | "
        f"exog_edges={exogenous_active}/{exogenous_total}"
    )


@torch.no_grad()
def _infer(
    model: GSEVARX,
    dataset: _TorchVARXLaggedDataset,
    config: GSEVARXTrainingConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    model.eval()
    endogenous_coefficients = []
    exogenous_coefficients = []
    for start in range(
        0,
        dataset.endogenous_predictors.shape[0],
        config.batch_size,
    ):
        stop = start + config.batch_size
        _, endogenous_batch, exogenous_batch = model(
            dataset.endogenous_predictors[start:stop],
            dataset.exogenous_predictors[start:stop],
        )
        endogenous_coefficients.append(endogenous_batch.cpu())
        exogenous_coefficients.append(exogenous_batch.cpu())

    endogenous_t = torch.cat(endogenous_coefficients)
    exogenous_t = torch.cat(exogenous_coefficients)
    endogenous_strength = model.coefficient_strength(
        endogenous_t,
        aggregation=config.strength_aggregation,
    )
    if model.exogenous_layout is None:
        exogenous_strength = model.coefficient_strength(exogenous_t, aggregation=config.strength_aggregation)
    else:
        exogenous_strength = torch.stack([
            model.coefficient_strength(
                exogenous_t[:, 0].index_select(-1, torch.tensor(group)).permute(0, 2, 1).unsqueeze(-1),
                aggregation=config.strength_aggregation).squeeze(-1)
            for group in model.exogenous_layout.groups
        ], dim=-1)
    endogenous_graph = model.endogenous_graph_from_gate(
        config.causal_threshold
    )
    exogenous_graph = model.exogenous_graph_from_gate(
        config.causal_threshold
    )
    return (
        endogenous_t.numpy(),
        exogenous_t.numpy(),
        endogenous_strength.numpy(),
        exogenous_strength.numpy(),
        endogenous_graph.cpu().numpy(),
        exogenous_graph.cpu().numpy(),
    )


def fit_gse_varx(
    endog: np.ndarray | list[np.ndarray],
    exog: np.ndarray | list[np.ndarray],
    config: GSEVARXTrainingConfig,
    model: GSEVARX | None = None,
    *,
    exog_features=None,
    exog_names=None,
    exog_lags=None,
    known_future_exog=(),
    endog_names=None,
    validation_data: VARXLaggedDataset | tuple | None = None,
) -> GSEVARXFitResult:
    """Fit GSEVARX without assuming application-specific variable names.

    Optional exog_features identifies already-expanded columns. Alternatively,
    exog_names/exog_lags describe raw variables and allowed physical lags;
    negative lags require known_future_exog. Layout-aware mode preserves one
    joint packed exogenous coefficient generator, groups gates by original
    source for each target, and returns source-level graphs/strengths while
    keeping the coefficient tensor packed. Legacy uniform-lag mode is unchanged.

    Optional validation_data is a preconstructed VARXLaggedDataset or a tuple
    of raw/already-expanded (endog, exog) blocks using the same lag metadata.
    With early_stopping=True it is required. Monitor pure validation MSE once
    per epoch, stopping after patience consecutive epochs without an improvement
    exceeding min_delta. By default restore the lowest-MSE epoch's full state,
    including gates, before inferring coefficients/graphs on training inputs.
    min_delta controls patience resets, not which minimum-MSE state is saved.
    No automatic split, standardization, or validation-to-training concatenation
    is performed; chronological splits and observed context are caller-owned.
    """
    _validate_varx_config(config)
    if config.early_stopping and validation_data is None:
        raise ValueError("early_stopping=True requires explicit validation_data")
    if validation_data is not None and (config.max_epochs < 1 or config.batch_size < 1):
        raise ValueError("validation requires positive max_epochs and batch_size")
    if config.seed is not None:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    lag_kwargs = dict(
        order=config.order,
        exog_order=config.exog_order,
        include_current_exog=config.include_current_exog,
        exog_features=exog_features, exog_names=exog_names,
        exog_lags=exog_lags, known_future_exog=known_future_exog,
    )
    dataset_np = construct_varx_lagged_dataset(endog, exog, **lag_kwargs)
    validation_np = _prepare_validation_dataset(validation_data, dataset_np, lag_kwargs)
    device = _resolve_device(config.device)
    dataset = _to_torch_varx_dataset(dataset_np, device)
    validation = None if validation_np is None else _to_torch_varx_dataset(validation_np, device)
    num_endogenous = dataset.endogenous_predictors.shape[-1]
    num_exogenous = dataset.exogenous_predictors.shape[-1]

    if model is None:
        model = GSEVARX(
            num_endogenous=num_endogenous,
            num_exogenous=num_exogenous,
            order=config.order,
            exog_order=config.exog_order,
            include_current_exog=config.include_current_exog,
            hidden_layer_size=config.hidden_layer_size,
            num_hidden_layers=config.num_hidden_layers,
            gate_init=config.gate_init,
            exogenous_gate_init=config.exogenous_gate_init,
            exogenous_layout=dataset_np.exogenous_layout,
        )
    if model.num_endogenous != num_endogenous:
        raise ValueError(
            "model num_endogenous does not match endog variable count"
        )
    if model.num_exogenous != num_exogenous:
        raise ValueError(
            "model num_exogenous does not match exog variable count"
        )
    if model.order != config.order:
        raise ValueError("model order does not match config order")
    if model.exogenous_layout != dataset_np.exogenous_layout:
        raise ValueError("model exogenous identities do not match the dataset layout")
    if endog_names is not None and len(endog_names) != num_endogenous:
        raise ValueError("endog_names must match the endogenous variable count")
    if (
        model.exog_order != config.exog_order
        or model.include_current_exog != config.include_current_exog
    ):
        raise ValueError(
            "model exogenous lag configuration does not match config"
        )

    model.to(device)
    model.project_causal_gates_()
    endogenous_regularizer, exogenous_regularizer = _make_regularizers(config, dataset_np.exogenous_layout)
    optimizer = _make_optimizer(config, model)
    criterion = nn.MSELoss(reduction="mean")
    rng = np.random.default_rng(config.seed)
    metric_names = (
        "loss",
        "mse",
        "ngc",
        "ngc_endogenous",
        "ngc_exogenous",
        "smooth",
        "smooth_endogenous",
        "smooth_exogenous",
        "jacobian",
        "jacobian_endogenous",
        "jacobian_exogenous",
    )
    history = {name: [] for name in metric_names}
    if validation is not None:
        history["val_mse"] = []
    best_epoch, best_state = None, None
    best_validation_mse = math.inf
    patience_reference = math.inf
    bad_epochs = 0
    stopped_epoch = None

    for epoch in range(1, config.max_epochs + 1):
        metrics = _epoch(
            model,
            dataset,
            config,
            optimizer,
            endogenous_regularizer,
            exogenous_regularizer,
            criterion,
            device,
            rng,
        )
        for name, value in metrics.items():
            history[name].append(value)
        if validation is not None:
            val_mse = _validation_mse(model, validation, config.batch_size)
            metrics["val_mse"] = val_mse
            history["val_mse"].append(val_mse)
            if val_mse < best_validation_mse:
                best_validation_mse, best_epoch = val_mse, epoch
                if config.early_stopping and config.restore_best_weights:
                    best_state = {name: value.detach().cpu().clone()
                                  for name, value in model.state_dict().items()}
            if val_mse < patience_reference - config.early_stopping_min_delta:
                patience_reference, bad_epochs = val_mse, 0
            else:
                bad_epochs += 1
        _log_epoch(epoch, config, metrics, model)
        if config.early_stopping and bad_epochs >= config.early_stopping_patience:
            stopped_epoch = epoch
            if config.verbose > 0:
                print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}, "
                      f"best val_mse={best_validation_mse:.6g}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        if config.verbose > 0:
            print(f"Restored best weights from epoch {best_epoch}")

    (
        endogenous_coeffs,
        exogenous_coeffs,
        endogenous_strength,
        exogenous_strength,
        endogenous_graph,
        exogenous_graph,
    ) = _infer(model, dataset, config)
    return GSEVARXFitResult(
        model=model,
        history=history,
        endogenous_coeffs=endogenous_coeffs,
        exogenous_coeffs=exogenous_coeffs,
        endogenous_strength=endogenous_strength,
        exogenous_strength=exogenous_strength,
        endogenous_graph=endogenous_graph,
        exogenous_graph=exogenous_graph,
        exogenous_lags=dataset_np.exogenous_lags.copy(),
        exogenous_layout=dataset_np.exogenous_layout,
        endogenous_names=None if endog_names is None else tuple(endog_names),
        causal_threshold=config.causal_threshold,
        exogenous_term_graph=(model.exogenous_gate.detach().cpu().numpy()[0] > config.causal_threshold).astype(int)
        if dataset_np.exogenous_layout is not None else None,
        best_epoch=best_epoch,
        best_validation_mse=None if best_epoch is None else best_validation_mse,
        epochs_trained=len(history["loss"]),
        early_stopped=stopped_epoch is not None,
        stopped_epoch=stopped_epoch,
    )
