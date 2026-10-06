from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

RegularizerName = Literal[
    "none",
    "sparse_group_lasso",
    "group_lasso",
    "hierarchical_group_lasso",
]
ReductionName = Literal["sum", "mean"]


def _group_norms_by_pair(tensor: torch.Tensor, lag_dim: int) -> torch.Tensor:
    dims = tuple(
        dim
        for dim in range(tensor.ndim)
        if dim not in (lag_dim, tensor.ndim - 2, tensor.ndim - 1)
    )
    if dims:
        tensor = torch.linalg.vector_norm(tensor, ord=2, dim=dims)
    if lag_dim > 0:
        lag_dim = 0
    return torch.linalg.vector_norm(tensor, ord=2, dim=lag_dim)


def _reduce(values: torch.Tensor, reduction: ReductionName) -> torch.Tensor:
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError(f"unsupported reduction: {reduction}")


def group_lasso_penalty(
    tensor: torch.Tensor,
    *,
    lag_dim: int = 0,
    reduction: ReductionName = "sum",
) -> torch.Tensor:
    """Group lasso over lag blocks for every target-source pair.

    Supports gate tensors ``[lag, target, source]`` and coefficient tensors
    ``[batch, lag, target, source]``.
    """
    norms = _group_norms_by_pair(tensor, lag_dim=lag_dim)
    return _reduce(norms, reduction)


def sparse_group_lasso_penalty(
    tensor: torch.Tensor,
    *,
    lag_dim: int = 0,
    reduction: ReductionName = "sum",
    l1_lambda: float = 0.0,
    group_lambda: float = 0.0,
) -> torch.Tensor:
    """Sparse group lasso over lag blocks for every target-source pair.

    Neural-GC's cMLP implements ``GSGL`` as lag-level shrinkage followed by
    source-level group shrinkage. For the gate tensor used here, each lag-level
    gate is a scalar, so the lag-level term becomes an L1 penalty while the
    edge-level term is the usual group lasso across lags.
    """
    penalty = tensor.new_zeros(())
    if group_lambda:
        penalty = penalty + group_lambda * group_lasso_penalty(
            tensor,
            lag_dim=lag_dim,
            reduction=reduction,
        )
    if l1_lambda:
        penalty = penalty + l1_lambda * _reduce(tensor.abs(), reduction)
    return penalty


def hierarchical_group_lasso_penalty(
    tensor: torch.Tensor,
    *,
    lag_dim: int = 0,
    reduction: ReductionName = "sum",
) -> torch.Tensor:
    """Nested lag-prefix group lasso.

    Lag index 0 is assumed to be the most distant lag, matching the GVAR data
    layout and the Neural-GC cMLP hierarchical penalty convention.
    """
    tensor = tensor.movedim(lag_dim, 0)
    penalties = [
        group_lasso_penalty(
            tensor[: lag_idx + 1],
            lag_dim=0,
            reduction=reduction,
        )
        for lag_idx in range(tensor.shape[0])
    ]
    return torch.stack(penalties).sum()


@torch.no_grad()
def prox_lasso_(param: torch.Tensor, lam: float, step_size: float) -> torch.Tensor:
    """In-place elementwise soft-thresholding."""
    threshold = lam * step_size
    param.copy_(param.sign() * torch.clamp(param.abs() - threshold, min=0.0))
    return param


@torch.no_grad()
def prox_group_lasso_(
    param: torch.Tensor,
    lam: float,
    step_size: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """In-place proximal operator for group lasso on ``[lag, target, source]``."""
    if param.ndim != 3:
        raise ValueError("prox_group_lasso_ expects [lag, target, source]")
    threshold = lam * step_size
    norms = torch.linalg.vector_norm(param, ord=2, dim=0, keepdim=True)
    scale = torch.clamp(
        1.0 - threshold / torch.clamp(norms, min=eps),
        min=0.0,
    )
    param.mul_(scale)
    param.masked_fill_(norms <= threshold, 0.0)
    return param


@torch.no_grad()
def prox_sparse_group_lasso_(
    param: torch.Tensor,
    step_size: float,
    *,
    l1_lambda: float = 0.0,
    group_lambda: float = 0.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """In-place proximal operator for sparse group lasso.

    The update mirrors Neural-GC's ``GSGL`` order: first apply within-group
    sparsity, then shrink each target-source lag vector as one group.
    """
    if param.ndim != 3:
        raise ValueError("prox_sparse_group_lasso_ expects [lag, target, source]")
    if l1_lambda:
        prox_lasso_(param, l1_lambda, step_size)
    if group_lambda:
        prox_group_lasso_(param, group_lambda, step_size, eps=eps)
    return param


@torch.no_grad()
def prox_hierarchical_group_lasso_(
    param: torch.Tensor,
    lam: float,
    step_size: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """In-place nested proximal update for hierarchical group lasso.

    This follows the iterative nested-group shrinkage used in Neural-GC: each
    prefix ``param[:lag_idx + 1]`` is shrunk as one group for every
    target-source pair.
    """
    if param.ndim != 3:
        raise ValueError(
            "prox_hierarchical_group_lasso_ expects [lag, target, source]"
        )
    threshold = lam * step_size
    for lag_idx in range(param.shape[0]):
        block = param[: lag_idx + 1]
        norms = torch.linalg.vector_norm(block, ord=2, dim=0, keepdim=True)
        scale = torch.clamp(
            1.0 - threshold / torch.clamp(norms, min=eps),
            min=0.0,
        )
        block.mul_(scale)
        block.masked_fill_(norms <= threshold, 0.0)
    return param


@torch.no_grad()
def prox_nonnegative_lasso_(
    param: torch.Tensor,
    lam: float,
    step_size: float,
) -> torch.Tensor:
    """Proximal operator for ``lam * ||x||_1 + I[x >= 0]``.

    Unlike softplus or sigmoid parameterizations, this update can create exact
    zeros while preserving a non-negative structural gate.
    """
    threshold = lam * step_size
    param.sub_(threshold).clamp_(min=0.0)
    return param


@torch.no_grad()
def prox_nonnegative_group_lasso_(
    param: torch.Tensor,
    lam: float,
    step_size: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Non-negative group-lasso proximal update."""
    if param.ndim != 3:
        raise ValueError(
            "prox_nonnegative_group_lasso_ expects [lag, target, source]"
        )
    param.clamp_(min=0.0)
    return prox_group_lasso_(param, lam, step_size, eps=eps)


@torch.no_grad()
def prox_nonnegative_sparse_group_lasso_(
    param: torch.Tensor,
    step_size: float,
    *,
    l1_lambda: float = 0.0,
    group_lambda: float = 0.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Non-negative sparse-group-lasso proximal update.

    First applies the one-sided L1 threshold ``max(v - eta*lambda, 0)`` and
    then shrinks every target-source lag vector as a group.
    """
    if param.ndim != 3:
        raise ValueError(
            "prox_nonnegative_sparse_group_lasso_ expects "
            "[lag, target, source]"
        )
    if l1_lambda:
        prox_nonnegative_lasso_(param, l1_lambda, step_size)
    else:
        param.clamp_(min=0.0)
    if group_lambda:
        prox_group_lasso_(param, group_lambda, step_size, eps=eps)
    return param


@torch.no_grad()
def prox_nonnegative_hierarchical_group_lasso_(
    param: torch.Tensor,
    lam: float,
    step_size: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Non-negative nested hierarchical-group-lasso proximal update."""
    if param.ndim != 3:
        raise ValueError(
            "prox_nonnegative_hierarchical_group_lasso_ expects "
            "[lag, target, source]"
        )
    param.clamp_(min=0.0)
    return prox_hierarchical_group_lasso_(param, lam, step_size, eps=eps)


@dataclass(frozen=True)
class NGCRegularizer:
    name: RegularizerName = "sparse_group_lasso"
    lam: float = 0.0
    reduction: ReductionName = "sum"
    lag_dim: int = 0
    sparse_l1_lambda: float = 0.0
    sparse_group_lambda: float = 0.0

    def __post_init__(self) -> None:
        if self.name == "sparse_group_lasso" and self.lam != 0:
            raise ValueError(
                "sparse_group_lasso does not use lam/lambda_ngc. "
                "Use sparse_group_lambda and sparse_l1_lambda instead."
            )
        if self.name not in ("none", "sparse_group_lasso") and (
            self.sparse_l1_lambda != 0 or self.sparse_group_lambda != 0
        ):
            raise ValueError(
                "sparse_l1_lambda and sparse_group_lambda are only used by "
                "sparse_group_lasso. Use lam/lambda_ngc for this regularizer."
            )

    def penalty(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.name == "none":
            return tensor.new_zeros(())
        if self.name == "sparse_group_lasso":
            return sparse_group_lasso_penalty(
                tensor,
                lag_dim=self.lag_dim,
                reduction=self.reduction,
                l1_lambda=self.sparse_l1_lambda,
                group_lambda=self.sparse_group_lambda,
            )
        if self.lam == 0:
            return tensor.new_zeros(())
        if self.name == "group_lasso":
            return self.lam * group_lasso_penalty(
                tensor,
                lag_dim=self.lag_dim,
                reduction=self.reduction,
            )
        if self.name == "hierarchical_group_lasso":
            return self.lam * hierarchical_group_lasso_penalty(
                tensor,
                lag_dim=self.lag_dim,
                reduction=self.reduction,
            )
        raise ValueError(f"unsupported regularizer: {self.name}")

    @torch.no_grad()
    def prox_(
        self,
        param: torch.Tensor,
        step_size: float,
        *,
        nonnegative: bool = False,
    ) -> torch.Tensor:
        """Apply the configured proximal operator in place.

        ``nonnegative=True`` is intended for the structural ``causal_gate``.
        The default remains ``False`` so existing signed-weight users retain
        their previous behavior.
        """
        if self.lag_dim != 0:
            raise ValueError(
                "prox_ expects lag_dim=0 and param shape [lag, target, source]"
            )

        if self.name == "none":
            if nonnegative:
                param.clamp_(min=0.0)
            return param

        if self.name == "sparse_group_lasso":
            if nonnegative:
                return prox_nonnegative_sparse_group_lasso_(
                    param,
                    step_size,
                    l1_lambda=self.sparse_l1_lambda,
                    group_lambda=self.sparse_group_lambda,
                )
            return prox_sparse_group_lasso_(
                param,
                step_size,
                l1_lambda=self.sparse_l1_lambda,
                group_lambda=self.sparse_group_lambda,
            )

        if self.lam == 0:
            if nonnegative:
                param.clamp_(min=0.0)
            return param

        if self.name == "group_lasso":
            if nonnegative:
                return prox_nonnegative_group_lasso_(
                    param,
                    self.lam,
                    step_size,
                )
            return prox_group_lasso_(param, self.lam, step_size)

        if self.name == "hierarchical_group_lasso":
            if nonnegative:
                return prox_nonnegative_hierarchical_group_lasso_(
                    param,
                    self.lam,
                    step_size,
                )
            return prox_hierarchical_group_lasso_(param, self.lam, step_size)

        raise ValueError(f"unsupported regularizer: {self.name}")
