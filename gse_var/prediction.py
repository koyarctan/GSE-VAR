"""Evaluate effective coefficients on supplied inputs, without fitting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import LaggedDataset, VARXLaggedDataset


@dataclass
class CoefficientEvaluation:
    model: object
    coeffs: np.ndarray | None = None
    endogenous_coeffs: np.ndarray | None = None
    exogenous_coeffs: np.ndarray | None = None
    endogenous_names: tuple[str, ...] | None = None
    exogenous_lags: np.ndarray | None = None
    causal_threshold: float = 0.0


def evaluate_coefficients(result_or_model, data, *, exog_inputs=None, batch_size=256):
    """Evaluate *only* the supplied lagged predictors (train/valid/test, etc.).

    ``data`` is a LaggedDataset, VARXLaggedDataset, endogenous lag tensor, or
    ``(endogenous_predictors, exogenous_predictors)`` tuple. Responses are never
    read. No training coefficients are reused, no refitting is done, and the
    model's original train/eval state is restored. Input scales must match fit.
    """
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    model = getattr(result_or_model, "model", result_or_model)
    is_varx = hasattr(model, "endogenous_gate")
    if isinstance(data, VARXLaggedDataset):
        if not is_varx:
            raise ValueError("VARX inputs require a VARX model")
        if data.exogenous_layout != model.exogenous_layout:
            raise ValueError("evaluation input identities differ from the fitted model")
        y, x = data.endogenous_predictors, data.exogenous_predictors
    elif isinstance(data, LaggedDataset):
        y, x = data.predictors, exog_inputs
    elif isinstance(data, tuple) and len(data) == 2:
        if exog_inputs is not None:
            raise ValueError("do not supply exog_inputs twice")
        y, x = data
    else:
        y, x = data, exog_inputs
    if is_varx and x is None:
        raise ValueError("VARX evaluation requires exogenous predictors")
    if not is_varx and x is not None:
        raise ValueError("GSE-VAR evaluation does not accept exogenous predictors")
    parameter = next(model.parameters())
    y = torch.as_tensor(y, dtype=parameter.dtype, device=parameter.device)
    x = None if x is None else torch.as_tensor(x, dtype=parameter.dtype, device=parameter.device)
    if y.ndim != 3 or len(y) == 0 or not torch.isfinite(y).all():
        raise ValueError("endogenous predictors must be finite, nonempty [sample, lag, source]")
    if x is not None and (x.ndim != 3 or len(x) != len(y) or not torch.isfinite(x).all()):
        raise ValueError("exogenous predictors must be finite and aligned with endogenous predictors")
    endogenous, exogenous = [], []
    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(y), batch_size):
                if is_varx:
                    _, cy, cx = model(y[start:start + batch_size], x[start:start + batch_size])
                    exogenous.append(cx.cpu().numpy())
                else:
                    _, cy = model(y[start:start + batch_size])
                if not torch.isfinite(cy).all() or (is_varx and not torch.isfinite(cx).all()):
                    raise ValueError("model produced non-finite effective coefficients")
                endogenous.append(cy.cpu().numpy())
    finally:
        model.train(was_training)
    cy = np.concatenate(endogenous)
    if is_varx:
        layout = model.exogenous_layout
        lags = (layout.lags if layout is not None else np.arange(
            model.exog_order, -1 if model.include_current_exog else 0, -1))
    else:
        lags = None
    return CoefficientEvaluation(
        model=model, coeffs=None if is_varx else cy,
        endogenous_coeffs=cy if is_varx else None,
        exogenous_coeffs=np.concatenate(exogenous) if is_varx else None,
        endogenous_names=getattr(result_or_model, "endogenous_names", None),
        exogenous_lags=lags,
        causal_threshold=getattr(result_or_model, "causal_threshold", 0.0),
    )
