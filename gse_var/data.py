from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exogenous import ExogenousLayout, _construct_variable_lag_dataset


@dataclass(frozen=True)
class LaggedDataset:
    predictors: np.ndarray
    responses: np.ndarray
    time_index: np.ndarray
    series_index: np.ndarray


@dataclass(frozen=True)
class VARXLaggedDataset:
    """Aligned endogenous and exogenous predictors for GSEVARX."""

    endogenous_predictors: np.ndarray
    exogenous_predictors: np.ndarray
    responses: np.ndarray
    time_index: np.ndarray
    series_index: np.ndarray
    exogenous_lags: np.ndarray
    exogenous_layout: ExogenousLayout | None = None


def _as_series_list(data: np.ndarray | list[np.ndarray]) -> list[np.ndarray]:
    if isinstance(data, list):
        if not data:
            raise ValueError("data list must not be empty")
        return data
    return [data]


def construct_lagged_dataset(data: np.ndarray | list[np.ndarray], order: int) -> LaggedDataset:
    """Build lagged GVAR predictors.

    Predictors have shape ``[n_samples, order, n_vars]``. The lag axis follows
    the original GVAR repository convention: index 0 is the most distant lag
    ``t - order`` and index ``order - 1`` is the most recent lag ``t - 1``.
    """
    if order <= 0:
        raise ValueError("order must be positive")

    predictors = []
    responses = []
    time_index = []
    series_index = []
    expected_p = None

    offset = 0
    for r, series in enumerate(_as_series_list(data)):
        series = np.asarray(series, dtype=np.float32)
        if series.ndim != 2:
            raise ValueError("each time series must have shape [time, variables]")
        if series.shape[0] <= order:
            raise ValueError("each time series must be longer than order")
        if expected_p is None:
            expected_p = series.shape[1]
        elif series.shape[1] != expected_p:
            raise ValueError("all time series must have the same number of variables")

        for t in range(order, series.shape[0]):
            predictors.append(series[t - order : t])
            responses.append(series[t])
            time_index.append(offset + t)
            series_index.append(r)

        offset += series.shape[0] + order

    return LaggedDataset(
        predictors=np.stack(predictors).astype(np.float32),
        responses=np.stack(responses).astype(np.float32),
        time_index=np.asarray(time_index, dtype=np.int64),
        series_index=np.asarray(series_index, dtype=np.int64),
    )


def construct_varx_lagged_dataset(
    endog: np.ndarray | list[np.ndarray],
    exog: np.ndarray | list[np.ndarray],
    *,
    order: int,
    exog_order: int = 0,
    include_current_exog: bool = True,
    exog_features=None,
    exog_names=None,
    exog_lags=None,
    known_future_exog=(),
) -> VARXLaggedDataset:
    """Build aligned predictors for a generic GSEVARX model.

    ``endog`` has shape ``[time, n_endogenous]`` and is predicted from its
    previous ``order`` observations. ``exog`` has shape
    ``[time, n_exogenous]`` and is never predicted. Its included lag numbers
    are ``exog_order, ..., 1, 0`` when ``include_current_exog=True`` and
    ``exog_order, ..., 1`` otherwise. Thus the exogenous predictor axis is
    always ordered from the most distant observation to the most recent one.

    Lists of arrays are supported for multiple independent series. Endogenous
    and exogenous lists must be aligned one-to-one and have equal lengths.

    Alternatively, supply ``exog_names`` and a complete ``exog_lags`` mapping
    for raw columns, or ``exog_features`` for already expanded columns. These
    modes pack valid source/lag pairs into ``[sample, 1, feature]`` and retain
    an ExogenousLayout. They require exog_order=0/include_current_exog=True;
    this describes the packed representation, not the physical lag numbers.
    Raw known-ahead leads require an explicit known_future_exog declaration;
    extended exog rows may cover these leads at the end of an endog block.
    Without such rows, end-of-block responses needing leads are trimmed.
    """
    if exog_features is not None or exog_lags is not None:
        if exog_order != 0 or not include_current_exog:
            raise ValueError("variable-specific inputs require exog_order=0 and include_current_exog=True")
        return _construct_variable_lag_dataset(
            endog, exog, order=order, exog_features=exog_features,
            exog_names=exog_names, exog_lags=exog_lags,
            known_future_exog=known_future_exog,
        )
    if exog_names is not None or known_future_exog:
        raise ValueError("exog_names/known_future_exog require exog_lags")
    if order <= 0:
        raise ValueError("order must be positive")
    if exog_order < 0:
        raise ValueError("exog_order must be non-negative")
    if exog_order == 0 and not include_current_exog:
        raise ValueError(
            "include_current_exog must be True when exog_order is zero"
        )

    endogenous_series = _as_series_list(endog)
    exogenous_series = _as_series_list(exog)
    if len(endogenous_series) != len(exogenous_series):
        raise ValueError(
            "endog and exog must contain the same number of series"
        )

    exogenous_lags = np.arange(exog_order, 0, -1, dtype=np.int64)
    if include_current_exog:
        exogenous_lags = np.concatenate(
            [exogenous_lags, np.asarray([0], dtype=np.int64)]
        )
    max_lag = max(order, exog_order)

    endogenous_predictors = []
    exogenous_predictors = []
    responses = []
    time_index = []
    series_index = []
    expected_endogenous_dim = None
    expected_exogenous_dim = None
    offset = 0

    for series_idx, (endogenous, exogenous) in enumerate(
        zip(endogenous_series, exogenous_series)
    ):
        endogenous = np.asarray(endogenous, dtype=np.float32)
        exogenous = np.asarray(exogenous, dtype=np.float32)
        if endogenous.ndim != 2:
            raise ValueError(
                "each endogenous series must have shape [time, variables]"
            )
        if exogenous.ndim != 2:
            raise ValueError(
                "each exogenous series must have shape [time, variables]"
            )
        if endogenous.shape[0] != exogenous.shape[0]:
            raise ValueError(
                "each aligned endog/exog pair must have the same time length"
            )
        if endogenous.shape[0] <= max_lag:
            raise ValueError(
                "each aligned series must be longer than the largest lag"
            )
        if endogenous.shape[1] == 0 or exogenous.shape[1] == 0:
            raise ValueError("endog and exog must each contain variables")

        if expected_endogenous_dim is None:
            expected_endogenous_dim = endogenous.shape[1]
            expected_exogenous_dim = exogenous.shape[1]
        elif endogenous.shape[1] != expected_endogenous_dim:
            raise ValueError(
                "all endogenous series must have the same number of variables"
            )
        elif exogenous.shape[1] != expected_exogenous_dim:
            raise ValueError(
                "all exogenous series must have the same number of variables"
            )

        for t in range(max_lag, endogenous.shape[0]):
            endogenous_predictors.append(endogenous[t - order : t])
            exogenous_predictors.append(
                np.stack([exogenous[t - lag] for lag in exogenous_lags])
            )
            responses.append(endogenous[t])
            time_index.append(offset + t)
            series_index.append(series_idx)

        offset += endogenous.shape[0] + max_lag

    return VARXLaggedDataset(
        endogenous_predictors=np.stack(endogenous_predictors).astype(
            np.float32
        ),
        exogenous_predictors=np.stack(exogenous_predictors).astype(np.float32),
        responses=np.stack(responses).astype(np.float32),
        time_index=np.asarray(time_index, dtype=np.int64),
        series_index=np.asarray(series_index, dtype=np.int64),
        exogenous_lags=exogenous_lags,
    )
