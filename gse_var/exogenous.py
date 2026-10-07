"""Explicit source/lag identities for packed, variable-specific VARX inputs."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class ExogenousFeature:
    """One input ``source[t - lag]``; negative lags require known-ahead data.

    ``label`` labels the original source, not a particular lag. Calendar leads
    may be declared known ahead; this declaration is not proof of exogeneity.
    """

    source: str
    lag: int = 0
    label: str | None = None
    known_ahead: bool = False
    kind: str | None = None

    def __post_init__(self):
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("feature source must be a nonempty string")
        if isinstance(self.lag, bool) or not isinstance(self.lag, Integral):
            raise ValueError("feature lag must be an integer")
        if self.lag < 0 and not self.known_ahead:
            raise ValueError("negative lags require known_ahead=True")
        if self.label is not None and (not isinstance(self.label, str) or not self.label):
            raise ValueError("feature label must be a nonempty string or None")


@dataclass(frozen=True)
class ExogenousLayout:
    """Ordered features, retained from dataset construction through plotting.

    Neural inputs remain packed: ``[sample, 1, feature]``. Source-level groups
    and dense lag views are derived from the same identities, without parsing
    column names or changing the coefficient generator.
    """

    features: tuple[ExogenousFeature, ...]

    def __post_init__(self):
        object.__setattr__(self, "features", tuple(self.features))
        if not self.features or any(not isinstance(f, ExogenousFeature) for f in self.features):
            raise ValueError("features must be a nonempty sequence of ExogenousFeature")
        keys = [(f.source, f.lag) for f in self.features]
        if len(set(keys)) != len(keys):
            raise ValueError("each (source, lag) feature must be unique")
        for source in self.sources:
            labels = {f.label for f in self.features if f.source == source and f.label is not None}
            if len(labels) > 1:
                raise ValueError("all features of a source must share its display label")

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(f.source for f in self.features))

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(next((f.label for f in self.features if f.source == s and f.label), s)
                     for s in self.sources)

    @property
    def lags(self) -> np.ndarray:
        return np.asarray(sorted({f.lag for f in self.features}, reverse=True), dtype=np.int64)

    @property
    def groups(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(i for i, f in enumerate(self.features) if f.source == s)
                     for s in self.sources)

    @property
    def mask(self) -> np.ndarray:
        mask = np.zeros((len(self.lags), len(self.sources)), dtype=bool)
        lag_index = {int(l): i for i, l in enumerate(self.lags)}
        source_index = {s: i for i, s in enumerate(self.sources)}
        for f in self.features:
            mask[lag_index[f.lag], source_index[f.source]] = True
        return mask

    def _unpack(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        if values.shape[-1] != len(self.features):
            raise ValueError("packed feature count does not match exogenous layout")
        out = np.full(values.shape[:-1] + self.mask.shape, np.nan,
                      dtype=np.result_type(values.dtype, np.float32))
        lag_index = {int(l): i for i, l in enumerate(self.lags)}
        source_index = {s: i for i, s in enumerate(self.sources)}
        for k, f in enumerate(self.features):
            out[..., lag_index[f.lag], source_index[f.source]] = values[..., k]
        return out

    def gate_by_lag(self, gate: np.ndarray) -> np.ndarray:
        """Return ``[lag, target, original_source]``; unused slots are NaN."""
        gate = np.asarray(gate)
        if gate.ndim != 3 or gate.shape[0] != 1:
            raise ValueError("layout requires packed gates [1, target, feature]")
        return np.moveaxis(self._unpack(gate[0]), -2, 0)

    def coefficients_by_lag(self, coefficients: np.ndarray) -> np.ndarray:
        """Return ``[sample, lag, target, original_source]`` with NaN holes."""
        coefficients = np.asarray(coefficients)
        if coefficients.ndim != 4 or coefficients.shape[1] != 1:
            raise ValueError("layout requires packed coefficients [sample, 1, target, feature]")
        return np.moveaxis(self._unpack(coefficients[:, 0]), -2, 1)


def as_exogenous_layout(features) -> ExogenousLayout:
    return features if isinstance(features, ExogenousLayout) else ExogenousLayout(tuple(features))


def _construct_variable_lag_dataset(
    endog, exog, *, order, exog_features=None, exog_names=None,
    exog_lags=None, known_future_exog=(),
):
    """Construct packed predictors; each independent block is indexed alone."""
    from .data import VARXLaggedDataset, _as_series_list

    if order <= 0:
        raise ValueError("order must be positive")
    raw = exog_lags is not None
    if raw and exog_features is not None:
        raise ValueError("provide exog_lags for raw inputs OR exog_features for expanded inputs")
    if raw:
        if exog_names is None or not exog_names:
            raise ValueError("raw variable-specific lags require exog_names")
        names = tuple(exog_names)
        if len(set(names)) != len(names):
            raise ValueError("exog_names must be unique")
        if set(exog_lags) != set(names):
            raise ValueError("exog_lags must specify every exog_name exactly once")
        known = set(known_future_exog)
        if not known.issubset(names):
            raise ValueError("known_future_exog contains an unknown source")
        layout = ExogenousLayout(tuple(
            ExogenousFeature(source=name, lag=lag, known_ahead=name in known)
            for name in names for lag in exog_lags[name]
        ))
        if any(not tuple(exog_lags[name]) for name in names):
            raise ValueError("each source must have at least one lag")
        source_indices = {name: i for i, name in enumerate(names)}
        expected_q = len(names)
    else:
        if exog_names is not None or known_future_exog:
            raise ValueError("expanded inputs use ExogenousFeature identities and known_ahead")
        layout = as_exogenous_layout(exog_features)
        expected_q = len(layout.features)

    ys, xs = _as_series_list(endog), _as_series_list(exog)
    if len(ys) != len(xs):
        raise ValueError("endog and exog must contain the same number of series")
    y_inputs, x_inputs, targets, times, series_ids = [], [], [], [], []
    expected_p = None
    offset = 0
    for block, (y, x) in enumerate(zip(ys, xs)):
        y, x = np.asarray(y, dtype=np.float32), np.asarray(x, dtype=np.float32)
        if y.ndim != 2 or x.ndim != 2 or not y.shape[1]:
            raise ValueError("each input must have shape [time, variables]")
        if x.shape[1] != expected_q:
            raise ValueError("exog column count does not match the supplied identities")
        if expected_p is None:
            expected_p = y.shape[1]
        elif y.shape[1] != expected_p:
            raise ValueError("all endogenous blocks must have the same variable count")
        if (not raw and len(x) != len(y)) or (raw and len(x) < len(y)):
            raise ValueError("aligned exog must cover endog; expanded input lengths must match")
        start = max(order, max(0, int(layout.lags.max()))) if raw else order
        stop = min(len(y), len(x) + min(0, int(layout.lags.min()))) if raw else len(y)
        if stop <= start:
            raise ValueError("each block must contain responses after lag/lead trimming")
        for t in range(start, stop):
            features = (np.asarray([x[t - f.lag, source_indices[f.source]] for f in layout.features])
                        if raw else x[t])
            predictors = y[t - order:t]
            if not (np.isfinite(predictors).all() and np.isfinite(y[t]).all()
                    and np.isfinite(features).all()):
                raise ValueError("used predictors and responses must be finite; split missing-date blocks first")
            y_inputs.append(predictors)
            x_inputs.append(features[None, :])
            targets.append(y[t])
            times.append(offset + t)
            series_ids.append(block)
        offset += len(y) + order
    return VARXLaggedDataset(
        endogenous_predictors=np.stack(y_inputs).astype(np.float32),
        exogenous_predictors=np.stack(x_inputs).astype(np.float32),
        responses=np.stack(targets).astype(np.float32),
        time_index=np.asarray(times, dtype=np.int64),
        series_index=np.asarray(series_ids, dtype=np.int64),
        exogenous_lags=layout.lags.copy(), exogenous_layout=layout,
    )
