from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gse_var import ExogenousFeature, ExogenousLayout, GSEVARXTrainingConfig, fit_gse_varx
from gse_var.regularizers import ExogenousGroupRegularizer, NGCRegularizer
from gse_var.varx import _make_regularizers, _validate_varx_config


def config_for(regularizer):
    weights = (
        dict(sparse_group_lambda=0.02, sparse_l1_lambda=0.01)
        if regularizer == "sparse_group_lasso"
        else dict(lambda_ngc=0.03 if regularizer != "none" else 0.0)
    )
    return GSEVARXTrainingConfig(
        order=2, hidden_layer_size=3, regularizer=regularizer,
        max_epochs=1, batch_size=20, verbose=0, device="cpu", **weights,
    )


def layout():
    return ExogenousLayout((
        ExogenousFeature("promo", 0), ExogenousFeature("promo", 2),
        ExogenousFeature("oil", 1), ExogenousFeature("holiday", 0),
    ))


@pytest.mark.parametrize("endog_name", [
    "none", "group_lasso", "hierarchical_group_lasso", "sparse_group_lasso",
])
@pytest.mark.parametrize("exog_name", [
    None, "none", "group_lasso", "hierarchical_group_lasso", "sparse_group_lasso",
])
@pytest.mark.parametrize("packed", [False, True])
def test_omitted_weights_inherit_only_when_used(endog_name, exog_name, packed):
    config = replace(config_for(endog_name), regularizer_exog=exog_name)
    _validate_varx_config(config)
    spec = layout() if packed else None
    endogenous, exogenous = _make_regularizers(config, spec)
    selected = exog_name or endog_name
    expected = dict(
        name=selected,
        lam=config.lambda_ngc if selected in ("group_lasso", "hierarchical_group_lasso") else 0.0,
        sparse_group_lambda=config.sparse_group_lambda if selected == "sparse_group_lasso" else 0.0,
        sparse_l1_lambda=config.sparse_l1_lambda if selected == "sparse_group_lasso" else 0.0,
    )
    reference = (ExogenousGroupRegularizer(layout=spec, **expected) if packed
                 else NGCRegularizer(**expected))
    assert endogenous.name == endog_name
    assert endogenous.lam == config.lambda_ngc
    assert endogenous.sparse_group_lambda == config.sparse_group_lambda
    assert endogenous.sparse_l1_lambda == config.sparse_l1_lambda
    for key, value in expected.items():
        assert getattr(exogenous, key) == value
    # Check both loss penalty and proximal update use the same resolved values.
    gate = torch.arange(1, 9, dtype=torch.float64).reshape(1, 2, 4) / 10
    torch.testing.assert_close(exogenous.penalty(gate), reference.penalty(gate))
    actual, wanted = gate.clone(), gate.clone()
    exogenous.prox_(actual, 0.5, nonnegative=True)
    reference.prox_(wanted, 0.5, nonnegative=True)
    torch.testing.assert_close(actual, wanted)
    assert config.lambda_ngc_exog is None
    assert config.sparse_group_lambda_exog is None
    assert config.sparse_l1_lambda_exog is None


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("exog_name, overrides", [
    ("sparse_group_lasso", dict(lambda_ngc_exog=0.0,
                               sparse_group_lambda_exog=0.07, sparse_l1_lambda_exog=0.09)),
    ("group_lasso", dict(lambda_ngc_exog=0.08)),
    ("hierarchical_group_lasso", dict(lambda_ngc_exog=0.08)),
])
def test_explicit_compatible_weights_override_inheritance(packed, exog_name, overrides):
    config = replace(config_for("group_lasso"), regularizer_exog=exog_name, **overrides)
    _validate_varx_config(config)
    _, exogenous = _make_regularizers(config, layout() if packed else None)
    for field, value in overrides.items():
        attribute = {"lambda_ngc_exog": "lam", "sparse_group_lambda_exog": "sparse_group_lambda",
                     "sparse_l1_lambda_exog": "sparse_l1_lambda"}[field]
        assert getattr(exogenous, attribute) == value


@pytest.mark.parametrize("exog_name, override, message", [
    ("sparse_group_lasso", dict(lambda_ngc_exog=0.1), "lambda_ngc_exog"),
    ("group_lasso", dict(sparse_group_lambda_exog=0.1), "only"),
    ("group_lasso", dict(sparse_l1_lambda_exog=0.1), "only"),
    ("hierarchical_group_lasso", dict(sparse_group_lambda_exog=0.1), "only"),
    ("hierarchical_group_lasso", dict(sparse_l1_lambda_exog=0.1), "only"),
    ("none", dict(sparse_group_lambda_exog=0.1), "only"),
    ("none", dict(sparse_l1_lambda_exog=0.1), "only"),
])
def test_explicit_incompatible_nonzero_weights_are_still_rejected(exog_name, override, message):
    config = replace(config_for("group_lasso"), regularizer_exog=exog_name, **override)
    with pytest.raises(ValueError, match=message):
        _validate_varx_config(config)


@pytest.mark.parametrize("field", [
    "lambda_ngc_exog", "sparse_group_lambda_exog", "sparse_l1_lambda_exog",
])
@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_invalid_explicit_weights_are_not_silently_ignored(field, value):
    config = replace(config_for("group_lasso"), regularizer_exog="none", **{field: value})
    with pytest.raises(ValueError, match=field):
        _validate_varx_config(config)


@pytest.mark.parametrize("exog_name", ["none", "group_lasso", "hierarchical_group_lasso"])
@pytest.mark.parametrize("packed", [False, True])
def test_fit_switch_from_sparse_group_does_not_require_zero_overrides(exog_name, packed):
    rng = np.random.default_rng(41)
    y, x = rng.normal(size=(12, 2)), rng.normal(size=(12, 4))
    config = replace(config_for("sparse_group_lasso"), regularizer_exog=exog_name,
                     lambda_ngc_exog=0.01)
    result = fit_gse_varx(y, x, config, **({"exog_features": layout()} if packed else {}))
    assert result.exogenous_graph.shape == (2, 3 if packed else 4)
    assert all(np.isfinite(values).all() for values in result.history.values())
