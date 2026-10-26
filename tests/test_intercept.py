from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gse_var import (
    ExogenousFeature, ExogenousLayout, GSEVAR, GSEVARX,
    GSEVARTrainingConfig, GSEVARXTrainingConfig,
    construct_lagged_dataset, construct_varx_lagged_dataset,
    fit_gse_var, fit_gse_varx,
)
from gse_var.regularizers import NGCRegularizer
from gse_var.training import _make_endogenous_regularizer, _make_optimizer
from gse_var.varx import _make_optimizer as _make_varx_optimizer, _make_regularizers


def test_both_training_defaults_use_edge_sgl_and_separate_intercept():
    for config_type in (GSEVARTrainingConfig, GSEVARXTrainingConfig):
        config = config_type(order=2, hidden_layer_size=3)
        assert config.regularizer == "sparse_group_lasso"
        assert not hasattr(config, "endogenous_grouping")
        assert not hasattr(config, "fit_intercept")
        assert isinstance(_make_endogenous_regularizer(config), NGCRegularizer)
    endogenous, exogenous = _make_regularizers(GSEVARXTrainingConfig(order=2, hidden_layer_size=3))
    assert isinstance(endogenous, NGCRegularizer)
    assert isinstance(exogenous, NGCRegularizer)
    assert exogenous.name == "sparse_group_lasso"


@pytest.mark.parametrize("regularizer", ["none", "group_lasso", "hierarchical_group_lasso"])
def test_explicit_non_sgl_regularizers_keep_their_existing_definition(regularizer):
    config = GSEVARTrainingConfig(order=2, hidden_layer_size=3, regularizer=regularizer)
    assert isinstance(_make_endogenous_regularizer(config), NGCRegularizer)


def make_case(varx):
    if varx:
        model = GSEVARX(num_endogenous=2, num_exogenous=3, order=2, exog_order=0,
                        include_current_exog=True, hidden_layer_size=3).double()
        inputs = (torch.randn(5, 2, 2, dtype=torch.float64),
                  torch.randn(5, 1, 3, dtype=torch.float64))
        config = GSEVARXTrainingConfig(order=2, hidden_layer_size=3)
    else:
        model = GSEVAR(num_vars=2, order=2, hidden_layer_size=3).double()
        inputs = (torch.randn(5, 2, 2, dtype=torch.float64),)
        config = GSEVARTrainingConfig(order=2, hidden_layer_size=3)
    return model, inputs, config


@pytest.mark.parametrize("varx", [False, True])
def test_intercept_changes_predictions_but_not_coefficients_jacobians_or_graphs(varx):
    torch.manual_seed(23)
    model, inputs, _ = make_case(varx)
    jacobian_forward = model.forward_with_jacobians if varx else model.forward_with_jacobian
    before = jacobian_forward(*inputs)
    with torch.no_grad():
        model.intercept.copy_(torch.tensor([3.0, -2.0], dtype=torch.float64))
    after = jacobian_forward(*inputs)
    torch.testing.assert_close(after[0], before[0] + model.intercept)
    torch.testing.assert_close(model(*inputs)[0], after[0])
    for left, right in zip(before[1:], after[1:]):
        torch.testing.assert_close(left, right)
    if varx:
        expected_y = after[3]
        expected_x = after[4]
    else:
        expected_y = after[2]
    # Direct differentiation verifies the optimized Jacobian with nonzero intercepts.
    differentiable_inputs = tuple(x.detach().clone().requires_grad_(True) for x in inputs)
    predictions = model(*differentiable_inputs)[0]
    for source_axis, wanted in enumerate((expected_y, expected_x) if varx else (expected_y,)):
        actual = torch.stack([torch.autograd.grad(predictions[:, target].sum(),
                             differentiable_inputs[source_axis], retain_graph=True)[0]
                             for target in range(2)], dim=2)
        torch.testing.assert_close(actual, wanted)
    with torch.no_grad():
        if varx:
            model.endogenous_gate.zero_()
            model.exogenous_gate.zero_()
            assert model.endogenous_graph_from_gate().count_nonzero() == 0
            assert model.exogenous_graph_from_gate().count_nonzero() == 0
        else:
            model.causal_gate.zero_()
            assert model.causal_graph_from_gate().count_nonzero() == 0
    torch.testing.assert_close(model(*inputs)[0], model.intercept.expand(5, -1))


@pytest.mark.parametrize("varx", [False, True])
def test_intercept_is_optimized_without_weight_decay(varx):
    model, _, config = make_case(varx)
    config = replace(config, coefficient_weight_decay=2.0)
    optimizer = (_make_varx_optimizer if varx else _make_optimizer)(config, model)
    group = next(group for group in optimizer.param_groups
                 if any(parameter is model.intercept for parameter in group["params"]))
    assert group["weight_decay"] == 0.0
    coefficient = next(model.endogenous_coeff_net.parameters() if varx else model.coeff_net.parameters())
    coefficient_group = next(group for group in optimizer.param_groups
                             if any(parameter is coefficient for parameter in group["params"]))
    assert coefficient_group["weight_decay"] == 0.0
    assert coefficient_group["objective_weight_decay"] == 2.0
    with torch.no_grad():
        model.intercept.fill_(4.0)
    model.intercept.grad = torch.zeros_like(model.intercept)
    optimizer.step()
    torch.testing.assert_close(model.intercept, model.intercept.new_full((2,), 4.0))


@pytest.mark.parametrize("varx", [False, True])
def test_intercept_first_update_uses_training_residuals_from_zero(varx):
    rng = np.random.default_rng(32)
    # The first residual is the response because gates and intercept start
    # at zero. The gate prox keeps these strongly penalized routes at zero.
    y = np.tile(np.array([[4.0, -2.0]]), (18, 1))
    x = rng.normal(size=(18, 3))
    config_type = GSEVARXTrainingConfig if varx else GSEVARTrainingConfig
    config = config_type(order=2, hidden_layer_size=3, max_epochs=1, batch_size=50,
                         gate_init=0.0, sparse_group_lambda=100.0, sparse_l1_lambda=100.0,
                         coefficient_weight_decay=1.0,
                         verbose=0, device="cpu")
    if varx:
        # A validation-only offset must never enter the training update.
        validation = construct_varx_lagged_dataset(y + 100, x, order=2)
        result = fit_gse_varx(y, x, config, validation_data=validation)
        data = construct_varx_lagged_dataset(y, x, order=2)
        args = (torch.as_tensor(data.endogenous_predictors),
                torch.as_tensor(data.exogenous_predictors))
        assert result.endogenous_graph.sum() == result.exogenous_graph.sum() == 0
    else:
        result = fit_gse_var(y, config)
        data = construct_lagged_dataset(y, order=2)
        args = (torch.as_tensor(data.predictors),)
        assert result.causal_graph.sum() == 0
    gradient = -2 * data.responses.mean(axis=0) / data.responses.shape[1]
    expected = -config.learning_rate * gradient / (np.abs(gradient) + 1e-8)
    np.testing.assert_allclose(result.intercept, expected, atol=1e-5)
    prediction = result.model(*args)[0].detach().numpy()
    np.testing.assert_allclose(prediction, np.broadcast_to(expected, prediction.shape), atol=1e-5)
    copied = result.intercept
    copied[:] = 0
    np.testing.assert_allclose(result.intercept, expected, atol=1e-5)


@pytest.mark.parametrize("varx", [False, True])
def test_provided_models_keep_their_intercept_and_intercept_mismatch_is_rejected(varx):
    model, _, config = make_case(varx)
    model.float()  # Fit APIs use float32 training data.
    with torch.no_grad():
        model.intercept.fill_(7.0)
    y, x = np.ones((7, 2)), np.ones((7, 3))
    config = replace(config, max_epochs=0, verbose=0, device="cpu")
    fit = (lambda cfg, m: fit_gse_varx(y, x, cfg, model=m)) if varx else (
        lambda cfg, m: fit_gse_var(y, cfg, model=m))
    result = fit(config, model)
    np.testing.assert_allclose(result.intercept, [7.0, 7.0])
    model.register_parameter("intercept", None)
    with pytest.raises(ValueError, match="intercept"):
        fit(config, model)


@pytest.mark.parametrize("varx", [False, True])
@pytest.mark.parametrize("gate_init", [0.0, 1.0])
def test_new_model_intercept_starts_at_zero_without_using_responses(varx, gate_init):
    rng = np.random.default_rng(19)
    y = rng.normal(size=(13, 2)) + [4.0, -2.0]
    x = rng.normal(size=(13, 3))
    config_type = GSEVARXTrainingConfig if varx else GSEVARTrainingConfig
    config = config_type(order=2, hidden_layer_size=3, max_epochs=0,
                         batch_size=3, gate_init=gate_init, verbose=0, device="cpu")
    if varx:
        data = construct_varx_lagged_dataset(y, x, order=2)
        result = fit_gse_varx(y, x, config)
        args = (torch.as_tensor(data.endogenous_predictors),
                torch.as_tensor(data.exogenous_predictors))
    else:
        data = construct_lagged_dataset(y, order=2)
        result = fit_gse_var(y, config)
        args = (torch.as_tensor(data.predictors),)
    prediction = result.model(*args)[0].detach().numpy()
    np.testing.assert_array_equal(result.intercept, np.zeros(2))
    latest = data.endogenous_predictors[:, -1] if varx else data.predictors[:, -1]
    np.testing.assert_allclose(prediction, gate_init * latest, atol=1e-6)


@pytest.mark.parametrize("varx", [False, True])
def test_nonzero_intercept_state_dict_roundtrip(varx):
    model, inputs, _ = make_case(varx)
    with torch.no_grad():
        model.intercept.copy_(torch.tensor([4.0, -2.0], dtype=torch.float64))
    restored, _, _ = make_case(varx)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(*inputs)[0], model(*inputs)[0])


@pytest.mark.parametrize("varx", [False, True])
def test_removed_intercept_option_is_rejected(varx):
    config_type = GSEVARXTrainingConfig if varx else GSEVARTrainingConfig
    with pytest.raises(TypeError, match="fit_intercept"):
        config_type(order=2, hidden_layer_size=3, fit_intercept=False)


def test_ungated_gvar_baseline_default_is_unchanged():
    model = GSEVAR(num_vars=2, order=2, hidden_layer_size=3, use_causal_gate=False)
    assert model.intercept is None
    assert "intercept" not in model.state_dict()
