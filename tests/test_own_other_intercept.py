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
from gse_var.regularizers import NGCRegularizer, OwnOtherSparseGroupRegularizer
from gse_var.training import _make_endogenous_regularizer, _make_optimizer, _validate_config
from gse_var.varx import _make_optimizer as _make_varx_optimizer, _make_regularizers


@pytest.mark.parametrize("nonnegative", [False, True])
@pytest.mark.parametrize("num_vars", [1, 2, 4])
def test_own_other_sgl_matches_disjoint_group_closed_form(nonnegative, num_vars):
    values = torch.linspace(-0.4, 0.9, 3 * num_vars**2, dtype=torch.float64).reshape(
        3, num_vars, num_vars,
    )
    reg = OwnOtherSparseGroupRegularizer(sparse_group_lambda=0.3, sparse_l1_lambda=0.1)
    diagonal = torch.eye(num_vars, dtype=torch.bool)
    groups = (diagonal, ~diagonal) if num_vars > 1 else (diagonal,)
    expected_penalty = 0.1 * values.abs().sum()
    wanted = values.clone()
    for lag in range(3):
        for group in groups:
            block = values[lag][group]
            expected_penalty += 0.3 * block.norm()
            shrunk = ((block - 0.05).clamp(min=0) if nonnegative else
                      block.sign() * (block.abs() - 0.05).clamp(min=0))
            norm = shrunk.norm().item()
            wanted[lag][group] = shrunk * max(0, 1 - 0.15 / max(norm, 1e-12))
    torch.testing.assert_close(reg.penalty(values), expected_penalty)
    got = values.clone()
    reg.prox_(got, 0.5, nonnegative=nonnegative)
    torch.testing.assert_close(got, wanted)


def test_own_and_other_are_selected_independently_without_cardinality_weights():
    reg = OwnOtherSparseGroupRegularizer(sparse_group_lambda=0.4, sparse_l1_lambda=0.1)
    gate = torch.tensor([[[1.0, 0.01], [0.01, 1.0]],
                         [[0.01, 1.0], [1.0, 0.01]]], dtype=torch.float64)
    reg.prox_(gate, 1.0, nonnegative=True)
    assert gate[0].diagonal().count_nonzero() == 2
    assert gate[0, 0, 1] == gate[0, 1, 0] == 0
    assert gate[1].diagonal().count_nonzero() == 0
    assert gate[1, 0, 1] > 0 and gate[1, 1, 0] > 0
    # Own edges are not protected from zeroing.
    reg.prox_(gate, 100.0, nonnegative=True)
    assert gate.count_nonzero() == 0


def test_own_other_penalty_gradient_is_correct():
    gate = torch.linspace(0.1, 1.8, 18, dtype=torch.float64).reshape(2, 3, 3)
    gate.requires_grad_(True)
    reg = OwnOtherSparseGroupRegularizer(sparse_group_lambda=0.2, sparse_l1_lambda=0.1)
    assert torch.autograd.gradcheck(reg.penalty, (gate,))


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4), (1, 2, 3, 3)])
def test_own_other_rejects_non_endogenous_gate_shapes(shape):
    reg = OwnOtherSparseGroupRegularizer()
    with pytest.raises(ValueError, match="square"):
        reg.penalty(torch.ones(shape))
    with pytest.raises(ValueError, match="square"):
        reg.prox_(torch.ones(shape), 1.0)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["sparse_group_lambda", "sparse_l1_lambda"])
def test_own_other_rejects_invalid_strengths(field, value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        OwnOtherSparseGroupRegularizer(**{field: value})


def test_both_training_defaults_use_own_other_sgl_and_separate_intercept():
    for config_type in (GSEVARTrainingConfig, GSEVARXTrainingConfig):
        config = config_type(order=2, hidden_layer_size=3)
        assert config.regularizer == "sparse_group_lasso"
        assert config.endogenous_grouping == "own_other"
        assert config.fit_intercept is True
        assert isinstance(_make_endogenous_regularizer(config), OwnOtherSparseGroupRegularizer)
        assert isinstance(_make_endogenous_regularizer(replace(config, endogenous_grouping="edge")),
                          NGCRegularizer)
    endogenous, exogenous = _make_regularizers(GSEVARXTrainingConfig(order=2, hidden_layer_size=3))
    assert isinstance(endogenous, OwnOtherSparseGroupRegularizer)
    assert isinstance(exogenous, NGCRegularizer)
    assert exogenous.name == "sparse_group_lasso"


@pytest.mark.parametrize("regularizer", ["none", "group_lasso", "hierarchical_group_lasso"])
def test_explicit_non_sgl_regularizers_keep_their_existing_definition(regularizer):
    config = GSEVARTrainingConfig(order=2, hidden_layer_size=3, regularizer=regularizer)
    assert isinstance(_make_endogenous_regularizer(config), NGCRegularizer)


@pytest.mark.parametrize("field,value", [("endogenous_grouping", "unknown"),
                                          ("fit_intercept", 1), ("fit_intercept", None)])
def test_new_config_options_are_validated(field, value):
    config = GSEVARTrainingConfig(order=2, hidden_layer_size=3, **{field: value})
    with pytest.raises(ValueError, match=field):
        _validate_config(config)


def make_case(varx, *, fit_intercept=True):
    if varx:
        model = GSEVARX(num_endogenous=2, num_exogenous=3, order=2, exog_order=0,
                        include_current_exog=True, hidden_layer_size=3,
                        fit_intercept=fit_intercept).double()
        inputs = (torch.randn(5, 2, 2, dtype=torch.float64),
                  torch.randn(5, 1, 3, dtype=torch.float64))
        config = GSEVARXTrainingConfig(order=2, hidden_layer_size=3)
    else:
        model = GSEVAR(num_vars=2, order=2, hidden_layer_size=3,
                       fit_intercept=fit_intercept).double()
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
@pytest.mark.parametrize("optimizer_name", ["ista", "adam"])
def test_intercept_is_optimized_without_weight_decay(varx, optimizer_name):
    model, _, config = make_case(varx)
    config = replace(config, optimizer=optimizer_name, coefficient_weight_decay=2.0)
    optimizer = (_make_varx_optimizer if varx else _make_optimizer)(config, model)
    group = next(group for group in optimizer.param_groups
                 if any(parameter is model.intercept for parameter in group["params"]))
    assert group["weight_decay"] == 0.0
    coefficient = next(model.endogenous_coeff_net.parameters() if varx else model.coeff_net.parameters())
    coefficient_group = next(group for group in optimizer.param_groups
                             if any(parameter is coefficient for parameter in group["params"]))
    assert coefficient_group["weight_decay"] == 2.0
    with torch.no_grad():
        model.intercept.fill_(4.0)
    model.intercept.grad = torch.zeros_like(model.intercept)
    optimizer.step()
    torch.testing.assert_close(model.intercept, model.intercept.new_full((2,), 4.0))


@pytest.mark.parametrize("varx", [False, True])
@pytest.mark.parametrize("optimizer", ["ista", "adam"])
def test_zero_gate_fit_learns_unpenalized_intercept_from_zero(varx, optimizer):
    rng = np.random.default_rng(32)
    # Strong proximal penalties keep all gates zero. The intercept must learn
    # the constant response rather than being initialized from its mean.
    y = np.tile(np.array([[4.0, -2.0]]), (18, 1))
    x = rng.normal(size=(18, 3))
    config_type = GSEVARXTrainingConfig if varx else GSEVARTrainingConfig
    config = config_type(order=2, hidden_layer_size=3, max_epochs=1, batch_size=50,
                         gate_init=0.0, sparse_group_lambda=100.0, sparse_l1_lambda=100.0,
                         optimizer=optimizer, coefficient_weight_decay=1.0,
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
    expected = (-config.learning_rate * gradient if optimizer == "ista" else
                -config.learning_rate * gradient / (np.abs(gradient) + 1e-8))
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
    with pytest.raises(ValueError, match="fit_intercept"):
        fit(replace(config, fit_intercept=False), model)


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
def test_nonzero_intercept_state_dict_roundtrip_and_legacy_state_dict(varx):
    model, inputs, _ = make_case(varx)
    with torch.no_grad():
        model.intercept.copy_(torch.tensor([4.0, -2.0], dtype=torch.float64))
    restored, _, _ = make_case(varx)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(*inputs)[0], model(*inputs)[0])
    legacy, _, _ = make_case(varx, fit_intercept=False)
    legacy_restored, _, _ = make_case(varx, fit_intercept=False)
    legacy_restored.load_state_dict(legacy.state_dict())
    torch.testing.assert_close(legacy_restored(*inputs)[0], legacy(*inputs)[0])


@pytest.mark.parametrize("varx", [False, True])
def test_fit_without_intercept_retains_intercept_free_model(varx):
    y, x = np.ones((7, 2)), np.ones((7, 3))
    config_type = GSEVARXTrainingConfig if varx else GSEVARTrainingConfig
    config = config_type(order=2, hidden_layer_size=3, max_epochs=0,
                         fit_intercept=False, verbose=0, device="cpu")
    result = fit_gse_varx(y, x, config) if varx else fit_gse_var(y, config)
    assert result.intercept is None
    assert "intercept" not in result.model.state_dict()


@pytest.mark.parametrize("varx", [False, True])
def test_no_intercept_mode_and_legacy_model_inference_remain_available(varx):
    model, inputs, _ = make_case(varx, fit_intercept=False)
    assert "intercept" not in model.state_dict()
    original = model(*inputs)[0]
    # Simulate an older pickled model that predates the new parameter entirely.
    del model.intercept
    torch.testing.assert_close(model(*inputs)[0], original)


def test_ungated_gvar_baseline_default_is_unchanged():
    model = GSEVAR(num_vars=2, order=2, hidden_layer_size=3, use_causal_gate=False)
    assert model.intercept is None
    assert "intercept" not in model.state_dict()
