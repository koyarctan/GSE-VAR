from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gse_var import (
    GSEVAR, GSEVARX, GSEVARTrainingConfig, GSEVARXTrainingConfig, construct_lagged_dataset,
    construct_varx_lagged_dataset, fit_gse_var, fit_gse_varx,
)
from gse_var.regularizers import NGCRegularizer
from gse_var.training import (
    _alternating_proximal_step, _make_endogenous_regularizer,
    temporal_smoothness_penalty,
)
from gse_var.varx import _make_regularizers


def quadratic_case(step=0.001, *, theta=0.0, gate=1.0, ridge=0.0):
    coefficient = torch.nn.Parameter(torch.tensor([theta], dtype=torch.float64))
    mask = torch.nn.Parameter(torch.full((2, 1, 1), gate, dtype=torch.float64))
    optimizer = torch.optim.Adam([
        dict(params=[coefficient], weight_decay=0.0, objective_weight_decay=ridge),
        dict(params=[mask], weight_decay=0.0),
        dict(params=[], weight_decay=0.0),
    ], lr=step)
    regularizer = NGCRegularizer(sparse_l1_lambda=0.001, sparse_group_lambda=0.003)

    def evaluate(create_graph):
        with torch.set_grad_enabled(create_graph):
            mse = (coefficient - 2).square().sum()
            zero = mse.new_zeros(())
            decay = 0.5 * ridge * coefficient.square().sum()
            penalty = regularizer.penalty(mask)
            return dict(loss=mse + decay + penalty, mse=mse, smooth=zero,
                        jacobian=zero, weight_decay=decay, ngc=penalty)

    return coefficient, mask, optimizer, regularizer, evaluate


def test_unused_gate_uses_the_requested_fixed_proximal_step():
    _, gate, optimizer, regularizer, evaluate = quadratic_case()
    for _ in range(25):
        metrics = _alternating_proximal_step(optimizer, evaluate, ((gate, regularizer),))
    fixed = torch.ones_like(gate)
    for _ in range(25):
        regularizer.prox_(fixed, 0.001, nonnegative=True)
    torch.testing.assert_close(gate, fixed)
    assert metrics["gate_step_size"] == pytest.approx(0.001)
    assert metrics["coefficient_step_size"] == pytest.approx(0.001)
    assert all(group["lr"] == 0.001 and "next_step_size" not in group for group in optimizer.param_groups)


def test_fixed_updates_include_ridge_once_and_log_the_pre_update_objective():
    coefficient, gate, optimizer, regularizer, evaluate = quadratic_case(step=0.01, theta=1, ridge=0.4)
    before = {name: value.detach().clone() for name, value in evaluate(False).items()}
    metrics = _alternating_proximal_step(optimizer, evaluate, ((gate, regularizer),))
    gradient = 2*(1-2)+.4
    assert coefficient.item() == pytest.approx(1 - .01*gradient/(abs(gradient)+1e-8))
    for name in before:
        torch.testing.assert_close(metrics[name], before[name])
    assert all(parameter.grad is None for group in optimizer.param_groups for parameter in group["params"])


def test_exactly_two_gradient_evaluations_without_trials_or_logging_forward():
    coefficient, gate, optimizer, regularizer, evaluate = quadratic_case(step=0.01)
    calls = []

    def counted(create_graph):
        calls.append((create_graph, coefficient.item()))
        return evaluate(create_graph)

    _alternating_proximal_step(optimizer, counted, ((gate, regularizer),))
    assert calls[0] == (True, 0.0)
    assert calls[1][0] is True and calls[1][1] == pytest.approx(.01)


def test_gate_gradient_is_recomputed_after_coefficient_update():
    coefficient, gate, optimizer, regularizer, _ = quadratic_case(step=0.01, theta=1)

    def evaluate(create_graph):
        with torch.set_grad_enabled(create_graph):
            mse = (coefficient * gate.sum() - 3).square().sum()
            zero = mse.new_zeros(())
            penalty = regularizer.penalty(gate)
            return dict(loss=mse + penalty, mse=mse, smooth=zero,
                        jacobian=zero, weight_decay=zero, ngc=penalty)

    gradient = 2*(2-3)*2
    expected_coefficient = 1 - .01*gradient/(abs(gradient)+1e-8)
    expected_gate = torch.ones_like(gate) - 0.01 * (2 * (expected_coefficient * 2 - 3) * expected_coefficient)
    regularizer.prox_(expected_gate, 0.01, nonnegative=True)
    _alternating_proximal_step(optimizer, evaluate, ((gate, regularizer),))
    assert coefficient.item() == pytest.approx(expected_coefficient)
    torch.testing.assert_close(gate, expected_gate)


def test_nonfinite_gradient_is_reported_before_mutating_parameters():
    coefficient, gate, optimizer, regularizer, _ = quadratic_case(theta=-1)

    def evaluate(create_graph):
        with torch.set_grad_enabled(create_graph):
            mse = coefficient.sqrt().sum()
            zero = mse.new_zeros(())
            return dict(loss=mse, mse=mse, ngc=zero, smooth=zero, jacobian=zero, weight_decay=zero)

    with pytest.raises(FloatingPointError, match="non-finite loss or gradient"):
        _alternating_proximal_step(optimizer, evaluate, ((gate, regularizer),))
    assert coefficient.item() == -1
    assert torch.all(gate == 1)


@pytest.mark.parametrize("varx", [False, True])
@pytest.mark.parametrize("regularizer", ["sparse_group_lasso", "group_lasso", "hierarchical_group_lasso"])
def test_real_fit_removes_prediction_irrelevant_gates_and_preserves_nonnegativity(varx, regularizer):
    config_type = GSEVARXTrainingConfig if varx else GSEVARTrainingConfig
    config = config_type(order=2, hidden_layer_size=3, max_epochs=25, batch_size=50,
                         learning_rate=20.0,
                         regularizer=regularizer, shuffle=False, verbose=0, device="cpu")
    config = (replace(config, sparse_group_lambda=0.003, sparse_l1_lambda=0.001)
              if regularizer == "sparse_group_lasso" else replace(config, lambda_ngc=0.003))
    data = np.zeros((14, 2), dtype="float32")
    result = fit_gse_varx(data, data, config) if varx else fit_gse_var(data, config)
    gates = ((result.model.endogenous_gate, result.model.exogenous_gate) if varx
             else (result.model.causal_gate,))
    assert all(gate.count_nonzero() == 0 for gate in gates)
    assert result.history["mse"] == [0.0] * config.max_epochs
    assert np.all(np.diff(result.history["loss"]) <= 1e-6)


@pytest.mark.parametrize("varx", [False, True])
def test_full_batch_logged_components_describe_the_same_pre_update_model(varx):
    rng = np.random.default_rng(812)
    y, x = rng.normal(size=(16, 2)).astype("float32"), rng.normal(size=(16, 3)).astype("float32")
    config_type = GSEVARXTrainingConfig if varx else GSEVARTrainingConfig
    config = config_type(order=2, hidden_layer_size=3, max_epochs=1, batch_size=100,
                         lambda_smooth=0.01, coefficient_weight_decay=0.2,
                         sparse_group_lambda=0.02, sparse_l1_lambda=0.01,
                         shuffle=False, verbose=0, device="cpu")
    if varx:
        model = GSEVARX(num_endogenous=2, num_exogenous=3, order=2, hidden_layer_size=3,
                        exog_order=0, include_current_exog=True)
        ds = construct_varx_lagged_dataset(y, x, order=2)
        args = (torch.tensor(ds.endogenous_predictors), torch.tensor(ds.exogenous_predictors))
        regs = _make_regularizers(config)
        gates = (model.endogenous_gate, model.exogenous_gate)
        networks = (model.endogenous_coeff_net, model.exogenous_coeff_net)
    else:
        model = GSEVAR(num_vars=2, order=2, hidden_layer_size=3)
        ds = construct_lagged_dataset(y, order=2)
        args = (torch.tensor(ds.predictors),)
        regs, gates = (_make_endogenous_regularizer(config),), (model.causal_gate,)
        networks = (model.coeff_net,)
    with torch.no_grad():
        outputs = model(*args)
        mse = (outputs[0] - torch.tensor(ds.responses)).square().mean()
        smooth = sum(config.lambda_smooth * temporal_smoothness_penalty(c, torch.tensor(ds.time_index))
                     for c in outputs[1:])
        decay = 0.5 * config.coefficient_weight_decay * sum(
            p.square().sum() for network in networks for p in network.parameters())
        penalty = sum(reg.penalty(gate) for reg, gate in zip(regs, gates))
    expected = dict(mse=mse, smooth=smooth, ngc=penalty, weight_decay=decay,
                    loss=mse + smooth + penalty + decay)
    result = fit_gse_varx(y, x, config, model=model) if varx else fit_gse_var(y, config, model=model)
    for name, value in expected.items():
        assert result.history[name][-1] == pytest.approx(float(value), rel=1e-5, abs=1e-7)


@pytest.mark.parametrize("field", ["learning_rate", "lambda_ngc", "lambda_smooth",
                                    "sparse_l1_lambda", "sparse_group_lambda", "coefficient_weight_decay"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_training_rejects_invalid_scales(field, value):
    config = GSEVARTrainingConfig(order=1, hidden_layer_size=2, **{field: value}, verbose=0)
    with pytest.raises(ValueError, match=field):
        fit_gse_var(np.zeros((5, 2)), config)


@pytest.mark.parametrize("varx", [False, True])
@pytest.mark.parametrize("frozen", ["gate", "coefficient", "all"])
def test_alternating_updates_preserve_frozen_parameters(varx, frozen):
    rng = np.random.default_rng(84)
    y, x = rng.normal(size=(10, 2)), rng.normal(size=(10, 3))
    config_type = GSEVARXTrainingConfig if varx else GSEVARTrainingConfig
    config = config_type(order=2, hidden_layer_size=3, max_epochs=2, batch_size=50,
                         coefficient_weight_decay=0.1, sparse_group_lambda=0.01,
                         shuffle=False, verbose=0, device="cpu")
    model = (GSEVARX(num_endogenous=2, num_exogenous=3, order=2, hidden_layer_size=3,
                     exog_order=0, include_current_exog=True) if varx else
             GSEVAR(num_vars=2, order=2, hidden_layer_size=3))
    for name, parameter in model.named_parameters():
        is_gate = name.endswith("gate")
        if frozen == "all" or (frozen == "gate" and is_gate) or (frozen == "coefficient" and not is_gate):
            parameter.requires_grad_(False)
    originals = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    result = fit_gse_varx(y, x, config, model=model) if varx else fit_gse_var(y, config, model=model)
    for name, parameter in result.model.named_parameters():
        if name in originals:
            torch.testing.assert_close(parameter, originals[name], rtol=0, atol=0)
