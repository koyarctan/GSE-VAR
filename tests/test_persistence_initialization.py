import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gse_var import (
    GSEVAR, GSEVARX, GSEVARTrainingConfig, GSEVARXTrainingConfig,
    fit_gse_var, fit_gse_varx,
)
from gse_var.models import LagwiseMLP


def make_model(varx, order=3, layers=2, fit_intercept=True, gate_init=1.0):
    common = dict(order=order, hidden_layer_size=5, num_hidden_layers=layers,
                  fit_intercept=fit_intercept, gate_init=gate_init)
    if varx:
        return GSEVARX(num_endogenous=3, num_exogenous=2, exog_order=2,
                       include_current_exog=True, **common)
    return GSEVAR(num_vars=3, **common)


def make_inputs(varx, order):
    endogenous = torch.randn(4, order, 3)
    return (endogenous, torch.randn(4, 3, 2)) if varx else (endogenous,)


def assert_persistence(model, inputs, varx):
    output = model(*inputs)
    expected = torch.zeros_like(output[1])
    expected[:, -1] = torch.eye(3, device=expected.device, dtype=expected.dtype)
    torch.testing.assert_close(output[1], expected, rtol=0, atol=0)
    torch.testing.assert_close(output[0], inputs[0][:, -1], rtol=0, atol=0)
    if varx:
        assert torch.count_nonzero(output[2]) == 0
        assert torch.all(model.endogenous_gate == 1)
        assert torch.all(model.exogenous_gate == 1)
    else:
        assert torch.all(model.causal_gate == 1)
    if model.intercept is not None:
        assert torch.count_nonzero(model.intercept) == 0
    assert all(parameter.requires_grad for parameter in model.parameters())


@pytest.mark.parametrize("varx", [False, True])
@pytest.mark.parametrize("order", [1, 3])
@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_default_predictions_copy_only_the_most_recent_observation(
    varx, order, layers, fit_intercept,
):
    torch.manual_seed(142)
    model = make_model(varx, order, layers, fit_intercept)
    assert_persistence(model, make_inputs(varx, order), varx)


@pytest.mark.parametrize("varx", [False, True])
def test_reset_parameters_restores_coefficient_and_intercept_initialization(varx):
    model = make_model(varx)
    networks = ((model.endogenous_coeff_net, model.exogenous_coeff_net)
                if varx else (model.coeff_net,))
    with torch.no_grad():
        for network in networks:
            for parameter in network.parameters():
                parameter.normal_()
        model.intercept.fill_(9.0)
    model.reset_parameters()
    assert_persistence(model, make_inputs(varx, 3), varx)


@pytest.mark.parametrize("varx", [False, True])
@pytest.mark.parametrize("gate_init", [0.0, 0.25, 2.0])
def test_existing_nondefault_gate_initialization_is_respected(varx, gate_init):
    model = make_model(varx, gate_init=gate_init)
    inputs = make_inputs(varx, 3)
    torch.testing.assert_close(model(*inputs)[0], gate_init * inputs[0][:, -1])
    gate = model.endogenous_gate if varx else model.causal_gate
    assert torch.all(gate == gate_init)
    if gate_init == 0:
        # No unpenalized self-edge or persistence skip path bypasses a zero gate.
        assert torch.count_nonzero(model(*inputs)[0]) == 0


@pytest.mark.parametrize("varx", [False, True])
def test_zero_output_weights_do_not_freeze_hidden_layers_or_gates(varx):
    torch.manual_seed(144)
    model = make_model(varx)
    inputs = make_inputs(varx, 3)
    target = torch.randn(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    networks = ((model.endogenous_coeff_net, model.exogenous_coeff_net)
                if varx else (model.coeff_net,))
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        (model(*inputs)[0] - target).square().mean().backward()
        optimizer.step()
    for network in networks:
        assert torch.count_nonzero(network.weights[-1]) > 0
        assert torch.count_nonzero(network.weights[0].grad) > 0
    gates = ((model.endogenous_gate, model.exogenous_gate)
             if varx else (model.causal_gate,))
    for gate in gates:
        assert torch.count_nonzero(gate.grad) > 0


@pytest.mark.parametrize("varx", [False, True])
def test_fit_initial_mse_matches_persistence_without_a_manual_model(varx):
    rng = np.random.default_rng(145)
    y = rng.normal(size=(15, 3)).astype("float32") + np.array([8, -3, 2], dtype="float32")
    x = rng.normal(size=(15, 2)).astype("float32")
    config_type = GSEVARXTrainingConfig if varx else GSEVARTrainingConfig
    config = config_type(order=3, hidden_layer_size=5, max_epochs=1,
                         batch_size=50, shuffle=False, verbose=0, device="cpu")
    result = fit_gse_varx(y, x, config) if varx else fit_gse_var(y, config)
    expected = np.square(y[3:] - y[2:-1]).mean()
    assert result.history["mse"][0] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("varx", [False, True])
def test_supplied_model_is_not_reinitialized(varx):
    rng = np.random.default_rng(146)
    model = make_model(varx)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.3)
        model.intercept.fill_(7.0)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    config_type = GSEVARXTrainingConfig if varx else GSEVARTrainingConfig
    config = config_type(order=3, hidden_layer_size=5, max_epochs=0,
                         verbose=0, device="cpu")
    y, x = rng.normal(size=(9, 3)), rng.normal(size=(9, 2))
    if varx:
        from dataclasses import replace
        config = replace(config, exog_order=2)
        result = fit_gse_varx(y, x, config, model=model)
    else:
        result = fit_gse_var(y, config, model=model)
    for key, value in result.model.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)


def test_ungated_gvar_keeps_random_output_initialization():
    torch.manual_seed(147)
    model = GSEVAR(num_vars=3, order=2, hidden_layer_size=5,
                   use_causal_gate=False)
    assert isinstance(model.coeff_net, LagwiseMLP)
    assert torch.count_nonzero(model.coeff_net.weights[-1]) > 0
    for bias in model.coeff_net.biases:
        torch.testing.assert_close(bias, torch.full_like(bias, 0.1))
    assert model.intercept is None
