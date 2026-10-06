import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gse_var import (
    GSEVARX,
    GSEVARXTrainingConfig,
    construct_varx_lagged_dataset,
    evaluate_varx_jacobian_agreement,
    fit_gse_varx,
)


def _make_model():
    return GSEVARX(
        num_endogenous=2,
        num_exogenous=3,
        order=2,
        exog_order=1,
        include_current_exog=True,
        hidden_layer_size=5,
        num_hidden_layers=2,
    ).double()


def test_varx_forward_and_fast_jacobians_match_explicit_autograd():
    torch.manual_seed(11)
    model = _make_model()
    with torch.no_grad():
        model.endogenous_gate[0, 1, 0] = 0.0
        model.exogenous_gate[1, 0, 2] = 0.0
    endogenous = torch.randn(3, 2, 2, dtype=torch.float64)
    exogenous = torch.randn(3, 2, 3, dtype=torch.float64)

    outputs = model.forward_with_jacobians(endogenous, exogenous)
    predictions, endogenous_coeffs, exogenous_coeffs = outputs[:3]
    endogenous_jacobian, exogenous_jacobian = outputs[3:5]
    endogenous_mismatch, exogenous_mismatch = outputs[5:]

    explicit_endogenous = endogenous.clone().requires_grad_(True)
    explicit_exogenous = exogenous.clone().requires_grad_(True)
    explicit_predictions, _, _ = model(
        explicit_endogenous,
        explicit_exogenous,
    )
    endogenous_gradients = []
    exogenous_gradients = []
    for target in range(model.num_endogenous):
        endogenous_gradient, exogenous_gradient = torch.autograd.grad(
            explicit_predictions[:, target].sum(),
            (explicit_endogenous, explicit_exogenous),
            retain_graph=True,
        )
        endogenous_gradients.append(endogenous_gradient)
        exogenous_gradients.append(exogenous_gradient)
    explicit_endogenous_jacobian = torch.stack(
        endogenous_gradients,
        dim=2,
    )
    explicit_exogenous_jacobian = torch.stack(
        exogenous_gradients,
        dim=2,
    )

    assert predictions.shape == (3, 2)
    assert endogenous_coeffs.shape == (3, 2, 2, 2)
    assert exogenous_coeffs.shape == (3, 2, 2, 3)
    assert torch.allclose(
        endogenous_jacobian,
        explicit_endogenous_jacobian,
        atol=1e-10,
        rtol=1e-8,
    )
    assert torch.allclose(
        exogenous_jacobian,
        explicit_exogenous_jacobian,
        atol=1e-10,
        rtol=1e-8,
    )
    assert torch.allclose(
        endogenous_mismatch,
        endogenous_jacobian - endogenous_coeffs,
    )
    assert torch.allclose(
        exogenous_mismatch,
        exogenous_jacobian - exogenous_coeffs,
    )
    assert torch.count_nonzero(endogenous_jacobian[:, 0, 1, 0]) == 0
    assert torch.count_nonzero(exogenous_jacobian[:, 1, 0, 2]) == 0


def test_varx_jacobian_regularizer_does_not_update_either_gate():
    torch.manual_seed(12)
    model = _make_model()
    endogenous = torch.randn(3, 2, 2, dtype=torch.float64)
    exogenous = torch.randn(3, 2, 3, dtype=torch.float64)

    endogenous_mismatch, exogenous_mismatch = (
        model.coefficient_jacobian_mismatches_for_regularization(
            endogenous,
            exogenous,
        )
    )
    (
        endogenous_mismatch.pow(2).mean()
        + exogenous_mismatch.pow(2).mean()
    ).backward()

    assert model.endogenous_gate.grad is None
    assert model.exogenous_gate.grad is None
    assert model.endogenous_coeff_net.weights[0].grad is not None
    assert model.exogenous_coeff_net.weights[0].grad is not None

    model.zero_grad(set_to_none=True)
    predictions, _, _ = model(endogenous, exogenous)
    predictions.pow(2).mean().backward()

    assert model.endogenous_gate.grad is not None
    assert model.exogenous_gate.grad is not None
    assert torch.count_nonzero(model.endogenous_gate.grad) > 0
    assert torch.count_nonzero(model.exogenous_gate.grad) > 0


@pytest.mark.parametrize("optimizer", ["ista", "adam"])
def test_fit_gse_varx_smoke_and_diagnostics(optimizer):
    rng = np.random.default_rng(13)
    endog = rng.normal(size=(20, 2)).astype("float32")
    exog = rng.normal(size=(20, 3)).astype("float32")
    config = GSEVARXTrainingConfig(
        order=2,
        exog_order=1,
        include_current_exog=True,
        hidden_layer_size=4,
        max_epochs=2,
        batch_size=8,
        learning_rate=1e-2,
        regularizer="group_lasso",
        lambda_ngc=1e-3,
        lambda_ngc_exog=2e-3,
        lambda_jacobian=0.1,
        lambda_jacobian_exog=0.2,
        optimizer=optimizer,
        verbose=0,
    )

    result = fit_gse_varx(endog, exog, config)

    assert len(result.history["loss"]) == 2
    assert result.endogenous_coeffs.shape == (18, 2, 2, 2)
    assert result.exogenous_coeffs.shape == (18, 2, 2, 3)
    assert result.endogenous_strength.shape == (2, 2)
    assert result.exogenous_strength.shape == (2, 3)
    assert result.endogenous_graph.shape == (2, 2)
    assert result.exogenous_graph.shape == (2, 3)
    np.testing.assert_array_equal(result.exogenous_lags, [1, 0])

    dataset = construct_varx_lagged_dataset(
        endog,
        exog,
        order=2,
        exog_order=1,
        include_current_exog=True,
    )
    agreement = evaluate_varx_jacobian_agreement(
        result.model,
        dataset.endogenous_predictors,
        dataset.exogenous_predictors,
        batch_size=7,
    )
    assert agreement.endogenous.coefficients.shape == (18, 2, 2, 2)
    assert agreement.exogenous.coefficients.shape == (18, 2, 2, 3)
    assert agreement.endogenous.mse >= 0.0
    assert agreement.exogenous.mse >= 0.0
