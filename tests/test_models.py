import copy

import pytest

torch = pytest.importorskip("torch")

from gse_var.models import GSEVAR


def test_vectorized_gse_var_forward_shapes_and_gradients():
    model = GSEVAR(
        num_vars=3,
        order=2,
        hidden_layer_size=5,
        num_hidden_layers=2,
    )
    inputs = torch.randn(4, 2, 3)

    preds, coeffs = model(inputs)
    loss = preds.pow(2).mean() + coeffs.pow(2).mean()
    loss.backward()

    assert preds.shape == (4, 3)
    assert coeffs.shape == (4, 2, 3, 3)
    assert model.causal_gate.grad is not None


def test_fast_jacobian_matches_explicit_autograd():
    torch.manual_seed(7)
    model = GSEVAR(
        num_vars=3,
        order=2,
        hidden_layer_size=5,
        num_hidden_layers=2,
    ).double()
    with torch.no_grad():
        model.causal_gate[0, 1, 2] = 0.0
    inputs = torch.randn(4, 2, 3, dtype=torch.float64)

    _, coeffs, jacobian, mismatch = model.forward_with_jacobian(inputs)

    explicit_inputs = inputs.detach().clone().requires_grad_(True)
    explicit_preds, explicit_coeffs = model(explicit_inputs)
    explicit_jacobian = torch.stack(
        [
            torch.autograd.grad(
                explicit_preds[:, target].sum(),
                explicit_inputs,
                retain_graph=True,
            )[0]
            for target in range(model.num_vars)
        ],
        dim=2,
    )

    assert torch.allclose(coeffs, explicit_coeffs, atol=1e-10, rtol=1e-8)
    assert torch.allclose(jacobian, explicit_jacobian, atol=1e-10, rtol=1e-8)
    assert torch.allclose(mismatch, jacobian - coeffs, atol=1e-12, rtol=1e-10)
    assert torch.count_nonzero(jacobian[:, 0, 1, 2]) == 0


def test_fast_jacobian_penalty_gradient_matches_explicit_autograd():
    torch.manual_seed(8)
    fast_model = GSEVAR(
        num_vars=2,
        order=2,
        hidden_layer_size=4,
        num_hidden_layers=2,
    ).double()
    explicit_model = copy.deepcopy(fast_model)
    inputs = torch.randn(3, 2, 2, dtype=torch.float64)

    _, _, _, mismatch = fast_model.forward_with_jacobian(
        inputs,
        create_graph=True,
    )
    mismatch.pow(2).mean().backward()

    explicit_inputs = inputs.detach().clone().requires_grad_(True)
    preds, coeffs = explicit_model(explicit_inputs)
    jacobian = torch.stack(
        [
            torch.autograd.grad(
                preds[:, target].sum(),
                explicit_inputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            for target in range(explicit_model.num_vars)
        ],
        dim=2,
    )
    (jacobian - coeffs).pow(2).mean().backward()

    explicit_parameters = dict(explicit_model.named_parameters())
    for name, fast_parameter in fast_model.named_parameters():
        explicit_parameter = explicit_parameters[name]
        fast_gradient = (
            fast_parameter.grad
            if fast_parameter.grad is not None
            else torch.zeros_like(fast_parameter)
        )
        explicit_gradient = (
            explicit_parameter.grad
            if explicit_parameter.grad is not None
            else torch.zeros_like(explicit_parameter)
        )
        assert torch.allclose(
            fast_gradient,
            explicit_gradient,
            atol=1e-10,
            rtol=1e-8,
        )


def test_regularization_mismatch_stops_only_causal_gate_gradient():
    torch.manual_seed(10)
    model = GSEVAR(
        num_vars=2,
        order=2,
        hidden_layer_size=4,
        num_hidden_layers=2,
    ).double()
    inputs = torch.randn(3, 2, 2, dtype=torch.float64)

    with torch.no_grad():
        _, _, _, expected_mismatch = model.forward_with_jacobian(inputs)
    mismatch = model.coefficient_jacobian_mismatch_for_regularization(inputs)

    assert torch.allclose(mismatch, expected_mismatch, atol=1e-12, rtol=1e-10)

    mismatch.pow(2).mean().backward()

    assert model.causal_gate.grad is None
    coefficient_gradient = model.coeff_net.weights[0].grad
    assert coefficient_gradient is not None
    assert torch.isfinite(coefficient_gradient).all()
    assert torch.count_nonzero(coefficient_gradient) > 0
