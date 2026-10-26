from dataclasses import replace
from itertools import product
import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gse_var import GSEVAR, GSEVARDisARM, GSEVARDisARMTrainingConfig, fit_gse_var_disarm
from gse_var.disarm import (
    _antithetic_masks, _disarm_gradient, _disarm_step, _masked_losses, _smoothness_losses,
)
from gse_var.training import temporal_smoothness_penalty


def test_disarm_expectation_matches_exact_enumeration_for_nonlinear_loss():
    # Integrate exactly over all antithetic-mask intervals, not a noisy MC test.
    logits = torch.tensor([-1.1, 0.0, 0.7], dtype=torch.float64, requires_grad=True)
    probabilities = logits.sigmoid()
    def loss(mask):
        a, b, c = mask.to(logits.dtype)
        return (a + 2 * b - c).square() + 3 * a * b * c
    exact_loss = sum(loss(torch.tensor(mask)) * torch.where(torch.tensor(mask), probabilities,
                                                           1 - probabilities).prod()
                     for mask in product([False, True], repeat=3))
    exact = torch.autograd.grad(exact_loss, logits)[0]
    intervals = []
    for probability in probabilities.detach().tolist():
        boundaries = sorted(set([0., probability, 1 - probability, 1.]))
        intervals.append([((a + b) / 2, b - a) for a, b in zip(boundaries, boundaries[1:])])
    estimated = torch.zeros_like(logits)
    for cells in product(*intervals):
        uniform = torch.tensor([cell[0] for cell in cells], dtype=logits.dtype)
        first, second = uniform < probabilities, (1 - uniform) < probabilities
        weight = np.prod([cell[1] for cell in cells])
        estimated += weight * _disarm_gradient(logits, first, second, loss(first), loss(second))
    torch.testing.assert_close(estimated, exact, rtol=1e-12, atol=1e-12)


def test_equal_mask_entries_have_zero_disarm_gradient():
    logits = torch.tensor([-3., 0., 3.])
    first, second = torch.tensor([False, False, True]), torch.tensor([False, True, True])
    gradient = _disarm_gradient(logits, first, second, torch.tensor(3.), torch.tensor(1.))
    torch.testing.assert_close(gradient, torch.tensor([0., -0.5, 0.]))


def _exact_estimator_moments(logits, loss_function):
    """Integrate the piecewise-constant antithetic estimator exactly."""
    probabilities = logits.detach().sigmoid()
    intervals = []
    for probability in probabilities.flatten().tolist():
        boundaries = sorted(set([0., probability, 1-probability, 1.]))
        intervals.append([((a+b)/2, b-a) for a, b in zip(boundaries, boundaries[1:])])
    mean, second_moment = torch.zeros_like(logits), torch.zeros_like(logits)
    for cells in product(*intervals):
        uniform = torch.tensor([cell[0] for cell in cells], dtype=logits.dtype).reshape_as(logits)
        first, second = uniform < probabilities, (1-uniform) < probabilities
        gradient = _disarm_gradient(logits, first, second, loss_function(first), loss_function(second))
        weight = np.prod([cell[1] for cell in cells])
        mean += weight * gradient
        second_moment += weight * gradient.square()
    return mean, second_moment-mean.square()


@pytest.mark.parametrize("coupled", [False, True])
def test_target_local_estimator_is_unbiased_and_additive_variance_does_not_increase(coupled):
    logits = torch.tensor([[[-1.1, .3], [.7, -.4]]], dtype=torch.float64, requires_grad=True)
    probabilities = logits.sigmoid()
    def losses(mask):
        a, b, c, d = mask.to(logits.dtype).flatten()
        local = torch.stack(((a+2*b-1).square()+a*b, 3*(c-d).square()+.2*c))/2
        shared = ((a+b-c).square()/(1+a+2*d) if coupled else logits.new_zeros(()))
        return local, shared
    def objective(mask):
        local, shared = losses(mask)
        return local.sum()+shared
    def target_scores(mask):
        local, shared = losses(mask)
        return (local+shared)[None, :, None]
    exact_loss = sum(
        objective(torch.tensor(mask).reshape_as(logits))
        * torch.where(torch.tensor(mask).reshape_as(logits), probabilities, 1-probabilities).prod()
        for mask in product([False, True], repeat=logits.numel())
    )
    exact_gradient = torch.autograd.grad(exact_loss, logits)[0]
    local_mean, local_variance = _exact_estimator_moments(logits, target_scores)
    global_mean, global_variance = _exact_estimator_moments(logits, objective)
    torch.testing.assert_close(local_mean, exact_gradient, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(global_mean, exact_gradient, rtol=1e-12, atol=1e-12)
    if not coupled:
        assert torch.all(local_variance <= global_variance+1e-12)
        assert torch.any(local_variance < global_variance-1e-5)


@pytest.mark.parametrize("mode", ["absolute", "relative"])
@pytest.mark.parametrize("times", [[7], [2, 4, 6], [9, 6, 7, 2, 3]])
def test_smoothness_scalar_and_parameter_gradient_match_existing_objective(mode, times):
    coefficients = torch.randn(len(times), 2, 3, 3, dtype=torch.float64, requires_grad=True)
    times = torch.tensor(times)
    scalar, target_scores = _smoothness_losses(coefficients, times, mode, 1e-8)
    reference = temporal_smoothness_penalty(coefficients, times, mode, 1e-8)
    torch.testing.assert_close(scalar, reference, rtol=0, atol=0)
    assert target_scores.shape == (3,)
    assert not target_scores.requires_grad
    if mode == "absolute":
        expected = torch.stack([temporal_smoothness_penalty(
            coefficients[:, :, target:target+1, :], times, mode, 1e-8,
        )/3 for target in range(3)])
    else:
        expected = reference.expand(3)
    torch.testing.assert_close(target_scores, expected)
    if scalar.requires_grad:
        actual_gradient = torch.autograd.grad(scalar, coefficients)[0]
        reference_gradient = torch.autograd.grad(reference, coefficients)[0]
        torch.testing.assert_close(actual_gradient, reference_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["absolute", "relative"])
def test_target_score_losses_keep_global_mean_scaling_and_coupled_relative_penalty(mode):
    torch.manual_seed(18)
    model = GSEVARDisARM(3, 2, 4)
    with torch.no_grad():
        for parameter in model.coeff_net.parameters():
            parameter.uniform_(-.3, .3)
    config = GSEVARDisARMTrainingConfig(order=2, hidden_layer_size=4,
        lambda_smooth=.13, lambda_jacobian=.07, smoothness_mode=mode, verbose=0)
    inputs, targets, times = torch.randn(5, 2, 3), torch.randn(5, 3), torch.tensor([9, 6, 7, 2, 3])
    mask = torch.ones_like(model.causal_gate)
    mask[0, 1, 2] = 0
    scalars, target_scores = _masked_losses(model, inputs, targets, times, mask, config)
    prediction, coefficients, mismatch = model._masked_forward(inputs, mask, need_mismatch=True)
    torch.testing.assert_close(scalars["mse"], (prediction-targets).square().mean())
    torch.testing.assert_close(scalars["smooth"], config.lambda_smooth * temporal_smoothness_penalty(
        coefficients, times, mode, config.smoothness_eps,
    ))
    torch.testing.assert_close(scalars["jacobian"], config.lambda_jacobian * mismatch.square().mean())
    expected = []
    for target in range(3):
        mse = (prediction[:, target]-targets[:, target]).square().mean()/3
        smooth = config.lambda_smooth * temporal_smoothness_penalty(
            coefficients[:, :, target:target+1, :] if mode == "absolute" else coefficients,
            times, mode, config.smoothness_eps,
        ) / (3 if mode == "absolute" else 1)
        jacobian = config.lambda_jacobian * mismatch[:, :, target, :].square().mean()/3
        expected.append(mse+smooth+jacobian)
    torch.testing.assert_close(target_scores, torch.stack(expected))
    if mode == "absolute":
        torch.testing.assert_close(target_scores.sum(), sum(scalars.values()))
    else:
        torch.testing.assert_close(target_scores.sum(),
            scalars["mse"]+scalars["jacobian"]+3*scalars["smooth"])


def test_unrelated_targets_do_not_change_target_gate_update_for_absolute_smoothness():
    torch.manual_seed(19)
    model = GSEVARDisARM(2, 2, 3, gate_probability_init=.6)
    with torch.no_grad():
        for parameter in model.coeff_net.parameters():
            parameter.uniform_(-.4, .4)
    reference = pickle.loads(pickle.dumps(model))
    config = GSEVARDisARMTrainingConfig(order=2, hidden_layer_size=3,
        lambda_l0=.01, lambda_smooth=.03, lambda_jacobian=.02,
        learning_rate=.05, verbose=0)
    inputs, targets, times = torch.randn(5, 2, 2), torch.randn(5, 2), torch.arange(5)
    changed_targets = targets.clone()
    changed_targets[:, 1] += 1000
    for candidate, responses in ((model, targets), (reference, changed_targets)):
        _disarm_step(candidate, inputs, responses, times, config,
            torch.optim.SGD(candidate.parameters(), lr=config.learning_rate),
            torch.Generator().manual_seed(23))
    torch.testing.assert_close(model.gate_logits[:, 0, :], reference.gate_logits[:, 0, :], rtol=0, atol=0)
    assert not torch.equal(model.gate_logits[:, 1, :], reference.gate_logits[:, 1, :])


def test_binary_masks_preserve_the_existing_prediction_formula_and_initialization():
    torch.manual_seed(7)
    reference = GSEVAR(3, 2, 4)
    torch.manual_seed(7)
    model = GSEVARDisARM(3, 2, 4)
    inputs = torch.randn(5, 2, 3)
    for actual, expected in zip(model(inputs), reference(inputs)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert "causal_gate" not in dict(model.named_parameters())
    assert "causal_gate" in dict(model.named_buffers())
    assert torch.allclose(model.gate_probabilities, torch.full_like(model.causal_gate, .9))
    mask = torch.ones_like(model.causal_gate)
    mask[:, 1, 0] = 0
    model.causal_gate.copy_(mask)
    reference.causal_gate.data.copy_(mask)
    with torch.no_grad():
        for parameter in model.coeff_net.parameters():
            parameter.uniform_(-.3, .3)
        reference.coeff_net.load_state_dict(model.coeff_net.state_dict())
    for actual, expected in zip(model.forward_with_mask(inputs, mask), reference(inputs)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    changed = inputs.clone()
    changed[:, :, 0] += 10
    torch.testing.assert_close(model(changed)[0][:, 1], model(inputs)[0][:, 1], rtol=0, atol=0)
    assert torch.count_nonzero(model(inputs)[1][:, :, 1, 0]) == 0
    _, jacobian, _ = model._jacobian_components(inputs, create_graph=False, detach_causal_gate=True)
    assert torch.count_nonzero(jacobian[:, :, 1, 0]) == 0


def test_masked_forward_never_mutates_the_deterministic_inference_mask():
    model = GSEVARDisARM(2, 1, 3)
    original = model.causal_gate.clone()
    model.forward_with_mask(torch.randn(4, 1, 2), torch.zeros_like(original))
    torch.testing.assert_close(model.causal_gate, original)
    with pytest.raises(ValueError, match="only 0 and 1"):
        model.forward_with_mask(torch.randn(4, 1, 2), original * .5)


def test_unused_gates_follow_analytic_expected_l0_gradient_and_only_two_forwards():
    model = GSEVARDisARM(2, 2, 3, gate_probability_init=.7)
    with torch.no_grad():
        for parameter in model.coeff_net.parameters():
            parameter.zero_()
    config = GSEVARDisARMTrainingConfig(order=2, hidden_layer_size=3, learning_rate=.2,
                                       lambda_l0=.4, verbose=0)
    initial = model.gate_logits.detach().clone()
    generator = torch.Generator().manual_seed(4)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    calls = []
    handle = model.coeff_net.register_forward_hook(lambda *args: calls.append(1))
    metrics = _disarm_step(model, torch.zeros(6, 2, 2), torch.zeros(6, 2), torch.arange(6),
                           config, optimizer, generator)
    handle.remove()
    expected_gradient = config.lambda_l0 * .7 * .3
    torch.testing.assert_close(model.gate_logits, initial - config.learning_rate * expected_gradient)
    assert len(calls) == 2
    assert metrics["expected_lag_edges"] == pytest.approx(8 * .7)
    assert metrics["l0"] == pytest.approx(config.lambda_l0 * 8 * .7)
    assert metrics["loss"] == pytest.approx(float(metrics["l0"]))


@pytest.mark.parametrize("smoothness_mode", ["absolute", "relative"])
def test_coefficient_update_averages_both_masks_and_applies_l2_exactly_once(smoothness_mode):
    torch.manual_seed(17)
    model = GSEVARDisARM(2, 1, 3, gate_probability_init=.6)
    with torch.no_grad():
        for parameter in model.coeff_net.parameters():
            parameter.uniform_(-.3, .3)
    reference = pickle.loads(pickle.dumps(model))
    config = GSEVARDisARMTrainingConfig(order=1, hidden_layer_size=3, learning_rate=.02,
        lambda_smooth=.03, lambda_jacobian=.02, coefficient_weight_decay=.4,
        smoothness_mode=smoothness_mode, verbose=0)
    inputs, targets, times = torch.randn(5, 1, 2), torch.randn(5, 2), torch.arange(5)
    masks = _antithetic_masks(reference.gate_logits, torch.Generator().manual_seed(12))
    losses = [sum(_masked_losses(reference, inputs, targets, times, mask, config)[0].values())
              for mask in masks]
    decay = .5 * config.coefficient_weight_decay * sum(p.square().sum() for p in reference.coeff_net.parameters())
    objective = .5 * sum(losses) + decay
    parameters = list(reference.coeff_net.parameters()) + [reference.intercept]
    gradients = torch.autograd.grad(objective, parameters)
    expected = [parameter.detach() - config.learning_rate * gradient
                for parameter, gradient in zip(parameters, gradients)]
    record = _disarm_step(model, inputs, targets, times, config,
        torch.optim.SGD(model.parameters(), lr=config.learning_rate), torch.Generator().manual_seed(12))
    for actual, wanted in zip(list(model.coeff_net.parameters()) + [model.intercept], expected):
        torch.testing.assert_close(actual, wanted)
    assert record["loss"] == pytest.approx(float(objective.detach()) + config.lambda_l0 * .6 * 4)


@pytest.mark.parametrize("smoothness_mode", ["absolute", "relative"])
def test_fit_supports_regularizers_and_callbacks_use_the_fixed_mask(smoothness_mode):
    data = np.random.default_rng(123).normal(size=(13, 2)).astype("float32")
    config = GSEVARDisARMTrainingConfig(order=2, hidden_layer_size=3, max_epochs=3,
        batch_size=5, lambda_smooth=.02, lambda_jacobian=.01, coefficient_weight_decay=.03,
        smoothness_mode=smoothness_mode, verbose=0, device="cpu")
    callbacks = []
    def callback(epoch, model, metrics):
        assert torch.equal(model.causal_gate.bool(), model.gate_logits >= 0)
        assert metrics["loss"] == pytest.approx(sum(metrics[key] for key in
            ("mse", "smooth", "jacobian", "weight_decay", "l0")), rel=1e-5)
        callbacks.append(epoch)
    result = fit_gse_var_disarm(data, config, epoch_callback=callback)
    assert callbacks == [1, 2, 3]
    assert all(len(values) == 3 and np.isfinite(values).all() for values in result.history.values())
    assert result.coeffs.shape == (11, 2, 2, 2)
    np.testing.assert_array_equal(result.causal_graph, result.model.causal_gate.bool().any(0).numpy())
    restored = GSEVARDisARM(2, 2, 3)
    restored.load_state_dict(result.model.state_dict())
    inputs = torch.randn(4, 2, 2)
    torch.testing.assert_close(restored(inputs)[0], result.model(inputs)[0], rtol=0, atol=0)
    restored = pickle.loads(pickle.dumps(result.model))
    torch.testing.assert_close(restored(inputs)[0], result.model(inputs)[0], rtol=0, atol=0)


def test_seeded_fit_is_reproducible_with_minibatches():
    data = np.random.default_rng(4).normal(size=(12, 2))
    config = GSEVARDisARMTrainingConfig(order=2, hidden_layer_size=3, max_epochs=4,
                                       batch_size=4, shuffle=True, verbose=0, device="cpu")
    first, second = fit_gse_var_disarm(data, config), fit_gse_var_disarm(data, config)
    assert first.history == second.history
    for key in first.model.state_dict():
        torch.testing.assert_close(first.model.state_dict()[key], second.model.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("field,value", [
    ("lambda_l0", -1), ("lambda_l0", float("nan")), ("learning_rate", 0),
    ("gate_probability_init", 0), ("gate_probability_init", 1),
    ("max_epochs", 0), ("batch_size", 0),
])
def test_invalid_disarm_config_is_rejected(field, value):
    config = replace(GSEVARDisARMTrainingConfig(order=1, hidden_layer_size=2), **{field: value})
    with pytest.raises(ValueError, match=field):
        fit_gse_var_disarm(np.zeros((5, 2)), config)


def test_nonfinite_gradient_is_rejected_without_parameter_update():
    model = GSEVARDisARM(2, 1, 3)
    config = GSEVARDisARMTrainingConfig(order=1, hidden_layer_size=3, verbose=0)
    originals = {name: parameter.clone() for name, parameter in model.named_parameters()}
    with pytest.raises(FloatingPointError, match="non-finite DisARM"):
        _disarm_step(model, torch.full((4, 1, 2), float("nan")), torch.zeros(4, 2),
                      torch.arange(4), config, torch.optim.Adam(model.parameters()),
                      torch.Generator().manual_seed(1))
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, originals[name], rtol=0, atol=0)
