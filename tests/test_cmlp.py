from copy import deepcopy

import numpy as np
import pytest
import torch

pytest.importorskip("torch")

from gse_var import CMLP, CMLPTrainingConfig, construct_lagged_dataset, fit_cmlp


def test_vectorized_cmlp_forward_matches_target_networks():
    torch.manual_seed(7)
    model = CMLP(num_vars=4, order=3, hidden_layer_size=5, num_hidden_layers=2)
    inputs = torch.randn(6, 3, 4)

    expected = torch.cat([network(inputs) for network in model.networks], dim=1)
    actual = model(inputs)

    torch.testing.assert_close(actual, expected)


def test_ista_uses_sum_of_target_losses_like_neural_gc():
    data = np.random.default_rng(4).normal(size=(9, 3)).astype("float32")
    config = CMLPTrainingConfig(
        order=2,
        hidden_layer_size=4,
        max_epochs=1,
        batch_size=100,
        learning_rate=1e-2,
        regularizer="none",
        optimizer="ista",
        shuffle=False,
        verbose=0,
    )
    torch.manual_seed(9)
    actual = CMLP(num_vars=3, order=2, hidden_layer_size=4)
    expected = deepcopy(actual)
    dataset = construct_lagged_dataset(data, order=2)
    inputs = torch.as_tensor(dataset.predictors)
    targets = torch.as_tensor(dataset.responses)

    loss = torch.nn.functional.mse_loss(expected(inputs), targets) * expected.num_vars
    loss.backward()
    with torch.no_grad():
        for parameter in expected.parameters():
            parameter.add_(parameter.grad, alpha=-config.learning_rate)

    fit_cmlp(data, config, model=actual)

    for actual_parameter, expected_parameter in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(actual_parameter, expected_parameter)


def test_fit_cmlp_sparse_group_lasso_smoke():
    data = np.random.default_rng(0).normal(size=(20, 3)).astype("float32")
    config = CMLPTrainingConfig(
        order=2,
        hidden_layer_size=4,
        max_epochs=2,
        batch_size=8,
        learning_rate=1e-2,
        regularizer="sparse_group_lasso",
        sparse_group_lambda=1e-3,
        sparse_l1_lambda=1e-4,
        optimizer="ista",
        verbose=0,
    )

    result = fit_cmlp(data, config)

    assert len(result.history["loss"]) == 2
    assert result.causal_strength.shape == (3, 3)
    assert result.causal_graph.shape == (3, 3)


def test_cmlp_sparse_group_lasso_rejects_lambda_ngc():
    data = np.random.default_rng(1).normal(size=(12, 2)).astype("float32")
    config = CMLPTrainingConfig(
        order=2,
        hidden_layer_size=4,
        max_epochs=1,
        lambda_ngc=1e-3,
        regularizer="sparse_group_lasso",
        verbose=0,
    )

    with pytest.raises(ValueError, match="sparse_group_lasso does not use lambda_ngc"):
        fit_cmlp(data, config)
