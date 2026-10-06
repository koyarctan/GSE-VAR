import numpy as np
import pytest

pytest.importorskip("torch")

from gse_var import GSEVARTrainingConfig, fit_gse_var


def test_sparse_group_lasso_config_rejects_lambda_ngc():
    data = np.random.default_rng(0).normal(size=(12, 2)).astype("float32")
    config = GSEVARTrainingConfig(
        order=2,
        hidden_layer_size=4,
        max_epochs=1,
        lambda_ngc=1e-3,
        regularizer="sparse_group_lasso",
        verbose=0,
    )

    with pytest.raises(ValueError, match="sparse_group_lasso does not use lambda_ngc"):
        fit_gse_var(data, config)


def test_hierarchical_group_lasso_config_uses_lambda_ngc():
    data = np.random.default_rng(0).normal(size=(12, 2)).astype("float32")
    config = GSEVARTrainingConfig(
        order=2,
        hidden_layer_size=4,
        max_epochs=1,
        lambda_ngc=1e-3,
        regularizer="hierarchical_group_lasso",
        verbose=0,
    )

    result = fit_gse_var(data, config)

    assert len(result.history["loss"]) == 1


@pytest.mark.parametrize("optimizer", ["ista", "adam"])
def test_optional_jacobian_regularization_is_logged(optimizer):
    data = np.random.default_rng(2).normal(size=(16, 2)).astype("float32")
    config = GSEVARTrainingConfig(
        order=2,
        hidden_layer_size=4,
        max_epochs=1,
        batch_size=7,
        lambda_jacobian=0.1,
        optimizer=optimizer,
        verbose=0,
    )

    result = fit_gse_var(data, config)

    assert len(result.history["jacobian"]) == 1
    assert result.history["jacobian"][0] >= 0.0
    assert np.isfinite(result.history["jacobian"][0])


def test_negative_jacobian_regularization_is_rejected():
    data = np.random.default_rng(3).normal(size=(12, 2)).astype("float32")
    config = GSEVARTrainingConfig(
        order=2,
        hidden_layer_size=4,
        max_epochs=1,
        lambda_jacobian=-0.1,
        verbose=0,
    )

    with pytest.raises(ValueError, match="lambda_jacobian"):
        fit_gse_var(data, config)
