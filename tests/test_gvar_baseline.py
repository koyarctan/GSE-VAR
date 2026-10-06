import numpy as np
import pytest

pytest.importorskip("torch")

from gse_var import GVARBaselineTrainingConfig, fit_gvar


def test_fit_gvar_baseline_smoke():
    data = np.random.default_rng(0).normal(size=(20, 3)).astype("float32")
    config = GVARBaselineTrainingConfig(
        order=2,
        hidden_layer_size=4,
        max_epochs=2,
        batch_size=8,
        learning_rate=1e-2,
        lambda_coeff=1e-3,
        lambda_smooth=1e-3,
        verbose=0,
    )

    result = fit_gvar(data, config)

    assert len(result.history["loss"]) == 2
    assert result.coeffs.shape == (18, 2, 3, 3)
    assert result.causal_strength.shape == (3, 3)
    assert result.causal_graph.shape == (3, 3)


def test_gvar_baseline_rejects_invalid_elastic_net_alpha():
    data = np.random.default_rng(1).normal(size=(12, 2)).astype("float32")
    config = GVARBaselineTrainingConfig(
        order=2,
        hidden_layer_size=4,
        max_epochs=1,
        elastic_net_alpha=1.5,
        verbose=0,
    )

    with pytest.raises(ValueError, match="elastic_net_alpha"):
        fit_gvar(data, config)
