import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gse_var import evaluate_jacobian_agreement
from gse_var.models import GSEVAR


def test_evaluate_jacobian_agreement_shapes_metrics_and_mode():
    torch.manual_seed(9)
    model = GSEVAR(
        num_vars=3,
        order=2,
        hidden_layer_size=5,
    )
    with torch.no_grad():
        model.causal_gate[1, 2, 0] = 0.0
    model.train()
    inputs = np.random.default_rng(9).normal(size=(7, 2, 3)).astype("float32")

    result = evaluate_jacobian_agreement(model, inputs, batch_size=3)

    assert model.training
    assert result.coefficients.shape == (7, 2, 3, 3)
    assert result.jacobian.shape == result.coefficients.shape
    assert result.mismatch.shape == result.coefficients.shape
    np.testing.assert_allclose(
        result.mismatch,
        result.jacobian - result.coefficients,
        atol=1e-6,
    )
    assert result.active_count == 7 * (2 * 3 * 3 - 1)
    assert result.mse >= 0.0
    assert result.mae >= 0.0
    assert result.active_mse >= 0.0
    assert result.active_mae >= 0.0
    assert 0.0 <= result.active_sign_agreement <= 1.0
    assert -1.0 <= result.cosine_similarity <= 1.0


def test_evaluate_jacobian_agreement_rejects_ungated_model():
    model = GSEVAR(
        num_vars=2,
        order=1,
        hidden_layer_size=3,
        use_causal_gate=False,
    )

    with pytest.raises(ValueError, match="use_causal_gate=True"):
        evaluate_jacobian_agreement(model, np.zeros((2, 1, 2), dtype="float32"))
