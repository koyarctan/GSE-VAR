"""Fixed training policy and numerical parity with the retained algorithms."""
from dataclasses import fields
import inspect
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gse_var import (
    GSEVAR, GSEVARX, GSEVARDisARM, GSEVARTrainingConfig, GSEVARXTrainingConfig,
    GSEVARDisARMTrainingConfig, fit_gse_var, fit_gse_varx, fit_gse_var_disarm,
)


def numerical_snapshot():
    """Exercise both regularizers, minibatching and Jacobian/smoothness terms."""
    rng = np.random.default_rng(751)
    y, x = rng.normal(size=(24, 3)), rng.normal(size=(24, 2))
    snapshots = {}
    for name, cls in (("proximal", GSEVARTrainingConfig), ("varx", GSEVARXTrainingConfig),
                      ("disarm", GSEVARDisARMTrainingConfig)):
        kwargs = dict(order=2, hidden_layer_size=3, max_epochs=4, batch_size=7,
                      learning_rate=.003, gate_learning_rate=.03 if name == "disarm" else .2,
                      coefficient_weight_decay=.03, lambda_smooth=.02, lambda_jacobian=.01,
                      shuffle=True, seed=17, verbose=0, device="cpu")
        if name == "disarm":
            kwargs.update(lambda_l0=.02, gate_probability_init=.5)
        else:
            kwargs.update(sparse_l1_lambda=.01, sparse_group_lambda=.02)
        if name == "varx":
            kwargs.update(exog_order=1)
        config = cls(**kwargs)
        result = (fit_gse_varx(y, x, config) if name == "varx" else
                  fit_gse_var_disarm(y, config) if name == "disarm" else fit_gse_var(y, config))
        arrays = (("endogenous_coeffs", "exogenous_coeffs", "endogenous_graph", "exogenous_graph")
                  if name == "varx" else ("coeffs", "causal_graph"))
        snapshots[name] = dict(state={key:value.detach().clone() for key,value in result.model.state_dict().items()},
                               history=result.history,
                               arrays={key:torch.as_tensor(getattr(result, key)) for key in arrays})
    return snapshots


def test_retained_algorithms_match_pre_refactor_weights_losses_and_inference():
    path = os.environ.get("GSE_REFACTOR_BASELINE")
    if not path:
        pytest.skip("set GSE_REFACTOR_BASELINE to a captured pre-refactor snapshot")
    expected = torch.load(path, weights_only=True)
    actual = numerical_snapshot()
    for method in expected:
        assert actual[method]["history"] == expected[method]["history"]
        for group in ("state", "arrays"):
            assert actual[method][group].keys() == expected[method][group].keys()
            for name in expected[method][group]:
                torch.testing.assert_close(actual[method][group][name], expected[method][group][name],
                                           rtol=0, atol=0)


@pytest.mark.parametrize("cls", [GSEVARTrainingConfig, GSEVARXTrainingConfig, GSEVARDisARMTrainingConfig])
def test_fixed_policies_are_not_configuration_options(cls):
    assert not {"optimizer", "fit_intercept", "causal_threshold"} & {field.name for field in fields(cls)}


@pytest.mark.parametrize("cls", [GSEVAR, GSEVARX, GSEVARDisARM])
def test_models_do_not_offer_an_intercept_switch(cls):
    assert "fit_intercept" not in inspect.signature(cls).parameters


def test_graph_selection_uses_exact_nonzero_routes_not_a_tunable_threshold():
    model = GSEVAR(2, 2, 3)
    with torch.no_grad():
        model.causal_gate.zero_()
        model.causal_gate[0, 0, 1] = 1e-30
    expected = torch.tensor([[0, 1], [0, 0]])
    torch.testing.assert_close(model.causal_graph_from_gate(), expected)
    assert "threshold" not in inspect.signature(model.causal_graph_from_gate).parameters


if __name__ == "__main__":
    import sys
    torch.set_num_threads(1)
    path = Path(sys.argv[1])
    if path.exists():
        raise FileExistsError(path)
    torch.save(numerical_snapshot(), path)
    print("Captured", str(path))
