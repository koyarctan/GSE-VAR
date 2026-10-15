import pytest

torch = pytest.importorskip("torch")

from gse_var import GSEVAR, GSEVARX, GSEVARTrainingConfig, GSEVARXTrainingConfig
from gse_var.training import _make_optimizer as make_var_optimizer
from gse_var.varx import _make_optimizer as make_varx_optimizer


def make_case(varx, optimizer, fit_intercept):
    config_type = GSEVARXTrainingConfig if varx else GSEVARTrainingConfig
    config = config_type(order=2, hidden_layer_size=4, optimizer=optimizer,
                         learning_rate=0.003, coefficient_weight_decay=0.125,
                         fit_intercept=fit_intercept)
    if varx:
        model = GSEVARX(num_endogenous=2, num_exogenous=3, order=2,
                        exog_order=0, include_current_exog=True,
                        hidden_layer_size=4, fit_intercept=fit_intercept)
        return config, model, make_varx_optimizer, ("endogenous_gate", "exogenous_gate")
    model = GSEVAR(num_vars=2, order=2, hidden_layer_size=4,
                   fit_intercept=fit_intercept)
    return config, model, make_var_optimizer, ("causal_gate",)


@pytest.mark.parametrize("varx", [False, True])
@pytest.mark.parametrize("optimizer", ["ista", "adam"])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("freeze_parameters", [False, True])
def test_optimizer_preserves_parameter_order_groups_and_decay_policy(
    varx, optimizer, fit_intercept, freeze_parameters,
):
    config, model, factory, gate_names = make_case(varx, optimizer, fit_intercept)
    if freeze_parameters:
        for name, parameter in model.named_parameters():
            if name in (gate_names[0], "intercept") or name.endswith("weights.0"):
                parameter.requires_grad_(False)

    named_parameters = dict(model.named_parameters())
    trainable = [name for name, parameter in named_parameters.items() if parameter.requires_grad]
    expected = [
        [name for name in trainable if name not in gate_names and name != "intercept"],
        [name for name in trainable if name in gate_names],
        [name for name in trainable if name == "intercept"],
    ]
    by_id = {id(parameter): name for name, parameter in named_parameters.items()}
    actual_optimizer = factory(config, model)
    actual = [[by_id[id(parameter)] for parameter in group["params"]]
              for group in actual_optimizer.param_groups]

    assert actual == expected  # Include empty groups and preserve declaration order.
    assert [group["weight_decay"] for group in actual_optimizer.param_groups] == [0.125, 0.0, 0.0]
    assert all(group["lr"] == config.learning_rate for group in actual_optimizer.param_groups)
    flattened = [name for group in actual for name in group]
    assert len(flattened) == len(set(flattened)) == len(trainable)
    assert set(flattened) == set(trainable)
    expected_type = torch.optim.Adam if optimizer == "adam" else torch.optim.SGD
    assert isinstance(actual_optimizer, expected_type)


@pytest.mark.parametrize("varx", [False, True])
def test_unsupported_optimizer_preserves_error_contract(varx):
    config, model, factory, _ = make_case(varx, "unknown", True)
    with pytest.raises(ValueError, match="^unsupported optimizer: unknown$"):
        factory(config, model)
