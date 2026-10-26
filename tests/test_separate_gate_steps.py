"""Public optimizer routes must reproduce the independently tested prototypes."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gse_var import (
    GSEVARTrainingConfig, GSEVARDisARMTrainingConfig, fit_gse_var, fit_gse_var_disarm,
)
from gse_var.training import _make_optimizer, _alternating_proximal_step
from tests.test_proximal_optimization import quadratic_case


@pytest.mark.parametrize("varx", [False, True])
def test_adam_prox_logs_the_same_objective_components_with_smoothness(varx):
    from tests.test_proximal_optimization import (
        test_full_batch_logged_components_describe_the_same_pre_update_model,
    )
    test_full_batch_logged_components_describe_the_same_pre_update_model(varx)


@pytest.mark.parametrize("varx", [False, True])
@pytest.mark.parametrize("regularizer", ["sparse_group_lasso", "group_lasso", "hierarchical_group_lasso"])
def test_adam_prox_keeps_all_three_exact_proximal_operators(varx, regularizer):
    from gse_var import GSEVARXTrainingConfig, fit_gse_varx
    cls = GSEVARXTrainingConfig if varx else GSEVARTrainingConfig
    penalty = (dict(sparse_l1_lambda=.4, sparse_group_lambda=.4)
               if regularizer == "sparse_group_lasso" else dict(lambda_ngc=.8))
    config = cls(order=2, hidden_layer_size=3, max_epochs=4, batch_size=100,
                 learning_rate=.003, gate_learning_rate=1.,
                 regularizer=regularizer, verbose=0, shuffle=False, device="cpu", **penalty)
    data = np.zeros((12, 2), dtype="float32")
    result = fit_gse_varx(data, data, config) if varx else fit_gse_var(data, config)
    gates = ((result.model.endogenous_gate, result.model.exogenous_gate) if varx
             else (result.model.causal_gate,))
    assert all(gate.count_nonzero() == 0 for gate in gates)
    assert np.isfinite(result.history["loss"]).all()


def test_adam_coefficients_and_euclidean_prox_have_separate_steps_and_no_double_decay():
    theta, gate, old, regularizer, evaluate = quadratic_case(theta=1., ridge=.4)
    groups = old.param_groups
    groups[1]["lr"] = .2
    optimizer = torch.optim.Adam(groups, lr=.001)
    reference = torch.nn.Parameter(theta.detach().clone())
    adam = torch.optim.Adam([reference], lr=.001)
    calls = []
    def counted(create_graph):
        calls.append(1)
        return evaluate(create_graph)
    for _ in range(3):
        adam.zero_grad(set_to_none=True)
        ((reference-2).square().sum()+.2*reference.square().sum()).backward()
        adam.step()
        metrics = _alternating_proximal_step(optimizer, counted, ((gate, regularizer),))
    torch.testing.assert_close(theta, reference, rtol=0, atol=0)
    expected = torch.ones_like(gate)
    for _ in range(3):
        regularizer.prox_(expected, .2, nonnegative=True)
    torch.testing.assert_close(gate, expected, rtol=0, atol=0)
    assert len(calls) == 6
    assert gate not in optimizer.state  # Adam never touches or stores gate moments.
    assert metrics["gate_step_size"] == pytest.approx(.2)


@pytest.mark.parametrize("method", ["prox_blocks", "disarm"])
def test_public_fit_reproduces_research_updates(method):
    pytest.importorskip("experiments.gate_research.bench")
    from experiments.gate_research.bench import FitSpec, fit_candidate, generate_data
    data = generate_data(101)
    train = tuple(value[:24] for value in data["train"])
    validation = tuple(value[:12] for value in data["validation"])
    binary = method == "disarm"
    spec = FitSpec(method, .03, epochs=5, check_every=5, hidden=3,
                   gate_lr=.03 if binary else 1.)
    expected, _ = fit_candidate(train, validation, spec, 42)
    sequence = np.concatenate((train[0][0].numpy(), train[1].numpy()))
    common = dict(order=1, hidden_layer_size=3, max_epochs=5, batch_size=1000,
                  learning_rate=.003, gate_learning_rate=spec.gate_lr,
                  coefficient_weight_decay=.0001, seed=42, shuffle=False,
                  verbose=0, device="cpu")
    config = (GSEVARDisARMTrainingConfig(**common, lambda_l0=.03/16, gate_probability_init=.5)
              if binary else GSEVARTrainingConfig(**common,
                                                  regularizer="group_lasso", lambda_ngc=.03/16))
    epochs = []
    callback = lambda epoch, model, metrics: epochs.append(epoch)
    actual = (fit_gse_var_disarm(sequence, config, epoch_callback=callback) if binary else
              fit_gse_var(sequence, config, epoch_callback=callback))
    assert epochs == [1, 2, 3, 4, 5]
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(actual.model.state_dict()[name], value, rtol=1e-6, atol=1e-7)


def test_default_always_uses_adam_with_an_exact_gate_prox():
    config = GSEVARTrainingConfig(order=1, hidden_layer_size=2)
    from gse_var import GSEVAR
    model = GSEVAR(2, 1, 2)
    optimizer = _make_optimizer(config, model)
    assert isinstance(optimizer, torch.optim.Adam)
    assert config.gate_learning_rate is None
    assert all(group["lr"] == config.learning_rate for group in optimizer.param_groups)
    assert all("objective_weight_decay" in group for group in optimizer.param_groups)
    assert GSEVARDisARMTrainingConfig(order=1, hidden_layer_size=2).gate_probability_init == .9


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
def test_invalid_separate_gate_rate_is_rejected(binary, value):
    cls = GSEVARDisARMTrainingConfig if binary else GSEVARTrainingConfig
    config = cls(order=1, hidden_layer_size=2, gate_learning_rate=value, verbose=0)
    fit = fit_gse_var_disarm if binary else fit_gse_var
    with pytest.raises(ValueError, match="gate_learning_rate"):
        fit(np.zeros((6, 2)), config)
