"""Audit the research protocol's data, mask semantics, and selection isolation."""
import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("experiments.gate_research.bench")

from experiments.gate_research.bench import (
    FitSpec, PROTOCOL, conditional_mean, fit_candidate, generate_data, make_graph, structure_metrics,
)


def test_registered_graph_is_sparse_and_every_target_has_two_other_parents():
    parents, signs, truth = make_graph(101)
    p = PROTOCOL["data"]["variables"]
    assert truth.shape == (p, p)
    assert truth.sum() == 3*p
    for target, sources in enumerate(parents):
        assert len(set(sources)) == 2
        assert target not in sources
    assert set(np.unique(signs)) == {-1., 1.}
    assert not set(map(tuple, PROTOCOL["development"])) & set(map(tuple, PROTOCOL["confirmation"]))


def test_generator_has_genuine_multiplicative_interaction_and_only_declared_inputs():
    parents, signs, truth = make_graph(101)
    target = 0
    a, b = parents[target]
    inputs = np.zeros((4, len(parents)))
    inputs[:, a] = [0, 1, 0, 1]
    inputs[:, b] = [0, 0, 1, 1]
    means = conditional_mean(inputs, parents, signs)[:, target]
    assert abs(means[3]-means[2]-means[1]+means[0]) > .5
    changed = inputs.copy()
    changed[:, ~truth[target]] = 100
    np.testing.assert_array_equal(conditional_mean(changed, parents, signs)[:, target], means)


def test_generated_splits_are_reproducible_and_finite():
    first, second = generate_data(101), generate_data(101)
    for split in ("train", "validation", "test"):
        x, y = first[split]
        assert x.shape == (PROTOCOL["data"][split+"_responses"], 1, 16)
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        torch.testing.assert_close(x, second[split][0], rtol=0, atol=0)
        torch.testing.assert_close(y, second[split][1], rtol=0, atol=0)
    assert not torch.equal(first["train"][0][:256], first["validation"][0])
    assert first["oracle_test"].shape == first["test"][1].shape


@pytest.mark.parametrize("method", ["prox_blocks", "disarm", "dense"])
def test_fit_selects_using_only_validation_and_restores_that_state(method):
    assert list(inspect.signature(fit_candidate).parameters) == ["train", "validation", "spec", "seed"]
    data = generate_data(101)
    train = tuple(value[:16] for value in data["train"])
    validation = tuple(value[:8] for value in data["validation"])
    spec = FitSpec(method, .03, epochs=6, hidden=3, check_every=2, gate_lr=.03)
    model, record = fit_candidate(train, validation, spec, 42)
    with torch.no_grad():
        restored_mse = float((model(validation[0])[0]-validation[1]).square().mean())
    assert restored_mse == pytest.approx(record["validation_mse"])
    assert record["validation_mse"] == min(point["validation_mse"] for point in record["curve"])
    if method == "disarm":
        assert torch.all((model.causal_gate == 0) | (model.causal_gate == 1))
    else:
        assert torch.all(model.causal_gate >= 0)


def test_structure_metrics_exclude_diagonal_but_count_false_edges():
    truth = np.eye(3, dtype=bool)
    truth[0, 1] = True
    exact = structure_metrics(truth, truth)
    assert exact["f1"] == 1 and exact["tp"] == 1
    dense = structure_metrics(np.ones_like(truth), truth)
    assert dense["fp"] == 5
