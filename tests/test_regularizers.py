import pytest

torch = pytest.importorskip("torch")

from gse_var.regularizers import (
    NGCRegularizer,
    group_lasso_penalty,
    hierarchical_group_lasso_penalty,
    prox_group_lasso_,
    prox_sparse_group_lasso_,
    sparse_group_lasso_penalty,
)


def test_group_lasso_penalty_groups_over_lags():
    gate = torch.zeros(2, 2, 2)
    gate[:, 0, 1] = torch.tensor([3.0, 4.0])

    assert group_lasso_penalty(gate).item() == pytest.approx(5.0)


def test_group_lasso_prox_sets_whole_group_to_exact_zero():
    gate = torch.zeros(2, 1, 1)
    gate[:, 0, 0] = torch.tensor([3.0, 4.0])

    prox_group_lasso_(gate, lam=10.0, step_size=1.0)

    assert torch.count_nonzero(gate).item() == 0


def test_group_lasso_prox_shrinks_group_norm():
    gate = torch.zeros(2, 1, 1)
    gate[:, 0, 0] = torch.tensor([3.0, 4.0])

    prox_group_lasso_(gate, lam=1.0, step_size=1.0)

    assert gate[:, 0, 0].tolist() == pytest.approx([2.4, 3.2])


def test_sparse_group_lasso_penalty_adds_group_and_l1_terms():
    gate = torch.zeros(2, 1, 1)
    gate[:, 0, 0] = torch.tensor([3.0, 4.0])

    penalty = sparse_group_lasso_penalty(gate, group_lambda=1.0, l1_lambda=1.0)

    assert penalty.item() == pytest.approx(12.0)


def test_sparse_group_lasso_regularizer_uses_direct_lambdas():
    gate = torch.zeros(2, 1, 1)
    gate[:, 0, 0] = torch.tensor([3.0, 4.0])
    regularizer = NGCRegularizer(
        name="sparse_group_lasso",
        sparse_group_lambda=1.0,
        sparse_l1_lambda=1.0,
    )

    assert regularizer.penalty(gate).item() == pytest.approx(12.0)


def test_sparse_group_lasso_regularizer_rejects_outer_lambda():
    with pytest.raises(ValueError, match="does not use"):
        NGCRegularizer(name="sparse_group_lasso", lam=1.0)


def test_sparse_group_lasso_prox_applies_l1_then_group_shrinkage():
    gate = torch.zeros(2, 1, 1)
    gate[:, 0, 0] = torch.tensor([3.0, 4.0])

    prox_sparse_group_lasso_(gate, step_size=1.0, group_lambda=1.0, l1_lambda=1.0)

    expected_norm = (2.0**2 + 3.0**2) ** 0.5
    expected_scale = 1.0 - 1.0 / expected_norm
    assert gate[:, 0, 0].tolist() == pytest.approx([2.0 * expected_scale, 3.0 * expected_scale])


def test_sparse_group_lasso_prox_can_zero_individual_lags():
    gate = torch.zeros(2, 1, 1)
    gate[:, 0, 0] = torch.tensor([0.5, 4.0])

    prox_sparse_group_lasso_(gate, step_size=1.0, group_lambda=1.0, l1_lambda=1.0)

    assert gate[0, 0, 0].item() == 0.0
    assert gate[1, 0, 0].item() > 0.0


def test_hierarchical_penalty_matches_nested_prefixes():
    gate = torch.ones(3, 1, 1)

    penalty = hierarchical_group_lasso_penalty(gate)

    assert penalty.item() == pytest.approx(1.0 + 2.0**0.5 + 3.0**0.5)


def test_hierarchical_prox_uses_oldest_lag_prefixes():
    gate = torch.zeros(3, 1, 1)
    gate[:, 0, 0] = torch.tensor([1.0, 1.0, 100.0])

    from gse_var.regularizers import prox_hierarchical_group_lasso_

    prox_hierarchical_group_lasso_(gate, lam=2.0, step_size=1.0)

    assert gate[0, 0, 0].item() == 0.0
    assert gate[1, 0, 0].item() == 0.0
    assert gate[2, 0, 0].item() > 0.0
