from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gse_var import (
    ExogenousFeature, ExogenousLayout, GSEVARX, GSEVARXTrainingConfig,
    construct_varx_lagged_dataset, fit_gse_varx, evaluate_coefficients,
    causal_gate_matrices,
)
from gse_var.regularizers import (
    ExogenousGroupRegularizer,
    prox_nonnegative_hierarchical_group_lasso_,
)


def layout():
    return ExogenousLayout((ExogenousFeature("promo", 0), ExogenousFeature("promo", 2),
                            ExogenousFeature("oil", 1), ExogenousFeature("payday", -2, known_ahead=True),
                            ExogenousFeature("payday", 0), ExogenousFeature("holiday", 0)))


def model():
    return GSEVARX(num_endogenous=2, num_exogenous=6, order=2, exog_order=0,
                   include_current_exog=True, hidden_layer_size=3, exogenous_layout=layout())


def test_layout_keeps_identity_mask_and_roundtrip():
    spec = layout()
    np.testing.assert_array_equal(spec.lags, [2, 1, 0, -2])
    assert spec.sources == ("promo", "oil", "payday", "holiday")
    assert spec.mask.sum() == 6
    gate = np.arange(12).reshape(1, 2, 6)
    dense = spec.gate_by_lag(gate)
    assert dense.shape == (4, 2, 4)
    assert np.isnan(dense[0, :, 1]).all()
    np.testing.assert_array_equal(dense[2, :, 0], gate[0, :, 0])
    np.testing.assert_array_equal(dense[0, :, 0], gate[0, :, 1])
    coefficients = np.stack([gate, gate + 10])
    unpacked = spec.coefficients_by_lag(coefficients)
    assert unpacked.shape == (2, 4, 2, 4)
    np.testing.assert_array_equal(unpacked[:, 3, :, 2], coefficients[:, 0, :, 3])


@pytest.mark.parametrize("feature", [("x", True), ("x", 1.5), ("", 0), ("x", -1)])
def test_invalid_identity_is_rejected(feature):
    with pytest.raises(ValueError):
        ExogenousFeature(*feature)


def test_duplicate_and_inconsistent_labels_rejected():
    with pytest.raises(ValueError, match="unique"):
        ExogenousLayout((ExogenousFeature("x"), ExogenousFeature("x")))
    with pytest.raises(ValueError, match="label"):
        ExogenousLayout((ExogenousFeature("x", 0, "A"), ExogenousFeature("x", 1, "B")))


def test_raw_construction_indexes_actual_lags_and_future_only_when_declared():
    y = np.arange(24).reshape(12, 2)
    x = np.column_stack([100 + np.arange(14), 200 + np.arange(14), 300 + np.arange(14)])
    args = dict(order=2, exog_names=["promo", "oil", "payday"],
                exog_lags={"promo": [0, 1, 4], "oil": [1, 5], "payday": [-2, -1, 0, 1, 2]})
    with pytest.raises(ValueError, match="known_ahead"):
        construct_varx_lagged_dataset(y, x, **args)
    ds = construct_varx_lagged_dataset(y, x, **args, known_future_exog=["payday"])
    assert ds.endogenous_predictors.shape == (7, 2, 2)
    assert ds.exogenous_predictors.shape == (7, 1, 10)
    np.testing.assert_array_equal(ds.time_index, np.arange(5, 12))
    for sample, t in enumerate(ds.time_index):
        np.testing.assert_array_equal(ds.endogenous_predictors[sample], y[t-2:t])
        expected = [x[t-f.lag, args["exog_names"].index(f.source)] for f in ds.exogenous_layout.features]
        np.testing.assert_array_equal(ds.exogenous_predictors[sample, 0], expected)
    # No trailing known-ahead rows => trim the final two responses, not pad/leak.
    short = construct_varx_lagged_dataset(y, x[:12], **args, known_future_exog=["payday"])
    assert short.time_index[-1] == 9


def test_expanded_inputs_not_shifted_twice_and_blocks_not_crossed():
    y = [np.ones((6, 2)), np.ones((7, 2)) * 10]
    x = [np.ones((6, 6)) * 2, np.ones((7, 6)) * 20]
    ds = construct_varx_lagged_dataset(y, x, order=2, exog_features=layout())
    assert ds.responses.shape == (9, 2)
    np.testing.assert_array_equal(ds.exogenous_predictors[0, 0], x[0][2])
    np.testing.assert_array_equal(ds.exogenous_predictors[4, 0], x[1][2])
    assert ds.time_index[4] - ds.time_index[3] > 1
    with pytest.raises(ValueError, match="exog_order"):
        construct_varx_lagged_dataset(y, x, order=2, exog_order=1, exog_features=layout())


def test_group_penalty_and_nonnegative_prox_match_closed_form():
    spec = layout()
    values = torch.tensor([[[0.4, 0.3, -0.2, 0.8, 0.1, 0.01],
                            [0.01, 0.02, 0.5, 0.3, 0.4, 0.7]]], dtype=torch.float64)
    reg = ExogenousGroupRegularizer(spec, sparse_group_lambda=0.2,
                                    sparse_l1_lambda=0.1)
    expected_penalty = 0.1 * values.abs().sum()
    expected_prox = values.clone()
    for group in spec.groups:
        expected_penalty += 0.2 * torch.linalg.vector_norm(values[0, :, group], dim=-1).sum()
        block = (values[0, :, group] - 0.1).clamp(min=0)
        norm = torch.linalg.vector_norm(block, dim=-1, keepdim=True)
        expected_prox[0, :, group] = block * (1 - 0.2 / norm.clamp(min=1e-12)).clamp(min=0)
    torch.testing.assert_close(reg.penalty(values), expected_penalty)
    got = values.clone()
    reg.prox_(got, 1.0, nonnegative=True)
    torch.testing.assert_close(got, expected_prox)
    assert got[0, 1, :2].count_nonzero() == 0
    assert got[0, 0, 2].item() == 0


def test_regularizer_rejects_arbitrary_lead_hierarchy():
    with pytest.raises(ValueError, match="leads"):
        ExogenousGroupRegularizer(layout(), name="hierarchical_group_lasso", lam=0.1)


def test_group_prox_treats_equal_norm_sources_equally_with_different_lag_counts():
    spec = ExogenousLayout(tuple(ExogenousFeature("promo", lag) for lag in (0, 1, 2)) +
                            (ExogenousFeature("holiday", 0),))
    reg = ExogenousGroupRegularizer(spec, name="group_lasso", lam=0.4)
    gate = torch.tensor([[[0.6, 0.8, 0.0, 1.0]]], dtype=torch.float64)
    torch.testing.assert_close(reg.penalty(gate), gate.new_tensor(0.8))
    reg.prox_(gate, 1.0, nonnegative=True)
    torch.testing.assert_close(gate, gate.new_tensor([[[0.36, 0.48, 0.0, 0.6]]]))


def test_hierarchical_penalty_and_prox_are_unweighted_for_variable_lags():
    # Deliberately interleave sources and lag order in the packed columns.
    spec = ExogenousLayout((ExogenousFeature("promo", 0), ExogenousFeature("oil", 3),
                            ExogenousFeature("promo", 2), ExogenousFeature("holiday", 0),
                            ExogenousFeature("oil", 1), ExogenousFeature("promo", 1)))
    values = torch.tensor([[[0.4, 0.03, -0.2, 0.01, 0.5, 0.3],
                            [0.1, 0.8, 0.2, 0.7, 0.4, 0.01],
                            [0.6, 0.4, 0.8, -0.2, 0.3, 0.5]]], dtype=torch.float64)
    reg = ExogenousGroupRegularizer(spec, name="hierarchical_group_lasso", lam=0.4)
    gate = values[0]
    expected_penalty = 0.4 * (
        gate[:, 2].abs().sum()
        + gate[:, [2, 5]].norm(dim=-1).sum()
        + gate[:, [2, 5, 0]].norm(dim=-1).sum()
        + gate[:, 1].abs().sum()
        + gate[:, [1, 4]].norm(dim=-1).sum()
        + gate[:, 3].abs().sum()
    )
    torch.testing.assert_close(reg.penalty(values), expected_penalty)
    expected_prox = values.clone()
    for indices in ([2, 5, 0], [1, 4], [3]):
        # The existing endogenous HGL is the unweighted reference operator.
        block = values[0, :, indices].T.unsqueeze(-1).clone()
        prox_nonnegative_hierarchical_group_lasso_(block, 0.4, 0.5)
        expected_prox[0, :, indices] = block[..., 0].T
    got = values.clone()
    reg.prox_(got, 0.5, nonnegative=True)
    torch.testing.assert_close(got, expected_prox)
    assert got[0, 0, 3].item() == 0


@pytest.mark.parametrize("name", ["group_lasso", "hierarchical_group_lasso", "sparse_group_lasso"])
def test_single_lag_source_has_no_size_multiplier(name):
    spec = ExogenousLayout((ExogenousFeature("holiday", 0),))
    kwargs = (dict(sparse_group_lambda=0.2, sparse_l1_lambda=0.1)
              if name == "sparse_group_lasso" else dict(lam=0.3))
    reg = ExogenousGroupRegularizer(spec, name=name, **kwargs)
    gate = torch.tensor([[[0.1], [0.4]]], dtype=torch.float64)
    torch.testing.assert_close(reg.penalty(gate), gate.new_tensor(0.15))
    reg.prox_(gate, 0.5, nonnegative=True)
    torch.testing.assert_close(gate, gate.new_tensor([[[0.0], [0.25]]]))


def test_group_size_weighting_options_are_removed():
    with pytest.raises(TypeError, match="size_weights"):
        ExogenousGroupRegularizer(layout(), size_weights=True)
    with pytest.raises(TypeError, match="exogenous_group_size_weights"):
        GSEVARXTrainingConfig(exogenous_group_size_weights=True)


def test_aggregate_then_threshold_not_threshold_then_or():
    m = model()
    with torch.no_grad():
        m.exogenous_gate.zero_()
        m.exogenous_gate[0, 0, :2] = 0.08
    matrices = causal_gate_matrices(m, branch="exogenous", threshold=0.1)
    assert np.nansum(matrices.by_lag[:, 0, 0]) == 0
    assert matrices.aggregated[0, 0] == 1
    np.testing.assert_array_equal(m.exogenous_graph_from_gate(0.1), matrices.aggregated)
    past = causal_gate_matrices(m, branch="exogenous", lag_scope="past", threshold=0)
    assert (past.lags > 0).all()
    assert np.isnan(past.aggregated[:, 2:]).all()


@pytest.mark.parametrize("optimizer", ["ista", "adam"])
@pytest.mark.parametrize("aggregation", ["max", "mean", "median"])
def test_training_separate_regularizers_shapes_and_jacobian_gate_detachment(optimizer, aggregation):
    rng = np.random.default_rng(8)
    y, x = rng.normal(size=(12, 2)), rng.normal(size=(12, 6))
    cfg = GSEVARXTrainingConfig(order=2, hidden_layer_size=3, max_epochs=1, batch_size=16,
                                regularizer="hierarchical_group_lasso", lambda_ngc=0.01,
                                regularizer_exog="sparse_group_lasso",
                                sparse_group_lambda_exog=0.02, sparse_l1_lambda_exog=0.01,
                                lambda_jacobian=0.1, optimizer=optimizer, strength_aggregation=aggregation,
                                verbose=0, device="cpu")
    result = fit_gse_varx(y, x, cfg, exog_features=layout(), endog_names=["salesA", "salesB"])
    assert result.exogenous_coeffs.shape == (10, 1, 2, 6)
    assert result.exogenous_graph.shape == (2, 4)
    assert result.exogenous_strength.shape == (2, 4)
    assert result.exogenous_graph_by_lag.shape == (4, 2, 4)
    assert result.exogenous_coeffs_by_lag.shape == (10, 4, 2, 4)
    assert result.exogenous_term_graph.shape == (2, 6)
    assert all(np.isfinite(v).all() for v in result.history.values())
    for source, group in enumerate(layout().groups):
        c = torch.from_numpy(np.take(result.exogenous_coeffs[:, 0], group, axis=-1)).permute(0, 2, 1).unsqueeze(-1)
        expected = result.model.coefficient_strength(c, aggregation).numpy().ravel()
        np.testing.assert_allclose(result.exogenous_strength[:, source], expected)
    ds = construct_varx_lagged_dataset(y, x, order=2, exog_features=layout())
    result.model.zero_grad(set_to_none=True)
    mismatches = result.model.coefficient_jacobian_mismatches_for_regularization(
        torch.tensor(ds.endogenous_predictors), torch.tensor(ds.exogenous_predictors))
    sum(v.square().mean() for v in mismatches).backward()
    assert result.model.endogenous_gate.grad is None
    assert result.model.exogenous_gate.grad is None


def test_evaluation_reads_only_predictors_and_restores_model_state():
    m = model()
    y, x = np.ones((5, 2, 2)), np.ones((5, 1, 6))
    ds = construct_varx_lagged_dataset(np.ones((7, 2)), np.ones((7, 6)), order=2, exog_features=layout())
    ds.responses[:] = np.nan  # Deliberately unusable: must never be accessed.
    m.train()
    saved = {k: v.clone() for k, v in m.state_dict().items()}
    evaluation = evaluate_coefficients(m, ds, batch_size=2)
    direct = evaluate_coefficients(m, (y, x), batch_size=5)
    np.testing.assert_allclose(evaluation.exogenous_coeffs, direct.exogenous_coeffs, rtol=1e-5)
    assert m.training
    for k, v in m.state_dict().items():
        torch.testing.assert_close(v, saved[k])
    with pytest.raises(ValueError, match="identities"):
        evaluate_coefficients(m, replace(ds, exogenous_layout=None))


def test_three_targets_and_five_lags_group_prox_preserves_axes():
    spec = ExogenousLayout(tuple(ExogenousFeature("x", k) for k in range(5)) +
                            (ExogenousFeature("z", 0),))
    reg = ExogenousGroupRegularizer(spec, name="group_lasso", lam=0.4)
    gate = torch.arange(18, dtype=torch.float64).reshape(1, 3, 6) / 10
    expected = gate.clone()
    for row in range(3):
        for group in spec.groups:
            block = gate[0, row, list(group)]
            norm = block.norm()
            expected[0, row, list(group)] = block * max(0, 1 - 0.4 / norm.item())
    reg.prox_(gate, 1, nonnegative=True)
    torch.testing.assert_close(gate, expected)


def test_raw_api_fit_and_evaluation_layouts_agree():
    y, x = np.ones((10, 2)), np.arange(24).reshape(12, 2).astype(float)
    kwargs = dict(exog_names=["promo", "calendar"], exog_lags={"promo": [0, 3], "calendar": [-2, 0]},
                  known_future_exog=["calendar"])
    cfg = GSEVARXTrainingConfig(order=2, hidden_layer_size=2, max_epochs=1,
                                regularizer="group_lasso", lambda_ngc=0.01,
                                regularizer_exog="sparse_group_lasso",
                                sparse_group_lambda_exog=0.01, verbose=0)
    result = fit_gse_varx(y, x, cfg, **kwargs)
    ds = construct_varx_lagged_dataset(y, x, order=2, **kwargs)
    ev = evaluate_coefficients(result, ds)
    np.testing.assert_allclose(ev.exogenous_coeffs, result.exogenous_coeffs, rtol=1e-5)
    assert result.exogenous_graph.shape == (2, 2)
