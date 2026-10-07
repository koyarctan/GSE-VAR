from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gse_var import (
    ExogenousFeature, ExogenousLayout, GSEVARX, GSEVARXTrainingConfig,
    construct_varx_lagged_dataset, fit_gse_varx,
)
import gse_var.varx as varx


METRICS = (
    "loss", "mse", "ngc", "ngc_endogenous", "ngc_exogenous", "smooth",
    "smooth_endogenous", "smooth_exogenous", "jacobian", "jacobian_endogenous",
    "jacobian_exogenous",
)


def setup(packed=True, **overrides):
    rng = np.random.default_rng(27)
    train = (rng.normal(size=(12, 2)), rng.normal(size=(12, 4)))
    valid = (rng.normal(size=(8, 2)), rng.normal(size=(8, 4)))
    layout = ExogenousLayout((
        ExogenousFeature("promo", 0), ExogenousFeature("promo", 1),
        ExogenousFeature("calendar", -1, known_ahead=True), ExogenousFeature("oil", 1),
    ))
    kwargs = {"exog_features": layout} if packed else {}
    config = GSEVARXTrainingConfig(
        order=2, hidden_layer_size=3, max_epochs=8, batch_size=4,
        regularizer="hierarchical_group_lasso", lambda_ngc=0.01,
        regularizer_exog="sparse_group_lasso",
        sparse_group_lambda_exog=0.02, sparse_l1_lambda_exog=0.01,
        early_stopping=True, early_stopping_patience=2, verbose=0, device="cpu",
    )
    validation = construct_varx_lagged_dataset(*valid, order=2, **kwargs)
    return train, valid, validation, replace(config, **overrides), kwargs


def mock_epochs(monkeypatch, scores):
    states = []
    scores = iter(scores)

    def epoch(model, *args):
        model.train()
        number = len(states) + 1
        with torch.no_grad():
            for i, parameter in enumerate(model.parameters()):
                parameter.fill_(number / 20 + i / 1000)
        states.append({key: value.clone() for key, value in model.state_dict().items()})
        return {key: float(number) for key in METRICS}

    monkeypatch.setattr(varx, "_epoch", epoch)
    monkeypatch.setattr(varx, "_validation_mse", lambda *args: next(scores))
    return states


def assert_state(model, expected):
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("restore", [False, True])
def test_patience_and_restore_all_weights_before_inference(monkeypatch, packed, restore):
    train, _, validation, config, kwargs = setup(packed, restore_best_weights=restore)
    states = mock_epochs(monkeypatch, [1.0, 0.5, 0.6, 0.7])
    result = fit_gse_varx(*train, config, validation_data=validation, **kwargs)
    assert result.early_stopped and result.stopped_epoch == 4
    assert result.epochs_trained == 4 and result.best_epoch == 2
    assert result.best_validation_mse == 0.5
    assert result.history["val_mse"] == [1.0, 0.5, 0.6, 0.7]
    assert_state(result.model, states[1 if restore else 3])
    train_ds = construct_varx_lagged_dataset(*train, order=2, **kwargs)
    with torch.no_grad():
        _, yc, xc = result.model(torch.tensor(train_ds.endogenous_predictors),
                                  torch.tensor(train_ds.exogenous_predictors))
    np.testing.assert_array_equal(result.endogenous_coeffs, yc.numpy())
    np.testing.assert_array_equal(result.exogenous_coeffs, xc.numpy())
    np.testing.assert_array_equal(result.exogenous_graph,
                                  result.model.exogenous_graph_from_gate(config.causal_threshold).numpy())
    assert result.endogenous_coeffs.shape[0] == 10  # never append validation responses


def test_monitor_only_does_not_stop_or_restore(monkeypatch):
    train, _, validation, config, kwargs = setup(early_stopping=False, max_epochs=5)
    states = mock_epochs(monkeypatch, [1.0, 0.5, 0.6, 0.7, 0.8])
    result = fit_gse_varx(*train, config, validation_data=validation, **kwargs)
    assert not result.early_stopped and result.stopped_epoch is None
    assert result.epochs_trained == 5 and result.best_epoch == 2
    assert_state(result.model, states[-1])


def test_best_state_is_restored_even_when_max_epochs_is_reached(monkeypatch):
    train, _, validation, config, kwargs = setup(max_epochs=2)
    states = mock_epochs(monkeypatch, [0.4, 0.5])
    result = fit_gse_varx(*train, config, validation_data=validation, **kwargs)
    assert result.epochs_trained == 2 and not result.early_stopped
    assert result.best_epoch == 1
    assert_state(result.model, states[0])


@pytest.mark.parametrize("scores, delta, patience, stopped, best", [
    ([1.0, 0.96, 0.93], 0.1, 2, 3, 3),
    ([1.0, 0.96, 0.89, 0.9, 0.91], 0.1, 2, 5, 3),
    ([1.0, 0.5], 0.5, 1, 2, 2),
    ([1.0, 1.0], 0.0, 1, 2, 1),
])
def test_min_delta_resets_patience_but_still_restores_actual_minimum(
    monkeypatch, scores, delta, patience, stopped, best,
):
    train, _, validation, config, kwargs = setup(
        early_stopping_min_delta=delta, early_stopping_patience=patience,
    )
    states = mock_epochs(monkeypatch, scores)
    result = fit_gse_varx(*train, config, validation_data=validation, **kwargs)
    assert result.stopped_epoch == stopped and result.best_epoch == best
    assert result.best_validation_mse == min(scores)
    assert_state(result.model, states[best - 1])


def test_missing_validation_is_rejected_before_training():
    train, _, _, config, kwargs = setup()
    with pytest.raises(ValueError, match="requires explicit validation_data"):
        fit_gse_varx(*train, config, **kwargs)


@pytest.mark.parametrize("field, value", [
    ("early_stopping", 1), ("restore_best_weights", 1),
    ("early_stopping_patience", 0), ("early_stopping_patience", -1),
    ("early_stopping_patience", True), ("early_stopping_patience", 1.5),
    ("early_stopping_min_delta", -0.1), ("early_stopping_min_delta", float("nan")),
    ("early_stopping_min_delta", float("inf")), ("max_epochs", 0),
])
def test_invalid_early_stopping_config_is_rejected(field, value):
    train, _, validation, config, kwargs = setup()
    with pytest.raises(ValueError):
        fit_gse_varx(*train, replace(config, **{field: value}), validation_data=validation, **kwargs)


@pytest.mark.parametrize("case", [
    "empty", "order", "endog_width", "exog_width", "targets", "identities",
    "lags", "time_index", "series_index", "nan_y", "nan_x", "nan_targets",
])
def test_invalid_validation_is_rejected_without_modifying_supplied_model(case):
    train, _, validation, config, kwargs = setup()
    changes = {}
    if case == "empty":
        changes["responses"] = validation.responses[:0]
    elif case == "order":
        changes["endogenous_predictors"] = validation.endogenous_predictors[:, :1]
    elif case == "endog_width":
        changes["endogenous_predictors"] = validation.endogenous_predictors[:, :, :1]
    elif case == "exog_width":
        changes["exogenous_predictors"] = validation.exogenous_predictors[:, :, :1]
    elif case == "targets":
        changes["responses"] = validation.responses[:, :1]
    elif case == "identities":
        changes["exogenous_layout"] = None
    elif case == "lags":
        changes["exogenous_lags"] = np.array([0])
    elif case in ("time_index", "series_index"):
        changes[case] = getattr(validation, case)[:-1]
    else:
        field = {"nan_y": "endogenous_predictors", "nan_x": "exogenous_predictors",
                 "nan_targets": "responses"}[case]
        changes[field] = getattr(validation, field).copy()
        changes[field].flat[0] = np.nan
    model = GSEVARX(num_endogenous=2, num_exogenous=4, order=2,
                    exog_order=0, include_current_exog=True,
                    hidden_layer_size=3, exogenous_layout=kwargs["exog_features"])
    state = {key: value.clone() for key, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="validation"):
        fit_gse_varx(*train, config, model=model, validation_data=replace(validation, **changes), **kwargs)
    assert_state(model, state)


@pytest.mark.parametrize("packed", [False, True])
def test_raw_tuple_and_preconstructed_validation_are_equivalent(packed):
    train, valid, validation, config, kwargs = setup(packed, max_epochs=2)
    results = [fit_gse_varx(*train, config, validation_data=data, **kwargs)
               for data in (valid, validation)]
    assert results[0].history == results[1].history
    np.testing.assert_array_equal(results[0].exogenous_coeffs, results[1].exogenous_coeffs)


def test_raw_variable_specific_validation_preserves_leads_and_metadata():
    rng = np.random.default_rng(12)
    train = (rng.normal(size=(12, 2)), rng.normal(size=(14, 2)))
    valid = (rng.normal(size=(8, 2)), rng.normal(size=(10, 2)))
    _, _, _, config, _ = setup(max_epochs=1)
    kwargs = dict(exog_names=["promo", "calendar"],
                  exog_lags={"promo": [0, 2], "calendar": [-2, 0]},
                  known_future_exog=["calendar"])
    result = fit_gse_varx(*train, config, validation_data=valid, **kwargs)
    ds = construct_varx_lagged_dataset(*valid, order=2, **kwargs)
    assert result.exogenous_coeffs.shape == (10, 1, 2, 4)
    assert ds.responses.shape == (6, 2)
    assert result.exogenous_layout == ds.exogenous_layout
    assert result.best_epoch == 1


@pytest.mark.parametrize("was_training", [False, True])
@pytest.mark.parametrize("batch_size", [1, 4, 100])
def test_validation_mse_is_sample_weighted_and_read_only(was_training, batch_size):
    _, _, validation, _, kwargs = setup()
    data = varx._to_torch_varx_dataset(validation, torch.device("cpu"))
    model = GSEVARX(num_endogenous=2, num_exogenous=4, order=2,
                    exog_order=0, include_current_exog=True,
                    hidden_layer_size=3, exogenous_layout=kwargs["exog_features"])
    model.train(was_training)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.25)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        predictions, _, _ = model(data.endogenous_predictors, data.exogenous_predictors)
        expected = nn_mse(predictions, data.responses)
    actual = varx._validation_mse(model, data, batch_size)
    assert actual == pytest.approx(expected, rel=1e-6)
    assert model.training == was_training
    assert_state(model, state)
    for parameter in model.parameters():
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 0.25))


def nn_mse(predictions, responses):
    return float((predictions - responses).square().mean())


def test_nonfinite_validation_loss_is_rejected_and_mode_is_restored(monkeypatch):
    _, _, validation, _, kwargs = setup()
    data = varx._to_torch_varx_dataset(validation, torch.device("cpu"))
    model = GSEVARX(num_endogenous=2, num_exogenous=4, order=2,
                    exog_order=0, include_current_exog=True,
                    hidden_layer_size=3, exogenous_layout=kwargs["exog_features"])
    model.train()
    monkeypatch.setattr(model, "forward", lambda y, x: (torch.full((len(y), 2), float("nan")), None, None))
    with pytest.raises(ValueError, match="MSE must be finite"):
        varx._validation_mse(model, data, 4)
    assert model.training


def test_monitoring_validation_does_not_change_training_updates():
    train, _, validation, config, kwargs = setup(early_stopping=False, max_epochs=2)
    without = fit_gse_varx(*train, config, **kwargs)
    monitored = fit_gse_varx(*train, config, validation_data=validation, **kwargs)
    assert "val_mse" not in without.history
    assert without.best_epoch is None and without.best_validation_mse is None
    assert without.epochs_trained == 2
    assert_state(monitored.model, without.model.state_dict())
    for key in without.history:
        assert without.history[key] == monitored.history[key]


def test_verbose_logging_includes_validation_and_stop_information(monkeypatch, capsys):
    train, _, validation, config, kwargs = setup(verbose=1, early_stopping_patience=1)
    mock_epochs(monkeypatch, [1.0, 2.0])
    fit_gse_varx(*train, config, validation_data=validation, **kwargs)
    output = capsys.readouterr().out
    assert "val_mse=" in output
    assert "Early stopping at epoch 2" in output
    assert "Restored best weights from epoch 1" in output


@pytest.mark.parametrize("case", ["array", "list", "short_tuple", "long_tuple"])
def test_invalid_validation_container_is_rejected(case):
    train, valid, _, config, kwargs = setup()
    invalid = {"array": valid[0], "list": list(valid),
               "short_tuple": valid[:1], "long_tuple": valid + (valid[0],)}[case]
    with pytest.raises(ValueError, match="validation_data must be"):
        fit_gse_varx(*train, config, validation_data=invalid, **kwargs)


def test_default_disabled_and_numpy_integer_patience():
    train, _, validation, config, kwargs = setup()
    assert GSEVARXTrainingConfig(order=2, hidden_layer_size=3).early_stopping is False
    result = fit_gse_varx(*train, replace(config, max_epochs=1,
                          early_stopping_patience=np.int64(2)), validation_data=validation, **kwargs)
    assert result.epochs_trained == 1
