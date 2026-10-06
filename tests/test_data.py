import numpy as np

from gse_var.data import (
    construct_lagged_dataset,
    construct_varx_lagged_dataset,
)


def test_construct_lagged_dataset_uses_oldest_to_newest_lag_order():
    data = np.arange(10, dtype=np.float32).reshape(5, 2)

    dataset = construct_lagged_dataset(data, order=2)

    assert dataset.predictors.shape == (3, 2, 2)
    np.testing.assert_array_equal(dataset.predictors[0], data[0:2])
    np.testing.assert_array_equal(dataset.responses[0], data[2])


def test_construct_varx_dataset_aligns_current_and_lagged_exog():
    endog = np.arange(12, dtype=np.float32).reshape(6, 2)
    exog = np.arange(100, 106, dtype=np.float32).reshape(6, 1)

    dataset = construct_varx_lagged_dataset(
        endog,
        exog,
        order=2,
        exog_order=1,
        include_current_exog=True,
    )

    assert dataset.endogenous_predictors.shape == (4, 2, 2)
    assert dataset.exogenous_predictors.shape == (4, 2, 1)
    np.testing.assert_array_equal(dataset.endogenous_predictors[0], endog[0:2])
    np.testing.assert_array_equal(
        dataset.exogenous_predictors[0, :, 0],
        exog[[1, 2], 0],
    )
    np.testing.assert_array_equal(dataset.responses[0], endog[2])
    np.testing.assert_array_equal(dataset.exogenous_lags, [1, 0])


def test_construct_varx_dataset_can_exclude_current_exog():
    endog = np.arange(12, dtype=np.float32).reshape(6, 2)
    exog = np.arange(100, 106, dtype=np.float32).reshape(6, 1)

    dataset = construct_varx_lagged_dataset(
        endog,
        exog,
        order=1,
        exog_order=2,
        include_current_exog=False,
    )

    np.testing.assert_array_equal(
        dataset.exogenous_predictors[0, :, 0],
        exog[[0, 1], 0],
    )
    np.testing.assert_array_equal(dataset.exogenous_lags, [2, 1])
