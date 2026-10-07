import numpy as np
import pytest

torch = pytest.importorskip("torch")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gse_var import (
    GSEVAR, GSEVARX, ExogenousFeature, ExogenousLayout,
    plot_causal_gate_by_lag, plot_causal_graph_matrix,
    plot_edge_lag_boxplots, plot_edge_lag_forest,
)
from gse_var.varx import GSEVARXFitResult


@pytest.fixture
def fitted():
    layout = ExogenousLayout((ExogenousFeature("promo", 0, "Promotion"),
                              ExogenousFeature("promo", 1, "Promotion"),
                              ExogenousFeature("oil", 2, "Oil"),
                              ExogenousFeature("payday", -1, "Payday", known_ahead=True)))
    m = GSEVARX(num_endogenous=2, num_exogenous=4, order=2, exog_order=0,
                include_current_exog=True, hidden_layer_size=1, exogenous_layout=layout)
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(1)
    return GSEVARXFitResult(model=m, exogenous_layout=layout,
                            endogenous_names=("SalesA", "SalesB"),
                            exogenous_coeffs=np.full((4, 1, 2, 4), 9999.0))


def test_gate_plot_rectangular_binary_lag_labels_and_unused_masks(fitted):
    fig, axes = plot_causal_gate_by_lag(fitted, branch="exogenous", binary=True, show=False)
    fig.canvas.draw()
    images = [ax.images[0] for ax in axes.flat if ax.images]
    assert len(images) == 5
    assert images[0].get_array().shape == (2, 3)
    assert np.ma.getmaskarray(images[0].get_array())[:, 0].all()
    assert {ax.get_title() for ax in axes.flat if ax.images} == {
        "lag 2", "lag 1", "current", "lead 1", "Aggregated graph"}
    plt.close(fig)
    fig, ax = plot_causal_graph_matrix(fitted, branch="exogenous", show=False)
    assert ax.images[0].get_array().shape == (2, 3)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Promotion", "Oil", "Payday"]
    plt.close(fig)


def test_forest_uses_supplied_data_not_training_coefficients(fitted):
    y = np.ones((5, 2, 2), dtype="float32")
    train = np.ones((5, 1, 4), dtype="float32")
    valid = np.arange(20, dtype="float32").reshape(5, 1, 4) + 2
    fig1, ax1 = plot_edge_lag_forest(fitted, (y, train), branch="exogenous",
                                     edges=[(0, 0)], dataset_label="train", show=False)
    fig2, ax2 = plot_edge_lag_forest(fitted, (y, valid), branch="exogenous",
                                     edges=[(0, 0)], dataset_label="valid", show=False)
    c1, c2 = ax1.collections[1].get_offsets()[:, 0], ax2.collections[1].get_offsets()[:, 0]
    assert not np.allclose(c1, c2)
    assert (c1 < 9999).all()
    assert "valid" in ax2.get_title() and "not CI" in ax2.get_title() and "n=5" in ax2.get_title()
    assert len(ax2.get_yticklabels()) == 2  # only the two valid promo lags
    fig1.canvas.draw()
    fig2.canvas.draw()
    plt.close(fig1)
    plt.close(fig2)
    with pytest.raises(TypeError):
        plot_edge_lag_forest(fitted)
    with pytest.raises(ValueError, match="explicit"):
        plot_edge_lag_forest(fitted, None)


def test_boxplots_external_equal_indices_not_mistaken_for_self_edges(fitted):
    data = (np.ones((5, 2, 2)), np.ones((5, 1, 4)))
    fig, axes = plot_edge_lag_boxplots(fitted, branch="exogenous", data=data,
                                      edges=[(0, 0)], show=False)
    assert axes[0, 0].get_title() == "Promotion -> SalesA"
    assert len(axes[0, 0].patches) == 4  # two boxes, two unused-lag spans
    fig.canvas.draw()
    plt.close(fig)


def test_legacy_var_and_uniform_varx_plot_paths():
    m = GSEVAR(num_vars=2, order=2, hidden_layer_size=2)
    inputs = np.ones((4, 2, 2))
    fig, _ = plot_causal_gate_by_lag(m, show=False)
    plt.close(fig)
    fig, _ = plot_edge_lag_forest(m, inputs, edges=[(0, 1)], show=False)
    plt.close(fig)
    x = GSEVARX(num_endogenous=2, num_exogenous=3, order=2, exog_order=1,
                include_current_exog=True, hidden_layer_size=2)
    fig, axes = plot_causal_gate_by_lag(x, branch="exogenous", show=False)
    assert axes.flat[0].images[0].get_array().shape == (2, 3)
    plt.close(fig)
