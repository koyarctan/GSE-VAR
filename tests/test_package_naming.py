import importlib
from pathlib import Path
import pickle

import numpy as np
import pytest

import gse_var


def test_project_metadata_and_package_discovery():
    tomllib = pytest.importorskip("tomllib")
    config = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    assert config["project"]["name"] == "gse-var"
    assert config["project"]["urls"]["Repository"].endswith("/GSE-VAR")
    assert config["tool"]["setuptools"]["packages"]["find"]["include"] == [
        "gse_var*",
        "xneural_var*",
    ]


def test_package_import_stays_lightweight():
    import subprocess
    import sys

    code = (
        "import sys; import gse_var; import xneural_var; "
        "assert 'torch' not in sys.modules; "
        "assert xneural_var.construct_lagged_dataset "
        "is gse_var.construct_lagged_dataset"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    "module_name",
    ["cmlp", "data", "gvar", "interpretability", "models",
     "regularizers", "training", "varx", "visualization"],
)
def test_legacy_submodules_are_canonical_module_objects(module_name):
    if module_name not in ("data", "visualization"):
        pytest.importorskip("torch")
    legacy = importlib.import_module(f"xneural_var.{module_name}")
    canonical = importlib.import_module(f"gse_var.{module_name}")
    assert legacy is canonical


@pytest.mark.parametrize("name", gse_var.__all__)
def test_canonical_public_exports_resolve(name):
    if name.startswith("plot_"):
        # Importing visualization must not require Matplotlib.
        pass
    elif name not in (
        "LaggedDataset", "VARXLaggedDataset", "construct_lagged_dataset",
        "construct_varx_lagged_dataset",
    ):
        pytest.importorskip("torch")
    assert getattr(gse_var, name) is not None


@pytest.mark.parametrize(
    ("previous", "canonical"),
    [
        ("GVARWithNGCGates", "GSEVAR"),
        ("GVARTrainingConfig", "GSEVARTrainingConfig"),
        ("FitResult", "GSEVARFitResult"),
        ("fit_gvar_ngc", "fit_gse_var"),
        ("XNeuralVARX", "GSEVARX"),
        ("XNeuralVARXTrainingConfig", "GSEVARXTrainingConfig"),
        ("XNeuralVARXFitResult", "GSEVARXFitResult"),
        ("XNeuralVARXJacobianAgreementResult", "GSEVARXJacobianAgreementResult"),
        ("fit_xneural_varx", "fit_gse_varx"),
    ],
)
def test_previous_public_api_names_are_identical_aliases(previous, canonical):
    pytest.importorskip("torch")
    import xneural_var

    expected = getattr(gse_var, canonical)
    assert getattr(gse_var, previous) is expected
    assert getattr(xneural_var, previous) is expected


@pytest.mark.parametrize(
    ("canonical_name", "legacy_name", "arguments"),
    [
        ("GSEVAR", "GVARWithNGCGates", {"num_vars": 2}),
        (
            "GSEVARX", "XNeuralVARX",
            {"num_endogenous": 2, "num_exogenous": 1,
             "exog_order": 0, "include_current_exog": True},
        ),
    ],
)
def test_legacy_full_model_pickle_loads_with_canonical_class(
    canonical_name, legacy_name, arguments,
):
    pytest.importorskip("torch")

    # Protocol zero makes module/class globals easy to substitute, emulating
    # a full model saved before the rename without keeping a binary fixture.
    canonical_class = getattr(gse_var, canonical_name)
    model = canonical_class(order=1, hidden_layer_size=3, **arguments)
    payload = pickle.dumps(model, protocol=0)
    original_global = f"cgse_var.models\n{canonical_name}\n".encode()
    assert original_global in payload
    payload = payload.replace(
        original_global,
        f"cxneural_var.models\n{legacy_name}\n".encode(),
    ).replace(
        b"cgse_var.models\nTargetLagwiseMLP\n",
        b"cxneural_var.models\nTargetLagwiseMLP\n",
    )
    restored = pickle.loads(payload)
    assert type(restored) is canonical_class
    assert restored.state_dict().keys() == model.state_dict().keys()


def test_previous_training_api_still_runs():
    pytest.importorskip("torch")
    from xneural_var import GVARTrainingConfig, fit_gvar_ngc

    config = GVARTrainingConfig(
        order=1,
        hidden_layer_size=3,
        max_epochs=1,
        verbose=0,
        device="cpu",
    )
    data = np.random.default_rng(21).normal(size=(8, 2)).astype("float32")
    result = fit_gvar_ngc(data, config)
    assert type(result).__name__ == "GSEVARFitResult"
    assert type(result.model).__name__ == "GSEVAR"
    assert len(result.history["loss"]) == 1
