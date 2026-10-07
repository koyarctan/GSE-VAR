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
    ]


def test_package_import_stays_lightweight():
    import subprocess
    import sys

    code = (
        "import sys; import gse_var; "
        "assert 'torch' not in sys.modules; "
        "assert callable(gse_var.construct_lagged_dataset)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_legacy_package_has_no_implementation_or_training_api():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    old_folder = root / "xneural_var"
    # Windows can keep a deleted package's empty directory locked. An empty
    # namespace directory is not a compatibility implementation or a wheel
    # package; it must contain neither source files nor cached modules.
    if old_folder.exists():
        assert list(old_folder.iterdir()) == []
    spec = importlib.util.find_spec("xneural_var")
    if spec is not None:
        assert spec.origin is None
    with pytest.raises(ImportError, match="xneural_var"):
        from xneural_var import GVARTrainingConfig


@pytest.mark.parametrize(
    "module_name",
    [
        "cmlp", "data", "gvar", "interpretability", "models",
        "regularizers", "training", "varx", "visualization",
    ],
)
def test_canonical_submodules_resolve(module_name):
    if module_name not in ("data", "visualization"):
        pytest.importorskip("torch")
    module = importlib.import_module(f"gse_var.{module_name}")
    assert module.__name__ == f"gse_var.{module_name}"


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
    ("previous", "module_name"),
    [
        ("GVARWithNGCGates", "models"),
        ("GVARTrainingConfig", "training"),
        ("FitResult", "training"),
        ("fit_gvar_ngc", "training"),
        ("XNeuralVARX", "models"),
        ("XNeuralVARXTrainingConfig", "varx"),
        ("XNeuralVARXFitResult", "varx"),
        ("XNeuralVARXJacobianAgreementResult", "interpretability"),
        ("fit_xneural_varx", "varx"),
    ],
)
def test_previous_public_api_names_are_not_exposed(previous, module_name):
    pytest.importorskip("torch")
    module = importlib.import_module(f"gse_var.{module_name}")
    assert not hasattr(gse_var, previous)
    assert not hasattr(module, previous)
    assert previous not in gse_var.__all__
    assert previous not in dir(gse_var)


_MODEL_CASES = [
    ("GSEVAR", {"num_vars": 2}),
    (
        "GSEVARX",
        {
            "num_endogenous": 2,
            "num_exogenous": 1,
            "exog_order": 0,
            "include_current_exog": True,
        },
    ),
]


@pytest.mark.parametrize(("canonical_name", "arguments"), _MODEL_CASES)
def test_canonical_full_model_pickle_roundtrip(canonical_name, arguments):
    pytest.importorskip("torch")
    canonical_class = getattr(gse_var, canonical_name)
    model = canonical_class(order=1, hidden_layer_size=3, **arguments)
    payload = pickle.dumps(model, protocol=0)
    assert f"cgse_var.models\n{canonical_name}\n".encode() in payload

    restored = pickle.loads(payload)

    assert type(restored) is canonical_class
    assert restored.state_dict().keys() == model.state_dict().keys()


@pytest.mark.parametrize(("canonical_name", "arguments"), _MODEL_CASES)
def test_canonical_state_dict_roundtrip_preserves_predictions(
    canonical_name, arguments,
):
    torch = pytest.importorskip("torch")
    canonical_class = getattr(gse_var, canonical_name)
    model = canonical_class(order=1, hidden_layer_size=3, **arguments)
    restored = canonical_class(order=1, hidden_layer_size=3, **arguments)
    restored.load_state_dict(model.state_dict())
    inputs = [torch.randn(4, 1, 2)]
    if canonical_name == "GSEVARX":
        inputs.append(torch.randn(4, 1, 1))

    with torch.no_grad():
        expected = model(*inputs)
        actual = restored(*inputs)

    for expected_tensor, actual_tensor in zip(expected, actual):
        assert torch.equal(expected_tensor, actual_tensor)


def test_canonical_training_api_still_runs():
    pytest.importorskip("torch")
    from gse_var import GSEVARFitResult, GSEVARTrainingConfig, fit_gse_var

    config = GSEVARTrainingConfig(
        order=1,
        hidden_layer_size=3,
        max_epochs=1,
        verbose=0,
        device="cpu",
    )
    data = np.random.default_rng(21).normal(size=(8, 2)).astype("float32")
    result = fit_gse_var(data, config)
    assert isinstance(result, GSEVARFitResult)
    assert type(result.model).__name__ == "GSEVAR"
    assert len(result.history["loss"]) == 1
