"""The standalone quick notebook must bundle the opt-in implementation exactly."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "experiments" / "11_GSE_VAR_lambda簡易探索.ipynb"


def notebook_code():
    if not NOTEBOOK.exists():
        pytest.skip("the locally maintained experiment notebook is not part of the wheel")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for cell in cells:
        compile("".join(cell["source"]), str(NOTEBOOK), "exec")
    training_cells = [cell for cell in cells if "SOURCE_PAYLOAD =" in "".join(cell["source"])]
    assert len(training_cells) == 1  # 診断セルを追加しても学習セルは1つだけ。
    return "".join(training_cells[0]["source"])


def test_standalone_bundle_and_checksums_match_current_library():
    code = notebook_code()
    assignments = {node.targets[0].id: node.value for node in ast.parse(code).body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    for prefix in ("SOURCE", "CORE", "DATA"):
        blob = zlib.decompress(base64.b64decode(ast.literal_eval(assignments[prefix + "_PAYLOAD"])))
        assert hashlib.sha256(blob).hexdigest() == ast.literal_eval(assignments[prefix + "_SHA"])
        if prefix == "SOURCE":
            sources = json.loads(blob)
            assert "gse_var/disarm.py" in sources
            for name, content in sources.items():
                assert content == (ROOT / name).read_text(encoding="utf-8")
                compile(content, name, "exec")
        elif prefix == "CORE":
            compile(blob.decode(), "citibike_core", "exec")
            assert blob.decode() == (ROOT / "experiments/_citibike_quick_core.py").read_text(encoding="utf-8")
    compile(code, str(NOTEBOOK), "exec")


def test_quick_restore_restores_logits_binary_gate_and_predictions_together():
    torch = pytest.importorskip("torch")
    from gse_var import GSEVARDisARM
    function = next(node for node in ast.parse(notebook_code()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "restore_disarm_quick")
    namespace = {"torch": torch, "np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "quick_restore", "exec"), namespace)
    model = GSEVARDisARM(40, 2, 3)
    with torch.no_grad():
        model.gate_logits[:, 2, 1] = -2
        model.update_causal_gate_()
        for parameter in model.coeff_net.parameters():
            parameter.uniform_(-.1, .1)
    arrays = {"cp_epoch": np.array([100])}
    arrays.update({"cp_weight__" + key: value.detach().numpy()[None].copy()
                   for key, value in model.state_dict().items()})
    profile = dict(order=2, hidden_layer_size=3, num_hidden_layers=1,
                   gate_probability_init=.9)
    restored = namespace["restore_disarm_quick"](arrays, 100, profile, "cpu")
    inputs = torch.randn(5, 2, 40)
    for actual, expected in zip(restored(inputs), model(inputs)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="一意"):
        namespace["restore_disarm_quick"](arrays, 200, profile, "cpu")


def test_notebook_exposes_independent_editable_settings_for_each_method():
    assignments = {node.targets[0].id: node.value for node in ast.parse(notebook_code()).body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    assert ast.literal_eval(assignments["CHECK_EVERY"]) > 0
    assert ast.literal_eval(assignments["GSE_METHOD"]) in ("proximal", "disarm")
    original_config = {keyword.arg: ast.literal_eval(keyword.value)
                       for keyword in assignments["MODEL_CONFIG"].keywords}
    binary_config = {keyword.arg: ast.literal_eval(keyword.value)
                     for keyword in assignments["DISARM_CONFIG"].keywords}
    assert {"sparse_group_lambda", "sparse_l1_lambda", "gate_init"} <= original_config.keys()
    assert {"lambda_l0", "gate_probability_init"} <= binary_config.keys()
    assert "lambda_l0" not in original_config
    assert "sparse_group_lambda" not in binary_config
    assert not {"optimizer", "fit_intercept", "causal_threshold"} & original_config.keys()
    assert not {"optimizer", "fit_intercept", "causal_threshold"} & binary_config.keys()


def diagnostic_summary_function():
    notebook_code()  # skip handling and syntax checks
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"]
                if cell.get("id") == "disarm-probability-diagnostics")
    function = next(node for node in ast.parse("".join(cell["source"])).body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "summarize_disarm_gate_probabilities")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "gate_diagnostics", "exec"), namespace)
    return namespace[function.name]


@pytest.mark.parametrize("binary", [False, True])
def test_quick_callbacks_and_checkpoint_restoration_use_the_fixed_public_api(binary):
    torch = pytest.importorskip("torch")
    from dataclasses import asdict
    from gse_var import GSEVARTrainingConfig, GSEVARDisARMTrainingConfig
    code = notebook_code()
    namespace = {"torch": torch, "np": np, "GSEVARDisARMTrainingConfig": GSEVARDisARMTrainingConfig,
                 "COEFFICIENT_POINTS": 2}
    core_path = ROOT / "experiments/_citibike_quick_core.py"
    # Prepared arrays and checkpoint logic need no dataframe or display imports.
    # Execute the actual numerical helpers, omitting only optional UI imports.
    core_tree = ast.parse(core_path.read_text(encoding="utf-8"))
    core_tree.body = [node for node in core_tree.body if not (
        isinstance(node, ast.Import) and any(alias.name in ("pandas", "matplotlib.pyplot") for alias in node.names)
        or isinstance(node, ast.ImportFrom) and node.module == "IPython.display"
    )]
    exec(compile(core_tree, str(core_path), "exec"), namespace)
    for node in ast.parse(code).body:
        if isinstance(node, ast.FunctionDef) and node.name in ("run_disarm_quick", "restore_disarm_quick"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "quick_callbacks", "exec"), namespace)
    rng = np.random.default_rng(285)
    prepared = dict(order=2, window=dict(name="cv_1"),
                    train=rng.normal(size=(12, 40)).astype("float32"),
                    x_eval=rng.normal(size=(3, 2, 40)).astype("float32"),
                    truth=np.ones((3, 40)), truth_log=np.full((3, 40), np.log(2)),
                    offset=np.zeros((3, 40)), scale=np.ones(40), mean=np.zeros(40),
                    calendar_beta=np.zeros((1, 40)), dates=np.arange(3).astype(str))
    cls = GSEVARDisARMTrainingConfig if binary else GSEVARTrainingConfig
    config = cls(order=2, hidden_layer_size=3, max_epochs=3, batch_size=100,
                 learning_rate=.003, gate_learning_rate=.03 if binary else .2,
                 coefficient_weight_decay=.01, shuffle=False, seed=17, verbose=0, device="cpu")
    profile = asdict(config)
    import gse_var.training as training
    original_epoch = training._epoch
    if binary:
        row, arrays = namespace["run_disarm_quick"](prepared, profile, "cpu", [1, 2, 3])
        restore = namespace["restore_disarm_quick"]
    else:
        profile["sparse_group_lambda"] = .01
        row, arrays = namespace["run_gse"](prepared, .01, 17, profile, "cpu", checkpoints=[1, 2, 3])
        restore = namespace["restore_checkpoint"]
    assert training._epoch is original_epoch
    assert "causal_threshold" not in row
    np.testing.assert_array_equal(arrays["cp_epoch"], [1, 2, 3])
    for index, epoch in enumerate(arrays["cp_epoch"]):
        restored = restore(arrays, int(epoch), profile, "cpu")
        scores, _ = namespace["predict_gse"](restored, prepared, "cpu")
        for name in namespace["METRICS"]:
            assert scores[name] == pytest.approx(arrays["cp_"+name][index], rel=1e-12, abs=1e-12)


def test_diagnostics_count_lag_routes_and_any_lag_edges_without_mutating_checkpoints():
    summarize = diagnostic_summary_function()
    logits = np.full((3, 2, 2, 2), 2.0)
    logits[0, 0, 0, 0] = 0  # p=0.5 is open, not closed.
    logits[1, 0, 0, 1] = -1
    logits[2, 0, 0, 1] = 1  # A route may reopen after a closed checkpoint.
    logits[2, 1, 0, 1] = -2
    logits[2, :, 1, 0] = [-4, -3]
    arrays = {"cp_epoch": np.array([100, 200, 300]),
              "cp_weight__gate_logits": logits,
              "cp_weight__causal_gate": (logits >= 0).astype(float)}
    before = {key: value.copy() for key, value in arrays.items()}
    rng_before = np.random.get_state()
    result = summarize(arrays, 200)
    np.testing.assert_array_equal(result["fixed_lag_count"], [8, 7, 5])
    np.testing.assert_array_equal(result["fixed_graph_count"], [4, 4, 3])
    assert result["best_index"] == 1
    assert result["first_observed_closed_epoch"][0, 0, 1] == 200
    assert result["first_observed_closed_epoch"][0, 0, 0] == -1
    expected_probabilities = 1 / (1 + np.exp(-logits))
    np.testing.assert_allclose(result["probabilities"], expected_probabilities)
    np.testing.assert_allclose(result["mean"], expected_probabilities.mean(axis=(1, 2, 3)))
    for key in before:
        np.testing.assert_array_equal(arrays[key], before[key])
    rng_after = np.random.get_state()
    assert rng_before[0] == rng_after[0]
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    assert rng_before[2:] == rng_after[2:]


def test_diagnostics_stable_sigmoid_and_invalid_checkpoints():
    summarize = diagnostic_summary_function()
    logits = np.array([[[[10000., -10000.], [0., 2.]]]])
    arrays = {"cp_epoch": np.array([100]), "cp_weight__gate_logits": logits,
              "cp_weight__causal_gate": (logits >= 0).astype(float)}
    with np.errstate(over="raise", invalid="raise"):
        result = summarize(arrays, 100)
    np.testing.assert_allclose(result["probabilities"][0, 0, 0], [1, 0])
    with pytest.raises(ValueError, match="checkpoint"):
        summarize({}, 100)
    with pytest.raises(ValueError, match="best epoch"):
        summarize(arrays, 200)
    with pytest.raises(ValueError, match="一致"):
        summarize({**arrays, "cp_weight__causal_gate": np.ones_like(logits)}, 100)
    with pytest.raises(ValueError, match="昇順"):
        summarize({**arrays, "cp_epoch": np.array([100.5])}, 100)
    with pytest.raises(ValueError, match="有限"):
        summarize({**arrays, "cp_weight__gate_logits": np.full_like(logits, np.nan)}, 100)
