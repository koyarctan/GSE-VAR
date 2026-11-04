"""Reference-data parity, signed inference, selection boundaries and resumption."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.lotka_volterra.data import generate, published, source_manifest
from experiments.lotka_volterra.protocol import configuration, tasks, select, reference_trgc
from experiments.lotka_volterra.run import completed
from experiments.shared import write_json

ROOT = Path(__file__).resolve().parents[1]


def test_full_protocol_matches_paper_and_saved_author_grid():
    c = configuration()
    assert (c['length'], c['p'], c['order']) == (2000, 20, 1)
    assert (c['gvar_layers'], c['gvar_hidden'], c['gvar_epochs'], c['gvar_batch']) == (2, 50, 500, 256)
    np.testing.assert_allclose(c['gvar_lambdas'], np.linspace(0, 1, 5))
    np.testing.assert_allclose(c['gvar_gammas'], np.linspace(0, .01, 5))
    assert len(tasks(c, 'pilot')) == 135
    main = tasks(c, 'main', {'GSE-VAR':{'hidden':8}})
    assert len(main) == 180
    assert len({t['id'] for t in main}) == len(main)
    assert not set(c['pilot_seeds']) & set(range(42, 47))
    for t in main:
        if t['method'] == 'GVAR':
            assert t['training_seed'] == 42+t['profile']['grid_i']+t['profile']['grid_j']


def test_published_replicate_is_the_author_generator_to_roundoff():
    raw_folder = ROOT/'data/lotka_volterra/published'
    if not raw_folder.exists():
        pytest.skip('Run the documented prepare command to fetch verified publication data.')
    saved = published(ROOT, download=False)[0]
    generated = generate(42)
    np.testing.assert_allclose(saved['train'], generated['train'], atol=2e-12, rtol=2e-12)
    np.testing.assert_array_equal(saved['truth'], generated['truth'])
    np.testing.assert_array_equal(saved['truth_signed'], generated['truth_signed'])
    assert np.sum(saved['truth_signed'][:10, 10:] == -1) == 20
    assert np.sum(saved['truth_signed'][10:, :10] == 1) == 20


def test_pilot_selection_requires_all_independent_series_and_freezes_one_setting():
    c = configuration('smoke')
    rows = []
    for t in tasks(c, 'pilot'):
        rows.append(dict(t, edges=5, mean_sign_ba=.7+.1*t['profile'].get('budget', 0)))
    choice = select(c, rows)
    assert choice['GSE-VAR']['budget'] == 3
    assert {t['profile']['hidden'] for t in tasks(c, 'main', choice) if t['method'] == 'GSE-VAR'} == {4}
    with pytest.raises(ValueError, match='Incomplete'):
        select(c, rows[:-1])


def test_trgc_matches_author_guards_including_diagonal_count():
    # Nonself edges alone do not bypass the author's total-edge <=p guard.
    f = np.array([[.1, .8, .5], [.2, .1, .7], [.9, .4, .1]])
    b = np.array([[.1, .3, .9], [.8, .1, .4], [.5, .7, .1]])
    graph, agreement, best = reference_trgc(f, b)
    assert best == int(np.argmax(agreement))
    assert agreement[0] == 0 and agreement[-1] == 0
    assert graph.sum() > 3 and graph.sum() < 9
    constant, scores, index = reference_trgc(np.ones((3, 3)), np.ones((3, 3)))
    assert np.all(scores == 0) and index == 0 and constant.all()


def test_linear_reference_metrics_match_published_paper():
    if not (ROOT/'data/lotka_volterra/published').exists():
        pytest.skip('Publication data were not downloaded.')
    from experiments.reference.gvar.linvar import LinVAR
    from experiments.shared import graph_metrics, sign_metrics
    metrics = []
    for case in published(ROOT, download=False):
        model = LinVAR(case['train'], K=1)
        pvalues, graph, _ = model.infer_causal_structure(adjust=True, signed=True)
        result = graph_metrics(case['truth'], graph, -pvalues)
        result['accuracy'] = (result['tp']+result['tn'])/380
        signs, _ = sign_metrics(case['truth_signed'], {'coefficients':model.model_results.coefs[None]}, graph)
        metrics.append(dict(result, **signs))
    for field, (mean, sd) in source_manifest()['paper_table3']['Linear-VAR'].items():
        values = [r[field] for r in metrics]
        assert np.mean(values) == pytest.approx(mean, abs=.00051)
        assert np.std(values) == pytest.approx(sd, abs=.00051)


def test_completed_tasks_require_matching_spec_and_unchanged_evaluation(tmp_path):
    task = tasks(configuration('smoke'), 'pilot')[0]
    folder = tmp_path/'jobs'/task['id']
    folder.mkdir(parents=True)
    data = b'evaluation'
    (folder/'evaluation.npz').write_bytes(data)
    write_json(folder/'metrics.json', dict(task, evaluation_sha256=hashlib.sha256(data).hexdigest()))
    assert completed(tmp_path, task)['id'] == task['id']
    (folder/'evaluation.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        completed(tmp_path, task)


def test_notebook_is_safe_to_open_and_uses_canonical_runner():
    notebook = json.loads((ROOT/'experiments/lotka_volterra/experiment.ipynb').read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
            assert cell['outputs'] == [] and cell['execution_count'] is None
    source = ''.join(''.join(cell['source']) for cell in notebook['cells'])
    assert 'TRAIN = False' in source and 'experiments.lotka_volterra.run' in source
