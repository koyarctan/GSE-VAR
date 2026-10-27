"""Remote notebook protocols, leakage boundaries, persistence and score semantics."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.paper_experiments.common import (score_metrics, linear_fit, lagged, bh_select,
    write_npz, read_npz, gse_profile, Run, fit_gse, restore_model, snapshot, digest)
from experiments.paper_experiments.marketing import load_data, MarketingExperiment, inverse
from experiments.paper_experiments.synthetic import default_config, SyntheticExperiment

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('prefix',['11_','12_','13_'])
def test_current_notebooks_are_remote_and_outputs_are_cleared(prefix):
    files=[p for p in (ROOT/'experiments').glob(prefix+'*.ipynb') if 'lambda' not in p.name]
    assert len(files)==1
    notebook=json.loads(files[0].read_text(encoding='utf-8'))
    text='\n'.join(''.join(c['source']) for c in notebook['cells'])
    assert 'google.colab' not in text and 'drive.mount' not in text
    assert 'full_393ba5' not in text and 'full_cabd5' not in text
    assert 'Citi Bike' not in text
    assert 'DisARM' in text and 'SGL' in text
    assert 'get_ipython' in text and 'selection = experiment.tune()' in text
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            ast.parse(''.join(cell['source']))
            assert cell['outputs']==[] and cell['execution_count'] is None


@pytest.mark.parametrize('truth,score,ap,auc',[
    ([1,1,0,0],[4,3,2,1],1.,1.),
    ([1,1,0,0],[1,1,1,1],.5,.5),
    ([1,1,0,0],[1,2,3,4],5/12,0.),
])
def test_average_precision_handles_ties_without_invented_endpoints(truth,score,ap,auc):
    result=score_metrics(truth,score)
    assert result['auprc']==pytest.approx(ap)
    assert result['auroc']==pytest.approx(auc)


def test_bh_and_linear_ols_have_expected_orientation():
    np.testing.assert_array_equal(bh_select([.001,.02,.9,.8]),[True,True,False,False])
    rng=np.random.default_rng(17)
    z=rng.normal(size=(500,3))
    for t in range(1,len(z)):
        z[t,1]+=.8*z[t-1,0]
    x,y=lagged(z,1)
    fit=linear_fit(z,x,1)
    assert fit['pvalues'][1,0]<1e-10
    assert fit['coefficients'][0,0,1,0]>.6
    np.testing.assert_allclose(fit['predictions'],np.einsum('bkj,kij->bi',x,fit['coefficients'][0])+fit['intercept'])


def test_source_panel_has_only_the_available_seven_queries_and_192_months():
    data=load_data(ROOT/'data')
    assert data['raw'].shape==(192,8)
    assert data['months'][0]=='2004-01' and data['months'][-1]=='2019-12'
    assert 'Japan itinerary' not in data['names']
    assert data['raw'][0,0]==52463 and data['raw'][-1,0]==144498
    assert len(data['manifest']['files'])==8


def test_marketing_transform_does_not_estimate_scaling_from_future(tmp_path):
    source=load_data(ROOT/'data')
    def make(path,raw):
        item=object.__new__(MarketingExperiment)
        item.config={'order':3}
        item.data=dict(source,raw=raw)
        item.run=SimpleNamespace(path=path)
        return item.case(84,108)
    first=make(tmp_path/'a',source['raw'].copy())
    changed=source['raw'].copy()
    changed[96:]*=1.5
    second=make(tmp_path/'b',changed)
    for field in ('train','mean','scale'):
        np.testing.assert_array_equal(first[field],second[field])
    assert not np.array_equal(first['y_eval'],second['y_eval'])
    recovered,_=inverse(first['y_eval'],first)
    np.testing.assert_allclose(recovered,first['truth'],rtol=2e-6,atol=.02)


def test_synthetic_conditions_share_trajectory_but_not_normalization(tmp_path):
    config=default_config('linear_var','smoke','cpu');config['output_root']=str(tmp_path)
    item=SyntheticExperiment(ROOT,'linear_var',config)
    a,b=item.case(48,42),item.case(64,42)
    np.testing.assert_array_equal(a['raw'],b['raw'])
    np.testing.assert_allclose(a['mean'],a['raw'][:48].mean(0))
    assert not np.array_equal(a['scale'],b['scale'])
    assert not set(config['pilot_seeds']) & set(config['main_seeds'])
    np.testing.assert_array_equal(np.sign(a['true_standardized']),a['truth_signed'])


@pytest.mark.parametrize('method',['GSE-SGL','GSE-DisARM'])
def test_checkpoints_restore_exact_mask_predictions_and_training_only_monitor(tmp_path,method):
    config=dict(mode='smoke',device='cpu',check_every=1,output_root=str(tmp_path))
    run=Run(ROOT,'test',config,{'fixture':'white noise'})
    train=np.random.default_rng(7).normal(size=(18,3)).astype('float32')
    x,y=lagged(train,1)
    case=dict(id='test',train=train,x_eval=x+2,y_eval=y+3)
    profile=gse_profile(method,1,3,3,.1,2)
    arrays=fit_gse(run,case,method,profile,42,score=False)
    ident=digest(json.loads(str(arrays['fit_spec'])))
    checkpoint=read_npz(run.path/'checkpoints'/ident/'000002.npz')
    model=restore_model(method,profile,3,checkpoint)
    current=snapshot(model,case['x_eval'],jacobian=True)
    for key in ('predictions','coefficients','jacobian','gate'):
        np.testing.assert_array_equal(current[key],arrays[key])
    training=snapshot(model,x)
    metric=float(np.mean((training['predictions']-y)**2))
    fields=arrays['checkpoint_fields'].tolist()
    assert arrays['checkpoint_metrics'][-1,fields.index('mse')]==metric
    if method=='GSE-DisARM':
        np.testing.assert_array_equal(arrays['gate'],arrays['state__gate_logits']>=0)
    for path in run.path.rglob('*.npz'):
        assert not any(v.dtype.hasobject for v in read_npz(path).values())


def test_run_controls_do_not_change_result_identity(tmp_path):
    config=dict(mode='smoke',device='cpu',check_every=1,output_root=str(tmp_path))
    a=Run(ROOT,'test',config,{})
    b=Run(ROOT,'test',dict(config,run_experiments=False,retry_errors=True),{})
    assert a.path==b.path


def test_object_arrays_are_never_serialized(tmp_path):
    with pytest.raises(TypeError,match='pickle'):
        write_npz(tmp_path/'bad.npz',unsafe=np.array([{}],dtype=object))


def test_analysis_bundle_includes_all_evaluation_arrays_but_not_large_checkpoints(tmp_path):
    import zipfile
    from experiments.paper_experiments.common import write_json,write_csv
    from experiments.paper_experiments.reports import make_analysis_bundle,read_results
    write_json(tmp_path/'manifest.json',{'experiment':'fixture'})
    write_json(tmp_path/'source_snapshot.json',{})
    write_json(tmp_path/'completion.json',{'complete':1,'expected':1,'failed':0})
    write_json(tmp_path/'main_index.json',[{'method':'GSE-SGL','dataset':'datasets/a.npz','result':'evaluations/a.npz'}])
    write_csv(tmp_path/'main_metrics.csv',[{'method':'GSE-SGL','rmse':1.}])
    write_npz(tmp_path/'datasets/a.npz',x_eval=np.zeros((2,1,2)))
    write_npz(tmp_path/'evaluations/a.npz',predictions=np.zeros((2,2)),coefficients=np.ones((2,1,2,2)))
    write_npz(tmp_path/'checkpoints/fit/000100.npz',weight=np.zeros(3))
    archive=make_analysis_bundle(tmp_path)
    with zipfile.ZipFile(archive) as z:
        assert 'evaluations/a.npz' in z.namelist()
        assert 'source_snapshot.json' in z.namelist()
        assert not any(name.startswith('checkpoints/') for name in z.namelist())
    archive=make_analysis_bundle(tmp_path,include_checkpoints=True)
    with zipfile.ZipFile(archive) as z:
        assert 'checkpoints/fit/000100.npz' in z.namelist()


def test_full_linear_settings_describe_the_actual_fixed_author_generator():
    config=default_config('linear_var')
    assert config['noise_sd']==.4 and config['p']==4 and config['order']==1
    assert config['lengths']==[250,500] and config['epochs']==10000
    assert config['main_seeds']==list(range(42,52))
