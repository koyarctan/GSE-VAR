"""Validation selection and training-only transformations for empirical data."""
import numpy as np
import pytest
from experiments.favorita.study import choose_candidate
from experiments.favorita.data import make_cases


def test_minimum_mse_is_selected_even_when_a_sparser_candidate_is_close():
    rows=[dict(mse=.2,se=.1,epoch=100,candidate=0,edges=10),
          dict(mse=.21,se=.1,epoch=100,candidate=1,edges=0)]
    assert choose_candidate(rows)['candidate']==0
    rows.append(dict(mse=.2,se=.1,epoch=20,candidate=2,edges=20))
    assert choose_candidate(rows)['candidate']==2


def test_future_sales_do_not_change_training_transforms_and_missing_days_break_windows():
    dates=np.arange(np.datetime64('2016-01-01'),np.datetime64('2016-05-01')).astype(str)
    raw=np.random.default_rng(13).uniform(1,50,(len(dates),2))
    raw[20]=np.nan
    panel=dict(dates=dates,raw=raw)
    config=dict(order=3,train_end='2016-02-29',validation_end='2016-03-31',end='2016-04-30')
    segments,cases,processing=make_cases(panel,config)
    changed=raw.copy();changed[dates>config['train_end']]*=100
    other_segments,_,other=make_cases(dict(panel,raw=changed),config)
    np.testing.assert_array_equal(processing['weekday_mean'],other['weekday_mean'])
    np.testing.assert_array_equal(processing['scale'],other['scale'])
    for a,b in zip(segments,other_segments):np.testing.assert_array_equal(a,b)
    assert len(segments)==2
    assert all(np.isfinite(s).all() for s in segments)
    for case in cases.values():
        assert all(processing['known'][t-3:t+1].all() for t in case['indices'])


@pytest.mark.parametrize('method',['GSE-VAR','Linear-SGL'])
def test_favorita_pipeline_selects_restores_and_exports_on_an_independent_project(tmp_path,monkeypatch,method):
    import json
    from experiments.favorita import study
    from experiments.favorita.run import configuration
    from experiments.shared import read_npz
    dates=np.arange(np.datetime64('2016-01-01'),np.datetime64('2016-05-01')).astype(str)
    panel=dict(dates=dates,raw=np.random.default_rng(81).uniform(1,50,(len(dates),2)),
               names=np.array(['a','b']),labels=np.array(['A','B']))
    config=configuration()
    config.update(order=2,hidden=2,epochs=2,check_every=1,mode='smoke',device='cpu',
                  train_end='2016-02-29',validation_end='2016-03-31',end='2016-04-30')
    monkeypatch.setattr(study,'prepare_data',lambda project,cfg:(panel,{'fixture':'synthetic'},tmp_path/'prepared'))
    profiles=study.candidate_profiles
    monkeypatch.setattr(study,'candidate_profiles',lambda cfg,method,p:profiles(cfg,method,p)[:1])
    output=study.execute(tmp_path,config,method,device='cpu')
    assert output.name.startswith('smoke_')
    assert json.loads((output/'completed.json').read_text())['seeds']==[42]
    selection=json.loads((output/'selection.json').read_text())
    saved=read_npz(output/'evaluation/seed_42.npz')
    assert int(saved['selected_epoch'])==selection['epoch']
    assert saved['predictions'].shape==saved['observed'].shape==(30,2)
    assert np.isfinite(saved['predictions']).all()
    assert saved['gate'].shape==(2,2,2)
