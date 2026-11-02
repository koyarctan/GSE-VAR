from pathlib import Path
import json
import numpy as np
import torch
from gse_var import GSEVARTrainingConfig,fit_gse_var

def test_binary_var_preserves_reference_weights_coefficients_and_history():
    data=np.random.default_rng(751).normal(size=(24,3))
    cfg=GSEVARTrainingConfig(order=2,hidden_layer_size=3,max_epochs=4,batch_size=7,
        learning_rate=.003,gate_learning_rate=.03,lambda_l0=.02,gate_probability_init=.5,
        coefficient_weight_decay=.03,lambda_smooth=.02,lambda_jacobian=.01,
        shuffle=True,seed=17,verbose=0,device='cpu')
    r=fit_gse_var(data,cfg)
    folder=Path(__file__).with_name('fixtures')
    with np.load(folder/'binary_var_reference.npz',allow_pickle=False) as old:
        actual=dict(coefficients=r.coeffs,graph=r.causal_graph,**{k:v.detach().numpy() for k,v in r.model.state_dict().items()})
        assert set(actual)==set(old.files)
        for k,v in actual.items():np.testing.assert_allclose(v,old[k],rtol=2e-6,atol=2e-7)
    history=json.loads((folder/'binary_var_reference_history.json').read_text())
    for k,v in r.history.items():np.testing.assert_allclose(v,history[k],rtol=2e-6,atol=2e-7)
