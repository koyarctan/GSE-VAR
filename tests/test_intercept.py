import numpy as np
import pytest
import torch
from gse_var import GSEVAR,GSEVARX,GSEVARTrainingConfig,GSEVARXTrainingConfig,fit_gse_var,fit_gse_varx

@pytest.mark.parametrize('varx',[False,True])
def test_intercept_changes_predictions_only(varx):
    m=GSEVARX(2,3,2,1,True,3) if varx else GSEVAR(2,2,3)
    inputs=[torch.randn(4,2,2)]+([torch.randn(4,2,3)] if varx else [])
    before=m(*inputs)
    with torch.no_grad():m.intercept.copy_(torch.tensor([1.,-2.]))
    after=m(*inputs)
    torch.testing.assert_close(after[0]-before[0],m.intercept.expand(4,-1))
    for a,b in zip(after[1:],before[1:]):torch.testing.assert_close(a,b,rtol=0,atol=0)

@pytest.mark.parametrize('varx',[False,True])
def test_intercept_learns_without_weight_decay(varx):
    y=np.full((12,2),.7,dtype='float32')
    cls=GSEVARXTrainingConfig if varx else GSEVARTrainingConfig
    cfg=cls(2,2,max_epochs=1,batch_size=20,learning_rate=.01,coefficient_weight_decay=2.,verbose=0,device='cpu')
    m=GSEVARX(2,1,2,0,True,2) if varx else GSEVAR(2,2,2)
    with torch.no_grad():
        for p in m.parameters():p.zero_()
    r=fit_gse_varx(y,np.zeros((12,1)),cfg,model=m) if varx else fit_gse_var(y,cfg,model=m)
    np.testing.assert_allclose(r.intercept,.01,atol=1e-6)
