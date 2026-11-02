from dataclasses import fields,replace
import numpy as np
import pytest
from gse_var import GSEVARTrainingConfig,GSEVARXTrainingConfig,fit_gse_varx

@pytest.mark.parametrize('cls',[GSEVARTrainingConfig,GSEVARXTrainingConfig])
def test_configuration_contains_only_binary_structure_learning(cls):
    names={f.name for f in fields(cls)}
    assert {'lambda_l0','gate_probability_init'}<=names
    assert not {'regularizer','sparse_l1_lambda','sparse_group_lambda','lambda_ngc','gate_init'}&names

@pytest.mark.parametrize('field,value',[('lambda_l0_exog',-1),('lambda_l0_exog',float('nan')),('exogenous_gate_probability_init',0),('exogenous_gate_probability_init',1)])
def test_invalid_exogenous_settings_are_rejected(field,value):
    cfg=replace(GSEVARXTrainingConfig(1,2,max_epochs=1,verbose=0),**{field:value})
    with pytest.raises(ValueError):fit_gse_varx(np.zeros((5,1)),np.zeros((5,1)),cfg)
