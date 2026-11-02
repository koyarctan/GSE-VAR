import inspect
import torch
import gse_var
from gse_var import GSEVAR,GSEVARX

def test_single_canonical_public_training_api():
    for n in ('GSEVARDisARM','GSEVARDisARMTrainingConfig','fit_gse_var_disarm'):assert not hasattr(gse_var,n)
    for cls in (GSEVAR,GSEVARX):assert not {'use_causal_gate','gate_init','fit_intercept'}&set(inspect.signature(cls).parameters)

def test_inference_is_fixed_binary_and_does_not_consume_rng():
    m=GSEVAR(2,2,3,gate_probability_init=.5)
    with torch.no_grad():m.gate_logits[0,0,1]=-.1;m.update_causal_gate_()
    x=torch.randn(4,2,2);state=torch.random.get_rng_state().clone()
    for mode in (True,False):
        m.train(mode);torch.testing.assert_close(m(x)[0],m(x)[0],rtol=0,atol=0)
    assert torch.equal(state,torch.random.get_rng_state())
    assert set(m.causal_gate.unique().tolist())<={0.,1.}
    assert 'causal_gate' not in dict(m.named_parameters())
