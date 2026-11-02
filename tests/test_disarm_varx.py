"""Joint Bernoulli optimization across endogenous and exogenous branches."""
from itertools import product
import numpy as np
import torch
from gse_var import GSEVARX, GSEVARXTrainingConfig, fit_gse_varx
from gse_var.training import _adam_disarm_step
from gse_var.disarm import _disarm_gradient
from gse_var.varx import _masked_losses


def test_varx_estimator_matches_exact_expectation_for_both_branches():
    torch.manual_seed(9)
    model = GSEVARX(1, 1, 1, 0, True, 2, gate_probability_init=.3,
                    exogenous_gate_probability_init=.7).double()
    with torch.no_grad():
        for net in (model.endogenous_coeff_net, model.exogenous_coeff_net):
            net.weights[-1].normal_(std=.2)
    config = GSEVARXTrainingConfig(1, 2, lambda_smooth=.07, lambda_smooth_exog=.13,
                                 lambda_jacobian=.02, lambda_jacobian_exog=.03,
                                 smoothness_mode='relative', verbose=0)
    y, x = torch.randn(3, 1, 1, dtype=torch.float64), torch.randn(3, 1, 1, dtype=torch.float64)
    targets, times = torch.randn(3, 1, dtype=torch.float64), torch.arange(3)
    logits = [model.endogenous_gate_logits, model.exogenous_gate_logits]
    ps = [v.sigmoid().squeeze() for v in logits]
    objective = torch.zeros((), dtype=torch.float64)
    for bits in product([0., 1.], repeat=2):
        masks = {n: torch.full_like(logits[i], bits[i]) for i, n in enumerate(('endogenous','exogenous'))}
        loss = sum(_masked_losses(model, y, x, targets, times, masks, config)[0].values())
        weight = (ps[0] if bits[0] else 1-ps[0]) * (ps[1] if bits[1] else 1-ps[1])
        objective = objective + weight * loss
    exact = torch.autograd.grad(objective, logits)
    intervals=[]
    for p in ps:
        bounds=sorted(set([0.,float(p.detach()),1-float(p.detach()),1.]))
        intervals.append([((a+b)/2,b-a) for a,b in zip(bounds,bounds[1:])])
    mean=[torch.zeros_like(v) for v in logits]
    for cells in product(*intervals):
        first={};second={}
        for i,n in enumerate(('endogenous','exogenous')):
            first[n]=torch.full_like(logits[i],cells[i][0])<ps[i]
            second[n]=torch.full_like(logits[i],1-cells[i][0])<ps[i]
        f=_masked_losses(model,y,x,targets,times,first,config)[1]
        s=_masked_losses(model,y,x,targets,times,second,config)[1]
        for i,n in enumerate(('endogenous','exogenous')):
            mean[i] += cells[0][1]*cells[1][1]*_disarm_gradient(logits[i],first[n],second[n],f,s)
    for a,b in zip(mean,exact):torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)


def test_separate_costs_use_two_evaluations_before_one_joint_update():
    model=GSEVARX(1,1,1,0,True,2,gate_probability_init=.6,exogenous_gate_probability_init=.7)
    cfg=GSEVARXTrainingConfig(1,2,learning_rate=.1,coefficient_weight_decay=0,verbose=0)
    optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    initial={n:p.clone() for n,p in model.gate_logits_by_branch.items()}
    seen=[]
    def evaluate(masks):
        seen.append({n:p.clone() for n,p in model.gate_logits_by_branch.items()})
        return {k:model.intercept.sum()*0 for k in ('mse','smooth','jacobian')}, torch.zeros(1)
    _adam_disarm_step(model,evaluate,{'endogenous':.2,'exogenous':.8},cfg,optimizer,torch.Generator().manual_seed(1))
    assert len(seen)==2
    for n,p in model.gate_logits_by_branch.items():
        for state in seen:torch.testing.assert_close(state[n],initial[n])
        probability=initial[n].sigmoid();cost=.2 if n=='endogenous' else .8
        torch.testing.assert_close(p,initial[n]-.1*cost*probability*(1-probability))


def test_both_logits_learn_reproducibly_with_binary_masks():
    rng=np.random.default_rng(9);y,x=rng.normal(size=(15,2)),rng.normal(size=(15,3))
    config=GSEVARXTrainingConfig(2,3,max_epochs=3,batch_size=5,gate_probability_init=.5,
                              lambda_l0=.03,lambda_l0_exog=.1,verbose=0,device='cpu')
    results=[fit_gse_varx(y,x,config) for _ in range(2)]
    assert results[0].history==results[1].history
    m=results[0].model
    for logits,gate in ((m.endogenous_gate_logits,m.endogenous_gate),(m.exogenous_gate_logits,m.exogenous_gate)):
        assert not torch.equal(logits,torch.zeros_like(logits))
        assert torch.equal(gate.bool(),logits>=0)
        assert set(gate.unique().tolist()) <= {0.,1.}
    assert 'endogenous_gate' not in dict(m.named_parameters())
    assert 'exogenous_gate' not in dict(m.named_parameters())
