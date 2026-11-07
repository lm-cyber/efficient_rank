"""Check spectral definitions, selection controls, exact slicing and graph training."""
import copy
import json

from runtime import RUN, engine, save_json, setup
from tiny_model import allocate_aligned, base_spec, build, cost, install, linear_cost, prune
from subsets import entropy_rank, select
from compress import allocate
from models import to_gpu
from capture_step import CapturedStep
import numpy as np
import pandas as pd
import torch


def main():
    setup();install();checks=[]
    eigen=torch.tensor([[0.,0.,0.],[1.,1.,1.],[0.,0.,5.]],device='cuda',dtype=torch.float64)
    assert torch.allclose(entropy_rank(eigen),torch.tensor([0.,3.,1.],device='cuda',dtype=torch.float64))
    checks.append('entropy rank zero, full, rank-one cases')
    rng=np.random.default_rng(17)
    values=rng.normal(size=(4,5,7))
    centered=values-values.mean(axis=1,keepdims=True)
    eig=np.linalg.eigvalsh(centered@centered.transpose(0,2,1)).clip(0)
    from measures import erank
    computed=entropy_rank(torch.as_tensor(eig,device='cuda')).cpu().numpy()
    expected=np.array([erank(np.linalg.svd(x,compute_uv=False)**2) for x in centered])
    assert np.allclose(computed,expected,atol=1e-8)
    checks.append('per-image covariance entropy agrees with direct SVD')
    # Equal class balance, disjoint extremes, deterministic random indices.
    frame=pd.DataFrame({'id':range(2500),'label':np.repeat(np.arange(100),25),
                        'score_cov':np.tile(np.arange(25),100)})
    chosen,_=select(frame,100)
    assert not set(chosen['low']).intersection(chosen['high'])
    assert chosen==select(frame,100)[0]
    checks.append('class-balanced fixed low/random/high selection, no overlap of extremes')
    engine.seed_all(999)
    model=to_gpu(build(base_spec())).eval()
    assert cost(model)['params']>10_000_000
    x=torch.randn(8,3,64,64,device='cuda').contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        reference=model(x)
    orders=[list(range(w)) for w in model.spec['widths']]
    full,_=prune(model,model.spec['widths'],orders)
    with torch.inference_mode():
        assert torch.allclose(full.eval()(x),reference,atol=1e-5,rtol=1e-5)
    checks.append('full-width ResNet-18 slicing preserves outputs')
    for budget in (.75,.6):
        widths=allocate(model,np.array(model.spec['widths']),cost(model)['macs']*budget)
        changed,masks=prune(model,widths,orders)
        full_widths,coefficients,fixed=linear_cost(model)
        assert cost(changed)['macs']==int(fixed+coefficients@widths)
        masked=copy.deepcopy(model).eval();handles=[]
        for block,mask in zip(masked.blocks,masks):
            keep=torch.zeros(block.conv2.in_channels,device='cuda');keep[mask]=1
            handles.append(block.conv2.register_forward_pre_hook(lambda mod,inputs,k=keep:(inputs[0]*k[None,:,None,None],)))
        with torch.inference_mode():
            assert torch.allclose(changed.eval()(x),masked(x),atol=2e-4,rtol=2e-4)
        for h in handles:h.remove()
        del changed,masked
    checks.append('two budgets: physical slicing equals dense channel masking; actual MACs match exact allocation')
    for budget in (.75,.6):
        target=cost(model)['macs']*budget
        a=allocate_aligned(model,np.array(model.spec['widths']),target)
        b=allocate_aligned(model,np.array(model.spec['widths'])*np.linspace(.6,1.2,8),target)
        _,coefficients,fixed=linear_cost(model)
        assert all(w%8==0 for w in a+b)
        assert coefficients@a==coefficients@b
        assert fixed+coefficients@a<=target
    checks.append('aligned-8 profiles: exact common attainable MAC budget, all widths multiples of 8')
    model.train();other=copy.deepcopy(model).train()
    a=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=.0005,foreach=True)
    b=torch.optim.SGD(other.parameters(),lr=.01,momentum=.9,weight_decay=.0005,foreach=True)
    y=torch.arange(8,device='cuda')
    step=CapturedStep(other,b,x,y)
    for _ in range(2):
        a.zero_grad(set_to_none=True)
        with engine.amp():loss=torch.nn.functional.cross_entropy(model(x),y)
        loss.backward();a.step();step(x,y)
    diffs={name:float((value.float()-other.state_dict()[name].float()).abs().max()) for name,value in model.state_dict().items()}
    assert max(diffs.values())==0.,max(diffs.values())
    checks.append('two full ResNet-18 optimizer steps: CUDA Graphs and eager states bitwise identical, including all BN buffers')
    save_json(RUN/'verification.json',{'passed':checks,'max_training_state_difference':max(diffs.values()),'dense_cost':cost(model)})
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':
    main()
