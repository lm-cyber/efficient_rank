"""Checkpoint-compatible SegFormer, physical Mix-FFN slicing, exact MAC allocation."""
import copy
import math

from common import CFG, WEIGHTS
import numpy as np
import torch
from torch import nn
from transformers import SegformerConfig,SegformerForSemanticSegmentation
from transformers.models.segformer.modeling_segformer import SegformerEfficientSelfAttention


def blocks(model):
    return [block for stage in model.segformer.encoder.block for block in stage]


def widths(model):
    return [block.mlp.dense1.out_features for block in blocks(model)]


@torch.no_grad()
def slice_ffn(ffn,keep):
    a,d,b=ffn.dense1,ffn.dwconv.dwconv,ffn.dense2
    keep=torch.as_tensor(keep,dtype=torch.long,device=a.weight.device)
    width=len(keep);device=a.weight.device;dtype=a.weight.dtype
    first=nn.Linear(a.in_features,width,bias=a.bias is not None,device=device,dtype=dtype)
    depthwise=nn.Conv2d(width,width,d.kernel_size,d.stride,d.padding,groups=width,
                        bias=d.bias is not None,device=device,dtype=dtype)
    second=nn.Linear(width,b.out_features,bias=b.bias is not None,device=device,dtype=dtype)
    first.weight.copy_(a.weight[keep]);depthwise.weight.copy_(d.weight[keep]);second.weight.copy_(b.weight[:,keep])
    if a.bias is not None:first.bias.copy_(a.bias[keep])
    if d.bias is not None:depthwise.bias.copy_(d.bias[keep])
    if b.bias is not None:second.bias.copy_(b.bias)
    ffn.dense1,ffn.dwconv.dwconv,ffn.dense2=first,depthwise,second


def build(pruned_widths=None,pretrained=False):
    config=SegformerConfig.from_json_file(WEIGHTS/'config.json')
    model=SegformerForSemanticSegmentation(config)
    if pretrained:
        state=torch.load(WEIGHTS/'pytorch_model.bin',map_location='cpu',weights_only=True)
        model.load_state_dict(state,strict=True)
    if pruned_widths is not None:
        assert len(pruned_widths)==len(blocks(model))
        for block,width in zip(blocks(model),pruned_widths):slice_ffn(block.mlp,list(range(width)))
    return model


def load(path,device='cuda'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=build(checkpoint['widths']);model.load_state_dict(checkpoint['state_dict'],strict=True)
    return model.to(device).eval()


@torch.no_grad()
def prune(model,pruned_widths,orders):
    result=copy.deepcopy(model).cpu();masks=[]
    for block,width,order in zip(blocks(result),pruned_widths,orders):
        keep=sorted(order[:int(width)]);masks.append(keep);slice_ffn(block.mlp,keep)
    return result,masks


@torch.inference_mode()
def cost(model):
    counts={};handles=[]
    def hook(name):
        def count(module,inputs,output):
            if isinstance(module,nn.Conv2d):value=output.numel()*module.in_channels*math.prod(module.kernel_size)//module.groups
            elif isinstance(module,nn.Linear):value=output.numel()*module.in_features
            else:
                x,h,w=inputs[:3];n=x.shape[1];k=(h//module.sr_ratio)*(w//module.sr_ratio)
                value=2*x.shape[0]*n*k*module.all_head_size
            counts[name]=int(value)
        return count
    for name,module in model.named_modules():
        if isinstance(module,(nn.Conv2d,nn.Linear,SegformerEfficientSelfAttention)):
            handles.append(module.register_forward_hook(hook(name)))
    training=model.training;model.eval()
    try:model(torch.zeros(1,3,CFG['image_size'],CFG['image_size'],device=next(model.parameters()).device))
    finally:
        for handle in handles:handle.remove()
        model.train(training)
    full=np.array(widths(model));coefficients=[]
    for stage,stage_blocks in enumerate(model.segformer.encoder.block):
        for index,block in enumerate(stage_blocks):
            prefix=f'segformer.encoder.block.{stage}.{index}.mlp.'
            n=block.mlp.dense1.out_features
            value=sum(counts[prefix+name] for name in ['dense1','dwconv.dwconv','dense2'])
            assert value%n==0
            coefficients.append(value//n)
    macs=sum(counts.values());fixed=int(macs-np.array(coefficients)@full)
    return {'params':sum(p.numel() for p in model.parameters()),'macs':macs,'flops':2*macs,
            'widths':full.tolist(),'coefficients':coefficients,'fixed_macs':fixed,
            'prunable_macs_pct':100*(macs-fixed)/macs,'operation_scope':'Conv/Linear and both attention matmuls'}


def target_for_budget(measured,budget):
    full=np.array(measured['widths']);coeff=np.array(measured['coefficients']);fixed=measured['fixed_macs']
    q=CFG['width_multiple'];maximum=full//q
    desired=measured['macs']*budget
    assert fixed+coeff@(np.ones(len(full),dtype=int)*q)<=desired, 'Infeasible global budget for MLP-only compression'
    ratio=(desired-fixed)/(coeff@full)
    ideal=maximum*ratio;units=np.clip(np.floor(ideal).astype(int),1,maximum)
    while True:
        available=[i for i in range(len(full)) if units[i]<maximum[i] and fixed+coeff@(units*q)+coeff[i]*q<=desired]
        if not available:break
        chosen=max(available,key=lambda i:((ideal[i]-units[i])/maximum[i],-i))
        units[chosen]+=1
    return int(fixed+coeff@(units*q))


def allocate(measured,profile,target):
    """Exact common MAC budget, minimizing relative squared width deviations.

    B1 has two blocks with equal linear MAC coefficients per stage. Enumerating
    the four stage totals avoids a large channel-level knapsack table.
    """
    full=np.array(measured['widths']);coeff=np.array(measured['coefficients']);fixed=measured['fixed_macs']
    q=CFG['width_multiple'];capacity=full//q;weights=coeff*q
    assert len(full)==8 and np.all(full%q==0)
    assert np.all(weights[::2]==weights[1::2])
    assert np.all(capacity[::2]==capacity[1::2])
    profile=np.maximum(np.asarray(profile,dtype=float)/q,1e-8)
    lo,hi=0.,1e6
    for _ in range(70):
        mid=(lo+hi)/2
        if fixed+weights@np.clip(profile*mid,1,capacity)<=target:lo=mid
        else:hi=mid
    ideal=np.clip(profile*lo,1,capacity);options=[]
    for stage in range(4):
        i=2*stage;cap=int(capacity[i]);entries=[]
        for total in range(2,2*cap+1):
            lower=max(1,total-cap);upper=min(cap,total-1)
            first=int(np.clip(np.rint((total+ideal[i]-ideal[i+1])/2),lower,upper));second=total-first
            penalty=((first-ideal[i])/cap)**2+((second-ideal[i+1])/cap)**2
            entries.append((int(total*weights[i]),float(penalty),(first*q,second*q)))
        options.append(entries)
    right={}
    for c,pc,wc in options[2]:
        for d,pd,wd in options[3]:
            key=c+d;value=(pc+pd,wc+wd)
            if key not in right or value[0]<right[key][0]:right[key]=value
    best=None;feasible=0
    for a,pa,wa in options[0]:
        for b,pb,wb in options[1]:
            remaining=target-fixed-a-b
            if remaining not in right:continue
            feasible+=1;penalty,wr=right[remaining]
            candidate=(pa+pb+penalty,wa+wb+wr)
            if best is None or candidate<best:best=candidate
    assert best is not None,'Reference budget must be attainable'
    result=list(best[1]);assert fixed+coeff@result==target
    return result,{'feasible_stage_combinations':feasible,'profile_penalty':best[0]}
