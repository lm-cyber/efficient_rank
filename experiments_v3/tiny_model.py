"""ResNet-18 for 64px inputs, plus generic exact internal-channel compression."""
import copy
import math

from runtime import engine
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import models as v2_models
import compress as v2_compress

OLD_BUILD=v2_models.build
OLD_PRUNE=v2_compress.prune
OLD_LINEAR_COST=v2_compress.linear_cost
OUTER=[64,64,128,128,256,256,512,512]


class Block(nn.Module):
    def __init__(self,cin,cout,width,stride):
        super().__init__()
        self.conv1=nn.Conv2d(cin,width,3,stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(width)
        self.conv2=nn.Conv2d(width,cout,3,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(cout)
        self.downsample=nn.Sequential(nn.Conv2d(cin,cout,1,stride,bias=False),nn.BatchNorm2d(cout)) if cin!=cout or stride!=1 else nn.Identity()

    def forward(self,x):
        residual=self.downsample(x)
        x=F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(x))+residual)


class ResNet18(nn.Module):
    def __init__(self,spec):
        super().__init__()
        self.spec=copy.deepcopy(spec)
        self.spec['widths']=list(spec.get('widths',OUTER))
        self.conv=nn.Conv2d(3,64,3,padding=1,bias=False)
        self.bn=nn.BatchNorm2d(64)
        self.blocks=nn.ModuleList()
        cin=64
        for i,width in enumerate(self.spec['widths']):
            self.blocks.append(Block(cin,OUTER[i],width,2 if i in (2,4,6) else 1))
            cin=OUTER[i]
        self.fc=nn.Linear(512,spec['classes'])
        for module in self.modules():
            if isinstance(module,nn.Conv2d):
                nn.init.kaiming_normal_(module.weight,mode='fan_out',nonlinearity='relu')

    def forward(self,x):
        x=F.relu(self.bn(self.conv(x)))
        for block in self.blocks:
            x=block(x)
        return self.fc(x.mean((2,3)))


def base_spec():
    return {'family':'resnet18','classes':200,'widths':OUTER.copy(),'input_size':64,'stem':'3x3_stride1_no_maxpool'}


def build(spec):
    return ResNet18(spec) if spec['family']=='resnet18' else OLD_BUILD(spec)


@torch.inference_mode()
def cost(model,details=False):
    handles=[];counts={}
    def hook(name):
        def count(module,inputs,output):
            counts[name]=int(output.numel()*module.in_channels*math.prod(module.kernel_size)//module.groups) if isinstance(module,nn.Conv2d) else int(output.numel()*module.in_features)
        return count
    for name,module in model.named_modules():
        if isinstance(module,(nn.Conv2d,nn.Linear)):
            handles.append(module.register_forward_hook(hook(name)))
    training=model.training
    model.eval()
    try:
        n=model.spec.get('input_size',32)
        model(torch.zeros(1,3,n,n,device=next(model.parameters()).device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(training)
    macs=sum(counts.values())
    if model.spec['family']=='mini_vit':
        tokens=(32//model.spec['patch_size'])**2
        macs+=sum(2*tokens**2*b.attn.heads*b.attn.head_dim for b in model.blocks)
    result={'params':sum(p.numel() for p in model.parameters()),'macs':macs,'flops':2*macs}
    return (result,counts) if details else result


def linear_cost(model):
    if model.spec['family']!='resnet18':
        return OLD_LINEAR_COST(model)
    full=np.array(model.spec['widths'])
    measured,counts=cost(model,details=True)
    coefficients=np.array([(counts[f'blocks.{i}.conv1']+counts[f'blocks.{i}.conv2'])//w for i,w in enumerate(full)])
    return full,coefficients,int(measured['macs']-coefficients@full)


@torch.no_grad()
def prune(model,widths,orders):
    if model.spec['family']!='resnet18':
        return OLD_PRUNE(model,widths,orders)
    result=copy.deepcopy(model).cpu();masks=[]
    for block,width,order in zip(result.blocks,widths,orders):
        keep=np.sort(np.asarray(order[:int(width)],dtype=int));masks.append(keep.tolist())
        a,b,c=block.conv1,block.bn1,block.conv2
        conv1=nn.Conv2d(a.in_channels,len(keep),a.kernel_size,a.stride,a.padding,bias=False)
        bn=nn.BatchNorm2d(len(keep),eps=b.eps,momentum=b.momentum)
        conv2=nn.Conv2d(len(keep),c.out_channels,c.kernel_size,c.stride,c.padding,bias=False)
        conv1.weight.copy_(a.weight[keep]);conv2.weight.copy_(c.weight[:,keep])
        for name in ('weight','bias','running_mean','running_var'):
            getattr(bn,name).copy_(getattr(b,name)[keep])
        bn.num_batches_tracked.copy_(b.num_batches_tracked)
        block.conv1,block.bn1,block.conv2=conv1,bn,conv2
    result.spec['widths']=list(map(int,widths))
    return v2_models.to_gpu(result),masks


def feature_modules(model):
    return [b.mlp.fc2 for b in model.blocks] if model.spec['family']=='mini_vit' else [b.conv2 for b in model.blocks]


def install():
    engine.build=build
    engine.cost=cost
    v2_compress.linear_cost=linear_cost


def allocate_aligned(model,profile,target_macs,quantum=8):
    """Same profile objective as v2, with widths constrained to a fixed multiple."""
    full,coefficients,fixed=linear_cost(model)
    assert np.all(full%quantum==0)
    capacities=full//quantum
    unit=int(np.gcd.reduce(coefficients))*quantum
    weights=coefficients*quantum//unit
    target=int(math.floor((target_macs-fixed)/unit))
    profile=np.maximum(np.asarray(profile,dtype=float)/quantum,1e-8)
    lo,hi=0.,1e6
    for _ in range(70):
        mid=(lo+hi)/2
        if weights@np.clip(profile*mid,1,capacities)<=target:lo=mid
        else:hi=mid
    ideal=np.clip(profile*lo,1,capacities)
    dp=np.full(target+1,np.inf);dp[0]=0
    choices=np.full((len(full),target+1),-1,dtype=np.int16)
    for i,(capacity,weight) in enumerate(zip(capacities,weights)):
        new=np.full_like(dp,np.inf)
        for width in range(1,int(capacity)+1):
            shift=int(width*weight)
            if shift>target:continue
            candidate=dp[:target+1-shift]+((width-ideal[i])/capacity)**2
            better=candidate<new[shift:]
            new[shift:][better]=candidate[better];choices[i,shift:][better]=width
        dp=new
    # All methods use this same highest reachable budget.
    reachable=np.flatnonzero(np.isfinite(dp))
    assert len(reachable)
    state=int(reachable[-1]);widths=[]
    for i in range(len(full)-1,-1,-1):
        width=int(choices[i,state]);assert width>=1
        widths.append(width*quantum);state-=width*int(weights[i])
    assert state==0
    return widths[::-1]
