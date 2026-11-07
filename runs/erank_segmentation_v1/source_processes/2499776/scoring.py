"""Rank images, match semantic strata, and score actual Mix-FFN channels."""
import json
from pathlib import Path
import time

from common import CFG,RUN,save_json,setup
from dataset import ADE,evaluation_loader,manifest
from network import blocks,build,cost,widths
import numpy as np
import pandas as pd
from scipy.linalg import qr
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


def entropy_rank(values):
    total=values.clamp_min(0).sum(-1,keepdim=True)
    p=values.clamp_min(0)/total.clamp_min(torch.finfo(values.dtype).tiny)
    entropy=-(p*p.clamp_min(torch.finfo(values.dtype).tiny).log()).sum(-1)
    return torch.where(total.squeeze(-1)>0,entropy.exp(),torch.zeros_like(entropy))


@torch.inference_mode()
def collect(model,split='calibration',save_features=True):
    destination=RUN/f'{split}_image_scores.csv'
    if destination.exists():return pd.read_csv(destination)
    loader=evaluation_loader(split,batch_size=4)
    n=len(loader.dataset);sizes=widths(model);offset=[0];handles=[];selected={};rows=[]
    sigma=np.empty((n,len(sizes)));cov=np.empty_like(sigma);caps={}
    folder=RUN/'feature_cache';folder.mkdir(exist_ok=True)
    cached=[np.lib.format.open_memmap(folder/f'layer{i}.npy',mode='w+',dtype=np.float32,
             shape=(n,CFG['pooled_rank_positions'],width)) for i,width in enumerate(sizes)] if save_features else []
    for i,block in enumerate(blocks(model)):
        def capture(module,inputs,i=i):
            features=inputs[0].detach().float();b,p,c=features.shape
            if i not in selected:
                rng=np.random.default_rng(17000+i)
                sample=rng.choice(p,min(CFG['image_rank_positions'],p),replace=False)
                pooled=np.random.default_rng(4000+i).choice(p,CFG['pooled_rank_positions'],replace=False)
                selected[i]=(torch.as_tensor(sample,device='cuda'),torch.as_tensor(pooled,device='cuda'))
            sample,pool=selected[i]
            x=features[:,sample].double();x=x-x.mean(1,keepdim=True)
            eigen=torch.linalg.eigvalsh(x@x.transpose(1,2)).clamp_min(0)
            cap=min(c,len(sample)-1);caps[i]=cap;target=slice(offset[0],offset[0]+b)
            cov[target,i]=entropy_rank(eigen).cpu().numpy()/cap
            sigma[target,i]=entropy_rank(eigen.sqrt()).cpu().numpy()/cap
            if save_features:cached[i][target]=features[:,pool].cpu().numpy()
        handles.append(block.mlp.dense2.register_forward_pre_hook(capture))
    model.eval()
    try:
        for x,labels,names in loader:
            model(x.cuda(non_blocking=True))  # FP32 activations, float64 spectra.
            for label,name in zip(labels,names):
                values=label.numpy();valid=values[values!=255]
                counts=np.bincount(valid,minlength=CFG['classes'])
                rows.append({'name':name,'dominant_class':int(counts.argmax()) if len(valid) else -1,
                             'classes_present':int(np.count_nonzero(counts)),
                             'dominant_fraction':float(counts.max()/len(valid)) if len(valid) else 0.})
            offset[0]+=len(x)
            if offset[0]%128==0:print('RANK',split,offset[0],n,flush=True)
    finally:
        for handle in handles:handle.remove()
    assert offset[0]==n
    for a in cached:a.flush()
    frame=pd.DataFrame(rows);frame['score_cov']=cov.mean(1);frame['score_sigma']=sigma.mean(1)
    for i in range(len(sizes)):
        frame[f'cov_l{i}']=cov[:,i];frame[f'sigma_l{i}']=sigma[:,i]
    frame.to_csv(destination,index=False)
    save_json(RUN/f'{split}_rank_metadata.json',{'positions':CFG['image_rank_positions'],'caps':caps,
              'widths':sizes,'precision':'FP32 activations; float64 Gram/eigenvalues','at':time.time()})
    return frame


def selection(frame):
    counts=frame.groupby('dominant_class').size().to_dict();n=CFG['subset_size']
    ideal={c:count*n/len(frame) for c,count in counts.items()}
    quota={c:min(count//2,int(np.floor(ideal[c]))) for c,count in counts.items()}
    while sum(quota.values())<n:
        eligible=[c for c in counts if quota[c]<counts[c]//2]
        assert eligible,'Insufficient examples for two disjoint extremes'
        chosen=max(eligible,key=lambda c:(ideal[c]-quota[c],-c));quota[chosen]+=1
    selected={'low':[],'random':[],'high':[]};thresholds={}
    rng=np.random.default_rng(7781)
    for c,k in sorted(quota.items()):
        if not k:continue
        group=frame[frame.dominant_class==c].sort_values(['score_cov','name'])
        selected['low'].extend(group.index[:k]);selected['high'].extend(group.index[-k:])
        selected['random'].extend(rng.choice(np.sort(group.index),k,replace=False))
        scores=group.score_cov.to_numpy()
        thresholds[str(c)]={'low':float((scores[k-1]+scores[k])/2),'high':float((scores[-k-1]+scores[-k])/2)}
    for group,indices in selected.items():
        selected[group]=[int(i) for i in indices]
        assert len(indices)==n and len(set(indices))==n
        actual=frame.iloc[indices].groupby('dominant_class').size().to_dict()
        assert actual=={c:k for c,k in quota.items() if k}
    assert not set(selected['low']).intersection(selected['high'])
    return selected,quota,thresholds


def pooled_profiles(frame,groups):
    sizes=json.loads((RUN/'calibration_rank_metadata.json').read_text())['widths']
    profiles={name:[] for name in groups};orders=[]
    for i,width in enumerate(sizes):
        source=np.load(RUN/'feature_cache'/f'layer{i}.npy',mmap_mode='r')
        for group,ids in groups.items():
            rows=np.asarray(source[ids]).reshape(-1,width).astype(np.float64)
            rows-=rows.mean(0,keepdims=True)
            x=torch.from_numpy(rows).cuda()
            eigen=torch.linalg.eigvalsh(x.T@x).clamp_min(0)
            profiles[group].append({'width':width,'erank_sigma':float(entropy_rank(eigen.sqrt())),
                                    'erank_cov':float(entropy_rank(eigen))})
            del x,eigen
            if group=='random':
                chosen=np.random.default_rng(151).choice(len(rows),min(CFG['qr_rows'],len(rows)),replace=False)
                _,_,order=qr(rows[chosen],mode='economic',pivoting=True,check_finite=False)
                orders.append(order.tolist())
        print('POOLED LAYER',i+1,len(sizes),flush=True)
    return profiles,orders


def taylor(model,names):
    scores=[torch.zeros(w,device='cuda',dtype=torch.float64) for w in widths(model)]
    saved=[None]*len(scores);handles=[]
    for i,block in enumerate(blocks(model)):
        def capture(module,inputs,i=i):saved[i]=inputs[0]
        handles.append(block.mlp.dense2.register_forward_pre_hook(capture))
    loader=DataLoader(ADE('calibration',filenames=names),batch_size=2,num_workers=4,pin_memory=True)
    seen=0;model.eval()
    try:
        for x,y,_ in loader:
            x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
            model.zero_grad(set_to_none=True)
            logits=model(x).logits
            up=F.interpolate(logits,size=y.shape[-2:],mode='bilinear',align_corners=False)
            pixel_loss=F.cross_entropy(up,y,ignore_index=255,reduction='none')
            valid=(y!=255).sum((1,2)).clamp_min(1)
            loss=(pixel_loss.sum((1,2))/valid).sum()
            grads=torch.autograd.grad(loss,saved)
            for index,(activation,gradient) in enumerate(zip(saved,grads)):
                scores[index]+=(activation.detach()*gradient.detach()).sum(1).square().sum(0).double()
            seen+=len(x)
            if seen%64==0:print('TAYLOR',seen,len(names),flush=True)
    finally:
        for h in handles:h.remove()
        model.zero_grad(set_to_none=True)
    return [(s/seen).cpu().tolist() for s in scores]


def main():
    setup();manifest()
    model=build(pretrained=True).cuda().eval()
    if not (RUN/'model_cost.json').exists():save_json(RUN/'model_cost.json',cost(model))
    frame=collect(model)
    target=RUN/'scores.json'
    if target.exists():return
    groups,quotas,thresholds=selection(frame)
    profiles,orders=pooled_profiles(frame,groups)
    names=frame.iloc[groups['random']].name.tolist()
    importance=taylor(model,names)
    l1=[b.mlp.dense1.weight.detach().abs().sum(1).cpu().tolist() for b in blocks(model)]
    save_json(target,{'groups':{name:{'indices':ids,'names':frame.iloc[ids].name.tolist(),
              'mean_image_rank_cov':float(frame.iloc[ids].score_cov.mean()),'layers':profiles[name]} for name,ids in groups.items()},
              'quotas':{str(k):v for k,v in quotas.items()},'thresholds':thresholds,'qr_orders':orders,
              'l1':l1,'taylor':importance,'at':time.time()})
    print('SCORING COMPLETE',flush=True)


if __name__=='__main__':main()
