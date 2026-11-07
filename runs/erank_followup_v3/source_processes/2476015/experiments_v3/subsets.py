"""Per-image rank strata and separately pooled rank profiles, without test selection."""
import json
from pathlib import Path

from runtime import CFG, RUN, save_json
from tiny_model import feature_modules
import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.stats import spearmanr
import torch
from torch.nn import functional as F
from measures import erank


def entropy_rank(eigenvalues):
    eigenvalues=eigenvalues.clamp_min(0)
    total=eigenvalues.sum(-1,keepdim=True)
    probabilities=eigenvalues/total.clamp_min(torch.finfo(eigenvalues.dtype).tiny)
    entropy=-(probabilities*probabilities.clamp_min(torch.finfo(eigenvalues.dtype).tiny).log()).sum(-1)
    return torch.where(total.squeeze(-1)>0,entropy.exp(),torch.zeros_like(entropy))


@torch.inference_mode()
def score_images(model,data,split,prefix,save_features=False):
    folder=RUN/'diagnostics'/prefix
    folder.mkdir(parents=True,exist_ok=True)
    destination=folder/f'{split}_image_scores.csv'
    metadata_path=folder/f'{split}_score_metadata.json'
    if destination.exists() and metadata_path.exists():
        return pd.read_csv(destination)
    ids=np.arange(10000) if split=='test' else data.ids_cpu[split]
    count=len(ids);widths=model.spec['widths']
    feature_folder=RUN/'feature_cache'/prefix
    arrays=[]
    if save_features:
        feature_folder.mkdir(parents=True,exist_ok=True)
        arrays=[np.lib.format.open_memmap(feature_folder/f'layer{i}.npy',mode='w+',dtype=np.float32,
                  shape=(count,CFG['rank_positions'],width)) for i,width in enumerate(widths)]
    all_cov=np.zeros((count,len(widths)));all_sigma=np.zeros_like(all_cov)
    normalized_cov=np.zeros_like(all_cov);normalized_sigma=np.zeros_like(all_cov)
    actual_caps={};offset=[0];position_indices={};handles=[]
    for i,module in enumerate(feature_modules(model)):
        def capture(layer,inputs,i=i):
            value=inputs[0].detach().float()
            rows=value.flatten(2).transpose(1,2) if value.ndim==4 else value
            b,p,c=rows.shape
            rows64=rows.double()
            centered=rows64-rows64.mean(1,keepdim=True)
            gram=centered.transpose(1,2)@centered if c<=p else centered@centered.transpose(1,2)
            eigenvalues=torch.linalg.eigvalsh(gram).clamp_min(0)
            cov=entropy_rank(eigenvalues).cpu().numpy()
            sigma=entropy_rank(eigenvalues.sqrt()).cpu().numpy()
            cap=min(c,p-1);actual_caps[i]=cap
            target=slice(offset[0],offset[0]+b)
            all_cov[target,i]=cov;all_sigma[target,i]=sigma
            normalized_cov[target,i]=cov/cap;normalized_sigma[target,i]=sigma/cap
            if save_features:
                if i not in position_indices:
                    selected=np.random.default_rng(4000+i).choice(p,CFG['rank_positions'],replace=False)
                    position_indices[i]=torch.as_tensor(selected,device='cuda')
                arrays[i][target]=rows[:,position_indices[i]].cpu().numpy()
        handles.append(module.register_forward_pre_hook(capture))
    labels=[];losses=[];correct=[]
    model.eval()
    try:
        for x,y in data.batches(split,batch_size=32):
            output=model(x)  # FP32, identical rank-scoring precision to v2.
            labels.extend(y.cpu().tolist())
            losses.extend(F.cross_entropy(output,y,reduction='none').cpu().tolist())
            correct.extend((output.argmax(1)==y).cpu().tolist())
            offset[0]+=len(y)
    finally:
        for handle in handles:
            handle.remove()
    assert offset[0]==count
    for array in arrays:
        array.flush()
    frame=pd.DataFrame({'id':ids,'label':labels,'score_cov':normalized_cov.mean(1),
                        'score_sigma':normalized_sigma.mean(1),'dense_loss_fp32':losses,'dense_correct_fp32':correct})
    for i in range(len(widths)):
        frame[f'cov_l{i}']=all_cov[:,i];frame[f'sigma_l{i}']=all_sigma[:,i]
    frame.to_csv(destination,index=False)
    save_json(metadata_path,{'split':split,'n':count,'widths':widths,'per_image_rank_caps':actual_caps,
               'selection_score':'mean per-layer covariance ERank / min(C, P-1)',
               'pooled_positions':CFG['rank_positions'] if save_features else None,'scoring_precision':'FP32 activations, float64 Gram/eigenvalues',
               'features_saved':save_features})
    return frame


def select(frame,classes):
    each=CFG['calibration_size']//classes
    assert each*classes==CFG['calibration_size']
    rng=np.random.default_rng(CFG['subset_seed'])
    positions={'low':[],'random':[],'high':[]};thresholds={}
    for c in range(classes):
        group=frame[frame.label==c].sort_values(['score_cov','id'])
        assert len(group)>=2*each
        positions['low'].extend(group.index[:each].tolist())
        positions['high'].extend(group.index[-each:].tolist())
        positions['random'].extend(rng.choice(np.sort(group.index),each,replace=False).tolist())
        values=group.score_cov.to_numpy()
        thresholds[str(c)]={'low':float((values[each-1]+values[each])/2),
                            'high':float((values[-each-1]+values[-each])/2)}
    for name,indices in positions.items():
        assert len(indices)==CFG['calibration_size'] and len(set(indices))==len(indices)
        assert frame.iloc[indices].groupby('label').size().eq(each).all()
    return positions,thresholds


def pooled_profiles(frame,positions,prefix):
    folder=RUN/'diagnostics'/prefix
    metadata=json.loads((folder/'calibration_holdout_score_metadata.json').read_text())
    records={name:[] for name in positions};common_orders=[]
    for i,width in enumerate(metadata['widths']):
        array=np.load(RUN/'feature_cache'/prefix/f'layer{i}.npy',mmap_mode='r')
        for group,indices in positions.items():
            rows=np.asarray(array[indices]).reshape(-1,width).astype(np.float64)
            rows-=rows.mean(0,keepdims=True)
            eigenvalues=np.linalg.eigvalsh(rows.T@rows).clip(0)
            records[group].append({'width':width,'erank_sigma':erank(np.sqrt(eigenvalues)),
                                  'erank_cov':erank(eigenvalues),
                                  'mean_individual_sigma':float(frame.iloc[indices][f'sigma_l{i}'].mean()),
                                  'mean_individual_cov':float(frame.iloc[indices][f'cov_l{i}'].mean())})
            if group=='random':
                ids=np.random.default_rng(151).choice(len(rows),min(4096,len(rows)),replace=False)
                _,_,order=qr(rows[ids],mode='economic',pivoting=True,check_finite=False)
                common_orders.append(order.tolist())
    return records,common_orders


def prepare(model,data,prefix):
    folder=RUN/'diagnostics'/prefix
    target=folder/'subsets.json'
    if target.exists():
        return json.loads(target.read_text())
    frame=score_images(model,data,'calibration_holdout',prefix,save_features=True)
    positions,thresholds=select(frame,data.classes)
    profiles,orders=pooled_profiles(frame,positions,prefix)
    value={'groups':{name:{'ids':frame.iloc[indices].id.tolist(),
                          'mean_score_cov':float(frame.iloc[indices].score_cov.mean()),
                          'dense_accuracy_fp32':float(100*frame.iloc[indices].dense_correct_fp32.mean()),
                          'layers':profiles[name]} for name,indices in positions.items()},
           'thresholds':thresholds,'common_qr_orders':orders,
           'score_loss_spearman':float(spearmanr(frame.score_cov,frame.dense_loss_fp32).statistic),
           'n_per_group':CFG['calibration_size'],'classes':data.classes}
    save_json(target,value)
    return value


def assign_strata(frame,selection):
    labels=[]
    for row in frame.itertuples():
        bounds=selection['thresholds'][str(row.label)]
        labels.append('low' if row.score_cov<=bounds['low'] else 'high' if row.score_cov>=bounds['high'] else 'middle')
    return np.array(labels)
