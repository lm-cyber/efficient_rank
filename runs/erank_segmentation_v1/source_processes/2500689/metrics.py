"""Dataset-level semantic mIoU from pooled confusion counts, with ignored labels."""
from common import CFG,amp
import numpy as np
import torch
from torch.nn import functional as F


def summarize(confusion):
    matrix=np.asarray(confusion,dtype=np.float64)
    intersection=np.diag(matrix);truth=matrix.sum(1);prediction=matrix.sum(0)
    union=truth+prediction-intersection
    present=union>0
    iou=np.divide(intersection,union,out=np.zeros_like(intersection),where=present)
    return {'miou':float(100*iou[present].mean()),'pixel_accuracy':float(100*intersection.sum()/matrix.sum()),
            'present_classes':int(present.sum()),'class_iou':[float(100*x) if valid else None for x,valid in zip(iou,present)],
            'valid_pixels':int(matrix.sum())}


def update(confusion,prediction,labels):
    valid=labels!=CFG['ignore_index']
    encoded=labels[valid]*CFG['classes']+prediction[valid]
    confusion+=torch.bincount(encoded,minlength=CFG['classes']**2).reshape(CFG['classes'],CFG['classes'])


@torch.inference_mode()
def evaluate(model,loader,per_image=False):
    model.eval();confusion=torch.zeros(CFG['classes'],CFG['classes'],device='cuda',dtype=torch.int64)
    individual=[];names=[]
    for x,labels,batch_names in loader:
        x=x.cuda(non_blocking=True)
        with amp():logits=model(x).logits
        for output,target,name in zip(logits,labels,batch_names):
            target=target.cuda(non_blocking=True)
            resized=F.interpolate(output[None].float(),size=target.shape,mode='bilinear',align_corners=False)[0]
            prediction=resized.argmax(0)
            if per_image:
                counts=torch.zeros_like(confusion);update(counts,prediction,target)
                confusion+=counts;individual.append(counts.cpu().numpy());names.append(name)
            else:update(confusion,prediction,target)
            del resized,prediction,target
    matrix=confusion.cpu().numpy();result=summarize(matrix)
    result['images']=len(loader.dataset)
    return (result,matrix,np.stack(individual),names) if per_image else (result,matrix)


def loss_function(logits,labels):
    resized=F.interpolate(logits.float(),size=labels.shape[-2:],mode='bilinear',align_corners=False)
    return F.cross_entropy(resized,labels,ignore_index=CFG['ignore_index'])
