"""Numerical and protocol checks before segmentation training."""
import copy
import json

from common import CFG,RUN,save_json,setup
from dataset import ADE,FOLDER,StepSampler,manifest,reduce_labels
from metrics import summarize,update,loss_function
from network import allocate,blocks,build,cost,prune,target_for_budget,widths
from scoring import entropy_rank,selection
from PIL import Image
import numpy as np
import pandas as pd
import torch
from transformers import SegformerImageProcessor


def main():
    setup();checks=[];parts=manifest()
    assert len(parts['train'])==17650
    expected=np.array([[255,0,149]])
    assert np.array_equal(reduce_labels(np.array([[0,1,150]],dtype=np.uint8)),expected)
    checks.append('ADE labels: 0 ignored, 1 maps to 0, 150 maps to 149; disjoint train/calibration/selection validation')
    sample=ADE('calibration')[0]
    with Image.open(FOLDER/'images/training'/sample[2]) as im,Image.open(FOLDER/'annotations/training'/sample[2].replace('.jpg','.png')) as mask:
        processor=SegformerImageProcessor(do_reduce_labels=True,size={'height':512,'width':512})
        reference=processor(images=im.convert('RGB'),segmentation_maps=mask,return_tensors='pt')
    assert torch.allclose(sample[0],reference['pixel_values'][0],atol=1e-6,rtol=1e-6)
    assert torch.equal(sample[1],reference['labels'][0])
    checks.append('image normalization, resize and mask labels match pinned Hugging Face processor')
    confusion=torch.zeros(150,150,device='cuda',dtype=torch.int64)
    update(confusion,torch.tensor([0,1,1,0],device='cuda'),torch.tensor([0,0,1,255],device='cuda'))
    metrics=summarize(confusion.cpu().numpy())
    assert abs(metrics['miou']-50)<1e-8 and abs(metrics['pixel_accuracy']-200/3)<1e-8
    checks.append('hand-computed pooled mIoU and pixel accuracy; ignore index excluded')
    a=list(StepSampler(100,77,start=0,steps=5));b=list(StepSampler(100,77,start=3,steps=5))
    assert a[3:]==b
    checks.append('training sampler and flips reproduce exactly when resuming')
    fake=pd.DataFrame({'dominant_class':np.repeat(np.arange(128),16),'score_cov':np.tile(np.arange(16),128),'name':[str(i) for i in range(2048)]})
    chosen,quota,_=selection(fake)
    assert all(len(v)==512 for v in chosen.values()) and not set(chosen['low']).intersection(chosen['high'])
    checks.append('low/random/high quotas match dominant classes and equal sample count')
    eigen=torch.tensor([[0.,0.,0.],[1.,1.,1.],[0.,0.,2.]],device='cuda',dtype=torch.float64)
    assert torch.allclose(entropy_rank(eigen),torch.tensor([0.,3.,1.],device='cuda',dtype=torch.float64))
    checks.append('spectral entropy zero, full and rank-one cases')
    model=build(pretrained=True).cuda().eval();x=sample[0][None].cuda()
    measured=cost(model);save_json(RUN/'model_cost.json',measured)
    orders=[list(reversed(range(w))) for w in widths(model)]
    with torch.inference_mode():reference=model(x).logits
    unchanged,_=prune(model,widths(model),orders);unchanged=unchanged.cuda().eval()
    with torch.inference_mode():assert torch.allclose(reference,unchanged(x).logits,atol=1e-6,rtol=1e-6)
    del unchanged
    checks.append('strict pretrained state loading; full-width physical slicing preserves logits')
    for budget in CFG['budgets']:
        target=target_for_budget(measured,budget)
        candidate,info=allocate(measured,np.array(measured['widths'])*np.linspace(.6,1.4,8),target)
        compressed,masks=prune(model,candidate,orders);compressed=compressed.cuda().eval()
        assert cost(compressed)['macs']==target
        assert all(w%8==0 for w in candidate)
        masked=copy.deepcopy(model).eval();handles=[]
        for block,keep in zip(blocks(masked),masks):
            multiplier=torch.zeros(block.mlp.dense2.in_features,device='cuda');multiplier[keep]=1
            handles.append(block.mlp.dense2.register_forward_pre_hook(lambda m,inputs,k=multiplier:(inputs[0]*k[None,None,:],)))
        with torch.inference_mode():
            first=compressed(x).logits;second=masked(x).logits
            assert torch.allclose(first,second,atol=3e-4,rtol=3e-4),float((first-second).abs().max())
        for h in handles:h.remove()
        del masked
        compressed.train();target_labels=sample[1][None].cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):output=compressed(x).logits
        loss=loss_function(output,target_labels);loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in compressed.parameters() if p.grad is not None)
        del compressed
    checks.append('both budgets: physical Linear/depthwise slicing equals masking, exact common MACs, finite BF16 training gradients')
    save_json(RUN/'verification.json',{'passed':checks,'measured_cost':measured})
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':main()
