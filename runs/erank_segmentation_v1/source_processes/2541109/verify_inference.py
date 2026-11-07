"""Functional CUDA Graph check only: contains no scientific latency measurements."""
import gc
import json
import time

from common import CFG,RUN,save_json,setup
from network import build,prune
import torch
from torch.nn import functional as F


@torch.inference_mode()
def main():
    setup();jobs=json.loads((RUN/'planned_runs.json').read_text())
    compressed=next(j for j in jobs if j['seed']==42 and j['method']=='uniform' and j['budget']==.8)
    rows=[]
    for spec in (None,compressed):
        model=build(pretrained=True)
        if spec is not None:model,_=prune(model,spec['widths'],spec['orders'])
        model=model.cuda().eval()
        for batch in CFG['latency_batches']:
            x=torch.randn(batch,3,CFG['image_size'],CFG['image_size'],device='cuda')
            def forward():
                logits=model(x).logits
                return F.interpolate(logits.float(),size=x.shape[-2:],mode='bilinear',align_corners=False).argmax(1)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                for _ in range(3):forward()
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):output=forward()
            for _ in range(2):
                x.copy_(torch.randn_like(x))
                with torch.autocast('cuda',dtype=torch.bfloat16):reference=forward()
                graph.replay()
                assert torch.equal(output,reference),'Graph map differs from cached eager map'
            rows.append({'model':'original' if spec is None else 'uniform_minus20','batch':batch,'inputs_checked':2,'exact_map_match':True})
            del graph,output,reference,x,stream;gc.collect();torch.cuda.empty_cache()
        del model;gc.collect();torch.cuda.empty_cache()
    save_json(RUN/'inference_verification.json',{'at':time.time(),'passed':rows,
              'scope':'Synthetic random inputs and exact prediction-map equality only; no timings recorded'})
    print('CUDA Graph output agrees with eager: 2 architectures, both batches, 2 changed inputs each',flush=True)


if __name__=='__main__':main()
