"""GPU segmentation-map latency: model, bilinear upsample, and argmax at 512px."""
import gc
import time

from common import CFG,external_gpu_processes
import numpy as np
import torch
from torch.nn import functional as F


@torch.inference_mode()
def measure(model,batch,mode):
    assert not external_gpu_processes(),'Latency requires exclusive GPU 0'
    model.eval()
    generator=torch.Generator(device='cuda').manual_seed(7717)
    x=torch.randn(batch,3,CFG['image_size'],CFG['image_size'],device='cuda',generator=generator)
    def forward():
        logits=model(x).logits
        return F.interpolate(logits.float(),size=x.shape[-2:],mode='bilinear',align_corners=False).argmax(1)
    graph=None;output=None
    if mode=='cuda_graph':
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            for _ in range(CFG['latency_warmup']):reference=forward()
        torch.cuda.current_stream().wait_stream(stream)
        reference=reference.clone();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):output=forward()
        graph.replay();assert torch.equal(output,reference),'Graph changed the predicted segmentation map'
        operation=graph.replay
    else:
        assert mode=='eager'
        operation=forward
    starts=[torch.cuda.Event(enable_timing=True) for _ in range(CFG['latency_repeats'])]
    ends=[torch.cuda.Event(enable_timing=True) for _ in starts]
    # Original cached autocast for eager; graph already contains its BF16 casts.
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for _ in range(CFG['latency_warmup']):operation()
        torch.cuda.synchronize();wall_start=time.perf_counter()
        for start,end in zip(starts,ends):
            start.record();operation();end.record()
        torch.cuda.synchronize();wall_ms=1000*(time.perf_counter()-wall_start)/len(starts)
    values=[a.elapsed_time(b) for a,b in zip(starts,ends)]
    assert not external_gpu_processes(),'Concurrent GPU workload contaminated measurement'
    result={'batch':batch,'mode':mode,'median_ms':float(np.median(values)),'wall_mean_ms':wall_ms,
            'milliseconds':values,'scope':'GPU model + resize logits to 512x512 + argmax; no CPU preprocessing/copy',
            'graph_map_equality_checked':mode=='cuda_graph'}
    del x,graph,output,operation
    gc.collect()
    return result
