"""Compare graph replay against the original eager optimizer trajectory before enabling it."""
import copy
import json
import time

import torch
from torch import nn
from torch.nn import functional as F

from capture_step import CapturedStep
from data import CFG, RUN
from engine import amp, save_json, seed_all, setup
from models import base_spec, build, sparse_parameters, to_gpu


def main():
    setup()
    results = []
    for family in CFG['families']:
        for sparse in (False, True):
            seed_all(991)
            eager = to_gpu(build(base_spec(family, 10, CFG['vit'], gated=sparse)))
            graph_model = copy.deepcopy(eager)
            is_vit = family == 'mini_vit'
            def optimizer(model):
                if is_vit:
                    return torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05, fused=True)
                return torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=.0005, foreach=True)
            opt_e, opt_g = optimizer(eager), optimizer(graph_model)
            graphs = {}
            differences = []
            for step in range(8):
                batch = 32 if step % 3 else 20
                x = torch.randn(batch, 3, 32, 32, device='cuda').contiguous(memory_format=torch.channels_last)
                y = torch.arange(batch, device='cuda') % 10
                y2 = y.flip(0)
                lam = 0.61
                eager.train()
                opt_e.zero_grad(set_to_none=True)
                with amp():
                    logits = eager(x)
                    if is_vit:
                        loss = lam * F.cross_entropy(logits, y, label_smoothing=.1) + (1-lam)*F.cross_entropy(logits, y2, label_smoothing=.1)
                    else:
                        loss = F.cross_entropy(logits, y)
                    if sparse:
                        loss = loss + CFG['gate_l1' if is_vit else 'bn_l1'] * sum(p.abs().sum() for p in sparse_parameters(eager))
                loss.backward()
                if is_vit:
                    nn.utils.clip_grad_norm_(eager.parameters(), 1.0)
                opt_e.step()
                if batch not in graphs:
                    graphs[batch] = CapturedStep(graph_model, opt_g, x, y, sparse=sparse)
                graphs[batch](x, y, y2, lam)
                pairs = zip(eager.state_dict().items(), graph_model.state_dict().items())
                error = 0.0
                for (name, a), (_, b) in pairs:
                    error = max(error, float((a.float() - b.float()).abs().max()))
                    assert torch.allclose(a, b, rtol=1e-5, atol=2e-6), (family, sparse, step, name, error)
                differences.append(error)
            # Compare latency of replay+optimizer to an actual eager step on the same architecture.
            batch = 256
            x = torch.randn(batch, 3, 32, 32, device='cuda').contiguous(memory_format=torch.channels_last)
            y = torch.arange(batch, device='cuda') % 10
            captured = CapturedStep(graph_model, opt_g, x, y, sparse=sparse)
            torch.cuda.synchronize()
            start = time.monotonic()
            for _ in range(20):
                captured(x, y, y.flip(0), 0.61)
            torch.cuda.synchronize()
            replay_ms = 1000 * (time.monotonic() - start) / 20
            results.append({'family': family, 'sparse': sparse, 'max_state_difference': max(differences),
                            'graph_train_step_ms_batch256': replay_ms})
            print(results[-1], flush=True)
            del captured, graphs, eager, graph_model
            torch.cuda.empty_cache()
    save_json(RUN / 'verification_capture.json', {'passed': results})


if __name__ == '__main__':
    main()
