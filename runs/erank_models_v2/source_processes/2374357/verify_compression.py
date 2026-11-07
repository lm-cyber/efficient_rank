import copy
import json

import numpy as np
import torch

from compress import allocate, architectural_variant, factorize, linear_cost, prune, prune_heads, random_widths, slimming_widths
from data import CFG, RUN
from engine import load_model, save_json, save_torch, seed_all, setup
from models import base_spec, build, cost, to_gpu


def main():
    setup()
    passed = []
    for family in CFG['families']:
        seed_all(915)
        model = to_gpu(build(base_spec(family, 10, CFG['vit'])))
        model.eval()
        x = torch.randn(4, 3, 32, 32, device='cuda')
        orders = [list(range(w - 1, -1, -1)) for w in model.spec['widths']]
        full, masks = prune(model, model.spec['widths'], orders)
        full.eval()
        with torch.inference_mode():
            assert torch.equal(model(x), full(x))
        passed.append(f'{family}: full-width pruning preserves exact outputs')
        full_macs = cost(model)['macs']
        for budget in CFG['budgets']:
            target = full_macs * budget
            profile = np.random.default_rng(20).uniform(0.5, 1, len(orders)) * model.spec['widths']
            a = allocate(model, model.spec['widths'], target)
            b = allocate(model, profile, target)
            r = random_widths(model, b, 321)
            utility = [np.random.default_rng(i).uniform(0, 1, w) for i, w in enumerate(model.spec['widths'])]
            g = allocate(model, model.spec['widths'], target, utility=utility)
            slimming = slimming_widths(model, utility, target)
            coefficients = linear_cost(model)[1]
            assert coefficients @ a == coefficients @ b == coefficients @ r == coefficients @ g == coefficients @ slimming
            assert sum(r) == sum(b)
            compressed, _ = prune(model, b, orders)
            expected = linear_cost(model)[2] + int(coefficients @ b)
            assert cost(compressed)['macs'] == expected
            factored, detail = factorize(model, target)
            assert cost(factored)['macs'] <= target
            assert detail['budget_gap_fraction'] < 0.015
            path = RUN / 'smoke' / f'{family}_svd_b{budget}.pt'
            save_torch(path, {'architecture': factored.spec, 'state_dict': factored.state_dict()})
            restored = load_model(path)
            restored.eval(), factored.eval()
            with torch.inference_mode():
                assert torch.equal(restored(x), factored(x))
                loss = torch.nn.functional.cross_entropy(factored(x), torch.arange(4, device='cuda'))
                assert torch.isfinite(loss)
            factored.train()
            factored(x).square().mean().backward()
            assert all(torch.isfinite(p.grad).all() for p in factored.parameters() if p.grad is not None)
            passed.append(f'{family} b{budget}: equal channel budgets, random constraints, SVD cost/roundtrip/backward')
            del compressed, factored, restored
        if family == 'mini_vit':
            headed, _ = prune_heads(model, [3] * 6)
            headed.eval()
            with torch.inference_mode():
                assert torch.equal(headed(x), model(x))
            for heads in (1, 2):
                headed, _ = prune_heads(model, [heads] * 6)
                assert cost(headed)['macs'] < full_macs
                headed(x).square().mean().backward()
            for variant in ('depth4_dim96', 'depth8_dim64'):
                spec = architectural_variant(model, variant, full_macs * 0.75)
                variant_model = to_gpu(build(spec))
                assert abs(cost(variant_model)['macs'] / full_macs - 0.75) < 0.001
                variant_model(x).square().mean().backward()
            passed.append('ViT: head pruning identity/gradient; both depth alternatives match 75% compute')
        del model, full
    save_json(RUN / 'verification_compression.json', {'passed': passed})
    print(json.dumps({'passed': passed}, indent=2))


if __name__ == '__main__':
    main()
