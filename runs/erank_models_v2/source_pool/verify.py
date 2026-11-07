"""Scientific and implementation checks. Tiny training here is not a reported experiment."""
import copy
import json

import numpy as np
import torch

from data import CFG, RUN, Data
from engine import evaluate, load_model, save_json, seed_all, setup, train
from measures import erank, taylor_scores
from models import base_spec, build, cost, to_gpu


def main():
    setup()
    checks = []
    assert erank(np.zeros(3)) == 0
    assert abs(erank(np.ones(7)) - 7) < 1e-10
    assert erank(np.array([1., 0, 0])) == 1
    checks.append('entropy rank zero/one/full cases')
    data = Data('cifar10')
    assert data.test_images is None
    for family in CFG['families']:
        seed_all(999)
        model = to_gpu(build(base_spec(family, 10, CFG['vit'], gated=family == 'mini_vit')))
        measured = cost(model)
        if family == 'resnet20':
            assert measured['macs'] == 40551040
        else:
            n, d, h, m = 64, 96, 96, 384
            expected = n * d * 3 * 4 * 4 + 6 * (4*n*d*h + 2*n*n*h + 2*n*d*m) + d*10
            assert measured['macs'] == expected, (measured, expected)
        checks.append(f'{family} actual module MAC count agrees with formula')
        initial = evaluate(model, data, limit=512)
        assert np.isfinite(initial['loss'])
        # Verify per-example Taylor invariance to the batch partition on a small subset.
        from measures import feature_modules
        tensors, handles = [], []
        layer = feature_modules(model)[0]
        handle = layer.register_forward_pre_hook(lambda mod, inputs: tensors.append(inputs[0]))
        x, y = next(data.batches('calibration', batch_size=4))
        model.eval()
        logits = model(x)
        gradient = torch.autograd.grad(torch.nn.functional.cross_entropy(logits, y, reduction='sum'), tensors[-1])[0]
        batch_score = (tensors[-1] * gradient).detach()
        total_dims = (2, 3) if batch_score.ndim == 4 else (1,)
        batch_score = batch_score.sum(total_dims).square().mean(0)
        each = []
        for index in range(4):
            logits = model(x[index:index+1])
            gradient = torch.autograd.grad(torch.nn.functional.cross_entropy(logits, y[index:index+1]), tensors[-1])[0]
            each.append((tensors[-1] * gradient).detach().sum(total_dims).square()[0])
        handle.remove()
        assert torch.allclose(batch_score, torch.stack(each).mean(0), rtol=5e-3, atol=1e-7)
        checks.append(f'{family} per-example Taylor invariant to scoring batch size')
        for sparse in (False, True):
            seed_all(777)
            model = build(base_spec(family, 10, CFG['vit'], gated=sparse))
            directory = RUN / 'smoke' / f'{family}_{sparse}'
            metadata = {'family': family, 'dataset': 'cifar10', 'model_seed': 777, 'train_seed': 777,
                        'phase': 'verification', 'method': 'smoke'}
            trained = train(model, data, directory, metadata, 2, sparse=sparse, limit=512)
            restored = load_model(directory / 'best.pt')
            x, _ = next(data.batches('val', batch_size=8))
            trained.eval(), restored.eval()
            with torch.inference_mode():
                assert torch.equal(trained(x), restored(x))
            checks.append(f'{family} sparse={sparse}: optimizer, finite loss, checkpoint exact roundtrip')
    assert data.test_images is None
    checks.append('verification never loads the test set')
    save_json(RUN / 'verification_initial.json', {'passed': checks})
    print(json.dumps({'passed': checks}, indent=2), flush=True)


if __name__ == '__main__':
    main()
