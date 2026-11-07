"""ERank, actual spatial HRank, per-example first-order Taylor, and QR channel selection."""
import numpy as np
import torch
from scipy.linalg import qr
from torch import nn
from torch.nn import functional as F

from data import CFG
from models import feature_modules, sparse_parameters


def erank(values):
    values = np.maximum(values, 0).astype(np.float64)
    if values.sum() <= 0:
        return 0.0
    p = values / values.sum()
    return float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))))


@torch.inference_mode()
def collect_features(model, data, limit=None, hrank=False):
    pieces = [[] for _ in model.blocks]
    ranks = [[] for _ in model.blocks]
    positions = {}
    seen = [0]
    handles = []
    for i, module in enumerate(feature_modules(model)):
        def capture(layer, inputs, i=i):
            x = inputs[0].detach().float()
            if x.ndim == 4:
                spatial = x
                rows = x.flatten(2).transpose(1, 2)
            else:
                side = int(round(x.shape[1] ** 0.5))
                assert side * side == x.shape[1]
                spatial = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], side, side)
                rows = x
            if i not in positions:
                selected = np.random.default_rng(4000 + i).choice(rows.shape[1], min(CFG['rank_positions'], rows.shape[1]), replace=False)
                positions[i] = torch.as_tensor(selected, device='cuda')
            pieces[i].append(rows[:, positions[i]].cpu().numpy())
            if hrank and seen[0] < CFG['score_images']:
                maps = spatial[:CFG['score_images'] - seen[0]].cpu().numpy()
                # Same numerical matrix-rank definition for CNN maps and ViT patch grids.
                singular = np.linalg.svd(maps, compute_uv=False)
                tol = max(maps.shape[-2:]) * np.finfo(np.float32).eps * singular[..., :1]
                ranks[i].append((singular > tol).sum(-1).astype(np.float64))
        handles.append(module.register_forward_pre_hook(capture))
    model.eval()
    try:
        for x, _ in data.batches('calibration', limit=limit, batch_size=64):
            model(x)  # FP32 scoring, including both spectral definitions.
            seen[0] += len(x)
    finally:
        for handle in handles:
            handle.remove()
    arrays = [np.concatenate(p) for p in pieces]
    spatial_scores = [np.concatenate(r).mean(0).tolist() for r in ranks] if hrank else None
    return arrays, spatial_scores


def spectral_description(array, with_order=True):
    width = array.shape[-1]
    rows = array.reshape(-1, width).astype(np.float64)
    rows -= rows.mean(0, keepdims=True)
    gram = rows.T @ rows
    eig = np.maximum(np.linalg.eigvalsh(gram), 0)
    sigma, cov = erank(np.sqrt(eig)), erank(eig)
    result = {'width': width, 'erank_sigma': sigma, 'erank_cov': cov,
              'normalized_sigma': sigma / width, 'normalized_cov': cov / width,
              'observations': len(rows), 'singular_values': np.sqrt(eig[::-1]).tolist()}
    if with_order:
        per_image = array.astype(np.float64)
        per_image -= per_image.mean(axis=1, keepdims=True)
        spatial_gram = np.einsum('npc,nqc->npq', per_image, per_image, optimize=True)
        spatial_eig = np.maximum(np.linalg.eigvalsh(spatial_gram), 0)
        result.update(mean_per_image_sigma=float(np.mean([erank(np.sqrt(e)) for e in spatial_eig])),
                      mean_per_image_cov=float(np.mean([erank(e) for e in spatial_eig])),
                      positions_per_image=array.shape[1],
                      per_image_rank_cap=min(array.shape[1] - 1, width))
        # Bounded, fixed QR calibration sample; both architecture criteria use this same ordering.
        selected = np.random.default_rng(151).choice(len(rows), min(4096, len(rows)), replace=False)
        _, _, order = qr(rows[selected], mode='economic', pivoting=True, check_finite=False)
        result['qr_order'] = order.tolist()
    return result


def taylor_scores(model, data):
    scores = [np.zeros(w, dtype=np.float64) for w in model.spec['widths']]
    saved = [None] * len(scores)
    handles = []
    for i, module in enumerate(feature_modules(model)):
        def capture(layer, inputs, i=i):
            saved[i] = inputs[0]
        handles.append(module.register_forward_pre_hook(capture))
    n = 0
    model.eval()
    try:
        for x, y in data.batches('calibration', limit=CFG['score_images'], batch_size=64):
            model.zero_grad(set_to_none=True)
            logits = model(x)
            # Sum loss keeps each sample's gradient independent of batch size.
            loss = F.cross_entropy(logits, y, reduction='sum')
            grads = torch.autograd.grad(loss, saved)
            for i, (activation, gradient) in enumerate(zip(saved, grads)):
                contribution = activation.detach() * gradient.detach()
                value = contribution.sum((2, 3)) if contribution.ndim == 4 else contribution.sum(1)
                scores[i] += value.square().sum(0).cpu().numpy()
            n += len(y)
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
    return [(s / n).tolist() for s in scores]


def all_scores(model, data):
    arrays, hrank = collect_features(model, data, hrank=True)
    descriptions = [spectral_description(array) for array in arrays]
    taylor = taylor_scores(model, data)
    l1 = []
    for block in model.blocks:
        weight = block.conv1.weight if model.spec['family'] == 'resnet20' else block.mlp.fc1.weight
        l1.append(weight.detach().abs().flatten(1).sum(1).cpu().tolist())
    return {'layers': descriptions, 'l1': l1, 'taylor': taylor, 'hrank': hrank,
            'calibration_images': len(arrays[0]), 'channel_score_images': CFG['score_images']}, arrays


def descending(scores):
    # Stable channel-index tie breaking, relevant when spatial ranks saturate.
    return [np.argsort(-np.asarray(s), kind='stable').tolist() for s in scores]
