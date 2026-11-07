import copy
import math

import numpy as np
import torch
from scipy.linalg import qr
from torch import nn

from common import CONFIG, FIXED_MACS, FULL_MACS, MAC_UNIT, MAC_WEIGHTS, WIDTHS, on_gpu


def entropy_rank(values, axis=-1):
    values = np.maximum(np.asarray(values, dtype=np.float64), 0)
    total = values.sum(axis=axis, keepdims=True)
    p = np.divide(values, total, out=np.zeros_like(values), where=total > 0)
    logs = np.zeros_like(p)
    np.log(p, out=logs, where=p > 0)
    rank = np.exp(-(p * logs).sum(axis=axis))
    return np.where(np.squeeze(total, axis=axis) > 0, rank, 0)


def spectral_scores(x):
    centered = np.asarray(x, dtype=np.float64) - np.mean(x, axis=0, dtype=np.float64)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular**2
    return {"erank_sigma": float(entropy_rank(singular)),
            "erank_cov": float(entropy_rank(energy)),
            "energy_rank95": int(np.searchsorted(np.cumsum(energy), 0.95 * energy.sum()) + 1)
                              if energy.sum() > 0 else 0}, singular, centered


@torch.inference_mode()
def collect_activations(model, data):
    pieces, handles = [[] for _ in model.blocks], []
    position_ids = {}
    for i, block in enumerate(model.blocks):
        def capture(layer, inputs, i=i):
            x = inputs[0].detach().float().flatten(2).transpose(1, 2)
            if i not in position_ids:
                count = min(CONFIG["calibration_positions"], x.shape[1])
                ids = np.random.default_rng(4000 + i).choice(x.shape[1], count, replace=False)
                position_ids[i] = torch.as_tensor(ids, device="cuda")
            pieces[i].append(x[:, position_ids[i]].cpu().numpy())
        handles.append(block.conv2.register_forward_pre_hook(capture))
    model.eval()
    try:
        for x, _ in data.batches("calibration", batch_size=128):
            model(x)  # FP32 feature extraction, shared by all spectral variants.
    finally:
        for handle in handles:
            handle.remove()
    return [np.concatenate(values) for values in pieces]


def describe_activations(arrays):
    rows, orders = [], []
    for i, array in enumerate(arrays):
        n, positions, channels = array.shape
        scores, singular, centered = spectral_scores(array.reshape(-1, channels))
        _, _, pivots = qr(centered, mode="economic", pivoting=True, check_finite=False)
        per_image = array.astype(np.float64)
        per_image -= per_image.mean(axis=1, keepdims=True)
        gram = np.einsum("npc,npd->ncd", per_image, per_image, optimize=True)
        eigen = np.maximum(np.linalg.eigvalsh(gram), 0)
        per_cov = entropy_rank(eigen)
        per_sigma = entropy_rank(np.sqrt(eigen))
        rows.append({"block": i + 1, "stage": i // 3 + 1, "channels": channels,
                     "observations": n * positions, "positions_per_image": positions,
                     "per_image_rank_cap": min(positions - 1, channels), **scores,
                     "normalized_sigma": scores["erank_sigma"] / channels,
                     "normalized_cov": scores["erank_cov"] / channels,
                     "mean_per_image_sigma": float(per_sigma.mean()),
                     "mean_per_image_cov": float(per_cov.mean()),
                     "singular_values": singular.tolist()})
        orders.append(pivots.tolist())
    return rows, orders


def widths_at_budget(scores, retained):
    """Round a scaled profile using DP, with exactly shared integer MAC budgets."""
    scores = np.asarray(scores, dtype=np.float64)
    full = np.array(WIDTHS)
    minimum = CONFIG["min_width"]
    target = math.floor((FULL_MACS * retained - FIXED_MACS) / MAC_UNIT)
    lo, hi = 0.0, 1000.0
    for _ in range(80):
        mid = (lo + hi) / 2
        proposed = np.clip(mid * scores, minimum, full)
        if MAC_WEIGHTS @ proposed <= target:
            lo = mid
        else:
            hi = mid
    ideal = np.clip(lo * scores, minimum, full)
    dp = np.full(target + 1, np.inf)
    dp[0] = 0.0
    choices = np.full((len(full), target + 1), -1, dtype=np.int16)
    for i, (capacity, cost) in enumerate(zip(full, MAC_WEIGHTS)):
        next_dp = np.full_like(dp, np.inf)
        for width in range(minimum, capacity + 1):
            shift = int(width * cost)
            if shift > target:
                break
            candidate = dp[:target + 1 - shift] + ((width - ideal[i]) / capacity)**2
            better = candidate < next_dp[shift:]
            next_dp[shift:][better] = candidate[better]
            choices[i, shift:][better] = width
        dp = next_dp
    if not np.isfinite(dp[target]):
        raise RuntimeError(f"The common MAC target is not feasible: {target}")
    widths, state = [], target
    for i in range(len(full) - 1, -1, -1):
        width = int(choices[i, state])
        assert minimum <= width <= full[i]
        widths.append(width)
        state -= width * int(MAC_WEIGHTS[i])
    widths.reverse()
    assert state == 0 and MAC_WEIGHTS @ widths == target
    return widths


def global_random_widths(removed, seed, target_units=None):
    """Uniform sampling of deletion slots without replacement, optionally conditional on MACs."""
    rng = np.random.default_rng(seed)
    full = np.array(WIDTHS)
    slots = full - CONFIG["min_width"]
    if target_units is None:
        deletions = rng.multivariate_hypergeometric(slots, removed)
        return (full - deletions).tolist(), 1
    tried = 0
    while tried < 1_000_000:
        deletions = rng.multivariate_hypergeometric(slots, removed, size=4096)
        widths = full - deletions
        matches = np.flatnonzero(widths @ MAC_WEIGHTS == target_units)
        if len(matches):
            index = int(matches[0])
            return widths[index].tolist(), tried + index + 1
        tried += len(deletions)
    raise RuntimeError("Random conditional sampling did not reach the exact MAC target")


def uniform_with_deletion_count(removed):
    full = np.array(WIDTHS)
    ideal = removed * full / full.sum()
    counts = np.minimum(np.floor(ideal).astype(int), full - CONFIG["min_width"])
    while counts.sum() < removed:
        allowed = np.flatnonzero(counts < full - CONFIG["min_width"])
        best = allowed[np.argmax(ideal[allowed] - counts[allowed])]
        counts[best] += 1
    return (full - counts).tolist()


def random_orders(seed):
    rng = np.random.default_rng(seed)
    return [rng.permutation(width).tolist() for width in WIDTHS]


def fold_deleted_channels(weight, activations, keep):
    """Uncentered ridge reconstruction, folded into the next convolution's weights."""
    channels = weight.shape[1]
    deleted = np.setdiff1d(np.arange(channels), keep)
    folded = weight[:, keep].copy()
    if not len(deleted):
        return folded
    x = activations.reshape(-1, channels).astype(np.float64)
    retained = x[:, keep]
    gram = retained.T @ retained
    ridge = max(1e-12, 1e-6 * np.trace(gram) / len(keep))
    coefficients = np.linalg.solve(gram + ridge * np.eye(len(keep)), retained.T @ x[:, deleted])
    folded += np.einsum("ocij,kc->okij", weight[:, deleted], coefficients).astype(weight.dtype)
    return folded


def prune(base, widths, orders, arrays=None, compensate=False):
    result = copy.deepcopy(base).cpu()
    masks = []
    for i, (block, width, order) in enumerate(zip(result.blocks, widths, orders)):
        width = int(width)
        keep = np.sort(np.asarray(order[:width], dtype=int))
        masks.append(keep.tolist())
        old1, oldbn, old2 = block.conv1, block.bn1, block.conv2
        conv1 = nn.Conv2d(old1.in_channels, width, 3, old1.stride, 1, bias=False)
        bn1 = nn.BatchNorm2d(width, eps=oldbn.eps, momentum=oldbn.momentum)
        conv2 = nn.Conv2d(width, old2.out_channels, 3, padding=1, bias=False)
        with torch.no_grad():
            conv1.weight.copy_(old1.weight[keep])
            if compensate:
                folded = fold_deleted_channels(old2.weight.detach().numpy(), arrays[i], keep)
                conv2.weight.copy_(torch.from_numpy(folded))
            else:
                conv2.weight.copy_(old2.weight[:, keep])
            for name in ("weight", "bias", "running_mean", "running_var"):
                getattr(bn1, name).copy_(getattr(oldbn, name)[keep])
            bn1.num_batches_tracked.copy_(oldbn.num_batches_tracked)
        block.conv1, block.bn1, block.conv2 = conv1, bn1, conv2
    result.widths = list(map(int, widths))
    return on_gpu(result), masks
