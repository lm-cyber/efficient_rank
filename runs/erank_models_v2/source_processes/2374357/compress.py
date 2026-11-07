"""Physical channel pruning and weight factorization with explicit compute budgets."""
import copy
import math

import numpy as np
import torch
from torch import nn

from models import FactorConv, FactorLinear, MLP, build, cost, replace_module, to_gpu


def linear_cost(model):
    full = np.asarray(model.spec['widths'], dtype=int)
    if model.spec['family'] == 'resnet20':
        coefficients = 18432 * np.array([16, 16, 16, 6, 8, 8, 3, 4, 4])
    else:
        n = (32 // model.spec['patch_size']) ** 2
        coefficients = np.full(len(full), 2 * n * model.spec['dim'], dtype=int)
    fixed = cost(model)['macs'] - int(coefficients @ full)
    return full, coefficients, fixed


def allocate(model, profile, target_macs, utility=None):
    full, coefficients, fixed = linear_cost(model)
    unit = int(np.gcd.reduce(coefficients))
    weights = coefficients // unit
    target = math.floor((target_macs - fixed) / unit)
    minimum = 4 if model.spec['family'] == 'resnet20' else 8
    minimum = np.minimum(minimum, full)
    if target < int(weights @ minimum) or target > int(weights @ full):
        raise ValueError(f'Infeasible internal-width MAC budget: {target_macs}, fixed={fixed}')
    profile = np.maximum(np.asarray(profile, dtype=float), 1e-8)
    lo, hi = 0., 1e6
    for _ in range(70):
        mid = (lo + hi) / 2
        if weights @ np.clip(profile * mid, minimum, full) <= target:
            lo = mid
        else:
            hi = mid
    ideal = np.clip(profile * lo, minimum, full)
    dp = np.full(target + 1, np.inf)
    dp[0] = 0
    choices = np.full((len(full), target + 1), -1, dtype=np.int16)
    for i, (capacity, weight) in enumerate(zip(full, weights)):
        next_dp = np.full_like(dp, np.inf)
        cumulative = np.r_[0, np.cumsum(np.sort(utility[i])[::-1])] if utility is not None else None
        for width in range(int(minimum[i]), int(capacity) + 1):
            shift = int(width * weight)
            if shift > target:
                continue
            penalty = -cumulative[width] if cumulative is not None else ((width - ideal[i]) / capacity)**2
            candidate = dp[:target + 1 - shift] + penalty
            better = candidate < next_dp[shift:]
            next_dp[shift:][better] = candidate[better]
            choices[i, shift:][better] = width
        dp = next_dp
    if not np.isfinite(dp[target]):
        raise RuntimeError('Common integer MAC budget unreachable')
    widths, state = [], target
    for i in range(len(full) - 1, -1, -1):
        width = int(choices[i, state])
        widths.append(width)
        state -= width * int(weights[i])
    widths.reverse()
    assert state == 0
    return widths


def random_widths(model, reference, seed):
    full, coefficients, _ = linear_cost(model)
    minimum = 4 if model.spec['family'] == 'resnet20' else 8
    slots = full - minimum
    removed = int(full.sum() - sum(reference))
    target = int(coefficients @ reference)
    rng = np.random.default_rng(seed)
    for attempt in range(250):
        deletions = rng.multivariate_hypergeometric(slots, removed, size=4096)
        widths = full - deletions
        valid = np.flatnonzero(widths @ coefficients == target)
        if len(valid):
            return widths[int(valid[0])].tolist()
    raise RuntimeError('Conditional global random sampling exhausted')


def slimming_widths(model, scores, target_macs):
    """Global gamma/gate threshold, minimum widths, then nearest exact-budget rounding."""
    full, coefficients, fixed = linear_cost(model)
    minimum = 4 if model.spec['family'] == 'resnet20' else 8
    widths = full.copy()
    current = int(fixed + coefficients @ widths)
    ordered = sorted((float(value), layer, channel)
                     for layer, values in enumerate(scores) for channel, value in enumerate(values))
    for _, layer, channel in ordered:
        if current <= target_macs:
            break
        if widths[layer] <= minimum:
            continue
        widths[layer] -= 1
        current -= int(coefficients[layer])
    return allocate(model, widths, target_macs)


@torch.no_grad()
def prune(model, widths, orders):
    result = copy.deepcopy(model).cpu()
    masks = []
    for i, (block, width, order) in enumerate(zip(result.blocks, widths, orders)):
        keep = np.sort(np.asarray(order[:int(width)], dtype=int))
        masks.append(keep.tolist())
        if model.spec['family'] == 'resnet20':
            old1, oldbn, old2 = block.conv1, block.bn1, block.conv2
            conv1 = nn.Conv2d(old1.in_channels, len(keep), 3, old1.stride, 1, bias=False)
            bn = nn.BatchNorm2d(len(keep), eps=oldbn.eps, momentum=oldbn.momentum)
            conv2 = nn.Conv2d(len(keep), old2.out_channels, 3, padding=1, bias=False)
            conv1.weight.copy_(old1.weight[keep])
            conv2.weight.copy_(old2.weight[:, keep])
            for name in ('weight', 'bias', 'running_mean', 'running_var'):
                getattr(bn, name).copy_(getattr(oldbn, name)[keep])
            bn.num_batches_tracked.copy_(oldbn.num_batches_tracked)
            block.conv1, block.bn1, block.conv2 = conv1, bn, conv2
        else:
            old = block.mlp
            mlp = MLP(result.spec['dim'], len(keep), gated=False)
            mlp.fc1.weight.copy_(old.fc1.weight[keep])
            mlp.fc1.bias.copy_(old.fc1.bias[keep])
            mlp.fc2.weight.copy_(old.fc2.weight[:, keep] * old.gate[keep][None, :])
            mlp.fc2.bias.copy_(old.fc2.bias)
            block.mlp = mlp
    result.spec['widths'] = list(map(int, widths))
    if model.spec['family'] == 'mini_vit':
        result.spec['gated'] = False
    return to_gpu(result), masks


@torch.no_grad()
def prune_heads(model, counts):
    result = copy.deepcopy(model).cpu()
    kept_heads = []
    for block, count in zip(result.blocks, counts):
        old = block.attn
        h, d, embedding = old.heads, old.head_dim, old.qkv.in_features
        # Fixed magnitude selector; architectural conclusions do not claim optimal head selection.
        importance = old.proj.weight.abs().reshape(embedding, h, d).sum((0, 2))
        heads = torch.argsort(importance, descending=True, stable=True)[:count].sort().values
        features = (heads[:, None] * d + torch.arange(d)).flatten()
        qkv_indices = torch.cat([features + j * h * d for j in range(3)])
        qkv = nn.Linear(embedding, 3 * count * d)
        proj = nn.Linear(count * d, embedding)
        qkv.weight.copy_(old.qkv.weight[qkv_indices])
        qkv.bias.copy_(old.qkv.bias[qkv_indices])
        proj.weight.copy_(old.proj.weight[:, features])
        proj.bias.copy_(old.proj.bias)
        old.qkv, old.proj, old.heads = qkv, proj, int(count)
        kept_heads.append(heads.tolist())
    result.spec['head_counts'] = list(map(int, counts))
    return to_gpu(result), kept_heads


def architectural_variant(base, name, target_macs):
    spec = copy.deepcopy(base.spec)
    spec['gated'] = False
    if name == 'depth4_dim96':
        spec.update(depth=4, dim=96, heads=3, head_dim=32)
    elif name == 'depth8_dim64':
        spec.update(depth=8, dim=64, heads=2, head_dim=32)
    else:
        raise ValueError(name)
    spec['head_counts'] = [spec['heads']] * spec['depth']
    # Large feasible caps permit matching the same compute while changing depth/embedding.
    spec['widths'] = [1024] * spec['depth']
    candidate = build(spec)
    spec['widths'] = allocate(candidate, np.ones(spec['depth']), target_macs)
    return spec


@torch.no_grad()
def factorize(model, target_macs):
    base_cost = cost(model)['macs']
    working = copy.deepcopy(model).cpu()
    entries = []
    spatial_sizes = {}
    handles = []
    for name, module in working.named_modules():
        if name.startswith('blocks.') and isinstance(module, (nn.Conv2d, nn.Linear)):
            def capture(layer, inputs, output, name=name):
                spatial_sizes[name] = output.numel() // (layer.out_channels if isinstance(layer, nn.Conv2d) else layer.out_features)
            handles.append(module.register_forward_hook(capture))
    working.eval()
    working(torch.zeros(1, 3, 32, 32))
    for handle in handles:
        handle.remove()
    for name, module in working.named_modules():
        if name not in spatial_sizes:
            continue
        weight = module.weight.detach().reshape(module.weight.shape[0], -1).double()
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        out, inp = weight.shape
        original_cost = spatial_sizes[name] * out * inp
        per_rank = spatial_sizes[name] * (out + inp)
        cap = min(len(s), (original_cost - 1) // per_rank)
        if cap < 1:
            continue
        entries.append({'name': name, 'source': module, 'u': u, 's': s, 'vh': vh,
                        'energy': s.square().numpy(), 'cost': original_cost,
                        'per_rank': per_rank, 'cap': cap, 'rank': None})
    current = base_cost
    while current > target_macs:
        choices = []
        for i, entry in enumerate(entries):
            rank = entry['rank']
            if rank == 1:
                continue
            if rank is None:
                next_rank = entry['cap']
                saving = entry['cost'] - next_rank * entry['per_rank']
                error = entry['energy'][next_rank:].sum()
            else:
                next_rank, saving = rank - 1, entry['per_rank']
                error = entry['energy'][next_rank]
            normalized_error = error / max(1e-30, entry['energy'].sum())
            choices.append((normalized_error / saving, i, next_rank, saving))
        if not choices:
            raise RuntimeError('SVD target is infeasible')
        _, index, next_rank, saving = min(choices)
        entries[index]['rank'] = int(next_rank)
        current -= int(saving)
    factors = {}
    for entry in entries:
        rank = entry['rank']
        if rank is None:
            continue
        old = entry['source']
        factored = FactorConv(old, rank) if isinstance(old, nn.Conv2d) else FactorLinear(old, rank)
        a = entry['vh'][:rank]
        b = entry['u'][:, :rank] * entry['s'][:rank]
        factored.a.weight.copy_(a.reshape_as(factored.a.weight).float())
        factored.b.weight.copy_(b.reshape_as(factored.b.weight).float())
        if old.bias is not None:
            factored.b.bias.copy_(old.bias)
        replace_module(working, entry['name'], factored)
        factors[entry['name']] = rank
    working.spec['factors'] = factors
    measured = cost(working)['macs']
    assert measured == current, (measured, current)
    assert measured <= target_macs
    return to_gpu(working), {'factor_ranks': factors, 'target_macs': target_macs,
                            'actual_macs': measured, 'budget_gap_fraction': (target_macs - measured) / base_cost}
