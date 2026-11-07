"""Execute the frozen pruning, recovery, scratch and architecture experiments."""
import argparse
import copy
import json
import time
import traceback

import numpy as np
import torch

from compress import allocate, architectural_variant, factorize, prune, prune_heads, random_widths, slimming_widths
from data import CFG, RUN, Data
from engine import evaluate, load_model, recalibrate, save_json, seed_all, setup, train
from manifest import jobs
from measures import all_scores, collect_features, descending, spectral_description
from models import build, cost, sparse_parameters


def complete(job):
    return (RUN / 'models' / job['task'] / 'train_done.json').exists()


def fit_pruned(model, data, job, details=None):
    directory = RUN / 'models' / job['task']
    if complete(job):
        return load_model(directory / 'best.pt')
    before = evaluate(model, data)
    recalibrate(model, data)
    after = evaluate(model, data)
    save_json(directory / 'pruning.json', {'before': before, 'after_bn': after,
                                          'architecture': model.spec, **(details or {})})
    return train(model, data, directory, {**job, 'train_seed': 100000 + job['model_seed']},
                 job['epochs'], ft=True)


def scratch(architecture, data, job):
    if complete(job):
        return
    seed_all(30000 + job['model_seed'])
    train(build(architecture), data, RUN / 'models' / job['task'],
          {**job, 'train_seed': 30000 + job['model_seed']}, job['epochs'])


def diagnostics(base, data, directory, scores, arrays, seed):
    orders = [r['qr_order'] for r in scores['layers']]
    fit_path = directory / 'fit_diagnostic.json'
    if not fit_path.exists():
        save_json(fit_path, {'train': evaluate(base, data, 'train'),
                            'validation': evaluate(base, data, 'val')})
    path = directory / 'sensitivity.json'
    if not path.exists():
        dense = copy.deepcopy(base)
        dense_before = evaluate(dense, data)
        recalibrate(dense, data)
        dense_after = evaluate(dense, data)
        del dense
        rows = []
        for layer in range(len(base.blocks)):
            for fraction in (0.25, 0.5):
                widths = list(base.spec['widths'])
                old = widths[layer]
                widths[layer] = int(old * (1 - fraction))
                changed, masks = prune(base, widths, orders)
                before = evaluate(changed, data)
                recalibrate(changed, data)
                after = evaluate(changed, data)
                rows.append({'layer': layer, 'fraction': fraction, 'old_width': old,
                             'new_width': widths[layer], 'erank_sigma': scores['layers'][layer]['erank_sigma'],
                             'erank_cov': scores['layers'][layer]['erank_cov'],
                             'normalized_sigma': scores['layers'][layer]['normalized_sigma'],
                             'normalized_cov': scores['layers'][layer]['normalized_cov'],
                             'before': before, 'after_bn': after,
                             'loss_increase': after['loss'] - dense_after['loss'],
                             'accuracy_drop': dense_after['accuracy'] - after['accuracy']})
                del changed
        save_json(path, {'dense_before': dense_before, 'dense_after_bn': dense_after, 'rows': rows})
    path = directory / 'rescaling.json'
    if not path.exists():
        x, _ = next(data.batches('calibration', batch_size=128))
        base.eval()
        with torch.inference_mode():
            reference = base(x)
        records = []
        for layer in (0, len(base.blocks) // 2, len(base.blocks) - 1):
            changed = copy.deepcopy(base).eval()
            scale = np.exp(np.random.default_rng(17070 + seed + layer).uniform(-2.3, 2.3, base.spec['widths'][layer]))
            multiplier = torch.tensor(scale, device='cuda', dtype=torch.float32)
            with torch.inference_mode():
                block = changed.blocks[layer]
                if base.spec['family'] == 'resnet20':
                    block.bn1.weight.mul_(multiplier)
                    block.bn1.bias.mul_(multiplier)
                    block.conv2.weight.div_(multiplier[None, :, None, None])
                else:
                    block.mlp.gate.mul_(multiplier)
                    block.mlp.fc2.weight.div_(multiplier[None, :])
                actual = changed(x)
                error = float((actual - reference).abs().max())
                if not torch.allclose(actual, reference, atol=2e-4, rtol=2e-4):
                    raise RuntimeError(f'Reparameterization changed outputs: {error}')
            modified = spectral_description(arrays[layer] * scale, with_order=False)
            records.append({'layer': layer, 'original_sigma': scores['layers'][layer]['erank_sigma'],
                            'original_cov': scores['layers'][layer]['erank_cov'],
                            'rescaled_sigma': modified['erank_sigma'], 'rescaled_cov': modified['erank_cov'],
                            'max_abs_logit_difference': error})
            del changed
        save_json(path, records)


def seed_pool(family, dataset, seed, data, inventory):
    prefix = f'{family}_{dataset}_s{seed}'
    def job(suffix):
        return inventory[prefix + '_' + suffix]
    base = load_model(RUN / 'models' / job('dense')['task'] / 'best.pt')
    sparse = load_model(RUN / 'models' / job('sparse')['task'] / 'best.pt')
    directory = RUN / 'diagnostics' / prefix
    directory.mkdir(parents=True, exist_ok=True)
    print('SCORING', prefix, flush=True)
    cached_score_size = None
    if (directory / 'scores.json').exists():
        scores = json.loads((directory / 'scores.json').read_text())
        cached_score_size = scores.get('channel_score_images', 512)
    if cached_score_size == CFG['score_images']:
        arrays, _ = collect_features(base, data)
    else:
        if (directory / 'scores.json').exists():
            save_json(directory / 'scores_before_equal_calibration.json', scores)
        scores, arrays = all_scores(base, data)
        save_json(directory / 'scores.json', scores)
    diagnostics(base, data, directory, scores, arrays, seed)
    del arrays
    full_cost = cost(base)['macs']
    orders = {'qr': [r['qr_order'] for r in scores['layers']]}
    orders.update({method: descending(scores[method]) for method in ('l1', 'taylor', 'hrank')})
    gamma = [p.detach().abs().cpu().numpy() for p in sparse_parameters(sparse)]
    gamma_orders = descending(gamma)
    save_json(directory / 'sparse_scores.json', {'scores': [g.tolist() for g in gamma], 'orders': gamma_orders})
    architectures, chosen = {}, {}
    for budget in CFG['budgets']:
        target = full_cost * budget
        chosen[budget] = {method: allocate(base, profile, target) for method, profile in (
            ('uniform_qr', base.spec['widths']),
            ('erank_sigma', [r['erank_sigma'] for r in scores['layers']]),
            ('erank_cov', [r['erank_cov'] for r in scores['layers']]))}
        chosen[budget]['random_width'] = random_widths(base, chosen[budget]['erank_sigma'], seed * 1000 + round(100 * budget))
        chosen[budget]['slimming'] = slimming_widths(sparse, gamma, target)
        for method in CFG['main_methods']:
            current_job = job(f'ft_b{round(100 * budget)}_{method}')
            if complete(current_job):
                architectures[(budget, method)] = json.loads((RUN / 'models' / current_job['task'] / 'train_done.json').read_text())['architecture']
                continue
            if method == 'svd':
                compressed, details = factorize(base, target)
            else:
                reference = sparse if method == 'slimming' else base
                widths = chosen[budget].get(method, chosen[budget]['uniform_qr'])
                order = gamma_orders if method == 'slimming' else orders.get(method, orders['qr'])
                compressed, masks = prune(reference, widths, order)
                details = {'channels': masks, 'target_macs': target,
                           'allocation': 'uniform' if method in ('l1', 'taylor', 'hrank') else method,
                           'selector': method if method in ('l1', 'taylor', 'hrank', 'slimming') else 'qr'}
            architectures[(budget, method)] = copy.deepcopy(compressed.spec)
            fit_pruned(compressed, data, current_job, details)
            del compressed
    for method in ('l1', 'taylor'):
        current_job = job(f'ablation_b75_erank_{method}')
        if not complete(current_job):
            compressed, masks = prune(base, chosen[0.75]['erank_sigma'], orders[method])
            fit_pruned(compressed, data, current_job, {'channels': masks, 'selector': method,
                                                      'allocation': 'erank_sigma'})
            del compressed
    if not complete(job('dense_ft')):
        train(copy.deepcopy(base), data, RUN / 'models' / job('dense_ft')['task'],
              {**job('dense_ft'), 'train_seed': 100000 + seed}, CFG['finetune_epochs'], ft=True)
    if not complete(job('sparse_uniform_ft')):
        compressed, masks = prune(sparse, chosen[0.75]['uniform_qr'], gamma_orders)
        fit_pruned(compressed, data, job('sparse_uniform_ft'), {'channels': masks, 'source': 'sparse'})
        del compressed
    for method in CFG['scratch_methods']:
        scratch(architectures[(0.75, method)], data, job(f'scratch_b75_{method}'))
    current = copy.deepcopy(base)
    trajectory = []
    for iteration in range(1, CFG['iterative_rounds'] + 1):
        current_job = job(f'iter_r{iteration}')
        if complete(current_job):
            current = load_model(RUN / 'models' / current_job['task'] / 'best.pt')
        else:
            arrays, _ = collect_features(current, data)
            ranks = [spectral_description(array) for array in arrays]
            minimum = 4 if family == 'resnet20' else 8
            widths = np.clip(np.ceil([r['erank_sigma'] for r in ranks]).astype(int),
                             minimum, current.spec['widths']).tolist()
            compressed, masks = prune(current, widths, [r['qr_order'] for r in ranks])
            prior = current.spec['widths']
            current = fit_pruned(compressed, data, current_job, {'previous_widths': prior,
                       'channels': masks, 'ranks': ranks, 'stable_widths': prior == widths})
            del arrays, compressed
        trajectory.append({'round': iteration, 'widths': current.spec['widths'], **cost(current)})
    save_json(directory / 'iterative_trajectory.json', trajectory)
    del current
    if family == 'mini_vit':
        for variant in ('depth4_dim96', 'depth8_dim64'):
            current_job = job(f'architecture_{variant}')
            if not complete(current_job):
                architecture = architectural_variant(base, variant, full_cost * 0.75)
                scratch(architecture, data, current_job)
        for method in ('attention_only', 'joint_heads_mlp'):
            current_job = job(f'architecture_ft_{method}')
            if complete(current_job):
                architecture = json.loads((RUN / 'models' / current_job['task'] / 'train_done.json').read_text())['architecture']
            else:
                if method == 'attention_only':
                    n, d = 64, base.spec['dim']
                    cost_per_head = 4 * n * d * base.spec['head_dim'] + 2 * n * n * base.spec['head_dim']
                    counts = [base.spec['heads']] * len(base.blocks)
                    current_cost, next_layer = full_cost, len(counts) - 1
                    while current_cost > full_cost * 0.75 and max(counts) > 1:
                        if counts[next_layer] > 1:
                            counts[next_layer] -= 1
                            current_cost -= cost_per_head
                        next_layer = (next_layer - 1) % len(counts)
                    compressed, heads = prune_heads(base, counts)
                else:
                    compressed, heads = prune_heads(base, [2] * len(base.blocks))
                    arrays, _ = collect_features(compressed, data)
                    ranks = [spectral_description(array) for array in arrays]
                    widths = allocate(compressed, [r['erank_sigma'] for r in ranks], full_cost * 0.75)
                    compressed, _ = prune(compressed, widths, [r['qr_order'] for r in ranks])
                    del arrays
                architecture = copy.deepcopy(compressed.spec)
                fit_pruned(compressed, data, current_job, {'kept_heads': heads, 'actual_cost': cost(compressed),
                                                           'dense_macs': full_cost})
                del compressed
            scratch(architecture, data, job(f'architecture_scratch_{method}'))
    del base, sparse
    torch.cuda.empty_cache()


def run(family, dataset):
    setup()
    data = Data(dataset)
    inventory = {j['task']: j for j in jobs() if j['family'] == family and j['dataset'] == dataset}
    for seed in CFG['seeds']:
        seed_pool(family, dataset, seed, data, inventory)
    missing = [name for name, j in inventory.items() if not complete(j)]
    if missing:
        raise RuntimeError(f'Unfinished inventory: {missing}')
    save_json(RUN / 'status' / f'{family}_{dataset}.json', {'state': 'trainings_complete',
                                                       'completed': len(inventory), 'updated_at': time.time()})
    print('GROUP COMPLETE', family, dataset, len(inventory), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('family')
    parser.add_argument('dataset')
    args = parser.parse_args()
    try:
        run(args.family, args.dataset)
    except Exception:
        save_json(RUN / 'failures' / f'{args.family}_{args.dataset}.json', {'phase': 'pool', 'traceback': traceback.format_exc()})
        raise
