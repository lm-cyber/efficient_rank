"""Frozen inventory; every planned training must finish before the final test phase."""
from data import CFG, RUN
from engine import save_json


def jobs():
    result = []
    for family in CFG['families']:
        for dataset in CFG['datasets']:
            for seed in CFG['seeds']:
                prefix = f'{family}_{dataset}_s{seed}'
                def add(suffix, phase, method, epochs, budget=None):
                    result.append({'task': f'{prefix}_{suffix}', 'family': family, 'dataset': dataset,
                                   'model_seed': seed, 'phase': phase, 'method': method,
                                   'epochs': epochs, 'budget': budget})
                add('dense', 'base', 'dense', CFG['base_epochs'], 1.0)
                add('sparse', 'sparse_base', 'sparse', CFG['base_epochs'], 1.0)
                for budget in CFG['budgets']:
                    for method in CFG['main_methods']:
                        add(f'ft_b{round(100*budget)}_{method}', 'main', method, CFG['finetune_epochs'], budget)
                for method in ('dense_ft', 'sparse_uniform_ft'):
                    add(method, 'control', method, CFG['finetune_epochs'], 1.0 if method == 'dense_ft' else 0.75)
                for method in CFG['scratch_methods']:
                    add(f'scratch_b75_{method}', 'scratch', method, CFG['scratch_epochs'], 0.75)
                for method in ('erank_l1', 'erank_taylor'):
                    add(f'ablation_b75_{method}', 'selection_ablation', method, CFG['finetune_epochs'], 0.75)
                for iteration in range(1, CFG['iterative_rounds'] + 1):
                    add(f'iter_r{iteration}', 'iterative', f'erank_ceil_r{iteration}', CFG['finetune_epochs'])
                if family == 'mini_vit':
                    for variant in ('depth4_dim96', 'depth8_dim64'):
                        add(f'architecture_{variant}', 'architecture_scratch', variant, CFG['scratch_epochs'], 0.75)
                    for method in ('attention_only', 'joint_heads_mlp'):
                        add(f'architecture_ft_{method}', 'architecture_ft', method, CFG['finetune_epochs'], 0.75)
                        add(f'architecture_scratch_{method}', 'architecture_scratch', method, CFG['scratch_epochs'], 0.75)
    return result


def freeze():
    import csv
    import json
    existing = RUN / 'config.json'
    if existing.exists():
        assert json.loads(existing.read_text()) == CFG, 'Config changed after preregistration'
    save_json(existing, CFG)
    manifest = jobs()
    path = RUN / 'planned_runs.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    save_json(path, manifest)
    with (RUN / 'planned_runs.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f'Frozen {len(manifest)} trainings, {sum(j["epochs"] for j in manifest)} epochs', flush=True)


if __name__ == '__main__':
    freeze()
