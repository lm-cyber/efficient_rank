"""Reports are derived from completed runs, stored predictions and actual timing events."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t

from data import CFG, RUN
from manifest import jobs


def read(path):
    return json.loads(Path(path).read_text())


def markdown(frame, digits=3):
    def value(item):
        if isinstance(item, (float, np.floating)):
            return f'{item:.{digits}f}' if np.isfinite(item) else '—'
        return str(item)
    rows = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
            '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    rows += ['| ' + ' | '.join(value(item) for item in row) + ' |' for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(rows)


def seed_intervals(values):
    values = np.asarray(values, dtype=float)
    n, mean, sd = len(values), float(values.mean()), float(values.std(ddof=1))
    # Three rounded accuracies can have identical paired differences. A zero
    # estimated variance does not establish a zero-width population interval.
    estimable = n > 1 and sd > 1e-10
    half = float(t.ppf(.975, n - 1)) * sd / np.sqrt(n) if estimable else np.nan
    adjusted = float(t.ppf(1 - .05 / (2 * 24), n - 1)) * sd / np.sqrt(n) if estimable else np.nan
    return n, mean, sd, half, adjusted, estimable


def make_results():
    rows, timings, trajectories, sensitivities, scales = [], [], [], [], []
    for job in jobs():
        folder = RUN / 'models' / job['task']
        if not (folder / 'test.json').exists():
            raise RuntimeError(f'Cannot report unfinished task: {job["task"]}')
        done, test = read(folder / 'train_done.json'), read(folder / 'test.json')
        prediction = np.load(folder / 'test_predictions.npz')['predictions']
        data_folder = Path(__file__).resolve().parents[1] / 'data' / ('cifar100' if job['dataset'] == 'cifar100' else '')
        labels = np.load(data_folder / 'test_labels.npy')
        assert prediction.shape == (10000,) and test['n'] == 10000
        assert abs(100 * np.mean(prediction == labels) - test['accuracy']) < 1e-10
        record = {**job, 'params': done['params'], 'macs': done['macs'], 'flops': done['flops'],
                  'test_accuracy': test['accuracy'], 'test_loss': test['loss'],
                  'best_val_accuracy': done['best_val_accuracy'], 'best_epoch': done['best_epoch'],
                  'training_seconds': done['training_seconds'], 'widths': json.dumps(done['architecture']['widths']),
                  'architecture': json.dumps(done['architecture'])}
        if (folder / 'pruning.json').exists():
            pruning = read(folder / 'pruning.json')
            record.update(raw_val_accuracy=pruning['before']['accuracy'],
                          bn_val_accuracy=pruning['after_bn']['accuracy'])
        for batch in CFG['latency_batches']:
            series, round_medians, wall = [], [], []
            for repetition in range(CFG['latency_rounds']):
                path = RUN / 'latency' / f'{job["task"]}_r{repetition}.json'
                if not path.exists():
                    raise RuntimeError(f'Latency not complete: {path}')
                timing = next(m for m in read(path)['measurements'] if m['batch'] == batch)
                series.extend(timing['milliseconds'])
                round_medians.append(timing['median_ms'])
                wall.append(timing['wall_mean_ms'])
                timings.append({**job, 'batch': batch, 'round': repetition,
                                **{k: v for k, v in timing.items() if k != 'milliseconds'}})
            record[f'latency_b{batch}_ms'] = float(np.median(series))
            record[f'latency_b{batch}_q25'] = float(np.quantile(series, .25))
            record[f'latency_b{batch}_q75'] = float(np.quantile(series, .75))
            record[f'latency_b{batch}_round_sd'] = float(np.std(round_medians, ddof=1))
            record[f'images_s_b{batch}'] = float(1000 * batch / np.median(series))
            record[f'wall_latency_b{batch}_ms'] = float(np.mean(wall))
            record[f'wall_images_s_b{batch}'] = float(1000 * batch / np.mean(wall))
        rows.append(record)
        for path in (folder / 'rank_dynamics').glob('e*.json'):
            snapshot = read(path)
            for layer, values in enumerate(snapshot['layers']):
                trajectories.append({**job, 'epoch': snapshot['epoch'], 'layer': layer,
                                     **{k: v for k, v in values.items() if k != 'singular_values'}})
    result = pd.DataFrame(rows)
    references = result[result.phase == 'base'].set_index(['family', 'dataset', 'model_seed'])
    for index, row in result.iterrows():
        original = references.loc[(row.family, row.dataset, row.model_seed)]
        result.loc[index, 'actual_flops_retained'] = row.flops / original.flops
        result.loc[index, 'accuracy_change_from_dense_pp'] = row.test_accuracy - original.test_accuracy
        for batch in CFG['latency_batches']:
            result.loc[index, f'speedup_b{batch}'] = original[f'latency_b{batch}_ms'] / row[f'latency_b{batch}_ms']
    result.to_csv(RUN / 'results.csv', index=False)
    pd.DataFrame(timings).to_csv(RUN / 'latency_rounds.csv', index=False)
    pd.DataFrame(trajectories).to_csv(RUN / 'rank_dynamics.csv', index=False)
    for family in CFG['families']:
        for dataset in CFG['datasets']:
            for seed in CFG['seeds']:
                folder = RUN / 'diagnostics' / f'{family}_{dataset}_s{seed}'
                key = {'family': family, 'dataset': dataset, 'model_seed': seed}
                for row in read(folder / 'sensitivity.json')['rows']:
                    sensitivities.append({**key, **{k: v for k, v in row.items() if not isinstance(v, dict)}})
                scales.extend({**key, **row} for row in read(folder / 'rescaling.json'))
    pd.DataFrame(sensitivities).to_csv(RUN / 'sensitivity.csv', index=False)
    pd.DataFrame(scales).to_csv(RUN / 'rescaling.csv', index=False)
    return result, pd.DataFrame(trajectories), pd.DataFrame(sensitivities)


def primary_comparisons(results):
    main = results[results.phase == 'main']
    records = []
    for (family, dataset, budget), part in main.groupby(['family', 'dataset', 'budget']):
        reference = part[part.method == 'uniform_qr'].set_index('model_seed')
        for method in ('erank_sigma', 'erank_cov'):
            current = part[part.method == method].set_index('model_seed')
            paired = (current.test_accuracy - reference.test_accuracy).sort_index()
            n, mean, sd, half, adjusted, estimable = seed_intervals(paired)
            assert n == 3
            folder = Path(__file__).resolve().parents[1] / 'data' / ('cifar100' if dataset == 'cifar100' else '')
            labels = np.load(folder / 'test_labels.npy')
            differences = []
            for seed in sorted(current.index):
                candidate = np.load(RUN / 'models' / current.loc[seed, 'task'] / 'test_predictions.npz')['predictions']
                baseline = np.load(RUN / 'models' / reference.loc[seed, 'task'] / 'test_predictions.npz')['predictions']
                differences.append((candidate == labels).astype(np.int8) - (baseline == labels).astype(np.int8))
            image_low, image_high = paired_image_interval(np.asarray(differences), labels)
            records.append({'family': family, 'dataset': dataset, 'budget': budget, 'method': method,
                            'difference_pp': mean, 'seed_sd': sd, 'n': n,
                            'seed_interval_estimable': estimable,
                            'ci95_low': mean - half, 'ci95_high': mean + half,
                            'bonferroni24_low': mean - adjusted, 'bonferroni24_high': mean + adjusted,
                            'conditional_image_ci95_low': image_low, 'conditional_image_ci95_high': image_high,
                            'paired_seed_differences': json.dumps(paired.to_dict())})
    frame = pd.DataFrame(records)
    frame.to_csv(RUN / 'primary_comparisons.csv', index=False)
    return frame


def paired_image_interval(differences, labels):
    """Class-stratified paired bootstrap, conditional on the realized trained models.

    Multinomial sampling of the discrete per-image differences is exactly equivalent
    to resampling image indices, while avoiding enormous temporary index matrices.
    """
    n_models, n_images = differences.shape
    signed = differences.sum(axis=0)
    rng = np.random.default_rng(11060)
    bootstrap_sum = np.zeros(CFG['bootstrap_repeats'], dtype=float)
    for label in np.unique(labels):
        values, counts = np.unique(signed[labels == label], return_counts=True)
        draws = rng.multinomial(int(counts.sum()), counts / counts.sum(), size=CFG['bootstrap_repeats'])
        bootstrap_sum += draws @ values
    interval = np.quantile(100 * bootstrap_sum / (n_models * n_images), [.025, .975])
    return float(interval[0]), float(interval[1])


def write_findings(results, summary, primary, architecture_summary, scale_summary):
    rows = []
    for family in CFG['families']:
        for dataset in CFG['datasets']:
            baseline = results[(results.family == family) & (results.dataset == dataset) &
                               (results.phase == 'base')].test_accuracy
            row = {'model': family, 'dataset': dataset,
                   'dense': f'{baseline.mean():.2f} ± {baseline.std(ddof=1):.2f}'}
            part = summary[(summary.family == family) & (summary.dataset == dataset) &
                           (summary.budget == .75)].set_index('method')
            for method in ('uniform_qr', 'erank_sigma', 'erank_cov'):
                item = part.loc[method]
                row[method] = f'{item.accuracy_mean:.2f} ± {item.accuracy_sd:.2f}'
            rows.append(row)
    positive = int((primary.bonferroni24_low > 0).sum())
    negative = int((primary.bonferroni24_high < 0).sum())
    unestimable = int((~primary.seed_interval_estimable).sum())
    at75 = primary[primary.budget == .75][['family', 'dataset', 'method', 'difference_pp',
                                         'bonferroni24_low', 'bonferroni24_high', 'seed_interval_estimable']]
    timings = summary[(summary.budget == .75) &
                      summary.method.isin(['uniform_qr', 'erank_sigma', 'erank_cov', 'svd'])][
                          ['family', 'dataset', 'method', 'speedup_b1', 'speedup_b128']]
    rank_snapshots = len(list((RUN / 'models').glob('*/rank_dynamics/e*.json')))
    timing_series = len(list((RUN / 'latency').glob('*_r*.json')))
    lines = ['# Результаты расширенного пула', '',
        f'Выполнены **{len(results)} обучений**, **{int(results.epochs.sum()):,} эпох**, '
        f'**{len(results)} test-оценок**, **{rank_snapshots} снимков динамики ранга** и '
        f'**{timing_series} серий измерения скорости** на GPU 0.', '',
        'ResNet-20 и Mini-ViT проверены на CIFAR-10/CIFAR-100, по трём seed. Сопоставлены Uniform, '
        'два определения ERank, случайные ширины, L1, Taylor, HRank, Slimming и весовая SVD. '
        'Для ViT HRank и gate-slimming являются адаптациями. '
        'Все методы и бюджеты приведены в [полном отчёте](report.md).', '',
        '## ERank и равномерное сужение', '',
        'Test accuracy при сохранении 75% FLOPs: среднее ± SD между seed. Dense имеет полный бюджет; '
        'остальные модели прошли 20 эпох fine-tuning. Абсолютные значения относятся к разбиению v2.', '',
        markdown(pd.DataFrame(rows)), '',
        'Парная разница ERank − Uniform, в процентных пунктах, с интервалами Бонферрони по всем '
        '24 заранее выбранным сравнениям:', '', markdown(at75), '',
        f'Во всём наборе из 24 сравнений интервалы целиком выше нуля в **{positive}** случаях, '
        f'целиком ниже нуля в **{negative}**; неоцениваемых интервалов — **{unestimable}**. '
        'При трёх seed эти оценки предварительные. Интервал, содержащий ноль, не доказывает '
        'эквивалентности; результат одного бюджета не переносится на все архитектуры.', '',
        '## Реальная скорость', '',
        'Ниже ускорение относительно своей исходной dense-модели при бюджете 75% FLOPs. '
        'Значение больше 1 означает ускорение; меньше 1 — замедление. Batch 1 и 128 измерены '
        'на GPU отдельно от обучения, в eager/BF16, без загрузки данных и копирования входа с CPU.', '',
        markdown(timings), '',
        '## Архитектуры Mini-ViT', '',
        'Описательное сравнение обучения с нуля за 150 эпох: обычная сеть, найденные MLP-ширины, '
        'варианты глубины/embedding и удаления attention heads. Бюджеты и размер моделей указаны '
        'в таблице; dense имеет полный бюджет, attention-only может отличаться от целевых 75%.', '',
        markdown(architecture_summary), '',
        '![Архитектуры ViT](vit_architectures.png)', '',
        'ERank-scratch дополнительно требует обучения исходной сети для поиска ширин. '
        'Поэтому одинаковое число эпох обучения небольшой модели не означает одинаковых '
        'полных затрат на поиск и обучение.', '',
        '## Что ограничивает трактовку ранга', '',
        'Компенсированное масштабирование каналов сохраняет функцию сети с точностью вычислений. '
        'Таблица показывает максимальные изменения ранга и logits в предусмотренных вмешательствах:', '',
        markdown(scale_summary, digits=6), '',
        'Такое изменение ранга ограничивает его трактовку как необходимой ширины функции. '
        'Оно не исключает полезности ранга как эвристики в фиксированной процедуре обучения. '
        'Проверка этой полезности содержится в равнобюджетных сравнениях выше. '
        'Динамика ранга, послойная чувствительность, выбор каналов, sparse-контроли и итерации '
        'подробно разобраны в [отчёте](report.md).', '',
        'Сфера выводов: две небольшие модели, два CIFAR-набора, один split и фиксированные '
        'гиперпараметры. Это не проверка больших pretrained ViT и не доказательство универсальной '
        'полезности или бесполезности ERank.', '',
        '- [План и определения](../../experiment_pool_v2.md).',
        '- [Все отдельные результаты](results.csv).',
        '- [Основные сравнения](main_summary.csv).',
        '- [Парные интервалы и различия по seed](primary_comparisons.csv).',
        '- [Код и воспроизведение](../../experiments_v2/README.md).', '']
    (RUN / 'findings.md').write_text('\n'.join(lines))


def main():
    results, dynamics, sensitivity = make_results()
    primary = primary_comparisons(results)
    main = results[results.phase == 'main']
    summary = main.groupby(['family', 'dataset', 'budget', 'method']).agg(
        accuracy_mean=('test_accuracy', 'mean'), accuracy_sd=('test_accuracy', 'std'),
        params_mean=('params', 'mean'), flops_mean=('flops', 'mean'),
        actual_retained=('actual_flops_retained', 'mean'),
        latency_b1_ms=('latency_b1_ms', 'mean'), latency_b128_ms=('latency_b128_ms', 'mean'),
        speedup_b1=('speedup_b1', 'mean'), speedup_b128=('speedup_b128', 'mean')).reset_index()
    summary.to_csv(RUN / 'main_summary.csv', index=False)
    contrasts = []
    for row in main[main.method.isin(['erank_sigma', 'erank_cov', 'random_width'])].itertuples():
        reference = main[(main.family == row.family) & (main.dataset == row.dataset) &
                         (main.model_seed == row.model_seed) & (main.budget == row.budget) &
                         (main.method == 'uniform_qr')].iloc[0]
        a, b = np.asarray(json.loads(row.widths)), np.asarray(json.loads(reference.widths))
        contrasts.append({'family': row.family, 'dataset': row.dataset, 'model_seed': row.model_seed,
                          'budget': row.budget, 'method': row.method,
                          'width_l1_distance_to_uniform': int(np.abs(a-b).sum()),
                          'different_layers': int((a != b).sum()), 'same_architecture': bool(np.array_equal(a, b))})
    pd.DataFrame(contrasts).to_csv(RUN / 'architecture_contrasts.csv', index=False)
    additional = results[results.phase.isin(['scratch', 'control', 'selection_ablation', 'architecture_ft', 'architecture_scratch', 'iterative'])]
    additional_summary = additional.groupby(['family', 'dataset', 'phase', 'method']).agg(
        accuracy_mean=('test_accuracy', 'mean'), accuracy_sd=('test_accuracy', 'std'),
        actual_retained=('actual_flops_retained', 'mean'), params_mean=('params', 'mean'),
        speedup_b1=('speedup_b1', 'mean'), speedup_b128=('speedup_b128', 'mean')).reset_index()
    additional_summary.to_csv(RUN / 'additional_summary.csv', index=False)
    correlation_rows = []
    for key, part in sensitivity.groupby(['family', 'dataset', 'model_seed', 'fraction']):
        for method in ('sigma', 'cov'):
            statistic = spearmanr(part[f'normalized_{method}'], part.loss_increase).statistic
            correlation_rows.append(dict(zip(['family', 'dataset', 'model_seed', 'fraction'], key),
                                         method=method, spearman=float(statistic), blocks=len(part)))
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(RUN / 'sensitivity_correlations.csv', index=False)
    correlation_summary = correlations.groupby(['family', 'dataset', 'fraction', 'method']).agg(
        spearman_mean=('spearman', 'mean'), spearman_sd=('spearman', 'std'),
        seeds=('model_seed', 'nunique')).reset_index()
    scales = pd.read_csv(RUN / 'rescaling.csv')
    for method in ('sigma', 'cov'):
        scales[f'{method}_relative_change_pct'] = 100 * (
            scales[f'rescaled_{method}'] - scales[f'original_{method}']).abs() / scales[f'original_{method}']
    scale_summary = scales.groupby(['family', 'dataset']).agg(
        interventions=('layer', 'size'),
        max_sigma_change_pct=('sigma_relative_change_pct', 'max'),
        max_cov_change_pct=('cov_relative_change_pct', 'max'),
        max_logit_difference=('max_abs_logit_difference', 'max')).reset_index()
    score_rows = []
    fit_rows = []
    for family in CFG['families']:
        for dataset in CFG['datasets']:
            for seed in CFG['seeds']:
                scores = read(RUN / 'diagnostics' / f'{family}_{dataset}_s{seed}' / 'scores.json')
                fit = read(RUN / 'diagnostics' / f'{family}_{dataset}_s{seed}' / 'fit_diagnostic.json')
                fit_rows.append({'family': family, 'dataset': dataset, 'model_seed': seed,
                                 'train_accuracy': fit['train']['accuracy'],
                                 'validation_accuracy': fit['validation']['accuracy'],
                                 'gap_pp': fit['train']['accuracy'] - fit['validation']['accuracy']})
                for layer, values in enumerate(scores['layers']):
                    spatial = np.asarray(scores['hrank'][layer])
                    energy = np.square(values['singular_values'])
                    rank95 = int(np.searchsorted(np.cumsum(energy), .95 * energy.sum()) + 1) if energy.sum() else 0
                    score_rows.append({'family': family, 'dataset': dataset, 'model_seed': seed,
                         'layer': layer, **{k: v for k, v in values.items() if not isinstance(v, list)},
                         'energy_rank95': rank95,
                         'hrank_min': float(spatial.min()), 'hrank_max': float(spatial.max()),
                         'hrank_sd_across_channels': float(spatial.std()),
                         'hrank_unique_scores': len(np.unique(spatial)),
                         'hrank_fraction_at_modal_score': float(np.unique(spatial, return_counts=True)[1].max() / len(spatial))})
    pd.DataFrame(score_rows).to_csv(RUN / 'score_diagnostics.csv', index=False)
    fit_frame = pd.DataFrame(fit_rows)
    fit_frame.to_csv(RUN / 'fit_diagnostics.csv', index=False)
    colors = dict(zip(CFG['main_methods'], plt.get_cmap('tab10').colors))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (family, dataset) in zip(axes.ravel(), [(f, d) for f in CFG['families'] for d in CFG['datasets']]):
        part = summary[(summary.family == family) & (summary.dataset == dataset)]
        for method in CFG['main_methods']:
            selected = part[part.method == method].sort_values('flops_mean')
            ax.errorbar(selected.flops_mean / 1e6, selected.accuracy_mean, yerr=selected.accuracy_sd,
                        label=method, color=colors[method], marker='o', capsize=2, linewidth=1)
        dense = results[(results.family == family) & (results.dataset == dataset) & (results.phase == 'base')]
        ax.scatter([dense.flops.mean()/1e6], [dense.test_accuracy.mean()], marker='*', s=90, color='black', label='dense')
        ax.set(title=f'{family} / {dataset}', xlabel='MFLOPs / image', ylabel='Test accuracy (%)')
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle('Fine-tuned accuracy versus measured operation count (mean ± seed SD)')
    fig.tight_layout()
    fig.savefig(RUN / 'accuracy_flops.png', dpi=160)
    fig.savefig(RUN / 'accuracy_flops.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (family, dataset) in zip(axes.ravel(), [(f, d) for f in CFG['families'] for d in CFG['datasets']]):
        part = summary[(summary.family == family) & (summary.dataset == dataset)]
        for method in CFG['main_methods']:
            selected = part[part.method == method].sort_values('params_mean')
            ax.errorbar(selected.params_mean / 1e6, selected.accuracy_mean, yerr=selected.accuracy_sd,
                        label=method, color=colors[method], marker='o', capsize=2, linewidth=1)
        ax.set(title=f'{family} / {dataset}', xlabel='Parameters (millions)', ylabel='Test accuracy (%)')
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(RUN / 'accuracy_params.png', dpi=160)
    fig.savefig(RUN / 'accuracy_params.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (family, dataset) in zip(axes.ravel(), [(f, d) for f in CFG['families'] for d in CFG['datasets']]):
        part = summary[(summary.family == family) & (summary.dataset == dataset)]
        for method in CFG['main_methods']:
            selected = part[part.method == method].sort_values('latency_b128_ms')
            ax.plot(selected.latency_b128_ms, selected.accuracy_mean, marker='o', color=colors[method], label=method, linewidth=1)
        ax.set(title=f'{family} / {dataset}', xlabel='Measured GPU latency, batch 128 (ms)', ylabel='Test accuracy (%)')
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(RUN / 'accuracy_latency.png', dpi=160)
    fig.savefig(RUN / 'accuracy_latency.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (family, dataset) in zip(axes.ravel(), [(f, d) for f in CFG['families'] for d in CFG['datasets']]):
        part = dynamics[(dynamics.family == family) & (dynamics.dataset == dataset) & (dynamics.phase == 'base')]
        table = part.groupby(['layer', 'epoch']).normalized_sigma.mean().unstack('epoch')
        im = ax.imshow(table.values, aspect='auto', vmin=0, vmax=1, cmap='viridis')
        ax.set_xticks(range(len(table.columns)), [str(e) for e in table.columns])
        ax.set_yticks(range(len(table.index)), [str(i+1) for i in table.index])
        ax.set(title=f'{family} / {dataset}', xlabel='Training epoch', ylabel='Internal layer / MLP block')
    fig.subplots_adjust(left=.07, right=.83, bottom=.08, top=.94, hspace=.35, wspace=.25)
    color_axis = fig.add_axes([.88, .15, .018, .70])
    fig.colorbar(im, cax=color_axis, label='ERank-sigma / width (mean over 3 seeds)')
    fig.savefig(RUN / 'rank_dynamics.png', dpi=160)
    fig.savefig(RUN / 'rank_dynamics.pdf')
    plt.close(fig)
    arch = results[(results.family == 'mini_vit') & results.phase.isin(['base', 'scratch', 'architecture_scratch'])]
    arch_summary = arch.groupby(['dataset', 'phase', 'method']).agg(
        accuracy_mean=('test_accuracy', 'mean'), accuracy_sd=('test_accuracy', 'std'),
        flops_mean=('flops', 'mean'), params_mean=('params', 'mean')).reset_index()
    arch_summary.to_csv(RUN / 'vit_architecture_summary.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, dataset in zip(axes, CFG['datasets']):
        part = arch_summary[arch_summary.dataset == dataset].sort_values('accuracy_mean')
        names = [r.method for r in part.itertuples()]
        ax.errorbar(part.accuracy_mean, range(len(part)), xerr=part.accuracy_sd,
                    fmt='o', capsize=3, color='steelblue')
        ax.set_yticks(range(len(part)), names, fontsize=8)
        ax.set(title=f'Mini-ViT / {dataset}', xlabel='Test accuracy after 150 epochs (mean ± seed SD)')
        ax.set_xlim(max(0, part.accuracy_mean.min() - 5), min(100, part.accuracy_mean.max() + 2))
        ax.grid(axis='x', alpha=.2)
    fig.tight_layout()
    fig.savefig(RUN / 'vit_architectures.png', dpi=160)
    fig.savefig(RUN / 'vit_architectures.pdf')
    plt.close(fig)
    dense_summary = results[results.phase.isin(['base', 'sparse_base'])].groupby(['family', 'dataset', 'method']).agg(
        accuracy_mean=('test_accuracy', 'mean'), accuracy_sd=('test_accuracy', 'std'),
        params=('params', 'mean'), flops=('flops', 'mean')).reset_index()
    significant = primary[(primary.bonferroni24_low > 0) | (primary.bonferroni24_high < 0)]
    lines = ['# Расширенный экспериментальный пул ERank', '',
        f'Завершены **{len(results)} обучений**, **{int(results.epochs.sum()):,} эпох** и столько же оценок на официальных test-наборах по 10 000 изображений.', '',
        'План: [experiment_pool_v2.md](../../experiment_pool_v2.md). Все отдельные результаты: [results.csv](results.csv).', '',
        'Все обучения и измерения выполнены на GPU 0. Сумма длительностей отдельных обучений при параллельной работе процессов не равна общему времени выполнения.', '',
        '## Исходные модели и sparse-контроли', '', markdown(dense_summary), '',
        'Качество исходных моделей на train без аугментаций и на checkpoint-validation:', '',
        markdown(fit_frame), '',
        'Разрыв train/validation учитывается при интерпретации сжатия и сравнения семейств; высокая train accuracy сама по себе не означает хорошего обобщения.', '',
        '## Основные сравнения', '',
        'В таблицах mean/SD считаются по трём независимым seed. L1, Taylor и HRank используют ширины Uniform; Slimming использует отдельно обученную sparse-модель. В ViT HRank и gate-slimming являются явно обозначенными адаптациями.', '',
        'SVD сравнивается по фактическим FLOPs: дискретные ранги допускают небольшое отличие от номинального бюджета. FLOPs учитывают Conv, Linear и оба attention matmul, но не нормализации, GELU/ReLU и softmax.', '']
    for (family, dataset), part in summary.groupby(['family', 'dataset']):
        lines += [f'### {family} / {dataset}', '', markdown(part.drop(columns=['family', 'dataset'])), '']
    lines += ['![Accuracy versus FLOPs](accuracy_flops.png)', '',
        '![Accuracy versus parameters](accuracy_params.png)', '',
        '## Парные проверки ERank против Uniform', '',
        '24 первичных сравнения анализируются совместно. Приведены t-интервалы между seed с поправкой Бонферрони; при n=3 их надёжность ограничена. Дополнительные conditional_image интервалы получены парным bootstrap изображений с сохранением числа примеров каждого класса; они условны на уже обученных моделях и не заменяют вариативность обучения. Отсутствие значимости не означает эквивалентности.', '',
        'При одинаковых парных различиях по всем seed t-интервал не оценивается (поле seed_interval_estimable=False); наблюдаемый нулевой разброс не считается доказательством точного эффекта или эквивалентности.', '',
        markdown(primary.drop(columns=['paired_seed_differences', 'seed_sd', 'ci95_low', 'ci95_high'])), '',
        f'Число первичных интервалов, не содержащих ноль: **{len(significant)} из 24**. Знак и размер каждого эффекта указаны в таблице; вывод не переносится на другие семейства и данные.', '',
        '## Отбор каналов, обучение с нуля и итерации', '', markdown(additional_summary), '',
        'После определения архитектуры scratch использует 150 эпох небольшой модели, fine-tuning — 20. Если учитывать всю процедуру, ERank-scratch также требует предварительных 150 эпох исходной сети для поиска ширин: 150 dense + 150 small против 150 dense + 20 small у ERank fine-tuning. Uniform-scratch не требует такого поиска. Стоимость исходной сети может амортизироваться между несколькими бюджетами; равенство полного compute не заявляется. Sparse-методы используют собственные 150 эпох подготовки.', '',
        'Три итерации прямого ceil(ERank) не гарантируют сходимости: фактические ширины и бюджеты каждого раунда сохранены в results.csv и diagnostics/*/iterative_trajectory.json.', '',
        'Изменение глубины/embedding ViT проверяется с подобранной под общий бюджет шириной MLP. Attention-only использует целые heads и может отличаться от номинальных 75% FLOPs; его нельзя объявлять точным равнобюджетным сравнением.', '',
        'На следующем графике все показанные модели обучены с нуля 150 эпох. Dense имеет исходный бюджет; для остальных точные FLOPs и параметры сохранены в [vit_architecture_summary.csv](vit_architecture_summary.csv). Стоимость поиска архитектур ERank в эти 150 эпох не включена.', '',
        '![ViT architecture comparison](vit_architectures.png)', '',
        '## Скорость инференса', '',
        'Измерены batch 1 и 128: вход на GPU, eager/BF16, deterministic cuDNN без benchmark, 30 warmup и 100 повторов в каждой из пяти перемешанных серий. Сохранены CUDA event latency и синхронизированное wall-clock время серии. Обучающие процессы к этому моменту завершены; серии с посторонними процессами на GPU 0 не принимаются. Скорость не включает загрузку данных, CPU preprocessing или копирование CPU→GPU.', '',
        'Значение speedup > 1 означает ускорение, < 1 — замедление. Снижение FLOPs само по себе ускорения не гарантирует. Медианы, IQR, разброс медиан серий и исходные CUDA event timings сохранены.', '',
        '![Accuracy versus latency](accuracy_latency.png)', '',
        '## Динамика ранга', '',
        'Ранги измерены на фиксированных 256 calibration-изображениях на эпохах 0, 1, 10, 30, 60, 100, 150. Тепловые карты усредняют три seed исходных dense-моделей. Полные dense и sparse данные: [rank_dynamics.csv](rank_dynamics.csv).', '',
        '![Rank dynamics](rank_dynamics.png)', '',
        '## Проверка интерпретации ранга', '',
        'В каждом блоке отдельно удаляются 25% или 50% внутренних каналов по QR без дообучения; для CNN пересчитывается BN. Ниже описательные корреляции между нормированным рангом блока и ростом validation loss относительно исходной модели с тем же пересчётом BN. Среднее и SD относятся к трём seed; одна корреляция использует 9 блоков ResNet или 6 блоков ViT.', '',
        markdown(correlation_summary), '',
        'Эта корреляция не изолирует причинную роль ранга: блоки также различаются положением, шириной и функцией. Она проверяет ограниченную предсказательную связь при данном вмешательстве. Мгновенная чувствительность не заменяет качество после fine-tuning.', '',
        'В трёх выбранных блоках каждой исходной сети выполняется компенсированное масштабирование каналов: следующий линейный оператор получает обратный множитель. Проверяется сохранение logits на 128 calibration-изображениях. Ниже максимальные относительные изменения ранга и максимальное абсолютное изменение logits по этим вмешательствам; это описательная диагностика, не независимые повторные обучения.', '',
        markdown(scale_summary, digits=6), '',
        'Зависимость ранга от такого представления параметров ограничивает его интерпретацию как необходимой ширины функции сети. Энтропийный ранг не задаёт гарантированную ошибку усечения и не указывает, какие координатные каналы следует оставить. Весовая SVD в этом пуле также не является проверкой PCA/SVD-проекции самих активаций.', '',
        '## Ограничения', '',
        '- Три seed дают предварительную статистику. Изображения, слои, эпохи и повторы тайминга не подменяют независимые обучения.',
        '- Использованы один фиксированный split, одна общая калибровочная подвыборка и один набор гиперпараметров на семейство. Неопределённость по другим разбиениям и настройкам не измерена.',
        '- Результаты относятся к двум небольшим моделям и двум CIFAR-наборам; это не проверка больших pretrained ViT или языковых моделей.',
        '- CNN и ViT обучались с различными подходящими оптимизаторами и регуляризацией; сравнение между ними не изолирует только архитектуру.',
        '- HRank на ViT использует patch-grid карты, gate-slimming — обучаемые множители MLP. Это адаптации, а не неизменённые оригинальные алгоритмы.',
        '- CIFAR-10 test уже анализировался в v1. V2-план фиксировался перед новой оценкой; CIFAR-100 является дополнительной проверкой.',
        '- Источники методов, определения бюджетов и все режимы обучения зафиксированы в плане.', '',
        '## Артефакты', '',
        '- `models/`: best/last checkpoints, архитектуры, выбранные каналы, эпохи, test и предсказания.',
        '- `latency/`: все отдельные времена, порядок серий и состояние GPU.',
        '- `rank_dynamics.csv`, `sensitivity.csv`, `rescaling.csv`: динамика и диагностические вмешательства.',
        '- `planned_runs.csv`, `config.json`, `splits/`, `frozen_test_manifest.json`: протокол и воспроизводимость.', '']
    (RUN / 'report.md').write_text('\n'.join(lines))
    write_findings(results, summary, primary, arch_summary, scale_summary)
    # Present saved measurements in relative units and add descriptive baseline contrasts.
    from readable_report import main as render_percentages
    render_percentages()
    print('REPORT WRITTEN', RUN / 'report.md', flush=True)


if __name__ == '__main__':
    main()
