"""Render percentage reports and descriptive ERank/baseline comparisons from saved results."""
import hashlib
import json
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'experiments_v2/config.json').read_text())
RUN = ROOT / 'runs' / CFG['protocol']
GROUPS = [(f, d) for f in CFG['families'] for d in CFG['datasets']]
ERANK = ['erank_sigma', 'erank_cov']
BASELINES = ['uniform_qr', 'random_width', 'l1', 'slimming', 'taylor', 'hrank', 'svd']
ORDER = ['dense', 'sparse', *CFG['main_methods']]
NAMES = {
    'dense': 'Исходная', 'sparse': 'Исходная + sparsity', 'uniform_qr': 'Uniform + QR',
    'erank_sigma': 'ERank-σ + QR', 'erank_cov': 'ERank-cov + QR', 'random_width': 'Random + QR',
    'l1': 'L1', 'slimming': 'Slimming', 'taylor': 'Taylor', 'hrank': 'HRank', 'svd': 'SVD весов',
    'dense_ft': 'Исходная + fine-tuning', 'sparse_uniform_ft': 'Sparse → Uniform',
    'erank_l1': 'Ширины ERank-σ + L1', 'erank_taylor': 'Ширины ERank-σ + Taylor',
    'erank_ceil_r1': 'ERank: итерация 1', 'erank_ceil_r2': 'ERank: итерация 2',
    'erank_ceil_r3': 'ERank: итерация 3', 'attention_only': 'Только attention heads',
    'joint_heads_mlp': 'Heads + MLP по ERank', 'depth4_dim96': '4 блока / dim 96',
    'depth8_dim64': '8 блоков / dim 64',
}
METRICS = ['accuracy_pct', 'quality_pct', 'accuracy_change_pct', 'loss_pct', 'params_pct',
           'flops_pct', 'latency_b1_pct', 'latency_b128_pct', 'speed_b1_change_pct', 'speed_b128_change_pct']
KEY = ['family', 'dataset', 'model_seed']
LEGEND = [
    'Исходная dense-модель того же семейства, датасета и seed — точка отсчёта. '
    'Сначала вычисляются отношения внутри каждого seed, затем среднее и SD по трём seed.',
    '',
    '| Показатель | Как читать |',
    '| --- | --- |',
    '| Качество, % | 100 × Accuracy / Accuracy исходной. 99% означает относительную потерю качества 1%. |',
    '| Accuracy, % | Обычная доля правильных ответов; сохранена рядом для ориентира. |',
    '| Loss, % | 100 × cross-entropy / cross-entropy исходной; меньше лучше. |',
    '| Параметры и FLOPs, % | Сколько осталось от исходной модели. 75% означает сокращение на 25%. |',
    '| Δскорости, % | 100 × (время исходной / время модели − 1). +10% — быстрее, −10% — медленнее. |',
    '| Время, % | 100 × время модели / время исходной. 110% — инференс длится на 10% дольше. |',
    '',
    'Проценты изменения accuracy отличаются от процентных пунктов: переход с 90% на 89,1% '
    'даёт −1% относительно прежней accuracy и −0,9 п.п. Скорость −10% соответствует времени '
    'примерно 111,1% от исходного. Отношения скорости и времени считаются отдельно до усреднения.',
]
COMPARISON_NOTE = (
    'Сравнения ERank со всеми бейзлайнами добавлены после получения результатов по запросу пользователя. '
    'Они описательные: среднее ± SD трёх парных различий по одинаковым seed, без новых заявлений '
    'о статистической значимости. Первоначальные 24 проверки ERank против Uniform сохранены отдельно. '
    'В таблицах ΔAccuracy = 100 × (Accuracy ERank / Accuracy бейзлайна − 1); плюс означает преимущество ERank. '
    'Для ресурсов точкой отсчёта служит указанный бейзлайн, а не dense.'
)
LIMITATIONS = [
    'Три seed, один split и одна калибровочная выборка дают ограниченную оценку устойчивости. '
    'SD — разброс обучений, а не доверительный интервал и не разброс отдельных изображений.',
    'L1, Taylor и HRank используют ширины Uniform. ERank определяет распределение ширин и использует QR. '
    'Поэтому сравнение с ними описывает полную процедуру; отдельные абляции ERank + L1/Taylor изолируют выбор каналов.',
    'Slimming использует отдельно обученную sparse-модель. Это другой способ подготовки весов; '
    'контроль Sparse → Uniform показан отдельно. На ViT HRank и gate-slimming являются адаптациями.',
    'Бюджеты совпадают по FLOPs, но не обязательно по параметрам. У SVD дискретные ранги дают небольшое '
    'отклонение FLOPs; фактическое отношение показано в сравнении. FLOPs считают Conv, Linear и attention matmul.',
    'Скорость относится к A100, eager/BF16, batch 1/128, deterministic cuDNN без benchmark. '
    'Вход уже на GPU; загрузка данных и копирование с CPU исключены. На другой реализации ускорение может отличаться.',
    'После поиска архитектуры scratch длится 150 эпох, fine-tuning — 20. ERank-scratch дополнительно требует '
    '150 эпох исходной сети для выбора ширин; Uniform-scratch такого поиска не требует. Полный compute не уравнен.',
    'Три итерации ceil(ERank) не доказывают сходимости или оптимальности. Нет парного итеративного Uniform-контроля.',
    'CNN и ViT используют разные оптимизаторы и регуляризацию. Различия между ними нельзя приписать только архитектуре. '
    'Выводы не переносятся автоматически на большие pretrained-модели; CIFAR-10 test уже анализировался в v1.',
]


def number(value, digits=2, signed=False):
    if not np.isfinite(value):
        return '—'
    value = 0.0 if abs(value) < .5 * 10 ** -digits else value
    spec = f'+.{digits}f' if signed and value else f'.{digits}f'
    return format(value, spec).replace('.', ',')


def mean_sd(mean, sd, signed=False):
    return f'{number(mean, signed=signed)} ± {number(sd)}'


def table(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return ''
    header = '| ' + ' | '.join(frame.columns) + ' |'
    body = ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)]
    return '\n'.join([header, '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |', *body])


def group_name(family, dataset):
    return f'{"ResNet-20" if family == "resnet20" else "Mini-ViT"} / {dataset.upper()}'


def name(method):
    return NAMES.get(method, method)


def row_name(row):
    if row.phase == 'scratch' and row.method in ['uniform_qr', *ERANK]:
        return {'uniform_qr': 'Uniform: MLP/каналы', 'erank_sigma': 'ERank-σ: MLP/каналы',
                'erank_cov': 'ERank-cov: MLP/каналы'}[row.method]
    return name(row.method)


def subset(frame, **values):
    mask = pd.Series(True, index=frame.index)
    for key, value in values.items():
        mask &= frame[key] == value
    return frame[mask]


def normalize(results):
    columns = ['test_accuracy', 'test_loss', 'params', 'flops', 'latency_b1_ms', 'latency_b128_ms']
    dense = results[results.phase == 'base'][KEY + columns]
    reference = dense.rename(columns={c: f'dense_{c}' for c in columns})
    out = results.merge(reference, on=KEY, how='left', validate='many_to_one', sort=False)
    assert len(out) == len(results) and out[[f'dense_{c}' for c in columns]].notna().all().all()
    out['accuracy_pct'] = out.test_accuracy
    for source, metric in [('test_accuracy', 'quality_pct'), ('test_loss', 'loss_pct'),
                           ('params', 'params_pct'), ('flops', 'flops_pct'),
                           ('latency_b1_ms', 'latency_b1_pct'), ('latency_b128_ms', 'latency_b128_pct')]:
        out[metric] = 100 * out[source] / out[f'dense_{source}']
    out['accuracy_change_pct'] = out.quality_pct - 100
    for batch in (1, 128):
        out[f'speed_b{batch}_change_pct'] = 100 * (
            out[f'dense_latency_b{batch}_ms'] / out[f'latency_b{batch}_ms'] - 1)
    return out


def summarize(frame):
    keys = ['family', 'dataset', 'phase', 'method', 'budget']
    aggregations = {f'{metric}_{stat}': (metric, stat) for metric in METRICS for stat in ('mean', 'std')}
    result = frame.groupby(keys, dropna=False).agg(n=('model_seed', 'nunique'), **aggregations).reset_index()
    assert (result.n == 3).all()
    return result


def ordered(frame):
    return frame.assign(_order=frame.method.map({m: i for i, m in enumerate(ORDER)}).fillna(100)).sort_values('_order')


def quality_table(part):
    return table([{
        'Метод': row_name(r),
        'Accuracy, %': mean_sd(r.accuracy_pct_mean, r.accuracy_pct_std),
        'Качество, % от исходной': mean_sd(r.quality_pct_mean, r.quality_pct_std),
        'Loss, % от исходной': mean_sd(r.loss_pct_mean, r.loss_pct_std),
    } for r in ordered(part).itertuples()])


def resource_table(part):
    return table([{
        'Метод': row_name(r), 'Параметры, %': number(r.params_pct_mean, 1),
        'FLOPs, %': number(r.flops_pct_mean), 'Δскорости, % (1)': number(r.speed_b1_change_pct_mean, 1, True),
        'Δскорости, % (128)': number(r.speed_b128_change_pct_mean, 1, True),
        'Время, % (128)': number(r.latency_b128_pct_mean, 1),
    } for r in ordered(part).itertuples()])


def compact_table(part):
    return table([{
        'Метод': row_name(r), 'Accuracy, %': number(r.accuracy_pct_mean),
        'Качество, %': mean_sd(r.quality_pct_mean, r.quality_pct_std),
        'Параметры, %': number(r.params_pct_mean, 1), 'FLOPs, %': number(r.flops_pct_mean),
        'Δскорости, % (128)': number(r.speed_b128_change_pct_mean, 1, True),
    } for r in ordered(part).itertuples()])


def compare_baselines(results):
    records = []
    main = results[results.phase == 'main']
    for (family, dataset, budget), part in main.groupby(['family', 'dataset', 'budget']):
        for method in ERANK:
            current = part[part.method == method].set_index('model_seed').sort_index()
            for baseline in BASELINES:
                ref = part[part.method == baseline].set_index('model_seed').sort_index()
                assert list(current.index) == list(ref.index) == sorted(CFG['seeds'])
                for seed in current.index:
                    a, b = current.loc[seed], ref.loc[seed]
                    row = {'family': family, 'dataset': dataset, 'budget': budget, 'method': method,
                           'baseline': baseline, 'model_seed': seed,
                           'accuracy_change_pct': 100 * (a.test_accuracy / b.test_accuracy - 1),
                           'accuracy_difference_pp': a.test_accuracy - b.test_accuracy,
                           'loss_pct_baseline': 100 * a.test_loss / b.test_loss,
                           'params_pct_baseline': 100 * a.params / b.params,
                           'flops_pct_baseline': 100 * a.flops / b.flops}
                    for batch in (1, 128):
                        ta, tb = a[f'latency_b{batch}_ms'], b[f'latency_b{batch}_ms']
                        row[f'latency_b{batch}_pct_baseline'] = 100 * ta / tb
                        row[f'speed_b{batch}_change_pct'] = 100 * (tb / ta - 1)
                    records.append(row)
    paired = pd.DataFrame(records)
    keys = ['family', 'dataset', 'budget', 'method', 'baseline']
    metrics = [c for c in paired if c not in keys + ['model_seed']]
    summary = paired.groupby(keys).agg(
        n=('model_seed', 'nunique'),
        **{f'{m}_{s}': (m, s) for m in metrics for s in ('mean', 'std')}).reset_index()
    assert len(summary) == 168 and (summary.n == 3).all() and len(paired) == 504
    return paired, summary


def comparison_quality(part):
    rows = []
    for baseline in BASELINES:
        row = {'Бейзлайн': name(baseline)}
        for method, title in [('erank_sigma', 'ERank-σ: ΔAccuracy, %'),
                              ('erank_cov', 'ERank-cov: ΔAccuracy, %')]:
            item = subset(part, baseline=baseline, method=method).iloc[0]
            row[title] = mean_sd(item.accuracy_change_pct_mean, item.accuracy_change_pct_std, True)
        rows.append(row)
    return table(rows)


def comparison_resources(part, method):
    rows = []
    for baseline in BASELINES:
        r = subset(part, baseline=baseline, method=method).iloc[0]
        rows.append({'Бейзлайн = 100%': name(baseline), 'Loss, %': number(r.loss_pct_baseline_mean),
                     'Параметры, %': number(r.params_pct_baseline_mean),
                     'FLOPs, %': number(r.flops_pct_baseline_mean),
                     'Δскорости, % (1)': number(r.speed_b1_change_pct_mean, 1, True),
                     'Δскорости, % (128)': number(r.speed_b128_change_pct_mean, 1, True)})
    return table(rows)


def save_figure(fig, stem):
    for suffix in ('png', 'pdf'):
        fig.savefig(RUN / f'{stem}.{suffix}', dpi=160)
    plt.close(fig)


def plots(summary, comparisons):
    colors = dict(zip(CFG['main_methods'], plt.get_cmap('tab10').colors))
    for stem, xkey, xlabel, original in [
        ('accuracy_flops', 'flops_pct_mean', 'FLOPs, % от исходной модели', 100),
        ('accuracy_params', 'params_pct_mean', 'Параметры, % от исходной модели', 100),
        ('accuracy_latency', 'speed_b128_change_pct_mean', 'Изменение скорости, % (batch 128)', 0),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, (family, dataset) in zip(axes.ravel(), GROUPS):
            part = subset(summary, family=family, dataset=dataset, phase='main')
            for method in CFG['main_methods']:
                selected = subset(part, method=method).sort_values(xkey)
                ax.errorbar(selected[xkey], selected.quality_pct_mean, yerr=selected.quality_pct_std,
                            label=name(method), color=colors[method], marker='o', capsize=2, linewidth=1)
            ax.scatter([original], [100], marker='*', s=85, color='black', label='Исходная')
            ax.set(title=group_name(family, dataset), xlabel=xlabel, ylabel='Accuracy, % от исходной')
            ax.yaxis.set_major_formatter(PercentFormatter(100))
            ax.xaxis.set_major_formatter(PercentFormatter(100))
            ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8, ncol=2)
        fig.suptitle('Fine-tuning: относительное качество (среднее ± SD, 3 seed)')
        fig.tight_layout()
        save_figure(fig, stem)
    dynamics = pd.read_csv(RUN / 'rank_dynamics.csv')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (family, dataset) in zip(axes.ravel(), GROUPS):
        part = subset(dynamics, family=family, dataset=dataset, phase='base')
        values = 100 * part.groupby(['layer', 'epoch']).normalized_sigma.mean().unstack('epoch')
        im = ax.imshow(values.values, aspect='auto', vmin=0, vmax=100, cmap='viridis')
        ax.set_xticks(range(len(values.columns)), [str(e) for e in values.columns])
        ax.set_yticks(range(len(values.index)), [str(i + 1) for i in values.index])
        ax.set(title=group_name(family, dataset), xlabel='Эпоха обучения', ylabel='Блок')
    fig.subplots_adjust(left=.07, right=.83, bottom=.08, top=.94, hspace=.35, wspace=.25)
    color_axis = fig.add_axes([.88, .15, .018, .70])
    fig.colorbar(im, cax=color_axis, label='ERank-σ / ширина, % (среднее по 3 seed)',
                 format=PercentFormatter(100))
    save_figure(fig, 'rank_dynamics')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, dataset in zip(axes, CFG['datasets']):
        part = subset(summary, family='mini_vit', dataset=dataset)
        part = part[part.phase.isin(['base', 'scratch', 'architecture_scratch'])].sort_values('quality_pct_mean')
        ax.errorbar(part.quality_pct_mean, range(len(part)), xerr=part.quality_pct_std,
                    fmt='o', capsize=3, color='steelblue')
        ax.set_yticks(range(len(part)), [row_name(r) for r in part.itertuples()], fontsize=9)
        ax.axvline(100, linestyle='--', color='gray', linewidth=1)
        ax.set(title=group_name('mini_vit', dataset), xlabel='Accuracy, % от исходной (среднее ± SD)')
        ax.xaxis.set_major_formatter(PercentFormatter(100))
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Архитектуры Mini-ViT: обучение с нуля, 150 эпох')
    fig.tight_layout()
    save_figure(fig, 'vit_architectures')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    column_keys = [(b, m) for b in (.9, .75, .6) for m in ERANK]
    for ax, (family, dataset) in zip(axes.ravel(), GROUPS):
        part = subset(comparisons, family=family, dataset=dataset)
        values = np.array([[subset(part, baseline=b, budget=budget, method=m).iloc[0].accuracy_change_pct_mean
                            for budget, m in column_keys] for b in BASELINES])
        signs = np.sign(np.round(values, 2))
        ax.imshow(signs, vmin=-1, vmax=1, cmap=ListedColormap(['#f4c4c4', '#f3f3f3', '#c9e7ce']), aspect='auto')
        for y, x in np.ndindex(values.shape):
            ax.text(x, y, number(values[y, x], signed=True), ha='center', va='center', fontsize=9)
        ax.set_xticks(range(6), [f'−{round(100*(1-b))}%\n{"σ" if m == "erank_sigma" else "cov"}' for b, m in column_keys])
        ax.set_yticks(range(len(BASELINES)), [name(b) for b in BASELINES], fontsize=9)
        ax.set(title=group_name(family, dataset), xlabel='Сокращение FLOPs и вариант ERank')
    fig.suptitle('ERank относительно бейзлайнов: ΔAccuracy, %\n'
                 'Цвет — знак округлённого среднего; не статистическая значимость', fontsize=12)
    fig.tight_layout()
    save_figure(fig, 'erank_vs_baselines')


def primary_text(primary):
    positive = int((primary.bonferroni24_low > 0).sum())
    negative = int((primary.bonferroni24_high < 0).sum())
    return (f'В первоначальных 24 сравнениях с поправкой Бонферрони: '
            f'{positive} подтверждённых преимуществ ERank и {negative} подтверждённых ухудшений. '
            'Отсутствие значимости не доказывает эквивалентности; мощность при трёх seed мала. '
            'Исходные интервалы в процентных пунктах и bootstrap изображений доступны в '
            '[primary_comparisons.csv](primary_comparisons.csv). Эти проверки не пересчитывались '
            'как новые проверки относительных процентов.')


def reports(summary, comparisons, primary):
    introduction = [
        'Все обучения, test-оценки и замеры инференса завершены на GPU 0.',
        'ResNet-20 и Mini-ViT × CIFAR-10/CIFAR-100; три seed; сокращение FLOPs на 10%, 25% и 40%.',
    ]
    leaders = []
    for family, dataset in GROUPS:
        candidates = subset(summary, family=family, dataset=dataset, phase='main', budget=.75)
        candidates = candidates[candidates.method.isin(BASELINES)]
        leader = candidates.loc[candidates.accuracy_pct_mean.idxmax()]
        row = {'Модель': group_name(family, dataset), 'Бейзлайн с наибольшей средней Accuracy': name(leader.method)}
        for method, column in [('erank_sigma', 'ERank-σ: ΔAccuracy, %'), ('erank_cov', 'ERank-cov: ΔAccuracy, %')]:
            effect = subset(comparisons, family=family, dataset=dataset, budget=.75,
                            method=method, baseline=leader.method).iloc[0]
            row[column] = mean_sd(effect.accuracy_change_pct_mean, effect.accuracy_change_pct_std, True)
        leaders.append(row)
    quick = ['# Результаты: проценты и сравнение с бейзлайнами', '', *introduction, '',
             '## ERank и лучший бейзлайн по средней Accuracy при −25% FLOPs', '',
             table(leaders), '',
             'Бейзлайн в этой короткой таблице выбран по средней Accuracy уже выполненных экспериментов. '
             'Это описательный ориентир, не статистически доказанный победитель. В таблицах ниже показаны '
             'все бейзлайны, включая случаи, где ERank выше по качеству. ΔAccuracy здесь — относительный процент, '
             'а не процентные пункты.', '',
             '[Сравнения всех бюджетов и ресурсов](erank_vs_baselines.md) · [Полный отчёт](report.md).', '',
             '## Как читать', '', *LEGEND, '',
             '## Основной срез: сокращение FLOPs на 25%', '',
             'Качество и ресурсы ниже нормированы на свою исходную модель. Скорость — изменение '
             'пропускной способности при batch 128; отрицательные значения означают замедление.', '']
    full = ['# Полный отчёт: относительные метрики', '', *introduction, '',
            '[Краткая сводка](findings.md) · [Все прямые сравнения ERank](erank_vs_baselines.md) · '
            '[План](../../experiment_pool_v2.md)', '',
            '## Единицы и точка отсчёта', '', *LEGEND, '',
            'В таблицах качества приведены среднее ± SD. В таблицах ресурсов — средние отношения. '
            'Для batch 1 и 128 используются одни и те же сохранённые замеры; новые обучения не запускались.', '',
            '## Исходные модели и sparse-контроли', '']
    for family, dataset in GROUPS:
        reference = subset(summary, family=family, dataset=dataset)
        reference = reference[reference.phase.isin(['base', 'sparse_base'])]
        full += [f'### {group_name(family, dataset)}', '', quality_table(reference), '', resource_table(reference), '']
    full += ['## Основные сравнения при одинаковом бюджете FLOPs', '',
             'Для ViT Slimming означает gate-slimming, HRank использует карты patch-grid. '
             'SVD может немного отклоняться от целевого бюджета; в таблицах приведён фактический процент.', '']
    for family, dataset in GROUPS:
        part = subset(summary, family=family, dataset=dataset, phase='main')
        dense = subset(summary, family=family, dataset=dataset, phase='base').iloc[0]
        base_note = f'Исходная Accuracy: {mean_sd(dense.accuracy_pct_mean, dense.accuracy_pct_std)}%.'
        quick += [f'### {group_name(family, dataset)}', '', base_note, '',
                  compact_table(subset(part, budget=.75)), '',
                  'Прямое сравнение качества ERank с каждым бейзлайном; плюс — лучше ERank:', '',
                  comparison_quality(subset(comparisons, family=family, dataset=dataset, budget=.75)), '']
        full += [f'### {group_name(family, dataset)}', '', base_note, '']
        for budget in (.9, .75, .6):
            full += [f'#### FLOPs: осталось {round(100*budget)}%, сокращение {round(100*(1-budget))}%', '',
                     quality_table(subset(part, budget=budget)), '',
                     resource_table(subset(part, budget=budget)), '',
                     'ERank против каждого бейзлайна: относительное изменение Accuracy, %.', '',
                     comparison_quality(subset(comparisons, family=family, dataset=dataset, budget=budget)), '']
    quick += ['## Что означает сравнение', '', COMPARISON_NOTE, '', primary_text(primary), '',
              '![ERank против всех бейзлайнов](erank_vs_baselines.png)', '',
              'Подробные сравнения параметров, FLOPs, loss и скорости относительно каждого бейзлайна: '
              '[erank_vs_baselines.md](erank_vs_baselines.md).', '']
    full += ['## Статистическая интерпретация', '', COMPARISON_NOTE, '', primary_text(primary), '',
             '![ERank против всех бейзлайнов](erank_vs_baselines.png)', '',
             '## Кривые качества и ресурсов', '',
             'На всех графиках качество нормировано на исходную модель внутри seed; '
             'звезда обозначает исходную модель. Проценты качества не являются обычной Accuracy.', '',
             '![Качество и FLOPs](accuracy_flops.png)', '',
             '![Качество и параметры](accuracy_params.png)', '',
             '![Качество и скорость](accuracy_latency.png)', '']
    architecture = summary[(summary.family == 'mini_vit') &
                           summary.phase.isin(['base', 'scratch', 'architecture_scratch'])]
    architecture_text = [
        'Все модели в этом разделе обучены с нуля 150 эпох. Исходная модель имеет полный бюджет. '
        'Для attention-only бюджет отличается от целевых 75% из-за дискретности числа heads; '
        'фактические проценты показаны. Различия средних описательные, не доказательство превосходства.', '',
    ]
    quick += ['## Зависимость Mini-ViT от архитектуры', '', *architecture_text]
    full += ['## Зависимость Mini-ViT от архитектуры', '', *architecture_text]
    for dataset in CFG['datasets']:
        section = [f'### {dataset.upper()}', '', compact_table(subset(architecture, dataset=dataset)), '']
        quick += section
        full += section
    quick += ['![Архитектуры ViT](vit_architectures.png)', '']
    full += ['![Архитектуры ViT](vit_architectures.png)', '',
             '## Обучение с нуля, выбор каналов, контроли и итерации', '']
    for phase, title in [('scratch', 'Обучение найденных ширин с нуля'),
                         ('selection_ablation', 'Одинаковые ширины ERank, разный выбор каналов'),
                         ('control', 'Контроли дообучения и sparse-модели'),
                         ('iterative', 'Три последовательные итерации ERank'),
                         ('architecture_ft', 'Изменение attention heads: fine-tuning')]:
        full += [f'### {title}', '']
        for family, dataset in GROUPS:
            part = subset(summary, family=family, dataset=dataset, phase=phase)
            if not part.empty:
                full += [f'#### {group_name(family, dataset)}', '', quality_table(part), '', resource_table(part), '']
    full += ['## Реальная скорость', '',
             'Проценты в таблицах рассчитаны из измеренных времён. Проведено пять перемешанных серий: '
             '30 warmup и 100 повторов на модель, batch 1 и 128. Сохранены CUDA events, wall-clock, '
             'IQR и разброс между сериями. Вычисляющих посторонних процессов на GPU 0 во время принятых '
             'серий не обнаружено. Уменьшение FLOPs не гарантирует ускорения в этой реализации.', '',
             '## Динамика ранга', '',
             'На цветовой шкале теперь проценты ширины, а не дроби 0–1. Данные и определения ранга не менялись. '
             'Измерения используют фиксированные 256 calibration-изображений на эпохах 0, 1, 10, 30, 60, 100, 150. '
             'График показывает средние dense-моделей; sparse-данные сохранены в CSV.', '',
             '![Динамика ранга](rank_dynamics.png)', '',
             '[Траектории двух определений ранга](rank_dynamics_dense.md).', '']
    fit = pd.read_csv(RUN / 'fit_diagnostics.csv')
    full += ['## Разрыв train / validation', '',
             table([{'Модель': group_name(r.family, r.dataset), 'Seed': str(r.model_seed),
                     'Train Accuracy, %': number(r.train_accuracy),
                     'Validation Accuracy, %': number(r.validation_accuracy),
                     'Validation / train, %': number(100*r.validation_accuracy/r.train_accuracy)}
                    for r in fit.itertuples()]), '',
             'Высокая train accuracy не означает хорошего обобщения; отношение validation/train здесь описательное.', '',
             '## Послойная чувствительность', '',
             'Связь нормированного ранга с ростом validation loss после удаления части каналов по QR '
             'без fine-tuning, с пересчётом BN для CNN. Коэффициент Spearman остаётся коэффициентом '
             'от −1 до 1; преобразовывать его в «процент качества» некорректно.', '']
    correlations = pd.read_csv(RUN / 'sensitivity_correlations.csv')
    corr = correlations.groupby(['family', 'dataset', 'fraction', 'method']).spearman.agg(['mean', 'std']).reset_index()
    full += [table([{'Модель': group_name(r.family, r.dataset), 'Удалено каналов, %': number(100*r.fraction, 0),
                     'Ранг': 'σ' if r.method == 'sigma' else 'cov', 'Spearman, среднее ± SD': mean_sd(r.mean, r.std)}
                    for r in corr.itertuples()]), '',
             'Корреляции по 6 или 9 блокам не изолируют причинную роль ранга: положение, ширина и функция блоков '
             'различаются. Мгновенная чувствительность не равна качеству после дообучения.', '']
    scales = pd.read_csv(RUN / 'rescaling.csv')
    for method in ('sigma', 'cov'):
        scales[f'change_{method}'] = 100 * (scales[f'rescaled_{method}']/scales[f'original_{method}'] - 1).abs()
    scale = scales.groupby(['family', 'dataset']).agg(sigma=('change_sigma', 'max'), cov=('change_cov', 'max'),
                                                     logits=('max_abs_logit_difference', 'max')).reset_index()
    full += ['## Масштабирование каналов без изменения функции', '',
             table([{'Модель': group_name(r.family, r.dataset), 'Макс. изменение ERank-σ, %': number(r.sigma),
                     'Макс. изменение ERank-cov, %': number(r.cov),
                     'Макс. ошибка logits, абс.': f'{r.logits:.1e}'} for r in scale.itertuples()]), '',
             'Ранг меняется при компенсированном масштабировании каналов, хотя logits практически сохраняются. '
             'Ошибка logits показана в абсолютных единицах: для её перевода в проценты нет заданного '
             'содержательного знаменателя. Это ограничивает трактовку ERank как необходимой ширины функции, '
             'но не исключает его использования как эвристики. Весовая SVD не проверяет PCA/SVD самих активаций.', '']
    for lines in (quick, full):
        lines += ['## Границы выводов', '', *['- ' + item for item in LIMITATIONS], '',
                  '## Данные', '',
                  '- [Все показатели в процентах](relative_metrics.csv).',
                  '- [Средние и SD относительных показателей](relative_summary.csv).',
                  '- [Все 168 сравнений ERank с бейзлайнами](erank_vs_baselines.csv).',
                  '- [Парные значения по seed](erank_vs_baselines_per_seed.csv).',
                  '- [Исходные абсолютные измерения](results.csv).',
                  '- [Первичные статистические проверки](primary_comparisons.csv).', '']
    (RUN / 'findings.md').write_text('\n'.join(quick))
    (RUN / 'report.md').write_text('\n'.join(full))
    detailed = ['# ERank против всех бейзлайнов', '', COMPARISON_NOTE, '',
                'В таблице качества показано относительное изменение Accuracy (среднее ± SD). '
                'В таблицах ресурсов показаны средние отношения: указанный бейзлайн = 100%. '
                'Меньше параметров/FLOPs/loss — лучше; положительное изменение скорости — быстрее.', '']
    for family, dataset in GROUPS:
        detailed += [f'## {group_name(family, dataset)}', '']
        for budget in (.9, .75, .6):
            part = subset(comparisons, family=family, dataset=dataset, budget=budget)
            detailed += [f'### Сокращение FLOPs на {round(100*(1-budget))}%', '',
                         comparison_quality(part), '']
            for method in ERANK:
                detailed += [f'Ресурсы и loss для **{name(method)}** относительно каждого бейзлайна:', '',
                             comparison_resources(part, method), '']
    detailed += ['## Ограничения сравнения', '', *['- ' + item for item in LIMITATIONS[:4]], '',
                 '[Полный отчёт](report.md) · [Краткая сводка](findings.md).', '']
    (RUN / 'erank_vs_baselines.md').write_text('\n'.join(detailed))


def main():
    protected = ['results.csv', 'main_summary.csv', 'primary_comparisons.csv', 'rank_dynamics.csv',
                 'additional_summary.csv', 'latency_rounds.csv', 'vit_architecture_summary.csv']
    hashes = {name: hashlib.sha256((RUN / name).read_bytes()).hexdigest() for name in protected}
    archive = RUN / 'report_history' / 'absolute_units'
    if not archive.exists():
        archive.mkdir(parents=True)
        for pattern in ('*.md', '*.png', '*.pdf'):
            for path in RUN.glob(pattern):
                shutil.copy2(path, archive / path.name)
        shutil.copy2(ROOT / 'experiments_v2/report.py', archive / 'report.py')
    raw = pd.read_csv(RUN / 'results.csv')
    assert len(raw) == 504 and raw.task.nunique() == 504
    results = normalize(raw)
    summary = summarize(results)
    paired, comparisons = compare_baselines(results)
    identity = ['task', *KEY, 'phase', 'method', 'budget']
    results[identity + METRICS].to_csv(RUN / 'relative_metrics.csv', index=False)
    summary.to_csv(RUN / 'relative_summary.csv', index=False)
    paired.to_csv(RUN / 'erank_vs_baselines_per_seed.csv', index=False)
    comparisons.to_csv(RUN / 'erank_vs_baselines.csv', index=False)
    primary = pd.read_csv(RUN / 'primary_comparisons.csv')
    reports(summary, comparisons, primary)
    plots(summary, comparisons)
    from rank_preview import main as rank_preview
    rank_preview()
    assert all(hashlib.sha256((RUN / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
    metadata = {'rendered_at': time.time(), 'comparisons': len(comparisons), 'paired_rows': len(paired),
                'comparison_status': 'Post-hoc descriptive comparisons requested after test; no new significance claims',
                'aggregation': 'Ratios within matched model seed, then mean and sample SD',
                'speed_definition': '100 * (reference_latency / candidate_latency - 1)',
                'quality_definition': '100 * candidate_accuracy / reference_accuracy',
                'unchanged_raw_sha256': hashes,
                'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (RUN / 'relative_report_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
    print('Percentage reports written; raw results and primary statistics unchanged.')


if __name__ == '__main__':
    main()
