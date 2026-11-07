"""Plot completed dense rank trajectories without opening test data."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUN = Path(__file__).resolve().parents[1] / 'runs/erank_models_v2'
EPOCHS = [0, 1, 10, 30, 60, 100, 150]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    lines = ['# Динамика ранга: все 12 исходных dense-моделей', '',
             'Все базовые dense-обучения завершены. Точки — среднее нормированного ранга по блокам и трём seed; полосы — минимум и максимум послойных средних, а не доверительные интервалы. Использованы фиксированные 256 calibration-изображений. Test не используется.', '',
             'Ранг измерен в состоянии модели на указанной эпохе. Основные эксперименты сжатия отдельно используют лучший validation-checkpoint и 1024 calibration-изображения; эти оценки могут отличаться.', '',
             '| Модель | Датасет | Seed | Sigma / width, эпоха 0 | Эпоха 150 | Cov / width, эпоха 0 | Эпоха 150 |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for ax, (family, dataset) in zip(axes.ravel(), [(f, d) for f in ('resnet20', 'mini_vit') for d in ('cifar10', 'cifar100')]):
        arrays = []
        for seed in (42, 123, 2026):
            folder = RUN / 'models' / f'{family}_{dataset}_s{seed}_dense'
            assert (folder / 'train_done.json').exists(), folder
            snapshots = [json.loads(p.read_text()) for p in sorted((folder / 'rank_dynamics').glob('e*.json'))]
            assert [s['epoch'] for s in snapshots] == EPOCHS
            arrays.append(np.array([[[r['normalized_sigma'], r['normalized_cov']] for r in s['layers']] for s in snapshots]))
        values = np.array(arrays)
        for i, (label, color) in enumerate((('sigma', 'tab:blue'), ('cov', 'tab:orange'))):
            block_mean = values[:, :, :, i].mean(0)
            ax.plot(EPOCHS, block_mean.mean(1), '-o', label=label, color=color, markersize=3)
            ax.fill_between(EPOCHS, block_mean.min(1), block_mean.max(1), alpha=.12, color=color)
        ax.set(title=f'{family} / {dataset} (3 seeds)', xlabel='Training epoch', ylabel='ERank / width', ylim=(0, 1.03))
        ax.grid(alpha=.2)
        start, end = values.mean((0, 2))[[0, -1]]
        lines.append(f'| {family} | {dataset} | 42, 123, 2026 | {start[0]:.3f} | {end[0]:.3f} | {start[1]:.3f} | {end[1]:.3f} |')
    axes[0, 0].legend()
    fig.suptitle('Rank trajectories of completed dense models')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(RUN / f'rank_dynamics_dense.{suffix}', dpi=160)
    plt.close(fig)
    lines += ['', '![Динамика ранга](rank_dynamics_dense.png)', '',
              'ERank-sigma и ERank-cov используют один центрированный массив активаций; веса спектра равны sigma и sigma² соответственно. Значения не являются доказательством возможности удалить соответствующую долю каналов без потери качества.', '',
              'Исходные послойные данные: `models/*_dense/rank_dynamics/e*.json`. Аналогичные семь измерений сохранены для всех 12 sparse-моделей. Эта страница описывает только динамику базовых моделей; состояние полного сравнения методов сжатия указано в [progress.md](progress.md).', '']
    (RUN / 'rank_dynamics_dense.md').write_text('\n'.join(lines))
    print('\n'.join(lines[6:12]))


if __name__ == '__main__':
    main()
