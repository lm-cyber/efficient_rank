"""Small live progress file; only reads project artifacts, never touches training or test data."""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/erank_models_v2'


def update():
    planned = json.loads((RUN / 'planned_runs.json').read_text())
    done = {p.parent.name: json.loads(p.read_text()) for p in (RUN / 'models').glob('*/train_done.json')}
    states = []
    for family in ('resnet20', 'mini_vit'):
        for dataset in ('cifar10', 'cifar100'):
            path = RUN / 'status' / f'{family}_{dataset}.json'
            state = json.loads(path.read_text()) if path.exists() else {'state': 'pending'}
            expected = [j for j in planned if j['family'] == family and j['dataset'] == dataset]
            states.append({'family': family, 'dataset': dataset, 'planned': len(expected),
                           'completed': sum(j['task'] in done for j in expected), **state})
    epochs = sum(v['epochs'] for v in done.values())
    epochs += sum(s.get('epoch', 0) for s in states if s.get('task') not in done)
    main = json.loads((RUN / 'status.json').read_text()) if (RUN / 'status.json').exists() else {'state': 'training'}
    failures = [{'file': str(path.relative_to(RUN)), **json.loads(path.read_text())}
                for path in sorted((RUN / 'failures').glob('*.json'))]
    record = {'updated_at_utc': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
              'completed_trainings': len(done), 'planned_trainings': len(planned),
              'completed_epochs_including_active': epochs, 'planned_epochs': sum(j['epochs'] for j in planned),
              'completed_tests': len(list((RUN / 'models').glob('*/test.json'))),
              'completed_latency_series': len(list((RUN / 'latency').glob('*_r*.json'))),
              'state': 'failed' if failures else main['state'], 'groups': states,
              'failures': failures}
    temporary = RUN / 'progress.tmp'
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    temporary.replace(RUN / 'progress.json')
    lines = ['# Текущий прогресс расширенного пула', '', record['updated_at_utc'], '',
             f'GPU 0. Обучения: **{len(done)} / {len(planned)}**. Эпохи, включая активные задачи: **{epochs} / {record["planned_epochs"]}**.', '',
             f'Test: **{record["completed_tests"]} / {len(planned)}**. Серии измерения инференса: **{record["completed_latency_series"]} / {len(planned)*5}**.', '',
             f'Состояние: **{record["state"]}**. Зарегистрированных ошибок: **{len(failures)}**.', '',
             '| Модель | Датасет | Завершено | Текущая задача | Эпоха | Лучшее validation accuracy |',
             '| --- | --- | --- | --- | --- | --- |']
    for s in states:
        lines.append(f'| {s["family"]} | {s["dataset"]} | {s["completed"]}/{s["planned"]} | '
                     f'{s.get("task",s["state"])} | {s.get("epoch","—")} / {s.get("epochs","—")} | {s.get("best_accuracy","—")} |')
    if failures:
        lines += ['', 'Ошибки требуют проверки; незавершённые задачи не входят в результаты.', '']
        lines += [f'- [{failure["file"]}]({failure["file"]})' for failure in failures]
    conclusion = ('Все обучения, test-оценки и измерения инференса завершены. '
                  '[Краткие результаты](findings.md) · [Полный отчёт](report.md) · [Все запуски](results.csv).'
                  if record['state'] == 'complete' else
                  'Validation-метрики служат для выбора checkpoint. Финальное сравнение методов будет рассчитано по test после завершения всего пула.')
    lines += ['', conclusion, '',
              'План: [experiment_pool_v2.md](../../experiment_pool_v2.md).']
    temporary = RUN / 'progress_md.tmp'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(RUN / 'progress.md')
    return main['state'] == 'complete'


if __name__ == '__main__':
    while not update():
        time.sleep(30)
