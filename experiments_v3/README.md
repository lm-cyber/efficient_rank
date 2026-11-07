# Аппаратный аудит и архивный план v3

Выполнена только аппаратная часть этого плана: повторные измерения **72 checkpoint на CIFAR-100**, три раунда, batch 1/128. В `runs/erank_followup_v3/` сохранены 216 серий eager/graph-аудита и ещё 216 серий точного воспроизведения исходного eager-контекста v2.

[Результаты аудита](../runs/erank_followup_v3/hardware_audit.md) показывают скорость в процентах относительно несжатой модели в том же режиме. В парных eager/graph-контролях кеш autocast отключён. Отдельная колонка Original v2 eager использует исходный контекст с кешем. Tuned включает подбор алгоритмов cuDNN и отключает требование детерминизма; это отдельные условия измерения.

**Обучения Tiny ImageNet и классификационные low/random/high-прогоны v3 не запускались.** Пользователь заменил их на семантическую сегментацию [ADE20K / SegFormer-B1](../experiments_segmentation/README.md). История замены записана в [task_revision.json](../runs/erank_followup_v3/task_revision.json). Контроллер v3 проверяет этот маркер и предотвращает случайный запуск отменённой очереди. Старый [протокол](../experiment_pool_v3.md), конфигурация и подготовленный код оставлены как история работы, а не как выполненные результаты.

## Пересборка таблиц

GPU и веса для пересборки опубликованного аппаратного отчёта не нужны. Используется окружение из `experiments_v2/requirements.txt`. Из корня репозитория:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONPATH=experiments_v3 python3 -c 'from hardware_audit import report; report()'
```

Для повторения самих измерений нужны checkpoint из `runs/erank_models_v2/models/`; они исключены из Git и должны быть воспроизведены по инструкции v2. Сначала сохраните существующий `runs/erank_followup_v3/` под другим именем, чтобы не использовать маркеры готовых серий. На свободной физической GPU 0 выполните последовательно:

```bash
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python3 experiments_v3/hardware_audit.py
python3 experiments_v3/hardware_original.py
```

Эти две команды выполняют только измерения и пересборку таблиц. Скрипты отклоняют замер при обнаружении другого процесса на GPU 0.
