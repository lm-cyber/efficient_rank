"""Report actual completed runs; distinguish seed uncertainty from image uncertainty."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import CONFIG, ROOT, RUN, FULL_MACS


def markdown_table(frame, columns=None, digits=3):
    if columns is not None:
        frame = frame[columns]
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}" if np.isfinite(value) else "—"
        return str(value).replace("|", "/")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines += ["| " + " | ".join(fmt(value) for value in row) + " |"
              for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def conditional_image_ci(differences, seed):
    """Paired image bootstrap, conditional on the already trained models."""
    rng = np.random.default_rng(seed)
    n = len(differences)
    estimates = []
    for start in range(0, CONFIG["bootstrap_repeats"], 100):
        size = min(100, CONFIG["bootstrap_repeats"] - start)
        indices = rng.integers(0, n, size=(size, n), dtype=np.int32)
        estimates.extend(100 * differences[indices].mean(axis=1))
    return np.percentile(estimates, [2.5, 97.5])


def generate_report():
    records, correctness = [], {}
    labels = np.load(ROOT / "data/test_labels.npy")
    for directory in sorted((RUN / "models").iterdir()):
        if not (directory / "test.json").exists():
            continue
        trained = json.loads((directory / "train_done.json").read_text())
        test = json.loads((directory / "test.json").read_text())
        result = json.loads((directory / "result.json").read_text()) if (directory / "result.json").exists() else {}
        row = {"task": directory.name, "phase": trained["phase"], "method": trained["method"],
               "model_seed": trained["model_seed"], "budget": trained.get("budget", 1.0),
               "repeat": trained.get("repeat"), "params": trained["params"],
               "flops": trained["flops"], "flops_retained_actual": trained["flops"] / (2 * FULL_MACS),
               "widths": str(trained["widths"]), "epochs": trained["epochs"],
               "best_epoch": trained["best_epoch"], "val_accuracy": trained["best_val_accuracy"],
               "test_accuracy": test["accuracy"], "test_loss": test["loss"],
               "training_seconds": trained["training_seconds"],
               "raw_val_accuracy": result.get("before", {}).get("accuracy", np.nan),
               "bn_val_accuracy": result.get("after_bn", {}).get("accuracy", np.nan)}
        records.append(row)
        predictions = np.load(directory / "test_predictions.npz")["predictions"]
        correctness[directory.name] = (predictions == labels).astype(np.float64)
    runs = pd.DataFrame(records)
    assert len(runs) == 63, f"Incomplete report: {len(runs)} / 63"
    runs.to_csv(RUN / "results.csv", index=False)
    main = runs[runs.phase == "main"].copy()
    per_seed = main.groupby(["budget", "method", "model_seed"], as_index=False).agg(
        test_accuracy=("test_accuracy", "mean"), test_loss=("test_loss", "mean"),
        val_accuracy=("val_accuracy", "mean"), params=("params", "mean"), flops=("flops", "mean"))
    per_seed.to_csv(RUN / "main_per_seed.csv", index=False)
    summary = per_seed.groupby(["budget", "method"], as_index=False).agg(
        test_mean=("test_accuracy", "mean"), test_std_between_seeds=("test_accuracy", "std"),
        n_seeds=("model_seed", "size"), params=("params", "mean"), flops=("flops", "mean"))
    summary.to_csv(RUN / "main_summary.csv", index=False)
    paired = []
    for budget in CONFIG["flops_retained"]:
        for method in ("erank_sigma", "erank_cov", "random"):
            deltas, per_image = [], []
            for seed in CONFIG["model_seeds"]:
                target = main[(main.budget == budget) & (main.method == method) & (main.model_seed == seed)]
                reference = main[(main.budget == budget) & (main.method == "uniform") & (main.model_seed == seed)]
                a = np.mean([correctness[task] for task in target.task], axis=0)
                b = correctness[reference.task.iloc[0]]
                per_image.append(a - b)
                deltas.append(float(100 * (a - b).mean()))
            deltas = np.asarray(deltas)
            mean = float(deltas.mean())
            half = float(stats.t.ppf(0.975, len(deltas) - 1) * deltas.std(ddof=1) / np.sqrt(len(deltas)))
            image_low, image_high = conditional_image_ci(np.mean(per_image, axis=0), 991 + round(budget * 100))
            paired.append({"budget": budget, "method_minus_uniform": method, "mean_delta_pp": mean,
                           "seed_t95_low_pp": mean - half, "seed_t95_high_pp": mean + half,
                           "conditional_image95_low_pp": image_low,
                           "conditional_image95_high_pp": image_high,
                           **{f"seed_{seed}_delta_pp": value for seed, value in zip(CONFIG["model_seeds"], deltas)}})
    paired = pd.DataFrame(paired)
    paired.to_csv(RUN / "paired_comparisons.csv", index=False)
    sensitivity, representation, rescaling, subsets, native = [], [], [], [], []
    for seed in CONFIG["model_seeds"]:
        directory = RUN / "diagnostics" / f"s{seed}"
        sensitivity += json.loads((directory / "sensitivity.json").read_text())
        rep = json.loads((directory / "representation.json").read_text())
        representation += [{"model_seed": seed, **{k: v for k, v in row.items() if k != "singular_values"}}
                           for row in rep["layers"]]
        rescaling += json.loads((directory / "rescaling.json").read_text())
        subsets += json.loads((directory / "calibration_subsets.json").read_text())
        native += json.loads((directory / "native_ceil_rule.json").read_text())
    sens = pd.DataFrame(sensitivity)
    sens.drop(columns=["before", "after_bn", "widths"]).to_csv(RUN / "sensitivity.csv", index=False)
    rep = pd.DataFrame(representation)
    rep.to_csv(RUN / "rank_profiles.csv", index=False)
    rescale = pd.DataFrame(rescaling).drop(columns="scales")
    rescale.to_csv(RUN / "rescaling.csv", index=False)
    pd.DataFrame(subsets).to_csv(RUN / "calibration_subsets.csv", index=False)
    native_flat = [{**{k: v for k, v in row.items() if k not in ("before", "after_bn")},
                    "raw_val_accuracy": row["before"]["accuracy"],
                    "bn_val_accuracy": row["after_bn"]["accuracy"]} for row in native]
    pd.DataFrame(native_flat).to_csv(RUN / "native_rule.csv", index=False)
    associations = []
    for (seed, fraction), frame in sens.groupby(["model_seed", "removed_fraction_requested"]):
        for score in ("normalized_sigma", "normalized_cov"):
            for stage in (0, 1, 2, 3):
                selected = frame if stage == 0 else frame[frame.stage == stage]
                rho = float(stats.spearmanr(selected[score], selected.loss_increase_after_bn).statistic)
                associations.append({"model_seed": seed, "removed_fraction": fraction,
                                     "score": score, "stage": stage, "n_blocks": len(selected),
                                     "spearman_descriptive": rho})
    correlations = pd.DataFrame(associations)
    correlations.to_csv(RUN / "sensitivity_correlations.csv", index=False)
    extra = runs[runs.phase.isin(["ablation", "scratch", "dense_control", "base"])].groupby(
        ["phase", "method", "budget"], as_index=False).agg(
            test_mean=("test_accuracy", "mean"), test_std=("test_accuracy", "std"), n=("model_seed", "size"))
    extra.to_csv(RUN / "ablation_scratch_summary.csv", index=False)
    ablation_deltas = []
    for seed in CONFIG["model_seeds"]:
        base_ft = runs[(runs.phase == "main") & (runs.method == "erank_sigma") &
                       (runs.budget == 0.75) & (runs.model_seed == seed)].iloc[0]
        for method in ("erank_sigma_random_channels", "erank_sigma_compensated"):
            row = runs[(runs.method == method) & (runs.model_seed == seed)].iloc[0]
            ablation_deltas.append({"model_seed": seed, "method": method,
                         "raw_val_delta_pp": row.raw_val_accuracy - base_ft.raw_val_accuracy,
                         "bn_val_delta_pp": row.bn_val_accuracy - base_ft.bn_val_accuracy,
                         "ft_test_delta_pp": row.test_accuracy - base_ft.test_accuracy})
    ablation_deltas = pd.DataFrame(ablation_deltas)
    ablation_deltas.to_csv(RUN / "ablation_paired.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, frame in summary.groupby("method"):
        frame = frame.sort_values("flops")
        ax.errorbar(frame.flops / 1e6, frame.test_mean, yerr=frame.test_std_between_seeds,
                    marker="o", capsize=4, label=method)
    dense = runs[runs.phase == "dense_control"]
    ax.errorbar([2 * FULL_MACS / 1e6], [dense.test_accuracy.mean()],
                yerr=[dense.test_accuracy.std(ddof=1)], marker="*", markersize=12,
                capsize=4, color="black", label="Original + fine-tuning")
    ax.set(xlabel="FLOPs (millions; convolutions + classifier)", ylabel="Test accuracy (%)",
           title="ResNet-20 / CIFAR-10: mean ± SD over 3 base seeds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RUN / "accuracy_flops.png", dpi=180)
    fig.savefig(RUN / "accuracy_flops.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, score in zip(axes, ("normalized_sigma", "normalized_cov")):
        for fraction, frame in sens.groupby("removed_fraction_requested"):
            ax.scatter(frame[score], frame.loss_increase_after_bn, alpha=0.7, label=f"remove {fraction:.0%}")
        ax.set(xlabel=score, ylabel="Validation loss increase vs dense BN control",
               title="Layer interventions (points are not independent seeds)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(RUN / "layer_sensitivity.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, field in zip(axes, ("normalized_sigma", "normalized_cov")):
        for seed, frame in rep.groupby("model_seed"):
            ax.plot(frame.block, frame[field], marker="o", label=f"seed {seed}")
        ax.set(xlabel="Residual block", ylabel=field, title="Activation rank profiles")
        ax.legend()
    fig.tight_layout()
    fig.savefig(RUN / "rank_profiles.png", dpi=180)
    plt.close(fig)

    lines = ["# Результаты нового пула ERank", "",
             "Протокол: `erank_audit_v1`. Все результаты ниже получены фактическими запусками на физическом GPU 0. "
             "Исходный план: [experiment_pool_v1.md](../../experiment_pool_v1.md).", "",
             f"Завершено **{len(runs)} обучений**, **{int(runs.epochs.sum())} эпох**. "
             f"Сумма измеренного времени обучений: **{runs.training_seconds.sum()/3600:.2f} ч**; "
             "диагностика, загрузка данных и финальная оценка в эту сумму не входят.", "",
             "## Основное сравнение на официальном test", "",
             "`budget` — сохранённая доля FLOPs. Среднее и SD рассчитаны между тремя исходными seed; "
             "для Random две реализации сначала усредняются внутри seed. Все методы одного бюджета имеют точное совпадение MACs.", "",
             markdown_table(summary, ["budget", "method", "test_mean", "test_std_between_seeds", "n_seeds"]), "",
             "![Качество и FLOPs](accuracy_flops.png)", "",
             "## Парные различия относительно Uniform", "",
             "Разницы выражены в процентных пунктах. Интервал `seed_t95` основан на трёх парных разностях и t-распределении "
             "с двумя степенями свободы; нормальность при таком объёме не проверяется, поэтому интервал предварительный. "
             "Отдельные условные bootstrap-интервалы по test-изображениям находятся в CSV и не заменяют вариативность обучения.", "",
             markdown_table(paired, ["budget", "method_minus_uniform", "mean_delta_pp", "seed_t95_low_pp", "seed_t95_high_pp"]), "",
             "Порог практической значимости зафиксирован до запуска: ±0,2 п.п. Широкий интервал означает неопределённость, "
             "а не равенство методов.", "",
             "## Отбор каналов, компенсация и обучение с нуля", "",
             markdown_table(extra), "",
             "Абляции используют те же ширины ERank-sigma при budget=0,75. Ниже сравниваются случайные идентичности каналов "
             "и линейная компенсация с основным QR-вариантом. Первые два столбца — validation до fine-tuning, последний — итоговый test.", "",
             markdown_table(ablation_deltas), "",
             "Обучение с нуля занимает 150 эпох. Ветка переноса весов включает 150 эпох исходной модели и 20 эпох fine-tuning. "
             "Равенство суммарного compute-бюджета между этими ветками не заявляется.", "",
             "## Чувствительность слоёв", "",
             "54 отдельных вмешательства: 9 блоков × 2 доли удаления × 3 seed. После BN сравнение проводится с "
             "несжатой моделью, прошедшей тот же пересчёт BN. Корреляции описательные; блоки одной сети не являются независимыми повторениями.", "",
             markdown_table(correlations[correlations.stage == 0],
                            ["model_seed", "removed_fraction", "score", "n_blocks", "spearman_descriptive"]), "",
             "![Чувствительность слоёв](layer_sensitivity.png)", "",
             "## Устойчивость определения ранга", "",
             "В следующем контроле положительное масштабирование BN-каналов компенсируется обратным масштабированием следующей свёртки. "
             "Ранги вычислены по тем же активациям и изображениям; разница logits проверена в FP32.", "",
             markdown_table(rescale, ["model_seed", "block", "original_sigma", "rescaled_sigma",
                                      "original_cov", "rescaled_cov", "max_abs_logit_difference"], digits=6), "",
             "![Ранги по слоям](rank_profiles.png)", "",
             "Оценки per-image, ограничения их максимального ранга и чувствительность к подвыборкам сохранены в "
             "`rank_profiles.csv` и `calibration_subsets.csv`. Они не использовались для выбора лучшей test-конфигурации.", "",
             "## Ограничения", "",
             "Три исходных seed, одна архитектура и один датасет. Random в основном сравнении условен на число удалений и точный MAC-бюджет; "
             "исходный глобальный Random без ограничения стоимости сохранён отдельно в `native_rule.csv`. "
             "Основные методы масштабируют профиль ERank под бюджет, поэтому не тождественны прямому `ceil(ERank)`. "
             "Фиксированные 20 эпох fine-tuning не доказывают достижение оптимума восстановления. Измерения аппаратного ускорения не проводились.", "",
             "## Артефакты", "",
             "- `results.csv`: все 63 модели, параметры, FLOPs, качество и время обучения.",
             "- `main_per_seed.csv`, `main_summary.csv`, `paired_comparisons.csv`: основные сравнения и неопределённость.",
             "- `sensitivity.csv`, `sensitivity_correlations.csv`: послойные вмешательства.",
             "- `rank_profiles.csv`, `rescaling.csv`, `calibration_subsets.csv`: спектральная диагностика.",
             "- `ablation_paired.csv`, `ablation_scratch_summary.csv`, `native_rule.csv`: контрольные эксперименты.",
             "- `models/`: checkpoints, маски каналов, журналы эпох, предсказания test.",
             "- `config.json`, `split.npz`, `environment.json`, `source_base/`, `source_pool/`: воспроизведение.", ""]
    (RUN / "report.md").write_text("\n".join(lines))
    print("REPORT WRITTEN", RUN / "report.md", flush=True)


if __name__ == "__main__":
    generate_report()
