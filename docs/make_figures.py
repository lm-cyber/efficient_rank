"""Сборка графиков для презентации и итогового отчёта.

Использует только опубликованные CSV из runs/. GPU и checkpoint не нужны.
Запуск из корня репозитория:

    python3 docs/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Палитра: ERank — синий/оранжевый, бейзлайны — нейтральные серые.
BLUE = "#257de0"
ORANGE = "#f47230"
RED = "#ec4647"
GRAY = "#8c8c8c"
DARK = "#4d4d4d"
INK = "#191919"
INK2 = "#666666"
GRID = "#e6e6e6"

for name in ("PT Sans", "Helvetica Neue", "Arial", "DejaVu Sans"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        FONT = name
        break
else:  # pragma: no cover
    FONT = "DejaVu Sans"

plt.rcParams.update(
    {
        "font.family": [FONT, "DejaVu Sans"],
        "font.size": 13,
        "axes.edgecolor": "#bdbdbd",
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "xtick.color": INK2,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)

SETTING = {
    ("resnet20", "cifar10"): "ResNet-20 / CIFAR-10",
    ("resnet20", "cifar100"): "ResNet-20 / CIFAR-100",
    ("mini_vit", "cifar10"): "Mini-ViT / CIFAR-10",
    ("mini_vit", "cifar100"): "Mini-ViT / CIFAR-100",
}
METHOD = {
    "uniform_qr": "Uniform + QR",
    "random_width": "Random + QR",
    "erank_sigma": "ERank-σ + QR",
    "erank_cov": "ERank-cov + QR",
    "l1": "L1",
    "taylor": "Taylor",
    "hrank": "HRank",
    "slimming": "Slimming",
    "svd": "SVD весов",
}
METHOD_COLOR = {"erank_sigma": BLUE, "erank_cov": ORANGE}


def fmt(x, digits=1, sign=False):
    s = f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"
    return s.replace(".", ",").replace("-", "−")


def comma_axes(fig):
    """Десятичная запятая и типографский минус на числовых осях."""
    from matplotlib.ticker import FuncFormatter

    f = FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",").replace("-", "−"))
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if isinstance(axis.get_major_formatter(), matplotlib.ticker.ScalarFormatter):
                axis.set_major_formatter(f)


def save(fig, name):
    comma_axes(fig)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("saved", OUT / name)


# 1. ERank − Uniform: 24 заранее заданных сравнения (v2).
def fig_erank_vs_uniform():
    p = pd.read_csv(RUNS / "erank_models_v2" / "primary_comparisons.csv")
    order = [k for k in SETTING]
    rows = []
    for fam, ds in order:
        for budget in (0.9, 0.75, 0.6):
            rows.append((fam, ds, budget))
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.axvspan(-0.2, 0.2, color="#f2f2f2", zorder=0)
    ax.axvline(0, color=DARK, lw=1.2, zorder=1)
    ylabels = []
    for i, (fam, ds, budget) in enumerate(rows):
        y = len(rows) - 1 - i
        cut = round((1 - budget) * 100)
        ylabels.append((y, f"{SETTING[(fam, ds)]}  −{cut}%"))
        for method, dy, color, label in (
            ("erank_sigma", 0.16, BLUE, "ERank-σ"),
            ("erank_cov", -0.16, ORANGE, "ERank-cov"),
        ):
            r = p[(p.family == fam) & (p.dataset == ds) & (p.budget == budget) & (p.method == method)].iloc[0]
            ax.plot([r.ci95_low, r.ci95_high], [y + dy] * 2, color=color, lw=2, solid_capstyle="round", zorder=2)
            ax.scatter(r.difference_pp, y + dy, s=55, color=color, edgecolor="white", linewidth=1.5, zorder=3,
                       label=label if i == 0 else None)
    for k in (3, 6, 9):
        ax.axhline(len(rows) - k - 0.5, color=GRID, lw=1, zorder=0)
    ax.set_yticks([y for y, _ in ylabels], [t for _, t in ylabels], fontsize=11)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ERank − Uniform + QR, п.п. test accuracy  (точка — среднее 3 seed, линия — 95% ДИ без поправки)")
    ax.text(0, len(rows) - 0.25, "зона ±0,2 п.п.", ha="center", va="bottom", fontsize=10, color=INK2)
    ax.legend(loc="lower right", ncols=2, fontsize=11)
    save(fig, "erank_vs_uniform.png")


# 2. Все методы при −25% FLOPs (v2).
def fig_methods_25():
    r = pd.read_csv(RUNS / "erank_models_v2" / "relative_summary.csv")
    r = r[(r.phase == "main") & (r.budget == 0.75)]
    methods = list(METHOD)
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.6), sharey=True)
    for ax, (key, title) in zip(axes, SETTING.items()):
        sub = r[(r.family == key[0]) & (r.dataset == key[1])].set_index("method")
        vals = sub.loc[methods, "quality_pct_mean"]
        sds = sub.loc[methods, "quality_pct_std"]
        main_vals = vals.drop("svd") if vals["svd"] < vals.drop("svd").min() - 1.5 else vals
        lo = min(main_vals - sds[main_vals.index]) - 0.4
        hi = max(max(vals + sds), 100) + 0.3
        for j, m in enumerate(methods):
            y = len(methods) - 1 - j
            color = METHOD_COLOR.get(m, DARK if m == "uniform_qr" else GRAY)
            v, sd = vals[m], sds[m]
            if v < lo:
                ax.annotate(f"{fmt(v)}% ←", (lo, y), xytext=(4, 0), textcoords="offset points",
                            va="center", ha="left", fontsize=10, color=INK2)
                continue
            ax.plot([v - sd, v + sd], [y, y], color=color, lw=2, solid_capstyle="round")
            ax.scatter(v, y, s=60 if m in METHOD_COLOR else 42, color=color, edgecolor="white", linewidth=1.4, zorder=3)
        ax.axvline(100, color=DARK, lw=1, ls=(0, (3, 3)))
        ax.set_xlim(lo, hi)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(methods)), [METHOD[m] for m in reversed(methods)])
    for t in axes[0].get_yticklabels():
        if t.get_text().startswith("ERank"):
            t.set_fontweight("bold")
    fig.supxlabel("Сохранённое качество, % от исходной модели (среднее ± SD, 3 seed); пунктир — 100%",
                  fontsize=11, color=INK2, y=0.0)
    fig.tight_layout()
    save(fig, "methods_flops25.png")


# 3. FLOPs не равны скорости (v2, batch 128, eager BF16, A100).
def fig_speed():
    r = pd.read_csv(RUNS / "erank_models_v2" / "relative_summary.csv")
    r = r[r.budget.isin([0.75, 0.735]) | (r.flops_pct_mean.between(73, 76))]
    panels = [
        ("ResNet-20 / CIFAR-100, −25% FLOPs", "resnet20", "cifar100",
         [("main", "uniform_qr", "Uniform + QR"), ("main", "l1", "L1"), ("main", "taylor", "Taylor"),
          ("main", "erank_sigma", "ERank-σ"), ("main", "random_width", "Random + QR"),
          ("main", "slimming", "Slimming"), ("main", "svd", "SVD весов"), ("main", "erank_cov", "ERank-cov")]),
        ("Mini-ViT / CIFAR-100, −25% FLOPs", "mini_vit", "cifar100",
         [("main", "uniform_qr", "Uniform: ширина MLP"), ("main", "erank_sigma", "ERank-σ: ширина MLP"),
          ("main", "svd", "SVD весов"), ("architecture_scratch", "attention_only", "Меньше attention heads"),
          ("architecture_scratch", "depth8_dim64", "8 блоков, dim 64"),
          ("architecture_scratch", "depth4_dim96", "4 блока, dim 96")]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, (title, fam, ds, items) in zip(axes, panels):
        sub = r[(r.family == fam) & (r.dataset == ds)]
        labels, vals = [], []
        for phase, m, label in items:
            row = sub[(sub.phase == phase) & (sub.method == m)].iloc[0]
            labels.append(label)
            vals.append(row.speed_b128_change_pct_mean)
        ys = list(range(len(vals)))[::-1]
        colors = [BLUE if v > 0 else RED for v in vals]
        ax.barh(ys, vals, color=colors, height=0.62)
        ax.axvline(0, color=DARK, lw=1.2)
        span = max(abs(min(vals)), abs(max(vals)))
        for y, v in zip(ys, vals):
            ax.text(v + (0.02 * span if v >= 0 else -0.02 * span), y, fmt(v, sign=True) + "%",
                    va="center", ha="left" if v >= 0 else "right", fontsize=11, color=INK)
        ax.set_yticks(ys, labels)
        ax.set_xlim(min(vals) - 0.35 * span if min(vals) < 0 else -0.1 * span,
                    max(vals) + 0.35 * span if max(vals) > 0 else 0.1 * span)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", visible=False)
    fig.supxlabel("Изменение пропускной способности относительно исходной модели, %  (+ быстрее, − медленнее; "
                  "A100, BF16, batch 128)", fontsize=11, color=INK2, y=0.0)
    fig.tight_layout()
    save(fig, "flops_vs_speed.png")


# 4. ADE20K / SegFormer-B1.
def fig_ade20k():
    s = pd.read_csv(RUNS / "erank_segmentation_v1" / "summary.csv").set_index(["method", "budget"])
    items = [
        ("taylor", "Taylor", DARK), ("uniform", "Uniform + QR", DARK),
        ("erank_sigma_low", "ERank-σ, low", BLUE), ("erank_sigma_random", "ERank-σ, random", BLUE),
        ("erank_sigma_high", "ERank-σ, high", BLUE),
        ("erank_cov_low", "ERank-cov, low", ORANGE), ("erank_cov_random", "ERank-cov, random", ORANGE),
        ("erank_cov_high", "ERank-cov, high", ORANGE),
        ("random_width", "Random + QR", GRAY), ("l1", "L1", GRAY),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, (budget, title) in zip(axes, ((0.9, "−10% FLOPs"), (0.8, "−20% FLOPs"))):
        ys = list(range(len(items)))[::-1]
        for y, (m, label, color) in zip(ys, items):
            v = s.loc[(m, budget), "quality_pct_mean"]
            ax.barh(y, v, color=color, height=0.62)
            ax.text(v + 1, y, fmt(v) + "%", va="center", fontsize=11, color=INK)
        ax.axvline(100, color=DARK, lw=1, ls=(0, (3, 3)))
        ax.set_xlim(0, 115)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", visible=False)
        ax.set_yticks(ys, [lab for _, lab, _ in items])
    fig.supxlabel("Сохранённый mIoU, % от исходной модели (40,94 mIoU); среднее 3 seed fine-tuning",
                  fontsize=11, color=INK2, y=0.0)
    fig.tight_layout()
    save(fig, "ade20k_quality.png")


# 5. Динамика ранга во время обучения исходных моделей.
def fig_rank_dynamics():
    d = pd.read_csv(RUNS / "erank_models_v2" / "rank_dynamics.csv")
    d = d[(d.phase == "base") & (d.method == "dense")]
    g = d.groupby(["family", "dataset", "epoch"])[["normalized_sigma", "normalized_cov"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, fam, title in zip(axes, ("resnet20", "mini_vit"), ("ResNet-20", "Mini-ViT")):
        for ds, ls in (("cifar10", "-"), ("cifar100", (0, (4, 3)))):
            sub = g[(g.family == fam) & (g.dataset == ds)]
            for col, color, name in (("normalized_sigma", BLUE, "ERank-σ"), ("normalized_cov", ORANGE, "ERank-cov")):
                ax.plot(sub.epoch, 100 * sub[col], color=color, ls=ls, lw=2, marker="o", ms=4,
                        label=f"{name}, {ds.upper().replace('CIFAR', 'CIFAR-')}")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Эпоха обучения")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("ERank / ширина слоя, %")
    axes[0].legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    save(fig, "rank_dynamics.png")


# 6. Немедленное качество не предсказывает итог (v1, ResNet-20 / CIFAR-10, −25% FLOPs).
def fig_immediate_vs_ft():
    a = pd.read_csv(RUNS / "erank_audit_v1" / "ablation_paired.csv")
    sc = pd.read_csv(RUNS / "erank_audit_v1" / "ablation_scratch_summary.csv").set_index("method")
    qr_immediate = 59.36  # findings.md, раздел 3: ERank-σ + QR, validation сразу после удаления
    comp = qr_immediate + a[a.method == "erank_sigma_compensated"].raw_val_delta_pp.mean()
    rnd = qr_immediate + a[a.method == "erank_sigma_random_channels"].raw_val_delta_pp.mean()
    qr_ft = 91.000  # main summary, ERank-σ, budget 0.75
    labels = ["Случайные\nканалы", "QR", "QR + линейная\nкомпенсация"]
    immediate = [rnd, qr_immediate, comp]
    final = [sc.loc["erank_sigma_random_channels", "test_mean"], qr_ft, sc.loc["erank_sigma_compensated", "test_mean"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, vals, title in zip(axes, (immediate, final),
                               ("Сразу после удаления каналов (validation)", "После 20 эпох fine-tuning (test)")):
        xs = range(3)
        ax.bar(xs, vals, color=[GRAY, BLUE, DARK], width=0.6)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.5, fmt(v, 2 if v > 85 else 1) + "%", ha="center", fontsize=12, color=INK, fontweight="bold")
        ax.set_xticks(list(xs), labels)
        ax.set_ylim(0, 105)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Accuracy, %")
    fig.tight_layout()
    save(fig, "immediate_vs_finetuned.png")


if __name__ == "__main__":
    fig_erank_vs_uniform()
    fig_methods_25()
    fig_speed()
    fig_ade20k()
    fig_rank_dynamics()
    fig_immediate_vs_ft()
