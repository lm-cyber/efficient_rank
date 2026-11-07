"""List every planned training before the pruning results are available."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / "experiments/config.json").read_text())
    run = ROOT / "runs" / config["protocol"]
    rows = []
    def add(task, experiment, phase, method, seed, budget, epochs, repeat=None):
        rows.append(dict(task=task, experiment=experiment, phase=phase, method=method,
                         model_seed=seed, budget=budget, epochs=epochs, repeat=repeat))
    for seed in config["model_seeds"]:
        add(f"base_s{seed}", "E0", "base", "original", seed, 1.0, config["base_epochs"])
        for budget in config["flops_retained"]:
            tag = round(budget * 100)
            for method in ("uniform", "erank_sigma", "erank_cov"):
                add(f"s{seed}_b{tag}_{method}", "E3", "main", method, seed, budget, config["finetune_epochs"])
            for repeat in range(config["random_repeats"]):
                add(f"s{seed}_b{tag}_random_{repeat}", "E3", "main", "random", seed, budget,
                    config["finetune_epochs"], repeat)
        budget = config["ablation_flops_retained"]
        tag = round(budget * 100)
        for method in ("erank_sigma_random_channels", "erank_sigma_compensated"):
            add(f"s{seed}_b{tag}_{method}", "E4", "ablation", method, seed, budget, config["finetune_epochs"])
        add(f"s{seed}_dense_ft", "E5", "dense_control", "original_ft", seed, 1.0, config["finetune_epochs"])
        for method in config["scratch_methods"]:
            add(f"s{seed}_b{tag}_scratch_{method}", "E6", "scratch", method, seed, budget, config["scratch_epochs"])
    assert len(rows) == 63 and len({row['task'] for row in rows}) == len(rows)
    path = run / "planned_runs.json"
    if path.exists():
        assert json.loads(path.read_text()) == rows, "A frozen training manifest already exists"
    else:
        path.write_text(json.dumps(rows, indent=2))
    with (run / "planned_runs.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Frozen training manifest: {len(rows)} jobs, {sum(row['epochs'] for row in rows)} epochs")


if __name__ == "__main__":
    main()
