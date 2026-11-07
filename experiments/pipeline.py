"""Execute the preregistered diagnostic, pruning, scratch, and held-out phases."""
import copy
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from common import (CONFIG, ROOT, RUN, WIDTHS, MAC_WEIGHTS, Data, ResNet20, evaluate,
                    json_dump, load_model, model_cost, on_gpu, recalibrate_bn, seed_all,
                    setup, train_model)
from methods import (collect_activations, describe_activations, global_random_widths,
                     prune, random_orders, spectral_scores, uniform_with_deletion_count,
                     widths_at_budget)


def np_savez(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def stage_status(phase, **details):
    json_dump(RUN / "status.json", {"state": phase, **details, "updated_at": time.time()})
    print("PHASE", phase, details, flush=True)


def representation_diagnostics(base, data, seed, directory):
    cache = directory / "activations.npz"
    if cache.exists():
        with np.load(cache) as stored:
            arrays = [stored[f"block_{i}"] for i in range(9)]
    else:
        stage_status("collect_activations", model_seed=seed)
        arrays = collect_activations(base, data)
        np_savez(cache, **{f"block_{i}": x for i, x in enumerate(arrays)})
    description = directory / "representation.json"
    if description.exists():
        values = json.loads(description.read_text())
        rows, orders = values["layers"], values["orders"]
    else:
        stage_status("spectral_diagnostics", model_seed=seed)
        rows, orders = describe_activations(arrays)
        json_dump(description, {"model_seed": seed, "layers": rows, "orders": orders})
    subsets_path = directory / "calibration_subsets.json"
    if not subsets_path.exists():
        subsets = []
        for subset_seed in (101, 202, 303):
            selected = np.random.default_rng(subset_seed).choice(len(arrays[0]), 512, replace=False)
            for i, array in enumerate(arrays):
                scores, _, _ = spectral_scores(array[selected].reshape(-1, array.shape[-1]))
                subsets.append({"model_seed": seed, "subset_seed": subset_seed,
                                "n_images": 512, "block": i + 1, **scores})
        json_dump(subsets_path, subsets)
    invariance_path = directory / "rescaling.json"
    if not invariance_path.exists():
        stage_status("rescaling_diagnostic", model_seed=seed)
        x, _ = next(data.batches("calibration", batch_size=128))
        base.eval()
        with torch.inference_mode():
            reference = base(x)
        records = []
        for i in (0, 4, 8):
            modified = copy.deepcopy(base).eval()
            scales = np.exp(np.random.default_rng(17070 + seed + i).uniform(-2.3, 2.3, WIDTHS[i]))
            scale = torch.as_tensor(scales, device="cuda", dtype=torch.float32)
            with torch.inference_mode():
                block = modified.blocks[i]
                block.bn1.weight.mul_(scale)
                block.bn1.bias.mul_(scale)
                block.conv2.weight.div_(scale[None, :, None, None])
                actual = modified(x)
                error = float((actual - reference).abs().max().item())
                invariant = bool(torch.allclose(actual, reference, atol=1e-4, rtol=1e-4))
                if not invariant:
                    raise RuntimeError(f"Functional rescaling check failed: {error}")
                agreement = float((actual.argmax(1) == reference.argmax(1)).float().mean().item())
            altered, _, _ = spectral_scores((arrays[i] * scales).reshape(-1, WIDTHS[i]))
            records.append({"model_seed": seed, "block": i + 1,
                            "original_sigma": rows[i]["erank_sigma"],
                            "original_cov": rows[i]["erank_cov"],
                            "rescaled_sigma": altered["erank_sigma"],
                            "rescaled_cov": altered["erank_cov"],
                            "max_abs_logit_difference": error,
                            "prediction_agreement": agreement, "scales": scales.tolist()})
            del modified
        json_dump(invariance_path, records)
    return arrays, rows, orders


def sensitivity(base, data, seed, directory, ranks, orders):
    path = directory / "sensitivity.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    done = {(r["block"], r["removed_fraction_requested"]) for r in rows}
    baseline_path = directory / "dense_bn_control.json"
    if baseline_path.exists():
        control = json.loads(baseline_path.read_text())
    else:
        dense = copy.deepcopy(base)
        before = evaluate(dense, data)
        recalibrate_bn(dense, data)
        control = {"before": before, "after_bn": evaluate(dense, data)}
        json_dump(baseline_path, control)
        del dense
    for block in range(9):
        for fraction in CONFIG["sensitivity_removed"]:
            if (block + 1, fraction) in done:
                continue
            stage_status("layer_sensitivity", model_seed=seed, block=block + 1, removed=fraction)
            widths = list(WIDTHS)
            widths[block] = max(CONFIG["min_width"], int(np.floor(widths[block] * (1 - fraction))))
            model, masks = prune(base, widths, orders)
            before = evaluate(model, data)
            recalibrate_bn(model, data)
            after = evaluate(model, data)
            rows.append({"model_seed": seed, "block": block + 1, "stage": block // 3 + 1,
                         "removed_fraction_requested": fraction,
                         "removed_fraction_actual": 1 - widths[block] / WIDTHS[block],
                         "normalized_sigma": ranks[block]["normalized_sigma"],
                         "normalized_cov": ranks[block]["normalized_cov"],
                         "before": before, "after_bn": after,
                         "accuracy_drop_before": control["before"]["accuracy"] - before["accuracy"],
                         "accuracy_drop_after_bn": control["after_bn"]["accuracy"] - after["accuracy"],
                         "loss_increase_before": before["loss"] - control["before"]["loss"],
                         "loss_increase_after_bn": after["loss"] - control["after_bn"]["loss"],
                         **model_cost(model)})
            json_dump(path, rows)
            del model


def native_rule_diagnostic(base, data, seed, directory, ranks, orders):
    path = directory / "native_ceil_rule.json"
    if path.exists():
        return
    widths = np.clip(np.ceil([r["erank_sigma"] for r in ranks]).astype(int),
                     CONFIG["min_width"], WIDTHS).tolist()
    removed = sum(WIDTHS) - sum(widths)
    specs = [("erank_ceil", widths), ("uniform_count", uniform_with_deletion_count(removed))]
    for repeat in range(2):
        random_widths, _ = global_random_widths(removed, seed * 100 + repeat)
        specs.append((f"global_random_count_{repeat}", random_widths))
    rows = []
    for method, widths in specs:
        model, masks = prune(base, widths, orders)
        before = evaluate(model, data)
        recalibrate_bn(model, data)
        rows.append({"method": method, "model_seed": seed, "removed": removed,
                     "before": before, "after_bn": evaluate(model, data), **model_cost(model)})
        del model
    json_dump(path, rows)


def run_pruned(base, data, name, spec, widths, orders, arrays=None, compensate=False):
    directory = RUN / "models" / name
    result_path = directory / "result.json"
    if result_path.exists():
        return
    stage_status("pruning", task=name)
    model, masks = prune(base, widths, orders, arrays=arrays, compensate=compensate)
    before = evaluate(model, data)
    recalibrate_bn(model, data)
    after_bn = evaluate(model, data)
    directory.mkdir(parents=True, exist_ok=True)
    json_dump(directory / "channels.json", masks)
    json_dump(directory / "pre_ft.json", {"before": before, "after_bn": after_bn})
    model, result = train_model(model, data, directory, CONFIG["finetune_epochs"],
                 CONFIG["finetune_lr"], 100000 + spec["model_seed"], spec)
    json_dump(result_path, {**result, "before": before, "after_bn": after_bn,
                           "after_ft": evaluate(model, data)})
    del model


def model_pool(base, data, seed, ranks, orders, arrays):
    profiles = {"uniform": WIDTHS,
                "erank_sigma": [r["erank_sigma"] for r in ranks],
                "erank_cov": [r["erank_cov"] for r in ranks]}
    chosen = {}
    for budget in CONFIG["flops_retained"]:
        tag = round(100 * budget)
        chosen[budget] = {name: widths_at_budget(profile, budget) for name, profile in profiles.items()}
        for method, widths in chosen[budget].items():
            spec = {"phase": "main", "method": method, "model_seed": seed,
                    "budget": budget, "selection": "qr", "compensation": False, "repeat": None}
            run_pruned(base, data, f"s{seed}_b{tag}_{method}", spec, widths, orders)
        reference_widths = chosen[budget]["erank_sigma"]
        removed = sum(WIDTHS) - sum(reference_widths)
        target_units = int(MAC_WEIGHTS @ reference_widths)
        for repeat in range(CONFIG["random_repeats"]):
            random_seed = seed * 10000 + tag * 10 + repeat
            widths, attempts = global_random_widths(removed, random_seed, target_units)
            spec = {"phase": "main", "method": "random", "model_seed": seed,
                    "budget": budget, "selection": "qr", "compensation": False, "repeat": repeat,
                    "random_seed": random_seed, "sampling_attempts": attempts}
            run_pruned(base, data, f"s{seed}_b{tag}_random_{repeat}", spec, widths, orders)
    budget = CONFIG["ablation_flops_retained"]
    tag = round(budget * 100)
    widths = chosen[budget]["erank_sigma"]
    for method, selected_orders, compensated in (
            ("erank_sigma_random_channels", random_orders(70000 + seed), False),
            ("erank_sigma_compensated", orders, True)):
        spec = {"phase": "ablation", "method": method, "model_seed": seed,
                "budget": budget, "selection": "qr" if compensated else "random",
                "compensation": compensated, "repeat": None}
        run_pruned(base, data, f"s{seed}_b{tag}_{method}", spec, widths, selected_orders,
                   arrays=arrays, compensate=compensated)
    directory = RUN / "models" / f"s{seed}_dense_ft"
    if not (directory / "result.json").exists():
        model, result = train_model(copy.deepcopy(base), data, directory,
                  CONFIG["finetune_epochs"], CONFIG["finetune_lr"], 100000 + seed,
                  {"phase": "dense_control", "method": "original_ft", "model_seed": seed,
                   "budget": 1.0, "repeat": None})
        json_dump(directory / "result.json", {**result, "after_ft": evaluate(model, data)})
        del model
    for method in CONFIG["scratch_methods"]:
        name = f"s{seed}_b{tag}_scratch_{method}"
        directory = RUN / "models" / name
        if (directory / "result.json").exists():
            continue
        seed_all(30000 + seed)
        model, result = train_model(on_gpu(ResNet20(chosen[budget][method])), data, directory,
                      CONFIG["scratch_epochs"], CONFIG["base_lr"], 30000 + seed,
                      {"phase": "scratch", "method": method, "model_seed": seed,
                       "budget": budget, "repeat": None, "initialization_seed": 30000 + seed})
        json_dump(directory / "result.json", {**result, "after_training": evaluate(model, data)})
        del model


def final_test(data):
    models = sorted(path.parent for path in (RUN / "models").glob("*/train_done.json"))
    planned = json.loads((RUN / "planned_runs.json").read_text())
    expected = {row["task"] for row in planned}
    actual = {path.name for path in models}
    if actual != expected:
        raise RuntimeError(f"Training manifest mismatch before opening test: missing={expected-actual}, extra={actual-expected}")
    manifest = [{"task": d.name, "spec": json.loads((d / "train_done.json").read_text())} for d in models]
    json_dump(RUN / "frozen_test_manifest.json", manifest)
    for directory in models:
        if (directory / "test.json").exists():
            continue
        stage_status("held_out_test", task=directory.name)
        model = load_model(directory / "best.pt")
        result = evaluate(model, data, "test", predictions=True)
        predictions = result.pop("predictions")
        np_savez(directory / "test_predictions.npz", predictions=predictions)
        json_dump(directory / "test.json", result)
        del model
    json_dump(RUN / "test_complete.json", {"models": len(models), "completed_at": time.time()})


def main():
    setup()
    if not (RUN / "base_complete.json").exists():
        raise RuntimeError("Complete base training first")
    data = Data()
    source = RUN / "source_pool"
    source.mkdir(exist_ok=True)
    for path in (ROOT / "experiments").glob("*.py"):
        if not (source / path.name).exists():
            shutil.copy2(path, source / path.name)
    for seed in CONFIG["model_seeds"]:
        base = load_model(RUN / "models" / f"base_s{seed}" / "best.pt")
        directory = RUN / "diagnostics" / f"s{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        arrays, ranks, orders = representation_diagnostics(base, data, seed, directory)
        sensitivity(base, data, seed, directory, ranks, orders)
        native_rule_diagnostic(base, data, seed, directory, ranks, orders)
        model_pool(base, data, seed, ranks, orders, arrays)
        del base, arrays
        torch.cuda.empty_cache()
    final_test(data)
    stage_status("building_report")
    from report import generate_report
    generate_report()
    stage_status("complete", completed_trainings=63)
    print("EXPERIMENT POOL COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        json_dump(RUN / "failure.json", {"phase": "pool", "traceback": traceback.format_exc(),
                                         "time": time.time()})
        raise
