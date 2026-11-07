import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import scipy
import torch

from common import CONFIG, ROOT, RUN, Data, ResNet20, json_dump, model_cost, on_gpu, seed_all, setup, train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    setup()
    data = Data()
    assert model_cost(ResNet20())["params"] == 269722
    assert model_cost(ResNet20())["macs"] == 40551040
    source = RUN / "source_base"
    source.mkdir(exist_ok=True)
    for name in ("base.py", "common.py", "config.json", "download_data.py"):
        target = source / name
        if not target.exists():
            shutil.copy2(ROOT / "experiments" / name, target)
    json_dump(RUN / "environment.json", {"python": sys.version, "platform": platform.platform(),
              "torch": torch.__version__, "cuda": torch.version.cuda, "numpy": np.__version__,
              "scipy": scipy.__version__, "gpu": torch.cuda.get_device_name(0),
              "visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
              "started_at": time.time()})
    if args.smoke:
        seed_all(987654)
        model = on_gpu(ResNet20())
        train_model(model, data, RUN / "smoke", 2, 0.1, 987654,
                    {"phase": "smoke", "excluded_from_analysis": True}, limit=2048)
        print("SMOKE PASS", flush=True)
        return
    for seed in CONFIG["model_seeds"]:
        seed_all(seed)
        model, result = train_model(on_gpu(ResNet20()), data, RUN / "models" / f"base_s{seed}",
                    CONFIG["base_epochs"], CONFIG["base_lr"], seed,
                    {"phase": "base", "method": "original", "model_seed": seed})
        print("BASE COMPLETE", seed, result["best_val_accuracy"], flush=True)
        del model
        torch.cuda.empty_cache()
    json_dump(RUN / "base_complete.json", {"seeds": CONFIG["model_seeds"], "completed_at": time.time()})
    json_dump(RUN / "status.json", {"state": "base_complete", "updated_at": time.time()})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        json_dump(RUN / "failure.json", {"phase": "base", "traceback": traceback.format_exc(),
                                         "time": time.time()})
        raise
