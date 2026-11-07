"""Shared, resumable ResNet-20 experiments. No torchvision dependency."""
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/config.json"
CONFIG = json.loads(CONFIG_PATH.read_text())
RUN = ROOT / "runs" / CONFIG["protocol"]
WIDTHS = (16, 16, 16, 32, 32, 32, 64, 64, 64)
MAC_UNIT = 18432
MAC_WEIGHTS = np.array([16, 16, 16, 6, 8, 8, 3, 4, 4], dtype=int)
FIXED_MACS = 3 * 16 * 9 * 32 * 32 + 64 * 10
FULL_MACS = int(FIXED_MACS + MAC_UNIT * (MAC_WEIGHTS @ WIDTHS))


def json_dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Launch with exactly physical GPU 0 exposed through its UUID")
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(CONFIG["gpu_memory_gib"] * 1024**3 / total, 0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = CONFIG["deterministic"]
    torch.use_deterministic_algorithms(CONFIG["deterministic"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    RUN.mkdir(parents=True, exist_ok=True)
    existing = RUN / "config.json"
    if existing.exists() and json.loads(existing.read_text()) != CONFIG:
        raise RuntimeError("Frozen run configuration differs; create a new protocol version")
    json_dump(existing, CONFIG)


def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16,
                          enabled=CONFIG["precision"] == "bf16_amp")


class Block(nn.Module):
    def __init__(self, c_in, c_out, hidden, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, hidden, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.stride, self.c_out = stride, c_out

    def forward(self, x):
        shortcut = x[:, :, ::self.stride, ::self.stride]
        extra = self.c_out - shortcut.shape[1]
        if extra:
            shortcut = F.pad(shortcut, (0, 0, 0, 0, extra // 2, extra - extra // 2))
        x = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(x)) + shortcut)


class ResNet20(nn.Module):
    def __init__(self, widths=WIDTHS):
        super().__init__()
        self.widths = list(map(int, widths))
        self.conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.blocks = nn.ModuleList()
        c_in = 16
        for i, hidden in enumerate(self.widths):
            self.blocks.append(Block(c_in, WIDTHS[i], hidden, 2 if i in (3, 6) else 1))
            c_in = WIDTHS[i]
        self.fc = nn.Linear(64, 10)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        for block in self.blocks:
            x = block(x)
        return self.fc(x.mean((2, 3)))


def on_gpu(model):
    return model.to("cuda", memory_format=torch.channels_last)


def model_cost(model):
    return {"params": sum(p.numel() for p in model.parameters()),
            "macs": int(FIXED_MACS + MAC_UNIT * (MAC_WEIGHTS @ model.widths)),
            "flops": int(2 * (FIXED_MACS + MAC_UNIT * (MAC_WEIGHTS @ model.widths))),
            "widths": list(model.widths)}


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ResNet20(checkpoint["widths"])
    model.load_state_dict(checkpoint["state_dict"])
    return on_gpu(model)


class Data:
    def __init__(self):
        images = np.load(ROOT / "data/train_images.npy")
        labels = np.load(ROOT / "data/train_labels.npy")
        assert images.shape == (50000, 3, 32, 32)
        rng = np.random.default_rng(CONFIG["split_seed"])
        train, val = [], []
        for label in range(10):
            ids = rng.permutation(np.flatnonzero(labels == label))
            val.extend(ids[:CONFIG["val_size"] // 10])
            train.extend(ids[CONFIG["val_size"] // 10:])
        rng.shuffle(train)
        rng.shuffle(val)
        self.ids_cpu = {"train": np.array(train), "val": np.array(val)}
        # Fixed class-balanced calibration subset, entirely inside train.
        cal = []
        cal_rng = np.random.default_rng(1701)
        for label in range(10):
            eligible = self.ids_cpu["train"][labels[self.ids_cpu["train"]] == label]
            count = CONFIG["calibration_size"] // 10 + (label < CONFIG["calibration_size"] % 10)
            cal.extend(cal_rng.permutation(eligible)[:count])
        cal_rng.shuffle(cal)
        self.ids_cpu["calibration"] = np.array(cal)
        self.ids_cpu["bn"] = np.array(cal[:CONFIG["bn_calibration_size"]])
        assert not set(train).intersection(val)
        assert set(cal).issubset(train)
        RUN.mkdir(parents=True, exist_ok=True)
        split_path = RUN / "split.npz"
        if split_path.exists():
            stored = np.load(split_path)
            assert all(np.array_equal(stored[k], v) for k, v in self.ids_cpu.items())
        else:
            np.savez(split_path, **self.ids_cpu)
        self.images = torch.from_numpy(images).to("cuda")
        self.labels = torch.from_numpy(labels).to("cuda")
        self.ids = {k: torch.as_tensor(v, device="cuda") for k, v in self.ids_cpu.items()}
        self.mean = torch.tensor((0.4914, 0.4822, 0.4465), device="cuda")[None, :, None, None]
        self.std = torch.tensor((0.247, 0.2435, 0.2616), device="cuda")[None, :, None, None]
        self.test_images = self.test_labels = None

    def batches(self, split, augment=False, seed=0, limit=None, batch_size=None):
        if split == "test":
            if self.test_images is None:
                self.test_images = torch.from_numpy(np.load(ROOT / "data/test_images.npy")).to("cuda")
                self.test_labels = torch.from_numpy(np.load(ROOT / "data/test_labels.npy")).to("cuda")
            images, labels = self.test_images, self.test_labels
            ids = torch.arange(len(labels), device="cuda")
        else:
            images, labels, ids = self.images, self.labels, self.ids[split]
        if limit is not None:
            ids = ids[:limit]
        generator = torch.Generator(device="cuda").manual_seed(seed)
        if augment:
            ids = ids[torch.randperm(len(ids), generator=generator, device="cuda")]
        size = batch_size or CONFIG["batch_size" if augment else "eval_batch_size"]
        for start in range(0, len(ids), size):
            chosen = ids[start:start + size]
            x = images[chosen].float().div_(255)
            if augment:
                x = F.pad(x, (4, 4, 4, 4))
                offsets = torch.randint(0, 9, (len(chosen), 2), generator=generator, device="cuda")
                base = torch.arange(32, device="cuda")
                rows = offsets[:, 0, None] + base
                cols = offsets[:, 1, None] + base
                batch = torch.arange(len(chosen), device="cuda")[:, None, None]
                x = x.permute(0, 2, 3, 1)[batch, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)
                flip = torch.rand((len(chosen), 1, 1, 1), generator=generator, device="cuda") < 0.5
                x = torch.where(flip, x.flip(-1), x)
            x = ((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
            yield x, labels[chosen]


@torch.inference_mode()
def evaluate(model, data, split="val", predictions=False, limit=None):
    model.eval()
    good = torch.zeros((), device="cuda", dtype=torch.long)
    loss = torch.zeros((), device="cuda", dtype=torch.float64)
    count = 0
    saved = []
    for x, y in data.batches(split, limit=limit):
        with amp():
            logits = model(x)
        logits = logits.float()
        predicted = logits.argmax(1)
        good += (predicted == y).sum()
        loss += F.cross_entropy(logits, y, reduction="sum").double()
        count += len(y)
        if predictions:
            saved.append(predicted.cpu().numpy())
    result = {"accuracy": 100 * good.item() / count, "loss": loss.item() / count, "n": count}
    if predictions:
        result["predictions"] = np.concatenate(saved)
    return result


@torch.no_grad()
def recalibrate_bn(model, data):
    layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    momentum = [m.momentum for m in layers]
    for m in layers:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    for x, _ in data.batches("bn", batch_size=128):
        with amp():
            model(x)
    for m, old in zip(layers, momentum):
        m.momentum = old
    model.eval()


def train_model(model, data, directory, epochs, lr, seed, metadata, limit=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    done_path, best_path, last_path = (directory / n for n in ("train_done.json", "best.pt", "last.pt"))
    if done_path.exists():
        return load_model(best_path), json.loads(done_path.read_text())
    model = on_gpu(model)
    seed_all(seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=CONFIG["momentum"],
                                weight_decay=CONFIG["weight_decay"], foreach=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    initial = evaluate(model, data)
    best_acc, best_epoch, best_state = initial["accuracy"], 0, cpu_state(model)
    first_epoch = 0
    previous_seconds = 0.0
    if last_path.exists():
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(last["state_dict"])
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        first_epoch = last["epoch"]
        best_acc, best_epoch, best_state = last["best_acc"], last["best_epoch"], last["best_state"]
        previous_seconds = last["seconds"]
        print("RESUME", directory.name, first_epoch, flush=True)
    json_dump(directory / "spec.json", {**metadata, "epochs": epochs, "lr": lr, "training_seed": seed,
                                         **model_cost(model)})
    start_time = time.monotonic()
    for epoch in range(first_epoch, epochs):
        epoch_start = time.monotonic()
        model.train()
        total_loss = torch.zeros((), device="cuda")
        total_n = 0
        for x, y in data.batches("train", augment=True, seed=seed * 1000 + epoch, limit=limit):
            optimizer.zero_grad(set_to_none=True)
            with amp():
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * len(y)
            total_n += len(y)
        validation = evaluate(model, data)
        if not math.isfinite(validation["loss"]):
            raise RuntimeError(f"Nonfinite validation loss in {directory}")
        if validation["accuracy"] > best_acc:
            best_acc, best_epoch, best_state = validation["accuracy"], epoch + 1, cpu_state(model)
            torch_save(best_path, {"state_dict": best_state, "widths": model.widths,
                                   "best_accuracy": best_acc, "best_epoch": best_epoch})
        row = {"epoch": epoch + 1, "epochs": epochs, "train_loss": total_loss.item() / total_n,
               "val_accuracy": validation["accuracy"], "val_loss": validation["loss"],
               "best_accuracy": best_acc, "lr": optimizer.param_groups[0]["lr"],
               "epoch_seconds": time.monotonic() - epoch_start}
        with (directory / "epochs.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        json_dump(RUN / "status.json", {"state": "training", "task": str(directory.relative_to(RUN)),
                                       **row, "updated_at": time.time()})
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            torch_save(last_path, {"state_dict": cpu_state(model), "widths": model.widths,
                       "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                       "epoch": epoch + 1, "best_acc": best_acc, "best_epoch": best_epoch,
                       "best_state": best_state, "seconds": previous_seconds + time.monotonic() - start_time})
        if epoch == first_epoch or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f"{directory.name}: {epoch+1}/{epochs} val={validation['accuracy']:.2f} "
                  f"best={best_acc:.2f} epoch={row['epoch_seconds']:.2f}s", flush=True)
    model.load_state_dict(best_state)
    torch_save(best_path, {"state_dict": best_state, "widths": model.widths,
                           "best_accuracy": best_acc, "best_epoch": best_epoch})
    summary = {**metadata, **model_cost(model), "best_val_accuracy": best_acc,
               "best_epoch": best_epoch, "epochs": epochs, "initial_val": initial,
               "training_seconds": previous_seconds + time.monotonic() - start_time}
    json_dump(done_path, summary)
    return model, summary
