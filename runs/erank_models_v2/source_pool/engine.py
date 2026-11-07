"""Deterministic, resumable training with saved rank trajectories and no early test access."""
import copy
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

from data import CFG, ROOT, RUN
from models import build, cost, sparse_parameters, to_gpu


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup():
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expose only physical GPU 0 through CUDA_VISIBLE_DEVICES=UUID')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(0.18)
    RUN.mkdir(parents=True, exist_ok=True)


def amp():
    return torch.autocast('cuda', dtype=torch.bfloat16)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_model(path):
    stored = torch.load(path, map_location='cpu', weights_only=False)
    model = build(stored['architecture'])
    model.load_state_dict(stored['state_dict'])
    return to_gpu(model)


@torch.inference_mode()
def evaluate(model, data, split='val', predictions=False, limit=None):
    model.eval()
    good = torch.zeros((), device='cuda', dtype=torch.int64)
    loss = torch.zeros((), device='cuda', dtype=torch.float64)
    n, saved = 0, []
    for x, y in data.batches(split, limit=limit):
        with amp():
            logits = model(x)
        logits = logits.float()
        pred = logits.argmax(1)
        good += (pred == y).sum()
        loss += F.cross_entropy(logits, y, reduction='sum').double()
        n += len(y)
        if predictions:
            saved.append(pred.cpu().numpy())
    result = {'accuracy': 100 * good.item() / n, 'loss': loss.item() / n, 'n': n}
    if predictions:
        result['predictions'] = np.concatenate(saved)
    return result


@torch.no_grad()
def recalibrate(model, data):
    layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    if not layers:
        return
    previous = [m.momentum for m in layers]
    for module in layers:
        module.reset_running_stats()
        module.momentum = None
    model.train()
    for x, _ in data.batches('bn', batch_size=128):
        with amp():
            model(x)
    for module, momentum in zip(layers, previous):
        module.momentum = momentum
    model.eval()


def rank_snapshot(model, data, directory, epoch):
    path = directory / 'rank_dynamics' / f'e{epoch:03d}.json'
    if path.exists():
        return
    from measures import collect_features, spectral_description
    arrays, _ = collect_features(model, data, limit=256, hrank=False)
    rows = [spectral_description(x, with_order=False) for x in arrays]
    save_json(path, {'epoch': epoch, 'layers': rows, 'architecture': model.spec,
                     'calibration_images': 256, 'positions': CFG['rank_positions']})


def train(model, data, directory, metadata, epochs, ft=False, sparse=False, dynamics=False,
          limit=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'train_done.json').exists():
        return load_model(directory / 'best.pt')
    model = to_gpu(model)
    seed = metadata['train_seed']
    family = model.spec['family']
    is_vit = family == 'mini_vit'
    lr = CFG[('vit_' if is_vit else 'resnet_') + ('ft_lr' if ft else 'lr')]
    seed_all(seed)
    if is_vit:
        decay, no_decay = [], []
        for name, parameter in model.named_parameters():
            (no_decay if parameter.ndim <= 1 or name.endswith('pos') else decay).append(parameter)
        optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': CFG['vit_weight_decay']},
                                       {'params': no_decay, 'weight_decay': 0.0}], lr=lr, fused=True)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                    weight_decay=CFG['resnet_weight_decay'], foreach=True)
    initial = evaluate(model, data)
    best_accuracy, best_epoch, best_state = initial['accuracy'], 0, cpu_state(model)
    first_epoch, elapsed_previous = 0, 0.0
    last_path = directory / 'last.pt'
    if last_path.exists():
        last = torch.load(last_path, map_location='cpu', weights_only=False)
        assert last['architecture'] == model.spec
        model.load_state_dict(last['state_dict'])
        optimizer.load_state_dict(last['optimizer'])
        first_epoch, elapsed_previous = last['epoch'], last['seconds']
        best_accuracy, best_epoch, best_state = last['best_accuracy'], last['best_epoch'], last['best_state']
        if (directory / 'epochs.jsonl').exists():
            rows = [json.loads(line) for line in (directory / 'epochs.jsonl').read_text().splitlines()]
            (directory / 'epochs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows if r['epoch'] <= first_epoch))
        print('RESUME', directory.name, first_epoch, flush=True)
    model_cost = cost(model)
    save_json(directory / 'spec.json', {**metadata, 'architecture': model.spec, **model_cost,
                                        'epochs': epochs, 'ft': ft, 'sparse': sparse, 'lr': lr})
    if not (directory / 'best.pt').exists():
        save_torch(directory / 'best.pt', {'architecture': model.spec, 'state_dict': best_state,
                                          'epoch': 0, 'val_accuracy': best_accuracy})
    if dynamics and first_epoch == 0:
        rank_snapshot(model, data, directory, 0)
    start = time.monotonic()
    warmup = min(CFG['vit_warmup_epochs'], epochs // 3) if is_vit else 0
    use_capture = os.environ.get('ERANK_USE_CUDA_GRAPH', '1') == '1' and limit is None
    captured_steps = {}
    if use_capture:
        from capture_step import CapturedStep
    for epoch in range(first_epoch, epochs):
        seed_all(seed * 1000 + epoch)
        if epoch < warmup:
            epoch_lr = lr * (epoch + 1) / warmup
        else:
            epoch_lr = lr * 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, epochs - warmup)))
        for group in optimizer.param_groups:
            group['lr'] = epoch_lr
        started_epoch = time.monotonic()
        model.train()
        accumulated_loss = torch.zeros((), device='cuda')
        samples = 0
        mix_rng = np.random.default_rng(seed * 1000 + epoch)
        for x, y in data.batches('train', augment=True, seed=seed * 1000 + epoch, limit=limit):
            if is_vit:
                lam = float(mix_rng.beta(CFG['vit_mixup_alpha'], CFG['vit_mixup_alpha']))
                permutation = torch.randperm(len(y), device='cuda')
                x = lam * x + (1 - lam) * x[permutation]
            if use_capture:
                if len(y) not in captured_steps:
                    captured_steps[len(y)] = CapturedStep(model, optimizer, x, y, sparse=sparse)
                loss = captured_steps[len(y)](x, y, y[permutation] if is_vit else None,
                                              lam if is_vit else 1.0)
            else:
                optimizer.zero_grad(set_to_none=True)
                with amp():
                    logits = model(x)
                    if is_vit:
                        loss = lam * F.cross_entropy(logits, y, label_smoothing=CFG['vit_label_smoothing'])
                        loss += (1 - lam) * F.cross_entropy(logits, y[permutation], label_smoothing=CFG['vit_label_smoothing'])
                    else:
                        loss = F.cross_entropy(logits, y)
                    if sparse:
                        penalty = sum(p.abs().sum() for p in sparse_parameters(model))
                        loss = loss + CFG['gate_l1' if is_vit else 'bn_l1'] * penalty
                loss.backward()
                if is_vit:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            accumulated_loss += loss.detach() * len(y)
            samples += len(y)
        validation = evaluate(model, data)
        if not math.isfinite(validation['loss']):
            raise RuntimeError(f'Nonfinite loss in {directory}')
        if validation['accuracy'] > best_accuracy:
            best_accuracy, best_epoch, best_state = validation['accuracy'], epoch + 1, cpu_state(model)
            save_torch(directory / 'best.pt', {'architecture': model.spec, 'state_dict': best_state,
                                              'epoch': best_epoch, 'val_accuracy': best_accuracy})
        if dynamics and epoch + 1 in CFG['rank_epochs']:
            rank_snapshot(model, data, directory, epoch + 1)
        row = {'epoch': epoch + 1, 'epochs': epochs, 'train_loss': accumulated_loss.item() / samples,
               'val_accuracy': validation['accuracy'], 'val_loss': validation['loss'],
               'best_accuracy': best_accuracy, 'lr': epoch_lr, 'seconds': time.monotonic() - started_epoch}
        with (directory / 'epochs.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        group = f'{metadata["family"]}_{metadata["dataset"]}'
        save_json(RUN / 'status' / f'{group}.json', {'state': 'training', 'task': directory.name,
                                                   **row, 'updated_at': time.time()})
        if (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            save_torch(last_path, {'architecture': model.spec, 'state_dict': cpu_state(model),
                                   'optimizer': optimizer.state_dict(), 'epoch': epoch + 1,
                                   'best_accuracy': best_accuracy, 'best_epoch': best_epoch,
                                   'best_state': best_state, 'seconds': elapsed_previous + time.monotonic() - start})
        if epoch == first_epoch or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f'{directory.name}: {epoch+1}/{epochs} val={validation["accuracy"]:.2f} '
                  f'best={best_accuracy:.2f} epoch={row["seconds"]:.2f}s', flush=True)
    model.load_state_dict(best_state)
    save_json(directory / 'train_done.json', {**metadata, 'architecture': model.spec, **model_cost,
              'epochs': epochs, 'initial_val': initial, 'best_val_accuracy': best_accuracy,
              'best_epoch': best_epoch, 'training_seconds': elapsed_previous + time.monotonic() - start})
    return model
