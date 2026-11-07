"""After every training finishes: held-out test, exclusive latency benchmark, report."""
import json
import os
import subprocess
import time
import traceback

import numpy as np
import torch

from data import CFG, RUN, Data
from engine import amp, evaluate, load_model, save_json, setup
from manifest import jobs


def check_inventory():
    inventory = jobs()
    actual = {p.parent.name for p in (RUN / 'models').glob('*/train_done.json')}
    expected = {job['task'] for job in inventory}
    if actual != expected:
        raise RuntimeError(f'Incomplete training inventory: missing={expected-actual}, extra={actual-expected}')
    total_epochs = 0
    summaries = {}
    for job in inventory:
        folder = RUN / 'models' / job['task']
        saved = json.loads((folder / 'train_done.json').read_text())
        summaries[job['task']] = saved
        assert saved['epochs'] == job['epochs']
        assert all(saved[k] == job[k] for k in ('family', 'dataset', 'model_seed', 'phase', 'method', 'budget'))
        assert saved['architecture']['family'] == job['family']
        assert saved['architecture']['classes'] == (100 if job['dataset'] == 'cifar100' else 10)
        assert (folder / 'best.pt').exists()
        rows = [json.loads(line) for line in (folder / 'epochs.jsonl').read_text().splitlines()]
        assert [row['epoch'] for row in rows] == list(range(1, job['epochs'] + 1))
        total_epochs += saved['epochs']
    for family in CFG['families']:
        for dataset in CFG['datasets']:
            for seed in CFG['seeds']:
                prefix = f'{family}_{dataset}_s{seed}'
                full_macs = summaries[prefix + '_dense']['macs']
                for budget in CFG['budgets']:
                    tag = round(100 * budget)
                    channel_costs = {summaries[f'{prefix}_ft_b{tag}_{method}']['macs']
                                     for method in CFG['main_methods'] if method != 'svd'}
                    assert len(channel_costs) == 1, (prefix, budget, channel_costs)
                    svd_cost = summaries[f'{prefix}_ft_b{tag}_svd']['macs']
                    assert 0 <= (full_macs * budget - svd_cost) / full_macs < .015
    manifest = {'trainings': len(inventory), 'epochs': total_epochs,
                'latest_training_mtime': max((RUN / 'models' / j['task'] / 'train_done.json').stat().st_mtime for j in inventory),
                'checked_at': time.time(), 'tasks': inventory}
    path = RUN / 'frozen_test_manifest.json'
    if not path.exists():
        save_json(path, manifest)
    return inventory


def final_test(inventory):
    for dataset in CFG['datasets']:
        data = Data(dataset)
        for job in inventory:
            if job['dataset'] != dataset:
                continue
            directory = RUN / 'models' / job['task']
            if (directory / 'test.json').exists():
                continue
            save_json(RUN / 'status.json', {'state': 'held_out_test', 'task': job['task'], 'updated_at': time.time()})
            model = load_model(directory / 'best.pt')
            checkpoint = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
            summary = json.loads((directory / 'train_done.json').read_text())
            assert checkpoint['epoch'] == summary['best_epoch']
            assert checkpoint['architecture'] == summary['architecture']
            del checkpoint
            result = evaluate(model, data, 'test', predictions=True)
            predictions = result.pop('predictions')
            np.savez_compressed(directory / 'test_predictions.npz', predictions=predictions)
            save_json(directory / 'test.json', {**result, 'evaluated_at': time.time()})
            print('TEST', job['task'], result, flush=True)
            del model
        del data
        torch.cuda.empty_cache()
    save_json(RUN / 'test_complete.json', {'count': len(inventory), 'at': time.time()})


def external_gpu_processes():
    uuid = os.environ['CUDA_VISIBLE_DEVICES']
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                      '--format=csv,noheader,nounits'], text=True)
    external = []
    for line in output.splitlines():
        fields = [x.strip() for x in line.split(',', 2)]
        if len(fields) == 3 and fields[0] == uuid and int(fields[1]) != os.getpid():
            external.append({'pid': int(fields[1]), 'name': fields[2]})
    return external


def wait_exclusive():
    while True:
        external = external_gpu_processes()
        if not external:
            return
        save_json(RUN / 'status.json', {'state': 'waiting_for_exclusive_latency',
                                       'external_processes': external, 'updated_at': time.time()})
        print('Latency deferred: external GPU 0 compute processes', external, flush=True)
        time.sleep(30)


@torch.inference_mode()
def timed_model(model, batch):
    model.eval()
    generator = torch.Generator(device='cuda').manual_seed(1107)
    x = torch.randn(batch, 3, 32, 32, device='cuda', generator=generator).contiguous(memory_format=torch.channels_last)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(CFG['latency_repeats'])]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(CFG['latency_repeats'])]
    with amp():
        for _ in range(CFG['latency_warmup']):
            model(x)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        for start, end in zip(starts, ends):
            start.record()
            model(x)
            end.record()
        torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - wall_start
    values = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return {'batch': batch, 'milliseconds': values, 'median_ms': float(np.median(values)),
            'p25_ms': float(np.quantile(values, 0.25)), 'p75_ms': float(np.quantile(values, 0.75)),
            'images_per_second': float(1000 * batch / np.median(values)),
            'wall_mean_ms': 1000 * wall_seconds / CFG['latency_repeats'],
            'wall_images_per_second': batch * CFG['latency_repeats'] / wall_seconds}


def latency(inventory):
    folder = RUN / 'latency'
    folder.mkdir(exist_ok=True)
    for repetition in range(CFG['latency_rounds']):
        order = np.random.default_rng(6011 + repetition).permutation(len(inventory))
        for index in order:
            job = inventory[int(index)]
            path = folder / f'{job["task"]}_r{repetition}.json'
            if path.exists():
                continue
            wait_exclusive()
            model = load_model(RUN / 'models' / job['task'] / 'best.pt')
            measurements = [timed_model(model, b) for b in CFG['latency_batches']]
            if external_gpu_processes():
                del model
                print('Discarding contaminated timing', job['task'], flush=True)
                # Retry this model once exclusive access is restored; never accept a contaminated round.
                while True:
                    wait_exclusive()
                    model = load_model(RUN / 'models' / job['task'] / 'best.pt')
                    measurements = [timed_model(model, b) for b in CFG['latency_batches']]
                    if not external_gpu_processes():
                        break
                    del model
            save_json(path, {'task': job['task'], 'round': repetition, 'measurements': measurements,
                             'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'], 'measured_at': time.time()})
            del model
            save_json(RUN / 'status.json', {'state': 'latency', 'task': job['task'],
                                           'round': repetition + 1, 'updated_at': time.time()})
        environment = subprocess.check_output(['nvidia-smi', '-i', '0',
            '--query-gpu=name,uuid,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem',
            '--format=csv'], text=True)
        (folder / f'environment_r{repetition}.txt').write_text(environment)
        print('LATENCY ROUND COMPLETE', repetition + 1, flush=True)


def main():
    setup()
    inventory = check_inventory()
    final_test(inventory)
    latency(inventory)
    from report import main as report
    report()
    save_json(RUN / 'status.json', {'state': 'complete', 'completed_trainings': len(inventory),
                                   'completed_tests': len(inventory), 'epochs': sum(j['epochs'] for j in inventory),
                                   'latency_rounds': CFG['latency_rounds'], 'updated_at': time.time()})
    print('FULL V2 POOL COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        save_json(RUN / 'failures' / 'finalize.json', {'traceback': traceback.format_exc()})
        raise
