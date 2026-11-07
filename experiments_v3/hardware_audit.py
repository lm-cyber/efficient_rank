"""Pair unchanged trained networks under eager/CUDA Graphs and cuDNN policies."""
import argparse
import gc
import json
import os
import time

from runtime import CFG, OLD, RUN, engine, environment, save_json, setup
import numpy as np
import pandas as pd
import torch
from finalize import external_gpu_processes

MODES = ('eager_deterministic', 'graph_deterministic', 'eager_tuned', 'graph_tuned')
REPORT_MODES = ('original_v2_eager', *MODES)


def policy(mode):
    tuned = mode.endswith('_tuned')
    torch.use_deterministic_algorithms(not tuned)
    torch.backends.cudnn.deterministic = not tuned
    torch.backends.cudnn.benchmark = tuned


@torch.inference_mode()
def prepare(model, x, mode):
    policy(mode)
    model.eval()
    def eager():
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            return model(x)
    if not mode.startswith('graph'):
        for _ in range(CFG['latency_warmup']):
            eager()
        return eager, {'mode_output_error': 0., 'graph_output_error': None}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(CFG['latency_warmup']):
            reference = eager()
    torch.cuda.current_stream().wait_stream(stream)
    reference = reference.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = eager()
    graph.replay()
    error = float((reference.float() - output.float()).abs().max())
    assert torch.allclose(reference, output, atol=1e-4, rtol=1e-4), error
    def replay():
        graph.replay()
        return output
    return replay, {'mode_output_error': 0., 'graph_output_error': error}


@torch.inference_mode()
def timed(operation):
    for _ in range(10):
        operation()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(CFG['latency_repeats'])]
    ends = [torch.cuda.Event(enable_timing=True) for _ in starts]
    start_wall = time.perf_counter()
    for begin, end in zip(starts, ends):
        begin.record()
        operation()
        end.record()
    torch.cuda.synchronize()
    wall_ms = 1000 * (time.perf_counter() - start_wall) / len(starts)
    values = [begin.elapsed_time(end) for begin, end in zip(starts, ends)]
    return {'median_ms': float(np.median(values)), 'wall_mean_ms': wall_ms,
            'milliseconds': values}


def inventory():
    rows = pd.read_csv(OLD / 'results.csv')
    return rows[(rows.dataset == 'cifar100') & (
        (rows.method == 'dense') | ((rows.phase == 'main') & (rows.budget == .75)) |
        (rows.phase == 'architecture_scratch'))].to_dict('records')


def report():
    records = []
    for path in [*(RUN / 'hardware_audit').glob('*.json'), *(RUN / 'hardware_original').glob('*.json')]:
        item = json.loads(path.read_text())
        if 'measurements' not in item:
            continue
        for measurement in item['measurements']:
            records.append({**{k: item[k] for k in ('task','family','model_seed','method','phase','round')},
                            **{k: v for k, v in measurement.items() if k != 'milliseconds'}})
    frame = pd.DataFrame(records)
    if frame.empty:
        return
    frame.to_csv(RUN / 'hardware_audit_rounds.csv', index=False)
    grouped = frame.groupby(['task','family','model_seed','method','phase','mode','batch'], as_index=False).median(numeric_only=True)
    dense = grouped[grouped.method == 'dense'][['family','model_seed','mode','batch','median_ms']]
    dense = dense.rename(columns={'median_ms': 'dense_ms'})
    paired = grouped.merge(dense, on=['family','model_seed','mode','batch'], how='left', validate='many_to_one')
    paired['speed_change_pct'] = 100 * (paired.dense_ms / paired.median_ms - 1)
    paired.to_csv(RUN / 'hardware_audit_per_seed.csv', index=False)
    summary = paired.groupby(['family','method','phase','mode','batch'], as_index=False).agg(
        speed_change_pct=('speed_change_pct','mean'), speed_sd=('speed_change_pct','std'),
        latency_ms=('median_ms','mean'), n=('speed_change_pct','count'))
    summary.to_csv(RUN / 'hardware_audit_summary.csv', index=False)
    lines = ['# Аудит реальной скорости', '',
             'CIFAR-100; те же checkpoint, BF16, входы на GPU; mean по трём seed после median трёх раундов. '
             'Скорость относительно dense **в том же режиме**. Графы статические, создание и копирование входа исключены. '
             'Original v2 eager повторяет исходный autocast-контекст с кешем. В остальных режимах кеш autocast отключён. '
             'Tuned включает autotuning cuDNN и отключает требование детерминизма. '
             'Неполные строки не являются итогом.', '']
    for family in ('resnet20','mini_vit'):
        for batch in CFG['latency_batches']:
            part = summary[(summary.family == family) & (summary.batch == batch)]
            lines += [f'## {family}, batch {batch}', '',
                      '| Метод | Original v2 eager | Eager deterministic, uncached | Graph deterministic, uncached | Eager tuned, uncached | Graph tuned, uncached |',
                      '| --- | ---: | ---: | ---: | ---: | ---: |']
            for method in part.method.unique():
                values = []
                for mode in REPORT_MODES:
                    row = part[(part.method == method) & (part['mode'] == mode)]
                    values.append('—' if row.empty else f'{row.iloc[0].speed_change_pct:+.1f}% (n={int(row.iloc[0].n)})')
                lines.append('| ' + ' | '.join([method,*values]) + ' |')
            lines += ['']
    (RUN / 'hardware_audit.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    setup()
    save_json(RUN / 'hardware_environment.json', environment())
    jobs = inventory()
    if args.seed is not None:
        jobs = [j for j in jobs if j['model_seed'] == args.seed]
    folder = RUN / 'hardware_audit'
    folder.mkdir(exist_ok=True)
    for repetition in range(CFG['latency_rounds']):
        for index in np.random.default_rng(8111 + repetition).permutation(len(jobs)):
            job = jobs[index]
            path = folder / f'{job["task"]}_r{repetition}.json'
            if path.exists():
                continue
            if external_gpu_processes():
                raise RuntimeError('GPU 0 must be exclusive for latency audit')
            model = engine.load_model(OLD / 'models' / job['task'] / 'best.pt').eval()
            measurements = []
            for batch in CFG['latency_batches']:
                generator = torch.Generator(device='cuda').manual_seed(1107)
                x = torch.randn(batch,3,32,32,device='cuda',generator=generator).contiguous(memory_format=torch.channels_last)
                policy('eager_deterministic')
                with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                    reference = model(x).clone()
                for mode in np.random.default_rng(2000 + repetition).permutation(MODES):
                    operation, validation = prepare(model, x, str(mode))
                    with torch.inference_mode():
                        output = operation()
                        validation['mode_output_error'] = float((output.float()-reference.float()).abs().max())
                        validation['argmax_agreement_pct'] = float((output.argmax(1)==reference.argmax(1)).float().mean()*100)
                    measurements.append({'batch': batch, 'mode': str(mode), **timed(operation), **validation})
                    del operation, output
                    gc.collect()
                del x, reference
            if external_gpu_processes():
                raise RuntimeError('Another process appeared during latency measurement; current series discarded')
            save_json(path, {**{k:job[k] for k in ('task','family','model_seed','method','phase')},
                             'round': repetition, 'measurements': measurements, 'at': time.time()})
            del model
            gc.collect()
            torch.cuda.empty_cache()
            print('AUDIT', repetition, job['task'], flush=True)
            report()
    save_json(RUN / 'hardware_audit_complete.json', {'at': time.time(), 'tasks': len(jobs), 'rounds': CFG['latency_rounds']})


if __name__ == '__main__':
    main()
