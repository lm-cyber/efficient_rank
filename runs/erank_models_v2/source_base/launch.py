"""Launch four independent workers, all pinned to physical GPU 0; no publication."""
import argparse
import json
import os
import subprocess
import sys
import time

from data import CFG, ROOT, RUN
from engine import save_json
from manifest import freeze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['base', 'pool', 'finalize', 'verify'])
    args = parser.parse_args()
    freeze()
    uuid = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    marker = RUN / 'processes.json'
    previous = json.loads(marker.read_text()) if marker.exists() else []
    for item in previous:
        proc = ROOT.__class__(f'/proc/{item["pid"]}/cmdline')
        if proc.exists() and b'experiments_v2' in proc.read_bytes():
            raise RuntimeError(f'A worker remains active: {item}')
    if args.phase in ('base', 'pool'):
        if not (ROOT / 'data/cifar100/train_images.npy').exists():
            raise RuntimeError('CIFAR-100 download must finish before launching all four groups')
        pairs = [(family, dataset) for family in CFG['families'] for dataset in CFG['datasets']]
    else:
        pairs = [()]
    records = []
    for pair in pairs:
        name = args.phase + ('_' + '_'.join(pair) if pair else '')
        command = [sys.executable, '-u', str(ROOT / 'experiments_v2' / f'{args.phase}.py'), *pair]
        logfile = RUN / f'{name}.log'
        with logfile.open('a') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        records.append({'pid': process.pid, 'phase': args.phase, 'group': pair, 'gpu_index': 0,
                        'gpu_uuid': uuid, 'log': str(logfile), 'started_at': time.time()})
    save_json(marker, records)
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
