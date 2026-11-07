"""Continue each group's pool when its base phase ends, then finalize all results."""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from data import CFG, ROOT, RUN
from engine import save_json


def alive(pid):
    path = Path(f'/proc/{pid}/cmdline')
    return path.exists() and b'experiments_v2' in path.read_bytes()


def main():
    env = os.environ.copy()
    uuid = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    env.update(CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    records = json.loads((RUN / 'processes.json').read_text())
    children = {}
    while True:
        complete = 0
        for record in records:
            family, dataset = record['group']
            group = f'{family}_{dataset}'
            if group in children and children[group].poll() is not None:
                if children[group].returncode:
                    raise RuntimeError(f'Worker exited with failure: {record}')
            if alive(record['pid']):
                continue
            status = json.loads((RUN / 'status' / f'{group}.json').read_text())
            if status['state'] == 'trainings_complete':
                complete += 1
                continue
            if record['phase'] != 'base' or status['state'] != 'bases_complete':
                raise RuntimeError(f'Worker stopped before completion: {record}; status={status}')
            logfile = RUN / f'pool_{group}.log'
            with logfile.open('a') as stream:
                process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'experiments_v2/pool.py'), family, dataset],
                           cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            children[group] = process
            record.update(pid=process.pid, phase='pool', log=str(logfile), started_at=time.time())
            save_json(RUN / 'processes.json', records)
            print('LAUNCHED POOL', group, process.pid, flush=True)
        if complete == len(records):
            break
        time.sleep(10)
    with (RUN / 'finalize.log').open('a') as stream:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'experiments_v2/finalize.py')],
                     cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    save_json(RUN / 'finalize_process.json', {'pid': process.pid, 'gpu_uuid': uuid, 'started_at': time.time()})
    print('LAUNCHED FINAL TEST AND LATENCY', process.pid, flush=True)
    code = process.wait()
    if code:
        raise RuntimeError(f'Finalization exited {code}')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        save_json(RUN / 'failures' / 'controller.json', {'traceback': traceback.format_exc()})
        raise
