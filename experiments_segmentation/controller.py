"""Run the preregistered segmentation pool on physical GPU 0."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from common import CFG,ROOT,RUN,freeze,save_json


def start(script,name,env,*args):
    with (RUN/(name+'.log')).open('a') as stream:
        process=subprocess.Popen([sys.executable,'-u',f'experiments_segmentation/{script}.py',*args],
            cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    save_json(RUN/(name+'_process.json'),{'pid':process.pid,'script':script,'args':args,'at':time.time()})
    return process


def wait(process,name):
    while process.poll() is None:time.sleep(10)
    if process.returncode:raise RuntimeError(f'{name} failed with exit {process.returncode}; see {name}.log')


def main():
    freeze()
    uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    while not (ROOT/'data/ade20k/extracted.json').exists() or not (RUN/'model_source.json').exists():
        save_json(RUN/'status.json',{'state':'preparing_data','at':time.time()})
        pid=json.loads((RUN/'download_process.json').read_text())['pid']
        path=Path(f'/proc/{pid}/stat')
        if not path.exists() or path.read_text().split()[2]=='Z':raise RuntimeError('Dataset/checkpoint preparation stopped')
        time.sleep(15)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name',
                                       '--format=csv,noheader,nounits'],text=True)
    busy=[line for line in processes.splitlines() if line.split(',')[0].strip()==uuid]
    if busy:raise RuntimeError('GPU 0 is occupied; do not launch a duplicate controller or overlap another experiment: '+str(busy))
    for script,marker in [('verify','verification.json'),('scoring','scores.json'),('plan','planned_runs.json')]:
        if not (RUN/marker).exists():
            save_json(RUN/'status.json',{'state':script,'at':time.time()})
            wait(start(script,script,env),script)
    workers={}
    for seed in CFG['seeds']:
        if not (RUN/'worker_complete'/f's{seed}.json').exists():
            workers[seed]=start('train_worker',f'train_s{seed}',env,str(seed))
    while workers:
        for seed,process in list(workers.items()):
            if process.poll() is None:continue
            if process.returncode:raise RuntimeError(f'Training seed {seed} failed with exit {process.returncode}')
            del workers[seed]
        done=len(list((RUN/'models').glob('*/train_done.json')))
        save_json(RUN/'status.json',{'state':'training','completed':done,'planned':63,
                  'active':{str(s):p.pid for s,p in workers.items()},'at':time.time()})
        if workers:time.sleep(10)
    assert len(list((RUN/'models').glob('*/train_done.json')))==63
    if not (RUN/'complete.json').exists():
        save_json(RUN/'status.json',{'state':'finalize','at':time.time()})
        wait(start('finalize','finalize',env),'finalize')
    save_json(RUN/'status.json',{'state':'complete','at':time.time()})


if __name__=='__main__':
    try:main()
    except Exception:
        save_json(RUN/'failures/controller.json',{'at':time.time(),'traceback':traceback.format_exc()})
        raise
