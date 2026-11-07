"""Wait for exclusive hardware audit, validate, run four workers, then final evaluation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from runtime import CFG, ROOT, RUN, freeze, save_json
from run_group import inventory, save_inventory


def launch(command,name,env):
    logfile=RUN/(name+'.log')
    with logfile.open('a') as stream:
        process=subprocess.Popen([sys.executable,'-u',*command],cwd=ROOT,env=env,
                                 stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    save_json(RUN/(name+'_process.json'),{'pid':process.pid,'command':command,'at':time.time()})
    return process


def wait_success(process,label):
    while process.poll() is None:
        time.sleep(10)
    if process.returncode:
        raise RuntimeError(f'{label} exited {process.returncode}; see {label}.log')


def main():
    freeze();save_inventory()
    uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    while not (RUN/'hardware_audit_complete.json').exists():
        save_json(RUN/'controller_status.json',{'state':'waiting_hardware_audit','at':time.time()})
        pid=json.loads((RUN/'hardware_process.json').read_text())['pid']
        path=Path(f'/proc/{pid}/stat')
        if not path.exists() or path.read_text().split()[2]=='Z':
            raise RuntimeError('Hardware audit stopped without completion marker')
        time.sleep(15)
    if not (RUN/'hardware_original_complete.json').exists():
        save_json(RUN/'controller_status.json',{'state':'hardware_original_eager','at':time.time()})
        wait_success(launch(['experiments_v3/hardware_original.py'],'hardware_original',env),'hardware_original')
    while not (ROOT/'data/tinyimagenet/decoded.json').exists():
        save_json(RUN/'controller_status.json',{'state':'waiting_tiny_data','at':time.time()})
        pid=json.loads((RUN/'download_process.json').read_text())['pid']
        if not Path(f'/proc/{pid}/cmdline').exists():raise RuntimeError('Tiny data preparation stopped')
        time.sleep(15)
    if not (RUN/'verification.json').exists():
        wait_success(launch(['experiments_v3/verify.py'],'verify',env),'verify')
    # Prioritize the three new dense models; the fourth worker processes existing CIFAR models.
    queue=[('resnet18','tinyimagenet',s) for s in CFG['seeds']]
    queue += [(family,dataset,s) for family,dataset in CFG['groups'] if family!='resnet18' for s in CFG['seeds']]
    active={}
    while queue or active:
        for name,process in list(active.items()):
            if process.poll() is None:continue
            if process.returncode:raise RuntimeError(f'{name} failed: exit {process.returncode}')
            del active[name]
        while queue and len(active)<4:
            family,dataset,seed=queue.pop(0);name=f'{family}_{dataset}_s{seed}'
            if (RUN/'group_complete'/f'{name}.json').exists():continue
            active[name]=launch(['experiments_v3/run_group.py',family,dataset,str(seed)],name,env)
            print('LAUNCH',name,active[name].pid,flush=True)
        save_json(RUN/'controller_status.json',{'state':'training','active':{n:p.pid for n,p in active.items()},
                                              'queued':queue,'at':time.time()})
        if active:time.sleep(10)
    expected=len(inventory());actual=len(list((RUN/'models').glob('*/train_done.json')))
    assert expected==actual,(expected,actual)
    save_json(RUN/'controller_status.json',{'state':'final_evaluation','trainings':actual,'at':time.time()})
    wait_success(launch(['experiments_v3/finalize.py'],'finalize',env),'finalize')
    save_json(RUN/'controller_status.json',{'state':'complete','trainings':actual,'at':time.time()})
    print('FOLLOWUP COMPLETE',flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        save_json(RUN/'failures/controller.json',{'traceback':traceback.format_exc(),'at':time.time()})
        raise
