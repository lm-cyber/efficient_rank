"""Frozen segmentation configuration and isolated run state."""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'experiments_segmentation/config.json').read_text())
RUN=ROOT/'runs'/CFG['protocol']
WEIGHTS=ROOT/'data/segformer_b1'


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def freeze():
    RUN.mkdir(parents=True,exist_ok=True)
    for source,name in [(ROOT/'experiments_segmentation/config.json','config.json'),(ROOT/'experiment_pool_segmentation.md','protocol.md')]:
        destination=RUN/name
        if destination.exists():assert destination.read_bytes()==source.read_bytes(),f'Frozen {name} changed'
        else:shutil.copy2(source,destination)
    marker=RUN/'preregistration.json'
    if not marker.exists():
        save_json(marker,{'at':time.time(),'scope':'Exploratory follow-up, before new scoring/validation/training',
                         'sha256':{name:hashlib.sha256((RUN/name).read_bytes()).hexdigest() for name in ['protocol.md','config.json']}})


def seed_all(seed):
    random.seed(seed);np.random.seed(seed%2**32);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def setup(gpu=True):
    freeze()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    if gpu:
        uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==uuid
        assert torch.cuda.device_count()==1
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
    snapshot=RUN/'source_processes'/str(os.getpid())
    snapshot.mkdir(parents=True,exist_ok=True)
    for path in (ROOT/'experiments_segmentation').glob('*'):
        if path.is_file():shutil.copy2(path,snapshot/path.name)
    save_json(snapshot/'process.json',{'pid':os.getpid(),'argv':sys.argv,'at':time.time()})


def amp():
    return torch.autocast('cuda',dtype=torch.bfloat16)


def external_gpu_processes():
    uuid=os.environ['CUDA_VISIBLE_DEVICES']
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'],text=True)
    return [line for line in lines.splitlines() if line.split(',')[0].strip()==uuid and int(line.split(',')[1])!=os.getpid()]


def save_checkpoint(path,model,**extra):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(f'.{os.getpid()}.tmp')
    from network import widths
    torch.save({'widths':widths(model),'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},**extra},temporary)
    temporary.replace(path)
