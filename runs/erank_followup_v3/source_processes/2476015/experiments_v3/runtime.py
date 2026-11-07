"""Shared configuration for the isolated follow-up; never writes to v1/v2 runs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments_v2'))
import torch
import data as original_data
import engine

CFG = json.loads((ROOT / 'experiments_v3/config.json').read_text())
RUN = ROOT / 'runs' / CFG['protocol']
OLD = ROOT / 'runs/erank_models_v2'
save_json = engine.save_json


def freeze():
    RUN.mkdir(parents=True, exist_ok=True)
    destination = RUN / 'config.json'
    if destination.exists():
        assert json.loads(destination.read_text()) == CFG, 'Frozen v3 config changed'
    else:
        save_json(destination, CFG)
    destination = RUN / 'protocol.md'
    protocol = ROOT / 'experiment_pool_v3.md'
    if not destination.exists():
        shutil.copy2(protocol, destination)
        save_json(RUN / 'preregistration.json', {'frozen_at': time.time(),
            'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
            'config_sha256': hashlib.sha256((ROOT / 'experiments_v3/config.json').read_bytes()).hexdigest(),
            'prior_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'scope': 'Exploratory follow-up after v2; not a new confirmatory study'})


def setup():
    freeze()
    expected = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == expected
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    engine.RUN = original_data.RUN = RUN
    snapshot = RUN / 'source_processes' / str(os.getpid())
    snapshot.mkdir(parents=True, exist_ok=True)
    for directory in ('experiments', 'experiments_v2', 'experiments_v3'):
        for path in (ROOT / directory).glob('*'):
            if path.is_file():
                target = snapshot / directory / path.name
                target.parent.mkdir(exist_ok=True)
                shutil.copy2(path, target)
    save_json(snapshot / 'process.json', {'pid': os.getpid(), 'argv': sys.argv, 'at': time.time()})


def environment():
    return {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': subprocess.check_output(['nvidia-smi', '-i', '0',
                '--query-gpu=name,uuid,driver_version', '--format=csv'], text=True), 'at': time.time()}
