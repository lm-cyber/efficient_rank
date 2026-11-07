"""Repeat the original v2 autocast context to distinguish it from uncached controls."""
import gc
import json
import time

from runtime import CFG, OLD, RUN, engine, save_json, setup
from hardware_audit import inventory, policy, report
from finalize import external_gpu_processes, timed_model
import numpy as np
import torch


def main():
    setup();policy('eager_deterministic')
    folder=RUN/'hardware_original';folder.mkdir(exist_ok=True)
    jobs=inventory()
    for repetition in range(CFG['latency_rounds']):
        for index in np.random.default_rng(8111+repetition).permutation(len(jobs)):
            job=jobs[index];path=folder/f'{job["task"]}_r{repetition}.json'
            if path.exists():continue
            assert not external_gpu_processes(),'GPU 0 must be exclusive'
            model=engine.load_model(OLD/'models'/job['task']/'best.pt').eval()
            measurements=[{'mode':'original_v2_eager',**timed_model(model,b)} for b in CFG['latency_batches']]
            assert not external_gpu_processes(),'Discard contaminated measurement'
            save_json(path,{**{k:job[k] for k in ('task','family','model_seed','method','phase')},
                            'round':repetition,'measurements':measurements,'at':time.time()})
            del model;gc.collect();torch.cuda.empty_cache()
            print('ORIGINAL EAGER',repetition,job['task'],flush=True)
    report()
    save_json(RUN/'hardware_original_complete.json',{'at':time.time(),'tasks':len(jobs),'rounds':CFG['latency_rounds']})


if __name__=='__main__':main()
