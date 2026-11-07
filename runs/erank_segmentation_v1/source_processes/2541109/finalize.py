"""Open final ADE validation only after every prescribed training has finished."""
import gc
import hashlib
import json
import time
import traceback

from common import CFG,RUN,amp,external_gpu_processes,save_json,setup
from dataset import evaluation_loader
from metrics import evaluate
from network import build,cost,load,prune,widths
from scoring import collect
from benchmark import measure
import numpy as np
import torch


def inventory():
    jobs=json.loads((RUN/'planned_runs.json').read_text())
    assert len(jobs)==63
    for job in jobs:
        folder=RUN/'models'/job['task']
        done=json.loads((folder/'train_done.json').read_text())
        assert done['steps']==CFG['steps'] and done['cost']['macs']==job['target_macs']
        state=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        assert state['step']==done['best_step'] and state['widths']==job['widths']
    save_json(RUN/'frozen_test_manifest.json',{'tasks':jobs,'at':time.time(),'completed_trainings':len(jobs),
              'total_updates':sum(j['steps'] for j in jobs),'test_opened_after_all_new_training':True})
    return [{'task':'original','method':'original','seed':0,'budget':1.}]+jobs


def get_model(job):
    return build(pretrained=True).cuda().eval() if job['task']=='original' else load(RUN/'models'/job['task']/'best.pt')


def native_diagnostics():
    path=RUN/'native_ceil_validation.json'
    if path.exists():return
    model=build(pretrained=True).cuda().eval();scores=json.loads((RUN/'scores.json').read_text())
    loader=evaluation_loader('validation');baseline,_=evaluate(model,loader);rows=[]
    q=CFG['width_multiple'];full=np.array(widths(model))
    for group in ('low','random','high'):
        for definition in ('sigma','cov'):
            raw=np.array([layer['erank_'+definition] for layer in scores['groups'][group]['layers']])
            candidate=np.clip(np.ceil(raw/q)*q,q,full).astype(int).tolist()
            compressed,_=prune(model,candidate,scores['qr_orders']);compressed=compressed.cuda().eval()
            metrics,_=evaluate(compressed,loader)
            rows.append({'group':group,'definition':definition,'widths':candidate,**cost(compressed),**metrics})
            del compressed
    save_json(path,{'baseline':baseline,'rows':rows,'scope':'Before fine-tuning, checkpoint-validation only; ceil aligned upward to 8; not maximum compressibility'})
    del model,loader;gc.collect();torch.cuda.empty_cache()


def final_test(jobs):
    score_path=RUN/'test_image_scores.csv'
    if not score_path.exists() or not (RUN/'test_rank_metadata.json').exists():
        base=build(pretrained=True).cuda().eval();collect(base,'test',save_features=False);del base
        gc.collect();torch.cuda.empty_cache()
    loader=evaluation_loader('test',batch_size=4)
    for job in jobs:
        folder=RUN/'evaluation'/job['task'];folder.mkdir(parents=True,exist_ok=True)
        if (folder/'test.json').exists():continue
        save_json(RUN/'status.json',{'state':'held_out_evaluation','task':job['task'],'at':time.time()})
        model=get_model(job)
        metrics,total,per_image,names=evaluate(model,loader,per_image=True)
        assert len(names)==2000
        np.savez_compressed(folder/'confusion.npz',total=total,per_image=per_image.astype(np.uint32))
        filename_path=RUN/'evaluation_filenames.json'
        if filename_path.exists():assert json.loads(filename_path.read_text())==names
        else:save_json(filename_path,names)
        save_json(folder/'test.json',{**job,**metrics,'evaluated_at':time.time()})
        print('EVALUATED',job['task'],'mIoU',round(metrics['miou'],3),flush=True)
        del model,per_image,total;gc.collect();torch.cuda.empty_cache()
    save_json(RUN/'test_complete.json',{'models':len(jobs),'images_per_model':2000,'at':time.time()})


def latency(jobs):
    folder=RUN/'latency';folder.mkdir(exist_ok=True)
    for repetition in range(CFG['latency_rounds']):
        for i in np.random.default_rng(8817+repetition).permutation(len(jobs)):
            job=jobs[int(i)];path=folder/f'{job["task"]}_r{repetition}.json'
            if path.exists():continue
            while external_gpu_processes():
                save_json(RUN/'status.json',{'state':'waiting_exclusive_latency','at':time.time()});time.sleep(15)
            model=get_model(job);measurements=[]
            for batch in CFG['latency_batches']:
                for mode in ('eager','cuda_graph'):
                    measurements.append(measure(model,batch,mode))
            save_json(path,{'task':job['task'],'round':repetition,'measurements':measurements,'at':time.time()})
            save_json(RUN/'status.json',{'state':'latency','task':job['task'],'round':repetition+1,'at':time.time()})
            print('LATENCY',repetition,job['task'],flush=True)
            del model;gc.collect();torch.cuda.empty_cache()


def main():
    setup();jobs=inventory();native_diagnostics();final_test(jobs);latency(jobs)
    from report import main as report
    report()
    save_json(RUN/'complete.json',{'trainings':63,'evaluated_models':64,'latency_series':64*CFG['latency_rounds'],
              'updates':63*CFG['steps'],'at':time.time()})


if __name__=='__main__':
    try:main()
    except Exception:
        save_json(RUN/'failures/finalize.json',{'at':time.time(),'traceback':traceback.format_exc()})
        raise
