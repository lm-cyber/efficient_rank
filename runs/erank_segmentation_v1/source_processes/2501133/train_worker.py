"""Paired fine-tuning from one fixed pretrained source; never reads final test images."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import traceback

from common import CFG,RUN,amp,save_checkpoint,save_json,seed_all,setup
from dataset import evaluation_loader,training_loader
from metrics import evaluate,loss_function
from network import build,cost,load,prune
import torch


def train(job,validation):
    folder=RUN/'models'/job['task'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'train_done.json').exists():return
    seed_all(job['seed'])
    source=build(pretrained=True)
    model,masks=prune(source,job['widths'],job['orders'])
    del source
    model=model.cuda()
    actual=cost(model);assert actual['macs']==job['target_macs']
    save_json(folder/'spec.json',{**job,**actual,'masks':masks,'source_snapshot':f'source_processes/{os.getpid()}'})
    decay=[];no_decay=[]
    for name,p in model.named_parameters():(no_decay if p.ndim<=1 or name.endswith('.bias') else decay).append(p)
    optimizer=torch.optim.AdamW([{'params':decay,'weight_decay':CFG['weight_decay']},
                                {'params':no_decay,'weight_decay':0.}],lr=CFG['learning_rate'],fused=True)
    resume=folder/'last.pt';start=0;elapsed=0.
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=True)
        model.load_state_dict(state['state_dict'],strict=True);optimizer.load_state_dict(state['optimizer'])
        start=state['step'];best=state['best_miou'];best_step=state['best_step'];elapsed=state['training_seconds']
        log=folder/'training.jsonl'
        if log.exists():
            rows=[json.loads(line) for line in log.read_text().splitlines()]
            log.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['step']<=start))
        del state
    else:
        initial,_=evaluate(model,validation)
        best=initial['miou'];best_step=0
        save_json(folder/'initial_validation.json',initial)
        save_checkpoint(folder/'best.pt',model,step=0,validation=initial)
    loader=training_loader(job['seed'],start=start)
    clock=time.monotonic();window_loss=0.;window_steps=0
    for step,(x,y,_) in enumerate(loader,start=start+1):
        seed_all(job['seed']*100000+step)
        model.train()
        if step<=CFG['warmup_steps']:lr=CFG['learning_rate']*step/CFG['warmup_steps']
        else:lr=CFG['learning_rate']*.5*(1+math.cos(math.pi*(step-CFG['warmup_steps'])/(CFG['steps']-CFG['warmup_steps'])))
        for group in optimizer.param_groups:group['lr']=lr
        x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp():output=model(x).logits
        loss=loss_function(output,y)
        assert torch.isfinite(loss),f'Nonfinite loss: {job["task"]}, step {step}'
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        window_loss+=float(loss.detach());window_steps+=1
        del output,loss,x,y
        if step%50==0:
            seconds=elapsed+time.monotonic()-clock
            print(job['task'],f'{step}/{CFG["steps"]}',f'loss={window_loss/window_steps:.4f}',f'seconds={seconds:.1f}',flush=True)
            save_json(RUN/'status'/f's{job["seed"]}.json',{'task':job['task'],'step':step,'steps':CFG['steps'],
                      'loss':window_loss/window_steps,'seconds':seconds,'at':time.time()})
        if step%CFG['validation_interval']==0 or step==CFG['steps']:
            metrics,_=evaluate(model,validation)
            if metrics['miou']>best:
                best=metrics['miou'];best_step=step
                save_checkpoint(folder/'best.pt',model,step=step,validation=metrics)
            seconds=elapsed+time.monotonic()-clock
            row={'step':step,'train_loss':window_loss/window_steps,'val_miou':metrics['miou'],
                 'val_pixel_accuracy':metrics['pixel_accuracy'],'best_miou':best,'best_step':best_step,
                 'learning_rate':lr,'seconds':seconds}
            with (folder/'training.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            save_checkpoint(resume,model,optimizer=optimizer.state_dict(),step=step,
                            best_miou=best,best_step=best_step,training_seconds=seconds)
            window_loss=0.;window_steps=0
    save_json(folder/'train_done.json',{'task':job['task'],'seed':job['seed'],'method':job['method'],
              'budget':job['budget'],'steps':CFG['steps'],'best_miou':best,'best_step':best_step,
              'training_seconds':elapsed+time.monotonic()-clock,'cost':actual,'at':time.time()})
    del model,optimizer,loader
    torch.cuda.empty_cache()
    print('COMPLETE',job['task'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('seed',type=int);args=parser.parse_args()
    setup();torch.cuda.set_per_process_memory_fraction(.31)
    inventory=json.loads((RUN/'planned_runs.json').read_text())
    jobs=[r for r in inventory if r['seed']==args.seed]
    assert len(jobs)==21
    validation=evaluation_loader('validation',batch_size=4)
    for job in jobs:train(job,validation)
    save_json(RUN/'worker_complete'/f's{args.seed}.json',{'at':time.time(),'jobs':len(jobs)})


if __name__=='__main__':
    try:main()
    except Exception:
        save_json(RUN/'failures'/f'worker_{os.getpid()}.json',{'at':time.time(),'traceback':traceback.format_exc()})
        raise
