"""One paired seed group: source model, fixed subsets, diagnostic and FT candidates."""
import argparse
import copy
import json
import math
import time
import traceback

from runtime import CFG, OLD, RUN, engine, save_json, setup
from tiny_data import TinyData
from tiny_model import allocate_aligned, base_spec, build, cost, feature_modules, install, linear_cost, prune
from subsets import prepare
from data import Data
from compress import allocate
from models import to_gpu
import numpy as np
import torch


def prefix_for(family,dataset,seed):
    return f'{family}_{dataset}_s{seed}'


def dense_path(family,dataset,seed):
    root=RUN if family=='resnet18' else OLD
    return root/'models'/f'{prefix_for(family,dataset,seed)}_dense'


def inventory():
    jobs=[]
    for family,dataset in CFG['groups']:
        for seed in CFG['seeds']:
            prefix=prefix_for(family,dataset,seed)
            if family=='resnet18':
                jobs.append({'task':prefix+'_dense','family':family,'dataset':dataset,
                             'model_seed':seed,'train_seed':seed,'phase':'base','method':'dense',
                             'budget':1.,'epochs':CFG['tiny_base_epochs']})
            for budget in CFG['budgets']:
                for method in CFG['methods']:
                    jobs.append({'task':f'{prefix}_b{round(100*budget)}_{method}','family':family,
                                 'dataset':dataset,'model_seed':seed,'train_seed':100000+seed,
                                 'phase':'main','method':method,'budget':budget,
                                 'epochs':CFG['finetune_epochs']})
    return jobs


def save_inventory():
    jobs=inventory()
    destination=RUN/'planned_runs.json'
    if destination.exists():
        assert json.loads(destination.read_text())==jobs
    else:
        save_json(destination,jobs)
    return jobs


def native_diagnostics(base,data,selection,prefix):
    path=RUN/'diagnostics'/prefix/'native_ceil.json'
    if path.exists():
        return
    full=np.array(base.spec['widths']);minimum=4 if base.spec['family']=='resnet20' else 8
    baseline=engine.evaluate(base,data)
    records=[];orders=selection['common_qr_orders']
    for group in CFG['selection_groups']:
        for definition in ('sigma','cov'):
            widths=np.clip(np.ceil([r['erank_'+definition] for r in selection['groups'][group]['layers']]),minimum,full).astype(int).tolist()
            model,masks=prune(base,widths,orders)
            engine.recalibrate(model,data)
            measured=cost(model)
            records.append({'group':group,'definition':definition,'method':'erank_ceil','widths':widths,
                            **measured,**engine.evaluate(model,data)})
            del model
            uniform_widths=allocate(base,full,measured['macs'])
            model,_=prune(base,uniform_widths,orders)
            engine.recalibrate(model,data)
            assert cost(model)['macs']==measured['macs']
            records.append({'group':group,'definition':definition,'method':'matched_uniform',
                            'widths':uniform_widths,**cost(model),**engine.evaluate(model,data)})
            del model
    save_json(path,{'baseline':baseline,'baseline_cost':cost(base),'records':records,
                   'scope':'Validation after BN recalibration, before fine-tuning; not a maximum compressibility estimate'})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('family',choices=['resnet20','mini_vit','resnet18'])
    parser.add_argument('dataset',choices=['cifar100','tinyimagenet'])
    parser.add_argument('seed',type=int)
    args=parser.parse_args()
    setup();install();jobs=save_inventory()
    family,dataset,seed=args.family,args.dataset,args.seed
    assert [family,dataset] in CFG['groups']
    torch.cuda.set_per_process_memory_fraction(.28 if family=='resnet18' else .15)
    prefix=prefix_for(family,dataset,seed)
    data=TinyData() if dataset=='tinyimagenet' else Data(dataset)
    if family=='resnet18':
        job=next(j for j in jobs if j['task']==prefix+'_dense')
        engine.seed_all(seed)
        base=engine.train(build(base_spec()),data,dense_path(family,dataset,seed),job,job['epochs'])
    else:
        base=engine.load_model(dense_path(family,dataset,seed)/'best.pt')
    assert data.test_images is None
    diagnostic_dir=RUN/'diagnostics'/prefix
    diagnostic_dir.mkdir(parents=True,exist_ok=True)
    save_json(diagnostic_dir/'dense_fit.json',{'train':engine.evaluate(base,data,'train'),
                                            'validation':engine.evaluate(base,data),'cost':cost(base)})
    print('SCORING',prefix,flush=True)
    selection=prepare(base,data,prefix)
    native_diagnostics(base,data,selection,prefix)
    full=np.array(base.spec['widths']);full_macs=cost(base)['macs']
    qr_orders=selection['common_qr_orders']
    l1_orders=[]
    for block in base.blocks:
        weight=block.mlp.fc1.weight if family=='mini_vit' else block.conv1.weight
        values=weight.detach().abs().flatten(1).sum(1).cpu().numpy()
        l1_orders.append(np.argsort(-values,kind='stable').tolist())
    group_jobs=[j for j in jobs if j['family']==family and j['dataset']==dataset and j['model_seed']==seed and j['phase']=='main']
    for job in group_jobs:
        folder=RUN/'models'/job['task']
        if (folder/'train_done.json').exists():
            continue
        method=job['method'];target=full_macs*job['budget']
        aligned=method.endswith('_aligned8')
        if aligned:
            profile=full if method=='uniform_aligned8' else [r['erank_sigma'] for r in selection['groups']['random']['layers']]
            widths=allocate_aligned(base,profile,target)
        elif method in ('uniform','l1'):
            widths=allocate(base,full,target)
        else:
            _,definition,group=method.split('_')
            profile=[r['erank_'+definition] for r in selection['groups'][group]['layers']]
            widths=allocate(base,profile,target)
        orders=l1_orders if method=='l1' else qr_orders
        model,masks=prune(base,widths,orders)
        expected_widths=allocate_aligned(base,full,target) if aligned else allocate(base,full,target)
        _,coefficients,fixed=linear_cost(base)
        expected=int(fixed+coefficients@expected_widths)
        assert cost(model)['macs']==expected
        before=engine.evaluate(model,data)
        engine.recalibrate(model,data)
        after=engine.evaluate(model,data)
        save_json(folder/'pruning.json',{'widths':widths,'masks':masks,'before':before,'after_bn':after,
                'source':str(dense_path(family,dataset,seed).relative_to(RUN.parent.parent)),
                'common_qr':method!='l1','rank_definition':method,'calibration_images':CFG['calibration_size']})
        engine.train(model,data,folder,job,job['epochs'],ft=True)
        assert data.test_images is None
        del model
        torch.cuda.empty_cache()
    save_json(RUN/'group_complete'/f'{prefix}.json',{'at':time.time(),'jobs':len(group_jobs),'test_opened':False})
    print('GROUP COMPLETE',prefix,flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception:
        import os
        save_json(RUN/'failures'/f'worker_{os.getpid()}.json',{'traceback':traceback.format_exc()})
        raise
