"""Freeze all candidate architectures without looking at final validation labels."""
import hashlib
import json
import time

from common import CFG,RUN,save_json
from network import allocate,target_for_budget
import numpy as np


def main():
    path=RUN/'planned_runs.json'
    if path.exists():return json.loads(path.read_text())
    scores=json.loads((RUN/'scores.json').read_text())
    measured=json.loads((RUN/'model_cost.json').read_text())
    result=[]
    for seed in CFG['seeds']:
        common={'seed':seed,'steps':CFG['steps'],'batch_size':CFG['batch_size']}
        result.append({**common,'task':f's{seed}_dense_ft','method':'dense_ft','budget':1.,
                       'widths':measured['widths'],'orders':[list(range(w)) for w in measured['widths']],
                       'target_macs':measured['macs']})
        for budget in CFG['budgets']:
            target=target_for_budget(measured,budget)
            uniform,uniform_info=allocate(measured,measured['widths'],target)
            for method in CFG['methods']:
                orders=scores['qr_orders']
                if method in ('uniform','l1','taylor'):
                    candidate,info=uniform,uniform_info
                    if method!='uniform':orders=[np.argsort(-np.asarray(row),kind='stable').tolist() for row in scores[method]]
                else:
                    if method=='random_width':
                        profile=np.array(measured['widths'])*np.random.default_rng(9901+seed).uniform(.3,1.7,len(measured['widths']))
                    else:
                        _,definition,group=method.split('_')
                        profile=[layer['erank_'+definition] for layer in scores['groups'][group]['layers']]
                    candidate,info=allocate(measured,profile,target)
                result.append({**common,'task':f's{seed}_b{round(100*budget)}_{method}',
                               'method':method,'budget':budget,'widths':candidate,'orders':orders,
                               'target_macs':target,'allocation':info,
                               'architecture_l1_from_uniform':int(np.abs(np.array(candidate)-uniform).sum())})
    assert len(result)==63
    save_json(path,result)
    save_json(RUN/'plan_metadata.json',{'at':time.time(),'trainings':len(result),'steps':sum(r['steps'] for r in result),
              'scope':'Architectures fixed from calibration before training and final evaluation',
              'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return result


if __name__=='__main__':
    jobs=main();print('Frozen',len(jobs),'training runs',flush=True)
