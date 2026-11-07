"""Check reporting arithmetic using disposable artificial data, never experiment outputs."""
import json
from pathlib import Path
import tempfile
import time

from common import CFG,RUN,save_json
from metrics import summarize
import numpy as np
import report
from scipy.stats import t


def main():
    checks=[]
    with tempfile.TemporaryDirectory(prefix='segmentation-report-check-') as directory:
        root=Path(directory);report.RUN=root;report.WEIGHTS=root/'weights'
        save_json(root/'weights/config.json',{'id2label':{str(i):f'class_{i}' for i in range(150)}})
        jobs=[]
        for seed in CFG['seeds']:
            jobs.append({'task':f's{seed}_dense_ft','method':'dense_ft','seed':seed,'budget':1.})
            for budget in CFG['budgets']:
                for method in CFG['methods']:
                    jobs.append({'task':f's{seed}_{method}_{budget}','method':method,'seed':seed,'budget':budget})
        save_json(root/'planned_runs.json',jobs)
        save_json(root/'model_cost.json',{'params':2000,'flops':1000})
        save_json(root/'evaluation_filenames.json',['a','b','c'])
        save_json(root/'scores.json',{'thresholds':{'0':{'low':.2,'high':.8}}})
        (root/'test_image_scores.csv').write_text('name,dominant_class,score_cov\na,0,0.1\nb,0,0.5\nc,0,0.9\n')
        all_jobs=[{'task':'original','method':'original','seed':0,'budget':1.}]+jobs
        for job in all_jobs:
            original=job['method']=='original'
            seed_offset=0 if original else CFG['seeds'].index(job['seed'])
            tp=75 if original else 80+CFG['methods'].index(job['method'])//2+seed_offset if job['method']!='dense_ft' else 76+seed_offset
            matrix=np.zeros((150,150),dtype=np.int64);matrix[:2,:2]=[[tp,100-tp],[100-tp,tp]]
            per_image=np.stack([matrix]*3);total=per_image.sum(0);metric=summarize(total)
            folder=root/'evaluation'/job['task'];folder.mkdir(parents=True)
            np.savez_compressed(folder/'confusion.npz',total=total,per_image=per_image)
            save_json(folder/'test.json',metric)
            if not original:
                training=root/'models'/job['task'];training.mkdir(parents=True)
                save_json(training/'train_done.json',{'cost':{'params':2000*job['budget'],'flops':1000*job['budget']},
                          'best_step':1500,'training_seconds':100})
                save_json(training/'initial_validation.json',metric)
                (training/'training.jsonl').write_text(''.join(json.dumps({'step':s,'val_miou':metric['miou'],'train_loss':1.})+'\n' for s in (500,1000,1500)))
            for repetition in range(3):
                rows=[]
                for mode in ('eager','cuda_graph'):
                    for batch in CFG['latency_batches']:
                        value=(10. if original else 8.)*batch
                        rows.append({'mode':mode,'batch':batch,'median_ms':value,'wall_mean_ms':value})
                save_json(root/'latency'/f'{job["task"]}_r{repetition}.json',{'task':job['task'],'round':repetition,'measurements':rows})
        frame,summary=report.gather()
        uniform=frame[(frame.method=='uniform')&(frame.seed==42)&(frame.budget==.9)].iloc[0]
        assert np.isclose(uniform.miou,100*80/120)
        assert np.isclose(uniform.quality_pct,100*(80/120)/(75/125))
        assert uniform.params_pct==uniform.flops_pct==90.
        assert uniform.time_eager_b1_pct==80. and uniform.speed_eager_b1_pct==25.
        checks.append('Hand-computed mIoU ratio, parameter/FLOP fractions, and 20% less time = 25% more speed')
        paired=report.comparisons(frame);assert len(paired)==8
        row=paired.iloc[0]
        left=frame[(frame.method==row.candidate)&(frame.budget==row.budget)].set_index('seed').miou
        right=frame[(frame.method==row.reference)&(frame.budget==row.budget)].set_index('seed').miou
        differences=np.array([100*(left[s]/right[s]-1) for s in CFG['seeds']])
        half=t.ppf(1-.05/16,2)*differences.std(ddof=1)/np.sqrt(3)
        assert np.isclose(row.ci_high_pct,differences.mean()+half)
        missing=frame[~((frame.method==row.candidate)&(frame.seed==42)&(frame.budget==row.budget))]
        try:report.comparisons(missing)
        except AssertionError:pass
        else:raise AssertionError('Missing seed should not silently change the comparison')
        constant=frame.copy();constant.loc[constant.budget<1,'miou']=60.
        assert report.comparisons(constant).ci_low_pct.isna().all()
        paired=report.comparisons(frame)
        checks.append('Eight seed-paired Bonferroni intervals, rejection of missing seed, undefined zero-SD interval')
        report.class_and_strata(frame)
        counts=json.loads((root/'strata_definition.json').read_text())['counts']
        assert counts=={'low':1,'middle':1,'high':1}
        report.plots(summary);report.learning_curves();report.write_report(frame,summary,paired)
        assert (root/'report.md').exists() and (root/'findings.md').exists()
        assert all((root/name).exists() for name in ['quality_resources.png','learning_curves.png','learning_curves.csv'])
        checks.append('Fixed-stratum counts, consistency of stored mIoU with confusion matrices, complete report and figures')
    save_json(RUN/'report_verification.json',{'passed':checks,'at':time.time(),'scope':'Artificial temporary fixtures; not scientific results'})
    print('\n'.join(checks))


if __name__=='__main__':main()
