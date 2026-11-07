"""Readable percentage results from saved segmentation counts and timing series."""
import hashlib
import json
from pathlib import Path
import time

from common import CFG,RUN,WEIGHTS,save_json
from metrics import summarize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.stats import t

LABELS={'original':'Исходная','dense_ft':'Исходная + FT','uniform':'Uniform + QR',
        'random_width':'Random-width + QR','l1':'L1','taylor':'Taylor',
        **{f'erank_{definition}_{group}':f'ERank {symbol}, {group}'
           for definition,symbol in [('sigma','σ'),('cov','cov')] for group in ['random','low','high']}}


def fmt(value,digits=2,signed=False):
    if pd.isna(value):return '—'
    return format(value,('+' if signed else '')+f'.{digits}f').replace('.',',')


def calibration_report():
    if not (RUN/'scores.json').exists():return
    scores=json.loads((RUN/'scores.json').read_text());frame=pd.read_csv(RUN/'calibration_image_scores.csv')
    rows=[]
    for group,values in scores['groups'].items():
        indices=values['indices'];part=frame.iloc[indices]
        rows.append({'group':group,'images':len(part),'image_rank_pct':100*part.score_cov.mean(),
                     'pooled_sigma_pct':100*np.mean([l['erank_sigma']/l['width'] for l in values['layers']]),
                     'pooled_cov_pct':100*np.mean([l['erank_cov']/l['width'] for l in values['layers']]),
                     'mean_classes_present':part.classes_present.mean()})
    summary=pd.DataFrame(rows);summary.to_csv(RUN/'calibration_summary.csv',index=False)
    lines=['# Что изменяет выбор изображений низкого ранга','',
           '512 изображений в каждой группе; одинаковые квоты по доминирующему классу. '
           'Индивидуальный ранг нормирован на предел выборочного спектра, pooled-ранг — на ширину канала. '
           'Это разные величины. Таблица не содержит результатов сжатия по mIoU.','',
           '| Группа | Индивидуальный ERank-cov, % предела | Pooled ERank-σ, % ширины | Pooled ERank-cov, % ширины |',
           '| --- | ---: | ---: | ---: |']
    for row in rows:lines.append(f'| {row["group"]} | {fmt(row["image_rank_pct"])} | {fmt(row["pooled_sigma_pct"])} | {fmt(row["pooled_cov_pct"])} |')
    check=json.loads((RUN/'rank_sampling_stability.json').read_text())
    lines+=['',f'Контроль на 32 калибровочных изображениях: Spearman между выбранными 256 позициями и полным пространственным спектром — {check["spearman_with_full_spatial"]["256"]:.3f}. Это ограниченная проверка, не гарантия точности для всех изображений.',
            '', '![Распределение ранга](rank_histogram.png)','',
            '![Послойные pooled-ранги](pooled_ranks.png)','',
            'Low-отбор сам по себе не доказывает возможность большего сжатия без потери качества. Это проверяют основные парные эксперименты.']
    (RUN/'calibration.md').write_text('\n'.join(lines)+'\n')
    fig,ax=plt.subplots(figsize=(9,4.5))
    bins=np.linspace(100*frame.score_cov.min(),100*frame.score_cov.max(),30)
    ax.hist(100*frame.score_cov,bins=bins,weights=np.full(len(frame),100/len(frame)),color='.8',label='Весь calibration pool')
    for group,color in [('low','#2676a6'),('random','#7355a3'),('high','#d1822e')]:
        part=frame.iloc[scores['groups'][group]['indices']]
        ax.hist(100*part.score_cov,bins=bins,weights=np.full(len(part),100/len(part)),histtype='step',linewidth=2,color=color,label=group)
    ax.set_xlabel('Индивидуальный ERank-cov, % теоретического предела');ax.set_ylabel('Доля изображений в интервале, %')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(RUN/'rank_histogram.png',dpi=180);fig.savefig(RUN/'rank_histogram.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,definition in zip(axes,['sigma','cov']):
        for group,values in scores['groups'].items():
            ax.plot(range(1,9),[100*l['erank_'+definition]/l['width'] for l in values['layers']],marker='o',label=group)
        ax.set_title('Pooled ERank-'+('σ' if definition=='sigma' else 'cov'))
        ax.set_xlabel('Mix-FFN блок');ax.grid(alpha=.2);ax.yaxis.set_major_formatter(PercentFormatter(100))
    axes[0].set_ylabel('Ранг относительно ширины');axes[1].legend();fig.tight_layout()
    fig.savefig(RUN/'pooled_ranks.png',dpi=180);fig.savefig(RUN/'pooled_ranks.pdf');plt.close(fig)


def gather():
    jobs=[{'task':'original','method':'original','seed':0,'budget':1.}]+json.loads((RUN/'planned_runs.json').read_text())
    raw=[];original_cost=json.loads((RUN/'model_cost.json').read_text())
    for job in jobs:
        result=json.loads((RUN/'evaluation'/job['task']/'test.json').read_text())
        done=json.loads((RUN/'models'/job['task']/'train_done.json').read_text()) if job['task']!='original' else None
        measured=done['cost'] if done else original_cost
        raw.append({k:job[k] for k in ['task','method','seed','budget']}|{
            'miou':result['miou'],'pixel_accuracy':result['pixel_accuracy'],
            'params':measured['params'],'flops':measured['flops'],'best_step':done['best_step'] if done else 0,
            'training_seconds':done['training_seconds'] if done else 0.})
    frame=pd.DataFrame(raw);assert len(frame)==64
    timing=[]
    for path in (RUN/'latency').glob('*.json'):
        item=json.loads(path.read_text())
        for row in item['measurements']:
            timing.append({'task':item['task'],'round':item['round'],**{k:row[k] for k in ['mode','batch','median_ms','wall_mean_ms']}})
    timing=pd.DataFrame(timing)
    assert len(timing)==64*3*2*2
    assert timing.groupby(['task','mode','batch']).size().eq(3).all()
    timing.to_csv(RUN/'latency_rounds.csv',index=False)
    median=timing.groupby(['task','mode','batch']).median(numeric_only=True).reset_index()
    for mode in ['eager','cuda_graph']:
        for batch in CFG['latency_batches']:
            values=median[(median['mode']==mode)&(median.batch==batch)].set_index('task').median_ms
            frame[f'latency_{mode}_b{batch}_ms']=frame.task.map(values)
            frame[f'speed_{mode}_b{batch}_pct']=100*(values['original']/frame.task.map(values)-1)
            frame[f'time_{mode}_b{batch}_pct']=100*frame.task.map(values)/values['original']
    original=frame[frame.method=='original'].iloc[0]
    for source,target in [('miou','quality_pct'),('pixel_accuracy','pixel_quality_pct'),('params','params_pct'),('flops','flops_pct')]:
        frame[target]=100*frame[source]/original[source]
    frame.to_csv(RUN/'results.csv',index=False)
    metrics=['miou','pixel_accuracy','quality_pct','params_pct','flops_pct','best_step']+[f'speed_{m}_b{b}_pct' for m in ['eager','cuda_graph'] for b in CFG['latency_batches']]
    summary=frame.groupby(['method','budget']).agg({k:['mean','std'] for k in metrics})
    summary.columns=['_'.join(k) for k in summary.columns];summary=summary.reset_index()
    summary.to_csv(RUN/'summary.csv',index=False)
    return frame,summary


def comparisons(frame):
    rows=[];paired=[];family_size=8;critical=t.ppf(1-.05/(2*family_size),2)
    for budget in CFG['budgets']:
        for definition in ['sigma','cov']:
            candidate=f'erank_{definition}_low'
            for reference in [f'erank_{definition}_random','uniform']:
                left=frame[(frame.method==candidate)&(frame.budget==budget)].set_index('seed')
                right=frame[(frame.method==reference)&(frame.budget==budget)].set_index('seed')
                assert set(left.index)==set(right.index)==set(CFG['seeds'])
                changes=100*(left.miou/right.miou-1);sd=float(changes.std(ddof=1))
                for seed,value in changes.items():paired.append({'budget':budget,'candidate':candidate,'reference':reference,'seed':seed,'delta_miou_relative_pct':float(value)})
                margin=critical*sd/np.sqrt(3) if sd>0 else np.nan
                rows.append({'budget':budget,'candidate':candidate,'reference':reference,'mean_change_pct':float(changes.mean()),
                             'sd_change_pct':sd,'ci_low_pct':float(changes.mean())-margin,'ci_high_pct':float(changes.mean())+margin,
                             'n':3,'family_size':family_size,'scope':'conditional on one source checkpoint and one calibration split'})
    pd.DataFrame(paired).to_csv(RUN/'paired_comparisons_per_seed.csv',index=False)
    result=pd.DataFrame(rows);result.to_csv(RUN/'paired_comparisons.csv',index=False)
    return result


def class_and_strata(frame):
    names=json.loads((RUN/'evaluation_filenames.json').read_text())
    scores=pd.read_csv(RUN/'test_image_scores.csv').set_index('name').loc[names]
    thresholds=json.loads((RUN/'scores.json').read_text())['thresholds'];strata=[]
    for row in scores.itertuples():
        bounds=thresholds.get(str(row.dominant_class))
        strata.append('middle' if bounds is None else 'low' if row.score_cov<=bounds['low'] else 'high' if row.score_cov>=bounds['high'] else 'middle')
    strata=np.array(strata);records=[];classes=[]
    labels=json.loads((WEIGHTS/'config.json').read_text())['id2label']
    reference=np.load(RUN/'evaluation/original/confusion.npz')['total'];truth=reference.sum(1)
    for row in frame.itertuples():
        payload=np.load(RUN/'evaluation'/row.task/'confusion.npz');total=payload['total'];counts=payload['per_image']
        assert np.array_equal(total.sum(1),truth)
        assert np.array_equal(counts.sum(0,dtype=np.int64),total)
        metric=summarize(total)
        assert np.isclose(metric['miou'],row.miou,rtol=0,atol=1e-8),'Stored mIoU disagrees with confusion counts'
        for index,value in enumerate(metric['class_iou']):
            classes.append({'task':row.task,'method':row.method,'budget':row.budget,'seed':row.seed,
                            'class_id':index,'class':labels[str(index)],'iou':value,'ground_truth_pixels':int(truth[index])})
        for stratum in ['low','middle','high']:
            mask=strata==stratum
            if not mask.any():continue
            matrix=counts[mask].sum(0,dtype=np.int64);present=matrix.sum(1)>0
            intersection=np.diag(matrix);union=matrix.sum(1)+matrix.sum(0)-intersection
            miou=100*np.mean(intersection[present]/union[present])
            records.append({'task':row.task,'method':row.method,'budget':row.budget,'seed':row.seed,'stratum':stratum,
                            'images':int(mask.sum()),'classes_present_in_gt':int(present.sum()),'miou':float(miou)})
    pd.DataFrame(classes).to_csv(RUN/'class_iou.csv',index=False)
    pd.DataFrame(records).to_csv(RUN/'stratified_miou.csv',index=False)
    save_json(RUN/'strata_definition.json',{'assignment':'Source checkpoint scores with calibration thresholds by dominant GT class',
              'scope':'Exploratory diagnosis, not a deployable routing policy; classes averaged only when present in the stratum GT',
              'counts':{name:int(np.count_nonzero(strata==name)) for name in ['low','middle','high']}})


def tables(summary,budget):
    part=summary[summary.budget==budget]
    if budget!=1.:part=part.set_index('method').loc[CFG['methods']].reset_index()
    quality=['| Метод | mIoU, % | Сохранено качества, % | Параметры, % | FLOPs, % |',
             '| --- | ---: | ---: | ---: | ---: |']
    speed=['| Метод | Eager, batch 1 | Graph, batch 1 | Eager, batch 8 | Graph, batch 8 |',
           '| --- | ---: | ---: | ---: | ---: |']
    for r in part.itertuples():
        quality.append(f'| {LABELS[r.method]} | {fmt(r.miou_mean)} | {fmt(r.quality_pct_mean)} ± {fmt(r.quality_pct_std)} | {fmt(r.params_pct_mean,1)} | {fmt(r.flops_pct_mean,2)} |')
        speed.append('| '+' | '.join([LABELS[r.method],*[fmt(getattr(r,f'speed_{m}_b{b}_pct_mean'),1,True)+'%' for b in CFG['latency_batches'] for m in ['eager','cuda_graph']]])+' |')
    return '\n'.join(quality),'\n'.join(speed)


def plots(summary):
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for method in CFG['methods']:
        part=summary[summary.method==method].sort_values('budget')
        axes[0].errorbar(part.flops_pct_mean,part.quality_pct_mean,yerr=part.quality_pct_std,marker='o',capsize=3,label=LABELS[method])
        axes[1].plot(part.speed_cuda_graph_b8_pct_mean,part.quality_pct_mean,marker='o',label=LABELS[method])
    axes[0].set_xlabel('Осталось FLOPs от исходной модели');axes[1].set_xlabel('Изменение скорости, CUDA Graphs / batch 8')
    for ax in axes:
        ax.set_ylabel('mIoU относительно исходной модели');ax.yaxis.set_major_formatter(PercentFormatter(100));ax.xaxis.set_major_formatter(PercentFormatter(100))
        ax.axhline(100,color='.5',linestyle=':');ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=5,fontsize=8)
    fig.tight_layout(rect=[0,.13,1,1]);fig.savefig(RUN/'quality_resources.png',dpi=180);fig.savefig(RUN/'quality_resources.pdf');plt.close(fig)


def learning_curves():
    records=[]
    for job in json.loads((RUN/'planned_runs.json').read_text()):
        folder=RUN/'models'/job['task']
        initial=json.loads((folder/'initial_validation.json').read_text())
        rows=[{'step':0,'val_miou':initial['miou'],'train_loss':np.nan}]
        rows += [json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
        for row in rows:
            records.append({**{k:job[k] for k in ['task','method','seed','budget']},
                            **{k:row[k] for k in ['step','val_miou','train_loss']}})
    frame=pd.DataFrame(records);frame.to_csv(RUN/'learning_curves.csv',index=False)
    average=frame.groupby(['method','budget','step']).val_miou.agg(['mean','std']).reset_index()
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,budget in zip(axes,CFG['budgets']):
        for method in ['dense_ft',*CFG['methods']]:
            values=average[(average.method==method)&(average.budget==(1. if method=='dense_ft' else budget))]
            ax.plot(values.step,values['mean'],marker='.',label=LABELS[method])
            ax.fill_between(values.step,values['mean']-values['std'],values['mean']+values['std'],alpha=.1)
        ax.set_title(f'Сокращение FLOPs: {round(100*(1-budget))}%')
        ax.set_xlabel('Обновления fine-tuning');ax.grid(alpha=.2)
    axes[0].set_ylabel('Checkpoint-validation mIoU, %; среднее ± SD')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,fontsize=8)
    fig.tight_layout(rect=[0,.16,1,1]);fig.savefig(RUN/'learning_curves.png',dpi=180)
    fig.savefig(RUN/'learning_curves.pdf');plt.close(fig)


def write_report(frame,summary,paired):
    original=frame[frame.method=='original'].iloc[0]
    late=int((frame[frame.method!='original'].best_step==CFG['steps']).sum())
    coverage=CFG['steps']*CFG['batch_size']/17650
    dense_quality,dense_speed=tables(summary,1.)
    lines=['# ERank на ADE20K: результаты сегментации','',
           f'SegFormer-B1, 512×512, 150 классов; один опубликованный исходный checkpoint, три seed fine-tuning. '
           f'Исходный mIoU на официальных 2000 validation-изображениях: **{fmt(original.miou)}%**.', '',
           f'Завершены 63 fine-tuning по {CFG["steps"]} обновлений. Это примерно **{coverage:.2f} эпохи** на выделенном train для каждого кандидата. '
           f'У {late} из 63 кандидатов лучшим оказался последний проверенный шаг: фиксированный бюджет не является доказательством сходимости.','',
           '## Единицы и условия','',
           '- Качество = 100 × mIoU / mIoU исходной модели. Разница в относительных процентах не равна разнице mIoU в процентных пунктах.',
           '- Параметры и FLOPs — доля от исходной модели. Все методы внутри бюджета имеют одинаковые реальные MACs и ширины, кратные 8.',
           '- Скорость = 100 × (время исходной / время кандидата − 1), плюс означает ускорение. Dense и кандидат сравниваются в одном режиме и batch.',
           '- Время включает GPU forward, bilinear resize до 512×512 и argmax. CPU preprocessing, передача входа и создание CUDA Graph исключены. Графы имеют фиксированный размер входа.',
           '- В строках качество ± SD трёх seed fine-tuning. Это условная вариативность при одном исходном обучении и одной калибровке.', '',
           '## Несжатый контроль','',dense_quality,'',dense_speed,'']
    for budget in CFG['budgets']:
        quality,speed=tables(summary,budget)
        lines += [f'## Сокращение общих FLOPs примерно на {round(100*(1-budget))}%','',quality,'',speed,'']
    lines += ['## Low-калибровка: парные эффекты','',
              'Среднее и SD относительного изменения mIoU. Одновременные 95% t-интервалы с поправкой Бонферрони на 8 контрастов; '
              'при нулевом SD интервал не оценивается. Малое n и общий pretrained checkpoint ограничивают выводы.','',
              '| Сокращение FLOPs | ERank-low | Сравнение | ΔmIoU, % | Интервал, % |',
              '| --- | --- | --- | ---: | --- |']
    for row in paired.itertuples():
        lines.append(f'| {round(100*(1-row.budget))}% | {LABELS[row.candidate]} | {LABELS[row.reference]} | {fmt(row.mean_change_pct,signed=True)} ± {fmt(row.sd_change_pct)} | [{fmt(row.ci_low_pct,signed=True)}; {fmt(row.ci_high_pct,signed=True)}] |')
    lines += ['', '## Ранг и качество','',
              '[Гистограммы и различие индивидуального и pooled-ранга](calibration.md). '
              '[Диагностика прямого ceil-правила до fine-tuning](native_ceil_validation.json). '
              'Снижение ранга само по себе не доказывает меньшую необходимую ширину.','',
              '![Качество, FLOPs и реальная скорость](quality_resources.png)','',
              '![Восстановление на checkpoint-validation](learning_curves.png)','',
              '## Ограничения','',
              'Это пилот с ограниченным восстановлением весов, а не доказательство бесполезности или универсального преимущества прунинга. '
              'Опубликованный исходный checkpoint уже обучался на ADE train, включая выделенные сейчас calibration и checkpoint-validation. '
              'Авторы исходной модели могли использовать официальный validation при её разработке. '
              'Наши новые кандидаты и гиперпараметры не выбирались по итоговой оценке.', '',
              'Все веса исходной модели общие между seed; они не являются независимыми pretrained-моделями. '
              'Квантизация, дистилляция, другие модели сегментации и обучение с нуля в этот пул не входят. '
              'Результаты скорости относятся к A100 и указанным режимам исполнения.','',
              '## Данные','',
              '- [Все результаты и относительные показатели](results.csv).',
              '- [Средние и SD](summary.csv).',
              '- [Парные сравнения](paired_comparisons.csv) и [значения по seed](paired_comparisons_per_seed.csv).',
              '- [IoU каждого класса](class_iou.csv).',
              '- [mIoU на фиксированных low/middle/high-стратах](stratified_miou.csv): диагностический анализ с классами, присутствующими в GT каждой страты.',
              '- [Исходные серии времени](latency_rounds.csv).','']
    (RUN/'report.md').write_text('\n'.join(lines))
    short=['# Краткие результаты ADE20K','',f'Исходный mIoU: {fmt(original.miou)}%. 63 завершённых fine-tuning, три seed одного pretrained checkpoint.','']
    for budget in CFG['budgets']:
        part=summary[summary.budget==budget].sort_values('miou_mean',ascending=False)
        top=part.iloc[0]
        short.append(f'- При −{round(100*(1-budget))}% FLOPs наибольший средний mIoU: **{LABELS[top.method]} — {fmt(top.miou_mean)}%**. Это описательный максимум, не статистически доказанный победитель.')
    short+=['', 'Качество ниже — доля исходного mIoU (исходная модель = 100%), среднее трёх seed.', '',
            '| Сокращение FLOPs | Uniform | ERank σ, low | ERank cov, low | Taylor |',
            '| --- | ---: | ---: | ---: | ---: |']
    for budget in CFG['budgets']:
        part=summary[summary.budget==budget].set_index('method')
        values=[fmt(part.loc[m,'quality_pct_mean'])+'%' for m in ['uniform','erank_sigma_low','erank_cov_low','taylor']]
        short.append('| '+str(round(100*(1-budget)))+'% | '+' | '.join(values)+' |')
    low_random=paired[paired.reference.str.startswith('erank_')]
    positive=int((low_random.ci_low_pct>0).sum())
    short+=['', f'У low-отбора **{positive} из {len(low_random)}** заранее заданных сравнений с random имеют '
                 'полностью положительный интервал после поправки на множественные сравнения. '
                 'Отсутствие подтверждённого преимущества не доказывает равенство методов.']
    speed=summary[(summary.method=='erank_sigma_low')&(summary.budget==.8)].iloc[0]
    short+=['', f'При −20% FLOPs у ERank σ, low измеренное ускорение: '
                 f'**{fmt(speed.speed_cuda_graph_b1_pct_mean,1,True)}%** в CUDA Graphs / batch 1, '
                 f'**{fmt(speed.speed_cuda_graph_b8_pct_mean,1,True)}%** в CUDA Graphs / batch 8; '
                 f'в eager / batch 1 — **{fmt(speed.speed_eager_b1_pct_mean,1,True)}%**. '
                 'Это ускорение измеренного GPU-пути, включающего resize и argmax, без CPU preprocessing и копирования входа.']
    eligible=summary[(summary.budget<1)&(summary.quality_pct_mean>=99)]
    if len(eligible):
        best=eligible.sort_values('speed_cuda_graph_b8_pct_mean',ascending=False).iloc[0]
        short+=['',f'Среди вариантов со средним сохранением не менее 99% исходного mIoU самый быстрый по измерениям Graph/batch 8: '
                  f'{LABELS[best.method]}, −{round(100*(1-best.budget))}% FLOPs; изменение скорости {fmt(best.speed_cuda_graph_b8_pct_mean,1,True)}%. '
                  'Порог 99% — описательный ориентир, не доказательство эквивалентности качества.']
    else:short+=['','Ни один сжатый вариант не сохранил в среднем 99% исходного mIoU при данном бюджете восстановления.']
    short+=['',f'Важно: {coverage:.2f} эпохи fine-tuning на кандидата; {late}/63 лучших checkpoint выбраны на последней проверке. '
                 'Если восстановление ещё продолжается, отрицательный результат нельзя переносить на более длинное обучение.',
            '', '[Полные таблицы, скорость и ограничения](report.md) · [Что изменил low-отбор](calibration.md).','']
    (RUN/'findings.md').write_text('\n'.join(short))


def main():
    calibration_report();frame,summary=gather();paired=comparisons(frame)
    class_and_strata(frame);plots(summary);learning_curves();write_report(frame,summary,paired)
    save_json(RUN/'report_metadata.json',{'at':time.time(),'models':64,'trainings':63,
        'normalization':'Within candidate/source ratio, same execution mode and batch; three fine-tune seeds share one source checkpoint',
        'source_sha256':hashlib.sha256((Path(__file__)).read_bytes()).hexdigest()})
    print('Segmentation report written',flush=True)


if __name__=='__main__':main()
