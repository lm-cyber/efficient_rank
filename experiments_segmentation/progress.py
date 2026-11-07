"""Read actual run markers; do not infer completed evaluations from trained weights."""
import json
from statistics import median
import time

from common import CFG,RUN


def main():
    jobs=json.loads((RUN/'planned_runs.json').read_text()) if (RUN/'planned_runs.json').exists() else []
    done={path.parent.name:json.loads(path.read_text()) for path in (RUN/'models').glob('*/train_done.json')}
    state=json.loads((RUN/'status.json').read_text()) if (RUN/'status.json').exists() else {'state':'not_started'}
    updates=sum(item['steps'] for item in done.values());active=[];remaining=[]
    durations=[item['training_seconds'] for item in done.values()]
    for seed in CFG['seeds']:
        path=RUN/'status'/f's{seed}.json'
        status=json.loads(path.read_text()) if path.exists() else None
        partial=status['step'] if status and status['task'] not in done else 0
        updates+=partial
        if status and status['task'] not in done:
            active.append(f'- Seed {seed}: `{status["task"]}`, {100*partial/CFG["steps"]:.1f}% обновлений.')
        pending=sum(job['seed']==seed and job['task'] not in done for job in jobs)
        if durations:remaining.append(max(0,pending-partial/CFG['steps'])*median(durations))
    planned=len(jobs) or 63
    evaluations=len(list((RUN/'evaluation').glob('*/test.json')))
    timing=len(list((RUN/'latency').glob('*.json')))
    lines=['# Прогресс ADE20K', '', f'Обновлено: {time.strftime("%Y-%m-%d %H:%M:%S UTC",time.gmtime())}.', '',
           f'Этап: **{state["state"]}**.', '',
           f'- Завершённые fine-tuning: **{len(done)}/{planned} ({100*len(done)/planned:.1f}%)**.',
           f'- Выполнено обновлений: **{100*updates/(planned*CFG["steps"]):.1f}%**.',
           f'- Итоговая оценка: **{evaluations}/64 ({100*evaluations/64:.1f}%)** моделей.',
           f'- Серии замеров скорости: **{timing}/192 ({100*timing/192:.1f}%)**.', *active]
    if state['state']=='training' and remaining:
        minutes=max(remaining)/60
        lines+=['',f'Ориентир оставшегося обучения: {0.75*minutes:.0f}–{1.25*minutes:.0f} минут, '
                 'по длительности уже завершённых кандидатов. Это грубая оценка, не доверительный интервал. '
                 'Итоговая оценка и замеры скорости потребуют дополнительного времени.']
    failures=list((RUN/'failures').glob('*.json'))
    if failures:lines+=['',f'Сохранены сообщения об ошибках: {len(failures)}; проверьте `failures/` и время записей.']
    if (RUN/'complete.json').exists():lines+=['','Весь пул завершён: [выводы](findings.md), [полный отчёт](report.md).']
    else:lines+=['','Итоговые выводы о качестве сжатия ещё не сформированы.']
    result='\n'.join(lines)+'\n';(RUN/'progress.md').write_text(result);print(result)


if __name__=='__main__':main()
