import argparse
import traceback

from data import CFG, RUN, Data
from engine import save_json, seed_all, setup, train
from manifest import jobs
from models import base_spec, build


def run(family, dataset):
    setup()
    data = Data(dataset)
    selected = [j for j in jobs() if j['family'] == family and j['dataset'] == dataset and j['phase'] in ('base', 'sparse_base')]
    for job in selected:
        sparse = job['phase'] == 'sparse_base'
        seed_all(job['model_seed'])
        model = build(base_spec(family, data.classes, CFG['vit'], gated=sparse))
        train(model, data, RUN / 'models' / job['task'], {**job, 'train_seed': job['model_seed']},
              job['epochs'], sparse=sparse, dynamics=True)
    save_json(RUN / 'status' / f'{family}_{dataset}.json', {'state': 'bases_complete', 'models': len(selected)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('family')
    parser.add_argument('dataset')
    args = parser.parse_args()
    try:
        run(args.family, args.dataset)
    except Exception:
        save_json(RUN / 'failures' / f'{args.family}_{args.dataset}.json', {'traceback': traceback.format_exc()})
        raise
