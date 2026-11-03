"""Freeze HPO, dependent reruns, extra controls and modern model jobs."""
import copy
from collections import Counter
import itertools
import numpy as np

from common import ROOT, PLAN, read, write_json
from prepare import CORE, SEEDS, pair_code
from core import load_case
from supplement_prepare import copy_common, degree_preserving_shuffle, finish_dataset, SOURCES
import run as base


def extra_shuffle():
    for seed in [123, 456]:
        c = load_case(ROOT / 'benchmark_data', 't4_a', seed)
        a, x = c['arrays'], c['triples']
        held = np.r_[a['valid'], a['test'], a['negatives'], a['conflict']]
        forbidden = set(pair_code(x[held], len(c['seen']), True))
        ids = [c['relations'].index(r) for r in c['meta']['scoring_relations']]
        shuffled, stats = degree_preserving_shuffle(c['train'], ids, forbidden, len(c['seen']), seed)
        out = ROOT / 'revision_data/shuffle' / f't4_a_seed{seed}'
        copy_common(out)
        changed = np.array(x); changed[a['train']] = shuffled
        np.save(out / 'triples.npy', changed, allow_pickle=False)
        target = out / 'splits/t4_a'; target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target / f'seed_{seed}.npz', **a)
        meta = copy.deepcopy(c['meta'])
        meta['perturbation'] = dict(per_relation=stats, filtering='original unshuffled train truths',
                                    protected='held-out pairs, endpoint degrees, auxiliary edges, masks')
        write_json(target / f'seed_{seed}.json', meta)
        finish_dataset(out, [meta], dict(experiment='revision_T4_shuffle', evaluation_filter_base='benchmark_data'))


def make_plan():
    jobs = []
    defaults = vars(base.parser().parse_args(['--setting', 't1_pair', '--model', 'ComplEx']))
    defaults = {k: v for k, v in defaults.items() if k not in ['data', 'out']}
    defaults.update(batch=4096, eval_batch=128, diagnostic_norms=True)
    for model in ['ComplEx', 'DistMult']:
        grid = ([dict(init_scale=s, nssa_margin=m, regularizer_weight=.01)
                 for s, m in itertools.product([.05, .1, .2], [0., 9.])]
                if model == 'ComplEx' else
                [dict(init_scale=1., nssa_margin=m, regularizer_weight=w)
                 for w, m in itertools.product([.00001, .0001, .001], [0., 9.])])
        grid.append(dict(init_scale=1., nssa_margin=9., regularizer_weight=.01 if model == 'ComplEx' else .1))
        for i, candidate in enumerate(grid):
            cfg = dict(defaults, **candidate, model=model, loss='nssa', validation_only=True)
            jobs.append(dict(id=f'hpo__{model.lower()}__c{i}', kind='hpo', stage='hpo',
                             model=model, setting='t1_pair', seed=42, data='benchmark_data', config=cfg, heavy=False))
    t1 = CORE[:4] + ['t1_loso_'+s.lower() for s in SOURCES]
    for model, setting, seed in itertools.product(['ComplEx', 'DistMult'], t1, SEEDS):
        jobs.append(dict(id=f'main__{setting}__{model.lower()}__s{seed}', kind='train', stage='train',
                         model=model, selected_model=model, setting=setting, seed=seed,
                         data='supplement_data/loso' if 'loso' in setting else 'benchmark_data', heavy=setting=='t1_bridges'))
    for model, seed in itertools.product(['ComplEx', 'DistMult'], SEEDS):
        jobs.append(dict(id=f'shuffle__t1_pair__{model.lower()}__s{seed}', kind='shuffle', stage='train',
                         model=model, selected_model=model, setting='t1_pair', seed=seed,
                         data=f'supplement_data/shuffle/t1_pair_seed{seed}', heavy=False))
    for model, seed in itertools.product(['RGCN', 'DistMult'], [123, 456]):
        original = next((ROOT / 'autodl/results/main').glob(f't4_a_{model.lower()}_*.json'))
        cfg = dict(read(original)['config'], seed=seed, split_seed=seed)
        jobs.append(dict(id=f'shuffle__t4_a__{model.lower()}__s{seed}', kind='shuffle', stage='train',
                         model=model, setting='t4_a', seed=seed, config=cfg,
                         data=f'revision_data/shuffle/t4_a_seed{seed}', heavy=True))
    for seed in SEEDS:
        jobs.append(dict(id=f'ultra__t2_a__s{seed}', kind='foundation', stage='foundation',
                         model='ULTRA', setting='t2_a', seed=seed, data='benchmark_data',
                         heavy=True, eval_batch=8, timeout_seconds=7200))
    for setting, seed in itertools.product(CORE + t1[4:], SEEDS):
        jobs.append(dict(id=f'kgicl__{setting}__s{seed}', kind='foundation', stage='foundation',
                         model='KG-ICL', setting=setting, seed=seed,
                         data='supplement_data/loso' if 'loso' in setting else 'benchmark_data',
                         heavy=True, eval_batch=1, timeout_seconds=7200))
    for job in list(jobs):
        if (job['kind']=='train' and job['setting'] in CORE[:4]) or (
                job['model']=='KG-ICL' and job['setting'] in CORE[:4]+['t4_a','t4_b']):
            jobs.append(dict(id='matched__'+job['id'], kind='matched', stage='matched', model=job['model'],
                             setting=job['setting'], seed=job['seed'], source_job=job['id'], heavy=True,
                             depends_on=[job['id']], eval_batch=1, timeout_seconds=7200))
    write_json(PLAN, dict(version='revision-v1', seeds=SEEDS,
        hpo_selection='seed42 fixed validation subset only; freeze before any new test evaluation',
        scope='T1 bilinear correction; KG-ICL second KG foundation model; ULTRA T2-A; T4-A shuffle seeds; P0/P1 offline analyses',
        legacy_results='immutable inputs; T2/T4 bilinear configurations unchanged',
        counts=dict(Counter(j['kind'] for j in jobs)), jobs=jobs))
    print(dict(Counter(j['kind'] for j in jobs)), 'total', len(jobs))


if __name__ == '__main__':
    extra_shuffle()
    make_plan()
