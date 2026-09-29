"""Prepare frozen LOSO, target-shuffle and matched-control inputs; never train."""
from collections import defaultdict
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from core import load_case, LightScorer, discrimination_metrics
from prepare import ROOT, SEEDS, TAX, closure_flags, pair_code, partition, save_split, sha256, write_json

BASE = ROOT / 'benchmark_data'
OUT = ROOT / 'supplement_data'
SOURCES = ['Peryton', 'gutMDisorder_v2', 'HMDAD']
MATCH_SETTINGS = ['t1_pair', 't1_bridges', 't1_cold_microbe', 't1_cold_disease', 't4_a', 't4_b']


def finish_dataset(folder, metas, extra):
    original = json.loads((BASE / 'manifest.json').read_text())
    write_json(folder / 'manifest.json', dict(protocol='supplement-v1', seeds=sorted({m['seed'] for m in metas}),
        original_edges_sha256=original['original_edges_sha256'], base_manifest_sha256=sha256(BASE / 'manifest.json'),
        splits=metas, **extra, files={p.relative_to(folder).as_posix(): sha256(p)
            for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'manifest.json'}))


def copy_common(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name in ['entities.tsv', 'relations.json', 'triples.npy']:
        shutil.copy2(BASE / name, folder / name)


def source_index():
    """Source union is over both target directions at the typed pair level."""
    target = OUT / 't1_pair_sources.json'
    digest = sha256(ROOT / 'kg_build/final_kg_edges.tsv')
    if target.exists():
        saved = json.loads(target.read_text())
        if saved['original_edges_sha256'] == digest:
            return {int(k): set(v) for k, v in saved['pair_sources'].items()}
    nodes = pd.read_csv(BASE / 'entities.tsv', sep='\t', dtype={'node_id': str})
    lookup = {(r.node_type, r.node_id): r.entity_index for r in nodes.itertuples()}
    result = defaultdict(set)
    cols = ['head_id', 'head_type', 'relation', 'tail_id', 'tail_type', 'source']
    for df in pd.read_csv(ROOT / 'kg_build/final_kg_edges.tsv', sep='\t', usecols=cols, dtype=str,
                          keep_default_na=False, chunksize=200000):
        part = df[(df.head_type == 'microbe') & (df.tail_type == 'disease') & df.relation.isin(['enriched_in', 'depleted_in'])]
        for r in part.itertuples():
            code = lookup[('microbe', r.head_id)] * len(nodes) + lookup[('disease', r.tail_id)]
            result[code].update(r.source.split('|'))
    write_json(target, dict(original_edges_sha256=digest,
        definition='Union of source labels across enriched_in/depleted_in for each typed microbe-disease pair',
        pair_sources={str(k): sorted(v) for k, v in sorted(result.items())}))
    return result


def prepare_loso():
    folder = OUT / 'loso'; copy_common(folder)
    sources = source_index()
    x = np.load(BASE / 'triples.npy', mmap_mode='r')
    nodes = pd.read_csv(BASE / 'entities.tsv', sep='\t')
    types = nodes.node_type.to_numpy(); n = len(nodes)
    rels = json.loads((BASE / 'relations.json').read_text())
    md = (types[x[:, 0]] == 'microbe') & (types[x[:, 2]] == 'disease')
    pos = np.flatnonzero(md & np.isin(x[:, 1], [rels.index(r) for r in ['enriched_in', 'depleted_in']]))
    aux = np.flatnonzero(np.isin(x[:, 1], [rels.index(r) for r in TAX + ['is_a', 'co_occurs_with']]))
    conflict = np.flatnonzero(md & (x[:, 1] == rels.index('inconsistent_association')))
    codes = pair_code(x, n); unordered = pair_code(x, n, True)
    metas, reports = [], []
    for source in SOURCES:
        exclusive = np.array([p for p, labels in sources.items() if labels == {source}], dtype=np.int64)
        shared = {p for p, labels in sources.items() if source in labels and len(labels) > 1}
        test = pos[np.isin(codes[pos], exclusive)]
        remaining = pos[~np.isin(codes[pos], exclusive)]
        assert len(test), ('No exclusive target pairs', source)
        for seed in SEEDS:
            # Source-shared pairs always stay in train, as required by the checklist.
            eligible_valid = remaining[~np.isin(codes[remaining], list(shared))]
            _, valid_pairs, _ = partition(codes[eligible_valid], seed, (0.9, 0.1))
            valid = remaining[np.isin(codes[remaining], valid_pairs)]
            train = np.r_[remaining[~np.isin(codes[remaining], valid_pairs)], aux]
            train = train[~np.isin(unordered[train], unordered[np.r_[valid, test]])]
            cf = conflict[np.isin(codes[conflict], exclusive)]
            setting = 't1_loso_' + source.lower()
            meta = save_split(folder, setting, seed, x, types, rels, train, valid, test,
                              ['enriched_in', 'depleted_in'], conflict=cf)
            meta['source_holdout'] = dict(source=source, exclusive_pairs=len(exclusive),
                shared_pairs_kept_in_train=len(shared), definition='Source-exclusive target pairs; auxiliary graph retained; not a temporal or strictly unseen-entity split')
            write_json(folder / 'splits' / setting / f'seed_{seed}.json', meta)
            metas.append(meta)
            reports.append(dict(source=source, seed=seed, exclusive_pairs=len(exclusive), test_triples=len(test),
                                shared_pairs=len(shared), **meta['audit']))
    finish_dataset(folder, metas, dict(experiment='N7', sources=SOURCES))
    write_json(OUT / 'loso_counts.json', reports)


def degree_preserving_shuffle(train, target_relations, forbidden_pairs, n, seed):
    """Simple directed-graph double-edge swaps preserve each relation's in/out degree."""
    result = train.copy(); rng = np.random.default_rng(seed); reports = []
    for relation in target_relations:
        positions = np.flatnonzero(result[:, 1] == relation)
        before = result[positions].copy()
        edges = {(int(h), int(t)) for h, _, t in before}
        assert len(edges) == len(before)
        accepted = 0
        for _ in range(20 * len(positions)):
            i, j = rng.integers(len(positions), size=2)
            h1, _, t1 = result[positions[i]]; h2, _, t2 = result[positions[j]]
            if h1 == h2 or t1 == t2 or (int(h1), int(t2)) in edges or (int(h2), int(t1)) in edges:
                continue
            pairs = [int(min(h1, t2)) * n + int(max(h1, t2)), int(min(h2, t1)) * n + int(max(h2, t1))]
            if any(p in forbidden_pairs for p in pairs):
                continue
            edges.remove((int(h1), int(t1))); edges.remove((int(h2), int(t2)))
            edges.add((int(h1), int(t2))); edges.add((int(h2), int(t1)))
            result[positions[i], 2], result[positions[j], 2] = t2, t1
            accepted += 1
            if accepted >= 5 * len(positions):
                break
        after = result[positions]
        for col in [0, 2]:
            assert np.array_equal(np.bincount(before[:, col], minlength=n), np.bincount(after[:, col], minlength=n))
        reports.append(dict(relation=int(relation), triples=len(positions), accepted_swaps=accepted,
                            changed_fraction=float(np.mean(before[:, 2] != after[:, 2]))))
    return result, reports


def prepare_shuffle():
    for setting, seeds in [('t1_pair', SEEDS), ('t4_a', [42])]:
        for seed in seeds:
            case = load_case(BASE, setting, seed)
            a = case['arrays']; x = case['triples']; n = len(case['seen'])
            held = np.r_[a['valid'], a['test'], a['negatives'], a['conflict']]
            forbidden = set(pair_code(x[held], n, True))
            ids = [case['relations'].index(r) for r in case['meta']['scoring_relations']]
            shuffled, stats = degree_preserving_shuffle(case['train'], ids, forbidden, n, seed)
            folder = OUT / 'shuffle' / f'{setting}_seed{seed}'; copy_common(folder)
            changed_x = np.array(x); changed_x[a['train']] = shuffled
            np.save(folder / 'triples.npy', changed_x, allow_pickle=False)
            assert not np.isin(pair_code(shuffled, n, True), list(forbidden)).any()
            assert np.array_equal(np.bincount(shuffled[:, [0, 2]].ravel(), minlength=n),
                                  np.bincount(case['train'][:, [0, 2]].ravel(), minlength=n))
            arrays = dict(a)
            if setting.startswith('t1'):
                arrays['taxonomy_hop'], arrays['ontology_hop'] = closure_flags(shuffled, x[a['test']], case['relations'])
            split = folder / 'splits' / setting; split.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(split / f'seed_{seed}.npz', **arrays)
            meta = copy.deepcopy(case['meta'])
            meta['perturbation'] = dict(method='5 accepted degree-preserving swaps per target edge, at most 20 attempts per edge',
                per_relation=stats, protected='All validation/test/control pairs, exact target in/out degrees, auxiliary triples, candidate and eligibility masks',
                filtering='Evaluation uses original unshuffled train truths; supplementary cell runner enforces this')
            if setting.startswith('t1'):
                th, dh = arrays['taxonomy_hop'], arrays['ontology_hop']
                meta['audit'].update(taxonomy_hop1=int((th == 1).sum()), taxonomy_hop2=int((th == 2).sum()),
                    ontology_hop1=int((dh == 1).sum()), ontology_hop2=int((dh == 2).sum()), no_closure_within_2hop=int(((th == 0) & (dh == 0)).sum()))
            write_json(split / f'seed_{seed}.json', meta)
            finish_dataset(folder, [meta], dict(experiment='N3', evaluation_filter_base='benchmark_data'))
            print('Shuffled', setting, seed, stats, flush=True)


def match_controls(positive, negative, train, all_triples, types, seen, seed):
    """1:1 matching by both endpoint degree buckets; unknown fallbacks remain labelled."""
    n = len(types); degree = np.bincount(train[:, [0, 2]].ravel(), minlength=n)
    buckets = np.floor(np.log2(np.maximum(degree, 1))).astype(int)
    log_degree = np.log2(np.maximum(degree, 1))
    pools = defaultdict(list)
    for i, (h, _, t) in enumerate(negative):
        pools[(buckets[h], buckets[t])].append(i)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(positive))
    pi, ni, missing = [], [], []
    for i in order:
        h, _, t = positive[i]; pool = pools.get((buckets[h], buckets[t]), [])
        if not pool:
            missing.append(i); continue
        # Select using degrees only; never model scores or test labels.
        rows = negative[pool]
        distance = abs(log_degree[rows[:, 0]] - log_degree[h]) + abs(log_degree[rows[:, 2]] - log_degree[t])
        tied = np.flatnonzero(distance == distance.min())
        chosen = int(rng.choice(tied)); pi.append(i); ni.append(pool.pop(chosen))
    forbidden = set(pair_code(all_triples, n, True))
    tail_type = types[positive[0, 2]]
    tail_pools = {b: np.flatnonzero((types == tail_type) & seen & (buckets == b)) for b in np.unique(buckets)}
    fallback_pi, fallback_rows = [], []
    for i in missing:
        h, r, t = positive[i]
        for cand in rng.permutation(tail_pools[buckets[t]]):
            code = int(min(h, cand)) * n + int(max(h, cand))
            if code not in forbidden:
                fallback_pi.append(i); fallback_rows.append([h, r, cand]); forbidden.add(code); break
    pi, ni = np.asarray(pi, np.int32), np.asarray(ni, np.int32)
    fallback = np.asarray(fallback_rows, np.int32).reshape(-1, 3)
    scorer = LightScorer(train, n, 'Popularity', seed)
    balance = discrimination_metrics(scorer.triples(positive[pi]), scorer.triples(negative[ni]), n_boot=0)
    return dict(positive_indices=pi, negative_indices=ni, fallback_positive_indices=np.asarray(fallback_pi, np.int32),
                fallback_triples=fallback), dict(original_positives=len(positive), original_controls=len(negative),
                matched_recorded_controls=len(pi), unknown_fallback=len(fallback), unmatched=len(positive)-len(pi)-len(fallback),
                fallback_fraction=len(fallback)/len(positive), matched_positive_coverage=len(pi)/len(positive),
                popularity_matched=balance, popularity_balance_within_003=balance['auroc'] is not None and abs(balance['auroc']-0.5) <= 0.03)


def prepare_matching():
    reports = []
    for setting in MATCH_SETTINGS:
        for seed in SEEDS:
            case = load_case(BASE, setting, seed); a = case['arrays']; x = case['triples']
            if setting.startswith('t1'):
                pi, ni = a['probe_positive'], a['conflict'][a['conflict_eligible']]
                label = 'direction_conflict_probe'
            else:
                pi, ni = a['test'][a['test_eligible']], a['negatives'][a['negative_eligible']]
                label = 'association_without_recorded_therapy'
            arrays, report = match_controls(x[pi], x[ni], case['train'], x,
                case['entities'].node_type.to_numpy(), case['eval_seen'], seed)
            arrays.update(original_positive_ids=pi, original_negative_ids=ni)
            folder = OUT / 'matched' / setting; folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / f'seed_{seed}.npz', **arrays)
            report.update(setting=setting, seed=seed, label=label, ratio='1:1 without replacement',
                eligibility='Original held-out control pool only; no training-observed pair admitted',
                fallback_semantics='Unknown pairs are scored and reported separately, never treated as recorded conflict/association controls')
            write_json(folder / f'seed_{seed}.json', report); reports.append(report)
            print('Matched', setting, seed, len(arrays['positive_indices']), 'fallback', len(arrays['fallback_triples']), flush=True)
    write_json(OUT / 'matched_counts.json', reports)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=['all', 'loso', 'shuffle', 'matched'], default='all')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [('loso', prepare_loso), ('shuffle', prepare_shuffle), ('matched', prepare_matching)]:
        if args.stage in ['all', name]: fn()
    write_json(OUT / 'manifest.json', dict(protocol='supplement-v1', base_manifest_sha256=sha256(BASE / 'manifest.json'),
        files={p.relative_to(OUT).as_posix(): sha256(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p != OUT / 'manifest.json'}))


if __name__ == '__main__':
    main()
