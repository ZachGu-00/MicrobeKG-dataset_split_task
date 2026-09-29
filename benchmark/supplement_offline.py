"""N2/N5 and S1-S4 analyses from saved observations; no model training."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from core import LEARNED, MODELS, known_maps, load_case, discrimination_metrics
from prepare import CORE, ROOT, sha256, write_json
from supplement_cell import BASELINE, OUTPUT, job_output


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def gini(v):
    v = np.sort(np.asarray(v, dtype=float))
    return float(2*np.dot(np.arange(1, len(v)+1), v)/(len(v)*v.sum()) - (len(v)+1)/len(v)) if v.sum() else 0.0


def collect():
    observations = [(p, read(p), 'baseline') for p in (BASELINE / 'main').glob('*.json')]
    plan = read(ROOT / 'supplement_plan.json')
    for job in plan['jobs']:
        receipt = job_output(job) / 'receipt.json'
        if not receipt.exists(): continue
        d = read(receipt)
        if d['status'] != 'complete': continue
        from supplement import verified_job
        assert verified_job(job, accept_limits=False), str(receipt)
        if job['kind'] in ['replicate', 'loso', 'shuffle', 'diagnostic']:
            paths = list((job_output(job) / 'main').glob('*.json'))
            assert len(paths) == 1
            observations.append((paths[0], read(paths[0]), job['kind']))
        elif job['kind'] == 'ultra':
            p = job_output(job) / 'result.json'; observations.append((p, read(p), 'ultra'))
    rows = []
    for path, d, group in observations:
        c = d['config']; rank = d.get('ranking') or {}
        fit = d.get('training', {})
        if not isinstance(fit, dict): fit = {}
        row = dict(group=group, model=c['model'], setting=c['setting'], seed=c['seed'],
            result_file=path.relative_to(ROOT).as_posix(), mrr=rank.get('mrr'), macro_mrr=rank.get('macro_mrr'),
            elapsed_sec=d.get('elapsed_sec'), best_epoch=fit.get('best_epoch'),
            epochs_run=fit.get('epochs_run'))
        for side in ['head', 'tail']: row[side + '_mrr'] = rank.get('by_side', {}).get(side, {}).get('mrr')
        for label, metrics in d.get('discrimination', {}).items():
            for key in ['auroc', 'auprc', 'floor', 'auprc_over_floor']: row[label + '_' + key] = metrics.get(key)
        rows.append(row)
    return observations, pd.DataFrame(rows)


def main():
    out = OUTPUT / 'analysis'; out.mkdir(parents=True, exist_ok=True)
    observations, table = collect()
    table.to_csv(out / 'all_observed_results.csv', index=False)
    baseline = table[table.group == 'baseline']
    primary = table[table.group.isin(['baseline', 'replicate', 'ultra', 'loso'])].copy()
    primary['score'] = primary['mrr']
    mask = primary.setting == 't4_b'
    primary.loc[mask, 'score'] = primary.loc[mask, 'association_without_recorded_therapy_auroc']
    aggregate = primary.groupby(['setting', 'model']).score.agg(['mean', 'std', 'count']).reset_index()
    aggregate.to_csv(out / 'main_mean_std.csv', index=False)
    pop = primary[primary.model == 'Popularity'][['setting', 'seed', 'score']].rename(columns={'score': 'popularity_score'})
    differences = primary.merge(pop, on=['setting', 'seed'])
    differences['difference'] = differences.score - differences.popularity_score
    differences = differences.groupby(['setting', 'model']).difference.agg(['mean', 'std', 'count']).reset_index()
    differences['absolute_mean_exceeds_std'] = np.where(differences['count'] >= 3, differences['mean'].abs() > differences['std'], None)
    differences.to_csv(out / 'paired_seed_difference_vs_popularity.csv', index=False)
    shuffled = table[table.group == 'shuffle'][['setting', 'model', 'seed', 'mrr']].rename(columns={'mrr': 'shuffled_mrr'})
    shuffled = shuffled.merge(primary[['setting', 'model', 'seed', 'mrr']], on=['setting', 'model', 'seed'])
    shuffled = shuffled.merge(pop, on=['setting', 'seed'])
    shuffled['change_from_original'] = shuffled.shuffled_mrr - shuffled.mrr
    shuffled['difference_from_popularity'] = shuffled.shuffled_mrr - shuffled.popularity_score
    shuffled.to_csv(out / 'shuffle_comparisons.csv', index=False)
    matched_rows = []
    from supplement import verified_job
    for job in read(ROOT / 'supplement_plan.json')['jobs']:
        if job['kind'] != 'matched' or not verified_job(job, accept_limits=False): continue
        result = read(job_output(job) / 'result.json')
        for control, metrics in result['metrics'].items():
            matched_rows.append(dict(setting=job['setting'], model=job['model'], seed=job['seed'], control=control,
                                     result_file=(job_output(job) / 'result.json').relative_to(ROOT).as_posix(), **metrics))
    matched_all = pd.DataFrame(matched_rows) if matched_rows else pd.DataFrame(columns=['setting', 'model', 'seed', 'control', 'auroc', 'auprc'])
    matched_all.to_csv(out / 'matched_all_observed.csv', index=False)
    matched_all.groupby(['setting', 'model', 'control'])[['auroc', 'auprc']].agg(['mean', 'std', 'count']).to_csv(out / 'matched_mean_std.csv')
    hpo = [read(p) for p in (BASELINE / 'hpo').glob('*.json')]
    hrows = [dict(model=d['config']['model'], setting=d['config']['setting'], dim=d['config']['dim'],
        negatives=d['config']['num_negs'], loss=d['config']['loss'], mrr=d['training']['best_validation_mrr'],
        best_epoch=d['training']['best_epoch'], epochs=d['training']['epochs_run']) for d in hpo]
    hd = pd.DataFrame(hrows); hd.to_csv(out / 'hpo_sensitivity.csv', index=False)
    summaries = hd.groupby(['setting', 'model']).mrr.agg(['min', 'max', 'std']).reset_index()
    summaries['span'] = summaries['max'] - summaries['min']
    summaries.to_csv(out / 'hpo_spans.csv', index=False)
    diag = [d for d in hpo if d['config']['setting'] == 't1_pair' and d['config']['model'] == 'ComplEx']
    reruns = [dict(result_file=p.relative_to(ROOT).as_posix(), config=d['config'],
        test_mrr=(d.get('ranking') or {}).get('mrr'), training=d.get('training'))
        for p, d, group in observations if group == 'diagnostic']
    write_json(out / 'complex_diagnostic.json', dict(status='saved-run diagnosis and second-config rerun complete; cause unresolved' if reruns else 'saved-run diagnosis complete; second-config rerun pending',
        second_config_reruns=reruns,
        configs=len(diag), all_best_epoch_10=all(d['training']['best_epoch'] == 10 for d in diag),
        all_losses_finite=all(np.isfinite(h['train_loss']) for d in diag for h in d['training']['history']),
        all_restore_checks_passed=all(d['training']['best_weights_restored'] for d in diag),
        interpretation='All eight trials peak at the first validation while loss decreases. No recorded NaN or failed restoration; this is poor validation performance, not evidence of a broken complex-tensor checkpoint. Full training trajectory norms were not saved.',
        runs=hd[(hd.model == 'ComplEx') & (hd.setting == 't1_pair')].to_dict('records')))
    runtimes = []
    for path, d, group in observations:
        if group != 'baseline': continue
        hw = d.get('training', {}).get('hardware', {})
        runtimes.append(dict(model=d['config']['model'], setting=d['config']['setting'], elapsed_sec=d['elapsed_sec'],
            gpu=hw.get('gpu'), peak_allocated_gib=(hw.get('peak_gpu_allocated_bytes') or 0)/2**30,
            epochs=d.get('training', {}).get('epochs_run')))
    pd.DataFrame(runtimes).to_csv(out / 'runtime_memory.csv', index=False)
    strata, stats, ratios, matched = [], [], [], []
    for setting in CORE:
        case = load_case(ROOT / 'benchmark_data', setting, 42)
        x, a = case['triples'], case['arrays']; types = case['entities'].node_type.to_numpy()
        degrees = np.bincount(case['train'][:, [0, 2]].ravel(), minlength=len(types))
        node_stats = {}
        for typ in ['microbe', 'disease']:
            vals = degrees[types == typ]; top = max(1, int(np.ceil(len(vals)*.01)))
            node_stats[typ] = dict(nodes=len(vals), gini_including_zero_degree=gini(vals),
                top_one_percent_edge_endpoint_share=float(np.sort(vals)[-top:].sum()/max(1, vals.sum())))
        kt, kh = known_maps(case)
        eligible = a['test'][a['test_eligible']]
        query_stats = {}
        for side, mapping, col in [('tail', kt, 0), ('head', kh, 2)]:
            if case['meta']['ranking_sides'] == 'tail' and side == 'head': continue
            wanted = case['meta'][side + '_type']
            rows = x[eligible].copy()
            if len(case['meta']['scoring_relations']) == 1:
                rows[:, 1] = case['relations'].index(case['meta']['scoring_relations'][0])
            queries = {(int(h), int(r)) if side == 'tail' else (int(r), int(t)) for h, r, t in rows}
            answers = [len(mapping.get(q, ())) for q in queries]
            query_stats[side] = dict(unique_test_queries=len(queries), average_known_true_answers=float(np.mean(answers)),
                typed_training_visible_candidates=int(((types == wanted) & case['eval_seen']).sum()))
        stats.append(dict(setting=setting, node_statistics=node_stats, query_statistics=query_stats,
            test_taxonomy_closure_fraction=float(np.mean((a.get('taxonomy_hop', np.array([0])) > 0))),
            notes='Total train endpoint degree; node Gini includes zero-degree typed nodes. Query answers use frozen filtered-evaluation truths. A closure is available evidence, not proof that L3 used that path.'))
        sub = baseline[baseline.setting == setting]
        if setting != 't4_b':
            best = sub.loc[sub.mrr.idxmax()]; pop = sub[sub.model == 'Popularity'].iloc[0]
            ratios.append(dict(setting=setting, best_model=best.model, best_mrr=best.mrr, popularity_mrr=pop.mrr,
                               popularity_over_best=pop.mrr/best.mrr))
        for path, d, group in observations:
            if group != 'baseline' or d['config']['setting'] != setting: continue
            with np.load(path.with_suffix('.npz')) as z: raw = dict(z)
            if setting.startswith('t1') and d['config']['model'] in ['RotatE', 'TuckER', 'L3', 'Popularity']:
                deg = degrees[raw['ranking_triples'][:, 2]]
                for label, lo, hi in [('0', 0, 0), ('1-2', 1, 2), ('3-9', 3, 9), ('10-49', 10, 49), ('50+', 50, np.inf)]:
                    mask = (deg >= lo) & (deg <= hi)
                    for side in ['head', 'tail']:
                        ranks = raw[side + '_ranks'][mask]
                        strata.append(dict(setting=setting, model=d['config']['model'], degree_bucket=label,
                            side=side, n=len(ranks), mrr=float(np.mean(1/ranks)) if len(ranks) else None))
            match_file = ROOT / 'supplement_data/matched' / setting / 'seed_42.npz'
            if match_file.exists():
                with np.load(match_file) as z: match = dict(z)
                label = 'direction_conflict_probe' if setting.startswith('t1') else 'association_without_recorded_therapy'
                pi, ni = match['positive_indices'], match['negative_indices']
                metric = discrimination_metrics(raw[label + '_pos_scores'][pi], raw[label + '_neg_scores'][ni], n_boot=0)
                matched.append(dict(setting=setting, model=d['config']['model'], seed=42,
                    original_auroc=d['discrimination'][label]['auroc'], original_auprc=d['discrimination'][label]['auprc'], **metric))
    pd.DataFrame(strata).to_csv(out / 'disease_degree_strata.csv', index=False)
    pd.DataFrame(ratios).to_csv(out / 'popularity_share.csv', index=False)
    pd.DataFrame(matched).to_csv(out / 'recorded_degree_matched_seed42.csv', index=False)
    write_json(out / 'graph_query_statistics.json', stats)
    ranked = primary[primary.seed == 42].pivot(index='model', columns='setting', values='score')
    tau = ranked.corr(method='kendall'); tau.to_csv(out / 'rank_stability_kendall.csv')
    loso_correlations = []
    for seed in sorted(primary.seed.unique()):
        per_seed = primary[primary.seed == seed].pivot(index='model', columns='setting', values='score')
        for setting in per_seed.columns:
            if 'loso' not in setting: continue
            for anchor in ['t1_pair', 't1_cold_microbe']:
                if anchor not in per_seed: continue
                pair = per_seed[[anchor, setting]].dropna()
                if len(pair) < 2: continue
                loso_correlations.append(dict(seed=int(seed), anchor=anchor, setting=setting, n_models=len(pair),
                    spearman=float(spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic),
                    kendall=float(kendalltau(pair.iloc[:, 0], pair.iloc[:, 1]).statistic)))
    write_json(out / 'loso_rank_correlations.json', dict(status='observed' if loso_correlations else 'pending_server_results', rows=loso_correlations))
    entities = pd.read_csv(ROOT / 'benchmark_data/entities.tsv', sep='\t', dtype={'node_id': str})
    triples = np.load(ROOT / 'benchmark_data/triples.npy', mmap_mode='r')
    candidates = []
    for setting, model, label in [('t2_b', 'ComplEx', 'experimental_nonutilization'), ('t4_b', 'DistMult', 'association_without_recorded_therapy')]:
        path, d, _ = next(v for v in observations if v[2] == 'baseline' and v[1]['config']['setting'] == setting and v[1]['config']['model'] == model)
        with np.load(path.with_suffix('.npz')) as z:
            ids, scores = z[label + '_pos_ids'], z[label + '_pos_scores']
            for rank, i in enumerate(np.argsort(-scores)[:10], 1):
                h, r, t = triples[ids[i]]
                candidates.append(dict(setting=setting, model=model, rank=rank, triple_row_id=int(ids[i]), score=float(scores[i]),
                    head_id=entities.iloc[h].node_id, head_name=entities.iloc[h].node_name,
                    tail_id=entities.iloc[t].node_id, tail_name=entities.iloc[t].node_name,
                    evidence_status='Held-out known positive for manual case review; independent evidence not verified'))
    pd.DataFrame(candidates).to_csv(out / 'case_review_candidates.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout='constrained')
    sd = pd.DataFrame(strata)
    for model in ['RotatE', 'TuckER', 'L3', 'Popularity']:
        part = sd[(sd.setting == 't1_pair') & (sd.model == model) & (sd.side == 'tail')]
        axes[0].plot(part.degree_bucket, part.mrr, marker='o', label=model)
    axes[0].set(xlabel='Disease train degree', ylabel='Tail MRR', title='T1 pair: performance by disease degree'); axes[0].legend(fontsize=8)
    for i, model in enumerate(LEARNED):
        vals = hd[(hd.setting == 't1_pair') & (hd.model == model)].mrr.to_numpy()
        axes[1].scatter([i]*len(vals), vals, s=18)
    axes[1].set_xticks(range(len(LEARNED)), LEARNED, rotation=35)
    axes[1].set(ylabel='Validation MRR', title='Eight HPO configurations per model')
    im = axes[2].imshow(tau.to_numpy(), vmin=-1, vmax=1, cmap='coolwarm')
    axes[2].set_xticks(range(len(tau)), tau.columns, rotation=90, fontsize=7)
    axes[2].set_yticks(range(len(tau)), tau.index, fontsize=7)
    axes[2].set_title('Model-rank Kendall tau'); fig.colorbar(im, ax=axes[2], shrink=.7)
    fig.savefig(out / 'offline_diagnostics.png', dpi=200); fig.savefig(out / 'offline_diagnostics.pdf'); plt.close(fig)
    matched_frame = pd.DataFrame(matched)
    write_json(out / 'offline_status.json', dict(status='completed_from_existing_observations',
        existing_main_cells=99, hpo_cells=96, observed_supplement_cells=len(observations)-99,
        completed=['N2 saved trajectory/restore inspection', 'N4 recorded-control reweighting on seed42 raw scores',
                   'N5 disease-degree strata / degree inequality / answer counts / closure fraction', 'S1 HPO sensitivity', 'S2 runtime/memory', 'S3 rank stability'],
        partial=['S4 exports 20 manual review candidates; no independent literature confirmation'],
        expected_123_456_cell_seconds=sum(d['elapsed_sec'] for _, d, g in observations if g == 'baseline')*2,
        matched_popularity=matched_frame[matched_frame.model == 'Popularity'].to_dict('records'),
        caveats=['Main and LOSO aggregates use observed seeds; graph/degree/HPO diagnostics retain the explicitly stated seed-42 scope.',
                 'No significance claim from marginal rank alone; no outcome-dependent deletion of low scores.',
                 'Recorded-control matching and completed unknown-fallback inference are separate estimands; inspect matched_all_observed.csv.']))
    print('Offline analyses updated:', out, 'observed supplementary cells:', len(observations)-99, flush=True)


if __name__ == '__main__':
    main()
