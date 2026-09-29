"""Recompute point metrics and index all final results, without model training."""
from collections import Counter
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmark'))
from core import discrimination_metrics, ranking_metrics
from prepare import write_json

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

errors = []
def compare(actual, saved, name):
    for key, value in actual.items():
        if key.endswith('ci95') or value is None:
            continue
        if isinstance(value, dict):
            compare(value, saved[key], name + '/' + key)
        elif isinstance(value, (int, float)):
            delta = abs(float(value) - float(saved[key]))
            assert np.isfinite(delta) and delta < 1e-12, (name, key, value, saved[key])
            errors.append(delta)

index = []
audited = 0
def index_result(path, group, job_id=''):
    d = read(path)
    c = d.get('config', d)
    index.append(dict(group=group, job_id=job_id, model=c['model'], setting=c['setting'], seed=c['seed'],
        status=d['status'], result_file=path.relative_to(ROOT).as_posix(),
        raw_file=path.with_suffix('.npz').relative_to(ROOT).as_posix(),
        weight_file=path.with_suffix('.pt').relative_to(ROOT).as_posix() if path.with_suffix('.pt').exists() else ''))
    return d

for group, folder in [('hpo', 'hpo'), ('baseline', 'main')]:
    for path in sorted((ROOT / 'autodl/results' / folder).glob('*.json')):
        d = index_result(path, group)
        if group == 'hpo':
            continue
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as z:
            raw = dict(z)
        if d.get('ranking'):
            compare(ranking_metrics(raw, n_boot=0), d['ranking'], str(path))
        for label, metrics in d.get('discrimination', {}).items():
            compare(discrimination_metrics(raw[label+'_pos_scores'], raw[label+'_neg_scores'], n_boot=0), metrics, label)
        audited += 1

plan = read(ROOT / 'supplement_plan.json')
for job in plan['jobs']:
    folder = ROOT / 'autodl/supplement_results/jobs' / job['id']
    receipt = read(folder / 'receipt.json')
    assert receipt['status'] == 'complete' and receipt['job'] == job
    paths = list((folder / 'main').glob('*.json')) if 'config' in job else [folder / 'result.json']
    assert len(paths) == 1
    d = index_result(paths[0], job['kind'], job['id'])
    with np.load(paths[0].with_suffix('.npz'), allow_pickle=False) as z:
        raw = dict(z)
    if job['kind'] == 'matched':
        label = d['matching']['label']
        with np.load((ROOT / job['source_result']).with_suffix('.npz'), allow_pickle=False) as z:
            original_pos, original_neg = z[label+'_pos_scores'], z[label+'_neg_scores']
        pairs = dict(original=(original_pos, original_neg),
            recorded_degree_matched=(raw['matched_positive_scores'], raw['matched_negative_scores']),
            original_controls_on_matched_positive_subset=(raw['matched_positive_scores'], original_neg))
        if 'unknown_fallback_only' in d['metrics']:
            pairs['unknown_fallback_only'] = (raw['fallback_positive_scores'], raw['fallback_negative_scores'])
        for control, values in pairs.items():
            compare(discrimination_metrics(*values, n_boot=0), d['metrics'][control], job['id']+'/'+control)
    else:
        if d.get('ranking'):
            compare(ranking_metrics(raw, n_boot=0), d['ranking'], job['id'])
        for label, metrics in d.get('discrimination', {}).items():
            compare(discrimination_metrics(raw[label+'_pos_scores'], raw[label+'_neg_scores'], n_boot=0), metrics, job['id']+'/'+label)
        training = d.get('training')
        if isinstance(training, dict) and 'history' in training:
            assert training['best_weights_restored']
            assert all(np.isfinite(row['train_loss']) for row in training['history'])
    audited += 1
assert len(index) == 762 and audited == 666
assert len({row['result_file'] for row in index}) == 762
out = ROOT / 'autodl/supplement_results/analysis/experiment_index.csv'
with out.open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(index[0]))
    writer.writeheader(); writer.writerows(index)
summary = dict(status='passed', checked_utc=datetime.now(timezone.utc).isoformat(),
    total_result_pairs=762, indexed_by_group=dict(Counter(r['group'] for r in index)),
    original_weights=54, supplemental_weights=183, supplemental_jobs=567,
    point_metrics_recomputed_cells=audited, point_metric_comparisons=len(errors),
    max_absolute_difference=max(errors), confidence_intervals_recomputed=False,
    model_inference_rerun=False, index=out.relative_to(ROOT).as_posix(),
    download_integrity='archive/2026-09-17_complete_import/import_merge.json')
write_json(ROOT / 'autodl/supplement_results/local_checks/final_audit.json', summary)
print(json.dumps(summary, indent=2), flush=True)
