#!/usr/bin/env python3
"""Score KGNMDA predictions with the audit's own protocol code (hmdad_audit.py, imported not re-implemented).

For every (seed, protocol A/B, fold) the held-out positives are ranked against ALL pairs unknown in the full benchmark
(AUROC with tie correction, AP), plus the degree-matched-negative AUROC with the audit's RNG formula. Protocol C is the
cold-microbe subset of the protocol-B test fold (same trained model), exactly as in `hmdad_audit.cv`.
"""
import argparse, sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--audit-dir', required=True)
ap.add_argument('--preds', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--dataset', default='HMDAD')
ap.add_argument('--seeds', default='0,1,2,3,4')
ap.add_argument('--method', default='KGNMDA')
ap.add_argument('--score', default='logit', choices=['logit', 'prob'])
a = ap.parse_args()
sys.path.insert(0, a.audit_dir)
import hmdad_audit as H

ds = H.load_dataset(a.dataset)
unknown = ds.A == 0
rows = []
for seed in [int(s) for s in a.seeds.split(',')]:
    for protocol, fold, train_ids, test_ids, _ in H.build_splits(ds, seed):
        P = np.load(Path(a.preds) / f'{a.dataset}_seed{seed}_{protocol}_fold{fold}.npz')
        S = P[a.score].astype(np.float64)
        assert S.shape == ds.A.shape and np.isfinite(S).all()
        T = np.zeros_like(ds.A); T[tuple(ds.edges[train_ids].T)] = 1
        testsets = {protocol: test_ids}
        if protocol == 'B':
            testsets['C'] = test_ids[T.sum(1)[ds.edges[test_ids, 0]] == 0]
        matching = {tp: H.matched_negatives(T, ds, ids, np.random.default_rng(seed * 1000 + fold * 10 + (0 if protocol == 'A' else 1) + 50000))
                    for tp, ids in testsets.items()}
        for tp, ids in testsets.items():
            test = ds.edges[ids]
            auc, ap_ = H.rank_metrics(S[tuple(test.T)], S[unknown])
            negpairs, keep, gaps = matching[tp]
            kp = ds.edges[keep]
            ma, mp = H.rank_metrics(S[tuple(kp.T)], S[tuple(negpairs.T)]) if len(keep) else (float('nan'), float('nan'))
            rows.append({'dataset': ds.name, 'method': a.method, 'protocol': tp, 'kernel_mode': 'train', 'seed': seed, 'fold': fold,
                         'n_train': len(train_ids), 'n_test': len(ids), 'n_negative': int(unknown.sum()),
                         'auroc': auc, 'auprc': ap_, 'auroc_matched': ma, 'auprc_matched': mp, 'n_matched': len(keep),
                         'train_seconds': float(P['train_seconds']), 'status': 'ok' if len(ids) else 'no_test_positives'})
df = pd.DataFrame(rows)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / f'{a.dataset.lower()}_per_fold.csv', index=False)

summ = []
for (d, m, p), g in df.groupby(['dataset', 'method', 'protocol'], sort=False):
    row = {'dataset': d, 'method': m, 'protocol': p}
    for metric in ['auroc', 'auprc', 'auroc_matched']:
        x = g[metric].dropna()
        row[f'{metric}_mean'] = x.mean()
        if metric == 'auroc':
            row['auroc_sd'] = x.std(ddof=1)
    row['n_rows'] = len(g)
    # extras (after the audit-schema columns)
    row['n_rows_valid'] = int(g.auroc.notna().sum())
    row['auroc_seedmean_sd'] = g.groupby('seed').auroc.mean().std(ddof=1)
    row['n_test_sum'] = int(g.n_test.sum())
    row['train_seconds_mean'] = g.train_seconds.mean()
    summ.append(row)
cols = ['dataset', 'method', 'protocol', 'auroc_mean', 'auroc_sd', 'auprc_mean', 'auroc_matched_mean', 'n_rows']
S_ = pd.DataFrame(summ)
S_[cols].to_csv(out / f'{a.dataset.lower()}_results.csv', index=False)
S_.to_csv(out / f'{a.dataset.lower()}_results_extended.csv', index=False)
pd.set_option('display.width', 250)
print(S_.to_string())
