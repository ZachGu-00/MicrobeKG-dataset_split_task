#!/usr/bin/env python3
"""Create one training job per audit split (dataset, seed, protocol in {A,B}, fold) for KGNMDA.

Everything split-dependent comes from the audit module itself (`hmdad_audit.build_splits`, `gip`), so folds and
seeds are the audit's own. Protocol C reuses the protocol-B models (cold-microbe subset of the same test fold), exactly as
in `hmdad_audit.cv`.

Per job (training fold only):
  * Gaussian (GIP) similarity kernels from the training adjacency  -> KGNMDA's `GM` / `GD` feature vectors
  * training positives = training edges; training negatives = |train| pairs drawn uniformly from all pairs that are NOT
    training positives (KGNMDA's "equal number of randomly sampled unknown pairs"). Held-out positives are NOT excluded
    from the pool, because excluding them would use test labels.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--audit-dir', default='/root/autodl-tmp/mkg/baselines_v3/kgnmda/audit_hmdad')
ap.add_argument('--map', default='/root/autodl-tmp/mkg/baselines_v3/hmdad_entity_map.json')
ap.add_argument('--out', default='/root/autodl-tmp/mkg/baselines_v3/kgnmda/jobs')
ap.add_argument('--dataset', default='HMDAD')
ap.add_argument('--seeds', default='0,1,2,3,4')
a = ap.parse_args()
sys.path.insert(0, a.audit_dir)
import hmdad_audit as H

ds = H.load_dataset(a.dataset)
emap = json.load(open(a.map))
assert emap['microbes'] == ds.microbes and emap['diseases'] == ds.diseases, 'entity map is not aligned with the audit dataset'
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
n = 0
for seed in [int(s) for s in a.seeds.split(',')]:
    for protocol, fold, train_ids, test_ids, _ in H.build_splits(ds, seed):
        T = np.zeros_like(ds.A); T[tuple(ds.edges[train_ids].T)] = 1
        assert T[tuple(ds.edges[test_ids].T)].sum() == 0
        km, kd = H.gip(T)                       # training-fold kernels only
        rng = np.random.default_rng(900000 + seed * 100 + fold * 10 + (0 if protocol == 'A' else 1))
        unlabeled = np.argwhere(T == 0)         # (microbe, disease) pairs not known as training positives
        neg = unlabeled[rng.choice(len(unlabeled), len(train_ids), replace=False)]
        np.savez_compressed(out / f'{a.dataset}_seed{seed}_{protocol}_fold{fold}.npz', seed=seed, fold=fold, protocol=protocol,
                            train_pos=ds.edges[train_ids], train_neg=neg, test_ids=test_ids, km=km, kd=kd,
                            micro_eid=np.array(emap['micro_eid']), dis_eid=np.array(emap['dis_eid']),
                            n_entities=emap['n_entities_total'])
        n += 1
print(f'wrote {n} jobs to {out}')
