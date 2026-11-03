"""Tested-null probe v3 (P0-2), computed offline from the probe score matrices written by rescore_v3.py.

Null source.  GMrepo v3 disease contrasts recomputed by the resource authors (family_tests.tsv.gz; 78 diseases,
170,783 tests): for disease D, case runs = valid metagenomic runs of D, controls = 12,205 healthy runs pooled over
129 projects; species present in >= 1 case or control run; two-sided Mann-Whitney U (exact conditional p when the
species is present in < 50 runs, tie-corrected normal approximation otherwise); Benjamini-Hochberg within D over
the whole family.  A tested null is a pair with fdr > F, |log2FC| < L, present in >= K runs, and no enriched_in /
depleted_in / inconsistent_association assertion in the graph.  Default (F, L, K) = (0.2, 0.5, 20).

All AUROCs below compare held-out positives P with negatives of the SAME disease (within-disease AUROC), and the
`*_matched` variants compare only within strata of the same disease.  Per (seed, disease) we store the numerator
sum(AUC_g * nP_g * nN_g) and denominator sum(nP_g * nN_g); the summary resamples seeds, then diseases.
"""
from __future__ import annotations

import itertools
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

MKG = Path("/root/autodl-tmp/mkg")
ROOT = Path("/root/autodl-tmp/MicrobeKG_data_20261031")
RS = MKG / "results3/rescore"
OUT = MKG / "results3/analysis"
GM = ROOT / ("paper_work_collaboration/overleaf/microbekg_p0024_overleaf/review/"
             "20260928_corrected/claude/release/gmrepo/work/family_tests.tsv.gz")
NAME = re.compile(r"(t1_\w+?)_([a-z0-9-]+)_split(\d+)_seed(\d+)_d(\d+)_n(\d+)_(\w+)\.npz")
RNG = np.random.default_rng(11)
DEFAULT = (0.2, 0.5, 20)
GRID = list(itertools.product([0.1, 0.2, 0.5], [0.25, 0.5, 1.0], [10, 20, 50]))
DET_EDGES = [0.02, 0.05, 0.1, 0.2, 0.4]


def root_of(s):
    if s == "t1_external":
        return MKG / "bench_ext"
    return MKG / ("bench3" if (MKG / "bench3/splits" / s).exists() else "bench2")


def model_key(model, loss, d):
    k = f"{model}-{loss}" if loss in ("zeroshot", "finetune") else model
    return k


def auc_parts(p, q):
    """AUROC(p > q) with ties 1/2, returned as (AUC * n_p * n_q, n_p * n_q)."""
    if len(p) == 0 or len(q) == 0:
        return 0.0, 0.0
    r = rankdata(np.r_[p, q])
    u = r[:len(p)].sum() - len(p) * (len(p) + 1) / 2
    return float(u), float(len(p) * len(q))


def strat_parts(p, q, gp, gq):
    num = den = 0.0
    for g in np.intersect1d(np.unique(gp), np.unique(gq)):
        a, b = auc_parts(p[gp == g], q[gq == g])
        num += a
        den += b
    return num, den


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ents = pd.read_csv(MKG / "bench2/entities.tsv", sep="\t", dtype=str, keep_default_na=False)
    n = len(ents)
    types = ents.node_type.to_numpy()
    node = {(t, i): j for j, (t, i) in enumerate(zip(ents.node_type, ents.node_id))}
    rels = json.loads((MKG / "bench2/relations.json").read_text())
    rid = {r: i for i, r in enumerate(rels)}
    REL = [rid["enriched_in"], rid["depleted_in"]]
    gm = pd.read_csv(GM, sep="\t", usecols=["disease", "head_id", "n_case", "present_case", "present_total", "log2fc",
                                             "fdr"])
    gm = gm[[("microbe", h) in node and ("disease", d) in node for h, d in zip(gm.head_id, gm.disease)]].copy()
    gm["mi"] = [node[("microbe", h)] for h in gm.head_id]
    gm["di"] = [node[("disease", d)] for d in gm.disease]
    gm["lfc"] = pd.to_numeric(gm.log2fc, errors="coerce").abs()
    gm["det"] = gm.present_case / gm.n_case
    gm["code"] = gm.mi.astype(np.int64) * n + gm.di
    det_of = dict(zip(gm.code, gm.det))
    tested = set(gm.code.tolist())

    rec = pd.read_csv(MKG / "out2/records_v2.tsv", sep="\t", dtype=str, keep_default_na=False,
                      usecols=["head_id", "tail_id", "site_sys", "record_source"])
    rec = rec[~rec.record_source.isin(["GMrepo_v3", "mBodyMap"])]
    gut = set((node[("microbe", h)] * n + node[("disease", d)]) for h, d, s in
              zip(rec.head_id, rec.tail_id, rec.site_sys) if s == "gut" and ("microbe", h) in node and ("disease", d) in node)

    triples, known = {}, {}
    cells = defaultdict(dict)
    pooled = defaultdict(dict)
    diag = []
    for f in sorted(RS.glob("*.npz")):
        m = NAME.match(f.name)
        if not m:
            continue
        s, model, split, seed, dim, negs, loss = m.groups()
        z = np.load(f)
        if "probe_scores" not in z.files:
            continue
        rs = loss.endswith("__rs")
        key = model_key(model, loss.replace("__rs", ""), dim) + ("-rs" if rs else "")
        r = root_of(s)
        if r not in triples:
            x = np.load(r / "triples.npy")
            md = (types[x[:, 0]] == "microbe") & (types[x[:, 2]] == "disease")
            kn = md & np.isin(x[:, 1], REL + [rid["inconsistent_association"]])
            triples[r] = x
            known[r] = set((x[kn, 0].astype(np.int64) * n + x[kn, 2]).tolist())
        x, kn = triples[r], known[r]
        x = x.astype(np.int64) if x.dtype != np.int64 else x
        mic, dis, S = z["probe_microbes"], z["probe_diseases"], z["probe_scores"]
        Smax = S.max(0)
        Sfwd = z["probe_scores_fwd"].max(0) if "probe_scores_fwd" in z.files else None
        pooled_P, pooled_N, det_rho = [], [], []
        mi = np.full(n, -1); mi[mic] = np.arange(len(mic))
        di = np.full(n, -1); di[dis] = np.arange(len(dis))
        test = x[z["ranking_row_ids"]]
        test = test[(di[test[:, 2]] >= 0) & (mi[test[:, 0]] >= 0)]
        sp = np.load(r / "splits" / s / f"seed_{split}.npz")
        tr = x[sp["train"]]
        tr = tr[np.isin(tr[:, 1], REL) & (types[tr[:, 0]] == "microbe") & (types[tr[:, 2]] == "disease")]
        deg = np.bincount(tr[:, 0], minlength=n)
        dbin = np.digitize(np.log2(deg + 1), [1, 2, 3, 4, 5, 6])
        res = defaultdict(dict)
        nulls = {}
        for thr in GRID:
            F, L, K = thr
            g = gm[(gm.fdr > F) & (gm.lfc < L) & (gm.present_total >= K)]
            g = g[(mi[g.mi.to_numpy()] >= 0) & (di[g.di.to_numpy()] >= 0)]
            g = g[[c not in kn for c in g.code]]
            nulls[thr] = g
        g0 = nulls[DEFAULT]
        for d in np.unique(test[:, 2]).tolist():
            j = di[d]
            Pt = test[test[:, 2] == d]
            pcode = np.unique(Pt[:, 0].astype(np.int64) * n + d)
            pm = (pcode // n).astype(int)
            ps = Smax[j, mi[pm]]
            uni = np.array([c in tested for c in pcode])
            gu = np.array([c in gut for c in pcode])
            unk = np.flatnonzero([(int(mm) * n + d) not in kn for mm in mic])
            N = g0[g0.di == d]
            nm = N.mi.to_numpy()
            ns = Smax[j, mi[nm]]
            nkeep = np.array([(c not in tested) or True for c in N.code])
            unk = np.setdiff1d(unk, mi[nm])
            us = Smax[j, unk]
            res["P_vs_allunknown"][int(d)] = auc_parts(ps, us)
            res["N_vs_allunknown"][int(d)] = auc_parts(ns, us)
            res["P_vs_N_within"][int(d)] = auc_parts(ps, ns)
            res["P_vs_N_tested"][int(d)] = auc_parts(ps[uni], ns)
            res["P_vs_N_tested_gut"][int(d)] = auc_parts(ps[uni & gu], ns)
            pdet = np.digitize([det_of.get(c, 0.0) for c in pcode], DET_EDGES)
            ndet = np.digitize(N.det.to_numpy(), DET_EDGES)
            res["P_vs_N_detection_matched"][int(d)] = strat_parts(ps[uni], ns, pdet[uni], ndet)
            res["P_vs_N_degree_matched"][int(d)] = strat_parts(ps, ns, dbin[pm], dbin[nm])
            res["P_vs_N_det_degree_matched"][int(d)] = strat_parts(ps[uni], ns, (pdet * 10 + dbin[pm])[uni],
                                                                   ndet * 10 + dbin[nm])
            res["P_vs_allunknown_degree_matched"][int(d)] = strat_parts(ps, us, dbin[pm], dbin[mic[unk]])
            if Sfwd is not None:
                res["P_vs_N_within_fwd"][int(d)] = auc_parts(Sfwd[j, mi[pm]], Sfwd[j, mi[nm]])
            pooled_P.append(ps); pooled_N.append(ns)
            fam = gm[gm.di == d]
            fam = fam[mi[fam.mi.to_numpy()] >= 0]
            if len(fam) > 20 and np.std(Smax[j, mi[fam.mi.to_numpy()]]) > 0:
                det_rho.append(spearmanr(Smax[j, mi[fam.mi.to_numpy()]], fam.det.to_numpy()).statistic)
            # per relation: positives of relation r scored with r against nulls scored with r
            for k, rname in enumerate(["enriched", "depleted"]):
                pr = np.unique(Pt[Pt[:, 1] == REL[k], 0])
                a = auc_parts(S[k, j, mi[pr]], S[k, j, mi[nm]])
                res[f"P_vs_N_rel_{rname}"][int(d)] = a
                prev = res["P_vs_N_rel_own"].get(int(d), (0.0, 0.0))
                res["P_vs_N_rel_own"][int(d)] = (prev[0] + a[0], prev[1] + a[1])
            for thr in GRID:
                Nt = nulls[thr]
                Nt = Nt[Nt.di == d]
                res[f"grid_{thr[0]}_{thr[1]}_{thr[2]}"][int(d)] = auc_parts(ps, Smax[j, mi[Nt.mi.to_numpy()]])
        cells[(s, key)][int(split)] = {k: v for k, v in res.items()}
        u, w = auc_parts(np.concatenate(pooled_P), np.concatenate(pooled_N))
        pooled[(s, key)][int(split)] = u / w if w else np.nan
        # anomaly diagnostics: does the score track microbe degree?
        rho = [spearmanr(Smax[j], deg[mic]).statistic for j in range(len(dis)) if np.std(Smax[j]) > 0]
        Pm = np.unique(test[:, 0])
        diag.append(dict(setting=s, model=key, seed=int(split), score_degree_spearman=float(np.nanmean(rho)),
                         score_detection_spearman=float(np.nanmean(det_rho)) if det_rho else np.nan,
                         mean_log_deg_P=float(np.log2(deg[Pm] + 1).mean()),
                         mean_log_deg_N=float(np.log2(deg[g0.mi.to_numpy()] + 1).mean()),
                         share_P_tested=float(np.mean([int(a) * n + int(b) in tested for a, b in test[:, [0, 2]]])),
                         n_P=int(len(np.unique(test[:, 0].astype(np.int64) * n + test[:, 2]))), n_N=int(len(g0))))
        print(f.name, "done", flush=True)

    rows = []
    for (s, key), per_seed in cells.items():
        if len(per_seed) < 3:
            continue
        variants = set().union(*[set(v) for v in per_seed.values()])
        for var in sorted(variants):
            seeds = sorted(per_seed)
            mats = []
            for sd in seeds:
                dct = per_seed[sd].get(var, {})
                arr = np.array([v for v in dct.values() if v[1] > 0])
                mats.append(arr if len(arr) else np.zeros((0, 2)))
            if not all(len(a) for a in mats):
                continue
            est = float(np.mean([a[:, 0].sum() / a[:, 1].sum() for a in mats]))
            draws = []
            for _ in range(1000):
                vals = []
                for k in RNG.integers(0, len(mats), len(mats)):
                    a = mats[k]
                    b = a[RNG.integers(0, len(a), len(a))]
                    vals.append(b[:, 0].sum() / b[:, 1].sum() if b[:, 1].sum() > 0 else np.nan)
                draws.append(np.nanmean(vals))
            lo, hi = np.nanquantile(draws, [0.025, 0.975])
            rows.append(dict(setting=s, model=key, variant=var, auroc=est, lo=lo, hi=hi, n_seeds=len(seeds),
                             n_diseases=float(np.mean([len(a) for a in mats])),
                             pairs_compared=float(np.mean([a[:, 1].sum() for a in mats]))))
    for (s, key), v in pooled.items():
        if len(v) >= 3:
            rows.append(dict(setting=s, model=key, variant="P_vs_N_pooled_v2def", auroc=float(np.mean(list(v.values()))),
                             lo=np.nan, hi=np.nan, n_seeds=len(v)))
    pd.DataFrame(rows).to_csv(OUT / "probe_v3.csv", index=False)
    pd.DataFrame(diag).to_csv(OUT / "probe_v3_diagnostics.csv", index=False)
    print("written", len(rows), flush=True)


if __name__ == "__main__":
    main()
