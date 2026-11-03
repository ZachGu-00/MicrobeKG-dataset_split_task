"""Feature-based pair scorer (P1-4): LightGBM on taxonomy, ontology, degree and AGORA2 metabolic-trait features.

For a candidate (m, r, d) and the training target graph P_r (microbe x disease, relation r):
  tax_L_same / tax_L_other  log1p of the number of training pairs (m', r, d) / (m', r', d) with m' != m in the same
                            taxon as m at level L (strain->species, genus, family, order, class, phylum)
  onto_same / onto_other    log1p of training pairs (m, r, d') / (m, r', d') with d' an is_a neighbour of d (1-2 hops)
  deg_m_same/other, deg_d_same/other   log1p training degrees (leave-one-out for training positives)
  metab_cos                 cosine between m's AGORA2 profile (SVD of can_utilize + produces) and the mean profile of
                            d's training microbes for r (leave-one-out for training positives); has_metab flag
  rel                       relation indicator
Training rows: training positives and four corrupted negatives each (two random diseases, two random microbes,
excluding known targets).  Default (crossfit): positives' features come from the training graph without their own
evidence cluster (5 study-grouped folds), as at test time; insample: features from the full training graph with
leave-one-out degrees.  Early stopping on 10% of training pairs.  CPU only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, "/root/autodl-tmp/mkg/gnn")
sys.path.insert(0, "/root/autodl-tmp/mkg")
import common  # noqa: E402
from common import evaluate_and_save, load  # noqa: E402

V3 = ["t1_v3_strict_nocohort", "t1_v3_strict_nocooc", "t1_v3_temporal_nocooc", "t1_v3_temporal_dated"]
common.SETTINGS.extend(s for s in V3 if s not in common.SETTINGS)
LEVELS = ["is_strain_of", "belongs_to_genus", "belongs_to_family", "belongs_to_order", "belongs_to_class",
          "belongs_to_phylum"]
FULL_X = {}


def full_triples(setting):
    r = common.data_root(None, setting)
    if r not in FULL_X:
        FULL_X[r] = np.load(r / "triples.npy", mmap_mode="r")
    return FULL_X[r]


class Features:
    def __init__(self, case, k_svd=32):
        t0 = time.perf_counter()
        ents = case["entities"]
        types = ents.node_type.to_numpy()
        n = len(types)
        rels = case["relations"]
        self.R = [rels.index("enriched_in"), rels.index("depleted_in")]
        tr = case["train"]
        mics = np.flatnonzero((types == "microbe") & case["eval_seen"])
        diss = np.flatnonzero((types == "disease") & case["eval_seen"])
        self.mics, self.diss = mics, diss
        self.mi = np.full(n, -1); self.mi[mics] = np.arange(len(mics))
        self.di = np.full(n, -1); self.di[diss] = np.arange(len(diss))
        allm = np.flatnonzero(types == "microbe")
        am = np.full(n, -1); am[allm] = np.arange(len(allm))
        ad = np.full(n, -1); ad[np.flatnonzero(types == "disease")] = np.arange((types == "disease").sum())
        nd = int((types == "disease").sum())
        P = []
        for r in self.R:
            z = tr[(tr[:, 1] == r) & (types[tr[:, 0]] == "microbe") & (types[tr[:, 2]] == "disease")]
            P.append(sp.csr_matrix((np.ones(len(z), np.float32), (am[z[:, 0]], ad[z[:, 2]])), shape=(len(allm), nd)))
        P = [(p > 0).astype(np.float32) for p in P]
        rows_m = am[mics]; cols_d = ad[diss]
        self.P = [p[rows_m][:, cols_d].toarray() for p in P]
        F = {}
        for lev in LEVELS:
            z = tr[tr[:, 1] == rels.index(lev)]
            z = z[(am[z[:, 0]] >= 0) & (am[z[:, 2]] >= 0)]
            M = sp.csr_matrix((np.ones(len(z), np.float32), (am[z[:, 0]], am[z[:, 2]])), shape=(len(allm), len(allm)))
            M = (M > 0).astype(np.float32)
            diag = np.asarray(M.multiply(M).sum(1)).ravel()[rows_m]
            for k, p in enumerate(P):
                pc = p[:, cols_d]
                E = (M[rows_m] @ (M.T @ pc)).toarray() - diag[:, None] * self.P[k]
                F[(lev, k)] = np.log1p(np.maximum(E, 0)).astype(np.float32)
                F[(lev + "_parent", k)] = np.log1p((M[rows_m] @ pc).toarray()).astype(np.float32)
                F[(lev + "_child", k)] = np.log1p((M.T.tocsr()[rows_m] @ pc).toarray()).astype(np.float32)
        z = tr[tr[:, 1] == rels.index("is_a")]
        z = z[(ad[z[:, 0]] >= 0) & (ad[z[:, 2]] >= 0)]
        O = sp.csr_matrix((np.ones(len(z), np.float32), (ad[z[:, 0]], ad[z[:, 2]])), shape=(nd, nd))
        O = ((O + O.T) > 0).astype(np.float32)
        O2 = ((O + O @ O) > 0).astype(np.float32)
        O2.setdiag(0)
        O2.eliminate_zeros()
        O2 = O2[cols_d][:, cols_d]
        for k, p in enumerate(P):
            F[("onto", k)] = np.log1p((p[rows_m][:, cols_d] @ O2).toarray()).astype(np.float32) if O2.nnz else \
                np.zeros((len(mics), len(diss)), np.float32)
        tot = np.bincount(tr[:, [0, 2]].ravel(), minlength=n).astype(np.float32)   # degree prior's degree
        self.totm, self.totd = np.log1p(tot[mics]), np.log1p(tot[diss])
        self.degm = [p.sum(1).astype(np.float32) for p in self.P]     # over eval-seen diseases
        self.degd = [p.sum(0).astype(np.float32) for p in self.P]
        # metabolic traits from the full graph (non-target AGORA2 edges)
        x = np.asarray(full_triples(case["meta"]["setting"]))
        mt = x[np.isin(x[:, 1], [rels.index("can_utilize"), rels.index("produces")])]
        mt = mt[am[mt[:, 0]] >= 0]
        if len(mt):
            cols = np.unique(mt[:, 2]); ci = np.searchsorted(cols, mt[:, 2])
            A = sp.csr_matrix((np.ones(len(mt), np.float32), (am[mt[:, 0]], ci)), shape=(len(allm), len(cols)))
            A = (A > 0).astype(np.float32)
            from sklearn.decomposition import TruncatedSVD
            Z = TruncatedSVD(k_svd, random_state=0).fit_transform(A)
            nz = np.linalg.norm(Z, axis=1)
            Z = Z / np.maximum(nz, 1e-9)[:, None]
            self.has = (nz > 0)[rows_m].astype(np.float32)
            self.Z = Z[rows_m].astype(np.float32)
        else:
            self.has = np.zeros(len(mics), np.float32); self.Z = np.zeros((len(mics), k_svd), np.float32)
        self.prof = [self.P[k].T @ self.Z for k in range(2)]           # disease x k (sum of member profiles)
        self.F = F
        print(f"features built {time.perf_counter() - t0:.0f}s, {len(mics)} microbes x {len(diss)} diseases", flush=True)

    def names(self):
        out = []
        for lev in LEVELS + ["onto"]:
            out += [f"{lev}_same", f"{lev}_other"]
        for lev in LEVELS:
            out += [f"{lev}_parent_same", f"{lev}_child_same"]
        return out + ["degm_same", "degm_other", "degd_same", "degd_other", "metab_cos", "has_metab", "rel",
                      "log_totdeg_m", "log_totdeg_d", "log_degree_prior"]

    def rows(self, im, idd, k, loo):
        """features for index arrays im (microbe rows), idd (disease cols), relation k; loo: 1 for training positives."""
        o = 1 - k
        cols = []
        for lev in LEVELS + ["onto"]:
            cols += [self.F[(lev, k)][im, idd], self.F[(lev, o)][im, idd]]
        for lev in LEVELS:
            cols += [self.F[(lev + "_parent", k)][im, idd], self.F[(lev + "_child", k)][im, idd]]
        cols += [np.log1p(self.degm[k][im] - loo), np.log1p(self.degm[o][im]),
                 np.log1p(self.degd[k][idd] - loo), np.log1p(self.degd[o][idd])]
        prof = self.prof[k][idd] - loo[:, None] * self.Z[im]
        pn = np.linalg.norm(prof, axis=1)
        cols += [np.einsum("ij,ij->i", self.Z[im], prof) / np.maximum(pn, 1e-9), self.has[im],
                 np.full(len(im), float(k), np.float32)]
        tm = np.log(np.maximum(np.expm1(self.totm[im]) - loo, 0) + 1)
        td = np.log(np.maximum(np.expm1(self.totd[idd]) - loo, 0) + 1)
        cols += [tm, td, tm + td]
        return np.column_stack(cols).astype(np.float32)


class GBMScorer:
    def __init__(self, feats, S):
        self.f, self.S = feats, S                                   # S[k]: microbes x diseases

    def _rel(self, r):
        return self.f.R.index(int(r))

    def candidates(self, queries, relations, pool, side):
        out = np.empty((len(queries), len(pool)), np.float32)
        for i, (q, r) in enumerate(zip(queries, relations)):
            k = self._rel(r)
            if side == "tail":
                out[i] = self.S[k][self.f.mi[q], self.f.di[pool]]
            else:
                out[i] = self.S[k][self.f.mi[pool], self.f.di[q]]
        return out

    def triples(self, rows):
        rows = np.asarray(rows)
        out = np.empty(len(rows), np.float32)
        for k, r in enumerate(self.f.R):
            m = rows[:, 1] == r
            out[m] = self.S[k][self.f.mi[rows[m, 0]], self.f.di[rows[m, 2]]]
        m = ~np.isin(rows[:, 1], self.f.R)
        if m.any():
            out[m] = np.maximum(self.S[0], self.S[1])[self.f.mi[rows[m, 0]], self.f.di[rows[m, 2]]]
        return out


def training_rows(case, f_eval, a, rng):
    """Training matrix.  insample: features of training positives from the full training graph (leave-one-out
    degrees).  crossfit: training target pairs are split into 5 folds by evidence cluster (first literature study,
    else cohort group); features of fold-k positives and their corrupted negatives come from the training graph
    without fold k, which mimics scoring pairs from a new study."""
    import pandas as pd
    nm, nd = len(f_eval.mics), len(f_eval.diss)
    known = f_eval.P[0] + f_eval.P[1] > 0
    X, y, pid = [], [], []

    def add(f, im, idd, k, loo_pos):
        X.append(f.rows(im, idd, k, loo_pos * np.ones(len(im), np.float32))); y.append(np.ones(len(im)))
        pid.append(im * nd + idd)
        for _ in range(2):
            d2 = rng.integers(0, nd, len(im)); ok = ~known[im, d2]
            X.append(f.rows(im[ok], d2[ok], k, np.zeros(ok.sum(), np.float32))); y.append(np.zeros(ok.sum()))
            pid.append(im[ok] * nd + d2[ok])
            m2 = rng.integers(0, nm, len(im)); ok = ~known[m2, idd]
            X.append(f.rows(m2[ok], idd[ok], k, np.zeros(ok.sum(), np.float32))); y.append(np.zeros(ok.sum()))
            pid.append(m2[ok] * nd + idd[ok])

    if a.features == "insample":
        for k in range(2):
            im, idd = np.nonzero(f_eval.P[k])
            add(f_eval, im, idd, k, 1.0)
        return np.vstack(X), np.concatenate(y), np.concatenate(pid)
    sup = pd.read_csv("/root/autodl-tmp/mkg/bench2/pair_support.tsv", sep="\t", dtype=str, keep_default_na=False)
    l1, c1 = sup.lit_studies.str.split(";").str[0], sup.cohort_studies.str.split(";").str[0]
    clus = dict(zip(sup.code.astype(np.int64), np.where(l1 != "", l1, c1)))
    tr = case["train"]; n = len(case["entities"])
    tgt = np.isin(tr[:, 1], f_eval.R)
    code = tr[:, 0].astype(np.int64) * n + tr[:, 2]
    uc = sorted({clus.get(c, "none") for c in code[tgt].tolist()})
    fold_of = dict(zip(uc, rng.permutation(len(uc)) % 5))
    fold = np.array([fold_of[clus.get(c, "none")] if t else -1 for c, t in zip(code.tolist(), tgt)])
    for k5 in range(5):
        sub = dict(case); sub["train"] = tr[fold != k5]
        f = Features(sub)
        held = tr[fold == k5]
        for k, r in enumerate(f_eval.R):
            h = held[held[:, 1] == r]
            im, idd = f_eval.mi[h[:, 0]], f_eval.di[h[:, 2]]
            ok = (im >= 0) & (idd >= 0)
            add(f, im[ok], idd[ok], k, 0.0)
        del f
    return np.vstack(X), np.concatenate(y), np.concatenate(pid)


def main(a):
    import lightgbm as lgb
    started = time.perf_counter()
    case = load(None, a.setting, a.split_seed)
    f = Features(case)
    rng = np.random.default_rng(a.seed)
    nm, nd = len(f.mics), len(f.diss)
    X, y, pid = training_rows(case, f, a, rng)
    val = np.isin(pid, rng.permutation(np.unique(pid))[: len(np.unique(pid)) // 10])
    model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=a.seed,
                               n_jobs=a.threads, verbose=-1)
    model.fit(X[~val], y[~val], eval_set=[(X[val], y[val])], callbacks=[lgb.early_stopping(50, verbose=False)])
    print("trained", model.best_iteration_, f"{time.perf_counter() - started:.0f}s", flush=True)
    # score only what evaluation and probes read: full rows of query microbes, full columns of query diseases
    import pandas as pd
    import rescore_v3 as R
    x = np.asarray(case["triples"]); arr = case["arrays"]
    te = x[arr["test"][arr["test_eligible"]]]
    cf = x[arr["conflict"][arr["conflict_eligible"]]] if len(arr["conflict"]) else np.zeros((0, 3), int)
    gm = pd.read_csv(R.GM, sep="\t", usecols=["disease"]).disease.unique()
    node = {v: i for i, v in enumerate(case["entities"].node_id)}
    types = case["entities"].node_type.to_numpy()
    probe_d = [node[d] for d in gm if d in node and types[node[d]] == "disease"]
    Mq = np.unique(f.mi[np.r_[te[:, 0], cf[:, 0]]]); Mq = Mq[Mq >= 0]
    Dq = np.unique(f.di[np.r_[te[:, 2], cf[:, 2], probe_d]]); Dq = Dq[Dq >= 0]
    S = []
    for k in range(2):
        Sk = np.zeros((nm, nd), np.float32)
        for im_b, id_b in [(np.repeat(Mq, nd), np.tile(np.arange(nd), len(Mq))),
                           (np.repeat(np.arange(nm), len(Dq)), np.tile(Dq, nm))]:
            for st in range(0, len(im_b), 4_000_000):
                sl = slice(st, min(st + 4_000_000, len(im_b)))
                Xi = f.rows(im_b[sl], id_b[sl], k, np.zeros(sl.stop - sl.start, np.float32))
                Sk[im_b[sl], id_b[sl]] = model.predict(Xi, raw_score=True, num_iteration=model.best_iteration_,
                                                       num_threads=a.threads)
        S.append(Sk)
    print("scored", len(Mq), "query microbes,", len(Dq), "query diseases", f"{time.perf_counter() - started:.0f}s",
          flush=True)
    scorer = GBMScorer(f, S)
    imp = dict(zip(f.names(), model.booster_.feature_importance("gain").round(1).tolist()))
    fit = dict(best_iteration=int(model.best_iteration_), feature_importance_gain=imp, n_train_rows=int((~val).sum()),
               runner_elapsed_sec=time.perf_counter() - started)
    method = "GBM" if a.features == "crossfit" else "GBM-insample"
    cfg = vars(a).copy(); cfg.update(model=method, loss="pairfeat")
    evaluate_and_save(case, scorer, method, 0, 4, "pairfeat", a.out, cfg, fit, dict(device="cpu"), 1000, 64, False)
    # rescore outputs (pools + probe matrix) in the format of rescore_v3.py
    n = len(types); rid = f.R
    md = np.isin(x[:, 1], rid) & (types[x[:, 0]] == "microbe") & (types[x[:, 2]] == "disease")
    trmd = case["train"][np.isin(case["train"][:, 1], rid)]
    pools = {"all": np.ones(n, bool), "target_all": R.target_mask(x[md], types, n),
             "target_train": R.target_mask(trmd, types, n)}
    res = R.ranks_pools(case, scorer, pools)
    mic, dis, PS = R.probe_matrix(case, scorer, probe_d, "head")
    stem = f"{a.setting}_{method.lower()}_split{a.split_seed}_seed{a.seed}_d0_n4_pairfeat"
    R.OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(R.OUT / f"{stem}.npz", **res, probe_microbes=mic, probe_diseases=dis, probe_scores=PS)
    print("saved", stem, f"{time.perf_counter() - started:.0f}s", json.dumps(imp)[:300], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--setting", required=True)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=24)
    p.add_argument("--features", choices=["crossfit", "insample"], default="crossfit")
    p.add_argument("--out", default="/root/autodl-tmp/mkg/results3/gnn")
    main(p.parse_args())
