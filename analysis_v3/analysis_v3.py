"""MicrobeKG-Bench v3 analyses (P0-3, P0-4, P1-1, P1-2, P1-3).

Statistics are fixed before the v3 runs:
  * hierarchical bootstrap, 2,000 draws: resample split seeds, then evidence clusters within each sampled seed
    (an independent cluster resample for every sampled seed);
  * paired differences to the degree prior on identical test rows, for MRR, Hits@10 and study-macro MRR;
  * labels: better / worse when the 95% interval excludes 0; equivalent when the 90% interval lies inside +-delta
    (delta = 0.005 MRR, 0.010 Hits@10, 0.005 macro MRR); otherwise inconclusive.
Outputs: results3/analysis/*.csv
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

MKG = Path("/root/autodl-tmp/mkg")
OUT = MKG / "results3/analysis"
NBOOT = 2000
RNG = np.random.default_rng(20261102)
DELTA = {"mrr": 0.005, "h10": 0.010, "macro": 0.005}
QS = [0.025, 0.05, 0.95, 0.975]
NAME = re.compile(r"(t1_\w+?)_([a-z0-9-]+)_split(\d+)_seed(\d+)_d(\d+)_n(\d+)_(\w+)\.npz")
DIRS = [(MKG / "results2/main/main", ""), (MKG / "results2/gnn/main", ""), (MKG / "results3/variants/main", ""),
        (MKG / "results3/gnn", ""), (MKG / "results3/hpo_random/final", "-rs")]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def root_of(setting):
    if setting == "t1_external":
        return MKG / "bench_ext"
    return MKG / ("bench3" if (MKG / "bench3/splits" / setting).exists() else "bench2")


def collect():
    cells = {}
    for d, tag in DIRS:
        if not d.exists():
            continue
        for p in d.glob("**/*.npz"):
            m = NAME.match(p.name)
            if not m or not p.with_suffix(".json").exists():
                continue
            s, model, split, seed, dim, negs, loss = m.groups()
            key = (f"{model}-{loss}" if loss in ("zeroshot", "finetune") else model) + tag
            st = json.loads(p.with_suffix(".json").read_text()).get("status")
            if st not in ("complete", None):
                continue
            cells[(s, key, int(split))] = p
    return cells


# ------------------------------------------------------------------------------------------------ bootstrap
def cluster_sums(v, w, c):
    u, inv = np.unique(c, return_inverse=True)
    return np.bincount(inv, weights=w * v), np.bincount(inv, weights=w)


def hboot(per_seed, macro=False, nboot=NBOOT):
    """per_seed: list of (values, weights, clusters). Returns estimate and quantiles QS."""
    agg = []
    for v, w, c in per_seed:
        ok = np.isfinite(v) & (w > 0)
        if ok.any():
            sv, sw = cluster_sums(v[ok], w[ok], c[ok])
            agg.append((sv, sw))
    if not agg:
        return np.nan, [np.nan] * len(QS)

    def stat(sv, sw):
        return np.mean(sv[sw > 0] / sw[sw > 0]) if macro else sv.sum() / sw.sum()

    est = float(np.mean([stat(sv, sw) for sv, sw in agg]))
    S = len(agg)
    per = []
    for sv, sw in agg:
        k = RNG.integers(0, len(sv), (nboot, S, len(sv)))
        num, den = sv[k], sw[k]
        if macro:
            r = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
            per.append(np.nanmean(r, axis=2))
        else:
            per.append(num.sum(2) / den.sum(2))
    per = np.stack(per)                                   # seeds x nboot x S
    pick = RNG.integers(0, S, (nboot, S))
    draws = np.nanmean(per[pick, np.arange(nboot)[:, None], np.arange(S)[None, :]], axis=1)
    return est, list(np.nanquantile(draws, QS))


def label(est, q, delta):
    lo95, lo90, hi90, hi95 = q
    if lo95 > 0:
        lab = "better"
    elif hi95 < 0:
        lab = "worse"
    elif lo90 > -delta and hi90 < delta:
        lab = "equivalent"
    else:
        lab = "inconclusive"
    return lab, bool(lo90 > -delta and hi90 < delta)


# ------------------------------------------------------------------------------------------------ data
class Data:
    def __init__(self):
        self.ents = pd.read_csv(MKG / "bench2/entities.tsv", sep="\t", dtype=str, keep_default_na=False)
        self.n = len(self.ents)
        sup = pd.read_csv(MKG / "bench2/pair_support.tsv", sep="\t", dtype=str, keep_default_na=False)
        sup["code"] = sup.code.astype(np.int64)
        lit1 = sup.lit_studies.str.split(";").str[0]
        coh1 = sup.cohort_studies.str.split(";").str[0]
        sup["cluster"] = np.where(lit1 != "", lit1, coh1)
        sup["n_lit"] = sup.n_lit.astype(int)
        self.S = sup.set_index("code")
        ext = pd.read_csv(MKG / "bench_ext/external_pairs.tsv", sep="\t", dtype={"code": np.int64, "pmid": str})
        ext["study"] = ext.bsdb.astype(str).str.split("/").str[0]      # BugSigDB study id (PMID- or DOI-keyed)
        g = ext.groupby("code")
        self.ext_study = g.study.first().to_dict()
        self.ext_pmid = g.pmid.first().to_dict()
        self.ext_nstudy = g.study.nunique().to_dict()
        self.x = {}

    def triples(self, setting):
        r = root_of(setting)
        if r not in self.x:
            self.x[r] = np.load(r / "triples.npy", mmap_mode="r")
        return self.x[r]

    def rows_meta(self, setting, rows):
        t = np.asarray(self.triples(setting)[rows])
        code = t[:, 0].astype(np.int64) * self.n + t[:, 2]
        if setting == "t1_external":
            return code, np.array([self.ext_study.get(c, "ext") for c in code])
        return code, self.S.loc[code].cluster.to_numpy()


def load_ranks(p, pool=None):
    if pool is None or pool == "all":
        z = np.load(p)
        t, h = z["tail_ranks"], z["head_ranks"]
        rows = z["ranking_row_ids"]
    else:
        z = np.load(MKG / "results3/rescore" / (p.stem + ("__rs" if "hpo_random" in str(p) else "") + ".npz"))
        t, h = z[f"{pool}_tail_ranks"], z[f"{pool}_head_ranks"]
        rows = z["ranking_row_ids"]
    rr = np.nanmean(np.c_[1 / t, 1 / h], axis=1)
    h10 = np.nanmean(np.c_[(t <= 10) + 0.0 * t, (h <= 10) + 0.0 * h], axis=1)
    h10[np.isnan(t) & np.isnan(h)] = np.nan
    return rows, rr, h10


# ------------------------------------------------------------------------------------------------ tables
def metric_table(D, cells, settings, models, pool="all", tag=""):
    main_rows, vs_rows = [], []
    cache = {}
    for s in settings:
        for m in models + ["popularity"]:
            for sd in sorted(sd for (ss, mm, sd) in cells if ss == s and mm == m):
                p = cells[(s, m, sd)]
                if pool != "all" and not (MKG / "results3/rescore" /
                                          (p.stem + ("__rs" if "hpo_random" in str(p) else "") + ".npz")).exists():
                    continue
                rows, rr, h10 = load_ranks(p, pool)
                code, clu = D.rows_meta(s, rows)
                cache[(s, m, sd)] = (rows, rr, h10, clu)
    for s in settings:
        for m in sorted({mm for (ss, mm, _) in cache if ss == s}):
            seeds = sorted(sd for (ss, mm, sd) in cache if ss == s and mm == m)
            if len(seeds) < 3:
                continue
            res = dict(setting=s, model=m, pool=pool, n_seeds=len(seeds),
                       n_test=float(np.mean([np.isfinite(cache[(s, m, sd)][1]).sum() for sd in seeds])))
            for met, j, macro in [("mrr", 1, False), ("h10", 2, False), ("macro", 1, True)]:
                per = [(cache[(s, m, sd)][j], np.ones(len(cache[(s, m, sd)][j])), cache[(s, m, sd)][3]) for sd in seeds]
                est, q = hboot(per, macro)
                res.update({met: est, f"{met}_lo": q[0], f"{met}_hi": q[3]})
            main_rows.append(res)
            if m == "popularity":
                continue
            both = [sd for sd in seeds if (s, "popularity", sd) in cache]
            if len(both) < 3:
                continue
            for sd in both:
                assert np.array_equal(cache[(s, m, sd)][0], cache[(s, "popularity", sd)][0]), (s, m, sd)
            r = dict(setting=s, model=m, pool=pool, n_seeds=len(both))
            for met, j, macro in [("mrr", 1, False), ("h10", 2, False), ("macro", 1, True)]:
                per = [(cache[(s, m, sd)][j] - cache[(s, "popularity", sd)][j], np.ones(len(cache[(s, m, sd)][j])),
                        cache[(s, m, sd)][3]) for sd in both]
                est, q = hboot(per, macro)
                lab, eq = label(est, q, DELTA[met])
                r.update({f"{met}_delta": est, f"{met}_lo95": q[0], f"{met}_lo90": q[1], f"{met}_hi90": q[2],
                          f"{met}_hi95": q[3], f"{met}_label": lab, f"{met}_within_delta": eq,
                          f"{met}_mde80": 2.8 * (q[3] - q[0]) / (2 * 1.96)})
            vs_rows.append(r)
    pd.DataFrame(main_rows).to_csv(OUT / f"main_table{tag}.csv", index=False)
    pd.DataFrame(vs_rows).to_csv(OUT / f"vs_degree{tag}.csv", index=False)
    log("written", tag, len(main_rows), len(vs_rows))
    return cache


def subset_table(D, cache, setting, subsets, tag):
    """subsets: name -> function(code array, cluster array) -> bool mask over test rows."""
    rows = []
    models = sorted({m for (s, m, _) in cache if s == setting})
    for name, f in subsets.items():
        for m in models:
            seeds = sorted(sd for (s, mm, sd) in cache if s == setting and mm == m)
            if len(seeds) < 3:
                continue
            per, perd = [], []
            for sd in seeds:
                r, rr, h10, clu = cache[(setting, m, sd)]
                code, _ = D.rows_meta(setting, r)
                w = (f(code, clu)).astype(float)
                per.append((rr, w, clu))
                if m != "popularity" and (setting, "popularity", sd) in cache:
                    perd.append((rr - cache[(setting, "popularity", sd)][1], w, clu))
            est, q = hboot(per)
            row = dict(setting=setting, subset=name, model=m, n_test=float(np.mean([p[1].sum() for p in per])),
                       mrr=est, mrr_lo=q[0], mrr_hi=q[3])
            if perd:
                e2, q2 = hboot(perd)
                lab, eq = label(e2, q2, DELTA["mrr"])
                row.update(delta=e2, lo95=q2[0], lo90=q2[1], hi90=q2[2], hi95=q2[3], label=lab,
                           mde80=2.8 * (q2[3] - q2[0]) / (2 * 1.96))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"subsets_{tag}.csv", index=False)
    return df


# ------------------------------------------------------------------------------------------------ P1-1
def p11(D, cache):
    rows, perm = [], []
    arms = ["t1_v3_p11_base", "t1_v3_p11_same", "t1_v3_p11_other"]
    models = sorted({m for (s, m, _) in cache if s == "t1_v3_p11_base"})
    for m in models:
        seeds = [sd for sd in [42, 123, 456, 789, 2024] if all((a, m, sd) in cache for a in arms)]
        if len(seeds) < 3:
            continue
        for sd in seeds:
            for a in arms[1:]:
                assert np.array_equal(cache[(a, m, sd)][0], cache[(arms[0], m, sd)][0])
        for name, a, b in [("same_minus_base", arms[1], arms[0]), ("other_minus_base", arms[2], arms[0]),
                           ("same_minus_other", arms[1], arms[2])]:
            per = [(cache[(a, m, sd)][1] - cache[(b, m, sd)][1], np.ones(len(cache[(a, m, sd)][1])),
                    cache[(a, m, sd)][3]) for sd in seeds]
            est, q = hboot(per)
            rows.append(dict(model=m, contrast=name, n_seeds=len(seeds), delta=est, lo95=q[0], hi95=q[3]))
        # study-label permutation: is the same-minus-other gain larger in studies with more revealed pairs added?
        stud = defaultdict(list)
        for sd in seeds:
            pp = np.load(MKG / f"bench3/p11_pairs_seed{sd}.npz")
            k = pd.Series([D.S.loc[c].lit_studies.split(";")[0] for c in pp["matched"]]).value_counts()
            g = cache[(arms[1], m, sd)][1] - cache[(arms[2], m, sd)][1]
            clu = cache[(arms[1], m, sd)][3]
            for c, v in pd.Series(g).groupby(clu).mean().items():
                stud["seed"].append(sd); stud["study"].append(c); stud["gain"].append(v)
                stud["k"].append(int(k.get(c, 0)))
        st = pd.DataFrame(stud)
        from scipy.stats import spearmanr
        rho = spearmanr(st.k, st.gain).statistic
        null = []
        for _ in range(10000):
            ks = st.groupby("seed").k.transform(lambda v: RNG.permutation(v.to_numpy()))
            null.append(spearmanr(ks, st.gain).statistic)
        null = np.asarray(null)
        hi = st.k > 0
        perm.append(dict(model=m, n_studies=len(st), rho=rho, p_perm=float((np.sum(null >= rho) + 1) / (len(null) + 1)),
                         gain_k_pos=float(st.gain[hi].mean()), gain_k0=float(st.gain[~hi].mean()),
                         n_k_pos=int(hi.sum()), n_k0=int((~hi).sum())))
    pd.DataFrame(rows).to_csv(OUT / "p11_contrasts.csv", index=False)
    pd.DataFrame(perm).to_csv(OUT / "p11_permutation.csv", index=False)


# ------------------------------------------------------------------------------------------------ variants vs parent
VPARENT = {"t1_v3_strict_nocohort": "t1_study_strict", "t1_v3_strict_nooverlap": "t1_study_strict",
           "t1_v3_strict_nocooc": "t1_study_strict", "t1_v3_temporal_nocooc": "t1_temporal",
           "t1_v3_temporal_dated": "t1_temporal"}


def variant_vs_parent(D, cache):
    """Same model, variant training graph minus parent training graph, on the test rows both rank (a variant can
    lose a few rows whose entities are no longer seen).  Also the change of the learned-minus-prior gap."""
    rows = []
    for v, par in VPARENT.items():
        for m in sorted({mm for (s, mm, _) in cache if s == v}):
            seeds = [sd for sd in [42, 123, 456, 789, 2024] if (v, m, sd) in cache and (par, m, sd) in cache]
            if len(seeds) < 3:
                continue
            per, gap = [], []
            for sd in seeds:
                a, b = cache[(v, m, sd)], cache[(par, m, sd)]
                common, ia, ib = np.intersect1d(a[0], b[0], return_indices=True)
                per.append((a[1][ia] - b[1][ib], np.ones(len(common)), a[3][ia]))
                if m != "popularity" and (v, "popularity", sd) in cache and (par, "popularity", sd) in cache:
                    pa, pb = cache[(v, "popularity", sd)], cache[(par, "popularity", sd)]
                    ja = pd.Index(pa[0]).get_indexer(common); jb = pd.Index(pb[0]).get_indexer(common)
                    assert (ja >= 0).all() and (jb >= 0).all()
                    gap.append(((a[1][ia] - pa[1][ja]) - (b[1][ib] - pb[1][jb]), np.ones(len(common)), a[3][ia]))
            est, q = hboot(per)
            r = dict(variant=v, parent=par, model=m, n_seeds=len(seeds),
                     n_common=float(np.mean([len(p[0]) for p in per])), delta=est, lo95=q[0], hi95=q[3])
            if gap:
                e2, q2 = hboot(gap)
                r.update(gap_change=e2, gap_lo95=q2[0], gap_hi95=q2[3])
            rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "variant_vs_parent.csv", index=False)


# ------------------------------------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = Data()
    cells = collect()
    log("cells", len(cells))
    models = sorted({m for (_, m, _) in cells if m not in ("popularity",)})
    base_settings = ["t1_pair", "t1_study_reveal", "t1_study", "t1_study_strict", "t1_temporal", "t1_temporal_nodis",
                     "t1_external", "t1_cold_microbe", "t1_cold_disease", "t1_study_ctx_open"]
    v3 = sorted({s for (s, _, _) in cells if s.startswith("t1_v3")})
    cache = metric_table(D, cells, base_settings + v3, models)
    for pool in ["target_all", "target_train"]:
        metric_table(D, cells, ["t1_pair", "t1_study", "t1_study_strict", "t1_temporal", "t1_external"], models,
                     pool=pool, tag=f"_{pool}")

    # P1-3 out-of-time publication date subsets
    dates = pd.read_csv(MKG / "indep/external_pmid_dates.tsv", sep="\t", dtype={"pmid": str})
    after = set(dates[dates.after_cutoff.astype(str) == "True"].pmid)
    y25 = set(dates[pd.to_numeric(dates.earliest_year, errors="coerce") >= 2025].pmid)
    ext_pairs = pd.read_csv(MKG / "bench_ext/external_pairs.tsv", sep="\t", dtype={"code": np.int64, "pmid": str})
    late_doi = set(ext_pairs[ext_pairs.pmid == "10"].code)          # two DOI-only studies, dated 2026-09/10
    pm = D.ext_pmid
    subset_table(D, cache, "t1_external", {
        "all": lambda c, k: np.ones(len(c), bool),
        "published_after_snapshot": lambda c, k: np.array([pm.get(v) in after or v in late_doi for v in c]),
        "published_2025_or_later": lambda c, k: np.array([pm.get(v) in y25 or v in late_doi for v in c]),
        "reported_by_2plus_new_studies": lambda c, k: np.array([D.ext_nstudy.get(v, 1) >= 2 for v in c]),
    }, "external")

    # P1-2 learnability strata on Strict / Study-exclusive / Temporal
    nl = D.S.n_lit.to_dict()
    rr = {}
    rp = MKG / "results3/analysis/rereported_pairs.tsv"
    if rp.exists():
        t = pd.read_csv(rp, sep="\t")
        rr = dict(zip(t.code, t.n_new_studies))
    for s in ["t1_study_strict", "t1_study", "t1_temporal"]:
        subset_table(D, cache, s, {
            "all": lambda c, k: np.ones(len(c), bool),
            "single_study": lambda c, k: np.array([nl.get(v, 0) == 1 for v in c]),
            "replicated_2plus_studies": lambda c, k: np.array([nl.get(v, 0) >= 2 for v in c]),
            "rereported_after_snapshot": lambda c, k: np.array([rr.get(v, 0) >= 1 for v in c]),
        }, f"learnability_{s}")
    p11(D, cache)
    variant_vs_parent(D, cache)
    log("done")


if __name__ == "__main__":
    main()
