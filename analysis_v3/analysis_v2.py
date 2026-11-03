"""Analyses for MicrobeKG-Bench v2 on per-triple ranks of finished cells.

Outputs (results2/analysis/):
  main_table.csv       setting x model: seed mean/sd, hierarchical-bootstrap CI, study-macro MRR, Hits@10
  vs_degree.csv        setting x model: paired difference to the degree prior, hierarchical CI
  decomposition.csv    Pair -> composition-matched Pair -> Study+R -> Study -> Strict, per model
  closure.csv          closure-reachable vs unreachable test pairs, raw and degree-standardised
  cold_sides.csv       cold settings, cold entity as query vs as target
  external.csv         out-of-time set, by side and closure

Hierarchical bootstrap: resample split seeds with replacement, then studies (clusters) within each
sampled seed; the per-seed statistic is a weighted mean over test triples.  The unit of resampling is
the study, so the intervals include between-study and between-seed variation.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/autodl-tmp/MicrobeKG_data")
sys.path.insert(0, str(ROOT / "benchmark"))
from core import load_case  # noqa: E402

B2, BE = Path("/root/autodl-tmp/mkg/bench2"), Path("/root/autodl-tmp/mkg/bench_ext")
RES = [Path("/root/autodl-tmp/mkg/results2/main/main"), Path("/root/autodl-tmp/mkg/results2/gnn/main")]
OUT = Path("/root/autodl-tmp/mkg/results2/analysis")
NAME = re.compile(r"(t1_\w+?)_([a-z0-9-]+)_split(\d+)_seed(\d+)_d(\d+)_n(\d+)_(\w+)\.npz")
NBOOT = 2000
RNG = np.random.default_rng(20261101)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def model_key(model, loss):
    return f"{model}-{loss}" if loss in ("zeroshot", "finetune") else model


def collect():
    cells = {}
    for d in RES:
        if not d.exists():
            continue
        for p in d.glob("*.npz"):
            m = NAME.match(p.name)
            if not m:
                continue
            s, model, split, seed, dim, negs, loss = m.groups()
            js = p.with_suffix(".json")
            if not js.exists() or json.loads(js.read_text()).get("status") not in ("complete", None):
                continue
            cells[(s, model_key(model, loss), int(split))] = p
    return cells


def load_cell(p):
    z = np.load(p)
    sides = [s for s in ("tail", "head") if f"{s}_ranks" in z.files]
    out = dict(rows=z["ranking_row_ids"], tail=1 / z["tail_ranks"], head=1 / z["head_ranks"] if "head_ranks" in z.files else None)
    out["rr"] = np.mean([1 / z[f"{s}_ranks"] for s in sides], axis=0)
    out["h10"] = np.mean([(z[f"{s}_ranks"] <= 10).astype(float) for s in sides], axis=0)
    out["noclosure"] = z["no_closure_2hop_mask"] if "no_closure_2hop_mask" in z.files else None
    return out


def agg_clusters(v, w, c):
    u, inv = np.unique(c, return_inverse=True)
    return np.bincount(inv, weights=w * v), np.bincount(inv, weights=w)


def hboot(per_seed, nboot=NBOOT):
    """per_seed: list of (values, weights, clusters). Returns estimate, (lo, hi)."""
    agg = [agg_clusters(*x) for x in per_seed if len(x[0])]
    if not agg:
        return np.nan, (np.nan, np.nan)
    est = float(np.mean([sv.sum() / sw.sum() for sv, sw in agg]))
    S = len(agg)
    draws = np.empty(nboot)
    for b in range(nboot):
        vals = []
        for s in RNG.integers(0, S, S):
            sv, sw = agg[s]
            k = RNG.integers(0, len(sv), len(sv))
            den = sw[k].sum()
            vals.append(sv[k].sum() / den if den > 0 else np.nan)
        draws[b] = np.nanmean(vals)
    return est, tuple(np.nanquantile(draws, [0.025, 0.975]))


def boot_draws(per_seed, nboot=NBOOT):
    agg = [agg_clusters(*x) for x in per_seed if len(x[0])]
    S = len(agg)
    draws = np.empty(nboot)
    for b in range(nboot):
        vals = []
        for s in RNG.integers(0, S, S):
            sv, sw = agg[s]
            k = RNG.integers(0, len(sv), len(sv))
            vals.append(sv[k].sum() / sw[k].sum())
        draws[b] = np.mean(vals)
    return draws


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells = collect()
    log("cells", len(cells))
    ents = pd.read_csv(B2 / "entities.tsv", sep="\t", dtype=str, keep_default_na=False)
    n = len(ents)
    x = np.load(B2 / "triples.npy", mmap_mode="r")
    xe = np.load(BE / "triples.npy", mmap_mode="r")
    sup = pd.read_csv(B2 / "pair_support.tsv", sep="\t", dtype=str, keep_default_na=False)
    sup["code"] = sup.code.astype(np.int64)
    sup["n_lit"] = sup.n_lit.astype(int)
    sup["has_cohort"] = sup.has_cohort.map({"True": True, "False": False})
    lit_first = sup.lit_studies.str.split(";").str[0]
    coh_first = sup.cohort_studies.str.split(";").str[0]
    sup["cluster"] = np.where(lit_first != "", lit_first, coh_first)
    sup["study_like"] = (sup.n_lit == 1) & ~sup.has_cohort
    litsrc = sup.sources.str.split("|").map(lambda s: sorted(set(s) - {"GMrepo_v3", "mBodyMap"}))
    sup["src_group"] = litsrc.map(lambda s: "none" if not s else (s[0] if len(s) == 1 else "multi"))
    mdeg = sup.groupby("head_id").size(); ddeg = sup.groupby("tail_id").size()
    sup["mbin"] = np.digitize(sup.head_id.map(mdeg), [2, 5, 17])
    sup["dbin"] = np.digitize(sup.tail_id.map(ddeg), [17, 65, 257])
    S = sup.set_index("code")
    ext = pd.read_csv(BE / "external_pairs.tsv", sep="\t", dtype={"code": np.int64})
    ext_pmid = dict(zip(ext.code, ext.pmid.astype(str)))

    def meta_rows(setting, rows):
        xx = xe if setting == "t1_external" else x
        trip = np.asarray(xx[rows])
        code = trip[:, 0].astype(np.int64) * n + trip[:, 2]
        if setting == "t1_external":
            return trip, code, np.array([ext_pmid.get(c, "ext") for c in code]), None
        m = S.loc[code]
        return trip, code, m.cluster.to_numpy(), m

    settings = sorted({k[0] for k in cells})
    models = sorted({k[1] for k in cells})
    data = {}
    for k, p in cells.items():
        d = load_cell(p)
        trip, code, clu, meta = meta_rows(k[0], d["rows"])
        d.update(trip=trip, code=code, cluster=clu, meta=meta)
        data[k] = d
    log("loaded")

    # ------------------------------------------------------------ main table and vs-degree
    rows, vs = [], []
    for s in settings:
        for m in models:
            seeds = sorted(sd for (ss, mm, sd) in data if ss == s and mm == m)
            if len(seeds) < 3:
                continue
            per = [(data[(s, m, sd)]["rr"], np.ones(len(data[(s, m, sd)]["rr"])), data[(s, m, sd)]["cluster"]) for sd in seeds]
            est, ci = hboot(per)
            seedm = [data[(s, m, sd)]["rr"].mean() for sd in seeds]
            macro = []
            for sd in seeds:
                dd = data[(s, m, sd)]
                g = pd.Series(dd["rr"]).groupby(dd["cluster"]).mean()
                macro.append(g.mean())
            h10 = np.mean([data[(s, m, sd)]["h10"].mean() for sd in seeds])
            rows.append(dict(setting=s, model=m, n_seeds=len(seeds), mrr=est, mrr_lo=ci[0], mrr_hi=ci[1],
                             seed_sd=float(np.std(seedm, ddof=1)), study_macro_mrr=float(np.mean(macro)),
                             hits10=float(h10), n_test=float(np.mean([len(data[(s, m, sd)]["rr"]) for sd in seeds]))))
            if m != "popularity":
                both = [sd for sd in seeds if (s, "popularity", sd) in data]
                per = []
                for sd in both:
                    a, b = data[(s, m, sd)], data[(s, "popularity", sd)]
                    assert np.array_equal(a["rows"], b["rows"]), (s, m, sd)
                    per.append((a["rr"] - b["rr"], np.ones(len(a["rr"])), a["cluster"]))
                if len(per) >= 3:
                    est, ci = hboot(per)
                    vs.append(dict(setting=s, model=m, n_seeds=len(per), delta=est, lo=ci[0], hi=ci[1],
                                   excludes_zero=bool(ci[0] > 0 or ci[1] < 0)))
    pd.DataFrame(rows).to_csv(OUT / "main_table.csv", index=False)
    pd.DataFrame(vs).to_csv(OUT / "vs_degree.csv", index=False)
    log("main table done")

    # ------------------------------------------------------------ composition decomposition
    study_test = pd.concat([data[k]["meta"] for k in data if k[0] == "t1_study" and k[1] == "popularity"])
    strata = ["src_group", "mbin", "dbin"]
    target = study_test.groupby(strata).size() / len(study_test)
    dec = []
    for m in models:
        need = ["t1_pair", "t1_study", "t1_study_reveal", "t1_study_strict"]
        if not all(any((s, m, sd) in data for sd in range(10**5)) for s in need[:1]):
            pass
        res = {}
        # Pair: all, matched, matched+reweighted
        P = [(sd, data[("t1_pair", m, sd)]) for (ss, mm, sd) in data if ss == "t1_pair" and mm == m]
        if len(P) < 3:
            continue
        pm = pd.concat([d["meta"][d["meta"].study_like] for _, d in P])
        src = pm.groupby(strata).size() / len(pm)
        wmap = (target / src).dropna()
        cover = float(target[target.index.isin(wmap.index)].sum())
        per_all, per_match, per_rw = [], [], []
        for sd, d in P:
            like = d["meta"].study_like.to_numpy()
            key = list(zip(*(d["meta"][c].to_numpy() for c in strata)))
            w = np.array([wmap.get(kk, 0.0) for kk in key]) * like
            per_all.append((d["rr"], np.ones(len(d["rr"])), d["cluster"]))
            per_match.append((d["rr"][like], np.ones(like.sum()), d["cluster"][like]))
            per_rw.append((d["rr"], w, d["cluster"]))
        res["pair_all"] = hboot(per_all)
        res["pair_matched"] = hboot(per_match)
        res["pair_matched_rw"] = hboot(per_rw)
        res["pair_share_study_like"] = (float(np.mean([d["meta"].study_like.mean() for _, d in P])), (np.nan, np.nan))
        draws = {"pair_all": boot_draws(per_all), "pair_matched_rw": boot_draws(per_rw)}
        for s, tag in [("t1_study_reveal", "study_reveal"), ("t1_study", "study"), ("t1_study_strict", "study_strict")]:
            Q = [data[(s, m, sd)] for (ss, mm, sd) in data if ss == s and mm == m]
            if len(Q) >= 3:
                per = [(q["rr"], np.ones(len(q["rr"])), q["cluster"]) for q in Q]
                res[tag] = hboot(per)
                draws[tag] = boot_draws(per)
        # paired differences on shared test rows
        for a_, b_, tag in [("t1_study_reveal", "t1_study", "sibling_effect"), ("t1_study", "t1_study_strict", "cohort_backdoor")]:
            per = []
            for (ss, mm, sd) in list(data):
                if ss == a_ and mm == m and (b_, m, sd) in data:
                    A, Bq = data[(a_, m, sd)], data[(b_, m, sd)]
                    assert np.array_equal(A["rows"], Bq["rows"])
                    per.append((A["rr"] - Bq["rr"], np.ones(len(A["rr"])), A["cluster"]))
            if len(per) >= 3:
                res[tag] = hboot(per)
        if "study" in draws:
            dd = draws["pair_all"] - draws["pair_matched_rw"]
            res["composition_effect"] = (float(res["pair_all"][0] - res["pair_matched_rw"][0]), tuple(np.quantile(dd, [.025, .975])))
            dd = draws["pair_matched_rw"] - draws["study"]
            res["holdout_effect"] = (float(res["pair_matched_rw"][0] - res["study"][0]), tuple(np.quantile(dd, [.025, .975])))
            tot = res["pair_all"][0] - res["study"][0]
            res["share_composition"] = (float(res["composition_effect"][0] / tot) if tot else np.nan, (np.nan, np.nan))
        for k, (e, ci) in res.items():
            dec.append(dict(model=m, quantity=k, estimate=e, lo=ci[0], hi=ci[1], strata_coverage=cover))
    pd.DataFrame(dec).to_csv(OUT / "decomposition.csv", index=False)
    log("decomposition done")

    # ------------------------------------------------------------ closure, degree-standardised
    clo = []
    deg_cache = {}
    for s in ["t1_pair", "t1_study", "t1_study_strict", "t1_temporal"]:
        seeds = sorted({sd for (ss, mm, sd) in data if ss == s})
        for sd in seeds:
            case = load_case(B2, s, sd)
            types = case["entities"].node_type.to_numpy()
            tr = case["train"]
            rel = [case["relations"].index(r) for r in ["enriched_in", "depleted_in"]]
            md = tr[np.isin(tr[:, 1], rel)]
            dm = np.bincount(md[:, 0], minlength=n); dd_ = np.bincount(md[:, 2], minlength=n)
            deg_cache[(s, sd)] = (dm, dd_)
        for m in models:
            ks = [(s, m, sd) for sd in seeds if (s, m, sd) in data]
            if len(ks) < 3:
                continue
            out = defaultdict(list)
            for k in ks:
                d = data[k]
                if d["noclosure"] is None:
                    continue
                dm, dd_ = deg_cache[(k[0], k[2])]
                b = np.digitize(np.log2(dm[d["trip"][:, 0]] + 1), [1, 2, 3, 5]) * 10 + np.digitize(np.log2(dd_[d["trip"][:, 2]] + 1), [5, 7, 9])
                nc = d["noclosure"].astype(bool)
                # standardise the no-closure subset to the bin distribution of the closure subset
                p_c = pd.Series(b[~nc]).value_counts(normalize=True)
                p_n = pd.Series(b[nc]).value_counts(normalize=True)
                w_std = np.array([p_c.get(v, 0) / p_n.get(v, np.inf) for v in b]) * nc
                one = np.ones(len(b))
                base = data.get((k[0], "popularity", k[2]))
                out["closure"].append((d["rr"], one * ~nc, d["cluster"]))
                out["noclosure"].append((d["rr"], one * nc, d["cluster"]))
                out["noclosure_std"].append((d["rr"], w_std, d["cluster"]))
                out["share_noclosure"].append((nc.astype(float), one, d["cluster"]))
                if base is not None and m != "popularity":
                    diff = d["rr"] - base["rr"]
                    out["vsdeg_closure"].append((diff, one * ~nc, d["cluster"]))
                    out["vsdeg_noclosure"].append((diff, one * nc, d["cluster"]))
                    out["vsdeg_noclosure_std"].append((diff, w_std, d["cluster"]))
            for q, per in out.items():
                e, ci = hboot(per)
                clo.append(dict(setting=s, model=m, quantity=q, estimate=e, lo=ci[0], hi=ci[1]))
    pd.DataFrame(clo).to_csv(OUT / "closure.csv", index=False)
    log("closure done")

    # ------------------------------------------------------------ cold sides
    cold = []
    for s, query_side in [("t1_cold_microbe", "tail"), ("t1_cold_disease", "head")]:
        for m in models:
            ks = [k for k in data if k[0] == s and k[1] == m]
            if len(ks) < 3:
                continue
            target_side = "head" if query_side == "tail" else "tail"
            for label, side in [("cold_entity_is_query", query_side), ("cold_entity_is_target", target_side)]:
                per = [(data[k][side], np.ones(len(data[k][side])), data[k]["cluster"]) for k in ks]
                e, ci = hboot(per)
                cold.append(dict(setting=s, model=m, side=label, mrr=e, lo=ci[0], hi=ci[1]))
    pd.DataFrame(cold).to_csv(OUT / "cold_sides.csv", index=False)

    # ------------------------------------------------------------ external
    exr = []
    for m in models:
        ks = [k for k in data if k[0] == "t1_external" and k[1] == m]
        if len(ks) < 3:
            continue
        for label, sel in [("all", None), ("closure", False), ("noclosure", True)]:
            for side in ["rr", "tail", "head"]:
                per = []
                for k in ks:
                    d = data[k]
                    w = np.ones(len(d["rr"]))
                    if sel is not None and d["noclosure"] is not None:
                        w = w * (d["noclosure"].astype(bool) == sel)
                    per.append((d[side], w, d["cluster"]))
                e, ci = hboot(per)
                exr.append(dict(model=m, subset=label, side=side, mrr=e, lo=ci[0], hi=ci[1]))
        base = [k for k in ks if ("t1_external", "popularity", k[2]) in data]
        if m != "popularity" and len(base) >= 3:
            per = [(data[k]["rr"] - data[("t1_external", "popularity", k[2])]["rr"], np.ones(len(data[k]["rr"])), data[k]["cluster"]) for k in base]
            e, ci = hboot(per)
            exr.append(dict(model=m, subset="vs_degree", side="rr", mrr=e, lo=ci[0], hi=ci[1]))
    pd.DataFrame(exr).to_csv(OUT / "external.csv", index=False)
    log("all done")


if __name__ == "__main__":
    main()
