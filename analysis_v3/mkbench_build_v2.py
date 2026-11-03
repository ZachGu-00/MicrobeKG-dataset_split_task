"""MicrobeKG-Bench v2: canonical study identities, five split seeds and paired split families.

Families (all in minimal-v1 format, readable by benchmark/run.py):
  t1_pair                  pair-level 80/10/10 (the conventional protocol)
  t1_study                 literature studies 80/10/10; test = half (T) of the pairs supported only by
                           held-out studies; the other half (R) is filter-only
  t1_study_reveal          t1_study + R in training                       (same test rows and candidates)
  t1_study_strict          t1_study minus every training pair that shares a literature study with the
                           held-out (validation or test) studies           (same test rows and candidates)
  t1_study_ctx_open        t1_study + metabolic/chemical/host context, CTD- and HMDB-only edges excluded
  t1_study_ctx_full        t1_study + all context                          (same test rows and candidates)
  t1_temporal              first literature report <=2021 train, 2022 valid, >=2023 test
  t1_temporal_nodis        same rule with Disbiome removed (sensitivity)
  t1_cold_microbe/disease  10% test-cold and 10% validation-cold endpoints with >=5 pairs

Cohort pseudo-studies (GMrepo phenotype, mBodyMap site x disease) always stay in training except in
t1_study_strict, where a cohort-supported pair is still dropped if a held-out study also reports it.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/autodl-tmp/MicrobeKG_data")
SRC = Path("/root/autodl-tmp/mkg/bench")
OUT = Path("/root/autodl-tmp/mkg/bench2")
REC = Path("/root/autodl-tmp/mkg/out2/records_v2.tsv")
PUB = Path("/root/autodl-tmp/mkg/out2/pubmed_meta.tsv")
GM = ROOT / ("paper_work_collaboration/overleaf/microbekg_p0024_overleaf/review/"
             "20260928_corrected/claude/release/gmrepo/work/family_tests.tsv.gz")
sys.path.insert(0, str(ROOT / "benchmark"))
from prepare import TAX, pair_code, partition, save_split, sha256, write_json  # noqa: E402

SEEDS = [42, 123, 456, 789, 2024]
SCORING = ["enriched_in", "depleted_in"]
LIT = {"BugSigDB", "Disbiome", "Peryton", "GMMAD", "Lit44", "gutMDisorder_v2", "HMDAD", "Piccinno_2025"}
COHORT = {"GMrepo_v3", "mBodyMap"}
RESTRICTED = {"CTD", "HMDB"}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ["triples.npy", "entities.tsv", "relations.json"]:
        shutil.copy(SRC / f, OUT / f)
    x = np.load(OUT / "triples.npy")
    ents = pd.read_csv(OUT / "entities.tsv", sep="\t", dtype=str, keep_default_na=False)
    rels = json.loads((OUT / "relations.json").read_text())
    types = ents.node_type.to_numpy()
    n, R = len(ents), len(rels)
    rid = {r: i for i, r in enumerate(rels)}
    idx = {(t, i): j for j, (t, i) in enumerate(zip(ents.node_type, ents.node_id))}
    relation = lambda names: np.isin(x[:, 1], [rid[r] for r in names])

    # ---------------- per-triple source labels (for the licence-aware context variants)
    code_of = lambda h, r, t: (h.astype(np.int64) * R + r) * n + t
    xcode = code_of(x[:, 0], x[:, 1], x[:, 2])
    assert np.all(np.diff(xcode) > 0)
    open_ok = np.zeros(len(x), bool)
    cols = ["head_id", "head_type", "relation", "tail_id", "tail_type", "source"]
    for df in pd.read_csv(ROOT / "kg_build_audited/final_kg_edges.tsv", sep="\t", usecols=cols,
                          dtype=str, keep_default_na=False, chunksize=500000):
        h = np.fromiter((idx[(a, b)] for a, b in zip(df.head_type, df.head_id)), np.int64, len(df))
        t = np.fromiter((idx[(a, b)] for a, b in zip(df.tail_type, df.tail_id)), np.int64, len(df))
        r = np.fromiter((rid[v] for v in df.relation), np.int64, len(df))
        ok = np.fromiter((bool(set(s.split("|")) - RESTRICTED) for s in df.source), bool, len(df))
        pos_ = np.searchsorted(xcode, code_of(h, r, t))
        open_ok[pos_[ok]] = True
    md = (types[x[:, 0]] == "microbe") & (types[x[:, 2]] == "disease")
    pos = np.flatnonzero(md & relation(SCORING))
    conflict = np.flatnonzero(md & relation(["inconsistent_association"]))
    aux = np.flatnonzero(relation(TAX + ["is_a", "co_occurs_with"]))
    context = np.flatnonzero(~md & ~relation(TAX + ["is_a", "co_occurs_with"]))
    ctx_open, ctx_full = context[open_ok[context]], context
    log("targets", len(pos), "aux", len(aux), "context open/full", len(ctx_open), len(ctx_full))
    pair = pair_code(x, n)

    # ---------------- study support per pair
    rec = pd.read_csv(REC, sep="\t", dtype=str, keep_default_na=False)
    pm = pd.read_csv(PUB, sep="\t", dtype=str, keep_default_na=False)
    pm_year = {p: int(y) for p, y in zip(pm.pmid, pm.year) if str(y).isdigit()}

    def year_of(row):
        if row.pmid and row.pmid in pm_year:
            return pm_year[row.pmid]
        y = str(row.year).split(".")[0]
        return int(y) if y.isdigit() else None

    rec["y"] = [year_of(r) for r in rec.itertuples(index=False)]
    rec["code"] = [idx[("microbe", h)] * n + idx[("disease", t)] for h, t in zip(rec.head_id, rec.tail_id)]
    sup = defaultdict(lambda: dict(lit=set(), lit_nodis=set(), coh=set(), y=None, y_nodis=None, srcs=set()))
    for r in rec.itertuples(index=False):
        s = sup[r.code]
        s["srcs"].add(r.record_source)
        if r.record_source in COHORT:
            s["coh"].add(r.study_id)
        elif r.record_source in LIT and r.study_id:
            s["lit"].add(r.study_id)
            if r.y is not None:
                s["y"] = r.y if s["y"] is None else min(s["y"], r.y)
            if r.record_source != "Disbiome":
                s["lit_nodis"].add(r.study_id)
                if r.y is not None:
                    s["y_nodis"] = r.y if s["y_nodis"] is None else min(s["y_nodis"], r.y)
    pos_pairs = np.unique(pair[pos])
    missing = [c for c in pos_pairs.tolist() if c not in sup]
    assert not missing, ("target pairs without a record", len(missing))
    support = {int(c): sup[int(c)] for c in pos_pairs}
    studies = np.array(sorted(set().union(*(s["lit"] for s in support.values()))))
    log("pairs", len(support), "literature studies", len(studies),
        "pairs w/o year", sum(1 for s in support.values() if s["lit"] and s["y"] is None))

    names = ents.node_id.to_numpy()
    pd.DataFrame([dict(code=c, head_id=names[c // n], tail_id=names[c % n],
                       n_lit=len(s["lit"]), has_cohort=bool(s["coh"]), first_year=s["y"] or "",
                       first_year_nodis=s["y_nodis"] or "", sources="|".join(sorted(s["srcs"])),
                       lit_studies=";".join(sorted(s["lit"])), cohort_studies=";".join(sorted(s["coh"])))
                  for c, s in sorted(support.items())]).to_csv(OUT / "pair_support.tsv", sep="\t", index=False)

    triples_of = lambda codes: pos[np.isin(pair[pos], np.asarray(sorted(codes), dtype=np.int64))]
    seen_of = lambda rows: np.bincount(x[rows][:, [0, 2]].ravel(), minlength=n) > 0
    metas, audit = [], dict(pairs=len(support), studies=len(studies), seeds={})

    def save(setting, seed, train, valid, test, **kw):
        cf = conflict[np.isin(pair[conflict], np.unique(pair[test]))]
        meta = save_split(OUT, setting, seed, x, types, rels, train, valid, test, SCORING, conflict=cf, **kw)
        metas.append(meta)
        return meta

    for seed in SEEDS:
        a = audit["seeds"][str(seed)] = {}
        # ---------- pair
        trp, vap, tep = partition(pair[pos], seed)
        tr, va, te = [pos[np.isin(pair[pos], p)] for p in (trp, vap, tep)]
        save("t1_pair", seed, np.r_[tr, aux], va, te)
        a["pair"] = dict(train=len(trp), valid=len(vap), test=len(tep))

        # ---------- study family
        s_tr, s_va, s_te = (set(v.tolist()) for v in partition(studies, seed))
        held = s_va | s_te
        train, valid, pool = [], [], []
        for c, s in support.items():
            lit, coh = s["lit"], s["coh"]
            if coh or (lit & s_tr) or not lit:
                train.append(c)
            elif lit & s_te:
                pool.append(c)
            else:
                valid.append(c)
        pool.sort()
        reveal = np.random.default_rng(seed).random(len(pool)) < 0.5
        R_ = [c for c, r in zip(pool, reveal) if r]
        T_ = [c for c, r in zip(pool, reveal) if not r]
        strict = [c for c in train if not (support[c]["lit"] & held)]
        dropped = sorted(set(train) - set(strict))
        tr_s, tr_strict, va_s, te_s, rv_s, dr_s = (triples_of(v) for v in (train, strict, valid, T_, R_, dropped))
        base, base_strict = np.r_[tr_s, aux], np.r_[tr_strict, aux]
        elig = seen_of(base_strict)
        save("t1_study", seed, base, va_s, te_s, known_extra=rv_s, eligible_nodes=elig)
        save("t1_study_reveal", seed, np.r_[base, rv_s], va_s, te_s, eligible_nodes=elig)
        save("t1_study_strict", seed, base_strict, va_s, te_s, known_extra=np.r_[rv_s, dr_s], eligible_nodes=elig)
        save("t1_study_ctx_open", seed, np.r_[base, ctx_open], va_s, te_s, known_extra=rv_s, eligible_nodes=elig)
        save("t1_study_ctx_full", seed, np.r_[base, ctx_full], va_s, te_s, known_extra=rv_s, eligible_nodes=elig)
        a["study"] = dict(studies=[len(s_tr), len(s_va), len(s_te)], train=len(train), strict_train=len(strict),
                          dropped_by_strict=len(dropped), valid=len(valid), test=len(T_), revealed=len(R_),
                          test_single_study=float(np.mean([len(support[c]["lit"]) == 1 for c in T_])))

        # ---------- temporal (deterministic; the seed changes model initialisation only)
        for setting, key, litkey in [("t1_temporal", "y", "lit"), ("t1_temporal_nodis", "y_nodis", "lit_nodis")]:
            tr_c, va_c, te_c, gone = [], [], [], []
            for c, s in support.items():
                y, lit, coh = s[key], s[litkey], s["coh"]
                if not lit and not coh:
                    gone.append(c)           # supported only by Disbiome in the no-Disbiome variant
                elif coh or y is None or y <= 2021:
                    tr_c.append(c)
                elif y == 2022:
                    va_c.append(c)
                else:
                    te_c.append(c)
            base_t = np.r_[triples_of(tr_c), aux]
            save(setting, seed, base_t, triples_of(va_c), triples_of(te_c),
                 known_extra=triples_of(gone) if gone else None)
            a[setting] = dict(train=len(tr_c), valid=len(va_c), test=len(te_c), removed=len(gone))

        # ---------- cold endpoints with a cold validation set
        for setting, col in [("t1_cold_microbe", 0), ("t1_cold_disease", 2)]:
            ends = pd.Series([c // n if col == 0 else c % n for c in support])
            deg = ends.value_counts()
            elig_nodes = np.sort(deg[deg >= 5].index.to_numpy())
            perm = np.random.default_rng(seed).permutation(elig_nodes)
            k = int(round(0.1 * len(perm)))
            te_n, va_n = set(perm[:k].tolist()), set(perm[k:2 * k].tolist())
            end = x[pos][:, col]
            te = pos[np.isin(end, list(te_n))]
            va = pos[np.isin(end, list(va_n))]
            tr = pos[~np.isin(end, list(te_n | va_n))]
            save(setting, seed, np.r_[tr, aux], va, te)
            a[setting] = dict(cold_test=len(te_n), cold_valid=len(va_n), train=len(tr), valid=len(va), test=len(te))
        log(seed, json.dumps(a))

    # ---------------- GMrepo tested-null pairs (hard negatives)
    g = pd.read_csv(GM, sep="\t")
    g = g[g.disease.isin(set(ents[ents.node_type == "disease"].node_id)) &
          g.head_id.isin(set(ents[ents.node_type == "microbe"].node_id))]
    g["lfc"] = pd.to_numeric(g.log2fc, errors="coerce")
    null = g[(g.fdr > 0.2) & (g.lfc.abs() < 0.5) & (pd.to_numeric(g.present_total, errors="coerce") >= 20)]
    assoc = set(pair[np.flatnonzero(md & relation(SCORING + ["inconsistent_association"]))].tolist())
    null = null.assign(mi=[idx[("microbe", h)] for h in null.head_id], di=[idx[("disease", d)] for d in null.disease])
    null = null[[(m * n + d) not in assoc for m, d in zip(null.mi, null.di)]]
    null[["mi", "di", "head_id", "disease", "lfc", "fdr", "present_total"]].to_csv(
        OUT / "tested_null.tsv", sep="\t", index=False)
    audit["tested_null"] = dict(pairs=len(null), diseases=int(null.di.nunique()), microbes=int(null.mi.nunique()))
    log("tested-null negatives", audit["tested_null"])

    write_json(OUT / "build_audit.json", audit)
    manifest = dict(protocol="mkbench-v2", graph="kg_build_audited", seeds=SEEDS,
                    settings=sorted({m["setting"] for m in metas}),
                    original_edges_sha256=sha256(ROOT / "kg_build_audited/final_kg_edges.tsv"),
                    original_nodes_sha256=sha256(ROOT / "kg_build_audited/final_kg_nodes.tsv"),
                    files={str(p.relative_to(OUT)): sha256(p) for p in sorted(OUT.rglob("*"))
                           if p.is_file() and p.name != "manifest.json"},
                    splits=metas)
    write_json(OUT / "manifest.json", manifest)
    log("done", len(metas), "split files")


if __name__ == "__main__":
    main()
