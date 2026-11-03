"""MicrobeKG-Bench v3 training-graph variants.  Every variant keeps the parent's validation rows, test rows, candidate
pools and filters (removed training pairs become filter-only), so differences are paired on identical test rows.

  t1_v3_strict_nocohort   Strict minus target pairs without training-literature support (cohort-only pairs from
                          GMrepo or mBodyMap) and minus all co_occurs_with edges (all GMrepo-derived)        [P0-1]
  t1_v3_strict_nooverlap  Strict minus pairs whose only support is a GMrepo contrast that contains case runs from a
                          BioProject of a held-out study (PMID->BioProject map, results3/indep)            [P0-1]
  t1_v3_strict_nocooc     Strict minus co_occurs_with edges                                                  [P0-2]
  t1_v3_temporal_nocooc   Temporal minus co_occurs_with edges                                                [P0-2]
  t1_v3_temporal_dated    Temporal where cohort support counts only if its data are dated <= 2021: a GMrepo
                          contrast is dated by its latest case-run project year; mBodyMap groups have no project
                          dates and count as undated                                                         [P1-3]
  t1_v3_p11_base          Study-exclusive minus a reserve of other-study pairs matched to the revealed pairs R
  t1_v3_p11_same          p11_base + R' (revealed pairs of the held-out studies that found a reserve match)  [P1-1]
  t1_v3_p11_other         p11_base + reserve (= the Study-exclusive training graph)                          [P1-1]
                          All three arms share one candidate set (entities seen in p11_base) and one test set.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/autodl-tmp/MicrobeKG_data")
B2, OUT = Path("/root/autodl-tmp/mkg/bench2"), Path("/root/autodl-tmp/mkg/bench3")
IND = Path("/root/autodl-tmp/mkg/indep")
sys.path.insert(0, str(ROOT / "benchmark"))
from prepare import partition, save_split, sha256, write_json  # noqa: E402

SEEDS = [42, 123, 456, 789, 2024]
SCORING = ["enriched_in", "depleted_in"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ["triples.npy", "entities.tsv", "relations.json"]:
        if not (OUT / f).exists():
            os.symlink(B2 / f, OUT / f)
    x = np.load(B2 / "triples.npy")
    ents = pd.read_csv(B2 / "entities.tsv", sep="\t", dtype=str, keep_default_na=False)
    rels = json.loads((B2 / "relations.json").read_text())
    types = ents.node_type.to_numpy()
    n = len(types)
    rid = {r: i for i, r in enumerate(rels)}
    cooc = rid["co_occurs_with"]
    is_target = np.isin(x[:, 1], [rid[r] for r in SCORING]) & (types[x[:, 0]] == "microbe") & (types[x[:, 2]] == "disease")
    pair = x[:, 0].astype(np.int64) * n + x[:, 2]

    sup = pd.read_csv(B2 / "pair_support.tsv", sep="\t", dtype=str, keep_default_na=False)
    sup["code"] = sup.code.astype(np.int64)
    lit = {c: set(v.split(";")) - {""} for c, v in zip(sup.code, sup.lit_studies)}
    coh = {c: set(v.split(";")) - {""} for c, v in zip(sup.code, sup.cohort_studies)}
    year = {c: int(y) if y.isdigit() else None for c, y in zip(sup.code, sup.first_year)}
    n_lit = dict(zip(sup.code, sup.n_lit.astype(int)))
    studies = np.array(sorted(set().union(*lit.values())))

    cy = pd.read_csv(IND / "gmrepo_contrast_years.tsv", sep="\t")
    contrast_year = {f"GMREPO:{d}": int(y) for d, y in zip(cy.disease, cy.latest_case_project_year) if pd.notna(y)}
    ov = pd.read_csv(IND / "overlap_by_seed.csv")
    ov = ov[ov.tier == "all_sources"].set_index("seed")
    affected = {int(s): set(f"GMREPO:{d}" for d in str(v).split(";") if d and d != "nan")
                for s, v in ov.affected_case_contrasts_ids.items()}

    load = lambda s, sd: dict(np.load(B2 / "splits" / s / f"seed_{sd}.npz"))
    metas, audit = [], {}

    def save(setting, seed, parent, train, known_extra, notes):
        meta = save_split(OUT, setting, seed, x, types, rels, train, parent["valid"], parent["test"], SCORING,
                          conflict=parent["conflict"], known_extra=known_extra,
                          eligible_nodes=parent["candidate_eligible"])
        new = np.load(OUT / "splits" / setting / f"seed_{seed}.npz")
        same_rows = bool(np.array_equal(new["test"][new["test_eligible"]], parent["test"][parent["test_eligible"]]))
        same_pool = bool(np.array_equal(new["candidate_eligible"], parent["candidate_eligible"]))
        notes.update(train_triples=int(len(np.unique(train))), same_test_rows=same_rows, same_candidate_flags=same_pool,
                     test_evaluable=int(new["test_eligible"].sum()))
        audit.setdefault(setting, {})[seed] = notes
        metas.append(meta)
        print(setting, seed, notes, flush=True)

    for seed in SEEDS:
        st, rv, sc, tp = (load(s, seed) for s in ["t1_study", "t1_study_reveal", "t1_study_strict", "t1_temporal"])
        s_tr = set(partition(studies, seed)[0].tolist())

        # ---------------- V1: Strict without cohort sources
        T = sc["train"]
        tgt = T[is_target[T]]
        drop = tgt[[not lit.get(int(c)) for c in pair[tgt]]]
        cooc_rows = T[x[T, 1] == cooc]
        train = np.setdiff1d(T, np.r_[drop, cooc_rows])
        save("t1_v3_strict_nocohort", seed, sc, train, np.union1d(sc["known_extra"], drop),
             dict(removed_target_triples=int(len(drop)), removed_target_pairs=int(len(np.unique(pair[drop]))),
                  removed_cooc=int(len(cooc_rows))))

        # ---------------- V3: Strict without GMrepo contrasts that share a BioProject with a held-out study
        aff = affected.get(seed, set())
        drop = tgt[[(not lit.get(int(c))) and bool(coh.get(int(c))) and coh[int(c)] <= aff for c in pair[tgt]]]
        save("t1_v3_strict_nooverlap", seed, sc, np.setdiff1d(T, drop), np.union1d(sc["known_extra"], drop),
             dict(affected_contrasts=sorted(aff), removed_target_triples=int(len(drop)),
                  removed_target_pairs=int(len(np.unique(pair[drop])))))

        # ---------------- V2: no co-occurrence edges (Strict, Temporal)
        save("t1_v3_strict_nocooc", seed, sc, np.setdiff1d(T, cooc_rows), sc["known_extra"],
             dict(removed_cooc=int(len(cooc_rows))))
        TT = tp["train"]
        cooc_t = TT[x[TT, 1] == cooc]
        save("t1_v3_temporal_nocooc", seed, tp, np.setdiff1d(TT, cooc_t), tp["known_extra"],
             dict(removed_cooc=int(len(cooc_t))))

        # ---------------- V4: Temporal with dated cohort support
        tgt_t = TT[is_target[TT]]

        def dated_ok(c):
            c = int(c)
            if lit.get(c) and (year.get(c) is None or year[c] <= 2021):
                return True
            return any(contrast_year.get(k, 9999) <= 2021 for k in coh.get(c, ()) if k.startswith("GMREPO:"))

        drop = tgt_t[[not dated_ok(c) for c in pair[tgt_t]]]
        save("t1_v3_temporal_dated", seed, tp, np.setdiff1d(TT, drop), np.union1d(tp["known_extra"], drop),
             dict(removed_target_triples=int(len(drop)), removed_target_pairs=int(len(np.unique(pair[drop])))))

        # ---------------- V5: same-study vs other-study pairs (P1-1)
        S = st["train"]
        s_tgt = S[is_target[S]]
        R_rows = np.setdiff1d(rv["train"], S)
        assert is_target[R_rows].all()
        R_pairs = np.unique(pair[R_rows])
        mbin = np.floor(np.log2(np.bincount(np.unique(pair[s_tgt]) // n, minlength=n) + 1)).astype(int)
        pool = defaultdict(list)
        for c in np.unique(pair[s_tgt]).tolist():
            L = lit.get(c, set())
            if not coh.get(c) and n_lit.get(c) == 1 and L and L <= s_tr:
                pool[(c % n, mbin[c // n])].append(c)
        rng = np.random.default_rng(seed)
        for k in pool:
            pool[k] = list(rng.permutation(pool[k]))
        matched, reserve, exact = [], [], 0
        for c in rng.permutation(R_pairs).tolist():
            d, b = c % n, mbin[c // n]
            for off in (0, -1, 1, -2, 2):
                cand = pool.get((d, b + off))
                if cand:
                    reserve.append(int(cand.pop()))
                    matched.append(c)
                    exact += off == 0
                    break
        matched, reserve = np.array(sorted(matched)), np.array(sorted(reserve))
        Rp_rows = R_rows[np.isin(pair[R_rows], matched)]
        Res_rows = s_tgt[np.isin(pair[s_tgt], reserve)]
        base_train = np.setdiff1d(S, Res_rows)
        notes = dict(revealed_pairs=int(len(R_pairs)), matched_pairs=int(len(matched)), exact_bin=int(exact),
                     revealed_triples_matched=int(len(Rp_rows)), reserve_triples=int(len(Res_rows)),
                     diseases=int(len(np.unique(matched % n))))
        # one candidate set and one set of evaluable test rows for all three arms: entities seen in the base graph
        seen_base = np.zeros(n, bool)
        seen_base[x[base_train][:, [0, 2]].ravel()] = True
        arm = dict(st, candidate_eligible=st["candidate_eligible"] & seen_base)
        save("t1_v3_p11_base", seed, arm, base_train, np.union1d(st["known_extra"], Res_rows), dict(notes))
        save("t1_v3_p11_same", seed, arm, np.union1d(base_train, Rp_rows),
             np.union1d(np.setdiff1d(st["known_extra"], Rp_rows), Res_rows), dict(notes))
        save("t1_v3_p11_other", seed, arm, S, st["known_extra"], dict(notes))
        for k in ["t1_v3_p11_base", "t1_v3_p11_same", "t1_v3_p11_other"]:
            audit[k][seed]["same_as_base"] = bool(np.array_equal(
                np.load(OUT / "splits" / k / f"seed_{seed}.npz")["candidate_eligible"], arm["candidate_eligible"]))
        np.savez_compressed(OUT / f"p11_pairs_seed{seed}.npz", revealed=R_pairs, matched=matched, reserve=reserve)

    m2 = json.loads((B2 / "manifest.json").read_text())
    write_json(OUT / "manifest.json", dict(protocol="mkbench-v3-variants", parent="bench2",
                                           original_edges_sha256=m2["original_edges_sha256"], seeds=SEEDS,
                                           settings=sorted(audit), splits=metas,
                                           files={str(p.relative_to(OUT)): sha256(p) for p in sorted(OUT.rglob("*"))
                                                  if p.is_file() and not p.is_symlink() and p.name != "manifest.json"}))
    write_json(OUT / "build_audit.json", audit)


if __name__ == "__main__":
    main()
