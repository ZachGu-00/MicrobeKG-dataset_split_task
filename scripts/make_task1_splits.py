"""Build Task 1 (Microbe-Disease Association Prediction) splits.

Target: enriched_in + depleted_in (microbe->disease, both directions predicted).
Hard-neg: inconsistent_association (+ sampled negatives at eval, report imbalance).

G_train vs supervision (cross-task principle): the split unit is the
(microbe, disease) PAIR -- NOT the (head,tail,body_site,evidence_type) qualifier
tuple the old version used. All edges of a held-out pair (enriched_in /
depleted_in / inconsistent_association, any body_site/evidence_type) leave
G_train together -> co-location scrub by construction (this is what the old
tuple split leaked: B-pair-overlap in the leakage audit).

G_train = train microbe-disease edges + ontology (microbe taxonomy, disease is_a)
+ co_occurs_with. ALL metabolite/substrate/host_gene bridges are EXCLUDED
(anti-shortcut). The `transductive_with_bridges` control adds the bridges back to
test whether they help or the model is just doing matrix factorization.

Cold settings report by proximity layer; the `none` layer is the headline:
  - cold-microbe: taxonomic proximity of held microbes to train microbes
  - cold-disease: MeSH is_a parent sharing of held diseases with train diseases

Output: splits/task1_microbe_disease/<setting>/seed_<N>/{train,valid,test,test_hardneg}.tsv
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from split_utils import (
    SPLITS_DIR, EDGE_COLS,
    load_final_edges, write_tsv, extract_body_site, split_three_way,
    extract_microbe_lineage, nearest_shared_taxon, nearest_shared_disease_parent,
)

TASK1_DIR = SPLITS_DIR / "task1_microbe_disease"
POS_RELS = ["enriched_in", "depleted_in"]
HARDNEG_REL = "inconsistent_association"
AUX_RELS = [
    "is_strain_of", "is_clade_of",
    "belongs_to_genus", "belongs_to_family", "belongs_to_order",
    "belongs_to_class", "belongs_to_phylum",
    "is_a", "co_occurs_with",
]
TRANSDUCTIVE_SEEDS = [42]
INDUCTIVE_SEEDS = [42]


def filter_scope(edges):
    pos = edges[(edges.head_type == "microbe") & (edges.tail_type == "disease")
                & edges.relation.isin(POS_RELS)].copy().reset_index(drop=True)
    neg = edges[(edges.head_type == "microbe") & (edges.tail_type == "disease")
                & (edges.relation == HARDNEG_REL)].copy().reset_index(drop=True)
    aux = edges[edges.relation.isin(AUX_RELS)].copy().reset_index(drop=True)
    # bridges = everything NOT microbe-disease(pos/neg) and NOT aux (for the control)
    md_mask = (edges.head_type == "microbe") & (edges.tail_type == "disease") \
        & edges.relation.isin(POS_RELS + [HARDNEG_REL])
    bridges = edges[~md_mask & ~edges.relation.isin(AUX_RELS)].copy().reset_index(drop=True)
    return pos, neg, aux, bridges


def _pairs(df):
    return list(df[["head_id", "tail_id"]].drop_duplicates().itertuples(index=False, name=None))


def _by_pairs(df, pairset):
    mask = [(h, t) in pairset for h, t in zip(df.head_id, df.tail_id)]
    return df[mask]


def make_transductive(pos, neg, aux, bridges, with_bridges=False):
    name = "transductive_with_bridges" if with_bridges else "transductive"
    seeds = [42] if with_bridges else TRANSDUCTIVE_SEEDS
    for seed in seeds:
        out = TASK1_DIR / name / f"seed_{seed}"
        tr_p, va_p, te_p = split_three_way(_pairs(pos), (0.8, 0.1, 0.1), seed)
        tr_s, va_s, te_s = set(tr_p), set(va_p), set(te_p)
        train_pos, valid_pos, test_pos = _by_pairs(pos, tr_s), _by_pairs(pos, va_s), _by_pairs(pos, te_s)
        # hard-neg follows the SAME pair partition (co-location): a pair's
        # inconsistent edge goes to whichever fold the pair landed in.
        train_neg = _by_pairs(neg, tr_s)
        test_neg = _by_pairs(neg, te_s | va_s)
        parts = [train_pos[EDGE_COLS], train_neg[EDGE_COLS], aux[EDGE_COLS]]
        if with_bridges:
            parts.append(bridges[EDGE_COLS])
        train = pd.concat(parts, ignore_index=True)
        write_tsv(train, out / "train.tsv")
        write_tsv(valid_pos[EDGE_COLS], out / "valid.tsv")
        write_tsv(test_pos[EDGE_COLS], out / "test.tsv")
        write_tsv(test_neg[EDGE_COLS], out / "test_hardneg.tsv")
        print(f"  {name} seed={seed}: train={len(train):,} valid={len(valid_pos):,} "
              f"test={len(test_pos):,} test_hardneg={len(test_neg):,}")


def make_cold_microbe(pos, neg, aux, lineage):
    pool = sorted(set(pos.head_id))
    print(f"  cold-microbe pool: {len(pool):,} microbes with positive edges")
    for seed in INDUCTIVE_SEEDS:
        out = TASK1_DIR / "cold_microbe" / f"seed_{seed}"
        rng = random.Random(seed)
        sh = list(pool); rng.shuffle(sh)
        n_hold = int(round(len(sh) * 0.10))
        held = set(sh[:n_hold])
        # split held microbes into valid/test halves
        held_list = sorted(held); rng.shuffle(held_list)
        cut = len(held_list) // 2
        valid_m, test_m = set(held_list[:cut]), set(held_list[cut:])
        train_pos = pos[~pos.head_id.isin(held)]
        train_neg = neg[~neg.head_id.isin(held)]
        train = pd.concat([train_pos[EDGE_COLS], train_neg[EDGE_COLS], aux[EDGE_COLS]], ignore_index=True)
        write_tsv(train, out / "train.tsv")
        write_tsv(pos[pos.head_id.isin(valid_m)][EDGE_COLS], out / "valid.tsv")
        write_tsv(pos[pos.head_id.isin(test_m)][EDGE_COLS], out / "test.tsv")
        write_tsv(neg[neg.head_id.isin(test_m)][EDGE_COLS], out / "test_hardneg.tsv")
        # PROBE: taxonomic proximity of test microbes to train microbes (with pos edges)
        prox, bucket = nearest_shared_taxon(test_m, set(train_pos.head_id), lineage)
        write_tsv(prox, out / "test_microbe_tax_proximity.tsv")
        print(f"  cold-microbe seed={seed}: held={len(held)} (valid={len(valid_m)} "
              f"test={len(test_m)}); tax proximity of test: {dict(bucket)}")


def make_cold_disease(pos, neg, aux, is_a_edges):
    parents = set(is_a_edges.head_id)
    pool = sorted(d for d in set(pos.tail_id) if d in parents and d.startswith("D"))
    print(f"  cold-disease pool: {len(pool):,} D-coded diseases with pos edge + MeSH parent")
    for seed in INDUCTIVE_SEEDS:
        out = TASK1_DIR / "cold_disease" / f"seed_{seed}"
        rng = random.Random(seed)
        sh = list(pool); rng.shuffle(sh)
        n_hold = int(round(len(sh) * 0.10))
        held = set(sh[:n_hold])
        held_list = sorted(held); rng.shuffle(held_list)
        cut = len(held_list) // 2
        valid_d, test_d = set(held_list[:cut]), set(held_list[cut:])
        train_pos = pos[~pos.tail_id.isin(held)]
        train_neg = neg[~neg.tail_id.isin(held)]
        train = pd.concat([train_pos[EDGE_COLS], train_neg[EDGE_COLS], aux[EDGE_COLS]], ignore_index=True)
        write_tsv(train, out / "train.tsv")
        write_tsv(pos[pos.tail_id.isin(valid_d)][EDGE_COLS], out / "valid.tsv")
        write_tsv(pos[pos.tail_id.isin(test_d)][EDGE_COLS], out / "test.tsv")
        write_tsv(neg[neg.tail_id.isin(test_d)][EDGE_COLS], out / "test_hardneg.tsv")
        prox, bucket = nearest_shared_disease_parent(test_d, set(train_pos.tail_id), is_a_edges)
        write_tsv(prox, out / "test_disease_ontology_proximity.tsv")
        print(f"  cold-disease seed={seed}: held={len(held)} (valid={len(valid_d)} "
              f"test={len(test_d)}); ontology proximity of test: {dict(bucket)}")


def main():
    print("Loading KG...")
    edges = load_final_edges()
    print(f"  {len(edges):,} edges")
    pos, neg, aux, bridges = filter_scope(edges)
    pos["body_site_uberon"] = pos["evidence"].map(extract_body_site)
    lineage = extract_microbe_lineage(edges)
    is_a_edges = aux[aux.relation == "is_a"]
    print(f"  positives (enriched/depleted microbe->disease): {len(pos):,} "
          f"({pos[['head_id','tail_id']].drop_duplicates().shape[0]:,} unique pairs)")
    print(f"  hard-neg (inconsistent_association): {len(neg):,}")
    print(f"  aux (taxonomy+is_a+co_occurs): {len(aux):,}; bridges (control): {len(bridges):,}")

    print("\n>>> 1A transductive (seed 42, pair-split)")
    make_transductive(pos, neg, aux, bridges, with_bridges=False)
    print("\n>>> 1A control: transductive_with_bridges (seed 42)")
    make_transductive(pos, neg, aux, bridges, with_bridges=True)
    print("\n>>> 1B cold-microbe (seed 42 + tax proximity)")
    make_cold_microbe(pos, neg, aux, lineage)
    print("\n>>> 1C cold-disease (seed 42 + ontology proximity)")
    make_cold_disease(pos, neg, aux, is_a_edges)
    print("\nTask 1 splits written to", TASK1_DIR)


if __name__ == "__main__":
    main()
