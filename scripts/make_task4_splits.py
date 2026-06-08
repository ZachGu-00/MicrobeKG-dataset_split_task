"""Build Task 4 (Metabolite->Disease Therapeutic Prediction via host-gene mechanism).

Target: treats_disease (metabolite->disease, CTD curated). Secondary: assoc(met).

G_train vs supervision separation (the cross-task principle): leakage lives in
G_train. For a held-out target pair (M, D) we remove from G_train ALL direct
metabolite->disease edges on that pair -- treats / associated_with_disease /
enriched_in / depleted_in -- i.e. the co-located edges. We KEEP the mechanism
bridge (host_gene<->metabolite, metabolite->gene, gene->disease, therapeutic_target):
the (metabolite, disease) pair-scrub never touches gene-path edges (their node
pairs differ), so the mechanism path stays as signal -- only direct co-location is
removed. metabolite->gene->disease is mechanism, NOT leakage.

hard-neg = "associated but NOT treating": (M, D) pairs with enriched/depleted/assoc
but no treats edge -- biomarker-vs-therapy discrimination. Emitted as a pool;
ranking eval also draws sampled negatives (see README, report the true imbalance).

Settings:
  4A transductive (main): treats 80/10/10. G_train = full - colocated(valid+test).
  4B cross-evidence zero-shot: train supervision = enriched/depleted/assoc (minus
     test-pair co-located); test = ALL treats; valid = an assoc slice (so test
     stays pure zero-shot on treats). Maps association/mechanism -> therapy.
  4C cold-metabolite: hold out metabolites; their treats = test. G_train removes
     those metabolites' direct met-dis edges (gene bridge kept). Chemical-class
     stratification is TODO (needs ChEBI/ClassyFire ontology, absent from the KG).

Training graph = final_kg_edges.tsv - <setting>/gtrain_removed.tsv.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from split_utils import (
    SPLITS_DIR, EDGE_COLS,
    load_final_edges, write_tsv, split_three_way, colocated_removed,
)

TASK4_DIR = SPLITS_DIR / "task4_metabolite_disease"
SEED = 42
MD_RELS = ["treats_disease", "associated_with_disease", "enriched_in", "depleted_in"]


def md_edges(edges):
    return edges[(edges.head_type == "metabolite") & (edges.tail_type == "disease")
                 & edges.relation.isin(MD_RELS)].reset_index(drop=True)


def setting_a(edges, treats):
    out = TASK4_DIR / "setting_a_transductive"
    idx = list(range(len(treats)))
    tr, va, te = split_three_way(idx, (0.8, 0.1, 0.1), SEED)
    write_tsv(treats.iloc[tr].reset_index(drop=True), out / "train.tsv")
    write_tsv(treats.iloc[va].reset_index(drop=True), out / "valid.tsv")
    write_tsv(treats.iloc[te].reset_index(drop=True), out / "test.tsv")
    eval_pairs = set(zip(treats.iloc[va].head_id, treats.iloc[va].tail_id)) \
        | set(zip(treats.iloc[te].head_id, treats.iloc[te].tail_id))
    removed = colocated_removed(edges, eval_pairs)
    write_tsv(removed, out / "gtrain_removed.tsv")
    print(f"  4A transductive: train={len(tr):,} valid={len(va):,} test={len(te):,} "
          f"gtrain_removed={len(removed):,}")


def setting_b(edges, treats, assoc, enr_dep, treats_pairs):
    out = TASK4_DIR / "setting_b_cross_evidence"
    train_sup = pd.concat([assoc, enr_dep], ignore_index=True)
    keep = [(h, t) not in treats_pairs
            for h, t in zip(train_sup.head_id, train_sup.tail_id)]
    train_sup = train_sup[keep].reset_index(drop=True)
    idx = list(range(len(train_sup)))
    tr, va, _ = split_three_way(idx, (0.9, 0.1, 0.0), SEED)
    write_tsv(train_sup.iloc[tr].reset_index(drop=True), out / "train.tsv")
    write_tsv(train_sup.iloc[va].reset_index(drop=True), out / "valid.tsv")
    write_tsv(treats.reset_index(drop=True), out / "test.tsv")     # all treats, zero-shot
    removed = colocated_removed(edges, treats_pairs)
    write_tsv(removed, out / "gtrain_removed.tsv")
    print(f"  4B cross-evidence: train_sup={len(tr):,} valid={len(va):,} "
          f"test(all treats)={len(treats):,} gtrain_removed={len(removed):,}")


def setting_c(edges, treats, md, n_substrate_classes=None):
    out = TASK4_DIR / "setting_c_cold_metabolite"
    mets = sorted(set(treats.head_id))
    rng = random.Random(SEED)
    rng.shuffle(mets)
    n = len(mets)
    n_test, n_valid = int(round(n * 0.20)), int(round(n * 0.10))
    test_m = set(mets[:n_test])
    valid_m = set(mets[n_test:n_test + n_valid])
    held = test_m | valid_m
    write_tsv(treats[~treats.head_id.isin(held)].reset_index(drop=True), out / "train.tsv")
    write_tsv(treats[treats.head_id.isin(valid_m)].reset_index(drop=True), out / "valid.tsv")
    write_tsv(treats[treats.head_id.isin(test_m)].reset_index(drop=True), out / "test.tsv")
    held_md = md[md.head_id.isin(held)]
    held_pairs = set(zip(held_md.head_id, held_md.tail_id))
    removed = colocated_removed(edges, held_pairs)
    write_tsv(removed, out / "gtrain_removed.tsv")
    (out / "test_metabolites.txt").write_text("\n".join(sorted(test_m)), encoding="utf-8")
    (out / "chemical_class_stratification_TODO.md").write_text(
        "Chemical-class stratification (none-class = honest layer, the metabolite "
        "analog of Task 3 tax proximity) needs ChEBI / ClassyFire ontology, which is "
        "NOT in the KG. TODO: map metabolite HMDB/CHEBI ids to a ClassyFire superclass, "
        "then stratify the cold-metabolite test set by whether a same-class metabolite "
        "with treats edges remains in train; the none-class layer is the headline.\n",
        encoding="utf-8")
    print(f"  4C cold-metabolite: test={len(test_m)} valid={len(valid_m)} "
          f"train treats={len(treats)-len(treats[treats.head_id.isin(held)]):,} "
          f"gtrain_removed={len(removed):,}")


def write_hardneg_pool(treats_pairs, assoc, enr_dep):
    """associated-but-not-treating (M,D) pairs as a hard-negative pool."""
    pool = pd.concat([assoc, enr_dep], ignore_index=True)
    keep = [(h, t) not in treats_pairs
            for h, t in zip(pool.head_id, pool.tail_id)]
    pool = pool[keep].drop_duplicates(subset=["head_id", "tail_id"]).reset_index(drop=True)
    write_tsv(pool, TASK4_DIR / "hardneg_pool_associated_not_treating.tsv")
    return len(pool)


def write_readme(stats):
    (TASK4_DIR / "README.md").write_text(f"""# Task 4: Metabolite->Disease Therapeutic Prediction

## Target
`treats_disease` (metabolite->disease, {stats['treats']:,}, CTD curated). Secondary:
`associated_with_disease`(met) ({stats['assoc']:,}).

## G_train vs supervision (cross-task principle)
`train_graph = final_kg_edges.tsv - <setting>/gtrain_removed.tsv`.
For each held-out (M,D) target, `gtrain_removed.tsv` contains ALL direct
metabolite->disease edges on that pair (treats/assoc/enriched/depleted) -- the
co-located edges. The mechanism bridge (host_gene<->metabolite, metabolite->gene,
gene->disease) is KEPT: it is a different node pair, so the pair-scrub leaves it
intact. metabolite->gene->disease is mechanism, not leakage.

## Hard negatives
`hardneg_pool_associated_not_treating.tsv` ({stats['hardneg']:,} pairs): (M,D) with
enriched/depleted/assoc but NO treats -> biomarker-vs-therapy discrimination.
Ranking eval should also draw sampled negatives and report the TRUE imbalance
(do not silently use a fixed 1:k).

## Settings
- 4A transductive (main result): treats 80/10/10. Standard link prediction; does
  NOT hit the closure trap (treats has no closed-form 2-edge definition).
- 4B cross-evidence zero-shot: train supervision = enriched/depleted/assoc (minus
  test-pair co-located); test = ALL treats; valid = assoc slice (keeps test pure
  zero-shot on treats). Maps association/mechanism -> therapy.
- 4C cold-metabolite: hold out metabolites; their treats = test. G_train removes
  those metabolites' direct met-dis edges (gene bridge kept). Chemical-class
  stratification TODO (needs ChEBI/ClassyFire; see file in setting_c dir).

## Role
Task 4 is the insurance for Task 3: curated gold + real mechanism, and 4A is
standard link prediction that does not touch the closure trap. If Task 3 Setting
B's `none` layer collapses to baseline, Task 4 carries the discovery story.
""", encoding="utf-8")


def main():
    print("Loading KG...")
    edges = load_final_edges()
    print(f"  {len(edges):,} edges")
    md = md_edges(edges)
    treats = md[md.relation == "treats_disease"].reset_index(drop=True)
    assoc = md[md.relation == "associated_with_disease"].reset_index(drop=True)
    enr_dep = md[md.relation.isin(["enriched_in", "depleted_in"])].reset_index(drop=True)
    treats_pairs = set(zip(treats.head_id, treats.tail_id))
    print(f"  metabolite->disease: treats={len(treats):,} assoc={len(assoc):,} "
          f"enriched/depleted={len(enr_dep):,}")

    print("\n>>> Setting A (transductive)")
    setting_a(edges, treats)
    print("\n>>> Setting B (cross-evidence zero-shot)")
    setting_b(edges, treats, assoc, enr_dep, treats_pairs)
    print("\n>>> Setting C (cold-metabolite)")
    setting_c(edges, treats, md)
    n_hn = write_hardneg_pool(treats_pairs, assoc, enr_dep)
    print(f"\n  hard-neg pool (associated not treating): {n_hn:,}")

    write_readme(dict(treats=len(treats), assoc=len(assoc), hardneg=n_hn))
    print("\nTask 4 splits written to", TASK4_DIR)


if __name__ == "__main__":
    main()
