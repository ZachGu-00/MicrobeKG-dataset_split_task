"""Build Task 2 (Capacity -> Realization Transfer) splits.

Target: utilizes (microbe->substrate, experimental). Capacity = can_utilize (GEM
computational). The task: does computationally-predicted metabolic CAPACITY transfer
to experimentally-observed REALIZATION?

hard-neg: does_not_utilize (literature-explicit negation; very FEW -> report n,
draw sampled negatives at eval, state the true imbalance).

G_train vs supervision: for the zero-shot settings, valid is a can_utilize SLICE
(NOT a utilizes slice) so test stays pure zero-shot on the utilizes relation.
Leakage-scrub: any (microbe, substrate) pair in the utilizes / does_not_utilize
eval set is removed from the can_utilize train pool (self-pair).
PROBE (was missing): test microbes stratified by taxonomic proximity to train
can_utilize microbes, excluding self -- the `none` layer is the honest transfer
signal (no same-genus can_utilize to copy). 0.626 was likely inflated by copying.

Settings:
  2A within-relation: can_utilize 90/5/5 (ceiling/control, NOT the real task).
  2B cross-relation zero-shot (flagship): train = can_utilize(scrubbed) + taxonomy;
     valid = can_utilize slice; test = ALL utilizes (+ does_not_utilize hard-neg).
  2C few-shot: inject N in {50,100,500} utilizes into train; test = remaining utilizes.
"""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pandas as pd

from split_utils import (
    SPLITS_DIR, EDGE_COLS,
    load_final_edges, write_tsv, split_three_way,
    extract_microbe_lineage, nearest_shared_taxon,
)

TASK2_DIR = SPLITS_DIR / "task2_capacity_realization"
SEED = 42
TAX_RELS = [
    "is_strain_of", "is_clade_of", "belongs_to_genus", "belongs_to_family",
    "belongs_to_order", "belongs_to_class", "belongs_to_phylum",
]
FEWSHOT_N = [50, 100, 500]


def main():
    print("Loading KG...")
    edges = load_final_edges()
    print(f"  {len(edges):,} edges")
    cu = edges[edges.relation == "can_utilize"].reset_index(drop=True)
    util = edges[edges.relation == "utilizes"].reset_index(drop=True)
    dnu = edges[edges.relation == "does_not_utilize"].reset_index(drop=True)
    tax = edges[edges.relation.isin(TAX_RELS)].reset_index(drop=True)
    lineage = extract_microbe_lineage(edges)
    print(f"  can_utilize={len(cu):,} utilizes={len(util):,} "
          f"does_not_utilize={len(dnu):,} tax={len(tax):,}")

    # ---- 2A within-relation (ceiling) ----
    a = TASK2_DIR / "setting_a_within_relation"
    tr, va, te = split_three_way(list(range(len(cu))), (0.90, 0.05, 0.05), SEED)
    write_tsv(cu.iloc[tr][EDGE_COLS], a / "train.tsv")
    write_tsv(cu.iloc[va][EDGE_COLS], a / "valid.tsv")
    write_tsv(cu.iloc[te][EDGE_COLS], a / "test.tsv")
    print(f"  2A: train={len(tr):,} valid={len(va):,} test={len(te):,}")

    # ---- 2B cross-relation zero-shot ----
    b = TASK2_DIR / "setting_b_cross_relation"
    eval_pairs = set(zip(util.head_id, util.tail_id)) | set(zip(dnu.head_id, dnu.tail_id))
    keep = [(h, t) not in eval_pairs for h, t in zip(cu.head_id, cu.tail_id)]
    cu_scrub = cu[keep].reset_index(drop=True)
    n_scrubbed = len(cu) - len(cu_scrub)
    cu_tr, cu_va, _ = split_three_way(list(range(len(cu_scrub))), (0.95, 0.05, 0.0), SEED)
    train_b = pd.concat([cu_scrub.iloc[cu_tr][EDGE_COLS], tax[EDGE_COLS]], ignore_index=True)
    write_tsv(train_b, b / "train.tsv")
    write_tsv(cu_scrub.iloc[cu_va][EDGE_COLS], b / "valid.tsv")
    write_tsv(util[EDGE_COLS], b / "test.tsv")
    write_tsv(dnu[EDGE_COLS], b / "test_hardneg.tsv")
    # T2 stratification axis = CAPACITY (does the microbe have its OWN can_utilize),
    # NOT the cold-entity tax axis. capacity->realization is a WITHIN-microbe transfer
    # (its computed capacity -> its observed realization); a same-genus sibling is only
    # secondary leakage. Three layers:
    #   has_own_capacity = microbe keeps its own can_utilize in train -> HEADLINE
    #                      (where AGORA2's 1.5M computed edges can actually transfer)
    #   tax_only         = no own can_utilize, but a same-clade sibling has one -> leakage-risk
    #   no_capacity      = neither -> capacity->realization UNTESTABLE here (no capacity to
    #                      transfer); a different question (closer to T3's 8,106), report as
    #                      floor / exclude from the transfer headline.
    test_microbes = set(util.head_id)
    train_cu_microbes = set(cu_scrub.iloc[cu_tr].head_id)
    has_own = test_microbes & train_cu_microbes
    no_own = test_microbes - train_cu_microbes
    prox, _ = nearest_shared_taxon(no_own, train_cu_microbes, lineage, exclude_query=True)
    prox_map = dict(zip(prox["microbe"], prox["nearest_shared_rank"]))
    rows, layer_count = [], Counter()
    for m in sorted(test_microbes):
        if m in has_own:
            layer, near = "has_own_capacity", "self"
        elif prox_map.get(m, "none") != "none":
            layer, near = "tax_only", prox_map[m]
        else:
            layer, near = "no_capacity", "none"
        layer_count[layer] += 1
        rows.append({"microbe": m, "capacity_layer": layer, "nearest_shared_rank": near})
    write_tsv(pd.DataFrame(rows, columns=["microbe", "capacity_layer", "nearest_shared_rank"]),
              b / "test_microbe_capacity_layer.tsv")
    print(f"  2B: train={len(train_b):,} (cu_scrub={len(cu_scrub):,}, "
          f"scrubbed self-pairs={n_scrubbed:,}) valid={len(cu_va):,} "
          f"test(all utilizes)={len(util):,} hardneg(dnu)={len(dnu):,}")
    print(f"      capacity layer (HEADLINE=has_own_capacity): {dict(layer_count)}")

    # ---- 2C few-shot ----
    for N in FEWSHOT_N:
        c = TASK2_DIR / f"setting_c_fewshot_n{N}"
        rng = random.Random(SEED)
        u_idx = list(range(len(util)))
        rng.shuffle(u_idx)
        if N > len(u_idx):
            print(f"  2C n={N}: skipped (only {len(u_idx)} utilizes)")
            continue
        inject = util.iloc[u_idx[:N]]
        test = util.iloc[u_idx[N:]]
        train_c = pd.concat(
            [cu_scrub.iloc[cu_tr][EDGE_COLS], tax[EDGE_COLS], inject[EDGE_COLS]],
            ignore_index=True)
        write_tsv(train_c, c / "train.tsv")
        write_tsv(test[EDGE_COLS], c / "test.tsv")
        write_tsv(dnu[EDGE_COLS], c / "test_hardneg.tsv")
        print(f"  2C n={N}: train={len(train_c):,} (inject {N} utilizes) test={len(test):,}")

    (TASK2_DIR / "README.md").write_text(
        "# Task 2: Capacity -> Realization Transfer\n\n"
        "Target `utilizes` (microbe->substrate, experimental); capacity = `can_utilize`"
        " (GEM computational). Does predicted capacity transfer to observed realization?\n\n"
        "- 2A within-relation: can_utilize 90/5/5 (ceiling, NOT the task).\n"
        "- 2B cross-relation zero-shot (flagship): train = can_utilize(self-pair scrubbed)"
        " + taxonomy; valid = can_utilize slice (so test stays pure zero-shot); test = ALL"
        " utilizes; hard-neg = does_not_utilize.\n"
        "- 2C few-shot: inject N utilizes into train.\n\n"
        "## Taxonomic-shortcut probe (NEW)\n"
        "`setting_b_cross_relation/test_microbe_tax_proximity.tsv` stratifies test"
        " utilizes microbes by nearest rank sharing a taxon with a train can_utilize"
        " microbe (self excluded). The `none` layer is the honest transfer signal --"
        " elsewhere the model can copy a same-genus microbe's can_utilize. Report 2B"
        " metrics stratified by this; consider a taxonomy-ablated run.\n\n"
        "## Hard-neg imbalance\n"
        "does_not_utilize is tiny -- report its n and the sampled-negative ratio"
        " explicitly; do not report AUPRC at a hidden imbalance.\n",
        encoding="utf-8")
    print("\nTask 2 splits written to", TASK2_DIR)


if __name__ == "__main__":
    main()
