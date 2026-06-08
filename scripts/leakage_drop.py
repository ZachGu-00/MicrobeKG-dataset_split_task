"""Generate leakage-free splits from original Task 1 transductive splits.

Two surgical drops, both motivated by the audit (kg_build/leakage_audit/):

  1. DROP from train: edges with relation in {enriched_in, depleted_in} whose
     (microbe, disease) pair also appears in test_hardneg. This removes the
     structural shortcut that makes 93% of hard-negs trivially scorable -- a
     pure structure-only KGE no longer sees the pair in train, so AUROC-HN can
     no longer be driven by pair-level overlap.
  2. DROP from test: edges with a 1-hop closure leakage path via train's
     auxiliary taxonomy (C-hop1) or MeSH `is_a` (D-hop1) edges. The auxiliary
     hierarchies stay in train (the model still sees them as structural prior).

valid.tsv and test_hardneg.tsv are unchanged.

The closure is recomputed on the *post-drop* train, so the C/D drop count may
differ slightly from the original audit numbers (some pair-overlap train edges
also contributed to closure neighbors).
"""
import argparse
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = REPO / "splits" / "task1_microbe_disease" / "transductive" / "seed_42"
DEFAULT_OUT = REPO / "splits" / "task1_microbe_disease" / "transductive_leakage_free" / "seed_42"

TARGET_RELS = {"enriched_in", "depleted_in"}
HARDNEG_RELS = {"inconsistent_association"}
EVAL_RELS = TARGET_RELS | HARDNEG_RELS

TAX_RELS = {
    "belongs_to_phylum", "belongs_to_order", "belongs_to_class",
    "belongs_to_family", "belongs_to_genus",
    "is_strain_of", "is_clade_of",
}
MESH_REL = "is_a"


def load(path):
    return pd.read_csv(path, sep="\t", dtype=str)


def drop_train_hardneg_pair_overlap(train, hardneg):
    """Remove train target-relation edges whose (h,t) pair is in hardneg."""
    hardneg_pairs = set(zip(hardneg.head_id, hardneg.tail_id))
    pair_in_hn = [(h, t) in hardneg_pairs for h, t in zip(train.head_id, train.tail_id)]
    drop_mask = train.relation.isin(TARGET_RELS).values & pd.Series(pair_in_hn).values
    return train[~drop_mask].copy(), int(drop_mask.sum())


def find_test_closure_leak(train, test):
    """Return set of (h,r,t) in test that have 1-hop taxonomy or MeSH leakage."""
    train_md = train[train.relation.isin(EVAL_RELS)]
    train_md_set = set(zip(train_md.head_id, train_md.relation, train_md.tail_id))

    train_rd_to_microbes = defaultdict(set)
    train_mr_to_diseases = defaultdict(set)
    for h, r, t in train_md_set:
        train_rd_to_microbes[(r, t)].add(h)
        train_mr_to_diseases[(h, r)].add(t)

    G_mi = nx.from_pandas_edgelist(
        train[train.relation.isin(TAX_RELS)], "head_id", "tail_id")
    di_edges = train[(train.relation == MESH_REL) &
                     (train.head_type == "disease") &
                     (train.tail_type == "disease")]
    G_di = nx.from_pandas_edgelist(di_edges, "head_id", "tail_id")

    def neighbors_1hop(G, n):
        return set(G.neighbors(n)) if n in G else set()

    leak = set()
    for h, r, t in zip(test.head_id, test.relation, test.tail_id):
        if (h, r, t) in train_md_set:
            continue  # exact duplicate (audit reports 0 for Task 1 seed_42)
        if neighbors_1hop(G_mi, h) & train_rd_to_microbes.get((r, t), set()):
            leak.add((h, r, t))
            continue
        if neighbors_1hop(G_di, t) & train_mr_to_diseases.get((h, r), set()):
            leak.add((h, r, t))
    return leak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", default=str(DEFAULT_SPLITS))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    src = Path(args.splits_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = load(src / "train.tsv")
    valid = load(src / "valid.tsv")
    test = load(src / "test.tsv")
    hardneg = load(src / "test_hardneg.tsv")

    new_train, n_t = drop_train_hardneg_pair_overlap(train, hardneg)
    leak = find_test_closure_leak(new_train, test)
    test_keys = [(h, r, t) for h, r, t in zip(test.head_id, test.relation, test.tail_id)]
    drop_mask = [k in leak for k in test_keys]
    new_test = test[~pd.Series(drop_mask).values].copy()
    n_d = sum(drop_mask)

    new_train.to_csv(out / "train.tsv", sep="\t", index=False)
    valid.to_csv(out / "valid.tsv", sep="\t", index=False)
    new_test.to_csv(out / "test.tsv", sep="\t", index=False)
    hardneg.to_csv(out / "test_hardneg.tsv", sep="\t", index=False)

    print(f"train         : {len(train):,} -> {len(new_train):,}  (-{n_t}, hardneg-pair-overlap drop)")
    print(f"valid         : {len(valid):,} -> {len(valid):,}  (unchanged)")
    print(f"test          : {len(test):,} -> {len(new_test):,}  (-{n_d}, closure-1hop drop)")
    print(f"test_hardneg  : {len(hardneg):,} -> {len(hardneg):,}  (unchanged)")
    print(f"\nLeakage-free splits -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
