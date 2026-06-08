"""Leakage audit for KG splits -- 4 categories on microbe-disease edges:

  A. Duplicate (h,r,t) appearing in both train and test.
  B. Pair-level overlap: same (microbe, disease) in train and test with a
     different relation (e.g. train enriched_in, test depleted_in). Also run
     on test_hardneg vs train -- this directly quantifies why ComplEx/CompGCN
     score hard-negs above positives in Task 1 (Gap 1 sell-point B).
  C. Microbe-taxonomy closure: test (m, r, d) where some m' != m has (m', r, d)
     in train AND m, m' are connected in train's taxonomy subgraph (closest-hop
     bucketed in 1..max_hop).
  D. Disease-MeSH closure: same idea on the disease `is_a` hierarchy.

First version targets Task 1 transductive seed_42 but the CLI is reusable on
any TSV-split-dir with the canonical schema. Only reports counts + sample leak
edges; does NOT touch the splits (cleanup policy decided after seeing numbers).
"""
import argparse
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = REPO / "splits" / "task1_microbe_disease" / "transductive" / "seed_42"
DEFAULT_OUT = REPO / "kg_build" / "leakage_audit" / "task1_transductive_seed42"

TARGET_RELS = {"enriched_in", "depleted_in"}            # microbe -> disease positives
HARDNEG_RELS = {"inconsistent_association"}             # microbe -> disease hard-neg
EVAL_RELS = TARGET_RELS | HARDNEG_RELS

TAX_RELS = {                                            # microbe -> microbe taxonomy
    "belongs_to_phylum", "belongs_to_order", "belongs_to_class",
    "belongs_to_family", "belongs_to_genus",
    "is_strain_of", "is_clade_of",
}
MESH_REL = "is_a"                                       # disease -> disease MeSH


def load(path):
    return pd.read_csv(
        path, sep="\t", dtype=str,
        usecols=["head_id", "head_type", "relation", "tail_id", "tail_type"],
    )


def build_closure_graph(df, relations, head_type=None, tail_type=None):
    sub = df[df.relation.isin(relations)]
    if head_type:
        sub = sub[sub.head_type == head_type]
    if tail_type:
        sub = sub[sub.tail_type == tail_type]
    # undirected: parent and child are both "related" for leakage purposes
    return nx.from_pandas_edgelist(sub, "head_id", "tail_id")


def closure_by_hop(G, source, max_hop):
    """{hop_k: set_of_nodes_at_exactly_distance_k} for k in 1..max_hop."""
    out = {k: set() for k in range(1, max_hop + 1)}
    if source not in G:
        return out
    for n, d in nx.single_source_shortest_path_length(G, source, cutoff=max_hop).items():
        if 1 <= d <= max_hop:
            out[d].add(n)
    return out


def closure_leak(test_list, train_set, train_idx, G, source_field, max_hop):
    """Walk each test edge, look up the closest-hop leakage neighbor in train.
    `source_field` is 'head' (microbe-taxonomy) or 'tail' (disease-MeSH).
    `train_idx` maps the *fixed* axis (relation,disease) or (microbe,relation)
    to the set of neighbors on the closure axis.
    """
    leak = {k: [] for k in range(1, max_hop + 1)}
    cache = {}
    for h, r, t in test_list:
        if (h, r, t) in train_set:
            continue  # exact dup is category A
        if source_field == "head":
            src, key = h, (r, t)
        else:
            src, key = t, (h, r)
        train_neighbors = train_idx.get(key, set())
        if not train_neighbors:
            continue
        if src not in cache:
            cache[src] = closure_by_hop(G, src, max_hop)
        for k in range(1, max_hop + 1):
            hit = cache[src][k] & train_neighbors
            if hit:
                leak[k].append((h, r, t, sorted(hit)[:3]))
                break  # closest hop only -- avoid double-counting
    return leak


def audit(splits_dir, out_dir, max_hop=2, n_samples=20):
    splits_dir = Path(splits_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading splits from {splits_dir}")
    train = load(splits_dir / "train.tsv")
    test = load(splits_dir / "test.tsv")
    hardneg = load(splits_dir / "test_hardneg.tsv")

    train_md = train[train.relation.isin(EVAL_RELS)]
    test_md = test[test.relation.isin(EVAL_RELS)]
    hardneg_md = hardneg[hardneg.relation.isin(EVAL_RELS)]
    print(f"  train microbe-disease edges (incl. hard-neg): {len(train_md):,}")
    print(f"  test  microbe-disease edges (positives only): {len(test_md):,}")
    print(f"  hardneg microbe-disease edges:                {len(hardneg_md):,}")

    train_set = set(zip(train_md.head_id, train_md.relation, train_md.tail_id))
    test_list = list(zip(test_md.head_id, test_md.relation, test_md.tail_id))
    hardneg_list = list(zip(hardneg_md.head_id, hardneg_md.relation, hardneg_md.tail_id))
    test_n = len(test_list)
    hardneg_n = len(hardneg_list)

    # --- A: exact duplicate (h,r,t) in both train and test ---
    dup = [t for t in test_list if t in train_set]
    print(f"\nA. Exact duplicate: {len(dup)} / {test_n}")

    # --- B: pair-level overlap (cross-relation) ---
    train_pair_to_rels = defaultdict(set)
    for h, r, t in train_set:
        train_pair_to_rels[(h, t)].add(r)

    def pair_overlap(probe):
        rows = []
        for h, r, t in probe:
            other = train_pair_to_rels.get((h, t), set()) - {r}
            if other:
                rows.append((h, r, t, sorted(other)))
        return rows

    pair_overlap_test = pair_overlap(test_list)
    pair_overlap_hn = pair_overlap(hardneg_list)
    print(f"B. Pair-level overlap on test:    {len(pair_overlap_test)} / {test_n}")
    print(f"B. Pair-level overlap on hardneg: {len(pair_overlap_hn)} / {hardneg_n}")

    # --- index train microbe-disease for closure lookups ---
    train_rd_to_microbes = defaultdict(set)
    train_mr_to_diseases = defaultdict(set)
    for h, r, t in train_set:
        train_rd_to_microbes[(r, t)].add(h)
        train_mr_to_diseases[(h, r)].add(t)

    # --- C: microbe-taxonomy closure ---
    G_mi = build_closure_graph(train, TAX_RELS)
    print(f"  microbe taxonomy subgraph: {G_mi.number_of_nodes():,} nodes, "
          f"{G_mi.number_of_edges():,} edges")
    tax_leak = closure_leak(test_list, train_set, train_rd_to_microbes, G_mi,
                            source_field="head", max_hop=max_hop)
    for k in range(1, max_hop + 1):
        print(f"C. Microbe-taxonomy closure (closest hop = {k}): "
              f"{len(tax_leak[k])} / {test_n}")

    # --- D: disease-MeSH closure ---
    G_di = build_closure_graph(train, {MESH_REL}, head_type="disease", tail_type="disease")
    print(f"  disease MeSH subgraph: {G_di.number_of_nodes():,} nodes, "
          f"{G_di.number_of_edges():,} edges")
    mesh_leak = closure_leak(test_list, train_set, train_mr_to_diseases, G_di,
                             source_field="tail", max_hop=max_hop)
    for k in range(1, max_hop + 1):
        print(f"D. Disease-MeSH closure (closest hop = {k}): "
              f"{len(mesh_leak[k])} / {test_n}")

    # --- write sample leak edges ---
    def save(name, rows, cols):
        if rows:
            pd.DataFrame(rows[:n_samples], columns=cols).to_csv(
                out_dir / f"leak_{name}.tsv", sep="\t", index=False)

    save("duplicate", dup, ["head_id", "relation", "tail_id"])
    save("pair_overlap_test", pair_overlap_test,
         ["head_id", "test_relation", "tail_id", "train_other_relations"])
    save("pair_overlap_hardneg", pair_overlap_hn,
         ["head_id", "hardneg_relation", "tail_id", "train_other_relations"])
    for k in range(1, max_hop + 1):
        save(f"taxonomy_hop{k}", tax_leak[k],
             ["head_id", "relation", "tail_id", "train_microbes_in_closure"])
        save(f"mesh_hop{k}", mesh_leak[k],
             ["head_id", "relation", "tail_id", "train_diseases_in_closure"])

    # --- markdown report ---
    def pct(n, d):
        return f"{100 * n / d:.2f}%" if d else "-"

    lines = [
        "# Task 1 transductive seed_42 -- leakage audit",
        "",
        f"- splits: `{splits_dir.relative_to(REPO)}`",
        f"- train microbe-disease edges (positives + hard-neg): {len(train_md):,}",
        f"- test  microbe-disease edges (positives only): {test_n:,}",
        f"- test_hardneg microbe-disease edges: {hardneg_n:,}",
        f"- microbe taxonomy subgraph (from train edges): "
        f"{G_mi.number_of_nodes():,} nodes / {G_mi.number_of_edges():,} edges "
        f"(relations: {sorted(TAX_RELS)})",
        f"- disease MeSH subgraph (from train edges): "
        f"{G_di.number_of_nodes():,} nodes / {G_di.number_of_edges():,} edges "
        f"(relation: `{MESH_REL}`)",
        "",
        "## Summary",
        "",
        "| Audit | Count | % of probe |",
        "|---|---:|---:|",
        f"| A. Exact duplicate `(h,r,t)` in train (test probe) | {len(dup)} | {pct(len(dup), test_n)} |",
        f"| B. Pair-level overlap (test probe, cross-relation) | {len(pair_overlap_test)} | {pct(len(pair_overlap_test), test_n)} |",
        f"| B. Pair-level overlap (**hardneg probe**, cross-relation) | {len(pair_overlap_hn)} | {pct(len(pair_overlap_hn), hardneg_n)} |",
    ]
    for k in range(1, max_hop + 1):
        lines.append(
            f"| C. Microbe-taxonomy closure (closest hop = {k}) | "
            f"{len(tax_leak[k])} | {pct(len(tax_leak[k]), test_n)} |")
    for k in range(1, max_hop + 1):
        lines.append(
            f"| D. Disease-MeSH closure (closest hop = {k}) | "
            f"{len(mesh_leak[k])} | {pct(len(mesh_leak[k]), test_n)} |")

    lines += [
        "",
        "## Notes",
        "",
        "- *Closest hop* = an edge with a 1-hop leakage neighbor is counted in "
        "hop=1 only, never double-counted in hop=2. C-hop buckets and D-hop "
        "buckets are each pairwise disjoint.",
        "- A and B are disjoint: B is the *cross-relation* slice, so a test "
        "edge matching a train edge exactly is counted in A only.",
        "- C and D skip A-duplicates (exact dups are counted in A and removed "
        "from the closure pool to avoid trivial \"leakage\").",
        "- B is reported on **both** the test probe (positives) and the "
        "test_hardneg probe -- the hardneg row directly quantifies why "
        "ComplEx/CompGCN score hard-negs above positives (Gap 1 sell-point B): "
        "if `(m, d)` has *any* train edge of another relation, a structure-only "
        "model gives the pair a high score regardless of which relation is "
        "actually being queried.",
        "- Sample leak edges (first 20 per category) are in `leak_*.tsv`.",
        "",
        "## Interpretation",
        "",
        "- A > 0 means the split is broken (a test edge literally lives in train).",
        "- B-test > 0 means the model can shortcut some test positives via a "
        "different-relation train edge on the same `(microbe, disease)` pair.",
        "- B-hardneg > 0 means the hard-neg `inconsistent_association` edges "
        "share their `(m, d)` pair with positive train edges of `enriched_in` "
        "or `depleted_in` -- a structure-only KGE then ranks the hardneg above "
        "true positives because the pair *does* have edges in train.",
        "- C / D quantify how much of the test set can be solved by looking "
        "up train neighbors through the *auxiliary* taxonomy / MeSH edges. "
        "1-hop leakage is the hardest to defend (essentially the same microbe "
        "or disease at a different node in the hierarchy).",
        "- These numbers feed the design of the leakage-free splits "
        "(Gap 3 paper deliverable: original splits + leakage-free splits "
        "released together as an audited benchmark).",
    ]
    (out_dir / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport -> {(out_dir / 'audit_report.md').relative_to(REPO)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", default=str(DEFAULT_SPLITS))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--max_hop", type=int, default=2,
                    help="C/D closure depth to report (closest-hop bucketed)")
    ap.add_argument("--n_samples", type=int, default=20)
    args = ap.parse_args()
    audit(args.splits_dir, args.out_dir, args.max_hop, args.n_samples)


if __name__ == "__main__":
    main()
