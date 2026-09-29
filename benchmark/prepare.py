"""Prepare the minimal benchmark from the frozen, evidence-annotated KG.

No training is performed. Entity identity is (node_type, node_id). The indexed
files are a compact server input; the original KG remains the provenance source.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "minimal-v1"
SEEDS = [42, 123, 456]
CORE = ["t1_pair", "t1_bridges", "t1_cold_microbe", "t1_cold_disease",
        "t2_a", "t2_b", "t4_a", "t4_b", "t4_c"]
OPTIONAL = ["t2_c100", "t1_triple_random"]
TAX = ["belongs_to_genus", "belongs_to_family", "belongs_to_order",
       "belongs_to_class", "belongs_to_phylum", "is_strain_of", "is_clade_of"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        temporary = Path(f.name)
    temporary.replace(path)


def pair_code(x, n, unordered=False):
    h, t = x[:, 0].astype(np.int64), x[:, 2].astype(np.int64)
    if unordered:
        h, t = np.minimum(h, t), np.maximum(h, t)
    return h * n + t


def partition(values, seed, ratios=(0.8, 0.1)):
    values = np.sort(np.unique(values))
    values = np.random.default_rng(seed).permutation(values)
    a, b = round(len(values) * ratios[0]), round(len(values) * ratios[1])
    return values[:a], values[a:a+b], values[a+b:]


def encode_kg(root, out):
    nodes_file = root / "kg_build/final_kg_nodes.tsv"
    edges_file = root / "kg_build/final_kg_edges.tsv"
    nodes = pd.read_csv(nodes_file, sep="\t", dtype=str, keep_default_na=False)
    nodes = nodes.sort_values(["node_type", "node_id"]).reset_index(drop=True)
    assert not nodes.duplicated(["node_type", "node_id"]).any()
    nodes.insert(0, "entity_index", np.arange(len(nodes)))
    mapping = {(t, i): j for j, t, i in nodes[["entity_index", "node_type", "node_id"]].itertuples(index=False, name=None)}
    relations = set()
    blocks, source_counts, evidence_counts, rel_counts = [], Counter(), Counter(), Counter()
    direction = {s: defaultdict(set) for s in ["BugSigDB", "GMrepo_v3", "Disbiome"]}
    cols = ["head_id", "head_type", "relation", "tail_id", "tail_type", "source", "evidence_type"]
    for df in pd.read_csv(edges_file, sep="\t", usecols=cols, dtype=str,
                          keep_default_na=False, chunksize=200000):
        h = np.fromiter((mapping[(t, i)] for t, i in zip(df.head_type, df.head_id)), dtype=np.int32)
        t = np.fromiter((mapping[(t, i)] for t, i in zip(df.tail_type, df.tail_id)), dtype=np.int32)
        blocks.append((h, df.relation.to_numpy(), t))
        relations.update(df.relation)
        rel_counts.update(df.relation)
        evidence_counts.update(df.evidence_type)
        for label, count in df.source.value_counts().items():
            for source in label.split("|"):
                source_counts[source] += int(count)
        md = df[(df.head_type == "microbe") & (df.tail_type == "disease")
                & df.relation.isin(["enriched_in", "depleted_in"])]
        for row in md.itertuples(index=False):
            for source in row.source.split("|"):
                if source in direction:
                    direction[source][(row.head_id, row.tail_id)].add(row.relation)
    rels = sorted(relations)
    rid = {r: i for i, r in enumerate(rels)}
    raw = np.concatenate([np.column_stack([h, np.fromiter((rid[r] for r in rvec), dtype=np.int32), t])
                          for h, rvec, t in blocks])
    triples = np.unique(raw, axis=0).astype(np.int32)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "triples.npy", triples, allow_pickle=False)
    nodes.to_csv(out / "entities.tsv", sep="\t", index=False)
    write_json(out / "relations.json", rels)
    agreement = []
    for a, b in itertools.combinations(direction, 2):
        overlap = set(direction[a]) & set(direction[b])
        clean = [p for p in overlap if len(direction[a][p]) == len(direction[b][p]) == 1]
        concordant = sum(direction[a][p] == direction[b][p] for p in clean)
        agreement.append(dict(source_a=a, source_b=b, overlapping_pairs=len(overlap),
                              unambiguous_pairs=len(clean), concordant_pairs=concordant,
                              concordance=concordant / len(clean) if clean else None,
                              within_source_conflicting_pairs=len(overlap)-len(clean)))
    stats = dict(raw_edge_records=len(raw), unique_typed_triples=len(triples),
                 duplicate_typed_triples_collapsed=len(raw)-len(triples),
                 typed_entities=len(nodes), bare_ids=nodes.node_id.nunique(),
                 multi_type_ids=int((nodes.groupby("node_id").node_type.nunique() > 1).sum()),
                 node_types=nodes.node_type.value_counts().to_dict(), relations=rel_counts,
                 sources=source_counts, evidence_types=evidence_counts,
                 cross_source_direction_agreement=agreement,
                 original_edges_sha256=sha256(edges_file), original_nodes_sha256=sha256(nodes_file))
    write_json(out / "data_quality.json", stats)
    return triples, nodes, rels, stats


def closure_flags(train, test, rels):
    """Closest 1/2-hop taxonomy and ontology shortcuts under the project's rule."""
    tax_ids = {rels.index(r) for r in TAX}
    isa = rels.index("is_a")
    pos_ids = {rels.index("enriched_in"), rels.index("depleted_in")}
    tax, ont, rd, hr = (defaultdict(set) for _ in range(4))
    for h, r, t in train:
        if r in tax_ids or r == isa:
            g = tax if r in tax_ids else ont
            g[int(h)].add(int(t)); g[int(t)].add(int(h))
        if r in pos_ids:
            rd[(int(r), int(t))].add(int(h))
            hr[(int(h), int(r))].add(int(t))

    def closest(g, x, targets):
        adjacent = g.get(int(x), set())
        if adjacent & targets:
            return 1
        if any(g.get(v, set()) & targets for v in adjacent):
            return 2
        return 0

    th = np.array([closest(tax, h, rd.get((int(r), int(t)), set())) for h, r, t in test], dtype=np.int8)
    dh = np.array([closest(ont, t, hr.get((int(h), int(r)), set())) for h, r, t in test], dtype=np.int8)
    return th, dh


def save_split(out, setting, seed, x, types, rels, train, valid, test,
               score_rel, negatives=None, conflict=None, known_extra=None, eligible_nodes=None):
    train, valid, test = [np.sort(np.unique(v)).astype(np.int32) for v in (train, valid, test)]
    assert not np.intersect1d(train, valid).size
    assert not np.intersect1d(train, test).size
    assert not np.intersect1d(valid, test).size
    n = len(types)
    seen = np.zeros(n, bool); seen[x[train][:, [0, 2]].ravel()] = True
    train_pairs = pair_code(x[train], n, unordered=True)
    vpair = np.isin(pair_code(x[valid], n, True), train_pairs)
    tpair = np.isin(pair_code(x[test], n, True), train_pairs)
    if setting != "t1_triple_random":
        assert not vpair.any() and not tpair.any(), (setting, "pair leakage")
    evaluation_seen = seen if eligible_nodes is None else seen & eligible_nodes
    available = lambda rows: evaluation_seen[x[rows, 0]] & evaluation_seen[x[rows, 2]]
    vm, tm = available(valid), available(test)
    assert vm.any() and tm.any(), (setting, "empty evaluable partition")
    neg = np.asarray(negatives if negatives is not None else [], dtype=np.int32)
    cf = np.asarray(conflict if conflict is not None else [], dtype=np.int32)
    if len(neg):
        assert not np.intersect1d(pair_code(x[test], n), pair_code(x[neg], n)).size
        assert not np.isin(pair_code(x[neg], n, True), train_pairs).any()
    if len(cf):
        assert not np.isin(pair_code(x[cf], n, True), train_pairs).any()
    arrays = dict(train=train, valid=valid, test=test, candidate_eligible=evaluation_seen, valid_eligible=vm,
                  test_eligible=tm, negatives=neg, negative_eligible=available(neg),
                  conflict=cf, conflict_eligible=available(cf),
                  known_extra=np.asarray(known_extra if known_extra is not None else [], dtype=np.int32))
    audit = dict(train_triples=len(train), valid_total=len(valid), test_total=len(test),
                 valid_evaluable=int(vm.sum()), test_evaluable=int(tm.sum()),
                 test_oov=int((~(seen[x[test, 0]] & seen[x[test, 2]])).sum()),
                 test_excluded=int((~tm).sum()), negative_total=len(neg),
                 negative_evaluable=int(available(neg).sum()),
                 exact_train_valid=0, exact_train_test=0,
                 train_valid_pair_overlap=int(vpair.sum()), train_test_pair_overlap=int(tpair.sum()))
    if setting.startswith("t1"):
        th, dh = closure_flags(x[train], x[test], rels)
        arrays.update(taxonomy_hop=th, ontology_hop=dh)
        audit.update(taxonomy_hop1=int((th == 1).sum()), taxonomy_hop2=int((th == 2).sum()),
                     ontology_hop1=int((dh == 1).sum()), ontology_hop2=int((dh == 2).sum()),
                     no_closure_within_2hop=int(((th == 0) & (dh == 0)).sum()))
        # Clean association vs conflict is a separate probe; never contradictory labels for one pair.
        conflict_pairs = pair_code(x[np.flatnonzero(x[:, 1] == rels.index("inconsistent_association"))], n)
        clean = test[tm & ~np.isin(pair_code(x[test], n), conflict_pairs)]
        _, first = np.unique(pair_code(x[clean], n), return_index=True)
        clean = clean[first]
        arrays["probe_positive"] = clean
        # Same head, same log2 train-degree bin for the disease; known pairs excluded.
        degrees = np.bincount(x[train][:, [0, 2]].ravel(), minlength=n)
        bins = np.floor(np.log2(np.maximum(degrees, 1))).astype(int)
        disease = np.flatnonzero((types == "disease") & evaluation_seen)
        bin_pools = {b: disease[bins[disease] == b] for b in np.unique(bins[disease])}
        known_md = set(pair_code(x[np.isin(x[:, 1], [rels.index(r) for r in
                               ["enriched_in", "depleted_in", "inconsistent_association"]])], n))
        controls, matched = [], []
        rng = np.random.default_rng(seed)
        used = set()
        for idx in clean:
            h, r, t = x[idx]
            for candidate in rng.permutation(bin_pools[bins[t]]):
                code = int(h) * n + int(candidate)
                if code not in known_md and code not in used:
                    controls.append([h, r, candidate]); matched.append(idx); used.add(code)
                    break
        arrays["random_control"] = np.asarray(controls, np.int32).reshape(-1, 3)
        arrays["random_control_positive"] = np.asarray(matched, np.int32)
        audit.update(conflict_evaluable=int(available(cf).sum()), clean_probe_pairs=len(clean),
                     degree_matched_controls=len(controls), degree_match_failures=len(clean)-len(controls))
    folder = out / "splits" / setting
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"seed_{seed}.npz", **arrays)
    meta = dict(protocol=PROTOCOL, setting=setting, seed=seed, scoring_relations=score_rel,
                head_type=str(types[x[test[0], 0]]), tail_type=str(types[x[test[0], 2]]),
                ranking_sides="tail" if setting.startswith("t2") else "both", audit=audit)
    write_json(folder / f"seed_{seed}.json", meta)
    print(setting, seed, audit, flush=True)
    return meta


def prepare_splits(out, x, nodes, rels, seeds):
    types = nodes.node_type.to_numpy(); n = len(types); rid = {r: i for i, r in enumerate(rels)}
    relation = lambda names: np.isin(x[:, 1], [rid[r] for r in names])
    md = (types[x[:, 0]] == "microbe") & (types[x[:, 2]] == "disease")
    pos = np.flatnonzero(md & relation(["enriched_in", "depleted_in"]))
    conflict = np.flatnonzero(md & relation(["inconsistent_association"]))
    aux = np.flatnonzero(relation(TAX + ["is_a", "co_occurs_with"]))
    tax = np.flatnonzero(relation(TAX))
    bridges = np.flatnonzero(~md & ~relation(TAX + ["is_a", "co_occurs_with"]))
    pair = pair_code(x, n)
    upair = pair_code(x, n, True)
    cu = np.flatnonzero(relation(["can_utilize"]))
    util = np.flatnonzero(relation(["utilizes"]))
    dnu = np.flatnonzero(relation(["does_not_utilize"]))
    metdis = (types[x[:, 0]] == "metabolite") & (types[x[:, 2]] == "disease")
    therapy = np.flatnonzero(metdis & relation(["treats_disease"]))
    assoc = np.flatnonzero(metdis & relation(["associated_with_disease", "enriched_in", "depleted_in"]))
    assoc = assoc[~np.isin(pair[assoc], pair[therapy])]
    _, first = np.unique(pair[assoc], return_index=True)
    negative_pool = assoc[first]
    all_ids = np.arange(len(x), dtype=np.int32)
    output = []

    def save(name, seed, tr, va, te, scoring, **kw):
        output.append(save_split(out, name, seed, x, types, rels, tr, va, te, scoring, **kw))

    def scrub(indices, withheld):
        return indices[~np.isin(upair[indices], upair[withheld])]

    for seed in seeds:
        trp, vap, tep = partition(pair[pos], seed)
        tr, va, te = [pos[np.isin(pair[pos], p)] for p in (trp, vap, tep)]
        cf = conflict[np.isin(pair[conflict], tep)]
        base = np.r_[tr, aux]
        base_seen = np.zeros(n, bool); base_seen[x[base][:, [0, 2]].ravel()] = True
        save("t1_pair", seed, base, va, te, ["enriched_in", "depleted_in"], conflict=cf)
        save("t1_bridges", seed, np.r_[base, bridges], va, te,
             ["enriched_in", "depleted_in"], conflict=cf, eligible_nodes=base_seen)
        rt, rv, re = partition(pos, seed)
        save("t1_triple_random", seed, np.r_[rt, aux], rv, re,
             ["enriched_in", "depleted_in"])
        for name, col in [("t1_cold_microbe", 0), ("t1_cold_disease", 2)]:
            entities = np.unique(x[pos, col])
            if col == 2:
                entities = np.intersect1d(entities, x[x[:, 1] == rid["is_a"], 0])
            _, valid_entities, test_entities = partition(entities, seed, (0.9, 0.05))
            va = pos[np.isin(x[pos, col], valid_entities)]
            te = pos[np.isin(x[pos, col], test_entities)]
            tr = pos[~np.isin(x[pos, col], np.r_[valid_entities, test_entities])]
            cf = conflict[np.isin(x[conflict, col], test_entities)]
            save(name, seed, np.r_[tr, aux], va, te, ["enriched_in", "depleted_in"], conflict=cf)
        tr, va, te = partition(cu, seed, (0.9, 0.05))
        save("t2_a", seed, tr, va, te, ["can_utilize"])
        # 10% observed utilization for validation, 100 disjoint optional hints;
        # B and C100 share exactly the remaining test triples and candidate masks.
        tr = np.r_[scrub(cu, np.r_[util, dnu]), tax]
        remainder, va, _ = partition(util, seed, (0.9, 0.1))
        seen = np.unique(x[tr][:, [0, 2]])
        hint_pool = remainder[np.isin(x[remainder, 0], seen) & np.isin(x[remainder, 2], seen)]
        hints = hint_pool[:100]
        assert len(hints) == 100
        te = remainder[~np.isin(remainder, hints)]
        save("t2_b", seed, tr, va, te, ["can_utilize"], negatives=dnu, known_extra=util)
        # Inject hints using the trained capacity predicate, making adaptation
        # and evaluation explicitly share a scoring relation.
        save("t2_c100", seed, np.r_[tr, hints], va, te, ["can_utilize"], negatives=dnu, known_extra=util)
        for name in ["t4_a", "t4_b", "t4_c"]:
            if name == "t4_a":
                _, va, te = partition(therapy, seed)
            elif name == "t4_b":
                _, vap, _ = partition(pair[assoc], seed, (0.9, 0.1))
                # Hold out only relation-compatible association validation,
                # and scrub all co-located relations for those pairs.
                va = assoc[np.isin(pair[assoc], vap) & (x[assoc, 1] == rid["associated_with_disease"])]
                te = therapy
            else:
                _, vm, tm = partition(x[therapy, 0], seed, (0.7, 0.1))
                va = therapy[np.isin(x[therapy, 0], vm)]
                te = therapy[np.isin(x[therapy, 0], tm)]
            pool = negative_pool[~np.isin(pair[negative_pool], pair[va])]
            if name == "t4_c":
                pool = pool[np.isin(x[pool, 0], x[te, 0])]
            rng = np.random.default_rng(seed)
            neg = rng.choice(pool, size=min(10 * len(te), len(pool)), replace=False)
            withheld = np.r_[va, te, neg]
            if name == "t4_c":
                withheld = np.r_[withheld, np.flatnonzero(metdis & np.isin(x[:, 0], np.r_[vm, tm]))]
            tr = scrub(all_ids, withheld)
            save(name, seed, tr, va, te,
                 ["associated_with_disease" if name == "t4_b" else "treats_disease"], negatives=neg)
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=ROOT / "benchmark_data")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()
    x, nodes, rels, stats = encode_kg(args.root, args.out)
    splits = prepare_splits(args.out, x, nodes, rels, args.seeds)
    manifest = dict(protocol=PROTOCOL, seeds=args.seeds, core_settings=CORE,
                    optional_settings=OPTIONAL, original_edges_sha256=stats["original_edges_sha256"],
                    files={str(p.relative_to(args.out)).replace("\\", "/"): sha256(p)
                           for p in sorted(args.out.rglob("*")) if p.is_file() and p.name != "manifest.json"},
                    splits=splits)
    write_json(args.out / "manifest.json", manifest)
    print("Prepared", len(splits), "audited splits at", args.out, flush=True)


if __name__ == "__main__":
    main()
