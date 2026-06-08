"""Build Task 3 v2 (Substrate-Utilization Inference for Substrate->Disease Discovery).

REFRAME (2026-05-29). The old "derived (S,D) pair" target was a closed-form AND
of two bridge edges that always sit in final_kg_edges.tsv:

    derived_pair(S, D)  <=  exists microbe M.
        can_utilize(M, S) AND (enriched_in(M, D) OR depleted_in(M, D))

Splitting the PAIRS (random or disease-grouped) cannot make this a real task: the
two bridge edges stay in the training graph, so the model just composes them back
(IBD H@10 = 0.93 was this closure reconstruction). To kill the closure we REMOVE a
bridge edge: Task 3 v2 holds out can_utilize (microbe->substrate) edges, so the
model must FIRST predict the missing substrate-utilization edge and only THEN
compose (S,D). That is inference, not reconstruction.

This turns the mediator bottleneck into the discovery target: only ~580 microbes
have BOTH can_utilize and DA edges; ~8,100 have a DA edge but NO can_utilize edge.

Settings (main metric, existing data):
  A. transductive can_utilize completion -- random edge hold-out; the microbe keeps
     >=1 can_utilize edge in train. Matrix completion; transductive ceiling.
  B. relation-level zero-shot can_utilize (partially-observed node) -- hold out ALL
     can_utilize edges of a set of mediator microbes. The microbe is STILL in the
     graph (DA + taxonomy edges, so it has an embedding); only the can_utilize
     relation is zero-shot for it. This is NOT unseen-node inductive -- naming
     matters for review. It mimics the ~8,100 target microbes (DA-only in graph).

KNOWN BIAS (probe, do not hide): the 116/58 cold microbes can only be drawn from
the 579 mediators = AGORA2-covered taxa, which are common, cultured, and have
taxonomic siblings in the training graph. A model can shortcut via "a same-genus
sibling utilizes S, so do I" (AGORA2 substrate profiles are genus-conserved)
rather than truly inferring. That shortcut does NOT exist for the isolated 8,100,
so scores are optimistically biased. Mitigations emitted here:
  - cold_microbe_tax_proximity.tsv: nearest shared rank (genus..phylum / none) to
    a train microbe that still has can_utilize. Stratify metrics by this; the
    `none` layer is the honest proxy for the 8,100.
  - taxonomy_edges.tsv: all microbe-microbe taxonomy edges, so a taxonomy-ablated
    run (train minus these) measures how much signal is genome-level inference vs
    taxonomic copying.

Discovery metric candidate pool (see README): positives = genuine (S,D) only
(closure-reachable pairs are trivial -> excluded entirely, neither pos nor neg);
negatives = full substrate candidate space minus positives minus trivial, ranked
full (no synthetic 1:k sampling).

External gold (Setting C, headline) -- see external_gold/README.md.

Training graph for A/B: final_kg_edges.tsv MINUS <setting>/removed_can_utilize.tsv.
"""
from __future__ import annotations

import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from split_utils import SPLITS_DIR, EDGE_COLS, load_final_edges, write_tsv

TASK3_DIR = SPLITS_DIR / "task3_substrate_disease"
SEED = 42
DA_RELS = ["enriched_in", "depleted_in"]
# closest -> farthest; first shared level is the taxonomic distance bucket
RANK_RELS = [
    ("belongs_to_genus", "genus"),
    ("belongs_to_family", "family"),
    ("belongs_to_order", "order"),
    ("belongs_to_class", "class"),
    ("belongs_to_phylum", "phylum"),
]
TAX_RELS = {r for r, _ in RANK_RELS} | {"is_strain_of", "is_clade_of"}
EXTERNAL_GOLD_SRC = (
    Path(__file__).resolve().parent.parent
    / "original_data" / "external_gold" / "prebiotic_disease_gold.tsv"
)


def index_bridges(edges):
    cu = edges[(edges.relation == "can_utilize")
               & (edges.head_type == "microbe")
               & (edges.tail_type == "substrate")].reset_index(drop=True)
    m2s = defaultdict(set)
    for h, t in zip(cu.head_id, cu.tail_id):
        m2s[h].add(t)
    da = edges[(edges.relation.isin(DA_RELS))
               & (edges.head_type == "microbe")
               & (edges.tail_type == "disease")]
    m2d = defaultdict(lambda: defaultdict(set))
    for h, t, r in zip(da.head_id, da.tail_id, da.relation):
        m2d[h][t].add("enriched" if r == "enriched_in" else "depleted")
    return cu, m2s, m2d


def extract_lineage(edges):
    """microbe -> {rank: tax_node_id} from belongs_to_* edges."""
    lineage = defaultdict(dict)
    for rel, rank in RANK_RELS:
        sub = edges[(edges.relation == rel)
                    & (edges.head_type == "microbe")
                    & (edges.tail_type == "microbe")]
        for h, t in zip(sub.head_id, sub.tail_id):
            lineage[h][rank] = t
    return lineage


def tax_proximity(cold_ms, train_canutil, lineage):
    """For each cold microbe, the nearest rank at which it shares a taxon node with
    SOME training microbe that still has a can_utilize edge. 'none' = no neighbour."""
    train_nodes = {rank: set() for _, rank in RANK_RELS}
    for m in train_canutil:
        lin = lineage.get(m, {})
        for _, rank in RANK_RELS:
            if rank in lin:
                train_nodes[rank].add(lin[rank])
    rows, bucket = [], Counter()
    for m in sorted(cold_ms):
        lin = lineage.get(m, {})
        nearest = "none"
        for _, rank in RANK_RELS:
            node = lin.get(rank)
            if node and node in train_nodes[rank]:
                nearest = rank
                break
        bucket[nearest] += 1
        rows.append({"microbe": m, "nearest_shared_rank": nearest})
    return pd.DataFrame(rows, columns=["microbe", "nearest_shared_rank"]), bucket


def compose_sd(microbes, m2s, m2d):
    sd = defaultdict(set)
    for m in microbes:
        subs = m2s.get(m)
        dis = m2d.get(m)
        if not subs or not dis:
            continue
        for s in subs:
            for d in dis:
                sd[(s, d)].add(m)
    return sd


def sd_rows(sd_pairs, m2d, genuine_set=None):
    rows = []
    for (s, d), ms in sorted(sd_pairs.items()):
        ms_sorted = sorted(ms)
        dirs = set()
        for m in ms:
            dirs |= m2d[m].get(d, set())
        ev = (f"mediating_microbes={','.join(ms_sorted)}"
              f"|n_microbes={len(ms_sorted)}"
              f"|directions={';'.join(sorted(dirs))}")
        if genuine_set is not None:
            ev += f"|genuine_discovery={'yes' if (s, d) in genuine_set else 'no'}"
        rows.append({
            "head_id": s, "head_type": "substrate",
            "relation": "derived_substrate_disease_link",
            "tail_id": d, "tail_type": "disease",
            "confidence": "derived_2hop", "species_source": "human",
            "source": "derived_2hop", "evidence": ev, "evidence_type": "derived",
        })
    return pd.DataFrame(rows, columns=EDGE_COLS)


def setting_a(cu, mediators, m2d):
    out = TASK3_DIR / "setting_a_transductive_completion"
    rng = random.Random(SEED)
    cu_med = cu[cu.head_id.isin(mediators)].reset_index(drop=True)
    by_m = defaultdict(list)
    for pos, h in enumerate(cu_med.head_id):
        by_m[h].append(pos)
    valid_pos, test_pos = [], []
    for m, idxs in by_m.items():
        rng.shuffle(idxs)
        for ix in idxs[1:]:
            r = rng.random()
            if r < 0.10:
                valid_pos.append(ix)
            elif r < 0.20:
                test_pos.append(ix)
    valid_df = cu_med.iloc[valid_pos].reset_index(drop=True)
    test_df = cu_med.iloc[test_pos].reset_index(drop=True)
    write_tsv(pd.concat([valid_df, test_df], ignore_index=True), out / "removed_can_utilize.tsv")
    write_tsv(valid_df, out / "valid.tsv")
    write_tsv(test_df, out / "test.tsv")
    test_m2s = defaultdict(set)
    for h, t in zip(test_df.head_id, test_df.tail_id):
        test_m2s[h].add(t)
    sd = compose_sd(set(test_m2s), test_m2s, m2d)
    write_tsv(sd_rows(sd, m2d), out / "downstream_sd_test.tsv")
    print(f"  Setting A (transductive): valid={len(valid_df):,} test={len(test_df):,} "
          f"can_utilize held; downstream (S,D)={len(sd):,}")


def setting_b(edges, cu, mediators, m2s, m2d, lineage, n_substrates):
    out = TASK3_DIR / "setting_b_relzeroshot"
    rng = random.Random(SEED)
    meds = sorted(mediators)
    rng.shuffle(meds)
    n = len(meds)
    n_test, n_valid = int(round(n * 0.20)), int(round(n * 0.10))
    test_m = set(meds[:n_test])
    valid_m = set(meds[n_test:n_test + n_valid])
    train_m = set(meds[n_test + n_valid:])
    held = test_m | valid_m

    write_tsv(cu[cu.head_id.isin(held)].reset_index(drop=True), out / "removed_can_utilize.tsv")
    test_edges = cu[cu.head_id.isin(test_m)].reset_index(drop=True)
    valid_edges = cu[cu.head_id.isin(valid_m)].reset_index(drop=True)
    write_tsv(test_edges, out / "test.tsv")
    write_tsv(valid_edges, out / "valid.tsv")
    (out / "test_microbes.txt").write_text("\n".join(sorted(test_m)), encoding="utf-8")
    (out / "valid_microbes.txt").write_text("\n".join(sorted(valid_m)), encoding="utf-8")

    # genuine-discovery filter
    train_closure = set(compose_sd(train_m, m2s, m2d).keys())
    test_sd = compose_sd(test_m, m2s, m2d)
    valid_sd = compose_sd(valid_m, m2s, m2d)
    test_genuine = {k for k in test_sd if k not in train_closure}
    valid_genuine = {k for k in valid_sd if k not in train_closure}
    write_tsv(sd_rows(test_sd, m2d, test_genuine), out / "downstream_sd_test.tsv")
    write_tsv(sd_rows(valid_sd, m2d, valid_genuine), out / "downstream_sd_valid.tsv")

    # PROBE 1: taxonomic proximity of cold microbes to train microbes with can_utilize
    train_canutil = set(m2s) - held          # microbes still holding can_utilize in train
    prox_df, bucket = tax_proximity(test_m, train_canutil, lineage)
    write_tsv(prox_df, out / "cold_microbe_tax_proximity.tsv")

    # PROBE 2: taxonomy edges to ablate (train minus these = genome-level only)
    tax = edges[(edges.relation.isin(TAX_RELS))
                & (edges.head_type == "microbe") & (edges.tail_type == "microbe")]
    write_tsv(tax.reset_index(drop=True), out / "taxonomy_edges.tsv")

    # candidate pool imbalance for the genuine-discovery ranking (per disease)
    pos_by_d, triv_by_d = defaultdict(set), defaultdict(set)
    for (s, d) in test_genuine:
        pos_by_d[d].add(s)
    for (s, d) in train_closure:
        triv_by_d[d].add(s)
    tot_pos = tot_neg = 0
    for d, pos in pos_by_d.items():
        triv = triv_by_d.get(d, set())
        tot_pos += len(pos)
        tot_neg += max(n_substrates - len(pos) - len(triv), 0)

    def pct(a, b):
        return f"{100 * a / b:.1f}%" if b else "-"
    print(f"  Setting B (relation-level zero-shot): microbes train={len(train_m)} "
          f"valid={len(valid_m)} test={len(test_m)}")
    print(f"    held can_utilize: test={len(test_edges):,} valid={len(valid_edges):,}")
    print(f"    test downstream (S,D): {len(test_sd):,} total; "
          f"genuine (not in train closure): {len(test_genuine):,} "
          f"({pct(len(test_genuine), len(test_sd))})")
    print(f"    cold-microbe tax proximity (nearest train-canutil neighbour): "
          f"{dict(bucket)}")
    print(f"    discovery ranking pool: pos={tot_pos:,} neg={tot_neg:,} "
          f"(imbalance ~1:{tot_neg // max(tot_pos, 1)}) over {len(pos_by_d)} diseases")
    return dict(n_train=len(train_m), n_valid=len(valid_m), n_test=len(test_m),
                test_edges=len(test_edges), test_sd=len(test_sd),
                test_genuine=len(test_genuine), tax_bucket=dict(bucket),
                pos=tot_pos, neg=tot_neg, n_disease=len(pos_by_d))


def external_gold():
    out = TASK3_DIR / "external_gold"
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(
        "# Task 3 Setting C: external gold (prebiotic -> disease)\n\n"
        "Headline validation with a label INDEPENDENT of the two KG bridge edges.\n"
        "The KG path is a feature only; the (substrate, disease) label comes from\n"
        "real prebiotic intervention evidence.\n\n"
        "## Required file\n"
        "`original_data/external_gold/prebiotic_disease_gold.tsv` (tab-separated):\n"
        "substrate_id, disease_id, direction(beneficial/harmful), source(PMID), evidence\n\n"
        "## Sources\n"
        "Prebiotic RCT literature reporting substrate->disease therapeutic pairs:\n"
        "inulin / resistant starch / GOS / FOS / pectin / beta-glucan -> IBD / T2D /\n"
        "CRC / obesity / constipation.\n\n"
        "## Use\n"
        "Positives = these airtight pairs. Model trained on A/B predicts can_utilize,\n"
        "composes (S,D), scored against this gold. Currently ABSENT (TODO: literature\n"
        "extraction); this split is generated automatically once the file exists.\n",
        encoding="utf-8")
    if not EXTERNAL_GOLD_SRC.exists():
        print("  Setting C: no gold file -> wrote spec README only (TODO: lit extraction).")
        return
    gold = pd.read_csv(EXTERNAL_GOLD_SRC, sep="\t", dtype=str, keep_default_na=False)
    rng = random.Random(SEED)
    idx = list(range(len(gold)))
    rng.shuffle(idx)
    cut = int(round(len(idx) * 0.5))
    write_tsv(gold.iloc[idx[:cut]], out / "valid.tsv")
    write_tsv(gold.iloc[idx[cut:]], out / "test.tsv")
    print(f"  Setting C: external gold {len(gold):,} pairs -> valid/test 50/50")


def write_readme(s):
    tax = s["tax_bucket"]
    tax_line = " / ".join(f"{k}:{tax.get(k, 0)}" for k in
                          ["genus", "family", "order", "class", "phylum", "none"])
    readme = f"""# Task 3 v2: Substrate-Utilization Inference for Substrate->Disease Discovery

## Why this replaces the old derived-pair Task 3
`derived_pair(S,D) = can_utilize(M,S) AND DA(M,D)` is a closed-form AND of two
edges that always live in `final_kg_edges.tsv`. Splitting the pairs cannot make it
a task -- the model composes the bridge edges back (0.93 H@10 = closure
reconstruction). We instead REMOVE `can_utilize` edges so the model must predict
the missing substrate-utilization edge before composing (S,D). Inference, not
reconstruction. This makes the mediator bottleneck the target: only {s['mediators']}
microbes have both can_utilize and DA; {s['target_microbes']:,} have DA but NO
can_utilize.

## Training graph
`train = final_kg_edges.tsv  MINUS  <setting>/removed_can_utilize.tsv` (we output
removed edges, not a copy of the KG).

## Setting A -- transductive can_utilize completion (ceiling / control)
Per-microbe edge hold-out; each mediator keeps >=1 can_utilize edge in train.
Matrix completion; expected high. seed={SEED}.

## Setting B -- relation-level zero-shot can_utilize (the real task)
**Naming**: NOT unseen-node inductive. The cold microbe is still in the graph via
its DA + taxonomy edges (it has an embedding); only the `can_utilize` relation is
zero-shot for it. This matches the real scenario (the {s['target_microbes']:,}
target microbes live in the graph through their DA edges).

All can_utilize edges of {s['n_test']} test + {s['n_valid']} valid mediator
microbes are removed; target = recover them; downstream = compose (S,D).

### Genuine-discovery filter (candidate pool)
Of {s['test_sd']:,} test (S,D), only {s['test_genuine']:,}
({100*s['test_genuine']/max(s['test_sd'],1):.1f}%) are NOT reachable via any other
training microbe -- the genuine discovery positives (no leakage by construction).
The other {100*(s['test_sd']-s['test_genuine'])/max(s['test_sd'],1):.1f}% are
trivial (a redundant mediator stays in train) and are EXCLUDED entirely -- neither
positive nor negative (marked `genuine_discovery=no`).

**Ranking protocol (write this into any eval)**: per disease with >=1 genuine
positive, rank ALL substrate candidates.
- positives = genuine (S,D) for that disease
- excluded = trivial closure-reachable substrates (removed from the pool)
- negatives = all_substrates - positives - trivial (FULL ranking, no synthetic
  1:k down-sampling)
Real imbalance: pos={s['pos']:,} vs neg={s['neg']:,} (~1:{s['neg']//max(s['pos'],1)})
over {s['n_disease']} diseases. Report MRR / Hits@k AND AUPRC at this true
imbalance; if a sampled negative ratio is ever used, state it explicitly.

### Taxonomic-shortcut probes (do not skip)
The {s['n_test']} cold microbes come from the {s['mediators']} mediators =
AGORA2-covered, common, cultured taxa with siblings in the graph. A model can
cheat via genus-conserved substrate profiles instead of inferring. Two probes:
- `cold_microbe_tax_proximity.tsv` -- nearest rank at which each cold microbe
  shares a taxon with a train microbe that still has can_utilize. Stratify metrics
  by this; the **`none`** layer is the honest proxy for the isolated
  {s['target_microbes']:,}. Current test split distribution: {tax_line}.
- `taxonomy_edges.tsv` -- run a taxonomy-ablated B (train minus these edges); the
  gap to the full-graph B is the share of score that is taxonomic copying vs
  genome-level inference.

## Setting C -- external gold (headline)
Real prebiotic->disease pairs, label independent of the bridge edges. See
`external_gold/README.md`. Generated when the gold file is supplied.

## Evaluation summary
1. can_utilize recovery (primary): MRR / Hits@k over removed edges, **stratified by
   tax-proximity bucket**; the `none` layer is the headline. A = transductive
   ceiling. Far below the old 0.93 -- that drop is the closure inflation removed.
2. downstream genuine (S,D) recovery at the true imbalance above.
3. taxonomy-ablated B vs full B = taxonomic-copying share.
4. external gold (Setting C) when available.
"""
    (TASK3_DIR / "README.md").write_text(readme, encoding="utf-8")


def main():
    print("Loading final KG edges...")
    edges = load_final_edges()
    print(f"  loaded {len(edges):,} edges")

    cu, m2s, m2d = index_bridges(edges)
    lineage = extract_lineage(edges)
    n_substrates = edges.loc[edges.tail_type == "substrate", "tail_id"].nunique()
    canutil_microbes, da_microbes = set(m2s), set(m2d)
    mediators = canutil_microbes & da_microbes
    target_microbes = da_microbes - canutil_microbes
    print(f"  can_utilize microbes: {len(canutil_microbes):,}")
    print(f"  DA microbes: {len(da_microbes):,}")
    print(f"  mediators (both, leakage source): {len(mediators):,}")
    print(f"  target microbes (DA but NO can_utilize, discovery goal): {len(target_microbes):,}")
    print(f"  substrate candidate pool: {n_substrates:,}")

    for old in ("primary", "disease_grouped", "setting_b_cold_microbe"):
        p = TASK3_DIR / old
        if p.exists():
            shutil.rmtree(p)
            print(f"  removed stale dir: {old}/")

    print("\n>>> Setting A (transductive completion)")
    setting_a(cu, mediators, m2d)
    print("\n>>> Setting B (relation-level zero-shot)")
    b = setting_b(edges, cu, mediators, m2s, m2d, lineage, n_substrates)
    print("\n>>> Setting C (external gold)")
    external_gold()

    write_readme(dict(mediators=len(mediators), target_microbes=len(target_microbes), **b))
    print("\nTask 3 v2 splits written to", TASK3_DIR)


if __name__ == "__main__":
    main()
