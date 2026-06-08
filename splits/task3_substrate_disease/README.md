# Task 3 v2: Substrate-Utilization Inference for Substrate->Disease Discovery

## Why this replaces the old derived-pair Task 3
`derived_pair(S,D) = can_utilize(M,S) AND DA(M,D)` is a closed-form AND of two
edges that always live in `final_kg_edges.tsv`. Splitting the pairs cannot make it
a task -- the model composes the bridge edges back (0.93 H@10 = closure
reconstruction). We instead REMOVE `can_utilize` edges so the model must predict
the missing substrate-utilization edge before composing (S,D). Inference, not
reconstruction. This makes the mediator bottleneck the target: only 579
microbes have both can_utilize and DA; 8,106 have DA but NO
can_utilize.

## Training graph
`train = final_kg_edges.tsv  MINUS  <setting>/removed_can_utilize.tsv` (we output
removed edges, not a copy of the KG).

## Setting A -- transductive can_utilize completion (ceiling / control)
Per-microbe edge hold-out; each mediator keeps >=1 can_utilize edge in train.
Matrix completion; expected high. seed=42.

## Setting B -- relation-level zero-shot can_utilize (the real task)
**Naming**: NOT unseen-node inductive. The cold microbe is still in the graph via
its DA + taxonomy edges (it has an embedding); only the `can_utilize` relation is
zero-shot for it. This matches the real scenario (the 8,106
target microbes live in the graph through their DA edges).

All can_utilize edges of 116 test + 58 valid mediator
microbes are removed; target = recover them; downstream = compose (S,D).

### Genuine-discovery filter (candidate pool)
Of 106,439 test (S,D), only 18,499
(17.4%) are NOT reachable via any other
training microbe -- the genuine discovery positives (no leakage by construction).
The other 82.6% are
trivial (a redundant mediator stays in train) and are EXCLUDED entirely -- neither
positive nor negative (marked `genuine_discovery=no`).

**Ranking protocol (write this into any eval)**: per disease with >=1 genuine
positive, rank ALL substrate candidates.
- positives = genuine (S,D) for that disease
- excluded = trivial closure-reachable substrates (removed from the pool)
- negatives = all_substrates - positives - trivial (FULL ranking, no synthetic
  1:k down-sampling)
Real imbalance: pos=18,499 vs neg=183,996 (~1:9)
over 317 diseases. Report MRR / Hits@k AND AUPRC at this true
imbalance; if a sampled negative ratio is ever used, state it explicitly.

### Taxonomic-shortcut probes (do not skip)
The 116 cold microbes come from the 579 mediators =
AGORA2-covered, common, cultured taxa with siblings in the graph. A model can
cheat via genus-conserved substrate profiles instead of inferring. Two probes:
- `cold_microbe_tax_proximity.tsv` -- nearest rank at which each cold microbe
  shares a taxon with a train microbe that still has can_utilize. Stratify metrics
  by this; the **`none`** layer is the honest proxy for the isolated
  8,106. Current test split distribution: genus:40 / family:21 / order:2 / class:1 / phylum:0 / none:52.
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
