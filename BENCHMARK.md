# MicrobeKG — Benchmark

Four tasks turn the graph into a KGC benchmark that stresses **generalization,
transfer, and mechanism**, not closure memorization. Baseline numbers are in
[RESULTS.md](RESULTS.md); the released splits live under `splits/`.

## Design principles

- **Anti-shortcut training graphs.** Each task trains on
  `G_train = KG − <leakage edges>`. Direct shortcut edges (e.g. metabolite→disease,
  host-gene closure, or a relation's own gold) are scrubbed so a model cannot
  recover a target by composing 2–3 hops.
- **Explicit hard negatives**, not random sampling: `inconsistent_association`,
  `does_not_utilize`, `does_not_produce`, and an "associated-but-not-treating" pool.
- **Stratified evaluation**: by evidence type, body-site, and **taxonomy
  proximity** (`genus / family / none`) — the `none` layer is the honest signal.
- **Leakage audited**: closure inflation and taxonomic copying are measured, not
  assumed away.
- **Fixed seeds**; Task 1 transductive ships 5 seeds, others seed 42.

| Task | Target | Settings | Core question |
|---|---|---|---|
| **1** Microbe→Disease | `enriched_in` / `depleted_in` | transductive · cold-microbe · cold-disease | Recover / generalize microbe–disease associations |
| **2** Capacity→Realization | `utilizes` | A within-relation · B zero-shot · C few-shot | Does predicted capacity transfer to observed realization? |
| **3** Substrate→Disease | `can_utilize` recovery → (S,D) | A transductive · B relation-zero-shot | Infer missing utilization, then discover substrate–disease links |
| **4** Metabolite→Disease | `treats_disease` | A transductive · B cross-evidence · C cold-metabolite | Distinguish therapy from biomarker association |

---

## Task 1 — Microbe→Disease association

**Predicate space.** Positives are **microbe→disease directional** edges only:
**57,377** (`enriched_in` 30,672 + `depleted_in` 26,705). Hard negatives:
`inconsistent_association` **2,552**. Train-only auxiliary: **72,925** (taxonomy +
`is_a` + `co_occurs_with`). **Excluded** (anti-shortcut): all microbe–substrate /
microbe–metabolite / metabolite–disease / host-gene / intervention /
`treats_disease` / `therapeutic_target_for` edges — this prevents 2/3-hop leakage.
`associated_with_disease` is excluded because in this KG it is host-gene→disease or
metabolite→disease, never microbe→disease.

| Paradigm | Rule | Seeds |
|---|---|--:|
| **Transductive** | Split unit = unique `(head_id, tail_id, body_site_uberon, evidence_type)`, 80/10/10; hard-neg split separately → `test_hardneg.tsv` | 5 |
| **Cold-microbe** | Hold out 10% of microbes with positives; their edges → 50/50 valid/test. Taxonomy parents stay in train (structural signal) | 3 |
| **Cold-disease** | Hold out 10% of D-coded MeSH diseases that have an `is_a` parent; parents stay in train | 3 |

**Reporting:** MRR / Hits@{1,3,5,10,20} (micro + macro), stratified by evidence
type and body-site; hard-negative **AUROC** on `test_hardneg.tsv` (positives vs
`inconsistent_association`).

---

## Task 2 — Capacity→Realization transfer

Capacity = `can_utilize` (1,501,799, AGORA2 genome-scale, *computational*).
Realization = `utilizes` (3,609, *experimental*). The question is whether a model
trained on predicted capacity transfers to observed realization.

| Setting | Train | Valid / test | Purpose |
|---|---|---|---|
| **A** within-relation | 90% of `can_utilize` | held-out `can_utilize` | Same-relation **ceiling** (not the task) |
| **B** zero-shot *(flagship)* | `can_utilize` (self-pair scrubbed) + taxonomy | test = **all** `utilizes`; hard-neg = `does_not_utilize` | Capacity→realization gap |
| **C** few-shot | B + N injected `utilizes` hints | same transfer test | Adaptation curve, N = 50 / 100 / 500 |

**Taxonomic-shortcut probe.** `test_microbe_tax_proximity.tsv` buckets each test
microbe by the nearest rank sharing a taxon with a train `can_utilize` microbe.
The **`none`** layer is the honest transfer signal (elsewhere the model can copy a
same-genus microbe's capacity). `does_not_utilize` is tiny (146) — report its `n`
and the negative ratio explicitly; do not report AUPRC at a hidden imbalance.

---

## Task 3 — Substrate→Disease discovery (v2)

**Why the old derived-pair task was replaced.** `derived_pair(S,D) =
can_utilize(M,S) ∧ DA(M,D)` is a closed-form AND of two edges already in the KG —
splitting the pairs only tests **closure reconstruction** (≈0.93 H@10). Instead we
**remove `can_utilize` edges** so the model must *infer* the missing utilization
edge before composing (S,D). The mediator bottleneck is the point: only **579**
microbes have both `can_utilize` and a disease association; **8,106** have a
disease association but **no** `can_utilize`.

`train = final_kg_edges − <setting>/removed_can_utilize.tsv`.

| Setting | Rule | Role |
|---|---|---|
| **A** transductive completion | per-microbe hold-out, each mediator keeps ≥1 `can_utilize` | ceiling / control |
| **B** relation-level zero-shot *(real task)* | remove **all** `can_utilize` of 116 test + 58 valid mediator microbes; recover them, then compose (S,D) | inference, not reconstruction |

**Genuine-discovery filter.** Of **106,439** test (S,D) pairs, only **18,499**
(17.4%) are *not* reachable via any other training microbe — the genuine-discovery
positives (no leakage by construction). The trivial 82.6% are excluded entirely.
Downstream ranking is at the **true imbalance**: pos **18,499** vs neg **183,996**
(~1:9) over **317** diseases — report MRR / Hits@k **and** AUPRC at this imbalance.

**Taxonomic-shortcut probe.** `cold_microbe_tax_proximity.tsv` buckets the 116 cold
microbes (genus 40 / family 21 / order 2 / class 1 / **none 52**). The **`none`**
layer is the honest proxy for the isolated 8,106. A taxonomy-ablated B (train minus
taxonomy edges) measures the share of score that is taxonomic copying vs inference.

---

## Task 4 — Metabolite→Disease therapeutic prediction

Target `treats_disease` (6,285, CTD curated). This is the **insurance for Task 3**:
curated gold + real mechanism; setting A is standard link prediction with no
closure trap.

`train_graph = final_kg_edges − <setting>/gtrain_removed.tsv`, where
`gtrain_removed.tsv` scrubs **all direct** metabolite→disease edges on each held
pair (treats / assoc / enriched / depleted). The mechanism bridge
(metabolite→gene→disease) is **kept** — a different node pair, so it survives the
pair-scrub; this is mechanism, not leakage.

| Setting | Rule |
|---|---|
| **A** transductive *(main)* | `treats` 80/10/10; standard link prediction |
| **B** cross-evidence zero-shot | train supervision = enriched/depleted/assoc; test = all `treats` (maps association/mechanism → therapy) |
| **C** cold-metabolite | hold out metabolites; their `treats` = test (chemical-class stratification TODO) |

**Hard negatives.** `hardneg_pool_associated_not_treating.tsv` (199,230 pairs):
(M,D) with enriched/depleted/assoc but **no** `treats` → biomarker-vs-therapy
discrimination. Sample at a stated ratio and report the true imbalance.

---

## Recommended reporting (all tasks)

1. **Ranking** — full filtered rank over the same-type candidate pool, both
   directions, micro **and** macro, MRR + Hits@{1,3,5,10,20} + bootstrap CI.
2. **Discrimination** — hard-negative **AUROC** (primary on positive-heavy sets)
   and **AUPRC + prevalence floor** (primary on negative-heavy sets); always state
   `n+ : n−`. See [RESULTS.md](RESULTS.md) for why the metric flips by set.
3. **Stratified** — by evidence type / body-site (Task 1) and by **tax-proximity
   `none` layer** (Tasks 2/3).
4. **Seed** — report seed; Task 1 transductive averages 5 seeds.
