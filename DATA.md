# MicrobeKG — Data

MicrobeKG v3.3 integrates **19 sources** into a single evidence-annotated graph of
**69,090 nodes**, **3,687,015 edges**, **6 node types**, and **25 relations**.
Every edge keeps its provenance, confidence, free-text evidence, and a primary
evidence class, so models can be evaluated *by source, body-site, evidence type,
and confidence* — not just as `(head, relation, tail)` triples.

All counts on this page come from the authoritative build report
`kg_build/reports/final_kg_report.txt`.

---

## 1. Data sources

| # | Source | Edges* | Main contribution |
|--:|---|--:|---|
| 1 | **AGORA2** | 1,501,799 | Genome-scale metabolic reconstructions → microbe–substrate metabolic **capacity** (`can_utilize`) |
| 2 | **GMMAD** | 1,048,789 | Microbe–metabolite **production** and association (`produces`, `associated_with_metabolite`) |
| 3 | **HMDB** | 851,347 | Human Metabolome DB → metabolite knowledge and metabolite–disease links |
| 4 | **CTD** | 144,419 | Comparative Toxicogenomics DB → chemical/gene–disease + chemical–gene regulation (`up/downregulates_gene`, `treats_disease`, `associated_with_disease`, `therapeutic_target_for`) |
| 5 | **NCBI Taxonomy** | 58,297 | Microbial taxonomic hierarchy (`belongs_to_*`, `is_strain_of`, `is_clade_of`) |
| 6 | **GMrepo_v3** | 23,719 | Cohort-derived microbe–disease enrichment / depletion |
| 7 | **BugSigDB** | 21,731 | Curated published microbial signatures (and cross-study conflicts) |
| 8 | **mBodyMap** | 8,252 | Body-site-resolved microbiome–disease evidence |
| 9 | **MeSH** | 8,076 | Disease ontology hierarchy (`is_a`) |
| 10 | **Disbiome** | 6,918 | Curated disease–microbe associations |
| 11 | **NJC19** | 5,037 | Experimental substrate utilization incl. explicit negatives (`utilizes`, `does_not_utilize`, `does_not_produce`) |
| 12 | **gutMGene** | 4,788 | Microbe–metabolite and host-gene regulation evidence |
| 13 | **Peryton** | 4,060 | Curated disease–microbe associations |
| 14 | **Lit44** | 3,701 | Literature-curated microbiome–disease evidence (44-study set) |
| 15 | **gutMDisorder_v2** | 2,415 | Gut microbiota–disorder associations |
| 16 | **cross_source_conflict** | 1,304 | *Derived:* edges where sources disagree, materialized as `inconsistent_association` hard negatives |
| 17 | **HMDAD** | 407 | Human Microbe–Disease Association DB (curated) |
| 18 | **Piccinno_2025** | 191 | SGB-level (species-genome-bin) disease–microbe associations |
| 19 | **Thomas_2019** | 52 | Colorectal-cancer microbiome meta-analysis |

\* *Per-source occurrence counts. A multi-source edge is counted under every
contributing source, so this column sums to more than the de-duplicated
3,687,015 total.*

---

## 2. Node inventory (6 types)

| Node type | Count | Role |
|---|--:|---|
| `metabolite` | 25,184 | Compounds **produced** by microbes; mechanistic bridge to host phenotypes |
| `host_gene` | 21,008 | Host molecular context (CTD chemical–gene–disease mechanism layer) |
| `microbe` | 16,502 | Central biological entity in every task |
| `disease` | 5,256 | Prediction / ranking targets |
| `substrate` | 1,018 | Nutrient / compound **consumed** by microbes (capacity input, Task 3 target) |
| `intervention` | 122 | Probiotic / FMT / dietary perturbation evidence |

> **`substrate` and `metabolite` are deliberately separate node types** —
> *consumed* vs *produced* semantics — even though 167 compounds play both roles.
> They are **not** silently merged.

---

## 3. Relation inventory (25 relations)

| Group | Relation | Edges |
|---|---|--:|
| **Metabolic capacity & production** | `can_utilize` | 1,501,799 |
| | `produces` | 860,841 |
| | `associated_with_metabolite` | 851,444 |
| | `utilizes` *(experimental realization)* | 3,609 |
| | `does_not_utilize` *(hard-neg)* | 146 |
| | `does_not_produce` *(hard-neg)* | 17 |
| **Microbe–disease association** | `enriched_in` | 129,936 |
| | `depleted_in` | 115,199 |
| | `inconsistent_association` *(hard-neg)* | 2,552 |
| **Chemical / gene–disease knowledge** | `upregulates_gene` | 52,387 |
| | `downregulates_gene` | 45,598 |
| | `associated_with_disease` | 41,266 |
| | `treats_disease` | 6,285 |
| | `therapeutic_target_for` | 2,237 |
| **Microbe taxonomy** | `belongs_to_phylum` | 10,725 |
| | `belongs_to_order` | 10,634 |
| | `belongs_to_class` | 10,632 |
| | `belongs_to_family` | 10,504 |
| | `belongs_to_genus` | 10,196 |
| | `is_strain_of` | 5,606 |
| | `is_clade_of` | 83 |
| **Ontology / ecology / intervention** | `is_a` *(disease ontology)* | 8,076 |
| | `co_occurs_with` | 6,469 |
| | `intervention_increases` | 431 |
| | `intervention_decreases` | 343 |

### Hard-negative relations (the anti-inflation core)

Standard microbiome KGs rely on random negative sampling. MicrobeKG ships
**explicit, biologically grounded negatives** so discrimination is non-trivial:

| Relation | Edges | Meaning |
|---|--:|---|
| `inconsistent_association` | 2,552 | Same microbe–disease pair with **conflicting** direction across studies/sources |
| `does_not_utilize` | 146 | Experimentally confirmed **non**-utilization (NJC19) |
| `does_not_produce` | 17 | Experimentally confirmed **non**-production (NJC19) |
| **Total hard-negative pool** | **2,715** | |

---

## 4. Evidence composition

Each edge carries a primary `evidence_type`. When an edge is supported by several
sources, the tag is **pipe-joined** with this precedence:
`experimental > curated_database > cohort_statistical > computational > ontology`.

| Evidence class | Edges |
|---|--:|
| `computational` | 2,548,423 |
| `curated_database` | 995,733 |
| `ontology` | 66,373 |
| `experimental` | 45,987 |
| `cohort_statistical` | 27,103 |
| mixed (pipe-joined tags) | 3,396 |

The large `computational` mass is the AGORA2 capacity layer + GMMAD/HMDB
association layers; `experimental` and `cohort_statistical` are the curated and
cohort-derived microbe–disease and utilization evidence used for evaluation.

---

## 5. Edge schema (10 columns)

Every KG and split TSV uses the same evidence-aware schema:

| Column | Meaning |
|---|---|
| `head_id` | Source node identifier |
| `head_type` | Source node type |
| `relation` | Edge predicate (one of the 25 relations) |
| `tail_id` | Target node identifier |
| `tail_type` | Target node type |
| `confidence` | Source-specific confidence or normalized score |
| `species_source` | Species / taxonomy provenance when available |
| `source` | Database or construction source |
| `evidence` | Source evidence, support text, or derived-path metadata |
| `evidence_type` | Primary evidence class (see §4) |

Keeping more than `(head, relation, tail)` is what enables **stratified**
evaluation (by source, body-site, evidence class, confidence) and the
leakage audits in [BENCHMARK.md](BENCHMARK.md).

---

## 6. Identifier conventions

| Entity | ID scheme (priority) |
|---|---|
| **Disease** | bare `D######` (MeSH) · `OMIM:###` · `C######` (MeSH supplementary) |
| **Microbe** | `NCBI_TAXON:<int>` · `SGB:<int>` (Piccinno) · `VMH:<model>` · `UNMAPPED:<name>` |
| **Metabolite / Substrate** | `HMDB > CHEBI > PUBCHEM > KEGG > VMH_MET > MESH > GMNAME` (first available wins) |
| **Host gene** | NCBI Gene / CTD gene identifiers |

---

## 7. Provenance & reproducibility

- Raw downloads live under `original_data/` (16 source downloads); per-source
  cleaning under `preprocessed/`.
- Per-source edge builders: `scripts/step_<source>.py` → `kg_build/edges/<source>.tsv`.
- Merge: `scripts/step_merge_kg.py` → `kg_build/final_kg_{edges,nodes}.tsv`.
- Authoritative statistics: `kg_build/reports/final_kg_report.txt`.
- The `source → evidence_type` mapping is fixed in
  `step_merge_kg.py::SOURCE_TO_EVIDENCE_TYPE`.

For the benchmark tasks and splits built on top of this graph, see
**[BENCHMARK.md](BENCHMARK.md)**; for baseline results,
**[RESULTS.md](RESULTS.md)**.
