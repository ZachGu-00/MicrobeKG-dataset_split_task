# Task 1: Microbe-Disease HRKG Prediction

## Scope

Edge target predicate space:
- POSITIVE (microbe → disease only): 57,377 directional edges
  - enriched_in: 30,672
  - depleted_in: 26,705
- HARD-NEG: inconsistent_association 2,552
- AUXILIARY (train-only): 72,925 edges
  (taxonomy + MeSH is_a + co_occurs_with)
- EXCLUDED: all microbe-substrate / microbe-metabolite / metabolite-disease /
  host_gene / intervention / treats_disease / therapeutic_target_for edges
  (anti-shortcut: prevents 2/3-hop leakage through metabolite or host_gene)

`associated_with_disease` (40,801) is excluded because in this KG it is
host_gene→disease (CTD) or metabolite→disease (HMDB/CTD) but NEVER
microbe→disease. The Gap-1 claim is preserved by the 57,377 microbe-disease
directional edges (130× HMDAD / 8× Disbiome).

## Three paradigms

### (A) Transductive — 5 seeds [42, 123, 456, 789, 2024]
Split unit = unique qualifier-tuple `(head_id, tail_id, body_site_uberon,
evidence_type)`. Ratio 80/10/10. Hard-neg pool split 80/10/10 separately;
test-side hard-neg goes to `test_hardneg.tsv`.

### (B) Inductive Cold-Microbe — 3 seeds [42, 123, 456]
Random 10 % of microbes with positive edges held out. Their positive edges
→ 50/50 valid/test. Taxonomy parents (genus/family/...) of held microbes
remain in train, so the model has structural signal to generalize.

### (C) Inductive Cold-Disease — 3 seeds [42, 123, 456]
Random 10 % of D-coded MeSH diseases that have a MeSH `is_a` parent held
out. Parents stay in train.

## File format

Each split file is the standard 10-column edge TSV
(`head_id, head_type, relation, tail_id, tail_type, confidence,
species_source, source, evidence, evidence_type`).

## Body-site distribution (top 10 UBERON, on positive set)

```
body_site_uberon
none              33669
UBERON:0001988    12896
UBERON:0001836     1475
UBERON:0006524     1132
UBERON:0016484      733
UBERON:0002116      673
UBERON:0000996      545
UBERON:0001052      416
UBERON:0000167      374
UBERON:0001728      313
```

## Evidence-type distribution (positive set)

```
evidence_type
experimental                                     34170
cohort_statistical                               19379
experimental|computational                        1891
experimental|cohort_statistical                   1157
computational                                      556
experimental|cohort_statistical|computational      195
cohort_statistical|computational                    29
```

## Reporting

Stratified MRR / Hits@1 / Hits@10 by evidence_type (5 buckets) and by
body-site bucket (gut / oral / vaginal / respiratory / urinary / other /
unspecified). Hard-neg discrimination = AUROC (positive scores vs
inconsistent_association scores) on `test_hardneg.tsv`.
