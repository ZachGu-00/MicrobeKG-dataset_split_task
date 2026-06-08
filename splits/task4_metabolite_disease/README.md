# Task 4: Metabolite->Disease Therapeutic Prediction

## Target
`treats_disease` (metabolite->disease, 6,285, CTD curated). Secondary:
`associated_with_disease`(met) (13,395).

## G_train vs supervision (cross-task principle)
`train_graph = final_kg_edges.tsv - <setting>/gtrain_removed.tsv`.
For each held-out (M,D) target, `gtrain_removed.tsv` contains ALL direct
metabolite->disease edges on that pair (treats/assoc/enriched/depleted) -- the
co-located edges. The mechanism bridge (host_gene<->metabolite, metabolite->gene,
gene->disease) is KEPT: it is a different node pair, so the pair-scrub leaves it
intact. metabolite->gene->disease is mechanism, not leakage.

## Hard negatives
`hardneg_pool_associated_not_treating.tsv` (199,230 pairs): (M,D) with
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
