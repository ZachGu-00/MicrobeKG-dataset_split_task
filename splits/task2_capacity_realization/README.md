# Task 2: Capacity -> Realization Transfer

Target `utilizes` (microbe->substrate, experimental); capacity = `can_utilize` (GEM computational). Does predicted capacity transfer to observed realization?

- 2A within-relation: can_utilize 90/5/5 (ceiling, NOT the task).
- 2B cross-relation zero-shot (flagship): train = can_utilize(self-pair scrubbed) + taxonomy; valid = can_utilize slice (so test stays pure zero-shot); test = ALL utilizes; hard-neg = does_not_utilize.
- 2C few-shot: inject N utilizes into train.

## Taxonomic-shortcut probe (NEW)
`setting_b_cross_relation/test_microbe_tax_proximity.tsv` stratifies test utilizes microbes by nearest rank sharing a taxon with a train can_utilize microbe (self excluded). The `none` layer is the honest transfer signal -- elsewhere the model can copy a same-genus microbe's can_utilize. Report 2B metrics stratified by this; consider a taxonomy-ablated run.

## Hard-neg imbalance
does_not_utilize is tiny -- report its n and the sampled-negative ratio explicitly; do not report AUPRC at a hidden imbalance.
