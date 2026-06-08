# Task 3 Setting C: external gold (prebiotic -> disease)

Headline validation with a label INDEPENDENT of the two KG bridge edges.
The KG path is a feature only; the (substrate, disease) label comes from
real prebiotic intervention evidence.

## Required file
`original_data/external_gold/prebiotic_disease_gold.tsv` (tab-separated):
substrate_id, disease_id, direction(beneficial/harmful), source(PMID), evidence

## Sources
Prebiotic RCT literature reporting substrate->disease therapeutic pairs:
inulin / resistant starch / GOS / FOS / pectin / beta-glucan -> IBD / T2D /
CRC / obesity / constipation.

## Use
Positives = these airtight pairs. Model trained on A/B predicts can_utilize,
composes (S,D), scored against this gold. Currently ABSENT (TODO: literature
extraction); this split is generated automatically once the file exists.
