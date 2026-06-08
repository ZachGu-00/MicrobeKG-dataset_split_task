# MicrobeKG — Baseline Results

A reproducible baseline leaderboard across all four tasks, plus a negative-ratio
robustness analysis and a multi-relation ablation. Aggregation is offline and
re-runnable: `experiments/aggregate.py`, `experiments/analyze_robustness.py`,
`experiments/ablation_multirelation.py`. Full per-cell tables are in
`experiments/leaderboard/`.

## Setup

- **13 baselines** — KGE: TransE, DistMult, ComplEx, RotatE, PairRE, ConvE,
  TuckER; GNN: RGCN; structural heuristics: CN, RA, L3; trivial floors: Random,
  Popularity.
- **Single seed (42)**, **182 model × setting cells** populated.
- **Ranking** — full filtered rank over the same-type candidate pool, both
  directions, micro **and** macro, MRR + Hits@{1,3,5,10,20} + 95% bootstrap CI.
- **Discrimination** — hard-negative AUPRC (+ prevalence floor + fold-enrichment)
  and AUROC (+ CI), plus `fpr@median`.

## Headline findings

1. **RotatE leads three task ceilings** (T1-transductive, T2-A, T4-A). On T3-A,
   TuckER reaches 0.4504, narrowly above RotatE at 0.4465.
2. **On honest cold-start / zero-shot splits, structural heuristics match or beat
   KGE.** T1 cold-microbe: L3 > all KGE; T3-B: RA/L3/CN are the top three; T2-B:
   CN/RA (AUROC 0.75) > best KGE TransE (0.62). KGC has **no advantage** on the
   honest splits — the central anti-inflation result.
3. **Models collapse on hard negatives.** On T1/T2 the discrimination AUROC is
   near or **below 0.5** (PairRE 0.035, ComplEx 0.15): a model ranks standard
   links but scores a *contradictory* association **above** a clean one — driven
   by degree/popularity (Popularity's own AUROC is 0.30, also inverted).
4. **The "0.9x AUPRC" on T1/T2 is a floor artifact**, not skill — see robustness.
5. **Honesty flag:** T1 cold-disease collapses to ≈Random, leaving substantial
   headroom on the hardest setting.

---

## Complete benchmark tables

All **182 model × setting cells** are printed in the following Markdown pages.
Each ranking table includes MRR, macro MRR, micro–macro gap,
Hits@{1,3,5,10,20}, confidence interval, and evaluation count. Each
discrimination table includes AUPRC, prevalence floor, AUPRC/floor, AUROC,
confidence intervals, `fpr@median`, and positive/negative counts.

| Task | Settings | Complete results |
|---|---|---|
| Task 1 — microbe→disease association | transductive, transductive with bridges, cold-microbe, cold-disease | **[RESULTS_TASK1.md](RESULTS_TASK1.md)** |
| Task 2 — capacity→realization transfer | A, B, C50, C100, C500 | **[RESULTS_TASK2.md](RESULTS_TASK2.md)** |
| Task 3 — substrate-utilization recovery | A, B | **[RESULTS_TASK3.md](RESULTS_TASK3.md)** |
| Task 4 — metabolite→disease therapeutic prediction | A, B, C | **[RESULTS_TASK4.md](RESULTS_TASK4.md)** |

Machine-readable versions remain available in
`experiments/leaderboard/ranking.csv` and
`experiments/leaderboard/discrimination.csv`.

---

## Negative-ratio robustness

Each discrimination set's raw pos/neg score arrays are resampled offline to
different neg:pos ratios. **AUROC is near-flat across ratios (rank-based,
prevalence-invariant — the real signal); AUPRC slides with prevalence.** So the
high "orig" AUPRC on the positive-heavy T1/T2 sets is a *floor artifact*.

![Negative-ratio robustness: AUROC flat, AUPRC prevalence-driven](figures/robustness.png)

*Each track = one setting swept over neg:pos ∈ {0.5:1 … 10:1}; near-vertical
tracks ⇒ AUROC is locked while AUPRC ranges ~0.06–0.98. ★ = as-reported (orig).
Red zone = AUROC < 0.5 (worse than chance).*

Median-over-models, two contrasting sets (full table:
`experiments/leaderboard/robustness.md`):

| Set | metric | orig | 1:1 | 10:1 |
|---|---|--:|--:|--:|
| T1 transductive (pos-heavy) | AUROC | 0.300 | 0.311 | 0.271 |
| | AUPRC | 0.873 | 0.394 | 0.064 |
| T4-A (neg-heavy) | AUROC | 0.777 | 0.779 | 0.777 |
| | AUPRC | 0.384 | 0.801 | 0.384 |

AUROC barely moves; AUPRC swings by an order of magnitude with the ratio.

---

## Multi-relation ablation (Task 1, same test set)

Does the multi-relational structure help? T1 has a clean ±bridge ablation:
identical test (5,531), train graph **120,868** edges (microbe-disease only) vs
**3,675,029** edges (+ substrate/metabolite/gene bridges).

| Model | MRR only | MRR +bridge | ΔMRR | AUROC only | AUROC +bridge | ΔAUROC |
|---|--:|--:|--:|--:|--:|--:|
| TransE | 0.0657 | 0.0517 | −0.014 | 0.349 | **0.751** | **+0.402** |
| ConvE | 0.0642 | 0.0382 | −0.026 | 0.186 | 0.375 | +0.189 |
| ComplEx | 0.0357 | 0.0808 | +0.045 | 0.152 | 0.347 | +0.195 |
| RotatE | 0.1225 | 0.0815 | −0.041 | 0.285 | 0.301 | +0.017 |
| DistMult | 0.0631 | 0.0602 | −0.003 | 0.459 | 0.127 | **−0.333** |
| L3 | 0.1012 | 0.0887 | −0.013 | 0.196 | 0.208 | +0.012 |

(Full 13-model table: `experiments/leaderboard/ablation_multirelation.md`.)

**Bridges are not a free lunch — the effect is bidirectional.** Adding the
multi-relational bridges **dilutes ranking** (ΔMRR < 0 for 12/13 models) but, for
path-additive KGE like **TransE, flips hard-negative discrimination from inverted
(0.35) to strong (0.75)** — while a bilinear model like DistMult *degrades*. The
value of "multi-relation" is **discrimination reliability and substrate→disease
reachability, not a ranking bump.**

---

## Reading the metrics

- **AUROC is the primary discrimination metric on positive-heavy sets** (T1, T2):
  AUPRC there is pinned to a 0.92–0.97 prevalence floor and is uninformative
  (Random ≈ best model ≈ floor).
- **AUPRC-over-floor is primary on negative-heavy sets** (T4): the floor is 0.09,
  so fold-enrichment is meaningful (up to 7×).
- Always report `n+ : n−`. The robustness figure is the formal justification.

## Coverage & known gaps

| Item | Status |
|---|---|
| Model × setting cells | **182/182** populated (single seed 42) |
| Models | All 13 baselines reported for every setting |
| Seed policy | One released seed only: **42** |
| Task 3 (S,D) genuine-discovery scoring | **TODO** — data ready (`downstream_sd_test.tsv`, 18,499 genuine); needs the composition evaluation |
| Tax-proximity `none`-layer stratification | **TODO** — `*_proximity.tsv` + raw ranks ready; offline recompute |
| Chemical-class stratification (Task 4-C) | TODO (needs ChEBI/ClassyFire) |

All raw per-instance dumps are in `experiments/<task>/results/raw/*.npz`, so the
TODO stratified/discovery metrics are recomputable **without re-running the sweep**.
