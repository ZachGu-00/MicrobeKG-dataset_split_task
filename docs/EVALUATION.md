# Evaluation protocol

Entity identity is `(node_type, node_id)`. The fixed reference splits use seeds 42, 123 and 456. Keep their typed candidate sets, eligibility masks, known-answer filters and recorded-control arrays unchanged when comparing models.

| Paper setting | File identifier | Primary metric |
|---|---|---|
| S1 | `t1_pair` | Filtered MRR, both directions |
| S1 with context | `t1_bridges` | Filtered MRR, both directions |
| S2 | `t1_cold_microbe` | Head MRR |
| S3 | `t1_cold_disease` | Tail MRR |
| T2-A / T2-B | `t2_a` / `t2_b` | Tail MRR |
| LOSO-P/G/H | `t1_loso_peryton` / `t1_loso_gutmdisorder_v2` / `t1_loso_hmdad` | MRR, both directions |
| T3-A | `t4_a` | MRR, both directions |
| T3-B | `t4_b` with matched-control arrays | Degree-matched recorded-control AUROC |
| T3-C | `t4_c` | Head MRR |

The historical `t4_*` identifiers are retained for the paper's T3 family. Primary MRR averages held-out records; query-balanced MRR is a separate measure. Report coverage alongside scores over eligible records. Unknown-pair controls and recorded negatives are distinct evaluation conditions.

The scorer assigns larger scores to more plausible triples. Filtering uses training and validation positives during validation, and also the frozen test positives and specified `known_extra` records during test evaluation. Ties receive average ranks. Preserve `scored_triples` relation remapping.

S2/S3 hold out target-relation edges at the selected endpoint while retaining its other graph context. They are target-relation-cold settings. The generic evaluator may return both sides; select the primary side in the table above.

`benchmark/run.py` supports six trained models (TransE, DistMult, ComplEx, RotatE, TuckER, RGCN) and five baselines (Random, Popularity, CN, RA, L3). `benchmark/ultra_adapter.py` and `benchmark/kgicl_adapter.py` use separately obtained pretrained weights. `revision/plan.json` defines the fixed additional search and evaluation jobs. Hyperparameter selection uses validation only.

Train/evaluation results are generated locally and excluded from Git. `revision/report.py` contains paired typed-pair bootstrap and matched-control statistics. The resource graph can be updated without rewriting the benchmark's fixed reference inputs or scores.
