# Multi-relation ablation — Task 1 (microbe→disease), same test set

Train graph: **microbe-disease only = 120,868 edges** vs **+ bridges = 3,675,029 edges**. Identical test (5,531 eval / 195 oov), seed 42. `MRR`/`H@10` = ranking (both, micro); `AUROC`/`AUPRC/fl` = hard-negative discrimination (positives vs `inconsistent_association`). Δ = (+bridges) − (only). **Bridges dilute ranking (ΔMRR mostly < 0) but can flip discrimination (ΔAUROC ≫ 0 for path-additive KGE like TransE).**

| Model | type | MRR only | MRR +brg | ΔMRR | H@10 only | H@10 +brg | AUROC only | AUROC +brg | ΔAUROC | AUPRC/fl only | AUPRC/fl +brg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TransE | kge | 0.0657 | 0.0517 | -0.0140 | 0.1483 | 0.1102 | 0.3494 | 0.7512 | +0.4018 | 0.9568 | 1.0594 |
| DistMult | kge | 0.0631 | 0.0602 | -0.0029 | 0.1440 | 0.1334 | 0.4594 | 0.1268 | -0.3326 | 1.0022 | 0.8749 |
| ComplEx | kge | 0.0357 | 0.0808 | +0.0451 | 0.0765 | 0.1701 | 0.1522 | 0.3469 | +0.1947 | 0.8859 | 0.9705 |
| RotatE | kge | 0.1225 | 0.0815 | -0.0410 | 0.2457 | 0.1718 | 0.2849 | 0.3014 | +0.0165 | 0.9474 | 0.9512 |
| PairRE | kge | 0.0823 | 0.0669 | -0.0154 | 0.1798 | 0.1427 | 0.0355 | 0.0222 | -0.0133 | 0.8526 | 0.8487 |
| ConvE | kge | 0.0642 | 0.0382 | -0.0260 | 0.1353 | 0.0683 | 0.1859 | 0.3750 | +0.1892 | 0.9092 | 0.9538 |
| TuckER | kge | 0.0948 | - | - | 0.2033 | - | 0.3649 | - | - | 0.9777 | - |
| RGCN | kge | 0.1067 | - | - | 0.2299 | - | 0.1452 | - | - | 0.8798 | - |
| CN | structural | 0.0517 | 0.0496 | -0.0022 | 0.1133 | 0.1046 | 0.4001 | 0.3386 | -0.0615 | 0.9732 | 0.9576 |
| RA | structural | 0.0550 | 0.0543 | -0.0007 | 0.1182 | 0.1125 | 0.3638 | 0.3284 | -0.0354 | 0.9557 | 0.9493 |
| L3 | structural | 0.1012 | 0.0887 | -0.0125 | 0.2262 | 0.1850 | 0.1960 | 0.2078 | +0.0118 | 0.9107 | 0.9166 |
| Random | trivial | 0.0086 | 0.0082 | -0.0004 | 0.0135 | 0.0133 | 0.4997 | 0.4866 | -0.0131 | 0.9976 | 0.9969 |
| Popularity | trivial | 0.0864 | 0.0701 | -0.0163 | 0.1875 | 0.1394 | 0.3004 | 0.2930 | -0.0074 | 0.9503 | 0.9470 |
