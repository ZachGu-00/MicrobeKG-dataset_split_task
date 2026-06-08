# Negative-ratio robustness of hard-negative discrimination (seed 42)

Each setting's raw pos/neg score arrays resampled OFFLINE to different neg:pos ratios (no model rerun); cells = **median across models**. `k:1` = k negatives per positive (majority side subsampled to hit the ratio); `orig` = full set. **AUROC is near-flat across columns (rank-based, prevalence-invariant → the real signal); AUPRC slides with prevalence — the high `orig` AUPRC on pos-heavy T1/T2 is a floor artifact, not skill.** Per-model detail in robustness.csv.


### AUROC — median across models (read: flat across columns)

| task·setting | n+ | n- | orig | 0.5:1 | 1:1 | 2:1 | 5:1 | 10:1 |
|---|---|---|---|---|---|---|---|---|
| task1·transductive | 5531 | 504 | 0.300 | 0.307 | 0.311 | 0.295 | 0.325 | 0.271 |
| task1·transductive_with_bridges | 5531 | 504 | 0.328 | 0.332 | 0.331 | 0.332 | 0.342 | 0.286 |
| task1·cold_microbe | 2687 | 94 | 0.457 | 0.490 | 0.467 | 0.457 | - | - |
| task1·cold_disease | 1928 | 78 | 0.469 | 0.464 | 0.454 | 0.461 | - | - |
| task2·B | 2329 | 81 | 0.583 | 0.544 | 0.610 | 0.571 | - | - |
| task2·C50 | 2655 | 102 | 0.515 | 0.542 | 0.526 | 0.504 | - | - |
| task2·C100 | 2714 | 104 | 0.534 | 0.533 | 0.527 | 0.490 | - | - |
| task2·C500 | 2767 | 140 | 0.549 | 0.560 | 0.552 | 0.564 | - | - |
| task4·A | 629 | 6290 | 0.777 | 0.783 | 0.779 | 0.780 | 0.777 | 0.777 |
| task4·B | 6279 | 62850 | 0.648 | 0.646 | 0.649 | 0.647 | 0.648 | 0.648 |
| task4·C | 1161 | 11630 | 0.544 | 0.540 | 0.545 | 0.542 | 0.546 | 0.544 |

### AUPRC — median across models (read: slides with ratio)

| task·setting | n+ | n- | orig | 0.5:1 | 1:1 | 2:1 | 5:1 | 10:1 |
|---|---|---|---|---|---|---|---|---|
| task1·transductive | 5531 | 504 | 0.873 | 0.571 | 0.394 | 0.240 | 0.121 | 0.064 |
| task1·transductive_with_bridges | 5531 | 504 | 0.872 | 0.567 | 0.404 | 0.255 | 0.129 | 0.065 |
| task1·cold_microbe | 2687 | 94 | 0.961 | 0.652 | 0.473 | 0.328 | - | - |
| task1·cold_disease | 1928 | 78 | 0.955 | 0.627 | 0.475 | 0.325 | - | - |
| task2·B | 2329 | 81 | 0.977 | 0.724 | 0.633 | 0.441 | - | - |
| task2·C50 | 2655 | 102 | 0.966 | 0.714 | 0.535 | 0.357 | - | - |
| task2·C100 | 2714 | 104 | 0.966 | 0.691 | 0.540 | 0.344 | - | - |
| task2·C500 | 2767 | 140 | 0.962 | 0.726 | 0.569 | 0.421 | - | - |
| task4·A | 629 | 6290 | 0.384 | 0.885 | 0.801 | 0.704 | 0.512 | 0.384 |
| task4·B | 6279 | 62850 | 0.185 | 0.792 | 0.666 | 0.504 | 0.305 | 0.185 |
| task4·C | 1161 | 11630 | 0.142 | 0.730 | 0.589 | 0.433 | 0.243 | 0.142 |

### AUPRC / floor — median (read: ≈flat = enrichment is the stable quantity)

| task·setting | n+ | n- | orig | 0.5:1 | 1:1 | 2:1 | 5:1 | 10:1 |
|---|---|---|---|---|---|---|---|---|
| task1·transductive | 5531 | 504 | 0.950 | 0.856 | 0.789 | 0.719 | 0.723 | 0.703 |
| task1·transductive_with_bridges | 5531 | 504 | 0.951 | 0.851 | 0.808 | 0.764 | 0.773 | 0.713 |
| task1·cold_microbe | 2687 | 94 | 0.994 | 0.978 | 0.946 | 0.983 | - | - |
| task1·cold_disease | 1928 | 78 | 0.994 | 0.940 | 0.949 | 0.974 | - | - |
| task2·B | 2329 | 81 | 1.010 | 1.086 | 1.266 | 1.323 | - | - |
| task2·C50 | 2655 | 102 | 1.004 | 1.071 | 1.071 | 1.072 | - | - |
| task2·C100 | 2714 | 104 | 1.004 | 1.037 | 1.080 | 1.033 | - | - |
| task2·C500 | 2767 | 140 | 1.009 | 1.089 | 1.138 | 1.262 | - | - |
| task4·A | 629 | 6290 | 4.224 | 1.327 | 1.602 | 2.112 | 3.071 | 4.224 |
| task4·B | 6279 | 62850 | 2.036 | 1.187 | 1.331 | 1.513 | 1.828 | 2.036 |
| task4·C | 1161 | 11630 | 1.559 | 1.095 | 1.178 | 1.300 | 1.459 | 1.559 |
