# MicrobeKG-Bench v3 release package

Loader and evaluator for **MicrobeKG-Bench: Study-Aware Evaluation of Microbe–Disease Link Prediction**. Code and documentation are published in [GitHub](https://github.com/ZachGu-00/MicrobeKG-dataset_split_task) and linked from [Hugging Face](https://huggingface.co/datasets/ZachGu/MicrobeKG). The v3 data archive will be distributed separately through Google Drive; its download link is pending. The existing Hugging Face nodes/edges tables are the audited graph, not v3 benchmark splits.

## Complete audited graph download

[Download the complete audited graph ZIP from Google Drive](https://drive.google.com/file/d/1q8DHrFgZF8uigf4wINlZfc9zMCenWIfx/view?usp=sharing).
This archive contains the complete audited graph (67,485 typed nodes and
3,647,004 assertions). It is separate from `MicrobeKG-Bench_release_v3.zip`,
which contains filtered graph triples, benchmark splits and associated tables.
Only the complete audited graph archive is currently available from this Drive link;
the benchmark v3 archive has not been uploaded there. The Drive archive may extract
to a differently named directory; pass that actual directory to `--graph-dir`.


## Version names and graph scope

`v3` / `mkbench-v3` identifies the benchmark protocol and release package, not a
third revision of the complete knowledge graph. `bench_v3/` contains its loader,
evaluator and release tools; `analysis_v3/` contains the supplied research scripts.

The source graph is named `microbekg-audited` (local directory `kg_build_audited`):
67,485 typed nodes and 3,647,004 assertions. The separate v3 data archive contains
the same typed node identities and 2,671,916 graph triples, obtained by excluding
975,088 assertions supported only by CTD/HMDB. Its benchmark graph is therefore
a filtered derivative, not a byte-for-byte copy of the complete audited graph.
The archive also supplies benchmark splits, record tables, tested-null candidates
and an external test set. Hugging Face's existing `nodes` / `edges` configurations
remain the complete audited graph; they are not the v3 benchmark splits.

Date-free names are publication aliases. Source-review dates and temporal-cutoff
dates remain provenance facts, and random seeds remain unchanged. Archive hashes
and source hashes identify the exact payload. For an existing local graph folder
with another name, pass its actual path with `--graph-dir`; it need not be renamed.


## Install and test

From this `bench_v3/` directory, use Python >= 3.12:

```bash
python -m pip install -e ".[test]"
python -m pytest -q  # rank test; data-dependent checks skip without the data
```

Extract `MicrobeKG-Bench_release_v3.zip` separately and point tests at its `v3/` directory:

```bash
# Linux / macOS
MKB_DATA_ROOT=/path/to/v3 python -m pytest -q
MKB_DATA_ROOT=/path/to/v3 MKB_ALL_SEEDS=1 python -m pytest -q
```

```powershell
# PowerShell
$env:MKB_DATA_ROOT = 'D:/data/v3'
python -m pytest -q
$env:MKB_ALL_SEEDS = '1'
python -m pytest -q
```

The tests rebuild the degree prior and L3 on the released splits and compare MRR, Hits@10 and study-macro MRR with `tests/reference_metrics.json`, which was written from the paper's frozen evaluator outputs (tolerance 1e-9).

## Contents of the separate data archive

* `benchmark/`: licence-safe graph (`entities.tsv`, `relations.json`, `triples.parquet`, 2,671,916 triples), pair support (`pair_support.parquet`) and split tables for 17 settings × 5 seeds (42, 123, 456, 789, 2024). Split tables hold explicit `(head_id, head_type, relation, tail_id, tail_type)` rows; validation and test tables add `study_id`.
* `benchmark/p11_pairs/`: per seed, the revealed same-study pairs, the matched subset and the matched other-study reserve used by the same-study gain control.
* `benchmark/variants_build_audit.json`: per-variant counts of removed pairs and the identical-test-row checks.
* `benchmark/t1_study_ctx_full_REBUILD_REQUIRED.json`: recipe only; its CTD/HMDB-only context is not redistributed.
* `external/`: the out-of-time BugSigDB test set (5,424 triples, 238 studies) and its five seeds.
* `records/records_v2.parquet`: 65,948 record-level rows without personal-information columns; `records/schema.json`.
* `tested_null.parquet`: 4,461 GMrepo tested-null candidates (not ranking negatives).
* `SOURCE_TERMS.*`, `DATASHEET.md`, `MAINTENANCE.md`, `croissant.json`, `RELEASE_MANIFEST.json`, `checksums.sha256`.

The packaging step removed 975,088 triples whose only support was CTD or HMDB. Multi-source assertions stay when at least one other source supports them.

## Settings

| Setting id | Paper name | Notes |
|---|---|---|
| `t1_pair` | Pair | random pair split |
| `t1_study` | Study-exclusive | held-out literature studies; pairs of a held-out study may still have cohort or other-study support |
| `t1_study_strict` | Strict (main) | Study-exclusive without any training support for test pairs |
| `t1_study_reveal` | Study + R | Study-exclusive with some pairs of each held-out study revealed |
| `t1_temporal`, `t1_temporal_nodis` | Temporal | first-report year split; `_nodis` ignores Disbiome years |
| `t1_cold_microbe`, `t1_cold_disease` | Cold endpoints | unseen microbes / diseases |
| `t1_study_ctx_open` | Study + open context | public non-target context only |
| `t1_external` | Out-of-time | BugSigDB curation after the snapshot |
| `t1_v3_strict_nocohort` | Strict − cohort | no GMrepo/mBodyMap-only pairs, no co-occurrence edges |
| `t1_v3_strict_nooverlap` | Strict − BioProject overlap | GMrepo contrasts sharing projects with held-out studies removed |
| `t1_v3_strict_nocooc`, `t1_v3_temporal_nocooc` | − co-occurrence | GMrepo co-occurrence edges removed |
| `t1_v3_temporal_dated` | Temporal-dated | cohort support only if its data are dated ≤ 2021 |
| `t1_v3_p11_base/same/other` | Same-study control | one candidate set and test set; see `p11_pairs/` |

Every `t1_v3_*` variant keeps the parent's validation rows, test rows and filters; removed training pairs become filter-only (`known_extra`).

## Load and evaluate

```python
from microbekg_bench import load, Evaluator
from microbekg_bench.baselines import DegreePrior
case = load("t1_study_strict", 42, root="/path/to/v3")
result = Evaluator(case).evaluate(DegreePrior(case.train, case.n_entities))
print(result["metrics"]["mrr"], result["metrics"]["macro_mrr"])
```

A scorer implements `candidates(queries, relations, pool, side)` and returns a score matrix. Ranking is filtered, covers both sides (tail: disease given microbe; head: microbe given disease), and gives half credit to each tied candidate. `macro_mrr` is the study-macro MRR: reciprocal ranks are averaged within each `study_id`, then across studies. Pass `groups=` for another grouping.

## Submission format

One JSON object per setting and seed:

```json
{"schema_version": "microbekg-bench-result-1", "model": "YourModel", "setting": "t1_study_strict", "seed": 42,
 "metrics": {"mrr": 0.0, "macro_mrr": 0.0, "hits_at_1": 0.0, "hits_at_3": 0.0, "hits_at_10": 0.0},
 "n_triples": 0, "code_url": "https://example.invalid/replace-me",
 "environment": {"python": "3.12", "package_version": "3.0.0"}}
```

Report the paired difference to the degree prior on the same test rows. The paper labels a difference better or worse only when its 95% interval excludes 0, and equivalent when the 90% interval lies within ±0.005 MRR.

## Distribution

Dataset page: [ZachGu/MicrobeKG](https://huggingface.co/datasets/ZachGu/MicrobeKG). The v3 data archive is pending a Google Drive download link. Original data checksums and manifest are in `release_metadata/`; they describe the separate archive. BugSigDB exports carry CC BY 4.0 in the export header; other sources follow `SOURCE_TERMS.*`.

## Rebuilding

`make_release.py` rebuilds the data part from the benchmark build outputs (all paths are arguments; it refuses to overwrite an existing output directory). `finalize_release.py` writes `RELEASE_MANIFEST.json` and `checksums.sha256`; the checksum list covers the manifest.
