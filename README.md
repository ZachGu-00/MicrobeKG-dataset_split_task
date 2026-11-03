# MicrobeKG

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


## MicrobeKG-Bench v3

The study-aware benchmark code is available in [bench_v3/](bench_v3/README.md):
an installable loader, filtered head/tail evaluator, CPU baselines, release builders
and reference-metric tests. [analysis_v3/](analysis_v3/README.md) contains the supplied
analysis and experiment scripts with their original workspace requirements.

Dataset page: [ZachGu/MicrobeKG](https://huggingface.co/datasets/ZachGu/MicrobeKG).
The v3 data archive will be released separately through Google Drive; the link is
pending. The existing Hugging Face nodes/edges tables remain the microbekg-audited
graph. See [the release index](RELEASE_V3.json) for archive hashes and versions.

## MicrobeKG resource and earlier benchmark code

Code for **MicrobeKG: An Evidence-Annotated Knowledge Graph for Microbiome Research**. The resource integrates microbial associations, metabolism, chemical and host context, taxonomies, source provenance, recorded negatives and conflicts.

The current audited resource contains **67,485 typed nodes and 3,647,004 assertions**, including **15,821 microbial nodes**, across 25 relations and 31 typed patterns. The benchmark provides fixed splits, typed candidates, known-answer filters and protocols for thirteen models.

This repository contains code, runtime configuration, tests and documentation. Graph tables, original downloads, splits, model weights, experimental outputs and manuscript files are distributed separately. A data download DOI has not yet been assigned.

## Repository layout

| Directory | Purpose |
|---|---|
| `benchmark/` | Split preparation, six trained models, five graph baselines and foundation-model adapters |
| `revision/` | Fixed experiment configurations, staged execution and statistical comparisons |
| `resource_tools/` | Trace association evidence and compare capacity with recorded utilization |
| `scripts/`, `tools/rebuild_graph.py` | Retained source normalization and reference-graph merge |
| `tools/` | External-input verification and CPU baseline evaluation |
| `vendor/` | Required ULTRA and KG-ICL source, patches and upstream licenses; no checkpoints |

## CPU evaluation

Use Python 3.11 or 3.13 in an isolated environment:

```bash
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install -r requirements-cpu.txt
python tools/verify_data.py --data /path/to/data/benchmark_data
python tools/evaluate_cpu.py --data /path/to/data/benchmark_data --setting t2_b --model Popularity --seeds 42 123 456 --out outputs/t2_b
```

These commands require the external indexed inputs described in [docs/DATA.md](docs/DATA.md). They use every eligible test record and the full typed candidate pool. The output records coverage and selects the paper's primary ranking side. See [docs/EVALUATION.md](docs/EVALUATION.md) for setting identifiers, filtering and discrimination protocols.

To evaluate a new model, implement `candidates(queries, relations, pool, side)` and `triples(rows)`, with higher scores meaning more plausible triples, then call `benchmark/core.py`'s evaluator. Preserve the supplied split masks, candidate pools and filters.

## Evidence queries

```bash
python -m pip install -r resource_tools/requirements.lock.txt
python resource_tools/query_resource.py --project-root /path/to/data --graph-dir kg_build_audited --mode demo --out outputs/resource_demo
```

The two examples return original-record locations for a microbe–disease pair and compare capacity, utilization and recorded non-utilization for a microbe. Inputs and graph hashes are checked without modifying the graph. See [resource_tools/README.md](resource_tools/README.md) for required source files and the audited support directory.

## Prepare splits and train

```bash
python benchmark/prepare.py --root /path/to/reference-data --out benchmark_data --seeds 42 123 456
python benchmark/run.py --data benchmark_data --setting t1_pair --model ComplEx --dim 128 --init-scale 0.1 --loss nssa --nssa-margin 9 --regularizer-weight 0.01 --num-negs 32 --batch 4096 --out outputs/training
```

The preparation command reads `kg_build/final_kg_nodes.tsv` and `kg_build/final_kg_edges.tsv` beneath the supplied root. It creates new split files; to reproduce a fixed reference evaluation, use its existing versioned inputs instead of regenerating them from another graph.

Training uses the separately listed [reference environment](environment-training.reference.yml) and [training dependencies](requirements-training.reference.txt), including PyTorch/PyKEEN. Foundation models require the upstream checkpoints listed in `vendor/*/UPSTREAM.json`. Historical multi-stage execution additionally uses the external inputs and result layout in [revision/README.md](revision/README.md).

The audited resource and fixed reference benchmark are versioned independently. This code update preserves the model/evaluation implementations and does not claim a new model evaluation on the audited graph.

## Code checks

```bash
python -m unittest discover -s tests -v
python tools/audit_repository.py
```

These checks use generated fixtures and do not need graph data, a GPU or model weights. Data-dependent research checks remain alongside the benchmark code.

Original MicrobeKG software is MIT licensed. The KG-ICL integration is GPL-3.0; retained upstream licenses and [NOTICE](NOTICE) describe the scope. These code licenses do not license third-party data.
