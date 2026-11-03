# External data layout

No data files or model checkpoints are included in this code repository. Obtain an authorized data release separately and retain its version and manifest. The current resource is `microbekg-audited`; the reference benchmark retains its original fixed graph/split inputs.

## Indexed evaluation inputs

Each directory passed to `tools/evaluate_cpu.py --data` contains:

```text
manifest.json
entities.tsv
relations.json
triples.npy
splits/<setting>/seed_<seed>.json
splits/<setting>/seed_<seed>.npz
```

The manifest lists file hashes. Split arrays contain train, validation and test indices, eligibility masks, negatives, conflicts and known-extra filtering records. LOSO inputs use `supplement_data/loso/`; matched-control arrays live in `supplement_data/matched/` and are consumed by supplemental evaluation code.

## Graph and evidence inputs

The current evidence query reads:

```text
kg_build_audited/
  final_kg_nodes.tsv
  final_kg_edges.tsv
  edges/
  mapping/
  provenance/
    literature_release_ledger.tsv
    query_support/
      lit/literature_edge_support.tsv
      curated/contrast_labels_all.tsv
original_data/
literature_evidence/
```

`--support-dir` selects an alternative support directory relative to `--project-root`. The two tables provide retained assertion support and comparison-level source IDs; they are data inputs, not code. Raw source files are required for original-cell, JSON-record and SBML checks. Details appear in the [query guide](../resource_tools/README.md).

## Reference graph construction

`scripts/` preserves source normalization and the original merge. With the retained standardized source tables, mappings, preprocessing tables and GMrepo phenotype-name lookup, run:

```bash
python tools/rebuild_graph.py --data-root /path/to/reference-data --out outputs/rebuilt_reference
```

The wrapper checks input hashes and compares the resulting node and edge hashes with that reference dataset's `kg_build/` files. It reconstructs the reference graph from retained intermediate inputs; it is not the rule pipeline for the later audited graph. Code and data versions should be kept together.

Source attribution and redistribution terms belong with each data release. The software license does not grant redistribution rights for original database downloads or literature supplements.
