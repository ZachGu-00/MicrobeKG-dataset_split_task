# Read-only evidence queries

Use Python 3.13 with `requirements.lock.txt`. Run from the repository root:

```bash
python resource_tools/query_resource.py --project-root /path/to/data --graph-dir kg_build_audited_20260928 --mode demo --out outputs/resource_demo
```

The graph and all source records stay outside this code release. Under the selected data root, preserve the graph's `edges/`, `mapping/` and `provenance/` directories, the `original_data/` downloads and `literature_evidence/` parsing tables and supplements.

For the audited graph, place `lit/literature_edge_support.tsv` and `curated/contrast_labels_all.tsv` in `<graph>/provenance/query_support/`, or pass `--support-dir path/relative/to/data/root` for their retained location.

Required raw inputs include `Peryton-results.tsv`, `BugSigDBE.csv`, `disbiome_experiment_data.txt`, the retained NJC19 JSON directory, `AGORA2_SBML.zip`, literature resolution/extraction tables and literature supplements. The query identifies missing support tables rather than inventing source coordinates. `inflate64` reads Deflate64 SBML archive members; decompressed lengths and CRCs are checked.

## Queries

- Association: `--mode association --microbe NCBI_TAXON:1017280 --disease D015179`.
- Utilization: `--mode utilization --microbe NCBI_TAXON:166486`.
- One substrate: add `--substrate HMDB:HMDB0000283` to utilization mode.

Keep the same root, graph and output arguments. Without a substrate argument, utilization lists the substrates with experimental records and the capacity assertions on those substrates.

The audited examples return three association assertions backed by 11 upstream records plus one derived conflict record, with eight original locations; the utilization example returns 34 assertions over 24 experimental substrates (10 capacity, 20 utilization, four recorded non-utilization), 19 NJC19 records and three original SBML-bound checks.

Outputs are `queries.json` and `summary.json`. The graph SHA-256 is checked before and after execution. Output must be outside source and graph directories; files in a previously selected output directory may be replaced. Historical graph directories can be selected explicitly and use their matching provenance ledger. Counts vary with the selected graph version.
