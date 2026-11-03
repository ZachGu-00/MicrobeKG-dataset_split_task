#!/usr/bin/env python3
"""Write RELEASE_MANIFEST.json and checksums.sha256 for a release directory.

The manifest lists every file except itself and checksums.sha256 (SHA-256, bytes, Parquet row counts, source labels,
licence status).  checksums.sha256 is written last and covers every file including RELEASE_MANIFEST.json, so
`sha256sum -c checksums.sha256` verifies the whole tree.  Caches and build leftovers are refused.

Usage: python finalize_release.py <release_dir>
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import pandas as pd

SKIP = {"RELEASE_MANIFEST.json", "checksums.sha256"}
BAD = ("__pycache__", ".pytest_cache", ".egg-info", ".ipynb_checkpoints")
SOURCES = {
    "benchmark/entities.tsv": ["NCBI_Taxonomy", "MeSH", "multiple upstream sources"],
    "benchmark/triples.parquet": ["NCBI_Taxonomy", "MeSH", "BugSigDB", "Disbiome", "GMMAD", "GMrepo_v3", "HMDAD",
                                  "Peryton", "Lit44", "gutMDisorder_v2", "Piccinno_2025", "Thomas_2019", "mBodyMap",
                                  "AGORA2", "gutMGene", "NJC19"],
    "records/records_v2.parquet": ["see SOURCE_TERMS.json"],
    "external/external_pairs.parquet": ["BugSigDB"],
    "tested_null.parquet": ["GMrepo_v3"],
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(root: Path):
    files = sorted(p for p in root.rglob("*") if p.is_file())
    bad = [str(p.relative_to(root)) for p in files if any(b in str(p) for b in BAD)]
    if bad:
        raise SystemExit(f"remove build leftovers first: {bad[:5]}")
    info = json.loads((root / "build_info.json").read_text())
    man = {"release_version": "3.0.0", "graph_version": info["graph_version"],
           "benchmark_version": info["benchmark_version"], "settings": info["settings"], "seeds": info["seeds"],
           "ctd_hmdb_only_triples_removed": info["ctd_hmdb_only_triples_removed"],
           "distribution_url": "https://huggingface.co/datasets/ANONYMIZED/MicrobeKG-Bench (placeholder)",
           "ctx_full": "rebuild-only", "files": []}
    for p in files:
        rel = p.relative_to(root).as_posix()
        if rel in SKIP:
            continue
        item = {"path": rel, "sha256": sha256(p), "bytes": p.stat().st_size, "source_labels": SOURCES.get(rel, []),
                "license_status": ("restricted-source-safe derived release" if rel.startswith("benchmark") else
                                   "BugSigDB CC BY 4.0" if rel.startswith("external") else "see SOURCE_TERMS.json")}
        if p.suffix == ".parquet":
            item["rows"] = int(pd.read_parquet(p).shape[0])
        man["files"].append(item)
    (root / "RELEASE_MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n")
    rows = [f"{sha256(p)}  {p.relative_to(root).as_posix()}" for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "checksums.sha256"]
    (root / "checksums.sha256").write_text("\n".join(rows) + "\n")
    print(json.dumps({"files": len(rows), "bytes": sum(i["bytes"] for i in man["files"])}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
