"""CTD chem-gene interactions → metabolite→host_gene edges (Plan B).

Filters
-------
- Organism == 'Homo sapiens'
- ChemicalID resolves to a KG metabolite node via ctd_chemical_xref.tsv
  (HMDB > CHEBI > PUBCHEM > KEGG > MESH priority)
- InteractionActions contains 'increases^expression' OR 'decreases^expression'
  (Plan A schema reuse — upregulates_gene / downregulates_gene only;
  affects/binding/methylation/etc. are not imported)
- Metabolite filter (default Plan A, all gold/host metabolites): import every
  CTD chem-gene edge whose metabolite maps to a KG node. Serves the
  metabolite->gene->disease gene-path discovery target. treats/assoc gold
  metabolites are mostly NOT microbe-linked, so the old Plan B filter dropped
  the very edges the gene path needs; dropping it raises treats gene-path
  reachability 6.2% -> 43.8% (see kg_build/discovery_audit/ctd_supply_ceiling.md).
- Plan B (--microbe-filter): keep only metabolites with a microbe->metabolite
  edge. Original 2026-05-21 behaviour, kept as a superset-preserving opt-in
  for the microbe->metabolite->gene multi-hop goal.

Output schema matches step_ctd.py (9 cols + dedupe key=(head_id, relation, tail_id)):
  head_id, head_type, relation, tail_id, tail_type,
  species_source, confidence, source, pmids, evidence_detail
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CTD_IXN_FILE = (
    ROOT
    / "original_data"
    / "CTD-chemical-gene"
    / "CTD_chem_gene_ixns.tsv"
    / "CTD_chem_gene_ixns.tsv"
)
KG_NODES = ROOT / "kg_build" / "final_kg_nodes.tsv"
KG_EDGES = ROOT / "kg_build" / "final_kg_edges.tsv"
CHEM_XREF = ROOT / "kg_build" / "mapping" / "ctd_chemical_xref.tsv"

EDGE_DIR = ROOT / "kg_build" / "edges"
REPORT_DIR = ROOT / "kg_build" / "reports"
OUTPUT_PATH = EDGE_DIR / "ctd_chem_gene_edges.tsv"
REPORT_PATH = REPORT_DIR / "ctd_chem_gene_edges_report.txt"

ACTION_TO_RELATION = {
    "increases^expression": "upregulates_gene",
    "decreases^expression": "downregulates_gene",
}

csv.field_size_limit(sys.maxsize)


def ensure_dirs() -> None:
    EDGE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_kg_metabolites() -> tuple[set[str], dict[str, str], dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    """Return (all_ids, by_hmdb, by_chebi, by_pubchem, by_kegg, by_mesh)."""
    ids: set[str] = set()
    by_hmdb: dict[str, str] = {}
    by_chebi: dict[str, str] = {}
    by_pubchem: dict[str, str] = {}
    by_kegg: dict[str, str] = {}
    by_mesh: dict[str, str] = {}
    with KG_NODES.open(encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            if row["node_type"] != "metabolite":
                continue
            nid = row["node_id"]
            ids.add(nid)
            if nid.startswith("HMDB:"):
                by_hmdb[nid.split(":", 1)[1]] = nid
            elif nid.startswith("CHEBI:"):
                by_chebi[nid.split(":", 1)[1]] = nid
            elif nid.startswith("PUBCHEM:"):
                by_pubchem[nid.split(":", 1)[1]] = nid
            elif nid.startswith("KEGG:"):
                by_kegg[nid.split(":", 1)[1]] = nid
            elif nid.startswith("MESH:"):
                by_mesh[nid.split(":", 1)[1]] = nid
    return ids, by_hmdb, by_chebi, by_pubchem, by_kegg, by_mesh


def load_microbe_linked_metabolites() -> set[str]:
    """Return set of metabolite ids that appear as the tail of at least one
    `microbe → metabolite` edge in the current final_kg_edges.tsv. Used by
    Plan B filter."""
    out: set[str] = set()
    if not KG_EDGES.exists():
        print(f"WARNING: {KG_EDGES} not found — Plan B microbe-linked filter "
              f"will keep ALL metabolites (effectively Plan A).")
        return out
    with KG_EDGES.open(encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            if row["head_type"] == "microbe" and row["tail_type"] == "metabolite":
                out.add(row["tail_id"])
    return out


def load_kg_host_genes() -> set[str]:
    """Return set of bare gene symbols already in KG (strip 'HGNC:' prefix)."""
    syms: set[str] = set()
    with KG_NODES.open(encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            if row["node_type"] != "host_gene":
                continue
            nid = row["node_id"]
            if nid.startswith("HGNC:"):
                syms.add(nid.split(":", 1)[1])
    return syms


def norm_hmdb(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    return s if s.startswith("HMDB") else f"HMDB{s.zfill(7)}"


def build_chem_to_kgid(by_hmdb, by_chebi, by_pubchem, by_kegg, by_mesh) -> dict[str, str]:
    """CTD ChemicalID (raw form, no MESH: prefix) → KG metabolite id.

    Match priority: HMDB > CHEBI > PUBCHEM > KEGG > MESH-direct.
    """
    mapping: dict[str, str] = {}
    with CHEM_XREF.open(encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            mesh_id = row.get("mesh_id", "").strip()
            if not mesh_id:
                continue
            bare = mesh_id.replace("MESH:", "")
            kg_id = None
            h = norm_hmdb(row.get("hmdb", ""))
            if h and h in by_hmdb:
                kg_id = by_hmdb[h]
            elif row.get("chebi", "").strip():
                ch = row["chebi"].strip()
                if ch in by_chebi:
                    kg_id = by_chebi[ch]
            if not kg_id and row.get("pubchem_cid", "").strip():
                pc = row["pubchem_cid"].strip()
                if pc in by_pubchem:
                    kg_id = by_pubchem[pc]
            if not kg_id and row.get("kegg", "").strip():
                kg = row["kegg"].strip()
                if kg in by_kegg:
                    kg_id = by_kegg[kg]
            if not kg_id and bare in by_mesh:
                kg_id = by_mesh[bare]
            if kg_id:
                mapping[bare] = kg_id
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser(description="CTD chem-gene -> metabolite->gene edges")
    ap.add_argument(
        "--microbe-filter", action="store_true",
        help="Plan B: keep only microbe-linked metabolites. Default OFF = keep "
             "all gold/host metabolites (Plan A) for metabolite->gene->disease "
             "gene-path discovery.")
    args = ap.parse_args()

    ensure_dirs()
    print("[1/4] Loading KG metabolite + host_gene + microbe-linked sets...", flush=True)
    _, by_hmdb, by_chebi, by_pubchem, by_kegg, by_mesh = load_kg_metabolites()
    kg_gene_syms = load_kg_host_genes()
    microbe_linked = load_microbe_linked_metabolites() if args.microbe_filter else set()
    print(f"  plan: {'B (microbe-linked filter)' if args.microbe_filter else 'A (all gold/host metabolites)'}")
    print(f"  KG metabolites by HMDB={len(by_hmdb):,} CHEBI={len(by_chebi):,} "
          f"PUBCHEM={len(by_pubchem):,} KEGG={len(by_kegg):,} MESH={len(by_mesh):,}")
    print(f"  KG host_gene HGNC symbols: {len(kg_gene_syms):,}")
    print(f"  microbe-linked metabolites (Plan B filter set): {len(microbe_linked):,}")

    print("[2/4] Building CTD MESH -> KG metabolite map...", flush=True)
    chem_to_kgid = build_chem_to_kgid(by_hmdb, by_chebi, by_pubchem, by_kegg, by_mesh)
    print(f"  CTD chemicals mappable to KG metabolite: {len(chem_to_kgid):,}")

    print("[3/4] Scanning CTD chem-gene-ixns...", flush=True)
    # key=(metabolite_id, relation, gene_symbol) -> list of (pmid_str, ixn_text, gene_ncbi, action_raw)
    edge_records: dict[tuple[str, str, str], list[tuple[str, str, str, str]]] = defaultdict(list)
    total_human = 0
    matched_chem = 0
    dropped_not_microbe_linked = 0
    after_action_filter = 0
    new_gene_syms: set[str] = set()
    with CTD_IXN_FILE.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 11:
                continue
            (
                chem_name,
                chem_id,
                cas_rn,
                gene_sym,
                gene_id,
                gene_forms,
                organism,
                org_id,
                ixn_text,
                actions,
                pmids,
            ) = parts[:11]
            if organism != "Homo sapiens":
                continue
            total_human += 1
            if chem_id not in chem_to_kgid:
                continue
            matched_chem += 1
            kg_mid = chem_to_kgid[chem_id]
            # Plan B: drop metabolites with no microbe edge
            if microbe_linked and kg_mid not in microbe_linked:
                dropped_not_microbe_linked += 1
                continue
            actions_split = [a.strip() for a in actions.split("|") if a.strip()]
            tail_id = f"HGNC:{gene_sym}"
            if gene_sym not in kg_gene_syms:
                new_gene_syms.add(gene_sym)
            for act in actions_split:
                rel = ACTION_TO_RELATION.get(act)
                if rel is None:
                    continue
                after_action_filter += 1
                edge_records[(kg_mid, rel, tail_id)].append((pmids, ixn_text, gene_id, act))

    print(f"  human ixns:                  {total_human:,}")
    print(f"  matched on metabolite:       {matched_chem:,}")
    print(f"  dropped (Plan B not microbe-linked): {dropped_not_microbe_linked:,}")
    print(f"  after expression-action filter (raw rows): {after_action_filter:,}")
    print(f"  unique edges:                {len(edge_records):,}")
    print(f"  CTD gene symbols NOT in KG (will create new host_gene nodes): {len(new_gene_syms):,}")

    print("[4/4] Writing edges + report...", flush=True)
    rows = []
    for (head_id, rel, tail_id), evidence_list in edge_records.items():
        # Aggregate PMIDs (union, pipe-joined). Sample one interaction_text/gene_id/action_raw
        all_pmids: set[str] = set()
        for pmid_str, _, _, _ in evidence_list:
            for p in pmid_str.split("|"):
                p = p.strip()
                if p:
                    all_pmids.add(p)
        # representative evidence detail (first record's ixn_text + gene_id + action)
        first = evidence_list[0]
        gene_ncbi = first[2]
        # condense ixn_text (some are very long); cap at 200 chars
        ixn_txt = first[1][:200]
        action_raw = first[3]
        detail = f"action={action_raw}|gene_id=NCBI:{gene_ncbi}|interaction_text={ixn_txt}|n_pmids={len(all_pmids)}|n_records={len(evidence_list)}"
        rows.append({
            "head_id": head_id,
            "head_type": "metabolite",
            "relation": rel,
            "tail_id": tail_id,
            "tail_type": "host_gene",
            "species_source": "human",
            "confidence": "literature_curated",
            "source": "CTD",
            "pmids": "|".join(sorted(all_pmids)) if all_pmids else "",
            "evidence_detail": detail,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, sep="\t", index=False)

    # ---- Report ----
    rel_counts = df["relation"].value_counts().to_dict()
    head_count = df["head_id"].nunique()
    tail_count = df["tail_id"].nunique()
    bidir_pairs = (
        df.groupby(["head_id", "tail_id"])["relation"].nunique()
        .reset_index().query("relation > 1").shape[0]
    )
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("CTD chem-gene-ixns edges report\n\n")
        f.write(f"input_file\t{CTD_IXN_FILE}\n")
        f.write(f"output_edges\t{OUTPUT_PATH}\n\n")
        f.write(f"total_human_ixns_scanned\t{total_human}\n")
        f.write(f"matched_to_kg_metabolite\t{matched_chem}\n")
        f.write(f"matched_pct_of_human\t{matched_chem / max(total_human,1) * 100:.2f}\n")
        f.write(f"dropped_not_microbe_linked (Plan B)\t{dropped_not_microbe_linked}\n")
        f.write(f"microbe_linked_metabolite_set_size\t{len(microbe_linked)}\n")
        f.write(f"rows_after_expression_action_filter\t{after_action_filter}\n")
        f.write(f"unique_edges_written\t{len(df)}\n")
        f.write(f"unique_metabolites_in_edges\t{head_count}\n")
        f.write(f"unique_genes_in_edges\t{tail_count}\n")
        f.write(f"ctd_chemicals_mappable_via_xref\t{len(chem_to_kgid)}\n")
        f.write(f"new_host_gene_nodes_to_create\t{len(new_gene_syms)}\n")
        f.write(f"bidirectional_pairs (both up and down on same (m,g))\t{bidir_pairs}\n\n")
        f.write("relation_counts:\n")
        for k, v in sorted(rel_counts.items()):
            f.write(f"  {k}\t{v}\n")
    print(f"  wrote {OUTPUT_PATH}  ({len(df):,} edges)")
    print(f"  wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
