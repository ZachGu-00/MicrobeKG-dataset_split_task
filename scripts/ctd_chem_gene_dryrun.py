"""CTD chem_gene_ixns dry-run — estimate metabolite->host_gene edges feasible to merge.

Pipeline:
  1. Load KG metabolite IDs from final_kg_nodes.tsv.
  2. Load CTD MESH -> {hmdb, chebi, pubchem_cid, inchikey, cas_rn} from ctd_chemical_xref.tsv.
  3. Build CTD-MESH -> KG-metabolite-id lookup (priority: HMDB > CHEBI > PUBCHEM > KEGG > InChIKey).
  4. Scan CTD_chem_gene_ixns.tsv (Homo sapiens only), match on ChemicalID -> metabolite,
     bucket InteractionActions, count.
"""
from __future__ import annotations
import csv, sys, re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("D:/Paperwork_code/microbe")
NODES = ROOT / "kg_build/final_kg_nodes.tsv"
XREF = ROOT / "kg_build/mapping/ctd_chemical_xref.tsv"
IXN = ROOT / "original_data/CTD-chemical-gene/CTD_chem_gene_ixns.tsv/CTD_chem_gene_ixns.tsv"
EDGES = ROOT / "kg_build/final_kg_edges.tsv"

csv.field_size_limit(sys.maxsize)

print("[1/4] Loading KG metabolite nodes...", flush=True)
kg_metabolite_ids: set[str] = set()
kg_metabolite_by_hmdb: dict[str, str] = {}
kg_metabolite_by_chebi: dict[str, str] = {}
kg_metabolite_by_pubchem: dict[str, str] = {}
kg_metabolite_by_kegg: dict[str, str] = {}
kg_metabolite_by_mesh: dict[str, str] = {}
with NODES.open(encoding="utf-8") as f:
    r = csv.DictReader(f, delimiter="\t")
    for row in r:
        if row["node_type"] != "metabolite":
            continue
        nid = row["node_id"]
        kg_metabolite_ids.add(nid)
        if nid.startswith("HMDB:"):
            kg_metabolite_by_hmdb[nid.split(":", 1)[1]] = nid
        elif nid.startswith("CHEBI:"):
            kg_metabolite_by_chebi[nid.split(":", 1)[1]] = nid
        elif nid.startswith("PUBCHEM:"):
            kg_metabolite_by_pubchem[nid.split(":", 1)[1]] = nid
        elif nid.startswith("KEGG:"):
            kg_metabolite_by_kegg[nid.split(":", 1)[1]] = nid
        elif nid.startswith("MESH:"):
            kg_metabolite_by_mesh[nid.split(":", 1)[1]] = nid

print(f"  metabolites: {len(kg_metabolite_ids):,}")
print(f"  by HMDB={len(kg_metabolite_by_hmdb):,} CHEBI={len(kg_metabolite_by_chebi):,} "
      f"PUBCHEM={len(kg_metabolite_by_pubchem):,} KEGG={len(kg_metabolite_by_kegg):,} "
      f"MESH={len(kg_metabolite_by_mesh):,}")

print("[2/4] Loading CTD chemical xref...", flush=True)
def norm_hmdb(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    return s if s.startswith("HMDB") else f"HMDB{s.zfill(7)}"

ctd_mesh_to_kgid: dict[str, str] = {}  # CTD MESH:Cxxx -> KG metabolite id
match_reason = Counter()
with XREF.open(encoding="utf-8") as f:
    r = csv.DictReader(f, delimiter="\t")
    for row in r:
        mesh = row.get("mesh_id", "").strip()
        if not mesh:
            continue
        ctd_key = mesh if mesh.startswith("MESH:") else f"MESH:{mesh}"
        # Priority: HMDB > CHEBI > PUBCHEM > KEGG > direct MESH
        h = norm_hmdb(row.get("hmdb", ""))
        kg_id = None
        if h and h in kg_metabolite_by_hmdb:
            kg_id = kg_metabolite_by_hmdb[h]; match_reason["HMDB"] += 1
        elif row.get("chebi") and row["chebi"].strip():
            ch = row["chebi"].strip()
            ch_key = ch if ch.startswith("CHEBI:") else ch
            if ch_key in kg_metabolite_by_chebi:
                kg_id = kg_metabolite_by_chebi[ch_key]; match_reason["CHEBI"] += 1
        if not kg_id and row.get("pubchem_cid") and row["pubchem_cid"].strip():
            pc = row["pubchem_cid"].strip()
            if pc in kg_metabolite_by_pubchem:
                kg_id = kg_metabolite_by_pubchem[pc]; match_reason["PUBCHEM"] += 1
        if not kg_id and row.get("kegg") and row["kegg"].strip():
            kg = row["kegg"].strip()
            if kg in kg_metabolite_by_kegg:
                kg_id = kg_metabolite_by_kegg[kg]; match_reason["KEGG"] += 1
        # Direct MESH match (rare)
        if not kg_id:
            bare = mesh.replace("MESH:", "")
            if bare in kg_metabolite_by_mesh:
                kg_id = kg_metabolite_by_mesh[bare]; match_reason["MESH"] += 1
        if kg_id:
            ctd_mesh_to_kgid[ctd_key] = kg_id
            # also raw form (CTD ChemicalID column comes raw without MESH: prefix)
            ctd_mesh_to_kgid[bare := mesh.replace("MESH:", "")] = kg_id

print(f"  CTD MESH -> KG metabolite mappings: {len(ctd_mesh_to_kgid)//2 if ctd_mesh_to_kgid else 0:,} unique")
print(f"  match reasons: {dict(match_reason)}")

print("[3/4] Scanning CTD chem-gene-ixns (human only)...", flush=True)
action_counter = Counter()
ixn_edge_tuples: set[tuple[str, str, str]] = set()  # (kg_mid, action, gene_symbol)
unique_chem = set()
unique_genes = set()
total_human = 0
matched_human = 0
with IXN.open(encoding="utf-8") as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 11:
            continue
        chem_name, chem_id, cas, gene_sym, gene_id, gene_forms, organism, org_id, ixn_text, actions, pmids = parts[:11]
        if organism != "Homo sapiens":
            continue
        total_human += 1
        if chem_id not in ctd_mesh_to_kgid:
            continue
        kg_mid = ctd_mesh_to_kgid[chem_id]
        matched_human += 1
        unique_chem.add(chem_id)
        unique_genes.add(gene_sym)
        # split actions
        for a in actions.split("|"):
            a = a.strip()
            if not a:
                continue
            action_counter[a] += 1
            ixn_edge_tuples.add((kg_mid, a, gene_sym))

print(f"  total human ixns scanned: {total_human:,}")
print(f"  matched to KG metabolite: {matched_human:,}  ({matched_human/total_human*100:.1f}%)")
print(f"  unique CTD chemicals matched: {len(unique_chem):,}")
print(f"  unique gene symbols touched: {len(unique_genes):,}")
print(f"  unique (metabolite, action, gene) tuples: {len(ixn_edge_tuples):,}")

print("[4/4] InteractionActions distribution (top 30):")
for act, n in action_counter.most_common(30):
    print(f"  {n:>10,}  {act}")

# Bucket into KG-style relations
EXPRESSION_UP = ("increases^expression",)
EXPRESSION_DOWN = ("decreases^expression",)
print("\n[buckets] mapping to KG-style relations:")
up = sum(action_counter[a] for a in EXPRESSION_UP)
down = sum(action_counter[a] for a in EXPRESSION_DOWN)
other = sum(n for a, n in action_counter.items() if a not in EXPRESSION_UP + EXPRESSION_DOWN)
print(f"  upregulates_gene (increases^expression):     {up:,}")
print(f"  downregulates_gene (decreases^expression):   {down:,}")
print(f"  other actions (binding/activity/...):        {other:,}")

# Also estimate unique edges per relation bucket
up_edges = {t for t in ixn_edge_tuples if t[1] == "increases^expression"}
down_edges = {t for t in ixn_edge_tuples if t[1] == "decreases^expression"}
other_edges = {t for t in ixn_edge_tuples if t[1] not in ("increases^expression", "decreases^expression")}
print(f"\n[buckets unique edges]")
print(f"  upregulates_gene unique:    {len(up_edges):,}")
print(f"  downregulates_gene unique:  {len(down_edges):,}")
print(f"  other unique:               {len(other_edges):,}")

# Check overlap with existing gutMGene edges
print("\n[overlap with existing gutMGene metabolite->host_gene edges]")
existing = set()
with EDGES.open(encoding="utf-8") as f:
    r = csv.DictReader(f, delimiter="\t")
    for row in r:
        if row["head_type"] == "metabolite" and row["tail_type"] == "host_gene":
            existing.add((row["head_id"], row["relation"], row["tail_id"]))
print(f"  existing metabolite->host_gene edges: {len(existing):,}")
# CTD gene id format is HGNC:SYMBOL in gutMGene; CTD raw uses gene_sym
# rough overlap on (metabolite_id, gene_symbol) ignoring HGNC: prefix
exist_pairs = {(h, t.replace("HGNC:", "")) for (h, _, t) in existing}
new_up_pairs = {(m, g) for (m, a, g) in up_edges}
new_down_pairs = {(m, g) for (m, a, g) in down_edges}
print(f"  CTD up pairs overlapping gutMGene: {len(new_up_pairs & exist_pairs):,}")
print(f"  CTD down pairs overlapping gutMGene: {len(new_down_pairs & exist_pairs):,}")
print(f"  CTD up net new: {len(new_up_pairs - exist_pairs):,}")
print(f"  CTD down net new: {len(new_down_pairs - exist_pairs):,}")
