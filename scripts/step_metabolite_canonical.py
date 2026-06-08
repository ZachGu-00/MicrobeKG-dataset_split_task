"""Build a unified canonical-id map for every metabolite/substrate id that
appears anywhere in the KG.

Strategy
--------
1.  Collect ID-pair evidence from every source we have:
      * UniChem `src7src22.txt`   (CHEBI  <-> PubChem,  ~181k pairs)
      * UniChem `src18src22.txt`  (HMDB   <-> PubChem,  ~204k pairs)
      * CTD `CTD_chemicals.tsv`   (MESH   <-> PubChem / CAS / InChIKey)
      * gutMGene preprocessed tables (per-row {PubChem, CHEBI, HMDB, KEGG})
      * VMH metabolite mapping       (VMH_MET <-> HMDB / KEGG / PubChem)
      * SBML extras                  (VMH_MET <-> HMDB / CHEBI)
      * CTD chemical xref            (MESH -> CHEBI/HMDB/KEGG/PubChem)
      * Every distinct ID actually used in `kg_build/edges/*.tsv` as
        head/tail of a metabolite|substrate edge (seeded as singletons so
        even orphan IDs end up in the map).
2.  Normalize ids to `NAMESPACE:value` form, run union-find over all pairs.
3.  For each connected component pick a canonical id with priority
        HMDB > CHEBI > PUBCHEM > KEGG > VMH_MET > MESH > GMNAME > <other>
    HMDB is preferred because the microbe-side subgraph already uses it
    for ~93% of its 1.6M edges; CHEBI is second because it is the most
    interoperable public ontology and UniChem links it directly.
    VMH_MET is kept as a fallback so that the ~590k AGORA2 substrate
    edges with no public-DB cross-reference are not orphaned (the user's
    downstream task requires every dietary substrate to remain in the
    graph).
4.  When a cluster contains several IDs in the same priority namespace
    (e.g. one PubChem CID maps to several CHEBI IDs because of stereo-
    isomerism in UniChem), the lexicographically smallest is chosen and
    the conflict is logged in the report.

Outputs
-------
    kg_build/mapping/metabolite_canonical.tsv
        original_id, canonical_id, cluster_id, cluster_size
    kg_build/reports/metabolite_canonical_report.txt
"""

from __future__ import annotations

import csv
import gzip
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ORIG = ROOT / "original_data"
PRE = ROOT / "preprocessed"
KG = ROOT / "kg_build"
EDGES = KG / "edges"
MAP_DIR = KG / "mapping"
REP_DIR = KG / "reports"

UNICHEM_CHEBI = ORIG / "src7src22.txt" / "src7src22.txt"
UNICHEM_HMDB = ORIG / "src18src22.txt" / "src18src22.txt"
CTD_CHEM = ORIG / "CTD_chemicals.tsv" / "CTD_chemicals.tsv"
GUTMGENE_FILES = [
    PRE / "gutmgene" / "gut_microbe_metabolite_all.tsv",
    PRE / "gutmgene" / "metabolite_host_gene_all.tsv",
]
VMH_HMDB = MAP_DIR / "vmh_metabolite_to_hmdb.tsv"
SBML_EXTRA = MAP_DIR / "sbml_metabolite_extra.tsv"
CTD_XREF = MAP_DIR / "ctd_chemical_xref.tsv"

OUT_PATH = MAP_DIR / "metabolite_canonical.tsv"
REPORT_PATH = REP_DIR / "metabolite_canonical_report.txt"

CANONICAL_PRIORITY = [
    "HMDB",
    "CHEBI",
    "PUBCHEM",
    "KEGG",
    "VMH_MET",
    "MESH",
    "GMNAME",
]
PRIORITY_RANK = {ns: i for i, ns in enumerate(CANONICAL_PRIORITY)}

CTD_FIELDS = [
    "ChemicalName", "ChemicalID", "CasRN", "PubChemCID", "PubChemSID",
    "DTXSID", "InChIKey", "Definition", "ParentIDs", "TreeNumbers",
    "ParentTreeNumbers", "MESHSynonyms", "CTDCuratedSynonyms",
]


# ----------------------------- normalization -----------------------------

def _strip(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    return "" if s.lower() in {"", "nan", "none", "na"} else s


def norm_pubchem(value: object) -> str:
    s = _strip(value)
    if not s:
        return ""
    s = s.replace("PUBCHEM:", "").replace("CID:", "").strip()
    try:
        return f"PUBCHEM:{int(float(s))}"
    except Exception:
        return ""


def norm_chebi(value: object) -> str:
    s = _strip(value)
    if not s:
        return ""
    s = s.upper()
    while s.startswith("CHEBI:"):
        s = s[len("CHEBI:"):]
    s = s.lstrip("0") or "0"
    return f"CHEBI:{s}"


def norm_hmdb(value: object) -> str:
    s = _strip(value)
    if not s:
        return ""
    if s.upper().startswith("HMDB:"):
        s = s.split(":", 1)[1]
    s = s.upper()
    if not s.startswith("HMDB"):
        return ""
    return f"HMDB:{s}"


def norm_kegg(value: object) -> str:
    s = _strip(value)
    if not s:
        return ""
    if s.upper().startswith("KEGG:"):
        s = s.split(":", 1)[1]
    return f"KEGG:{s}"


def norm_vmh(value: object) -> str:
    s = _strip(value)
    if not s:
        return ""
    if s.upper().startswith("VMH_MET:"):
        s = s.split(":", 1)[1]
    return f"VMH_MET:{s}"


def norm_mesh(value: object) -> str:
    s = _strip(value)
    if not s:
        return ""
    if s.upper().startswith("MESH:"):
        s = s.split(":", 1)[1]
    return f"MESH:{s}"


def norm_prefixed(value: object) -> str:
    """Accept an id that already carries a namespace prefix; normalize per
    namespace if known, otherwise return the trimmed string."""
    s = _strip(value)
    if not s or ":" not in s:
        return ""
    prefix = s.split(":", 1)[0].upper()
    if prefix == "PUBCHEM":
        return norm_pubchem(s)
    if prefix == "CHEBI":
        return norm_chebi(s)
    if prefix == "HMDB":
        return norm_hmdb(s)
    if prefix == "KEGG":
        return norm_kegg(s)
    if prefix == "VMH_MET":
        return norm_vmh(s)
    if prefix == "MESH":
        return norm_mesh(s)
    if prefix == "GMNAME":
        return s
    return s


# ------------------------------ union-find -------------------------------

class DSU:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def add(self, x: str) -> None:
        if x and x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        if not a or not b:
            return
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


# ------------------------------- loaders ---------------------------------

def load_unichem_pairs(path: Path, normalizer, dsu: DSU, stats: Counter) -> None:
    if not path.exists():
        stats[f"missing:{path.name}"] += 1
        return
    with open(path, encoding="utf-8") as f:
        next(f, None)  # header `From src:'X'\tTo src:'22'`
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            a = normalizer(parts[0])
            b = norm_pubchem(parts[1])
            if a and b:
                dsu.union(a, b)
                stats[f"unichem:{path.name}"] += 1


def load_ctd_chemicals(dsu: DSU, stats: Counter) -> None:
    if not CTD_CHEM.exists():
        stats["missing:CTD_chemicals"] += 1
        return
    df = pd.read_csv(
        CTD_CHEM, sep="\t", comment="#", header=None, names=CTD_FIELDS,
        dtype=str, keep_default_na=False, na_values=[""],
    )
    for chem_id, cid in zip(df["ChemicalID"], df["PubChemCID"]):
        mesh = norm_mesh(chem_id)
        pc = norm_pubchem(cid)
        if mesh:
            dsu.add(mesh)
        if pc:
            dsu.add(pc)
        if mesh and pc:
            dsu.union(mesh, pc)
            stats["ctd_chemicals:mesh_pubchem"] += 1


def load_gutmgene(dsu: DSU, stats: Counter) -> None:
    # IMPORTANT: a gutMGene row is a microbial *reaction* (substrate -> metabolite),
    # so the substrate and the metabolite are DIFFERENT molecules. We must union
    # cross-reference ids WITHIN each (substrate / metabolite) group only, NEVER
    # across the two — otherwise e.g. D-Glucose(substrate) and Acetate(metabolite)
    # get merged, which then cascades through shared PubChem hubs into a giant
    # false cluster (observed: a single 825-id cluster merging glucose, acetyl-CoA,
    # cholate, tryptophan, ... — see kg_build/reports/metabolite_canonical_report.txt).
    metabolite_cols = {
        "metabolite_pubchem_cid": norm_pubchem,
        "metabolite_chebi": norm_chebi,
        "metabolite_hmdb": norm_hmdb,
        "metabolite_kegg": norm_kegg,
        "metabolite_id": norm_prefixed,
    }
    substrate_cols = {
        "substrate_id": norm_prefixed,
        "substrate_pubchem_cid": norm_pubchem,
    }
    for path in GUTMGENE_FILES:
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, low_memory=False)
        for group in (metabolite_cols, substrate_cols):
            present = [c for c in group if c in df.columns]
            if not present:
                continue
            for _, row in df[present].iterrows():
                ids = [group[c](row[c]) for c in present]
                ids = [i for i in ids if i]
                if not ids:
                    continue
                anchor = ids[0]
                dsu.add(anchor)
                for other in ids[1:]:
                    dsu.union(anchor, other)
                    stats[f"gutmgene:{path.name}"] += 1


def load_vmh_hmdb(dsu: DSU, stats: Counter) -> None:
    if not VMH_HMDB.exists():
        return
    df = pd.read_csv(VMH_HMDB, sep="\t", dtype=str, keep_default_na=False)
    for _, row in df.iterrows():
        vmh = norm_vmh(row.get("vmh_metabolite_abbr", ""))
        ids = [
            vmh,
            norm_hmdb(row.get("hmdb", "")),
            norm_kegg(row.get("kegg", "")),
            norm_pubchem(row.get("pubchem", "")),
            norm_prefixed(row.get("mapped_metabolite_id", "")),
        ]
        ids = [i for i in ids if i]
        if not ids:
            continue
        anchor = ids[0]
        dsu.add(anchor)
        for other in ids[1:]:
            dsu.union(anchor, other)
            stats["vmh_hmdb"] += 1


def load_sbml_extra(dsu: DSU, stats: Counter) -> None:
    if not SBML_EXTRA.exists():
        return
    df = pd.read_csv(SBML_EXTRA, sep="\t", dtype=str, keep_default_na=False)
    for _, row in df.iterrows():
        vmh = norm_vmh(row.get("vmh_abbr", ""))
        mapped = norm_prefixed(row.get("mapped_id", ""))
        if vmh and mapped:
            dsu.union(vmh, mapped)
            stats["sbml_extra"] += 1


def load_ctd_xref(dsu: DSU, stats: Counter) -> None:
    if not CTD_XREF.exists():
        return
    df = pd.read_csv(CTD_XREF, sep="\t", dtype=str, keep_default_na=False)
    for _, row in df.iterrows():
        ids = [
            norm_mesh(row.get("mesh_id", "")),
            norm_pubchem(row.get("pubchem_cid", "")),
            norm_chebi(row.get("chebi", "")),
            norm_hmdb(row.get("hmdb", "")),
            norm_kegg(row.get("kegg", "")),
            norm_prefixed(row.get("preferred_id", "")),
        ]
        ids = [i for i in ids if i]
        if not ids:
            continue
        anchor = ids[0]
        dsu.add(anchor)
        for other in ids[1:]:
            dsu.union(anchor, other)
            stats["ctd_xref"] += 1


def seed_from_edges(dsu: DSU, stats: Counter) -> None:
    """Make sure every metabolite/substrate id that ever appears in the KG
    edges is at least a singleton in the DSU."""
    for path in sorted(EDGES.glob("*.tsv")):
        if path.name == "agora2_edges_legacy_with_invalid_produces.tsv":
            continue
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                for kind, id_col in [("head", "head_id"), ("tail", "tail_id")]:
                    t = row.get(f"{kind}_type", "")
                    if t in {"metabolite", "substrate"}:
                        nid = norm_prefixed(row.get(id_col, ""))
                        if nid:
                            dsu.add(nid)
                            stats[f"edges:{path.name}"] += 1


# --------------------------- canonical picking ---------------------------

def namespace_of(node_id: str) -> str:
    if ":" not in node_id:
        return "OTHER"
    return node_id.split(":", 1)[0]


def pick_canonical(members: list[str]) -> tuple[str, list[tuple[str, list[str]]]]:
    """Return (canonical_id, conflicts) where conflicts is a list of
    (namespace, [ids...]) for namespaces that contain more than one id."""
    by_ns: dict[str, list[str]] = defaultdict(list)
    for m in members:
        by_ns[namespace_of(m)].append(m)

    conflicts: list[tuple[str, list[str]]] = []
    for ns, ids in by_ns.items():
        if len(ids) > 1 and ns in PRIORITY_RANK:
            conflicts.append((ns, sorted(ids)))

    def rank(node_id: str) -> tuple[int, str]:
        ns = namespace_of(node_id)
        return (PRIORITY_RANK.get(ns, 99), node_id)

    canonical = min(members, key=rank)
    return canonical, conflicts


# --------------------------------- main ---------------------------------

def main() -> None:
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)

    dsu = DSU()
    stats: Counter = Counter()

    print("Loading UniChem CHEBI<->PubChem ...")
    load_unichem_pairs(UNICHEM_CHEBI, norm_chebi, dsu, stats)
    print("Loading UniChem HMDB<->PubChem ...")
    load_unichem_pairs(UNICHEM_HMDB, norm_hmdb, dsu, stats)
    print("Loading CTD chemicals MESH<->PubChem ...")
    load_ctd_chemicals(dsu, stats)
    print("Loading gutMGene per-row id bundles ...")
    load_gutmgene(dsu, stats)
    print("Loading VMH metabolite mapping ...")
    load_vmh_hmdb(dsu, stats)
    print("Loading SBML extras ...")
    load_sbml_extra(dsu, stats)
    print("Loading CTD chemical xref ...")
    load_ctd_xref(dsu, stats)
    print("Seeding singletons from KG edges ...")
    seed_from_edges(dsu, stats)

    print(f"Total nodes in DSU: {len(dsu.parent):,}")

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for node in dsu.parent:
        members_by_root[dsu.find(node)].append(node)
    print(f"Total clusters: {len(members_by_root):,}")

    rows: list[tuple[str, str, str, int]] = []
    canonical_ns_counter: Counter = Counter()
    cluster_size_hist: Counter = Counter()
    conflict_clusters: list[tuple[str, list[tuple[str, list[str]]]]] = []

    for root, members in members_by_root.items():
        canonical, conflicts = pick_canonical(members)
        cluster_id = canonical
        cluster_size_hist[len(members)] += 1
        canonical_ns_counter[namespace_of(canonical)] += 1
        if conflicts:
            conflict_clusters.append((canonical, conflicts))
        for m in members:
            rows.append((m, canonical, cluster_id, len(members)))

    rows.sort(key=lambda r: (r[2], r[0]))

    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["original_id", "canonical_id", "cluster_id", "cluster_size"])
        w.writerows(rows)
    print(f"Saved {len(rows):,} mapping rows to {OUT_PATH}")

    remapped = sum(1 for r in rows if r[0] != r[1])
    write_report(stats, dsu, members_by_root, canonical_ns_counter,
                 cluster_size_hist, conflict_clusters, remapped, len(rows))
    print(f"Saved report to {REPORT_PATH}")


def write_report(stats: Counter, dsu: DSU, clusters: dict, canonical_ns: Counter,
                 size_hist: Counter, conflicts: list, remapped: int, total: int) -> None:
    lines = [
        "Metabolite canonical mapping report",
        "",
        f"output_file\t{OUT_PATH}",
        f"total_mapped_ids\t{total}",
        f"total_clusters\t{len(clusters)}",
        f"ids_remapped_to_other_canonical\t{remapped}",
        "",
        "evidence_pair_counts (how many union/add events each loader contributed)",
    ]
    for k, v in sorted(stats.items()):
        lines.append(f"{k}\t{v}")

    lines.extend(["", "canonical_namespace_counts"])
    for ns in CANONICAL_PRIORITY + ["OTHER"]:
        if ns in canonical_ns:
            lines.append(f"{ns}\t{canonical_ns[ns]}")

    lines.extend(["", "cluster_size_distribution (size:count)"])
    for size in sorted(size_hist):
        lines.append(f"{size}\t{size_hist[size]}")

    lines.extend([
        "",
        f"intra_namespace_conflict_clusters\t{len(conflicts)}",
        "  (cluster contains >=2 ids in the same priority namespace, e.g. one PubChem -> two ChEBI stereoisomers; canonical chosen lexicographically)",
    ])
    for canonical, confs in conflicts[:30]:
        for ns, ids in confs:
            lines.append(f"  {canonical}\t{ns}\t{','.join(ids)}")
    if len(conflicts) > 30:
        lines.append(f"  ... ({len(conflicts) - 30} more)")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
