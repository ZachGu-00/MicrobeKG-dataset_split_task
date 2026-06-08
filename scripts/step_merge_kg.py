from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
EDGE_DIR = ROOT / "kg_build" / "edges"
REPORT_DIR = ROOT / "kg_build" / "reports"
MAPPING_DIR = ROOT / "kg_build" / "mapping"
GUTMGENE_DIR = ROOT / "preprocessed" / "gutmgene"
AGORA2_DIR = ROOT / "preprocessed" / "agora2"

FINAL_EDGES_PATH = ROOT / "kg_build" / "final_kg_edges.tsv"
FINAL_NODES_PATH = ROOT / "kg_build" / "final_kg_nodes.tsv"
REPORT_PATH = REPORT_DIR / "final_kg_report.txt"
METABOLITE_CANONICAL_PATH = MAPPING_DIR / "metabolite_canonical.tsv"
MICROBE_FALLBACK_PATH = MAPPING_DIR / "microbe_name_fallback.tsv"
COMPOUND_TYPES = {"metabolite", "substrate"}

# Source -> standardized evidence_type for the new `evidence_type` column.
# Used by add_evidence_type() to populate per-edge evidence-type tags so that
# the Gap-2 evidence-typed split (docs/idea_kg_llm_rlvr.md §0.2) can stratify
# train/test by evidence quality without re-parsing the source field.
SOURCE_TO_EVIDENCE_TYPE = {
    "AGORA2": "computational",
    "GMMAD": "computational",
    "HMDB": "curated_database",
    "CTD": "curated_database",
    "gutMGene": "experimental",
    "BugSigDB": "experimental",
    "Disbiome": "experimental",
    "Peryton": "experimental",
    "HMDAD": "experimental",
    "mBodyMap": "experimental",
    "gutMDisorder_v2": "experimental",
    "Piccinno_2025": "experimental",
    "Thomas_2019": "experimental",
    "NJC19": "experimental",
    "GMrepo_v3": "cohort_statistical",
    "Lit44": "cohort_statistical",
    "cross_source_conflict": "cohort_statistical",
    "NCBI_Taxonomy": "ontology",
    "MeSH": "ontology",
}
EVIDENCE_TYPE_PRIORITY = {
    "experimental": 0,
    "curated_database": 1,
    "cohort_statistical": 2,
    "computational": 3,
    "ontology": 4,
    "unknown": 99,
}


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_metabolite_canonical() -> dict[str, str]:
    if not METABOLITE_CANONICAL_PATH.exists():
        print("WARNING: metabolite_canonical.tsv not found; skipping ID normalization. Run scripts/step_metabolite_canonical.py first.")
        return {}
    df = pd.read_csv(METABOLITE_CANONICAL_PATH, sep="\t", dtype=str, keep_default_na=False)
    return dict(zip(df["original_id"], df["canonical_id"]))


def load_microbe_fallback() -> dict[str, str]:
    if not MICROBE_FALLBACK_PATH.exists():
        return {}
    df = pd.read_csv(MICROBE_FALLBACK_PATH, sep="\t", dtype=str, keep_default_na=False)
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        if row.get("ncbi_taxon_id"):
            out[row["original_id"]] = row["ncbi_taxon_id"]
    return out


def apply_microbe_fallback(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    if not mapping or df.empty:
        return df
    out = df.copy()
    head_mask = out["head_type"] == "microbe"
    tail_mask = out["tail_type"] == "microbe"
    if head_mask.any():
        out.loc[head_mask, "head_id"] = out.loc[head_mask, "head_id"].map(
            lambda x: mapping.get(str(x), x)
        )
    if tail_mask.any():
        out.loc[tail_mask, "tail_id"] = out.loc[tail_mask, "tail_id"].map(
            lambda x: mapping.get(str(x), x)
        )
    return out


def apply_metabolite_canonical(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rewrite head_id/tail_id whose type is metabolite|substrate using the
    canonical-id map; ids absent from the map are left unchanged."""
    if not mapping or df.empty:
        return df
    out = df.copy()
    head_mask = out["head_type"].isin(COMPOUND_TYPES)
    tail_mask = out["tail_type"].isin(COMPOUND_TYPES)
    if head_mask.any():
        out.loc[head_mask, "head_id"] = out.loc[head_mask, "head_id"].map(
            lambda x: mapping.get(str(x), x)
        )
    if tail_mask.any():
        out.loc[tail_mask, "tail_id"] = out.loc[tail_mask, "tail_id"].map(
            lambda x: mapping.get(str(x), x)
        )
    return out


def canonical_source_label(source: str) -> str:
    text = str(source)
    if "GMrepo" in text:
        return "GMrepo"
    if "gutMGene" in text:
        return "gutMGene"
    if "AGORA2" in text:
        return "AGORA2"
    return text


def standardize_gmrepo_disease() -> pd.DataFrame:
    df = pd.read_csv(EDGE_DIR / "gmrepo_disease_edges.tsv", sep="\t", low_memory=False)
    df["species_source"] = "human"
    df["evidence"] = (
        "phenotype=" + df["phenotype"].astype(str)
        + "|effect=" + df["effect_direction"].astype(str)
        + "|log2fc=" + df["log2fc"].astype(str)
        + "|pvalue=" + df["pvalue"].astype(str)
        + "|fdr=" + df["fdr"].astype(str)
        + "|n_case=" + df["n_case"].astype(str)
        + "|n_control=" + df["n_control"].astype(str)
    )
    return df[
        ["head_id", "head_type", "relation", "tail_id", "tail_type", "confidence", "species_source", "source", "evidence"]
    ].copy()


def standardize_gutmgene() -> pd.DataFrame:
    df = pd.read_csv(EDGE_DIR / "gutmgene_edges.tsv", sep="\t", low_memory=False)
    df["evidence"] = "pmids=" + df["pmids"].fillna("").astype(str) + "|detail=" + df["evidence_detail"].fillna("").astype(str)
    return df[
        ["head_id", "head_type", "relation", "tail_id", "tail_type", "confidence", "species_source", "source", "evidence"]
    ].copy()


def standardize_agora2() -> pd.DataFrame:
    return pd.read_csv(EDGE_DIR / "agora2_edges.tsv", sep="\t", low_memory=False)[
        ["head_id", "head_type", "relation", "tail_id", "tail_type", "confidence", "species_source", "source", "evidence"]
    ].copy()


def standardize_taxonomy() -> pd.DataFrame:
    path = EDGE_DIR / "taxonomy_edges.tsv"
    if not path.exists():
        print("WARNING: taxonomy_edges.tsv not found, skipping taxonomy hierarchy edges.")
        return pd.DataFrame(
            columns=["head_id", "head_type", "relation", "tail_id", "tail_type",
                     "confidence", "species_source", "source", "evidence"]
        )
    return pd.read_csv(path, sep="\t", low_memory=False)[
        ["head_id", "head_type", "relation", "tail_id", "tail_type", "confidence", "species_source", "source", "evidence"]
    ].copy()


def standardize_gutmdisorder() -> pd.DataFrame:
    path = ROOT / "kg_build" / "edges" / "gutmdisorder_edges.tsv"
    if not path.exists():
        print("WARNING: kg_build/edges/gutmdisorder_edges.tsv not found, skipping.")
        return pd.DataFrame(
            columns=["head_id", "head_type", "relation", "tail_id", "tail_type",
                     "confidence", "species_source", "source", "evidence"]
        )
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df["evidence"] = (
        "pmids=" + df["pmids"].fillna("").astype(str)
        + "|condition=" + df["condition"].fillna("").astype(str)
        + "|taxonomy_level=" + df["taxonomy_level"].fillna("").astype(str)
        + "|intervention=" + df["intervention_name"].fillna("").astype(str)
    )
    # 2026-05-09 schema cleanup: drop direction-ambiguous intervention_affects
    # (87 edges); keep only intervention_increases / intervention_decreases.
    df = df[df["relation"] != "intervention_affects"].reset_index(drop=True)
    return df[
        ["head_id", "head_type", "relation", "tail_id", "tail_type", "confidence", "species_source", "source", "evidence"]
    ].copy()


def _standardize_disease_edge_table(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        print(f"WARNING: {path} not found, skipping {label}.")
        return pd.DataFrame(
            columns=["head_id", "head_type", "relation", "tail_id", "tail_type",
                     "confidence", "species_source", "source", "evidence"]
        )
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df["evidence"] = (
        "pmids=" + df.get("pmids", "").fillna("").astype(str)
        + "|detail=" + df.get("evidence_detail", "").fillna("").astype(str)
    )
    return df[
        ["head_id", "head_type", "relation", "tail_id", "tail_type", "confidence", "species_source", "source", "evidence"]
    ].copy()


def standardize_disbiome() -> pd.DataFrame:
    return _standardize_disease_edge_table(EDGE_DIR / "disbiome_edges.tsv", "Disbiome")


def standardize_gutmdisorder_disease() -> pd.DataFrame:
    return _standardize_disease_edge_table(EDGE_DIR / "gutmdisorder_disease_edges.tsv", "gutMDisorder disease")


def standardize_hmdad() -> pd.DataFrame:
    return _standardize_disease_edge_table(EDGE_DIR / "hmdad_edges.tsv", "HMDAD")


def standardize_peryton() -> pd.DataFrame:
    return _standardize_disease_edge_table(EDGE_DIR / "peryton_edges.tsv", "Peryton")


def standardize_mbodymap() -> pd.DataFrame:
    return _standardize_disease_edge_table(EDGE_DIR / "mbodymap_edges.tsv", "mBodyMap")


def standardize_gmmad() -> pd.DataFrame:
    path = EDGE_DIR / "gmmad_edges.tsv"
    if not path.exists():
        return pd.DataFrame(
            columns=["head_id", "head_type", "relation", "tail_id", "tail_type",
                     "confidence", "species_source", "source", "evidence"]
        )
    return pd.read_csv(path, sep="\t", low_memory=False)[
        ["head_id", "head_type", "relation", "tail_id", "tail_type", "confidence", "species_source", "source", "evidence"]
    ].copy()


def standardize_segata_crc() -> pd.DataFrame:
    path = EDGE_DIR / "segata_crc_edges.tsv"
    if not path.exists():
        return pd.DataFrame(
            columns=["head_id", "head_type", "relation", "tail_id", "tail_type",
                     "confidence", "species_source", "source", "evidence"]
        )
    return pd.read_csv(path, sep="\t", low_memory=False)[
        ["head_id", "head_type", "relation", "tail_id", "tail_type", "confidence", "species_source", "source", "evidence"]
    ].copy()


def standardize_hmdb() -> pd.DataFrame:
    path = EDGE_DIR / "hmdb_edges.tsv"
    if not path.exists():
        print("WARNING: hmdb_edges.tsv not found, skipping HMDB.")
        return pd.DataFrame(
            columns=["head_id", "head_type", "relation", "tail_id", "tail_type",
                     "confidence", "species_source", "source", "evidence"]
        )
    return pd.read_csv(path, sep="\t", low_memory=False)[
        ["head_id", "head_type", "relation", "tail_id", "tail_type", "confidence", "species_source", "source", "evidence"]
    ].copy()


def standardize_bugsigdb_conflict() -> pd.DataFrame:
    path = EDGE_DIR / "bugsigdb_conflict_edges.tsv"
    if not path.exists():
        print("WARNING: bugsigdb_conflict_edges.tsv not found, skipping conflict-mined hard negatives.")
        return pd.DataFrame(
            columns=["head_id", "head_type", "relation", "tail_id", "tail_type",
                     "confidence", "species_source", "source", "evidence"]
        )
    return pd.read_csv(path, sep="\t", low_memory=False)[
        ["head_id", "head_type", "relation", "tail_id", "tail_type", "confidence", "species_source", "source", "evidence"]
    ].copy()


def standardize_cross_source_conflict() -> pd.DataFrame:
    path = EDGE_DIR / "cross_source_conflict_edges.tsv"
    if not path.exists():
        print("WARNING: cross_source_conflict_edges.tsv not found, skipping cross-source hard negatives.")
        return pd.DataFrame(
            columns=["head_id", "head_type", "relation", "tail_id", "tail_type",
                     "confidence", "species_source", "source", "evidence"]
        )
    return pd.read_csv(path, sep="\t", low_memory=False)[
        ["head_id", "head_type", "relation", "tail_id", "tail_type", "confidence", "species_source", "source", "evidence"]
    ].copy()


def standardize_njc19() -> pd.DataFrame:
    path = EDGE_DIR / "njc19_edges.tsv"
    if not path.exists():
        print("WARNING: njc19_edges.tsv not found, skipping NJC19.")
        return pd.DataFrame(
            columns=["head_id", "head_type", "relation", "tail_id", "tail_type",
                     "confidence", "species_source", "source", "evidence"]
        )
    return pd.read_csv(path, sep="\t", low_memory=False)[
        ["head_id", "head_type", "relation", "tail_id", "tail_type", "confidence", "species_source", "source", "evidence"]
    ].copy()


def standardize_bugsigdb() -> pd.DataFrame:
    path = EDGE_DIR / "bugsigdb_edges.tsv"
    if not path.exists():
        print("WARNING: bugsigdb_edges.tsv not found, skipping BugSigDB.")
        return pd.DataFrame(
            columns=["head_id", "head_type", "relation", "tail_id", "tail_type",
                     "confidence", "species_source", "source", "evidence"]
        )
    return pd.read_csv(path, sep="\t", low_memory=False)[
        ["head_id", "head_type", "relation", "tail_id", "tail_type", "confidence", "species_source", "source", "evidence"]
    ].copy()


def standardize_literature44() -> pd.DataFrame:
    """44 literature supplements parsed by parser/parse_supplements.py and
    filtered for case-vs-healthy comparisons (parser/filter_for_kg.py +
    scripts/step_literature44.py)."""
    path = EDGE_DIR / "literature44_edges.tsv"
    if not path.exists():
        print("WARNING: literature44_edges.tsv not found, skipping Literature44.")
        return pd.DataFrame(
            columns=["head_id", "head_type", "relation", "tail_id", "tail_type",
                     "confidence", "species_source", "source", "evidence"]
        )
    return pd.read_csv(path, sep="\t", low_memory=False)[
        ["head_id", "head_type", "relation", "tail_id", "tail_type", "confidence", "species_source", "source", "evidence"]
    ].copy()


def standardize_mesh_tree() -> pd.DataFrame:
    path = EDGE_DIR / "mesh_tree_edges.tsv"
    if not path.exists():
        print("WARNING: mesh_tree_edges.tsv not found, skipping disease-ontology hierarchy edges.")
        return pd.DataFrame(
            columns=["head_id", "head_type", "relation", "tail_id", "tail_type",
                     "confidence", "species_source", "source", "evidence"]
        )
    return pd.read_csv(path, sep="\t", low_memory=False)[
        ["head_id", "head_type", "relation", "tail_id", "tail_type", "confidence", "species_source", "source", "evidence"]
    ].copy()


def standardize_ctd() -> pd.DataFrame:
    path = EDGE_DIR / "ctd_edges.tsv"
    if not path.exists():
        return pd.DataFrame(
            columns=["head_id", "head_type", "relation", "tail_id", "tail_type",
                     "confidence", "species_source", "source", "evidence"]
        )
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df["evidence"] = (
        "pmids=" + df["pmids"].fillna("").astype(str)
        + "|detail=" + df["evidence_detail"].fillna("").astype(str)
    )
    return df[
        ["head_id", "head_type", "relation", "tail_id", "tail_type", "confidence", "species_source", "source", "evidence"]
    ].copy()


def standardize_ctd_chem_gene() -> pd.DataFrame:
    """CTD chem-gene-ixns (metabolite -> host_gene) edges produced by
    scripts/step_ctd_chem_gene.py. Plan A schema reuse: only
    increases^expression / decreases^expression actions are imported and
    mapped to upregulates_gene / downregulates_gene (same relation namespace
    as gutMGene). Source tag = 'CTD' (same database as chem-disease layer).
    """
    path = EDGE_DIR / "ctd_chem_gene_edges.tsv"
    if not path.exists():
        return pd.DataFrame(
            columns=["head_id", "head_type", "relation", "tail_id", "tail_type",
                     "confidence", "species_source", "source", "evidence"]
        )
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df["evidence"] = (
        "pmids=" + df["pmids"].fillna("").astype(str)
        + "|detail=" + df["evidence_detail"].fillna("").astype(str)
    )
    return df[
        ["head_id", "head_type", "relation", "tail_id", "tail_type", "confidence", "species_source", "source", "evidence"]
    ].copy()


def standardize_cooccurrence(agora2: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    # 2026-05-09 schema cleanup: merge co_occurs_negative (18 edges) into
    # co_occurs_with — the spearman_r sign in the evidence column already
    # encodes direction. The previous AGORA2-substrate-overlap upgrade to
    # `competes_with` was dropped (only 2 edges in full KG, not worth a
    # dedicated relation type).
    _ = agora2  # legacy positional arg, no longer used
    df = pd.read_csv(EDGE_DIR / "cooccurrence_edges.tsv", sep="\t", low_memory=False)
    df["relation"] = df["relation"].replace({"co_occurs_negative": "co_occurs_with"})
    return df[
        ["head_id", "head_type", "relation", "tail_id", "tail_type", "confidence", "species_source", "source", "evidence"]
    ].copy(), 0


def remove_residual_non_ncbi_microbes(edges: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop microbe nodes whose IDs are VMH:* or UNMAPPED:*. These cannot be
    grounded to NCBI taxonomy so they don't participate in inductive split or
    leakage audit. SGB:* is kept (bridged to NCBI via is_clade_of)."""
    bad: set[str] = set()
    for col, type_col in [("head_id", "head_type"), ("tail_id", "tail_type")]:
        m = edges.loc[edges[type_col] == "microbe", col]
        bad.update(m[m.str.startswith("VMH:") | m.str.startswith("UNMAPPED:")].unique())
    if not bad:
        return edges, 0
    keep = ~(
        ((edges["head_type"] == "microbe") & (edges["head_id"].isin(bad)))
        | ((edges["tail_type"] == "microbe") & (edges["tail_id"].isin(bad)))
    )
    return edges[keep].reset_index(drop=True), int((~keep).sum())


def remove_ctd_only_metabolites(edges: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop metabolite nodes whose source set is exactly {CTD}. CTD's chemical
    pool is dominated by drugs / environmental toxins / industrial chemicals
    that are semantically distinct from microbial / endogenous metabolites.
    Multi-source CTD nodes (e.g. CTD|HMDB) are kept — HMDB co-occurrence
    confirms an endogenous role."""
    head = edges.loc[edges["head_type"] == "metabolite", ["head_id", "source"]].rename(
        columns={"head_id": "node_id"}
    )
    tail = edges.loc[edges["tail_type"] == "metabolite", ["tail_id", "source"]].rename(
        columns={"tail_id": "node_id"}
    )
    nodes = pd.concat([head, tail], ignore_index=True)
    src_per_node = nodes.groupby("node_id")["source"].apply(
        lambda s: {v for item in s for v in str(item).split("|") if v}
    )
    ctd_only = {n for n, srcs in src_per_node.items() if srcs == {"CTD"}}
    if not ctd_only:
        return edges, 0
    keep = ~(
        ((edges["head_type"] == "metabolite") & (edges["head_id"].isin(ctd_only)))
        | ((edges["tail_type"] == "metabolite") & (edges["tail_id"].isin(ctd_only)))
    )
    return edges[keep].reset_index(drop=True), int((~keep).sum())


def remove_diseases_without_microbe_or_ontology(edges: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop disease nodes that have neither a microbe edge nor an is_a
    ontology link. Pure chemical/gene-only diseases (typically OMIM long-tail
    or MeSH C-supplementary leaves) lose Gap-1 relevance because the KGC
    target task is microbe→disease prediction; without ontology backbone they
    also can't help inductive cold-start (Gap 2)."""
    has_microbe = set(
        edges.loc[
            (edges["head_type"] == "microbe") & (edges["tail_type"] == "disease"),
            "tail_id",
        ]
    )
    isa = edges["relation"] == "is_a"
    in_ontology = set(edges.loc[isa, "head_id"]) | set(edges.loc[isa, "tail_id"])
    keep_ids = has_microbe | in_ontology

    all_disease_ids = set(edges.loc[edges["head_type"] == "disease", "head_id"]) | set(
        edges.loc[edges["tail_type"] == "disease", "tail_id"]
    )
    drop = all_disease_ids - keep_ids
    if not drop:
        return edges, 0
    keep = ~(
        ((edges["head_type"] == "disease") & (edges["head_id"].isin(drop)))
        | ((edges["tail_type"] == "disease") & (edges["tail_id"].isin(drop)))
    )
    return edges[keep].reset_index(drop=True), int((~keep).sum())


def add_evidence_type(edges: pd.DataFrame) -> pd.DataFrame:
    """Populate a standardized `evidence_type` column from the multi-source
    `source` column. When multiple sources back an edge, evidence types are
    pipe-joined sorted by priority (experimental > curated_database >
    cohort_statistical > computational > ontology)."""
    def derive(source_str: str) -> str:
        srcs = [s for s in str(source_str).split("|") if s]
        types = sorted(
            {SOURCE_TO_EVIDENCE_TYPE.get(s, "unknown") for s in srcs},
            key=lambda t: EVIDENCE_TYPE_PRIORITY.get(t, 99),
        )
        return "|".join(types)

    edges = edges.copy()
    edges["evidence_type"] = edges["source"].map(derive)
    return edges


def merge_edges(frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat(frames, ignore_index=True)
    merged["confidence"] = merged["confidence"].fillna("").astype(str)
    merged["species_source"] = merged["species_source"].fillna("").astype(str)
    merged["source"] = merged["source"].fillna("").astype(str)
    merged["evidence"] = merged["evidence"].fillna("").astype(str)

    final_edges = (
        merged.groupby(["head_id", "head_type", "relation", "tail_id", "tail_type"], as_index=False, dropna=False)
        .agg(
            confidence=("confidence", lambda s: "|".join(sorted(set(v for v in s if v)))),
            species_source=("species_source", lambda s: "|".join(sorted(set(v for v in s if v)))),
            source=("source", lambda s: "|".join(sorted(set(v for item in s for v in str(item).split("|") if v)))),
            evidence=("evidence", lambda s: "|".join(sorted(set(v for v in s if v)))),
        )
        .sort_values(["head_type", "head_id", "relation", "tail_type", "tail_id"])
        .reset_index(drop=True)
    )
    return final_edges


def build_name_maps() -> dict[str, str]:
    name_map: dict[str, str] = {}

    phen = pd.read_csv(ROOT / "original_data" / "gmrepo" / "gmrepo_all_phenotypes.csv", low_memory=False)
    for _, row in phen.iterrows():
        if pd.notna(row.get("disease")) and pd.notna(row.get("term")):
            name_map[str(row["disease"])] = str(row["term"])

    for table_name in ["gut_microbe_host_gene_all.tsv", "gut_microbe_metabolite_all.tsv"]:
        df = pd.read_csv(GUTMGENE_DIR / table_name, sep="\t", low_memory=False)
        if "microbe_taxon_id" in df.columns and "gut_microbiota" in df.columns:
            for _, row in df[["microbe_taxon_id", "gut_microbiota"]].dropna().drop_duplicates().iterrows():
                name_map[str(row["microbe_taxon_id"])] = str(row["gut_microbiota"])

    df = pd.read_csv(GUTMGENE_DIR / "gut_microbe_metabolite_all.tsv", sep="\t", low_memory=False)
    for _, row in df[["metabolite_id", "metabolite"]].dropna().drop_duplicates().iterrows():
        name_map[str(row["metabolite_id"])] = str(row["metabolite"])
    for _, row in df[["substrate_id", "substrate"]].dropna().drop_duplicates().iterrows():
        name_map[str(row["substrate_id"])] = str(row["substrate"])

    df = pd.read_csv(GUTMGENE_DIR / "metabolite_host_gene_all.tsv", sep="\t", low_memory=False)
    for _, row in df[["metabolite_id", "metabolite"]].dropna().drop_duplicates().iterrows():
        name_map[str(row["metabolite_id"])] = str(row["metabolite"])

    vmh_microbes = pd.read_csv(MAPPING_DIR / "vmh_to_ncbi_taxon.tsv", sep="\t", low_memory=False)
    for _, row in vmh_microbes.dropna(subset=["mapped_taxon_id", "organism"]).iterrows():
        name_map[str(row["mapped_taxon_id"])] = str(row["organism"])
        name_map[f"VMH:{row['vmh_model_id']}"] = str(row["organism"])

    manifest = pd.read_csv(AGORA2_DIR / "agora2_manifest.tsv.gz", sep="\t", low_memory=False)
    for _, row in manifest.iterrows():
        name_map[f"VMH:{row['vmh_model_id']}"] = str(row["organism_label"])

    vmh_mets = pd.read_csv(MAPPING_DIR / "vmh_metabolite_to_hmdb.tsv", sep="\t", low_memory=False)
    for _, row in vmh_mets.iterrows():
        full_name = str(row.get("full_name", "")).strip()
        if not full_name:
            continue
        mapped = str(row.get("mapped_metabolite_id", "")).strip()
        abbr = str(row.get("vmh_metabolite_abbr", "")).strip()
        if mapped:
            name_map[mapped] = full_name
        if abbr:
            name_map[f"VMH_MET:{abbr}"] = full_name

    # gutMDisorder intervention nodes
    gmd_path = ROOT / "kg_build" / "edges" / "gutmdisorder_edges.tsv"
    if gmd_path.exists():
        gmd = pd.read_csv(gmd_path, sep="\t", low_memory=False)
        for _, row in gmd[["head_id", "intervention_name"]].dropna().drop_duplicates().iterrows():
            name_map[str(row["head_id"])] = str(row["intervention_name"])

    return name_map


def default_node_name(node_id: str) -> str:
    if node_id.startswith("HGNC:"):
        return node_id.split(":", 1)[1]
    if node_id.startswith("UNMAPPED:"):
        return node_id.split(":", 1)[1]
    if node_id.startswith("VMH:"):
        return node_id.split(":", 1)[1].replace("_", " ")
    if node_id.startswith("VMH_MET:"):
        return node_id.split(":", 1)[1]
    return node_id


def build_nodes(final_edges: pd.DataFrame) -> pd.DataFrame:
    name_map = build_name_maps()

    head_nodes = final_edges[["head_id", "head_type", "source"]].rename(
        columns={"head_id": "node_id", "head_type": "node_type"}
    )
    tail_nodes = final_edges[["tail_id", "tail_type", "source"]].rename(
        columns={"tail_id": "node_id", "tail_type": "node_type"}
    )
    nodes = pd.concat([head_nodes, tail_nodes], ignore_index=True)

    final_nodes = (
        nodes.groupby(["node_id", "node_type"], as_index=False)
        .agg(source_databases=("source", lambda s: "|".join(sorted(set(v for item in s for v in str(item).split("|") if v)))))
        .sort_values(["node_type", "node_id"])
        .reset_index(drop=True)
    )
    final_nodes["node_name"] = final_nodes["node_id"].map(lambda x: name_map.get(x, default_node_name(x)))
    final_nodes = final_nodes[["node_id", "node_type", "node_name", "source_databases"]]
    return final_nodes


def compute_graph_stats(final_edges: pd.DataFrame) -> tuple[int, int]:
    parent: dict[str, str] = {}
    size: dict[str, int] = {}
    degree: defaultdict[str, int] = defaultdict(int)

    def find(x: str) -> str:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for node in pd.unique(pd.concat([final_edges["head_id"], final_edges["tail_id"]], ignore_index=True)):
        parent[node] = node
        size[node] = 1

    for _, row in final_edges.iterrows():
        head = row["head_id"]
        tail = row["tail_id"]
        degree[head] += 1
        degree[tail] += 1
        union(head, tail)

    component_sizes: defaultdict[str, int] = defaultdict(int)
    for node in parent:
        component_sizes[find(node)] += 1

    largest_component = max(component_sizes.values()) if component_sizes else 0
    low_degree_nodes = sum(1 for node in parent if degree[node] <= 1)
    return largest_component, low_degree_nodes


def count_shared_microbes(final_nodes: pd.DataFrame) -> int:
    microbes = final_nodes[final_nodes["node_type"] == "microbe"].copy()
    return int(
        microbes["source_databases"]
        .map(lambda x: len(set(v for v in str(x).split("|") if v)) > 1)
        .sum()
    )


def write_report(final_edges: pd.DataFrame, final_nodes: pd.DataFrame, upgraded_competition: int) -> None:
    largest_component, low_degree_nodes = compute_graph_stats(final_edges)
    shared_microbes = count_shared_microbes(final_nodes)

    lines = [
        "Final KG report",
        "",
        f"final_edges_file\t{FINAL_EDGES_PATH}",
        f"final_nodes_file\t{FINAL_NODES_PATH}",
        f"total_nodes\t{len(final_nodes)}",
        f"total_edges\t{len(final_edges)}",
        f"shared_microbe_nodes_across_sources\t{shared_microbes}",
        f"largest_connected_component_nodes\t{largest_component}",
        f"degree_le_1_nodes\t{low_degree_nodes}",
        f"upgraded_competes_with_edges\t{upgraded_competition}",
        "",
        "node_type_counts",
    ]
    for node_type, count in final_nodes["node_type"].value_counts().sort_index().items():
        lines.append(f"{node_type}\t{count}")

    lines.extend(["", "relation_counts"])
    for relation, count in final_edges["relation"].value_counts().sort_index().items():
        lines.append(f"{relation}\t{count}")

    lines.extend(["", "source_edge_counts"])
    exploded = final_edges.assign(source=final_edges["source"].str.split("|")).explode("source")
    for source, count in exploded["source"].value_counts().sort_index().items():
        lines.append(f"{source}\t{count}")

    if "evidence_type" in final_edges.columns:
        lines.extend(["", "evidence_type_counts (per-edge primary tag, pipe-joined when multi-source)"])
        for et, count in final_edges["evidence_type"].value_counts().sort_index().items():
            lines.append(f"{et}\t{count}")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()

    gmrepo = standardize_gmrepo_disease()
    gutmgene = standardize_gutmgene()
    agora2 = standardize_agora2()
    cooccurrence, upgraded_competition = standardize_cooccurrence(agora2)
    taxonomy = standardize_taxonomy()
    gutmdisorder = standardize_gutmdisorder()
    disbiome = standardize_disbiome()
    gutmdisorder_disease = standardize_gutmdisorder_disease()
    hmdad = standardize_hmdad()
    peryton = standardize_peryton()
    mbodymap = standardize_mbodymap()
    ctd = standardize_ctd()
    ctd_chem_gene = standardize_ctd_chem_gene()
    segata = standardize_segata_crc()
    gmmad = standardize_gmmad()
    mesh_tree = standardize_mesh_tree()
    hmdb = standardize_hmdb()
    bugsigdb = standardize_bugsigdb()
    bugsigdb_conflict = standardize_bugsigdb_conflict()
    cross_source_conflict = standardize_cross_source_conflict()
    njc19 = standardize_njc19()
    # Literature48 (48 paper supplements: 44 round1 + 4 round2) parsed by
    # literature_evidence/parser/* + scripts/step_literature44.py. Source
    # tag = 'Lit44'. The standalone version lives in
    # literature_evidence/kg_ready/ and is also referenced as
    # kg_build/edges/literature44_edges.tsv when merged.
    literature44 = standardize_literature44()

    canonical_map = load_metabolite_canonical()
    microbe_fallback = load_microbe_fallback()
    frames = [
        gmrepo, gutmgene, agora2, cooccurrence, taxonomy,
        gutmdisorder, disbiome, gutmdisorder_disease,
        hmdad, peryton, mbodymap, ctd, ctd_chem_gene, segata, gmmad, mesh_tree,
        hmdb, bugsigdb, bugsigdb_conflict, cross_source_conflict, njc19, literature44,
    ]
    frames = [apply_microbe_fallback(f, microbe_fallback) for f in frames]
    frames = [apply_metabolite_canonical(f, canonical_map) for f in frames]

    final_edges = merge_edges(frames)

    # Purity cleanup (2026-05-09): drop schema-noise nodes/edges before nodes
    # are materialised. Order matters — disease filter looks at microbe edges
    # so it should run after microbe / metabolite cleanup propagates removals
    # of orphan disease links.
    final_edges, n_microbe_removed = remove_residual_non_ncbi_microbes(final_edges)
    final_edges, n_ctd_removed = remove_ctd_only_metabolites(final_edges)
    final_edges, n_disease_removed = remove_diseases_without_microbe_or_ontology(final_edges)
    print(f"Purity cleanup: -{n_microbe_removed:,} edges (residual non-NCBI microbe), "
          f"-{n_ctd_removed:,} edges (CTD-only metabolite), "
          f"-{n_disease_removed:,} edges (disease without microbe/ontology link)")

    final_edges = add_evidence_type(final_edges)

    final_nodes = build_nodes(final_edges)

    final_edges.to_csv(FINAL_EDGES_PATH, sep="\t", index=False)
    final_nodes.to_csv(FINAL_NODES_PATH, sep="\t", index=False)
    write_report(final_edges, final_nodes, upgraded_competition)

    print(f"Saved {len(final_edges):,} final KG edges to {FINAL_EDGES_PATH}")
    print(f"Saved {len(final_nodes):,} final KG nodes to {FINAL_NODES_PATH}")


if __name__ == "__main__":
    main()
