"""Common helpers for building KGC benchmark splits over the microbe KG.

All split scripts read `kg_build/final_kg_edges.tsv` (10-col schema with
`evidence_type`) and write per-split TSVs into `splits/taskN_*/`.

Edge TSV schema (input + output identical, 10 cols):
  head_id, head_type, relation, tail_id, tail_type,
  confidence, species_source, source, evidence, evidence_type
"""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
FINAL_EDGES = ROOT / "kg_build" / "final_kg_edges.tsv"
SPLITS_DIR = ROOT / "splits"

EDGE_COLS = [
    "head_id", "head_type", "relation", "tail_id", "tail_type",
    "confidence", "species_source", "source", "evidence", "evidence_type",
]

UBERON_RE = re.compile(r"body_site_uberon=(UBERON:[0-9X]+)")
NAME_RE = re.compile(r"body_site_name=([^|;]+)")

# Coarse body_site bucket for stratified reporting in Task 1.
# Maps any UBERON id substring → coarse bucket; "other" catches the rest.
BODY_SITE_BUCKETS = {
    "UBERON:0001988": "gut",       # Feces
    "UBERON:0000160": "gut",       # Intestine
    "UBERON:0001179": "gut",       # Colonic mucosa
    "UBERON:0001157": "gut",       # Cecum / colon
    "UBERON:0002116": "gut",       # Ileal mucosa
    "UBERON:0000945": "gut",       # Stomach
    "UBERON:0000167": "oral",      # Oral cavity
    "UBERON:0001723": "oral",      # Tongue
    "UBERON:0003409": "oral",      # Subgingival plaque
    "UBERON:0001091": "oral",      # Saliva / tooth
    "UBERON:0001098": "oral",      # Tonsil
    "UBERON:0000996": "vaginal",   # Vagina
    "UBERON:0006524": "respiratory",   # BALF
    "UBERON:0007311": "respiratory",   # Sputum
    "UBERON:0001088": "urinary",   # Urine
    "UBERON:0001134": "skin",
}


def extract_body_site(evidence: str) -> str:
    """Return raw UBERON id from edge evidence column, or "none" if absent."""
    if not evidence:
        return "none"
    m = UBERON_RE.search(str(evidence))
    return m.group(1) if m else "none"


def body_site_bucket(uberon: str) -> str:
    """Map UBERON id (or 'none') → coarse bucket for stratified reporting."""
    if uberon == "none":
        return "unspecified"
    return BODY_SITE_BUCKETS.get(uberon, "other")


def load_final_edges() -> pd.DataFrame:
    """Load full final KG edges. ~3.4M rows / 10 cols / ~620MB on disk."""
    return pd.read_csv(FINAL_EDGES, sep="\t", dtype=str, keep_default_na=False, low_memory=False)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)


def split_three_way(items, ratios=(0.8, 0.1, 0.1), seed=42):
    """Deterministic 3-way split of a sequence into (train, valid, test).

    Uses python's random.Random so results are stable across machines for a
    given seed.
    """
    import random
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    n_train = int(round(n * ratios[0]))
    n_valid = int(round(n * ratios[1]))
    train = items[:n_train]
    valid = items[n_train:n_train + n_valid]
    test = items[n_train + n_valid:]
    return train, valid, test


def df_from_keys(edges: pd.DataFrame, key_cols: list[str], keys: list[tuple]) -> pd.DataFrame:
    """Inner-join edges with a key set by multi-column tuple match."""
    if not keys:
        return edges.iloc[0:0].copy()
    key_df = pd.DataFrame(keys, columns=key_cols)
    return edges.merge(key_df, on=key_cols, how="inner")


def _undirected_key(h, t):
    return h + "\x01" + t if h <= t else t + "\x01" + h


def colocated_removed(edges: pd.DataFrame, eval_pairs) -> pd.DataFrame:
    """G_train co-location scrub. Return every edge whose UNORDERED node pair
    (head_id, tail_id) is in `eval_pairs` -- i.e. the held-out target triples,
    their reverse edges, and any co-located edge (same node pair, any relation).
    G_train := full_edges minus this. Output the removed set (not a KG copy).

    The pair-level match means a gene-path bridge edge (different node pair) is
    never removed -- only direct edges between the held-out pair are scrubbed.
    """
    eval_keys = {_undirected_key(a, b) for a, b in eval_pairs}
    keys = [_undirected_key(h, t)
            for h, t in zip(edges["head_id"].values, edges["tail_id"].values)]
    mask = pd.Series([k in eval_keys for k in keys], index=edges.index)
    return edges[mask]


# --- proximity stratification (cold / zero-shot settings; none-layer = headline) ---
RANK_RELS_ORDERED = [
    ("belongs_to_genus", "genus"), ("belongs_to_family", "family"),
    ("belongs_to_order", "order"), ("belongs_to_class", "class"),
    ("belongs_to_phylum", "phylum"),
]


def extract_microbe_lineage(edges):
    """microbe -> {rank: tax_node_id} from belongs_to_* edges."""
    lineage = defaultdict(dict)
    for rel, rank in RANK_RELS_ORDERED:
        sub = edges[(edges.relation == rel)
                    & (edges.head_type == "microbe") & (edges.tail_type == "microbe")]
        for h, t in zip(sub.head_id, sub.tail_id):
            lineage[h][rank] = t
    return lineage


def nearest_shared_taxon(query_microbes, ref_microbes, lineage, exclude_query=False):
    """For each query microbe, the closest rank at which it shares a taxon node with
    ANY ref microbe (genus first .. phylum), else 'none'. Returns (df, Counter).
    exclude_query=True drops the query set from ref (so a microbe's own taxon nodes
    don't count as a 'sibling' -- use when query microbes may also be in ref)."""
    if exclude_query:
        ref_microbes = set(ref_microbes) - set(query_microbes)
    ref_nodes = {rank: set() for _, rank in RANK_RELS_ORDERED}
    for m in ref_microbes:
        lin = lineage.get(m, {})
        for _, rank in RANK_RELS_ORDERED:
            if rank in lin:
                ref_nodes[rank].add(lin[rank])
    rows, bucket = [], Counter()
    for m in sorted(query_microbes):
        lin = lineage.get(m, {})
        near = "none"
        for _, rank in RANK_RELS_ORDERED:
            node = lin.get(rank)
            if node and node in ref_nodes[rank]:
                near = rank
                break
        bucket[near] += 1
        rows.append({"microbe": m, "nearest_shared_rank": near})
    return pd.DataFrame(rows, columns=["microbe", "nearest_shared_rank"]), bucket


def nearest_shared_disease_parent(query_diseases, ref_diseases, is_a_edges):
    """For each query disease, whether it shares a MeSH `is_a` parent with any ref
    disease (1-hop), else 'none'. Returns (df, Counter). is_a: head is_a tail."""
    parents = defaultdict(set)
    for h, t in zip(is_a_edges.head_id, is_a_edges.tail_id):
        parents[h].add(t)
    ref_parents = set()
    for d in ref_diseases:
        ref_parents |= parents.get(d, set())
    rows, bucket = [], Counter()
    for d in sorted(query_diseases):
        shared = "parent" if (parents.get(d, set()) & ref_parents) else "none"
        bucket[shared] += 1
        rows.append({"disease": d, "shares_parent_with_train": shared})
    return pd.DataFrame(rows, columns=["disease", "shares_parent_with_train"]), bucket
