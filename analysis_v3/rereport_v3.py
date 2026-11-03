"""P1-2: graph target pairs that BugSigDB signatures curated after the graph snapshot report again, from a study whose
PMID is not in the graph (same mapping rules as external_v2.py; pairs already in the graph are KEPT here)."""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/autodl-tmp/MicrobeKG_data")
MKG = Path("/root/autodl-tmp/mkg")
sys.path.insert(0, str(MKG))
from external_v2 import SNAP, norm  # noqa: E402

ents = pd.read_csv(MKG / "bench2/entities.tsv", sep="\t", dtype=str, keep_default_na=False)
n = len(ents)
idx = {(t, i): j for j, (t, i) in enumerate(zip(ents.node_type, ents.node_id))}
edges = pd.read_csv(ROOT / "kg_build_audited/final_kg_edges.tsv", sep="\t", dtype=str, keep_default_na=False,
                    usecols=["head_type", "tail_id", "tail_type", "source", "evidence"])
b = edges[(edges.head_type == "microbe") & (edges.tail_type == "disease") & edges.source.str.contains("BugSigDB")]
votes = defaultdict(Counter)
for tail, ev in zip(b.tail_id, b.evidence):
    for efo in re.findall(r"efo=(EFO:\d+)", ev):
        votes[efo][tail] += 1
efo_map = {k: v.most_common(1)[0][0] for k, v in votes.items()}
mesh_names = json.loads((MKG / "out/mesh_index.json").read_text())["names"]
known_pmids = set(pd.read_csv(MKG / "out2/records_v2.tsv", sep="\t", dtype=str, keep_default_na=False).pmid) - {""}
d = pd.read_csv(MKG / "external/bugsigdb_full_dump.csv", skiprows=1, low_memory=False, dtype=str).fillna("")
d["curated"] = pd.to_datetime(d["Curated date"], errors="coerce", format="mixed")
col = {c: i for i, c in enumerate(d.columns)}
sup = pd.read_csv(MKG / "bench2/pair_support.tsv", sep="\t", dtype=str, keep_default_na=False)
graph_pairs = set(sup.code.astype(np.int64).tolist())
rows = []
for r in d.itertuples(index=False):
    if not (pd.notna(r[col["curated"]]) and r[col["curated"]] > SNAP):
        continue
    pmid = str(r[col["PMID"]]).split(".")[0]
    if pmid and pmid in known_pmids:
        continue
    mesh = efo_map.get(str(r[col["EFO ID"]])) or mesh_names.get(norm(r[col["Condition"]]))
    direction = str(r[col["Abundance in Group 1"]]).lower()
    if not mesh or ("disease", mesh) not in idx or direction not in ("increased", "decreased"):
        continue
    study = re.search(r"(bsdb:\d+)", str(r[col["BSDB ID"]]))
    study = study.group(1) if study else "pmid:" + pmid
    for lineage in str(r[col["NCBI Taxonomy IDs"]]).split(";"):
        ids = [t for t in lineage.split("|") if t.isdigit()]
        m = ("microbe", "NCBI_TAXON:" + ids[-1]) if ids else None
        if m and m in idx:
            c = idx[m] * n + idx[("disease", mesh)]
            if c in graph_pairs:
                rows.append(dict(code=c, study=study, pmid=pmid, direction=direction))
t = pd.DataFrame(rows)
g = t.groupby("code").agg(n_new_studies=("study", "nunique"), n_signatures=("study", "size")).reset_index()
(MKG / "results3/analysis").mkdir(parents=True, exist_ok=True)
g.to_csv(MKG / "results3/analysis/rereported_pairs.tsv", sep="\t", index=False)
print("re-reported graph pairs", len(g), "studies", t.study.nunique(), "pairs with >=2 new studies",
      int((g.n_new_studies >= 2).sum()))
