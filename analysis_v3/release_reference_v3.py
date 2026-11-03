"""Reference per-seed metrics of the degree prior and L3 for the offline release tests (tests/reference_metrics.json).
mrr / hits_at_10 average both ranking sides over eligible test triples; study_macro_mrr averages per-triple reciprocal
ranks within each study_id first (paper definition)."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

MKG = Path("/root/autodl-tmp/mkg")
NAME = re.compile(r"(t1_\w+?)_(popularity|l3)_split(\d+)_seed(\d+)_d(\d+)_n(\d+)_(\w+)\.npz")
ents = pd.read_csv(MKG / "bench2/entities.tsv", sep="\t", dtype=str, keep_default_na=False)
n = len(ents)
sup = pd.read_csv(MKG / "bench2/pair_support.tsv", sep="\t", dtype=str, keep_default_na=False)
l1, c1 = sup.lit_studies.str.split(";").str[0], sup.cohort_studies.str.split(";").str[0]
study = dict(zip(sup.code.astype(np.int64), np.where(l1 != "", l1, c1)))
ext = pd.read_csv(MKG / "bench_ext/external_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
ext["study"] = ext.bsdb.str.split("/").str[0]
ext_study = ext.groupby(ext.code.astype(np.int64)).study.first().to_dict()       # as in analysis_v3.py
X = {"bench2": np.load(MKG / "bench2/triples.npy", mmap_mode="r"),
     "ext": np.load(MKG / "bench_ext/triples.npy", mmap_mode="r")}
out = {}
for d in [MKG / "results2/main/main", MKG / "results3/variants/main"]:
    for p in sorted(d.glob("*.npz")):
        m = NAME.match(p.name)
        if not m or m.group(3) != m.group(4):
            continue
        s, model, seed = m.group(1), m.group(2), int(m.group(3))
        if s.startswith("t1_study_ctx_full"):
            continue
        z = np.load(p)
        t, h = z["tail_ranks"], z["head_ranks"]
        rows = np.asarray((X["ext"] if s == "t1_external" else X["bench2"])[z["ranking_row_ids"]])
        code = rows[:, 0].astype(np.int64) * n + rows[:, 2]
        st = np.array([(ext_study if s == "t1_external" else study).get(int(c), "") for c in code])
        rr = (1 / t + 1 / h) / 2
        macro = float(pd.Series(rr).groupby(st).mean().mean())
        out[f"{s}|{'DegreePrior' if model == 'popularity' else 'L3'}|{seed}"] = dict(
            mrr=float(np.mean(np.r_[1 / t, 1 / h])), hits_at_10=float(np.mean(np.r_[t <= 10, h <= 10])),
            study_macro_mrr=macro, n_triples=int(len(t)))
(MKG / "results3/reference_metrics.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
print(len(out), sorted({k.split("|")[0] for k in out}))
