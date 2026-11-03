"""Same-study sibling audit split by evidence unit (P0-1).  A test pair (m, d) has a sibling of unit U if a training
target pair of the same disease d shares a U-unit with it:
  literature  a PMID (or Disbiome publication) reporting both pairs
  gmrepo      the GMrepo contrast GMREPO:<d> (every cohort pair of that contrast shares it)
  mbodymap    an mBodyMap site-disease group MBODYMAP:<site>:<d>
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

MKG = Path("/root/autodl-tmp/mkg")
SEEDS = [42, 123, 456, 789, 2024]
SETTINGS = {"t1_pair": "bench2", "t1_study_reveal": "bench2", "t1_study": "bench2", "t1_study_strict": "bench2",
            "t1_temporal": "bench2", "t1_temporal_nodis": "bench2", "t1_cold_microbe": "bench2",
            "t1_cold_disease": "bench2", "t1_v3_strict_nocohort": "bench3", "t1_v3_temporal_dated": "bench3"}
ents = pd.read_csv(MKG / "bench2/entities.tsv", sep="\t", dtype=str, keep_default_na=False)
n = len(ents); types = ents.node_type.to_numpy()
x = np.load(MKG / "bench2/triples.npy", mmap_mode="r")
sup = pd.read_csv(MKG / "bench2/pair_support.tsv", sep="\t", dtype=str, keep_default_na=False)
sup["code"] = sup.code.astype(np.int64)
lit = {c: set(v.split(";")) - {""} for c, v in zip(sup.code, sup.lit_studies)}
coh = {c: set(v.split(";")) - {""} for c, v in zip(sup.code, sup.cohort_studies)}
unit = {"literature": lambda c: lit.get(c, set()),
        "gmrepo": lambda c: {u for u in coh.get(c, ()) if u.startswith("GMREPO:")},
        "mbodymap": lambda c: {u for u in coh.get(c, ()) if u.startswith("MBODYMAP:")}}
rows = []
for s, root in SETTINGS.items():
    for sd in SEEDS:
        z = np.load(MKG / root / "splits" / s / f"seed_{sd}.npz")
        tr = np.asarray(x[z["train"]]); te = np.asarray(x[z["test"][z["test_eligible"]]])
        md = (types[tr[:, 0]] == "microbe") & (types[tr[:, 2]] == "disease") & (tr[:, 1] != tr[:, 1] + 1)
        trc = set(np.unique(tr[md][:, 0].astype(np.int64) * n + tr[md][:, 2]).tolist()) & set(sup.code.tolist())
        tec = np.unique(te[:, 0].astype(np.int64) * n + te[:, 2]).tolist()
        idx = {k: set() for k in unit}
        for c in trc:
            for k, f in unit.items():
                for u in f(c):
                    idx[k].add((u, c % n))
        hit = {k: np.array([any((u, c % n) in idx[k] for u in unit[k](c)) for c in tec]) for k in unit}
        rows.append(dict(setting=s, seed=sd, test_pairs=len(tec),
                         **{f"sib_{k}": float(v.mean()) for k, v in hit.items()},
                         sib_any=float(np.mean(hit["literature"] | hit["gmrepo"] | hit["mbodymap"])),
                         sib_cohort=float(np.mean(hit["gmrepo"] | hit["mbodymap"])),
                         test_has_lit=float(np.mean([bool(lit.get(c)) for c in tec])),
                         test_has_gmrepo=float(np.mean([bool(unit["gmrepo"](c)) for c in tec])),
                         test_has_mbodymap=float(np.mean([bool(unit["mbodymap"](c)) for c in tec]))))
df = pd.DataFrame(rows)
(MKG / "results3/analysis").mkdir(parents=True, exist_ok=True)
df.to_csv(MKG / "results3/analysis/sibling_units.csv", index=False)
print(df.groupby("setting").mean(numeric_only=True).drop(columns="seed").round(4).to_string())
