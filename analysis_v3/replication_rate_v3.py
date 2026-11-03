"""P1-2 noise ceiling: inter-study replication of literature pairs (computed from the released tables).
Among microbe-disease pairs whose disease has >= 2 literature studies, the share reported by >= 2 literature studies,
and the share reported by >= 2 studies in the same direction.  Writes results3/analysis/replication_rate.csv."""
from pathlib import Path

import pandas as pd

R = Path("/home/user/ClawsGO/MicrobeKG")
rec = pd.read_parquet(R / "release/v3/records/records_v2.parquet")
rec = rec[~rec.record_source.isin(["GMrepo_v3", "mBodyMap"]) & rec.study_id.ne("") & rec.direction.isin(["up", "down"])]
nstud_d = rec.groupby("tail_id").study_id.nunique()
rec = rec[rec.tail_id.map(nstud_d) >= 2]
g = rec.groupby(["head_id", "tail_id"])
n_any = g.study_id.nunique()
n_dir = rec.groupby(["head_id", "tail_id", "direction"]).study_id.nunique().groupby(level=[0, 1]).max()
out = pd.DataFrame([dict(pairs=len(n_any), diseases=int((nstud_d >= 2).sum()), replicated_any=float((n_any >= 2).mean()),
                         replicated_same_direction=float((n_dir >= 2).mean()))])
out.to_csv(R / "results3/analysis/replication_rate.csv", index=False)
print(out.round(4).to_string())
