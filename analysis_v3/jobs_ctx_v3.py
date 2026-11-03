"""V6: complete the Study + open-context seeds with the configuration of the existing cells (results2/main)."""
import json
from pathlib import Path

MKG = Path("/root/autodl-tmp/mkg")
CFG = {"RotatE": ["--dim", "256"], "TuckER": ["--dim", "128"]}
MISSING = {"RotatE": [123, 456, 789, 2024], "TuckER": [789, 2024]}
jobs = []
for m, seeds in MISSING.items():
    for sd in seeds:
        jobs.append(dict(
            name=f"ctx_open {m} {sd}",
            cmd=["/root/miniconda3/bin/python", str(MKG / "mk2_cell.py"), "--data", str(MKG / "bench2"),
                 "--out", str(MKG / "results2/main"), "--setting", "t1_study_ctx_open", "--model", m,
                 "--split-seed", str(sd), "--seed", str(sd), "--num-negs", "32", "--loss", "default", "--lr", "0.001",
                 "--epochs", "60", "--batch", "1024", "--frequency", "10", "--patience", "3", "--device", "cuda"]
            + CFG[m],
            done=str(MKG / f"results2/main/main/t1_study_ctx_open_{m.lower()}_split{sd}_seed{sd}_*.npz"),
            cwd=str(MKG / "gnn"),
            running=f"mk2_cell.py.*--setting t1_study_ctx_open --model {m} --split-seed {sd} "))
(MKG / "results3/jobs_ctx.json").write_text(json.dumps(jobs, indent=1))
print(len(jobs))
