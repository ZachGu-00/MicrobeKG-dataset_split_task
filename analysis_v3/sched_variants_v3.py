"""Run the v3 variant cells: heuristics, the six PyKEEN models (Study/Temporal-tuned configs) and ULTRA zero-shot."""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MKG = Path("/root/autodl-tmp/mkg")
OUT = MKG / "results3/variants"
SEL = json.loads((MKG / "results2/selected_configs.json").read_text())
SEEDS = [42, 123, 456, 789, 2024]
LIGHT = ["Popularity", "CN", "RA", "L3"]
LEARNED = ["TransE", "DistMult", "ComplEx", "RotatE", "TuckER", "RGCN"]
PARENT = {"t1_v3_p11_base": "t1_study", "t1_v3_p11_same": "t1_study", "t1_v3_p11_other": "t1_study",
          "t1_v3_strict_nocohort": "t1_study", "t1_v3_strict_nocooc": "t1_study", "t1_v3_strict_nooverlap": "t1_study",
          "t1_v3_temporal_nocooc": "t1_temporal", "t1_v3_temporal_dated": "t1_temporal"}
ORDER = list(PARENT)


def args_of(c):
    out = ["--dim", str(c["dim"]), "--loss", c["loss"], "--lr", str(c["lr"])]
    for k, flag in [("nssa_margin", "--nssa-margin"), ("init_scale", "--init-scale"),
                    ("regularizer_weight", "--regularizer-weight")]:
        if k in c:
            out += [flag, str(c[k])]
    return out


def done(setting, model, seed):
    key = "ultra" if model == "ULTRA" else model.lower()
    return any((OUT / "main").glob(f"{setting}_{key}_split{seed}_seed{seed}_*.npz"))


def cmd_of(setting, model, seed):
    if model == "ULTRA":
        return [sys.executable, str(MKG / "ultra_v3.py"), "--data", str(MKG / "bench3"), "--out", str(OUT / "main"),
                "--setting", setting, "--split-seed", str(seed), "--seed", str(seed), "--mode", "zeroshot",
                "--batch", "8", "--eval-batch", "8"]
    conf = dict(loss="default", dim=128, lr=1e-3) if model in LIGHT else SEL[f"{PARENT[setting]}|{model}"]["config"]
    return [sys.executable, str(MKG / "mk3_cell.py"), "--data", str(MKG / "bench3"), "--out", str(OUT),
            "--setting", setting, "--model", model, "--split-seed", str(seed), "--seed", str(seed),
            "--device", "cpu" if model in LIGHT else "cuda", "--epochs", "150", "--frequency", "10",
            "--patience", "3"] + args_of(conf)


def go(job):
    s, m, sd = job
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd_of(s, m, sd), capture_output=True, text=True, timeout=4 * 3600, cwd=str(MKG / "gnn"))
        rc, tail = p.returncode, (p.stdout[-500:] + p.stderr[-1500:]) if p.returncode else ""
    except subprocess.TimeoutExpired:
        rc, tail = -9, "timeout"
    return f"DONE {s} {m} s{sd} rc={rc} {time.perf_counter() - t0:.0f}s" + ("\n" + tail if tail else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=ORDER)
    ap.add_argument("--models", nargs="+", default=LIGHT + LEARNED + ["ULTRA"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    (OUT / "main").mkdir(parents=True, exist_ok=True)
    jobs = [(s, m, sd) for s in a.settings for m in a.models for sd in SEEDS if not done(s, m, sd)]
    # RGCN first within each priority block so it does not become the tail
    jobs.sort(key=lambda j: (ORDER.index(j[0]) // 3, j[1] != "RGCN"))
    print(len(jobs), "jobs", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(go, jobs):
            print(msg, flush=True)
    print("QUEUE DONE", flush=True)


if __name__ == "__main__":
    main()
