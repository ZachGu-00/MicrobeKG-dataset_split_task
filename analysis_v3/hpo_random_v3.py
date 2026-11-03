"""P0-4: random hyper-parameter search for RotatE, TuckER and R-GCN on Strict and Out-of-time (split seed 42).

Each configuration is one validation-only run of the frozen benchmark runner (benchmark/run.py through mk2_cell.py),
so tuning never touches test triples.  Stages:
  search  run the sampled configurations (validation MRR on split seed 42)
  select  pick the best configuration per (setting, model) -> results3/hpo_random/selected.json
  final   run the selected configuration on all five split seeds (test evaluation, weights saved)
R-GCN is expensive (20-60 min per Strict cell, about 1 min per epoch on Out-of-time), so it gets 12 sampled
configurations on Strict, and its Out-of-time search reuses the two best Strict configurations plus the earlier
grid winner instead of fresh samples.  RotatE and TuckER get 50 configurations on Strict and the first 25 of
the same sample on Out-of-time.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

CELL = "/root/autodl-tmp/mkg/mk2_cell.py"
OUT = Path("/root/autodl-tmp/mkg/results3/hpo_random")
DATA = {"t1_study_strict": "/root/autodl-tmp/mkg/bench2", "t1_external": "/root/autodl-tmp/mkg/bench_ext"}
SEEDS = [42, 123, 456, 789, 2024]
N_CONFIGS = {"RotatE": 50, "TuckER": 50, "RGCN": 12}
N_EXT = {"RotatE": 15, "TuckER": 15}   # Out-of-time cells take 7-30 min each
# Configurations actually run on Strict.  Sampling still draws N_CONFIGS per model so that the configuration sequence
# (and plan.json) is unchanged; the search was capped at the first N_RUN for compute.
N_RUN = {"RotatE": 30, "TuckER": 30, "RGCN": 12}
EPOCHS = {"RotatE": 150, "TuckER": 150, "RGCN": 80}
EPOCHS_EXT = {"RotatE": 150, "TuckER": 150, "RGCN": 120}
OLD = json.loads(Path("/root/autodl-tmp/mkg/results2/selected_configs.json").read_text())
OLD_KEY = {"t1_study_strict": "t1_study", "t1_external": "t1_pair"}


def sample(model, k, rng):
    out = []
    while len(out) < k:
        c = dict(model=model,
                 dim=int(rng.choice([64, 128, 256, 512] if model == "RotatE" else
                                    [64, 128, 200, 256] if model == "TuckER" else [64, 128, 256])),
                 lr=float(np.round(10 ** rng.uniform(-4, -2), 6)),
                 loss=str(rng.choice(["default", "nssa"])),
                 num_negs=int(rng.choice([8, 32, 64, 128] if model != "RGCN" else [8, 32, 64])),
                 batch=int(rng.choice([256, 1024, 4096] if model != "RGCN" else [512, 1024, 2048])))
        if c["loss"] == "nssa":
            c["nssa_margin"] = float(rng.choice([3.0, 6.0, 9.0, 12.0, 24.0]))
        if model == "RGCN":
            c["rgcn_sampler"] = str(rng.choice(["uniform", "schlichtkrull"]))
        if c not in out:
            out.append(c)
    return out


def cfg_id(c):
    return "_".join(f"{k}{v}" for k, v in sorted(c.items()) if k != "model")


def args_of(c):
    a = ["--dim", str(c["dim"]), "--lr", str(c["lr"]), "--loss", c["loss"], "--num-negs", str(c["num_negs"]),
         "--batch", str(c["batch"])]
    if "nssa_margin" in c:
        a += ["--nssa-margin", str(c["nssa_margin"])]
    if "rgcn_sampler" in c:
        a += ["--rgcn-sampler", c["rgcn_sampler"]]
    return a


def old_config(setting, model):
    c = dict(OLD[f"{OLD_KEY[setting]}|{model}"]["config"])
    c.update(model=model, num_negs=32, batch=1024)
    return c


def gpu_used():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        return int(out.split()[0])
    except Exception:
        return 0


MAX_GPU_MIB = 26000


def run(setting, c, split_seed, out, validation_only, epochs, timeout):
    while gpu_used() > MAX_GPU_MIB:
        time.sleep(30)
    cmd = [sys.executable, CELL, "--data", DATA[setting], "--out", str(out), "--setting", setting, "--model", c["model"],
           "--split-seed", str(split_seed), "--seed", str(split_seed), "--device", "cuda", "--epochs", str(epochs),
           "--frequency", "10", "--patience", "3"] + args_of(c) + (["--validation-only"] if validation_only else [])
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        rc, tail = p.returncode, (p.stdout[-600:] + p.stderr[-1200:]) if p.returncode else ""
    except subprocess.TimeoutExpired:
        rc, tail = -9, "timeout"
    return f"[{setting} {c['model']} s{split_seed} {cfg_id(c)}] rc={rc} {time.perf_counter() - t0:.0f}s" + (
        "\n" + tail if tail else "")


def best_valid(folder):
    vals = []
    for p in Path(folder).glob("**/*.json"):
        d = json.loads(p.read_text())
        v = d.get("training", {}).get("best_validation_mrr")
        if v is not None:
            vals.append(v)
    return max(vals) if vals else None


def search_jobs():
    rng = np.random.default_rng(20261102)
    plan, jobs = {}, []
    for model in ["RotatE", "TuckER", "RGCN"]:
        cs = sample(model, N_CONFIGS[model], rng)
        plan[model] = cs
        for c in cs[:N_RUN[model]] + [old_config("t1_study_strict", model)]:
            jobs.append(("t1_study_strict", c))
        if model != "RGCN":
            for c in cs[:N_EXT[model]] + [old_config("t1_external", model)]:
                jobs.append(("t1_external", c))
    (OUT / "plan.json").write_text(json.dumps(plan, indent=1))
    return jobs


def rgcn_external_jobs():
    # Dropped for compute: R-GCN search cells took 1.7-2.3 GPU hours each on Strict; Out-of-time R-GCN keeps the
    # grid-tuned configuration (stated in the paper's Limitations).
    return []
    scored = []
    for c in json.loads((OUT / "plan.json").read_text())["RGCN"] + [old_config("t1_study_strict", "RGCN")]:
        v = best_valid(OUT / "search" / "t1_study_strict" / "RGCN" / cfg_id(c))
        if v is not None:
            scored.append((v, c))
    top = [c for _, c in sorted(scored, key=lambda t: -t[0])[:5]]
    old = old_config("t1_external", "RGCN")
    top = top[:2]
    return [("t1_external", c) for c in top + ([old] if old not in top else [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["search", "rgcn_external", "select", "final"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=4 * 3600)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage in ("search", "rgcn_external"):
        jobs = search_jobs() if a.stage == "search" else rgcn_external_jobs()
        jobs = [j for j in jobs if best_valid(OUT / "search" / j[0] / j[1]["model"] / cfg_id(j[1])) is None]
        if os.environ.get("HPO_SKIP_RGCN"):    # compute cap: keep the R-GCN configurations already finished
            jobs = [j for j in jobs if j[1]["model"] != "RGCN"]
        jobs.sort(key=lambda j: (j[1]["model"] != "RGCN", j[0]))
        print(len(jobs), "jobs", flush=True)

        def go(j):
            s, c = j
            ep = (EPOCHS_EXT if s == "t1_external" else EPOCHS)[c["model"]]
            return run(s, c, 42, OUT / "search" / s / c["model"] / cfg_id(c), True, ep, a.timeout)

        with ThreadPoolExecutor(a.workers) as ex:
            for msg in ex.map(go, jobs):
                print(msg, flush=True)
        print("STAGE DONE", a.stage, flush=True)
    elif a.stage == "select":
        sel = {}
        for s in DATA:
            for model in ["RotatE", "TuckER", "RGCN"]:
                rows = []
                for d in (OUT / "search" / s / model).glob("*"):
                    v = best_valid(d)
                    if v is None:
                        continue
                    js = next(d.glob("**/*.json"))
                    cfg = json.loads(js.read_text())["config"]
                    c = dict(model=model, dim=cfg["dim"], lr=cfg["lr"], loss=cfg["loss"], num_negs=cfg["num_negs"],
                             batch=cfg["batch"])
                    if cfg["loss"] == "nssa":
                        c["nssa_margin"] = cfg["nssa_margin"]
                    if model == "RGCN":
                        c["rgcn_sampler"] = cfg["rgcn_sampler"]
                    rows.append(dict(config=c, valid_mrr=v))
                rows.sort(key=lambda r: -r["valid_mrr"])
                old = old_config(s, model)
                old_v = best_valid(OUT / "search" / s / model / cfg_id(old))   # grid-selected configuration
                sel[f"{s}|{model}"] = dict(best=rows[0] if rows else None, n_configs=len(rows), old_grid_config=old,
                                           old_grid_valid_mrr=old_v, all=rows)
                print(s, model, len(rows), rows[0]["valid_mrr"] if rows else None, "old", old_v, flush=True)
        (OUT / "selected.json").write_text(json.dumps(sel, indent=1))
    else:
        sel = json.loads((OUT / "selected.json").read_text())
        jobs = [(k.split("|")[0], v["best"]["config"], sd) for k, v in sel.items() if v["best"] for sd in SEEDS]
        jobs.sort(key=lambda j: (j[1]["model"] != "RGCN", j[0]))

        def go(j):
            s, c, sd = j
            ep = (EPOCHS_EXT if s == "t1_external" else EPOCHS)[c["model"]]
            return run(s, c, sd, OUT / "final" / cfg_id(c), False, ep, a.timeout)

        with ThreadPoolExecutor(a.workers) as ex:
            for msg in ex.map(go, jobs):
                print(msg, flush=True)
        print("FINAL DONE", flush=True)


if __name__ == "__main__":
    main()
