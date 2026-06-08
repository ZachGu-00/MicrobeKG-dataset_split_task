"""One-click runner for the full 4-task KGC benchmark. Single seed, resumable.

Drives run.py directly (one subprocess per task/setting/model), so this file owns
all routing. Safe to Ctrl-C / re-run: any finished result JSON is skipped (断点续跑).

Speed policy (measured 2026-05-30):
- Heavy models (ConvE/TuckER/RGCN) are GPU-bound and cost 4-8h each on the 3.6M+
  graphs while barely training (best_epoch<=25). They are SKIPPED on big-graph
  settings (task1 with_bridges, all of task3/task4) and kept only on the smaller
  graphs. Big graphs there run the 5 light KGE + heuristics + trivial floors.
- Light KGE are CPU/data-pipeline bound (GPU-util ~19%), so a bigger batch barely
  helps them but does no harm (1.4GB at batch 8192); heavy KGE ARE GPU-bound, so a
  bigger batch speeds them up. Hence per-model batch below.

Run:  /root/miniconda3/bin/python -u experiments/run_all.py
  --seed 42         single seed (default)
  --tasks 1 3       subset (default 1 2 3 4)
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

LIGHT_KGE = ["TransE", "DistMult", "ComplEx", "RotatE", "PairRE"]
HEAVY = ["ConvE", "TuckER", "RGCN"]          # skipped on big graphs
HEUR = ["CN", "RA", "L3"]
TRIVIAL = ["Random", "Popularity"]

# per-task settings; `big` = 3.6M-edge settings where HEAVY models are skipped.
# `gnn` = whether RGCN is in the model set at all (off for task3/4: message passing
# on 3.66M is infeasible).
PLAN = {
    "1": {"dir": "task1",
          "settings": ["transductive", "transductive_with_bridges", "cold_microbe", "cold_disease"],
          "big": {"transductive_with_bridges"}, "gnn": True},
    "2": {"dir": "task2", "settings": ["A", "B", "C50", "C100", "C500"], "big": set(), "gnn": True},
    "3": {"dir": "task3", "settings": ["A", "B"], "big": {"A", "B"}, "gnn": False},
    "4": {"dir": "task4", "settings": ["A", "B", "C"], "big": {"A", "B", "C"}, "gnn": False},
}

# per-model training batch (light: harmless big batch; heavy: big batch helps the
# GPU-bound ones; only heavy KGE land on graphs where it matters = task2).
BATCH = {m: 8192 for m in LIGHT_KGE}
BATCH.update({m: 4096 for m in HEAVY})


def models_for(plan, setting):
    if setting in plan["big"]:
        kge = LIGHT_KGE                       # drop ConvE/TuckER/RGCN on big graphs
    else:
        kge = LIGHT_KGE + ["ConvE", "TuckER"] + (["RGCN"] if plan["gnn"] else [])
    return kge + HEUR + TRIVIAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tasks", nargs="+", default=["1", "2", "3", "4"], choices=["1", "2", "3", "4"])
    args = ap.parse_args()

    jobs = []
    for tk in args.tasks:
        plan = PLAN[tk]
        for s in plan["settings"]:
            for m in models_for(plan, s):
                jobs.append((plan["dir"], s, m))
    total = len(jobs)
    print(f"=== {total} jobs, seed {args.seed} ===", flush=True)

    failures = []
    for i, (d, s, m) in enumerate(jobs, 1):
        results = HERE / d / "results"
        name = f"{m.lower()}_{s}_seed{args.seed}"
        if (results / f"{name}.json").exists():
            print(f">>> [{i}/{total}] SKIP {d}/{name}", flush=True)
            continue
        cmd = [sys.executable, str(HERE / d / "run.py"), "--model", m,
               "--setting", s, "--seed", str(args.seed)]
        if m in BATCH:
            cmd += ["--batch", str(BATCH[m])]
        tag = f" batch={BATCH[m]}" if m in BATCH else ""
        print(f">>> [{i}/{total}] {d} {m}/{s}{tag}", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"    FAILED rc={rc} {d}/{name}", flush=True)
            failures.append(f"{d}/{name}")

    print("\n=== COVERAGE ===", flush=True)
    for tk in args.tasks:
        d = PLAN[tk]["dir"]
        rd = HERE / d / "results"
        n = len(list(rd.glob("*.json"))) if rd.is_dir() else 0
        print(f"{d}: {n} json", flush=True)
    if failures:
        print("FAILED: " + ", ".join(failures), flush=True)


if __name__ == "__main__":
    main()
