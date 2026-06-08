"""One-click runner for the full 4-task KGC benchmark. Single seed, resumable.

Drives run.py directly (one subprocess per task/setting/model), so this file owns
all routing. Safe to Ctrl-C / re-run: any finished result JSON is skipped (断点续跑).

All 13 baselines are scheduled for every setting. Light KGE use batch 8192;
ConvE/TuckER/RGCN use batch 4096.

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
HEAVY = ["ConvE", "TuckER", "RGCN"]
HEUR = ["CN", "RA", "L3"]
TRIVIAL = ["Random", "Popularity"]

PLAN = {
    "1": {"dir": "task1",
          "settings": ["transductive", "transductive_with_bridges", "cold_microbe", "cold_disease"]},
    "2": {"dir": "task2", "settings": ["A", "B", "C50", "C100", "C500"]},
    "3": {"dir": "task3", "settings": ["A", "B"]},
    "4": {"dir": "task4", "settings": ["A", "B", "C"]},
}

BATCH = {m: 8192 for m in LIGHT_KGE}
BATCH.update({m: 4096 for m in HEAVY})


def models_for(plan, setting):
    return LIGHT_KGE + HEAVY + HEUR + TRIVIAL


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
