"""Launch Task 4 sweep. KGE + structural heuristics (CN/RA/L3, sparse so feasible
on the ~3.68M graph); no GNN (message-passing on 3.68M is infeasible). Run:
D:\\Anaconda\\envs\\torch\\python.exe experiments/task4/launch.py  [--smoke]
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
RUN = HERE / "run.py"
RESULTS = HERE / "results"
DEFAULT_MODELS = ["TransE", "DistMult", "ComplEx", "RotatE", "PairRE", "ConvE",
                  "TuckER", "CN", "RA", "L3", "Random", "Popularity"]
SETTINGS = ["A", "B", "C"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--settings", nargs="+", default=SETTINGS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    jobs = [("TransE", "A")] if a.smoke else [(m, s) for m in a.models for s in a.settings]
    extra = ["--max_epochs", "1", "--tag", "smoke"] if a.smoke else []
    for i, (m, s) in enumerate(jobs, 1):
        name = f"{m.lower()}_{s}_seed{a.seed}" + ("_smoke" if a.smoke else "")
        if (RESULTS / f"{name}.json").exists():
            print(f">>> [{i}/{len(jobs)}] SKIP {name}")
            continue
        print(f">>> [{i}/{len(jobs)}] {m}/{s}")
        subprocess.run([sys.executable, str(RUN), "--model", m, "--setting", s,
                        "--seed", str(a.seed)] + extra)


if __name__ == "__main__":
    main()
