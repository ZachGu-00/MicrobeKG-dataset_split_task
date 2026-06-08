"""Launch Task 2 sweep: models x settings, skipping finished results.
Run: D:\\Anaconda\\envs\\torch\\python.exe experiments/task2/launch.py  [--smoke]
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
RUN = HERE / "run.py"
RESULTS = HERE / "results"
DEFAULT_MODELS = ["TransE", "DistMult", "ComplEx", "RotatE", "PairRE", "ConvE",
                  "TuckER", "RGCN", "CN", "RA", "L3", "Random", "Popularity"]
SETTINGS = ["A", "B", "C50", "C100", "C500"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--settings", nargs="+", default=SETTINGS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        joblist = [("TransE", "B")]
        extra = ["--max_epochs", "1", "--tag", "smoke"]
    else:
        joblist = [(m, s) for m in args.models for s in args.settings]
        extra = []

    total = len(joblist)
    for i, (m, setting) in enumerate(joblist, 1):
        name = f"{m.lower()}_{setting}_seed{args.seed}" + ("_smoke" if args.smoke else "")
        if (RESULTS / f"{name}.json").exists():
            print(f">>> [{i}/{total}] SKIP {name}")
            continue
        print(f">>> [{i}/{total}] {m} / {setting}")
        rc = subprocess.run([sys.executable, str(RUN), "--model", m,
                             "--setting", setting, "--seed", str(args.seed)] + extra).returncode
        if rc != 0:
            print(f"    FAILED rc={rc}")


if __name__ == "__main__":
    main()
