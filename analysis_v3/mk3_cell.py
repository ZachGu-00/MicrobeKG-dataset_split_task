"""One frozen benchmark/run.py cell on the v3 training-graph variants (bench3)."""
from pathlib import Path
import sys

ROOT = Path("/root/autodl-tmp/MicrobeKG_data")
sys.path.insert(0, str(ROOT / "benchmark"))
import prepare  # noqa: E402

V3 = ["t1_v3_strict_nocohort", "t1_v3_strict_nooverlap", "t1_v3_strict_nocooc", "t1_v3_temporal_nocooc",
      "t1_v3_temporal_dated", "t1_v3_p11_base", "t1_v3_p11_same", "t1_v3_p11_other",
      "t1_study", "t1_study_reveal", "t1_study_strict", "t1_study_ctx_open", "t1_study_ctx_full",
      "t1_temporal", "t1_temporal_nodis", "t1_external"]
prepare.OPTIONAL.extend(s for s in V3 if s not in prepare.OPTIONAL)
import run  # noqa: E402

if __name__ == "__main__":
    run.run(run.parser().parse_args())
