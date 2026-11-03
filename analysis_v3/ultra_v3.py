"""ULTRA runner for v3: registers the v3 settings and saves fine-tuned weights next to the result json."""
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/mkg/gnn")
import common  # noqa: E402

V3 = ["t1_v3_strict_nocohort", "t1_v3_strict_nooverlap", "t1_v3_strict_nocooc", "t1_v3_temporal_nocooc",
      "t1_v3_temporal_dated", "t1_v3_p11_base", "t1_v3_p11_same", "t1_v3_p11_other"]
common.SETTINGS.extend(s for s in V3 if s not in common.SETTINGS)
import torch  # noqa: E402
import ultra_run  # noqa: E402

_orig = ultra_run.evaluate_and_save


def _save(case, sc, method, dim, negs, loss, out, cfg, fit, hw, *rest):
    res = _orig(case, sc, method, dim, negs, loss, out, cfg, fit, hw, *rest)
    if loss == "finetune":
        m = case["meta"]
        stem = f"{m['setting']}_ultra_split{m['seed']}_seed{cfg['seed']}_d{dim}_n{negs}_{loss}"
        torch.save(sc.model.state_dict(), Path(out) / f"{stem}.pt")
    return res


ultra_run.evaluate_and_save = _save

if __name__ == "__main__":
    ultra_run.main(ultra_run.parser().parse_args())
