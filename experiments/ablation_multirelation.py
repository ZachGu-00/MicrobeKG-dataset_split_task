"""Multi-relation ablation (Task 1): microbe-disease-only vs + multi-relation bridges.

The two Task-1 settings `transductive` and `transductive_with_bridges` share the
EXACT same test set (5,531 eval / 195 oov); only the train graph differs —
120,868 microbe-disease edges vs 3,675,029 with the substrate/metabolite/gene
bridges added. So this is a clean ±bridge ablation. Reports ranking (both-MRR
micro, H@10) and hard-negative discrimination (AUROC, AUPRC/floor) side by side,
making the BIDIRECTIONAL effect explicit: bridges dilute ranking but — for
path-additive models like TransE — flip hard-negative discrimination from
inverted (<0.5) to strong (>0.7).

Out: experiments/leaderboard/ablation_multirelation.{md,csv}
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "leaderboard"
T1 = HERE / "task1" / "results"
ONLY, BRIDGE = "transductive", "transductive_with_bridges"

MODEL_ORDER = ["TransE", "DistMult", "ComplEx", "RotatE", "PairRE", "ConvE", "TuckER",
               "RGCN", "CN", "RA", "L3", "Random", "Popularity"]
TYPE = {**{m: "kge" for m in MODEL_ORDER[:8]},
        **{m: "structural" for m in ["CN", "RA", "L3"]},
        **{m: "trivial" for m in ["Random", "Popularity"]}}
TYPE["RGCN"] = "gnn"


def load(setting):
    out = {}
    for p in T1.glob(f"*_{setting}_seed42.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        out[d["model"]] = d
    return out


def rank(d):
    if not d or "ranking" not in d:
        return None, None
    b = d["ranking"]["both"]["micro"]
    return b["mrr"], b["hits_at_10"]


def disc(d):
    a = d.get("hardneg_auroc") if d else None
    if not a or a.get("auroc") is None:
        return None, None
    return a["auroc"], a["auprc_over_floor"]


def fn(x, sign=False):
    if not isinstance(x, (int, float)):
        return "-"
    return f"{x:+.4f}" if sign else f"{x:.4f}"


def main():
    OUT.mkdir(exist_ok=True)
    only, brid = load(ONLY), load(BRIDGE)
    n_only = next(iter(only.values()))["num_train"]
    n_brid = next(iter(brid.values()))["num_train"]

    md = ["# Multi-relation ablation — Task 1 (microbe→disease), same test set\n",
          f"Train graph: **microbe-disease only = {n_only:,} edges** vs "
          f"**+ bridges = {n_brid:,} edges**. Identical test (5,531 eval / 195 oov), "
          "seed 42. `MRR`/`H@10` = ranking (both, micro); `AUROC`/`AUPRC/fl` = "
          "hard-negative discrimination (positives vs `inconsistent_association`). "
          "Δ = (+bridges) − (only). **Bridges dilute ranking (ΔMRR mostly < 0) but "
          "can flip discrimination (ΔAUROC ≫ 0 for path-additive KGE like TransE).**\n",
          "| Model | type | MRR only | MRR +brg | ΔMRR | H@10 only | H@10 +brg "
          "| AUROC only | AUROC +brg | ΔAUROC | AUPRC/fl only | AUPRC/fl +brg |",
          "|" + "|".join("---" for _ in range(13)) + "|"]
    csv_rows = []
    for m in MODEL_ORDER:
        mo, ho = rank(only.get(m))
        mb, hb = rank(brid.get(m))
        ao, fo = disc(only.get(m))
        ab, fb = disc(brid.get(m))
        if mo is None and mb is None and ao is None and ab is None:
            continue
        dmrr = (mb - mo) if (isinstance(mo, float) and isinstance(mb, float)) else None
        dauc = (ab - ao) if (isinstance(ao, float) and isinstance(ab, float)) else None
        md.append("| " + " | ".join([
            m, TYPE.get(m, "-"), fn(mo), fn(mb), fn(dmrr, True), fn(ho), fn(hb),
            fn(ao), fn(ab), fn(dauc, True), fn(fo), fn(fb)]) + " |")
        csv_rows.append({"model": m, "type": TYPE.get(m, ""),
                         "MRR_only": mo, "MRR_bridge": mb, "dMRR": dmrr,
                         "H10_only": ho, "H10_bridge": hb,
                         "AUROC_only": ao, "AUROC_bridge": ab, "dAUROC": dauc,
                         "AUPRC_over_floor_only": fo, "AUPRC_over_floor_bridge": fb})

    (OUT / "ablation_multirelation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with (OUT / "ablation_multirelation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"ablation_multirelation.{{md,csv}} <- {len(csv_rows)} models")


if __name__ == "__main__":
    main()
