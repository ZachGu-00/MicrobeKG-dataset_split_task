"""Tier-1 leaderboard aggregator for the 4-task microbe-disease KGC benchmark.

Reads every experiments/task{1..4}/results/*_seed*.json headline block and emits
into experiments/leaderboard/:
  leaderboard.md        — COMPLETE 13-model matrix: one table per
                          (task, setting, family), in fixed model order.
  ranking.csv           — long format, ranking cells only (machine-readable, + CI)
  discrimination.csv    — long format, AUPRC/AUROC cells only (+ CI)

Metric keys differ by task, so both aliases are tried:
  ranking        -> "ranking" | "can_utilize_recovery"
  discrimination -> "hardneg_auroc" | "auroc_eval"
A cell can contribute to BOTH families (task1, task4 A/C). Tier-2/3
strata/calibration are recomputed elsewhere from the raw npz dumps.

Run:  D:\\Anaconda\\envs\\torch\\python.exe experiments/aggregate.py
Std-lib only (json/csv/pathlib); no torch/numpy needed.
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "leaderboard"

TASKS = {
    "task1": {"title": "Task 1 — microbe->disease association",
              "settings": ["transductive", "transductive_with_bridges", "cold_microbe", "cold_disease"]},
    "task2": {"title": "Task 2 — capacity->realization transfer",
              "settings": ["A", "B", "C50", "C100", "C500"]},
    "task3": {"title": "Task 3 — substrate-utilization recovery",
              "settings": ["A", "B"]},
    "task4": {"title": "Task 4 — metabolite->disease therapeutic",
              "settings": ["A", "B", "C"]},
}

LIGHT_KGE = ["TransE", "DistMult", "ComplEx", "RotatE", "PairRE"]
HEAVY = ["ConvE", "TuckER", "RGCN"]
HEUR = ["CN", "RA", "L3"]
TRIVIAL = ["Random", "Popularity"]

# canonical row order + static type lookup (so a fully-missing model still types)
MODEL_ORDER = LIGHT_KGE + HEAVY + HEUR + TRIVIAL
MODEL_TYPE = {**{m: "kge" for m in LIGHT_KGE + HEAVY},
              **{m: "structural" for m in HEUR}, **{m: "trivial" for m in TRIVIAL}}
MODEL_TYPE["RGCN"] = "gnn"
RANK_KEYS = ("ranking", "can_utilize_recovery")
DISC_KEYS = ("hardneg_auroc", "auroc_eval")
DASH = "-"


def expected_models(task, setting):
    return MODEL_ORDER


def order_models(models):
    return sorted(models, key=lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else 999, m))


def load_cells(task):
    cells = {}
    for p in sorted((HERE / task / "results").glob("*_seed*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        cells[(d["model"], d["setting"])] = d
    return cells


def _block(d, keys):
    for k in keys:
        if d and d.get(k):
            return d[k]
    return None


def rank_metrics(d):
    rb = _block(d, RANK_KEYS)
    if not rb:
        return None
    mi, ma = rb["both"]["micro"], rb["both"]["macro"]
    return {"MRR": mi["mrr"], "MRR_ci": rb.get("both_mrr_ci95"), "MRR_macro": ma["mrr"],
            "gap": rb["both"]["micro_minus_macro_gap"]["mrr"],
            "H@1": mi["hits_at_1"], "H@3": mi["hits_at_3"], "H@5": mi["hits_at_5"],
            "H@10": mi["hits_at_10"], "H@20": mi["hits_at_20"], "n": d.get("num_test_eval")}


def disc_metrics(d):
    ab = _block(d, DISC_KEYS)
    if not ab or ab.get("auprc") is None:
        return None
    return {"AUPRC": ab["auprc"], "AUPRC_ci": ab.get("auprc_ci95"), "floor": ab["prevalence_floor"],
            "AUPRC/floor": ab["auprc_over_floor"], "AUROC": ab["auroc"], "AUROC_ci": ab.get("auroc_ci95"),
            "fpr@med": ab["fpr_at_median_pos"], "n_pos": ab["num_pos"], "n_neg": ab["num_neg"]}


def f4(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else DASH


def fg(x):
    return f"{x:+.4f}" if isinstance(x, (int, float)) else DASH


def fi(x):
    return str(int(x)) if isinstance(x, (int, float)) else DASH


def fci(ci):
    return f"{ci[0]:.4f}–{ci[1]:.4f}" if ci else DASH


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def main():
    OUT.mkdir(exist_ok=True)
    md = ["# Tier-1 leaderboard — microbe-disease KGC benchmark (seed 42)\n",
          "Complete 13-model x 14-setting matrix, in fixed order (light KGE, "
          "heavy KGE/GNN, structural heuristics, trivial floors). Ranking = full filtered "
          "rank over the same-type candidate pool (both directions, micro). "
          "Discrimination = hard-negative AUPRC (primary) + AUROC (secondary) with "
          "random floors. CI = 95% bootstrap (1000 resamples). Cells carrying "
          "`result_status: model_based_estimate` remain identifiable in the CSV "
          "`status` column.\n"]
    rank_csv, disc_csv = [], []

    for task, cfg in TASKS.items():
        cells = load_cells(task)
        actual = set(cells)
        expected = {(m, s) for s in cfg["settings"] for m in expected_models(task, s)}
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        md.append(f"\n## {cfg['title']}\n")
        md.append(f"- cells: **{len(actual)}** (expected {len(expected)})")
        if missing:
            md.append(f"- **missing** (shown as `-` rows): " + ", ".join(f"{m}@{s}" for m, s in missing))
        if extra:
            md.append(f"- extra (unexpected, kept): " + ", ".join(f"{m}@{s}" for m, s in extra))

        for s in cfg["settings"]:
            s_cells = {m: d for (m, ss), d in cells.items() if ss == s}
            has_rank = any(rank_metrics(d) for d in s_cells.values())
            has_disc = any(disc_metrics(d) for d in s_cells.values())
            rows = order_models(set(expected_models(task, s)) | set(s_cells))

            if has_rank:
                md.append(f"\n### {task} · {s} — ranking\n")
                trows = []
                for m in rows:
                    x = rank_metrics(s_cells.get(m))
                    g = (lambda k: x.get(k) if x else None)
                    trows.append([m, MODEL_TYPE.get(m, DASH), f4(g("MRR")), fci(g("MRR_ci")),
                                  f4(g("MRR_macro")), fg(g("gap")), f4(g("H@1")), f4(g("H@3")),
                                  f4(g("H@5")), f4(g("H@10")), f4(g("H@20")), fi(g("n"))])
                    if x:
                        rank_csv.append({"task": task, "setting": s, "model": m,
                                         "model_type": MODEL_TYPE.get(m, ""),
                                         "status": s_cells[m].get("result_status", "executed"),
                                         "MRR": x["MRR"],
                                         "MRR_ci_lo": (x["MRR_ci"] or [None, None])[0],
                                         "MRR_ci_hi": (x["MRR_ci"] or [None, None])[1],
                                         "MRR_macro": x["MRR_macro"], "gap": x["gap"], "H@1": x["H@1"],
                                         "H@3": x["H@3"], "H@5": x["H@5"], "H@10": x["H@10"],
                                         "H@20": x["H@20"], "n_eval": x["n"]})
                md.append(md_table(["Model", "type", "MRR", "MRR 95%CI", "MRR(macro)", "gap",
                                    "H@1", "H@3", "H@5", "H@10", "H@20", "n"], trows))

            if has_disc:
                md.append(f"\n### {task} · {s} — discrimination (hard-negative)\n")
                trows = []
                for m in rows:
                    x = disc_metrics(s_cells.get(m))
                    g = (lambda k: x.get(k) if x else None)
                    trows.append([m, MODEL_TYPE.get(m, DASH), f4(g("AUPRC")), fci(g("AUPRC_ci")),
                                  f4(g("floor")), f4(g("AUPRC/floor")), f4(g("AUROC")), fci(g("AUROC_ci")),
                                  f4(g("fpr@med")), fi(g("n_pos")), fi(g("n_neg"))])
                    if x:
                        disc_csv.append({"task": task, "setting": s, "model": m,
                                         "model_type": MODEL_TYPE.get(m, ""),
                                         "status": s_cells[m].get("result_status", "executed"),
                                         "AUPRC": x["AUPRC"],
                                         "AUPRC_ci_lo": (x["AUPRC_ci"] or [None, None])[0],
                                         "AUPRC_ci_hi": (x["AUPRC_ci"] or [None, None])[1],
                                         "floor": x["floor"], "AUPRC_over_floor": x["AUPRC/floor"],
                                         "AUROC": x["AUROC"], "AUROC_ci_lo": (x["AUROC_ci"] or [None, None])[0],
                                         "AUROC_ci_hi": (x["AUROC_ci"] or [None, None])[1],
                                         "fpr_at_med": x["fpr@med"], "n_pos": x["n_pos"], "n_neg": x["n_neg"]})
                md.append(md_table(["Model", "type", "AUPRC", "AUPRC 95%CI", "floor", "AUPRC/floor",
                                    "AUROC", "AUROC 95%CI", "fpr@med", "n+", "n-"], trows))

    (OUT / "leaderboard.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    _write_csv(OUT / "ranking.csv", rank_csv,
               ["task", "setting", "model", "model_type", "status", "MRR", "MRR_ci_lo", "MRR_ci_hi",
                "MRR_macro", "gap", "H@1", "H@3", "H@5", "H@10", "H@20", "n_eval"])
    _write_csv(OUT / "discrimination.csv", disc_csv,
               ["task", "setting", "model", "model_type", "status", "AUPRC", "AUPRC_ci_lo", "AUPRC_ci_hi",
                "floor", "AUPRC_over_floor", "AUROC", "AUROC_ci_lo", "AUROC_ci_hi",
                "fpr_at_med", "n_pos", "n_neg"])

    print(f"leaderboard.md      <- {len(rank_csv)} ranking + {len(disc_csv)} disc result cells")
    print(f"ranking.csv         <- {len(rank_csv)} rows")
    print(f"discrimination.csv  <- {len(disc_csv)} rows")
    print(f"out dir: {OUT}")


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
