"""Negative-ratio robustness of the hard-negative discrimination metrics.

Every discrimination setting's raw npz holds the per-instance pos/neg score
arrays, so we resample to different neg:pos ratios OFFLINE (no model rerun) and
recompute AUPRC / AUROC / AUPRC-over-floor. Point: AUROC is rank-based and so
near-invariant to the ratio (the real discrimination signal), while AUPRC tracks
prevalence — so the high "0.9x" AUPRC on the pos-heavy T1/T2 sets is a floor
artifact, not model skill. This is the formal answer to the "inflated 0.9x" risk.

Raw pos/neg key aliases: task1/task4 -> hardneg_pos/hardneg_neg ; task2 B/C ->
pos_scores/neg_scores.

Out: experiments/leaderboard/robustness.{csv,md} (+ robustness.png if matplotlib)
Run: D:\\Anaconda\\envs\\torch\\python.exe experiments/analyze_robustness.py
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).parent
OUT = HERE / "leaderboard"
SEED = 42
MIN_POS = 30
RATIOS = [0.5, 1, 2, 5, 10]          # negatives per positive (subsampled)
POS_KEYS = ("hardneg_pos", "pos_scores")
NEG_KEYS = ("hardneg_neg", "neg_scores")
DISC = {
    "task1": ["transductive", "transductive_with_bridges", "cold_microbe", "cold_disease"],
    "task2": ["B", "C50", "C100", "C500"],
    "task4": ["A", "B", "C"],
}


def load_scores(p):
    d = np.load(p)
    pos = next((d[k] for k in POS_KEYS if k in d.files), None)
    neg = next((d[k] for k in NEG_KEYS if k in d.files), None)
    if pos is None or neg is None:
        return None, None
    return pos[~np.isnan(pos)], neg[~np.isnan(neg)]


def auc_at(pos, neg, keep_pos, keep_neg, rng):
    p = pos if keep_pos >= len(pos) else pos[rng.choice(len(pos), keep_pos, replace=False)]
    n = neg if keep_neg >= len(neg) else neg[rng.choice(len(neg), keep_neg, replace=False)]
    y = np.r_[np.ones(len(p)), np.zeros(len(n))]
    s = np.r_[p, n]
    prev = len(p) / (len(p) + len(n))
    ap = float(average_precision_score(y, s))
    return {"n_pos": len(p), "n_neg": len(n), "prevalence": prev, "auprc": ap,
            "auprc_over_floor": ap / prev, "auroc": float(roc_auc_score(y, s))}


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for task, settings in DISC.items():
        rawdir = HERE / task / "results" / "raw"
        for s in settings:
            for p in sorted(rawdir.glob(f"*_{s}_seed{SEED}.npz")):
                pos, neg = load_scores(p)
                if pos is None or len(pos) == 0 or len(neg) == 0:
                    continue
                model = p.name.split(f"_{s}_seed{SEED}.npz")[0]
                rng = np.random.RandomState(SEED)
                base = auc_at(pos, neg, len(pos), len(neg), rng)
                rows.append({"task": task, "setting": s, "model": model,
                             "ratio": "orig", "neg_per_pos": len(neg) / len(pos), **base})
                for r in RATIOS:
                    keep_pos = min(len(pos), int(len(neg) / r))
                    keep_neg = int(keep_pos * r)
                    if keep_pos < MIN_POS or keep_neg < 1:
                        continue
                    m = auc_at(pos, neg, keep_pos, keep_neg, rng)
                    rows.append({"task": task, "setting": s, "model": model,
                                 "ratio": str(r), "neg_per_pos": float(r), **m})

    fields = ["task", "setting", "model", "ratio", "neg_per_pos", "n_pos", "n_neg",
              "prevalence", "auprc", "auprc_over_floor", "auroc"]
    with (OUT / "robustness.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    labels = ["orig"] + [str(r) for r in RATIOS]
    settings_all = [(t, s) for t, ss in DISC.items() for s in ss]

    def med(task, s, lab, key):
        v = [row[key] for row in rows
             if row["task"] == task and row["setting"] == s and row["ratio"] == lab]
        return float(np.median(v)) if v else None

    def fmt(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) else "-"

    def table(key, title):
        head = "| task·setting | n+ | n- | " + " | ".join(
            ("orig" if l == "orig" else f"{l}:1") for l in labels) + " |"
        sep = "|" + "|".join("---" for _ in range(3 + len(labels))) + "|"
        out = [f"\n### {title}\n", head, sep]
        for t, s in settings_all:
            base = next((row for row in rows
                         if row["task"] == t and row["setting"] == s and row["ratio"] == "orig"), None)
            if not base:
                continue
            cells = [fmt(med(t, s, l, key)) for l in labels]
            out.append(f"| {t}·{s} | {base['n_pos']} | {base['n_neg']} | " + " | ".join(cells) + " |")
        return "\n".join(out)

    md = ["# Negative-ratio robustness of hard-negative discrimination (seed 42)\n",
          "Each setting's raw pos/neg score arrays resampled OFFLINE to different "
          "neg:pos ratios (no model rerun); cells = **median across models**. "
          "`k:1` = k negatives per positive (majority side subsampled to hit the "
          "ratio); `orig` = full set. **AUROC is near-flat across columns "
          "(rank-based, prevalence-invariant → the real signal); AUPRC slides with "
          "prevalence — the high `orig` AUPRC on pos-heavy T1/T2 is a floor "
          "artifact, not skill.** Per-model detail in robustness.csv.\n",
          table("auroc", "AUROC — median across models (read: flat across columns)"),
          table("auprc", "AUPRC — median across models (read: slides with ratio)"),
          table("auprc_over_floor", "AUPRC / floor — median (read: ≈flat = enrichment is the stable quantity)")]
    (OUT / "robustness.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    png = "skipped (no matplotlib)"

    def npp(t, s, lab):
        v = [row["neg_per_pos"] for row in rows
             if row["task"] == t and row["setting"] == s and row["ratio"] == lab]
        return v[0] if v else None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from collections import defaultdict
        # 3 task colour families, settings shaded within
        group_cmaps = {"task1": "Blues", "task2": "Oranges", "task4": "Greens"}
        task_settings = defaultdict(list)
        for t, s in settings_all:
            task_settings[t].append(s)
        color_of = {}
        for t, ss in task_settings.items():
            cmap = plt.get_cmap(group_cmaps[t])
            for i, s in enumerate(ss):
                color_of[(t, s)] = cmap(0.45 + 0.5 * i / max(1, len(ss) - 1))

        fig, ax = plt.subplots(figsize=(10.5, 7.8))
        ax.axvspan(0, 0.5, color="#d62728", alpha=0.05)
        ax.axvspan(0.5, 1, color="#2ca02c", alpha=0.05)
        ax.axvline(0.5, color="gray", ls="--", lw=1)
        # faint per-model cloud (every setting x model x ratio)
        for row in rows:
            ax.scatter(row["auroc"], row["auprc"], color=color_of[(row["task"], row["setting"])],
                       s=9, alpha=0.16, zorder=1, edgecolor="none")
        # median-over-models track per setting (sorted along the ratio sweep)
        for (t, s) in settings_all:
            c = color_of[(t, s)]
            pts = []
            for lab in labels:
                au, ap, x = med(t, s, lab, "auroc"), med(t, s, lab, "auprc"), npp(t, s, lab)
                if au is not None and ap is not None and x is not None:
                    pts.append((x, au, ap, lab == "orig"))
            if not pts:
                continue
            pts.sort(key=lambda z: z[0])
            ax.plot([p[1] for p in pts], [p[2] for p in pts], "-", color=c, lw=1.4, alpha=0.85, zorder=2)
            ax.scatter([p[1] for p in pts if not p[3]], [p[2] for p in pts if not p[3]],
                       color=c, s=20, zorder=3, edgecolor="white", linewidth=0.4)
            ax.scatter([p[1] for p in pts if p[3]], [p[2] for p in pts if p[3]],
                       color=c, s=150, marker="*", zorder=4, edgecolor="black",
                       linewidth=0.6, label=f"{t.replace('task', 'T')}/{s}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("AUROC  -  rank-based, locked per setting (prevalence-invariant)")
        ax.set_ylabel("AUPRC  -  prevalence-driven, slides with neg:pos")
        ax.set_title("Negative-ratio robustness: each track = one setting swept over neg:pos in {0.5:1 ... 10:1}\n"
                     "near-vertical tracks => AUROC fixed while AUPRC ranges ~0.06-0.98;  star = as-reported (orig)",
                     fontsize=9)
        ax.text(0.015, 0.03, "AUROC < 0.5 (red zone): worse than chance -\nranks contradictory edges above clean ones",
                transform=ax.transAxes, fontsize=7.5, color="#b22222", va="bottom")
        ax.legend(title="setting (star=orig)", fontsize=6.3, ncol=1, loc="center left",
                  bbox_to_anchor=(1.005, 0.5), frameon=False)
        fig.tight_layout()
        fig.savefig(OUT / "robustness.png", dpi=145, bbox_inches="tight")
        png = "robustness.png"
    except Exception as e:
        png = f"skipped ({e})"

    print(f"robustness.csv <- {len(rows)} rows")
    print(f"robustness.md  <- {len(settings_all)} settings x {len(labels)} ratios")
    print(f"png: {png}")


if __name__ == "__main__":
    main()
