"""Plot the scale and benchmark readiness of microbiome knowledge resources."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "kg_landscape_comparison.png"

# Counts are from the cited releases in README.md. HMDAD and Disbiome are
# commonly used association-matrix subsets rather than multi-relational KGs.
SCALE = {
    "MicrobeKG v3.3": (69_090, 3_687_015),
    "MetagenomicKG": (1_250_000, 56_000_000),
    "MGMLink": (782_466, 5_076_297),
    "MicrobiomeKG v2.1": (27_772, 112_118),
    "HMDAD": (292 + 39, 450),
    "Disbiome subset": (1_582 + 352, 8_645),
}

RESOURCES = [
    "MicrobeKG v3.3",
    "MetagenomicKG",
    "MGMLink",
    "MicrobiomeKG v2.1",
    "KG-Microbe",
    "HMDAD",
    "Disbiome",
]
FEATURES = [
    "Multi-\nrelational",
    "Edge\nprovenance",
    "Standard\nKGC splits",
    "Inductive /\ncold-start",
    "Leakage\naudit",
    "Multi-model\nleaderboard",
]

# 1 = explicitly provided, 0.5 = partial/contextual support, 0 = not reported
# in the cited public release. "Not reported" is not evidence of absence.
READINESS = np.array(
    [
        [1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0.5, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
    ],
    dtype=float,
)


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.2, 5.3), gridspec_kw={"width_ratios": [1.05, 1.35]}
    )

    colors = {
        "MicrobeKG v3.3": "#c92a2a",
        "MetagenomicKG": "#1864ab",
        "MGMLink": "#5f3dc4",
        "MicrobiomeKG v2.1": "#2b8a3e",
        "HMDAD": "#868e96",
        "Disbiome subset": "#495057",
    }
    offsets = {
        "MicrobeKG v3.3": (7, 8),
        "MetagenomicKG": (-82, 8),
        "MGMLink": (7, -14),
        "MicrobiomeKG v2.1": (7, 7),
        "HMDAD": (7, 5),
        "Disbiome subset": (7, 7),
    }
    for name, (nodes, edges) in SCALE.items():
        size = 105 if name == "MicrobeKG v3.3" else 58
        ax1.scatter(
            nodes,
            edges,
            s=size,
            color=colors[name],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax1.annotate(
            name,
            (nodes, edges),
            xytext=offsets[name],
            textcoords="offset points",
            fontweight="bold" if name == "MicrobeKG v3.3" else "normal",
        )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", color="#dee2e6", linewidth=0.6, alpha=0.8)
    ax1.set_xlabel("Nodes (log scale)")
    ax1.set_ylabel("Edges or association records (log scale)")
    ax1.set_title("A. Scale is not the differentiator", loc="left", fontweight="bold")
    ax1.text(
        0.02,
        0.90,
        "HMDAD/Disbiome points are bipartite benchmark subsets.",
        transform=ax1.transAxes,
        fontsize=8,
        color="#495057",
    )

    cmap = ListedColormap(["#f1f3f5", "#ffd43b", "#2f9e44"])
    ax2.imshow(READINESS, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(FEATURES)), FEATURES)
    ax2.set_yticks(range(len(RESOURCES)), RESOURCES)
    ax2.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)
    ax2.tick_params(length=0)
    for y in range(len(RESOURCES)):
        for x in range(len(FEATURES)):
            value = READINESS[y, x]
            label = "Yes" if value == 1 else ("Partial" if value == 0.5 else "NR")
            ax2.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value == 1 else "#343a40",
                fontweight="bold" if value == 1 else "normal",
            )
    ax2.set_xticks(np.arange(-0.5, len(FEATURES), 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, len(RESOURCES), 1), minor=True)
    ax2.grid(which="minor", color="white", linewidth=1.5)
    ax2.set_title(
        "B. Benchmark readiness in the cited public releases",
        loc="left",
        fontweight="bold",
        pad=34,
    )
    ax2.text(
        0,
        -0.13,
        "NR = not reported; Partial = contextual rather than benchmark-grade support.",
        transform=ax2.transAxes,
        fontsize=8,
        color="#495057",
    )

    fig.suptitle(
        "MicrobeKG occupies a benchmark-oriented design point",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
