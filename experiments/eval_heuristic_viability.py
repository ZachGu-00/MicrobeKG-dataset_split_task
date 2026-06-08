"""One-off probe: can structural heuristics (CN/RA/L3) even work per task/setting?

For each setting we build EXACTLY the same G_train the learning models use
(common.build_training_array), then for a sample of test positives report:
  - head_cov / tail_cov : frac of test heads/tails that have >=1 edge in G_train
  - CN>0 / RA>0 / L3>0  : frac of sampled positives whose TRUE (head,tail) pair
                          gets a nonzero heuristic score (direct intersection;
                          L3 = degree-normalised length-3 path count)
If head/tail coverage ~0 -> nodes truly isolated -> heuristics meaningless.
If CN/RA are ~0 -> graph is bipartite for this prediction (a microbe and a
substrate/disease share no neighbour) -> only odd-length L3 could help.

CN/RA are computed by ONE set-intersection on the true pair (cheap on any
graph). L3 is only attempted on small graphs (it 3-hop-expands).

Run: D:\\Anaconda\\envs\\torch\\python.exe experiments/eval_heuristic_viability.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SPLITS, load_hrt, build_training_array

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

SAMPLE = 300
_EMPTY = frozenset()


def build_adj(arr):
    adj = {}
    for h, _, t in arr:
        adj.setdefault(h, set()).add(t)
        adj.setdefault(t, set()).add(h)
    return adj


def cn_ra_pair(adj, h, t):
    """Direct CN / RA on one pair via a single intersection (cheap)."""
    common = adj.get(h, _EMPTY) & adj.get(t, _EMPTY)
    if not common:
        return 0.0, 0.0
    cn = float(len(common))
    ra = float(sum(1.0 / len(adj[w]) for w in common if adj[w]))
    return cn, ra


def l3_pair(adj, h, t):
    """Degree-normalised length-3 path score for one pair (small graphs only)."""
    nt = adj.get(t, _EMPTY)
    s = 0.0
    for x in adj.get(h, _EMPTY):
        kx = len(adj[x])
        if not kx:
            continue
        for y in adj[x] & nt:
            ky = len(adj[y])
            if ky:
                s += 1.0 / np.sqrt(kx * ky)
    return s


def probe(name, train_arr, test_arr, l3=True):
    adj = build_adj(train_arr)
    n = len(test_arr)
    if n == 0:
        print(f"  {name:42s} EMPTY test", flush=True); return
    head_cov = np.mean([h in adj for h in test_arr[:, 0]])
    tail_cov = np.mean([t in adj for t in test_arr[:, 2]])
    idx = np.random.RandomState(0).choice(n, size=min(SAMPLE, n), replace=False)
    sample = test_arr[idx]
    cn_hit = ra_hit = l3_hit = 0
    for h, _, t in sample:
        cn, ra = cn_ra_pair(adj, h, t)
        cn_hit += cn > 0
        ra_hit += ra > 0
        if l3 and l3_pair(adj, h, t) > 0:
            l3_hit += 1
    ns = len(sample)
    l3s = f" L3pos={l3_hit/ns:4.2f}" if l3 else " L3=skip(too large)"
    print(f"  {name:30s} n={n:>7,} head_cov={head_cov:5.2f} tail_cov={tail_cov:5.2f} | "
          f"CNpos={cn_hit/ns:4.2f} RApos={ra_hit/ns:4.2f}{l3s}", flush=True)


def main():
    T1 = SPLITS / "task1_microbe_disease"
    T2 = SPLITS / "task2_capacity_realization"
    T3 = SPLITS / "task3_substrate_disease"
    T4 = SPLITS / "task4_metabolite_disease"

    print("TASK 1 (microbe->disease)  [train.tsv IS G_train, ~121k edges]", flush=True)
    for s in ["transductive", "transductive_with_bridges", "cold_microbe", "cold_disease"]:
        d = T1 / s / "seed_42"
        if d.is_dir():
            probe(s, build_training_array(d), load_hrt(d / "test.tsv"), l3=True)

    print("\nTASK 2 (utilizes: microbe->substrate)  [G_train ~1.35M]", flush=True)
    for k, sd in {"A": "setting_A_within_relation", "B": "setting_B_cross_relation",
                  "C50": "setting_c_fewshot_n50", "C100": "setting_c_fewshot_n100",
                  "C500": "setting_c_fewshot_n500"}.items():
        d = T2 / sd
        if d.is_dir():
            probe(k, build_training_array(d), load_hrt(d / "test.tsv"), l3=False)

    print("\nTASK 3 (can_utilize: microbe->substrate)  [G_train ~3.66M]", flush=True)
    for k, sd in {"A": "setting_a_transductive_completion", "B": "setting_b_relzeroshot"}.items():
        d = T3 / sd
        if d.is_dir():
            probe(k, build_training_array(d, removed_name="removed_can_utilize.tsv"),
                  load_hrt(d / "test.tsv"), l3=False)

    print("\nTASK 4 (treats: metabolite->disease)  [G_train ~3.68M, gene bridge kept]", flush=True)
    for k, sd in {"A": "setting_a_transductive", "B": "setting_b_cross_evidence",
                  "C": "setting_c_cold_metabolite"}.items():
        d = T4 / sd
        if d.is_dir():
            probe(k, build_training_array(d, removed_name="gtrain_removed.tsv"),
                  load_hrt(d / "test.tsv"), l3=False)


if __name__ == "__main__":
    main()
