"""Tier-1 metric core + raw per-instance dumps for the 4-task KGC benchmark.

Design: every run writes a raw dump (per-instance head/tail filtered ranks +
grouping ids; or pos/neg score arrays for AUROC settings). EVERY Tier-1/2/3
number is recomputable from those dumps, so the GPU sweep never has to be re-run
to add a metric. The JSON carries the Tier-1 headline; aggregate.py reads the
dumps for Tier-2/3 (inflation gap, paired tests, calibration, novelty precision).

Conventions
- AUPRC is the primary discrimination metric; AUROC is secondary. Every AUROC
  block ships its random baseline: AUROC floor = 0.5 (imbalance-independent),
  AUPRC floor = positive prevalence. `auprc_over_floor` (= AUPRC / prevalence)
  is the cross-task-comparable fold-enrichment over random.
- Ranking is FULL (every same-type candidate scored, filtered by known gold),
  never sampled-negative. Realistic rank = (optimistic + pessimistic) / 2.
- Two-level averaging: micro (per test instance) AND macro (mean over per-query
  groups), with the gap reported. Tail-prediction groups by head; head-prediction
  groups by tail -- so macro-over-microbe and macro-over-disease both appear.
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

HITS_K = [1, 3, 5, 10, 20]
N_BOOT = 1000
_EMPTY = frozenset()


# ---- ranking -----------------------------------------------------------------
def realistic_rank(scores, true_pos, known_mask):
    """Filtered realistic rank of the true candidate (1-based).
    scores: 1-D over the candidate pool. true_pos: index of the true candidate.
    known_mask: bool over the pool, True = a known gold to filter out (the true
    candidate's own entry is ignored automatically)."""
    ts = scores[true_pos]
    comp = ~known_mask
    comp[true_pos] = False
    cs = scores[comp]
    g = int(np.count_nonzero(cs > ts))
    e = int(np.count_nonzero(cs == ts))
    return g + e / 2.0 + 1.0


def _mrr(r):
    return float(np.mean(1.0 / r))


def rank_summary(ranks):
    r = np.asarray(ranks, float)
    out = {"mrr": _mrr(r), "mean_rank": float(r.mean()), "n": int(r.size)}
    for k in HITS_K:
        out[f"hits_at_{k}"] = float(np.mean(r <= k))
    return out


def macro_micro(ranks, groups):
    """micro (per instance) + macro (mean over per-group score) + micro-macro gap."""
    r = np.asarray(ranks, float)
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    cnt = np.bincount(inv)

    def grp_mean(x):
        return np.bincount(inv, weights=x, minlength=cnt.size) / cnt

    micro = rank_summary(r)
    macro = {"mrr": float(grp_mean(1.0 / r).mean()), "n_groups": int(cnt.size)}
    for k in HITS_K:
        macro[f"hits_at_{k}"] = float(grp_mean((r <= k).astype(float)).mean())
    keys = ["mrr"] + [f"hits_at_{k}" for k in HITS_K]
    gap = {m: micro[m] - macro[m] for m in keys}
    return {"micro": micro, "macro": macro, "micro_minus_macro_gap": gap}


def bootstrap_ci(values, stat, n=N_BOOT, seed=0, alpha=0.05):
    v = np.asarray(values, float)
    if v.size == 0:
        return None
    rng = np.random.RandomState(seed)
    samples = [stat(v[rng.randint(0, v.size, v.size)]) for _ in range(n)]
    return [float(np.percentile(samples, 100 * alpha / 2)),
            float(np.percentile(samples, 100 * (1 - alpha / 2)))]


def ranking_block(tail_ranks, tail_groups, head_ranks, head_groups, seed=0):
    """Headline ranking: tail + head + both, micro/macro/gap, MRR bootstrap CI."""
    tr = np.asarray(tail_ranks, float)
    hr = np.asarray(head_ranks, float)
    both = np.concatenate([tr, hr])
    block = {
        "tail": macro_micro(tr, tail_groups),
        "head": macro_micro(hr, head_groups),
        "both": macro_micro(both, np.concatenate([np.asarray(tail_groups, object),
                                                   np.asarray(head_groups, object)])),
        "both_mrr_ci95": bootstrap_ci(both, _mrr, seed=seed),
        "protocol": "full ranking over same-type candidate pool, filtered "
                    "(train+valid+test), realistic rank; tail grouped by head, "
                    "head grouped by tail",
    }
    # convenience flat keys for the leaderboard (both/micro headline)
    bm = block["both"]["micro"]
    block["both_mean_reciprocal_rank"] = bm["mrr"]
    for k in HITS_K:
        block[f"both_hits_at_{k}"] = bm[f"hits_at_{k}"]
    return block


# ---- discrimination (AUPRC primary) ------------------------------------------
def auc_block(pos_scores, neg_scores, seed=0):
    ps = np.asarray(pos_scores, float); ps = ps[~np.isnan(ps)]
    ns = np.asarray(neg_scores, float); ns = ns[~np.isnan(ns)]
    npos, nneg = int(ps.size), int(ns.size)
    if npos == 0 or nneg == 0:
        return {"num_pos": npos, "num_neg": nneg, "auprc": None, "auroc": None}
    y = np.concatenate([np.ones(npos), np.zeros(nneg)])
    s = np.concatenate([ps, ns])
    prev = npos / (npos + nneg)
    auprc, auroc = float(average_precision_score(y, s)), float(roc_auc_score(y, s))
    rng = np.random.RandomState(seed)
    bpr, bro = [], []
    for _ in range(N_BOOT):
        pi, ni = rng.randint(0, npos, npos), rng.randint(0, nneg, nneg)
        sb = np.concatenate([ps[pi], ns[ni]])
        bpr.append(average_precision_score(y, sb))
        bro.append(roc_auc_score(y, sb))
    ci = lambda b: [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    return {
        "num_pos": npos, "num_neg": nneg, "prevalence_floor": float(prev),
        "auprc": auprc, "auprc_ci95": ci(bpr),
        "auprc_over_floor": float(auprc / prev),
        "auroc": auroc, "auroc_ci95": ci(bro),
        "auroc_floor": 0.5,
        "fpr_at_median_pos": float((ns >= np.median(ps)).mean()),
    }


# ---- raw dump ----------------------------------------------------------------
def save_raw(results_dir, run_name, **arrays):
    raw = Path(results_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    p = raw / f"{run_name}.npz"
    np.savez_compressed(p, **{k: np.asarray(v) for k, v in arrays.items()})
    return p
