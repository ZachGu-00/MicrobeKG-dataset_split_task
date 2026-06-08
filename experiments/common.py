"""Shared infrastructure for the 4-task KGC benchmark (PyKEEN backend).

Every task's run.py imports this. Encodes the cross-task design:
- G_train vs supervision: for tasks whose split dir has `gtrain_removed.tsv`
  (T3/T4), the training graph is full_kg MINUS those edges; for T1/T2 the
  split's own train.tsv already IS the constructed G_train + supervision.
- proximity-stratified eval: cold/zero-shot settings ship a *_proximity.tsv;
  ranking metrics are reported per layer, `none` = headline.
- surrogate scoring: zero-shot settings score an unseen target relation with a
  seen surrogate relation (e.g. utilizes via can_utilize).

Run with: D:\\Anaconda\\envs\\torch\\python.exe
"""
import os
import warnings
import logging
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("PYKEEN_HOME", str(REPO / ".pykeen_home"))

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"pykeen\.utils")
warnings.filterwarnings("ignore", message=r".*has parameters, but no reset_parameters.*")
warnings.filterwarnings("ignore", message=r".*RGCN without graph-based sampling.*")
for _n in ("pykeen.training.training_loop", "pykeen.nn.representation",
           "pykeen.models.base", "pykeen.utils", "pykeen.triples.triples_factory"):
    logging.getLogger(_n).setLevel(logging.ERROR)

import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

from metrics import realistic_rank

SPLITS = REPO / "splits"
FINAL_EDGES = REPO / "kg_build" / "final_kg_edges.tsv"

HRT = ["head_id", "relation", "tail_id"]
RANK_METRICS = ["mean_reciprocal_rank", "adjusted_mean_rank_index", "mean_rank"]
HITS_K = [1, 3, 5, 10, 20]
SIDES = ["head", "tail", "both"]


def load_hrt(path: Path) -> np.ndarray:
    df = pd.read_csv(path, sep="\t", usecols=HRT, dtype=str)
    return df[HRT].to_numpy()


def load_valid(setting_dir: Path, train_arr, seed=0, frac=0.02, cap=5000) -> np.ndarray:
    """Validation triples for early stopping. Some settings (T2 few-shot) ship no
    valid.tsv -> fall back to a small deterministic slice of train as an
    early-stopping monitor (not held out; our metrics never read valid except as
    a known-positive filter, where train edges are already known)."""
    p = setting_dir / "valid.tsv"
    if p.exists():
        return load_hrt(p)
    n = min(cap, max(1, int(len(train_arr) * frac)))
    idx = np.random.RandomState(seed).choice(len(train_arr), size=n, replace=False)
    return train_arr[idx]


def build_training_array(setting_dir: Path, removed_name=None) -> np.ndarray:
    """Construct G_train. If a removed-edge file exists (T3: removed_can_utilize.tsv,
    T4: gtrain_removed.tsv), G_train = final_kg_edges - removed. Else = train.tsv (T1/T2)."""
    names = [removed_name] if removed_name else ["gtrain_removed.tsv", "removed_can_utilize.tsv"]
    for nm in names:
        rmf = setting_dir / nm
        if nm and rmf.exists():
            full = load_hrt(FINAL_EDGES)
            rm = load_hrt(rmf)
            rm_keys = {tuple(r) for r in rm}
            return full[np.array([tuple(r) not in rm_keys for r in full])]
    return load_hrt(setting_dir / "train.tsv")


def make_factories(train_arr, valid_arr, inverse=False):
    train = TriplesFactory.from_labeled_triples(train_arr, create_inverse_triples=inverse)
    valid = TriplesFactory.from_labeled_triples(
        valid_arr, entity_to_id=train.entity_to_id, relation_to_id=train.relation_to_id)
    return train, valid


def run_training(model, train, valid, test_for_pk, args):
    return pipeline(
        training=train, validation=valid, testing=test_for_pk,
        model=model, model_kwargs=dict(embedding_dim=args.dim),
        training_kwargs=dict(num_epochs=args.max_epochs, batch_size=args.batch,
                             num_workers=0, use_tqdm_batch=False, sampler=args.sampler),
        optimizer="Adam", optimizer_kwargs=dict(lr=args.lr),
        negative_sampler="basic",
        negative_sampler_kwargs=dict(num_negs_per_pos=args.num_negs),
        stopper="early",
        stopper_kwargs=dict(frequency=args.frequency, patience=args.patience,
                            relative_delta=args.relative_delta,
                            metric="both.realistic.inverse_harmonic_mean_rank"),
        evaluator="rankbased",
        evaluator_kwargs=dict(filtered=True, metrics=["hits_at_k"],
                              metrics_kwargs=[{"k": 10}], add_defaults=True),
        evaluation_kwargs=dict(batch_size=args.eval_batch, use_tqdm=False),
        random_seed=args.seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def build_known(arrays, target_rels=None):
    """Gold for filtered ranking: known_tails[(h,r)] and known_heads[(r,t)],
    pooled over arrays (train+valid+test), restricted to target relations."""
    kt, kh = {}, {}
    for arr in arrays:
        if arr is None or len(arr) == 0:
            continue
        for h, r, t in arr:
            if target_rels is None or r in target_rels:
                kt.setdefault((h, r), set()).add(t)
                kh.setdefault((r, t), set()).add(h)
    return kt, kh


def relation_pools(train_arr, test_arr, target_rels=None):
    """Typed candidate pools (head_pool, tail_pool) = unique heads/tails of the
    target relations across train+test. Ranking is full over these pools."""
    rels = set(target_rels) if target_rels else set(test_arr[:, 1])
    hp, tp = set(test_arr[:, 0]), set(test_arr[:, 2])
    m = np.isin(train_arr[:, 1], list(rels))
    hp.update(train_arr[m, 0])
    tp.update(train_arr[m, 2])
    return sorted(hp), sorted(tp)


def _mask_from(known_set, pos_map, n):
    m = np.zeros(n, dtype=bool)
    for c in known_set:
        i = pos_map.get(c)
        if i is not None:
            m[i] = True
    return m


def kge_rank_instances(model, tf, test_arr, head_pool, tail_pool,
                       known_tails, known_heads, batch=128):
    """Per-instance filtered realistic ranks (tail & head) over the typed pools,
    by scoring every pool candidate. Returns rank + grouping arrays + n_oov."""
    e2i, r2i = tf.entity_to_id, tf.relation_to_id
    tp = [c for c in tail_pool if c in e2i]
    hp = [c for c in head_pool if c in e2i]
    tp_gi = np.array([e2i[c] for c in tp])
    hp_gi = np.array([e2i[c] for c in hp])
    tp_pos = {c: i for i, c in enumerate(tp)}
    hp_pos = {c: i for i, c in enumerate(hp)}
    rows, oov = [], 0
    for h, r, t in test_arr:
        if h in e2i and r in r2i and t in e2i and t in tp_pos and h in hp_pos:
            rows.append((h, r, t))
        else:
            oov += 1
    tr, trg, hk, hg = [], [], [], []
    for s0 in range(0, len(rows), batch):
        ch = rows[s0:s0 + batch]
        hrb = torch.tensor([[e2i[h], r2i[r]] for h, r, _ in ch], device=model.device)
        rtb = torch.tensor([[r2i[r], e2i[t]] for _, r, t in ch], device=model.device)
        with torch.no_grad():
            st = model.predict_t(hrb).detach().cpu().numpy()[:, tp_gi]
            sh = model.predict_h(rtb).detach().cpu().numpy()[:, hp_gi]
        for j, (h, r, t) in enumerate(ch):
            km = _mask_from(known_tails.get((h, r), _EMPTY), tp_pos, len(tp))
            tr.append(realistic_rank(st[j], tp_pos[t], km)); trg.append(h)
            hm = _mask_from(known_heads.get((r, t), _EMPTY), hp_pos, len(hp))
            hk.append(realistic_rank(sh[j], hp_pos[h], hm)); hg.append(t)
    return {"tail_rank": np.array(tr), "tail_group": np.array(trg, object),
            "head_rank": np.array(hk), "head_group": np.array(hg, object),
            "n_eval": len(rows), "n_oov": oov}


def kge_pair_scores(model, tf, pairs, rel):
    """Score (head, tail) pairs under a FIXED relation `rel`; NaN where head/tail/
    rel OOV. Used for AUROC discrimination incl. surrogate scoring (true relation
    unseen in train -> score with a seen surrogate)."""
    e2i, r2i = tf.entity_to_id, tf.relation_to_id
    out = np.full(len(pairs), np.nan)
    if rel not in r2i:
        return out
    rid = r2i[rel]
    rows, where = [], []
    for i, (h, t) in enumerate(pairs):
        if h in e2i and t in e2i:
            rows.append([e2i[h], rid, e2i[t]]); where.append(i)
    if rows:
        with torch.no_grad():
            sc = model.predict_hrt(torch.tensor(rows, dtype=torch.long,
                                   device=model.device)).cpu().numpy().ravel()
        out[np.array(where)] = sc
    return out


def kge_triple_scores(model, tf, triples):
    """Score (h, r, t) triples under their OWN relation; NaN where any id OOV.
    Used where positives and negatives carry different relations (T1 hard-neg:
    enriched/depleted vs inconsistent_association)."""
    e2i, r2i = tf.entity_to_id, tf.relation_to_id
    out = np.full(len(triples), np.nan)
    rows, where = [], []
    for i, (h, r, t) in enumerate(triples):
        if h in e2i and r in r2i and t in e2i:
            rows.append([e2i[h], r2i[r], e2i[t]]); where.append(i)
    if rows:
        with torch.no_grad():
            sc = model.predict_hrt(torch.tensor(rows, dtype=torch.long,
                                   device=model.device)).cpu().numpy().ravel()
        out[np.array(where)] = sc
    return out


def stopper_summary(result):
    st = result.stopper
    return {"stopped_early": getattr(st, "stopped", False),
            "best_epoch": getattr(st, "best_epoch", None),
            "best_metric": getattr(st, "best_metric", None)}


def save_results(obj, results_dir: Path, run_name: str):
    results_dir.mkdir(parents=True, exist_ok=True)
    p = results_dir / f"{run_name}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    return p


def add_common_args(ap):
    ap.add_argument("--model", required=True,
                    choices=["TransE", "DistMult", "ComplEx", "RotatE", "TuckER",
                             "PairRE", "ConvE", "RGCN", "CompGCN",
                             "CN", "RA", "L3", "Random", "Popularity"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_negs", type=int, default=32)
    ap.add_argument("--eval_batch", type=int, default=16)
    ap.add_argument("--sampler", default=None)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--frequency", type=int, default=5)
    ap.add_argument("--relative_delta", type=float, default=0.001)
    ap.add_argument("--tag", default="")
    return ap


# --- non-learning baselines: structural heuristics + trivial floors ---------
# All scored over the SAME G_train the KGE models use (fair, leakage-equivalent).
# Structural (sparse matrix propagation -> every candidate scored per head, so
# full filtered ranking is tractable even on the ~3.66M-edge graphs):
#   CN(h,c) = (A A)[h,c]        common neighbours       (even path)
#   RA(h,c) = (A D^-1 A)[h,c]   resource allocation     (even path, Zhou 2009)
#   L3(h,c) = (A W A W A)[h,c]  len-3 paths, W=D^-1/2    (odd path, Kovacs 2019;
#             survives where the target is a bipartite/cross-type edge and CN=0)
# Trivial floors (Tier-2 lower bracket -- the real test is beating Popularity on
# the hardest layer): Popularity ranks by node degree; Random ranks by noise.
HEURISTICS = ["CN", "RA", "L3"]
TRIVIAL = ["Random", "Popularity"]
LIGHT = HEURISTICS + TRIVIAL
_EMPTY = frozenset()


class _LightScorer:
    """Base for non-learning scorers. set_pools fixes the typed candidate pools;
    tail_vec(h)/head_vec(t) return scores over the tail/head pool; pair_score for
    AUROC. has_head/has_tail drop OOV (= not seen in G_train), matching the KGE
    factory, so every model is evaluated on the same instances."""

    def __init__(self, train_arr):
        self.nodes = set(train_arr[:, 0]) | set(train_arr[:, 2])

    def has_head(self, x):
        return x in self.nodes

    def has_tail(self, x):
        return x in self.nodes

    def set_pools(self, head_ids, tail_ids):
        self.head_ids, self.tail_ids = list(head_ids), list(tail_ids)
        self._prep()

    def _prep(self):
        pass


class StructuralScorer(_LightScorer):
    def __init__(self, train_arr, method):
        super().__init__(train_arr)
        self.method = method
        h, t = train_arr[:, 0], train_arr[:, 2]
        codes, uniques = pd.factorize(np.concatenate([h, t]))
        self.node2i = {v: i for i, v in enumerate(uniques)}
        n = len(uniques)
        hc, tc = codes[:len(h)], codes[len(h):]
        rows, cols = np.concatenate([hc, tc]), np.concatenate([tc, hc])
        A = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
        self.A = (A > 0).astype(np.float32).tocsr()   # collapse multi-edges to 1
        deg = np.asarray(self.A.sum(1)).ravel()
        self.Dinv = sp.diags(np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0))
        self.Dsq = sp.diags(np.divide(1.0, np.sqrt(deg), out=np.zeros_like(deg), where=deg > 0))

    def _prep(self):
        self.tail_gi = np.array([self.node2i.get(c, -1) for c in self.tail_ids])
        self.head_gi = np.array([self.node2i.get(c, -1) for c in self.head_ids])

    def _row(self, x):
        r = self.A.getrow(self.node2i[x])
        if self.method == "CN":
            v = r @ self.A
        elif self.method == "RA":
            v = (r @ self.Dinv) @ self.A
        else:   # L3
            v = (((r @ self.Dsq) @ self.A) @ self.Dsq) @ self.A
        return np.asarray(v.todense()).ravel()

    def _slice(self, x, gi):
        out = np.zeros(len(gi), np.float32)
        if x in self.node2i:
            row = self._row(x)
            m = gi >= 0
            out[m] = row[gi[m]]
        return out

    def tail_vec(self, h):
        return self._slice(h, self.tail_gi)

    def head_vec(self, t):
        return self._slice(t, self.head_gi)

    def pair_score(self, h, t):
        if h not in self.node2i or t not in self.node2i:
            return np.nan
        return float(self._row(h)[self.node2i[t]])


class PopularityScorer(_LightScorer):
    """Score a candidate by its G_train degree (predict the most-connected
    disease/substrate regardless of the query) -- the popularity floor."""

    def __init__(self, train_arr):
        super().__init__(train_arr)
        d = {}
        for h, _, t in train_arr:
            d[h] = d.get(h, 0) + 1
            d[t] = d.get(t, 0) + 1
        self.deg = d

    def _prep(self):
        self.tail_pop = np.array([self.deg.get(c, 0) for c in self.tail_ids], float)
        self.head_pop = np.array([self.deg.get(c, 0) for c in self.head_ids], float)

    def tail_vec(self, h):
        return self.tail_pop

    def head_vec(self, t):
        return self.head_pop

    def pair_score(self, h, t):
        return float(self.deg.get(t, 0))


class RandomScorer(_LightScorer):
    def __init__(self, train_arr, seed):
        super().__init__(train_arr)
        self.rng = np.random.RandomState(seed)

    def tail_vec(self, h):
        return self.rng.random(len(self.tail_ids))

    def head_vec(self, t):
        return self.rng.random(len(self.head_ids))

    def pair_score(self, h, t):
        return float(self.rng.random())


def make_light_scorer(model, train_arr, seed=0):
    if model in HEURISTICS:
        return StructuralScorer(train_arr, model)
    if model == "Popularity":
        return PopularityScorer(train_arr)
    if model == "Random":
        return RandomScorer(train_arr, seed)
    raise ValueError(model)


def light_rank_instances(scorer, test_arr, head_pool, tail_pool,
                         known_tails, known_heads):
    """Per-instance filtered realistic ranks (tail & head) for a non-learning
    scorer; grouped so each query entity is scored once. Same return shape as
    kge_rank_instances."""
    from collections import defaultdict
    tp, hp = list(tail_pool), list(head_pool)
    tp_pos = {c: i for i, c in enumerate(tp)}
    hp_pos = {c: i for i, c in enumerate(hp)}
    scorer.set_pools(hp, tp)
    rows, oov = [], 0
    for h, r, t in test_arr:
        if scorer.has_head(h) and scorer.has_tail(t) and t in tp_pos and h in hp_pos:
            rows.append((h, r, t))
        else:
            oov += 1
    tr, trg = [], []
    by_hr = defaultdict(list)
    for h, r, t in rows:
        by_hr[(h, r)].append(t)
    for (h, r), tails in by_hr.items():
        sv = scorer.tail_vec(h)
        km = _mask_from(known_tails.get((h, r), _EMPTY), tp_pos, len(tp))
        for t in tails:
            tr.append(realistic_rank(sv, tp_pos[t], km)); trg.append(h)
    hk, hg = [], []
    by_rt = defaultdict(list)
    for h, r, t in rows:
        by_rt[(r, t)].append(h)
    for (r, t), heads in by_rt.items():
        sh = scorer.head_vec(t)
        hm = _mask_from(known_heads.get((r, t), _EMPTY), hp_pos, len(hp))
        for h in heads:
            hk.append(realistic_rank(sh, hp_pos[h], hm)); hg.append(t)
    return {"tail_rank": np.array(tr), "tail_group": np.array(trg, object),
            "head_rank": np.array(hk), "head_group": np.array(hg, object),
            "n_eval": len(rows), "n_oov": oov}


def light_pair_scores(scorer, pairs):
    return np.array([scorer.pair_score(h, t) for h, t in pairs], float)
