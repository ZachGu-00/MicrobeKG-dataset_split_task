"""Shared, typed evaluation and compact split loading for the minimal benchmark."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score

LIGHT = ["CN", "RA", "L3", "Random", "Popularity"]
LEARNED = ["TransE", "DistMult", "ComplEx", "RotatE", "TuckER", "RGCN"]
MODELS = LEARNED + LIGHT
HITS = [1, 3, 5, 10, 20]


def load_case(folder, setting, split_seed, smoke=False):
    folder = Path(folder)
    meta = json.loads((folder / "splits" / setting / f"seed_{split_seed}.json").read_text())
    raw = np.load(folder / "splits" / setting / f"seed_{split_seed}.npz", allow_pickle=False)
    a = {k: raw[k] for k in raw.files}
    x = np.load(folder / "triples.npy", mmap_mode="r")
    entities = pd.read_csv(folder / "entities.tsv", sep="\t", dtype={"node_id": str})
    relations = json.loads((folder / "relations.json").read_text())
    if smoke:
        # Real data, at most 512 training triples and four held-out test triples.
        # Preserve a train edge for every evaluated endpoint so smoke tests can
        # exercise prediction rather than silently dropping all tiny-test rows.
        for key, mask in [("valid", "valid_eligible"), ("test", "test_eligible"),
                          ("negatives", "negative_eligible"), ("conflict", "conflict_eligible")]:
            a[key] = a[key][a[mask]][:4]
        for key in ["probe_positive", "random_control_positive"]:
            if key in a:
                a[key] = a[key][:4]
        if "random_control" in a:
            a["random_control"] = a["random_control"][:4]
        endpoints = set()
        for key in ["valid", "test", "negatives", "conflict", "probe_positive", "random_control_positive"]:
            if key in a:
                endpoints.update(x[a[key]][:, [0, 2]].ravel())
        if "random_control" in a:
            endpoints.update(a["random_control"][:, [0, 2]].ravel())
        selected = list(np.random.default_rng(0).choice(a["train"], min(256, len(a["train"])), replace=False))
        for idx in a["train"]:
            h, _, t = x[idx]
            if h in endpoints or t in endpoints:
                selected.append(idx); endpoints.discard(h); endpoints.discard(t)
            if not endpoints:
                break
        for name in meta["scoring_relations"]:
            selected.extend(a["train"][x[a["train"], 1] == relations.index(name)][:4])
        a["train"] = np.unique(selected).astype(np.int32)
        assert len(a["train"]) <= 512
        # Never use full target/test labels in smoke validation filtering.
        a["known_extra"] = np.array([], dtype=np.int32)
        for key in ["taxonomy_hop", "ontology_hop"]:
            if key in a:
                a.pop(key)
    train = np.array(x[a["train"]])
    if setting == "t2_c100":
        train[train[:, 1] == relations.index("utilizes"), 1] = relations.index("can_utilize")
    seen = np.zeros(len(entities), bool); seen[train[:, [0, 2]].ravel()] = True
    eval_seen = seen & a["candidate_eligible"]
    eligible = lambda rows: eval_seen[rows[:, 0]] & eval_seen[rows[:, 2]]
    for key, mask in [("valid", "valid_eligible"), ("test", "test_eligible"),
                      ("negatives", "negative_eligible"), ("conflict", "conflict_eligible")]:
        calculated = eligible(x[a[key]])
        if not smoke:
            assert np.array_equal(calculated, a[mask]), (setting, mask)
        a[mask] = calculated
    return dict(meta=meta, arrays=a, triples=x, train=train, seen=seen, eval_seen=eval_seen,
                entities=entities, relations=relations, smoke=smoke)


def scored_triples(case, rows):
    rows = np.array(rows, copy=True)
    scoring = case["meta"]["scoring_relations"]
    if len(scoring) == 1:
        rows[:, 1] = case["relations"].index(scoring[0])
    return rows


def known_maps(case, validation=False):
    a, x = case["arrays"], case["triples"]
    blocks = [case["train"], scored_triples(case, x[a["valid"]])]
    if not validation:
        blocks += [scored_triples(case, x[a["test"]]), scored_triples(case, x[a["known_extra"]])]
    kt, kh = defaultdict(set), defaultdict(set)
    scoring = {case["relations"].index(r) for r in case["meta"]["scoring_relations"]}
    for block in blocks:
        for h, r, t in block[np.isin(block[:, 1], list(scoring))]:
            kt[(int(h), int(r))].add(int(t)); kh[(int(r), int(t))].add(int(h))
    return kt, kh


def filtered_rank(scores, target_position, known_positions):
    assert np.isfinite(scores).all(), "non-finite model predictions"
    keep = np.ones(len(scores), bool)
    keep[np.asarray(known_positions, dtype=int)] = False
    keep[target_position] = False
    other, truth = scores[keep], scores[target_position]
    return 1 + np.count_nonzero(other > truth) + 0.5 * np.count_nonzero(other == truth)


class LightScorer:
    def __init__(self, train, n, model, seed):
        self.model, self.seed, self.n = model, seed, n
        h, t = train[:, 0], train[:, 2]
        self.degree = np.bincount(np.r_[h, t], minlength=n)
        if model in ["CN", "RA", "L3"]:
            self.adj = (sp.csr_matrix((np.ones(2*len(h)), (np.r_[h, t], np.r_[t, h])), shape=(n, n)) > 0).astype(np.float32)
            deg = np.asarray(self.adj.sum(axis=1)).ravel()
            self.inv = sp.diags(1 / np.maximum(deg, 1))
            self.sqrt_inv = sp.diags(1 / np.sqrt(np.maximum(deg, 1)))

    def random_pairs(self, h, t):
        # Stable per-pair scores, identical in pair/head/tail evaluation.
        with np.errstate(over="ignore"):
            z = np.asarray(h, dtype=np.uint64) * np.uint64(0x9e3779b97f4a7c15)
            z = z + np.asarray(t, dtype=np.uint64) + np.uint64(self.seed)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
            z = z ^ (z >> np.uint64(31))
        return (z >> np.uint64(11)).astype(float) / 2**53

    def paths(self, queries):
        v = self.adj[queries]
        if self.model == "CN":
            return v @ self.adj
        if self.model == "RA":
            return (v @ self.inv) @ self.adj
        return (((v @ self.sqrt_inv) @ self.adj) @ self.sqrt_inv) @ self.adj

    def candidates(self, queries, relations, pool, side):
        if self.model == "Random":
            h, t = (queries[:, None], pool[None, :]) if side == "tail" else (pool[None, :], queries[:, None])
            return self.random_pairs(h, t)
        if self.model == "Popularity":
            return self.degree[queries, None] * self.degree[pool][None, :]
        return self.paths(queries)[:, pool].toarray()

    def triples(self, rows):
        if self.model == "Random":
            return self.random_pairs(rows[:, 0], rows[:, 2])
        if self.model == "Popularity":
            return self.degree[rows[:, 0]] * self.degree[rows[:, 2]]
        values = np.empty(len(rows))
        for h in np.unique(rows[:, 0]):
            mask = rows[:, 0] == h
            values[mask] = self.paths(np.array([h]))[:, rows[mask, 2]].toarray().ravel()
        return values


class LearnedScorer:
    def __init__(self, model, global_to_local, slice_size=2048):
        self.model, self.mapping, self.slice_size = model, global_to_local, slice_size

    def candidates(self, queries, relations, pool, side):
        import torch
        q, p = self.mapping[queries], self.mapping[pool]
        assert (q >= 0).all() and (p >= 0).all()
        tensor = lambda z: torch.as_tensor(z, dtype=torch.long, device=self.model.device)
        with torch.inference_mode():
            if side == "tail":
                out = self.model.predict_t(tensor(np.column_stack([q, relations])), tails=tensor(p), slice_size=self.slice_size)
            else:
                out = self.model.predict_h(tensor(np.column_stack([relations, q])), heads=tensor(p), slice_size=self.slice_size)
        return out.cpu().numpy()

    def triples(self, rows):
        import torch
        mapped = rows.copy(); mapped[:, 0] = self.mapping[rows[:, 0]]; mapped[:, 2] = self.mapping[rows[:, 2]]
        assert (mapped >= 0).all()
        values = []
        with torch.inference_mode():
            for start in range(0, len(mapped), 1024):
                scores = self.model.predict_hrt(torch.as_tensor(mapped[start:start+1024], dtype=torch.long, device=self.model.device))
                values.extend(scores.cpu().numpy().ravel())
        return np.asarray(values)


def evaluate_ranks(case, scorer, validation=False, valid_cap=2048, batch=16):
    a, x, meta = case["arrays"], case["triples"], case["meta"]
    key = "valid" if validation else "test"
    idx = a[key][a[f"{key}_eligible"]]
    if validation and len(idx) > valid_cap:
        idx = np.sort(np.random.default_rng(meta["seed"]).choice(idx, valid_cap, replace=False))
    rows = scored_triples(case, x[idx])
    known = known_maps(case, validation)
    sides = ["tail"] if meta["ranking_sides"] == "tail" else ["tail", "head"]
    raw = {"ranking_row_ids": idx, "ranking_triples": rows}
    for side in sides:
        query_col, target_col = (0, 2) if side == "tail" else (2, 0)
        wanted = meta["tail_type"] if side == "tail" else meta["head_type"]
        pool = np.flatnonzero((case["entities"].node_type.to_numpy() == wanted) & case["eval_seen"])
        pos = {int(v): i for i, v in enumerate(pool)}
        groups = defaultdict(list)
        for i, (h, r, t) in enumerate(rows):
            groups[(int(rows[i, query_col]), int(r))].append(i)
        ranks = np.empty(len(rows))
        queries = list(groups)
        for start in range(0, len(queries), batch):
            chunk = queries[start:start+batch]
            scores = scorer.candidates(np.array([q for q, _ in chunk]), np.array([r for _, r in chunk]), pool, side)
            for j, (q, r) in enumerate(chunk):
                gold = known[0].get((q, r), ()) if side == "tail" else known[1].get((r, q), ())
                positions = [pos[v] for v in gold if v in pos]
                for i in groups[(q, r)]:
                    ranks[i] = filtered_rank(scores[j], pos[int(rows[i, target_col])], positions)
        raw[f"{side}_ranks"] = ranks
        raw[f"{side}_pool_size"] = np.array(len(pool))
    return raw


def ranking_metrics(raw, seed=42, n_boot=1000):
    if not len(raw["ranking_row_ids"]):
        return None
    sides = [s for s in ["tail", "head"] if f"{s}_ranks" in raw]
    per_side, macros = {}, []
    for side in sides:
        ranks = raw[f"{side}_ranks"]
        groups = raw["ranking_triples"][:, 0 if side == "tail" else 2]
        _, inv = np.unique(groups, return_inverse=True)
        macro = float((np.bincount(inv, weights=1/ranks) / np.bincount(inv)).mean())
        per_side[side] = dict(mrr=float((1/ranks).mean()), macro_mrr=macro,
                              **{f"hits_at_{k}": float((ranks <= k).mean()) for k in HITS})
        macros.append(macro)
    ranks = np.concatenate([raw[f"{s}_ranks"] for s in sides])
    # Resample triples, keeping head/tail ranks of the same triple together.
    reciprocal = np.mean([1/raw[f"{s}_ranks"] for s in sides], axis=0)
    rng = np.random.default_rng(seed)
    boot = [float(rng.choice(reciprocal, len(reciprocal), replace=True).mean()) for _ in range(n_boot)]
    return dict(mrr=float((1/ranks).mean()), macro_mrr=float(np.mean(macros)),
                mrr_ci95=np.quantile(boot, [0.025, 0.975]).tolist() if boot else None,
                n_triples=len(reciprocal), n_ranks=len(ranks), by_side=per_side,
                **{f"hits_at_{k}": float((ranks <= k).mean()) for k in HITS})


def discrimination_metrics(pos, neg, seed=42, n_boot=1000):
    assert np.isfinite(pos).all() and np.isfinite(neg).all()
    if not len(pos) or not len(neg):
        return dict(n_pos=len(pos), n_neg=len(neg), auroc=None, auprc=None)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    values = np.r_[pos, neg]
    floor = len(pos) / len(values)
    auc, ap = float(roc_auc_score(y, values)), float(average_precision_score(y, values))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        s = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        boot.append([roc_auc_score(y, s), average_precision_score(y, s)])
    ci = np.quantile(boot, [0.025, 0.975], axis=0) if boot else None
    return dict(n_pos=len(pos), n_neg=len(neg), auroc=auc, auprc=ap, floor=floor,
                auprc_over_floor=ap/floor, auroc_ci95=ci[:, 0].tolist() if ci is not None else None,
                auprc_ci95=ci[:, 1].tolist() if ci is not None else None)


def fingerprint(case):
    h = hashlib.sha256()
    for key in ["train", "valid", "test", "negatives", "conflict"]:
        h.update(case["arrays"][key].tobytes())
    return h.hexdigest()
