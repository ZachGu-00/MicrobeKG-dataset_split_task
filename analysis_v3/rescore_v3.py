"""Rescoring of finished cells without retraining (P0-2, P0-4).

For every finished cell (heuristic, PyKEEN checkpoint, ULTRA zero-shot, NCN, and the v3 runners that save weights):
  1. filtered ranks of the test triples under three candidate pools
       all          every eval-seen entity of the target type (the paper's pool; checked against the saved ranks)
       target_all   eval-seen entities with >= 1 enriched_in/depleted_in edge anywhere in the setting's graph
       target_train eval-seen entities with >= 1 enriched_in/depleted_in edge in the training graph
     A rank is NaN when the target entity is outside the pool (only possible for target_train).
  2. per-relation scores of every (eval-seen microbe, GMrepo probe disease) pair, so that every tested-null probe
     variant (thresholds, prevalence matching, species-only, per relation) can be computed offline.  Scores are the
     disease-query (head-side) ranking scores, i.e. the scores a model uses when it ranks microbes for a disease;
     for models with a symmetric score (heuristics, PyKEEN) they equal the forward triple scores of the v2 probes.
Output: results3/rescore/<cell stem>.npz.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/autodl-tmp/MicrobeKG_data")
MKG = Path("/root/autodl-tmp/mkg")
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(MKG))
sys.path.insert(0, str(MKG / "gnn"))
from core import LightScorer, filtered_rank, known_maps, load_case, scored_triples  # noqa: E402

OUT = MKG / "results3/rescore"
GM = ROOT / ("paper_work_collaboration/overleaf/microbekg_p0024_overleaf/review/"
             "20260928_corrected/claude/release/gmrepo/work/family_tests.tsv.gz")
RESULT_DIRS = [MKG / "results2/main/main", MKG / "results2/gnn/main", MKG / "results3/variants/main",
               MKG / "results3/gnn", MKG / "results3/hpo_random/final"]
NAME = re.compile(r"(t1_\w+?)_([a-z0-9-]+)_split(\d+)_seed(\d+)_d(\d+)_n(\d+)_(\w+)\.npz")
PROBE_SETTINGS = {"t1_pair", "t1_study", "t1_study_strict", "t1_temporal", "t1_external"}
LIGHT = {"popularity": "Popularity", "cn": "CN", "ra": "RA", "l3": "L3", "random": "Random"}
PYKEEN = {"transe", "distmult", "complex", "rotate", "tucker", "rgcn"}


def data_root(setting):
    if setting == "t1_external":
        return MKG / "bench_ext"
    if (MKG / "bench3/splits" / setting).exists():
        return MKG / "bench3"
    return MKG / "bench2"


def model_key(model, loss):
    return f"{model}-{loss}" if loss in ("zeroshot", "finetune") else model


def target_mask(rows, types, n):
    m = np.zeros(n, bool)
    m[rows[:, 0]] = True
    m[rows[:, 2]] = True
    return m


def ranks_pools(case, scorer, pools, batch=16):
    a, x, meta = case["arrays"], case["triples"], case["meta"]
    idx = a["test"][a["test_eligible"]]
    rows = scored_triples(case, x[idx])
    known = known_maps(case, False)
    types = case["entities"].node_type.to_numpy()
    out = {"ranking_row_ids": idx}
    for side in ["tail", "head"]:
        qc, tc = (0, 2) if side == "tail" else (2, 0)
        wanted = meta["tail_type"] if side == "tail" else meta["head_type"]
        base = np.flatnonzero((types == wanted) & case["eval_seen"])
        sub = {k: np.flatnonzero(m[base]) for k, m in pools.items()}
        pos = {k: {int(base[j]): i for i, j in enumerate(v)} for k, v in sub.items()}
        groups = defaultdict(list)
        for i in range(len(rows)):
            groups[(int(rows[i, qc]), int(rows[i, 1]))].append(i)
        queries = list(groups)
        res = {k: np.full(len(rows), np.nan) for k in pools}
        for st in range(0, len(queries), batch):
            chunk = queries[st:st + batch]
            scores = scorer.candidates(np.array([q for q, _ in chunk]), np.array([r for _, r in chunk]), base, side)
            for j, (q, r) in enumerate(chunk):
                gold = known[0].get((q, r), ()) if side == "tail" else known[1].get((r, q), ())
                for k in pools:
                    s = scores[j][sub[k]]
                    p = pos[k]
                    positions = [p[v] for v in gold if v in p]
                    for i in groups[(q, r)]:
                        t = int(rows[i, tc])
                        if t in p:
                            res[k][i] = filtered_rank(s, p[t], positions)
        for k in pools:
            out[f"{k}_{side}_ranks"] = res[k]
    return out


def probe_matrix(case, scorer, dis_nodes, direction, batch=16):
    types = case["entities"].node_type.to_numpy()
    rel = [case["relations"].index("enriched_in"), case["relations"].index("depleted_in")]
    mic = np.flatnonzero((types == "microbe") & case["eval_seen"])
    dis = np.asarray([d for d in dis_nodes if case["eval_seen"][d]], dtype=np.int64)
    S = np.empty((2, len(dis), len(mic)), np.float32)
    for k, r in enumerate(rel):
        if direction == "head":           # query = disease, rank microbes
            for st in range(0, len(dis), batch):
                q = dis[st:st + batch]
                S[k, st:st + len(q)] = scorer.candidates(q, np.full(len(q), r), mic, "head")
        else:                             # query = microbe (forward direction, as scorer.triples)
            for st in range(0, len(mic), batch):
                q = mic[st:st + batch]
                S[k, :, st:st + len(q)] = scorer.candidates(q, np.full(len(q), r), dis, "tail").T
    return mic, dis, S


def build_scorer(model, cfg, case, json_path, device):
    if model in LIGHT:
        return LightScorer(case["train"], len(case["seen"]), LIGHT[model], cfg["seed"]), "head"
    if model in PYKEEN:
        from probes_v2 import rebuild_learned
        return rebuild_learned(case, cfg, json_path.with_suffix(".pt"), device), "head"
    if model == "ultra-zeroshot" or model == "ultra-finetune":
        import ultra_run
        sc = ultra_run.UltraScorer(case, device, "ultra_50g.pth", batch=8, load_weights=model == "ultra-zeroshot")
        if model == "ultra-finetune":
            import torch
            sc.model.load_state_dict(torch.load(json_path.with_suffix(".pt"), map_location="cpu", weights_only=True))
            sc.model.eval()
        return sc, "tail"
    if model == "ncn":
        sys.path.insert(0, str(MKG / "baselines_v3"))
        import ncn_run
        return ncn_run.build_scorer(case, json_path.with_suffix(".pt"), device), "head"
    if model == "nbfnet":
        sys.path.insert(0, str(MKG))
        import nbfnet_v3
        return nbfnet_v3.build_scorer(case, json_path.with_suffix(".pt"), device), "tail"
    raise ValueError(model)


def cells(models, settings):
    found = {}
    for d in RESULT_DIRS:
        if not d.exists():
            continue
        for p in d.glob("**/*.npz"):
            m = NAME.match(p.name)
            if not m:
                continue
            s, model, split, seed, dim, negs, loss = m.groups()
            key = model_key(model, loss)
            js = p.with_suffix(".json")
            if not js.exists():
                continue
            if (models and key not in models) or (settings and s not in settings):
                continue
            if key in PYKEEN | {"ncn", "nbfnet", "ultra-finetune"} and not p.with_suffix(".pt").exists():
                continue
            stem = p.stem + ("__rs" if "hpo_random" in str(p) else "")
            found[stem] = (s, key, int(split), p, js)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--settings", nargs="+", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--no-probe", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gm = pd.read_csv(GM, sep="\t", usecols=["disease"]).disease.unique()
    todo = sorted(cells(a.models, a.settings).items())
    todo = [t for i, t in enumerate(todo) if i % a.nshards == a.shard]
    print(len(todo), "cells", flush=True)
    for stem, (s, model, split, npz, js) in todo:
        out = OUT / f"{stem}.npz"
        if out.exists():
            continue
        t0 = time.perf_counter()
        cfg = json.loads(js.read_text())["config"]
        case = load_case(data_root(s), s, split)
        ents = case["entities"]
        n = len(ents)
        types = ents.node_type.to_numpy()
        x = np.asarray(case["triples"])
        rid = [case["relations"].index(r) for r in ["enriched_in", "depleted_in"]]
        md = np.isin(x[:, 1], rid) & (types[x[:, 0]] == "microbe") & (types[x[:, 2]] == "disease")
        tr = case["train"]
        trmd = tr[np.isin(tr[:, 1], rid)]
        pools = {"all": np.ones(n, bool), "target_all": target_mask(x[md], types, n),
                 "target_train": target_mask(trmd, types, n)}
        scorer, direction = build_scorer(model, cfg, case, js, a.device)
        res = ranks_pools(case, scorer, pools)
        old = np.load(npz)
        diff = max(float(np.nanmax(np.abs(res[f"all_{sd}_ranks"] - old[f"{sd}_ranks"]))) for sd in ["tail", "head"])
        same_rows = bool(np.array_equal(res["ranking_row_ids"], old["ranking_row_ids"]))
        extra = {}
        if not a.no_probe and (s in PROBE_SETTINGS or (MKG / "bench3/splits" / s).exists()):
            node = {v: i for i, v in enumerate(ents.node_id)}
            dis_nodes = [node[d] for d in gm if d in node and types[node[d]] == "disease"]
            mic, dis, S = probe_matrix(case, scorer, dis_nodes, "head")
            extra = dict(probe_microbes=mic, probe_diseases=dis, probe_scores=S)
            if direction == "tail":   # asymmetric scorers: also keep the forward (microbe-query) scores
                extra["probe_scores_fwd"] = probe_matrix(case, scorer, dis_nodes, "tail")[2]
        np.savez_compressed(out, **res, check_all_pool_max_abs_rank_diff=np.array(diff), check_same_rows=np.array(same_rows),
                            **extra)
        print(f"{stem} diff={diff:.3g} rows={same_rows} {time.perf_counter() - t0:.0f}s", flush=True)
        del scorer
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
