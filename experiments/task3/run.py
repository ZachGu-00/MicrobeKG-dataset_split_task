"""Task 3 (substrate-utilization inference) — one model on one setting, full metrics.

train graph = final_kg_edges - removed_can_utilize.tsv (bridge edges held out; the
model must RECOVER the missing substrate-utilization edges). Settings: A
(transductive completion, ceiling) | B (cold-microbe relation-level zero-shot).
Models: KGE / structural (CN/RA/L3) / trivial (Random/Popularity). No GNN.

Metric = can_utilize recovery: full ranking (tail+head, micro/macro, MRR CI) over
the substrate/microbe pools, filtered. Raw dump -> tax-layer stratification + the
genuine-(S,D) downstream / novelty-precision recomputed offline (aggregate.py).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (
    SPLITS, load_hrt, build_training_array, make_factories, run_training,
    relation_pools, build_known, kge_rank_instances,
    light_rank_instances, make_light_scorer, LIGHT, HEURISTICS,
    stopper_summary, save_results, add_common_args, load_valid,
)
from metrics import ranking_block, save_raw

TASK3 = SPLITS / "task3_substrate_disease"
RESULTS = Path(__file__).parent / "results"
SETTING_DIRS = {"A": "setting_a_transductive_completion", "B": "setting_b_relzeroshot"}
TARGET = {"can_utilize"}


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--setting", required=True, choices=list(SETTING_DIRS))
    args = ap.parse_args()
    sdir = TASK3 / SETTING_DIRS[args.setting]
    assert sdir.is_dir(), f"missing {sdir}"

    train_arr = build_training_array(sdir, removed_name="removed_can_utilize.tsv")
    valid_arr = load_valid(sdir, train_arr, args.seed)
    test_arr = load_hrt(sdir / "test.tsv")

    head_pool, tail_pool = relation_pools(train_arr, test_arr, TARGET)
    kt, kh = build_known([train_arr, valid_arr, test_arr], TARGET)
    is_light = args.model in LIGHT
    name = f"{args.model.lower()}_{args.setting}_seed{args.seed}" + (f"_{args.tag}" if args.tag else "")
    print(f"[T3/{args.model}/{args.setting} seed={args.seed}] {'light' if is_light else 'kge'} "
          f"train={len(train_arr):,} test(held can_utilize)={len(test_arr):,} "
          f"pools(h/t)={len(head_pool)}/{len(tail_pool)} cuda={torch.cuda.is_available()}", flush=True)

    t0 = time.time()
    if is_light:
        scorer = make_light_scorer(args.model, train_arr, args.seed)
        ranks = light_rank_instances(scorer, test_arr, head_pool, tail_pool, kt, kh)
        early = None
    else:
        train, valid = make_factories(train_arr, valid_arr, args.model in ("CompGCN", "ConvE"))
        result = run_training(args.model, train, valid, valid, args)
        ranks = kge_rank_instances(result.model, train, test_arr, head_pool, tail_pool, kt, kh)
        early = stopper_summary(result)
    elapsed = time.time() - t0

    rk = ranking_block(ranks["tail_rank"], ranks["tail_group"], ranks["head_rank"], ranks["head_group"], seed=args.seed)
    save_raw(RESULTS, name, tail_rank=ranks["tail_rank"], tail_group=ranks["tail_group"],
             head_rank=ranks["head_rank"], head_group=ranks["head_group"])

    out = {"task": "task3", "model": args.model, "setting": args.setting, "seed": args.seed,
           "model_type": ("structural" if args.model in HEURISTICS else "trivial") if is_light else "kge",
           "elapsed_sec": round(elapsed, 1), "num_train": len(train_arr),
           "num_test_eval": ranks["n_eval"], "num_test_oov": ranks["n_oov"],
           "early_stopping": early, "can_utilize_recovery": rk,
           "stratified_note": "recompute tax-layer strata offline from raw + cold_microbe_tax_proximity.tsv"}
    if args.setting == "B":
        ds = pd.read_csv(sdir / "downstream_sd_test.tsv", sep="\t", dtype=str)
        out["downstream_genuine_pairs"] = int(ds["evidence"].str.contains("genuine_discovery=yes").sum())
        out["downstream_total_pairs"] = int(len(ds))
    save_results(out, RESULTS, name)
    bm = rk["both"]
    print(f"  can_utilize recovery: MRR(both,micro)={bm['micro']['mrr']:.4f} macro={bm['macro']['mrr']:.4f} "
          f"gap={bm['micro_minus_macro_gap']['mrr']:.4f} H@10={bm['micro']['hits_at_10']:.4f} "
          f"(n={ranks['n_eval']}, oov={ranks['n_oov']})", flush=True)
    print(f"  saved -> {RESULTS / (name + '.json')}", flush=True)


if __name__ == "__main__":
    main()
