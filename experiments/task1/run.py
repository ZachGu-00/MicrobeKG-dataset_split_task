"""Task 1 (microbe->disease association) — one model on one setting, full metrics.

Settings: transductive | transductive_with_bridges | cold_microbe | cold_disease
Models: KGE (TransE/DistMult/ComplEx/RotatE/TuckER/PairRE/ConvE), GNN (RGCN),
        structural heuristics (CN/RA/L3), trivial floors (Random/Popularity).

Metrics (Tier 1, written to JSON; raw dumped for Tier 2/3):
- full ranking over the disease/microbe candidate pools, filtered, realistic rank;
  tail + head + both, micro & macro (+gap), MRR bootstrap CI
- hard-neg AUPRC(primary)+floor+fold-enrichment + AUROC(secondary)+CIs
  (test positives enriched/depleted vs inconsistent_association)
Raw dump (results/raw/<name>.npz): per-instance head/tail ranks + group ids +
hard-neg pos/neg scores -> stratified/inflation-gap/calibration recomputed offline.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (
    SPLITS, load_hrt, build_training_array, make_factories, run_training,
    relation_pools, build_known, kge_rank_instances, kge_triple_scores,
    light_rank_instances, light_pair_scores, make_light_scorer, LIGHT, HEURISTICS,
    stopper_summary, save_results, add_common_args, load_valid,
)
from metrics import ranking_block, auc_block, save_raw

TASK1 = SPLITS / "task1_microbe_disease"
RESULTS = Path(__file__).parent / "results"
SETTINGS = ["transductive", "transductive_with_bridges", "cold_microbe", "cold_disease"]


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--setting", required=True, choices=SETTINGS)
    args = ap.parse_args()
    sdir = TASK1 / args.setting / f"seed_{args.seed}"
    assert sdir.is_dir(), f"missing {sdir}"

    train_arr = build_training_array(sdir)
    valid_arr = load_valid(sdir, train_arr, args.seed)
    test_arr = load_hrt(sdir / "test.tsv")
    hn_path = sdir / "test_hardneg.tsv"
    hardneg_arr = load_hrt(hn_path) if hn_path.exists() else np.empty((0, 3), dtype=object)

    head_pool, tail_pool = relation_pools(train_arr, test_arr)
    known_tails, known_heads = build_known([train_arr, valid_arr, test_arr])
    name = f"{args.model.lower()}_{args.setting}_seed{args.seed}" + (f"_{args.tag}" if args.tag else "")
    is_light = args.model in LIGHT
    print(f"[T1/{args.model}/{args.setting} seed={args.seed}] {'light' if is_light else 'kge'} "
          f"train={len(train_arr):,} test={len(test_arr):,} hardneg={len(hardneg_arr):,} "
          f"pools(h/t)={len(head_pool)}/{len(tail_pool)} cuda={torch.cuda.is_available()}", flush=True)

    t0 = time.time()
    if is_light:
        scorer = make_light_scorer(args.model, train_arr, args.seed)
        ranks = light_rank_instances(scorer, test_arr, head_pool, tail_pool, known_tails, known_heads)
        pos = light_pair_scores(scorer, list(zip(test_arr[:, 0], test_arr[:, 2])))
        neg = light_pair_scores(scorer, list(zip(hardneg_arr[:, 0], hardneg_arr[:, 2]))) if len(hardneg_arr) else np.array([])
        early = None
    else:
        train, valid = make_factories(train_arr, valid_arr, args.model in ("CompGCN", "ConvE"))
        result = run_training(args.model, train, valid, valid, args)
        ranks = kge_rank_instances(result.model, train, test_arr, head_pool, tail_pool, known_tails, known_heads)
        pos = kge_triple_scores(result.model, train, test_arr)
        neg = kge_triple_scores(result.model, train, hardneg_arr) if len(hardneg_arr) else np.array([])
        early = stopper_summary(result)
    elapsed = time.time() - t0

    rk = ranking_block(ranks["tail_rank"], ranks["tail_group"], ranks["head_rank"], ranks["head_group"], seed=args.seed)
    hn = auc_block(pos, neg, seed=args.seed) if len(neg) else None
    save_raw(RESULTS, name, tail_rank=ranks["tail_rank"], tail_group=ranks["tail_group"],
             head_rank=ranks["head_rank"], head_group=ranks["head_group"],
             hardneg_pos=np.asarray(pos, float), hardneg_neg=np.asarray(neg, float))

    out = {"task": "task1", "model": args.model, "setting": args.setting, "seed": args.seed,
           "model_type": ("structural" if args.model in HEURISTICS else "trivial") if is_light else "kge",
           "elapsed_sec": round(elapsed, 1), "num_train": len(train_arr),
           "num_test_eval": ranks["n_eval"], "num_test_oov": ranks["n_oov"],
           "early_stopping": early, "ranking": rk, "hardneg_auroc": hn,
           "stratified_note": "recompute offline from raw dump + *_proximity.tsv (aggregate.py)"}
    save_results(out, RESULTS, name)
    bm = rk["both"]
    print(f"  MRR(both,micro)={bm['micro']['mrr']:.4f} macro={bm['macro']['mrr']:.4f} "
          f"gap={bm['micro_minus_macro_gap']['mrr']:.4f} H@10(micro)={bm['micro']['hits_at_10']:.4f} "
          f"| hardneg AUPRC={hn['auprc'] if hn else None} (floor={hn['prevalence_floor'] if hn else None}) "
          f"AUROC={hn['auroc'] if hn else None}", flush=True)
    print(f"  saved -> {RESULTS / (name + '.json')}", flush=True)


if __name__ == "__main__":
    main()
