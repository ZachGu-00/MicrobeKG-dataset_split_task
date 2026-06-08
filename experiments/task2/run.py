"""Task 2 (capacity->realization transfer) — one model on one setting, full metrics.

Settings: A (within-relation can_utilize ranking, ceiling) | B (cross-relation
zero-shot: utilizes scored by surrogate can_utilize) | C50/C100/C500 (few-shot).
Models: KGE / GNN(RGCN) / structural (CN/RA/L3) / trivial (Random/Popularity).

A: full ranking (tail+head, micro/macro, MRR CI) over the substrate/microbe pools.
B/C: utilizes vs does_not_utilize -> AUPRC(primary)+floor+fold-enrichment +
AUROC(secondary)+CIs. B's does_not_utilize is tiny -> n reported explicitly.
Raw dump: A ranks+groups; B/C pos/neg scores. Capacity-layer stratification of B
is recomputed offline from the raw dump + test_microbe_capacity_layer.tsv.
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
    relation_pools, build_known, kge_rank_instances, kge_pair_scores,
    light_rank_instances, light_pair_scores, make_light_scorer, LIGHT, HEURISTICS,
    stopper_summary, save_results, add_common_args, load_valid,
)
from metrics import ranking_block, auc_block, save_raw

TASK2 = SPLITS / "task2_capacity_realization"
RESULTS = Path(__file__).parent / "results"
SETTING_DIRS = {
    "A": "setting_A_within_relation", "B": "setting_B_cross_relation",
    "C50": "setting_c_fewshot_n50", "C100": "setting_c_fewshot_n100",
    "C500": "setting_c_fewshot_n500",
}
TARGET = {"can_utilize"}


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--setting", required=True, choices=list(SETTING_DIRS))
    args = ap.parse_args()
    sdir = TASK2 / SETTING_DIRS[args.setting]
    assert sdir.is_dir(), f"missing {sdir}"

    train_arr = build_training_array(sdir)
    valid_arr = load_valid(sdir, train_arr, args.seed)
    test_arr = load_hrt(sdir / "test.tsv")
    hn_path = sdir / "test_hardneg.tsv"
    dnu_arr = load_hrt(hn_path) if hn_path.exists() else np.empty((0, 3), dtype=object)

    is_light = args.model in LIGHT
    needs_surrogate = ("utilizes" not in set(train_arr[:, 1])) and args.setting != "A"
    name = f"{args.model.lower()}_{args.setting}_seed{args.seed}" + (f"_{args.tag}" if args.tag else "")
    print(f"[T2/{args.model}/{args.setting} seed={args.seed}] {'light' if is_light else 'kge'} "
          f"train={len(train_arr):,} test={len(test_arr):,} dnu={len(dnu_arr):,} "
          f"surrogate={needs_surrogate} cuda={torch.cuda.is_available()}", flush=True)

    out = {"task": "task2", "model": args.model, "setting": args.setting, "seed": args.seed,
           "model_type": ("structural" if args.model in HEURISTICS else "trivial") if is_light else "kge",
           "num_train": len(train_arr), "needs_surrogate": needs_surrogate}

    t0 = time.time()
    if is_light:
        scorer = make_light_scorer(args.model, train_arr, args.seed)
        model_obj, tf = None, None
        early = None
    else:
        train, valid = make_factories(train_arr, valid_arr, args.model in ("CompGCN", "ConvE"))
        result = run_training(args.model, train, valid, valid, args)
        model_obj, tf, early = result.model, train, stopper_summary(result)

    if args.setting == "A":
        head_pool, tail_pool = relation_pools(train_arr, test_arr, TARGET)
        kt, kh = build_known([train_arr, valid_arr, test_arr], TARGET)
        if is_light:
            ranks = light_rank_instances(scorer, test_arr, head_pool, tail_pool, kt, kh)
        else:
            ranks = kge_rank_instances(model_obj, tf, test_arr, head_pool, tail_pool, kt, kh)
        rk = ranking_block(ranks["tail_rank"], ranks["tail_group"], ranks["head_rank"], ranks["head_group"], seed=args.seed)
        out["ranking"] = rk
        out["num_test_eval"], out["num_test_oov"] = ranks["n_eval"], ranks["n_oov"]
        save_raw(RESULTS, name, tail_rank=ranks["tail_rank"], tail_group=ranks["tail_group"],
                 head_rank=ranks["head_rank"], head_group=ranks["head_group"])
        bm = rk["both"]
        msg = (f"MRR(both,micro)={bm['micro']['mrr']:.4f} macro={bm['macro']['mrr']:.4f} "
               f"gap={bm['micro_minus_macro_gap']['mrr']:.4f} H@10={bm['micro']['hits_at_10']:.4f}")
    else:
        pos_pairs = list(zip(test_arr[:, 0], test_arr[:, 2]))
        neg_pairs = list(zip(dnu_arr[:, 0], dnu_arr[:, 2])) if len(dnu_arr) else []
        if is_light:
            pos = light_pair_scores(scorer, pos_pairs)
            neg = light_pair_scores(scorer, neg_pairs) if neg_pairs else np.array([])
            scoring = "structural"
        else:
            scoring = "can_utilize" if needs_surrogate else "utilizes"
            pos = kge_pair_scores(model_obj, tf, pos_pairs, scoring)
            neg = kge_pair_scores(model_obj, tf, neg_pairs, scoring) if neg_pairs else np.array([])
        au = auc_block(pos, neg, seed=args.seed)
        au["scoring_relation"] = scoring
        out["auroc_eval"] = au
        out["stratified_note"] = "recompute capacity-layer strata offline from raw + test_microbe_capacity_layer.tsv"
        save_raw(RESULTS, name, pos_scores=np.asarray(pos, float), neg_scores=np.asarray(neg, float),
                 pos_head=test_arr[:, 0], pos_tail=test_arr[:, 2])
        msg = (f"AUPRC={au.get('auprc')} (floor={au.get('prevalence_floor')}) "
               f"AUROC={au.get('auroc')} n_pos={au.get('num_pos')} n_neg={au.get('num_neg')}")

    out["early_stopping"] = early
    out["elapsed_sec"] = round(time.time() - t0, 1)
    save_results(out, RESULTS, name)
    print(f"  {msg}", flush=True)
    print(f"  saved -> {RESULTS / (name + '.json')}", flush=True)


if __name__ == "__main__":
    main()
