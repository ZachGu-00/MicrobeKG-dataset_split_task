"""Task 4 (metabolite->disease therapeutic prediction) — one model, one setting.

train graph = final_kg_edges - gtrain_removed.tsv (co-located met-dis edges of held
pairs removed; metabolite->gene->disease mechanism bridge KEPT). Settings: A
(transductive treats) | B (cross-evidence zero-shot: treats scored by surrogate
associated_with_disease) | C (cold-metabolite). Models: KGE / structural / trivial.

Ranking (treats; reported when treats is in train = A, C): full ranking tail+head,
micro/macro, MRR CI. Hard-neg AUROC (treats vs associated-not-treating, seeded
sample): AUPRC(primary)+floor+fold-enrichment + AUROC(secondary)+CIs, TRUE
imbalance reported. Raw dump -> chemical-class strata / calibration offline.
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
    relation_pools, build_known, kge_rank_instances, kge_pair_scores,
    light_rank_instances, light_pair_scores, make_light_scorer, LIGHT, HEURISTICS,
    stopper_summary, save_results, add_common_args, load_valid,
)
from metrics import ranking_block, auc_block, save_raw

TASK4 = SPLITS / "task4_metabolite_disease"
RESULTS = Path(__file__).parent / "results"
SETTING_DIRS = {"A": "setting_a_transductive", "B": "setting_b_cross_evidence",
                "C": "setting_c_cold_metabolite"}
HARDNEG_POOL = TASK4 / "hardneg_pool_associated_not_treating.tsv"
TARGET = {"treats_disease"}


def sample_hardneg_pairs(pos_pairs, neg_ratio, seed):
    """Seeded sample of the associated-not-treating pool, ratio:pos."""
    hn = pd.read_csv(HARDNEG_POOL, sep="\t", usecols=["head_id", "tail_id"], dtype=str)
    hn_pairs = list(zip(hn.head_id, hn.tail_id))
    rng = np.random.RandomState(seed)
    k = min(len(hn_pairs), neg_ratio * len(pos_pairs))
    sel = rng.choice(len(hn_pairs), size=k, replace=False)
    return [hn_pairs[i] for i in sel]


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--setting", required=True, choices=list(SETTING_DIRS))
    ap.add_argument("--neg_ratio", type=int, default=10, help="sampled hard-neg : pos ratio")
    args = ap.parse_args()
    sdir = TASK4 / SETTING_DIRS[args.setting]
    assert sdir.is_dir(), f"missing {sdir}"

    train_arr = build_training_array(sdir, removed_name="gtrain_removed.tsv")
    valid_arr = load_valid(sdir, train_arr, args.seed)
    test_arr = load_hrt(sdir / "test.tsv")
    treats_in_train = "treats_disease" in set(train_arr[:, 1])
    is_light = args.model in LIGHT
    name = f"{args.model.lower()}_{args.setting}_seed{args.seed}" + (f"_{args.tag}" if args.tag else "")
    print(f"[T4/{args.model}/{args.setting} seed={args.seed}] {'light' if is_light else 'kge'} "
          f"train={len(train_arr):,} test(treats)={len(test_arr):,} "
          f"treats_in_train={treats_in_train} cuda={torch.cuda.is_available()}", flush=True)

    out = {"task": "task4", "model": args.model, "setting": args.setting, "seed": args.seed,
           "model_type": ("structural" if args.model in HEURISTICS else "trivial") if is_light else "kge",
           "num_train": len(train_arr)}

    t0 = time.time()
    if is_light:
        scorer = make_light_scorer(args.model, train_arr, args.seed)
        model_obj, tf, early = None, None, None
    else:
        train, valid = make_factories(train_arr, valid_arr, args.model in ("CompGCN", "ConvE"))
        result = run_training(args.model, train, valid, valid, args)
        model_obj, tf, early = result.model, train, stopper_summary(result)

    raw = {}
    if treats_in_train:
        head_pool, tail_pool = relation_pools(train_arr, test_arr, TARGET)
        kt, kh = build_known([train_arr, valid_arr, test_arr], TARGET)
        if is_light:
            ranks = light_rank_instances(scorer, test_arr, head_pool, tail_pool, kt, kh)
        else:
            ranks = kge_rank_instances(model_obj, tf, test_arr, head_pool, tail_pool, kt, kh)
        out["ranking"] = ranking_block(ranks["tail_rank"], ranks["tail_group"],
                                       ranks["head_rank"], ranks["head_group"], seed=args.seed)
        out["num_test_eval"], out["num_test_oov"] = ranks["n_eval"], ranks["n_oov"]
        raw.update(tail_rank=ranks["tail_rank"], tail_group=ranks["tail_group"],
                   head_rank=ranks["head_rank"], head_group=ranks["head_group"])

    pos_pairs = list(zip(test_arr[:, 0], test_arr[:, 2]))
    neg_pairs = sample_hardneg_pairs(pos_pairs, args.neg_ratio, args.seed)
    if is_light:
        pos = light_pair_scores(scorer, pos_pairs)
        neg = light_pair_scores(scorer, neg_pairs)
        scoring = "structural"
    else:
        scoring = "treats_disease" if treats_in_train else "associated_with_disease"
        pos = kge_pair_scores(model_obj, tf, pos_pairs, scoring)
        neg = kge_pair_scores(model_obj, tf, neg_pairs, scoring)
    au = auc_block(pos, neg, seed=args.seed)
    au["scoring_relation"], au["neg_ratio"] = scoring, args.neg_ratio
    out["hardneg_auroc"] = au
    raw.update(hardneg_pos=np.asarray(pos, float), hardneg_neg=np.asarray(neg, float))
    if args.setting == "C":
        out["chemical_class_stratification"] = "TODO (needs ChEBI/ClassyFire, see split dir)"

    save_raw(RESULTS, name, **raw)
    out["early_stopping"] = early
    out["elapsed_sec"] = round(time.time() - t0, 1)
    save_results(out, RESULTS, name)
    rmsg = ""
    if "ranking" in out:
        bm = out["ranking"]["both"]
        rmsg = (f"MRR(both,micro)={bm['micro']['mrr']:.4f} macro={bm['macro']['mrr']:.4f} "
                f"H@10={bm['micro']['hits_at_10']:.4f} | ")
    print(f"  {rmsg}hardneg AUPRC={au.get('auprc')} (floor={au.get('prevalence_floor')}) "
          f"AUROC={au.get('auroc')} n_pos={au.get('num_pos')} n_neg={au.get('num_neg')} "
          f"scoring={scoring}", flush=True)
    print(f"  saved -> {RESULTS / (name + '.json')}", flush=True)


if __name__ == "__main__":
    main()
