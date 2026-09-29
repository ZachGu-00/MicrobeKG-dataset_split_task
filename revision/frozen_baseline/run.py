"""Run one corrected benchmark cell. --smoke uses a tiny real-data subgraph."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from core import (LIGHT, MODELS, LearnedScorer, LightScorer, discrimination_metrics,
                  evaluate_ranks, fingerprint, load_case, ranking_metrics, scored_triples)
from prepare import CORE, OPTIONAL, PROTOCOL, ROOT, sha256, write_json

RESTORE_MRR_ATOL = 1e-5


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "benchmark_data")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--setting", choices=CORE + OPTIONAL, required=True)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--num-negs", type=int, default=32)
    ap.add_argument("--loss", choices=["default", "nssa"], default="default")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--frequency", type=int, default=10)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--valid-cap", type=int, default=2048)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--validation-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--profile", action="store_true", help="Two full-graph forward/backward batches; no test evaluation")
    ap.add_argument("--tucker-training", choices=["sampled_query", "legacy"], default="sampled_query")
    ap.add_argument("--rgcn-sampler", choices=["uniform", "schlichtkrull"], default="uniform")
    return ap


def run_name(args):
    return (f"{args.setting}_{args.model.lower()}_split{args.split_seed}_seed{args.seed}"
            f"_d{args.dim}_n{args.num_negs}_{args.loss}")


def compatible_hpo_result(path, previous, code_hash):
    """Reuse individually pinned results across restoration-only fixes."""
    if previous.get("code_sha256") == code_hash:
        return True
    reuse_path = path.parent.parent / "server/restoration_reuse.json"
    if path.parent.name in ["hpo", "main"] and reuse_path.exists():
        receipt = json.loads(reuse_path.read_text())
        entry = receipt.get("files", {}).get(f"{path.parent.name}/{path.name}", {})
        if (receipt.get("target_code_sha256") == code_hash
                and receipt.get("data_manifest_sha256") == previous.get("data_manifest_sha256")
                and entry.get("source_code_sha256") == previous.get("code_sha256")
                and entry.get("json_sha256") == sha256(path)):
            return True
    receipt_path = path.parent.parent / "server/hpo_reuse.json"
    if path.parent.name != "hpo" or previous.get("status") != "validation_complete" or not receipt_path.exists():
        return False
    receipt = json.loads(receipt_path.read_text())
    entry = receipt.get("files", {}).get(path.name, {})
    return (receipt.get("target_code_sha256") == code_hash
            and entry.get("source_code_sha256") == previous.get("code_sha256")
            and entry.get("json_sha256") == sha256(path))


def check_restored_mrr(restored, best):
    """Allow bounded numerical drift after the independent exact-tensor check."""
    delta = abs(restored - best)
    assert np.isfinite(delta) and delta <= RESTORE_MRR_ATOL, (
        "validation MRR drift after exact weight restore", restored, best, delta, RESTORE_MRR_ATOL)
    return delta


def restore_best_state(model, state):
    """Restore exact tensors and clear graph caches without reapplying constraints."""
    import torch
    from pykeen.nn.message_passing import RGCNRepresentation
    model.load_state_dict(state, strict=True)
    for module in model.modules():
        if isinstance(module, RGCNRepresentation):
            module.enriched_embeddings = None
    for name, value in model.state_dict().items():
        assert torch.equal(value.detach().cpu(), state[name].detach().cpu()), ("restored tensor differs", name)


def tucker_query_scores(model, positives, negative_tails):
    """One head/relation projection per query, shared with its sampled tails."""
    import torch
    tails = torch.cat([positives[:, 2, None], negative_tails], dim=1)
    h = model.entity_representations[0](indices=positives[:, 0])
    r = model.relation_representations[0](indices=positives[:, 1])
    t = model.entity_representations[0](indices=tails)
    return model.interaction(h[:, None, :], r[:, None, :], t)


def tucker_epoch(model, optimizer, triples, args, max_batches=None):
    import torch
    model.train()
    order = torch.randperm(len(triples), device=triples.device)
    total = torch.zeros((), device=model.device)
    count, timestamps, start = 0, [], 0
    while start < len(triples):
        stop = min(start + args.batch, len(triples))
        if len(triples) - stop == 1:
            stop += 1  # keep BatchNorm's final batch at least size two
        positives = triples[order[start:stop]]
        offsets = torch.randint(1, model.num_entities, (len(positives), args.num_negs), device=model.device)
        negatives = (positives[:, 2, None] + offsets) % model.num_entities
        optimizer.zero_grad(set_to_none=True)
        scores = tucker_query_scores(model, positives, negatives)
        loss = model.loss.process_slcwa_scores(positive_scores=scores[:, :1].contiguous(),
                                             negative_scores=scores[:, 1:].contiguous(),
                                             label_smoothing=0.0, num_entities=model.num_entities)
        loss = loss + model.collect_regularization_term()
        loss.backward()
        optimizer.step()
        model.post_parameter_update()
        total += loss.detach() * len(positives)
        count += len(positives)
        start = stop
        if max_batches is not None:
            if model.device.type == "cuda":
                torch.cuda.synchronize()
            timestamps.append(time.perf_counter())
            if len(timestamps) >= max_batches:
                break
    return float(total) / count, timestamps


def train(case, args):
    os.environ.setdefault("PYKEEN_HOME", str(ROOT / ".pykeen_home"))
    import torch
    from pykeen.models import model_resolver
    from pykeen.triples import CoreTriplesFactory
    from pykeen.training import SLCWATrainingLoop
    from pykeen.utils import set_random_seed
    torch.set_num_threads(int(os.environ.get('MICROBE_TORCH_THREADS', '4')))
    set_random_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    mapping = np.full(len(case["seen"]), -1, dtype=np.int32)
    mapping[case["seen"]] = np.arange(case["seen"].sum())
    triples = case["train"].copy()
    triples[:, 0] = mapping[triples[:, 0]]; triples[:, 2] = mapping[triples[:, 2]]
    tf = CoreTriplesFactory(torch.as_tensor(triples, dtype=torch.long),
                           num_entities=int(case["seen"].sum()), num_relations=len(case["relations"]))
    kwargs = dict(triples_factory=tf, embedding_dim=args.dim, random_seed=args.seed)
    if args.loss == "nssa":
        kwargs.update(loss="NSSA", loss_kwargs=dict(margin=9.0, adversarial_temperature=1.0))
    model = model_resolver.make(args.model, kwargs).to(device)
    scorer = LearnedScorer(model, mapping)
    optimizer = torch.optim.Adam(model.get_grad_params(), lr=args.lr)
    query_training = args.model == "TuckER" and args.tucker_training == "sampled_query"
    sampler = "schlichtkrull" if args.model == "RGCN" and args.rgcn_sampler == "schlichtkrull" else None
    query_triples = tf.mapped_triples.to(device) if query_training else None
    loop = SLCWATrainingLoop(model=model, triples_factory=tf, optimizer=optimizer,
                             negative_sampler="basic", negative_sampler_kwargs=dict(num_negs_per_pos=args.num_negs),
                             automatic_memory_optimization=not args.profile)
    if args.profile:
        from pykeen.training.callbacks import TrainingCallback
        timestamps = []
        class BatchTimer(TrainingCallback):
            def post_batch(self, **kwargs):
                if device == "cuda":
                    torch.cuda.synchronize()
                timestamps.append(time.perf_counter())
        started = time.perf_counter()
        if query_training:
            _, timestamps = tucker_epoch(model, optimizer, query_triples, args, max_batches=2)
        else:
            loop.train(tf, num_epochs=1, batch_size=args.batch, only_size_probing=True,
                       sampler=sampler, use_tqdm=False, use_tqdm_batch=False, num_workers=0,
                       clear_optimizer=False, drop_last=False, callbacks=BatchTimer())
        if device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        batches = min(2, int(np.ceil(len(triples) / args.batch)))
        return scorer, dict(profile_batches=batches, profile_seconds=seconds,
                            steady_batch_seconds=timestamps[1]-timestamps[0] if len(timestamps)>1 else None,
                            seconds_per_batch_including_setup=seconds / batches,
                            training_triples=len(triples), best_weights_restored=None,
                            hardware=dict(device=device, gpu=torch.cuda.get_device_name(0) if device == "cuda" else None,
                                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
                                peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if device == "cuda" else None)), None
    best, best_epoch, best_state, stale = -np.inf, 0, None, 0
    history = []
    for start in range(0, args.epochs, args.frequency):
        end = min(start + args.frequency, args.epochs)
        # Evaluation may cache graph representations; invalidate before training.
        model.post_parameter_update()
        train_started = time.perf_counter()
        if query_training:
            losses = [tucker_epoch(model, optimizer, query_triples, args)[0] for _ in range(start, end)]
        else:
            losses = loop.train(tf, num_epochs=end, batch_size=args.batch, continue_training=start > 0,
                                sampler=sampler, use_tqdm=False, use_tqdm_batch=False, num_workers=0,
                                clear_optimizer=False, drop_last=False)
        if device == "cuda":
            torch.cuda.synchronize()
        train_seconds = time.perf_counter() - train_started
        validation_started = time.perf_counter()
        val = evaluate_ranks(case, scorer, validation=True, valid_cap=args.valid_cap, batch=args.eval_batch)
        value = ranking_metrics(val, n_boot=0)["mrr"]
        validation_seconds = time.perf_counter() - validation_started
        history.append(dict(epoch=end, train_loss=float(losses[-1]), validation_mrr=value,
                            train_seconds=train_seconds, validation_seconds=validation_seconds))
        peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0
        print(f"epoch={end} validation_mrr={value:.6f} train_sec={train_seconds:.2f} "
              f"valid_sec={validation_seconds:.2f} peak_gpu_gib={peak:.2f}", flush=True)
        if value > best:
            best, best_epoch, stale = value, end, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    assert best_state is not None
    restore_best_state(model, best_state)
    checked = evaluate_ranks(case, scorer, validation=True, valid_cap=args.valid_cap, batch=args.eval_batch)
    restored = ranking_metrics(checked, n_boot=0)["mrr"]
    restore_delta = check_restored_mrr(restored, best)
    if restore_delta:
        print(f"restored_validation_mrr_abs_diff={restore_delta:.9g} accepted_atol={RESTORE_MRR_ATOL:g}", flush=True)
    hardware = dict(device=device, gpu=torch.cuda.get_device_name(0) if device == "cuda" else None,
                    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
                    peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if device == "cuda" else None,
                    python=platform.python_version(), platform=platform.platform(),
                    packages={p: importlib.metadata.version(p) for p in
                              ["torch", "pykeen", "numpy", "pandas", "scipy", "scikit-learn"]})
    return scorer, dict(best_epoch=best_epoch, best_validation_mrr=best,
                        restored_validation_mrr=restored, best_weights_restored=True,
                        restoration_check="exact_tensors_and_validation_mrr_v3",
                        restored_validation_mrr_abs_diff=restore_delta, restored_validation_mrr_atol=RESTORE_MRR_ATOL,
                        epochs_run=history[-1]["epoch"], history=history, hardware=hardware), best_state


def probes(case, scorer, args):
    a, x, meta = case["arrays"], case["triples"], case["meta"]
    blocks, raw = {}, {}

    def save(label, pos, neg, pos_ids, neg_ids):
        blocks[label] = discrimination_metrics(pos, neg, args.seed, args.bootstrap)
        raw.update({label + "_pos_scores": pos, label + "_neg_scores": neg,
                    label + "_pos_ids": pos_ids, label + "_neg_ids": neg_ids})

    if meta["setting"].startswith("t1"):
        def max_direction(rows):
            if not len(rows):
                return np.array([])
            scores = []
            for name in meta["scoring_relations"]:
                q = rows.copy(); q[:, 1] = case["relations"].index(name)
                scores.append(scorer.triples(q))
            return np.max(scores, axis=0)
        pi = a["probe_positive"]
        ni = a["conflict"][a["conflict_eligible"]]
        if len(ni):
            save("direction_conflict_probe", max_direction(x[pi]), max_direction(x[ni]), pi, ni)
        ci = a["random_control_positive"]
        if len(ci):
            save("degree_matched_unknown_control", max_direction(x[ci]), max_direction(a["random_control"]),
                 ci, a["random_control"])
    elif len(a["negatives"]):
        pi = a["test"][a["test_eligible"]]
        ni = a["negatives"][a["negative_eligible"]]
        label = "experimental_nonutilization" if meta["setting"].startswith("t2") else "association_without_recorded_therapy"
        save(label, scorer.triples(scored_triples(case, x[pi])), scorer.triples(scored_triples(case, x[ni])), pi, ni)
    return blocks, raw


def run(args):
    assert not (args.smoke and args.profile), "Smoke and profiling are separate modes"
    assert not args.profile or args.model not in LIGHT, "Profile measures learned model GPU batches"
    if args.smoke:
        args.dim, args.num_negs, args.batch = 16, 2, 64
        args.epochs, args.frequency, args.patience = 2, 1, 2
        args.bootstrap = 20
    case = load_case(args.data, args.setting, args.split_seed, args.smoke)
    if args.profile:
        stage = "profile"
    elif args.smoke:
        stage = "smoke"
    else:
        stage = "hpo" if args.validation_only else "main"
    folder = args.out / stage
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (run_name(args) + ".json")
    raw_path = path.with_suffix(".npz")
    manifest = json.loads((args.data / "manifest.json").read_text())
    code_hash = hashlib.sha256("".join(sha256(Path(__file__).parent / p) for p in ["run.py", "core.py"]).encode()).hexdigest()
    config = {k: v for k, v in vars(args).items() if k not in ["data", "out"]}
    if path.exists() and raw_path.exists():
        previous = json.loads(path.read_text())
        if (previous.get("config") == config and compatible_hpo_result(path, previous, code_hash)
                and previous.get("data_manifest_sha256") == sha256(args.data / "manifest.json")
                and previous.get("raw_sha256") == sha256(raw_path)):
            print("Verified existing cell:", path.name, flush=True)
            return previous
    started = time.perf_counter()
    print(f"{stage} {args.model} {args.setting} train={len(case['train'])} seed={args.seed}", flush=True)
    state = None
    if args.model in LIGHT:
        scorer = LightScorer(case["train"], len(case["seen"]), args.model, args.seed)
        val = evaluate_ranks(case, scorer, validation=True, valid_cap=args.valid_cap, batch=args.eval_batch)
        fit = dict(best_validation_mrr=ranking_metrics(val, n_boot=0)["mrr"], best_weights_restored=None)
    else:
        scorer, fit, state = train(case, args)
    output = dict(protocol=PROTOCOL, status="profile_only" if args.profile else "smoke_only" if args.smoke else "validation_complete" if args.validation_only else "complete",
                  training_protocol="server-scalable-v1",
                  config=config, data_manifest_sha256=sha256(args.data / "manifest.json"),
                  original_edges_sha256=manifest["original_edges_sha256"], code_sha256=code_hash,
                  split_fingerprint=fingerprint(case), counts=case["meta"]["audit"], training=fit)
    raw = {}
    if args.profile:
        pass
    elif args.validation_only:
        raw = evaluate_ranks(case, scorer, validation=True, valid_cap=args.valid_cap, batch=args.eval_batch)
    else:
        # T4-B is a cross-evidence discrimination probe. Its association ranking
        # is used only for validation, not presented as therapeutic ranking.
        if args.setting != "t4_b":
            raw = evaluate_ranks(case, scorer, batch=args.eval_batch)
            output["ranking"] = ranking_metrics(raw, args.seed, args.bootstrap)
            if "taxonomy_hop" in case["arrays"]:
                eligible = case["arrays"]["test_eligible"]
                mask = ((case["arrays"]["taxonomy_hop"] == 0) & (case["arrays"]["ontology_hop"] == 0))[eligible]
                subset = {k: v[mask] if getattr(v, "ndim", 0) else v for k, v in raw.items()}
                output["ranking_no_closure_2hop"] = ranking_metrics(subset, args.seed, args.bootstrap)
                raw["no_closure_2hop_mask"] = mask
        disc, scores = probes(case, scorer, args)
        output["discrimination"] = disc
        raw.update(scores)
    if args.smoke:
        output["smoke_counts"] = dict(train=len(case["train"]), valid=len(case["arrays"]["valid"]), test=len(case["arrays"]["test"]))
    np.savez_compressed(raw_path, **raw)
    output["raw_sha256"] = sha256(raw_path)
    output["elapsed_sec"] = round(time.perf_counter() - started, 3)
    if state is not None and not args.validation_only and not args.smoke:
        import torch
        torch.save(dict(state_dict=state, config=config, global_entity_indices=np.flatnonzero(case["seen"]).tolist(),
                        relations=case["relations"], data_manifest_sha256=output["data_manifest_sha256"]), path.with_suffix(".pt"))
    write_json(path, output)
    print("Saved", path, flush=True)
    return output


if __name__ == "__main__":
    run(parser().parse_args())
