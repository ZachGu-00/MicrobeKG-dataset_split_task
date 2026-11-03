"""NBFNet for v3 (P0-4).  Same model and target-edge-removed training as gnn/nbfnet_run.py, with three changes:
  * triples() groups rows by (disease, relation) and propagates once from the disease with the inverse relation
    (one propagation per disease instead of one per four rows), which removes the probe bottleneck that made the
    v2 runs time out;
  * a fixed step budget per epoch, validation every epoch, early stopping with patience 1;
  * weights are saved next to the result (<stem>.pt) and build_scorer() rebuilds the scorer for rescoring.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/mkg/gnn")
import common  # noqa: E402
from common import evaluate_and_save, load  # noqa: E402
import nbfnet_run as nbf  # noqa: E402

V3 = ["t1_v3_strict_nocohort", "t1_v3_strict_nooverlap", "t1_v3_strict_nocooc", "t1_v3_temporal_nocooc",
      "t1_v3_temporal_dated", "t1_v3_p11_base", "t1_v3_p11_same", "t1_v3_p11_other"]
common.SETTINGS.extend(s for s in V3 if s not in common.SETTINGS)


class FastScorer(nbf.Scorer):
    def triples(self, rows):
        rows = np.asarray(rows)
        out = np.empty(len(rows), np.float32)
        groups = defaultdict(list)
        for i, (h, r, t) in enumerate(rows):
            groups[(int(t), int(r))].append(i)
        keys = list(groups)
        dev = self.device
        with torch.inference_mode():
            for st in range(0, len(keys), self.batch):
                ks = keys[st:st + self.batch]
                width = max(len(groups[k]) for k in ks)
                src = self.mapping[np.array([k[0] for k in ks])]
                rel = np.array([k[1] for k in ks]) + self.model.r
                tg = np.zeros((len(ks), width), np.int64)
                for j, k in enumerate(ks):
                    hh = self.mapping[rows[groups[k], 0]]
                    tg[j, :] = hh[0]
                    tg[j, :len(hh)] = hh
                assert (src >= 0).all() and (tg >= 0).all()
                s = self.model.score(torch.as_tensor(src, device=dev), torch.as_tensor(rel, device=dev),
                                     torch.as_tensor(tg, device=dev), *self.edges).detach().cpu().numpy()
                for j, k in enumerate(ks):
                    out[groups[k]] = s[j, :len(groups[k])]
        return out

    def candidates(self, queries, relations, pool, side):
        with torch.inference_mode():
            return super().candidates(queries, relations, pool, side)


def graph(case, device):
    mapping = np.full(len(case["seen"]), -1, np.int64)
    mapping[case["seen"]] = np.arange(case["seen"].sum())
    x = case["train"].astype(np.int64).copy()
    x[:, 0] = mapping[x[:, 0]]
    x[:, 2] = mapping[x[:, 2]]
    R = len(case["relations"])
    t = lambda v: torch.as_tensor(v, device=device, dtype=torch.long)
    edges = (t(np.r_[x[:, 0], x[:, 2]]), t(np.r_[x[:, 1], x[:, 1] + R]), t(np.r_[x[:, 2], x[:, 0]]))
    return mapping, x, int(case["seen"].sum()), R, edges


def build_scorer(case, pt_path, device="cuda", batch=8):
    cfg = json.loads(Path(pt_path).with_suffix(".json").read_text())["config"]
    device = torch.device(device)
    mapping, _, n, R, edges = graph(case, device)
    model = nbf.NBFNet(n, R, cfg["dim"], cfg["layers"]).to(device)
    model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    model.eval()
    return FastScorer(model, edges, device, batch, mapping)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None)
    p.add_argument("--out", default="/root/autodl-tmp/mkg/results3/gnn")
    p.add_argument("--setting", choices=common.SETTINGS, required=True)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--steps-per-epoch", type=int, default=2000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--num-negs", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--patience", type=int, default=1)
    p.add_argument("--valid-cap", type=int, default=1024)
    p.add_argument("--eval-batch", type=int, default=8)
    p.add_argument("--adversarial-temperature", type=float, default=1.0)
    return p


def main(a):
    started = time.perf_counter()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    case = load(a.data, a.setting, a.split_seed)
    device = torch.device("cuda" if a.device == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    mapping, x, n, R, edges = graph(case, device)
    model = nbf.NBFNet(n, R, a.dim, a.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    types_seen = case["entities"].node_type.to_numpy()[case["seen"]]
    pools = {t: np.flatnonzero(types_seen == t) for t in np.unique(types_seen)}
    rng = np.random.default_rng(a.seed)
    best, best_state, stale, hist = -1.0, None, 0, []
    from core import evaluate_ranks, ranking_metrics
    for ep in range(1, a.epochs + 1):
        model.train()
        order = rng.permutation(len(x))
        losses, ts = [], time.perf_counter()
        for step in range(a.steps_per_epoch):
            z = x[order[(step * a.batch) % len(x):(step * a.batch) % len(x) + a.batch]]
            if not len(z):
                continue
            B = len(z)
            neg = np.stack([rng.choice(pools.get(types_seen[row[2]], np.arange(n)), a.num_negs) for row in z])
            rem = np.unique(np.r_[z[:, 0] * ((2 * R) * n) + z[:, 1] * n + z[:, 2],
                                  z[:, 2] * ((2 * R) * n) + (z[:, 1] + R) * n + z[:, 0]]).astype(np.int64)
            opt.zero_grad(set_to_none=True)
            score = model.score(torch.as_tensor(z[:, 0], device=device), torch.as_tensor(z[:, 1], device=device),
                                torch.as_tensor(np.c_[z[:, 2], neg], device=device), *edges,
                                remove_keys=torch.as_tensor(rem, device=device))
            pos_loss = torch.nn.functional.softplus(-score[:, 0]).mean()
            w = torch.softmax(score[:, 1:] * a.adversarial_temperature, dim=-1).detach()
            loss = pos_loss + (w * torch.nn.functional.softplus(score[:, 1:])).sum(-1).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        vs = time.perf_counter()
        sc = FastScorer(model, edges, device, a.eval_batch, mapping)
        vm = ranking_metrics(evaluate_ranks(case, sc, validation=True, valid_cap=a.valid_cap, batch=a.eval_batch),
                             n_boot=0)["mrr"]
        hist.append(dict(epoch=ep, train_loss=float(np.mean(losses)), validation_mrr=vm,
                         train_seconds=vs - ts, validation_seconds=time.perf_counter() - vs))
        print(f"epoch={ep} validation_mrr={vm:.6f} train_sec={vs - ts:.0f}", flush=True)
        if vm > best:
            best, stale = vm, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= a.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    hw = dict(device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
              peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
              python=platform.python_version(), torch=torch.__version__)
    fit = dict(best_epoch=int(np.argmax([h["validation_mrr"] for h in hist]) + 1), best_validation_mrr=best,
               epochs_run=len(hist), history=hist, target_edge_removal=True,
               runner_elapsed_sec=time.perf_counter() - started)
    cfg = vars(a).copy()
    cfg.update(model="NBFNet", loss="advbce_target_edge_removed")
    sc = FastScorer(model, edges, device, a.eval_batch, mapping)
    evaluate_and_save(case, sc, "NBFNet", a.dim, a.num_negs, "advbce", a.out, cfg, fit, hw, 1000, a.eval_batch, False)
    stem = f"{a.setting}_nbfnet_split{a.split_seed}_seed{a.seed}_d{a.dim}_n{a.num_negs}_advbce"
    torch.save(best_state, Path(a.out) / f"{stem}.pt")
    print("saved", stem, round(time.perf_counter() - started), "s", flush=True)


if __name__ == "__main__":
    main(parser().parse_args())
