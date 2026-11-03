#!/usr/bin/env python3
"""NCN (Neural Common Neighbor, Wang, Yang & Zhang, ICLR 2024) on MicrobeKG-Bench.

Official code (GraphPKU/NeuralCommonNeighbor, commit 11d5970) is imported unchanged from
/root/autodl-tmp/mkg/vendor/NCN: `GCN` encoder and `CNLinkPredictor` (predictor name `cn1`, i.e. NCN).

Adaptation
* Message-passing graph: all training triples of the case (all relations), as ONE undirected homogeneous
  graph over the compact seen-entity id space (relation types and directions are dropped, duplicates merged).
* Node input: learnable embeddings (official `max_x` path, orthogonal init). No node features.
* Target edges: unique (microbe, disease) pairs of the scoring relations (enriched_in, depleted_in) in
  case['train']. NCN is relation-agnostic: both scoring relations get the same score.
* Negatives: per positive (m, d), k pairs (m, random disease) and k pairs (random microbe, d).
  Sampled pairs that are training positives are masked out of the loss.
* Target link removal (official `--maskinput` behaviour): both directions of the batch's target edges are
  removed from the message-passing graph (GCN and common-neighbour computation) for each training batch.
* Ranking: candidates() computes the NCN score of every (query, candidate) pair at once with a sparse
  product, A[pool] @ (1[N(q)] * x), which equals sum_{k in N(q) & N(c)} x_k for all candidates c.
"""
from __future__ import annotations
import argparse, importlib.util, json, platform, sys, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, '/root/autodl-tmp/mkg/gnn')
VENDOR = Path('/root/autodl-tmp/mkg/vendor/NCN')


def _vendor():
    for name in ('utils', 'model'):
        spec = importlib.util.spec_from_file_location(name, VENDOR / f'{name}.py')
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules['model']


vendor_model = _vendor()
from torch_sparse import SparseTensor  # noqa: E402
from common import load, evaluate_and_save, evaluate_ranks, ranking_metrics, SETTINGS, SEEDS  # noqa: E402

# Fixed (not tuned) architecture/regularisation choices; hidden size and lr are the tuned axes.
FIXED = dict(conv='gcn', mplayers=2, nnlayers=3, ln=True, lnnn=True, res=True, jk=False,
             use_xlin=False, tailact=True, twolayerlin=False, xdp=0.0, tdp=0.0,
             gnndp=0.2, gnnedp=0.1, predp=0.2, preedp=0.1, predictor='cn1', beta=1.0)


class Graph:
    """Compact undirected homogeneous message-passing graph + target-pair list from a case."""

    def __init__(self, case, device):
        seen = case['seen']
        self.n = n = int(seen.sum()); self.device = device
        self.mapping = np.full(len(seen), -1, np.int64); self.mapping[seen] = np.arange(n)
        x = case['train'].astype(np.int64)
        u, v = self.mapping[x[:, 0]], self.mapping[x[:, 2]]
        assert (u >= 0).all() and (v >= 0).all()
        ne = u != v
        a, b = np.minimum(u, v)[ne], np.maximum(u, v)[ne]
        und = np.unique(a * n + b); a, b = und // n, und % n
        row, col = np.r_[a, b], np.r_[b, a]
        key = row * n + col; o = np.argsort(key, kind='stable'); row, col, key = row[o], col[o], key[o]
        self.n_undirected = len(und)
        self.row = torch.as_tensor(row, device=device); self.col = torch.as_tensor(col, device=device)
        self.ekey = torch.as_tensor(key, device=device)
        self.csr = sp.csr_matrix((np.ones(len(row), np.float32), (row, col)), shape=(n, n))
        scoring = [case['relations'].index(r) for r in case['meta']['scoring_relations']]
        tm = np.isin(x[:, 1], scoring)
        pk = np.unique(u[tm] * n + v[tm]); self.pos_m, self.pos_d = pk // n, pk % n
        types = case['entities'].node_type.to_numpy()[seen]
        self.micro = np.flatnonzero(types == case['meta']['head_type'])
        self.dis = np.flatnonzero(types == case['meta']['tail_type'])
        assert types[self.pos_m].tolist().count(case['meta']['head_type']) == len(self.pos_m)
        assert types[self.pos_d].tolist().count(case['meta']['tail_type']) == len(self.pos_d)
        self.pos_keys = torch.as_tensor(pk, device=device)
        self.xid = torch.arange(n, device=device)
        self._cache = {}

    def adj(self, remove_keys=None):
        if remove_keys is None:
            return SparseTensor(row=self.row, col=self.col, sparse_sizes=(self.n, self.n), is_sorted=True, trust_data=True)
        keep = ~torch.isin(self.ekey, remove_keys)
        return SparseTensor(row=self.row[keep], col=self.col[keep], sparse_sizes=(self.n, self.n), is_sorted=True, trust_data=True)

    def pool_csr(self, p):
        k = (len(p), hash(p.tobytes()))
        if k not in self._cache:
            sub = self.csr[p].tocsr()
            self._cache = {k: torch.sparse_csr_tensor(torch.as_tensor(sub.indptr, dtype=torch.int64), torch.as_tensor(sub.indices, dtype=torch.int64),
                                                      torch.as_tensor(sub.data), size=(len(p), self.n)).to(self.device)}
        return self._cache[k]

    def dense_rows(self, q):
        sub = self.csr[q].tocsr(); out = torch.zeros(len(q), self.n, device=self.device)
        rows = np.repeat(np.arange(len(q)), np.diff(sub.indptr))
        out[torch.as_tensor(rows, device=self.device), torch.as_tensor(sub.indices, dtype=torch.long, device=self.device)] = 1.0
        return out


def build_modules(n, hid, cfg, device):
    GCN, pred = vendor_model.GCN, vendor_model.predictor_dict[cfg['predictor']]
    model = GCN(hid, hid, hid, cfg['mplayers'], cfg['gnndp'], cfg['ln'], cfg['res'], n - 1, cfg['conv'], cfg['jk'], cfg['gnnedp'],
                xdropout=cfg['xdp'], taildropout=cfg['tdp']).to(device)
    predictor = pred(hid, hid, 1, cfg['nnlayers'], cfg['predp'], cfg['preedp'], cfg['lnnn'], cndeg=-1,
                     use_xlin=cfg['use_xlin'], tailact=cfg['tailact'], twolayerlin=cfg['twolayerlin'], beta=cfg['beta']).to(device)
    return model, predictor


class NCNScorer:
    """Benchmark scorer. NCN is relation-agnostic: `relations` is ignored."""

    def __init__(self, graph, model, predictor, device, qchunk=16, pair_batch=16384):
        self.g, self.model, self.predictor, self.device = graph, model, predictor, device
        self.qchunk, self.pair_batch = qchunk, pair_batch
        self.adj = graph.adj(); self.refresh()

    @torch.no_grad()
    def refresh(self):
        self.model.eval(); self.predictor.eval()
        self.h = self.model(self.g.xid, self.adj)
        self.xn = self.h + self.predictor.xlin(self.h)   # node features that are summed over common neighbours

    @torch.no_grad()
    def candidates(self, queries, relations, pool, side):
        q = self.g.mapping[np.asarray(queries)]; p = self.g.mapping[np.asarray(pool)]
        assert (q >= 0).all() and (p >= 0).all()
        pr, D, N = self.predictor, self.h.shape[1], len(p)
        apool = self.g.pool_csr(p); hp = self.h[torch.as_tensor(p, device=self.device)]
        out = []
        for st in range(0, len(q), self.qchunk):
            qs = q[st:st + self.qchunk]; B = len(qs)
            aq = self.g.dense_rows(qs)                                    # [B, n] 0/1 neighbour indicators
            z = (aq.T[:, :, None] * self.xn[:, None, :]).reshape(self.g.n, B * D)
            cn = torch.sparse.mm(apool, z).reshape(N * B, D)              # sum of x over common neighbours, [N, B, D]
            hq = self.h[torch.as_tensor(qs, device=self.device)]
            xij = pr.xijlin((hp[:, None, :] * hq[None, :, :]).reshape(N * B, D))
            s = pr.lin(pr.xcnlin(cn) * pr.beta + xij).reshape(N, B).T
            out.append(s.float().cpu().numpy())
        return np.concatenate(out, 0) if out else np.empty((0, N), np.float32)

    @torch.no_grad()
    def triples(self, rows):
        """Pair scores through the official predictor forward, batched over pairs (not one pass per row)."""
        rows = np.asarray(rows); m = self.g.mapping[rows[:, 0]]; d = self.g.mapping[rows[:, 2]]
        assert (m >= 0).all() and (d >= 0).all()
        out = np.empty(len(rows), np.float32)
        for st in range(0, len(rows), self.pair_batch):
            tar = torch.as_tensor(np.stack([m[st:st + self.pair_batch], d[st:st + self.pair_batch]]), device=self.device)
            out[st:st + self.pair_batch] = self.predictor(self.h, self.adj, tar).squeeze(-1).float().cpu().numpy()
        return out


def build_scorer(case, pt_path, device='cuda', qchunk=16):
    """Rebuild a trained NCN scorer from a saved `<stem>.pt` for rescoring."""
    device = torch.device(device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    ck = torch.load(pt_path, map_location='cpu', weights_only=False)
    g = Graph(case, device); assert g.n == ck['n'], (g.n, ck['n'])
    model, predictor = build_modules(g.n, ck['hid'], ck['fixed'], device)
    model.load_state_dict(ck['model']); predictor.load_state_dict(ck['predictor'])
    return NCNScorer(g, model, predictor, device, qchunk)


def selftest(sc, g, k=2000, seed=0):
    """Vectorised candidates() must equal the official pair forward."""
    rng = np.random.default_rng(seed); d = sc.h.device
    res = {}
    for side, qn, pn in (('tail', g.micro, g.dis), ('head', g.dis, g.micro)):
        qi = rng.choice(qn, 6, replace=False); pi = pn[:700]
        inv = np.flatnonzero(g.mapping >= 0)  # compact -> global is not needed: identity mapping below
        ident = np.arange(g.n)
        old = g.mapping; g.mapping = ident
        try:
            C = sc.candidates(qi, np.zeros(len(qi), int), pi, side)
            rows = np.array([[a, 0, b] if side == 'tail' else [b, 0, a] for a in qi for b in pi])
            T = sc.triples(rows).reshape(len(qi), len(pi))
        finally:
            g.mapping = old
        res[side] = float(np.abs(C - T).max()); res[side + '_scale'] = float(np.abs(T).max())
    return res


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=None); p.add_argument('--out', default='/root/autodl-tmp/mkg/results2/gnn/main')
    p.add_argument('--setting', choices=SETTINGS, required=True)
    p.add_argument('--split-seed', type=int, default=42); p.add_argument('--seed', type=int, default=None)
    p.add_argument('--device', default='cuda')
    p.add_argument('--hid', type=int, default=64); p.add_argument('--lr', type=float, default=5e-3)
    p.add_argument('--batch', type=int, default=2048); p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--negs-per-side', type=int, default=1)
    p.add_argument('--eval-every', type=int, default=2); p.add_argument('--patience', type=int, default=6)
    p.add_argument('--valid-cap', type=int, default=1024); p.add_argument('--eval-batch', type=int, default=16)
    p.add_argument('--qchunk', type=int, default=16)
    p.add_argument('--max-train-minutes', type=float, default=40.0)
    p.add_argument('--mem-cap-gib', type=float, default=7.5)
    p.add_argument('--validation-only', action='store_true'); p.add_argument('--tune-out', default='/root/autodl-tmp/mkg/baselines_v3/tune')
    p.add_argument('--selftest', action='store_true'); p.add_argument('--smoke', action='store_true')
    return p


def main(a):
    t_run = time.perf_counter()
    if a.seed is None: a.seed = a.split_seed
    if a.smoke: a.hid, a.epochs, a.eval_every, a.patience, a.eval_batch, a.valid_cap = 16, 2, 1, 1, 2, 64
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    device = torch.device('cuda' if a.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(0.95, a.mem_cap_gib * 2**30 / total)); torch.cuda.reset_peak_memory_stats()
    case = load(a.data, a.setting, a.split_seed, a.smoke)
    g = Graph(case, device)
    print(f'setting={a.setting} split={a.split_seed} n={g.n} undirected_edges={g.n_undirected} target_pairs={len(g.pos_m)} '
          f'microbes={len(g.micro)} diseases={len(g.dis)}', flush=True)
    model, predictor = build_modules(g.n, a.hid, FIXED, device)
    opt = torch.optim.Adam([{'params': model.parameters(), 'lr': a.lr}, {'params': predictor.parameters(), 'lr': a.lr}])
    sc = NCNScorer(g, model, predictor, device, a.qchunk)
    if a.selftest:
        print('selftest max|candidates - official pair forward|:', selftest(sc, g), flush=True)
    pos_m = torch.as_tensor(g.pos_m, device=device); pos_d = torch.as_tensor(g.pos_d, device=device)
    micro = torch.as_tensor(g.micro, device=device); dis = torch.as_tensor(g.dis, device=device)
    n, P, k = g.n, len(g.pos_m), a.negs_per_side
    best, best_state, best_ep, stale, hist = -1.0, None, 0, 0, []
    t_train = time.perf_counter(); stopped = 'epochs'
    for ep in range(1, a.epochs + 1):
        model.train(); predictor.train(); t0 = time.perf_counter(); losses = []
        perm = torch.randperm(P, device=device)
        for st in range(0, P, a.batch):
            idx = perm[st:st + a.batch]; pm, pd = pos_m[idx], pos_d[idx]; B = len(idx)
            nd = dis[torch.randint(len(dis), (B, k), device=device)]               # (m, random disease)
            nm = micro[torch.randint(len(micro), (B, k), device=device)]           # (random microbe, d)
            m_all = torch.cat([pm, pm.repeat_interleave(k), nm.reshape(-1)])
            d_all = torch.cat([pd, nd.reshape(-1), pd.repeat_interleave(k)])
            rem = torch.cat([pm * n + pd, pd * n + pm])                           # target link removal, both directions
            adj = g.adj(rem)
            h = model(g.xid, adj)
            out = predictor.multidomainforward(h, adj, torch.stack([m_all, d_all]), cndropprobs=[]).squeeze(-1)
            pos, neg = out[:B], out[B:]
            neg_key = m_all[B:] * n + d_all[B:]
            ok = (~torch.isin(neg_key, g.pos_keys)).float()                        # mask sampled true positives
            loss = torch.nn.functional.softplus(-pos).mean() + (torch.nn.functional.softplus(neg) * ok).sum() / ok.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        row = {'epoch': ep, 'train_loss': float(np.mean(losses)), 'train_seconds': time.perf_counter() - t0}
        if ep % a.eval_every == 0 or ep == a.epochs:
            tv = time.perf_counter(); sc.refresh()
            vr = evaluate_ranks(case, sc, validation=True, valid_cap=a.valid_cap, batch=a.eval_batch)
            vm = ranking_metrics(vr, n_boot=0)['mrr']; row.update(validation_mrr=vm, validation_seconds=time.perf_counter() - tv)
            peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0.0
            print(f'epoch={ep} loss={row["train_loss"]:.4f} validation_mrr={vm:.6f} train_sec={row["train_seconds"]:.2f} '
                  f'val_sec={row["validation_seconds"]:.1f} peak_alloc_gib={peak:.2f}', flush=True)
            if vm > best:
                best, best_ep, stale = vm, ep, 0
                best_state = ({k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()},
                              {k_: v.detach().cpu().clone() for k_, v in predictor.state_dict().items()})
            else:
                stale += 1
            if stale >= a.patience: stopped = 'patience'; hist.append(row); break
        hist.append(row)
        if (time.perf_counter() - t_train) / 60 > a.max_train_minutes: stopped = 'time_cap'; print('train time cap reached', flush=True); break
    train_seconds = time.perf_counter() - t_train
    if best_state is not None:
        model.load_state_dict(best_state[0]); predictor.load_state_dict(best_state[1])
    sc.refresh()
    peak_alloc = torch.cuda.max_memory_allocated() if device.type == 'cuda' else None
    hardware = {'device': str(device), 'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                'peak_gpu_allocated_bytes': peak_alloc, 'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved() if device.type == 'cuda' else None,
                'python': platform.python_version(), 'torch': torch.__version__}
    fit = {'best_epoch': best_ep, 'best_validation_mrr': best, 'epochs_run': len(hist), 'stopped_by': stopped, 'train_seconds': train_seconds,
           'history': hist, 'hardware': hardware, 'target_edge_removal': True, 'relation_agnostic': True, 'fixed': FIXED,
           'message_passing_undirected_edges': g.n_undirected, 'target_pairs': int(P), 'ncn_commit': '11d5970'}
    cfg = vars(a).copy(); cfg['model'] = 'NCN'; cfg['loss'] = 'bce_target_edge_removed'; cfg['seed'] = a.seed; cfg.update({f'fixed_{k_}': v for k_, v in FIXED.items()})
    if a.validation_only:
        out = Path(a.tune_out); out.mkdir(parents=True, exist_ok=True)
        fit['runner_elapsed_sec'] = time.perf_counter() - t_run; fit['config'] = cfg
        name = f'{a.setting}_ncn_tune_split{a.split_seed}_seed{a.seed}_h{a.hid}_lr{a.lr:g}.json'
        (out / name).write_text(json.dumps(fit, ensure_ascii=False, indent=2) + '\n'); print('Saved validation-only', out / name, flush=True)
        return fit
    stem = f'{a.setting}_ncn_split{a.split_seed}_seed{a.seed}_d{a.hid}_n{2 * k}_bce'
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'predictor': predictor.state_dict(), 'n': g.n, 'hid': a.hid, 'fixed': FIXED,
                'setting': a.setting, 'split_seed': a.split_seed, 'seed': a.seed}, outdir / (stem + '.pt'))
    fit['runner_elapsed_sec'] = time.perf_counter() - t_run
    res = evaluate_and_save(case, sc, 'NCN', a.hid, 2 * k, 'bce', outdir, cfg, fit, hardware, 20 if a.smoke else 1000, a.eval_batch, a.smoke)
    wall = time.perf_counter() - t_run
    print(f'CELL_DONE {stem} test_mrr={res["ranking"]["mrr"]:.6f} wall_sec={wall:.1f} train_sec={train_seconds:.1f} '
          f'peak_alloc_gib={(peak_alloc or 0) / 2**30:.2f}', flush=True)
    return res


if __name__ == '__main__':
    main(parser().parse_args())
