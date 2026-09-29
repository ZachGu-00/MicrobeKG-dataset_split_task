"""Official frozen ULTRA weights with the benchmark's typed, filtered evaluator."""
from pathlib import Path
import sys
import types

import numpy as np
import torch

from prepare import ROOT


def reference_rspmm(edge_index, edge_type, edge_weight, relation, input, sum='add', mul='mul'):
    """Tiny-test reference for ULTRA's sum/DistMult kernel, with bounded edge chunks."""
    assert sum == 'add' and mul == 'mul'
    output = torch.zeros_like(input)
    # Match rspmm.cpp exactly: edge_index[0] is output row, [1] is input column.
    for start in range(0, edge_type.numel(), 16384):
        target, source = edge_index[:, start:start+16384]
        message = input[source] * relation[edge_type[start:start+16384]]
        output.index_add_(0, target, message * edge_weight[start:start+16384, None])
    return output


def import_ultra(reference=False):
    sys.path.insert(0, str(ROOT / 'vendor/ULTRA'))
    if reference:
        # Local CPU smoke only; production uses official compiled dependencies.
        import torch_geometric
        from torch_geometric.utils import scatter
        try:
            import torch_scatter
        except ImportError:
            module = types.ModuleType('torch_scatter')
            module.scatter = scatter
            module.scatter_add = lambda src, index, dim=0, dim_size=None: scatter(src, index, dim=dim, dim_size=dim_size, reduce='sum')
            sys.modules['torch_scatter'] = module
        module = types.ModuleType('ultra.rspmm')
        module.generalized_rspmm = reference_rspmm
        sys.modules['ultra.rspmm'] = module
    from ultra.models import Ultra
    from ultra.tasks import build_relation_graph
    return Ultra, build_relation_graph


class UltraScorer:
    def __init__(self, case, device='cuda', reference=False, batch=8):
        from torch_geometric.data import Data
        Ultra, build_relation_graph = import_ultra(reference)
        self.device = torch.device(device); self.batch = batch
        self.mapping = np.full(len(case['seen']), -1, np.int64)
        self.mapping[case['seen']] = np.arange(case['seen'].sum())
        x = case['train'].copy()
        x[:, 0] = self.mapping[x[:, 0]]; x[:, 2] = self.mapping[x[:, 2]]
        self.num_relations = len(case['relations'])
        # ULTRA triples use h,t,r; benchmark triples use h,r,t.
        edge_index = torch.as_tensor(np.r_[x[:, [0, 2]], x[:, [2, 0]]].T.copy(), dtype=torch.long)
        edge_type = torch.as_tensor(np.r_[x[:, 1], x[:, 1] + self.num_relations], dtype=torch.long)
        graph = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=int(case['seen'].sum()), num_relations=2*self.num_relations)
        self.graph = build_relation_graph(graph).to(self.device)
        shared = dict(input_dim=64, hidden_dims=[64]*6, message_func='distmult', aggregate_func='sum', short_cut=True, layer_norm=True)
        self.model = Ultra(dict(shared, **{'class': 'RelNBFNet'}), dict(shared, **{'class': 'EntityNBFNet'})).to(self.device)
        self.checkpoint = ROOT / 'vendor/ULTRA/ckpts/ultra_50g.pth'
        payload = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict(payload['model'], strict=True)
        self.model.eval()

    def candidates(self, queries, relations, pool, side):
        q, p = self.mapping[queries], self.mapping[pool]
        assert (q >= 0).all() and (p >= 0).all()
        relations = relations + (self.num_relations if side == 'head' else 0)
        result = []
        with torch.inference_mode():
            for start in range(0, len(q), self.batch):
                query = q[start:start+self.batch]; rel = relations[start:start+self.batch]
                rows = np.empty((len(query), len(p), 3), np.int64)
                rows[:, :, 0] = query[:, None]; rows[:, :, 1] = p[None, :]; rows[:, :, 2] = rel[:, None]
                result.append(self.model(self.graph, torch.as_tensor(rows, device=self.device)).cpu().numpy())
        return np.concatenate(result)

    def triples(self, rows):
        # Evaluation messages depend only on (head, relation). Score all requested
        # tails together so a high-degree head does not repeat full-graph inference.
        result = np.empty(len(rows), np.float32)
        groups = {}
        for i, (h, r, _) in enumerate(rows): groups.setdefault((int(h), int(r)), []).append(i)
        queries = list(groups)
        with torch.inference_mode():
            for start in range(0, len(queries), self.batch):
                chunk = queries[start:start+self.batch]
                width = max(len(groups[q]) for q in chunk)
                batch = np.empty((len(chunk), width, 3), np.int64)
                for j, (h, r) in enumerate(chunk):
                    indices = groups[h, r]
                    tails = self.mapping[rows[indices, 2]]
                    batch[j, :, 0] = self.mapping[h]; batch[j, :, 2] = r
                    batch[j, :, 1] = tails[0]; batch[j, :len(tails), 1] = tails
                assert (batch[:, :, :2] >= 0).all()
                scores = self.model(self.graph, torch.as_tensor(batch, device=self.device)).cpu().numpy()
                for j, query in enumerate(chunk):
                    indices = groups[query]; result[indices] = scores[j, :len(indices)]
        return result
