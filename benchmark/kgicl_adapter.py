"""KG-ICL-6L, frozen upstream weights, train-only prompts and common evaluation.

The upstream GPL-3.0 source is retained in vendor/KG-ICL. Obtain its checkpoint separately.
No target validation/test labels enter graph construction or prompt selection.
"""
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import random
import sys
import types

import numpy as np
import torch

from prepare import ROOT


def import_kgicl(reference=False):
    if reference:
        try:
            import torch_scatter
        except ImportError:
            from torch_geometric.utils import scatter
            module = types.ModuleType('torch_scatter')
            module.scatter = scatter
            module.scatter_add = lambda src, index, dim=0, dim_size=None: scatter(src, index, dim=dim, dim_size=dim_size, reduce='sum')
            module.scatter_mean = lambda src, index, dim=0, dim_size=None: scatter(src, index, dim=dim, dim_size=dim_size, reduce='mean')
            module.scatter_max = lambda src, index, dim=0, dim_size=None: (scatter(src, index, dim=dim, dim_size=dim_size, reduce='max'), None)
            sys.modules['torch_scatter'] = module
    sys.path.insert(0, str(ROOT / 'vendor/KG-ICL/src'))
    sys.path.insert(0, str(ROOT / 'vendor/KG-ICL/run_your_own_dataset'))
    from encoder.EntityEncoder import EntityEncoder
    from data_loader import DataLoader, KG
    from utils_all import KG as PromptGraph
    return EntityEncoder, DataLoader, KG, PromptGraph


def make_prompts(triples, n_entities, n_relations, query_relations, seed, shot=5):
    """Upstream 3-hop prompt rule with explicit node/label alignment."""
    _, _, _, PromptGraph = import_kgicl()
    random.seed(seed); np.random.seed(seed)
    generator = PromptGraph(triples, n_entities, n_relations)
    generator.ht2r = defaultdict(set)
    for h, r, t in triples:
        generator.ht2r[int(h), int(t)].add(int(r))
    generator.build_nx_graph()
    cases, supports = {}, {}
    for relation in sorted(query_relations):
        ids = np.flatnonzero(triples[:, 1] == relation)
        assert len(ids), ('scoring relation has no training examples', relation)
        chosen = np.random.permutation(ids)[:min(shot * 3, len(ids))]
        cases[relation] = {}; cases[relation+n_relations] = {}; supports[str(relation)] = []
        for idx in chosen:
            head, _, tail = map(int, triples[idx])
            if head == tail:
                continue
            labels, nodes = generator.get_enclosing_nodes_large(head, tail, enclosing=False, hop=3)
            nodes = list(map(int, nodes))
            assert nodes[:2] == [head, tail]
            mapping = {node: i for i, node in enumerate(nodes)}
            prompt = []
            for u, v in generator.G.subgraph(nodes).edges():
                for a, b in [(u, v), (v, u)]:
                    prompt.extend((mapping[a], r, mapping[b]) for r in sorted(generator.ht2r[a, b]))
            prompt = np.asarray(prompt, dtype=np.int64).reshape(-1, 3)
            assert len(prompt) and np.max(labels) <= 3
            edges = np.r_[prompt[:, [0, 2]], prompt[:, [2, 0]]].T.copy()
            edge_types = np.r_[prompt[:, 1], prompt[:, 1]+n_relations].copy()
            number = len(cases[relation])
            cases[relation][number] = (edges, edge_types, relation, labels, len(nodes))
            cases[relation+n_relations][number] = (edges, edge_types, relation+n_relations, labels, len(nodes))
            supports[str(relation)].append([head, relation, tail])
            if number + 1 == shot:
                break
        assert cases[relation], ('no usable train-only prompt', relation)
    return cases, supports


class KGICLScorer:
    def __init__(self, case, device='cuda', reference=False, batch=1, seed=42):
        EntityEncoder, DataLoader, KG, _ = import_kgicl(reference)
        self.device = torch.device(device); self.batch = batch
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.mapping = np.full(len(case['seen']), -1, np.int64)
        self.mapping[case['seen']] = np.arange(case['seen'].sum())
        x = case['train'].copy().astype(np.int64)
        x[:, 0] = self.mapping[x[:, 0]]; x[:, 2] = self.mapping[x[:, 2]]
        self.num_relations = len(case['relations'])
        n = int(case['seen'].sum())
        args = SimpleNamespace(device=self.device, shot=5, hidden_dim=32, attn_dim=5,
            n_layer=6, n_relation_encoder_layer=3, act='idd', dropout=0., use_rspmm=False,
            use_attn=True, attn_type='Sigmoid', AGG='max', AGG_rel='max', MSG='concat',
            use_augment=False, use_token_set=True, use_prompt_graph=True, path_hop=3,
            relation_mask_rate=0.)
        kg = KG.__new__(KG)
        kg.args = args; kg.entity_num = n; kg.relation_num = 2*self.num_relations
        # Disable the optional distance prior; do not derive it from held-out labels.
        kg.answer_distance = defaultdict(lambda: 1.)
        query_relations = [case['relations'].index(r) for r in case['meta']['scoring_relations']]
        kg.cases, self.prompt_supports = make_prompts(x, n, self.num_relations, query_relations, seed)
        self.loader = DataLoader.__new__(DataLoader)
        self.loader.args = args; self.loader.kg = kg
        inverse = x[:, [2, 1, 0]].copy(); inverse[:, 1] += self.num_relations
        self.loader.build_graph(np.r_[x, inverse])
        self.model = EntityEncoder(args).to(self.device)
        self.checkpoint = ROOT / 'vendor/KG-ICL/checkpoint/KG-ICL-6L/model_best.tar'
        payload = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        state = payload['state_dict']
        # These upstream fused-conv parameters are declared but never used by
        # forward/message/aggregate/update. Assert an exact whitelist, then load
        # every parameter of the official non-fused evaluation path strictly.
        unused = {f'gnn_layers.{i}.conv.{suffix}' for i in range(6) for suffix in
                  ['linear.weight', 'linear.bias', 'relation_projection.0.weight',
                   'relation_projection.0.bias', 'relation_projection.2.weight', 'relation_projection.2.bias']}
        extra = set(state) - set(self.model.state_dict())
        assert extra <= unused, ('unexpected checkpoint keys', extra-unused)
        self.model.load_state_dict({k: v for k, v in state.items() if k not in extra}, strict=True)
        self.model.eval()

    def candidates(self, queries, relations, pool, side):
        q, p = self.mapping[queries], self.mapping[pool]
        assert (q >= 0).all() and (p >= 0).all()
        r = np.asarray(relations) + (self.num_relations if side == 'head' else 0)
        result = []
        with torch.inference_mode():
            for start in range(0, len(q), self.batch):
                scores = self.model(q[start:start+self.batch], r[start:start+self.batch],
                                    loader=self.loader, training=False)[0]
                result.append(scores[:, p].cpu().numpy())
        return np.concatenate(result)

    def triples(self, rows):
        result = np.empty(len(rows), np.float32)
        groups = defaultdict(list)
        for i, (h, r, _) in enumerate(rows): groups[int(h), int(r)].append(i)
        for (h, r), indices in groups.items():
            result[indices] = self.candidates(np.array([h]), np.array([r]), rows[indices, 2], 'tail')[0]
        return result
