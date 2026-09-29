"""Exercise nonzero official-model inference before scheduling expensive jobs."""
import argparse
import numpy as np
import torch
from common import ROOT, OUT, write_json, sha256
from core import load_case
from kgicl_adapter import KGICLScorer
from ultra_adapter import UltraScorer


def check(device='cpu'):
    torch.set_num_threads(4)
    case=load_case(ROOT/'benchmark_data','t1_pair',42,smoke=True)
    rows=case['train'][np.isin(case['train'][:,1],[case['relations'].index(r) for r in case['meta']['scoring_relations']])][:3]
    pool=np.flatnonzero(case['seen'])
    output={}
    for name, scorer in [('KG-ICL',KGICLScorer(case,device=device,reference=device=='cpu',batch=1)),
                         ('ULTRA',UltraScorer(case,device=device,reference=device=='cpu',batch=1))]:
        a=scorer.candidates(rows[:,0],rows[:,1],pool,'tail')
        b=scorer.candidates(rows[:,0],rows[:,1],pool,'tail')
        assert np.isfinite(a).all() and np.ptp(a)>1e-6, (name,'constant or invalid scores')
        np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-5)
        direct=scorer.triples(rows)
        np.testing.assert_allclose(direct,a[np.arange(len(rows)),np.searchsorted(pool,rows[:,2])],atol=1e-5,rtol=1e-5)
        output[name]=dict(status='passed',score_range=[float(a.min()),float(a.max())],checkpoint_sha256=sha256(scorer.checkpoint))
    write_json(OUT/f'preflight_{device}.json',output)
    print(output)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    check(ap.parse_args().device)
