from __future__ import annotations
from collections import defaultdict
import numpy as np

HITS=(1,3,5,10,20)

def filtered_rank(scores,target_position,known_positions):
    scores=np.asarray(scores,float); assert np.isfinite(scores).all()
    keep=np.ones(len(scores),bool); keep[np.asarray(list(known_positions),dtype=int)]=False; keep[target_position]=False
    return 1+np.count_nonzero(scores[keep]>scores[target_position])+0.5*np.count_nonzero(scores[keep]==scores[target_position])

def ranking_metrics(ranks, groups=None, seed=42, n_boot=1000):
    ranks=np.asarray(ranks,float)
    if not len(ranks): return None
    out={'mrr':float(np.mean(1/ranks)),**{f'hits_at_{k}':float(np.mean(ranks<=k)) for k in HITS},'n_ranks':len(ranks)}
    if groups is None: groups=np.arange(len(ranks))
    groups=np.asarray(groups); vals=[]
    for g in np.unique(groups): vals.append(float(np.mean(1/ranks[groups==g])))
    out['macro_mrr']=float(np.mean(vals)); rng=np.random.default_rng(seed); boot=[float(np.mean(rng.choice(vals,len(vals),replace=True))) for _ in range(n_boot)]
    out['macro_mrr_ci95']=np.quantile(boot,[.025,.975]).tolist()
    return out

class Evaluator:
    """Frozen-core-compatible filtered ranking for explicit triples.

    `metrics['macro_mrr']` is the study-macro MRR: reciprocal ranks of both
    sides are averaged within each study_id, then across studies. Pass
    `groups=` to use another grouping.

    A scorer may implement candidates(queries, relations, pool, side), returning
    a score matrix. Alternatively pass score arrays to evaluate_scores.
    """
    def __init__(self,case): self.case=case
    def _known(self, rows):
        kt,kh=defaultdict(set),defaultdict(set)
        for h,r,t in rows:
            kt[(int(h),int(r))].add(int(t)); kh[(int(r),int(t))].add(int(h))
        return kt,kh
    def evaluate(self,scorer,split='test',batch=16,groups=None):
        rows=getattr(self.case,split)
        mask=self.case.arrays.get(f'{split}_eligible')
        study=self.case.study_ids.get(split) if hasattr(self.case,'study_ids') else None
        if mask is not None and len(mask)==len(rows): rows=rows[mask]; study=None if study is None else study[mask]
        train=self.case.train; extra=self.case.arrays.get('known_extra',np.empty((0,3),np.int32))
        known_rows=np.concatenate([train,self.case.valid,extra,rows]) if split=='test' else np.concatenate([train])
        kt,kh=self._known(known_rows); sides=['tail','head']; out={}
        for side in sides:
            wanted='disease' if side=='tail' else 'microbe'; pool=np.flatnonzero((self.case.node_types==wanted)&self.case.candidate_mask); pos={int(v):i for i,v in enumerate(pool)}
            query_groups=defaultdict(list)
            for i,(h,r,t) in enumerate(rows): query_groups[(int(h if side=='tail' else t),int(r))].append(i)
            ranks=np.empty(len(rows),float)
            for start in range(0,len(query_groups),batch):
                chunk=list(query_groups)[start:start+batch]; scores=scorer.candidates(np.array([q for q,_ in chunk]),np.array([r for _,r in chunk]),pool,side)
                for j,(q,r) in enumerate(chunk):
                    gold=kt.get((q,r),()) if side=='tail' else kh.get((r,q),()); known=[pos[v] for v in gold if v in pos]
                    for i in query_groups[(q,r)]:
                        target=rows[i,2] if side=='tail' else rows[i,0]; ranks[i]=filtered_rank(scores[j],pos[int(target)],known)
            out[side]=ranks
        groups=(rows[:,0] if study is None else study) if groups is None else np.asarray(groups)
        merged=np.concatenate([out['tail'],out['head']]); return {'ranks':out,'metrics':ranking_metrics(merged, np.tile(groups,2), seed=self.case.seed), 'ranking_triples':rows}
    def evaluate_scores(self,scores_tail,scores_head=None,split='test',groups=None):
        rows=getattr(self.case,split)
        mask=self.case.arrays.get(f'{split}_eligible')
        study=self.case.study_ids.get(split) if hasattr(self.case,'study_ids') else None
        if mask is not None and len(mask)==len(rows): rows=rows[mask]; study=None if study is None else study[mask]
        a=np.asarray(scores_tail); b=a if scores_head is None else np.asarray(scores_head)
        known=np.concatenate([self.case.train,self.case.valid,self.case.arrays.get('known_extra',np.empty((0,3),np.int32)),rows]) if split=='test' else self.case.train
        kt,kh=self._known(known)
        ranks={}
        for side,scores in [('tail',a),('head',b)]:
            wanted='disease' if side=='tail' else 'microbe'; pool=np.flatnonzero((self.case.node_types==wanted)&self.case.candidate_mask); pos={int(v):i for i,v in enumerate(pool)}; rr=[]
            for i,(h,r,t) in enumerate(rows):
                q=int(h if side=='tail' else t); target=int(t if side=='tail' else h); gold=kt.get((q,int(r)),()) if side=='tail' else kh.get((int(r),q),()); rr.append(filtered_rank(scores[i],pos[target],[pos[v] for v in gold if v in pos]))
            ranks[side]=np.asarray(rr)
        groups=(rows[:,0] if study is None else study) if groups is None else np.asarray(groups)
        return {'ranks':ranks,'metrics':ranking_metrics(np.concatenate([ranks['tail'],ranks['head']]),np.tile(groups,2),seed=self.case.seed)}
