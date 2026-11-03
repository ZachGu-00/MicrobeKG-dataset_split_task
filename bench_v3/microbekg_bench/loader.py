from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass
class Case:
    setting: str
    seed: int
    entities: pd.DataFrame
    relations: list[str]
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    candidate_mask: np.ndarray
    filter_triples: np.ndarray
    arrays: dict[str, np.ndarray]
    metadata: dict
    root: Path
    study_ids: dict[str, np.ndarray]
    @property
    def n_entities(self): return len(self.entities)
    @property
    def node_types(self): return self.entities.node_type.to_numpy()
    @property
    def relation_ids(self): return {r:i for i,r in enumerate(self.relations)}

def _read_rows(path, ent, rid):
    df=pd.read_parquet(path)
    if len(df)==0: return np.empty((0,3),np.int32)
    idx={(t,i):j for j,t,i in ent[['entity_index','node_type','node_id']].itertuples(index=False,name=None)}
    return np.asarray([[idx[(ht,h)],rid[r],idx[(tt,t)]] for h,ht,r,t,tt in df[['head_id','head_type','relation','tail_id','tail_type']].itertuples(index=False,name=None)],dtype=np.int32)

def load(setting: str, seed: int, root='.', *, external=False) -> Case:
    """Load explicit triple tables, candidate mask, and filter set.

    `root` is the local release directory. `external=True` loads the
    t1_external split under external/. `filter_triples` is train plus
    known_extra; the evaluator additionally filters validation and test
    triples in test mode. `study_ids['valid' | 'test']` holds the evidence
    study of every row (used for study-macro MRR).
    """
    root=Path(root); base=root/'external' if external or setting=='t1_external' else root/'benchmark'
    ent=pd.read_csv(base/'entities.tsv',sep='\t',dtype=str,keep_default_na=False); rel=json.loads((base/'relations.json').read_text()); rid={r:i for i,r in enumerate(rel)}
    sd=base/'splits'/setting; prefix=sd/f'seed_{seed}'
    meta=json.loads(prefix.with_suffix('.json').read_text()); raw=np.load(prefix.with_name(prefix.name+'_aux.npz'),allow_pickle=False)
    read=lambda part:_read_rows(sd/f'seed_{seed}_{part}.parquet',ent,rid)
    train,valid,test=read('train'),read('valid'),read('test'); known=read('known_extra')
    candidate=np.zeros(len(ent),bool); c=pd.read_parquet(sd/f'seed_{seed}_candidate_entities.parquet'); candidate[c.entity_index.to_numpy(dtype=int)]=True
    seen=np.zeros(len(ent),bool); seen[np.concatenate([train[:,[0,2]].ravel()])]=True
    candidate &= seen
    # Known filters are all training assertions, plus known_extra. The evaluator adds validation/test assertions for test mode.
    filt=np.concatenate([train,known]) if len(known) else train.copy()
    arrays={k:raw[k] for k in raw.files}
    arrays['known_extra']=known
    study={part:pd.read_parquet(sd/f'seed_{seed}_{part}.parquet',columns=['study_id']).study_id.to_numpy(dtype=str) for part in ('valid','test')}
    return Case(setting,seed,ent,rel,train,valid,test,candidate,filt,arrays,meta,root,study)
