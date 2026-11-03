from __future__ import annotations
import numpy as np
from scipy import sparse

class DegreePrior:
    """Popularity scorer matching core.LightScorer(model='Popularity')."""
    def __init__(self,train,n):
        self.degree=np.bincount(np.r_[train[:,0],train[:,2]],minlength=n)
    def candidates(self,queries,relations,pool,side):
        q=np.asarray(queries); p=np.asarray(pool)
        return self.degree[q,None]*self.degree[p][None,:]
    def triples(self,rows): return self.degree[rows[:,0]]*self.degree[rows[:,2]]

Popularity=DegreePrior

class _GraphScorer:
    def __init__(self,train,n):
        h,t=train[:,0],train[:,2]; self.adj=(sparse.csr_matrix((np.ones(2*len(h)),(np.r_[h,t],np.r_[t,h])),shape=(n,n))>0).astype(np.float32)
        deg=np.asarray(self.adj.sum(axis=1)).ravel(); self.inv=sparse.diags(1/np.maximum(deg,1)); self.sqrt_inv=sparse.diags(1/np.sqrt(np.maximum(deg,1)))
    def _paths(self,q):
        v=self.adj[q]
        if self.model=='CN': return v@self.adj
        if self.model=='RA': return (v@self.inv)@self.adj
        return (((v@self.sqrt_inv)@self.adj)@self.sqrt_inv)@self.adj
    def candidates(self,queries,relations,pool,side): return self._paths(np.asarray(queries))[:,np.asarray(pool)].toarray()
    def triples(self,rows):
        out=np.empty(len(rows))
        for h in np.unique(rows[:,0]):
            m=rows[:,0]==h; out[m]=self._paths(np.array([h]))[:,rows[m,2]].toarray().ravel()
        return out
class CN(_GraphScorer): model='CN'
class RA(_GraphScorer): model='RA'
class L3(_GraphScorer): model='L3'
