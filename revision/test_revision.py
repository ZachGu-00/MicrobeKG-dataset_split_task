"""Scientific protocol checks for changes introduced by this supplement."""
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
from common import ROOT, PLAN, read
from report import paired_vectors,cluster_bootstrap,weighted_auc,control_bootstrap


class RevisionTests(unittest.TestCase):
    def test_pair_alignment_and_cluster(self):
        a=dict(ranking_triples=np.array([[1,0,3],[1,1,3],[2,0,4]]),head_ranks=np.array([1.,2.,4.]),
               tail_ranks=np.array([2.,4.,8.]),head_pool_size=np.array([10,10,10]),tail_pool_size=np.array([8,8,8]))
        b={k:v[[2,0,1]] for k,v in a.items()}
        pairs,delta=paired_vectors(a,b)
        boot,n=cluster_bootstrap(pairs,delta,100,42)
        self.assertEqual(n,2); np.testing.assert_array_equal(boot,0)
        b['tail_pool_size'][0]=7
        with self.assertRaises(AssertionError): paired_vectors(a,b)

    def test_weighted_auc_matches_expanded_sample(self):
        from sklearn.metrics import roc_auc_score
        pos=np.array([1.,2.,2.]); neg=np.array([0.,1.,2.,3.])
        pw=np.array([2,0,3]); nw=np.array([1,2,3,2])
        pp=np.repeat(pos,pw); nn=np.repeat(neg,nw)
        expected=roc_auc_score(np.r_[np.ones(len(pp)),np.zeros(len(nn))],np.r_[pp,nn])
        self.assertAlmostEqual(weighted_auc(pos,neg,pw,nw),expected)

    def test_paired_control_null(self):
        label='association_without_recorded_therapy'
        source={label+'_pos_scores':np.array([2.,4.,3.]),label+'_neg_scores':np.array([1.,3.,2.]),
                label+'_pos_ids':np.arange(3),label+'_neg_ids':np.arange(3)+10}
        match=dict(positive_indices=np.arange(3),negative_indices=np.arange(3),
                   original_positive_ids=np.arange(3),original_negative_ids=np.arange(3)+10)
        point,boot,n=control_bootstrap(source,match,100,42)
        self.assertEqual(point,0); np.testing.assert_array_equal(boot,0)

    def test_plan_no_test_selection(self):
        jobs=read(PLAN)['jobs']; self.assertEqual(len(jobs),147)
        hpo=[j for j in jobs if j['kind']=='hpo']; self.assertEqual(len(hpo),14)
        self.assertTrue(all(j['setting']=='t1_pair' and j['seed']==42 and j['config']['validation_only'] for j in hpo))
        self.assertEqual(len([j for j in jobs if j.get('selected_model')]),48)
        for j in jobs:
            if j['kind']=='matched': self.assertIn(j['source_job'],[k['id'] for k in jobs])
        from launcher import select
        with patch('launcher.verified',return_value=False):
            with self.assertRaises(AssertionError): select()

    def test_extra_t4_shuffle_preserves_protocol(self):
        from core import load_case
        for seed in [123,456]:
            old=load_case(ROOT/'benchmark_data','t4_a',seed)
            new=load_case(ROOT/f'revision_data/shuffle/t4_a_seed{seed}','t4_a',seed)
            for key in old['arrays']: np.testing.assert_array_equal(old['arrays'][key],new['arrays'][key])
            for relation in range(len(old['relations'])):
                a=old['train'][old['train'][:,1]==relation]; b=new['train'][new['train'][:,1]==relation]
                for col in [0,2]:
                    np.testing.assert_array_equal(np.bincount(a[:,col],minlength=len(old['seen'])),np.bincount(b[:,col],minlength=len(old['seen'])))
            for name in ['valid','test','negatives','conflict']:
                np.testing.assert_array_equal(old['triples'][old['arrays'][name]],new['triples'][new['arrays'][name]])

    def test_bilinear_config_roundtrip(self):
        import torch
        from pykeen.models import model_resolver
        from pykeen.triples import CoreTriplesFactory
        from run import model_kwargs,restore_best_state
        from common import write_json
        from supplement_cell import restored_scorer
        factory=CoreTriplesFactory(torch.tensor([[0,0,1],[1,0,2]]),num_entities=3,num_relations=1)
        for name in ['ComplEx','DistMult']:
            cfg=dict(model=name,dim=128,seed=42,loss='nssa',nssa_margin=0.,init_scale=.1 if name=='ComplEx' else 1.,regularizer_weight=.0001)
            a=model_resolver.make(name,model_kwargs(factory,cfg)); state={k:v.clone() for k,v in a.state_dict().items()}
            b=model_resolver.make(name,model_kwargs(factory,cfg)); restore_best_state(b,state)
            for k,v in state.items(): self.assertTrue(torch.equal(v,b.state_dict()[k]))
            with TemporaryDirectory() as temporary:
                path=Path(temporary)/'result.json';write_json(path,dict(config=cfg))
                torch.save(dict(state_dict=state,config=cfg,global_entity_indices=[0,1,2],relations=['r']),path.with_suffix('.pt'))
                case=dict(seen=np.ones(3,bool),train=np.array([[0,0,1],[1,0,2]]),relations=['r'])
                scorer=restored_scorer(case,path,'cpu'); a.eval()
                np.testing.assert_allclose(scorer.triples(case['train']),a.predict_hrt(factory.mapped_triples).detach().numpy().ravel(),atol=1e-6)
            if name=='ComplEx':
                self.assertLess(float(a.entity_representations[0]._embeddings.weight.detach().std()),.15)

    def test_train_only_prompts_and_inverse(self):
        import torch
        torch.set_num_threads(2)
        from core import load_case
        from kgicl_adapter import KGICLScorer
        c=load_case(ROOT/'benchmark_data','t1_pair',42,smoke=True)
        scorer=KGICLScorer(c,device='cpu',reference=True,batch=2)
        mapped=c['train'].copy(); mapped[:,0]=scorer.mapping[mapped[:,0]]; mapped[:,2]=scorer.mapping[mapped[:,2]]
        allowed=set(map(tuple,mapped))
        self.assertTrue(all(tuple(t) in allowed for ss in scorer.prompt_supports.values() for t in ss))
        rows=c['train'][np.isin(c['train'][:,1],[c['relations'].index(r) for r in c['meta']['scoring_relations']])][:2]
        pool=np.flatnonzero(c['seen'])
        batched=scorer.candidates(rows[:,0],rows[:,1],pool,'tail')
        scorer.batch=1
        single=scorer.candidates(rows[:,0],rows[:,1],pool,'tail')
        np.testing.assert_allclose(batched,single,atol=1e-5,rtol=1e-5)
        heads=scorer.candidates(rows[:,2],rows[:,1],pool,'head')
        with torch.inference_mode():
            inverse=scorer.model(scorer.mapping[rows[:1,2]],rows[:1,1]+scorer.num_relations,loader=scorer.loader,training=False)[0]
        np.testing.assert_allclose(heads[:1],inverse[:,scorer.mapping[pool]].numpy(),atol=1e-5,rtol=1e-5)


if __name__=='__main__': unittest.main()
