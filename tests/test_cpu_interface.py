"""Data-free checks of publication entry points and scientific ranking rules."""
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib,json,subprocess,sys,unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmark'));sys.path.insert(0,str(ROOT/'tools'))
from core import filtered_rank,load_case,evaluate_ranks
from evaluate_cpu import primary_sides
from verify_data import verify,safe_path

def fixture(root):
    root.mkdir();split=root/'splits/t2_b';split.mkdir(parents=True)
    types=['microbe','microbe','substrate','substrate','substrate','substrate','disease']
    pd.DataFrame({'entity_index':range(7),'node_type':types,'node_id':[str(i) for i in range(7)]}).to_csv(root/'entities.tsv',sep='\t',index=False)
    (root/'relations.json').write_text(json.dumps(['can_utilize','context']))
    triples=np.array([[0,0,2],[1,0,3],[1,0,4],[0,1,6],[0,0,3],[0,0,4]],dtype=np.int32)
    np.save(root/'triples.npy',triples)
    arrays={'train':np.arange(4,dtype=np.int32),'valid':np.array([4]),'test':np.array([5]),
            'valid_eligible':np.array([True]),'test_eligible':np.array([True]),
            'candidate_eligible':np.ones(7,dtype=bool),'known_extra':np.array([],dtype=np.int32)}
    for key in ['negatives','conflict']:
        arrays[key]=np.array([],dtype=np.int32);arrays['negative_eligible' if key=='negatives' else 'conflict_eligible']=np.array([],dtype=bool)
    np.savez(split/'seed_42.npz',**arrays)
    (split/'seed_42.json').write_text(json.dumps({'seed':42,'ranking_sides':'tail','head_type':'microbe','tail_type':'substrate','scoring_relations':['can_utilize']}))
    files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    (root/'manifest.json').write_text(json.dumps({'files':files,'original_edges_sha256':'generated-test-fixture'}))

class CPUInterfaceTests(unittest.TestCase):
    def test_filter_known_answers_and_average_ties(self):
        self.assertEqual(filtered_rank(np.array([9.,5.,5.,1.]),1,[0]),1.5)

    def test_typed_train_visible_candidates_and_filtering(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'data';fixture(root);case=load_case(root,'t2_b',42)
            class ConstantScorer:
                def candidates(self,queries,relations,pool,side):
                    np.testing.assert_array_equal(pool,[2,3,4])
                    return np.ones((len(queries),len(pool)))
            raw=evaluate_ranks(case,ConstantScorer())
            np.testing.assert_array_equal(raw['tail_ranks'],[1.])
            self.assertEqual(int(raw['tail_pool_size']),3)
            case['triples']._mmap.close()

    def test_primary_metric_sides(self):
        both={'head_ranks':np.array([1.]),'tail_ranks':np.array([2.])}
        for setting in ['t1_cold_microbe','t4_c']:self.assertEqual(primary_sides(setting,both),['head'])
        for setting in ['t1_cold_disease','t2_a','t2_b']:self.assertEqual(primary_sides(setting,both),['tail'])
        self.assertEqual(primary_sides('t1_pair',both),['head','tail'])

    def test_manifest_tampering_and_path_escape(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'data';fixture(root);self.assertTrue(verify(root)['passed'])
            (root/'relations.json').write_text('[]');self.assertFalse(verify(root)['passed'])
            with self.assertRaises(ValueError):safe_path(root,'../outside')

    def test_cpu_cli_and_input_protection(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'data';fixture(root);out=Path(tmp)/'output'
            args=[sys.executable,str(ROOT/'tools/evaluate_cpu.py'),'--data',str(root),'--setting','t2_b','--seeds','42','--out',str(out)]
            done=subprocess.run(args,capture_output=True,text=True);self.assertEqual(done.returncode,0,done.stderr)
            result=json.loads((out/'evaluation.json').read_text())['results'][0]
            self.assertEqual(result['record_weighted_mrr'],1.)
            self.assertEqual(result['primary_sides'],['tail']);self.assertEqual(result['coverage'],1.)
            args[-1]=str(root/'bad-output')
            bad=subprocess.run(args,capture_output=True,text=True);self.assertNotEqual(bad.returncode,0)
            self.assertFalse((root/'bad-output').exists())

if __name__=='__main__':unittest.main()
