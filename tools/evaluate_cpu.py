"""Run a CPU baseline on externally supplied, manifest-verified benchmark splits."""
from pathlib import Path
import argparse,json,sys
import numpy as np

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'benchmark'))
from core import LIGHT,LightScorer,load_case,evaluate_ranks
from verify_data import verify

def primary_sides(setting,raw):
    if setting in {'t1_cold_microbe','t4_c'}:return ['head']
    if setting in {'t1_cold_disease','t2_a','t2_b','t2_c100'}:return ['tail']
    return [s for s in ['head','tail'] if s+'_ranks' in raw]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True,help='One indexed dataset directory containing manifest.json')
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--setting',required=True)
    ap.add_argument('--model',choices=LIGHT,default='Popularity')
    ap.add_argument('--seeds',nargs='+',type=int,default=[42,123,456]);ap.add_argument('--batch',type=int,default=128)
    args=ap.parse_args();root=args.data.resolve();out=args.out.resolve()
    if out.is_relative_to(root):ap.error('--out must be outside the input directory')
    if args.setting=='t4_b':ap.error('T3-B is discrimination: use benchmark/run.py and its recorded-control protocol')
    verification=verify(root)
    if not verification['passed']:raise ValueError(verification)
    out.mkdir(parents=True,exist_ok=True);reports=[]
    for seed in args.seeds:
        case=load_case(root,args.setting,seed)
        scorer=LightScorer(case['train'],len(case['entities']),args.model,seed)
        raw=evaluate_ranks(case,scorer,batch=args.batch);sides=primary_sides(args.setting,raw)
        count=len(raw['ranking_row_ids']);total=len(case['arrays']['test'])
        mrr=float(np.mean([np.mean(1/raw[s+'_ranks']) for s in sides])) if count else None
        row={'setting':args.setting,'model':args.model,'seed':seed,'primary_sides':sides,
             'record_weighted_mrr':mrr,'eligible_records':count,'test_records':total,
             'coverage':count/total if total else None}
        np.savez_compressed(out/f'{args.setting}_{args.model.lower()}_s{seed}.npz',**raw)
        reports.append(row);print(json.dumps(row),flush=True)
    report={'input_verification':verification,'results':reports}
    (out/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
