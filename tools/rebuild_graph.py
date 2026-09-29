"""Rebuild from retained standard source tables. This is not raw-source ETL."""
from pathlib import Path
import argparse, contextlib, importlib.util, json, sys, time
sys.dont_write_bytecode=True
from verify_data import sha
CODE=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); root=args.data_root.resolve(); out=args.out.resolve()
    if out.is_relative_to(root): raise ValueError('Outputs must be outside immutable data')
    out.mkdir(parents=True,exist_ok=True)
    input_files=[p for name in ['kg_build/edges','kg_build/mapping','preprocessed'] for p in (root/name).rglob('*') if p.is_file()]
    input_files.append(root/'original_data/gmrepo/gmrepo_all_phenotypes.csv')
    before={p:sha(p) for p in input_files}
    spec=importlib.util.spec_from_file_location('retained_merge',CODE/'scripts/step_merge_kg.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.ROOT=root;mod.EDGE_DIR=root/'kg_build/edges';mod.MAPPING_DIR=root/'kg_build/mapping'
    mod.GUTMGENE_DIR=root/'preprocessed/gutmgene';mod.AGORA2_DIR=root/'preprocessed/agora2'
    mod.METABOLITE_CANONICAL_PATH=mod.MAPPING_DIR/'metabolite_canonical.tsv'
    mod.MICROBE_FALLBACK_PATH=mod.MAPPING_DIR/'microbe_name_fallback.tsv'
    mod.FINAL_EDGES_PATH=out/'final_kg_edges.tsv';mod.FINAL_NODES_PATH=out/'final_kg_nodes.tsv'
    mod.REPORT_DIR=out;mod.REPORT_PATH=out/'final_kg_report.txt'
    start=time.monotonic()
    with (out/'run.log').open('w',encoding='utf-8',buffering=1) as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f): mod.main()
    checks=[dict(file=name,expected=sha(root/'kg_build'/name),actual=sha(out/name))
        for name in ['final_kg_edges.tsv','final_kg_nodes.tsv']]
    unchanged=all(sha(p)==h for p,h in before.items())
    report=dict(seconds=time.monotonic()-start,input_files_unchanged=unchanged,checks=checks,
        passed=unchanged and all(x['expected']==x['actual'] for x in checks),
        scope='Saved standardized source tables, mappings, preprocessing and GMrepo phenotype-name lookup to frozen graph; upstream extraction is not reproduced.')
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not report['passed']: raise SystemExit(1)

if __name__=='__main__': main()
