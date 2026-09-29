"""Upload-package verification, sequential stages and resumable per-GPU jobs."""
import argparse
from collections import Counter
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from common import ROOT, OUT, PLAN, read, write_json, sha256, code_hash, folder, resolve, inputs, verified
from supplement import visible_gpus, available_cpu_threads, dispatch_jobs


def verify():
    manifest = ROOT / 'revision/bundle_manifest.json'
    if manifest.exists():
        for name, digest in read(manifest)['files'].items():
            assert sha256(ROOT/name) == digest, name
    for data in ['benchmark_data', 'supplement_data', 'revision_data/shuffle/t4_a_seed123', 'revision_data/shuffle/t4_a_seed456']:
        for name, value in read(ROOT/data/'manifest.json')['files'].items():
            assert sha256(ROOT/data/name) == (value['sha256'] if isinstance(value, dict) else value), (data, name)
    jobs = read(PLAN)['jobs']
    assert len(jobs) == 147 and len({j['id'] for j in jobs}) == 147
    print('All packaged inputs verified; 147 planned jobs.', flush=True)


def select():
    hpo = [j for j in read(PLAN)['jobs'] if j['kind']=='hpo']
    assert all(verified(j) for j in hpo), 'HPO incomplete: selection requires all 14 validated runs'
    selected = {}
    for model in ['ComplEx', 'DistMult']:
        runs = [(read(folder(j)/'result.json'), j) for j in hpo if j['model']==model]
        assert all(r['status']=='validation_complete' and r['config']['validation_only'] for r, j in runs)
        result, job = max(runs, key=lambda x: x[0]['training']['best_validation_mrr'])
        selected[model] = dict(job_id=job['id'], config=job['config'],
            validation_mrr=result['training']['best_validation_mrr'],
            source_sha256=sha256(folder(job)/'result.json'), criterion='fixed seed42 validation MRR only')
    path = OUT/'selected_configs.json'
    if path.exists():
        assert read(path)==selected, 'Frozen selection differs. Use a separate output snapshot before changing it.'
    else: write_json(path, selected)
    print('Frozen validation-only configurations:', {m:d['validation_mrr'] for m,d in selected.items()}, flush=True)


def setup():
    import torch
    assert torch.cuda.is_available(), 'Use a CUDA PyTorch environment on the experiment server'
    wanted = {'pykeen':'pykeen==1.11.1', 'torch_geometric':'torch-geometric==2.6.1',
              'torch_scatter':None, 'ninja':'ninja', 'networkx':'networkx', 'filelock':'filelock', 'tqdm':'tqdm'}
    missing = [p for m,p in wanted.items() if p and importlib.util.find_spec(m) is None]
    if missing: subprocess.run([sys.executable,'-m','pip','install',*missing], check=True)
    import importlib.metadata
    if importlib.metadata.version('pykeen')!='1.11.1':
        subprocess.run([sys.executable,'-m','pip','install','pykeen==1.11.1'],check=True)
    if importlib.util.find_spec('torch_scatter') is None:
        version = '.'.join(torch.__version__.split('+')[0].split('.')[:2])+'.0'
        cuda = 'cu'+torch.version.cuda.replace('.','')
        subprocess.run([sys.executable,'-m','pip','install','torch-scatter','-f',
                        f'https://data.pyg.org/whl/torch-{version}+{cuda}.html'], check=True)
    subprocess.run([sys.executable,str(ROOT/'revision/preflight.py'),'--device','cuda'], check=True)
    OUT.mkdir(exist_ok=True)
    with (OUT/'environment.txt').open('w',encoding='utf-8') as f:
        subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)


def launch(args):
    OUT.mkdir(exist_ok=True)
    if args.detach:
        with (OUT/'queue.log').open('a',encoding='utf-8') as log:
            kw = dict(stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            if os.name=='nt': kw['creationflags']=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            else: kw['start_new_session']=True
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run','--workers',str(args.workers)],**kw)
        print('Detached PID',p.pid,'log:',OUT/'queue.log'); return
    from filelock import FileLock
    with FileLock(str(OUT/'queue.lock'),timeout=0):
        verify()
        jobs=read(PLAN)['jobs']; by_id={j['id']:j for j in jobs}
        devices=visible_gpus(); threads=available_cpu_threads()
        selection_ready=False
        write_json(OUT/'queue_status.json',dict(status='running',pid=os.getpid(),devices=devices,total=len(jobs)))
        def one(job,gpu):
            if verified(job): return
            dest=folder(job); dest.mkdir(parents=True,exist_ok=True)
            if job.get('selected_model') and not selection_ready:
                write_json(dest/'receipt.json',dict(status='blocked_hpo',job=job)); return
            for dep in job.get('depends_on',[]):
                if not verified(by_id[dep]):
                    write_json(dest/'receipt.json',dict(status='blocked_dependency',dependency=dep)); return
            # Preserve every unsuccessful attempt, including runner-internal caches.
            old=[p for p in dest.iterdir() if p.name!='prior_attempts']
            if old:
                archive=dest/'prior_attempts'/str(time.time_ns()); archive.mkdir(parents=True)
                assert archive.resolve().is_relative_to(OUT.resolve())
                for p in old: p.rename(archive/p.name)
            env=os.environ.copy(); n=str(max(1,min(4,threads//(len(devices)*(1 if job['heavy'] else args.workers)))))
            env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS=n,MKL_NUM_THREADS=n,OPENBLAS_NUM_THREADS=n,
                       MICROBE_TORCH_THREADS=n,MAX_JOBS='4',PYTHONHASHSEED='0',
                       TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            print('START',job['id'],'GPU',gpu,flush=True)
            write_json(dest/'receipt.json',dict(status='running',job=job,gpu=gpu))
            with (dest/'run.log').open('w',encoding='utf-8') as log:
                try:
                    p=subprocess.run([sys.executable,'-u',str(ROOT/'revision/cell.py'),'--job-id',job['id']],
                                     stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT,timeout=job.get('timeout_seconds'))
                    if p.returncode and read(dest/'receipt.json').get('status')=='running':
                        write_json(dest/'receipt.json',dict(status='failed',returncode=p.returncode,job=job))
                except subprocess.TimeoutExpired:
                    write_json(dest/'receipt.json',dict(status='oot',timeout_seconds=job['timeout_seconds'],job=job))
            print('END',job['id'],read(dest/'receipt.json')['status'],flush=True)
        for stage in ['hpo','train','foundation','matched']:
            if stage=='train':
                try:
                    select(); selection_ready=True
                except AssertionError as exc:
                    print(str(exc),flush=True)
                    print('Dependent T1 jobs blocked; independent T4 and foundation jobs continue.',flush=True)
            for heavy, workers in [(False,args.workers),(True,1)]:
                dispatch_jobs([j for j in jobs if j['stage']==stage and j['heavy']==heavy],devices,workers,one)
        counts=Counter('complete' if verified(j) else
                       ('invalid_complete' if read(folder(j)/'receipt.json').get('status')=='complete' else read(folder(j)/'receipt.json').get('status','invalid'))
                       if (folder(j)/'receipt.json').exists() else 'missing' for j in jobs)
        write_json(OUT/'queue_status.json',dict(status='complete' if all(verified(j) for j in jobs) else 'finished_with_exceptions',counts=dict(counts)))
        subprocess.run([sys.executable,str(ROOT/'revision/report.py')],check=True)
        collect()


def collect():
    target=ROOT/'MicrobeKG_revision_results.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and 'prior_attempts' not in p.parts and p.suffix!='.lock':
                # result.pt is the canonical checkpoint; avoid runner-internal duplicate copies.
                if p.suffix=='.pt' and p.name!='result.pt': continue
                z.write(p,p.relative_to(ROOT).as_posix())
        z.write(PLAN,PLAN.relative_to(ROOT).as_posix())
    target.with_suffix('.sha256').write_text(sha256(target)+'  '+target.name+'\n')
    print('Results:',target,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['verify','setup','run','report','collect','status'])
    ap.add_argument('--workers',type=int,choices=[1,2,3],default=2)
    ap.add_argument('--detach',action='store_true'); args=ap.parse_args()
    if args.action=='verify': verify()
    elif args.action=='setup': setup()
    elif args.action=='run': launch(args)
    elif args.action=='report': subprocess.run([sys.executable,str(ROOT/'revision/report.py')],check=True)
    elif args.action=='collect': collect()
    else: print(read(OUT/'queue_status.json') if (OUT/'queue_status.json').exists() else 'not started')
