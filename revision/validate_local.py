"""Bounded tests only. Formal server jobs are never started by this validator."""
import subprocess
import sys
from common import ROOT, OUT, write_json, sha256, code_hash


def main():
    dest=ROOT/'revision/validation';dest.mkdir(exist_ok=True)
    commands=[
      ['revision/test_revision.py'],
      ['revision/preflight.py','--device','cpu'],
      ['revision/cell.py','--job-id','hpo__complex__c0','--device','cuda','--smoke'],
      ['revision/cell.py','--job-id','hpo__distmult__c0','--device','cuda','--smoke'],
      ['revision/cell.py','--job-id','kgicl__t4_b__s42','--device','cpu','--smoke'],
      ['revision/cell.py','--job-id','ultra__t2_a__s42','--device','cpu','--smoke'],
      ['revision/cell.py','--job-id','shuffle__t4_a__rgcn__s123','--device','cuda','--smoke'],
    ]
    checks=[]
    for i,args in enumerate(commands):
        log=dest/f'check_{i+1}.log'
        with log.open('w',encoding='utf-8') as f:
            subprocess.run([sys.executable,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=240)
        checks.append(dict(command=args,status='passed',log=log.relative_to(ROOT).as_posix(),sha256=sha256(log)))
        print('PASS',args,flush=True)
    write_json(dest/'validation.json',dict(status='passed',code_sha256=code_hash(),checks=checks,
        limits='Tiny real-data smoke only; full formal 147-cell suite and native Linux CUDA rspmm still run on target server.'))


if __name__=='__main__': main()
