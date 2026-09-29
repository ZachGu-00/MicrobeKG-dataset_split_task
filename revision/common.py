"""Shared paths and receipt checks for the September revision package."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmark'))
from prepare import sha256, write_json

OUT = ROOT / 'revision_results'
PLAN = ROOT / 'revision/plan.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def code_hash():
    paths = sorted((ROOT / 'revision').glob('*.py')) + sorted((ROOT / 'benchmark').glob('*.py'))
    paths += sorted((ROOT / 'vendor/KG-ICL/src').rglob('*.py'))
    paths += [ROOT / 'vendor/KG-ICL/run_your_own_dataset/utils_all.py']
    paths += sorted(p for p in (ROOT / 'vendor/ULTRA/ultra').rglob('*') if p.suffix in ['.py','.cpp','.h','.cu','.cuh'])
    paths += [ROOT / 'vendor/KG-ICL/checkpoint/KG-ICL-6L/model_best.tar', ROOT / 'vendor/ULTRA/ckpts/ultra_50g.pth']
    return hashlib.sha256(''.join(sha256(p) for p in paths).encode()).hexdigest()


def folder(job, smoke=False):
    return OUT / ('smoke' if smoke else 'jobs') / job['id']


def resolve(job):
    job = dict(job)
    if job.get('selected_model'):
        chosen = read(OUT / 'selected_configs.json')[job['selected_model']]
        job['config'] = dict(chosen['config'], setting=job['setting'], seed=job['seed'],
                             split_seed=job['seed'], validation_only=False)
    if job.get('source_job'):
        job['source_result'] = (OUT / 'jobs' / job['source_job'] / 'result.json').relative_to(ROOT).as_posix()
    return job


def inputs(job):
    paths = [ROOT / 'benchmark_data/manifest.json']
    if job.get('data'): paths.append(ROOT / job['data'] / 'manifest.json')
    if job.get('selected_model'): paths.append(OUT / 'selected_configs.json')
    if job['kind'] == 'matched':
        p = ROOT / job['source_result']
        paths += [p, p.with_suffix('.npz'), ROOT / 'supplement_data/matched' / job['setting'] / f"seed_{job['seed']}.npz"]
        if job['model'] != 'KG-ICL': paths.append(p.with_suffix('.pt'))
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def verified(job, smoke=False):
    try:
        job = resolve(job)
        r = read(folder(job, smoke) / 'receipt.json')
        return (r['status'] == ('smoke_only' if smoke else 'complete') and r['job'] == job
                and r['code_sha256'] == code_hash() and r['inputs'] == inputs(job)
                and bool(r['files']) and all((ROOT / p).is_file() and sha256(ROOT / p) == h for p, h in r['files'].items()))
    except (FileNotFoundError, KeyError):
        return False
