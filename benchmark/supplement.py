"""Plan, verify, launch and resume the checklist's supplemental experiments."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import time

from core import LEARNED, MODELS
from prepare import CORE, ROOT, SEEDS, sha256, write_json
from run import run_name
from supplement_cell import BASELINE, OUTPUT, code_hash, job_output
from supplement_prepare import MATCH_SETTINGS, SOURCES


def training_result(job):
    from argparse import Namespace
    return (job_output(job) / 'main' / (run_name(Namespace(**job['config'])) + '.json')).relative_to(ROOT).as_posix()


def make_plan():
    originals = [json.loads(p.read_text()) for p in sorted((BASELINE / 'main').glob('*.json'))]
    lookup = {(d['config']['setting'], d['config']['model']): d['config'] for d in originals}
    selected = json.loads((BASELINE / 'selected_configs.json').read_text())
    jobs = []

    def train(kind, setting, model, seed, data='benchmark_data', config=None):
        anchor_setting = setting if setting in CORE else 't1_pair'
        c = dict(config or lookup[anchor_setting, model])
        c.update(setting=setting, seed=seed, split_seed=seed)
        job = dict(id=f'{kind}__{setting}__{model.lower()}__s{seed}', kind=kind, setting=setting, model=model,
                   seed=seed, data=data, data_manifest_sha256=sha256(ROOT / data / 'manifest.json'),
                   config=c, heavy=model in ['RGCN', 'TuckER'], tf32=model in ['RGCN', 'TuckER'])
        jobs.append(job); return job

    # N2 is separate evidence: no original result or selected configuration is overwritten.
    hpo = [json.loads(p.read_text()) for p in (BASELINE / 'hpo').glob('t1_pair_complex*.json')]
    second = sorted(hpo, key=lambda d: d['training']['best_validation_mrr'], reverse=True)[1]['config'].copy()
    second['validation_only'] = False
    train('diagnostic', 't1_pair', 'ComplEx', 42, config=second)
    for setting in CORE:
        for model in MODELS:
            for seed in [123, 456]: train('replicate', setting, model, seed)
    for seed in SEEDS:
        for model in LEARNED:
            train('shuffle', 't1_pair', model, seed, f'supplement_data/shuffle/t1_pair_seed{seed}')
    for model in ['RGCN', 'DistMult']:
        train('shuffle', 't4_a', model, 42, 'supplement_data/shuffle/t4_a_seed42')
    for source in SOURCES:
        for seed in SEEDS:
            for model in MODELS:
                train('loso', 't1_loso_' + source.lower(), model, seed, 'supplement_data/loso')
    for setting in [s for s in CORE if s != 't2_a'] + ['t1_loso_' + s.lower() for s in SOURCES]:
        for seed in SEEDS:
            jobs.append(dict(id=f'ultra__{setting}__s{seed}', kind='ultra', model='ULTRA', setting=setting, seed=seed,
                data='supplement_data/loso' if 'loso' in setting else 'benchmark_data', heavy=True, tf32=False,
                eval_batch=8, timeout_seconds=7200, checkpoint='vendor/ULTRA/ckpts/ultra_50g.pth'))
    sources = {(j['setting'], j['model'], j['seed']): j for j in jobs if j['kind'] in ['replicate', 'ultra']}
    for setting in MATCH_SETTINGS:
        for model in MODELS + ['ULTRA']:
            for seed in SEEDS:
                dependencies = []
                if model == 'ULTRA':
                    parent = sources[setting, model, seed]
                    source_result = (job_output(parent) / 'result.json').relative_to(ROOT).as_posix()
                    dependencies = [parent['id']]
                elif seed == 42:
                    from argparse import Namespace
                    source_result = (BASELINE / 'main' / (run_name(Namespace(**lookup[setting, model])) + '.json')).relative_to(ROOT).as_posix()
                else:
                    parent = sources[setting, model, seed]
                    source_result = training_result(parent); dependencies = [parent['id']]
                jobs.append(dict(id=f'matched__{setting}__{model.lower()}__s{seed}', kind='matched', model=model,
                    setting=setting, seed=seed, source_result=source_result, depends_on=dependencies,
                    heavy=model in ['RGCN', 'TuckER', 'ULTRA'], tf32=model in ['RGCN', 'TuckER'],
                    eval_batch=8, timeout_seconds=7200 if model == 'ULTRA' else None))
    assert len({j['id'] for j in jobs}) == len(jobs)
    for job in jobs:
        if 'data' in job: job['data_manifest_sha256'] = sha256(ROOT / job['data'] / 'manifest.json')
        if job['kind'] == 'matched': job['matching_sha256'] = sha256(ROOT / 'supplement_data/matched' / job['setting'] / f"seed_{job['seed']}.npz")
    plan = dict(status='prepared_not_run', seeds=SEEDS, frozen_seed42_main=99, frozen_hpo=96,
        selected_configs_sha256=sha256(BASELINE / 'selected_configs.json'),
        base_manifest_sha256=sha256(ROOT / 'benchmark_data/manifest.json'),
        supplement_manifest_sha256=sha256(ROOT / 'supplement_data/manifest.json'),
        scope='N1-N7 experimental components; S1-S3 offline; S4 candidate export only; N8/N9 publication/manual work pending; S5-S10 optional work excluded',
        protocol_notes=['No score threshold is used to delete, label as failed, or select a test result.',
            'N1 changes split and initialization seeds together; fixed seed-42 validation-selected hyperparameters.',
            'N3 retains original evaluation truths, candidate masks, auxiliary edges and exact per-relation endpoint degrees.',
            'N4 recorded controls and unknown fallback results are separate; unmatched and degree imbalance rates are reported.',
            'N6 uses the same train-visible eligibility masks; ULTRA capability does not make these splits strictly inductive.',
            'ULTRA OOM/OOT are recorded only after an actual attempt; timeout is 2 hours per ULTRA cell.',
            'Resume skips checksum-verified complete jobs; no epoch-level checkpoint or optimizer-state resume.'],
        counts=dict(Counter(j['kind'] for j in jobs)), jobs=jobs)
    write_json(ROOT / 'supplement_plan.json', plan)
    print(json.dumps({k: v for k, v in plan.items() if k != 'jobs'}, ensure_ascii=False, indent=2))


def verify_inputs():
    for folder in [ROOT / 'benchmark_data', ROOT / 'supplement_data']:
        manifest = json.loads((folder / 'manifest.json').read_text())
        for name, value in manifest['files'].items():
            digest = value['sha256'] if isinstance(value, dict) else value
            assert sha256(folder / name) == digest, str(folder / name)
    plan = json.loads((ROOT / 'supplement_plan.json').read_text())
    assert plan['base_manifest_sha256'] == sha256(ROOT / 'benchmark_data/manifest.json')
    assert plan['supplement_manifest_sha256'] == sha256(ROOT / 'supplement_data/manifest.json')
    assert plan['selected_configs_sha256'] == sha256(BASELINE / 'selected_configs.json')
    audit = json.loads((BASELINE / 'local_checks/download_audit.json').read_text())
    for entry in audit['cells']:
        p = BASELINE / entry['file']
        assert sha256(p) == entry['json_sha256'] and sha256(p.with_suffix('.npz')) == entry['raw_sha256']
    for entry in audit['checkpoints']:
        assert sha256(ROOT / entry['file']) == entry['sha256']
    completed = 0
    for job in plan['jobs']:
        receipt = job_output(job) / 'receipt.json'
        if receipt.exists() and json.loads(receipt.read_text())['status'] == 'complete':
            assert verified_job(job, accept_limits=False), f"Completed supplemental job failed verification: {job['id']}"
            completed += 1
    print(f'Verified frozen inputs, 195 original result pairs, 54 weights, and {completed} completed supplemental jobs; {len(plan["jobs"])-completed} remain', flush=True)


def verified_job(job, accept_limits=True):
    p = job_output(job) / 'receipt.json'
    if not p.exists(): return False
    receipt = json.loads(p.read_text())
    if receipt.get('job') != job or receipt.get('code_sha256') != code_hash(): return False
    if any(not (ROOT / name).exists() or sha256(ROOT / name) != digest for name, digest in receipt.get('inputs', {}).items()): return False
    if accept_limits and job['model'] == 'ULTRA' and receipt['status'] in ['oom', 'oot']: return True
    return receipt['status'] == 'complete' and bool(receipt.get('inputs')) and bool(receipt.get('files')) and all(
        (ROOT / name).exists() and sha256(ROOT / name) == digest for name, digest in receipt['files'].items())


def archive_attempt(folder):
    # The frozen runner has its own JSON/NPZ reuse check. A rejected receipt must
    # not reuse those files when its checkpoint is missing or corrupted.
    existing = [folder / name for name in ['main', 'result.json', 'result.npz', 'weight_diagnostics.json', 'receipt.json'] if (folder / name).exists()]
    if not existing: return
    destination = (folder / 'prior_attempts' / str(time.time_ns())).resolve()
    assert destination.is_relative_to(ROOT.resolve())
    assert all(p.resolve().is_relative_to(ROOT.resolve()) for p in existing)
    destination.mkdir(parents=True)
    for path in existing: path.rename(destination / path.name)


def visible_gpus():
    import torch
    count = torch.cuda.device_count()
    if not count:
        raise RuntimeError('No visible CUDA GPU. Use CPU smoke tests locally; run the formal queue on the server.')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = visible.split(',') if visible is not None else [str(i) for i in range(count)]
    assert len(devices) == count and all(d.strip() for d in devices), devices
    return [d.strip() for d in devices]


def available_cpu_threads():
    count = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    quota = Path('/sys/fs/cgroup/cpu.max')
    if quota.exists():
        limit, period = quota.read_text().split()
        if limit != 'max': count = min(count, max(1, int(limit) // int(period)))
    else:
        limit = Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
        period = Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
        if limit.exists() and period.exists() and int(limit.read_text()) > 0:
            count = min(count, max(1, int(limit.read_text()) // int(period.read_text())))
    return count


def dispatch_jobs(jobs, devices, workers_per_gpu, execute):
    pending = Queue()
    for job in jobs: pending.put(job)

    def worker(gpu):
        while True:
            try: job = pending.get_nowait()
            except Empty: return
            execute(job, gpu)

    with ThreadPoolExecutor(max_workers=len(devices) * workers_per_gpu) as pool:
        futures = [pool.submit(worker, gpu) for gpu in devices for _ in range(workers_per_gpu)]
        for future in futures: future.result()


def launch(args):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.detach:
        command = [sys.executable, '-u', str(Path(__file__).resolve()), 'run', '--workers', str(args.workers)]
        if args.retry_limits: command.append('--retry-limits')
        with (OUTPUT / 'queue.log').open('a', encoding='utf-8') as log:
            kwargs = dict(stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            if os.name == 'nt': kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            else: kwargs['start_new_session'] = True
            process = subprocess.Popen(command, **kwargs)
        print('Detached queue PID', process.pid, 'log', OUTPUT / 'queue.log'); return
    from filelock import FileLock
    with FileLock(str(OUTPUT / 'queue.lock'), timeout=0):
        verify_inputs()
        plan = json.loads((ROOT / 'supplement_plan.json').read_text())
        devices = visible_gpus()
        cpu_threads = available_cpu_threads()
        status = dict(status='running', pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
            total_jobs=len(plan['jobs']), gpu_devices=devices, ordinary_workers_per_gpu=args.workers,
            heavy_workers_per_gpu=1, actual_cpu_threads=cpu_threads)
        write_json(OUTPUT / 'queue_status.json', status)
        jobs_by_id = {j['id']: j for j in plan['jobs']}

        def one(job, gpu):
            if verified_job(job, accept_limits=not args.retry_limits): return
            folder = job_output(job); folder.mkdir(parents=True, exist_ok=True)
            for dep in job.get('depends_on', []):
                receipt = job_output(jobs_by_id[dep]) / 'receipt.json'
                if not verified_job(jobs_by_id[dep], accept_limits=False):
                    write_json(folder / 'receipt.json', dict(status='blocked_dependency', job=job, code_sha256=code_hash(), dependency=dep))
                    return
            env = os.environ.copy()
            slots = len(devices) * (1 if job['heavy'] else args.workers)
            threads = str(max(1, min(4, cpu_threads // slots)))
            env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
                MICROBE_TORCH_THREADS=threads, OMP_WAIT_POLICY='PASSIVE',
                CUDA_VISIBLE_DEVICES=gpu,
                TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='1' if job['tf32'] else '0',
                PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', MAX_JOBS='4')
            command = [sys.executable, '-u', str(ROOT / 'benchmark/supplement_cell.py'), '--job-id', job['id']]
            print('START', job['id'], 'GPU', gpu, 'CPU_THREADS', threads, flush=True)
            started = time.time()
            archive_attempt(folder)
            write_json(folder / 'receipt.json', dict(status='running', job=job, code_sha256=code_hash(),
                started_utc=datetime.now(timezone.utc).isoformat(), gpu_device=gpu))
            with (folder / 'run.log').open('a', encoding='utf-8') as log:
                try:
                    rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                        cwd=ROOT, timeout=job.get('timeout_seconds')).returncode
                except subprocess.TimeoutExpired:
                    write_json(folder / 'receipt.json', dict(status='oot', job=job, code_sha256=code_hash(),
                        timeout_seconds=job['timeout_seconds'], elapsed_sec=time.time()-started))
                    rc = 124
            receipt = folder / 'receipt.json'
            if not receipt.exists() or json.loads(receipt.read_text()).get('status') == 'running':
                write_json(receipt, dict(status='failed', job=job, code_sha256=code_hash(), returncode=rc, elapsed_sec=time.time()-started))
            print('END', job['id'], json.loads(receipt.read_text())['status'], flush=True)

        # Independent jobs use separate GPUs; large models remain exclusive per GPU.
        for kinds in [{'diagnostic'}, {'replicate', 'shuffle', 'loso'}, {'ultra'}, {'matched'}]:
            group = [j for j in plan['jobs'] if j['kind'] in kinds]
            for heavy, workers in [(False, args.workers), (True, 1)]:
                dispatch_jobs([j for j in group if j['heavy'] == heavy], devices, workers, one)
        counts = Counter()
        for job in plan['jobs']:
            receipt = job_output(job) / 'receipt.json'
            counts[json.loads(receipt.read_text())['status'] if receipt.exists() else 'missing'] += 1
        status.update(status='complete' if counts['complete'] == len(plan['jobs']) else 'finished_with_exceptions',
                      counts=dict(counts), finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(OUTPUT / 'queue_status.json', status)
        subprocess.run([sys.executable, str(ROOT / 'benchmark/supplement_offline.py')], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ROOT / 'benchmark/supplement_report.py')], cwd=ROOT, check=True)
        print(json.dumps(status, indent=2), flush=True)


def setup():
    # Keep a working PyTorch installation. Install only missing dependencies.
    wanted = {'pykeen': 'pykeen==1.11.1', 'pandas': 'pandas', 'scipy': 'scipy', 'sklearn': 'scikit-learn',
              'filelock': 'filelock', 'matplotlib': 'matplotlib', 'torch_geometric': 'torch-geometric', 'ninja': 'ninja'}
    missing = [package for module, package in wanted.items() if importlib.util.find_spec(module) is None]
    if missing: subprocess.run([sys.executable, '-m', 'pip', 'install', *missing], check=True)
    import torch
    if importlib.util.find_spec('torch_scatter') is None:
        version = '.'.join(torch.__version__.split('+')[0].split('.')[:2]) + '.0'
        cuda = 'cpu' if torch.version.cuda is None else 'cu' + torch.version.cuda.replace('.', '')
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch-scatter', '-f',
                        f'https://data.pyg.org/whl/torch-{version}+{cuda}.html'], check=True)
    from ultra_adapter import UltraScorer
    from core import load_case
    torch.set_num_threads(4)
    case = load_case(ROOT / 'benchmark_data', 't1_pair', 42, smoke=True)
    scorer = UltraScorer(case, device='cuda', reference=False, batch=1)
    scores = scorer.triples(case['train'][:2])
    assert len(scores) == 2 and all(math.isfinite(float(x)) for x in scores)
    write_json(OUTPUT / 'native_ultra_preflight.json', dict(status='passed', checkpoint_sha256=sha256(scorer.checkpoint)))
    print('Native ULTRA GPU preflight passed; no full experiment run during setup')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['plan', 'verify', 'run', 'setup', 'report'])
    ap.add_argument('--workers', type=int, default=2, help='Concurrent ordinary jobs per visible GPU; heavy jobs always use one slot per GPU')
    ap.add_argument('--detach', action='store_true')
    ap.add_argument('--retry-limits', action='store_true')
    args = ap.parse_args()
    if args.action == 'plan': make_plan()
    elif args.action == 'verify': verify_inputs()
    elif args.action == 'run':
        assert 1 <= args.workers <= 3
        launch(args)
    elif args.action == 'setup': setup()
    else:
        subprocess.run([sys.executable, str(ROOT / 'benchmark/supplement_offline.py')], check=True)
        subprocess.run([sys.executable, str(ROOT / 'benchmark/supplement_report.py')], check=True)


if __name__ == '__main__':
    main()
