"""One supplemental job. The frozen run.py/core.py and seed-42 files stay intact."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import core
import run as baseline_run
from core import LEARNED, LearnedScorer, LightScorer, discrimination_metrics, evaluate_ranks, ranking_metrics, scored_triples
from prepare import ROOT, sha256, write_json

BASELINE = ROOT / 'autodl/results'
OUTPUT = ROOT / 'autodl/supplement_results'


def code_hash():
    files = [Path(__file__).with_name(p) for p in ['supplement_cell.py', 'supplement_prepare.py',
             'run.py', 'core.py', 'prepare.py', 'ultra_adapter.py']]
    files += sorted(p for p in (ROOT / 'vendor/ULTRA/ultra').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files += [ROOT / 'vendor/ULTRA/UPSTREAM.json', ROOT / 'vendor/ULTRA/ckpts/ultra_50g.pth']
    return hashlib.sha256(''.join(sha256(p) for p in files).encode()).hexdigest()


def job_output(job, smoke=False):
    return OUTPUT / ('smoke' if smoke else 'jobs') / job['id']


def source_path(job):
    return ROOT / job['source_result']


def input_fingerprints(job):
    paths = [ROOT / job['data'] / 'manifest.json'] if 'data' in job else []
    if job['kind'] == 'matched':
        paths += [source_path(job), source_path(job).with_suffix('.npz'),
                  ROOT / 'benchmark_data/manifest.json',
                  ROOT / 'supplement_data/matched' / job['setting'] / f"seed_{job['seed']}.npz"]
        if job['model'] in LEARNED: paths.append(source_path(job).with_suffix('.pt'))
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def restored_scorer(case, result_path, device):
    import torch
    from pykeen.models import model_resolver
    from pykeen.triples import CoreTriplesFactory
    torch.set_num_threads(int(os.environ.get('MICROBE_TORCH_THREADS', '4')))
    saved = json.loads(result_path.read_text())
    c = saved['config']
    if c['model'] not in LEARNED:
        return LightScorer(case['train'], len(case['seen']), c['model'], c['seed'])
    checkpoint = torch.load(result_path.with_suffix('.pt'), map_location='cpu', weights_only=True, mmap=True)
    assert checkpoint['config'] == c
    assert checkpoint['global_entity_indices'] == np.flatnonzero(case['seen']).tolist()
    assert checkpoint['relations'] == case['relations']
    mapping = np.full(len(case['seen']), -1, np.int32)
    mapping[case['seen']] = np.arange(case['seen'].sum())
    triples = case['train'].copy(); triples[:, 0] = mapping[triples[:, 0]]; triples[:, 2] = mapping[triples[:, 2]]
    factory = CoreTriplesFactory(torch.as_tensor(triples, dtype=torch.long),
        num_entities=int(case['seen'].sum()), num_relations=len(case['relations']))
    kw = dict(triples_factory=factory, embedding_dim=c['dim'], random_seed=c['seed'])
    if c['loss'] == 'nssa': kw.update(loss='NSSA', loss_kwargs=dict(margin=9.0, adversarial_temperature=1.0))
    model = model_resolver.make(c['model'], kw).to(device)
    baseline_run.restore_best_state(model, checkpoint['state_dict'])
    model.eval()
    return LearnedScorer(model, mapping)


def run_matched(job, device, smoke):
    folder = job_output(job, smoke); folder.mkdir(parents=True, exist_ok=True)
    source = source_path(job)
    saved = json.loads(source.read_text())
    assert saved['status'] == 'complete' and saved['raw_sha256'] == sha256(source.with_suffix('.npz'))
    setting, seed = job['setting'], job['seed']
    match_file = ROOT / 'supplement_data/matched' / setting / f'seed_{seed}.npz'
    meta = json.loads(match_file.with_suffix('.json').read_text())
    with np.load(match_file) as z: match = dict(z)
    with np.load(source.with_suffix('.npz')) as z: raw = dict(z)
    label = meta['label']
    for side in ['positive', 'negative']:
        key = 'pos' if side == 'positive' else 'neg'
        assert np.array_equal(raw[label + '_' + key + '_ids'], match['original_' + side + '_ids'])
    positive, negative = raw[label + '_pos_scores'], raw[label + '_neg_scores']
    pi, ni = match['positive_indices'], match['negative_indices']
    fi, fallback = match['fallback_positive_indices'], match['fallback_triples']
    if smoke: pi, ni, fi, fallback = pi[:4], ni[:4], fi[:4], fallback[:4]
    n_boot = 20 if smoke else 1000
    metrics = dict(original=saved['discrimination'][label],
        recorded_degree_matched=discrimination_metrics(positive[pi], negative[ni], seed, n_boot),
        original_controls_on_matched_positive_subset=discrimination_metrics(positive[pi], negative, seed, n_boot))
    result_raw = dict(matched_positive_scores=positive[pi], matched_negative_scores=negative[ni],
        matched_positive_ids=match['original_positive_ids'][pi], matched_negative_ids=match['original_negative_ids'][ni])
    if len(fi):
        case = core.load_case(ROOT / 'benchmark_data', setting, seed)
        if job['model'] == 'ULTRA':
            from ultra_adapter import UltraScorer
            scorer = UltraScorer(case, device=device, reference=smoke, batch=job.get('eval_batch', 8))
        else:
            scorer = restored_scorer(case, source, device)
        if setting.startswith('t1'):
            scores = []
            for name in case['meta']['scoring_relations']:
                rows = fallback.copy(); rows[:, 1] = case['relations'].index(name)
                scores.append(scorer.triples(rows))
            fallback_scores = np.max(scores, axis=0)
        else:
            fallback_scores = scorer.triples(scored_triples(case, fallback))
        metrics['unknown_fallback_only'] = discrimination_metrics(positive[fi], fallback_scores, seed, n_boot)
        result_raw.update(fallback_positive_scores=positive[fi], fallback_negative_scores=fallback_scores,
                          fallback_positive_ids=match['original_positive_ids'][fi], fallback_negative_triples=fallback)
    np.savez_compressed(folder / 'result.npz', **result_raw)
    output = dict(status='smoke_only' if smoke else 'complete', setting=setting, model=job['model'], seed=seed,
        protocol='degree-matched-v1', matching=meta, metrics=metrics,
        source_json_sha256=sha256(source), source_raw_sha256=saved['raw_sha256'], matching_sha256=sha256(match_file),
        raw_sha256=sha256(folder / 'result.npz'),
        interpretation='Recorded controls and unknown fallbacks are separate estimands; do not merge them into a conflict/therapy headline')
    write_json(folder / 'result.json', output)
    return [folder / 'result.json', folder / 'result.npz']


def run_ultra(job, device, smoke):
    import torch
    from ultra_adapter import UltraScorer
    torch.set_num_threads(int(os.environ.get('MICROBE_TORCH_THREADS', '4')))
    folder = job_output(job, smoke); folder.mkdir(parents=True, exist_ok=True)
    data = ROOT / job['data']
    case = core.load_case(data, job['setting'], job['seed'], smoke=smoke)
    scorer = UltraScorer(case, device=device, reference=smoke, batch=2 if smoke else job['eval_batch'])
    raw, ranking = {}, None
    if job['setting'] != 't4_b':
        raw = evaluate_ranks(case, scorer, batch=2 if smoke else job['eval_batch'])
        ranking = ranking_metrics(raw, job['seed'], 20 if smoke else 1000)
    args = baseline_run.parser().parse_args(['--setting', 't1_pair', '--model', 'TransE'])
    args.seed = job['seed']; args.bootstrap = 20 if smoke else 1000
    discrimination, scores = baseline_run.probes(case, scorer, args)
    raw.update(scores)
    np.savez_compressed(folder / 'result.npz', **raw)
    write_json(folder / 'result.json', dict(status='smoke_only' if smoke else 'complete', config=dict(model='ULTRA',
        setting=job['setting'], seed=job['seed']), ranking=ranking, discrimination=discrimination,
        counts=case['meta']['audit'], raw_sha256=sha256(folder / 'result.npz'), data_manifest_sha256=sha256(data / 'manifest.json'),
        checkpoint_sha256=sha256(scorer.checkpoint), training='zero-shot; no target fine-tuning',
        inference_backend='CPU reference sum/DistMult kernel' if smoke else 'official fused rspmm',
        eligibility='Same training-visible typed candidates and test mask as all other baselines; not a strict unseen-entity benchmark',
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None))
    return [folder / 'result.json', folder / 'result.npz']


def run_training(job, device, smoke):
    args = baseline_run.parser().parse_args(['--setting', 't1_pair', '--model', job['model']])
    for k, v in job['config'].items(): setattr(args, k, v)
    args.data = ROOT / job['data']; args.out = job_output(job, smoke)
    args.device = device; args.smoke = smoke; args.supplement_code_sha256 = code_hash()
    # Only this child process changes the filtering function. Frozen source files stay identical.
    if job['kind'] == 'shuffle':
        original = core.load_case(ROOT / 'benchmark_data', job['setting'], job['seed'], smoke=smoke)
        original_known_maps = core.known_maps
        core.known_maps = lambda case, validation=False: original_known_maps(original, validation)
    result = baseline_run.run(args)
    folder = args.out / ('smoke' if smoke else 'main')
    path = folder / (baseline_run.run_name(args) + '.json')
    paths = [path, path.with_suffix('.npz')]
    if path.with_suffix('.pt').exists(): paths.append(path.with_suffix('.pt'))
    if job['kind'] == 'diagnostic' and not smoke:
        import torch
        state = torch.load(path.with_suffix('.pt'), weights_only=True, map_location='cpu')['state_dict']
        write_json(args.out / 'weight_diagnostics.json', {k: dict(shape=list(v.shape), finite=bool(torch.isfinite(v).all()),
            rms=float(v.abs().float().square().mean().sqrt()), max_abs=float(v.abs().max())) for k, v in state.items() if v.numel()})
        paths.append(args.out / 'weight_diagnostics.json')
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--job-id', required=True)
    ap.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    plan = json.loads((ROOT / 'supplement_plan.json').read_text())
    job = next(j for j in plan['jobs'] if j['id'] == args.job_id)
    started = time.time(); folder = job_output(job, args.smoke); folder.mkdir(parents=True, exist_ok=True)
    try:
        if job['kind'] == 'matched': paths = run_matched(job, args.device, args.smoke)
        elif job['kind'] == 'ultra': paths = run_ultra(job, args.device, args.smoke)
        else: paths = run_training(job, args.device, args.smoke)
        receipt = dict(status='smoke_only' if args.smoke else 'complete', job=job, code_sha256=code_hash(),
            elapsed_sec=time.time()-started, inputs=input_fingerprints(job), files={p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})
    except Exception as exc:
        import torch
        oom = isinstance(exc, torch.cuda.OutOfMemoryError)
        receipt = dict(status='oom' if oom and job['model'] == 'ULTRA' else 'failed', job=job,
            code_sha256=code_hash(), error=repr(exc), elapsed_sec=time.time()-started)
        write_json(folder / 'receipt.json', receipt)
        raise
    write_json(folder / 'receipt.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ['job', 'files']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
