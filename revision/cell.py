"""Run one revision job without writing into historical result directories."""
import argparse
import shutil
import time
import numpy as np

from common import ROOT, PLAN, OUT, read, write_json, sha256, code_hash, folder, resolve, inputs
import core
import run as base
import supplement_cell as old


def training(job, device, smoke):
    args = base.parser().parse_args(['--setting', 't1_pair', '--model', job['model']])
    for k, v in job['config'].items(): setattr(args, k, v)
    dest = folder(job, smoke)
    args.data = ROOT / job['data']; args.out = dest; args.device = device; args.smoke = smoke
    args.revision_code_sha256 = code_hash()
    if job['kind'] == 'shuffle':
        original = core.load_case(ROOT / 'benchmark_data', job['setting'], job['seed'], smoke=smoke)
        original_maps = core.known_maps
        core.known_maps = lambda case, validation=False: original_maps(original, validation)
    result = base.run(args)
    stage = 'smoke' if smoke else 'hpo' if args.validation_only else 'main'
    source = dest / stage / (base.run_name(args)+'.json')
    paths = []
    for suffix in ['.json', '.npz', '.pt']:
        p = source.with_suffix(suffix)
        if p.exists():
            target = dest / ('result'+suffix)
            shutil.copy2(p, target); paths.append(target)
    return paths


def foundation(job, device, smoke):
    import torch
    torch.set_num_threads(4)
    from ultra_adapter import UltraScorer
    from kgicl_adapter import KGICLScorer
    data = ROOT / job['data']; dest = folder(job, smoke)
    case = core.load_case(data, job['setting'], job['seed'], smoke=smoke)
    batch = 1 if smoke else job['eval_batch']
    scorer = (KGICLScorer(case, device=device, reference=smoke, batch=batch, seed=job['seed'])
              if job['model']=='KG-ICL' else UltraScorer(case, device=device, reference=smoke, batch=batch))
    raw, ranking = {}, None
    if job['setting'] != 't4_b':
        raw = core.evaluate_ranks(case, scorer, batch=batch)
        ranking = core.ranking_metrics(raw, job['seed'], 20 if smoke else 1000)
    args = argparse.Namespace(seed=job['seed'], bootstrap=20 if smoke else 1000)
    discrimination, scores = base.probes(case, scorer, args); raw.update(scores)
    np.savez_compressed(dest/'result.npz', **raw)
    write_json(dest/'result.json', dict(status='smoke_only' if smoke else 'complete',
        config=dict(model=job['model'], setting=job['setting'], seed=job['seed']),
        ranking=ranking, discrimination=discrimination, counts=case['meta']['audit'],
        training='frozen official pretrained checkpoint; zero target fine-tuning',
        data_manifest_sha256=sha256(data/'manifest.json'), checkpoint_sha256=sha256(scorer.checkpoint),
        raw_sha256=sha256(dest/'result.npz'),
        prompt_supports=getattr(scorer, 'prompt_supports', None),
        protocol_notes='Train-only graph and KG-ICL prompts; common typed candidates, eligibility and realistic ties; no target distance prior',
        backend='reference smoke' if smoke else 'official KG-ICL non-fused' if job['model']=='KG-ICL' else 'official ULTRA fused'))
    return [dest/'result.json', dest/'result.npz']


def matched(job, device, smoke):
    old.OUTPUT = OUT
    original = old.restored_scorer
    def scorer(case, path, device):
        if job['model'] == 'KG-ICL':
            from kgicl_adapter import KGICLScorer
            return KGICLScorer(case, device=device, reference=smoke, batch=1, seed=job['seed'])
        return original(case, path, device)
    old.restored_scorer = scorer
    return old.run_matched(job, device, smoke)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job-id', required=True); ap.add_argument('--device', default='cuda')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    job = resolve(next(j for j in read(PLAN)['jobs'] if j['id']==args.job_id))
    dest = folder(job, args.smoke); dest.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        paths = (matched(job, args.device, args.smoke) if job['kind']=='matched' else
                 foundation(job, args.device, args.smoke) if job['kind']=='foundation' else
                 training(job, args.device, args.smoke))
        receipt = dict(status='smoke_only' if args.smoke else 'complete', job=job,
                       code_sha256=code_hash(), inputs=inputs(job), elapsed_sec=time.time()-started,
                       files={p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})
    except Exception as exc:
        import torch
        write_json(dest/'receipt.json', dict(status='oom' if isinstance(exc, torch.cuda.OutOfMemoryError) else 'failed',
                   job=job, error=repr(exc), code_sha256=code_hash(), elapsed_sec=time.time()-started))
        raise
    write_json(dest/'receipt.json', receipt)


if __name__ == '__main__':
    main()
