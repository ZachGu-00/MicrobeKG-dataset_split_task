"""Small protocol checks and explicitly tiny, real-data execution tests."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory

import numpy as np

from core import load_case, known_maps
from prepare import ROOT, CORE, SEEDS, pair_code, sha256, write_json
from supplement_prepare import degree_preserving_shuffle, SOURCES, MATCH_SETTINGS


class ProtocolTests(unittest.TestCase):
    def test_multigpu_dispatch_has_unique_jobs_and_bounded_slots(self):
        from supplement import dispatch_jobs
        for workers in [1, 2]:
            active = {'0': 0, '1': 0}
            peak = dict(active)
            observed = []
            lock = threading.Lock()
            barrier = threading.Barrier(2 * workers, timeout=5)

            def execute(job, gpu):
                with lock:
                    active[gpu] += 1
                    peak[gpu] = max(peak[gpu], active[gpu])
                    observed.append((job, gpu))
                if job < 2 * workers: barrier.wait()
                time.sleep(.005)
                with lock: active[gpu] -= 1

            dispatch_jobs(range(12), ['0', '1'], workers, execute)
            self.assertEqual(sorted(job for job, _ in observed), list(range(12)))
            self.assertEqual(peak, {'0': workers, '1': workers})
            self.assertEqual(active, {'0': 0, '1': 0})

    def test_gpu_binding_respects_inherited_visible_devices(self):
        from supplement import visible_gpus
        with patch('torch.cuda.device_count', return_value=2):
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3,7'}):
                self.assertEqual(visible_gpus(), ['3', '7'])
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-first,GPU-second'}):
                self.assertEqual(visible_gpus(), ['GPU-first', 'GPU-second'])
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(visible_gpus(), ['0', '1'])
        with patch('torch.cuda.device_count', return_value=0):
            with self.assertRaisesRegex(RuntimeError, 'No visible CUDA GPU'):
                visible_gpus()

    def test_ultra_grouped_scoring_matches_direct_inference(self):
        import torch
        from ultra_adapter import UltraScorer
        torch.set_num_threads(1)
        case = load_case(ROOT / 'benchmark_data', 't1_pair', 42, smoke=True)
        scorer = UltraScorer(case, device='cpu', reference=True, batch=2)
        rows = case['train'][:7].copy()
        rows[1:4, :2] = rows[0, :2]
        rows = rows[[3, 5, 0, 6, 1, 4, 2]]
        mapped = np.column_stack([scorer.mapping[rows[:, 0]], scorer.mapping[rows[:, 2]], rows[:, 1]])[:, None, :]
        with torch.inference_mode():
            direct = np.concatenate([scorer.model(scorer.graph, torch.as_tensor(mapped[i:i+2])).numpy().ravel() for i in range(0, len(rows), 2)])
        np.testing.assert_allclose(scorer.triples(rows), direct, rtol=1e-5, atol=1e-5)

    def test_resume_rejects_changed_inputs_outputs_and_smoke(self):
        import supplement
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'input'; source.write_text('frozen')
            output = root / 'output'; output.write_text('observed')
            job = dict(id='test', model='DistMult')
            receipt = dict(status='complete', job=job, code_sha256='fixed',
                           inputs={'input': sha256(source)}, files={'output': sha256(output)})
            def save(): write_json(root / 'receipt.json', receipt)
            with patch.object(supplement, 'ROOT', root), patch.object(supplement, 'job_output', return_value=root), patch.object(supplement, 'code_hash', return_value='fixed'):
                save(); self.assertTrue(supplement.verified_job(job))
                source.write_text('changed'); self.assertFalse(supplement.verified_job(job))
                source.write_text('frozen'); output.write_text('broken'); self.assertFalse(supplement.verified_job(job))
                output.write_text('observed'); self.assertFalse(supplement.verified_job(dict(job, seed=123)))
                receipt['status'] = 'smoke_only'; save(); self.assertFalse(supplement.verified_job(job))
                receipt.update(status='oom', job=dict(job, model='ULTRA')); save()
                self.assertTrue(supplement.verified_job(receipt['job']))
                self.assertFalse(supplement.verified_job(receipt['job'], accept_limits=False))
                main = root / 'main'; main.mkdir()
                (main / 'model.json').write_text('old result without weights')
                old_digest = sha256(main / 'model.json')
                supplement.archive_attempt(root)
                self.assertFalse(main.exists())
                self.assertFalse((root / 'receipt.json').exists())
                archived = list((root / 'prior_attempts').glob('*/main/model.json'))
                self.assertEqual(len(archived), 1)
                self.assertEqual(sha256(archived[0]), old_digest)

    def test_shuffle_preserves_degrees_and_forbidden_pairs(self):
        train = np.array([[h, 0, t] for h, t in [(0, 4), (0, 5), (1, 5), (1, 6), (2, 6), (2, 7), (3, 7), (3, 4)]] + [[0, 1, 1]])
        forbidden = {0*8+6}
        out, _ = degree_preserving_shuffle(train, [0], forbidden, 8, 42)
        np.testing.assert_array_equal(out[-1], train[-1])
        for col in [0, 2]:
            np.testing.assert_array_equal(np.bincount(out[:-1, col], minlength=8), np.bincount(train[:-1, col], minlength=8))
        self.assertEqual(len(np.unique(out, axis=0)), len(out))
        self.assertFalse(np.isin(pair_code(out, 8, True), list(forbidden)).any())

    def test_reference_rspmm_matches_official_row_column_convention(self):
        import torch
        from ultra_adapter import reference_rspmm
        edge = torch.tensor([[0, 0, 2], [1, 2, 0]])
        rel = torch.tensor([[2., 3.], [4., 5.]])
        inp = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        out = reference_rspmm(edge, torch.tensor([0, 1, 0]), torch.ones(3), rel, inp)
        expected = torch.stack([inp[1]*rel[0]+inp[2]*rel[1], torch.zeros(2), inp[0]*rel[0]])
        torch.testing.assert_close(out, expected)


def full_data_checks():
    records = []
    source_index = json.loads((ROOT / 'supplement_data/t1_pair_sources.json').read_text())['pair_sources']
    for source in SOURCES:
        for seed in SEEDS:
            case = load_case(ROOT / 'supplement_data/loso', 't1_loso_' + source.lower(), seed)
            a, x = case['arrays'], case['triples']; n = len(case['seen'])
            train_pairs = set(pair_code(case['train'], n, True))
            assert not train_pairs.intersection(pair_code(x[np.r_[a['valid'], a['test']]], n, True))
            assert all(source_index[str(int(p))] == [source] for p in pair_code(x[a['test']], n))
            shared = {int(p) for p, labels in source_index.items() if source in labels and len(labels) > 1}
            assert shared <= set(pair_code(case['train'], n))
            records.append(dict(kind='loso', setting=case['meta']['setting'], seed=seed, status='passed'))
    for setting, seeds in [('t1_pair', SEEDS), ('t4_a', [42])]:
        for seed in seeds:
            original = load_case(ROOT / 'benchmark_data', setting, seed)
            case = load_case(ROOT / f'supplement_data/shuffle/{setting}_seed{seed}', setting, seed)
            target = [case['relations'].index(r) for r in case['meta']['scoring_relations']]
            for rel in target:
                old, new = original['train'][original['train'][:, 1] == rel], case['train'][case['train'][:, 1] == rel]
                for col in [0, 2]:
                    np.testing.assert_array_equal(np.bincount(old[:, col], minlength=len(case['seen'])), np.bincount(new[:, col], minlength=len(case['seen'])))
            auxiliary = ~np.isin(original['train'][:, 1], target)
            np.testing.assert_array_equal(original['train'][auxiliary], case['train'][auxiliary])
            for key in ['valid', 'test', 'negatives', 'conflict', 'candidate_eligible', 'test_eligible', 'valid_eligible', 'negative_eligible']:
                np.testing.assert_array_equal(original['arrays'][key], case['arrays'][key])
            held = np.r_[case['arrays']['valid'], case['arrays']['test'], case['arrays']['negatives'], case['arrays']['conflict']]
            assert not np.isin(pair_code(case['train'], len(case['seen']), True), pair_code(case['triples'][held], len(case['seen']), True)).any()
            assert all(s['changed_fraction'] > .9 for s in case['meta']['perturbation']['per_relation'])
            records.append(dict(kind='shuffle', setting=setting, seed=seed, status='passed'))
    for setting in MATCH_SETTINGS:
        for seed in SEEDS:
            case = load_case(ROOT / 'benchmark_data', setting, seed); n = len(case['seen'])
            with np.load(ROOT / 'supplement_data/matched' / setting / f'seed_{seed}.npz') as a:
                pi, ni = a['positive_indices'], a['negative_indices']
                assert len(pi) == len(np.unique(pi)) and len(ni) == len(np.unique(ni))
                assert not np.intersect1d(pi, a['fallback_positive_indices']).size
                deg = np.bincount(case['train'][:, [0, 2]].ravel(), minlength=n)
                buckets = np.floor(np.log2(np.maximum(deg, 1)))
                pos = case['triples'][a['original_positive_ids'][pi]]
                neg = case['triples'][a['original_negative_ids'][ni]]
                np.testing.assert_array_equal(buckets[pos[:, [0, 2]]], buckets[neg[:, [0, 2]]])
                fp = case['triples'][a['original_positive_ids'][a['fallback_positive_indices']]]
                fn = a['fallback_triples']
                np.testing.assert_array_equal(fp[:, 0], fn[:, 0])
                np.testing.assert_array_equal(buckets[fp[:, 2]], buckets[fn[:, 2]])
                assert not np.isin(pair_code(fn, n, True), pair_code(case['triples'], n, True)).any()
            records.append(dict(kind='matched', setting=setting, seed=seed, status='passed'))
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    tests = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolTests))
    assert tests.wasSuccessful()
    checked = full_data_checks()
    smoke = []
    if args.smoke:
        jobs = ['diagnostic__t1_pair__complex__s42', 'replicate__t1_pair__rotate__s123',
                'shuffle__t1_pair__transe__s42', 'shuffle__t1_pair__tucker__s42',
                'shuffle__t4_a__rgcn__s42', 'loso__t1_loso_hmdad__distmult__s456',
                'ultra__t1_pair__s42', 'matched__t4_a__distmult__s42', 'matched__t1_pair__popularity__s42']
        for job in jobs:
            subprocess.run([sys.executable, str(ROOT / 'benchmark/supplement_cell.py'), '--job-id', job, '--device', 'cpu', '--smoke'], check=True)
            smoke.append(job)
    write_json(ROOT / 'autodl/supplement_results/checks.json', dict(status='passed', unit_tests=tests.testsRun,
        full_data_checks=checked, smoke_jobs=smoke, native_cuda_ultra_tested=False,
        baseline_manifest_sha256=sha256(ROOT / 'benchmark_data/manifest.json'),
        supplement_manifest_sha256=sha256(ROOT / 'supplement_data/manifest.json'),
        bounds='Training smoke uses at most 512 real triples, 2 epochs and 4 test triples; pretrained fallback scoring is limited to 4 pairs'))
    print('Passed', len(checked), 'full-data invariants and', len(smoke), 'tiny jobs')


if __name__ == '__main__':
    main()
