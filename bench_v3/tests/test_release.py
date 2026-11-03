"""Offline release tests: rebuild the degree prior and L3 on the released splits and compare MRR, Hits@10 and
study-macro MRR with the reference values of the paper's frozen evaluator (tests/reference_metrics.json).

By default every setting is checked on seed 42; set MKB_ALL_SEEDS=1 to check all five seeds.
Run from the release root:  python -m pytest -q
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from microbekg_bench import Evaluator, filtered_rank, load
from microbekg_bench.baselines import L3, DegreePrior

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MKB_DATA_ROOT", str(PACKAGE_ROOT)))
HAS_DATA = (ROOT / "benchmark/entities.tsv").is_file()
REF = json.loads((PACKAGE_ROOT / "tests" / "reference_metrics.json").read_text())
SEEDS = None if os.environ.get("MKB_ALL_SEEDS") else {42}
CASES = sorted(k for k in REF if SEEDS is None or int(k.split("|")[2]) in SEEDS)
SCORERS = {"DegreePrior": DegreePrior, "L3": L3}


def test_filtered_rank_ties_half_credit():
    # target 0.5; one higher score, two ties, one known (filtered) higher score
    assert filtered_rank([0.5, 0.9, 0.5, 0.5, 1.0], 0, [4]) == 1 + 1 + 0.5 * 2


@pytest.mark.skipif(not HAS_DATA, reason="Set MKB_DATA_ROOT to the extracted v3 data directory")
@pytest.mark.parametrize("key", CASES)
def test_reference_metrics(key):
    setting, model, seed = key.split("|")
    case = load(setting, int(seed), ROOT)
    res = Evaluator(case).evaluate(SCORERS[model](case.train, case.n_entities))
    m, ref = res["metrics"], REF[key]
    assert m["n_ranks"] == 2 * ref["n_triples"]
    assert m["mrr"] == pytest.approx(ref["mrr"], abs=1e-9)
    assert m["hits_at_10"] == pytest.approx(ref["hits_at_10"], abs=1e-9)
    assert m["macro_mrr"] == pytest.approx(ref["study_macro_mrr"], abs=1e-9)


@pytest.mark.skipif(not HAS_DATA, reason="Set MKB_DATA_ROOT to the extracted v3 data directory")
def test_variant_rows_match_parent():
    """v3 training-graph variants keep the parent's test rows."""
    for var, par in [("t1_v3_strict_nocooc", "t1_study_strict"), ("t1_v3_temporal_nocooc", "t1_temporal")]:
        a, b = load(var, 42, ROOT), load(par, 42, ROOT)
        assert np.array_equal(a.test, b.test)
        assert len(a.train) <= len(b.train)
