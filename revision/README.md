# Revision configurations and execution

`plan.json` is executable configuration for 147 jobs: 14 validation searches, 42 training jobs, 10 permutations, 39 frozen-foundation-model evaluations and 42 matched-control evaluations. `PROTOCOL.md` defines these settings.

The launcher uses external inputs at `benchmark_data/`, `supplement_data/` and `revision_data/`, pretrained assets under `vendor/`, and historical reference runs under `autodl/results/`. New output goes to `revision_results/`. None of these inputs or outputs is committed to this repository.

After assembling the matching external inputs and training environment:

```bash
python revision/launcher.py verify
python revision/launcher.py run --workers 1
python revision/launcher.py report
```

These are full research executions. For a CPU baseline using one externally supplied split directory, use `tools/evaluate_cpu.py` instead. `revision/test_revision.py` includes data/checkpoint-dependent tests; the root `tests/` suite uses generated fixtures only.

`frozen_baseline/` preserves the reference implementation used in historical comparisons. Missing data, selections or checkpoints must be supplied with their matching release; an unavailable input is not a reason to substitute another graph version.
