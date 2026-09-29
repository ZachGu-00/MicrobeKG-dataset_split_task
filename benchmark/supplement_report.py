"""Generate the single final report from verified completed outputs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import pandas as pd
from prepare import ROOT, CORE
from core import MODELS

ARCHIVE = 'archive/2026-09-17_complete_import'
ANALYSIS = ROOT / 'autodl/supplement_results/analysis'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---']*len(headers)) + '|'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def fmt(values):
    return f'{values.mean():.4f} ± {values.std(ddof=1):.4f}'


def main():
    plan = read(ROOT / 'supplement_plan.json')
    audit = read(ROOT / 'autodl/supplement_results/local_checks/final_audit.json')
    queue = read(ROOT / 'autodl/supplement_results/queue_status.json')
    assert audit['status'] == 'passed' and queue['counts'] == {'complete': 567}
    for job in plan['jobs']:
        receipt = read(ROOT / 'autodl/supplement_results/jobs' / job['id'] / 'receipt.json')
        assert receipt['status'] == 'complete' and receipt['job'] == job
    observed = pd.read_csv(ANALYSIS / 'all_observed_results.csv')
    summary = pd.read_csv(ANALYSIS / 'main_mean_std.csv')
    matched = pd.read_csv(ANALYSIS / 'matched_all_observed.csv')
    shuffle = pd.read_csv(ANALYSIS / 'shuffle_comparisons.csv')
    diag = read(ANALYSIS / 'complex_diagnostic.json')
    assert len(observed) == 450 and len(matched.groupby(['setting', 'model', 'seed'])) == 216
    assert (summary['count'] == 3).all()
    core = observed[observed.group.isin(['baseline', 'replicate'])]
    assert len(core.groupby(['model', 'setting'])) == 99 and len(core) == 297
    cold = core[(core.setting == 't1_cold_microbe') & (core.model == 'TuckER')]
    main_tables = []
    for settings, title in [(CORE, '主设置：三 seed 汇总'),
                            (sorted(s for s in summary.setting.unique() if 'loso' in s), 'LOSO：三 seed 汇总')]:
        rows = []
        for model in MODELS + ['ULTRA']:
            values = []
            for setting in settings:
                part = summary[(summary.model == model) & (summary.setting == setting)]
                if part.empty:
                    assert model == 'ULTRA' and setting == 't2_a'
                    values.append('未纳入协议')
                else:
                    r = part.iloc[0]
                    values.append(f"{r['mean']:.4f} ± {r['std']:.4f}")
            rows.append([model] + values)
        main_tables.append('## ' + title + '\n\n' + table(['模型'] + settings, rows))
    bridge = core[core.setting == 't1_bridges'][['model', 'seed', 'mrr']].merge(
        core[core.setting == 't1_pair'][['model', 'seed', 'mrr']], on=['model', 'seed'], suffixes=('_bridge', '_pair'))
    bridge['delta'] = bridge.mrr_bridge - bridge.mrr_pair
    bridge_rows = [[model, fmt(rows.delta), int((rows.delta > 0).sum()), int((rows.delta < 0).sum())]
                   for model, rows in bridge[bridge.model.isin(['DistMult', 'ComplEx', 'RotatE', 'TransE', 'TuckER', 'RGCN'])].groupby('model')]
    shuffle_rows = [[setting, model, len(rows), f'{rows.mrr.mean():.4f}', f'{rows.shuffled_mrr.mean():.4f}',
                    f'{rows.popularity_score.mean():.4f}', f'{rows.difference_from_popularity.max():+.4f}']
                   for (setting, model), rows in shuffle.groupby(['setting', 'model'], sort=False)]
    control_rows = []
    for setting in ['t4_a', 't4_b']:
        for model in ['DistMult', 'RGCN', 'ULTRA', 'CN', 'RA', 'Popularity']:
            rows = matched[(matched.setting == setting) & (matched.model == model)]
            control_rows.append([setting, model] + [fmt(rows[rows.control == control].auroc) for control in
                ['original', 'original_controls_on_matched_positive_subset', 'recorded_degree_matched', 'unknown_fallback_only']])
    coverage = []
    for setting in sorted(matched.setting.unique()):
        records = [read(ROOT / 'supplement_data/matched' / setting / f'seed_{seed}.json') for seed in [42, 123, 456]]
        first = records[0]
        coverage.append([setting, f"{first['matched_recorded_controls']}/{first['original_positives']}",
            f"{min(r['matched_positive_coverage'] for r in records):.1%}–{max(r['matched_positive_coverage'] for r in records):.1%}",
            f"{min(r['fallback_fraction'] for r in records):.1%}–{max(r['fallback_fraction'] for r in records):.1%}"])
    pop = matched[(matched.model == 'Popularity') & (matched.control == 'recorded_degree_matched')]
    imbalance = pop[(pop.auroc - .5).abs() > .03]
    imbalance_note = '；'.join(f'{r.setting} seed {r.seed}: {r.auroc:.4f}' for r in imbalance.itertuples()) or '无'
    correlations = pd.DataFrame(read(ANALYSIS / 'loso_rank_correlations.json')['rows'])
    corr_rows = [[setting, anchor, fmt(rows.spearman), fmt(rows.kendall)]
                 for (setting, anchor), rows in correlations.groupby(['setting', 'anchor'])]
    hours = (datetime.fromisoformat(queue['finished_utc']) - datetime.fromisoformat(queue['started_utc'])).total_seconds()/3600
    report = f'''# MicrobeKG：完整实验结果与复现记录

更新：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}。本文件为唯一主结果说明；原执行清单保留为历史计划，下面的实测结果优先于清单中的预期数字。

## 完成状态与结果入口

**本轮 567/567 项全部完成，无失败、OOM、超时或依赖阻塞。** 原 96 项 HPO、99 项 seed42 主实验保持不变；各轮结果已合并到 `autodl/`，不再从解压目录或旧快照读取。

| 内容 | 数量 | 位置 |
|---|---:|---|
| 原 HPO / seed42 主实验 | 96 / 99 | `autodl/results/` |
| 补 seed 123、456 | 198 | `autodl/supplement_results/jobs/replicate__*/` |
| ComplEx 第二优验证配置诊断 | 1 | `jobs/diagnostic__*/` |
| 度保持打乱 / LOSO | 20 / 99 | `jobs/shuffle__*/`、`jobs/loso__*/` |
| ULTRA 零样本 / 度匹配评估 | 33 / 216 | `jobs/ultra__*/`、`jobs/matched__*/` |
| 全部结果对 / 模型权重 | **762 / 237** | JSON+NPZ；原 54 份、新 183 份 `.pt` |

- [全部实验索引](autodl/supplement_results/analysis/experiment_index.csv)：762 行，统一定位结果、原始分数和权重；HPO、主实验、诊断与控制实验分组保留，不把不同协议混为一个均值。
- [非匹配结果总表](autodl/supplement_results/analysis/all_observed_results.csv)：450 行；[三 seed 均值与标准差](autodl/supplement_results/analysis/main_mean_std.csv)：143 个模型×设置组合，均为 n=3。
- [匹配评估总表](autodl/supplement_results/analysis/matched_all_observed.csv)：216 项 × 4 种对照口径；同目录 `matched_mean_std.csv` 给出 AUROC/AUPRC 汇总。
- [最终本地核验](autodl/supplement_results/local_checks/final_audit.json)：ZIP SHA-256 与解压后 3,173 文件一致，合并后原始输出保持校验和。另从 NPZ 重算 **{audit['point_metrics_recomputed_cells']} 项、{audit['point_metric_comparisons']:,} 个点指标**，最大差异 **{audit['max_absolute_difference']:.2e}**；没有重做模型推理或置信区间 bootstrap。

双 5090 续跑复用 128 项、完成剩余 439 项，墙钟 **{hours:.2f} 小时**；2026-09-17 10:04 UTC（北京时间 18:04）结束。此耗时不包括更早的原实验和 A40 阶段。

## 总体判断：支持哪些预期

1. **随机 pair 的成绩不能直接解释为迁移能力。** RotatE 的 T1 pair MRR 为 0.1488±0.0079；LOSO 最佳模型随来源改变。Peryton 上 Popularity 0.1640 最强，gutMDisorder 上 L3 0.0647 最强，HMDAD 上 TransE 0.0305 略高于 RotatE 0.0300。来源、样本与候选分布不同，不能把跨任务落差作为单一因素的因果效应。
2. **T4-B 的高 AUROC 叙事需要撤回。** DistMult 原始 AUROC 为 0.9149±0.0276，记录对照度匹配后为 0.4967±0.0114；固定到相同正例子集、仍用原对照时为 0.9291±0.0237。对照选择对结论影响很大；未知 pair 回退 AUROC 0.6058 属于另一个问题，不能替代已记录关联对照。
3. **T1 打乱支持目标边包含有用信号，但并非所有任务都退化到度数基线。** T1 六学习模型 × 三 seed 均不超过对应 Popularity +0.01；T4-A 的 RGCN/DistMult 在单 seed 打乱后仍高出 Popularity 0.0338/0.0173。辅助边保留，尚不能把残余分数归因于某一种结构或机制。
4. **ULTRA 补齐了现代零样本基线，但并非普遍领先。** T1 pair 为 0.1298±0.0006，低于 RotatE 和 TuckER；cold-D 为 0.0144±0.0021，是该列最高均值，但绝对检索能力仍低。ULTRA T2-A 未纳入原计划，不能当作漏跑或填零。
5. **ComplEx 和稳定性问题仍是限制。** ComplEx T1 pair 为 0.000762±0.000108，第二优验证配置诊断 MRR 为 {diag['second_config_reruns'][0]['test_mrr']:.6f}，并未修复低分。旧八配置在首个验证点（第 10 轮）达最佳，损失有限且权重恢复检查通过；完整范数轨迹未保存，原因未定。DistMult cold-M 跨 seed 为 0.075291、0.000353、0.001229；保留全部低分，不将其武断标为代码崩溃。

ComplEx T2-A/T2-B MRR 为 0.8681±0.0088、0.2158±0.0089，支持性能落差的描述，但不等于“证据类型决定可迁移性”的因果结论。Popularity 在 T1 pair 的 MRR 为最佳模型均值的约 56%，该比值不是“度数解释了 56% 性能”的效应分解。

## 指标与可比性

以下均为 **seed 42/123/456 的均值 ± 样本标准差**。主表 T4-B 是原始对照 **AUROC**，其余列是 **MRR**；T4-B 的论文结论应结合后面的度匹配表。不同列不是相同难度的同一任务，跨 seed 标准差也不是显著性检验。

原 11 模型 × 9 设置全部有三 seed；ULTRA 在计划的 11 设置上各有三 seed。双向 MRR 不能替代冷端检索：TuckER cold-M 双向均值 {cold.mrr.mean():.4f}，预测冷微生物的 head MRR 只有 {cold.head_mrr.mean():.4f}，tail MRR 为 {cold.tail_mrr.mean():.4f}。cold split 沿用训练可见候选与辅助图，不宣称严格未见实体归纳评测。

{chr(10).join(main_tables)}

LOSO 仅将单个来源独占支持的目标 pair 留出，跨源共同支持 pair 留在训练，辅助图保留。它是来源独占目标 pair 泛化测试，不是时间前瞻验证或完全独立外部队列。

## 多关系辅助边与打乱对照

下表为相同 seed 的 `+bridge − T1 pair` MRR 差；三个 seed 同方向不等于正式显著性证据。

{table(['模型', '差值均值 ± 标准差', '提升 seed 数', '下降 seed 数'], bridge_rows)}

打乱只作用于目标关系，保留端点度、辅助边、验证/测试、候选及资格掩码；过滤仍使用原始训练真值。下表为 seed 均值，最后一列是各 seed 中最大的“打乱 − Popularity”。T4-A 只有 seed42，单独标明 n=1。

{table(['设置', '模型', 'n', '原 MRR', '打乱 MRR', 'Popularity', '最大残余差'], shuffle_rows)}

## 度匹配与未知 pair 回退

下表均为 **AUROC，n=3**。记录对照与未知 pair 回退分别报告，不能合并称为治疗/冲突识别。匹配改变了正例覆盖与对照人群；另列同一匹配正例子集配原对照，以区分正例选择的影响。AUPRC 的正例比例基线及分数保存在机器表，不能跨比例直接比较。

{table(['设置', '模型', '原始', '同正例子集+原对照', '记录对照匹配', '未知回退'], control_rows)}

T4-A RGCN 匹配后仍有 0.7550±0.0245 的区分度，但不是疗效或因果证据。T4-B 匹配后最高均值为 CN 0.5975、RA 0.5944，不能继续把 DistMult 原高分作为强迁移结果。

{table(['设置', 'seed42 匹配数/正例数', '三 seed 匹配覆盖率', '三 seed 未知回退比例'], coverage)}

Popularity 在 18 个设置×seed 的记录对照匹配结果中，AUROC 范围为 {pop.auroc.min():.4f}–{pop.auroc.max():.4f}；**超出 0.50±0.03 的项为：{imbalance_note}**。不能宣称所有匹配均完全消除了度数残余；T4-A/B 均通过该数值检查。T1 冲突匹配覆盖率很低，大部分未知回退不能继续称为已记录冲突识别。

## 排名是否可迁移到 LOSO

每个相关系数比较同 seed 的 12 个模型；下表对三个相关系数取均值±标准差，仅作描述，不将三个 seed 当成独立来源重复。

{table(['留出源', '对照设置', 'Spearman', 'Kendall'], corr_rows)}

Peryton 与 T1 pair 的 Spearman 约 0.29–0.36，HMDAD 约 0.50–0.78，gutMDisorder 约 0.45–0.71。基准内排名的可迁移程度依赖来源，不支持一个统一的高泛化排行榜。

## 清单完成情况与剩余工作

| 清单 | 当前状态 |
|---|---|
| N1 | 三 seed 实验完成；结论稳定性逐项判断，尚未新增成对显著性检验 |
| N2 | 诊断与第二配置重跑完成；低分原因未解释，不能标为修复完成 |
| N3 | 20/20 完成；T1 满足清单数值预期，T4-A 两模型保留超阈值残余 |
| N4 | 216/216 完成；记录对照、同正例子集与未知回退分开；cold-D seed123 有残余度数可分性 |
| N5 | 度分桶、候选/答案规模、度偏斜及闭包离线分析完成；结构可用性不是模型使用该路径的证明 |
| N6 / N7 | ULTRA 33/33、LOSO 99/99 完成，未发生 OOM/OOT |
| N8 | 逐源再分发许可、LICENSE、DOI、完整源数据重建链待完成；没有执行公开发布 |
| N9 / N10 | 按当前结果重写论文叙事与图表；低分/负结果可移附录，不删除或改成预期分数 |
| S1–S4 | HPO/运行时/排名分析已有；20 个案例候选尚无独立文献核验 |

旧清单中的“修复后必须高于 Popularity”“headline 不出现低于随机”不是删改数据的规则。统计显著性、根因解释和投稿可用性不能由队列完成直接推出。T2-B 实验否定对照仅 81 条（seed42），AUPRC 需结合正例比例，不宜作为强结论。FB15k-237 每 query 平均答案数按当前明示口径为 3.03，不能照搬清单里未统一口径的 399.82。

![度分层、HPO 敏感性与排名诊断](autodl/supplement_results/analysis/offline_diagnostics.png)

此图的度分层、HPO 和排名相关面板基于原 seed42；三 seed 与 LOSO 结果见上表。PDF 在同目录。图表能描述现象，不能单独证明捷径的因果来源。

## 文件整理与复现

| 路径 | 当前用途 |
|---|---|
| `README.md` | 唯一主结果说明，所有新结论集中于此 |
| `autodl/results/` | 冻结的原 HPO/主实验、54 权重与原校验记录 |
| `autodl/supplement_results/jobs/` | 全部 567 项输出、逐样本分数、183 权重、日志与 receipt |
| `autodl/supplement_results/analysis/` | 合并索引、汇总表、诊断图与人工核查候选 |
| `autodl/supplement_results/local_checks/` | 本地最终完整性与点指标复算记录 |
| `benchmark/`、`supplement_plan.json` | 当前训练、评估、汇总代码及服务器实际执行计划 |
| `benchmark_data/`、`supplement_data/`、`vendor/` | 已校验输入、划分和 ULTRA 官方代码/权重 |
| `{ARCHIVE}/` | 完整下载 ZIP、旧续跑 ZIP、被替换版本、历史部署/打包记录及清理清单 |
| `original_data/`、`preprocessed/`、`kg_build/`、`literature_evidence/` | 原始数据、构建链和文献证据，保持原样 |

输入与训练代码指纹保持不变：`{read(ROOT / ARCHIVE / 'export_manifest.json')['code_sha256']}`。三 seed 同时改变划分与初始化，选参仍固定为 seed42 的验证选择；训练 batch、轮数、负采样和权重恢复规则不变。ULTRA 固定官方 50g 权重，不做目标训练；本轮推理 batch 从 32 调为 64（T1 pair/cold/LOSO）或 128（含完整辅助图），按验证查询吞吐与分数容差检查选择，不按测试成绩选择。容差检查不等价于所有排名逐位相同。

```bash
# 完整结果与依赖哈希检查；不训练
python benchmark/supplement.py verify
# 从 NPZ 复算点指标并更新统一索引；不训练、不重算置信区间
python benchmark/audit_results.py
# 从原始结果汇总 CSV/图，再生成本 README；不训练
python benchmark/supplement.py report
```

完整离线备份：`{ARCHIVE}/snapshots/MicrobeKG_2x5090_complete_20260917.zip`，12,483,674,748 字节；SHA-256：`619e466785a15293e9701d08b817c76cda47a835c88a31e76cd96161ab0e19b6`。备份保留服务器原始报告；当前结论以本文件为准。清理只针对已核对的副本、旧状态与缓存，不删低分结果或唯一权重。

重复解压副本位于 `{ARCHIVE}/verified_extraction/`：自动审批规则拒绝递归删除，因此改为归档，未物理删除，也未据此宣称释放磁盘空间。移动与清理记录见该归档目录下的 `file_moves.json`、`cleanup_check.json`；更早归档保持原样。

原计划保留于 [必要充分执行清单](MicrobeKG_KDD2027_必要充分执行清单.md)。其中 venue、文献与日期信息未在此次合并中重新联网核实；本文件只更新实际代码和结果支持的结论。
'''
    (ROOT / 'README.md').write_text(report, encoding='utf-8')
    print('Updated complete-results README:', ROOT / 'README.md')


if __name__ == '__main__':
    main()
