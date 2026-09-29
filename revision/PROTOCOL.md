# 本轮实验与统计协议

版本：revision-20260918-v1；固定 seeds 42、123、456。计划在 plan.json。

## 147 个任务

| 部分 | 数量 | 范围 |
|---|---:|---|
| 验证集 HPO | 14 | ComplEx、DistMult 各 7 个候选，只用 T1 pair seed42 验证子集 |
| 修正后 T1 | 42 | 两模型 × 3 seeds × pair/bridges/cold-M/cold-D/3 LOSO |
| T1 目标边打乱 | 6 | 两模型 × 3 seeds，冻结新的 T1 配置 |
| T4-A 打乱补 seed | 4 | RGCN、DistMult × 123/456，沿用原 T4 配置 |
| ULTRA T2-A | 3 | 官方 ultra_50g，零目标微调 |
| KG-ICL | 36 | 官方 6L × 9 主设置及 3 LOSO × 3 seeds |
| 度匹配复评 | 42 | 修正的两模型 4 个 T1 设置共 24 项；KG-ICL 4 个 T1、T4-A/B 共 18 项 |

HPO 在所有 14 项成功后才选择配置，以 best_validation_mrr 为唯一标准；平分时取计划顺序靠前者。
ComplEx 搜索初始化标准差 .05/.1/.2 × NSSA margin 0/9，关系正则 .01；附原始 std1/margin9 对照。
DistMult 搜索关系正则 1e-5/1e-4/1e-3 × margin 0/9；附原始 .1/margin9 对照。
其他条件：dim128、32 negatives、Adam lr=.001、batch4096、最多200 epochs、每10 epochs验证、patience5、固定最多2048验证样本。
每个验证点保存实体范数、关系范数、平均两两余弦。低测试分不会导致排除或重选参数。
桥接全图每 epoch 更新次数与 pair 不同，因此桥接增益不能全部归因于边语义。

T2/T4 已有双线性配置不变，仍使用原任务族的验证选择；新 T1 修正不是全任务统一 HPO。
旧低分保留在历史数据中，正式修订主表不与新配置跨 seed 混合；新 3 seeds 未齐时列 pending。

## KG-ICL 适配

官方仓库：https://github.com/nju-websoft/KG-ICL
固定 commit 与权重 SHA-256 见 vendor/KG-ICL/UPSTREAM.json。
使用预训练 6L、5-shot prompt、官方默认 non-fused 推理；不是 LLM，也无需 API key。
每个评分关系从训练事实中确定性采样支持例，遵循官方 3-hop/open-neighbor prompt 规则。
图只含训练事实及其逆边；不把验证或测试正例纳入推理图或提示。
禁用官方可选的验证答案距离先验。提示节点与距离标签按明确索引对齐。
官方模型权重严格加载；仅排除 non-fused 路径从不调用的 36 个 fused conv 声明参数，并核对键白名单。
两个小补丁：评估时的临时 RReLU 使用 training=False；多维索引改为兼容新 PyTorch 的写法。
源代码、GPL-3.0、补丁差异和模型权重随包提供。改动均明确披露，不声称未修改地复现官方跨数据集数值。

ULTRA 官方仓库：https://github.com/DeepGraphLearning/ULTRA
生产使用原生编译 rspmm。setup 阶段必须通过 CUDA 推理检查；CPU reference 只用于本地小图测试。
两个基础模型均沿用所有基线的训练可见、类型约束、filtered realistic-tie 排名候选。
cold split 的冷实体可经辅助边在训练图中出现；不能称为严格未见实体归纳测试。
基础模型每项上限2小时，超时/OOM只在实际执行后记录；不填0、不自动换 checkpoint 或测试子集。

## 四条发现的 bootstrap

2,000次 bootstrap；A−B 差值、95% percentile CI，完整报告所有预定比较。
排名以 typed microbe–disease pair 为簇，同 pair 的两个关系及 head/tail 一起重采样。
先核对两模型三元组身份与候选规模，再对齐结果，不能只按数组位置相减。
每 seed 独立配对重采样；三 seed 结果是三个固定划分条件下的平均差 CI，不能解释为重复划分总体的不确定性。
实体共享可能带来额外依赖；该 CI 针对 pair 簇抽样，不能代表独立临床研究间变异。

1. F1 来源迁移：random 和每个 LOSO 内分别比较 RotatE−Popularity。不同来源测试 pair 不相同，禁止跨设置假配对。跨来源排名变化作为描述性证据。
2. F2 对照选择：T4-B DistMult 固定相同 matched-positive 子集，比较记录度匹配对照与全部原记录对照的 AUROC。重采样匹配的正负 block；出现在两个估计中的对照共享权重，其他原对照独立重采样。未知回退不混入。
3. F3 打乱：T1 六学习模型、T4-A RGCN/DistMult；分别比较原图−打乱和打乱−Popularity。对应 seed 的测试 pair 相同，保持原图过滤真值。
4. F4 现代模型：ULTRA/KG-ICL 与预先指定的 RotatE/TuckER 比较 4 个 T1 与 T2-A。另报 cold-M head、cold-D tail。不会根据测试结果挑选比较对象。

附加诊断：修正双线性模型−Random，覆盖7个T1设置。置信区间未作多重比较校正，不能挑一项显著就宣称普遍领先。

## Peryton 与案例

按来源联合归属计算独占 pair；共享 pair 仍在 LOSO 训练中。报告训练总度数、目标关系度数、候选池及 held-out 分布。
共同 log2(head degree)×log2(tail degree) 区间标准化是描述性敏感性分析，必须报告覆盖率与同区间未加权分数。
不能将观察到的度分布差异直接解释成生物机制或因果效应。
未独立核验的20个候选不进入修订稿发现段，采用用户授权的“删段”方案。
