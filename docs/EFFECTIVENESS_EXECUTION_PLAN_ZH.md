# GPPO 方法有效性实验执行规划（100k Effectiveness Protocol）

> 本文档定义 L1 轻量随机事件仿真阶段的“方法有效性”实验主线。
> 目标不是追求最大训练预算，而是用最小但统计上可信、可复现、可审计的实验，回答：GPPO-Adaptive 在本项目的随机事件驱动资源分配场景下，是否比标准 PPO 更有效，优势具体体现在哪里。

---

## 0. 当前研究目标

当前阶段的首要目标是：

> **证明 GPPO-Adaptive 在随机事件驱动资源分配场景，特别是 Overlap、Burst 和 Unseen 条件下，相比 PPO-MLP 具有更好的可行性保持、恢复效率和泛化能力，并用明确的指标、效应量和跨 seed 结果支撑该结论。**

本阶段不以“把所有模型训练到最大步数”为成功标准。

本阶段必须回答 5 个问题：

1. GPPO-Adaptive 是否稳定优于 PPO-MLP？
2. 优势具体体现在哪些物理/任务指标？
3. 优势是否在复杂动态事件下更明显？
4. 优势是否跨训练 seed 稳定，而不是单次随机结果？
5. 优势中有多少来自图结构，有多少来自 adaptive gate？

---

# 1. 实验论证链

三个 learned methods 全部保留：

```text
PPO-MLP
GPPO-NoGate
GPPO-Adaptive
```

它们承担不同的论证职责：

```text
PPO-MLP
   ↓
用于证明 graph representation 是否带来收益
   ↓
GPPO-NoGate
   ↓
用于证明 adaptive gate 是否带来额外收益
   ↓
GPPO-Adaptive
```

因此：

- `GPPO-Adaptive vs PPO-MLP` 是主比较；
- `GPPO-Adaptive vs GPPO-NoGate` 是核心消融；
- `GPPO-NoGate vs PPO-MLP` 用于拆分 graph representation 的贡献。

不得只保留 Adaptive 和 PPO 而删除 NoGate 消融。

---

# 2. 核心实验假设

## H1：方法有效性

在相同训练预算、相同 PPO 超参数、相同训练 seed、相同事件协议和相同 Validation/Test tapes 下：

```text
GPPO-Adaptive > PPO-MLP
```

这里的“更优”优先指：

```text
final infeasible rate ↓
recovery latency ↓
```

而不是只看累计 reward。

## H2：图结构有效

如果：

```text
GPPO-NoGate > PPO-MLP
```

则说明图结构建模本身提供了有效信息归纳偏置。

## H3：Adaptive gate 有额外贡献

如果：

```text
GPPO-Adaptive > GPPO-NoGate
```

则说明性能提升不能仅归因于“用了图网络”，adaptive gating 在动态事件下提供了额外收益。

## H4：复杂动态事件下优势应更明显

重点不是要求 Adaptive 在每一个简单场景都全面碾压 PPO，而是检验：

```text
Single → Sequential → Overlap / Burst → Unseen
```

随着事件并发、突发性和分布偏移增强，Adaptive 的相对优势是否扩大。

如果 Single 差异很小，但 Overlap / Burst / Unseen 显著改善，这是符合方法动机的有效结果。

---

# 3. Frozen Experiment Contract

除本文件明确修改的训练预算和实验目标外，以下协议继续沿用 S4 已冻结合同：

```text
GraphObservationContract
ActionContract
mask
reward
PPOConfig
事件确认语义
并发/version/lease/fencing 语义
Validation/Test seed namespace 隔离
Test 不参与选模
```

任何实施前都必须重新生成并通过新的 Gate，证明该版本没有破坏这些合同。

模型之间唯一允许的算法结构差异仍然是：

```text
PPO-MLP       = canonical flattened graph → MLP
GPPO-NoGate   = graph → AHGNN
GPPO-Adaptive = graph → AHGNN + adaptive gate
```

---

# 4. Training Design

## 4.1 Training seeds

固定：

```text
1101
2202
3303
```

三个模型必须使用完全相同的 training seeds。

训练 seed 是统计上的独立训练单位。

不得：

- 某个模型换 seed；
- 删除表现差的 seed；
- Test 后补 seed；
- 因为某个 seed 结果不好而重新挑 seed。

## 4.2 Effectiveness budget

默认正式有效性预算冻结为：

```text
100,000 accepted decision steps / model / seed
```

checkpoint：

```text
25,000
50,000
75,000
100,000
```

总训练量：

```text
3 models × 3 seeds × 100,000
= 900,000 accepted decision steps
```

总 checkpoint：

```text
3 × 3 × 4 = 36
```

本阶段不要求默认训练到 300k。

## 4.3 为什么采用 100k

100k 的目的不是宣称“100k 必然完全收敛”，而是提供：

1. 25k/50k/75k/100k 四个 learning-curve 观测点；
2. 三个独立 training seed；
3. 足够进行 checkpoint selection；
4. 足够观察 sample efficiency；
5. 将训练成本控制在原 300k 协议的 1/3。

是否需要扩展预算由 Validation learning curve 决定，但必须遵守第 11 节的扩展规则。

---

# 5. Validation Protocol

Validation 固定：

```text
100 tapes total
Single       25
Sequential   25
Overlap      25
Burst        25
Unseen        0
```

Validation 可以用于：

```text
checkpoint selection
learning-curve analysis
判断 100k 是否已经足以形成稳定结论
```

Validation 不得用于：

```text
证明 Unseen 泛化
修改 Test 分布
Test 后反向调参
```

## 5.1 Checkpoint selection

每个 `variant × training seed` 独立按冻结字典序选择：

```text
1. lowest final infeasible rate
2. lowest cumulative weighted vacancy
3. lowest recovery latency
4. lowest fixed J
5. earliest checkpoint
```

选中后必须冻结：

```text
checkpoint path
checkpoint SHA-256
source tree hash
source commit SHA
protocol SHA
seed manifest SHA
PPOConfig
training seed
selected step
```

不得用 Test 替换 Validation 进行选模。

---

# 6. Test Protocol

正式 Test 固定：

```text
200 tapes total
Single       40
Sequential   40
Overlap      40
Burst        40
Unseen       40
```

Test 必须在：

```text
training complete
→ Validation complete
→ checkpoint selection complete
→ checkpoint freeze complete
```

之后才运行。

Test 禁止用于：

```text
checkpoint selection
training budget selection
reward tuning
architecture tuning
adaptive gate tuning
MLP width tuning
事件概率调整
```

正式 Test 对冻结 checkpoint 只运行一次。

---

# 7. Primary Metrics

向导师汇报时，Primary metrics 预先冻结为：

## 7.1 Final infeasible rate ↓

核心问题：

> 随机事件发生后，最终有多少情况无法形成满足约束的可执行资源分配？

这是最直接的系统鲁棒性指标之一。

## 7.2 Recovery latency ↓

核心问题：

> 系统受到 UAV damage、target discovery、通信异常或事件重叠扰动后，需要多长时间重新形成合理分配？

这是动态恢复能力的核心指标。

Primary claim 必须优先围绕这两个指标展开。

---

# 8. Secondary Metrics

至少报告：

```text
cumulative weighted vacancy ↓
fixed J
Unseen generalization
sample efficiency
event recovery success rate
pre-mask invalid probability
stale rejection / concurrency safety metrics
```

解释含义：

- `weighted vacancy`：恢复过程中有多少资源/任务位置长期空缺；
- `fixed J`：整体优化目标表现；
- `Unseen`：面对未见事件组合/扰动分布时是否保持性能；
- `sample efficiency`：达到同等性能所需训练步数；
- safety metrics：确保性能提升不是通过违反 mask、version、lease 或并发约束获得。

不得只报告 reward 而不报告物理任务指标。

---

# 9. Scene-wise Analysis

所有正式结果必须分别报告：

```text
Single
Sequential
Overlap
Burst
Unseen
```

禁止只给一个总体平均值。

重点分析：

```text
Overlap
Burst
Unseen
```

因为这些场景最直接对应本方法的研究动机：

- 多事件同时影响资源分配；
- 图结构和局部状态快速变化；
- 事件需要确认、合并和重新决策；
- 分布可能偏离训练阶段。

推荐最终结论形式：

> “简单 Single 场景中三种算法差异有限；随着事件并发和突发程度增加，GPPO-Adaptive 在 infeasible rate 和 recovery latency 上的优势扩大，并在 Unseen 场景中表现出更小的性能退化。”

只有数据支持时才能使用该表述。

---

# 10. Learning Curve 与 Sample Efficiency

必须使用：

```text
25k
50k
75k
100k
```

四个 checkpoint 形成 learning curve。

至少分析：

1. 三个模型的学习速度；
2. 75k → 100k 是否趋于稳定；
3. Adaptive 是否更早达到稳定性能；
4. Adaptive 50k/75k 是否达到或超过 PPO 100k；
5. 跨 3 seed 的趋势是否一致。

如果 Adaptive 在更少训练步数下达到 PPO 的最终水平，可以形成：

> “GPPO-Adaptive 具有更高的 sample efficiency。”

该结论必须由 checkpoint-level Validation 数据支持。

---

# 11. 何时允许扩展到 150k / 200k / 300k

100k 是默认 Effectiveness Protocol 的正式预算。

只有出现以下情况之一，才允许设计 Extended Benchmark：

```text
75k → 100k 仍在快速改善
三种方法尚未形成稳定排序
Adaptive vs PPO 差异仍被较大 seed 波动覆盖
关键 Primary metrics 明显尚未趋稳
```

扩展可以考虑：

```text
150k
200k
300k
```

但必须满足：

1. 扩展规则在查看正式 Test 之前决定；
2. Test 不得用于决定训练步数；
3. 所有模型和 seeds 使用相同扩展预算；
4. 不得只给 Adaptive 增加预算；
5. Extended Benchmark 与 100k Effectiveness Protocol 分开报告。

禁止因为 100k Test 结果“不够好看”而继续训练到更高预算。

---

# 12. Statistical Analysis

训练 seed 是独立统计单位。

必须报告：

```text
raw seed values
mean
standard deviation
seed-level uncertainty / 95% CI where appropriate
paired same-tape difference
effect size
Holm correction
```

主比较：

```text
GPPO-Adaptive vs PPO-MLP
```

核心消融：

```text
GPPO-Adaptive vs GPPO-NoGate
```

辅助比较：

```text
GPPO-NoGate vs PPO-MLP
```

可以保留 Greedy 等非学习方法作为 reference，但不能替代 PPO 主 baseline。

## 12.1 Paired evaluation

同一个 Test tape 上的算法结果必须进行 paired comparison。

例如：

```text
Δ infeasible
Δ recovery latency
Δ weighted vacancy
Δ fixed J
```

这比把所有 episode 混在一起做非配对统计更有解释力。

## 12.2 失败样本

不得删除未恢复/失败样本。

需要时同时报告：

```text
conditional recovered-only distribution
+
right-censored / failure-aware analysis
```

---

# 13. 最终导师汇报必须生成的结果

## Figure 1：Learning Curve

```text
x = training steps
25k / 50k / 75k / 100k

y = Validation metric

PPO-MLP
GPPO-NoGate
GPPO-Adaptive

3 seeds mean ± variability
```

用于回答：

```text
谁学得更快？
谁最终更好？
是否已经趋稳？
```

## Figure 2：Scene-wise Performance

比较：

```text
Single
Sequential
Overlap
Burst
Unseen
```

三个模型并列展示。

用于回答：

> Adaptive 的优势到底出现在哪类场景？

## Figure 3：Recovery Latency

重点展示 Overlap / Burst / Unseen。

用于回答：

> 动态事件以后谁恢复得更快？

## Figure 4：Final Infeasible Rate

直接比较三个模型的失败/不可行比例。

这是最适合导师快速理解的方法有效性图之一。

## Table 1：核心结果汇总

至少包含：

| Method | Infeasible ↓ | Recovery ↓ | Vacancy ↓ | Fixed J | Unseen |
|---|---:|---:|---:|---:|---:|
| PPO-MLP | ... | ... | ... | ... | ... |
| GPPO-NoGate | ... | ... | ... | ... | ... |
| GPPO-Adaptive | ... | ... | ... | ... | ... |

同时生成：

```text
Improvement vs PPO (%)
```

例如最终可报告：

```text
infeasible rate     ↓ xx%
recovery latency    ↓ xx%
weighted vacancy    ↓ xx%
Unseen degradation  ↓ xx%
```

`xx%` 必须由最终数据计算，不得提前填写或假设。

---

# 14. Success Criteria

本阶段成功标准不是：

```text
9 个 run 全部训练到 300k
```

而是：

在预先冻结、公平、可重复的实验协议下，有足够证据回答：

1. Adaptive vs PPO 的方向是否跨 seed 稳定；
2. Primary metrics 的 effect size 是否具有实际意义；
3. 优势主要出现在哪些指标；
4. 优势主要出现在哪些场景；
5. NoGate 消融是否支持 adaptive mechanism 的贡献；
6. Unseen Test 是否支持泛化能力；
7. 性能提升是否没有破坏 safety/protocol invariants。

如果这些问题在 100k budget 下已经得到清晰回答，则 Effectiveness Protocol 完成。

---

# 15. 允许与禁止的结论

如果数据支持，可以写：

> “在冻结的随机事件驱动资源分配环境中，GPPO-Adaptive 相比 PPO-MLP 在动态扰动后的可行性保持和恢复效率方面表现出稳定优势；该优势在 Overlap、Burst 以及 Unseen 场景中更加明显。”

如果 NoGate > PPO 且 Adaptive > NoGate，可以进一步写：

> “性能提升部分来自图结构建模，而 adaptive gating 在复杂动态事件下提供了额外收益。”

禁止在数据不支持时写：

```text
GPPO-Adaptive 在所有指标、所有 seed、所有场景全面优于 PPO
```

禁止把 L1 轻量随机事件仿真结果写成：

```text
已经证明在 SFC / Isaac Sim 高保真环境中更优
```

L2 结论必须等待独立高保真验证。

---

# 16. 当前 S4 300k Campaign 与新协议的隔离

截至本文档创建时，已有一个按旧 S4 300k protocol 启动的本地 campaign：

```text
preliminary_formal_s4_restart1
```

该 campaign 的原始协议是：

```text
3 models × 3 seeds × 300k
checkpoint every 25k
```

因此必须遵守以下规则：

1. 不得把它中途训练到 100k 后直接重新命名为新的 100k Effectiveness Formal；
2. 不得把旧 300k campaign 的部分 runs 与新 100k runs 拼接成一个正式结果集；
3. 如果继续让它运行，它只能作为 S4 Full Benchmark / pilot reference；
4. 如果决定停止，必须保留已有 artifacts，并明确标记：

```text
STOPPED_BY_PROTOCOL_CHANGE
usable_for_effectiveness_formal = false
```

5. 是否停止当前本地进程需要单独、明确的 operator authorization；本文档本身不构成 kill-process 指令。

新的 100k Effectiveness Formal 必须：

```text
new protocol version
new Gate
new source/protocol/seed hashes
fresh output directory
all 9 runs start from step 0
```

---

# 17. Implementation Plan

## Phase E0：协议冻结

修改并冻结：

```text
training budget = 100000
checkpoint interval = 25000
variants = 3
training seeds = 1101 / 2202 / 3303
Validation = 100 tapes
Test = 200 tapes
Primary metrics = infeasible rate + recovery latency
```

生成新的 protocol SHA 和 seed manifest SHA。

## Phase E1：代码最小修改

只做支持新 Effectiveness Protocol 所必需的修改：

```text
formal budget 100k
formal checkpoint grid 25/50/75/100k
progress reporting / read-only monitoring if needed
```

不得借此次协议调整顺便修改：

```text
reward
model architecture
confirmation semantics
mask logic
concurrency semantics
Validation/Test distributions
```

如果需要实现 parallel runner，必须单独审计，且保证每个 run 的算法语义不变。

## Phase E2：P0 Gate

重新运行完整 Gate：

```text
all required tests PASS
training_allowed = true
violations = []
source hashes present
protocol hash present
seed manifest hash present
no train/validation/test leakage
protected source clean
```

Gate 未通过不得正式训练。

## Phase E3：Smoke

至少继续覆盖：

```text
Single       >=20
Sequential   >=20
Overlap      >=20
Burst        >=20
```

并确认新 budget/checkpoint grid 行为正确。

## Phase E4：正式 100k Training

运行：

```text
PPO-MLP / 1101
PPO-MLP / 2202
PPO-MLP / 3303
GPPO-NoGate / 1101
GPPO-NoGate / 2202
GPPO-NoGate / 3303
GPPO-Adaptive / 1101
GPPO-Adaptive / 2202
GPPO-Adaptive / 3303
```

每个：

```text
fresh start
100000 accepted steps
25k checkpoint interval
4 checkpoints
```

总计：

```text
36 checkpoints
```

禁止 checkpoint resume，除非未来另行重新设计并审计正式 resume protocol。

## Phase E5：Training Completion Audit

训练完成后，先 STOP，不得直接进入 Validation。

检查：

```text
9/9 runs present
36/36 checkpoints present
checkpoint filename step == stored total_steps
all runs fresh-from-zero
no resume
all SHA256 generated
all source/protocol/seed hashes match
PPOConfig parity
no NaN/Inf anomalies except explicitly expected fields
stderr / exceptions reviewed
```

PASS 后才进入 Validation。

## Phase E6：Validation + Freeze

生成/加载冻结的 100-tape Validation bank。

评估全部 checkpoints。

每个 variant×seed 按固定字典序选 1 个 checkpoint。

最终冻结：

```text
9 selected checkpoints
```

保存 selection evidence 和 hashes。

## Phase E7：Formal Test Once

冻结后，对 9 个 selected checkpoints 运行一次 200-tape Test。

输出：

```text
per-tape metrics
per-mode metrics
seed-level summaries
paired differences
```

正式 Test 完成后禁止返回训练阶段调参。

## Phase E8：Statistics + Figures

生成：

```text
4 core figures
1 core result table
seed-level statistics
paired same-tape effects
Holm-adjusted comparisons
Improvement vs PPO percentages
```

形成导师汇报结论。

---

# 18. Parallelization / 加速原则

本协议允许未来把 9 个独立 `variant × seed` run 进行并行调度，因为它们本身统计上独立。

但是并行化只能改变调度，不得改变：

```text
seed
PPOConfig
event tape generation
training budget
checkpoint schedule
model architecture
reward
```

如果实现并行 runner：

```text
worker 1 → one variant×seed
worker 2 → another variant×seed
...
```

必须证明：

1. 每个 worker 从 step 0 开始；
2. 不共享模型/optimizer/RNG state；
3. 输出目录互斥；
4. 36 checkpoint 最终可合并为统一 index；
5. 任一 worker failure 不得通过 checkpoint splice/resume 静默修复。

GPU 加速不是默认前提。

当前 trainer 为小图 + Python-level per-graph PPO update，CUDA 是否真正加速必须先通过单独 benchmark 证明。

---

# 19. 最终导师汇报主线

最终汇报不应围绕“跑了多少步”，而应围绕：

```text
问题：动态随机事件导致资源分配拓扑不断变化，PPO 是否难以及时适应？

方法：图结构表示 + adaptive gating

证据：
1. infeasible rate
2. recovery latency
3. weighted vacancy
4. scene-wise robustness
5. unseen generalization
6. sample efficiency
7. NoGate ablation

结论：
Adaptive GPPO 是否、在哪里、以多大幅度优于 PPO。
```

最终最重要的不是一句“GPPO 比 PPO 好”，而是能够回答：

> **“好多少、好在哪里、在什么场景下最明显、是否跨 seed 稳定、是否能由消融解释。”**

---

# 20. 执行顺序

后续代理严格按以下顺序执行：

```text
E0 protocol freeze
→ E1 minimal implementation
→ E2 P0 Gate
→ E3 smoke
→ E4 3×3×100k training
→ E5 training completion audit
→ E6 Validation + checkpoint freeze
→ E7 Test once
→ E8 statistics + figures
→ advisor-facing conclusion
```

任何阶段失败：

```text
STOP → preserve evidence → fix → re-audit
```

不得通过修改报告、跳过 Gate、拼接 checkpoint、查看 Test 后再调协议等方式绕过失败。

---

# 21. 与原重构规划的关系

`docs/REFACTOR_EXECUTION_PLAN_ZH.md` 继续作为系统架构、事件协议、并发、Reward、P0 Gate、L0/L1/L2 分层的主规划。

本文档覆盖其中 Phase J/K 的“方法有效性实验”策略，并在新的 Effectiveness Protocol 中优先采用本文档定义。

如两者在训练预算上冲突：

```text
旧 Phase J 默认 300k = Full Benchmark Protocol
本文 100k = Effectiveness Protocol
```

不得把两个协议的产物混为同一个正式实验。
