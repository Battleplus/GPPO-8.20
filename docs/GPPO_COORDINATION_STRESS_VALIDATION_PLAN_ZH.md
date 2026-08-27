# GPPO 协调压力优势假设：证据总结与验证设计

> 状态：研究备忘录 / 预注册式设计草案，尚未执行新的压力验证实验  
> 存档日期：2026-08-27  
> 基础 evidence commit：`2afa8ec1cb481deb57645dbd30240d90d32d2233`  
> 训练 source commit：`32974ec85be71e192b12cae85d00eb877d5fe07d`

## 1. 目的

本文档固化目前关于 PPO-MLP 与 GPPO-Adaptive 的讨论、已有证据能够支持的结论，以及下一步应如何验证下面这条机制假设：

> **GPPO-Adaptive 的潜在优势主要出现在多事件重叠或突发、需要关系建模和多 UAV 协调的条件，而不是所有场景的平均回报。**

这不是已经证实的结论，而是根据固定 50k held-out 评估形成的、可被后续实验证伪的假设。

## 2. 已完成实验的可信边界

已有 minimum-validation 证据闭环满足：

- 模型仅包含 PPO-MLP、GPPO-Adaptive；
- training seeds 为 `1101、2202、3303`；
- 每个 run 精确训练 `50,000` decision steps；
- 每组保存 `25k、50k` checkpoint，共 `12/12`；
- 后续评估固定使用六个 50k checkpoint，不进行 checkpoint selection；
- held-out bank 共 100 cases：Single、Sequential、Overlap、Burst、Unseen 各 20；
- 六个 checkpoint 使用完全相同的 100-case bank，共产生 `600/600` 模型-case 结果；
- checkpoint、source、protocol、seed manifest provenance 和 SHA-256 均通过复验；
- 没有补训、续训、参数调整或根据结果修改协议。

因此，下面的数字可以用来描述**这一次冻结 minimum-validation 协议**，但由于只有三个独立训练 seeds，不能直接外推成普遍性的算法优越结论。

## 3. 已观察到的结果

所有差值均定义为：

```text
GPPO-Adaptive - PPO-MLP
```

### 3.1 总体结果

| 指标 | 差值 | 当前解释 |
|---|---:|---|
| Event success rate | `+0.00267` | GPPO 略高，但三-seed 不确定性跨零 |
| Legal coverage rate | `0` | 完全持平 |
| Recovery delay | `-0.00837` | GPPO 略低，但不稳定 |
| Cumulative uncovered time | `-0.05033` | GPPO 略低，但不稳定 |
| Normalized distance | `+0.01121` | 数值越低越好；三-seed 区间稳定倾向 PPO |
| Load gap | `-0.01072` | GPPO 略低，但不稳定 |
| Switch count | `-0.02000` | GPPO 略低，但不稳定 |
| Episode return | `+0.01441` | 几乎持平，seed 方向不一致 |
| Communication bytes | `0` | 完全持平 |
| Inference latency | `+14.553 ms` | 数值越低越好；GPPO 明显更慢 |

总体上，当前证据**不支持 GPPO-Adaptive 全面优于 PPO-MLP**。质量指标大多接近、相同或随 seed 改变方向；GPPO 的额外计算成本则明确保留下来。

### 3.2 按场景的 Episode return

| 场景 | GPPO − PPO | 三个 seed 的方向 | 当前解释 |
|---|---:|---|---|
| Single | `-0.27999` | 混合 | PPO 平均更好，但不稳定 |
| Sequential | `+0.10444` | 混合 | GPPO 有轻微正向趋势 |
| Overlap | `+0.13123` | 混合 | GPPO 有轻微正向趋势 |
| Burst | `+0.26223` | 三个 seed 均偏向 GPPO | 当前最值得继续验证的 GPPO 优势信号 |
| Unseen | `-0.14586` | 三个 seed 均偏向 PPO | 当前最一致的 PPO 泛化优势信号 |

五个场景的三-seed 95% 区间都跨零。因此，上表只能称为“方向性趋势”，不能写成已证明的场景优势。

### 3.3 按训练 seed 的总体 Episode return

| Seed | GPPO − PPO |
|---:|---:|
| 1101 | `-0.29722` |
| 2202 | `+0.33100` |
| 3303 | `+0.00946` |

一负、一正、一个接近零，说明当前回报差异小于训练随机性带来的波动，尚未形成 seed-stable 的质量优势。

## 4. 为什么会出现这种结果

以下是与观测一致、但尚未被因果实验确认的解释：

1. **硬约束和动作 mask 可能主导了决策。** Legal coverage 与 communication bytes 完全相同，说明大量行为由共同环境约束、合法动作集合和通信协议决定。
2. **额外表达能力不等于自动产生收益。** GPPO 的图结构和 adaptive gate 增加了模型容量，但当前 reward 与 50k 训练条件未必提供了足够清晰的关系建模信号。
3. **GPPO 的收益可能具有场景条件。** Sequential、Overlap、Burst 的平均方向为正，而 Single 和 Unseen 为负，符合“复杂协调时可能有用、简单或分布变化时未必有用”的假设。
4. **额外计算成本是确定存在的。** GPPO 推理延迟更高；如果策略质量没有同步改善，部署层面的净收益会变差。
5. **三个 seeds 太少，真实小效应容易被优化随机性覆盖。** 当前结果足以发现明显的大差异，但不足以稳定确认小幅质量提升。

这些解释不能仅凭最终 held-out 分数区分。要判断究竟是 gate 未发挥作用、图关系没有增量，还是训练样本不足，需要一个预先声明、只改变协调压力的机制验证实验。

## 5. 适用条件的当前判断

### 5.1 当前更倾向选择 GPPO-Adaptive 的条件

- 多个事件在短时间窗口内密集到达；
- 多个 UAV/区域关系同时发生改变；
- 任务更关注 Burst/Overlap 下的恢复与协调；
- 系统可以接受更高推理延迟；
- 后续压力实验确认优势随协调压力单调增强。

### 5.2 当前更倾向选择 PPO-MLP 的条件

- 单事件或低并发任务占主导；
- 部署要求低延迟、低算力消耗；
- 更关注 normalized distance 和路径效率；
- 面对 Test-Unseen 类分布变化；
- 尚无更强证据证明 GPPO 的质量收益能够覆盖其计算成本。

## 6. 下一步验证实验：协调压力梯度

### 6.1 核心研究问题

在模型、checkpoint、环境语义、reward 和 case seeds 不变的情况下，随着事件协调压力升高，GPPO 相对 PPO 的配对效果是否系统性改善？

形式化假设：

```text
H1: coordination_pressure 增加时，
    oriented_effect(GPPO - PPO) 呈正向趋势。

H0: GPPO - PPO 与 coordination_pressure 无稳定关系，
    或收益不足以覆盖 latency / distance 成本。
```

### 6.2 固定不变的内容

- 六个已经封存的 50k checkpoints；
- 两模型 × 三 training seeds 的完整矩阵；
- model、environment、reward、trainer 实现；
- action mask、event semantics、max decisions；
- 每个压力等级使用相同的实例 seed/event seed 对；
- 所有模型在同一 tape 上成对比较；
- 不读取 25k checkpoint 性能；
- 不训练、不续训、不做 checkpoint selection。

### 6.3 压力等级

第一版建议完全复用现有 scheduler 的四种时序合同，不发明新的环境语义：

| 压力等级 | 场景 | 解释 |
|---:|---|---|
| 0 | Single | 同一时刻只有一个待处理事件，最低协调压力 |
| 1 | Sequential | 事件连续到达，但主要按顺序处理 |
| 2 | Overlap | 后一事件可在前一事件恢复前到达 |
| 3 | Burst | 事件按簇同时进入，形成最高并发协调压力 |

每个等级建议至少 100 tapes、每 tape 8–12 个事件。四个等级必须共享配对 seed 索引，避免把场景差异和随机 case 差异混在一起。

该实验必须使用全新的开发性 namespace，例如：

```text
coordination_stress_v1_<source-sha>
```

它不能覆盖、重新消费或伪装成已经完成的一次性 formal held-out Test。

### 6.4 预先冻结的指标

继续使用代码中的 `PAIRED_METRICS`：

- event_success_rate；
- legal_coverage_rate；
- recovery_delay；
- cumulative_uncovered_time；
- normalized_distance；
- load_gap；
- switch_count；
- episode_return；
- communication_bytes；
- inference_latency_ms。

机制诊断只能从现有 trace 字段中读取，不改变 reward 或模型：

- 同 tape 上 PPO 与 GPPO 的动作分歧率；
- active/pending event 数量与模型差值的关系；
- gate_mean、gate_variance 随压力等级的变化；
- 每 episode decision_count；
- 每获得一单位回报改善所增加的 inference latency；
- 回报改善与 normalized distance 代价的联合变化。

机制诊断用于解释，不替代冻结质量指标，也不能在看结果后改变主结论标准。

### 6.5 统计单位与报告方式

- tape 内比较：同 checkpoint seed、同 tape 的 GPPO − PPO 配对差；
- 算法稳定性单位：training seed，而不是把数百个 tapes 当成独立模型训练；
- 每个压力等级报告三个原始 seed 效应；
- 报告 seed 均值、标准差、95% 区间和 case-pair bootstrap 描述；
- 四个压力等级的趋势检验必须在执行前固定；
- 十个指标 family 使用 Holm 校正；
- 同时报告质量、距离、通信和延迟，不允许只展示有利指标。

### 6.6 预注册式判断标准

只有同时满足以下条件，才把结果解释为“GPPO 在高协调压力下具有优势”：

1. Burst 的主质量效应在三个 training seeds 中方向一致；
2. Burst 的 seed-level 不确定性不再跨零，或扩大 seeds 后仍稳定为正；
3. 从 Single 到 Burst 的 oriented effect 呈预先声明的上升趋势；
4. 优势不是由单个异常 tape 或单个 seed 驱动；
5. GPPO 的质量收益能够明确量化，并与 latency、distance 代价同时呈现；
6. Unseen/低压力场景的退化不被隐去。

如果 Burst 优势消失、seed 方向继续混合，或者收益小于延迟/距离代价，则应接受当前更简单的解释：PPO-MLP 是这一预算与部署条件下更稳妥的模型。

## 7. 实施边界

后续代码应作为独立实验工具加入，不修改已有 formal minimum-validation 合同，也不覆盖正式 evidence：

```text
experiments/coordination_stress/
  README.md
  contract.json
  run_coordination_stress.py
  analyze_coordination_stress.py
  tests/
```

GitHub 只提交：

- 实验代码；
- 冻结合同；
- 单元测试；
- README；
- 小型机器可读汇总和 SHA inventory（实际执行后）。

不应提交六个 checkpoint 二进制；运行时由使用者显式提供训练 evidence campaign，并在启动前重新核验 SHA-256。

## 8. 当前结论

当前最准确的模型定位是：

> **GPPO-Adaptive 是“可能在复杂动态协调中产生收益、但更慢且泛化尚不稳定”的候选模型；PPO-MLP 是“更简单、更快、路径效率更好且当前证据更稳妥”的基线。**

下一步实验的任务不是寻找一个让 GPPO 看起来更好的数据集，而是严格检验：GPPO 的相对收益是否真的随协调压力增强，并且是否足以覆盖它的计算与路径成本。

