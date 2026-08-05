# Phase 6Q 五目标体推理因果审计报告

日期：2026-08-04。正式训练状态：**未开始**。

## 结论

当前 Phase 3/4 中“制导模型的 hard 地球物理响应仍接近基线”不能主要用非唯一性
解释。地震分支首先没有充分到达观测：在无 flow、无地质 loss、无停止阈值的最简
五目标体实验中，最好 hard seismic attainment 仍只有37.80%，远低于95%门槛。

原因不是单一的，而是分层出现：

1. **正演和观测配对没有总体程序错误。** Q0 在十二候选中枚举66个二体组合，
   property、blurred property、reflectivity、seismic、gravity 的真值组合均为唯一
   数值零且排名第一。
2. **信号在合适的低维结构参数化中足够。** Q1 候选 soft/STE 在 correct seismic
   上均精确选中两个隐藏体；zero/shuffled 不通过。
3. **从结构参数化放宽到自由体素后，地震优化严重停滞。** Q2 soft/STE hard
   attainment 只有3.30%/10.26%；Q2b hard-only 协调翻转提高到15.64%，仍未达到
   观测流形。更细的1–32体素 Q2c 反而降到4.00%，排除了“最小步长太粗”这一单因。
4. **soft-hard 桥接确实重新成为问题。** Q3 普通15类 soft embedding 的最终 soft
   seismic attainment 为12.59%，同时 hard attainment 为−87.64%；hard-only 最佳
   仍是原始 I0。这直接复现了“soft loss 下降但 hard 标签正演没有靠近真值”。
5. **hard-forward 能改善，但仍未解决。** Q3 rock-STE 的 correct seismic hard
   attainment 为37.80%，hidden IoU/两个岩体 recall 均为20%；正确观测明显优于
   shuffled（16.12%、IoU 0.22%），说明梯度含真值信号，但尚不足以恢复。
6. **15类物性竞争提供替代下降方向。** correct seismic 的 Q3 rock-STE 改了2,688
   个体素，只有256个为 label9，其余为 labels 4/5/6/8/10/12。限制为 label0/9 的
   Q3b 精确重现 Q2 的3.30%/10.26%，说明 checkpoint simplex embedding 几何本身
   不是新增故障；全类别替代可以提高物理拟合，但削弱目标岩性可辨识性。
7. **gravity 已经进入典型非唯一性区。** Q2 soft gravity hard attainment 93.85%，
   hidden IoU 仅9.53%；Q2b attainment 85.78% 而 IoU=0；Q3 rock-STE attainment
   70.67%，1,304个变更全部使用 label6，label9 恢复为零。这满足“物理接近但三维
   错误”，可以讨论非唯一性。地震不满足这一前提。

因此，Phase 3/4 结果同时包含 soft-hard 松弛、非局部地震算子的耦合/局部停滞、
自由体素缺少结构参数化、15类物性替代，以及冻结 flow/controller 的潜在冲突；其中
gravity 的主要表现是非唯一性，seismic 的首要问题仍是 hard 物理可达性。

## 关键结果

| 层级 | correct seismic 方法 | hard attainment | hidden IoU | 两体 recall |
|---|---|---:|---:|---:|
| Q1 | 候选 soft / top-2 STE | 100% / 100% | 100% / 100% | 100% / 100% |
| Q2 | 自由体素 soft / STE | 3.30% / 10.26% | 10% / 40% | 10%/10% / 40%/40% |
| Q2b | hard 坐标 32–512 | 15.64% | 70% | 70%/70% |
| Q2c | hard 坐标 1–32 | 4.00% | 12.5% | 15%/10% |
| Q3 | 15类 soft / rock-STE | 0% / 37.80% | 0% / 20% | 0%/0% / 20%/20% |
| Q3b | 二类 embedding soft / STE | 3.30% / 10.26% | 10% / 40% | 10%/10% / 40%/40% |

Q3 soft 的表中0%是 hard-only best；其最终 soft/hard attainment 分别为12.59%和
−87.64%，不能把 soft 改善当成 hard 地球物理拟合。

## 对原有结论的修正

- “mu/cap 上限不影响”的结论只在原 Phase 4/6P 控制器、原状态参数化和局部强度
  区间内成立。Phase 6Q 显示 hard-forward、参数化和类别约束可以显著改变结果，
  因此不能外推成“继续增大物理作用在任何实现中都无效”。
- “结果差是因为非唯一性”的表述必须按算子拆开。gravity 有直接证据；seismic
  尚未通过 hard physics attainment 门，不能先归因于观测流形内的非唯一解。
- Phase 1/2 的 soft-hard 解决是对直接、全分辨率、可辨识属性通道成立，不是对
  模糊/正演后的 acquisition-domain 观测自动成立。Phase 3/4 引入的信息损失和
  非局部算子使问题重新出现。

## 下一阶段与训练前验收门

正式训练前还需完成 Q4 冻结 flow 轨迹隔离，以及确认简单立方体是否在现有生成器
支持上。若 Q3 endpoint 失败，Q4 只能用来量化 flow 额外损失，不能替代当前失败。

Phase 6B adapter 训练必须至少满足：

- 数据 split/manifest 不变，训练、验证、测试和报告 case 隔离；
- all-class hard-aware endpoint/velocity 目标，不能只训练 soft expected property；
- correct observation 必须显著优于 zero/shuffled；
- hard seismic attainment ≥95%，不能用 soft loss 代替；
- hidden/body 或 held-out 地质门同时通过，防止用错误物性类别投机；
- 条件零违规、基模冻结 hash 不变、无基模梯度；
- gravity/seismic 分开报告物理拟合与地质恢复，不把高物理拟合自动解释为正确地质。

权威机器目录位于
`experiments/stage6_inference_causality/runs/five_body_cuboid_v1/`；具体冻结协议见
`docs/PHASE6Q_SIMPLE_CAUSAL_SPEC.md`，运行记录见
`experiments/stage6_inference_causality/WORKLOG.md`。
