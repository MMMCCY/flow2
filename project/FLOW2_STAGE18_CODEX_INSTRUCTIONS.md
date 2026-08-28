# Flow2 Stage18 Codex 开发指令
## 主题：Stage17 hard-seismic 审计与地球物理 evidence semantics 的最小修正实验

> 本阶段只解决 Stage17 已暴露出的一个明确问题：
> **成功的 Stage15-H / Stage17C 路径把 full-trace binary inversion score 当作“指向 label9 的正向置信度”，而不是完整的正/负 property target。**
> 不开发新方法族，不做阈值 sweep，不重新测试已经证明过的问题。

# 0. 开工前审计

修改任何文件前：

1. 读取并遵守：
   - `docs/AGENTS.md`
   - `docs/PROJECT_BASELINE.md`
   - `docs/EXPERIMENT_PROTOCOL.md`
   - `docs/RESEARCH_GOAL.md`
   - `docs/DEVELOPMENT_HANDOFF.md`
   - `docs/STAGE17_SPEC.md`
   - Stage17A/B/C formal reports
   - Stage15-H `binary_trace_property_flow_v2` 报告
   - `scripts/stage15/build_binary_trace_property_assets.py`
   - `scripts/stage15/run_binary_trace_boundary_inversion.py`
   - `guidance/binary_trace_boundary.py`
   - `guidance/property_volume.py`
   - `guidance/property_sampling.py`
   - `scripts/stage17/run_same_evidence_coupling.py`
   - `scripts/stage17/evaluate_same_evidence_coupling.py`

2. 记录：
   ```bash
   git status --short
   git branch --show-current
   git rev-parse HEAD
   ```

3. 检查 Stage18 是否已存在。若不存在，使用 `Stage18`；若已存在，使用下一个未占用编号并说明。

4. 不覆盖、不移动、不重写任何 Stage15/Stage17 既有结果。

5. 先汇报开发计划，再实施。

# 1. 必须保留的历史事实

## 1.1 成功的 Stage15-H / conference 版本没有使用 0.6 二值阈值

成功路径的 authoritative 语义是：

- full 320-sample trace inversion；
- `binary_impedance_score` 连续取值 `[0,1]`；
- normalized binary property codebook：
  - label9 = 1
  - every other class = 0
- Flow guidance 时：
  - `target_properties = 1`
  - `confidence = binary_impedance_score`
- **no thresholding**。

`binary_trace_property_flow_v2/REPORT.md` 明确写：
> continuous inversion score supplies confidence without a threshold.

`DEVELOPMENT_HANDOFF.md` 的 Stage15-H 段同样明确：
> inversion score is used as continuous positive confidence, not as calibrated occupancy and without thresholding.

会议图与 Stage15-H 成功实现同属 Stage15 commit lineage；不要把固定 0.5 diagnostic core 或早期 consensus 阈值当成成功 Flow bridge 的输入定义。

## 1.2 历史阈值属于其他分支

仓库中存在过：

- Stage15 consensus：
  - positive threshold = 0.8
  - negative threshold = 0.2
- Stage15-H evaluation：
  - fixed 0.5 core，仅作为 retrospective/display diagnostic

这些都不是 Stage15-H 成功 Flow bridge 的 guidance mapping。

本阶段**禁止自行引入 0.6 阈值**，因为：
- 当前成功 protocol 中没有它；
- `binary_impedance_score` 是 normalized impedance/anomaly-strength estimate，不是已校准的 `P(label9)`；
- 后验用 truth 挑一个 0.6 只会重新引入不可归因的 threshold tuning。

# 2. Stage18 科学问题

只回答两个问题。

## Q1：Stage17C 已生成的 hard geology 是否真的更符合原始 binary seismic observation？

这是一个纯 retrospective audit。

不重新采样，不修改 Stage17C，不生成新的 Flow 结果。

## Q2：Stage17C 的系统性过预测是否主要来自 positive-only evidence semantics？

当前 successful semantics 近似为：

\[
L_{pos} \sim \sum_x q(x)\,[p_9(x)-1]^2
\]

其中：

- `q(x)` = Stage17A `binary_impedance_score`
- score 高：强推 label9
- score 低：弱推 label9
- score = 0：不施加 geophysical guidance

本阶段测试最小替代语义：

\[
L_{prop} \sim \sum_x [p_9(x)-q(x)]^2
\]

注意：

**不要把 `q` 命名成 calibrated probability。**

它应被描述为：

`continuous normalized binary-property target`

因为 `q` 来源于已知 background/label9 impedance endpoints 之间的 normalized log-impedance estimate。

# 3. Stage18A：现有 Stage17C hard-seismic consistency audit

## 3.1 不运行新的 Flow

输入直接使用：

`experiments/stage17_evidence_coupling_attribution/stage17c/formal_all5_all3_v1`

中的全部既有：

- 5 geology cases
- seeds 42 / 142 / 242
- `FLOW_ONLY`
- `TRAJECTORY_EVIDENCE`

共 30 个 decoded hard geology。

## 3.2 hard seismic 计算必须复用 Stage17 observation model

对每个 decoded geology：

1. 转为 binary occupancy：
   ```python
   occupancy = (decoded_geology == 9).float()
   ```

2. 使用 Stage17 冻结的：
   - binary acoustic config
   - seismic config
   - subsurface support
   - observed seismic

3. 使用：
   - `binary_occupancy_to_acoustic`
   - `seismic_operator_from_config`

4. 计算：
   - hard seismic MSE
   - hard seismic RMSE

这是 **Stage17 binary observation model consistency**，不得描述成 full-lithology field seismic consistency。

## 3.3 必须输出

逐 pair：

- case
- seed
- arm
- hard seismic MSE / RMSE
- paired delta RMSE：guided - Flow-only
- 已有 target IoU / P / R / volume error，可从 Stage17C evaluator 读取，不重复重新定义

按 geology case 先对 3 seeds 取 median，再跨 5 cases 汇总。

## 3.4 结果语言

不设计复杂 gate，只分三种解释：

- `SURROGATE_OVERSHOOT_SIGNAL`：
  多数独立 cases（至少 3/5）在 target IoU 改善时 hard seismic RMSE 反而变差。
- `HARD_PHYSICS_COMPATIBLE_BUT_VOLUME_UNCALIBRATED`：
  多数 cases hard seismic RMSE 改善或持平，但 target volume 仍明显过预测。
- `MIXED_HARD_PHYSICS_RESPONSE`：
  以上两者均不成立。

这只是下一步归因，不是 final-goal PASS/FAIL。

# 4. Stage18B：只改变 evidence semantics

## 4.1 不重复已经完成的正式实验

不要重新正式运行：

- `FLOW_ONLY`
- Stage17C `TRAJECTORY_EVIDENCE` positive-only arm

它们已经是 immutable formal reference。

新的 formal run 只生成一个新 arm：

`CONTINUOUS_PROPERTY_TARGET`

每 case × seed 共 5 × 3 = 15 个新输出。

评估时引用 Stage17C 既有两组输出作为 paired reference。

## 4.2 唯一允许改变的科学变量

Stage17C 当前：

```python
score = binary_impedance_score
confidence = score * free_subsurface
target_properties = ones
```

Stage18B 改成：

```python
score = binary_impedance_score
confidence = free_subsurface.float()
target_properties = score
```

对于单通道 binary indicator property：

- label9 endpoint = 1
- all other classes = 0

因此 property loss 直接比较：

`Flow expected normalized binary property` vs `seismic-derived normalized binary property score`

### 必须保持不变

- Stage17A evidence tensor及 hash
- checkpoint
- EMA/raw policy
- 3D U-Net
- conditions
- case registry
- source seeds
- initial noise
- 32-step fixed Euler
- alpha
- max guidance ratio
- tau start/end/schedule
- guidance start/schedule
- grad clipping
- guidance scaling mode
- property sigmas
- property scale weights
- decoder
- condition projection
- target label
- binary property table

禁止 threshold、temperature sweep、alpha sweep、score rescaling、case-specific normalization。

# 5. pairing 与 provenance

每个 Stage18B 新输出必须验证：

- `case_id` 与 Stage17C 相同；
- evidence tensor SHA 与 Stage17C 相同；
- initial noise SHA 与对应 Stage17C seed 相同；
- checkpoint SHA 相同；
- condition tensors SHA 相同；
- solver / n_steps 相同；
- scientific guidance hyperparameters 相同。

如果任何一项不一致，该 pair 不可用于科学比较。

# 6. soft-hard 诊断嵌入同一个实验，不另开新 Stage

Stage18B runner 必须保存：

- final continuous state，或至少足以在 evaluator 中稳定重算 final soft class probabilities 的 tensor。

使用现有：

`soft_decode_to_probs`

在冻结的 `tau_end` 下计算：

- final soft label9 probability mass：
  \[
  M_{soft} = \sum_x p_9(x)
  \]
- final hard label9 voxel count：
  \[
  M_{hard}
  \]
- `M_hard - M_soft`

目的仅是判断：

- overprediction 是否已经存在于 soft state；
- 还是主要在 nearest-embedding hard decode 后放大。

不要因此开发新的 decoder、STE 或 discrete Flow。

# 7. Stage18B 必须计算 hard seismic

对新 `CONTINUOUS_PROPERTY_TARGET` hard geology，使用与 Stage18A 完全相同的 Stage17 binary hard-seismic evaluator。

因此每个新 pair 同时有：

- hard geology metrics
- hard seismic RMSE
- soft/hard volume diagnostic

# 8. 正式评价指标

主统计单位仍然是 independent geology case；3 seeds 是 case 内 stochastic replicates。

每个 case 对 3 seeds 报 median，并保留全部 15 pair。

至少报告：

- target IoU
- precision
- recall
- TP / FP / FN
- truth target volume
- predicted target volume
- absolute volume-error fraction
- centroid distance
- global fixed/truth-present mIoU
- global voxel accuracy
- hard-condition violations
- hard seismic RMSE
- `M_soft`
- `M_hard`

不要报告“提升了几倍”作为主要结论。

# 9. Stage18B 的简单结果分类

本阶段核心比较是：

`CONTINUOUS_PROPERTY_TARGET`
vs
Stage17C `TRAJECTORY_EVIDENCE`

并同时检查它是否仍优于 `FLOW_ONLY`。

## `VOLUME_REPAIR_WITH_LOCALIZATION_RETAINED`

同时满足：

- 至少 3/5 independent cases 的 case-median absolute volume error 低于 Stage17 positive-only；
- 至少 3/5 cases 的新 arm target IoU 仍高于对应 Flow-only。

不要求它必须超过 Stage17 positive-only 的 IoU，因为允许用一部分 recall 换取显著减少 FP / volume inflation。

## `VOLUME_REPAIR_LOCALIZATION_TRADEOFF`

- volume error 在至少 3/5 cases 改善；
- 但 target IoU 相对 Flow-only 的正向作用不足 3/5。

## `NO_VOLUME_REPAIR`

- volume error 改善少于 3/5 cases。

这些只是 mechanism/scientific interpretation，不是 final-goal success gate。

hard seismic结果单独报告，不允许一个指标补偿另一个指标。

# 10. smoke 与 formal

## unit tests

可使用小 tensor/mock：

- truth firewall
- non-empty output refusal
- evidence hash validation
- Stage17C initial-noise hash matching
- new target semantics：
  - `target_properties == score`
  - `confidence == free_subsurface`
- 明确验证代码中不存在 thresholding
- hard seismic evaluator binary decode
- condition projection
- final-state / soft-mass calculation

## CUDA smoke

只运行：

- first registered case
- first registered seed
- new arm `CONTINUOUS_PROPERTY_TARGET`

smoke 与 formal 使用完全相同科学参数。

smoke 不能减少：
- 32 Flow steps
- property scales
- guidance semantics
- evidence construction

smoke 只验证工程可运行性，不参与科学结论。

## formal

5 cases × 3 seeds × 1 new arm = 15 new Flow outputs。

不重复 Stage17C 的 30 个旧 outputs。

# 11. Stop rule

Stage18 完成后必须停止。

不要自动：

- 调 threshold
- 测 0.6
- 做 positive/negative threshold sweep
- 调 alpha/cap
- 做 D-Flow
- 做 PGDM/PnP
- 做 hard correction
- 做 topology stress
- 做 full 15-class inversion
- 加 noise / wavelet mismatch
- 加 gravity/magnetics
- 训练新网络

下一步由 Stage18A/B 的结果决定：

- 如果 hard seismic 恶化且 continuous-property semantics 仍不能控制 volume，才讨论 hard-physics correction；
- 如果 volume 明显修复且 localization 保留，则优先把同一方法推进到 full-lithology acoustic background；
- 如果 soft mass合理但 hard volume异常放大，才授权单独研究 soft-hard decode。

# 12. 建议目录

若 Stage18 未占用：

```text
project/geodata-3d-conditional/
├── docs/
│   └── STAGE18_SPEC.md
├── experiments/
│   └── stage18_evidence_semantics/
│       ├── README.md
│       ├── configs/
│       │   └── continuous_property_target_v1.json
│       ├── hard_seismic_audit/
│       ├── smoke/
│       ├── formal/
│       └── reports/
├── scripts/
│   └── stage18/
│       ├── audit_stage17c_hard_seismic.py
│       ├── run_continuous_property_target.py
│       └── evaluate_continuous_property_target.py
└── tests/
    └── test_stage18_*.py
```

允许按仓库现有命名规范调整，但职责不要混合。

# 13. 最终交付

Codex 必须报告：

1. repository audit；
2. 变更文件清单；
3. Stage17 references 和 hashes；
4. focused tests exact commands/results；
5. CUDA smoke exact command/result；
6. Stage18A hard-seismic audit report；
7. Stage18B 15-pair formal report；
8. 5-case primary summary；
9. soft-vs-hard mass diagnostics；
10. 当前 evidence semantics 是否支持“positive-only 导致 overprediction”的因果解释；
11. hard seismic 是否支持后续 hard-physics correction；
12. 已证明 / 未证明；
13. 更新 `docs/DEVELOPMENT_HANDOFF.md`；
14. 完成后停止，等待下一步研究决策。

# 14. 本阶段的核心研发纪律

本阶段只改变一个变量：

\[
\boxed{
\text{同一个 Stage17A score 应该作为“正向置信度”，
还是作为“连续 normalized property target”}
}
\]

不改变 Flow，不改变 seismic，不改变 case，不改变 guidance strength，不改变 solver。

任何同时修改 evidence tensor、threshold、guidance strength 或 sampler 的实验，都不能回答本阶段问题。
