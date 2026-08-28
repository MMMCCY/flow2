# Flow2 下一阶段 Codex 开发指令
## Stage17：地球物理证据泛化、D-Flow 校准与同证据耦合归因

> **用途**：将本文件直接交给 Codex 作为下一阶段开发任务说明。  
> **核心原则**：本阶段不是“寻找一个能让指标变好看的新方法”，而是用严格控制变量的实验，把“地球物理证据是否有用”和“证据如何进入冻结 CFM”两个问题分开验证。  
> **禁止把单个连续 loss 降低、倍数提升、单样本最好结果或 smoke-test 结果当成科学成功。**

---

## 0. 开工前必须做的事情

在修改任何文件前：

1. 读取并遵守：
   - `project/geodata-3d-conditional/docs/AGENTS.md`
   - `project/geodata-3d-conditional/docs/PROJECT_BASELINE.md`
   - `project/geodata-3d-conditional/docs/EXPERIMENT_PROTOCOL.md`
   - `project/geodata-3d-conditional/docs/RESEARCH_GOAL.md`
   - `project/geodata-3d-conditional/docs/DEVELOPMENT_HANDOFF.md`
   - `project/geodata-3d-conditional/docs/PHASE1_REPORT.md`
   - `project/geodata-3d-conditional/docs/PHASE2A_REPORT.md`
   - `project/geodata-3d-conditional/docs/PHASE3_REPORT.md`
   - `project/geodata-3d-conditional/docs/PHASE4C_REPORT.md`
   - `project/geodata-3d-conditional/docs/PHASE5C_REPORT.md`
   - Stage7 / Stage8A-v4 / Stage9A 报告
   - Stage15-H、five-body、topology-support 报告与相应 runner
   - Stage16 `README.md` 与 `reports/DEVELOPMENT_REPORT.md`
   - 仓库内的 D-Flow 论文 PDF；不要凭记忆复现论文方法。

2. 执行并记录：
   ```bash
   git status --short
   git branch --show-current
   git rev-parse HEAD
   ```
   不 reset、不 checkout 覆盖、不删除、不重排用户已有未提交文件，不覆盖任何既有实验输出。

3. 检查仓库是否已经存在 Stage17 或等价实验。若已存在，不创建重名阶段；使用下一个未占用编号，并在报告中说明编号变化。

4. 先提交一份**开发计划**，列出将复用的函数、需要新增的文件、不会修改的文件和实验顺序，再开始实现。

---

# 1. 本阶段科学问题

最终目标保持不变：

> 在冻结三维 CFM 地质先验和严格地表/钻井硬约束的前提下，让地球物理观测对未钻遇地下地质体提供有效全局软约束，同时由生成式地质先验限制地球物理反演的非唯一解；最终输出应是观测一致、条件严格满足、地质结构合理并保留不确定性的三维模型集合。

本阶段只回答两个相互独立的问题：

### Q1：Stage15-H 类型的地球物理空间证据是否跨地质 case 仍具有 case-specific 信息？

即先验证：

\[
d_i \rightarrow q_i(\mathbf x)
\]

本身是否有效，而不让 Flow 参与。

### Q2：在**完全相同的证据**下，trajectory guidance 与 D-Flow source optimization 哪种耦合方式能把证据转化为正确 hard geology？

即再验证：

\[
q_i(\mathbf x) + \text{frozen Flow} \rightarrow m_{\text{post}}
\]

本阶段**不直接开发 PGDM、DPS、PnP、SGLD、多物理联合、完整 15 类联合反演或新训练网络**。这些属于后续阶段，避免再次同时改变多个变量而无法归因。

---

# 2. 不可修改的全局实验边界

除非发现明确 bug 并单独报告，否则保持：

- 原 CFM checkpoint 不变；
- raw frozen `embedding.weight` 不变；
- 411 个 trainable model entries 使用既有 EMA 约定；
- 不修改 3D U-Net；
- 不重新训练原 CFM；
- surface / borehole hard conditions 的定义不变；
- hard conditions 在采样前和每个 fixed-Euler step 后重新投影；
- 严格配对使用既有 32-step fixed-Euler midpoint 路径；
- baseline/guided 或不同 coupling arms 必须共享：
  - checkpoint hash；
  - case tensors；
  - geophysical evidence tensor；
  - condition tensors；
  - source seed；
  - **完全相同的 initial noise tensor**；
  - solver；
  - time grid；
  - step count；
  - decode；
  - evaluation code。
- 不允许为了让新方法表现更好而修改 Phase1/Phase2/Stage15-H 已验证的 reference arm。
- 不允许根据 truth 指标挑选最好 seed、最好 case、最好参数作为正式结果。

---

# 3. 统一的结果解释规则

## 3.1 禁止的“成功”定义

以下任何一项**单独出现都不得写成成功**：

- seismic/property/oracle continuous loss 降低；
- 某指标“提高了几倍”；
- baseline label9 很小导致的高相对增长率；
- 单个 seed 有明显改善；
- 只看 recall 不看 precision/volume；
- 只看 global accuracy；
- 只看漂亮图片；
- 只看 soft probability 而不看 hard decode；
- 从若干样本中挑出最好一个；
- smoke-test 有好结果。

尤其禁止使用：

\[
\frac{M_{\rm guided}}{M_{\rm baseline}}
\]

这种在 baseline 接近 0 时会严重夸大的倍数指标作为主结论。

## 3.2 必须报告的绝对量

所有正式 Flow 结果至少报告：

- target IoU；
- target precision；
- target recall；
- TP / FP / FN 绝对 voxel 数；
- predicted target voxel count；
- truth target voxel count；
- absolute volume-error fraction；
- centroid distance；
-主要 truth components 的逐体 recall；
- predicted connected-component count；
- largest-component fraction / top-k mass；
- global voxel accuracy；
- truth-present fixed-class mIoU；
- hard-condition violations；
- paired hard-change voxel count 和占比；
- continuous evidence/objective；
- hard physics metric（若该 arm 有定义）。

不得以 global metric 掩盖目标体恶化。

## 3.3 三层结论，而不是单一 PASS/FAIL

报告使用：

1. **MECHANISM**：代码/梯度/搜索是否真实工作，hard conditions 是否保持；
2. **SCIENTIFIC EFFECT**：是否在多数固定配对中产生具有实际幅度的 hard-geology 改善；
3. **FINAL-GOAL**：是否同时满足形态、拓扑、物理一致性和不确定性要求。

例如：
- “mechanism active but not scientifically useful”
- “scientific target localization improves but topology remains insufficient”

不要把 progression gate 失败写成“整个方法族无效”。

---

# 4. Stage17A：Stage15-H 多 case 地球物理证据审计

## 4.1 目标

只回答：

> full-trace binary seismic inversion 生成的连续 score，在多个独立三维地质 case 上是否确实包含 case-specific 的 label9 空间信息？

**不运行 Flow。**

## 4.2 case 选择

优先复用仓库中已经冻结的独立 StructuralGeo benchmark cases，不要重新创造一套不必要的数据。

若现有 case 不足，则：

- 预先固定一个 StructuralGeo seed 序列；
- 按顺序选择前 `N >= 5` 个满足以下**任务资格条件**的 case：
  - 存在足够规模的未硬约束 label9；
  - label9 不是几乎全部被 borehole 命中；
  - 至少存在一个有实际体积的隐藏 target component；
- 资格条件只能用于保证“这个 case 能测试隐藏目标问题”，不得使用 seismic inversion/evidence/Flow 的任何表现进行筛选；
- 保存所有被跳过 case 的 seed 和跳过原因，防止 cherry-picking。

不要用“label9 数量刚好与 cond_generation_0 类似”作为筛选条件。

## 4.3 完全复用 Stage15-H 成功链

优先复用而不是重写：

- Stage15 binary acoustic config；
- Stage15 seismic operator；
- `run_binary_trace_boundary_inversion.py` 的 full 320-sample trace 逻辑；
- `binary_impedance_score.pt` 的定义；
- boundary strength；
- truth-blind runner；
- truth-aware evaluator 分离；
- score 不 threshold 后再提供给后续 Flow 的原则。

不得在 multi-case 中重新调：

- inversion prior weight；
- smoothness weight；
- refinement passes；
- wavelet；
- binary acoustic endpoints；
- trace window；
- score scaling；
- threshold；
- case-specific normalizer。

如果为泛化脚本需要重构代码，要求旧 `cond_generation_0` 结果可做 regression，数值定义不能被静默改变。

## 4.4 evidence 指标

每个 case 至少报告：

- voxel AUPRC；
- label prevalence（AUPRC 的随机基线）；
- `AUPRC - prevalence` 的**绝对差**；
- truth-target mean/median score；
- background mean/median score；
- boundary AUPRC；
- XY footprint AUPRC；
- fixed 0.5 core P/R/IoU 仅作为 diagnostic，不作为主 success gate。

### case-specificity 矩阵

对每个 evidence \(q_i\)，分别用所有 truth \(m_j\) 做 retrospective 评价：

\[
S_{ij} = \operatorname{AUPRC}(q_i, m_j)
\]

输出完整矩阵。

主结论使用：

\[
\Delta_i
=
S_{ii}
-
\operatorname{median}_{j\ne i}(S_{ij})
\]

而不是只看对角线绝对值。

建议 decision language：

- `EVIDENCE_CASE_SPECIFIC`: 至少 4/5 case 的 \(\Delta_i > 0\)，且 group median \(\Delta_i > 0\)；
- `EVIDENCE_INFORMATIVE_BUT_NOT_CASE_SPECIFIC`: AUPRC 高于 prevalence，但 diagonal/off-diagonal 不稳定；
- `EVIDENCE_NOT_SUPPORTED`: 多数 case 不高于 prevalence 或 diagonal specificity 不成立。

不要额外发明一个很高的绝对 AUPRC 门槛。

## 4.5 Stage17A stop rule

- 若 evidence 不具备 case specificity：**停止所有新的 D-Flow + geophysics 科学结论**，只允许独立进行 Stage17B 的 oracle/property 算法校准。
- 不允许通过“再调 Stage15-H 参数”挽救正式 multi-case 结果；任何新参数都必须成为新的独立协议。

---

# 5. Stage17B：D-Flow 机制校准

## 5.1 为什么必须先校准

Stage16 当前 one-seed smoke 只证明：

- through-Flow gradient 非零；
- source 被更新；
- endpoint objective 能降低；
- model 未改变；
- conditions 为 0 violation。

但 oracle/property hard geology 改善很弱，且当前 frozen config 是：

- Adam；
- lr=0.01；
- 20 iterations；
- source regularization=0；
- standard Gaussian init。

因此当前结果不足以区分：

1. D-Flow 本身不适合当前 3D categorical CFM；
2. Stage16 optimization setup 只是一个未充分校准的 smoke 配置。

## 5.2 校准只允许使用已知成功的 upper bound

只使用：

- Phase1 authoritative oracle probability；
- Phase2A authoritative ideal property。

不要使用 seismic truth 指标来选择 D-Flow 优化器。

## 5.3 reference arms

同一个 initial noise 下必须同时生成：

- `FLOW_ONLY`
- `REFERENCE_TRAJECTORY_GUIDANCE`
- `DFLOW`

其中 `REFERENCE_TRAJECTORY_GUIDANCE` 必须调用 Phase1/Phase2 已验证实现和原有参数，不得为了“公平”而改写 reference 方法。

D-Flow 必须复用 Stage16 end-to-end frozen-flow solver，不允许在这一阶段改成另一种生成模型或 latent representation。

## 5.4 D-Flow optimizer 开发限制

先审阅仓库内 D-Flow 论文，明确区分：

- 论文明确采用的做法；
- flow2 当前已有实现；
- 本项目为了 64³ categorical CFM 做出的工程选择。

允许一个很小的、**预先登记**的 calibration set，但只在 oracle/property development task 上：

- legacy Stage16 Adam 配置作为基线；
- 一个 paper-aligned LBFGS + line-search 配置。

不要进行大范围：
- lr sweep；
- iteration sweep；
- shell-weight sweep；
- seed-by-seed tuning。

如果要比较 Adam / LBFGS，报告：

- optimizer step 数；
- closure evaluation 数；
- endpoint solve 数；
- runtime；
- GPU peak memory。

不要因为 LBFGS 使用更多 forward/backward evaluations 就宣称“算法更好”而不报告计算量。

source-shell regularization：
- 可以实现并记录 diagnostic；
- 除非论文和预注册 calibration 明确要求，否则不要在看到结果后临时打开；
- 任何开启 shell regularization 的新配置都必须获得新的 config version 并重新 smoke。

## 5.5 D-Flow 的评价方式

不要用“比 baseline 提高 N 倍”。

对于每个 hard metric \(M\)，定义 reference recovery fraction：

\[
R_M =
\frac{M_{\rm DFLOW}-M_{\rm FLOW}}
     {M_{\rm REF}-M_{\rm FLOW}}
\]

只在 reference 分母具有明确、非微小改善时计算。

这个量表达：

> D-Flow 恢复了已验证 trajectory-guidance 增益的多少。

至少用于：

- target IoU；
- target recall；
- centroid-distance reduction；
- volume-error reduction。

同时保留原始绝对指标。

### D-Flow progression language

不要直接写 PASS/FAIL：

- `DFLOW_ENGINE_ACTIVE`：objective 降低、source 非零更新、hard geology 发生变化；
- `DFLOW_HARD_CONTROL_MATERIAL`：oracle 和 property 两类任务中，多数固定 seeds 均出现明确 hard-geometry 改善，且 median IoU recovery fraction 至少达到 reference gain 的约 25%；这是**进入 geophysical coupling 的工程 progression gate**，不是“D-Flow 理论成功”的通用结论；
- `DFLOW_ENGINE_ACTIVE_BUT_WEAK_HARD_CONTROL`：objective 能降但 hard recovery 很弱，此时停止把 D-Flow 用于 Stage17C geophysical arm，并报告 reachability/optimization 风险。

不得因为某一个 seed 恰好改善就进入 geophysical formal experiment。

---

# 6. Stage17C：同 evidence、不同 coupling 的严格归因实验

## 6.1 前提

- Stage17A evidence 达到 `EVIDENCE_CASE_SPECIFIC`，才允许正式运行任何 evidence-guided Flow multi-case experiment。
- `TRAJECTORY_GUIDANCE` arm 可在 Stage17A 通过后运行。
- `DFLOW` arm 只有 Stage17B 达到 `DFLOW_HARD_CONTROL_MATERIAL` 才允许正式运行。

## 6.2 关键控制变量

每个 case / source seed 的三种 arm：

1. `FLOW_ONLY`
2. `TRAJECTORY_EVIDENCE`
3. `DFLOW_EVIDENCE`（若 B 通过）

必须使用**完全同一个 Stage17A evidence tensor**。

不得出现：

- trajectory arm 使用 Stage15-H score；
- D-Flow arm 又改回 raw seismic MSE；

这种同时更换 evidence 和 coupling 的设计。

也不得重新构造一个“对 D-Flow 更友好”的 evidence。

## 6.3 trajectory arm

直接复用 Stage15-H 已成功的 Phase2-style property guidance：

- 相同 property endpoint 定义；
- 相同 continuous score-as-confidence 逻辑；
- 相同 Phase2 property sampler；
- 不重新调 alpha/cap/schedule/tau；
- 若 multi-case 需要泛化 assets，只泛化输入路径，不修改 loss 数学定义。

## 6.4 D-Flow arm

使用 Stage17B 冻结的 D-Flow config。

唯一变化是 endpoint objective 接收与 trajectory arm 完全相同的 Stage15-H evidence/property assets。

不得：
- 使用 raw seismic MSE 替换；
- 根据某个 geologic case 调 optimizer；
- 在 formal 中改变 iteration、lr、regularization、init；
- 用 truth 做 early stopping。

## 6.5 formal seeds

每个正式 case 使用固定的至少 3 个 source seeds；优先沿用：
- 42
- 142
- 242

若这些 seed 在所有 case 共享，则所有 arm 共享完全相同的 source tensor hash。

统计基于**全部 pair**，不以 seed mean 替代 pair 分布，也不挑最好 seed。

## 6.6 结果判读

主问题不是“谁的 IoU 最大”，而是：

### A. evidence 是否产生一致的 hard-geology 正向作用？

至少检查：

- majority paired target-IoU direction；
- recall 的**绝对百分点变化**；
- precision 是否因过预测大幅崩溃；
- volume-error 是否改善；
- centroid 是否接近；
- major-body recalls；
- global fixed-class mIoU；
- hard conditions。

避免使用诸如 “recall 从 0.01 到 0.04，增长 4 倍” 的语言。

### B. coupling 方法之间的比较

用：

- 每一 pair 的 absolute metric delta；
- group median delta；
- positive-pair fraction；
- 与 trajectory reference 的 recovery fraction。

如果 D-Flow objective 更低但 hard geology 更差，应明确写：

`PHYSICS/EVIDENCE OBJECTIVE BETTER, HARD GEOLOGY WORSE`

而不是“接近成功”。

### C. topology

普通 StructuralGeo multi-case 中：
- component metrics 作为重要 secondary metric；
- 不因为一个 component-count threshold 未满足就把所有定位改善写成失败。

five-body / ring 继续作为独立 stress test，不参与 Stage17C 的第一轮方法筛选。

---

# 7. smoke-test 与 formal experiment 的强制区分

这是本任务的重点要求。

## 7.1 unit test

目的：
- shape；
- hash；
- zero-guidance regression；
- condition projection；
- truth firewall；
- deterministic replay；
- D-Flow gradient；
- optimizer only contains source；
- old Stage15-H regression。

可以使用 tiny mock model。

它不是科学实验。

## 7.2 CUDA smoke

目的只有：
- 真实 checkpoint 能运行；
- 无 OOM；
- gradient/solver 正常；
- 输出文件完整；
- condition=0 violation；
- runtime 可接受。

**smoke 必须使用 formal config 的同一科学参数。**

允许 smoke 只减少：
- case 数；
- source seed 数；
- sample 数。

smoke **不得改变**：
- solver；
- 32 step；
- loss；
- alpha/cap；
- tau；
- optimizer；
- lr；
- optimization iterations；
- source regularization；
- evidence construction；
- decode；
- condition semantics。

也就是说，`--smoke` 应只选择正式预注册列表的第一个 case/seed/sample，而不是创建一种“轻量科学方法”。

## 7.3 smoke 后的代码修改规则

若 smoke 后发现工程 bug：

- 修复；
- 增加 regression test；
- 新建空输出目录；
- 重跑 smoke。

若修改会改变科学语义：

- config version 必须升级；
- 原 smoke 标记为 obsolete / superseded，但不得删除；
- 新配置重新冻结；
- 重新 smoke；
- 之后才能 formal。

不得将“smoke 用 Adam20，formal 改 LBFGS100”却仍称为同一个 v1 方法。

---

# 8. truth firewall

Stage17A inversion runner：

- truth 只可用于生成 synthetic observation 的离线资产构造；
- 运行 inversion 时不得读取 truth geology；
- evaluator 在 run 完成并冻结 hashes 后再加载 truth。

Stage17C：

- guidance runner 只能读取：
  - checkpoint；
  - conditions；
  - frozen evidence/property assets；
  - source tensor/config；
- 不得读取 target truth。

truth 仅在独立 evaluator 中出现。

所有 manifest 明确写：
- `truth_loaded_by_runner`
- `truth_role`
- input hashes
- output hashes
- config hash
- git commit
- source code hashes（沿用现有项目习惯）。

---

# 9. 代码复用优先级

不要重写已有稳定实现。

优先复用：

- `inference_runtime.py`
- `guidance/probability_sampling.py`
- `guidance/property_sampling.py`
- `guidance/property_volume.py`
- `guidance/binary_seismic_inversion.py`
- `guidance/binary_trace_boundary.py`
- `guidance/seismic.py`
- `guidance/dflow.py`
- Stage15 common utilities
- Stage15-H evaluator / asset builder
- Stage16 model freezing/hash audit

如确实需要抽公共函数：
- 先保证旧 Stage15-H / Stage16 regression；
- 不移动 immutable output；
- 不改变旧 manifest 的含义。

---

# 10. 建议目录

若 Stage17 未被占用：

```text
project/geodata-3d-conditional/
├── docs/
│   └── STAGE17_SPEC.md
├── experiments/
│   └── stage17_evidence_coupling_attribution/
│       ├── README.md
│       ├── configs/
│       │   ├── evidence_multicase_v1.json
│       │   ├── dflow_calibration_v1.json
│       │   └── coupling_v1.json
│       ├── cases/
│       ├── evidence/
│       ├── dflow_calibration/
│       ├── coupling/
│       └── reports/
├── scripts/
│   └── stage17/
│       ├── build_or_register_cases.py
│       ├── run_multicase_trace_evidence.py
│       ├── evaluate_multicase_evidence.py
│       ├── run_dflow_calibration.py
│       ├── evaluate_dflow_calibration.py
│       ├── run_same_evidence_coupling.py
│       └── evaluate_same_evidence_coupling.py
└── tests/
    └── test_stage17_*.py
```

具体命名可根据现有代码风格调整，但职责必须保持分离。

---

# 11. 配置冻结与版本规则

每个 formal config 必须：

- `status: frozen_before_run`
- schema version；
- exact asset hashes；
- exact source seeds/cases；
- exact solver；
- exact scientific hyperparameters；
- `parameter_sweep: false`；
- `training_performed: false`；
- truth firewall flags。

formal run 开始后：
- 不修改原 config；
- 不覆盖 output dir；
- 不后验改 threshold；
- 不根据正式结果追加“再试一个参数”。

若设计必须变化：
- 新 config version；
- 新 output dir；
- 新 scientific question；
- 原结果保留。

---

# 12. Figure 与报告规则

所有正式报告：

1. 先给数据来源与 protocol；
2. 再给完整 pair 表；
3. 再给 group summary；
4. 最后解释。

图片不得代替统计。

如果需要选择一个 case/seed 做主图：
- selection policy 必须预先声明，或者选择固定 seed；
- 若使用“最佳 IoU 样本”仅可作为可视化示例，标题明确 `selected for visualization only`；
- 科学结论必须使用所有 case/seed。

报告必须明确区分：
- source-derived facts；
- retrospective truth evaluation；
- interpretation；
- unresolved issues。

---

# 13. 测试要求

至少增加测试覆盖：

- multi-case runner 绝不加载 truth；
- off-diagonal specificity evaluator 的矩阵维度和 case mapping；
- same evidence tensor hash 在 trajectory/D-Flow arms 完全一致；
- baseline source hashes 完全一致；
- `--smoke` 不修改 scientific config；
- D-Flow optimizer 只更新 source；
- D-Flow model state hash 前后不变；
- condition projection；
- old Stage15-H cond_generation_0 regression；
- old Stage16 differentiable solver regression；
- non-empty output refusal；
- config frozen status validation；
- evaluation 不能在 run incomplete 时加载 truth。

运行：
- 新 stage focused tests；
- Stage15/Stage16 regression；
- 全 lightweight test suite。

报告 exact commands/results，不得只写“tests passed”。

---

# 14. 本阶段自动停止条件

Codex 不得无限推进。

### Stage17A 后：
- 若 evidence 不 case-specific：停止 Stage17C，输出报告，等待研究决策。

### Stage17B 后：
- 若 D-Flow 只有 endpoint objective 改善，但 oracle/property hard control 明显弱：
  - 可以记录 `DFLOW_ENGINE_ACTIVE_BUT_WEAK_HARD_CONTROL`；
  - **不要**继续跑 D-Flow geophysical formal arm；
  - trajectory arm 与 evidence multi-case 仍可独立研究。

### Stage17C 后：
- 不自动进入 PGDM/PnP/SGLD/topology/multiclass；
- 输出明确因果判断：
  - evidence bottleneck；
  - coupling bottleneck；
  - source reachability bottleneck；
  - 或 evidence + coupling 均有效但 topology 仍不足。

等待下一阶段授权。

---

# 15. Codex 最终交付物

完成后必须提供：

1. 改动文件清单；
2. 每个文件的职责；
3. git branch / commit / working-tree 状态；
4. 新增 config 及 hash；
5. unit/regression test 命令和结果；
6. smoke 命令和结果；
7. formal run 是否执行：
   - 未执行就明确写未执行；
   - 执行则给完整输出目录和 manifest；
8. Stage17A evidence report；
9. Stage17B D-Flow calibration report；
10. 若获准并满足前置条件，Stage17C coupling report；
11. 更新 `docs/DEVELOPMENT_HANDOFF.md`；
12. 明确写出：
    - 已证明什么；
    - 未证明什么；
    - 哪些 negative result 只关闭具体协议而不是整个方法族；
    - 下一步建议，但不要未经授权继续实现。

---

# 16. 最重要的研发纪律

本阶段始终遵循：

> **一次实验只改变一个科学变量。**

具体来说：

- 测 evidence：不碰 Flow coupling；
- 测 D-Flow：先用已验证 oracle/property，不碰 geophysics；
- 比 coupling：evidence 必须完全相同；
- 比 geophysical representation：coupling 必须完全相同。

任何同时改变两项以上的实验，都不得用于根因归因。

以及：

> **绝对 hard-geology 恢复效果优先于相对倍数；完整模型与几何指标优先于单个连续 loss；smoke 只验证工程可运行性，formal 才能产生科学结论。**
