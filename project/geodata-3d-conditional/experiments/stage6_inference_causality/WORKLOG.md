# Phase 6Q work log

## 2026-08-04：阶段启动

- 用户授权在远端 `/home/xmj/mcy/flowtrain_stochastic_interpolation-main`
  撰写并运行代码；正式训练仍不启动。
- 远端：`172.27.231.254:22`，branch `main`，commit `5b717ec`。
- GPU：NVIDIA GeForce RTX 4090 D，24564 MiB；`.venv` Python 3.10.12。
- 工作树已有大量用户/既有实验未提交文件；本阶段只新增
  `stage6_inference_causality`、Phase 6Q 文档、对应 guidance/script/test。
- 首轮顺序：Q0 硬候选枚举，然后 Q1 soft 与 STE 候选系数优化。

## 2026-08-04：Q0-Q2c

- Q0/Q1 权威目录：`runs/five_body_cuboid_v1/q0_q1_full_v2`；v1 因 STE
  hard-best 初值语义错误保留但废止。Q0 的66个二体组合中，五种观测的 truth pair
  均为唯一数值零、rank 1；Q1 correct 的 soft/STE 均精确恢复 `[4,6]`。
- Q2 权威目录：`q2_full_v2`；v1 因 entropy trace 浮点 NaN 保留但废止。correct
  seismic：soft hard attainment 3.30%、STE 10.26%；gravity soft/STE hard attainment
  93.85%/73.78%，但 hidden IoU 仅9.53%/6.54%。
- Q2b 权威目录：`q2b_full_v1`。hard-only 单调坐标求解得到 correct seismic
  attainment 15.64%、hidden recall 70%；gravity attainment 85.78% 但 hidden IoU=0。
- Q2c 事后细步长目录：`q2c_seismic_fine_v1`。将 proposal 改为1/2/4/8/16/32
  后 correct seismic 只达到4.00%、hidden recall 12.5%；因此 Q2b 的最小32体素并非
  停滞主因，较大的协调更新反而必要。

## 2026-08-04：Q3/Q3b checkpoint embedding endpoint

- 冻结配置：`configs/embedding_endpoint_v1.json`；检查点 SHA256
  `561e94bf...94c`，只读取 `embedding.weight [15,15]`，不加载 U-Net。
- property 冒烟：`q3_property_smoke_v1`，两方法均在 step30 得到 hard attainment=1、
  hidden IoU=1。
- 首次完整目录 `q3_full_v1` 在地表以上精确 one-hot air 进入旧声学 helper 时触发
  “non-air support empty”；它是未完成目录，保留不覆盖。修正为只在 subsurface
  检查/归一化 rock probability，新增回归测试。
- Q3 权威目录：`q3_full_v2`。correct seismic 的 `soft_embedding` hard-best 仍为
  I0（attainment=0），最终 soft attainment=12.59% 而 hard=-87.64%；
  `ste_embedding_rock` hard attainment=37.80%、hidden IoU/recall=20%。shuffled STE
  attainment=16.12% 但 IoU=0.22%。gravity STE attainment=70.67%，全部1,304个变更
  体素为 raw label6、没有 label9。
- 查看 Q3 后冻结 Q3b 全类别竞争诊断：`configs/embedding_binary_control_v1.json`，
  权威目录 `q3b_binary_full_v1`。限制为 raw label0/9 后，correct seismic 结果精确
  重现 Q2：soft attainment=3.30%、recall=10%，STE attainment=10.26%、recall=40%。
  因此真实 simplex embedding 连续几何本身不是新增主因；Q3 全类别物性替代提高了
  地震拟合但降低了目标类别可辨识性。
- 远端测试：Phase6Q 专项 `16 passed`；完整项目测试 `168 passed, 13 warnings`；
  warnings 均为既有 matplotlib/pyparsing deprecation。GPU peak 约221 MB。
- 已执行中文 Notebook：`flow2_phase6q_simple_causality_summary_zh.ipynb`，25 cells、
  9/9 code cells 已执行、5 个嵌入 PNG、0 error；SHA256 `ef91707e...7b89`。
- 人工结论：`docs/PHASE6Q_CAUSALITY_REPORT.md`。正式训练未启动。

## 2026-08-07：D0–D2 implementation gates

- 远端重新确认：`main @ 11caa498b15e6b89891604e9537830b30df504fa`，RTX 4090 D，`.venv` Python 3.10.12；任务开始时工作树 clean。
- D0 从实际代码冻结 Phase1–4 调用图。确认 canonical seismic 无 Gaussian blur，真正施加到状态上的 physics velocity 为 `-guidance_velocity`。
- D1 权威目录 `d1_observation_closure_v1`：property/reflectivity/seismic/gravity truth closure 全部逐位相等、truth loss=0，baseline 与 truth 分离。首次 runner metadata KeyError 在创建结果目录前修正，无物理变更。
- D2 首次目录 `d2_gradient_audit_v1` 保留：float64 FD 与 decoder 均通过，但生产 float32 raw step 小于可表示精度，被保守判为 probe failure。仅放大诊断 probe 后的权威目录为 `d2_gradient_audit_v1_fix1`；五条链相对误差 `2.71e-9`–`1.83e-8`，controller cosine 约1，实际 Euler soft loss 下降。

## 2026-08-07：D3–D5 causal isolation

- D3 `d3_soft_hard_transfer_v1`：L0–L2 soft/hard 约100%；首个分叉为 L3 reflectivity/TWT。L4 seismic 最大/最终 soft `19.09/19.09%`，hard `2.44/-0.80%`；最佳步126有64个 target crossing、0 wrong crossing、gradient hidden energy `99.9996%`；crop=0、projection erasure=0。
- D4 `d4_frozen_flow_trajectory_v1`：BASE 精确回归既有 fixed-Euler prior sampler。correct BASE_PLUS 最大/最终 paired hard advantage `30.64/11.52%`，zero `30.25/29.94%`，shuffled `30.35/17.52%`；PHYSICS_ONLY 最大仅9.30%。绝对 observed hard loss 的 best 是最终 step32，但末段 paired advantage 恶化，未触发2×预算。
- D5 `d5_native_geology_audit_v1`：五个 StructuralGeo IntrusionSpec 事件以临时标签9–13恢复 body masks 后合并为 audit label9。8/8 无制导先验样本在条件外含目标，7/8 有尺度相容分量；native correct BASE_PLUS 最大/最终 hard `31.48/14.44%`，且控制组同样改善，复现 cuboid 故障。

## 2026-08-07：D6 synthesis and stop

- 机器裁决：`diagnostic_verdict.json`；人工综合：`DIAGNOSTIC_SYNTHESIS.md`。
- 项目完整测试命令：`PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q project/geodata-3d-conditional/tests`；结果 `181 passed, 13 warnings in 14.75s`。
- warnings 均为既有 matplotlib/pyparsing deprecation。正式训练、U-Net 微调、新推理算法与广泛超参扫描均未执行。按 D6 stop condition 停止。

## 2026-08-07：D7 observation specificity

- 开始时仓库为 clean `main @ 85d5deb4555430117887a8ba173a0222c6b899ae`。
  D1–D4 历史 runner hash 来自当时 untracked 文件，因此只对 D1–D4 建立新
  provenance rerun；D5 全部 hash 已匹配，无需重跑。当前源码、配置、checkpoint
  均匹配，观测 hash 存在，`provenance_verified=true`。
- 权威 D7：`runs/five_body_cuboid_v1/d7_observation_specificity_v1`。在 BASE
  step 8/12/16/20/24/28/32 与共同 endpoint 上使用完全相同 state。correct 对
  controls 的 mean seismic residual/raw-gradient/applied-velocity cosine 为
  `0.92084/0.94507/0.94507`；controller 几乎不改变方向，故不授权 S3 可选控制。
- 机制排序：S1 residual similarity > S4 hard transition > S2 VJP > S3 controller。
  16列局部 JVP basis 从 probability 到 seismic 均为 effective rank 13/16，
  condition number 约 `1.16e3–4.43e3`，没有额外 L3 rank cliff，但已明显病态。

## 2026-08-07：Stage 7B structured hard geophysics

- 实现 mixed discrete/continuous object parameterization、hard-condition materialization、
  population beam 与 add/remove/translate/resize/rotate/change-shape/change-lithology
  trust region。接受与最终选择只读取 hard observed seismic RMSE；truth 指标均事后计算。
- 四控制为 correct/zero/shuffled/wrong-case，所有结果统一对 correct observation
  交叉评价。解析 cuboid correct arm 在不知道 truth indices 的情况下唯一选择
  candidate 4/6，hard attainment、hidden IoU/recall 均为1.0。
- 权威 native 结果：`reports/stage7_v1_final_v2`。seeds
  20260807/20260808/20260809 的 correct arm 在统一 correct 场下均严格rank 1；
  attainment `45.00/23.85/59.23%`，hidden IoU `0.969/0.914/0.987`，recall
  `0.996/0.916/0.993`，条件违例和错误岩性体积均为0。
- `stage7_v1`、`fix1`–`fix5` 与 partial `stage7_v1_final` 保留为开发证据，不是
  权威结论。最终完整测试 `186 passed, 13 warnings`；训练、fine-tune、LoRA 与
  broad controller sweep 均未运行。完成后按用户要求 STOP。
