# Flow2 Stage19 Codex 开发指令
## 主题：冻结地质 CFM + 小型地球物理 evidence-to-velocity residual adapter

> **阶段目标只有一个：**
>
> 在完全冻结现有 3D geological CFM 的前提下，训练一个约 5.4 万参数的小型 residual velocity adapter，使已经被 Stage17A 证明具有 case-specific 信息的连续 seismic evidence，能够被学习为**连贯的三维地质速度场修正**。
>
> 本阶段不再继续开发新的推理期 guidance 技巧，不加入 topology loss，不加入 hard-seismic loss，不修改 Flow 主网络，不处理多物性/多类别，不加入噪声或波子不确定性。
>
> **先解决最小、最清楚、已有实验结果充分支持的科学问题：**
>
> \[
> \boxed{
> q(\mathbf x)\;\xrightarrow{\text{learned small adapter}}\;\Delta v(\mathbf x,t)
> }
> \]
>
> 其中 `q(x)` 是 Stage17A 已验证的 continuous full-trace binary seismic evidence，`Δv` 是在冻结 Flow 速度场外部叠加的小型学习修正。

---

# 0. 开发基线与开工前审计

本规范基于 `MMMCCY/flow2` 当前 `main`：

```text
3134684cb9715a8cc58e3758d8770fde075cdd5d
commit message: stage18
```

Codex 修改任何文件前必须：

1. 阅读并遵守：
   - `project/geodata-3d-conditional/docs/AGENTS.md`
   - `project/geodata-3d-conditional/docs/PROJECT_BASELINE.md`
   - `project/geodata-3d-conditional/docs/EXPERIMENT_PROTOCOL.md`
   - `project/geodata-3d-conditional/docs/RESEARCH_GOAL.md`
   - `project/geodata-3d-conditional/docs/DEVELOPMENT_HANDOFF.md`
   - `project/FLOW2_STAGE18_CODEX_INSTRUCTIONS.md`
   - Stage17A / Stage17C formal outputs
   - Stage18 formal outputs
   - Stage6 adapter 源码与 oracle smoke 输出

2. 记录：

```bash
git status --short
git branch --show-current
git rev-parse HEAD
```

3. 不覆盖、不移动、不修改 Stage1–18 已有正式输出。

4. Stage19 必须建立新目录，旧实验只能作为 immutable reference。

5. 开工前先输出一次“实现计划 + 将复用的函数/文件 + 不会修改的内容”，确认后再实施。

---

# 1. 本阶段科学问题

当前实验已经不再支持“地球物理没有信息”或“冻结 Flow 完全不可控制”这两种解释。

本阶段只回答：

> **当 seismic evidence 已经具有可靠的目标空间信息时，是否可以仅通过一个小型可训练 residual adapter，学习出 evidence → Flow velocity correction 的映射，使冻结 geological CFM 在 held-out geology 上同时改善目标体恢复、hard seismic 一致性、体积和结构组织？**

形式化为：

\[
v_{\text{post}}(x_t,t,c,q)
=
v_\theta(x_t,t,c)
+
a_\phi(x_t,v_\theta,c,M_c,q,t)
\]

其中：

- `v_theta`：现有冻结 CFM；
- `a_phi`：Stage19 唯一训练模块；
- `c`：地表/钻井条件；
- `M_c`：hard-condition mask；
- `q`：Stage17A continuous seismic evidence；
- `theta` 全程冻结；
- 只优化 `phi`。

本阶段**不是**直接做完整 posterior sampler，也**不是** field-ready geophysical inversion。

---

# 2. 复用内容必须先审计：哪些可以复用，哪些不能直接复用

## 2.1 允许复用：Stage17A evidence pipeline

### 已验证成功的部分

权威结果：

`experiments/stage17_evidence_coupling_attribution/stage17a/formal_all5_v1/evaluation/per_case_evidence_metrics.csv`

五个 Full StructuralGeo cases：

| case | voxel AUPRC | prevalence | AP skill |
|---|---:|---:|---:|
| case01 | 0.65294 | 0.00078 | 0.65267 |
| case02 | 0.47005 | 0.07744 | 0.42557 |
| case03 | 0.59063 | 0.01153 | 0.58585 |
| case04 | 0.57151 | 0.07716 | 0.53568 |
| case05 | 0.56290 | 0.01859 | 0.55462 |

`specificity_deltas.csv` 的 diagonal AP-skill advantage：

```text
+0.66668
+0.37892
+0.58956
+0.50602
+0.55834
```

5/5 为正。

因此本阶段可以认为：

```text
Stage17A evidence 在当前 binary high-contrast synthetic regime 下具有足够信息和 case specificity。
```

### 必须原样复用的 Stage17A 设置

- full vertical trace；
- trace samples = `320`；
- no lateral filter；
- no threshold；
- two-pass linearized log-impedance + binary endpoint projection；
- `refinement_passes = 2`；
- `prior_relative_weight = 0.001`；
- `vertical_smoothness_relative_weight = 0.01`；
- target label = `9`；
- Stage15 binary acoustic upper bound；
- Stage4 full-cube noiseless inverse-crime seismic operator。

必须复用：

```text
experiments/stage15_binary_seismic_consensus/configs/binary_trace_boundary_inversion_v1.json
experiments/stage15_binary_seismic_consensus/configs/binary_acoustic_upper_bound_v1.json
experiments/stage4_seismic/configs/full_cube_noiseless_inverse_crime_v1.json
scripts/stage17/build_or_register_cases.py 中 observation 构造逻辑
scripts/stage17/run_multicase_trace_evidence.py 中 evidence inversion 逻辑
guidance/binary_trace_boundary.py
guidance/binary_seismic_inversion.py
guidance/seismic.py
```

### 明确不能继承的错误表述

`binary_acoustic_upper_bound_v1.json` 已明确：

```text
scientific_role = deliberately_binary_high_contrast_synthetic_upper_bound
realistic_multiclass_petrophysics = false
```

同时 seismic config 明确：

```text
noise = none
inverse_crime = true
measured_geophysics = false
```

因此 Stage19 **仍然是机制验证上限实验**。

禁止将 Stage19 结果表述为：

- realistic petrophysical inversion；
- field seismic inversion；
- full 15-class geophysical inversion；
- calibrated P(label9 | seismic)。

`q` 统一命名：

```text
continuous binary seismic evidence score
```

或：

```text
normalized binary-property evidence
```

---

## 2.2 允许复用：Stage6A residual adapter 架构与训练 loss

### 已验证成功的部分

`experiments/stage6_geo_adapter/runs/cond_generation_0/oracle_tiny_overfit_v1/sample_metrics.csv`

同一冻结 checkpoint、同一 sample：

```text
baseline target IoU     0.028596
adapter target IoU      0.511914
baseline precision      0.067484
adapter precision       0.833687
baseline recall         0.047279
adapter recall          0.570138
baseline global acc     0.587366
adapter global acc      0.745053
```

adapter 参数量：

```text
54,327
```

且：

```text
base_model_frozen = true
base_gradients_absent = true
base_model_tensor_hash_unchanged = true
```

这充分证明：

> 小型 external velocity adapter 有能力显著改变冻结 Flow 的 hard geology。

### 允许复用的部分

直接复用：

```text
guidance/residual_velocity_adapter.py
```

特别复用：

- `ResidualVelocityAdapter`
- `_TimeResidualBlock`
- `cap_residual_velocity`
- `class_balancing_weights`
- `residual_adapter_losses`
- `fixed_euler_adapter_sample`

### 不能直接复用的部分

Stage6A 是：

```text
legacy_truth_derived_oracle_engineering_only
truth_derived_oracle_input = true
publication_evidence = false
```

因此禁止：

- 使用 Stage6A truth acoustic 作为 Stage19 adapter 输入；
- 把 Stage6A 的成功结果当作泛化证据；
- 把 80-step same-case overfit training schedule 原样当成正式 Stage19 训练量；
- 使用 cond_generation_0 作为 Stage19 train/test case。

本阶段只复用**网络结构、速度残差约束和已验证 loss 形式**。

---

## 2.3 允许复用：Stage18 作为最佳 inference-only baseline

Stage18 已经证明在 5 个 cases 上：

- 5/5 volume error 比 Stage17 positive-only 改善；
- 4/5 target IoU 高于 Flow-only；
- 5/5 case-median hard seismic RMSE 低于 Flow-only；
- 但 component fragmentation 严重增加。

例如：

```text
case02 components: 152 -> 853
case03 components: 49  -> 328
case04 components: 26  -> 309
case05 components: 80  -> 406
```

因此：

- Stage18 是 Stage19 必须比较的 inference-only baseline；
- Stage18 不是 Stage19 的训练 target；
- 不把 Stage18 的 hand-designed property guidance 混进 adapter loss；
- 不再调 Stage18 alpha / cap / tau / threshold。

---

## 2.4 允许复用：Full StructuralGeo benchmark 的生成与 selection firewall

权威文件：

```text
experiments/full_structuralgeo_benchmark/configs/full_complexity_targeted_v1.json
experiments/full_structuralgeo_benchmark/benchmark_manifest.json
```

已有 benchmark：

- determinism passed；
- 5 accepted / 12 examined；
- selection 不使用 seismic / Flow / downstream metrics；
- fixed well XY 已冻结；
- complex geology eligibility 已冻结。

因此 Stage19 新 cohort 应复用同一生成 recipe、eligibility 和 fixed wells。

### 但不能直接使用原 5 cases 训练

原 5 cases 已参与 Stage10–18 多次方法开发和决策。

Stage19 必须生成一个**全新 cohort**。

原 `fullgeo_case01–05`：

```text
只能作为 historical sanity reference
不得进入 Stage19 train / validation / test
```

同时必须保留现有 benchmark 对 pretraining overlap 的诚实表述：

> 原始 streaming CFM 训练没有保存 seed/sample manifest，因此无法证明 Stage19 新 StructuralGeo cases 与历史 CFM pretraining 在 sample level 绝无重合。

这不阻止 Stage19，但最终报告必须注明。

---

# 3. 本阶段严格禁止扩大科学问题

Stage19 v1 禁止：

- alpha / mu / cap sweep；
- adapter width sweep；
- optimizer sweep；
- LR sweep；
- D-Flow；
- pCN / MCMC / SMC；
- new decoder / STE decoder；
- topology loss；
- persistent homology loss；
- SDF / StructuralGeo parametric inversion；
- raw seismic Transformer / CNN encoder；
- hard seismic training loss；
- joint fine-tune base Flow；
- 15-class geophysical evidence；
- gravity / magnetic data；
- seismic noise；
- wavelet mismatch；
- petrophysical uncertainty；
- thresholding `q`；
- case-specific score normalization；
- score sharpening / temperature calibration。

如果 Stage19 v1 失败，先停止并归因，不允许自动进入 sweep。

---

# 4. Stage19 方法总图

## 4.1 训练期

对每一个 synthetic geology：

```text
StructuralGeo truth m
        |
        |-- fixed surface/borehole conditions c
        |
        |-- binary acoustic forward
        v
synthetic seismic d
        |
        v
Stage17A full-trace inversion
        |
        v
continuous evidence q(x)
```

同时使用 truth 构造 CFM supervised velocity：

```text
X0 ~ N(0,I)
X1 = Embed(m)
(Xt, Vt) = model.interpolator.flow_objective(t, X0, X1)
```

冻结 base Flow：

```text
Vbase = frozen_model.net(Xt, conditioning, t)
```

训练：

```text
ΔV = adapter(Xt, Vbase, conditioning, condition_mask, q, t)
Vpred = Vbase + ΔV
```

目标：

```text
Vpred -> Vt
```

同时使用已有 all-class endpoint CE + Dice，避免只恢复 label9 而破坏其他类别。

---

## 4.2 推理期

测试时禁止读取 truth：

```text
conditions c
seismic d
    |
Stage17A evidence q
    |
X0 ~ N(0,I)
    |
32-step frozen Flow + learned adapter
    |
hard categorical geology
```

truth 只能由 evaluator 在推理完成后读取。

---

# 5. 新 StructuralGeo cohort：一次生成、一次冻结，不按结果挑 case

## 5.1 cohort 数量

固定：

```text
TRAIN = 64 independent geology cases
VAL   = 8 independent geology cases
TEST  = 12 independent geology cases
```

总计：

```text
84 independent geological histories
```

这不是超参数 sweep，而是 supervised training cohort + held-out evaluation cohort。

---

## 5.2 root-seed ranges

使用三个完全分离的 root-seed ranges：

```text
TRAIN candidate seeds:
start = 220260001
step  = 1
max candidates = 256
accept first 64 eligible

VAL candidate seeds:
start = 220270001
step  = 1
max candidates = 128
accept first 8 eligible

TEST candidate seeds:
start = 220280001
step  = 1
max candidates = 128
accept first 12 eligible
```

如果某个 range 在 maximum budget 内没有满足目标 accepted count：

```text
STOP
```

禁止：

- 扩大/修改 eligibility；
- 根据 seismic/evidence/Flow 结果补选；
- 根据 target volume/centroid“挑好看”的 case。

---

## 5.3 StructuralGeo recipe 原样继承

复用：

```json
{
  "bounds": [[-1920, 1920], [-1920, 1920], [-1920, 1920]],
  "resolution": [64, 64, 64],
  "markov_matrix": "packaged_default",
  "height_tracking": true,
  "normalize": true,
  "fill_nans": true,
  "raw_label_range": [-1, 13]
}
```

eligibility 原样继承：

```text
raw_label9_voxels_positive = true
hidden_label9_voxels_positive = true
base_strata_history_required = true
fold_or_fault_event_required = true
label9_producing_intrusion_event_required = true
```

---

## 5.4 fixed well XY 原样继承

```text
[8,46]
[9,5]
[10,24]
[27,17]
[35,26]
[39,59]
[44,60]
[48,6]
[57,32]
```

必须复用已有 condition 构造逻辑，不重新随机设计井位。

---

## 5.5 split firewall

在任何 adapter training 开始前，必须生成并 hash：

```text
cohort_manifest.json
train_registry.json
val_registry.json
test_registry.json
```

并记录每个 case：

- `case_id`
- root seed
- markov sequence
- event subtypes
- truth SHA256
- condition values SHA256
- condition mask SHA256
- target voxel count
- hidden target voxel count

`TEST` registry 一旦生成后不得改变。

---

# 6. Synthetic observation 与 Stage17A evidence 构造

## 6.1 observation

必须复用 Stage17 observation builder 的逻辑：

```python
binary_truth = ((truth == 9) & subsurface).float()
```

support：

```text
columnwise_fill_below_highest_nonair_v1
```

enclosed air acoustic：

```text
binary_background_endpoint
```

binary fixed values 仅由 hard conditions 构造。

使用：

```text
binary_occupancy_to_acoustic
seismic_operator_from_config
build_seismic_observation
```

要求 forward closure：

```text
max_abs_error <= 1e-7
```

---

## 6.2 seismic config 原样冻结

```text
grid = 64 x 64 x 64
cell size = [100m, 100m, 50m]
trace samples = 320
dt = 8 ms
Ricker = 25 Hz
duration = 128 ms
all XY columns
noise = none
```

不加噪声。

---

## 6.3 evidence inversion 原样冻结

Stage17A：

```text
method = complete_trace_two_pass_linearized_log_impedance_with_binary_endpoint_projection_v1
refinement_passes = 2
prior_relative_weight = 0.001
vertical_smoothness_relative_weight = 0.01
full_vertical_trace_used = true
lateral_filter_used = false
threshold_sweep = false
```

输出唯一正式 adapter geophysics channel：

```text
binary_impedance_score.pt
```

Stage19 v1：

```text
geophysics_channels = 1
q_input = binary_impedance_score * subsurface_mask.float()
```

不加入：

- boundary-strength second channel；
- raw seismic；
- acoustic reconstruction；
- thresholded core。

---

# 7. Evidence 复用前的一次性 gate：只确认没有跨 cohort 崩溃，不调参数

用户要求复用成功案例前确认其成功不是偶然，因此必须有一个非常小的、预注册的 reuse gate。

只在 `VAL=8` cases 上、训练前执行一次 retrospective evidence audit。

不筛 case、不调 inversion 参数。

必须计算：

- voxel AUPRC；
- prevalence；
- AP skill；
- correct-case vs wrong-case AP-skill delta。

Stage19 evidence gate：

```text
至少 6/8 VAL cases: AP skill > 0
VAL group median AP skill >= 0.30
至少 6/8 VAL cases: diagonal AP-skill > off-diagonal median
```

如果失败：

```text
STOP_BEFORE_ADAPTER_TRAINING
```

禁止根据失败结果重新调：

- inversion weights；
- refinement passes；
- threshold；
- lateral smoothing。

这一 gate 只验证：Stage17A 的已成功 evidence pipeline 是否仍工作在新 cohort。

---

# 8. Base Flow：完全冻结，原样加载

checkpoint 必须为当前已使用 checkpoint：

```text
project/geodata-3d-conditional/demo_model/conditional-weights.ckpt
SHA256 = 561e94bfda770ec41fc4cbed43436a7e2130eef5dfb7e5d666fcefc0724ff94c
```

加载策略：

```python
runtime.load_model_with_weight_policy(..., weight_source="ema")
```

必须满足：

```text
EMA applied = true
embedding.weight remains frozen raw checkpoint tensor
all base parameters requires_grad = false
base parameters receive no gradients
base tensor hash before == after training
```

禁止修改：

- `Geo3DStochInterp`
- `Unet3DCondV3`
- categorical embedding
- decoder
- checkpoint
- EMA policy。

---

# 9. Adapter 架构：只做一个必要修改

直接复用：

```text
guidance/residual_velocity_adapter.py
```

Stage19 config 固定：

```json
{
  "base_width": 12,
  "dilations": [1, 2, 4, 1],
  "geophysics_channels": 1,
  "max_parameters": 100000,
  "max_residual_ratio": 0.25,
  "final_layer_zero_initialized": true
}
```

解释：

- Stage6A `base_width=12` 已证明有足够控制能力；
- dilation `[1,2,4,1]` 已验证；
- Stage19 `q` 只有一个 channel，因此从 2 改为 1；
- 不扩大网络；
- output conv 必须 zero-init；
- correction 在 hard-condition mask 中必须严格为 0。

预计参数量仍约 `5.4e4`，必须：

```text
actual_parameter_count < 100000
```

---

# 10. Adapter 训练数据与数学目标

## 10.1 每个训练 sample

读取：

```text
truth
condition_values
condition_mask
subsurface_mask
binary_impedance_score
```

构造：

```python
X1 = model.embed(truth)
embedded_conditions = model.embed(condition_values)
conditioning = embedded_conditions * condition_mask.expand_as(embedded_conditions)
```

注意：

- conditioning 必须来自 `condition_values`；
- 不允许把 full truth embedding 作为 conditioning；
- truth 只用于 supervised target velocity 和 training loss。

---

## 10.2 CFM supervised state

严格复用现有 stochastic interpolator：

```python
Xt, Vt = model.interpolator.flow_objective(T, X0, X1)
```

训练时：

```text
X0 ~ standard Gaussian
```

然后对 `Xt` 强制 hard conditions：

```python
Xt = project_conditions(Xt, embedded_conditions, condition_mask)
```

base velocity：

```python
with torch.no_grad():
    Vbase = model.net(Xt, conditioning, T)
```

`Vbase` 必须 detach。

---

## 10.3 adapter 输入

```python
q = binary_impedance_score * subsurface_mask.float()

raw_delta = adapter(
    state=Xt,
    base_velocity=Vbase,
    conditioning=conditioning,
    condition_mask=condition_mask,
    geophysics=q,
    time=T,
)
```

然后：

```python
delta, used_ratio = cap_residual_velocity(
    raw_delta,
    Vbase,
    condition_mask,
    max_ratio=0.25,
)
```

不额外乘 Stage17/18 guidance schedule。

---

# 11. Loss：Stage19 v1 不发明新 loss

直接复用：

```text
residual_adapter_losses(...)
```

固定：

```json
{
  "flow_weight": 1.0,
  "cross_entropy_weight": 0.25,
  "dice_weight": 0.25,
  "residual_regularizer_weight": 0.0001,
  "logit_temperature": 0.1
}
```

即：

\[
L
=
L_{flow}
+0.25L_{CE}
+0.25L_{Dice}
+10^{-4}L_{residual}
\]

其中：

- `L_flow`：学习纠正 base velocity 到 CFM truth velocity；
- CE + Dice：让一步 endpoint 预测保持 categorical geology；
- residual regularizer：避免 adapter 覆盖 prior；
- active region：unconditioned non-air subsurface。

继续复用：

```text
class_balancing_weights
```

每个 geology case 按自身 active class count 计算 class weights。

### Stage19 v1 明确不加入

- label9-only Dice；
- topology loss；
- component loss；
- seismic loss；
- Stage18 property loss；
- oracle probability loss。

原因：首先只回答“learned coupling 本身是否足够”。

---

# 12. 正式训练 schedule：固定一次，不 sweep

## 12.1 optimizer

```text
optimizer = AdamW
learning_rate = 0.002
weight_decay = 0.0001
gradient_clip_norm = 1.0
```

复用 Stage6A 已稳定的 optimizer 超参数。

不使用 scheduler。

不比较 Adam / LBFGS。

---

## 12.2 interpolation times

复用 Stage6A 已验证代表性 times：

```text
T = [0.2, 0.4, 0.6, 0.8]
```

每个 train geology 每个 epoch 都覆盖这 4 个 T。

不使用 time sweep。

---

## 12.3 epochs / optimizer steps

固定：

```text
TRAIN cases = 64
T per case = 4
optimizer batch size = 1 geology-state
updates per epoch = 64 * 4 = 256
epochs = 4
total optimizer updates = 1024
```

这是 Stage19 唯一正式训练 schedule。

### noise policy

每一次 `(epoch, case, T)` 使用一个新的 Gaussian `X0`。

使用单个 CPU generator：

```text
training_state_generator_seed = 6100
```

顺序必须 deterministic，并记录每个 state 的：

```text
initial_noise_sha256
```

训练 case 顺序每 epoch 可 deterministic shuffle：

```text
shuffle_seed = 6201 + epoch_index
```

---

## 12.4 adapter initialization

```text
adapter initialization seed = 6200
final output layer = exact zeros
```

因此 step 0 adapter 必须等价于 frozen Flow。

---

## 12.5 validation

每 epoch 完成后，在 8 个 VAL cases 上计算同一个 supervised objective。

VAL 使用固定：

```text
T = [0.2,0.4,0.6,0.8]
1 deterministic X0 per case/T
```

只记录：

- val total loss；
- val flow loss；
- val CE；
- val Dice；
- val one-step endpoint accuracy。

**不 early-stop，不从 4 个 epoch 中挑 best checkpoint。**

Stage19 formal checkpoint 固定使用：

```text
epoch 4 final adapter
```

这样避免 validation 变成新的调参循环。

如果训练出现：

- NaN / inf；
- base hash 变化；
- adapter zero gradient；
- condition projection failure；

视为工程错误，修复代码后重跑。

如果 loss 有限但科学效果差，不允许据此改超参数重跑 v1。

---

# 13. 训练期间必须保存

```text
adapter_config.json
adapter_checkpoint.pt
adapter_checkpoint.sha256
training_manifest.json
training_trace.csv
validation_trace.csv
training_case_manifest.csv
```

`training_trace.csv` 至少包括：

- epoch；
- case id；
- T；
- X0 SHA；
- total loss；
- flow loss；
- CE；
- Dice；
- residual regularizer；
- endpoint accuracy；
- used residual ratio；
- gradient norm。

training manifest 必须记录：

```text
base model hash before/after
base grads absent
adapter parameter count
cohort registry hashes
evidence registry hashes
all frozen config hashes
```

---

# 14. Test inference：只运行必要的五个 arm

TEST = `12 cases`。

每个 case 固定 source seeds：

```text
42
142
242
```

每个 arm 同 case / seed 必须使用完全相同 `X0`。

固定：

```text
n_steps = 32
integrator = fixed Euler midpoint
adapter_scale = 1.0
max_residual_ratio = 0.25
```

## 14.1 Arm A — FLOW_ONLY

使用同一个 `fixed_euler_adapter_sample`：

```text
adapter_scale = 0.0
```

这应与 frozen Flow baseline 精确一致。

先在 smoke 中做 hash equivalence；formal 不需要额外两套 Flow runner。

---

## 14.2 Arm B — STAGE18_CONTINUOUS_TARGET

在新 TEST cases 上复用 Stage18 最终 inference-only semantics：

```python
target_properties = score
confidence = score * free_subsurface.float()
```

并复用 Stage17C/18 固定：

```text
alpha = 0.25
max_guidance_ratio = 0.25
tau_start = 0.5
tau_end = 0.1
tau_schedule = cosine
guidance_start = 0.25
guidance_schedule = windowed_sine
grad_clip_norm = 1.0
guidance_scaling_mode = reference_norm_relative_v2
property_sigmas = [0.0, 1.5, 3.0]
property_scale_weights = [0.5, 0.3, 0.2]
property_loss_mode = matched_multiscale_normalized_mse_v1
```

不 sweep。

此 arm 的作用只有一个：

```text
证明 learned adapter 是否比当前最佳 inference-only coupling 更好地组织结构。
```

---

## 14.3 Arm C — ADAPTER_CORRECT

```text
adapter = epoch4 final checkpoint
q = current test case correct evidence
adapter_scale = 1.0
```

这是唯一主实验 arm。

---

## 14.4 Arm D — ADAPTER_ZERO

```text
q = zeros_like(correct_q)
```

其他全部相同。

目的：检测 adapter 是否只学会了一个 generic geology correction，而没有使用 geophysical evidence。

---

## 14.5 Arm E — ADAPTER_WRONG_CASE

对 TEST case `i`：

```text
q_wrong = evidence of TEST case (i+1) mod 12
```

wrong-case mapping 必须在 config 中预先冻结。

目的：验证 case specificity。

不运行 shuffled arm，避免扩大 formal matrix。

---

# 15. Test truth firewall

`run_inference.py` 禁止加载：

```text
true_model.pt
binary_truth.pt
truth component records
truth history / event parameters
```

它只能读取：

- condition values；
- condition mask；
- subsurface support；
- observed seismic / evidence；
- base checkpoint；
- adapter checkpoint；
- frozen config。

truth 只能由：

```text
evaluate_stage19.py
```

在全部 inference outputs 写盘、hash 完成之后读取。

---

# 16. 必须评估的指标：保持简单

主统计单位：

```text
independent geology case
```

3 source seeds 是 case 内 stochastic replicates。

每个 case 先对 3 seeds 取 median，再跨 12 cases 汇总。

## 16.1 target label9

必须报告：

- IoU；
- precision；
- recall；
- TP / FP / FN；
- predicted target volume；
- absolute volume-error fraction；
- centroid distance。

---

## 16.2 global geology

必须报告：

- voxel accuracy；
- truth-present mean IoU；
- per-class IoU。

目标是防止只修 label9、破坏背景 geology。

---

## 16.3 topology / structure

Stage19 v1 只评估，不训练 topology。

只使用现有简单指标：

- target connected component count；
- largest-component fraction；
- top-4 component mass fraction；
- top-8 component mass fraction；
- tiny component mass fraction `<=5 voxels`。

不要引入新的 persistent homology package。

---

## 16.4 hard seismic

所有 hard decoded geology 必须使用同一个 frozen Stage17 binary acoustic/seismic forward model重新计算：

```text
hard seismic MSE
hard seismic RMSE
```

禁止使用 training loss 替代 hard seismic evaluation。

---

## 16.5 hard conditions

必须：

```text
condition_violation_count = 0
```

对所有 arm / case / seed 成立。

---

# 17. Stage19 简单、预注册的成功标准

不构造复杂 composite score。

## 17.1 工程 gate：必须全部通过

```text
base checkpoint hash correct
EMA applied
base tensor hash unchanged after training
base gradients absent
adapter parameter count < 100000
adapter scale=0 reproduces FLOW_ONLY
all condition violations = 0
all case/seed pairings valid
all evidence/input hashes valid
```

任何一项失败：

```text
ENGINEERING_FAIL
```

不得做科学解释。

---

## 17.2 Evidence reuse gate

见第 7 节。

失败：

```text
EVIDENCE_REUSE_NOT_VALIDATED
```

不进入 adapter training。

---

## 17.3 Learned coupling 主 gate

对 `ADAPTER_CORRECT` vs `FLOW_ONLY`：

必须同时满足：

```text
>= 9/12 test cases 的 case-median target IoU 提升
cross-case median delta target IoU > +0.05
```

这是最核心 gate。

---

## 17.4 Evidence specificity gate

`ADAPTER_CORRECT` 必须：

```text
cross-case median target IoU > ADAPTER_ZERO
cross-case median target IoU > ADAPTER_WRONG_CASE
```

并且：

```text
>= 8/12 cases correct > zero
>= 8/12 cases correct > wrong-case
```

如果失败：

```text
ADAPTER_NOT_USING_CASE_SPECIFIC_EVIDENCE
```

即使 correct arm 的 IoU 比 Flow 高，也不能宣称 geophysical conditioning 成功。

---

## 17.5 Hard physics gate

`ADAPTER_CORRECT` vs `FLOW_ONLY`：

```text
>= 8/12 cases hard seismic RMSE 降低
cross-case median delta hard seismic RMSE < 0
```

不要求每个 case 都优于 Stage18，但必须单独报告：

```text
ADAPTER_CORRECT vs STAGE18_CONTINUOUS_TARGET
```

---

## 17.6 Volume gate

要求：

```text
cross-case median absolute volume-error fraction
ADAPTER_CORRECT < FLOW_ONLY
```

同时和 Stage18 单独对比。

如果 adapter geometry 很好但 volume 再次出现 Stage17 式明显膨胀，则不能判为完整 Stage19 success。

---

## 17.7 Structure gate

Stage19 的目标不是要求 component count 完全匹配 truth，而是首先证明 learned coupling 不再产生 Stage18 式强碎片化。

要求：

```text
cross-case median target component count:
ADAPTER_CORRECT < STAGE18_CONTINUOUS_TARGET
```

同时：

```text
cross-case median top8 component mass fraction:
ADAPTER_CORRECT > STAGE18_CONTINUOUS_TARGET
```

这两个简单条件足够判断结构组织是否改善。

---

## 17.8 Global geology preservation gate

要求：

```text
cross-case median truth-present mean IoU delta
ADAPTER_CORRECT - FLOW_ONLY >= -0.01
```

即允许非常小的 trade-off，但禁止为 label9 恢复明显破坏全局 geology。

---

# 18. 最终结果分类

## `LEARNED_EVIDENCE_ADAPTER_VALIDATED`

同时通过：

- engineering gate；
- evidence reuse gate；
- learned coupling gate；
- specificity gate；
- hard physics gate；
- volume gate；
- structure gate；
- global geology preservation gate。

解释：

> 冻结 geological CFM 不需要重新训练即可通过一个很小的 learned evidence-to-velocity adapter 学会利用 seismic-derived 3D evidence，并在 held-out StructuralGeo cases 上产生比 Flow-only 和 hand-designed inference guidance 更有组织的 geologically meaningful update。

---

## `LEARNED_COUPLING_POSITIVE_BUT_STRUCTURE_UNRESOLVED`

如果：

- IoU / specificity / hard physics / volume 通过；
- 但 structure gate 未通过。

解释：

> learned coupling 已成立，但结构连通组织仍不足。

下一阶段才允许讨论：

- topology regularization；
- SDF / structural latent；
- StructuralGeo geometry prior。

---

## `ADAPTER_CAPACITY_OR_TRAINING_INSUFFICIENT`

如果：

- evidence reuse gate 通过；
- specificity 有效；
- 但 learned coupling gate 失败。

先停止。

不得自动加宽 adapter。

下一轮研究决策再判断：

- 增加 adapter capacity；或
- joint fine-tune base Flow。

---

## `ADAPTER_NOT_USING_CASE_SPECIFIC_EVIDENCE`

如果 correct 与 zero/wrong-case 无区别。

解释：

> adapter 可能只学习 generic correction，不能宣称 geophysics-conditioned generation。

停止，不做 topology/physics 扩展。

---

# 19. 为什么 Stage19 v1 不加入 hard-seismic training loss

这是刻意设计，而不是遗漏。

当前最清楚的 supervised target 已经存在：

\[
V_t = \text{CFM flow objective}(X_0,X_1,t)
\]

Stage19 想知道的是：

> `q` 能否告诉一个小网络“冻结 base velocity 在哪里应该怎样改”。

如果同时加入：

- CFM velocity loss；
- categorical loss；
- seismic loss；
- topology loss；

一旦成功就无法知道真正起作用的是哪一个。

因此 v1 先使用：

```text
现成的 supervised CFM target + 现成 categorical endpoint loss
```

hard seismic 只做 independent final evaluation。

如果 Stage19 geometry 明显改善但 hard seismic 不改善，下一阶段再授权增加 physics-consistency loss。

---

# 20. 为什么 Stage19 v1 不加入 topology loss

Stage15 oracle 已经证明 frozen Flow 能生成：

- separated five bodies；
- solid body；
- ring topology。

因此 Stage18 fragmentation 更可能是 coupling 方式问题，而不是 prior 完全没有结构能力。

Stage19 先验证：

```text
learned velocity response 本身是否已经能恢复结构组织
```

如果答案是能，则不需要 topology machinery。

只有当：

```text
learned coupling / hard physics / volume 都成功
但 components 仍严重碎片化
```

才有充分理由进入 topology/SDF 阶段。

---

# 21. 软件开发建议目录

```text
project/geodata-3d-conditional/
├── docs/
│   └── STAGE19_LEARNED_EVIDENCE_ADAPTER_SPEC.md
├── experiments/
│   └── stage19_learned_evidence_adapter/
│       ├── README.md
│       ├── configs/
│       │   ├── cohort_v1.json
│       │   ├── evidence_v1.json
│       │   ├── training_v1.json
│       │   └── inference_v1.json
│       ├── cohort/
│       │   ├── cohort_manifest.json
│       │   ├── train_registry.json
│       │   ├── val_registry.json
│       │   └── test_registry.json
│       ├── observations/
│       ├── evidence/
│       ├── evidence_audit/
│       ├── checkpoints/
│       ├── smoke/
│       ├── formal/
│       └── reports/
├── scripts/
│   └── stage19/
│       ├── build_cohort.py
│       ├── build_observations.py
│       ├── run_evidence.py
│       ├── audit_evidence.py
│       ├── train_adapter.py
│       ├── run_inference.py
│       └── evaluate.py
└── tests/
    └── test_stage19_learned_evidence_adapter.py
```

职责必须清楚，不把 training / inference / evaluation 混成一个脚本。

---

# 22. 代码复用原则

Codex 应优先 import / refactor 已有函数，不复制大段旧实现。

## observation/evidence

优先复用：

```text
scripts/stage17/build_or_register_cases.py
scripts/stage17/run_multicase_trace_evidence.py
guidance/binary_seismic_inversion.py
guidance/binary_trace_boundary.py
guidance/seismic.py
scripts/stage15/common.py
scripts/stage17/common.py
```

Stage17 scripts 本身存在固定 `len(cases)==5` 的 formal guard。

不要修改旧 Stage17 guard。

Stage19 新 runner 应复用底层函数，并支持 Stage19 新 registry size。

## adapter

直接 import：

```text
guidance/residual_velocity_adapter.py
```

若无需 bug fix，不新增第二套 adapter 实现。

## runtime/checkpoint

复用：

```text
inference_runtime.py
```

## evaluation

优先复用：

```text
guidance/property_evaluation.py
guidance/probability_evaluation.py
Stage18 hard seismic evaluator
```

不要重新定义 IoU / component / seismic RMSE 口径。

---

# 23. 必要 CPU/unit tests：只测工程不做科学 tuning

至少覆盖：

1. Stage19 config frozen-schema validation；
2. train/val/test root-seed ranges 不重叠；
3. historical `fullgeo_case01–05` 不在任何 Stage19 split；
4. Stage17 evidence parameters byte-for-byte/field-for-field frozen；
5. no thresholding of evidence；
6. adapter `geophysics_channels=1` shape correctness；
7. output conv exact zero initialization；
8. adapter parameter count <100000；
9. correction = 0 inside hard-condition mask；
10. residual cap <=0.25；
11. base parameters `requires_grad=False`；
12. backward 后 base `.grad is None`；
13. scale=0 sampler 与 Flow-only exact/hash equivalent；
14. test inference runner static truth firewall；
15. wrong-case mapping fixed and never self-maps；
16. output dir non-empty refusal；
17. condition projection exact；
18. hard seismic evaluator uses hard `decoded == 9` occupancy；
19. test evaluator case-first median aggregation。

不要为了测试引入新的方法参数。

---

# 24. CUDA smoke：只做一次工程验证

在 formal training 前允许一个 smoke。

smoke 只使用：

```text
2 TRAIN cases
1 VAL case
4 interpolation times
2 optimizer steps or a minimal full forward/backward sequence
1 inference case
1 source seed = 42
```

smoke 目的：

- GPU memory；
- base frozen；
- adapter gradients nonzero；
- q channel 接入正确；
- sampling 32 steps 可运行；
- condition exact；
- files/hashes 正常。

smoke 不得：

- 修改 formal LR；
- 修改 adapter width；
- 修改 residual cap；
- 根据 IoU 选择参数；
- 被写入科学报告。

---

# 25. Formal 执行顺序：禁止跳步

```text
Step 1  build new StructuralGeo cohort
Step 2  freeze train/val/test registries + hashes
Step 3  build binary seismic observations
Step 4  run frozen Stage17A evidence inversion
Step 5  run VAL-only evidence reuse audit
Step 6  evidence gate PASS -> continue
Step 7  run one Stage19 formal adapter training
Step 8  freeze epoch4 adapter checkpoint + hash
Step 9  run 12 test cases x 3 source seeds x 5 arms
Step 10 freeze all inference outputs + hashes
Step 11 evaluator reads test truth
Step 12 generate case-first summaries and decision
Step 13 update DEVELOPMENT_HANDOFF
Step 14 STOP
```

任何科学参数必须在 Step 1 前写进 frozen config。

---

# 26. Formal output 数量

Test：

```text
12 cases
x 3 source seeds
x 5 arms
= 180 final hard geology outputs
```

这是完整 formal matrix，不再增加 arm。

训练：

```text
64 cases x 4 times x 4 epochs
= 1024 optimizer updates
```

---

# 27. 推荐 frozen training config

建议直接实现为：

```json
{
  "schema": "stage19_learned_evidence_adapter_training_v1",
  "status": "frozen_before_cuda_training",
  "target_label": 9,
  "base_model": {
    "weight_source": "ema",
    "checkpoint_sha256": "561e94bfda770ec41fc4cbed43436a7e2130eef5dfb7e5d666fcefc0724ff94c",
    "freeze_all_parameters": true
  },
  "evidence": {
    "type": "stage17a_binary_impedance_score",
    "channels": 1,
    "input_policy": "score_times_subsurface_no_threshold"
  },
  "adapter": {
    "base_width": 12,
    "dilations": [1, 2, 4, 1],
    "geophysics_channels": 1,
    "max_parameters": 100000,
    "max_residual_ratio": 0.25,
    "final_layer_zero_initialized": true
  },
  "training": {
    "adapter_seed": 6200,
    "state_generator_seed": 6100,
    "epochs": 4,
    "times": [0.2, 0.4, 0.6, 0.8],
    "batch_size": 1,
    "optimizer": "AdamW",
    "learning_rate": 0.002,
    "weight_decay": 0.0001,
    "gradient_clip_norm": 1.0,
    "scheduler": null,
    "flow_weight": 1.0,
    "cross_entropy_weight": 0.25,
    "dice_weight": 0.25,
    "residual_regularizer_weight": 0.0001,
    "logit_temperature": 0.1,
    "early_stopping": false,
    "checkpoint_selection": "epoch4_final"
  },
  "sampling": {
    "n_steps": 32,
    "adapter_scale": 1.0,
    "source_seeds": [42, 142, 242]
  },
  "parameter_sweep": false,
  "topology_loss": false,
  "physics_training_loss": false,
  "base_finetuning": false
}
```

---

# 28. Codex 实现时最重要的 training pseudocode

```python
# truth is allowed here because this is supervised synthetic training.
truth = load_truth(case)
condition_values = load_condition_values(case)
condition_mask = load_condition_mask(case).bool()
subsurface = load_subsurface(case).bool()
score = load_binary_impedance_score(case).float()

with torch.no_grad():
    X1 = model.embed(truth)
    embedded_conditions = model.embed(condition_values)
    conditioning = embedded_conditions * condition_mask.expand_as(embedded_conditions)

X0 = torch.randn(X1.shape, generator=cpu_training_generator).to(device)
T = torch.tensor([time_value], device=device, dtype=X1.dtype)

with torch.no_grad():
    Xt, Vt = model.interpolator.flow_objective(T, X0, X1)
    Xt = project_conditions(Xt, embedded_conditions, condition_mask)
    Vbase = model.net(Xt, conditioning, T)

q = (score * subsurface.float()).to(device=device, dtype=Xt.dtype)

raw_delta = adapter(
    Xt,
    Vbase,
    conditioning,
    condition_mask,
    q,
    T,
)

delta, used_ratio = cap_residual_velocity(
    raw_delta,
    Vbase,
    condition_mask,
    max_ratio=0.25,
)

class_weights = class_balancing_weights(
    truth,
    (~condition_mask) & (truth != -1),
    model.num_categories,
)

loss, diagnostics = residual_adapter_losses(
    state=Xt,
    target_velocity=Vt,
    base_velocity=Vbase,
    correction=delta,
    truth=truth,
    condition_mask=condition_mask,
    embedding_weight=model.embedding.weight,
    time=T,
    class_weights=class_weights,
    logit_temperature=0.1,
    flow_weight=1.0,
    cross_entropy_weight=0.25,
    dice_weight=0.25,
    residual_regularizer_weight=0.0001,
)

optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
optimizer.step()
```

必须保证：

```text
Xt / Vt / Vbase 不通过 base model 建图反传
only adapter parameters update
```

---

# 29. 不允许的“看似合理优化”

Codex 不得自行做以下改动：

- 因为 `q` 稀疏就加 Gaussian blur；
- 因为 component 多就加 morphology；
- 因为 loss 不降就改 LR；
- 因为 target 小就复制 target-specific patches；
- 因为 case01 类似的小体难就剔除小 target case；
- 因为 wrong-case control 影响大就改变 mapping；
- 因为 Stage18 baseline 很强就改 adapter cap；
- 因为 hard seismic 不好就加 seismic loss；
- 因为 test IoU 不高就从 epoch1–4 挑最好 test checkpoint。

这些都会破坏 Stage19 的单一科学问题。

---

# 30. Stage19 后的唯一允许分支

完成后必须 STOP。

根据正式结果，只允许提出下一阶段建议，不自动开发。

## 如果 Stage19 全面成功

下一步优先：

```text
保持 frozen Flow + adapter 架构不变
只提高 geophysical realism
```

例如：

1. binary high-contrast → multiclass background；
2. noiseless → noisy seismic；
3. fixed petrophysics → uncertain petrophysics。

每次只改变一个问题。

## 如果 coupling 成功但 topology 仍失败

下一阶段才讨论：

- simple topology regularization；或
- continuous SDF / structural latent。

## 如果 correct evidence 不优于 zero/wrong

说明 learned geophysical conditioning 未成立。

停止，不讨论 topology。

## 如果 evidence gate 成功但 adapter 仍无法改善 geometry

才讨论：

```text
joint fine-tune base Flow + adapter
```

而不是回到大量 inference-time guidance sweep。

---

# 31. 最终 Codex 交付清单

Codex 完成后必须报告：

1. 开工时 repository audit；
2. starting git commit；
3. 变更文件清单；
4. 新 cohort 生成规则；
5. train/val/test registry + SHA；
6. evidence frozen configs + hashes；
7. VAL evidence reuse gate 结果；
8. adapter 参数量；
9. base model frozen/hash audit；
10. exact training command；
11. total optimizer steps；
12. training/validation traces；
13. epoch4 adapter checkpoint SHA；
14. exact inference command；
15. 180 formal outputs manifest；
16. case-first metrics CSV；
17. Flow / Stage18 / adapter-correct / zero / wrong-case 比较；
18. hard seismic comparison；
19. volume comparison；
20. component/top8 structural comparison；
21. condition violation audit；
22. specificity gate；
23. 最终 Stage19 classification；
24. “已证明 / 未证明”；
25. 更新 `docs/DEVELOPMENT_HANDOFF.md`；
26. 完成后停止，不继续 Stage20。

---

# 32. Stage19 最核心研发纪律

本阶段不要尝试一次解决所有地球物理联合建模问题。

我们已经分别证明：

```text
1. frozen Flow 可以表达目标 geology；
2. seismic evidence 在当前 upper-bound regime 下具有 spatial information；
3. hand-designed guidance 可以让 hard geology 发生大幅改变；
4. tiny adapter 可以强力控制 frozen Flow；
```

所以 Stage19 只连接最后缺失的一段：

\[
\boxed{
\text{seismic evidence}
\rightarrow
\text{learned geological velocity correction}
}
\]

如果这一段成立，后面的研究可以逐步提高物理真实性。

如果这一段在最有利的 binary/noiseless synthetic setting 下都不能成立，就没有必要先增加更多地球物理复杂度。

**简单先行，一次只回答一个问题。**
