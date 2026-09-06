# Flow2 Stage20 Codex 开发方案
## 主题：Phase4c 多类别地震观测 + Phase5a 连续阻抗反演 + Stage19R learned residual adapter

> **Stage20 的唯一核心科学问题：**
>
> 在不使用 target-specific binary probability/score 作为生成证据的情况下，能否把 **多类别地质模型产生的 acquisition-domain seismic** 先转换为一个 **连续三维 log-impedance evidence**，再由已经验证有效的 **小型 evidence-to-velocity residual adapter** 将该连续物性证据转化为更正确的三维 categorical geology？
>
> 本阶段不是重新设计 Flow，不是重新设计 seismic forward，不是重新设计 Phase5a inversion，不是继续调 inference-time guidance，也不是 topology method development。
>
> **只允许改变 evidence 语义：**
>
> ```text
> Stage19R: target-specific binary impedance score q(x)
> Stage20 : continuous inverted log-impedance q_Z(x)
> ```
>
> 其余已成功或已冻结组件尽量直接复用。

---

# 0. 开发基线与已有证据

## 0.1 预期代码基线

本开发方案按以下当前仓库状态编写：

```text
repository: MMMCCY/flow2
branch: main
expected commit: 2284d7235cd62295d741c3982eb5063fe6577ac6
commit message: stage19r
```

Codex 开始修改前必须：

```bash
git status --short
git rev-parse HEAD
```

若 HEAD 已变化，不要直接假定路径/接口不变；先重新检查本文件列出的复用对象和 schema。不得覆盖用户未提交修改。

## 0.2 Stage19R 已经验证的事实

Stage19R 正式结果已经得到：

```text
LEARNED_EVIDENCE_ADAPTER_VALIDATED
```

正式 TEST：

```text
12 cases × 3 source seeds × 5 arms = 180 outputs
```

关键结果：

```text
correct adapter target-IoU improvement vs Flow: 12/12
correct > zero adapter:                         12/12
correct > wrong-case adapter:                  12/12
hard seismic RMSE improvement:                 12/12
median target IoU:
    Flow-only        = 0.169669
    correct adapter  = 0.410565
median Δ target IoU  = +0.236709
median hard seismic ΔRMSE = -0.008943
median truth-present mIoU Δ = +0.039260
```

因此 Stage20 **不再验证“adapter 是否具有控制能力”**；Stage20 只验证：

```text
continuous seismic-inversion evidence
          ↓
learned adapter
          ↓
hard categorical geology
```

是否仍然成立。

## 0.3 Phase4c 已经验证的事实

Phase4c 已经证明：

1. 多类别 geology → density/Vp → convolutional seismic 的 forward operator 正常；
2. acquisition-domain seismic gradient 能改变 hard labels；
3. 但直接降低 seismic residual 并不能可靠恢复 label9 几何；
4. 因此 **Phase4c forward/evaluator 可以复用，direct trajectory gradient guidance 不复用**。

## 0.4 Phase5a 已经验证的事实

Phase5a 已经证明：

1. fixed Tikhonov linearized post-stack log-impedance inversion 工作正常；
2. 12/12 member 的 seismic RMSE 和 continuous logZ error 可以改善；
3. exact surface/borehole acoustics 可以保持；
4. 但 continuous property 的 pointwise nearest-code projection 不等于 geological recovery；
5. Phase5b 已经证明“posterior mean/spread + 手工 soft-property guidance”无效。

因此 Stage20：

```text
复用 Phase5a inversion kernel
不复用 Phase5a fixed-12 Flow-prior posterior construction
不复用 Phase5b soft guidance
```

---

# 1. Stage20 的冻结科学设计

## 1.1 总链路

正式方法冻结为：

```text
StructuralGeo categorical truth
        │
        ├── sparse hard conditions ─────────────────────┐
        │                                               │
        ↓                                               │
Phase4c multiclass acoustic codebook                    │
        ↓                                               │
full-cube noiseless post-stack seismic d_obs            │
        │                                               │
        ├───────────────┐                               │
        │               │                               │
        │      borehole-only low-frequency acoustic prior
        │               │
        │               ↓
        └──────→ Phase5a linearized logZ inversion
                        ↓
                 inverted logZ volume
                        ↓
                 fixed normalization
                        ↓
                      q_Z
                        ↓
            Stage19R residual velocity adapter
                        ↓
                  frozen geological CFM
                        ↓
             final hard categorical geology
                        ↓
       Phase4c multiclass hard seismic evaluator
```

核心概率/解释层面可理解为：

```text
geophysics supplies continuous evidence;
Flow supplies geological prior/support;
adapter learns how evidence should modify Flow velocity.
```

不要把 Stage20 描述为 exact Bayesian posterior sampler。

## 1.2 Stage20 明确不做的事情

禁止在本阶段加入：

```text
- Phase4c direct seismic-gradient trajectory guidance
- Stage18 continuous-target guidance
- Phase5b posterior-mean/spread soft guidance
- fixed12 Flow-prior inversion posterior
- D-Flow source optimization
- gravity fusion
- elastic/multi-angle data
- noise / wavelet mismatch / acquisition sparsity
- density/Vp two-channel inversion
- q threshold / q blur / q sharpening
- topology loss
- physics training loss
- base Flow fine-tuning
- adapter width/depth/dilation sweep
- learning-rate sweep
- inversion regularization sweep
- IDW power sweep
- early stopping / best-checkpoint selection
- TEST-based tuning
```

Stage20 必须是 **单一 frozen operating point**。

---

# 2. 复用矩阵

## 2.1 Phase4c：复用

必须复用当前已有文件/逻辑：

```text
experiments/stage4_seismic/configs/acoustic_distinct_label9_upper_bound_v1.json
experiments/stage4_seismic/configs/full_cube_noiseless_inverse_crime_v1.json
guidance/seismic.py
```

Phase4c observation 参数保持：

```text
grid_shape             = [64, 64, 64]
cell_size_m            = [100, 100, 50]
num_samples            = 320
sample_interval_ms     = 8.0
wavelet                = zero-phase Ricker
peak_frequency_hz      = 25.0
wavelet_duration_ms    = 128.0
trace_grid             = all_xy_columns
sample_mask            = all
uncertainty_amplitude  = 0.01
noise                   = none
local_surface_datum    = true
exclude_air_rock_interface = true
inverse_crime          = true
```

多类别 acoustic codebook 原样使用。不得改 label9 的 density/Vp，也不得为了 Stage20 成功重新制造更容易的属性表。

## 2.2 Phase4c：不复用

禁止调用/复制：

```text
Phase4c direct seismic residual guidance controller
alpha/cap controller sweep
```

Stage20 中 Phase4c 只承担：

```text
1. synthetic observation forward
2. final hard-geology seismic evaluation
```

## 2.3 Phase5a：复用

必须复用：

```text
guidance/seismic_inversion.py
experiments/stage5_acoustic_inversion/configs/model_based_log_impedance_v1.json
```

继续使用固定参数：

```text
inversion_mode                       = linearized_poststack_log_impedance_tikhonov_v1
prior_relative_weight                = 0.001
vertical_smoothness_relative_weight  = 0.01
regularization_scale                 = mean_diagonal_gtg
time_difference                      = forward_first_difference_last_row_zero
wavelet_boundary                     = zero_padding_same_length_no_wraparound
time_depth_mapping                   = cell-centre interpolation using fixed prior slowness
slowness_update                      = none
impedance_bounds                     = non-air codebook min/max
condition_policy                     = exact before/after inversion
```

**不要修改 `guidance/seismic_inversion.py` 的 Phase5a 数学实现。**

Stage20 可以直接调用：

```python
invert_acoustic_member(...)
```

或调用其内部已测试 utility，但不要重写另一套 inversion solver。

## 2.4 Phase5a：不复用

禁止复用：

```text
12 Flow priors
fixed12 posterior mean
fixed12 posterior std/spread
posterior_statistics 作为 Stage20 q
```

原因：Stage20 需要一个更干净的

```text
q = G(seismic, known conditions)
```

而不是：

```text
q = G(seismic, 12 Flow geological samples)
```

避免形成：

```text
Flow → inversion evidence → Flow
```

的循环 attribution。

注意：原 Phase5a config parser 中虽然仍包含

```text
posterior_statistics = fixed12_population_mean_std_v1
```

Stage20 仅把该 JSON 作为 **solver operating-point provenance** 使用；不得在 Stage20 构造 fixed12 statistics。不要为了改这个字符串而重构 Phase5a parser。

## 2.5 Stage19R：复用

必须复用：

```text
guidance/residual_velocity_adapter.py
Stage19R EMA/frozen-base load policy
Stage19R exact-condition projection
Stage19R residual cap
Stage19R training loss
Stage19R fixed-Euler adapter sampling path
Stage19R formal checkpoint guards / truth firewall 思路
```

当前成功 adapter 的正式参数：

```text
geophysics_channels     = 1
base_width              = 12
dilations               = [1, 2, 4, 1]
max_residual_ratio      = 0.25
final layer             = exact zero init
formal parameter count  = 54003
```

Stage20 必须保持 **1 个 geophysical channel**。

不要为了增加 missing-evidence mask、posterior spread 等额外引入第二通道。

## 2.6 Stage19R：不复用

Stage20 不读取：

```text
Stage17A binary impedance score
Stage18 target/confidence
Stage19R evidence_v2
```

除了 cohort / adapter implementation / training protocol 以外，Stage20 不依赖 binary evidence。

---

# 3. Cohort 设计

## 3.1 TRAIN / VAL：直接复用 Stage19R

TRAIN 与 VAL geology 不重新生成：

```text
TRAIN = Stage19R cohort_v2/train_registry.json = 64 cases
VAL   = Stage19R cohort_v2/val_registry.json   = 8 cases
```

路径：

```text
experiments/stage19_learned_evidence_adapter/cohort_v2/train_registry.json
experiments/stage19_learned_evidence_adapter/cohort_v2/val_registry.json
```

原因：Stage20 只希望改变 evidence semantics，不希望同时改变训练 geology 分布。

不得重新筛选这 72 cases。

## 3.2 TEST：新建独立 12-case cohort

Stage19R TEST 已经被审阅，因此 Stage20 formal TEST 使用新的 StructuralGeo roots。

冻结：

```json
{
  "accepted": 12,
  "start": 220290001,
  "step": 1,
  "max_candidates": 512
}
```

case ID：

```text
stage20_test_case001 ... stage20_test_case012
```

使用 **完全相同** 的 Full StructuralGeo recipe 和 eligibility：

```text
- final raw label9 positive
- hidden raw label9 positive
- BaseStrata in history
- Fold or Fault in history
- label9-producing intrusion event in history
```

不得根据：

```text
seismic
continuous evidence
Flow output
IoU
volume
centroid
topology
visual quality
```

进行 TEST selection。

若 512 candidates 内不足 12 个：

```text
STOP_STAGE20_TEST_COHORT_UNAVAILABLE
```

不要在同一次正式 protocol 中临时扩 seed range。

## 3.3 Generator reuse

不要复制新的 StructuralGeo recipe。

Stage20 开始时必须调用/复用：

```text
scripts/stage19/audit_generator_reuse.py
```

并要求：

```text
machine_decision == GENERATOR_REUSE_VALIDATED
```

同时 `build_test_cohort.py` 使用：

```python
from guidance.full_structuralgeo_benchmark import prepare_candidate
```

并复用 Stage19R `_save_case` 语义或等价逻辑。

不得修改 StructuralGeo generator source。

## 3.4 fixed wells

每个 case 必须使用 registry 中记录的同一 9 个 `well_xy`：

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

代码应优先从 case registry 的 `well_xy` 字段读取并验证等于 frozen list；不要在多个脚本中各维护一份可能漂移的列表。

---

# 4. 新建 Stage20 目录

建议：

```text
project/geodata-3d-conditional/
├── experiments/
│   └── stage20_continuous_impedance_adapter/
│       ├── README.md
│       ├── configs/
│       │   ├── cohort_v1.json
│       │   ├── evidence_v1.json
│       │   ├── training_v1.json
│       │   └── inference_v1.json
│       ├── cohort/
│       ├── observations/
│       ├── inversion_input_registry/
│       ├── evidence/
│       ├── evidence_audit/
│       ├── checkpoints/
│       │   └── formal_v1/
│       ├── smoke/
│       ├── formal/
│       │   └── inference_v1/
│       └── reports/
└── scripts/
    └── stage20/
        ├── __init__.py
        ├── common.py
        ├── audit_reuse.py
        ├── build_test_cohort.py
        ├── build_observations.py
        ├── build_continuous_evidence.py
        ├── audit_evidence.py
        ├── train_adapter.py
        ├── run_inference.py
        └── evaluate.py
```

**不要重构或改名 Stage19R 目录。**

Stage20 可以 import Stage19 通用 helper；若 helper 过度 Stage19-specific，则在 `scripts/stage20/common.py` 写薄 wrapper，不要为了“代码优雅”修改已经验证的 Stage19R 实验代码。

---

# 5. `cohort_v1.json` 明确内容

建议 schema：

```json
{
  "schema": "stage20_cohort_v1",
  "train": {
    "reuse_registry": "experiments/stage19_learned_evidence_adapter/cohort_v2/train_registry.json",
    "accepted": 64
  },
  "val": {
    "reuse_registry": "experiments/stage19_learned_evidence_adapter/cohort_v2/val_registry.json",
    "accepted": 8
  },
  "test": {
    "accepted": 12,
    "start": 220290001,
    "step": 1,
    "max_candidates": 512,
    "eligibility": "exact_full_structuralgeo_stage19r"
  },
  "generator_reuse_required": "GENERATOR_REUSE_VALIDATED",
  "selection_uses_seismic_evidence_flow_or_downstream_metrics": false
}
```

`build_test_cohort.py` 最终生成统一 Stage20 registry：

```text
cohort/train_registry.json  # pointers to Stage19R assets; no copies required
cohort/val_registry.json    # pointers to Stage19R assets; no copies required
cohort/test_registry.json   # new Stage20 cases
cohort/cohort_manifest.json
```

TRAIN/VAL registry 中必须记录：

```text
reused_from_stage19r = true
source registry path + SHA256
```

TEST：

```text
reused_from_stage19r = false
```

---

# 6. Phase4c multiclass observation builder

## 6.1 输入

`build_observations.py` 遍历 84 cases：

```text
64 TRAIN + 8 VAL + 12 new TEST
```

它可以读取 truth，**仅用于 synthetic observation generation**。

每 case 读取：

```text
truth
condition_values
condition_mask
well_xy
```

## 6.2 subsurface support

继续复用 Stage19 已验证的 column-wise surface support 语义：

```python
nonair = truth != -1
highest_nonair_z per (x,y)
support = z <= highest_nonair_z
```

这表示已知地表/air-rock support，不表示允许后续 inversion 或 inference 读取 unconstrained truth。

保存：

```text
subsurface_mask.pt
```

## 6.3 multiclass acoustic conversion

不能再生成 Stage19 binary occupancy acoustics。

必须使用 Phase4c 完整 acoustic codebook：

```text
raw geological label
      ↓
(density, Vp)
      ↓
impedance Z = density * Vp
slowness s = 1 / Vp
```

使用 `guidance/seismic.py` 中已有 hard-label/acoustic utility，避免手写不同映射。

## 6.4 seismic forward

使用 `full_cube_noiseless_inverse_crime_v1.json` 创建 operator。

每 case：

```python
observed = build_seismic_observation(...)
closure = operator(true_impedance, true_slowness, support)
```

要求：

```text
max_abs(closure - observed_seismic) <= 1e-7
```

否则：

```text
STOP_PHASE4C_FORWARD_REUSE_MISMATCH
```

## 6.5 observation outputs

每 case 保存：

```text
observed_seismic.pt
sample_mask.pt
uncertainty.pt
subsurface_mask.pt
condition_values.pt
condition_mask.pt
acoustic_property_table.pt   # or shared immutable source record
```

不要把：

```text
true_model.pt
truth_acoustic.pt
```

放入 inversion 输入目录。

## 6.6 双 registry truth firewall

`build_observations.py` 生成两个 registry：

### full observation registry

供 retrospective audit 使用，可保留 truth pointer：

```text
observations/observation_registry_full.json
```

### inversion-only registry

供 `build_continuous_evidence.py` 使用：

```text
inversion_input_registry/registry.json
```

该 registry 中 **禁止出现**：

```text
truth
true_model
truth_assets
binary_truth
target mask
label9 truth
```

允许：

```text
case_id
split
root_seed
well_xy
observed_seismic
sample_mask
uncertainty
subsurface_mask
condition_values
condition_mask
operator/config provenance
```

`build_continuous_evidence.py` 只能接受该 stripped registry。

---

# 7. Borehole-derived low-frequency acoustic prior

这是 Stage20 唯一新增的 geophysical-preprocessing 方法。

必须保持简单、确定性、无训练、无参数搜索。

## 7.1 插值对象

不直接对 categorical label 做数值平均。

对每个已知 borehole voxel 先由 Phase4c codebook 得到：

```text
logZ = log(density * Vp)
slowness = 1 / Vp
```

然后分别插值：

```text
logZ_LF(x,y,z)
slowness_LF(x,y,z)
```

## 7.2 只使用真正 well_xy

禁止用 combined `condition_mask & subsurface` 自动当作“井点”，因为 combined mask 还可能包含 surface conditions。

必须使用 registry 中的：

```text
well_xy
```

逐井提取该 `(x_i,y_i,:)` 的 condition values。

## 7.3 每个深度切片独立做 lateral IDW

冻结：

```text
method          = inverse_distance_weighting
power           = 2.0
distance_space  = xy grid-index distance
vertical mixing = none
```

对于 depth index `z`：

```python
valid wells = wells satisfying:
    subsurface_mask[xi, yi, z] == True
    condition_mask[xi, yi, z] == True
    condition_values[xi, yi, z] in raw labels 0..13
```

若某个 `z` 在任何 subsurface voxel 存在，但 valid well 数量为 0：

```text
STOP_LOW_FREQUENCY_PRIOR_UNDEFINED
```

不要自动使用邻近深度、global mean 或 truth 补洞。

## 7.4 IDW 公式

非井位置：

```text
d_i^2 = (x-x_i)^2 + (y-y_i)^2
w_i   = 1 / d_i^2
w_i   = w_i / sum_j(w_j)
```

分别：

```text
logZ_LF = Σ w_i logZ_i
s_LF    = Σ w_i s_i
```

### exact-well override

如果 `(x,y)` 正好等于一个 valid well：

```text
logZ_LF(x,y,z) = exact well logZ
s_LF(x,y,z)    = exact well slowness
```

不要使用 epsilon 让井值被其他井污染。

## 7.5 air / surface

对 `subsurface_mask == False`：

```text
impedance = Phase4c air impedance
slowness  = Phase4c air slowness
```

## 7.6 exact condition overwrite

IDW 完成后，把所有 `condition_mask == True` 的 acoustic properties 重写为由 `condition_values` 映射得到的 exact Phase4c codebook properties。

因此：

```text
condition acoustic violations = 0
```

构造：

```text
prior_acoustic = concat(exp(logZ_LF), slowness_LF)
shape = [1,2,64,64,64]
```

---

# 8. Phase5a continuous inversion evidence

## 8.1 输入

`build_continuous_evidence.py` 只读取 stripped inversion registry。

每 case：

```text
observed seismic
sample mask
uncertainty
subsurface mask
condition values/mask
well_xy
Phase4c codebook
Phase4c seismic operator config
Phase5a inversion config
```

它不得读取 truth。

## 8.2 调用 Phase5a solver

构造：

```text
prior_acoustic = [Z_LF, s_LF]
```

然后调用现有：

```python
prior_exact, inverted, fields, diagnostics = invert_acoustic_member(...)
```

其中：

```text
inverted impedance = updated Z
inverted slowness  = fixed prior slowness except exact-condition overwrite
```

必须断言：

```text
free-subsurface slowness is unchanged by inversion
condition acoustic violations == 0
all tensors finite
```

## 8.3 不生成 posterior ensemble

每 case 只有：

```text
1 deterministic low-frequency prior
1 deterministic inverted property volume
```

不要生成 12 prior members。

## 8.4 保存内容

每 case 至少保存：

```text
low_frequency_acoustic.pt       [1,2,64,64,64]
inverted_acoustic.pt            [1,2,64,64,64]
low_frequency_logz.pt           [1,1,64,64,64]
inverted_logz.pt                [1,1,64,64,64]
normalized_low_frequency_logz.pt
normalized_inverted_logz.pt
subsurface_mask.pt or reference
condition_mask.pt or reference
```

可以额外保存：

```text
delta_logz_depth.pt
```

仅用于 diagnostic；不得作为 Stage20 v1 adapter input。

---

# 9. Evidence normalization

## 9.1 禁止 per-case normalization

绝对禁止：

```python
(q - q.min(case)) / (q.max(case)-q.min(case))
```

因为会抹掉 absolute impedance semantics。

## 9.2 frozen global codebook normalization

只使用 Phase4c non-air raw labels `0..13` 的 acoustic codebook：

```text
L_min = min_k log(Z_k)
L_max = max_k log(Z_k)
```

固定：

```text
q_norm = 2 * (logZ - L_min) / (L_max - L_min) - 1
q_norm = clamp(q_norm, -1, +1)
```

TRAIN / VAL / TEST 全部使用同一个 `L_min/L_max`。

在 evidence manifest 中记录：

```text
L_min
L_max
source codebook SHA256
```

## 9.3 q 文件保存为“完整场”，mask 在消费时应用

保存：

```text
normalized_inverted_logz.pt
normalized_low_frequency_logz.pt
```

不要在文件生成阶段永久乘当前 free mask。

训练/推理时统一：

```python
free = subsurface_mask & (~condition_mask)
q = normalized_logz * free.float()
```

理由：formal wrong-case arm 必须用 **当前 case 的 free mask**，避免 wrong case 自己的 topography/condition mask 造成额外混淆。

---

# 10. Evidence gate：训练前唯一 scientific precondition

只在 8 个 Stage19R VAL cases 上执行。

**不得用 TEST gate。**

`audit_evidence.py` 可以在所有 evidence tensors 和 hashes 冻结后，单独读取 VAL truth 做 retrospective property audit。

## 10.1 integrity gate

要求 8/8：

```text
- finite tensors
- Phase4c forward closure <= 1e-7
- exact condition acoustic violations = 0
- inversion output within frozen non-air impedance bounds on subsurface
- free-subsurface slowness unchanged by Phase5a inversion
- no truth loaded by build_continuous_evidence.py
```

任一失败：

```text
STOP_CONTINUOUS_EVIDENCE_INTEGRITY
```

## 10.2 seismic improvement gate

每个 VAL case 分别计算：

```text
RMSE_prior = RMSE(F(Z_LF,s_LF), d_obs)
RMSE_post  = RMSE(F(Z_post,s_LF), d_obs)
```

冻结 gate：

```text
post < prior in at least 6/8 cases
median(post - prior) < 0
```

## 10.3 continuous property improvement gate

truth 只在 auditor 中用于计算：

```text
true_logZ from multiclass truth + Phase4c codebook
```

metric region：

```text
free_subsurface = subsurface & ~condition_mask
```

计算：

```text
RMSE_logZ_prior
RMSE_logZ_post
```

冻结 gate：

```text
post < prior in at least 6/8 cases
median(post - prior) < 0
```

不要要求 AUPRC。

Stage20 q 是 continuous property，不声称是 `P(label9|d)`。

## 10.4 evidence gate decision

全部通过：

```text
CONTINUOUS_IMPEDANCE_EVIDENCE_VALIDATED
adapter_training_authorized = true
```

否则：

```text
STOP_CONTINUOUS_IMPEDANCE_EVIDENCE_NOT_VALIDATED
adapter_training_authorized = false
```

此时 **不训练 adapter**。

不得在 gate 失败后调：

```text
IDW p
λprior
λsmooth
wavelet
normalization
```

然后继续同一 Stage20 v1。

---

# 11. Stage20 adapter training

## 11.1 训练数据

```text
TRAIN = 64 reused Stage19R geology cases
VAL   = 8 reused Stage19R geology cases
```

每 case Stage20 training input：

```text
truth                      # only supervised target
condition_values
condition_mask
subsurface_mask
normalized_inverted_logz   # q source
```

运行时：

```python
free = subsurface & (~condition_mask)
q = normalized_inverted_logz * free.float()
```

## 11.2 完整复用 Stage19R training hyperparameters

冻结：

```text
base checkpoint:
    demo_model/conditional-weights.ckpt
    SHA256 = 561e94bfda770ec41fc4cbed43436a7e2130eef5dfb7e5d666fcefc0724ff94c
    weight source = EMA
    freeze all = true

adapter:
    geophysics_channels = 1
    base_width = 12
    dilations = [1,2,4,1]
    max_residual_ratio = 0.25
    zero-init output = true
    expected parameter count = 54003

training:
    adapter_seed = 6200
    state_generator_seed = 6100
    epochs = 4
    times = [0.2,0.4,0.6,0.8]
    batch_size = 1
    optimizer = AdamW
    learning_rate = 0.002
    weight_decay = 0.0001
    gradient_clip_norm = 1.0
    scheduler = none
    flow_weight = 1.0
    cross_entropy_weight = 0.25
    dice_weight = 0.25
    residual_regularizer_weight = 0.0001
    logit_temperature = 0.1
    early_stopping = false
    checkpoint_selection = epoch4_final
    shuffle_seed_base = 6201
```

formal optimizer updates 必须严格：

```text
64 cases × 4 times × 4 epochs = 1024
```

## 11.3 objective 不改

继续 Stage19R：

```text
state,target_velocity = base CFM flow_objective
state = exact condition projection
base_velocity = frozen Flow velocity
correction = adapter(state, base_velocity, conditioning, mask, q, t)
correction = capped at 0.25 relative ratio
```

loss：

```text
L = 1.0 * flow
  + 0.25 * CE
  + 0.25 * Dice
  + 1e-4 * residual regularizer
```

不增加 logZ loss 或 seismic loss。

## 11.4 formal training guard

formal checkpoint 只允许在以下全部满足时写出：

```text
run_class == formal_training
smoke_subset == false
epochs == 4
optimizer_updates == 1024
checkpoint_selection == epoch4_final
adapter_parameter_count == 54003
base_checkpoint_sha256 exact match
training_config_sha256 exact match
base model hash before == after
base gradients absent
base_finetuning == false
all losses finite
```

不要根据 validation loss 选 best epoch；epoch4 是唯一正式 checkpoint。

---

# 12. Formal inference arms：只保留 4 个

Stage20 formal TEST：

```text
12 cases × 3 source seeds × 4 arms = 144 outputs
```

source seeds：

```text
42
142
242
```

sampling：

```text
n_steps = 32
integrator = fixed_euler_midpoint
adapter_scale = 1.0 for adapter arms
max_residual_ratio = 0.25
exact condition projection every step
```

## 12.1 Arm A — FLOW_ONLY

```text
adapter_scale = 0
```

使用同一 Stage20 sampling runner，以保证 strict pairing。

必须验证其输出与独立 frozen-Flow baseline 等价（smoke 中做一次 byte/hard-voxel regression）。

## 12.2 Arm B — ADAPTER_PRIOR_ONLY

```text
q = normalized_low_frequency_logz_current_case * current_free_mask
adapter_scale = 1
```

回答：

```text
adapter 的收益是否仅来自 borehole interpolation / low-frequency background？
```

这是 Stage20 最重要的新 control。

## 12.3 Arm C — ADAPTER_CORRECT

```text
q = normalized_inverted_logz_current_case * current_free_mask
adapter_scale = 1
```

这是正式方法。

## 12.4 Arm D — ADAPTER_WRONG_CASE

wrong-case mapping：

```text
next_test_case_cyclic
case001 <- q_correct(case002)
...
case012 <- q_correct(case001)
```

但必须：

```text
q_wrong = wrong_case_normalized_inverted_logz * CURRENT_CASE_free_mask
```

不得使用 wrong case 自己的 mask。

回答：

```text
continuous evidence 是否具有 case specificity？
```

## 12.5 为什么 formal 不保留 ADAPTER_ZERO

**不要实现 formal `ADAPTER_ZERO` arm。**

理由是 Stage20 normalized continuous q 的数值 `0` 本身对应 codebook 范围中间值，是一个合法连续物性值；把全零直接解释为“没有 evidence”语义不干净。

也不要为了恢复 zero arm 而增加第二个 evidence-mask channel，因为这会改变 adapter architecture，使 Stage20 同时改变两件事。

Stage20 需要的三个 attribution 已经由下面三组比较覆盖：

```text
CORRECT vs FLOW       → continuous evidence + learned adapter 是否总体有效
CORRECT vs PRIOR_ONLY → seismic inversion 是否在 wells 之外提供增量
CORRECT vs WRONG      → evidence 是否 case-specific
```

这比增加一个语义含糊的 zero arm 更简单、更严谨。

---

# 13. TEST truth firewall

正式 inference 之前生成：

```text
evidence/test_inference_registry.json
```

该文件只包含 TEST 所需：

```text
case_id
root_seed
condition_values
condition_mask
subsurface_mask
normalized_low_frequency_logz
normalized_inverted_logz
observed seismic metadata / assets required for output manifest
```

明确禁止：

```text
truth
truth_assets
true_model
true_logz
label9 mask
history fields that expose target geometry
retrospective metric files
```

`run_inference.py` formal mode只接受 stripped registry。

`evaluate.py` 在 144 outputs 全部完成、manifest/hash 冻结后，才读取 full registry 和 truth。

formal inference manifest 必须记录：

```text
truth_loaded_by_runner = false
```

---

# 14. Final hard evaluation

## 14.1 必须计算的 geology metrics

每个 case × seed × arm：

```text
global voxel accuracy
truth-present mIoU
per-class IoU
label9 IoU
label9 precision
label9 recall
label9 predicted voxel count
label9 absolute volume-error fraction
label9 centroid distance
label9 connected components
largest-component mass fraction
top4/top8 mass fraction
exact condition violations
```

case-first aggregation：

```text
先在 3 source seeds 内取 case median
再跨 12 cases 统计
```

禁止把 36 samples 当成 36 个独立 cases。

## 14.2 hard seismic evaluator

对于每个 final hard categorical sample：

```text
hard geology
   ↓ Phase4c multiclass codebook
(Z_hard, s_hard)
   ↓ exact same Phase4c forward operator
predicted seismic
```

计算：

```text
hard seismic RMSE vs d_obs
hard seismic MAE (diagnostic)
```

这一步必须使用完整 multiclass density/Vp codebook，而不是 Stage19 binary acoustic model。

## 14.3 property-only nearest-code baseline

在 evaluator 中为每个 TEST case 额外计算一次 deterministic baseline：

```text
inverted_logZ
   ↓ nearest non-air codebook logZ
pointwise categorical geology
```

规则：

```text
subsurface: nearest logZ among raw labels 0..13
above surface: -1
hard conditions: overwrite exact condition_values
tie: torch.argmin first index
```

命名：

```text
PROPERTY_ONLY_NEAREST_LOGZ
```

该 baseline：

```text
- 不参与 Flow sampling
- 不需要 3 seeds
- 不作为 Stage20 pass/fail gate
- 只用于解释“continuous inversion + pointwise classification”与“Flow geological interpretation”的差异
```

---

# 15. Stage20 final scientific gates

以下 gate 全部按 **12 case 的 seed-median** 计算。

## Gate A — learned continuous evidence coupling

要求：

```text
ADAPTER_CORRECT target-label IoU > FLOW_ONLY
in at least 9/12 cases
```

且：

```text
cross-case median( IoU_correct - IoU_flow ) > +0.05
```

## Gate B — seismic increment beyond boreholes

要求：

```text
ADAPTER_CORRECT target-label IoU > ADAPTER_PRIOR_ONLY
in at least 8/12 cases
```

且：

```text
median( IoU_correct - IoU_prior_only ) > +0.02
```

这一 gate 是 Stage20 最关键的 attribution gate。

如果 A 通过但 B 不通过，不能写“seismic 提升 geology”；只能判定 adapter 主要利用 low-frequency well background。

## Gate C — case specificity

要求：

```text
ADAPTER_CORRECT target-label IoU > ADAPTER_WRONG_CASE
in at least 8/12 cases
```

且：

```text
median( IoU_correct - IoU_wrong ) > +0.02
```

## Gate D — hard physics consistency

要求：

```text
hard seismic RMSE(correct) < hard seismic RMSE(flow)
in at least 8/12 cases
```

且：

```text
median( RMSE_correct - RMSE_flow ) < 0
```

## Gate E — target volume calibration

要求：

```text
median absolute label9 volume-error fraction(correct)
<
median absolute label9 volume-error fraction(flow)
```

不要求每 case 都改善。

## Gate F — global geology preservation

要求：

```text
median( truth_present_mIoU_correct - truth_present_mIoU_flow ) >= -0.01
```

## Gate G — hard conditions

要求所有 144 outputs：

```text
condition violation count == 0
```

## 不设 topology hard gate

记录：

```text
components
largest component fraction
top4/top8 mass
centroid
```

但 Stage20 v1 不因为单一 component-count 波动而 STOP。

如结果出现明显 catastrophic fragmentation，再在结果讨论阶段决定是否需要下一阶段 topology intervention；不要预先把 Stage20 复杂化。

---

# 16. Final decision vocabulary

## 16.1 全部主要 gates 通过

A/B/C/D/E/F/G 通过：

```text
CONTINUOUS_IMPEDANCE_ADAPTER_VALIDATED
```

允许结论：

```text
在当前 noiseless inverse-crime multiclass acoustic upper bound 下，
由 seismic + sparse wells 构建的连续 log-impedance evidence，
经 learned residual adapter 后，能够对 frozen 3-D categorical geological Flow
产生 case-specific、hard-physics-consistent 的正向更新；
且 improvement 超过仅使用 borehole-derived low-frequency property background。
```

仍然禁止声称：

```text
field validated
noise robust
realistic petrophysics solved
exact posterior inference
```

## 16.2 Evidence gate 通过，但 Gate A 失败

```text
CONTINUOUS_EVIDENCE_COUPLING_NOT_VALIDATED
```

解释：continuous inversion 本身工作，但 one-channel continuous property 没有通过当前 adapter 映射成 hard geology。

不要先调 adapter 超参数。

## 16.3 A 通过，B 失败

```text
WELL_PRIOR_DOMINATED_NO_SEISMIC_INCREMENT
```

解释：adapter 有效，但不能证明 seismic 在 sparse wells 之外贡献有效增量。

## 16.4 A/B 通过，C 失败

```text
CONTINUOUS_EVIDENCE_NOT_CASE_SPECIFIC
```

解释：存在 generic learned correction 风险。

## 16.5 geometry 通过，D 失败

```text
GEOLOGY_IMPROVES_WITHOUT_HARD_SEISMIC_SUPPORT
```

不能把它归类为完整 Stage20 成功。

---

# 17. `training_v1.json` 建议内容

直接从 Stage19 `training_v1.json` 复制后只修改：

```text
schema
status/path provenance
evidence.type
evidence.input_policy
```

冻结逻辑示意：

```json
{
  "schema": "stage20_continuous_impedance_adapter_training_v1",
  "status": "frozen_before_cuda_training",
  "parameter_sweep": false,
  "target_label": 9,
  "base_model": {
    "checkpoint": "demo_model/conditional-weights.ckpt",
    "weight_source": "ema",
    "checkpoint_sha256": "561e94bfda770ec41fc4cbed43436a7e2130eef5dfb7e5d666fcefc0724ff94c",
    "freeze_all_parameters": true
  },
  "evidence": {
    "type": "phase5a_inverted_multiclass_log_impedance",
    "channels": 1,
    "input_policy": "fixed_codebook_normalized_logz_times_current_free_subsurface"
  },
  "adapter": {
    "base_width": 12,
    "dilations": [1,2,4,1],
    "geophysics_channels": 1,
    "expected_parameters": 54003,
    "max_parameters": 100000,
    "max_residual_ratio": 0.25,
    "final_layer_zero_initialized": true
  },
  "training": {
    "adapter_seed": 6200,
    "state_generator_seed": 6100,
    "epochs": 4,
    "times": [0.2,0.4,0.6,0.8],
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
    "checkpoint_selection": "epoch4_final",
    "shuffle_seed_base": 6201
  },
  "sampling": {
    "n_steps": 32,
    "adapter_scale": 1.0,
    "source_seeds": [42,142,242]
  },
  "topology_loss": false,
  "physics_training_loss": false,
  "base_finetuning": false
}
```

---

# 18. `inference_v1.json` 建议内容

```json
{
  "schema": "stage20_inference_v1",
  "status": "frozen_before_run",
  "parameter_sweep": false,
  "truth_loaded_by_runner": false,
  "target_label": 9,
  "source_seeds": [42,142,242],
  "n_steps": 32,
  "integrator": "fixed_euler_midpoint",
  "arms": [
    "FLOW_ONLY",
    "ADAPTER_PRIOR_ONLY",
    "ADAPTER_CORRECT",
    "ADAPTER_WRONG_CASE"
  ],
  "adapter_scale": 1.0,
  "max_residual_ratio": 0.25,
  "wrong_case_mapping": "next_test_case_cyclic",
  "wrong_case_mask_policy": "always_use_current_case_free_subsurface",
  "expected_outputs": 144
}
```

---

# 19. `evidence_v1.json` 建议内容

应明确记录所有 provenance，至少：

```json
{
  "schema": "stage20_continuous_impedance_evidence_v1",
  "status": "frozen_before_build",
  "target_label": 9,
  "phase4c": {
    "acoustic_config": "experiments/stage4_seismic/configs/acoustic_distinct_label9_upper_bound_v1.json",
    "seismic_config": "experiments/stage4_seismic/configs/full_cube_noiseless_inverse_crime_v1.json"
  },
  "phase5a": {
    "inversion_config": "experiments/stage5_acoustic_inversion/configs/model_based_log_impedance_v1.json",
    "reuse_scope": "solver_only_no_fixed12_posterior"
  },
  "low_frequency_prior": {
    "method": "well_xy_depthwise_idw_logz_slowness_v1",
    "power": 2.0,
    "distance_space": "xy_grid_index",
    "vertical_mixing": false,
    "exact_well_override": true,
    "empty_valid_well_depth_policy": "stop"
  },
  "normalization": {
    "type": "global_nonair_codebook_logz_minus1_plus1",
    "per_case": false,
    "clamp": [-1.0,1.0],
    "runtime_mask": "current_free_subsurface"
  },
  "gate": {
    "val_cases": 8,
    "min_seismic_improved_cases": 6,
    "require_negative_median_seismic_delta": true,
    "min_logz_improved_cases": 6,
    "require_negative_median_logz_delta": true
  },
  "parameter_sweep": false
}
```

实现时必须把当前文件 SHA256 写入 generated manifest；不要凭本文件猜 hash。

---

# 20. Provenance / hash 规则

`audit_reuse.py` 必须在任何 Stage20 observation/evidence build 前检查：

```text
1. Stage19R generator reuse decision == GENERATOR_REUSE_VALIDATED
2. Stage19R train_registry complete 64
3. Stage19R val_registry complete 8
4. Phase4c acoustic config current file hash
5. Phase4c seismic config current file hash
6. guidance/seismic.py current source hash
7. Phase5a model_based_log_impedance_v1.json current file hash
8. guidance/seismic_inversion.py current source hash
9. guidance/residual_velocity_adapter.py current source hash
10. base checkpoint SHA256 exact
```

对于 Phase4c/Phase5a authoritative reuse：

- 优先读取它们原成功 experiment manifest 中已记录的 source/config SHA256；
- 如果 authoritative manifest 有记录，当前文件必须匹配；
- 如果旧 manifest 没有某项 hash，则 Stage20 preflight 记录当前 hash，之后在整个 Stage20 v1 中冻结，不允许静默变化。

不要修改旧 Phase4c/Phase5a artifacts 来补 hash。

reuse preflight 输出：

```text
experiments/stage20_continuous_impedance_adapter/reuse_audit/summary.json
```

通过：

```text
STAGE20_REUSE_VALIDATED
```

否则 STOP。

---

# 21. 必须实现的 focused tests

新建：

```text
tests/test_stage20_continuous_impedance_adapter.py
```

至少覆盖以下内容。

## 21.1 low-frequency prior

测试：

```text
- exact well voxel remains exact
- p=2 IDW numerically correct on a tiny synthetic slice
- interpolation is lateral-only at fixed z
- labels are mapped to logZ/slowness before interpolation
- no direct numeric averaging of class IDs
- invalid/air well at a depth excluded
- no valid well at an occupied depth raises STOP_LOW_FREQUENCY_PRIOR_UNDEFINED
- above-surface uses air acoustic
- final exact condition overwrite gives zero violations
```

## 21.2 normalization

```text
- codebook minimum rock logZ -> -1
- codebook maximum rock logZ -> +1
- same input gives same q across cases
- no per-case min/max
- runtime free mask zeros condition and air voxels
```

## 21.3 wrong-case semantics

```text
- wrong evidence tensor comes from next cyclic case
- mask always comes from current case
- condition voxels always q=0
```

## 21.4 truth firewall

```text
- inversion-only registry contains no truth key/path
- build_continuous_evidence.py source does not access truth assets
- formal TEST inference registry contains no truth assets
- run_inference.py does not load truth
```

## 21.5 reuse contracts

```text
- Phase4c config/source mismatch rejected
- Phase5a config/source mismatch rejected
- base checkpoint hash mismatch rejected
- Stage19R generator audit failure rejected
```

## 21.6 training guards

```text
- adapter parameter count exactly 54003
- optimizer contains adapter params only
- base params require_grad false
- base gradients absent
- 1024 formal updates required
- non-epoch4 checkpoint rejected
- smoke checkpoint rejected by formal inference
- config-hash mismatch rejected
```

## 21.7 inference

```text
- exact 4 arms
- exact 3 seeds
- expected formal outputs = 144
- scale-zero FLOW_ONLY regression
- exact hard conditions every arm
```

---

# 22. Smoke 策略

在 formal training 前只做一次工程 smoke。

目的仅检查：

```text
CUDA execution
memory
file paths
registry schema
truth firewall
adapter q shape/range
4-arm inference routing
hard-condition projection
hard seismic evaluator
```

建议：

```text
2 TRAIN cases
1 VAL case
1 TEST case
2 optimizer updates
1 source seed
4 arms
```

smoke 结果不得用于：

```text
改 IDW
改 inversion λ
改 adapter 参数
改 loss
改 gate
```

smoke 发现 engineering bug 可以修；修复后写新 `*_fix1` 目录，不覆盖失败诊断。

---

# 23. 正式执行顺序

严格执行：

```text
1. audit reuse
2. build/reuse cohort
3. build Phase4c multiclass observations
4. create stripped inversion-only registry
5. build deterministic low-frequency priors + Phase5a inverted logZ evidence
6. audit VAL evidence gate
7. STOP if gate fails
8. engineering CUDA smoke
9. exactly one formal adapter training
10. formal checkpoint integrity audit
11. exactly one formal 12×3×4 inference
12. freeze output manifest + hashes
13. retrospective full evaluation
14. write final report
15. STOP for research discussion
```

建议命令形态：

```bash
../../.venv/bin/python scripts/stage20/audit_reuse.py
../../.venv/bin/python scripts/stage20/build_test_cohort.py
../../.venv/bin/python scripts/stage20/build_observations.py --device cuda
../../.venv/bin/python scripts/stage20/build_continuous_evidence.py --device cuda
../../.venv/bin/python scripts/stage20/audit_evidence.py

../../.venv/bin/python scripts/stage20/train_adapter.py --smoke \
  --output-dir experiments/stage20_continuous_impedance_adapter/smoke/training_v1

../../.venv/bin/python scripts/stage20/run_inference.py --smoke \
  --training-manifest experiments/stage20_continuous_impedance_adapter/smoke/training_v1/training_manifest.json \
  --adapter-checkpoint experiments/stage20_continuous_impedance_adapter/smoke/training_v1/adapter_checkpoint.pt \
  --output-dir experiments/stage20_continuous_impedance_adapter/smoke/inference_v1

../../.venv/bin/python scripts/stage20/train_adapter.py
../../.venv/bin/python scripts/stage20/run_inference.py
../../.venv/bin/python scripts/stage20/evaluate.py --device cuda
```

实际参数名可以适应已有 Stage19 CLI，但必须保持上述逻辑和目录分离。

---

# 24. Codex 实现时的禁止性说明

以下属于设计错误，不允许发生：

## 24.1 不得把 combined condition mask 当作 borehole locations

必须使用 registry `well_xy`。

## 24.2 不得直接平均 geological class IDs

IDW 插值的是：

```text
logZ
slowness
```

不是 label number。

## 24.3 不得用 truth 构造 low-frequency prior

low-frequency prior 只能使用：

```text
well_xy
condition values
known subsurface support
Phase4c codebook
```

## 24.4 不得让 inversion builder 读取 truth

truth 只允许：

```text
synthetic observation generation
supervised adapter training target
retrospective VAL/TEST audit
```

## 24.5 不得使用 per-case q normalization

必须使用 global frozen codebook normalization。

## 24.6 不得在 wrong-case arm 使用 wrong-case mask

必须：

```text
wrong q values + current case free mask
```

## 24.7 不得重新引入 ADAPTER_ZERO formal arm

continuous zero 缺少明确 missing-evidence 语义，本阶段不为此改 adapter input schema。

## 24.8 不得把 Stage20 成功条件写成“continuous loss 降低”

最终主要 gate 必须是：

```text
hard geology
hard seismic
seismic increment beyond wells
case specificity
```

## 24.9 不得自动 tune TEST

正式 TEST 一旦运行，不因为结果差修改参数后重跑并覆盖。

---

# 25. Final report 必须回答的五个问题

最终 `STAGE20_REPORT.md` 按顺序回答：

### Q1. deterministic seismic inversion evidence 本身是否有效？

报告 VAL：

```text
seismic prior→post
logZ prior→post
exact conditions
```

### Q2. continuous q 能否改善 hard categorical geology？

报告：

```text
CORRECT vs FLOW
```

### Q3. 改善是否真正来自 seismic，而不是 well interpolation？

报告：

```text
CORRECT vs PRIOR_ONLY
```

### Q4. evidence 是否 case-specific？

报告：

```text
CORRECT vs WRONG_CASE
```

### Q5. hard geology 改善是否同时得到 acquisition-domain physics 支持？

报告：

```text
hard seismic RMSE correct vs Flow
```

然后再报告：

```text
PROPERTY_ONLY_NEAREST_LOGZ
```

用于说明 pointwise property classification 与 geological prior interpretation 的差异。

---

# 26. 成功后的唯一推荐下一步

只有在：

```text
CONTINUOUS_IMPEDANCE_ADAPTER_VALIDATED
```

后，下一阶段才讨论：

```text
- overlapping / ambiguous realistic petrophysics
- additive noise
- wavelet/model mismatch
- sparse acquisition
- second physical-property channel
```

Stage20 内不要提前做这些。

若 Stage20 因 continuous property identifiability 失败，优先讨论：

```text
第二个物性通道 / multi-property evidence
```

而不是先扩大 adapter 或调 alpha/learning rate。

---

# 27. 开发完成后的验收清单

Codex 在提交前逐项确认：

```text
[ ] Stage19R 原 artifacts 未修改
[ ] Phase4c 原 artifacts 未修改
[ ] Phase5a 原 artifacts 未修改
[ ] generator reuse audit passed
[ ] TRAIN64 / VAL8 exact reuse
[ ] new TEST12 generated without downstream selection
[ ] Phase4c multiclass observation closure passed 84/84
[ ] inversion-only registry has no truth
[ ] low-frequency prior uses well_xy only
[ ] IDW is on logZ/slowness, p=2, no vertical mixing
[ ] exact well override implemented
[ ] Phase5a λ = 0.001 / 0.01 unchanged
[ ] no fixed12 posterior construction
[ ] q uses global codebook normalization, not per-case normalization
[ ] VAL evidence gate frozen and audited before training
[ ] adapter input remains 1 channel
[ ] adapter parameter count = 54003
[ ] base checkpoint EMA + fully frozen
[ ] formal training exactly 4 epochs / 1024 updates
[ ] no physics/topology loss
[ ] formal arms exactly 4
[ ] formal outputs exactly 144
[ ] wrong-case uses current-case mask
[ ] TEST inference registry has no truth
[ ] condition violations = 0
[ ] final evaluator uses multiclass Phase4c hard seismic
[ ] case-first statistics used
[ ] no parameter sweep
[ ] no TEST tuning
[ ] focused tests pass
[ ] py_compile pass
[ ] git diff --check pass
[ ] final report stops for research discussion
```

---

# 28. 本阶段最简洁的方法定义

若需要在 README / report 中用一句话概括 Stage20：

> **Stage20 keeps the pretrained 3-D categorical Flow model frozen, reuses the Phase4c multiclass seismic forward model and the Phase5a fixed log-impedance inversion kernel, replaces the historical Flow-ensemble inversion prior with a deterministic borehole-derived low-frequency acoustic background, and trains only the already validated small residual velocity adapter to translate the resulting continuous inverted log-impedance field into geological flow corrections.**

中文：

> **Stage20 冻结原三维 categorical Flow，仅复用 Phase4c 多类别地震正演与 Phase5a 固定阻抗反演内核，用钻井条件构建确定性低频声学背景代替历史 fixed12 Flow-prior posterior，并只训练已经验证的小型 residual velocity adapter，将连续反演 log-impedance 证据转化为地质生成速度修正。**

这就是本阶段完整范围；不要在实现过程中继续扩展方法树。
