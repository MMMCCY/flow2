# Flow2 Stage19R Codex 开发修改方案
## 主题：Stage19 cohort availability 修复 + 复用链路审计加固；不修改 learned evidence adapter 科学方案

> **本修改不是新的方法实验，也不是对 Stage19 adapter 的重新设计。**
>
> Stage19 v1 已在 cohort 构建阶段按预注册规则正确停止：
>
> - TRAIN candidate range：`220260001..220260256`
> - examined：`256`
> - eligible：`33`
> - required：`64`
> - observed eligibility rate：`33/256 = 0.12890625`
> - stop：`RuntimeError: STOP: insufficient eligible train cases`
>
> 因此 Stage19 v1 没有产生 observation、evidence gate、adapter training、formal inference 或 scientific result。
>
> 本次只修复一个已经明确暴露出的**实验前置样本可用性问题**，并补上在代码审计中发现的几个 provenance / truth-firewall / formal-checkpoint 防护缺口。
>
> **禁止借此机会修改 adapter、loss、Flow、geophysics 方法或评价标准。**

---

# 0. 现有 Stage19 v1 的结论边界

必须保留并明确写入 handoff：

```text
Stage19 v1 = PRETRAINING_COHORT_AVAILABILITY_STOP
```

它不是：

```text
ADAPTER_FAIL
EVIDENCE_FAIL
FLOW_FAIL
TOPOLOGY_FAIL
GEOPHYSICS_FAIL
```

当前唯一可以支持的实验事实是：

```text
在固定的新 TRAIN seed range 内，
原 Full StructuralGeo eligibility 的实际通过率为 33/256 ≈ 12.89%，
因此 256 个 candidate 的预注册上限不足以得到 64 个 TRAIN cases。
```

禁止从这一结果讨论 adapter 泛化能力、hard seismic、target IoU 或 topology，因为这些阶段尚未执行。

Stage19 v1 的全部已有输出保持 immutable，不覆盖、不删除、不重写：

```text
experiments/stage19_learned_evidence_adapter/cohort/
experiments/stage19_learned_evidence_adapter/reports/DEVELOPMENT_REPORT.md
docs/DEVELOPMENT_HANDOFF.md 中 Stage19 v1 段落
```

---

# 1. 对现有实现的审计结论

## 1.1 可以继续使用，不修改的核心科学实现

以下设计与 Stage19 初衷一致：

1. base geological CFM 全冻结，EMA 加载；
2. 只训练 external `ResidualVelocityAdapter`；
3. adapter 输入仍为：
   - state
   - base velocity
   - geological conditioning
   - condition mask
   - one-channel Stage17A evidence
   - time
4. evidence 使用未阈值化 `binary_impedance_score`；
5. adapter correction 在 hard-condition mask 内严格为 0；
6. residual norm cap 固定为 `0.25`；
7. training truth 只用于 supervised CFM velocity / CE / Dice；
8. inference runner 当前代码没有主动读取 truth tensor；
9. TEST 仍为：
   - `FLOW_ONLY`
   - `STAGE18_CONTINUOUS_TARGET`
   - `ADAPTER_CORRECT`
   - `ADAPTER_ZERO`
   - `ADAPTER_WRONG_CASE`
10. source seeds 仍为 `42 / 142 / 242`；
11. case-first statistics 保持不变。

这些内容不要重新设计。

---

# 2. Stage19 v1 暴露出的真正问题

## 2.1 cohort budget 不是方法参数，但原先预算缺乏可行性保障

原配置：

```json
"train": {
  "accepted": 64,
  "start": 220260001,
  "step": 1,
  "max_candidates": 256
}
```

实际：

```text
33 eligible / 256 examined
```

即：

```text
p_hat = 0.12890625
```

若这一数量级具有代表性，则获得 64 个 eligible cases 所需 candidate 的期望数量约为：

```text
64 / 0.1289 ≈ 496
```

因此 `max_candidates=256` 本身不具备充分可行性。

这属于**cohort logistics / availability assumption 错误**，不是 adapter 科学设计失败。

---

# 3. Stage19R 的唯一 cohort 修改

建立新的、明确标记为 revision 的 cohort protocol，例如：

```text
configs/cohort_v2.json
cohort_v2/
```

禁止覆盖 `cohort_v1.json` 和 `cohort/`。

## 3.1 accepted 数量完全不变

```text
TRAIN = 64
VAL   = 8
TEST  = 12
```

不得为了利用现有 33 个 case 而把 TRAIN 改为 32/33。

## 3.2 seed start / step 完全不变

```text
TRAIN start = 220260001
VAL   start = 220270001
TEST  start = 220280001
step = 1
```

## 3.3 只扩大 maximum search envelope

冻结为：

```json
{
  "train": {
    "accepted": 64,
    "start": 220260001,
    "step": 1,
    "max_candidates": 1024
  },
  "val": {
    "accepted": 8,
    "start": 220270001,
    "step": 1,
    "max_candidates": 256
  },
  "test": {
    "accepted": 12,
    "start": 220280001,
    "step": 1,
    "max_candidates": 256
  }
}
```

说明：

- 这不是生成 1024 个训练模型；
- runner 仍然在得到第 64 个 eligible TRAIN case 后立即停止；
- `prepare_candidate` 具有 history-only 预筛，因此大量不合格 history 不需要完整 64³ materialization；
- accepted count、eligibility、geology recipe 完全不变；
- 不读取任何 seismic / evidence / Flow / downstream metric。

这是一个**新的 prospectively frozen cohort protocol**。Stage19 v1 的失败必须保留，不能把 v2 描述为“继续原正式运行”。

---

# 4. 必须增加的 StructuralGeo reuse preflight

这是本次最重要的复用防护。

现有 `build_cohort.py` 只显式检查：

```text
bounds
resolution
fixed wells
```

但它没有强制验证：

```text
StructuralGeo generator source
default Markov matrix
RNG implementation
conditioning implementation
prepare_candidate eligibility implementation
```

与原先成功 Full StructuralGeo benchmark 是否仍完全一致。

在生成 Stage19R cohort 前新增：

```text
scripts/stage19/audit_generator_reuse.py
```

或在 `build_cohort_v2.py` 的开头执行同等检查。

## 4.1 读取 authoritative reference

使用：

```text
experiments/full_structuralgeo_benchmark/benchmark_manifest.json
experiments/full_structuralgeo_benchmark/cases/fullgeo_case01/manifest.json
```

## 4.2 generator-critical source hash audit

至少验证旧 benchmark manifest 中记录的以下 source SHA：

```text
StructuralGeo-main/src/geogen/dataset/dataset.py
StructuralGeo-main/src/geogen/generation/categorical_events.py
StructuralGeo-main/src/geogen/generation/geowords.py
StructuralGeo-main/src/geogen/generation/model_generators.py
StructuralGeo-main/src/geogen/generation/rng_contract.py
StructuralGeo-main/src/geogen/model/geomodel.py
StructuralGeo-main/src/geogen/model/metaballs.py
StructuralGeo-main/src/geogen/probability/random_varibles.py
StructuralGeo-main/src/geogen/probability/sedimentbuilders.py
StructuralGeo-main/src/geogen/probability/wavegenerators.py
StructuralGeo-main/src/geogen/generation/markov_matrix/default_markov_matrix.csv
project/geodata-3d-conditional/boreholes.py
```

若任何 generator-critical hash 与旧 benchmark 不一致：

```text
STOP_GENERATOR_REUSE_MISMATCH
```

禁止自动继续。

不要因为 report/script 文本变化而误停；只比较真正影响 model generation / conditions 的 source。

## 4.3 historical deterministic replay gate

使用旧 benchmark 已知 accepted seed：

```text
root_seed = 120260003
```

调用当前 `prepare_candidate(120260003)`，必须验证至少：

```text
eligible == True
canonical truth tensor SHA
canonical condition_values SHA
canonical condition_mask SHA
markov_sequence
```

与：

```text
fullgeo_case01/manifest.json
```

保存的对应记录一致。

旧 manifest 已有 authoritative canonical hash，例如：

```text
tensor_content_hashes["truth/true_model.pt"]
tensor_content_hashes["condition/condition_values.pt"]
tensor_content_hashes["condition/condition_mask.pt"]
```

只有 exact replay 通过，才能生成新 cohort。

这一检查用于证明：

```text
Stage19R 确实复用了已经验证成功的 StructuralGeo generation + conditioning contract
```

而不是“文件名相同但实际 generator 已经发生变化”。

---

# 5. 修复 cohort failure provenance

现有 `build_cohort.py` 的一个明显缺口是：

```text
只有 split 达到 accepted target 后才写 registry；
如果中途因 insufficient eligible cases 失败，
examined candidate 的 rejection trace 不会被完整持久化。
```

因此当前我们只能知道：

```text
33/256 eligible
```

却不能从 formal artifact 中直接统计到底主要失败于：

```text
missing_fold_or_fault_event
missing_label9_producing_intrusion_event
final_raw_label9_absent
no_hidden_raw_label9_under_fixed_condition
```

这使失败原因不可审计。

Stage19R builder 必须在每个 split 中持续维护：

```text
candidate_trace
rejection_reason_counts
accepted_count
examined_count
```

至少在 split 完成或异常退出前写：

```text
cohort_v2/audit/train_candidate_trace.json
cohort_v2/audit/val_candidate_trace.json
cohort_v2/audit/test_candidate_trace.json
```

以及：

```text
cohort_v2/audit/rejection_summary.json
```

异常时也必须写完已有 trace 后再 raise。

注意：

这些 rejection statistics 只能用于 provenance/report，不能用于改变 eligibility。

---

# 6. 对 cohort config 的完整一致性验证

`build_cohort.py` 不应只验证 bounds/resolution/wells。

Stage19R 必须验证 `cohort_v2.json` 中以下字段逐项等于 authoritative Full StructuralGeo recipe：

```text
bounds
resolution
markov_matrix
height_tracking
normalize
fill_nans
raw_label_range
eligibility 全部字段
fixed_well_xy
```

不允许 config 中写了一套 eligibility，而代码实际上调用另一套 hard-coded `prepare_candidate` 规则却不报错。

建议新增 helper：

```python
validate_stage19_cohort_contract(config, reference_config)
```

只做 exact equality validation，不开发新选择逻辑。

---

# 7. Stage17A evidence reuse 也必须增加 hash enforcement

现有 Stage19 `evidence_v1.json` 记录的是路径，但 observation/evidence runner 没有完整强制校验 successful Stage17A authoritative asset hash。

Stage19R 不改变任何地球物理参数，但必须把 authoritative hash 写入 `evidence_v2.json`。

不要手工猜 hash；从：

```text
experiments/stage17_evidence_coupling_attribution/stage17a/formal_all5_v1/config.json
```

读取并冻结：

```text
binary_acoustic_config path + sha256
seismic_config path + sha256
inversion_config path + sha256
```

`build_observations` 和 `run_evidence` 开始前必须逐项验证。

同时继续验证：

```text
trace_samples = 320
refinement_passes = 2
prior_relative_weight = 0.001
vertical_smoothness_relative_weight = 0.01
full_vertical_trace_used = true
lateral_filter_used = false
thresholding = false
```

任何一项不一致：

```text
STOP_STAGE17A_REUSE_MISMATCH
```

不得继续。

---

# 8. 加固 TEST truth firewall

当前实际代码中：

```text
run_inference.py
```

本身没有调用 truth tensor，这是正确的。

但它读取的 `evidence_registry.json` 记录中仍包含：

```text
truth_assets
```

因此目前属于：

```text
代码没有读取 truth
```

而不是更严格的：

```text
inference 输入接口根本不提供 truth
```

Stage19R 必须建立一个 stripped TEST inference registry，例如：

```text
evidence/test_inference_registry.json
```

每个 TEST case 只允许包含：

```text
case_id
root_seed
condition values
condition mask
subsurface mask
binary_impedance_score
observation metadata/hash（如需要）
```

严格禁止出现：

```text
truth
truth_assets
binary_truth
true_model
history truth metadata
```

`run_inference.py` 只能读取这个 stripped registry。

`evaluate.py` 继续读取 full evaluation registry / cohort registry，并在 inference 完成后加载 truth。

新增静态/运行时测试：

```python
assert "truth_assets" not in serialized_inference_registry
assert "true_model" not in serialized_inference_registry
assert "binary_truth" not in serialized_inference_registry
```

---

# 9. 加固 formal adapter checkpoint 防误用

当前 `run_inference.py` 会检查：

```text
checkpoint schema
base checkpoint SHA
```

但不足以阻止误把 smoke / epoch1 adapter checkpoint 当 formal checkpoint。

Stage19R 必须要求 formal inference 同时读取：

```text
formal training_manifest.json
adapter_checkpoint.pt
```

并验证：

```text
training_manifest.run_status == "completed"
training_manifest.run_class == "formal_training"
training_manifest.smoke_subset == false
training_manifest.optimizer_updates == 1024
training_manifest.epochs == 4
training_manifest.base_model_unchanged == true
training_manifest.base_gradients_absent == true

adapter_checkpoint.schema == "stage19_adapter_checkpoint_v1"
adapter_checkpoint.epoch == 4
adapter_checkpoint.training_config_sha256 == current training config sha256
adapter_checkpoint.base_checkpoint_sha256 == frozen base checkpoint sha256
adapter_checkpoint.adapter_parameter_count < 100000
adapter checkpoint file SHA == training_manifest 中记录的正式 checkpoint SHA
```

任何一项失败：

```text
STOP_INVALID_FORMAL_ADAPTER_CHECKPOINT
```

这个修改只属于 engineering guard，不改变训练。

---

# 10. 不允许修改的 Stage19 scientific core

Stage19R 完成 cohort repair 后，以下内容保持 Stage19 v1 原样：

## Base Flow

```text
checkpoint:
demo_model/conditional-weights.ckpt

SHA256:
561e94bfda770ec41fc4cbed43436a7e2130eef5dfb7e5d666fcefc0724ff94c

weight_source = EMA
base parameters frozen
embedding unchanged
decoder unchanged
```

## Evidence

```text
Stage17A full-trace binary_impedance_score
1 channel
no threshold
q = score * subsurface
```

## Adapter

```text
base_width = 12
dilations = [1,2,4,1]
geophysics_channels = 1
max_residual_ratio = 0.25
zero initialized output
parameter_count < 100000
```

## Training

```text
AdamW
lr = 0.002
weight_decay = 0.0001
times = [0.2,0.4,0.6,0.8]
epochs = 4
batch_size = 1
optimizer updates = 1024
gradient clip = 1.0

flow_weight = 1.0
CE_weight = 0.25
Dice_weight = 0.25
residual_regularizer = 1e-4
logit_temperature = 0.1

checkpoint = epoch4 final
no early stopping
no scheduler
```

禁止：

```text
增加 topology loss
增加 seismic training loss
改 learning rate
改 adapter width
改 residual cap
改训练次数
改 T 采样
加 raw seismic encoder
解冻 Flow
做 parameter sweep
```

---

# 11. 正式执行顺序

## Step 0 — repository audit

记录：

```bash
git status --short
git branch --show-current
git rev-parse HEAD
```

确认 Stage19 v1 artifacts 不被修改。

## Step 1 — generator reuse audit

运行：

```text
audit_generator_reuse.py
```

必须输出：

```text
GENERATOR_REUSE_VALIDATED
```

否则停止。

## Step 2 — build `cohort_v2`

只按新的 search envelope 生成：

```text
64 / 8 / 12
```

如果 1024/256/256 仍不足：

```text
STOP_COHORT_V2_UNAVAILABLE
```

并保存完整 rejection trace。

禁止再次扩 budget。

## Step 3 — build observations

复用 Stage17 observation model。

要求每 case：

```text
forward_closure_max_abs <= 1e-7
```

## Step 4 — run Stage17A evidence

不加载 truth。

## Step 5 — VAL evidence reuse gate

保持原 Stage19 gate：

```text
>= 6/8 AP skill > 0
median AP skill >= 0.30
>= 6/8 correct-case specificity positive
```

失败：

```text
STOP_BEFORE_ADAPTER_TRAINING
```

不得调 inversion。

## Step 6 — CUDA smoke

仅验证工程：

```text
adapter train smoke
test inference smoke
formal checkpoint guard
truth firewall
condition exactness
```

smoke 不进入科学结论。

## Step 7 — exactly one formal adapter training

只跑 frozen config：

```text
4 epochs
1024 updates
```

## Step 8 — exactly one formal TEST inference

```text
12 cases
3 source seeds
5 arms
= 180 outputs
```

## Step 9 — retrospective evaluation

保持 Stage19 v1 evaluator/gates，不重新设计评价指标。

## Step 10 — STOP

完成后等待人工讨论。

---

# 12. 必须增加/修改的测试

至少增加以下 focused tests：

```text
1. historical seed 120260003 exact generator replay hash passes
2. generator-critical source hash mismatch -> hard stop
3. default Markov matrix hash mismatch -> hard stop
4. cohort_v2 与 cohort_v1 的科学差异只有 candidate search budget
5. cohort failure 时 candidate trace / rejection counts 仍写出
6. eligibility config 与 reference 不一致 -> hard stop
7. Stage17A source asset hash mismatch -> hard stop
8. TEST inference registry 不包含任何 truth pointer
9. run_inference 拒绝 smoke/epoch1 checkpoint
10. run_inference 拒绝 training-config SHA mismatch
11. run_inference 拒绝 non-formal training manifest
12. scale-zero FLOW_ONLY equivalence 继续通过
13. condition projection 继续 exact
14. base Flow hash training 前后完全不变
```

不要新增与新科学方法有关的 tests。

---

# 13. 对 Stage19R 结果的解释规则

## 如果 cohort_v2 成功

只说明：

```text
cohort availability engineering prerequisite repaired
```

不能把 cohort 构建成功当科学结果。

随后按原 Stage19 继续 evidence gate / adapter training。

## 如果 evidence gate 失败

解释为：

```text
Stage17A evidence pipeline does not generalize sufficiently to the new frozen cohort
```

此时停止，不训练 adapter。

## 如果 adapter training/inference 完成

才允许讨论：

```text
learned evidence-to-velocity coupling
correct-vs-zero/wrong specificity
hard seismic
volume calibration
topology/fragmentation
global geology preservation
```

---

# 14. 当前最重要的研发纪律

这次不要因为 Stage19 v1 停止而重新发明方法。

当前最合理的操作是：

\[
\boxed{
\text{保留同一科学实验}
+
\text{修复 cohort search budget}
+
\text{加强 reuse / truth / checkpoint 审计}
}
\]

不要：

```text
把 TRAIN 改成 33
降低 eligibility
根据已有 33 cases 的形态挑 case
调 Stage17A inversion
修改 adapter
加 topology loss
加 physics loss
解冻 Flow
```

Stage19 v1 的失败已经告诉我们的是**数据 cohort 可用性预算错误**，而不是 learned adapter 不可行。

---

# 15. Codex 最终交付要求

完成 Stage19R 开发后，必须报告：

1. repository starting commit/status；
2. Stage19 v1 immutable artifact audit；
3. generator-critical source/hash comparison；
4. historical seed replay exact hash result；
5. `cohort_v2.json` 与 v1 的字段级 diff；
6. candidate acceptance/rejection summary；
7. focused tests exact commands/results；
8. cohort_v2 exact execution command/result；
9. Stage17A source-hash audit；
10. VAL evidence gate；
11. CUDA smoke；
12. formal training manifest；
13. formal adapter checkpoint provenance；
14. truth-blind inference registry audit；
15. 180-output formal inference；
16. case-first evaluation；
17. 说明 Stage19R 改动是否只修复 cohort/provenance，而未改变科学方法；
18. 更新 `docs/DEVELOPMENT_HANDOFF.md`；
19. STOP，等待下一步研究讨论。
