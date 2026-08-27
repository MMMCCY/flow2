# Stage16 D-Flow source optimization 中文开发报告

日期：2026-08-24  
任务：`stage16_dflow_source_optimization`  
实现基线：分支 `stage15-binary-seismic-consensus`，commit
`357796884b86f4f7de97cabc2b85077b6313a6a1`

> 仓库在任务开始时并不位于用户文字所述的 `main`：本地和计算服务器均位于
> 上述 Stage15 分支；`main`/`origin/main` 是更早的
> `123d4dc3a1a33b8f03ff5b98894747b921a98fcd`。由于 Stage15 authoritative
> 资产在当前分支才完整存在，而且工作树已有用户未提交变更，本开发没有切分支、
> reset、覆盖或改写既有实验结果。仓库此前没有 Stage16，故使用请求中的编号。

## A. 实际阅读的既有文件

开发前阅读和交叉核对了以下实现与协议：

- 根目录 `AGENTS.md`，以及 `docs/AGENTS.md`、`PROJECT_BASELINE.md`、
  `EXPERIMENT_PROTOCOL.md`、`DEVELOPMENT_HANDOFF.md`、`RESEARCH_GOAL.md`；
- `inference_runtime.py`、`model_train_sh_inference_cond.py`；
- `guidance/probability_sampling.py`、`probability_volume.py`、
  `probability_evaluation.py`、`property_sampling.py`、`property_volume.py`、
  `property_evaluation.py`、`seismic.py`、`binary_seismic_inversion.py`；
- `guided_geophysical_sampling.py`；
- `scripts/stage14/run_gansim_style_geo_guidance.py`；
- `scripts/stage15/common.py`、`run_flow_demo.py`、
  `run_inversion_score_probability.py`、观测构建及 coarse inversion runner；
- Phase1 `README.md`、`docs/PHASE1_REPORT.md`、
  `reports/phase1b_v4_12pair/{REPORT.md,summary.json,paired_samples.csv}` 及其
  authoritative target/ROI/condition 资产；
- Phase2 `README.md`、`docs/PHASE2A_REPORT.md`、
  `configs/ideal_distinct_density_proxy_v1.json`、
  `reports/phase2a_v1_12pair/{REPORT.md,summary.json,paired_samples.csv}` 及其
  property table/target/confidence 资产；
- `experiments/stage15_binary_seismic_consensus/` 下 binary acoustic、
  observation、flow demo、inversion、报告和配置；
- `D-Flow_Differentiating_through_Flows_for_Controlled_Generation.pdf` 全文转换
  后核对。论文依据是 Ben-Hamu 等 ICML 2024 的 Algorithm 1：冻结生成 Flow，
  仅优化 source point，通过完整 ODE solve 反传；论文也讨论 Gaussian source
  shell regularization、LBFGS 与 gradient checkpointing。本阶段依项目协议采用
  authoritative 32-step fixed Euler，而不是照搬论文的特定 NFE/求解器设置。

由此确认：状态为 `[B,15,64,64,64]`；`embed` 将 raw label `-1..13` 映射到
category `0..14` 的 15 维嵌入，`decode` 以 cosine nearest embedding 返回
category，runner 再减一得到 raw geology；EMA loader 保留原始
`embedding.weight`，对 411 个可训练 state entries 应用 EMA。

## B. authoritative baseline sampler 逻辑

当前基线先把条件嵌入并将 source 的条件位置投影为条件 embedding。第 `k` 个
step 使用 midpoint 时间

`t_k = (k + 0.5) / 32, dt = 1 / 32`

计算 `velocity = model.net(state, conditioning, time)`，执行
`candidate = state + dt * velocity`，然后调用已有 `_project_conditions`，用
`torch.where` 在钻井/地表条件位置重新写入固定 embedding。该投影在初始时和每个
Euler step 后都执行。最终由 `model.decode(state) - 1` 得到离散地质。

## C. D-Flow 与旧 local guidance 的数学和代码区别

旧方法在每一个中间状态计算局部损失梯度，并把 `-g_t` 直接加到 instantaneous
velocity：它改变的是 ODE 的逐步速度场。D-Flow 不向 velocity 加任何 guidance；
它保持完整冻结 Flow map `Phi_theta` 不变，在 endpoint 计算
`L(Phi_theta(x0))`，通过 32 个 Euler step 的计算图取得
`(d x1 / d x0)^T grad_x1 L`，optimizer 唯一更新 `x0`。因此两者在数学对象、
梯度路径和代码入口上严格不同。

## D. 新增/修改文件

- `guidance/dflow.py`：通用 differentiable solver、source optimizer、冻结和
  state-dict hash 审计；
- `scripts/stage16/__init__.py`；
- `scripts/stage16/run_dflow.py`：三个 objective 共用的 truth-firewalled runner；
- `scripts/stage16/evaluate_dflow.py`：采样完成后的独立 truth-aware evaluator；
- `tests/test_dflow.py`：9 个 CPU tiny-flow 测试；
- `experiments/stage16_dflow_source_optimization/configs/*.json`：三个冻结配置；
- `experiments/stage16_dflow_source_optimization/README.md` 与本报告；
- 三个 objective 目录、reports 目录的 `.gitkeep`；
- 三个 one-seed smoke run 及其 retrospective evaluation immutable 输出。

既有 Phase/Stage 结果、checkpoint、EMA、embedding 和训练代码均未修改。

## E. differentiable solver 如何避免 `no_grad`

`differentiable_flow_solve` 内对 `model.net` 的调用没有任何
`torch.no_grad()`；开启 checkpoint 时使用
`torch.utils.checkpoint.checkpoint(model.net, ..., use_reentrant=False)`。
模型参数 `requires_grad=False` 只阻止参数梯度，不切断 evolving `state` 到
velocity 的图。不可变的 conditioning/embedded condition 会 detach，以避免
跨 optimizer iteration 复用旧 embedding graph，但 source/state 从未 detach。

## F. 模型参数冻结和零变化保证

加载 EMA 后立即 `model.eval()`，并对每个参数调用
`requires_grad_(False)`、清空 `.grad`。运行前断言 trainable model parameter
count 为 0；optimizer 参数 ID 集合必须且只能等于 source parameter ID。
运行前后计算完整 aggregate state-dict SHA256 和每个 tensor SHA256，逐项断言
完全一致，并断言模型参数不存在 gradient。三个 CUDA smoke run 均记录
`model_state_unchanged=true`。

## G. hard-condition projection

没有复制投影实现；直接复用 `guidance.probability_sampling` 的
`_project_conditions` 和 `_condition_violations`。投影使用 `torch.where`，受约束
位置恒等替换为固定 embedding，非约束位置的梯度保留。每一步检查 embedding
max absolute error，endpoint 再 hard decode 检查。全部 smoke optimization
iteration 和最终输出均为 0 violations。

## H. 显存处理

配置固定 `use_gradient_checkpointing=true`；每个 Euler step 的 `model.net`
采用 non-reentrant activation recomputation，不保存完整 U-Net 中间 activation。
网格保持 `64^3`，channel 15，step 保持 32，未降尺寸、未减 solver step、未换
科学问题。RTX 4090 D smoke 观测显存约 7.0 GiB，无 OOM。

## I. 三个 objective 的复用关系

- Oracle：复用 Phase1 target probability volume、target/core/ROI、condition 和
  calibrated soft BCE + hard Dice 定义；endpoint temperature 固定为
  authoritative final `tau=0.1`。
- Property：复用 Phase2A 两通道 property codebook、完整 target property、
  confidence、matched multiscale MSE（sigma 0/1.5/3，权重 0.5/0.3/0.2，
  channel 权重 0.5/0.5）；endpoint `tau=0.1`。
- Seismic：复用 Stage15 `observed_seismic.pt`、binary acoustic config、condition、
  TWT normal-incidence forward 和 25 Hz Ricker wavelet；soft label-9 occupancy 映射
  到 binary acoustic property，以 raw seismic MSE 为唯一梯度 objective。hard
  decode seismic MSE 只作 detached diagnostic，没有加入 straight-through、
  多尺度、topology 或新网络。

## J. seismic runner 的 truth firewall

是，seismic inference 完全 truth-blind。其 required asset key 集合不含
`truth_model`，并有程序级 `seismic_runner_requires_truth() == False` 检查和 CPU
单测。runner manifest 同时写入
`truth_loaded_by_runner=false` 与
`seismic_truth_loaded_by_runner=false`。runner 不读取 Stage15 observation
manifest，以避免通过 provenance 间接加载 truth。只有 run status 已为 complete
后，`evaluate_dflow.py` 才读取 truth，evaluation manifest 写明
`truth_role=retrospective_evaluation_only`。

## K. 测试命令与结果

```bash
PYTHONPATH=project/geodata-3d-conditional:src .venv/bin/pytest -q \
  project/geodata-3d-conditional/tests/test_dflow.py
```

结果：`9 passed in 2.13s`。覆盖“只改 x0”、非零有限 through-flow gradient、
硬条件、zero-optimization pair、确定性、mock loss 下降、checkpoint on/off、
seismic API 无 truth、model hash 不变。

```bash
PYTHONPATH=project/geodata-3d-conditional:src .venv/bin/pytest -q \
  project/geodata-3d-conditional/tests/test_stage1_probability_guidance.py \
  project/geodata-3d-conditional/tests/test_phase2_property_volume.py \
  project/geodata-3d-conditional/tests/test_phase2_property_sampling.py \
  project/geodata-3d-conditional/tests/test_phase4_seismic.py \
  project/geodata-3d-conditional/tests/test_stage15_binary_seismic_consensus.py \
  project/geodata-3d-conditional/tests/test_stage15_coarse_binary_seismic.py
```

结果：`74 passed in 2.31s`。

```bash
PYTHONPATH=project/geodata-3d-conditional:src .venv/bin/pytest -q \
  project/geodata-3d-conditional/tests
```

结果：`334 passed, 2 failed, 13 warnings in 15.48s`。两个失败均为本任务前已存在的
论文证据生成物漂移：`paper_evidence_summary` 中记录的 generation commit
`cd4b097...` 与当前 HEAD 不同；已有 manifest
`figure03_cfm_structured_cuboid_contrast.json` 未进入既有 figure-link 清单。
它们不扫描 Stage16 输出，也不是 D-Flow 改动引入。遵守“不修改无关论文资产”，
本任务未修复这两项。

此外，新文件 `py_compile`、配置 asset hash preflight、`git diff --check` 均通过。

## L. smoke experiment 命令与观察结果

以下命令在用户指定服务器的当前仓库目录、RTX 4090 D 上执行。三个配置在运行前
均为 `status=frozen_before_run`，固定 Adam、lr 0.01、20 iterations、source
regularization 0，不因结果修改。

```bash
PYTHONPATH=project/geodata-3d-conditional:src .venv/bin/python \
  project/geodata-3d-conditional/scripts/stage16/run_dflow.py \
  --config project/geodata-3d-conditional/experiments/stage16_dflow_source_optimization/configs/oracle_probability_v1.json \
  --output-dir project/geodata-3d-conditional/experiments/stage16_dflow_source_optimization/oracle_probability/smoke_seed42_v1 \
  --smoke

PYTHONPATH=project/geodata-3d-conditional:src .venv/bin/python \
  project/geodata-3d-conditional/scripts/stage16/run_dflow.py \
  --config project/geodata-3d-conditional/experiments/stage16_dflow_source_optimization/configs/property_v1.json \
  --output-dir project/geodata-3d-conditional/experiments/stage16_dflow_source_optimization/property/smoke_seed42_v1 \
  --smoke

PYTHONPATH=project/geodata-3d-conditional:src .venv/bin/python \
  project/geodata-3d-conditional/scripts/stage16/run_dflow.py \
  --config project/geodata-3d-conditional/experiments/stage16_dflow_source_optimization/configs/seismic_v1.json \
  --output-dir project/geodata-3d-conditional/experiments/stage16_dflow_source_optimization/seismic/smoke_seed42_v1 \
  --smoke
```

每个 run 完成后分别执行：

```bash
PYTHONPATH=project/geodata-3d-conditional:src .venv/bin/python \
  project/geodata-3d-conditional/scripts/stage16/evaluate_dflow.py \
  --run-dir <上述已完成的 run-dir>
```

seed 42 smoke 的关键结果如下；这只是 one-seed 机制检查，不构成 formal n-sample
科学结论：

| objective | FLOW_ONLY endpoint | DFLOW endpoint | 离散 retrospective 观察 |
|---|---:|---:|---|
| Oracle probability | 2.056511 | 1.784927 | target IoU `+0.013628`，precision `+0.452992`，recall `-0.003345`，mIoU `-0.041572` |
| Property | 1.540703 | 0.831861 | mIoU `+0.038209`，target IoU `-0.005566`，precision `+0.593909`，recall `-0.023974` |
| Seismic soft MSE | 0.00168326 | 0.00115233 | hard MSE 亦由 `0.00175883` 降到 `0.00119108`，但 accuracy `-0.156017`、mIoU `-0.078792`、target IoU `-0.002696` |

三个 case 的首轮 source gradient norm 分别为 `0.139186`、`0.322844`、
`0.00113071`，source update norm 非零；说明 through-Flow gradient 和 source
optimization 实际发生。结果没有触发任何自动调参。特别是 seismic 的物理失配
改善与地质指标恶化并存，是必须保留的负向机制证据。

## M. 输出文件位置

- Oracle：`oracle_probability/smoke_seed42_v1/`
- Property：`property/smoke_seed42_v1/`
- Seismic：`seismic/smoke_seed42_v1/`
- 各自 retrospective evaluation：上述目录下 `evaluation/`
- 配置：`configs/{oracle_probability_v1,property_v1,seismic_v1}.json`
- 本报告：`reports/DEVELOPMENT_REPORT.md`

每个 run 保存冻结 config 和 hash、checkpoint/input asset hashes、git branch/commit、
model load report、前后 tensor hashes、run manifest、全局及 sample trace、paired
manifest、initial/optimized source、FLOW_ONLY/DFLOW continuous state、decoded geology
及其 SHA256。每个 objective 目录约 65 MiB，未覆盖已有结果。

## N. 风险与未解决问题

1. 当前只有每个 objective 的 seed-42 smoke；formal preregistered seed 集合尚未运行，
   不能从单样本判定 Oracle/Property/Seismic PASS 或 FAIL。
2. Adam 20-step objective trace 不保证单调，尤其 seismic 早期有震荡；不能据此改
   optimizer、lr 或 iterations。formal 配置已冻结，不允许 sweep。
3. Soft categorical endpoint 仍可能利用连续 embedding/acoustic mapping；虽然本次
   hard seismic MSE 也下降，但 soft-hard gap 和地质恶化表明 observability/likelihood/
   decode geometry 仍是主要风险。
4. Oracle one-seed endpoint objective 下降但全局 mIoU 下降；按任务判据，formal
   Oracle 若失败应先审计 reachability/optimizer 实现，不能直接进入复杂方法。
5. 本阶段有意未实现 straight-through、hard-aware、coarse-to-fine、DPS/PGDM、
   MCMC、topology loss、adapter、fine-tuning 或新网络训练；进入下一阶段需用户确认。
6. 开发起点实际不是 `main`。后续合并/提交前必须由维护者决定如何处理当前分支差异
   和工作树中既有的文档删除；本任务没有触碰这些用户变更。

