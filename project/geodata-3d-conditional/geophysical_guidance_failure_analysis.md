# Inference-Time Geophysical Guidance Failure Analysis

本文档结合当前代码和已有运行结果，分析为什么在推理阶段新增的地球物理制导没有达到预期的地质恢复效果。这里的“预期效果”特指：在不改训练、不改 U-Net、不重训模型的前提下，仅通过采样阶段的地球物理梯度，使最终离散岩性模型在地球物理一致性和目标地质结构上都显著改善。

结论先行：当前制导不是完全没有生效。它确实能在连续 embedding 空间中降低轻量地球物理 proxy misfit；但这个下降大多没有稳定转化为最终 `model.decode(...).argmax` 之后的离散岩性结构变化。因此它更像一个弱的连续状态扰动，而不是一个真正的地球物理条件生成或反演机制。

## 1. 原模型学到的是什么

当前条件模型定义在 `model_train_sh_inference_cond.py` 的 `Geo3DStochInterp` 中。训练阶段先把离散岩性标签映射到 embedding 空间：

- `embed()` 中先执行 `x.squeeze(1).long() + 1`，把标签 `-1..13` 映射到 embedding index `0..14`。
- embedding 后的张量形状是 `[B, E, X, Y, Z]`。
- `decode()` 再把连续 embedding 向量归一化，与固定 embedding table 做 cosine similarity，并对类别维取 `argmax`。

关键代码位置：

- `model_train_sh_inference_cond.py:299-302`: embedding 初始化并冻结。
- `model_train_sh_inference_cond.py:344-355`: 离散标签到 embedding 的 `embed()`。
- `model_train_sh_inference_cond.py:358-388`: embedding 到离散类别的 `decode()`。

训练目标也很重要。`training_step()` 中，模型不是学习地球物理观测条件下的后验分布，而是学习从噪声 embedding 到真实 geology embedding 的 stochastic interpolation velocity：

- `X1 = self.embed(batch)` 得到真实 geology embedding。
- `X0 = torch.randn_like(X1)` 作为噪声初态。
- `XT, VT = self.interpolator.flow_objective(T, X0, X1)`。
- `VT_hat = self.net(XT, ATb, T)`。
- loss 主要是 `VT_hat` 与 `VT` 的 normalized MSE，加一个 borehole/surface 条件重建项。

关键代码位置：

- `model_train_sh_inference_cond.py:413-431`: 构造 `X0, X1, XT, VT, VT_hat`。
- `model_train_sh_inference_cond.py:440-451`: `flow_loss + reconstruct_loss`。
- `src/flowtrain/interpolation/interpolation.py:379-406`: 当前 `LinearInterpolant` 的 `alpha=1-t, beta=t, alpha_dot=-1, beta_dot=1`。

这意味着模型先验只知道两类条件：随机噪声初态和 `ATb` 形式的 borehole/surface 条件。它没有见过地球物理观测，也没有学过地球物理 misfit 的梯度方向是否对应合理地质结构。

## 2. 原始推理路径和新增制导路径并不一样

原始条件推理在 `model_inference_experiments.py` 中：

- 通过 wrapper `dxdt_cond(x, time)` 调用 `model.net.forward(x, ATb=ATb, time=time)`。
- 使用 `ODEFlowSolver(model=dxdt_cond, rtol=1e-6)`。
- `ODEFlowSolver.solve()` 内部用 `torchdiffeq.odeint(..., method="dopri5")` 做自适应 ODE 积分。
- 时间范围是 `t0=0.0001, tf=0.9999`。

关键代码位置：

- `model_inference_experiments.py:200-205`: 构造条件 velocity wrapper 和 ODE solver。
- `model_inference_experiments.py:237-239`: 调用 solver。
- `src/flowtrain/solvers/solvers.py:40-77`: `ODEFlowSolver.solve()`。

新增的 `guided_geophysical_sampling.py` 没有复用这个 solver，而是独立实现了固定步长 Euler：

- `dt = 1.0 / n_steps`。
- 每步取 `t_value = (step + 0.5) / n_steps`。
- 先算 `v_prior = model.net(x, conditioning, t)`。
- 再算地球物理 loss 对 `x` 的梯度。
- 最后 `v = v_prior - guidance_velocity`，并执行 `x = x.detach() + dt * v.detach()`。

关键代码位置：

- `guided_geophysical_sampling.py:282-402`: `guided_euler_sample()`。
- `guided_geophysical_sampling.py:340-342`: 当前状态重新 `requires_grad`，并计算 prior velocity。
- `guided_geophysical_sampling.py:360-370`: 计算 geophysical gradient 和合成 guided velocity。
- `guided_geophysical_sampling.py:400`: Euler 更新。

这带来两个后果：

1. baseline 与 guided 不再完全等价于原始推理路径，只是同一脚本内 `mu/alpha=0` 与 `>0` 可比。
2. 地球物理梯度被直接加到 velocity field 上，但这个 velocity field 原本是按训练分布学习的；外加梯度是否仍落在地质先验流形上没有保证。

## 3. 制导损失优化的是软概率密度，不是最终离散岩性

新增制导的 differentiable path 是：

```text
x
  -> soft_decode_to_probs(x, embedding_weight, tau)
  -> probs_to_density(probs, property_map)
  -> forward_model(density)
  -> normalized_misfit(predicted, observed)
```

关键代码位置：

- `guided_geophysical_sampling.py:34-59`: `soft_decode_to_probs()`。
- `guided_geophysical_sampling.py:62-83`: `probs_to_density()`。
- `guided_geophysical_sampling.py:90-102`: 单一 gravity loss。
- `guided_geophysical_sampling.py:105-173`: multi-physics loss。

`soft_decode_to_probs()` 与 `model.decode()` 使用了同一个 embedding table 和 cosine similarity，这一点是合理的。但两者的输出机制不同：

- 制导阶段：`softmax(similarities / tau)`，得到连续概率。
- 保存阶段：`argmax(logits)`，得到离散类别。

最终保存代码：

- `guided_geophysical_sampling.py:649`: `decoded = (model.decode(final_state).detach().cpu() - 1)[0]`。

这就是一个核心断层：地球物理 loss 可以通过微小改变类别概率来降低 predicted gravity，但只要 embedding 向量没有跨过 `argmax` 分类边界，最终岩性标签就不会变。已有结果正好印证这一点：

- `rel_alpha_0_05` 的 decoded changed voxel fraction 平均约 `0.000593`。
- `rel_alpha_0_10` 的 decoded changed voxel fraction 平均约 `0.001369`。
- `cond_generation_1_label10/guided_alpha0_05` 平均约 `0.001131`。

也就是说，大多数 guided run 最终只改变约千分之一量级的体素。对于希望恢复连续地质体、岩脉或局部目标结构的任务，这个变化幅度太小。

## 4. 制导强度和模型先验速度不在一个量级

`build_guidance_velocity()` 支持两种模式：

- absolute: `guidance_velocity = mu * w_t * grad_geo`
- relative: `guidance_velocity = alpha * w_t * ||v_prior|| * grad_geo / ||grad_geo||`

关键代码位置：

- `guided_geophysical_sampling.py:225-279`: `build_guidance_velocity()`。
- `guided_geophysical_sampling.py:176-206`: `guidance_weight()`。
- `guided_geophysical_sampling.py:209-218`: `clip_gradient_by_norm()`。

### 4.1 absolute 模式的问题

absolute 模式下，`grad_geo` 先经过 `clip_gradient_by_norm()`，默认 `grad_clip_norm=1.0`。因此在 active guidance 时间段内：

```text
||guidance_velocity|| <= mu * w_t
```

而已有 trace 中 `prior_velocity_norm` 通常在 `~2000` 量级。默认 `mu=0.01` 或 `mu=1.0` 时，外加速度与 prior velocity 相比几乎可以忽略。sweep 结果也说明这一点：

| run | geo_misfit_mean | voxel_accuracy_mean | mean_iou_mean |
| --- | ---: | ---: | ---: |
| `mu_0` | 0.200613 | 0.606055 | 0.166507 |
| `mu_001` | 0.200610 | 0.606056 | 0.166507 |
| `mu_01` | 0.200610 | 0.606055 | 0.166506 |
| `mu_1` | 0.200612 | 0.606054 | 0.166505 |
| `mu_500` | 0.200184 | 0.606059 | 0.166507 |
| `mu_10000` | 0.146048 | 0.606094 | 0.166506 |

只有 `mu=10000` 明显降低了 proxy misfit，但 voxel accuracy 和 mean IoU 仍基本不变。这说明强行加大 absolute gradient 可以改变连续 proxy response，却仍然难以形成有意义的离散地质结构变化。

数据来源：

- `guided-results/guidance_sweep_summary.csv:2-7`。

### 4.2 relative 模式的问题

relative 模式规避了 raw gradient 尺度问题，把 guidance norm 绑定为 `alpha * w_t` 倍 prior norm。这比 absolute 模式更可控，但默认 alpha sweep 仍偏弱：

| run | geo_misfit_mean | voxel_accuracy_mean | mean_iou_mean | borehole_consistency_mean |
| --- | ---: | ---: | ---: | ---: |
| `rel_alpha_0` | 0.212686 | 0.551294 | 0.145969 | 0.999964 |
| `rel_alpha_0_01` | 0.209421 | 0.551334 | 0.145979 | 0.999957 |
| `rel_alpha_0_03` | 0.193612 | 0.551332 | 0.145980 | 0.999874 |
| `rel_alpha_0_05` | 0.178490 | 0.551343 | 0.145980 | 0.999760 |
| `rel_alpha_0_10` | 0.166567 | 0.551391 | 0.145990 | 0.999606 |

地球物理 proxy misfit 随 alpha 下降，但 voxel accuracy 和 mean IoU 几乎不动。`rel_alpha_0_10` 比 baseline `rel_alpha_0` 的 proxy misfit 降低明显，但 decoded change ratio 仍只有约 `0.001369`。

数据来源：

- `guided-results/guidance_sweep_summary.csv:8-12`。
- `guided-results/rel_alpha_0_10/decoded_change_ratio.csv`。

这说明 relative 模式的外加速度能影响连续地球物理响应，但依然没有有效推动 embedding 跨越类别边界，或者跨越后没有形成空间连贯的目标结构。

## 5. 时间调度让制导主要发生在后半段，过早/过晚都有问题

默认 `guidance_schedule="late_quadratic"` 且 `guidance_start=0.5`：

```python
return 0.0 if t < start else ((t - start) / (1.0 - start)) ** 2
```

关键代码位置：

- `guided_geophysical_sampling.py:176-206`: `guidance_weight()`。

这种设计是为了避免在早期破坏生成先验，但它也带来一个限制：前半段完全没有地球物理梯度，后半段地质结构已经被 prior velocity 推向某个模式，外加梯度只能做末端修正。对于需要改变目标体位置、连通性、体积的任务，末端修正通常太晚。

如果把制导提前或加大，又会引入另一个风险：外加梯度没有训练过，可能把状态推离模型先验流形，产生地质不合理样本。当前代码没有任何机制约束 guidance 后的状态仍是模型高概率区域。

## 6. 地球物理 forward model 是轻量 proxy，非唯一性强

`geophysics.py` 中的 `SimpleGravityForward` 是一个轻量表面 gravity proxy：

- 输入密度体 `[B, 1, X, Y, Z]`。
- 对每个深度切片做 2D 卷积。
- 沿深度累加得到 `[B, 1, X, Y]` 的 surface anomaly。
- 默认 `remove_mean=True`，会去掉场均值。

关键代码位置：

- `geophysics.py:139-230`: `SimpleGravityForward`。
- `geophysics.py:424-466`: `normalized_misfit()`。

这个 proxy 对筛选和演示是有用的，但作为推理制导目标有天然局限：

1. 从 3D 岩性到 2D gravity field 是多对一映射，非唯一性很强。
2. 去均值后，绝对密度偏移信息被弱化。
3. 不同岩性组合可能产生相似 gravity anomaly。
4. 局部目标结构的可见性依赖密度 contrast、体积、深度和周围背景。
5. proxy misfit 下降不代表目标标签 IoU、recall 或地质连通性改善。

候选筛选结果已经显示这种分离：

- `cond_generation_1_label7`: `proxy misfit improves but target-label metrics do not improve enough`。
- `paper_cond_gen_0_label8`: 同样是 proxy misfit 改善但目标标签指标不足。
- `cond_generation_1_label10` 和 `paper_cond_gen_0_label7` 是较好的 demo candidate，但 target IoU/recall 改善也很小，更多是“可展示的轻微改善”，不是强反演结果。

数据来源：

- `dike-demo-manual/candidate_screening/candidate_screening.csv:2-5`。

## 7. 当前 ATb 条件只来自 borehole/surface，没有地球物理条件通道

新增脚本构造 `ATb_lith` 时保持了原推理约定：

```python
boreholes_mask = (boreholes != -1) | (truth == -1)
embedded_truth = model.embed(truth)
embedded_mask = boreholes_mask.expand(-1, embedded_truth.shape[1], -1, -1, -1)
ATb_lith = embedded_truth * embedded_mask
```

关键代码位置：

- `guided_geophysical_sampling.py:554-558`。
- `model_inference_experiments.py:283-291`。

这保证了兼容原条件模型，但也说明地球物理观测没有进入 U-Net 的条件分支。地球物理只在采样 ODE 外部通过 gradient correction 作用。模型内部不知道这个 correction 的语义，也不会主动学习如何用地球物理观测解释未约束体素。

因此，地球物理 guidance 和 learned prior 之间可能发生冲突：

- prior velocity 倾向于生成训练分布中的合理 geology。
- geophysical gradient 倾向于降低当前 proxy misfit。
- 二者没有共同训练过，`v_prior - guidance_velocity` 不一定仍对应合理的 posterior velocity。

## 8. borehole consistency 很高并不代表整体地质恢复成功

评价中 `borehole_consistency` 常接近 1，这是因为条件区域本来很小，而且 `ATb` 强约束了 borehole/surface。它只能说明条件点附近大体保住了，并不说明未观测区域的目标结构恢复成功。

例如 relative sweep 中：

- `rel_alpha_0` 的 `borehole_consistency_mean = 0.999964`。
- `rel_alpha_0_10` 的 `borehole_consistency_mean = 0.999606`。

但同一组的 `voxel_accuracy_mean` 只有约 `0.551`，`mean_iou_mean` 只有约 `0.146`，并且随 alpha 基本不动。

数据来源：

- `guided-results/guidance_sweep_summary.csv:8-12`。

所以，当前制导主要改善的是 geophysical proxy misfit，不是全局 geology accuracy，也不是稳定的 target structure recovery。

## 9. 逐项对应“为什么达不到预期”

### 原因 A：梯度路径与最终离散输出脱节

代码证据：

- `soft_decode_to_probs()` 使用 softmax 概率。
- `probs_to_density()` 用概率期望得到连续 density。
- 最终 `model.decode()` 使用 hard argmax。

结果证据：

- decoded change ratio 通常只有 `0.0005-0.0014` 量级。

影响：

- proxy field 能连续变化。
- 离散岩性标签几乎不变。
- 即使变，也多为零散体素，不一定形成地质结构。

### 原因 B：默认制导强度太小，强制导也只改善 proxy

代码证据：

- absolute 模式受 `grad_clip_norm` 和 `mu` 限制。
- relative 模式受 `alpha * w_t` 限制。
- 默认 `late_quadratic` 让前半程 guidance 为 0。

结果证据：

- `mu=0.01/0.1/1.0/500` 与 `mu=0` 基本一致。
- `mu=10000` 降低 proxy misfit，但 voxel/IoU 基本不动。
- relative alpha 增大能降 proxy misfit，但 voxel/IoU 基本不动。

影响：

- 小 guidance 过弱。
- 大 guidance 也没有明确地质语义，容易只调 proxy response。

### 原因 C：地球物理 proxy 非唯一，不能直接约束三维标签

代码证据：

- `SimpleGravityForward` 把 3D density 压缩到 2D surface anomaly。
- `normalized_misfit()` 只比较场残差 RMS。

结果证据：

- candidate screening 中存在 proxy misfit 改善但 target-label metrics 不改善的 case。

影响：

- 优化 gravity proxy 可以通过很多非目标结构的方式实现。
- 对局部岩脉、目标标签 recall/IoU 的约束不足。

### 原因 D：模型没有学习地球物理条件 posterior

代码证据：

- 训练 loss 没有 geophysical observation input。
- U-Net 条件输入只有 `ATb`。
- 新增 guidance 是推理时外部修改 velocity。

影响：

- `v_prior - guidance_velocity` 不是训练过的 velocity field。
- 没有保证 guidance 后的状态仍在 geology manifold 上。
- 地球物理梯度和 learned prior 只能临时折中。

### 原因 E：固定步长 Euler 改变了原推理数值路径

代码证据：

- 原推理使用 adaptive `dopri5`。
- 新脚本使用固定步长 Euler。

影响：

- 与原模型推理分布存在数值差异。
- 制导效果和 Euler 误差、步数、时间取点耦合。
- 当需要精确比较“原模型 vs guided”时，应把 `alpha=0/mu=0` 作为同脚本 baseline，而不是直接和原 `ODEFlowSolver` 输出混比。

## 10. 如果要改进，代码层面优先方向

下面按“仍不重训”和“允许训练改动”分开。

### 10.1 仍不重训，仅改采样/后处理

1. 把 analysis target 从全局 proxy misfit 改成更局部的目标函数  
   例如使用 observation mask、target-sensitive anomaly、局部 gravity-gradient/magnetic term，而不是只用全场 normalized RMS。

2. 在制导中显式监控 hard-decode 变化  
   每步或每若干步记录 `model.decode(x)` 的体素变化率、目标标签体积、目标标签重心、目标区域 recall proxy。当前 trace 主要记录 continuous loss 和 velocity norm，不能判断离散结构是否在变。

3. 做“投影式”或“重采样式”后处理  
   比直接连续梯度更稳的办法可能是：生成较大 ensemble，然后按 geophysical + target metrics 排序/筛选，或者在候选离散标签空间做局部替换搜索。现有 `rank_realizations_by_geophysics()` 更接近这个方向。

4. 使用 paired baseline 严格评估  
   对每个 seed 比较 `alpha=0` 与 `alpha>0` 的 decoded sample，避免把不同采样器、不同 seed、不同 case 混在一起解释。

5. 调整 schedule 时同时看 decoded metrics  
   单看 `geo_loss` 会误导。需要同时输出 hard decoded `geo_misfit`、voxel accuracy、target IoU/recall、changed voxel fraction。

### 10.2 如果允许改训练或模型条件

1. 把地球物理观测作为条件输入  
   例如增加 geophysical encoder，把 observed gravity/magnetic/gradient field 编码后注入 U-Net，而不是只在采样外部加梯度。

2. 训练时加入地球物理一致性或 posterior matching  
   让模型学习地球物理观测对应的 geology posterior，而不是推理时临时外加未训练梯度。

3. 使用可微但更贴近离散标签的 relaxation  
   当前 softmax density expectation 太容易出现“概率调密度但不改 argmax”的情况。可以考虑 Gumbel-softmax、straight-through estimator 或离散候选重评分，但这会改变训练/采样设计。

4. 加强正演模型和 petrophysical prior  
   如果 density/susceptibility 配置不够区分目标标签，或 proxy 对目标弱可见，任何 gradient guidance 都很难恢复目标结构。

## 11. 最简诊断清单

后续每次尝试 guidance 参数或代码改动，建议至少检查以下几项：

1. `guided_trace.csv` 中 `effective_guidance_ratio` 是否达到预期。
2. `decoded_change_ratio.csv` 中 hard-decode 体素变化是否足够大。
3. guided 相对 paired baseline 的 `geo_misfit` 是否下降。
4. guided 相对 paired baseline 的 `voxel_accuracy`、`mean_iou` 是否改善。
5. target label 的 IoU、recall、volume、centroid 是否改善。
6. borehole consistency 是否被破坏。
7. 改善是否只发生在 proxy，而没有发生在地质指标。

当前已有结果最重要的信息是：proxy misfit 下降是真实存在的，但 hard decoded geology 的变化太小，且目标结构指标并不稳定改善。这就是推理阶段外加地球物理制导达不到预期的根本原因。
