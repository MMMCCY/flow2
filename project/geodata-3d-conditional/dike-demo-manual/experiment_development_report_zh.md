# Dike Guidance Demo 试验开发复盘报告

## 1. 当前研究目标

本阶段目标是：在不修改训练代码、不修改 U-Net、不修改 embedding、不修改 Flow Matching training loss、不重新训练模型的前提下，将已训练好的 conditional Flow Matching 网络作为 learned geological prior，仅在推理 / 采样阶段加入 geophysical guidance，用于展示 sparse conditioning 下 dike-like / intrusion-like target label 的三维恢复差异。

论文叙事目标原本是：

- baseline conditional Flow Matching 在 sparse borehole conditioning 下，对 dike extent 存在较大不确定性。
- inference-time lightweight geophysical proxy guidance 能降低 geophysical proxy misfit。
- guided ensemble 的 target-label probability volume 更集中、更接近 truth。
- 所有表述应保持为 lightweight gravity-proxy / magnetic-proxy / gravity-gradient-proxy，而不是定量反演。

当前复盘结论是：**gravity-only 版本有弱正向证据，但 magnetic / gravity-gradient 扩展没有达到预期；继续盲目加 alpha 或继续叠加 proxy 很可能是在错误方向上推进。**

## 2. 代码开发时间线与目的

### 2.1 P0/P1/P2 Demo 工具链

新增和整理的工具模块包括：

- `geology_io_utils.py`
  - 目的：统一 `.pt` 体数据加载、shape 归一化、sample 文件索引、target mask、probability volume、CSV/JSON 工具。
  - 解决的问题：不同输出张量可能是 `[X,Y,Z]`、`[1,X,Y,Z]`、`[1,1,X,Y,Z]` 或 batch 形式，后处理脚本需要一致处理。

- `visualize_truth_model_labels.py`
  - 目的：人工 QA truth_model 中每个 label 的空间结构和 borehole 命中情况。
  - 后续路线调整：从“自动猜测 dike label”转为“人工可视化确认 dike-like label”。

- `create_density_config.py`
  - 目的：为人工确认的 target label 设置受控 density_config，使 target 与非 target lithology 具有显著 density contrast。

- `analyze_dike_observability.py`
  - 目的：删除 / 替换 target label，计算 truth gravity-proxy 是否显著变化，从而验证 target 对 lightweight gravity-proxy 是否可观测。

- `evaluate_target_feature.py`
  - 目的：计算 target-specific 指标：IoU、precision、recall、F1、volume error、centroid distance、probability overlap、probability entropy 等。

- `visualize_dike_ensemble.py`
  - 目的：生成 Figure 8 / Figure 9 风格的 baseline vs guided target realization 和 probability visualization。

- `compare_gravity_residuals.py`
  - 目的：生成 baseline/guided gravity-proxy residual side-by-side 图。

- `select_dike_demo_samples.py`
  - 目的：根据 global metrics + target metrics 自动选展示样本。

- `make_dike_guidance_demo.py`
  - 目的：串联 truth QA、observability、metrics、sample selection、residual、3D visualization、demo_report。

这部分工具链总体是有效的。它解决了“能不能系统筛选 case、评估 target、生成论文图”的问题，没有触碰训练流程。

### 2.2 Gravity-only guidance 路线

已有 `guided_geophysical_sampling.py` 的核心逻辑是：

- 对连续 embedding state `x` 做 soft decode，得到 category probabilities。
- 用 `LithologyPropertyMap` 将 soft probabilities 映射成 expected density volume。
- 用 `SimpleGravityForward` 得到 lightweight gravity-proxy。
- 计算 `normalized_misfit(predicted_gravity, observed_gravity)`。
- 对 `x` 求梯度，并在 sampling velocity 上加入 inference-time guidance。

相对制导模式中，关键是：

```text
guidance_velocity = alpha * w_t * ||v_prior|| * grad_geo / ||grad_geo||
v = v_prior - guidance_velocity
```

这意味着 guidance 是沿 loss descent 方向修正 Flow Matching prior velocity。

### 2.3 Manual target-label 路线

最初自动猜测 target label 的可靠性不足，因此改成：

```text
人工可视化 truth_model
    -> 确认 dike-like label
    -> 创建 density_config
    -> 用 truth_model + density_config 生成 observed_gravity
    -> baseline alpha=0 / guided alpha>0 采样
    -> global evaluation
    -> target-level evaluation
    -> sample selection + figures
```

这一调整是合理的，因为论文 Demo 不能建立在自动猜测 label 的不确定性上。

### 2.4 Candidate screening

候选筛选结果保存在：

```text
project/geodata-3d-conditional/dike-demo-manual/candidate_screening/candidate_screening.csv
```

四个候选中，当时较值得展示的是：

- `cond_generation_1_label10`
- `paper_cond_gen_0_label7`

它们在 gravity-only 指标下被分类为 `main_demo_candidate`。

筛选表中的关键结果：

| case | target | recommendation | geo misfit improvement | target IoU improvement | target recall improvement |
|---|---:|---|---:|---:|---:|
| cond_generation_1_label7 | 7 | limitation_case | 0.2651 | 0.00034 | -0.00011 |
| cond_generation_1_label10 | 10 | main_demo_candidate | 0.0933 | 0.00383 | 0.00055 |
| paper_cond_gen_0_label7 | 7 | main_demo_candidate | 0.0358 | 0.00144 | 0.00150 |
| paper_cond_gen_0_label8 | 8 | limitation_case | 0.2113 | -0.00032 | -0.00094 |

解释：gravity-only 能稳定降低 geo misfit，但 target-level 改善非常小。`main_demo_candidate` 是“弱正向改善”，不是强重建成功。

### 2.5 Magnetic-proxy + gravity-gradient-proxy 扩展

后续新增：

- `geophysics.py`
  - `GravityGradientForward`
  - `MagneticTMIForward`

- `geology_io_utils.py`
  - `load_susceptibility_config`
  - `property_map_from_susceptibility_config`
  - `susceptibility_config_metadata`

- `create_susceptibility_config.py`
  - 为 target label 设置高 susceptibility，用于 controlled magnetic-proxy demo。

- `generate_observed_geophysics.py`
  - 从 truth_model 生成 observed gravity / magnetic / gravity-gradient proxy。

- `guided_geophysical_sampling.py`
  - 新增 `multi_physics_guidance_loss`
  - 支持 `--physics-mode gravity|magnetic|gravity_gradient|joint`
  - 支持 `--gravity-weight`、`--magnetic-weight`、`--gravity-gradient-weight`
  - `guided_trace.csv` 新增 `gravity_loss`、`magnetic_loss`、`gravity_gradient_loss`

- `evaluate_geophysics.py`
  - 新增可选输出列：
    - `magnetic_proxy_misfit`
    - `gravity_gradient_proxy_misfit`
    - `joint_proxy_misfit`

这部分代码能运行，语法和 synthetic tensor smoke test 通过，但真实采样结果没有达到预期。

## 3. 已完成试验与结果

### 3.1 初始 label 9 gravity-only demo

路径：

```text
dike-demo-manual/cond_generation_0_label9/final_demo
```

结果：

| 指标 | baseline | guided | 变化 |
|---|---:|---:|---:|
| geo_misfit mean | 0.8834 | 0.7065 | 改善 0.1769 |
| voxel_accuracy mean | 0.55129 | 0.55130 | 基本不变 |
| mean_iou mean | 0.14597 | 0.14598 | 基本不变 |
| target_iou mean | 0.01271 | 0.01278 | 极小改善 |
| target_recall mean | 0.01385 | 0.01395 | 极小改善 |
| residual_rms_reduction | - | 0.09295 | gravity residual 有改善 |

解释：

- gravity-proxy misfit 降低。
- target label 9 几乎没有恢复。
- 这说明 gravity-only guidance 主要改善 proxy residual，而不是显著改善 dike reconstruction。

### 3.2 cond_generation_1_label10 gravity-only demo

路径：

```text
dike-demo-manual/candidate_screening/cond_generation_1_label10/final_demo
```

结果：

| 指标 | baseline | guided | 变化 |
|---|---:|---:|---:|
| geo_misfit mean | 0.32395 | 0.23063 | 改善 0.09332 |
| voxel_accuracy mean | 0.66993 | 0.67002 | 基本不变 |
| mean_iou mean | 0.19953 | 0.20071 | 小幅改善 |
| target_iou mean | 0.42636 | 0.43020 | 小幅改善 |
| target_recall mean | 0.68161 | 0.68216 | 极小改善 |
| target_f1 mean | 0.59568 | 0.59942 | 小幅改善 |
| residual_rms_reduction | - | 0.11685 | gravity residual 改善 |

解释：

- 这是目前最接近论文 Demo 目标的 case。
- 但 target 改善仍然很弱：IoU 只提升约 0.0038。
- 它可以支持“guidance 能部分影响结果”的弱表述，但不足以支持强结论。

### 3.3 paper_cond_gen_0_label7 gravity-only demo

路径：

```text
dike-demo-manual/candidate_screening/paper_cond_gen_0_label7/final_demo
```

结果：

| 指标 | baseline | guided | 变化 |
|---|---:|---:|---:|
| geo_misfit mean | 0.21538 | 0.17954 | 改善 0.03584 |
| voxel_accuracy mean | 0.78452 | 0.78480 | 小幅改善 |
| mean_iou mean | 0.26729 | 0.26796 | 小幅改善 |
| target_iou mean | 0.40705 | 0.40848 | 小幅改善 |
| target_recall mean | 0.54949 | 0.55099 | 小幅改善 |
| target_f1 mean | 0.57629 | 0.57776 | 小幅改善 |
| residual_rms_reduction | - | 0.19620 | gravity residual 改善 |

解释：

- gravity-only 同样没有变差。
- 但 target-level 改善仍然非常小。
- 可作为弱正向 case，但不是强视觉 Demo。

### 3.4 cond_generation_1_label10 magnetic/joint alpha=1.0

路径：

```text
dike-demo-manual/magnetic_joint/cond_generation_1_label10
```

配置：

- baseline：`alpha=0.0`
- joint：`alpha=1.0`
- `physics_mode=joint`
- `gravity_weight=0.5`
- `magnetic_weight=1.0`
- `gravity_gradient_weight=0.25`
- seed 相同，paired baseline 存在。

结果：

| 指标 | baseline | joint alpha=1.0 | 变化 |
|---|---:|---:|---:|
| geo_misfit mean | 0.32395 | 0.25654 | gravity 改善 |
| magnetic_proxy_misfit mean | 1.35057 | 2.91402 | 明显变差 |
| gravity_gradient_proxy_misfit mean | 0.21142 | 0.33752 | 明显变差 |
| joint_proxy_misfit mean | 1.56540 | 3.12667 | 明显变差 |
| voxel_accuracy mean | 0.66993 | 0.66945 | 略降 |
| mean_iou mean | 0.19953 | 0.19931 | 略降 |
| borehole_consistency mean | 0.99906 | 0.99675 | 下降 |
| target_iou mean | 0.42636 | 0.42611 | 略降 |
| target_recall mean | 0.68161 | 0.68091 | 略降 |
| target_f1 mean | 0.59568 | 0.59543 | 略降 |

trace 结果：

- 最后一步 `effective_guidance_ratio ≈ 0.959`
- `decoded_change_fraction mean ≈ 0.00262`

解释：

- alpha=1.0 已经很强，guidance velocity 明显进入采样动力学。
- 但 hard decoded categorical volume 只改变约 0.26% 体素。
- magnetic 和 gravity-gradient 最终 proxy metric 反而变差。
- target reconstruction 没有改善。

### 3.5 Magnetic-only alpha=1.0 诊断

路径：

```text
dike-demo-manual/magnetic_joint/cond_generation_1_label10/guided_magnetic_alpha1_00
```

可比核心指标：

| 指标 | baseline | magnetic-only alpha=1.0 | 变化 |
|---|---:|---:|---:|
| magnetic_proxy_misfit mean | 1.35057 | 3.28946 | 明显变差 |
| target_iou mean | 0.42636 | 0.42328 | 变差 |
| target_f1 mean | 0.59568 | 0.59263 | 变差 |
| target_centroid_distance mean | 6.94354 | 7.10496 | 变差 |
| probability_entropy_mean | 0.06443 | 0.06587 | 更分散 |
| decoded_change_fraction mean | - | 0.00248 | 改变极少 |

注意：该评估中的 `geo_misfit` 不应与 baseline_joint_evaluation 直接比较，因为 magnetic-only evaluation 命令没有传入同一套 `observed_gravity` 和 `density_config`。可比的是 `magnetic_proxy_misfit`。

结论：

- magnetic-only 本身没有降低 magnetic proxy misfit。
- 因此 joint 失败不只是权重冲突。

### 3.6 Gravity-gradient-only alpha=1.0 诊断

路径：

```text
dike-demo-manual/magnetic_joint/cond_generation_1_label10/guided_gravity_gradient_alpha1_00
```

结果：

| 指标 | baseline | gravity-gradient-only alpha=1.0 | 变化 |
|---|---:|---:|---:|
| gravity_gradient_proxy_misfit mean | 0.21142 | 0.36193 | 明显变差 |
| target_iou mean | 0.42636 | 0.42146 | 变差 |
| target_f1 mean | 0.59568 | 0.59089 | 变差 |
| target_centroid_distance mean | 6.94354 | 7.23051 | 变差 |
| probability_entropy_mean | 0.06443 | 0.06683 | 更分散 |
| decoded_change_fraction mean | - | 0.00387 | 改变仍很少 |
| borehole_consistency mean | 0.99906 | 0.96900 | 明显下降 |

结论：

- gravity-gradient-only 同样没有降低对应 proxy misfit。
- 而且 borehole consistency 明显下降，说明强 guidance 已经在破坏 sparse conditioning consistency。

## 4. 为什么简单 gravity 方法能有弱效果，而 magnetic/gradient 反而变差

### 4.1 Gravity-only 优化和评估是完全同构的

gravity-only 中：

```text
soft probabilities -> density_config -> SimpleGravityForward -> gravity loss
```

评估时也是：

```text
hard decoded samples -> same density_config -> SimpleGravityForward -> geo_misfit
```

因此优化目标和评估目标高度一致。即使 hard decode 改变很少，只要少量体素变化在 gravity field 上有有利影响，`geo_misfit` 就可能下降。

### 4.2 Simple gravity-proxy 是平滑、低频、全局场

`SimpleGravityForward` 本身是强平滑的 surface field proxy。它对局部 categorical boundary 的要求不高，少量整体质量分布变化就可能降低 field residual。

这解释了为什么 gravity-only 能“至少不变差”：它优化的是一个低频、平滑、相对宽容的目标。

### 4.3 Magnetic / gravity-gradient proxy 更局部、更高频

新增的 `MagneticTMIForward` 和 `GravityGradientForward` 使用类似二阶导 / dipole-style kernel：

```text
(3 z^2 - r^2) / (r^2 + eps)^2.5
```

它比 simple gravity 更局部、更敏感、更高频。对 target geometry、边界和位置更敏感。

在 hard categorical decode 只改变 0.2%-0.4% 体素的情况下，这种高频 proxy 更容易被少量错误边界变化放大，最终 metric 反而变差。

### 4.4 Soft guidance 与 hard decode 之间存在明显落差

当前 guidance 对连续 embedding state `x` 求梯度，损失定义在 soft probabilities 上。

但最终评估使用：

```python
decoded = model.decode(final_state) - 1
```

这是 hard categorical decode。当前结果表明：

- guidance velocity 很强；
- continuous trajectory 被显著改变；
- 但 hard decoded label 只改变极少体素；
- target metrics 几乎不动或变差。

这说明我们目前主要卡在：

```text
soft proxy loss gradient 有效
    不等于
hard categorical realization 改善
```

### 4.5 Magnetic 物性设定可能过强，导致 loss landscape 不适合当前 prior manifold

`create_susceptibility_config.py` 中 target susceptibility 设为 `5.0`，背景 scale 为 `0.01`。

这会让 target label 在 magnetic-proxy 中占绝对主导。好处是 target observability 强，坏处是：

- loss 对 target probability 极其敏感；
- gradient 可能把 continuous embedding 推向模型 prior manifold 外；
- U-Net prior velocity 和 magnetic gradient 方向冲突；
- hard decode 无法产生连续、合理的 dike morphology。

### 4.6 Gravity-gradient guidance 对 borehole consistency 的破坏是警告信号

gravity-gradient-only alpha=1.0 后：

```text
borehole_consistency: 0.99906 -> 0.96900
```

这说明强 gradient proxy guidance 已经不只是“没有改善 target”，而是在破坏条件约束。

这类结果不适合作为 paper demo，也不应该继续靠加 alpha 推进。

### 4.7 evaluation 中存在一个需要警惕的可比性问题

`evaluate_geophysics.py` 即使在 magnetic-only evaluation 中仍会输出 `geo_misfit`。如果命令没有传入同一个 `observed_gravity` 和 `density_config`，这个 `geo_misfit` 与 baseline_joint_evaluation 的 `geo_misfit` 不是严格可比。

因此 magnetic-only 诊断中应重点看：

```text
magnetic_proxy_misfit
```

gravity-gradient-only 诊断中应重点看：

```text
gravity_gradient_proxy_misfit
```

不能把 magnetic-only 中看起来较低的 `geo_misfit` 当成真实 gravity improvement。

## 5. 当前偏差总结

### 5.1 已经达到的部分

- 后处理 / demo 工具链基本完成。
- 手动 target-label QA 路线正确。
- density_config 受控物性设定可复现。
- gravity-only guidance 能降低 lightweight gravity-proxy misfit。
- `cond_generation_1_label10` 和 `paper_cond_gen_0_label7` 有弱 target-level 正向改善。
- 所有实验保持了 inference-time sampling-only，没有修改训练和模型结构。

### 5.2 没有达到的部分

- gravity-only target improvement 太小，不足以形成强 Figure 8 / Figure 9 叙事。
- magnetic-proxy guidance 没有降低 magnetic_proxy_misfit。
- gravity-gradient-proxy guidance 没有降低 gravity_gradient_proxy_misfit。
- joint guidance 中 magnetic/gradient 项反向变差。
- hard decoded target reconstruction 没有变好。
- strong alpha 下 decoded 变化仍然极少，但 borehole consistency 可能下降。

### 5.3 当前最可能的根本问题

优先级从高到低：

1. **soft guidance 与 hard categorical decode 之间存在不可忽略的离散边界落差。**
2. **magnetic/gradient proxy 比 simple gravity 更高频，和当前 learned prior 的可达形态不匹配。**
3. **target susceptibility contrast 过强，导致 gradient 方向过于尖锐，推离 geological prior manifold。**
4. **multi-physics loss 的尺度和时序没有校准，尤其在 ODE sampling 后期强推可能破坏已有结构。**
5. **当前 evaluation / trace 没有同时保存 pre-step / post-step soft proxy loss，因此无法判断每一步 guidance 是否真的在局部下降 soft loss。**

## 6. 是否陷入错误路线

是，至少 magnetic / gravity-gradient 这条路线目前已经进入高风险区。

不是说 geophysical guidance 方向完全错误，而是当前实现组合：

```text
soft_decode_to_probs
    + high-contrast target susceptibility
    + high-frequency proxy kernel
    + relative alpha=1.0
    + hard categorical decode
```

没有产生目标论文图需要的效果。

继续盲目做以下事情不建议：

- 继续提高 alpha。
- 继续增加 magnetic / gradient 权重。
- 继续更换 case 试运气。
- 继续把 magnetic/joint 结果包装成成功 demo。

## 7. 建议的暂停后复盘方向

### 7.1 先冻结可用成果

建议将当前可用结论冻结为：

- gravity-only lightweight proxy guidance 可以降低 proxy misfit。
- target-level 改善存在但很弱。
- 该方法目前更适合表述为 post-hoc / inference-time proxy consistency improvement，而不是强 dike reconstruction inversion。

### 7.2 必须补的诊断，而不是继续调参

如果后续继续研究，应先补诊断代码，而不是直接跑新实验：

1. 在每个 step 记录：
   - `loss_before_step`
   - `loss_after_step`
   - `soft_predicted_proxy_before`
   - `soft_predicted_proxy_after`

2. 保存 final continuous state 的：
   - soft target probability volume；
   - hard decoded target mask；
   - 二者的差异图。

3. 检查 guidance sign：
   - 对同一个 `x` 做一次小步 `x - eps * grad`；
   - 验证 soft proxy loss 是否下降；
   - 再 hard decode，看 hard proxy loss 是否同步下降。

4. 做 temperature sweep：
   - `tau=0.05, 0.1, 0.2, 0.5`
   - 判断 soft decode 是否过软或过硬。

5. 做物性 contrast sweep：
   - susceptibility target 从 `0.5, 1.0, 2.0, 5.0`
   - 不要一开始就用极端 contrast。

### 7.3 如果论文目标必须是 Figure 8 / Figure 9 风格

更现实的路线可能是：

- 先只使用 gravity-only。
- 强调 lightweight proxy consistency，不声称定量 inversion。
- 选择 `cond_generation_1_label10` 或 `paper_cond_gen_0_label7` 作为 weak positive case。
- 图中展示：
  - baseline uncertainty 大；
  - guided gravity residual 下降；
  - target probability 有轻微集中；
  - 明确承认 target reconstruction improvement is modest。

如果必须展示明显 dike reconstruction 改善，则当前方法还不够。

## 8. 一句话总括

之前简单 gravity 方法看起来有效，是因为它优化和评估的是同一个平滑、低频、宽容的 lightweight gravity-proxy；而 magnetic / gravity-gradient 扩展引入了更局部、更高频、更强 contrast 的 proxy，暴露了 soft guidance 无法有效跨越 hard categorical decode 边界的问题，因此即使 alpha=1.0、guidance velocity 很强，最终 hard decoded realization 的 magnetic/gradient proxy 和 target metrics 仍然变差。

