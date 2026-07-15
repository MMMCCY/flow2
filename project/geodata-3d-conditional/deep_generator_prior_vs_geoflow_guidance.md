# Deep Generator Priors Paper vs Current Geoflow Guidance

本文档分析论文 `Deep generator priors for Bayesian seismic inversion.pdf` 为什么能够在地球物理反演中发挥作用，以及它和当前 `flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional` 中推理阶段地球物理制导的本质差异。

核心结论：论文的方法不是简单地“在生成过程中加一个地球物理梯度”。它把生成器作为显式 Bayesian prior，把反演变量改成低维 latent vector，并在该 latent 空间中采样完整 posterior `p(x | d_obs)`。我们的当前实现则是在已经训练好的条件生成流采样过程中，对高维连续 embedding state 临时叠加一个轻量地球物理 proxy 的梯度。两者的数学对象、变量空间、物理 forward、prior-data 平衡方式、输出变量类型都不同。

## 1. 论文到底做了什么

论文的基本目标是 Bayesian seismic inversion。未知量是地下连续物性模型，例如速度或平方慢度 `m`。观测量是地震走时或波形数据 `d_obs`。传统 Bayesian 反演写成：

```text
p(m | d_obs) ∝ p(d_obs | m) p(m)
```

若观测噪声是 Gaussian：

```text
d_obs = F(m) + ε
p(d_obs | m) ∝ exp(-1/2 ||F(m) - d_obs||^2_{Σ^-1})
```

其中 `F(m)` 是真实地球物理 forward map，例如：

- Traveltime tomography: 解 Eikonal equation。
- Full waveform inversion: 解 acoustic wave equation。

论文的关键变化是：不用手工 prior `p(m)`，而是训练一个 GAN generator：

```text
m = G(x; Θ_G)
x ~ N(0, I)
```

这样反演变量从高维模型 `m ∈ R^ng` 变成低维 latent `x ∈ R^nl`，并且 `nl << ng`。论文示例中 generator 将 50 维 Gaussian latent 映射到 `64 x 64` 速度图像。

于是 posterior 变成：

```text
p(x | d_obs) ∝ p(d_obs | x) p(x)
             ∝ exp(
                 -1/2 ||F(G(x; Θ_G)) - d_obs||^2_{Σ^-1}
                 -1/(2σ^2) ||x||^2
               )
```

这里 `σ` 是控制对 generator prior 信任程度的参数。如果 generator 太限制，目标模型需要很大的 `||x||` 才能表示，就应该增大 `σ`，削弱 prior 项，避免过度相信生成器。

## 2. 论文为什么能做到“地球物理制导”

### 2.1 它的地球物理项是 posterior 的 likelihood，不是启发式附加项

论文中地球物理数据进入 posterior 的 likelihood：

```text
||F(G(x)) - d_obs||^2_{Σ^-1}
```

这意味着每一个候选 latent `x` 都对应一个完整的物理模型 `G(x)`，再通过地球物理 forward `F` 与观测数据比较。地球物理数据不是在生成后打分，也不是临时改一步速度，而是 Bayesian objective 的一部分。

我们的当前实现中，地球物理项是推理阶段外加到 velocity 的 correction：

```python
v_prior = model.net(x, conditioning, t)
loss_geo = multi_physics_guidance_loss(...)
grad_geo = torch.autograd.grad(loss_geo, x)[0]
guidance_velocity = build_guidance_velocity(...)
v = v_prior - guidance_velocity
x = x.detach() + dt * v.detach()
```

代码位置：

- `guided_geophysical_sampling.py:340-370`
- `guided_geophysical_sampling.py:400`

这不是 posterior sampling，也没有定义 `p(x | d_obs)`。`mu`、`alpha`、`guidance_start`、`grad_clip_norm` 都是启发式控制量，不对应数据噪声 covariance 或 prior confidence。

### 2.2 它始终在 generator manifold 上搜索

论文中任意一个 latent `x` 生成的模型都是：

```text
m = G(x)
```

因此 pCN/MCMC 每次接受的样本都在 generator 的输出流形上。地球物理数据只是在这个流形上筛选或推动 posterior。这样有一个重要后果：即使地球物理反问题非唯一，样本仍受生成器 prior 约束，倾向于保持训练数据的地质/速度图像统计特征。

我们的实现里，采样状态 `x` 是 flow ODE 中间的高维 embedding state，形状是：

```text
[B, E, X, Y, Z]
```

这个 `x` 不是一个低维 latent，也不是一个被简单 Gaussian prior 约束的反演变量。外加 `-guidance_velocity` 后，状态是否仍在模型训练时学到的生成轨迹附近没有保证。

代码对照：

- 原训练流目标：`model_train_sh_inference_cond.py:430`
- 当前制导 Euler 更新：`guided_geophysical_sampling.py:400`

### 2.3 它反演的是低维 latent，不是全体 voxel embedding

论文把 `64 x 64` 图像反演变成 50 维 latent 反演。低维 latent 的优点是：

1. 搜索空间小很多。
2. 每个 latent 方向都对应 generator 学到的结构变化。
3. latent prior 是标准 Gaussian，方便 pCN 采样。
4. latent norm `||x||` 可用于判断生成样本是否仍在合理 prior 区域。

我们的制导变量是整块 3D embedding volume。以当前条件模型配置，embedding dim 为 15，空间为 `64^3`，单样本连续状态维度约为：

```text
15 x 64 x 64 x 64 = 3,932,160
```

地球物理 proxy 给出的观测场是 2D surface field，例如 `[1, 1, 64, 64]`。用一个 2D gravity proxy 去制导近 400 万维的高维连续状态，本身就是极强欠定问题。论文则用 generator prior 把反问题先压到 50 维。

### 2.4 它使用物理 forward 的观测数据，而不是轻量代理筛选量

论文的两个 forward operator 是地震反演常用物理模型：

- Traveltime tomography: Eikonal equation。
- FWI: acoustic wave equation。

其 likelihood 明确包含观测噪声协方差 `Σ`，并且结果评估会比较 predicted data 与 observed data 是否达到 noise level。

我们的 `SimpleGravityForward` 是轻量 proxy：

- 对每个深度切片做 2D 卷积。
- 沿深度累加为 surface anomaly。
- 默认去均值。

代码位置：

- `geophysics.py:139-230`
- `geophysics.py:424-466`

这个 proxy 可用于后处理排序和演示，但不是完整重力反演 forward，也没有观测噪声模型和 covariance 标定。更重要的是，从 3D 离散岩性到 2D surface gravity 是高度非唯一的：许多不同 3D 岩性组合都可以给出相似 surface anomaly。

### 2.5 它的 prior-data 平衡有 Bayesian 参数 σ，我们只有 heuristic scale

论文显式写出：

```text
-1/2 ||F(G(x)) - d_obs||^2_{Σ^-1} - 1/(2σ^2) ||x||^2
```

其中：

- `Σ` 表示观测噪声和数据不确定性。
- `σ` 表示对 generator prior generality 的信任程度。

论文还专门讨论了 `σ` 的作用：当 target model 与训练集分布差异较大时，`σ=1` 会导致 prior over-trusted；增大 `σ` 可以削弱 prior，扩大 posterior STD 和 error bars。

我们的代码没有 posterior density，因此没有等价的 `Σ` 或 `σ`。当前参数是：

- `mu`: absolute guidance strength。
- `alpha`: relative guidance strength。
- `guidance_start` / `guidance_schedule`: 时间调度。
- `grad_clip_norm`: 梯度裁剪。

这些参数能调节扰动强弱，但不能解释为“观测噪声水平”或“prior 可信度”。因此调参时只能看经验结果，例如 `geo_misfit`、`decoded_change_ratio`、`voxel_accuracy`、`target IoU`。

### 2.6 它用 MCMC 接受/拒绝保证 posterior 逻辑

论文使用 pCN 采样：

```text
y = sqrt(1 - β^2) x_old + β r,  r ~ N(0, I)
accept with probability based on likelihood ratio
```

pCN 的 proposal 与 Gaussian prior 匹配，所以接受率只依赖 likelihood ratio。这样采样得到的是 posterior 样本，可以计算：

- MAP estimate。
- posterior STD。
- pointwise probability density。
- confidence interval。

我们的实现是确定性或近确定性的 guided Euler 采样。它没有接受/拒绝步骤，也没有 posterior samples 的统计语义。当前保存的是 guided generated realizations，不是严格 posterior chain。

## 3. 论文和我们的工作逐项对比

| 维度 | 论文方法 | 我们当前实现 |
| --- | --- | --- |
| 生成模型角色 | 显式 prior generator `m=G(x)` | 条件 flow 生成器，推理时输出 embedding trajectory |
| 反演变量 | 低维 latent `x ∈ R^50` | 高维连续 embedding state `[E,X,Y,Z]` |
| prior | `x ~ N(0,I)`，有明确 density | learned flow prior 隐含在 velocity field 中，无显式 posterior density |
| 地球物理数据 | likelihood `p(d_obs|x)` 的核心项 | sampling velocity 的外加 correction |
| forward model | Eikonal / acoustic wave equation | lightweight gravity/magnetic proxy |
| 输出物性 | 连续 velocity/slowness model | 离散 categorical lithology，经 hard decode 得到 |
| 优化/采样 | pCN MCMC 采样 posterior | fixed-step Euler + heuristic guidance |
| prior-data 平衡 | `Σ` 和 `σ` | `mu`、`alpha`、schedule、clip norm |
| 是否保持生成器流形 | 是，所有模型都是 `G(x)` | 不保证，外加梯度可能偏离训练轨迹 |
| 不确定性 | 可计算 posterior STD / confidence interval | 当前没有 posterior 不确定性语义 |
| 成功条件 | target 可被 generator latent 表示，forward/likelihood 足够约束 | guidance 必须能穿过 hard decode 边界并形成合理离散结构 |

## 4. 为什么论文结果更容易看起来“成功”

### 4.1 输出是连续速度，不存在 hard argmax 断层

论文 generator 直接输出连续速度图像。地球物理 misfit 对 `G(x)` 的变化可以直接反映到最终模型。

我们的最终样本是：

```python
decoded = model.decode(final_state) - 1
```

代码位置：

- `model_train_sh_inference_cond.py:358-388`
- `guided_geophysical_sampling.py:649`

制导损失使用 soft probability 和 expected density：

```text
x -> softmax probability -> expected density -> forward proxy
```

但最终输出使用 `argmax`。只要连续 embedding 没跨过类别边界，最终岩性不变。已有结果中 `decoded_change_ratio` 常在千分之一量级，这正是这个断层的表现。

### 4.2 论文的 generator prior 已经用于反演目标

论文训练 GAN 的目的就是让 `G(x)` 表示地质/速度图像 prior，然后 Bayesian inversion 在 `x` 上进行。

我们的模型训练目标不是地球物理反演，而是 stochastic interpolation：

```python
XT, VT = self.interpolator.flow_objective(T, X0, X1)
VT_hat = self.net(XT, ATb, T)
```

代码位置：

- `model_train_sh_inference_cond.py:430-431`

这个训练目标使模型学会从噪声到地质 embedding 的条件生成速度，但没有学会“给定 gravity/magnetic data 后如何修改未观测区域”。

### 4.3 论文先验维度低且有可校验的 generality

论文专门评估 generator quality：

1. 生成 50,000 samples，比较 pointwise STD 和 probability density。
2. 对 test model 求最优 latent，比较 relative model error。
3. 检查 latent norm 是否落在 Gaussian zone。

这很关键：如果 test model 不能被 generator 合理表示，论文预期反演会失败或 prior 过强。

我们的当前工作没有等价测试。我们没有回答：

- 真实 3D lithology 是否在 flow generator 的高概率区域？
- 目标标签结构是否能通过当前 conditional prior 自然生成？
- 需要多大的连续扰动才能让 hard decode 产生目标结构？
- 这个扰动是否仍是 geologically plausible？

### 4.4 论文承认并处理“prior 过信任”的失败模式

论文在 Sigsbee 测试中出现了一个很重要的现象：MAP estimate 能把 predicted data 拟合到 noise level，但模型仍明显不同于 true model。这说明生成器 prior 过强，反演只能找到 DGAN range 内的数据拟合解。

论文的处理方式是增大 `σ`，削弱 prior 影响，使 posterior STD 和 error bars 增大。

这和我们的失败很相似，但我们的代码没有 `σ` 这样的 Bayesian knob。我们看到的是：

- proxy misfit 改善；
- decoded geology 指标不稳定或几乎不变；
- target-label metric 有时不改善。

这可以理解为：我们也在一个 implicit generator prior 允许的范围内找 proxy data-fit 解，但没有显式 posterior 框架来判断 prior 是否过强，也没有不确定性估计来提醒“这个解只是 prior range 内的拟合”。

## 5. 我们不能直接照搬论文的原因

### 5.1 我们的生成模型不是 `m=G(z)` 型显式 generator

论文需要一个函数：

```text
z -> physical model m
```

并且 `z` 有简单 prior density。我们的 flow 模型更像：

```text
X0, ATb, ODE dynamics -> final embedding -> hard decode lithology
```

`X0` 虽然来自 Gaussian noise，但中间 ODE state 和最终 hard decoded lithology 没有像 GAN latent 那样简单的 posterior density。要照搬论文，需要重新定义反演变量和 posterior，例如把 `X0` 作为 latent，优化或采样：

```text
p(X0 | d_obs, ATb)
```

但这会是百万维 latent，不是论文那种 50 维 latent，MCMC 难度完全不同。

### 5.2 我们是离散岩性，论文是连续速度

论文的 velocity model 是连续变量。我们的 lithology 是离散类别。即使 soft relaxation 可微，最终 hard decode 仍然导致：

- 梯度可改变概率，不一定改变类别。
- 小幅连续扰动可能完全没有离散效果。
- 大幅扰动可能造成零散类别翻转，而不是空间连贯地质体。

因此我们需要额外机制连接 geophysical loss 和 hard categorical structure，例如 straight-through estimator、Gumbel-softmax、离散候选搜索、或生成大 ensemble 后按物理数据筛选。

### 5.3 我们的 forward proxy 信息量不足

论文的 seismic data 包含大量 source-receiver/time 维度信息。我们的 lightweight gravity proxy 是 `64 x 64` surface anomaly，而且从 3D 到 2D 压缩。对于 3D categorical lithology，它提供的约束远弱于 FWI/traveltime tomography 对 2D velocity model 的约束。

这不是说 gravity/magnetic 不能用，而是当前实现的 proxy 不能承担“强反演制导”的角色。

### 5.4 我们没有 posterior acceptance/rejection 或 ensemble posterior

论文不是只生成一个 guided sample。它跑长链 MCMC，丢掉 burn-in，用剩余样本计算统计量。我们的脚本生成 `sample_*.pt`，然后评价 misfit/accuracy，没有 posterior chain 语义。

如果要更接近论文，至少应考虑：

- 在 latent/initial-noise 空间定义 posterior score。
- 用 geophysical likelihood 对样本做 MCMC 接受/拒绝或 SMC/importance weighting。
- 输出 posterior ensemble，而不是单纯 guided trajectory。

## 6. 论文给我们的直接启示

### 启示 1：不要把地球物理制导只当作 ODE velocity correction

更稳的框架是先定义：

```text
p(latent | observed geophysics, boreholes)
```

然后采样或优化这个 posterior。当前的 `v_prior - guidance_velocity` 没有 posterior 语义，因此很难解释和控制。

### 启示 2：要选择合适的 latent 空间

论文成功的核心之一是低维 latent。我们的 `X0` 或 ODE state 太高维。可能的替代方向：

1. 训练一个低维 geology generator。
2. 在已生成 ensemble 的低维系数或 PCA/autoencoder latent 上做反演。
3. 使用 diffusion/flow 模型自身的 likelihood 或 score，如果可得，再构造 posterior sampling。

### 启示 3：必须校验 generator 是否能表示目标结构

论文先评估 generator generality，再做反演。我们也需要类似诊断：

- 从模型生成大量 baseline samples。
- 统计目标标签体积、连通性、位置分布。
- 判断 truth target 是否落在 prior ensemble 的支持范围内。
- 如果 truth 不在支持范围内，地球物理制导很难恢复它。

### 启示 4：prior-data 权重需要可解释标定

论文用 `Σ` 和 `σ`。我们目前用 `mu/alpha`。后续可以尝试把 guidance 参数改写成更接近 Bayesian 的形式：

```text
data term: ||F(sample) - d_obs||^2 / noise_variance
prior term: distance_to_prior_or_latent_norm / prior_scale
```

哪怕不是严格 Bayesian，也应让参数对应观测噪声、forward uncertainty、prior confidence，而不是纯经验缩放。

### 启示 5：proxy misfit 下降不等于真实结构恢复

论文在 Sigsbee case 中展示了“数据拟合到 noise level，但模型仍错”的情况。我们的当前结果也类似：gravity proxy misfit 可以下降，但 target-label metrics 不一定改善。后续报告中应明确区分：

- data-fit improvement；
- posterior uncertainty；
- geology/target reconstruction；
- prior range limitation。

## 7. 如果要把我们的工作改成论文式路线

### 路线 A：后处理式 Bayesian ensemble reweighting

最小改动路线：

1. 用当前条件模型生成大量 baseline samples。
2. 对每个 sample 计算 `F(sample)` 和 geophysical likelihood。
3. 用 likelihood 对 ensemble 加权或筛选。
4. 输出 weighted MAP、weighted uncertainty、top-k samples。

优点：不改训练，不需要可微 guidance。  
缺点：只在已有 ensemble 支持范围内选择，不能创造 prior 没生成过的结构。

### 路线 B：在初始噪声 `X0` 上做优化/采样

定义：

```text
sample = Decode(Flow(X0, ATb))
loss = data_misfit(F(sample), d_obs) + prior_penalty(X0)
```

但这里有两个问题：

1. `X0` 维度太高。
2. `Decode` 是 hard argmax，不可微；如果用 soft decode，又回到概率-离散断层。

需要配合 soft/straight-through relaxation 或低维参数化。

### 路线 C：训练低维 generator prior

更接近论文：

1. 训练 `G(z, ATb)` 或 `G(z)`，其中 `z` 低维。
2. `G` 直接输出 lithology probability/density/physical property。
3. 定义 `p(z | d_obs, ATb)`。
4. 用 MCMC/VI/optimization 求 posterior。

优点：理论更清楚。  
缺点：需要重新训练或至少新增模型，已经超出当前“只做推理阶段制导”的约束。

### 路线 D：联合地球物理条件训练

将 observed gravity/magnetic/seismic data 编码成条件，训练模型直接学习：

```text
p(geology | boreholes, geophysics)
```

这和论文不同，但更适合当前 flow/diffusion 类生成模型。缺点同样是需要改训练。

## 8. 最关键的一句话

论文能做地球物理制导，是因为它把“生成器 prior + 地球物理 forward + 噪声模型”统一在一个 Bayesian posterior 里，并且在低维 latent 空间中采样这个 posterior。我们的当前方法只是把一个轻量 geophysical proxy 的梯度外加到高维 embedding 采样轨迹上；它没有低维 latent posterior、没有 prior-data 权重标定、没有 acceptance/rejection、没有硬离散输出的一致可微反演路径。因此它可以降低 proxy misfit，但很难稳定地产生预期的地质结构恢复。
