# Phase 6Q：五目标体推理因果审计

冻结日期：2026-08-04，在首次 Phase 6Q 结果生成之前。

## 目的与阶段边界

Phase 6Q 是 Phase 6B 正式训练前新增的推理期机制实验。它回答：当前
acquisition-domain 地球物理失败首先来自物理不可辨识、soft/hard 松弛、优化
参数化、冻结 embedding，还是 flow 轨迹控制器。它不训练网络、不修改 checkpoint，
也不以解析立方体代表真实地质分布。

实验严格按下列边界逐级增加复杂性：

1. `Q0`：硬离散候选枚举，完全不使用梯度和生成网络；
2. `Q1`：候选体 soft 系数优化与 hard-forward/soft-backward 对照；
3. `Q2`：自由体素 logits、hard 坐标求解器及其步长诊断；
4. `Q3`：真实 checkpoint 的冻结 embedding endpoint，不加载 flow U-Net；
5. `Q4`：冻结 flow 轨迹；
6. `Q5`：StructuralGeo 原生五侵入体；
7. 只有前述原因链明确后才返回 `cond_generation_0`。

本规范首先冻结并授权实现/运行 `Q0-Q1`。Q0/Q1 完成后、Q2 运行前，于
2026-08-04 冻结 Q2 的独立配置；`Q2-Q5` 必须复用已冻结真值和观测，不得依据
Q0-Q1 的真值地质指标调整物性、位置或验收门。

## Q0/Q1 解析真值

- 网格：`64 x 64 x 64`，最后一维 z 向上；
- 空气：`z=56:64`，raw label `-1`；
- 地下背景：raw label `0`；
- 目标体：五个互不相交的 `8 x 8 x 10` raw label-9 立方体；
- 三个已钻目标体位于独立横向 footprint，三口单体素垂直井通过其中心；
- 两个未钻目标体从十二个预声明候选 footprint 中选择；
- 初始模型 `I0` 与真值完全相同，只删除两个未钻目标体；
- 条件包括完整空气、平面地表和三口井；候选体不得接触条件体素。

确切坐标、truth candidate indices 和哈希由
`experiments/stage6_inference_causality/configs/five_body_cuboid_v1.json`
冻结。候选体个数和“恰有两个未钻体”是 Q0 的强诊断先验，不是部署假设。

## 观测阶梯

同一真值和 I0 依次使用：

1. `property`：完整 label-9 占据体；
2. `blurred_property`：固定 sigma 的三维高斯模糊占据体；
3. `reflectivity_spikes`：时间采样后的未卷积反射系数；
4. `seismic`：Phase 4C 同定义的全覆盖、无噪声卷积地震；
5. `gravity`：Phase 4A 同定义的全覆盖、无噪声地表重力。

声学和重力物性继续使用 Phase 4 的 distinct label-9 上限码表。解析真值、码表和
正演来自同一算子，属于明确标注的 inverse crime。

## Q0：硬候选枚举

枚举十二个候选体中所有二体组合，使用 hard labels、hard properties 和 hard
forward 计算观测误差。必须保存全部组合，而不是只保存最佳组合。

主要输出：

- truth pair 的排名；
- truth pair 与第二名的 RMSE 间隔；
- 基线 RMSE；
- 低于数值阈值和观测不确定度阈值的候选数；
- 各算子的 hard-loss 排名与 body precision/recall。

若 truth pair 在 noiseless inverse-crime 下不是硬 loss 第一名，后续不得用该算子
讨论 soft/hard 或 flow；应先审计正演、观测和候选设计。若多组硬模型落入同一
误差容限，直接记录为该受限模型族内的物理非唯一性。

## Q1：候选系数优化

十二个候选各有一个 logit。固定三个已钻体和背景，只优化未钻候选贡献。

- `soft`：前向使用 sigmoid soft occupancy；hard 审计使用 0.5 阈值；
- `ste_top2`：前向使用恰好两个 hard 候选，反向使用 sigmoid straight-through；
- 两者使用同一零候选初始倾向、Adam、温度计划和更新数；
- 模型选择只按对应观测的 hard physics loss，不能用 truth body 指标；
- 每步同时记录 soft loss、hard loss、选择集合、soft/hard 达到率和类别熵。

地震 additionally 运行 `correct`、`zero`、`shuffled_xy` 三种观测控制。shuffled
使用冻结置换种子。正确观测应比 zero/shuffled 更好地选择两个真值未钻体。

## 预注册配置

- 优化更新：200；
- Adam learning rate：0.15；
- weight decay：0；
- 温度：`1.0 x 80, 0.5 x 60, 0.2 x 60`；
- soft 方法 cardinality penalty：0.01，目标和为 2；
- hard evaluation：每一步；
- primary device：CUDA；
- 所有输出写入新的、拒绝覆盖的 run tag；
- runner 不能加载 flow checkpoint 或计算任何训练梯度。

## 验收门

### 工程门

- 配置、代码、码表和生成 tensors 具有内容 hash；
- 同一配置重复构建 tensors 的 hash 一致；
- 真值/基线只在两个未钻体处不同；
- 三口井分别命中三个固定体且不命中候选体；
- 真值、基线和所有候选条件违反均为零；
- 全部损失、状态和梯度有限；
- 输出目录拒绝覆盖；
- 运行命令、主机、GPU、开始/结束时间和结果路径写入工作日志。

### Q0 物理门

- `property` truth pair 必须唯一排名第一并达到数值零误差；
- `blurred_property`、`reflectivity_spikes`、`seismic` 分别报告是否唯一，不预设
  非唯一结论；
- `gravity` 只作非唯一性对照，不要求唯一恢复。

### Q1 soft/hard 门

在正确完整地震上：

- hard physics attainment 至少 0.95；
- 选择的两个未钻候选均正确，body precision/recall 均为 1；
- soft 与 hard attainment 相差不超过 0.05；
- correct observation 的 body recall 必须高于 zero 和 shuffled；
- 不能以 soft loss 单独通过。

这些门首先用于解析候选参数化。通过不代表冻结 flow 或自由体素已经解决；失败
则明确说明 acquisition-domain soft/hard 问题在不使用网络时已经出现。

## 后续决策

- Q0 失败：先修正观测/正演或承认该算子不可辨识；
- Q0 通过、soft 失败、STE 通过：主要是 soft relaxation；
- Q1 两者通过、后续 endpoint 失败：主要是 embedding/decoder bridge；
- endpoint 通过、flow 失败：主要是 prior velocity/controller 冲突；
- 解析模型通过、StructuralGeo 原生模型失败：主要是生成支持或训练分布问题；
- hard physics 高拟合后 geology 仍多解：此时才把剩余差异归为非唯一性。

Phase 6Q 不授权正式训练。Phase 6B 的 manifest、held-out split 和训练门保持不变。

## Q2 补充冻结：自由体素二分类

Q2 不再提供十二体候选字典。它只声明一个不依赖隐藏体位置的宽搜索区：
`x=0:64, y=18:64, z=0:55`。搜索区包含 161,920 个未条件体素、所有十二个候选
footprint 及大量背景；三个固定目标体位于搜索区外并保持真值。优化器不知道目标
体个数、位置、形状或总体积。

- `soft_voxel`：soft occupancy 直接进入物性和正演；
- `ste_voxel`：前向使用 0.5 阈值 hard occupancy，反向使用 sigmoid surrogate；
- 初始 logit `-6.0`，初始 hard 模型严格等于 I0 baseline；
- 300 次 Adam，learning rate `0.15`，无 weight decay、无体积/平滑/真值 ROI
  正则；温度 `1.0 x 100, 0.5 x 100, 0.2 x 100`；
- 每5步做一次 hard 正演并只按 hard physics loss 选择 best；
- 五种 correct observation 均运行；地震另运行 zero/shuffled 控制；
- hidden union 与两个单体 recall 只在运行后审计，不能用于选择。

Q2 correct seismic 的预注册门仍为 hard attainment 至少0.95、hidden-union IoU
至少0.70、两个目标体 recall 各至少0.80，且 correct 必须优于 zero/shuffled。
若 soft 物理显著下降但 hard 门失败，直接记录为自由体素 soft/hard gap；若 STE
hard 物理通过而几何失败，则开始进入物理非唯一性/缺少几何先验的讨论。

## Q2b 补充冻结：单调 hard 坐标求解器

Q2 结果生成后、Q2b 运行前，于 2026-08-04 冻结一个不使用 soft endpoint 作为
验收对象的离散对照。它从相同 I0 hard baseline 开始，每轮在当前 hard 二分类体上
计算连续物理方向导数，按预测改善分数提出 `32/64/128/256/512` 个体素翻转的五个
hard proposal；全部 proposal 做 hard 正演，只接受 hard physics RMSE 严格下降的
最佳者，否则停止。已为 target 的体素允许翻回背景，因此不是只增不减。

- 最多80轮；无体积、平滑、候选形状、目标数量或 truth ROI；
- 每轮的模型选择只使用 hard physics；
- correct 五观测均运行，地震另有 zero/shuffled；
- hard loss 必须单调不增，所有几何指标均为运行后审计；
- 该方法用于判断“soft 代理能否由 hard 接受门修复”，不是部署算法声明。

若 Q2b 地震能够达到物理门而几何仍差，开始讨论 acquisition-domain 非唯一性；若
仍停留在低 hard attainment，则说明仅增加 hard 接受门不足，需要更强参数化先验、
冻结 embedding/flow 审计或学习型映射。

### Q2c 事后机制诊断（不改写 Q2b）

Q2b 的 correct seismic 在精确恢复两体各70%后停止，停止时最小 proposal 仍为
32体素。为区分“hard 方法本身失败”和“proposal 粒度过粗”，在查看 Q2b 后明确
冻结一次只针对该问题的 Q2c：从原始 I0 重新开始，flip counts 改为
`1/2/4/8/16/32`，最多160轮，仅运行 seismic correct/zero/shuffled，其他规则
完全不变。Q2c 是事后诊断，不得回写为 Q2b 预注册成功，也不据此继续搜索更多
步长集合。

## Q3 补充冻结：checkpoint embedding endpoint

Q2/Q2b/Q2c 结果生成后、Q3 首次运行前，于 2026-08-04 冻结真实 embedding
端点审计。Q3 读取正式 checkpoint 中冻结的 `embedding.weight`，但绝不实例化或
运行 flow U-Net；因此它只比 Q2 多出 15 类 cosine decoder 和连续 embedding
参数化，不混入生成先验或轨迹控制器。

- 真值、I0、搜索区、物性码表和五种观测完整复用 Q2；
- 搜索区外的 hard 类别固定为 I0，搜索区内每个体素从 raw label-0 的真实 embedding
  精确初始化；
- `soft_embedding` 使用当前部署定义的 15 类 cosine-softmax；地震 soft 正演继续使用
  地下 rock-conditional acoustic，gravity 继续使用全类别期望密度；
- `ste_embedding_rock` 是 hard-forward/soft-backward 对照；地下 hard/soft surrogate
  都排除物理上不允许的 air 类；
- 200 次 Adam，learning rate `0.02`，无 weight decay，gradient clip `1.0`；温度
  `0.5 x 40, 0.2 x 60, 0.1 x 60, 0.05 x 40`；每5步进行 hard 正演；
- 单体素 embedding norm 上限为 checkpoint 最大行范数的4倍；
- 模型选择只按 hard physics RMSE，所有 hidden-body 几何指标仅作事后审计；
- correct 五观测均运行，地震另运行 zero/shuffled；不使用体积、平滑、形状、数量或
  truth ROI 正则。

Q3 correct seismic 门仍为 hard attainment 至少0.95、hidden IoU 至少0.70、两个
目标体 recall 各至少0.80，并要求 correct 明显优于 zero/shuffled。若 Q2 hard 方法
明显优于 Q3，则真实 embedding/15类 decoder bridge 是新增故障源；若 Q3 endpoint
通过而 Q4 flow 失败，才把主要责任进一步定位到冻结生成速度或轨迹控制器。

### Q3b 事后全类别竞争诊断（不改写 Q3）

Q3 首次完整结果显示，correct seismic 的 `ste_embedding_rock` 会用 raw labels
4/5/6/8/9/10/12 共七种类别降低 hard loss，而 gravity 完全用 raw label 6 代替
目标 label 9。为了把“embedding 连续几何”与“15类物性竞争/非唯一性”分开，在
查看 Q3 后冻结 Q3b：保留同一个 checkpoint embedding、初始状态、搜索区、优化器、
温度、观测和 hard-only 选优，只把地下 decoder 限制为 raw label 0 与 raw label 9。

Q3b 同时运行 `soft_embedding_binary` 与 `ste_embedding_binary`，五观测和地震三控制
保持不变。若 Q3b seismic 显著优于 Q3 全类别方法，说明类别竞争是新增主因；若仍
低于 Q2/Q2b，则剩余差异属于 embedding 状态几何、梯度缩放或优化调度。Q3b 是查看
Q3 后的机制诊断，不得回写成 Q3 预注册结果。
