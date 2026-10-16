# 可运行集合后处理代码

这份代码完成硬标签集合筛选、体素计数、频率阈值mask、候选审计与Figure8/9式三维图。它不是Flow推理程序，也不声称已完成200个真实实验模型。代码与03提示词是本次补充；旧02面向既有资产的系统绘图。

## 输入

- `labels.npy`：整数硬岩性标签，形状`[N,X,Y,Z]`或`[N,1,X,Y,Z]`。`.npz`同理，key为`labels`。同一案例、协议、网格和方法。`.npy`支持mmap，计数逐样本累加。
- `ids.json`：按数组行顺序排列的N个唯一字符串ID。独立生成得到相同模型可以重复几何，不能重复ID。
- `provenance.json`：至少`case_id`、`method_id`，建议完整保存commit、checkpoint hash、条件/观测hash、参数及source生成规则。程序记录输入文件hash，但无法替你证明输入属于同一协议。
- 可选`truth.npy`：完整硬类别真值，三维XYZ。`--mode oracle-iou`必须提供。此模式按全类别IoU择优，不等同于未命中对象IoU。
- `--mode physics`需要score CSV：`sample_id`与指定的score column；默认`hard_physics_rmse`，默认越小越好。CSV由实验环境中真实hard-forward评分生成，本代码不负责正演。ID必须与整个集合完全一致。
- 条件评价需同时提供`--condition-mask`与`--condition-labels`，前者二值mask，后者原始硬类别。提供后全部模型必须满足硬条件；违反时退出，不静默筛除。`.npz`mask key为`mask`。
- 可选`--evaluation-mask`定义未命中对象/自由域评价范围，只用于事后指标，不进入physics筛选。若只覆盖真值体内，precision退化，需另设预先固定的周边ROI。

原仓库.pt模型需在已有torch环境按manifest读取后转换为.npy。Stage9的.pt.gz是批次，必须依据chunk与offset读取，不能把一个chunk当一个实现或把预测响应当模型。

## 命令

环境需要`numpy`、`Pillow`。根据你的实验Python环境安装已有依赖；本包不自动安装。

```powershell
python ensemble_occurrence.py --labels labels.npy --ids ids.json --provenance provenance.json --truth truth.npy --target-label 9 --mode oracle-iou --keep 100 --out result_oracle
python ensemble_occurrence.py --labels labels.npy --ids ids.json --provenance provenance.json --truth truth.npy --target-label 9 --mode physics --score-csv scores.csv --score-column hard_seismic_rmse --keep 100 --out result_physics
python -m unittest test_ensemble_occurrence.py
```

默认要求N200。实际N3时显式`--expected-count 3 --keep 3`，不能复制成200。更大池先按照冻结编号取前200并保存对应IDs；本程序不自动从更大池预筛200。`--mode all`输出整个集合；`--mode random`固定随机种子选keep个。oracle/physics运行同时输出all、selected与相同规模random_control。

默认严格超过40/60/80%及全部命中，N100对应41/61/81/100次，N200对应81/121/161/200次。加`--at-least`改为至少40/60/80%。`--spacing dx dy dz`提供实际体素间距；默认1为voxel units，不表示米。`--z-down`仅改变绘图Z方向。绘图使用固定等距及正视相机，保持各集合一致。

## 输出与限制

`audit.json`保存输入hash、选择模式、样本数、阈值和每阈值IoU/P/R/体积。`candidate_audit.csv`保存全部候选的分数、选中标记及事后几何指标。各集合保存selected IDs、`occurrence.npz`中的整数count、frequency、threshold_masks、minimum_counts，以及Figure8/9的SVG与PNG。Figure8按固定顺序或排序的等距分位展示6个实现，避免人工挑图；仓库论文版可扩为12个。

程序不计算连通体、对象合并、物理预测或地质有效性，这些需要实际实验环境的已有评价模块，详见03。当前代码不输出阈值套叠、16:9汇报排版或论文PDF，已在完整接入提示词中提出。参考代码的PNG/SVG可直接查看，是准确体素表面而非平滑等值面；出版排版可在不改变mask的基础上优化。

频率是所选模型集合的经验出现频率，未经校准不称后验概率或置信区间。阈值mask不是完整多岩性模型，不能直接继承单样本的物理残差。红色条件overlay独立于阈值mask，其出现不代表该阈值也覆盖那里。评价指标目前基于体素数；真实物理体积为体素数乘spacing乘积，单位由输入提供。

## 已执行验证

2026-10-09运行6个测试通过，覆盖严格/包含阈值、小集合离散性、硬标签计数、排序与稳定平局、物理选择不受truth影响、退化指标和重复ID拒绝。人工200成员、12³网格跑通oracle/all/random_control全部输出，并查看PNG确认三维与正视、阈值和空面板显示。Windows沙箱临时目录ACL阻止测试读写，提升权限后测试通过；后处理演示本身在工作区直接运行成功。

`software_demo/`全部是平移盒子人工数据，图题明确`SYNTHETIC SOFTWARE TEST ONLY`，**不是MCYR实验结果，禁止作为方法效果图**。demo脚本首次运行输出至新目录；再次运行须使用新的输出目录，程序拒绝覆盖。未读取真实checkpoint，未运行200个Flow模型，未验证真实隐藏体恢复改善。
