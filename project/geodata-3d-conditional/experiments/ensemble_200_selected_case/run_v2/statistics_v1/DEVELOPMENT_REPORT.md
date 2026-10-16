# 同案例 200 采样、筛选与绘图结果

**真实实验结果：** 200 个独立source draws；每个draw生成1个共享baseline、1个Stage15 trace v4模型和1个Stage19R adapter模型，共600个完整硬岩性模型。失败数0，硬条件违背0。

案例：`stage19_test_case004`。用户选择按历史 adapter 三种子的中位目标 IoU 最高案例进行展示。历史中位 IoU为0.6710347944；完整选择排名在protocol。这是事后选择案例，不能解释为无偏跨案例比较或新泛化证据。200个实现是一个案例内的随机重复，不是200个独立地质测试案例。

Stage15固定v4 positive-endpoint/confidence方法迁移至这个TEST案例；不冒称历史cond_generation_0结果。Stage19R使用正式epoch4已训练54,003参数adapter；此次不训练，不能把整条方法称training-free。两条路线共享相同checkpoint、条件、观测、证据和source draws。

观测筛选：完整硬地质→固定binary acoustic映射→卷积地震正演→未归一化hard seismic RMSE（振幅单位，越低越好）；不含真值指标。所有obs IDs均在本次后处理读取truth前写出。case选择与oracle筛选单独声明使用真值。

random100使用20261009种子无放回；obs/oracle平局按冻结source order；oracle按完整raw-label9真实IoU降序，仅事后候选支持诊断。每个集合独立重算count，不在不同Top100间进行伪配对。



## 全集合与筛选结果

| 方法 | 集合 | 中位候选IoU | 中位hard RMSE | >80%自由域IoU | 100%自由域IoU |
|---|---|---:|---:|---:|---:|
| FLOW_ONLY | all200 | 0.3259 | 0.06067 | 0.5074 | 0.0192 |
| FLOW_ONLY | random100 | 0.3262 | 0.06060 | 0.5002 | 0.0443 |
| FLOW_ONLY | obs_top100 | 0.3553 | 0.05882 | 0.5175 | 0.0381 |
| FLOW_ONLY | oracle_top100 | 0.3591 | 0.05903 | 0.5863 | 0.0578 |
| STAGE15_TRACE_V4 | all200 | 0.3488 | 0.06509 | 0.8198 | 0.5284 |
| STAGE15_TRACE_V4 | random100 | 0.3491 | 0.06517 | 0.8209 | 0.5719 |
| STAGE15_TRACE_V4 | obs_top100 | 0.3719 | 0.06345 | 0.8285 | 0.5614 |
| STAGE15_TRACE_V4 | oracle_top100 | 0.3738 | 0.06369 | 0.8385 | 0.5654 |
| STAGE19R_ADAPTER | all200 | 0.6270 | 0.04140 | 0.7914 | 0.2790 |
| STAGE19R_ADAPTER | random100 | 0.6261 | 0.04146 | 0.7929 | 0.3461 |
| STAGE19R_ADAPTER | obs_top100 | 0.6548 | 0.03991 | 0.7881 | 0.3577 |
| STAGE19R_ADAPTER | oracle_top100 | 0.6549 | 0.04017 | 0.7939 | 0.3639 |

表中IoU为完整目标类别的候选IoU；阈值IoU是在核实的地下自由域中评价。硬条件区的频率不能作为隐藏体恢复证据。

## STAGE15_TRACE_V4

1. 候选支持：同源200对的中位IoU增量 +0.020074，本案例内改善。中位hard RMSE增量 +0.004201；地质与物理改善分别判断。

2. 观测筛选：all200平均候选IoU 0.351640，random100 0.352371，obs_top100 0.373822，oracle_top100 0.378715。obs相对random增量 +0.021451；oracle与obs差 +0.004893。只比较集合统计，不把不同Top100的成员伪配对。

3. 频率摘要：all200地下自由区的四个严格阈值指标如下。高频一致性不等于正确性；提高阈值所丢失的真目标与保留下来的误报均明确报告。

>40%: IoU=0.5021, precision=0.5021, recall=1.0000, TP/FP/FN=10144/10061/0, voxels=20205, components6=3.

>60%: IoU=0.6441, precision=0.6445, recall=0.9992, TP/FP/FN=10136/5592/8, voxels=15728, components6=28.

>80%: IoU=0.8198, precision=0.8448, recall=0.9652, TP/FP/FN=9791/1799/353, voxels=11590, components6=9.

all-hit: IoU=0.5284, precision=0.9987, recall=0.5288, TP/FP/FN=5364/7/4780, voxels=5371, components6=10.

Unique decoded models: 200/200。独立draw即使同模型也保留多重性。

## STAGE19R_ADAPTER

1. 候选支持：同源200对的中位IoU增量 +0.294359，本案例内改善。中位hard RMSE增量 -0.019122；地质与物理改善分别判断。

2. 观测筛选：all200平均候选IoU 0.624161，random100 0.623745，obs_top100 0.653705，oracle_top100 0.660253。obs相对random增量 +0.029959；oracle与obs差 +0.006549。只比较集合统计，不把不同Top100的成员伪配对。

3. 频率摘要：all200地下自由区的四个严格阈值指标如下。高频一致性不等于正确性；提高阈值所丢失的真目标与保留下来的误报均明确报告。

>40%: IoU=0.8706, precision=0.8904, recall=0.9751, TP/FP/FN=9891/1217/253, voxels=11108, components6=1.

>60%: IoU=0.8869, precision=0.9721, recall=0.9101, TP/FP/FN=9232/265/912, voxels=9497, components6=1.

>80%: IoU=0.7914, precision=0.9926, recall=0.7960, TP/FP/FN=8075/60/2069, voxels=8135, components6=1.

all-hit: IoU=0.2790, precision=1.0000, recall=0.2790, TP/FP/FN=2830/0/7314, voxels=2830, components6=21.

Unique decoded models: 200/200。独立draw即使同模型也保留多重性。

## 图件与核验

Figure8按冻结顺序/排序等距分位展示12个成员。Figure9四个严格阈值：N200要求81/121/161/200次，N100要求41/61/81/100次。full/free_subsurface均有等距与正视、论文和1920×1080汇报PNG/SVG。红色是独立条件体素overlay；不能看成阈值命中。只知aggregate condition mask时不臆造surface/well分类。

绘图使用真实二值体素外露面，没有平滑、闭运算、填洞、删小体或改轴序。XYZ物理extent为6.4/6.4/3.2km、cell100/100/50m，Z向上。full与free区的统计和指标均独立保存。

自动产物核验PASS：200源IDs、600模型、同源噪声哈希、条件一致、100无放回、完整排序、所有count和frequency重算、严格阈值嵌套与all-hit、12个Figure8 IDs；0个PNG通过格式检查。PNG人工视觉检查记录另附。

软件测试：随附6个unittest通过；新增接入与Stage19回归36个pytest通过；出版渲染测试1个通过。13个既有Matplotlib弃用警告不影响结果。

## 范围与限制

binary high-contrast、noiseless inverse-crime机制上限；不是field seismic或15类实际物性反演。q为continuous binary evidence，频率为未校准empirical occurrence。阈值mask不是完整地质实现，不能继承成员的物理残差。本次未做五体任务、逐对象恢复或跨案例泛化评价。

原参考ensemble_occurrence.py缺失，已按README与测试重新实现；该差异详见ASSET_AUDIT_REPORT。checkpoint、数据和GPU均实际可用。旧实验和用户既有改动保持原样，没有push或发布。

详细路径、哈希与冻结参数见protocol.json；候选指标见candidate_metrics.csv；同源增量见paired_deltas.csv；图件见FIGURE_INDEX.md。
