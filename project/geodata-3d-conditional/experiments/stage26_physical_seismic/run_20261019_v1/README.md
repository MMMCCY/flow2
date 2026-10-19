# Stage26 完整交付

本轮物理链成功执行并可复核，但本案例D不优于A或无地震更新D0：连续拟合改善，分类/生成后的地质和物理一致性未改善。

- [中文报告](REPORT.md)
- [目标体对比](figures/fig02_target_body_comparison.png)
- [整体岩性图](figures/fig01_full_geology_comparison.png)
- [固定Y26地质剖面](figures/fig03_section_Y26.png)
- [D0独立剖面](figures/fig04_D0_section_Y26.png)
- [统一色标时间域地震图](figures/fig05_seismic_Y26.png)
- [条件质量](condition_quality.csv) · [改变机制](condition_transitions.csv)
- [形态指标](shape_metrics.csv) · [完整固定域指标](metrics.csv)
- [物理误差](physical_errors.csv) · [记录窗/空气诊断](physical_window_diagnostics.csv)
- [命令与无网络重绘](COMMANDS.md) · [最终核验](FINAL_CHECKS.json)
- [工程记录](ENGINEERING.md) · [阶段耗时](TIMING.json)

成功反演1次，新Flow D0/D各1份/32步，共64前向；旧A/B/C、旧观测和配置原样复用，地下空气不修补，负结果不筛掉。所有图有PNG/PDF及独立面板。

新源码在scripts/stage26/，执行源码快照在source_snapshot/；反演数值优先保存在preprocessing/numerical_bundle.pt，单独物性/地震/分类场也保留。arms/包含两臂条件和预测。旧资产路径依赖由INPUT_FREEZE记录，需连同仓库既有Stage25保存。
