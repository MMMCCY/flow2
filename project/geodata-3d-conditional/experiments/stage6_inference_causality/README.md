# Phase 6Q inference causality

该目录保存正式训练前的五目标体推理因果审计。权威规范为
`docs/PHASE6Q_SIMPLE_CAUSAL_SPEC.md`。

目录约定：

- `configs/`：结果前冻结的解析真值与优化协议；
- `runs/`：不可覆盖的机器运行产物；
- `reports/`：独立审计、汇总和 notebook 输入；
- `WORKLOG.md`：命令、环境、阶段状态与结果索引。

本阶段不得写入 Phase 6B 训练目录，不修改 flow checkpoint，不把解析立方体当作
StructuralGeo 分布内证据。Q0-Q2 不加载 checkpoint；Q3/Q3b 只读取正式 checkpoint
的冻结 `embedding.weight`，不实例化或运行 flow U-Net。

Stage 7 已完成。D7 权威目录为
`runs/five_body_cuboid_v1/d7_observation_specificity_v1/`；结构化 hard-geophysics
权威报告为 `reports/stage7_v1_final_v2/STAGE7_REPORT.md`。最终方法只用 hard
observed seismic mismatch 接受和选择 proposal，在解析 cuboid 上精确恢复隐藏体，
并在三个 deterministic StructuralGeo replicas 上通过 correct/zero/shuffled/
wrong-case 统一交叉评价。正式训练仍未启动。
