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
