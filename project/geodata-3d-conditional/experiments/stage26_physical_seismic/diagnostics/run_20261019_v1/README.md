# Stage26 离线诊断交付

[中文诊断报告](REPORT.md)给出已确认现象、合理推断及前端/候选接口的下一步建议。

- [六面板固定Y26分类图](figures/fig01_classification_Y26.png)
- [固定Y26时深/物性误差图](figures/fig02_twt_property_errors_Y26.png)
- [候选来源](candidate_sources.csv) · [六种条件质量](condition_quality.csv)
- [核心形态](core_morphology.csv) · [完整域指标](domain_metrics.csv)
- [错误转移](error_transfers.csv) · [逐类转移](class_transition_counts.csv)
- [连续阻抗误差](continuous_log_impedance_errors.csv)
- [原正确→错误位置](right_to_wrong_log_errors.csv)
- [时深差](twt_differences.csv) · [物理残差](physical_residuals.csv)
- [命令/重绘](COMMANDS.md) · [最终核验](FINAL_CHECKS.json) · [耗时](TIMING.json)

新增一次ORACLE_DIAGNOSTIC_ONLY反演，新Flow/checkpoint/训练/观测生成均0。原正式Stage26的151份封存资产全部核验未改动。六分类张量在classifications/；oracle prior/updated/模拟地震和数值bundle在oracle/，不生成正式推理注册表，绝不能送入Flow或作为可部署成功结果。

配置、输入身份及oracle干预定义见configs/、INPUT_FREEZE.json、ORACLE_PROTOCOL.json。源码scripts/stage26_diagnostics/，执行源快照source_snapshot/。两图均有PNG/PDF，不绘制新三维展示。原观察集合只来自原9井/地表，本例候选效果不保证实际未见岩性可安全剔除。
