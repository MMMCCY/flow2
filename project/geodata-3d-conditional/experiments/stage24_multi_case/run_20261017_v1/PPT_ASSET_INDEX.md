# PPT_ASSET_INDEX

不制作PPTX；以下图片可直接插入。所有图例/长方法说明与数字留在讲稿或外置图注。

## 第1页：fullgeo_case03 / seed42 — 原有主要隐藏体示例

[整体图](reports/fullgeo_case03/fig01_full_geology_comparison.png) · [目标图](reports/fullgeo_case03/fig02_target_body_comparison.png)；同名PDF、_clean版本与panels/九个单面板均在同目录。

可陈述的数字：伪井准确率0.609863→0.924441；holdout目标IoU0.022059→0.130112；新目标FP=2579，体积比=1.900750。不能把所有指标预写为改善；直接恢复分类IoU=0.967723必须保留。

限制：查看REPORT全部对象共同井外TP及匹配IoU，完整保留未恢复对象、碎裂和FP。

## 第2页：fullgeo_case02 / seed42 — 多个零井目标对象，保留失败

[整体图](reports/fullgeo_case02/fig01_full_geology_comparison.png) · [目标图](reports/fullgeo_case02/fig02_target_body_comparison.png)；同名PDF、_clean版本与panels/九个单面板均在同目录。

可陈述的数字：伪井准确率0.634283→0.930209；holdout目标IoU0.396172→0.464922；新目标FP=12197，体积比=1.608127。不能把所有指标预写为改善；直接恢复分类IoU=0.998069必须保留。

限制：查看REPORT全部对象共同井外TP及匹配IoU，完整保留未恢复对象、碎裂和FP。

## 第3页：fullgeo_case01 / seed42 — 小目标/全类表现，非主要隐藏体新成功

[整体图](reports/fullgeo_case01/fig01_full_geology_comparison.png) · [目标图](reports/fullgeo_case01/fig02_target_body_comparison.png)；同名PDF、_clean版本与panels/九个单面板均在同目录。

可陈述的数字：伪井准确率0.800918→0.968251；holdout目标IoU0.014286→0.160000；新目标FP=468，体积比=4.415254。不能把所有指标预写为改善；直接恢复分类IoU=1.000000必须保留。

限制：case01无原井零命中的major目标，不可作为第二个主要隐藏体恢复成功。

## 第4页：三例简洁数字表

[PPT_COMPARISON.csv](PPT_COMPARISON.csv)为7列紧凑三例数字表；[three_case_summary.csv](three_case_summary.csv)为扩展表：分别列转换、井外指标、FP/体积和直接分类；完整A/B/C参考见[three_case_metrics.csv](three_case_metrics.csv)。不画误差棒，不称未见地质泛化。

## 必须补充的结果解释

case02四个零条件major对象均出现共同井外TP增长，但最大预测分量仍合并8个真对象、原未知域FP11891；新方案总体FP与体积高估比旧C更大，不能宣称范围更准。case01主要对象已被原井命中，1/6/3体素的三个零井小对象仍全部漏失。详细匹配/任意覆盖/FP见REPORT复核结论。两例直接恢复分类均明显好于Flow；不隐去这一事实。
