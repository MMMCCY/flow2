# Three-case offline summary

执行（服务器仓库根目录）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python project/geodata-3d-conditional/scripts/reports/build_pseudo_well_three_case_summary.py --out project/geodata-3d-conditional/reports/pseudo_well_three_case_summary_v1
```

输出目录不可覆盖；重跑请选择新版本目录。

D_global_nonair=truth!=-1；类别仅由各例真值非空气类确定，两方法相同。全模型含原条件及新增伪井，不能单独证明井外恢复。E_holdout/E_original_free直接读取原registry的mask/classes；后者包含新增伪井，前者虽无新增井直接写入，但属性参与全局代理，非独立观测测试。

IoU=TP/(TP+FP+FN)，OA=正确数/域体素数，mIoU=指定类别中有定义IoU的算术平均（严格复用原voxel_metrics，空并集跳过，未定义保留NA）；预测空气不剔除。miou_classes记录指定集合，miou_defined_classes记录实际参与均值集合。所有域指标与逐类IoU/真值数见CSV。

12组历史域/方法重算均在1e-12容差内吻合Stage24原始CSV（逐类及目标计数一起检查），详见verification.json；39份输入文件哈希前后不变。18行主指标，6行global、6行holdout；9张图各PNG300dpi/PDF。

本次新增训练=0、Flow推理=0、属性恢复=0，仅CPU读既有标签与计算评价。历史结果、源码快照和认可3D图未覆盖。

限制：单seed、复用案例、人工可分属性代理、已知模糊核且无额外噪声，不构成泛化或真实物理反演证据。case01主要目标已有原井命中，小型零井目标仍遗漏；case02严重误报与对象合并仍在，体积偏大；case03也有误报和范围偏差。完整属性直接分类优于伪井Flow的历史数据保留于reference_metrics_sources.json索引，但其信息预算不同。

文件：

- CAPTIONS.md
- asset_manifest.json
- case01_center_slices_clean.pdf
- case01_center_slices_clean.png
- case01_center_slices_with_errors.pdf
- case01_center_slices_with_errors.png
- case02_center_slices_clean.pdf
- case02_center_slices_clean.png
- case02_center_slices_with_errors.pdf
- case02_center_slices_with_errors.png
- case03_center_slices_clean.pdf
- case03_center_slices_clean.png
- case03_center_slices_with_errors.pdf
- case03_center_slices_with_errors.png
- class_iou.csv
- global_comparison.csv
- global_comparison.pdf
- global_comparison.png
- holdout_comparison.csv
- holdout_comparison.pdf
- holdout_comparison.png
- metrics_all_domains.csv
- quantitative_comparison.pdf
- quantitative_comparison.png
- reference_metrics_sources.json
- verification.json
- README.md

人工图像复核：已打开参考原图及本轮全部9张PNG（三例切片通过并排预览检查），确认切片方向与参考一致、raw色表共享、空气白色、错误白/红、图例与文字无裁切。9对PNG/PDF齐全，PNG均为300dpi。
