# Figure captions

Ground truth=真值；Frozen Flow=冻结Flow；Proposed method=改进方法。目标类为 raw9。
三例均仅 seed42。Frozen Flow 是原9井与地表条件的 A_9；改进方法是恢复属性生成完整多岩性伪井后的冻结 Flow：case01/02 来自 Stage24，case03 来自 Stage23 v2。具体资产见 manifest。

## case01_center_slices_clean.png / .pdf

固定零基 X=32、Y=32、Z=32；横/纵轴依次为 Y/Z、X/Z、X/Y，坐标均为0–63体素索引。切片使用 np.take(volume,32,axis=axis).T、origin=lower、nearest。共享原参考图离散色表，空气−1白色，raw0…13固定映射。 三行分别为三个固定中心截面。

## case01_center_slices_with_errors.png / .pdf

固定零基 X=32、Y=32、Z=32；横/纵轴依次为 Y/Z、X/Z、X/Y，坐标均为0–63体素索引。切片使用 np.take(volume,32,axis=axis).T、origin=lower、nearest。共享原参考图离散色表，空气−1白色，raw0…13固定映射。 每个岩性行后是全截面 prediction != truth：包括空气，正确白色、错误红色；真值列全为正确。该图像域与非空气统计域不同。

## case02_center_slices_clean.png / .pdf

固定零基 X=32、Y=32、Z=32；横/纵轴依次为 Y/Z、X/Z、X/Y，坐标均为0–63体素索引。切片使用 np.take(volume,32,axis=axis).T、origin=lower、nearest。共享原参考图离散色表，空气−1白色，raw0…13固定映射。 三行分别为三个固定中心截面。

## case02_center_slices_with_errors.png / .pdf

固定零基 X=32、Y=32、Z=32；横/纵轴依次为 Y/Z、X/Z、X/Y，坐标均为0–63体素索引。切片使用 np.take(volume,32,axis=axis).T、origin=lower、nearest。共享原参考图离散色表，空气−1白色，raw0…13固定映射。 每个岩性行后是全截面 prediction != truth：包括空气，正确白色、错误红色；真值列全为正确。该图像域与非空气统计域不同。

## case03_center_slices_clean.png / .pdf

固定零基 X=32、Y=32、Z=32；横/纵轴依次为 Y/Z、X/Z、X/Y，坐标均为0–63体素索引。切片使用 np.take(volume,32,axis=axis).T、origin=lower、nearest。共享原参考图离散色表，空气−1白色，raw0…13固定映射。 三行分别为三个固定中心截面。

## case03_center_slices_with_errors.png / .pdf

固定零基 X=32、Y=32、Z=32；横/纵轴依次为 Y/Z、X/Z、X/Y，坐标均为0–63体素索引。切片使用 np.take(volume,32,axis=axis).T、origin=lower、nearest。共享原参考图离散色表，空气−1白色，raw0…13固定映射。 每个岩性行后是全截面 prediction != truth：包括空气，正确白色、错误红色；真值列全为正确。该图像域与非空气统计域不同。

## global_comparison.png / .pdf

D_global_nonair=truth!=-1；包含原条件与新增伪井位置，表示最终整体吻合，不能单独证明井外补全。OA/P/R为百分数，mIoU/IoU为0–1；仅较高且非平局的OA/mIoU/IoU/P/R加粗。体积比理想值1，不标为越大越好。

## holdout_comparison.png / .pdf

E_holdout沿用原注册mask、固定类别及空分母规则；无新增井条件直接写入，但属性参与全局代理，不是独立观测测试。数值格式与强调规则同上。

## quantitative_comparison.png / .pdf

上排全模型非空气域mIoU/目标IoU，下排E_holdout对应指标；统一0–1纵轴。每例seed42配对比较，无误差棒、置信区间或显著性检验。
