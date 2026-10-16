# 人工视觉核对（2026-10-09）

实际打开并检查：

- `FLOW_ONLY/all200/figure9_free_subsurface_iso_paper.png`：地下自由域、物理aspect、固定等距视角；100%区域明显缩小，红色条件overlay独立；没有降低阈值。
- `STAGE15_TRACE_V4/all200/figure9_free_subsurface_front_presentation.png`：正视沿+Y，XYZ extent6.4/6.4/3.2km，cell100/100/50m；真实不平滑体素轮廓与字体、图例均可读。
- `STAGE19R_ADAPTER/obs_top100/figure9_free_subsurface_iso_presentation.png`：N100严格41/61/81/100次；红色overlay不是阈值mask；1920×1080布局与固定坐标正确。
- `STAGE19R_ADAPTER/all200/figure8_full_iso_paper.png`：左侧独立truth/条件，右侧12个冻结等距ID成员；source000/018/.../199，未手工挑图。外壳线框、非目标透明、完整target橙色。
- 另外打开独立 `software_validation/edge_cases_iso.png` 核对空、满、触边mask。该图显式标注SOFTWARE TEST ONLY，不属于实验图件。

203个正式PNG均通过Pillow格式核验。数值版与出版版summary、12份
occurrence.npz全部逐数组一致；渲染没有改变count、frequency或mask。
没有平滑、填洞、删除小体或交换轴序。

等距图从上方看薄目标体时会弱化厚度差异，因此保留独立正视图和真实
三维体积/IoU/P/R。不能仅凭相似投影认定体积恢复成功。
SVG为准确体素外露面矢量图；PNG是对应栅格图。
