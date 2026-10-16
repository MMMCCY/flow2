# Stage23 figure captions / 图注

All main figures: **fullgeo_case03, seed42, synthetic property surrogate with known Gaussian kernel; frozen network**. No extra noise, no training, no new deconvolution, no new conditioning or model inference. These are retrospective single-case illustrations, not ensemble uncertainty or independent geological TEST. Original nine boreholes and surface constraints are retained in all models. The Stage23 model additionally uses full multi-lithology pseudo-wells from the old total512 layout (503 extra columns); wells are omitted visually. Shapes and values are unchanged.

## Figure 1 / 图1：Full geology comparison / 整体多岩性模型

Ground truth, frozen nine-well Flow A, and frozen Flow conditioned by restored-property pseudo-wells. Upper row: complete nonair64³ geology. Lower row: the SAME display-only quadrant removal X≥32 and Y<32. All panels use one camera, parallel scale, fixed scene bounds, unit voxel spacing and Stage15 discrete raw-label colours. Air (raw−1) is hidden, and raw labels0…13 have no invented geological names. Labels use XYZ array order, as verified against the VTK ImageData Fortran-order cell mapping. Fixed E_holdout all-class mIoU comes from the saved CSV; ground truth has no performance score. No independent panel trimming or rescaling.

真值、原9井冻结Flow及恢复属性完整伪井条件下的冻结Flow。上排完整非空气体，下排统一剖切（仅显示移除X≥32且Y<32的象限）。同相机、焦点、尺度、色表、光照和64³边框；无几何平滑修补。原条件保留但不绘井柱。原始张量轴为XYZ，体素边界0…64、中心0.5…63.5，不虚构米或海拔。

## Figure 2 / 图2：Complete target bodies / 完整目标岩体

All raw9 voxels in the FULL64³ scene, including every FP fragment, with opaque orange voxel surfaces and no truth ghost or target cutaway. Upper row uses a common isometric camera, lower row the same orthographic view along−Y. Truth-only object labels are retrospective annotations; there are no ROI restrictions on predictions. Original-unknown-domain (E_original_free) IoU, precision and recall are saved metrics while the scene displays the complete volume. For A versus Stage23, predicted/true target volume ratios are2.63993 versus1.90075. The common hidden-object subset is truth object1 intersect E_original_free outside the Stage23 NEW mask:510 voxels; A/C/COPY recover0, Stage23 recovers144 (28.24%). This percentage is coverage inside that object subset, NOT object IoU or global recall. Whole object1 any-prediction coverage is206/572, with one-to-one matched IoU0.122137. Missing parts, splitting and distant false positives remain visible.

三列保留全部目标预测，不按真值或连通分量筛选、不填洞、不叠加真值ghost。上下排为统一等轴和沿−Y正交视角。真值对象1标注仅用于回顾性解释。整场景展示与指标域分开：P/R/IoU在固定原未知域计算，144/510是同一井外对象子集内覆盖率，不能标成IoU。本轮仍有漏失、分裂和误报，不能称准确完整重建。

## Figure S1 / 补图：Bridge evidence / 桥接证据

Left: all-label pseudo-well accuracy on the identical18555-voxel added_mask, with target FN/FP; no truth-based removal of incorrect segments. Right: saved fixed E_holdout target IoU and all-class mIoU for A, smooth pseudo-well C, Stage23, and accurate-well B. No error bars for one seed. Direct restored classification is explicitly retained: E_original_free target IoU0.9677 and mIoU0.6627, much better than Flow, but it uses the FULL spatial field rather than sampled pseudo-well information. B uses accurate true well labels and is not an information-matched deployment competitor. Holdout has no added condition, but its property information participated in the global surrogate; it is not independent observation holdout.

左图保留伪井全部错误：准确率60.99%→92.44%，目标FN211→7、FP0→1。右图保留准确井B对照与直接分类的优势，避免只展示有利结果。转换和井外恢复改善不等于真实地球物理反演、跨案例稳健或论文总目标完成。

## Rendering and editable output / 渲染与编辑

PNG600dpi. PDF embeds raster3D renders, with vector titles, axes labels/triads, metrics, annotations and discrete legend (embedded TrueType fonts); it is NOT wholly vector3D geometry. Original1500×1360 panel PNGs are in panels/. Style reference: actual Stage15 figures. Structural reference: local Synthetic Geology PDF, DOI10.1029/2025JH000986, pages12–13 / Figures7–8; no source artwork or multiple-realization claim copied. Reproduction command is in README.md; RENDER_CONFIG and INPUT_ASSETS bind geometry, rendering and exact saved evidence.
