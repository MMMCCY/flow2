# Stage23 revised captions / 修订图注

Both plates show **fullgeo_case03, seed42, synthetic property surrogate with known Gaussian kernel; frozen network**. Column order: ground truth, original nine-well frozen Flow A, Stage23 frozen Flow supplied with restored-property full multi-lithology pseudo-wells. “Ours” denotes preprocessing/conditioning, not retraining. The original boreholes and surface constraints are retained; Stage23 uses the old total512 layout, including503 additional well columns. No new inference, training, restoration solve or condition generation was performed. Each PDF contains raster3D with editable vector titles/panel letters/legend, not vector3D geometry.

## Figure A / 图A

Upper row: full nonair geology. Lower row: identical display-only front-corner removal using the existing cutaway_mask with cut_fraction0.52 (X≥33.28 and Y<33.28 in cell-index coordinates). Actual cell centers/array ordering are not transformed: XYZ tensor axes, VTK ImageData dimensions=shape+1, unit cell spacing, Fortran-order ravel. All panels use the same orthographic camera, focal point, scale and opaque exact voxel faces; no contour interpolation, smoothing, repair or per-model image cropping. Air−1 is invisible. The shared discrete legend gives raw labels0…13 from the verified project LABEL_COLORS; label9 remains orange. No invented lithological names. Wells and coordinate decorations are omitted. The camera is uniformly centered lower than the target plate to fill the panels with geology rather than empty air; every nonair mesh point in all three FULL models is verified inside the viewport. Scene data remain the complete64³ volume.

上排完整多岩性实体，下排同一前角缺口剖切。统一使用Stage15低层体素外露面、色表和光照，不更改标签几何；所有非空气岩性不透明。相机只按图统一调整，没有按每个模型的目标bbox放大或隐藏误报。色标数字为raw岩性标签；原井/地表条件在模型中保留但不叠加密集井柱。干净版仅保留(a)…(f)，没有列名或色标。

## Figure B / 图B

The FULL raw9 body in each model is shown as opaque #D95F02 voxel surfaces. All predicted components, including remote FP fragments, remain; there is no target cutaway, truth ghost, outline, ROI restriction, small-component filtering, hole filling or error recolouring. One common oblique orthographic camera frames the complete0…64 cube, including original nine wells rendered identically in thin #168A89 tubes (radius0.12 voxel) and a faint gray cube. Lines are original condition-position references, NOT512 pseudo-wells, and do not change colour according to truth hits. No surface or non-target opacity fog is added. Clean version has only(a)…(c).

三列分别只显示各自完整目标体；预测列不叠真值。保持单行三格，与Stage15三面板风格一致；青色细线是同一组原9井位置参照，浅灰盒为完整64³空间参照。所有碎片、漏失和过大部分如实展示。目标体的阶梯面与明暗来自真实体素几何和同一光照。

## Saved evidence, kept OUTSIDE figures / 外置数值与限制

Saved fixed E_holdout all-class mIoU: A=0.1481, Stage23=0.5039. On E_original_free, A target P/R/IoU=0.008743/0.023081/0.006382; Stage23=0.217061/0.412579/0.165816. Volume ratios are2.63993 and1.90075. Displayed geometry is FULL volume, while these target metrics are in the fixed original-unknown domain.

On identical added_mask, old smooth versus restored pseudo-well accuracy is60.99%→92.44%; target FN211→7, FP0→1. Hidden object1 common subset outside the Stage23 condition mask is510 voxels: A=0, COPY=0, C=0, Stage23=144 correct target voxels.144/510 is coverage inside that truth-object subset, not object IoU or global recall.

Direct full restored-field classification remains much better: original-free target IoU0.967723, mIoU0.662712; it uses full-field information whereas Flow receives sampled wells. The result still misses/splits target bodies and has FP, so it is not accurate complete reconstruction. Known-kernel/no-extra-noise/artificial-surrogate/reused-case evidence does not establish real geophysical inversion, cross-case robustness or the paper's total objective. Holdout locations were not added as conditions but contributed to the global property surrogate; they are not independent observations/geological TEST.

视觉参照为实际Stage15两张PNG及本地Synthetic Geology论文图7/8（DOI10.1029/2025JH000986）；不复制旧任务ghost、粉色命中井、多实现阵列或论文原图。v1保留不覆盖。最终PNG需人工打开对照参考核验，配置和输入身份见同目录JSON。
