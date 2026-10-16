# Stage24 captions / 外置中英文图注

Ground truth / Frozen Flow / Ours均指完整64³标签体。Frozen Flow为原9井与地表条件的旧A，Ours为相同冻结网络接受恢复属性完整伪井；未重训。整体图为完整/同一缺口，目标图保留全部raw9分量，青色细线为原9井、灰框为0…64体素坐标范围，非米或海拔。相同approved_v1色表、镜头、尺度和照明。PDF三维为栅格，文字/图例为矢量。

Each case uses source seed42, a synthetic property surrogate with known Gaussian kernel and a frozen network. The complete model/target is shown; reported metrics use the fixed registered E_original_free or E_holdout domains. No target ROI/component filtering, geometric repairs, smoothing or truth ghost. All FP fragments remain. One realization per reused case does not establish held-out generalization.

## fullgeo_case01, seed42

Holdout target IoU: C 0.014286 → Ours 0.160000; holdout all-class mIoU 0.367135 → 0.519594. Original-free Ours P/R/IoU 0.101727/0.449153/0.090444; target volume ratio 4.415254. Direct restored FULL-field classification IoU 1.000000, mIoU 0.799359; unlike Flow it uses full-field information.

case01：major目标原井已命中，不能称为新零井主要隐藏体恢复。

## fullgeo_case02, seed42

Holdout target IoU: C 0.396172 → Ours 0.464922; holdout all-class mIoU 0.333983 → 0.482250. Original-free Ours P/R/IoU 0.527321/0.848000/0.481783; target volume ratio 1.608127. Direct restored FULL-field classification IoU 0.998069, mIoU 0.735885; unlike Flow it uses full-field information.

case02：所有零井目标对象及失败对象见REPORT与对象表，不挑选最好分量。

## fullgeo_case03, seed42

Holdout target IoU: C 0.022059 → Ours 0.130112; holdout all-class mIoU 0.288156 → 0.503877. Original-free Ours P/R/IoU 0.217061/0.412579/0.165816; target volume ratio 1.900750. Direct restored FULL-field classification IoU 0.967723, mIoU 0.662712; unlike Flow it uses full-field information.

case03：既有认可结果复用，新Flow/恢复次数均0；共同510井外对象体素恢复144，非IoU。
