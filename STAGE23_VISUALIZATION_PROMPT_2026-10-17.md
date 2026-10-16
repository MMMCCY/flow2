# Stage23正向结果可视化：完整提示词

可直接交给服务器VSCode Codex。任务是读取保存资产绘图，不开展新科学实验。

---

请为Synthetic Geology / Flow项目完成Stage23的科学结果可视化。使用实际三维标签体和保存指标生成图件，交付可复现绘图脚本、高清图片、可编辑文字图层的PDF及图注。不要只给方案，完成渲染并检查成图。

## 1. 科学结论和展示目标

本轮已有一个正向概念验证：在已知Gaussian核、无额外噪声、人工属性对比的case03/seed42中，先恢复属性再形成完整多岩性伪井，改善了冻结Flow对原井未命中对象的井外恢复，并缓解了只输入目标点时的过度膨胀。

不能称为“准确完整重建”“真实地球物理成功”“跨案例稳健”或“论文总目标已经完成”。目标仍漏失、分裂并有假阳性。图片重点展示正向变化，完整保留误差，不改变标签几何。

主图集中比较三列：

1. `Ground truth`
2. `Frozen Flow (9 wells)`：旧A_9，seed42
3. `Frozen Flow + restored-property pseudo-wells`：Stage23，seed42

第三列模型本身同样冻结，不得让标题暗示网络被重训；最终可使用两行短标题 `Frozen Flow + pseudo-wells` / `blur-aware preprocessing`，图注解释完整方法。

## 2. 仓库、资产及预算

服务器仓库通常 `/home/xmj/mcy/flowtrain_stochastic_interpolation-main`；P表示仓库内 `project/geodata-3d-conditional`，先确认实际结构、读适用AGENTS.md。

参考GitHub `MMMCCY/flow2` 提交 `90ca753f0413f48aa57e989ba41f327ae98573b7`。优先当前服务器实际封存资产，记录后续差异，不reset或覆盖用户工作。

有效结果：

`R23=P/experiments/stage23_full_pseudo_wells/run_20261017_v2`

旧对照：

`R21=P/experiments/stage21v2_dense_virtual_boreholes/run_20261016_v1`

属性恢复及直接分类：

`R22=P/experiments/stage22_blur_aware_bridge/run_20261016_v1`

通过registry、EXECUTION、receipt定位真值、A/B/C、Stage23、Stage22直接分类及条件mask；不要凭文件名猜样本。验证读取的tensor/file hash、case03/seed42和形状。大张量若GitHub无而服务器有，用服务器原件。不能从报告截图重建体素，不能重新推理。

本任务新增Flow前向=0、训练=0、去卷积求解=0、条件生成=0。只允许CPU读取与绘图；3D离屏渲染可以使用图形加速，它不是模型推理。不要加载checkpoint。

Stage23 v1没有有效预测，不能使用。v2经历保存收据恢复，按EXECUTION、completion_receipt及实际预测hash确认唯一输出；不要为了获取“干净收据”重跑模型。

## 3. 参考格式：先看图，再改绘图脚本

用户参考目录：

`P/experiments/stage15_binary_seismic_consensus/reports/conference_paper_v1`

优先查看 `figure2_complete_label9_flow_results.png`，可参考 `figure2_five_body_geophysical_guidance.png` 的布局。读取实际文件，不能只按文件名想象。

可复用：

- `P/scripts/reports/build_stage15_conference_figures.py`
- `P/scripts/paper_figures/style.py` 及相关成熟渲染工具

已知旧脚本有 `render_categorical_volume_3d`、`render_target_constraints_panel` 等函数。先读实现，确认是否默认加入真值ghost、裁剪、平滑、固定旧case井位或独立相机缩放，不可盲用。

原论文 *Synthetic Geology*，DOI `10.1029/2025JH000986`：

- 图7（PDF第12页）：上排完整多岩性模型，下排相同位置剖切，统一颜色和视角。
- 图8（PDF第13页上部）：透明背景中显示完整目标岩体和稀疏条件；它展示多随机实现。我们只有seed42，不仿造多实现阵列或概率图。

服务器能找到论文则直接核对；找不到无需阻塞，按上述已核实结构绘制。Stage15提供统一风格，图7/8提供完整体与目标分开展示的逻辑，不需要复制原文图片、页眉或面板数量。

## 4. 主图A：整体多岩性模型（2行×3列）

文件名建议 `fig01_full_geology_comparison`。

列为真值、A、Stage23。上排完整64³多岩性体，下排同一剖切几何。整体约180mm宽、125–140mm高，三列模型等大；不要大画布配小字体。

统一要求：

- 相同raw标签颜色表、离散图例、背景、光照、投影、相机、焦点、缩放和坐标轴。沿用Stage15/原论文岩性色系，目标类9突出但不能在各方法中换色。无法验证岩性名称时用 `Lithology 1` 或 `raw label 1`，不要杜撰具体地质名称。
- 真实数组轴顺序须读现有渲染函数确认，不能把文档的ZYX示例套到项目张量；三轴同尺度。只有体素坐标依据时标 `X/Y/Z (voxel index)`，不虚构米或海拔。
- 全体积边界固定为64³，空气按项目约定隐藏；不能依各臂非空bbox自动缩放。
- 下排剖切位置三列一致。优先沿用成熟固定中部切块，如x/y索引32；若有理由使用对象展示的既定坐标，统一固定并在caption/CONFIG记录。剖切仅用于显示，不修改保存预测或评价。
- 主图不叠加512根井柱，避免遮住内部结构。图注写原9井/地表与额外完整伪井条件；所有面板共享的原条件可以省略渲染。若保留原9井细线，三列保持一致并解释它们是原条件。不能把512伪井画在baseline上。
- 不标不存在的ensemble概率，不以平滑表面替换岩性标签。可用体素表面提取，不能做改变体积/连通性的几何修补。

在A和Stage23列下各标一个简短指标：固定holdout全类mIoU，A=0.1481、Stage23=0.5039，必须从当前CSV读取核对，不硬编码未经核对值。真值列不写预测性能。

## 5. 主图B：目标岩体展示（2行×3列）

文件名建议 `fig02_target_body_comparison`。三列顺序同图A，上排完整label9目标体的统一等轴视角，下排同一目标体的统一正交侧视/背面视角，用于减轻遮挡。相机由真值与完整64³盒子统一确定，所有臂共用，不逐臂挑最佳角度。

目标岩体用不透明、同一种醒目颜色，非目标背景不显示或以相同极低透明度显示空间参照。边框和坐标盒固定，保持细碎FP及完整预测分量；不能只画与真值重叠的分量、最大分量、匹配对象或真值ROI内部分。不能把目标预测与真值ghost混画后叫“预测目标”。

真值的两个26连通目标对象可用统一文字/细引线标记 `Hidden object 1 (572 voxels)` 和 `Object 2`；预测列若需定位提示，只能使用相同坐标的小箭头，并说明是回顾性参照。不得以真值对象1的范围裁掉远处FP。

完整场景必须保留；不要为突出小目标缩小Stage23而放大baseline或反之。若目标物体正被背景遮挡，统一降低背景透明度或使用第二视角，不对目标做切块。

文字指标以原未知域为主，图注说明显示完整模型，而指标在固定E_original_free上计算；如果配井外指标，则明确标E_holdout，不能把域混用。

需核对的原未知域值：

- A：目标P=0.008743、R=0.023081、IoU=0.006382，预测/真值体积比2.63993。
- Stage23：P=0.217061、R=0.412579、IoU=0.165816，体积比1.90075。

面板中只放IoU及P/R的一行短注，其他放caption或配套小表，避免挤满图。统一精度，百分数与小数不要混排。

重点解释不能只靠写入：对象1在本轮新增条件之外共同510体素，A/COPY/C=0，Stage23 TP=144、覆盖28.24%。可在整图下放一条醒目但克制的注释：`Hidden object: 144/510 previously unconstrained voxels recovered (baseline: 0/510)`，并明确这是该真值对象内覆盖，不是对象IoU或整个目标召回。预测对象1全部覆盖206/572、匹配IoU0.122137，不可误标成28.24% IoU。

## 6. 一张小型证据补图，避免只靠漂亮外观

生成 `figS01_bridge_evidence`，紧凑的两个面板即可：

- 左：旧平滑伪井与恢复属性伪井在同added_mask的全类正确率60.99%→92.44%，并标目标FN211→7、FP0→1。用简单柱图/数值表，不画多余装饰。
- 右：holdout目标IoU及全类mIoU的A、旧C、Stage23、准确井B对照。IoU=0.0081/0.0221/0.1301/0.1407；mIoU=0.1481/0.2882/0.5039/0.5289。从CSV读取，分类名称清楚。单样本没有误差棒。

补图或配套表必须保留恢复场直接分类的关键指标（原未知域目标IoU0.9677、mIoU0.6627），说明它使用完整恢复场而Flow只获得伪井采样，两者信息量不同。不要藏起直接分类更好的事实，也不要据此称Flow同信息条件下失败。

如180mm画幅无法清楚容纳上述表，将direct指标放在独立`figure_metrics.csv`和图注中，不把主图扩为密集九宫格。

## 7. 科学真实性与视觉质量

全程只读旧封存实验。不得按真值裁坏井/FP、滤掉小连通分量、填洞、插值新增类别、移动对象、平滑改边界或选择另一个seed。允许对显示进行统一抗锯齿和光照处理，不能改变语义体积。

两张主图主要用于病例比较，caption必须写明 `fullgeo_case03, seed42, synthetic property surrogate with known Gaussian kernel; frozen network`。图7/8只是表现形式参照，不借用其多实现/泛化结论。

使用PyVista/VTK等已有3D渲染结合Matplotlib排版；依赖尽量复用。禁止使用AI图像生成/编辑工具绘制研究结果。无离屏环境时先解决渲染依赖，不用手绘示意替代实际标签体。

正文最终排版后文字至少8pt，建议9–10pt；白底、细边框、紧凑留白，面板(a)…编号。字体可用现有Arial/DejaVu Sans，文字不遮挡体积。分类图例为离散色块而非连续数值色条；不使用无意义渐变。

输出每图300–600dpi PNG和PDF，3D可栅格嵌入PDF但文字/图例尽量矢量；不要把嵌入位图PDF宣称为全部矢量。附面板原始高分辨率渲染，方便后续重新排版。

## 8. 目录、交付与成图核验

新脚本放 `P/scripts/reports/` 独立文件或新的可视化目录，勿在封存stage21v2/stage22/stage23源目录增加py文件（旧source hash可能glob它们）。不覆盖Stage15图件。

输出建议：

`P/experiments/stage23_full_pseudo_wells/reports/conference_paper_v1/`

若已存在，先识别用途，不覆盖未知结果。保存：三张图的PNG/PDF、简短中英文CAPTIONS.md、figure_metrics.csv、RENDER_CONFIG.json（轴顺序、camera、切块、色表、透明度、物理/体素单位）、INPUT_ASSETS.json（张量身份）和一条重绘命令。无需新建庞大审计系统。

完成后必须打开PNG并检查：三列相机和比例一致、目标没有被误裁、图例无错位、文字可读、边界没有遮挡或截断、错误体素没有被渲染逻辑隐藏。原始图像读取与导出要区分，查看图不能只用“脚本无报错”替代。若某视角遮挡严重，统一调整三列；记录最终camera，而不逐臂美化。

指标从现有CSV读取并核对，不重算全部科学实验，只检查绘图使用资产和数值一致。发现缺失、hash不符或轴语义不确定时查代码和注册表，不凭外观猜测；确实无法确定则报告具体缺什么。

最终说明：图展示了哪些正向证据，哪些残余漏失/FP依然可见；给出图片、PDF、脚本和图注路径；明确新增模型前向、训练和属性恢复次数均为0。不得自动追加seed确认或其他研究工作。
