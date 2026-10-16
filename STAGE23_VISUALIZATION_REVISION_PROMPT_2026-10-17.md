# Stage23图件重绘：以实际参考图为准，图像为主、极少文字

本提示词替代先前 `STAGE23_VISUALIZATION_PROMPT_2026-10-17.md` 的版式、注释和补图要求。科学数据与资产身份不变。

---

请重绘Stage23结果图。用户不满意现有图片，要求接近仓库Stage15和原论文的三维岩体视觉效果，图内不放过多文字。此次只改可视化，不改实验，不重新推理。

## 先直接看参考图和现有成图

先在服务器定位实际项目根目录P（通常仓库内`project/geodata-3d-conditional`），读适用AGENTS.md，打开以下PNG，不能只读取绘图源码后声称已查看：

1. `P/experiments/stage15_binary_seismic_consensus/reports/conference_paper_v1/figure2_complete_label9_flow_results.png`
2. 同目录 `figure2_five_body_geophysical_guidance.png`
3. 先前本任务生成的Stage23主图，通常在 `P/experiments/stage23_full_pseudo_wells/reports/conference_paper_v1/`，确认实际路径。

原论文 *Synthetic Geology*，DOI `10.1029/2025JH000986`，图7位于PDF第12页，图8位于第13页上部。能找到PDF则渲染查看。已确认的视觉特征：

- 论文图7：三列，上排实心完整多岩性立方模型，下排同视角缺口剖切；背景白，体素块面清晰，模型占据主要画幅，单个共享岩性图例。
- 论文图8：目标体实心，周围地质背景透明，轻细空间参照，无性能指标；原图的多样本排布不适用于本轮单seed。
- Stage15 complete-label9图：完整橙色目标体，体素外表面不平滑，深浅面体现3D，细青色井线，浅灰立方框，无坐标刻度与指标表；每格只有很短的标题。
- Stage15 five-body图：一行三格，几乎整个画幅都用于模型，白底、统一斜视相机和尺度。其灰色真值ghost、粉色命中井是旧任务特定叠加，本轮不直接照搬。

请把这些实际图像作为视觉验收依据，不再自由设计报告型图板。先比较现图与参考在相机、体素几何、透明度、主体占比、留白和文字上的差别，再实施修正。现图未同步时找服务器原件；若确实不存在，如实说明，不凭空描述缺陷。

## 仅交付两张主图

三列顺序固定：真值、原9井冻结Flow A、Stage23改进后冻结Flow。

图内允许文字仅有：

- 顶部一次短列名：`Ground truth`、`Frozen Flow`、`Ours`。
- 左上角小面板号(a)、(b)…。
- 整体模型必要的一个共享离散岩性色标；可在完整带图例版保留，同时导出无图例干净版。

不要在图里放IoU、precision、recall、体积比、seed、case、方法长名称、结论句、对象体积、箭头、文本框、表格、柱图、流程图或大标题。不要生成上一版的证据补图。方法说明、指标、限制、图例解释全部写入独立`CAPTIONS.md`，不烧进PNG/PDF。不要把长图注另放在图像底部来规避此要求。

### 图A：整体模型，严格参照论文图7的2×3结构

上排：真值、A、Stage23的完整多岩性体；下排：相同三体的统一缺口剖切。

所有非空气岩性在整体图中不透明；不要用半透明整块岩体或体渲染雾效替代明确岩性块面。使用体素单元的外露面，平面明暗自然、边缘利落，不对标签体做平滑/重采样，不生成圆滑blob。

下排使用同一剖切keep-mask，例如复用原成熟cutaway_mask和cut_fraction=0.52，三个模型完全一致。只为显示去掉同一个前侧角块，暴露内部横纵剖面；这不是只显示三张二维切片，也不随机切碎模型。上排cutaway=False，下排True。

统一轴顺序、原标签颜色表、相机、尺度、光照、实体透明度与模型框。使用项目已验证的`LABEL_COLORS`并保留target9橙色。不能某列使用论文色表而另一列使用Stage15色表。若用户后续要求全套论文色表再整体切换，本轮不同时设计多套配色。

3列图像对齐，上下行留白紧凑。建议整图180mm左右宽；各格采用接近方形的同分辨率3D场景，立方体占面板约75–90%，同时不能裁掉边界。三维立方体边长在各列像素上相同，不按各臂内容自动适配。

不要画512根伪井，不在整体图上叠加密集条件。整体块图可省略井线与大坐标框，只保留可选的小方向三轴，避免遮盖结构。图注解释条件差异。

### 图B：目标体，一行三格，直接贴近Stage15 three-panel风格

仅显示各自完整raw label9的体素外表面，采用Stage15橙色 `#D95F02`，不透明。白底、浅灰细线立方体框，原9井用相同细青色线条（`#168A89`），无坐标数字、地面网格或黑色粗框。

真值列显示真值label9；A列只显示A的label9；Ours列只显示Stage23的label9。不能在预测列叠加真值ghost或真值轮廓，避免把真值误当成恢复结果。也不加入TP/FP/FN多色误差编码。只用一个目标颜色，让几何形状自己形成对比。

井线只作原9井位置参照，不能把512根井塞入Ours图，也不能根据真值是否命中将原井重新着色。可省略井线图例，在外置图注说明。井线过度遮挡时统一降低线宽，不能只在Ours删线。

完整预测分量全部保留，包括散落FP。不能截取真值ROI、只画最大分量、去小体、填洞、修边或让Ours的误报隐身。目标图不使用cutaway，目标岩体不能被成熟整体渲染函数默认切掉。

图B先只做一个统一斜视角，不强行增加第二行侧视来扩大画幅。三格同一完整64³空间盒，同一正交相机，同一缩放；空缺和细碎部分如实保留。不能按每臂目标bbox自动居中或单独放大。

## 复用正确渲染器，不盲用默认参数

读取实际Stage15参考图的生成脚本并追踪调用。可用 `scripts/paper_figures/style.py` 的低层体素网格、binary_surface、相机、光照和边框；不要另写Matplotlib mplot3d.voxels，造成渲染质感差异。

现有已核对风格参数可作起点：

```
camera.position_direction = [1.72, -1.35, 1.38]
camera.focal_point_fraction = [0.50, 0.50, 0.46]
camera.view_up = [0, 0, 1]
parallel_projection = True
parallel_scale = 0.70 * 64
zoom = 1.14
smooth_shading = False
show_edges = False
ambient = 0.34
diffuse = 0.66
specular = 0
```

PyVista `ImageData`采用真实体素cell、threshold后extract_surface，保留阶梯面，不用marching-cubes平滑形状；用SSAA改善锯齿。阅读grid赋值轴顺序，项目代码使用Fortran ravel匹配XYZ cell，不得随意转置或翻Z轴。

注意默认陷阱：`render_categorical_volume_3d`默认半透明且cutaway=True，整体上排必须显式改为不透明且不剖切；`render_target_constraints_panel`会叠加灰色真值ghost，本轮目标图不能直接原样调用。创建独立薄封装调用低层工具，不修改旧共享style以免影响历史图。

上述camera是在参考图基础上的起点，不是任意面板都保证不截边。允许统一微调留白、zoom或相机以匹配实际参考图；不得逐方法调视角。完成后相机数值写入配置。

## 数据、输出与预算

参考提交 `90ca753f0413f48aa57e989ba41f327ae98573b7`。

真值与A从旧Stage21v2 registry/receipt解析：

`P/experiments/stage21v2_dense_virtual_boreholes/run_20261016_v1`

改进输出只用有效Stage23 v2、seed42：

`P/experiments/stage23_full_pseudo_wells/run_20261017_v2`

校验现有张量身份和标签轴，CPU只读；不加载网络、训练、推理、去卷积或再生成条件。不把直接属性分类结果冒充Stage23 Flow输出。

单独写绘图脚本，不在封存stage21v2/22/23源码目录加py；不修改共享style和旧结果。新输出用 `P/experiments/stage23_full_pseudo_wells/reports/conference_paper_v2/`，保留不满意的v1用于对比，不覆盖。

交付两主图的高清PNG/PDF（300–600dpi），每张另有仅面板号的干净版本，便于用户后续排版。其余只是外置图注、相机色表配置、输入资产身份和重绘命令。不额外做报告海报或指标图。

必须直接打开最终成图，与两张Stage15参考图和论文图7/8并排检查，确认：模型占画幅足够、无大块无效留白、体素感/光照一致、无灰雾和错误ghost、三列镜头一致、所有预测分量保留、图内无指标或长文本。脚本跑完不算视觉验收完成。若只修文字却保留明显不同的渲染器或相机，也不算完成。

最后用简短文字说明改动并给出两张预览和文件路径。不增加科研实验或对科学方法做调整。
