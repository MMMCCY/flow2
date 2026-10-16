"""One fixed slice and a Chinese report with three separate interpretation levels."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scripts.stage23.common import read_json


def fmt(v):
    return 'NA' if v is None else f'{v:.6f}'


def report(root,result,predictions,truth):
    if (root/'REPORT.md').exists():
        raise FileExistsError('Immutable report')
    figures=root/'figures'
    figures.mkdir()
    panels=[('Truth',truth),('Restored direct',predictions['RESTORED_CLASSIFICATION']),
        ('A original 9',predictions['A_9']),('Old C full pseudo',predictions['C_512']),
        ('Stage22 sparse target',predictions['FLOW_STAGE22']),('Stage23 full pseudo',predictions['FLOW_STAGE23'])]
    cmap=ListedColormap(['white',*plt.cm.tab20(np.linspace(0,1,14))])
    norm=BoundaryNorm(np.arange(-1.5,14.5),15)
    fig,axes=plt.subplots(2,6,figsize=(18,6.5),layout='constrained')
    for j,(title,volume) in enumerate(panels):
        im=axes[0,j].imshow(volume[:,:,15].T,origin='lower',interpolation='nearest',cmap=cmap,norm=norm)
        axes[1,j].imshow((volume[:,:,15]==9).T,origin='lower',interpolation='nearest',cmap='Reds',vmin=0,vmax=1)
        axes[0,j].set_title(title,fontsize=10)
        axes[1,j].set_title('Target raw9',fontsize=9)
        for ax in axes[:,j]:
            ax.set(xticks=[0,32,63],yticks=[0,32,63],xlabel='X',ylabel='Y')
    fig.colorbar(im,ax=list(axes[0]),ticks=range(-1,14),label='Raw lithology',shrink=.9)
    fig.suptitle('case03 seed42 | full XY, fixed Z=15 | retrospective display')
    fig.savefig(figures/'fixed_object_slice.png',dpi=160)
    plt.close(fig)
    samples={s['arm']:s for s in result['samples']}
    seal=read_json(root/'CONDITION_RECEIPT.json')
    freeze=read_json(root/'METHOD_FREEZE.json')
    receipt=read_json(root/'predictions/seed42/receipt.json')
    quality={q['arm']:q for q in result['conversion']}
    nq,cq=quality['STAGE23_WELLS'],quality['C_512']
    off={r['arm']:r for r in result['object1_common_subset']}
    lines=['# Stage23：恢复属性 → 完整多岩性伪井 → 冻结 Flow','',
        '本轮已完成并停止：只使用已见 fullgeo_case03、源 seed42、旧总井布局512；新增1份Flow预测、实际32次速度前向。没有训练、新去卷积、参数扫描、后处理修剪或其他seed/case。','',
        f"初始提交 `{freeze['git']['head']}`，工作区状态见 METHOD_FREEZE。旧实验、旧stage21v2/stage22源码与模型保持只读；未commit/push。",'',
        f"与Stage22相比，本轮取消 pred==9、d9≤0.01、margin≥0.05 的目标专用筛选，在原512总井布局的全部非原条件深度写入恢复分类。新增{seal['added_voxels']}体素、{seal['added_columns']}列；512含原9井，不是新增512井。背景及所有其他预测岩性均保留，原井/地表（含硬空气）优先，mask外为−1。新mask逐元素等于旧B512/C512；holdout新增0。",'',
        '直接复用R22 classification张量，CPU一次重算仅核对原FP32分类一致性，未求解恢复场。码本、双通道权重、非空气max-min范围、raw0…13候选与argmin并列规则不变。候选不含空气，未用真值空气mask修正新增井。','',
        f"方法冻结 {freeze['frozen_at_utc']}；条件封存 {seal['sealed_at_utc']}；Flow开始 {receipt['started_at_utc']}、完成 {receipt['completed_at_utc']}；首次完整真值评价 {result['evaluation_started_at_utc']}。准备与推理有文件读取防火墙；封存前未读取完整真值/对象/q9，科学精度不用于决定是否运行。",'',
        f"推理复用原sample_native与32步中点时间Euler，411个EMA权重及原checkpoint冻结embedding，全部参数冻结、FP32无autocast/TF32。实际A噪声tensor SHA256 `{receipt['initial_noise_tensor_sha256']}`；模型before/after等于旧A `{receipt['base_hash_after']}`。原条件/新增条件违例均0。含加载耗时约{receipt['elapsed_seconds']:.2f}s（计时来源见receipt）。",'',
        '工程重试已保留：v1防火墙误拦截Python依赖sympy.geometry，模型初始化前失败、前向0次、无预测。仅把geometry限制收窄为experiments内资产；v1日志/receipt/源码快照不覆盖，v2重新封存相同科学条件。合计仍是1份新Flow、32前向。','',
        'v2采样成功后，save_tensor收到相对路径导致收据序列化失败；已保存的预测和32步trace由CPU脚本核验后恢复封存，无追加前向。失败收据和日志保留。after模型hash根据失败点之前已通过的运行时before/after与逐项相等断言恢复，未伪称重新测量；耗时由trace文件时间减持久化开始时间近似重建。正式执行源码见source_snapshot；交付源码随后修复入口路径resolve并增加无网络回归测试，差异见POST_RUN_CODE_FIX。','',
        '## 1. 转换改善（只在相同 added_mask 上）','',
        '|标签来源|体素数|错误数|准确率|真目标错分/FN|真非目标错分|目标TP|目标FP|目标precision|目标recall|目标IoU|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for q in result['conversion']:
        lines.append(f"|{q['arm']}|{q['voxels']}|{q['errors']}|{fmt(q['accuracy'])}|{q['truth_target_errors']}|{q['truth_non_target_errors']}|{q['target_tp']}|{q['target_fp']}|{fmt(q['target_precision'])}|{fmt(q['target_recall'])}|{fmt(q['target_iou'])}|")
    lines += ['',f"本轮−旧C：全岩性错误 {nq['errors']-cq['errors']:+d}，真目标漏失 {nq['target_fn']-cq['target_fn']:+d}，真非目标错分 {nq['truth_non_target_errors']-cq['truth_non_target_errors']:+d}；目标FP {nq['target_fp']-cq['target_fp']:+d}。该域真空气体素数{nq['truth_air_voxels']}，全部保留；未依据真值删除坏井。",'',
        '两份confusion CSV的行是真值、列是预测，包含raw−1…13；condition_class_metrics.csv给出每类TP/FP/FN、P/R/IoU。不能将新增条件位置的正确率当作井外生成精度。','',
        '## 2. 建模收益（两个固定域及共同井外对象子集）','',
        'E_holdout与E_original_free及各自类别集合完全沿用R21。后者保留新增伪井体素，未扣除新条件后冒用原域名称。holdout未直接写入条件，但其信息参与全局属性代理，不能称为独立观测留出或未知地质TEST。','',
        '|臂|域|TP|FP|FN|precision|recall|IoU|固定类mIoU|预测/真值体积|体积比|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|']
    for s in result['samples']:
        for d,m in s['metrics'].items():
            lines.append(f"|{s['arm']}|{d}|{m['target_tp']}|{m['target_fp']}|{m['target_fn']}|{fmt(m['target_precision'])}|{fmt(m['target_recall'])}|{fmt(m['target_iou'])}|{fmt(m['fixed_class_miou'])}|{m['target_pred_volume']}/{m['target_truth_volume']}|{fmt(m['target_pred_truth_volume_ratio'])}|")
    lines += ['', '所有旧Flow和恢复场直接结果均复用保存资产；只有COPY_ONLY在旧A上写入本轮条件，无网络，holdout与A逐元素相同。B512为同mask准确真井参考，直接恢复分类使用全空间信息；它们不是等信息预算的竞争臂。Stage22仅稀疏正类，其mask不同，本轮比较是完整流水线诊断，不能分离各因素因果作用。','',
        f"对象1完整572体素，原井/地表零命中。共同子集固定为 truthID1 ∩ E_original_free ∩ ~Stage23_new_mask，分母{off['FLOW_STAGE23']['voxels']}；所有臂使用同一子集：",'',
        '|臂|共同井外TP|共同分母|覆盖率|','|---|---:|---:|---:|']
    for r in result['object1_common_subset']:
        lines.append(f"|{r['arm']}|{r['tp']}|{r['voxels']}|{fmt(r['coverage'])}|")
    lines += ['',f"预定建模判断逐项：`{result['modeling_checks']}`。全部满足：**{result['modeling_all_criteria']}**。若有指标冲突，只能解释为部分收益。对象1井外TP相对A/COPY分别变化{off['FLOW_STAGE23']['tp']-off['A_9']['tp']:+d}/{off['FLOW_STAGE23']['tp']-off['COPY_ONLY_STAGE23']['tp']:+d}。",'',
        '## 3. 范围控制与对象匹配','',
        '完整体积先做26连通ID，再按固定域进行旧一对一最大IoU匹配和merge/split判定。下表保留两个真值对象；匹配0不等于任意预测覆盖0。高覆盖也不能抵消巨大FP。两域完整结果见object_metrics.csv。','',
        '|臂|真对象ID|完整体素|匹配IoU|匹配覆盖|任意预测覆盖|','|---|---:|---:|---:|---:|---:|']
    for s in result['samples']:
        for o in s['objects']['E_original_free']:
            lines.append(f"|{s['arm']}|{o['object_id']}|{572 if o['object_id']==1 else 1164}|{fmt(o['matched_iou'])}|{fmt(o['matched_coverage'])}|{fmt(o['any_prediction_coverage'])}|")
    lines += ['', '主要预测分量（每臂取原未知域体积最大的3个；全部分量保存在predicted_object_metrics.csv）：','',
        '|臂|分量ID|完整体积|原未知域体积|原未知域FP|显著重叠真对象数|','|---|---:|---:|---:|---:|---:|']
    with (root/'predicted_object_metrics.csv').open() as stream:
        components=list(csv.DictReader(stream))
    for arm in samples:
        selected=sorted([r for r in components if r['arm']==arm and r['domain']=='E_original_free'],key=lambda r:int(r['common_voxels']),reverse=True)[:3]
        for p in selected:
            lines.append(f"|{arm}|{p['predicted_object_id']}|{p['full_voxels']}|{p['common_voxels']}|{p['false_positive_common_voxels']}|{p['significant_truth_overlaps']}|")
    n=samples['FLOW_STAGE23']['metrics']['E_original_free']
    s22=samples['FLOW_STAGE22']['metrics']['E_original_free']
    lines += ['',f"相比Stage22，原未知域precision {fmt(s22['target_precision'])} → {fmt(n['target_precision'])}；绝对相对体积误差 {fmt(s22['target_absolute_relative_volume_error'])} → {fmt(n['target_absolute_relative_volume_error'])}；合并分量数 {s22['merge_predicted_object_count']} → {n['merge_predicted_object_count']}；分裂真对象数 {s22['split_truth_object_count']} → {n['split_truth_object_count']}。是否绝对可接受须结合上述FP、体积与对象结果，不能仅凭优于Stage22判成功。",'',
        '本轮−参考的原未知域差值：','',
        '|参考|Δprecision|Δrecall|ΔIoU|ΔmIoU|ΔFP|Δ体积比|','|---|---:|---:|---:|---:|---:|---:|']
    for arm in ('A_9','B_512','C_512','FLOW_STAGE22','RESTORED_CLASSIFICATION'):
        b=samples[arm]['metrics']['E_original_free']
        lines.append(f"|{arm}|{n['target_precision']-b['target_precision']:+.6f}|{n['target_recall']-b['target_recall']:+.6f}|{n['target_iou']-b['target_iou']:+.6f}|{n['fixed_class_miou']-b['fixed_class_miou']:+.6f}|{n['target_fp']-b['target_fp']:+d}|{n['target_pred_truth_volume_ratio']-b['target_pred_truth_volume_ratio']:+.6f}|")
    lines += ['', '![固定回顾性截面](figures/fixed_object_slice.png)','',
        '图为沿用此前对象展示的Z=15全64×64 XY截面，统一raw−1…13色标；第二行为统一目标二值色标。未为本轮结果重新搜索截面。','',
        '已达到本轮唯一预测预算，立即停止。三层判断分别据转换错误、井外生成收益、误报/范围控制讨论，不合并成单一宽松“通过”。仍需区分属性转换质量与冻结Flow几何质量；本轮没有多seed重复支持。','',
        '仅为已知核、无额外噪声、人工属性对比、已见案例的路线诊断；不证明真实地球物理反演、跨模型泛化或论文创新性。后续由用户与MCYR决定，本轮不追加实验。','',
        '交付：PROTOCOL / METHOD_FREEZE / CONDITION_RECEIPT / inference_registry保存方法、输入、代码与条件身份；COMMANDS和TEST_RESULTS记录命令与必要检查；EXECUTION保存预算；sample_metrics.csv、class_metrics.csv、condition_quality.csv、两份confusion CSV、对象表和本图保存评价；predictions/seed42/decoded_geology.pt及receipt保存唯一新Flow。','']
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
