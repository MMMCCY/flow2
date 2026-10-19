"""Chinese report from sealed measurements; no model or inversion computation."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage26.common import *

def n(value):return 'NA' if value is None else f'{value:.6f}'
def main():
    e=read_json(ROOT/'EVALUATION.json');inv=read_json(ROOT/'INVERSION_STATE.json');seal=read_json(ROOT/'CONDITIONS_FROZEN.json');render=read_json(ROOT/'figures/RENDER_CONFIG.json')
    assert render['visual_inspection']=='completed'
    if (ROOT/'REPORT.md').exists():raise FileExistsError('Completed report is immutable')
    shape={r['arm']:r for r in e['shape']};metrics={(r['arm'],r['domain']):r for r in e['metrics']};off={r['arm']:r for r in e['outside']};phys={r['arm']:r for r in e['physical_errors']};quality={r['arm']:r for r in e['condition_quality']};trans=e['condition_transitions']
    a,d0,d=(shape[k] for k in ('A','D0','D'))
    rows=['# Stage26：合成逐柱叠后卷积地震及模型基础线性化阻抗更新','',
        '**本案例未形成超过A及D0的井外形态改善。** 连续阻抗更新降低了地震数据误差，但分类伪井质量下降，随后冻结Flow D的整体目标IoU和井外覆盖低于A及D0，最终地质查表正演误差反而增加。这是完整交付的科学负结果，不是沿用Stage25的保存故障。','',
        '## 配对设计与信息隔离','',
        '复用Stage25唯一原生Laccolith（帽部+向下供给柱，2217目标体素）及旧A/B/C、原9井/地表、512布局、256holdout、同一15维CPU seed42源。没有新几何、噪声、seed或物性/正则/频率扫描，没有重跑A/B/C。原9井只有中心(35,26)命中，但A全目标质心误差为'+n(metrics['A','FULL']['centroid_error_m'])+'米，不是正确定位的理想物性先验。','',
        'D0将本次prior_exact的log阻抗直接分类为完整多岩性512井；D将一次观测驱动更新后的log阻抗按完全相同规则分类。两者原井/地表精确物性投影和地下空气处理相同，只改变是否应用阻抗更新。raw0…13完整非空气码本、log阻抗最近值、first argmin；地表定义外设空气，原条件优先。不把固定慢度当第二证据通道。两套分类场和条件均先封存再评价。','',
        f'D0/D条件不同：新增mask上{seal["added_labels_changed"]}体素标签改变（完整分类场改变{seal["classification_changed_voxels"]}体素）；因此分别运行两份正式预测。mask与旧B/C完全相同，holdout无新增写入。411EMA、冻结checkpoint embedding、32步原sample_native中点Euler、FP32/noTF32/noautocast和已保存源身份一致，条件违例0。','',
        '反演独立进程仅接收A、原井/地表、旧地震观测和固定配置；文件防火墙禁止真值、几何ROI、旧B/C、Gaussian观测/恢复场，允许本次自身输出写入及哈希/加载读回。Flow只接收已封存条件与源噪声。完整真值及预设帽部区只在D0/D预测固定后用于评价。','',
        '## 物理处理与真实数据误差','',
        '观测张量和三份物理配置逐字节复用旧Stage25 physics资产；核对其原始来源和当前算子hash，没有重新正演生成观测，也没有伪造旧包装层source hash。空间XYZ、Z向上，spacing=3840/63='+f'{SPACING:.9f}'+'米；全64×64逐柱、8ms/400样点、25Hz零相位Ricker/128ms、无额外噪声，记录到3192ms。记录窗由已知完整网格深度和码本最低岩石速度预设，观测正演本身允许使用真值速度；反演没有读取该速度。','',
        '复用invert_acoustic_member：先从A计算非线性反射/走时/卷积响应，再对残差解一次线性化log阻抗Tikhonov修正；G=W(0.5D)，正则0.001/0.01及原mean-diagonal尺度、差分和边界定义不变。直接Cholesky解一次，并非多次非线性优化。时深映射与更新前后慢度始终来自A，不估计新速度。A地下空气确定性替换数为'+str(inv['cleanup']['underground_air_voxels_replaced'])+'。旧fixed12兼容字段仅供parser校验，本次成员数1，不计算后验均值/方差/置信度。','',
        '方法族概念核对：[PyLops叠后教程](https://pylops.readthedocs.io/en/latest/tutorials/poststack.html)、[PoststackInversion文档](https://pylops.readthedocs.io/en/latest/api/generated/pylops.avo.poststack.PoststackInversion.html)。本轮运行仓库既有实现，没有采用教程中的真值平滑初始模型。','',
        '所有误差统一为完整1638400个数据样点上的 relative L2 = ‖d_pred−d_obs‖₂/‖d_obs‖₂；没有借真值慢度对齐事件。','',
        '|输出|相对L2|慢度/速度定义|','|---|---:|---|']
    for key in ('A_prior','continuous_updated','DIRECT_PRIOR','DIRECT_UPDATED','D0','D'):
        row=phys[key];rows.append(f'|{key}|{n(row["relative_l2"])}|'+('固定A慢度' if key in ('A_prior','continuous_updated') else '各自硬岩性重新查表阻抗和慢度')+'|')
    rows += ['',f'连续反演实际拟合从{n(phys["A_prior"]["relative_l2"])}降至{n(phys["continuous_updated"]["relative_l2"])}，但仍有较大残差。直接updated分类再正演误差{n(phys["DIRECT_UPDATED"]["relative_l2"])}，最终Flow D则升至{n(phys["D"]["relative_l2"])}；连续物性拟合改善没有保留到最终地质。D0正演误差{n(phys["D0"]["relative_l2"])}也不优于A先验。','',
        'D生成了2246个公开地表以下的空气标签，直接分类和D0均为0。按要求保留原始预测、用完整raw−1…13声学表重新正演，不在评价时修补空气或借真值清理。D共有218/212992个有效界面的到时晚于3192ms，最大4364.32ms；沿用既有有限记录窗裁剪预测到时的规则。其余D0/直接场无超窗到时。以上误差均对应同一完整已观测时间窗，不是全时间轴传播误差；详见physical_window_diagnostics.csv。地下空气和其慢速度是生成结果的物理缺陷，不修改时间窗或标签来改善分数。','',
        '## 同added_mask条件质量','',
        '|条件|全类准确率|错误数|目标TP/FP/FN|目标P/R/IoU|','|---|---:|---:|---|---|']
    for key in ('D0','D'):
        q=quality[key];rows.append(f'|{key}|{n(q["accuracy"])}|{q["errors"]}|{q["target_tp"]}/{q["target_fp"]}/{q["target_fn"]}|{n(q["target_precision"])}/{n(q["target_recall"])}/{n(q["target_iou"])}|')
    rows += ['',f'相同26156个新增体素中，D相对D0改变{trans["changed_added_voxels"]}个标签：错→对{trans["wrong_to_right"]}、对→错{trans["right_to_wrong"]}、错→其他错误{trans["wrong_to_wrong"]}。错误净增加{quality["D"]["errors"]-quality["D0"]["errors"]}。目标条件FP减少，precision/IoU提高，但TP减少、FN增加、recall降低；不能只用目标井上IoU提高概括转换成功。此分类问题已经发生在Flow之前。','',
        '## 完整体积形态','',
        '|输出|目标IoU|边界均距/米|体积比|FP|帽部区IoU|','|---|---:|---:|---:|---:|---:|']
    for key in ('A','D0','D','B','C','DIRECT_PRIOR','DIRECT_UPDATED','COPY_ONLY_D'):
        s=shape[key];rows.append(f'|{key}|{n(s["target_iou"])}|{n(s["assd_m"])}|{n(s["volume_ratio"])}|{s["target_fp"]}|{n(s["cap_box_iou"])}|')
    rows += ['',
        '边界距离完全复用Stage25：完整目标的6邻接边界体素、双向最近距离样本合并均值、实际spacing、全部预测分量参与，空预测记NA。均值小于一个网格间距不表示每处边界达到亚网格精度。帽部区预先按Stage25生成参数冻结，只作局部解释，不删除全体积FP。','',
        f'D整体IoU低于A和D0；边界均距{n(d["assd_m"])}米比D0的{n(d0["assd_m"])}米小，但仍大于A的{n(a["assd_m"])}米，且井外覆盖下降。D目标体积比{n(d["volume_ratio"])}、FP减少伴随明显漏失，不能称整体恢复改善。D0略增加覆盖，却增大FP和体积、降低IoU，表明把A自约束为密集井也不保证形态收益。','',
        '完整updated分类IoU0.139236、边界均距327.70米，优于D，但目标recall从A的0.249887降至0.195760，帽部区IoU也下降；这是减少误报/收缩目标与漏失的权衡，不是正确岩盖重建。直接场使用全场信息，Flow仅看到512井，二者不是等信息预算排名。','',
        '## 原未知域与固定holdout','',
        '|域|输出|TP/FP/FN|P/R/IoU|固定类mIoU|','|---|---|---|---|---:|']
    for domain in ('E_original_free','E_holdout'):
        for key in ('A','D0','D','DIRECT_UPDATED'):
            m=metrics[key,domain];rows.append(f'|{domain}|{key}|{m["target_tp"]}/{m["target_fp"]}/{m["target_fn"]}|{n(m["target_precision"])}/{n(m["target_recall"])}/{n(m["target_iou"])}|{n(m["fixed_class_miou"])}|')
    rows += ['',
        '域及各域真值存在的非空气类别集合原样复用Stage25；新增井位置未从原未知域排除。完整B/C/直接先验/COPY及质心误差见metrics.csv。holdout未被写入硬条件，但其上方/柱内地震观测参与更新，不能称独立观测泛化测试。','',
        '## 同一新增井外子集','',
        '所有臂统一使用 真目标 & 原未知域 & ~D新增mask，分母1933。','',
        '|输出|正确TP/分母|覆盖率|','|---|---|---:|']
    for key in ('A','D0','D','COPY_ONLY_D','DIRECT_UPDATED','B','C'):
        r=off[key];rows.append(f'|{key}|{r["tp"]}/{r["voxels"]}|{n(r["coverage"])}|')
    rows += ['',
        'COPY_ONLY_D只在A上写入D条件：holdout逐体素等于A，共同井外TP也等于A。D没有超过A或D0的新井外覆盖证据；这与4025处条件变化并不矛盾，变化本身不等于正确新信息。','',
        '## 机制判断、C参考与限制','',
        '物理观测确实影响了连续场和分类接口，并降低连续数据误差；但它未在本案例转化为超过A自约束的正确形态信息。伪井全类质量下降已显示反演场可分类性存在问题。固定不准确A的时深关系、单次线性化及单阻抗的类别歧义可能限制转换，当前单例不能唯一归因。D比全场updated分类进一步退化，并产生地下空气及更大的物理误差，说明井柱压缩和冻结模型对这些错误条件的利用还带来额外损失；不能只指责Flow或只指责物性转换。','',
        '旧C完整IoU0.786812、伪井准确率99.9541%，目标247/0/0；本次D条件只有59/135/188。B准确井仍可达IoU0.792091，说明该几何在高质量条件下可明显恢复；当前物理更新未提供同等级标签证据。已知Gaussian双属性代理接近完美的结果不能替代带反射系数、传播走时、带限记录和反演的物理链验证。','',
        '准确范围：合成逐柱叠后卷积地震及模型基础线性化阻抗更新。声学raw9密度3000kg/m³、Vp6500m/s为人工强对比；同算子生成/反演、已知波子、无额外噪声是inverse-crime理想设定。不含原始炮集处理、偏移成像、绕射/多路径、全三维波场或实测验证；也没有从数据估计新速度。本轮一个几何、一个seed，不做显著性或单seed置信区间。','',
        '## 执行与交付','',
        f'本轮成功保存反演1次（函数调用1次，耗时{inv["elapsed_seconds"]:.6f}秒）；没有再次Gaussian去卷积，没有重新生成观测。D0/D新Flow{e["new_Flow_outputs"]}份，共{e["velocity_evaluations"]}次速度前向，原A/B/C重跑0次。Stage25历史反演失败1次单独保留、不计作本轮成功。',
        *[f'{arm}采样/加载/冻结审计耗时'+f'{read_json(ROOT/f"arms/{arm}/predictions/seed42/receipt.json")["elapsed_seconds"]:.3f}'+'秒。' for arm in seal['canonical_arms']],
        '10项相关CPU测试通过；正式反演前实际小张量save_tensor写入→file_hash以rb读回→verify_tensor加载成功。原井/新增条件违例0，模型/源身份一致，D0/D及旧B/C mask一致，holdout无新增、COPY holdout=A，数值有限及误差算术通过。本轮没有反演/网络工程失败或重算；离屏VTK提示无X连接，但所有图成功保存并已实际打开核验。','',
        '数值优先原子保存于preprocessing/numerical_bundle.pt；若后续元数据步骤失败可加载它恢复，不重新反演。prior_exact、updated、模拟地震及两个直接分类单独保存；条件与预测在arms/D0和arms/D。旧观测原始来源和当前执行包装层的hash分开记录于INPUT_FREEZE.json及source_snapshot，未修改Stage25冻结记录或旧算子。','',
        '主图、目标图、固定Y26剖面见figures/fig01…fig03；D0独立目标体在figures/panels/target_D0，中心剖面fig04；同Y26时间域地震统一色标图fig05。均有PNG/PDF，高清独立面板及图外图注保留。条件质量/改变机制/完整地质/物理误差CSV均在本目录，命令与重绘见COMMANDS.md。','',
        '后续若研究者另行授权，优先讨论一个固定先验时深误差诊断，或同原井不同帽部形态对照；本轮不追加噪声、速度/正则扫描、新案例或训练。未推送GitHub。']
    (ROOT/'REPORT.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    write_json(ROOT/'COMPLETION.json',dict(status='complete',scientific_verdict='physical fit improved continuously, no off-well shape improvement over A and D0',inversion_calls=1,new_Flow_outputs=e['new_Flow_outputs'],velocity_evaluations=e['velocity_evaluations'],report_sha256=file_hash(ROOT/'REPORT.md'),completed_at_utc=utc()))
    print('STAGE26_COMPLETE',ROOT,flush=True)
if __name__=='__main__':main()
