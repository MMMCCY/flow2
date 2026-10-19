"""Assemble Chinese report from saved scientific outputs, no inference."""
import sys,time
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage25.common import *

def num(v):return 'NA' if v is None else f'{v:.6f}'
def main():
    e=read_json(ROOT/'EVALUATION.json');s=read_json(ROOT/'SOLVER_STATE.json');g=read_json(ROOT/'GENERATION_CHECKS.json');f=read_json(ROOT/'figures/RENDER_CONFIG.json')
    assert f['visual_inspection']=='completed'
    shapes={r['arm']:r for r in e['shape']};metrics={(r['arm'],r['domain']):r for r in e['metrics']}
    a,b,c,d=(shapes[k] for k in ('A','B','C','DIRECT'))
    improved=c['target_iou']>a['target_iou'] and c['assd_m'] is not None and a['assd_m'] is not None and c['assd_m']<a['assd_m']
    out=['# Stage25：原生拱形岩盖受控形态实验\n',
        ('本案例C相对A的整体目标IoU提高、平均表面距离减小；仍需同时核对体积、误报和残余碎裂，不能称精准恢复。' if improved else '本案例未同时达到整体目标IoU提高和表面距离减小；各项收益与代价分别报告，不宣称精准形态恢复。'),
        '\n## 科学问题与固定方法\n',
        f'原9井中{g["hit_well_count"]}口命中目标，中心井(35,26)提供存在及位置提示；帽部侧缘未被其他原井约束。真值是原生Laccolith（帽部+向下供给柱），不是盐丘。目标为一个连通主体，共{g["target_voxels"]}体素；无侧边截断、无目标出露。背景raw0/1/3/5，目标raw9，地表约Z52。',
        f'每轴[-1920,1920]含端点64点，实际间距{SPACING:.9f}米。XY中心(35,26)，帽底Z28、名义顶Z36，帽直径22格、高8格、柱直径4.4格，轴比1。完整事件、参数、冻结帽部解释区见GEOMETRY.json。未搜索几何或seed、未根据网络输出调整形态。',
        'A为原9真井+地表；B为同原条件+准确512井；C为同布局恢复属性完整多岩性伪井。B/C mask完全相等，256固定holdout柱无新增硬条件；原条件永远优先。新增井不从原未知域排除。原生生成、观测构建、恢复、Flow、事后评价分进程，恢复/Flow文件防火墙禁止读取完整真值/几何/评价注册表，Flow另外禁止恢复场和观测。',
        '只生成一次新CPU seed42 Gaussian源；三臂完全配对。411EMA、冻结embedding、32步中点时间Euler、FP32/noTF32/noautocast与sample_native均复用既有成功实现。双属性Gaussian sigma1/radius3/replicate观测、有界TV1e-4恢复、FP64运算/原FP32核系数、全非空气码本归一化及raw0…13首argmin规则均保持不变。没有训练、adapter、新输入通道或梯度制导。',
        '\n## 整体形态（完整体积、全部分量）\n',
        '|臂|IoU|ASSD/米|体积比|目标体素|预测分量|帽部解释区IoU|\n|---|---:|---:|---:|---:|---:|---:|']
    for k in ('A','B','C','DIRECT','COPY_ONLY'):
        r=shapes[k];out.append(f'|{k}|{num(r["target_iou"])}|{num(r["assd_m"])}|{num(r["volume_ratio"])}|{r["predicted_voxels"]}|{r["predicted_components"]}|{num(r["cap_box_iou"])}|')
    out += ['\nASSD对完整预测与真值raw9的6邻接边界求双向最近距离，将两方向的全部距离样本合并取均值，使用实际spacing；所有分量参与，无匹配分量筛选。空预测记NA且注明未检出。帽部区由生成参数预先冻结，只作局部解释；整体FP/体积未被裁除。',
        f'A整体IoU={num(a["target_iou"])}、ASSD={num(a["assd_m"])}米、体积比={num(a["volume_ratio"])}。中心井已提示位置并不保证帽顶/两肩精确。C−A IoU={c["target_iou"]-a["target_iou"]:+.6f}；ASSD变化={num(None if c["assd_m"] is None or a["assd_m"] is None else c["assd_m"]-a["assd_m"])}米；体积比从{num(a["volume_ratio"])}变为{num(c["volume_ratio"])}。预设中心剖面见fig03_section_Y26，不另挑切面。',
        f'B准确密集井整体IoU={num(b["target_iou"])}、ASSD={num(b["assd_m"])}米，提供冻结先验及条件利用能力的对照，但不是理论上界。C与B的差异需结合下述伪井误差解释；B若仍有残余误差，则不能全部归咎于属性转换。',
        f'直接全场分类IoU={num(d["target_iou"])}、ASSD={num(d["assd_m"])}米。它使用全场信息；Flow只接收512柱，柱间证据被丢弃，且受到冻结先验/条件利用限制。直接分类若优于C必须如实保留，不能作为等信息预算竞赛。',
        '\n## 固定域指标\n',
        '|域|臂|TP/FP/FN|P/R/IoU|固定类mIoU|体积比|质心偏差/米|\n|---|---|---|---|---:|---:|---:|']
    for domain in ('E_original_free','E_holdout'):
        for arm in ('A','B','C','DIRECT','COPY_ONLY'):
            r=metrics[arm,domain];out.append(f'|{domain}|{arm}|{r["target_tp"]}/{r["target_fp"]}/{r["target_fn"]}|{num(r["target_precision"])}/{num(r["target_recall"])}/{num(r["target_iou"])}|{num(r["fixed_class_miou"])}|{num(r["target_pred_truth_volume_ratio"])}|{num(r["centroid_error_m"])}|')
    out += ['\n各域类别集合在推理前按真值存在的非空气类别冻结；完整类别IoU、分母、质心体素单位等见metrics.csv。holdout未被硬写入，但其属性观测参与全场恢复，不是独立观测测试。',
        '\n## 同added_mask的井上转换质量\n',
        '|臂|准确率|错误数|目标TP/FP/FN|\n|---|---:|---:|---|']
    for r in e['condition_quality']:out.append(f'|{r["arm"]}|{num(r["accuracy"])}|{r["errors"]}|{r["target_tp"]}/{r["target_fp"]}/{r["target_fn"]}|')
    out += ['\n完整raw−1…13混淆矩阵和逐类质量另存well_confusion_B/C.csv、well_classes_B/C.csv。',
        '\n## 新条件之外的正确覆盖\n',
        '对全部臂统一使用 真目标 & 原未知域 & ~C新增mask 的同一子集。\n',
        '|臂|TP/分母|覆盖|\n|---|---|---:|']
    for r in e['outside']:out.append(f'|{r["arm"]}|{r["tp"]}/{r["voxels"]}|{num(r["coverage"])}|')
    out += ['\nCOPY_ONLY的holdout与A逐体素相同，且上述共同子集TP应与A相同；C在共同子集的增益才是直接写井之外的覆盖。覆盖提高不代替全体积IoU、表面误差和FP评价。',
        '\n## 计算与技术核验\n',
        f'实际新Flow为3份（A/B/C），共96次速度前向；每臂32次，seed42。Gaussian属性恢复仅1次，迭代{s["iterations"]}次、收敛={s["converged"]}、耗时{s["elapsed_seconds"]:.2f}秒。没有整网络smoke或历史样本重跑。',
        *[f'{k}采样/加载审计总耗时{read_json(ROOT/f"arms/{k}/predictions/seed42/receipt.json")["elapsed_seconds"]:.2f}秒。' for k in 'ABC'],
        '原条件/新增条件违例0；模型hash不变；噪声相同；B/C mask一致；holdout无新增；COPY holdout=A；TP/FP/FN与域分母一致。生成/恢复/推理/评价receipt和哈希均保存。CPU测试命令及日志见COMMANDS.md、CPU_TESTS.log。开发时首次CPU收集因新评价文件括号遗漏失败，在任何生成/恢复/网络前向前修正；属零前向工程错误。其他故障详见ENGINEERING.md。',
        '\n## 可选地震与结论边界\n',
        '可选地震尚未执行；主实验完整封存后另记录是否进入，见OPTIONAL_PHYSICS.json。未运行D、地震反演0次，不声称物理一致性。',
        '本轮只有一个预定几何、一个seed，已知Gaussian核、无额外噪声、人工双属性；只支持本案例上的受控判断，不做显著性、单seed置信区间、抗误差或未见地质泛化结论。不自动增加案例、seed、训练或参数扫描。',
        '\n## 文件与复现\n',
        '主图：figures/fig01_full_geology_comparison.{png,pdf}、fig02_target_body_comparison.{png,pdf}及clean版。中心剖面：fig03_section_Y26；准确井与直接分类剖面：fig04_controls_section_Y26。全部高清独立面板：figures/panels/（PNG/PDF），图外说明：figures/CAPTIONS.md。数字表位于本目录。',
        '新源码：project/geodata-3d-conditional/scripts/stage25/；科学源码快照：source_snapshot/；配置：PROTOCOL.json、GEOMETRY.json、layout.json、codebook.json；观测：observations/；原真值：truth/（仅生成/评价）；条件/预测：arms/A,B,C/；恢复场：preprocessing/。重绘命令见COMMANDS.md，必须选择新输出目录以保留本次图。',
        '建议后续由研究者决定是否进行“原井/地表条件张量完全相同、帽部宽度不同”的二模型形态歧义对照；本轮不执行。']
    (ROOT/'REPORT.md').write_text('\n'.join(out)+'\n',encoding='utf-8')
    write_json(ROOT/'MANDATORY_COMPLETE.json',dict(status='MANDATORY_COMPLETE',completed_at_utc=utc(),new_flow_outputs=3,velocity_evaluations=96,restoration_solves=1,report_sha256=file_hash(ROOT/'REPORT.md'),all_checks=e['checks'],figures_reviewed=True))
    print('MANDATORY_COMPLETE',ROOT,flush=True)
if __name__=='__main__':main()
