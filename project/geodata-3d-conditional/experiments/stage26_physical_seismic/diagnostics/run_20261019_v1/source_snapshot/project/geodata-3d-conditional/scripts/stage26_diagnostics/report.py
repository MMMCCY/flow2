"""Evidence-led diagnostic report; saved statistics only, no numerical experiments."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage26_diagnostics.common import *

def n(x):return 'NA' if x is None else f'{x:.6f}'
def main():
    e=read_json(ROOT/'EVALUATION.json');oracle=read_json(ROOT/'ORACLE_STATE.json');r=read_json(ROOT/'figures/RENDER_CONFIG.json')
    assert r['visual_inspection']=='completed'
    if (ROOT/'REPORT.md').exists():raise FileExistsError('Preserve diagnostic report')
    quality={(x['field'],x['candidates']):x for x in e['condition_quality']}
    shape={(x['field'],x['candidates']):x for x in e['morphology']}
    domains={(x['field'],x['candidates'],x['domain']):x for x in e['domain_metrics']}
    twt={x['region']:x for x in e['twt_statistics']}
    errors={(x['field'],x['region']):x['MAE'] for x in e['log_errors']}
    physical={(x['field'],x['candidates']):x for x in e['physical_residuals']}
    lines=['# Stage26离线诊断：固定速度/时深关系与完整岩性分类','',
        '**结论：完整码本接口确实放大了大量背景错分；但缩小候选集合没有带来新的井外目标覆盖，正确速度oracle也没有充分恢复帽部。当前不是只改速度、或只缩码本便可解决的问题。下一步应优先处理反演前端“阻抗证据是否足以可靠分类”，配合研究原条件支持的候选约束；不应直接追加Flow。**','',
        '本报告区分已确认现象和合理推断，不把oracle当作可部署方法。全部正式Stage26数值只读；新Flow=0、新checkpoint加载=0、新观测生成=0、新训练=0。仅新增一次ORACLE_DIAGNOSTIC_ONLY反演，其余使用保存场。','',
        '## 已确认：原分类一致，候选来源没有真值泄漏','',
        '从注册记录加载prior_exact/updated、原条件及分类张量。两份保存场重做FULL分类，与原函数及原classification_D0/D均逐体素完全一致；shape为[1,2,64,64,64]、FP64、阻抗kg m^-2 s^-1、慢度s/m，原条件精确、旧更新慢度不变。没有带着不同分类定义继续比较。','',
        'FULL固定raw0…13；OBSERVED只由原条件values[original_mask]的非空气标签提取，实际为'+str(e['candidates']['OBSERVED'])+'。该集合在prepare独立进程读取真值、B/C或评价域之前封存，未硬编码预期、未额外补9。两套候选均用同一log阻抗最近值/first argmin、相同空气定义及原条件优先。来源计数见candidate_sources.csv/CANDIDATES.json。该案例中原条件恰好覆盖真值类别，不代表现实未见岩性可被安全剔除。','',
        '## 六个离线分类：条件质量与核心形态','',
        '条件分母26156；新井外子集统一为 真目标 & 原未知域 & ~固定新增mask，共1933。帽部区原样使用Stage25预设参数，不按输出选择区域。','',
        '|物性场|候选|井上准确率|井上TP/FP/FN|原未知域IoU|帽部区IoU|新井外TP|全目标体积比|','|---|---|---:|---|---:|---:|---:|---:|']
    for field in FIELDS:
        for candidate in CANDIDATE_NAMES:
            q=quality[field,candidate];s=shape[field,candidate];m=domains[field,candidate,'E_original_free']
            lines.append(f'|{field}|{candidate}|{n(q["accuracy"])}|{q["target_tp"]}/{q["target_fp"]}/{q["target_fn"]}|{n(m["target_iou"])}|{n(s["cap_box_iou"])}|{s["common_outside_tp"]}/1933|{n(s["full_volume_ratio"])}|')
    lines += ['',
        '同一保存updated上，FULL→OBSERVED改3917个新增标签：2084错→对、1833错→另一错误、0对→错。准确率79.9778%→87.9454%，帽部IoU0.088983→0.147412。已确认完整候选接口有可利用的局部改进空间。但剩余1833并未变正确，例如真1的2→3有220个、13→0有197个，真3的8→1有191个、12→5有145个。删掉未观测类也会把错误重新分配到已知类。','',
        '必须与同OBSERVED的prior比较：准确率87.0202%→87.9454%（仅+0.9252个百分点）；帽部IoU0.131634→0.147412；完整目标召回24.9887%→25.0338%几乎不变；井外TP446→441。全目标IoU0.116191→0.151474伴随FP2551→1447、体积比1.40054→0.90302，主要是误报减少，不能称新增目标覆盖恢复。prior本身FULL→OBSERVED也修正99个错误，这部分属于类别先验收益，不能归给地震更新。','',
        '完整P/R/IoU、原未知域/holdout/FULL指标见condition_quality.csv、domain_metrics.csv、core_morphology.csv。错误转移见error_transfers.csv及class_transition_counts.csv。没有用这些指标重选候选集或修改正式井条件。','',
        '## 已确认：连续更新有局部收益，也破坏了原正确阻抗','',
        '|物性场|原未知域log阻抗MAE|真目标MAE（2180体素）|非目标MAE（210344体素）|','|---|---:|---:|---:|']
    for field in FIELDS:lines.append(f'|{field}|{n(errors[field,"original_free_all"])}|{n(errors[field,"original_free_target"])}|{n(errors[field,"original_free_non_target"])}|')
    lines += ['',
        '真值阻抗只在评价进程由真值+码本计算，不进入求解、初始阻抗、边界、正则或分类器。MAE按相同物理单位的自然log阻抗计算。保存updated在目标和背景的平均误差均下降，但FULL标签准确率下降，说明连续误差与分类正确率并非同一目标。','',
        '尤其在新增井1792个FULL“对→错”位置，prior阻抗原本等于真码值，log误差均值0；updated变为0.104118。主要真类→预测类：3→2为317个、5→12为302个、1→7为237个、3→11为209个、1→8为180个、5→6为161个。这里连续数值确实被推离真值，不能把损害全部说成“数值更正确但分类边界不合适”，也没有证据将原因先定为平滑。详见right_to_wrong_log_errors.csv。','',
        '## 已确认：时深差局部显著，但正确速度不充分','',
        '使用公开地表subsurface支持和原cell_center_twt_ms。该函数返回top-down数组，先flip回原XYZ再套同深度域；这不是已匹配地震反射事件之间的时差。','',
        '|域|中位数/ms|P90/ms|最大/ms|P90/8ms样点|P90/40ms周期|','|---|---:|---:|---:|---:|---:|']
    for name in ('original_free','fixed_cap_box'):
        x=twt[name];lines.append(f'|{name}|{n(x["median_ms"])}|{n(x["p90_ms"])}|{n(x["max_ms"])}|{n(x["p90_8ms_samples"])}|{n(x["p90_40ms_periods"])}|')
    lines += ['',
        '未知地下域最大差641.34ms约80.17个采样间隔/16.03个25Hz周期；帽部区最大73.77ms约9.22个采样间隔/1.84个周期。中位数0不能排除局部显著差异；仅凭P90/最大值也不能断言它是唯一主因。','',
        'oracle输入的初始阻抗逐体素等于旧prior_exact，仅固定慢度改为真值慢度。这同时干预先验地震到时与时间修正向深度映射，不是只更换后处理插值。原观测、码本、井/地表投影、波子与0.001/0.01正则完全不变；oracle更新前后慢度不变，原条件保持。没有真值阻抗参与初始值/边界/正则中心，没有真值岩性送给分类器。','',
        '|连续物性对照|更新前relative L2|更新后relative L2|','|---|---:|---:|',
        f'|固定A慢度（复核保存场）|{n(physical["saved_A_prior","none"]["relative_l2"])}|{n(physical["saved_A_updated","none"]["relative_l2"])}|',
        f'|ORACLE真值慢度（唯一新反演）|{n(physical["oracle_prior","none"]["relative_l2"])}|{n(physical["oracle_updated","none"]["relative_l2"])}|','',
        '正确速度使连续拟合更好，目标log阻抗MAE也进一步降至0.253628；但oracle FULL准确率79.0526%低于旧updated FULL的79.9778%，完整目标IoU0.135135也低于0.139236。其帽部IoU0.108181虽有局部增加，仍很低。','',
        '同OBSERVED下，oracle相对旧updated准确率87.9454%→87.9110%基本不变；帽部IoU0.147412→0.175040、井外TP441→489（覆盖+2.48个百分点），但FP1447→1724、体积比0.90302→1.05187，完整IoU仅0.151474→0.154276。目标召回有小幅增加而非岩盖正确恢复。只能说正确速度在部分目标区域有帮助，不能将全部收益或失败归于该模块。','',
        '## 硬分类后的物理误差另列','',
        '下表每个硬标签场都重新查表获得自己的阻抗和速度，使用同一观测的完整1638400样点。未在硬标签正演中借oracle慢度提高分数。','',
        '|分类前物性场|FULL硬标签relative L2|OBSERVED硬标签relative L2|','|---|---:|---:|']
    for field in FIELDS:lines.append(f'|{field}|{n(physical[field,"FULL"]["relative_l2"])}|{n(physical[field,"OBSERVED"]["relative_l2"])}|')
    lines += ['',
        'OBSERVED改善分类正确率，并不保证物理拟合继续改善。oracle连续误差0.825122转成硬标签后为1.305966/FULL或1.329898/OBSERVED；不能混报为oracle岩性仍有0.825的物理误差。旧Flow D的4.006694还含地下空气及生成后速度改变的影响，本轮未据其单独归因，也未清理D或增加生成实验。','',
        '## 下一步判断：先修前端可分类性，候选接口作为辅助','',
        '**已确认**：FULL多余候选会吸收大量背景标签；可从真实原条件提取的OBSERVED能局部修正错误，但其目标覆盖几乎没有新增收益。时深关系有局部明显偏差；提供正确固定速度后数据拟合更好，却仍不能生成可靠的阻抗岩性证据。原更新破坏原正确属性点的事实也已确认。','',
        '**合理推断，尚非唯一归因**：单次线性化log阻抗更新的幅值/低频证据不足，以及固定不准确阻抗先验与完整类别映射之间的目标不匹配，可能共同限制伪井质量。正确速度oracle并未使FULL显著变好，因而不支持“只修速度即可解决”；OBSERVED+oracle也只是有限局部改善，不能作为成功方法。','',
        '**建议**：优先由MCYR决定如何诊断/改善前端阻抗证据的保真与可分类性，并把从实际观测得到的候选类别约束作为辅助接口方向；暂不在当前配置上加井、加seed、调阈值或重跑Flow。下一步不宜仅用地震残差下降作为进入生成的证据。具体前端方法和新试验预算需另行决定，本轮不实现。','',
        'oracle的真值速度本身携带地下结构，甚至与类别一一关联，现实不可获得。即便出现改善，也只说明给定额外正确速度后的反事实可能性，不是可部署恢复成果，不证明速度唯一主因。本轮一个固定案例，不作显著性或泛化声明。','',
        '## 数值、工程与文件','',
        f'新增ORACLE_DIAGNOSTIC_ONLY反演调用1次、耗时{oracle["elapsed_seconds"]:.6f}秒，保存完整bundle和oracle prior/updated/前后地震场。新Flow/checkpoint/训练/观测生成均0；原正式反演没有重跑或改写。4项必要小测试通过：候选来源、同规则分类/优先级、TWT轴向、实际保存→哈希读回→加载及历史只读/禁止checkpoint检查。数值优先原子保存，已有完整oracle则恢复或拒绝覆盖，不重复求解。','',
        '六个完整分类张量见classifications/，oracle资产均在oracle/且带ORACLE_DIAGNOSTIC_ONLY标记；没有新正式推理注册表或Flow入口。配置/必要输入身份见configs/、INPUT_FREEZE.json、ORACLE_PROTOCOL.json。紧凑CSV在本目录，固定Y26两张解释图见figures/，PNG/PDF均已打开核验。黑线是真值目标轮廓，仅用于事后解释；无新三维展示或标签修补。源码scripts/stage26_diagnostics/，重现及只重绘命令见COMMANDS.md。未推送GitHub。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(ROOT/'COMPLETION.json',dict(status='complete',verdict='candidate interface improves background classification; correct slowness is insufficient; frontend classifiability remains limited',new_oracle_inversions=1,new_Flow=0,new_checkpoint_loads=0,new_training=0,new_observation_generations=0,report_sha256=file_hash(ROOT/'REPORT.md'),completed_at_utc=utc()))
    print('DIAGNOSTICS_COMPLETE',ROOT,flush=True)
if __name__=='__main__':main()
