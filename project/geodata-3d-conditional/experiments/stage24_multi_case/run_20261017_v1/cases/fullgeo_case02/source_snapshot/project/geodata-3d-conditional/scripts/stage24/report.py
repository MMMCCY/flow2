"""Three-case factual summary and external PPT captions; no new scientific computations."""
import csv
import os
import sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage24.common import *

def fmt(v):return 'NA' if v is None else f'{v:.6f}'
def records(path):
    with path.open() as f:return list(csv.DictReader(f))

def report():
    guard('report')
    if (ROOT/'REPORT.md').exists():raise FileExistsError('Preserve completed report')
    results={case:read_json(ROOT/'cases'/case/'EXECUTION.json') for case in CASES}
    original=read_json(R23/'EXECUTION.json')
    rename={'FLOW_STAGE23':'FLOW_RESTORED_PSEUDO','COPY_ONLY_STAGE23':'COPY_ONLY'}
    old=dict(case='fullgeo_case03',samples=[],conversion=[],objects_common_outside=[])
    for s in original['samples']:
        if s['arm']=='FLOW_STAGE22':continue
        old['samples'].append({**s,'arm':rename.get(s['arm'],s['arm'])})
    old['conversion']=[{**q,'case':'fullgeo_case03','arm':'RESTORED_WELLS' if q['arm']=='STAGE23_WELLS' else q['arm']} for q in original['conversion']]
    old['objects_common_outside']=[dict(case='fullgeo_case03',object_id=1,major=True,
        **{k:r[k] for k in ('voxels','tp','coverage')},arm=rename.get(r['arm'],r['arm']))
        for r in original['object1_common_subset'] if r['arm']!='FLOW_STAGE22']
    results['fullgeo_case03']=old
    metrics=[];quality=[];outside=[];concise=[]
    for case,r in results.items():
        for s in r['samples']:
            for domain,m in s['metrics'].items():
                metrics.append(dict(case=case,seed=42,arm=s['arm'],domain=domain,**m))
        quality.extend(r['conversion']);outside.extend(r['objects_common_outside'])
        by={s['arm']:s for s in r['samples']};a=by['A_9'];c=by['C_512'];n=by['FLOW_RESTORED_PSEUDO'];d=by['RESTORED_CLASSIFICATION']
        q={row['arm']:row for row in r['conversion']}
        h=n['metrics']['E_holdout'];ch=c['metrics']['E_holdout'];primary=n['metrics']['E_original_free']
        concise.append(dict(case=case,seed=42,old_C_well_accuracy=q['C_512']['accuracy'],
            restored_well_accuracy=q['RESTORED_WELLS']['accuracy'],
            old_C_well_errors=q['C_512']['errors'],restored_well_errors=q['RESTORED_WELLS']['errors'],
            restored_target_TP=q['RESTORED_WELLS']['target_tp'],restored_target_FP=q['RESTORED_WELLS']['target_fp'],
            restored_target_FN=q['RESTORED_WELLS']['target_fn'],
            holdout_A_target_iou=a['metrics']['E_holdout']['target_iou'],holdout_C_target_iou=ch['target_iou'],
            holdout_new_target_iou=h['target_iou'],holdout_C_recall=ch['target_recall'],holdout_new_recall=h['target_recall'],
            holdout_C_miou=ch['fixed_class_miou'],holdout_new_miou=h['fixed_class_miou'],
            primary_new_precision=primary['target_precision'],primary_new_recall=primary['target_recall'],
            primary_new_iou=primary['target_iou'],primary_new_TP=primary['target_tp'],primary_new_FP=primary['target_fp'],
            primary_new_FN=primary['target_fn'],primary_new_volume_ratio=primary['target_pred_truth_volume_ratio'],
            direct_primary_iou=d['metrics']['E_original_free']['target_iou'],
            direct_primary_miou=d['metrics']['E_original_free']['fixed_class_miou'],
            modeling_holdout_R_IoU_mIoU_all_improve=(h['target_recall']>ch['target_recall'] and h['target_iou']>ch['target_iou'] and h['fixed_class_miou']>=ch['fixed_class_miou'])))
    write_csv(ROOT/'three_case_metrics.csv',metrics)
    write_csv(ROOT/'three_case_condition_quality.csv',quality)
    write_csv(ROOT/'three_case_zero_condition_objects.csv',outside)
    write_csv(ROOT/'three_case_summary.csv',concise)
    # Original case03 object tables are reused by pointer; new cases contain full exposure tables.
    lines=['# Stage24：固定绘图标准与两例固定方法复用','',
        '已达到本轮预算并停止：case03科学结果与认可主图复用；case01/02各1次Stage22原solver、1份seed42冻结Flow、32次速度前向。合计2次新恢复、2份新Flow、64前向。无工程重试、训练、新seed、参数扫描或新增案例。','',
        '## 绘图标准已实际封存','',
        '`visualization_standards/geology_approved_v1/`保存真实认可入口build_stage23_reference_figures.py、全部必要本地依赖快照/hash、实际配置、原样图PNG/PDF、原输入身份及命令。renderer.py执行认可源码中未修改的camera/project/render/compose函数AST，并使用冻结style快照；只参数化输入、案例及输出。case03四张普通/干净PNG与认可样图逐像素完全一致。主图直接复制认可版原字节，未重设风格。','',
        '所有案例使用同一XYZ体素轴、色表、光照、相机与缩放。整体图上下排完整/固定缺口；目标图一行三列、原9井细青线、完整64³框。新案例与case03用相同镜头，未逐方法居中或缩放。全部目标分量保留，无ghost、ROI修剪、去小体、填洞或平滑。每例另有9份独立面板PNG。','',
        '## 方法保持不变','',
        '输入为R21保存的sigma1双属性观测；不从完整真值重新生成。直接复用Stage22 CPU FP64有界TV恢复：原FP32 Gaussian核提升、radius3/separable/replicate、固定尺度与含空气界限、原硬条件查表投影、TV1e−4、最多2000迭代及连续50次相对变化<1e−6。最终接受迭代，无真值挑选。固定原分类raw0…13/权重/尺度/并列规则，原512总井布局的所有非原条件深度写入完整多岩性标签。','',
        '前处理有真值/对象/诊断资产读取防火墙；只解析evaluation_registry中properties元数据，不打开truth。恢复场、条件封存及唯一预测完成后才进行独立评价。Flow仅接受条件与实际旧A噪声；原Stage23已修复入口直接复用，411EMA/原embedding冻结、FP32无TF32/autocast、32步中点Euler与硬投影不变。原条件/新增条件违例0，模型身份不变。无按精度筛选是否执行的科研gate。','',
        '## 三例分开判断','',
        '|案例|伪井准确率 C→恢复|holdout IoU C→新|holdout recall C→新|holdout mIoU C→新|新原未知域P/R/IoU|新目标体积比|直接分类IoU|',
        '|---|---|---|---|---|---|---:|---:|']
    for r in sorted(concise,key=lambda x:x['case']):
        lines.append(f"|{r['case']}|{fmt(r['old_C_well_accuracy'])}→{fmt(r['restored_well_accuracy'])}|{fmt(r['holdout_C_target_iou'])}→{fmt(r['holdout_new_target_iou'])}|{fmt(r['holdout_C_recall'])}→{fmt(r['holdout_new_recall'])}|{fmt(r['holdout_C_miou'])}→{fmt(r['holdout_new_miou'])}|{fmt(r['primary_new_precision'])}/{fmt(r['primary_new_recall'])}/{fmt(r['primary_new_iou'])}|{fmt(r['primary_new_volume_ratio'])}|{fmt(r['direct_primary_iou'])}|")
    lines += ['', '转换准确率、井外目标指标、全类mIoU及误报/体积应分别解读。B准确真井参考、C原平滑伪井和直接恢复分类均保留在three_case_metrics.csv。B不是理论几何上界；直接分类使用完整空间信息，Flow只获得伪井采样，信息预算不同。','',
        '## 全部原零硬条件对象的共同井外子集','',
        '完整真值26连通ID先建立，再按固定域进行旧一对一匹配和merge/split。两个新案例的object_exposure/object_metrics/predicted_object_metrics保留全部ID、体积、原井/硬条件命中、匹配IoU/coverage及任意预测覆盖。下表不挑最好的隐藏对象；同一对象对所有臂使用 object∩E_original_free∩~NEW_mask 的共同分母。','',
        '|案例|对象|major|共同体素|A TP|C TP|COPY TP|新Flow TP|新覆盖|','|---|---:|---|---:|---:|---:|---:|---:|---:|']
    keys=sorted({(r['case'],r['object_id']) for r in outside})
    for case,i in keys:
        rows={r['arm']:r for r in outside if r['case']==case and r['object_id']==i}
        n=rows['FLOW_RESTORED_PSEUDO']
        lines.append(f"|{case}|{i}|{n['major']}|{n['voxels']}|{rows['A_9']['tp']}|{rows['C_512']['tp']}|{rows['COPY_ONLY']['tp']}|{n['tp']}|{fmt(n['coverage'])}|")
    lines += ['', 'case01没有原井零命中的major（完整体积≥20）目标对象；其主要目标原井已命中，零井分量很小，不能称第二个主要隐藏体成功案例。case02全部零井/零硬条件对象均保留；没有恢复的对象也必须明示。匹配IoU低或匹配0不能替代任意覆盖描述，覆盖高也不抵消巨大FP。','',
        '本任务是已知核、无额外噪声、人工属性代理、已见案例上的展示扩充。三案例各只有seed42，holdout属性参与了全局代理，既不是独立观测留出，也不是独立未见地质TEST；不宣称跨案例稳健或泛化成立。无显著性检验、误差棒或新后续实验。','',
        '交付路径：标准visualization_standards/geology_approved_v1；新结果cases/fullgeo_case01与02；三例主图/干净版/单面板reports/<case>；统一数字表three_case_summary.csv及完整指标表；外置图注CAPTIONS.md；PPT_ASSET_INDEX.md给出逐页用图和必须说明的限制。','']
    (ROOT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    captions=['# Stage24 captions / 外置中英文图注','',
        'Ground truth / Frozen Flow / Ours均指完整64³标签体。Frozen Flow为原9井与地表条件的旧A，Ours为相同冻结网络接受恢复属性完整伪井；未重训。整体图为完整/同一缺口，目标图保留全部raw9分量，青色细线为原9井、灰框为0…64体素坐标范围，非米或海拔。相同approved_v1色表、镜头、尺度和照明。PDF三维为栅格，文字/图例为矢量。','',
        'Each case uses source seed42, a synthetic property surrogate with known Gaussian kernel and a frozen network. The complete model/target is shown; reported metrics use the fixed registered E_original_free or E_holdout domains. No target ROI/component filtering, geometric repairs, smoothing or truth ghost. All FP fragments remain. One realization per reused case does not establish held-out generalization.','']
    for r in concise:
        case=r['case'];captions.extend([f'## {case}, seed42','',
            f"Holdout target IoU: C {fmt(r['holdout_C_target_iou'])} → Ours {fmt(r['holdout_new_target_iou'])}; holdout all-class mIoU {fmt(r['holdout_C_miou'])} → {fmt(r['holdout_new_miou'])}. Original-free Ours P/R/IoU {fmt(r['primary_new_precision'])}/{fmt(r['primary_new_recall'])}/{fmt(r['primary_new_iou'])}; target volume ratio {fmt(r['primary_new_volume_ratio'])}. Direct restored FULL-field classification IoU {fmt(r['direct_primary_iou'])}, mIoU {fmt(r['direct_primary_miou'])}; unlike Flow it uses full-field information.",'',
            'case01：major目标原井已命中，不能称为新零井主要隐藏体恢复。' if case.endswith('01') else ('case02：所有零井目标对象及失败对象见REPORT与对象表，不挑选最好分量。' if case.endswith('02') else 'case03：既有认可结果复用，新Flow/恢复次数均0；共同510井外对象体素恢复144，非IoU。'),''])
    (ROOT/'CAPTIONS.md').write_text('\n'.join(captions),encoding='utf-8')
    pages=['# PPT_ASSET_INDEX','', '不制作PPTX；以下图片可直接插入。所有图例/长方法说明与数字留在讲稿或外置图注。','']
    for page,case,title in ((1,'fullgeo_case03','原有主要隐藏体示例'),(2,'fullgeo_case02','多个零井目标对象，保留失败'),(3,'fullgeo_case01','小目标/全类表现，非主要隐藏体新成功')):
        q=next(r for r in concise if r['case']==case)
        pages.extend([f'## 第{page}页：{case} / seed42 — {title}','',
            f'[整体图](reports/{case}/fig01_full_geology_comparison.png) · [目标图](reports/{case}/fig02_target_body_comparison.png)；同名PDF、_clean版本与panels/九个单面板均在同目录。','',
            f"可陈述的数字：伪井准确率{fmt(q['old_C_well_accuracy'])}→{fmt(q['restored_well_accuracy'])}；holdout目标IoU{fmt(q['holdout_C_target_iou'])}→{fmt(q['holdout_new_target_iou'])}；新目标FP={q['primary_new_FP']}，体积比={fmt(q['primary_new_volume_ratio'])}。不能把所有指标预写为改善；直接恢复分类IoU={fmt(q['direct_primary_iou'])}必须保留。",'',
            '限制：'+('case01无原井零命中的major目标，不可作为第二个主要隐藏体恢复成功。' if case.endswith('01') else '查看REPORT全部对象共同井外TP及匹配IoU，完整保留未恢复对象、碎裂和FP。'),''])
    pages.extend(['## 第4页：三例简洁数字表','', '[three_case_summary.csv](three_case_summary.csv)：分别列转换、井外指标、FP/体积和直接分类；完整A/B/C参考见[three_case_metrics.csv](three_case_metrics.csv)。不画误差棒，不称未见地质泛化。',''])
    (ROOT/'PPT_ASSET_INDEX.md').write_text('\n'.join(pages),encoding='utf-8')
    write_json(ROOT/'EXECUTION.json',dict(status='completed',new_cases=list(CASES),source_seed=42,
        new_property_solves=2,new_Flow_outputs=2,velocity_evaluations=64,engineering_retries=0,
        case03_scientific_compute=0,training=0,parameter_scans=0,stop_reason='two_case_budget_complete',
        standard_manifest_sha256=file_hash(STANDARD/'MANIFEST.json'),completed_at_utc=utc()))
    print('Three-case report and PPT index written; STOP at 2 solves / 2 predictions / 64 forwards',flush=True)

if __name__=='__main__':report()
