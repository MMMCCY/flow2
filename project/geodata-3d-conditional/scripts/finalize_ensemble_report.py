#!/usr/bin/env python3
"""Summarize the completed selected-case experiment without changing rankings."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); p.add_argument('--report',type=Path,required=True); args=p.parse_args()
    summary=json.loads((args.report/'summary.json').read_text()); protocol=json.loads((args.run/'protocol.json').read_text())
    validation=json.loads((args.report/'VALIDATION_REPORT.json').read_text())
    if summary['actual_count']!=200 or validation['status']!='PASS': raise ValueError('completed, audited 200 required')
    with (args.report/'candidate_metrics.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    by_arm={arm:{r['sample_id']:r for r in rows if r['arm']==arm} for arm in summary['arms']}
    lines=['# 同案例 200 采样、筛选与绘图结果',
           '**真实实验结果：** 200 个独立source draws；每个draw生成1个共享baseline、1个Stage15 trace v4模型和1个Stage19R adapter模型，共600个完整硬岩性模型。失败数0，硬条件违背0。',
           f'案例：`{summary["case_id"]}`。用户选择按历史 adapter 三种子的中位目标 IoU 最高案例进行展示。历史中位 IoU为0.6710347944；完整选择排名在protocol。这是事后选择案例，不能解释为无偏跨案例比较或新泛化证据。200个实现是一个案例内的随机重复，不是200个独立地质测试案例。',
           'Stage15固定v4 positive-endpoint/confidence方法迁移至这个TEST案例；不冒称历史cond_generation_0结果。Stage19R使用正式epoch4已训练54,003参数adapter；此次不训练，不能把整条方法称training-free。两条路线共享相同checkpoint、条件、观测、证据和source draws。',
           '观测筛选：完整硬地质→固定binary acoustic映射→卷积地震正演→未归一化hard seismic RMSE（振幅单位，越低越好）；不含真值指标。所有obs IDs均在本次后处理读取truth前写出。case选择与oracle筛选单独声明使用真值。',
           'random100使用20261009种子无放回；obs/oracle平局按冻结source order；oracle按完整raw-label9真实IoU降序，仅事后候选支持诊断。每个集合独立重算count，不在不同Top100间进行伪配对。',
           '']
    table=['| 方法 | 集合 | 中位候选IoU | 中位hard RMSE | >80%自由域IoU | 100%自由域IoU |','|---|---|---:|---:|---:|---:|']
    def fmt(v): return 'undefined' if v is None else f'{v:.4f}'
    detailed={}
    for arm,pools in summary['arms'].items():
        detailed[arm]={}
        for name in ('all200','random100','obs_top100','oracle_top100'):
            pool=pools[name]; metrics=pool['threshold_metrics']; selected=json.loads((args.report/arm/name/'selected_ids.json').read_text())
            table.append(f'| {arm} | {name} | {pool["median_candidate_iou"]:.4f} | {pool["median_candidate_hard_seismic_rmse"]:.5f} | {fmt(metrics[2]["free_subsurface"]["iou"])} | {fmt(metrics[3]["free_subsurface"]["iou"])} |')
            values=np.array([float(by_arm[arm][sid]['full_iou']) for sid in selected])
            detailed[arm][name]=dict(mean_iou=float(values.mean()),median_iou=float(np.median(values)),minimum_iou=float(values.min()),maximum_iou=float(values.max()),p05_iou=float(np.quantile(values,.05)),p95_iou=float(np.quantile(values,.95)))
    lines+=['## 全集合与筛选结果','\n'.join(table),'表中IoU为完整目标类别的候选IoU；阈值IoU是在核实的地下自由域中评价。硬条件区的频率不能作为隐藏体恢复证据。']
    for method in protocol['methods']:
        delta=summary['paired_summary'][method]; support='改善' if delta['median_delta_iou']>0 else '未改善'
        obs=detailed[method]['obs_top100']['mean_iou']; random=detailed[method]['random100']['mean_iou']; oracle=detailed[method]['oracle_top100']['mean_iou']; all_mean=detailed[method]['all200']['mean_iou']
        lines += [f'## {method}',
                  f'1. 候选支持：同源200对的中位IoU增量 {delta["median_delta_iou"]:+.6f}，本案例内{support}。中位hard RMSE增量 {delta["median_delta_hard_seismic_rmse"]:+.6f}；地质与物理改善分别判断。',
                  f'2. 观测筛选：all200平均候选IoU {all_mean:.6f}，random100 {random:.6f}，obs_top100 {obs:.6f}，oracle_top100 {oracle:.6f}。obs相对random增量 {obs-random:+.6f}；oracle与obs差 {oracle-obs:+.6f}。只比较集合统计，不把不同Top100的成员伪配对。']
        masks=summary['arms'][method]['all200']['threshold_metrics']
        lines.append('3. 频率摘要：all200地下自由区的四个严格阈值指标如下。高频一致性不等于正确性；提高阈值所丢失的真目标与保留下来的误报均明确报告。')
        for label,metric in zip(('>40%','>60%','>80%','all-hit'),masks):
            m=metric['free_subsurface']
            lines.append(f'{label}: IoU={fmt(m["iou"])}, precision={fmt(m["precision"])}, recall={fmt(m["recall"])}, TP/FP/FN={m["tp"]}/{m["fp"]}/{m["fn"]}, voxels={m["predicted_voxels"]}, components6={m["connected_components_6"]}.')
        lines.append(f'Unique decoded models: {summary["arms"][method]["unique_decoded_models"]}/200。独立draw即使同模型也保留多重性。')
    lines += ['## 图件与核验',
              'Figure8按冻结顺序/排序等距分位展示12个成员。Figure9四个严格阈值：N200要求81/121/161/200次，N100要求41/61/81/100次。full/free_subsurface均有等距与正视、论文和1920×1080汇报PNG/SVG。红色是独立条件体素overlay；不能看成阈值命中。只知aggregate condition mask时不臆造surface/well分类。',
              '绘图使用真实二值体素外露面，没有平滑、闭运算、填洞、删小体或改轴序。XYZ物理extent为6.4/6.4/3.2km、cell100/100/50m，Z向上。full与free区的统计和指标均独立保存。',
              f'自动产物核验PASS：200源IDs、600模型、同源噪声哈希、条件一致、100无放回、完整排序、所有count和frequency重算、严格阈值嵌套与all-hit、12个Figure8 IDs；{validation["validated_png_files"]}个PNG通过格式检查。PNG人工视觉检查记录另附。',
              '软件测试：随附6个unittest通过；新增接入与Stage19回归36个pytest通过；出版渲染测试1个通过。13个既有Matplotlib弃用警告不影响结果。',
              '## 范围与限制',
              'binary high-contrast、noiseless inverse-crime机制上限；不是field seismic或15类实际物性反演。q为continuous binary evidence，频率为未校准empirical occurrence。阈值mask不是完整地质实现，不能继承成员的物理残差。本次未做五体任务、逐对象恢复或跨案例泛化评价。历史CFM流式预训练没有逐样本seed manifest，不能排除与合成案例家族的历史训练样本重叠。',
              '原参考ensemble_occurrence.py缺失，已按README与测试重新实现；该差异详见ASSET_AUDIT_REPORT。checkpoint、数据和GPU均实际可用。旧实验和用户既有改动保持原样，没有push或发布。',
              '详细路径、哈希与冻结参数见protocol.json；候选指标见candidate_metrics.csv；同源增量见paired_deltas.csv；图件见FIGURE_INDEX.md。']
    (args.report/'DEVELOPMENT_REPORT.md').write_text('\n\n'.join(lines)+'\n')
    (args.report/'selection_diagnostics.json').write_text(json.dumps(detailed,indent=2)+'\n')
    artifacts=[]
    for path in sorted(args.report.rglob('*')):
        if path.is_file() and path.name!='ARTIFACT_MANIFEST.json':
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
            artifacts.append(dict(path=str(path.relative_to(args.report)),bytes=path.stat().st_size,sha256=h.hexdigest()))
    (args.report/'ARTIFACT_MANIFEST.json').write_text(json.dumps(dict(status='completed',case=summary['case_id'],actual_count=200,artifacts=artifacts),indent=2)+'\n')
    print('\n'.join(table))

if __name__=='__main__': main()
