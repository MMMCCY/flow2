"""Evaluate saved D and re-forward its hard labels; approved separate Truth/A/D figures."""
import sys,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage25_physics.common import *
from scripts.stage25.evaluate import surface_distance,centroid_distance
from scripts.stage21.evaluate import voxel_metrics,ratio
from scripts.stage23.evaluate import sealed_prediction,conversion
from scripts.stage24.plot import standard
from scripts.stage25.plot import sections,save_panel
import numpy as np
import torch

def main():
    guard('evaluate');torch.set_num_threads(4)
    inv=read_json(PHYS/'INVERSION_STATE.json');assert inv['status']=='completed' and inv['source_sha256']==physics_sources()
    if (PHYS/'EVALUATION.json').exists():raise FileExistsError('D evaluation immutable')
    r,dp=sealed_prediction(ROOT/'arms/D/predictions/seed42');ar,ap=sealed_prediction(ROOT/'arms/A/predictions/seed42')
    assert r['velocity_evaluations']==32 and r['base_hash_after']==EXPECTED_MODEL and r['initial_noise_tensor_sha256']==ar['initial_noise_tensor_sha256']
    dc=check_condition(read_json(ROOT/'arms/D/inference_registry.json')['conditions']);cc=check_condition(read_json(ROOT/'arms/C/inference_registry.json')['conditions'])
    assert torch.equal(dc['mask'],cc['mask']) and not ((dp!=dc['values'])&dc['mask']).any()
    # Physics consistency is computed before evaluation truth is opened.
    table,operator=setup();f=read_json(PHYS/'OBSERVATION_FREEZE.json');obs=verify_tensor(f['assets']['observation']);sub=verify_tensor(f['assets']['subsurface'])
    acoustic=hard_labels_to_acoustic(dp,table)
    field=operator(acoustic[:,0:1],acoustic[:,1:2],sub)
    derror=float((field-obs).square().sum().sqrt()/obs.square().sum().sqrt())
    forward_asset=save_tensor(PHYS/'forward_D_hard_seismic.pt',field)
    e=read_json(ROOT/'evaluation_registry.json');truth=verify_tensor(e['truth']).numpy()[0,0];d=dp.numpy()[0,0]
    domains={k:(verify_tensor(v['mask']).numpy()[0,0],v['classes']) for k,v in e['domains'].items()}
    domains['FULL']=(np.ones_like(truth,dtype=bool),sorted(int(x) for x in np.unique(truth) if x!=-1))
    metrics=[]
    for name,(domain,classes) in domains.items():
        m=voxel_metrics(d,truth,domain,classes);m.update(target_pred_truth_volume_ratio=ratio(m['target_pred_volume'],m['target_truth_volume']),centroid_error_voxels=centroid_distance((d==9)&domain,(truth==9)&domain,[1]*3),centroid_error_m=centroid_distance((d==9)&domain,(truth==9)&domain,[SPACING]*3))
        assert m['target_tp']+m['target_fp']==m['target_pred_volume'] and m['target_tp']+m['target_fn']==m['target_truth_volume']
        metrics.append(dict(arm='D',domain=name,**m))
    distance=surface_distance(d==9,truth==9,[SPACING]*3)
    full=next(m for m in metrics if m['domain']=='FULL')
    shape=dict(arm='D',target_iou=full['target_iou'],assd_m=distance,assd_voxels=None if distance is None else distance/SPACING,volume_ratio=full['target_pred_truth_volume_ratio'],target_pred_volume=full['target_pred_volume'],target_truth_volume=full['target_truth_volume'],detected=bool((d==9).any()))
    added=(dc['mask']&~dc['original_mask']).numpy()[0,0]
    subset=(truth==9)&domains['E_original_free'][0]&~added
    outside=dict(arm='D',tp=int(((d==9)&subset).sum()),voxels=int(subset.sum()),coverage=ratio(int(((d==9)&subset).sum()),int(subset.sum())))
    quality,classes,matrix=conversion(dc['values'].numpy()[0,0],truth,added)
    write_csv(PHYS/'metrics_D.csv',metrics);write_csv(PHYS/'shape_metrics_D.csv',[shape]);write_csv(PHYS/'outside_added_wells_D.csv',[outside]);write_csv(PHYS/'condition_quality_D.csv',[dict(arm='D',**quality)]);write_csv(PHYS/'well_classes_D.csv',classes)
    np.savetxt(PHYS/'well_confusion_D.csv',matrix,fmt='%d',delimiter=',')
    result=dict(status='completed',shape=shape,outside=outside,condition_quality=quality,metrics=metrics,seismic_relative_errors={**inv['relative_seismic_errors'],'D_hard_remapped':derror},D_hard_forward=forward_asset,underground_air_D=int(((dp==-1)&sub).sum()),D_prediction=r['prediction'],D_velocity_evaluations=32,total_velocity_evaluations=128,total_flow_outputs=4,physical_inversion_calls=1,sigma1_deconvolution_after_seismic=False,condition_violations=0,mask_B_C_D_equal=True,truth_used_only_after_D=True,completed_at_utc=utc())
    write_json(PHYS/'EVALUATION.json',result)
    approved=standard();approved.namespace['TITLES']=['Ground truth','Frozen Flow A','Seismic-well Flow D']
    cams=approved.CONFIG['cameras'];wells=read_json(ROOT/'layout.json')['original_wells'];out=PHYS/'figures';out.mkdir();(out/'panels').mkdir()
    arrays=[truth,ap.numpy()[0,0],d];keep=np.ones_like(truth,dtype=bool);cut=approved.cutaway_mask(truth.shape,.52)
    images=[];checks=[]
    for row in range(2):
        for arm,v in zip(('Truth','A','D'),arrays):
            im,check=approved.render(v,cut if row else keep,cams['geology']);save_panel(im,out/'panels'/f'geology_{arm}_r{row+1}');images.append(im);checks.append(dict(arm=arm,kind='geology',row=row,**check))
    for clean in (False,True):approved.compose(images,out,'full_geology',clean)
    images=[]
    for arm,v in zip(('Truth','A','D'),arrays):
        im,check=approved.render(v,keep,cams['target'],wells);save_panel(im,out/'panels'/f'target_{arm}');images.append(im);checks.append(dict(arm=arm,kind='target',**check))
    for clean in (False,True):approved.compose(images,out,'target',clean)
    sections(arrays,['Ground truth','Frozen Flow A','Seismic-well Flow D'],approved,out/'fig03_section_Y26')
    write_json(out/'RENDER_CONFIG.json',dict(standard='geology_approved_v1',source_sha256=physics_sources(),standard_sha256=file_hash(STANDARD/'MANIFEST.json'),cameras=cams,checks=checks,visual_inspection='pending',all_components_retained=True,model_calls_for_plotting=0))
    (out/'CAPTIONS.md').write_text('真值/A/D独立三列。D为一次单成员模型基础log阻抗反演形成完整伪井后的冻结Flow；与Gaussian恢复伪井C是不同证据链。所有分量保留、无修补，approved_v1相同风格，中心剖面固定Y26。合成叠后逐柱卷积、人工声学对比、无额外噪声、同算子inverse-crime；并非实测/全波形/3D传播。速度未被反演，时深关系仍依赖A慢度。数字及物理一致性另见../REPORT.md。')
    (PHYS/'REPORT.md').write_text(f'''# 可选地震：单成员固定操作点

主实验MANDATORY_COMPLETE之后才启动。本次只调用一次既有invert_acoustic_member，无fixed12构建、无后验统计、无参数扫描，D仅一次32前向，seed42和主模型/源不变。总Flow4份、128前向，Gaussian恢复1次、物理反演1次。

正演使用新真值经既有码本得到的阻抗/慢度；spacing={SPACING:.9f}米，固定8ms采样、25Hz Ricker/128ms波子、400个时间点（3192ms），记录窗由全网格深度和最低岩石码本速度预设，不以真值速度缩窗。地表下mask只来自原条件中公开空气信息。

反演只读取保存的A及原井/地表、地震观测。FP64直接Cholesky求解原线性化log阻抗Tikhonov操作点，prior_relative_weight=0.001、vertical_smoothness_relative_weight=0.01。A慢度不更新，时深映射不使用真值速度；不能称反演得到速度。旧配置的fixed12 posterior_statistics字符串仅为旧parser所需兼容字段，实际成员数1且不调用任何posterior_statistics。

用唯一真正更新的log阻抗按raw0…13码本最近值、首argmin分类，再在与B/C完全相同512布局形成完整多岩性伪井。未再次Gaussian去卷积，未把未更新慢度当第二证据通道。

相对观测L2误差：A清理后物性先验={inv['relative_seismic_errors']['prior']:.6f}，连续反演后={inv['relative_seismic_errors']['inverted']:.6f}，D硬岩性重映射物性再正演={derror:.6f}。这些分别对应连续物性拟合与最终离散生成物理一致性，不能互相替代。

D完整目标IoU={shape['target_iou']:.6f}，ASSD={distance if distance is not None else 'NA'}米，体积比={shape['volume_ratio']:.6f}。新增井外正确覆盖={outside['tp']}/{outside['voxels']}。井上全类准确率={quality['accuracy']:.6f}，目标TP/FP/FN={quality['target_tp']}/{quality['target_fp']}/{quality['target_fn']}。完整固定域数字见metrics_D.csv，A/B/C参考主目录CSV，不合并两种方法。

这是合成叠后卷积正演—模型基础反演：各柱独立，不含绕射、复杂照明、多路径及三维波场传播；人工目标声学对比、无额外噪声、同算子生成与反演均是理想inverse-crime设定。反演先验慢度来自不准确A，其误差会限制阻抗修正和岩性转换。不能宣称真实地震全面验证或可靠精确形态恢复。

观测生成耗时见OBSERVATION_FREEZE.json，反演耗时{inv['elapsed_seconds']:.3f}秒，D采样/加载审计{r['elapsed_seconds']:.3f}秒。源/配置快照、完整物性和地震场、条件、receipts、CSV和高清单面板均保留；独立真值/A/D成图见figures/，不改主实验C图。
''')
    print('D_EVALUATED_FIGURES_COMPLETE',shape,result['seismic_relative_errors'],flush=True)
if __name__=='__main__':main()
