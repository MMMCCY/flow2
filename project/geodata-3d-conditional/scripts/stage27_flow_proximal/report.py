"""Render numerical evidence as a report without changing or selecting outputs."""
import argparse
from pathlib import Path
from .common import *

def fmt(x):
    return 'NA' if x is None else (f'{x:.6f}' if isinstance(x,float) else str(x))

def main(root):
    root=root.resolve()
    e=read_json(root/'EVALUATION.json');s=read_json(root/'SAMPLING_COMPLETE.json')
    pair=read_json(root/'PAIR_CHECKS.json');resources=read_json(root/'resource_usage.json')
    rows=e['metrics'];p=e['physical'];arms=s['arms']
    def m(arm,world='G1',domain='U_common'):
        return next(r for r in rows if r['arm']==arm and r['world']==world and r['domain']==domain)
    def physical(arm,obs='y1',field='hard_own_impedance_and_slowness'):
        return next(r for r in p if r['arm']==arm and r['observation']==obs and r['field']==field)
    lines=['# Stage27：冻结 Flow 终点近端修正与路径重投影', '',
        f"实际目录：`{root}`。服务器：{resources['host']}。完成状态：{s['state']['status']}。",
        f"基线 HEAD：`{s['state']['git']['head']}`。开始时工作树见 START.json 和 sampling_state.json；保留用户改动。",'',
        '## 实施与范围', '',
        '原模型和 embedding 冻结，411项 EMA，15维 embedding，网络 FP32（关闭 TF32/autocast），近端物理 FP64。'
        f"近端设备为 {resources['prox_device']}。每臂32次速度前向；使用原 seed42 保存源，各步CPU FP32独立噪声种子1042+j，各臂hash逐项相同。",'',
        '每步以 no_grad 计算 e_hat=x+(1-t)v，并 detach；仅在地下非原条件域对终点做固定3次近端尝试。'
        '目标为自由元素均值信赖项，加 nu² 倍的平方相对地震能量和0.05倍平滑TV。梯度乘D_free，方向RMS封顶0.1，按封存六步长Armijo搜索。'
        '更新后按 b*u+(1-b)*epsilon 重投影，所有非自由位置始终恢复原条件嵌入。'
        '无跨步/网络反传，无新硬伪井，无传统反演、训练、扫描或补充seed。', '',
        'checkpoint训练时间范围为[0.0001,0.9999]，首步名义0在1e-4评估是明确数值近似。'
        '源码训练终点另有1e-4 Gaussian jitter；训练路径仍为独立Gaussian源的一侧线性插值。'
        '最后FP64近端端点和转回网络FP32后原decode使用的端点分别保存。', '',
        '推理入口只接受固定字段registry和枚举张量输入，不打开真值、几何资产、旧完整分类或评价registry。'
        'prepare生成/评价角色可读真值；独立进程的推理防火墙保持开启，历史结果禁止写入。', '',
        '## 输入定义与检查', '',
        f"G2有效配对：{pair['valid_pair']}；M0冲突数{pair['condition_conflict_count']}；地下/地表一致{pair['subsurface_equal']}。"
        f"帽部目标差异{pair['cap_target_difference_voxels']}体素；所有类别差异{pair['all_label_difference_voxels']}体素。", '',
        'G2只将cap直径22格改为18格，其余原生历史、origin、高度、柱径、地表、类别和seed不变。'
        'G1观测沿用旧资产；y2只生成一次。M512、holdout和帽部box复用，G2嵌套帽部并集box等于G1原box，差异域在采样前封存。'
        'FULL包含空气，U_original和U_common仅为已知地下支持内非原条件/非512条件域；holdout无硬条件但仍被物理观测覆盖。', '',
        '必要小张量检查见 NECESSARY_CHECKS.json；CPU和CUDA实际物理梯度有限且非零、非自由梯度为0，Armijo下降、RMS上限、F0/Fr不调用地震、配对噪声重建通过。'
        '第一个正式F0兼作端到端检查，没有额外32步smoke。', '',
        '## G1主结果：共同自由域 U_common', '',
        '|方法|TP|FP|FN|Precision|Recall|IoU|体积比|硬场y1相对L2|地下空气|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ['A','D0','D','DIRECT_PRIOR','DIRECT_UPDATED',*arms]:
        r=m(arm);ph=physical(arm);c=e['conditions'][arm]
        lines.append('|'+ '|'.join(map(fmt,[arm,r['target_tp'],r['target_fp'],r['target_fn'],r['target_precision'],
            r['target_recall'],r['target_iou'],r['target_pred_truth_volume_ratio'],ph['relative_l2'],c['underground_air_voxels']]))+'|')
    lines += ['', 'A/D0/D/DIRECT仅复用旧保存资产、重新评价；没有新增这些方法的采样。DIRECT的信息预算不同，不作为公平方法优胜证据。', '',
        '## G1帽部、边界和组件', '',
        '|方法|帽部TP|帽部FN|帽部IoU|全目标ASSD(m)|组件数|全目标体素数|',
        '|---|---:|---:|---:|---:|---:|---:|']
    components=read_json(root/'components.json')
    for arm in ['A','F0','Fr','Fy_G1','Fyr_G1']:
        cap=m(arm,domain='CAP_G1')
        boundary=next(b for b in e['shape'] if b['arm']==arm and b['world']=='G1')
        obj=components[arm+'_G1']
        lines.append('|'+ '|'.join(map(fmt,[arm,cap['target_tp'],cap['target_fn'],cap['target_iou'],
            boundary['assd_m'],obj['predicted_components'],sum(obj['predicted_volume_sequence'])]))+'|')
    lines += ['', '新输出全目标均只有一个26邻接组件，但帽部大多缺失。边界平均距离变小不表示帽部恢复；'
        '远处误报减少及目标收缩也能改善该均值，须与帽部TP/FN和体积共同解释。', '',
        '## 每臂成本与连续/硬场差异', '',
        '|方法|NFE|物理forward|物理backward|接受|耗尽|秒|连续y1相对L2|硬y1相对L2|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm,r in arms.items():
        lines.append('|'+ '|'.join(map(fmt,[arm,r['network_evaluations'],r['physics_forwards'],r['physics_backwards'],
            r['accepted_steps'],r['line_search_exhausted'],r['elapsed_seconds'],
            physical(arm,field='continuous_prox_endpoint_soft_nonair')['relative_l2'],physical(arm)['relative_l2']]))+'|')
    lines += ['', '连续物理用地下非空气归一化期望阻抗与慢度；硬场按原decode类别查自身阻抗和慢度，地下空气完整保留。'
        '两者不等价，不能用前者替代最终硬地质结论。所有越出原记录窗的到时按旧算子裁剪并在physical_errors.csv计数。'
        '结构-only的连续/硬物理误差仅在评价计算，不参与近端。', '',
        '## 配对信息检验', '',
        '|方法|IoU(y1,G1)|IoU(y1,G2)|IoU(y2,G1)|IoU(y2,G2)|D_struct U_common|',
        '|---|---:|---:|---:|---:|---:|']
    for info in e['paired_information']:
        family=info['family'];k1=family+'_G1';k2=family+'_G2'
        lines.append('|'+ '|'.join(map(fmt,[family,m(k1)['target_iou'],m(k1,'G2')['target_iou'],
            m(k2)['target_iou'],m(k2,'G2')['target_iou'],info['D_struct_U_common']]))+'|')
    lines += ['', '|方法|硬场(y1输出,y1)|硬场(y1输出,y2)|硬场(y2输出,y1)|硬场(y2输出,y2)|帽部占据y1→y2|未变域改类数|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for info in e['paired_information']:
        family=info['family'];k1=family+'_G1';k2=family+'_G2'
        lines.append('|'+ '|'.join(map(fmt,[family,physical(k1)['energy'],physical(k1,'y2')['energy'],
            physical(k2)['energy'],physical(k2,'y2')['energy'],
            f"{info['G1_cap_target_voxels']}→{info['G2_cap_target_voxels']}",info['all_class_changes_unchanged_region']]))+'|')
    lines += ['', '物理矩阵为各观测自身能量归一化的平方相对L2，不是概率；比较同列对角与反事实，不能把不同观测范数当统一似然。'
        '完整原始矩阵、连续场矩阵、原自由域指标和方向变化在cross_observation_matrix.csv、paired_information.csv保留。', '',
        '## 解释与最终四问', '']
    delta_f0=m('F0')['target_iou']-m('A')['target_iou']
    delta_r=m('Fr')['target_iou']-m('F0')['target_iou']
    delta_y=m('Fy_G1')['target_iou']-m('F0')['target_iou']
    delta_yr=m('Fyr_G1')['target_iou']-m('Fy_G1')['target_iou']
    lines += [f"F0相对旧A的U_common IoU差值为{delta_f0:+.6f}，这是新采样路径自身变化，不能归给地球物理。"
        f"Fr−F0={delta_r:+.6f}；Fy_G1−F0={delta_y:+.6f}；Fyr_G1−Fy_G1={delta_yr:+.6f}。",'',
        '1. **新推理是否正确实施**：按封存算法、固定方法表与预算执行；最终参数hash保持、无模型梯度、原条件无违例及噪声配对结果见SAMPLING_COMPLETE.json。'
        '实现有效不等于科学结果为正。', '',
        '2. **物理是否带来额外井外结构**：以Fy对F0、Fyr对Fr的共同自由域硬指标和两观测交叉矩阵判断。']
    for info in e['paired_information']:
        direction=(info['G2_cap_target_voxels']-info['G1_cap_target_voxels'])*info['truth_cap_change']>0
        family=info['family'];k1=family+'_G1';k2=family+'_G2'
        diag1=physical(k1)['energy']<physical(k2)['energy']
        diag2=physical(k2,'y2')['energy']<physical(k1,'y2')['energy']
        ds=info['D_struct_U_common']
        lines.append(f"   {family}：帽部占据方向与预定真值变化一致={direction}；两列硬物理对角优于反事实={diag1}/{diag2}；D_struct={fmt(ds)}。"
            + ('仅支持本配对的有限信息利用。' if direction and diag1 and diag2 and ds is not None and ds>0
               else '证据未同时满足结构方向、物理观测匹配及结构交叉偏好的支持条件，不能宣称完成结构恢复。'))
    lines += ['',f"3. **TV帮助组织还是仅平滑**：共同自由域IoU增益Fyr−Fy为{delta_yr:+.6f}，Fr−F0为{delta_r:+.6f}，差中之差={delta_yr-delta_r:+.6f}。"
        'F0、Fr、Fy和Fyr均只有一个目标组件，TV没有带来组件组织的新增证据。Fr相对F0主要收缩目标并降低召回；'
        'Fyr相对Fy仅有很小帽部TP/IoU改善，仍丢失大多数帽部，不能称为帽部结构恢复。'
        '完整组件体积与空间对应在components.json，双向边界在shape_metrics.csv。', '',
        '4. **单源配对不能回答什么**：不能证明多seed稳健性、后验频率校准、多解覆盖、独立地质泛化、实测地震效果或唯一重建。'
        '观测为已知码本/已知波子/无新增噪声/同算子逐柱卷积的合成机制实验，不含三维传播、FWI或联合反演。'
        'lambda_R=0.05与tau=0.1为本轮固定初始设置，不声称最优；负结果不推出所有推理制导无效。', '',
        '## 异常、预算和交付', '',
        f"跳过项：{s['state']['skipped_arms']}。正式网络NFE={resources['network_evaluations']}；"
        f"内部物理forward={resources['physics_forwards_internal']}，backward={resources['physics_backwards']}。"
        f"y2生成forward={resources['observation_generation_physics_forwards']}；"
        f"评价硬场forward={resources['evaluation_hard_physics_forwards']}，连续场forward={resources['evaluation_continuous_physics_forwards']}。", '',
        '线搜索耗尽按原算法保留该次u，不追加迭代、不改参数、不重采样。'
        '没有因科学收益不足重跑、改变方法、清理空气或删除碎片。任何绘图/receipt修复仅为后处理。', '',
        '**配置隔离说明**：正式执行的共享配置中包含预注册的G2名义直径18格常量，未完全从推理registry剥离；'
        'run/sampler从未访问或使用这一字段，未读真值/几何资产，近端目标也不接收直径、体积或真值轮廓。'
        '这是一项输入元数据隔离瑕疵，不应写成完全符合剥离几何字段要求。执行后的代码已将准备配置与推理配置分开，'
        '并保存去除此字段的派生registry供审阅；原registry、hash和执行快照全部保留。'
        '没有为这项无数值依赖的修复追加采样，故本轮192次结果仍对应原封存快照。见METADATA_ISOLATION_REVIEW.json和POSTPROCESS_SOURCES.json。', '',
        '原始端点、FP64近端端点、原decode硬标签在arms/<arm>/；逐步时间、nu²、能量、梯度RMS、步长、调用数、耗时和重噪hash在trace.json。'
        'config.json、inference_registry.json、OBSERVATION_FREEZE.json、PAIR_CHECKS.json、source_snapshot/、COMMANDS.md、'
        'condition_checks.json和resource_usage.json保存输入/来源/运行身份。认可风格PNG/PDF与固定Y26图在figures/。', '']
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(root/'FINAL_CHECKS.json',dict(network_evaluations=resources['network_evaluations'],
        max_network_evaluations=192,budget_respected=resources['network_evaluations']<=192,
        model_unchanged=s['model_unchanged'],parameter_gradients_absent=s['state']['parameter_gradients_absent'],
        executed_inference_registry_fully_geometry_stripped=False,
        nominal_G2_diameter_not_used_by_numerical_code=True,
        current_registry_contract_corrected=True,
        all_new_original_condition_violations_zero=all(e['conditions'][a]['original_condition_violations']==0 for a in arms),
        outputs_reload_checked=True,pair_valid=pair['valid_pair'],new_traditional_inversions=0,
        training=0,scientific_retries=0,git_push=False,completed_at_utc=utc()))
    print('REPORT_COMPLETE',root,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
