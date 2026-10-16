#!/usr/bin/env python3
"""Frozen Stage15-v4 / Stage19R paired ensemble workflow on a shared TEST case.

No training, parameter search or hidden truth in sampling / observation ranking.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
for path in (PROJECT,REPO): sys.path.insert(0,str(path))
import inference_runtime as runtime
import ensemble_occurrence as occurrence
from scripts.stage15.common import read_json, write_json, write_csv, normalize_volume, base_manifest
from scripts.stage19.common import validate_asset, freeze_base_model
from guidance.binary_seismic_inversion import binary_acoustic_properties_from_configs, binary_occupancy_to_acoustic
from guidance.seismic import seismic_operator_from_config, tensor_sha256
from guidance.property_sampling import fixed_euler_property_sample
from guidance.property_volume import property_table_from_config
from guidance.residual_velocity_adapter import ResidualVelocityAdapter, fixed_euler_adapter_sample
from scripts.stage19.run_inference import load_validated_adapter_checkpoint, _parameter_hash

S19 = PROJECT/'experiments/stage19_learned_evidence_adapter'
S15 = PROJECT/'experiments/stage15_binary_seismic_consensus'
METHODS = ('STAGE15_TRACE_V4','STAGE19R_ADAPTER')
CASE = 'stage19_test_case004'
BASE_HASH = '561e94bfda770ec41fc4cbed43436a7e2130eef5dfb7e5d666fcefc0724ff94c'


def atomic_json(path,value):
    path=Path(path); tmp=path.with_suffix('.pending.json'); write_json(tmp,value); tmp.replace(path)


def load_hard(path, offset=None):
    """Raw hard-label .pt/.pt.gz chunk reader: offset is mandatory for batches."""
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rb') as f: value=torch.load(f,map_location='cpu',weights_only=False)
    else: value=runtime.load_tensor(path)
    if value.ndim in (4,5):
        if offset is None:
            if value.shape[0]!=1: raise ValueError('batch chunk requires explicit offset')
            offset=0
        if not 0<=offset<value.shape[0]: raise ValueError('chunk offset out of range')
        value=value[offset]
    if value.ndim==4 and value.shape[0]==1: value=value[0]
    if value.ndim!=3 or not torch.isfinite(value).all() or not torch.equal(value,value.round()): raise ValueError('invalid hard XYZ model')
    if value.min() < -1 or value.max() > 13: raise ValueError('raw labels must be -1..13')
    return value.long().numpy()


def audit_assets(case_id):
    registry_path=S19/'evidence_v2/test_inference_registry.json'
    text=registry_path.read_text().lower()
    if any(word in text for word in ('truth','true_model','binary_truth')): raise ValueError('inference registry firewall failure')
    registry=read_json(registry_path)
    if registry['run_status']!='completed': raise ValueError('incomplete registry')
    case=next(c for c in registry['cases'] if c['case_id']==case_id)
    inputs={name:validate_asset(case[name],name) for name in ('condition_values','condition_mask','subsurface_mask','binary_impedance_score','observed_seismic','observation_manifest')}
    configs={
        'stage15_reference':S15/'trace_boundary/flow_property_seed42_v4/guided/config.json',
        'property':S15/'configs/binary_trace_property_indicator_v1.json',
        'seismic':PROJECT/'experiments/stage4_seismic/configs/full_cube_noiseless_inverse_crime_v1.json',
        'binary':S15/'configs/binary_acoustic_upper_bound_v1.json',
        'training':S19/'configs/training_v1.json',
        'inference':S19/'configs/inference_v1.json',
        'training_manifest':S19/'checkpoints/formal_v2/training_manifest.json',
        'adapter':S19/'checkpoints/formal_v2/adapter_checkpoint.pt',
        'checkpoint':PROJECT/'demo_model/conditional-weights.ckpt',
        'registry':registry_path,
        'stage19_evidence_config':S19/'configs/evidence_v2.json',
    }
    binary=read_json(configs['binary']); source=Path(binary['source_acoustic_config']['path'])
    if not source.is_absolute(): source=REPO/source
    configs['source_acoustic']=source
    from scripts.stage19.common import validate_stage17a_reuse
    validate_stage17a_reuse(read_json(configs['stage19_evidence_config']))
    assets={name:runtime.asset_record(path) for name,path in {**inputs,**configs}.items()}
    if assets['checkpoint']['sha256']!=BASE_HASH: raise ValueError('base checkpoint SHA mismatch')
    payload=load_validated_adapter_checkpoint(configs['training_manifest'],configs['adapter'],configs['training'],read_json(configs['training']),require_formal=True)
    del payload
    metadata=read_json(inputs['observation_manifest'])['operator_metadata']
    if metadata['axis_order']!=['x','y','z'] or metadata['cell_size_m']!=[100.,100.,50.]: raise ValueError('unexpected geometry')
    return inputs,configs,assets,metadata


def make_protocol(case_id,methods,selection):
    inputs,configs,assets,metadata=audit_assets(case_id)
    seeds=[42+100*i for i in range(200)]
    source=[dict(sample_id=f'source_{i:03d}',source_seed=seed,candidate_order=i) for i,seed in enumerate(seeds)]
    selection_record={}
    if selection=='historical_adapter_median_iou_best':
        metrics_path=S19/'reports/formal_v2/sample_metrics.csv'
        with metrics_path.open(newline='') as f: old_rows=list(csv.DictReader(f))
        case_scores={}
        for row in old_rows:
            if row['arm']=='ADAPTER_CORRECT': case_scores.setdefault(row['case_id'],[]).append(float(row['target_iou']))
        if len(case_scores)!=12 or any(len(v)!=3 for v in case_scores.values()): raise ValueError('historical selection requires 12 x 3 unique cases')
        ranking=sorted(case_scores,key=lambda key:(-float(np.median(case_scores[key])),key))
        if case_id!=ranking[0]: raise ValueError('case differs from historical median IoU selection')
        selection_record=dict(metrics=runtime.asset_record(metrics_path),ranking=[dict(case_id=key,median_target_iou=float(np.median(case_scores[key]))) for key in ranking])
    protocol=base_manifest('ensemble_200_frozen_v1',Path(__file__),configs['inference'])
    protocol.update(case_id=case_id,methods=list(methods),source_ids=source,planned_count=200,
        source_rule='independent CPU torch.Generator().manual_seed(42+100*i); float32 standard Gaussian; paired identical tensor',
        selection=selection,selection_record=selection_record,case_selection_uses_historical_truth_metrics=selection=='historical_adapter_median_iou_best',
        checkpoint_hash=BASE_HASH,assets=assets,geometry=metadata,
        stage15_parameters=read_json(configs['stage15_reference']),stage19_parameters=read_json(configs['inference']),
        training_performed_by_this_run=False,stage19_uses_pretrained_evidence_adapter=True,
        stage15_transfer='unchanged v4 positive-endpoint/confidence mapping on shared Stage19R TEST assets',
        observation_score=dict(name='hard_seismic_rmse',unit='unscaled_convolutional_amplitude',direction='lower',normalization='none',noise='none',inverse_crime=True),
        truth_loaded_by_sampling=False,source_files={str(p.relative_to(REPO)):runtime.file_sha256(p) for p in [Path(__file__),REPO/'ensemble_occurrence.py',PROJECT/'guidance/property_sampling.py',PROJECT/'guidance/residual_velocity_adapter.py',PROJECT/'guidance/seismic.py',PROJECT/'guidance/binary_seismic_inversion.py']})
    return protocol,inputs,configs


def sample(args):
    protocol,inputs,configs=make_protocol(args.case,args.methods,args.case_selection)
    if args.resume:
        old=read_json(args.out/'protocol.json')
        for key in ('case_id','methods','source_ids','assets','source_files','selection','selection_record'):
            if old[key]!=protocol[key]: raise ValueError(f'resume mismatch: {key}')
        protocol=old
    else:
        if args.out.exists(): raise FileExistsError(args.out)
        args.out.mkdir(parents=True); write_json(args.out/'protocol.json',protocol)
        write_json(args.out/'source_ids.json',protocol['source_ids'])
        write_json(args.out/'failures.json',[])
    state_path=args.out/'sampling_state.json'
    state=read_json(state_path) if state_path.exists() else dict(status='running',records=[],failed_ids=[])
    failures=json.loads((args.out/'failures.json').read_text())
    if state['failed_ids']: raise RuntimeError('failed source IDs retained; repair requires explicit new decision, no replacement seeds')
    device=torch.device(args.device)
    if device.type!='cuda' or not torch.cuda.is_available(): raise RuntimeError('sampling requires available CUDA')
    from model_train_sh_inference_cond import Geo3DStochInterp
    model,report=runtime.load_model_with_weight_policy(model_class=Geo3DStochInterp,checkpoint_path=configs['checkpoint'],map_location=device,weight_source='ema')
    model=model.to(device).eval(); freeze_base_model(model); before=_parameter_hash(model)
    tcfg=read_json(configs['training']); icfg=read_json(configs['inference']); scfg=read_json(configs['stage15_reference'])
    adapter=ResidualVelocityAdapter(model.embedding_dim,geophysics_channels=1,base_width=tcfg['adapter']['base_width'],dilations=tcfg['adapter']['dilations']).to(device)
    payload=load_validated_adapter_checkpoint(configs['training_manifest'],configs['adapter'],configs['training'],tcfg,require_formal=True,map_location=device)
    adapter.load_state_dict(payload['adapter_state_dict']); adapter.eval()
    for param in adapter.parameters(): param.requires_grad_(False)
    tensors={name:normalize_volume(runtime.load_tensor(inputs[name]),name) for name in ('condition_values','condition_mask','subsurface_mask','binary_impedance_score')}
    values=tensors['condition_values'].long(); mask=tensors['condition_mask'].bool(); support=tensors['subsurface_mask'].bool(); score=tensors['binary_impedance_score'].float()
    with torch.no_grad(): embedded=model.embed(values.to(device)); conditioning=embedded*mask.to(device).expand_as(embedded)
    table,weights,_=property_table_from_config(read_json(configs['property']),model.num_categories)
    q=(score*support.float()).to(device); confidence=(score*(support&~mask).float()).to(device)
    # Exactly Stage15 v4: target endpoint throughout support, air endpoint outside.
    target=support.float().to(device)
    operator,_=seismic_operator_from_config(read_json(configs['seismic']),grid_shape=tuple(model.data_shape))
    properties=binary_acoustic_properties_from_configs(read_json(configs['binary']),read_json(configs['source_acoustic']))
    observed=runtime.load_tensor(inputs['observed_seismic']).to(device)
    keys=('max_guidance_ratio','tau_start','tau_end','tau_schedule','guidance_start','guidance_schedule','grad_clip_norm','guidance_scaling_mode','property_sigmas','property_scale_weights')
    settings={key:scfg[key] for key in keys}
    def property_sample(initial,alpha):
        return fixed_euler_property_sample(model=model,initial_state=initial,conditioning=conditioning,embedded_truth=embedded,truth_model=values.to(device),condition_mask=mask.to(device),target_properties=target,property_table=table.to(device),confidence=confidence,property_channel_weights=weights.to(device),n_steps=scfg['n_steps'],alpha=alpha,loss_mode=scfg['property_loss_mode'],**settings)
    expected_arms=['FLOW_ONLY']+list(args.methods)
    for item in protocol['source_ids'][:args.limit]:
        sid=item['sample_id']; seed=item['source_seed']; start=time.monotonic()
        existing={r['arm']:r for r in state['records'] if r['sample_id']==sid}
        for arm,record in existing.items():
            if runtime.file_sha256(args.out/record['decoded_path'])!=record['decoded_file_sha256']: raise ValueError('saved model corrupted')
        if set(existing)==set(expected_arms): continue
        initial_cpu=torch.randn((1,model.embedding_dim,*model.data_shape),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=embedded.dtype).contiguous()
        noise_sha=tensor_sha256(initial_cpu)
        try:
            for arm in expected_arms:
                if arm in existing: continue
                if arm=='STAGE15_TRACE_V4': final,trace=property_sample(initial_cpu.to(device),scfg['alpha'])
                else:
                    final,trace=fixed_euler_adapter_sample(model=model,adapter=adapter,initial_state=initial_cpu.to(device),conditioning=conditioning,embedded_conditions=embedded,condition_mask=mask.to(device),geophysics=q,n_steps=icfg['n_steps'],adapter_scale=0.0 if arm=='FLOW_ONLY' else icfg['adapter_scale'],max_residual_ratio=icfg['max_residual_ratio'])
                if not torch.isfinite(final).all(): raise ValueError('nonfinite final state')
                with torch.no_grad(): decoded=(model.decode(final).detach().cpu()-1).unsqueeze(1).long()
                violations=int(((decoded!=values)&mask).sum())
                if violations: raise ValueError(f'condition violations {violations}')
                folder=args.out/'samples'/sid/arm; folder.mkdir(parents=True,exist_ok=True)
                if (folder/'decoded_geology.pt').exists():
                    # A crash after file creation but before state commit preserves the orphan.
                    raise RuntimeError(f'uncommitted output retained at {folder}; refusing overwrite')
                torch.save(decoded,folder/'decoded_geology.pt'); write_csv(folder/'sampling_trace.csv',trace)
                with torch.no_grad():
                    impedance,slowness=binary_occupancy_to_acoustic((decoded.to(device)==9).float(),support.to(device),properties)
                    prediction=operator(impedance,slowness,support.to(device)); rmse=float((prediction-observed).square().mean().sqrt())
                if not np.isfinite(rmse): raise ValueError('nonfinite forward score')
                torch.save(prediction.cpu(),folder/'hard_predicted_seismic.pt')
                record={**item,'hard_prediction_file_sha256':runtime.file_sha256(folder/'hard_predicted_seismic.pt'),'arm':arm,'initial_noise_sha256':noise_sha,'decoded_tensor_sha256':tensor_sha256(decoded),'decoded_path':str((folder/'decoded_geology.pt').relative_to(args.out)),'decoded_file_sha256':runtime.file_sha256(folder/'decoded_geology.pt'),'hard_seismic_rmse':rmse,'condition_violations':violations}
                state['records'].append(record); atomic_json(state_path,state)
                if arm=='FLOW_ONLY' and item['candidate_order']==0:
                    reference,_=property_sample(initial_cpu.to(device),0.0)
                    ref=(model.decode(reference).detach().cpu()-1).unsqueeze(1).long()
                    if not torch.equal(ref,decoded): raise ValueError('independent property alpha-zero / adapter scale-zero mismatch')
                    state['independent_baseline_equivalence']=True
                if arm in ('FLOW_ONLY','STAGE19R_ADAPTER') and seed in (42,142,242):
                    old_arm='FLOW_ONLY' if arm=='FLOW_ONLY' else 'ADAPTER_CORRECT'
                    old_path=S19/'formal/inference_v2'/args.case/f'seed_{seed}'/old_arm/'decoded_geology.pt'
                    if not np.array_equal(load_hard(old_path),decoded[0,0].numpy()): raise ValueError(f'historical replay mismatch: {seed}/{arm}')
                    record['historical_replay_exact']=True; atomic_json(state_path,state)
                print(f'{sid} seed={seed} {arm} hard_rmse={rmse:.6f}',flush=True)
            print(f'{sid} all arms elapsed={time.monotonic()-start:.1f}s',flush=True)
        except Exception as exc:
            failure=dict(sample_id=sid,source_seed=seed,error=f'{type(exc).__name__}: {exc}')
            failures.append(failure); state['failed_ids'].append(sid); state['status']='failed'
            atomic_json(args.out/'failures.json',failures); atomic_json(state_path,state); raise
    after=_parameter_hash(model)
    if before!=after: raise ValueError('base model changed')
    count=sum(set(r['arm'] for r in state['records'] if r['sample_id']==item['sample_id'])==set(expected_arms) for item in protocol['source_ids'])
    state.update(status='completed' if count==200 else 'partial_diagnostic',complete_source_count=count,base_model_unchanged=True,base_hash_before=before,base_hash_after=after,model_load_report=report)
    atomic_json(state_path,state); write_csv(args.out/'sample_manifest.csv',state['records'])


def topology(mask):
    from scipy import ndimage
    labels,n=ndimage.label(mask,structure=ndimage.generate_binary_structure(3,1))
    sizes=np.bincount(labels.ravel())[1:]
    return dict(connected_components_6=int(n),largest_component_fraction=float(sizes.max()/sizes.sum()) if sizes.size else None)


def summarize_mask(mask,truth,domains,spacing):
    return {name:{**occurrence.metrics(mask,truth,domain),**topology(mask&domain),
        'physical_volume_m3':float((mask&domain).sum()*np.prod(spacing))} for name,domain in domains.items()}


def postprocess(args):
    protocol=read_json(args.out/'protocol.json'); state=read_json(args.out/'sampling_state.json')
    if state['failed_ids']: raise ValueError('failed planned candidates; inspect failures.json')
    inputs,configs,assets,metadata=audit_assets(protocol['case_id'])
    if assets!=protocol['assets']: raise ValueError('input assets changed')
    rows=state['records']; planned=protocol['source_ids']; methods=protocol['methods']
    arms=['FLOW_ONLY']+methods
    complete=[x for x in planned if set(r['arm'] for r in rows if r['sample_id']==x['sample_id'])==set(arms)]
    n=len(complete)
    if n!=200 and not args.diagnostic: raise ValueError(f'only {n}/200 complete; use --diagnostic for actual-N outputs')
    if n==0: raise ValueError('no completed independent draws')
    output=args.report_dir or args.out/'postprocess_v1'
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    ids=[x['sample_id'] for x in complete]; keep=min(100,n); spacing=metadata['cell_size_m']
    label_arrays={}; group_ids={}; score_index={}; group_indices={}
    # Write all observation-selected IDs before reading evaluation registry/truth.
    for arm in arms:
        indexed={r['sample_id']:r for r in rows if r['arm']==arm}
        array=np.lib.format.open_memmap(output/f'{arm}_labels.npy',mode='w+',dtype=np.int8,shape=(n,64,64,64))
        scores=[]
        for i,sid in enumerate(ids):
            r=indexed[sid]
            if runtime.file_sha256(args.out/r['decoded_path'])!=r['decoded_file_sha256']: raise ValueError('model hash mismatch')
            array[i]=load_hard(args.out/r['decoded_path']); scores.append(r['hard_seismic_rmse'])
        array.flush(); label_arrays[arm]=array
        scores=np.asarray(scores); score_index[arm]=scores
        groups={f'all{n}':np.arange(n),f'random{keep}':occurrence.select_indices(ids,'random',keep),f'obs_top{keep}':occurrence.select_indices(ids,'physics',keep,scores)}
        group_indices[arm]=groups; group_ids[arm]={name:[ids[i] for i in idx] for name,idx in groups.items()}
        write_json(output/f'{arm}_observation_selected_ids.json',group_ids[arm])
    write_json(output/'inference_selection_complete.json',dict(truth_loaded=False,actual_count=n,keep=keep,score=protocol['observation_score']))
    registry=read_json(S19/'evidence_v2/evidence_registry.json')
    case=next(c for c in registry['cases'] if c['case_id']==protocol['case_id'])
    truth_path=validate_asset(case['truth_assets']['truth'],'retrospective truth')
    truth_labels=load_hard(truth_path); truth=truth_labels==9
    condition=load_hard(inputs['condition_values']); cm=normalize_volume(runtime.load_tensor(inputs['condition_mask']),'mask').bool()[0,0].numpy()
    support=normalize_volume(runtime.load_tensor(inputs['subsurface_mask']),'support').bool()[0,0].numpy()
    domains={'full':np.ones_like(cm),'free_subsurface':support&~cm,'condition':cm}
    for arm,a in label_arrays.items():
        for i in range(n):
            if np.any(a[i][cm]!=condition[cm]): raise ValueError(f'condition violation {arm}/{ids[i]}')
    metrics_rows=[]; summaries={}; candidate_metrics={}
    for arm,a in label_arrays.items():
        candidate_metrics[arm]=[summarize_mask(a[i]==9,truth,domains,spacing) for i in range(n)]
        ious=np.array([x['full']['iou'] for x in candidate_metrics[arm]])
        group_indices[arm][f'oracle_top{keep}']=occurrence.select_indices(ids,'oracle-iou',keep,ious,True)
        summaries[arm]={}
        for i,sid in enumerate(ids):
            present=np.unique(truth_labels[truth_labels!=-1])
            per_class={f'raw_label_{int(label)}_iou':occurrence.metrics(a[i]==label,truth_labels==label)['iou'] for label in present}
            metrics_rows.append(dict(global_voxel_accuracy=float((a[i]==truth_labels).mean()),truth_present_mean_iou=float(np.mean(list(per_class.values()))),**per_class,arm=arm,sample_id=sid,source_seed=complete[i]['source_seed'],hard_seismic_rmse=float(score_index[arm][i]),**{f'{domain}_{key}':value for domain,item in candidate_metrics[arm][i].items() for key,value in item.items()}))
        for name,indices in group_indices[arm].items():
            folder=output/arm/name; folder.mkdir(parents=True)
            occurrence.write_json(folder/'selected_ids.json',[ids[i] for i in indices])
            count=occurrence.counts_for(a,indices,9); cuts=occurrence.minimum_counts(len(indices)); masks=np.array([count>=c for c in cuts])
            np.savez_compressed(folder/'occurrence.npz',count=count,frequency=count/len(indices),threshold_masks=masks,minimum_counts=cuts)
            summaries[arm][name]=dict(n=len(indices),minimum_counts=cuts,
                median_candidate_iou=float(np.median(ious[indices])),median_candidate_hard_seismic_rmse=float(np.median(score_index[arm][indices])),
                threshold_metrics=[summarize_mask(m,truth,domains,spacing) for m in masks])
            member_indices=indices[np.unique(np.linspace(0,len(indices)-1,min(12,len(indices)),dtype=int))]
            write_json(folder/'figure8_ids.json',[ids[i] for i in member_indices])
            if not args.skip_plots: plot_group(folder,arm,name,a,indices,truth,cm,condition,support,spacing,protocol)
        stability={}
        for prefix in (50,100,200):
            if prefix<=n:
                count=occurrence.counts_for(a,np.arange(prefix),9); np.save(output/f'{arm}_prefix{prefix}_frequency.npy',count/prefix)
                stability[str(prefix)]=dict(mean_abs_frequency_difference_from_all=float(np.abs(count/prefix-occurrence.counts_for(a,np.arange(n),9)/n).mean()))
        summaries[arm]['stability']=stability
        summaries[arm]['unique_decoded_models']=len({__import__('hashlib').sha256(a[i].tobytes()).hexdigest() for i in range(n)})
    write_csv(output/'candidate_metrics.csv',metrics_rows)
    paired=[]
    for method in methods:
        for i,sid in enumerate(ids):
            records=[r for r in rows if r['sample_id']==sid and r['arm'] in ('FLOW_ONLY',method)]
            if len(records)!=2 or len({r['initial_noise_sha256'] for r in records})!=1: raise ValueError('false source pairing')
            paired.append(dict(method=method,sample_id=sid,delta_iou=candidate_metrics[method][i]['full']['iou']-candidate_metrics['FLOW_ONLY'][i]['full']['iou'],delta_hard_seismic_rmse=float(score_index[method][i]-score_index['FLOW_ONLY'][i])))
    write_csv(output/'paired_deltas.csv',paired)
    summary=dict(actual_count=n,planned_count=200,protocol_completed=n==200,failed_count=len(state['failed_ids']),case_id=protocol['case_id'],case_selection=protocol['selection'],case_selection_uses_historical_truth_metrics=protocol['case_selection_uses_historical_truth_metrics'],truth=runtime.asset_record(truth_path),arms=summaries,paired_summary={method:{'median_delta_iou':float(np.median([r['delta_iou'] for r in paired if r['method']==method])),'median_delta_hard_seismic_rmse':float(np.median([r['delta_hard_seismic_rmse'] for r in paired if r['method']==method]))} for method in methods},undefined_metric_convention='null for empty union/prediction/truth denominator',physical_score='binary acoustic upper bound; full multiclass hard labels preserved')
    write_json(output/'summary.json',summary)
    np.save(output/'truth_target.npy',truth); np.save(output/'condition_mask.npy',cm); np.save(output/'condition_labels.npy',condition); np.save(output/'subsurface_mask.npy',support)
    write_json(output/'plot_context.json',dict(protocol=protocol,groups={arm:{name:idx.tolist() for name,idx in groups.items()} for arm,groups in group_indices.items()},ids=ids,spacing=spacing))
    if not args.skip_plots:
        plot_paired(output,label_arrays,ids,truth,cm,condition,spacing,methods,protocol)
        plot_distributions(output,candidate_metrics,score_index)
    report(output,summary,protocol)
    print(f'postprocess complete actual N={n}: {output}',flush=True)


def plot_group(folder,arm,name,labels,indices,truth,cm,condition,support,spacing,protocol):
    data=np.load(folder/'occurrence.npz'); masks=data['threshold_masks']; cuts=data['minimum_counts']
    members=indices[np.unique(np.linspace(0,len(indices)-1,min(12,len(indices)),dtype=int))]
    ids=[x['sample_id'] for x in protocol['source_ids']]
    for domain,roi in [('full',np.ones_like(cm)),('free_subsurface',support&~cm)]:
        overlay=[(cm&(condition==9),'#c83232')]
        left=[('Truth target (evaluation)',[(truth&roi,'#e7aa32')]+overlay),('Observed target voxels',overlay)]
        panels8=left+[(ids[i],[((labels[i]==9)&roi,'#e7aa32')]+overlay) for i in members]
        panels9=left+[(f'C>={c}; N={len(indices)}',[(mask&roi,'#e7aa32')]+overlay) for mask,c in zip(masks,cuts)]
        title=f'{arm} / {name} / {domain} / N={len(indices)}; selected case (historical IoU)'
        for fig,panels in [('figure8',panels8),('figure9',panels9)]:
            for front in (False,True):
                for layout in ('paper','presentation'):
                    occurrence.render_panels(folder/f'{fig}_{domain}_{"front" if front else "iso"}_{layout}',panels,truth.shape,spacing,title,front,columns=4 if fig=='figure8' else 3,presentation=layout=='presentation')
    captions=(f'{arm}; {protocol["case_id"]}; {name}; actual N={len(indices)}. Case selected retrospectively by historical adapter median target IoU. '
              'Empirical occurrence from hard raw-label9 models; uncalibrated. Strict >40/60/80%, and all-hit. '
              'Gold = target; red = independent observed target voxel overlay, not a threshold hit. Only aggregate condition mask is used; no inferred surface/well split. '
              'X,Y,Z; 100,100,50 m cells; larger Z upward. Exact unsmoothed exposed voxel faces. '
              'Free-subsurface figures exclude hard conditions from the statistic region but retain independent red overlays. '
              'Figure8 uses fixed order/rank quantiles, at most 12 members. Oracle selection uses full true target IoU; retrospective only. '
              'Stage19R uses a trained evidence adapter; Stage15 uses frozen positive-endpoint property guidance. No retraining.\n')
    (folder/'CAPTIONS.md').write_text(captions)


def report(output,summary,protocol):
    lines=[f'# Ensemble run: {summary["case_id"]}',f'Actual independent draws: {summary["actual_count"]}; planned: 200; failed: {summary["failed_count"]}.',
        f'Case selection: {summary["case_selection"]}; uses historical true target IoU. This selected-case illustration is not an unbiased cross-case comparison.',
        'Baseline is computed once per source and shared across the two identical-baseline comparisons. No pseudo-pairing between different selected Top100 sets.',
        'Frozen Flow, 32-step fixed Euler, raw labels -1..13, target9; XYZ 100/100/50 m, Z upward. Stage19R has a trained adapter. Binary high-contrast noiseless inverse-crime upper bound.',
        'Observation IDs were written before evaluation truth was opened. Full-target oracle ranking is explicitly retrospective. No truth-controlled tuning or seed selection.',
        'Undefined empty-union IoU and empty-prediction precision are null. Threshold masks are frequency summaries, not full geological realizations and carry no inherited seismic RMSE.',
        'No new five-body experiment or per-object recovery claim. A mask-only condition overlay is labelled as condition voxels.',
        '']
    for method in protocol['methods']:
        paired=summary['paired_summary'][method]; arm=summary['arms'][method]
        lines += [f'## {method}',f'Paired median IoU delta: {paired["median_delta_iou"]:+.6f}; paired median hard-seismic RMSE delta: {paired["median_delta_hard_seismic_rmse"]:+.6f}.']
        for name,item in arm.items():
            if not isinstance(item,dict) or 'median_candidate_iou' not in item: continue
            lines.append(f'{name}: median candidate IoU {item["median_candidate_iou"]:.6f}, hard seismic RMSE {item["median_candidate_hard_seismic_rmse"]:.6f}; threshold IoUs '+str([m['free_subsurface']['iou'] for m in item['threshold_metrics']]))
        lines.append(f'Unique decoded models: {arm["unique_decoded_models"]}. Identical independent outputs retain multiplicity.')
    (output/'RUN_REPORT.md').write_text('\n\n'.join(lines)+'\n')
    images=sorted(p.relative_to(output).as_posix() for p in output.rglob('*.png'))
    (output/'FIGURE_INDEX.md').write_text('\n'.join(f'- [{p}]({p})' for p in images)+'\n')


def plot_paired(output,arrays,ids,truth,cm,condition,spacing,methods,protocol):
    chosen=np.unique(np.linspace(0,len(ids)-1,min(12,len(ids)),dtype=int))
    write_json(output/'paired_figure_ids.json',[ids[i] for i in chosen])
    overlay=[(cm&(condition==9),'#c83232')]
    for method in methods:
        panels=[]
        for i in chosen:
            for arm in ('FLOW_ONLY',method):
                panels.append((f'{ids[i]} / {arm}',[(arrays[arm][i]==9,'#e7aa32')]+overlay))
        occurrence.render_panels(output/f'paired_{method}',panels,truth.shape,spacing,f'Paired same-source members; {protocol["case_id"]}; retrospective selected case',columns=4)
        for name in ('all'+str(len(ids)),'random'+str(min(100,len(ids))),'obs_top'+str(min(100,len(ids))),'oracle_top'+str(min(100,len(ids)))):
            panels=[]
            for arm in ('FLOW_ONLY',method):
                data=np.load(output/arm/name/'occurrence.npz')
                for mask,cut in zip(data['threshold_masks'],data['minimum_counts']):
                    panels.append((f'{arm}: C>={cut}',[(mask,'#e7aa32')]+overlay))
            occurrence.render_panels(output/f'threshold_compare_{method}_{name}',panels,truth.shape,spacing,f'{name}; separately selected sets, not source-paired; empirical occurrence',columns=4)


def plot_distributions(output,metrics,score_index):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fields=['iou','precision','recall','predicted_voxels']
    fig,axes=plt.subplots(1,5,figsize=(18,3.5),constrained_layout=True)
    for arm,rows in metrics.items():
        for ax,field in zip(axes,fields):
            values=[r['full'][field] for r in rows if r['full'][field] is not None]
            ax.hist(values,bins=20,histtype='step',label=arm); ax.set_title(field)
        axes[-1].hist(score_index[arm],bins=20,histtype='step',label=arm)
    axes[-1].set_title('hard seismic RMSE'); axes[0].legend(fontsize=6)
    fig.suptitle('All candidates; retrospective selected case; no Top100 truncation')
    for suffix in ('png','svg','pdf'): fig.savefig(output/f'candidate_distributions.{suffix}',dpi=160)
    plt.close(fig)


def plot_only(args):
    folder=args.report_dir or args.out/'postprocess_v1'; context=read_json(folder/'plot_context.json')
    output=args.plot_dir
    if output is None: raise ValueError('--plot-dir required for plot-only')
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    truth=np.load(folder/'truth_target.npy'); cm=np.load(folder/'condition_mask.npy'); cv=np.load(folder/'condition_labels.npy'); support=np.load(folder/'subsurface_mask.npy')
    import shutil
    for arm,groups in context['groups'].items():
        labels=np.load(folder/f'{arm}_labels.npy',mmap_mode='r')
        for name,idx in groups.items():
            destination=output/arm/name; destination.mkdir(parents=True)
            shutil.copyfile(folder/arm/name/'occurrence.npz',destination/'occurrence.npz')
            plot_group(destination,arm,name,labels,np.array(idx),truth,cm,cv,support,context['spacing'],context['protocol'])
    summary=read_json(folder/'summary.json'); report(output,summary,context['protocol'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=['audit','sample','postprocess','plot-only','all'],default='all')
    parser.add_argument('--case',default=CASE); parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--case-selection',choices=['historical_adapter_median_iou_best','specified_without_iou_selection'],default='historical_adapter_median_iou_best')
    parser.add_argument('--out',type=Path,required=True); parser.add_argument('--resume',action='store_true')
    parser.add_argument('--limit',type=int,default=200,help='integration check only; still freezes 200 IDs')
    parser.add_argument('--diagnostic',action='store_true'); parser.add_argument('--device',default='cuda')
    parser.add_argument('--report-dir',type=Path); parser.add_argument('--plot-dir',type=Path); parser.add_argument('--skip-plots',action='store_true')
    args=parser.parse_args()
    if not 1<=args.limit<=200: raise ValueError('limit must be 1..200')
    if args.action=='audit':
        protocol,_,_=make_protocol(args.case,args.methods,args.case_selection)
        if args.out.exists(): raise FileExistsError(args.out)
        args.out.mkdir(parents=True); write_json(args.out/'asset_audit.json',protocol); return
    if args.action in ('sample','all'): sample(args)
    if args.action in ('postprocess','all'): postprocess(args)
    if args.action=='plot-only': plot_only(args)

if __name__=='__main__': main()
