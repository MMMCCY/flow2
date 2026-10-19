"""Offline evaluation only: six classifications, property errors and fixed-grid TWT."""
import sys,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage26_diagnostics.common import *
from scripts.stage26.common import relative_l2
from scripts.stage23.evaluate import conversion
from scripts.stage21.evaluate import voxel_metrics,ratio
from guidance.seismic_inversion import cell_center_twt_ms
import torch
import numpy as np

def transition_counts(old,new,truth,domain,name):
    changed=domain&(old!=new);count=int(changed.sum())
    if not count:return []
    codes=((truth[changed]+1)*225+(old[changed]+1)*15+new[changed]+1).long().numpy()
    hist=np.bincount(codes,minlength=3375);indices=np.flatnonzero(hist)
    rows=[]
    for i in sorted(indices,key=lambda j:int(hist[j]),reverse=True):
        truthlabel=int(i//225)-1;oldlabel=int((i%225)//15)-1;newlabel=int(i%15)-1
        rows.append(dict(comparison=name,truth_raw=truthlabel,from_raw=oldlabel,to_raw=newlabel,voxels=int(hist[i]),from_correct=oldlabel==truthlabel,to_correct=newlabel==truthlabel))
    assert sum(r['voxels'] for r in rows)==count
    return rows

def summary_error(error,mask):
    a=error[mask]
    return dict(voxels=a.numel(),MAE=None if not a.numel() else float(a.mean()))

def main():
    guard('evaluate');torch.set_num_threads(4);torch.set_grad_enabled(False)
    if (ROOT/'EVALUATION.json').exists():raise FileExistsError('Keep completed diagnostic evaluation')
    f=read_json(ROOT/'INPUT_FREEZE.json');oracle=read_json(ROOT/'ORACLE_STATE.json');assert oracle['status']=='completed'
    assert sources()==f['source_sha256'] and file_hash(ROOT/'CANDIDATES.json')==f['candidate_file_sha256']
    table,operator=setup();a=check_condition(f['original_conditions']);sub=verify_tensor(f['subsurface']);obs=verify_tensor(f['observation'])
    fields=dict(prior_exact=verify_tensor(f['old_assets']['prior_exact']),updated=verify_tensor(f['old_assets']['updated']),oracle_updated=verify_tensor(oracle['assets']['updated']))
    classifications={};classification_assets={};physical=[]
    # No truth labels are provided to any classifier, including ORACLE classifications.
    for name,field in fields.items():
        for candidate_name,candidates in f['candidates'].items():
            key=f'{name}_{candidate_name}'
            pred=classify(field,table,candidates,sub,a['values'],a['mask'])
            assert torch.equal(pred[a['mask']],a['values'][a['mask']]) and not (pred[sub]<0).any()
            classifications[key]=pred;classification_assets[key]=save_tensor(ROOT/'classifications'/f'{key}.pt',pred)
            # Hard-label consistency uses each classification's own impedance AND velocity.
            acoustic=hard_labels_to_acoustic(pred,table);data=operator(acoustic[:,0:1],acoustic[:,1:2],sub)
            physical.append(dict(field=name,candidates=candidate_name,representation='hard_labels',relative_l2=relative_l2(data,obs),slowness='own hard-label codebook; NEVER oracle slowness',data_samples=obs.numel()))
    old_inv=read_json(FORMAL/'INVERSION_STATE.json')
    for name,asset,role in [('saved_A_prior',f['old_assets']['prior_seismic'],'fixed A slowness'),('saved_A_updated',f['old_assets']['updated_seismic'],'fixed A slowness'),('oracle_prior',oracle['assets']['prior_seismic'],'ORACLE truth slowness'),('oracle_updated',oracle['assets']['updated_seismic'],'ORACLE truth slowness')]:
        data=verify_tensor(asset);error=relative_l2(data,obs)
        if name in ('saved_A_prior','saved_A_updated'):assert abs(error-old_inv['relative_l2']['A_prior' if name=='saved_A_prior' else 'continuous_updated'])<1e-12
        physical.append(dict(field=name,candidates='none',representation='continuous_acoustic',relative_l2=error,slowness=role,data_samples=obs.numel()))
    # Evaluation-only truth read: candidate choices were already saved by a separate truth-blind process.
    e=read_json(STAGE25/'evaluation_registry.json');truth=verify_tensor(e['truth']);truth_np=truth.numpy()[0,0]
    domains={name:(verify_tensor(info['mask']),info['classes']) for name,info in e['domains'].items()}
    free=domains['E_original_free'][0];dense=check_condition(f['dense_conditions']['D']);added=dense['mask']&~a['mask']
    cap=torch.zeros_like(truth,dtype=torch.bool);box=e['geometry']['cap_evaluation_box_indices']
    cap[:,:,box['x'][0]:box['x'][1]+1,box['y'][0]:box['y'][1]+1,box['z'][0]:box['z'][1]+1]=True
    cap&=sub
    subset=(truth==9)&free&~added;assert int(subset.sum())==1933
    allmask=torch.ones_like(truth,dtype=torch.bool)
    classes=sorted(int(i) for i in torch.unique(truth) if i!=-1)
    domains['FULL']=(allmask,classes)
    condition_rows=[];morphology=[];all_metrics=[]
    for key,pred in classifications.items():
        field,candidate=key.rsplit('_',1)
        q,_,_=conversion(pred.numpy()[0,0],truth_np,added.numpy()[0,0])
        condition_rows.append(dict(field=field,candidates=candidate,**q))
        for name,(domain,classes) in domains.items():
            m=voxel_metrics(pred.numpy()[0,0],truth_np,domain.numpy()[0,0],classes)
            assert m['target_tp']+m['target_fp']==m['target_pred_volume'] and m['target_tp']+m['target_fn']==m['target_truth_volume']
            m['volume_ratio']=ratio(m['target_pred_volume'],m['target_truth_volume'])
            all_metrics.append(dict(field=field,candidates=candidate,domain=name,**m))
        p=pred==9;t=truth==9
        morphology.append(dict(field=field,candidates=candidate,full_target_iou=ratio(int((p&t).sum()),int((p|t).sum())),full_target_recall=ratio(int((p&t).sum()),int(t.sum())),full_volume_ratio=ratio(int(p.sum()),int(t.sum())),target_fp=int((p&~t).sum()),cap_box_iou=ratio(int((p&t&cap).sum()),int(((p|t)&cap).sum())),cap_target_recall=ratio(int((p&t&cap).sum()),int((t&cap).sum())),common_outside_tp=int((p&subset).sum()),common_outside_voxels=int(subset.sum()),common_outside_coverage=ratio(int((p&subset).sum()),int(subset.sum()))))
    comparisons=[]
    for field in FIELDS:comparisons.append((f'{field}:FULL_to_OBSERVED',f'{field}_FULL',f'{field}_OBSERVED'))
    for candidate in CANDIDATE_NAMES:
        comparisons += [(f'{candidate}:prior_to_saved_updated',f'prior_exact_{candidate}',f'updated_{candidate}'),(f'{candidate}:saved_to_oracle_updated',f'updated_{candidate}',f'oracle_updated_{candidate}')]
    transfers=[];details=[]
    for name,oldkey,newkey in comparisons:
        for domain_name,domain in [('added',added),('original_free',free)]:transfers.append(dict(comparison=name,domain=domain_name,**transfer(classifications[oldkey],classifications[newkey],truth,domain)))
        details+=transition_counts(classifications[oldkey],classifications[newkey],truth,added,name)
    prior=fields['prior_exact'];updated=fields['updated'];oracle_up=fields['oracle_updated']
    # True impedance exists ONLY for evaluation of continuous errors, never oracle input/regularizer.
    true_log=table[0,truth.long()+1].log()
    errors={name:(value[:,0:1].log()-true_log).abs() for name,value in fields.items()}
    log_rows=[]
    regions=dict(original_free_all=free,original_free_target=free&(truth==9),original_free_non_target=free&(truth!=9))
    right_to_wrong=added&(classifications['prior_exact_FULL']==truth)&(classifications['updated_FULL']!=truth)
    assert int(right_to_wrong.sum())==1792
    regions['added_FULL_right_to_wrong']=right_to_wrong
    for name,error in errors.items():
        for region,mask in regions.items():log_rows.append(dict(field=name,region=region,error_definition='natural-log impedance MAE; identical physical units',**summary_error(error,mask)))
    rt_details=transition_counts(classifications['prior_exact_FULL'],classifications['updated_FULL'],truth,right_to_wrong,'added_FULL_right_to_wrong')
    for row in rt_details:
        mask=right_to_wrong&(truth==row['truth_raw'])&(classifications['updated_FULL']==row['to_raw'])
        row.update(prior_log_MAE=float(errors['prior_exact'][mask].mean()),updated_log_MAE=float(errors['updated'][mask].mean()),oracle_log_MAE=float(errors['oracle_updated'][mask].mean()))
    # cell_center_twt_ms returns TOP-DOWN: flip back before masking or plotting in original XYZ.
    truth_s=table[1,truth.long()+1]
    a_twt=cell_center_twt_ms(prior[:,1:2],sub,SPACING).flip(-1)
    truth_twt=cell_center_twt_ms(truth_s,sub,SPACING).flip(-1)
    twt_abs=(a_twt-truth_twt).abs();twt_rows=[]
    for region,mask in [('original_free',free),('fixed_cap_box',cap)]:
        vals=twt_abs[mask]
        med=float(vals.median());p90=float(torch.quantile(vals,.9));maximum=float(vals.max())
        twt_rows.append(dict(region=region,voxels=int(vals.numel()),median_ms=med,p90_ms=p90,max_ms=maximum,median_8ms_samples=med/8,p90_8ms_samples=p90/8,max_8ms_samples=maximum/8,median_40ms_periods=med/40,p90_40ms_periods=p90/40,max_40ms_periods=maximum/40,axis='original XYZ after flipping top-down output',comparison='same-depth TWT difference, not matched reflection events'))
    twt_assets={name:save_tensor(ROOT/'evaluation'/f'{name}.pt',value) for name,value in dict(A_twt_xyz_ms=a_twt,truth_twt_xyz_ms=truth_twt,absolute_twt_difference_ms=twt_abs).items()}
    write_csv(ROOT/'condition_quality.csv',condition_rows);write_csv(ROOT/'core_morphology.csv',morphology);write_csv(ROOT/'domain_metrics.csv',all_metrics)
    write_csv(ROOT/'error_transfers.csv',transfers);write_csv(ROOT/'class_transition_counts.csv',details);write_csv(ROOT/'right_to_wrong_log_errors.csv',rt_details)
    write_csv(ROOT/'continuous_log_impedance_errors.csv',log_rows);write_csv(ROOT/'twt_differences.csv',twt_rows);write_csv(ROOT/'physical_residuals.csv',physical)
    write_json(ROOT/'EVALUATION.json',dict(status='completed',role='OFFLINE_DIAGNOSTICS; oracle not deployable',classification_assets=classification_assets,candidates=f['candidates'],condition_quality=condition_rows,morphology=morphology,domain_metrics=all_metrics,transfers=transfers,log_errors=log_rows,twt_statistics=twt_rows,twt_assets=twt_assets,physical_residuals=physical,reproduction=f['reproduction'],added_voxels=int(added.sum()),right_to_wrong_voxels=int(right_to_wrong.sum()),cap_box=box,new_oracle_inversions=1,new_Flow=0,checkpoint_loads=0,new_observation_generations=0,evaluation_source_sha256=file_hash(Path(__file__)),completed_at_utc=utc()))
    print('SIX_OFFLINE_CLASSIFICATIONS_EVALUATED',condition_rows,morphology,physical,twt_rows,flush=True)
if __name__=='__main__':main()
