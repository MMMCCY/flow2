"""Saved-prediction evaluation: old Stage25 domains and distances, full-data seismic errors."""
import os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage26.common import *
from scripts.stage25.evaluate import surface_distance,centroid_distance
from scripts.stage21.evaluate import voxel_metrics,ratio
from scripts.stage23.evaluate import sealed_prediction,conversion
import numpy as np
import torch
from scipy import ndimage

def main():
    guard('evaluate');torch.set_num_threads(4);torch.set_grad_enabled(False)
    if (ROOT/'EVALUATION.json').exists():raise FileExistsError('Preserve completed evaluation')
    freeze=read_json(ROOT/'INPUT_FREEZE.json');assert freeze['source_sha256']==sources()
    seal=read_json(ROOT/'CONDITIONS_FROZEN.json');inv=read_json(ROOT/'INVERSION_STATE.json');assert inv['status']=='completed'
    predictions={};assets={};receipts={}
    for arm in ('A','B','C'):
        r,p=sealed_prediction(OLD/f'arms/{arm}/predictions/seed42');predictions[arm]=p;assets[arm]=r['prediction'];receipts[arm]=r
    conditions={arm:check_condition(seal['conditions'][arm]) for arm in ('D0','D')}
    assert torch.equal(conditions['D0']['mask'],conditions['D']['mask'])
    for arm in ('D0','D'):
        canonical=seal['alias'].get(arm,arm);r,p=sealed_prediction(ROOT/f'arms/{canonical}/predictions/seed42')
        assert r['velocity_evaluations']==32 and r['base_hash_after']==r['base_hash_before']==EXPECTED_MODEL
        assert r['initial_noise_tensor_sha256']==freeze['noise']['tensor_sha256']
        assert r['registry_sha256']==file_hash(ROOT/f'arms/{canonical}/inference_registry.json')
        assert not ((p!=conditions[arm]['values'])&conditions[arm]['mask']).any()
        predictions[arm]=p;assets[arm]=r['prediction'];receipts[arm]=r
    assert all(r['base_hash_after']==EXPECTED_MODEL and r['initial_noise_tensor_sha256']==freeze['noise']['tensor_sha256'] for r in receipts.values())
    original=conditions['D']['original_mask'];added=conditions['D']['mask']&~original
    # Numerical assets and sealed direct classifications were generated before any geological evaluation.
    direct_assets={k:inv['assets'][name] for k,name in [('DIRECT_PRIOR','classification_D0'),('DIRECT_UPDATED','classification_D')]}
    for k,asset in direct_assets.items():predictions[k]=verify_tensor(asset);assets[k]=asset
    copy=torch.where(conditions['D']['mask'],conditions['D']['values'],predictions['A'])
    predictions['COPY_ONLY_D']=copy;assets['COPY_ONLY_D']=save_tensor(ROOT/'offline/COPY_ONLY_D.pt',copy)
    table,operator=setup();obs=verify_tensor(freeze['observation']);sub=verify_tensor(freeze['subsurface'])
    physical=[]
    for k,field_asset in [('A_prior','prior_seismic'),('continuous_updated','updated_seismic')]:
        field=verify_tensor(inv['assets'][field_asset]);err=relative_l2(field,obs)
        assert abs(err-inv['relative_l2'][k])<1e-12
        physical.append(dict(arm=k,relative_l2=err,velocity_policy='fixed A prior slowness',data_samples=obs.numel(),prediction=inv['assets'][field_asset]))
    for arm in ('D0','D','DIRECT_PRIOR','DIRECT_UPDATED'):
        acoustic=hard_labels_to_acoustic(predictions[arm],table)
        field=operator(acoustic[:,0:1],acoustic[:,1:2],sub)
        field_asset=save_tensor(ROOT/'physics'/f'{arm}_hard_seismic.pt',field)
        physical.append(dict(arm=arm,relative_l2=relative_l2(field,obs),velocity_policy='each hard lithology mapped to codebook impedance AND slowness; no truth velocity',data_samples=obs.numel(),prediction=field_asset,underground_air_voxels=int(((predictions[arm]==-1)&sub).sum())))
    # First truth read in this stage occurs only after both D0/D predictions are fixed.
    evaluation_started=utc();e=read_json(OLD/'evaluation_registry.json');truth_t=verify_tensor(e['truth']);truth=truth_t.numpy()[0,0]
    b=check_condition(read_json(OLD/'arms/B/inference_registry.json')['conditions']);c=check_condition(read_json(OLD/'arms/C/inference_registry.json')['conditions'])
    assert torch.equal(b['mask'],conditions['D']['mask']) and torch.equal(c['mask'],conditions['D']['mask'])
    domains={k:(verify_tensor(v['mask']).numpy()[0,0],v['classes']) for k,v in e['domains'].items()}
    held=verify_tensor(e['domains']['E_holdout']['mask'])
    assert not (held&added).any() and torch.equal(copy[held],predictions['A'][held])
    domains['FULL']=(np.ones_like(truth,dtype=bool),sorted(int(i) for i in np.unique(truth) if i!=-1))
    capbox=e['geometry']['cap_evaluation_box_indices'];cap=np.zeros_like(truth,dtype=bool)
    cap[capbox['x'][0]:capbox['x'][1]+1,capbox['y'][0]:capbox['y'][1]+1,capbox['z'][0]:capbox['z'][1]+1]=True
    subset=(truth==9)&domains['E_original_free'][0]&~added.numpy()[0,0]
    rows=[];shape=[];outside=[];quality=[]
    for arm,pred in predictions.items():
        v=pred.numpy()[0,0];p=v==9;t=truth==9
        assert pred.shape==truth_t.shape and pred.min()>=-1 and pred.max()<=13
        for name,(domain,classes) in domains.items():
            m=voxel_metrics(v,truth,domain,classes)
            assert m['target_tp']+m['target_fp']==m['target_pred_volume'] and m['target_tp']+m['target_fn']==m['target_truth_volume']
            m.update(target_pred_truth_volume_ratio=ratio(m['target_pred_volume'],m['target_truth_volume']),centroid_error_voxels=centroid_distance(p&domain,t&domain,[1]*3),centroid_error_m=centroid_distance(p&domain,t&domain,[SPACING]*3))
            rows.append(dict(arm=arm,domain=name,**m))
        dist=surface_distance(p,t,[SPACING]*3)
        shape.append(dict(arm=arm,target_iou=ratio(int((p&t).sum()),int((p|t).sum())),assd_m=dist,assd_voxels=None if dist is None else dist/SPACING,detected=bool(p.any()),predicted_voxels=int(p.sum()),truth_voxels=int(t.sum()),volume_ratio=ratio(int(p.sum()),int(t.sum())),target_fp=int((p&~t).sum()),predicted_components=int(ndimage.label(p,np.ones((3,3,3)))[1]),cap_box_iou=ratio(int((p&t&cap).sum()),int(((p|t)&cap).sum()))))
        outside.append(dict(arm=arm,voxels=int(subset.sum()),tp=int((p&subset).sum()),coverage=ratio(int((p&subset).sum()),int(subset.sum()))))
    for arm in ('D0','D'):
        q,cls,matrix=conversion(conditions[arm]['values'].numpy()[0,0],truth,added.numpy()[0,0]);quality.append(dict(arm=arm,**q));write_csv(ROOT/f'well_classes_{arm}.csv',cls);np.savetxt(ROOT/f'well_confusion_{arm}.csv',matrix,fmt='%d',delimiter=',')
    v0=conditions['D0']['values'];vd=conditions['D']['values'];changed=(v0!=vd)&added
    w0=v0!=truth_t;wd=vd!=truth_t
    transitions=dict(changed_added_voxels=int(changed.sum()),wrong_to_right=int((changed&w0&~wd).sum()),right_to_wrong=int((changed&~w0&wd).sum()),wrong_to_wrong=int((changed&w0&wd).sum()),unchanged_voxels=int((added&~changed).sum()),added_voxels=int(added.sum()))
    assert transitions['wrong_to_right']+transitions['right_to_wrong']+transitions['wrong_to_wrong']==transitions['changed_added_voxels']
    assert quality[0]['errors']-transitions['wrong_to_right']+transitions['right_to_wrong']==quality[1]['errors']
    for name,rr in dict(metrics=rows,shape_metrics=shape,outside_added_wells=outside,condition_quality=quality,condition_transitions=[transitions],physical_errors=[{k:v for k,v in row.items() if k!='prediction'} for row in physical]).items():write_csv(ROOT/f'{name}.csv',rr)
    write_json(ROOT/'EVALUATION.json',dict(status='completed',evaluation_started_at_utc=evaluation_started,completed_at_utc=utc(),prediction_assets=assets,old_evaluation_registry_sha256=file_hash(OLD/'evaluation_registry.json'),shape=shape,metrics=rows,outside=outside,condition_quality=quality,condition_transitions=transitions,physical_errors=physical,new_Flow_outputs=len(seal['canonical_arms']),velocity_evaluations=32*len(seal['canonical_arms']),inversion_calls=1,old_stage25_failed_inversion_attempts=1,checks=dict(condition_violations=0,B_C_D0_D_masks_equal=True,holdout_added_voxels=0,copy_holdout_equals_A=True,model_identity=True,source_identity=True,no_A_B_C_reruns=True)))
    print('EVALUATION_COMPLETE',shape,physical,transitions,flush=True)
if __name__=='__main__':main()
