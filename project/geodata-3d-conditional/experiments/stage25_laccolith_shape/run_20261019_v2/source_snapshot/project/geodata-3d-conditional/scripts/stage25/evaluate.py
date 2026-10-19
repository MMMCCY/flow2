"""Retrospective fixed-domain/whole-target evaluation after all three sealed predictions."""
import os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage25.common import *
from scripts.stage21.evaluate import voxel_metrics,ratio
from scripts.stage23.evaluate import conversion,sealed_prediction
import numpy as np
import torch
from scipy import ndimage

def surface_distance(pred,truth,spacing):
    """ASSD: mean over both directed boundary-distance samples, all components."""
    if not pred.any() or not truth.any(): return None
    structure=ndimage.generate_binary_structure(3,1)
    p=pred&~ndimage.binary_erosion(pred,structure=structure,border_value=0)
    t=truth&~ndimage.binary_erosion(truth,structure=structure,border_value=0)
    a=ndimage.distance_transform_edt(~t,sampling=spacing)[p]
    b=ndimage.distance_transform_edt(~p,sampling=spacing)[t]
    return float((a.sum()+b.sum())/(a.size+b.size))

def centroid_distance(p,t,spacing):
    if not p.any() or not t.any(): return None
    delta=np.argwhere(p).mean(0)-np.argwhere(t).mean(0)
    return float(np.linalg.norm(delta*np.asarray(spacing)))

def main():
    guard('evaluate');torch.set_num_threads(4)
    if (ROOT/'EVALUATION.json').exists(): raise FileExistsError('Preserve evaluation')
    f=read_json(ROOT/'OBSERVATION_FREEZE.json');assert sources()==f['source_sha256']
    conditions={a:check_condition(read_json(ROOT/f'arms/{a}/inference_registry.json')['conditions']) for a in 'ABC'}
    assert torch.equal(conditions['B']['mask'],conditions['C']['mask'])
    predictions={};receipts={}
    for arm in 'ABC':
        r,p=sealed_prediction(ROOT/f'arms/{arm}/predictions/seed42')
        assert r['velocity_evaluations']==32 and r['base_hash_after']==r['base_hash_before']==EXPECTED_MODEL
        assert r['initial_noise_tensor_sha256']==f['noise']['tensor_sha256']
        assert r['registry_sha256']==file_hash(ROOT/f'arms/{arm}/inference_registry.json')
        assert not ((p!=conditions[arm]['values'])&conditions[arm]['mask']).any()
        predictions[arm]=p;receipts[arm]=r
    # First evaluation truth read, after all conditions and outputs have been sealed.
    evaluation_started=utc();e=read_json(ROOT/'evaluation_registry.json');truth_t=verify_tensor(e['truth']);truth=truth_t.numpy()[0,0]
    original=conditions['A']['mask'];c=conditions['C'];added=c['mask']&~original
    assert torch.equal(conditions['A']['values'][original],truth_t[original]) and torch.equal(conditions['B']['values'][conditions['B']['mask']],truth_t[conditions['B']['mask']])
    labels=verify_tensor(read_json(ROOT/'preprocessing_assets.json')['classification'])
    predictions['DIRECT']=torch.where(original,conditions['A']['values'],labels)
    predictions['COPY_ONLY']=torch.where(c['mask'],c['values'],predictions['A'])
    offline={a:save_tensor(ROOT/'offline'/f'{a}.pt',predictions[a]) for a in ('DIRECT','COPY_ONLY')}
    domains={k:(verify_tensor(v['mask']).numpy()[0,0],v['classes']) for k,v in e['domains'].items()}
    held=verify_tensor(e['domains']['E_holdout']['mask'])
    assert torch.equal(predictions['COPY_ONLY'][held],predictions['A'][held]) and not (held&added).any()
    domains['FULL']=(np.ones_like(truth,dtype=bool),sorted(int(i) for i in np.unique(truth) if i!=-1))
    rows=[];shape=[];outside=[];quality=[]
    subset=(truth==9)&domains['E_original_free'][0]&~added.numpy()[0,0]
    capbox=e['geometry']['cap_evaluation_box_indices'];cap=np.zeros_like(truth,dtype=bool)
    cap[capbox['x'][0]:capbox['x'][1]+1,capbox['y'][0]:capbox['y'][1]+1,capbox['z'][0]:capbox['z'][1]+1]=True
    for arm,pred in predictions.items():
        v=pred.numpy()[0,0];p=v==9;t=truth==9
        for name,(domain,classes) in domains.items():
            m=voxel_metrics(v,truth,domain,classes)
            assert m['target_tp']+m['target_fp']==m['target_pred_volume'] and m['target_tp']+m['target_fn']==m['target_truth_volume']
            m.update(target_pred_truth_volume_ratio=ratio(m['target_pred_volume'],m['target_truth_volume']),centroid_error_voxels=centroid_distance(p&domain,t&domain,[1]*3),centroid_error_m=centroid_distance(p&domain,t&domain,[SPACING]*3))
            rows.append(dict(arm=arm,domain=name,**m))
        distance=surface_distance(p,t,[SPACING]*3)
        shape.append(dict(arm=arm,target_iou=ratio(int((p&t).sum()),int((p|t).sum())),assd_m=distance,assd_voxels=None if distance is None else distance/SPACING,detected=bool(p.any()),predicted_voxels=int(p.sum()),truth_voxels=int(t.sum()),volume_ratio=ratio(int(p.sum()),int(t.sum())),predicted_components=int(ndimage.label(p,np.ones((3,3,3)))[1]),cap_box_iou=ratio(int((p&t&cap).sum()),int(((p|t)&cap).sum()))))
        outside.append(dict(arm=arm,subset='truth target & original unknown & ~C added mask',voxels=int(subset.sum()),tp=int((p&subset).sum()),coverage=ratio(int((p&subset).sum()),int(subset.sum()))))
    for arm in ('B','C'):
        q,classes,matrix=conversion(conditions[arm]['values'].numpy()[0,0],truth,added.numpy()[0,0])
        quality.append(dict(arm=arm,**q));write_csv(ROOT/f'well_classes_{arm}.csv',classes)
        np.savetxt(ROOT/f'well_confusion_{arm}.csv',matrix,fmt='%d',delimiter=',')
    for name,rr in dict(metrics=rows,shape_metrics=shape,outside_added_wells=outside,condition_quality=quality).items():write_csv(ROOT/f'{name}.csv',rr)
    write_json(ROOT/'EVALUATION.json',dict(status='completed',evaluation_started_at_utc=evaluation_started,completed_at_utc=utc(),source_sha256=sources(),samples={a:r['prediction'] for a,r in receipts.items()},offline=offline,shape=shape,condition_quality=quality,outside=outside,metrics=rows,checks=dict(original_conditions_match_truth=True,B_true_labels=True,B_C_masks_equal=True,holdout_added_voxels=0,copy_holdout_equals_A=True,condition_violations=0,model_unchanged=True,paired_noise=True),new_flow_outputs=3,velocity_evaluations=96,restoration_solves=1))
    print('EVALUATION_COMPLETE',shape,quality,flush=True)
if __name__=='__main__':main()
