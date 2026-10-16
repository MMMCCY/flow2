"""Fixed-domain, complete-object retrospective evaluation for the two new cases."""
import argparse
import os
import sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage24.common import *
from scripts.stage21.evaluate import voxel_metrics,connected,match_objects,ratio
from scripts.stage23.evaluate import conversion,sealed_prediction
import numpy as np
import torch

def evaluate(case):
    guard('evaluate');torch.set_num_threads(4);torch.set_grad_enabled(False)
    root=ROOT/'cases'/case
    if (root/'EXECUTION.json').exists():raise FileExistsError('Preserve completed evaluation')
    seal=read_json(root/'CONDITION_RECEIPT.json');reg=read_json(root/'inference_registry.json')
    freeze=read_json(root/'METHOD_FREEZE.json')
    assert case in CASES and seal['case']==case and seal['converged']
    assert freeze['source_sha256']==sources()
    assert file_hash(root/'METHOD_FREEZE.json')==seal['freeze_file_sha256']
    assert file_hash(root/'inference_registry.json')==seal['inference_registry_sha256']
    newr,new=sealed_prediction(root/'predictions/seed42')
    assert newr['velocity_evaluations']==32 and newr['registry_sha256']==file_hash(root/'inference_registry.json')
    assert newr['condition_record_sha256']==seal['condition_record_sha256']==json_hash(reg['conditions'])
    assert newr['base_hash_after']==newr['base_hash_before']==reg['noise']['42']['expected_model_hash']
    assert newr['initial_noise_tensor_sha256']==reg['noise']['42']['asset']['tensor_sha256']
    conditions=check_condition(reg['conditions']);added=verify_tensor(seal['added_mask'])
    assert not ((new!=conditions['values'])&conditions['mask']).any()
    assert torch.equal(added,conditions['mask']&~conditions['original_mask'])
    # First full truth access for this NEW case is after field, conditions and prediction seals.
    started=utc();e=read_json(OLD/'evaluation_registry.json')['cases'][case]
    truth_t=verify_tensor(e['truth']);truth=truth_t.numpy()[0,0]
    domains={d:(verify_tensor(a['tensor']).numpy()[0,0],a['classes']) for d,a in e['domains'].items()}
    ti,nt=connected(truth==9)
    original=conditions['original_mask'].numpy()[0,0];mask=conditions['mask'].numpy()[0,0]
    layout=read_json(OLD/'layout.json');well=well_mask(layout['original_wells']).numpy()[0,0]
    exposure=[]
    for i in range(1,nt+1):
        obj=ti==i
        exposure.append(dict(case=case,object_id=i,full_truth_voxels=int(obj.sum()),major=bool(obj.sum()>=20),
            original_well_hit_voxels=int((obj&well).sum()),original_condition_hit_voxels=int((obj&original).sum()),
            original_well_hit_columns=sum(bool(obj[x,y].any()) for x,y in layout['original_wells'])))
    samples=[];sample_rows=[];object_rows=[];pred_rows=[];outside_rows=[]
    def measure(arm,pred,asset,receipt=None):
        assert pred.shape==truth_t.shape and pred.min()>=-1 and pred.max()<=13
        v=pred.numpy()[0,0];pi,_=connected(v==9)
        s=dict(case=case,seed=42,arm=arm,prediction=asset,receipt=receipt,metrics={},objects={})
        for name,(domain,classes) in domains.items():
            m=voxel_metrics(v,truth,domain,classes)
            assert m['target_tp']+m['target_fp']==m['target_pred_volume']
            assert m['target_tp']+m['target_fn']==m['target_truth_volume']
            m['target_pred_truth_volume_ratio']=ratio(m['target_pred_volume'],m['target_truth_volume'])
            oo,pp,stats,_=match_objects(ti,pi,domain,RULES);m.update(stats)
            s['metrics'][name]=m;s['objects'][name]=oo
            sample_rows.append(dict(case=case,arm=arm,domain=name,**m))
            object_rows.extend({'arm':arm,'domain':name,**exposure[o['object_id']-1],**o} for o in oo)
            pred_rows.extend(dict(case=case,arm=arm,domain=name,**p) for p in pp)
        for meta in exposure:
            if meta['original_condition_hit_voxels']==0:
                subset=(ti==meta['object_id'])&domains['E_original_free'][0]&~mask
                count=int(subset.sum());tp=int(((v==9)&subset).sum())
                outside_rows.append(dict(case=case,arm=arm,object_id=meta['object_id'],major=meta['major'],
                    voxels=count,tp=tp,coverage=ratio(tp,count),subset='object & E_original_free & ~NEW_mask (same for every arm)'))
        samples.append(s)
    def rid(folder):return dict(path=str((folder/'receipt.json').relative_to(REPO)),file_sha256=file_hash(folder/'receipt.json'))
    predictions={}
    for arm in ('A_9','B_512','C_512'):
        folder=OLD/'predictions'/case/'seed42'/arm;r,p=sealed_prediction(folder)
        assert r['case_id']==case and r['source_seed']==42
        assert r['initial_noise_tensor_sha256']==reg['noise']['42']['asset']['tensor_sha256']
        assert r['base_hash_after']==reg['noise']['42']['expected_model_hash']
        predictions[arm]=p;measure(arm,p,r['prediction'],rid(folder))
    labels=verify_tensor(read_json(root/'preprocessing_assets.json')['classification'])
    direct=torch.where(conditions['original_mask'],conditions['original_values'].to(labels.dtype),labels)
    measure('RESTORED_CLASSIFICATION',direct,save_tensor(root/'offline/restored_direct.pt',direct))
    copy=torch.where(conditions['mask'],conditions['values'].to(predictions['A_9'].dtype),predictions['A_9'])
    held=torch.from_numpy(domains['E_holdout'][0]).reshape_as(copy)
    assert torch.equal(copy[held],predictions['A_9'][held])
    measure('COPY_ONLY',copy,save_tensor(root/'offline/copy_only.pt',copy))
    measure('FLOW_RESTORED_PSEUDO',new,newr['prediction'],rid(root/'predictions/seed42'))
    oldc=check_condition(read_json(OLD/'inference_registry.json')['conditions'][case]['C_512'])
    assert torch.equal(oldc['mask'],conditions['mask'])
    quality=[]
    for arm,v in (('C_512',oldc['values']),('RESTORED_WELLS',conditions['values'])):
        q,_,_=conversion(v.numpy()[0,0],truth,added.numpy()[0,0]);quality.append(dict(case=case,arm=arm,**q))
    for name,rows in (('sample_metrics',sample_rows),('condition_quality',quality),('object_exposure',exposure),
        ('object_metrics',object_rows),('predicted_object_metrics',pred_rows),('objects_common_outside',outside_rows)):
        write_csv(root/f'{name}.csv',rows)
    write_json(root/'EVALUATION_INPUTS.json',dict(case=case,truth=e['truth'],domains=e['domains'],
        role='retrospective only',evaluation_started_at_utc=started,condition_sealed_at_utc=seal['sealed_at_utc']))
    result=dict(case=case,seed=42,status='completed',evaluation_started_at_utc=started,
        samples=samples,conversion=quality,object_exposure=exposure,objects_common_outside=outside_rows,
        copy_holdout_equal_A=True,new_solve_invocations=1,new_Flow_outputs=1,velocity_evaluations=32,
        engineering_retries=0,completed_at_utc=utc())
    write_json(root/'EXECUTION.json',result)
    print(f'{case}: evaluated {nt} complete truth objects; no science-based selection or retries',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=CASES,required=True)
    evaluate(p.parse_args().case)
