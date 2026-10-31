"""Retrospective hard geology and two-observation cross-evaluation; never samples."""
import argparse
import os
import sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
PROJECT=Path(__file__).resolve().parents[2]
for p in (PROJECT,PROJECT.parents[1],PROJECT.parents[1]/'src'):
    sys.path.insert(0,str(p))
import numpy as np
import torch
from scipy import ndimage
from .common import *
from .sampler import soft_decode_to_probs
from guidance.seismic import (hard_labels_to_acoustic,probabilities_to_subsurface_acoustic,
    overwrite_exact_condition_acoustic)
from scripts.stage21.evaluate import voxel_metrics,ratio
from scripts.stage25.evaluate import surface_distance,centroid_distance

def component_summary(pred,truth):
    pi,pn=ndimage.label(pred==9,np.ones((3,3,3)))
    ti,tn=ndimage.label(truth==9,np.ones((3,3,3)))
    objects=[]
    for i in range(1,pn+1):
        region=pi==i
        ids,counts=np.unique(ti[region],return_counts=True)
        objects.append(dict(id=i,volume=int(region.sum()),
            truth_overlaps=[dict(id=int(j),voxels=int(c)) for j,c in zip(ids,counts) if j],
            centroid_indices=np.argwhere(region).mean(0).tolist()))
    return dict(connectivity=26,predicted_components=pn,truth_components=tn,
        predicted_volume_sequence=sorted([o['volume'] for o in objects],reverse=True),
        objects=objects,truth_volume_sequence=sorted(np.bincount(ti.ravel())[1:].tolist(),reverse=True))

def boundary_summary(pred,truth):
    p=pred==9;t=truth==9
    if not p.any() or not t.any():
        return dict(assd_m=None,pred_to_truth_mean_m=None,truth_to_pred_mean_m=None,
                    status='empty_predicted_target' if not p.any() else 'empty_truth_target')
    s=ndimage.generate_binary_structure(3,1)
    pb=p&~ndimage.binary_erosion(p,structure=s,border_value=0)
    tb=t&~ndimage.binary_erosion(t,structure=s,border_value=0)
    a=ndimage.distance_transform_edt(~tb,sampling=SPACING)[pb]
    b=ndimage.distance_transform_edt(~pb,sampling=SPACING)[tb]
    return dict(assd_m=float((a.sum()+b.sum())/(a.size+b.size)),
        pred_to_truth_mean_m=float(a.mean()),truth_to_pred_mean_m=float(b.mean()),
        pred_to_truth_p95_m=float(np.quantile(a,.95)),truth_to_pred_p95_m=float(np.quantile(b,.95)),
        status='valid',centroid_distance_m=centroid_distance(p,t,[SPACING]*3))

def main(root):
    root=root.resolve();install_firewall(root);torch.set_num_threads(4);torch.set_grad_enabled(False)
    if (root/'EVALUATION.json').exists():
        raise FileExistsError('Evaluation complete; preserve outputs')
    complete=read_json(root/'SAMPLING_COMPLETE.json');reg=read_json(root/'inference_registry.json')
    ev=read_json(root/'evaluation_registry.json')
    truths={k:verify_tensor(a) for k,a in ev['truths'].items()}
    values=verify_tensor(reg['conditions']['values']);m0=verify_tensor(reg['conditions']['mask']).bool()
    sub=verify_tensor(reg['subsurface']).bool();m512=verify_tensor(ev['M512']).bool()
    obs={k:verify_tensor(a) for k,a in reg['observations'].items()}
    cap=verify_tensor(ev['cap_union']).bool();diff=verify_tensor(ev['difference']).bool()
    domains=dict(FULL=torch.ones_like(sub),U_original=sub&~m0,U_common=sub&~m512,
        HOLDOUT=sub&~m0&well_mask(ev['holdout']),CAP_G1=cap,CAP_UNION=cap,
        GEOMETRY_DIFFERENCE=diff,UNCHANGED=sub&~diff&~m0)
    pred_assets={k:v['assets']['hard_prediction'] for k,v in complete['arms'].items()}
    # Existing A/D0/D/DIRECT are reference-only inputs, never new generation.
    for name,folder in [('A',R25),('D0',R26),('D',R26)]:
        pred_assets[name]=read_json(folder/f'arms/{name}/predictions/seed42/receipt.json')['prediction']
    inversion=read_json(R26/'INVERSION_STATE.json')
    for name,key in [('DIRECT_PRIOR','classification_D0'),('DIRECT_UPDATED','classification_D')]:
        pred_assets[name]=inversion['assets'][key]
    predictions={k:verify_tensor(a) for k,a in pred_assets.items()}
    rows=[];shape_rows=[];components={};condition_checks={};cross=[];phys_rows=[]
    table,operator=physics(reg['physics_config'])
    for name,pred in predictions.items():
        assert pred.shape==(1,1,64,64,64) and pred.min()>=-1 and pred.max()<=13
        violation=int(((pred!=values)&m0).sum())
        if name in complete['arms'] and violation:
            raise ValueError('New prediction violates original conditions')
        condition_checks[name]=dict(original_condition_violations=violation,
            underground_air_voxels=int(((pred==-1)&sub).sum()))
        ac=hard_labels_to_acoustic(pred,table)
        field=operator(ac[:,0:1],ac[:,1:2],sub)
        response=operator.interface_response(ac[:,0:1],ac[:,1:2],sub)
        out_of_window=int((response[2]&(response[1]>operator.recording_end_ms)).sum())
        for key,y in obs.items():
            energy=float((field-y).square().sum()/y.square().sum().clamp_min(1e-12))
            phys_rows.append(dict(arm=name,field='hard_own_impedance_and_slowness',observation=key,
                energy=energy,relative_l2=energy**.5,out_of_window_arrivals=out_of_window,
                underground_air_voxels=condition_checks[name]['underground_air_voxels']))
        v=pred.numpy()[0,0]
        for world,truth_t in truths.items():
            truth=truth_t.numpy()[0,0]
            classes=sorted(int(c) for c in np.unique(truth) if c!=-1)
            for domain,mask in domains.items():
                d=mask.numpy()[0,0];metrics=voxel_metrics(v,truth,d,classes)
                metrics['target_pred_truth_volume_ratio']=ratio(metrics['target_pred_volume'],metrics['target_truth_volume'])
                rows.append(dict(arm=name,world=world,domain=domain,**metrics))
                if name in complete['arms'] and domain in ('U_common','U_original','CAP_UNION','GEOMETRY_DIFFERENCE'):
                    cross.append(dict(kind='structure',arm=name,reference=world,domain=domain,
                        target_iou=metrics['target_iou'],target_tp=metrics['target_tp'],
                        target_fp=metrics['target_fp'],target_fn=metrics['target_fn']))
            shape_rows.append(dict(arm=name,world=world,**boundary_summary(v,truth)))
            components[f'{name}_{world}']=component_summary(v,truth)
    # Optimized continuous endpoints are evaluated independently from hard labels.
    # No new model load: the checkpoint embedding is immutable in the original model.
    # The run saves the exact constant used for proximal soft decoding.
    embedding=verify_tensor(read_json(root/'runtime_constants.json')['embedding'])
    exact=hard_labels_to_acoustic(values,table)
    for name,receipt in complete['arms'].items():
        u=verify_tensor(receipt['assets']['prox_endpoint_fp64']).double()
        q=soft_decode_to_probs(u,embedding.double(),tau=.1)
        ac=probabilities_to_subsurface_acoustic(q,table,sub)
        ac=overwrite_exact_condition_acoustic(ac,exact,m0)
        field=operator(ac[:,0:1],ac[:,1:2],sub)
        for key,y in obs.items():
            energy=float((field-y).square().sum()/y.square().sum().clamp_min(1e-12))
            phys_rows.append(dict(arm=name,field='continuous_prox_endpoint_soft_nonair',observation=key,
                energy=energy,relative_l2=energy**.5,out_of_window_arrivals=None,
                underground_air_voxels=None))
    cross += [dict(kind='physics',arm=p['arm'],reference=p['observation'],domain=p['field'],
                   energy=p['energy'],relative_l2=p['relative_l2']) for p in phys_rows if p['arm'] in complete['arms']]
    def metric(arm,world,domain):
        return next(m for m in rows if m['arm']==arm and m['world']==world and m['domain']==domain)
    pair=[]
    for family in ('Fy','Fyr'):
        k1=family+'_G1';k2=family+'_G2'
        if k2 not in predictions:
            continue
        p1=predictions[k1];p2=predictions[k2]
        info=dict(family=family,G1_cap_target_voxels=int(((p1==9)&cap).sum()),
            G2_cap_target_voxels=int(((p2==9)&cap).sum()),
            target_occupancy_change_on_true_difference=int(((p2==9)&diff).sum())-int(((p1==9)&diff).sum()),
            all_class_changes_unchanged_region=int(((p1!=p2)&domains['UNCHANGED']).sum()),
            target_changes_unchanged_region=int((((p1==9)^(p2==9))&domains['UNCHANGED']).sum()),
            truth_cap_change=int(((truths['G2']==9)&cap).sum())-int(((truths['G1']==9)&cap).sum()))
        for domain in ('U_common','U_original'):
            i11=metric(k1,'G1',domain)['target_iou'];i12=metric(k1,'G2',domain)['target_iou']
            i22=metric(k2,'G2',domain)['target_iou'];i21=metric(k2,'G1',domain)['target_iou']
            info['D_struct_'+domain]=None if any(i is None for i in (i11,i12,i22,i21)) else .5*((i11-i12)+(i22-i21))
        pair.append(info)
    for filename,data in [('metrics',rows),('shape_metrics',shape_rows),('physical_errors',phys_rows),
                          ('cross_observation_matrix',cross),('paired_information',pair)]:
        write_csv(root/f'{filename}.csv',data)
    write_json(root/'condition_checks.json',condition_checks)
    write_json(root/'components.json',components)
    resource=read_json(root/'resource_usage.json')
    resource.update(evaluation_hard_physics_forwards=len(predictions),
        evaluation_continuous_physics_forwards=len(complete['arms']),
        observation_generation_physics_forwards=read_json(root/'OBSERVATION_FREEZE.json')['y2_generation_physics_forwards'])
    write_json(root/'resource_usage.json',resource)
    write_json(root/'EVALUATION.json',dict(status='completed',completed_at_utc=utc(),
        prediction_assets=pred_assets,metrics=rows,shape=shape_rows,physical=phys_rows,
        paired_information=pair,conditions=condition_checks,resources=resource,
        role='retrospective only; never ranks or modifies scientific outputs',
        soft_hard_gap='Soft physics excludes underground air; hard physics retains every air voxel',
        boundary_convention='6-neighbor boundary voxel centers; all 26-connected objects retained',
        G2_evaluations_if_invalid='G2 metrics are retrospective only; no y2 output if input check failed'))
    print('EVALUATION_COMPLETE',root,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path)
    main(p.parse_args().out)
