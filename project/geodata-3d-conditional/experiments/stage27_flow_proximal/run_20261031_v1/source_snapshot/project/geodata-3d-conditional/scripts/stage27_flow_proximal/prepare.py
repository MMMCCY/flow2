"""Observation-generation role only. Freeze one prescribed G2 before inference."""
import argparse
import os
import sys
import shutil
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
PROJECT=Path(__file__).resolve().parents[2]
for p in (PROJECT,PROJECT.parents[1],PROJECT.parents[1]/'src',PROJECT.parents[1]/'StructuralGeo-main/src'):
    sys.path.insert(0,str(p))
import numpy as np
import torch
from .common import *
from guidance.seismic import hard_labels_to_acoustic

def build_g2():
    from geogen.model.geomodel import GeoModel
    from geogen.model.geoprocess import Bedrock,Sedimentation,Laccolith,UnconformityBase
    np.random.seed(20261019)
    coords=np.linspace(-1920,1920,64,dtype=np.float32)
    origin=tuple(float(coords[i]) for i in (35,26,28))
    model=GeoModel(bounds=(-1920,1920),resolution=(64,64,64),height_tracking=False,name='stage27_g2')
    model.add_history([Bedrock(float(coords[12]),0),
        Sedimentation([1,3,5],[8*SPACING,6*SPACING,35*SPACING],base=float(coords[12])),
        Laccolith(origin=origin,cap_diam=18*SPACING,stem_diam=4.4*SPACING,
                  height=8*SPACING,minor_axis_scale=1.,rotation=0.,value=9,upper=True,clip=False),
        UnconformityBase(float(coords[52]))])
    model.compute_model(keep_snapshots=False,normalize=False)
    return torch.from_numpy(np.nan_to_num(model.get_data_grid(),nan=-1).astype(np.int64))[None,None]

def main(root):
    root=root.resolve()
    if root.exists():
        raise FileExistsError('New immutable run directory required')
    root.mkdir(parents=True)
    install_firewall(root)
    torch.set_num_threads(4)
    write_json(root/'START.json',dict(started_at_utc=utc(),git=git_snapshot(),host=os.uname().nodename,
        stage='preparation',scientific_model_calls=0))
    write_json(root/'config.json',CONFIG)
    frozen_sources=sources()
    for rel in frozen_sources:
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(REPO/rel,dest)
    spec=REPO/'STAGE27_FLOW_PROXIMAL_EXECUTION_PROMPT_2026-10-30.md'
    shutil.copyfile(spec,root/'SPECIFICATION.md')
    old=read_json(R26/'INPUT_FREEZE.json')
    conditions={k:old['original_conditions'][k] for k in ('values','mask')}
    values=verify_tensor(conditions['values']);mask=verify_tensor(conditions['mask']).bool()
    sub=verify_tensor(old['subsurface']).bool()
    initial=verify_tensor(old['noise']);obs=verify_tensor(old['observation'])
    assert initial.shape==(1,15,64,64,64) and initial.dtype==torch.float32
    assert obs.shape==(1,1,64,64,400) and obs.dtype==torch.float64
    assert torch.equal(sub,~(mask&(values==-1)))
    if file_hash(REPO/CHECKPOINT)!=CHECKPOINT_SHA:
        raise ValueError('Checkpoint identity')
    pc={k:read_json(R26/'configs'/f'{name}.json') for k,name in
        [('acoustic','acoustic_codebook'),('observation','observation_config')]}
    for name,h in old['config_sha256'].items():
        assert file_hash(R26/'configs'/name)==h
    assert pc['observation']['sample_mask']=='all'
    assert pc['observation']['time_sampling']==dict(num_samples=400,sample_interval_ms=8.)
    assert pc['observation']['noise']['type']=='none'
    write_json(root/'physics_config.json',pc)
    # Only this process opens truth, old geometry or evaluation registries.
    evaluation=read_json(R25/'evaluation_registry.json')
    g1=verify_tensor(evaluation['truth']); g2=build_g2()
    assert torch.equal(g1[mask],values[mask])
    conflicts=mask&(g1!=g2)
    same_surface=torch.equal(g1==-1,g2==-1)
    # G2 is nested inside G1's predefined cap box. Union of their nominal boxes
    # is the unchanged G1 box, fixed before any generated predictions.
    capbox=evaluation['geometry']['cap_evaluation_box_indices']
    cap=torch.zeros_like(mask)
    cap[:,:,capbox['x'][0]:capbox['x'][1]+1,capbox['y'][0]:capbox['y'][1]+1,
        capbox['z'][0]:capbox['z'][1]+1]=True
    differing=g1!=g2
    checks=dict(shape_equal=g1.shape==g2.shape,condition_equal=not bool(conflicts.any()),
        condition_conflict_count=int(conflicts.sum()),condition_conflict_indices=conflicts.nonzero().tolist(),
        condition_conflict_G1=g1[conflicts].tolist(),condition_conflict_G2=g2[conflicts].tolist(),
        subsurface_equal=same_surface,cap_target_difference_voxels=int(((g1==9)^(g2==9))[cap].sum()),
        all_label_difference_voxels=int(differing.sum()),
        prescribed_change='cap_diam 22*spacing -> 18*spacing only; seed20261019 unchanged',
        g1_free_target_voxels=int(((g1==9)&sub&~mask).sum()),
        g2_free_target_voxels=int(((g2==9)&sub&~mask).sum()))
    checks['valid_pair']=checks['shape_equal'] and checks['condition_equal'] and same_surface and checks['cap_target_difference_voxels']>0
    g2_asset=save_tensor(root/'truth/G2.pt',g2)
    diff_asset=save_tensor(root/'evaluation/geometry_difference.pt',differing)
    cap_asset=save_tensor(root/'evaluation/cap_union_box.pt',cap)
    write_json(root/'PAIR_CHECKS.json',checks)
    observations={'y1':old['observation']}
    generated_forward=0
    if checks['valid_pair']:
        table,operator=physics(pc)
        ac=hard_labels_to_acoustic(g2,table)
        with torch.no_grad():
            y2=operator(ac[:,0:1],ac[:,1:2],sub)
        generated_forward=1
        observations['y2']=save_tensor(root/'observations/y2.pt',y2)
    m512=read_json(R25/'arms/B/inference_registry.json')['conditions']['mask']
    layout=read_json(R25/'layout.json')
    assert file_hash(R25/'layout.json')==old['layout_sha256']
    write_json(root/'layout.json',layout)
    write_json(root/'evaluation_registry.json',dict(truths={'G1':evaluation['truth'],'G2':g2_asset},
        cap_G1_box=capbox,cap_union=cap_asset,difference=diff_asset,M512=m512,
        holdout=layout['holdout'],valid_pair=checks['valid_pair'],
        truth_role='synthetic observation generation / retrospective evaluation only',
        freeze_time_utc=utc(),predictions_generated_before_freeze=0))
    registry=dict(schema='stage27_inference_v1',config=CONFIG,config_sha256=json_hash(CONFIG),
        checkpoint=CHECKPOINT,checkpoint_sha256=CHECKPOINT_SHA,conditions=conditions,
        subsurface=old['subsurface'],noise=old['noise'],observations=observations,
        physics_config=pc,source_sha256=frozen_sources,layout_sha256=old['layout_sha256'])
    validate_registry(json_roundtrip(registry))
    write_json(root/'inference_registry.json',registry)
    write_json(root/'OBSERVATION_FREEZE.json',dict(observations=observations,
        old_y1_freeze_sha256=file_hash(R25/'physics/OBSERVATION_FREEZE.json'),
        y1_recomputed=False,y2_generation_physics_forwards=generated_forward,
        new_traditional_inversions=0,registry_sha256=file_hash(root/'inference_registry.json'),
        source_sha256=frozen_sources,config_sha256=json_hash(CONFIG),frozen_at_utc=utc()))
    print('INPUTS_FROZEN',root,'valid_pair',checks['valid_pair'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
