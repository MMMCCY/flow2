"""Observation-generation only: one fixed native laccolith and oracle well samples."""
import os,sys,time,shutil
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage25.common import *
sys.path.insert(0,str(REPO/'StructuralGeo-main/src'))
import numpy as np
import torch
from scipy import ndimage
from geogen.model.geomodel import GeoModel
from geogen.model.geoprocess import Bedrock,Sedimentation,Laccolith,UnconformityBase
from boreholes import make_surface_mask
from guidance.property_volume import property_table_from_config,hard_labels_to_properties,gaussian_blur_property_channels
from scripts.stage21.run import initial_noise

def build_geometry():
    np.random.seed(20261019)
    coords=np.linspace(-1920,1920,64,dtype=np.float32)
    origin=(float(coords[35]),float(coords[26]),float(coords[28]))
    params=dict(seed=20261019,bounds=[[-1920,1920]]*3,shape=[64]*3,axes='XYZ, increasing Z upward',grid='np.linspace includes both endpoints; cell indices reference sample locations',spacing_m=[SPACING]*3,
        center_xy_indices=[35,26],cap_base_z_index=28,nominal_top_z_index=36,cap_diameter_voxels=22,cap_height_voxels=8,stem_diameter_voxels=4.4,origin_m=list(origin),
        cap_diameter_m=22*SPACING,height_m=8*SPACING,stem_diameter_m=4.4*SPACING,minor_axis_scale=1.,rotation=0.,raw_target=9,stem_depth='native np.inf, extends to bottom of grid',
        surface_z_index=52,normalization=False,geological_seed_search=False,technical_geometry_revisions=0,
        events=[dict(type='Bedrock',base_m=float(coords[12]),value=0),dict(type='Sedimentation',base_m=float(coords[12]),thicknesses_m=[8*SPACING,6*SPACING,35*SPACING],values=[1,3,5]),
        dict(type='Laccolith',native_components=['PushHemisphere','DikeHemisphere','DikeColumn']),dict(type='UnconformityBase',base_m=float(coords[52]),value='NaN/air')],
        cap_evaluation_box_indices=dict(x=[24,46],y=[15,37],z=[28,36]),section=dict(axis='Y',index=26))
    model=GeoModel(bounds=(-1920,1920),resolution=(64,64,64),height_tracking=False,name='stage25_native_laccolith')
    model.add_history([Bedrock(float(coords[12]),0),Sedimentation([1,3,5],[8*SPACING,6*SPACING,35*SPACING],base=float(coords[12])),
        Laccolith(origin=origin,cap_diam=22*SPACING,stem_diam=4.4*SPACING,height=8*SPACING,minor_axis_scale=1.,rotation=0.,value=9,upper=True,clip=False),UnconformityBase(float(coords[52]))])
    model.compute_model(keep_snapshots=False,normalize=False)
    labels=torch.from_numpy(np.nan_to_num(model.get_data_grid(),nan=-1).astype(np.int64))[None,None]
    return labels,params

def main():
    guard('generate');torch.set_num_threads(4)
    if ROOT.exists(): raise FileExistsError(ROOT)
    start=time.monotonic();ROOT.mkdir(parents=True)
    write_json(ROOT/'START.json',dict(started_at_utc=utc(),budget_seconds=10800,git=git_snapshot(),read_only_review_complete=True,remote_host='LabRS',max_flow_outputs=4,max_velocity_evaluations=128))
    labels,params=build_geometry();v=labels.numpy()[0,0];target=v==9
    _,n=ndimage.label(target,np.ones((3,3,3)))
    hits=[dict(x=x,y=y,target_voxels=int(target[x,y].sum()),cap_voxels=int(target[x,y,29:37].sum())) for x,y in W9]
    assert n==1 and target.any() and not target[[0,-1],:,:].any() and not target[:,[0,-1],:].any()
    assert not target[:,:,52:].any() and target[35,26,29:36].all() and target[35,26,:29].all()
    assert sum(h['target_voxels']>0 for h in hits)==1
    surface=make_surface_mask(labels).bool()
    # Surface observation includes air and the immediately underlying material.
    assert not (surface& (labels==9)).any()
    original=condition_assets(labels,surface,W9);ov=original['condition_values'];om=original['condition_mask']
    layout=read_json(OLD/'layout.json');assert layout['original_wells']==[list(p) for p in W9]
    bv,bm=dense_conditions(ov,om,labels,layout['levels']['512'])
    assert not ((bm&~om)&well_mask(layout['holdout'])).any()
    truth=save_tensor(ROOT/'truth/labels.pt',labels)
    codebook=read_json(OLD/'codebook.json');table,weights,_=property_table_from_config(codebook,15)
    obs=gaussian_blur_property_channels(hard_labels_to_properties(labels,table),1.)
    obs_asset=save_tensor(ROOT/'observations/gaussian_sigma1.pt',obs)
    noise=save_tensor(ROOT/'source/seed42.pt',initial_noise(42,20,(64,64,64)))
    write_json(ROOT/'codebook.json',codebook);write_json(ROOT/'layout.json',layout)
    write_json(ROOT/'GEOMETRY.json',params)
    protocol=dict(case='native_laccolith_20261019',seed=42,shape=[64]*3,max_main_outputs=3,max_main_velocity_evaluations=96,steps=32,sampler='unchanged Stage21 sample_native; midpoint-time Euler',model='411EMA/frozen checkpoint embedding',precision='FP32/noTF32/noautocast',
        max_restoration_solves=1,solver='unchanged Stage22 solve defaults, CPU FP64',sigma=1.,radius=3,TV=1e-4,max_iterations=2000,stop='50 consecutive relative objective changes <1e-6; final iterate',
        classifier='raw0..13; original normalized weighted distance; first argmin',total_dense_columns=512,holdout_columns=256,domains='(truth!=-1)&~original_mask; and intersection with fixed holdout columns',
        class_set='nonair truth classes present in each fixed domain, frozen before inference',surface='existing make_surface_mask from observed flat surface; all air and immediately lower voxel',
        surface_distance='ASSD = mean over concatenated bidirectional nearest-boundary distance samples, 6-neighbor boundary voxels, actual physical spacing, all target components; empty -> NA',
        specification_sha256=file_hash(REPO/'STAGE25_LACCOLITH_SHAPE_EXECUTION_PROMPT_2026-10-19.md'),optional_physics='only after MANDATORY_COMPLETE and remaining>=4500s; max one inversion and D/32 calls')
    write_json(ROOT/'PROTOCOL.json',protocol)
    source=sources()
    for rel in source:
        dest=ROOT/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/rel,dest)
    seal_arm('A',ov,om,ov,om,noise,source,protocol)
    seal_arm('B',bv,bm,ov,om,noise,source,protocol)
    domains={}
    for name,domain in dict(E_original_free=(labels!=-1)&~om,E_holdout=(labels!=-1)&~om&well_mask(layout['holdout'])).items():
        domains[name]=dict(mask=save_tensor(ROOT/'evaluation'/f'{name}.pt',domain),classes=sorted(int(x) for x in torch.unique(labels[domain]) if x!=-1))
    write_json(ROOT/'evaluation_registry.json',dict(truth=truth,domains=domains,geometry=params,role='observation generation and retrospective evaluation only'))
    write_json(ROOT/'GENERATION_CHECKS.json',dict(components=n,target_voxels=int(target.sum()),original_well_hits=hits,hit_well_count=sum(h['target_voxels']>0 for h in hits),surface_has_target=False,cap_sides_unconstrained_by_other_wells=True,elapsed_seconds=time.monotonic()-start))
    write_json(ROOT/'OBSERVATION_FREEZE.json',dict(sealed_at_utc=utc(),observations=obs_asset,noise=noise,source_sha256=source,protocol_sha256=json_hash(protocol),codebook_sha256=file_hash(ROOT/'codebook.json'),layout_sha256=file_hash(ROOT/'layout.json'),geometry_sha256=file_hash(ROOT/'GEOMETRY.json'),truth=truth,truth_role='synthetic generation/retrospective only',git=git_snapshot()))
    print('GEOMETRY_OBSERVATION_FROZEN',int(target.sum()),hits,flush=True)
if __name__=='__main__': main()
