"""Fixed contracts and minimal input allowlist; no historical main entrypoints."""
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
import torch
from scripts.stage21.common import (REPO, PROJECT, CHECKPOINT, CHECKPOINT_SHA,
    file_hash, tensor_hash, json_hash, read_json, write_json, write_csv,
    verify_tensor, git_snapshot, source_hashes, well_mask)

R25 = PROJECT/'experiments/stage25_laccolith_shape/run_20261019_v3'
R26 = PROJECT/'experiments/stage26_physical_seismic/run_20261019_v1'
STANDARD = PROJECT/'visualization_standards/geology_approved_v1'
SPACING = 3840/63
EXPECTED_MODEL = '10d2181fe3eeeb406afc3da5e78ded64e9f92abfb2b3c352ee57e60e06da9b68'
ARMS = [('F0',0,0,None), ('Fr',0,1,None), ('Fy_G1',1,0,'y1'),
        ('Fyr_G1',1,1,'y1'), ('Fy_G2',1,0,'y2'), ('Fyr_G2',1,1,'y2')]
CONFIG = dict(N=32, initial_source='old saved seed42', noise_seed_base=1042,
    noise_prng='torch.randn CPU FP32; fresh CPU torch.Generator.manual_seed(1042+j)',
    time_eval='clamp(j/32, checkpoint.time_range)', time_next='(j+1)/32',
    tau=.1, prox_steps=3, prox_gradient_scale='15*U.sum()', direction_rms_cap=.1,
    line_search_steps=[1.,.5,.25,.125,.0625,.03125], armijo_c=1e-4,
    lambda_structure=.05, tv_smoothing_eps=1e-6, energy_eps=1e-12,
    real_condition_projection='every stage; all non-U restored to embedding B',
    target_raw=9, target_category=10, G2_cap_diameter_voxels=18,
    same_model_weights=True, parameter_sweep=False, new_traditional_inversions=0,
    max_network_evaluations=192, max_physics_gradients=384,
    max_internal_physics_forwards=2688, precision_network='FP32_noTF32_noautocast',
    precision_prox_physics='FP64', new_observation_noise='none',
    source_seed=42, geometry_shape=[64,64,64], spacing_m=[SPACING]*3,
    nu2='(1-t_eval)^2/(t_eval^2+(1-t_eval)^2)', arms=ARMS)

def utc():
    return datetime.now(timezone.utc).isoformat()

def save_tensor(path, tensor):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    value=tensor.detach().cpu().contiguous()
    tmp=path.with_suffix('.partial'); torch.save(value,tmp); tmp.replace(path)
    return dict(path=str(path.relative_to(REPO)), file_sha256=file_hash(path),
                tensor_sha256=tensor_hash(value))

def sources():
    paths=list(Path(__file__).parent.glob('*.py'))
    paths += [PROJECT/p for p in ('guided_geophysical_sampling.py','guidance/seismic.py',
        'scripts/stage25/evaluate.py','scripts/stage24/plot.py','scripts/stage25/plot.py',
        'scripts/stage21/evaluate.py')]
    paths += list((REPO/'StructuralGeo-main/src/geogen/model').glob('*.py'))
    return {**source_hashes(), **{str(p.relative_to(REPO)):file_hash(p) for p in paths}}

def physics(config):
    from guidance.seismic import acoustic_tables_from_config, seismic_operator_from_config
    tables,_=acoustic_tables_from_config(config['acoustic'],15)
    operator,_=seismic_operator_from_config(config['observation'])
    return tables.property_table.double(),operator

def install_firewall(root, allowed_reads=None):
    """Historical results read-only; inference tensor inputs strictly enumerated."""
    root=Path(root).resolve()
    allowed=None if allowed_reads is None else {Path(p).resolve() for p in allowed_reads}
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        p=Path(os.fsdecode(args[0])).resolve()
        writing=(isinstance(args[1],str) and any(c in args[1] for c in 'wax+')) or (
            isinstance(args[2],int) and args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        own=p==root or root in p.parents
        if writing and ((PROJECT/'experiments') in p.parents or STANDARD in p.parents) and not own:
            raise PermissionError(f'Historical assets are read-only: {p}')
        if allowed is not None and not writing:
            if 'truth' in p.parts or p.name in ('evaluation_registry.json','GEOMETRY.json','PAIR_CHECKS.json'):
                raise PermissionError(f'Truth/geometry unavailable in inference: {p}')
            if p.suffix in ('.pt','.ckpt','.pth') and p not in allowed and not own:
                raise PermissionError(f'Unlisted inference tensor: {p}')
    sys.addaudithook(audit)

def validate_registry(reg):
    keys={'schema','config','config_sha256','checkpoint','checkpoint_sha256','conditions',
          'subsurface','noise','observations','physics_config','source_sha256','layout_sha256'}
    if set(reg)!=keys or reg['schema']!='stage27_inference_v1':
        raise ValueError('Inference registry allowlist mismatch')
    if reg['config']!=json_roundtrip(CONFIG) or reg['config_sha256']!=json_hash(CONFIG):
        raise ValueError('Fixed protocol mismatch')
    if reg['checkpoint']!=CHECKPOINT or reg['checkpoint_sha256']!=CHECKPOINT_SHA:
        raise ValueError('Checkpoint contract mismatch')
    if set(reg['conditions'])!={'values','mask'} or not set(reg['observations']) <= {'y1','y2'}:
        raise ValueError('Condition/observation allowlist mismatch')

def json_roundtrip(value):
    import json
    return json.loads(json.dumps(value))
