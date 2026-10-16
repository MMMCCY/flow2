"""Case-parameterized Stage22/23 reuse with read-only history and bounded budgets."""
import os
import sys
from datetime import datetime,timezone
from pathlib import Path
sys.dont_write_bytecode=True
from scripts.stage21.common import (
    PROJECT,REPO,CHECKPOINT,CHECKPOINT_SHA,read_json,write_json,write_csv,file_hash,
    tensor_hash,json_hash,verify_tensor,save_tensor,well_mask,git_snapshot,
)
from scripts.stage21v2.common import check_condition,dense_conditions,classify
OLD=PROJECT/'experiments/stage21v2_dense_virtual_boreholes/run_20261016_v1'
R22=PROJECT/'experiments/stage22_blur_aware_bridge/run_20261016_v1'
R23=PROJECT/'experiments/stage23_full_pseudo_wells/run_20261017_v2'
ROOT=PROJECT/'experiments/stage24_multi_case/run_20261017_v1'
STANDARD=PROJECT/'visualization_standards/geology_approved_v1'
CASES=('fullgeo_case01','fullgeo_case02')
RULES=dict(edge_min_intersection=5,edge_min_truth_coverage=.05,detection_min_iou=.1,detection_min_coverage=.1)

def utc():
    return datetime.now(timezone.utc).isoformat()

def sources():
    paths=list(Path(__file__).parent.glob('*.py'))
    paths += [STANDARD/'renderer.py',STANDARD/'MANIFEST.json']
    paths += [PROJECT/p for p in ('scripts/stage22/solver.py','scripts/stage23/run.py',
        'scripts/stage23/common.py','scripts/stage23/evaluate.py','scripts/stage21/common.py',
        'scripts/stage21/run.py','scripts/stage21/evaluate.py','scripts/stage21v2/common.py',
        'guidance/property_volume.py','inference_runtime.py','model_train_sh_inference_cond.py',
        'guidance/dflow.py','guidance/generator_posterior.py','scripts/reports/build_stage24_approved_figures.py')]
    return {str(p.relative_to(REPO)):file_hash(p) for p in sorted(paths)}

def guard(mode):
    protected=[OLD,R22,R23,PROJECT/'experiments/stage23_full_pseudo_wells/reports',
        PROJECT/'scripts/stage21v2',PROJECT/'scripts/stage22',PROJECT/'scripts/stage23',STANDARD]
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        writing=((isinstance(args[1],str) and any(c in args[1] for c in 'wax+')) or
            (isinstance(args[2],int) and args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if writing and any(p==q or q in p.parents for q in protected):
            raise PermissionError(f'Historical/approved files are read-only: {p}')
        if mode in ('prepare','flow') and ('truth' in p.parts or
            ('geometry' in p.parts and 'experiments' in p.parts) or
            'stage21v2_evidence_diagnostics' in p.parts or p.name=='object_metrics.csv'):
            raise PermissionError(f'{mode} cannot open truth/objects/diagnostics: {p}')
        if mode!='flow' and p.suffix=='.ckpt':raise PermissionError('No offline checkpoint load')
        if mode=='flow' and ('preprocessing' in p.parts or p.name=='evaluation_registry.json'):
            raise PermissionError('Flow inputs are sealed conditions and noise only')
    sys.addaudithook(audit)

def paired_noise(case):
    path=OLD/'predictions'/case/'seed42/A_9/receipt.json'
    a=read_json(path)
    assert a['status']=='completed' and a['source_seed']==42 and a['case_id']==case
    assert read_json(path.parent/'completion_receipt.json')['receipt_sha256']==file_hash(path)
    if 'noise' in a:asset=a['noise']
    else:
        old=a['reuse_audit']['old_record']
        asset=dict(path='project/geodata-3d-conditional/experiments/stage21_virtual_borehole_conditioning/run_20261014_v2/formal/noise/seed42.pt',
            file_sha256=old['initial_noise_file_sha256'],tensor_sha256=old['initial_noise_tensor_sha256'])
    assert tensor_hash(verify_tensor(asset))==a['initial_noise_tensor_sha256']
    return dict(asset=asset,expected_model_hash=a['base_hash_after'],
        A_receipt_path=str(path.relative_to(REPO)),A_receipt_sha256=file_hash(path))
