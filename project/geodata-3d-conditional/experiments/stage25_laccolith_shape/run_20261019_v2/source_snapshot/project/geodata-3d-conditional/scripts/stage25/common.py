"""Stage25 isolated assets and truth-role firewall; historical science is read-only."""
import os, sys, time
from datetime import datetime, timezone
from pathlib import Path
sys.dont_write_bytecode=True
from scripts.stage21.common import *
from scripts.stage21v2.common import check_condition, dense_conditions, classify
ROOT=PROJECT/'experiments/stage25_laccolith_shape/run_20261019_v2'
OLD=PROJECT/'experiments/stage21v2_dense_virtual_boreholes/run_20261016_v1'
R22=PROJECT/'experiments/stage22_blur_aware_bridge/run_20261016_v1'
STANDARD=PROJECT/'visualization_standards/geology_approved_v1'
SPACING=3840/63
EXPECTED_MODEL='10d2181fe3eeeb406afc3da5e78ded64e9f92abfb2b3c352ee57e60e06da9b68'
def utc(): return datetime.now(timezone.utc).isoformat()
def sources():
    result=source_hashes()
    paths=list((PROJECT/'scripts/stage25').glob('*.py'))
    paths += [PROJECT/p for p in ('scripts/stage22/solver.py','scripts/stage21v2/common.py','scripts/stage23/run.py','scripts/stage23/evaluate.py','scripts/stage24/plot.py','scripts/stage21/evaluate.py','guidance/property_volume.py','boreholes.py')]
    paths += list((REPO/'StructuralGeo-main/src/geogen/model').glob('*.py'))
    paths += [STANDARD/'renderer.py',STANDARD/'MANIFEST.json']
    result.update({str(p.relative_to(REPO)):file_hash(p) for p in paths})
    return result
def guard(mode):
    protected=[PROJECT/'scripts'/s for s in ('stage21v2','stage22','stage23','stage24')]
    protected += [STANDARD]+[PROJECT/'experiments'/s for s in ('stage21v2_dense_virtual_boreholes','stage22_blur_aware_bridge','stage23_full_pseudo_wells','stage24_multi_case')]
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)): return
        p=Path(os.fsdecode(args[0])).resolve()
        writing=(isinstance(args[1],str) and any(c in args[1] for c in 'wax+')) or (isinstance(args[2],int) and args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if writing and any(p==q or q in p.parents for q in protected): raise PermissionError(f'Sealed history: {p}')
        if mode in ('restore','flow') and ('truth' in p.parts or ('geometry' in p.parts and 'experiments' in p.parts) or p.name in ('evaluation_registry.json','GEOMETRY.json')): raise PermissionError(f'Truth forbidden in {mode}: {p}')
        if mode=='flow' and ('preprocessing' in p.parts or 'observations' in p.parts): raise PermissionError(f'Flow sees conditions/noise only: {p}')
        if mode!='flow' and p.suffix=='.ckpt': raise PermissionError('No offline checkpoint load')
    sys.addaudithook(audit)
def seal_arm(arm,values,mask,original_values,original_mask,noise,source,protocol):
    out=ROOT/'arms'/arm
    records={k:save_tensor(out/'conditions'/f'{k}.pt',v) for k,v in dict(values=values,mask=mask,original_values=original_values,original_mask=original_mask).items()}
    check_condition(records)
    r=dict(schema='stage23_conditions_only_v1',protocol_sha256=json_hash(protocol),source_sha256=source,checkpoint=CHECKPOINT,checkpoint_sha256=CHECKPOINT_SHA,conditions=records,noise={'42':dict(asset=noise,expected_model_hash=EXPECTED_MODEL)})
    write_json(out/'inference_registry.json',r)
    write_json(out/'CONDITION_RECEIPT.json',dict(arm=arm,sealed_at_utc=utc(),inference_registry_sha256=file_hash(out/'inference_registry.json'),condition_record_sha256=json_hash(records)))
    return records
