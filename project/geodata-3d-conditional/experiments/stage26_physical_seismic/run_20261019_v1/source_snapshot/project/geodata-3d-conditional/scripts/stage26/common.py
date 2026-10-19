"""Stage26 contracts: reusable old assets, isolated new outputs and process roles."""
import os,sys
from datetime import datetime,timezone
from pathlib import Path
sys.dont_write_bytecode=True
from scripts.stage21.common import *
from scripts.stage21v2.common import check_condition,dense_conditions
from guidance.seismic import acoustic_tables_from_config,seismic_operator_from_config,hard_labels_to_acoustic
OLD=PROJECT/'experiments/stage25_laccolith_shape/run_20261019_v3'
OLD_PHYS=OLD/'physics'
ROOT=PROJECT/'experiments/stage26_physical_seismic/run_20261019_v1'
STANDARD=PROJECT/'visualization_standards/geology_approved_v1'
SPACING=3840/63
EXPECTED_MODEL='10d2181fe3eeeb406afc3da5e78ded64e9f92abfb2b3c352ee57e60e06da9b68'
def utc():return datetime.now(timezone.utc).isoformat()
def sources():
    result=source_hashes()
    paths=[Path(__file__).parent/name for name in ('common.py','prepare.py','run.py')]
    paths += [PROJECT/name for name in ('guidance/seismic.py','guidance/seismic_inversion.py','scripts/stage21v2/common.py')]
    result.update({str(p.relative_to(REPO)):file_hash(p) for p in paths})
    return result

def guard(mode):
    """Own outputs can be written AND read back; historical science remains read-only."""
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        writing=(isinstance(args[1],str) and any(c in args[1] for c in 'wax+')) or (isinstance(args[2],int) and args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        in_own=p==ROOT or ROOT in p.parents
        if writing and ((PROJECT/'experiments') in p.parents or STANDARD in p.parents) and not in_own:raise PermissionError(f'Historical outputs are read-only: {p}')
        if mode in ('prepare','flow'):
            if 'truth' in p.parts or ('geometry' in p.parts and 'experiments' in p.parts) or p.name in ('evaluation_registry.json','GEOMETRY.json'):raise PermissionError(f'Truth unavailable in {mode}: {p}')
            for arm in ('B','C'):
                q=OLD/'arms'/arm
                if p==q or q in p.parents:raise PermissionError(f'No historical {arm} labels in inference: {p}')
            if not in_own and ('preprocessing' in p.parts or p.name=='gaussian_sigma1.pt'):raise PermissionError(f'No old Gaussian fields: {p}')
        if mode=='flow' and ('preprocessing' in p.parts or 'observations' in p.parts or p.name=='numerical_bundle.pt'):raise PermissionError(f'Flow conditions/noise only: {p}')
        if mode!='flow' and p.suffix=='.ckpt':raise PermissionError('No offline checkpoint loading')
    sys.addaudithook(audit)

def setup():
    tables,_=acoustic_tables_from_config(read_json(ROOT/'configs/acoustic_codebook.json'),15)
    operator,_=seismic_operator_from_config(read_json(ROOT/'configs/observation_config.json'))
    return tables.property_table.double(),operator

def classify_impedance(acoustic,table,subsurface,original_values,original_mask):
    import torch
    candidates=table[0,1:].log()[None,:,None,None,None]
    labels=(acoustic[:,0:1].log()-candidates).abs().argmin(1,keepdim=True).long()
    labels=torch.where(subsurface,labels,torch.full_like(labels,-1))
    return torch.where(original_mask,original_values.long(),labels)

def relative_l2(field,observation):
    import torch
    if field.shape!=observation.shape or not torch.isfinite(field).all():raise ValueError('Physical field shape/finiteness')
    norm=observation.double().square().sum().sqrt()
    if norm<=0:raise ValueError('Zero observation norm')
    return float((field.double()-observation.double()).square().sum().sqrt()/norm)

def seal_arm(arm,values,mask,original,noise,source,protocol):
    out=ROOT/'arms'/arm
    records={k:save_tensor(out/'conditions'/f'{k}.pt',v) for k,v in dict(values=values,mask=mask,original_values=original['values'],original_mask=original['mask']).items()}
    check_condition(records)
    r=dict(schema='stage23_conditions_only_v1',protocol_sha256=json_hash(protocol),source_sha256=source,checkpoint=CHECKPOINT,checkpoint_sha256=CHECKPOINT_SHA,conditions=records,noise={'42':dict(asset=noise,expected_model_hash=EXPECTED_MODEL)})
    write_json(out/'inference_registry.json',r)
    write_json(out/'CONDITION_RECEIPT.json',dict(arm=arm,sealed_at_utc=utc(),inference_registry_sha256=file_hash(out/'inference_registry.json'),condition_record_sha256=json_hash(records)))
    return records
