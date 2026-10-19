"""Isolated offline diagnostics; no model/checkpoint API and no formal inference registry."""
import os,sys
from datetime import datetime,timezone
from pathlib import Path
sys.dont_write_bytecode=True
from scripts.stage21.common import PROJECT,REPO,read_json,write_json,write_csv,file_hash,tensor_hash,json_hash,save_tensor,verify_tensor,git_snapshot
from scripts.stage21v2.common import check_condition
from guidance.seismic import acoustic_tables_from_config,seismic_operator_from_config,hard_labels_to_acoustic
FORMAL=PROJECT/'experiments/stage26_physical_seismic/run_20261019_v1'
STAGE25=PROJECT/'experiments/stage25_laccolith_shape/run_20261019_v3'
ROOT=PROJECT/'experiments/stage26_physical_seismic/diagnostics/run_20261019_v1'
SPACING=3840/63
FIELDS=('prior_exact','updated','oracle_updated')
CANDIDATE_NAMES=('FULL','OBSERVED')
def utc():return datetime.now(timezone.utc).isoformat()
def guard(mode):
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        writing=(isinstance(args[1],str) and any(c in args[1] for c in 'wax+')) or (isinstance(args[2],int) and args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        own=p==ROOT or ROOT in p.parents
        if p.suffix=='.ckpt':raise PermissionError('Diagnostics never load checkpoint')
        if writing and (PROJECT/'experiments') in p.parents and not own:raise PermissionError(f'Historical experiment read-only: {p}')
        if mode=='prepare' and ('truth' in p.parts or p.name in ('evaluation_registry.json','GEOMETRY.json') or (STAGE25/'arms/B') in p.parents or (STAGE25/'arms/C') in p.parents):raise PermissionError(f'Candidate selection cannot read truth/B/C: {p}')
    sys.addaudithook(audit)
def sources():
    paths=[Path(__file__).parent/name for name in ('common.py','prepare.py','oracle.py')]
    paths += [PROJECT/name for name in ('guidance/seismic.py','guidance/seismic_inversion.py','scripts/stage26/common.py','scripts/stage21/common.py','scripts/stage21v2/common.py')]
    return {str(p.relative_to(REPO)):file_hash(p) for p in paths}
def setup():
    tables,_=acoustic_tables_from_config(read_json(ROOT/'configs/acoustic_codebook.json'),15)
    operator,_=seismic_operator_from_config(read_json(ROOT/'configs/observation_config.json'))
    return tables.property_table.double(),operator
def candidates_from_conditions(values,mask):
    import torch
    return sorted(int(i) for i in torch.unique(values[mask]) if i>=0)
def classify(acoustic,table,candidates,subsurface,original_values,original_mask):
    import torch
    if not candidates or candidates!=sorted(set(candidates)):raise ValueError('Candidate labels must be nonempty, sorted, unique')
    raw=torch.tensor(candidates,dtype=torch.long,device=acoustic.device)
    logs=table[0,raw+1].log()[None,:,None,None,None]
    idx=(acoustic[:,0:1].log()-logs).abs().argmin(1,keepdim=True)
    labels=raw[idx]
    labels=torch.where(subsurface,labels,torch.full_like(labels,-1))
    return torch.where(original_mask,original_values.long(),labels)
def transfer(old,new,truth,domain):
    import torch
    changed=(old!=new)&domain;old_wrong=old!=truth;new_wrong=new!=truth
    d=dict(voxels=int(domain.sum()),changed=int(changed.sum()),wrong_to_right=int((changed&old_wrong&~new_wrong).sum()),right_to_wrong=int((changed&~old_wrong&new_wrong).sum()),wrong_to_wrong=int((changed&old_wrong&new_wrong).sum()))
    assert d['changed']==d['wrong_to_right']+d['right_to_wrong']+d['wrong_to_wrong']
    return d
