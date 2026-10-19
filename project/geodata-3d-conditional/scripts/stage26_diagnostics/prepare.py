"""Truth-blind candidate derivation and exact replay of saved FULL classification."""
import sys,os,shutil
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage26_diagnostics.common import *
from scripts.stage26.common import classify_impedance as original_classifier
import torch

def main():
    guard('prepare');torch.set_num_threads(4);ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'INPUT_FREEZE.json').exists():raise FileExistsError('Prepared diagnostic inputs are immutable')
    write_json(ROOT/'START.json',dict(started_at_utc=utc(),budget_seconds=3600,git=git_snapshot(),scope='offline diagnostic only',new_Flow=0,checkpoint_loads=0,max_new_oracle_inversions=1))
    fi=read_json(FORMAL/'INPUT_FREEZE.json');inv=read_json(FORMAL/'INVERSION_STATE.json');seal=read_json(FORMAL/'CONDITIONS_FROZEN.json')
    assert inv['status']=='completed' and read_json(FORMAL/'FINAL_CHECKS.json')['status']=='passed'
    for name,h in fi['config_sha256'].items():
        src=FORMAL/'configs'/name;assert file_hash(src)==h
        dest=ROOT/'configs'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
    table,operator=setup();a=check_condition(fi['original_conditions']);sub=verify_tensor(fi['subsurface'])
    assert torch.equal(sub,~(a['mask']&(a['values']==-1)))
    # Candidate selection is completed before any full truth or evaluation-domain read.
    observed=candidates_from_conditions(a['values'],a['mask']);candidates=dict(FULL=list(range(14)),OBSERVED=observed)
    counts=[]
    for name,labels in candidates.items():
        for label in labels:counts.append(dict(candidate_set=name,raw_label=label,original_condition_voxels=int(((a['values']==label)&a['mask']).sum()),source='fixed raw0..13 codebook' if name=='FULL' else 'ONLY original values within original mask; no truth/B/C'))
    write_json(ROOT/'CANDIDATES.json',dict(candidates=candidates,selected_at_utc=utc(),source_condition=fi['original_conditions'],used_full_truth=False,used_B_C=False,added_missing_target=False,not_a_deployable_universal_class_prior=True))
    write_csv(ROOT/'candidate_sources.csv',counts)
    condition=hard_labels_to_acoustic(a['values'],table)
    checks={};prior=verify_tensor(inv['assets']['prior_exact']);updated=verify_tensor(inv['assets']['updated'])
    assert prior.shape==updated.shape==(1,2,64,64,64) and prior.dtype==updated.dtype==torch.float64
    assert torch.equal(prior[:,1:2],updated[:,1:2])
    for field,asset,key in [('prior_exact',prior,'classification_D0'),('updated',updated,'classification_D')]:
        assert torch.isfinite(asset).all() and (asset>0).all()
        assert torch.equal(asset[a['mask'].expand_as(asset)],condition[a['mask'].expand_as(condition)])
        pred=classify(asset,table,candidates['FULL'],sub,a['values'],a['mask'])
        assert torch.equal(pred,original_classifier(asset,table,sub,a['values'],a['mask']))
        assert torch.equal(pred,verify_tensor(inv['assets'][key]))
        checks[field]='exact_tensor_equal_to_original_and_saved_labels'
    d=check_condition(seal['conditions']['D']);d0=check_condition(seal['conditions']['D0'])
    assert torch.equal(d['mask'],d0['mask']) and torch.equal(d['original_mask'],a['mask'])
    source=sources()
    for rel in source:
        dest=ROOT/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/rel,dest)
    meta={name:file_hash(FORMAL/name) for name in ('REPORT.md','INPUT_FREEZE.json','INVERSION_STATE.json','CONDITIONS_FROZEN.json','EVALUATION.json','FINAL_CHECKS.json')}
    write_json(ROOT/'INPUT_FREEZE.json',dict(prepared_at_utc=utc(),source_sha256=source,old_formal_metadata_sha256=meta,old_assets=inv['assets'],observation=fi['observation'],subsurface=fi['subsurface'],original_conditions=fi['original_conditions'],dense_conditions=seal['conditions'],candidates=candidates,candidate_file_sha256=file_hash(ROOT/'CANDIDATES.json'),config_sha256=fi['config_sha256'],reproduction=checks,specification_sha256=file_hash(REPO/'STAGE26_DIAGNOSTICS_TASK_2026-10-19.md'),roles=dict(truth='evaluation; ORACLE slowness only in separate entry',oracle='ORACLE_DIAGNOSTIC_ONLY; never sent to Flow'),git=git_snapshot()))
    print('CANDIDATES_FROM_ORIGINAL_CONDITIONS',observed,flush=True)
    print('SAVED_FULL_CLASSIFICATIONS_REPRODUCED',checks,flush=True)
if __name__=='__main__':main()
