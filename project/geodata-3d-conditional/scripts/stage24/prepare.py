"""One unchanged Stage22 CPU solve per new case, then complete Stage23 pseudo-wells."""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage24.common import *
from scripts.stage22.solver import solve
from guidance.property_volume import property_table_from_config,hard_labels_to_properties
import torch

def prepare(case,out):
    guard('prepare')
    if case not in CASES:raise ValueError('Only case01/02')
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(4)
    ri=read_json(OLD/'inference_registry.json')
    # Only registered property metadata is used; no truth tensor can be opened by this process.
    prop_record=read_json(OLD/'evaluation_registry.json')['cases'][case]['properties']
    props=verify_tensor(prop_record)
    arms=ri['conditions'][case]
    original=check_condition(arms['A_9'])
    table,weights,_=property_table_from_config(read_json(OLD/'codebook.json'),15)
    scales=table[:,1:].max(1).values-table[:,1:].min(1).values
    classifier=read_json(OLD/'classifier.json')
    assert scales.tolist()==classifier['scales'] and weights.tolist()==classifier['weights']
    assert props.dtype==torch.float32 and props.shape==(1,2,64,64,64)
    previous=read_json(R22/'METHOD_FREEZE.json')
    assert all(file_hash(REPO/p)==h for p,h in previous['source_sha256'].items())
    assert file_hash(OLD/'codebook.json')==previous['codebook_sha256']
    paired=paired_noise(case)
    out.mkdir(parents=True)
    protocol=dict(case=case,seed=42,restoration='EXACT unchanged scripts/stage22/solver.py solve defaults',
        sigma=1.,radius=3,TV=1e-4,dtype='CPU float64; original float32 kernel coefficients',
        max_iterations=2000,stop='50 consecutive relative changes<1e-6; final accepted iterate',
        classification='original raw0..13 first argmin, fixed weights/nonair ranges',
        condition='Stage23 dense_conditions, old total512 columns, original hard priority',
        scientific_gate=False,max_new_solves=1,max_new_predictions=1,max_velocity_evaluations=32,
        specification_sha256=file_hash(REPO/'STAGE24_MULTI_CASE_EXECUTION_PROMPT_2026-10-17.md'))
    write_json(out/'PROTOCOL.json',protocol)
    source=sources()
    for rel in source:
        dest=out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/rel,dest)
    freeze=dict(case=case,source_sha256=source,protocol_sha256=json_hash(protocol),
        properties=prop_record,original_conditions=arms['A_9'],noise=paired,
        historical_source_sha256={**ri['source_sha256'],**previous['source_sha256']},
        metadata_sha256={str(p.relative_to(REPO)):file_hash(p) for p in (OLD/'layout.json',OLD/'classifier.json',OLD/'codebook.json',OLD/'evaluation_registry.json',R22/'PROTOCOL.json')},
        git=git_snapshot(),frozen_at_utc=utc(),compatible_existing_restoration='none in Stage22/23 or Stage24 namespace')
    write_json(out/'METHOD_FREEZE.json',freeze)
    mask=original['mask']
    y=props.double()/scales.double()[None,:,None,None,None]
    known=hard_labels_to_properties(original['values'],table).double()/scales.double()[None,:,None,None,None]
    lower=(table.double().min(1).values/scales.double())[None,:,None,None,None]
    upper=(table.double().max(1).values/scales.double())[None,:,None,None,None]
    start=time.monotonic()
    write_json(out/'SOLVER_STATE.json',dict(case=case,status='running',solve_invocations=1,started_at_utc=utc()))
    def progress(row):
        if row['iteration']%100==0:
            write_json(out/'SOLVER_PROGRESS.json',row)
            print(f"{case} iteration{row['iteration']}: objective={row['objective']:.8g} stable={row['stable_count']}",flush=True)
    try:
        z,history,converged=solve(y,lower,upper,known,mask,callback=progress)
        assert float((z-known).abs().masked_select(mask.expand_as(z)).max())==0
        assert not ((z<lower)|(z>upper)).any() and torch.isfinite(z).all()
        restored=(z*scales.double()[None,:,None,None,None]).float()
        labels=classify(restored,table,weights)
        layout=read_json(OLD/'layout.json')
        values,newmask=dense_conditions(original['values'],mask,labels,layout['levels']['512'])
        added=newmask&~mask
        for arm in ('B_512','C_512'):assert torch.equal(newmask,verify_tensor(arms[arm]['mask']))
        assert not (added&well_mask(layout['holdout'])).any()
        records={k:save_tensor(out/'conditions'/f'{k}.pt',v) for k,v in (
            ('values',values),('mask',newmask),('original_values',original['values']),('original_mask',mask))}
        check_condition(records)
        assets={k:save_tensor(out/'preprocessing'/f'{k}.pt',v) for k,v in (
            ('z',z),('properties',restored),('classification',labels))}
        added_asset=save_tensor(out/'conditions/added_mask.pt',added)
        reg=dict(schema='stage23_conditions_only_v1',protocol_sha256=json_hash(protocol),source_sha256=source,
            checkpoint=CHECKPOINT,checkpoint_sha256=CHECKPOINT_SHA,conditions=records,noise={'42':paired})
        write_json(out/'inference_registry.json',reg)
        write_csv(out/'solver_history.csv',history)
        write_json(out/'preprocessing_assets.json',assets)
        state=dict(case=case,status='completed' if converged else 'unconverged',solve_invocations=1,
            iterations=len(history)-1,converged=converged,final=history[-1],elapsed_seconds=time.monotonic()-start,
            assets=assets,truth_access=False,completed_at_utc=utc())
        write_json(out/'SOLVER_STATE.json',state)
        write_json(out/'CONDITION_RECEIPT.json',dict(status='completed' if converged else 'unconverged',case=case,
            converged=converged,sealed_at_utc=utc(),freeze_file_sha256=file_hash(out/'METHOD_FREEZE.json'),
            protocol_sha256=json_hash(protocol),inference_registry_sha256=file_hash(out/'inference_registry.json'),
            condition_record_sha256=json_hash(records),added_mask=added_asset,added_voxels=int(added.sum()),
            truth_access=False,scientific_gate=False))
        assert sources()==source
        if not converged:raise RuntimeError('Numerical nonconvergence: stop without changing scientific parameters')
        print(f'{case} sealed: {len(history)-1} iterations, {int(added.sum())} full-column added voxels',flush=True)
    except Exception as exc:
        write_json(out/'PREPARE_FAILURE.json',dict(error=str(exc),solve_invocations=1,failed_at_utc=utc()))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=CASES,required=True)
    a=p.parse_args();prepare(a.case,(ROOT/'cases'/a.case).resolve())
