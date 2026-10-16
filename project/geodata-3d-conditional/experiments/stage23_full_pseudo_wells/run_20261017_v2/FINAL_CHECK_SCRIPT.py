"""Final CPU identity/accounting checks; no checkpoint load or network evaluation."""
import os,sys,json,shutil
from pathlib import Path
from datetime import datetime
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
repo=Path('/home/xmj/mcy/flowtrain_stochastic_interpolation-main')
sys.path.insert(0,str(repo/'project/geodata-3d-conditional'))
from scripts.stage23.common import *
import torch
import numpy as np
torch.set_num_threads(4)
r=ROOT
freeze=read_json(r/'METHOD_FREEZE.json');seal=read_json(r/'CONDITION_RECEIPT.json')
reg=read_json(r/'inference_registry.json');e=read_json(r/'EXECUTION.json')
assert file_hash(r/'METHOD_FREEZE.json')==seal['freeze_file_sha256']
assert file_hash(r/'inference_registry.json')==seal['inference_registry_sha256']
assert json_hash(read_json(r/'PROTOCOL.json'))==seal['protocol_sha256']
assert json_hash(reg['conditions'])==seal['condition_record_sha256']
assert all(file_hash(REPO/p)==h for p,h in freeze['historical_source_sha256'].items())
assert all(file_hash(REPO/p)==h for p,h in freeze['input_file_sha256'].items())
assert all(file_hash(r/'source_snapshot'/p)==h for p,h in freeze['source_sha256'].items())
fix=read_json(r/'POST_RUN_CODE_FIX.json')['changes']
assert all(file_hash(REPO/p)==(fix[p]['delivery_sha256'] if p in fix else h) for p,h in freeze['source_sha256'].items())
assert file_hash(REPO/CHECKPOINT)==CHECKPOINT_SHA
for asset in freeze['reused_assets'].values():
    verify_tensor(asset)
ts=check_condition(reg['conditions']);added=verify_tensor(seal['added_mask'])
old=read_json(OLD/'inference_registry.json')['conditions'][CASE]
assert torch.equal(ts['mask'],verify_tensor(old['B_512']['mask']))
assert torch.equal(ts['mask'],verify_tensor(old['C_512']['mask']))
assert torch.equal(added,ts['mask']&~ts['original_mask'])
assert not (added&well_mask(read_json(OLD/'layout.json')['holdout'])).any()
flow=r/'predictions/seed42'
receipt=read_json(flow/'receipt.json')
assert read_json(flow/'completion_receipt.json')['receipt_sha256']==file_hash(flow/'receipt.json')
pred=verify_tensor(receipt['prediction'])
assert not ((pred!=ts['values'])&ts['mask']).any()
trace=read_json(flow/'trace.json')
assert len(trace)==receipt['velocity_evaluations']==32
assert all(t['time']==(i+.5)/32 and t['velocity_finite'] and t['state_finite'] and t['condition_embedding_max_error']==0 for i,t in enumerate(trace))
assert receipt['trace_sha256']==file_hash(flow/'trace.json')
noise=verify_tensor(reg['noise']['42']['asset'])
assert tensor_hash(noise)==receipt['initial_noise_tensor_sha256']
assert receipt['base_hash_before']==receipt['base_hash_after']==reg['noise']['42']['expected_model_hash']
assert read_json(flow/'failed_save_receipt.json')['velocity_evaluations']==32
v1=r.parent/'run_20261017_v1'
assert read_json(v1/'predictions/seed42/receipt.json')['velocity_evaluations']==0
v1reg=read_json(v1/'inference_registry.json')
assert all(v1reg['conditions'][k]['tensor_sha256']==a['tensor_sha256'] for k,a in reg['conditions'].items())
assert datetime.fromisoformat(seal['sealed_at_utc'])<datetime.fromisoformat(receipt['started_at_utc'])<datetime.fromisoformat(receipt['completed_at_utc'])<datetime.fromisoformat(e['evaluation_started_at_utc'])
inputs=read_json(r/'EVALUATION_INPUTS.json')
held=verify_tensor(inputs['domains']['E_holdout']['tensor'])
samples={s['arm']:s for s in e['samples']}
a=verify_tensor(samples['A_9']['prediction']);copy=verify_tensor(samples['COPY_ONLY_STAGE23']['prediction'])
assert torch.equal(copy[held],a[held])
for s in samples.values():
    verify_tensor(s['prediction'])
    for m in s['metrics'].values():
        assert m['target_tp']+m['target_fp']==m['target_pred_volume']
        assert m['target_tp']+m['target_fn']==m['target_truth_volume']
        assert m['target_iou']==m['target_tp']/(m['target_tp']+m['target_fp']+m['target_fn'])
assert len({s['voxels'] for s in e['object1_common_subset']})==1
assert e['object1_common_subset'][0]['voxels']==510
assert len(list(r.glob('predictions/*/decoded_geology.pt')))==1
assert not list(v1.glob('predictions/*/decoded_geology.pt'))
assert e['new_Flow_outputs']==1 and e['velocity_evaluations']==32
shutil.copyfile(Path(__file__),r/'FINAL_CHECK_SCRIPT.py')
checks=dict(status='passed',checked_at_utc=utc(),historical_sources_unchanged=len(freeze['historical_source_sha256']),
    frozen_inputs_unchanged=True,executed_source_snapshot_verified=True,
    current_source_diff_explicitly_recorded=sorted(fix),same_v1_v2_condition_tensors=True,
    condition_mask_equal_B_C=True,holdout_additions=0,original_and_added_violations=0,
    classification_assets_verified=True,checkpoint_hash_verified=True,actual_noise_verified=True,
    model_after_identity_evidence=receipt['base_hash_after_evidence'],midpoint_trace_verified=32,
    copy_holdout_equal_A=True,metric_arithmetic=True,common_object1_denominator=510,
    source_cases=1,source_seeds=[42],new_Flow_outputs=1,total_velocity_evaluations=32,
    initial_dependency_failure_calls=0,saved_output_recovery_calls=0,new_training=0,new_deconvolution=0,
    final_CPU_tests=5)
write_json(r/'FINAL_CHECKS.json',checks)
write_json(r/'DELIVERY.json',dict(status='completed',run=str(r.relative_to(REPO)),new_Flow_outputs=1,
    velocity_evaluations=32,stop_reason=e['stop_reason'],engineering_issues_disclosed=True,
    executed_source='source_snapshot',delivery_patch='POST_RUN_CODE_FIX.diff',
    report='REPORT.md',comparison='comparison.csv',figure='figures/fixed_object_slice.png',
    final_checks_sha256=file_hash(r/'FINAL_CHECKS.json'),handoff_sha256=file_hash(PROJECT/'docs/DEVELOPMENT_HANDOFF.md'),
    initial_failed_run_receipt_sha256=file_hash(v1/'predictions/seed42/receipt.json')))
write_json(r/'ARTIFACT_HASHES.json',{str(p.relative_to(r)):file_hash(p) for p in sorted(r.rglob('*')) if p.is_file() and p.name!='ARTIFACT_HASHES.json'})
print(json.dumps(checks,ensure_ascii=False))
