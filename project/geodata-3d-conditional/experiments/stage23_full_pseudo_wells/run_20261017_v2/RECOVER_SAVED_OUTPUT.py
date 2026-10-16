"""CPU-only recovery of a successfully sampled tensor after receipt path serialization failed."""
import os,sys,shutil
from pathlib import Path
from datetime import datetime,timezone
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
repo=Path('/home/xmj/mcy/flowtrain_stochastic_interpolation-main')
sys.path.insert(0,str(repo/'project/geodata-3d-conditional'))
from scripts.stage23.common import *
import torch
guard('flow')
root=ROOT
folder=root/'predictions/seed42'
assert not (folder/'completion_receipt.json').exists()
state=read_json(folder/'receipt.json')
assert state['status']=='failed' and state['velocity_evaluations']==32
assert 'relative_to' in (root/'flow_seed42.log').read_text()
assert 'prediction=save_tensor' in (root/'flow_seed42.log').read_text()
assert source_hashes()==state['source_sha256']
registry=read_json(root/'inference_registry.json')
ts=check_condition(registry['conditions'])
noise=verify_tensor(state['source_noise_asset'])
assert tensor_hash(noise)==state['initial_noise_tensor_sha256']
predpath=folder/'decoded_geology.pt'
pred=torch.load(predpath,map_location='cpu',weights_only=True)
assert pred.shape==(1,1,64,64,64) and pred.dtype==torch.int64
assert pred.min()>=-1 and pred.max()<=13
extra=ts['mask']&~ts['original_mask']
ov=int(((pred!=ts['values'])&ts['original_mask']).sum())
av=int(((pred!=ts['values'])&extra).sum())
assert ov==av==0
trace=read_json(folder/'trace.json')
assert len(trace)==32
for i,t in enumerate(trace):
    assert t['step']==i and t['time']==(i+.5)/32
    assert t['condition_embedding_max_error']==0 and t['state_finite'] and t['velocity_finite']
shutil.copyfile(folder/'receipt.json',folder/'failed_save_receipt.json')
shutil.copyfile(Path(__file__),root/'RECOVER_SAVED_OUTPUT.py')
finished=datetime.fromtimestamp((folder/'trace.json').stat().st_mtime,timezone.utc)
start=datetime.fromisoformat(state['started_at_utc'])
state.update(status='completed',prediction=dict(path=str(predpath.relative_to(REPO)),
    file_sha256=file_hash(predpath),tensor_sha256=tensor_hash(pred)),
    trace_sha256=file_hash(folder/'trace.json'),original_condition_violations=ov,
    added_condition_violations=av,base_hash_after=state['base_hash_before'],
    completed_at_utc=utc(),elapsed_seconds=(finished-start).total_seconds(),
    elapsed_seconds_source='trace file mtime minus persisted start UTC; approximate, original monotonic timing lost',
    recovered_without_forward=True,save_failure_receipt_sha256=file_hash(folder/'failed_save_receipt.json'),
    base_hash_after_evidence='Recovered from successful runtime before==after and entry-by-entry equality check before save_tensor failure; exact frozen source and traceback retained, not a fresh model measurement',
    recovery_reason=state.pop('error'),sampling_finished_at_utc_estimate=finished.isoformat())
write_json(folder/'receipt.json',state)
write_json(folder/'completion_receipt.json',dict(receipt_sha256=file_hash(folder/'receipt.json'),
    recovered_saved_output=True,recovery_script_sha256=file_hash(root/'RECOVER_SAVED_OUTPUT.py'),additional_velocity_calls=0))
print('Recovered saved prediction without any new forward; 32 midpoint traces and all hard conditions verified')
