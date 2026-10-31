"""One frozen model load, six prescribed outputs, durable NFE budget and traces."""
import argparse
import os
import sys
import time
import platform
import traceback
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
sys.dont_write_bytecode=True
PROJECT=Path(__file__).resolve().parents[2]
for p in (PROJECT,PROJECT.parents[1],PROJECT.parents[1]/'src'):
    sys.path.insert(0,str(p))
import torch
from .common import *
from .sampler import EndpointObjective,sample
from inference_runtime import load_model_with_weight_policy
from guidance.dflow import model_state_dict_hashes

def main(root,prox_device):
    root=root.resolve(); reg=read_json(root/'inference_registry.json'); validate_registry(reg)
    assets=[*reg['conditions'].values(),reg['subsurface'],reg['noise'],*reg['observations'].values()]
    allowed=[REPO/a['path'] for a in assets]+[REPO/reg['checkpoint']]
    install_firewall(root,allowed)
    if sources()!=reg['source_sha256']:
        raise ValueError('Frozen source differs; do not run an unrecorded revision')
    if (root/'SAMPLING_COMPLETE.json').exists():
        raise FileExistsError('Sampling is complete; run postprocessing only')
    if (root/'sampling_state.json').exists():
        raise RuntimeError('Existing NFE ledger: inspect saved progress; no automatic resampling')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():
        raise RuntimeError('Network requires CUDA')
    values=verify_tensor(reg['conditions']['values']);mask=verify_tensor(reg['conditions']['mask']).bool()
    sub=verify_tensor(reg['subsurface']).bool(); initial=verify_tensor(reg['noise'])
    observations={k:verify_tensor(v) for k,v in reg['observations'].items()}
    if file_hash(REPO/reg['checkpoint'])!=CHECKPOINT_SHA:
        raise ValueError('Checkpoint hash changed')
    from model_train_sh_inference_cond import Geo3DStochInterp
    model,load_report=load_model_with_weight_policy(Geo3DStochInterp,REPO/reg['checkpoint'],
                                                  map_location='cpu',weight_source='ema')
    if load_report['ema_entry_count']!=411 or tuple(model.embedding.weight.shape)!=(15,15):
        raise ValueError('Checkpoint EMA/embedding mismatch')
    from flowtrain.interpolation import LinearInterpolant
    if not isinstance(model.interpolant,LinearInterpolant) or not model.interpolant.one_sided:
        raise ValueError('Version conflict: training interpolant is not one-sided linear')
    if list(model.time_range)!=[.0001,.9999]:
        raise ValueError('Checkpoint time range differs from reviewed metadata')
    for parameter in model.parameters():
        parameter.requires_grad_(False);parameter.grad=None
    model.eval().float().to('cuda')
    before,_=model_state_dict_hashes(model)
    if before!=EXPECTED_MODEL:
        raise ValueError('Frozen model identity mismatch')
    load_report.update(actual_time_range=list(model.time_range),embedding_dim=15,
        initial_nominal_time_endpoint_approximation='j0 nominal0 evaluated at1e-4',
        training_endpoint_jitter='training source adds1e-4 Gaussian to embedded X1; linear path retained',
        training_interpolant='LinearInterpolant(one_sided=True); independent Gaussian X0')
    write_json(root/'model_load_report.json',load_report)
    table,operator=physics(reg['physics_config']);pd=torch.device(prox_device)
    embedding=model.embedding.weight.detach().to(pd,dtype=torch.float64)
    embedding_asset=save_tensor(root/'constants/embedding_fp64.pt',embedding)
    write_json(root/'runtime_constants.json',dict(embedding=embedding_asset,
        role='immutable checkpoint embedding; no new trainable parameters'))
    table=table.to(pd);pv=values.to(pd);pm=mask.to(pd);ps=sub.to(pd)
    state=dict(status='running',started_at_utc=utc(),git=git_snapshot(),network_evaluations=0,
        completed_arms=[],skipped_arms=[],active_arm=None,active_step=None,
        physics_forwards=0,physics_backwards=0,prox_device=str(pd),model_hash_before=before,
        registry_sha256=file_hash(root/'inference_registry.json'))
    write_json(root/'sampling_state.json',state)
    resources=dict(host=platform.node(),python=platform.python_version(),torch=torch.__version__,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),network_device='cuda:0',
        prox_device=str(pd),cpu_threads=torch.get_num_threads(),model_loads=1,
        network_precision='FP32_noTF32_noautocast',prox_precision='FP64',
        deterministic_algorithms=True,prng='torch CPU FP32',scientific_retries=0)
    all_started=time.monotonic();arm_receipts={};counters=dict(physics_forwards=0,physics_backwards=0)
    try:
        for name,a,r,obs_key in ARMS:
            if obs_key and obs_key not in observations:
                state['skipped_arms'].append(dict(arm=name,reason='prescribed G2 failed same-condition input definition'))
                continue
            state['active_arm']=name;folder=root/'arms'/name;folder.mkdir(parents=True)
            started=time.monotonic()
            objective=EndpointObjective(embedding,table,operator,pv,pm,ps,
                observations[obs_key].to(pd) if obs_key else None,a,r,counters)
            arm_counts=counters.copy(); trace=[]
            def reserve(j):
                if state['network_evaluations']>=192:
                    raise RuntimeError('Network budget exhausted')
                state.update(network_evaluations=state['network_evaluations']+1,active_step=j)
                write_json(root/'sampling_state.json',state)
            def checkpoint_step(record,x,u):
                trace.append(record)
                # Save only latest numerical progress, not a 0.5GB noise stack.
                tmp=folder/'progress.partial'
                torch.save(dict(next_step=record['step']+1,state=x.detach().cpu(),
                                endpoint=u.detach().cpu()),tmp)
                tmp.replace(folder/'progress.pt')
                write_json(folder/'trace.json',trace)
                state.update(physics_forwards=counters['physics_forwards'],physics_backwards=counters['physics_backwards'])
                write_json(root/'sampling_state.json',state)
            pred,endpoint,prox_endpoint,trace=sample(model,initial,values,mask,sub,objective,checkpoint_step,reserve)
            # First post-sampling work: atomically save raw numerical tensors.
            assets_out={k:save_tensor(folder/f'{k}.pt',v) for k,v in
                [('hard_prediction',pred),('continuous_endpoint',endpoint),('prox_endpoint_fp64',prox_endpoint)]}
            receipt=dict(arm=name,a=a,r=r,observation=obs_key,assets=assets_out,
                network_evaluations=len(trace),initial_noise_tensor_sha256=tensor_hash(initial),
                physics_forwards=counters['physics_forwards']-arm_counts['physics_forwards'],
                physics_backwards=counters['physics_backwards']-arm_counts['physics_backwards'],
                condition_violations=int(((pred!=values)&mask).sum()),
                noise_hashes=[t['noise_tensor_sha256'] for t in trace],
                accepted_steps=sum(i['accepted_alpha'] is not None for t in trace for i in t['inner']),
                line_search_exhausted=sum(i['line_search_exhausted'] for t in trace for i in t['inner']),
                zero_gradient_steps=sum(i['zero_gradient'] for t in trace for i in t['inner']),
                elapsed_seconds=time.monotonic()-started,first_two_outer_seconds=[t['elapsed_seconds'] for t in trace[:2]])
            write_json(folder/'receipt.json',receipt);arm_receipts[name]=receipt
            state['completed_arms'].append(name)
            write_json(root/'sampling_state.json',state)
            print('ARM_COMPLETE',name,'nfe',state['network_evaluations'],'seconds',receipt['elapsed_seconds'],flush=True)
        after,_=model_state_dict_hashes(model)
        if after!=before or any(p.grad is not None or p.requires_grad for p in model.parameters()):
            raise ValueError('Frozen model/gradient invariant failed')
        assert counters['physics_forwards']<=2688 and counters['physics_backwards']<=384
        state.update(status='completed' if len(arm_receipts)==6 else 'G1_complete_pair_invalid',
            model_hash_after=after,parameter_gradients_absent=True,active_arm=None,
            finished_at_utc=utc(),elapsed_seconds=time.monotonic()-all_started)
        for name,receipt in arm_receipts.items():
            assert receipt['network_evaluations']==32
            assert receipt['noise_hashes']==arm_receipts['F0']['noise_hashes']
            for asset in receipt['assets'].values():
                verify_tensor(asset)
        write_json(root/'SAMPLING_COMPLETE.json',dict(state=state,arms=arm_receipts,
            paired_noise_hashes_equal=True,model_unchanged=True,training_calls=0,traditional_inversions=0))
    except Exception:
        state.update(status='engineering_failed',error=traceback.format_exc(),finished_at_utc=utc())
        raise
    finally:
        state.update(physics_forwards=counters['physics_forwards'],physics_backwards=counters['physics_backwards'])
        resources.update(network_evaluations=state['network_evaluations'],
            physics_forwards_internal=counters['physics_forwards'],physics_backwards=counters['physics_backwards'],
            elapsed_seconds=time.monotonic()-all_started,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(root/'sampling_state.json',state);write_json(root/'resource_usage.json',resources)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prox-device',choices=['cuda','cpu'],default='cuda')
    args=p.parse_args();main(args.out,args.prox_device)
