"""Reuse immutable seismic data; one recoverably saved inversion, then D0/D conditions."""
import os,sys,time,shutil
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage26.common import *
from scripts.stage23.evaluate import sealed_prediction
from guidance.seismic_inversion import labels_to_clean_prior_acoustic,parse_inversion_config,invert_acoustic_member
import torch

def numerical_bundle(prior,obs,subsurface,condition,mask,table,operator,config):
    path=ROOT/'preprocessing/numerical_bundle.pt'
    if path.exists():
        value=torch.load(path,map_location='cpu',weights_only=False)
        print('Recovered saved numerical bundle; no new inversion',flush=True)
        return value
    state=ROOT/'INVERSION_STATE.json'
    if state.exists():raise RuntimeError('Prior call has no saved bundle: inspect engineering failure before retry')
    started=time.monotonic();write_json(state,dict(status='running',inversion_calls=1,started_at_utc=utc()))
    result=invert_acoustic_member(prior,observed_seismic=obs,subsurface_mask=subsurface,condition_target=condition,condition_mask=mask,property_table=table,forward_operator=operator,config=config)
    # First operation after the numerical return: atomic raw tensor save. No hash/report work first.
    bundle=dict(prior_exact=result[0],updated=result[1],fields=result[2],diagnostics=result[3],inversion_elapsed_seconds=time.monotonic()-started,inversion_calls=1)
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.partial')
    torch.save(bundle,temporary);temporary.replace(path)
    print('NUMERICAL_BUNDLE_SAVED',flush=True)
    return bundle

def main():
    guard('prepare');torch.set_num_threads(4)
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'CONDITIONS_FROZEN.json').exists():raise FileExistsError('Complete conditions are immutable; do not recompute')
    protocol=dict(task='Stage26 synthetic tracewise poststack/model-based linearized log impedance update',seed=42,shape=[64]*3,spacing_m=[SPACING]*3,old_stage25_commit='e97a92677463f31162f5bec4e9a7bc78a5e0ac47',
        max_successful_inversions=1,max_new_flow_outputs=2,max_velocity_evaluations=64,steps=32,source='old saved CPU seed42 15-dimensional tensor',weight_policy='411EMA, checkpoint embedding unchanged',precision='FP32_no_TF32_no_autocast',
        arms=dict(D0='prior_exact log impedance classification + old512 wells',D='updated log impedance classification + same512 wells'),classifier='all raw0..13 nearest log impedance; first argmin; observed air and original conditions override',
        time_depth='fixed A prior slowness; no velocity inversion',one_member=True,posterior_statistics=False,parameter_sweeps=False,truth_gate=False,specification_sha256=file_hash(REPO/'STAGE26_PHYSICAL_SEISMIC_TASK_2026-10-19.md'))
    if not (ROOT/'INPUT_FREEZE.json').exists():
        write_json(ROOT/'START.json',dict(started_at_utc=utc(),budget_seconds=10800,git=git_snapshot(),reviewed_old_failure=True))
        f=read_json(OLD_PHYS/'OBSERVATION_FREEZE.json')
        for name,h in f['config_sha256'].items():
            assert file_hash(OLD_PHYS/name)==h
            target=ROOT/'configs'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OLD_PHYS/name,target)
        for name in ('guidance/seismic.py','guidance/seismic_inversion.py'):
            rel=str((PROJECT/name).relative_to(REPO));assert file_hash(PROJECT/name)==f['source_sha256'][rel]
        obs=verify_tensor(f['assets']['observation']);subsurface=verify_tensor(f['assets']['subsurface'])
        assert obs.shape==(1,1,64,64,400) and obs.dtype==torch.float64 and torch.isfinite(obs).all()
        cfg=read_json(ROOT/'configs/observation_config.json')
        assert cfg['time_sampling']==dict(num_samples=400,sample_interval_ms=8.)
        assert cfg['cell_size_m']==[SPACING]*3 and cfg['wavelet']['peak_frequency_hz']==25 and cfg['wavelet']['duration_ms']==128 and cfg['noise']['type']=='none'
        areg=read_json(OLD/'arms/A/inference_registry.json');a=check_condition(areg['conditions']);ar,ap=sealed_prediction(OLD/'arms/A/predictions/seed42')
        noise=read_json(OLD/'OBSERVATION_FREEZE.json')['noise'];assert verify_tensor(noise).shape==(1,15,64,64,64)
        assert ar['base_hash_after']==EXPECTED_MODEL and ar['initial_noise_tensor_sha256']==noise['tensor_sha256']
        assert torch.equal(subsurface,~(a['mask']&(a['values']==-1)))
        layout=read_json(OLD/'layout.json');write_json(ROOT/'layout.json',layout);write_json(ROOT/'PROTOCOL.json',protocol)
        source=sources()
        for rel in source:
            dest=ROOT/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/rel,dest)
        write_json(ROOT/'INPUT_FREEZE.json',dict(frozen_at_utc=utc(),source_sha256=source,protocol_sha256=json_hash(protocol),old_observation_freeze_sha256=file_hash(OLD_PHYS/'OBSERVATION_FREEZE.json'),old_observation_producer_source=f['source_sha256'],
            observation=f['assets']['observation'],subsurface=f['assets']['subsurface'],observation_reused_without_recompute=True,noise=noise,A_prediction=ar['prediction'],original_conditions=areg['conditions'],layout_sha256=file_hash(ROOT/'layout.json'),config_sha256=f['config_sha256'],git=git_snapshot(),historical_failed_inversion_attempts=1,new_budget=True))
    freeze=read_json(ROOT/'INPUT_FREEZE.json');assert sources()==freeze['source_sha256'] and json_hash(protocol)==freeze['protocol_sha256']
    assert all(file_hash(ROOT/'configs'/n)==h for n,h in freeze['config_sha256'].items())
    table,operator=setup();obs=verify_tensor(freeze['observation']);subsurface=verify_tensor(freeze['subsurface'])
    a=check_condition(freeze['original_conditions']);ap=verify_tensor(freeze['A_prediction'])
    prior,cleanup=labels_to_clean_prior_acoustic(ap,table,subsurface)
    condition=hard_labels_to_acoustic(a['values'],table)
    config=parse_inversion_config(read_json(ROOT/'configs/inversion_config.json'))
    bundle=numerical_bundle(prior,obs,subsurface,condition,a['mask'],table,operator,config)
    prior_exact,updated,fields=(bundle[k] for k in ('prior_exact','updated','fields'))
    assert torch.isfinite(updated).all() and torch.equal(prior_exact[:,1:2],updated[:,1:2])
    for acoustic in (prior_exact,updated):assert torch.equal(acoustic.masked_select(a['mask'].expand_as(acoustic)),condition.masked_select(a['mask'].expand_as(condition)))
    classifications={arm:classify_impedance(v,table,subsurface,a['values'],a['mask']) for arm,v in dict(D0=prior_exact,D=updated).items()}
    layout=read_json(ROOT/'layout.json');records={};masks={}
    for arm,labels in classifications.items():
        values,mask=dense_conditions(a['values'],a['mask'],labels,layout['levels']['512']);masks[arm]=mask
        assert not ((mask&~a['mask'])&well_mask(layout['holdout'])).any()
        records[arm]=seal_arm(arm,values,mask,a,freeze['noise'],sources(),protocol)
    assert torch.equal(masks['D0'],masks['D'])
    assets={k:save_tensor(ROOT/'preprocessing'/f'{k}.pt',v) for k,v in dict(prior_exact=prior_exact,updated=updated,prior_seismic=fields[0],updated_seismic=fields[1],classification_D0=classifications['D0'],classification_D=classifications['D']).items()}
    same=torch.equal(check_condition(records['D0'])['values'],check_condition(records['D'])['values'])
    added=masks['D']&~a['mask'];changed=classifications['D']!=classifications['D0']
    errors=dict(A_prior=relative_l2(fields[0],obs),continuous_updated=relative_l2(fields[1],obs))
    write_json(ROOT/'INVERSION_STATE.json',dict(status='completed',inversion_calls=1,successful_saved_inversions=1,old_stage25_failed_attempts=1,assets=assets,numerical_bundle=dict(path=str((ROOT/'preprocessing/numerical_bundle.pt').relative_to(REPO)),file_sha256=file_hash(ROOT/'preprocessing/numerical_bundle.pt')),diagnostics=bundle['diagnostics'],cleanup=cleanup,
        elapsed_seconds=bundle['inversion_elapsed_seconds'],relative_l2=errors,slowness_unchanged=True,truth_access=False,actual_members=1,posterior_statistics=False,completed_at_utc=utc()))
    write_json(ROOT/'CONDITIONS_FROZEN.json',dict(sealed_at_utc=utc(),conditions=records,assets=assets,D0_D_identical=same,classification_changed_voxels=int(changed.sum()),added_labels_changed=int((changed&added).sum()),mask_equal=True,holdout_added_voxels=0,canonical_arms=['D0'] if same else ['D0','D'],alias={'D':'D0'} if same else {},source_sha256=sources()))
    print('INVERSION_AND_D0_D_CONDITIONS_COMPLETE',errors,'changed_added',int((changed&added).sum()),'same_conditions',same,flush=True)
if __name__=='__main__':main()
