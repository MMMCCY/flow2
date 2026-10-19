"""One A-only model-based log-impedance inversion; single-quantity full pseudo-wells."""
import sys,os,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage25_physics.common import *
from guidance.seismic_inversion import labels_to_clean_prior_acoustic,invert_acoustic_member,parse_inversion_config
from scripts.stage23.evaluate import sealed_prediction
import torch

def main():
    inversion_guard();torch.set_num_threads(4)
    f=read_json(PHYS/'OBSERVATION_FREEZE.json');assert f['source_sha256']==physics_sources()
    assert all(file_hash(PHYS/n)==h for n,h in f['config_sha256'].items())
    if (PHYS/'INVERSION_STATE.json').exists():raise FileExistsError('Only one inversion allowed')
    table,operator=setup();obs=verify_tensor(f['assets']['observation']);subsurface=verify_tensor(f['assets']['subsurface'])
    _,prior_labels=sealed_prediction(ROOT/'arms/A/predictions/seed42')
    original=check_condition(read_json(ROOT/'arms/A/inference_registry.json')['conditions'])
    prior,cleanup=labels_to_clean_prior_acoustic(prior_labels,table,subsurface)
    condition=hard_labels_to_acoustic(original['values'],table)
    config=parse_inversion_config(read_json(PHYS/'inversion_config.json'))
    start=time.monotonic();write_json(PHYS/'INVERSION_STATE.json',dict(status='running',inversion_calls=1,started_at_utc=utc(),truth_access=False,prior_source='saved A only'))
    prior_exact,updated,fields,diagnostics=invert_acoustic_member(prior,observed_seismic=obs,subsurface_mask=subsurface,condition_target=condition,condition_mask=original['mask'],property_table=table,forward_operator=operator,config=config)
    assert torch.isfinite(updated).all() and torch.equal(updated[:,1:2],prior_exact[:,1:2])
    assert torch.equal(updated.masked_select(original['mask'].expand_as(updated)),condition.masked_select(original['mask'].expand_as(condition)))
    candidates=table[0,1:].log()[None,:,None,None,None]
    # The only updated quantity is log impedance. Prior slowness is never a second evidence channel.
    labels=(updated[:,0:1].log()-candidates).abs().argmin(1,keepdim=True).long()
    layout=read_json(ROOT/'layout.json');dv,dm=dense_conditions(original['values'],original['mask'],labels,layout['levels']['512'])
    expected=original['mask']|well_mask(layout['levels']['512'])
    assert torch.equal(dm,expected) and not ((dm&~original['mask'])&well_mask(layout['holdout'])).any()
    assets={k:save_tensor(PHYS/'preprocessing'/f'{k}.pt',v) for k,v in dict(prior_acoustic=prior_exact,updated_acoustic=updated,prior_seismic=fields[0],updated_seismic=fields[1],classification=labels).items()}
    norm=obs.square().sum().sqrt()
    errors={k:float((value-obs).square().sum().sqrt()/norm) for k,value in dict(prior=fields[0],inverted=fields[1]).items()}
    freeze=read_json(ROOT/'OBSERVATION_FREEZE.json')
    seal_arm('D',dv,dm,original['values'],original['mask'],freeze['noise'],sources(),read_json(ROOT/'PROTOCOL.json'))
    write_json(PHYS/'INVERSION_STATE.json',dict(status='completed',inversion_calls=1,actual_member_count=1,posterior_statistics_computed=False,truth_access=False,prior_source='saved A and original conditions only',slowness_updated=False,time_depth='A prior slowness, never truth velocity',classification='single updated log-impedance nearest raw0..13; first argmin',converged='direct Cholesky solve, no iterative/tuning gate',assets=assets,relative_seismic_errors=errors,cleanup=cleanup,diagnostics=diagnostics,elapsed_seconds=time.monotonic()-start,completed_at_utc=utc(),source_sha256=physics_sources()))
    print('SINGLE_INVERSION_COMPLETE_D_CONDITIONS_SEALED',errors,flush=True)
if __name__=='__main__':main()
