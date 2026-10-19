"""ORACLE_DIAGNOSTIC_ONLY: intervene on fixed slowness, NEVER truth impedance."""
import os,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage26_diagnostics.common import *
from guidance.seismic_inversion import invert_acoustic_member,parse_inversion_config
import torch

def main():
    guard('oracle');torch.set_num_threads(4)
    freeze=read_json(ROOT/'INPUT_FREEZE.json');assert freeze['source_sha256']==sources()
    path=ROOT/'oracle/ORACLE_DIAGNOSTIC_ONLY_bundle.pt'
    if (ROOT/'ORACLE_STATE.json').exists() and read_json(ROOT/'ORACLE_STATE.json')['status']=='completed':raise FileExistsError('Completed oracle is immutable')
    table,operator=setup();original=check_condition(freeze['original_conditions']);sub=verify_tensor(freeze['subsurface']);obs=verify_tensor(freeze['observation'])
    old_prior=verify_tensor(freeze['old_assets']['prior_exact'])
    freeze_sha=file_hash(ROOT/'INPUT_FREEZE.json')
    # Only this explicitly authorized ORACLE entry derives extra slowness from truth labels.
    # Do NOT compute truth impedance or pass truth labels to inversion/classification.
    truth_record=read_json(STAGE25/'evaluation_registry.json')['truth'];truth=verify_tensor(truth_record)
    truth_slowness=table[1,truth.long()+1]
    oracle_input=old_prior.clone();oracle_input[:,1:2]=truth_slowness
    condition=hard_labels_to_acoustic(original['values'],table)
    assert torch.equal(oracle_input[:,0:1],old_prior[:,0:1])
    assert torch.equal(oracle_input.masked_select(original['mask'].expand_as(oracle_input)),condition.masked_select(original['mask'].expand_as(condition)))
    protocol=dict(role='ORACLE_DIAGNOSTIC_ONLY',initial_impedance='identical voxelwise to saved prior_exact',changed_module='truth slowness affects BOTH prior seismic TWT and time-to-depth correction',truth_impedance_used=False,truth_labels_given_to_classifier=False,config_sha256=freeze['config_sha256'],truth_asset=truth_record,no_formal_registry=True,no_Flow=True)
    if path.exists():
        bundle=torch.load(path,map_location='cpu',weights_only=False)
        print('Recovered oracle numerical bundle; zero additional solve',flush=True)
    else:
        if (ROOT/'ORACLE_STATE.json').exists():raise RuntimeError('A previous oracle call lacks numerical output; diagnose before any retry')
        write_json(ROOT/'ORACLE_PROTOCOL.json',protocol)
        write_json(ROOT/'ORACLE_STATE.json',dict(status='running',role='ORACLE_DIAGNOSTIC_ONLY',inversion_calls=1,started_at_utc=utc()))
        started=time.monotonic()
        result=invert_acoustic_member(oracle_input,observed_seismic=obs,subsurface_mask=sub,condition_target=condition,condition_mask=original['mask'],property_table=table,forward_operator=operator,config=parse_inversion_config(read_json(ROOT/'configs/inversion_config.json')))
        # Durable numerical output before all assertions, hashing and reporting.
        bundle=dict(prior=result[0],updated=result[1],fields=result[2],diagnostics=result[3],elapsed_seconds=time.monotonic()-started,role='ORACLE_DIAGNOSTIC_ONLY',inversion_calls=1,source_sha256=freeze['source_sha256'],input_freeze_sha256=freeze_sha)
        path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.partial');torch.save(bundle,temp);temp.replace(path)
        print('ORACLE_NUMERICAL_BUNDLE_SAVED',flush=True)
    assert bundle['source_sha256']==freeze['source_sha256'] and bundle['input_freeze_sha256']==file_hash(ROOT/'INPUT_FREEZE.json')
    assert torch.equal(bundle['prior'][:,0:1],old_prior[:,0:1])
    assert torch.equal(bundle['prior'][:,1:2],truth_slowness) and torch.equal(bundle['updated'][:,1:2],truth_slowness)
    for value in (bundle['prior'],bundle['updated']):
        assert torch.isfinite(value).all()
        assert torch.equal(value.masked_select(original['mask'].expand_as(value)),condition.masked_select(original['mask'].expand_as(condition)))
    assets={k:save_tensor(ROOT/'oracle'/f'ORACLE_DIAGNOSTIC_ONLY_{k}.pt',v) for k,v in dict(prior=bundle['prior'],updated=bundle['updated'],prior_seismic=bundle['fields'][0],updated_seismic=bundle['fields'][1]).items()}
    write_json(ROOT/'ORACLE_STATE.json',dict(status='completed',role='ORACLE_DIAGNOSTIC_ONLY',inversion_calls=1,new_Flow=0,new_checkpoint_loads=0,new_observation_generations=0,initial_impedance_exactly_old_prior=True,only_slowness_module_changed=True,oracle_slowness_unchanged=True,original_conditions_exact=True,truth_impedance_used=False,assets=assets,numerical_bundle_sha256=file_hash(path),elapsed_seconds=bundle['elapsed_seconds'],diagnostics=bundle['diagnostics'],completed_at_utc=utc()))
    print('ORACLE_DIAGNOSTIC_ONLY_COMPLETED',bundle['elapsed_seconds'],flush=True)
if __name__=='__main__':main()
