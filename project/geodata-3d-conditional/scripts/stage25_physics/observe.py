"""Independent observation-only process: native truth -> one ideal seismic data cube."""
import sys,os,math,time,shutil
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage25_physics.common import *
from guidance.seismic import build_seismic_observation
import torch

def main():
    guard('generate');torch.set_num_threads(4)
    assert read_json(ROOT/'MANDATORY_COMPLETE.json')['status']=='MANDATORY_COMPLETE'
    if PHYS.exists():raise FileExistsError(PHYS)
    PHYS.mkdir();start=time.monotonic()
    tables,_=acoustic_tables_from_config(read_json(ACOUSTIC),15)
    config=read_json(OBS_CONFIG);config['id']='stage25_single_member_noiseless_actual_spacing'
    config['cell_size_m']=[SPACING]*3
    # Bound uses complete known grid depth and minimum ROCK codebook speed, no truth velocity.
    maximum_twt=2*64*SPACING/float(tables.velocity_m_s[1:].min())*1000
    config['time_sampling']['num_samples']=math.ceil((maximum_twt+config['wavelet']['duration_ms']/2)/config['time_sampling']['sample_interval_ms'])+1
    write_json(PHYS/'observation_config.json',config);write_json(PHYS/'acoustic_codebook.json',read_json(ACOUSTIC))
    inv=read_json(INV_CONFIG);inv['id']='stage25_single_member_fixed_operating_point';write_json(PHYS/'inversion_config.json',inv)
    original=check_condition(read_json(ROOT/'arms/A/inference_registry.json')['conditions'])
    subsurface=~(original['mask']&(original['values']==-1))
    table,operator=setup()
    truth=verify_tensor(read_json(ROOT/'evaluation_registry.json')['truth'])
    acoustic=hard_labels_to_acoustic(truth,table)
    obs=build_seismic_observation(acoustic,subsurface,operator,noise_std_amplitude=0.,noise_seed=4205)
    assets=dict(observation=save_tensor(PHYS/'observations/seismic.pt',obs.values),subsurface=save_tensor(PHYS/'conditions/subsurface.pt',subsurface))
    source=physics_sources()
    for rel in source:
        dest=PHYS/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/rel,dest)
    write_json(PHYS/'OBSERVATION_FREEZE.json',dict(sealed_at_utc=utc(),source_sha256=source,assets=assets,config_sha256={n:file_hash(PHYS/n) for n in ('observation_config.json','acoustic_codebook.json','inversion_config.json')},metadata=obs.metadata,maximum_twt_bound_ms=maximum_twt,time_bound_uses_truth_velocity=False,actual_member_count=1,posterior_statistics_computed=False,legacy_fixed12_parser_field='retained only for old parser compatibility, NOT a member-count claim; no posterior statistics function called',elapsed_seconds=time.monotonic()-start))
    print('SEISMIC_OBSERVATION_FROZEN',operator.num_time_samples,operator.recording_end_ms,flush=True)
if __name__=='__main__':main()
