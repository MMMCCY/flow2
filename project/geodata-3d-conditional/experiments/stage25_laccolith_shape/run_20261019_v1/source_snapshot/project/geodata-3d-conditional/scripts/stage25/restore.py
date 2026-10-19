"""One truth-blind Stage22 solve; full Stage23 column classification."""
import os,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage25.common import *
from scripts.stage22.solver import solve
from guidance.property_volume import property_table_from_config,hard_labels_to_properties
import torch

def main():
    guard('restore');torch.set_num_threads(4)
    if (ROOT/'SOLVER_STATE.json').exists(): raise FileExistsError('One solve only; preserve previous state')
    f=read_json(ROOT/'OBSERVATION_FREEZE.json');assert f['source_sha256']==sources()
    assert f['codebook_sha256']==file_hash(ROOT/'codebook.json') and f['layout_sha256']==file_hash(ROOT/'layout.json')
    old=read_json(R22/'METHOD_FREEZE.json')
    assert file_hash(ROOT/'codebook.json')==old['codebook_sha256']
    assert all(file_hash(REPO/p)==h for p,h in old['source_sha256'].items())
    obs=verify_tensor(f['observations']);original=check_condition(read_json(ROOT/'arms/A/inference_registry.json')['conditions'])
    table,weights,_=property_table_from_config(read_json(ROOT/'codebook.json'),15)
    scales=table[:,1:].max(1).values-table[:,1:].min(1).values
    mask=original['mask'];values=original['values']
    y=obs.double()/scales.double()[None,:,None,None,None]
    known=hard_labels_to_properties(values,table).double()/scales.double()[None,:,None,None,None]
    lower=(table.double().min(1).values/scales.double())[None,:,None,None,None]
    upper=(table.double().max(1).values/scales.double())[None,:,None,None,None]
    started=time.monotonic();write_json(ROOT/'SOLVER_STATE.json',dict(status='running',solve_invocations=1,started_at_utc=utc()))
    def progress(row):
        if row['iteration']%100==0:
            write_json(ROOT/'SOLVER_PROGRESS.json',row)
            print(f"iteration={row['iteration']} objective={row['objective']:.9g} stable={row['stable_count']}",flush=True)
    z,history,converged=solve(y,lower,upper,known,mask,callback=progress)
    assert torch.isfinite(z).all() and not ((z<lower)|(z>upper)).any() and torch.equal(z[mask.expand_as(z)],known[mask.expand_as(z)])
    restored=(z*scales.double()[None,:,None,None,None]).float()
    labels=classify(restored,table,weights)
    layout=read_json(ROOT/'layout.json');cv,cm=dense_conditions(values,mask,labels,layout['levels']['512'])
    b=check_condition(read_json(ROOT/'arms/B/inference_registry.json')['conditions'])
    assert torch.equal(cm,b['mask']) and not ((cm&~mask)&well_mask(layout['holdout'])).any()
    assets={k:save_tensor(ROOT/'preprocessing'/f'{k}.pt',v) for k,v in dict(z=z,properties=restored,classification=labels).items()}
    write_json(ROOT/'preprocessing_assets.json',assets)
    write_csv(ROOT/'solver_history.csv',history)
    seal_arm('C',cv,cm,values,mask,f['noise'],f['source_sha256'],read_json(ROOT/'PROTOCOL.json'))
    write_json(ROOT/'SOLVER_STATE.json',dict(status='completed' if converged else 'unconverged',solve_invocations=1,iterations=len(history)-1,converged=converged,final=history[-1],elapsed_seconds=time.monotonic()-started,truth_access=False,completed_at_utc=utc(),assets=assets))
    write_json(ROOT/'CONDITIONS_FROZEN.json',dict(sealed_at_utc=utc(),arms={a:file_hash(ROOT/f'arms/{a}/inference_registry.json') for a in 'ABC'},restored_assets=assets,mask_B_equals_C=True,holdout_added_voxels=0,truth_access=False))
    if not converged: raise RuntimeError('Numerical nonconvergence; no scientific parameter changes authorized')
    print('C_CONDITIONS_FROZEN',flush=True)
if __name__=='__main__': main()
