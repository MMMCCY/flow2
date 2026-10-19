import sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from scripts.stage26_diagnostics.common import candidates_from_conditions,classify
from guidance.seismic_inversion import cell_center_twt_ms

def test_observed_candidates_only_original_mask_no_target_injection():
    v=torch.tensor([0,5,-1,13,9]).reshape(1,1,1,1,5);m=torch.tensor([True,True,True,False,False]).reshape_as(v)
    assert candidates_from_conditions(v,m)==[0,5]

def test_classification_same_distance_tie_and_observed_air_priority():
    table=torch.stack((torch.exp(torch.arange(15,dtype=torch.float64)/5),torch.ones(15)))
    table[0,2]=table[0,1]
    a=table[:,[1,14,10,14]].reshape(1,2,1,1,4)
    sub=torch.tensor([True,True,False,True]).reshape(1,1,1,1,4)
    values=torch.tensor([-1,-1,-1,3]).reshape_as(sub);mask=torch.zeros_like(sub);mask[...,3]=True
    assert classify(a,table,list(range(14)),sub,values,mask).flatten().tolist()==[0,13,-1,3]
    pred=classify(a,table,[0,5],sub,values,mask)
    assert pred.flatten().tolist()==[0,5,-1,3]
    a[:,1]=999
    assert torch.equal(pred,classify(a,table,[0,5],sub,values,mask))

def test_twt_output_must_flip_back_to_original_xyz_for_domain_statistics():
    s=torch.full((1,1,1,1,4),.001,dtype=torch.float64);sub=torch.ones_like(s,dtype=torch.bool)
    top_down=cell_center_twt_ms(s,sub,1.)
    assert torch.equal(top_down,torch.tensor([1.,3.,5.,7.]).reshape_as(s))
    assert top_down.flip(-1).flatten().tolist()==[7.,5.,3.,1.]

def test_actual_save_hash_load_and_protected_inputs():
    code='''
import torch,os
from scripts.stage26_diagnostics.common import *
ROOT.mkdir(parents=True,exist_ok=True)
guard('prepare')
p=ROOT/'engineering'/f'probe_{os.getpid()}.pt';v=torch.arange(12,dtype=torch.float64)
record=save_tensor(p,v);assert torch.equal(verify_tensor(record),v);p.unlink()
for path,mode in ((STAGE25/'truth/labels.pt','rb'),(STAGE25/'arms/B/conditions/values.pt','rb'),(FORMAL/'REPORT.md','wb'),(ROOT/'fake.ckpt','rb')):
    try:path.open(mode)
    except PermissionError:continue
    raise AssertionError(path)
'''
    r=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
