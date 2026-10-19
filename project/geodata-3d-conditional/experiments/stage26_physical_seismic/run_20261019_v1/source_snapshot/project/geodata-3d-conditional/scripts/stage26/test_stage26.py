"""Focused lifecycle/firewall and single-quantity classification checks; zero network calls."""
import sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from scripts.stage26.common import classify_impedance

def test_actual_saved_tensor_hash_readback_and_load_under_firewall():
    code='''
from scripts.stage26.common import *
import torch,os
ROOT.mkdir(parents=True,exist_ok=True)
guard('prepare')
p=ROOT/'preprocessing'/f'.engineering_tensor_probe_{os.getpid()}.pt'
x=torch.arange(24,dtype=torch.float64).reshape(1,2,2,2,3)
a=save_tensor(p,x)
assert torch.equal(verify_tensor(a),x)
assert file_hash(p)==a['file_sha256']
p.unlink()
for forbidden in (OLD/'truth/labels.pt',OLD/'arms/B/conditions/values.pt',OLD/'arms/C/conditions/values.pt',OLD/'preprocessing/properties.pt',ROOT/'evaluation_registry.json'):
    try:forbidden.open('rb')
    except PermissionError:continue
    raise AssertionError(forbidden)
try:(OLD/'physics/preprocessing/illegal.pt').open('wb')
except PermissionError:pass
else:raise AssertionError('Historical write allowed')
print('PASS actual save -> rb hash -> verified load; own output reads allowed; forbidden inputs and historical writes blocked')
'''
    r=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True)
    assert r.returncode==0,r.stderr

def test_log_impedance_classifier_uses_all_classes_first_tie_air_and_original_priority():
    table=torch.stack((torch.exp(torch.arange(15,dtype=torch.float64)/5),torch.arange(15,dtype=torch.float64)+1))
    table[0,2]=table[0,1]
    acoustic=table[:,[1,14,10,14]].reshape(1,2,1,1,4)
    subsurface=torch.tensor([True,True,False,True]).reshape(1,1,1,1,4)
    original=torch.tensor([-1,-1,-1,3]).reshape_as(subsurface).long();mask=torch.zeros_like(subsurface);mask[...,3]=True
    pred=classify_impedance(acoustic,table,subsurface,original,mask)
    assert pred.flatten().tolist()==[0,13,-1,3]
    other=acoustic.clone();other[:,1]=999  # unchanged slowness is not classification evidence
    assert torch.equal(pred,classify_impedance(other,table,subsurface,original,mask))
