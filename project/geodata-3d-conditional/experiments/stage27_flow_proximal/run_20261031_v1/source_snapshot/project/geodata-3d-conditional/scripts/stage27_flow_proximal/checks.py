"""Small necessary checks only; zero velocity network evaluations."""
import argparse
import json
import torch
from .sampler import EndpointObjective,proximal,project,reproject_noise
from .common import physics,R26,tensor_hash
from guidance.seismic import ConvolutionalSeismic

def main(device):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(27)
    dtype=torch.float64;shape=(1,15,3,4,5)
    embedding=torch.eye(15,device=device,dtype=dtype)
    values=torch.zeros((1,1,3,4,5),device=device,dtype=torch.long)
    mask=torch.zeros_like(values,dtype=torch.bool);mask[:,:,:1]=True
    sub=torch.ones_like(mask);sub[:,:,:,:,-1]=False
    table,_=physics({'acoustic':json.loads((R26/'configs/acoustic_codebook.json').read_text()),
                     'observation':json.loads((R26/'configs/observation_config.json').read_text())})
    table=table.to(device)
    operator=ConvolutionalSeismic((3,4,5),cell_size_m=60.95238095238095,
        num_time_samples=400,sample_interval_ms=8,peak_frequency_hz=25,wavelet_duration_ms=128)
    center=torch.randn(shape,device=device,dtype=dtype)
    base=torch.zeros_like(center);free=(sub&~mask).expand_as(center)
    center=project(center,base,free)
    observation=torch.zeros((1,1,3,4,400),device=device,dtype=dtype)
    observation[...,10]=.1
    counts=dict(physics_forwards=0,physics_backwards=0)
    objective=EndpointObjective(embedding,table,operator,values,mask,sub,observation,1,1,counts)
    u=center.clone().requires_grad_();q,_=objective.terms(u,center,.5)
    g,=torch.autograd.grad(q,u)
    assert torch.isfinite(g).all() and g[free].abs().sum()>0 and not g[~free].any()
    refined,trace,_=proximal(center,base,objective,.5)
    assert torch.equal(refined[~free],base[~free])
    assert all(t['Q_after']<=t['Q_before']+1e-12 for t in trace)
    assert all(t['direction_rms']<=.1+1e-12 for t in trace)
    def forbidden(*args):
        raise AssertionError('Inactive arm called seismic')
    for a,r in [(0,0),(0,1)]:
        inactive=EndpointObjective(embedding,table,forbidden,values,mask,sub,None,a,r,counts)
        output,tr,_=proximal(center,base,inactive,.5)
        if not r:
            assert torch.equal(output,center) and tr==[]
    assert tensor_hash(reproject_noise(shape,0))==tensor_hash(reproject_noise(shape,0))
    assert tensor_hash(reproject_noise(shape,0))!=tensor_hash(reproject_noise(shape,1))
    print(json.dumps(dict(status='passed',device=device,physics_gradient_nonzero=True,
        nonfree_gradient_zero=True,armijo_monotone=True,direction_rms_cap=True,
        F0_and_Fr_no_seismic=True,paired_cpu_noise_replay=True,network_evaluations=0,
        checks_physics_forwards=counts['physics_forwards'],checks_backward='autograd.grad actual helper'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    main(p.parse_args().device)
