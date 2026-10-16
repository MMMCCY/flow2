"""Only small numerical/projection checks required by Stage22."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from scripts.stage22.solver import blur,tvmean,objective,project,solve
from scripts.stage22.common import sparse_conditions
from guidance.property_volume import gaussian_blur_property_channels


def test_operator_matches_old_and_preserves_constant():
    torch.set_num_threads(2)
    x=torch.randn(1,2,7,8,9,generator=torch.Generator().manual_seed(22))
    assert torch.equal(blur(x),gaussian_blur_property_channels(x,1.))
    assert (blur(x.double()).float()-blur(x)).abs().max()<4e-7
    assert (blur(torch.ones_like(x))-1).abs().max()<4e-7


def test_boundary_autograd_and_tv_directional_gradient():
    g=torch.Generator().manual_seed(22)
    x=torch.randn(1,2,5,6,7,generator=g,dtype=torch.float64).requires_grad_(True)
    y=torch.randn(x.shape,generator=g,dtype=torch.float64)
    direction=torch.randn(x.shape,generator=g,dtype=torch.float64)
    f=objective(x,y)[0]
    grad=torch.autograd.grad(f,x)[0]
    step=1e-5
    fd=(objective(x.detach()+step*direction,y)[0]-objective(x.detach()-step*direction,y)[0])/(2*step)
    assert torch.allclose(fd,(grad*direction).sum(),rtol=1e-6,atol=1e-8)
    # At replicate boundaries the actual transpose is obtained through autograd.
    assert torch.isfinite(grad).all()
    constant=torch.ones_like(x)
    assert abs(float(tvmean(constant))-1e-4)<1e-12


def test_toy_deconvolution_decreases_objective_with_exact_projection():
    x=torch.zeros(1,1,8,8,8,dtype=torch.float64)
    x[:,:,2:6,2:6,2:6]=1
    y=blur(x)
    mask=torch.zeros_like(x,dtype=torch.bool)
    mask[:,:,:,0,:]=True
    result,history,_=solve(y,torch.tensor(0.),torch.tensor(1.),x,mask,max_iterations=100)
    assert history[-1]["objective"]<history[0]["objective"]*.3
    assert torch.equal(result[mask],x[mask])
    assert result.min()>=0 and result.max()<=1
    assert ((blur(result)-y).square().mean())<((blur(y)-y).square().mean())


def test_target_only_conditions_original_priority_and_holdout():
    values=torch.full((1,1,3,3,3),-1.)
    mask=torch.zeros_like(values,dtype=torch.bool)
    mask[...,0]=True
    values[mask]=2
    pred=torch.full_like(values,9,dtype=torch.long)
    d9=torch.zeros_like(values)
    margin=torch.ones_like(values)
    eligible=torch.ones_like(mask)
    eligible[:,:,1,1,:]=False
    # Strict fixed filters retain all FP candidates as well; no truth argument.
    d9[:,:,0,0,1]=.02
    margin[:,:,2,2,1]=.04
    v,m,added=sparse_conditions(values,mask,pred,d9,margin,eligible)
    assert torch.equal(v[mask],values[mask]) and torch.equal(m,mask|added)
    assert (v[added]==9).all() and (v[~m]==-1).all()
    assert not added[:,:,1,1,:].any() and not (added&mask).any()
    assert not added[:,:,0,0,1].any() and not added[:,:,2,2,1].any()
