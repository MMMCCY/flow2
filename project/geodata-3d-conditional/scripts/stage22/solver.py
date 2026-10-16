"""Smooth convex bounded TV deconvolution with projected accelerated gradients."""
import math
import torch
import torch.nn.functional as F
from guidance.property_volume import _gaussian_kernel_1d, gaussian_blur_property_channels


def blur(value):
    # Freeze the SAME float32 coefficients used to create the observation. FP64
    # solver arithmetic does not silently change them to a different kernel.
    if value.dtype == torch.float32:
        return gaussian_blur_property_channels(value, 1.)
    kernel = _gaussian_kernel_1d(1., value.device, torch.float32).to(value.dtype)
    channels = value.shape[1]
    for shape, padding in [((-1,1,1),(0,0,0,0,3,3)), ((1,-1,1),(0,0,3,3,0,0)), ((1,1,-1),(3,3,0,0,0,0))]:
        weights = kernel.reshape(1,1,*shape).expand(channels,1,*([-1 if n == -1 else 1 for n in shape]))
        value = F.conv3d(F.pad(value,padding,mode="replicate"),weights,groups=channels)
    return value


def tvmean(z):
    dx = F.pad(z[:,:,1:,:,:]-z[:,:,:-1,:,:],(0,0,0,0,0,1))
    dy = F.pad(z[:,:,:,1:,:]-z[:,:,:,:-1,:],(0,0,0,1,0,0))
    dz = F.pad(z[:,:,:,:,1:]-z[:,:,:,:,:-1],(0,1,0,0,0,0))
    return (dx.square()+dy.square()+dz.square()+1e-8).sqrt().mean()


def objective(z,y):
    data = (blur(z)-y).square().mean()
    tv = tvmean(z)
    return data + 1e-4*tv, data, tv


def project(z,lower,upper,known,mask):
    return torch.where(mask,known,torch.maximum(lower,torch.minimum(upper,z)))


def solve(y,lower,upper,known,mask,max_iterations=2000,callback=None):
    """Backtracking uses scaled gradients to cancel the common mean denominator.

    Every accepted iterate is projected. Extrapolated points need not be
    feasible; the smooth objective is defined everywhere. If acceleration
    increases the objective, restart from the last accepted iterate. No truth
    or categorical score is available to this function.
    """
    if y.dtype != torch.float64:
        raise ValueError("solver uses float64")
    x=project(y.clone(),lower,upper,known,mask)
    v=x.clone()
    momentum=1.
    L=4.
    stable=0
    history=[]
    with torch.no_grad():
        previous,dat,tv=map(float,objective(x,y))
    history.append(dict(iteration=0,objective=previous,data_mse=dat,TVmean=tv,L=L,
                        relative_change=None,restarted=False,stable_count=0))
    for iteration in range(1,max_iterations+1):
        restarted=False
        backtracks=0
        while True:
            differentiable=v.detach().requires_grad_(True)
            value,_,_=objective(differentiable,y)
            # Same mean objective, scaled gradient; use mean inner products below.
            gradient=torch.autograd.grad(value,differentiable)[0]*y.numel()
            fv=float(value.detach())
            with torch.no_grad():
                while True:
                    candidate=project(v-gradient/L,lower,upper,known,mask)
                    f,data,tv=objective(candidate,y)
                    delta=candidate-v
                    majorant=fv+float((gradient*delta).mean())+.5*L*float(delta.square().mean())
                    if float(f)<=majorant+1e-15:
                        break
                    L*=2
                    backtracks+=1
                    if backtracks>60:
                        raise FloatingPointError("backtracking could not majorize smooth objective")
                if float(f)<=previous+1e-15:
                    break
                if restarted:
                    raise FloatingPointError("projected gradient failed monotone descent")
                v=x.clone()
                momentum=1.
                restarted=True
        current=float(f)
        if not math.isfinite(current) or not torch.isfinite(candidate).all():
            raise FloatingPointError("nonfinite solver iterate")
        relative=abs(previous-current)/max(abs(previous),1e-30)
        stable=stable+1 if relative<1e-6 else 0
        new_momentum=.5*(1+math.sqrt(1+4*momentum*momentum))
        with torch.no_grad():
            v=candidate+(momentum-1)/new_momentum*(candidate-x)
            x=candidate
        momentum=new_momentum
        previous=current
        row=dict(iteration=iteration,objective=current,data_mse=float(data),TVmean=float(tv),L=L,
                 relative_change=relative,restarted=restarted,backtracks=backtracks,stable_count=stable)
        history.append(row)
        if callback:
            callback(row)
        if stable>=50:
            return x,history,True
    return x,history,False
