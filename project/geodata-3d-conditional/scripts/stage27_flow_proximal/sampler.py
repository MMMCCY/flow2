"""Endpoint-only proximal gradients; no graph through velocity or past steps."""
import time
import torch
from guided_geophysical_sampling import soft_decode_to_probs
from guidance.seismic import (probabilities_to_subsurface_acoustic,
    overwrite_exact_condition_acoustic, hard_labels_to_acoustic)
from .common import CONFIG, tensor_hash

def project(u, base, free):
    return torch.where(free, u, base)

def structure_energy(q, subsurface, values, mask):
    rock=q[:,1:]; rock=rock/rock.sum(1,keepdim=True).clamp_min(1e-12)
    target=rock[:,9:10]
    target=torch.where(mask & subsurface,(values==9).to(q.dtype),target)
    total=target.new_zeros(()); count=0
    for axis in (2,3,4):
        left=[slice(None)]*5; right=left.copy()
        left[axis]=slice(None,-1); right[axis]=slice(1,None)
        left=tuple(left); right=tuple(right)
        edges=subsurface[left]&subsurface[right]
        delta=target[left]-target[right]
        total=total+((delta.square()+1e-6).sqrt()-1e-3).masked_select(edges).sum()
        count+=int(edges.sum())
    if not count:
        raise ValueError('Empty structure edge domain')
    return total/count

class EndpointObjective:
    def __init__(self, embedding, table, operator, values, mask, subsurface, observation,
                 data_active, structure_active, counters):
        self.embedding=embedding.detach()
        self.table=table.detach()
        self.operator=operator; self.values=values; self.mask=mask; self.sub=subsurface
        self.obs=observation; self.a=data_active; self.r=structure_active
        self.counters=counters
        self.exact=hard_labels_to_acoustic(values,table)
        self.free=(subsurface & ~mask).expand(-1,embedding.shape[1],-1,-1,-1)
        self.nfree=int(self.free.sum())
        if self.nfree==0:
            raise ValueError('No free elements')
        self.norm=None if observation is None else observation.square().sum().clamp_min(1e-12)

    def terms(self,u,center,nu2):
        trust=.5*(u-center).square().masked_select(self.free).mean()
        ey=u.new_zeros(()); reg=u.new_zeros(())
        if self.a or self.r:
            q=soft_decode_to_probs(u,self.embedding.to(u),tau=.1)
            if self.a:
                acoustic=probabilities_to_subsurface_acoustic(q,self.table,self.sub)
                acoustic=overwrite_exact_condition_acoustic(acoustic,self.exact,self.mask)
                field=self.operator(acoustic[:,0:1],acoustic[:,1:2],self.sub)
                self.counters['physics_forwards']+=1
                ey=(field-self.obs).square().sum()/self.norm
            if self.r:
                reg=structure_energy(q,self.sub,self.values,self.mask)
        objective=trust+nu2*(self.a*ey+self.r*.05*reg)
        if not torch.isfinite(objective):
            raise FloatingPointError('Nonfinite proximal objective')
        return objective,dict(trust=float(trust.detach()),
            data_energy=float(ey.detach()) if self.a else None,
            structure_energy=float(reg.detach()) if self.r else None)

def proximal(center,base,objective,nu2):
    if not objective.a and not objective.r:
        return center,[],dict(before=None,after=None)
    u=center.detach(); inner=[]; before=None; after=None
    for k in range(CONFIG['prox_steps']):
        u=u.detach().requires_grad_(True)
        current,terms=objective.terms(u,center,nu2)
        if before is None:
            before=terms.copy()
        grad,=torch.autograd.grad(current,u)
        if objective.a:
            objective.counters['physics_backwards']+=1
        g=torch.where(objective.free,grad*objective.nfree,torch.zeros_like(grad))
        if not torch.isfinite(g).all():
            raise FloatingPointError('Nonfinite proximal gradient')
        rms=g.masked_select(objective.free).square().mean().sqrt()
        direction=g/torch.clamp(rms/.1,min=1.)
        slope=(g*direction).masked_select(objective.free).mean()
        record=dict(inner=k,Q_before=float(current.detach()),**terms,
                    gradient_rms=float(rms),direction_rms=float(
                        direction.masked_select(objective.free).square().mean().sqrt()),
                    accepted_alpha=None,line_search_exhausted=False,zero_gradient=bool(rms==0),
                    candidate_evaluations=0)
        after=terms.copy()
        with torch.no_grad():
            if rms!=0:
                for alpha in CONFIG['line_search_steps']:
                    candidate=project(u-alpha*direction,base,objective.free)
                    candidate_q,candidate_terms=objective.terms(candidate,center,nu2)
                    record['candidate_evaluations']+=1
                    if candidate_q<=current.detach()-1e-4*alpha*slope:
                        u=candidate; after=candidate_terms
                        record.update(accepted_alpha=alpha,Q_after=float(candidate_q),after=candidate_terms)
                        break
                else:
                    record['line_search_exhausted']=True
        if record['accepted_alpha'] is None:
            record['Q_after']=record['Q_before']
        inner.append(record)
    return u.detach(),inner,dict(before=before,after=after)

def reproject_noise(shape,step):
    return torch.randn(shape,generator=torch.Generator(device='cpu').manual_seed(1042+step),
                       dtype=torch.float32,device='cpu')

def sample(model,initial,values,mask,subsurface,objective,record_step,ledger_nfe):
    device=next(model.parameters()).device
    with torch.no_grad():
        base=model.embed(values.to(device)); cond=base*mask.to(device)
        free=(subsurface & ~mask).to(device).expand_as(base)
        roundtrip=(model.decode(base)-1).unsqueeze(1)
        if not torch.equal(roundtrip[mask.to(device)],values.to(device)[mask.to(device)].long()):
            raise ValueError('Condition embed/decode roundtrip')
        x=project(initial.to(device),base,free)
    prox_device=objective.embedding.device
    prox_base=base.detach().to(device=prox_device,dtype=torch.float64)
    traces=[]
    for j in range(32):
        start=time.monotonic()
        t=max(float(model.time_range[0]),min(j/32,float(model.time_range[1])))
        nu2=(1-t)**2/(t*t+(1-t)**2)
        with torch.no_grad():
            # Reserve the NFE durably BEFORE the call; interrupted calls are never silently repeated.
            ledger_nfe(j)
            v=model.net(x,cond,torch.full((1,),t,device=device,dtype=torch.float32))
            center=project(x+(1-t)*v,base,free).detach()
            if not torch.isfinite(center).all():
                raise FloatingPointError('Nonfinite endpoint estimate')
        counters_before=objective.counters.copy()
        # F0 preserves its FP32 endpoint exactly; active proximal updates use FP64.
        if objective.a or objective.r:
            u,inner,terms=proximal(center.to(device=prox_device,dtype=torch.float64),prox_base,objective,nu2)
        else:
            u=center; inner=[]; terms=dict(before=None,after=None)
        with torch.no_grad():
            u_net=u.to(device=device,dtype=torch.float32)
            nh=None
            if j<31:
                noise=reproject_noise(tuple(x.shape),j); nh=tensor_hash(noise)
                b=(j+1)/32
                x=project(b*u_net+(1-b)*noise.to(device),base,free)
            else:
                x=project(u_net,base,free)
            if not torch.equal(x[~free],base[~free]):
                raise ValueError('Nonfree projection violated')
        if device.type=='cuda':
            torch.cuda.synchronize(device)
        trace=dict(step=j,nfe=j+1,time_nominal=j/32,time_eval=t,time_next=(j+1)/32,
            nu2=nu2,noise_seed=1042+j if j<31 else None,noise_tensor_sha256=nh,
            terms=terms,inner=inner,physics_forwards=objective.counters['physics_forwards']-counters_before['physics_forwards'],
            physics_backwards=objective.counters['physics_backwards']-counters_before['physics_backwards'],
            elapsed_seconds=time.monotonic()-start,condition_embedding_max_error=0.)
        traces.append(trace); record_step(trace,x,u)
        print(f'step={j+1}/32 t={t:.6f} seconds={trace["elapsed_seconds"]:.3f} physics={trace["physics_forwards"]}',flush=True)
    with torch.no_grad():
        decoded=(model.decode(x)-1).unsqueeze(1).long().cpu()
    if ((decoded!=values.cpu())&mask.cpu()).any():
        raise ValueError('Final hard condition violation')
    return decoded,x.detach().cpu(),u.detach().cpu(),traces
