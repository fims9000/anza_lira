"""Differentiable structure-aware continuity evidence from CT and candidate endpoints.

The Gaussian endpoint heatmap is the only source of endpoint locations. The
readout never sees labels, ground-truth masks or post-hoc path identities.
Physical vessel evidence is learned upstream, and soft bottlenecks on multiple
candidate paths make continuity matter. This is a structural inductive bias,
not a mathematical guarantee of preserving vessel topology.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def _peak_centers(heat:torch.Tensor,degree:int,min_gap:float=3.0):
    if heat.ndim!=4 or degree not in (2,3):
        raise ValueError("Expected B,D,H,W heatmap and degree 2 or 3")
    b,d,h,w=heat.shape
    work=heat.detach().reshape(b,-1).clone()
    z,y,x=torch.meshgrid(
        torch.arange(d,device=heat.device),
        torch.arange(h,device=heat.device),
        torch.arange(w,device=heat.device),indexing="ij",
    )
    grid=torch.stack((z,y,x),dim=-1).reshape(1,-1,3)
    chosen=[]
    valid=torch.ones(b,device=heat.device,dtype=torch.bool)
    strongest=None
    for _ in range(degree):
        peak,index=work.max(dim=1)
        if strongest is None:
            strongest=peak
        valid=valid & (peak>strongest*.20) & (peak>.05)
        center=grid.expand(b,-1,-1).gather(
            1,index[:,None,None].expand(-1,-1,3),
        ).squeeze(1).to(torch.float32)
        chosen.append(center)
        diff=grid.to(torch.float32)-center[:,None,:]
        work=work.masked_fill((diff*diff).sum(dim=-1)<min_gap**2,-1.)
    return torch.stack(chosen,dim=1),valid


def _sample_paths(vessel:torch.Tensor,positions:torch.Tensor):
    if vessel.ndim!=4 or positions.ndim!=4 or positions.shape[-1]!=3:
        raise ValueError("Expected vessel B,D,H,W and path points B,P,S,3")
    b,d,h,w=vessel.shape
    if positions.shape[0]!=b or min(d,h,w)<4:
        raise ValueError("Invalid batch or physical patch shape")
    scale=positions.new_tensor([d-1,h-1,w-1])
    # grid_sample expects x/y/z = W/H/D, but our coordinates are D/H/W.
    normalized=(2*positions/scale-1).flip(-1)
    paths,steps=positions.shape[1:3]
    grid=normalized.reshape(b,paths*steps,1,1,3)
    values=F.grid_sample(
        vessel[:,None],grid,align_corners=True,
        padding_mode="zeros",mode="bilinear",
    )[:,0,:,0,0]
    return values.reshape(b,paths,steps)


def _pair_paths(peaks:torch.Tensor,steps:int,curvature:float):
    first,last=peaks[:,0],peaks[:,1]
    segment=last-first
    axis=F.normalize(segment,dim=-1,eps=1e-6)
    reference=F.one_hot(axis.abs().argmin(dim=-1),num_classes=3).to(axis)
    cross1=F.normalize(torch.cross(axis,reference,dim=-1),dim=-1,eps=1e-6)
    cross2=torch.cross(axis,cross1,dim=-1)
    t=torch.linspace(0,1,steps,device=peaks.device,dtype=peaks.dtype)
    straight=first[:,None,:]+t[None,:,None]*segment[:,None,:]
    bend=curvature*torch.sin(t*math.pi)[None,:,None]
    options=[straight,straight+bend*cross1[:,None,:],
             straight-bend*cross1[:,None,:],
             straight+bend*cross2[:,None,:],
             straight-bend*cross2[:,None,:]]
    return torch.stack(options,dim=1)


def _junction_paths(peaks:torch.Tensor,steps:int):
    hub=peaks.mean(dim=1,keepdim=True)
    t=torch.linspace(0,1,steps,device=peaks.device,dtype=peaks.dtype)
    return (peaks[:,:,None,:]*(1-t)[None,None,:,None]+
            hub[:,:,None,:]*t[None,None,:,None])


def _soft_min(values:torch.Tensor,temperature:float):
    return (-temperature*torch.logsumexp(-values/temperature,dim=-1)
            +temperature*math.log(values.shape[-1]))


class VesselBridgeReadout(nn.Module):
    """PAIR: best of 5 straight/curved routes; JUNCTION: weakest of 3 arms."""
    def __init__(self,degree:int,steps:int=11,curvature:float=1.6,
                 temperature:float=.1):
        super().__init__()
        if degree not in (2,3) or steps<3 or curvature<0 or temperature<=0:
            raise ValueError("Invalid path readout configuration")
        self.degree=int(degree)
        self.steps=int(steps)
        self.curvature=float(curvature)
        self.temperature=float(temperature)

    def forward(self,vessel_probability:torch.Tensor,heatmap:torch.Tensor):
        if vessel_probability.shape!=heatmap.shape or heatmap.ndim!=4:
            raise ValueError("Vesselness and endpoint masks must be B,D,H,W")
        if not torch.isfinite(vessel_probability).all() or not torch.isfinite(heatmap).all():
            raise ValueError("Nonfinite structural evidence")
        peaks,valid=_peak_centers(heatmap,self.degree)
        paths=(_pair_paths(peaks,self.steps,self.curvature)
               if self.degree==2 else _junction_paths(peaks,self.steps))
        values=_sample_paths(vessel_probability,paths)
        min_per_path=_soft_min(values,self.temperature)
        if self.degree==2:
            bridge=min_per_path.max(dim=1).values
        else:
            bridge=min_per_path.min(dim=1).values
        backup=vessel_probability.flatten(1).mean(dim=1)
        return torch.where(valid,bridge,backup).clamp(1e-5,1-1e-5)


def soft_skeleton3d(mask:torch.Tensor,iterations:int=3):
    if mask.ndim!=5 or mask.shape[1]!=1 or iterations<0:
        raise ValueError("Expected vessel mask B,1,D,H,W")
    def minpool(x): return -F.max_pool3d(-x,3,1,1)
    def maxpool(x): return F.max_pool3d(x,3,1,1)
    skel=F.relu(mask-maxpool(minpool(mask)))
    active=mask
    for _ in range(iterations):
        active=minpool(active)
        excess=F.relu(active-maxpool(minpool(active)))
        skel=skel+F.relu(excess-skel*excess)
    return skel


def vessel_structure_loss(logits:torch.Tensor,truth:torch.Tensor,
                          skeleton_iterations:int=3):
    """Differentiable BCE + Dice + clDice on independently supervised train masks."""
    if logits.shape!=truth.shape or logits.ndim!=5 or logits.shape[1]!=1:
        raise ValueError("Vessel target must match dense B,1,D,H,W logits")
    if not torch.isfinite(truth).all() or bool(((truth<0)|(truth>1)).any()):
        raise ValueError("Nonfinite/out-of-range vessel target")
    p=torch.sigmoid(logits)
    # Thin coronary vessels occupy a tiny fraction of the patch. Without
    # balancing the dense BCE, an all-background prediction can dominate the
    # auxiliary task and make the topology bridge readout effectively constant.
    positives=truth.sum()
    negatives=truth.numel()-positives
    pos_weight=(negatives/positives.clamp_min(1.)).clamp(1.,32.).detach()
    bce=F.binary_cross_entropy_with_logits(logits,truth,pos_weight=pos_weight)
    dims=(1,2,3,4)
    intersect=(p*truth).sum(dim=dims)
    dice=(2*intersect+1e-5)/(p.sum(dim=dims)+truth.sum(dim=dims)+1e-5)
    skeleton_p=soft_skeleton3d(p,skeleton_iterations)
    skeleton_y=soft_skeleton3d(truth,skeleton_iterations)
    tprec=((skeleton_p*truth).sum(dim=dims)+1e-5)/(skeleton_p.sum(dim=dims)+1e-5)
    tsens=((skeleton_y*p).sum(dim=dims)+1e-5)/(skeleton_y.sum(dim=dims)+1e-5)
    cldice=(2*tprec*tsens+1e-5)/(tprec+tsens+1e-5)
    return .4*bce+.3*(1-dice.mean())+.3*(1-cldice.mean())
