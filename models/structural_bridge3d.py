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


def _peak_centers_from_channels(heatmap:torch.Tensor,degree:int):
    """Decode individual 3D endpoint channels without cross-tip suppression.

    Each candidate tip occupies its own Gaussian field. Exact endpoint
    positions may be arbitrarily close; a suppression radius or global
    summed-heatmap peak search would destroy PAIR/JUNCTION identity.
    """
    if heatmap.ndim!=5 or heatmap.shape[1]!=3 or degree not in (2,3):
        raise ValueError("Separate endpoint heatmap must be B,3,D,H,W")
    b,_,d,h,w=heatmap.shape
    selected=heatmap[:,:degree].flatten(2)
    peak=selected.max(dim=2).values
    # Each channel is ONE known Gaussian. Its squared-intensity centroid
    # recovers a continuous subvoxel tip coordinate even when two tips are
    # closer than 3 mm and their maxima occupy nearby/identical voxels.
    # Unlike integer argmax, a .2-mm position perturbation need not jump by
    # a complete voxel. Zero-padding remains a fail-closed invalid tip.
    weights=selected.clamp_min(0.).square()
    weight_sum=weights.sum(dim=2)
    coords=torch.stack(torch.meshgrid(
        torch.arange(d,device=heatmap.device,dtype=heatmap.dtype),
        torch.arange(h,device=heatmap.device,dtype=heatmap.dtype),
        torch.arange(w,device=heatmap.device,dtype=heatmap.dtype),
        indexing="ij",
    ),dim=-1).reshape(-1,3)
    centers=torch.einsum("bkn,nc->bkc",weights,coords)/weight_sum.clamp_min(1e-8)[...,None]
    # A missing/cropped endpoint must abstain rather than replacing it
    # by an unrelated high-vesselness image statistic.
    valid=(peak>.05).all(dim=1)&(weight_sum>1e-5).all(dim=1)&torch.isfinite(peak).all(dim=1)
    return centers,valid


def endpoint_tangent_alignment(
    vessel_probability:torch.Tensor,heatmap:torch.Tensor,*,radius_voxels:float=2.4
)->torch.Tensor:
    """Differentiable PAIR tangent/continuation evidence, diagnostic only.

    Uses local vessel covariance near each endpoint to test whether the
    candidate chord follows the *local* vessel direction. Unlike binary
    connectivity this can reject an incorrect turn within the SAME tree.
    It is not a proof of anatomical correctness at a true bifurcation.
    Input: vessel [B,D,H,W], individual tip fields [B,3,D,H,W].
    """
    if (vessel_probability.ndim!=4 or heatmap.ndim!=5 or
        heatmap.shape[1]!=3 or heatmap.shape[0]!=vessel_probability.shape[0] or
        heatmap.shape[2:]!=vessel_probability.shape[1:] or
        not math.isfinite(radius_voxels) or radius_voxels<=0):
        raise ValueError("Invalid 3D vessel/tip shape or covariance radius")
    if not torch.isfinite(vessel_probability).all() or not torch.isfinite(heatmap).all():
        raise ValueError("Nonfinite tangent evidence")
    b,d,h,w=vessel_probability.shape
    points,valid=_peak_centers_from_channels(heatmap,2)
    chord=points[:,1]-points[:,0]
    valid=valid&(torch.linalg.vector_norm(chord,dim=1)>1e-5)
    direction=F.normalize(chord,dim=1,eps=1e-6)
    coords=torch.stack(torch.meshgrid(
        torch.arange(d,device=vessel_probability.device,dtype=vessel_probability.dtype),
        torch.arange(h,device=vessel_probability.device,dtype=vessel_probability.dtype),
        torch.arange(w,device=vessel_probability.device,dtype=vessel_probability.dtype),
        indexing="ij",
    ),dim=-1).reshape(1,-1,3)
    vessel=vessel_probability.flatten(1).clamp(0,1)
    parts=[]
    for i in range(2):
        diff=coords-points[:,i,None,:]
        dist2=diff.square().sum(dim=-1)
        weights=vessel*torch.exp(-dist2/(2*radius_voxels**2))
        mass=weights.sum(dim=1).clamp_min(1e-8)
        mean=torch.einsum("bn,bni->bi",weights,diff)/mass[:,None]
        centered=diff-mean[:,None,:]
        cov=torch.einsum("bn,bni,bnj->bij",weights,centered,centered)/mass[:,None,None]
        trace=cov.diagonal(dim1=1,dim2=2).sum(dim=1)
        along=torch.einsum("bi,bij,bj->b",direction,cov,direction)
        anisotropy=(along-trace/3.)/(2*trace/3.+1e-6)
        parts.append(anisotropy.clamp(0,1))
    score=torch.stack(parts).mean(dim=0)
    return torch.where(valid,score,torch.zeros_like(score))


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


def _junction_adaptive_paths(peaks:torch.Tensor,steps:int,hub_radius_voxels:float=2.0):
    """3-arm routes to 27 nearby possible hubs instead of a single centroid.

    Return [B,27*3,S,3]. The 3 arms for each hub are scored jointly; a
    single nearby non-branch vessel cannot satisfy all three.
    """
    if peaks.ndim!=3 or peaks.shape[1:]!=(3,3):
        raise ValueError("Adaptive JUNCTION expects B,3,3 endpoint coordinates")
    device,dtype=peaks.device,peaks.dtype
    scale=float(hub_radius_voxels)
    offsets=torch.stack(torch.meshgrid(
        *(torch.tensor([-scale,0.,scale],device=device,dtype=dtype)
          for _ in range(3)),indexing="ij"
    ),dim=-1).reshape(27,3)
    centroid=peaks.mean(dim=1,keepdim=True)
    hubs=centroid+offsets[None]
    t=torch.linspace(0,1,steps,device=device,dtype=dtype)
    origins=peaks[:,None,:,None,:]       # B,1,3,1,3
    targets=hubs[:,:,None,None,:]        # B,27,1,1,3
    paths=origins*(1-t)[None,None,None,:,None]+targets*t[None,None,None,:,None]
    return paths.reshape(len(peaks),27*3,steps,3)


def _soft_min(values:torch.Tensor,temperature:float):
    return (-temperature*torch.logsumexp(-values/temperature,dim=-1)
            +temperature*math.log(values.shape[-1]))


class FuzzyConnectedness3d(nn.Module):
    """Differentiable max-min 26-neighbor reachability from one selected end.

    On a binary vessel mask, this is exactly connectivity after 'steps' waves.
    On learned probabilities, the value is the strongest bottleneck along a
    discrete 3D path. It does not distinguish branch identity at junctions.
    """
    def __init__(self,degree:int=3,steps:int=24):
        super().__init__()
        if degree not in (2,3) or steps<1 or steps>96:
            raise ValueError("Invalid fuzzy connectedness parameters")
        self.degree=degree
        self.steps=steps

    def forward(self,vessel_probability:torch.Tensor,heatmap:torch.Tensor):
        if vessel_probability.shape!=heatmap.shape or heatmap.ndim!=4:
            raise ValueError("Expected matched 3D vessel/endpoint probability fields")
        b,d,h,w=vessel_probability.shape
        peaks,valid=_peak_centers(heatmap,self.degree)
        coords=peaks.round().long()
        seed_index=coords[:,0,0]*h*w+coords[:,0,1]*w+coords[:,0,2]
        source=F.one_hot(seed_index,num_classes=d*h*w).to(vessel_probability.dtype)
        source=source.view(b,1,d,h,w)
        vessel=vessel_probability[:,None]
        reach=source*vessel
        for _ in range(self.steps):
            neigh=F.max_pool3d(reach,3,stride=1,padding=1)
            reach=torch.maximum(reach,torch.minimum(vessel,neigh))
        targets=coords[:,1:,0]*h*w+coords[:,1:,1]*w+coords[:,1:,2]
        value=reach.reshape(b,-1).gather(1,targets)
        output=value.min(dim=1).values
        backup=vessel_probability.flatten(1).mean(dim=1)
        return torch.where(valid,output,backup).clamp(1e-5,1-1e-5)


class VesselBridgeReadout(nn.Module):
    """PAIR: best of 5 straight/curved routes; JUNCTION: weakest of 3 arms."""
    def __init__(self,degree:int,steps:int=11,curvature:float=1.6,
                 temperature:float=.1,junction_strategy:str="centroid_star"):
        super().__init__()
        if degree not in (2,3) or steps<3 or curvature<0 or temperature<=0:
            raise ValueError("Invalid path readout configuration")
        if junction_strategy not in {"centroid_star","adaptive_star","fuzzy_reachability"}:
            raise ValueError("Unknown JUNCTION structure strategy")
        self.degree=int(degree)
        self.steps=int(steps)
        self.curvature=float(curvature)
        self.temperature=float(temperature)
        self.junction_strategy=junction_strategy
        self.connectedness=(FuzzyConnectedness3d(3,steps=24)
                            if degree==3 and junction_strategy=="fuzzy_reachability"
                            else None)

    def forward(self,vessel_probability:torch.Tensor,heatmap:torch.Tensor):
        separate=heatmap.ndim==5
        if vessel_probability.ndim!=4 or (
            (not separate and vessel_probability.shape!=heatmap.shape) or
            (separate and (heatmap.shape[0]!=vessel_probability.shape[0] or
                           heatmap.shape[1]!=3 or
                           heatmap.shape[2:]!=vessel_probability.shape[1:]))
        ):
            raise ValueError("Mismatched 3D vessel and summed/separate endpoint fields")
        if not torch.isfinite(vessel_probability).all() or not torch.isfinite(heatmap).all():
            raise ValueError("Nonfinite structural evidence")
        if separate and self.junction_strategy=="fuzzy_reachability":
            raise ValueError("Fuzzy connectivity diagnostic expects the legacy summed heatmap")
        if self.degree==3 and self.junction_strategy=="fuzzy_reachability":
            # Diagnostic only: raw component connectivity is NOT correct
            # branch identity. Oracle tests showed AUROC 0.5 for this task.
            return self.connectedness(vessel_probability,heatmap)
        peaks,valid=(
            _peak_centers_from_channels(heatmap,self.degree)
            if separate else
            _peak_centers(heatmap,self.degree)
        )
        if self.degree==3:
            if self.junction_strategy=="adaptive_star":
                paths=_junction_adaptive_paths(peaks,self.steps)
                bottleneck=_soft_min(
                    _sample_paths(vessel_probability,paths),self.temperature
                ).reshape(len(peaks),27,3)
                # All three arms must admit the *same* junction hub.
                bridge=bottleneck.min(dim=2).values.max(dim=1).values
            else:
                paths=_junction_paths(peaks,self.steps)
                bottleneck=_soft_min(_sample_paths(vessel_probability,paths),self.temperature)
                bridge=bottleneck.min(dim=1).values
        else:
            paths=_pair_paths(peaks,self.steps,self.curvature)
            bottleneck=_soft_min(_sample_paths(vessel_probability,paths),self.temperature)
            bridge=bottleneck.max(dim=1).values
        backup=(torch.full_like(bridge,1e-5) if separate else
                vessel_probability.flatten(1).mean(dim=1))
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
