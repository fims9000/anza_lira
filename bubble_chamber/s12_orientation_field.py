"""Differentiable direct supervision of original AZConv local orientation fields.

Designed to address the very weak geometry gradient from the original
AZConv2d.axis_alignment_loss on thin simulated bubble-chamber tracks. The
reference orientation is inferred from a structure tensor of a *known dense*
synthetic target. It must NOT be used on sparse H50Z keypoints/unknown pixels.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def reference_orientation(target:torch.Tensor):
    """Return sign-invariant (cos 2theta, sin 2theta) and reliability weights.

    For thin lines the gradient normal is perpendicular to the track tangent.
    The tensor representation avoids selecting an arbitrary + or - direction.
    """
    if target.ndim!=4 or target.shape[1]!=1:
        raise ValueError('Expected one-channel dense binary/soft target [B,1,H,W]')
    if target.shape[-1]<8 or target.shape[-2]<8:
        raise ValueError('Target spatial grid too small for tangent estimation')
    with torch.no_grad():
        z=target.to(dtype=torch.float32)
        z=F.avg_pool2d(z,3,stride=1,padding=1)
        kx=z.new_tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]).view(1,1,3,3)/8
        ky=z.new_tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]]).view(1,1,3,3)/8
        gx=F.conv2d(z,kx,padding=1);gy=F.conv2d(z,ky,padding=1)
        jxx=F.avg_pool2d(gx.square(),5,stride=1,padding=2)
        jyy=F.avg_pool2d(gy.square(),5,stride=1,padding=2)
        jxy=F.avg_pool2d(gx*gy,5,stride=1,padding=2)
        contrast=torch.sqrt((jxx-jyy).square()+4*jxy.square()+1e-10)
        trace=jxx+jyy
        # Dominant eigenvector is the image-gradient normal. Shift by 90deg
        # to describe the track tangent: negate the doubled-angle vector.
        cos2=(jyy-jxx)/contrast
        sin2=(-2*jxy)/contrast
        coherence=(contrast/(trace+1e-7)).clamp(0,1)
        strength=(trace/(trace.amax(dim=(-2,-1),keepdim=True)+1e-7)).clamp(0,1)
        band=F.max_pool2d(target.float(),3,stride=1,padding=1)
        reliability=(band*coherence*strength.sqrt()).clamp(0,1)
    return cos2,sin2,reliability


def az_orientation_field_loss(model:torch.nn.Module,target:torch.Tensor,
                               *,min_weight:float=1e-5)->tuple[torch.Tensor,dict]:
    """Use live `mu`, `theta_map` from actual AZConv2d stages (no kernel edits)."""
    from models.azconv import AZConv2d
    layers=[m for m in model.modules() if isinstance(m,AZConv2d)]
    if not layers:
        raise ValueError('Authentic AZConv stages not present')
    losses=[];weights=[]
    for layer in layers:
        live=layer._last_live_interpretation
        mu=live.get('mu');theta=live.get('theta_map')
        if mu is None or theta is None:
            raise ValueError('Must call original AZConv forward before orientation loss')
        size=theta.shape[-2:]
        t=F.interpolate(target.float(),size=size,mode='area')
        cgt,sgt,w=reference_orientation(t)
        c=(mu*torch.cos(2*theta)).sum(1,keepdim=True)
        s=(mu*torch.sin(2*theta)).sum(1,keepdim=True)
        # Unit-length MSE in double-angle space provides stable gradients,
        # even when competing AZ rules make the mixed direction ambiguous.
        err=(c-cgt).square()+(s-sgt).square()
        denom=w.sum()
        if float(denom.detach())<=min_weight:
            continue
        losses.append((err*w).sum()/denom)
        weights.append(float(denom.detach()))
    if not losses:
        return target.new_zeros(()),{'valid_stages':0,'mean_reference_weight':0.}
    return torch.stack(losses).mean(),{'valid_stages':len(losses),
           'mean_reference_weight':float(sum(weights)/len(weights))}
