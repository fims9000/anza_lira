"""Experimental volumetric ANZA-style fuzzy anisotropic kernel operator.

This is a new 3D implementation, *not* a claim of Anosov chaos or ergodicity.
Each local direction u and hyperbolicity h parameterizes a unit-determinant
positive definite metric

  G(u,h) = exp(2h) uu^T + exp(-h)(I - uu^T),  det(G) = 1.

For center p and voxel-offset neighbor q, the unnormalized pair kernel is

  K_r(p,q) = mu_r(p) mu_r(q)
             exp(-beta * 0.5 * [delta^T G_r(p) delta
                                  + delta^T G_r(q) delta]).

K_r(p,q) equals K_r(q,p) when the spatial offset is reversed. This is
defined on 3D CT patch coordinates with an explicit voxel-to-world basis;
it does NOT prove a benefit of ANZA over an ordinary 3D CNN on CCTA.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AZConv3dConfig:
    modes: int = 3
    kernel_size: int = 3
    value_channels: int = 8
    beta: float = 0.12
    max_hyperbolicity: float = 1.0
    fuzzy_temperature: float = 1.0
    anisotropy: bool = True
    fuzzy: bool = True
    axial_preference: bool = False  # opt-in: penalize cross-axis transport more than along-axis

    def __post_init__(self):
        if self.modes < 1 or self.kernel_size < 1 or self.kernel_size % 2 != 1:
            raise ValueError("Expected positive mode count and odd kernel")
        if self.value_channels < 1:
            raise ValueError("value_channels must be positive")
        if not all(math.isfinite(x) and x > 0 for x in (
            self.beta, self.max_hyperbolicity, self.fuzzy_temperature
        )):
            raise ValueError("Kernel energy and gating scales must be positive and finite")


def quadratic_metric(
    direction: torch.Tensor,
    hyperbolicity: torch.Tensor,
    delta_world: torch.Tensor,
) -> torch.Tensor:
    """Compute delta^T G delta, with 3D direction on axis=2.

    direction: [B,R,3,D,H,W], hyperbolicity: [B,R,D,H,W]
    delta_world: world-coordinate 3-vector.
    """
    if direction.ndim != 6 or direction.shape[2] != 3:
        raise ValueError("Direction must have 3 world-coordinate components")
    if hyperbolicity.shape != direction.shape[:2] + direction.shape[3:]:
        raise ValueError("Hyperbolicity and direction spatial shapes disagree")
    if delta_world.shape != (3,):
        raise ValueError("delta_world must be a physical 3-vector")
    offset=delta_world.to(dtype=direction.dtype,device=direction.device).reshape(1,1,3,1,1,1)
    norm2=(offset*offset).sum(dim=2)
    axis_projection=(direction*offset).sum(dim=2)
    return (
        torch.exp(2.0*hyperbolicity)*axis_projection.square()
        + torch.exp(-hyperbolicity)*(
            norm2-axis_projection.square()
        ).clamp_min(0.0)
    )


def pair_kernel(
    center_membership: torch.Tensor,
    neighbor_membership: torch.Tensor,
    center_direction: torch.Tensor,
    neighbor_direction: torch.Tensor,
    center_hyperbolicity: torch.Tensor,
    neighbor_hyperbolicity: torch.Tensor,
    offset_world: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive")
    q_center=quadratic_metric(
        center_direction,center_hyperbolicity,offset_world
    )
    q_neighbor=quadratic_metric(
        neighbor_direction,neighbor_hyperbolicity,-offset_world
    )
    return (
        center_membership*neighbor_membership*
        torch.exp(-0.5*float(beta)*(q_center+q_neighbor))
    )


class AZConv3d(nn.Module):
    """3D physical metric and fuzzy membership-based local aggregation.

    Tensor input [batch,channels,depth,height,width] -> same 3D spatial grid.
    The voxel_world_basis is the 3x3 affine Jacobian (millimeters per voxel).
    """
    def __init__(
        self,in_channels:int,out_channels:int,
        cfg:AZConv3dConfig|None=None,
    ):
        super().__init__()
        self.cfg=cfg or AZConv3dConfig()
        if in_channels<1 or out_channels<1:
            raise ValueError("Channel counts must be positive")
        r=self.cfg.modes
        v=self.cfg.value_channels
        self.membership=nn.Conv3d(in_channels,r,1)
        self.orientation=nn.Conv3d(in_channels,3*r,1)
        self.hyperbolicity=nn.Conv3d(in_channels,r,1)
        self.values=nn.Conv3d(in_channels,v,1,bias=False)
        self.mix=nn.Conv3d(r*v,out_channels,1)
        self.shortcut=(
            nn.Identity() if in_channels==out_channels else
            nn.Conv3d(in_channels,out_channels,1,bias=False)
        )
        self.residual_logit=nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,x:torch.Tensor,
        voxel_world_basis:torch.Tensor|None=None,
    )->torch.Tensor:
        if x.ndim!=5:
            raise ValueError("AZConv3d expects B,C,D,H,W")
        if not torch.isfinite(x).all():
            raise ValueError("AZConv3d input has nonfinite values")
        b,c,d,h,w=x.shape
        basis=(
            torch.eye(3,device=x.device,dtype=x.dtype)
            if voxel_world_basis is None else
            torch.as_tensor(voxel_world_basis,device=x.device,dtype=x.dtype)
        )
        if basis.shape!=(3,3) or not torch.isfinite(basis).all():
            raise ValueError("Expected finite 3x3 voxel-to-world Jacobian")
        if torch.abs(torch.linalg.det(basis))<1e-12:
            raise ValueError("Singular voxel-to-world Jacobian")
        modes=self.cfg.modes
        if self.cfg.fuzzy:
            mu=torch.softmax(
                self.membership(x)/self.cfg.fuzzy_temperature,dim=1
            )
        else:
            mu=x.new_full((b,modes,d,h,w),1.0/modes)
        raw_direction=self.orientation(x).reshape(b,modes,3,d,h,w)
        direction_norm=torch.linalg.vector_norm(raw_direction,dim=2,keepdim=True)
        fallback=torch.zeros_like(raw_direction)
        fallback[:,:,0]=1.0
        direction=torch.where(
            direction_norm>1e-8,
            raw_direction/direction_norm.clamp_min(1e-8),
            fallback,
        )
        if self.cfg.anisotropy:
            if self.cfg.axial_preference:
                # G's axial eigenvalue is exp(2h), transverse is exp(-h).
                # Negative h lowers cost ALONG u and raises it across u.
                # It is an inductive bias, not a proof of topology preservation.
                hyper=-self.cfg.max_hyperbolicity*torch.sigmoid(self.hyperbolicity(x))
            else:
                # Preserve the historical ANZA3D operator bit-for-bit.
                hyper=self.cfg.max_hyperbolicity*torch.tanh(self.hyperbolicity(x))
        else:
            hyper=x.new_zeros((b,modes,d,h,w))
        projected=self.values(x)
        radius=self.cfg.kernel_size//2
        # Boundary entries get zero membership. Center always exists, so
        # normalization retains a positive self-neighbor contribution.
        pad=(radius,radius,radius,radius,radius,radius)
        mu_pad=F.pad(mu,pad)
        direction_pad=F.pad(direction,pad)
        hyper_pad=F.pad(hyper,pad)
        projected_pad=F.pad(projected,pad)
        numerator=x.new_zeros((b,modes,self.cfg.value_channels,d,h,w))
        denominator=x.new_zeros((b,modes,d,h,w))
        for offset in product(range(-radius,radius+1),repeat=3):
            z,y,t=(radius+o for o in offset)
            neighbor_mu=mu_pad[:,:,z:z+d,y:y+h,t:t+w]
            neighbor_direction=direction_pad[:,:,:,z:z+d,y:y+h,t:t+w]
            neighbor_hyper=hyper_pad[:,:,z:z+d,y:y+h,t:t+w]
            neighbor_values=projected_pad[:,:,z:z+d,y:y+h,t:t+w]
            world=basis@x.new_tensor(offset)
            compatibility=pair_kernel(
                mu,neighbor_mu,direction,neighbor_direction,
                hyper,neighbor_hyper,world,beta=self.cfg.beta,
            )
            denominator=denominator+compatibility
            numerator=numerator+compatibility.unsqueeze(2)*neighbor_values.unsqueeze(1)
        aggregated=(numerator/denominator.clamp_min(1e-12).unsqueeze(2)).reshape(
            b,modes*self.cfg.value_channels,d,h,w
        )
        return self.shortcut(x)+torch.sigmoid(self.residual_logit)*self.mix(aggregated)


def count_trainable(module:nn.Module)->int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
