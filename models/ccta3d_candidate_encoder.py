"""Controlled ANZA-3D vs compact depthwise-CNN candidate encoders.

Architecture and training are research prototypes, NOT replacements for the
frozen CT28 K10/15 HGB candidate score streams. Use the same patch builder,
patient split, optimizer and evaluation for both comparisons.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .azconv3d import AZConv3d, AZConv3dConfig, count_trainable
from .structural_bridge3d import VesselBridgeReadout


class PermutationInvariantTipStem3d(nn.Module):
    """One HU projection plus the SAME convolution on each endpoint channel.

    The sum across tips makes the network invariant to their storage order:
    no endpoint-ID/coordinate sorting shortcut can enter via distinct weights.
    PAIR carries a zero third tip channel and is normalized by degree=2.
    """
    def __init__(self,channels:int,degree:int):
        super().__init__()
        if degree not in (2,3):
            raise ValueError("Unsupported candidate degree")
        self.degree=degree
        self.ct=nn.Conv3d(1,channels,3,padding=1,bias=False)
        self.tip=nn.Conv3d(1,channels,3,padding=1,bias=False)
        self.norm=nn.GroupNorm(1,channels)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        if x.ndim!=5 or x.shape[1]!=4:
            raise ValueError("Separate-tip stem requires HU + three endpoints")
        b,_,d,h,w=x.shape
        tips=x[:,1:].reshape(b*3,1,d,h,w)
        maps=self.tip(tips).reshape(b,3,-1,d,h,w)
        encoded=(maps.sum(dim=1)/self.degree)
        return F.gelu(self.norm(self.ct(x[:,:1])+encoded))


class CompactConv3d(nn.Module):
    """Depthwise 3x3x3 + pointwise 1x1 with residual, matched order of size."""
    def __init__(self, channels: int):
        super().__init__()
        self.depthwise=nn.Conv3d(
            channels,channels,kernel_size=3,padding=1,
            groups=channels,bias=True,
        )
        self.pointwise=nn.Conv3d(channels,channels,kernel_size=1)
        self.norm=nn.GroupNorm(1,channels)
        self.residual_logit=nn.Parameter(torch.tensor(-2.0))

    def forward(self,x):
        return x+torch.sigmoid(self.residual_logit)*F.gelu(
            self.norm(self.pointwise(self.depthwise(x)))
        )


class Candidate3dEncoder(nn.Module):
    """Score PAIR or JUNCTION using a normalized local CT+endpoint patch.

    Input [B,2,D,H,W]: channel 0 = clipped normalized CCTA HU, channel 1 =
    endpoint-identity-free Gaussian mask. Train separate type-specific heads.
    """
    def __init__(
        self,variant:str="anza3d",channels:int=12,
        cfg:AZConv3dConfig|None=None,
        voxel_mm:float=1.,
        candidate_type:str="PAIR",
    ):
        super().__init__()
        if variant not in {"anza3d","conv3d","anza3d_structure","conv3d_structure","anza3d_noaxial_structure",
                           "anza3d_sep_structure","conv3d_sep_structure","anza3d_noaxial_sep_structure"}:
            raise ValueError("Unknown candidate model variant")
        if candidate_type not in {"PAIR","JUNCTION"}:
            raise ValueError("Unknown candidate relation type")
        if channels<3:
            raise ValueError("channels must be >=3")
        if not math.isfinite(voxel_mm) or voxel_mm <= 0:
            raise ValueError("Physical patch voxel size must be positive finite mm")
        self.voxel_mm=float(voxel_mm)
        self.variant=variant
        self.candidate_type=candidate_type
        self.structure_aware=variant.endswith("_structure")
        self.separate_endpoints="_sep_" in variant
        self.stem=(
            PermutationInvariantTipStem3d(channels,2 if candidate_type=="PAIR" else 3)
            if self.separate_endpoints else
            nn.Sequential(
                nn.Conv3d(2,channels,kernel_size=3,padding=1,bias=False),
                nn.GroupNorm(1,channels),nn.GELU(),
            )
        )
        if variant.startswith("anza3d"):
            self.local=AZConv3d(
                channels,channels,
                cfg or AZConv3dConfig(
                    modes=3,value_channels=6,kernel_size=3,
                    beta=.12,max_hyperbolicity=1.,
                    axial_preference=(variant in {"anza3d_structure","anza3d_sep_structure"}),
                )
            )
        else:
            self.local=CompactConv3d(channels)
        self.head=nn.Sequential(
            nn.GroupNorm(1,channels),nn.GELU(),
            nn.AdaptiveAvgPool3d(1),nn.Flatten(),
            nn.Linear(channels,channels),nn.GELU(),
            nn.Linear(channels,1),
        )
        if self.structure_aware:
            # Identical dense supervision and geometric readout for matched CNN
            # and ANZA: only local operator/axial physical bias differs.
            self.vessel_head=nn.Conv3d(channels,1,1)
            self.bridge_readout=VesselBridgeReadout(
                2 if candidate_type=="PAIR" else 3,
                junction_strategy=("adaptive_star" if candidate_type=="JUNCTION" else "centroid_star"),
            )
            self.bridge_strength_logit=nn.Parameter(torch.tensor(-.5))

    def forward(self,x:torch.Tensor,*,return_structure:bool=False):
        expected_channels=4 if self.separate_endpoints else 2
        if x.ndim!=5 or x.shape[1]!=expected_channels:
            raise ValueError(f"Expected {expected_channels}-channel CCTA/endpoint 3D patch")
        if self.separate_endpoints and self.candidate_type=="PAIR" and not bool(
            (x[:,3].abs()<1e-7).all()
        ):
            raise ValueError("PAIR third endpoint channel must be empty")
        h=self.stem(x)
        if self.variant.startswith("anza3d"):
            # Resampled CT patches have an isotropic physical voxel grid
            # in the candidate-local frame, not arbitrary raw CT spacing.
            basis=torch.eye(3,device=h.device,dtype=h.dtype)*self.voxel_mm
            h=self.local(h,voxel_world_basis=basis)
        else:
            h=self.local(h)
        relation_logits=self.head(h).flatten()
        if not self.structure_aware:
            if return_structure:
                raise ValueError("Dense structural supervision is not enabled")
            return relation_logits
        vessel_logits=self.vessel_head(h)
        vessel_prob=torch.sigmoid(vessel_logits[:,0])
        path_support=self.bridge_readout(vessel_prob,x[:,1:] if self.separate_endpoints else x[:,1])
        # The positive coefficient makes more supported continuity raise the
        # candidate score. Dense mask supervision prevents a trivial all-ones
        # vessel field; raw classification BCE remains active.
        strength=F.softplus(self.bridge_strength_logit)
        relation_logits=relation_logits+strength*torch.logit(path_support)
        if return_structure:
            return relation_logits,vessel_logits,path_support
        return relation_logits

    def parameter_count(self)->int:
        return count_trainable(self)


def parameter_budget(anza:Candidate3dEncoder,conv:Candidate3dEncoder)->dict:
    az=anza.parameter_count()
    cnn=conv.parameter_count()
    return {
        "anza3d_trainable":az,
        "conv3d_trainable":cnn,
        "relative_gap":abs(az-cnn)/max(az,cnn),
        "matched_parameter_budget":abs(az-cnn)/max(az,cnn)<=0.20,
    }
