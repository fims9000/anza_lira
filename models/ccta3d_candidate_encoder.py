"""Controlled ANZA-3D vs compact depthwise-CNN candidate encoders.

Architecture and training are research prototypes, NOT replacements for the
frozen CT28 K10/15 HGB candidate score streams. Use the same patch builder,
patient split, optimizer and evaluation for both comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .azconv3d import AZConv3d, AZConv3dConfig, count_trainable


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
    ):
        super().__init__()
        if variant not in {"anza3d","conv3d"}:
            raise ValueError("variant must be anza3d or conv3d")
        if channels<3:
            raise ValueError("channels must be >=3")
        self.variant=variant
        self.stem=nn.Sequential(
            nn.Conv3d(2,channels,kernel_size=3,padding=1,bias=False),
            nn.GroupNorm(1,channels),nn.GELU(),
        )
        if variant=="anza3d":
            self.local=AZConv3d(
                channels,channels,
                cfg or AZConv3dConfig(
                    modes=3,value_channels=6,kernel_size=3,
                    beta=.12,max_hyperbolicity=1.,
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

    def forward(self,x:torch.Tensor)->torch.Tensor:
        if x.ndim!=5 or x.shape[1]!=2:
            raise ValueError("Expected 2-channel CCTA/endpoint 3D patch")
        h=self.stem(x)
        h=self.local(h)
        return self.head(h).flatten()

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
