"""Capacity-matched seed-41 K2 segmentation matrix."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.segmentation import ResidualConvBlock, UpAttentionBlock

from .block import SymbolicFeatureBlock


VARIANTS = ("M0_backbone", "M1_static", "M2_shear_ks", "M3_cat_raw", "M4_anza_ks")
METHOD = {"M1_static": "static", "M2_shear_ks": "shear_ks", "M3_cat_raw": "cat_raw", "M4_anza_ks": "cat_ks"}


class ANZAKSUNet(nn.Module):
    def __init__(self, variant: str, *, feature_norm: dict[str, torch.Tensor] | None = None, widths: tuple[int, int, int, int] = (16, 32, 48, 64)) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown K2 variant: {variant}")
        self.variant = variant
        w1, w2, w3, wb = widths
        self.enc1 = ResidualConvBlock(3, w1, use_se=True)
        self.enc2 = ResidualConvBlock(w1, w2, use_se=True)
        self.enc3 = ResidualConvBlock(w2, w3, use_se=True)
        self.bottleneck = ResidualConvBlock(w3, wb, use_se=True)
        self.pool = nn.MaxPool2d(2)
        self.up3 = UpAttentionBlock(wb, w3, w3)
        self.up2 = UpAttentionBlock(w3, w2, w2)
        self.up1 = UpAttentionBlock(w2, w1, w1)
        if variant == "M0_backbone":
            self.symbolic = None
        else:
            method = METHOD[variant]
            norm = feature_norm or {}
            self.symbolic = SymbolicFeatureBlock(w3, method, feature_mean=norm.get("mean"), feature_std=norm.get("std"))
        self.head = nn.Conv2d(w1, 1, 1)

    def forward(self, x: torch.Tensor, *, return_aux: bool = False) -> torch.Tensor | dict[str, Any]:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        y = self.up3(self.bottleneck(self.pool(x3)), x3)
        aux: dict[str, Any] = {}
        if self.symbolic is not None:
            y, aux = self.symbolic(y, return_aux=True)
        y = self.up2(y, x2)
        y = self.up1(y, x1)
        logits = self.head(y)
        if not return_aux:
            return logits
        return {"visible_logits": logits, "symbolic_aux": aux, "variant": self.variant}


def build_k2_model(variant: str, feature_norm: dict[str, torch.Tensor] | None = None) -> ANZAKSUNet:
    return ANZAKSUNet(variant, feature_norm=feature_norm)
