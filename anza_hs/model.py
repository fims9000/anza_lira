"""One controlled backbone with B0--B3 decoder-local operator choices."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.segmentation import ResidualConvBlock, UpAttentionBlock

from .operators import ANZAHyperbolicConv, GenericAnisoConv, IsotropicOrientConv


VARIANTS = ("B0_backbone", "B1_isotropic", "B2_generic_aniso", "B3_anza_hyperbolic")


class ANZAHSUNet(nn.Module):
    def __init__(self, variant: str, *, widths: tuple[int, int, int, int] = (16, 32, 48, 64)) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown H1 variant: {variant}")
        self.variant = variant; w1, w2, w3, wb = widths
        self.enc1 = ResidualConvBlock(3, w1, use_se=True)
        self.enc2 = ResidualConvBlock(w1, w2, use_se=True)
        self.enc3 = ResidualConvBlock(w2, w3, use_se=True)
        self.bottleneck = ResidualConvBlock(w3, wb, use_se=True)
        self.pool = nn.MaxPool2d(2)
        self.up3 = UpAttentionBlock(wb, w3, w3)
        self.up2 = UpAttentionBlock(w3, w2, w2)
        self.up1 = UpAttentionBlock(w2, w1, w1)
        operator = {
            "B1_isotropic": IsotropicOrientConv,
            "B2_generic_aniso": GenericAnisoConv,
            "B3_anza_hyperbolic": ANZAHyperbolicConv,
        }.get(variant)
        self.bank_quarter = operator(w3) if operator else None
        self.bank_half = operator(w2) if operator else None
        self.head = nn.Conv2d(w1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, *, return_aux: bool = False) -> torch.Tensor | dict[str, Any]:
        x1 = self.enc1(x); x2 = self.enc2(self.pool(x1)); x3 = self.enc3(self.pool(x2))
        y = self.up3(self.bottleneck(self.pool(x3)), x3); orientation_logits = []
        if self.bank_quarter is not None:
            y, bank = self.bank_quarter(y); orientation_logits.append(bank)
        y = self.up2(y, x2)
        if self.bank_half is not None:
            y, bank = self.bank_half(y); orientation_logits.append(bank)
        y = self.up1(y, x1); visible = self.head(y)
        if not return_aux:
            return visible
        return {"visible_logits": visible, "orientation_logits": orientation_logits, "variant": self.variant}


def build_h1_model(variant: str) -> ANZAHSUNet:
    return ANZAHSUNet(variant)
