"""Matched L0--L3 model matrix importing the frozen ANZA-HS operators."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from anza_hs.operators import ANZAHyperbolicConv, GenericAnisoConv, IsotropicOrientConv
from models.segmentation import ResidualConvBlock, UpAttentionBlock


LEADS_VARIANTS = ("L0_backbone", "L1_isotropic", "L2_generic_aniso", "L3_anza_hs")
WIDTHS = (16, 32, 48, 64)


class ANZALEADSUNet(nn.Module):
    """Exact H1 backbone/placements with equal orientation supervision."""

    def __init__(self, variant: str, *, widths: tuple[int, int, int, int] = WIDTHS) -> None:
        super().__init__()
        if variant not in LEADS_VARIANTS:
            raise ValueError(f"unknown LEADS variant: {variant}")
        self.variant = variant
        self.widths = tuple(int(value) for value in widths)
        w1, w2, w3, wb = self.widths
        self.enc1 = ResidualConvBlock(3, w1, use_se=True)
        self.enc2 = ResidualConvBlock(w1, w2, use_se=True)
        self.enc3 = ResidualConvBlock(w2, w3, use_se=True)
        self.bottleneck = ResidualConvBlock(w3, wb, use_se=True)
        self.pool = nn.MaxPool2d(2)
        self.up3 = UpAttentionBlock(wb, w3, w3)
        self.up2 = UpAttentionBlock(w3, w2, w2)
        self.up1 = UpAttentionBlock(w2, w1, w1)
        operator = {
            "L1_isotropic": IsotropicOrientConv,
            "L2_generic_aniso": GenericAnisoConv,
            "L3_anza_hs": ANZAHyperbolicConv,
        }.get(variant)
        self.bank_quarter = operator(w3) if operator else None
        self.bank_half = operator(w2) if operator else None
        # L0 gets the same auxiliary targets at the same stages. These heads are
        # discarded at inference and do not alter the segmentation path.
        self.aux_quarter = nn.Conv2d(w3, 8, kernel_size=3, padding=1) if operator is None else None
        self.aux_half = nn.Conv2d(w2, 8, kernel_size=3, padding=1) if operator is None else None
        self.head = nn.Conv2d(w1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, *, return_aux: bool = False) -> torch.Tensor | dict[str, Any]:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        y = self.up3(self.bottleneck(self.pool(x3)), x3)
        orientation_logits: list[torch.Tensor] = []
        if self.bank_quarter is not None:
            y, logits = self.bank_quarter(y)
            orientation_logits.append(logits)
        else:
            orientation_logits.append(self.aux_quarter(y))
        y = self.up2(y, x2)
        if self.bank_half is not None:
            y, logits = self.bank_half(y)
            orientation_logits.append(logits)
        else:
            orientation_logits.append(self.aux_half(y))
        y = self.up1(y, x1)
        visible = self.head(y)
        if not return_aux:
            return visible
        return {"visible_logits": visible, "orientation_logits": orientation_logits, "variant": self.variant}


def build_leads_model(variant: str) -> ANZALEADSUNet:
    return ANZALEADSUNet(variant)
