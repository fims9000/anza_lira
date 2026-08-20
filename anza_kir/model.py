"""Common IR1 base and frozen-backbone IR2 model matrix."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.segmentation import ResidualConvBlock, UpAttentionBlock

from .block import InnovationResidual, VARIANT_METHOD


KIR_VARIANTS = tuple(VARIANT_METHOD)


class KIRBaseUNet(nn.Module):
    def __init__(self, widths: tuple[int, int, int, int] = (16, 32, 48, 64)) -> None:
        super().__init__()
        w1, w2, w3, wb = widths
        self.enc1 = ResidualConvBlock(3, w1, use_se=True)
        self.enc2 = ResidualConvBlock(w1, w2, use_se=True)
        self.enc3 = ResidualConvBlock(w2, w3, use_se=True)
        self.bottleneck = ResidualConvBlock(w3, wb, use_se=True)
        self.pool = nn.MaxPool2d(2)
        self.up3 = UpAttentionBlock(wb, w3, w3)
        self.up2 = UpAttentionBlock(w3, w2, w2)
        self.up1 = UpAttentionBlock(w2, w1, w1)
        self.head = nn.Conv2d(w1, 1, 1)
        self.evidence_head = nn.Conv2d(w3, 1, 1)
        self.orientation_head = nn.Conv2d(w3, 8, 1)

    def encode_to_quarter(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        y = self.up3(self.bottleneck(self.pool(x3)), x3)
        return x1, x2, y

    def decode_from_quarter(self, x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = self.up2(y, x2)
        y = self.up1(y, x1)
        return self.head(y)

    def forward(self, x: torch.Tensor, *, return_aux: bool = False) -> torch.Tensor | dict[str, Any]:
        x1, x2, y = self.encode_to_quarter(x)
        evidence_logits = self.evidence_head(y)
        orientation_logits = self.orientation_head(y)
        logits = self.decode_from_quarter(x1, x2, y)
        if not return_aux:
            return logits
        return {
            "visible_logits": logits,
            "evidence_logits": evidence_logits,
            "evidence_probability": torch.sigmoid(evidence_logits),
            "orientation_logits": orientation_logits,
            "quarter_features": y,
        }


class FrozenBackboneKIR(nn.Module):
    def __init__(self, base: KIRBaseUNet, variant: str, feature_norm: dict[str, dict[str, torch.Tensor]]) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()
        self.variant = variant
        self.residual = InnovationResidual(48, variant, feature_norm=feature_norm)

    def train(self, mode: bool = True) -> "FrozenBackboneKIR":
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, x: torch.Tensor, *, return_aux: bool = False) -> torch.Tensor | dict[str, Any]:
        with torch.no_grad():
            x1, x2, y = self.base.encode_to_quarter(x)
            evidence_logits = self.base.evidence_head(y)
            evidence_probability = torch.sigmoid(evidence_logits)
            orientation_logits = self.base.orientation_head(y)
        corrected, residual_aux = self.residual(
            y.detach(), evidence_probability.detach(), orientation_logits.detach(), return_aux=True
        )
        logits = self.base.decode_from_quarter(x1.detach(), x2.detach(), corrected)
        if not return_aux:
            return logits
        return {
            "visible_logits": logits,
            "evidence_logits": evidence_logits,
            "evidence_probability": evidence_probability,
            "orientation_logits": orientation_logits,
            "residual_aux": residual_aux,
        }


def build_base_model() -> KIRBaseUNet:
    return KIRBaseUNet()


def build_kir_model(
    variant: str,
    base_state: dict[str, torch.Tensor],
    feature_norm: dict[str, dict[str, torch.Tensor]],
) -> FrozenBackboneKIR:
    base = build_base_model()
    base.load_state_dict(base_state)
    return FrozenBackboneKIR(base, variant, feature_norm)
