"""Capacity-matched Generic/B3 SPD metric heads on historical T1 decoder features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from models.segmentation_v2 import ComparableStructuralUNet, build_comparable_model


def normalize_double_angle(raw_cs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if raw_cs.shape[-3] != 2:
        raise ValueError("double-angle tensor must have two channels")
    return raw_cs / torch.sqrt(torch.sum(raw_cs * raw_cs, dim=-3, keepdim=True) + eps)


def metric_from_axial(c2: torch.Tensor, s2: torch.Tensor, d: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Build (...,2,2,H,W) SPD metric from axial doubled angle and bounded d/m."""
    theta = 0.5 * torch.atan2(s2, c2)
    cosine, sine = torch.cos(theta), torch.sin(theta)
    lambda_u = torch.exp(2.0 * (m + d))
    lambda_s = torch.exp(2.0 * (m - d))
    c00 = lambda_u * cosine.square() + lambda_s * sine.square()
    c01 = (lambda_u - lambda_s) * cosine * sine
    c11 = lambda_u * sine.square() + lambda_s * cosine.square()
    row0 = torch.stack((c00, c01), dim=1)
    row1 = torch.stack((c01, c11), dim=1)
    return torch.stack((row0, row1), dim=1)


@dataclass(frozen=True)
class GeometryOutput:
    c2: torch.Tensor
    s2: torch.Tensor
    d: torch.Tensor
    m: torch.Tensor
    metric: torch.Tensor


class GeometrySidecar(nn.Module):
    """3x3 Conv-BN-ReLU and a 1x1 geometry output, shared form for B2/B3."""

    def __init__(self, in_channels: int, variant: str, hidden: int = 16) -> None:
        super().__init__()
        if variant not in {"B2", "B3"}:
            raise ValueError("geometry sidecar exists only for B2/B3")
        self.variant = variant
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Conv2d(hidden, 4 if variant == "B2" else 3, 1)

    def forward(self, feature: torch.Tensor) -> GeometryOutput:
        raw = self.output(self.features(feature))
        cs = normalize_double_angle(raw[:, :2])
        d = 0.5 * torch.sigmoid(raw[:, 2:3])
        m = 0.5 * torch.tanh(raw[:, 3:4]) if self.variant == "B2" else torch.zeros_like(d)
        metric = metric_from_axial(cs[:, 0], cs[:, 1], d[:, 0], m[:, 0])
        return GeometryOutput(cs[:, 0], cs[:, 1], d[:, 0], m[:, 0], metric)


class V11StructuralModel(nn.Module):
    """Exact historical T1 U-Net topology with optional decoder-only sidecars."""

    def __init__(self, variant: str) -> None:
        super().__init__()
        if variant not in {"B0", "B1", "B2", "B3"}:
            raise ValueError(f"unknown V1.1 variant: {variant}")
        self.variant = variant
        self.backbone: ComparableStructuralUNet = build_comparable_model("unet")
        if variant in {"B2", "B3"}:
            self.geometry_quarter = GeometrySidecar(64, variant)
            self.geometry_half = GeometrySidecar(32, variant)
        else:
            self.geometry_quarter = None
            self.geometry_half = None

    def forward(self, x: torch.Tensor, *, return_geometry: bool = False) -> torch.Tensor | dict[str, Any]:
        backbone = self.backbone
        x1 = backbone.enc1(x)
        x2 = backbone.enc2(backbone.pool(x1))
        x3 = backbone.enc3(backbone.pool(x2))
        bottleneck = backbone.bottleneck(backbone.pool(x3))
        quarter = backbone.up3(bottleneck, x3)
        half = backbone.up2(quarter, x2)
        full = backbone.up1(half, x1)
        logits = backbone.visible_head(full)
        if not return_geometry:
            return logits
        geometry = []
        if self.geometry_quarter is not None and self.geometry_half is not None:
            geometry = [self.geometry_quarter(quarter), self.geometry_half(half)]
        return {"visible_logits": logits, "geometry": geometry, "variant": self.variant}


def parameter_audit() -> dict[str, Any]:
    models = {variant: V11StructuralModel(variant) for variant in ("B0", "B1", "B2", "B3")}
    counts = {variant: sum(parameter.numel() for parameter in model.parameters()) for variant, model in models.items()}
    backbone = sum(parameter.numel() for parameter in models["B0"].backbone.parameters())
    sidecars = {variant: counts[variant] - backbone for variant in ("B2", "B3")}
    relative = abs(counts["B2"] - counts["B3"]) / counts["B3"]
    return {"total_parameters": counts, "backbone_parameters": backbone, "sidecar_parameters": sidecars, "B2_B3_relative_difference": relative, "passes_one_percent": relative < 0.01}
