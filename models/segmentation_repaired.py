"""Comparable U-Net with one repaired residual router at encoder stage 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .azconv_repaired import AmbiguityGatedResidualANZA, RepairedAZConvConfig
from .segmentation_v2 import ComparableStructuralUNet, ComparableUNetConfig, _ComparableBlock, _UpBlock


class _RepairedBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_modes: int,
        repaired_cfg: RepairedAZConvConfig,
    ) -> None:
        super().__init__()
        self.spatial = AmbiguityGatedResidualANZA(
            in_channels,
            out_channels,
            num_rules=num_modes,
            cfg=repaired_cfg,
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.refine = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.last_diagnostics: dict[str, Any] | None = None

    def forward(self, x: torch.Tensor, *, collect_diagnostics: bool = False) -> torch.Tensor:
        if collect_diagnostics:
            self.last_diagnostics = self.spatial(x, return_diagnostics=True)
            x = self.last_diagnostics["output"]
        else:
            self.last_diagnostics = None
            x = self.spatial(x)
        x = F.relu(self.norm1(x), inplace=False)
        return F.relu(self.norm2(self.refine(x)), inplace=False)


@dataclass(frozen=True)
class RepairedUNetConfig:
    widths: tuple[int, int, int, int] = (16, 32, 64, 96)
    num_modes: int = 4
    routing_kernel_size: int = 5
    use_ambiguity_gate: bool = True
    transport_steps: int = 1


class RepairedStructuralUNet(nn.Module):
    """The v1 U-Net topology with one enc1 residual correction branch."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        cfg: RepairedUNetConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or RepairedUNetConfig()
        w1, w2, w3, wb = self.cfg.widths
        repaired_cfg = RepairedAZConvConfig(
            num_modes=self.cfg.num_modes,
            routing_kernel_size=self.cfg.routing_kernel_size,
            use_ambiguity_gate=self.cfg.use_ambiguity_gate,
            transport_steps=self.cfg.transport_steps,
        )
        self.enc1 = _RepairedBlock(
            in_channels,
            w1,
            num_modes=self.cfg.num_modes,
            repaired_cfg=repaired_cfg,
        )
        kwargs = {"operator": "v1", "num_modes": self.cfg.num_modes, "v2_cfg": None}
        self.enc2 = _ComparableBlock(w1, w2, **kwargs)
        self.enc3 = _ComparableBlock(w2, w3, **kwargs)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(w3, wb, 3, padding=1, bias=False),
            nn.BatchNorm2d(wb),
            nn.ReLU(inplace=True),
            nn.Conv2d(wb, wb, 3, padding=1, bias=False),
            nn.BatchNorm2d(wb),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)
        self.up3 = _UpBlock(wb, w3, w3)
        self.up2 = _UpBlock(w3, w2, w2)
        self.up1 = _UpBlock(w2, w1, w1)
        self.visible_head = nn.Conv2d(w1, out_channels, 1)

    def forward(self, x: torch.Tensor, *, return_diagnostics: bool = False) -> torch.Tensor | dict[str, Any]:
        x1 = self.enc1(x, collect_diagnostics=return_diagnostics)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        bottleneck = self.bottleneck(self.pool(x3))
        y = self.up3(bottleneck, x3)
        y = self.up2(y, x2)
        y = self.up1(y, x1)
        visible_logits = self.visible_head(y)
        if not return_diagnostics:
            return visible_logits
        if self.enc1.last_diagnostics is None:
            raise AssertionError("repaired enc1 diagnostics were not collected")
        return {
            "visible_logits": visible_logits,
            "transport_diagnostics": [self.enc1.last_diagnostics],
            "operator": "ambiguity_gated_residual_anza",
        }


def initialize_from_v1(
    repaired: RepairedStructuralUNet,
    v1: ComparableStructuralUNet,
) -> None:
    """Copy the full v1 segmentation path while leaving new router weights intact."""
    if v1.cfg.operator != "v1" or repaired.cfg.widths != v1.cfg.widths:
        raise ValueError("initialization requires matching v1 topology and widths")
    repaired.enc1.spatial.base.load_state_dict(v1.enc1.spatial.state_dict())
    for name in ("norm1", "refine", "norm2"):
        getattr(repaired.enc1, name).load_state_dict(getattr(v1.enc1, name).state_dict())
    for name in ("enc2", "enc3", "bottleneck", "up3", "up2", "up1", "visible_head"):
        getattr(repaired, name).load_state_dict(getattr(v1, name).state_dict())


def build_repaired_model(
    *,
    widths: tuple[int, int, int, int] = (16, 32, 64, 96),
    routing_kernel_size: int = 5,
    use_ambiguity_gate: bool = True,
    seed_matched_v1: ComparableStructuralUNet | None = None,
) -> RepairedStructuralUNet:
    model = RepairedStructuralUNet(
        cfg=RepairedUNetConfig(
            widths=widths,
            routing_kernel_size=routing_kernel_size,
            use_ambiguity_gate=use_ambiguity_gate,
        )
    )
    if seed_matched_v1 is not None:
        initialize_from_v1(model, seed_matched_v1)
    return model
