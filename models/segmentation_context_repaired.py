"""Comparable U-Net containing one context-gated residual ANZA block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .azconv_context_repaired import ContextGatedResidualANZA
from .azconv_repaired import RepairedAZConvConfig
from .segmentation_repaired import initialize_from_v1
from .segmentation_v2 import ComparableStructuralUNet, _ComparableBlock, _UpBlock


class _ContextBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, num_modes: int) -> None:
        super().__init__()
        self.spatial = ContextGatedResidualANZA(
            in_channels,
            out_channels,
            num_rules=num_modes,
            cfg=RepairedAZConvConfig(
                num_modes=num_modes,
                routing_kernel_size=3,
                use_ambiguity_gate=True,
            ),
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
class ContextRepairedUNetConfig:
    widths: tuple[int, int, int, int] = (16, 32, 64, 96)
    num_modes: int = 4


class ContextRepairedStructuralUNet(nn.Module):
    def __init__(self, cfg: ContextRepairedUNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ContextRepairedUNetConfig()
        w1, w2, w3, wb = self.cfg.widths
        self.enc1 = _ContextBlock(3, w1, num_modes=self.cfg.num_modes)
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
        self.visible_head = nn.Conv2d(w1, 1, 1)

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
            raise AssertionError("context diagnostics were not collected")
        return {
            "visible_logits": visible_logits,
            "transport_diagnostics": [self.enc1.last_diagnostics],
            "operator": "context_gated_residual_anza",
        }


def build_context_repaired_model(
    *,
    widths: tuple[int, int, int, int] = (16, 32, 64, 96),
    seed_matched_v1: ComparableStructuralUNet | None = None,
) -> ContextRepairedStructuralUNet:
    model = ContextRepairedStructuralUNet(ContextRepairedUNetConfig(widths=widths))
    if seed_matched_v1 is not None:
        # The initializer is structural and intentionally works for both
        # residual blocks: their v1/base and U-Net paths share the same names.
        initialize_from_v1(model, seed_matched_v1)  # type: ignore[arg-type]
    return model
