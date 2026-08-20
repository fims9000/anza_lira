"""Controlled C1--C3 U-Nets for the structural-affinity repair stream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .azconv import AZConvConfig
from .azconv_affinity import (
    AffinityAZConvConfig,
    IndependentFuzzyAZConv2d,
    StructuralAffinityAZConv2d,
)
from .segmentation_v2 import ComparableStructuralUNet, _UpBlock


class _AffinityStudyBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, affinity: bool, radius2: bool) -> None:
        super().__init__()
        cfg = AZConvConfig()
        if affinity:
            self.spatial: nn.Module = StructuralAffinityAZConv2d(
                in_channels,
                out_channels,
                num_rules=4,
                cfg=cfg,
                affinity_cfg=AffinityAZConvConfig(
                    context_channels=8,
                    use_radius2_supervision=radius2,
                ),
            )
        else:
            self.spatial = IndependentFuzzyAZConv2d(
                in_channels, out_channels, num_rules=4, cfg=cfg
            )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.refine = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.last_diagnostics: dict[str, Any] | None = None

    def forward(self, x: torch.Tensor, *, collect_diagnostics: bool = False) -> torch.Tensor:
        if collect_diagnostics:
            result = self.spatial(x, return_diagnostics=True)  # type: ignore[call-arg]
            self.last_diagnostics = result
            x = result["output"]
        else:
            self.last_diagnostics = None
            x = self.spatial(x)
        return F.relu(self.norm2(self.refine(F.relu(self.norm1(x), inplace=False))), inplace=False)


@dataclass(frozen=True)
class AffinityStructuralUNetConfig:
    widths: tuple[int, int, int, int] = (16, 32, 64, 96)
    use_affinity: bool = False
    use_radius2: bool = False


class AffinityStructuralUNet(nn.Module):
    def __init__(self, cfg: AffinityStructuralUNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or AffinityStructuralUNetConfig()
        w1, w2, w3, wb = self.cfg.widths
        self.enc1 = _AffinityStudyBlock(3, w1, affinity=self.cfg.use_affinity, radius2=self.cfg.use_radius2)
        self.enc2 = _AffinityStudyBlock(w1, w2, affinity=False, radius2=False)
        self.enc3 = _AffinityStudyBlock(w2, w3, affinity=False, radius2=False)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(w3, wb, 3, padding=1, bias=False), nn.BatchNorm2d(wb), nn.ReLU(inplace=True),
            nn.Conv2d(wb, wb, 3, padding=1, bias=False), nn.BatchNorm2d(wb), nn.ReLU(inplace=True),
        )
        self.up3 = _UpBlock(wb, w3, w3)
        self.up2 = _UpBlock(w3, w2, w2)
        self.up1 = _UpBlock(w2, w1, w1)
        self.visible_head = nn.Conv2d(w1, 1, 1)

    def forward(self, x: torch.Tensor, *, return_diagnostics: bool = False) -> torch.Tensor | dict[str, Any]:
        x1 = self.enc1(x, collect_diagnostics=return_diagnostics)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        y = self.up3(self.bottleneck(self.pool(x3)), x3)
        y = self.up2(y, x2)
        logits = self.visible_head(self.up1(y, x1))
        if not return_diagnostics:
            return logits
        return {
            "visible_logits": logits,
            "affinity_diagnostics": self.enc1.last_diagnostics,
            "operator": "structural_affinity_anza" if self.cfg.use_affinity else "independent_fuzzy_anza",
        }


def initialize_affinity_from_v1(model: AffinityStructuralUNet, v1: ComparableStructuralUNet) -> None:
    """Seed-match all shared weights; new context/affinity parameters stay neutral."""

    if v1.cfg.operator != "v1" or tuple(v1.cfg.widths) != tuple(model.cfg.widths):
        raise ValueError("initialization requires a width-matched v1 model")
    missing, unexpected = model.load_state_dict(v1.state_dict(), strict=False)
    if unexpected:
        raise ValueError(f"unexpected v1 state keys: {unexpected}")
    allowed = ("context_encoder", "affinity_mlp", "raw_beta")
    if any(not any(token in key for token in allowed) for key in missing):
        raise ValueError(f"non-affinity weights were not initialized: {missing}")


def build_affinity_model(
    candidate_id: str,
    *,
    widths: tuple[int, int, int, int] = (16, 32, 64, 96),
    seed_matched_v1: ComparableStructuralUNet | None = None,
) -> nn.Module:
    if candidate_id == "C0":
        if seed_matched_v1 is None:
            raise ValueError("C0 requires the explicit seed-matched v1 instance")
        return seed_matched_v1
    if candidate_id not in {"C1", "C2", "C3"}:
        raise ValueError(candidate_id)
    model = AffinityStructuralUNet(
        AffinityStructuralUNetConfig(
            widths=widths,
            use_affinity=candidate_id in {"C2", "C3"},
            use_radius2=candidate_id == "C3",
        )
    )
    if seed_matched_v1 is not None:
        initialize_affinity_from_v1(model, seed_matched_v1)
    return model

