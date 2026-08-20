"""Comparable lightweight models for the controlled ANZA-LIRA v2 study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .azconv import AZConv2d, AZConvConfig
from .azconv_v2 import AZConvV2Config, ModeResolvedAZConv2d


class NativeDeformConv2d(nn.Module):
    """Dependency-free stride-1 deformable convolution using grid_sample."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 != 1:
            raise ValueError("NativeDeformConv2d requires an odd positive kernel size")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.padding = kernel_size // 2
        patch_area = kernel_size**2
        self.offset_head = nn.Conv2d(in_channels, 2 * patch_area, kernel_size=3, padding=1)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, patch_area))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        offsets = [
            (column - self.padding, row - self.padding)
            for row in range(kernel_size)
            for column in range(kernel_size)
        ]
        self.register_buffer("kernel_offsets_xy", torch.tensor(offsets, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("NativeDeformConv2d input must be BxCxHxW")
        batch, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {channels}")
        patch_area = self.kernel_size**2
        learned = self.offset_head(x).reshape(batch, patch_area, 2, height, width)
        yy, xx = torch.meshgrid(
            torch.arange(height, device=x.device, dtype=x.dtype),
            torch.arange(width, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        base_x = xx.view(1, 1, height, width)
        base_y = yy.view(1, 1, height, width)
        sample_x = base_x + self.kernel_offsets_xy[:, 0].view(1, patch_area, 1, 1) + learned[:, :, 0]
        sample_y = base_y + self.kernel_offsets_xy[:, 1].view(1, patch_area, 1, 1) + learned[:, :, 1]
        normalized_x = 2.0 * sample_x / max(width - 1, 1) - 1.0
        normalized_y = 2.0 * sample_y / max(height - 1, 1) - 1.0
        grid = torch.stack([normalized_x, normalized_y], dim=-1)
        sampled = F.grid_sample(
            x.unsqueeze(1).expand(-1, patch_area, -1, -1, -1).reshape(
                batch * patch_area, channels, height, width
            ),
            grid.reshape(batch * patch_area, height, width, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).reshape(batch, patch_area, channels, height, width).permute(0, 2, 1, 3, 4)
        return torch.einsum("bckhw,ock->bohw", sampled, self.weight) + self.bias.view(1, -1, 1, 1)


class _ComparableBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        operator: str,
        *,
        num_modes: int,
        v2_cfg: AZConvV2Config | None,
    ) -> None:
        super().__init__()
        self.operator_name = operator
        self.last_diagnostics: dict[str, Any] | None = None
        if operator == "conv":
            self.spatial: nn.Module = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=False
            )
        elif operator == "deformable":
            self.spatial = NativeDeformConv2d(in_channels, out_channels, kernel_size=3)
        elif operator == "v1":
            self.spatial = AZConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                num_rules=num_modes,
                cfg=AZConvConfig(),
            )
        elif operator in {"v2a", "v2b"}:
            if v2_cfg is None or v2_cfg.variant != operator:
                raise ValueError("V2 block requires a matching V2 config")
            self.spatial = ModeResolvedAZConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                cfg=v2_cfg,
            )
        else:
            raise ValueError(f"Unknown comparable block operator: {operator}")
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.refine = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, *, collect_diagnostics: bool = False) -> torch.Tensor:
        if isinstance(self.spatial, ModeResolvedAZConv2d) and collect_diagnostics:
            diagnostics = self.spatial(x, return_diagnostics=True)
            self.last_diagnostics = diagnostics
            x = diagnostics["output"]
        else:
            self.last_diagnostics = None
            x = self.spatial(x)
        x = F.relu(self.norm1(x), inplace=False)
        return F.relu(self.norm2(self.refine(x)), inplace=False)


class _UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


@dataclass(frozen=True)
class ComparableUNetConfig:
    operator: str = "conv"
    widths: tuple[int, int, int, int] = (16, 32, 64, 96)
    num_modes: int = 4
    transport_steps: int = 2
    structural_completion_head: bool = False
    use_fuzzy: bool = True
    use_junction: bool = True
    use_cone: bool = True
    kappa_theta: float = 4.0
    kappa_direction: float = 4.0


class ComparableStructuralUNet(nn.Module):
    """One U-Net topology with a controlled choice of encoder operator."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        cfg: ComparableUNetConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or ComparableUNetConfig()
        if self.cfg.operator not in {"conv", "deformable", "v1", "v2a", "v2b"}:
            raise ValueError(f"Unknown model operator: {self.cfg.operator}")
        w1, w2, w3, wb = self.cfg.widths
        v2_cfg = None
        if self.cfg.operator in {"v2a", "v2b"}:
            v2_cfg = AZConvV2Config(
                num_modes=self.cfg.num_modes,
                transport_steps=self.cfg.transport_steps,
                variant=self.cfg.operator,
                kappa_theta=self.cfg.kappa_theta,
                kappa_direction=self.cfg.kappa_direction,
                use_fuzzy=self.cfg.use_fuzzy,
                use_junction=self.cfg.use_junction,
                use_cone=self.cfg.use_cone,
            )
        kwargs = {"operator": self.cfg.operator, "num_modes": self.cfg.num_modes, "v2_cfg": v2_cfg}
        self.enc1 = _ComparableBlock(in_channels, w1, **kwargs)
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
        self.completion_head = (
            nn.Conv2d(w1, out_channels, 1) if self.cfg.structural_completion_head else None
        )

    def forward(self, x: torch.Tensor, *, return_diagnostics: bool = False) -> torch.Tensor | dict[str, Any]:
        x1 = self.enc1(x, collect_diagnostics=return_diagnostics)
        x2 = self.enc2(self.pool(x1), collect_diagnostics=return_diagnostics)
        x3 = self.enc3(self.pool(x2), collect_diagnostics=return_diagnostics)
        bottleneck = self.bottleneck(self.pool(x3))
        y = self.up3(bottleneck, x3)
        y = self.up2(y, x2)
        y = self.up1(y, x1)
        visible_logits = self.visible_head(y)
        if not return_diagnostics:
            return visible_logits
        result: dict[str, Any] = {
            "visible_logits": visible_logits,
            "transport_diagnostics": [
                block.last_diagnostics
                for block in (self.enc1, self.enc2, self.enc3)
                if block.last_diagnostics is not None
            ],
            "operator": self.cfg.operator,
        }
        if self.completion_head is not None:
            result["completion_logits"] = self.completion_head(y)
        return result


def build_comparable_model(name: str, *, widths: tuple[int, int, int, int] = (16, 32, 64, 96)) -> ComparableStructuralUNet:
    configs = {
        "unet": ComparableUNetConfig(operator="conv", widths=widths),
        "deformable_unet": ComparableUNetConfig(operator="deformable", widths=widths),
        "anza_v1": ComparableUNetConfig(operator="v1", widths=widths),
        "anza_v2a": ComparableUNetConfig(operator="v2a", widths=widths),
        "anza_v2b": ComparableUNetConfig(operator="v2b", widths=widths),
        "anza_v2_full": ComparableUNetConfig(
            operator="v2b",
            widths=widths,
            structural_completion_head=True,
        ),
    }
    if name not in configs:
        raise ValueError(f"Unknown comparable model: {name}")
    return ComparableStructuralUNet(cfg=configs[name])
