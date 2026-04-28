"""Segmentation models for DRIVE-style vessel extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .azconv import AZConv2d, AZConvConfig


class ConvBlock(nn.Module):
    """Two-layer Conv-BN-ReLU block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AZConvBlock(nn.Module):
    """AZ block followed by a standard refinement conv."""

    def __init__(self, in_channels: int, out_channels: int, num_rules: int, cfg: AZConvConfig) -> None:
        super().__init__()
        self.block = nn.Sequential(
            AZConv2d(in_channels, out_channels, kernel_size=3, num_rules=num_rules, cfg=cfg),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Bilinear upsampling followed by skip fusion."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fuse = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel reweighting."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class AttentionGate(nn.Module):
    """Gate skip features using decoder context."""

    def __init__(self, skip_channels: int, gate_channels: int) -> None:
        super().__init__()
        inter = max(min(skip_channels, gate_channels) // 2, 8)
        self.theta = nn.Conv2d(skip_channels, inter, kernel_size=1, bias=False)
        self.phi = nn.Conv2d(gate_channels, inter, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(inter, 1, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(inter)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gate_up = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        attn = self.relu(self.bn(self.theta(skip) + self.phi(gate_up)))
        alpha = self.sigmoid(self.psi(attn))
        return skip * alpha


class ResidualConvBlock(nn.Module):
    """Residual conv block with optional Squeeze-Excitation."""

    def __init__(self, in_channels: int, out_channels: int, use_se: bool = True) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        y = self.se(y)
        return self.act(y + self.shortcut(x))


class AZResidualBlock(nn.Module):
    """Residual block with AZConv as the first spatial operator."""

    def __init__(self, in_channels: int, out_channels: int, num_rules: int, cfg: AZConvConfig, use_se: bool = True) -> None:
        super().__init__()
        self.body = nn.Sequential(
            AZConv2d(in_channels, out_channels, kernel_size=3, num_rules=num_rules, cfg=cfg),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        y = self.se(y)
        return self.act(y + self.shortcut(x))


class HybridAZResidualBlock(nn.Module):
    """Residual block that blends a standard conv branch with an AZ branch."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_rules: int,
        cfg: AZConvConfig,
        use_se: bool = True,
        mix_init: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 < float(mix_init) < 1.0:
            raise ValueError("mix_init must be in the open interval (0, 1).")
        self.conv_body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.az_body = nn.Sequential(
            AZConv2d(in_channels, out_channels, kernel_size=3, num_rules=num_rules, cfg=cfg),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        mix_value = float(min(max(mix_init, 1e-4), 1.0 - 1e-4))
        mix_logit = torch.logit(torch.tensor(mix_value, dtype=torch.float32))
        self.mix_logit = nn.Parameter(mix_logit)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_y = self.conv_body(x)
        az_y = self.az_body(x)
        mix = torch.sigmoid(self.mix_logit)
        y = (1.0 - mix) * conv_y + mix * az_y
        y = self.se(y)
        return self.act(y + self.shortcut(x))


class AZResidualRefineBlock(nn.Module):
    """Residual block where both spatial operators are AZConv2d."""

    def __init__(self, in_channels: int, out_channels: int, num_rules: int, cfg: AZConvConfig, use_se: bool = True) -> None:
        super().__init__()
        self.body = nn.Sequential(
            AZConv2d(in_channels, out_channels, kernel_size=3, num_rules=num_rules, cfg=cfg),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            AZConv2d(out_channels, out_channels, kernel_size=3, num_rules=num_rules, cfg=cfg),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        y = self.se(y)
        return self.act(y + self.shortcut(x))


class ASPP(nn.Module):
    """Lightweight ASPP for multi-scale context."""

    def __init__(self, in_channels: int, out_channels: int, dilations: tuple[int, ...] = (1, 2, 4, 8)) -> None:
        super().__init__()
        branches = []
        for d in dilations:
            branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d, dilation=d, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(dilations), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        return self.project(torch.cat(feats, dim=1))


class UpAttentionBlock(nn.Module):
    """Upsample, attention-gate skip, then residual fuse."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.gate = AttentionGate(skip_channels=skip_channels, gate_channels=in_channels)
        self.fuse = ResidualConvBlock(in_channels + skip_channels, out_channels, use_se=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.gate(skip, x)
        return self.fuse(torch.cat([x, skip], dim=1))


class UpAttentionConvBlock(nn.Module):
    """Upsample, attention-gate skip, then standard conv fuse."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.gate = AttentionGate(skip_channels=skip_channels, gate_channels=in_channels)
        self.fuse = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.gate(skip, x)
        return self.fuse(torch.cat([x, skip], dim=1))


class UpAttentionAZBlock(nn.Module):
    """Upsample, attention-gate skip, then AZ residual fuse."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, num_rules: int, cfg: AZConvConfig) -> None:
        super().__init__()
        self.gate = AttentionGate(skip_channels=skip_channels, gate_channels=in_channels)
        self.fuse = AZResidualRefineBlock(in_channels + skip_channels, out_channels, num_rules=num_rules, cfg=cfg, use_se=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.gate(skip, x)
        return self.fuse(torch.cat([x, skip], dim=1))


class _RegularizedSegmentationMixin:
    """Collect regularization terms from nested AZ layers."""

    def regularization_terms(self) -> dict[str, torch.Tensor]:
        totals: dict[str, torch.Tensor] = {}
        for module in self.modules():
            if isinstance(module, AZConv2d):
                for key, value in module.regularization_terms().items():
                    if key in totals:
                        totals[key] = totals[key] + value
                    else:
                        totals[key] = value
        mix_terms: list[torch.Tensor] = []
        for module in self.modules():
            mix_logit = getattr(module, "mix_logit", None)
            if isinstance(mix_logit, nn.Parameter):
                mix_terms.append(torch.sigmoid(mix_logit))

        if mix_terms:
            mix_values = torch.stack(mix_terms)
            mix_target = getattr(self, "hybrid_mix_target", None)
            if mix_target is None:
                totals["hybrid_mix_target"] = mix_values.new_zeros(())
            else:
                # L1-style attraction gives a stronger gradient than MSE when
                # the mix branch is far from the desired operating region.
                totals["hybrid_mix_target"] = (mix_values - float(mix_target)).abs().mean()
        if totals:
            totals.setdefault("hybrid_mix_target", next(self.parameters()).new_zeros(()))
            return totals

        zero = next(self.parameters()).new_zeros(())
        return {
            "membership_entropy": zero,
            "membership_smoothness": zero,
            "geometry_smoothness": zero,
            "hyperbolicity_penalty": zero,
            "anisotropy_gap": zero,
            "hybrid_mix_target": zero,
        }

    def axis_alignment_loss(self, target: torch.Tensor, valid_mask: torch.Tensor, num_iters: int = 8) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for module in self.modules():
            if isinstance(module, AZConv2d):
                losses.append(module.axis_alignment_loss(target, valid_mask, num_iters=num_iters))
        if not losses:
            return target.new_zeros(())
        return torch.stack(losses).mean()


class BaselineUNet(_RegularizedSegmentationMixin, nn.Module):
    """Small U-Net baseline for retinal vessel segmentation."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, widths: tuple[int, ...] = (32, 64, 128, 192)) -> None:
        super().__init__()
        w1, w2, w3, wb = widths
        self.enc1 = ConvBlock(in_channels, w1)
        self.enc2 = ConvBlock(w1, w2)
        self.enc3 = ConvBlock(w2, w3)
        self.bottleneck = ConvBlock(w3, wb)
        self.pool = nn.MaxPool2d(2)

        self.up3 = UpBlock(wb, w3, w3)
        self.up2 = UpBlock(w3, w2, w2)
        self.up1 = UpBlock(w2, w1, w1)
        self.head = nn.Conv2d(w1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        xb = self.bottleneck(self.pool(x3))

        y = self.up3(xb, x3)
        y = self.up2(y, x2)
        y = self.up1(y, x1)
        return self.head(y)


class AttentionUNet(_RegularizedSegmentationMixin, nn.Module):
    """Attention U-Net baseline for stronger segmentation comparisons."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, widths: tuple[int, ...] = (32, 64, 128, 192)) -> None:
        super().__init__()
        w1, w2, w3, wb = widths
        self.enc1 = ConvBlock(in_channels, w1)
        self.enc2 = ConvBlock(w1, w2)
        self.enc3 = ConvBlock(w2, w3)
        self.bottleneck = ConvBlock(w3, wb)
        self.pool = nn.MaxPool2d(2)

        self.up3 = UpAttentionConvBlock(wb, w3, w3)
        self.up2 = UpAttentionConvBlock(w3, w2, w2)
        self.up1 = UpAttentionConvBlock(w2, w1, w1)
        self.head = nn.Conv2d(w1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        xb = self.bottleneck(self.pool(x3))

        y = self.up3(xb, x3)
        y = self.up2(y, x2)
        y = self.up1(y, x1)
        return self.head(y)


class AZUNet(_RegularizedSegmentationMixin, nn.Module):
    """AZ-enhanced U-Net for DRIVE vessel segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        num_rules: int = 4,
        cfg: AZConvConfig | None = None,
        widths: tuple[int, ...] = (32, 64, 128, 192),
    ) -> None:
        super().__init__()
        cfg = cfg or AZConvConfig()
        w1, w2, w3, wb = widths
        self.enc1 = AZConvBlock(in_channels, w1, num_rules, cfg)
        self.enc2 = AZConvBlock(w1, w2, num_rules, cfg)
        self.enc3 = AZConvBlock(w2, w3, num_rules, cfg)
        self.bottleneck = ConvBlock(w3, wb)
        self.pool = nn.MaxPool2d(2)

        self.up3 = UpBlock(wb, w3, w3)
        self.up2 = UpBlock(w3, w2, w2)
        self.up1 = UpBlock(w2, w1, w1)
        self.head = nn.Conv2d(w1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        xb = self.bottleneck(self.pool(x3))

        y = self.up3(xb, x3)
        y = self.up2(y, x2)
        y = self.up1(y, x1)
        return self.head(y)


class AZSOTAUNet(_RegularizedSegmentationMixin, nn.Module):
    """
    Research-grade AZ architecture:
    - residual AZ encoder
    - ASPP multi-scale bottleneck
    - attention-gated decoder
    - deep supervision + boundary head
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        num_rules: int = 4,
        cfg: AZConvConfig | None = None,
        widths: tuple[int, ...] = (48, 96, 160, 224),
        pure_az: bool = False,
        encoder_az_stages: int = 3,
        encoder_block_mode: str = "az",
        hybrid_mix_init: float = 0.5,
        hybrid_mix_target: float | None = None,
        bottleneck_mode: str | None = None,
        decoder_mode: str | None = None,
        boundary_mode: str | None = None,
    ) -> None:
        super().__init__()
        cfg = cfg or AZConvConfig()
        w1, w2, w3, wb = widths

        if bottleneck_mode is None:
            bottleneck_mode = "az_double" if pure_az else "aspp"
        if decoder_mode is None:
            decoder_mode = "az" if pure_az else "residual"
        if boundary_mode is None:
            boundary_mode = "az" if pure_az else "conv"

        allowed_bottlenecks = {"aspp", "az_single", "az_double"}
        allowed_decoders = {"residual", "az"}
        allowed_boundaries = {"conv", "az"}
        allowed_encoder_modes = {"az", "hybrid", "hybrid_shallow"}
        if not 0 <= int(encoder_az_stages) <= 3:
            raise ValueError("encoder_az_stages must be in the range [0, 3].")
        if encoder_block_mode not in allowed_encoder_modes:
            raise ValueError(f"Unknown encoder_block_mode '{encoder_block_mode}'. Expected one of {sorted(allowed_encoder_modes)}.")
        if not 0.0 < float(hybrid_mix_init) < 1.0:
            raise ValueError("hybrid_mix_init must be in the open interval (0, 1).")
        if bottleneck_mode not in allowed_bottlenecks:
            raise ValueError(f"Unknown bottleneck_mode '{bottleneck_mode}'. Expected one of {sorted(allowed_bottlenecks)}.")
        if decoder_mode not in allowed_decoders:
            raise ValueError(f"Unknown decoder_mode '{decoder_mode}'. Expected one of {sorted(allowed_decoders)}.")
        if boundary_mode not in allowed_boundaries:
            raise ValueError(f"Unknown boundary_mode '{boundary_mode}'. Expected one of {sorted(allowed_boundaries)}.")

        self.encoder_az_stages = int(encoder_az_stages)
        self.encoder_block_mode = encoder_block_mode
        self.hybrid_mix_init = float(hybrid_mix_init)
        self.hybrid_mix_target = float(hybrid_mix_target) if hybrid_mix_target is not None else None

        def _encoder_block(stage_index: int, in_ch: int, out_ch: int) -> nn.Module:
            if stage_index <= self.encoder_az_stages:
                if self.encoder_block_mode == "hybrid":
                    return HybridAZResidualBlock(
                        in_ch,
                        out_ch,
                        num_rules,
                        cfg,
                        use_se=True,
                        mix_init=self.hybrid_mix_init,
                    )
                if self.encoder_block_mode == "hybrid_shallow" and stage_index == 1:
                    return HybridAZResidualBlock(
                        in_ch,
                        out_ch,
                        num_rules,
                        cfg,
                        use_se=True,
                        mix_init=self.hybrid_mix_init,
                    )
                return AZResidualBlock(in_ch, out_ch, num_rules, cfg, use_se=True)
            return ResidualConvBlock(in_ch, out_ch, use_se=True)

        self.enc1 = _encoder_block(1, in_channels, w1)
        self.enc2 = _encoder_block(2, w1, w2)
        self.enc3 = _encoder_block(3, w2, w3)
        self.pool = nn.MaxPool2d(2)
        self.pure_az = bool(pure_az)
        self.bottleneck_mode = bottleneck_mode
        self.decoder_mode = decoder_mode
        self.boundary_mode = boundary_mode

        if self.bottleneck_mode == "aspp":
            self.bottleneck = ASPP(w3, wb, dilations=(1, 2, 4, 8))
        elif self.bottleneck_mode == "az_single":
            self.bottleneck = AZResidualRefineBlock(w3, wb, num_rules=num_rules, cfg=cfg, use_se=True)
        else:
            self.bottleneck = nn.Sequential(
                AZResidualRefineBlock(w3, wb, num_rules=num_rules, cfg=cfg, use_se=True),
                AZResidualRefineBlock(wb, wb, num_rules=num_rules, cfg=cfg, use_se=True),
            )

        if self.decoder_mode == "az":
            self.up3 = UpAttentionAZBlock(wb, w3, w3, num_rules=num_rules, cfg=cfg)
            self.up2 = UpAttentionAZBlock(w3, w2, w2, num_rules=num_rules, cfg=cfg)
            self.up1 = UpAttentionAZBlock(w2, w1, w1, num_rules=num_rules, cfg=cfg)
        else:
            self.up3 = UpAttentionBlock(wb, w3, w3)
            self.up2 = UpAttentionBlock(w3, w2, w2)
            self.up1 = UpAttentionBlock(w2, w1, w1)

        if self.boundary_mode == "az":
            self.boundary_head = nn.Sequential(
                AZConv2d(w1, w1, kernel_size=3, num_rules=num_rules, cfg=cfg),
                nn.BatchNorm2d(w1),
                nn.ReLU(inplace=True),
                nn.Conv2d(w1, 1, kernel_size=1),
            )
        else:
            self.boundary_head = nn.Sequential(
                nn.Conv2d(w1, w1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(w1),
                nn.ReLU(inplace=True),
                nn.Conv2d(w1, 1, kernel_size=1),
            )

        self.main_head = nn.Conv2d(w1, out_channels, kernel_size=1)
        self.aux2_head = nn.Conv2d(w2, out_channels, kernel_size=1)
        self.aux3_head = nn.Conv2d(w3, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        xb = self.bottleneck(self.pool(x3))

        y3 = self.up3(xb, x3)
        y2 = self.up2(y3, x2)
        y1 = self.up1(y2, x1)

        main = self.main_head(y1)
        aux2 = F.interpolate(self.aux2_head(y2), size=main.shape[-2:], mode="bilinear", align_corners=False)
        aux3 = F.interpolate(self.aux3_head(y3), size=main.shape[-2:], mode="bilinear", align_corners=False)
        boundary = self.boundary_head(y1)

        return {
            "logits": main,
            "aux_logits": [aux2, aux3],
            "boundary_logits": boundary,
        }
