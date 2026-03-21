"""
Anosov-inspired / hyperbolic anisotropic + fuzzy-gated local aggregation (AZConv2d).

This is an engineering-inspired approximation; it is not a rigorous dynamical-systems operator.
See README for formulas and caveats.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AZConvConfig:
    """Ablation switches for AZConv2d."""

    use_fuzzy: bool = True
    use_anisotropy: bool = True
    learn_directions: bool = True


class AZConv2d(nn.Module):
    """
    Local k×k neighborhood aggregation with rule-wise anisotropic kernels and fuzzy compatibilities.

    Forward:
      x: [B, C_in, H, W]
      unfold patches: [B, C_in, S, L] with S=k*k, L=H*W
      gating fuzzy memberships: mu: [B, R, H, W]
      alpha: [B, R, S, L]
      aggregate per rule over neighbors: [B, R, C_in, L] -> reshape to [B, R*C_in, H, W]
      pointwise 1×1 mixes rules × channels to [B, C_out, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_rules: int = 4,
        cfg: Optional[AZConvConfig] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("AZConv2d expects odd kernel_size for same-size unfold/fold.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = kernel_size
        self.R = num_rules
        self.cfg = cfg or AZConvConfig()
        self.pad = kernel_size // 2

        # 1×1 gating: logits per rule -> softmax over R fuzzy memberships μ in [0,1]^R
        self.gate_conv = nn.Conv2d(in_channels, num_rules, kernel_size=1, bias=True)

        # Directions: stable/unstable axes encoded as angle θ_r; axes are orthonormal.
        if self.cfg.learn_directions:
            self.theta = nn.Parameter(torch.zeros(num_rules))
        else:
            # Fixed, evenly spaced orientations (not learned).
            if num_rules == 1:
                fixed = torch.zeros(1)
            else:
                fixed = torch.linspace(0.0, math.pi * (num_rules - 1) / num_rules, num_rules)
            self.register_buffer("theta", fixed)

        # Positive scales via softplus for stable/unstable directions.
        self.raw_sigma_u = nn.Parameter(torch.zeros(num_rules))
        self.raw_sigma_s = nn.Parameter(torch.zeros(num_rules))
        self.raw_sigma_iso = nn.Parameter(torch.zeros(num_rules))

        # Mix aggregated rule×channel volume to output channels.
        self.pointwise = nn.Conv2d(in_channels * num_rules, out_channels, kernel_size=1, bias=bias)

        self._register_offset_buffers()

    def _register_offset_buffers(self) -> None:
        """Integer offsets (dx, dy) matching torch.nn.functional.unfold patch order (row-major)."""

        k = self.k
        dx_list = []
        dy_list = []
        for row in range(k):
            for col in range(k):
                dy = row - k // 2
                dx = col - k // 2
                dx_list.append(float(dx))
                dy_list.append(float(dy))
        # Shapes are [1,1,S,1] so they broadcast with direction vectors [R,1,1,1].
        self.register_buffer("_dx", torch.tensor(dx_list).view(1, 1, -1, 1))
        self.register_buffer("_dy", torch.tensor(dy_list).view(1, 1, -1, 1))

    def _anisotropic_kernel(self, device: torch.device) -> torch.Tensor:
        """
        k_r(m) = exp( -||P_u m||²/σ_u² - ||P_s m||²/σ_s² )
        where u,s are orthonormal directions from angle θ_r and m=(dx,dy) offsets.

        Returns:
          k: [R, S]
        """

        theta = self.theta if self.cfg.learn_directions else self.theta.detach()

        # u=(cosθ,sinθ), s=(-sinθ,cosθ) in (dx,dy) coordinates.
        # Use [R,1,1,1] so broadcasting with dx [1,1,S,1] yields [R,1,S,1].
        cos_t = torch.cos(theta).view(self.R, 1, 1, 1)
        sin_t = torch.sin(theta).view(self.R, 1, 1, 1)
        u_x, u_y = cos_t, sin_t
        s_x, s_y = -sin_t, cos_t

        dx = self._dx.to(device)
        dy = self._dy.to(device)
        proj_u = u_x * dx + u_y * dy  # [R,1,S,1]
        proj_s = s_x * dx + s_y * dy  # [R,1,S,1]

        sigma_u = F.softplus(self.raw_sigma_u).view(self.R, 1, 1, 1) + 1e-4
        sigma_s = F.softplus(self.raw_sigma_s).view(self.R, 1, 1, 1) + 1e-4
        k = torch.exp(-(proj_u**2) / (sigma_u**2) - (proj_s**2) / (sigma_s**2))  # [R,1,S,1]
        return k.squeeze(-1).squeeze(1)  # [R,S]

    def _isotropic_kernel(self, device: torch.device) -> torch.Tensor:
        """Ablation: isotropic Gaussian kernel on ||m||² per rule."""

        dx = self._dx.to(device)
        dy = self._dy.to(device)
        dist2 = dx**2 + dy**2  # [1,1,S,1]
        sigma = F.softplus(self.raw_sigma_iso).view(self.R, 1, 1, 1) + 1e-4
        k = torch.exp(-dist2 / (sigma**2))  # [R,1,S,1] via broadcasting
        return k.squeeze(-1).squeeze(1)  # [R,S]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, H, W]
        B, C, H, W = x.shape
        k, R, pad = self.k, self.R, self.pad
        S = k * k
        L = H * W

        x_unfold = F.unfold(x, kernel_size=k, padding=pad, stride=1)
        assert x_unfold.shape == (B, C * S, L), f"unfold shape {x_unfold.shape} vs expected {(B, C * S, L)}"
        x_un = x_unfold.view(B, C, S, L)

        # Fuzzy memberships μ_r(center) and μ_r(neighbor).
        logits = self.gate_conv(x)  # [B, R, H, W]
        if self.cfg.use_fuzzy:
            mu = F.softmax(logits, dim=1)
        else:
            # Uniform memberships per rule: removes fuzzy gating signal.
            mu = torch.full_like(logits, 1.0 / R)

        mu_unfold = F.unfold(mu, kernel_size=k, padding=pad, stride=1)  # [B, R*S, L]
        assert mu_unfold.shape == (B, R * S, L)
        mu_un = mu_unfold.view(B, R, S, L)  # [B, R, S, L]

        # Extract μ_r(center) from the unfold window.
        s_center = (k // 2) * k + (k // 2)
        mu_center = mu_un[:, :, s_center : s_center + 1, :]  # [B,R,1,L]

        # Anisotropic vs isotropic kernel on offsets.
        device = x.device
        kern = self._anisotropic_kernel(device) if self.cfg.use_anisotropy else self._isotropic_kernel(device)
        # kern: [R,S] -> [1,R,S,1] for broadcast
        alpha = mu_center * mu_un * kern.view(1, R, S, 1)  # [B,R,S,L]

        # Normalize across rules and neighbors for each spatial location.
        alpha_sum = alpha.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        alpha = alpha / alpha_sum

        # Weighted aggregation over neighbors per rule.
        agg = torch.einsum("brsl,bcsl->brcl", alpha, x_un)  # [B,R,C,L]
        agg_flat = agg.reshape(B, R * C, H, W)
        out = self.pointwise(agg_flat)  # [B,out_channels,H,W]
        return out


def _conv_block_az(in_c: int, out_c: int, num_rules: int, cfg: AZConvConfig) -> nn.Sequential:
    return nn.Sequential(
        AZConv2d(in_c, out_c, kernel_size=3, num_rules=num_rules, cfg=cfg),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


def _conv_block_std(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class AZConvNet(nn.Module):
    """
    CIFAR-sized CNN: two AZConv stages + one standard 3×3 stage + classifier.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        num_rules: int = 4,
        cfg: Optional[AZConvConfig] = None,
    ) -> None:
        super().__init__()
        cfg = cfg or AZConvConfig()
        self.cfg = cfg
        self.features = nn.Sequential(
            _conv_block_az(in_channels, 64, num_rules, cfg),
            nn.MaxPool2d(2),
            _conv_block_az(64, 128, num_rules, cfg),
            nn.MaxPool2d(2),
            _conv_block_std(128, 192),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

