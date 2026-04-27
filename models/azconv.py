"""
AZConv2d: local Anosov-Zadeh kernel aggregation.

The mathematically clean object in this repo is the raw local kernel

    K_r(center, neighbor) = mu_r(center) * mu_r(neighbor) * kappa_r(center, neighbor, offset)

where kappa_r is aligned either with a fixed cat-map stable/unstable splitting,
with a global learned anisotropic frame, or with a local learned hyperbolic
geometry. In the local mode, pairwise geometry is built symmetrically from the
center and neighbor fields so the raw kernel remains symmetric in the pair.

For CNN training stability we optionally normalize these raw compatibilities
across rules and neighbors before aggregating value features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AZConvConfig:
    """Configuration and ablations for AZConv2d."""

    use_fuzzy: bool = True
    use_anisotropy: bool = True
    learn_directions: bool = True
    geometry_mode: str = "local_hyperbolic"
    use_value_projection: bool = True
    normalize_kernel: bool = True
    min_hyperbolicity: float = 0.1
    fuzzy_temperature: float = 1.0
    normalize_mode: str = "global"
    compatibility_floor: float = 0.0
    use_input_residual: bool = False
    residual_init: float = 0.0
    geometry_kernel_size: int = 1
    init_anisotropy_gap: float = 0.35


class AZConv2d(nn.Module):
    """
    Local k x k aggregation with rule-wise anisotropic kernels and fuzzy compatibilities.

    Forward:
      x: [B, C_in, H, W]
      value patches: [B, C_in, S, L] with S=k*k, L=H*W
      fuzzy memberships: [B, R, H, W]
      raw compatibilities: [B, R, S, L]
      aggregate per rule over neighbors: [B, R, C_in, L]
      pointwise 1x1 mixes rules x channels to [B, C_out, H, W]
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
        self._last_reg_terms: dict[str, torch.Tensor] = {}
        self._last_interpretation: dict[str, Any] = {}

        valid_modes = {"learned_angle", "fixed_cat_map", "learned_hyperbolic", "local_hyperbolic"}
        if self.cfg.geometry_mode not in valid_modes:
            raise ValueError(
                f"Unknown geometry_mode={self.cfg.geometry_mode!r}; "
                f"expected one of {sorted(valid_modes)}."
            )
        if float(self.cfg.fuzzy_temperature) <= 0.0:
            raise ValueError("fuzzy_temperature must be positive.")
        valid_norm_modes = {"global", "per_rule", "none"}
        if self.cfg.normalize_mode not in valid_norm_modes:
            raise ValueError(
                f"Unknown normalize_mode={self.cfg.normalize_mode!r}; "
                f"expected one of {sorted(valid_norm_modes)}."
            )
        if float(self.cfg.compatibility_floor) < 0.0:
            raise ValueError("compatibility_floor must be non-negative.")
        if not 0.0 <= float(self.cfg.residual_init) <= 1.0:
            raise ValueError("residual_init must be in [0, 1].")
        if int(self.cfg.geometry_kernel_size) <= 0 or int(self.cfg.geometry_kernel_size) % 2 != 1:
            raise ValueError("geometry_kernel_size must be a positive odd integer.")
        if float(self.cfg.init_anisotropy_gap) < 0.0:
            raise ValueError("init_anisotropy_gap must be non-negative.")

        self.gate_conv = nn.Conv2d(in_channels, num_rules, kernel_size=1, bias=True)
        self.value_conv = (
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
            if self.cfg.use_value_projection
            else nn.Identity()
        )

        cat_theta = math.atan2((3.0 + math.sqrt(5.0)) / 2.0 - 2.0, 1.0)
        theta_init = cat_theta + torch.linspace(0.0, math.pi * (num_rules - 1) / max(num_rules, 1), num_rules)

        if self.cfg.geometry_mode in {"learned_angle", "learned_hyperbolic"}:
            if self.cfg.learn_directions:
                self.theta = nn.Parameter(theta_init.clone())
            else:
                self.register_buffer("theta", theta_init.clone())

        if self.cfg.geometry_mode in {"learned_hyperbolic", "local_hyperbolic"}:
            if self.cfg.geometry_mode == "local_hyperbolic":
                geom_k = int(self.cfg.geometry_kernel_size)
                self.geometry_conv = nn.Conv2d(
                    in_channels,
                    3 * num_rules,
                    kernel_size=geom_k,
                    padding=geom_k // 2,
                    bias=True,
                )
                self._init_local_geometry_head(theta_init)
                self.register_parameter("raw_base_scale", None)
                self.register_parameter("raw_hyperbolicity", None)
            else:
                self.raw_base_scale = nn.Parameter(torch.zeros(num_rules))
                self.raw_hyperbolicity = nn.Parameter(torch.zeros(num_rules))
                self.geometry_conv = None
            self.register_parameter("raw_sigma_u", None)
            self.register_parameter("raw_sigma_s", None)
        else:
            self.raw_sigma_u = nn.Parameter(torch.zeros(num_rules))
            self.raw_sigma_s = nn.Parameter(torch.zeros(num_rules))
            self._init_global_sigma_anisotropy()
            self.register_parameter("raw_base_scale", None)
            self.register_parameter("raw_hyperbolicity", None)
            self.geometry_conv = None

        self.raw_sigma_iso = nn.Parameter(torch.zeros(num_rules))
        self.pointwise = nn.Conv2d(in_channels * num_rules, out_channels, kernel_size=1, bias=bias)
        self.use_input_residual = bool(self.cfg.use_input_residual)
        if self.use_input_residual:
            init = float(min(max(self.cfg.residual_init, 1e-4), 1.0 - 1e-4))
            self.residual_logit = nn.Parameter(torch.logit(torch.tensor(init, dtype=torch.float32)))
            if in_channels == out_channels:
                self.residual_proj = nn.Identity()
            else:
                self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.register_parameter("residual_logit", None)
            self.residual_proj = nn.Identity()

        self._register_offset_buffers()
        self._register_cat_map_buffers()

    def _init_local_geometry_head(self, theta_init: torch.Tensor) -> None:
        assert self.geometry_conv is not None
        nn.init.zeros_(self.geometry_conv.weight)
        nn.init.zeros_(self.geometry_conv.bias)
        with torch.no_grad():
            self.geometry_conv.bias[: self.R].copy_(theta_init)

    @staticmethod
    def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
        value = value.clamp_min(1e-6)
        return value + torch.log(-torch.expm1(-value))

    def _init_global_sigma_anisotropy(self) -> None:
        gap = float(self.cfg.init_anisotropy_gap)
        if gap <= 0.0:
            return
        base_sigma = F.softplus(torch.zeros((), dtype=torch.float32)) + 1e-4
        sigma_u = base_sigma * math.exp(0.5 * gap)
        sigma_s = base_sigma * math.exp(-0.5 * gap)
        raw_u = self._inverse_softplus(torch.tensor(float(sigma_u) - 1e-4, dtype=torch.float32))
        raw_s = self._inverse_softplus(torch.tensor(float(sigma_s) - 1e-4, dtype=torch.float32))
        with torch.no_grad():
            assert self.raw_sigma_u is not None and self.raw_sigma_s is not None
            self.raw_sigma_u.fill_(float(raw_u))
            self.raw_sigma_s.fill_(float(raw_s))

    def _register_offset_buffers(self) -> None:
        dx_list = []
        dy_list = []
        for row in range(self.k):
            for col in range(self.k):
                dy = row - self.k // 2
                dx = col - self.k // 2
                dx_list.append(float(dx))
                dy_list.append(float(dy))

        self.register_buffer("_dx", torch.tensor(dx_list).view(1, 1, -1, 1))
        self.register_buffer("_dy", torch.tensor(dy_list).view(1, 1, -1, 1))

    def _register_cat_map_buffers(self) -> None:
        sqrt5 = math.sqrt(5.0)
        lambda_u = (3.0 + sqrt5) / 2.0
        lambda_s = (3.0 - sqrt5) / 2.0

        unstable = torch.tensor([1.0, lambda_u - 2.0], dtype=torch.float32)
        stable = torch.tensor([1.0, lambda_s - 2.0], dtype=torch.float32)
        unstable = unstable / unstable.norm()
        stable = stable / stable.norm()

        self.register_buffer("_cat_unstable", unstable)
        self.register_buffer("_cat_stable", stable)

    def _direction_components_global(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        if self.cfg.geometry_mode == "fixed_cat_map":
            unstable = self._cat_unstable.to(device)
            stable = self._cat_stable.to(device)
            return (
                unstable[0].view(1, 1, 1, 1).expand(1, self.R, 1, 1),
                unstable[1].view(1, 1, 1, 1).expand(1, self.R, 1, 1),
                stable[0].view(1, 1, 1, 1).expand(1, self.R, 1, 1),
                stable[1].view(1, 1, 1, 1).expand(1, self.R, 1, 1),
            )

        theta = self.theta if self.cfg.learn_directions else self.theta.detach()
        cos_t = torch.cos(theta).view(1, self.R, 1, 1)
        sin_t = torch.sin(theta).view(1, self.R, 1, 1)
        return cos_t, sin_t, -sin_t, cos_t

    def _global_anisotropic_sigmas(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cfg.geometry_mode == "learned_hyperbolic":
            base = F.softplus(self.raw_base_scale).view(1, self.R, 1, 1) + 1e-4
            hyper = F.softplus(self.raw_hyperbolicity).view(1, self.R, 1, 1)
            sigma_u = base * torch.exp(hyper)
            sigma_s = base * torch.exp(-hyper)
            gap = hyper
            return sigma_u, sigma_s, gap

        sigma_u = F.softplus(self.raw_sigma_u).view(1, self.R, 1, 1) + 1e-4
        sigma_s = F.softplus(self.raw_sigma_s).view(1, self.R, 1, 1) + 1e-4
        gap = torch.abs(torch.log(sigma_u) - torch.log(sigma_s))
        return sigma_u, sigma_s, gap

    def _global_anisotropic_kernel(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        u_x, u_y, s_x, s_y = self._direction_components_global(device)
        sigma_u, sigma_s, gap = self._global_anisotropic_sigmas()
        dx = self._dx.to(device)
        dy = self._dy.to(device)
        proj_u = u_x * dx + u_y * dy
        proj_s = s_x * dx + s_y * dy
        kern = torch.exp(-(proj_u**2) / (sigma_u**2) - (proj_s**2) / (sigma_s**2))
        geom_smoothness = dx.new_zeros(())
        interp = {
            "geometry_mode": self.cfg.geometry_mode,
            "u_x": u_x.detach(),
            "u_y": u_y.detach(),
            "s_x": s_x.detach(),
            "s_y": s_y.detach(),
            "sigma_u": sigma_u.detach(),
            "sigma_s": sigma_s.detach(),
            "gap": gap.detach(),
        }
        return kern, gap, geom_smoothness, interp

    def _local_hyperbolic_kernel(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        assert self.geometry_conv is not None
        batch, _channels, height, width = x.shape
        patch_area = self.k * self.k
        locations = height * width

        geom = self.geometry_conv(x)
        theta_map, raw_base_map, raw_hyper_map = torch.chunk(geom, 3, dim=1)
        base_map = F.softplus(raw_base_map) + 1e-4
        hyper_map = F.softplus(raw_hyper_map)

        theta_center = theta_map.reshape(batch, self.R, 1, locations)
        base_center = base_map.reshape(batch, self.R, 1, locations)
        hyper_center = hyper_map.reshape(batch, self.R, 1, locations)

        theta_un = F.unfold(theta_map, kernel_size=self.k, padding=self.pad, stride=1).view(batch, self.R, patch_area, locations)
        base_un = F.unfold(base_map, kernel_size=self.k, padding=self.pad, stride=1).view(batch, self.R, patch_area, locations)
        hyper_un = F.unfold(hyper_map, kernel_size=self.k, padding=self.pad, stride=1).view(batch, self.R, patch_area, locations)

        c2_center = torch.cos(2.0 * theta_center)
        s2_center = torch.sin(2.0 * theta_center)
        c2_neighbor = torch.cos(2.0 * theta_un)
        s2_neighbor = torch.sin(2.0 * theta_un)
        c2_pair = c2_center + c2_neighbor
        s2_pair = s2_center + s2_neighbor
        norm = torch.sqrt(c2_pair**2 + s2_pair**2).clamp_min(1e-6)
        c2_pair = c2_pair / norm
        s2_pair = s2_pair / norm
        theta_pair = 0.5 * torch.atan2(s2_pair, c2_pair)

        u_x = torch.cos(theta_pair)
        u_y = torch.sin(theta_pair)
        s_x = -u_y
        s_y = u_x

        base_pair = 0.5 * (base_center + base_un)
        hyper_pair = 0.5 * (hyper_center + hyper_un)
        sigma_u = base_pair * torch.exp(hyper_pair)
        sigma_s = base_pair * torch.exp(-hyper_pair)

        dx = self._dx.to(x.device)
        dy = self._dy.to(x.device)
        proj_u = u_x * dx + u_y * dy
        proj_s = s_x * dx + s_y * dy
        kern = torch.exp(-(proj_u**2) / (sigma_u**2) - (proj_s**2) / (sigma_s**2))

        c2_map = torch.cos(2.0 * theta_map)
        s2_map = torch.sin(2.0 * theta_map)
        geom_smoothness = x.new_zeros(())
        if width > 1:
            geom_smoothness = geom_smoothness + (c2_map[:, :, :, 1:] - c2_map[:, :, :, :-1]).abs().mean()
            geom_smoothness = geom_smoothness + (s2_map[:, :, :, 1:] - s2_map[:, :, :, :-1]).abs().mean()
            geom_smoothness = geom_smoothness + (base_map[:, :, :, 1:] - base_map[:, :, :, :-1]).abs().mean()
            geom_smoothness = geom_smoothness + (hyper_map[:, :, :, 1:] - hyper_map[:, :, :, :-1]).abs().mean()
        if height > 1:
            geom_smoothness = geom_smoothness + (c2_map[:, :, 1:, :] - c2_map[:, :, :-1, :]).abs().mean()
            geom_smoothness = geom_smoothness + (s2_map[:, :, 1:, :] - s2_map[:, :, :-1, :]).abs().mean()
            geom_smoothness = geom_smoothness + (base_map[:, :, 1:, :] - base_map[:, :, :-1, :]).abs().mean()
            geom_smoothness = geom_smoothness + (hyper_map[:, :, 1:, :] - hyper_map[:, :, :-1, :]).abs().mean()

        interp = {
            "geometry_mode": self.cfg.geometry_mode,
            "theta_map": theta_map.detach(),
            "base_map": base_map.detach(),
            "hyper_map": hyper_map.detach(),
            "sigma_u_map": (base_map * torch.exp(hyper_map)).detach(),
            "sigma_s_map": (base_map * torch.exp(-hyper_map)).detach(),
            "gap": hyper_center.detach(),
        }
        return kern, hyper_center, geom_smoothness, interp

    def _isotropic_kernel(self, device: torch.device) -> torch.Tensor:
        dx = self._dx.to(device)
        dy = self._dy.to(device)
        dist2 = dx**2 + dy**2
        sigma = F.softplus(self.raw_sigma_iso).view(1, self.R, 1, 1) + 1e-4
        return torch.exp(-dist2 / (sigma**2))

    def _update_regularization_terms(
        self,
        mu: torch.Tensor,
        gap: torch.Tensor,
        geometry_smoothness: torch.Tensor,
    ) -> None:
        mu_clamped = mu.clamp_min(1e-8)
        entropy = -(mu_clamped * mu_clamped.log()).sum(dim=1).mean()

        smoothness = mu.new_zeros(())
        if mu.shape[-1] > 1:
            smoothness = smoothness + (mu[:, :, :, 1:] - mu[:, :, :, :-1]).abs().mean()
        if mu.shape[-2] > 1:
            smoothness = smoothness + (mu[:, :, 1:, :] - mu[:, :, :-1, :]).abs().mean()

        if self.cfg.use_anisotropy:
            hyper_penalty = F.relu(self.cfg.min_hyperbolicity - gap).pow(2).mean()
            anisotropy_gap = gap.mean()
        else:
            hyper_penalty = mu.new_zeros(())
            anisotropy_gap = mu.new_zeros(())

        self._last_reg_terms = {
            "membership_entropy": entropy,
            "membership_smoothness": smoothness,
            "geometry_smoothness": geometry_smoothness,
            "hyperbolicity_penalty": hyper_penalty,
            "anisotropy_gap": anisotropy_gap,
        }

    def regularization_terms(self) -> dict[str, torch.Tensor]:
        if self._last_reg_terms:
            return self._last_reg_terms

        zero = self.pointwise.weight.new_zeros(())
        return {
            "membership_entropy": zero,
            "membership_smoothness": zero,
            "geometry_smoothness": zero,
            "hyperbolicity_penalty": zero,
            "anisotropy_gap": zero,
        }

    @torch.no_grad()
    def metric_tensor_summary(self) -> dict[str, float | str]:
        """Return compact diagnostics for the quadratic metric used by the layer."""

        summary: dict[str, float | str] = {
            "geometry_mode": self.cfg.geometry_mode if self.cfg.use_anisotropy else "isotropic",
            "normalize_mode": self.cfg.normalize_mode,
            "use_fuzzy": float(bool(self.cfg.use_fuzzy)),
            "use_anisotropy": float(bool(self.cfg.use_anisotropy)),
        }
        if not self.cfg.use_anisotropy:
            summary.update({"metric_min_eig": 1.0, "metric_max_eig": 1.0, "metric_condition": 1.0})
            return summary
        if self.cfg.geometry_mode == "local_hyperbolic":
            reg = self.regularization_terms()
            summary["anisotropy_gap"] = float(reg["anisotropy_gap"].detach().cpu())
            return summary

        device = self.pointwise.weight.device
        u_x, u_y, s_x, s_y = self._direction_components_global(device)
        sigma_u, sigma_s, gap = self._global_anisotropic_sigmas()
        u = torch.stack([u_x.reshape(-1), u_y.reshape(-1)], dim=-1)
        s = torch.stack([s_x.reshape(-1), s_y.reshape(-1)], dim=-1)
        inv_u = 1.0 / sigma_u.reshape(-1).pow(2)
        inv_s = 1.0 / sigma_s.reshape(-1).pow(2)
        metric = inv_u[:, None, None] * (u[:, :, None] @ u[:, None, :])
        metric = metric + inv_s[:, None, None] * (s[:, :, None] @ s[:, None, :])
        eigvals = torch.linalg.eigvalsh(metric).clamp_min(1e-12)
        summary.update(
            {
                "metric_min_eig": float(eigvals[:, 0].min().detach().cpu()),
                "metric_max_eig": float(eigvals[:, 1].max().detach().cpu()),
                "metric_condition": float((eigvals[:, 1] / eigvals[:, 0]).max().detach().cpu()),
                "anisotropy_gap": float(gap.mean().detach().cpu()),
            }
        )
        return summary

    def _tensor_to_cpu(self, tensor: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return tensor.detach().to(dtype=dtype, device="cpu")

    def _update_interpretation_cache(
        self,
        mu: torch.Tensor,
        kern: torch.Tensor,
        compat: torch.Tensor,
        interp: dict[str, Any],
    ) -> None:
        kernel_mean = kern[0].mean(dim=-1).view(self.R, self.k, self.k)
        compat_mean = compat[0].mean(dim=-1).view(self.R, self.k, self.k)
        reg = {key: float(value.detach().cpu()) for key, value in self.regularization_terms().items()}
        snapshot: dict[str, Any] = {
            "num_rules": self.R,
            "kernel_size": self.k,
            "geometry_mode": str(interp.get("geometry_mode", self.cfg.geometry_mode)),
            "mu_map": self._tensor_to_cpu(mu[0], dtype=torch.float16),
            "mu_rule_mean": self._tensor_to_cpu(mu[0].mean(dim=(1, 2))),
            "kernel_map": self._tensor_to_cpu(kernel_mean),
            "compat_map": self._tensor_to_cpu(compat_mean),
            "regularization": reg,
        }

        if "u_x" in interp:
            snapshot.update(
                {
                    "u_vec": self._tensor_to_cpu(
                        torch.stack([interp["u_x"][0, :, 0, 0], interp["u_y"][0, :, 0, 0]], dim=1)
                    ),
                    "s_vec": self._tensor_to_cpu(
                        torch.stack([interp["s_x"][0, :, 0, 0], interp["s_y"][0, :, 0, 0]], dim=1)
                    ),
                    "sigma_u": self._tensor_to_cpu(interp["sigma_u"][0, :, 0, 0]),
                    "sigma_s": self._tensor_to_cpu(interp["sigma_s"][0, :, 0, 0]),
                    "gap": self._tensor_to_cpu(interp["gap"].reshape(-1)[: self.R]),
                }
            )

        if "theta_map" in interp:
            snapshot.update(
                {
                    "theta_map": self._tensor_to_cpu(interp["theta_map"][0], dtype=torch.float16),
                    "hyper_map": self._tensor_to_cpu(interp["hyper_map"][0], dtype=torch.float16),
                    "sigma_u_map": self._tensor_to_cpu(interp["sigma_u_map"][0], dtype=torch.float16),
                    "sigma_s_map": self._tensor_to_cpu(interp["sigma_s_map"][0], dtype=torch.float16),
                }
            )

        self._last_interpretation = snapshot

    def interpretation_snapshot(self) -> dict[str, Any]:
        return self._last_interpretation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        patch_area = self.k * self.k
        locations = height * width

        values = self.value_conv(x)
        v_unfold = F.unfold(values, kernel_size=self.k, padding=self.pad, stride=1)
        assert v_unfold.shape == (batch, channels * patch_area, locations), f"unfold shape {v_unfold.shape}"
        v_un = v_unfold.view(batch, channels, patch_area, locations)

        logits = self.gate_conv(x)
        if self.cfg.use_fuzzy:
            mu = F.softmax(logits / float(self.cfg.fuzzy_temperature), dim=1)
        else:
            mu = torch.full_like(logits, 1.0 / self.R)

        mu_unfold = F.unfold(mu, kernel_size=self.k, padding=self.pad, stride=1)
        assert mu_unfold.shape == (batch, self.R * patch_area, locations)
        mu_un = mu_unfold.view(batch, self.R, patch_area, locations)
        valid_un = F.unfold(
            torch.ones(batch, 1, height, width, device=x.device, dtype=x.dtype),
            kernel_size=self.k,
            padding=self.pad,
            stride=1,
        ).view(batch, 1, patch_area, locations)

        center_index = (self.k // 2) * self.k + (self.k // 2)
        mu_center = mu_un[:, :, center_index : center_index + 1, :]

        if self.cfg.use_anisotropy:
            if self.cfg.geometry_mode == "local_hyperbolic":
                kern, gap, geom_smoothness, interp = self._local_hyperbolic_kernel(x)
            else:
                kern, gap, geom_smoothness, interp = self._global_anisotropic_kernel(x.device)
        else:
            kern = self._isotropic_kernel(x.device)
            gap = x.new_zeros(1, self.R, 1, 1)
            geom_smoothness = x.new_zeros(())
            interp = {
                "geometry_mode": "isotropic",
                "sigma_u": torch.ones_like(gap),
                "sigma_s": torch.ones_like(gap),
                "gap": gap.detach(),
            }

        compat = mu_center * mu_un * kern * valid_un
        if self.cfg.compatibility_floor > 0.0:
            compat = compat + float(self.cfg.compatibility_floor) * valid_un
        if self.cfg.normalize_kernel:
            if self.cfg.normalize_mode == "global":
                compat_sum = compat.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
                compat = compat / compat_sum
            elif self.cfg.normalize_mode == "per_rule":
                # Normalize neighbors within each rule while preserving per-rule
                # center mass. This keeps fuzzy rule weighting and prevents
                # geometry from being washed out by cross-rule global scaling.
                neighbor_sum = compat.sum(dim=2, keepdim=True).clamp_min(1e-8)
                compat = (compat / neighbor_sum) * mu_center
            # normalize_mode == "none" -> keep raw compatibilities unchanged.

        self._update_regularization_terms(mu, gap, geom_smoothness)
        self._update_interpretation_cache(mu, kern, compat, interp)
        agg = torch.einsum("brsl,bcsl->brcl", compat, v_un)
        agg_flat = agg.reshape(batch, self.R * channels, height, width)
        out = self.pointwise(agg_flat)
        if self.use_input_residual and self.residual_logit is not None:
            alpha = torch.sigmoid(self.residual_logit)
            out = out + alpha * self.residual_proj(x)
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
    """CIFAR-sized CNN: two AZConv stages + one standard 3x3 stage + classifier."""

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

    def regularization_terms(self) -> dict[str, torch.Tensor]:
        totals: dict[str, torch.Tensor] = {}
        for module in self.modules():
            if isinstance(module, AZConv2d):
                for key, value in module.regularization_terms().items():
                    if key in totals:
                        totals[key] = totals[key] + value
                    else:
                        totals[key] = value
        return totals

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
