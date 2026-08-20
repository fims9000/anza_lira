"""Independent-fuzzy ANZA and direct structural-affinity modulation.

The published/current :class:`AZConv2d` is deliberately left untouched.  This
module is the isolated C1--C3 repair stream.  C1 replaces categorical softmax
memberships by independent fuzzy degrees.  C2/C3 change the *normalized ANZA
interaction weight* in log space; they do not add a second segmentation path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .azconv import AZConv2d, AZConvConfig


LOCAL8_OFFSETS = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
RADIUS2_OFFSETS = ((-2, -2), (0, -2), (2, -2), (-2, 0), (2, 0), (-2, 2), (0, 2), (2, 2))


@dataclass
class AffinityAZConvConfig:
    context_channels: int = 8
    beta_max: float = 4.0
    use_radius2_supervision: bool = False


def _shift_tensor(x: torch.Tensor, dx: int, dy: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return q=p+(dx,dy), padded with zero, and an in-bounds mask."""

    height, width = x.shape[-2:]
    shifted = torch.zeros_like(x)
    valid = torch.zeros((1, 1, height, width), device=x.device, dtype=x.dtype)
    dst_y0, dst_y1 = max(0, -dy), min(height, height - dy)
    dst_x0, dst_x1 = max(0, -dx), min(width, width - dx)
    src_y0, src_y1 = max(0, dy), min(height, height + dy)
    src_x0, src_x1 = max(0, dx), min(width, width + dx)
    shifted[..., dst_y0:dst_y1, dst_x0:dst_x1] = x[..., src_y0:src_y1, src_x0:src_x1]
    valid[..., dst_y0:dst_y1, dst_x0:dst_x1] = 1.0
    return shifted, valid


class IndependentFuzzyAZConv2d(AZConv2d):
    """C1: the v1 pair weight with independent sigmoid memberships."""

    def _memberships(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate_conv(x)
        if self.cfg.use_fuzzy:
            return torch.sigmoid(logits / float(self.cfg.fuzzy_temperature))
        return torch.ones_like(logits)

    def _base_terms(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        batch, _channels, height, width = x.shape
        patch_area = self.k * self.k
        locations = height * width
        mu = self._memberships(x)
        mu_un = F.unfold(mu, kernel_size=self.k, padding=self.pad).view(
            batch, self.R, patch_area, locations
        )
        center_index = (self.k // 2) * self.k + self.k // 2
        mu_center = mu_un[:, :, center_index : center_index + 1, :]
        valid = F.unfold(
            torch.ones(batch, 1, height, width, device=x.device, dtype=x.dtype),
            kernel_size=self.k,
            padding=self.pad,
        ).view(batch, 1, patch_area, locations)
        if self.cfg.use_anisotropy:
            if self.cfg.geometry_mode == "local_hyperbolic":
                kernel, gap, smoothness, interp = self._local_hyperbolic_kernel(x)
            else:
                kernel, gap, smoothness, interp = self._global_anisotropic_kernel(x.device)
        else:
            kernel = self._isotropic_kernel(x.device)
            gap = x.new_zeros(1, self.R, 1, 1)
            smoothness = x.new_zeros(())
            interp = {"geometry_mode": "isotropic", "gap": gap.detach()}
        raw = mu_center * mu_un * kernel * valid
        if self.cfg.compatibility_floor > 0.0:
            raw = raw + float(self.cfg.compatibility_floor) * valid
        return raw, valid, mu, gap, {**interp, "geometry_smoothness": smoothness, "kernel": kernel}

    def _normalize(self, raw: torch.Tensor, valid: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        if not self.cfg.normalize_kernel or self.cfg.normalize_mode == "none":
            return raw
        if self.cfg.normalize_mode == "global":
            return raw / raw.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        center = (self.k // 2) * self.k + self.k // 2
        mu_center = F.unfold(mu, kernel_size=self.k, padding=self.pad).view(
            mu.shape[0], self.R, self.k * self.k, -1
        )[:, :, center : center + 1]
        return raw / raw.sum(dim=2, keepdim=True).clamp_min(1e-8) * mu_center

    def _finish(
        self,
        x: torch.Tensor,
        compat: torch.Tensor,
        mu: torch.Tensor,
        gap: torch.Tensor,
        interp: dict[str, Any],
    ) -> torch.Tensor:
        batch, channels, height, width = x.shape
        values = self.value_conv(x)
        value_un = F.unfold(values, kernel_size=self.k, padding=self.pad).view(
            batch, channels, self.k * self.k, height * width
        )
        # The legacy entropy regularizer expects a categorical distribution;
        # use a diagnostic-only normalized copy without changing fuzzy weights.
        mu_diag = mu / mu.sum(dim=1, keepdim=True).clamp_min(1e-8)
        self._update_regularization_terms(
            mu_diag, gap, interp["geometry_smoothness"], interp
        )
        self._last_live_interpretation = {
            "mu": mu,
            "theta_map": interp.get("theta_map"),
        }
        self._update_interpretation_cache(mu, interp["kernel"], compat, interp)
        agg = torch.einsum("brsl,bcsl->brcl", compat, value_un)
        out = self.pointwise(agg.reshape(batch, self.R * channels, height, width))
        if self.use_input_residual and self.residual_logit is not None:
            out = out + torch.sigmoid(self.residual_logit) * self.residual_proj(x)
        return out

    def forward(self, x: torch.Tensor, *, return_diagnostics: bool = False) -> torch.Tensor | dict[str, Any]:
        raw, valid, mu, gap, interp = self._base_terms(x)
        weights = self._normalize(raw, valid, mu)
        output = self._finish(x, weights, mu, gap, interp)
        if not return_diagnostics:
            return output
        return {"output": output, "memberships": mu, "weights": weights}


class StructuralAffinityAZConv2d(IndependentFuzzyAZConv2d):
    """C2/C3: symmetric pair affinity directly biases ANZA interaction weights."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_rules: int = 4,
        cfg: AZConvConfig | None = None,
        affinity_cfg: AffinityAZConvConfig | None = None,
        bias: bool = True,
    ) -> None:
        if kernel_size != 3:
            raise ValueError("StructuralAffinityAZConv2d currently fixes the ANZA support to local 3x3")
        super().__init__(in_channels, out_channels, kernel_size, num_rules, cfg, bias)
        self.affinity_cfg = affinity_cfg or AffinityAZConvConfig()
        context_channels = int(self.affinity_cfg.context_channels)
        if context_channels <= 0:
            raise ValueError("context_channels must be positive")
        self.context_encoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, context_channels, 1),
            nn.GELU(),
            nn.Conv2d(context_channels, context_channels, 3, padding=1, groups=context_channels, bias=False),
            nn.Conv2d(context_channels, context_channels, 1),
            nn.GELU(),
        )
        pair_features = 3 * context_channels + 5
        hidden = max(16, 2 * context_channels)
        self.affinity_mlp = nn.Sequential(nn.Linear(pair_features, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.raw_beta = nn.Parameter(torch.zeros(()))
        self._last_affinity: dict[str, torch.Tensor] = {}

    @property
    def beta(self) -> torch.Tensor:
        raw_nonnegative = torch.clamp_min(self.raw_beta, 0.0)
        # Exactly zero at initialization, positive derivative at the boundary,
        # and never semantically inverts a supervised compatibility score.
        centered_softplus = F.softplus(raw_nonnegative) - math.log(2.0)
        return float(self.affinity_cfg.beta_max) * centered_softplus

    def set_affinity_enabled(self, enabled: bool) -> None:
        self._affinity_enabled = bool(enabled)

    def _theta_map_live(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.geometry_mode == "local_hyperbolic":
            assert self.geometry_conv is not None
            theta, _base, _hyper = torch.chunk(self.geometry_conv(x), 3, dim=1)
            return theta
        if self.cfg.geometry_mode in {"learned_angle", "learned_hyperbolic"}:
            return self.theta.view(1, self.R, 1, 1).expand(x.shape[0], -1, x.shape[2], x.shape[3])
        angle = torch.atan2(self._cat_unstable[1], self._cat_unstable[0])
        return angle.view(1, 1, 1, 1).expand(x.shape[0], self.R, x.shape[2], x.shape[3])

    def _pair_scores(
        self, x: torch.Tensor, offsets: Iterable[tuple[int, int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.context_encoder(x)
        theta = self._theta_map_live(x)
        scores: list[torch.Tensor] = []
        validity: list[torch.Tensor] = []
        for dx, dy in offsets:
            neighbor, valid = _shift_tensor(context, int(dx), int(dy))
            theta_q, _ = _shift_tensor(theta, int(dx), int(dy))
            diff = (context - neighbor).abs()
            # Symmetry is architectural: both endpoint orders pass through the
            # same MLP and are averaged.  Geometry uses axial, sign-free terms.
            axial = torch.cos(2.0 * (theta - theta_q))
            # Average *axes*, never directed arrows.  Doubled-angle averaging
            # makes every pair feature invariant under theta -> theta + pi.
            pair_c2 = torch.cos(2.0 * theta) + torch.cos(2.0 * theta_q)
            pair_s2 = torch.sin(2.0 * theta) + torch.sin(2.0 * theta_q)
            pair_norm = torch.sqrt(pair_c2.square() + pair_s2.square()).clamp_min(1e-6)
            pair_theta = 0.5 * torch.atan2(pair_s2 / pair_norm, pair_c2 / pair_norm)
            axis_x = torch.cos(pair_theta)
            axis_y = torch.sin(pair_theta)
            disp_norm = max(math.sqrt(float(dx * dx + dy * dy)), 1.0)
            along = (axis_x * (dx / disp_norm) + axis_y * (dy / disp_norm)).abs()
            across = (axis_x * (dy / disp_norm) - axis_y * (dx / disp_norm)).abs()
            distance = context.new_full((x.shape[0], self.R, x.shape[2], x.shape[3]), disp_norm / 2.0)
            cp = context.unsqueeze(1).expand(-1, self.R, -1, -1, -1)
            cq = neighbor.unsqueeze(1).expand_as(cp)
            cd = diff.unsqueeze(1).expand_as(cp)
            geom = torch.stack([axial, along, across, distance, valid.expand_as(axial)], dim=2)
            forward_features = torch.cat([cp, cq, cd, geom], dim=2).permute(0, 1, 3, 4, 2)
            reverse_features = torch.cat([cq, cp, cd, geom], dim=2).permute(0, 1, 3, 4, 2)
            score = 0.5 * (self.affinity_mlp(forward_features) + self.affinity_mlp(reverse_features))
            scores.append(score.squeeze(-1))
            validity.append(valid.expand(x.shape[0], self.R, -1, -1))
        return torch.stack(scores, dim=2), torch.stack(validity, dim=2)

    def edge_logits(self, x: torch.Tensor, *, include_radius2: bool | None = None) -> dict[str, torch.Tensor]:
        use_radius2 = self.affinity_cfg.use_radius2_supervision if include_radius2 is None else include_radius2
        offsets = LOCAL8_OFFSETS + (RADIUS2_OFFSETS if use_radius2 else ())
        by_rule, valid = self._pair_scores(x, offsets)
        aggregate = torch.logsumexp(by_rule, dim=1) - math.log(self.R)
        return {
            "logits": aggregate,
            "logits_by_rule": by_rule,
            "valid": valid[:, 0],
            "offsets_xy": x.new_tensor(offsets, dtype=torch.int64),
        }

    def forward(self, x: torch.Tensor, *, return_diagnostics: bool = False) -> torch.Tensor | dict[str, Any]:
        raw, valid, mu, gap, interp = self._base_terms(x)
        scores8, score_valid = self._pair_scores(x, LOCAL8_OFFSETS)
        local_scores_for_supervision = scores8
        radius2_scores = None
        radius2_valid = None
        if self.affinity_cfg.use_radius2_supervision:
            radius2_scores, radius2_valid = self._pair_scores(x, RADIUS2_OFFSETS)
            # Each sparse radius-2 edge is collinear with its local-8 partner.
            # It provides longer-range structural evidence while aggregation
            # remains the normalized local ANZA interaction.  Thus C3 uses the
            # radius-2 field causally and beta=0 still equals clean C1 exactly.
            scores8 = 0.5 * (scores8 + radius2_scores)
        center = (self.k * self.k) // 2
        scores = raw.new_zeros(raw.shape)
        positions = [index for index in range(self.k * self.k) if index != center]
        scores[:, :, positions] = scores8.reshape(x.shape[0], self.R, 8, -1)
        enabled = getattr(self, "_affinity_enabled", True)
        beta = self.beta if enabled else self.raw_beta.new_zeros(())
        # Algebraically identical to log(w0)+beta*s, but this product form is
        # intentionally used so beta=0 is bit-exact (exp(0)==1) while beta
        # still receives a gradient on the first step.
        modulated = raw * torch.exp(beta * scores) * valid
        # Preserve a neutral self-edge.  Affinity is defined only between two
        # distinct local elements; beta=0 is therefore exactly C1.
        modulated[:, :, center] = raw[:, :, center]
        weights = self._normalize(modulated, valid, mu)
        output = self._finish(x, weights, mu, gap, interp)
        aggregate = torch.logsumexp(scores8, dim=1) - math.log(self.R)
        self._last_affinity = {
            "logits": aggregate,
            "logits_by_rule": scores8,
            "valid": score_valid[:, 0],
            "beta": beta,
            "radius2_logits_by_rule": radius2_scores,
        }
        if not return_diagnostics:
            return output
        radius2 = None
        if self.affinity_cfg.use_radius2_supervision:
            assert radius2_scores is not None and radius2_valid is not None
            all_scores = torch.cat([local_scores_for_supervision, radius2_scores], dim=2)
            all_valid = torch.cat([score_valid, radius2_valid], dim=2)
            radius2 = {
                "logits": torch.logsumexp(all_scores, dim=1) - math.log(self.R),
                "logits_by_rule": all_scores,
                "valid": all_valid[:, 0],
                "offsets_xy": x.new_tensor(LOCAL8_OFFSETS + RADIUS2_OFFSETS, dtype=torch.int64),
            }
        return {
            "output": output,
            "memberships": mu,
            "weights": weights,
            "affinity": self._last_affinity,
            "radius2_affinity": radius2,
        }
