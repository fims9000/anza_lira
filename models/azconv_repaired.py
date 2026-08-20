"""Ambiguity-gated residual axial transport on an unchanged AZConv2d base.

The repaired branch is deliberately isolated from the frozen v2 implementation.
It has no persistent directional half states and applies source membership once,
when the initial mode states are created.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .azconv import AZConv2d, AZConvConfig
from .azconv_v2 import orientation_compatibility, paired_sigmas


@dataclass(frozen=True)
class RepairedAZConvConfig:
    num_modes: int = 4
    state_channels: int | None = None
    transport_steps: int = 1
    routing_kernel_size: int = 5
    use_ambiguity_gate: bool = True
    kappa_theta: float = 4.0
    kappa_displacement: float = 4.0
    max_hyperbolicity: float = 1.0
    ambiguity_sharpness_initial: float = 8.0
    ambiguity_threshold_initial: float = 0.25
    residual_lambda_max: float = 1.0
    epsilon: float = 1e-8


def ambiguity_components(
    membership: torch.Tensor,
    theta: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mode diversity D, angular diversity A, and J=D*A in [0,1]."""
    if membership.shape != theta.shape or membership.ndim != 4:
        raise ValueError("membership and theta must be matching BxRxHxW tensors")
    modes = membership.shape[1]
    mu = membership / membership.sum(dim=1, keepdim=True).clamp_min(epsilon)
    if modes <= 1:
        zero = mu[:, 0].new_zeros(mu.shape[0], *mu.shape[2:])
        return zero, zero, zero
    diversity = ((1.0 - mu.square().sum(dim=1)) / (1.0 - 1.0 / modes)).clamp(0.0, 1.0)
    angular_numerator = mu[:, 0].new_zeros(mu.shape[0], *mu.shape[2:])
    angular_denominator = torch.zeros_like(angular_numerator)
    for first in range(modes):
        for second in range(first + 1, modes):
            weight = mu[:, first] * mu[:, second]
            angular_numerator = angular_numerator + weight * torch.sin(
                theta[:, first] - theta[:, second]
            ).square()
            angular_denominator = angular_denominator + weight
    angular = (angular_numerator / angular_denominator.clamp_min(epsilon)).clamp(0.0, 1.0)
    return diversity, angular, (diversity * angular).clamp(0.0, 1.0)


class _AxialModeUpdate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.self_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.message_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(1, channels)
        self.update_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, state: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        batch, modes, channels, height, width = state.shape
        flat_state = state.reshape(batch * modes, channels, height, width)
        flat_message = message.reshape_as(flat_state)
        update = self.norm(
            self.self_projection(flat_state) + self.message_projection(flat_message)
        )
        update = F.gelu(update).reshape_as(state)
        return state + torch.sigmoid(self.update_logit) * update


class AmbiguityGatedResidualANZA(nn.Module):
    """AZConv2d v1 plus a sparse-use, zero-scaled multimode correction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_rules: int = 4,
        *,
        base_cfg: AZConvConfig | None = None,
        cfg: RepairedAZConvConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or RepairedAZConvConfig(num_modes=num_rules)
        if self.cfg.num_modes != num_rules:
            raise ValueError("num_rules must equal cfg.num_modes")
        if self.cfg.routing_kernel_size <= 0 or self.cfg.routing_kernel_size % 2 != 1:
            raise ValueError("routing_kernel_size must be a positive odd integer")
        if self.cfg.transport_steps <= 0:
            raise ValueError("transport_steps must be positive")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(self.cfg.num_modes)
        self.state_channels = int(self.cfg.state_channels or out_channels)
        self.routing_kernel_size = int(self.cfg.routing_kernel_size)
        self.routing_padding = self.routing_kernel_size // 2

        self.base = AZConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            num_rules=num_rules,
            cfg=base_cfg or AZConvConfig(),
        )
        self.membership_head = nn.Conv2d(in_channels, self.modes, 1)
        self.geometry_head = nn.Conv2d(in_channels, 3 * self.modes, 1)
        self.value_projection = nn.Conv2d(in_channels, self.state_channels, 1, bias=False)
        self.updates = nn.ModuleList(
            [_AxialModeUpdate(self.state_channels) for _ in range(self.cfg.transport_steps)]
        )
        self.delta_projection = nn.Conv2d(self.state_channels, out_channels, 1, bias=False)
        # lambda=0 alone gives exact base equivalence while retaining a non-zero
        # wake-up gradient. Zeroing both lambda and W_delta would deadlock the branch.
        nn.init.kaiming_uniform_(self.delta_projection.weight, a=math.sqrt(5))
        self.residual_lambda_raw = nn.Parameter(torch.zeros(()))
        self.ambiguity_sharpness_raw = nn.Parameter(
            torch.tensor(self._inverse_softplus(self.cfg.ambiguity_sharpness_initial))
        )
        threshold = min(max(float(self.cfg.ambiguity_threshold_initial), 1e-4), 1.0 - 1e-4)
        self.ambiguity_threshold_logit = nn.Parameter(torch.logit(torch.tensor(threshold)))
        self._initialize_geometry()
        offsets = [
            (column - self.routing_padding, row - self.routing_padding)
            for row in range(self.routing_kernel_size)
            for column in range(self.routing_kernel_size)
        ]
        self.register_buffer("offset_xy", torch.tensor(offsets, dtype=torch.float32))

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        tensor = torch.tensor(float(value)).clamp_min(1e-6)
        return float(tensor + torch.log(-torch.expm1(-tensor)))

    def _initialize_geometry(self) -> None:
        nn.init.zeros_(self.membership_head.weight)
        nn.init.zeros_(self.membership_head.bias)
        nn.init.zeros_(self.geometry_head.weight)
        nn.init.zeros_(self.geometry_head.bias)
        theta = torch.linspace(0.0, math.pi * (self.modes - 1) / self.modes, self.modes)
        with torch.no_grad():
            self.geometry_head.bias[: self.modes].copy_(theta)
            self.geometry_head.bias[self.modes : 2 * self.modes].fill_(0.5413)
            self.geometry_head.bias[2 * self.modes :].fill_(-1.5078)

    @property
    def residual_lambda(self) -> torch.Tensor:
        return float(self.cfg.residual_lambda_max) * torch.tanh(self.residual_lambda_raw)

    def geometry(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        membership = torch.softmax(self.membership_head(x), dim=1)
        theta, raw_base, raw_hyper = torch.chunk(self.geometry_head(x), 3, dim=1)
        base = F.softplus(raw_base) + 1e-4
        hyper = F.softplus(raw_hyper).clamp_max(float(self.cfg.max_hyperbolicity))
        sigma_u, sigma_s = paired_sigmas(base, hyper)
        diversity, angular, ambiguity = ambiguity_components(
            membership, theta, epsilon=self.cfg.epsilon
        )
        sharpness = F.softplus(self.ambiguity_sharpness_raw)
        threshold = torch.sigmoid(self.ambiguity_threshold_logit)
        learned_gate = torch.sigmoid(sharpness * (ambiguity - threshold))
        gate = learned_gate if self.cfg.use_ambiguity_gate else torch.ones_like(learned_gate)
        return {
            "membership": membership,
            "theta": theta,
            "base_scale": base,
            "hyperbolicity": hyper,
            "sigma_u": sigma_u,
            "sigma_s": sigma_s,
            "mode_diversity": diversity,
            "angular_diversity": angular,
            "ambiguity": ambiguity,
            "ambiguity_gate": gate,
            "learned_ambiguity_gate": learned_gate,
            "ambiguity_sharpness": sharpness,
            "ambiguity_threshold": threshold,
        }

    def _unfold(self, field: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = field.shape
        return F.unfold(
            field,
            self.routing_kernel_size,
            padding=self.routing_padding,
        ).reshape(batch, channels, self.routing_kernel_size**2, height * width)

    def _transition(self, geometry: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        membership = geometry["membership"]
        theta = geometry["theta"]
        sigma_u = geometry["sigma_u"]
        sigma_s = geometry["sigma_s"]
        batch, _modes, height, width = membership.shape
        patch_area = self.routing_kernel_size**2
        locations = height * width

        mu_destination = membership.reshape(batch, self.modes, 1, 1, locations)
        theta_destination = theta.reshape(batch, self.modes, 1, 1, locations)
        theta_source_unfolded = self._unfold(theta)
        theta_source = theta_source_unfolded.reshape(batch, 1, self.modes, patch_area, locations)

        dx = self.offset_xy[:, 0].view(1, 1, patch_area, 1)
        dy = self.offset_xy[:, 1].view(1, 1, patch_area, 1)
        offset_norm = torch.sqrt(dx.square() + dy.square()).clamp_min(1.0)
        unit_dx = dx / offset_norm
        unit_dy = dy / offset_norm
        center = (dx == 0) & (dy == 0)

        theta_dst_flat = theta.reshape(batch, self.modes, 1, locations)
        dst_dot = (
            torch.cos(theta_dst_flat) * unit_dx + torch.sin(theta_dst_flat) * unit_dy
        ).abs()
        dst_dot = torch.where(center, torch.ones_like(dst_dot), dst_dot)
        src_dot = (
            torch.cos(theta_source_unfolded) * unit_dx
            + torch.sin(theta_source_unfolded) * unit_dy
        ).abs()
        src_dot = torch.where(center, torch.ones_like(src_dot), src_dot)
        displacement = torch.exp(
            -float(self.cfg.kappa_displacement) * (1.0 - dst_dot).square()
        ).reshape(batch, self.modes, 1, patch_area, locations)
        displacement = displacement * torch.exp(
            -float(self.cfg.kappa_displacement) * (1.0 - src_dot).square()
        ).reshape(batch, 1, self.modes, patch_area, locations)

        projection_u_dst = torch.cos(theta_dst_flat) * dx + torch.sin(theta_dst_flat) * dy
        projection_s_dst = -torch.sin(theta_dst_flat) * dx + torch.cos(theta_dst_flat) * dy
        sigma_u_dst = sigma_u.reshape(batch, self.modes, 1, locations)
        sigma_s_dst = sigma_s.reshape(batch, self.modes, 1, locations)
        geometry_dst = torch.exp(
            -0.5 * (projection_u_dst / sigma_u_dst).square()
            -0.5 * (projection_s_dst / sigma_s_dst).square()
        ).reshape(batch, self.modes, 1, patch_area, locations)

        sigma_u_src = self._unfold(sigma_u).clamp_min(float(self.cfg.epsilon))
        sigma_s_src = self._unfold(sigma_s).clamp_min(float(self.cfg.epsilon))
        projection_u_src = torch.cos(theta_source_unfolded) * dx + torch.sin(theta_source_unfolded) * dy
        projection_s_src = -torch.sin(theta_source_unfolded) * dx + torch.cos(theta_source_unfolded) * dy
        geometry_src = torch.exp(
            -0.5 * (projection_u_src / sigma_u_src).square()
            -0.5 * (projection_s_src / sigma_s_src).square()
        ).reshape(batch, 1, self.modes, patch_area, locations)
        symmetric_geometry = torch.sqrt(
            (geometry_dst * geometry_src).clamp_min(float(self.cfg.epsilon))
        )
        axial = orientation_compatibility(
            theta_destination, theta_source, self.cfg.kappa_theta
        )
        valid_source = self._unfold(torch.ones_like(membership[:, :1])).reshape(
            batch, 1, 1, patch_area, locations
        )
        raw = mu_destination * symmetric_geometry * axial * displacement * valid_source

        source_patch_mass = raw.sum(dim=1).reshape(
            batch * self.modes, patch_area, locations
        )
        source_mass = F.fold(
            source_patch_mass,
            output_size=(height, width),
            kernel_size=self.routing_kernel_size,
            padding=self.routing_padding,
        ).reshape(batch, self.modes, height, width)
        denominator = self._unfold(source_mass).reshape(
            batch, 1, self.modes, patch_area, locations
        )
        transition = raw / denominator.clamp_min(float(self.cfg.epsilon))
        normalized_source_mass = F.fold(
            transition.sum(dim=1).reshape(batch * self.modes, patch_area, locations),
            output_size=(height, width),
            kernel_size=self.routing_kernel_size,
            padding=self.routing_padding,
        ).reshape(batch, self.modes, height, width)
        return transition, normalized_source_mass

    def _aggregate(self, state: torch.Tensor, transition: torch.Tensor) -> torch.Tensor:
        batch, modes, channels, height, width = state.shape
        source = F.unfold(
            state.reshape(batch * modes, channels, height, width),
            self.routing_kernel_size,
            padding=self.routing_padding,
        ).reshape(batch, modes, channels, self.routing_kernel_size**2, height * width)
        message = torch.einsum("brskl,bsckl->brcl", transition, source)
        return message.reshape(batch, modes, channels, height, width)

    def forward(self, x: torch.Tensor, *, return_diagnostics: bool = False) -> torch.Tensor | dict[str, Any]:
        if x.ndim != 4:
            raise ValueError("AmbiguityGatedResidualANZA input must be BxCxHxW")
        base_output = self.base(x)
        geometry = self.geometry(x)
        value = self.value_projection(x)
        initial_state = geometry["membership"].unsqueeze(2) * value.unsqueeze(1)
        transition, source_mass = self._transition(geometry)
        state = initial_state
        for update in self.updates:
            state = update(state, self._aggregate(state, transition))
        routed = state.sum(dim=1)
        delta = self.delta_projection(routed - value)
        correction = self.residual_lambda * geometry["ambiguity_gate"].unsqueeze(1) * delta
        output = base_output + correction
        if not return_diagnostics:
            return output
        return {
            "output": output,
            "base_output": base_output,
            "correction": correction,
            "delta": delta,
            "residual_lambda": self.residual_lambda,
            "initial_state": initial_state,
            "mode_states": state,
            "transport": transition,
            "transport_source_mass": source_mass,
            **geometry,
        }
