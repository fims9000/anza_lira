"""ANZA-LIRA v2 mode-resolved local structural transport.

The implementation is intentionally separate from :class:`AZConv2d`. It uses
local determinant-one paired expansion/contraction as a geometric
parameterization; it does not claim Anosov dynamics or ergodicity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class AZConvV2Config:
    num_modes: int = 4
    state_channels: int | None = None
    transport_steps: int = 2
    variant: str = "v2a"
    kappa_theta: float = 4.0
    kappa_direction: float = 4.0
    max_hyperbolicity: float = 1.0
    use_fuzzy: bool = True
    use_junction: bool = True
    use_cone: bool = True
    epsilon: float = 1e-8


def axial_distance(theta_a: torch.Tensor, theta_b: torch.Tensor) -> torch.Tensor:
    """Pi-periodic angle distance in [0, pi/2] with stable gradients."""
    delta = theta_a - theta_b
    # Equivalent to 0.5*acos(cos(2*delta)), without the singular acos
    # derivative at exactly aligned self-neighbors.
    return torch.atan2(torch.sin(delta).abs(), torch.cos(delta).abs())


def orientation_compatibility(
    theta_a: torch.Tensor,
    theta_b: torch.Tensor,
    kappa_theta: float,
) -> torch.Tensor:
    return torch.exp(-float(kappa_theta) * torch.sin(theta_a - theta_b).square())


def directional_compatibility(
    direction_xy: torch.Tensor,
    theta: torch.Tensor,
    half_sign: torch.Tensor | float,
    kappa_direction: float,
) -> torch.Tensor:
    """Compatibility of q->p direction with a signed local axial half-mode."""
    tangent_x = torch.cos(theta) * torch.as_tensor(half_sign, device=theta.device, dtype=theta.dtype)
    tangent_y = torch.sin(theta) * torch.as_tensor(half_sign, device=theta.device, dtype=theta.dtype)
    dot = direction_xy[..., 0] * tangent_x + direction_xy[..., 1] * tangent_y
    return torch.exp(-float(kappa_direction) * (1.0 - dot).square())


def paired_sigmas(base: torch.Tensor, hyperbolicity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return base * torch.exp(hyperbolicity), base * torch.exp(-hyperbolicity)


def junction_score(membership: torch.Tensor, theta: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Geometry-native mode and angular diversity score in [0,1]."""
    if membership.shape != theta.shape or membership.ndim != 4:
        raise ValueError("membership and theta must be matching BxRxHxW tensors")
    modes = membership.shape[1]
    normalized = membership / membership.sum(dim=1, keepdim=True).clamp_min(epsilon)
    if modes <= 1:
        return normalized[:, 0].new_zeros(normalized.shape[0], *normalized.shape[2:])
    diversity = (1.0 - normalized.square().sum(dim=1)) / (1.0 - 1.0 / modes)
    angular_numerator = normalized[:, 0].new_zeros(normalized.shape[0], *normalized.shape[2:])
    angular_denominator = torch.zeros_like(angular_numerator)
    for first in range(modes):
        for second in range(first + 1, modes):
            pair_weight = normalized[:, first] * normalized[:, second]
            angular_numerator = angular_numerator + pair_weight * torch.sin(
                theta[:, first] - theta[:, second]
            ).square()
            angular_denominator = angular_denominator + pair_weight
    angular = angular_numerator / angular_denominator.clamp_min(epsilon)
    return (diversity * angular).clamp(0.0, 1.0)


class _ModeUpdate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.self_projection = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.message_projection = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, channels)
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, state: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        batch, modes, channels, height, width = state.shape
        state_flat = state.reshape(batch * modes, channels, height, width)
        message_flat = message.reshape(batch * modes, channels, height, width)
        update = self.norm(self.self_projection(state_flat) + self.message_projection(message_flat))
        update = F.gelu(update).reshape_as(state)
        return state + torch.sigmoid(self.residual_scale) * update


class ModeResolvedAZConv2d(nn.Module):
    """V2A mode-resolved axial transport with delayed fusion."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        cfg: AZConvV2Config | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or AZConvV2Config()
        if kernel_size <= 0 or kernel_size % 2 != 1:
            raise ValueError("ModeResolvedAZConv2d requires an odd positive kernel size")
        if self.cfg.num_modes <= 0 or self.cfg.transport_steps <= 0:
            raise ValueError("num_modes and transport_steps must be positive")
        if self.cfg.variant not in {"v2a", "v2b"}:
            raise ValueError("variant must be v2a or v2b")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(self.cfg.num_modes)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.state_channels = int(self.cfg.state_channels or out_channels)
        self.gate_head = nn.Conv2d(in_channels, self.modes, kernel_size=1)
        self.geometry_head = nn.Conv2d(in_channels, 3 * self.modes, kernel_size=1)
        self.value_projection = nn.Conv2d(in_channels, self.state_channels, kernel_size=1, bias=False)
        self.updates = nn.ModuleList(
            [_ModeUpdate(self.state_channels) for _ in range(int(self.cfg.transport_steps))]
        )
        self.output_projection = nn.Conv2d(self.state_channels, out_channels, kernel_size=1)
        self._initialize_geometry()
        offsets = []
        for row in range(self.kernel_size):
            for column in range(self.kernel_size):
                offsets.append((column - self.padding, row - self.padding))
        self.register_buffer("offset_xy", torch.tensor(offsets, dtype=torch.float32))

    def _initialize_geometry(self) -> None:
        nn.init.zeros_(self.geometry_head.weight)
        nn.init.zeros_(self.geometry_head.bias)
        theta = torch.linspace(0.0, math.pi * (self.modes - 1) / self.modes, self.modes)
        with torch.no_grad():
            self.geometry_head.bias[: self.modes].copy_(theta)
            self.geometry_head.bias[self.modes : 2 * self.modes].fill_(0.5413)  # softplus ~= 1
            self.geometry_head.bias[2 * self.modes :].fill_(-1.5078)  # softplus ~= 0.2

    def geometry(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.gate_head(x)
        if self.cfg.use_fuzzy:
            membership = torch.softmax(logits, dim=1)
        else:
            membership = torch.full_like(logits, 1.0 / self.modes)
        theta, raw_base, raw_hyper = torch.chunk(self.geometry_head(x), 3, dim=1)
        base = F.softplus(raw_base) + 1e-4
        hyper = F.softplus(raw_hyper).clamp_max(float(self.cfg.max_hyperbolicity))
        sigma_u, sigma_s = paired_sigmas(base, hyper)
        return {
            "membership": membership,
            "theta": theta,
            "base": base,
            "hyperbolicity": hyper,
            "sigma_u": sigma_u,
            "sigma_s": sigma_s,
            "junction_score": junction_score(membership, theta, self.cfg.epsilon),
        }

    def _unfold_field(self, field: torch.Tensor) -> torch.Tensor:
        batch, modes, height, width = field.shape
        patch_area = self.kernel_size**2
        return F.unfold(field, self.kernel_size, padding=self.padding).view(
            batch, modes, patch_area, height * width
        )

    def _v2a_transition(self, geometry: dict[str, torch.Tensor]) -> torch.Tensor:
        membership = geometry["membership"]
        theta = geometry["theta"]
        sigma_u = geometry["sigma_u"]
        sigma_s = geometry["sigma_s"]
        batch, _modes, height, width = membership.shape
        locations = height * width
        patch_area = self.kernel_size**2

        mu_destination = membership.reshape(batch, self.modes, 1, 1, locations)
        mu_source = self._unfold_field(membership).reshape(batch, 1, self.modes, patch_area, locations)
        theta_destination = theta.reshape(batch, self.modes, 1, 1, locations)
        theta_source = self._unfold_field(theta).reshape(batch, 1, self.modes, patch_area, locations)

        dx = self.offset_xy[:, 0].view(1, 1, patch_area, 1)
        dy = self.offset_xy[:, 1].view(1, 1, patch_area, 1)
        theta_dst_flat = theta.reshape(batch, self.modes, 1, locations)
        projection_u = torch.cos(theta_dst_flat) * dx + torch.sin(theta_dst_flat) * dy
        projection_s = -torch.sin(theta_dst_flat) * dx + torch.cos(theta_dst_flat) * dy
        sigma_u_dst = sigma_u.reshape(batch, self.modes, 1, locations)
        sigma_s_dst = sigma_s.reshape(batch, self.modes, 1, locations)
        geometry_destination = torch.exp(
            -0.5 * (projection_u / sigma_u_dst).square() - 0.5 * (projection_s / sigma_s_dst).square()
        ).reshape(batch, self.modes, 1, patch_area, locations)

        theta_src_flat = self._unfold_field(theta)
        # ``unfold`` pads field maps with zero. Clamp only the source
        # denominators so padded locations yield finite geometry and are then
        # removed by their zero source membership.
        sigma_u_src = self._unfold_field(sigma_u).clamp_min(float(self.cfg.epsilon))
        sigma_s_src = self._unfold_field(sigma_s).clamp_min(float(self.cfg.epsilon))
        projection_u_src = torch.cos(theta_src_flat) * (-dx) + torch.sin(theta_src_flat) * (-dy)
        projection_s_src = -torch.sin(theta_src_flat) * (-dx) + torch.cos(theta_src_flat) * (-dy)
        geometry_source = torch.exp(
            -0.5 * (projection_u_src / sigma_u_src).square()
            - 0.5 * (projection_s_src / sigma_s_src).square()
        ).reshape(batch, 1, self.modes, patch_area, locations)
        symmetric_geometry = torch.sqrt(
            (geometry_destination * geometry_source).clamp_min(float(self.cfg.epsilon))
        )
        axial = orientation_compatibility(
            theta_destination,
            theta_source,
            self.cfg.kappa_theta,
        )
        raw = mu_destination * mu_source * symmetric_geometry * axial
        return raw / raw.sum(dim=(2, 3), keepdim=True).clamp_min(float(self.cfg.epsilon))

    def _aggregate_v2a(self, state: torch.Tensor, transition: torch.Tensor) -> torch.Tensor:
        batch, modes, channels, height, width = state.shape
        patch_area = self.kernel_size**2
        source = F.unfold(
            state.reshape(batch * modes, channels, height, width),
            self.kernel_size,
            padding=self.padding,
        ).view(batch, modes, channels, patch_area, height * width)
        message = torch.einsum("brskl,bsckl->brcl", transition, source)
        return message.reshape(batch, modes, channels, height, width)

    def _v2b_transition(self, geometry: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        membership = geometry["membership"]
        theta = geometry["theta"]
        sigma_u = geometry["sigma_u"]
        sigma_s = geometry["sigma_s"]
        batch, _modes, height, width = membership.shape
        locations = height * width
        patch_area = self.kernel_size**2

        mu_destination = membership.reshape(batch, self.modes, 1, 1, 1, 1, locations)
        mu_source = self._unfold_field(membership).reshape(
            batch, 1, 1, self.modes, 1, patch_area, locations
        )
        theta_destination = theta.reshape(batch, self.modes, 1, 1, 1, 1, locations)
        theta_source_unfolded = self._unfold_field(theta)
        theta_source = theta_source_unfolded.reshape(
            batch, 1, 1, self.modes, 1, patch_area, locations
        )

        dx = self.offset_xy[:, 0].view(1, 1, patch_area, 1)
        dy = self.offset_xy[:, 1].view(1, 1, patch_area, 1)
        theta_dst_flat = theta.reshape(batch, self.modes, 1, locations)
        projection_u = torch.cos(theta_dst_flat) * dx + torch.sin(theta_dst_flat) * dy
        projection_s = -torch.sin(theta_dst_flat) * dx + torch.cos(theta_dst_flat) * dy
        sigma_u_dst = sigma_u.reshape(batch, self.modes, 1, locations)
        sigma_s_dst = sigma_s.reshape(batch, self.modes, 1, locations)
        geometry_destination = torch.exp(
            -0.5 * (projection_u / sigma_u_dst).square() - 0.5 * (projection_s / sigma_s_dst).square()
        ).reshape(batch, self.modes, 1, 1, 1, patch_area, locations)

        sigma_u_source = self._unfold_field(sigma_u).clamp_min(float(self.cfg.epsilon))
        sigma_s_source = self._unfold_field(sigma_s).clamp_min(float(self.cfg.epsilon))
        projection_u_source = torch.cos(theta_source_unfolded) * (-dx) + torch.sin(theta_source_unfolded) * (-dy)
        projection_s_source = -torch.sin(theta_source_unfolded) * (-dx) + torch.cos(theta_source_unfolded) * (-dy)
        geometry_source = torch.exp(
            -0.5 * (projection_u_source / sigma_u_source).square()
            - 0.5 * (projection_s_source / sigma_s_source).square()
        ).reshape(batch, 1, 1, self.modes, 1, patch_area, locations)
        symmetric_geometry = torch.sqrt(
            (geometry_destination * geometry_source).clamp_min(float(self.cfg.epsilon))
        )
        axial = orientation_compatibility(
            theta_destination,
            theta_source,
            self.cfg.kappa_theta,
        )

        offset_norm = torch.sqrt(
            self.offset_xy[:, 0].square() + self.offset_xy[:, 1].square()
        ).clamp_min(float(self.cfg.epsilon))
        direction_x = (-self.offset_xy[:, 0] / offset_norm).view(1, 1, 1, 1, 1, patch_area, 1)
        direction_y = (-self.offset_xy[:, 1] / offset_norm).view(1, 1, 1, 1, 1, patch_area, 1)
        half_signs = theta.new_tensor((1.0, -1.0))
        destination_sign = half_signs.view(1, 1, 2, 1, 1, 1, 1)
        source_sign = half_signs.view(1, 1, 1, 1, 2, 1, 1)
        destination_dot = destination_sign * (
            direction_x * torch.cos(theta_destination) + direction_y * torch.sin(theta_destination)
        )
        source_dot = source_sign * (
            direction_x * torch.cos(theta_source) + direction_y * torch.sin(theta_source)
        )
        directional_destination = torch.exp(
            -float(self.cfg.kappa_direction) * (1.0 - destination_dot).square()
        )
        directional_source = torch.exp(
            -float(self.cfg.kappa_direction) * (1.0 - source_dot).square()
        )
        raw = (
            mu_destination
            * mu_source
            * symmetric_geometry
            * axial
            * directional_destination
            * directional_source
        )

        # Convert destination-centered patch contributions into one denominator
        # per valid source state (q,s,xi). This is the exact local row-stochastic
        # normalization required by the V2B transport interpretation.
        source_patch_mass = raw.sum(dim=(1, 2)).reshape(
            batch * self.modes * 2, patch_area, locations
        )
        source_mass = F.fold(
            source_patch_mass,
            output_size=(height, width),
            kernel_size=self.kernel_size,
            padding=self.padding,
        ).reshape(batch, self.modes, 2, height, width)
        denominator_patch = F.unfold(
            source_mass.reshape(batch * self.modes * 2, 1, height, width),
            self.kernel_size,
            padding=self.padding,
        ).reshape(batch, self.modes, 2, patch_area, locations)
        transition = raw / denominator_patch.unsqueeze(1).unsqueeze(2).clamp_min(
            float(self.cfg.epsilon)
        )
        normalized_source_mass = F.fold(
            transition.sum(dim=(1, 2)).reshape(batch * self.modes * 2, patch_area, locations),
            output_size=(height, width),
            kernel_size=self.kernel_size,
            padding=self.padding,
        ).reshape(batch, self.modes, 2, height, width)
        return transition, normalized_source_mass

    def _aggregate_v2b(self, state: torch.Tensor, transition: torch.Tensor) -> torch.Tensor:
        batch, modes, halves, channels, height, width = state.shape
        patch_area = self.kernel_size**2
        source = F.unfold(
            state.reshape(batch * modes * halves, channels, height, width),
            self.kernel_size,
            padding=self.padding,
        ).view(batch, modes, halves, channels, patch_area, height * width)
        message = torch.einsum("brhsikl,bsickl->brhcl", transition, source)
        return message.reshape(batch, modes, halves, channels, height, width)

    def forward(self, x: torch.Tensor, *, return_diagnostics: bool = False) -> torch.Tensor | dict[str, Any]:
        if x.ndim != 4:
            raise ValueError("ModeResolvedAZConv2d input must be BxCxHxW")
        geometry = self.geometry(x)
        value = self.value_projection(x)
        if self.cfg.variant == "v2a":
            state = geometry["membership"].unsqueeze(2) * value.unsqueeze(1)
            transition = self._v2a_transition(geometry)
            for update in self.updates:
                state = update(state, self._aggregate_v2a(state, transition))
            fused = (geometry["membership"].unsqueeze(2) * state).sum(dim=1)
            transport_mass = transition.sum(dim=(2, 3))
        else:
            half_membership = (
                0.5 * geometry["membership"].unsqueeze(2).unsqueeze(3)
            ).expand(-1, -1, 2, self.state_channels, -1, -1)
            state = half_membership * value.unsqueeze(1).unsqueeze(2)
            transition, transport_mass = self._v2b_transition(geometry)
            for update in self.updates:
                flat_state = state.reshape(
                    state.shape[0], self.modes * 2, self.state_channels, state.shape[-2], state.shape[-1]
                )
                flat_message = self._aggregate_v2b(state, transition).reshape_as(flat_state)
                state = update(flat_state, flat_message).reshape_as(state)
            fusion_weight = 0.5 * geometry["membership"].unsqueeze(2).unsqueeze(3)
            fused = (fusion_weight * state).sum(dim=(1, 2))
        output = self.output_projection(fused)
        if not return_diagnostics:
            return output
        return {
            "output": output,
            "mode_states": state,
            "membership": geometry["membership"],
            "theta": geometry["theta"],
            "sigma_u": geometry["sigma_u"],
            "sigma_s": geometry["sigma_s"],
            "hyperbolicity": geometry["hyperbolicity"],
            "junction_score": geometry["junction_score"],
            "transport": transition,
            "transport_mass": transport_mass,
            "variant": self.cfg.variant,
        }
