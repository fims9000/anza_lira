"""Context-aware residual ANZA repair for the frozen B1-B3 cycle.

The proven AZConv2d v1 path remains untouched.  Only membership, geometry, and
the direct ambiguity gate receive an effective 5x5 context field; transport
itself remains 3x3.  Axial orientation is represented by normalized
``(cos(2 theta), sin(2 theta))`` rather than a directed angle.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .azconv_repaired import (
    AmbiguityGatedResidualANZA,
    RepairedAZConvConfig,
    ambiguity_components,
)


def doubled_angle_vector(theta: torch.Tensor) -> torch.Tensor:
    """Return the sign-invariant axial representation [cos(2θ), sin(2θ)]."""
    angle = torch.as_tensor(theta)
    return torch.stack((torch.cos(2.0 * angle), torch.sin(2.0 * angle)), dim=-1)


class ContextGatedResidualANZA(AmbiguityGatedResidualANZA):
    """A3 residual transport with contextual heads and a direct learned gate."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_rules: int = 4,
        *,
        base_cfg=None,
        cfg: RepairedAZConvConfig | None = None,
    ) -> None:
        context_cfg = cfg or RepairedAZConvConfig(
            num_modes=num_rules,
            routing_kernel_size=3,
            use_ambiguity_gate=True,
        )
        if context_cfg.routing_kernel_size != 3:
            raise ValueError("Context repair freezes transport at 3x3")
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            num_rules=num_rules,
            base_cfg=base_cfg,
            cfg=context_cfg,
        )
        channels = int(out_channels)
        self.context_dw1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.context_dw2 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.context_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.membership_head = nn.Conv2d(channels, self.modes, 1)
        self.geometry_head = nn.Conv2d(channels, 4 * self.modes, 1)
        self.direct_gate_head = nn.Conv2d(channels, 1, 1)
        self._initialize_context_heads()

    def _initialize_context_heads(self) -> None:
        nn.init.dirac_(self.context_dw1.weight, groups=self.out_channels)
        nn.init.dirac_(self.context_dw2.weight, groups=self.out_channels)
        nn.init.dirac_(self.context_projection.weight)
        nn.init.zeros_(self.membership_head.weight)
        nn.init.zeros_(self.membership_head.bias)
        nn.init.zeros_(self.geometry_head.weight)
        nn.init.zeros_(self.geometry_head.bias)
        nn.init.zeros_(self.direct_gate_head.weight)
        nn.init.constant_(self.direct_gate_head.bias, -2.0)
        angles = torch.linspace(0.0, math.pi * (self.modes - 1) / self.modes, self.modes)
        with torch.no_grad():
            self.geometry_head.bias[: self.modes].copy_(torch.cos(2.0 * angles))
            self.geometry_head.bias[self.modes : 2 * self.modes].copy_(torch.sin(2.0 * angles))
            self.geometry_head.bias[2 * self.modes : 3 * self.modes].fill_(0.5413)
            self.geometry_head.bias[3 * self.modes :].fill_(-1.5078)

    def context_features(self, base_features: torch.Tensor) -> torch.Tensor:
        context = F.gelu(self.context_dw1(base_features))
        context = F.gelu(self.context_dw2(context))
        return self.context_projection(context)

    def geometry(self, base_features: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self.context_features(base_features)
        membership = torch.softmax(self.membership_head(context), dim=1)
        raw_cos2, raw_sin2, raw_base, raw_hyper = torch.chunk(
            self.geometry_head(context), 4, dim=1
        )
        norm = torch.sqrt(raw_cos2.square() + raw_sin2.square()).clamp_min(self.cfg.epsilon)
        cos2 = raw_cos2 / norm
        sin2 = raw_sin2 / norm
        theta = torch.remainder(0.5 * torch.atan2(sin2, cos2), math.pi)
        base = F.softplus(raw_base) + 1e-4
        hyper = F.softplus(raw_hyper).clamp_max(float(self.cfg.max_hyperbolicity))
        from .azconv_v2 import paired_sigmas

        sigma_u, sigma_s = paired_sigmas(base, hyper)
        diversity, angular, ambiguity = ambiguity_components(
            membership, theta, epsilon=self.cfg.epsilon
        )
        direct_gate = torch.sigmoid(self.direct_gate_head(context)).squeeze(1)
        gate = direct_gate if self.cfg.use_ambiguity_gate else torch.ones_like(direct_gate)
        return {
            "context_features": context,
            "membership": membership,
            "theta": theta,
            "orientation_cos2": cos2,
            "orientation_sin2": sin2,
            "base_scale": base,
            "hyperbolicity": hyper,
            "sigma_u": sigma_u,
            "sigma_s": sigma_s,
            "mode_diversity": diversity,
            "angular_diversity": angular,
            "ambiguity": ambiguity,
            "ambiguity_gate": gate,
            "learned_ambiguity_gate": direct_gate,
        }

    def forward(self, x: torch.Tensor, *, return_diagnostics: bool = False) -> torch.Tensor | dict[str, Any]:
        if x.ndim != 4:
            raise ValueError("ContextGatedResidualANZA input must be BxCxHxW")
        base_output = self.base(x)
        geometry = self.geometry(base_output)
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


def context_head_parameter_count(module: ContextGatedResidualANZA) -> int:
    names = (
        "context_dw1",
        "context_dw2",
        "context_projection",
        "membership_head",
        "geometry_head",
        "direct_gate_head",
    )
    return sum(parameter.numel() for name in names for parameter in getattr(module, name).parameters())


def context_head_macs_per_pixel(module: ContextGatedResidualANZA) -> int:
    channels = module.out_channels
    return int(2 * 9 * channels + channels * channels + channels * module.modes + channels * 4 * module.modes + channels)
