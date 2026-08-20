"""Capacity-matched static-plus-dynamic innovation residuals for ANZA-KIR."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from anza_ks_k2.dense_features import FEATURE_WIDTH, dense_orientation_features


VARIANT_METHOD = {
    "R0_static_residual": "static",
    "R1_shear_ks_residual": "shear_ks",
    "R2_cat_raw_residual": "cat_raw",
    "R3_anza_kir": "cat_ks",
}


class InnovationResidual(nn.Module):
    """Apply a zero-initialized correction based on [Static(E), Dynamic(E)]."""

    def __init__(
        self,
        channels: int,
        variant: str,
        *,
        feature_norm: dict[str, dict[str, torch.Tensor]] | None = None,
    ) -> None:
        super().__init__()
        if variant not in VARIANT_METHOD:
            raise ValueError(f"unknown ANZA-KIR residual variant: {variant}")
        self.variant = variant
        self.dynamic_method = VARIANT_METHOD[variant]
        self.readout = nn.Sequential(nn.Linear(2 * FEATURE_WIDTH, 32), nn.GELU(), nn.Linear(32, 16))
        self.output_projection = nn.Conv2d(16, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(()))
        norm = feature_norm or {}
        for label, method in (("static", "static"), ("dynamic", self.dynamic_method)):
            values = norm.get(method, {})
            mean = torch.as_tensor(values.get("mean", torch.zeros(FEATURE_WIDTH)), dtype=torch.float32)
            std = torch.as_tensor(values.get("std", torch.ones(FEATURE_WIDTH)), dtype=torch.float32)
            if mean.shape != (FEATURE_WIDTH,) or std.shape != (FEATURE_WIDTH,) or torch.any(std <= 0):
                raise ValueError("feature normalization must contain positive 104-vectors")
            self.register_buffer(f"{label}_mean", mean)
            self.register_buffer(f"{label}_std", std)

    def forward(
        self,
        x: torch.Tensor,
        evidence_probability: torch.Tensor,
        orientation_logits: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        if evidence_probability.ndim != 4 or evidence_probability.shape[1] != 1:
            raise ValueError("evidence_probability must be Bx1xHxW")
        if evidence_probability.requires_grad:
            # IR2 freezes the evidence field. This guard prevents silent semantic drift.
            raise ValueError("IR2 evidence_probability must come from the frozen IR1 evidence head")
        with torch.no_grad():
            static = dense_orientation_features(evidence_probability, "static")
            dynamic = static if self.dynamic_method == "static" else dense_orientation_features(evidence_probability, self.dynamic_method)
            static = (static - self.static_mean) / self.static_std
            dynamic = (dynamic - self.dynamic_mean) / self.dynamic_std
            joined = torch.cat((static, dynamic), dim=-1)
            uncertainty = 4.0 * evidence_probability * (1.0 - evidence_probability)
            orientation = torch.sigmoid(orientation_logits).permute(0, 2, 3, 1)[..., None]
        responses = self.readout(joined)
        aggregate = (orientation * responses).sum(dim=-2) / orientation.sum(dim=-2).clamp_min(1e-6)
        correction = self.output_projection(aggregate.permute(0, 3, 1, 2))
        gated_correction = uncertainty * correction
        output = x + self.gamma * gated_correction
        if not return_aux:
            return output
        return output, {
            "evidence_probability": evidence_probability,
            "uncertainty": uncertainty,
            "orientation_logits": orientation_logits,
            "gated_correction": self.gamma * gated_correction,
            "gamma": self.gamma,
            "dynamic_method": self.dynamic_method,
        }
