from __future__ import annotations

import torch

from models.azconv import AZConv2d, AZConvConfig


def test_v1_global_interaction_weights_sum_to_one_per_destination() -> None:
    layer = AZConv2d(3, 5, num_rules=4, cfg=AZConvConfig(normalize_mode="global"))
    captured = {}
    original = layer._update_interpretation_cache

    def capture(mu, kernel, compatibility, interpretation):
        captured["compatibility"] = compatibility.detach()
        return original(mu, kernel, compatibility, interpretation)

    layer._update_interpretation_cache = capture
    layer(torch.randn(2, 3, 11, 13))
    mass = captured["compatibility"].sum(dim=(1, 2))
    assert torch.allclose(mass, torch.ones_like(mass), atol=1e-6)
