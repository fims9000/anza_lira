from __future__ import annotations

import torch

from models.azconv_v2 import AZConvV2Config, ModeResolvedAZConv2d


def test_v2_memberships_are_nonnegative_and_sum_to_one() -> None:
    layer = ModeResolvedAZConv2d(3, 5, cfg=AZConvV2Config(num_modes=4, state_channels=6))
    diagnostics = layer(torch.randn(2, 3, 11, 13), return_diagnostics=True)
    membership = diagnostics["membership"]
    assert torch.all(membership >= 0)
    assert torch.allclose(membership.sum(dim=1), torch.ones_like(membership[:, 0]), atol=1e-6)
    assert diagnostics["mode_states"].shape == (2, 4, 6, 11, 13)
    assert diagnostics["output"].shape == (2, 5, 11, 13)


def test_no_fuzzy_limit_is_uniform() -> None:
    layer = ModeResolvedAZConv2d(2, 2, cfg=AZConvV2Config(num_modes=4, use_fuzzy=False))
    membership = layer.geometry(torch.randn(1, 2, 5, 5))["membership"]
    assert torch.allclose(membership, torch.full_like(membership, 0.25))
