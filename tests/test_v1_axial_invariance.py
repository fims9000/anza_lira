from __future__ import annotations

import math

import torch

from models.azconv import AZConv2d


def test_v1_local_geometry_is_exactly_pi_periodic() -> None:
    torch.manual_seed(9)
    first = AZConv2d(3, 4, num_rules=4).eval()
    second = AZConv2d(3, 4, num_rules=4).eval()
    second.load_state_dict(first.state_dict())
    with torch.no_grad():
        second.geometry_conv.bias[: second.R].add_(math.pi)
    image = torch.randn(2, 3, 17, 19)
    with torch.inference_mode():
        assert torch.allclose(first(image), second(image), atol=2e-6, rtol=1e-6)
