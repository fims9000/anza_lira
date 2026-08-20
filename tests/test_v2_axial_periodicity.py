from __future__ import annotations

import math

import torch

from models.azconv_v2 import axial_distance


def test_axial_distance_limit_cases() -> None:
    theta = torch.tensor([0.2, -0.7])
    assert torch.allclose(axial_distance(theta, theta), torch.zeros_like(theta), atol=1e-6)
    assert torch.allclose(axial_distance(theta, theta + math.pi), torch.zeros_like(theta), atol=2e-4)
    orthogonal = axial_distance(torch.tensor(0.0), torch.tensor(math.pi / 2))
    assert torch.allclose(orthogonal, torch.tensor(math.pi / 2), atol=1e-6)
