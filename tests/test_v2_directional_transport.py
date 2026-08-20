from __future__ import annotations

import math

import torch

from models.azconv_v2 import AZConvV2Config, ModeResolvedAZConv2d, directional_compatibility


def test_directional_compatibility_prefers_the_matching_half_mode() -> None:
    direction = torch.tensor([1.0, 0.0])
    theta = torch.tensor(0.0)
    forward = directional_compatibility(direction, theta, 1.0, 4.0)
    backward = directional_compatibility(direction, theta, -1.0, 4.0)
    orthogonal = directional_compatibility(direction, torch.tensor(math.pi / 2), 1.0, 4.0)
    assert torch.allclose(forward, torch.tensor(1.0))
    assert forward > orthogonal > backward


def test_zero_directional_kappa_disables_half_mode_preference() -> None:
    directions = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    theta = torch.tensor([0.2, 1.1])
    plus = directional_compatibility(directions, theta, 1.0, 0.0)
    minus = directional_compatibility(directions, theta, -1.0, 0.0)
    assert torch.allclose(plus, torch.ones_like(plus))
    assert torch.allclose(minus, torch.ones_like(minus))


def test_v2b_transition_routes_positive_x_travel_to_positive_half_mode() -> None:
    layer = ModeResolvedAZConv2d(
        1,
        1,
        cfg=AZConvV2Config(
            num_modes=1,
            state_channels=1,
            transport_steps=1,
            variant="v2b",
            kappa_direction=4.0,
        ),
    )
    diagnostics = layer(torch.zeros(1, 1, 5, 5), return_diagnostics=True)
    transition = diagnostics["transport"]
    destination_location = 2 * 5 + 3
    source_left_offset = 3  # row-major 3x3 offset (-1, 0): q -> p travels +x
    plus_to_plus = transition[0, 0, 0, 0, 0, source_left_offset, destination_location]
    plus_to_minus = transition[0, 0, 1, 0, 0, source_left_offset, destination_location]
    assert plus_to_plus > plus_to_minus


def test_v2b_forward_backward_and_half_states_are_finite() -> None:
    torch.manual_seed(44)
    layer = ModeResolvedAZConv2d(
        2,
        3,
        cfg=AZConvV2Config(
            num_modes=3,
            state_channels=4,
            transport_steps=2,
            variant="v2b",
        ),
    )
    inputs = torch.randn(1, 2, 7, 8, requires_grad=True)
    diagnostics = layer(inputs, return_diagnostics=True)
    loss = diagnostics["output"].square().mean()
    loss.backward()
    assert diagnostics["mode_states"].shape == (1, 3, 2, 4, 7, 8)
    assert torch.isfinite(diagnostics["transport"]).all()
    assert torch.isfinite(inputs.grad).all()
