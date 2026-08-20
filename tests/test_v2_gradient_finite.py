from __future__ import annotations

import torch

from models.azconv_v2 import AZConvV2Config, ModeResolvedAZConv2d


def test_v2a_forward_and_backward_are_finite() -> None:
    torch.manual_seed(42)
    layer = ModeResolvedAZConv2d(
        2,
        3,
        cfg=AZConvV2Config(num_modes=3, state_channels=4, transport_steps=2),
    )
    inputs = torch.randn(2, 2, 8, 10, requires_grad=True)
    diagnostics = layer(inputs, return_diagnostics=True)
    loss = diagnostics["output"].square().mean() + 0.01 * diagnostics["mode_states"].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    gradients = [parameter.grad for parameter in layer.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_v2a_geometry_diagnostics_are_finite_and_bounded() -> None:
    layer = ModeResolvedAZConv2d(3, 3, cfg=AZConvV2Config(num_modes=4))
    diagnostics = layer(torch.randn(1, 3, 6, 6), return_diagnostics=True)
    for name in ("membership", "theta", "sigma_u", "sigma_s", "hyperbolicity", "junction_score"):
        assert torch.isfinite(diagnostics[name]).all()
    assert torch.all(diagnostics["sigma_u"] > 0)
    assert torch.all(diagnostics["sigma_s"] > 0)
    assert torch.all((diagnostics["junction_score"] >= 0) & (diagnostics["junction_score"] <= 1))
