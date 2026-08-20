from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models.segmentation_v2 import NativeDeformConv2d, build_comparable_model


def test_zero_offset_native_deformable_matches_standard_convolution() -> None:
    torch.manual_seed(51)
    deformable = NativeDeformConv2d(2, 3, kernel_size=3)
    standard = nn.Conv2d(2, 3, kernel_size=3, padding=1)
    with torch.no_grad():
        deformable.weight.copy_(standard.weight.reshape(3, 2, 9))
        deformable.bias.copy_(standard.bias)
    inputs = torch.randn(2, 2, 9, 11)
    assert torch.allclose(deformable(inputs), standard(inputs), atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize(
    "name",
    ("unet", "deformable_unet", "anza_v1", "anza_v2a", "anza_v2b", "anza_v2_full"),
)
def test_comparable_models_share_visible_output_contract(name: str) -> None:
    torch.manual_seed(52)
    model = build_comparable_model(name, widths=(4, 8, 12, 16))
    inputs = torch.randn(1, 3, 32, 32)
    output = model(inputs)
    assert output.shape == (1, 1, 32, 32)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("name", ("anza_v2a", "anza_v2b", "anza_v2_full"))
def test_v2_models_expose_unfused_transport_diagnostics(name: str) -> None:
    model = build_comparable_model(name, widths=(4, 8, 12, 16))
    diagnostics = model(torch.randn(1, 3, 16, 16), return_diagnostics=True)
    assert len(diagnostics["transport_diagnostics"]) == 3
    first = diagnostics["transport_diagnostics"][0]
    if name == "anza_v2a":
        assert first["mode_states"].ndim == 5
    else:
        assert first["mode_states"].ndim == 6
    assert torch.isfinite(first["transport_mass"]).all()
    assert ("completion_logits" in diagnostics) == (name == "anza_v2_full")


def test_full_v2_visible_and_completion_heads_receive_finite_gradients() -> None:
    model = build_comparable_model("anza_v2_full", widths=(4, 8, 12, 16))
    diagnostics = model(torch.randn(1, 3, 16, 16), return_diagnostics=True)
    loss = diagnostics["visible_logits"].square().mean() + diagnostics["completion_logits"].square().mean()
    loss.backward()
    for head in (model.visible_head, model.completion_head):
        assert head is not None
        assert all(parameter.grad is not None for parameter in head.parameters())
        assert all(torch.isfinite(parameter.grad).all() for parameter in head.parameters())
