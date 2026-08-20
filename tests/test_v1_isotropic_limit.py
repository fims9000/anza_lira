from __future__ import annotations

import torch

from models.azconv import AZConv2d, AZConvConfig


def test_v1_isotropic_kernel_depends_only_on_offset_radius() -> None:
    layer = AZConv2d(3, 4, kernel_size=3, num_rules=4, cfg=AZConvConfig(use_anisotropy=False))
    kernel = layer._isotropic_kernel(torch.device("cpu"))[0]
    for rule in range(layer.R):
        assert torch.allclose(kernel[rule, 1], kernel[rule, 3])
        assert torch.allclose(kernel[rule, 1], kernel[rule, 5])
        assert torch.allclose(kernel[rule, 1], kernel[rule, 7])
        assert torch.allclose(kernel[rule, 0], kernel[rule, 2])
        assert torch.allclose(kernel[rule, 0], kernel[rule, 6])
        assert torch.allclose(kernel[rule, 0], kernel[rule, 8])


def test_v1_scales_are_positive_finite() -> None:
    layer = AZConv2d(3, 4, num_rules=4)
    layer(torch.randn(1, 3, 12, 12))
    snapshot = layer.interpretation_snapshot()
    for name in ("sigma_u_map", "sigma_s_map"):
        value = snapshot[name]
        assert torch.isfinite(value).all()
        assert torch.all(value > 0)
