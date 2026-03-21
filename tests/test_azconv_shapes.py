"""Shape and finiteness checks for AZConv2d (pytest)."""

import torch

from models.azconv import AZConv2d, AZConvConfig


def test_azconv_output_shape():
    device = torch.device("cpu")
    B, C, H, W, R, k, out_c = 2, 8, 32, 32, 4, 3, 16
    x = torch.randn(B, C, H, W, device=device)
    layer = AZConv2d(C, out_c, kernel_size=k, num_rules=R).to(device)
    y = layer(x)
    assert y.shape == (B, out_c, H, W)


def test_unfold_matches_expected_length():
    B, C, H, W, k = 1, 3, 32, 32, 3
    S, L = k * k, H * W
    x = torch.randn(B, C, H, W)
    u = torch.nn.functional.unfold(x, k, padding=k // 2, stride=1)
    assert u.shape == (B, C * S, L)


def test_ablations_run():
    x = torch.randn(2, 4, 16, 16)
    for cfg in [
        AZConvConfig(True, True, True),
        AZConvConfig(False, True, True),
        AZConvConfig(True, False, True),
        AZConvConfig(True, True, False),
    ]:
        m = AZConv2d(4, 8, kernel_size=3, num_rules=3, cfg=cfg)
        y = m(x)
        assert y.shape == (2, 8, 16, 16)
        assert torch.isfinite(y).all()

