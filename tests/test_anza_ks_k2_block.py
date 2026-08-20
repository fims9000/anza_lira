import torch

from anza_ks_k2.block import SymbolicFeatureBlock
from anza_ks_k2.model import VARIANTS, build_k2_model


def test_gamma_zero_is_exact_identity_and_aux_contract() -> None:
    block = SymbolicFeatureBlock(8, "cat_ks")
    x = torch.randn(2, 8, 18, 18)
    y, aux = block(x, return_aux=True)
    torch.testing.assert_close(y, x, rtol=0.0, atol=0.0)
    assert aux["orientation_logits"].shape == (2, 8, 18, 18)
    assert aux["occupancy_logits"].shape == (2, 1, 18, 18)


def test_occupancy_gate_suppresses_correction() -> None:
    block = SymbolicFeatureBlock(8, "static")
    block.gamma.data.fill_(1.0)
    block.occupancy_head.weight.data.zero_(); block.occupancy_head.bias.data.fill_(-40.0)
    x = torch.randn(1, 8, 18, 18)
    y = block(x)
    torch.testing.assert_close(y, x, rtol=0.0, atol=1e-6)


def test_all_k2_models_share_output_contract() -> None:
    x = torch.randn(1, 3, 96, 96)
    for variant in VARIANTS:
        model = build_k2_model(variant)
        result = model(x, return_aux=True)
        assert result["visible_logits"].shape == (1, 1, 96, 96)
