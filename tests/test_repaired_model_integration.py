from __future__ import annotations

import torch

from models.segmentation_repaired import build_repaired_model
from models.segmentation_v2 import build_comparable_model


def test_full_repaired_network_equals_seed_matched_v1_at_zero_lambda() -> None:
    torch.manual_seed(100)
    v1 = build_comparable_model("anza_v1", widths=(4, 8, 12, 16)).eval()
    repaired = build_repaired_model(
        widths=(4, 8, 12, 16),
        routing_kernel_size=3,
        seed_matched_v1=v1,
    ).eval()
    sample = torch.randn(1, 3, 24, 24)
    with torch.no_grad():
        expected = v1(sample)
        result = repaired(sample, return_diagnostics=True)
    assert torch.equal(result["visible_logits"], expected)
    assert len(result["transport_diagnostics"]) == 1
    assert result["transport_diagnostics"][0]["transport"].ndim == 5


def test_only_enc1_contains_a_repaired_router() -> None:
    model = build_repaired_model(widths=(4, 8, 12, 16))
    assert model.enc1.spatial.__class__.__name__ == "AmbiguityGatedResidualANZA"
    assert model.enc2.spatial.__class__.__name__ == "AZConv2d"
    assert model.enc3.spatial.__class__.__name__ == "AZConv2d"
