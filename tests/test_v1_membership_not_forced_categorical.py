from __future__ import annotations

import torch

from affinity_repair.v1_audit import v1_source_facts
from models.azconv import AZConv2d


def test_current_v1_is_documented_as_categorical_softmax_mismatch() -> None:
    layer = AZConv2d(3, 4, num_rules=4)
    captured = {}
    original = layer._update_regularization_terms

    def capture(mu, gap, smoothness, interpretation):
        captured["mu"] = mu.detach()
        return original(mu, gap, smoothness, interpretation)

    layer._update_regularization_terms = capture
    layer(torch.randn(2, 3, 9, 9))
    assert torch.allclose(captured["mu"].sum(dim=1), torch.ones(2, 9, 9), atol=1e-6)
    assert v1_source_facts()["membership_is_categorical_simplex"] is True
    assert v1_source_facts()["fuzzy_independent_candidate_required"] is True
