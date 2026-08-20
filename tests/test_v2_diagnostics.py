from __future__ import annotations

import torch

from synthetic.diagnostics import effective_mode_count


def test_effective_mode_count_detects_collapse_and_uniform_usage() -> None:
    collapsed = torch.tensor([[[[1.0]], [[0.0]], [[0.0]], [[0.0]]]])
    uniform = torch.full((1, 4, 1, 1), 0.25)
    assert torch.allclose(effective_mode_count(collapsed), torch.ones(1, 1, 1))
    assert torch.allclose(effective_mode_count(uniform), torch.full((1, 1, 1), 4.0))


def test_effective_mode_count_is_finite_and_bounded() -> None:
    membership = torch.softmax(torch.randn(2, 5, 4, 3), dim=1)
    effective = effective_mode_count(membership)
    assert torch.isfinite(effective).all()
    assert torch.all((effective >= 1.0) & (effective <= 5.0 + 1e-6))
