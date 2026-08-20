from __future__ import annotations

import torch

from models.azconv_repaired import AmbiguityGatedResidualANZA


def test_zero_residual_is_exactly_the_unchanged_v1_base() -> None:
    torch.manual_seed(11)
    operator = AmbiguityGatedResidualANZA(3, 5).eval()
    sample = torch.randn(2, 3, 12, 13)
    with torch.no_grad():
        expected = operator.base(sample)
        diagnostics = operator(sample, return_diagnostics=True)
    assert diagnostics["residual_lambda"].item() == 0.0
    assert torch.equal(diagnostics["output"], expected)
    assert torch.count_nonzero(diagnostics["correction"]) == 0


def test_zero_lambda_keeps_nonzero_wakeup_gradient() -> None:
    torch.manual_seed(12)
    operator = AmbiguityGatedResidualANZA(2, 3)
    sample = torch.randn(1, 2, 9, 10)
    loss = operator(sample).square().mean()
    loss.backward()
    gradient = operator.residual_lambda_raw.grad
    assert gradient is not None
    assert torch.isfinite(gradient)
    assert gradient.abs() > 0
