from __future__ import annotations

import torch

from models.azconv_repaired import AmbiguityGatedResidualANZA


def test_repaired_operator_gradients_are_finite_after_wakeup() -> None:
    torch.manual_seed(40)
    operator = AmbiguityGatedResidualANZA(3, 4)
    with torch.no_grad():
        operator.residual_lambda_raw.fill_(0.1)
    sample = torch.randn(2, 3, 11, 12, requires_grad=True)
    output = operator(sample)
    loss = output.square().mean()
    loss.backward()
    assert torch.isfinite(output).all()
    assert sample.grad is not None and torch.isfinite(sample.grad).all()
    for name, parameter in operator.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
