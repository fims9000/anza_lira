from __future__ import annotations

import torch

from anza_kir.block import InnovationResidual
from anza_kir.model import KIR_VARIANTS, build_base_model, build_kir_model
from anza_kir.training import balanced_evidence_loss


def _norm():
    return {method: {"mean": torch.zeros(104), "std": torch.ones(104)} for method in ("static", "shear_ks", "cat_raw", "cat_ks")}


def test_gamma_zero_is_exact_identity_and_uses_evidence_probability():
    block = InnovationResidual(4, "R3_anza_kir", feature_norm=_norm()).eval()
    x = torch.randn(1, 4, 24, 24); evidence = torch.full((1, 1, 24, 24), 0.5); orientation = torch.zeros(1, 8, 24, 24)
    output, aux = block(x, evidence, orientation, return_aux=True)
    assert torch.equal(output, x)
    assert torch.equal(aux["evidence_probability"], evidence)
    assert torch.allclose(aux["uncertainty"], torch.ones_like(evidence))


def test_confident_evidence_disables_correction():
    block = InnovationResidual(4, "R1_shear_ks_residual", feature_norm=_norm()).eval()
    with torch.no_grad(): block.gamma.fill_(1.0)
    x = torch.randn(1, 4, 24, 24); orientation = torch.zeros(1, 8, 24, 24)
    for value in (0.0, 1.0):
        output = block(x, torch.full((1, 1, 24, 24), value), orientation)
        assert torch.allclose(output, x)


def test_all_residual_controls_have_equal_trainable_capacity():
    base = build_base_model().state_dict(); counts = []
    for variant in KIR_VARIANTS:
        model = build_kir_model(variant, base, _norm())
        counts.append(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
        assert not any(parameter.requires_grad for parameter in model.base.parameters())
    assert len(set(counts)) == 1


def test_evidence_loss_rewards_correct_fault_evidence():
    target = torch.zeros(1, 1, 12, 12); target[:, :, 4:8, 5:7] = 1
    correct = torch.where(target.bool(), torch.tensor(5.0), torch.tensor(-5.0))
    wrong = -correct
    assert balanced_evidence_loss(correct, target) < balanced_evidence_loss(wrong, target)
