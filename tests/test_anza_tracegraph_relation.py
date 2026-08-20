from __future__ import annotations

import torch

from anza_tracegraph.batch import K_MAX, RelationDataset
from anza_tracegraph.models import RelationTransformer, build_model
from anza_tracegraph.training import p0_loss


def _batch():
    return next(iter(torch.utils.data.DataLoader(RelationDataset("train", [0, 1, 2, 3]), batch_size=4)))


def test_p1_p2_capacity_differs_only_by_beta_and_none_is_present():
    p1 = build_model("P1_tracegraph"); p2 = build_model("P2_anza_tracegraph")
    assert sum(p.numel() for p in p2.parameters()) == sum(p.numel() for p in p1.parameters()) + 1
    output = p1(_batch()); assert output.shape == (4, K_MAX + 1)
    assert torch.allclose(torch.softmax(output, -1).sum(-1), torch.ones(4))


def test_padded_candidates_are_masked():
    batch = _batch(); output = build_model("P1_tracegraph")(batch)
    assert torch.all(output[:, :K_MAX][~batch["candidate_mask"]] == -30.0)


def test_beta_zero_matches_generic_transformer_exactly():
    torch.manual_seed(7); p1 = RelationTransformer(use_anza_bias=False).eval(); p2 = RelationTransformer(use_anza_bias=True).eval(); p2.load_state_dict(p1.state_dict(), strict=False)
    with torch.no_grad(): p2.beta_raw.fill_(-50.0)
    batch = _batch(); assert torch.allclose(p1(batch), p2(batch), atol=1e-6, rtol=1e-6)


def test_beta_gradient_is_finite_and_p1_has_no_anza_parameter():
    p1 = RelationTransformer(use_anza_bias=False); assert not hasattr(p1, "beta_raw")
    p2 = RelationTransformer(use_anza_bias=True); loss = torch.nn.functional.cross_entropy(p2(_batch()), _batch()["label"]); loss.backward(); assert p2.beta_raw.grad is not None and torch.isfinite(p2.beta_raw.grad)


def test_p0_balanced_loss_is_finite_with_none_and_positive():
    batch = _batch(); logits = build_model("P0_pair")(batch); loss = p0_loss(logits, batch["label"], batch["candidate_mask"]); assert torch.isfinite(loss) and loss > 0
