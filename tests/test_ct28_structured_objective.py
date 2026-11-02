from pathlib import Path
import sys
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research/ccta_graph_lira_safe_repair"))
from ct28_structured_objective import (
    structured_log_probs, structure_nll, jensen_shannon, consistency_regularized_loss,
)


def build(a=0.0):
    relation = torch.tensor([0.2, 0.3, -0.1, 1.1 + a], dtype=torch.float64, requires_grad=True)
    p = torch.tensor([0.1, 0.6], dtype=torch.float64, requires_grad=True)
    j = torch.tensor([0.2, 0.7], dtype=torch.float64, requires_grad=True)
    pairs = [("E00", "E01"), ("E02", "E03")]
    junctions = [("E01", "E04", "E05"), ("E00", "E04", "E05")]
    return relation, p, j, pairs, junctions


def test_constraint_normalization_and_gradient():
    relation, p, j, pe, je = build()
    q = structured_log_probs(relation, p, j, pe, je)
    assert ("NONE", -1, -1) in q
    assert ("BOTH", 0, 0) not in q
    assert ("BOTH", 0, 1) not in q
    assert ("BOTH", 1, 0) in q
    assert ("BOTH", 1, 1) in q
    assert abs(sum(x.exp().item() for x in q.values()) - 1.0) < 1e-12
    loss = structure_nll(q, ("BOTH", 1, 0))
    loss.backward()
    for t in (relation, p, j):
        assert t.grad is not None and torch.isfinite(t.grad).all()
        assert torch.any(t.grad != 0)


def test_js_is_symmetric_nonnegative_zero_at_identity():
    relation, p, j, pe, je = build()
    same = structured_log_probs(relation, p, j, pe, je)
    relation2, p2, j2, _, _ = build(a=-0.5)
    other = structured_log_probs(relation2, p2, j2, pe, je)
    assert abs(jensen_shannon(same, same).item()) < 1e-12
    assert jensen_shannon(same, other).item() > 0
    assert torch.allclose(jensen_shannon(same, other), jensen_shannon(other, same))
    loss = consistency_regularized_loss(same, [other], ("BOTH", 1, 0), js_weight=2.0)
    loss.backward()
    assert relation.grad is not None and torch.isfinite(relation.grad).all()


def test_missing_candidate_identity_and_unfeasible_truth_fail():
    relation, p, j, pe, je = build()
    q = structured_log_probs(relation, p, j, pe, je)
    with pytest.raises(ValueError, match="Ground truth"):
        structure_nll(q, ("BOTH", 0, 1))
    with pytest.raises(ValueError, match="identity"):
        jensen_shannon(q, {("NONE",-1,-1): next(iter(q.values()))})
    with pytest.raises(ValueError, match="nonnegative"):
        consistency_regularized_loss(q, [], ("NONE", -1, -1), js_weight=-1)


def test_joint_loss_passes_numerical_gradient_check():
    relation, p, j, pe, je = build()
    relation2, p2, j2, _, _ = build(a=0.1)
    def objective(t, ps, js, tt, pps, jjs):
        clean = structured_log_probs(t, ps, js, pe, je)
        perturbed = structured_log_probs(tt, pps, jjs, pe, je)
        return consistency_regularized_loss(clean, [perturbed], ("BOTH", 1, 0), js_weight=0.7)
    assert torch.autograd.gradcheck(objective, (relation, p, j, relation2, p2, j2), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_no_candidates_gives_only_none_and_mismatched_shapes_raise():
    logits = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    empty = torch.empty(0, dtype=torch.float64)
    logq = structured_log_probs(logits, empty, empty, [], [])
    assert list(logq) == [("NONE", -1, -1)]
    assert logq[("NONE", -1, -1)].exp().item() == 1.0
    with pytest.raises(ValueError, match="expected"):
        structured_log_probs(logits, empty, torch.ones(1, dtype=torch.float64), [], [])
