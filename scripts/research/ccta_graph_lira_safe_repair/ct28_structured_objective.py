"""Experimental differentiable, feasible-set structural objective for Graph-LIRA.

Train-only research primitive. It does NOT replace the frozen K10/15 inference.
No patient data, selection thresholds, or test evaluation are included here.
"""
from __future__ import annotations

from typing import Sequence
import math

import torch


RELATIONS = ("NONE", "PAIR", "JUNCTION", "BOTH")
Decision = tuple[str, int, int]


def _validate_tensor(name: str, value: torch.Tensor, size: int | None = None) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional torch tensor")
    if size is not None and value.numel() != size:
        raise ValueError(f"{name} expected {size} values")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} has nonfinite values")


def _check_endpoints(rows: Sequence[Sequence[str]], arity: int, name: str) -> None:
    for row in rows:
        if len(row) != arity or len(set(row)) != arity or not all(
            isinstance(s, str) and s for s in row
        ):
            raise ValueError(f"Invalid {name} endpoints")


def structured_log_probs(
    relation_logits: torch.Tensor,
    pair_scores: torch.Tensor,
    junction_scores: torch.Tensor,
    pair_endpoints: Sequence[Sequence[str]],
    junction_endpoints: Sequence[Sequence[str]],
) -> dict[Decision, torch.Tensor]:
    """Log distribution over NONE/PAIR/JUNCTION/feasible BOTH decisions.

    Normalize candidate choices within each relation type. If some relation
    type has no feasible candidate set, normalize remaining structural mass
    for this exploratory training objective. Do not use these normalized scores
    as calibrated confidence without separate validation.
    """
    _validate_tensor("relation_logits", relation_logits, 4)
    _validate_tensor("pair_scores", pair_scores, len(pair_endpoints))
    _validate_tensor("junction_scores", junction_scores, len(junction_endpoints))
    if not (relation_logits.device == pair_scores.device == junction_scores.device):
        raise ValueError("All tensors must share device")
    if not (relation_logits.dtype == pair_scores.dtype == junction_scores.dtype):
        raise ValueError("All tensors must share dtype")
    _check_endpoints(pair_endpoints, 2, "PAIR")
    _check_endpoints(junction_endpoints, 3, "JUNCTION")

    typelog = torch.log_softmax(relation_logits, dim=0)
    decisions = [("NONE", -1, -1)]
    unnormalized = [typelog[0]]
    if pair_scores.numel():
        pair_cond = torch.log_softmax(pair_scores, dim=0)
        for i in range(len(pair_endpoints)):
            decisions.append(("PAIR", i, -1))
            unnormalized.append(typelog[1] + pair_cond[i])
    if junction_scores.numel():
        junction_cond = torch.log_softmax(junction_scores, dim=0)
        for j in range(len(junction_endpoints)):
            decisions.append(("JUNCTION", -1, j))
            unnormalized.append(typelog[2] + junction_cond[j])

    allowed = [
        (i, j)
        for i, pair in enumerate(pair_endpoints)
        for j, junction in enumerate(junction_endpoints)
        if set(pair).isdisjoint(junction)
    ]
    if allowed:
        joint = torch.stack([pair_scores[i] + junction_scores[j] for i, j in allowed])
        joint_cond = torch.log_softmax(joint, dim=0)
        for k, (i, j) in enumerate(allowed):
            decisions.append(("BOTH", i, j))
            unnormalized.append(typelog[3] + joint_cond[k])
    logmass = torch.stack(unnormalized)
    logmass = logmass - torch.logsumexp(logmass, dim=0)
    return {key: logmass[i] for i, key in enumerate(decisions)}


def structure_nll(log_probs: dict[Decision, torch.Tensor], truth: Decision) -> torch.Tensor:
    if truth not in log_probs:
        raise ValueError("Ground truth is not in the feasible structural set")
    return -log_probs[truth]


def jensen_shannon(
    clean: dict[Decision, torch.Tensor],
    perturbed: dict[Decision, torch.Tensor],
) -> torch.Tensor:
    """Symmetric JS divergence with natural logarithm and autograd support.

    Requires the same candidate universe and stable candidate-index identity.
    A changing candidate pool is a recall failure, not a missing zero entry.
    """
    if list(clean) != list(perturbed) or not clean:
        raise ValueError("Candidate identity/order changed across perturbations")
    p_log = torch.stack([clean[k] for k in clean])
    q_log = torch.stack([perturbed[k] for k in clean])
    m_log = torch.logaddexp(p_log, q_log) - math.log(2.0)
    return 0.5 * (torch.sum(p_log.exp() * (p_log - m_log)) +
                  torch.sum(q_log.exp() * (q_log - m_log)))


def consistency_regularized_loss(
    clean: dict[Decision, torch.Tensor],
    perturbed: Sequence[dict[Decision, torch.Tensor]],
    true_decision: Decision,
    *,
    js_weight: float,
) -> torch.Tensor:
    if not math.isfinite(js_weight) or js_weight < 0:
        raise ValueError("js_weight must be nonnegative")
    nll = structure_nll(clean, true_decision)
    if not perturbed:
        return nll
    regularizer = torch.stack([jensen_shannon(clean, state) for state in perturbed]).mean()
    return nll + js_weight * regularizer
