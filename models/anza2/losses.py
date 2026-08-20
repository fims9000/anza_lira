"""Permutation-invariant field supervision and axial target transforms."""

from __future__ import annotations

import math

import torch


def axis_set_coverage_loss(
    predicted_orientation: torch.Tensor,
    target_orientation: torch.Tensor,
    target_valid: torch.Tensor,
) -> torch.Tensor:
    """Cover every valid target axis with its closest predicted mode."""

    if predicted_orientation.ndim != 5 or predicted_orientation.shape[2] != 2:
        raise ValueError("predicted_orientation must be BxRx2xHxW")
    if target_orientation.ndim != 5 or target_orientation.shape[2] != 2:
        raise ValueError("target_orientation must be BxKx2xHxW")
    if target_valid.shape != target_orientation.shape[:2] + target_orientation.shape[-2:]:
        raise ValueError("target_valid must be BxKxHxW")
    similarity = torch.einsum("brchw,bkchw->brkhw", predicted_orientation, target_orientation)
    distance = 1.0 - similarity.amax(dim=1)
    weight = target_valid.to(distance.dtype)
    return (distance * weight).sum() / weight.sum().clamp_min(1.0)


def membership_weighted_axis_set_coverage_loss(
    predicted_orientation: torch.Tensor,
    membership: torch.Tensor,
    target_orientation: torch.Tensor,
    target_valid: torch.Tensor,
) -> torch.Tensor:
    """Cover each target axis with an axis-aligned *active* fuzzy mode.

    Plain set coverage can place the correct angle in an inactive channel.  The
    structural affinity then sees a different mode than the supervised angle.
    Weighting axial similarity by independent membership closes that gap while
    retaining permutation invariance and allowing several active modes.
    """

    if membership.shape != predicted_orientation.shape[:2] + predicted_orientation.shape[-2:]:
        raise ValueError("membership must be BxRxHxW and align with predicted orientations")
    if target_valid.shape != target_orientation.shape[:2] + target_orientation.shape[-2:]:
        raise ValueError("target_valid must be BxKxHxW")
    similarity = torch.einsum("brchw,bkchw->brkhw", predicted_orientation, target_orientation)
    axial_similarity = ((similarity + 1.0) / 2.0).clamp(0.0, 1.0)
    supported = membership.unsqueeze(2) * axial_similarity
    distance = 1.0 - supported.amax(dim=1)
    weight = target_valid.to(distance.dtype)
    return (distance * weight).sum() / weight.sum().clamp_min(1.0)


def active_mode_count_loss(membership: torch.Tensor, k_max: torch.Tensor | float, valid: torch.Tensor) -> torch.Tensor:
    excess = torch.relu(membership.sum(dim=1) - torch.as_tensor(k_max, device=membership.device))
    weight = valid.to(excess.dtype)
    return (excess.square() * weight).sum() / weight.sum().clamp_min(1.0)


def membership_axis_coverage_loss(
    predicted_orientation: torch.Tensor,
    membership: torch.Tensor,
    target_orientation: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    gamma: float = 2.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Require every target axis to be covered by an aligned active mode.

    Modes are an unordered local set.  The max over predicted modes therefore
    makes the objective permutation invariant without assigning global mode
    identities.  Doubled-angle vectors make antipodal physical directions the
    same axis.
    """

    if predicted_orientation.ndim != 5 or predicted_orientation.shape[2] != 2:
        raise ValueError("predicted_orientation must be BxRx2xHxW")
    if membership.shape != predicted_orientation.shape[:2] + predicted_orientation.shape[-2:]:
        raise ValueError("membership must be BxRxHxW and align with predicted_orientation")
    if target_orientation.ndim != 5 or target_orientation.shape[2] != 2:
        raise ValueError("target_orientation must be BxKx2xHxW")
    if target_valid.shape != target_orientation.shape[:2] + target_orientation.shape[-2:]:
        raise ValueError("target_valid must be BxKxHxW")
    if gamma <= 0 or epsilon <= 0:
        raise ValueError("gamma and epsilon must be positive")
    similarity = torch.einsum("brchw,bkchw->brkhw", predicted_orientation, target_orientation)
    compatibility = ((1.0 + similarity) / 2.0).clamp(0.0, 1.0).pow(float(gamma))
    coverage = (membership.unsqueeze(2) * compatibility).amax(dim=1)
    valid = target_valid.bool()
    if not valid.any():
        return membership.sum() * 0.0
    return -torch.log(coverage[valid].clamp_min(float(epsilon))).mean()


def positive_exact_mode_count_loss(
    membership: torch.Tensor,
    target_mode_count: torch.Tensor,
) -> torch.Tensor:
    """Match fuzzy mass to the exact target count on structural pixels only."""

    if membership.ndim != 4:
        raise ValueError("membership must be BxRxHxW")
    if target_mode_count.shape != membership.shape[:1] + membership.shape[-2:]:
        raise ValueError("target_mode_count must be BxHxW")
    positive = target_mode_count > 0
    if not positive.any():
        return membership.sum() * 0.0
    error = membership.sum(dim=1) - target_mode_count.to(membership.dtype)
    return error[positive].square().mean()


def background_membership_loss(
    membership: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Suppress fuzzy union on background with separate class normalization."""

    if membership.ndim != 4:
        raise ValueError("membership must be BxRxHxW")
    if target_valid.ndim != 4 or target_valid.shape[0] != membership.shape[0] or target_valid.shape[-2:] != membership.shape[-2:]:
        raise ValueError("target_valid must be BxKxHxW and align spatially")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    background = ~target_valid.bool().any(dim=1)
    if not background.any():
        return membership.sum() * 0.0
    fuzzy_union = 1.0 - torch.prod(1.0 - membership, dim=1)
    return -torch.log((1.0 - fuzzy_union[background]).clamp_min(float(epsilon))).mean()


def rc1_membership_loss(
    predicted_orientation: torch.Tensor,
    membership: torch.Tensor,
    target_orientation: torch.Tensor,
    target_valid: torch.Tensor,
    target_mode_count: torch.Tensor,
    *,
    lambda_bg: float,
    lambda_count: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Frozen RC1 objective with separately normalized structural/background terms."""

    cover = membership_axis_coverage_loss(
        predicted_orientation, membership, target_orientation, target_valid, gamma=2.0
    )
    background = background_membership_loss(membership, target_valid)
    count = positive_exact_mode_count_loss(membership, target_mode_count)
    total = cover + float(lambda_bg) * background + float(lambda_count) * count
    return total, {"cover": cover, "background": background, "count_positive": count}


def transform_doubled_angle(orientation: torch.Tensor, transform: str) -> torch.Tensor:
    """Apply exact D4 spatial and component transforms to an axial field."""

    if orientation.shape[-3] != 2:
        raise ValueError("orientation must have a size-2 component axis")
    c, s = orientation.unbind(dim=-3)
    if transform == "identity":
        c2, s2 = c, s
    elif transform == "hflip":
        c2, s2 = torch.flip(c, (-1,)), -torch.flip(s, (-1,))
    elif transform == "vflip":
        c2, s2 = torch.flip(c, (-2,)), -torch.flip(s, (-2,))
    elif transform == "rot90":
        c2, s2 = -torch.rot90(c, 1, (-2, -1)), -torch.rot90(s, 1, (-2, -1))
    elif transform == "rot270":
        c2, s2 = -torch.rot90(c, 3, (-2, -1)), -torch.rot90(s, 3, (-2, -1))
    elif transform == "rot180":
        c2, s2 = torch.rot90(c, 2, (-2, -1)), torch.rot90(s, 2, (-2, -1))
    elif transform == "transpose":
        c2, s2 = -c.transpose(-2, -1), s.transpose(-2, -1)
    else:
        raise ValueError(f"unsupported transform: {transform}")
    return torch.stack((c2, s2), dim=-3)
