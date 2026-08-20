"""Direct permutation-invariant mode and identity-preserving route supervision."""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F

from models.azconv_v2 import axial_distance


def axial_mode_set_loss(
    predicted_theta: torch.Tensor,
    predicted_membership: torch.Tensor,
    gt_theta_set: torch.Tensor,
    gt_theta_valid: torch.Tensor,
    *,
    membership_weight: float = 1.0,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match unordered GT axial directions to distinct predicted modes."""
    if predicted_theta.shape != predicted_membership.shape or predicted_theta.ndim != 4:
        raise ValueError("predicted theta and membership must be matching BxRxHxW")
    if gt_theta_set.shape != gt_theta_valid.shape or gt_theta_set.ndim != 4:
        raise ValueError("GT theta set and validity must be matching BxMxHxW")
    if predicted_theta.shape[0] != gt_theta_set.shape[0] or predicted_theta.shape[2:] != gt_theta_set.shape[2:]:
        raise ValueError("predicted and GT tangent fields must share batch/spatial shape")
    batch, modes, height, width = predicted_theta.shape
    max_targets = gt_theta_set.shape[1]
    if max_targets > modes:
        raise ValueError("GT tangent cardinality exceeds predicted mode count")
    membership = predicted_membership / predicted_membership.sum(dim=1, keepdim=True).clamp_min(epsilon)
    assignment = torch.full(
        (batch, max_targets, height, width),
        -1,
        dtype=torch.long,
        device=predicted_theta.device,
    )
    orientation_terms = []
    membership_terms = []
    count_map = gt_theta_valid.sum(dim=1)
    for count in range(1, max_targets + 1):
        selected = count_map == count
        if not selected.any():
            continue
        pred = predicted_theta.permute(0, 2, 3, 1)[selected]
        truth = gt_theta_set[:, :count].permute(0, 2, 3, 1)[selected]
        mu = membership.permute(0, 2, 3, 1)[selected]
        cost = axial_distance(pred.unsqueeze(2), truth.unsqueeze(1)).square()
        permutations = torch.tensor(
            list(itertools.permutations(range(modes), count)),
            device=predicted_theta.device,
            dtype=torch.long,
        )
        candidate_costs = torch.stack(
            [cost[:, permutation, torch.arange(count, device=cost.device)].mean(dim=1) for permutation in permutations],
            dim=1,
        )
        best_index = candidate_costs.argmin(dim=1)
        best_modes = permutations[best_index]
        orientation_terms.append(candidate_costs.gather(1, best_index[:, None]).squeeze(1))
        matched_mu = mu.gather(1, best_modes).clamp_min(epsilon)
        target_mass = 1.0 / count
        membership_terms.append(
            (target_mass * (torch.log(torch.tensor(target_mass, device=mu.device, dtype=mu.dtype)) - matched_mu.log())).sum(dim=1)
        )
        for target_index in range(count):
            target_map = assignment[:, target_index]
            target_map[selected] = best_modes[:, target_index]
    zero = predicted_theta.sum() * 0.0
    orientation_loss = torch.cat(orientation_terms).mean() if orientation_terms else zero
    membership_loss = torch.cat(membership_terms).mean() if membership_terms else zero
    total = orientation_loss + float(membership_weight) * membership_loss
    return total, {
        "orientation_set_loss": orientation_loss,
        "membership_set_kl": membership_loss,
        "assignment": assignment,
        "supervised_pixels": (count_map > 0).sum(),
    }


def branch_mode_masks_from_tangents(
    predicted_theta: torch.Tensor,
    gt_branch_theta: torch.Tensor,
    gt_branch_theta_valid: torch.Tensor,
) -> torch.Tensor:
    """Assign each lineage branch pixel to its closest predicted axial mode."""
    if predicted_theta.ndim != 4 or gt_branch_theta.ndim != 4:
        raise ValueError("theta fields must be BxRxHxW and BxNxHxW")
    if gt_branch_theta.shape != gt_branch_theta_valid.shape:
        raise ValueError("branch theta and validity must match")
    if predicted_theta.shape[0] != gt_branch_theta.shape[0] or predicted_theta.shape[2:] != gt_branch_theta.shape[2:]:
        raise ValueError("predicted and branch theta fields must share batch/spatial shape")
    distances = axial_distance(
        predicted_theta.unsqueeze(1),
        gt_branch_theta.unsqueeze(2),
    )
    closest = distances.argmin(dim=2)
    one_hot = F.one_hot(closest, num_classes=predicted_theta.shape[1]).permute(0, 1, 4, 2, 3)
    return one_hot.to(dtype=predicted_theta.dtype) * gt_branch_theta_valid.unsqueeze(2).to(predicted_theta.dtype)


def mode_specific_branch_transition_logits(
    transport: torch.Tensor,
    branch_mode_masks: torch.Tensor,
    *,
    kernel_size: int,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Aggregate branch relations while retaining matched source/destination modes."""
    transition = torch.as_tensor(transport)
    masks = torch.as_tensor(branch_mode_masks, device=transition.device, dtype=transition.dtype)
    if transition.ndim != 5 or transition.shape[0] != 1:
        raise ValueError("transport must be 1xRdestinationxRsourcexKxL")
    if masks.ndim != 4:
        raise ValueError("branch_mode_masks must be NxRxHxW for one sample")
    branches, modes, height, width = masks.shape
    if transition.shape[1:3] != (modes, modes):
        raise ValueError("transport and branch mode counts differ")
    patch_area = kernel_size**2
    locations = height * width
    if transition.shape[3:] != (patch_area, locations):
        raise ValueError("transport spatial shape does not match branch masks")
    destination = masks.reshape(branches, modes, locations)
    source = F.unfold(
        masks.reshape(branches * modes, 1, height, width),
        kernel_size,
        padding=kernel_size // 2,
    ).reshape(branches, modes, patch_area, locations)
    score = torch.einsum("drl,eskl,rskl->de", destination, source, transition[0])
    support = torch.einsum("dl,ekl->de", destination.sum(dim=1), source.sum(dim=1))
    return torch.log((score / support.clamp_min(epsilon)).clamp_min(epsilon))
